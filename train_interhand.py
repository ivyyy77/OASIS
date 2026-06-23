# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import pdb
import sys

from sklearn.preprocessing import maxabs_scale

from LHM.runners import REGISTRY_RUNNERS
# from splatformer.models.feature_predictor import FeaturePredictor
from torch.nn import Tanh, Identity

from data.debug import ReconstructionDataset, HandDataset
from data.interhand.train import Dataset, HandAvatarDataset, HanCo, MixedDataset
from data.debug import make_dataloader
import torch.nn as nn
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
import time
import json
import re
import shutil
from torch.nn.parallel import DistributedDataParallel as DDP
# from splatformer.utils import gpu_utils, gs_utils, loss_utils
# from splatformer.utils.optimizers import build_optimizer, build_scheduler
# from splatformer.utils.metrics import MetricComputer
# from splatformer.utils.log_utils import ProcessSafeLogger
import wandb
# wandb.login(key=os.getenv("WANDB_API_KEY"))  # configure locally
from tqdm import tqdm
from data.hand_dataset import merge_batch

from torch.utils.tensorboard import SummaryWriter
from LHM.losses import _is_better
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)



flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', './wandb/', 'Wandbs Output directory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
# CLI flags for checkpoint / output and resume (compatible with absl.flags)
flags.DEFINE_string('checkpoint-path', None, 'Checkpoint directory to save/load checkpoints')
flags.DEFINE_string('output-path', None, 'Output/test directory to write results')
flags.DEFINE_boolean('resume', False, 'Enable auto-resume from latest checkpoint')
flags.DEFINE_string('checkpoint-file', None, 'Specific checkpoint file to load (overrides checkpoint-path)')
flags.DEFINE_multi_string(
  'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')
# flags.DEFINE_integer('iter', '2', 'Training iteration')
# flags.DEFINE_integer('iter', '11757', 'Training iteration')
flags.DEFINE_integer('iter', '40000', 'Training iteration')
flags.DEFINE_integer('num_workers', 4, 'Number of DataLoader workers')
flags.DEFINE_integer('batch_size', 2, 'Default training batch size')
flags.DEFINE_boolean('use_amp', True, 'Use automatic mixed precision (AMP)')
flags.DEFINE_boolean('pin_memory', True, 'Use pin_memory for DataLoader')


FLAGS = flags.FLAGS

@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_runner_checkpoint(runner, iteration=None, checkpoint_path=None, checkpoint_file=None):
    """Load checkpoint with explicit file support when available."""
    if hasattr(runner, 'load_checkpoint'):
        return runner.load_checkpoint(
            iteration=iteration,
            is_latest=False,
            checkpoint_path=checkpoint_path,
            checkpoint_file=checkpoint_file,
        )
    return runner.load(iteration=iteration, is_latest=False, checkpoint_path=checkpoint_path)


def _infer_num_frames_from_test_batch(test_batch):
    """Best-effort frame count inference for FPS computation."""
    sample = test_batch
    if isinstance(test_batch, (list, tuple)) and len(test_batch) > 0:
        sample = test_batch[0]

    if isinstance(sample, dict) and 'original_image' in sample:
        img = sample['original_image']
        if torch.is_tensor(img):
            # [N, C, H, W] or [B, N, C, H, W]
            if img.dim() == 4:
                return int(img.shape[0])
            if img.dim() >= 5:
                return int(img.shape[1])
    return 1


def _run_test_with_fps(runner, test_dataloader, gs_model_list, gs_densify_list, query_points, iteration):
    psnr_full, ssim_full, lpips_full, image_full = 0.0, 0.0, 0.0, 0.0
    count = 0
    total_render_time = 0.0
    total_rendered_views = 0

    for idx, test_batch in enumerate(test_dataloader):
        with tqdm(
            total=1,
            desc=f"Iter {iteration} | Sample {idx+1}/{len(test_dataloader)}",
            leave=True,
            position=idx + 1,
            dynamic_ncols=True,
        ) as pbar:
            with torch.no_grad():
                loss_dict = runner.test_handavatar(
                    gs_model_list,
                    gs_densify_list,
                    query_points,
                    test_batch,
                    idx,
                    iteration,
                    pbar,
                )

            # Extract render timing from loss_dict (computed inside forward_animate_gs)
            total_render_time += loss_dict.get('render_time', 0.0)
            total_rendered_views += loss_dict.get('render_views', 0)

            psnr_full += loss_dict['psnr'].item()
            ssim_full += loss_dict['ssim'].item()
            lpips_full += loss_dict['lpips'].item()
            count += 1

            # FPS calculation based on render time only
            current_fps = total_rendered_views / max(total_render_time, 1e-8)
            pbar.set_postfix({
                "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
                "ssim": f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
                "psnr": f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
                "fps": f"{current_fps:.2f}",
            })
            pbar.update(1)
            torch.cuda.empty_cache()

    avg_fps = total_rendered_views / max(total_render_time, 1e-8)
    print(
        f"\033[91m[Summary @iter {iteration}]\033[0m "
        f"PSNR: {psnr_full/count:.4f}, "
        f"SSIM: {ssim_full/count:.4f}, "
        f"LPIPS: {lpips_full/count:.4f}, "
        f"L1: {image_full/count:.4f}, "
        f"FPS: {avg_fps:.2f}, "
        f"Frames: {total_rendered_views}, "
        f"Time: {total_render_time:.3f}s\n"
    )

    return {
        'lpips': lpips_full / count,
        'ssim': ssim_full / count,
        'psnr': psnr_full / count,
        'image_l1': image_full / count,
        'fps': avg_fps,
        'test_total_frames': int(total_rendered_views),
        'test_total_time_sec': float(total_render_time),
    }



def main(argv):

    # train_loader = HandDataset(split='train')
    # # test_loader = ReconstructionDataset(split='test')
    # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=1)
    # # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)


    # data_iterator = iter(dataloader)
    # # data_iterator_1 = iter(dataloader_1)
    # batches = next(data_iterator)


    # FLAGS.parse_args()  # 手动解析命令行参数
    parser = argparse.ArgumentParser(description="OpenLRM launcher")
    parser.add_argument("runner", type=str, help="Runner to launch")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Optional checkpoint directory to save/load checkpoints (overrides env)")
    parser.add_argument("--checkpoint-file", type=str, default=None,
                        help="Optional specific checkpoint file to load (overrides checkpoint-path)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Optional output/test directory to write results (overrides env)")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Enable auto-resume from latest checkpoint")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Disable auto-resume from checkpoint")
    parser.set_defaults(resume=False)

    # keep compatibility: accept arbitrary extra CLI bindings (passed to runners/config parsing)
    args, unknown = parser.parse_known_args()

    # prepare CLI values (no env writes) — will pass into runner when possible
    cli_checkpoint = args.checkpoint_path
    cli_checkpoint_file = args.checkpoint_file
    cli_output = args.output_path
    cli_resume = args.resume

    # Prefer absl FLAGS if argparse didn't get them (absl parsing happens before main)
    try:
        from absl import flags as _absl_flags
        absl_f = _absl_flags.FLAGS
        if cli_checkpoint is None and getattr(absl_f, 'checkpoint_path', None):
            cli_checkpoint = absl_f.checkpoint_path
        if cli_checkpoint_file is None and getattr(absl_f, 'checkpoint_file', None):
            cli_checkpoint_file = absl_f.checkpoint_file
        if cli_output is None and getattr(absl_f, 'output_path', None):
            cli_output = absl_f.output_path
        # argparse default is False; prefer explicit argparse, otherwise use absl
        if not hasattr(args, 'resume') or args.resume is None:
            cli_resume = getattr(absl_f, 'resume', False)
    except Exception:
        pass

    # backward-compat: pick up unknown 'checkpoint=..' or 'output=..' only if not provided
    # DEBUG: print CLI argument parsing
    print(f"[DEBUG CLI] args.checkpoint_path={args.checkpoint_path}, args.output_path={args.output_path}")
    print(f"[DEBUG CLI] unknown args: {unknown}")
    for u in unknown:
        if isinstance(u, str) and u.lower().startswith("checkpoint-path=") and cli_checkpoint is None:
            cli_checkpoint = u.split("=", 1)[1]
            print(f"[DEBUG CLI] Picked up checkpoint-path from unknown: {cli_checkpoint}")
        elif isinstance(u, str) and u.lower().startswith("checkpoint=") and cli_checkpoint is None:
            cli_checkpoint = u.split("=", 1)[1]
            print(f"[DEBUG CLI] Picked up checkpoint from unknown: {cli_checkpoint}")
        if isinstance(u, str) and u.lower().startswith("output-path=") and cli_output is None:
            cli_output = u.split("=", 1)[1]
            print(f"[DEBUG CLI] Picked up output-path from unknown: {cli_output}")
        elif isinstance(u, str) and u.lower().startswith("output=") and cli_output is None:
            cli_output = u.split("=", 1)[1]
            print(f"[DEBUG CLI] Picked up output from unknown: {cli_output}")
    print(f"[DEBUG CLI] Final: cli_checkpoint={cli_checkpoint}, cli_output={cli_output}, cli_resume={cli_resume}")

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError("Runner {} not found".format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]
    # Try to construct runner with explicit CLI args if it accepts them
    runner = None
    try:
        runner = RunnerClass(checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file, output_path=cli_output, resume=cli_resume)
    except TypeError:
        # fallback: default constructor then try to set attributes or call setter
        runner = RunnerClass()
        # set attributes if available
        try:
            if hasattr(runner, 'set_cli_args'):
                runner.set_cli_args(checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file, output_path=cli_output, resume=cli_resume)
            else:
                if cli_checkpoint is not None:
                    setattr(runner, 'cli_checkpoint_path', cli_checkpoint)
                if cli_checkpoint_file is not None:
                    setattr(runner, 'cli_checkpoint_file', cli_checkpoint_file)
                if cli_output is not None:
                    setattr(runner, 'cli_output_path', cli_output)
                setattr(runner, 'cli_resume', cli_resume)
        except Exception:
            pass

    # # features_offsets = ['means',  'scales', 'opacities', 'quats', 'features_dc', 'features_rest']
    # # features_activation = {'means': Tanh(), 'features_dc': Identity(), 'features_rest': Identity(), 'scales': Identity(), 'opacities': Identity(), 'quats': Identity()}
    # features_offsets = ['xyz',  'scaling', 'opacity', 'rotation', 'shs']
    # features_activation = {'xyz': Tanh(), 'shs': Identity(), 'scaling': Identity(), 'opacity': Identity(), 'rotation': Identity()}
    # # model = FeaturePredictor()
    # model = FeaturePredictor(backbone_type='PT', grid_resolution=384,
    #                          input_embed_to_mlp=False, input_feat_to_mlp=True, input_features=features_offsets, max_scale_normalized=0.01,
    #                          output_features=features_offsets, output_features_type='res',
    #                          # output_head_nlayer=4, output_head_type='mlp-relu', output_head_width=128,
    #                          output_head_nlayer=3, output_head_type='mlp-relu', output_head_width=128,
    #                          res_feature_activation=features_activation, resume_ckpt=None, sh_degree=1, zeroinit=True)

    # ==============================================================================================
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29502',#'env://',
                            rank=0,       # Set appropriate rank per process
                            world_size=1,
                            # master_addr="127.0.0.1",
                            # master_port=29500
                            )   # Total number of processes)
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    # gin.bind_parameter('training.output_dir', FLAGS.output_dir)
    # gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    # gin.parse_config_files_and_bindings(FLAGS.gin_file, bindings=None)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)
    iteration = FLAGS.iter
    # ========================================= DEBUG =========================================
    # model = runner.model.renderer.gs_net

    # ========================================= DEBUG =========================================
    #
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters: {num_params}")
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # # print(RunnerClass)
    # # sys.exit()
    # model = model.to(device_id)
    # model = DDP(model, device_ids=[device_id])
    # if FLAGS.only_eval == False:
    #   model.train()
    #   # 3. Optimizer
    #   optimizer, scheduler = {}, {}
    #   with gin.config_scope('pretrain'):
    #     optimizer['pretrain'] = build_optimizer(model.module)
    #     scheduler['pretrain'] = build_scheduler(optimizer['pretrain'])
    #   with gin.config_scope('train2D'):
    #     optimizer['train2D'] = build_optimizer(model.module)
    #     scheduler['train2D'] = build_scheduler(optimizer['train2D'])
    # # if rank==0:
    # #     wandb_run = wandb.init(project='debug_new_project', dir=FLAGS.wandb_dir)
    # #     if FLAGS.output_dir[-1] == '/':
    # #         FLAGS.output_dir = FLAGS.output_dir[:-1]
    # #     wandb.run.name = '/'.join(FLAGS.output_dir.split('/')[-2:])
    # # ==============================================================================================
    # # if rank == 0:
    # #     wandb_run = wandb.init(project='sign', dir=FLAGS.wandb_dir)  # resume=?

    # train_loader = ReconstructionDataset(split='train')
    # test_loader = ReconstructionDataset(split='test')
    # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)

    # from data.hand_dataset import create_dataset, merge_batch
    # print('===============================')
    # print("\033[91m[Loading MANO Training Dataset]\033[0m")
    # hand_train_loader = create_dataset('train')
    # dataloader = make_dataloader(hand_train_loader, shuffle=True, batch_size=2)

    #
    # train_loader = HandDataset(split='train')
    # # test_loader = ReconstructionDataset(split='test')
    # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)
    # # dataloader_1 = make_dataloader(train_loader, shuffle=True, batch_size=2)
    #
    #
    # data_iterator = iter(dataloader)
    # # data_iterator_1 = iter(dataloader_1)

    
    train_loader = Dataset(dataset_path='./data/interhand',
                           data_type='train')
    test_loader = Dataset(dataset_path='./data/interhand',
                           data_type='val',
                           skip=100)
    # test_handavatar = Dataset(dataset_path='./data/interhand',
    #                        data_type='val',
    #                        skip=100)
    # train_loader = HanCo(dataset_path='./data/hanco')


    # interhand_ds = Dataset(dataset_path='./data/interhand',
    #                        data_type='train')
    # handco_ds = HanCo(dataset_path='./data/hanco')


    train_dataloader = make_dataloader(train_loader, shuffle=True, batch_size=FLAGS.batch_size,
                                       num_workers=FLAGS.num_workers,
                                       persistent_workers=(FLAGS.num_workers>0),
                                       pin_memory=FLAGS.pin_memory)
    # test_dataloader = make_dataloader(test_loader, shuffle=True, batch_size=1)

    test_handavatar = HandAvatarDataset(dataset_path='/scratch/groups/su004-neuralnet/zh174/InterHand/5',
                                                d_type='progress',
                                                skip=200)
    
    test_dataloader = make_dataloader(test_handavatar, shuffle=False, batch_size=1,
                                      num_workers=FLAGS.num_workers,
                                      persistent_workers=(FLAGS.num_workers>0),
                                      pin_memory=FLAGS.pin_memory)


    data_iterator = iter(train_dataloader)
    # data_iterator_1 = iter(dataloader_1)
    writer = SummaryWriter(log_dir=os.path.join(FLAGS.output_dir, 'exp1'))

    if FLAGS.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Evaluation-only mode: load target checkpoint and run test once.
    if FLAGS.only_eval:
        ckpt_file_flag = None
        ckpt_path_flag = None
        try:
            ckpt_file_flag = FLAGS['checkpoint-file'].value
            ckpt_path_flag = FLAGS['checkpoint-path'].value
        except Exception:
            ckpt_file_flag = getattr(FLAGS, 'checkpoint_file', None)
            ckpt_path_flag = getattr(FLAGS, 'checkpoint_path', None)

        ckpt_file = cli_checkpoint_file or ckpt_file_flag or './checkpoint/interhand/iteration_4000.ckpt'
        ckpt_path = cli_checkpoint or ckpt_path_flag

        iter_from_name = None
        m = re.search(r'iteration_(\d+)\.ckpt$', ckpt_file)
        if m:
            iter_from_name = int(m.group(1))

        _load_runner_checkpoint(
            runner,
            iteration=iter_from_name,
            checkpoint_path=ckpt_path,
            checkpoint_file=ckpt_file,
        )
        print(f"\033[91m[========== Running ONLY-EVAL from checkpoint: {ckpt_file} ==========]\033[0m")

        infer_batch = test_handavatar.get_img()
        gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)
        eval_iteration = iter_from_name if iter_from_name is not None else int(FLAGS.iter)
        current_metrics = _run_test_with_fps(
            runner,
            test_dataloader,
            gs_model_list,
            gs_densify_list,
            query_points,
            eval_iteration,
        )
        current_metrics['iteration_tested_on'] = int(eval_iteration)
        current_metrics['timestamp'] = time.time()
        current_metrics['checkpoint_file'] = ckpt_file

        results_base = runner.checkpoint_path
        if ckpt_file is not None and os.path.isfile(ckpt_file):
            results_base = os.path.dirname(ckpt_file)
        test_results_dir = os.path.join(results_base, 'test_results')
        os.makedirs(test_results_dir, exist_ok=True)
        latest_results_path = os.path.join(test_results_dir, 'latest_test_results.json')
        with open(latest_results_path, 'w') as f:
            json.dump(current_metrics, f, indent=2)

        writer.close()
        print("ONLY-EVAL completed.")
        return

    torch.autograd.set_detect_anomaly(False)
    # ==============================================================================================
    # pbar = tqdm(range(1, 2000 + 1))
    pbar = tqdm(range(1, iteration + 1), desc='Training', dynamic_ncols=True)
    # num_epochs = 20
    for iteration in pbar:
    # for epoch in range(num_epochs):
        # batches = next(data_iterator)   # 随机采样一个 subject ID, 并查找当前 ID 的所有视频
        # batches = merge_batch(batches)
        try:
            batches = next(data_iterator)
            # batches_1 = next(data_iterator_1)
        except:
            data_iterator = iter(train_dataloader)
            batches = next(data_iterator)
        # batches = merge_batch(batches)
        runner.run(batch=batches, scaler=scaler, iteration=iteration, writer=writer, pbar=pbar)    # hand
        # runner.run(batch=batches, scaler=scaler, optimizer=optimizer, scheduler=scheduler)    # human
        # runner.run()    # infer funtion


        # if iteration % 5 == 0:
        if iteration >= 1000 and iteration % 1000 == 0:

            # runner.save(iteration=iteration, is_latest=False)

            runner.save(is_latest=True)

            runner.load(is_latest=True)
            print(f"\033[91m[========== Running test ! Loading model for iteration {iteration} ==========]\033[0m")

            infer_batch = test_handavatar.get_img()
            # infer_img = infer_batch['original_image'].permute(1,2,0).to('cuda')
            # infer_pts = infer_batch['big_pose_world_vertex'].to('cuda')
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

            current_metrics = _run_test_with_fps(
                runner,
                test_dataloader,
                gs_model_list,
                gs_densify_list,
                query_points,
                iteration,
            )


            # aggregate metrics
            current_metrics['iteration_tested_on'] = int(iteration)
            current_metrics['timestamp'] = time.time()

            # create test results folder
            test_results_dir = os.path.join(runner.checkpoint_path, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)

            latest_results_path = os.path.join(test_results_dir, 'latest_test_results.json')
            best_results_path = os.path.join(test_results_dir, 'best_test_results.json')

            # save latest results
            with open(latest_results_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)

            # # load best if exists
            # best_metrics = None
            # if os.path.isfile(best_results_path):
            #     with open(best_results_path, 'r') as f:
            #         best_metrics = json.load(f)

            # 5) 判断是否已有历史 best
            if os.path.exists(best_results_path):
                try:
                    with open(best_results_path, 'r') as f:
                        best_metrics = json.load(f)
                except Exception as e:
                    print(f"[Warning] failed to load best_results_path ({best_results_path}): {e}")
                    best_metrics = None
            else:
                # 第一次测试：没有历史 best -> 直接把当前设为 best（并保存 checkpoint）
                best_metrics = None

            # ---- 比较是否更优 ----
            # 6) 若没有历史 best（第一次）或当前比 best 更优，则更新 best 并保存 iteration_ckpt
            if (best_metrics is None) or _is_better(current_metrics, best_metrics):
                # 更新 best json
                best_metrics = current_metrics
                with open(best_results_path, 'w') as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"\033[92m[+] New best model found at iteration {iteration}! Saving...]\033[0m")
                runner.save_checkpoint(iteration=iteration, is_latest=False)
            else:
                print(f"[ ] No improvement at iter {iteration}. Best stays at iteration {best_metrics.get('iteration_tested_on', 'N/A')} (LPIPS {best_metrics['lpips']:.6f}, PSNR {best_metrics['psnr']:.4f})")



    runner.save(iteration)
    writer.close()

    print("Training completed.")



if __name__ == "__main__":
    # main()
    app.run(main)



# python train_interhand.py infer.hand_lrm  model_name=LHM-1B