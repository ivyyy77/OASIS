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
from data.interhand.train import Dataset, HandAvatarDataset, HanCo, MixedDataset, Mix_Dataset, Hands11k_Dataset
from data.debug import make_dataloader
# ensure subdivided MANO cache exists to avoid repeated expensive upsampling
try:
    import os
    import torch
    from tools.model.smplx.manohd.subdivide import sub_mano
    from tools_utils.model.ohta.configs import cfg as _ohta_cfg
    from tools_utils.model import smplx as _smplx_module

    _t = int(_ohta_cfg.smpl_cfg.get("manohd", 0))
    cache_path = os.path.join("pretrained_models", "mano_subdiv", f"mano_subdiv_{_t}.pth")
    if _t > 0 and not os.path.exists(cache_path):
        print(f"[PREWARM] subdivided MANO cache missing for t={_t}, computing once to warm cache (this may take time)...")
        mano_tmp = _smplx_module.create(**_ohta_cfg.smpl_cfg)
        try:
            sub_mano(mano_tmp, _t)
            print(f"[PREWARM] cached subdivided MANO saved to {cache_path}")
        except Exception as e:
            print(f"[PREWARM] Failed to prewarm subdivided MANO: {e}")
except Exception:
    # don't block training if prewarm fails; fallback to on-demand behavior
    pass
import torch.nn as nn
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
import time
import json
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
from torch.utils.data import DataLoader

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



def main(argv):


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
        # argparse default is True; prefer explicit argparse, otherwise use absl
        if not hasattr(args, 'resume') or args.resume is None:
            cli_resume = getattr(absl_f, 'resume', True)
    except Exception:
        pass

    # backward-compat: pick up unknown 'checkpoint=..' or 'output=..' only if not provided
    for u in unknown:
        if isinstance(u, str) and u.lower().startswith("checkpoint=") and cli_checkpoint is None:
            cli_checkpoint = u.split("=", 1)[1]
        if isinstance(u, str) and u.lower().startswith("output=") and cli_output is None:
            cli_output = u.split("=", 1)[1]

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

    # ==============================================================================================
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500',#'env://',
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
    
    # train_loader = Dataset(dataset_path='./data/interhand',
    #                        data_type='train')
    # test_loader = Dataset(dataset_path='./data/interhand',
    #                        data_type='val',
    #                        skip=100)
    # train_loader = HanCo(dataset_path='./data/hanco')
    hands11k_ds = Hands11k_Dataset(dataset_path='./data/hands11k/processed_test',
                                   data_type='train')
    # train_dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)

    interhand_ds = Dataset(dataset_path='./data/interhand',
                           data_type='train')
    handco_ds = HanCo(dataset_path='./data/hanco', data_type='train')

    train_loader = MixedDataset([interhand_ds, handco_ds, hands11k_ds], [0.25, 0.25, 0.5])
    # train_loader = MixedDataset([interhand_ds, handco_ds, hands11k_ds], [0.1, 0.1, 0.8])
    # train_loader = MixedDataset([interhand_ds, handco_ds], [0.5, 0.5])
    
    train_dataloader = make_dataloader(train_loader, shuffle=True, batch_size=FLAGS.batch_size,
                                       num_workers=FLAGS.num_workers,
                                       persistent_workers=(FLAGS.num_workers>0),
                                       pin_memory=FLAGS.pin_memory)

    
    # test_hand = HanCo(dataset_path='./data/hanco',
    #                         split='test')
    test_hand = HandAvatarDataset(dataset_path='/scratch/groups/su004-neuralnet/zh174/InterHand/5',
                                                data_type='progress',
                                                skip=200)
    
    
    test_dataloader = make_dataloader(test_hand, shuffle=False, batch_size=1,
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
    torch.autograd.set_detect_anomaly(False)
    # ==============================================================================================
    # pbar = tqdm(range(1, 2000 + 1))
    pbar = tqdm(range(1, iteration + 1), desc='Training', dynamic_ncols=True)
    # num_epochs = 20
    for iteration in pbar:
        try:
            batches = next(data_iterator)
            # batches_1 = next(data_iterator_1)
        except:
            data_iterator = iter(train_dataloader)
            batches = next(data_iterator)
        # batches = merge_batch(batches)
        runner.run(batch=batches, scaler=scaler, iteration=iteration, writer=writer, pbar=pbar)    # hand

        # if iteration % 5 == 0:
        if iteration >= 1000 and iteration % 1000 == 0:

            # runner.save(iteration=iteration, is_latest=False)

            runner.save(is_latest=True)

            runner.load(is_latest=True)
            print(f"\033[91m[========== Running test ! Loading model for iteration {iteration} ==========]\033[0m")

            psnr_full = 0
            ssim_full = 0
            lpips_full = 0
            image_full = 0

            count = 0

            infer_batch = test_hand.get_img()
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)


            # 外层枚举整个 test_dataloader
            for idx, test_batch in enumerate(test_dataloader):
                # 每个 batch 单独一个 tqdm，并且 position = idx+1 保证占用新的一行
                with tqdm(
                    total=1,                              # 只处理一个 batch
                    desc=f"Iter {iteration} | Sample {idx+1}/{len(test_dataloader)}",
                    leave=True,                            # 保留进度条
                    position=idx + 1,                       # 每个 batch 占一行
                    dynamic_ncols=True
                ) as pbar:

                    # 关闭梯度避免显存爆炸
                    with torch.no_grad():
                        loss_dict = runner.test_handavatar( gs_model_list, gs_densify_list, query_points,
                                                            test_batch, idx, iteration, pbar)

                    psnr_full  += loss_dict['psnr'].item()
                    ssim_full  += loss_dict['ssim'].item()
                    lpips_full += loss_dict['lpips'].item()
                    # image_full += loss_dict['image_l1'].item()
                    count += 1

                    # 在该 batch 的进度条上显示平均值与当前值
                    pbar.set_postfix({
                        "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
                        "ssim":  f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
                        "psnr":  f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
                        # "L1":    f"{image_full/count:.4f} ({loss_dict['image_l1'].item():.4f})",
                    })

                    pbar.update(1)

                    # 及时释放 GPU 缓存
                    torch.cuda.empty_cache()

            print(f"\033[91m[Summary @iter {iteration}]\033[0m "
                f"PSNR: {psnr_full/count:.4f}, "
                f"SSIM: {ssim_full/count:.4f}, "
                f"LPIPS: {lpips_full/count:.4f}, "
                f"L1: {image_full/count:.4f}\n")


            # aggregate metrics
            current_metrics = {
                'lpips': lpips_full / count,
                'ssim': ssim_full / count,
                'psnr': psnr_full / count,
                'image_l1': image_full / count,
                'iteration_tested_on': int(iteration),
                'timestamp': time.time()
            }

            # create test results folder
            test_results_dir = os.path.join(runner.checkpoint_path, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)

            latest_results_path = os.path.join(test_results_dir, 'latest_test_results.json')
            best_results_path = os.path.join(test_results_dir, 'best_test_results.json')

            # save latest results
            with open(latest_results_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)

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





if __name__ == "__main__":
    # main()
    app.run(main)



# python train_interhand.py infer.hand_lrm  model_name=LHM-1B