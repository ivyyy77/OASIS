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

from tqdm import tqdm
from sklearn.preprocessing import maxabs_scale

from LHM.runners import REGISTRY_RUNNERS
# from splatformer.models.feature_predictor import FeaturePredictor
from torch.nn import Tanh, Identity

from data.debug import ReconstructionDataset, HandDataset
from data.interhand.train import Dataset, HandAvatarDataset
from data.debug import make_dataloader
import torch.nn as nn
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
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
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)



flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', './wandb/', 'Wandbs Output directory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_multi_string(
  'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')
# flags.DEFINE_integer('iter', '2', 'Training iteration')
# flags.DEFINE_integer('iter', '11757', 'Training iteration')
flags.DEFINE_integer('iter', '50000', 'Training iteration')

# DataLoader and runtime options (align with train_mix_3)
flags.DEFINE_integer('num_workers', 4, 'Number of DataLoader workers')
flags.DEFINE_integer('batch_size', 2, 'Default training batch size')
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

    args, unknown = parser.parse_known_args()

    # Initialize CLI placeholders so later logic can safely check them
    cli_checkpoint = None
    cli_checkpoint_file = None
    cli_output = None
    cli_resume = None

    # Collect tokens to inspect: prefer argparse `unknown` (unrecognized args),
    # otherwise fall back to raw sys.argv. Support forms:
    #  - key=value
    #  - key value
    #  - single token with escaped space (e.g. "checkpoint-file\ ./path")
    raw_tokens = unknown if unknown else sys.argv[1:]
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        # skip conventional flags
        if tok.startswith('--') or tok.startswith('-'):
            i += 1
            continue

        # key=value style in one token
        if '=' in tok and not tok.startswith('-'):
            k, value = tok.split('=', 1)
            k = k.strip()
            value = value.strip()
            i += 1

        # token that already contains a space due to escaped-space in launcher
        elif ' ' in tok and not tok.startswith('-'):
            k, value = tok.split(' ', 1)
            k = k.strip()
            value = value.strip()
            i += 1

        # key and value are separate tokens
        elif not tok.startswith('-'):
            if i + 1 < len(raw_tokens) and not raw_tokens[i + 1].startswith('-') and '=' not in raw_tokens[i + 1]:
                k = tok.strip()
                value = raw_tokens[i + 1].strip()
                i += 2
            else:
                i += 1
                continue
        else:
            i += 1
            continue

        # assign normalized values if not already set
        if k in ('checkpoint-file', 'checkpoint_file') and cli_checkpoint_file is None:
            cli_checkpoint_file = value
        elif k in ('checkpoint-path', 'checkpoint_path') and cli_checkpoint is None:
            cli_checkpoint = value
        elif k in ('output-path', 'output_path') and cli_output is None:
            cli_output = value
        elif k in ('resume',) and cli_resume is None:
            try:
                cli_resume = bool(value)
            except Exception:
                pass

    # Prepare CLI values (compat with absl flags). Only fill from argparse
    # when not already provided by the debugger/launcher normalization above.
    if cli_checkpoint is None:
        cli_checkpoint = args.checkpoint_path
    if cli_checkpoint_file is None:
        cli_checkpoint_file = args.checkpoint_file
    if cli_output is None:
        cli_output = args.output_path
    if cli_resume is None and hasattr(args, 'resume'):
        cli_resume = args.resume
    try:
        from absl import flags as _absl_flags
        absl_f = _absl_flags.FLAGS
        if cli_checkpoint is None and getattr(absl_f, 'checkpoint_path', None):
            cli_checkpoint = absl_f.checkpoint_path
        if cli_checkpoint_file is None and getattr(absl_f, 'checkpoint_file', None):
            cli_checkpoint_file = absl_f.checkpoint_file
        if cli_output is None and getattr(absl_f, 'output_path', None):
            cli_output = absl_f.output_path
    except Exception:
        pass

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError("Runner {} not found".format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]
    # Try to construct runner with explicit CLI args if it accepts them
    try:
        runner = RunnerClass(checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file, output_path=cli_output, resume=cli_resume)
    except TypeError:
        runner = RunnerClass()
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
        except Exception:
            pass

    # ==============================================================================================
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29501',#'env://',
                            rank=0,       # Set appropriate rank per process
                            world_size=1,
                            # master_addr="127.0.0.1",
                            # master_port=29500
                            )   # Total number of processes)
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)
    iteration = FLAGS.iter
    # ========================================= DEBUG =========================================
    # model = runner.model.renderer.gs_net

    # test_loader = Dataset(dataset_path='./data/interhand',
    #                        data_type='val',
    #                        skip=100)
    # test_dataloader = make_dataloader(test_loader, shuffle=False, batch_size=2)
    

    test_handavatar = HandAvatarDataset(dataset_path='/scratch/groups/su004-neuralnet/zh174/InterHand/5',
                                                data_type='progress',
                                                skip=200)
    
    test_dataloader = make_dataloader(test_handavatar, shuffle=False, batch_size=1,
                                      num_workers=FLAGS.num_workers,
                                      persistent_workers=(FLAGS.num_workers>0),
                                      pin_memory=FLAGS.pin_memory)

    # ================================  Train ==============================================================
    # ================================  Train ==============================================================
    # ================================  Train ==============================================================
    
    # pbar = tqdm(range(1, iteration + 1), desc='Training', dynamic_ncols=True)
    # # num_epochs = 20
    # for iteration in pbar:
    # # for epoch in range(num_epochs):
    #     # batches = next(data_iterator)   # 随机采样一个 subject ID, 并查找当前 ID 的所有视频
    #     # batches = merge_batch(batches)
    #     try:
    #         batches = next(data_iterator)
    #         # batches_1 = next(data_iterator_1)
    #     except:
    #         data_iterator = iter(train_dataloader)
    #         batches = next(data_iterator)
    #     # batches = merge_batch(batches)
    #     runner.run(batch=batches, scaler=scaler, iteration=iteration, writer=writer, pbar=pbar)    # hand
    #     # runner.run(batch=batches, scaler=scaler, optimizer=optimizer, scheduler=scheduler)    # human
    #     # runner.run()    # infer funtion

    #     # if iteration % 5 == 0:
    #     if iteration % 5000 == 0:
    #         runner.save(iteration)
    #         # runner.test(merge_batch(batch_test), iteration)

    #         # # 测试：对整个 test_dataloader 做一次逐-batch 测试，用一个临时的子进度条
    #         # test_pbar = tqdm(test_dataloader, desc=f"Test @iter {iteration}", leave=False, position=1, dynamic_ncols=True)
    #         # for test_batch in test_pbar:
    #         #     # runner.test 应该处理单个 batch 的推理并返回/记录度量
    #         #     # runner.save(iteration)
    #         #     runner.test(test_batch, iteration, test_pbar)   # hand
                
    #         #     # 可选择把一些实时 metric 填到 test_pbar.postfix（如果 runner.test 返回 metric）
    #         #     # test_pbar.set_postfix({'some_metric': f"{metric:.4f}"})
    #         # test_pbar.close()            
    #         # # runner.test(batches, iteration)
    #         # # exit(0)
    # runner.save(iteration)
    # writer.close()

    # ================================  TEST ==============================================================
    # ================================  TEST ==============================================================
    # ================================  TEST ==============================================================

    # Determine which iteration(s) to test:
    # - prefer an explicit checkpoint file passed via CLI when available
    # - else fall back to a sensible default set
    import re
    # try to pick up checkpoint-file from args/absl
    try:
        cli_ckpt = None
        # prefer local variable if set earlier
        cli_ckpt = globals().get('cli_checkpoint_file', None)
        if cli_ckpt is None:
            # inspect raw argv for common debugger-style tokens
            raw = sys.argv[1:]
            # debug output to help trace what the launcher passed
            print(f"[DEBUG] sys.argv: {sys.argv}")
            print(f"[DEBUG] argparse unknown tokens: {unknown}")
            # normalize tokens: expand any escaped-space sequences like "\\ "
            norm_tokens = []
            for tok in raw:
                t = tok.replace('\\ ', ' ')
                t = t.strip()
                if ' ' in t:
                    # if normalization created a space, split into parts
                    parts = [p for p in t.split(' ') if p]
                    norm_tokens.extend(parts)
                else:
                    norm_tokens.append(t)
            print(f"[DEBUG] normalized argv tokens: {norm_tokens}")
            for i, tok in enumerate(norm_tokens):
                if tok.startswith('checkpoint-file=') or tok.startswith('checkpoint_file='):
                    cli_ckpt = tok.split('=', 1)[1]
                    break
                if tok in ('checkpoint-file', 'checkpoint_file') and i + 1 < len(norm_tokens):
                    cli_ckpt = norm_tokens[i + 1]
                    break
        if cli_ckpt:
            base = os.path.basename(cli_ckpt)
            m = re.search(r'iteration[_-]?(\d+)', base)
            if m:
                test_iters = {int(m.group(1))}
            else:
                m2 = re.search(r'(\d{3,7})', base)
                if m2:
                    test_iters = {int(m2.group(1))}
                else:
                    test_iters = {14000}
        else:
            test_iters = {14000}
    except Exception:
        test_iters = {14000}

    for iteration in test_iters:
        # runner.load_model(iteration)
        # runner.load(is_latest=True)
        # prefer explicit checkpoint path/file when loading
        try:
            if hasattr(runner, 'load_checkpoint'):
                runner.load_checkpoint(iteration=iteration, is_latest=False, checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file)
            else:
                runner.load(iteration=iteration, is_latest=False, checkpoint_path=cli_checkpoint)

            # runner.load(iteration=iteration, is_latest=False, checkpoint_path=globals().get('cli_checkpoint', None), checkpoint_file=globals().get('cli_checkpoint_file', None))
        except TypeError:
            # older runner.load signature may not accept checkpoint args
            runner.load(iteration=iteration, is_latest=False)
        print(f"\033[91m[========== Running test ! Loading model for iteration {iteration} ==========]\033[0m")

        psnr_full = 0
        ssim_full = 0
        lpips_full = 0
        image_full = 0

        count = 0

        infer_batch = test_handavatar.get_img()
        # infer_batch = test_handavatar.get_img_04()
        # infer_img = infer_batch['original_image'].permute(1,2,0).to('cuda')
        # infer_pts = infer_batch['big_pose_world_vertex'].to('cuda')
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
        # f"\033[91m[========== Running test for iteration {iteration} ==========]\033[0m"


if __name__ == "__main__":
    # main()
    app.run(main)



