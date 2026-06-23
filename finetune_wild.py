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

import time
import json

from tqdm import tqdm
# from sklearn.preprocessing import maxabs_scale

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
from LHM.losses import _is_better

from torch.utils.tensorboard import SummaryWriter
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)



flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', './wandb/', 'Wandbs Output directory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
# CLI flags for checkpoint / output and resume (compatible with absl.flags)
flags.DEFINE_string('checkpoint-path', None, 'Checkpoint directory to save/load checkpoints')
flags.DEFINE_string('output-path', './output/finetune', 'Output/test directory to write results')
flags.DEFINE_boolean('resume', True, 'Enable auto-resume from latest checkpoint')
flags.DEFINE_string('checkpoint-file', './checkpoint/base_feat_1/iteration_4000.ckpt', 'Specific checkpoint file to load (overrides checkpoint-path)')
flags.DEFINE_string('input-dir', None, 'Optional directory containing in-the-wild images to process (process all files)')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_boolean('use_amp', True, 'Use automatic mixed precision (AMP)')
flags.DEFINE_multi_string(
  'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')
# flags.DEFINE_integer('iter', '2', 'Training iteration')
flags.DEFINE_integer('test-iter', 9000, 'Test iteration')
flags.DEFINE_integer('iter', '200', 'Finetuning iteration')
flags.DEFINE_boolean('animate_to_handavatar', True, 'Whether to animate inferred GS to HandAvatar test poses')


FLAGS = flags.FLAGS

# Default finetune launcher values so runner/model_name can be omitted from CLI.
DEFAULT_RUNNER = "infer.hand_lrm"
DEFAULT_MODEL_NAME = "LHM-1B"
os.environ.setdefault("APP_MODEL_NAME", DEFAULT_MODEL_NAME)

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
    parser.add_argument("--runner", default=DEFAULT_RUNNER, type=str, help="Runner to launch")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Optional checkpoint directory to save/load checkpoints (overrides env)")
    parser.add_argument("--checkpoint-file", type=str, default='./checkpoint/base_feat_1/iteration_4000.ckpt',
                        help="Optional specific checkpoint file to load (overrides checkpoint-path)")
    parser.add_argument("--output-path", type=str, default='./output/finetune',
                        help="Optional output/test directory to write results (overrides env)")
    parser.add_argument("--handavatar-path", type=str, default=None,
                        help="Optional HandAvatar dataset path for mapping")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Optional directory containing in-the-wild images to process (process all files)")
    parser.add_argument("--test-iter", type=int, default=None,
                        help="Iteration to load and test (overrides hardcoded value)")
    args, unknown = parser.parse_known_args()

    # Initialize CLI placeholders so later logic can safely check them
    cli_checkpoint = None
    cli_checkpoint_file = None
    cli_output = None
    cli_handavatar = None
    cli_input_dir = None

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
        elif k in ('handavatar-path', 'handavatar_path') and cli_handavatar is None:
            cli_handavatar = value
        elif k in ('input-dir', 'input_dir') and cli_input_dir is None:
            cli_input_dir = value
        elif k in ('test-iter', 'test_iter') and getattr(args, 'test_iter', None) is None:
            try:
                args.test_iter = int(value)
            except Exception:
                pass

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError("Runner {} not found".format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]
    # Prepare CLI values (compat with absl flags). Only fill from argparse
    # when not already provided by the debugger/launcher normalization above.
    if cli_checkpoint is None:
        cli_checkpoint = args.checkpoint_path
    if cli_checkpoint_file is None:
        cli_checkpoint_file = args.checkpoint_file
    if cli_output is None:
        cli_output = args.output_path
    if cli_handavatar is None:
        cli_handavatar = args.handavatar_path
    if cli_input_dir is None:
        cli_input_dir = args.input_dir
    try:
        from absl import flags as _absl_flags
        absl_f = _absl_flags.FLAGS
        if cli_checkpoint is None and getattr(absl_f, 'checkpoint_path', None):
            cli_checkpoint = absl_f.checkpoint_path
        if cli_checkpoint_file is None and getattr(absl_f, 'checkpoint_file', None):
            cli_checkpoint_file = absl_f.checkpoint_file
        if cli_output is None and getattr(absl_f, 'output_path', None):
            cli_output = absl_f.output_path
        if cli_handavatar is None and getattr(absl_f, 'handavatar_path', None):
            cli_handavatar = absl_f.handavatar_path
    except Exception:
        pass

    # Try to construct runner with explicit CLI args if it accepts them
    try:
        runner = RunnerClass(checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file, output_path=cli_output)
    except TypeError:
        runner = RunnerClass()
        try:
            if hasattr(runner, 'set_cli_args'):
                runner.set_cli_args(checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file, output_path=cli_output)
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

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)
    # ========================================= DEBUG / in-the-wild =========================================
    # model = runner.model.renderer.gs_net
    # If an input directory is provided, process all images in it one-by-one.
    import glob
    img_glob = []  # Initialize img_glob
    if cli_input_dir is not None and os.path.exists(cli_input_dir):
        # Case 1: input is a directory -> collect images under it
        if os.path.isdir(cli_input_dir):
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                img_glob.extend(glob.glob(os.path.join(cli_input_dir, ext)))
            img_glob = sorted(img_glob)
            # only treat as "multi-image mode" when there are multiple images
            multi_image_mode = (len(img_glob) > 1)

        # Case 2: input is a file -> treat as a single image if extension matches
        elif os.path.isfile(cli_input_dir):
            _, ext = os.path.splitext(cli_input_dir)
            if ext.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                img_glob = [cli_input_dir]
                multi_image_mode = False
    else:
        multi_image_mode = False


    # Extract image name for checkpoint naming (same as finetune_ohta)
    if multi_image_mode or not img_glob:
        image_name = "default"
    else:
        image_name = os.path.splitext(os.path.basename(img_glob[0]))[0]

    # Determine base output root: prefer explicit --output-path, fallback to FLAGS.output_dir
    base_output_root = cli_output or FLAGS.output_dir

    # Create image-specific checkpoint directory under the chosen root
    image_checkpoint_dir = os.path.join(base_output_root, f"finetune_{image_name}")
    os.makedirs(image_checkpoint_dir, exist_ok=True)

    # Update runner's checkpoint path and test output path to use image-specific directory
    if hasattr(runner, 'checkpoint_path'):
        runner.checkpoint_path = image_checkpoint_dir
    else:
        setattr(runner, 'checkpoint_path', image_checkpoint_dir)

    # Set test_path to be inside checkpoint directory (test images saved together with checkpoints)
    test_output_dir = os.path.join(image_checkpoint_dir, 'test_images')
    os.makedirs(test_output_dir, exist_ok=True)
    if hasattr(runner, 'test_path'):
        runner.test_path = test_output_dir
    else:
        setattr(runner, 'test_path', test_output_dir)

    # Print configuration summary
    print(f"\033[94m{'='*80}\033[0m")
    print(f"\033[94mFine-tuning (wild) Configuration:\033[0m")
    print(f"  Image: {image_name}")
    print(f"  Base output root: {base_output_root}")
    print(f"  Checkpoint dir: {image_checkpoint_dir}")
    print(f"  Test output dir: {test_output_dir}")
    print(f"  Input image path: {img_glob[0] if img_glob else 'N/A'}")
    print(f"\033[94m{'='*80}\033[0m\n")

    if FLAGS.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    torch.autograd.set_detect_anomaly(False)
    writer = SummaryWriter(log_dir=os.path.join(image_checkpoint_dir, 'logs'))


    # If not in multi-image mode, keep legacy single-sample behavior
    if not multi_image_mode:
        if not img_glob:
            raise ValueError("No input image provided. Please specify --input-dir with a valid image path.")
        test_hand = HandDataset(split='test_wild', img_path=img_glob[0])
        # Some HandDataset variants may not expose `video_ids`; using a dataloader keeps code paths compatible
        test_dataloader = make_dataloader(test_hand, shuffle=False, batch_size=1)
    # try:
    #     # prefer a DataLoader when dataset is well-formed
    #     if hasattr(test_hand, 'video_ids'):
    #         test_dataloader = make_dataloader(test_hand, shuffle=False, batch_size=1)
    #     else:
    #         print("[Warning] HandDataset missing `video_ids`; skipping test dataloader enumeration.")
    #         test_dataloader = []
    # except Exception as e:
    #     print(f"[Warning] failed to create test dataloader: {e}; skipping enumeration.")
    #     test_dataloader = []
    

    # determine which iteration to test:
    # - prefer CLI `--test-iter` when given
    # - else if a `--checkpoint-file` was provided, try to extract iteration from its filename
    # - otherwise fall back to a sensible default
    import re
    if args.test_iter is not None:
        test_iters = {args.test_iter}
    elif cli_checkpoint_file is not None:
        # try patterns like iteration_9000.ckpt or iteration-9000.ckpt
        base = os.path.basename(cli_checkpoint_file)
        m = re.search(r'iteration[_-]?(\d+)', base)
        if m:
            test_iters = int(m.group(1))
        else:
            # fallback: first long digit sequence in filename
            m2 = re.search(r'(\d{3,7})', base)
            if m2:
                test_iters = int(m2.group(1))
            else:
                test_iters = 12000
    else:
        test_iters = 12000

    
    try:
        if hasattr(runner, 'load_checkpoint'):
            # runner.load_checkpoint(iteration=iteration, is_latest=False, checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file)
            runner.finetune_model(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file)
        else:
            runner.load(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint)
    except TypeError:
        # fallback to old loader
        runner.load(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint)
    
    print(f"\033[91m[========== Running Finetune ! Loading model for iteration {test_iters} ==========]\033[0m")

    # Reset checkpoint_path AFTER loading pretrained model so fine-tuned
    # checkpoints and tests are written into our per-image directory.
    runner.checkpoint_path = image_checkpoint_dir
    print(f"\033[94m[Debug] Reset runner.checkpoint_path to: {runner.checkpoint_path}\033[0m")
    if hasattr(runner, 'hand_model') and hasattr(runner.hand_model, 'checkpoint_path'):
        runner.hand_model.checkpoint_path = image_checkpoint_dir
        print(f"\033[94m[Debug] Reset runner.hand_model.checkpoint_path to: {runner.hand_model.checkpoint_path}\033[0m")


    # Load HandAvatar test dataset if animate-to-handavatar is enabled
    handavatar_dataloader = None
    if FLAGS.animate_to_handavatar:
        try:
            handavatar_path = '/scratch/groups/su004-neuralnet/zh174/InterHand/5'
            print(f"\033[94mLoading HandAvatar dataset from: {handavatar_path}\033[0m")
            handavatar_ds = HandAvatarDataset(dataset_path=handavatar_path, data_type='progress', skip=200)
            handavatar_dataloader = make_dataloader(handavatar_ds, shuffle=False, batch_size=1)
            print(f"\033[92m[✓] Loaded HandAvatar dataset with {len(handavatar_dataloader)} samples\033[0m")
        except Exception as e:
            print(f"\033[91m[Warning] Failed to load HandAvatar dataset: {e}\033[0m")
            handavatar_dataloader = None
    # ====================== Eval-only branch ======================
    if FLAGS.only_eval:
        # Use the loaded model (pretrained or existing checkpoint) and run
        # evaluation directly without any further fine-tuning.
        eval_iter = FLAGS.iter
        if isinstance(eval_iter, set):
            # extract a single integer if we accidentally stored a set
            eval_iter = next(iter(eval_iter))
        print(f"\033[91m[========== Running EVAL ONLY @ iter {eval_iter} ==========]\033[0m")
        runner.load_checkpoint(iteration=eval_iter, is_latest=False, checkpoint_path=image_checkpoint_dir)

        psnr_full = 0
        ssim_full = 0
        lpips_full = 0
        image_full = 0
        count = 0

        # Build GS once from the input image
        # infer_batch = test_hand.get_img()
        infer_batch = test_hand.get_img_rainbow()
        gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

        # Optionally animate GS to HandAvatar poses/cameras
        if handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
            print(f"\033[94m[Animating GS to HandAvatar poses for iteration {eval_iter}]\033[0m")
            for map_idx, map_batch in enumerate(handavatar_dataloader):
                if isinstance(map_batch, dict) and 'dataset_id' not in map_batch:
                    map_batch['dataset_id'] = 0
                elif not isinstance(map_batch, dict):
                    try:
                        setattr(map_batch, 'dataset_id', 0)
                    except Exception:
                        pass
                try:
                    map_batch[0]['dataset_id'] = 0
                except Exception:
                    pass

                with torch.no_grad():
                    _ = runner.test_handavatar(gs_model_list, gs_densify_list, query_points,
                                               map_batch, map_idx, eval_iter, pbar=None)
            print(f"\033[92m[✓] Completed HandAvatar pose animation\033[0m")

        for idx, test_batch in enumerate(test_dataloader):
            with tqdm(
                total=1,
                desc=f"Eval {eval_iter} | Sample {idx+1}/{len(test_dataloader)}",
                leave=True,
                position=idx + 1,
                dynamic_ncols=True,
            ) as pbar:
                with torch.no_grad():
                    # test_batch[0]['dataset_id'] = 2
                    loss_dict = runner.test_handavatar(
                        gs_model_list,
                        gs_densify_list,
                        query_points,
                        test_batch,
                        idx,
                        eval_iter,
                        pbar,
                    )

                psnr_full += loss_dict['psnr'].item()
                ssim_full += loss_dict['ssim'].item()
                lpips_full += loss_dict['lpips'].item()
                count += 1

                pbar.set_postfix(
                    {
                        "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
                        "ssim": f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
                        "psnr": f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
                    }
                )
                pbar.update(1)
                torch.cuda.empty_cache()

        print(
            f"\033[91m[Summary @eval {eval_iter}]\033[0m "
            f"PSNR: {psnr_full/count:.4f}, "
            f"SSIM: {ssim_full/count:.4f}, "
            f"LPIPS: {lpips_full/count:.4f}\n"
        )

        current_metrics = {
            'image_name': image_name,
            'lpips': psnr_full / count if count > 0 else 0.0,
            'ssim': ssim_full / count if count > 0 else 0.0,
            'psnr': psnr_full / count if count > 0 else 0.0,
            'image_l1': image_full / count if count > 0 else 0.0,
            'iteration': int(eval_iter),
            'timestamp': time.time(),
        }

        metrics_path = os.path.join(image_checkpoint_dir, f'metrics_eval_iter{int(eval_iter)}.json')
        with open(metrics_path, 'w') as f:
            json.dump(current_metrics, f, indent=2)
        print(f"\033[92m[✓] Saved eval metrics to {metrics_path}\033[0m")
        return


    # ====================== Finetune + test branch ======================

    iteration = FLAGS.iter

    data_iterator = iter(test_dataloader)
    pbar = tqdm(range(1, iteration + 1), desc='Finetuning', dynamic_ncols=True)
    # num_epochs = 20
    for iteration in pbar:
        try:
            batches = next(data_iterator)
            # batches_1 = next(data_iterator_1)
        except:
            data_iterator = iter(test_dataloader)
            batches = next(data_iterator)
        # batches = merge_batch(batches)
        # batches[0]['dataset_id'] = 2   # handavatar dataset id = 2
        if 'text-to-avatar' in cli_input_dir:
            batches[0]['dataset_id'] = 0        
        runner.run_wild(batch=batches, scaler=scaler, iteration=iteration, writer=writer, pbar=pbar, total_iters=FLAGS.iter)    # hand

        # Save final checkpoint and run testing once at the last iteration,
        # mirroring the simpler behavior in finetune_ohta. 
        
        # if iteration % 100 == 0 and iteration >= 300:
        #     if iteration == 600:
        #         exit(0)
        # if iteration % 500 == 0 or iteration == 400 or iteration == FLAGS.iter: 
        # if iteration == 201:
        #     exit(0)
        # if iteration >= 100 and iteration % 100 == 0: # or (iteration >=60 and iteration % 10 == 0): 
        # if iteration == FLAGS.iter or iteration % 1000 == 0 or iteration == 500 or (iteration > 100 and iteration <= 500 and iteration % 100 == 0): 
        if iteration == FLAGS.iter or iteration % 500 == 0:
            checkpoint_name = f"{image_name}_iter{iteration}"
            runner.save(iteration=iteration, is_latest=False)
            print(f"\033[92m[✓] Saved checkpoint: {checkpoint_name} at iteration {iteration}\033[0m")

            runner.load(iteration=iteration, is_latest=False, checkpoint_path=image_checkpoint_dir)

            print(f"\033[91m[========== Running test ! Loading model for iteration {iteration} ==========]\033[0m")

            psnr_full = 0
            ssim_full = 0
            lpips_full = 0
            image_full = 0

            count = 0

            infer_batch = test_hand.get_img_rainbow()
            # if 'text-to-avatar' in cli_input_dir:
            #     infer_batch = test_hand.get_img_rainbow()
            # else:
            #     infer_batch = test_hand.get_img()
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)


            # --- Animate inferred GS to HandAvatar test poses/cameras ---
            if handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
                print(f"\033[94m[Animating GS to HandAvatar poses for iteration {iteration}]\033[0m")
                for map_idx, map_batch in enumerate(handavatar_dataloader):
                    # Mark batch as HandAvatar dataset (dataset_id=0 for InterHand)
                    if isinstance(map_batch, dict) and 'dataset_id' not in map_batch:
                        map_batch['dataset_id'] = 0
                    elif not isinstance(map_batch, dict):
                        try:
                            setattr(map_batch, 'dataset_id', 0)
                        except Exception:
                            pass
                    try:
                        map_batch[0]['dataset_id'] = 0
                    except Exception:
                        pass

                    # Render GS onto HandAvatar pose/camera
                    with torch.no_grad():
                        _ = runner.test_handavatar(gs_model_list, gs_densify_list, query_points,
                                                   map_batch, map_idx, iteration, pbar=None)
                print(f"\033[92m[✓] Completed HandAvatar pose animation\033[0m")


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
                        # test_batch[0]['dataset_id'] = 2
                        if 'text-to-avatar' in cli_input_dir:
                            test_batch[0]['dataset_id'] = 0
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


            # aggregate metrics (simple per-iteration record, same style as finetune_ohta)
            current_metrics = {
                'image_name': image_name,
                'lpips': lpips_full / count,
                'ssim': ssim_full / count,
                'psnr': psnr_full / count,
                'image_l1': image_full / count,
                'iteration': int(iteration),
                'timestamp': time.time()
            }

            metrics_path = os.path.join(image_checkpoint_dir, f'metrics_iter{iteration}.json')
            with open(metrics_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)
            print(f"\033[92m[✓] Saved metrics to {metrics_path}\033[0m")





if __name__ == "__main__":
    # main()
    app.run(main)



