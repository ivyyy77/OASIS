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
import re
import cv2

from tqdm import tqdm

from LHM.runners import REGISTRY_RUNNERS
# from splatformer.models.feature_predictor import FeaturePredictor
from torch.nn import Tanh, Identity

from data.debug import ReconstructionDataset, HandDataset
from data.interhand.train import Dataset, HandAvatarDataset, HanCo
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


#  PYTHONPATH=$PWD LOCAL_CONDA/bin/conda run -n lhm --no-capture-output python LOCAL_GPFS_HOME/.vscode-server/extensions/ms-python.python-2026.0.0-linux-x64/python_files/get_output_via_markers.py test_hanco.py infer.hand_lrm model_name=LHM-1B --finetune_before_test --finetune_iters=200 --checkpoint-file ./checkpoint/base_feat_1/iteration_4000.ckpt --output-path ./output/finetune


flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', './wandb/', 'Wandbs Output directory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_boolean('use_amp', True, 'Use AMP during finetuning')
flags.DEFINE_string('checkpoint-path', None, 'Checkpoint directory to save/load checkpoints')
flags.DEFINE_string('checkpoint-file', './checkpoint/iteration_6000.ckpt', 'Specific checkpoint file to load (overrides checkpoint-path)')
flags.DEFINE_string('output-path', './output/finetune', 'Output/test directory to write results')
flags.DEFINE_multi_string(
  'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')
flags.DEFINE_integer('test-iter', 9000, 'Test iteration')
flags.DEFINE_integer('iter', '500', 'Total finetuning iteration')
# Two-stage inversion flags
flags.DEFINE_boolean('use_two_stage_inversion', False, 'Whether to use two-stage inversion (stage1 + stage2 with color consistency)')
flags.DEFINE_integer('iter_inversion_stage1', 100, 'Stage1 inversion iters (single-view color optimization)')
flags.DEFINE_integer('iter_inversion_stage2', 0, 'Stage2 inversion iters (pseudo-view color consistency)')
flags.DEFINE_float('color-consistency-weight', 1.0, 'Weight for color consistency loss in inversion stage2')
flags.DEFINE_integer('pseudo-views', 8, 'Number of pseudo-GT views')
flags.DEFINE_boolean('enable_pose_refine', False, 'Enable pose refinement during stage2 finetune')
flags.DEFINE_float('pose_refine_lr', 1e-4, 'Learning rate for pose refiner')
# HanCo image selection flags
flags.DEFINE_string('input_img', None, 'HanCo image path, e.g. rgb/0185/cam6/00000010.jpg (auto-parses vid/cam/imgid)')
flags.DEFINE_list('test_cams', None, 'Comma-separated list of camera IDs for testing (e.g. 0,3,6)')


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


def _run_hanco_test(runner, test_dataloader, gs_model_list, gs_densify_list, query_points, iteration):
    psnr_full = 0
    ssim_full = 0
    lpips_full = 0
    image_full = 0
    count = 0

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
        f"\033[91m[Summary @iter {iteration}]\033[0m "
        f"PSNR: {psnr_full/count:.4f}, "
        f"SSIM: {ssim_full/count:.4f}, "
        f"LPIPS: {lpips_full/count:.4f}, "
        f"L1: {image_full/count:.4f}\n"
    )


def _load_runner_checkpoint(runner, iteration, checkpoint_path=None, checkpoint_file=None):
    """Helper to load a checkpoint from various sources."""
    if hasattr(runner, 'load_checkpoint'):
        if checkpoint_file is not None:
            return runner.load_checkpoint(
                iteration=iteration,
                is_latest=False,
                checkpoint_path=checkpoint_path,
                checkpoint_file=checkpoint_file,
            )
        return runner.load_checkpoint(
            iteration=iteration,
            is_latest=False,
            checkpoint_path=checkpoint_path,
        )
    # Fallback to older load API without checkpoint_file support
    return runner.load(iteration=iteration, is_latest=False, checkpoint_path=checkpoint_path)



def main(argv):

    # train_loader = HandDataset(split='train')
    # # test_loader = ReconstructionDataset(split='test')
    # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=1)
    # # dataloader = make_dataloader(train_loader, shuffle=True, batch_size=2)


    # data_iterator = iter(dataloader)
    # # data_iterator_1 = iter(dataloader_1)
    # batches = next(data_iterator)


    # FLAGS.parse_args()  # 手动解析命令行参数
    # Inject defaults for runner and model_name so they don't need to be typed every time
    if not any(a == 'infer.hand_lrm' for a in sys.argv[1:]):
        sys.argv.insert(1, 'infer.hand_lrm')
    if not any(a.startswith('model_name=') for a in sys.argv[1:]):
        sys.argv.insert(2, 'model_name=LHM-1B')

    parser = argparse.ArgumentParser(description="OpenLRM launcher")
    parser.add_argument("runner", type=str, nargs='?', default='infer.hand_lrm', help="Runner to launch (default: infer.hand_lrm)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Optional checkpoint directory to save/load checkpoints (overrides env)")
    parser.add_argument("--checkpoint-file", type=str, default='./checkpoint/iteration_6000.ckpt',
                        help="Optional specific checkpoint file to load (overrides checkpoint-path)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Optional output/test directory to write results (overrides env)")
    args, unknown = parser.parse_known_args()

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError("Runner {} not found".format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]

    # Initialize CLI placeholders so later logic can safely check them
    cli_checkpoint = None
    cli_checkpoint_file = None
    cli_output = None

    # Collect tokens to inspect: prefer argparse `unknown`, otherwise fall back to raw sys.argv.
    # Support forms:
    #  - key=value
    #  - key value
    #  - single token with escaped space (e.g. "checkpoint-file\ ./path")
    raw_tokens = unknown if unknown else sys.argv[1:]
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        if tok.startswith('--') or tok.startswith('-'):
            i += 1
            continue

        if '=' in tok and not tok.startswith('-'):
            k, value = tok.split('=', 1)
            k = k.strip()
            value = value.strip()
            i += 1
        elif ' ' in tok and not tok.startswith('-'):
            k, value = tok.split(' ', 1)
            k = k.strip()
            value = value.strip()
            i += 1
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

        if k in ('checkpoint-file', 'checkpoint_file') and cli_checkpoint_file is None:
            cli_checkpoint_file = value
        elif k in ('checkpoint-path', 'checkpoint_path') and cli_checkpoint is None:
            cli_checkpoint = value
        elif k in ('output-path', 'output_path') and cli_output is None:
            cli_output = value

    # Prepare CLI values (compat with absl flags). Only fill from argparse when not set.
    if cli_checkpoint is None:
        cli_checkpoint = args.checkpoint_path
    if cli_checkpoint_file is None:
        cli_checkpoint_file = args.checkpoint_file
    if cli_output is None:
        cli_output = args.output_path
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
    # For single-process execution, use gloo backend with dynamic port
    if not dist.is_available():
        print("[Warning] torch.distributed not available, skipping init_process_group")
    else:
        try:
            port = random.randint(30000, 40000)
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(port)
            dist.init_process_group(backend='gloo', init_method='env://',
                                    rank=0, world_size=1, timeout=torch.distributed.timedelta(minutes=30))
            print(f"[✓] Distributed process group initialized on port {port}")
        except Exception as e:
            print(f"[Warning] Failed to init_process_group: {e}, continuing without distributed")

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)

    # ========================================= Setup Dataset =========================================
    hanco_kwargs = {}
    input_img = getattr(FLAGS, 'input_img', None)
    if input_img is not None:
        # Auto-parse vid/cam/imgid from path like: rgb/0185/cam6/00000010.jpg
        # or ./data/hanco/rgb/0185/cam6/00000010.jpg
        m = re.search(r'(\d{4})/cam(\d+)/(\d+)\.jpg', input_img)
        if m:
            hanco_kwargs['test_vid'] = int(m.group(1))
            hanco_kwargs['test_cam'] = int(m.group(2))
            hanco_kwargs['img_id'] = int(m.group(3))
        else:
            raise ValueError(f"Cannot parse vid/cam/imgid from --input_img='{input_img}'. "
                             f"Expected pattern like 'rgb/0185/cam6/00000010.jpg'")
    if getattr(FLAGS, 'test_cams', None) is not None:
        hanco_kwargs['test_cams'] = [int(c) for c in FLAGS.test_cams]
    if hanco_kwargs:
        print(f"\033[94m[HanCo CLI overrides] {hanco_kwargs}\033[0m")
    test_hand = HanCo(dataset_path='./data/hanco', split='test', **hanco_kwargs)
    test_dataloader = make_dataloader(test_hand, shuffle=False, batch_size=1)

    # Determine base output root: prefer explicit --output-path, fallback to FLAGS.output_dir
    base_output_root = cli_output or FLAGS.output_dir
    image_checkpoint_dir = os.path.join(base_output_root, 'finetune_hanco')
    os.makedirs(image_checkpoint_dir, exist_ok=True)

    # Set test_path for saving test images
    test_output_dir = os.path.join(image_checkpoint_dir, 'test_images')
    os.makedirs(test_output_dir, exist_ok=True)

    # Update runner's checkpoint path and test output path
    if hasattr(runner, 'checkpoint_path'):
        runner.checkpoint_path = image_checkpoint_dir
    else:
        setattr(runner, 'checkpoint_path', image_checkpoint_dir)

    if hasattr(runner, 'test_path'):
        runner.test_path = test_output_dir
    else:
        setattr(runner, 'test_path', test_output_dir)

    # Print configuration summary
    print(f"\033[94m{'='*80}\033[0m")
    print(f"\033[94mHanCo Two-Stage Fine-tuning Configuration:\033[0m")
    print(f"  Base output root: {base_output_root}")
    print(f"  Checkpoint dir: {image_checkpoint_dir}")
    print(f"  Test output dir: {test_output_dir}")
    print(f"\033[94m{'='*80}\033[0m\n")

    if FLAGS.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    torch.autograd.set_detect_anomaly(False)
    writer = SummaryWriter(log_dir=os.path.join(image_checkpoint_dir, 'logs'))

    # Determine which iteration to load from checkpoint
    test_iters = 7000  # default
    if cli_checkpoint_file:
        base = os.path.basename(cli_checkpoint_file)
        m = re.search(r'iteration[_-]?(\d+)', base)
        if m:
            test_iters = int(m.group(1))
        else:
            m2 = re.search(r'(\d{3,7})', base)
            if m2:
                test_iters = int(m2.group(1))

    # Load pretrained checkpoint
    try:
        if hasattr(runner, 'load_checkpoint'):
            runner.finetune_model(iteration=test_iters, is_latest=False, 
                                  checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file)
        else:
            runner.load(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint)
    except TypeError:
        runner.load(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint)

    print(f"\033[91m[========== Running Finetune ! Loading model for iteration {test_iters} ==========]\033[0m")

    # Reset checkpoint_path AFTER loading pretrained model
    runner.checkpoint_path = image_checkpoint_dir
    print(f"\033[94m[Debug] Reset runner.checkpoint_path to: {runner.checkpoint_path}\033[0m")
    if hasattr(runner, 'hand_model') and hasattr(runner.hand_model, 'checkpoint_path'):
        runner.hand_model.checkpoint_path = image_checkpoint_dir
        print(f"\033[94m[Debug] Reset runner.hand_model.checkpoint_path to: {runner.hand_model.checkpoint_path}\033[0m")

    # ====================== Eval-only branch ======================
    if FLAGS.only_eval:
        eval_iter = FLAGS.iter
        if isinstance(eval_iter, set):
            eval_iter = next(iter(eval_iter))
        print(f"\033[91m[========== Running EVAL ONLY @ iter {eval_iter} ==========]\033[0m")
        runner.load_checkpoint(iteration=eval_iter, is_latest=False, checkpoint_path=image_checkpoint_dir)

        infer_batch = test_hand.get_img()
        gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)
        _run_hanco_test(runner, test_dataloader, gs_model_list, gs_densify_list, query_points, eval_iter)
        return

    # ====================== Finetune + test branch ======================
    total_iters = FLAGS.iter
    # Two-stage inversion: stage1 (single-view) + stage2 (pseudo-view color consistency)
    use_two_stage_inversion = getattr(FLAGS, 'use_two_stage_inversion', False)
    inv_stage1_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage1', 200) or 200))
    inv_stage2_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage2', 0) or 0)) if use_two_stage_inversion else 0
    inv_iters = inv_stage1_iters + inv_stage2_iters  # Total inversion iterations
    color_consistency_weight = float(getattr(FLAGS, 'color_consistency_weight', 1.0) or 1.0)

    finetune_iters = max(0, total_iters - inv_iters)
    pseudo_batch = None
    batches = None

    data_iterator = iter(test_dataloader)

    print(f"\033[94m{'='*60}\033[0m")
    print(f"\033[94mInversion Training Plan:\033[0m")
    print(f"  Stage 1 (single-view): {inv_stage1_iters} iters")
    print(f"  Stage 2 (color consistency): {inv_stage2_iters} iters")
    print(f"  Color consistency weight: {color_consistency_weight}")
    print(f"  Finetune stage: {finetune_iters} iters")
    print(f"\033[94m{'='*60}\033[0m\n")

    # ------------------------- Inversion Stage 1: Single-view optimization -------------------------
    if inv_stage1_iters > 0:
        inv_pbar = tqdm(range(1, inv_stage1_iters + 1), desc='Inversion Stage1', dynamic_ncols=True)
        for step in inv_pbar:
            # try:
            #     batches = next(data_iterator)
            # except Exception:
            #     data_iterator = iter(test_dataloader)
            #     batches = next(data_iterator)

            infer_batch = [test_hand.get_img()]
            runner.run_wild_ohta(
                batch=infer_batch,
                # batch=batches,
                scaler=scaler,
                iteration=step,
                writer=writer,
                pbar=inv_pbar,
                total_iters=inv_stage1_iters,
            )

        # Save stage1 checkpoint
        runner.save(iteration=inv_stage1_iters, is_latest=False)
        print(f"\033[92m[✓] Completed Inversion Stage 1 ({inv_stage1_iters} iters)\033[0m")

    # ------------------------- Generate pseudo-GT views for Stage 2 -------------------------
    if inv_stage2_iters > 0:
        # Load color_shift/scale for pseudo-GT rendering
        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_stage1_iters,
                is_latest=False,
                checkpoint_path=image_checkpoint_dir,
            )

        # Ensure we have a batch for pseudo-GT generation
        if batches is None:
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)

        # Generate pseudo-GT views
        if hasattr(runner, 'build_pseudo_gt_batch'):
            save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
            save_prefix = "hanco_inv_stage1"
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=save_dir,
                save_prefix=save_prefix,                
                use_canonical_root=True,  # Use canonical pose for pseudo-GT generation            
                )
            print(f"\033[92m[✓] Generated {len(pseudo_batch)} pseudo-GT batches for Stage 2\033[0m")
        else:
            pseudo_batch = batches
            print(f"\033[93m[Warning] build_pseudo_gt_batch not available, using original batch\033[0m")

        # Inversion Stage 2: Pseudo-view color consistency
        inv_stage2_pbar = tqdm(range(1, inv_stage2_iters + 1), desc='Inversion Stage2', dynamic_ncols=True)
        for step in inv_stage2_pbar:
            runner.run_wild_inversion_stage2(
                batch=batches,
                pseudo_batch=pseudo_batch,
                scaler=scaler,
                iteration=step,
                writer=writer,
                pbar=inv_stage2_pbar,
                total_iters=inv_stage2_iters,
                color_consistency_weight=color_consistency_weight,
            )

        # Save stage2 checkpoint
        runner.save(iteration=inv_iters, is_latest=False)
        print(f"\033[92m[✓] Completed Inversion Stage 2 ({inv_stage2_iters} iters)\033[0m")

        # Reload with updated color params for subsequent stages
        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_iters,
                is_latest=False,
                checkpoint_path=image_checkpoint_dir,
            )

        # Regenerate pseudo-GT with updated color parameters
        if hasattr(runner, 'build_pseudo_gt_batch'):
            save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
            save_prefix = "hanco_inv_final"
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=save_dir,
                save_prefix=save_prefix,                
                use_canonical_root=False,  # Use reference pose for Stage 2 rendering            
                )

        # Render Inversion Stage 2 Final Model to Pseudo-GT Poses
        if pseudo_batch is not None:
            print(f"\033[94m[Rendering Inversion Stage 2 model to pseudo-GT poses]\033[0m")
            runner.hand_model.eval()
            if hasattr(runner, '_sync_color_inversion_params'):
                runner._sync_color_inversion_params()

            infer_batch = test_hand.get_img()
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

            final_pseudo_batch = pseudo_batch[0]
            inv_stage2_pseudo_dir = os.path.join(test_output_dir, 'debug_vis', 'inv_stage2_pseudo_views')
            os.makedirs(inv_stage2_pseudo_dir, exist_ok=True)

            n_views = final_pseudo_batch['original_image'].shape[0]
            for view_idx in range(n_views):
                with torch.no_grad():
                    cam = runner.hand_model.renderer.get_single_view_cam(final_pseudo_batch, view_idx)
                    smplx_params = final_pseudo_batch['smpl_param']
                    smplx_data = runner.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx)
                    smplx_data = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in smplx_data.items()}

                    render_res = runner.hand_model.renderer.forward_animate_gs(
                        gs_model_list[0],
                        gs_densify_list[0],
                        query_points[0],
                        cam,
                        smplx_data,
                        256, 256,
                        runner.bg_color,
                    )

                    if 'comp_rgb' in render_res:
                        comp_rgb = render_res['comp_rgb']
                        if comp_rgb.dim() == 5:
                            comp_rgb = comp_rgb[0, 0]
                        elif comp_rgb.dim() == 4:
                            comp_rgb = comp_rgb[0]
                        if comp_rgb.dim() == 3:
                            if comp_rgb.shape[-1] != 3 and comp_rgb.shape[0] == 3:
                                comp_rgb = comp_rgb.permute(1, 2, 0)
                            comp_rgb_np = (comp_rgb.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                            comp_rgb_np = cv2.cvtColor(comp_rgb_np, cv2.COLOR_RGB2BGR)
                            out_path = os.path.join(inv_stage2_pseudo_dir, f'inv_stage2_view_{view_idx:02d}.jpg')
                            cv2.imwrite(out_path, comp_rgb_np)

            print(f"\033[92m[✓] Rendered Inv Stage 2 model to {n_views} pseudo-GT poses → {inv_stage2_pseudo_dir}\033[0m")
            runner.hand_model.train()

    elif inv_stage1_iters > 0:
        # Only stage1, still need to generate pseudo batch for finetune
        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_stage1_iters,
                is_latest=False,
                checkpoint_path=image_checkpoint_dir,
            )

        if batches is None:
            # try:
            #     batches = next(data_iterator)
            # except Exception:
            #     data_iterator = iter(test_dataloader)
            #     batches = next(data_iterator)
            batches = [test_hand.get_img()]

        if hasattr(runner, 'build_pseudo_gt_batch'):
            save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
            save_prefix = f"hanco_inv{inv_stage1_iters}"
            batches = [test_hand.get_img()]
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=save_dir,
                save_prefix=save_prefix,
            )

            # Render Inversion Stage 1 Final Model to Pseudo-GT Poses
            print(f"\033[94m[Rendering Inversion Stage 1 model to pseudo-GT poses]\033[0m")
            runner.hand_model.eval()
            if hasattr(runner, '_sync_color_inversion_params'):
                runner._sync_color_inversion_params()

            infer_batch = test_hand.get_img()
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

            final_pseudo_batch = pseudo_batch[0]
            inv_stage1_pseudo_dir = os.path.join(test_output_dir, 'debug_vis', 'inv_stage1_pseudo_views')
            os.makedirs(inv_stage1_pseudo_dir, exist_ok=True)

            n_views = final_pseudo_batch['original_image'].shape[0]
            for view_idx in range(n_views):
                with torch.no_grad():
                    cam = runner.hand_model.renderer.get_single_view_cam(final_pseudo_batch, view_idx)
                    smplx_params = final_pseudo_batch['smpl_param']
                    smplx_data = runner.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx)
                    smplx_data = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in smplx_data.items()}

                    render_res = runner.hand_model.renderer.forward_animate_gs(
                        gs_model_list[0],
                        gs_densify_list[0],
                        query_points[0],
                        cam,
                        smplx_data,
                        256, 256,
                        runner.bg_color,
                    )

                    if 'comp_rgb' in render_res:
                        comp_rgb = render_res['comp_rgb']
                        if comp_rgb.dim() == 5:
                            comp_rgb = comp_rgb[0, 0]
                        elif comp_rgb.dim() == 4:
                            comp_rgb = comp_rgb[0]
                        if comp_rgb.dim() == 3:
                            if comp_rgb.shape[-1] != 3 and comp_rgb.shape[0] == 3:
                                comp_rgb = comp_rgb.permute(1, 2, 0)
                            comp_rgb_np = (comp_rgb.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                            comp_rgb_np = cv2.cvtColor(comp_rgb_np, cv2.COLOR_RGB2BGR)
                            out_path = os.path.join(inv_stage1_pseudo_dir, f'inv_stage1_view_{view_idx:02d}.jpg')
                            cv2.imwrite(out_path, comp_rgb_np)

            print(f"\033[92m[✓] Rendered Inv Stage 1 model to {n_views} pseudo-GT poses → {inv_stage1_pseudo_dir}\033[0m")
            runner.hand_model.train()
        else:
            pseudo_batch = batches

    # --------------------- Stage 2: pseudo-gt finetune ---------------------
    if finetune_iters > 0:
        if pseudo_batch is None:
            # try:
            #     batches = next(data_iterator)
            # except Exception:
            #     data_iterator = iter(test_dataloader)
            #     batches = next(data_iterator)
            batches = [test_hand.get_img()]
            if hasattr(runner, 'build_pseudo_gt_batch'):
                save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
                save_prefix = f"hanco_inv{inv_iters}"
                pseudo_batch = runner.build_pseudo_gt_batch(
                    batches,
                    n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                    save_dir=save_dir,
                    save_prefix=save_prefix,
                )
            else:
                pseudo_batch = batches

        finetune_pbar = tqdm(range(1, finetune_iters + 1), desc='Finetuning', dynamic_ncols=True)
        use_pose_refine = getattr(FLAGS, 'enable_pose_refine', False)
        if use_pose_refine:
            print(f"\033[94m[Pose Refinement ENABLED] lr={FLAGS.pose_refine_lr}\033[0m")
        for step in finetune_pbar:
            total_step = inv_iters + step
            runner.run_wild_stage_2(
                batch=pseudo_batch,
                scaler=scaler,
                iteration=step - 1,
                writer=writer,
                pbar=finetune_pbar,
                total_iters=finetune_iters,
                enable_pose_refine=use_pose_refine,
                pose_refine_lr=getattr(FLAGS, 'pose_refine_lr', 1e-4),
            )

            # Save checkpoint and optionally run testing
            is_last = (total_step == total_iters)
            do_test = is_last # or (total_step % 500 == 0)

            if do_test:
                iteration = total_step
                runner.save(iteration=iteration, is_latest=False)
                print(f"\033[92m[✓] Saved checkpoint at iteration {iteration}\033[0m")

                # Switch to eval mode for inference
                runner.hand_model.eval()
                if hasattr(runner, '_sync_color_inversion_params'):
                    runner._sync_color_inversion_params()

                print(f"\033[91m[========== Running test @ iteration {iteration} ==========]\033[0m")

                infer_batch = test_hand.get_img()
                gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)
                _run_hanco_test(runner, test_dataloader, gs_model_list, gs_densify_list, query_points, iteration)

                # Restore train mode
                runner.hand_model.train()

        # Final Evaluation: Render to pseudo-GT poses
        if pseudo_batch is not None:
            print(f"\033[94m[Rendering final model to pseudo-GT poses]\033[0m")
            runner.hand_model.eval()
            if hasattr(runner, '_sync_color_inversion_params'):
                runner._sync_color_inversion_params()

            infer_batch = test_hand.get_img()
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

            final_pseudo_batch = pseudo_batch[0]
            final_pseudo_dir = os.path.join(test_output_dir, 'debug_vis', 'final_pseudo_views')
            os.makedirs(final_pseudo_dir, exist_ok=True)

            n_views = final_pseudo_batch['original_image'].shape[0]
            for view_idx in range(n_views):
                with torch.no_grad():
                    cam = runner.hand_model.renderer.get_single_view_cam(final_pseudo_batch, view_idx)
                    smplx_params = final_pseudo_batch['smpl_param']
                    smplx_data = runner.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx)
                    smplx_data = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in smplx_data.items()}

                    render_res = runner.hand_model.renderer.forward_animate_gs(
                        gs_model_list[0],
                        gs_densify_list[0],
                        query_points[0],
                        cam,
                        smplx_data,
                        256, 256,
                        runner.bg_color,
                    )

                    if 'comp_rgb' in render_res:
                        comp_rgb = render_res['comp_rgb']
                        if comp_rgb.dim() == 5:
                            comp_rgb = comp_rgb[0, 0]
                        elif comp_rgb.dim() == 4:
                            comp_rgb = comp_rgb[0]
                        if comp_rgb.dim() == 3:
                            if comp_rgb.shape[-1] != 3 and comp_rgb.shape[0] == 3:
                                comp_rgb = comp_rgb.permute(1, 2, 0)
                            comp_rgb_np = (comp_rgb.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                            comp_rgb_np = cv2.cvtColor(comp_rgb_np, cv2.COLOR_RGB2BGR)
                            out_path = os.path.join(final_pseudo_dir, f'pseudo_gt_view_{view_idx:02d}.jpg')
                            cv2.imwrite(out_path, comp_rgb_np)

            print(f"\033[92m[✓] Rendered final model to {n_views} pseudo-GT poses → {final_pseudo_dir}\033[0m")
            runner.hand_model.train()

    writer.close()
    print(f"\033[92m[✓] Training complete!\033[0m")


if __name__ == "__main__":
    app.run(main)
