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
flags.DEFINE_string('checkpoint-file', '../checkpoint/interhand-correct/iteration_6000.ckpt', 'Specific checkpoint file to load (overrides checkpoint-path)')
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
flags.DEFINE_integer('iter-inversion', 200, 'Inversion iterations before pseudo-GT finetune')
flags.DEFINE_boolean('use_two_stage_inversion', False, 'Whether to use two-stage inversion (stage1 + stage2 with color consistency)')
flags.DEFINE_integer('iter_inversion_stage1', 100, 'Stage1 inversion iters (single-view color optimization)')
flags.DEFINE_integer('iter_inversion_stage2', 0, 'Stage2 inversion iters (pseudo-view color consistency)')
flags.DEFINE_float('color-consistency-weight', 1.0, 'Weight for color consistency loss in inversion stage2')
flags.DEFINE_integer('pseudo-views', 8, 'Number of pseudo-GT views')
flags.DEFINE_boolean('animate_to_handavatar', True, 'Whether to animate inferred GS to HandAvatar test poses')
flags.DEFINE_boolean('save_test_gt_images', False, 'Whether to save transformed GT images during testing')
flags.DEFINE_boolean('enable_pose_refine', False, 'Enable pose refinement during stage2 finetune')
flags.DEFINE_float('pose_refine_lr', 1e-4, 'Learning rate for pose refiner')
flags.DEFINE_boolean('no_pretrain', False, 'Ablation: skip pretrained checkpoint loading (train from scratch)')
flags.DEFINE_integer('edit-unmask-iter', 300, 'Stage2 iterations before unmasking edit region')
flags.DEFINE_float('edit-mask-weight', 30.0, 'Loss weight multiplier on edit region in unmasked phase')
flags.DEFINE_float('pseudo-view-weight', 0.1, 'Weight for pseudo-view supervision in unmasked (edit) phase')
flags.DEFINE_string('edit', 'True', 'Enable edit mode in dataset (auto-detected from input path)')


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
    parser.add_argument("--checkpoint-file", type=str, default='./checkpoint/interhand-correct/iteration_6000.ckpt',
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
    # When --no_pretrain is set, pass resume=False so runner.__init__ does NOT
    # load the checkpoint — keeping custom modules (transformer, adapter, etc.)
    # at their random-init state while DINOv2 encoder retains HuggingFace weights.
    _resume = not FLAGS.no_pretrain
    try:
        runner = RunnerClass(checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file, output_path=cli_output, resume=_resume)
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
    # dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500',#'env://',
    #                         rank=0,       # Set appropriate rank per process
    #                         world_size=1,
    #                         # master_addr="127.0.0.1",
    #                         # master_port=29500
    #                         )   # Total number of processes)
    # rank = dist.get_rank()
    # torch.cuda.set_device(rank % torch.cuda.device_count())
    # print(f"Start running basic DDP example on rank {rank}.")
    # device_id = rank % torch.cuda.device_count()

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
    ablation_suffix = '_no_pretrain' if FLAGS.no_pretrain else ''
    image_checkpoint_dir = os.path.join(base_output_root, f"finetune_{image_name}{ablation_suffix}")
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
    runner.save_test_gt_images = bool(getattr(FLAGS, 'save_test_gt_images', False))

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
        is_edit_mode = 'editing' in (cli_input_dir or '')
        if is_edit_mode:
            test_hand = HandDataset(split='test_wild', img_path=img_glob[0], edit=FLAGS.edit)
            print(f"\033[93m[Edit Mode] Detected 'editing' in input path, using edit-aware pipeline\033[0m")
        else:
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

    
    # Propagate no_pretrain flag to runner so finetune methods can unfreeze all modules
    runner.no_pretrain = FLAGS.no_pretrain

    if FLAGS.no_pretrain:
        # Ablation: skip pretrained checkpoint, use model initial weights
        print(f"\033[93m[========== ABLATION: No pretrained checkpoint (training from scratch) ==========]\033[0m")
        # Zero-init the SHS (color) output layer in the GS decoder.
        # By default, use_rgb=True causes this layer to keep random Kaiming init
        # (all other GS layers are zero-init'd). With random transformer weights,
        # the 1024-dim dot product produces large logits → sigmoid → ~1.0 → white.
        # Zero-init gives sigmoid(0) = 0.5 → neutral gray at start.
        gs_net = runner.hand_model.renderer.gs_net
        for layer_dict_name in ('out_layers', 'densify_out_layers'):
            layer_dict = getattr(gs_net, layer_dict_name, None)
            if layer_dict is not None and 'shs' in layer_dict:
                shs_layer = layer_dict['shs']
                if hasattr(shs_layer, 'weight'):
                    torch.nn.init.constant_(shs_layer.weight, 0)
                    torch.nn.init.constant_(shs_layer.bias, 0)
                    print(f"\033[93m[no_pretrain] Zero-init'd {layer_dict_name}['shs'] to avoid white render\033[0m")
    else:
        try:
            if hasattr(runner, 'load_checkpoint'):
                runner.finetune_model(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file)
            else:
                runner.load(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint)
        except TypeError:
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
            handavatar_path = cli_handavatar if cli_handavatar is not None else './data/interhand'
            print(f"\033[94mLoading HandAvatar dataset from: {handavatar_path}\033[0m")
            handavatar_ds = HandAvatarDataset(dataset_path=handavatar_path, data_type='progress', skip=100)
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
        infer_batch = test_hand.get_img()
        # infer_batch = test_hand.get_img_rainbow()
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
                    test_batch[0]['dataset_id'] = 2
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

    total_iters = FLAGS.iter
    # Two-stage inversion: stage1 (single-view) + stage2 (pseudo-view color consistency)
    use_two_stage_inversion = getattr(FLAGS, 'use_two_stage_inversion', True)
    inv_stage1_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage1', 200) or 200))
    inv_stage1_iters = int(getattr(FLAGS, 'iter_inversion_stage1', 200))
    inv_stage2_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage2', 200) or 200)) if use_two_stage_inversion else 0
    inv_iters = inv_stage1_iters + inv_stage2_iters  # Total inversion iterations
    color_consistency_weight = float(getattr(FLAGS, 'color_consistency_weight', 1.0) or 1.0)
    
    # Use visible-only strategy (no pseudo-GT, gradient masking on invisible regions)
    _vis_only_raw = getattr(FLAGS, 'use_visible_only', False)  # Default to False (use pseudo-GT)
    if isinstance(_vis_only_raw, str):
        use_visible_only_strategy = _vis_only_raw.lower() not in ('false', '0', 'no')
    else:
        use_visible_only_strategy = bool(_vis_only_raw)
    pretrain_reg_weight = float(getattr(FLAGS, 'pretrain_reg', 0.1) or 0.1)
    
    finetune_iters = max(0, total_iters - inv_iters)
    pseudo_batch = None
    batches = None

    # Edit mode parameters
    edit_unmask_iter = max(0, int(getattr(FLAGS, 'edit_unmask_iter', 300)))
    edit_mask_weight = float(getattr(FLAGS, 'edit_mask_weight', 100.0))
    pseudo_view_weight = float(getattr(FLAGS, 'pseudo_view_weight', 0.1))

    data_iterator = iter(test_dataloader)

    print(f"\033[94m{'='*60}\033[0m")
    print(f"\033[94mInversion Training Plan:\033[0m")
    print(f"  Edit mode: {is_edit_mode}")
    print(f"  Stage 1 (single-view): {inv_stage1_iters} iters")
    print(f"  Stage 2 (color consistency): {inv_stage2_iters} iters")
    print(f"  Color consistency weight: {color_consistency_weight}")
    print(f"  Finetune stage: {finetune_iters} iters")
    print(f"  \033[93mStrategy: {'VISIBLE-ONLY (no pseudo-GT)' if use_visible_only_strategy else 'PSEUDO-GT'}\033[0m")
    if use_visible_only_strategy:
        print(f"  Pretrain regularization: {pretrain_reg_weight}")
    if is_edit_mode:
        print(f"  Edit unmask iter: {edit_unmask_iter}")
        print(f"  Edit mask weight: {edit_mask_weight}")
        print(f"  Pseudo-view weight (unmasked): {pseudo_view_weight}")
    print(f"\033[94m{'='*60}\033[0m\n")

    # ------------------------- Inversion Stage 1: Single-view optimization -------------------------
    if inv_stage1_iters > 0:
        inv_pbar = tqdm(range(1, inv_stage1_iters + 1), desc='Edit Inversion' if is_edit_mode else 'Inversion Stage1', dynamic_ncols=True)
        for step in inv_pbar:
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)

            if 'text-to-avatar' in cli_input_dir:
                batches[0]['dataset_id'] = 0
            else:
                batches[0]['dataset_id'] = 2

            if is_edit_mode:
                runner.run_edit_wild_inversion(
                    batch=batches,
                    scaler=scaler,
                    iteration=step,
                    writer=writer,
                    pbar=inv_pbar,
                    total_iters=inv_stage1_iters,
                )
            else:
                runner.run_wild_ohta(
                    batch=batches,
                    scaler=scaler,
                    iteration=step,
                    writer=writer,
                    pbar=inv_pbar,
                    total_iters=inv_stage1_iters,
                )

        # Save stage1 checkpoint
        runner.save(iteration=inv_stage1_iters, is_latest=False)
        print(f"\033[92m[✓] Completed Inversion Stage 1 ({inv_stage1_iters} iters)\033[0m")

        # Ablation: when finetune stage is disabled, directly evaluate the
        # stage1 inversion rendering on HandAvatar deformation poses.
        if finetune_iters == 0:
            eval_iter = inv_stage1_iters
            print(f"\033[91m[========== Running Stage1 Inversion Eval @ iter {eval_iter} ==========]\033[0m")

            runner.hand_model.eval()
            runner._sync_color_inversion_params()

            infer_batch = test_hand.get_img()
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

            psnr_full = 0
            ssim_full = 0
            lpips_full = 0
            image_full = 0
            count = 0

            if handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
                with tqdm(
                    total=len(handavatar_dataloader),
                    desc=f"Stage1 Eval {eval_iter} | HandAvatar",
                    dynamic_ncols=True,
                ) as pbar:
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
                            loss_dict = runner.test_handavatar(
                                gs_model_list,
                                gs_densify_list,
                                query_points,
                                map_batch,
                                map_idx,
                                eval_iter,
                                pbar,
                            )

                        if isinstance(loss_dict, dict) and 'psnr' in loss_dict:
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
            else:
                print("\033[93m[Warning] HandAvatar dataloader unavailable, fallback to wild test dataloader.\033[0m")
                for idx, test_batch in enumerate(test_dataloader):
                    with tqdm(
                        total=1,
                        desc=f"Stage1 Eval {eval_iter} | Sample {idx+1}/{len(test_dataloader)}",
                        leave=True,
                        position=idx + 1,
                        dynamic_ncols=True,
                    ) as pbar:
                        with torch.no_grad():
                            test_batch[0]['dataset_id'] = 2
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

            if count > 0:
                print(
                    f"\033[91m[Stage1 Summary @iter {eval_iter}]\033[0m "
                    f"PSNR: {psnr_full/count:.4f}, "
                    f"SSIM: {ssim_full/count:.4f}, "
                    f"LPIPS: {lpips_full/count:.4f}\n"
                )

            metrics = {
                'image_name': image_name,
                'lpips': lpips_full / count if count > 0 else 0.0,
                'ssim': ssim_full / count if count > 0 else 0.0,
                'psnr': psnr_full / count if count > 0 else 0.0,
                'image_l1': image_full / count if count > 0 else 0.0,
                'iteration': int(eval_iter),
                'timestamp': time.time(),
                'eval_scope': 'handavatar_stage1_inversion' if handavatar_dataloader is not None else 'wild_fallback_stage1_inversion',
            }

            metrics_path = os.path.join(image_checkpoint_dir, f'metrics_stage1_iter{int(eval_iter)}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\033[92m[✓] Saved stage1 eval metrics to {metrics_path}\033[0m")
            return

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

        # Generate pseudo-GT views for Stage 2 training (use canonical root)
        if hasattr(runner, 'build_pseudo_gt_batch'):
            save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
            save_prefix = f"{image_name}_inv_stage1"
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=save_dir,
                save_prefix=save_prefix,
                use_canonical_root=True,
            )
            print(f"\033[92m[✓] Generated {len(pseudo_batch)} pseudo-GT batches for Stage 2\033[0m")
        else:
            pseudo_batch = batches
            print(f"\033[93m[Warning] build_pseudo_gt_batch not available, using original batch\033[0m")

        # ------------------------- Inversion Stage 2: Pseudo-view color consistency -------------------------
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

        # Regenerate pseudo-GT for visualization (use reference root pose)
        if hasattr(runner, 'build_pseudo_gt_batch'):
            save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
            save_prefix = f"{image_name}_inv_final"
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=save_dir,
                save_prefix=save_prefix,
                use_canonical_root=False,
            )

    elif inv_stage1_iters > 0:
        # Only stage1, still need to generate pseudo batch for finetune
        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_stage1_iters,
                is_latest=False,
                checkpoint_path=image_checkpoint_dir,
            )

        if hasattr(runner, 'build_pseudo_gt_batch'):
            save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
            save_prefix = f"{image_name}_inv_final"
            try:
                pseudo_batch = runner.build_pseudo_gt_batch(
                    batches,
                    n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                    save_dir=save_dir,
                    save_prefix=save_prefix,
                    use_canonical_root=False,
                )
            except Exception as e:
                print(f"\033[93m[Warning] Failed to generate pseudo batch: {e}\033[0m")
                pseudo_batch = batches
        else:
            pseudo_batch = batches

    # --------------------- Stage 3: Finetune with pseudo-GT ---------------------
    if finetune_iters > 0:
        if pseudo_batch is None:
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)

            # Set dataset_id so MANO uses the correct center_add behavior
            if 'text-to-avatar' in cli_input_dir:
                batches[0]['dataset_id'] = 0
            else:
                batches[0]['dataset_id'] = 2

            if hasattr(runner, 'build_pseudo_gt_batch'):
                save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
                save_prefix = f"{image_name}_inv{inv_iters}"
                pseudo_batch = runner.build_pseudo_gt_batch(
                    batches,
                    n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                    save_dir=save_dir,
                    save_prefix=save_prefix,
                    use_canonical_root=True,
                )
            else:
                pseudo_batch = batches

        finetune_pbar = tqdm(range(1, finetune_iters + 1), desc='Finetuning', dynamic_ncols=True)
        for step in finetune_pbar:
            total_step = inv_iters + step
            
            if use_visible_only_strategy:
                # NEW STRATEGY: Only learn from input view, no pseudo-GT
                # Use original batches (not pseudo_batch) to avoid pseudo-GT pollution
                if batches is None:
                    try:
                        batches = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(test_dataloader)
                        batches = next(data_iterator)
                
                runner.run_visible_only(
                    batch=batches,
                    scaler=scaler,
                    iteration=step - 1,
                    writer=writer,
                    pbar=finetune_pbar,
                    total_iters=finetune_iters,
                    gradient_mask_invisible=True,
                    pretrain_regularization=pretrain_reg_weight,
                )
            else:
                # OLD STRATEGY: Use pseudo-GT views (has invisible region issues)
                runner.run_wild_stage_2(
                    batch=pseudo_batch,
                    scaler=scaler,
                    iteration=step - 1,
                    writer=writer,
                    pbar=finetune_pbar,
                    total_iters=finetune_iters,
                    enable_pose_refine=getattr(FLAGS, 'enable_pose_refine', False),
                    pose_refine_lr=getattr(FLAGS, 'pose_refine_lr', 1e-4),
                    edit_mask_weight=edit_mask_weight if is_edit_mode else 0.0,
                )

            # Save checkpoint and optionally run testing.
            # Testing uses the current in-memory model directly (no checkpoint
            # reload) so that optimizer state, learning rate, and stage flags
            # are preserved for continued training.
            is_last = (total_step == total_iters)
            do_test = is_last #or (total_step % 500 == 0)

            if do_test:
                iteration = total_step
                checkpoint_name = f"{image_name}_iter{iteration}"
                runner.save(iteration=iteration, is_latest=False)
                print(f"\033[92m[✓] Saved checkpoint: {checkpoint_name} at iteration {iteration}\033[0m")

                # Switch to eval mode for inference; do NOT reload checkpoint
                runner.hand_model.eval()
                runner._sync_color_inversion_params()

                print(f"\033[91m[========== Running test @ iteration {iteration} ==========]\033[0m")

                psnr_full = 0
                ssim_full = 0
                lpips_full = 0
                image_full = 0
                count = 0

                if 'text-to-avatar' in cli_input_dir:
                    infer_batch = test_hand.get_img_rainbow()
                else:
                    infer_batch = test_hand.get_img()
                gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

                # --- Animate inferred GS to HandAvatar test poses/cameras ---
                if handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
                    print(f"\033[94m[Animating GS to HandAvatar poses for iteration {iteration}]\033[0m")
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
                                                       map_batch, map_idx, iteration, pbar=None)
                    print(f"\033[92m[✓] Completed HandAvatar pose animation\033[0m")

                for idx, test_batch in enumerate(test_dataloader):
                    with tqdm(
                        total=1,
                        desc=f"Iter {iteration} | Sample {idx+1}/{len(test_dataloader)}",
                        leave=True,
                        position=idx + 1,
                        dynamic_ncols=True
                    ) as pbar:
                        with torch.no_grad():
                            # Set dataset_id for test batch
                            if 'text-to-avatar' in cli_input_dir:
                                test_batch[0]['dataset_id'] = 0
                            else:
                                test_batch[0]['dataset_id'] = 2
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

                        pbar.set_postfix({
                            "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
                            "ssim": f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
                            "psnr": f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
                        })
                        pbar.update(1)
                        torch.cuda.empty_cache()

                print(
                    f"\033[91m[Summary @iter {iteration}]\033[0m "
                    f"PSNR: {psnr_full/count:.4f}, "
                    f"SSIM: {ssim_full/count:.4f}, "
                    f"LPIPS: {lpips_full/count:.4f}\n"
                )

                # Switch back to train mode for next iteration
                runner.hand_model.train()

                current_metrics = {
                    'image_name': image_name,
                    'lpips': lpips_full / count,
                    'ssim': ssim_full / count,
                    'psnr': psnr_full / count,
                    'image_l1': image_full / count,
                    'iteration': int(iteration),
                    'timestamp': time.time(),
                }

                metrics_path = os.path.join(image_checkpoint_dir, f'metrics_iter{iteration}.json')
                with open(metrics_path, 'w') as f:
                    json.dump(current_metrics, f, indent=2)
                print(f"\033[92m[✓] Saved metrics to {metrics_path}\033[0m")




if __name__ == "__main__":
    # main()
    app.run(main)


