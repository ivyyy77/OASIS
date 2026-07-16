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
import sys

import time
import json

from tqdm import tqdm


from LHM.runners import REGISTRY_RUNNERS

from torch.nn import Tanh, Identity

from data.wild_hand_dataset import HandDataset
from data.interhand.train import Dataset, HandAvatarDataset
from data.wild_hand_dataset import make_dataloader
import torch.nn as nn
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP




from tqdm import tqdm
from data.hand_dataset import merge_batch
from LHM.losses import _is_better

from torch.utils.tensorboard import SummaryWriter




flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')

flags.DEFINE_string('checkpoint-path', None, 'Checkpoint directory to save/load checkpoints')
flags.DEFINE_string('output-path', './output/finetune', 'Output/test directory to write results')
flags.DEFINE_boolean('resume', True, 'Enable auto-resume from latest checkpoint')
flags.DEFINE_string('checkpoint-file', None, 'Specific checkpoint file to load (overrides checkpoint-path)')
flags.DEFINE_string('input-dir', None, 'Optional directory containing in-the-wild images to process (process all files)')
flags.DEFINE_string('handavatar-path', os.environ.get('HANDAVATAR_ROOT'), 'HandAvatar dataset root')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input')
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_boolean('use_amp', True, 'Use automatic mixed precision (AMP)')
flags.DEFINE_multi_string(
  'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')

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













    parser = argparse.ArgumentParser(description="OpenLRM launcher")
    parser.add_argument("--runner", default=DEFAULT_RUNNER, type=str, help="Runner to launch")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Optional checkpoint directory to save/load checkpoints (overrides env)")
    parser.add_argument("--checkpoint-file", type=str, default=None,
                        help="Optional specific checkpoint file to load (overrides checkpoint-path)")
    parser.add_argument("--output-path", type=str, default='./output/finetune',
                        help="Optional output/test directory to write results (overrides env)")
    parser.add_argument("--handavatar-path", type=str, default=os.environ.get("HANDAVATAR_ROOT"),
                        help="Optional HandAvatar dataset path for mapping")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Optional directory containing in-the-wild images to process (process all files)")
    parser.add_argument("--test-iter", type=int, default=None,
                        help="Iteration to load and test (overrides hardcoded value)")
    args, unknown = parser.parse_known_args()


    cli_checkpoint = None
    cli_checkpoint_file = None
    cli_output = None
    cli_handavatar = None
    cli_input_dir = None






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













    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)



    import glob
    img_glob = []
    if cli_input_dir is not None and os.path.exists(cli_input_dir):

        if os.path.isdir(cli_input_dir):
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                img_glob.extend(glob.glob(os.path.join(cli_input_dir, ext)))
            img_glob = sorted(img_glob)

            multi_image_mode = (len(img_glob) > 1)


        elif os.path.isfile(cli_input_dir):
            _, ext = os.path.splitext(cli_input_dir)
            if ext.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                img_glob = [cli_input_dir]
                multi_image_mode = False
    else:
        multi_image_mode = False



    if multi_image_mode or not img_glob:
        image_name = "default"
    else:
        image_name = os.path.splitext(os.path.basename(img_glob[0]))[0]


    base_output_root = cli_output or FLAGS.output_dir


    ablation_suffix = '_no_pretrain' if FLAGS.no_pretrain else ''
    image_checkpoint_dir = os.path.join(base_output_root, f"finetune_{image_name}{ablation_suffix}")
    os.makedirs(image_checkpoint_dir, exist_ok=True)


    if hasattr(runner, 'checkpoint_path'):
        runner.checkpoint_path = image_checkpoint_dir
    else:
        setattr(runner, 'checkpoint_path', image_checkpoint_dir)


    test_output_dir = os.path.join(image_checkpoint_dir, 'test_images')
    os.makedirs(test_output_dir, exist_ok=True)
    if hasattr(runner, 'test_path'):
        runner.test_path = test_output_dir
    else:
        setattr(runner, 'test_path', test_output_dir)
    runner.save_test_gt_images = bool(getattr(FLAGS, 'save_test_gt_images', False))


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



    if not multi_image_mode:
        if not img_glob:
            raise ValueError("No input image provided. Please specify --input-dir with a valid image path.")
        is_edit_mode = 'editing' in (cli_input_dir or '')
        if is_edit_mode:
            test_hand = HandDataset(split='test_wild', img_path=img_glob[0], edit=FLAGS.edit)
            print(f"\033[93m[Edit Mode] Detected 'editing' in input path, using edit-aware pipeline\033[0m")
        else:
            test_hand = HandDataset(split='test_wild', img_path=img_glob[0])

        test_dataloader = make_dataloader(test_hand, shuffle=False, batch_size=1)
















    import re
    if args.test_iter is not None:
        test_iters = {args.test_iter}
    elif cli_checkpoint_file is not None:

        base = os.path.basename(cli_checkpoint_file)
        m = re.search(r'iteration[_-]?(\d+)', base)
        if m:
            test_iters = int(m.group(1))
        else:

            m2 = re.search(r'(\d{3,7})', base)
            if m2:
                test_iters = int(m2.group(1))
            else:
                test_iters = 12000
    else:
        test_iters = 12000



    runner.no_pretrain = FLAGS.no_pretrain

    if FLAGS.no_pretrain:

        print(f"\033[93m[========== ABLATION: No pretrained checkpoint (training from scratch) ==========]\033[0m")





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



    runner.checkpoint_path = image_checkpoint_dir
    print(f"\033[94mCheckpoint output: {runner.checkpoint_path}\033[0m")
    if hasattr(runner, 'hand_model') and hasattr(runner.hand_model, 'checkpoint_path'):
        runner.hand_model.checkpoint_path = image_checkpoint_dir
        print(f"\033[94mModel checkpoint output: {runner.hand_model.checkpoint_path}\033[0m")



    handavatar_dataloader = None
    if FLAGS.animate_to_handavatar:
        try:
            if cli_handavatar is None:
                raise ValueError("--handavatar-path is required when animation is enabled")
            handavatar_path = cli_handavatar
            print(f"\033[94mLoading HandAvatar dataset from: {handavatar_path}\033[0m")
            handavatar_ds = HandAvatarDataset(dataset_path=handavatar_path, data_type='progress', skip=100)
            handavatar_dataloader = make_dataloader(handavatar_ds, shuffle=False, batch_size=1)
            print(f"\033[92m[✓] Loaded HandAvatar dataset with {len(handavatar_dataloader)} samples\033[0m")
        except Exception as e:
            print(f"\033[91m[Warning] Failed to load HandAvatar dataset: {e}\033[0m")
            handavatar_dataloader = None

    if FLAGS.only_eval:


        eval_iter = FLAGS.iter
        if isinstance(eval_iter, set):

            eval_iter = next(iter(eval_iter))
        print(f"\033[91m[========== Running EVAL ONLY @ iter {eval_iter} ==========]\033[0m")
        runner.load_checkpoint(iteration=eval_iter, is_latest=False, checkpoint_path=image_checkpoint_dir)

        psnr_full = 0
        ssim_full = 0
        lpips_full = 0
        image_full = 0
        count = 0


        infer_batch = test_hand.get_img()

        gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)


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




    total_iters = FLAGS.iter

    use_two_stage_inversion = getattr(FLAGS, 'use_two_stage_inversion', True)
    inv_stage1_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage1', 200) or 200))
    inv_stage1_iters = int(getattr(FLAGS, 'iter_inversion_stage1', 200))
    inv_stage2_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage2', 200) or 200)) if use_two_stage_inversion else 0
    inv_iters = inv_stage1_iters + inv_stage2_iters
    color_consistency_weight = float(getattr(FLAGS, 'color_consistency_weight', 1.0) or 1.0)


    _vis_only_raw = getattr(FLAGS, 'use_visible_only', False)
    if isinstance(_vis_only_raw, str):
        use_visible_only_strategy = _vis_only_raw.lower() not in ('false', '0', 'no')
    else:
        use_visible_only_strategy = bool(_vis_only_raw)
    pretrain_reg_weight = float(getattr(FLAGS, 'pretrain_reg', 0.1) or 0.1)

    finetune_iters = max(0, total_iters - inv_iters)
    pseudo_batch = None
    batches = None


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


        runner.save(iteration=inv_stage1_iters, is_latest=False)
        print(f"\033[92m[✓] Completed Inversion Stage 1 ({inv_stage1_iters} iters)\033[0m")



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


    if inv_stage2_iters > 0:

        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_stage1_iters,
                is_latest=False,
                checkpoint_path=image_checkpoint_dir,
            )


        if batches is None:
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)


        if hasattr(runner, 'build_pseudo_gt_batch'):
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=None,
                save_prefix=None,
                use_canonical_root=True,
            )
            print(f"\033[92m[✓] Generated {len(pseudo_batch)} pseudo-GT batches for Stage 2\033[0m")
        else:
            pseudo_batch = batches
            print(f"\033[93m[Warning] build_pseudo_gt_batch not available, using original batch\033[0m")


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


        runner.save(iteration=inv_iters, is_latest=False)
        print(f"\033[92m[✓] Completed Inversion Stage 2 ({inv_stage2_iters} iters)\033[0m")


        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_iters,
                is_latest=False,
                checkpoint_path=image_checkpoint_dir,
            )


        if hasattr(runner, 'build_pseudo_gt_batch'):
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=None,
                save_prefix=None,
                use_canonical_root=False,
            )

    elif inv_stage1_iters > 0:

        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_stage1_iters,
                is_latest=False,
                checkpoint_path=image_checkpoint_dir,
            )

        if hasattr(runner, 'build_pseudo_gt_batch'):
            try:
                pseudo_batch = runner.build_pseudo_gt_batch(
                    batches,
                    n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                    save_dir=None,
                    save_prefix=None,
                    use_canonical_root=False,
                )
            except Exception as e:
                print(f"\033[93m[Warning] Failed to generate pseudo batch: {e}\033[0m")
                pseudo_batch = batches
        else:
            pseudo_batch = batches


    if finetune_iters > 0:
        if pseudo_batch is None:
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)


            if 'text-to-avatar' in cli_input_dir:
                batches[0]['dataset_id'] = 0
            else:
                batches[0]['dataset_id'] = 2

            if hasattr(runner, 'build_pseudo_gt_batch'):
                pseudo_batch = runner.build_pseudo_gt_batch(
                    batches,
                    n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                    save_dir=None,
                    save_prefix=None,
                    use_canonical_root=True,
                )
            else:
                pseudo_batch = batches

        finetune_pbar = tqdm(range(1, finetune_iters + 1), desc='Finetuning', dynamic_ncols=True)
        for step in finetune_pbar:
            total_step = inv_iters + step

            if use_visible_only_strategy:


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





            is_last = (total_step == total_iters)
            do_test = is_last

            if do_test:
                iteration = total_step
                checkpoint_name = f"{image_name}_iter{iteration}"
                runner.save(iteration=iteration, is_latest=False)
                print(f"\033[92m[✓] Saved checkpoint: {checkpoint_name} at iteration {iteration}\033[0m")


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

    app.run(main)
