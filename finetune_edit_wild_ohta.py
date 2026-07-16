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
import cv2
from tqdm import tqdm
from data.hand_dataset import merge_batch
from LHM.losses import _is_better

from torch.utils.tensorboard import SummaryWriter


flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_string('checkpoint-path', None, 'Checkpoint directory to save/load checkpoints')
flags.DEFINE_string('output-path', './output/finetune_edit', 'Output/test directory to write results')
flags.DEFINE_boolean('resume', True, 'Enable auto-resume from latest checkpoint')
flags.DEFINE_string('checkpoint-file', None,
                    'Specific checkpoint file to load (overrides checkpoint-path)')
flags.DEFINE_string('input-dir', None,
                    'Optional directory containing in-the-wild images to process (process all files)')
flags.DEFINE_string('handavatar-path', os.environ.get('HANDAVATAR_ROOT'), 'HandAvatar dataset root')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input')
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_boolean('use_amp', True, 'Use automatic mixed precision (AMP)')
flags.DEFINE_multi_string(
    'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')
flags.DEFINE_integer('test-iter', 9000, 'Test iteration')
flags.DEFINE_integer('iter', '800', 'Total finetuning iterations (inversion + stage2)')
flags.DEFINE_integer('iter_inversion', 100, 'Inversion iterations (color optimization with edit masked)')
flags.DEFINE_integer('edit-unmask-iter', 0, 'Stage2 iterations before unmasking edit region')
flags.DEFINE_float('edit-mask-weight', 30.0, 'Loss weight multiplier on edit region in unmasked phase')
flags.DEFINE_float('pseudo-view-weight', 0.1, 'Weight for pseudo-view supervision in unmasked (edit) phase')
flags.DEFINE_integer('pseudo-views', 8, 'Number of pseudo-GT views')
flags.DEFINE_boolean('animate_to_handavatar', True, 'Whether to animate inferred GS to HandAvatar test poses')
flags.DEFINE_boolean('save_test_gt_images', False, 'Whether to save transformed GT images during testing')
flags.DEFINE_string('edit', 'True', 'Enable edit mode in dataset')


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

    parser = argparse.ArgumentParser(description="OHTA-style Edit Wild Finetuner")
    parser.add_argument("--runner", default=DEFAULT_RUNNER, type=str, help="Runner to launch")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--output-path", type=str, default='./output/finetune_edit')
    parser.add_argument("--handavatar-path", type=str, default=os.environ.get("HANDAVATAR_ROOT"))
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--test-iter", type=int, default=None)
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
            k = k.strip(); value = value.strip(); i += 1
        elif ' ' in tok and not tok.startswith('-'):
            k, value = tok.split(' ', 1)
            k = k.strip(); value = value.strip(); i += 1
        elif not tok.startswith('-'):
            if i + 1 < len(raw_tokens) and not raw_tokens[i + 1].startswith('-') and '=' not in raw_tokens[i + 1]:
                k = tok.strip(); value = raw_tokens[i + 1].strip(); i += 2
            else:
                i += 1; continue
        else:
            i += 1; continue
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
    if cli_checkpoint is None: cli_checkpoint = args.checkpoint_path
    if cli_checkpoint_file is None: cli_checkpoint_file = args.checkpoint_file
    if cli_output is None: cli_output = args.output_path
    if cli_handavatar is None: cli_handavatar = args.handavatar_path
    if cli_input_dir is None: cli_input_dir = args.input_dir

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


    if not dist.is_available():
        print("[Warning] torch.distributed not available, skipping init_process_group")
    else:
        try:
            port = random.randint(30000, 40000)
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(port)
            dist.init_process_group(backend='gloo', init_method='env://',
                                    rank=0, world_size=1, timeout=torch.distributed.timedelta(minutes=30))
            print(f"[+] Distributed process group initialized on port {port}")
        except Exception as e:
            print(f"[Warning] Failed to init_process_group: {e}, continuing without distributed")

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device_id = rank % torch.cuda.device_count()

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
    image_checkpoint_dir = os.path.join(base_output_root, f"finetune_edit_{image_name}")
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
    print(f"\033[94mEdit Wild OHTA Fine-tuning Configuration:\033[0m")
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
        test_hand = HandDataset(split='test_wild', img_path=img_glob[0], edit=FLAGS.edit)
        test_dataloader = make_dataloader(test_hand, shuffle=False, batch_size=1)


    import re
    if args.test_iter is not None:
        try:
            test_iters = int(args.test_iter)
        except Exception:
            test_iters = 12000
    elif cli_checkpoint_file is not None:
        base = os.path.basename(cli_checkpoint_file)
        m = re.search(r'iteration[_-]?(\d+)', base)
        if m:
            test_iters = int(m.group(1))
        else:
            m2 = re.search(r'(\d{3,7})', base)
            test_iters = int(m2.group(1)) if m2 else 12000
    else:
        test_iters = 12000

    try:
        if hasattr(runner, 'load_checkpoint'):
            runner.finetune_model(iteration=test_iters, is_latest=False,
                                  checkpoint_path=cli_checkpoint, checkpoint_file=cli_checkpoint_file)
        else:
            runner.load(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint)
    except TypeError as e:
        print(f"\033[91mCheckpoint loading failed: {e}\033[0m")
        runner.load(iteration=test_iters, is_latest=False, checkpoint_path=cli_checkpoint)

    print(f"\033[91m[========== Loading model for iteration {test_iters} ==========]\033[0m")

    runner.checkpoint_path = image_checkpoint_dir
    if hasattr(runner, 'hand_model') and hasattr(runner.hand_model, 'checkpoint_path'):
        runner.hand_model.checkpoint_path = image_checkpoint_dir


    handavatar_dataloader = None
    if FLAGS.animate_to_handavatar:
        try:
            if cli_handavatar is None:
                raise ValueError("--handavatar-path is required when animation is enabled")
            handavatar_path = cli_handavatar
            handavatar_ds = HandAvatarDataset(dataset_path=handavatar_path, data_type='progress', skip=200)
            handavatar_dataloader = make_dataloader(handavatar_ds, shuffle=False, batch_size=1)
            print(f"\033[92m[+] Loaded HandAvatar dataset with {len(handavatar_dataloader)} samples\033[0m")
        except Exception as e:
            print(f"\033[91m[Warning] Failed to load HandAvatar dataset: {e}\033[0m")
            handavatar_dataloader = None


    total_iters = FLAGS.iter
    inv_iters = int(getattr(FLAGS, 'iter_inversion', 200))
    edit_unmask_iter = max(0, int(getattr(FLAGS, 'edit_unmask_iter', 300)))
    edit_unmask_iter = int(getattr(FLAGS, 'edit-unmask-iter'))
    edit_mask_weight = float(getattr(FLAGS, 'edit_mask_weight', 30.0))
    pseudo_view_weight = float(getattr(FLAGS, 'pseudo_view_weight', 0.1))
    finetune_iters = max(0, total_iters - inv_iters)
    batches = None

    data_iterator = iter(test_dataloader)

    print(f"\033[94m{'='*60}\033[0m")
    print(f"\033[94mEdit Wild OHTA Training Plan:\033[0m")
    print(f"  Inversion (edit-masked color optim): {inv_iters} iters")
    print(f"  Stage2 total: {finetune_iters} iters")
    print(f"    - Masked (learn hand texture): first {min(edit_unmask_iter, finetune_iters)} iters")
    print(f"    - Unmasked (learn edit pattern): remaining {max(0, finetune_iters - edit_unmask_iter)} iters")
    print(f"  Edit mask weight: {edit_mask_weight}")
    print(f"  Pseudo-view weight (unmasked): {pseudo_view_weight}")
    print(f"\033[94m{'='*60}\033[0m\n")


    if inv_iters > 0:
        inv_pbar = tqdm(range(1, inv_iters + 1), desc='Edit Inversion', dynamic_ncols=True)
        for step in inv_pbar:
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)

            batches[0]['dataset_id'] = 2
            runner.run_edit_wild_inversion(
                batch=batches,
                scaler=scaler,
                iteration=step,
                writer=writer,
                pbar=inv_pbar,
                total_iters=inv_iters,
            )

        runner.save(iteration=inv_iters, is_latest=False)
        print(f"\033[92m[+] Completed Edit Inversion ({inv_iters} iters)\033[0m")


    pseudo_batch = None
    if inv_iters > 0:
        if hasattr(runner, 'load_checkpoint_with_color_shift_scale'):
            runner.load_checkpoint_with_color_shift_scale(
                iteration=inv_iters, is_latest=False, checkpoint_path=image_checkpoint_dir,
            )
        if hasattr(runner, 'build_pseudo_gt_batch') and batches is not None:
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=None,
                save_prefix=None,
                use_canonical_root=True,
            )
            print(f"\033[92m[+] Generated pseudo-GT views for reference\033[0m")


    if finetune_iters > 0:
        finetune_pbar = tqdm(range(1, finetune_iters + 1), desc='Edit Stage2', dynamic_ncols=True)
        for step in finetune_pbar:
            total_step = inv_iters + step
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)

            batches[0]['dataset_id'] = 2
            runner.run_edit_wild_stage2(
                batch=batches,
                pseudo_batch=pseudo_batch,
                scaler=scaler,
                iteration=step - 1,
                writer=writer,
                pbar=finetune_pbar,
                total_iters=finetune_iters,
                edit_unmask_iter=edit_unmask_iter,
                edit_mask_weight=edit_mask_weight,
                pseudo_view_weight=pseudo_view_weight,
            )


            is_last = (total_step == total_iters)
            do_test = is_last or (total_step % 1000 == 0)

            if do_test:
                iteration = total_step
                runner.save(iteration=iteration, is_latest=False)
                print(f"\033[92m[+] Saved checkpoint at iteration {iteration}\033[0m")

                runner.hand_model.eval()
                runner._sync_color_inversion_params()

                print(f"\033[91m[========== Testing @ iteration {iteration} ==========]\033[0m")

                psnr_full = 0
                ssim_full = 0
                lpips_full = 0
                count = 0

                infer_batch = test_hand.get_img()
                gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

                if handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
                    for map_idx, map_batch in enumerate(handavatar_dataloader):
                        try:
                            map_batch[0]['dataset_id'] = 0
                        except Exception:
                            pass
                        with torch.no_grad():
                            _ = runner.test_handavatar(gs_model_list, gs_densify_list, query_points,
                                                       map_batch, map_idx, iteration, pbar=None)
                    print(f"\033[92m[+] Completed HandAvatar pose animation\033[0m")

                for idx, test_batch in enumerate(test_dataloader):
                    with tqdm(total=1, desc=f"Iter {iteration} | Sample {idx+1}/{len(test_dataloader)}",
                              leave=True, position=idx + 1, dynamic_ncols=True) as _pbar:
                        with torch.no_grad():
                            test_batch[0]['dataset_id'] = 2
                            loss_dict = runner.test_handavatar(
                                gs_model_list, gs_densify_list, query_points,
                                test_batch, idx, iteration, _pbar)

                        psnr_full += loss_dict['psnr'].item()
                        ssim_full += loss_dict['ssim'].item()
                        lpips_full += loss_dict['lpips'].item()
                        count += 1

                        _pbar.set_postfix({
                            "lpips": f"{lpips_full/count:.4f}",
                            "ssim": f"{ssim_full/count:.4f}",
                            "psnr": f"{psnr_full/count:.4f}",
                        })
                        _pbar.update(1)
                        torch.cuda.empty_cache()

                if count > 0:
                    print(f"\033[91m[Summary @iter {iteration}]\033[0m "
                          f"PSNR: {psnr_full/count:.4f}, "
                          f"SSIM: {ssim_full/count:.4f}, "
                          f"LPIPS: {lpips_full/count:.4f}\n")

                current_metrics = {
                    'image_name': image_name,
                    'lpips': lpips_full / count if count > 0 else 0.0,
                    'ssim': ssim_full / count if count > 0 else 0.0,
                    'psnr': psnr_full / count if count > 0 else 0.0,
                    'iteration': int(iteration),
                    'timestamp': time.time(),
                }
                metrics_path = os.path.join(image_checkpoint_dir, f'metrics_iter{iteration}.json')
                with open(metrics_path, 'w') as f:
                    json.dump(current_metrics, f, indent=2)
                print(f"\033[92m[+] Saved metrics to {metrics_path}\033[0m")

                runner.hand_model.train()


    runner._finetune_edit_stage = None
    runner.hand_model.renderer.edit_mask_mode = False
    runner.hand_model.renderer.edit_vis_mask = None
    print(f"[edit_wild_stage2] Finished. Visibility masking cleared.")


if __name__ == "__main__":
    app.run(main)
