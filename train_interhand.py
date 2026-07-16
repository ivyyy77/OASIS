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

from LHM.runners import REGISTRY_RUNNERS
from data.interhand.train import Dataset, HandAvatarDataset
from data.wild_hand_dataset import make_dataloader
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
import time
import json
import re
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from LHM.losses import _is_better




flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input')
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')

flags.DEFINE_string('checkpoint-path', None, 'Checkpoint directory to save/load checkpoints')
flags.DEFINE_string('output-path', None, 'Output/test directory to write results')
flags.DEFINE_boolean('resume', False, 'Enable auto-resume from latest checkpoint')
flags.DEFINE_string('checkpoint-file', None, 'Specific checkpoint file to load (overrides checkpoint-path)')
flags.DEFINE_string('dataset-path', os.environ.get('INTERHAND_ROOT'), 'InterHand2.6M root')
flags.DEFINE_string('handavatar-path', os.environ.get('HANDAVATAR_ROOT'), 'HandAvatar evaluation root')
flags.DEFINE_multi_string(
  'gin_file', 'splatformer/configs/train/default.gin', 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '"build_trainloader.batch_size="32', 'Newline separated list of Gin parameter bindings.')


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


            total_render_time += loss_dict.get('render_time', 0.0)
            total_rendered_views += loss_dict.get('render_views', 0)

            psnr_full += loss_dict['psnr'].item()
            ssim_full += loss_dict['ssim'].item()
            lpips_full += loss_dict['lpips'].item()
            count += 1


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













    parser = argparse.ArgumentParser(description="OpenLRM launcher")
    parser.add_argument("runner", type=str, help="Runner to launch")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Optional checkpoint directory to save/load checkpoints (overrides env)")
    parser.add_argument("--checkpoint-file", type=str, default=None,
                        help="Optional specific checkpoint file to load (overrides checkpoint-path)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Optional output/test directory to write results (overrides env)")
    parser.add_argument("--dataset-path", default=os.environ.get("INTERHAND_ROOT"),
                        help="InterHand2.6M preprocessing root (or set INTERHAND_ROOT)")
    parser.add_argument("--handavatar-path", default=os.environ.get("HANDAVATAR_ROOT"),
                        help="HandAvatar preprocessing root used for evaluation (or set HANDAVATAR_ROOT)")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Enable auto-resume from latest checkpoint")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Disable auto-resume from checkpoint")
    parser.set_defaults(resume=False)


    args, unknown = parser.parse_known_args()

    args.dataset_path = args.dataset_path or FLAGS['dataset-path'].value
    args.handavatar_path = args.handavatar_path or FLAGS['handavatar-path'].value

    if not args.dataset_path:
        parser.error("--dataset-path is required when INTERHAND_ROOT is not set")
    if not args.handavatar_path:
        parser.error("--handavatar-path is required when HANDAVATAR_ROOT is not set")


    cli_checkpoint = args.checkpoint_path
    cli_checkpoint_file = args.checkpoint_file
    cli_output = args.output_path
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

        if not hasattr(args, 'resume') or args.resume is None:
            cli_resume = getattr(absl_f, 'resume', False)
    except Exception:
        pass


    for u in unknown:
        if isinstance(u, str) and u.lower().startswith("checkpoint-path=") and cli_checkpoint is None:
            cli_checkpoint = u.split("=", 1)[1]
        elif isinstance(u, str) and u.lower().startswith("checkpoint=") and cli_checkpoint is None:
            cli_checkpoint = u.split("=", 1)[1]
        if isinstance(u, str) and u.lower().startswith("output-path=") and cli_output is None:
            cli_output = u.split("=", 1)[1]
        elif isinstance(u, str) and u.lower().startswith("output=") and cli_output is None:
            cli_output = u.split("=", 1)[1]

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError("Runner {} not found".format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]

    runner = None
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
                setattr(runner, 'cli_resume', cli_resume)
        except Exception:
            pass














    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29502',
                            rank=0,
                            world_size=1,


                            )
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()



    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed(42)
    iteration = FLAGS.iter




















































    train_loader = Dataset(dataset_path=args.dataset_path,
                           data_type='train')











    train_dataloader = make_dataloader(train_loader, shuffle=True, batch_size=FLAGS.batch_size,
                                       num_workers=FLAGS.num_workers,
                                       persistent_workers=(FLAGS.num_workers>0),
                                       pin_memory=FLAGS.pin_memory)


    test_handavatar = HandAvatarDataset(dataset_path=args.handavatar_path,
                                                d_type='progress',
                                                skip=200)

    test_dataloader = make_dataloader(test_handavatar, shuffle=False, batch_size=1,
                                      num_workers=FLAGS.num_workers,
                                      persistent_workers=(FLAGS.num_workers>0),
                                      pin_memory=FLAGS.pin_memory)


    data_iterator = iter(train_dataloader)

    writer = SummaryWriter(log_dir=os.path.join(FLAGS.output_dir, 'exp1'))

    if FLAGS.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


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


    pbar = tqdm(range(1, iteration + 1), desc='Training', dynamic_ncols=True)

    for iteration in pbar:



        try:
            batches = next(data_iterator)

        except:
            data_iterator = iter(train_dataloader)
            batches = next(data_iterator)

        runner.run(batch=batches, scaler=scaler, iteration=iteration, writer=writer, pbar=pbar)





        if iteration >= 1000 and iteration % 1000 == 0:



            runner.save(is_latest=True)

            runner.load(is_latest=True)
            print(f"\033[91m[========== Running test ! Loading model for iteration {iteration} ==========]\033[0m")

            infer_batch = test_handavatar.get_img()


            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

            current_metrics = _run_test_with_fps(
                runner,
                test_dataloader,
                gs_model_list,
                gs_densify_list,
                query_points,
                iteration,
            )



            current_metrics['iteration_tested_on'] = int(iteration)
            current_metrics['timestamp'] = time.time()


            test_results_dir = os.path.join(runner.checkpoint_path, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)

            latest_results_path = os.path.join(test_results_dir, 'latest_test_results.json')
            best_results_path = os.path.join(test_results_dir, 'best_test_results.json')


            with open(latest_results_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)








            if os.path.exists(best_results_path):
                try:
                    with open(best_results_path, 'r') as f:
                        best_metrics = json.load(f)
                except Exception as e:
                    print(f"[Warning] failed to load best_results_path ({best_results_path}): {e}")
                    best_metrics = None
            else:

                best_metrics = None



            if (best_metrics is None) or _is_better(current_metrics, best_metrics):

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

    app.run(main)
