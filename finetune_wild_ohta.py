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

from tqdm import tqdm
# from sklearn.preprocessing import maxabs_scale

from LHM.runners import REGISTRY_RUNNERS
# from splatformer.models.feature_predictor import FeaturePredictor
from torch.nn import Tanh, Identity

from data.debug import ReconstructionDataset, HandDataset
from data.interhand.train import Dataset, HandAvatarDataset
from data.lisa_test_dataset import LisaTestDataset, make_lisa_test_dataloader
from data.debug import make_dataloader
import torch.nn as nn
import torch.distributed as dist
import torch, os, random, gin
from absl import flags, app
import numpy as np
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP
# ==== Attention Visualization (Ablation Study) ====
from LHM.models.transformer_dit import set_attn_viz_enabled, get_attn_visualizer, set_vis_bias_enabled
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
flags.DEFINE_string('checkpoint-file', './checkpoint/interhand-correct/iteration_6000.ckpt', 'Specific checkpoint file to load (overrides checkpoint-path)')
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
flags.DEFINE_integer('iter_inversion_stage1', 200, 'Stage1 inversion iters (single-view color optimization)')
flags.DEFINE_integer('iter_inversion_stage2', 0, 'Stage2 inversion iters (pseudo-view color consistency)')
flags.DEFINE_float('color-consistency-weight', 1.0, 'Weight for color consistency loss in inversion stage2')
flags.DEFINE_integer('pseudo-views', 8, 'Number of pseudo-GT views')
flags.DEFINE_boolean('animate_to_handavatar', True, 'Whether to animate inferred GS to HandAvatar test poses')
flags.DEFINE_boolean('save_test_gt_images', False, 'Whether to save transformed GT images during testing')
flags.DEFINE_integer('edit-unmask-iter', 300, 'Stage2 iterations before unmasking edit region')
flags.DEFINE_float('edit-mask-weight', 30.0, 'Loss weight multiplier on edit region in unmasked phase')
flags.DEFINE_float('pseudo-view-weight', 0.1, 'Weight for pseudo-view supervision in unmasked (edit) phase')
flags.DEFINE_string('edit', 'True', 'Enable edit mode in dataset (auto-detected from input path)')
# ==== Attention Visualization (Ablation Study) ====
flags.DEFINE_boolean('viz_attn', False, 'Enable attention score visualization for ablation study')
flags.DEFINE_string('viz_attn_layers', None, 'Comma-separated layer indices to visualize (e.g., "0,3,5"). None=all')
flags.DEFINE_string('viz_attn_heads', None, 'Comma-separated head indices to visualize. None=all')
flags.DEFINE_integer('viz_attn_step_interval', 50, 'Capture attention every N steps')
flags.DEFINE_string('viz_attn_save_dir', 'output/attn_debug_v2', 'Directory for v2 attention ratio visualizations')
flags.DEFINE_boolean('viz_attn_canonical', False, 'Save posed and canonical point-level attention ratio maps')
flags.DEFINE_string('viz_attn_canonical_dir', 'output/attn_canonical', 'Directory for canonical attention visualizations')
flags.DEFINE_boolean('viz_attn_enable_vis_bias', True, 'Enable visibility attention bias during attention visualization runs')
flags.DEFINE_boolean('viz_attn_forward_only', False, 'Run one latent forward for attention capture, then save visualizations and exit')


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


def _render_mano_mesh_overlay(world_vertex_np, K_np, R_np, T_np, faces, H, W, bg_rgb_np=None):
    """Render posed MANO mesh overlay using pyrender with actual camera intrinsics.

    Args:
        world_vertex_np: (N, 3) numpy array of MANO vertices in world space.
        K_np: (3, 3) camera intrinsic matrix.
        R_np: (3, 3) world-to-camera rotation.
        T_np: (3,) world-to-camera translation.
        faces: (F, 3) face indices (numpy or tensor).
        H, W: image height / width.
        bg_rgb_np: optional (H, W, 3) uint8 RGB background image.
    Returns:
        (H, W, 3) uint8 RGB image with mesh overlay.
    """
    import pyrender as _pyrender
    import trimesh as _trimesh

    # Convert faces to numpy if needed
    if isinstance(faces, torch.Tensor):
        faces_np = faces.cpu().numpy()
    else:
        faces_np = np.array(faces)

    # World → camera space (OpenCV convention: +Z towards scene)
    v_cam = (R_np @ world_vertex_np.T).T + T_np.reshape(1, 3)  # (N, 3)

    # Build trimesh in camera space then convert OpenCV → OpenGL (flip Y and Z)
    mesh = _trimesh.Trimesh(v_cam.copy(), faces_np.copy())
    rot = _trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    material = _pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.40, 0.55, 0.85, 1.0),
    )
    mesh_pr = _pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = _pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4))
    scene.add(mesh_pr, 'mesh')

    fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
    cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
    camera = _pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
    scene.add(camera, pose=np.eye(4))

    # Two directional lights for decent shading
    light = _pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))
    lpose = np.eye(4); lpose[:3, 3] = [0, -1, 1]
    scene.add(light, pose=lpose)

    renderer = _pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    try:
        color, _ = renderer.render(scene, flags=_pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    color_f = color.astype(np.float32) / 255.0
    valid_mask = color_f[:, :, 3:4]
    if bg_rgb_np is not None:
        bg = np.clip(bg_rgb_np.astype(np.float32) / 255.0, 0, 1)
        out = color_f[:, :, :3] * valid_mask + bg * (1.0 - valid_mask)
    else:
        out = color_f[:, :, :3] * valid_mask + np.ones((H, W, 3), dtype=np.float32) * (1.0 - valid_mask)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


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

    def _is_text_to_avatar_input(input_path):
        if not input_path:
            return False
        path_str = str(input_path)
        if os.path.isabs(path_str):
            try:
                path_str = os.path.relpath(path_str, os.getcwd())
            except Exception:
                pass
        return 'text-to-avatar' in path_str

    is_text_to_avatar = _is_text_to_avatar_input(cli_input_dir)
    setattr(runner, '_is_text_to_avatar', is_text_to_avatar)

    # ==============================================================================================
    # For single-process execution, use gloo backend with dynamic port
    import random
    if not dist.is_available():
        print("[Warning] torch.distributed not available, skipping init_process_group")
    else:
        try:
            # Try to use an available port
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

    # Edit mode detection (works for both single and multi-image paths)
    is_edit_mode = 'editing' in (cli_input_dir or '')


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

    # ==== Enable Attention Visualization if requested ====
    if getattr(FLAGS, 'viz_attn', False):
        viz_save_dir = getattr(FLAGS, 'viz_attn_save_dir', 'output/attn_debug_v2')
        capture_layers = None
        capture_heads = None
        if getattr(FLAGS, 'viz_attn_layers', None):
            capture_layers = [int(x) for x in FLAGS.viz_attn_layers.split(',')]
        if getattr(FLAGS, 'viz_attn_heads', None):
            capture_heads = [int(x) for x in FLAGS.viz_attn_heads.split(',')]
        
        set_attn_viz_enabled(
            enabled=True,
            save_dir=viz_save_dir,
            capture_layers=capture_layers,
            capture_heads=capture_heads,
        )
        set_vis_bias_enabled(bool(getattr(FLAGS, 'viz_attn_enable_vis_bias', True)))
        print(f"\033[95m{'='*60}\033[0m")
        print(f"\033[95m[Attention Visualization ENABLED]\033[0m")
        print(f"  Save dir: {viz_save_dir}")
        print(f"  Capture layers: {capture_layers or 'ALL'}")
        print(f"  Capture heads: {capture_heads or 'ALL'}")
        print(f"  Step interval: {getattr(FLAGS, 'viz_attn_step_interval', 50)}")
        print(f"\033[95m{'='*60}\033[0m\n")
    # =====================================================

    def _attn_file_prefix():
        ckpt_for_tag = str(cli_checkpoint_file or getattr(FLAGS, 'checkpoint_file', '') or '')
        iter_match = re.search(r'iteration_(\d+)', ckpt_for_tag)
        image_match = re.search(r'image(\d+)', str(image_name))
        iter_tag = iter_match.group(1) if iter_match else str(test_iters)
        image_tag = image_match.group(1) if image_match else image_name
        return f"iter{iter_tag}-image{image_tag}"

    def _save_rgb_and_mesh(batches_in):
        """Save input RGB image and posed MANO mesh render into viz_attn_canonical_dir."""
        _save_dir = FLAGS.viz_attn_canonical_dir if getattr(FLAGS, 'viz_attn_canonical', False) else None
        if _save_dir is None:
            return
        os.makedirs(_save_dir, exist_ok=True)
        _prefix = _attn_file_prefix()
        try:
            batch = batches_in[0]

            # ---- Save original RGB input image ----
            orig = batch['original_image']
            if isinstance(orig, torch.Tensor):
                img_v = orig[0] if orig.dim() == 4 else orig  # [3, H, W]
                img_np = img_v.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = np.array(orig).squeeze()
                if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):
                    img_np = img_np.transpose(1, 2, 0)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            H_img, W_img = img_np.shape[:2]

            rgb_path = os.path.join(_save_dir, f"{_prefix}_input_rgb.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            print(f"\033[92m[AttnViz] Saved input RGB → {rgb_path}\033[0m")

            # ---- Render posed MANO mesh ----
            def _to_np(arr, shape):
                if isinstance(arr, torch.Tensor):
                    return arr[0].cpu().numpy().reshape(shape)
                return np.array(arr[0]).reshape(shape)

            wv_np = _to_np(batch['world_vertex'], (-1, 3))
            K_np  = _to_np(batch['K'],  (3, 3))
            R_np  = _to_np(batch['R'],  (3, 3))
            T_np  = np.array(batch['T'][0]).reshape(-1)[:3]

            faces = runner.hand_model.renderer.mano_model.mano.faces
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()

            mesh_img = _render_mano_mesh_overlay(wv_np, K_np, R_np, T_np, faces, H_img, W_img, img_np)
            mesh_path = os.path.join(_save_dir, f"{_prefix}_posed_mesh.png")
            cv2.imwrite(mesh_path, cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR))
            print(f"\033[92m[AttnViz] Saved posed mesh → {mesh_path}\033[0m")

        except Exception as _e:
            print(f"\033[91m[AttnViz] Warning: RGB/mesh save failed: {_e}\033[0m")
            import traceback as _tb; _tb.print_exc()

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
    lisa_test_dataloader = None  # Lisa experiment test dataloader
    
    # Check if this is a Lisa experiment (output path contains 'lisa_experiment')
    is_lisa_experiment = cli_output is not None and 'lisa_experiment' in cli_output
    
    if is_lisa_experiment:
        # Lisa experiment: use cam400053 and cam400018 as test views
        print(f"\033[95m{'='*60}\033[0m")
        print(f"\033[95m[Lisa Experiment Mode Detected]\033[0m")
        print(f"\033[95mUsing cam400053_image4734 and cam400018_image4734 as test views\033[0m")
        print(f"\033[95m{'='*60}\033[0m\n")
        try:
            lisa_test_ds = LisaTestDataset(
                data_root='example_data/interhand2.6m',
                # test_views=['cam400053_image4734', 'cam400018_image4734']
                test_views=['cam400042_image6077', 'cam400018_image6077']
            )
            lisa_test_dataloader = make_dataloader(lisa_test_ds, shuffle=False, batch_size=1)
            print(f"\033[92m[✓] Loaded Lisa test dataset with {len(lisa_test_dataloader)} test views\033[0m")
        except Exception as e:
            print(f"\033[91m[Warning] Failed to load Lisa test dataset: {e}\033[0m")
            import traceback
            traceback.print_exc()
            lisa_test_dataloader = None
    
    if FLAGS.animate_to_handavatar and not is_lisa_experiment:
        # Only load HandAvatar if not in Lisa experiment mode
        try:
            handavatar_path = '/scratch/groups/su004-neuralnet/zh174/InterHand/5'
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
        # infer_batch = test_hand.get_img()
        infer_batch = test_hand.get_img_rainbow()
        gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

        # ============= Lisa Experiment: Render to test views =============
        if is_lisa_experiment and lisa_test_dataloader is not None:
            print(f"\033[95m[Lisa Experiment] Rendering to cam400053 and cam400018 test views\033[0m")
            lisa_psnr = 0
            lisa_ssim = 0
            lisa_lpips = 0
            lisa_count = 0
            
            for lisa_idx, lisa_batch in enumerate(lisa_test_dataloader):
                try:
                    lisa_batch[0]['dataset_id'] = 0
                except Exception:
                    pass
                
                with torch.no_grad():
                    lisa_loss_dict = runner.test_handavatar(
                        gs_model_list, gs_densify_list, query_points,
                        lisa_batch, lisa_idx, eval_iter, pbar=None
                    )
                
                if lisa_loss_dict is not None:
                    lisa_psnr += lisa_loss_dict.get('psnr', torch.tensor(0)).item()
                    lisa_ssim += lisa_loss_dict.get('ssim', torch.tensor(0)).item()
                    lisa_lpips += lisa_loss_dict.get('lpips', torch.tensor(0)).item()
                    lisa_count += 1
                    print(f"  [View {lisa_idx}] PSNR: {lisa_loss_dict['psnr'].item():.4f}, "
                          f"SSIM: {lisa_loss_dict['ssim'].item():.4f}, "
                          f"LPIPS: {lisa_loss_dict['lpips'].item():.4f}")
            
            if lisa_count > 0:
                print(f"\033[95m[Lisa Summary @eval {eval_iter}]\033[0m "
                      f"PSNR: {lisa_psnr/lisa_count:.4f}, "
                      f"SSIM: {lisa_ssim/lisa_count:.4f}, "
                      f"LPIPS: {lisa_lpips/lisa_count:.4f}")
                
                # Save Lisa metrics
                lisa_metrics = {
                    'experiment': 'lisa',
                    'train_view': 'cam400012_image4734',
                    'test_views': ['cam400053_image4734', 'cam400018_image4734'],
                    'psnr': lisa_psnr / lisa_count,
                    'ssim': lisa_ssim / lisa_count,
                    'lpips': lisa_lpips / lisa_count,
                    'iteration': int(eval_iter),
                }
                lisa_metrics_path = os.path.join(image_checkpoint_dir, f'lisa_metrics_eval_iter{int(eval_iter)}.json')
                with open(lisa_metrics_path, 'w') as f:
                    json.dump(lisa_metrics, f, indent=2)
                print(f"\033[92m[✓] Saved Lisa metrics to {lisa_metrics_path}\033[0m")
            
            print(f"\033[92m[✓] Completed Lisa experiment evaluation\033[0m")
        
        # Optionally animate GS to HandAvatar poses/cameras (skip if Lisa experiment)
        elif handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
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

        if not is_lisa_experiment:
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
        else:
            print(f"\033[95m[Lisa Experiment] Skip default input-view eval in eval-only mode\033[0m")
        
        # ==== Save Attention Visualizations ====
        if getattr(FLAGS, 'viz_attn', False):
            visualizer = get_attn_visualizer()
            if visualizer is not None and visualizer.captured_data:
                print(f"\033[95m[AttnViz] Generating attention visualizations...\033[0m")
                summary = visualizer.visualize_all(
                    canonical_root_dir=FLAGS.viz_attn_canonical_dir if FLAGS.viz_attn_canonical else None,
                    image_name=image_name,
                    file_prefix=_attn_file_prefix(),
                )
                visualizer.save_raw_attention()
                if FLAGS.viz_attn_canonical:
                    visualizer.append_summary_csv(
                        os.path.join(FLAGS.viz_attn_canonical_dir, 'summary_stats.csv'),
                        summary.get('canonical_rows', []),
                    )
                print(f"\033[92m[✓] Attention visualizations saved\033[0m")
        # ========================================
        
        return


    # ====================== Finetune + test branch ======================

    total_iters = FLAGS.iter
    # Two-stage inversion: stage1 (single-view) + stage2 (pseudo-view color consistency)
    use_two_stage_inversion = getattr(FLAGS, 'use_two_stage_inversion', True)
    inv_stage1_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage1', 200) or 200))
    inv_stage2_iters = max(0, int(getattr(FLAGS, 'iter_inversion_stage2', 200) or 200)) if use_two_stage_inversion else 0
    inv_iters = inv_stage1_iters + inv_stage2_iters  # Total inversion iterations
    color_consistency_weight = float(getattr(FLAGS, 'color_consistency_weight', 1.0) or 1.0)
    
    finetune_iters = max(0, total_iters - inv_iters)
    pseudo_batch = None

    # Edit mode parameters
    edit_mask_weight = float(getattr(FLAGS, 'edit_mask_weight', 100.0))

    data_iterator = iter(test_dataloader)

    if getattr(FLAGS, 'viz_attn_forward_only', False):
        if not getattr(FLAGS, 'viz_attn', False):
            raise ValueError("--viz_attn_forward_only requires --viz_attn")
        try:
            batches = next(data_iterator)
        except Exception:
            data_iterator = iter(test_dataloader)
            batches = next(data_iterator)
        setattr(runner, 'attn_forward_only', True)
        runner._render_edit_core(batches)
        visualizer = get_attn_visualizer()
        if visualizer is not None and visualizer.captured_data:
            print(f"\033[95m[AttnViz] Generating forward-only attention visualizations...\033[0m")
            summary = visualizer.visualize_all(
                canonical_root_dir=FLAGS.viz_attn_canonical_dir if FLAGS.viz_attn_canonical else None,
                image_name=image_name,
                file_prefix=_attn_file_prefix(),
            )
            visualizer.save_raw_attention()
            if FLAGS.viz_attn_canonical:
                visualizer.append_summary_csv(
                    os.path.join(FLAGS.viz_attn_canonical_dir, 'summary_stats.csv'),
                    summary.get('canonical_rows', []),
                )
        setattr(runner, 'attn_forward_only', False)
        return

    print(f"\033[94m{'='*60}\033[0m")
    print(f"\033[94mInversion Training Plan:\033[0m")
    print(f"  Edit mode: {is_edit_mode}")
    print(f"  Stage 1 (single-view): {inv_stage1_iters} iters")
    print(f"  Stage 2 (color consistency): {inv_stage2_iters} iters")
    print(f"  Color consistency weight: {color_consistency_weight}")
    print(f"  Finetune stage: {finetune_iters} iters")
    if is_edit_mode:
        print(f"  Edit mask weight: {edit_mask_weight}")
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

            # batches[0]['dataset_id'] = 2
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
                use_canonical_root=True,  # Use fixed canonical pose for pseudo-GT generation
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
                use_canonical_root=False,  # Use reference pose for Stage 2 rendering
            )
        
        # ------------------- Render Inversion Stage 2 Final Model to Pseudo-GT Poses -------------------
        if pseudo_batch is not None:
            print(f"\033[94m[Rendering Inversion Stage 2 model to pseudo-GT poses]\033[0m")
            
            runner.hand_model.eval()
            runner._sync_color_inversion_params()
            
            # infer_batch = test_hand.get_img()
            infer_batch = test_hand.get_img_rainbow()
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

        if hasattr(runner, 'build_pseudo_gt_batch'):
            save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
            save_prefix = f"{image_name}_inv{inv_stage1_iters}"
            pseudo_batch = runner.build_pseudo_gt_batch(
                batches,
                n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                save_dir=save_dir,
                save_prefix=save_prefix,
                use_canonical_root=True,  # Use canonical pose for pseudo-GT finetune
            )
            
            # ------------------- Render Inversion Stage 1 Final Model to Pseudo-GT Poses -------------------
            print(f"\033[94m[Rendering Inversion Stage 1 model to pseudo-GT poses]\033[0m")
            
            runner.hand_model.eval()
            runner._sync_color_inversion_params()
            
            # infer_batch = test_hand.get_img()
            infer_batch = test_hand.get_img_rainbow()
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
            try:
                batches = next(data_iterator)
            except Exception:
                data_iterator = iter(test_dataloader)
                batches = next(data_iterator)
            if hasattr(runner, 'build_pseudo_gt_batch'):
                save_dir = os.path.join(test_output_dir, 'debug_vis', 'pseudo_views')
                save_prefix = f"{image_name}_inv{inv_iters}"
                pseudo_batch = runner.build_pseudo_gt_batch(
                    batches,
                    n_views=int(getattr(FLAGS, 'pseudo_views', 8)),
                    save_dir=save_dir,
                    save_prefix=save_prefix,
                    use_canonical_root=True,  # Use canonical pose for pseudo-GT finetune
                )
            else:
                pseudo_batch = batches

        finetune_pbar = tqdm(range(1, finetune_iters + 1), desc='Finetuning', dynamic_ncols=True)
        for step in finetune_pbar:
            total_step = inv_iters + step
            runner.run_wild_stage_2(
                batch=pseudo_batch,
                scaler=scaler,
                iteration=step - 1,
                writer=writer,
                pbar=finetune_pbar,
                total_iters=finetune_iters,
                is_text_to_avatar=is_text_to_avatar,
                edit_mask_weight=edit_mask_weight if is_edit_mode else 0.0,
            )

            # Save checkpoint and optionally run testing.
            # Testing uses the current in-memory model directly (no checkpoint
            # reload) so that optimizer state, learning rate, and stage flags
            # are preserved for continued training.
            is_last = (total_step == total_iters)
            do_test = is_last # or (total_step % 500 == 0)

            if do_test:
                iteration = total_step
                checkpoint_name = f"{image_name}_iter{iteration}"
                runner.save(iteration=iteration, is_latest=False)
                print(f"\033[92m[✓] Saved checkpoint: {checkpoint_name} at iteration {iteration}\033[0m")

                # Switch to eval mode for inference; do NOT reload checkpoint
                # (reloading replaces modules and invalidates optimizer refs).
                runner.hand_model.eval()
                # Ensure renderer has up-to-date color_shift/scale refs
                runner._sync_color_inversion_params()

                print(f"\033[91m[========== Running test @ iteration {iteration} ==========]\033[0m")

                psnr_full = 0
                ssim_full = 0
                lpips_full = 0
                image_full = 0

                count = 0

                # infer_batch = test_hand.get_img()
                infer_batch = test_hand.get_img_rainbow()
                gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

                # ============= Lisa Experiment: Render to test views =============
                if is_lisa_experiment and lisa_test_dataloader is not None:
                    print(f"\033[95m[Lisa Experiment] Rendering to cam400053 and cam400018 test views\033[0m")
                    lisa_psnr = 0
                    lisa_ssim = 0
                    lisa_lpips = 0
                    lisa_count = 0
                    
                    for lisa_idx, lisa_batch in enumerate(lisa_test_dataloader):
                        try:
                            lisa_batch[0]['dataset_id'] = 0
                        except Exception:
                            pass
                        
                        with torch.no_grad():
                            lisa_loss_dict = runner.test_handavatar(
                                gs_model_list, gs_densify_list, query_points,
                                lisa_batch, lisa_idx, iteration, pbar=None
                            )
                        
                        if lisa_loss_dict is not None:
                            lisa_psnr += lisa_loss_dict.get('psnr', torch.tensor(0)).item()
                            lisa_ssim += lisa_loss_dict.get('ssim', torch.tensor(0)).item()
                            lisa_lpips += lisa_loss_dict.get('lpips', torch.tensor(0)).item()
                            lisa_count += 1
                            print(f"  [View {lisa_idx}] PSNR: {lisa_loss_dict['psnr'].item():.4f}, "
                                  f"SSIM: {lisa_loss_dict['ssim'].item():.4f}, "
                                  f"LPIPS: {lisa_loss_dict['lpips'].item():.4f}")
                    
                    if lisa_count > 0:
                        print(f"\033[95m[Lisa Summary @iter {iteration}]\033[0m "
                              f"PSNR: {lisa_psnr/lisa_count:.4f}, "
                              f"SSIM: {lisa_ssim/lisa_count:.4f}, "
                              f"LPIPS: {lisa_lpips/lisa_count:.4f}")
                        
                        # Save Lisa metrics
                        lisa_metrics = {
                            'experiment': 'lisa',
                            'train_view': 'cam400012_image4734',
                            'test_views': ['cam400053_image4734', 'cam400018_image4734'],
                            'psnr': lisa_psnr / lisa_count,
                            'ssim': lisa_ssim / lisa_count,
                            'lpips': lisa_lpips / lisa_count,
                            'iteration': int(iteration),
                        }
                        lisa_metrics_path = os.path.join(image_checkpoint_dir, f'lisa_metrics_iter{int(iteration)}.json')
                        with open(lisa_metrics_path, 'w') as f:
                            json.dump(lisa_metrics, f, indent=2)
                        print(f"\033[92m[✓] Saved Lisa metrics to {lisa_metrics_path}\033[0m")
                    
                    print(f"\033[92m[✓] Completed Lisa experiment evaluation\033[0m")

                # --- Animate inferred GS to HandAvatar test poses/cameras (skip if Lisa experiment) ---
                elif handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
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

                if not is_lisa_experiment:
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
                                # test_batch[0]['dataset_id'] = 2
                                loss_dict = runner.test_handavatar(gs_model_list, gs_densify_list, query_points,
                                                                    test_batch, idx, iteration, pbar)

                            psnr_full  += loss_dict['psnr'].item()
                            ssim_full  += loss_dict['ssim'].item()
                            lpips_full += loss_dict['lpips'].item()
                            count += 1

                            pbar.set_postfix({
                                "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
                                "ssim":  f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
                                "psnr":  f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
                            })

                            pbar.update(1)
                            torch.cuda.empty_cache()

                    print(f"\033[91m[Summary @iter {iteration}]\033[0m "
                        f"PSNR: {psnr_full/count:.4f}, "
                        f"SSIM: {ssim_full/count:.4f}, "
                        f"LPIPS: {lpips_full/count:.4f}, "
                        f"L1: {image_full/count:.4f}\n")

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
                else:
                    print(f"\033[95m[Lisa Experiment] Skip default input-view eval in finetune mode\033[0m")

                # Restore train mode; do NOT reset _finetune_wild_stage2_only
                # so optimizer/lr/momentum state are preserved for continued training.
                runner.hand_model.train()

        # ------------------- Final Evaluation: Render to 8 Pseudo-GT poses -------------------
        if is_last and pseudo_batch is not None:
            print(f"\033[94m[Rendering final model to pseudo-GT poses]\033[0m")
            
            # Use final model in eval mode
            runner.hand_model.eval()
            runner._sync_color_inversion_params()
            
            # Infer GS from input image
            # infer_batch = test_hand.get_img()
            infer_batch = test_hand.get_img_rainbow()
            gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)
            
            # Render to pseudo-GT poses directly (without metric computation to avoid shape mismatches)
            final_pseudo_batch = pseudo_batch[0]
            final_pseudo_dir = os.path.join(test_output_dir, 'debug_vis', 'final_pseudo_views')
            os.makedirs(final_pseudo_dir, exist_ok=True)
            
            n_views = final_pseudo_batch['original_image'].shape[0]  # 8
            
            for view_idx in range(n_views):
                
                with torch.no_grad():
                    # Extract single view data
                    cam = runner.hand_model.renderer.get_single_view_cam(final_pseudo_batch, view_idx)
                    smplx_params = final_pseudo_batch['smpl_param']
                    smplx_data = runner.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx)
                    
                    # Ensure smplx_data tensors are on CUDA
                    smplx_data = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in smplx_data.items()}
                    
                    # Direct rendering without post-processing
                    render_res = runner.hand_model.renderer.forward_animate_gs(
                        gs_model_list[0],
                        gs_densify_list[0],
                        query_points[0],
                        cam,
                        smplx_data,
                        256, 256,
                        runner.bg_color,
                    )
                    
                    # Save the rendered RGB
                    if 'comp_rgb' in render_res:
                        comp_rgb = render_res['comp_rgb']
                        
                        # Handle different output shapes
                        if comp_rgb.dim() == 5:  # [bs, nv, H, W, C] or [bs, nv, C, H, W]
                            comp_rgb = comp_rgb[0, 0]  # Take first batch, first view
                        elif comp_rgb.dim() == 4:  # [nv, H, W, C] or [bs, H, W, C]
                            comp_rgb = comp_rgb[0]
                        
                        # Ensure [H, W, C] format
                        if comp_rgb.dim() == 3:
                            if comp_rgb.shape[-1] != 3 and comp_rgb.shape[0] == 3:
                                comp_rgb = comp_rgb.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                            
                            comp_rgb_np = (comp_rgb.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                            comp_rgb_np = cv2.cvtColor(comp_rgb_np, cv2.COLOR_RGB2BGR)
                            out_path = os.path.join(final_pseudo_dir, f'pseudo_gt_view_{view_idx:02d}.jpg')
                            cv2.imwrite(out_path, comp_rgb_np)
            
            print(f"\033[92m[✓] Rendered final model to {n_views} pseudo-GT poses → {final_pseudo_dir}\033[0m")
            
            print(f"\033[92m[✓] Rendered final model to {len(pseudo_batch)} pseudo-GT poses → {final_pseudo_dir}\033[0m")
            
            # Restore train mode (though training is done)
            runner.hand_model.train()

        # ==== Save Attention Visualizations ====
        if getattr(FLAGS, 'viz_attn', False):
            visualizer = get_attn_visualizer()
            if visualizer is not None and visualizer.captured_data:
                print(f"\033[95m[AttnViz] Generating attention visualizations...\033[0m")
                summary = visualizer.visualize_all(
                    canonical_root_dir=FLAGS.viz_attn_canonical_dir if FLAGS.viz_attn_canonical else None,
                    image_name=image_name,
                    file_prefix=_attn_file_prefix(),
                )
                visualizer.save_raw_attention()
                if FLAGS.viz_attn_canonical:
                    visualizer.append_summary_csv(
                        os.path.join(FLAGS.viz_attn_canonical_dir, 'summary_stats.csv'),
                        summary.get('canonical_rows', []),
                    )
                print(f"\033[92m[✓] Attention visualizations saved\033[0m")
        # ========================================

        return

    # # No finetune stage: still allow evaluation on inversion-only result.
    # if inv_iters > 0:
    #     iteration = inv_iters
    #     checkpoint_name = f"{image_name}_iter{iteration}"
    #     runner.save(iteration=iteration, is_latest=False)
    #     print(f"\033[92m[✓] Saved checkpoint: {checkpoint_name} at iteration {iteration}\033[0m")

    #     runner.load(iteration=iteration, is_latest=False, checkpoint_path=image_checkpoint_dir)

    #     print(f"\033[91m[========== Running test ! Loading model for iteration {iteration} ==========]\033[0m")

    #     psnr_full = 0
    #     ssim_full = 0
    #     lpips_full = 0
    #     image_full = 0

    #     count = 0

    #     infer_batch = test_hand.get_img_rainbow()
    #     gs_model_list, gs_densify_list, query_points = runner.infer_handavatar(infer_batch)

    #     if handavatar_dataloader is not None and FLAGS.animate_to_handavatar:
    #         print(f"\033[94m[Animating GS to HandAvatar poses for iteration {iteration}]\033[0m")
    #         for map_idx, map_batch in enumerate(handavatar_dataloader):
    #             if isinstance(map_batch, dict) and 'dataset_id' not in map_batch:
    #                 map_batch['dataset_id'] = 0
    #             elif not isinstance(map_batch, dict):
    #                 try:
    #                     setattr(map_batch, 'dataset_id', 0)
    #                 except Exception:
    #                     pass
    #             try:
    #                 map_batch[0]['dataset_id'] = 0
    #             except Exception:
    #                 pass

    #             with torch.no_grad():
    #                 _ = runner.test_handavatar(gs_model_list, gs_densify_list, query_points,
    #                                            map_batch, map_idx, iteration, pbar=None)
    #         print(f"\033[92m[✓] Completed HandAvatar pose animation\033[0m")

    #     for idx, test_batch in enumerate(test_dataloader):
    #         with tqdm(
    #             total=1,
    #             desc=f"Iter {iteration} | Sample {idx+1}/{len(test_dataloader)}",
    #             leave=True,
    #             position=idx + 1,
    #             dynamic_ncols=True
    #         ) as pbar:

    #             with torch.no_grad():
    #                 if 'text-to-avatar' in cli_input_dir:
    #                     test_batch[0]['dataset_id'] = 0
    #                 loss_dict = runner.test_handavatar(gs_model_list, gs_densify_list, query_points,
    #                                                     test_batch, idx, iteration, pbar)

    #             psnr_full  += loss_dict['psnr'].item()
    #             ssim_full  += loss_dict['ssim'].item()
    #             lpips_full += loss_dict['lpips'].item()
    #             count += 1

    #             pbar.set_postfix({
    #                 "lpips": f"{lpips_full/count:.4f} ({loss_dict['lpips'].item():.4f})",
    #                 "ssim":  f"{ssim_full/count:.4f} ({loss_dict['ssim'].item():.4f})",
    #                 "psnr":  f"{psnr_full/count:.4f} ({loss_dict['psnr'].item():.4f})",
    #             })

    #             pbar.update(1)
    #             torch.cuda.empty_cache()

    #     print(f"\033[91m[Summary @iter {iteration}]\033[0m "
    #         f"PSNR: {psnr_full/count:.4f}, "
    #         f"SSIM: {ssim_full/count:.4f}, "
    #         f"LPIPS: {lpips_full/count:.4f}, "
    #         f"L1: {image_full/count:.4f}\n")

    #     current_metrics = {
    #         'image_name': image_name,
    #         'lpips': lpips_full / count,
    #         'ssim': ssim_full / count,
    #         'psnr': psnr_full / count,
    #         'image_l1': image_full / count,
    #         'iteration': int(iteration),
    #         'timestamp': time.time()
    #     }

    #     metrics_path = os.path.join(image_checkpoint_dir, f'metrics_iter{iteration}.json')
    #     with open(metrics_path, 'w') as f:
    #         json.dump(current_metrics, f, indent=2)
    #     print(f"\033[92m[✓] Saved metrics to {metrics_path}\033[0m")





if __name__ == "__main__":
    # main()
    app.run(main)
