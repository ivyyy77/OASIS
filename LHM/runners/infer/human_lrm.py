import argparse
import os
import time
import random
import math
from collections import defaultdict
import torch.nn as nn
from pathlib import Path

from typing import Optional
import cv2
import numpy as np
import torch
import logging
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from tools_utils import libcore
from LHM.losses import *
from sklearn.cluster import KMeans

from data.interhand.train import Renderer_mesh

from engine.SegmentAPI.base import Bbox
from scene.mlp_delta_pose import BodyPoseRefiner

from LHM.datasets.cam_utils import (
    build_camera_principle,
    build_camera_standard,
    compose_extrinsic_RT,
    create_intrinsics,
    surrounding_views_linspace,
    center_looking_at_camera_pose,
)
from LHM.models.rendering.gs_renderer import Camera as GSCamera
from LHM.models.modeling_hand_lrm import ModelHandLRM
from LHM.runners import REGISTRY_RUNNERS
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    prepare_sign_motion_seqs,
    resize_image_keepaspect_np,
)
from LHM.utils.download_utils import download_extract_tar_from_url
from PIL import Image

from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.logging import configure_logger
from LHM.utils.model_card import MODEL_CARD, MODEL_CONFIG


def prior_check():
    if os.environ.get("LHM_SKIP_PRIOR_CHECK", "0") == "1":
        print("[prior_check] Skipped because LHM_SKIP_PRIOR_CHECK=1")
        return
    if not os.path.exists('./pretrained_models'):
        prior_data = MODEL_CARD['prior_model']
        download_extract_tar_from_url(prior_data)

from .base_inferrer_hand import Inferrer_hand

logger = logging.getLogger(__name__)


def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device

def resize_with_padding(img, target_size, padding_color=(255, 255, 255)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    dw = target_w - new_w
    dh = target_h - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return padded

def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)

    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box

def get_bbox_hand(mask):
    height, width = mask.shape[0], mask.shape[1]
    pha = mask
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    _h, _w, _ = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)

    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box

def query_model_name(model_name):
    """Query and resolve model name/path. Simply returns the model_name as-is."""
    return model_name

def query_model_config(model_name):
    try:
        model_params = model_name.split('-')[1]

        return MODEL_CONFIG[model_params]
    except:
        return None

def infer_preprocess_image(
    rgb_path,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]

    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard

    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        offset_h = int(target_h - h)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )
    return rgb, mask, intr

def preprocess_hand_image(
    rgb,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    bbox = get_bbox(mask)

    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]

    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard

    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        offset_h = int(target_h - h)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )
    return rgb

def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    if "export_mesh" not in cli_cfg:
        cli_cfg.export_mesh = None
    if "export_video" not in cli_cfg:
        cli_cfg.export_video= None

    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    model_name = os.environ.get("APP_MODEL_NAME") or cli_cfg.get("model_name", "LHM-1B")
    model_name = query_model_name(model_name)
    cli_cfg.model_name = model_name

    model_config = query_model_config(model_name)
    cfg_train = None
    cfg.source_size = 1024
    cfg.src_head_size = 112
    cfg.render_size = 512

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)
        cfg.mesh_dump = os.path.join("exps", "meshs", _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train

@REGISTRY_RUNNERS.register("infer.hand_lrm")
class HandLRMInferrer(Inferrer_hand):
    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"

    def __init__(self, checkpoint_path: str = None, checkpoint_file: str = None, output_path: str = None, resume: bool = True):
        self.uv_bias_regularization_weight = 0.01
        super().__init__()

        self.cfg, cfg_train = parse_configs()

        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )

        prior_check()

        self.hand_model = ModelHandLRM().to(
            device=avaliable_device(),
            dtype=torch.float32,
        )

        ckpt = checkpoint_path if checkpoint_path is not None else getattr(self, 'cli_checkpoint_path', None)
        explicit_ckpt_file = checkpoint_file if checkpoint_file is not None else getattr(self, 'cli_checkpoint_file', None)
        if ckpt is None:
            ckpt = getattr(self.hand_model, 'checkpoint_path', None) or getattr(self.cfg, 'checkpoint_path', None)
        self.checkpoint_path = ckpt
        if self.checkpoint_path is None:
            self.checkpoint_path = os.path.join('./checkpoint/exp-mix3-1-school')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        outp = output_path if output_path is not None else getattr(self, 'cli_output_path', None)
        if outp is None:
            outp = getattr(self.cfg, 'save_tmp_dump', None) or getattr(self.cfg, 'image_dump', None)
        self.test_path = outp or os.path.join('./output', 'exp')
        os.makedirs(self.test_path, exist_ok=True)

        self.hand_model.checkpoint_path = self.checkpoint_path

        resume_flag = resume if resume is not None else getattr(self, 'cli_resume', True)
        if resume_flag:
            try:
                iter_loaded = self.load_checkpoint(is_latest=True, checkpoint_path=self.checkpoint_path, checkpoint_file=explicit_ckpt_file)
                if iter_loaded is not None:
                    print(f"\033[92m[Resumed model from checkpoint at iteration {iter_loaded}]\033[0m")
            except Exception as e:
                print(f"\033[93m[Warning] failed to load checkpoint: {e}\033[0m")

        self.accum_step = 4

        self.renderer_mesh = Renderer_mesh()

        bg_color = [0, 0, 0]
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.motion_dict = dict()
        self.ball_loss = Heuristic_ASAP_Loss()
        self.lpips_loss = LPIPSLoss(device='cuda')
        self.pixel_loss = PixelLoss()
        self.tv_loss = TVLoss()
        self.offset_loss = ACAP_Loss()

        self.prev_shs = None
        self.prev_scaling = None
        self.color_stability_weight = 1.0
        self.vis_mask_threshold = 0.5

        self.warmup = WarmupScheduler(self.hand_model.pcl_embed, use_iteration=True, warmup_iters=10000)
        self.annealer = AnnealingScheduler(start=1.0, end=0.1, total_iters=20000, mode='cosine')

    def save_checkpoint(self, iteration, is_latest):
        self.hand_model.save_checkpoint(iteration, is_latest)

    def _build_model(self, cfg):
        return ModelHandLRM()

    def _default_source_camera(
            self,
            dist_to_center: float = 2.0,
            batch_size: int = 1,
            device: torch.device = torch.device("cpu"),
    ):
        canonical_camera_extrinsics = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, -dist_to_center],
                    [0, 1, 0, 0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(
            canonical_camera_extrinsics, canonical_camera_intrinsics
        )
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(
            self,
            n_views: int,
            batch_size: int = 1,
            device: torch.device = torch.device("cpu"),
    ):
        render_camera_extrinsics = surrounding_views_linspace(
            n_views=n_views, device=device
        )
        render_camera_intrinsics = (
            create_intrinsics(
                f=0.75,
                c=0.5,
                device=device,
            )
            .unsqueeze(0)
            .repeat(render_camera_extrinsics.shape[0], 1, 1)
        )
        render_cameras = build_camera_standard(
            render_camera_extrinsics, render_camera_intrinsics
        )
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_single(
            self,
            image_path: str,
            motion_seqs,
            motion_img_dir,
            motion_video_read_fps,
            export_video: bool,
            export_mesh: bool,
            dump_tmp_dir: str,
            dump_image_dir: str,
            dump_video_path: str,
            shape_param=None,
            render_model=None,
            batch=None,
            image_paths=None,
            mask_paths=None,

            optimizer=None, scheduler=None, scaler=None
    ):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = self.cfg.render_fps
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)

        mask_path = image_path.replace('images', 'masks')
        parsing_mask = np.array(Image.open(mask_path))

        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )

        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        motion_name = 'sign'

        if motion_name in self.motion_dict:
            motion_seq = self.motion_dict[motion_name]
            camera_size = motion_seq["render_c2ws"].size(1)

        motion_seq = prepare_sign_motion_seqs(
            motion_seqs,
            motion_img_dir,
            save_root=dump_tmp_dir,
            fps=motion_video_read_fps,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
        )
        self.motion_dict[motion_name] = motion_seq
        camera_size = motion_seq["render_c2ws"].size(1)

        device = "cuda"
        dtype = torch.float32

        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.hand_model.to(dtype)
        smplx_params = motion_seq['smplx_params']

        gs_model_list, query_points, transform_mat_neutral_pose = self.hand_model.infer_single_view(

            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),

            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        batch_list = []
        batch_size = 40
        for batch_i in range(0, camera_size, batch_size):
            keys = [
                "root_pose",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "lhand_pose",
                "rhand_pose",
                "trans",
                "focal",
                "princpt",
                "img_size_wh",
                "expr",
            ]

            batch_smplx_params = dict()

            batch_smplx_params["betas"] = smplx_params['betas'].to('cuda')
            batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
            for key in keys:
                batch_smplx_params[key] = motion_seq["smplx_params"][key][
                                          :, batch_i: batch_i + batch_size
                                          ].to(device)

            res = self.hand_model.animation_infer(gs_model_list, query_points, batch_smplx_params,
                                             render_c2ws=motion_seq["render_c2ws"][
                                                         :, batch_i: batch_i + batch_size
                                                         ].to(device),
                                             render_intrs=motion_seq["render_intrs"][
                                                          :, batch_i: batch_i + batch_size
                                                          ].to(device),
                                             render_bg_colors=motion_seq["render_bg_colors"][
                                                              :, batch_i: batch_i + batch_size
                                                              ].to(device),
                                             model=render_model,
                                             gt_img=image_paths[
                                                    batch_i: batch_i + batch_size, :
                                                    ].to(device),
                                             gt_msk=mask_paths[
                                                    batch_i: batch_i + batch_size, :
                                                    ].to(device),
                                             optimizer=optimizer, scheduler=scheduler, scaler=scaler
                                             )

            comp_rgb = res["comp_rgb"]
            comp_mask = res["comp_mask"]

            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)

            del res
            torch.cuda.empty_cache()

        rgb = np.concatenate(batch_list, axis=0)

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        print(f"save video to {dump_video_path}")

        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=render_fps,
        )

    def train_single(
            self,
            image_path: str,
            motion_seqs,
            motion_img_dir,
            motion_video_read_fps,
            export_video: bool,
            export_mesh: bool,
            dump_tmp_dir: str,
            dump_image_dir: str,
            dump_video_path: str,
            shape_param=None,
            render_model=None,
            batch=None,
            image_paths=None,
            mask_paths=None,

            optimizer=None, scheduler=None, scaler=None
    ):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size

        render_fps = self.cfg.render_fps
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)

        mask_path = image_path.replace('images', 'masks')
        parsing_mask = np.array(Image.open(mask_path))

        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )

        try:
            src_head_rgb = self.crop_face_image(image_path)
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)

        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )

        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        motion_seq = prepare_sign_motion_seqs(
            motion_seqs,
            motion_img_dir,
            save_root=dump_tmp_dir,
            fps=motion_video_read_fps,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
        )
        motion_name = 'sign'
        self.motion_dict[motion_name] = motion_seq
        camera_size = motion_seq["render_c2ws"].size(1)

        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.hand_model.to(dtype)
        smplx_params = motion_seq['smplx_params']

        gs_model_list, query_points, transform_mat_neutral_pose = self.hand_model.infer_single_view(

            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),

            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        num_samples = 4
        total_frames = motion_seq["smplx_params"]["root_pose"].shape[1]
        sampled_indices = sorted(random.sample(range(total_frames), num_samples))

        keys = ["root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                "lhand_pose", "rhand_pose", "trans", "focal", "princpt", "img_size_wh", "expr"]

        batch_smplx_params = {
            "betas": smplx_params['betas'].to(device),
            "transform_mat_neutral_pose": transform_mat_neutral_pose,
        }

        for key in keys:
            batch_smplx_params[key] = motion_seq["smplx_params"][key][:, sampled_indices].to(device)

        if any(idx >= total_frames or idx < 0 for idx in sampled_indices):
            print(f"索引越界：sampled_indices 中最大为 {max(sampled_indices)}，"
                  f"但总帧数只有 {total_frames} 帧。请重新采样。")
        else:
            res = self.hand_model.finetune_train(gs_model_list, query_points, batch_smplx_params,
                                            render_c2ws=motion_seq["render_c2ws"][:, sampled_indices].to(device),
                                            render_intrs=motion_seq["render_intrs"][:, sampled_indices].to(device),
                                            render_bg_colors=motion_seq["render_bg_colors"][:, sampled_indices].to(
                                                device),
                                            model=render_model,
                                            gt_img=image_paths[sampled_indices, :].to(device),
                                            gt_msk=mask_paths[sampled_indices, :].to(device),
                                            optimizer=optimizer, scheduler=scheduler, scaler=scaler)

    def train_hand(
            self,
            image_path: str,
            motion_seqs,
            motion_img_dir,
            motion_video_read_fps,
            dump_tmp_dir: str,

            shape_param=None,
            render_model=None,
            batch=None,
            image_paths=None,
            mask_paths=None,

            optimizer=None, scheduler=None, scaler=None
    ):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size

        render_fps = self.cfg.render_fps
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)

        image = batch[0].original_image
        mask = batch[0].bkgd_mask

        image = preprocess_hand_image(
            image,
            mask=mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=5.0 / 3,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=256,
            multiply=14,
            need_mask=True,
        )

        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        device = "cuda"
        dtype = torch.float32

        self.hand_model.to(dtype)
        smplx_params = batch[0].smpl_params

        gs_model_list, query_points, transform_mat_neutral_pose = self.hand_model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            render_c2ws=batch[0].extrinsic.to(device),
            render_intrs=batch[0].K.to(device),
            render_bg_colors=batch[0].bg_color.to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        num_samples = 4
        total_frames = motion_seq["smplx_params"]["root_pose"].shape[1]
        sampled_indices = sorted(random.sample(range(total_frames), num_samples))

        keys = ["root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                "lhand_pose", "rhand_pose", "trans", "focal", "princpt", "img_size_wh", "expr"]

        batch_smplx_params = {
            "betas": smplx_params['betas'].to(device),
            "transform_mat_neutral_pose": transform_mat_neutral_pose,
        }

        for key in keys:
            batch_smplx_params[key] = motion_seq["smplx_params"][key][:, sampled_indices].to(device)

        if any(idx >= total_frames or idx < 0 for idx in sampled_indices):
            print(f"索引越界：sampled_indices 中最大为 {max(sampled_indices)}，"
                  f"但总帧数只有 {total_frames} 帧。请重新采样。")
        else:
            res = self.hand_model.finetune_train(gs_model_list, query_points, batch_smplx_params,
                                            render_c2ws=motion_seq["render_c2ws"][:, sampled_indices].to(device),
                                            render_intrs=motion_seq["render_intrs"][:, sampled_indices].to(device),
                                            render_bg_colors=motion_seq["render_bg_colors"][:, sampled_indices].to(
                                                device),
                                            model=render_model,
                                            gt_img=image_paths[sampled_indices, :].to(device),
                                            gt_msk=mask_paths[sampled_indices, :].to(device),
                                            optimizer=optimizer, scheduler=scheduler, scaler=scaler)

    def infer_ori(self, batch=None, scaler=None, iter=None, writer=None):
        all_gt = []
        all_render = []
        all_mask_gt = []
        all_mask_render = []
        loss_dict = {}
        loss_dict['image_l1'], loss_dict['lpips'], loss_dict['mask_l2'],\
        loss_dict['ssim'], loss_dict['offset'], loss_dict['scaling'],\
        loss_dict['color_stability'] = 0, 0, 0, 0, 0, 0, 0

        for i in range(len(batch)):
            infer_image = batch[i]['original_image'][0, ...].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch[i]['bkgd_mask'][0,...].unsqueeze(0).to('cuda')
            image = infer_image * bkgd_mask + self.bg_color * (1 - bkgd_mask)
            smplx_params = batch[i]['smpl_param']

            cano_pts = batch[i]['big_pose_world_vertex'][0].unsqueeze(0)

            render_pkg = self.hand_model.infer_single_view(
                image,
                batch[i],
                cano_pts=cano_pts,
                render_bg_colors=self.bg_color,
                smplx_params={
                    k: v.to('cuda') for k, v in smplx_params.items()
                },
            )

            gt_image_bs = batch[i]['original_image'].permute(0, 2, 3, 1).to('cuda')
            bkgd_mask_bs = batch[i]['bkgd_mask'].to('cuda')
            render_image_bs = render_pkg['comp_rgb']
            render_mask_bs = render_pkg['comp_mask'].repeat(1,1,1,3)

            all_render.append(render_image_bs)
            all_gt.append(gt_image_bs)
            all_mask_gt.append(bkgd_mask_bs)
            all_mask_render.append(render_mask_bs)

            loss_dict['ssim'] += (1 - ssim(render_image_bs, gt_image_bs))
            loss_dict['scaling'] += 0.5 * self.ball_loss(render_pkg['3dgs'][0][0][0].scaling)
            loss_dict['offset'] += self.offset_loss(render_pkg['offset'][0])

        all_render = torch.stack(all_render, dim=0)
        all_gt = torch.stack(all_gt, dim=0)
        all_mask_render = torch.stack(all_mask_render, dim=0)
        all_mask_gt = torch.stack(all_mask_gt, dim=0)

        loss_dict['mask_l2'] += self.pixel_loss(all_mask_render.float().permute(0, 1, 4, 2, 3),
                        all_mask_gt.float().permute(0, 1, 4, 2, 3))
        loss_dict['lpips'] += self.lpips_loss(all_render.float().permute(0, 1, 4, 2, 3),
                        all_gt.permute(0, 1, 4, 2, 3))
        loss_dict['image_l1'] += self.pixel_loss(all_render.permute(0, 1, 4, 2, 3),
                                                 all_gt.permute(0, 1, 4, 2, 3))

        writer.add_scalar('train/loss_image_l1', loss_dict['image_l1'], iter)
        writer.add_scalar('train/loss_mask_l2', loss_dict['mask_l2'], iter)
        writer.add_scalar('train/loss_lpips', loss_dict['lpips'], iter)
        writer.add_scalar('train/loss_ssim', loss_dict['ssim'], iter)

        total_loss = sum(loss_dict.values())

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
            has_grad = any(p.grad is not None for g in self.hand_model.optimizer.param_groups for p in g['params'])
            if has_grad:
                scaler.step(self.hand_model.optimizer)
                if getattr(self.hand_model, "scheduler", None) is not None:
                    self.hand_model.scheduler.step()
                scaler.update()
                self.hand_model.optimizer.zero_grad()
        else:
            total_loss.backward(retain_graph=True)
            has_grad = any(p.grad is not None for g in self.hand_model.optimizer.param_groups for p in g['params'])
            if has_grad:
                self.hand_model.optimizer.step()
                if getattr(self.hand_model, "scheduler", None) is not None:
                    self.hand_model.scheduler.step()
                self.hand_model.optimizer.zero_grad()

    def vis_pts(self, batch, verts_cam, nail_image, return_mask: bool = False, threshold: float = 0.5):
        device = next(self.hand_model.parameters()).device
        device = 'cuda'
        E = torch.eye(4, device=device)

        R = torch.as_tensor(batch['R'][0, ...], device=device, dtype=torch.float32)
        T = torch.as_tensor(batch['T'][0, ...], device=device, dtype=torch.float32)
        E[:3, :3] = R
        E[:3, 3] = T

        world_vertex = torch.as_tensor(batch['world_vertex'][0, ...], device=device, dtype=torch.float32)
        K = torch.as_tensor(batch['K'][0, ...], device=device, dtype=torch.float32)
        depth_map = self.renderer_mesh(world_vertex, K, E, self.hand_model.renderer.mano_model.mano.faces)

        if depth_map.dim() == 3 and depth_map.size(0) == 1:
            depth_map = depth_map.squeeze(0)
        depth_map = depth_map.to(device).to(torch.float32)

        depth_map = torch.where(torch.isnan(depth_map), torch.tensor(float('inf'), device=device), depth_map)

        z = torch.as_tensor(verts_cam.reshape(-1), device=device, dtype=torch.float32)

        mask_inside = z > 0

        x = torch.as_tensor(nail_image[..., 0], device=device)
        y = torch.as_tensor(nail_image[..., 1], device=device)

        H, W = depth_map.shape[-2], depth_map.shape[-1]

        x_valid = torch.clamp(x[mask_inside].long(), 0, W - 1)
        y_valid = torch.clamp(y[mask_inside].long(), 0, H - 1)

        if x_valid.numel() == 0:
            p_vis = torch.zeros(z.shape[0], dtype=torch.float32, device=device)
            if return_mask:
                vis_mask = torch.zeros_like(p_vis)
                return p_vis, vis_mask
            return p_vis

        z_buf = depth_map[y_valid, x_valid]
        z_in = z[mask_inside]

        tau = 5e-2
        delta_z = z_buf - z_in
        p_vis_local = 1.0 / (1.0 + torch.exp(-delta_z / tau))

        p_vis = torch.zeros(z.shape[0], dtype=torch.float32, device=device)

        if p_vis_local.dtype != p_vis.dtype:
            p_vis_local = p_vis_local.to(p_vis.dtype)
        p_vis[mask_inside] = p_vis_local

        if return_mask:
            visible_local = z_in <= (z_buf + 5e-3)
            visible_mask = torch.zeros(world_vertex.shape[1], dtype=torch.bool, device=device)
            visible_mask[mask_inside] = visible_local
            return p_vis, visible_mask

        return p_vis

    def vis_pts_with_face_centers(self, batch, verts_cam, nail_image, verts_world, return_mask: bool = False, threshold: float = 0.5):
        """
        Compute visibility for both vertices and face-center points.

        Face-center points are defined using barycentric coordinates (1/3, 1/3, 1/3).

        Args:
            batch: batch data containing camera parameters
            verts_cam: [1, N_verts, 3] vertices in camera coordinates
            nail_image: [N_verts, 2] projected pixel coordinates of vertices
            verts_world: [N_verts, 3] vertices in world coordinates
            return_mask: whether to return hard visibility mask
            threshold: threshold for hard visibility mask

        Returns:
            p_vis: [B, N_verts + N_faces] visibility probabilities
            (optional) vis_mask: hard visibility mask
        """
        device = next(self.hand_model.parameters()).device
        device = 'cuda'

        if verts_world.dim() == 3:
            verts_world = verts_world[0]
        if verts_cam.dim() == 3 and verts_cam.shape[0] == 1:
            verts_cam = verts_cam[0]
        if nail_image.dim() == 3 and nail_image.shape[0] == 1:
            nail_image = nail_image[0]

        faces = self.hand_model.renderer.mano_model.mano.faces
        N_verts = verts_world.shape[0]
        N_faces = faces.shape[0]

        E = torch.eye(4, device=device)
        R = torch.as_tensor(batch['R'][0, ...], device=device, dtype=torch.float32)
        T = torch.as_tensor(batch['T'][0, ...], device=device, dtype=torch.float32)
        E[:3, :3] = R
        E[:3, 3] = T

        world_vertex = torch.as_tensor(batch['world_vertex'][0, ...], device=device, dtype=torch.float32)
        K = torch.as_tensor(batch['K'][0, ...], device=device, dtype=torch.float32)
        depth_map = self.renderer_mesh(world_vertex, K, E, faces)

        if depth_map.dim() == 3 and depth_map.size(0) == 1:
            depth_map = depth_map.squeeze(0)
        depth_map = depth_map.to(device).to(torch.float32)
        depth_map = torch.where(torch.isnan(depth_map), torch.tensor(float('inf'), device=device), depth_map)

        H, W = depth_map.shape[-2], depth_map.shape[-1]

        z_verts = torch.as_tensor(verts_cam[..., 2].reshape(-1), device=device, dtype=torch.float32)
        mask_inside_verts = z_verts > 0

        x_verts = torch.as_tensor(nail_image[..., 0], device=device)
        y_verts = torch.as_tensor(nail_image[..., 1], device=device)

        x_valid_verts = torch.clamp(x_verts[mask_inside_verts].long(), 0, W - 1)
        y_valid_verts = torch.clamp(y_verts[mask_inside_verts].long(), 0, H - 1)

        p_vis_verts = torch.zeros(N_verts, dtype=torch.float32, device=device)

        if x_valid_verts.numel() > 0:
            z_buf_verts = depth_map[y_valid_verts, x_valid_verts]
            z_in_verts = z_verts[mask_inside_verts]
            tau = 5e-2
            delta_z_verts = z_buf_verts - z_in_verts
            p_vis_local_verts = 1.0 / (1.0 + torch.exp(-delta_z_verts / tau))
            p_vis_verts[mask_inside_verts] = p_vis_local_verts

        faces_tensor = torch.as_tensor(faces, device=device, dtype=torch.long)
        verts_cam_verts = torch.as_tensor(verts_cam, device=device, dtype=torch.float32)
        v0 = verts_cam_verts[faces_tensor[:, 0]]
        v1 = verts_cam_verts[faces_tensor[:, 1]]
        v2 = verts_cam_verts[faces_tensor[:, 2]]

        face_centers_cam = (v0 + v1 + v2) / 3.0

        ones = torch.ones(N_faces, 1, device=device, dtype=torch.float32)
        face_centers_cam_h = torch.cat([face_centers_cam, ones], dim=1)

        face_centers_cam_3d = face_centers_cam[:, :3]

        proj_2d = (K @ face_centers_cam_3d.T).T

        x_faces = proj_2d[:, 0] / proj_2d[:, 2]
        y_faces = proj_2d[:, 1] / proj_2d[:, 2]

        z_faces = face_centers_cam[:, 2]
        mask_inside_faces = z_faces > 0

        x_valid_faces = torch.clamp(x_faces[mask_inside_faces].long(), 0, W - 1)
        y_valid_faces = torch.clamp(y_faces[mask_inside_faces].long(), 0, H - 1)

        p_vis_faces = torch.zeros(N_faces, dtype=torch.float32, device=device)

        if x_valid_faces.numel() > 0:
            z_buf_faces = depth_map[y_valid_faces, x_valid_faces]
            z_in_faces = z_faces[mask_inside_faces]
            tau = 5e-2
            delta_z_faces = z_buf_faces - z_in_faces
            p_vis_local_faces = 1.0 / (1.0 + torch.exp(-delta_z_faces / tau))
            p_vis_faces[mask_inside_faces] = p_vis_local_faces

        p_vis = torch.cat([p_vis_verts, p_vis_faces], dim=0)

        if return_mask:
            visible_mask_verts = torch.zeros(N_verts, dtype=torch.bool, device=device)
            if x_valid_verts.numel() > 0:
                visible_local_verts = z_verts[mask_inside_verts] <= (z_buf_verts + 5e-3)
                visible_mask_verts[mask_inside_verts] = visible_local_verts

            visible_mask_faces = torch.zeros(N_faces, dtype=torch.bool, device=device)
            if x_valid_faces.numel() > 0:
                visible_local_faces = z_faces[mask_inside_faces] <= (z_buf_faces + 5e-3)
                visible_mask_faces[mask_inside_faces] = visible_local_faces

            visible_mask = torch.cat([visible_mask_verts, visible_mask_faces], dim=0)
            return p_vis, visible_mask

        return p_vis

    def get_boundary_points(self, vis_prob: torch.Tensor, threshold_low: float = 5e-3, threshold_high: float = 5e-2):
        """
        Identify boundary points based on visibility probability.

        Args:
            vis_prob: [B, N] tensor of visibility probabilities
            threshold_low: Lower threshold for boundary region (default 5e-3)
            threshold_high: Upper threshold for boundary region (default 5e-2)

        Returns:
            boundary_mask: [B, N] bool tensor marking boundary points
            visible_mask: [B, N] bool tensor marking highly visible points (vis_prob > threshold_high)
            invisible_mask: [B, N] bool tensor marking highly invisible points (vis_prob < threshold_low)
        """

        boundary_mask = (vis_prob >= threshold_low) & (vis_prob <= threshold_high)

        visible_mask = vis_prob > threshold_high

        invisible_mask = vis_prob < threshold_low

        return boundary_mask, visible_mask, invisible_mask

    def infer(self, batch=None, scaler=None, iter=None, writer=None, pbar=None):
        gt_image_bs = []

        gt_mask_bs = []

        infer_images = []
        nail_images = []
        verts_cams = []

        nail_masks = []
        vis_masks = []
        vis_prob = []
        cano_pts_list = []
        ref_view = 0

        loss_dict = {}
        loss_dict['image_l1'], loss_dict['lpips'], loss_dict['mask_l2'],\
        loss_dict['ssim'], loss_dict['offset'], loss_dict['scaling'],\
        loss_dict['color_stability'] = 0, 0, 0, 0, 0, 0, 0

        for i in range(len(batch)):
            batch_i = batch[i]
            gt_image = batch_i['original_image'].permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch_i['bkgd_mask'].to('cuda')
            gt_image_bs.append(gt_image)
            gt_mask_bs.append(bkgd_mask)

            infer_image = batch_i['original_image'][ref_view, ...].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch_i['bkgd_mask'][ref_view, ...].unsqueeze(0).to('cuda')
            infer_images.append(infer_image)
            nail_images.append(batch_i['nail_image'][ref_view, ...].unsqueeze(0).to('cuda'))
            nail_masks.append(batch_i['nail_mask'][ref_view, ...].unsqueeze(0).to('cuda'))
            verts_cams.append(batch_i['verts_cam'][ref_view, ...].unsqueeze(0).to('cuda'))

            p_vis, vis_mask = self.vis_pts(
                batch_i,
                batch_i['verts_cam'][ref_view, ..., 2],
                batch_i['nail_image'][ref_view, ...],
                return_mask=True,
                threshold=0.5,
            )
            vis_masks.append(vis_mask)
            vis_prob.append(p_vis.float())
            cano_pts_list.append(batch_i['big_pose_world_vertex'][ref_view].to('cuda'))

        gt_image_bs = torch.stack(gt_image_bs, dim=0)
        gt_mask_bs = torch.stack(gt_mask_bs, dim=0)

        nail_pts = batch[0]['nail_image'][ref_view, ...].shape[0]

        image = torch.stack(infer_images, dim=0)

        bs, n, h, w, c = gt_image_bs.shape

        nail_images = torch.stack(nail_images, dim=0).view(bs, nail_pts, -1)
        nail_masks = torch.stack(nail_masks, dim=0).squeeze(-1)
        verts_cams = torch.stack(verts_cams, dim=0).view(bs, nail_pts, -1)
        vis_masks = torch.stack(vis_masks, dim=0)
        vis_prob = torch.stack(vis_prob, dim=0)

        all_gt = gt_image_bs.reshape(bs*n, h, w, c)

        cano_pts = torch.stack(cano_pts_list, dim=0)

        render_pkg = self.hand_model.infer_single_view(
            vis_prob,
            vis_masks,
            nail_images,
            nail_masks,
            verts_cams,
            image.reshape(bs, h, w, c),
            batch,
            cano_pts=cano_pts,
            render_bg_colors=self.bg_color,
        )
        gau_num = render_pkg['scaling'].shape[1]
        all_render = render_pkg['comp_rgb']
        render_image_bs = all_render.reshape(bs, n, h, w, c)

        all_render_mask = render_pkg['comp_mask'].repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)

        loss_dict['ssim'] += (1 - ssim(all_render, all_gt))
        loss_dict['scaling'] += 0.5 * self.ball_loss(render_pkg['scaling'])
        loss_dict['offset'] += self.offset_loss(render_pkg['offset'])

        loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
        loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                        gt_image_bs.permute(0, 1, 4, 2, 3))
        loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs[:, 0, ...].unsqueeze(1).permute(0, 1, 4, 2, 3),
                        gt_image_bs[:, 0, ...].unsqueeze(1).permute(0, 1, 4, 2, 3))

        loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                                 gt_image_bs.permute(0, 1, 4, 2, 3))
        loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs[:, 0, ...].unsqueeze(1).permute(0, 1, 4, 2, 3),
                                                 gt_image_bs[:, 0, ...].unsqueeze(1).permute(0, 1, 4, 2, 3))

        writer.add_scalar('train/loss_image_l1', loss_dict['image_l1'], iter)
        writer.add_scalar('train/loss_mask_l2', loss_dict['mask_l2'], iter)
        writer.add_scalar('train/loss_lpips', loss_dict['lpips'], iter)
        writer.add_scalar('train/loss_ssim', loss_dict['ssim'], iter)
        writer.add_scalar('train/loss_color_stability', loss_dict['color_stability'], iter)
        pbar.set_postfix({
            "\033[91m#Gau\033[0m": gau_num,
            "L1_loss": f"{loss_dict['image_l1']:.4f}",
            "mask_l2": f"{loss_dict['mask_l2']:.4f}",
            'lpips': f"{loss_dict['lpips']:.4f}",
            'ssim': f"{loss_dict['ssim']:.4f}",
        })

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
            has_grad = any(p.grad is not None for g in self.hand_model.optimizer.param_groups for p in g['params'])
            if has_grad and (iter + 1) % self.accum_step == 0:
                scaler.step(self.hand_model.optimizer)
                if getattr(self.hand_model, "scheduler", None) is not None:
                    self.hand_model.scheduler.step()
                scaler.update()
                self.hand_model.optimizer.zero_grad()
        else:
            total_loss.backward(retain_graph=True)
            has_grad = any(p.grad is not None for g in self.hand_model.optimizer.param_groups for p in g['params'])
            if has_grad and (iter + 1) % self.accum_step == 0:
                self.hand_model.optimizer.step()
                if getattr(self.hand_model, "scheduler", None) is not None:
                    self.hand_model.scheduler.step()
                self.hand_model.optimizer.zero_grad()

    def infer_edit(self, batch=None, scaler=None, iter=None, writer=None, pbar=None):
        gt_image_bs = []

        gt_mask_bs = []
        bound_mask_bs = []

        infer_images = []
        nail_images = []
        verts_cams = []

        nail_masks = []
        vis_masks = []
        vis_prob = []
        cano_pts_list = []
        ref_view = 0

        loss_dict = {}
        loss_dict['image_l1'], loss_dict['lpips'], loss_dict['mask_l2'],\
        loss_dict['ssim'], loss_dict['offset'], loss_dict['scaling'], loss_dict['color_stability'],\
        loss_dict['uv_bias_reg'], loss_dict['color_bias_reg'], loss_dict['opacity_bias_reg'] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        for i in range(len(batch)):
            batch_i = batch[i]
            gt_image = batch_i['original_image'].permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch_i['bkgd_mask'].to('cuda')
            bound_mask = batch_i['bound_mask'].to('cuda')
            gt_image_bs.append(gt_image)
            gt_mask_bs.append(bkgd_mask)
            bound_mask_bs.append(bound_mask)

            infer_image = batch_i['original_image'][ref_view, ...].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')

            bkgd_mask = batch_i['bkgd_mask'][ref_view, ...].unsqueeze(0).to('cuda')
            infer_images.append(infer_image)
            nail_images.append(batch_i['nail_image'][ref_view, ...].unsqueeze(0).to('cuda'))
            nail_masks.append(batch_i['nail_mask'][ref_view, ...].unsqueeze(0).to('cuda'))
            verts_cams.append(batch_i['verts_cam'][ref_view, ...].unsqueeze(0).to('cuda'))

            p_vis, visible_mask = self.vis_pts(
                batch_i,
                batch_i['verts_cam'][ref_view, ..., 2],
                batch_i['nail_image'][ref_view, ...],
                return_mask=True,
            )
            vis_masks.append(visible_mask)
            vis_prob.append(p_vis)
            cano_pts_list.append(batch_i['big_pose_world_vertex'][ref_view].to('cuda'))

        gt_image_bs = torch.stack(gt_image_bs, dim=0)
        gt_mask_bs = torch.stack(gt_mask_bs, dim=0)
        bound_mask_bs = torch.stack(bound_mask_bs, dim=0)

        nail_pts = batch[0]['nail_image'][ref_view, ...].shape[0]

        image = torch.stack(infer_images, dim=0)

        bs, n, h, w, c = gt_image_bs.shape

        nail_images = torch.stack(nail_images, dim=0).view(bs, nail_pts, -1)
        nail_masks = torch.stack(nail_masks, dim=0).squeeze(-1)
        verts_cams = torch.stack(verts_cams, dim=0).view(bs, nail_pts, -1)
        vis_masks = torch.stack(vis_masks, dim=0)
        vis_prob = torch.stack(vis_prob, dim=0)

        all_gt = gt_image_bs.reshape(bs*n, h, w, c)

        cano_pts = torch.stack(cano_pts_list, dim=0)

        render_pkg = self.hand_model.infer_single_view(
            vis_prob,
            vis_masks,
            nail_images,
            nail_masks,
            verts_cams,
            image.reshape(bs, h, w, c),
            batch,
            cano_pts=cano_pts,
            render_bg_colors=self.bg_color,
        )
        gau_num = render_pkg['scaling'].shape[1]
        all_render = render_pkg['comp_rgb']
        render_image_bs = all_render.reshape(bs, n, h, w, c)

        all_render_mask = render_pkg['comp_mask'].repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)

        loss_dict['ssim'] += (1 - ssim(all_render, all_gt))
        loss_dict['scaling'] += 0.5 * self.ball_loss(render_pkg['scaling'])
        loss_dict['offset'] += self.offset_loss(render_pkg['offset'])

        bias = getattr(self.hand_model, "uv_map_bias", None)
        if bias is not None and bias.requires_grad:
            uv_bias_reg = (bias ** 2).mean()
            loss_dict['uv_bias_reg'] += 0.01 * uv_bias_reg

        color_b = getattr(self.hand_model, "color_b_map", None)
        if color_b is not None and color_b.requires_grad:
            color_bias_reg = torch.abs(color_b).mean()
            loss_dict['color_bias_reg'] += 100.0 * color_bias_reg

            if iter % 50 == 0:
                cb_norm = torch.norm(color_b).item()
                cb_grad_norm = torch.norm(color_b.grad).item() if color_b.grad is not None else 0.0
                cb_min, cb_max = color_b.min().item(), color_b.max().item()
                print(f"\033[95m[ColorBias Edit] norm={cb_norm:.6e}, grad_norm={cb_grad_norm:.6e}, reg_loss={color_bias_reg.item():.6e}, range=[{cb_min:.6e}, {cb_max:.6e}]\033[0m")

        opacity_b = getattr(self.hand_model, "opacity_b_map", None)
        if opacity_b is not None and opacity_b.requires_grad:
            opacity_bias_reg = (opacity_b ** 2).mean()
            loss_dict['opacity_bias_reg'] += 1 * opacity_bias_reg

        loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
        loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                gt_image_bs.permute(0, 1, 4, 2, 3))

        masked_render = render_image_bs[:, 0, ...] * bound_mask_bs[:, 0, ...]
        masked_gt = gt_image_bs[:, 0, ...] * bound_mask_bs[:, 0, ...]
        loss_dict['lpips'] += 5 * self.lpips_loss(masked_render.unsqueeze(1).permute(0, 1, 4, 2, 3),
                masked_gt.unsqueeze(1).permute(0, 1, 4, 2, 3))

        loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                             gt_image_bs.permute(0, 1, 4, 2, 3))

        masked_render_l1 = masked_render
        masked_gt_l1 = masked_gt
        loss_dict['image_l1'] += 10 * self.pixel_loss(masked_render_l1.unsqueeze(1).permute(0, 1, 4, 2, 3),
                             masked_gt_l1.unsqueeze(1).permute(0, 1, 4, 2, 3))

        writer.add_scalar('train/loss_image_l1', loss_dict['image_l1'], iter)
        writer.add_scalar('train/loss_lpips', loss_dict['lpips'], iter)
        writer.add_scalar('train/loss_ssim', loss_dict['ssim'], iter)
        writer.add_scalar('train/loss_uv_bias_reg', loss_dict['uv_bias_reg'], iter)
        writer.add_scalar('train/loss_color_bias_reg', loss_dict['color_bias_reg'], iter)
        writer.add_scalar('train/loss_opacity_bias_reg', loss_dict['opacity_bias_reg'], iter)

        pbar.set_postfix({
            "\033[91m#Gau\033[0m": gau_num,
            "L1_loss": f"{loss_dict['image_l1']:.4f}",
            'lpips': f"{loss_dict['lpips']:.4f}",
            'ssim': f"{loss_dict['ssim']:.4f}",
            'uv_bias': f"{loss_dict['uv_bias_reg']:.4f}",
            'color_b': f"{loss_dict['color_bias_reg']:.4f}",
            'opacity_b': f"{loss_dict['opacity_bias_reg']:.4f}"
        })

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter+1) % self.accum_step == 0:
            scaler.step(self.hand_model.optimizer)
            self.hand_model.scheduler.step()
            scaler.update()
            self.hand_model.optimizer.zero_grad()

    def infer_wild(self, batch=None, scaler=None, iter=None, writer=None, pbar=None):
        gt_image_bs = []

        gt_mask_bs = []
        bound_mask_bs = []

        infer_images = []
        nail_images = []
        verts_cams = []

        nail_masks = []
        vis_masks = []
        vis_prob = []
        cano_pts_list = []
        ref_view = 0

        loss_dict = {}
        loss_dict['image_l1'], loss_dict['lpips'], loss_dict['mask_l2'],\
        loss_dict['ssim'], loss_dict['offset'], loss_dict['scaling'], loss_dict['color_stability'],\
        loss_dict['uv_bias_reg'], loss_dict['color_bias_reg'], loss_dict['opacity_bias_reg'] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        for i in range(len(batch)):
            batch_i = batch[i]

            gt_image = batch_i['original_image'].permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch_i['bkgd_mask'].to('cuda')
            bound_mask = batch_i['bound_mask'].to('cuda')
            gt_image_bs.append(gt_image)
            gt_mask_bs.append(bkgd_mask)
            bound_mask_bs.append(bound_mask)

            infer_image = batch_i['original_image'][ref_view, ...].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')

            bkgd_mask = batch_i['bkgd_mask'][ref_view, ...].unsqueeze(0).to('cuda')
            infer_images.append(infer_image)
            nail_images.append(batch_i['nail_image'][ref_view, ...].unsqueeze(0).to('cuda'))
            nail_masks.append(batch_i['nail_mask'][ref_view, ...].unsqueeze(0).to('cuda'))
            verts_cams.append(batch_i['verts_cam'][ref_view, ...].unsqueeze(0).to('cuda'))

            p_vis, visible_mask = self.vis_pts(
                batch_i,
                batch_i['verts_cam'][ref_view, ..., 2],
                batch_i['nail_image'][ref_view, ...],
                return_mask=True,
            )
            vis_masks.append(visible_mask)
            vis_prob.append(p_vis.float())
            cano_pts_list.append(batch_i['big_pose_world_vertex'][ref_view].to('cuda'))

        gt_image_bs = torch.stack(gt_image_bs, dim=0)
        gt_mask_bs = torch.stack(gt_mask_bs, dim=0)
        bound_mask_bs = torch.stack(bound_mask_bs, dim=0)

        nail_pts = batch[0]['nail_image'][ref_view, ...].shape[0]

        image = torch.stack(infer_images, dim=0)

        bs, n, h, w, c = gt_image_bs.shape

        nail_images = torch.stack(nail_images, dim=0).view(bs, nail_pts, -1)
        nail_masks = torch.stack(nail_masks, dim=0).squeeze(-1)
        verts_cams = torch.stack(verts_cams, dim=0).view(bs, nail_pts, -1)
        vis_masks = torch.stack(vis_masks, dim=0)
        vis_prob = torch.stack(vis_prob, dim=0)

        all_gt = gt_image_bs.reshape(bs*n, h, w, c)

        cano_pts = torch.stack(cano_pts_list, dim=0)

        render_pkg = self.hand_model.infer_single_view(
            vis_prob,
            vis_masks,
            nail_images,
            nail_masks,
            verts_cams,
            image.reshape(bs, h, w, c),
            batch,
            cano_pts=cano_pts,
            render_bg_colors=self.bg_color,
        )

        gau_num = render_pkg['scaling'].shape[1]
        all_render = render_pkg['comp_rgb']
        render_image_bs = all_render.reshape(bs, n, h, w, c)

        all_render_mask = render_pkg['comp_mask'].repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)

        loss_dict['ssim'] += (1 - ssim(all_render, all_gt))
        loss_dict['scaling'] += 0.5 * self.ball_loss(render_pkg['scaling'])
        loss_dict['offset'] += self.offset_loss(render_pkg['offset'])

        bias = getattr(self.hand_model, "uv_map_bias", None)
        if bias is not None and bias.requires_grad:
            uv_bias_reg = (bias ** 2).mean()
            loss_dict['uv_bias_reg'] += 0.01 * uv_bias_reg

        color_b = getattr(self.hand_model, "color_b_map", None)
        if color_b is not None and color_b.requires_grad:
            color_bias_reg = torch.abs(color_b).mean()
            loss_dict['color_bias_reg'] += 50.0 * color_bias_reg

        opacity_b = getattr(self.hand_model, "opacity_b_map", None)
        if opacity_b is not None and opacity_b.requires_grad:
            opacity_bias_reg = (opacity_b ** 2).mean()
            loss_dict['opacity_bias_reg'] += 0.5 * opacity_bias_reg
            if iter % 50 == 0:
                ob_norm = torch.norm(opacity_b).item()
                ob_grad_norm = torch.norm(opacity_b.grad).item() if opacity_b.grad is not None else 0.0
                print(f"\033[95m[OpacityBias Edit] norm={ob_norm:.6e}, grad_norm={ob_grad_norm:.6e}, reg_loss={opacity_bias_reg.item():.6e}\033[0m")

        loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
        loss_dict['lpips'] += 10 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                gt_image_bs.permute(0, 1, 4, 2, 3))

        loss_dict['image_l1'] += 20 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                             gt_image_bs.permute(0, 1, 4, 2, 3))

        writer.add_scalar('train/loss_image_l1', loss_dict['image_l1'], iter)
        writer.add_scalar('train/loss_lpips', loss_dict['lpips'], iter)
        writer.add_scalar('train/loss_ssim', loss_dict['ssim'], iter)
        writer.add_scalar('train/loss_uv_bias_reg', loss_dict['uv_bias_reg'], iter)
        writer.add_scalar('train/loss_color_bias_reg', loss_dict['color_bias_reg'], iter)
        writer.add_scalar('train/loss_opacity_bias_reg', loss_dict['opacity_bias_reg'], iter)

        pbar.set_postfix({
            "\033[91m#Gau\033[0m": gau_num,
            "L1_loss": f"{loss_dict['image_l1']:.4f}",
            'lpips': f"{loss_dict['lpips']:.4f}",
            'ssim': f"{loss_dict['ssim']:.4f}",
            'uv_bias': f"{loss_dict['uv_bias_reg']:.4f}",
            'color_b': f"{loss_dict['color_bias_reg']:.4f}",
            'opacity_b': f"{loss_dict['opacity_bias_reg']:.4f}"
        })

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter+1) % self.accum_step == 0:
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

    def _collect_params(self, keywords):
        """Utility: collect parameters whose names contain any keyword."""
        params = []
        for name, p in self.hand_model.named_parameters():
            if any(k in name for k in keywords):
                params.append(p)
        return params

    def _render_edit_core(self, batch_list):
        """
        A slimmed-down forward pass for editing finetune. It mirrors infer_single_view
        but also returns latent points for gradient masking.
        """
        gt_image_bs, gt_mask_bs, bound_mask_bs = [], [], []
        infer_images, nail_images, nail_masks, verts_cams = [], [], [], []
        vis_masks, vis_prob = [], []
        cano_pts_list = []
        ref_view = 0

        for i in range(len(batch_list)):
            batch_i = batch_list[i]
            gt_image = batch_i['original_image']
            if torch.is_tensor(gt_image) and gt_image.dim() == 4:
                if gt_image.shape[1] in (1, 3, 4):
                    gt_image = gt_image.permute(0, 2, 3, 1)
            gt_image = gt_image.to('cuda')
            bkgd_mask = batch_i['bkgd_mask'].to('cuda')
            bound_mask = batch_i['bound_mask'].to('cuda')
            gt_image_bs.append(gt_image)
            gt_mask_bs.append(bkgd_mask)
            bound_mask_bs.append(bound_mask)

            infer_image = batch_i['original_image'][ref_view].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')
            infer_images.append(infer_image)
            nail_images.append(batch_i['nail_image'][ref_view].unsqueeze(0).to('cuda'))
            nail_masks.append(batch_i['nail_mask'][ref_view].unsqueeze(0).to('cuda'))
            verts_cams.append(batch_i['verts_cam'][ref_view].unsqueeze(0).to('cuda'))

            p_vis, visible_mask = self.vis_pts(
                batch_i,
                batch_i['verts_cam'][ref_view, ..., 2],
                batch_i['nail_image'][ref_view, ...],
                return_mask=True,
            )
            vis_masks.append(visible_mask)
            vis_prob.append(p_vis)
            cano_pts_list.append(batch_i['big_pose_world_vertex'][ref_view].to('cuda'))

        gt_image_bs = torch.stack(gt_image_bs, dim=0)
        gt_mask_bs = torch.stack(gt_mask_bs, dim=0)
        bound_mask_bs = torch.stack(bound_mask_bs, dim=0)

        nail_pts = batch_list[0]['nail_image'][ref_view].shape[0]
        bs, n, h, w, c = gt_image_bs.shape

        image = torch.stack(infer_images, dim=0)
        nail_images = torch.stack(nail_images, dim=0).view(bs, nail_pts, -1)
        nail_masks = torch.stack(nail_masks, dim=0).squeeze(-1)
        verts_cams = torch.stack(verts_cams, dim=0).view(bs, nail_pts, -1)
        vis_masks = torch.stack(vis_masks, dim=0)
        vis_prob = torch.stack(vis_prob, dim=0)

        cano_pts = torch.stack(cano_pts_list, dim=0)

        def _select_first_view(tensor: Optional[torch.Tensor]):
            if torch.is_tensor(tensor) and tensor.ndim > 2:
                return tensor[0]
            return tensor

        for b in batch_list:
            vert_uv = b.get('vert_uv') if isinstance(b, dict) else None
            face_uv = b.get('face_uv') if isinstance(b, dict) else None
            face_uv_xy = b.get('face_uv_xy') if isinstance(b, dict) else None
            posed_points = b.get('world_vertex') if isinstance(b, dict) else None
            cam = b.get('full_proj_transform') if isinstance(b, dict) else None
            if not (torch.is_tensor(vert_uv) and torch.is_tensor(face_uv) and torch.is_tensor(face_uv_xy)):
                if isinstance(b, dict):
                    b['uv_map'] = None
                continue
            b['uv_map'] = {
                'vert_uv': _select_first_view(vert_uv),
                'face_uv': _select_first_view(face_uv),
                'face_uv_xy': _select_first_view(face_uv_xy),
                'posed_points': _select_first_view(posed_points),
                'cam': _select_first_view(cam),
            }

        latent_points, global_texture_feature, bias_dict = self.hand_model.forward_latent_points(
            vis_prob,
            nail_images,
            batch_list[0]['uv_map'],
            image[:, 0].permute(0, 3, 1, 2),
            vis_msk=vis_masks,
            camera=None,
            query_points=cano_pts,
            posed_points=verts_cams,
        )

        edit_stage = getattr(self, '_finetune_edit_stage', None)
        wild_stage = getattr(self, '_finetune_wild_stage', None)

        if (edit_stage == 'stage2') or (edit_stage == 'stage2a') or (wild_stage == 'stage2b'):
            renderer = self.hand_model.renderer
            renderer.edit_mask_mode = True
            renderer.edit_vis_mask = vis_masks

        gs_model_list, gs_densify_list, query_points = self.hand_model.renderer.forward_gs(
            gs_hidden_features=latent_points.to('cuda'),
            query_points=cano_pts,
            global_feature=global_texture_feature,
            batches=batch_list,
            verts_cam=verts_cams,
            nail_image=nail_images,
            color_bias=bias_dict.get("color_bias"),
            opacity_bias=bias_dict.get("opacity_bias"),
            vis_prob=vis_prob,
        )

        render_res_list = []
        for i in range(len(batch_list)):
            num_views = batch_list[i]['original_image'].shape[0]
            smplx_params = {k: v.to('cuda') for k, v in batch_list[i]['smpl_param'].items()}
            for view_idx in range(num_views):
                render_res = self.hand_model.renderer.forward_animate_gs(
                    gs_model_list[i],
                    gs_densify_list[i],
                    query_points[i],
                    self.hand_model.renderer.get_single_view_cam(batch_list[i], view_idx),
                    self.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                    batch_list[i]['width'][0],
                    batch_list[i]['height'][0],
                    self.bg_color,
                )
                render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                if k in {'offset', 'scaling', 'shs', 'nail_3d_mask'}:
                    out[k] = torch.cat([v[0]], dim=0)
                else:
                    out[k] = torch.concat(v, dim=1)
                    if k in {"comp_rgb", "comp_mask", "comp_depth", "comp_obj"}:
                        out[k] = out[k][0].permute(0, 2, 3, 1)
            else:
                out[k] = v

        render_image_bs = out['comp_rgb']
        render_mask_bs = out['comp_mask'].repeat(1, 1, 1, 3) if 'comp_mask' in out else None
        render_depth_bs = out.get('comp_depth', None)
        gau_scaling = out.get('scaling', None)
        nail_3d_mask = out.get('nail_3d_mask', None)

        render_image_bs = render_image_bs.reshape(bs, n, h, w, c)
        render_mask_bs = render_mask_bs.reshape(bs, n, h, w, c) if render_mask_bs is not None else None
        render_depth_bs = render_depth_bs.reshape(bs, n, h, w, -1) if torch.is_tensor(render_depth_bs) else None

        return {
            'gt_image_bs': gt_image_bs,
            'gt_mask_bs': gt_mask_bs,
            'bound_mask_bs': bound_mask_bs,
            'render_image_bs': render_image_bs,
            'render_mask_bs': render_mask_bs,
            'render_depth_bs': render_depth_bs,
            'latent_points': latent_points,
            'vis_masks': vis_masks,
            'vis_prob': vis_prob,
            'scaling': gau_scaling,
            'shs': out.get('shs', None),
            'nail_3d_mask': nail_3d_mask,
        }

    def _ensure_color_inversion_params(self, device: torch.device, use_bias=False, bias_size=32):
        if not hasattr(self.hand_model, "color_shift") or self.hand_model.color_shift is None:
            self.hand_model.color_shift = nn.Parameter(
                torch.zeros(1, 1, 1, 3, device=device, dtype=torch.float32)
            )
        else:
            self.hand_model.color_shift.data = self.hand_model.color_shift.data.to(device)

        if not hasattr(self.hand_model, "color_scale") or self.hand_model.color_scale is None:
            self.hand_model.color_scale = nn.Parameter(
                torch.ones(1, 1, 1, 3, device=device, dtype=torch.float32)
            )
        else:
            self.hand_model.color_scale.data = self.hand_model.color_scale.data.to(device)

        self.hand_model.color_shift.requires_grad = True
        self.hand_model.color_scale.requires_grad = True
        params = [self.hand_model.color_shift, self.hand_model.color_scale]

        renderer = getattr(self.hand_model, "renderer", None)
        gs_net = getattr(renderer, "gs_net", None) if renderer is not None else None
        if gs_net is not None:
            for name in ("color_latent_gamma", "color_latent_beta"):
                latent_param = getattr(gs_net, name, None)
                if latent_param is not None:
                    latent_param.requires_grad = False
            for name in ("stage1_color_coeff", "stage1_color_rgb"):
                stage1_param = getattr(gs_net, name, None)
                if stage1_param is not None:
                    stage1_param.requires_grad = True
                    params.append(stage1_param)

        self._sync_color_inversion_params()

        if use_bias:
            if not hasattr(self.hand_model, "color_bias_lowres") or self.hand_model.color_bias_lowres is None:
                self.hand_model.color_bias_lowres = nn.Parameter(
                    torch.zeros(1, 3, bias_size, bias_size, device=device, dtype=torch.float32)
                )
            else:
                self.hand_model.color_bias_lowres.data = self.hand_model.color_bias_lowres.data.to(device)
            self.hand_model.color_bias_lowres.requires_grad = True
            params.append(self.hand_model.color_bias_lowres)

        return params

    def _sync_color_inversion_params(self):
        renderer = getattr(self.hand_model, "renderer", None)
        if renderer is None:
            return
        if hasattr(self.hand_model, "color_shift") and self.hand_model.color_shift is not None:
            renderer.color_shift = self.hand_model.color_shift
        if hasattr(self.hand_model, "color_scale") and self.hand_model.color_scale is not None:
            renderer.color_scale = self.hand_model.color_scale
        if hasattr(self.hand_model, "color_bias_lowres") and self.hand_model.color_bias_lowres is not None:
            renderer.color_bias_lowres = self.hand_model.color_bias_lowres

    def _repeat_first_view(self, tensor: torch.Tensor, n_views: int):
        if not torch.is_tensor(tensor):
            return tensor
        if tensor.dim() == 0:
            tensor = tensor.view(1)
        if tensor.shape[0] == n_views:
            return tensor
        base = tensor[:1]
        repeat_shape = [n_views] + [1] * (base.dim() - 1)
        return base.repeat(*repeat_shape)

    def _build_ohta_pseudo_poses(self, base_batch: dict, n_views: int, device: torch.device, use_canonical_root: bool = False):
        """
        OHTA-style pseudo-view generation via pose rotation (NOT camera rotation).
        For each pseudo view i:
          1. Compute rotation axis from wrist(0) → MCP(4) using MANO FK at original hand pose
          2. Rotate root_pose by 360*i/n_views degrees around that axis
          3. Zero out hand_pose (fingers flat)
          4. Camera stays unchanged

        Args:
            base_batch: Input batch containing smpl_param with poses/shape
            n_views: Number of pseudo views to generate
            device: Target device
            use_canonical_root: If True, use fixed canonical root pose (-π/2, 0, 0) for pseudo-GT generation;
                                If False, use the original reference pose for Stage 2 rendering

        Returns dict with 'poses' [n_views, 48], 'shape' [n_views, 10], 'trans' [n_views, 3],
               'posed_verts' [n_views, V, 3]
        """
        import cv2
        from scipy.spatial.transform import Rotation as ScipyRotation

        mano_model = self.hand_model.renderer.mano_model
        mano_layer = mano_model.mano

        orig_poses = base_batch["smpl_param"]["poses"]
        if torch.is_tensor(orig_poses):
            orig_poses = orig_poses.detach().cpu().numpy()
        if orig_poses.ndim > 1:
            orig_poses = orig_poses[0]
        orig_poses = orig_poses.copy().astype(np.float32)

        if use_canonical_root:
            print(f"[_build_ohta_pseudo_poses] Using canonical root pose (-π/2, 0, 0) for pseudo-GT generation")
        else:
            print(f"[_build_ohta_pseudo_poses] Using reference root pose {orig_poses[:3].tolist()} for Stage 2 rendering")

        orig_shape = base_batch["smpl_param"]["shape"]
        if torch.is_tensor(orig_shape):
            orig_shape = orig_shape.detach().cpu().numpy()
        if orig_shape.ndim > 1:
            orig_shape = orig_shape[0]
        orig_shape = orig_shape.copy().astype(np.float32)

        with torch.no_grad():
            ori_posed_res = mano_layer(
                torch.from_numpy(orig_shape)[None].float(),
                torch.zeros(1, 3).float(),
                torch.from_numpy(orig_poses[3:])[None].float(),
                return_verts=True,
            )

        joints = ori_posed_res.joints[0].detach().cpu().numpy()
        rot_axis = joints[4] - joints[0]
        rot_axis_norm = rot_axis / (np.linalg.norm(rot_axis) + 1e-8)

        ori_rot_mat = cv2.Rodrigues(orig_poses[:3])[0]

        all_poses = []
        all_verts = []
        for i in range(n_views):
            angle_deg = 360.0 * i / n_views
            delta_r = ScipyRotation.from_rotvec(
                np.deg2rad(angle_deg) * rot_axis_norm
            ).as_matrix().astype(np.float32)

            new_rot_mat = ori_rot_mat @ delta_r
            new_root = cv2.Rodrigues(new_rot_mat)[0].reshape(3).astype(np.float32)

            new_pose = np.zeros(48, dtype=np.float32)
            new_pose[:3] = new_root

            all_poses.append(new_pose)

            with torch.no_grad():
                posed_res = mano_layer(
                    torch.from_numpy(orig_shape)[None].float(),
                    torch.zeros(1, 3).float(),
                    torch.zeros(1, 45).float(),
                    return_verts=True,
                )
            verts = posed_res.vertices[0].detach().cpu().numpy()
            all_verts.append(verts)

        all_poses = np.stack(all_poses, axis=0)
        all_verts = np.stack(all_verts, axis=0)

        result = {
            "poses": torch.from_numpy(all_poses).float(),
            "shape": torch.from_numpy(
                np.tile(orig_shape, (n_views, 1))
            ).float(),
            "trans": torch.zeros(n_views, 3, dtype=torch.float32),
            "posed_verts": torch.from_numpy(all_verts).float(),
        }

        return result

    def robust_color_statistics(self, image: torch.Tensor, mask: torch.Tensor,
                                  n_clusters: int = 3, return_all_clusters: bool = False):
        """
        Robust skin color extraction using K-means clustering.
        Selects the largest cluster as the dominant skin color, avoiding interference
        from patterns (e.g., white rabbit on dark skin) or specular highlights.

        Args:
            image: [H, W, 3] or [B, H, W, 3] RGB image tensor in [0, 1]
            mask: [H, W, 1] or [B, H, W, 1] binary mask (1 = hand region)
            n_clusters: number of K-means clusters (default 3: skin, highlight, pattern)
            return_all_clusters: if True, return all cluster info for debugging

        Returns:
            dominant_color: [3] tensor of the dominant (largest cluster) RGB color
            cluster_info: (optional) dict with cluster centers and sizes
        """
        from sklearn.cluster import KMeans

        if image.dim() == 4:
            image = image.reshape(-1, image.shape[-2], image.shape[-1], 3)
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1], 1)
            image = image.reshape(-1, 3)
            mask = mask.reshape(-1, 1)
        elif image.dim() == 3:
            image = image.reshape(-1, 3)
            mask = mask.reshape(-1, 1)

        valid_idx = (mask.squeeze(-1) > 0.5)
        if valid_idx.sum() < n_clusters * 10:
            if valid_idx.sum() > 0:
                return image[valid_idx].mean(dim=0), None
            else:
                return torch.ones(3, device=image.device) * 0.5, None

        valid_pixels = image[valid_idx].detach().cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
        labels = kmeans.fit_predict(valid_pixels)

        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        largest_cluster_idx = np.argmax(cluster_sizes)
        dominant_color = torch.from_numpy(kmeans.cluster_centers_[largest_cluster_idx]).float().to(image.device)

        if return_all_clusters:
            cluster_info = {
                'centers': torch.from_numpy(kmeans.cluster_centers_).float().to(image.device),
                'sizes': cluster_sizes,
                'dominant_idx': largest_cluster_idx
            }
            return dominant_color, cluster_info

        return dominant_color, None

    @torch.no_grad()
    def build_pseudo_gt_batch(self, batch_list, n_views: int = 8,
                              save_dir: str | None = None, save_prefix: str | None = None,
                              use_canonical_root: bool = True):
        """
        OHTA-style pseudo-GT generation:
        - Keep view 0 as the original input view (real supervision)
        - Rotate hand root pose around MCP→wrist axis for remaining (n_views - 1) pseudo views
        - Zero out hand (finger) pose
        - Keep camera parameters unchanged
        - Render self-supervised pseudo-GT images

        Args:
            batch_list: List of input batches
            n_views: Total number of views (including original)
            save_dir: Directory to save rendered pseudo-GT images
            save_prefix: Prefix for saved image filenames
            use_canonical_root: If True, use fixed canonical root (-π/2,0,0) for pseudo-GT;
                                If False, use reference pose (for Stage 2 rendering)
        """
        if batch_list is None or len(batch_list) == 0:
            raise ValueError("batch_list is required for pseudo-GT generation")

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pseudo_batches = []

        for batch_idx, base_batch in enumerate(batch_list):
            gs_model_list, gs_densify_list, query_points = self.infer_handavatar(base_batch)

            n_pseudo = max(0, n_views - 1)
            pseudo_smpl = None
            if n_pseudo > 0:
                pseudo_smpl = self._build_ohta_pseudo_poses(base_batch, n_pseudo, device=device,
                                                             use_canonical_root=use_canonical_root)
                pseudo_smpl = {k: v.to(device) for k, v in pseudo_smpl.items()}

            h = base_batch["height"]
            if torch.is_tensor(h):
                height_val = int(h[0].item()) if h.numel() > 1 else int(h.item())
            elif isinstance(h, (list, tuple)):
                height_val = int(h[0])
            else:
                height_val = int(h)

            w = base_batch["width"]
            if torch.is_tensor(w):
                width_val = int(w[0].item()) if w.numel() > 1 else int(w.item())
            elif isinstance(w, (list, tuple)):
                width_val = int(w[0])
            else:
                width_val = int(w)

            render_images = []
            render_masks = []

            for view_idx in range(n_pseudo):
                view_cam = self.hand_model.renderer.get_single_view_cam(base_batch, 0)

                smpl_view = self.hand_model.renderer.get_single_view_smpl_data(pseudo_smpl, view_idx)
                smpl_view = {k: v.to(device) if torch.is_tensor(v) else v for k, v in smpl_view.items()}

                render_res = self.hand_model.renderer.forward_animate_gs(
                    gs_model_list[0],
                    gs_densify_list[0],
                    query_points[0],
                    view_cam,
                    smpl_view,
                    height_val,
                    width_val,
                    self.bg_color,
                )

                comp_rgb = render_res["comp_rgb"][0, 0].detach().cpu()
                if comp_rgb.dim() == 3 and comp_rgb.shape[0] == 3:
                    comp_rgb = comp_rgb.permute(1, 2, 0)
                render_images.append(comp_rgb)

                comp_mask = render_res.get("comp_mask", None)
                if comp_mask is not None:
                    mask = comp_mask[0, 0]
                    if mask.dim() == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(-1)
                    render_masks.append(mask.detach().cpu())
                else:
                    render_masks.append(torch.ones(height_val, width_val, 1, dtype=comp_rgb.dtype))

            orig_img = base_batch["original_image"]
            if torch.is_tensor(orig_img):
                orig_img = orig_img[0].detach().cpu()
            else:
                orig_img = torch.tensor(orig_img[0])
            if orig_img.dim() == 3 and orig_img.shape[0] == 3:
                orig_img = orig_img.permute(1, 2, 0)

            orig_mask = base_batch.get("bkgd_mask", None)
            if torch.is_tensor(orig_mask):
                orig_mask = orig_mask[0].detach().cpu()
            elif orig_mask is None:
                orig_mask = torch.ones(height_val, width_val, 1, dtype=orig_img.dtype)
            else:
                orig_mask = torch.tensor(orig_mask[0])
            if orig_mask.dim() == 3 and orig_mask.shape[0] == 1:
                orig_mask = orig_mask.squeeze(0)
            if orig_mask.dim() == 2:
                orig_mask = orig_mask.unsqueeze(-1)
            if orig_mask.dim() == 3 and orig_mask.shape[-1] != 1:
                orig_mask = orig_mask[..., :1]

            if n_pseudo > 0:
                render_images = torch.stack(render_images, dim=0)
                render_masks = torch.stack(render_masks, dim=0)
                if render_masks.dim() == 4 and render_masks.shape[-1] != 1:
                    render_masks = render_masks[..., :1]
                render_images = torch.cat([orig_img.unsqueeze(0), render_images], dim=0)
                render_masks = torch.cat([orig_mask.unsqueeze(0), render_masks], dim=0)
            else:
                render_images = orig_img.unsqueeze(0)
                render_masks = orig_mask.unsqueeze(0)

            if render_masks.shape[-1] == 1:
                mask_3c = render_masks.repeat(1, 1, 1, 3)
            else:
                mask_3c = render_masks

            if save_dir is not None and batch_idx == 0:
                os.makedirs(save_dir, exist_ok=True)
                prefix = save_prefix or "pseudo"
                max_views = min(n_views, 8)
                for view_idx in range(max_views):
                    libcore.write_tensor_image(
                        os.path.join(save_dir, f"{prefix}_view{view_idx}.jpg"),
                        render_images[view_idx],
                        rgb2bgr=True,
                    )

            pseudo = dict(base_batch)
            pseudo["original_image"] = render_images
            pseudo["bkgd_mask"] = mask_3c

            orig_bound = base_batch.get("bound_mask", None)
            if torch.is_tensor(orig_bound):
                orig_bound_v0 = orig_bound[0].detach().cpu()
                if orig_bound_v0.dim() == 3 and orig_bound_v0.shape[0] in (1, 3):
                    orig_bound_v0 = orig_bound_v0.permute(1, 2, 0)
                if orig_bound_v0.dim() == 2:
                    orig_bound_v0 = orig_bound_v0.unsqueeze(-1)
                if orig_bound_v0.shape[-1] == 1:
                    orig_bound_v0 = orig_bound_v0.repeat(1, 1, 3)
                n_total = render_images.shape[0]
                zeros_pseudo = torch.zeros(n_total - 1, orig_bound_v0.shape[0], orig_bound_v0.shape[1], 3,
                                           dtype=orig_bound_v0.dtype)
                pseudo["bound_mask"] = torch.cat([orig_bound_v0.unsqueeze(0), zeros_pseudo], dim=0)
            else:
                pseudo["bound_mask"] = mask_3c

            base_smpl = base_batch.get("smpl_param", {})
            pseudo_smpl_cpu = {k: v.detach().cpu() for k, v in pseudo_smpl.items()} if pseudo_smpl else {}
            combined_smpl = {}
            for k, v in base_smpl.items():
                if torch.is_tensor(v):
                    base_v = v.detach().cpu()
                else:
                    base_v = torch.tensor(v)
                if base_v.dim() == 1:
                    base_v = base_v.unsqueeze(0)
                if k in pseudo_smpl_cpu:
                    combined_smpl[k] = torch.cat([base_v, pseudo_smpl_cpu[k]], dim=0)
                else:
                    combined_smpl[k] = base_v
            pseudo["smpl_param"] = combined_smpl

            skip_keys = {"original_image", "bkgd_mask", "bound_mask", "smpl_param"}
            for key, value in pseudo.items():
                if key in skip_keys:
                    continue
                if torch.is_tensor(value):
                    if value.dim() >= 1 and value.shape[0] == 1:
                        pseudo[key] = self._repeat_first_view(value, n_views)
                elif isinstance(value, (list, tuple)):
                    if len(value) == 1:
                        pseudo[key] = list(value) * n_views
                    elif 0 < len(value) < n_views:
                        pseudo[key] = [value[0]] * n_views

            if "image_name" in pseudo and isinstance(pseudo["image_name"], (list, tuple)):
                if len(pseudo["image_name"]) > 0:
                    name0 = pseudo["image_name"][0]
                    names = [name0]
                    names.extend([f"{name0}_pseudo{i}" for i in range(1, n_views)])
                    pseudo["image_name"] = names

            pseudo_batches.append(pseudo)

        return pseudo_batches

    def finetune_wild_inversion(self, batch=None, scaler=None, iter=None, writer=None, pbar=None,
                                total_iters=200, lr=1e-2, downscale=1, use_hand_mask=True,

                                mean_weight=20.0, std_weight=1.0,
                                use_bias=False, bias_size=32, bias_weight=0.5, bias_tv_weight=0.1,
                                use_base_feat=True):
        """
        OHTA-style inversion stage: only optimize global color shift/scale to match the input.
        This is designed to adjust color while keeping texture stable.
        """
        import torch.nn.functional as F
        if batch is None:
            raise ValueError("batch is required for finetune_wild_inversion")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        freeze_all(self.hand_model)
        params = self._ensure_color_inversion_params(device, use_bias=use_bias, bias_size=bias_size)

        if not getattr(self, "_inversion_reset_done", False):
            self.hand_model.color_shift.data.zero_()
            self.hand_model.color_scale.data.fill_(1.0)
            renderer = getattr(self.hand_model, "renderer", None)
            gs_net = getattr(renderer, "gs_net", None) if renderer is not None else None
            if gs_net is not None:
                if hasattr(gs_net, "color_latent_gamma") and gs_net.color_latent_gamma is not None:
                    gs_net.color_latent_gamma.data.zero_()
                if hasattr(gs_net, "color_latent_beta") and gs_net.color_latent_beta is not None:
                    gs_net.color_latent_beta.data.zero_()
                if hasattr(gs_net, "stage1_color_coeff") and gs_net.stage1_color_coeff is not None:
                    gs_net.stage1_color_coeff.data.zero_()
                if hasattr(gs_net, "stage1_color_rgb") and gs_net.stage1_color_rgb is not None:
                    gs_net.stage1_color_rgb.data.fill_(1.0)
            if use_bias and hasattr(self.hand_model, "color_bias_lowres") and self.hand_model.color_bias_lowres is not None:
                self.hand_model.color_bias_lowres.data.zero_()
            self._inversion_reset_done = True

        enable_base_feat_late = False

        def _build_inversion_param_groups():
            param_groups = [{
                "params": [self.hand_model.color_shift, self.hand_model.color_scale],
                "lr": lr,
            }]

            renderer_local = getattr(self.hand_model, "renderer", None)
            gs_net_local = getattr(renderer_local, "gs_net", None) if renderer_local is not None else None
            if gs_net_local is not None:
                field_params = []
                use_stage1_color_field = os.getenv("LHM_STAGE1_ENABLE_COLOR_FIELD", "0") in ("1", "true", "True")
                if os.getenv("LHM_STAGE1_GLOBAL_ONLY", "0") in ("1", "true", "True"):
                    use_stage1_color_field = False
                if use_stage1_color_field:
                    for name in ("stage1_color_coeff", "stage1_color_rgb"):
                        stage1_param = getattr(gs_net_local, name, None)
                        if stage1_param is not None and stage1_param.requires_grad:
                            field_params.append(stage1_param)
                if field_params:
                    param_groups.append({
                        "params": field_params,
                        "lr": lr * 1.5,
                    })

            if use_bias and hasattr(self.hand_model, "color_bias_lowres") and self.hand_model.color_bias_lowres is not None:
                param_groups.append({
                    "params": [self.hand_model.color_bias_lowres],
                    "lr": lr * 0.5,
                })

            return param_groups

        if getattr(self, 'no_pretrain', False):
            _no_pretrain_modules = [
                self.hand_model.pcl_embed,
                self.hand_model.motion_embed_mlp,
                self.hand_model.adapter,
                self.hand_model.aggregator,
                self.hand_model.transformer,
                self.hand_model.encoder,
                self.hand_model.renderer,
            ]
            unfreeze_modules(_no_pretrain_modules, train_mode=True)

            backbone_lr = 1e-4
            all_params = []
            for m in _no_pretrain_modules:
                all_params.extend([p for p in m.parameters() if p.requires_grad])
            optimizer_groups = [{'params': all_params, 'lr': backbone_lr}]
            if iter == 1:
                n_total = sum(p.numel() for p in all_params)
                print(f"\033[93m[no_pretrain] Unfroze checkpoint components: {len(all_params)} param tensors, {n_total} params total, lr={backbone_lr}\033[0m")
        else:
            optimizer_groups = _build_inversion_param_groups()

        flat_opt_params = [p for group in optimizer_groups for p in group["params"]]
        if getattr(self, "_inversion_optimizer", None) is None:
            self._inversion_optimizer = torch.optim.Adam(optimizer_groups, weight_decay=0.0)
            self._inversion_scheduler = None
        else:
            opt_params = set()
            for group in self._inversion_optimizer.param_groups:
                opt_params.update(group["params"])
            if any(p not in opt_params for p in flat_opt_params):
                self._inversion_optimizer = torch.optim.Adam(optimizer_groups, weight_decay=0.0)
                self._inversion_scheduler = None

        renderer = getattr(self.hand_model, "renderer", None)
        gs_net = getattr(renderer, "gs_net", None) if renderer is not None else None
        use_stage1_color_field = os.getenv("LHM_STAGE1_ENABLE_COLOR_FIELD", "0") in ("1", "true", "True")
        if os.getenv("LHM_STAGE1_GLOBAL_ONLY", "0") in ("1", "true", "True"):
            use_stage1_color_field = False
        field_alpha = 1.0
        if gs_net is not None:
            if use_stage1_color_field:
                field_warmup_start = min(8, max(1, total_iters // 20))
                field_ramp = max(18, total_iters // 7)
                if iter is None or iter <= field_warmup_start:
                    field_alpha = 0.0
                else:
                    field_alpha = min(1.0, float(iter - field_warmup_start) / float(field_ramp))
            else:
                field_alpha = 0.0
            gs_net.stage1_color_field_alpha = field_alpha

        render_pkg = self._render_edit_core(batch)

        render_image_bs = render_pkg['render_image_bs']
        gt_image_bs = render_pkg['gt_image_bs'].detach()
        hand_mask = render_pkg.get('gt_mask_bs', None)
        fullres_render_image_bs = render_image_bs
        fullres_gt_image_bs = gt_image_bs

        if hand_mask is None:
            hand_mask = torch.ones_like(gt_image_bs[..., 0:1])
        if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
            hand_mask = hand_mask[..., 0:1]
        hand_mask = hand_mask.detach()

        if use_hand_mask:
            mask = hand_mask
        else:
            mask = torch.ones_like(hand_mask)
        fullres_mask = mask

        effective_downscale = downscale
        if effective_downscale is None:
            effective_downscale = 1.0

        if effective_downscale is not None and effective_downscale < 1.0:
            b, n, h, w, c = render_image_bs.shape
            render_flat = render_image_bs.reshape(b * n, h, w, c).permute(0, 3, 1, 2)
            gt_flat = gt_image_bs.reshape(b * n, h, w, c).permute(0, 3, 1, 2)
            mask_flat = mask.reshape(b * n, h, w, 1).permute(0, 3, 1, 2)
            target_h = max(1, int(h * effective_downscale))
            target_w = max(1, int(w * effective_downscale))
            render_flat = F.interpolate(render_flat, size=(target_h, target_w), mode="area")
            gt_flat = F.interpolate(gt_flat, size=(target_h, target_w), mode="area")
            mask_flat = F.interpolate(mask_flat, size=(target_h, target_w), mode="area")
            render_image_bs = render_flat.permute(0, 2, 3, 1).reshape(b, n, target_h, target_w, c)
            gt_image_bs = gt_flat.permute(0, 2, 3, 1).reshape(b, n, target_h, target_w, c)
            mask = mask_flat.permute(0, 2, 3, 1).reshape(b, n, target_h, target_w, 1)

        render_color = render_image_bs
        fullres_render_color = fullres_render_image_bs
        if use_bias and hasattr(self.hand_model, "color_bias_lowres") and self.hand_model.color_bias_lowres is not None:
            b, n, h, w, _ = render_color.shape
            bias = self.hand_model.color_bias_lowres
            bias_up = F.interpolate(bias, size=(h, w), mode="bilinear", align_corners=False)
            bias_up = bias_up.permute(0, 2, 3, 1).unsqueeze(1)
            render_color = render_color + bias_up
            b_full, n_full, h_full, w_full, _ = fullres_render_color.shape
            bias_up_full = F.interpolate(bias, size=(h_full, w_full), mode="bilinear", align_corners=False)
            bias_up_full = bias_up_full.permute(0, 2, 3, 1).unsqueeze(1)
            fullres_render_color = fullres_render_color + bias_up_full

        def _erode_mask_inversion(mask_bhw1, kernel_size=9):
            """Erode mask to avoid boundary noise. Input: [B, N, H, W, 1], Output: same shape."""
            b, n, h, w, c = mask_bhw1.shape
            mask_flat = mask_bhw1.reshape(b * n, h, w, c).permute(0, 3, 1, 2)

            eroded = -F.max_pool2d(-mask_flat, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            return eroded.permute(0, 2, 3, 1).reshape(b, n, h, w, c)

        def _soft_mask_inversion(mask_bhw1, erode_kernel=11, blur_kernel=13, blur_sigma=3.0):
            """Create soft boundary mask: erode then Gaussian-blur for smooth falloff.
            Far from boundary → weight ~1.0; at boundary → smooth ramp to 0."""
            b, n, h, w, c = mask_bhw1.shape
            eroded = _erode_mask_inversion(mask_bhw1, kernel_size=erode_kernel)

            flat = eroded.reshape(b * n, h, w, c).permute(0, 3, 1, 2)
            pad = blur_kernel // 2

            coords = torch.arange(blur_kernel, dtype=flat.dtype, device=flat.device) - pad
            gauss_1d = torch.exp(-0.5 * (coords / blur_sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()

            kernel_h = gauss_1d.view(1, 1, blur_kernel, 1)
            kernel_w = gauss_1d.view(1, 1, 1, blur_kernel)
            blurred = F.conv2d(F.pad(flat, (0, 0, pad, pad), mode='replicate'), kernel_h)
            blurred = F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode='replicate'), kernel_w)
            return blurred.permute(0, 2, 3, 1).reshape(b, n, h, w, c)

        eroded_mask = _erode_mask_inversion(mask, kernel_size=11)
        soft_mask = _soft_mask_inversion(mask, erode_kernel=11, blur_kernel=13, blur_sigma=3.0)
        loss_mask = soft_mask if use_hand_mask else mask
        loss_mask_3c = loss_mask.repeat(1, 1, 1, 1, 3)
        diff = (render_color - gt_image_bs) * loss_mask_3c
        fullres_soft_mask = _soft_mask_inversion(fullres_mask, erode_kernel=11, blur_kernel=13, blur_sigma=3.0)
        fullres_loss_mask = fullres_soft_mask if use_hand_mask else fullres_mask
        fullres_loss_mask_3c = fullres_loss_mask.repeat(1, 1, 1, 1, 3)

        denom = loss_mask_3c.sum().clamp_min(1e-6)
        loss_main = diff.abs().sum() / denom

        mean_r = (render_color * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        mean_g = (gt_image_bs * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        loss_mean = (mean_r - mean_g).abs().mean()

        var_r = ((render_color - mean_r) ** 2 * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        var_g = ((gt_image_bs - mean_g) ** 2 * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        std_r = torch.sqrt(var_r + 1e-6)
        std_g = torch.sqrt(var_g + 1e-6)
        loss_std = (std_r - std_g).abs().mean()

        with torch.no_grad():
            if not hasattr(self, '_initial_psnr') or iter == 1:
                mse = ((render_image_bs - gt_image_bs) ** 2 * loss_mask_3c).sum() / denom
                psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
                self._initial_psnr = psnr.item()
                print(f"[Inversion] Initial PSNR (render vs GT): {self._initial_psnr:.2f} dB")

                mean_original = (render_image_bs * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
                mean_gt = (gt_image_bs * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
                color_diff = (mean_original - mean_gt).abs()
                print(f"[Inversion] Mean color diff (RGB): R={color_diff[0]:.4f}, G={color_diff[1]:.4f}, B={color_diff[2]:.4f}")

            initial_psnr = self._initial_psnr

            if initial_psnr < 10:
                adaptive_color_reg = 1e-4
            elif initial_psnr < 12:
                adaptive_color_reg = 5e-4
            elif initial_psnr < 14:
                adaptive_color_reg = 1e-3
            elif initial_psnr < 16:
                adaptive_color_reg = 3e-3
            elif initial_psnr < 20:
                adaptive_color_reg = 1e-2
            else:
                adaptive_color_reg = 2e-2

        color_reg_weight = adaptive_color_reg
        reg = color_reg_weight * (self.hand_model.color_shift ** 2).mean()
        reg += color_reg_weight * ((self.hand_model.color_scale - 1.0) ** 2).mean()
        renderer = getattr(self.hand_model, "renderer", None)
        gs_net = getattr(renderer, "gs_net", None) if renderer is not None else None
        stage1_field_norm = torch.tensor(0.0, device=device)
        stage1_rgb_norm = torch.tensor(0.0, device=device)
        if gs_net is not None:
            stage1_color_coeff = getattr(gs_net, "stage1_color_coeff", None)
            stage1_color_rgb = getattr(gs_net, "stage1_color_rgb", None)
            if use_stage1_color_field and stage1_color_coeff is not None:
                reg += max(5.0 * color_reg_weight, 5e-4) * (stage1_color_coeff ** 2).mean()
                stage1_field_norm = stage1_color_coeff.detach().norm()
            if use_stage1_color_field and stage1_color_rgb is not None:
                reg += max(2.0 * color_reg_weight, 2e-4) * (stage1_color_rgb ** 2).mean()
                stage1_rgb_norm = stage1_color_rgb.detach().norm()

        if iter is not None and iter % 50 == 0:
            current_mse = ((render_color - gt_image_bs) ** 2 * loss_mask_3c).sum() / denom
            current_psnr = 10 * torch.log10(1.0 / current_mse.clamp_min(1e-10)).item()
            fullres_denom = fullres_loss_mask_3c.sum().clamp_min(1e-6)
            fullres_mse = ((fullres_render_color - fullres_gt_image_bs) ** 2 * fullres_loss_mask_3c).sum() / fullres_denom
            current_fullres_psnr = 10 * torch.log10(1.0 / fullres_mse.clamp_min(1e-10)).item()
            color_shift_flat = self.hand_model.color_shift.detach().view(-1)
            color_scale_flat = self.hand_model.color_scale.detach().view(-1)
            print(
                f"[Inversion] iter={iter} PSNR(train/full): "
                f"{initial_psnr:.2f}->{current_psnr:.2f}/{current_fullres_psnr:.2f} dB, "
                f"reg_weight={color_reg_weight:.4f}"
            )
            print(f"[Inversion] color_shift=[{color_shift_flat[0].item():.4f}, {color_shift_flat[1].item():.4f}, {color_shift_flat[2].item():.4f}]")
            print(f"[Inversion] color_scale=[{color_scale_flat[0].item():.4f}, {color_scale_flat[1].item():.4f}, {color_scale_flat[2].item():.4f}]")
            print(
                f"[Inversion] stage1_color_field alpha={field_alpha:.3f}, "
                f"coeff_norm={stage1_field_norm.item():.4f}, rgb_norm={stage1_rgb_norm.item():.4f}, "
                f"downscale={effective_downscale:.2f}"
            )
        bias_reg = 0.0
        if use_bias and hasattr(self.hand_model, "color_bias_lowres") and self.hand_model.color_bias_lowres is not None:
            bias = self.hand_model.color_bias_lowres
            tv_h = (bias[:, :, 1:, :] - bias[:, :, :-1, :]).abs().mean()
            tv_w = (bias[:, :, :, 1:] - bias[:, :, :, :-1]).abs().mean()
            bias_reg = bias_weight * (bias ** 2).mean() + bias_tv_weight * (tv_h + tv_w)

        invisible_color_reg = torch.tensor(0.0, device=device)

        if iter is not None and iter % 50 == 0:
            if use_stage1_color_field:
                print(f"[Inversion Stage1 iter={iter}] Optimizing renderer color_shift/scale + smooth canonical 3D color field (no geometry/base_feat)")
            else:
                print(f"[Inversion Stage1 iter={iter}] Optimizing renderer color_shift/scale only (global-only, no canonical 3D color field)")

        total_loss = (
            10.0 * loss_main
            + mean_weight * loss_mean
            + std_weight * loss_std
            + 10.0 * reg

        )

        self._inversion_optimizer.zero_grad(set_to_none=True)

        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(flat_opt_params, max_norm=1.0)

        has_grad = any(p.grad is not None for p in flat_opt_params)
        if has_grad:
            self._inversion_optimizer.step()
        if getattr(self, "_inversion_scheduler", None) is not None:
            self._inversion_scheduler.step()

        if writer is not None and iter is not None:
            writer.add_scalar('train/inversion_total', total_loss, iter)
            writer.add_scalar('train/inversion_loss', loss_main, iter)
            writer.add_scalar('train/inversion_mean', loss_mean, iter)
            writer.add_scalar('train/inversion_std', loss_std, iter)
            writer.add_scalar('train/inversion_reg', reg, iter)
            writer.add_scalar('train/inversion_stage1_color_field_norm', stage1_field_norm, iter)
            writer.add_scalar('train/inversion_stage1_color_rgb_norm', stage1_rgb_norm, iter)
            writer.add_scalar('train/inversion_stage1_color_field_alpha', field_alpha, iter)
            writer.add_scalar('train/inversion_downscale', effective_downscale, iter)
            writer.add_scalar('train/inversion_invisible_color_reg', invisible_color_reg, iter)
            if use_bias:
                writer.add_scalar('train/inversion_bias_reg', bias_reg, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": "inversion",
                "loss": float(loss_main),
                "total": float(total_loss),
                "psnr": float(current_fullres_psnr) if iter is not None and iter % 50 == 0 else None,
                "invis_reg": float(invisible_color_reg),
            })

    def finetune_wild_inversion_stage2(self, batch=None, pseudo_batch=None, scaler=None,
                                        iter=None, writer=None, pbar=None,
                                        total_iters=200, lr=5e-4, downscale=1,
                                        color_consistency_weight=1.0, n_clusters=3):
        """
        Second stage of inversion (iterations 201-400): optimize color parameters while
        enforcing color consistency between pseudo-GT views and the reference image.

        Uses K-means clustering to extract the dominant skin color from both:
        1. Reference image (real input)
        2. Each pseudo-GT rendered view

        The loss pulls the pseudo-view dominant colors toward the reference dominant color.
        This ensures view-independent color consistency while avoiding interference from
        small patterns (e.g., tattoos, accessories) on the hand.
        """
        import torch.nn.functional as F
        if batch is None or pseudo_batch is None:
            raise ValueError("Both batch and pseudo_batch are required for inversion_stage2")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = self._ensure_color_inversion_params(device, use_bias=False, bias_size=32)
        base_feat_params = self._collect_params(
            ['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping']
        )
        for p in base_feat_params:
            p.requires_grad = True
        params = list(dict.fromkeys(params + base_feat_params))

        if getattr(self, "_inversion_stage2_optimizer", None) is None:
            self._inversion_stage2_optimizer = torch.optim.Adam(params, lr=lr * 0.5, weight_decay=0.0)
            self._inversion_stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._inversion_stage2_optimizer,
                T_max=total_iters,
                eta_min=max(lr * 0.05, 1e-6),
            )

        ref_render_pkg = self._render_edit_core(batch)
        ref_render = ref_render_pkg['render_image_bs']
        ref_gt = ref_render_pkg['gt_image_bs'].detach()
        ref_mask = ref_render_pkg.get('gt_mask_bs', None)

        if ref_mask is None:
            ref_mask = torch.ones_like(ref_gt[..., 0:1])
        if ref_mask.dim() == 5 and ref_mask.shape[-1] > 1:
            ref_mask = ref_mask[..., 0:1]
        ref_mask = ref_mask.detach()

        ref_gt_flat = ref_gt[0, 0]
        ref_mask_flat = ref_mask[0, 0]
        ref_dominant_color, _ = self.robust_color_statistics(ref_gt_flat, ref_mask_flat, n_clusters=n_clusters)

        pseudo_render_pkg = self._render_edit_core(pseudo_batch)
        pseudo_render = pseudo_render_pkg['render_image_bs']
        pseudo_mask = pseudo_render_pkg.get('gt_mask_bs', None)

        if pseudo_mask is None:
            pseudo_mask = torch.ones_like(pseudo_render[..., 0:1])
        if pseudo_mask.dim() == 5 and pseudo_mask.shape[-1] > 1:
            pseudo_mask = pseudo_mask[..., 0:1]
        pseudo_mask = pseudo_mask.detach()

        color_diff_loss = torch.tensor(0.0, device=device, requires_grad=True)
        b, n_views, h, w, c = pseudo_render.shape

        for view_idx in range(n_views):
            view_render = pseudo_render[0, view_idx]
            view_mask = pseudo_mask[0, view_idx]
            if view_mask.shape[-1] > 1:
                view_mask = view_mask[..., 0:1]

            view_dominant_color, _ = self.robust_color_statistics(
                view_render.detach(), view_mask, n_clusters=n_clusters
            )

            valid_mask = (view_mask > 0.5).float()
            denom = valid_mask.sum().clamp_min(1e-6)
            view_mean_color = (view_render * valid_mask).sum(dim=(0, 1)) / denom

            color_diff = (view_mean_color - ref_dominant_color.detach()).pow(2).mean()
            color_diff_loss = color_diff_loss + color_diff

        color_diff_loss = color_diff_loss / max(n_views, 1)

        if downscale is not None and downscale < 1.0:
            b_r, n_r, h_r, w_r, c_r = ref_render.shape
            render_flat = ref_render.reshape(b_r * n_r, h_r, w_r, c_r).permute(0, 3, 1, 2)
            gt_flat = ref_gt.reshape(b_r * n_r, h_r, w_r, c_r).permute(0, 3, 1, 2)
            mask_flat = ref_mask.reshape(b_r * n_r, h_r, w_r, 1).permute(0, 3, 1, 2)
            target_h = max(1, int(h_r * downscale))
            target_w = max(1, int(w_r * downscale))
            render_flat = F.interpolate(render_flat, size=(target_h, target_w), mode="area")
            gt_flat = F.interpolate(gt_flat, size=(target_h, target_w), mode="area")
            mask_flat = F.interpolate(mask_flat, size=(target_h, target_w), mode="area")
            ref_render = render_flat.permute(0, 2, 3, 1).reshape(b_r, n_r, target_h, target_w, c_r)
            ref_gt = gt_flat.permute(0, 2, 3, 1).reshape(b_r, n_r, target_h, target_w, c_r)
            ref_mask = mask_flat.permute(0, 2, 3, 1).reshape(b_r, n_r, target_h, target_w, 1)

        mask_3c = ref_mask.repeat(1, 1, 1, 1, 3)
        diff = (ref_render - ref_gt)
        loss_main = ((diff.abs() + 1e-5) ** 0.3).mean()

        color_reg_weight = 1e-3
        reg = color_reg_weight * (self.hand_model.color_shift ** 2).mean()
        reg += color_reg_weight * ((self.hand_model.color_scale - 1.0) ** 2).mean()

        total_loss = loss_main + color_consistency_weight * color_diff_loss + reg

        self._inversion_stage2_optimizer.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=True)
        has_grad = any(p.grad is not None for p in params)
        if has_grad:
            self._inversion_stage2_optimizer.step()
        if getattr(self, "_inversion_stage2_scheduler", None) is not None:
            self._inversion_stage2_scheduler.step()

        if writer is not None and iter is not None:
            writer.add_scalar('train/inv_stage2_main', loss_main, iter)
            writer.add_scalar('train/inv_stage2_color_consistency', color_diff_loss, iter)
            writer.add_scalar('train/inv_stage2_reg', reg, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": "inv_stage2",
                "main": f"{float(loss_main):.4f}",
                "color": f"{float(color_diff_loss):.4f}",
            })

    def _render_nail_core(self, batch_list):
        """
        Forward pass for nail finetune with stage2b nail_3d_mask support.
        Extends _render_edit_core by pre-computing and combining nail_3d_mask with vis_masks.
        """
        gt_image_bs, gt_mask_bs, bound_mask_bs = [], [], []
        infer_images, nail_images, nail_masks, verts_cams = [], [], [], []
        vis_masks, vis_prob = [], []
        cano_pts_list = []
        ref_view = 0

        for i in range(len(batch_list)):
            batch_i = batch_list[i]
            gt_image = batch_i['original_image'].permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch_i['bkgd_mask'].to('cuda')
            bound_mask = batch_i['bound_mask'].to('cuda')
            gt_image_bs.append(gt_image)
            gt_mask_bs.append(bkgd_mask)
            bound_mask_bs.append(bound_mask)

            infer_image = batch_i['original_image'][ref_view].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')
            infer_images.append(infer_image)
            nail_images.append(batch_i['nail_image'][ref_view].unsqueeze(0).to('cuda'))
            nail_masks.append(batch_i['nail_mask'][ref_view].unsqueeze(0).to('cuda'))
            verts_cams.append(batch_i['verts_cam'][ref_view].unsqueeze(0).to('cuda'))

            p_vis, visible_mask = self.vis_pts(
                batch_i,
                batch_i['verts_cam'][ref_view, ..., 2],
                batch_i['nail_image'][ref_view, ...],
                return_mask=True,
            )
            vis_masks.append(visible_mask)
            vis_prob.append(p_vis)
            cano_pts_list.append(batch_i['big_pose_world_vertex'][ref_view].to('cuda'))

        gt_image_bs = torch.stack(gt_image_bs, dim=0)
        gt_mask_bs = torch.stack(gt_mask_bs, dim=0)
        bound_mask_bs = torch.stack(bound_mask_bs, dim=0)

        nail_pts = batch_list[0]['nail_image'][ref_view].shape[0]
        bs, n, h, w, c = gt_image_bs.shape

        image = torch.stack(infer_images, dim=0)
        nail_images = torch.stack(nail_images, dim=0).view(bs, nail_pts, -1)
        nail_masks = torch.stack(nail_masks, dim=0).squeeze(-1)
        verts_cams = torch.stack(verts_cams, dim=0).view(bs, nail_pts, -1)
        vis_masks = torch.stack(vis_masks, dim=0)
        vis_prob = torch.stack(vis_prob, dim=0)

        cano_pts = torch.stack(cano_pts_list, dim=0)

        def _select_first_view(tensor: Optional[torch.Tensor]):
            if torch.is_tensor(tensor) and tensor.ndim > 2:
                return tensor[0]
            return tensor

        for b in batch_list:
            vert_uv = b.get('vert_uv') if isinstance(b, dict) else None
            face_uv = b.get('face_uv') if isinstance(b, dict) else None
            face_uv_xy = b.get('face_uv_xy') if isinstance(b, dict) else None
            posed_points = b.get('world_vertex') if isinstance(b, dict) else None
            cam = b.get('full_proj_transform') if isinstance(b, dict) else None
            if not (torch.is_tensor(vert_uv) and torch.is_tensor(face_uv) and torch.is_tensor(face_uv_xy)):
                if isinstance(b, dict):
                    b['uv_map'] = None
                continue
            b['uv_map'] = {
                'vert_uv': _select_first_view(vert_uv),
                'face_uv': _select_first_view(face_uv),
                'face_uv_xy': _select_first_view(face_uv_xy),
                'posed_points': _select_first_view(posed_points),
                'cam': _select_first_view(cam),
            }

        latent_points, global_texture_feature, bias_dict = self.hand_model.forward_latent_points(
            vis_prob,
            nail_images,
            batch_list[0]['uv_map'],
            image[:, 0].permute(0, 3, 1, 2),
            vis_msk=vis_masks,
            camera=None,
            query_points=cano_pts,
        )

        nail_stage = getattr(self, '_finetune_nail_stage', None)
        nail_3d_mask_combined = None

        if nail_stage == 'stage2b':
            gs_model_list_temp, gs_densify_list_temp, query_points_temp = self.hand_model.renderer.forward_gs(
                gs_hidden_features=latent_points.to('cuda'),
                query_points=cano_pts,
                global_feature=global_texture_feature,
                batches=batch_list,
                verts_cam=verts_cams,
                nail_image=nail_images,
                color_bias=bias_dict.get("color_bias"),
                opacity_bias=bias_dict.get("opacity_bias"),
                vis_prob=vis_prob,
            )

            nail_3d_masks = []
            for i in range(len(batch_list)):
                num_views = batch_list[i]['original_image'].shape[0]
                smplx_params = {k: v.to('cuda') for k, v in batch_list[i]['smpl_param'].items()}
                for view_idx in range(num_views):
                    render_res = self.hand_model.renderer.forward_animate_gs(
                        gs_model_list_temp[i],
                        gs_densify_list_temp[i],
                        query_points_temp[i],
                        self.hand_model.renderer.get_single_view_cam(batch_list[i], view_idx),
                        self.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                        batch_list[i]['width'][0],
                        batch_list[i]['height'][0],
                        self.bg_color,
                    )
                    if 'nail_3d_mask' in render_res and render_res['nail_3d_mask'] is not None:
                        nail_3d_masks.append(render_res['nail_3d_mask'])

            if nail_3d_masks:
                nail_3d_mask = render_res['nail_3d_mask'].squeeze(-1).to(vis_masks.device)

                nail_3d_mask_combined = (vis_masks > 0.5).float() * (nail_3d_mask > 0.5).float()

        edit_stage = getattr(self, '_finetune_edit_stage', 'stage1')
        wild_stage = getattr(self, '_finetune_wild_stage', None)

        if (edit_stage == 'stage2') or (wild_stage == 'stage2b') or (nail_stage == 'stage2b'):
            renderer = self.hand_model.renderer
            renderer.edit_mask_mode = True
            renderer.edit_vis_mask = vis_masks
            if nail_stage == 'stage2b' and nail_3d_mask_combined is not None:
                renderer.edit_vis_mask = nail_3d_mask_combined

        gs_model_list, gs_densify_list, query_points = self.hand_model.renderer.forward_gs(
            gs_hidden_features=latent_points.to('cuda'),
            query_points=cano_pts,
            global_feature=global_texture_feature,
            batches=batch_list,
            verts_cam=verts_cams,
            nail_image=nail_images,
            color_bias=bias_dict.get("color_bias"),
            opacity_bias=bias_dict.get("opacity_bias"),
            vis_prob=vis_prob,
        )

        render_res_list = []
        for i in range(len(batch_list)):
            num_views = batch_list[i]['original_image'].shape[0]
            smplx_params = {k: v.to('cuda') for k, v in batch_list[i]['smpl_param'].items()}
            for view_idx in range(num_views):
                render_res = self.hand_model.renderer.forward_animate_gs(
                    gs_model_list[i],
                    gs_densify_list[i],
                    query_points[i],
                    self.hand_model.renderer.get_single_view_cam(batch_list[i], view_idx),
                    self.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                    batch_list[i]['width'][0],
                    batch_list[i]['height'][0],
                    self.bg_color,
                )
                render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                if k in {'offset', 'scaling', 'shs', 'nail_3d_mask'}:
                    out[k] = torch.cat([v[0]], dim=0)
                else:
                    out[k] = torch.concat(v, dim=1)
                    if k in {"comp_rgb", "comp_mask", "comp_depth", "comp_obj"}:
                        out[k] = out[k][0].permute(0, 2, 3, 1)
            else:
                out[k] = v

        render_image_bs = out['comp_rgb']
        render_mask_bs = out['comp_mask'].repeat(1, 1, 1, 3) if 'comp_mask' in out else None
        gau_scaling = out.get('scaling', None)
        nail_3d_mask = out.get('nail_3d_mask', None)

        render_image_bs = render_image_bs.reshape(bs, n, h, w, c)
        render_mask_bs = render_mask_bs.reshape(bs, n, h, w, c) if render_mask_bs is not None else None

        return {
            'gt_image_bs': gt_image_bs,
            'gt_mask_bs': gt_mask_bs,
            'bound_mask_bs': bound_mask_bs,
            'render_image_bs': render_image_bs,
            'render_mask_bs': render_mask_bs,
            'latent_points': latent_points,
            'vis_masks': vis_masks,
            'scaling': gau_scaling,
            'nail_3d_mask': nail_3d_mask,
        }

    def finetune_edit(self, batch=None, scaler=None, iter=None, writer=None, pbar=None, total_iters=2000,
                      mask_weight: float = 5.0, use_lora: bool = True):
        """
        Auto 2-stage editing finetune:
          - Stage1 (first 5% iters): optimize base features under edit mask only.
          - Stage2 (remaining 95%): freeze base features, optimize renderer with visibility masking.

        Args:
            use_lora: If True, use LoRA adapters for stage2 (lighter, faster, less overfitting).
                      If False, directly finetune renderer.mlp_net (full capacity, more memory).
        """
        if batch is None:
            raise ValueError("batch is required for finetune_edit")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule stage1/2 switch")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        stage_boundary = int(max(1, total_iters * 0.1))
        stage_boundary = 200
        stage = 'stage1' if (iter is None or iter < stage_boundary) else 'stage2'

        if getattr(self, '_finetune_edit_stage', None) != stage:
            base_feat_params = self._collect_params(['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping'])

            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                        print(f"[finetune_wild] LoRA adapters initialized/ensured (rank=16, alpha=1.0)")

                if renderer is not None:
                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))

                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.mlp_net.parameters()))

                if not stage2_params:
                    raise RuntimeError(
                        "use_lora=True but no LoRA params found. "
                        "Please ensure renderer.lora_mlp exists or set use_lora=False."
                    )
                stage2_name = "LoRA adapters"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())
                    stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.lora_mlp.parameters()))
                if not stage2_params:
                    raise RuntimeError(
                        "No renderer.mlp_net params found for stage2 direct finetuning. "
                        "Try use_lora=True if LoRA adapters are available."
                    )
                stage2_name = "renderer.mlp_net (direct)"

            if stage == 'stage1':
                freeze_all(self.hand_model)
                if not stage2_params:
                    raise RuntimeError("No renderer params found for stage1")
                for p in stage2_params:
                    p.requires_grad = True

                lr_stage1 = 1e-4
                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage1, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-5)
                print(f"[Stage1] Training {len(stage2_params)} renderer {stage2_name} parameters (lr={lr_stage1:.0e})")
            else:
                freeze_all(self.hand_model)
                for p in stage2_params:
                    p.requires_grad = True

                lr_stage2 = 1e-3 if use_lora else 1e-4
                lr_stage2 = 1e-4
                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-5)
                print(f"[Stage2] Training {len(stage2_params)} {stage2_name} parameters (lr={lr_stage2:.0e}) with visibility masking")

            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_edit_stage = stage

        render_pkg = self._render_edit_core(batch)

        gt_image_bs = render_pkg['gt_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)

        edit_mask = bound_mask_bs.float()
        inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)
        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(edit_mask)

        loss_dict = defaultdict(float)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask.repeat(1,1,1,1,3)

        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)

        if stage == 'stage1':
            base_render = render_image_bs
            base_gt = gt_image_bs

            loss_dict['lpips'] += 10 * self.lpips_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
            loss_dict['image_l1'] += 20 * self.pixel_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
            loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
            loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

            scaling_now = render_pkg['scaling']
            loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)
            prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
            if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
                scaling_stability = ((scaling_now - prev_scaling)).mean().abs()
                loss_dict['scaling'] += 1e3 * scaling_stability
            self.prev_scaling_finetune_edit = scaling_now.detach().clone()
        else:
            hand_render = render_image_bs * hand_mask
            hand_gt = gt_image_bs * hand_mask

            if iter >= 2000:
                mask_weight = 50.0
            else:
                mask_weight = 30.0
            loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                gt_image_bs.permute(0, 1, 4, 2, 3))

            loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                gt_image_bs.permute(0, 1, 4, 2, 3))

            edit_render = render_image_bs * edit_mask
            edit_gt = gt_image_bs * edit_mask

            loss_dict['lpips'] += mask_weight * self.lpips_loss(edit_render.permute(0, 1, 4, 2, 3),
                    edit_gt.permute(0, 1, 4, 2, 3))

            loss_dict['image_l1'] += mask_weight * 2 * self.pixel_loss(edit_render.permute(0, 1, 4, 2, 3),
                                edit_gt.permute(0, 1, 4, 2, 3))

            loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
            loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

            scaling_now = render_pkg['scaling']
            loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)
            if loss_dict['scaling'] != 0:
                print(f"\033[94m[Stage2 Scaling Loss] ball_loss={loss_dict['scaling']:.6f}\033[0m")
            prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
            if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
                scaling_stability = ((scaling_now - prev_scaling)).mean().abs()
                loss_dict['scaling'] += 5e3 * scaling_stability
            self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k,v in loss_dict.items():
                writer.add_scalar(f'train/{stage}_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": stage,
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips'])
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print(f"\033[92m[Finetune Complete] Disabled visibility masking in mlp_net\033[0m")

    def finetune_edit_wild(self, batch=None, scaler=None, iter=None, writer=None, pbar=None, total_iters=2000,
                      mask_weight: float = 5.0, use_lora: bool = True):
        """
        Finetune on in-the-wild inputs with the original 2-stage editing logic:
          - Stage1 (~10% iters): train base features only, losses masked by edit region.
          - Stage2 (remaining): freeze base features, train renderer (LoRA or mlp_net),
            enable visibility masking.
        """
        if batch is None:
            raise ValueError("batch is required for finetune_wild")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule stage1/2 switch")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        is_coco = 'coco' in batch[0]['image_name'][0]

        if not is_coco:
            stage_boundary = int(max(1, total_iters * 0.1))
        else:
            stage_boundary = 200

        stage = 'stage1' if (iter is None or iter < stage_boundary) else 'stage2'

        if is_coco and stage == 'stage2':
            stage2a_ratio = 0.3

            stage2_mid = stage_boundary + int(min(400, (total_iters - stage_boundary) * stage2a_ratio))
            stage2_mid = stage_boundary + 200

            stage = 'stage2a' if iter < stage2_mid else 'stage2b'

        if getattr(self, '_finetune_edit_stage', None) != stage:
            base_feat_params = self._collect_params(['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping'])
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)

                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))

                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))

                    stage2_params.extend(list(renderer.mlp_net.parameters()))

                if not stage2_params:
                    raise RuntimeError(
                        "use_lora=True but no LoRA params found. "
                        "Please ensure renderer has ensure_lora_adapters method or set use_lora=False."
                    )
                stage2_name = "LoRA adapters + mlp_net"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())
                    stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.lora_mlp.parameters()))
                if not stage2_params:
                    raise RuntimeError(
                        "No renderer.mlp_net params found for stage2 direct finetuning. "
                        "Try use_lora=True if LoRA adapters are available."
                    )
                stage2_name = "renderer.mlp_net (direct)"

            if stage == 'stage1':
                freeze_all(self.hand_model)

                if not base_feat_params:
                    print("[finetune_t2a:Stage1] No base feature params found; falling back to renderer params.")
                    base_feat_params = stage2_params
                for p in base_feat_params:
                    p.requires_grad = True

                self.hand_model.optimizer = torch.optim.AdamW(base_feat_params, lr=1e-4, weight_decay=0.0)

                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-5)
            else:
                joint_training = True
                joint_training = False

                freeze_all(self.hand_model)

                for p in base_feat_params:
                    p.requires_grad = False
                for p in stage2_params:
                    p.requires_grad = True
                lr_stage2 = 1e-3 if use_lora else 1e-4
                lr_stage2 = 1e-4

                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6)

            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_wild_stage = stage

        render_pkg = self._render_edit_core(batch)

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)

        edit_mask = bound_mask_bs.float()
        inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)

        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(edit_mask)

        loss_dict = defaultdict(float)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask.repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)

        if stage == 'stage1' or stage == 'stage2a':
            base_render = render_image_bs * inv_edit_mask
            base_gt = gt_image_bs * inv_edit_mask

            loss_dict['lpips'] += 10 * self.lpips_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
            loss_dict['image_l1'] += 20 * self.pixel_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
            loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
            loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

            scaling_now = render_pkg['scaling']

            loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)
            prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
            if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
                scaling_stability = ((scaling_now - prev_scaling)).mean().abs()
                loss_dict['scaling'] += 1e3 * scaling_stability
            self.prev_scaling_finetune_edit = scaling_now.detach().clone()
        else:
            hand_render = render_image_bs * hand_mask
            hand_gt = gt_image_bs * hand_mask

            if iter >= 2000:
                mask_weight = 50.0
            else:
                mask_weight = 30.0
            loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                gt_image_bs.permute(0, 1, 4, 2, 3))

            loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                gt_image_bs.permute(0, 1, 4, 2, 3))

            edit_render = render_image_bs * edit_mask
            edit_gt = gt_image_bs * edit_mask

            loss_dict['lpips'] += mask_weight * self.lpips_loss(edit_render.permute(0, 1, 4, 2, 3),
                    edit_gt.permute(0, 1, 4, 2, 3))

            loss_dict['image_l1'] += mask_weight * 2 * self.pixel_loss(edit_render.permute(0, 1, 4, 2, 3),
                                edit_gt.permute(0, 1, 4, 2, 3))

            loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
            loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

            scaling_now = render_pkg['scaling']
            loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)

            prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
            if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
                scaling_stability = ((scaling_now - prev_scaling)).mean().abs()
                loss_dict['scaling'] += 5e3 * scaling_stability
            self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{stage}_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": stage,
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips'])
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print(f"[finetune_wild] Finished. Visibility masking cleared.")

    def finetune_edit_wild_inversion(self, batch=None, scaler=None, iter=None, writer=None, pbar=None,
                                      total_iters=200, lr=1e-2, downscale=1, use_hand_mask=True,
                                      mean_weight=20.0, std_weight=1.0,
                                      use_bias=False, bias_size=32, bias_weight=0.5, bias_tv_weight=0.1):
        """
        Edit-aware inversion: same as finetune_wild_inversion, but masks out the edit
        region (bound_mask) so that the edit pattern does NOT influence global color
        optimization. Only the non-edit hand region is used for color shift/scale fitting.
        """
        import torch.nn.functional as F
        if batch is None:
            raise ValueError("batch is required for finetune_edit_wild_inversion")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        freeze_all(self.hand_model)
        params = self._ensure_color_inversion_params(device, use_bias=use_bias, bias_size=bias_size)

        if not getattr(self, "_edit_inversion_reset_done", False):
            self.hand_model.color_shift.data.zero_()
            self.hand_model.color_scale.data.fill_(1.0)
            renderer = getattr(self.hand_model, "renderer", None)
            gs_net = getattr(renderer, "gs_net", None) if renderer is not None else None
            if gs_net is not None:
                for attr_name in ("color_latent_gamma", "color_latent_beta", "stage1_color_coeff"):
                    attr = getattr(gs_net, attr_name, None)
                    if attr is not None:
                        attr.data.zero_()
                attr = getattr(gs_net, "stage1_color_rgb", None)
                if attr is not None:
                    attr.data.fill_(1.0)
            if use_bias and hasattr(self.hand_model, "color_bias_lowres") and self.hand_model.color_bias_lowres is not None:
                self.hand_model.color_bias_lowres.data.zero_()
            self._edit_inversion_reset_done = True

        def _build_param_groups():
            return [{
                "params": [self.hand_model.color_shift, self.hand_model.color_scale],
                "lr": lr,
            }]

        optimizer_groups = _build_param_groups()
        flat_opt_params = [p for group in optimizer_groups for p in group["params"]]
        if getattr(self, "_edit_inversion_optimizer", None) is None:
            self._edit_inversion_optimizer = torch.optim.Adam(optimizer_groups, weight_decay=0.0)

        render_pkg = self._render_edit_core(batch)
        render_image_bs = render_pkg['render_image_bs']
        gt_image_bs = render_pkg['gt_image_bs'].detach()
        hand_mask = render_pkg.get('gt_mask_bs', None)
        bound_mask_bs = render_pkg['bound_mask_bs']

        if hand_mask is None:
            hand_mask = torch.ones_like(gt_image_bs[..., 0:1])
        if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
            hand_mask = hand_mask[..., 0:1]
        hand_mask = hand_mask.detach()

        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)
        if bound_mask_bs.dim() == 5 and bound_mask_bs.shape[-1] > 1:
            bound_mask_bs = bound_mask_bs[..., 0:1]
        edit_mask = bound_mask_bs.float()
        inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)

        if use_hand_mask:
            mask = hand_mask * inv_edit_mask
        else:
            mask = inv_edit_mask

        def _erode_mask_inv(mask_bhw1, kernel_size=9):
            b, n, h, w, c = mask_bhw1.shape
            mask_flat = mask_bhw1.reshape(b * n, h, w, c).permute(0, 3, 1, 2)
            eroded = -F.max_pool2d(-mask_flat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            return eroded.permute(0, 2, 3, 1).reshape(b, n, h, w, c)

        eroded_mask = _erode_mask_inv(mask, kernel_size=11)
        loss_mask_3c = eroded_mask.repeat(1, 1, 1, 1, 3)

        render_color = render_image_bs
        diff = (render_color - gt_image_bs) * loss_mask_3c

        denom = loss_mask_3c.sum().clamp_min(1e-6)
        loss_main = diff.abs().sum() / denom

        mean_r = (render_color * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        mean_g = (gt_image_bs * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        loss_mean = (mean_r - mean_g).abs().mean()

        var_r = ((render_color - mean_r) ** 2 * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        var_g = ((gt_image_bs - mean_g) ** 2 * loss_mask_3c).sum(dim=(0, 1, 2, 3)) / denom
        loss_std = (torch.sqrt(var_r + 1e-6) - torch.sqrt(var_g + 1e-6)).abs().mean()

        with torch.no_grad():
            if not hasattr(self, '_edit_inv_initial_psnr') or iter == 1:
                mse = ((render_image_bs - gt_image_bs) ** 2 * loss_mask_3c).sum() / denom
                psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
                self._edit_inv_initial_psnr = psnr.item()
                print(f"[Edit Inversion] Initial PSNR (masked, no-edit region): {self._edit_inv_initial_psnr:.2f} dB")
            initial_psnr = self._edit_inv_initial_psnr
            if initial_psnr < 10:
                color_reg_weight = 1e-4
            elif initial_psnr < 14:
                color_reg_weight = 1e-3
            elif initial_psnr < 20:
                color_reg_weight = 1e-2
            else:
                color_reg_weight = 2e-2

        reg = color_reg_weight * (self.hand_model.color_shift ** 2).mean()
        reg += color_reg_weight * ((self.hand_model.color_scale - 1.0) ** 2).mean()

        total_loss = 10.0 * loss_main + mean_weight * loss_mean + std_weight * loss_std + 10.0 * reg

        if iter is not None and iter % 50 == 0:
            current_mse = ((render_color - gt_image_bs) ** 2 * loss_mask_3c).sum() / denom
            current_psnr = 10 * torch.log10(1.0 / current_mse.clamp_min(1e-10)).item()
            cs = self.hand_model.color_shift.detach().view(-1)
            cc = self.hand_model.color_scale.detach().view(-1)
            print(f"[Edit Inv] iter={iter} PSNR={current_psnr:.2f}, shift=[{cs[0]:.4f},{cs[1]:.4f},{cs[2]:.4f}], scale=[{cc[0]:.4f},{cc[1]:.4f},{cc[2]:.4f}]")

        self._edit_inversion_optimizer.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(flat_opt_params, max_norm=1.0)
        if any(p.grad is not None for p in flat_opt_params):
            self._edit_inversion_optimizer.step()

        if writer is not None and iter is not None:
            writer.add_scalar('train/edit_inv_total', total_loss, iter)
            writer.add_scalar('train/edit_inv_loss', loss_main, iter)
            writer.add_scalar('train/edit_inv_mean', loss_mean, iter)

        if pbar is not None:
            pbar.set_postfix({"stage": "edit_inversion", "loss": float(loss_main), "total": float(total_loss)})

    def finetune_edit_wild_stage2(self, batch=None, pseudo_batch=None, scaler=None, iter=None, writer=None, pbar=None,
                                   total_iters=2000, use_lora: bool = True,
                                   edit_unmask_iter: int = 200, edit_mask_weight: float = 30.0,
                                   pseudo_view_weight: float = 0.1):
        """
        Edit-aware Stage2: same loss logic as finetune_wild_stage2_only,
        plus edit mask handling (masked first N iters, then unmasked with enhanced edit region).

        Robustness operations (boundary weighting, erosion, inner ring) are only applied
        to the hand mask. The edit mask is assumed to be accurate and used as-is.

        Sub-stages:
          - masked  (iter < edit_unmask_iter): learn hand texture, edit region excluded from loss
          - unmasked (iter >= edit_unmask_iter): full image + enhanced edit region loss
        """
        if batch is None:
            raise ValueError("batch is required")
        if total_iters is None:
            raise ValueError("total_iters is required")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        sub_stage = 'masked' if (iter is not None and iter < edit_unmask_iter) else 'unmasked'

        if getattr(self, '_finetune_edit_wild_stage2_init', None) != 'done':
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.mlp_net.parameters()))
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())

            if not stage2_params:
                raise RuntimeError("No stage2 params found for edit wild stage2")

            freeze_all(self.hand_model)
            for p in stage2_params:
                p.requires_grad = True

            lr_stage2 = 1e-4
            self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
            self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6
            )
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_edit_wild_stage2_init = 'done'
            print(f"[edit_wild_stage2] Initialized with {len(stage2_params)} params, lr={lr_stage2}")

        if iter == edit_unmask_iter:
            print(f"[edit_wild_stage2] Transitioning from MASKED to UNMASKED @ iter={iter}")

        if sub_stage == 'masked' and pseudo_batch is not None:
            render_batch = pseudo_batch
            self._finetune_edit_stage = None
        elif sub_stage == 'unmasked':
            render_batch = batch
            self._finetune_edit_stage = 'stage2'
        else:
            render_batch = batch
            self._finetune_edit_stage = None

        render_pkg = self._render_edit_core(render_batch)

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']

        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                pass
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(gt_image_bs[..., 0:1])

        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)
        if bound_mask_bs.dim() == 5 and bound_mask_bs.shape[-1] > 1:
            bound_mask_bs = bound_mask_bs[..., 0:1]
        edit_mask = bound_mask_bs.float()
        inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask[..., :1] if hand_mask.shape[-1] > 1 else hand_mask
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        render_mask_bs = render_mask_bs[..., :1] if render_mask_bs.shape[-1] > 1 else render_mask_bs

        device = gt_image_bs.device
        loss_dict = defaultdict(lambda: torch.zeros((), device=device))

        def _build_view_weights(batch_list, _bs, _n, _device, input_weight=10.0, pseudo_weight=1.0):
            weights = torch.full((_bs, _n), pseudo_weight, device=_device)
            for b_idx in range(_bs):
                names = None
                if isinstance(batch_list[b_idx], dict):
                    names = batch_list[b_idx].get('image_name', None)
                if isinstance(names, (list, tuple)) and len(names) >= _n:
                    for v_idx in range(_n):
                        name = str(names[v_idx])
                        if "_pseudo" not in name:
                            weights[b_idx, v_idx] = input_weight
                elif isinstance(names, str):
                    weights[b_idx, :] = input_weight
                else:
                    weights[b_idx, :] = input_weight
            return weights

        view_weights = _build_view_weights(render_batch, bs, n, device, input_weight=10.0, pseudo_weight=1.0)

        use_gt_inner_ring = os.getenv("LHM_STAGE2_USE_GT_INNER_RING", "0") in ("1", "true", "True")
        adaptive_inner_ring = os.getenv("LHM_STAGE2_ADAPTIVE_INNER_RING", "0") in ("1", "true", "True")
        boundary_teacher_blend = float(os.getenv("LHM_STAGE2_BOUNDARY_TEACHER_BLEND", "0.0"))
        boundary_teacher_blend = max(0.0, min(1.0, boundary_teacher_blend))
        inner_ring_kernel = 9
        inner_ring_weight = 0.25
        inner_ring_local_kernel = 9
        inner_ring_color_tau = 1.25

        if iter == 0:
            self._stage2_boundary_teacher = render_image_bs[:, 0, ...].detach().clone()

        def _erode_mask(mask_hwc, kernel_size=5):
            """Erode mask to avoid boundary noise. Input: [H, W, C], Output: [H, W, C]"""
            import torch.nn.functional as F
            if mask_hwc.dim() == 3:
                mask_chw = mask_hwc.permute(2, 0, 1).unsqueeze(0)
            else:
                mask_chw = mask_hwc
            eroded = -F.max_pool2d(-mask_chw, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            return eroded.squeeze(0).permute(1, 2, 0)

        def _soft_erode_mask(mask_hwc, erode_kernel=9, blur_kernel=13, blur_sigma=3.0):
            """Erode then Gaussian-blur for smooth boundary falloff. [H,W,C] -> [H,W,C]"""
            import torch.nn.functional as F
            eroded = _erode_mask(mask_hwc, kernel_size=erode_kernel)
            if eroded.dim() == 3:
                flat = eroded.permute(2, 0, 1).unsqueeze(0)
            else:
                flat = eroded
            pad = blur_kernel // 2
            coords = torch.arange(blur_kernel, dtype=flat.dtype, device=flat.device) - pad
            gauss_1d = torch.exp(-0.5 * (coords / blur_sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel_h = gauss_1d.view(1, 1, blur_kernel, 1)
            kernel_w = gauss_1d.view(1, 1, 1, blur_kernel)
            blurred = F.conv2d(F.pad(flat, (0, 0, pad, pad), mode='replicate'), kernel_h)
            blurred = F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode='replicate'), kernel_w)
            return blurred.squeeze(0).permute(1, 2, 0)

        def _compute_adaptive_inner_ring_weight(gt_rgb, core_mask, boundary_mask):
            import torch.nn.functional as F
            core_mask = core_mask[..., :1]
            boundary_mask = boundary_mask[..., :1]
            if boundary_mask.sum() <= 1e-6:
                return inner_ring_weight * boundary_mask
            gt_rgb_chw = gt_rgb.permute(2, 0, 1).unsqueeze(0)
            core_mask_chw = core_mask.permute(2, 0, 1).unsqueeze(0)
            kernel_area = float(inner_ring_local_kernel * inner_ring_local_kernel)
            local_core_sum = F.avg_pool2d(
                gt_rgb_chw * core_mask_chw, kernel_size=inner_ring_local_kernel,
                stride=1, padding=inner_ring_local_kernel // 2) * kernel_area
            local_core_weight = F.avg_pool2d(
                core_mask_chw, kernel_size=inner_ring_local_kernel,
                stride=1, padding=inner_ring_local_kernel // 2) * kernel_area
            core_count = core_mask.sum().clamp_min(1.0)
            global_core_mean = (gt_rgb * core_mask.repeat(1, 1, 3)).sum(dim=(0, 1), keepdim=True) / core_count
            global_core_var = (((gt_rgb - global_core_mean) ** 2) * core_mask.repeat(1, 1, 3)).sum(dim=(0, 1), keepdim=True) / core_count
            global_core_std = global_core_var.sqrt().clamp_min(0.03)
            local_core_mean = local_core_sum / local_core_weight.clamp_min(1e-6)
            local_core_mean = local_core_mean.squeeze(0).permute(1, 2, 0)
            valid_local = (local_core_weight.squeeze(0).permute(1, 2, 0) > 0.5)
            local_core_mean = torch.where(valid_local.repeat(1, 1, 3), local_core_mean, global_core_mean)
            color_diff = (((gt_rgb - local_core_mean) / global_core_std) ** 2).mean(dim=-1, keepdim=True)
            adaptive_conf = torch.exp(-color_diff / inner_ring_color_tau).clamp(0.0, 1.0)
            return inner_ring_weight * adaptive_conf * boundary_mask

        lpips_total = torch.zeros((), device=device)
        l1_total = torch.zeros((), device=device)
        ssim_total = torch.zeros((), device=device)

        weight_sum = view_weights.sum().clamp_min(1e-6)
        for b_idx in range(bs):
            for v_idx in range(n):
                w = view_weights[b_idx, v_idx]
                render_v = render_image_bs[b_idx, v_idx]
                gt_v = gt_image_bs[b_idx, v_idx]

                if sub_stage == 'masked':
                    em = inv_edit_mask[b_idx, v_idx]
                    em_3c = em.repeat(1, 1, 3) if em.shape[-1] == 1 else em
                    render_v = render_v * em_3c
                    gt_v = gt_v * em_3c
                else:
                    em_3c = None

                render_v_chw = render_v.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                gt_v_chw = gt_v.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

                if v_idx == 0:
                    mask_source = gt_mask_bs[b_idx, v_idx][..., :1] if use_gt_inner_ring else render_mask_bs[b_idx, v_idx][..., :1]

                    soft_weight = _soft_erode_mask(mask_source, erode_kernel=inner_ring_kernel, blur_kernel=13, blur_sigma=3.0)
                    soft_weight_3c = soft_weight.repeat(1, 1, 3)

                    if em_3c is not None:
                        soft_weight_3c = soft_weight_3c * em_3c

                    l1_soft = ((render_v - gt_v).abs() * soft_weight_3c).sum()
                    norm = soft_weight_3c.sum().clamp_min(1e-6)
                    l1_total += w * l1_soft / norm
                else:
                    l1_total += w * self.pixel_loss(render_v_chw, gt_v_chw)

                lpips_total += w * self.lpips_loss(render_v_chw, gt_v_chw)
                ssim_total += w * (1 - ssim(render_v.unsqueeze(0), gt_v.unsqueeze(0)))

        loss_dict['lpips'] += 25 * (lpips_total / weight_sum)
        loss_dict['image_l1'] += 15 * (l1_total / weight_sum)
        loss_dict['ssim'] += 3.0 * (ssim_total / weight_sum)

        if sub_stage == 'unmasked' and edit_mask_weight > 0:
            edit_render = render_image_bs * edit_mask
            edit_gt = gt_image_bs * edit_mask
            loss_dict['edit_lpips'] = edit_mask_weight * self.lpips_loss(
                edit_render.permute(0, 1, 4, 2, 3), edit_gt.permute(0, 1, 4, 2, 3))
            loss_dict['edit_l1'] = edit_mask_weight * 2 * self.pixel_loss(
                edit_render.permute(0, 1, 4, 2, 3), edit_gt.permute(0, 1, 4, 2, 3))

        render_mask_1c = render_mask_bs[..., :1]
        render_masked = render_image_bs * render_mask_1c
        mask_sum = render_mask_1c.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        view_means = (render_masked.sum(dim=(2, 3), keepdim=True) / mask_sum).squeeze(2).squeeze(2)
        global_mean = view_means.mean(dim=1, keepdim=True)
        view_color_var = ((view_means - global_mean) ** 2).mean()
        loss_dict['view_consistency'] = 1.5 * view_color_var

        if sub_stage == 'unmasked' and pseudo_batch is not None and pseudo_view_weight > 0:
            _saved_edit_stage = getattr(self, '_finetune_edit_stage', None)
            self._finetune_edit_stage = None
            pseudo_pkg = self._render_edit_core(pseudo_batch)
            self._finetune_edit_stage = _saved_edit_stage

            pseudo_render = pseudo_pkg['render_image_bs']
            pseudo_gt = pseudo_pkg['gt_image_bs']
            pseudo_render_mask = pseudo_pkg['render_mask_bs']
            pseudo_hand_mask = pseudo_pkg.get('gt_mask_bs', None)
            if pseudo_hand_mask is not None:
                if pseudo_hand_mask.dim() == 5 and pseudo_hand_mask.shape[-1] > 1:
                    pseudo_hand_mask = pseudo_hand_mask[..., 0:1]
            else:
                pseudo_hand_mask = torch.ones_like(pseudo_gt[..., 0:1])
            pseudo_gt_mask = pseudo_hand_mask.repeat(1, 1, 1, 1, 3)
            _, pn, ph, pw, pc = pseudo_gt.shape
            pseudo_render_mask = pseudo_render_mask.reshape(1, pn, ph, pw, pc)

            loss_dict['pseudo_lpips'] = pseudo_view_weight * 10 * self.lpips_loss(
                pseudo_render.permute(0, 1, 4, 2, 3), pseudo_gt.permute(0, 1, 4, 2, 3))
            loss_dict['pseudo_l1'] = pseudo_view_weight * 20 * self.pixel_loss(
                pseudo_render.permute(0, 1, 4, 2, 3), pseudo_gt.permute(0, 1, 4, 2, 3))
            loss_dict['pseudo_mask'] = pseudo_view_weight * self.pixel_loss(
                pseudo_render_mask.permute(0, 1, 4, 2, 3), pseudo_gt_mask.permute(0, 1, 4, 2, 3))
            pseudo_all_render = pseudo_render.reshape(-1, ph, pw, pc)
            pseudo_all_gt = pseudo_gt.reshape(-1, ph, pw, pc)
            loss_dict['pseudo_ssim'] = pseudo_view_weight * (1 - ssim(pseudo_all_render, pseudo_all_gt))

        scaling_now = render_pkg['scaling']
        if torch.is_tensor(scaling_now) and scaling_now.device != device:
            scaling_now = scaling_now.to(device)
        loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)
        prev_scaling = getattr(self, 'prev_scaling_edit_wild_stage2', None)
        if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
            if torch.is_tensor(prev_scaling) and prev_scaling.device != device:
                prev_scaling = prev_scaling.to(device)
            loss_dict['scaling'] += 1e3 * (scaling_now - prev_scaling).mean().abs()
        self.prev_scaling_edit_wild_stage2 = scaling_now.detach().clone()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/edit_stage2_{sub_stage}_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": f"edit_s2_{sub_stage}",
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips']),
            })

    def finetune_wild(self, batch=None, scaler=None, iter=None, writer=None, pbar=None, total_iters=2000,
                      mask_weight: float = 5.0, use_lora: bool = True):
        """
        Finetune on in-the-wild inputs with the original 2-stage editing logic:
          - Stage1 (~10% iters): train base features only, losses masked by edit region.
          - Stage2 (remaining): freeze base features, train renderer (LoRA or mlp_net),
            enable visibility masking.
        """
        if batch is None:
            raise ValueError("batch is required for finetune_wild")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule stage1/2 switch")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        is_coco = 'coco' in batch[0]['image_name'][0]

        if not is_coco:
            stage_boundary = int(max(1, total_iters * 0.1))
        else:
            stage_boundary = 200

        stage = 'stage1' if (iter is None or iter < stage_boundary) else 'stage2'

        if total_iters >= 1000 and stage == 'stage2':
            stage2a_ratio = 0.3

            stage2_mid = stage_boundary + int(min(400, (total_iters - stage_boundary) * stage2a_ratio))
            stage2_mid = stage_boundary + 800

            stage = 'stage2a' if iter < stage2_mid else 'stage2b'

        if getattr(self, '_finetune_wild_stage', None) != stage:
            base_feat_params = self._collect_params(['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping'])
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                        print(f"[finetune_wild] LoRA adapters initialized/ensured (rank=16, alpha=1.0)")

                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                        print(f"[finetune_wild] Added {len(list(renderer.lora_mlp.parameters()))} lora_mlp params")

                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                        print(f"[finetune_wild] Added {len(list(renderer.lora_gs.parameters()))} lora_gs params")

                    stage2_params.extend(list(renderer.mlp_net.parameters()))
                    print(f"[finetune_wild] Added {len(list(renderer.mlp_net.parameters()))} mlp_net params")

                if not stage2_params:
                    raise RuntimeError(
                        "use_lora=True but no LoRA params found. "
                        "Please ensure renderer has ensure_lora_adapters method or set use_lora=False."
                    )
                stage2_name = "LoRA adapters + mlp_net"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())
                    stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.lora_mlp.parameters()))
                if not stage2_params:
                    raise RuntimeError(
                        "No renderer.mlp_net params found for stage2 direct finetuning. "
                        "Try use_lora=True if LoRA adapters are available."
                    )
                stage2_name = "renderer.mlp_net (direct)"

            if stage == 'stage1':
                freeze_all(self.hand_model)

                if not base_feat_params:
                    print("[finetune_t2a:Stage1] No base feature params found; falling back to renderer params.")
                    base_feat_params = stage2_params
                for p in base_feat_params:
                    p.requires_grad = True

                self.hand_model.optimizer = torch.optim.AdamW(base_feat_params, lr=1e-4, weight_decay=0.0)

                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-5)
                print(f"[finetune_wild:Stage1] Training {len(base_feat_params)} base feature parameters")
            else:
                freeze_all(self.hand_model)
                for p in base_feat_params:
                    p.requires_grad = False
                for p in stage2_params:
                    p.requires_grad = True
                lr_stage2 = 1e-3 if use_lora else 1e-4
                lr_stage2 = 1e-4

                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6)
                print(f"[finetune_wild:Stage2] Training {len(stage2_params)} {stage2_name} (lr={lr_stage2:.0e})")
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_wild_stage = stage

        render_pkg = self._render_edit_core(batch)

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)

        if 'coco' in batch[0]['image_name'][0]:
            edit_mask = bound_mask_bs.float()
            inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)
        else:
            edit_mask = None
            inv_edit_mask = None

        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(edit_mask)

        loss_dict = defaultdict(float)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask.repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)
        if is_coco:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-edit-render.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)
        else:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-render.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-gt.jpg'), gt_image_bs[0, 0, ...], rgb2bgr=True)

        if stage == 'stage1':
            loss_dict['image_l1'] = ((torch.abs(render_image_bs - gt_image_bs) + 1e-5) ** 0.3).mean()
        else:
                    loss_dict['lpips'] += 10 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                    gt_image_bs.permute(0, 1, 4, 2, 3))

                    loss_dict['image_l1'] += 20 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                            gt_image_bs.permute(0, 1, 4, 2, 3))

                    loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
                    loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

        scaling_now = render_pkg['scaling']
        loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)

        prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
        if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
            scaling_stability = ((scaling_now - prev_scaling)).mean().abs()

            loss_dict['scaling'] += 1e3 * scaling_stability

        self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{stage}_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": stage,
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips'])
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print(f"[finetune_wild] Finished. Visibility masking cleared.")

    def finetune_wild_with_boundary_color(self, batch=None, scaler=None, iter=None, writer=None, pbar=None,
                                          total_iters=2000, mask_weight: float = 5.0, use_lora: bool = True,
                                          boundary_threshold_low: float = 5e-3, boundary_threshold_high: float = 5e-2,
                                          boundary_knn: int = 16, boundary_weight: float = 1.0,
                                          invisible_weight: float = 0.5):
        """
        Enhanced finetune with boundary point color smoothing and invisible point color regularization.

        Extends finetune_wild with:
          - Boundary point (B): Color smoothing towards weighted average of nearby visible points in 3D space
          - Invisible point (O): Color change constrained to match overall visible point color trend

        Args:
            batch: Input batch data
            scaler: Gradient scaler for AMP
            iter: Current iteration
            writer: TensorBoard writer
            pbar: Progress bar
            total_iters: Total training iterations
            mask_weight: Weight for masked losses
            use_lora: Whether to use LoRA adapters
            boundary_threshold_low: Lower threshold for boundary region (5e-3)
            boundary_threshold_high: Upper threshold for boundary region (5e-2)
            boundary_knn: Number of nearest visible points to use for boundary smoothing
            boundary_weight: Weight for boundary color smoothing loss
            invisible_weight: Weight for invisible point color regularization
        """
        if batch is None:
            raise ValueError("batch is required for finetune_wild_with_boundary_color")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule stage1/2 switch")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        is_coco = 'coco' in batch[0]['image_name'][0]
        is_coco = True

        if not is_coco:
            stage_boundary = int(max(1, total_iters * 0.1))
        else:
            stage_boundary = 2

        stage = 'stage1' if (iter is None or iter < stage_boundary) else 'stage2'

        if is_coco and stage == 'stage2':
            stage2a_ratio = 0.4

            stage2_mid = stage_boundary + int((total_iters - stage_boundary) * stage2a_ratio)

            stage = 'stage2a' if iter < stage2_mid else 'stage2b'

        if getattr(self, '_finetune_wild_stage', None) != stage:
            base_feat_params = self._collect_params(['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping'])
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                        print(f"[finetune_wild_boundary] LoRA adapters initialized/ensured (rank=16, alpha=1.0)")

                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.mlp_net.parameters()))

                if not stage2_params:
                    raise RuntimeError("use_lora=True but no LoRA params found")
                stage2_name = "LoRA adapters + mlp_net"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())
                    stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.lora_mlp.parameters()))
                if not stage2_params:
                    raise RuntimeError("No renderer.mlp_net params found for stage2")
                stage2_name = "renderer.mlp_net (direct)"

            if stage == 'stage1':
                freeze_all(self.hand_model)
                if not base_feat_params:
                    print("[finetune_t2a:Stage1] No base feature params found; falling back to renderer params.")
                    base_feat_params = stage2_params
                for p in base_feat_params:
                    p.requires_grad = True

                self.hand_model.optimizer = torch.optim.AdamW(base_feat_params, lr=1e-4, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6)
                print(f"[finetune_wild_boundary:Stage1] Training {len(base_feat_params)} base feature parameters")
            elif stage == 'stage2a':
                freeze_all(self.hand_model)
                for p in base_feat_params:
                    p.requires_grad = False
                for p in stage2_params:
                    p.requires_grad = True
                lr_stage2 = 1e-3 if use_lora else 1e-4

                lr_stage2 = 1e-4

                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-4)
                print(f"[finetune_wild_boundary:Stage2] Training {len(stage2_params)} {stage2_name} (lr={lr_stage2:.0e})")
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_wild_stage = stage

        render_pkg = self._render_edit_core(batch)
        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)

        if 'coco' in batch[0]['image_name'][0]:
            edit_mask = bound_mask_bs.float()
            inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)
        else:
            edit_mask = None
            inv_edit_mask = None

        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(edit_mask) if edit_mask is not None else torch.ones_like(gt_image_bs[..., :1])

        loss_dict = defaultdict(float)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask[..., :1] if hand_mask.shape[-1] > 1 else hand_mask
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        render_mask_bs = render_mask_bs[..., :1] if render_mask_bs.shape[-1] > 1 else render_mask_bs
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)

        def _build_view_weights(batch_list, bs, n, device, input_weight=4.0, pseudo_weight=1.0):
            weights = torch.full((bs, n), pseudo_weight, device=device)
            for b_idx in range(bs):
                names = None
                if isinstance(batch_list[b_idx], dict):
                    names = batch_list[b_idx].get('image_name', None)
                if isinstance(names, (list, tuple)) and len(names) >= n:
                    for v_idx in range(n):
                        name = str(names[v_idx])
                        if "_pseudo" not in name:
                            weights[b_idx, v_idx] = input_weight
                elif isinstance(names, str):
                    weights[b_idx, :] = input_weight
                else:
                    weights[b_idx, :] = input_weight
            return weights

        view_weights = _build_view_weights(batch, bs, n, gt_image_bs.device)

        if is_coco:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-boundary-edit-render.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)

            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-boundary-edit-gt.jpg'), gt_image_bs[0, 0, ...], rgb2bgr=True)
        else:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-boundary-render.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-boundary-gt.jpg'), gt_image_bs[0, 0, ...], rgb2bgr=True)

        loss_dict['lpips'] += 10 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                                    gt_image_bs.permute(0, 1, 4, 2, 3))
        loss_dict['image_l1'] += 20 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                                       gt_image_bs.permute(0, 1, 4, 2, 3))
        loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                                                 gt_mask_bs.permute(0, 1, 4, 2, 3))
        loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

        if 'shs' in render_pkg and 'vis_prob' in render_pkg and (stage == 'stage2a' or stage == 'stage1'):
            shs_now = render_pkg['shs']

            if shs_now is not None and shs_now.dim() == 4:
                shs_now = shs_now.squeeze(2)

            vis_prob = render_pkg['vis_prob']

            n_vertex = vis_prob.shape[1]
            n_total = shs_now.shape[1]

            cano_pts = batch[0]['big_pose_world_vertex'][0].unsqueeze(0).to('cuda')
            if cano_pts.shape[0] == 1:
                cano_pts = cano_pts.expand(bs, -1, -1)
            cano_pts_vertex = cano_pts[:, :n_vertex, :]

            faces = self.hand_model.renderer.mano_model.mano.faces
            faces_tensor = torch.as_tensor(faces, device=cano_pts_vertex.device, dtype=torch.long)
            n_face_expected = faces_tensor.shape[0]
            n_face_use = max(0, min(n_face_expected, n_total - n_vertex))
            if n_face_use > 0:
                v0 = cano_pts_vertex[:, faces_tensor[:n_face_use, 0]]
                v1 = cano_pts_vertex[:, faces_tensor[:n_face_use, 1]]
                v2 = cano_pts_vertex[:, faces_tensor[:n_face_use, 2]]
                cano_face_centers = (v0 + v1 + v2) / 3.0
                cano_pts_all = torch.cat([cano_pts_vertex, cano_face_centers], dim=1)
            else:
                cano_pts_all = cano_pts_vertex

            boundary_mask, visible_mask, invisible_mask = self.get_boundary_points(
                vis_prob,
                threshold_low=boundary_threshold_low,
                threshold_high=boundary_threshold_high,
            )

            vis_mask_all_list = []
            for b_idx in range(bs):
                verts_cam_single = batch[b_idx]['verts_cam'][0, ...]
                nail_image_single = batch[b_idx]['nail_image'][0, ...]
                verts_world_single = batch[b_idx]['world_vertex'][0, ...]
                _, vis_mask_all_b = self.vis_pts_with_face_centers(
                    batch[b_idx],
                    verts_cam_single.unsqueeze(0),
                    nail_image_single,
                    verts_world_single,
                    return_mask=True,
                    threshold=0.5,
                )
                vis_mask_all_list.append(vis_mask_all_b)

            vis_mask_all = torch.stack(vis_mask_all_list, dim=0) if vis_mask_all_list else None
            if vis_mask_all is not None:
                if vis_mask_all.shape[1] > n_total:
                    vis_mask_all = vis_mask_all[:, :n_total]
                elif vis_mask_all.shape[1] < n_total:
                    pad = torch.zeros((bs, n_total - vis_mask_all.shape[1]), dtype=vis_mask_all.dtype, device=vis_mask_all.device)
                    vis_mask_all = torch.cat([vis_mask_all, pad], dim=1)

            visible_mask_all = (vis_mask_all > 0.5) if vis_mask_all is not None else None
            invisible_mask_all = (~visible_mask_all) if visible_mask_all is not None else None

            if n_face_use > 0:
                boundary_face_mask = boundary_mask[:, faces_tensor[:n_face_use]].any(dim=2)
                boundary_mask_all = torch.cat([boundary_mask, boundary_face_mask], dim=1)
            else:
                boundary_mask_all = boundary_mask

            if not hasattr(self, 'prev_shs_boundary_finetune'):
                self.prev_shs_boundary_finetune = None

            prev_shs = self.prev_shs_boundary_finetune
            if prev_shs is not None:
                if prev_shs.dim() == 4:
                    prev_shs = prev_shs.squeeze(2)
                if prev_shs.shape[1] > n_total:
                    prev_shs = prev_shs[:, :n_total, :]
                elif prev_shs.shape[1] < n_total:
                    pad = torch.zeros((bs, n_total - prev_shs.shape[1], prev_shs.shape[2]), dtype=prev_shs.dtype, device=prev_shs.device)
                    prev_shs = torch.cat([prev_shs, pad], dim=1)

            if boundary_mask_all is not None and visible_mask_all is not None and boundary_mask_all.any() and visible_mask_all.any():
                for b_idx in range(bs):
                    visible_pts_idx = visible_mask_all[b_idx].nonzero(as_tuple=True)[0]
                    boundary_pts_idx = boundary_mask_all[b_idx].nonzero(as_tuple=True)[0]

                    if len(visible_pts_idx) > 0 and len(boundary_pts_idx) > 0:
                        visible_cano = cano_pts_all[b_idx, visible_pts_idx]
                        boundary_cano = cano_pts_all[b_idx, boundary_pts_idx]

                        dists = torch.cdist(boundary_cano, visible_cano)

                        k = min(boundary_knn, len(visible_pts_idx))
                        knn_dists, knn_indices = torch.topk(dists, k=k, dim=1, largest=False)

                        weights = 1.0 / (knn_dists + 1e-6)
                        weights = weights / weights.sum(dim=1, keepdim=True)

                        shs_visible = shs_now[b_idx, visible_pts_idx]

                        knn_colors_list = []
                        for i in range(len(boundary_pts_idx)):
                            knn_idx_for_point = knn_indices[i]
                            knn_colors_for_point = shs_visible[knn_idx_for_point]
                            knn_colors_list.append(knn_colors_for_point)
                        knn_colors = torch.stack(knn_colors_list, dim=0)

                        weighted_colors = (knn_colors * weights.unsqueeze(-1)).sum(dim=1)

                        boundary_colors = shs_now[b_idx, boundary_pts_idx]

                        boundary_color_diff = boundary_colors - weighted_colors
                        loss_dict['boundary_color'] += boundary_weight * (boundary_color_diff ** 2).mean()

            if invisible_mask_all is not None and invisible_mask_all.any() and prev_shs is not None:
                visible_shs_now = shs_now[visible_mask_all].reshape(-1, shs_now.shape[-1])
                prev_visible_shs = prev_shs[visible_mask_all].reshape(-1, prev_shs.shape[-1])

                if len(visible_shs_now) > 0:
                    color_delta_trend = (visible_shs_now - prev_visible_shs).mean(dim=0, keepdim=True)
                    color_delta_std = (visible_shs_now - prev_visible_shs).std(dim=0, keepdim=True)

                    invisible_shs_now = shs_now[invisible_mask_all].reshape(-1, shs_now.shape[-1])
                    prev_invisible_shs = prev_shs[invisible_mask_all].reshape(-1, prev_shs.shape[-1])

                    if len(invisible_shs_now) > 0:
                        invisible_color_delta = invisible_shs_now - prev_invisible_shs

                        trend_deviation = (invisible_color_delta - color_delta_trend).abs()

                        max_deviation = color_delta_std * 0.3

                        violation = (trend_deviation - max_deviation).clamp(min=0.0)
                        violation = trend_deviation

                        invisible_weight = 10
                        invisible_weight = 1e3
                        loss_dict['invisible_color'] += invisible_weight * (violation ** 2).mean()

            self.prev_shs_boundary_finetune = shs_now.detach().clone()

        scaling_now = render_pkg['scaling']

        loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)

        prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
        if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
            scaling_stability = ((scaling_now - prev_scaling)).mean().abs()
            loss_dict['scaling'] += 1e3 * scaling_stability
        self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        if 'invisible_mask_all' in locals() and invisible_mask_all is not None:
            scaling_all = scaling_now.reshape(-1, 3)
            scale_mag_all = scaling_all.norm(dim=1)

            global_min_threshold = 0.004
            global_min_threshold = 0.001
            global_scale_deficit = (global_min_threshold - scale_mag_all).clamp(min=0.0)
            loss_dict['scaling_global_lower_bound'] += 5e1 * (global_scale_deficit ** 2).mean()

            if invisible_mask_all.any():
                invisible_scaling = scaling_now[invisible_mask_all].reshape(-1, 3)
                invisible_scale_mag = invisible_scaling.norm(dim=1)

                invisible_min_threshold = 0.01
                invisible_min_threshold = 0.005
                invisible_scale_deficit = (invisible_min_threshold - invisible_scale_mag).clamp(min=0.0)
                loss_dict['invisible_scaling_lower_bound'] += 1e4 * (invisible_scale_deficit ** 2).mean()

            if 'visible_mask_all' in locals() and visible_mask_all is not None and visible_mask_all.any():
                visible_scaling = scaling_now[visible_mask_all].reshape(-1, 3)
                visible_scale_mag = visible_scaling.norm(dim=1)

                visible_soft_threshold = 0.004
                visible_soft_threshold = 0.001
                visible_scale_deficit = (visible_soft_threshold - visible_scale_mag).clamp(min=0.0)
                loss_dict['visible_scaling_soft_bound'] += 5e1 * (visible_scale_deficit ** 2).mean()

            if iter is not None and iter % 20 == 0:
                n_invisible_too_small = (invisible_scale_mag < invisible_min_threshold).sum().item() if invisible_mask_all.any() else 0
                n_visible_too_small = (visible_scale_mag < visible_soft_threshold).sum().item() if 'visible_mask_all' in locals() and visible_mask_all is not None and visible_mask_all.any() else 0
                print(f"\n[Multi-level Scaling Constraints @ iter={iter}]")
                print(f"  Global: scale_mag min/max/mean={scale_mag_all.min():.6f}/{scale_mag_all.max():.6f}/{scale_mag_all.mean():.6f}")
                if invisible_mask_all.any():
                    print(f"  Invisible: {len(invisible_scale_mag)} points, N_too_small={n_invisible_too_small}")
                    print(f"    scale_mag min/max/mean={invisible_scale_mag.min():.6f}/{invisible_scale_mag.max():.6f}/{invisible_scale_mag.mean():.6f}")
                    print(f"    loss_invisible_lower_bound={loss_dict['invisible_scaling_lower_bound']:.6f}")
                if 'visible_mask_all' in locals() and visible_mask_all is not None and visible_mask_all.any():
                    print(f"  Visible: {len(visible_scale_mag)} points, N_too_small={n_visible_too_small}")
                    print(f"    scale_mag min/max/mean={visible_scale_mag.min():.6f}/{visible_scale_mag.max():.6f}/{visible_scale_mag.mean():.6f}")
                    print(f"    loss_visible_soft_bound={loss_dict['visible_scaling_soft_bound']:.6f}")
                print(f"  loss_global_lower_bound={loss_dict['scaling_global_lower_bound']:.6f}")

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{stage}_boundary_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": stage,
                "L1": float(loss_dict['image_l1']),

                "invisible": float(loss_dict.get('invisible_color', 0.0)),
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print(f"[finetune_wild_with_boundary_color] Finished. Visibility masking cleared.")

    def _t2a_geometry_regularization(self, render_pkg, render_mask_bs, iter=None):
        """Optional, differentiable geometry smoothing for text-to-avatar fine-tuning."""
        enabled = os.environ.get("LHM_T2A_GEOM_REG", "0").lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return {}

        import torch.nn.functional as F

        depth = render_pkg.get('render_depth_bs', None)
        if not torch.is_tensor(depth) or render_mask_bs is None:
            return {}

        depth_curv_weight = float(os.environ.get("LHM_T2A_DEPTH_CURV_WEIGHT", "0.0"))
        normal_smooth_weight = float(os.environ.get("LHM_T2A_NORMAL_SMOOTH_WEIGHT", "0.0"))
        if depth_curv_weight <= 0.0 and normal_smooth_weight <= 0.0:
            return {}

        mask_thresh = float(os.environ.get("LHM_T2A_GEOM_MASK_THRESH", "0.05"))
        edge_beta = float(os.environ.get("LHM_T2A_GEOM_EDGE_BETA", "80.0"))
        depth_scale = float(os.environ.get("LHM_T2A_GEOM_DEPTH_SCALE", "50.0"))
        start_iter = int(os.environ.get("LHM_T2A_GEOM_START_ITER", "0"))
        if iter is not None and iter + 1 < start_iter:
            return {}
        ramp_iters = max(1.0, float(os.environ.get("LHM_T2A_GEOM_RAMP_ITERS", "100.0")))
        ramp_pos = 1.0 if iter is None else max(0.0, float(iter + 1 - start_iter))
        ramp = 1.0 if iter is None else min(1.0, ramp_pos / ramp_iters)

        z = depth[..., 0].float()
        valid = (render_mask_bs[..., 0].detach().float() > mask_thresh).float()
        if valid.sum() < 16:
            return {}

        z_scaled = z * depth_scale
        eps = 1e-6

        def weighted_mean(value, weight):
            return (value * weight).sum() / weight.sum().clamp_min(eps)

        dzx = z[..., 1:] - z[..., :-1]
        dzy = z[..., 1:, :] - z[..., :-1, :]
        wx = valid[..., 1:] * valid[..., :-1] * torch.exp(-edge_beta * dzx.detach().abs())
        wy = valid[..., 1:, :] * valid[..., :-1, :] * torch.exp(-edge_beta * dzy.detach().abs())

        losses = {}
        if depth_curv_weight > 0.0:
            ddx = z_scaled[..., 2:] - 2.0 * z_scaled[..., 1:-1] + z_scaled[..., :-2]
            ddy = z_scaled[..., 2:, :] - 2.0 * z_scaled[..., 1:-1, :] + z_scaled[..., :-2, :]
            wx2 = valid[..., 2:] * valid[..., 1:-1] * valid[..., :-2]
            wy2 = valid[..., 2:, :] * valid[..., 1:-1, :] * valid[..., :-2, :]
            wx2 = wx2 * torch.exp(
                -edge_beta
                * ((z[..., 2:] - z[..., 1:-1]).detach().abs()
                   + (z[..., 1:-1] - z[..., :-2]).detach().abs())
            )
            wy2 = wy2 * torch.exp(
                -edge_beta
                * ((z[..., 2:, :] - z[..., 1:-1, :]).detach().abs()
                   + (z[..., 1:-1, :] - z[..., :-2, :]).detach().abs())
            )
            curv_loss = weighted_mean(torch.sqrt(ddx.square() + eps), wx2)
            curv_loss = curv_loss + weighted_mean(torch.sqrt(ddy.square() + eps), wy2)
            losses['geom_depth_curv'] = depth_curv_weight * ramp * curv_loss

        if normal_smooth_weight > 0.0:
            dzdx = F.pad((z_scaled[..., 2:] - z_scaled[..., :-2]) * 0.5, (1, 1, 0, 0))
            dzdy = F.pad((z_scaled[..., 2:, :] - z_scaled[..., :-2, :]) * 0.5, (0, 0, 1, 1))
            normal = F.normalize(
                torch.stack((-dzdx, -dzdy, torch.ones_like(z_scaled)), dim=-1),
                dim=-1,
                eps=eps,
            )
            ndx = 1.0 - (normal[..., 1:, :] * normal[..., :-1, :]).sum(dim=-1).clamp(-1.0, 1.0)
            ndy = 1.0 - (normal[..., 1:, :, :] * normal[..., :-1, :, :]).sum(dim=-1).clamp(-1.0, 1.0)
            normal_loss = weighted_mean(ndx, wx) + weighted_mean(ndy, wy)
            losses['geom_normal_smooth'] = normal_smooth_weight * ramp * normal_loss

        return losses

    def finetune_t2a(self, batch=None, scaler=None, iter=None, writer=None, pbar=None, total_iters=2000,
                      mask_weight: float = 5.0, use_lora: bool = True):
        """
        Finetune on in-the-wild inputs with the original 2-stage editing logic:
          - Stage1 (~10% iters): train base features only, losses masked by edit region.
          - Stage2 (remaining): freeze base features, train renderer (LoRA or mlp_net),
            enable visibility masking.
        """
        if batch is None:
            raise ValueError("batch is required for finetune_wild")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule stage1/2 switch")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        is_coco = 'coco' in batch[0]['image_name'][0]
        is_coco = True

        if not is_coco:
            stage_boundary = int(max(1, total_iters * 0.1))
        else:
            stage_boundary = 200

        stage = 'stage1' if (iter is None or iter < stage_boundary) else 'stage2'

        if is_coco and stage == 'stage2':
            stage2a_ratio = 0.3

            stage2_mid = stage_boundary + int(min(200, (total_iters - stage_boundary) * stage2a_ratio))

            stage = 'stage2a' if iter < stage2_mid else 'stage2b'

        force_stage2b_iter = int(os.environ.get("LHM_T2A_FORCE_STAGE2B_ITER", "0"))
        if force_stage2b_iter > 0 and iter is not None and iter >= force_stage2b_iter:
            stage = 'stage2b'

        if getattr(self, '_finetune_wild_stage', None) != stage:
            base_feat_params = self._collect_params(['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping'])
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                        print(f"[finetune_wild] LoRA adapters initialized/ensured (rank=16, alpha=1.0)")

                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                        print(f"[finetune_wild] Added {len(list(renderer.lora_mlp.parameters()))} lora_mlp params")

                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                        print(f"[finetune_wild] Added {len(list(renderer.lora_gs.parameters()))} lora_gs params")

                    stage2_params.extend(list(renderer.mlp_net.parameters()))
                    print(f"[finetune_wild] Added {len(list(renderer.mlp_net.parameters()))} mlp_net params")

                if not stage2_params:
                    raise RuntimeError(
                        "use_lora=True but no LoRA params found. "
                        "Please ensure renderer has ensure_lora_adapters method or set use_lora=False."
                    )
                stage2_name = "LoRA adapters + mlp_net"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())
                    stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.lora_mlp.parameters()))
                if not stage2_params:
                    raise RuntimeError(
                        "No renderer.mlp_net params found for stage2 direct finetuning. "
                        "Try use_lora=True if LoRA adapters are available."
                    )
                stage2_name = "renderer.mlp_net (direct)"

            if stage == 'stage1':
                freeze_all(self.hand_model)

                if not base_feat_params:
                    print("[finetune_t2a:Stage1] No base feature params found; falling back to renderer params.")
                    base_feat_params = stage2_params
                for p in base_feat_params:
                    p.requires_grad = True

                self.hand_model.optimizer = torch.optim.AdamW(base_feat_params, lr=1e-4, weight_decay=0.0)

                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-5)
                print(f"[finetune_wild:Stage1] Training {len(base_feat_params)} base feature parameters")
            elif stage == 'stage2a':
                joint_training = True
                joint_training = False

                freeze_all(self.hand_model)

                for p in base_feat_params:
                    p.requires_grad = False
                for p in stage2_params:
                    p.requires_grad = True
                lr_stage2 = 1e-3 if use_lora else 1e-4
                lr_stage2 = 1e-4

                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6)
                print(f"[finetune_wild:Stage2] Training {len(stage2_params)} {stage2_name} (lr={lr_stage2:.0e})")
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_wild_stage = stage

        render_pkg = self._render_edit_core(batch)

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)

        if 'coco' in batch[0]['image_name'][0]:
            edit_mask = bound_mask_bs.float()
            inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)
        else:
            edit_mask = None
            inv_edit_mask = None

        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(edit_mask)

        loss_dict = defaultdict(float)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask.repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)
        if is_coco:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-edit-render-2.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-edit-gt-2.jpg'), gt_image_bs[0, 0, ...], rgb2bgr=True)
        else:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-render.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-gt.jpg'), gt_image_bs[0, 0, ...], rgb2bgr=True)

        if edit_mask is None:
            loss_dict['lpips'] += 10 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
            gt_image_bs.permute(0, 1, 4, 2, 3))

            loss_dict['image_l1'] += 20 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                            gt_image_bs.permute(0, 1, 4, 2, 3))

            loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                    gt_mask_bs.permute(0, 1, 4, 2, 3))
            loss_dict['ssim'] += (1 - ssim(all_render, all_gt))
        else:
            if stage == 'stage1' or stage == 'stage2a':
                base_render = render_image_bs * inv_edit_mask
                base_gt = gt_image_bs * inv_edit_mask

                loss_dict['lpips'] += 10 * self.lpips_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
                loss_dict['image_l1'] += 20 * self.pixel_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
                loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
                loss_dict['ssim'] += (1 - ssim(all_render, all_gt))
            else:
                loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                            gt_image_bs.permute(0, 1, 4, 2, 3))

                loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                gt_image_bs.permute(0, 1, 4, 2, 3))

                edit_render = render_image_bs * edit_mask
                edit_gt = gt_image_bs * edit_mask

                loss_dict['lpips'] += mask_weight * self.lpips_loss(edit_render.permute(0, 1, 4, 2, 3),
                    edit_gt.permute(0, 1, 4, 2, 3))

                loss_dict['image_l1'] += mask_weight * 2 * self.pixel_loss(edit_render.permute(0, 1, 4, 2, 3),
                                edit_gt.permute(0, 1, 4, 2, 3))

                loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
                loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

        for reg_name, reg_loss in self._t2a_geometry_regularization(render_pkg, render_mask_bs, iter).items():
            loss_dict[reg_name] += reg_loss

        scaling_now = render_pkg['scaling']
        loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)

        prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
        if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
            scaling_stability = ((scaling_now - prev_scaling)).mean().abs()

            loss_dict['scaling'] += 1e3 * scaling_stability

        self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{stage}_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": stage,
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips']),
                "geom": float(loss_dict.get('geom_depth_curv', 0.0) + loss_dict.get('geom_normal_smooth', 0.0)),
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print(f"[finetune_wild] Finished. Visibility masking cleared.")

    def finetune_wild_stage2_only(self, batch=None, scaler=None, iter=None, writer=None, pbar=None,
                                  total_iters=2000, use_lora: bool = True, enable_vis_mask: bool = True,
                                  enable_scaling_constraint: bool = False, scaling_min_threshold: float = 0.004,
                                  invisible_scaling_threshold: float = 0.01, visible_scaling_threshold: float = 0.003,
                                  enable_pose_refine: bool = False, pose_refine_lr: float = 1e-4,
                                  edit_mask_weight: float = 0.0):
        """
        Stage2-only finetune for in-the-wild inputs:
          - No stage1 (base_feat is assumed to be handled in inversion).
          - Directly finetune renderer (LoRA or mlp_net) from the first iteration.

        Args:
            enable_scaling_constraint: If True, adds multi-level scaling constraints:
                                       - Level 1: Global lower bound for all points
                                       - Level 2: Stricter constraint for invisible points
                                       - Level 3: Soft constraint for visible points
                                       Default: False for backward compatibility.
            scaling_min_threshold: Global minimum scaling magnitude threshold (default: 0.004).
            invisible_scaling_threshold: Minimum scaling for invisible points (default: 0.01, stricter).
            visible_scaling_threshold: Soft minimum scaling for visible points (default: 0.004).
        """
        if batch is None:
            raise ValueError("batch is required for finetune_wild_stage2_only")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule finetune")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        stage = 'stage2'
        if getattr(self, '_finetune_wild_stage2_only', None) != stage:
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                        print("[finetune_wild_stage2_only] LoRA adapters initialized/ensured (rank=16, alpha=1.0)")
                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.mlp_net.parameters()))
                if not stage2_params:
                    raise RuntimeError("use_lora=True but no LoRA params found")
                stage2_name = "LoRA adapters + mlp_net"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())

                if not stage2_params:
                    raise RuntimeError("No renderer.mlp_net params found for stage2")
                stage2_name = "renderer.mlp_net (direct)"

            freeze_all(self.hand_model)

            if getattr(self, 'no_pretrain', False):
                _no_pretrain_modules = [
                    self.hand_model.pcl_embed,
                    self.hand_model.motion_embed_mlp,
                    self.hand_model.adapter,
                    self.hand_model.aggregator,
                    self.hand_model.transformer,
                    self.hand_model.encoder,
                    self.hand_model.renderer,
                ]
                unfreeze_modules(_no_pretrain_modules, train_mode=True)
                stage2_params = []
                for m in _no_pretrain_modules:
                    stage2_params.extend([p for p in m.parameters() if p.requires_grad])
                stage2_name = "checkpoint components (no_pretrain ablation)"
                n_total = sum(p.numel() for p in stage2_params)
                print(f"\033[93m[no_pretrain] Unfroze checkpoint components: {len(stage2_params)} param tensors, {n_total} params total\033[0m")
            else:
                for p in stage2_params:
                    p.requires_grad = True

            if getattr(self, 'no_pretrain', False):
                lr_stage2 = 1e-4
            else:
                lr_stage2 = 1e-3 if use_lora else 1e-4

            if enable_pose_refine:
                pose_refiner = self._init_pose_refiner(device='cuda')
                pose_params = list(pose_refiner.parameters())
                n_pose_params = sum(p.numel() for p in pose_params)
                self.hand_model.optimizer = torch.optim.AdamW([
                    {'params': stage2_params, 'lr': lr_stage2, 'weight_decay': 0.0},
                    {'params': pose_params, 'lr': pose_refine_lr, 'weight_decay': 0.0},
                ], lr=lr_stage2, weight_decay=0.0)
                print(f"[finetune_wild_stage2_only] Training {len(stage2_params)} {stage2_name} (lr={lr_stage2:.0e})"
                      f" + PoseRefiner ({n_pose_params} params, lr={pose_refine_lr:.0e})")
            else:
                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                print(f"[finetune_wild_stage2_only] Training {len(stage2_params)} {stage2_name} (lr={lr_stage2:.0e})")
            self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6
            )
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_wild_stage2_only = stage

        enable_vis_mask = False
        if enable_vis_mask:
            self._finetune_wild_stage = 'stage2b'
        else:
            self._finetune_wild_stage = 'stage2'

        _pose_backups = None
        if enable_pose_refine and hasattr(self, '_pose_refiner'):
            _pose_backups, _all_delta_Rs = self._apply_pose_refinement(batch)

        render_pkg = self._render_edit_core(batch)

        if _pose_backups is not None:
            self._restore_poses(batch, _pose_backups)

            if iter is not None and iter % 50 == 0:
                with torch.no_grad():
                    sample_delta = _all_delta_Rs[0]
                    eye = torch.eye(3, device=sample_delta.device).unsqueeze(0).unsqueeze(0)
                    delta_norm = ((sample_delta - eye) ** 2).sum(dim=(-2, -1)).sqrt().mean().item()
                    print(f"[PoseRefiner @iter={iter}] delta_norm={delta_norm:.6f}")

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(gt_image_bs[..., :1])

        is_text_to_avatar = bool(getattr(self, '_is_text_to_avatar', False))

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask[..., :1] if hand_mask.shape[-1] > 1 else hand_mask
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        render_mask_bs = render_mask_bs[..., :1] if render_mask_bs.shape[-1] > 1 else render_mask_bs
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)

        def _build_view_weights(batch_list, bs, n, device, input_weight=10.0, pseudo_weight=1.0):
            weights = torch.full((bs, n), pseudo_weight, device=device)
            for b_idx in range(bs):
                names = None
                if isinstance(batch_list[b_idx], dict):
                    names = batch_list[b_idx].get('image_name', None)
                if isinstance(names, (list, tuple)) and len(names) >= n:
                    for v_idx in range(n):
                        name = str(names[v_idx])
                        if "_pseudo" not in name:
                            weights[b_idx, v_idx] = input_weight
                elif isinstance(names, str):
                    weights[b_idx, :] = input_weight
                else:
                    weights[b_idx, :] = input_weight
            return weights

        input_weight = 10.0
        if is_text_to_avatar and iter is not None and total_iters is not None:
            if iter >= max(0, total_iters - total_iters * 0.6):
                input_weight = 30.0

        view_weights = _build_view_weights(batch, bs, n, gt_image_bs.device, input_weight=input_weight, pseudo_weight=1.0)

        use_gt_inner_ring = os.getenv("LHM_STAGE2_USE_GT_INNER_RING", "0") in ("1", "true", "True")
        adaptive_inner_ring = os.getenv("LHM_STAGE2_ADAPTIVE_INNER_RING", "0") in ("1", "true", "True")
        boundary_teacher_blend = float(os.getenv("LHM_STAGE2_BOUNDARY_TEACHER_BLEND", "0.0"))
        boundary_teacher_blend = max(0.0, min(1.0, boundary_teacher_blend))
        inner_ring_kernel = 9
        inner_ring_weight = 0.25
        inner_ring_local_kernel = 9
        inner_ring_color_tau = 1.25

        if iter == 0:
            self._cached_pseudo_gt_renders = gt_image_bs[:, 1:, ...].detach().clone() if n > 1 else None
            if self._cached_pseudo_gt_renders is not None:
                print(f"[finetune_wild_stage2_only] Cached {n-1} pseudo-GT views for color consistency")
            self._stage2_boundary_teacher = render_image_bs[:, 0, ...].detach().clone()
            if use_gt_inner_ring:
                if adaptive_inner_ring:
                    mode_name = "gt_inner_ring_adaptive"
                elif boundary_teacher_blend > 0.0:
                    mode_name = f"gt_inner_ring_boundary_teacher_{boundary_teacher_blend:.2f}"
                else:
                    mode_name = "gt_inner_ring"
            else:
                mode_name = "baseline_render_mask_boundary"
            print(f"[finetune_wild_stage2_only] Stage2 texture mode: {mode_name}")

        device = gt_image_bs.device
        loss_dict = defaultdict(lambda: torch.zeros((), device=device))
        lpips_total = torch.zeros((), device=device)
        l1_total = torch.zeros((), device=device)
        mask_total = torch.zeros((), device=device)
        ssim_total = torch.zeros((), device=device)

        def _erode_mask(mask_hwc, kernel_size=5):
            """Erode mask to avoid boundary noise. Input: [H, W, C], Output: [H, W, C]"""
            import torch.nn.functional as F
            if mask_hwc.dim() == 3:
                mask_chw = mask_hwc.permute(2, 0, 1).unsqueeze(0)
            else:
                mask_chw = mask_hwc
            eroded = -F.max_pool2d(-mask_chw, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            return eroded.squeeze(0).permute(1, 2, 0)

        def _soft_erode_mask(mask_hwc, erode_kernel=9, blur_kernel=13, blur_sigma=3.0):
            """Erode then Gaussian-blur for smooth boundary falloff. [H,W,C] -> [H,W,C]
            Core region → weight ~1.0, boundary → smooth ramp to 0."""
            import torch.nn.functional as F
            eroded = _erode_mask(mask_hwc, kernel_size=erode_kernel)
            if eroded.dim() == 3:
                flat = eroded.permute(2, 0, 1).unsqueeze(0)
            else:
                flat = eroded
            pad = blur_kernel // 2
            coords = torch.arange(blur_kernel, dtype=flat.dtype, device=flat.device) - pad
            gauss_1d = torch.exp(-0.5 * (coords / blur_sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel_h = gauss_1d.view(1, 1, blur_kernel, 1)
            kernel_w = gauss_1d.view(1, 1, 1, blur_kernel)
            blurred = F.conv2d(F.pad(flat, (0, 0, pad, pad), mode='replicate'), kernel_h)
            blurred = F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode='replicate'), kernel_w)
            return blurred.squeeze(0).permute(1, 2, 0)

        def _compute_adaptive_inner_ring_weight(gt_rgb, core_mask, boundary_mask):
            import torch.nn.functional as F

            core_mask = core_mask[..., :1]
            boundary_mask = boundary_mask[..., :1]
            if boundary_mask.sum() <= 1e-6:
                return inner_ring_weight * boundary_mask

            gt_rgb_chw = gt_rgb.permute(2, 0, 1).unsqueeze(0)
            core_mask_chw = core_mask.permute(2, 0, 1).unsqueeze(0)
            kernel_area = float(inner_ring_local_kernel * inner_ring_local_kernel)

            local_core_sum = F.avg_pool2d(
                gt_rgb_chw * core_mask_chw,
                kernel_size=inner_ring_local_kernel,
                stride=1,
                padding=inner_ring_local_kernel // 2,
            ) * kernel_area
            local_core_weight = F.avg_pool2d(
                core_mask_chw,
                kernel_size=inner_ring_local_kernel,
                stride=1,
                padding=inner_ring_local_kernel // 2,
            ) * kernel_area

            core_count = core_mask.sum().clamp_min(1.0)
            global_core_mean = (gt_rgb * core_mask.repeat(1, 1, 3)).sum(dim=(0, 1), keepdim=True) / core_count
            global_core_var = (((gt_rgb - global_core_mean) ** 2) * core_mask.repeat(1, 1, 3)).sum(dim=(0, 1), keepdim=True) / core_count
            global_core_std = global_core_var.sqrt().clamp_min(0.03)

            local_core_mean = local_core_sum / local_core_weight.clamp_min(1e-6)
            local_core_mean = local_core_mean.squeeze(0).permute(1, 2, 0)
            valid_local = (local_core_weight.squeeze(0).permute(1, 2, 0) > 0.5)
            local_core_mean = torch.where(valid_local.repeat(1, 1, 3), local_core_mean, global_core_mean)

            color_diff = (((gt_rgb - local_core_mean) / global_core_std) ** 2).mean(dim=-1, keepdim=True)
            adaptive_conf = torch.exp(-color_diff / inner_ring_color_tau).clamp(0.0, 1.0)
            return inner_ring_weight * adaptive_conf * boundary_mask

        weight_sum = view_weights.sum().clamp_min(1e-6)
        for b_idx in range(bs):
            for v_idx in range(n):
                w = view_weights[b_idx, v_idx]
                render_v = render_image_bs[b_idx, v_idx]
                gt_v = gt_image_bs[b_idx, v_idx]
                render_v_chw = render_v.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                gt_v_chw = gt_v.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

                if v_idx == 0:
                    mask_source = gt_mask_bs[b_idx, v_idx][..., :1] if use_gt_inner_ring else render_mask_bs[b_idx, v_idx][..., :1]

                    soft_weight = _soft_erode_mask(mask_source, erode_kernel=inner_ring_kernel, blur_kernel=13, blur_sigma=3.0)
                    soft_weight_3c = soft_weight.repeat(1, 1, 3)

                    l1_soft = ((render_v - gt_v).abs() * soft_weight_3c).sum()
                    norm = soft_weight_3c.sum().clamp_min(1e-6)
                    l1_total += w * l1_soft / norm
                else:
                    l1_total += w * self.pixel_loss(render_v_chw, gt_v_chw)

                lpips_total += w * self.lpips_loss(render_v_chw, gt_v_chw)
                ssim_total += w * (1 - ssim(render_v.unsqueeze(0), gt_v.unsqueeze(0)))

        loss_dict['lpips'] += 25 * (lpips_total / weight_sum)
        loss_dict['image_l1'] += 15 * (l1_total / weight_sum)
        loss_dict['mask_l2'] += (mask_total / weight_sum)
        loss_dict['ssim'] += 3.0 * (ssim_total / weight_sum)

        if edit_mask_weight > 0:
            _bound_mask_bs = render_pkg.get('bound_mask_bs', None)
            if _bound_mask_bs is not None:
                _bm0 = _bound_mask_bs[:, 0, ...]
                if _bm0.dim() == 3:
                    _bm0 = _bm0.unsqueeze(-1)
                if _bm0.dim() == 4 and _bm0.shape[-1] > 1:
                    _bm0 = _bm0[..., 0:1]

                _edit_mask_list = []
                for _bi in range(_bm0.shape[0]):
                    _edit_mask_list.append(_soft_erode_mask(_bm0[_bi], erode_kernel=7, blur_kernel=11, blur_sigma=2.5))
                _edit_mask = torch.stack(_edit_mask_list, dim=0)
                if _edit_mask.sum() > 0:
                    _em3 = _edit_mask.expand_as(render_image_bs[:, 0])
                    edit_render = (render_image_bs[:, 0] * _em3).unsqueeze(1)
                    edit_gt = (gt_image_bs[:, 0] * _em3).unsqueeze(1)
                    loss_dict['edit_lpips'] = edit_mask_weight * self.lpips_loss(
                        edit_render.permute(0, 1, 4, 2, 3), edit_gt.permute(0, 1, 4, 2, 3))
                    loss_dict['edit_l1'] = edit_mask_weight * 2 * self.pixel_loss(
                        edit_render.permute(0, 1, 4, 2, 3), edit_gt.permute(0, 1, 4, 2, 3))
                    if iter is not None and iter % 100 == 0:
                        _em_ratio = _edit_mask.sum() / max(_edit_mask.numel(), 1)
                        print(f"  [edit_mask_enhanced] lpips={float(loss_dict['edit_lpips']):.4f}, "
                              f"l1={float(loss_dict['edit_l1']):.4f}, mask_coverage={_em_ratio:.3f}")

        render_mask = render_mask_bs[..., :1]
        render_masked = render_image_bs * render_mask
        mask_sum = render_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        view_means = (render_masked.sum(dim=(2, 3), keepdim=True) / mask_sum).squeeze(2).squeeze(2)
        global_mean = view_means.mean(dim=1, keepdim=True)
        view_color_var = ((view_means - global_mean) ** 2).mean()
        loss_dict['view_consistency'] = 1.5 * view_color_var

        if hasattr(self, '_cached_pseudo_gt_renders') and self._cached_pseudo_gt_renders is not None and n > 1:
            pseudo_renders_current = render_image_bs[:, 1:, ...]
            cached_renders = self._cached_pseudo_gt_renders
            if pseudo_renders_current.shape == cached_renders.shape:
                pseudo_masks = render_mask_bs[:, 1:, ..., :1]
                pseudo_masks_3c = pseudo_masks.repeat(1, 1, 1, 1, 3)
                color_diff = (pseudo_renders_current - cached_renders).abs() * pseudo_masks_3c
                color_consistency_loss = color_diff.sum() / (pseudo_masks_3c.sum().clamp_min(1e-6))
                loss_dict['invisible_color_consistency'] = 10.0 * color_consistency_loss
                if iter is not None and iter % 50 == 0:
                    print(f"  [invisible_color_consistency] loss={color_consistency_loss.item():.6f}")

        scaling_now = render_pkg['scaling']
        if torch.is_tensor(scaling_now) and scaling_now.device != device:
            scaling_now = scaling_now.to(device)
        loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)
        prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
        if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
            if torch.is_tensor(prev_scaling) and prev_scaling.device != device:
                prev_scaling = prev_scaling.to(device)
            scaling_stability = (scaling_now - prev_scaling).mean().abs()
            loss_dict['scaling'] += 1e3 * scaling_stability
        self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        if enable_scaling_constraint:
            scaling_all = scaling_now.reshape(-1, 3)
            scale_mag_all = scaling_all.norm(dim=1)
            n_total = scaling_now.shape[1] if scaling_now.dim() >= 2 else len(scale_mag_all)

            global_scale_deficit = (scaling_min_threshold - scale_mag_all).clamp(min=0.0)
            loss_dict['scaling_global_lower_bound'] = 5e2 * (global_scale_deficit ** 2).mean()

            vis_mask_all = None
            try:
                vis_mask_all_list = []
                for b_idx in range(bs):
                    if 'verts_cam' in batch[b_idx] and 'nail_image' in batch[b_idx] and 'world_vertex' in batch[b_idx]:
                        verts_cam_single = batch[b_idx]['verts_cam'][0, ...]
                        nail_image_single = batch[b_idx]['nail_image'][0, ...]
                        verts_world_single = batch[b_idx]['world_vertex'][0, ...]
                        _, vis_mask_all_b = self.vis_pts_with_face_centers(
                            batch[b_idx],
                            verts_cam_single.unsqueeze(0),
                            nail_image_single,
                            verts_world_single,
                            return_mask=True,
                            threshold=0.5,
                        )
                        vis_mask_all_list.append(vis_mask_all_b)

                if vis_mask_all_list:
                    vis_mask_all = torch.stack(vis_mask_all_list, dim=0)

                    if vis_mask_all.shape[1] > n_total:
                        vis_mask_all = vis_mask_all[:, :n_total]
                    elif vis_mask_all.shape[1] < n_total:
                        pad = torch.zeros((bs, n_total - vis_mask_all.shape[1]), dtype=vis_mask_all.dtype, device=vis_mask_all.device)
                        vis_mask_all = torch.cat([vis_mask_all, pad], dim=1)
            except Exception as e:
                if iter is not None and iter % 100 == 0:
                    print(f"[finetune_wild_stage2_only] Warning: Failed to compute visibility mask: {e}")
                vis_mask_all = None

            if vis_mask_all is not None:
                visible_mask_all = (vis_mask_all > 0.5).reshape(-1)
                invisible_mask_all = ~visible_mask_all

                if invisible_mask_all.any():
                    invisible_scaling = scale_mag_all[invisible_mask_all]
                    invisible_scale_deficit = (invisible_scaling_threshold - invisible_scaling).clamp(min=0.0)
                    loss_dict['invisible_scaling_lower_bound'] = 5e4 * (invisible_scale_deficit ** 2).mean()

                if visible_mask_all.any():
                    visible_scaling = scale_mag_all[visible_mask_all]
                    visible_scaling_threshold_eff = 0.005 if is_text_to_avatar else visible_scaling_threshold
                    visible_scale_deficit = (visible_scaling_threshold_eff - visible_scaling).clamp(min=0.0)
                    visible_soft_weight = 5e2 if is_text_to_avatar else 5e1
                    loss_dict['visible_scaling_soft_bound'] = visible_soft_weight * (visible_scale_deficit ** 2).mean()

                if iter is not None and iter % 50 == 0:
                    n_invisible = invisible_mask_all.sum().item()
                    n_visible = visible_mask_all.sum().item()
                    n_invisible_too_small = (invisible_scaling < invisible_scaling_threshold).sum().item() if invisible_mask_all.any() else 0
                    visible_scaling_threshold_eff = 0.005 if is_text_to_avatar else visible_scaling_threshold
                    n_visible_too_small = (visible_scaling < visible_scaling_threshold_eff).sum().item() if visible_mask_all.any() else 0
                    print(f"\n[finetune_wild_stage2_only Multi-level Scaling @ iter={iter}]")
                    print(f"  Global: scale_mag min/max/mean={scale_mag_all.min():.6f}/{scale_mag_all.max():.6f}/{scale_mag_all.mean():.6f}")
                    print(f"  Invisible: {n_invisible} points, N_too_small={n_invisible_too_small} (threshold={invisible_scaling_threshold})")
                    if invisible_mask_all.any():
                        print(f"    scale_mag min/max/mean={invisible_scaling.min():.6f}/{invisible_scaling.max():.6f}/{invisible_scaling.mean():.6f}")
                        print(f"    loss_invisible_lower_bound={loss_dict.get('invisible_scaling_lower_bound', 0):.6f}")
                    print(f"  Visible: {n_visible} points, N_too_small={n_visible_too_small} (threshold={visible_scaling_threshold_eff})")
                    if visible_mask_all.any():
                        print(f"    scale_mag min/max/mean={visible_scaling.min():.6f}/{visible_scaling.max():.6f}/{visible_scaling.mean():.6f}")
                        print(f"    loss_visible_soft_bound={loss_dict.get('visible_scaling_soft_bound', 0):.6f}")
                    print(f"  loss_global_lower_bound={loss_dict['scaling_global_lower_bound']:.6f}")
            else:
                if iter is not None and iter % 50 == 0:
                    n_too_small = (scale_mag_all < scaling_min_threshold).sum().item()
                    print(f"\n[finetune_wild_stage2_only Scaling Constraint @ iter={iter}] (visibility unavailable)")
                    print(f"  scale_mag min/max/mean={scale_mag_all.min():.6f}/{scale_mag_all.max():.6f}/{scale_mag_all.mean():.6f}")
                    print(f"  N_points={len(scale_mag_all)}, N_too_small={n_too_small} (threshold={scaling_min_threshold})")
                    print(f"  loss_scaling_global_lower_bound={loss_dict['scaling_global_lower_bound']:.6f}")

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                val = v if torch.is_tensor(v) else torch.tensor(float(v))
                writer.add_scalar(f'train/stage2_only_{k}', val, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": "stage2_only",
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips'])
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print("[finetune_wild_stage2_only] Finished. Visibility masking cleared.")

    def _init_pose_refiner(self, device='cuda'):
        """Initialize the BodyPoseRefiner for MANO hand model (16 joints).
        Following GauHuman: total_bones=16, embedding_size=45 (15 hand joints * 3),
        mlp_width=128, mlp_depth=2.
        """
        if not hasattr(self, '_pose_refiner') or self._pose_refiner is None:
            self._pose_refiner = BodyPoseRefiner(
                total_bones=16,
                embedding_size=3 * 15,
                mlp_width=128,
                mlp_depth=2,
            ).to(device)
            print(f"[PoseRefiner] Initialized: 16 bones, input=45, width=128, depth=2")
        return self._pose_refiner

    def _apply_pose_refinement(self, batch_list):
        """Apply learned pose delta to all views in batch_list.
        Modifies smpl_param['poses'] in-place and returns backup for restoration.

        The BodyPoseRefiner takes hand_pose (45-dim axis-angle for 15 joints) as input,
        outputs delta rotation matrices Rs [B, 15, 3, 3].
        We compose: R_refined = R_delta @ R_original for each joint.
        Root pose (first 3 dims) is kept unchanged.

        Returns:
            backups: list of original poses tensors for restoration
            all_delta_Rs: list of delta rotation matrices for regularization
        """
        from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

        pose_refiner = self._pose_refiner
        backups = []
        all_delta_Rs = []

        for batch_i in batch_list:
            poses = batch_i['smpl_param']['poses']
            original_poses = poses.detach().clone()
            backups.append(original_poses)

            device = poses.device if poses.is_cuda else 'cuda'
            poses_d = poses.detach().to(device)

            n_views = poses_d.shape[0]

            hand_poses = poses_d[:, 3:].clone()

            delta_dict = pose_refiner(hand_poses)
            delta_Rs = delta_dict["Rs"]
            all_delta_Rs.append(delta_Rs)

            hand_poses_reshape = hand_poses.reshape(n_views, 15, 3)
            orig_Rs = axis_angle_to_matrix(hand_poses_reshape)

            refined_Rs = torch.bmm(
                delta_Rs.reshape(-1, 3, 3),
                orig_Rs.reshape(-1, 3, 3),
            ).reshape(n_views, 15, 3, 3)

            refined_hand_poses = matrix_to_axis_angle(refined_Rs).contiguous()
            refined_hand_poses = refined_hand_poses.reshape(n_views, 45)

            refined_poses = torch.cat([poses_d[:, :3], refined_hand_poses], dim=1).clone()
            batch_i['smpl_param']['poses'] = refined_poses

        return backups, all_delta_Rs

    def _restore_poses(self, batch_list, backups):
        """Restore original poses from backup."""
        for batch_i, backup in zip(batch_list, backups):
            batch_i['smpl_param']['poses'] = backup

    def finetune_visible_only(self, batch=None, scaler=None, iter=None, writer=None, pbar=None,
                              total_iters=300, use_lora: bool = True,
                              gradient_mask_invisible: bool = True,
                              pretrain_regularization: float = 0.1):
        """
        New strategy: Only learn from input view with visibility-aware gradient masking.

        Core idea:
        1. Only use the input view (1 view) for loss, NO pseudo-GT views
        2. Mask gradients on invisible regions to preserve pretrained prior
        3. Add regularization to keep features close to pretrained output

        This prevents texture collapse on invisible regions by:
        - Not using rendered pseudo-GT as supervision (which is already wrong)
        - Freezing invisible region features via gradient masking
        - Regularizing towards pretrained prior
        """
        if batch is None:
            raise ValueError("batch is required for finetune_visible_only")
        if total_iters is None:
            raise ValueError("total_iters is required")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        stage = 'visible_only'
        if getattr(self, '_finetune_visible_only_stage', None) != stage:
            stage_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                        print("[finetune_visible_only] LoRA adapters initialized (rank=16, alpha=1.0)")
                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage_params.extend(list(renderer.lora_mlp.parameters()))
                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage_params.extend(list(renderer.lora_gs.parameters()))
                    stage_params.extend(list(renderer.mlp_net.parameters()))
                stage_name = "LoRA + mlp_net"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage_params = list(renderer.mlp_net.parameters())
                stage_name = "mlp_net"

            if not stage_params:
                raise RuntimeError("No params found for finetune_visible_only")

            freeze_all(self.hand_model)
            for p in stage_params:
                p.requires_grad = True

            lr = 5e-4
            self.hand_model.optimizer = torch.optim.AdamW(stage_params, lr=lr, weight_decay=1e-4)
            self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6
            )
            print(f"[finetune_visible_only] Training {len(stage_params)} {stage_name} params (lr={lr:.0e})")
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_visible_only_stage = stage

            if pretrain_regularization > 0:
                with torch.no_grad():
                    render_pkg_pretrain = self._render_edit_core(batch)
                    self._cached_pretrain_render = render_pkg_pretrain['render_image_bs'].detach().clone()
                    print(f"[finetune_visible_only] Cached pretrained render for regularization")

        self._finetune_wild_stage = 'stage2b' if gradient_mask_invisible else None

        input_only_batch = []
        for b in batch:
            b_single = dict(b)

            for key, value in b_single.items():
                if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] > 1:
                    b_single[key] = value[:1]
                elif isinstance(value, (list, tuple)) and len(value) > 1:
                    b_single[key] = [value[0]]

            if 'smpl_param' in b_single:
                smpl = b_single['smpl_param']
                for k, v in smpl.items():
                    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] > 1:
                        smpl[k] = v[:1]
            input_only_batch.append(b_single)

        render_pkg = self._render_edit_core(input_only_batch)

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        hand_mask = render_pkg.get('gt_mask_bs', None)
        vis_masks = render_pkg.get('vis_masks', None)

        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
        else:
            hand_mask = torch.ones_like(gt_image_bs[..., :1])

        bs, n, h, w, c = gt_image_bs.shape
        device = gt_image_bs.device
        loss_dict = defaultdict(lambda: torch.zeros((), device=device))

        render_v = render_image_bs[0, 0]
        gt_v = gt_image_bs[0, 0]
        mask_v = hand_mask[0, 0]

        import torch.nn.functional as F
        mask_chw = mask_v.permute(2, 0, 1).unsqueeze(0)
        eroded = -F.max_pool2d(-mask_chw, kernel_size=9, stride=1, padding=4)
        mask_eroded = eroded.squeeze(0).permute(1, 2, 0)
        mask_eroded_3c = mask_eroded.repeat(1, 1, 3)

        render_masked = render_v * mask_eroded_3c
        gt_masked = gt_v * mask_eroded_3c

        render_chw = render_masked.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        gt_chw = gt_masked.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        loss_dict['lpips'] = 20 * self.lpips_loss(render_chw, gt_chw)
        loss_dict['image_l1'] = 10 * self.pixel_loss(render_chw, gt_chw)
        loss_dict['ssim'] = 2.0 * (1 - ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)))

        if pretrain_regularization > 0 and hasattr(self, '_cached_pretrain_render'):
            cached_render = self._cached_pretrain_render[0, 0]
            current_render = render_v

            if vis_masks is not None:
                vis_weight = 1.0 - mask_eroded
                vis_weight_3c = vis_weight.repeat(1, 1, 3)
                reg_diff = ((current_render - cached_render) ** 2) * (1.0 + 2.0 * vis_weight_3c)
            else:
                reg_diff = (current_render - cached_render) ** 2

            loss_dict['pretrain_reg'] = pretrain_regularization * reg_diff.mean()

        scaling = render_pkg.get('scaling', None)
        if scaling is not None:
            if torch.is_tensor(scaling) and scaling.device != device:
                scaling = scaling.to(device)
            loss_dict['scaling'] = 500 * self.ball_loss(scaling)

            scale_mag = scaling.reshape(-1, 3).norm(dim=1)
            min_scale_deficit = (0.005 - scale_mag).clamp(min=0.0)
            loss_dict['scaling_min'] = 1000 * (min_scale_deficit ** 2).mean()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if gradient_mask_invisible and vis_masks is not None:
            def _mask_invisible_gradients(grad, mask):
                """Mask gradients based on visibility. mask: [N] boolean, True=visible"""
                if grad is None:
                    return grad

                if mask.dim() == 1 and grad.dim() >= 2:
                    mask_expanded = mask.unsqueeze(-1).expand_as(grad[:mask.shape[0]])
                    grad_masked = grad.clone()
                    grad_masked[:mask.shape[0]] = grad[:mask.shape[0]] * mask_expanded.float()
                    return grad_masked
                return grad

            latent_points = render_pkg.get('latent_points', None)
            if latent_points is not None and latent_points.requires_grad:
                vis_bool = vis_masks[0] > 0.5 if vis_masks is not None else None
                if vis_bool is not None:
                    latent_points.register_hook(lambda g: _mask_invisible_gradients(g, vis_bool))

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                val = v if torch.is_tensor(v) else torch.tensor(float(v))
                writer.add_scalar(f'train/visible_only_{k}', val, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": "visible_only",
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips']),
                "reg": float(loss_dict.get('pretrain_reg', 0))
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print("[finetune_visible_only] Finished.")

    def finetune_nail(self, batch=None, scaler=None, iter=None, writer=None, pbar=None, total_iters=2000,
                      mask_weight: float = 5.0, use_lora: bool = True):
        """
        Finetune on in-the-wild inputs with the original 2-stage editing logic:
          - Stage1 (~10% iters): train base features only, losses masked by edit region.
          - Stage2 (remaining): freeze base features, train renderer (LoRA or mlp_net),
            enable visibility masking.
        """
        if batch is None:
            raise ValueError("batch is required for finetune_wild")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule stage1/2 switch")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        is_coco = 'coco' in batch[0]['image_name'][0]
        is_coco = True

        if not is_coco:
            stage_boundary = int(max(1, total_iters * 0.1))
        else:
            stage_boundary = 200

        stage = 'stage1' if (iter is None or iter < stage_boundary) else 'stage2'

        if is_coco and stage == 'stage2':
            stage2a_ratio = 0.3

            stage2_mid = stage_boundary + int(min(200, (total_iters - stage_boundary) * stage2a_ratio))
            stage2_mid = stage_boundary + 200

            stage = 'stage2a' if iter < stage2_mid else 'stage2b'

        if getattr(self, '_finetune_nail_stage', None) != stage:
            base_feat_params = self._collect_params(['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping'])
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if use_lora:
                if renderer is not None:
                    if hasattr(renderer, 'ensure_lora_adapters'):
                        renderer.ensure_lora_adapters(rank=16, alpha=1.0)
                        print(f"[finetune_wild] LoRA adapters initialized/ensured (rank=16, alpha=1.0)")

                    if hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                        print(f"[finetune_wild] Added {len(list(renderer.lora_mlp.parameters()))} lora_mlp params")

                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                        print(f"[finetune_wild] Added {len(list(renderer.lora_gs.parameters()))} lora_gs params")

                    stage2_params.extend(list(renderer.mlp_net.parameters()))
                    print(f"[finetune_wild] Added {len(list(renderer.mlp_net.parameters()))} mlp_net params")

                if not stage2_params:
                    raise RuntimeError(
                        "use_lora=True but no LoRA params found. "
                        "Please ensure renderer has ensure_lora_adapters method or set use_lora=False."
                    )
                stage2_name = "LoRA adapters + mlp_net"
            else:
                if renderer is not None and hasattr(renderer, 'mlp_net'):
                    stage2_params = list(renderer.mlp_net.parameters())
                    stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.lora_mlp.parameters()))
                if not stage2_params:
                    raise RuntimeError(
                        "No renderer.mlp_net params found for stage2 direct finetuning. "
                        "Try use_lora=True if LoRA adapters are available."
                    )
                stage2_name = "renderer.mlp_net (direct)"

            if stage == 'stage1':
                freeze_all(self.hand_model)

                if not base_feat_params:
                    raise RuntimeError("No base feature params found for stage1")
                for p in base_feat_params:
                    p.requires_grad = True

                self.hand_model.optimizer = torch.optim.AdamW(base_feat_params, lr=1e-4, weight_decay=0.0)

                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-5)
                print(f"[finetune_wild:Stage1] Training {len(base_feat_params)} base feature parameters")
            elif stage == 'stage2a':
                joint_training = True
                joint_training = False

                freeze_all(self.hand_model)

                for p in base_feat_params:
                    p.requires_grad = False
                for p in stage2_params:
                    p.requires_grad = True
                lr_stage2 = 1e-3 if use_lora else 1e-4
                lr_stage2 = 1e-4

                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6)
                print(f"[finetune_wild:Stage2] Training {len(stage2_params)} {stage2_name} (lr={lr_stage2:.0e})")
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_nail_stage = stage

        render_pkg = self._render_nail_core(batch)

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)

        nail_mask_tensor = batch[0]['nail_mask'].to('cuda').unsqueeze(0)
        if nail_mask_tensor.dim() == 3:
            nail_mask_tensor = nail_mask_tensor.unsqueeze(0).unsqueeze(-1)
        elif nail_mask_tensor.dim() == 4:
            nail_mask_tensor = nail_mask_tensor.permute(1, 0, 2, 3).unsqueeze(-1)
        edit_mask = nail_mask_tensor.float()
        inv_edit_mask = (1.0 - edit_mask).clamp_min(0.0)

        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(edit_mask)

        loss_dict = defaultdict(float)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask.repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)
        if is_coco:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-edit-render-1.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-edit-gt-1.jpg'), gt_image_bs[0, 0, ...], rgb2bgr=True)
        else:
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-render.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)
            libcore.write_tensor_image(os.path.join('./hand', 'finetune-wild-gt.jpg'), gt_image_bs[0, 0, ...], rgb2bgr=True)

        if edit_mask is None:
            loss_dict['lpips'] += 10 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
            gt_image_bs.permute(0, 1, 4, 2, 3))

            loss_dict['image_l1'] += 20 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                            gt_image_bs.permute(0, 1, 4, 2, 3))

            loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                    gt_mask_bs.permute(0, 1, 4, 2, 3))
            loss_dict['ssim'] += (1 - ssim(all_render, all_gt))
        else:
            if stage == 'stage1' or stage == 'stage2a':
                base_render = render_image_bs * inv_edit_mask
                base_gt = gt_image_bs * inv_edit_mask

                loss_dict['lpips'] += 10 * self.lpips_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
                loss_dict['image_l1'] += 20 * self.pixel_loss(base_render.permute(0,1,4,2,3), base_gt.permute(0,1,4,2,3))
                loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
                loss_dict['ssim'] += (1 - ssim(all_render, all_gt))
            else:
                loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                            gt_image_bs.permute(0, 1, 4, 2, 3))

                loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                gt_image_bs.permute(0, 1, 4, 2, 3))

                edit_render = render_image_bs * edit_mask
                edit_gt = gt_image_bs * edit_mask

                loss_dict['lpips'] += mask_weight * self.lpips_loss(edit_render.permute(0, 1, 4, 2, 3),
                    edit_gt.permute(0, 1, 4, 2, 3))

                loss_dict['image_l1'] += mask_weight * 2 * self.pixel_loss(edit_render.permute(0, 1, 4, 2, 3),
                                edit_gt.permute(0, 1, 4, 2, 3))

                loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))
                loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

        scaling_now = render_pkg['scaling']
        loss_dict['scaling'] += 1e3 * self.ball_loss(scaling_now)

        prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
        if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
            scaling_stability = ((scaling_now - prev_scaling)).mean().abs()

            loss_dict['scaling'] += 1e3 * scaling_stability

        self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{stage}_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": stage,
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips'])
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print(f"[finetune_wild] Finished. Visibility masking cleared.")

    def finetune_interhand(self, batch=None, scaler=None, iter=None, writer=None, pbar=None, total_iters=2000,
                      mask_weight: float = 5.0, use_lora: bool = True):
        """
        Finetune on in-the-wild inputs with the original 2-stage editing logic:
          - Stage1 (~10% iters): train base features only, losses masked by edit region.
          - Stage2 (remaining): freeze base features, train renderer (LoRA or mlp_net),
            enable visibility masking.
        """
        if batch is None:
            raise ValueError("batch is required for finetune_wild")
        if total_iters is None:
            raise ValueError("total_iters is required to schedule stage1/2 switch")

        if not hasattr(self, 'accum_step'):
            self.accum_step = 1

        stage_boundary = int(max(1, total_iters * 0.4))
        stage_boundary = int(max(1, total_iters * 0.1))
        stage = 'stage1' if (iter is None or iter < stage_boundary) else 'stage2'

        if getattr(self, '_finetune_wild_stage', None) != stage:
            base_feat_params = self._collect_params(['vertex_base_feat', 'base_feature', 'base_feat', 'vertex_global_mapping'])
            stage2_params = []
            renderer = getattr(self.hand_model, 'renderer', None)

            if stage == 'stage1':
                freeze_all(self.hand_model)
                if not base_feat_params:
                    raise RuntimeError("No base feature params found for stage1")
                for p in base_feat_params:
                    p.requires_grad = True

                self.hand_model.optimizer = torch.optim.AdamW(base_feat_params, lr=5e-5, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-5)
                print(f"[finetune_wild:Stage1] Training {len(base_feat_params)} base feature parameters")
            else:
                joint_training = True
                joint_training = False

                freeze_all(self.hand_model)
                if use_lora:
                    if renderer is not None and hasattr(renderer, 'lora_mlp') and renderer.lora_mlp is not None:
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                    if hasattr(renderer, 'lora_gs') and renderer.lora_gs is not None:
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                    stage2_params.extend(list(renderer.mlp_net.parameters()))
                    if not stage2_params:
                        raise RuntimeError("use_lora=True but no LoRA params found")
                    stage2_name = 'LoRA adapters'
                    lr_stage2 = 1e-4
                else:
                    if renderer is not None and hasattr(renderer, 'mlp_net') and renderer.mlp_net is not None:
                        stage2_params = list(renderer.mlp_net.parameters())
                        stage2_params.extend(list(renderer.lora_gs.parameters()))
                        stage2_params.extend(list(renderer.lora_mlp.parameters()))
                    if not stage2_params:
                        raise RuntimeError("No renderer.mlp_net params found for stage2")
                    stage2_name = 'renderer.mlp_net (direct)'
                    lr_stage2 = 5e-5

                if joint_training:
                    for p in base_feat_params:
                        p.requires_grad = True
                    stage2_params.extend(base_feat_params)
                    stage2_name += ' + base_feat (joint)'
                    print(f"[finetune_wild:Stage2] JOINT TRAINING enabled: {len(stage2_params)} params ({len(base_feat_params)} base_feat + mlp_net)")
                else:
                    for p in base_feat_params:
                        p.requires_grad = False

                for p in stage2_params:
                    p.requires_grad = True
                self.hand_model.optimizer = torch.optim.AdamW(stage2_params, lr=lr_stage2, weight_decay=0.0)
                self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.hand_model.optimizer, T_max=total_iters, eta_min=1e-6)
                print(f"[finetune_wild:Stage2] Training {len(stage2_params)} {stage2_name} (lr={lr_stage2:.0e})")
            self.hand_model.optimizer.zero_grad(set_to_none=True)
            self._finetune_wild_stage = stage

        render_pkg = self._render_edit_core(batch)

        gt_image_bs = render_pkg['gt_image_bs']
        render_image_bs = render_pkg['render_image_bs']
        bound_mask_bs = render_pkg['bound_mask_bs']
        if bound_mask_bs.dim() == 4:
            bound_mask_bs = bound_mask_bs.unsqueeze(-1)

        edit_mask = None
        inv_edit_mask = None
        hand_mask = render_pkg.get('gt_mask_bs', None)
        if hand_mask is not None:
            if hand_mask.dim() == 5 and hand_mask.shape[-1] > 1:
                hand_mask = hand_mask[..., 0:1]
            elif hand_mask.dim() == 5 and hand_mask.shape[-1] == 1:
                hand_mask = hand_mask
            else:
                hand_mask = None
        if hand_mask is None:
            hand_mask = torch.ones_like(edit_mask)

        loss_dict = defaultdict(float)

        bs, n, h, w, c = gt_image_bs.shape
        all_render_mask = render_pkg['render_mask_bs']
        gt_mask_bs = hand_mask.repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)
        all_render = render_image_bs.reshape(-1, h, w, c)
        all_gt = gt_image_bs.reshape(-1, h, w, c)
        libcore.write_tensor_image(os.path.join('./hand', 'finetune-interhand-render.jpg'), render_image_bs[0, 0, ...], rgb2bgr=True)

        loss_dict['lpips'] += 10 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
            gt_image_bs.permute(0, 1, 4, 2, 3))

        loss_dict['image_l1'] += 20 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                            gt_image_bs.permute(0, 1, 4, 2, 3))

        loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                    gt_mask_bs.permute(0, 1, 4, 2, 3))
        loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

        scaling_now = render_pkg['scaling']
        loss_dict['scaling'] += 0.5 * self.ball_loss(scaling_now)
        prev_scaling = getattr(self, 'prev_scaling_finetune_edit', None)
        if prev_scaling is not None and prev_scaling.shape == scaling_now.shape:
            scaling_stability = ((scaling_now - prev_scaling)).mean().abs()
            loss_dict['scaling'] += 1e4 * scaling_stability
        self.prev_scaling_finetune_edit = scaling_now.detach().clone()

        total_loss = sum(loss_dict.values()) / self.accum_step

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward(retain_graph=True)
        else:
            total_loss.backward(retain_graph=True)

        if (iter is None) or ((iter + 1) % self.accum_step == 0):
            if scaler is not None and scaler.is_enabled():
                scaler.step(self.hand_model.optimizer)
                scaler.update()
            else:
                self.hand_model.optimizer.step()
            if getattr(self.hand_model, 'scheduler', None) is not None:
                self.hand_model.scheduler.step()
            self.hand_model.optimizer.zero_grad()

        if writer is not None and iter is not None:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{stage}_{k}', v, iter)

        if pbar is not None:
            pbar.set_postfix({
                "stage": stage,
                "L1": float(loss_dict['image_l1']),
                "lpips": float(loss_dict['lpips'])
            })

        if iter is not None and iter == total_iters - 1:
            self.hand_model.renderer.edit_mask_mode = False
            self.hand_model.renderer.edit_vis_mask = None
            print(f"[finetune_wild] Finished. Visibility masking cleared.")

    def infer_handavatar(self, batch):
        infer_images = batch['original_image'].permute(0, 2, 3, 1).cuda()
        nail_image = batch['nail_image'].cuda()
        cano_pts = batch['big_pose_world_vertex'][0].unsqueeze(0).to('cuda')
        verts_cam = batch['verts_cam'][0, ...].unsqueeze(0).to('cuda')
        nail_mask = batch['nail_mask'][0, ...].unsqueeze(0).to('cuda')

        vis_mask = self.vis_pts(batch, batch['verts_cam'][0, ..., 2], batch['nail_image'][0, ...]).unsqueeze(0)

        def _select_first_view(tensor: Optional[torch.Tensor]):
            if torch.is_tensor(tensor) and tensor.ndim > 2:
                return tensor[0]
            return tensor

        vert_uv = batch.get('vert_uv') if isinstance(batch, dict) else None
        face_uv = batch.get('face_uv') if isinstance(batch, dict) else None
        face_uv_xy = batch.get('face_uv_xy') if isinstance(batch, dict) else None
        posed_points = batch.get('world_vertex') if isinstance(batch, dict) else None
        cam = batch.get('full_proj_transform') if isinstance(batch, dict) else None
        batch['uv_map'] = {
                    'vert_uv': _select_first_view(vert_uv),
                    'face_uv': _select_first_view(face_uv),
                    'face_uv_xy': _select_first_view(face_uv_xy),
                    'posed_points': _select_first_view(posed_points),
                    'cam': _select_first_view(cam),
                }

        with torch.no_grad():
            latent_points, global_texture_feature, bias_dict = self.hand_model.forward_latent_points(
            vis_mask, nail_image, batch['uv_map'], infer_images.permute(0, 3, 1, 2), camera=None, query_points=cano_pts, posed_points=verts_cam)

            gs_model_list, gs_densify_list, query_points = self.hand_model.renderer.forward_gs(
                gs_hidden_features=latent_points.to('cuda'),
                query_points=cano_pts,
                global_feature=global_texture_feature,
                batches=batch,
                verts_cam=verts_cam,
                nail_image=nail_image,
                color_bias=bias_dict.get("color_bias"),
                opacity_bias=bias_dict.get("opacity_bias"),
            )

        return gs_model_list, gs_densify_list, query_points

    def test_handavatar(self, gs_model_list, gs_densify_list, query_points,
                        batches, idx, iteration, pbar):

        loss_dict = {}
        loss_dict['image_l1'], loss_dict['lpips'], loss_dict['ssim'], loss_dict['psnr'] =  0, 0, 0, 0
        all_gt, all_render = [], []
        gt_image_bs = []
        gt_mask_bs = []

        infer_images = []
        view_idx = 0

        render_res_list = []
        total_render_time = 0.0
        total_rendered_views = 0

        for i in range(len(batches)):
            gt_image = batches[i]['original_image'].permute(0, 2, 3, 1).to('cuda')
            gt_image_bs.append(gt_image)

            num_views = batches[i]['original_image'].shape[0]
            smplx_params={
                k: v.to('cuda') for k, v in batches[i]['smpl_param'].items()
            }

            render_res = self.hand_model.renderer.forward_animate_gs(
                gs_model_list[0],
                gs_densify_list[0],
                query_points[0],

                self.hand_model.renderer.get_single_view_cam(batches[i], view_idx),
                self.hand_model.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                256,
                256,
                self.bg_color,
            )

            total_render_time += render_res.get('render_time', 0.0)
            total_rendered_views += render_res.get('render_views', 0)

            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                if k == 'offset' or k == 'scaling':
                    selected = [v[0]]
                    out[k] = torch.cat(selected, dim=0)
                else:
                    out[k] = torch.concat(v, dim=1)
                    if k in ["comp_rgb", "comp_mask", "comp_depth", "comp_obj"]:
                        out[k] = out[k][0].permute(

                            0, 2, 3, 1
                        )
            else:
                out[k] = v

        render_pkg = out

        all_render = render_pkg['comp_rgb']
        gt_image_bs = torch.stack(gt_image_bs, dim=0)
        bs, n, h, w, c = gt_image_bs.shape
        all_gt = gt_image_bs.reshape(bs*n, h, w, c)
        render_image_bs = all_render.reshape(bs, n, h, w, c)

        for i in range(len(batches)):
            compare = torch.concat([gt_image_bs[i, :].squeeze(), render_image_bs[i, :].squeeze()], dim=1)
            compare = compare.cpu().numpy()
            ssim_ = ssim(all_render[i], all_gt[i])
            psnr_ = (20.0 * torch.log10(1.0 /
                        torch.sqrt(((all_render[i].permute(2,0,1) - all_gt[i].permute(2,0,1))**2      ).mean())))
            lpips_ = self.lpips_loss.test(all_render[i].permute(2,0,1), all_gt[i].permute(2,0,1))
            compare = cv2.putText(compare, f'psnr/ssim/lpips', (20, compare.shape[0] - 50), 0, 0.5, (255, 0, 0))
            img_name = Path(batches[i]['image_name'][0]).stem
            compare = cv2.putText(compare,
                                f"{psnr_:.4f}/{ssim_:.4f}/{lpips_:.4f}",
                                (20, compare.shape[0] - 10), 0, 0.5, (255, 0, 0) )

            n_views_i = render_image_bs.shape[1]
            for view_j in range(n_views_i):
                out_name = f'iter{iteration}-{img_name}-eval{idx}-batch{i}-view{view_j}.jpg'
                libcore.write_tensor_image(
                    os.path.join(self.test_path, out_name),
                    render_image_bs[i, view_j, ...],
                    rgb2bgr=True
                )

                if getattr(self, 'save_test_gt_images', True):
                    gt_name = f'iter{iteration}-{img_name}-eval{idx}-batch{i}-view{view_j}-GT.jpg'
                    libcore.write_tensor_image(
                        os.path.join(self.test_path, gt_name),
                        gt_image_bs[i, view_j, ...],
                        rgb2bgr=True
                    )

            loss_dict['ssim'] += ssim_
            loss_dict['psnr'] += psnr_
            loss_dict['lpips'] += lpips_

        loss_dict['render_time'] = total_render_time
        loss_dict['render_views'] = total_rendered_views

        return loss_dict

    def train_handavatar(self, batch=None, scaler=None, iter=None, writer=None, pbar=None):
        gt_image_bs = []

        gt_mask_bs = []

        infer_images = []
        nail_images = []
        verts_cams = []

        nail_masks = []
        cano_pts_list = []
        ref_view = 0

        loss_dict = {}
        loss_dict['image_l1'], loss_dict['lpips'], loss_dict['mask_l2'],\
        loss_dict['ssim'], loss_dict['offset'], loss_dict['scaling'] = 0, 0, 0, 0, 0, 0

        for i in range(len(batch)):
            batch_i = batch[i]

            gt_image = batch_i['original_image'].permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch_i['bkgd_mask'].to('cuda')
            gt_image_bs.append(gt_image)
            gt_mask_bs.append(bkgd_mask)

            infer_image = batch_i['original_image'][ref_view, ...].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')
            bkgd_mask = batch_i['bkgd_mask'][ref_view, ...].unsqueeze(0).to('cuda')
            infer_image = infer_image * bkgd_mask + self.bg_color * (1 - bkgd_mask)
            infer_images.append(infer_image)
            nail_images.append(batch_i['nail_image'][ref_view, ...].unsqueeze(0).to('cuda'))
            nail_masks.append(batch_i['nail_mask'][ref_view, ...].unsqueeze(0).to('cuda'))
            verts_cams.append(batch_i['verts_cam'][ref_view, ...].unsqueeze(0).to('cuda'))
            cano_pts_list.append(batch_i['big_pose_world_vertex'][ref_view].to('cuda'))

        gt_image_bs = torch.stack(gt_image_bs, dim=0)
        gt_mask_bs = torch.stack(gt_mask_bs, dim=0)

        nail_pts = batch[0]['nail_image'][ref_view, ...].shape[0]

        image = torch.stack(infer_images, dim=0)

        bs, n, h, w, c = gt_image_bs.shape

        nail_images = torch.stack(nail_images, dim=0).view(bs, nail_pts, -1)
        nail_masks = torch.stack(nail_masks, dim=0).squeeze(-1)
        verts_cams = torch.stack(verts_cams, dim=0).view(bs, nail_pts, -1)

        all_gt = gt_image_bs.reshape(bs*n, h, w, c)

        cano_pts = torch.stack(cano_pts_list, dim=0)

        render_pkg = self.hand_model.infer_single_view(

            nail_images,
            nail_masks,
            verts_cams,
            image.reshape(bs, h, w, c),
            batch,
            cano_pts=cano_pts,
            render_bg_colors=self.bg_color,
        )

        gau_num = render_pkg['scaling'].shape[1]
        all_render = render_pkg['comp_rgb']
        render_image_bs = all_render.reshape(bs, n, h, w, c)

        all_render_mask = render_pkg['comp_mask'].repeat(1,1,1,1,3)
        render_mask_bs = all_render_mask.reshape(bs, n, h, w, c)

        loss_dict['ssim'] += (1 - ssim(all_render, all_gt))

        loss_dict['scaling'] += 0.5 * self.ball_loss(render_pkg['scaling'])
        loss_dict['offset'] += self.offset_loss(render_pkg['offset'])

        loss_dict['mask_l2'] += self.pixel_loss(render_mask_bs.permute(0, 1, 4, 2, 3),
                        gt_mask_bs.permute(0, 1, 4, 2, 3))

        loss_dict['lpips'] += 5 * self.lpips_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                        gt_image_bs.permute(0, 1, 4, 2, 3))

        loss_dict['image_l1'] += 10 * self.pixel_loss(render_image_bs.permute(0, 1, 4, 2, 3),
                                                 gt_image_bs.permute(0, 1, 4, 2, 3))

        writer.add_scalar('train/loss_image_l1', loss_dict['image_l1'], iter)
        writer.add_scalar('train/loss_mask_l2', loss_dict['mask_l2'], iter)
        writer.add_scalar('train/loss_lpips', loss_dict['lpips'], iter)
        writer.add_scalar('train/loss_ssim', loss_dict['ssim'], iter)

        pbar.set_postfix({
            "\033[91m#Gau\033[0m": gau_num,
            "L1_loss": f"{loss_dict['image_l1']:.4f}",
            "mask_l2": f"{loss_dict['mask_l2']:.4f}",
            'lpips': f"{loss_dict['lpips']:.4f}",
            'ssim': f"{loss_dict['ssim']:.4f}"
        })

        total_loss = sum(loss_dict.values()) / self.accum_step

        scaler.scale(total_loss).backward(retain_graph=True)

        if (iter+1) % self.accum_step == 0:
            scaler.step(self.hand_model.optimizer)
            self.hand_model.scheduler.step()
            scaler.update()
            self.hand_model.optimizer.zero_grad()

    def test(self, batch, idx, iteration, pbar):
        loss_dict = {}
        loss_dict['image_l1'], loss_dict['lpips'], loss_dict['ssim'], loss_dict['psnr'] =  0, 0, 0, 0
        all_gt, all_render = [], []
        gt_image_bs = []
        gt_mask_bs = []

        infer_images = []
        cano_pts_list = []
        ref_view = 0

        for i in range(len(batch)):
            batch_i = batch[i]

            infer_image = batch_i['original_image'][ref_view, ...].unsqueeze(0).permute(0, 2, 3, 1).to('cuda')
            gt_image = batch_i['original_image'].permute(0, 2, 3, 1).to('cuda')

            gt_image_bs.append(gt_image)

            infer_images.append(infer_image)
            cano_pts_list.append(batch_i['big_pose_world_vertex'][ref_view].to('cuda'))

        num_views = batch[0]['original_image'].shape[0]
        gt_image_bs = torch.stack(gt_image_bs, dim=0)

        image = torch.stack(infer_images, dim=0)
        bs, n, h, w, c = gt_image_bs.shape

        cano_pts = torch.stack(cano_pts_list, dim=0)

        with torch.no_grad():
            render_pkg = self.hand_model.infer_single_view(

                image.reshape(bs, h, w, c),
                batch,
                cano_pts=cano_pts,
                render_bg_colors=self.bg_color,
                )
        all_render = render_pkg['comp_rgb']
        all_gt = gt_image_bs.reshape(bs*n, h, w, c)
        render_image_bs = all_render.reshape(bs, n, h, w, c)

        for i in range(len(batch)):
            compare = torch.concat([gt_image_bs[i, :].squeeze(), render_image_bs[i, :].squeeze()], dim=1)
            compare = compare.cpu().numpy()
            ssim_ = ssim(all_render[i], all_gt[i])
            psnr_ = (20.0 * torch.log10(1.0 /
                        torch.sqrt(((all_render[i].permute(2,0,1) - all_gt[i].permute(2,0,1))**2      ).mean())))
            lpips_ = self.lpips_loss.test(all_render[i].permute(2,0,1), all_gt[i].permute(2,0,1))
            compare = cv2.putText(compare, f'psnr/ssim/lpips', (20, compare.shape[0] - 50), 0, 0.8, (255, 0, 0))
            compare = cv2.putText(compare,
                                f"{psnr_:.4f}/{ssim_:.4f}/{lpips_:.4f}",
                                (20, compare.shape[0] - 10), 0, 0.8, (255, 0, 0) )

            libcore.write_tensor_image(os.path.join(self.test_path, f'iter{iteration}-{idx}-batch{i}.jpg'), torch.tensor(compare), rgb2bgr=True)

            loss_dict['ssim'] += ssim_
            loss_dict['psnr'] += psnr_
            loss_dict['lpips'] += lpips_

        return loss_dict

    def load_model(self, iteration=5000, checkpoint_path=None):
        if checkpoint_path is not None:
            path = checkpoint_path
        else:
            path = self.checkpoint_path
        os.makedirs(path, exist_ok=True)

        self.hand_model.pcl_embed.load_state_dict(torch.load(os.path.join(path, 'pcl_embed', 'iteration_' + str(iteration), 'ckpt.pth'))
                                                  ['pcl_embed'])
        self.hand_model.pcl_embed.eval()

        self.hand_model.motion_embed_mlp.load_state_dict(torch.load(os.path.join(path, 'motion_embed', 'iteration_' + str(iteration), 'ckpt.pth'))
                                                         ['motion_embed'])
        self.hand_model.motion_embed_mlp.eval()

        self.hand_model.transformer = (torch.load(os.path.join(path, 'transformer', 'iteration_' + str(iteration), 'ckpt.pth'))
                                                    ['transformer'])
        self.hand_model.transformer.eval()

        self.hand_model.encoder = (torch.load(os.path.join(path, 'encoder', 'iteration_' + str(iteration), 'ckpt.pth'))
                                                    ['encoder'])
        self.hand_model.encoder.eval()

        self.hand_model.renderer.load_state_dict(torch.load(os.path.join(path, 'renderer', 'iteration_' + str(iteration), 'ckpt.pth'))
                                                        ['renderer'])
        self.hand_model.renderer.eval()

        if self.hand_model.renderer.neural_refiner is not None:
            self.hand_model.renderer.neural_refiner.load_state_dict(torch.load(os.path.join(path, 'refiner', 'iteration_' + str(iteration), 'ckpt.pth'))
                                                                     ['neural_refiner'])
            self.hand_model.renderer.neural_refiner.eval()

    def load_checkpoint(self, iteration=None, is_latest=False, checkpoint_path=None, checkpoint_file=None):
        """
        加载模型检查点。
        """

        if checkpoint_file is not None:
            ckpt_file = checkpoint_file
        else:
            base = checkpoint_path if checkpoint_path is not None else getattr(self, 'checkpoint_path', None) or os.path.join('.', 'checkpoint')

            if os.path.isfile(base):
                ckpt_file = base
            else:
                if is_latest:
                    ckpt_file = os.path.join(base, "latest.ckpt")
                else:
                    ckpt_file = os.path.join(base, f"iteration_{iteration}.ckpt")

        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location="cuda", weights_only=False)

        print(f"\033[93m[Loading checkpoint: {ckpt_file}]\033[0m")

        self.hand_model.pcl_embed.load_state_dict(ckpt["pcl_embed"])
        self.hand_model.motion_embed_mlp.load_state_dict(ckpt["motion_embed_mlp"])

        self.hand_model.adapter.load_state_dict(ckpt["adapter"])
        self.hand_model.aggregator.load_state_dict(ckpt["aggregator"])
        self.hand_model.transformer = ckpt["transformer"]
        self.hand_model.transformer
        for blk in self.hand_model.transformer.layers:
            if hasattr(blk, "part_aware_point"):
                blk.part_aware_point.to('cuda:0')

        self.hand_model.encoder = ckpt["encoder"]

        self.hand_model.renderer.load_state_dict(ckpt["renderer"], strict=False)

        if "optimizer" in ckpt and hasattr(self.hand_model, "optimizer") and self.hand_model.optimizer is not None:
            try:
                self.hand_model.optimizer.load_state_dict(ckpt["optimizer"])
                if torch.cuda.is_available():
                    for param, state in list(self.hand_model.optimizer.state.items()):
                        if isinstance(param, torch.Tensor):
                            target_device = param.device
                            target_dtype = param.dtype
                        else:
                            target = next(self.hand_model.parameters())
                            target_device = target.device
                            target_dtype = target.dtype

                        for k, v in list(state.items()):
                            if isinstance(v, torch.Tensor):
                                try:
                                    state[k] = v.to(device=target_device, dtype=target_dtype)
                                except Exception:
                                    state[k] = v.to(device=target_device)
            except Exception as e:
                print(f"[Warning] failed to restore optimizer state: {e}")

        if "scheduler" in ckpt and hasattr(self.hand_model, "scheduler") and self.hand_model.scheduler is not None:
            try:
                self.hand_model.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[Warning] failed to restore scheduler state: {e}")

        if "neural_refiner" in ckpt and self.hand_model.renderer.neural_refiner is not None:
            self.hand_model.renderer.neural_refiner.load_state_dict(ckpt["neural_refiner"])

        self.hand_model.eval()
        print(f"\033[92m[Checkpoint loaded successfully.]\033[0m")
        return ckpt.get("iteration", None)

    def load_checkpoint_with_color_shift_scale(self, iteration=None, is_latest=False,
                                               checkpoint_path=None, checkpoint_file=None):
        """
        Load checkpoint and explicitly restore color_shift / color_scale if present.
        This does not modify the original load_checkpoint behavior.
        """
        loaded_iter = self.load_checkpoint(
            iteration=iteration,
            is_latest=is_latest,
            checkpoint_path=checkpoint_path,
            checkpoint_file=checkpoint_file,
        )

        if checkpoint_file is not None:
            ckpt_file = checkpoint_file
        else:
            base = checkpoint_path if checkpoint_path is not None else getattr(self, 'checkpoint_path', None) or os.path.join('.', 'checkpoint')
            if os.path.isfile(base):
                ckpt_file = base
            else:
                if is_latest:
                    ckpt_file = os.path.join(base, "latest.ckpt")
                else:
                    ckpt_file = os.path.join(base, f"iteration_{iteration}.ckpt")

        if not os.path.exists(ckpt_file):
            print(f"[Warn] Color shift/scale load skipped (missing): {ckpt_file}")
            return loaded_iter

        ckpt = torch.load(ckpt_file, map_location="cuda", weights_only=False)
        model_state = ckpt.get("model_state", {})

        device = next(self.hand_model.parameters()).device
        self._ensure_color_inversion_params(device)

        if "color_shift" in ckpt:
            self.hand_model.color_shift.data.copy_(ckpt["color_shift"].to(device))
        elif "color_shift" in model_state:
            self.hand_model.color_shift.data.copy_(model_state["color_shift"].to(device))

        if "color_scale" in ckpt:
            self.hand_model.color_scale.data.copy_(ckpt["color_scale"].to(device))
        elif "color_scale" in model_state:
            self.hand_model.color_scale.data.copy_(model_state["color_scale"].to(device))

        if "color_bias_lowres" in ckpt and hasattr(self.hand_model, "color_bias_lowres"):
            try:
                self.hand_model.color_bias_lowres.data.copy_(ckpt["color_bias_lowres"].to(device))
            except Exception:
                pass

        self._sync_color_inversion_params()

        return loaded_iter

    def finetune_model(self, iteration=5000, is_latest=False, checkpoint_path=None, checkpoint_file=None):
        if checkpoint_file is not None:
            ckpt_file = checkpoint_file
        else:
            base = checkpoint_path if checkpoint_path is not None else getattr(self, 'checkpoint_path', None) or os.path.join('.', 'checkpoint')

            if os.path.isfile(base):
                ckpt_file = base
            else:
                if is_latest:
                    ckpt_file = os.path.join(base, "latest.ckpt")
                else:
                    ckpt_file = os.path.join(base, f"iteration_{iteration}.ckpt")

        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location="cuda", weights_only=False)

        print(f"\033[93m[Loading checkpoint: {ckpt_file}]\033[0m")

        model_state = ckpt.get("model_state", None)
        if model_state is not None:
            missing, unexpected = self.hand_model.load_state_dict(model_state, strict=False)
            if len(missing) > 0:
                print(f"[Warn] Missing keys when loading model_state: {missing}")
            if len(unexpected) > 0:
                print(f"[Warn] Unexpected keys when loading model_state: {unexpected}")
        else:
            module_map = {
                "pcl_embed": getattr(self.hand_model, "pcl_embed", None),
                "motion_embed_mlp": getattr(self.hand_model, "motion_embed_mlp", None),
                "adapter": getattr(self.hand_model, "adapter", None),
                "aggregator": getattr(self.hand_model, "aggregator", None),
                "transformer": getattr(self.hand_model, "transformer", None),
                "encoder": getattr(self.hand_model, "encoder", None),
                "renderer": getattr(self.hand_model, "renderer", None),
                "neural_refiner": getattr(getattr(self.hand_model, "renderer", None), "neural_refiner", None),
                "vertex_global_mapping": getattr(self.hand_model, "vertex_global_mapping", None),
            }

            for key, module in module_map.items():
                if module is None or key not in ckpt:
                    continue
                try:
                    payload = ckpt[key]
                    if isinstance(payload, nn.Module):
                        module.load_state_dict(payload.state_dict(), strict=False)
                    elif isinstance(payload, dict):
                        module.load_state_dict(payload, strict=False)
                    else:
                        setattr(self.hand_model, key, payload)
                except Exception as e:
                    print(f"[Warn] Failed to load {key}: {e}")

        self._sync_color_inversion_params()

        if getattr(self.hand_model, "transformer", None) is not None:
            self.hand_model.transformer
            for blk in self.hand_model.transformer.layers:
                if hasattr(blk, "part_aware_point"):
                    blk.part_aware_point.to('cuda:0')

        freeze_all(self.hand_model)

        feature_dim = 1024
        uvmap_size = 256

        bias_h, bias_w = 1024, 1024
        self.hand_model.color_b_map = torch.nn.Parameter(
            torch.zeros(1, 3, bias_h, bias_w, dtype=torch.float32, device='cuda'),

            requires_grad=True,
        )
        self.hand_model.opacity_b_map = torch.nn.Parameter(
            torch.zeros(1, 1, bias_h, bias_w, dtype=torch.float32, device='cuda'),

            requires_grad=True,
        )
        print(f"\033[92m[Finetune] Initialized frozen color/opacity bias maps (LoRA-only finetune): {self.hand_model.color_b_map.shape}, {self.hand_model.opacity_b_map.shape}\033[0m")

        if hasattr(self.hand_model, "bias_style_unet") and self.hand_model.bias_style_unet is not None:
            for p in self.hand_model.bias_style_unet.parameters():
                p.requires_grad = False

        lora_params = []
        renderer = getattr(self.hand_model, "renderer", None)
        if renderer is not None:
            if hasattr(renderer, "lora_mlp") and renderer.lora_mlp is not None:
                for p in renderer.lora_mlp.parameters():
                    p.requires_grad = True
                lora_params.extend(list(renderer.lora_mlp.parameters()))

        self.color_bias_reg_weight = 100.0
        self.opacity_bias_reg_weight = 1.0
        self.map_bias_reg_weight = 0.01

        optim_params = []
        if len(lora_params) > 0:
            optim_params.append({"params": lora_params, "lr": 1e-4
                                 })

        optim_params.append({"params": self.hand_model.color_b_map, "lr": 1e-4})
        optim_params.append({"params": self.hand_model.opacity_b_map, "lr": 1e-4})

        self.hand_model.optimizer = torch.optim.AdamW(optim_params, weight_decay=0.0)

        self.hand_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.hand_model.optimizer,
            T_max=iteration,
            eta_min=1e-5,
        )

        self.hand_model.optimizer.zero_grad(set_to_none=True)

        self.accum_step = 1

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_all(model: nn.Module):
    set_requires_grad(model, False)
    model.eval()

def unfreeze_modules(modules, train_mode=True):
    """
    modules: iterable of nn.Module
    train_mode: True 则 .train()，否则保持 eval()
    """
    for m in modules:
        set_requires_grad(m, True)
        if train_mode:
            m.train()

class WarmupScheduler:
    """
    自动控制 HierarchicalPointEmbed 的参数冻结与解冻。
    可基于 epoch 或 iteration 调度。
    """
    def __init__(self, model: nn.Module, use_iteration=False, warmup_iters=10000):
        """
        Args:
            model: HierarchicalPointEmbed 实例。
            warmup_epochs: warmup 阶段持续的 epoch 数。
            use_iteration: 是否基于 iteration 调度（而不是 epoch）。
            warmup_iters: 若基于 iteration，则 warm-up 的 iteration 数。
        """
        self.model = model
        self.use_iteration = use_iteration
        self.warmup_iters = warmup_iters
        self._frozen = None

        self.high_level_keys = [
            'part_pool_proj', 'part_processor',
            'part_broadcast_proj', 'detail_mlp'
        ]

    def step(self, epoch=None, iteration=None):
        """
        每个 epoch 或 iteration 调用一次。
        自动调整 requires_grad。
        """

        if self.use_iteration:
            warmup = iteration is not None and iteration < self.warmup_iters
        else:
            warmup = epoch is not None and epoch < self.warmup_epochs

        if warmup == self._frozen:
            return
        self._frozen = warmup

        for name, param in self.model.named_parameters():
            if any(k in name for k in self.high_level_keys):
                param.requires_grad = not warmup
            else:
                param.requires_grad = True

    def print_trainable_summary(self):
        """
        打印当前可训练参数比例（用于 sanity check）
        """
        total, trainable = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"[WarmupScheduler] Trainable params: {trainable/total:.2%} ({trainable}/{total})")

class AnnealingScheduler:
    """
    简单退火调度器，支持 linear / exponential / cosine 三种模式。
    用法：
        annealer = AnnealingScheduler(start=1.0, end=0.1, total_iters=10000, mode='cosine')
        val = annealer.step(iteration)
    """
    def __init__(self, start=1.0, end=0.1, total_iters=20000, mode='cosine'):
        self.start = float(start)
        self.end = float(end)
        self.total_iters = max(1, int(total_iters))
        assert mode in ('linear', 'exponential', 'cosine')
        self.mode = mode

    def step(self, iteration: int):
        it = max(0, min(int(iteration), self.total_iters))
        t = it / self.total_iters
        if self.mode == 'linear':
            return self.start + (self.end - self.start) * t
        if self.mode == 'exponential':
            return self.start * ((self.end / self.start) ** t)

        import math
        return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * t))
