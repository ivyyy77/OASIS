
import argparse
import os
import pdb
import time
import random

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
import imageio
from tools import libcore


from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.utils.model_download_utils import AutoModelQuery


try:
    from engine.SegmentAPI.SAM import SAM2Seg
except:
    print("\033[31mNo SAM2 found! Try using rembg to remove the background. This may slightly degrade the quality of the results!\033[0m")
    from rembg import remove

from LHM.datasets.cam_utils import (
    build_camera_principle,
    build_camera_standard,
    create_intrinsics,
    surrounding_views_linspace,
)
from LHM.models.modeling_human_lrm import ModelHumanLRM
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
from LHM.utils.face_detector import FaceDetector
from PIL import Image

# from LHM.utils.video import images_to_video
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.logging import configure_logger
from LHM.utils.model_card import MODEL_CARD, MODEL_CONFIG
from LHM.runners.infer.base_inferrer import Inferrer

@REGISTRY_RUNNERS.register("infer.hand_lrm")
class HandLRMInferrer(Inferrer):
    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"

    # EXP_TYPE: str = "human_lrm_sd3"

    def __init__(self):
        super().__init__()

        self.cfg, cfg_train = parse_configs()

        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )  # logger function

        # if do not download prior model, we automatically download them.
        prior_check()

        try:
            self.parsingnet = SAM2Seg()
        except:
            self.parsingnet = None

        self.hand_model = ModelHandLRM()

        # self.pose_decoder = BodyPoseRefiner(total_bones=totol_bones, embedding_size=3*(totol_bones-1), mlp_width=128, mlp_depth=2)
        # self.pose_decoder.to('cuda')

        self.motion_dict = dict()

    def _build_model(self, cfg):
        from LHM.models import model_dict

        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])  # ModelHumanLRMSapdinoBodyHeadSD3_5

        model = hf_model_cls.from_pretrained(cfg.model_name)  # 从这里加载的预训练的！！！
        return model

    def _default_source_camera(
            self,
            dist_to_center: float = 2.0,
            batch_size: int = 1,
            device: torch.device = torch.device("cpu"),
    ):
        # return: (N, D_cam_raw)
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
        # return: (N, M, D_cam_render)
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


    @torch.no_grad()
    def parsing(self, img_path):

        parsing_out = self.parsingnet(img_path=img_path, bbox=None)

        alpha = (parsing_out.masks * 255).astype(np.uint8)

        return alpha


    def infer_single(
            self,
            image_path: str,
            motion_seqs,
            motion_img_dir,
            motion_video_read_fps,
            export_video: bool,
            export_mesh: bool,
            dump_tmp_dir: str,  # require by extracting motion seq from video, to save some results
            dump_image_dir: str,
            dump_video_path: str,
            shape_param=None,
            splatformer_model=None,
            batch=None,
            image_paths=None,
            mask_paths=None,
            # mask_paths=None,
            optimizer=None, scheduler=None, scaler=None
    ):

        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = self.cfg.render_fps
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False

        mask_path = image_path.replace('images', 'masks')
        parsing_mask = np.array(Image.open(mask_path))

        # prepare reference image
        image, _, _ = infer_preprocess_image(  # 只把人裁剪出来的图片，且 mask 了背景
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

        # save masked image for vis
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
            # camera_size = len(motion_seq["motion_seqs"])
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
            vis_motion=vis_motion,
        )
        self.motion_dict[motion_name] = motion_seq
        camera_size = motion_seq["render_c2ws"].size(1)

        device = "cuda"
        dtype = torch.float32
        # shape_param = torch.tensor([-0.2310,  1.8540, -0.9160,  0.5136,  1.6672,
        #                             1.7144, -0.1274, -0.0522, 0.3267,  0.3119]).unsqueeze(0)
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.hand_model.to(dtype)
        smplx_params = motion_seq['smplx_params']
        # smplx_params['betas'] = shape_param.to(device)
        gs_model_list, query_points, transform_mat_neutral_pose = self.hand_model.infer_single_view(
            # transform_mat_neutral_pose: 1,55,4,4
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            # render_trans=motion_seq["render_trans"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()  # concat all smplx params
            },
        )
        # =========================================== 下面是变形了 ===========================================

        batch_list = []
        batch_size = 40  # avoid memeory out!

        # render_intrs = motion_seq["render_intrs"][:, 0].to(device).repeat(batch_size, 1, 1).unsqueeze(0)

        for batch_i in range(0, camera_size, batch_size):
            # with torch.no_grad():   # 取消
            # TODO check device and dtype
            # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])

            print(f"batch: {batch_i}, total: {camera_size // batch_size + 1} ")

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
            # batch_smplx_params["betas"] = shape_param.to(device)
            batch_smplx_params["betas"] = smplx_params['betas'].to('cuda')
            batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
            for key in keys:
                batch_smplx_params[key] = motion_seq["smplx_params"][key][
                                          :, batch_i: batch_i + batch_size
                                          ].to(device)

            # def animation_infer(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors, render_h, render_w):
            res = self.hand_model.animation_infer(gs_model_list, query_points, batch_smplx_params,
                                             render_c2ws=motion_seq["render_c2ws"][
                                                         :, batch_i: batch_i + batch_size
                                                         ].to(device),
                                             render_intrs=motion_seq["render_intrs"][
                                                          :, batch_i: batch_i + batch_size
                                                          ].to(device),
                                             # render_intrs=render_intrs,
                                             # render_trans=motion_seq["render_trans"][
                                             # :, batch_i: batch_i + batch_size
                                             # ].to(device),
                                             render_bg_colors=motion_seq["render_bg_colors"][
                                                              :, batch_i: batch_i + batch_size
                                                              ].to(device),
                                             model=splatformer_model,
                                             gt_img=image_paths[
                                                    batch_i: batch_i + batch_size, :
                                                    ].to(device),
                                             gt_msk=mask_paths[
                                                    batch_i: batch_i + batch_size, :
                                                    ].to(device),
                                             optimizer=optimizer, scheduler=scheduler, scaler=scaler
                                             )

            comp_rgb = res["comp_rgb"]  # [Nv, H, W, 3], 0-1
            comp_mask = res["comp_mask"]  # [Nv, H, W, 3], 0-1

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
            gradio_codec=False,
            verbose=True,
        )

    def train_single(
            self,
            image_path: str,
            motion_seqs,
            motion_img_dir,
            motion_video_read_fps,
            export_video: bool,
            export_mesh: bool,
            dump_tmp_dir: str,  # require by extracting motion seq from video, to save some results
            dump_image_dir: str,
            dump_video_path: str,
            shape_param=None,
            splatformer_model=None,
            batch=None,
            image_paths=None,
            mask_paths=None,
            # mask_paths=None,
            optimizer=None, scheduler=None, scaler=None
    ):

        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        # render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False

        mask_path = image_path.replace('images', 'masks')
        parsing_mask = np.array(Image.open(mask_path))
        # if self.parsingnet is not None:
        #     parsing_mask = self.parsing(image_path)
        # else:
        #     img_np = cv2.imread(image_path)
        #     remove_np = remove(img_np)
        #     parsing_mask = remove_np[...,3]

        # prepare reference image
        image, _, _ = infer_preprocess_image(  # 只把人裁剪出来的图片，且 mask 了背景
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

        # from LHM.models.hmr2_extension import load_hmr_predictor
        # gaussian_predictor = load_hmr_predictor()

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
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # read motion seq

        # motion_name = os.path.dirname(
        #     motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        # )
        # motion_name = os.path.basename(motion_name)

        # normalized_path = os.path.normpath(motion_seqs_dir)  # 输出: train_data/motion_video/sign
        # 提取最后一个目录名
        # motion_name = os.path.basename(normalized_path)

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
            vis_motion=vis_motion,
        )
        motion_name = 'sign'
        self.motion_dict[motion_name] = motion_seq
        camera_size = motion_seq["render_c2ws"].size(1)

        device = "cuda"
        dtype = torch.float32
        # shape_param = torch.tensor([-0.2310,  1.8540, -0.9160,  0.5136,  1.6672,
        #                             1.7144, -0.1274, -0.0522, 0.3267,  0.3119]).unsqueeze(0)
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.hand_model.to(dtype)
        smplx_params = motion_seq['smplx_params']
        # smplx_params['betas'] = shape_param.to(device)
        gs_model_list, query_points, transform_mat_neutral_pose = self.hand_model.infer_single_view(
            # transform_mat_neutral_pose: 1,55,4,4
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            # render_trans=motion_seq["render_trans"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()  # concat all smplx params
            },
        )

        # =========================================== 下面是变形了 ===========================================
        # 1. 随机选取 4 个 index
        num_samples = 4
        total_frames = motion_seq["smplx_params"]["root_pose"].shape[1]
        sampled_indices = sorted(random.sample(range(total_frames), num_samples))

        # 2. 构造 smplx 参数的 batch
        keys = ["root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                "lhand_pose", "rhand_pose", "trans", "focal", "princpt", "img_size_wh", "expr"]

        batch_smplx_params = {
            "betas": smplx_params['betas'].to(device),  # 形状参数保持不变
            "transform_mat_neutral_pose": transform_mat_neutral_pose,
        }

        for key in keys:
            # 只采样选中的帧
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
                                            model=splatformer_model,
                                            gt_img=image_paths[sampled_indices, :].to(device),
                                            gt_msk=mask_paths[sampled_indices, :].to(device),
                                            optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        # =========================================== 下面是变形了 ===========================================

    def train_hand(
            self,
            image_path: str,
            motion_seqs,
            motion_img_dir,
            motion_video_read_fps,
            # export_video: bool,
            # export_mesh: bool,
            dump_tmp_dir: str,  # require by extracting motion seq from video, to save some results
            # dump_image_dir: str,
            # dump_video_path: str,
            shape_param=None,
            splatformer_model=None,
            batch=None,
            image_paths=None,
            mask_paths=None,
            # mask_paths=None,
            optimizer=None, scheduler=None, scaler=None
    ):

        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        # render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False

        mask_path = image_path.replace('images', 'masks')
        parsing_mask = np.array(Image.open(mask_path))

        # prepare reference image
        image, _, _ = infer_preprocess_image(  # 只把人裁剪出来的图片，且 mask 了背景
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

        # save masked image for vis
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
            vis_motion=vis_motion,
        )
        motion_name = 'sign'
        self.motion_dict[motion_name] = motion_seq
        camera_size = motion_seq["render_c2ws"].size(1)

        device = "cuda"
        dtype = torch.float32
        # shape_param = torch.tensor([-0.2310,  1.8540, -0.9160,  0.5136,  1.6672,
        #                             1.7144, -0.1274, -0.0522, 0.3267,  0.3119]).unsqueeze(0)
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.hand_model.to(dtype)
        smplx_params = motion_seq['smplx_params']
        # smplx_params['betas'] = shape_param.to(device)
        gs_model_list, query_points, transform_mat_neutral_pose = self.hand_model.infer_single_view(
            # transform_mat_neutral_pose: 1,55,4,4
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            # render_trans=motion_seq["render_trans"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()  # concat all smplx params
            },
        )

        # =========================================== 下面是变形了 ===========================================
        # 1. 随机选取 4 个 index
        num_samples = 4
        total_frames = motion_seq["smplx_params"]["root_pose"].shape[1]
        sampled_indices = sorted(random.sample(range(total_frames), num_samples))

        # 2. 构造 smplx 参数的 batch
        keys = ["root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                "lhand_pose", "rhand_pose", "trans", "focal", "princpt", "img_size_wh", "expr"]

        batch_smplx_params = {
            "betas": smplx_params['betas'].to(device),  # 形状参数保持不变
            "transform_mat_neutral_pose": transform_mat_neutral_pose,
        }

        for key in keys:
            # 只采样选中的帧
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
                                            model=splatformer_model,
                                            gt_img=image_paths[sampled_indices, :].to(device),
                                            gt_msk=mask_paths[sampled_indices, :].to(device),
                                            optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        # =========================================== 下面是变形了 ===========================================
        print('finish')

    def infer(self, splatformer_model=None, batch=None, optimizer=None, scheduler=None, scaler=None):

        if len(batch) == 1:
            source_img = batch[0]['image_path']
            # image_paths = batch[0]['image_path']
            # omit_prefix = os.path.dirname(source_img)
            source_msk = batch[0]['mask_path']
            source_ann = batch[0]['annotation']
        else:
            source_img = [item['image_path'] for item in batch]

        # for image_path in tqdm(image_paths,
        for image_path in tqdm(source_img,
                               disable=not self.accelerator.is_local_main_process,
                               ):

            omit_prefix = os.path.dirname(image_path)

            # prepare dump paths
            image_name = os.path.basename(image_path)
            uid = image_name.split(".")[0]
            subdir_path = os.path.dirname(image_path).replace(omit_prefix, "")
            subdir_path = (
                subdir_path[1:] if subdir_path.startswith("/") else subdir_path
            )
            print("subdir_path and uid:", subdir_path, uid)

            # setting config
            motion_seqs_all = batch[0]['all_annotations']
            image_paths_all = batch[0]['all_images']
            mask_paths_all = batch[0]['all_masks']
            keys = list(motion_seqs_all.keys())  # 将键转为列表

            key = random.choice(keys)
            motion_seqs = motion_seqs_all[key]
            image_paths = image_paths_all[key]
            mask_paths = mask_paths_all[key]

            # 批量读取图像
            sample_img = cv2.imread(image_paths[0])
            h, w, c = sample_img.shape
            num_images = len(image_paths)

            # 预分配内存（避免动态扩容）
            masked_images = np.empty((num_images, h, w, c), dtype=np.float32)
            masks_images = np.empty((num_images, h, w), dtype=np.float32)
            # masked_images = np.empty((num_images, h, w, c), dtype=np.uint8)

            for i, (img_path, msk_path) in enumerate(zip(image_paths, mask_paths)):
                img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
                msk = imageio.imread(msk_path)
                msk = (msk != 0).astype(np.uint8)
                img[msk == 0] = 1
                masked_images[i] = img
                masks_images[i] = msk

            masked_images = torch.tensor(masked_images)
            masks_images = torch.tensor(masks_images)
            # 堆叠为 (N, H, W, 3) 的张量
            # image_array = np.stack(images, axis=0)

            # ============================================================================================
            motion_seqs_dir = self.cfg.motion_seqs_dir
            motion_name = os.path.dirname(
                motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
            )
            motion_name = os.path.basename(motion_name)
            dump_video_path = os.path.join(
                self.cfg.video_dump,
                subdir_path,
                motion_name,
                f"{uid}.mp4",
            )
            dump_image_dir = os.path.join(
                self.cfg.image_dump,
                subdir_path,
            )
            dump_mesh_dir = os.path.join(
                self.cfg.mesh_dump,
                subdir_path,
            )
            dump_tmp_dir = os.path.join(self.cfg.image_dump, subdir_path, "tmp_res")
            os.makedirs(dump_image_dir, exist_ok=True)
            os.makedirs(dump_tmp_dir, exist_ok=True)
            os.makedirs(dump_mesh_dir, exist_ok=True)

            shape_pose = self.pose_estimator(image_path)

            self.train_single(
                    image_path,
                    motion_seqs=motion_seqs,
                    motion_img_dir=self.cfg.motion_img_dir,
                    motion_video_read_fps=self.cfg.motion_video_read_fps,
                    export_video=self.cfg.export_video,
                    export_mesh=self.cfg.export_mesh,
                    dump_tmp_dir=dump_tmp_dir,
                    dump_image_dir=dump_image_dir,
                    dump_video_path=dump_video_path,
                    shape_param=shape_pose.beta,
                    splatformer_model=splatformer_model,
                    batch=batch,
                    image_paths=masked_images,
                    mask_paths=masks_images,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    # single_img=single_img,
            )

