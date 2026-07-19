import os
import random
import glob
import pickle
import json
import math

import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path
import cv2
from data.hand_dataset import CameraInfo

from data.hand_dataset import merge_batch

from pytorch3d.transforms import matrix_to_axis_angle
from tools_utils.model.handavatar.configs import cfg
import tools_utils.model.smplx
from tools_utils.model.smplx.manohd.subdivide import sub_mano
from tools_utils.model.smplx_ours.manohd.subdivide import sub_mano as sub_mano_ori
from tools_utils.camera_utils import loadCam_aug_bs, get_rays_from_KRT, rays_intersect_3d_bbox
from data.utils.camera_util import apply_global_tfm_to_camera

from simple_knn._C import distCUDA2

from plyfile import PlyData
import pandas as pd

class HandDataset(Dataset):
    def __init__(self, split='test_wild', frm_list=None, num_frames=1, img_path=None, edit=False):
        if split != 'test_wild':
            raise ValueError("HandDataset only supports split='test_wild'")
        if not img_path:
            raise ValueError('img_path is required')
        self.split = split
        self.img_path_abs = os.path.abspath(img_path)
        images_dir = os.path.dirname(self.img_path_abs)
        self.dat_dir = os.path.dirname(images_dir)
        self.num_frames = num_frames
        self.video_ids = [os.path.basename(self.img_path_abs)]
        self.edit = edit

        self.height, self.width = 256, 256

        self.mano = tools_utils.model.smplx_ours.create(**cfg.smpl_cfg)
        self.mano_wild_778 = tools_utils.model.smplx_ours.create(**cfg.smpl_cfg)
        self.mano_ori = tools_utils.model.smplx.create(**cfg.smpl_cfg)
        self.mano_778 = tools_utils.model.smplx.create(**cfg.smpl_cfg)

        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano_ori(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights

        if cfg.smpl_cfg['manohd'] > 0:
            self.mano_ori, _, _ = sub_mano_ori(self.mano_ori, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano_ori.lbs_weights = lbs_weights

        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype == 'left':
            self.mano.shapedirs[:, 0, :] *= -1

        mano_res = self.mano(
            torch.zeros(1, 10).float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(), return_verts=True)

        self.canonical_verts = mano_res.vertices.detach().numpy().squeeze()

        self.faces = mano_res.faces_tensor
        self.big_pose_xyz = mano_res.vertices.detach().numpy().squeeze()
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)

        big_pose_smpl_param['faces'] = self.faces.detach()
        self.big_pose_smpl_param = big_pose_smpl_param
        big_pose_min_xyz = np.min(self.big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(self.big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        self.big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(self.canonical_verts)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]
        self.pcd_scales = torch.exp(scales)

        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        print(f'Loading MANO semantic labels from {labels_path}')
        ply = PlyData.read(labels_path)
        if ply.elements:
            pc = pd.DataFrame(ply.elements[0].data).values
        self.labels = torch.tensor(pc[:, 6].astype(np.uint8))

        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        nail_indices = np.isin(self.labels, list(NAIL_PARTS.values()))

        self.nail_labels = self.labels[nail_indices]

        nail_vertex_indices = np.nonzero(nail_indices)[0]
        faces_np = self.mano.faces.cpu().numpy() if torch.is_tensor(self.mano.faces) else self.mano.faces
        face_nail_mask = np.all(np.isin(faces_np, nail_vertex_indices), axis=1)
        self.nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))
        self.nail_mask_1 = self.nail_mask.to(torch.float32).unsqueeze(1)

        self.nail_faces = faces_np[face_nail_mask]

        self._uv_template = self._build_right_hand_uv_template()

    def _build_right_hand_uv_template(self):
        if self.handtype != 'right':
            return None
        try:
            uv_root = _resolve_mano_uv_root()
        except FileNotFoundError as exc:
            print(f'[WARNING] {exc}')
            return None

        obj_path = uv_root / 'original mano template' / 'hand.obj'
        vt_raw, ft_raw, _ = _read_mano_uv_obj(str(obj_path))
        if vt_raw.size == 0 or ft_raw.size == 0:
            print(f'[WARNING] Failed to read MANO UV data from {obj_path}')
            return None

        vt_right = vt_raw / 2.0
        change_path = uv_root / 'change' / 'change_r.npy'
        if not change_path.exists():
            print(f'[WARNING] Missing MANO UV index file at {change_path}')
            return None
        change_idx = np.load(change_path).astype(np.int64)

        face_uv = ft_raw.astype(np.int64)
        face_uv_xy = vt_right[face_uv]

        return {
            'change_idx': change_idx,
            'face_idx': torch.from_numpy(face_uv.astype(np.int64)),
            'face_uv_xy': torch.from_numpy(face_uv_xy.astype(np.float32)),
        }

    @staticmethod
    def skeleton_to_box(skeleton):
        min_xyz = np.min(skeleton, axis=0) - 0.01
        max_xyz = np.max(skeleton, axis=0) + 0.01
        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def get_patch_ray_indices(
            self,
            N_patch,
            ray_mask,
            subject_mask,
            bbox_mask,
            patch_size,
            H, W):

        assert subject_mask.dtype == np.bool_
        assert bbox_mask.dtype == np.bool_

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max =\
                self._get_patch_ray_indices(ray_mask, candidate_mask,
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)

            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices

    def _get_patch_ray_indices(
            self,
            ray_mask,
            candidate_mask,
            patch_size,
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool_
        assert candidate_mask.dtype == np.bool_

        valid_ys, valid_xs = np.where(candidate_mask)

        select_idx = np.random.choice(valid_ys.shape[0],
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size,
                        a_min=0,
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]

        inter_mask = inter_mask.reshape(H, W)

        return select_inds,\
                inter_mask[y_min:y_max, x_min:x_max],\
                np.array([x_min, y_min]), np.array([x_max, y_max])

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        ray_alpha = ray_alpha[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, ray_alpha, near, far

    def sample_patch_rays(self, img, alpha, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, ray_alpha, near, far):

        select_inds, patch_info, patch_div_indices =\
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches,
                ray_mask=ray_mask,
                subject_mask=subject_mask,
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size,
                H=H, W=W)

        rays_o, rays_d, ray_img, ray_alpha, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far)

        targets = []
        targets_alpha = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i]
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
            targets_alpha.append(alpha[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0)
        target_alpha_patches = np.stack(targets_alpha, axis=0)

        patch_masks = patch_info['mask']

        return rays_o, rays_d, ray_img, ray_alpha, near, far,\
                target_patches, target_alpha_patches, patch_masks, patch_div_indices

    def __len__(self):
        if self.split in ['test_wild']:
            return 1
        else:
            return len(self.video_ids)

    def process_image(self, img, mask, bg_color=(0, 0, 0)):
        """
        Apply mask to image, set background to bg_color. Returns segmented image and mask.
        img: RGB image (H,W,3), mask: grayscale (H,W) or (H,W,1)
        """
        if len(mask.shape) == 2:
            mask = mask[..., None]
        mask_f = mask.astype(np.float32) / 255.
        bg = np.array(bg_color, dtype=np.float32).reshape(1,1,3)
        img = img.astype(np.float32) / 255.
        seg_img = mask_f * img + (1.0 - mask_f) * bg

        return seg_img

    @staticmethod
    def resize_mask(mask, target_size):
        """Resize alpha/binary masks with downsampling-aware interpolation."""
        src_h, src_w = mask.shape[:2]
        dst_w, dst_h = target_size
        if (src_w, src_h) == (dst_w, dst_h):
            return mask

        if dst_w < src_w or dst_h < src_h:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_NEAREST

        return cv2.resize(mask, (dst_w, dst_h), interpolation=interp)

    def refine_mask_after_mano(self, mask, img):
        if mask is None or img is None:
            return mask
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.max() < 20:
            return mask

        _, mask_bin = cv2.threshold(mask_u8[..., 0], 255, 255, cv2.THRESH_BINARY)
        mask_bool = mask_bin > 0
        if not np.any(mask_bool):
            return mask

        sel_img = img[mask_bool].mean(axis=-1)
        if sel_img.size == 0 or sel_img.max() < 20:
            return mask

        keep = np.bitwise_and(sel_img > 10, sel_img < 200)
        mask_bool[mask_bool] = keep.astype(np.int32)
        mask = mask_u8 * mask_bool[..., None]

        contours, _ = cv2.findContours(mask[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        poly = contours[0].transpose(1, 0, 2).astype(np.int32)
        poly_mask = np.zeros_like(img)
        poly_mask = cv2.fillPoly(poly_mask, poly, (1, 1, 1))
        mask = mask * poly_mask

        return mask

    def get_img(self):
        frame_name = self.video_ids[0]
        img_path = os.path.join(self.dat_dir, 'images', frame_name)
        mask_path = img_path.replace('images', 'masks')
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if self.edit:
            mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'
        mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'
        if mask_edit_path is not None and os.path.exists(mask_edit_path):
            mask_edit = cv2.imread(mask_edit_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

        anno_path = img_path.replace('images', 'anno')
        anno_path = os.path.splitext(anno_path)[0] + '.pkl'

        if not os.path.exists(mask_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            mask_path = os.path.join(self.dat_dir, 'masks', generate_name)
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if not os.path.exists(anno_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            anno_path = os.path.join(self.dat_dir, 'anno', generate_name)
            anno_path = os.path.splitext(anno_path)[0] + '.pkl'

        if self.edit:
            mask_edit = cv2.imread(mask_edit_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

        with open(anno_path, 'rb') as f:
            ann = pickle.load(f)

        if isinstance(ann, (list, tuple)):
            return self.get_img_rainbow()
        if not isinstance(ann, dict):
            raise TypeError(f"Unsupported annotation format in {anno_path}: {type(ann).__name__}")

        if ann.get('pred_mano_params', {}) == {}:
            ann = ann.get('frames', {})
            ann = ann.get(os.path.splitext(frame_name)[0], {})

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)
        mask = self.resize_mask(mask, (self.width, self.height))

        mask = self.refine_mask_after_mano(mask, img)

        img = self.process_image(img, mask)
        cam_info_list = []

        w2c = np.eye(4, dtype=np.float32)
        T = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(3,1)
        T[:2] *= 10
        T = T.reshape(3)
        R = np.eye(3, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3:4] = T[..., None]

        raw_K = get_camera_parameters(

            max(self.height, self.width),

            fov=30,

            p_x=None,
            p_y=None,
            device='cuda',
        )
        K = raw_K.numpy()

        focal_length_y = K[1, 1]
        focal_length_x = K[0, 0]
        FovY = focal2fov(focal_length_y, self.height)
        FovX = focal2fov(focal_length_x, self.width)

        bkgd_mask = mask.astype('float32') / 255.0

        if os.path.exists(mask_edit_path):
            bound_mask = mask_edit.astype('float32') / 255.0
        else:
            bound_mask = bkgd_mask

        pred_mano_params = ann.get('pred_mano_params', {})

        global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
        hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)

        rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)

        axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats))
        axis_angle = axis_angle.reshape(-1).numpy()

        posed_res = self.mano(
            torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
            torch.from_numpy(axis_angle)[None, :3].float(),
            torch.from_numpy(axis_angle)[None, 3:].float(),

            return_verts=True)
        world_vertex = posed_res.vertices.detach().numpy()
        smpl_param = {
            'poses': axis_angle,
            'shape': np.array(pred_mano_params.get('betas', []), dtype=np.float32),
            'posed_verts': world_vertex.squeeze(),

        }

        mano_shape = self.mano(
            torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(), return_verts=True)

        self.canonical_verts = mano_shape.vertices.detach().numpy().squeeze()

        posed_778 = self.mano_wild_778(
            torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
            torch.from_numpy(axis_angle[:3])[None].float(),
            torch.from_numpy(axis_angle[3:])[None].float(),
        return_verts=True)

        posed_verts_778 = posed_778.vertices.detach().numpy().squeeze()

        uv_mapping = None
        if self._uv_template is not None:
            vert_uv_tensor = torch.from_numpy(
                posed_verts_778.reshape(-1, 3)[self._uv_template['change_idx']].astype(np.float32)
            )
            uv_mapping = {
                'vert_uv': vert_uv_tensor,
                'face_uv': self._uv_template['face_idx'].clone(),
                'face_uv_xy': self._uv_template['face_uv_xy'].clone(),
                }

        min_xyz = np.min(world_vertex.squeeze(), axis=0)
        max_xyz = np.max(world_vertex.squeeze(), axis=0)
        max_xyz -= 0.05
        min_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None].repeat(3, axis=2)

        verts_cam = world_vertex.reshape(-1, 3)
        verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = (verts_img[:, :2]).astype(np.float32)

        cam_info = CameraInfo(
                    uid=id, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                    image=img, nail_image=nail_img,
                    verts_cam=verts_cam,
                    nail_mask=nail_mask,
                    bound_mask=bound_mask,
                    bkgd_mask=bkgd_mask,
                    image_path=img_path,
                    mask_path=mask_path,
                    image_name=frame_name,
                    width=self.width,
                    height=self.height,
                    smpl_param=smpl_param,
                    world_vertex=world_vertex,
                    world_bound=world_bound,
                    big_pose_smpl_param=self.big_pose_smpl_param,
                    big_pose_world_vertex=self.canonical_verts,
                    big_pose_world_bound=self.big_pose_world_bound
                )

        cam_info = loadCam_aug_bs(id, cam_info)

        if uv_mapping is not None:
            cam_info.vert_uv = uv_mapping['vert_uv']
            cam_info.face_uv = uv_mapping['face_uv']
            cam_info.face_uv_xy = uv_mapping['face_uv_xy']

        cam_info_list.append(cam_info)
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)

        return final_results

    def get_img_rainbow(self):
        frame_name = self.video_ids[0]
        img_path = os.path.join(self.dat_dir, 'images', frame_name)
        mask_path = img_path.replace('images', 'masks')
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if self.edit:
            mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'

        anno_path = img_path.replace('images', 'anno')
        anno_path = os.path.splitext(anno_path)[0] + '.pkl'

        if not os.path.exists(mask_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            mask_path = os.path.join(self.dat_dir, 'masks', generate_name)
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if not os.path.exists(anno_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            anno_path = os.path.join(self.dat_dir, 'anno', generate_name)
            anno_path = os.path.splitext(anno_path)[0] + '.pkl'

        with open(anno_path, 'rb') as fi:
            cameras, mesh_infos, bbox, img_type = pickle.load(fi)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)
        mask = self.resize_mask(mask, (self.width, self.height))
        if self.edit:
            mask_edit = cv2.imread(mask_edit_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

        img = self.process_image(img, mask)
        cam_info_list = []

        dst_skel_info = query_dst_skeleton(mesh_infos)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_cam_joints = dst_skel_info['joint_cam']
        dst_valid_joints = dst_skel_info['joint_valid']

        E = apply_global_tfm_to_camera(
                E=np.eye(4),
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        K = cameras['intrinsics'][:3, :3].copy()

        focal_length_y = K[1, 1]
        focal_length_x = K[0, 0]
        FovY = focal2fov(focal_length_y, self.height)
        FovX = focal2fov(focal_length_x, self.width)

        bkgd_mask = mask.astype('float32') / 255.0
        if self.edit:
            bound_mask = mask_edit.astype('float32') / 255.0
        else:
            bound_mask = bkgd_mask

        posed_res = self.mano_ori(
            torch.from_numpy(dst_shape)[None].float(),
            torch.from_numpy(dst_poses[:3])[None].float(),
            torch.from_numpy(dst_poses[3:])[None].float(),
        return_verts=True)

        posed_778 = self.mano_778(
            torch.from_numpy(dst_shape)[None].float(),
            torch.from_numpy(dst_poses[:3])[None].float(),
            torch.from_numpy(dst_poses[3:])[None].float(),
        return_verts=True)

        cano_res = self.mano_ori(
            torch.from_numpy(dst_shape)[None].float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
        return_verts=True)
        self.canonical_verts = cano_res.vertices[0].detach().numpy()

        world_vertex = posed_res.vertices.detach().numpy()
        posed_verts_778 = posed_778.vertices.detach().numpy()

        uv_mapping = None
        if self._uv_template is not None:
            vert_uv_tensor = torch.from_numpy(
                posed_verts_778.reshape(-1, 3)[self._uv_template['change_idx']].astype(np.float32)
            )
            uv_mapping = {
                'vert_uv': vert_uv_tensor,
                'face_uv': self._uv_template['face_idx'].clone(),
                'face_uv_xy': self._uv_template['face_uv_xy'].clone(),
            }

        smpl_param ={
            'poses': dst_poses,
            'shape': dst_shape,
            'posed_verts': world_vertex.squeeze(),
        }

        min_xyz = np.min(world_vertex.squeeze(), axis=0)
        max_xyz = np.max(world_vertex.squeeze(), axis=0)
        max_xyz -= 0.05
        min_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None].repeat(3, axis=2)

        verts_cam = world_vertex.reshape(-1, 3)
        verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = (verts_img[:, :2]).astype(np.float32)

        cam_info = CameraInfo(
                    uid=id, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                    image=img, nail_image=nail_img,
                    verts_cam=verts_cam,
                    nail_mask=nail_mask,
                    bound_mask=bound_mask,
                    bkgd_mask=bkgd_mask,
                    image_path=img_path,
                    mask_path=mask_path,
                    image_name=frame_name,
                    width=self.width,
                    height=self.height,
                    smpl_param=smpl_param,
                    world_vertex=world_vertex,
                    world_bound=world_bound,
                    big_pose_smpl_param=self.big_pose_smpl_param,
                    big_pose_world_vertex=self.canonical_verts,
                    big_pose_world_bound=self.big_pose_world_bound
                )

        cam_info = loadCam_aug_bs(id, cam_info)
        if uv_mapping is not None:
            cam_info.vert_uv = uv_mapping['vert_uv']
            cam_info.face_uv = uv_mapping['face_uv']
            cam_info.face_uv_xy = uv_mapping['face_uv_xy']

        cam_info_list.append(cam_info)
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)

        return final_results

    def __getitem__(self, idx, attempt=0, max_attempts=10):
        if self.split in ['test_wild']:
            frame_name = self.video_ids[0]

            dat_dir_parts = set(Path(self.dat_dir).as_posix().split('/'))

            if ('coco-ours' in dat_dir_parts) or ('coco' in self.img_path_abs):
                final_results = self.get_img()
            elif 'in_the_wild' in dat_dir_parts:
                if frame_name in ['02023.jpg', '12023.jpg']:
                    final_results = self.get_img_rainbow()
                else:
                    final_results = self.get_img()
            else:
                final_results = self.get_img_rainbow()

            return final_results

        while attempt < max_attempts:
            video_id = self.video_ids[idx]

            img_dir = os.path.join(self.dat_dir, 'images', video_id)
            mask_dir = os.path.join(self.dat_dir, 'masks', video_id)
            vis_dir = os.path.join(self.dat_dir, 'vis', video_id)
            ann_path = os.path.join(self.dat_dir, 'annotations', f"{video_id}.json")

            img_paths = sorted([f for f in glob.glob(os.path.join(img_dir, '*.png'))])
            mask_paths = sorted([f for f in glob.glob(os.path.join(mask_dir, '*.png'))])

            if not os.path.isfile(ann_path):
                print(f"Missing annotation for video {video_id}")
                idx = (idx + 1) % len(self.video_ids)
                attempt += 1
                continue
            with open(ann_path, 'r') as f:
                annotation = json.load(f)

            frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths if os.path.splitext(os.path.basename(p))[0].endswith('_1')]

            mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}

            ann_map = annotation.get('frames', {})

            if len(frame_ids) < self.num_frames:
                if self.split == 'test':
                    idx = (idx + 1) % len(self.video_ids)
                    attempt += 1
                    continue
                frame_ids_left = [os.path.splitext(os.path.basename(p))[0] for p in img_paths if os.path.splitext(os.path.basename(p))[0].endswith('_0')]
                if len(frame_ids) <= 1 and len(frame_ids_left) > 1:
                    frame_ids = frame_ids_left

                    if len(frame_ids) < self.num_frames:
                        idx = (idx + 1) % len(self.video_ids)
                        attempt += 1
                        continue
                    else:
                        chosen = random.sample(frame_ids, self.num_frames)
                else:
                    idx = (idx + 1) % len(self.video_ids)
                    attempt += 1
                    continue
            else:
                if self.split == 'test':
                    chosen = frame_ids[:1]
                chosen = random.sample(frame_ids, self.num_frames)

            cam_info_list = []
            for fid in chosen:
                img_path = os.path.join(img_dir, f'{fid}.png')

                mask_path = mask_map.get(fid, None)
                ann = ann_map.get(fid, {})
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

                img = self.process_image(img, mask)

                w2c = np.eye(4, dtype=np.float32)
                T = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(3,1)

                T[:2] *= 10
                T = T.reshape(3)

                R = np.eye(3, dtype=np.float32)
                w2c[:3, :3] = R
                w2c[:3, 3:4] = T[..., None]

                raw_K = get_camera_parameters(

                    max(self.height, self.width),

                    fov=30,

                    p_x=None,
                    p_y=None,
                    device='cuda',
                )
                K = raw_K.numpy()

                focal_length_y = K[1, 1]
                focal_length_x = K[0, 0]
                FovY = focal2fov(focal_length_y, self.height)
                FovX = focal2fov(focal_length_x, self.width)

                image_name = os.path.basename(img_path)

                bkgd_mask = mask.astype('float32') / 255.0

                pred_mano_params = ann.get('pred_mano_params', {})

                global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
                hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)

                rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)

                axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats))
                axis_angle = axis_angle.reshape(-1).numpy()

                posed_res = self.mano(
                    torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
                    torch.from_numpy(axis_angle)[None, :3].float(),
                    torch.from_numpy(axis_angle)[None, 3:].float(),

                    return_verts=True)
                world_vertex = posed_res.vertices.detach().numpy()
                smpl_param = {
                    'poses': axis_angle,
                    'shape': np.array(pred_mano_params.get('betas', []), dtype=np.float32),
                    'posed_verts': world_vertex.squeeze(),

                }

                min_xyz = np.min(world_vertex.squeeze(), axis=0)
                max_xyz = np.max(world_vertex.squeeze(), axis=0)
                max_xyz -= 0.05
                min_xyz += 0.05
                world_bound = np.stack([min_xyz, max_xyz], axis=0)
                bound_mask = get_bound_2d_mask(world_bound, raw_K.squeeze().cpu(), w2c, self.width, self.height)
                bound_mask = (np.array(bound_mask*255.0, dtype=np.byte))

                semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                                faces=torch.from_numpy(self.mano.faces).long())
                nail_mask = (semantic_mask).astype('float32') / 255.
                nail_mask = nail_mask[..., None].repeat(3, axis=2)

                verts_cam = world_vertex.reshape(-1, 3)
                verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
                verts_img = np.dot(K, verts_cam.T).T
                verts_img[:, :2] /= verts_img[:, 2:3]
                nail_img = (verts_img[:, :2]).astype(np.float32)

                cam_info = CameraInfo(
                    uid=fid, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                    image=img,
                    nail_image=nail_img,
                    nail_mask=nail_mask,
                    verts_cam=verts_cam,
                    bound_mask=bound_mask,
                    bkgd_mask=bkgd_mask,
                    image_path=img_path,
                    mask_path=mask_path,
                    image_name=image_name,
                    width=self.width,
                    height=self.height,
                    smpl_param=smpl_param,
                    world_vertex=world_vertex,
                    world_bound=world_bound,
                    big_pose_smpl_param=self.big_pose_smpl_param,
                    big_pose_world_vertex=self.big_pose_xyz,
                    big_pose_world_bound=self.big_pose_world_bound
                )

                cam_info = loadCam_aug_bs(fid, cam_info)
                cam_info_list.append(cam_info)

            cam_info_dicts = [vars(c) for c in cam_info_list]
            final_results = merge_batch(cam_info_dicts)

            return final_results

    def image_semantic_mask(self, verts_cam, K, R, T, faces, height=None, width=None):
        """
        根据MANO顶点和相机参数，投影到图像并绘制每个类别mask
        Args:
            verts_cam: (N_v, 3) 手部顶点（世界/相机空间）
            K: (3,3) 内参矩阵
            R: (3,3) 旋转矩阵
            T: (3,) 平移向量
            faces: (N_f, 3) 面索引
            image_size: (H, W)
        Returns:
            masks: (num_classes, H, W) np.uint8
        """

        h = self.height if height is None else height
        w = self.width if width is None else width
        masks = np.zeros((h, w), dtype=np.uint8)

        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        verts_img = np.round(verts_img[:, :2]).astype(np.int32)

        for tri in self.nail_faces:
            tri_pts = verts_img[tri, :]
            if np.any(tri_pts < 0) or np.any(tri_pts[:, 0] >= w) or np.any(tri_pts[:, 1] >= h):
                continue
            cv2.fillConvexPoly(masks, tri_pts, 255)

        return masks

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """

    xyz = np.dot(xyz, RT[:3, :3]) + RT[:3, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]

    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def make_dataloader(frameset, shuffle=True, batch_size=1,
                    num_workers=0, prefetch_factor=None,
                    persistent_workers=False, pin_memory=False):
    dataloader = torch.utils.data.DataLoader(frameset, shuffle=shuffle,
                                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                                             batch_size=batch_size,
                                             persistent_workers=persistent_workers,
                                             pin_memory=pin_memory,
                                             collate_fn=frameset_collate_fn)
    return dataloader

from collections import defaultdict

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device("cuda")):
    """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)

    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal

    if p_x is not None and p_y is not None:
        K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
    else:
        K[0, -1], K[1, -1] = img_size // 2, img_size // 2

    return K

def get_focalLength_from_fieldOfView(fov=60, img_size=512):
    """
    Compute the focal length of the camera lens by assuming a certain FOV for the entire image
    Args:
        - fov: float, expressed in degree
        - img_size: int
    Return:
        focal: float
    """
    focal = img_size / (2 * np.tan(np.radians(fov) / 2))
    return focal

def frameset_collate_fn(batches):
    return batches

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def query_dst_skeleton(mesh_infos):
    return {
        'poses': mesh_infos['poses'].astype('float32'),
        'shape': mesh_infos['shape'].astype('float32'),
        'dst_tpose_joints':\
            mesh_infos['tpose_joints'].astype('float32'),
        'bbox': mesh_infos['bbox'].copy(),
        'Rh': mesh_infos['Rh'].astype('float32'),
        'Th': mesh_infos['Th'].astype('float32'),
        'joint_img': mesh_infos['joint_img'].astype('int32'),
        'joint_cam': mesh_infos['joint_cam'].astype('float32'),
        'joint_valid': mesh_infos['joint_valid'].astype('float32')
    }

def _read_mano_uv_obj(filename: str):
    vt, ft, faces = [], [], []
    with open(filename, 'r') as obj_file:
        for raw_line in obj_file:
            if raw_line.startswith('#'):
                continue
            parts = raw_line.strip().split()
            if not parts:
                continue
            if parts[0] == 'vt':
                vt.append([float(val) for val in parts[1:]])
            elif parts[0] == 'f':
                tex_idx = []
                geo_idx = []
                for token in parts[1:]:
                    if not token:
                        continue
                    segments = token.split('/')
                    if len(segments) < 2:
                        continue
                    geo_idx.append(int(segments[0]))
                    tex_idx.append(int(segments[1]))
                if tex_idx:
                    ft.append(tex_idx)
                if geo_idx:
                    faces.append(geo_idx)
    vt = np.asarray(vt, dtype=np.float64)
    ft = np.asarray(ft, dtype=np.int64) - 1
    faces = np.asarray(faces, dtype=np.int64) - 1
    if vt.size:
        vt[:, 1] = 1.0 - vt[:, 1]
    return vt, ft, faces

def _resolve_mano_uv_root():
    base_path = Path(__file__).resolve()

    base_str = str(base_path)
    if '/gpfs/' in base_str:
        base_str = base_str.replace('/gpfs', '', 1)
        base_path = Path(base_str)

    candidates = [
        base_path.parents[1] / 'mano_uv',
        base_path.parents[2] / 'mano_uv',
        base_path.parents[3] / 'mano_uv',
        base_path.parents[3] / 'GuassianHand' / 'mano_uv',
    ]
    for candidate in candidates:
        change_file = candidate / 'change' / 'change_r.npy'
        obj_file = candidate / 'original mano template' / 'hand.obj'
        if change_file.exists() and obj_file.exists():
            print(f"Using MANO UV assets from: {candidate}")
            return candidate
    raise FileNotFoundError('Unable to locate mano_uv.')
