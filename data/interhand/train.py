# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..
# *************************************************************************


import os
from pathlib import Path
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
try:
    _torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    _orig_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if _torch_lib_dir not in _orig_ld.split(':'):
        os.environ['LD_LIBRARY_PATH'] = (
            (_orig_ld + ':' if _orig_ld else '') + _torch_lib_dir
        )
except Exception:
    pass

import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional

from plyfile import PlyData
import pandas as pd
from collections import Counter
from simple_knn._C import distCUDA2

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    MeshRenderer,
)
from pytorch3d.structures import Meshes

import copy
import sys
import pickle
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from data.utils.image_util import load_image
from tools_utils.map import *
from data.utils.hand_util import\
    body_pose_to_body_RTs,\
    get_canonical_global_tfms,\
    approx_gaussian_bone_volumes
from data.utils.camera_util import\
    apply_global_tfm_to_camera,\
    get_rays_from_KRT,\
    rays_intersect_3d_bbox,\
    cam2pixel
from data.utils.hand_util import MANOHand, INTERHAND2MANO
from data.utils.augm_util import process_bbox, augmentation, trans_point2d
import json
from pycocotools.coco import COCO
from tools_utils.model.ohta.configs import cfg
from data.interhand.handavatar.configs import cfg as cfg_handavatar
from tools_utils.model import smplx as smplx_
import tools_utils.model.smplx as smplx
import tools_utils.model.smplx_ours
import glob
import random
import math

import torch.distributed as dist
from data.hand_dataset import merge_batch
from tools_utils.camera_utils import loadCam_aug_bs
from pytorch3d.transforms import matrix_to_axis_angle
from tools_utils.model.smplx.manohd.subdivide import sub_mano
from tools_utils.model.smplx_ours.manohd.subdivide import sub_mano as sub_mano_ori
from data.hand_dataset import CameraInfo
from data.wild_hand_dataset import focal2fov, get_focalLength_from_fieldOfView, get_camera_parameters, get_bound_2d_mask
from torch.utils.data import Sampler
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

SEMANTIC_GROUPS = {
    0: 'palm',
    1: 'index',  2: 'index',  3: 'index',
    4: 'middle', 5: 'middle', 6: 'middle',
    7: 'pinky',  8: 'pinky',  9: 'pinky',
    10: 'ring',  11: 'ring',  12: 'ring',
    13: 'thumb', 14: 'thumb', 15: 'thumb',
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
    candidates = [
        base_path.parents[2] / 'runtime_assets' / 'mano_uv',
        base_path.parents[2] / 'mano_uv',
        base_path.parents[3] / 'mano_uv',
        base_path.parents[3] / 'GuassianHand' / 'mano_uv',
    ]
    for candidate in candidates:
        change_file = candidate / 'change' / 'change_r.npy'
        obj_file = candidate / 'original mano template' / 'hand.obj'
        if change_file.exists() and obj_file.exists():
            return candidate
    raise FileNotFoundError('Unable to locate runtime_assets/mano_uv.')

class Dataset(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(
            self,
            dataset_path,

            maxframes=-1,
            bgcolor=[0.0, 0.0, 0.0],

            skip=1,
            subject=None,
            data_type='train',

            ):

        print('[Dataset Path]', dataset_path)

        self.mano = smplx.create(**cfg.smpl_cfg)

        self.mano_ori = smplx_.create(**cfg.smpl_cfg)

        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights

        self.renderer = Renderer_mesh()

        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1

        self.phase = data_type
        print(f'[INFO] Phase: {self.phase}')

        if subject is None:
            if self.phase == 'train':
                subject = cfg.subject
            else:
                subject = 'train/prior_learning_data'

        print(f'[INFO] Subject: {subject}')

        if isinstance(subject, list):
            subject = subject[0]

        self.subject = subject
        self.height, self.width = 256, 256

        if data_type == 'train':
            self.num_frames = 5
        else:
            self.num_frames = 1

        self.epoch_size = 20000

        if 'prior_learning' in subject:
            action_root = os.path.join(
                dataset_path,
                'InterHand2.6M_5fps_batch1',
                'images',
                'train',
                'Capture0',
                '00*',
            )
            action = ['/' + os.path.basename(item) for item in glob.glob(action_root)]
            all_dir_name = [
                'train/Capture0',
                'train/Capture1',
                'train/Capture2',
                'train/Capture3',
                'train/Capture5',
                'train/Capture6',
                'train/Capture7',
                'train/Capture8',
                'train/Capture9',
                'train/Capture10',
                'train/Capture11',
                'train/Capture12',
                'train/Capture13',
                'train/Capture14',
                'train/Capture15',
                'train/Capture16',
                'train/Capture20',
                'train/Capture22',
                'train/Capture23',
                'train/Capture24',
                'train/Capture25',
                ]
            self.subject = all_dir_name
            all_dir_name = [x.split('/')[-1] + act for x in all_dir_name for act in action]

            test_split = ['0000_neutral_relaxed', '0009_thumbtucknormal', '0019_alligator_closed', '0029_indextip', '0039_fingerspreadrigid', '0048_index_point', '0058_middlefinger']
        else:
            all_dir_name = ['/'.join(subject.split('/')[1:])]
            test_split = [subject.split('/')[-1]]

        self.image_dir = os.path.join(dataset_path, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')

        anno_name = os.path.join(self.image_dir.replace('images', 'preprocess_ohta_our_full'), subject, 'anno_cam.pkl')
        print('Load annotation', anno_name)
        if not os.path.exists(anno_name):
            print('Preprocessing ...')
            self.preprocess(dataset_path, subject, anno_name, self.phase, all_dir_name)
        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)

        mano_res = self.mano(
            torch.zeros(1, 10).float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
            return_verts=True)
        self.canonical_bbox = mano_res.joints[0].numpy()
        self.canonical_verts = mano_res.vertices[0].numpy()
        self.canonical_bbox = self.skeleton_to_bbox(self.canonical_verts)
        big_pose_min_xyz = np.min(self.canonical_verts, axis=0)
        big_pose_max_xyz = np.max(self.canonical_verts, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        self.big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(self.canonical_verts)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]
        self.pcd_scales = torch.exp(scales)

        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)

        self.big_pose_smpl_param = big_pose_smpl_param
        self.cano_face_area = calc_face_areas(torch.tensor(self.canonical_verts), mano_res.faces_tensor)

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

        new_framelist = []
        for i in range(len(self.framelist)):
            action = self.framelist[i].split('/')[2]
            if self.phase == 'train':
                if action not in test_split:
                    new_framelist.append(self.framelist[i])
            else:
                if action in test_split:
                    new_framelist.append(self.framelist[i])
        self.framelist = new_framelist[:]

        self.framelist = self.framelist[::skip]

        try:
            exclude = cfg[data_type].get('exclude_idx', None)
        except:
            print(f'[WARNING] Invalid data_type: {data_type}, set exclude_idx: None')
            exclude = None

        if isinstance(exclude, list):
            sel_idx = list(range(len(self.framelist)))
            self.framelist = [self.framelist[i] for i in sel_idx if i not in exclude]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]

        self.frames_by_subject = defaultdict(list)
        for f in self.framelist:
            parts = f.split('/')
            subject_prefix = parts[0] + '/' + parts[1]
            self.frames_by_subject[subject_prefix].append(f)

        self.subject_ids = [s for s, lst in self.frames_by_subject.items() if len(lst) > 0]

        print(f' -- Total Frames: {self.get_total_frames()}')

        print(f'[INFO] Found {len(self.subject_ids)} subjects. Frames per subject for \033[91m{self.phase}\033[0m: min={min(len(self.frames_by_subject[s]) for s in self.subject_ids)}, max={max(len(self.frames_by_subject[s]) for s in self.subject_ids)}')

        self.bgcolor = bgcolor

        if A is not None:
            self.transform_c = A.Compose([A.CLAHE(p=0.3),
                                            A.FancyPCA(alpha=0.1, p=0.2),
                                            A.RandomGamma(p=0.3),
                                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                                            ], additional_targets={'mask':'mask'})
        else:
            self.transform_c = None
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

    @torch.no_grad()
    def image_semantic_mask(self, verts_cam, K, R, T, faces):
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

        masks = np.zeros((self.height, self.width), dtype=np.uint8)

        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        verts_img = np.round(verts_img[:, :2]).astype(np.int32)

        for tri in self.nail_faces:
            tri_pts = verts_img[tri, :]
            if np.any(tri_pts < 0) or np.any(tri_pts[:, 0] >= self.width) or np.any(tri_pts[:, 1] >= self.height):
                continue
            cv2.fillConvexPoly(masks, tri_pts, 255)

        return masks

    @torch.no_grad()
    def preprocess(self, dataset_path, subject, anno_name, data_type, all_dir_name, with_mask=True):
        th_hands_mean_right = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                                        0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                                        0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                                        -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                                        0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                                        -0.0266, -0.0529, 0.5356, -0.0460, 0.2774])
        th_hands_mean_left = th_hands_mean_right.copy().reshape(-1, 3)
        th_hands_mean_left[:, 1:] *= -1
        th_hands_mean_left = th_hands_mean_left.reshape(-1)

        phase = subject.split('/')[0]

        self.annot_path = os.path.join(dataset_path, 'annotations')
        print("Load annotation from  " + os.path.join(self.annot_path, phase))
        db = COCO(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_data.json'))
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_camera.json')) as f:
            cameras = json.load(f)
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_joint_3d.json')) as f:
            joints_interhand = json.load(f)
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)

        self.cameras = {}
        self.mesh_infos = {}
        self.bbox = {}
        self.framelist = []

        for i, aid in enumerate(db.anns.keys()):
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            if '/'.join(img['file_name'].split('/')[:2]) not in all_dir_name:
                continue
            if i%5000==0:
                print(i)
            capture_id = img['capture']
            cam = img['camera']
            frame_idx = img['frame_idx']
            image_name = img['file_name']
            hand_type = ann['hand_type']
            joint_valid = np.array(ann['joint_valid'])
            try:
                mano_param = mano_params[str(capture_id)][str(frame_idx)][self.handtype]
            except:
                print('cannot read mano params', image_name)
                continue
            if hand_type != self.handtype or mano_param is None:
                print(f'{i}, Discard {image_name}, {hand_type} is not agree with {self.handtype}')
                continue

            frame_name = img['file_name']
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'], dtype=np.float32)
            if data_type != 'infer':
                if bbox[0]<10 or bbox[1]<10 or max(bbox[2], bbox[3])<80 or bbox[0]+bbox[2]>img_width-10 or bbox[1]+bbox[3]>img_height-10:
                    continue

                img_path = os.path.join(self.image_dir, f'{phase}/{image_name}')
                img = cv2.imread(img_path)
                if img.max() < 20:
                    continue
                if np.allclose(img[..., 0], img[..., 1], atol=1) or np.allclose(img[..., 2], img[..., 1], atol=1) or np.allclose(img[..., 0], img[..., 2], atol=1):
                    continue

                mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                if not os.path.exists(mask_path):
                    continue
                mask = cv2.imread(img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png'))
                mask_sum = mask[..., 0].astype('bool').sum()
                if mask.max() < 255 or mask_sum < 3000:
                    continue

                mask_bool = mask[..., 0]==255
                sel_img = img[mask_bool].mean(axis=-1)
                if sel_img.max()<20:
                    print(i, frame_name, 'sel_img is too dark')
                    continue

            print(f'[INFO] Preprocessing {frame_name}')

            bbox = process_bbox(bbox, img_width, img_height)
            self.bbox[f'{phase}/{image_name}'] = bbox
            self.framelist.append(f'{phase}/{image_name}')

            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32),\
                            np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            cam_T, cam_R = campos, camrot
            E = np.eye(4)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            K = np.eye(3)
            K[[0, 1], [0, 1]] = focal
            K[[0, 1], [2, 2]] = princpt
            self.cameras[f'{phase}/{image_name}'] = {
                'intrinsics': K,
                'extrinsics': [cam_R, cam_T],
                'distortions': np.zeros(5)
            }

            poses = np.array(mano_param['pose'])
            betas = np.array(mano_param['shape'])
            Rh = poses[:3].copy()
            Rh_mat = np.dot(camrot, R.from_rotvec(Rh).as_matrix())
            Rh = R.from_matrix(Rh_mat).as_rotvec()
            poses[:3] = Rh
            poses[3:] += (th_hands_mean_left, th_hands_mean_right)[self.handtype=='right']
            tres = self.mano(
                torch.from_numpy(betas)[None].float(),
                torch.zeros(1, 3).float(),
                torch.zeros(1, 45).float(),
                return_verts=True)
            tpose_joints = tres.joints[0].numpy()
            tverts = tres.vertices[0].numpy()

            posed_res = self.mano(
                torch.from_numpy(betas)[None].float(),
                torch.from_numpy(poses)[None, :3].float(),
                torch.from_numpy(poses)[None, 3:].float(),
                return_verts=True)
            joints = posed_res.joints[0].numpy()
            verts = posed_res.vertices[0].numpy()

            joint_world = np.array(joints_interhand[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32) / 1000 * cfg.smpl_cfg.scale
            joint_cam = np.dot(camrot, joint_world.T).T - np.dot(camrot, campos) / 1000 * cfg.smpl_cfg.scale
            joint_cam = joint_cam[:21][INTERHAND2MANO] if self.handtype=='right' else joint_cam[21:][INTERHAND2MANO]
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2].astype('int32')
            joint_valid = joint_valid[:21][INTERHAND2MANO] if self.handtype=='right' else joint_valid[21:][INTERHAND2MANO]

            self.mesh_infos[f'{phase}/{image_name}'] = {
                'Rh': np.zeros(3),
                'Th': joint_cam[0] - joints[0],
                'poses': poses.reshape(-1),
                'shape': betas,
                'joints': joints,
                'tpose_joints': tpose_joints,
                'bbox': self.skeleton_to_bbox(verts),
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid
            }

        os.makedirs(os.path.dirname(anno_name), exist_ok=True)
        with open(anno_name, 'wb') as f:
            pickle.dump([self.cameras, self.mesh_infos, self.bbox, self.framelist], f)

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'shape': self.mesh_infos[frame_name]['shape'].astype('float32'),
            'dst_tpose_joints':\
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
            'joint_img': self.mesh_infos[frame_name]['joint_img'].astype('int32'),
            'joint_cam': self.mesh_infos[frame_name]['joint_cam'].astype('float32'),
            'joint_valid': self.mesh_infos[frame_name]['joint_valid'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        ray_alpha = ray_alpha[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, ray_alpha, near, far

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

    def load_image(self, frame_name, bg_color, use_mask=True):
        imagepath = os.path.join(self.image_dir, frame_name)
        orig_img = np.array(load_image(imagepath))

        if use_mask:
            if 'prior_learning' not in self.subject and self.phase == 'progress':
                maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                alpha_mask = np.array(load_image(maskpath))
            else:
                maskpath = imagepath.replace('images', 'masks_SAM').replace('.jpg', '.png')
                try:
                    alpha_mask = np.array(load_image(maskpath))
                except:
                    maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                    alpha_mask = np.array(load_image(maskpath))
        else:
            alpha_mask = np.ones_like(orig_img) * 255

        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        img = alpha_mask / 255. * orig_img + (1.0 - alpha_mask / 255.) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None,
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None,
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)

        return img, alpha_mask

    def get_total_frames(self):
        return len(self.framelist)

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
        return self.get_total_frames()

    def __getitem__(self, idx):
        n_sub = len(self.subject_ids)

        subject_idx = idx % n_sub

        subject_prefix = self.subject_ids[subject_idx]
        subject_frames = self.frames_by_subject[subject_prefix]
        chosen = random.sample(subject_frames, self.num_frames)
        id = subject_prefix

        cam_info_list = []

        for frame_name in chosen:
            img_path = os.path.join(self.image_dir, frame_name)
            mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')

            bgcolor = np.array(self.bgcolor, dtype='float32')

            img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])

            bbox = self.bbox[frame_name]
            img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(img, self.bbox[frame_name], 'eval',
                                                                                        exclude_flip=True,
                                                                                        input_img_shape=(256, 256), mask=alpha,
                                                                                        base_scale=1.3,
                                                                                        scale_factor=0.2,
                                                                                        rot_factor=0,
                                                                                        shift_wh=[bbox[2], bbox[3]],
                                                                                        gaussian_std=3,
                                                                                        bordervalue=bgcolor.tolist())

            img = (img / 255.).astype('float32')
            alpha = alpha.astype('float32') / 255.0

            dst_skel_info = self.query_dst_skeleton(frame_name)

            dst_poses = dst_skel_info['poses']
            dst_shape = dst_skel_info['shape']

            assert frame_name in self.cameras
            K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
            K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
            K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])

            focal_length_x = K[0, 0]
            focal_length_y = K[1, 1]
            FovX = focal2fov(focal_length_x, self.height)
            FovY = focal2fov(focal_length_y, self.width)

            E = self.cameras[frame_name]['extrinsics']

            E = apply_global_tfm_to_camera(
                    E=np.eye(4),
                    Rh=dst_skel_info['Rh'],
                    Th=dst_skel_info['Th'])
            R = E[:3, :3]
            T = E[:3, 3]

            posed_res = self.mano(
                torch.from_numpy(dst_shape)[None].float(),
                torch.from_numpy(dst_poses[:3])[None].float(),
                torch.from_numpy(dst_poses[3:])[None].float(),

            return_verts=True)

            cano_res = self.mano(
                torch.from_numpy(dst_shape)[None].float(),
                torch.zeros(1, 3).float(),
                torch.zeros(1, 45).float(),

            return_verts=True)
            self.canonical_verts = cano_res.vertices[0].detach().numpy()

            world_vertex = posed_res.vertices.detach().numpy()
            posed_vertices = world_vertex.reshape(-1, 3)
            uv_mapping = None
            if self._uv_template is not None:
                vert_uv_tensor = torch.from_numpy(
                    posed_vertices[self._uv_template['change_idx']].astype(np.float32)
                )
                uv_mapping = {
                    'vert_uv': vert_uv_tensor,
                    'face_uv': self._uv_template['face_idx'].clone(),
                    'face_uv_xy': self._uv_template['face_uv_xy'].clone(),
                }

            smpl_param ={
                'poses': dst_poses,
                'shape': dst_shape,
                'trans': torch.zeros(3).float(),
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
            nail_mask = nail_mask[..., None]

            verts_cam = np.dot(R, posed_vertices.T).T + T[None, :]
            verts_img = np.dot(K, verts_cam.T).T
            verts_img[:, :2] /= verts_img[:, 2:3]
            nail_img = (verts_img[:, :2]).astype(np.float32)

            cam_info = CameraInfo(
                        uid=id, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                        image=img,
                        nail_image=nail_img,
                        nail_mask=nail_mask,
                        verts_cam=verts_cam,
                        bound_mask=alpha,
                        bkgd_mask=alpha,
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

        img_aug = skin_tone_perturb_shared(final_results['original_image'])
        final_results['original_image'] = img_aug

        return final_results

class HandAvatarDataset(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(
            self,
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=[0.0, 0.0, 0.0],
            ray_shoot_mode='image',
            data_type='val',
            skip=200,
            subject=None,
            finetune=False,
            **kwargs):

        print('[Dataset Path]', dataset_path)

        self.mano = smplx.create(**cfg.smpl_cfg)
        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1

        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights

        self.renderer = Renderer_mesh()

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

        self.phase = kwargs.get('data_type', 'train')

        if subject is None:
            if self.phase == 'train':
                subject = cfg_handavatar.subject
            else:
                subject = cfg_handavatar[kwargs['data_type']].subject
        subject = 'test/Capture0/ROM03_RT_No_Occlusion'

        subject_1 = 'train/prior_learning_data'
        if isinstance(subject, list):
            subject = subject[0]
        self.image_dir = os.path.join(dataset_path, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')

        anno_name = os.path.join(self.image_dir.replace('images', 'preprocess'), subject, 'anno_cam.pkl')
        if not os.path.exists(anno_name):
            for preprocess_name in ('preprocess_ohta', 'preprocess_ohta_our_full'):
                alt_anno = os.path.join(self.image_dir.replace('images', preprocess_name), subject, 'anno_cam.pkl')
                if os.path.exists(alt_anno):
                    anno_name = alt_anno
                    break
        anno_name_1 = os.path.join(self.image_dir.replace('images', 'preprocess'), subject_1, 'anno_cam.pkl')
        img_prior = os.environ.get('LHM_HANDAVATAR_PRIOR_ROOT', dataset_path)
        img_prior_ = os.path.join(img_prior, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')
        self.img_prior_dir = img_prior_
        anno_name_1 = os.path.join(img_prior_.replace('images', 'preprocess_ohta_our_full'), subject_1, 'anno_cam.pkl')
        if not os.path.exists(anno_name_1):
            for preprocess_name in ('preprocess_ohta', 'preprocess'):
                alt_anno_1 = os.path.join(img_prior_.replace('images', preprocess_name), subject_1, 'anno_cam.pkl')
                if os.path.exists(alt_anno_1):
                    anno_name_1 = alt_anno_1
                    break

        print('Load annotation', anno_name)
        if not os.path.exists(anno_name):
            print('Preprocessing ...')
            self.preprocess(dataset_path, subject, anno_name, self.phase)
            exit(0)
        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)

        with open(anno_name_1, 'rb') as f:
            self.cameras_04, self.mesh_infos_04, self.bbox_04, self.framelist_04 = pickle.load(f)

        self.height, self.width = 256, 256

        mano_res = self.mano(
            torch.zeros(1, 10).float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
            return_verts=True)
        self.canonical_bbox = mano_res.joints[0].numpy()
        self.canonical_verts = mano_res.vertices[0].numpy()
        self.canonical_bbox = self.skeleton_to_bbox(self.canonical_verts)

        big_pose_min_xyz = np.min(self.canonical_verts, axis=0)
        big_pose_max_xyz = np.max(self.canonical_verts, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        self.big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)

        self.big_pose_smpl_param = big_pose_smpl_param
        self.cano_face_area = calc_face_areas(torch.tensor(self.canonical_verts), mano_res.faces_tensor)

        self.framelist = self.framelist[::skip]

        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        print(f' -- Total Frames: {self.get_total_frames()}')

        self.bgcolor = bgcolor
        self.num_frames = 5

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(self.canonical_verts)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]
        self.pcd_scales = torch.exp(scales)

        self._uv_template = self._build_right_hand_uv_template()
        self.finetune = finetune

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
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_image(self, frame_name, bg_color, use_mask=True):
        imagepath = os.path.join(self.image_dir, frame_name)
        orig_img = np.array(load_image(imagepath))

        maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
        alpha_mask = np.array(load_image(maskpath))

        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        img = alpha_mask / 255. * orig_img + (1.0 - alpha_mask / 255.) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None,
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None,
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)

        return img, alpha_mask

    def load_image_09(self, frame_name, bg_color, use_mask=True):
        imagepath = os.path.join(self.img_prior_dir, frame_name)
        orig_img = np.array(load_image(imagepath))

        maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
        alpha_mask = np.array(load_image(maskpath))

        if frame_name in self.cameras_04 and 'distortions' in self.cameras_04[frame_name]:
            K = self.cameras_04[frame_name]['intrinsics']
            D = self.cameras_04[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        img = alpha_mask / 255. * orig_img + (1.0 - alpha_mask / 255.) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None,
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None,
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)

        return img, alpha_mask

    def get_total_frames(self):
        return len(self.framelist)

    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'shape': self.mesh_infos[frame_name]['shape'].astype('float32'),
            'dst_tpose_joints':\
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
            'joint_img': self.mesh_infos[frame_name]['joint_img'].astype('int32'),
            'joint_cam': self.mesh_infos[frame_name]['joint_cam'].astype('float32'),
            'joint_valid': self.mesh_infos[frame_name]['joint_valid'].astype('float32')
        }

    def query_dst_skeleton_04(self, frame_name):
        return {
            'poses': self.mesh_infos_04[frame_name]['poses'].astype('float32'),
            'shape': self.mesh_infos_04[frame_name]['shape'].astype('float32'),
            'dst_tpose_joints':\
                self.mesh_infos_04[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos_04[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos_04[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos_04[frame_name]['Th'].astype('float32'),
            'joint_img': self.mesh_infos_04[frame_name]['joint_img'].astype('int32'),
            'joint_cam': self.mesh_infos_04[frame_name]['joint_cam'].astype('float32'),
            'joint_valid': self.mesh_infos_04[frame_name]['joint_valid'].astype('float32')
        }

    def __len__(self):
        return self.get_total_frames()

    def get_img(self, img_path='./data/image15012.jpg', mask_path='./data/image15012.png'):
        frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400272/image15012.jpg'

        bgcolor = np.array(self.bgcolor, dtype='float32')
        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])

        bbox = self.bbox[frame_name]
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(img, self.bbox[frame_name], 'eval',
                                                                                    exclude_flip=True,
                                                                                    input_img_shape=(256, 256), mask=alpha,
                                                                                    base_scale=1.3,
                                                                                    scale_factor=0.2,
                                                                                    rot_factor=0,
                                                                                    shift_wh=[bbox[2], bbox[3]],
                                                                                    gaussian_std=3,
                                                                                    bordervalue=bgcolor.tolist())
        img = (img / 255.).astype('float32')
        alpha = alpha.astype('float32') / 255.0

        dst_skel_info = self.query_dst_skeleton(frame_name)

        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']

        cam_info_list = []
        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])

        focal_length_x = K[1, 1]
        focal_length_y = K[0, 0]
        FovX = focal2fov(focal_length_x, self.height)
        FovY = focal2fov(focal_length_y, self.width)

        E = self.cameras[frame_name]['extrinsics']

        E = apply_global_tfm_to_camera(
                E=np.eye(4),
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        posed_res = self.mano(
            torch.from_numpy(dst_shape)[None].float(),
            torch.from_numpy(dst_poses[:3])[None].float(),
            torch.from_numpy(dst_poses[3:])[None].float(),
        return_verts=True)

        cano_res = self.mano(
            torch.from_numpy(dst_shape)[None].float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
        return_verts=True)
        self.canonical_verts = cano_res.vertices[0].detach().numpy()

        world_vertex = posed_res.vertices.detach().numpy()

        posed_vertices = world_vertex.reshape(-1, 3)

        uv_mapping = None
        if self._uv_template is not None:
            vert_uv_tensor = torch.from_numpy(
                posed_vertices[self._uv_template['change_idx']].astype(np.float32)
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

        verts_cam = world_vertex.reshape(-1, 3)
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None]

        depth_map = self.renderer(torch.from_numpy(world_vertex).cuda(), torch.from_numpy(K), torch.from_numpy(E), self.mano.faces)
        x = verts_img[:, 0]
        y = verts_img[:, 1]
        z = verts_cam[:, 2].reshape(-1)

        depth_map = np.asarray(depth_map.cpu())
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze()
        H, W = depth_map.shape
        depth_map[np.isnan(depth_map)] = np.inf

        mask_inside = (
            (x >= 0) & (x < W) &
            (y >= 0) & (y < H) &
            (z > 0)
        )

        x_valid = np.clip(x[mask_inside].astype(np.int32), 0, W - 1)
        y_valid = np.clip(y[mask_inside].astype(np.int32), 0, H - 1)

        z_buf = depth_map[y_valid, x_valid].reshape(-1)
        z_in = z[mask_inside].reshape(-1)

        visible_local = z_in <= (z_buf + 5e-2)

        visible_mask = np.zeros(world_vertex.shape[1], dtype=bool)
        visible_mask[mask_inside] = visible_local

        cam_info = CameraInfo(
                    uid=id, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                    image=img, nail_image=nail_img,
                    verts_cam=verts_cam,
                    nail_mask=nail_mask,
                    bound_mask=alpha,
                    bkgd_mask=alpha,
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

    def get_img_04(self, img_path='./data/image15012.jpg', mask_path='./data/image15012.png'):
        frame_name = 'train/Capture8/0007_thumbup_normal/cam400002/image2402.jpg'

        bgcolor = np.array(self.bgcolor, dtype='float32')
        img, alpha = self.load_image_09(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])

        bbox = self.bbox_04[frame_name]
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(img, self.bbox_04[frame_name], 'eval',
                                                                                    exclude_flip=True,
                                                                                    input_img_shape=(256, 256), mask=alpha,
                                                                                    base_scale=1.3,
                                                                                    scale_factor=0.2,
                                                                                    rot_factor=0,
                                                                                    shift_wh=[bbox[2], bbox[3]],
                                                                                    gaussian_std=3,
                                                                                    bordervalue=bgcolor.tolist())
        img = (img / 255.).astype('float32')
        alpha = alpha.astype('float32') / 255.0

        dst_skel_info = self.query_dst_skeleton_04(frame_name)

        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']

        cam_info_list = []
        assert frame_name in self.cameras_04
        K = self.cameras_04[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])

        focal_length_x = K[1, 1]
        focal_length_y = K[0, 0]
        FovX = focal2fov(focal_length_x, self.height)
        FovY = focal2fov(focal_length_y, self.width)

        E = self.cameras_04[frame_name]['extrinsics']

        E = apply_global_tfm_to_camera(
                E=np.eye(4),
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        posed_res = self.mano(
            torch.from_numpy(dst_shape)[None].float(),
            torch.from_numpy(dst_poses[:3])[None].float(),
            torch.from_numpy(dst_poses[3:])[None].float(),
        return_verts=True)

        cano_res = self.mano(
            torch.from_numpy(dst_shape)[None].float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
        return_verts=True)
        self.canonical_verts = cano_res.vertices[0].detach().numpy()

        world_vertex = posed_res.vertices.detach().numpy()

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

        verts_cam = world_vertex.reshape(-1, 3)
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None]

        depth_map = self.renderer(torch.from_numpy(world_vertex).cuda(), torch.from_numpy(K), torch.from_numpy(E), self.mano.faces)
        x = verts_img[:, 0]
        y = verts_img[:, 1]
        z = verts_cam[:, 2].reshape(-1)

        depth_map = np.asarray(depth_map.cpu())
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze()
        H, W = depth_map.shape
        depth_map[np.isnan(depth_map)] = np.inf

        mask_inside = (
            (x >= 0) & (x < W) &
            (y >= 0) & (y < H) &
            (z > 0)
        )

        x_valid = np.clip(x[mask_inside].astype(np.int32), 0, W - 1)
        y_valid = np.clip(y[mask_inside].astype(np.int32), 0, H - 1)

        z_buf = depth_map[y_valid, x_valid].reshape(-1)
        z_in = z[mask_inside].reshape(-1)

        visible_local = z_in <= (z_buf + 5e-2)

        visible_mask = np.zeros(world_vertex.shape[1], dtype=bool)
        visible_mask[mask_inside] = visible_local

        cam_info = CameraInfo(
                    uid=id, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                    image=img, nail_image=nail_img,
                    verts_cam=verts_cam,
                    nail_mask=nail_mask,
                    bound_mask=alpha,
                    bkgd_mask=alpha,
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
        cam_info_list.append(cam_info)
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)

        return final_results

    @torch.no_grad()
    def image_semantic_mask(self, verts_cam, K, R, T, faces):
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

        masks = np.zeros((self.height, self.width), dtype=np.uint8)

        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        verts_img = np.round(verts_img[:, :2]).astype(np.int32)

        for tri in self.nail_faces:
            tri_pts = verts_img[tri, :]
            if np.any(tri_pts < 0) or np.any(tri_pts[:, 0] >= self.width) or np.any(tri_pts[:, 1] >= self.height):
                continue
            cv2.fillConvexPoly(masks, tri_pts, 255)

        return masks

    def __getitem__(self, idx):
        if self.finetune:
            return self.get_img()

        frame_name = self.framelist[idx]

        img_path = os.path.join(self.image_dir, frame_name)
        mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')

        bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])

        bbox = self.bbox[frame_name]
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(img, self.bbox[frame_name], 'eval',
                                                                                    exclude_flip=True,
                                                                                    input_img_shape=(256, 256), mask=alpha,
                                                                                    base_scale=1.3,
                                                                                    scale_factor=0.2,
                                                                                    rot_factor=0,
                                                                                    shift_wh=[bbox[2], bbox[3]],
                                                                                    gaussian_std=3,
                                                                                    bordervalue=bgcolor.tolist())

        img = (img / 255.).astype('float32')
        alpha = alpha.astype('float32') / 255.0

        dst_skel_info = self.query_dst_skeleton(frame_name)

        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])

        focal_length_x = K[1, 1]
        focal_length_y = K[0, 0]
        FovX = focal2fov(focal_length_x, self.height)
        FovY = focal2fov(focal_length_y, self.width)

        E = self.cameras[frame_name]['extrinsics']

        E = apply_global_tfm_to_camera(
                E=np.eye(4),
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        posed_res = self.mano(
            torch.from_numpy(dst_shape)[None].float(),
            torch.from_numpy(dst_poses[:3])[None].float(),
            torch.from_numpy(dst_poses[3:])[None].float(),
        return_verts=True)

        cano_res = self.mano(
            torch.from_numpy(dst_shape)[None].float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
        return_verts=True)
        self.canonical_verts = cano_res.vertices[0].detach().numpy()

        world_vertex = posed_res.vertices.detach().numpy()
        posed_vertices = world_vertex.reshape(-1, 3)

        uv_mapping = None
        if self._uv_template is not None:
            vert_uv_tensor = torch.from_numpy(
                posed_vertices[self._uv_template['change_idx']].astype(np.float32)
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

        verts_cam = world_vertex.reshape(-1, 3)
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None]

        cam_info = CameraInfo(
                    uid=id, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                    image=img, nail_image=nail_img,
                    nail_mask=nail_mask,
                    bound_mask=alpha,
                    verts_cam=verts_cam,
                    bkgd_mask=alpha,
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
        cam_info_list = []
        cam_info = loadCam_aug_bs(id, cam_info)
        if uv_mapping is not None:
            cam_info.vert_uv = uv_mapping['vert_uv']
            cam_info.face_uv = uv_mapping['face_uv']
            cam_info.face_uv_xy = uv_mapping['face_uv_xy']
        cam_info_list.append(cam_info)
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)

        return final_results

def compute_visibility_with_radius_discrete(verts_cam, proj_xy, H, W, radius_px_per_point, z_eps=1e-3, chunk=100000):
    """
    verts_cam: (N,3) torch
    proj_xy: (N,2) torch (float pixel coords)
    radius_px_per_point: (N,) torch or scalar, radius in pixels (>=0)
    返回: depth_buffer (H,W), visible_mask (N,)
    """
    device = verts_cam.device
    dtype = verts_cam.dtype
    N = verts_cam.shape[0]

    if isinstance(radius_px_per_point, (int, float)) or (torch.is_tensor(radius_px_per_point) and radius_px_per_point.ndim==0):
        radius_px = torch.full((N,), float(radius_px_per_point), device=device, dtype=dtype)
    else:
        radius_px = radius_px_per_point.to(device=device, dtype=dtype)

    HxW = H * W
    depth_flat = torch.full((HxW,), float('inf'), device=device, dtype=dtype)

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        px = proj_xy[start:end]
        z = verts_cam[start:end, 2]
        r = radius_px[start:end]

        C = px.shape[0]

        max_r = int(math.ceil(r.max().item()))
        if max_r == 0:
            us = torch.round(px[:,0]).long().clamp(0, W-1)
            vs = torch.round(px[:,1]).long().clamp(0, H-1)
            inds = vs * W + us
            depth_flat.scatter_reduce_(0, inds, z, reduce='amin')
            continue

        offsets = torch.stack(torch.meshgrid(torch.arange(-max_r, max_r+1, device=device),
                                            torch.arange(-max_r, max_r+1, device=device)), dim=-1).reshape(-1,2)
        K = offsets.shape[0]

        px_exp = px.unsqueeze(1).expand(-1, K, 2).reshape(-1,2)
        offs_exp = offsets.unsqueeze(0).expand(C, K, 2).reshape(-1,2)

        pxy = px_exp + offs_exp

        r_rep = r.unsqueeze(1).expand(-1, K).reshape(-1)
        dx = pxy[:,0] - px_exp[:,0]
        dy = pxy[:,1] - px_exp[:,1]
        dist2 = dx*dx + dy*dy
        mask = dist2 <= (r_rep + 1e-9)**2

        if mask.sum() == 0:
            continue
        valid_px = pxy[mask]

        u = torch.round(valid_px[:,0]).long().clamp(0, W-1)
        v = torch.round(valid_px[:,1]).long().clamp(0, H-1)
        inds = v * W + u

        z_rep = z.unsqueeze(1).expand(-1, K).reshape(-1)[mask]

        depth_flat.scatter_reduce_(0, inds, z_rep, reduce='amin')

    depth_buffer = depth_flat.view(H, W)

    depth_at_point = torch.full((N,), float('inf'), device=device, dtype=dtype)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        px = proj_xy[start:end]
        r = radius_px[start:end]
        C = px.shape[0]
        max_r = int(math.ceil(r.max().item()))
        if max_r == 0:
            us = torch.round(px[:,0]).long().clamp(0, W-1)
            vs = torch.round(px[:,1]).long().clamp(0, H-1)
            depth_at_point[start:end] = depth_buffer[vs, us]
            continue
        offsets = torch.stack(torch.meshgrid(torch.arange(-max_r, max_r+1, device=device),
                                            torch.arange(-max_r, max_r+1, device=device)), dim=-1).reshape(-1,2)
        K = offsets.shape[0]
        px_exp = px.unsqueeze(1).expand(-1, K, 2).reshape(-1,2)
        offs_exp = offsets.unsqueeze(0).expand(C, K, 2).reshape(-1,2)
        pxy = px_exp + offs_exp
        r_rep = r.unsqueeze(1).expand(-1, K).reshape(-1)
        dx = pxy[:,0] - px_exp[:,0]; dy = pxy[:,1] - px_exp[:,1]
        dist2 = dx*dx + dy*dy
        mask = dist2 <= (r_rep + 1e-9)**2
        if mask.sum() == 0:
            continue
        valid_px = pxy[mask]
        u = torch.round(valid_px[:,0]).long().clamp(0, W-1)
        v = torch.round(valid_px[:,1]).long().clamp(0, H-1)
        inds = v * W + u

        vals = depth_flat[inds]

        point_idx_rep = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, K).reshape(-1)[mask]

        for i in range(start, end):
            sel = (point_idx_rep == i)
            if sel.sum()>0:
                depth_at_point[i] = torch.min(vals[sel])
            else:
                u0 = int(round(px[i-start,0].item()))
                v0 = int(round(px[i-start,1].item()))
                u0 = max(0, min(W-1,u0)); v0 = max(0, min(H-1,v0))
                depth_at_point[i] = depth_buffer[v0, u0]

    z_all = verts_cam[:,2]
    visible_mask = (z_all <= depth_at_point + z_eps)
    return depth_buffer, visible_mask

def compute_visibility_from_depth(verts_cam, proj_xy, H, W, eps=1e-3):
    """
    输入:
      verts_cam: (N,3) 相机坐标系下的顶点 (x,y,z)
      proj_xy:   (N,2) 对应的像素坐标 (u,v)
      H, W:      图像分辨率
      eps:       容差（判断z相等的阈值）
    输出:
      depth_buffer: (H,W) 每个像素的最小深度
      visible_mask: (N,) 每个点是否可见 (True=可见)
    """
    device = verts_cam.device
    dtype = verts_cam.dtype

    z = verts_cam[:,2]
    u = proj_xy[:,0]; v = proj_xy[:,1]

    u_int = torch.round(u).long().clamp(0, W-1)
    v_int = torch.round(v).long().clamp(0, H-1)
    inds = v_int * W + u_int

    depth_flat = torch.full((H*W,), float('inf'), device=device, dtype=dtype)

    depth_flat.scatter_reduce_(0, inds, z, reduce='amin', include_self=True)
    depth_buffer = depth_flat.view(H, W)

    depth_at_point = depth_buffer[v_int, u_int]

    visible_mask = (z <= depth_at_point + eps)

    return depth_buffer, visible_mask

def clamp01(x: torch.Tensor):
    return x.clamp(0.0, 1.0)

def skin_tone_perturb_shared(
    imgs: torch.Tensor,
    p = 0.5,
    brightness: float = 0.05,
    contrast: float = 0.05,
    saturation: float = 0.06,
    hue: float = 0.02,
    wb_strength: float = 0.02,
    gamma_min: float = 0.98,
    gamma_max: float = 1.02,
):
    """
    对同一 subject 的多张图片执行一致的轻微肤色扰动。
    imgs: [B, 3, H, W], 归一化到 [0,1]
    """

    if random.random() > p:
        return imgs

    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast, contrast)
    s = 1.0 + random.uniform(-saturation, saturation)
    h = random.uniform(-hue, hue)
    gamma = random.uniform(gamma_min, gamma_max)

    r_noise = random.uniform(-wb_strength, wb_strength) + wb_strength * 0.2
    g_noise = random.uniform(-wb_strength, wb_strength) + wb_strength * 0.1
    b_noise = random.uniform(-wb_strength, wb_strength) - wb_strength * 0.1

    imgs_aug = TF.adjust_brightness(imgs, b)
    imgs_aug = TF.adjust_contrast(imgs_aug, c)
    imgs_aug = TF.adjust_saturation(imgs_aug, s)
    imgs_aug = TF.adjust_hue(imgs_aug, h)
    imgs_aug = TF.adjust_gamma(imgs_aug, gamma)

    imgs_aug = torch.stack([
        (imgs_aug[:, 0] * (1 + r_noise)).clamp(0, 1),
        (imgs_aug[:, 1] * (1 + g_noise)).clamp(0, 1),
        (imgs_aug[:, 2] * (1 + b_noise)).clamp(0, 1),
    ], dim=1)

    return imgs_aug

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

def ndc_T_world(xyzs_world, K, E, H, W):
    E = E.cuda()
    K = K.cuda()
    xyzs_cam = cam_T_world(xyzs_world, E)
    xys_2d = img_T_cam(xyzs_cam, K)

    if H < W:
        xs = -((xys_2d[:, 0, :] / H) * 2. - (W / H))
        ys = -((xys_2d[:, 1, :] / H) * 2. - 1.)
    else:
        xs = -((xys_2d[:, 0, :] / W) * 2. - 1.)
        ys = -((xys_2d[:, 1, :] / W) * 2. - (H / W))

    zs = xyzs_cam[:, 2]
    xyzs_ndc = torch.stack([xs, ys, zs], dim=-1)
    return xyzs_ndc

def cam_T_world(xyzs_world, E):
    E=E.unsqueeze(0).float()
    xyzs_world_ = torch.cat([xyzs_world, torch.ones_like(xyzs_world[:, :1])], dim=1)
    xyzs_cam_ = torch.bmm(E, xyzs_world_)
    xyzs_cam = xyzs_cam_[:, :3] / xyzs_cam_[:, 3:]
    return xyzs_cam

def img_T_cam(xyzs_cam, K):
    K = K.unsqueeze(0).float()
    xys_ = torch.bmm(K, xyzs_cam)
    xys = xys_[:, :2] / xys_[:, 2:]
    return xys

def compute_visibility_fast_np(verts_cam, proj_xy, pcd_scales_world, K, H=256, W=256, z_eps=1e-3):
    """
    Input: numpy arrays (or torch tensors; will be converted)
    Returns dict with r_px, depth_buf (H,W), depth_at_point (N,), residual (z - depth_at_point), vis_mask
    """

    if hasattr(verts_cam, 'cpu'): verts_cam = verts_cam.detach().cpu().numpy()
    if hasattr(proj_xy, 'cpu'): proj_xy = proj_xy.detach().cpu().numpy()
    if hasattr(pcd_scales_world, 'cpu'): pcd_scales_world = pcd_scales_world.detach().cpu().numpy().reshape(-1)
    verts_cam = np.asarray(verts_cam)
    proj_xy = np.asarray(proj_xy)
    pcd_scales_world = np.asarray(pcd_scales_world).reshape(-1)
    K = np.asarray(K)
    N = proj_xy.shape[0]

    z = np.maximum(verts_cam[:,2].astype(np.float64), 1e-6)
    fx = float(K[0,0])
    r_px = (pcd_scales_world * fx) / z

    r_px = np.clip(r_px, 0.5, max(1.0, np.percentile(r_px, 98)))
    r_int = np.ceil(r_px).astype(int)

    depth_flat = np.full((H*W,), np.inf, dtype=np.float64)
    ux = proj_xy[:,0].astype(np.float64)
    uy = proj_xy[:,1].astype(np.float64)
    u_floor = np.floor(ux).astype(int)
    v_floor = np.floor(uy).astype(int)

    unique_r = np.unique(r_int)

    for r in unique_r:
        idxs = np.where(r_int == r)[0]
        if idxs.size == 0: continue
        if r <= 0:
            u0 = np.clip(u_floor[idxs], 0, W-1); v0 = np.clip(v_floor[idxs], 0, H-1)
            flat_idxs = v0 * W + u0
            np.minimum.at(depth_flat, flat_idxs, z[idxs])
            continue
        rr = int(r)
        xs = np.arange(-rr, rr+1)
        ys = np.arange(-rr, rr+1)
        xu, yv = np.meshgrid(xs, ys)
        mask_circle = (xu*xu + yv*yv) <= (r*r + 1e-9)
        dxs = xu[mask_circle].ravel(); dys = yv[mask_circle].ravel()
        Koff = dxs.size
        u_centers = u_floor[idxs]
        v_centers = v_floor[idxs]

        u_t = (u_centers[:,None] + dxs[None,:]).astype(int)
        v_t = (v_centers[:,None] + dys[None,:]).astype(int)

        inb = (u_t >= 0) & (u_t < W) & (v_t >= 0) & (v_t < H)
        if not inb.any():
            continue
        flat_inds = (v_t * W + u_t).ravel()
        mask_flat = inb.ravel()
        flat_inds_valid = flat_inds[mask_flat]
        z_rep = np.repeat(z[idxs], Koff).ravel()
        z_valid = z_rep[mask_flat]
        np.minimum.at(depth_flat, flat_inds_valid, z_valid)

    depth_buf = depth_flat.reshape(H, W)

    depth_at_point = np.full((N,), np.inf, dtype=np.float64)
    for r in unique_r:
        idxs = np.where(r_int == r)[0]
        if idxs.size == 0: continue
        if r <= 0:
            u0 = np.clip(u_floor[idxs], 0, W-1); v0 = np.clip(v_floor[idxs], 0, H-1)
            depth_at_point[idxs] = depth_buf[v0, u0]
            continue
        rr = int(r)
        xs = np.arange(-rr, rr+1); ys = np.arange(-rr, rr+1)
        xu, yv = np.meshgrid(xs, ys)
        mask_circle = (xu*xu + yv*yv) <= (r*r + 1e-9)
        dxs = xu[mask_circle].ravel(); dys = yv[mask_circle].ravel()
        Koff = dxs.size
        u_centers = u_floor[idxs]; v_centers = v_floor[idxs]
        u_t = (u_centers[:,None] + dxs[None,:]).astype(int)
        v_t = (v_centers[:,None] + dys[None,:]).astype(int)

        u_t_cl = np.clip(u_t, 0, W-1); v_t_cl = np.clip(v_t, 0, H-1)
        flat_inds = (v_t_cl * W + u_t_cl)
        vals = depth_flat[flat_inds.ravel()].reshape(flat_inds.shape)

        mins = np.min(vals, axis=1)
        depth_at_point[idxs] = mins

    inf_mask = ~np.isfinite(depth_at_point)
    if inf_mask.any():
        uc = np.clip(np.round(ux[inf_mask]).astype(int), 0, W-1)
        vc = np.clip(np.round(uy[inf_mask]).astype(int), 0, H-1)
        depth_at_point[inf_mask] = depth_buf[vc, uc]

    residual = z - depth_at_point
    vis_mask = residual <= z_eps

    return dict(r_px=r_px, z=z, depth_buf=depth_buf, depth_at_point=depth_at_point, residual=residual, vis_mask=vis_mask)

def generate_uv_mask(uv_mapping, uv_resolution=256):
    """
    Generate a binary mask for the UV map indicating which pixels are valid (foreground).

    Args:
        uv_mapping: dict with keys 'face_uv_xy' containing triangle vertices in UV space [n_faces, 3, 2]
        uv_resolution: resolution of the UV map (assumed to be square)

    Returns:
        uv_mask: [uv_resolution, uv_resolution, 1] binary mask (1 for valid, 0 for background)
    """
    if uv_mapping is None or 'face_uv_xy' not in uv_mapping:
        return None

    import cv2

    uv_mask = np.zeros((uv_resolution, uv_resolution, 1), dtype=np.uint8)

    face_uv_xy = uv_mapping['face_uv_xy']
    if isinstance(face_uv_xy, torch.Tensor):
        face_uv_xy = face_uv_xy.detach().cpu().numpy()

    face_uv_xy_pixels = face_uv_xy * (uv_resolution - 1)

    for triangle in face_uv_xy_pixels:
        pts = np.array(triangle, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(uv_mask, [pts], 255)

    return uv_mask.astype(np.float32) / 255.0

class Renderer:
    def __init__(self, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """

        self.focal_length = 5000

        self.img_res = 256

        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.faces_left = self.faces[:,[0,2,1]]
        render_res = [self.img_res, self.img_res]
        self.renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)

        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        self.camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

    def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = True,
                imgname: Optional[str] = None,
                side_view=False, rot_angle=90,
                mesh_base_color=(0.25098039,  0.274117647,  0.65882353),
                scene_bg_color=(1,1,1),
                return_rgba=False,
                ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1,0,0], rot_angle=0, is_right=1):

        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
            mesh.export('./hand/mesh.ply')
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)

        rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t = None,
            rot=None,
            rot_axis=[1,0,0],
            rot_angle=0,
            camera_z=3,

            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=True,
        ):

        focal_length = self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array([0,0,0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)

        camera_node = pyrender.Node(camera=self.camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        self.renderer.delete()

        return color

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)

        camera_node = pyrender.Node(camera=self.camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose

            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

class Renderer_mesh(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cameras = FoVOrthographicCameras(R=(torch.eye(3, dtype=torch.float32)[None, ...]).cuda(),
                                         T=torch.zeros((1, 3), dtype=torch.float32).cuda(),
                                         znear=[0.01],
                                         zfar=[100],
                                         device='cuda'
                                         )

        raster_settings = RasterizationSettings(
            image_size=(256, 256),
            blur_radius=0.0,

            bin_size=0,
        )
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        self.soft_mask = True

        self.sigma = 1e-5
        raster_settings_soft = RasterizationSettings(
            image_size=(256, 256),
            blur_radius=np.log(1. / 1e-4 - 1.) * self.sigma,
            faces_per_pixel=50,
            bin_size=0,
        )
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_soft
            ),
            shader=SoftSilhouetteShader()
        )

    def forward(self, xyzs_observation, K, E, faces):
        xyzs_ndc = ndc_T_world(xyzs_observation.permute(0,2,1), K, E, 256, 256)

        mesh = Meshes(xyzs_ndc, torch.tensor(faces).unsqueeze(0).cuda())

        fragments = self.rasterizer(mesh)

        depth_map = fragments.zbuf[0, :, :, 0]

        depth_map[torch.isnan(depth_map)] = 0.0

        return depth_map
