# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2024) B-
# ytedance Inc..
# *************************************************************************


import os
from pathlib import Path
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Ensure PyTorch CUDA libraries are discoverable before importing simple_knn.
# This avoids "libc10_cuda.so: cannot open shared object file" when the
# debugger/launcher does not propagate LD_LIBRARY_PATH.
import torch  # noqa: E402
try:
    _torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    _orig_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if _torch_lib_dir not in _orig_ld.split(':'):
        os.environ['LD_LIBRARY_PATH'] = (
            (_orig_ld + ':' if _orig_ld else '') + _torch_lib_dir
        )
except Exception:
    # Fallback silently; ImportError will still surface if CUDA libs are truly missing.
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
# import faiss
import copy
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import pickle
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from data.utils.image_util import load_image
from tools_utils.map import *
from data.utils.hand_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from data.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox, \
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
from data.hand_dataset import create_dataset, merge_batch
from tools_utils.camera_utils import loadCam_aug_bs
from pytorch3d.transforms import matrix_to_axis_angle
from tools_utils.model.smplx.manohd.subdivide import sub_mano
from tools_utils.model.smplx_ours.manohd.subdivide import sub_mano as sub_mano_ori
from data.hand_dataset import CameraInfo
from data.debug import focal2fov, get_focalLength_from_fieldOfView, get_camera_parameters, get_bound_2d_mask
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from torch.utils.data import Sampler
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

from data.hanco.utils import *
from data.hanco.utils.mano_utils import pred_to_mano, project, trafoPoints

HANCO_CAMS = [3,5,6,7]

SEMANTIC_GROUPS = {
    0: 'palm',                # wrist
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
        base_path.parents[2] / 'mano_uv',
        base_path.parents[3] / 'mano_uv',
        base_path.parents[3] / 'GuassianHand' / 'mano_uv',
    ]
    for candidate in candidates:
        change_file = candidate / 'change' / 'change_r.npy'
        obj_file = candidate / 'original mano template' / 'hand.obj'
        if change_file.exists() and obj_file.exists():
            return candidate
    raise FileNotFoundError('Unable to locate mano_uv assets. Please place the directory inside lhm-hand-new or alongside GuassianHand.')


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, ratios):
        self.datasets = datasets
        self.ratios = np.array(ratios) / sum(ratios)
        self.lengths = [len(d) for d in datasets]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        ds_idx = np.random.choice(len(self.datasets), p=self.ratios)
        sub_idx = np.random.randint(len(self.datasets[ds_idx]))
        sample = self.datasets[ds_idx][sub_idx]
        # 注入标签（推荐同时给 int 和 str）
        sample["dataset_id"] = ds_idx   # 0 是 interhand, 1 是 HanCo
        return sample

        # --- 注入标签（推荐同时给 int 和 str）---
        sample["dataset_id"] = ds_idx   # 0 是 interhand, 1 是 HanCo
        # sample["dataset_name"] = self.names[ds_idx]
        return sample


class Dataset(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(
            self,
            dataset_path,
            # keyfilter=None,
            maxframes=-1,
            bgcolor=[0.0, 0.0, 0.0],
            # ray_shoot_mode='image',
            skip=1,
            subject=None,
            data_type='train',
            # **kwargs
            ):

        print('[Dataset Path]', dataset_path)

        # MANO
        self.mano = smplx.create(**cfg.smpl_cfg)
        # self.mano_ori = copy.deepcopy(self.mano)  # ✅ 深拷贝
        self.mano_ori = smplx_.create(**cfg.smpl_cfg)

        # MANO-HD ===================================
        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights
        # MANO-HD ===================================

        # self.renderer = Renderer(faces=self.mano.faces)
        self.renderer = Renderer_mesh()

        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1

        # annotation
        self.phase = data_type
        print(f'[INFO] Phase: {self.phase}')

        if subject is None:
            if self.phase == 'train':
                subject = cfg.subject
            else:
                # subject = cfg[kwargs['data_type']].subject
                subject = 'train/prior_learning_data'

        print(f'[INFO] Subject: {subject}')

        if isinstance(subject, list):
            subject = subject[0]

        self.subject = subject
        self.height, self.width = 256, 256

        if data_type == 'train':
            self.num_frames = 5 #4
        else:
            self.num_frames = 1

        self.epoch_size = 20000

        # for prior learning
        if 'prior_learning' in subject:
            action = ['/'+ item.split('/')[-1] for item in glob.glob('/home/z/zh174/interhand/InterHand2.6M_5fps_batch1/images/train/Capture0/00*')]
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
            # select train and test
            test_split = ['0000_neutral_relaxed', '0009_thumbtucknormal', '0019_alligator_closed', '0029_indextip', '0039_fingerspreadrigid', '0048_index_point', '0058_middlefinger']
        else:
            all_dir_name = ['/'.join(subject.split('/')[1:])]
            test_split = [subject.split('/')[-1]]

        # print('[INFO] All using sequence:', all_dir_name)
        # print('[INFO] Testing split:', test_split)

        self.image_dir = os.path.join(dataset_path, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')
        # anno_name = os.path.join(self.image_dir.replace('images', 'preprocess_ohta_our'), subject, 'anno_cam.pkl')
        anno_name = os.path.join(self.image_dir.replace('images', 'preprocess_ohta_our_full'), subject, 'anno_cam.pkl')
        print('Load annotation', anno_name)
        if not os.path.exists(anno_name):
            print('Preprocessing ...')
            self.preprocess(dataset_path, subject, anno_name, self.phase, all_dir_name)
        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)
        # canonical
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
        scales = torch.log(torch.sqrt(dist2))[...,None]#.repeat(1, 3)
        self.pcd_scales = torch.exp(scales)

        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)
        # big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        # big_pose_smpl_param['faces'] = self.faces.detach()  # 1538,3
        self.big_pose_smpl_param = big_pose_smpl_param
        self.cano_face_area = calc_face_areas(torch.tensor(self.canonical_verts), mano_res.faces_tensor)
        # self.aug = True
        # exit()

        # 加载每个顶点的语义标签
        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        print(f'Loading MANO semantic labels from {labels_path}')
        ply = PlyData.read(labels_path)
        if ply.elements:
            pc = pd.DataFrame(ply.elements[0].data).values
        self.labels = torch.tensor(pc[:, 6].astype(np.uint8))  # (N,)

        # SEMANTIC_GROUPS = {
        #     0: 'palm',                # wrist
        #     1: 'index',  2: 'index',  3: 'index',
        #     4: 'middle', 5: 'middle', 6: 'middle',
        #     7: 'pinky',  8: 'pinky',  9: 'pinky',
        #     10: 'ring',  11: 'ring',  12: 'ring',
        #     13: 'thumb', 14: 'thumb', 15: 'thumb',
        # }    # nail: 4, 7, 10, 13, 16

        # map16to6 = torch.tensor([
        #     0, 1,1,1,   # index
        #     2,2,2,      # middle
        #     3,3,3,      # pinky
        #     4,4,4,      # ring
        #     5,5,5       # thumb
        # ])  # wrist可选合并到 palm=0


        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        # 找出所有指甲顶点
        nail_indices = np.isin(self.labels, list(NAIL_PARTS.values()))
        # self.nail_coords = pc_coords[nail_indices]
        # self.objects_dc = F.one_hot((self.labels-1).to(torch.int64), num_classes=16).unsqueeze(1).to(torch.float32).cuda()
        # 找到哪些点是指甲
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

        # post process
        self.framelist = self.framelist[::skip]
        # data_type = kwargs['data_type']
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

        # 构建 subject -> frames 映射，subject_prefix 格式与 __getitem__ 中保持一致 "phase/CaptureX"
        self.frames_by_subject = defaultdict(list)
        for f in self.framelist:
            parts = f.split('/')
            subject_prefix = parts[0] + '/' + parts[1]   # e.g., "train/Capture0"
            self.frames_by_subject[subject_prefix].append(f)
        # 列出所有 subject id（去掉那些帧数为0的）
        self.subject_ids = [s for s, lst in self.frames_by_subject.items() if len(lst) > 0]

        print(f' -- Total Frames: {self.get_total_frames()}')
        # 可选：统计信息打印
        print(f'[INFO] Found {len(self.subject_ids)} subjects. Frames per subject for \033[91m{self.phase}\033[0m: min={min(len(self.frames_by_subject[s]) for s in self.subject_ids)}, max={max(len(self.frames_by_subject[s]) for s in self.subject_ids)}')
        # exit()
        # self.keyfilter = keyfilter
        self.bgcolor = bgcolor
        # self.ray_shoot_mode = ray_shoot_mode
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
        # h, w = image_size
        masks = np.zeros((self.height, self.width), dtype=np.uint8)

        # 相机变换
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        verts_img = np.round(verts_img[:, :2]).astype(np.int32)   # 12337, 2

        # # 计算每个面所属的语义类别（取面顶点标签的众数）
        # face_labels = self.labels[faces].numpy()
        # per_face_class = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, face_labels)

        # # 绘制mask
        # for f_idx, f in enumerate(faces):
        #     cls = int(per_face_class[f_idx])
        #     tri = verts_img[f, :]
        #     if np.any(tri < 0) or np.any(tri[:, 0] >= self.width) or np.any(tri[:, 1] >= self.height):
        #         continue
        #     cv2.fillConvexPoly(masks[cls], tri, 255)


         # 绘制指甲区域
        for tri in self.nail_faces:
            tri_pts = verts_img[tri, :]
            if np.any(tri_pts < 0) or np.any(tri_pts[:, 0] >= self.width) or np.any(tri_pts[:, 1] >= self.height):
                continue
            cv2.fillConvexPoly(masks, tri_pts, 255)

        return masks


    # def image_semantic_mask(self, verts_img, faces, image_size):
    #     """
    #     可选：快速在CPU生成一张6通道的mask，不保存，只返回numpy。
    #     verts_img: (N_v, 2) 图像空间坐标
    #     faces: (N_f, 3)
    #     image_size: (H, W)
    #     """

    #     semantic_masks = np.zeros((2, self.height, self.width), dtype=np.uint8)
    #     face_labels = self.nail_labels[faces]
    #     per_face_class = torch.mode(face_labels, dim=1)[0].cpu().numpy()

    #     verts_img_int = verts_img.astype(np.int32)
    #     for f_idx, f in enumerate(faces):
    #         label = int(per_face_class[f_idx])
    #         tri = np.array([[verts_img_int[f[0]][0], verts_img_int[f[0]][1]],
    #                         [verts_img_int[f[1]][0], verts_img_int[f[1]][1]],
    #                         [verts_img_int[f[2]][0], verts_img_int[f[2]][1]]])
    #         cv2.fillConvexPoly(semantic_masks[label], tri, 255)

    #     return semantic_masks  # (6, H, W)


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
        # dir_name = '/'.join(subject.split('/')[1:])

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

            # bbox
            frame_name = img['file_name']
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'], dtype=np.float32) # x,y,w,h
            if data_type != 'infer':
                if bbox[0]<10 or bbox[1]<10 or max(bbox[2], bbox[3])<80 or bbox[0]+bbox[2]>img_width-10 or bbox[1]+bbox[3]>img_height-10:
                    # print(f'{i}, Discard {image_name}, bbox is too biased/small: {bbox.tolist()}')
                    continue

                # frame
                img_path = os.path.join(self.image_dir, f'{phase}/{image_name}')
                img = cv2.imread(img_path)
                if img.max() < 20:
                    # print(f'{i}, Discard {image_name}, RGB is too dark: {img.max()}')
                    continue
                if np.allclose(img[..., 0], img[..., 1], atol=1) or np.allclose(img[..., 2], img[..., 1], atol=1) or np.allclose(img[..., 0], img[..., 2], atol=1):
                    # print(f'{i}, Discard {image_name}, Gray scale')
                    continue

                # mask
                mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                if not os.path.exists(mask_path):
                    # print(f'{i}, Discard {image_name}, w/o mask')
                    continue
                mask = cv2.imread(img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png'))
                mask_sum = mask[..., 0].astype('bool').sum()
                if mask.max() < 255 or mask_sum < 3000:
                    # print(f'{i}, Discard {image_name}, mask is too dark: {mask.max()}, {mask_sum}')
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

            # camera
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), \
                            np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            cam_T, cam_R = campos, camrot
            E = np.eye(4)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            K = np.eye(3)
            K[[0, 1], [0, 1]] = focal
            K[[0, 1], [2, 2]] = princpt
            self.cameras[f'{phase}/{image_name}'] = {
                'intrinsics': K,
                'extrinsics': [cam_R, cam_T],  # E,
                'distortions': np.zeros(5)
            }

            # mesh
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
            'dst_tpose_joints': \
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
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
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

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0],
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
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

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]

        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
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
                    # print('[WARNING] No SAM mask, use MANO mask instead')
                    maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                    alpha_mask = np.array(load_image(maskpath))

        else:
            alpha_mask = np.ones_like(orig_img) * 255

        # undistort image
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

        select_inds, patch_info, patch_div_indices = \
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
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)
        target_alpha_patches = np.stack(targets_alpha, axis=0)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, ray_alpha, near, far, \
                target_patches, target_alpha_patches, patch_masks, patch_div_indices


    def __len__(self):
        return self.get_total_frames()
        # return self.epoch_size



    def __getitem__(self, idx):

        # 将 idx 映射到 subject 索引
        n_sub = len(self.subject_ids)
        # 如果你希望 dataset 长度大于 subject 数，可在 __len__ 返回期望的 epoch_size（见下文）
        subject_idx = idx % n_sub
        # print(subject_idx)
        subject_prefix = self.subject_ids[subject_idx]
        subject_frames = self.frames_by_subject[subject_prefix]
        chosen = random.sample(subject_frames, self.num_frames)
        id = subject_prefix

        # aug_prob = random.random()
        # #######################################################################################################################
        # frame_name = self.framelist[idx]
        # parts = frame_name.split('/')
        # id = int(frame_name.split('/')[1][7:])

        # # ---------- 3. 找出同 subject 下的所有 frame ----------
        # subject_prefix = parts[0] + '/' + parts[1]   # e.g., "train/Capture0"
        # # subject_dir = os.path.join(self.image_dir, subject_prefix)
        # same_subject_frames = [f for f in self.framelist if f.startswith(subject_prefix)]
        # chosen = random.sample(same_subject_frames, self.num_frames)
        # #######################################################################################################################
        # ---------- 5. 对每个 chosen frame 重用你原有的处理流程 ----------

        cam_info_list = []

        for frame_name in chosen:

            img_path = os.path.join(self.image_dir, frame_name)
            mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')

            bgcolor = np.array(self.bgcolor, dtype='float32')

            img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])
            # from tools_utils import libcore
            # libcore.write_tensor_image('./hand/interhand_gt_debug.jpg', torch.tensor(img))
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
            # dst_bbox = dst_skel_info['bbox']
            dst_poses = dst_skel_info['poses']
            dst_shape = dst_skel_info['shape']
            # dst_tpose_joints = dst_skel_info['dst_tpose_joints']
            # dst_cam_joints = dst_skel_info['joint_cam']
            # dst_valid_joints = dst_skel_info['joint_valid']

            # dst_poses = np.zeros_like(dst_poses)

            # # pose = np.zeros(48)

            # # 举例：让模型绕 X 轴转 90°
            # dst_poses[0:3] = np.array([-np.pi/2, 0, 0])


            assert frame_name in self.cameras
            K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
            K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
            K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])
            # K

            focal_length_x = K[0, 0]
            focal_length_y = K[1, 1]
            FovX = focal2fov(focal_length_x, self.height)
            FovY = focal2fov(focal_length_y, self.width)

            E = self.cameras[frame_name]['extrinsics']
            # R = E[0]

            E = apply_global_tfm_to_camera(
                    E=np.eye(4),
                    Rh=dst_skel_info['Rh'],
                    Th=dst_skel_info['Th'])
            R = E[:3, :3]
            T = E[:3, 3]
            # T[:1] = 0.0

            # T = E[1].reshape(3, 1)
            # T = np.zeros_like(T)
            # w2c = np.eye(4)
            # w2c[:3, :3] = R
            # w2c[:3, 3:4] = T
            # R = np.transpose(w2c[:3, :3])

            # dst_Th = dst_skel_info['Th']
            posed_res = self.mano(
                torch.from_numpy(dst_shape)[None].float(),
                torch.from_numpy(dst_poses[:3])[None].float(),
                torch.from_numpy(dst_poses[3:])[None].float(),
                # transl=torch.from_numpy(dst_Th)[None].float(),
            return_verts=True)

            # posed_res_ori = self.mano_ori(
            #     torch.from_numpy(dst_shape)[None].float(),
            #     torch.from_numpy(dst_poses[:3])[None].float(),
            #     torch.from_numpy(dst_poses[3:])[None].float(),
            #     # transl=torch.from_numpy(dst_Th)[None].float(),
            # return_verts=True)

            cano_res = self.mano(
                torch.from_numpy(dst_shape)[None].float(),
                torch.zeros(1, 3).float(),
                torch.zeros(1, 45).float(),
                # transl=torch.from_numpy(dst_Th)[None].float(),
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
            # posed_face_area = calc_face_areas(posed_res.vertices.squeeze(), posed_res.faces_tensor).numpy()
            # 相对变化（变化比例）
            # area_ratio = posed_face_area / (self.cano_face_area + 1e-8)  # 大于1放大，小于1表示缩小
            # torch.sum(area_ratio>1.5)   # 1043个

            # T_ = T.copy()
            # T_[2] /= 10.0
            # xyz_vertex = ndc_T_world(world_vertex.reshape(-1, 3), K, E, self.height, self.width)
            # cam_view = self.renderer(xyz_vertex.reshape(-1, 3), T_, img)
            # input_img_overlay = img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            # 得到每个面的语义类别，以及每个顶点的 one-hot 语义标签
            semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                         faces=torch.from_numpy(self.mano.faces).long())
            nail_mask = (semantic_mask).astype('float32') / 255.
            nail_mask = nail_mask[..., None]#.repeat(3, axis=2)
            # nail_img = nail_mask * img + (1.0 - nail_mask / 255.) * bgcolor[None, None, :]

            # # 计算顶点的图像坐标，用于后续采样
            verts_cam = np.dot(R, posed_vertices.T).T + T[None, :]
            verts_img = np.dot(K, verts_cam.T).T
            verts_img[:, :2] /= verts_img[:, 2:3]
            nail_img = (verts_img[:, :2]).astype(np.float32)   # 12337, 2
            # # nail_img = None
            # out = visualize_three_modes(verts_cam, nail_img, self.pcd_scales, K, img=img, H=256, W=256,
            #                             z_eps=1e-3, show_top_k=20, save_prefix='./hand/vis_compare')
            # depth_buffer, vis_mask = compute_visibility_from_depth(torch.tensor(verts_cam), torch.tensor(nail_img),
            #                                                        self.height, self.width, eps=0)

            # print(f"可见点数量: {vis_mask.sum().item()} / {len(vis_mask)}")

            # import matplotlib.pyplot as plt

            # visible_xy = nail_img[vis_mask].cpu().numpy()
            # plt.scatter(visible_xy[:,0], visible_xy[:,1], s=2, c='lime', label='Visible')
            # plt.gca().invert_yaxis()
            # plt.legend()
            # plt.title('Visible points')
            # plt.show()
            # plt.savefig("./hand/occlu.png", dpi=300, bbox_inches='tight')

            # a=compute_visibility_with_radius_discrete(torch.tensor(verts_cam), torch.tensor(nail_img), 256, 256, self.pcd_scales)
            # m=a[1]
            # mesh1 = trimesh.Trimesh(world_vertex.reshape(12337,3)[m])
            # mesh1.export('./hand/depth-4.ply')




            # depth_map = self.renderer(torch.from_numpy(world_vertex).cuda(), torch.from_numpy(K), torch.from_numpy(E), self.mano.faces)
            # x = verts_img[:, 0]
            # y = verts_img[:, 1]
            # z = verts_cam[:, 2].reshape(-1)   # 👈 这一行保证 z 是 (N,)

            # # depth_map 处理
            # depth_map = np.asarray(depth_map.cpu())
            # if depth_map.ndim == 3:
            #     depth_map = depth_map.squeeze()
            # H, W = depth_map.shape
            # depth_map[np.isnan(depth_map)] = np.inf

            # # 计算在图像范围内的点
            # mask_inside = (
            #     (x >= 0) & (x < W) &
            #     (y >= 0) & (y < H) &
            #     (z > 0)
            # )

            # # 有效像素索引
            # x_valid = np.clip(x[mask_inside].astype(np.int32), 0, W - 1)
            # y_valid = np.clip(y[mask_inside].astype(np.int32), 0, H - 1)

            # # 深度采样
            # z_buf = depth_map[y_valid, x_valid].reshape(-1)
            # z_in = z[mask_inside].reshape(-1)

            # # 深度比较
            # visible_local = z_in <= (z_buf + 5e-2)

            # # 构造 mask
            # visible_mask = np.zeros(world_vertex.shape[1], dtype=bool)
            # visible_mask[mask_inside] = visible_local

            # print(f"可见点数量: {visible_mask.sum()} / {len(visible_mask)}")

            # # if aug_prob > 0.5:
            # #     img=self.transform_c(image=img, mask=alpha)['image']

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
                        # uv_mapping=uv_mapping,
                        world_vertex=world_vertex,
                        # world_vertex=posed_face_area,  # world_vertex
                        # posed_face_area=posed_face_area,
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
                # Generate and save UV mask (marking foreground vs background pixels)
                # cam_info.uv_mask = generate_uv_mask(uv_mapping, uv_resolution=256)
                # debug_path = f'./uv_debug/{frame_name.replace("/", "_")}.png'
                # os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                # visualize_uv_mapping(uv_mapping, save_path=debug_path, title=f'UV debug {frame_name}')
            cam_info_list.append(cam_info)

        # Convert CameraInfo objects to dicts for merge_batch
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)

        img_aug = skin_tone_perturb_shared(final_results['original_image'])
        final_results['original_image'] = img_aug

        return final_results
        # return results


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

        # MANO
        self.mano = smplx.create(**cfg.smpl_cfg)
        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1


        # MANO-HD ===================================
        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights
        # MANO-HD ===================================
        self.renderer = Renderer_mesh()


        # 加载每个顶点的语义标签
        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        print(f'Loading MANO semantic labels from {labels_path}')
        ply = PlyData.read(labels_path)
        if ply.elements:
            pc = pd.DataFrame(ply.elements[0].data).values
        self.labels = torch.tensor(pc[:, 6].astype(np.uint8))  # (N,)

        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        # 找出所有指甲顶点
        nail_indices = np.isin(self.labels, list(NAIL_PARTS.values()))
        # self.nail_coords = pc_coords[nail_indices]
        # self.objects_dc = F.one_hot((self.labels-1).to(torch.int64), num_classes=16).unsqueeze(1).to(torch.float32).cuda()
        # 找到哪些点是指甲
        self.nail_labels = self.labels[nail_indices]

        nail_vertex_indices = np.nonzero(nail_indices)[0]
        faces_np = self.mano.faces.cpu().numpy() if torch.is_tensor(self.mano.faces) else self.mano.faces
        face_nail_mask = np.all(np.isin(faces_np, nail_vertex_indices), axis=1)
        self.nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))
        self.nail_mask_1 = self.nail_mask.to(torch.float32).unsqueeze(1)

        self.nail_faces = faces_np[face_nail_mask]


        # annotation
        self.phase = kwargs.get('data_type', 'train')
        # self.phase = data_type
        # subject = 'test/Capture0/ROM03_RT_No_Occlusion'
        if subject is None:
            if self.phase == 'train':
                subject = cfg_handavatar.subject
            else:
                subject = cfg_handavatar[kwargs['data_type']].subject
        subject = 'test/Capture0/ROM03_RT_No_Occlusion'
        # subject = 'test/Capture0/ROM04_RT_Occlusion'
        subject_1 = 'train/prior_learning_data'
        if isinstance(subject, list):
            subject = subject[0]
        self.image_dir = os.path.join(dataset_path, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')
        # anno_name = os.path.join(self.image_dir.replace('images', 'preprocess_handavatar'), subject, 'anno_cam.pkl')
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

        # canonical
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
        # big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        # big_pose_smpl_param['faces'] = self.faces.detach()  # 1538,3
        self.big_pose_smpl_param = big_pose_smpl_param
        self.cano_face_area = calc_face_areas(torch.tensor(self.canonical_verts), mano_res.faces_tensor)


        # post process
        self.framelist = self.framelist[::skip]
        # exclude = cfg[kwargs['data_type']].get('exclude_idx', None)
        # if isinstance(exclude, list):
        #     sel_idx = list(range(len(self.framelist)))
        #     self.framelist = [self.framelist[i] for i in sel_idx if i not in exclude]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        print(f' -- Total Frames: {self.get_total_frames()}')
        # self.keyfilter = keyfilter
        self.bgcolor = bgcolor
        self.num_frames = 5 # 4
        # self.ray_shoot_mode = ray_shoot_mode

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(self.canonical_verts)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]#.repeat(1, 3)
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

        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        # img = orig_img
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

        # undistort image
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
            'dst_tpose_joints': \
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
            'dst_tpose_joints': \
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
        # example_data/interhand2.6m/images/test_Capture0_ROM03_RT_No_Occlusion_cam400266_image14388.jpg
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400266/image14388.jpg'
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400447/image14472.jpg'
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400279/image13848.jpg'
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400500/image13170.jpg'
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400481/image15300.jpg'
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400451/image18214.jpg'
        bgcolor = np.array(self.bgcolor, dtype='float32')
        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])
        # from tools_utils import libcore
        # libcore.write_tensor_image('./hand/interhand_gt_debug.jpg', torch.tensor(img))
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
        # a = alpha.astype('float32')
        # cv2.imwrite('./hand/fig1-example-1.png', a*255)

        # # 保存mask（兼容 0–1 和 0–255，以及单/三通道）
        # alpha_to_save = alpha
        # if alpha_to_save.dtype != np.uint8:
        #     if np.max(alpha_to_save) <= 1.0 + 1e-6:
        #         alpha_to_save = alpha_to_save * 255.0
        #     alpha_to_save = np.clip(alpha_to_save, 0, 255).astype('uint8')
        # if alpha_to_save.ndim == 3:
        #     # 若为三通道，取最大值聚合为单通道以便可视化
        #     alpha_to_save = np.max(alpha_to_save, axis=2)
        # cv2.imwrite(mask_path, alpha_to_save)

        # flipped_img = torch.flip(torch.tensor(img), dims=[1])  # dim=2 表示水平翻转 (W维)
        # from tools_utils import libcore
        # libcore.write_tensor_image('./hand/1-debug-left-flip.jpg', flipped_img, rgb2bgr=True)


        dst_skel_info = self.query_dst_skeleton(frame_name)
        # dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']
        # dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        # dst_cam_joints = dst_skel_info['joint_cam']
        # dst_valid_joints = dst_skel_info['joint_valid']

        cam_info_list = []
        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])
        # K

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

        # 计算顶点的图像坐标，用于后续采样
        verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None]#.repeat(3, axis=2)
        # out = visualize_three_modes(verts_cam, nail_img, self.pcd_scales, K, img=img, H=256, W=256,
        #                                 z_eps=1e-3, show_top_k=20, save_prefix='./hand/vis_compare')

        depth_map = self.renderer(torch.from_numpy(world_vertex).cuda(), torch.from_numpy(K), torch.from_numpy(E), self.mano.faces)
        x = verts_img[:, 0]
        y = verts_img[:, 1]
        z = verts_cam[:, 2].reshape(-1)   # 👈 这一行保证 z 是 (N,)

        # depth_map 处理
        depth_map = np.asarray(depth_map.cpu())
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze()
        H, W = depth_map.shape
        depth_map[np.isnan(depth_map)] = np.inf

        # 计算在图像范围内的点
        mask_inside = (
            (x >= 0) & (x < W) &
            (y >= 0) & (y < H) &
            (z > 0)
        )

        # 有效像素索引
        x_valid = np.clip(x[mask_inside].astype(np.int32), 0, W - 1)
        y_valid = np.clip(y[mask_inside].astype(np.int32), 0, H - 1)

        # 深度采样
        z_buf = depth_map[y_valid, x_valid].reshape(-1)
        z_in = z[mask_inside].reshape(-1)

        # 深度比较
        visible_local = z_in <= (z_buf + 5e-2)

        # 构造 mask
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
        # final_results = cam_info_dicts

        return final_results



    def get_img_04(self, img_path='./data/image15012.jpg', mask_path='./data/image15012.png'):
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400272/image15012.jpg'
        # frame_name = 'test/Capture0/ROM04_RT_Occlusion/cam400451/image18214.jpg'
        # frame_name = 'test/Capture0/ROM04_RT_Occlusion/cam400262/image17806.jpg'
        frame_name = 'train/Capture8/0007_thumbup_normal/cam400002/image2402.jpg'
        # frame_name = 'train/Capture9/0004_star_trek/cam400007/image1485.jpg'
        # frame_name = 'train/Capture9/0000_neutral_relaxed/cam400007/image0195.jpg'
        # frame_name = 'train/Capture9/0005_star_trek_extended_thumb/cam400029/image1908.jpg'
        # frame_name = 'test/Capture0/ROM04_RT_Occlusion/cam400460/image17788.jpg'
        bgcolor = np.array(self.bgcolor, dtype='float32')
        img, alpha = self.load_image_09(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])
        # from tools_utils import libcore
        # libcore.write_tensor_image('./hand/interhand_gt_debug.jpg', torch.tensor(img))
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

        # flipped_img = torch.flip(torch.tensor(img), dims=[1])  # dim=2 表示水平翻转 (W维)
        # from tools_utils import libcore
        # libcore.write_tensor_image('./hand/1-debug-left-flip.jpg', flipped_img, rgb2bgr=True)


        dst_skel_info = self.query_dst_skeleton_04(frame_name)
        # dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']
        # dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        # dst_cam_joints = dst_skel_info['joint_cam']
        # dst_valid_joints = dst_skel_info['joint_valid']

        cam_info_list = []
        assert frame_name in self.cameras_04
        K = self.cameras_04[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])
        # K

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

        # 计算顶点的图像坐标，用于后续采样
        verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None]#.repeat(3, axis=2)
        # out = visualize_three_modes(verts_cam, nail_img, self.pcd_scales, K, img=img, H=256, W=256,
        #                                 z_eps=1e-3, show_top_k=20, save_prefix='./hand/vis_compare')

        depth_map = self.renderer(torch.from_numpy(world_vertex).cuda(), torch.from_numpy(K), torch.from_numpy(E), self.mano.faces)
        x = verts_img[:, 0]
        y = verts_img[:, 1]
        z = verts_cam[:, 2].reshape(-1)   # 👈 这一行保证 z 是 (N,)

        # depth_map 处理
        depth_map = np.asarray(depth_map.cpu())
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze()
        H, W = depth_map.shape
        depth_map[np.isnan(depth_map)] = np.inf

        # 计算在图像范围内的点
        mask_inside = (
            (x >= 0) & (x < W) &
            (y >= 0) & (y < H) &
            (z > 0)
        )

        # 有效像素索引
        x_valid = np.clip(x[mask_inside].astype(np.int32), 0, W - 1)
        y_valid = np.clip(y[mask_inside].astype(np.int32), 0, H - 1)

        # 深度采样
        z_buf = depth_map[y_valid, x_valid].reshape(-1)
        z_in = z[mask_inside].reshape(-1)

        # 深度比较
        visible_local = z_in <= (z_buf + 5e-2)

        # 构造 mask
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
        # final_results = cam_info_dicts

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
        # h, w = image_size
        masks = np.zeros((self.height, self.width), dtype=np.uint8)

        # 相机变换
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        verts_img = np.round(verts_img[:, :2]).astype(np.int32)   # 12337, 2

        # # 计算每个面所属的语义类别（取面顶点标签的众数）
        # face_labels = self.labels[faces].numpy()
        # per_face_class = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, face_labels)

        # # 绘制mask
        # for f_idx, f in enumerate(faces):
        #     cls = int(per_face_class[f_idx])
        #     tri = verts_img[f, :]
        #     if np.any(tri < 0) or np.any(tri[:, 0] >= self.width) or np.any(tri[:, 1] >= self.height):
        #         continue
        #     cv2.fillConvexPoly(masks[cls], tri, 255)


         # 绘制指甲区域
        for tri in self.nail_faces:
            tri_pts = verts_img[tri, :]
            if np.any(tri_pts < 0) or np.any(tri_pts[:, 0] >= self.width) or np.any(tri_pts[:, 1] >= self.height):
                continue
            cv2.fillConvexPoly(masks, tri_pts, 255)

        return masks


    def __getitem__(self, idx):

        if self.finetune:
            # return self.get_img_04()
            return self.get_img()

        frame_name = self.framelist[idx]
        # frame_name = 'test/Capture0/ROM03_RT_No_Occlusion/cam400272/image15012.jpg'

        img_path = os.path.join(self.image_dir, frame_name)
        mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')

        bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])
        # from tools_utils import libcore
        # libcore.write_tensor_image('./hand/interhand_gt_debug.jpg', torch.tensor(img))
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
        # dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']

        # dst_poses = np.zeros_like(dst_poses)
        # # pose = np.zeros(48)

        # # 举例：让模型绕 X 轴转 90°
        # dst_poses[0:3] = np.array([-np.pi/2, 0, 0])

        # dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        # dst_cam_joints = dst_skel_info['joint_cam']
        # dst_valid_joints = dst_skel_info['joint_valid']


        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])
        # K

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

        # 计算顶点的图像坐标，用于后续采样
        verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None]#.repeat(3, axis=2)


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



class HanCo(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(
            self,
            dataset_path='/home/z/zh174/hanco/',
            keyfilter=None,
            maxframes=-1,
            bgcolor=[0.0, 0.0, 0.0],
            ray_shoot_mode='image',
            data_type='val',
            skip=200,
            subject=None,
            test_vid=None,
            test_cam=None,
            img_id=None,
            test_cams=None,
            **kwargs):

        print('[Dataset Path]', dataset_path)
        self.dataset_path = dataset_path

        # if cfg.smpl_cfg['flat_hand_mean'] == True:
        #     cfg.smpl_cfg['flat_hand_mean'] = False
        #     # cfg.smpl_cfg['scale'] = None
        #     # cfg.smpl_cfg['center_id'] = 4
        #     # cfg.smpl_cfg['center_id'] = None

        self.mano = smplx.create(**cfg.smpl_cfg)
        self.mano_ori = smplx_.create(**cfg.smpl_cfg)

        # MANO-HD ===================================
        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights
        # MANO-HD ===================================


        self.hand_mean = ([ 0.11167872, -0.04289217,  0.41644184,  0.10881133,  0.06598568,
        0.75622001, -0.09639297,  0.09091566,  0.18845929, -0.11809504,
       -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.70485714,
       -0.01918292,  0.09233685,  0.33791352, -0.45703298,  0.19628395,
        0.62545753, -0.21465238,  0.06599829,  0.50689421, -0.36972436,
        0.06034463,  0.07949023, -0.14186969,  0.08585263,  0.63552826,
       -0.30334159,  0.05788098,  0.63138921, -0.17612089,  0.13209308,
        0.37335458,  0.85096428, -0.27692274,  0.09154807, -0.49983944,
       -0.02655647, -0.05288088,  0.53555915, -0.04596104,  0.27735802])

        self.renderer = Renderer_mesh()
        self.height, self.width = 256, 256
        # self.height, self.width = 224, 224
        self.img_scale = self.height / 224.0


        self.image_dir = os.path.join(dataset_path, 'rgb')
        self.mask_dir = os.path.join(dataset_path, 'mask_hand')

        self.exclude_subjects = set()

        # image 75
        self.test_subj = 10
        self.test_cam = 6
        self.test_vid = 191
        self.img_id = 75
        self.test_cams = [3,5,6,7]
        self.skip = 10


        # image 24
        self.test_subj = 10
        self.test_cam = 0
        self.test_vid = 168
        self.img_id = 24
        self.test_cams = [0,1,3,4,6,7]
        self.skip = 1

        # image 24
        self.test_subj = 10
        self.test_cam = 0
        self.test_vid = 165
        self.img_id = 32
        self.test_cams = [0]
        self.skip = 1

        # image 24
        self.test_subj = 10
        self.test_cam = 7
        self.test_vid = 192
        self.img_id = 57
        self.test_cams = [0]
        self.skip = 1


        # image 24
        self.test_subj = 10
        self.test_cam = 6
        self.test_vid = 185
        self.img_id = 10
        self.test_cams = [0]
        self.skip = 1

        # --- CLI overrides (take precedence over hardcoded defaults above) ---
        if test_vid is not None:
            self.test_vid = int(test_vid)
        if test_cam is not None:
            self.test_cam = int(test_cam)
        if img_id is not None:
            self.img_id = int(img_id)
        if test_cams is not None:
            self.test_cams = [int(c) for c in test_cams]

        # # image 83
        # self.test_subj = 10
        # self.test_cam = 2
        # self.test_vid = 305
        # self.img_id = 83
        # self.test_cams = [2,3,4]
        # # self.test_cams = [2]
        # self.skip = 5

        # # image 6
        # self.test_subj = 10
        # self.test_cam = 7
        # self.test_vid = 154
        # self.img_id = 6
        # self.test_cams = [0,3,5,6,7]
        # self.skip = 1


        # # image 8
        # self.test_subj = 10
        # self.test_cam = 7
        # self.test_vid = 166
        # self.img_id = 8
        # self.test_cams = [0,3,4,7]
        # self.skip = 5




        if data_type == 'train':
            self.exclude_subjects.add(self.test_subj)   # subject 10 不进训练

        self.phase = data_type
        # 你要保留的“每个 subject 的视频数量”
        max_sids_per_subject = 10   # <-- 改成你要的 xx

        with open(os.path.join(dataset_path, "meta.json"), "r") as f:
            meta = json.load(f)

        subject_id_meta = meta["subject_id"]   # [1518][num_frames]
        num_seq = len(subject_id_meta)

        # subj -> sid -> list[fid]
        subj_to_sid_fids = defaultdict(lambda: defaultdict(list))

        for sid in range(num_seq):
            for fid, subj in enumerate(subject_id_meta[sid]):
                if subj is None:
                    continue
                subj = int(subj)
                subj_to_sid_fids[subj][sid].append(fid)

        self.frames_by_subject = defaultdict(list)

        for subj, sid_map in subj_to_sid_fids.items():
            # ===== 核心：训练阶段直接跳过 subject 10 =====
            if subj in self.exclude_subjects:
                continue
            kept_sids = sorted(sid_map.keys())[:max_sids_per_subject]

            for sid in kept_sids:
                valid_pairs = []

                for fid in sid_map[sid]:
                    has_valid = False
                    for cid in HANCO_CAMS:
                        img_path = os.path.join(
                            self.dataset_path,
                            f"rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg"
                        )
                        mask_path = img_path.replace("rgb", "mask_hand")
                        mano_path = os.path.join(
                            self.dataset_path,
                            f"shape/{sid:04d}/cam{cid}/{fid:08d}.json"
                        )
                        calib_path = os.path.join(
                            self.dataset_path,
                            f"calib/{sid:04d}/{fid:08d}.json"
                        )

                        if (os.path.exists(img_path)
                            and os.path.exists(mask_path)
                            and os.path.exists(mano_path)
                            and os.path.exists(calib_path)):
                            has_valid = True
                            break

                    if has_valid:
                        valid_pairs.append((sid, fid))

                # 只把“至少有一个合法 cam 的帧”加入
                self.frames_by_subject[subj].extend(valid_pairs)

        self.subject_ids = sorted(self.frames_by_subject.keys())

#################################################################################################################

        if data_type != 'train':
            # 固定 subject 011, sid 0191, cam6
            self.progress_frames = []

            sid = self.test_vid  # 191

            # cams = [6]
            # cams = [2,3,4]
            # cams = [3,5,6,7]
            # cams = [0, 3, 5, 6, 7]
            cams = [0]

            for cam in self.test_cams:
                img_dir = os.path.join(self.dataset_path, f"rgb/{sid:04d}/cam{cam}")
                if not os.path.isdir(img_dir):
                    continue  # 防御式：某些 cam 目录不存在

                all_imgs = sorted(os.listdir(img_dir))


                # img_dir = os.path.join(self.dataset_path, f"rgb/{sid:04d}/cam{cam}")
                # all_imgs = sorted(os.listdir(img_dir))

                for fname in all_imgs:
                    if fname == "00000075.jpg":
                        continue
                    fid = int(fname.split('.')[0])
                    self.progress_frames.append((sid, cam, fid))
            self.progress_frames = self.progress_frames[::self.skip]

#################################################################################################################

        if self.phase == 'train':
            self.num_frames = 5 # 4
        else:
            self.num_frames = 1

        # canonical
        mano_res = self.mano(
            torch.zeros(1, 10).float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
            return_verts=True)
        # self.canonical_bbox = mano_res.joints[0].numpy()
        self.canonical_verts = mano_res.vertices[0].numpy()
        # self.canonical_bbox = self.skeleton_to_bbox(self.canonical_verts)
        big_pose_min_xyz = np.min(self.canonical_verts, axis=0)
        big_pose_max_xyz = np.max(self.canonical_verts, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        self.big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(self.canonical_verts)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]#.repeat(1, 3)
        self.pcd_scales = torch.exp(scales)

        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)
        self.big_pose_smpl_param = big_pose_smpl_param


        # 加载每个顶点的语义标签
        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        print(f'Loading MANO semantic labels from {labels_path}')
        ply = PlyData.read(labels_path)
        if ply.elements:
            pc = pd.DataFrame(ply.elements[0].data).values
        self.labels = torch.tensor(pc[:, 6].astype(np.uint8))  # (N,)

        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        # 找出所有指甲顶点
        nail_indices = np.isin(self.labels, list(NAIL_PARTS.values()))
        # self.nail_coords = pc_coords[nail_indices]
        # self.objects_dc = F.one_hot((self.labels-1).to(torch.int64), num_classes=16).unsqueeze(1).to(torch.float32).cuda()
        # 找到哪些点是指甲
        self.nail_labels = self.labels[nail_indices]

        nail_vertex_indices = np.nonzero(nail_indices)[0]
        faces_np = self.mano.faces.cpu().numpy() if torch.is_tensor(self.mano.faces) else self.mano.faces
        face_nail_mask = np.all(np.isin(faces_np, nail_vertex_indices), axis=1)
        self.nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))
        self.nail_mask_1 = self.nail_mask.to(torch.float32).unsqueeze(1)

        self.nail_faces = faces_np[face_nail_mask]

        self.bgcolor = bgcolor


    #     return semantic_masks  # (6, H, W)


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
        # h, w = image_size
        masks = np.zeros((self.height, self.width), dtype=np.uint8)

        # 相机变换
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        verts_img = np.round(verts_img[:, :2]).astype(np.int32)   # 12337, 2

        # # 计算每个面所属的语义类别（取面顶点标签的众数）
        # face_labels = self.labels[faces].numpy()
        # per_face_class = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, face_labels)

        # # 绘制mask
        # for f_idx, f in enumerate(faces):
        #     cls = int(per_face_class[f_idx])
        #     tri = verts_img[f, :]
        #     if np.any(tri < 0) or np.any(tri[:, 0] >= self.width) or np.any(tri[:, 1] >= self.height):
        #         continue
        #     cv2.fillConvexPoly(masks[cls], tri, 255)


         # 绘制指甲区域
        for tri in self.nail_faces:
            tri_pts = verts_img[tri, :]
            if np.any(tri_pts < 0) or np.any(tri_pts[:, 0] >= self.width) or np.any(tri_pts[:, 1] >= self.height):
                continue
            cv2.fillConvexPoly(masks, tri_pts, 255)

        return masks


    # def get_img(self, img_idx=75):
    def get_img(self, img_idx=83):
    # def get_img(self, img_idx=24):

        img_idx = self.img_id
        cam_info_list = []
        frame_name = f"{self.test_vid:04d}/cam{self.test_cam}/{img_idx:08d}.jpg"
        bgcolor = np.array(self.bgcolor, dtype='float32')
        img_path = os.path.join(
                            self.dataset_path,
                            f"rgb/{self.test_vid:04d}/cam{self.test_cam}/{img_idx:08d}.jpg"
                        )
        mask_path = img_path.replace("rgb", "mask_hand") #.replace(".jpg", ".png")
        mano_path = img_path.replace("rgb", "shape").replace(".jpg", ".json")
        calib_path = os.path.join(self.dataset_path, f"calib/{self.test_vid:04d}/{img_idx:08d}.json")

        img = cv2.imread(img_path)[:, :, ::-1].astype('float32') / 255.0
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Read mask, create default white mask if file doesn't exist
        alpha_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if alpha_raw is None:
            # print(f"Warning: Mask not found at {mask_path}, using default white mask")
            alpha = np.ones((self.height, self.width, 3), dtype='float32')
        else:
            alpha = alpha_raw[..., None].repeat(3, axis=-1).astype('float32') / 255.0
            alpha = cv2.resize(alpha, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        img = alpha * img + (1.0 - alpha) * bgcolor[None, None, :]

        # ===== 相机 =====
        calib = json.load(open(calib_path, "r"))
        K = np.array(calib["K"][self.test_cam])
        K_ = K.copy()
        K[0,0] *= self.img_scale
        K[1,1] *= self.img_scale
        K[0,2] *= self.img_scale
        K[1,2] *= self.img_scale

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovX = focal2fov(focal_length_x, self.height)
        FovY = focal2fov(focal_length_y, self.width)

        M = np.eye(4)
        R = M[:3, :3]

        with open(mano_path, 'r') as fi:
            mano_cam = np.array(json.load(fi))[None]

        pose_cam, shape_cam, global_t_cam = pred_to_mano(mano_cam, K_[None], fw=np)

        T = global_t_cam.reshape(-1)
        T *= 10   # z 轴取反

        root_pose = pose_cam[0, :3]      # (3,)
        hand_pose_F = pose_cam[0, 3:]    # (45,)

        hand_pose_T = hand_pose_F + self.hand_mean

        pose_cam[0, :3] = root_pose
        pose_cam[0, 3:] = hand_pose_T

        dst_poses = pose_cam.reshape(-1)
        dst_shape = shape_cam.reshape(-1)


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

        # 计算顶点的图像坐标，用于后续采样
        verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None]#.repeat(3, axis=2)


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
        # final_results = cam_info_dicts

        return final_results



    def __len__(self):
        if self.phase != 'train':
            return len(self.progress_frames)
        else:
            return len(self.frames_by_subject)


    def __getitem__(self, idx, attempt=0, max_attempts=10):

        if self.phase == 'train':
            subject_idx = idx % len(self.subject_ids)
            subject_id = self.subject_ids[subject_idx]

            frame_pairs = self.frames_by_subject[subject_id]

            chosen = random.sample(frame_pairs, self.num_frames)
            cid = random.choice(HANCO_CAMS)
        else:
            # final_results = self.get_img()
            # return final_results
            sid, cid, fid = self.progress_frames[idx]
            subject_id = self.test_subj
            chosen = [(sid, fid)]

        cam_info_list = []
        bgcolor = np.array(self.bgcolor, dtype='float32')

        for (sid, fid) in chosen:

            img_path = os.path.join(self.dataset_path,
                f"rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg")

            mask_path = img_path.replace('rgb', 'mask_hand') #.replace('.jpg', '.png')

            mano_path = os.path.join(self.dataset_path,
                f"shape/{sid:04d}/cam{cid}/{fid:08d}.json")

            calib_path = os.path.join(self.dataset_path,
                f"calib/{sid:04d}/{fid:08d}.json")

            frame_name = f"{sid:04d}/cam{cid}/{fid:08d}.jpg"

            img = cv2.imread(img_path)[:, :, ::-1].astype('float32') / 255.0
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            # Read mask, create default white mask if file doesn't exist
            alpha_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if alpha_raw is None:
                # print(f"Warning: Mask not found at {mask_path}, using default white mask")
                alpha = np.ones((self.height, self.width, 3), dtype='float32')
            else:
                alpha = alpha_raw[..., None].repeat(3, axis=-1).astype('float32') / 255.0
                alpha = cv2.resize(alpha, (self.width, self.height), interpolation=cv2.INTER_NEAREST)


            img = alpha * img + (1.0 - alpha) * bgcolor[None, None, :]

            # ===== 相机 =====
            calib = json.load(open(calib_path, "r"))
            K = np.array(calib["K"][cid])
            K_ = K.copy()
            K[0,0] *= self.img_scale
            K[1,1] *= self.img_scale
            K[0,2] *= self.img_scale
            K[1,2] *= self.img_scale

            M = np.array(calib["M"][cid])
            M = np.eye(4)
            # E = M
            R = M[:3, :3]
            # T = M[:3, 3]

            focal_length_x = K[0, 0]
            focal_length_y = K[1, 1]
            FovX = focal2fov(focal_length_x, self.height)
            FovY = focal2fov(focal_length_y, self.width)

            with open(mano_path, 'r') as fi:
                mano_cam = np.array(json.load(fi))[None]

            pose_cam, shape_cam, global_t_cam = pred_to_mano(mano_cam, K_[None], fw=np)

            root_pose = pose_cam[0, :3]      # (3,)
            hand_pose_F = pose_cam[0, 3:]    # (45,)

            # pose_mean_hand = self.hand_mean    # 取 45 维；不同实现字段名可能略有差异

            hand_pose_T = hand_pose_F + self.hand_mean

            pose_cam[0, :3] = root_pose
            pose_cam[0, 3:] = hand_pose_T

            dst_poses = pose_cam.reshape(-1)
            # dst_shape = np.array(mano_w["shapes"]).reshape(-1)
            dst_shape = shape_cam.reshape(-1)
            # dst_Th = np.array(mano_w["global_t"]).reshape(-1)

            T = global_t_cam.reshape(-1)
            T *= 10   # z 轴取反

            # dst_poses = np.zeros_like(dst_poses)
            # # pose = np.zeros(48)

            # # 举例：让模型绕 X 轴转 90°
            # dst_poses[0:3] = np.array([np.pi/2, 0, 0])


            posed_res = self.mano(
                torch.from_numpy(dst_shape)[None].float(),
                torch.from_numpy(dst_poses[:3])[None].float(),
                torch.from_numpy(dst_poses[3:])[None].float(),
                # transl=torch.from_numpy(dst_Th)[None].float(),
                # transl=torch.from_numpy(global_t_cam).reshape(-1)[None].float(),
            return_verts=True)

            cano_res = self.mano(
                torch.from_numpy(dst_shape)[None].float(),
                torch.zeros(1, 3).float(),
                torch.zeros(1, 45).float(),
                # transl=torch.from_numpy(dst_Th)[None].float(),
            return_verts=True)
            self.canonical_verts = cano_res.vertices[0].detach().numpy()

            world_vertex = posed_res.vertices.detach().numpy()
            # world_vertex_ori = posed_res_ori.vertices.detach().numpy()

            smpl_param ={
                'poses': dst_poses,
                'shape': dst_shape,
                # 'trans': dst_Th,
                'posed_verts': world_vertex.squeeze(),
            }

            min_xyz = np.min(world_vertex.squeeze(), axis=0)
            max_xyz = np.max(world_vertex.squeeze(), axis=0)
            max_xyz -= 0.05
            min_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            # 得到每个面的语义类别，以及每个顶点的 one-hot 语义标签
            semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())

            nail_mask = (semantic_mask).astype('float32') / 255.
            nail_mask = nail_mask[..., None]#.repeat(3, axis=2)
            # nail_img = nail_mask * img + (1.0 - nail_mask / 255.) * bgcolor[None, None, :]

            # # 计算顶点的图像坐标，用于后续采样
            verts_cam = world_vertex.reshape(-1, 3) # [self.nail_mask]
            verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
            verts_img = np.dot(K, verts_cam.T).T
            verts_img[:, :2] /= verts_img[:, 2:3]
            nail_img = (verts_img[:, :2]).astype(np.float32)   # 12337, 2

            # out = visualize_three_modes(verts_cam, nail_img, self.pcd_scales, K, img=img, H=self.height, W=self.width,
            #                         z_eps=1e-3, show_top_k=20, save_prefix='/home/z/zh174/lhm-hand/hand/hanco_vis')

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
                        # world_vertex=posed_face_area,  # world_vertex
                        # posed_face_area=posed_face_area,
                        world_bound=world_bound,
                        big_pose_smpl_param=self.big_pose_smpl_param,
                        big_pose_world_vertex=self.canonical_verts,
                        big_pose_world_bound=self.big_pose_world_bound
                    )

            cam_info = loadCam_aug_bs(subject_id, cam_info)
            cam_info_list.append(cam_info)

        final_results = merge_batch([vars(c) for c in cam_info_list])

        img_aug = skin_tone_perturb_shared(final_results['original_image'])
        final_results['original_image'] = img_aug

        return final_results



class Hands11k_Dataset(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(
            self,
            dataset_path,
            # keyfilter=None,
            maxframes=-1,
            bgcolor=[0.0, 0.0, 0.0],
            # ray_shoot_mode='image',
            skip=1,
            subject=None,
            data_type='train',
            # **kwargs
            ):

        self.split = data_type
        # prefer explicit dataset_path if provided (e.g. '/home/z/zh174/hands11k/processed_test')
        if dataset_path is not None and dataset_path != '':
            self.dat_dir = dataset_path
        else:
            if data_type == 'train':
                self.dat_dir = '/home/z/zh174/Hands11k/processed_test'
            else:
                self.dat_dir = '/home/z/zh174/eval_debug'

        # number of frames sampled per subject (train: multiple, test: single)
        if data_type == 'train':
            print("\033[91m[========== Loading Hands11k Training Dataset ==========]\033[0m")
            self.num_frames = 5 #4
        else:
            print("\033[91m[========== Loading Hands11k Testing Dataset ==========]\033[0m")
            self.num_frames = 1

        # image / mask folders
        images_root = os.path.join(self.dat_dir, 'img')
        masks_root = os.path.join(self.dat_dir, 'mask')

        # collect subject folders (each subject corresponds to a subdirectory)
        if os.path.exists(images_root):
            self.video_ids = [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
            self.video_ids.sort()
        else:
            self.video_ids = []

        # build mapping subject -> list of frame relative paths (subject/filename)
        self.frames_by_subject = {}
        self.subject_ids = []
        self.image_dir = images_root
        self.mask_dir = masks_root
        for vid in self.video_ids:
            subdir = os.path.join(images_root, vid)
            files = [f for f in sorted(os.listdir(subdir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(files) < 4:
                continue

            rel_paths = [os.path.join(vid, f) for f in files]

            self.frames_by_subject[vid] = rel_paths
            self.subject_ids.append(vid)

            # rel_paths = [os.path.join(vid, f) for f in files]
            # if len(rel_paths) >= 4:
            #     self.frames_by_subject[vid] = rel_paths
            #     self.subject_ids.append(vid)

        # flattened framelist
        self.framelist = []
        for vid in self.subject_ids:
            self.framelist.extend(self.frames_by_subject[vid])

        print(f'[Hands11k] Found {len(self.subject_ids)} subjects, {len(self.framelist)} frames for {self.split}')

        # MANO
        # self.mano = smplx.create(**cfg.smpl_cfg)
        self.mano = tools_utils.model.smplx_ours.create(**cfg.smpl_cfg)
        # self.mano_ori = copy.deepcopy(self.mano)  # ✅ 深拷贝
        # self.mano_ori = smplx_.create(**cfg.smpl_cfg)

        # MANO-HD ===================================
        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano_ori(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights
        # MANO-HD ===================================

        # self.renderer = Renderer(faces=self.mano.faces)
        # self.renderer = Renderer_mesh()

        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1

        self._uv_template = self._build_right_hand_uv_template()

        # # annotation
        # self.phase = data_type
        # print(f'[INFO] Phase: {self.phase}')

        # if subject is None:
        #     if self.phase == 'train':
        #         subject = cfg.subject
        #     else:
        #         # subject = cfg[kwargs['data_type']].subject
        #         subject = 'train/prior_learning_data'

        # print(f'[INFO] Subject: {subject}')

        # if isinstance(subject, list):
        #     subject = subject[0]

        # self.subject = subject
        self.height, self.width = 256, 256

        # canonical
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
        scales = torch.log(torch.sqrt(dist2))[...,None]#.repeat(1, 3)
        self.pcd_scales = torch.exp(scales)

        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)
        # big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        # big_pose_smpl_param['faces'] = self.faces.detach()  # 1538,3
        self.big_pose_smpl_param = big_pose_smpl_param
        self.cano_face_area = calc_face_areas(torch.tensor(self.canonical_verts), mano_res.faces_tensor)
        # self.aug = True
        # exit()

        # 加载每个顶点的语义标签
        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        print(f'Loading MANO semantic labels from {labels_path}')
        ply = PlyData.read(labels_path)
        if ply.elements:
            pc = pd.DataFrame(ply.elements[0].data).values
        self.labels = torch.tensor(pc[:, 6].astype(np.uint8))  # (N,)

        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        # 找出所有指甲顶点
        nail_indices = np.isin(self.labels, list(NAIL_PARTS.values()))
        # self.nail_coords = pc_coords[nail_indices]
        # self.objects_dc = F.one_hot((self.labels-1).to(torch.int64), num_classes=16).unsqueeze(1).to(torch.float32).cuda()
        # 找到哪些点是指甲
        self.nail_labels = self.labels[nail_indices]

        nail_vertex_indices = np.nonzero(nail_indices)[0]
        faces_np = self.mano.faces.cpu().numpy() if torch.is_tensor(self.mano.faces) else self.mano.faces
        face_nail_mask = np.all(np.isin(faces_np, nail_vertex_indices), axis=1)
        self.nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))
        self.nail_mask_1 = self.nail_mask.to(torch.float32).unsqueeze(1)

        self.nail_faces = faces_np[face_nail_mask]

        self.bgcolor = bgcolor



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
        # h, w = image_size
        masks = np.zeros((self.height, self.width), dtype=np.uint8)

        # 相机变换
        verts_cam = np.dot(R, verts_cam.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        verts_img = np.round(verts_img[:, :2]).astype(np.int32)   # 12337, 2

        # # 计算每个面所属的语义类别（取面顶点标签的众数）
        # face_labels = self.labels[faces].numpy()
        # per_face_class = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, face_labels)

        # # 绘制mask
        # for f_idx, f in enumerate(faces):
        #     cls = int(per_face_class[f_idx])
        #     tri = verts_img[f, :]
        #     if np.any(tri < 0) or np.any(tri[:, 0] >= self.width) or np.any(tri[:, 1] >= self.height):
        #         continue
        #     cv2.fillConvexPoly(masks[cls], tri, 255)


         # 绘制指甲区域
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
        # dir_name = '/'.join(subject.split('/')[1:])

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

            # bbox
            frame_name = img['file_name']
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'], dtype=np.float32) # x,y,w,h
            if data_type != 'infer':
                if bbox[0]<10 or bbox[1]<10 or max(bbox[2], bbox[3])<80 or bbox[0]+bbox[2]>img_width-10 or bbox[1]+bbox[3]>img_height-10:
                    # print(f'{i}, Discard {image_name}, bbox is too biased/small: {bbox.tolist()}')
                    continue

                # frame
                img_path = os.path.join(self.image_dir, f'{phase}/{image_name}')
                img = cv2.imread(img_path)
                if img.max() < 20:
                    # print(f'{i}, Discard {image_name}, RGB is too dark: {img.max()}')
                    continue
                if np.allclose(img[..., 0], img[..., 1], atol=1) or np.allclose(img[..., 2], img[..., 1], atol=1) or np.allclose(img[..., 0], img[..., 2], atol=1):
                    # print(f'{i}, Discard {image_name}, Gray scale')
                    continue

                # mask
                mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                if not os.path.exists(mask_path):
                    # print(f'{i}, Discard {image_name}, w/o mask')
                    continue
                mask = cv2.imread(img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png'))
                mask_sum = mask[..., 0].astype('bool').sum()
                if mask.max() < 255 or mask_sum < 3000:
                    # print(f'{i}, Discard {image_name}, mask is too dark: {mask.max()}, {mask_sum}')
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

            # camera
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), \
                            np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            cam_T, cam_R = campos, camrot
            E = np.eye(4)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            K = np.eye(3)
            K[[0, 1], [0, 1]] = focal
            K[[0, 1], [2, 2]] = princpt
            self.cameras[f'{phase}/{image_name}'] = {
                'intrinsics': K,
                'extrinsics': [cam_R, cam_T],  # E,
                'distortions': np.zeros(5)
            }

            # mesh
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
                    # print('[WARNING] No SAM mask, use MANO mask instead')
                    maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                    alpha_mask = np.array(load_image(maskpath))

        else:
            alpha_mask = np.ones_like(orig_img) * 255

        # undistort image
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

    def __len__(self):
        return self.get_total_frames()
        # return self.epoch_size

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
        # seg_img = seg_img.astype(np.uint8)
        return seg_img


    def __getitem__(self, idx):
        # Map idx -> subject
        n_sub = len(self.subject_ids)
        # if n_sub == 0:
        #     raise RuntimeError(f"No subjects found in Hands11k dataset at {self.dat_dir}")

        subject_idx = idx % n_sub
        subject_prefix = self.subject_ids[subject_idx]
        subject_frames = self.frames_by_subject[subject_prefix]

        # load per-subject annotation (cache)
        if not hasattr(self, 'anno_cache'):
            self.anno_cache = {}
        if subject_prefix not in self.anno_cache:
            anno_path = os.path.join(self.dat_dir, 'anno', f'{subject_prefix}.json')
            if os.path.exists(anno_path):
                with open(anno_path, 'r') as f:
                    try:
                        ann = json.load(f)
                    except Exception:
                        ann = {}
            else:
                ann = {}
            frames_map = ann.get('frames', {}) if isinstance(ann, dict) else {}
            self.anno_cache[subject_prefix] = frames_map
        else:
            frames_map = self.anno_cache[subject_prefix]

        # choose frames
        if len(subject_frames) <= self.num_frames:
            chosen = list(subject_frames)
        else:
            chosen = random.sample(subject_frames, self.num_frames)

        cam_info_list = []

        for rel in chosen:
            img_path = os.path.join(self.image_dir, rel)
            fid = os.path.splitext(os.path.basename(rel))[0]
            mask_path = os.path.join(self.mask_dir, rel)

            # load image/mask
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Read mask, create default white mask if file doesn't exist
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                # print(f"Warning: Mask not found at {mask_path}, using default white mask")
                mask = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            else:
                mask = mask_raw[..., None].repeat(3, axis=2)
            img = mask * img.astype(np.float32)/255.0 + (1.0 - mask) * self.bgcolor[None, None, :]
            img = img.astype(np.float32) / 255.0
            # if img is not None:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # else:
            #     continue
            # if os.path.exists(mask_path):
            #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)
            # else:
            #     mask = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

            bkgd_mask = mask.astype('float32') / 255.0

            ann_entry = frames_map.get(fid, {})
            pred_mano_params = ann_entry.get('pred_mano_params', {}) if isinstance(ann_entry, dict) else {}

            # axis-angle from rot mats
            global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
            hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)
            if global_orient_mat.size == 0 and hand_pose_mat.size == 0:
                axis_angle = np.zeros((48,), dtype=np.float32)
            else:
                try:
                    rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)
                    axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats)).reshape(-1).numpy()
                except Exception:
                    axis_angle = np.zeros((48,), dtype=np.float32)

            betas = np.array(pred_mano_params.get('betas', []), dtype=np.float32)

            # pose MANO
            posed_res = self.mano(
                    torch.from_numpy(betas)[None].float() if betas.size else torch.zeros(1,10).float(),
                    torch.from_numpy(axis_angle)[None, :3].float(),
                    torch.from_numpy(axis_angle)[None, 3:].float(),
                    return_verts=True)
            world_vertex = posed_res.vertices.detach().numpy()

            smpl_param = {
                'poses': axis_angle,
                'shape': betas,
                'posed_verts': world_vertex.squeeze(),
            }


            min_xyz = np.min(world_vertex.squeeze(), axis=0)
            max_xyz = np.max(world_vertex.squeeze(), axis=0)
            max_xyz -= 0.05
            min_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            # intrinsics estimate
            raw_K = get_camera_parameters(max(self.height, self.width),
                                          fov=30, p_x=None, p_y=None)
            K = raw_K.squeeze().cpu().numpy()
            focal_length_y = K[1, 1]
            focal_length_x = K[0, 0]
            FovY = focal2fov(focal_length_y, self.height)
            FovX = focal2fov(focal_length_x, self.width)

            # extrinsics: try pred_cam_t
            T = np.array(ann_entry.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(-1)
            if T.size >= 2:
                T[:2] *= 10
            R = np.eye(3, dtype=np.float32)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T.reshape(3,1)

            try:
                bound_mask = get_bound_2d_mask(world_bound, raw_K.squeeze().cpu(), w2c, self.width, self.height)
                bound_mask = (np.array(bound_mask*255.0, dtype=np.byte))
            except Exception:
                bound_mask = (mask[...,0] > 0).astype(np.uint8) * 255

            # project verts to image for nail sampling
            verts_cam = np.dot(R, world_vertex.reshape(-1,3).T).T + T[None,:]
            verts_img = np.dot(K, verts_cam.T).T
            verts_img[:, :2] /= verts_img[:, 2:3]
            nail_img = verts_img[:, :2].astype(np.float32)

            # out = visualize_three_modes(verts_cam, nail_img, self.pcd_scales, K, img=img, H=256, W=256,
            #                            z_eps=1e-3, show_top_k=20, save_prefix='./hands11k_compare')

            semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1,3), K=K, R=R, T=T, faces=torch.from_numpy(self.mano.faces).long())
            nail_mask = (semantic_mask).astype('float32') / 255.
            nail_mask = nail_mask[..., None]

            image_name = os.path.basename(img_path)

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
                image_name=rel,
                width=self.width,
                height=self.height,
                smpl_param=smpl_param,
                world_vertex=world_vertex,
                world_bound=world_bound,
                big_pose_smpl_param=self.big_pose_smpl_param,
                big_pose_world_vertex=self.canonical_verts,
                big_pose_world_bound=self.big_pose_world_bound
            )

            cam_info = loadCam_aug_bs(fid, cam_info)
            cam_info_list.append(cam_info)

        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)
        img_aug = skin_tone_perturb_shared(final_results['original_image'])
        final_results['original_image'] = img_aug

        return final_results



class DistributedSameDatasetBatchSampler(Sampler):
    """
    Yields batches of indices. Each batch contains items from the SAME dataset.
    Each yielded element is a list of (ds_idx, sub_idx) pairs of length batch_size.

    DDP: batches are sharded by rank (rank r takes batches r, r+world_size, ...).
    """
    def __init__(self, dataset_lengths, ratios, batch_size, num_batches,
                 seed=1234, drop_last=True):
        self.lengths = list(dataset_lengths)
        self.ratios = np.array(ratios, dtype=np.float64)
        self.ratios = self.ratios / self.ratios.sum()
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DistributedSameDatasetBatchSampler requires torch.distributed initialized.")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        # 每个 rank 拿到的 batch 数
        return (self.num_batches + self.world_size - 1) // self.world_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        # 先生成全局 batch 列表（长度 num_batches），然后按 rank 取子序列
        all_batches = []
        for _ in range(self.num_batches):
            ds_idx = int(rng.choice(len(self.lengths), p=self.ratios))
            # 在选中的 dataset 内抽 batch_size 个 sub_idx（可重复抽样，适合你这种 epoch_size 训练）
            sub_idxs = rng.randint(0, self.lengths[ds_idx], size=self.batch_size).tolist()
            batch = [(ds_idx, si) for si in sub_idxs]
            all_batches.append(batch)

        # shard：rank r 取 r, r+world_size, ...
        for i in range(self.rank, self.num_batches, self.world_size):
            yield all_batches[i]



class Mix_Dataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]

    def __len__(self):
        # 不重要：DDP 下我们用 batch_sampler 控制步数
        return sum(self.lengths)

    def __getitem__(self, index):
        # index 由 batch sampler 生成，形如 (ds_idx, sub_idx)
        ds_idx, sub_idx = index
        sample = self.datasets[int(ds_idx)][int(sub_idx)]
        # 注入标签（建议 tensor，便于模型里分支）
        sample["dataset_id"] = torch.tensor(int(ds_idx), dtype=torch.long)
        # --- DEBUG: 找出 set 字段 ---
        for k, v in sample.items():
            if isinstance(v, set):
                raise TypeError(f"[DEBUG] Found set in sample: key={k}, ds_idx={ds_idx}, type={type(v)}, value={v}")

        return sample



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

    # if radius scalar -> expand
    if isinstance(radius_px_per_point, (int, float)) or (torch.is_tensor(radius_px_per_point) and radius_px_per_point.ndim==0):
        radius_px = torch.full((N,), float(radius_px_per_point), device=device, dtype=dtype)
    else:
        radius_px = radius_px_per_point.to(device=device, dtype=dtype)

    # We'll create lists of pixel inds and corresponding z values in chunks to avoid huge memory
    HxW = H * W
    depth_flat = torch.full((HxW,), float('inf'), device=device, dtype=dtype)

    # process in chunks
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        px = proj_xy[start:end]  # (C,2)
        z = verts_cam[start:end, 2]  # (C,)
        r = radius_px[start:end]     # (C,)

        C = px.shape[0]
        # For each point, generate pixel grid covering its radius
        # We'll vectorize by generating offsets for maximal radius in chunk
        max_r = int(math.ceil(r.max().item()))
        if max_r == 0:
            # fallback to center-only
            us = torch.round(px[:,0]).long().clamp(0, W-1)
            vs = torch.round(px[:,1]).long().clamp(0, H-1)
            inds = vs * W + us
            depth_flat.scatter_reduce_(0, inds, z, reduce='amin')
            continue

        # generate offsets grid
        offsets = torch.stack(torch.meshgrid(torch.arange(-max_r, max_r+1, device=device),
                                            torch.arange(-max_r, max_r+1, device=device)), dim=-1).reshape(-1,2)  # K x 2
        K = offsets.shape[0]

        # expand px to px_exp (C*K, 2)
        px_exp = px.unsqueeze(1).expand(-1, K, 2).reshape(-1,2)
        offs_exp = offsets.unsqueeze(0).expand(C, K, 2).reshape(-1,2)
        # effective pixel coords
        pxy = px_exp + offs_exp  # (C*K, 2)
        # mask by each point's radius: keep only offsets where sqrt(dx^2+dy^2) <= r_i
        # compute squared distances: for point i repeated K times, compare with r_i^2
        r_rep = r.unsqueeze(1).expand(-1, K).reshape(-1)  # (C*K,)
        dx = pxy[:,0] - px_exp[:,0]  # equals offs_exp[:,0]
        dy = pxy[:,1] - px_exp[:,1]
        dist2 = dx*dx + dy*dy
        mask = dist2 <= (r_rep + 1e-9)**2

        if mask.sum() == 0:
            continue
        valid_px = pxy[mask]  # (M,2)
        # pixel indices int clamp
        u = torch.round(valid_px[:,0]).long().clamp(0, W-1)
        v = torch.round(valid_px[:,1]).long().clamp(0, H-1)
        inds = v * W + u  # (M,)
        # corresponding z values: need to expand z to match mask ordering
        z_rep = z.unsqueeze(1).expand(-1, K).reshape(-1)[mask]  # (M,)

        depth_flat.scatter_reduce_(0, inds, z_rep, reduce='amin')

    depth_buffer = depth_flat.view(H, W)

    # now for each point, compute depth_at_point as minimal depth over pixels it covers
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
        # get depth buffer at these pixel positions
        vals = depth_flat[inds]  # (M,)
        # we need per-point min: map each valid entry back to point index
        # compute point indices for each mask entry
        point_idx_rep = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, K).reshape(-1)[mask]
        # scatter min per point
        for i in range(start, end):
            sel = (point_idx_rep == i)
            if sel.sum()>0:
                depth_at_point[i] = torch.min(vals[sel])
            else:
                # fallback to center pixel
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

    # 像素索引
    u_int = torch.round(u).long().clamp(0, W-1)
    v_int = torch.round(v).long().clamp(0, H-1)
    inds = v_int * W + u_int

    # 初始化 +inf 深度
    depth_flat = torch.full((H*W,), float('inf'), device=device, dtype=dtype)

    # scatter-reduce 取每个像素最小深度
    depth_flat.scatter_reduce_(0, inds, z, reduce='amin', include_self=True)
    depth_buffer = depth_flat.view(H, W)

    # 取每个点对应像素的最小深度
    depth_at_point = depth_buffer[v_int, u_int]

    # 可见性判断：点深度 ≈ 该像素最前的深度
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
    # -----------------------------
    # 1. 生成共享扰动参数
    # -----------------------------

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

    # -----------------------------
    # 2. 应用一致扰动到 batch
    # -----------------------------
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
    # Convert cam_bbox to full image
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
    # get lights in a circle around origin at elevation
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
    xyzs_cam = cam_T_world(xyzs_world, E) # xyzs_world (1,3,778)
    xys_2d = img_T_cam(xyzs_cam, K)

    # normalize to NDC space. flip xy because the ndc coord definition
    # IMPORTANT: CHECK THE DEFINITION OF NDC
    if H < W:
        xs = -((xys_2d[:, 0, :] / H) * 2. - (W / H))
        ys = -((xys_2d[:, 1, :] / H) * 2. - 1.)
    else:
        xs = -((xys_2d[:, 0, :] / W) * 2. - 1.)
        ys = -((xys_2d[:, 1, :] / W) * 2. - (H / W))
    # xs = -((xys_2d[:, 0, :] / W) * 2. - 1.)
    # ys = -((xys_2d[:, 1, :] / W) * 2. - (H / W))
    zs = xyzs_cam[:, 2]  # 1,12337
    xyzs_ndc = torch.stack([xs, ys, zs], dim=-1)   # 1,12337,3
    return xyzs_ndc


def cam_T_world(xyzs_world, E):  # xyzs_world: 1,3,12337    E:4,4  both tensor
    E=E.unsqueeze(0).float()
    xyzs_world_ = torch.cat([xyzs_world, torch.ones_like(xyzs_world[:, :1])], dim=1)   # xyzs_world (1,3,778) --> (1,4,778)
    xyzs_cam_ = torch.bmm(E, xyzs_world_)
    xyzs_cam = xyzs_cam_[:, :3] / xyzs_cam_[:, 3:]
    return xyzs_cam


def img_T_cam(xyzs_cam, K):
    K = K.unsqueeze(0).float()
    xys_ = torch.bmm(K, xyzs_cam)
    xys = xys_[:, :2] / xys_[:, 2:]
    return xys


# --- helper: fast visibility + depth (vectorized splat) ---
def compute_visibility_fast_np(verts_cam, proj_xy, pcd_scales_world, K, H=256, W=256, z_eps=1e-3):
    """
    Input: numpy arrays (or torch tensors; will be converted)
    Returns dict with r_px, depth_buf (H,W), depth_at_point (N,), residual (z - depth_at_point), vis_mask
    """
    # convert to numpy if needed
    if hasattr(verts_cam, 'cpu'): verts_cam = verts_cam.detach().cpu().numpy()
    if hasattr(proj_xy, 'cpu'): proj_xy = proj_xy.detach().cpu().numpy()
    if hasattr(pcd_scales_world, 'cpu'): pcd_scales_world = pcd_scales_world.detach().cpu().numpy().reshape(-1)
    verts_cam = np.asarray(verts_cam)
    proj_xy = np.asarray(proj_xy)
    pcd_scales_world = np.asarray(pcd_scales_world).reshape(-1)
    K = np.asarray(K)
    N = proj_xy.shape[0]

    # compute r_px and z
    z = np.maximum(verts_cam[:,2].astype(np.float64), 1e-6)
    fx = float(K[0,0])
    r_px = (pcd_scales_world * fx) / z
    # clamp to sane range to avoid explosion
    r_px = np.clip(r_px, 0.5, max(1.0, np.percentile(r_px, 98)))  # keep extreme outliers limited
    r_int = np.ceil(r_px).astype(int)

    # create flat depth buffer
    depth_flat = np.full((H*W,), np.inf, dtype=np.float64)
    ux = proj_xy[:,0].astype(np.float64)
    uy = proj_xy[:,1].astype(np.float64)
    u_floor = np.floor(ux).astype(int)
    v_floor = np.floor(uy).astype(int)

    unique_r = np.unique(r_int)
    # vectorized splat by grouping by radius
    for r in unique_r:
        idxs = np.where(r_int == r)[0]
        if idxs.size == 0: continue
        if r <= 0:
            # center only
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
        u_centers = u_floor[idxs]  # (M,)
        v_centers = v_floor[idxs]
        # broadcast and compute targets
        u_t = (u_centers[:,None] + dxs[None,:]).astype(int)
        v_t = (v_centers[:,None] + dys[None,:]).astype(int)
        # mask bounds
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

    # compute depth_at_point (min over covered pixels)
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
        # clamp
        u_t_cl = np.clip(u_t, 0, W-1); v_t_cl = np.clip(v_t, 0, H-1)
        flat_inds = (v_t_cl * W + u_t_cl)
        vals = depth_flat[flat_inds.ravel()].reshape(flat_inds.shape)
        # if all inf, fallback later
        mins = np.min(vals, axis=1)
        depth_at_point[idxs] = mins

    # fallback center if inf
    inf_mask = ~np.isfinite(depth_at_point)
    if inf_mask.any():
        uc = np.clip(np.round(ux[inf_mask]).astype(int), 0, W-1)
        vc = np.clip(np.round(uy[inf_mask]).astype(int), 0, H-1)
        depth_at_point[inf_mask] = depth_buf[vc, uc]

    residual = z - depth_at_point
    vis_mask = residual <= z_eps

    return dict(r_px=r_px, z=z, depth_buf=depth_buf, depth_at_point=depth_at_point, residual=residual, vis_mask=vis_mask)



# --- visualization: create 3-panel figure ---
def visualize_three_modes(verts_cam, proj_xy, pcd_scales_world, K, img=None, H=256, W=256,
                          z_eps=1e-3, show_top_k=20, save_prefix='./vis_compare'):
    # compute visibility & depth
    out = compute_visibility_fast_np(verts_cam, proj_xy, pcd_scales_world, K, H=H, W=W, z_eps=z_eps)
    r_px, z, depth_buf, depth_at_point, residual, vis_mask = out['r_px'], out['z'], out['depth_buf'], out['depth_at_point'], out['residual'], out['vis_mask']

    # prepare background image
    if img is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        bg_gray = True
    else:
        canvas = img.copy()
        if canvas.dtype != np.uint8:
            if canvas.max() <= 1.0:
                canvas = (canvas * 255).astype(np.uint8)
            else:
                canvas = canvas.astype(np.uint8)
        bg_gray = False

    # create three panels
    fig = plt.figure(figsize=(15,5))
    gs = fig.add_gridspec(2, 3, height_ratios=[3,1], hspace=0.25, wspace=0.2)
    ax_vis = fig.add_subplot(gs[0,0])
    ax_depth = fig.add_subplot(gs[0,1])
    ax_rad = fig.add_subplot(gs[0,2])

    # --- Panel 1: visibility overlay (outline + center points) ---
    vis_canvas = canvas.copy()
    # draw outlines: visible green, occluded red
    for i in range(proj_xy.shape[0]):
        x = proj_xy[i,0]; y = proj_xy[i,1]
        if not np.isfinite(x) or not np.isfinite(y): continue
        if x < -5 or x > W+5 or y < -5 or y > H+5: continue
        rr = int(max(1, round(r_px[i])))
        color = (0,255,0) if vis_mask[i] else (0,0,255)  # BGR
        cv2.circle(vis_canvas, (int(round(x)), int(round(y))), rr, color=color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(vis_canvas, (int(round(x)), int(round(y))), 1, (0,0,0), thickness=-1, lineType=cv2.LINE_AA)
    ax_vis.imshow(vis_canvas[..., ::-1])
    ax_vis.set_title("Visibility (green visible, red occluded)")
    ax_vis.axis('off')

    # --- Panel 2: depth heatmap overlay (use depth_buf) ---
    # normalize depth for colormap (clip extreme percentiles)
    depth_img = depth_buf.copy()
    mask = np.isfinite(depth_img)
    if mask.any():
        vmin = np.percentile(depth_img[mask], 2)
        vmax = np.percentile(depth_img[mask], 98)
    else:
        vmin, vmax = 0.0, 1.0
    norm = (depth_img - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = plt.get_cmap('plasma')
    heat = cmap(norm)[:,:,:3]  # RGB float 0..1
    # overlay with a slight alpha
    base = (canvas.astype(np.float32) / 255.0)
    overlay_depth = (0.55 * base + 0.45 * heat)
    ax_depth.imshow(overlay_depth)
    ax_depth.set_title("Depth buffer overlay (plasma)")
    ax_depth.axis('off')
    # add colorbar
    im2 = ax_depth.imshow(norm, cmap='plasma', alpha=0)
    cbar = fig.colorbar(im2, ax=ax_depth, fraction=0.046, pad=0.02)
    cbar.set_label('normalized depth')

    # --- Panel 3: radius heatmap (per-pixel average radius) ---
    # accumulate per-pixel avg r_px by rounding projection to pixel centers
    pix_sum = np.zeros((H,W), dtype=np.float64)
    pix_cnt = np.zeros((H,W), dtype=np.int32)
    u_round = np.clip(np.round(proj_xy[:,0]).astype(int), 0, W-1)
    v_round = np.clip(np.round(proj_xy[:,1]).astype(int), 0, H-1)
    for i in range(proj_xy.shape[0]):
        pix_sum[v_round[i], u_round[i]] += r_px[i]
        pix_cnt[v_round[i], u_round[i]] += 1
    pix_avg = np.zeros_like(pix_sum)
    mask2 = pix_cnt>0
    pix_avg[mask2] = pix_sum[mask2] / pix_cnt[mask2]
    # normalize and colormap
    if mask2.any():
        vmin_r = np.percentile(pix_avg[mask2], 2)
        vmax_r = np.percentile(pix_avg[mask2], 98)
    else:
        vmin_r, vmax_r = 0.0, 1.0
    norm_r = (pix_avg - vmin_r) / (vmax_r - vmin_r + 1e-12)
    norm_r = np.clip(norm_r, 0.0, 1.0)
    cmap_r = plt.get_cmap('viridis')
    heat_r = cmap_r(norm_r)[:,:,:3]
    overlay_rad = (0.6 * base + 0.4 * heat_r)
    ax_rad.imshow(overlay_rad)
    ax_rad.set_title("Radius heatmap (per-pixel avg r_px)")
    ax_rad.axis('off')
    im3 = ax_rad.imshow(norm_r, cmap='viridis', alpha=0)
    cbar3 = fig.colorbar(im3, ax=ax_rad, fraction=0.046, pad=0.02)
    cbar3.set_label('normalized radius')

    # --- Bottom row: show top-K worst occluded patches + histograms ---
    ax_patches = fig.add_subplot(gs[1, :])
    # prepare top-k worst occluded (largest residual)
    worst_idx = np.argsort(-residual)[:show_top_k]
    # arrange small patches in a grid (rows x cols)
    cols = min(10, show_top_k)
    rows = int(np.ceil(show_top_k / cols))
    patch_h = 64; patch_w = 64
    # build mosaic
    mosaic = np.ones((rows*patch_h, cols*patch_w, 3), dtype=np.uint8) * 30
    for k, idx in enumerate(worst_idx):
        r = int(max(1, round(r_px[idx])))
        cx = int(round(proj_xy[idx,0])); cy = int(round(proj_xy[idx,1]))
        col = k % cols; row = k // cols
        x0 = max(0, cx - patch_w//2); x1 = min(W, cx + patch_w//2)
        y0 = max(0, cy - patch_h//2); y1 = min(H, cy + patch_h//2)
        patch = canvas[y0:y1, x0:x1].copy()
        # place into mosaic slot (center it)
        slot_x0 = col*patch_w + (patch_w - (x1-x0))//2
        slot_y0 = row*patch_h + (patch_h - (y1-y0))//2
        mosaic[slot_y0:slot_y0+(y1-y0), slot_x0:slot_x0+(x1-x0), :] = patch
        # overlay occluding pixels (where depth equals pixel min)
        # find occluding pixel positions inside this patch
        local_depth = depth_buf[y0:y1, x0:x1]
        occ_mask = np.isfinite(local_depth) & (np.abs(local_depth - depth_at_point[idx]) <= 1e-6)
        ys, xs = np.where(occ_mask)
        for (yy, xx) in zip(ys, xs):
            ox = slot_x0 + xx; oy = slot_y0 + yy
            cv2.circle(mosaic, (ox, oy), 1, (0,255,0), thickness=-1)  # green occluder pixel
        # mark center
        center_x = slot_x0 + (cx - x0); center_y = slot_y0 + (cy - y0)
        cv2.circle(mosaic, (center_x, center_y), 3, (0,0,255), thickness=1)  # red center
    ax_patches.imshow(mosaic[..., ::-1])
    ax_patches.set_title("Top-K worst occluded points (red center) with occluding pixels (green)")
    ax_patches.axis('off')

    # save combined figure
    fig.tight_layout()
    fig.savefig(save_prefix + "_3panel.png", dpi=200, bbox_inches='tight')
    print("Saved combined visualization to:", save_prefix + "_3panel.png")

    # Also print some numeric diagnostics
    print("r_px stats: min {:.3f}, median {:.3f}, max {:.3f}".format(np.min(r_px), np.median(r_px), np.max(r_px)))
    print("Visible count: {} / {}".format(np.sum(out['vis_mask']), proj_xy.shape[0]))
    print("Residual stats: min {:.6f}, max {:.6f}, mean {:.6f}, median {:.6f}".format(np.min(residual), np.max(residual), np.mean(residual), np.median(residual)))

    return out


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

    # Initialize empty mask
    uv_mask = np.zeros((uv_resolution, uv_resolution, 1), dtype=np.uint8)

    # Get face UVs and convert to pixel coordinates
    face_uv_xy = uv_mapping['face_uv_xy']
    if isinstance(face_uv_xy, torch.Tensor):
        face_uv_xy = face_uv_xy.detach().cpu().numpy()

    # Convert normalized UV coordinates [0, 1] to pixel coordinates [0, resolution]
    face_uv_xy_pixels = face_uv_xy * (uv_resolution - 1)

    # Draw filled triangles on the mask
    for triangle in face_uv_xy_pixels:
        # Convert to integer coordinates for OpenCV
        pts = np.array(triangle, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(uv_mask, [pts], 255)

    # Normalize to [0, 1] and return as float
    return uv_mask.astype(np.float32) / 255.0


def visualize_uv_mapping(uv_mapping, save_path='./uv_debug.png', title='UV mapping debug'):
    """Render side-by-side 3D/2D views to sanity-check UV assignments."""
    if uv_mapping is None:
        raise ValueError('uv_mapping must not be None')

    required_keys = ('vert_uv', 'face_uv', 'face_uv_xy')
    missing = [key for key in required_keys if key not in uv_mapping]
    if missing:
        raise KeyError(f'uv_mapping is missing keys: {missing}')

    def _to_numpy(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        return np.asarray(val)

    vert_uv = _to_numpy(uv_mapping['vert_uv']).astype(np.float64)
    face_idx = _to_numpy(uv_mapping['face_uv']).astype(np.int64)
    face_uv_xy = _to_numpy(uv_mapping['face_uv_xy']).astype(np.float64)

    if vert_uv.ndim != 2 or vert_uv.shape[1] != 3:
        raise ValueError('uv_mapping["vert_uv"] must have shape (N, 3)')

    num_uv_verts = vert_uv.shape[0]
    uv_coords = np.zeros((num_uv_verts, 2), dtype=np.float64)
    counts = np.zeros((num_uv_verts, 1), dtype=np.float64)
    for tri, coords in zip(face_idx, face_uv_xy):
        uv_coords[tri] += coords
        counts[tri] += 1.0
    counts[counts == 0] = 1.0
    uv_coords /= counts

    u = uv_coords[:, 0]
    v = uv_coords[:, 1]
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    u_norm = (u - u_min) / (u_max - u_min + 1e-6)
    v_norm = (v - v_min) / (v_max - v_min + 1e-6)
    colors = np.stack([u_norm, v_norm, 1.0 - 0.5 * (u_norm + v_norm)], axis=1)
    face_colors = colors[face_idx].mean(axis=1)

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    tris3d = vert_uv[face_idx]
    mesh = Poly3DCollection(tris3d, facecolors=face_colors, linewidths=0.05, alpha=0.95)
    mesh.set_edgecolor((0, 0, 0, 0.2))
    ax3d.add_collection3d(mesh)
    ax3d.scatter(vert_uv[:, 0], vert_uv[:, 1], vert_uv[:, 2], c=colors, s=2)
    ranges = vert_uv.max(axis=0) - vert_uv.min(axis=0)
    max_range = float(ranges.max())
    if max_range == 0:
        max_range = 1.0
    centers = (vert_uv.max(axis=0) + vert_uv.min(axis=0)) / 2.0
    for setter, center in zip((ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim), centers):
        setter(center - max_range / 2.0, center + max_range / 2.0)
    ax3d.set_title('3D mesh colored by UV')
    ax3d.set_axis_off()
    # Set viewing angle for right hand: hand back facing viewer, thumb up
    ax3d.view_init(elev=20, azim=270)

    ax2d = fig.add_subplot(1, 2, 2)
    uv_tri = face_uv_xy.copy()
    u2_min, u2_max = float(uv_tri[:, :, 0].min()), float(uv_tri[:, :, 0].max())
    v2_min, v2_max = float(uv_tri[:, :, 1].min()), float(uv_tri[:, :, 1].max())
    uv_tri[:, :, 0] = (uv_tri[:, :, 0] - u2_min) / (u2_max - u2_min + 1e-6)
    uv_tri[:, :, 1] = (uv_tri[:, :, 1] - v2_min) / (v2_max - v2_min + 1e-6)
    for idx, tri in enumerate(uv_tri):
        ax2d.add_patch(Polygon(tri, facecolor=face_colors[idx], edgecolor='k', linewidth=0.1))
    ax2d.set_xlim(0.0, 1.0)
    ax2d.set_ylim(1.0, 0.0)
    ax2d.set_aspect('equal')
    ax2d.set_title('UV layout (normalized)')
    ax2d.axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[UV Debug] Saved visualization to {save_path}')

    stats = {
        'u_range': (u_min, u_max),
        'v_range': (v_min, v_max),
        'num_faces': int(face_idx.shape[0]),
        'num_vertices': int(num_uv_verts),
    }
    return stats




class Renderer:

    def __init__(self, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        # self.cfg = cfg
        self.focal_length = 5000
        # self.focal_length = 5000 / 256 * 256
        self.img_res = 256

        # add faces that make the hand mesh watertight
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

        # if full_frame:
        #     image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        # else:
        #     image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3,1,1)
        #     image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3,1,1)
        #     image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.
        # camera_translation = np.array([0,0,0])

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
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
            mesh.export('./hand/mesh.ply')
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

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
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=True,
        ):

        # renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
        #                                       viewport_height=render_res[1],
        #                                       point_size=1.0)
        # # material = pyrender.MetallicRoughnessMaterial(
        # #     metallicFactor=0.0,
        # #     alphaMode='OPAQUE',
        # #     baseColorFactor=(*mesh_base_color, 1.0))

        # focal_length = focal_length if focal_length is not None else self.focal_length
        focal_length = self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array([0,0,0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        # camera_center = [render_res[0] / 2., render_res[1] / 2.]
        # camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
        #                                    cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
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
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)

        # # camera_pose[:3, 3] = camera_translation
        # camera_center = [render_res[0] / 2., render_res[1] / 2.]
        # focal_length = focal_length if focal_length is not None else self.focal_length
        # camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
        #                                    cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
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
        # from phalp.visualize.py_renderer import get_light_poses
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
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
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
            # faces_per_pixel=50,
            # max_faces_per_bin=20000,
            bin_size=0,
        )
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        # self.shader = NormalShader(device='cuda')

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
        xyzs_ndc = ndc_T_world(xyzs_observation.permute(0,2,1), K, E, 256, 256)  # 1,12337,3

        mesh = Meshes(xyzs_ndc, torch.tensor(faces).unsqueeze(0).cuda())

        fragments = self.rasterizer(mesh)
        # 5️⃣ 从 fragments 中取出深度
        # fragments.zbuf 的 shape: [batch_size, H, W, faces_per_pixel]
        depth_map = fragments.zbuf[0, :, :, 0]  # 取最近的那个面深度

        # depth_map 中无效像素（未被任何面覆盖）是 NaN
        depth_map[torch.isnan(depth_map)] = 0.0

        return depth_map



if __name__ == '__main__':

    from core.data.dataset_args import DatasetArgs
    from configs import cfg, make_cfg, args
    args.cfg = 'ohta/configs/interhand/ohta_train.yaml'
    cfg = make_cfg(args)

    args = DatasetArgs.get('interhand_train')
    args['data_type'] = 'train'
    dataset = Dataset(**args)
    for i in range(0, len(dataset), len(dataset)//10):
        print(f'{i} / {len(dataset)}')
        data = dataset.__getitem__(i)
        dataset.vis(data, i)
