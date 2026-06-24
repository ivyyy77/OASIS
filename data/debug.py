import os
import random
import logging
import codecs as cs
from os.path import join as pjoin
import glob
import pickle
import json
import math

import numpy as np
from rich.progress import track

import torch
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path
import cv2
from data.hand_dataset import CameraInfo

# from .scripts.motion_process import recover_from_ric
from .utils.word_vectorizer import WordVectorizer
from torch.utils.data import Sampler

# from data.interhand.train import _read_mano_uv_obj, _resolve_mano_uv_root

from data.hand_dataset import create_dataset, merge_batch

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


logger = logging.getLogger(__name__)


PIANO_RESOLUTION = [1080, 1920]
FX = 37500


class MotionDataset(Dataset):
    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 split_file: str, motion_dir: str, window_size: int,
                 tiny: bool = False, progress_bar: bool = True, **kwargs) -> None:
        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        maxdata = 10 if tiny else 1e10
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if motion.shape[0] < window_size:
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)
        if not tiny:
            logger.info("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

        self.mean = mean
        self.std = std
        self.window_size = window_size

    def __len__(self) -> int:
        return self.cumsum[-1]

    def __getitem__(self, item: int) -> tuple:
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        return motion, self.window_size


class Text2MotionDataset(Dataset):

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        w_vectorizer: WordVectorizer,
        max_motion_length: int,
        min_motion_length: int,
        max_text_len: int,
        unit_length: int,
        motion_dir: str,
        text_dir: str,
        fps: int,
        padding_to_max: bool,
        njoints: int,
        tiny: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.padding_to_max = padding_to_max
        self.njoints = njoints

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        maxdata = 10 if tiny else 1e10
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if len(motion) < self.min_motion_length or len(motion) >= self.max_motion_length:
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < self.min_motion_length or \
                                        len(n_motion) >= self.max_motion_length:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except ValueError:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except Exception as e:
                print(e)
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if not tiny:
            logger.info(f"Reading {len(self.id_list)} motions from {split_file}.")
            logger.info(f"Total {len(name_list)} motions are used.")
            logger.info(f"{bad_count} motion sequences not within the length range of "
                        f"[{self.min_motion_length}, {self.max_motion_length}) are filtered out.")

        self.mean = mean
        self.std = std

        control_args = kwargs['control_args']
        self.control_mode = None
        if os.path.exists(control_args.MEAN_STD_PATH):
            self.raw_mean = np.load(pjoin(control_args.MEAN_STD_PATH, 'Mean_raw.npy'))
            self.raw_std = np.load(pjoin(control_args.MEAN_STD_PATH, 'Std_raw.npy'))
        else:
            self.raw_mean = self.raw_std = None
        if not tiny and control_args.CONTROL:
            self.t_ctrl = control_args.TEMPORAL
            self.training_control_joints = np.array(control_args.TRAIN_JOINTS)
            self.testing_control_joints = np.array(control_args.TEST_JOINTS)
            self.training_density = control_args.TRAIN_DENSITY
            self.testing_density = control_args.TEST_DENSITY

            self.control_mode = 'val' if ('test' in split_file or 'val' in split_file) else 'train'
            if self.control_mode == 'train':
                logger.info(f'Training Control Joints: {self.training_control_joints}')
                logger.info(f'Training Control Density: {self.training_density}')
            else:
                logger.info(f'Testing Control Joints: {self.testing_control_joints}')
                logger.info(f'Testing Control Density: {self.testing_density}')
            logger.info(f"Temporal Control: {self.t_ctrl}")

        self.data_dict = data_dict
        self.name_list = name_list

    def __len__(self) -> int:
        return len(self.name_list)

    def random_mask(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        choose_joint = self.testing_control_joints

        length = joints.shape[0]
        density = self.testing_density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, self.njoints, 3))
        for cj in choose_joint:
            mask_seq[choose_seq, cj] = 1.0

        joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * mask_seq
        return joints, mask_seq

    def random_mask_train(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.t_ctrl:
            choose_joint = self.training_control_joints
        else:
            num_joints = len(self.training_control_joints)
            num_joints_control = 1
            choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
            choose_joint = self.training_control_joints[choose_joint]

        length = joints.shape[0]

        if self.training_density == 'random':
            choose_seq_num = np.random.choice(length - 1, 1) + 1
        else:
            choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, self.njoints, 3))
        for cj in choose_joint:
            mask_seq[choose_seq, cj] = 1

        joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * mask_seq
        return joints, mask_seq

    def __getitem__(self, idx: int) -> tuple:
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        hint, hint_mask = None, None
        if self.control_mode is not None:
            joints = recover_from_ric(torch.from_numpy(motion).float(), self.njoints)
            joints = joints.numpy()
            if self.control_mode == 'train':
                hint, hint_mask = self.random_mask_train(joints)
            else:
                hint, hint_mask = self.random_mask(joints)

            if self.padding_to_max:
                padding = np.zeros((self.max_motion_length - m_length, *hint.shape[1:]))
                hint = np.concatenate([hint, padding], axis=0)
                hint_mask = np.concatenate([hint_mask, padding], axis=0)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.padding_to_max:
            padding = np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)

        return (word_embeddings,
                pos_one_hots,
                caption,
                sent_len,
                motion,
                m_length,
                "_".join(tokens),
                (hint, hint_mask))


class ReconstructionDataset(Dataset):
    def __init__(self, split='train', frm_list=None):
        # self.config = config
        self.split = split

        self.dat_dir = '/home/ubuntu/hanhan/LHM/data/language2motion/'  # config.dat_dir
        # self.cameras_extent = config.get('cameras_extent', 1.0)

        self.base_dir = '/home/ubuntu/hanhan/LHM/data/new-data/'
        # self.base_dir = os.path.abspath(os.path.join(self.dat_dir, '../new-data'))

        self.id_list = self.load_txt_file(f"{split}_ids.txt")   # 是个 list
        self.frame_list = self.load_txt_file(f"{split}.txt")   # 是个 list
        self.text_list = self.load_txt_file(f"{split}_texts.txt")

        # self.num_for_train = config.get('num_for_train', -350)

        # self.load_flame_json()
        self.num_frames = len(self.frame_list)
        print(f'[SignAvatar Reconstruction Dataset][{self.split}] num_videos = {self.num_frames}')

        # # npy_to_obj('./data/0.npy', 'zju-mesh.obj')
        # from data.smplx2bvh import smpl2bvh
        # smpl2bvh()

    ##################################################
    # load flame_params.json
    def load_flame_json(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'flame_params.json')):
            print('[ERROR] flame_params.json not existed in the data folder')
            raise NotImplementedError

        with open(os.path.join(self.dat_dir, 'flame_params.json')) as fp:
            contents = json.load(fp)
            self.intrinsics = contents['intrinsics']
            self.shape_params = torch.tensor(contents['shape_params']).unsqueeze(0)

            if self.split == 'train':
                self.frames_info = contents['frames'][:self.num_for_train]
            else:
                self.frames_info = contents['frames'][self.num_for_train:]

            self.frm_list = []
            self.flame_params = []
            for frame in self.frames_info:
                frm_idx = os.path.basename(frame['file_path'])
                self.frm_list.append(frm_idx)
                self.flame_params.append({
                    'full_pose': torch.tensor(frame['pose']).unsqueeze(0),
                    'expression_params': torch.tensor(frame['expression']).unsqueeze(0),
                })
        self.flame = FLAME('model/imavatar/FLAME2020/generic_model.pkl',
                           'model/imavatar/FLAME2020/landmark_embedding.npy',
                           n_shape=100,
                           n_exp=50,
                           shape_params=self.shape_params,
                           canonical_expression=None,
                           canonical_pose=None)
        self.mesh_py3d = py3d_meshes.Meshes(self.flame.v_template[None, ...].float(),
                                            torch.from_numpy(self.flame.faces[None, ...].astype(int)))

    ##################################################
    def load_txt_file(self, filename):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        # print(f'getitem called with idx: {idx}')
        # 1. 随机采样一个 id（如果 idx 为 None）
        if idx is None:
            idx   = torch.randint(0, len(self.id_list), (1,)).item()
        person_id = self.id_list[int(idx)]

        # ====================================== DEBUG =======================================
        person_id = '--7E2sU6zP4'
        # person_id = '--8pSDeC-fg'
        # person_id = '--dANj_01AU'
        # person_id = 'furfAAKJFvw'

        # 2. 在 images/ 下找到所有以 "{person_id}_" 开头的子文件夹
        images_root = os.path.join(self.dat_dir, 'images')
        video_dirs  = [
            d for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d)) and d.startswith(f"{person_id}_")
        ]

        # 3. 构建该 id 下所有视频的列表，包括 images、masks、annotation
        data_entries = []
        for video_name in video_dirs:
            img_dir  = os.path.join(images_root, video_name)
            mask_dir = os.path.join(self.dat_dir, 'masks', video_name)

            img_paths  = sorted(glob.glob(os.path.join(img_dir, '[0-9][0-9][0-9][0-9][0-9].png')))
            mask_paths  = sorted(glob.glob(os.path.join(mask_dir, '[0-9][0-9][0-9][0-9][0-9].png')))
            # mask_paths = [os.path.join(mask_dir, os.path.basename(p))
            #               for p in img_paths
            #               if os.path.isfile(os.path.join(mask_dir, os.path.basename(p)))]

            # 读取 annotation
            ann_path = os.path.join(self.dat_dir, 'annotations', f"{video_name}.pkl")
            if not os.path.isfile(ann_path):
                raise FileNotFoundError(f"Missing annotation for video {video_name}")
            with open(ann_path, 'rb') as f:
                annotation = pickle.load(f)
                # annotation = torch.load(f, map_location='cpu')  # ✅ 改成这样

            data_entries.append({
                'video_name': video_name,
                'images':     img_paths,
                'masks':      mask_paths,
                'annotation': annotation,
            })

        if not data_entries:
            raise RuntimeError(f"No videos found for ID {person_id}")

        # 4. 从所有帧中随机挑一帧作本次 sample
        #    我们先 flatten 所有 (video_name, img_path, mask_path)
        frame_records = []
        for entry in data_entries:
            vn = entry['video_name']
            for img_path, mask_path in zip(entry['images'], entry['masks']):
                frame_records.append((vn, img_path, mask_path, entry['annotation']))

        video_name, image_path, mask_path, annotation = random.choice(frame_records)

        # 5. 组织“所有视频”级的返回结构
        all_annotations = {e['video_name']: e['annotation'] for e in data_entries}
        all_images      = {e['video_name']: e['images']     for e in data_entries}
        all_masks       = {e['video_name']: e['masks']      for e in data_entries}

        # 6. 最终返回（保留原有字段 + 新增全集信息）
        return {
            'id':             person_id,       # e.g. "-fZc293MpJk"
            'video_name':     video_name,      # e.g. "-fZc293MpJk_3-1-rgb_front"
            'image_path':     image_path,      # 单帧图片路径
            'mask_path':      mask_path,       # 单帧掩码路径
            'annotation':     annotation,      # 单帧对应的 .pkl 内容

            # —— 新增 —— 全部视频级信息
            'all_annotations': all_annotations,  # { video_name: pkl 内容, … }
            'all_images':      all_images,       # { video_name: [所有帧路径,…], … }
            'all_masks':       all_masks,        # { video_name: [所有掩码路径,…], … }
        }


class HandDataset(Dataset):
    def __init__(self, split='train', frm_list=None, num_frames=4, img_path='./example_data/coco-ours/images/000000042103_0.png', edit=False):
        self.split = split
        if split == 'train':
            self.dat_dir = '/scratch/groups/su004-neuralnet/zh174/hand-data/our_data'
            print("\033[91m[========== Loading Our Training Dataset ==========]\033[0m")
            self.num_frames = num_frames
        elif split == 'test':
            self.dat_dir = '/home/z/zh174/eval_debug'
            print("\033[91m[========== Loading Our Testing Dataset ==========]\033[0m")
            self.num_frames = 1
        elif split == 'test_wild':
            img_path_abs = os.path.abspath(img_path)
            self.img_path_abs = img_path_abs
            images_dir = os.path.dirname(img_path_abs)          # .../coco-ours/images
            self.dat_dir = os.path.dirname(images_dir)          # .../coco-ours

            print("\033[91m[========== Loading In_the_Wild Data ==========]\033[0m")
            self.num_frames = 1
            self.video_ids = [os.path.basename(img_path_abs)]
            self.edit = edit


        if split in ['train', 'test']:
            # List all video ids (subfolders in images/)
            images_root = os.path.join(self.dat_dir, 'images')
            self.video_ids = [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
            self.video_ids.sort()
            print(f'[Our Hand Dataset] Found {len(self.video_ids)} videos for {self.split}')

        self.height, self.width = 256, 256

        self.mano = tools_utils.model.smplx_ours.create(**cfg.smpl_cfg)
        self.mano_wild_778 = tools_utils.model.smplx_ours.create(**cfg.smpl_cfg)
        self.mano_ori = tools_utils.model.smplx.create(**cfg.smpl_cfg)
        self.mano_778 = tools_utils.model.smplx.create(**cfg.smpl_cfg)

        # if cfg.smpl_cfg['manohd'] > 0:
        #     print('MANO-HD in Model')
        #     smpl_body, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        #     lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        #     smpl_body.lbs_weights = lbs_weights
        #     self.mano = smpl_body


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
            torch.zeros(1, 10).float(),  # betas
            torch.zeros(1, 3).float(),  # global_orient 确定！
            torch.zeros(1, 45).float(), return_verts=True)
        # self.canonical_bbox = mano_res.joints[0].numpy()
        self.canonical_verts = mano_res.vertices.detach().numpy().squeeze()
        # self.canonical_bbox = self.skeleton_to_bbox(self.canonical_verts)
        # vertices = mano_res.vertices
        self.faces = mano_res.faces_tensor
        self.big_pose_xyz = mano_res.vertices.detach().numpy().squeeze()  # ndarray: 778,3
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)
        # big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        big_pose_smpl_param['faces'] = self.faces.detach()  # 1538,3
        self.big_pose_smpl_param = big_pose_smpl_param
        big_pose_min_xyz = np.min(self.big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(self.big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        self.big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(self.canonical_verts)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]#.repeat(1, 3)
        self.pcd_scales = torch.exp(scales)

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
        # seg_img = seg_img.astype(np.uint8)
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

        # _, mask_bin = cv2.threshold(mask_u8[..., 0], 127, 255, cv2.THRESH_BINARY)
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
        # image_name = os.path.basename(img_path)
        # frame_name = '000000087811_1.png'
        frame_name = self.video_ids[0]
        img_path = os.path.join(self.dat_dir, 'images', frame_name)
        mask_path = img_path.replace('images', 'masks')#.replace('.jpg', '.png')
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if self.edit:
            mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'
        mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'
        if mask_edit_path is not None and os.path.exists(mask_edit_path):
            mask_edit = cv2.imread(mask_edit_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

        # anno_path = img_path.replace('images', 'anno').replace('.png', '.pkl')
        anno_path = img_path.replace('images', 'anno')
        anno_path = os.path.splitext(anno_path)[0] + '.pkl'

        # if anno_path is None or mask_path is None:
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

        if ann.get('pred_mano_params', {}) == {}:
            ann = ann.get('frames', {})
            ann = ann.get(os.path.splitext(frame_name)[0], {})

        # bgcolor = np.array(self.bgcolor, dtype='float32')
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

        #########################################################################################
        raw_K = get_camera_parameters(
            # max(512, 314),
            # fov=60,
            max(self.height, self.width),
            # fov=5000, # 60
            # fov=25, # 60   interhand 有点大
            # fov=40,  # 30
            # fov=35,  # 30
            fov=30,  # 30
            # fov=32,  # 30
            # fov=60, # 物体太小
            p_x=None,
            p_y=None,
            device='cuda',
        )
        K = raw_K.numpy()
        # K = raw_K.squeeze().cpu().numpy()
        focal_length_y = K[1, 1]
        focal_length_x = K[0, 0]
        FovY = focal2fov(focal_length_y, self.height)  # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
        FovX = focal2fov(focal_length_x, self.width)  # 与 splatting中操作一致
        #########################################################################################

        # Fill missing fields with None or zeros
        bkgd_mask = mask.astype('float32') / 255.0

        # if self.edit:
        # if mask_edit_path is not None:
        if os.path.exists(mask_edit_path):
            bound_mask = mask_edit.astype('float32') / 255.0
        else:
            bound_mask = bkgd_mask

        # bound_mask = None
        # Fill smpl_param from pred_mano_params
        pred_mano_params = ann.get('pred_mano_params', {})

        # Convert (16,3,3) rotation matrices to (16,3) axis-angle as in hand_dataset.py
        global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
        hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)
        # Concatenate to (16,3,3) if needed
        rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)
        # Convert to torch tensor for matrix_to_axis_angle
        axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats))  # (16,3)
        axis_angle = axis_angle.reshape(-1).numpy()   # (48,)

        ##########################################################################3
        posed_res = self.mano(
            torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
            torch.from_numpy(axis_angle)[None, :3].float(),
            torch.from_numpy(axis_angle)[None, 3:].float(),
            # torch.from_numpy(T).float(),
            return_verts=True)
        world_vertex = posed_res.vertices.detach().numpy()
        smpl_param = {
            'poses': axis_angle,
            'shape': np.array(pred_mano_params.get('betas', []), dtype=np.float32),
            'posed_verts': world_vertex.squeeze(),
            # 'trans': np.array(T).reshape(-1),
        }


        mano_shape = self.mano(
            torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),  # betas
            torch.zeros(1, 3).float(),  # global_orient 确定！
            torch.zeros(1, 45).float(), return_verts=True)
        # self.canonical_bbox = mano_res.joints[0].numpy()
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
        # bound_mask = get_bound_2d_mask(world_bound, raw_K.squeeze().cpu(), w2c, self.width, self.height)
        # bound_mask = (np.array(bound_mask*255.0, dtype=np.byte))

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None].repeat(3, axis=2)
        # nail_img = nail_mask * img + (1.0 - nail_mask / 255.) * bgcolor[None, None, :]

        # # 计算顶点的图像坐标，用于后续采样
        verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
        verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = (verts_img[:, :2]).astype(np.float32)   # 12337, 2


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
            # from data.interhand.train import visualize_uv_mapping
            # debug_path = f'./uv_debug/{frame_name.replace("/", "_")}.png'
            # os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            # visualize_uv_mapping(uv_mapping, save_path=debug_path, title=f'UV debug {frame_name}')

        cam_info_list.append(cam_info)
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)
        # final_results = cam_info_dicts

        return final_results



    def get_img_rainbow(self):
        # image_name = os.path.basename(img_path)
        # frame_name = '000000087811_1.png'
        frame_name = self.video_ids[0]
        img_path = os.path.join(self.dat_dir, 'images', frame_name)
        mask_path = img_path.replace('images', 'masks')#.replace('.jpg', '.png')
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if self.edit:
            mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'
        # anno_path = img_path.replace('images', 'anno').replace('.png', '.pkl')
        anno_path = img_path.replace('images', 'anno')
        anno_path = os.path.splitext(anno_path)[0] + '.pkl'

        # if anno_path is None or mask_path is None:
        if not os.path.exists(mask_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            mask_path = os.path.join(self.dat_dir, 'masks', generate_name)
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if not os.path.exists(anno_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            anno_path = os.path.join(self.dat_dir, 'anno', generate_name)
            anno_path = os.path.splitext(anno_path)[0] + '.pkl'

        # print(f'Loading image from {anno_path}')
        # exit(0)
        with open(anno_path, 'rb') as fi:
            cameras, mesh_infos, bbox, img_type = pickle.load(fi)

        # bgcolor = np.array(self.bgcolor, dtype='float32')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)
        mask = self.resize_mask(mask, (self.width, self.height))
        if self.edit:
            mask_edit = cv2.imread(mask_edit_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

        img = self.process_image(img, mask)
        cam_info_list = []

        # w2c = np.eye(4, dtype=np.float32)
        # T = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(3,1)
        # T[:2] *= 10
        # T = T.reshape(3)
        # R = np.eye(3, dtype=np.float32)
        # w2c[:3, :3] = R
        # w2c[:3, 3:4] = T[..., None]

        dst_skel_info = query_dst_skeleton(mesh_infos)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']#.reshape(1, 10)
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_cam_joints = dst_skel_info['joint_cam']
        dst_valid_joints = dst_skel_info['joint_valid']

        E = apply_global_tfm_to_camera(
                E=np.eye(4),
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]


        #########################################################################################
        K = cameras['intrinsics'][:3, :3].copy()
        # K = raw_K.squeeze().cpu().numpy()
        focal_length_y = K[1, 1]
        focal_length_x = K[0, 0]
        FovY = focal2fov(focal_length_y, self.height)  # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
        FovX = focal2fov(focal_length_x, self.width)  # 与 splatting中操作一致
        #########################################################################################

        # Fill missing fields with None or zeros
        bkgd_mask = mask.astype('float32') / 255.0
        if self.edit:
            bound_mask = mask_edit.astype('float32') / 255.0
        else:
            bound_mask = bkgd_mask
        ##########################################################################3
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
        # bound_mask = get_bound_2d_mask(world_bound, K.squeeze().cpu(), w2c, self.width, self.height)
        # bound_mask = (np.array(bound_mask*255.0, dtype=np.byte))

        semantic_mask = self.image_semantic_mask(world_vertex.reshape(-1, 3), K=K, R=R, T=T,
                                                                        faces=torch.from_numpy(self.mano.faces).long())
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None].repeat(3, axis=2)
        # nail_img = nail_mask * img + (1.0 - nail_mask / 255.) * bgcolor[None, None, :]

        # # 计算顶点的图像坐标，用于后续采样
        verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
        verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = (verts_img[:, :2]).astype(np.float32)   # 12337, 2


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
            # from data.interhand.train import visualize_uv_mapping
            # debug_path = f'./uv_debug/{frame_name.replace("/", "_")}.png'
            # os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            # visualize_uv_mapping(uv_mapping, save_path=debug_path, title=f'UV debug {frame_name}')
        cam_info_list.append(cam_info)
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)
        # final_results = cam_info_dicts

        return final_results



    def __getitem__(self, idx, attempt=0, max_attempts=10):

        # if self.split in ['test_wild']:
        #     frame_name = self.video_ids[0]  # 例如 "000000087811_1.png"
        #     final_results = self.get_img()
        #     return final_results

        if self.split in ['test_wild']:
            frame_name = self.video_ids[0]  # 例如 "000000087811_1.png"

            dat_dir_parts = set(Path(self.dat_dir).as_posix().split('/'))
            # if 'coco-ours' in dat_dir_parts:
            if ('coco-ours' in dat_dir_parts) or ('coco' in self.img_path_abs):
                final_results = self.get_img()
            elif 'in_the_wild' in dat_dir_parts:
                if frame_name in ['02023.jpg', '12023.jpg']:
                    final_results = self.get_img_rainbow()
                else:
                    final_results = self.get_img()
            else:
                # final_results = self.render_piano()
                final_results = self.get_img_rainbow()
                # final_results = self.get_img()

            return final_results

        while attempt < max_attempts:
            video_id = self.video_ids[idx]
            # video_id = 'w5xhbo7tCvH' # '00eT0ZOThXq'  # DEBUG
            # For the given video_id, list all image files in images/ and masks/, and load annotation json
            img_dir = os.path.join(self.dat_dir, 'images', video_id)
            mask_dir = os.path.join(self.dat_dir, 'masks', video_id)
            vis_dir = os.path.join(self.dat_dir, 'vis', video_id)
            ann_path = os.path.join(self.dat_dir, 'annotations', f"{video_id}.json")
            # List all image files in img_dir
            img_paths = sorted([f for f in glob.glob(os.path.join(img_dir, '*.png'))])
            mask_paths = sorted([f for f in glob.glob(os.path.join(mask_dir, '*.png'))])
            # Load annotation json
            # import json
            if not os.path.isfile(ann_path):
                print(f"Missing annotation for video {video_id}")
                idx = (idx + 1) % len(self.video_ids)
                attempt += 1
                continue
            with open(ann_path, 'r') as f:
                annotation = json.load(f)


            # Frame id is usually the stem of the filename (without extension)
            # 只保留右手（fid以'_1'结尾）
            frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths if os.path.splitext(os.path.basename(p))[0].endswith('_1')]
            # Build a mapping from frame_id to mask path
            mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
            # Build a mapping from frame_id to annotation (assume annotation['frames'][frame_id] exists)
            ann_map = annotation.get('frames', {})
            # Sample 4 frames (randomly, or first 4 if not enough)
            if len(frame_ids) < self.num_frames:
                # chosen = frame_ids
                # print(f'Video {video_id} 只有 {len(frame_ids)} 帧, 总数少于 {self.num_frames}')
                # 尝试采样左手
                if self.split == 'test':
                    idx = (idx + 1) % len(self.video_ids)
                    attempt += 1
                    continue
                frame_ids_left = [os.path.splitext(os.path.basename(p))[0] for p in img_paths if os.path.splitext(os.path.basename(p))[0].endswith('_0')]
                if len(frame_ids) <= 1 and len(frame_ids_left) > 1:
                    frame_ids = frame_ids_left
                    # print(f'尝试采样左手，当前有 {len(frame_ids)} 帧')
                    if len(frame_ids) < self.num_frames:
                        # chosen = frame_ids
                        idx = (idx + 1) % len(self.video_ids)
                        attempt += 1
                        continue
                    else:
                        chosen = random.sample(frame_ids, self.num_frames)
                else:
                    # print(f'Video {video_id} 只有 {len(frame_ids)} frames, 总数少于 {self.num_frames}')
                    idx = (idx + 1) % len(self.video_ids)
                    attempt += 1
                    continue
            else:
                if self.split == 'test':
                    chosen = frame_ids[:1]   # 只取第一帧
                chosen = random.sample(frame_ids, self.num_frames)




            # print(f'Video {video_id} 采样 frames: {chosen}')
            cam_info_list = []
            for fid in chosen:
                img_path = os.path.join(img_dir, f'{fid}.png')
                # vis_path = os.path.join(vis_dir, f'{fid}.png')
                mask_path = mask_map.get(fid, None)
                ann = ann_map.get(fid, {})
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

                img = self.process_image(img, mask)

                # from tools_utils import libcore
                # vis_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)
                # libcore.write_tensor_image('./hand/debug.jpg', torch.tensor(img), rgb2bgr=True)
                # libcore.write_tensor_image('./hand/debug_vis.jpg', torch.tensor(vis_img/255.), rgb2bgr=True)
                # height, width = img.shape[:2] if img is not None else (0, 0)
                # Camera parameters from WiLoR annotation
                # Use pred_cam_t_full as T, identity for R, default K
                w2c = np.eye(4, dtype=np.float32)
                T = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(3,1)
                # T_1 = T
                # T[0], T[1] = T_1[1], T_1[0]
                T[:2] *= 10
                T = T.reshape(3)
                # T *= 10.0
                # T[0] /= -1
                # T[0] /= -10.0
                R = np.eye(3, dtype=np.float32)
                w2c[:3, :3] = R
                w2c[:3, 3:4] = T[..., None]

                #########################################################################################
                # # No intrinsics in WiLoR, use identity or estimate if possible
                # K = np.eye(3, dtype=np.float32)  # No intrinsics in WiLoR, use identity or estimate if possible
                # # FovY/FovX: estimate from K and image size if possible
                # fy = K[1,1]
                # fx = K[0,0]
                # FovY = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi if fy > 0 and height > 0 else 0.0
                # FovX = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi if fx > 0 and width > 0 else 0.0

                #########################################################################################
                raw_K = get_camera_parameters(
                    # max(512, 314),
                    # fov=60,
                    max(self.height, self.width),
                    # fov=5000, # 60
                    # fov=25, # 60   interhand 有点大
                    fov=30,  # 30
                    # fov=60, # 物体太小
                    p_x=None,
                    p_y=None,
                    device='cuda',
                )
                K = raw_K.numpy()
                # K = raw_K.squeeze().cpu().numpy()
                focal_length_y = K[1, 1]
                focal_length_x = K[0, 0]
                FovY = focal2fov(focal_length_y, self.height)  # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
                FovX = focal2fov(focal_length_x, self.width)  # 与 splatting中操作一致
                #########################################################################################

                image_name = os.path.basename(img_path)
                # Fill missing fields with None or zeros
                bkgd_mask = mask.astype('float32') / 255.0
                # bound_mask = None
                # Fill smpl_param from pred_mano_params
                pred_mano_params = ann.get('pred_mano_params', {})

                # Convert (16,3,3) rotation matrices to (16,3) axis-angle as in hand_dataset.py
                global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
                hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)
                # Concatenate to (16,3,3) if needed
                rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)
                # Convert to torch tensor for matrix_to_axis_angle
                axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats))  # (16,3)
                axis_angle = axis_angle.reshape(-1).numpy()   # (48,)

                # trans = np.zeros_like(T).reshape(-1)
                # trans[2] = T[2]  # keep z translation only
                # T = np.zeros_like(T)  # set T to zero, since it's included in smpl_param
                ##########################################################################3
                posed_res = self.mano(
                    torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
                    torch.from_numpy(axis_angle)[None, :3].float(),
                    torch.from_numpy(axis_angle)[None, 3:].float(),
                    # torch.from_numpy(T).float(),
                    return_verts=True)
                world_vertex = posed_res.vertices.detach().numpy()
                smpl_param = {
                    'poses': axis_angle,
                    'shape': np.array(pred_mano_params.get('betas', []), dtype=np.float32),
                    'posed_verts': world_vertex.squeeze(),
                    # 'trans': np.array(T).reshape(-1),
                }

                # verts = world_vertex.squeeze()
                # hand_center = np.mean(verts, axis=0, keepdims=True).T  # (3,1)
                # T = hand_center - T

                # bbox = self.skeleton_to_box(world_vertex.squeeze())
                # T_bbox = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(-1)
                # T_bbox[2] /= 10
                # # vis2 = draw_3d_bbox_on_image(img, bbox, K, R, T_bbox, color=(0,0,255), thickness=2,
                # #              save_path=f'./hand/bbox_vis.jpg')
                # bound_mask = project_2d_bbox(K, R, T_bbox, bbox, self.height, self.width)
                # bound_mask = bound_mask[..., None].astype('float32')
                # # print(bound_mask.shape, bound_mask.dtype, bound_mask.max())

                # # exit()
                ##############################################################################
                # world_bound = ann.get('box_center', None)  # or ann.get('bbox', None) if available
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
                # nail_img = nail_mask * img + (1.0 - nail_mask / 255.) * bgcolor[None, None, :]

                # # 计算顶点的图像坐标，用于后续采样
                verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
                verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
                verts_img = np.dot(K, verts_cam.T).T
                verts_img[:, :2] /= verts_img[:, 2:3]
                nail_img = (verts_img[:, :2]).astype(np.float32)   # 12337, 2

                # # debug
                # from .interhand.train import visualize_three_modes
                # out = visualize_three_modes(verts_cam, nail_img, self.pcd_scales, K, img=img, H=256, W=256,
                #                             z_eps=1e-3, show_top_k=20, save_prefix='./hand/vis_compare')

                # cam_info.append(CameraInfo(
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

            # Convert CameraInfo objects to dicts for merge_batch
            cam_info_dicts = [vars(c) for c in cam_info_list]
            final_results = merge_batch(cam_info_dicts)

            # return cam_info_list
            return final_results


    def _build_piano_intrinsics(self, piano_meta, target_height=None, target_width=None):
        src_w = float(piano_meta.get('width', PIANO_RESOLUTION[1]))
        src_h = float(piano_meta.get('height', PIANO_RESOLUTION[0]))
        fx = float(piano_meta.get('fx', FX))
        tgt_w = float(self.width if target_width is None else target_width)
        tgt_h = float(self.height if target_height is None else target_height)
        scale_x = np.float32(tgt_w) / np.float32(src_w)
        scale_y = np.float32(tgt_h) / np.float32(src_h)
        fx_scaled = fx * scale_x
        fy_scaled = fx * scale_y
        cx = (src_w * 0.5) * scale_x
        cy = (src_h * 0.5) * scale_y
        K = np.array(
            [[fx_scaled, 0.0, cx],
             [0.0, fy_scaled, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        FovY = focal2fov(float(fy_scaled), tgt_h)
        FovX = focal2fov(float(fx_scaled), tgt_w)
        return K, FovY, FovX


    def render_piano(self):
        # image_name = os.path.basename(img_path)
        # frame_name = '000000087811_1.png'
        frame_name = self.video_ids[0]
        img_path = os.path.join(self.dat_dir, 'images', frame_name)
        mask_path = img_path.replace('images', 'masks')#.replace('.jpg', '.png')
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if self.edit:
            mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'
        anno_piano = '/home/z/zh174/lhm-hand-new/hand/BV1ac411g7RJ_seq_0000.json'

        # if anno_path is None or mask_path is None:
        if not os.path.exists(mask_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            mask_path = os.path.join(self.dat_dir, 'masks', generate_name)
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if not os.path.exists(anno_piano):
            raise FileNotFoundError(f'Piano annotation not found: {anno_piano}')

        with open(anno_piano, 'r') as f:
            piano_meta = json.load(f)

        # bgcolor = np.array(self.bgcolor, dtype='float32')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)
        mask = self.resize_mask(mask, (self.width, self.height))
        if self.edit:
            mask_edit = cv2.imread(mask_edit_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

        img = self.process_image(img, mask)
        cam_info_list = []


        mano_piano_right = np.array(piano_meta.get('right', []), dtype=np.float32)
        if mano_piano_right.size == 0:
            raise ValueError('Missing right-hand pose data in piano annotation')

        frame_stem = os.path.splitext(os.path.basename(frame_name))[0]
        frame_digits = ''.join(ch for ch in frame_stem if ch.isdigit())
        frame_id = int(frame_digits) if frame_digits else 0
        start_frame = int(piano_meta.get('start_frame', 0))
        row_idx = frame_id - start_frame
        if row_idx < 0 or row_idx >= mano_piano_right.shape[0]:
            row_idx = 0

        pose_row = mano_piano_right[row_idx]
        cam_t = pose_row[:3].astype(np.float32)
        global_orient = pose_row[3:6].astype(np.float32)
        hand_pose = pose_row[6:51].astype(np.float32)
        dst_poses = np.concatenate([global_orient, hand_pose], axis=0)
        dst_shape = np.zeros(10, dtype=np.float32)
        generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
        shape_path = os.path.join(self.dat_dir, 'anno', generate_name)
        shape_path = os.path.splitext(shape_path)[0] + '.pkl'
        if os.path.exists(shape_path):
            with open(shape_path, 'rb') as fi:
                _, mesh_infos, _, _ = pickle.load(fi)
            dst_skel_info = query_dst_skeleton(mesh_infos)
            dst_shape = dst_skel_info['shape']

        ############################# PIANO

        R = np.eye(3, dtype=np.float32)
        # scale_mats = np.eye(4)
        # scale_mats[:3, :3] = R
        T = np.zeros(3, dtype=np.float32)
        # scale_mats[:3, 3] = cam_t

        #########################################################################################
        K, FovY, FovX = self._build_piano_intrinsics(piano_meta)
        #########################################################################################

        # Fill missing fields with None or zeros
        bkgd_mask = mask.astype('float32') / 255.0
        if self.edit:
            bound_mask = mask_edit.astype('float32') / 255.0
        else:
            bound_mask = bkgd_mask
        ##########################################################################3
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
        world_vertex = world_vertex + cam_t.reshape(1, 1, 3)
        posed_verts_778 = posed_778.vertices.detach().numpy()
        posed_verts_778 = posed_verts_778 + cam_t.reshape(1, 1, 3)

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

        smpl_param = {
            'poses': dst_poses,
            'shape': dst_shape,
            'posed_verts': world_vertex.squeeze(),
            'trans': cam_t,
        }

        min_xyz = np.min(world_vertex.squeeze(), axis=0)
        max_xyz = np.max(world_vertex.squeeze(), axis=0)
        max_xyz -= 0.05
        min_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)
        # bound_mask = get_bound_2d_mask(world_bound, K.squeeze().cpu(), w2c, self.width, self.height)
        # bound_mask = (np.array(bound_mask*255.0, dtype=np.byte))

        semantic_mask = self.image_semantic_mask(
            world_vertex.reshape(-1, 3),
            K=K,
            R=R,
            T=T,
            faces=torch.from_numpy(self.mano.faces).long(),
        )
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None].repeat(3, axis=2)
        # nail_img = nail_mask * img + (1.0 - nail_mask / 255.) * bgcolor[None, None, :]

        # # 计算顶点的图像坐标，用于后续采样
        verts_cam = world_vertex.reshape(-1, 3)#[self.nail_mask]
        verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = (verts_img[:, :2]).astype(np.float32)   # 12337, 2


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
            # from data.interhand.train import visualize_uv_mapping
            # debug_path = f'./uv_debug/{frame_name.replace("/", "_")}.png'
            # os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            # visualize_uv_mapping(uv_mapping, save_path=debug_path, title=f'UV debug {frame_name}')
        cam_info_list.append(cam_info)
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)
        # final_results = cam_info_dicts

        return final_results





    @torch.no_grad()
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
        # h, w = image_size
        # T = T.reshape(3)
        h = self.height if height is None else height
        w = self.width if width is None else width
        masks = np.zeros((h, w), dtype=np.uint8)

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
            if np.any(tri_pts < 0) or np.any(tri_pts[:, 0] >= w) or np.any(tri_pts[:, 1] >= h):
                continue
            cv2.fillConvexPoly(masks, tri_pts, 255)

        return masks


    def render_piano_fullres(self):
        frame_name = self.video_ids[0]
        img_path = os.path.join(self.dat_dir, 'images', frame_name)
        mask_path = img_path.replace('images', 'masks')
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if self.edit:
            mask_edit_path = os.path.splitext(mask_path)[0] + '_edit.png'
        anno_piano = '/home/z/zh174/lhm-hand-new/hand/BV1ac411g7RJ_seq_0000.json'

        if not os.path.exists(mask_path):
            generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
            mask_path = os.path.join(self.dat_dir, 'masks', generate_name)
            mask_path = os.path.splitext(mask_path)[0] + '.png'
        if not os.path.exists(anno_piano):
            raise FileNotFoundError(f'Piano annotation not found: {anno_piano}')

        with open(anno_piano, 'r') as f:
            piano_meta = json.load(f)

        target_h = int(piano_meta.get('height', PIANO_RESOLUTION[0]))
        target_w = int(piano_meta.get('width', PIANO_RESOLUTION[1]))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)
        mask = self.resize_mask(mask, (target_w, target_h))
        if self.edit:
            mask_edit = cv2.imread(mask_edit_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

        img = self.process_image(img, mask)
        cam_info_list = []

        mano_piano_right = np.array(piano_meta.get('right', []), dtype=np.float32)
        if mano_piano_right.size == 0:
            raise ValueError('Missing right-hand pose data in piano annotation')

        frame_stem = os.path.splitext(os.path.basename(frame_name))[0]
        frame_digits = ''.join(ch for ch in frame_stem if ch.isdigit())
        frame_id = int(frame_digits) if frame_digits else 0
        start_frame = int(piano_meta.get('start_frame', 0))
        row_idx = frame_id - start_frame
        if row_idx < 0 or row_idx >= mano_piano_right.shape[0]:
            row_idx = 0

        pose_row = mano_piano_right[row_idx]
        cam_t = pose_row[:3].astype(np.float32)
        # scale = torch.tensor([1.5, 1.5, 25]).cuda()
        # cam_t = cam_t / scale.cpu().numpy()
        global_orient = pose_row[3:6].astype(np.float32)
        hand_pose = pose_row[6:51].astype(np.float32)
        dst_poses = np.concatenate([global_orient, hand_pose], axis=0)
        dst_shape = np.zeros(10, dtype=np.float32)
        generate_name = 'test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012'
        shape_path = os.path.join(self.dat_dir, 'anno', generate_name)
        shape_path = os.path.splitext(shape_path)[0] + '.pkl'
        if os.path.exists(shape_path):
            with open(shape_path, 'rb') as fi:
                _, mesh_infos, _, _ = pickle.load(fi)
            dst_skel_info = query_dst_skeleton(mesh_infos)
            dst_shape = dst_skel_info['shape']

        R = np.eye(3, dtype=np.float32)
        T = np.zeros(3, dtype=np.float32)
        T = cam_t

        K, FovY, FovX = self._build_piano_intrinsics(piano_meta, target_h, target_w)

        bkgd_mask = mask.astype('float32') / 255.0
        if self.edit:
            bound_mask = mask_edit.astype('float32') / 255.0
        else:
            bound_mask = bkgd_mask

        posed_res = self.mano_ori(
            torch.from_numpy(dst_shape)[None].float(),
            torch.from_numpy(dst_poses[:3])[None].float(),
            torch.from_numpy(dst_poses[3:])[None].float(),
            # torch.from_numpy(cam_t)[None].float(),
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
        world_vertex = world_vertex + cam_t.reshape(1, 1, 3)
        posed_verts_778 = posed_778.vertices.detach().numpy()
        posed_verts_778 = posed_verts_778 + cam_t.reshape(1, 1, 3)

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

        smpl_param = {
            'poses': dst_poses,
            'shape': dst_shape,
            'posed_verts': world_vertex.squeeze(),
            'trans': cam_t,
        }

        min_xyz = np.min(world_vertex.squeeze(), axis=0)
        max_xyz = np.max(world_vertex.squeeze(), axis=0)
        max_xyz -= 0.05
        min_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        semantic_mask = self.image_semantic_mask(
            world_vertex.reshape(-1, 3),
            K=K,
            R=R,
            T=T,
            faces=torch.from_numpy(self.mano.faces).long(),
            height=target_h,
            width=target_w,
        )
        nail_mask = (semantic_mask).astype('float32') / 255.
        nail_mask = nail_mask[..., None].repeat(3, axis=2)

        verts_cam = world_vertex.reshape(-1, 3)
        verts_cam = np.dot(R, verts_cam.T).T + T[None, ...]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = (verts_img[:, :2]).astype(np.float32)

        cam_info = CameraInfo(
            uid=id,
            R=R,
            T=T,
            K=K,
            FovY=FovY,
            FovX=FovX,
            image=img,
            nail_image=nail_img,
            verts_cam=verts_cam,
            nail_mask=nail_mask,
            bound_mask=bound_mask,
            bkgd_mask=bkgd_mask,
            image_path=img_path,
            mask_path=mask_path,
            image_name=frame_name,
            width=target_w,
            height=target_h,
            smpl_param=smpl_param,
            world_vertex=world_vertex,
            world_bound=world_bound,
            big_pose_smpl_param=self.big_pose_smpl_param,
            big_pose_world_vertex=self.canonical_verts,
            big_pose_world_bound=self.big_pose_world_bound,
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



    def __getitem_ori__(self, idx, attempt=0, max_attempts=10):

        while attempt < max_attempts:
            video_id = self.video_ids[idx]
            # video_id = 'w5xhbo7tCvH' # '00eT0ZOThXq'  # DEBUG
            # For the given video_id, list all image files in images/ and masks/, and load annotation json
            img_dir = os.path.join(self.dat_dir, 'images', video_id)
            mask_dir = os.path.join(self.dat_dir, 'masks', video_id)
            vis_dir = os.path.join(self.dat_dir, 'vis', video_id)
            ann_path = os.path.join(self.dat_dir, 'annotations', f"{video_id}.json")
            # List all image files in img_dir
            img_paths = sorted([f for f in glob.glob(os.path.join(img_dir, '*.png'))])
            mask_paths = sorted([f for f in glob.glob(os.path.join(mask_dir, '*.png'))])
            # Load annotation json
            # import json
            if not os.path.isfile(ann_path):
                print(f"Missing annotation for video {video_id}")
                idx = (idx + 1) % len(self.video_ids)
                attempt += 1
                continue
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
            # Frame id is usually the stem of the filename (without extension)
            # 只保留右手（fid以'_1'结尾）
            frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths if os.path.splitext(os.path.basename(p))[0].endswith('_1')]
            # Build a mapping from frame_id to mask path
            mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
            # Build a mapping from frame_id to annotation (assume annotation['frames'][frame_id] exists)
            ann_map = annotation.get('frames', {})
            # Sample 4 frames (randomly, or first 4 if not enough)
            if len(frame_ids) < self.num_frames:
                # chosen = frame_ids
                # print(f'Video {video_id} 只有 {len(frame_ids)} 帧, 总数少于 {self.num_frames}')
                # 尝试采样左手
                frame_ids_left = [os.path.splitext(os.path.basename(p))[0] for p in img_paths if os.path.splitext(os.path.basename(p))[0].endswith('_0')]
                if len(frame_ids) <= 1 and len(frame_ids_left) > 1:
                    frame_ids = frame_ids_left
                    # print(f'尝试采样左手，当前有 {len(frame_ids)} 帧')
                    if len(frame_ids) < self.num_frames:
                        # chosen = frame_ids
                        idx = (idx + 1) % len(self.video_ids)
                        attempt += 1
                        continue
                    else:
                        chosen = random.sample(frame_ids, self.num_frames)
                else:
                    # print(f'Video {video_id} 只有 {len(frame_ids)} frames, 总数少于 {self.num_frames}')
                    idx = (idx + 1) % len(self.video_ids)
                    attempt += 1
                    continue
            else:
                chosen = random.sample(frame_ids, self.num_frames)
            # print(f'Video {video_id} 采样 frames: {chosen}')
            cam_info_list = []
            for fid in chosen:
                img_path = os.path.join(img_dir, f'{fid}.png')
                vis_path = os.path.join(vis_dir, f'{fid}.png')
                mask_path = mask_map.get(fid, None)
                ann = ann_map.get(fid, {})
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

                img = self.process_image(img, mask)

                # from tools_utils import libcore
                # vis_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)
                # libcore.write_tensor_image('./hand/debug.jpg', torch.tensor(img), rgb2bgr=True)
                # libcore.write_tensor_image('./hand/debug_vis.jpg', torch.tensor(vis_img/255.), rgb2bgr=True)
                # height, width = img.shape[:2] if img is not None else (0, 0)
                # Camera parameters from WiLoR annotation
                # Use pred_cam_t_full as T, identity for R, default K
                w2c = np.eye(4, dtype=np.float32)
                T = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(3,1)
                # T_1 = T
                # T[0], T[1] = T_1[1], T_1[0]
                T[:2] *= 10
                # T *= 10.0
                # T[0] /= -1
                # T[0] /= -10.0
                R = np.eye(3, dtype=np.float32)
                w2c[:3, :3] = R
                w2c[:3, 3:4] = T

                #########################################################################################
                # # No intrinsics in WiLoR, use identity or estimate if possible
                # K = np.eye(3, dtype=np.float32)  # No intrinsics in WiLoR, use identity or estimate if possible
                # # FovY/FovX: estimate from K and image size if possible
                # fy = K[1,1]
                # fx = K[0,0]
                # FovY = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi if fy > 0 and height > 0 else 0.0
                # FovX = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi if fx > 0 and width > 0 else 0.0

                #########################################################################################
                raw_K = get_camera_parameters(
                    # max(512, 314),
                    # fov=60,
                    max(self.height, self.width),
                    # fov=5000, # 60
                    # fov=25, # 60   interhand 有点大
                    fov=30,  # 30
                    # fov=60, # 物体太小
                    p_x=None,
                    p_y=None,
                    device='cuda',
                )
                K = raw_K.numpy()
                # K = raw_K.squeeze().cpu().numpy()
                focal_length_y = K[1, 1]
                focal_length_x = K[0, 0]
                FovY = focal2fov(focal_length_y, self.height)  # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
                FovX = focal2fov(focal_length_x, self.width)  # 与 splatting中操作一致
                #########################################################################################

                image_name = os.path.basename(img_path)
                # Fill missing fields with None or zeros
                bkgd_mask = mask.astype('float32') / 255.0
                # bound_mask = None
                # Fill smpl_param from pred_mano_params
                pred_mano_params = ann.get('pred_mano_params', {})

                # Convert (16,3,3) rotation matrices to (16,3) axis-angle as in hand_dataset.py
                global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
                hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)
                # Concatenate to (16,3,3) if needed
                rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)
                # Convert to torch tensor for matrix_to_axis_angle
                axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats))  # (16,3)
                axis_angle = axis_angle.reshape(-1).numpy()   # (48,)

                # trans = np.zeros_like(T).reshape(-1)
                # trans[2] = T[2]  # keep z translation only
                smpl_param = {
                    'poses': axis_angle,
                    'shape': np.array(pred_mano_params.get('betas', []), dtype=np.float32),
                    # 'trans': trans,
                    # 'trans': np.array(T).reshape(-1),
                }
                # T = np.zeros_like(T)  # set T to zero, since it's included in smpl_param
                ##########################################################################3
                posed_res = self.mano(
                    torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
                    torch.from_numpy(axis_angle)[None, :3].float(),
                    torch.from_numpy(axis_angle)[None, 3:].float(),
                    # torch.from_numpy(T).float(),
                    return_verts=True)
                world_vertex = posed_res.vertices.detach().numpy()
                # verts = world_vertex.squeeze()
                # hand_center = np.mean(verts, axis=0, keepdims=True).T  # (3,1)
                # T = hand_center - T

                # bbox = self.skeleton_to_box(world_vertex.squeeze())
                # T_bbox = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(-1)
                # T_bbox[2] /= 10
                # # vis2 = draw_3d_bbox_on_image(img, bbox, K, R, T_bbox, color=(0,0,255), thickness=2,
                # #              save_path=f'./hand/bbox_vis.jpg')
                # bound_mask = project_2d_bbox(K, R, T_bbox, bbox, self.height, self.width)
                # bound_mask = bound_mask[..., None].astype('float32')
                # # print(bound_mask.shape, bound_mask.dtype, bound_mask.max())

                # # exit()
                ##############################################################################
                # world_bound = ann.get('box_center', None)  # or ann.get('bbox', None) if available
                min_xyz = np.min(world_vertex.squeeze(), axis=0)
                max_xyz = np.max(world_vertex.squeeze(), axis=0)
                max_xyz -= 0.05
                min_xyz += 0.05
                world_bound = np.stack([min_xyz, max_xyz], axis=0)
                bound_mask = get_bound_2d_mask(world_bound, raw_K.squeeze().cpu(), w2c, self.width, self.height)
                bound_mask = (np.array(bound_mask*255.0, dtype=np.byte))
                # cam_info.append(CameraInfo(
                cam_info = CameraInfo(
                    uid=fid, R=R, T=T, K=K, FovY=FovY, FovX=FovX,
                    image=img,
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

            # Convert CameraInfo objects to dicts for merge_batch
            cam_info_dicts = [vars(c) for c in cam_info_list]
            final_results = merge_batch(cam_info_dicts)

            # return cam_info_list
            return final_results



    def __getitem_nerf__(self, idx):

        video_id = self.video_ids[idx]
        # video_id = '2WP8upOky6c' # '00eT0ZOThXq'  # DEBUG
        # For the given video_id, list all image files in images/ and masks/, and load annotation json
        img_dir = os.path.join(self.dat_dir, 'images', video_id)
        mask_dir = os.path.join(self.dat_dir, 'masks', video_id)
        vis_dir = os.path.join(self.dat_dir, 'vis', video_id)
        ann_path = os.path.join(self.dat_dir, 'annotations', f"{video_id}.json")
        # List all image files in img_dir
        img_paths = sorted([f for f in glob.glob(os.path.join(img_dir, '*.png'))])
        mask_paths = sorted([f for f in glob.glob(os.path.join(mask_dir, '*.png'))])
        # Load annotation json
        # import json
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Missing annotation for video {video_id}")
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        # Frame id is usually the stem of the filename (without extension)
        # frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
        frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths if os.path.splitext(os.path.basename(p))[0].endswith('_1')]
        # Build a mapping from frame_id to mask path
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
        # Build a mapping from frame_id to annotation (assume annotation['frames'][frame_id] exists)
        ann_map = annotation.get('frames', {})
        # Sample 4 frames (randomly, or first 4 if not enough)
        if len(frame_ids) < self.num_frames:
            chosen = frame_ids
        else:
            chosen = random.sample(frame_ids, self.num_frames)
        cam_info_list = []
        for fid in chosen:
            img_path = os.path.join(img_dir, f'{fid}.png')
            vis_path = os.path.join(vis_dir, f'{fid}.png')
            mask_path = mask_map.get(fid, None)
            ann = ann_map.get(fid, {})
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)

            img = self.process_image(img, mask)

            # from tools_utils import libcore
            # vis_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)
            # libcore.write_tensor_image('./hand/debug.jpg', torch.tensor(img), rgb2bgr=True)
            # libcore.write_tensor_image('./hand/debug_vis.jpg', torch.tensor(vis_img/255.), rgb2bgr=True)
            # height, width = img.shape[:2] if img is not None else (0, 0)
            # Camera parameters from WiLoR annotation
            # Use pred_cam_t_full as T, identity for R, default K
            w2c = np.eye(4, dtype=np.float32)
            T = np.array(ann.get('pred_cam_t', [0,0,0]), dtype=np.float32).reshape(-1)
            # T_1 = T
            # T[0], T[1] = T_1[1], T_1[0]
            T[:2] *= 10
            # T *= 10.0
            # T[0] /= -1
            # T[0] /= -10.0
            R = np.eye(3, dtype=np.float32)
            w2c[:3, :3] = R
            w2c[:3, 3:4] = T.reshape(3,1)

            #########################################################################################
            # # No intrinsics in WiLoR, use identity or estimate if possible
            # K = np.eye(3, dtype=np.float32)  # No intrinsics in WiLoR, use identity or estimate if possible
            # # FovY/FovX: estimate from K and image size if possible
            # fy = K[1,1]
            # fx = K[0,0]
            # FovY = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi if fy > 0 and height > 0 else 0.0
            # FovX = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi if fx > 0 and width > 0 else 0.0

            #########################################################################################
            raw_K = get_camera_parameters(
                # max(512, 314),
                # fov=60,
                max(self.height, self.width),
                # fov=5000, # 60
                # fov=25, # 60   interhand 有点大
                fov=28,  # 30
                # fov=60, # 物体太小
                p_x=None,
                p_y=None,
                # device='cuda',
            )
            K = raw_K.squeeze().cpu().numpy()
            focal_length_y = K[1, 1]
            focal_length_x = K[0, 0]
            FovY = focal2fov(focal_length_y, self.height)  # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
            FovX = focal2fov(focal_length_x, self.width)  # 与 splatting中操作一致
            #########################################################################################

            image_name = os.path.basename(img_path)
            # Fill missing fields with None or zeros
            bkgd_mask = mask.astype('float32') / 255.0
            # bound_mask = None
            # Fill smpl_param from pred_mano_params
            pred_mano_params = ann.get('pred_mano_params', {})

            # Convert (16,3,3) rotation matrices to (16,3) axis-angle as in hand_dataset.py
            global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
            hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)
            # Concatenate to (16,3,3) if needed
            rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)
            # Convert to torch tensor for matrix_to_axis_angle
            axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats))  # (16,3)
            axis_angle = axis_angle.reshape(-1).numpy()   # (48,)

            # trans = np.zeros_like(T).reshape(-1)
            # trans[2] = T[2]  # keep z translation only
            smpl_param = {
                'poses': axis_angle,
                'shape': np.array(pred_mano_params.get('betas', []), dtype=np.float32),
                # 'trans': trans,
                # 'trans': np.array(T).reshape(-1),
            }
            # T = np.zeros_like(T)  # set T to zero, since it's included in smpl_param
            ##########################################################################3
            # interhand 中的处理
            tres = self.mano(
                torch.from_numpy(smpl_param['shape'])[None].float(),
                torch.zeros(1, 3).float(),
                torch.zeros(1, 45).float(),
                return_verts=True)
            tpose_joints = tres.joints.detach().numpy()   # 1,21,3
            tverts = tres.vertices.detach().numpy()    # 1,12337,3
            ##########################################################################3
            posed_res = self.mano(
                # torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float(),
                torch.from_numpy(smpl_param['shape'])[None].float(),
                torch.from_numpy(axis_angle)[None, :3].float(),
                torch.from_numpy(axis_angle)[None, 3:].float(),
                # torch.from_numpy(T).float(),
                return_verts=True)
            world_vertex = posed_res.vertices.detach().numpy()
            bbox = self.skeleton_to_box(world_vertex.squeeze())
            posed_joints = posed_res.joints.detach().numpy().squeeze()
            world_joints = posed_joints - T[None, :]
            joint_valid = np.ones((21,), dtype=np.float32)

            joints_uv = np.dot(raw_K.squeeze().cpu().numpy(), posed_joints.T).T  # (21, 3)
            joints_uv = joints_uv[:, :2] / joints_uv[:, 2:3]
            ## 光线处理  #######################################################################################
            rays_o, rays_d = get_rays_from_KRT(self.height, self.width, raw_K.squeeze().cpu(), R, T)
            ray_img = img.reshape(-1, 3)
            rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
            rays_d = rays_d.reshape(-1, 3)
            ray_alpha = mask.reshape(-1, 3)

            near, far, ray_mask = rays_intersect_3d_bbox(bbox, rays_o, rays_d)
            # 这里是9393，但是用interhand数据集测试的时候是19156
            rays_o = rays_o[ray_mask]
            rays_d = rays_d[ray_mask]
            ray_img = ray_img[ray_mask]
            ray_alpha = ray_alpha[ray_mask]

            near = near[:, None].astype('float32')
            far = far[:, None].astype('float32')

            rays_o, rays_d, ray_img, ray_alpha, near, far, \
            target_patches, target_alpha_patches, patch_masks, patch_div_indices = \
                self.sample_patch_rays(img=img, alpha=mask, H=self.height, W=self.width,
                                       subject_mask=mask[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(self.height, self.width),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o,
                                       rays_d=rays_d,
                                       ray_img=ray_img,
                                       ray_alpha=ray_alpha,
                                       near=near,
                                       far=far)

            batch_rays = np.stack([rays_o, rays_d], axis=0)
            #########################################################################################
            ##############################################################################
            # verts = world_vertex.squeeze()
            # hand_center = np.mean(verts, axis=0, keepdims=True).T  # (3,1)
            # T = hand_center - T
            ##############################################################################
            # world_bound = ann.get('box_center', None)  # or ann.get('bbox', None) if available
            min_xyz = np.min(world_vertex.squeeze(), axis=0)
            max_xyz = np.max(world_vertex.squeeze(), axis=0)
            max_xyz -= 0.05
            min_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            bound_mask = get_bound_2d_mask(world_bound, raw_K.squeeze().cpu(), w2c, self.width, self.height)
            bound_mask = np.array(bound_mask*255.0, dtype=np.byte)[..., None].repeat(3, axis=2)
            results = {
                'target_alpha_img': mask,
                'cam_T': T,
                'cam_R': R,
                'cam_K': raw_K.squeeze().cpu().numpy(),
                'img_width': self.width,
                'img_height': self.height,
                'ray_mask': ray_mask, # (65536,)
                'rays': batch_rays,   # (2, 256, 3)
                'near': near,   # 512,1,
                'far': far,     # 512,1,   # 但是用interhand数据集测试的时候是 1024，1
                'bg_color': np.array([0,0,0], dtype=np.float32),
                'patch_div_indices': patch_div_indices,
                'patch_masks': patch_masks,
                'target_patches': target_patches,
                'target_alpha_patches': target_alpha_patches[..., 0],
                'dst_posevec':axis_angle[3:],
                'dst_shape': smpl_param['shape'],
                'dst_global_orient': smpl_param['poses'].reshape(-1)[:3],
                'dst_cam_joints': world_joints,
                'dst_valid_joints': joint_valid,
                'img_path': img_path,
                'joints_uv': joints_uv,
            }

        # return cam_info_list
        return results






class HandDataset_new(Dataset):
    def __init__(self, split='train', frm_list=None, num_frames=4):
        self.split = split
        self.dat_dir = '/scratch/groups/su004-neuralnet/zh174/hand-data/our_data'
        self.num_frames = num_frames
        self.height, self.width = 256, 256

        # 预加载所有视频ID和元数据
        self._preload_metadata()

        # 预加载MANO模型和相关数据
        self._preload_mano_data()

        # 预加载相机参数
        self._preload_camera_params()

        print(f'[Our Hand Dataset] Found {len(self.video_ids)} videos')

    def _preload_metadata(self):
        """预加载所有视频元数据"""
        images_root = os.path.join(self.dat_dir, 'images')
        self.video_ids = [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
        self.video_ids.sort()

        # 预加载所有annotation文件
        self.annotations = {}
        ann_dir = os.path.join(self.dat_dir, 'annotations')
        for vid in self.video_ids:
            ann_path = os.path.join(ann_dir, f"{vid}.json")
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    self.annotations[vid] = json.load(f)

        # 预构建帧索引
        self._build_frame_index()

    def _build_frame_index(self):
        """构建帧索引，避免在__getitem__中频繁进行文件操作"""
        self.video_frame_map = {}

        for vid in self.video_ids:
            img_dir = os.path.join(self.dat_dir, 'images', vid)
            img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))

            # 只处理右手帧
            frame_ids = [os.path.splitext(os.path.basename(p))[0]
                        for p in img_paths
                        if os.path.splitext(os.path.basename(p))[0].endswith('_1')]

            # 如果右手帧不足，考虑左手帧
            if len(frame_ids) < self.num_frames:
                frame_ids_left = [os.path.splitext(os.path.basename(p))[0]
                                 for p in img_paths
                                 if os.path.splitext(os.path.basename(p))[0].endswith('_0')]
                frame_ids = frame_ids_left

            self.video_frame_map[vid] = frame_ids

    def _preload_mano_data(self):
        """预加载MANO模型和相关数据"""
        # 您的MANO初始化代码
        self.mano = tools_utils.model.smplx.create(**cfg.smpl_cfg)

        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights

        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype == 'left':
            self.mano.shapedirs[:, 0, :] *= -1

        # 预计算大姿势数据
        with torch.no_grad():
            mano_res = self.mano(
                torch.zeros(1, 10).float(),  # betas
                torch.zeros(1, 3).float(),  # global_orient
                torch.zeros(1, 45).float(), return_verts=True)

        self.faces = mano_res.faces_tensor
        self.big_pose_xyz = mano_res.vertices.detach().numpy().squeeze()

        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)
        big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        big_pose_smpl_param['faces'] = self.faces.detach()
        self.big_pose_smpl_param = big_pose_smpl_param

        big_pose_min_xyz = np.min(self.big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(self.big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        self.big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    def _preload_camera_params(self):
        """预加载相机参数"""
        self.raw_K = get_camera_parameters(
            max(self.height, self.width),
            fov=28,
            p_x=None,
            p_y=None,
            device='cpu',  # 使用CPU而不是CUDA
        ).squeeze().cpu().numpy()

        self.focal_length_y = self.raw_K[1, 1]
        self.focal_length_x = self.raw_K[0, 0]
        self.FovY = focal2fov(self.focal_length_y, self.height)
        self.FovX = focal2fov(self.focal_length_x, self.width)

    def __len__(self):
        return len(self.video_ids)

    def _precompute_mano_vertices(self, pred_mano_params):
        """预计算MANO顶点"""
        betas = torch.from_numpy(np.array(pred_mano_params.get('betas', []), dtype=np.float32))[None].float()
        axis_angle = self._get_axis_angle(pred_mano_params)

        with torch.no_grad():
            posed_res = self.mano(
                betas,
                axis_angle[None, :3].float(),
                axis_angle[None, 3:].float(),
                return_verts=True)

        return posed_res.vertices.detach().numpy()

    def _get_axis_angle(self, pred_mano_params):
        """从MANO参数获取轴角表示"""
        global_orient_mat = np.array(pred_mano_params.get('global_orient', []), dtype=np.float32)
        hand_pose_mat = np.array(pred_mano_params.get('hand_pose', []), dtype=np.float32)
        rot_mats = np.concatenate([global_orient_mat, hand_pose_mat], axis=0)
        axis_angle = matrix_to_axis_angle(torch.from_numpy(rot_mats))
        return axis_angle.reshape(-1).numpy()

    def _get_world_bound(self, world_vertex):
        """计算世界边界"""
        min_xyz = np.min(world_vertex.squeeze(), axis=0)
        max_xyz = np.max(world_vertex.squeeze(), axis=0)
        min_xyz += 0.05
        max_xyz -= 0.05
        return np.stack([min_xyz, max_xyz], axis=0)

    def process_image(self, img, mask, bg_color=(0, 0, 0)):
        """处理图像和掩码"""
        if len(mask.shape) == 2:
            mask = mask[..., None]
        mask_f = mask.astype(np.float32) / 255.
        bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        img = img.astype(np.float32) / 255.
        seg_img = mask_f * img + (1.0 - mask_f) * bg
        return seg_img

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        # 获取预计算的帧ID
        frame_ids = self.video_frame_map.get(video_id, [])
        if len(frame_ids) < self.num_frames:
            # 如果帧不足，返回空数据或跳过
            return self._get_empty_data()

        # 随机选择帧
        chosen = random.sample(frame_ids, self.num_frames)

        # 获取预加载的annotation
        annotation = self.annotations.get(video_id, {})
        ann_map = annotation.get('frames', {})

        cam_info_list = []
        for fid in chosen:
            # 构建路径
            img_dir = os.path.join(self.dat_dir, 'images', video_id)
            mask_dir = os.path.join(self.dat_dir, 'masks', video_id)
            img_path = os.path.join(img_dir, f'{fid}.png')
            mask_path = os.path.join(mask_dir, f'{fid}.png')

            # 加载图像和掩码
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # 处理图像
            img = self.process_image(img, mask)

            # 获取annotation
            ann = ann_map.get(fid, {})

            # 处理相机参数
            T = np.array(ann.get('pred_cam_t', [0, 0, 0]), dtype=np.float32).reshape(3, 1)
            T[:2] *= 10
            R = np.eye(3, dtype=np.float32)

            # 处理MANO参数
            pred_mano_params = ann.get('pred_mano_params', {})
            axis_angle = self._get_axis_angle(pred_mano_params)

            # 预计算顶点和边界
            world_vertex = self._precompute_mano_vertices(pred_mano_params)
            world_bound = self._get_world_bound(world_vertex)

            # 创建SMPL参数
            smpl_param = {
                'poses': axis_angle,
                'shape': np.array(pred_mano_params.get('betas', []), dtype=np.float32),
            }

            # 创建CameraInfo
            cam_info = CameraInfo(
                uid=fid, R=R, T=T, K=self.raw_K,
                FovY=self.FovY, FovX=self.FovX,
                image=img,
                bkgd_mask=mask,
                image_path=img_path,
                mask_path=mask_path,
                image_name=os.path.basename(img_path),
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

        # 如果帧不足，返回空数据
        if len(cam_info_list) < self.num_frames:
            return self._get_empty_data()

        # 合并批次
        cam_info_dicts = [vars(c) for c in cam_info_list]
        final_results = merge_batch(cam_info_dicts)

        return final_results

    def _get_empty_data(self):
        """返回空数据"""
        # 根据您的merge_batch函数返回适当结构的空数据
        # 这里需要根据您的实际需求实现
        return {}



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
    # xyz = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    # xyz = np.dot(xyz, K.T)
    # xy = xyz[:, :2] / xyz[:, 2:]

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
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
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

def group_indices_by_subject(dataset):
    subject_to_indices = defaultdict(list)
    for idx, subject_id in enumerate(dataset.subjects):
        subject_to_indices[subject_id].append(idx)
    return list(subject_to_indices.values())  # list of index lists per subject





# class SubjectBatchSampler(Sampler):
#     def __init__(self, grouped_indices, shuffle=True):
#         self.grouped_indices = grouped_indices
#         self.shuffle = shuffle
#
#     def __iter__(self):
#         if self.shuffle:
#             random.shuffle(self.grouped_indices)
#         for group in self.grouped_indices:
#             yield group  # one group = one batch
#
#     def __len__(self):
#         return len(self.grouped_indices)


class SubjectBatchSampler(Sampler):
    def __init__(self, subjects, batch_size, max_batches_per_epoch=1000):
        self.subject_to_indices = defaultdict(list)
        for idx, subject in enumerate(subjects):
            self.subject_to_indices[subject].append(idx)

        self.subjects = list(self.subject_to_indices.keys())
        self.batch_size = batch_size
        self.max_batches_per_epoch = max_batches_per_epoch

    def __iter__(self):
        for _ in range(self.max_batches_per_epoch):
            # 随机选择一个 subject
            subject = random.choice(self.subjects)
            indices = self.subject_to_indices[subject]

            # 如果该 subject 样本数量不足一个 batch，就跳过
            if len(indices) < self.batch_size:
                continue

            # 从该 subject 中随机选择一个 batch
            batch_indices = random.sample(indices, self.batch_size)
            yield batch_indices

    def __len__(self):
        return self.max_batches_per_epoch



    # def __iter__(self):
    #     random.shuffle(self.subjects)
    #     for subject in self.subjects:
    #         indices = self.subject_to_indices[subject]
    #         # 将同一 subject 下的数据分批输出
    #         for i in range(0, len(indices), self.batch_size):
    #             yield indices[i:i + self.batch_size]
    #
    # def __len__(self):
    #     # 粗略估计：所有 subject 的 batch 数之和
    #     return sum(
    #         (len(indices) + self.batch_size - 1) // self.batch_size
    #         for indices in self.subject_to_indices.values()
    #     )


def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device("cuda")):
    """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
        K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
    else:
        K[0, -1], K[1, -1] = img_size // 2, img_size // 2

    # Add batch dimension
    # K = K.unsqueeze(0).to(device)
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


def save_obj(vertices, faces, save_path):
    with open(save_path, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            # OBJ 格式是 1-based index
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def npy_to_obj(npy_path, obj_path):
    data = np.load(npy_path, allow_pickle=True).item()

    vertices = data.get("vertices", None)
    faces = data.get("faces", None)

    if vertices is None or faces is None:
        raise ValueError("The .npy file must contain 'vertices' and 'faces'.")

    save_obj(vertices, faces, obj_path)
    print(f"Saved OBJ file to: {obj_path}")




def query_dst_skeleton(mesh_infos):
    return {
        'poses': mesh_infos['poses'].astype('float32'),
        'shape': mesh_infos['shape'].astype('float32'),
        'dst_tpose_joints': \
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

    # Handle GPFS symlink: /gpfs/home/... -> /home/...
    base_str = str(base_path)
    if '/gpfs/' in base_str:
        base_str = base_str.replace('/gpfs', '', 1)
        base_path = Path(base_str)

    candidates = [
        base_path.parents[1] / 'mano_uv',  # lhm-hand-new/mano_uv
        base_path.parents[2] / 'mano_uv',  # fallback to zh174/mano_uv
        base_path.parents[3] / 'mano_uv',  # fallback to z/mano_uv
        base_path.parents[3] / 'GuassianHand' / 'mano_uv',
    ]
    for candidate in candidates:
        change_file = candidate / 'change' / 'change_r.npy'
        obj_file = candidate / 'original mano template' / 'hand.obj'
        if change_file.exists() and obj_file.exists():
            print(f"[DEBUG] Found mano_uv at: {candidate}")
            return candidate
    raise FileNotFoundError('Unable to locate mano_uv assets. Please place the directory inside lhm-hand-new or alongside GuassianHand.')
