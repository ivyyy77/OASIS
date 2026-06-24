
import os
import sys

# import scene.dataset_readers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import pickle

import numpy as np
import cv2
import torch
from PIL import Image
# import point_cloud_utils as pcu
# from hand_semantic import *
import glob
import re

from tools_utils.model.handavatar.core.utils.image_util import load_image
from tools_utils.model.handavatar.core.utils.math_util import convert
from tools_utils.model.handavatar.core.utils.hand_util import MANOHand, INTERHAND2MANO
from tools_utils.model.handavatar.core.utils.augm_util import process_bbox, augmentation, trans_point2d

import json
from pycocotools.coco import COCO
from tools_utils.model.handavatar.configs import cfg
import tools_utils.model.smplx
from tools_utils.model.smplx.manohd.subdivide import sub_mano

from typing import NamedTuple, Optional
from tools_utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from tools_utils.camera_utils import loadCam, loadCam_aug_bs
# import pytorch3d.structures.meshes as py3d_meshes
from pytorch3d.structures import Meshes


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    # extrinsics: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    nail_image: Optional[np.array]
    nail_mask: Optional[np.array]
    verts_cam: Optional[np.array]
    bkgd_mask: np.array
    bound_mask: np.array
    image_path: str
    mask_path: str
    image_name: str
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array

def frameset_collate_fn(batches):
    return batches

def make_dataloader(frameset, shuffle=False, batch_size= 4 , num_workers=0, prefetch_factor=None, persistent_workers=None):
    dataloader = torch.utils.data.DataLoader(frameset, shuffle=shuffle,
                                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                                             batch_size=batch_size,
                                             persistent_workers=persistent_workers,
                                             collate_fn=frameset_collate_fn)
    return dataloader

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def getNerfppNorm_bs(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    W2C = getWorld2View2(cam_info.R, cam_info.T)
    C2W = np.linalg.inv(W2C)
    cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def create_dataset(data_type='train', subject=None):
    dataset_name = cfg[data_type].dataset # 加载配置文件的 ‘train’下的 dataset

    args = DatasetArgs.get(dataset_name)

    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
    args['data_type'] = data_type
    if data_type == 'progress':
        if cfg.progress.get('skip', -1) > 0:
            args['skip'] = cfg.progress.skip
            args['maxframes'] = -1
        else:
            total_train_imgs = 20000  # _get_total_train_imgs(os.path.join(args['dataset_path'],
            # f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images',
            # cfg.subject))
            args['skip'] = 16
            args['maxframes'] = 16

    if data_type in ['freeview', 'tpose', 'freepose', 'infer']:
        args['skip'] = cfg[data_type].get('skip', 1)
    if data_type in ['infer']:
        args['subject'] = subject
    args['use_mean_pose'] = False
    args['use_var_pose'] = False
    dataset = Dataset(**args)
    return dataset

class DatasetArgs(object):
    dataset_attrs = {}

    if cfg.category in ['handavatar', ] and cfg.task == 'interhand':
        dataset_attrs.update({
            "interhand_train": {
                "dataset_path": f'/mnt/data/InterHand/',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "interhand_test": {
                "dataset_path": f'/mnt/data/InterHand/',
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'interhand',
            },
        })

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()


class Dataset(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(
            self, dataset_path, keyfilter=None, maxframes=-1, bgcolor=None,
            ray_shoot_mode='image', skip=1, subject=None, **kwargs):

        print('[Dataset Path]', dataset_path)
        # self.img_res = [512,334]
        self.img_res = [256, 256]

        # MANO
        self.mano = tools_utils.model.smplx.create(**cfg.smpl_cfg)

        # if cfg.smpl_cfg['manohd'] > 0:
        #     print('MANO-HD in Model')
        #     smpl_body, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        #     lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        #     smpl_body.lbs_weights = lbs_weights
        #     self.mano = smpl_body


        if cfg.smpl_cfg['manohd'] > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights

        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype == 'left':
            self.mano.shapedirs[:, 0, :] *= -1


        # labels = []
        # for w in lbs_weights:
        #     labels.append(MANO_JOINT_TO_PART[np.argmax(w)])
        #
        # colors = get_colors_from_labels(labels)
        #
        # ply = generate_ply(self.mano.v_template, colors, labels)
        # save_path = save_path = './assets/manohd_semantic.ply'
        # save_plyfile(ply, save_path)

        # if cfg.smpl_cfg['manohd'] > 0:
        #     print('MANO-HD in Model')
        #     self.mano = sub_mano(mano, cfg.smpl_cfg['manohd'])


        # annotation
        self.phase = kwargs.get('data_type', 'train')
        if subject is None:
            if self.phase == 'train':
                subject = cfg.subject
            else:
                subject = cfg[kwargs['data_type']].subject
        if isinstance(subject, list):
            subject = subject[0]

        # ###########################################################################################################################################
        # subject = 'train/prior_learning_data'
        # if 'prior_learning' in subject:
        #     action = ['/'+ item.split('/')[-1] for item in glob.glob('/mnt/data/InterHand/InterHand2.6M_5fps_batch1/images/train/Capture0/00*')]
        #     all_dir_name = [
        #         'train/Capture0',
        #         'train/Capture1',
        #         'train/Capture2',
        #         'train/Capture3',
        #         'train/Capture5',
        #         'train/Capture6',
        #         'train/Capture7',
        #         'train/Capture8',
        #         'train/Capture9',
        #         'train/Capture10',
        #         'train/Capture11',
        #         'train/Capture12',
        #         'train/Capture13',
        #         'train/Capture14',
        #         'train/Capture15',
        #         'train/Capture16',
        #         'train/Capture20',
        #         'train/Capture22',
        #         'train/Capture23',
        #         'train/Capture24',
        #         'train/Capture25',
        #         ]
        #     self.subjects = [int(name.split('/')[-1].replace('Capture', '')) for name in all_dir_name]
        #     # list [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 22, 23, 24, 25]
        #     all_dir_name = [x.split('/')[-1] + act for x in all_dir_name for act in action]
        #     # select train and test
        #     test_split = ['0000_neutral_relaxed', '0009_thumbtucknormal', '0019_alligator_closed', '0029_indextip', '0039_fingerspreadrigid', '0048_index_point', '0058_middlefinger']
        # else:
        #     all_dir_name = ['/'.join(subject.split('/')[1:])]
        #     test_split = [subject.split('/')[-1]]
        #
        # print('[INFO] All using sequence:', all_dir_name)
        # print('[INFO] Test split:', test_split)
        #
        # #####################################################################################################################################################

        self.image_dir = os.path.join(dataset_path, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')
        if 'prior_learning' in subject:
            anno_name = os.path.join(self.image_dir.replace('images', 'preprocess'), subject, 'anno_cam.pkl')
        else:
            anno_name = os.path.join(self.image_dir.replace('images', 'anno_preprocess_gauhand_manohd'), subject, 'anno_cam.pkl')
        anno_name_handAvatar = os.path.join(self.image_dir.replace('images', 'preprocess'), subject, 'anno_cam.pkl')
        print('Load annotation', anno_name)
        if not os.path.exists(anno_name):
            print('Preprocessing ...')
            self.anno_preprocess(dataset_path, subject, anno_name, self.phase)
        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)
        if 'prior_learning' not in subject:
            with open(anno_name_handAvatar, 'rb') as f:
                _, _, _, framelist_handAvatar = pickle.load(f)


        mano_res = self.mano(
            torch.zeros(1, 10).float(), # betas
            torch.zeros(1, 3).float(),  # global_orient 确定！
            torch.zeros(1, 45).float(), # hand_pose
            return_verts=True)
        self.canonical_bbox = mano_res.joints[0].numpy() # 提取其中的关节位置 和 手部的顶点位置
        self.canonical_verts = mano_res.vertices.numpy()
        self.canonical_faces = mano_res.faces_tensor.numpy()
        self.canonical_bbox = self.skeleton_to_bbox(self.canonical_verts) # 用关节位置进一步构造 bbox
        #self.mesh_py3d = py3d_meshes.Meshes(mano_res.vertices[None,...].float(), torch.from_numpy(mano_res.face[None,...].astype(int)))
        # self.mesh_py3d = py3d_meshes.Meshes(torch.from_numpy(self.canonical_verts).float(),
        #                                     torch.from_numpy(self.canonical_faces[None, ...].astype(int)))
        # self.cano_mesh_info = {
        #     'mesh_verts':self.mesh_py3d.verts_packed(),
        #     'mesh_norms':self.mesh_py3d.verts_normals_packed(),
        #     'mesh_faces':self.mesh_py3d.faces_packed()
        # }


        # post process
        # self.framelist = framelist_handAvatar[::skip]
        # ###########################################################################################################################
        # new_framelist = []
        # for i in range(len(self.framelist)):
        #     action = self.framelist[i].split('/')[2]
        #     if self.phase == 'train':
        #         if action not in test_split:
        #             new_framelist.append(self.framelist[i])
        #     else:
        #         if action in test_split:
        #             new_framelist.append(self.framelist[i])
        # self.framelist = new_framelist[:]
        #
        # # post process
        # self.framelist = self.framelist[::skip]
        #
        # data_type = kwargs['data_type']
        # try:
        #     exclude = cfg[data_type].get('exclude_idx', None)
        # except:
        #     print(f'[WARNING] Invalid data_type: {data_type}, set exclude_idx: None')
        #     exclude = None
        #
        # if isinstance(exclude, list):
        #     sel_idx = list(range(len(self.framelist)))
        #     self.framelist = [self.framelist[i] for i in sel_idx if i not in exclude]
        # if maxframes > 0:
        #     self.framelist = self.framelist[:maxframes]
        # ########################################################################################################################################
        self.cameras = {k : self.cameras[k] for k in self.framelist}
        self.bbox = {k : self.bbox[k] for k in self.framelist}
        self.mesh_infos = {k : self.mesh_infos[k] for k in self.framelist}

        exclude = cfg[kwargs['data_type']].get('exclude_idx', None)
        if isinstance(exclude, list):
            sel_idx = list(range(len(self.framelist)))
            self.framelist = [self.framelist[i] for i in sel_idx if i not in exclude]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]

        pattern = re.compile(r'Capture(\d+)')

        self.ids = [int(pattern.search(p).group(1)) for p in self.framelist]
        self.framelist = list(zip(self.framelist, self.ids))

        print(f' -- Total Frames: {self.get_total_frames()}')
        self.keyfilter = keyfilter
        self.bgcolor = [0,0,0]
        self.ray_shoot_mode = ray_shoot_mode
        self.aug = True
        self.resolution_arg = -1
        self.resolution_scales = [1.0]
        poses = [item['poses'] for item in self.mesh_infos.values()]
        self.poses = torch.tensor(poses).float()
        shapes = [item['shape'] for item in self.mesh_infos.values()]
        self.shapes = torch.tensor(shapes).float()
        if kwargs['use_mean_pose']:
            self.mean_pose = torch.mean(poses, 0, keepdim=True)
        else:
            self.mean_pose = torch.zeros_like(self.poses)
        if kwargs['use_var_pose']:
            self.var_pose = torch.var(self.poses, 0, keepdim=True)
        else:
            self.var_pose = None
        # self.mean_shape = torch.zeros_like(self.shapes[[0], :])
        self.mean_shape = torch.zeros_like(self.shapes)

    @torch.no_grad()
    def anno_preprocess(self, dataset_path, subject, anno_name, data_type, with_mask=True):

        th_hands_mean_right = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                                        0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                                        0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                                        -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                                        0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                                        -0.0266, -0.0529, 0.5356, -0.0460, 0.2774])
        th_hands_mean_left = th_hands_mean_right.copy().reshape(-1, 3)
        th_hands_mean_left[:, 1:] *= -1


        phase = subject.split('/')[0]
        dir_name = '/'.join(subject.split('/')[1:])

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
            if dir_name + '/' not in img['file_name']:
                continue
            if i % 5000 == 0:
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
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
            if data_type != 'infer':
                if bbox[0] < 10 or bbox[1] < 10 or max(bbox[2], bbox[3]) < 80 or bbox[0] + bbox[2] > img_width - 10 or \
                        bbox[1] + bbox[3] > img_height - 10:
                    # print(f'{i}, Discard {image_name}, bbox is too biased/small: {bbox.tolist()}')
                    continue

                # frame
                img_path = os.path.join(self.image_dir, f'{phase}/{image_name}')

                # if not os.path.exists(img_path):
                #     print(f"File not found: {img_path}. Skipping...")
                #     return None

                img = cv2.imread(img_path)
                if img.max() < 20:
                    # print(f'{i}, Discard {image_name}, RGB is too dark: {img.max()}')
                    continue
                if np.allclose(img[..., 0], img[..., 1], atol=1) or np.allclose(img[..., 2], img[..., 1],
                                                                                atol=1) or np.allclose(img[..., 0],
                                                                                                       img[..., 2],
                                                                                                       atol=1):
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

                mask_bool = mask[..., 0] == 255
                sel_img = img[mask_bool].mean(axis=-1)
                if sel_img.max() < 20:
                    print(i, frame_name, 'sel_img is too dark')
                    continue

            bbox = process_bbox(bbox, img_width, img_height)
            self.bbox[f'{phase}/{image_name}'] = bbox
            self.framelist.append(f'{phase}/{image_name}')

            # camera
            T, R = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            # T[1:] = -T[1:]
            # R[1:,:] = -R[1:,:] # Opencv需要将第 2，3行（ y，z轴）变为负数
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            K = np.eye(3)
            K[[0, 1], [0, 1]] = focal
            K[[0, 1], [2, 2]] = princpt

            poses = np.array(mano_param['pose'])
            betas = np.array(mano_param['shape'])
            # Rh = poses[:3].copy()
            # Rh_mat = np.dot( R, convert(Rh, 'axangle', 'rotmat') )
            # Rh = convert(Rh_mat, 'rotmat',  'axangle')
            poses[3:] += (th_hands_mean_left, th_hands_mean_right)[self.handtype == 'right']

            posed_res = self.mano(
                torch.from_numpy(betas)[None].float(),
                torch.from_numpy(poses)[None, :3].float(),
                torch.from_numpy(poses)[None, 3:].float(),
                return_verts=True)
            vertices = posed_res.vertices
            faces = posed_res.faces_tensor
            joints = posed_res.joints[0].numpy()
            # verts = posed_res.vertices[0].numpy()
            verts = posed_res.vertices.numpy()
            center = posed_res.center[0].numpy

            # mesh_py3d = Meshes(vertices.float(), torch.from_numpy(faces[None, ...].detach().numpy()))
            # frame_mesh = mesh_py3d.update_padded(vertices)
            # mesh_norms = frame_mesh.verts_normals_packed()
            # fid, bc = pcu.sample_mesh_poisson_disk(vertices.detach().numpy().squeeze(), faces.squeeze().numpy(), num_samples=-1, radius=0.01)
            # rand_normals = pcu.interpolate_barycentric_coords(faces, fid, bc, mesh_norms.detach().numpy())

            ###orthographic projection
            joint_world = np.array(joints_interhand[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32) / 1000 * cfg.smpl_cfg.scale
            joint_cam = np.dot(R, joint_world.T).T - np.dot(R, T) / 1000 * cfg.smpl_cfg.scale
            joint_cam = joint_cam[:21][INTERHAND2MANO] if self.handtype=='right' else joint_cam[21:][INTERHAND2MANO]
            T = joint_cam[0] - np.dot(R,joints[0])


            self.cameras[f'{phase}/{image_name}'] = {
                'intrinsics': K,
                'extrinsics': [R,T],
                'img_width' :img_width,
                'img_height':img_height
            }

            # mesh
            self.mesh_infos[f'{phase}/{image_name}'] = {
                'poses': poses.reshape(-1),
                'shape': betas,
                'vertices': verts,
                'joints': joints,
                'center': center,
                'Rh': np.zeros(3),
                'Th': joint_cam[0] - joints[0]
            }
        os.makedirs(os.path.dirname(anno_name), exist_ok=True)
        with open(anno_name, 'wb') as f:
            pickle.dump([self.cameras, self.mesh_infos, self.bbox, self.framelist], f)


    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'shape': self.mesh_infos[frame_name]['shape'].astype('float32'),
            'vertices': self.mesh_infos[frame_name]['vertices'].astype('float32'),
            'joints': self.mesh_infos[frame_name]['joints'].astype('float32'),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),

            # 'mesh_norms': self.mesh_infos[frame_name]['sampled_norms'].astype('float32'),
        }

    def load_image(self, frame_name, bg_color, use_mask=True):
        imagepath = os.path.join(self.image_dir, frame_name)
        orig_img = np.array(load_image(imagepath))
        orig_img = np.array(load_image('./data/12023.jpg'))

        if use_mask:
            maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
            alpha_mask = np.array(load_image(maskpath))
            alpha_mask = np.array(load_image('./data/12023.png'))
        else:
            alpha_mask = np.ones_like(orig_img) * 255

        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        # debug
        # if alpha_mask.any() == None or orig_img.any() == None or bg_color.any() == None:
        #     bg_color = [1,1,1]
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


    def mask_from_mano(self, frame_name,filtering=True):
        save_name = os.path.join(self.image_dir, frame_name).replace('images', 'masks_removeblack').replace('.jpg',
                                                                                                            '.png')
        dst_skel_info = self.query_dst_skeleton(frame_name)
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        R,T = self.cameras[frame_name]['extrinsics']

        poses = dst_skel_info['poses']
        betas = dst_skel_info['shape']

        posed_res = self.mano(
            torch.from_numpy(betas)[None].float(),
            torch.from_numpy(poses[:3])[None].float(),
            torch.from_numpy(poses[3:])[None].float(),
            return_verts=True)
        joints = posed_res.joints[0].detach().numpy()
        verts = posed_res.vertices[0].detach().numpy()
        verts = np.dot(R, verts.T).T + T # 将顶点位置从世界坐标系转换到相机坐标系
        verts_img = np.dot(K, verts.T).T # 将顶点位置从相机坐标系投影到图像平面。K内参，用于将三维点投影到图像平面
        verts_img[:, :2] = np.round(verts_img[:, :2] / verts_img[:, 2:3]) # verts_img[:, :2]是图像平面的 (x, y) 坐标
        verts_img = verts_img.astype(np.int32)

        img = cv2.imread(os.path.join(self.image_dir, frame_name))

        mask = np.zeros_like(img)
        for f in self.mano.faces:
            triangle = np.array([[verts_img[f[0]][0], verts_img[f[0]][1]], [verts_img[f[1]][0], verts_img[f[1]][1]],
                                 [verts_img[f[2]][0], verts_img[f[2]][1]]])
            cv2.fillConvexPoly(mask, triangle, (255, 255, 255))

        if filtering:

            mask_bool = mask[..., 0] == 255
            sel_img = img[mask_bool].mean(axis=-1)

            sel_img = np.bitwise_and(sel_img > 10, sel_img < 200)
            mask_bool[mask_bool] = sel_img.astype('int32')
            mask = mask * mask_bool[..., None]

            contours, _ = cv2.findContours(mask[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours)
            contours.sort(key=cnt_area, reverse=True)
            poly = contours[0].transpose(1, 0, 2).astype(np.int32)
            poly_mask = np.zeros_like(img)
            poly_mask = cv2.fillPoly(poly_mask, poly, (1, 1, 1))
            mask = mask * poly_mask
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            cv2.imwrite(save_name, mask)

    def calculate_elevation_azimuth(self, R):
        """
        计算相机的 elevation 和 azimuth。

        参数:
        - R: 3x3 的旋转矩阵，从世界坐标系到相机坐标系的旋转

        返回:
        - elevation: 仰角（单位为度数）
        - azimuth: 方位角（单位为度数）
        """
        # 视线方向为 R 的第三列向量
        r13, r23, r33 = R[0, 2], R[1, 2], R[2, 2]

        # 计算 elevation 和 azimuth（结果转换为度数）
        elevation = np.degrees(np.arcsin(r23))  # 仰角
        azimuth = np.degrees(np.arctan2(r13, r33))  # 方位角

        return elevation, azimuth


##############################################################################################
    def readHandCameras(self):
       # 输入好像只需要一个 self就行?
        cam_info = []
        mano_res = self.mano(
            torch.zeros(1, 10).float(),  # betas
            torch.zeros(1, 3).float(),  # global_orient 确定！
            torch.zeros(1, 45).float(),  # hand_pose
            return_verts=True)
        vertices = mano_res.vertices
        faces = mano_res.faces_tensor
        big_pose_xyz = mano_res.vertices.detach().numpy().squeeze()
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1,10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1,48)).astype(np.float32)
        big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        big_pose_smpl_param['faces'] = faces.detach()
        # big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose())+big_pose_smpl_param['Th']).astype(np.float32)
        # big_pose_smpl_param['poses'] = torch.zeros((1,48))
        mesh_py3d = Meshes(vertices.float(), torch.from_numpy(faces[None, ...].detach().numpy()))


        big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.05
        big_pose_max_xyz += 0.05
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)


        for idx in range(0, len(self.framelist)):
            # if frame_name == self.framelist[idx]:
            # 此处的 framelist是从anno_name_handAvatar路径中加载的，对应 HandAvatar数据处理130行，是全部数据集的list
            # 应该把原本全部的图片文件命名为 44，部分的还是 4，然后再查找？
            # framelist应该还是原来的 4中的，但是bbox已经变成 44的了？

            frame_name = self.framelist[idx]
            bbox = self.bbox[frame_name]
            cam_extrinsics, cam_intrinsics, width, height = self.cameras[frame_name]['extrinsics'], self.cameras[frame_name]['intrinsics'], \
                                                        self.cameras[frame_name]['img_width'], self.cameras[frame_name]['img_height']
            sys.stdout.flush()

            key = frame_name
            uid = key
            R = cam_extrinsics[0]
            T = cam_extrinsics[1].reshape((3, 1))
            # w2c = np.stack(R,T)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T
            R = np.transpose(w2c[:3,:3])

            focal_length_x = cam_intrinsics[0, 0]
            focal_length_y = cam_intrinsics[1, 1]
            FovY = focal2fov(focal_length_y, height) # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
            FovX = focal2fov(focal_length_x, width) # 与 splatting中操作一致

            dst_skel_info = self.query_dst_skeleton(frame_name)
            poses = dst_skel_info['poses']
            dst_skel_info.update({'Rh':poses[:3]})
            # poses[:3] = 0
            betas = dst_skel_info['shape']
            Rh = dst_skel_info['Rh']
            dst_skel_info['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            dst_skel_info.update({'Th': torch.zeros(1,3)})
            posed_verts = dst_skel_info['vertices']

            posed_verts_tensor = torch.from_numpy(posed_verts)
            frame_mesh = mesh_py3d.update_padded(posed_verts_tensor)
            mesh_norms = frame_mesh.verts_normals_packed()

            dst_skel_info.update({'mesh_norms': mesh_norms})

            # if np.any(betas != 0) and idx == 0:
            #     cano_with_shape_offset = self.mano(
            #         torch.from_numpy(betas)[None].float(),  # betas
            #         torch.zeros(1, 3).float(),  # global_orient 确定！
            #         torch.zeros(1, 45).float(),  # hand_pose
            #         return_verts=True)
            #     big_pose_xyz = cano_with_shape_offset.vertices.detach().numpy().squeeze()


            # mano_posed = self.mano(
            #     torch.from_numpy(betas)[None].float(), # betas
            #     torch.from_numpy(poses[:3])[None].float(),  # global_orient 确定！
            #     torch.from_numpy(poses[3:])[None].float(), # hand_pose
            #     return_verts=True)
            # posed_verts = mano_posed.vertices.detach().numpy().squeeze() # 相当于593行，xyz指的是变形的位置

            min_xyz = np.min(posed_verts.squeeze(), axis=0)
            max_xyz = np.max(posed_verts.squeeze(), axis=0)
            max_xyz -= 0.05
            min_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz],axis=0)
            bound_mask = get_bound_2d_mask(world_bound, cam_intrinsics, w2c, width, height)
            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            image_path = os.path.join(self.image_dir, key)
            image_name = os.path.basename(image_path).split(".")[0]
            image_cam = image_path.split("/")[-2]
            image_name = image_cam + '_' +image_name
            image = Image.open(image_path)

            # Check if the image path exists
            # if not os.path.exists(image_path):
            #     print(f"File not found: {image_path}. Skipping...")
            #     return None  # or continue, depending on your loop logic
            # with Image.open(image_path) as image:
            #     # 处理 mask 文件对象
            #     return image

            mask_path = image_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                # with Image.open(mask_path) as mask:
                #     # 处理 mask 文件对象
                #     pass
            else:
                self.mask_from_mano(key)
                mask = Image.open(mask_path)

            cam_info.append(CameraInfo(uid=uid, R=R, T=T, K=cam_intrinsics, FovY=FovY, FovX=FovX,
                                       image=image, bkgd_mask=mask, bound_mask=bound_mask,
                                       image_path=image_path, mask_path=mask_path, image_name=image_name, width=width, height=height,
                                       smpl_param=dst_skel_info, world_vertex=posed_verts, world_bound=bbox,
                                       big_pose_world_vertex=big_pose_xyz, big_pose_world_bound=big_pose_world_bound,
                                       big_pose_smpl_param=big_pose_smpl_param))

        # cam_info = loadCam(self.resolution_arg,uid, cam_info, resolution_scale)

        # if len(cam_infos)>10:
        #     break

        return cam_info

###################################################################################################
    def readHandCameras_aug(self):
        cam_info = []
        mano_res = self.mano(
            torch.zeros(1, 10).float(),  # betas
            torch.zeros(1, 3).float(),  # global_orient 确定！
            torch.zeros(1, 45).float(),  # hand_pose
            return_verts=True)
        vertices = mano_res.vertices
        faces = mano_res.faces_tensor
        big_pose_xyz = mano_res.vertices.detach().numpy().squeeze()  # ndarray: 778,3
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1,10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1,48)).astype(np.float32)
        big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        big_pose_smpl_param['faces'] = faces.detach()  # 1538,3
        big_pose_smpl_param['vertex_index'] = np.arange(big_pose_xyz.shape[0])
        big_pose_smpl_param['finger_category'] = mano_res.finger_category
        # big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose())+big_pose_smpl_param['Th']).astype(np.float32)
        # big_pose_smpl_param['poses'] = torch.zeros((1,48))
        mesh_py3d = Meshes(vertices.float(), torch.from_numpy(faces[None, ...].detach().numpy()))

        # verts_tensor = torch.from_numpy(vertices)
        frame_mesh = mesh_py3d.update_padded(vertices)
        mesh_norms = frame_mesh.verts_normals_packed()
        big_pose_smpl_param['normal'] = mesh_norms

        # fid, bc = pcu.sample_mesh_poisson_disk(vertices.detach().numpy().squeeze(), faces.squeeze().numpy(), num_samples=-1, radius=0.01)
        # rand_normals = pcu.interpolate_barycentric_coords(faces, fid, bc, mesh_norms.detach().numpy())
        # mesh_norms = rand_normals
        # big_pose_smpl_param['normal'] = mesh_norms

        big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)


        for idx in range(0, len(self.framelist)):

            frame_name = self.framelist[idx]
            bbox = self.bbox[frame_name]
            cam_extrinsics, cam_intrinsics = self.cameras[frame_name]['extrinsics'], self.cameras[frame_name]['intrinsics']
            sys.stdout.flush()

            key = frame_name  # str: 'test/Capture/ROM04_RT_Occlusion/cam4000262/image17746.jpg'
            uid = key

            image_path = os.path.join(self.image_dir, key)
            image_name = os.path.basename(image_path).split(".")[0]
            image_cam = image_path.split("/")[-2]
            image_name = image_cam + '_' + image_name

            # mask_path = image_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')
            mask_path = image_path.replace('images', 'masks_new').replace('.jpg', '.png')

            if self.bgcolor is None:
                bgcolor = (np.random.rand(3) * 255.).astype('float32')
            else:
                bgcolor = np.array(self.bgcolor, dtype = 'float32')

            img, mask = self.load_image(key, bgcolor, use_mask=(True, True)[self.phase == 'infer'])
            img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox,
                                                                                            'eval',
                                                                                            exclude_flip=True,
                                                                                            input_img_shape=(256, 256),
                                                                                            mask=mask,
                                                                                            base_scale=1.3,
                                                                                            scale_factor=0.2,
                                                                                            rot_factor=0,
                                                                                            shift_wh=[bbox[2], bbox[3]],
                                                                                            gaussian_std=3,
                                                                                            bordervalue=bgcolor.tolist())

            # cv2.imwrite('test.png', img.astype('uint8'))
            # cv2.waitKey(0)

            img = (img / 255.).astype('float32')
            img = img[:, :, [2, 1, 0]]

            mask = mask.astype('float32')
            height, width = img.shape[0:2]

            R_ = cam_extrinsics[0]
            T = cam_extrinsics[1].reshape((3, 1))
            w2c = np.eye(4)
            w2c[:3,:3] = R_
            w2c[:3,3:4] = T
            R = np.transpose(w2c[:3, :3])
            elevation, azimuth = self.calculate_elevation_azimuth(R_)


            K = self.cameras[key]['intrinsics'][:3, :3].copy()
            if self.aug:
                K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
                K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2] * aug_param[1])

            focal_length_x = cam_intrinsics[0, 0]
            focal_length_y = cam_intrinsics[1, 1]
            FovY = focal2fov(focal_length_y, height) # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
            FovX = focal2fov(focal_length_x, width) # 与 splatting中操作一致

            dst_skel_info = self.query_dst_skeleton(frame_name)
            poses = dst_skel_info['poses']
            dst_skel_info.update({'Rh':poses[:3]})
            # poses[:3] = 0
            betas = dst_skel_info['shape']
            Rh = dst_skel_info['Rh']
            dst_skel_info['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            dst_skel_info.update({'Th': torch.zeros(1,3)})
            posed_verts = dst_skel_info['vertices']

            posed_verts_tensor = torch.from_numpy(posed_verts)
            frame_mesh = mesh_py3d.update_padded(posed_verts_tensor)
            mesh_norms = frame_mesh.verts_normals_packed()  # 778,3

            dst_skel_info.update({'mesh_norms': mesh_norms})

            # target_radius = np.linalg.norm(posed_verts.max(0) - posed_verts.min(0)) * 0.0025 #* 0.015  # 0.02629
            # fid, bc = pcu.sample_mesh_poisson_disk(posed_verts.squeeze(), faces.squeeze().numpy(), num_samples=-1, radius=0.015)
            # print('11111111111111111111')
            # rand_normals = pcu.interpolate_barycentric_coords(faces, fid, bc, mesh_norms)
            # print('22222222222222222')

            # dst_skel_info.update({'sampled_norms': rand_normals})

            # mano_posed = self.mano(
            #     torch.from_numpy(betas)[None].float(), # betas
            #     torch.from_numpy(poses[:3])[None].float(),  # global_orient 确定！
            #     torch.from_numpy(poses[3:])[None].float(), # hand_pose
            #     return_verts=True)
            # posed_verts = mano_posed.vertices.detach().numpy().squeeze() # 相当于593行，xyz指的是变形的位置

            # if np.any(betas != 0) and idx == 0:
            #     cano_with_shape_offset = self.mano(
            #         torch.from_numpy(betas)[None].float(),  # betas
            #         torch.zeros(1, 3).float(),  # global_orient 确定！
            #         torch.zeros(1, 45).float(),  # hand_pose
            #         return_verts=True)
            #     big_pose_xyz = cano_with_shape_offset.vertices.detach().numpy().squeeze()

            min_xyz = np.min(posed_verts.squeeze(), axis=0)
            max_xyz = np.max(posed_verts.squeeze(), axis=0)
            max_xyz -= 0.05
            min_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz],axis=0)
            bound_mask = get_bound_2d_mask(world_bound, cam_intrinsics, w2c, width, height)
            # bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))
            bound_mask = np.array(bound_mask * 255.0, dtype=np.byte)

            cam_info.append(CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, extrinsics=w2c, # 为啥新加这个外参啊？？干啥用了？下次记下来行不
                                       image=img, bkgd_mask=mask, bound_mask=bound_mask,
                                       image_path=image_path, mask_path=mask_path, image_name=image_name, width=width, height=height,
                                       smpl_param=dst_skel_info, world_vertex=posed_verts, world_bound=bbox,
                                       big_pose_world_vertex=big_pose_xyz, big_pose_world_bound=big_pose_world_bound,
                                       big_pose_smpl_param=big_pose_smpl_param))


        return cam_info


###################################################################################################

    def readHandCameras_aug_bs(self, idx):

        mano_res = self.mano(
            torch.zeros(1, 10).float(),  # betas
            torch.zeros(1, 3).float(),  # global_orient 确定！
            torch.zeros(1, 45).float(), return_verts=True)
        vertices = mano_res.vertices
        faces = mano_res.faces_tensor
        big_pose_xyz = mano_res.vertices.detach().numpy().squeeze()  # ndarray: 778,3
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shape'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 48)).astype(np.float32)
        big_pose_smpl_param['center'] = mano_res.center.detach().numpy().squeeze()
        big_pose_smpl_param['faces'] = faces.detach()  # 1538,3
        # big_pose_smpl_param['vertex_index'] = np.arange(big_pose_xyz.shape[0])
        # big_pose_smpl_param['finger_category'] = mano_res.finger_category

        # mesh_py3d = Meshes(vertices.float(), torch.from_numpy(faces[None, ...].detach().numpy()))
        #
        # frame_mesh = mesh_py3d.update_padded(vertices)
        # mesh_norms = frame_mesh.verts_normals_packed()
        # big_pose_smpl_param['normal'] = mesh_norms

        big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.3
        big_pose_max_xyz += 0.3
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        frame_name = self.framelist[idx][0]
        bbox = self.bbox[frame_name]
        cam_extrinsics, cam_intrinsics = self.cameras[frame_name]['extrinsics'], self.cameras[frame_name]['intrinsics']
        # sys.stdout.flush()

        key = frame_name  # str: 'test/Capture/ROM04_RT_Occlusion/cam4000262/image17746.jpg'
        uid = key
        id = self.framelist[idx][1]
        image_path = os.path.join(self.image_dir, key)
        image_name = os.path.basename(image_path).split(".")[0]
        image_cam = image_path.split("/")[-2]
        image_name = image_cam + '_' + image_name

        mask_path = image_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, mask = self.load_image(key, bgcolor, use_mask=(True, True)[self.phase == 'infer'])
        # img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox,
        #                                                                                 'eval',
        #                                                                                 exclude_flip=True,
        #                                                                                 input_img_shape=(256, 256),
        #                                                                                 mask=mask,
        #                                                                                 base_scale=1.3,
        #                                                                                 scale_factor=0.2,
        #                                                                                 rot_factor=0,
        #                                                                                 shift_wh=[bbox[2], bbox[3]],
        #                                                                                 gaussian_std=3,
        #                                                                                 bordervalue=bgcolor.tolist())

        img = (img / 255.).astype('float32')
        img = img[:, :, [2, 1, 0]]

        mask = mask.astype('float32')
        height, width = img.shape[0:2]

        R = cam_extrinsics[0]
        T = cam_extrinsics[1].reshape((3, 1))
        # R = cam_extrinsics[:3, :3]
        # T = cam_extrinsics[:3, 3].reshape((3, 1))
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3:4] = T
        R = np.transpose(w2c[:3, :3])

        K = self.cameras[key]['intrinsics'][:3, :3].copy()
        # if self.aug:
        #     K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        #     K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2] * aug_param[1])

        focal_length_x = cam_intrinsics[0, 0]
        focal_length_y = cam_intrinsics[1, 1]

        #############################################################################
        R = np.eye(3)
        # R[:, 0] *= -1
        # R[:, 2] *= -1
        T = np.array([[.08284344, -0.05660701,  3.572253]], dtype=np.float32).reshape(1,3)
        T = np.array([[-0.13709098, -0.06628575, 3.9246328]], dtype=np.float32).reshape(1, 3)    # ohta
        raw_K = get_camera_parameters(
            # max(512, 314),
            # fov=60,
            max(height, width),
            fov=30, # 60
            p_x=None,
            p_y=None,
            device='cuda:0',
        )
        K = raw_K.squeeze().cpu().numpy()
        focal_length_y = K[1, 1]
        focal_length_x = K[0, 0]

        #############################################################################

        FovY = focal2fov(focal_length_y, height)  # 函数的操作为：2*math.atan( height /( 2*focal_length_y ))
        FovX = focal2fov(focal_length_x, width)  # 与 splatting中操作一致

        dst_skel_info = self.query_dst_skeleton(frame_name)
        poses = dst_skel_info['poses']
        dst_skel_info.update({'Rh': poses[:3]})

        betas = dst_skel_info['shape']
        Rh = dst_skel_info['Rh']
        dst_skel_info['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
        dst_skel_info.update({'Th': torch.zeros(1, 3)})
        posed_verts = dst_skel_info['vertices']

        # posed_verts_tensor = torch.from_numpy(posed_verts)
        # frame_mesh = mesh_py3d.update_padded(posed_verts_tensor)
        # mesh_norms = frame_mesh.verts_normals_packed()  # 778,3

        # dst_skel_info.update({'mesh_norms': mesh_norms})

        ##############################################################################################
        # from smplx.lbs import rotation_matrix_to_angle_axis
        from pytorch3d.transforms import matrix_to_axis_angle
        rot_mats = torch.tensor([[[[-0.4013,  0.0466, -0.9148],
                                  [ 0.8680, -0.2997, -0.3960],
                                  [-0.2926, -0.9529,  0.0798]],
                                  [[9.9263e-01, -1.1986e-01, 1.7655e-02],
                                   [1.2094e-01, 9.8897e-01, -8.5564e-02],
                                   [-7.2045e-03, 8.7069e-02, 9.9618e-01]],
                                  [[9.9716e-01, 6.1495e-02, 4.3499e-02],
                                   [-5.8041e-02, 9.9537e-01, -7.6652e-02],
                                   [-4.8011e-02, 7.3909e-02, 9.9611e-01]],
                                  [[9.9810e-01, 5.6488e-03, -6.1294e-02],
                                   [-5.7095e-03, 9.9998e-01, -8.1566e-04],
                                   [6.1288e-02, 1.1641e-03, 9.9812e-01]],
                                  [[9.9789e-01, -2.2381e-02, -6.0992e-02],
                                   [1.5757e-02, 9.9413e-01, -1.0699e-01],
                                   [6.3029e-02, 1.0581e-01, 9.9239e-01]],
                                  [[9.9765e-01, 3.2454e-02, -6.0268e-02],
                                   [-2.3603e-02, 9.8956e-01, 1.4216e-01],
                                   [6.4253e-02, -1.4040e-01, 9.8801e-01]],
                                  [[9.8866e-01, -1.4969e-01, 1.2317e-02],
                                   [1.4948e-01, 9.8862e-01, 1.6619e-02],
                                   [-1.4664e-02, -1.4590e-02, 9.9979e-01]],
                                  [[9.8836e-01, 1.2120e-01, -9.1901e-02],
                                   [-1.3593e-01, 9.7496e-01, -1.7602e-01],
                                   [6.8266e-02, 1.8646e-01, 9.8009e-01]],
                                  [[9.9482e-01, -2.7790e-03, -1.0165e-01],
                                   [-3.1356e-03, 9.9831e-01, -5.7979e-02],
                                   [1.0164e-01, 5.7997e-02, 9.9313e-01]],
                                  [[9.9951e-01, 1.3183e-02, 2.8522e-02],
                                   [-1.9208e-02, 9.7472e-01, 2.2260e-01],
                                   [-2.4866e-02, -2.2303e-01, 9.7449e-01]],
                                  [[9.8468e-01, -1.1820e-02, -1.7397e-01],
                                   [-2.2306e-02, 9.8096e-01, -1.9290e-01],
                                   [1.7294e-01, 1.9383e-01, 9.6567e-01]],
                                  [[9.9790e-01, 6.4758e-02, 4.3122e-04],
                                   [-6.3760e-02, 9.8131e-01, 1.8156e-01],
                                   [1.1334e-02, -1.8120e-01, 9.8338e-01]],
                                  [[9.8079e-01, -1.9208e-01, 3.3903e-02],
                                   [1.8509e-01, 9.7138e-01, 1.4886e-01],
                                   [-6.1526e-02, -1.3972e-01, 9.8828e-01]],
                                  [[9.4531e-01, 3.2618e-01, 1.5721e-03],
                                   [-3.1466e-01, 9.1316e-01, -2.5911e-01],
                                   [-8.5952e-02, 2.4444e-01, 9.6585e-01]],
                                  [[9.8685e-01, -1.5215e-01, -5.4516e-02],
                                   [1.5832e-01, 8.4217e-01, 5.1544e-01],
                                   [-3.2512e-02, -5.1730e-01, 8.5519e-01]],
                                  [[9.8956e-01, -5.2252e-02, -1.3434e-01],
                                   [-1.9667e-03, 9.2701e-01, -3.7504e-01],
                                   [1.4413e-01, 3.7139e-01, 9.1722e-01]]
                                  ]]).squeeze(0)

        # rot_mats =  torch.tensor([[
        #                          [[-9.5801e-01,  1.3578e-01,  2.5256e-01],
        #                           [ 2.5013e-01, -3.4944e-02,  9.6758e-01],
        #                           [ 1.4020e-01,  9.9012e-01, -4.8555e-04]],
        #                          [[ 0.8615, -0.4717, -0.1879],
        #                           [ 0.4377,  0.8775, -0.1959],
        #                           [ 0.2573,  0.0865,  0.9624]], [[ 0.6393, -0.7603,  0.1149],
        #                           [ 0.7680,  0.6386, -0.0483],
        #                           [-0.0367,  0.1191,  0.9922]], [[ 0.9744, -0.2226, -0.0306],
        #                           [ 0.2244,  0.9710,  0.0824],
        #                           [ 0.0113, -0.0871,  0.9961]], [[ 0.8154, -0.5702, -0.0999],
        #                           [ 0.5776,  0.7901,  0.2052],
        #                           [-0.0381, -0.2250,  0.9736]], [[ 0.6381, -0.7646, -0.0908],
        #                           [ 0.7690,  0.6271,  0.1238],
        #                           [-0.0377, -0.1488,  0.9881]], [[ 0.9150, -0.3955,  0.0798],
        #                           [ 0.3898,  0.9176,  0.0776],
        #                           [-0.1039, -0.0399,  0.9938]], [[ 0.7131, -0.6388,  0.2888],
        #                           [ 0.2911,  0.6445,  0.7070],
        #                           [-0.6378, -0.4201,  0.6455]], [[ 0.9419, -0.3337,  0.0374],
        #                           [ 0.3357,  0.9345, -0.1182],
        #                           [ 0.0045,  0.1239,  0.9923]], [[ 0.9613,  0.2756, -0.0029],
        #                           [-0.2017,  0.7106,  0.6741],
        #                           [ 0.1879, -0.6474,  0.7387]], [[ 0.7696, -0.6262,  0.1248],
        #                           [ 0.5705,  0.7622,  0.3061],
        #                           [-0.2868, -0.1644,  0.9438]], [[ 0.9054, -0.4243, -0.0135],
        #                           [ 0.4081,  0.8611,  0.3034],
        #                           [-0.1171, -0.2802,  0.9528]], [[ 0.9099, -0.4112,  0.0541],
        #                           [ 0.3737,  0.8694,  0.3232],
        #                           [-0.1799, -0.2739,  0.9448]], [[ 0.8277, -0.4318, -0.3583],
        #                           [-0.1692,  0.4168, -0.8931],
        #                           [ 0.5350,  0.7999,  0.2719]], [[ 0.9760, -0.2153, -0.0326],
        #                           [ 0.1880,  0.7578,  0.6248],
        #                           [-0.1098, -0.6160,  0.7801]], [[ 0.9938,  0.0763,  0.0810],
        #                           [-0.0194,  0.8360, -0.5483],
        #                           [-0.1096,  0.5434,  0.8323]]]]).squeeze(0)
        # axis_angle = rotation_matrix_to_angle_axis(rot_mats)
        axis_angle = matrix_to_axis_angle(rot_mats)
        axis_angle = axis_angle.reshape(-1).numpy()
        # a = apply_extra_rotation_to_global_orient(torch.tensor(axis_angle.reshape(1, 48)), rot_axis=[1, 0, 0],
        #                                           rot_angle_deg=180)
        # dst_skel_info['poses'] = a.squeeze().cpu().numpy()
        dst_skel_info['poses'] = axis_angle
        dst_skel_info['shape'] = np.array([-0.3097, -0.0901, -0.1211, -0.0245,  0.0185,  0.1052,  0.0570,  0.0015, -0.0388,  0.0508])
        ##########################################################################################################################

        min_xyz = np.min(posed_verts.squeeze(), axis=0)
        max_xyz = np.max(posed_verts.squeeze(), axis=0)
        max_xyz -= 0.05
        min_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)
        bound_mask = get_bound_2d_mask(world_bound, cam_intrinsics, w2c, width, height)
        bound_mask = np.array(bound_mask * 255.0, dtype=np.byte)

        # ########################################## VGGT
        # from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        # from vggt.models.vggt import VGGT
        # from vggt.utils.load_fn import load_and_preprocess_images
        #
        # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        # # model = VGGT.from_pretrained("./pretrained_models/huggingface/model.pt").to('cuda:1')
        # model = VGGT()
        # state_dict = torch.load("./pretrained_models/huggingface/model.pt", map_location='cpu')
        # model.load_state_dict(state_dict)
        # model = model.to('cuda:1')
        # # image_names = ["/home/ubuntu/hanhan/LHM/interhand.jpg"]
        # # images = load_and_preprocess_images(image_names).to('cuda:1')
        # image_names = os.path.join('/mnt/data/InterHand/InterHand2.6M_5fps_batch1/images/', uid)
        # images = load_and_preprocess_images([image_names]).to('cuda:1')
        #
        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(dtype=dtype):
        #         images = images[None]  # add batch dimension
        #         aggregated_tokens_list, ps_idx = model.aggregator(images)
        #
        #     # Predict Cameras
        #     pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        #     # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        #     # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])    shape: 518,518
        #     # extrinsic, intrinsic, FovX, FovY = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        #     # extrinsic = closed_form_inverse_se3(extrinsic.squeeze(0))[:, :3, :]
        #     extrinsic, intrinsic, FovX, FovY = pose_encoding_to_extri_intri(pose_enc, [512, 334])
        #     K = intrinsic.squeeze().cpu().numpy()
        #     # FovX = 2 * np.arctan2(518/2, K[0, 0])
        #     # FovY = 2 * np.arctan2(518/2, K[1, 1])
        #     if self.aug:
        #         K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        #         K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2] * aug_param[1])
        #     R = extrinsic.squeeze()[:3, :3].cpu().numpy()
        #     T = extrinsic.squeeze()[:3, 3].unsqueeze(-1).cpu().numpy()
        #     # R[1:, ] = -R[1:, ]
        #     # T[1:] = -T[1:]
        # # ########################################## VGGT


        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, # extrinsics=w2c,
                              image=img, bkgd_mask=mask, bound_mask=bound_mask, image_path=image_path,
                              mask_path=mask_path, image_name=image_name, width=width, height=height,
                              smpl_param=dst_skel_info, world_vertex=posed_verts, world_bound=bbox,
                              big_pose_world_vertex=big_pose_xyz, big_pose_world_bound=big_pose_world_bound,
                              big_pose_smpl_param=big_pose_smpl_param)

        cam_info = loadCam_aug_bs(id, cam_info)

        return cam_info



        # image_path = os.path.join(images_folder, key)
        # image_name = os.path.basename(image_path).split(".")[0]
        # image_cam = image_path.split("/")[-2]
        # image_name = image_cam + '_' +image_name
        # mask_path = image_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')
        # if not os.path.exists(mask_path):
        #     self.mask_from_mano(key)
        # img, mask = self.load_image(key, bgcolor, use_mask=(True, True)[self.phase == 'infer'])
        # bbox = self.bbox[key]
        # (img, img2bb_trans, bb2img_trans,
        #  aug_param, do_flip, scale, mask) = augmentation(img, bbox, 'eval', exclude_flip=True, input_img_shape=(256, 256),
        #                                                  mask=mask, base_scale=1.3, scale_factor=0.2, rot_factor=0,
        #                                                  shift_wh=[bbox[2], bbox[3]], gaussian_std=3, bordervalue=bgcolor.tolist())
        #
        # img = (img / 255.).astype('float32')
        # mask = mask.astype('float32')
        # height, width = img.shape[0:2]
        # cam_extrinsics, cam_intrinsics = cameras['extrinsics'], cameras['intrinsics']
        #
        # sys.stdout.flush()
        #
        # uid = key
        # R = cam_extrinsics[0]
        # T = cam_extrinsics[1]
        # K = self.cameras[key]['intrinsics'][:3, :3].copy()
        # if self.aug:
        #     K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        #     K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])
        #
        # focal_length_x = K[0, 0]
        # focal_length_y = K[1, 1]
        # FovY = focal2fov(focal_length_y, height)
        # FovX = focal2fov(focal_length_x, width)
        #
        # cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=img, mask=mask,
        #                       image_path=image_path, image_name=image_name, width=width, height=height)
        # cam_info = loadCam_aug(uid, cam_info)
        #
        # # if len(cam_infos)>10:
        # #     break
        #
        # return cam_info



    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        # frame_name = self.framelist[idx]
        # cam_info = {}

        # if self.bgcolor is None:  # 上面定义为 0 了
        #     bgcolor = (np.random.rand(3) * 255.).astype('float32')
        # else:
        #     bgcolor = np.array(self.bgcolor, dtype='float32')

        # for resolution_scale in self.resolution_scales:
        #     if not self.aug:
        #         cam_info[resolution_scale] = self.readHandCameras(cameras=self.cameras[frame_name], images_folder=self.image_dir, key =frame_name,resolution_scale=resolution_scale)
        #     else:
        #         cam_info[resolution_scale] = self.readHandCameras_aug_bs(idx)



        cam_info = self.readHandCameras_aug_bs(idx)
        results = cam_info

        results = {
            'FoVx': cam_info.FoVx,
            'FoVy': cam_info.FoVy,
            'K': cam_info.K,
            'R': cam_info.R,
            'T': cam_info.T,
            'bkgd_mask': cam_info.bkgd_mask,
            'bound_mask': cam_info.bound_mask,
            'camera_center': cam_info.camera_center,
            # 'extrinsics': cam_info.extrinsic,
            'full_proj_transform': cam_info.full_proj_transform,
            'width': cam_info.width,
            'height': cam_info.height,
            'original_image': cam_info.original_image,
            'projection_matrix': cam_info.projection_matrix,
            'smpl_param': cam_info.smpl_param,
            'world_view_transform': cam_info.world_view_transform,
            'zfar': cam_info.zfar,
            'znear': cam_info.znear,
            'uid': cam_info.uid,
            'trans': cam_info.trans,
            'scale': cam_info.scale,
            'image_name': cam_info.image_name,
            'colmap_id': cam_info.colmap_id,
            'big_pose_world_vertex': cam_info.big_pose_world_vertex,
            'big_pose_world_bound': cam_info.big_pose_world_bound,
            'big_pose_smpl_param': cam_info.big_pose_smpl_param,
        }

        # nerf_normalization = getNerfppNorm_bs(results)
        # results.update({'gt': nerf_normalization["radius"]})
        return results

        # results = {
        #     'uid': frame_name
        # }
        #
        # bbox = self.bbox[frame_name]
        # W = self.cameras[frame_name]['img_width']
        # H = self.cameras[frame_name]['img_height']
        #
        # dst_skel_info = self.query_dst_skeleton(frame_name)
        # dst_poses = dst_skel_info['poses']
        # dst_shape = dst_skel_info['shape']
        # dst_vertices = dst_skel_info['vertices']
        #
        # assert frame_name in self.cameras
        # K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        # E = self.cameras[frame_name]['extrinsics']
        #
        #
        # R = E[0]
        # T = E[1]
        #
        #
        # results.update({
        #     'cam_K': K,
        #     'cam_R': R,
        #     'cam_T': T,
        # })
        #
        # results.update({
        #     'img_width': W,
        #     'img_height': H,
        #     'bgcolor': bgcolor,
        #     'bbox':bbox,
        # })
        #
        # results.update({
        #     'dst_posevec': dst_poses[3:],
        #     'dst_shape': dst_shape,
        #     'dst_global_orient': dst_poses[:3],
        #     'camera_info': cam_info
        # })
        #
        # mano_mesh = self.mesh_py3d.update_padded(torch.from_numpy(dst_vertices).float())
        # results['mesh_info'] = {
        #     'mesh_verts': mano_mesh.verts_packed(),
        #     'mesh_norms': mano_mesh.verts_normals_packed(),
        #     'mesh_faces': mano_mesh.faces_packed(),
        # }
        #
        #
        # intrinsics = cam_info[resolution_scale].projection_matrix.T
        # extrinsics = cam_info[resolution_scale].world_view_transform
        # ###R.T for pytorch3D
        #
        # ###coordinate transfer for pytorch3D
        # if self.img_res[0]!=self.img_res[1]:
        #     extrinsics = torch.matmul(extrinsics, torch.diag(torch.tensor([-1, -1, 1,1]).float())).float()
        # intrinsics[3, 2] = -1
        # if self.img_res[0]!=cam_info[resolution_scale].original_image.shape[1]:
        #     a=1
        #
        #
        # sample = {
        #     "idx": torch.LongTensor([idx]),
        #     "img_name": frame_name,
        #     "intrinsics": intrinsics,
        #     "shape": dst_shape,
        #     "pose": dst_poses[3:],
        #     "global_orient":  dst_poses[:3],
        #     "cam_pose": extrinsics,
        #     "cam_info": cam_info[resolution_scale],
        #     "cam_name": '/'.join(frame_name.split('/')[-2:]),
        #     "img_index": '_'.join(frame_name.split('/')[-2:])[:-4],
        #     "mask": cam_info[resolution_scale].mask
        #     }
        #
        # ground_truth = {
        #     'rgb': cam_info[resolution_scale].original_image.flatten(1).T,
        #     'object_mask': cam_info[resolution_scale].mask.flatten(),
        # }
        #
        # return idx, sample, ground_truth, results

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



def merge_batch(batch_list):
    merged = {}
    keys = batch_list[0].keys()

    for key in keys:
        values = [b[key] for b in batch_list]

        # torch.Tensor
        if isinstance(values[0], torch.Tensor):
            try:
                merged[key] = torch.stack(values)
            except RuntimeError:
                # 非统一shape，例如 vertices
                merged[key] = values

        # np.ndarray
        elif isinstance(values[0], np.ndarray):
            try:
                merged[key] = np.stack(values)
            except ValueError:
                merged[key] = values

        # float / int / str
        elif isinstance(values[0], (float, int, str)):
            merged[key] = values

        # nested dict
        elif isinstance(values[0], dict):
            merged[key] = merge_batch(values)

        # fallback
        else:
            merged[key] = values

    return merged

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


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
    K = K.unsqueeze(0).to(device)
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
