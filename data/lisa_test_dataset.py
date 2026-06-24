"""
Lisa Experiment Test Dataset

This dataset loads preprocessed InterHand images for the Lisa experiment.
It provides cam400053 and cam400018 views for evaluating the reconstructed 3D hand.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from tools_utils.model.ohta.configs import cfg
from tools_utils.model import smplx as smplx_
import tools_utils.model.smplx as smplx
from tools_utils.model.smplx.manohd.subdivide import sub_mano
from data.utils.camera_util import apply_global_tfm_to_camera
from data.hand_dataset import CameraInfo, merge_batch
from tools_utils.camera_utils import loadCam_aug_bs
from data.debug import focal2fov


class LisaTestDataset(Dataset):
    """
    Dataset for Lisa experiment testing.

    Loads cam400053_image4734 and cam400018_image4734 for evaluation.
    These are novel views of the same hand pose as the training image (cam400012_image4734).
    """

    def __init__(
        self,
        data_root='example_data/interhand2.6m',
        test_views=None,
        bgcolor=[0.0, 0.0, 0.0],
        **kwargs
    ):
        """
        Args:
            data_root: Root directory containing images/, masks/, anno/
            test_views: List of test view names (without extension), e.g., ['cam400053_image4734', 'cam400018_image4734']
            bgcolor: Background color
        """
        self.data_root = data_root
        self.bgcolor = np.array(bgcolor, dtype='float32')
        self.height, self.width = 256, 256

        # Default test views for Lisa experiment
        if test_views is None:
            test_views = ['cam400053_image4734', 'cam400018_image4734']
        self.test_views = test_views

        # Initialize MANO model
        print('[LisaTestDataset] Initializing MANO model...')
        self.mano = smplx.create(**cfg.smpl_cfg)
        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype == 'left':
            self.mano.shapedirs[:, 0, :] *= -1

        # MANO-HD
        if cfg.smpl_cfg.get('manohd', 0) > 0:
            self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights

        # Load canonical pose
        mano_res = self.mano(
            torch.zeros(1, 10).float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
            return_verts=True
        )
        self.canonical_verts = mano_res.vertices[0].detach().numpy()
        self.canonical_bbox = self._skeleton_to_bbox(self.canonical_verts)

        # Big pose parameters
        big_pose_min_xyz = np.min(self.canonical_verts, axis=0) - 0.3
        big_pose_max_xyz = np.max(self.canonical_verts, axis=0) + 0.3
        self.big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        self.big_pose_smpl_param = {
            'R': np.eye(3).astype(np.float32),
            'Th': np.zeros((1, 3)).astype(np.float32),
            'shape': np.zeros((1, 10)).astype(np.float32),
            'poses': np.zeros((1, 48)).astype(np.float32),
        }

        # Load all annotations
        self.annotations = {}
        for view_name in self.test_views:
            anno_path = os.path.join(data_root, 'anno', f'{view_name}.pkl')
            if os.path.exists(anno_path):
                with open(anno_path, 'rb') as f:
                    self.annotations[view_name] = pickle.load(f)
                print(f'[LisaTestDataset] Loaded annotation: {view_name}')
            else:
                print(f'[LisaTestDataset] Warning: Missing annotation: {anno_path}')

        print(f'[LisaTestDataset] Total test views: {len(self.test_views)}')

    @staticmethod
    def _skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0)
        max_xyz = np.max(skeleton, axis=0)
        return np.stack([min_xyz, max_xyz], axis=0)

    def __len__(self):
        return len(self.test_views)

    def get_total_frames(self):
        return len(self.test_views)

    def __getitem__(self, idx):
        view_name = self.test_views[idx]

        # Load image
        img_path = os.path.join(self.data_root, 'images', f'{view_name}.jpg')
        mask_path = os.path.join(self.data_root, 'masks', f'{view_name}.png')

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Cannot load image: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.).astype('float32')

        # Load mask
        mask = cv2.imread(mask_path)
        if mask is not None:
            alpha = (mask[:, :, 0] / 255.).astype('float32')
        else:
            alpha = np.ones((self.height, self.width), dtype='float32')

        # Load annotation
        anno = self.annotations.get(view_name)
        if anno is None:
            raise ValueError(f'No annotation for view: {view_name}')

        camera_info, mesh_info, bbox_dict, _ = anno

        # Extract camera parameters
        K = camera_info['intrinsics'][:3, :3].copy()

        # Extract mesh parameters
        poses = mesh_info['poses'].reshape(-1)
        shape = mesh_info['shape'].reshape(-1)
        Rh = mesh_info['Rh']
        Th = mesh_info['Th']

        # Compute extrinsics
        E = apply_global_tfm_to_camera(
            E=np.eye(4),
            Rh=Rh,
            Th=Th
        )
        R = E[:3, :3]
        T = E[:3, 3]

        # Compute posed mesh
        posed_res = self.mano(
            torch.from_numpy(shape)[None].float(),
            torch.from_numpy(poses[:3])[None].float(),
            torch.from_numpy(poses[3:])[None].float(),
            return_verts=True
        )
        world_vertex = posed_res.vertices.detach().numpy()
        posed_vertices = world_vertex.reshape(-1, 3)


        cano_res = self.mano(
            torch.from_numpy(shape)[None].float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
            return_verts=True
        )
        self.canonical_verts = cano_res.vertices[0].detach().numpy()

        # Compute world bounds
        min_xyz = np.min(posed_vertices, axis=0) + 0.05
        max_xyz = np.max(posed_vertices, axis=0) - 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        # Compute vertices in camera space
        verts_cam = np.dot(R, posed_vertices.T).T + T[None, :]
        verts_img = np.dot(K, verts_cam.T).T
        verts_img[:, :2] /= verts_img[:, 2:3]
        nail_img = np.round(verts_img[:, :2]).astype(np.int32)

        # FOV
        focal_length_x = K[1, 1]
        focal_length_y = K[0, 0]
        FovX = focal2fov(focal_length_x, self.height)
        FovY = focal2fov(focal_length_y, self.width)

        # SMPL parameters
        smpl_param = {
            'poses': poses,
            'shape': shape,
            'posed_verts': world_vertex.squeeze(),
        }

        # Create CameraInfo
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            K=K,
            FovY=FovY,
            FovX=FovX,
            image=img,
            nail_image=nail_img,
            nail_mask=alpha[..., None].astype('float32'),
            bound_mask=alpha,
            verts_cam=verts_cam,
            bkgd_mask=alpha,
            image_path=img_path,
            mask_path=mask_path,
            image_name=view_name,
            width=self.width,
            height=self.height,
            smpl_param=smpl_param,
            world_vertex=world_vertex,
            world_bound=world_bound,
            big_pose_smpl_param=self.big_pose_smpl_param,
            big_pose_world_vertex=self.canonical_verts,
            big_pose_world_bound=self.big_pose_world_bound,
        )

        # Process camera info
        cam_info = loadCam_aug_bs(idx, cam_info)
        cam_info_dicts = [vars(cam_info)]
        final_results = merge_batch(cam_info_dicts)

        return final_results


def make_lisa_test_dataloader(data_root='example_data/interhand2.6m', test_views=None, batch_size=1):
    """Create a dataloader for Lisa test dataset."""
    from data.debug import make_dataloader

    ds = LisaTestDataset(data_root=data_root, test_views=test_views)
    return make_dataloader(ds, shuffle=False, batch_size=batch_size)


if __name__ == '__main__':
    # Test the dataset
    ds = LisaTestDataset()
    print(f'Dataset length: {len(ds)}')

    for i in range(len(ds)):
        batch = ds[i]
        print(f'View {i}: {ds.test_views[i]}')
        print(f'  original_image shape: {batch["original_image"].shape}')
        print(f'  K shape: {batch["K"].shape}')
