import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
if os.environ.get("LHM_USE_DGR32", "0") == "1":
    from diff_gaussian_rasterization_32 import (
        GaussianRasterizationSettings,
        GaussianRasterizer_32 as GaussianRasterizer,
    )
else:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
from pytorch3d.ops import knn_points


from data.interhand.train import Renderer_mesh


from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply

from pytorch3d.ops import knn_points
import torch.nn.functional as F

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from LHM.models.rendering.smpl_x import SMPLXModel
from LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel, MANOVoxelMeshModel
from LHM.models.rendering.utils.sh_utils import RGB2SH, SH2RGB
from LHM.models.rendering.utils.typing import *
from LHM.models.rendering.utils.utils import MLP, trunc_exp
from LHM.models.utils import LinerParameterTuner, StaticParameterTuner
from LHM.outputs.output import GaussianAppOutput, GaussianDensifyOutput
from scene.hexplane import HexPlaneField
import torch.nn.init as init
from tools.general_util import get_expon_lr_func
from LHM.models.embedder import BodyPoseRefiner
from tools_utils.map import *

from diffusers.models.normalization import AdaLayerNormZero


def auto_repeat_size(tensor, repeat_num, axis=0):
    repeat_size = [1] * tensor.dim()
    repeat_size[axis] = repeat_num
    return repeat_size

def aabb(xyz):
    return torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values

def inverse_sigmoid(x):
    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))

def ndc_T_world(xyzs_world, K, E, H, W):
    E = E.cuda()
    K = K.cuda()

    xyzs_cam = xyzs_world
    xys_2d = img_T_cam(xyzs_cam, K)

    xs = ((xys_2d[:, 0, :] / W) * 2. - 1.)
    ys = ((xys_2d[:, 1, :] / W) * 2. - (H / W))
    zs = xyzs_cam[:, 2]
    xyzs_ndc = torch.stack([xs, ys, zs], dim=-1)
    return xyzs_ndc

def img_T_cam(xyzs_cam, K):
    K = K.unsqueeze(0).float()
    xys_ = torch.bmm(K, xyzs_cam)
    xys = xys_[:, :2] / xys_[:, 2:]
    return xys

def img_T_world(xyzs_world, K, E):
    xyzs_cam = cam_T_world(xyzs_world, E)
    xys = img_T_cam(xyzs_cam, K)
    return xys

def cam_T_world(xyzs_world, E):
    E=E.unsqueeze(0).float()
    xyzs_world_ = torch.cat([xyzs_world, torch.ones_like(xyzs_world[:, :1])], dim=1)
    xyzs_cam_ = torch.bmm(E, xyzs_world_)
    xyzs_cam = xyzs_cam_[:, :3] / xyzs_cam_[:, 3:]
    return xyzs_cam

def world_from_ndc(xyzs_ndc, K, E, H, W):
    """
    根据 NDC 空间的点恢复到世界坐标系。

    参数
    ----
    xyzs_ndc : Tensor, shape (..., 3)
        NDC 空间下的点，最后一个维度是 (x_ndc, y_ndc, z_cam)。
        其中 z_cam 是它在相机坐标系下的深度。
    K : Tensor, shape (3, 3)
        相机内参矩阵。
    E : Tensor, shape (3, 4)
        外参矩阵，将世界坐标（齐次）映射到相机坐标（非齐次）。
    H, W : int
        图像的高和宽，用于 NDC ↔ 像素坐标的转换。

    返回
    ----
    xyzs_world : Tensor, shape (..., 3)
        恢复得到的世界坐标下的三维点。
    """

    x_ndc, y_ndc, z_cam = xyzs_ndc[..., 0], xyzs_ndc[..., 1], xyzs_ndc[..., 2]

    if H < W:
        u = (W - x_ndc * H) / 2

        v = H * (1 - y_ndc) / 2
    else:
        u = W * (1 - x_ndc) / 2

        v = (H - y_ndc * W) / 2

    x_h = u * z_cam
    y_h = v * z_cam
    z_h = z_cam
    xyzs_pix = torch.stack([x_h, y_h, z_h], dim=-1)

    K_inv = torch.inverse(K)

    original_shape = xyzs_pix.shape
    xyzs_flat = xyzs_pix.reshape(-1, 3).unsqueeze(-1)
    cam_xyz_flat = torch.bmm(K_inv.unsqueeze(0).expand(xyzs_flat.size(0), -1, -1), xyzs_flat)
    cam_xyz = cam_xyz_flat.squeeze(-1).reshape(original_shape)

    R = E[:3, :3]
    t = E[:3, 3:]

    E_h = torch.zeros(4, 4, device=E.device, dtype=E.dtype)
    E_h[:3, :3] = R
    E_h[:3, 3:] = t
    E_h[3, 3] = 1.0
    E_h_inv = torch.inverse(E_h)

    ones = torch.ones(*original_shape[:-1], 1, device=cam_xyz.device, dtype=cam_xyz.dtype)
    cam_xyz_h = torch.cat([cam_xyz, ones], dim=-1)

    cam_xyz_h_flat = cam_xyz_h.reshape(-1, 4).unsqueeze(-1)
    world_h_flat = torch.bmm(E_h_inv.unsqueeze(0).expand(cam_xyz_h_flat.size(0), -1, -1), cam_xyz_h_flat)
    world_xyz = world_h_flat.squeeze(-1)[..., :3].reshape(original_shape)

    return world_xyz

def generate_rotation_matrix_y(degrees):
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)

def getProjectionMatrix_new(K, zfar=100, znear=0.01):
    focalx, focaly = K[0, 0].item(), K[1, 1].item()
    px, py = K[0, 2].item(), K[1, 2].item()
    h, w = 720, 1280

    K_ndc = torch.tensor([
                [2 * focalx / w, 0, (2 * px - w) / w, 0],
                [0, 2 * focaly / h, (2 * py - h) / h, 0],
                [0, 0, zfar / (zfar - znear), -zfar * znear / (zfar - znear)],
                [0, 0, 1, 0]
            ]).float().to(K.device)
    return K_ndc

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix_refine(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    s = 0
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y

def getWorld2View2(c2w, translate=np.array([.0, .0, .0]), scale=1.0):
    C2W = torch.linalg.inv(c2w)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + torch.tensor(translate).cuda()) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return (Rt)

class Camera:
    def __init__(
        self,
        w2c,
        intrinsic,
        FoVx,
        FoVy,
        height,
        width,

        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(w2c, self.trans, self.scale)

        self.zfar = 100.0
        self.znear = 0.01

        self.projection_matrix = (
            getProjectionMatrix_refine(
                intrinsic, self.height, self.width)
            .transpose(0, 1)
            .to(w2c.device)
        )

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = intrinsic

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(
            intrinsic,
            w=torch.tensor(width, device=w2c.device),
            h=torch.tensor(height, device=w2c.device),
        )
        return Camera(
            w2c=w2c,
            intrinsic=intrinsic,
            FoVx=FoVx,
            FoVy=FoVy,
            height=height,
            width=width,

        )

class Hand_Camera:
    def __init__( self, w2c, intrinsic,
                    FoVx, FoVy, height, width,

                    trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.R = R
        self.T = T
        self.K = K

        self.height = height
        self.width = width
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        self.zfar = 100.0
        self.znear = 0.01

        self.projection_matrix = (
            getProjectionMatrix_refine(
                intrinsic, self.height, self.width)
            .transpose(0, 1)
            .to(w2c.device)
        )

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = intrinsic

class GaussianModel32:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(self, xyz, opacity, rotation, scaling, shs, use_rgb=False) -> None:
        """
        Initializes the GSRenderer object.
        Args:
            xyz (Tensor): The xyz coordinates.
            opacity (Tensor): The opacity values.
            rotation (Tensor): The rotation values.
            scaling (Tensor): The scaling values.
            before_activate: if True, the output appearance is needed to process by activation function.
            shs (Tensor): The spherical harmonics coefficients.
            use_rgb (bool, optional): Indicates whether shs represents RGB values. Defaults to False.
        """

        self.setup_functions()

        self.xyz: Tensor = xyz
        self.opacity: Tensor = opacity
        self.rotation: Tensor = rotation
        self.scaling: Tensor = scaling
        self.shs: Tensor = shs

        self.use_rgb = False

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None

        self.spatial_lr_scale = 0
        self.training_setup()

    def get_xyz(self):
        return self.xyz

    def training_setup(self):
        self._xyz = nn.Parameter(self.xyz.requires_grad_(True))
        self._features_color = nn.Parameter(self.shs.requires_grad_(True))

        self._scaling = nn.Parameter(self.scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self.rotation.requires_grad_(True))
        self._opacity = nn.Parameter(self.opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': 0.001, "name": "xyz"},
            {'params': [self._features_color], 'lr': 0.001, "name": "f_color"},

            {'params': [self._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.005, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=0.001,
                                                    lr_final=0.00002,
                                                    lr_delay_mult=0.02,
                                                    max_steps=500)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]

        for i in range(self.shs.shape[1]):
            l.append("f_color_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.use_rgb:
            shs = RGB2SH(self.shs)
        else:
            shs = self.shs

        features_color = shs

        opacities = (
            inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3))
            .detach()
            .cpu()
            .numpy()
        )

        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, features_color, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]

        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        sh_degree = int(math.sqrt((len(extra_f_names) + 3) / 3)) - 1

        print("load sh degree: ", sh_degree)

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).to(self.xyz)
        opacities = torch.from_numpy(opacities).to(self.opacity)
        rotation = torch.from_numpy(rots).to(self.rotation)
        scales = torch.from_numpy(scales).to(self.scaling)
        features_dc = torch.from_numpy(features_dc).to(self.shs)
        features_rest = torch.from_numpy(features_extra).to(self.shs)

        shs = torch.cat([features_dc, features_rest], dim=2)

        if self.use_rgb:
            shs = SH2RGB(shs)
        else:
            shs = shs

        self.xyz: Tensor = xyz
        self.opacity: Tensor = self.opacity_activation(opacities)
        self.rotation: Tensor = self.rotation_activation(rotation)
        self.scaling: Tensor = self.scaling_activation(scales)
        self.shs: Tensor = shs.permute(0, 2, 1)

        self.active_sh_degree = sh_degree

    def clone(self):
        xyz = self.xyz.clone()
        opacity = self.opacity.clone()
        rotation = self.rotation.clone()
        scaling = self.scaling.clone()
        shs = self.shs.clone()
        use_rgb = self.use_rgb
        return GaussianModel(xyz, opacity, rotation, scaling, shs, use_rgb)

class GaussianModel:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(self, xyz, opacity, rotation, scaling, shs, labels, use_rgb,
                 xyz_densify, opacity_densify, rotation_densify, scaling_densify, shs_densify, labels_densify,
                 skip_training_setup=False,
                 ) -> None:
        """
        Initializes the GSRenderer object.
        Args:
            xyz (Tensor): The xyz coordinates.
            opacity (Tensor): The opacity values.
            rotation (Tensor): The rotation values.
            scaling (Tensor): The scaling values.
            before_activate: if True, the output appearance is needed to process by activation function.
            shs (Tensor): The spherical harmonics coefficients.
            use_rgb (bool, optional): Indicates whether shs represents RGB values. Defaults to False.
            skip_training_setup (bool): If True, skip creating nn.Parameter wrappers
                and internal optimizer. Set True when GaussianModel is used as a
                transient container during the rendering forward pass.
        """

        self.setup_functions()

        self.xyz: Tensor = xyz
        self.opacity: Tensor = opacity
        self.rotation: Tensor = rotation
        self.scaling: Tensor = scaling
        self.shs: Tensor = shs
        self.object_dc: Tensor = labels

        self.xyz: Tensor = torch.cat([xyz, xyz_densify], dim=0)
        self.opacity: Tensor = torch.cat([opacity, opacity_densify], dim=0)
        self.rotation: Tensor = torch.cat([rotation, rotation_densify], dim=0)
        self.scaling: Tensor = torch.cat([scaling, scaling_densify], dim=0)

        self.shs: Tensor = torch.cat([shs, shs_densify], dim=0)
        self.object_dc: Tensor = torch.cat([labels, labels_densify], dim=0)

        self.nail_mask = labels.cuda()

        self.use_rgb = use_rgb

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None

        self.spatial_lr_scale = 0
        if not skip_training_setup:
            self.training_setup()

    def get_xyz(self):
        return self.xyz

    def training_setup(self):
        self._xyz = nn.Parameter(self.xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(self.shs[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(self.shs[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(self.scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self.rotation.requires_grad_(True))
        self._opacity = nn.Parameter(self.opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': 0.001, "name": "xyz"},
            {'params': [self._features_dc], 'lr': 0.01, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.01/20, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.005, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=0.001,
                                                    lr_final=0.00002,
                                                    lr_delay_mult=0.02,
                                                    max_steps=500)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.use_rgb:
            shs = RGB2SH(self.shs)
        else:
            shs = self.shs

        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]

        f_dc = (
            features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            features_rest.float()
            .detach()
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = (
            inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3))
            .detach()
            .cpu()
            .numpy()
        )

        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]

        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        sh_degree = int(math.sqrt((len(extra_f_names) + 3) / 3)) - 1

        print("load sh degree: ", sh_degree)

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).to(self.xyz)
        opacities = torch.from_numpy(opacities).to(self.opacity)
        rotation = torch.from_numpy(rots).to(self.rotation)
        scales = torch.from_numpy(scales).to(self.scaling)
        features_dc = torch.from_numpy(features_dc).to(self.shs)
        features_rest = torch.from_numpy(features_extra).to(self.shs)

        shs = torch.cat([features_dc, features_rest], dim=2)

        if self.use_rgb:
            shs = SH2RGB(shs)
        else:
            shs = shs

        self.xyz: Tensor = xyz
        self.opacity: Tensor = self.opacity_activation(opacities)
        self.rotation: Tensor = self.rotation_activation(rotation)
        self.scaling: Tensor = self.scaling_activation(scales)
        self.shs: Tensor = shs.permute(0, 2, 1)

        self.active_sh_degree = sh_degree

    def clone(self):
        xyz = self.xyz.clone()
        opacity = self.opacity.clone()
        rotation = self.rotation.clone()
        scaling = self.scaling.clone()
        shs = self.shs.clone()
        use_rgb = self.use_rgb
        return GaussianModel(xyz, opacity, rotation, scaling, shs, use_rgb)

class GSLayer(nn.Module):
    """W/O Activation Function"""

    def setup_functions(self):
        self.scaling_activation = trunc_exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(
        self,
        in_channels,
        use_rgb,
        clip_scaling=0.2,
        init_scaling=-5.0,
        init_density=0.1,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
        use_fine_feat=False,
        feature_map=False,
    ):
        super().__init__()
        self.setup_functions()

        if isinstance(clip_scaling, omegaconf.listconfig.ListConfig) or isinstance(
            clip_scaling, list
        ):
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)

        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat

        if not feature_map:
            self.attr_dict = {
                "shs": (sh_degree + 1) ** 2 * 3,
                "scaling": 3,
                "xyz": 3,
                "opacity": None,
                "rotation": None,
            }
        else:
            self.attr_dict = {
                "shs": 32,
                "scaling": 3,
                "xyz": 3,
                "opacity": None,
                "rotation": None,
            }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs":
                    shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)

            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = layer

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def constrain_forward(self, ret, constrain_dict):
        is_upper_body = constrain_dict['is_upper_body']
        scaling = ret['scaling']

        scaling[is_upper_body] = scaling[is_upper_body].clamp(max = 0.02)

        ret['scaling'] = scaling

        return ret

    def constrain_expr(self, ret, constrain_head):
        head_mask = constrain_head['head']
        face_mask = constrain_head['face']
        xyz = ret['offset_xyz']

        head_only_mask = head_mask & (~face_mask)

        xyz[head_only_mask] = xyz[head_only_mask] * 0.5

        ret['offset_xyz'] = xyz

        return ret

    def forward(self, x, pts, x_fine=None, constrain_dict=None, constrain_head=None):
        assert len(x.shape) == 2
        ret = {}
        for k in self.attr_dict:
            layer = self.out_layers[k]

            v = layer(x)
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(
                        torch.eye(3).type_as(x)[None, :, :].repeat(x.shape[0], 1, 1)
                    )
                else:
                    v = self.rotation_activation(v)
            elif k == "scaling":
                v = self.scaling_activation(v)

                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    v = self.opacity_activation(v)
            elif k == "shs":
                if self.use_rgb:
                    v = self.rgb_activation(v)

                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    v = self.rgb_activation(v)
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
            elif k == "xyz":
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step

                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v

        ret["use_rgb"] = self.use_rgb

        if constrain_dict is not None:
            ret = self.constrain_forward(ret, constrain_dict)

        if constrain_head is not None:
            ret = self.constrain_expr(ret, constrain_head)

        return GaussianAppOutput(**ret)

class GSLayer_Hand(nn.Module):
    """W/O Activation Function"""

    def setup_functions(self):
        self.scaling_activation = trunc_exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(
            self,
            pcl_embed,
            mano,
            in_channels,
            use_rgb,
            clip_scaling=0.2,
            init_scaling=-5.0,
            init_density=0.1,
            sh_degree=None,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=None,
            fix_opacity=False,
            fix_rotation=False,
            use_fine_feat=False,
            feature_map=False,
    ):
        super().__init__()
        self.setup_functions()
        self.in_channels = in_channels

        if isinstance(clip_scaling, omegaconf.listconfig.ListConfig) or isinstance(
                clip_scaling, list
        ):
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)

        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat

        self.mano_face = mano.faces_tensor.long()
        self.mano_verts = mano.v_template
        self.face_area = calc_face_areas(self.mano_verts, self.mano_face)
        self.mano_face = self.mano_face.to('cuda')

        F_num = self.mano_face.shape[0]
        init_bary = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)

        self.barycentric_raw = nn.Parameter(
                            torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32).unsqueeze(0).repeat(F_num, 1).cuda(), requires_grad=True)

        self.color_latent_gamma = nn.Parameter(torch.zeros(1, in_channels, device='cuda'))
        self.color_latent_beta = nn.Parameter(torch.zeros(1, in_channels, device='cuda'))

        self.register_buffer(
            "stage1_color_field_freqs",
            torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda'),
        )

        self.stage1_color_coeff = nn.Parameter(torch.zeros(16, 1, device='cuda'))
        self.stage1_color_rgb = nn.Parameter(torch.ones(1, 3, device='cuda'))
        self.stage1_color_field_alpha = 1.0

        if not feature_map:
            self.attr_dict = {
                "shs": (sh_degree + 1) ** 2 * 3,
                "scaling": 3,
                "xyz": 3,
                "opacity": None,
                "rotation": None,
            }
        else:
            self.attr_dict = {
                "shs": 32,
                "scaling": 3,
                "xyz": 3,
                "opacity": None,
                "rotation": None,
            }

        self.densify_attr_dict = {
                "activation": 1,
                "shs": (sh_degree + 1) ** 2 * 3,
                "bary": 3,
                "opacity": None,
                "rotation": None,
                "scaling": 3,
        }

        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
            self.densify_attr_dict["opacity"] = 1
        self.densify_attr_dict["activation"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4
            self.densify_attr_dict["rotation"] = 4

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs" and not use_rgb:
                    out_ch = (sh_degree + 1) ** 2 * 3

                layer = nn.Linear(in_channels, out_ch)

            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))

            self.out_layers[key] = layer

        self.out_layers = self.out_layers.cuda()

        self.densify_out_layers = nn.ModuleDict()
        for key, out_ch in self.densify_attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs" and not use_rgb:
                    out_ch = (sh_degree + 1) ** 2 * 3

                layer = nn.Linear(in_channels, out_ch)

            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            elif key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "bary":
                init_bary = 1/3
                nn.init.constant_(layer.bias, init_bary)
            self.densify_out_layers[key] = layer

        self.densify_out_layers = self.densify_out_layers.cuda()

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        with open(labels_path, 'rb') as f:
            pc = PlyData.read(f)
        if pc.elements:
            pc = pd.DataFrame(pc.elements[0].data).values
        pc_coords = pc[:, :3]
        self.labels = torch.tensor(pc[:, 6].astype(np.uint8))

        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        nail_indices = np.isin(self.labels, list(NAIL_PARTS.values()))

        nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))
        self.nail_mask = nail_mask.cuda()

        self.objects_dc = nail_mask.to(torch.float32).unsqueeze(1).cuda()

    def _expand_shared_style(self, tensor, feat):
        if tensor is None:
            return None
        out = tensor.to(device=feat.device, dtype=feat.dtype)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and feat.shape[0] > 1:
            out = out.expand(feat.shape[0], -1)
        elif out.shape[0] != feat.shape[0]:
            out = out[:1].expand(feat.shape[0], -1)
        return out

    def _apply_color_latent_modulation(self, feat, global_feat=None):
        gamma = self._expand_shared_style(self.color_latent_gamma, feat)
        beta = self._expand_shared_style(self.color_latent_beta, feat)
        if gamma is None or beta is None:
            return feat

        if global_feat is not None:
            style = self._expand_shared_style(global_feat, feat)
            style = F.layer_norm(style, (style.shape[-1],))
        else:
            style = torch.ones_like(feat)

        gamma = 0.5 * torch.tanh(gamma)
        beta = 0.25 * torch.tanh(beta)
        return feat * (1.0 + gamma * style) + beta * style

    def _build_stage1_color_basis(self, pts, center=None, scale=None):
        if pts is None or pts.numel() == 0:
            return None, center, scale

        if center is None:
            center = pts.mean(dim=0, keepdim=True)
        pts_centered = pts - center

        if scale is None:
            scale = pts_centered.norm(dim=-1).amax().clamp_min(1e-6)

        p = pts_centered / scale
        r2 = (p * p).sum(dim=-1, keepdim=True)

        basis_parts = [p]
        freqs = self.stage1_color_field_freqs.to(device=pts.device, dtype=pts.dtype)
        for freq in freqs:
            omega = float(freq.item()) * math.pi
            basis_parts.append(torch.sin(omega * p))

            basis_parts.append(torch.cos(omega * p) - 1.0)
        basis_parts.append(r2)

        basis = torch.cat(basis_parts, dim=-1)
        return basis, center, scale

    def compute_stage1_color_offsets(self, canonical_pts, base_shs=None, densify_shs=None):
        coeff = getattr(self, "stage1_color_coeff", None)
        if coeff is None or canonical_pts is None or canonical_pts.numel() == 0:
            return None, None

        alpha = float(getattr(self, "stage1_color_field_alpha", 1.0))
        if alpha <= 0.0:
            return None, None

        basis, center, scale = self._build_stage1_color_basis(canonical_pts)
        if basis is None:
            return None, None

        coeff_eff = 0.35 * torch.tanh(coeff.to(device=canonical_pts.device, dtype=canonical_pts.dtype))
        scalar_delta = basis @ coeff_eff
        rgb_eff = 0.25 * torch.tanh(
            self.stage1_color_rgb.to(device=canonical_pts.device, dtype=canonical_pts.dtype)
        )
        base_delta = scalar_delta * rgb_eff

        bary = torch.softmax(self.barycentric_raw, dim=-1).to(
            device=canonical_pts.device, dtype=canonical_pts.dtype
        )
        densify_pts = (canonical_pts[self.mano_face] * bary.unsqueeze(-1)).sum(dim=1)
        densify_basis, _, _ = self._build_stage1_color_basis(densify_pts, center=center, scale=scale)
        densify_scalar = densify_basis @ coeff_eff
        densify_delta = densify_scalar * rgb_eff

        return alpha * base_delta.unsqueeze(1), alpha * densify_delta.unsqueeze(1)

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def constrain_forward(self, ret, constrain_dict):
        is_upper_body = constrain_dict['is_upper_body']
        scaling = ret['scaling']

        scaling[is_upper_body] = scaling[is_upper_body].clamp(max=0.02)

        ret['scaling'] = scaling

        return ret

    def constrain_expr(self, ret, constrain_head):
        head_mask = constrain_head['head']
        face_mask = constrain_head['face']
        xyz = ret['offset_xyz']

        head_only_mask = head_mask & (~face_mask)

        xyz[head_only_mask] = xyz[head_only_mask] * 0.5

        ret['offset_xyz'] = xyz

        return ret

    def densify_by_mesh(self, ret, threshold=0.8):
        densify_mask = (ret['activation']>threshold)
        densify_face = self.mano_face[densify_mask]
        densify_bary = self.barycentric_raw[densify_mask]

        ret_densify = {}
        if torch.sum(densify_mask) > 0:
            color = (ret['shs'][densify_mask] * densify_bary).sum(dim=2)
            opacity = (ret['shs'][densify_mask] * densify_bary).sum(dim=2)
            scaling = (ret['scaling'][densify_mask] * densify_bary).sum(dim=2)
            rotation = (ret['rotation'][densify_mask] * densify_bary).sum(dim=2)

        return ret_densify

    def forward(self, x, pts, x_fine=None, constrain_dict=None, constrain_head=None,
                global_feat=None, film_param=None, vis_mask=None):
        assert len(x.shape) == 2

        ret = {}
        color_x = self._apply_color_latent_modulation(x, global_feat)

        for k in self.attr_dict:
            layer = self.out_layers[k]

            feat_in = color_x if k == "shs" else x
            v = layer(feat_in)

            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(
                        torch.eye(3).type_as(x)[None, :, :].repeat(x.shape[0], 1, 1)
                    )
                else:
                    v = self.rotation_activation(v)
            elif k == "scaling":
                v = self.scaling_activation(v)

                if self.clip_scaling is not None:
                    v[self.labels != 1] = torch.clamp(v[self.labels != 1], min=0.001, max=0.3)
                    v[self.labels == 1] = torch.clamp(v[self.labels == 1], min=0.005, max=0.4)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    v = self.opacity_activation(v)
            elif k == "shs":
                if self.use_rgb:
                    v = self.rgb_activation(v)
                    v = torch.reshape(v, (v.shape[0], -1, 3))
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    v = self.rgb_activation(v)
                    v = torch.reshape(v, (v.shape[0], -1, 3))
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
            elif k == "xyz":
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step

                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v

        densify_ret = {}

        bary = torch.softmax(self.barycentric_raw, dim=-1).unsqueeze(0).unsqueeze(-1)
        face_feats = x.unsqueeze(0)[:, self.mano_face, :]
        face_feats = (face_feats * bary).sum(dim=2).squeeze()
        color_face_feats = color_x.unsqueeze(0)[:, self.mano_face, :]
        color_face_feats = (color_face_feats * bary).sum(dim=2).squeeze()

        for j in self.densify_attr_dict:
            densify_layer = self.densify_out_layers[j]

            x = color_face_feats if j == "shs" else face_feats
            v = densify_layer(x)

            if j == "activation":
                v = self.opacity_activation(v)
            elif j == "bary":
                v = torch.softmax(v, dim=-1)
            elif j == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    v = self.opacity_activation(v)
            elif j == "scaling":
                v = self.scaling_activation(v)
            elif j == "rotation":
                v = self.rotation_activation(v)
            elif j == "shs":
                if self.use_rgb:
                    v = self.rgb_activation(v)
                    v = torch.reshape(v, (v.shape[0], -1, 3))
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    v = self.rgb_activation(v)
                    v = torch.reshape(v, (v.shape[0], -1, 3))
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
            densify_ret[j] = v

        ret["use_rgb"] = self.use_rgb

        if constrain_dict is not None:
            ret = self.constrain_forward(ret, constrain_dict)

        if constrain_head is not None:
            ret = self.constrain_expr(ret, constrain_head)

        return GaussianAppOutput(**ret), GaussianDensifyOutput(**densify_ret)

    def face_vertex_latents(self, vertex_latent):
        """
        vertex_latent: (B, V, C)
        returns L0,L1,L2: each (B, F, C) corresponding to faces' v0/v1/v2 latents
        """
        faces = self.mano_face.long().to(vertex_latent.device)
        v0_idx = faces[:,0]; v1_idx = faces[:,1]; v2_idx = faces[:,2]
        L0 = vertex_latent[:, v0_idx, :]
        L1 = vertex_latent[:, v1_idx, :]
        L2 = vertex_latent[:, v2_idx, :]
        return L0, L1, L2

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )

        self.register_buffer("basis", e)

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)
        self.norm = nn.LayerNorm(dim)

        self.to('cuda')

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)

        return embeddings

    def forward(self, input_):
        input = self.normalize_coords(input_)
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2)
        )
        embed = self.norm(embed)

        return embed

    def normalize_coords(self, verts_pos):
        """
        Normalize verts_pos to roughly [-1,1] per-batch, center and scale by max abs.
        verts_pos: [B, N, 3]
        """
        B, N, C = verts_pos.shape
        center = verts_pos.mean(dim=1, keepdim=True)
        v = verts_pos - center

        max_val = v.abs().amax(dim=(1,2), keepdim=True)
        v = v / (max_val + 1e-6)
        return v

class HierarchicalPointEmbed_ori(nn.Module):
    def __init__(
        self,
        in_pos_dim=3,
        in_feat_dim=0,
        out_dim=1024,
        base_dim=64,
        scales=[8, 16, 32],
        part_dim=128,
        joint_dim=128,
        use_image_feat=False,
    ):
        super().__init__()
        self.scales = scales
        self.use_image_feat = use_image_feat and in_feat_dim>0
        self.base_proj = nn.Linear(in_pos_dim + (in_feat_dim if self.use_image_feat else 0), base_dim)

        self.neigh_aggrs = nn.ModuleList([
            NeighborAggregation(in_feat_dim + base_dim, base_dim, K=k, agg='max') for k in scales
        ])

        ms_in = base_dim * len(scales) + base_dim
        self.ms_fuse = nn.Sequential(
            nn.Linear(ms_in, 256), nn.GELU(),
            nn.Linear(256, 256)
        )

        self.vertex_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, out_dim)
        )

        self.part_pool_proj = nn.Linear(out_dim, part_dim)
        self.part_processor = nn.Sequential(
            nn.Linear(part_dim, part_dim), nn.GELU(),
            nn.Linear(part_dim, part_dim)
        )
        self.part_broadcast_proj = nn.Linear(part_dim, out_dim)

        self.output_norm = nn.LayerNorm(out_dim)

        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        with open(labels_path, 'rb') as f:
            pc = PlyData.read(f)
        if pc.elements:
            pc = pd.DataFrame(pc.elements[0].data).values
        pc_coords = pc[:, :3]

        self.labels = torch.tensor(pc[:, 6].astype(np.int64)).cuda()

        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        self.nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))

    def forward(self, verts_pos, part_idx=None, joint_xyz=None, img_feat=None, nail_mask=None):
        """
        inputs:
          verts_pos: (B, N, 3)
          part_idx:  (B, N) int labels in [0..P-1]  OR (N,) broadcastable. Optional.
          joint_xyz: (B, J, 3) optional
          img_feat:  (B, N, F_img) optional per-vertex image features (from DINOV2 projection)
          nail_mask: (B, N) bool optional
        outputs:
          vertex_feat: (B, N, out_dim)
          importance_score: (B, N, 1)
        """
        B, N, _ = verts_pos.shape
        device = verts_pos.device

        base_in = verts_pos
        base = self.base_proj(base_in)

        center_feat = base

        scale_outs = []
        for aggr in self.neigh_aggrs:
            out_s = aggr(verts_pos, torch.cat([center_feat, img_feat], dim=-1) if self.use_image_feat else center_feat)
            scale_outs.append(out_s)
        ms_cat = torch.cat([center_feat] + scale_outs, dim=-1)

        ms_f = self.ms_fuse(ms_cat)

        v_pre = self.vertex_mlp(ms_f)

        part_idx = self.labels.unsqueeze(0).repeat(B, 1)
        P = int(part_idx.max().item()+1)

        part_token = scatter_mean_torch(v_pre, part_idx, dim_size=P)

        part_token = self.part_pool_proj(part_token)
        part_token = self.part_processor(part_token)

        gather_idx = part_idx

        part_bcast = part_token[torch.arange(B)[:,None], gather_idx]
        part_bcast = self.part_broadcast_proj(part_bcast)
        v_pre = v_pre + part_bcast

        if joint_xyz is not None:
            j_embed = self.joint_embed(joint_xyz)
            j_global = j_embed.mean(dim=1, keepdim=True)
            j_bcast = self.joint_broadcast_proj(j_global)
            v_pre = v_pre + j_bcast

        v_refined = v_pre

        flat = v_pre.view(B*N, -1)

        v_refined = flat.view(B, N, -1)

        v_refined = self.output_norm(v_refined)

        return v_refined

class HierarchicalPointEmbed(nn.Module):
    def __init__(
        self,
        in_pos_dim=3,
        in_feat_dim=0,
        out_dim=1024,
        base_dim=64,
        scales=[8, 16, 32],
        part_dim=128,
        joint_dim=128,
        use_image_feat=False,
        pos_embed_dim=128,
        pos_hidden_dim=48,
    ):
        super().__init__()
        self.scales = scales
        self.use_image_feat = use_image_feat and in_feat_dim>0
        self.pos_embed_dim = pos_embed_dim

        self.point_pos_embed = PointEmbed(hidden_dim=pos_hidden_dim, dim=pos_embed_dim)

        in_base_dim = pos_embed_dim + (in_feat_dim if self.use_image_feat else 0)
        self.base_proj = nn.Linear(in_base_dim, base_dim)

        self.neigh_aggrs = nn.ModuleList([
            NeighborAggregation((in_feat_dim if self.use_image_feat else 0) + base_dim, base_dim, K=k, agg='max')
            for k in scales
        ])

        ms_in = base_dim * len(scales) + base_dim
        self.ms_fuse = nn.Sequential(
            nn.Linear(ms_in, 256), nn.GELU(),
            nn.Linear(256, 256)
        )

        self.vertex_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, out_dim)
        )

        self.part_pool_proj = nn.Linear(out_dim, part_dim)
        self.part_processor = nn.Sequential(
            nn.Linear(part_dim, part_dim), nn.GELU(),
            nn.Linear(part_dim, part_dim)
        )
        self.part_broadcast_proj = nn.Linear(part_dim, out_dim)

        self.output_norm = nn.LayerNorm(out_dim)

        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        with open(labels_path, 'rb') as f:
            pc = PlyData.read(f)
        if pc.elements:
            pc = pd.DataFrame(pc.elements[0].data).values
        pc_coords = pc[:, :3]
        self.labels = torch.tensor(pc[:, 6].astype(np.int64)).cuda()
        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }
        self.nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))

    def normalize_coords(self, verts_pos):
        """
        Normalize verts_pos to roughly [-1,1] per-batch, center and scale by max abs.
        verts_pos: [B, N, 3]
        """
        B, N, C = verts_pos.shape
        center = verts_pos.mean(dim=1, keepdim=True)
        v = verts_pos - center

        max_val = v.abs().amax(dim=(1,2), keepdim=True)
        v = v / (max_val + 1e-6)
        return v

    def forward(self, verts_pos, part_idx=None, joint_xyz=None, img_feat=None, nail_mask=None):
        """
        verts_pos: (B, N, 3)
        img_feat:  (B, N, F_img) optional
        """
        B, N, _ = verts_pos.shape
        device = verts_pos.device

        verts_norm = self.normalize_coords(verts_pos)

        pos_feat = self.point_pos_embed(verts_norm)

        if self.use_image_feat:
            assert img_feat is not None, "use_image_feat=True but img_feat is None"
            base_in = torch.cat([pos_feat, img_feat], dim=-1)
        else:
            base_in = pos_feat

        base = self.base_proj(base_in)

        center_feat = base
        scale_outs = []
        for aggr in self.neigh_aggrs:
            if self.use_image_feat:
                aggr_feat_input = torch.cat([center_feat, img_feat], dim=-1)
            else:
                aggr_feat_input = center_feat
            out_s = aggr(verts_pos, aggr_feat_input)

            scale_outs.append(out_s)
        ms_cat = torch.cat([center_feat] + scale_outs, dim=-1)

        ms_f = self.ms_fuse(ms_cat)
        v_pre = self.vertex_mlp(ms_f)

        part_idx = self.labels.unsqueeze(0).repeat(B, 1)
        P = int(part_idx.max().item()+1)
        part_token = scatter_mean_torch(v_pre, part_idx, dim_size=P)

        part_token = self.part_pool_proj(part_token)
        part_token = self.part_processor(part_token)
        part_bcast = part_token[torch.arange(B)[:,None], part_idx]
        part_bcast = self.part_broadcast_proj(part_bcast)
        v_pre = v_pre + part_bcast

        v_refined = v_pre.view(B, N, -1)

        v_refined = self.output_norm(v_refined)

        return v_refined

class HierarchicalPointEmbedTransformer(nn.Module):
    def __init__(
        self,
        in_feat_dim=1024,
        out_dim=1024,
        base_dim=512,

        scales=[4, 8, 16],
        part_dim=1024,

        joint_dim=1024,
        use_image_feat=False,
    ):
        super().__init__()
        self.scales = scales
        self.use_image_feat = use_image_feat and in_feat_dim > 0

        in_base_dim = in_feat_dim + (in_feat_dim if self.use_image_feat else 0)

        self.base_proj = nn.Linear(in_base_dim, base_dim)

        self.neigh_aggrs = nn.ModuleList([
            NeighborAggregation(
                (in_feat_dim if self.use_image_feat else 0) + base_dim,
                base_dim,
                K=k,
                agg='max'
            )
            for k in scales
        ])

        ms_in = base_dim * (len(scales) + 1)
        self.ms_fuse = nn.Sequential(
            nn.Linear(ms_in, 256), nn.GELU(),
            nn.Linear(256, 256)
        )

        self.vertex_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, out_dim)
        )

        self.part_pool_proj = nn.Linear(out_dim, part_dim)
        self.part_processor = nn.Sequential(
            nn.Linear(part_dim, part_dim), nn.GELU(),
            nn.Linear(part_dim, part_dim)
        )
        self.part_broadcast_proj = nn.Linear(part_dim, out_dim)

        self.output_norm = nn.LayerNorm(out_dim)

        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'

        with open(labels_path, 'rb') as f:
            pc = PlyData.read(f)

        if pc.elements:
            pc = pd.DataFrame(pc.elements[0].data).values
        pts = pc[:, :3]

        self.labels = torch.tensor(pc[:, 6].astype(np.int64)).cuda()

    def forward(
        self,
        point_feat,
        verts_pos=None,
        img_feat=None,
        nail_mask=None,
    ):
        """
        point_feat: (B, N, C_in)  主输入特征
        verts_pos:  (B, N, 3)     可选：如果要做 NeighborAggregation 的 KNN，就必须提供
        """
        B, N, C_in = point_feat.shape
        device = point_feat.device

        v_pre = point_feat

        part_idx = self.labels.unsqueeze(0).repeat(B, 1).to(device)
        P = int(part_idx.max().item() + 1)

        part_token = scatter_mean_torch(v_pre, part_idx, dim_size=P)

        part_token = self.part_processor(part_token)

        part_bcast = part_token[torch.arange(B)[:, None], part_idx]

        v_pre = v_pre + part_bcast

        v_refined = self.output_norm(v_pre)

        return v_refined

class NeighborAggregation_new(nn.Module):
    """
    Memory-friendly NeighborAggregation:
      - 不在内部计算 KNN，而是从外部传入 idx (B, N, K_used)
      - 通过按邻居维度 K 的 for 循环，避免构造 (B, N, K, *) 大张量
      - 输出形状仍为 (B, N, out_dim)
    """
    def __init__(self, in_feat_dim, out_dim, K=8, agg='max'):
        super().__init__()
        self.K = K
        self.agg = agg
        self.out_dim = out_dim

        self.inp_dim = 3 + 1 + in_feat_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.inp_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, pos, feat, idx=None):
        B, N, C = feat.shape
        device = pos.device

        if idx is None:
            idx = knn_idx(pos, self.K)
        else:
            if idx.size(-1) < self.K:
                raise ValueError(f"idx.size(-1)={idx.size(-1)} < K={self.K}")
            idx = idx[..., :self.K]

        batch_idx = torch.arange(B, device=device)[:, None, None]

        neigh_pos  = pos[batch_idx, idx]
        neigh_feat = feat[batch_idx, idx]

        center = pos.unsqueeze(2)
        rel    = neigh_pos - center
        dist   = torch.norm(rel, dim=-1, keepdim=True)

        center_feat = feat.unsqueeze(2).expand(-1, -1, self.K, -1)

        inp = torch.cat([rel, dist, center_feat, neigh_feat], dim=-1)

        BnK = B * N * self.K
        inp_flat = inp.view(BnK, -1)
        out_flat = self.mlp(inp_flat)
        out = out_flat.view(B, N, self.K, -1)

        if self.agg == 'max':
            out_agg, _ = out.max(2)
        else:
            out_agg = out.mean(2)
        return out_agg

class NeighborAggregation(nn.Module):
    """
    For each vertex, gathers neighbor relative coords + features and aggregates.
    input feat dims:
      pos: (B,N,3)
      feat: (B,N,C)
    returns: aggregated per-vertex feature (B,N, out_dim)
    """
    def __init__(self, in_feat_dim, out_dim, K=8, agg='max'):
        super().__init__()
        self.K = K
        self.agg = agg

        self.mlp = nn.Sequential(
            nn.Linear(3 + 1 + in_feat_dim*2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, pos, feat):
        B, N, C = feat.shape
        idx = knn_idx(pos, self.K)

        device = pos.device
        batch_idx = torch.arange(B, device=device)[:, None, None]
        neigh_pos = pos[batch_idx, idx]
        neigh_feat = feat[batch_idx, idx]

        center = pos.unsqueeze(2)
        rel = neigh_pos - center
        dist = torch.norm(rel, dim=-1, keepdim=True)

        center_feat = feat.unsqueeze(2).expand(-1, -1, self.K, -1)

        inp = torch.cat([rel, dist, center_feat, neigh_feat], dim=-1)

        BnK = B * N * self.K
        inp_flat = inp.view(BnK, -1)
        out_flat = self.mlp(inp_flat)
        out = out_flat.view(B, N, self.K, -1)
        if self.agg == 'max':
            out_agg, _ = out.max(2)
        else:
            out_agg = out.mean(2)
        return out_agg

def pairwise_dist(x, y):
    x2 = (x**2).sum(-1, keepdim=True)
    y2 = (y**2).sum(-1, keepdim=True).transpose(1, 2)
    xy = x @ y.transpose(1, 2)
    dist = x2 + y2 - 2*xy
    return dist

def knn_idx(pts, K):
    idx = knn_points(pts, pts, K=K).idx
    return idx

def knn_faiss(x, K):
    B, N, D = x.shape
    x_np = x.detach().cpu().numpy()
    idx_all = []

    for b in range(B):
        index = faiss.IndexFlatL2(D)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(x_np[b])
        _, idx = index.search(x_np[b], K)
        idx_all.append(torch.from_numpy(idx).long())

    return torch.stack(idx_all).to(x.device)

def scatter_mean_torch(src, idx, dim_size):
    B, N, C = src.shape
    out = src.new_zeros(B, dim_size, C)
    count = src.new_zeros(B, dim_size, 1)
    for b in range(B):
        out[b].index_add_(0, idx[b], src[b])
        ones = src.new_ones(N, 1, device=src.device)
        count[b].index_add_(0, idx[b], ones)
    count = count.clamp(min=1.0)
    out = out / count
    return out

class CrossAttnBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """

    def __init__(
        self,
        inner_dim: int,
        cond_dim: int,
        num_heads: int,
        eps: float = None,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        feedforward=False,
    ):
        super().__init__()

        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )

        self.mlp = None
        if feedforward:
            self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                dropout=attn_drop,
                bias=attn_bias,
                batch_first=True,
            )
            self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
            self.mlp = nn.Sequential(
                nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(mlp_drop),
                nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
                nn.Dropout(mlp_drop),
            )

    def forward(self, x, cond):
        x = self.cross_attn(
            self.norm_q(x), self.norm_k(cond), cond, need_weights=False
        )[0]
        if self.mlp is not None:
            before_sa = self.norm2(x)
            x = (
                x
                + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
            )
            x = x + self.mlp(self.norm3(x))
        return x

class SwiGLU_FF(nn.Module):
    def __init__(self, dim, inner_dim, dropout=0.0):
        super().__init__()
        self.proj_in = nn.Linear(dim, inner_dim * 2)
        self.proj_out = nn.Linear(inner_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = self.proj_in(x).chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.proj_out(x)
        return x

class DecoderCrossAttn(nn.Module):
    def __init__(
        self, query_dim, context_dim, num_heads, mlp=False, decode_with_extra_info=None
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.cross_attn = CrossAttnBlock(
            inner_dim=query_dim,
            cond_dim=context_dim,
            num_heads=num_heads,
            feedforward=mlp,
            eps=1e-5,
        )
        self.decode_with_extra_info = decode_with_extra_info
        if decode_with_extra_info is not None:
            if decode_with_extra_info["type"] == "dinov2p14_feat":
                context_dim = decode_with_extra_info["cond_dim"]
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=context_dim,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                from LHM.models.encoders.dinov2_wrapper import Dinov2Wrapper

                self.encoder = Dinov2Wrapper(
                    model_name="dinov2_vits14_reg", freeze=False, encoder_feat_dim=384
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=384,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_resnet18_feat":
                from LHM.models.encoders.xunet_wrapper import XnetWrapper

                self.encoder = XnetWrapper(
                    model_name="resnet18", freeze=False, encoder_feat_dim=64
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=64,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )

    def resize_image(self, image, multiply):
        B, _, H, W = image.shape
        new_h, new_w = (
            math.ceil(H / multiply) * multiply,
            math.ceil(W / multiply) * multiply,
        )
        image = F.interpolate(
            image, (new_h, new_w), align_corners=True, mode="bilinear"
        )
        return image

    def forward(self, pcl_query, pcl_latent, extra_info=None):
        out = self.cross_attn(pcl_query, pcl_latent)
        if self.decode_with_extra_info is not None:
            out_dict = {}
            out_dict["coarse"] = out
            if self.decode_with_extra_info["type"] == "dinov2p14_feat":
                out = self.cross_attn_color(out, extra_info["image_feats"])
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                img_feat = self.encoder(extra_info["image"])
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_resnet18_feat":
                image = extra_info["image"]
                image = self.resize_image(image, multiply=32)
                img_feat = self.encoder(image)
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
        return out

class Head_Res_Net(nn.Module):
    def __init__(self, W, H):
        super(Head_Res_Net, self).__init__()
        self.W = W
        self.H = H

        self.feature_out = [Linear_Res(self.W)]
        self.feature_out.append(nn.Linear(W, self.H))
        self.feature_out = nn.Sequential(*self.feature_out)

    def initialize_weights(self, ):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)

class Linear_Res(nn.Module):
    def __init__(self, W):
        super(Linear_Res, self).__init__()
        self.main_stream = nn.Linear(W, W)

    def forward(self, x):
        x = F.relu(x)
        return x + self.main_stream(x)

def initialize_zeros_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)

class GridOffset(nn.Module):
    def __init__(self):
        super(GridOffset, self).__init__()

        kplanes_config = {'grid_dimensions': 2,
                          'input_coordinate_dim': 4,
                          'output_coordinate_dim': 32,
                          'resolution': [32, 32, 32, 12]}

        self.W, self.D = 64, 1
        self.input_ch = 27

        self.grid = HexPlaneField(bounds=1.6, multires=[1], planeconfig=kplanes_config)

        self.pos_, self.sceles_, self.rots_, self.opacity_ = self.create_res_net()

        self.apply(initialize_weights)
        self.pos_.initialize_weights()

    def create_res_net(self):
        self.feature_out = [nn.Linear(self.grid.feat_dim, self.W)]

        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        output_dim = self.W
        return\
            Head_Res_Net(self.W, 3),\
                Head_Res_Net(self.W, 3),\
                Head_Res_Net(self.W, 4),\
                Head_Res_Net(self.W, 1)

class HyperFiLM(nn.Module):
    """
    Hypernetwork that maps a global latent z (B, zdim) into FiLM parameters.
    We create per-attribute gamma and beta vectors of length 'in_channels'.
    Attributes: list of attribute keys, e.g. ['scaling','xyz','opacity','rotation','shs']
    """
    def __init__(self, z_dim=1024, in_channels=1024, attributes=None, hidden=512):
        super().__init__()
        if attributes is None:
            attributes = ["activation", "bary"]
        self.attrs = attributes
        self.in_ch = in_channels

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.attr_proj = nn.Linear(hidden, in_channels * 2)
        nn.init.constant_(self.attr_proj.weight, 0.0)
        nn.init.constant_(self.attr_proj.bias, 0.0)

        with torch.no_grad():
            self.attr_proj.bias[:in_channels].fill_(1.0)

    def forward(self, z):
        """
        z: (B, z_dim)
        returns dict attr -> (gamma (B, C), beta (B, C))
        """
        h = self.mlp(z)

        v = self.attr_proj(h)
        gamma, beta = v[:, :self.in_ch], v[:, self.in_ch:]
        out = (gamma.unsqueeze(1), beta.unsqueeze(1))
        return out

class FiLMModulator(nn.Module):
    """
    Produce FiLM parameters for a list of layer hidden sizes.
    - global_vec: [B, gdim]
    returns:
    - list of (gamma, beta), each gamma/beta is [B, layer_dim]
    """
    def __init__(self, gdim:int, layer_dims: list, hidden: int = 256):
        super().__init__()
        self.gdim = gdim
        self.layer_dims = list(layer_dims)
        self.hidden = hidden

        self.mlp_per_layer = nn.ModuleList()
        for dim in self.layer_dims:
            self.mlp_per_layer.append(
                nn.Sequential(
                    nn.Linear(gdim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, dim * 2)
                )
            )

    def forward(self, global_vec: torch.Tensor):
        out = []
        for mlp in self.mlp_per_layer:
            gb = mlp(global_vec)
            dim = gb.shape[-1] // 2
            gamma = gb[:, :dim]
            beta = gb[:, dim:]
            out.append((gamma, beta))
        return out

class LoRAAdapter(nn.Module):
    """Simple LoRA adapter: low-rank update for a vector of dim `dim`.

    y = x + scale * U(V(x)),  where V: dim->rank, U: rank->dim.
    """

    def __init__(self, dim: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return x
        delta = self.up(self.down(x)) * (self.alpha / self.rank)
        return x + delta

    def try_1(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return x
        delta = self.up(self.down(x)) * (self.alpha / self.rank)
        return delta

class GS_Hand_3DRenderer(nn.Module):
    def __init__(
            self,
            pcl_embed,
            feat_dim,
            query_dim,
            use_rgb,
            sh_degree,
            xyz_offset_max_step,
            mlp_network_config,
            clip_scaling=0.2,
            decoder_mlp=False,
            skip_decoder=True,
            fix_opacity=False,
            fix_rotation=False,
            decode_with_extra_info=None,
            gradient_checkpointing=False,
            apply_pose_blendshape=False,
                feature_map=False,
                dataset_center_add_map=None,
                use_lora=False,
            ):

        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder
        self.query_dim = query_dim

        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree

        self.mano_model = MANOVoxelMeshModel()

        self.mano_models = {}

        if dataset_center_add_map is None:
            self.dataset_center_add_map = {0: True, 1: True, 2: False}
        else:
            self.dataset_center_add_map = dataset_center_add_map

        if not self.skip_decoder:
            self.pcl_embed = PointEmbed(dim=query_dim)
            self.decoder_cross_attn = DecoderCrossAttn(
                query_dim=query_dim,
                context_dim=feat_dim,
                num_heads=1,
                mlp=decoder_mlp,
                decode_with_extra_info=decode_with_extra_info,
            )

        self.mlp_network_config = mlp_network_config
        self.use_lora = use_lora

        if self.mlp_network_config is not None:
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config).to('cuda')

            if self.use_lora:
                self.lora_mlp = LoRAAdapter(query_dim, rank=16, alpha=1.0).to('cuda')
            else:
                self.lora_mlp = None
        else:
            self.mlp_net = None
            self.lora_mlp = None

        self.feature_map = feature_map

        if feature_map:
            raise ValueError('feature_map refinement is not part of the OASIS hand runtime')
        self.neural_refiner = None

        self.edit_mask_mode = False
        self.edit_vis_mask = None
        self.edit_invisible_cache = None

        self.edit_vis_prob = None
        self.edit_soft_vis_blend = False
        self.edit_vis_prob_low = 0.2
        self.edit_vis_prob_high = 0.8

        self.gs_net = GSLayer_Hand(
            pcl_embed,
            self.mano_model.mano,
            in_channels=query_dim,
            use_rgb=use_rgb,
            sh_degree=self.sh_degree,
            clip_scaling=clip_scaling,
            init_scaling=-5.0,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=xyz_offset_max_step,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            use_fine_feat=(
                True
                if decode_with_extra_info is not None
                   and decode_with_extra_info["type"] is not None
                else False
            ),
            feature_map=self.feature_map,
        )

        if self.use_lora:
            self.lora_gs = LoRAAdapter(query_dim, rank=16, alpha=1.0).to('cuda')
        else:
            self.lora_gs = None

        self.renderer_mesh = Renderer_mesh()

        self.quat_helper = PerVertQuaternion(self.gs_net.mano_verts.to('cuda'), self.gs_net.mano_face.to('cuda')).to('cuda')

    def hyper_step(self, step):
        self.gs_net.hyper_step(step)

    def ensure_lora_adapters(self, rank=16, alpha=1.0):
        """
        Dynamically create LoRA adapters if they don't exist.
        This is useful for finetuning when the model was originally trained without LoRA.

        Args:
            rank: LoRA rank parameter
            alpha: LoRA alpha scaling parameter
        """
        query_dim = getattr(self, 'query_dim', 1024)

        if self.mlp_net is not None and self.lora_mlp is None:
            self.lora_mlp = LoRAAdapter(query_dim, rank=rank, alpha=alpha).to('cuda')
            self.use_lora = True
            print(f"[ensure_lora_adapters] Created lora_mlp with rank={rank}, alpha={alpha}")

        if self.lora_gs is None:
            self.lora_gs = LoRAAdapter(query_dim, rank=rank, alpha=alpha).to('cuda')
            print(f"[ensure_lora_adapters] Created lora_gs with rank={rank}, alpha={alpha}")

    def forward_single_view(
            self,
            gs: GaussianModel,
            viewpoint_camera: Camera,
            background_color: Optional[Float[Tensor, "3"]],
            ret_mask: bool = True,
    ):
        screenspace_points = (torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0)

        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color

        tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
        tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera['height']),

            image_width=int(viewpoint_camera['width']),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera['world_view_transform'],
            projmatrix=viewpoint_camera['full_proj_transform'].float(),

            sh_degree=self.sh_degree,
            campos=viewpoint_camera['camera_center'],

            prefiltered=False,
            debug=False,
            **({"antialiasing": False} if os.environ.get("LHM_USE_DGR32", "0") == "1" else {}),
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = None
            shs = gs.shs.float()

        means3D = torch.nan_to_num(means3D, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=1e-4, posinf=1e-4, neginf=1e-4).clamp_min(1e-6)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=0.0, neginf=0.0)
        opacity = torch.nan_to_num(opacity, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
        if shs is not None:
            shs = torch.nan_to_num(shs, nan=0.0, posinf=0.0, neginf=0.0)
        if colors_precomp is not None:
            colors_precomp = torch.nan_to_num(colors_precomp, nan=0.0, posinf=0.0, neginf=0.0)
        if (
            os.environ.get("LHM_USE_DGR32", "0") == "1"
            and colors_precomp is not None
            and colors_precomp.shape[-1] < 32
        ):
            colors_precomp = F.pad(colors_precomp, (0, 32 - colors_precomp.shape[-1]), value=0.0)

        color_scale = getattr(self, "color_scale", None)
        color_shift = getattr(self, "color_shift", None)
        if not getattr(self, "_color_applied_in_gs", False) and (color_scale is not None or color_shift is not None):
            def _as_rgb_vec(value, device, dtype, default):
                if value is None:
                    return default
                if torch.is_tensor(value):
                    tensor = value.to(device=device, dtype=dtype)
                else:
                    tensor = torch.tensor(value, device=device, dtype=dtype)
                if tensor.numel() == 3:
                    return tensor.view(1, 3)
                return tensor.reshape(-1, 3).mean(dim=0, keepdim=True)

            device = gs.xyz.device
            dtype = (colors_precomp if colors_precomp is not None else shs).dtype
            cs = _as_rgb_vec(color_scale, device, dtype, default=torch.ones(1, 3, device=device, dtype=dtype))
            ct = _as_rgb_vec(color_shift, device, dtype, default=torch.zeros(1, 3, device=device, dtype=dtype))

            if colors_precomp is not None:
                colors_precomp = colors_precomp * cs + ct
            else:
                shs = shs * cs
                shs[:, :1, :] = shs[:, :1, :] + ct.view(1, 1, 3)

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            _n3d = means3D.shape[0]
            for _nm, _t in [('means2D', means2D), ('opacity', opacity), ('scales', scales), ('rotations', rotations)]:
                if _t is not None and _t.shape[0] != _n3d:
                    print(f"\033[91m[SHAPE MISMATCH] means3D={_n3d} but {_nm}={_t.shape}\033[0m")
            if shs is not None and shs.shape[0] != _n3d:
                print(f"\033[91m[SHAPE MISMATCH] means3D={_n3d} but shs={shs.shape}\033[0m")
            if colors_precomp is not None and colors_precomp.shape[0] != _n3d:
                print(f"\033[91m[SHAPE MISMATCH] means3D={_n3d} but colors_precomp={colors_precomp.shape}\033[0m")

            raster_out = rasterizer(
                means3D=means3D.float(),
                means2D=means2D.float(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=cov3D_precomp,
            )
            if len(raster_out) == 4:
                rendered_image, radii, rendered_depth, rendered_alpha = raster_out
            elif len(raster_out) == 3:
                rendered_image, radii, rendered_depth = raster_out
                rendered_alpha = (rendered_depth > 0).to(rendered_image.dtype)
            else:
                rendered_image, radii = raster_out
                rendered_depth = torch.zeros(
                    1, rendered_image.shape[-2], rendered_image.shape[-1],
                    device=rendered_image.device, dtype=rendered_image.dtype
                )
                rendered_alpha = (rendered_image.abs().sum(dim=0, keepdim=True) > 0).to(rendered_image.dtype)
            if rendered_image.shape[0] > 3:
                rendered_image = rendered_image[:3]

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),
            "comp_rgb_bg": bg_color,
            "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),

        }

        return ret

    def forward_single_view_32(
            self,
            gs: GaussianModel32,
            viewpoint_camera: Camera,
            background_color: Optional[Float[Tensor, "3"]],
            ret_mask: bool = True,
    ):
        screenspace_points = (torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0)

        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color

        tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
        tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)

        raster_settings = GaussianRasterizationSettings_32(
            image_height=int(viewpoint_camera['height']),

            image_width=int(viewpoint_camera['width']),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera['world_view_transform'],
            projmatrix=viewpoint_camera['full_proj_transform'].float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera['camera_center'],

            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

        rasterizer = GaussianRasterizer_32(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = gs.shs.float()
            shs = None

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth = rasterizer(
                means3D=means3D.float(),
                means2D=means2D.float(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=cov3D_precomp,
            )

        raw_images = rendered_image.unsqueeze(0)
        rendered_image = self.neural_refiner(raw_images)

        ret = {
            "comp_rgb": rendered_image.squeeze().permute(1,2,0),
            "comp_rgb_bg": bg_color,

            "comp_depth": rendered_depth.permute(1, 2, 0),
        }

        return ret

    def get_mano_model(self, dataset_id):
        """Return a MANOVoxelMeshModel instance for the given dataset_id.
        If dataset_id is None or not provided, return the default `self.mano_model`.
        """
        if dataset_id is None:
            return self.mano_model

        try:
            if hasattr(dataset_id, "item"):
                dataset_id = int(dataset_id.item())
            else:
                dataset_id = int(dataset_id)
        except Exception:
            return self.mano_model

        if dataset_id in self.mano_models:
            return self.mano_models[dataset_id]

        center_add = self.dataset_center_add_map.get(dataset_id, True)
        model = MANOVoxelMeshModel(center_add=center_add)
        self.mano_models[dataset_id] = model
        return model

    def calc_face_areas(self, mesh_verts, mesh_faces):
        vertices_faces = mesh_verts[mesh_faces]

        faces_normals = torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        )

        face_areas = faces_normals.norm(dim=-1, keepdim=True) / 2.0
        return face_areas

    def animate_gs_model(
            self, gs_attr: GaussianAppOutput, gs_densify_attr, query_points,
            smplx_data, dataset_id=None, debug=False,
    ):
        """
        query_points: [N, 3]
        """

        device = gs_attr.offset_xyz.device

        cano_smplx_data_keys = [

            "shape",
            "poses",

        ]

        merge_smplx_data = dict()
        for cano_smplx_data_key in cano_smplx_data_keys:
            warp_data = smplx_data[cano_smplx_data_key]
            cano_pose = torch.zeros_like(warp_data)
            merge_pose = torch.cat([warp_data, cano_pose], dim=0)
            merge_smplx_data[cano_smplx_data_key] = merge_pose

        merge_smplx_data["shape"] = smplx_data["shape"]

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            mean_3d = (
                    query_points + gs_attr.offset_xyz
            )

            num_view = merge_smplx_data["poses"].shape[0]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)

            mano_for_dataset = self.get_mano_model(dataset_id)
            mean_3d, transform_matrix = (
                mano_for_dataset.transform_to_posed_verts_from_neutral_pose(
                    mean_3d,
                    smplx_data,

                )
            )

            _, N, _, _ = transform_matrix.shape
            transform_rotation = transform_matrix[:, :, :3, :3]

            rigid_rotation_matrix = torch.nn.functional.normalize(
                matrix_to_quaternion(transform_rotation), dim=-1
            )

            rotation_neutral_pose = gs_attr.rotation.unsqueeze(0).repeat(num_view, 1, 1)

            rotation_pose_verts = quaternion_multiply(
                rigid_rotation_matrix, rotation_neutral_pose
            )

            densify_mask = (gs_densify_attr.activation>=0.5).squeeze(-1)

            triangle_verts = mean_3d[1, :][self.gs_net.mano_face]

            densify_gau_mean = torch.einsum('nij, ni->nj', triangle_verts, gs_densify_attr.bary)[densify_mask, :]
            per_vert_quat = self.quat_helper(mean_3d[1, :])
            tri_quats = per_vert_quat[self.gs_net.mano_face]
            densify_gau_rot = torch.nn.functional.normalize(torch.einsum('bij, bi->bj', tri_quats.squeeze(), gs_densify_attr.bary))

            densify_gau_rot = torch.nn.functional.normalize(quaternion_multiply(densify_gau_rot, gs_densify_attr.rotation))

            densify_gau_scaling_alter = self.quat_helper.calc_face_area_change(smplx_data["posed_verts"].squeeze())
            mean_val = densify_gau_scaling_alter.mean()
            if not torch.isfinite(mean_val):
                mean_val = torch.tensor(1.0, device=mean_val.device, dtype=mean_val.dtype)
            mean_val = mean_val.clamp_min(1e-6)
            densify_gau_scaling_alter = densify_gau_scaling_alter / mean_val
            densify_gau_scaling_alter = torch.nan_to_num(
                densify_gau_scaling_alter,
                nan=1.0,
                posinf=1.0,
                neginf=1.0,
            )

            densify_gau_scaling = (gs_densify_attr.scaling * densify_gau_scaling_alter)[densify_mask, :]

            knn = knn_points(densify_gau_mean.unsqueeze(0), mean_3d[1, :].unsqueeze(0), K=1)
            knn_idx = knn.idx.squeeze()

            densify_labels = self.gs_net.objects_dc[knn_idx]

        gs_list = []
        cano_gs_list = []
        for i in range(num_view):
            gs_copy = GaussianModel(

                xyz=mean_3d[i],
                opacity=gs_attr.opacity,

                rotation=rotation_pose_verts[i],
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                labels=self.gs_net.objects_dc,
                use_rgb=self.gs_net.use_rgb,

                xyz_densify=densify_gau_mean,
                opacity_densify=gs_densify_attr.opacity[densify_mask, :],
                rotation_densify=densify_gau_rot[densify_mask, :],
                scaling_densify=densify_gau_scaling,
                shs_densify=gs_densify_attr.shs[densify_mask, :],
                labels_densify=densify_labels,

                skip_training_setup=True,
            )

            if i == num_view - 2:
                cano_gs_list.append(gs_copy)
            else:
                gs_list.append(gs_copy)

        return gs_list, cano_gs_list

    def forward_gs_attr(self, x, query_points, global_feature=None,
                        batch=None, verts_cam=None, nail_image=None, debug=False, x_fine=None, vis_prob=None):
        device = x.device

        if vis_prob is not None:
            self.edit_vis_prob = vis_prob

        if not getattr(self, "edit_mask_mode", False) or getattr(self, "edit_vis_mask", None) is None:
            self.edit_invisible_cache = None

        batched = (x.dim() == 3) or (query_points is not None and query_points.dim() == 3)

        if self.mlp_network_config is not None and self.mlp_net is not None:
            if batched:
                B = x.shape[0]
                x_flat = x.reshape(-1, x.shape[-1])
                x_ori = x_flat
                x_flat = self.mlp_net(x_flat)

                if getattr(self, "edit_mask_mode", False) and torch.is_tensor(getattr(self, "edit_vis_mask", None)):
                    vis_mask = self.edit_vis_mask
                    mask_flat = (vis_mask > 0.5).reshape(-1, 1).float().to(x_flat.device)

                    weight_flat = mask_flat
                    if torch.is_tensor(getattr(self, "edit_vis_prob", None)):
                        vis_prob = self.edit_vis_prob
                        prob_flat = vis_prob.reshape(-1, 1).to(x_flat.device)
                        low, high = float(self.edit_vis_prob_low), float(self.edit_vis_prob_high)
                        soft = ((prob_flat - low) / max(high - low, 1e-6)).clamp(0.0, 1.0)
                        weight_flat = mask_flat * soft

                    inv_weight_flat = 1.0 - weight_flat

                    if self.edit_invisible_cache is None or not isinstance(self.edit_invisible_cache, dict):
                        self.edit_invisible_cache = {}
                    if "coarse_mlp" not in self.edit_invisible_cache or self.edit_invisible_cache["coarse_mlp"].shape != x_flat.shape:
                        self.edit_invisible_cache["coarse_mlp"] = x_flat.detach()
                    cached_mlp = self.edit_invisible_cache["coarse_mlp"].to(x_flat.device)

                    if getattr(self, "lora_mlp", None) is not None:
                        x_flat_lora = self.lora_mlp(x_flat)
                        if "coarse_lora" not in self.edit_invisible_cache or self.edit_invisible_cache["coarse_lora"].shape != x_flat.shape:
                            with torch.no_grad():
                                self.edit_invisible_cache["coarse_lora"] = self.lora_mlp(cached_mlp).detach()
                        cached_after = self.edit_invisible_cache["coarse_lora"].to(x_flat.device)
                        x_flat = x_flat_lora * weight_flat + cached_after * inv_weight_flat
                    else:
                        x_flat = x_flat * weight_flat + cached_mlp * inv_weight_flat
                else:
                    if getattr(self, "lora_mlp", None) is not None:
                        x_lora = self.lora_mlp.try_1(x_ori)
                        x_flat = x_lora + x_flat

                x = x_flat.view(B, x.shape[1], -1)
            else:
                x = self.mlp_net(x)

                if getattr(self, "edit_mask_mode", False) and torch.is_tensor(getattr(self, "edit_vis_mask", None)):
                    vis_mask = self.edit_vis_mask.squeeze(0) if self.edit_vis_mask.dim() > 1 else self.edit_vis_mask
                    mask_2d = (vis_mask > 0.5).unsqueeze(-1).float().to(x.device)
                    inv_mask_2d = 1.0 - mask_2d

                    if self.edit_invisible_cache is None or not isinstance(self.edit_invisible_cache, dict):
                        self.edit_invisible_cache = {}
                    if "coarse_nb_mlp" not in self.edit_invisible_cache or self.edit_invisible_cache["coarse_nb_mlp"].shape != x.shape:
                        self.edit_invisible_cache["coarse_nb_mlp"] = x.detach()
                    cached_mlp = self.edit_invisible_cache["coarse_nb_mlp"].to(x.device)

                    if getattr(self, "lora_mlp", None) is not None:
                        x_lora = self.lora_mlp(x)
                        if "coarse_nb_lora" not in self.edit_invisible_cache or self.edit_invisible_cache["coarse_nb_lora"].shape != x.shape:
                            with torch.no_grad():
                                self.edit_invisible_cache["coarse_nb_lora"] = self.lora_mlp(cached_mlp).detach()
                        cached_after = self.edit_invisible_cache["coarse_nb_lora"].to(x.device)
                        x = x_lora * mask_2d + cached_after * inv_mask_2d
                    else:
                        x = x * mask_2d + cached_mlp * inv_mask_2d
                else:
                    if getattr(self, "lora_mlp", None) is not None:
                        x = self.lora_mlp(x)

                if x_fine is not None:
                    x_fine = self.mlp_net(x_fine)

                    if getattr(self, "edit_mask_mode", False) and torch.is_tensor(getattr(self, "edit_vis_mask", None)):
                        vis_mask = self.edit_vis_mask.squeeze(0) if self.edit_vis_mask.dim() > 1 else self.edit_vis_mask
                        mask_2d = (vis_mask > 0.5).unsqueeze(-1).float().to(x_fine.device)
                        inv_mask_2d = 1.0 - mask_2d

                        if self.edit_invisible_cache is None or not isinstance(self.edit_invisible_cache, dict):
                            self.edit_invisible_cache = {}
                        if "fine_nb_mlp" not in self.edit_invisible_cache or self.edit_invisible_cache["fine_nb_mlp"].shape != x_fine.shape:
                            self.edit_invisible_cache["fine_nb_mlp"] = x_fine.detach()
                        cached_mlp = self.edit_invisible_cache["fine_nb_mlp"].to(x_fine.device)

                        if getattr(self, "lora_mlp", None) is not None:
                            x_fine_lora = self.lora_mlp(x_fine)
                            if "fine_nb_lora" not in self.edit_invisible_cache or self.edit_invisible_cache["fine_nb_lora"].shape != x_fine.shape:
                                with torch.no_grad():
                                    self.edit_invisible_cache["fine_nb_lora"] = self.lora_mlp(cached_mlp).detach()
                            cached_after = self.edit_invisible_cache["fine_nb_lora"].to(x_fine.device)
                            x_fine = x_fine_lora * mask_2d + cached_after * inv_mask_2d
                        else:
                            x_fine = x_fine * mask_2d + cached_mlp * inv_mask_2d
                    else:
                        if getattr(self, "lora_mlp", None) is not None:
                            x_fine = self.lora_mlp(x_fine)

        constrain_dict, constrain_head = None, None
        film_param = None
        visible_mask = None

        if not batched:
            if getattr(self, "lora_gs", None) is not None:
                x = self.lora_gs(x)
                if x_fine is not None:
                    x_fine = self.lora_gs(x_fine)

            gs_attr, gs_density_attr = self.gs_net(
                x, query_points, x_fine, constrain_dict, constrain_head, global_feature, film_param, visible_mask
            )
            return gs_attr, gs_density_attr, visible_mask

        B, N, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        pts_flat = query_points.reshape(-1, query_points.shape[-1])
        x_fine_flat = None
        if x_fine is not None:
            x_fine_flat = x_fine.reshape(-1, x_fine.shape[-1])

        if getattr(self, "lora_gs", None) is not None:
            x_flat = self.lora_gs(x_flat)
            if x_fine_flat is not None:
                x_fine_flat = self.lora_gs(x_fine_flat)

        global_feat_flat = None
        if global_feature is not None:
            try:
                if global_feature.dim() == 2 and global_feature.shape[0] == B:
                    global_feat_flat = global_feature.repeat_interleave(N, dim=0)
                else:
                    global_feat_flat = global_feature
            except Exception:
                global_feat_flat = global_feature

        gs_attr_flat, gs_density_flat = self.gs_net(
            x_flat, pts_flat, x_fine_flat, constrain_dict, constrain_head, global_feat_flat, film_param, visible_mask
        )

        batched_gs = {}
        for name, val in gs_attr_flat.__dict__.items():
            if torch.is_tensor(val) and val.shape[0] == B * N:
                new_shape = (B, N) + tuple(val.shape[1:])
                batched_gs[name] = val.view(*new_shape)
            else:
                batched_gs[name] = val

        batched_densify = {}
        for name, val in gs_density_flat.__dict__.items():
            if torch.is_tensor(val) and hasattr(self.gs_net, 'mano_face') and val.shape[0] == B * (self.gs_net.mano_face.shape[0]):
                F = self.gs_net.mano_face.shape[0]
                batched_densify[name] = val.view(B, F, *val.shape[1:])
            elif torch.is_tensor(val) and val.shape[0] == B * N:
                batched_densify[name] = val.view(B, N, *val.shape[1:])
            else:
                batched_densify[name] = val

        return GaussianAppOutput(**batched_gs), GaussianDensifyOutput(**batched_densify), visible_mask

    def get_query_points(self, smplx_data, device):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                positions, _, transform_mat_neutral_pose = (
                    self.mano_model.get_query_points(smplx_data, device=device)
                )
        smplx_data["transform_mat_neutral_pose"] = (
            transform_mat_neutral_pose
        )
        return positions, smplx_data

    def decoder_cross_attn_wrapper(self, pcl_embed, latent_feat, extra_info):
        gs_feats = self.decoder_cross_attn(
            pcl_embed.to(dtype=latent_feat.dtype), latent_feat, extra_info
        )
        return gs_feats

    def query_latent_feat(
            self,
            positions: Float[Tensor, "*B N1 3"],

            latent_feat: Float[Tensor, "*B N2 C"],
            extra_info,
    ):
        device = latent_feat.device
        if self.skip_decoder:
            gs_feats = latent_feat
            assert positions is not None
        else:
            assert positions is None
            if positions is None:
                positions, smplx_data = self.get_query_points(smplx_data, device)

            with torch.autocast(device_type=device.type, dtype=torch.float32):
                pcl_embed = self.pcl_embed(positions)

            gs_feats = self.decoder_cross_attn_wrapper(
                pcl_embed, latent_feat, extra_info
            )

        return gs_feats, positions,

    def forward_single_batch(
            self,
            gs_list: list[GaussianModel32],
            batch,

            height: int,
            width: int,
            background_color: Optional[Float[Tensor, "Nv 3"]],
            offset_xyz: Optional[Float[Tensor, "Nv 3"]] = None,
            debug: bool = False,
    ):
        out_list = []
        self.device = gs_list[0].xyz.device

        render_elapsed_time = 0.0
        for v_idx in range(len(gs_list)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            render_start = time.perf_counter()

            out_list.append(
                self.forward_single_view(

                    gs_list[0],
                    batch,

                    background_color,
                )
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            render_elapsed_time += time.perf_counter() - render_start

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}

        out["scaling"] = gs_list[0].scaling
        out['offset'] = offset_xyz
        out['shs'] = gs_list[0].shs
        out['nail_3d_mask'] = gs_list[0].nail_mask

        out['render_time'] = render_elapsed_time
        out['render_views'] = len(gs_list)

        return out

    def get_single_batch_smpl_data(self, smpl_data, bidx):
        smpl_data_single_batch = {}
        for k, v in smpl_data.items():
            smpl_data_single_batch[k] = v[
                bidx
            ]
            if k == "betas" or (k == "joint_offset") or (k == "face_offset"):
                smpl_data_single_batch[k] = v[
                                            bidx: bidx + 1
                                            ]
        return smpl_data_single_batch

    def get_single_view_cam(self, cam_data, bidx):
        cam_data_single_view = {}

        for k, v in cam_data.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    raise ValueError(f"[{k}] got scalar tensor, expect batch dim, v={v}")

                if not (0 <= bidx < v.size(0)):
                    raise IndexError(
                        f"[{k}] bidx={bidx} out of range for v.shape={tuple(v.shape)}"
                    )

                try:
                    cam_data_single_view[k] = v[bidx].to('cuda')
                except RuntimeError as e:
                    print("\n[CAM ERROR] key:", k)
                    print("  type:", type(v))
                    print("  shape:", v.shape)
                    print("  device:", v.device)
                    print("  bidx:", bidx)
                    print("  exception:", e)
                    print()
                    raise
            elif isinstance(v, list):
                if not (0 <= bidx < len(v)):
                    raise IndexError(
                        f"[{k}] list index out of range: bidx={bidx}, len={len(v)}"
                    )
                cam_data_single_view[k] = v[bidx]
            else:
                if k in ("dataset_id", "dataset_name"):
                    cam_data_single_view[k] = v
                else:
                    continue

        return cam_data_single_view

    def get_single_view_smpl_data(self, smpl_data, vidx):
        smpl_data_single_view = {}
        for k, v in smpl_data.items():
            smpl_data_single_view[k] = v[vidx: vidx + 1, ...]

        return smpl_data_single_view

    def forward_gs(
            self,
            gs_hidden_features: Float[Tensor, "B Np Cp"],
            query_points: Float[Tensor, "B Np_q 3"],
            global_feature: Float[Tensor, "B C"],
            batches,
            verts_cam,
            nail_image,

            additional_features: Optional[dict] = None,
            color_bias: Optional[Float[Tensor, "B N 3"]] = None,
            opacity_bias: Optional[Float[Tensor, "B N 1"]] = None,
            vis_prob: Optional[Float[Tensor, "B Np_q"]] = None,
            debug: bool = False,
            **kwargs,
    ):
        batch_size = gs_hidden_features.shape[0]

        query_gs_features, query_points = self.query_latent_feat(
            query_points, gs_hidden_features, additional_features
        )

        gs_attr_list = []
        densify_gs_attr_list = []

        try:
            batched_out = self.forward_gs_attr(
                query_gs_features, query_points, global_feature, batches, verts_cam, nail_image, debug, vis_prob=vis_prob
            )

            if isinstance(batched_out, tuple) and len(batched_out) >= 2:
                gs_attr_batch, densify_batch = batched_out[0], batched_out[1]

                if hasattr(gs_attr_batch, "__dict__"):
                    for b in range(batch_size):
                        kwargs = {}
                        for name, val in gs_attr_batch.__dict__.items():
                            if torch.is_tensor(val) and val.shape[0] == batch_size:
                                kwargs[name] = val[b]
                            else:
                                try:
                                    kwargs[name] = val[b]
                                except Exception:
                                    kwargs[name] = val
                        gs_attr_list.append(GaussianAppOutput(**kwargs))
                else:
                    try:
                        for b in range(batch_size):
                            gs_attr_list.append(gs_attr_batch[b])
                    except Exception:
                        raise RuntimeError("Batched forward_gs_attr returned unexpected format")

                if hasattr(densify_batch, "__dict__"):
                    for b in range(batch_size):
                        kwargs = {}
                        for name, val in densify_batch.__dict__.items():
                            if torch.is_tensor(val) and val.shape[0] == batch_size:
                                kwargs[name] = val[b]
                            else:
                                try:
                                    kwargs[name] = val[b]
                                except Exception:
                                    kwargs[name] = val
                        densify_gs_attr_list.append(GaussianDensifyOutput(**kwargs))
                else:
                    try:
                        for b in range(batch_size):
                            densify_gs_attr_list.append(densify_batch[b])
                    except Exception:
                        raise RuntimeError("Batched forward_gs_attr returned unexpected format for densify")
            else:
                raise RuntimeError("Batched forward_gs_attr returned unexpected format")
        except Exception:
            for b in range(batch_size):
                if batch_size == 1:
                    gs_attr, densify_gs_attr, vis_mask = self.forward_gs_attr(
                        query_gs_features[b], query_points[b], global_feature[b],
                        batches, verts_cam[b, :, 2], nail_image[b, ...], debug,
                    )
                else:
                    gs_attr, densify_gs_attr, vis_mask = self.forward_gs_attr(
                        query_gs_features[b], query_points[b], global_feature[b],
                        batches[b], verts_cam[b, :, 2], nail_image[b, ...], debug,
                    )
                gs_attr_list.append(gs_attr)
                densify_gs_attr_list.append(densify_gs_attr)

        if color_bias is not None or opacity_bias is not None:
            for b_idx, gs_attr in enumerate(gs_attr_list):
                cb = None
                ob = None
                if color_bias is not None:
                    cb = color_bias
                    if cb.dim() == 3:
                        if cb.shape[0] == len(gs_attr_list):
                            cb = cb[b_idx]
                        elif cb.shape[0] == 1:
                            cb = cb[0]
                    cb = cb.to(gs_attr.shs.device, gs_attr.shs.dtype)
                    if cb.dim() == 2:
                        cb = cb.unsqueeze(1)
                    gs_attr.shs = gs_attr.shs + cb
                if opacity_bias is not None:
                    ob = opacity_bias
                    if ob.dim() == 3:
                        if ob.shape[0] == len(gs_attr_list):
                            ob = ob[b_idx]
                        elif ob.shape[0] == 1:
                            ob = ob[0]
                    ob = ob.to(gs_attr.opacity.device, gs_attr.opacity.dtype)
                    if ob.dim() == 1:
                        ob = ob.unsqueeze(-1)
                    gs_attr.opacity = gs_attr.opacity + ob

        self._color_applied_in_gs = False
        color_shift = getattr(self, "color_shift", None)
        color_scale = getattr(self, "color_scale", None)

        if color_shift is not None or color_scale is not None:
            def _normalize_global_color_param(param, ref):
                if param is None:
                    return None
                p = param.to(ref.device, ref.dtype)
                if p.dim() >= 4:
                    p = p.view(-1, p.shape[-1])
                if p.dim() == 2 and p.shape[0] == 1:
                    p = p[0]
                if p.dim() == 1:
                    p = p.view(1, 1, 3)
                elif p.dim() == 2 and p.shape[-1] == 3:
                    p = p.unsqueeze(1)
                return p

            for gs_attr in gs_attr_list:
                if not hasattr(gs_attr, "shs") or gs_attr.shs is None:
                    continue
                if gs_attr.shs.shape[-1] != 3:
                    continue
                shift = _normalize_global_color_param(color_shift, gs_attr.shs)
                scale = _normalize_global_color_param(color_scale, gs_attr.shs)
                if scale is None:
                    scale = 1.0
                if shift is None:
                    shift = 0.0
                shs_scaled = gs_attr.shs * scale
                shs_dc = shs_scaled[:, :1, :] + shift
                if shs_scaled.shape[1] > 1:
                    gs_attr.shs = torch.cat([shs_dc, shs_scaled[:, 1:, :]], dim=1)
                else:
                    gs_attr.shs = shs_dc

            for densify_attr in densify_gs_attr_list:
                if not hasattr(densify_attr, "shs") or densify_attr.shs is None:
                    continue
                if densify_attr.shs.shape[-1] != 3:
                    continue
                shift = _normalize_global_color_param(color_shift, densify_attr.shs)
                scale = _normalize_global_color_param(color_scale, densify_attr.shs)
                if scale is None:
                    scale = 1.0
                if shift is None:
                    shift = 0.0
                shs_scaled = densify_attr.shs * scale
                shs_dc = shs_scaled[:, :1, :] + shift
                if shs_scaled.shape[1] > 1:
                    densify_attr.shs = torch.cat([shs_dc, shs_scaled[:, 1:, :]], dim=1)
                else:
                    densify_attr.shs = shs_dc

            self._color_applied_in_gs = True

        if hasattr(self, "gs_net") and self.gs_net is not None:
            for b_idx, gs_attr in enumerate(gs_attr_list):
                if not hasattr(gs_attr, "shs") or gs_attr.shs is None or gs_attr.shs.shape[-1] != 3:
                    continue

                pts_b = query_points[b_idx] if torch.is_tensor(query_points) and query_points.dim() == 3 else query_points
                densify_attr = densify_gs_attr_list[b_idx] if b_idx < len(densify_gs_attr_list) else None
                base_delta, densify_delta = self.gs_net.compute_stage1_color_offsets(
                    pts_b,
                    base_shs=gs_attr.shs,
                    densify_shs=getattr(densify_attr, "shs", None),
                )

                if base_delta is not None:
                    delta = base_delta.to(gs_attr.shs.device, gs_attr.shs.dtype)
                    shs_dc = gs_attr.shs[:, :1, :] + delta
                    if gs_attr.shs.shape[1] > 1:
                        gs_attr.shs = torch.cat([shs_dc, gs_attr.shs[:, 1:, :]], dim=1)
                    else:
                        gs_attr.shs = shs_dc

                if densify_delta is not None and densify_attr is not None:
                    if hasattr(densify_attr, "shs") and densify_attr.shs is not None and densify_attr.shs.shape[-1] == 3:
                        delta = densify_delta.to(densify_attr.shs.device, densify_attr.shs.dtype)
                        shs_dc = densify_attr.shs[:, :1, :] + delta
                        if densify_attr.shs.shape[1] > 1:
                            densify_attr.shs = torch.cat([shs_dc, densify_attr.shs[:, 1:, :]], dim=1)
                        else:
                            densify_attr.shs = shs_dc

        return gs_attr_list, densify_gs_attr_list, query_points

    def forward_animate_gs(
            self,
            gs_attr_list,
            gs_densify_attr_list,
            query_points,

            batch,
            smplx_data,

            height,
            width,
            background_color,

            debug=False,
            df_data=None,
    ):
        batch_size = len(gs_attr_list)
        out_list = []
        cano_out_list = []

        N_view = smplx_data["poses"].shape[0]

        gs_attr = gs_attr_list
        query_pt = query_points

        dataset_id = None
        try:
            if isinstance(batch, dict) and "dataset_id" in batch:
                val = batch["dataset_id"]

                if isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        dataset_id = int(val.item())
                    else:
                        dataset_id = int(val[0].item())
                elif isinstance(val, (list, tuple)):
                    dataset_id = int(val[0])
                else:
                    dataset_id = int(val)
        except Exception:
            dataset_id = None

        merge_animatable_gs_model_list, cano_gs_model_list = self.animate_gs_model(
            gs_attr,
            gs_densify_attr_list,
            query_pt,
            smplx_data,
            dataset_id=dataset_id,
            debug=debug,
        )

        animatable_gs_model_list = merge_animatable_gs_model_list[:N_view]

        out_list.append(
            self.forward_single_batch(
                animatable_gs_model_list,
                batch,

                height,
                width,
                background_color,
                gs_attr.offset_xyz,
                debug=debug,
            )
        )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v[0] if len(v) > 0 else v

        out["comp_rgb"] = out["comp_rgb"].permute(
            0, 1, 4, 2, 3
        )

        out["comp_depth"] = out["comp_depth"].permute(
            0, 1, 4, 2, 3
        )

        return out

    def forward(
            self,
            gs_hidden_features: Float[Tensor, "B Np Cp"],
            query_points: Float[Tensor, "B Np 3"],
            smplx_data,
            c2w: Float[Tensor, "B Nv 4 4"],
            intrinsic: Float[Tensor, "B Nv 4 4"],
            height,
            width,
            additional_features: Optional[Float[Tensor, "B C H W"]] = None,
            background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
            debug: bool = False,
            **kwargs,
    ):
        gs_attr_list, query_points, smplx_data = self.forward_gs(
            gs_hidden_features,
            query_points,
            smplx_data=smplx_data,
            additional_features=additional_features,
            debug=debug,
        )

        out = self.forward_animate_gs(
            gs_attr_list,
            query_points,
            smplx_data,
            c2w,
            intrinsic,
            height,
            width,
            background_color,
            debug,
            df_data=kwargs["df_data"],
        )
        out["gs_attr"] = gs_attr_list

        return out
