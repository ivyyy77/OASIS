import copy
import math
import os
import pdb
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
# import faiss
# from knn_cuda import KNN
from data.interhand.train import Renderer_mesh
# from diff_gaussian_rasterization_32 import GaussianRasterizer_32
# from diff_gaussian_rasterization_32 import GaussianRasterizationSettings as GaussianRasterizationSettings_32
# import diff_gaussian_rasterization_obj as dgro
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply
# from knn_cuda import KNN
from pytorch3d.ops import knn_points
import torch.nn.functional as F

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from LHM.models.rendering.smpl_x import SMPLXModel, read_smplx_param
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
    # xyzs_cam = cam_T_world(xyzs_world, E) # xyzs_world (1,3,778)
    xyzs_cam = xyzs_world  # xyzs_world (1,3,778)
    xys_2d = img_T_cam(xyzs_cam, K)

    # normalize to NDC space. flip xy because the ndc coord definition
    # IMPORTANT: CHECK THE DEFINITION OF NDC
    # if H < W:
    #     xs = -((xys_2d[:, 0, :] / H) * 2. - (W / H))
    #     ys = -((xys_2d[:, 1, :] / H) * 2. - 1.)
    # else:
    #     xs = -((xys_2d[:, 0, :] / W) * 2. - 1.)
    #     ys = -((xys_2d[:, 1, :] / W) * 2. - (H / W))
    xs = ((xys_2d[:, 0, :] / W) * 2. - 1.)
    ys = ((xys_2d[:, 1, :] / W) * 2. - (H / W))
    zs = xyzs_cam[:, 2]  # 1,12337
    xyzs_ndc = torch.stack([xs, ys, zs], dim=-1)   # 1,12337,3
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


def cam_T_world(xyzs_world, E):  # xyzs_world: 1,3,12337    E:4,4  both tensor
    E=E.unsqueeze(0).float()
    xyzs_world_ = torch.cat([xyzs_world, torch.ones_like(xyzs_world[:, :1])], dim=1)   # xyzs_world (1,3,778) --> (1,4,778)
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

    # 分离 NDC 坐标和深度
    x_ndc, y_ndc, z_cam = xyzs_ndc[..., 0], xyzs_ndc[..., 1], xyzs_ndc[..., 2]

    # —— 1. NDC → 像素坐标 (u, v) ——
    # 注意：和正向时条件分支相反，需要对公式做反解
    if H < W:
        # 正向： x_ndc = (W - 2*u) / H  =>  u = (W - x_ndc * H) / 2
        u = (W - x_ndc * H) / 2
        # 正向： y_ndc = 1 - 2*v / H      =>  v = H * (1 - y_ndc) / 2
        v = H * (1 - y_ndc) / 2
    else:
        # 正向： x_ndc = 1 - 2*u / W     =>  u = W * (1 - x_ndc) / 2
        u = W * (1 - x_ndc) / 2
        # 正向： y_ndc = (H - 2*v) / W   =>  v = (H - y_ndc * W) / 2
        v = (H - y_ndc * W) / 2

    # —— 2. 像素坐标 + 深度 → 相机坐标齐次形式 (x_cam, y_cam, z_cam) ——
    # 在正向流程中，图像坐标归一化前为 x_ = K @ [x_cam, y_cam, z_cam]
    # 此时 x_ = [u * z_cam, v * z_cam, z_cam]
    # 拼成齐次向量以便左乘 K⁻¹
    x_h = u * z_cam
    y_h = v * z_cam
    z_h = z_cam
    xyzs_pix = torch.stack([x_h, y_h, z_h], dim=-1)  # (..., 3)

    # 计算 K⁻¹
    K_inv = torch.inverse(K)
    # 如果 xyzs_pix 是 (..., 3)，需要先 reshape 到 (N, 3, 1) 再做批量相乘，这里假设已经是 (N,3)
    # 对一般 (...,3) 的场景，可改写成：
    original_shape = xyzs_pix.shape
    xyzs_flat = xyzs_pix.reshape(-1, 3).unsqueeze(-1)  # (N, 3, 1)
    cam_xyz_flat = torch.bmm(K_inv.unsqueeze(0).expand(xyzs_flat.size(0), -1, -1), xyzs_flat)
    cam_xyz = cam_xyz_flat.squeeze(-1).reshape(original_shape)  # (..., 3)

    # —— 3. 相机坐标 → 世界坐标 ——
    # 外参 E 是 (3×4) 矩阵 [R | t], world_h = [Xw, Yw, Zw, 1] -> cam = E @ world_h
    # 我们需要 E⁻¹ 扩展到 4×4，然后左乘 cam_h = [x_cam, y_cam, z_cam, 1]
    R = E[:3, :3]  # (3,3)
    t = E[:3, 3:]  # (3,1)
    # 构造 4×4 外参齐次矩阵
    E_h = torch.zeros(4, 4, device=E.device, dtype=E.dtype)
    E_h[:3, :3] = R
    E_h[:3, 3:] = t
    E_h[3, 3] = 1.0
    E_h_inv = torch.inverse(E_h)  # (4,4)

    # 将每个 cam_xyz 补齐成齐次坐标 [x, y, z, 1]
    ones = torch.ones(*original_shape[:-1], 1, device=cam_xyz.device, dtype=cam_xyz.dtype)
    cam_xyz_h = torch.cat([cam_xyz, ones], dim=-1)  # (..., 4)

    # 同样 reshape 批量变换
    cam_xyz_h_flat = cam_xyz_h.reshape(-1, 4).unsqueeze(-1)  # (N,4,1)
    world_h_flat = torch.bmm(E_h_inv.unsqueeze(0).expand(cam_xyz_h_flat.size(0), -1, -1), cam_xyz_h_flat)
    world_xyz = world_h_flat.squeeze(-1)[..., :3].reshape(original_shape)  # (..., 3)

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
    # K_ndc = torch.tensor([
    #             [2 * focalx / w, 0, (2 * px - w) / w - 1, 0],
    #             [0, 2 * focaly / h, 1 - (2 * py - h) / h, 0],
    #             [0, 0, (zfar + znear) / (zfar - znear), - 2 * zfar * znear / (zfar - znear)],
    #             [0, 0, 1, 0]
    #         ]).float().to(K.device)
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
    # u, v = W/2, H/2
    # cx = u - cx
    # cy = cy - v
    s = 0 # K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0
    # z_sign = -1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)
    # P[0, 2] = 1 - 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    # # DEBUG
    # P = torch.tensor([
    #     [22.25269874, 0, 0.0993494, 0],
    #     [0, 39.56035442, -0.20949485, 0],
    #     [0, 0, 1.001005, -0.10005003],
    #     [0,0,1,0]
    # ])

    return P


def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y
    # fov_x_ = 2 * math.atan(w / (2 * fx))
    # fov_y_ = 2 * math.atan(h / (2 * fy))
    # return fov_x_, fov_y_


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
        # trans=np.array([-2.6987e-04, 6.5616e-01, 1.7932e+01]),
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
        # self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01
        # self.zfar = 1000
        # self.znear = 0.001

        # self.projection_matrix = (
        #     getProjectionMatrix(
        #         znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        #     )
        #     .transpose(0, 1)
        #     .to(w2c.device)
        # )

        self.projection_matrix = (
            getProjectionMatrix_refine(
                intrinsic, self.height, self.width)
            .transpose(0, 1)
            .to(w2c.device)
        )

        # self.projection_matrix = getProjectionMatrix_new(intrinsic).transpose(0, 1)

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
            # trans=trans,
        )



class Hand_Camera:
    def __init__( self, w2c, intrinsic,
                    FoVx, FoVy, height, width,
                    # trans=np.array([-2.6987e-04, 6.5616e-01, 1.7932e+01]),
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

        # self.world_view_transform = getWorld2View2(w2c, self.trans, self.scale)
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        self.zfar = 100.0
        self.znear = 0.01
        # self.zfar = 1000
        # self.znear = 0.001

        # self.projection_matrix = (
        #     getProjectionMatrix(
        #         znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        #     )
        #     .transpose(0, 1)
        #     .to(w2c.device)
        # )

        self.projection_matrix = (
            getProjectionMatrix_refine(
                intrinsic, self.height, self.width)
            .transpose(0, 1)
            .to(w2c.device)
        )

        # self.projection_matrix = getProjectionMatrix_new(intrinsic).transpose(0, 1)

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

        # rgb activation function
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
        self.shs: Tensor = shs  # [B, SH_Coeff, 3]

        self.use_rgb = False #use_rgb  # shs indicates rgb?

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        # self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.training_setup()

    def get_xyz(self):
        return self.xyz
    #
    def training_setup(self):
        # self.percent_dense = training_args.percent_dense
        self._xyz = nn.Parameter(self.xyz.requires_grad_(True))
        self._features_color = nn.Parameter(self.shs.requires_grad_(True))
        # self._features_dc = nn.Parameter(self.shs[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(self.shs[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(self.scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self.rotation.requires_grad_(True))
        self._opacity = nn.Parameter(self.opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': 0.001, "name": "xyz"},
            {'params': [self._features_color], 'lr': 0.001, "name": "f_color"},
            # {'params': [self._features_dc], 'lr': 0.01, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': 0.01/20, "name": "f_rest"},
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
        # features_dc = self.shs[:, :1]
        # features_rest = self.shs[:, 1:]
        # features_color = self.shs

        # for i in range(features_dc.shape[1] * features_dc.shape[2]):
        #     l.append("f_dc_{}".format(i))
        # for i in range(features_rest.shape[1] * features_rest.shape[2]):
        #     l.append("f_rest_{}".format(i))
        
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

        # features_dc = shs[:, :1]
        # features_rest = shs[:, 1:]
        features_color = shs

        # f_dc = (
        #     features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        # )
        # f_rest = (
        #     features_rest.float()
        #     .detach()
        #     .flatten(start_dim=1)
        #     .contiguous()
        #     .cpu()
        #     .numpy()
        # )
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
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # 0, 3, 8, 15
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

        # rgb activation function
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
        self.shs: Tensor = shs  # [B, SH_Coeff, 3]
        self.object_dc: Tensor = labels
        # self.labels = 
        
        self.xyz: Tensor = torch.cat([xyz, xyz_densify], dim=0)
        self.opacity: Tensor = torch.cat([opacity, opacity_densify], dim=0)
        self.rotation: Tensor = torch.cat([rotation, rotation_densify], dim=0)
        self.scaling: Tensor = torch.cat([scaling, scaling_densify], dim=0)
        # print(shs.shape, shs_densify.shape)
        self.shs: Tensor = torch.cat([shs, shs_densify], dim=0)  # [B, SH_Coeff, 3]
        self.object_dc: Tensor = torch.cat([labels, labels_densify], dim=0)

        # NAIL_PARTS = {
        #     "index3": 4,
        #     "middle3": 7,
        #     "pinky3": 10,
        #     "ring3": 13,
        #     "thumb3": 16
        # }

        # # nail_mask = torch.isin(self.object_dc.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.object_dc.device))
        # nail_mask = torch.isin(labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.object_dc.device))
        # self.nail_mask = nail_mask.cuda()        
        self.nail_mask = labels.cuda()

        self.use_rgb = use_rgb  # shs indicates rgb?

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        # self.percent_dense = 0
        self.spatial_lr_scale = 0
        if not skip_training_setup:
            self.training_setup()

    def get_xyz(self):
        return self.xyz
    #
    def training_setup(self):
        # self.percent_dense = training_args.percent_dense
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
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # 0, 3, 8, 15
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

        self.scaling_activation = trunc_exp  # proposed by torch-ngp
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
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)  # this way
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)
        
        # use_rgb = False   # false
        self.use_rgb = use_rgb   # true
        self.restrict_offset = restrict_offset   # true
        self.xyz_offset = xyz_offset   # true
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32    # 我debug是1.0
        self.fix_opacity = fix_opacity      # False
        self.fix_rotation = fix_rotation      # False
        self.use_fine_feat = use_fine_feat      # False

        # feature_map = True
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
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:   # 这两个 'fix_xx' 都是 false，就走下面的 else 路径
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)     # init_scaling = -5
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))       # init_density = 0.1
            self.out_layers[key] = layer    # 例如，shs的层，输入1024，输出3
            # 输入都是一样的，输出根据高斯属性的需求而来

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def constrain_forward(self, ret, constrain_dict):

        # body scaling constrain
        # gs_attr.scaling[is_constrain_body] = gs_attr.scaling[is_constrain_body].clamp(max=0.02)  # magic number, which is used to constrain 
        # hand opacity constrain 

        # force the hand's opacity to be 0.95
        # gs_attr.opacity[is_hand] = gs_attr.opacity[is_hand].clamp(min=0.95)

        # body scaling constrain
        # is_constrain_body = constrain_dict['is_constrain_body']

        # is_hand = constrain_dict['is_hands']
        # opacity = ret['opacity']
        # opacity[is_hand] = opacity[is_hand].clamp(min=0.95)
        # ret['opacity'] = opacity
        
        is_upper_body = constrain_dict['is_upper_body']
        scaling = ret['scaling'] 
        # scaling[is_constrain_body] body_constrain= scaling[is_constrain_body].clamp(max = 0.02)
        scaling[is_upper_body] = scaling[is_upper_body].clamp(max = 0.02)
        # scaling = scaling.clamp(max=0.02)
        ret['scaling'] = scaling

        return ret


    def constrain_expr(self, ret, constrain_head):
        head_mask = constrain_head['head']  # shape: (N,), bool tensor
        face_mask = constrain_head['face']  # shape: (N,), bool tensor
        xyz = ret['offset_xyz']  # shape: (N, 3), tensor

        # 找出属于 head 但不属于 face 的点
        head_only_mask = head_mask & (~face_mask)
        # head_only_mask = (~face_mask)

        # 对这些点做缩放
        xyz[head_only_mask] = xyz[head_only_mask] * 0.5
        # xyz[head_only_mask] = xyz[head_only_mask] * 1

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
                    )  # constant rotation
                else:
                    # v = torch.nn.functional.normalize(v)
                    v = self.rotation_activation(v)
            elif k == "scaling":
                # v = trunc_exp(v)
                v = self.scaling_activation(v)

                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    # v = torch.sigmoid(v)
                    v = self.opacity_activation(v)
            elif k == "shs":
                if self.use_rgb:
                    # v = torch.sigmoid(v)
                    v = self.rgb_activation(v)
                    # v = torch.reshape(v, (v.shape[0], -1, 3))

                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    v = self.rgb_activation(v)
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
                # v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                # TODO check
                if self.restrict_offset:   # this way
                    max_step = self.xyz_offset_max_step
                    # max_step = 0.5    # DEBUG
                    v = (torch.sigmoid(v) - 0.5) * max_step     # 这个在 gs_renderer.py 1019行，于网格的顶点相加了
                if self.xyz_offset:   # this way
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v

        ret["use_rgb"] = self.use_rgb

        if constrain_dict is not None:
            ret = self.constrain_forward(ret, constrain_dict)     # 限制上半身体素的最大尺寸，防止其变得过大

        if constrain_head is not None:
            ret = self.constrain_expr(ret, constrain_head)

        return GaussianAppOutput(**ret)



class GSLayer_Hand(nn.Module):
    """W/O Activation Function"""

    def setup_functions(self):

        self.scaling_activation = trunc_exp  # proposed by torch-ngp
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
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)  # this way
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)
        # self.clip_scaling = 0.03

        self.use_rgb = use_rgb  # true
        self.restrict_offset = restrict_offset  # true
        self.xyz_offset = xyz_offset  # true
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32    # 我debug是1.0
        self.fix_opacity = fix_opacity  # False
        self.fix_rotation = fix_rotation  # False
        self.use_fine_feat = use_fine_feat  # False

        # self.mano = mano
        self.mano_face = mano.faces_tensor.long()#.to('cuda')
        self.mano_verts = mano.v_template
        self.face_area = calc_face_areas(self.mano_verts, self.mano_face)
        self.mano_face = self.mano_face.to('cuda')

        # K_max = 3
        F_num = self.mano_face.shape[0]
        init_bary = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
        # # self.barycentric_raw = nn.Parameter(init_uv.unsqueeze(0).unsqueeze(0)
        #                             # .repeat(F, K_max, 1) + 0.01*torch.randn(F, K_max, 2))
        # self.barycentric_raw = nn.Parameter(
        #                     init_bary.unsqueeze(0).repeat(F_num, 1), requires_grad=True).to('cuda')   # [F, 3]
        
        self.barycentric_raw = nn.Parameter(
                            torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32).unsqueeze(0).repeat(F_num, 1).cuda(), requires_grad=True)

        # Shared appearance modulation for the color branch.
        # Stage1 optimizes these low-DOF latent gates so visible supervision
        # propagates coherently to invisible regions through the prior features.
        self.color_latent_gamma = nn.Parameter(torch.zeros(1, in_channels, device='cuda'))
        self.color_latent_beta = nn.Parameter(torch.zeros(1, in_channels, device='cuda'))

        # Stage1 canonical color field: a low-frequency RGB residual defined
        # directly in canonical 3D MANO space.
        # Use only the first two Fourier bands, inspired by PointEmbed/NeRF PE,
        # to stay smooth enough for Stage1 color bias correction.
        self.register_buffer(
            "stage1_color_field_freqs",
            torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda'),
        )
        # Scalar spatial field times a shared RGB direction.
        # This keeps Stage1 as an overall color adaptation instead of a
        # per-part semantic recoloring.
        self.stage1_color_coeff = nn.Parameter(torch.zeros(16, 1, device='cuda'))
        self.stage1_color_rgb = nn.Parameter(torch.ones(1, 3, device='cuda'))
        self.stage1_color_field_alpha = 1.0

        # self.norm = AdaLayerNormZero(1024).to('cuda')
        # # self.linear_cond_proj = nn.Linear(1024, 1024)
        # G = 1024
        # self.linear_cond_proj = nn.Sequential( nn.Linear(G, G//2),
        #                                 nn.GELU(),
        #                                 nn.Linear(G//2, G)
        #                             ).to('cuda')
        # # VERY IMPORTANT: zero-init last linear so initial emb = 0
        # nn.init.zeros_(self.linear_cond_proj[-1].weight)
        # nn.init.zeros_(self.linear_cond_proj[-1].bias)

        # self.alpha_prior_raw = nn.Parameter(torch.full((F_num, 1), 0.0) + 0.01 * torch.randn(F_num, 1), requires_grad=True).to('cuda')  
        # self.activate_faces = nn.Parameter(torch.randn(F), requires_grad=True)
        # # learnable prior activations (optional)
        # self.alpha_raw = nn.Parameter(torch.full((F, K_max), -4.0))  # bias -> sigmoid ~ 0.018 (mostly off)

        # self.attn = Attention(
        #     query_dim=1024,
        #     cross_attention_dim=1024,
        #     heads=8,
        #     dim_head=128,
        #     qk_norm='rms_norm',
        #     out_dim=1024,
        #     out_context_dim=1024,
        # ).to('cuda:1')
        
        # self.norm = nn.LayerNorm(1024, elementwise_affine=False, eps=0.0).cuda()
        # # self.norm_2 = nn.LayerNorm(1024, elementwise_affine=False, eps=0.0)
        # self.ff = FeedForward(dim=1024, dim_out=1024, activation_fn="gelu-approximate").cuda()
        # # self.pcl_embed = pcl_embed
        # # self.mlp = nn.Linear(128, 1024)
        # # self.norm = nn.LayerNorm(1024)


        # self.num_layers = 1
        # # create per-layer modules (reuse same hyperparams as original single-layer)
        # self.attns = nn.ModuleList([
        #     Attention(
        #         query_dim=in_channels,
        #         cross_attention_dim=in_channels,
        #         heads=4,
        #         dim_head=128,
        #         qk_norm='rms_norm',
        #         out_dim=in_channels,
        #         out_context_dim=in_channels,
        #     ) for _ in range(self.num_layers)
        # ])
        # self.norms = nn.ModuleList([
        #     nn.LayerNorm(in_channels, elementwise_affine=False, eps=0.0)
        #     for _ in range(self.num_layers)
        # ])
        # self.ffs = nn.ModuleList([
        #     # FeedForward(dim=in_channels, dim_out=in_channels, activation_fn="gelu-approximate")
        #     SwiGLU_FF(dim=in_channels, inner_dim=in_channels*4, dropout=0.0)
        #     for _ in range(self.num_layers)
        # ])


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
                    # shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:  # 这两个 'fix_xx' 都是 false，就走下面的 else 路径
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)  # init_scaling = -5
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))  # init_density = 0.1
            # elif key == "bary":
            #     init_bary = 1/3
            #     nn.init.constant_(layer.bias, init_bary)
            self.out_layers[key] = layer  # 例如，shs的层，输入1024，输出3
            # 输入都是一样的，输出根据高斯属性的需求而来
        self.out_layers = self.out_layers.cuda()

        # for k, dim in self.attr_dict.items():
        #     if k in ["xyz", "scaling", "opacity", "shs"]:  # 这些输出概率
        #         self.out_layers[k] = nn.Linear(hidden_dim, dim * 2)
        #     else:
        #         self.out_layers[k] = nn.Linear(hidden_dim, dim)


        self.densify_out_layers = nn.ModuleDict()
        for key, out_ch in self.densify_attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs" and not use_rgb:
                    out_ch = (sh_degree + 1) ** 2 * 3
                    # shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:  # 这两个 'fix_xx' 都是 false，就走下面的 else 路径
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))  # init_density = 0.1
            elif key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)  # init_scaling = -5
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)            
            elif key == "bary":
                init_bary = 1/3
                nn.init.constant_(layer.bias, init_bary)
            self.densify_out_layers[key] = layer  # 例如，shs的层，输入1024，输出3
            # 输入都是一样的，输出根据高斯属性的需求而来
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
        self.labels = torch.tensor(pc[:, 6].astype(np.uint8)) # (N,)

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
        nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))
        self.nail_mask = nail_mask.cuda()
        # 转成 0/1
        self.objects_dc = nail_mask.to(torch.float32).unsqueeze(1).cuda()

        # self.labels = self.objects_dc.cuda()

        # self.weights = torch.ones_like(self.labels).cuda()
        # self.weights[self.labels == 1] = 0.8

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
            # Remove the constant component; global shift/scale already handles it.
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

        # body scaling constrain
        # gs_attr.scaling[is_constrain_body] = gs_attr.scaling[is_constrain_body].clamp(max=0.02)  # magic number, which is used to constrain
        # hand opacity constrain

        # force the hand's opacity to be 0.95
        # gs_attr.opacity[is_hand] = gs_attr.opacity[is_hand].clamp(min=0.95)

        # body scaling constrain
        # is_constrain_body = constrain_dict['is_constrain_body']

        # is_hand = constrain_dict['is_hands']
        # opacity = ret['opacity']
        # opacity[is_hand] = opacity[is_hand].clamp(min=0.95)
        # ret['opacity'] = opacity

        is_upper_body = constrain_dict['is_upper_body']
        scaling = ret['scaling']
        # scaling[is_constrain_body] body_constrain= scaling[is_constrain_body].clamp(max = 0.02)
        scaling[is_upper_body] = scaling[is_upper_body].clamp(max=0.02)
        # scaling = scaling.clamp(max=0.02)
        ret['scaling'] = scaling

        return ret


    def constrain_expr(self, ret, constrain_head):
        head_mask = constrain_head['head']  # shape: (N,), bool tensor
        face_mask = constrain_head['face']  # shape: (N,), bool tensor
        xyz = ret['offset_xyz']  # shape: (N, 3), tensor

        # 找出属于 head 但不属于 face 的点
        head_only_mask = head_mask & (~face_mask)
        # head_only_mask = (~face_mask)

        # 对这些点做缩放
        xyz[head_only_mask] = xyz[head_only_mask] * 0.5
        # xyz[head_only_mask] = xyz[head_only_mask] * 1

        ret['offset_xyz'] = xyz

        return ret
    

    def densify_by_mesh(self, ret, threshold=0.8):
        # 把插值之后的特征做成每个mesh上高斯的特征，然后再解码为重心坐标/所有高斯的属性。
        densify_mask = (ret['activation']>threshold)
        densify_face = self.mano_face[densify_mask]
        densify_bary = self.barycentric_raw[densify_mask]
        
        ret_densify = {}
        if torch.sum(densify_mask) > 0:
            color = (ret['shs'][densify_mask] * densify_bary).sum(dim=2)
            opacity = (ret['shs'][densify_mask] * densify_bary).sum(dim=2)
            scaling = (ret['scaling'][densify_mask] * densify_bary).sum(dim=2)
            rotation = (ret['rotation'][densify_mask] * densify_bary).sum(dim=2)
            

        # check len(ret_densify)
        return ret_densify


    def forward(self, x, pts, x_fine=None, constrain_dict=None, constrain_head=None, 
                global_feat=None, film_param=None, vis_mask=None):
        assert len(x.shape) == 2
        # x: (B, N, C)
        # B, N, C = x.shape
        ret = {}
        color_x = self._apply_color_latent_modulation(x, global_feat)

        # # faces: [F, 3], 每个面对应的3个顶点 index
        # faces = self.mano_face  # long tensor

        # # x: [B, V, C], V=12337
        # # 先取出每个 face 对应的顶点特征
        # face_feats = x[:, faces, :]   # [B, F, 3, C]

        # # barycentric = torch.softmax(self.barycentric_raw, dim=-1)  # [F, 3]
        # # # 用 bary 插值
        # # bary = barycentric.unsqueeze(0).unsqueeze(-1)   # [1, F, 3, 1]
        # # face_feats = (face_feats * bary).sum(dim=2)     # [B, F, C]


        for k in self.attr_dict:
            layer = self.out_layers[k]

            # # global_feat = self.linear_cond_proj(global_feat.unsqueeze(0))   # global: 1,1024
            # vis_feat, *_ = self.norm(x[vis_mask].unsqueeze(0), emb=global_feat)     # x: N,1024
            # x[vis_mask] = vis_feat.squeeze(0)

            # gamma, beta = film_param  # (B,1,C)
            # # x_mod = (1 + gamma) * x + beta     # broadcast to (B,N,C)
            # x_mod = (gamma) * x + beta     # broadcast to (B,N,C)
            # x = torch.where(~vis_mask.unsqueeze(-1), x_mod, x).squeeze()

            ##################################################
            # un_vis_pts = self.pcl_embed(pts[~vis_mask].unsqueeze(0))   # (1, N, 128)
            # un_vis_pts = self.mlp(un_vis_pts)   # (N, 1024)
            # un_vis_pts = un_vis_pts + x[~vis_mask].unsqueeze(0)   # (N, 1024)
            
            # ##################################################
            # un_vis_pts = x[~vis_mask].unsqueeze(0)   # 取出所有vis_mask为True的点
            
            # attn_output = self.attn(hidden_states=un_vis_pts.to('cuda:1'), 
            #               encoder_hidden_states=global_feat.reshape(-1, 1024).unsqueeze(0).to('cuda:1')).to('cuda')
            #             #   encoder_hidden_states=global_feat.unsqueeze(0).unsqueeze(0))

            # # concat = torch.cat([un_vis_pts, global_feat.reshape(-1, 1024).unsqueeze(0)], dim=1).to('cuda:1')
            # # attn_output = self.attn(
            # #     hidden_states=concat, encoder_hidden_states=concat,
            # # ).cuda()
            # # attn_output = attn_output[:, :un_vis_pts.shape[1], :]

            # hidden_states_unvis = attn_output + un_vis_pts
            # norm_pts = self.norm(hidden_states_unvis)
            # ff_output = self.ff(norm_pts)

            # x_mod = hidden_states_unvis + ff_output.squeeze(0)
            # # x[~vis_mask] = x_mod
            # result = x.clone()
            # result[~vis_mask] = x_mod
            # x = result
            # ##################################################



            # un_vis_pts = x[~vis_mask]   # (N_unvis, C)
            # # multi-layer stacked cross-attn + FFN over the un-visual points
            # for layer_idx in range(self.num_layers):
            #     attn_module = self.attns[layer_idx]
            #     norm_module = self.norms[layer_idx]
            #     ff_module = self.ffs[layer_idx]

            #     # cross-attn expects batched inputs; keep previous convention
            #     attn_out = attn_module(
            #         hidden_states=un_vis_pts.unsqueeze(0),
            #         encoder_hidden_states=global_feat.unsqueeze(0).unsqueeze(0),
            #     )

            #     hidden_states_unvis = attn_out + un_vis_pts.unsqueeze(0)  # residual
            #     norm_pts = norm_module(hidden_states_unvis)
            #     ff_output = ff_module(norm_pts)

            #     x_mod = hidden_states_unvis + ff_output.squeeze(0)
            #     # write back into x at un-visual positions
            #     result = x.clone()
            #     result[~vis_mask] = x_mod
            #     x = result



            # x = torch.where(~vis_mask.unsqueeze(-1), x_mod, x).squeeze()
            feat_in = color_x if k == "shs" else x
            v = layer(feat_in)    # x:N,1024   v:N,3

            # if k == "activation":
            
            #     # activation = torch.sigmoid(self.activate_faces)
            #     bary = torch.softmax(self.barycentric_raw, dim=-1).unsqueeze(0).unsqueeze(-1)
            #     face_feats = x_in.unsqueeze(0)[:, self.mano_face, :]
            #     face_feats = (face_feats * bary).sum(dim=2)

            #     v = layer(face_feats)
            #     v = v.squeeze()
            # #     v = self.opacity_activation(v)
            # #     activation_mask = (v>=0.8)

            # # elif k == "bary":

            #     v = torch.softmax(v, dim=-1)
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(
                        torch.eye(3).type_as(x)[None, :, :].repeat(x.shape[0], 1, 1)
                    )  # constant rotation
                else:
                    # v = torch.nn.functional.normalize(v)
                    v = self.rotation_activation(v)
            elif k == "scaling":
                # v = trunc_exp(v)
                v = self.scaling_activation(v)
                # print(v[0,...])
                if self.clip_scaling is not None:
                    # print(torch.sum(v > self.clip_scaling))
                    # v = torch.clamp(v, min=0, max=self.clip_scaling)
                    v[self.labels != 1] = torch.clamp(v[self.labels != 1], min=0.001, max=0.3)
                    v[self.labels == 1] = torch.clamp(v[self.labels == 1], min=0.005, max=0.4)
                    #####################################################################
                    # v[self.labels != 1] = torch.clamp(v[self.labels != 1], min=0, max=0.015)
                    # v[self.labels == 1] = torch.clamp(v[self.labels == 1], min=0, max=0.02)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    # v = torch.sigmoid(v)
                    v = self.opacity_activation(v)

            elif k == "shs":
                if self.use_rgb:
                    # v = torch.sigmoid(v)
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
                # TODO check
                if self.restrict_offset:  # this way
                    max_step = self.xyz_offset_max_step
                    # max_step = 0.5    # DEBUG
                    v = (torch.sigmoid(v) - 0.5) * max_step  # 这个在 gs_renderer.py 1019行，于网格的顶点相加了
                if self.xyz_offset:  # this way
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v


        ############################################# FOR DENSIFICATION ###############################################
        densify_ret = {}
        # activation = torch.sigmoid(self.activate_faces)
        bary = torch.softmax(self.barycentric_raw, dim=-1).unsqueeze(0).unsqueeze(-1)
        face_feats = x.unsqueeze(0)[:, self.mano_face, :]
        face_feats = (face_feats * bary).sum(dim=2).squeeze()
        color_face_feats = color_x.unsqueeze(0)[:, self.mano_face, :]
        color_face_feats = (color_face_feats * bary).sum(dim=2).squeeze()
            
                
        for j in self.densify_attr_dict:
            densify_layer = self.densify_out_layers[j]

            # if film_param is not None:
            #     gamma, beta = film_param#[k]  # (B,1,C)
            #     x_mod = gamma * face_feats + beta     # broadcast to (B,N,C)
            #     x = x_mod
            # else:
            #     x = face_feats

            x = color_face_feats if j == "shs" else face_feats
            v = densify_layer(x)

            if j == "activation":
                # v += self.alpha_prior_raw
                v = self.opacity_activation(v)

            elif j == "bary":
                v = torch.softmax(v, dim=-1)

            elif j == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    # v = torch.sigmoid(v)
                    v = self.opacity_activation(v)

            elif j == "scaling":
                v = self.scaling_activation(v)
            elif j == "rotation":
                v = self.rotation_activation(v)

            elif j == "shs":
                if self.use_rgb:
                    # v = torch.sigmoid(v)
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
            ret = self.constrain_forward(ret, constrain_dict)  # 限制上半身体素的最大尺寸，防止其变得过大

        if constrain_head is not None:
            ret = self.constrain_expr(ret, constrain_head)

        # if film_params is not None:
        #     ret_densify = self.densify_by_mesh(ret)

        return GaussianAppOutput(**ret), GaussianDensifyOutput(**densify_ret)
    

    def face_vertex_latents(self, vertex_latent):
        """
        vertex_latent: (B, V, C)
        returns L0,L1,L2: each (B, F, C) corresponding to faces' v0/v1/v2 latents
        """
        faces = self.mano_face.long().to(vertex_latent.device)  # (F,3)
        v0_idx = faces[:,0]; v1_idx = faces[:,1]; v2_idx = faces[:,2]
        L0 = vertex_latent[:, v0_idx, :]  # (B, F, C)
        L1 = vertex_latent[:, v1_idx, :]
        L2 = vertex_latent[:, v2_idx, :]
        return L0, L1, L2




class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0   # 8

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

        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)
        self.norm = nn.LayerNorm(dim)

        # ###########################################################################
        # scales = [8, 16, 32]
        # base_dim = dim // 4
        # self.base_proj = nn.Linear(dim, base_dim)
        # self.neigh_aggrs = nn.ModuleList([
        #     NeighborAggregation(base_dim, base_dim, K=k, agg='max')
        #     for k in scales
        # ])

        # # fuse multi-scale
        # ms_in = base_dim * len(scales) + base_dim  # concatenation of center + scales
        # self.ms_fuse = nn.Sequential(
        #     nn.Linear(ms_in, 256), nn.GELU(),
        #     nn.Linear(256, 256)
        # )
        # # per-vertex refinement
        # self.vertex_mlp = nn.Sequential(
        #     nn.Linear(256, 512), nn.GELU(),
        #     nn.Linear(512, dim)
        # )        
        # self.output_norm = nn.LayerNorm(dim)
        # ###########################################################################
        
        self.to('cuda')

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)

        return embeddings

    def forward(self, input_):
        # input: B x N x 3
        input = self.normalize_coords(input_)
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2)    # 2, 12337, 51
        )  # B x N x C
        embed = self.norm(embed)

        # # new added neighbor aggregation module
        # #####################################################################################################
        # # 1) base projection
        # base = self.base_proj(embed)   # [B,N, base_dim] 2,12337,64

        # # 2) multi-scale neighbor aggregation
        # center_feat = base  # (B,N, base_dim)
        # scale_outs = []
        # for aggr in self.neigh_aggrs:
        #     aggr_feat_input = center_feat
        #     out_s = aggr(input_, aggr_feat_input)  # NOTE: neighbor search uses raw verts_pos (world coords)
        #     # aggr_feat_input: 2,12337,64       out_s: 2,12337,64
        #     # 对于每个顶点，找到其 K 个最近邻. 从邻居中聚合特征（max / mean / attention），并输出一个局部 summary（B,N,C）。
        #     # 多尺度（不同 K）让网络同时看到非常局部的微细结构和较大范围的上下文（
            
        #     scale_outs.append(out_s)
        # ms_cat = torch.cat([center_feat] + scale_outs, dim=-1)  # (B,N, base_dim*(len(scales)+1))
        # # 2,12337,256

        # # 3) fuse + vertex mlp
        # ms_f = self.ms_fuse(ms_cat)  # (B, N, 256)
        # embed = self.vertex_mlp(ms_f)  # (B, N, out_dim)  2,12337,1024

        # embed = self.output_norm(embed)

        return embed


    def normalize_coords(self, verts_pos):
        """
        Normalize verts_pos to roughly [-1,1] per-batch, center and scale by max abs.
        verts_pos: [B, N, 3]
        """
        B, N, C = verts_pos.shape
        center = verts_pos.mean(dim=1, keepdim=True)           # [B,1,3]
        v = verts_pos - center
        # amax over N and channels
        max_val = v.abs().amax(dim=(1,2), keepdim=True)        # [B,1,1]
        v = v / (max_val + 1e-6)
        return v




# ----------------- main HierarchicalPointEmbed -----------------
class HierarchicalPointEmbed_ori(nn.Module):
    def __init__(
        self,
        in_pos_dim=3,
        in_feat_dim=0,   # optional per-vertex image feature dim
        out_dim=1024,
        base_dim=64,
        scales=[8, 16, 32],   # K neighbors per scale
        part_dim=128,
        joint_dim=128,
        use_image_feat=False,
    ):
        super().__init__()
        self.scales = scales
        self.use_image_feat = use_image_feat and in_feat_dim>0
        self.base_proj = nn.Linear(in_pos_dim + (in_feat_dim if self.use_image_feat else 0), base_dim)
        # per-scale neighbor aggr
        self.neigh_aggrs = nn.ModuleList([
            NeighborAggregation(in_feat_dim + base_dim, base_dim, K=k, agg='max') for k in scales
        ])
        # fuse multi-scale
        ms_in = base_dim * len(scales) + base_dim  # concatenation of center + scales
        self.ms_fuse = nn.Sequential(
            nn.Linear(ms_in, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        # per-vertex refinement
        self.vertex_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, out_dim)
        )
        # part pooling + broadcast path
        self.part_pool_proj = nn.Linear(out_dim, part_dim)
        self.part_processor = nn.Sequential(
            nn.Linear(part_dim, part_dim), nn.GELU(),
            nn.Linear(part_dim, part_dim)
        )
        self.part_broadcast_proj = nn.Linear(part_dim, out_dim)
        # # joint embedder
        # self.joint_embed = nn.Sequential(
        #     nn.Linear(3, joint_dim), nn.GELU(),
        #     nn.Linear(joint_dim, joint_dim)
        # )
        # self.joint_broadcast_proj = nn.Linear(joint_dim, out_dim)
        # detail branch (nails)
        # self.detail_mlp = nn.Sequential(
        #     nn.Linear(out_dim, out_dim//2), nn.GELU(),
        #     nn.Linear(out_dim//2, out_dim)
        # )
        self.output_norm = nn.LayerNorm(out_dim)

        # # importance head
        # self.importance_head = nn.Sequential(
        #     nn.Linear(out_dim, 128), nn.GELU(),
        #     nn.Linear(128, 1), nn.Sigmoid()
        # )

        ###########################################################################
        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        with open(labels_path, 'rb') as f:
            pc = PlyData.read(f)
        if pc.elements:
            pc = pd.DataFrame(pc.elements[0].data).values
        pc_coords = pc[:, :3]
        # self.labels = torch.tensor(pc[:, 6].astype(np.uint8)).cuda() # (N,)
        self.labels = torch.tensor(pc[:, 6].astype(np.int64)).cuda()


        NAIL_PARTS = {
            "index3": 4,
            "middle3": 7,
            "pinky3": 10,
            "ring3": 13,
            "thumb3": 16
        }

        # 找出所有指甲顶点
        # nail_indices = np.isin(self.labels, list(NAIL_PARTS.values()))
        # self.nail_coords = pc_coords[nail_indices]
        # self.objects_dc = F.one_hot((self.labels-1).to(torch.int64), num_classes=16).unsqueeze(1).to(torch.float32).cuda()
        # 找到哪些点是指甲
        self.nail_mask = torch.isin(self.labels.to(torch.int64), torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device))
        # label_bias = torch.tensor([[1.0, 0.0],   # palm/普通区域 → expert_0
        #                    [0.0, 1.5]])  # 指甲区域 → expert_1，给偏置
        # self.label_bias = nn.Parameter(label_bias, requires_grad=True)
        ###########################################################################





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

        # # 1) base projection (pos + optional image feats)
        # if self.use_image_feat:
        #     assert img_feat is not None
        #     base_in = torch.cat([verts_pos, img_feat], dim=-1)  # (B,N, 3+F)
        # else:
        #     base_in = verts_pos
        base_in = verts_pos
        base = self.base_proj(base_in)  # (B, N, base_dim)

        # 2) multi-scale neighbor aggregation
        # at each scale we feed center_feat + neighbor feats (center neighbor combine)
        # prepare a center_feat by concatenating base with zero neighbor feature placeholder (we use base as "center_feat")
        center_feat = base  # (B,N, base_dim)
        # to feed NeighborAggregation we need feat input (B,N,C) that represents some per-vertex feat
        # we choose to use base as the "feat" and also pass it inside neighbor aggregation
        scale_outs = []
        for aggr in self.neigh_aggrs:
            out_s = aggr(verts_pos, torch.cat([center_feat, img_feat], dim=-1) if self.use_image_feat else center_feat)
            scale_outs.append(out_s)  # (B,N, base_dim)
        ms_cat = torch.cat([center_feat] + scale_outs, dim=-1)  # (B,N, base_dim*(len(scales)+1))

        # 3) fuse multi-scale
        ms_f = self.ms_fuse(ms_cat)  # (B, N, 256)

        # 4) per-vertex preliminary feature
        v_pre = self.vertex_mlp(ms_f)  # (B, N, out_dim)

        # 5) optional part-level conditioning
        # ensure part_idx shape (B,N)
        # if part_idx.dim() == 1:
        #     part_idx = part_idx.unsqueeze(0).repeat(B, 1)
        part_idx = self.labels.unsqueeze(0).repeat(B, 1)
        P = int(part_idx.max().item()+1)
        # compute per-part pooled token
        part_token = scatter_mean_torch(v_pre, part_idx, dim_size=P)  # (B, P, out_dim)
        # project & process
        part_token = self.part_pool_proj(part_token)  # (B,P,part_dim)
        part_token = self.part_processor(part_token)  # (B,P,part_dim)
        # broadcast back to vertices
        # gather per-vertex broadcast
        gather_idx = part_idx  # (B,N)
        # build gather by indexing per-batch
        part_bcast = part_token[torch.arange(B)[:,None], gather_idx]  # (B,N,part_dim)
        part_bcast = self.part_broadcast_proj(part_bcast)  # (B,N,out_dim)
        v_pre = v_pre + part_bcast

        # 6) optional joint conditioning
        if joint_xyz is not None:
            # joint_xyz: (B, J, 3) -> embed and global avg
            j_embed = self.joint_embed(joint_xyz)  # (B,J,joint_dim)
            j_global = j_embed.mean(dim=1, keepdim=True)  # (B,1,joint_dim)
            j_bcast = self.joint_broadcast_proj(j_global)  # (B,1,out_dim)
            v_pre = v_pre + j_bcast  # broadcast to (B,N,out_dim)

        # 7) detail branch for nails
        v_refined = v_pre
        
        # nail_mask: (B,N) bool
        # apply detail mlp only to mask positions (but keep shape) -> residual
        # flatten
        flat = v_pre.view(B*N, -1)
        # mask_flat = self.nail_mask.unsqueeze(0).expand(B,-1).reshape(B*N)
        # if mask_flat.any():
        #     refined = self.detail_mlp(flat[mask_flat])  # (num_nail, out_dim)
        #     flat[mask_flat] = flat[mask_flat] + refined
        v_refined = flat.view(B, N, -1)

        v_refined = self.output_norm(v_refined)

        # 8) importance score head
        # importance = self.importance_head(v_refined)  # (B, N, 1) in (0,1)

        return v_refined #, importance



class HierarchicalPointEmbed(nn.Module):
    def __init__(
        self,
        in_pos_dim=3,
        in_feat_dim=0,   # optional per-vertex image feature dim
        out_dim=1024,
        base_dim=64,
        scales=[8, 16, 32],   # K neighbors per scale
        part_dim=128,
        joint_dim=128,
        use_image_feat=False,
        pos_embed_dim=128,     # <-- 新增：positional embedding 输出维度
        pos_hidden_dim=48,     # 传给 PointEmbed 的 hidden_dim（频率数量等）
    ):
        super().__init__()
        self.scales = scales
        self.use_image_feat = use_image_feat and in_feat_dim>0
        self.pos_embed_dim = pos_embed_dim

        # ---------- 新增：位置编码器 ----------
        # 这里直接使用你给的 PointEmbed 类；输出 dim = pos_embed_dim
        self.point_pos_embed = PointEmbed(hidden_dim=pos_hidden_dim, dim=pos_embed_dim)

        # ---------- base_proj 现在接收 pos_embed （而不是原始 xyz） ----------
        in_base_dim = pos_embed_dim + (in_feat_dim if self.use_image_feat else 0)
        self.base_proj = nn.Linear(in_base_dim, base_dim)

        # per-scale neighbor aggr
        # NeighborAggregation 原本是 NeighborAggregation(in_feat_dim + base_dim, base_dim, K=k)
        # 现在 base_dim 是我们投影后的 base_dim；若 use_image_feat, aggr 输入为 base + img_feat
        self.neigh_aggrs = nn.ModuleList([
            NeighborAggregation((in_feat_dim if self.use_image_feat else 0) + base_dim, base_dim, K=k, agg='max')
            for k in scales
        ])

        # fuse multi-scale
        ms_in = base_dim * len(scales) + base_dim  # concatenation of center + scales
        self.ms_fuse = nn.Sequential(
            nn.Linear(ms_in, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        # per-vertex refinement
        self.vertex_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, out_dim)
        )
        # part pooling + broadcast path
        self.part_pool_proj = nn.Linear(out_dim, part_dim)
        self.part_processor = nn.Sequential(
            nn.Linear(part_dim, part_dim), nn.GELU(),
            nn.Linear(part_dim, part_dim)
        )
        self.part_broadcast_proj = nn.Linear(part_dim, out_dim)

        self.output_norm = nn.LayerNorm(out_dim)

        ###########################################################################
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
        ###########################################################################

    def normalize_coords(self, verts_pos):
        """
        Normalize verts_pos to roughly [-1,1] per-batch, center and scale by max abs.
        verts_pos: [B, N, 3]
        """
        B, N, C = verts_pos.shape
        center = verts_pos.mean(dim=1, keepdim=True)           # [B,1,3]
        v = verts_pos - center
        # amax over N and channels
        max_val = v.abs().amax(dim=(1,2), keepdim=True)        # [B,1,1]
        v = v / (max_val + 1e-6)
        return v

    def forward(self, verts_pos, part_idx=None, joint_xyz=None, img_feat=None, nail_mask=None):
        """
        verts_pos: (B, N, 3)
        img_feat:  (B, N, F_img) optional
        """
        B, N, _ = verts_pos.shape
        device = verts_pos.device

        # ---------- 位置归一化（非常重要） ----------
        verts_norm = self.normalize_coords(verts_pos)  # [B,N,3] roughly in [-1,1]

        # ---------- pos embedding ----------
        pos_feat = self.point_pos_embed(verts_norm)    # [B,N,pos_embed_dim] # 2,12337,128

        # ---------- base projection: combine pos_feat + optional image feat ----------
        if self.use_image_feat:
            assert img_feat is not None, "use_image_feat=True but img_feat is None"
            base_in = torch.cat([pos_feat, img_feat], dim=-1)   # [B,N, pos + F_img]
        else:
            base_in = pos_feat   # [B,N,pos_embed_dim] 2,12337,128

        base = self.base_proj(base_in)   # [B,N, base_dim] 2,12337,64

        # 2) multi-scale neighbor aggregation
        center_feat = base  # (B,N, base_dim)
        scale_outs = []
        for aggr in self.neigh_aggrs:
            # NeighborAggregation expects (positions, feat) — ensure we pass in correct feat
            if self.use_image_feat:
                aggr_feat_input = torch.cat([center_feat, img_feat], dim=-1)
            else:
                aggr_feat_input = center_feat
            out_s = aggr(verts_pos, aggr_feat_input)  # NOTE: neighbor search uses raw verts_pos (world coords)
            # aggr_feat_input: 2,12337,64       out_s: 2,12337,64
            # 对于每个顶点，找到其 K 个最近邻. 从邻居中聚合特征（max / mean / attention），并输出一个局部 summary（B,N,C）。
            # 多尺度（不同 K）让网络同时看到非常局部的微细结构和较大范围的上下文（
            
            scale_outs.append(out_s)
        ms_cat = torch.cat([center_feat] + scale_outs, dim=-1)  # (B,N, base_dim*(len(scales)+1))
        # 2,12337,256

        # 3) fuse + vertex mlp
        ms_f = self.ms_fuse(ms_cat)  # (B, N, 256)
        v_pre = self.vertex_mlp(ms_f)  # (B, N, out_dim)  2,12337,1024

        # part pooling and broadcast as before
        part_idx = self.labels.unsqueeze(0).repeat(B, 1) # 2,12337
        P = int(part_idx.max().item()+1) # 17
        part_token = scatter_mean_torch(v_pre, part_idx, dim_size=P)  # (B, P, out_dim)
        # (2,17,1024) 对每一个 part，把属于该 part 的所有点的 v_pre 特征求平均。

        part_token = self.part_pool_proj(part_token)  # (B,P,part_dim) # 2,17,128
        part_token = self.part_processor(part_token)  # (B,P,part_dim) # 2,17,128
        part_bcast = part_token[torch.arange(B)[:,None], part_idx]  # (B,N,part_dim)  # 2,12337,128
        part_bcast = self.part_broadcast_proj(part_bcast)  # (B,N,out_dim)  # 2,12337,1024
        v_pre = v_pre + part_bcast

        # optional joint conditioning (same as original)
        # ... (保持你原有的 joint 代码)

        v_refined = v_pre.view(B, N, -1)  # 2,12337,1024

        v_refined = self.output_norm(v_refined)  # 2,12337,1024

        return v_refined



class HierarchicalPointEmbedTransformer(nn.Module):
    def __init__(
        self,
        in_feat_dim=1024,      # ⭐ 主输入特征维度（来自 transformer 的点特征）
        out_dim=1024,
        base_dim=512, #64,
        # scales=[8, 16, 32],    # K neighbors per scale
        scales=[4, 8, 16],    # K neighbors per scale
        part_dim=1024, #128,
        # part_dim=512, #128,
        joint_dim=1024, #128,
        use_image_feat=False,
    ):
        super().__init__()
        self.scales = scales
        self.use_image_feat = use_image_feat and in_feat_dim > 0

        # -------- base_proj: 从输入点特征 (+可选 img_feat) 映射到 base_dim ----------
        in_base_dim = in_feat_dim + (in_feat_dim if self.use_image_feat else 0)
        # ⚠️ 如果 img_feat 维度跟 in_feat_dim 不一样，把上面这一行改成具体维度之和
        self.base_proj = nn.Linear(in_base_dim, base_dim)

        # -------- per-scale neighbor aggregation ----------
        # NeighborAggregation expects (positions, feat)
        # 这里 feat = concat(base_feat, img_feat) 或者只是 base_feat
        self.neigh_aggrs = nn.ModuleList([
            NeighborAggregation(
                (in_feat_dim if self.use_image_feat else 0) + base_dim,
                base_dim,
                K=k,
                agg='max'
            )
            for k in scales
        ])

        # -------- fuse multi-scale ----------
        ms_in = base_dim * (len(scales) + 1)  # center + |scales|
        self.ms_fuse = nn.Sequential(
            nn.Linear(ms_in, 256), nn.GELU(),
            nn.Linear(256, 256)
        )

        # -------- vertex refinement MLP ----------
        self.vertex_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, out_dim)
        )

        # -------- part pooling + broadcast ----------
        self.part_pool_proj = nn.Linear(out_dim, part_dim)
        self.part_processor = nn.Sequential(
            nn.Linear(part_dim, part_dim), nn.GELU(),
            nn.Linear(part_dim, part_dim)
        )
        self.part_broadcast_proj = nn.Linear(part_dim, out_dim)

        self.output_norm = nn.LayerNorm(out_dim)

        ###########################################################################
        # 语义 part label & nail mask 读取（保持你原来的逻辑）
        labels_path = 'pretrained_models/dense_sample_points/manohd_semantic.ply'
        # labels_side_path = './manohd_semantic_sideaware.ply'    # 1-16 是手背，17 之后是手掌心
        with open(labels_path, 'rb') as f:
            pc = PlyData.read(f)
        # with open(labels_side_path, 'rb') as f:
        #     pc_side = PlyData.read(f)
        if pc.elements:
            pc = pd.DataFrame(pc.elements[0].data).values
        pts = pc[:, :3]
        # if pc_side.elements:
        #     pc_side = pd.DataFrame(pc_side.elements[0].data).values

        # pts_t = torch.from_numpy(pts).float()     # (N,3)

        # # 计算 pairwise 距离
        # d = torch.cdist(pts_t.unsqueeze(0), pts_t.unsqueeze(0)).squeeze(0)  # (N,N)

        # K_max = 32
        # _, idx = torch.topk(d, K_max, largest=False)  # (N,K)

        # # idx = idx.cpu().numpy()

        # self.register_buffer("knn_idx_all", idx.long())

        self.labels = torch.tensor(pc[:, 6].astype(np.int64)).cuda()

        # self.labels_side = torch.tensor(pc_side[:, 7].astype(np.int64)).cuda()

        # NAIL_PARTS = {
        #     "index3": 4,
        #     "middle3": 7,
        #     "pinky3": 10,
        #     "ring3": 13,
        #     "thumb3": 16
        # }

        # self.nail_mask = torch.isin(
        #     self.labels.to(torch.int64),
        #     torch.tensor(list(NAIL_PARTS.values())).to(self.labels.device)
        # )
        ###########################################################################

    def forward(
        self,
        point_feat,          # ⭐ (B, N, in_feat_dim) 来自 transformer 的点特征
        verts_pos=None,      # (B, N, 3) 可选，仅用于 KNN，如果没有邻域就传 None
        img_feat=None,       # (B, N, F_img) 可选
        nail_mask=None,      # 不在这里用，你后面可以扩展
    ):
        """
        point_feat: (B, N, C_in)  主输入特征
        verts_pos:  (B, N, 3)     可选：如果要做 NeighborAggregation 的 KNN，就必须提供
        """
        B, N, C_in = point_feat.shape
        device = point_feat.device

        # # ---------- 1) base projection: 从输入特征到 base_dim ----------
        # if self.use_image_feat:
        #     assert img_feat is not None, "use_image_feat=True 但 img_feat 是 None"
        #     base_in = torch.cat([point_feat, img_feat], dim=-1)  # (B, N, C_in + F_img)
        # else:
        #     base_in = point_feat                                   # (B, N, C_in)

        # base = self.base_proj(base_in)   # (B, N, base_dim)
        # # base = base_in

        # # ---------- 2) 多尺度邻域聚合 ----------
        # center_feat = base                # (B, N, base_dim)
        # scale_outs = []

        # if verts_pos is not None:
        #     # 有 xyz 的情况下才能做 KNN + 聚合
        #     for aggr in self.neigh_aggrs:
        #     # for k, aggr in zip(self.scales, self.neigh_aggrs):
        #         if self.use_image_feat:
        #             aggr_feat_input = torch.cat([center_feat, img_feat], dim=-1)
        #         else:
        #             aggr_feat_input = center_feat
        #         # NeighborAggregation(positions, feat)
        #         # idx_k = self.knn_idx_all.unsqueeze(0).expand(B, -1, -1)[..., :k]  # (B, N, k)
        #         # out_s = aggr(verts_pos, aggr_feat_input, idx=idx_k)  # (B, N, base_dim)
        #         out_s = aggr(verts_pos, aggr_feat_input)  # (B, N, base_dim)
        #         # out_s = aggr(verts_pos, aggr_feat_input)  # (B, N, base_dim)
        #         scale_outs.append(out_s)
        # else:
        #     # 没有 verts_pos，就不做邻域，只保留 center_feat
        #     scale_outs = []

        # # concat center + multi-scale outputs
        # if len(scale_outs) > 0:
        #     ms_cat = torch.cat([center_feat] + scale_outs, dim=-1)  # (B, N, base_dim*(1+len(scales)))
        # else:
        #     # 无邻域时，直接用 center_feat 进 ms_fuse
        #     ms_cat = center_feat                                     # (B, N, base_dim)

        # # ---------- 3) fuse + vertex mlp ----------
        # ms_f = self.ms_fuse(ms_cat)        # (B, N, 256)
        # v_pre = self.vertex_mlp(ms_f)      # (B, N, out_dim)
        
        v_pre = point_feat
        # ---------- 4) part pooling + broadcast ----------
        # 这里仍用预定义 self.labels 作为每个顶点的 part index
        # part_idx = self.labels_side.unsqueeze(0).repeat(B, 1).to(device)  # (B, N)
        part_idx = self.labels.unsqueeze(0).repeat(B, 1).to(device)  # (B, N)
        P = int(part_idx.max().item() + 1)                           # #parts, e.g. 17

        # scatter_mean_torch: (B, N, out_dim), part_idx -> (B, P, out_dim)
        part_token = scatter_mean_torch(v_pre, part_idx, dim_size=P)  # (B, P, out_dim)

        # part_token = self.part_pool_proj(part_token)   # (B, P, part_dim)
        part_token = self.part_processor(part_token)   # (B, P, part_dim)

        # broadcast 回每个顶点
        part_bcast = part_token[torch.arange(B)[:, None], part_idx]   # (B, N, part_dim)
        # part_bcast = self.part_broadcast_proj(part_bcast)             # (B, N, out_dim)

        v_pre = v_pre + part_bcast       # (B, N, out_dim)

        # ---------- 5) optional joint conditioning（你原来那段可以继续加在这里） ----------
        # TODO: 如果你有 joint_xyz / joint_feat 之类，可以在这里做 FiLM 或者再加一层 MLP

        v_refined = self.output_norm(v_pre)   # (B, N, out_dim)

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

        # 输入是 [rel_pos(3), dist(1), center_feat(C), neigh_feat(C)]
        self.inp_dim = 3 + 1 + in_feat_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.inp_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, pos, feat, idx=None):
        # pos: (B, N, 3), feat: (B, N, C)
        B, N, C = feat.shape
        device = pos.device

        # 1) 决定用哪个 idx
        if idx is None:
            # 回退到原来的 knn_idx 实现
            idx = knn_idx(pos, self.K)  # (B, N, K)
        else:
            # 外部传进来的 idx 形状 (B, N, K')，只取前 self.K 个
            if idx.size(-1) < self.K:
                raise ValueError(f"idx.size(-1)={idx.size(-1)} < K={self.K}")
            idx = idx[..., :self.K]     # (B, N, K)

        # 2) 原版实现：一次性构造 (B, N, K, *)，一次 MLP 调用
        batch_idx = torch.arange(B, device=device)[:, None, None]  # (B,1,1)

        # neighbor positions / features
        neigh_pos  = pos[batch_idx, idx]    # (B, N, K, 3)
        neigh_feat = feat[batch_idx, idx]   # (B, N, K, C)

        # center pos / rel / dist
        center = pos.unsqueeze(2)           # (B, N, 1, 3)
        rel    = neigh_pos - center         # (B, N, K, 3)
        dist   = torch.norm(rel, dim=-1, keepdim=True)  # (B, N, K, 1)

        # tile center_feat
        center_feat = feat.unsqueeze(2).expand(-1, -1, self.K, -1)  # (B, N, K, C)

        # concat features
        inp = torch.cat([rel, dist, center_feat, neigh_feat], dim=-1)  # (B, N, K, 3+1+2C)

        # flatten for MLP
        BnK = B * N * self.K
        inp_flat = inp.view(BnK, -1)                 # (B*N*K, 3+1+2C)
        out_flat = self.mlp(inp_flat)                # (B*N*K, out_dim)
        out = out_flat.view(B, N, self.K, -1)        # (B, N, K, out_dim)

        # aggregate along neighbor dim
        if self.agg == 'max':
            out_agg, _ = out.max(2)  # (B, N, out_dim)
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
        # process concatenated [rel_pos(3), dist(1), neighbor_feat(C)]
        self.mlp = nn.Sequential(
            nn.Linear(3 + 1 + in_feat_dim*2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        # self.knn = KNN(k=K, transpose_mode=True)
        # self.knn_s2 = KNN(k=8, transpose_mode=True)
        # self.knn_s3 = KNN(k=16, transpose_mode=True)


    def forward(self, pos, feat):
        # pos: (B,N,3), feat: (B,N,C)
        B, N, C = feat.shape
        idx = knn_idx(pos, self.K)  # (B, N, K)
        # _, idx = self.knn(pos.detach(), pos.detach())
        # idx = knn_faiss(pos, self.K)  # (B, N, K)
        # gather neighbor positions and features
        device = pos.device
        batch_idx = torch.arange(B, device=device)[:, None, None]
        neigh_pos = pos[batch_idx, idx]   # (B, N, K, 3)
        neigh_feat = feat[batch_idx, idx] # (B, N, K, C)
        # center pos
        center = pos.unsqueeze(2)  # (B,N,1,3)
        rel = neigh_pos - center   # (B,N,K,3)
        dist = torch.norm(rel, dim=-1, keepdim=True)  # (B,N,K,1)
        # tile center_feat
        center_feat = feat.unsqueeze(2).expand(-1, -1, self.K, -1)  # (B,N,K,C)
        # concat
        inp = torch.cat([rel, dist, center_feat, neigh_feat], dim=-1)  # (B,N,K, 3+1+C+C)
        # flatten for mlp
        BnK = B * N * self.K
        inp_flat = inp.view(BnK, -1)
        out_flat = self.mlp(inp_flat)  # (BnK, out_dim)
        out = out_flat.view(B, N, self.K, -1)
        if self.agg == 'max':
            out_agg, _ = out.max(2)  # (B,N,out_dim)
        else:
            out_agg = out.mean(2)
        return out_agg



# ----------------- helpers -----------------
def pairwise_dist(x, y):
    # x: (B, N, D), y: (B, M, D) -> returns (B, N, M) distances squared
    # Memory: O(N*M) per batch
    x2 = (x**2).sum(-1, keepdim=True)  # (B,N,1)
    y2 = (y**2).sum(-1, keepdim=True).transpose(1, 2)  # (B,1,M)
    xy = x @ y.transpose(1, 2)  # (B,N,M)
    dist = x2 + y2 - 2*xy
    return dist



def knn_idx(pts, K):
    # # pts: (B, N, 3)
    # # return idx: (B, N, K)
    # with torch.no_grad():
    #     D = pairwise_dist(pts, pts)  # (B, N, N)
    #     # large diag values to ignore self? we keep self included optionally
    #     # choose topk smallest distances
    #     vals, idx = torch.topk(D, k=K, dim=-1, largest=False, sorted=True)
    idx = knn_points(pts, pts, K=K).idx
    return idx  # (B, N, K)


def knn_faiss(x, K):
    # x: (B, N, 3)
    B, N, D = x.shape
    x_np = x.detach().cpu().numpy()
    idx_all = []

    for b in range(B):
        index = faiss.IndexFlatL2(D)
        index = faiss.index_cpu_to_all_gpus(index)  # multi-GPU support
        index.add(x_np[b])
        _, idx = index.search(x_np[b], K)
        idx_all.append(torch.from_numpy(idx).long())

    return torch.stack(idx_all).to(x.device)



def scatter_mean_torch(src, idx, dim_size):
    # src: (B, N, C)
    # idx: (B, N) int labels in [0..dim_size-1]
    B, N, C = src.shape
    out = src.new_zeros(B, dim_size, C)
    count = src.new_zeros(B, dim_size, 1)
    for b in range(B):
        out[b].index_add_(0, idx[b], src[b])  # idx[b] length N
        ones = src.new_ones(N, 1, device=src.device)
        count[b].index_add_(0, idx[b], ones)
    count = count.clamp(min=1.0)
    out = out / count
    return out  # (B, dim_size, C)




class CrossAttnBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """

    # Block contains a cross-attention layer, a self-attention layer, and an MLP
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
        # TODO check already apply normalization
        # self.norm_q = nn.LayerNorm(inner_dim, eps=eps)
        # self.norm_k = nn.LayerNorm(cond_dim, eps=eps)
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
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
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
        self.proj_in = nn.Linear(dim, inner_dim * 2)  # split for gate
        self.proj_out = nn.Linear(inner_dim, dim)
        self.act = nn.SiLU()   # 或 torch.nn.functional.silu
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.proj_in(x).chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.proj_out(x)
        return x# self.dropout(x)



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

        # 初始化 grid
        self.grid = HexPlaneField(bounds=1.6, multires=[1], planeconfig=kplanes_config)

        # 创建子网络
        self.pos_, self.sceles_, self.rots_, self.opacity_ = self.create_res_net()

        # 权重初始化
        self.apply(initialize_weights)
        self.pos_.initialize_weights()


    def create_res_net(self):

        self.feature_out = [nn.Linear(self.grid.feat_dim, self.W)]

        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        output_dim = self.W
        return \
            Head_Res_Net(self.W, 3), \
                Head_Res_Net(self.W, 3), \
                Head_Res_Net(self.W, 4), \
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
            # attributes = ['scaling','xyz','opacity','rotation','shs']
            attributes = ["activation", "bary"]
        self.attrs = attributes
        self.in_ch = in_channels
        # self.linear = nn.Linear(z_dim, 6*z_dim)
        # self.silu = nn.SiLU()

        # small MLP backbone
        self.mlp = nn.Sequential(
            nn.SiLU(),  # ReLU
            nn.Linear(z_dim, hidden),
            nn.SiLU(),  # ReLU
            nn.Linear(hidden, hidden),
        )
        # for each attribute we output gamma and beta (per-channel)
        # to reduce params we implement a small projection for each attr
        # self.attr_proj = nn.ModuleDict()
        # for a in self.attrs:
        #     self.attr_proj[a] = nn.Linear(hidden, in_channels * 2)  # [gamma||beta]

        self.attr_proj = nn.Linear(hidden, in_channels * 2) 
        nn.init.constant_(self.attr_proj.weight, 0.0)
        nn.init.constant_(self.attr_proj.bias, 0.0)
        # set bias gamma to 1 for first in_channels entries
        # but since bias is length 2*in_ch: first half gamma, second beta
        with torch.no_grad():
            self.attr_proj.bias[:in_channels].fill_(1.0)  # gamma init = 1


        # # initialization: gamma close to 1, beta close to 0
        # for m in self.attr_proj.values():
        #     nn.init.constant_(m.weight, 0.0)
        #     nn.init.constant_(m.bias, 0.0)
        #     # set bias gamma to 1 for first in_channels entries
        #     # but since bias is length 2*in_ch: first half gamma, second beta
        #     with torch.no_grad():
        #         m.bias[:in_channels].fill_(1.0)  # gamma init = 1


    def forward(self, z):
        """
        z: (B, z_dim)
        returns dict attr -> (gamma (B, C), beta (B, C))
        """
        h = self.mlp(z)  # (B, hidden)
        # h = self.linear(self.silu(z))  # (B, 6*z_dim)

        v = self.attr_proj(h)  # (B, 2*C)
        gamma, beta = v[:, :self.in_ch], v[:, self.in_ch:]
        out = (gamma.unsqueeze(1), beta.unsqueeze(1))  # keep dim for broadcasting to (B, N, C)
        return out

        # for a in self.attrs:
        #     v = self.attr_proj[a](h)  # (B, 2*C)
        #     gamma, beta = v[:, :self.in_ch], v[:, self.in_ch:]
        #     out[a] = (gamma.unsqueeze(1), beta.unsqueeze(1))  # keep dim for broadcasting to (B, N, C)
        # return out




# -------------------------
# FiLM modulator: produce gamma/beta for each decoder layer
# -------------------------
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
        # for each layer produce gamma and beta
        self.mlp_per_layer = nn.ModuleList()
        for dim in self.layer_dims:
            # small MLP -> 2*dim
            self.mlp_per_layer.append(
                nn.Sequential(
                    nn.Linear(gdim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, dim * 2)  # gamma|beta flattened
                )
            )

    def forward(self, global_vec: torch.Tensor):
        out = []
        for mlp in self.mlp_per_layer:
            gb = mlp(global_vec)  # [B, 2*dim]
            dim = gb.shape[-1] // 2
            gamma = gb[:, :dim]  # [B, dim]
            beta = gb[:, dim:]
            out.append((gamma, beta))
        return out  # list length = num layers




class LoRAAdapter(nn.Module):
    """Simple LoRA adapter: low-rank update for a vector of dim `dim`.

    y = x + scale * U(V(x)),  where V: dim->rank, U: rank->dim.
    """

    def __init__(self, dim: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.down = nn.Linear(dim, rank, bias=False)  # V: dim -> rank
        self.up = nn.Linear(rank, dim, bias=False)  # U: rank -> dim

        # Init following common LoRA practice: down Kaiming, up zeros so initial delta ~ 0
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
            query_dim,  # 1024
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
                use_lora=False,  # Only enable LoRA adapters for finetuning
            ):

        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder
        self.query_dim = query_dim  # Store for dynamic LoRA creation

        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree

        self.mano_model = MANOVoxelMeshModel()  # false
        # per-dataset mano model cache (allows different center_add flags per dataset)
        self.mano_models = {}
        # mapping dataset_id -> center_add flag. Default: interhand/hanco True, hands11k False
        if dataset_center_add_map is None:
            self.dataset_center_add_map = {0: True, 1: True, 2: False}
        else:
            self.dataset_center_add_map = dataset_center_add_map

        # self.pcl_embed = pcl_embed
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
        self.use_lora = use_lora  # Store flag for conditional LoRA usage

        # using to mapping transformer decode feature to regression features. as decode feature is processed by NormLayer.
        if self.mlp_network_config is not None:
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config).to('cuda')
            # LoRA adapter on top of mlp_net output (only for finetuning)
            if self.use_lora:
                self.lora_mlp = LoRAAdapter(query_dim, rank=16, alpha=1.0).to('cuda')
            else:
                self.lora_mlp = None
        else:
            self.mlp_net = None
            self.lora_mlp = None

            # # 让初始输出恒为 0：最后一层 Linear 置零
            # last_linear = None
            # for m in reversed(self.mlp_net.layers):
            #     if isinstance(m, nn.Linear):
            #         last_linear = m
            #         break
            # assert last_linear is not None

            # nn.init.zeros_(last_linear.weight)
            # nn.init.zeros_(last_linear.bias)

        self.feature_map = feature_map
        
        if feature_map: 
            from LHM.models.styleunet import StyleUNet
            self.neural_refiner = StyleUNet().to('cuda')
            use_rgb = False
        else:
            self.neural_refiner = None
        
        # ===== Edit mode visibility masking interface =====
        # For finetune_edit stage2: freeze invisible points in mlp_net
        self.edit_mask_mode = False  # Enable/disable masking
        self.edit_vis_mask = None  # [B, N] visibility mask for points
        self.edit_invisible_cache = None  # cached mlp outputs for invisible points (kept frozen)
        # Soft visibility blending（可选）：结合 vis_prob 在可见边界减缓更新
        self.edit_vis_prob = None  # [B, N] visibility probability（不传则关闭）
        self.edit_soft_vis_blend = False  # 开关：True 时启用 vis_prob 融合
        self.edit_vis_prob_low = 0.2  # 低阈值，低于此视为不可信
        self.edit_vis_prob_high = 0.8  # 高阈值，高于此视为可信
        # ===== End edit mode interface =====         

        # self.film_param = HyperFiLM().to('cuda')
        # self.norm = AdaLayerNormZero(1024).to('cuda')
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
        )#.to('cuda')  # 这个就是普通的，还没加载权重

        # LoRA adapter on gs_net input features (only for finetuning)
        if self.use_lora:
            self.lora_gs = LoRAAdapter(query_dim, rank=16, alpha=1.0).to('cuda')
        else:
            self.lora_gs = None
        
        self.renderer_mesh = Renderer_mesh()
        # quaternion from cano to pose
        self.quat_helper = PerVertQuaternion(self.gs_net.mano_verts.to('cuda'), self.gs_net.mano_face.to('cuda')).to('cuda')
        # self.face_area = self.calc_face_areas(self.gs_net.mano_verts, self.gs_net.mano_face)
        # self.pose_embed = BodyPoseRefiner().to('cuda')
        
        # self.grid_offset = GridOffset()
        # is_face_expr = self.smplx_model.is_face_expr  # (3084)
        # is_face = self.smplx_model.is_face
        # self.constrain_head = dict(head=is_face, face=is_face_expr)
        #
        # head_mask = self.constrain_head['head']  # shape: (N,), bool tensor
        # self.face_mask = self.constrain_head['face']  # shape: (N,), bool tensor
        # self.head_only_mask = (head_mask & (~self.face_mask))

        # self.grid = HexPlaneField(bounds = 1.6, multires = [1], planeconfig = kplanes_config)
        # self.W, self.D = 64, 1
        # self.input_ch = 27
        # self.pos_, self.sceles_, self.rots_, self.opacity_ = self.create_res_net()
        # self.apply(initialize_weights)
        # self.pos_.initialize_weights()

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
        query_dim = getattr(self, 'query_dim', 1024)  # fallback to default if not stored
        
        # Create lora_mlp if mlp_net exists but lora_mlp doesn't
        if self.mlp_net is not None and self.lora_mlp is None:
            self.lora_mlp = LoRAAdapter(query_dim, rank=rank, alpha=alpha).to('cuda')
            self.use_lora = True
            print(f"[ensure_lora_adapters] Created lora_mlp with rank={rank}, alpha={alpha}")
        
        # Create lora_gs if it doesn't exist
        if self.lora_gs is None:
            self.lora_gs = LoRAAdapter(query_dim, rank=rank, alpha=alpha).to('cuda')
            print(f"[ensure_lora_adapters] Created lora_gs with rank={rank}, alpha={alpha}")

    def forward_single_view(
            self,
            gs: GaussianModel,#32,
            viewpoint_camera: Camera,
            background_color: Optional[Float[Tensor, "3"]],
            ret_mask: bool = True,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0)
        
        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color
        # Set up rasterization configuration
        # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
        tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera['height']),
            # image_height=int(viewpoint_camera.height),
            # image_width=int(viewpoint_camera.width),
            image_width=int(viewpoint_camera['width']),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera['world_view_transform'],
            projmatrix=viewpoint_camera['full_proj_transform'].float(),
            # projmatrix=viewpoint_camera['projection_maxtrix'].float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera['camera_center'],
            # viewmatrix=viewpoint_camera.world_view_transform,
            # projmatrix=viewpoint_camera.full_proj_transform.float(),
            # sh_degree=self.sh_degree,
            # campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            **({"antialiasing": False} if os.environ.get("LHM_USE_DGR32", "0") == "1" else {}),
        )

        # raster_settings_obj = dgro.GaussianRasterizationSettings(
        #     image_height=int(viewpoint_camera['height']),
        #     image_width=int(viewpoint_camera['width']),
        #     tanfovx=tanfovx,
        #     tanfovy=tanfovy,
        #     bg=bg_color,
        #     scale_modifier=self.scaling_modifier,
        #     viewmatrix=viewpoint_camera['world_view_transform'],
        #     projmatrix=viewpoint_camera['full_proj_transform'].float(),
        #     sh_degree=self.sh_degree,
        #     campos=viewpoint_camera['camera_center'],
        #     prefiltered=False,
        #     debug=False
        # )
        # rasterizer_obj = dgro.GaussianRasterizer(raster_settings=raster_settings_obj)

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = None
            shs = gs.shs.float()

        # Guard against NaN/Inf inputs to the rasterizer.
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

        # Apply global color scale/shift on Gaussian colors (OHTA-style inversion params).
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

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # NOTE that dadong tries to regress rgb not shs
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # ── DEBUG shape check ──
            _n3d = means3D.shape[0]
            for _nm, _t in [('means2D', means2D), ('opacity', opacity), ('scales', scales), ('rotations', rotations)]:
                if _t is not None and _t.shape[0] != _n3d:
                    print(f"\033[91m[SHAPE MISMATCH] means3D={_n3d} but {_nm}={_t.shape}\033[0m")
            if shs is not None and shs.shape[0] != _n3d:
                print(f"\033[91m[SHAPE MISMATCH] means3D={_n3d} but shs={shs.shape}\033[0m")
            if colors_precomp is not None and colors_precomp.shape[0] != _n3d:
                print(f"\033[91m[SHAPE MISMATCH] means3D={_n3d} but colors_precomp={colors_precomp.shape}\033[0m")
            # ── end ──
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
            # _, _, rendered_objects = rasterizer_obj(
            #     means3D = means3D.float(),
            #     means2D = means2D.float(),
            #     shs = shs,
            #     sh_objs = gs.object_dc.float(),
            #     colors_precomp = colors_precomp,
            #     opacities = opacity.float(),
            #     scales = scales.float(),
            #     rotations = rotations.float(),
            #     cov3D_precomp = cov3D_precomp)

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),
            # "comp_obj": rendered_objects.permute(1, 2, 0)
        }

        return ret


    def forward_single_view_32(
            self,
            gs: GaussianModel32,
            viewpoint_camera: Camera,
            background_color: Optional[Float[Tensor, "3"]],
            ret_mask: bool = True,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0)
        
        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color
        # Set up rasterization configuration
        # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
        tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)

        raster_settings = GaussianRasterizationSettings_32(
            image_height=int(viewpoint_camera['height']),
            # image_height=int(viewpoint_camera.height),
            # image_width=int(viewpoint_camera.width),
            image_width=int(viewpoint_camera['width']),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera['world_view_transform'],
            projmatrix=viewpoint_camera['full_proj_transform'].float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera['camera_center'],
            # viewmatrix=viewpoint_camera.world_view_transform,
            # projmatrix=viewpoint_camera.full_proj_transform.float(),
            # sh_degree=self.sh_degree,
            # campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

        rasterizer = GaussianRasterizer_32(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are pr   ovided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = gs.shs.float()
            shs = None

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # NOTE that dadong tries to regress rgb not shs
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
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

        raw_images = rendered_image.unsqueeze(0)  # 1,32,H,W
        rendered_image = self.neural_refiner(raw_images)  # 1,3,H,W


        ret = {
            "comp_rgb": rendered_image.squeeze().permute(1,2,0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            # "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),
        }

        return ret


    def get_mano_model(self, dataset_id):
        """Return a MANOVoxelMeshModel instance for the given dataset_id.
        If dataset_id is None or not provided, return the default `self.mano_model`.
        """
        if dataset_id is None:
            return self.mano_model

        # unwrap tensor/np types
        try:
            if hasattr(dataset_id, "item"):
                dataset_id = int(dataset_id.item())
            else:
                dataset_id = int(dataset_id)
        except Exception:
            # fallback to default
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
            self, gs_attr: GaussianAppOutput, gs_densify_attr, query_points, # dataset_id,
            smplx_data, dataset_id=None, debug=False,
    ):
        """
        query_points: [N, 3]
        """

        device = gs_attr.offset_xyz.device

        # build cano_dependent_pose
        cano_smplx_data_keys = [
            # "R",
            # "Th",
            "shape",
            "poses",
            # "center",
            # "faces",
            # "vertex_index",
        ]

        merge_smplx_data = dict()
        for cano_smplx_data_key in cano_smplx_data_keys:  # make cano pose (A-pose process for body pose params)
            warp_data = smplx_data[cano_smplx_data_key]#.unsqueeze(0)
            cano_pose = torch.zeros_like(warp_data)
            merge_pose = torch.cat([warp_data, cano_pose], dim=0)  # concate posed-params and cano-params
            merge_smplx_data[cano_smplx_data_key] = merge_pose

        merge_smplx_data["shape"] = smplx_data["shape"]
        # merge_smplx_data["transform_mat_neutral_pose"] = smplx_data[
        #     "transform_mat_neutral_pose"]

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            mean_3d = (
                    query_points + gs_attr.offset_xyz
            )  # [N, 3]  # canonical space offset.

            # grid_feature = self.grid_offset.grid(mean_3d)
            # hidden = self.grid_offset.feature_out(grid_feature)
            # grid_offset = self.grid_offset.pos_(hidden)
            #
            # grid_offset[self.face_mask] = grid_offset[self.face_mask] * 0
            #
            # mean_3d = mean_3d + grid_offset

            # matrix to warp predefined pose to zero-pose
            # transform_mat_neutral_pose = merge_smplx_data[
            #     "transform_mat_neutral_pose"]  # [55, 4, 4]
            num_view = merge_smplx_data["poses"].shape[0]  # [Nv, 21, 3]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
            # query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)
            # transform_mat_neutral_pose = transform_mat_neutral_pose.unsqueeze(0).repeat(
            #     num_view, 1, 1, 1)

            # # print(mean_3d.shape, transform_mat_neutral_pose.shape, query_points.shape, smplx_data["body_pose"].shape, smplx_data["betas"].shape)
            # if dataset_id == 0:
            #     mean_3d, transform_matrix = (
            #         self.mano_model.transform_to_posed_verts_from_neutral_pose(
            #             mean_3d,
            #             smplx_data, # merge_smplx_data,
            #             # query_points,
            #             # transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
            #             # device=device,
            #         )
            #     )  # [B, N, 3]
            # else:
            #     mean_3d, transform_matrix = (
            #         self.mano_model_hanco.transform_to_posed_verts_from_neutral_pose(
            #             mean_3d,
            #             smplx_data, # merge_smplx_data,
            #             # query_points,
            #             # transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
            #             # device=device,
            #         )
            #    )  # [B, N, 3]




            # select dataset-specific MANO model (controls center subtraction via center_add)
            mano_for_dataset = self.get_mano_model(dataset_id)
            mean_3d, transform_matrix = (
                mano_for_dataset.transform_to_posed_verts_from_neutral_pose(
                    mean_3d,
                    smplx_data, # merge_smplx_data,
                    # query_points,
                    # transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                    # device=device,
                )
            )  # [B, N, 3]





            # rotation appearance from canonical space to view_posed

            _, N, _, _ = transform_matrix.shape
            transform_rotation = transform_matrix[:, :, :3, :3]

            rigid_rotation_matrix = torch.nn.functional.normalize(
                matrix_to_quaternion(transform_rotation), dim=-1
            )
            # I = matrix_to_quaternion(torch.eye(3)).to(device)

            # # inference constrain
            # is_constrain_body = self.mano_model.is_constrain_body
            # rigid_rotation_matrix[:, is_constrain_body] = I
            rotation_neutral_pose = gs_attr.rotation.unsqueeze(0).repeat(num_view, 1, 1)

            # TODO do not move underarm gs

            # QUATERNION MULTIPLY
            rotation_pose_verts = quaternion_multiply(
                rigid_rotation_matrix, rotation_neutral_pose
            )
            # rotation_pose_verts = rotation_neutral_pose
            # if torch.sum(gs_densify_attr.activation>0.5) > 100:
            densify_mask = (gs_densify_attr.activation>=0.5).squeeze(-1)   # 24608, 1
            # STE trick: forward uses hard_mask, backward uses alpha
            # mask_for_backward = (densify_mask.float() - gs_densify_attr.activation).detach() + gs_densify_attr.activation
            # if not densify_mask.all():  # 说明不是全True
            #     num_active = densify_mask.sum().item()   # True 的数量
            #     total = densify_mask.shape[0]           # 总数
            #     print(f"[DEBUG] Activated faces: {num_active}/{total}")
            # densify_gau_mean = (mean_3d[1, :].unsqueeze(0)[:, self.gs_net.mano_face, :]
            #                     * gs_densify_attr.bary.unsqueeze(0).unsqueeze(-1)).sum(dim=-1)   # 1, 24608, 3
            # densify_gau_mean = densify_gau_mean.squeeze(0)[densify_mask, :]
            # selected_idx = torch.nonzero(densify_mask.float()).squeeze(1)
            triangle_verts = mean_3d[1, :][self.gs_net.mano_face]   # 24608,3,3
            # debug = torch.einsum('nij, ni->nj', triangle_verts, gs_densify_attr.bary)[selected_idx]#
            densify_gau_mean = torch.einsum('nij, ni->nj', triangle_verts, gs_densify_attr.bary)[densify_mask, :]
            per_vert_quat = self.quat_helper(mean_3d[1, :])   # 1, 24608, 3
            tri_quats = per_vert_quat[self.gs_net.mano_face]
            densify_gau_rot = torch.nn.functional.normalize(torch.einsum('bij, bi->bj', tri_quats.squeeze(), gs_densify_attr.bary)) # 24608, 4
            
            densify_gau_rot = torch.nn.functional.normalize(quaternion_multiply(densify_gau_rot, gs_densify_attr.rotation))  # 24608, 4
            # (tri_quats * gs_densify_attr.bary.unsqueeze(0).unsqueeze(-1)).sum(dim=-1).squeeze(0)[densify_mask, :]

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
            
            #
            knn = knn_points(densify_gau_mean.unsqueeze(0), mean_3d[1, :].unsqueeze(0), K=1)
            knn_idx = knn.idx.squeeze()
            # 根据原始标签获取 densify 点的标签
            densify_labels = self.gs_net.objects_dc[knn_idx]
            # densify_gau_scaling = (self.calc_face_areas(mean_3d, self.face_area) + 1e-4) / (self.face_area + 1e-4)
            # 不透明度和颜色直接并进去，不用变形的操作

        # import trimesh
        # mesh1 = trimesh.Trimesh(densify_gau_mean.detach().cpu().numpy())
        # mesh1.export('./hand/desify3.ply')

        gs_list = []
        cano_gs_list = []
        for i in range(num_view):
            gs_copy = GaussianModel(
            # gs_copy = GaussianModel32(
                xyz=mean_3d[i],
                opacity=gs_attr.opacity,
                # rotation=gs_attr.rotation,
                rotation=rotation_pose_verts[i],
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                labels=self.gs_net.objects_dc,
                use_rgb=self.gs_net.use_rgb,
                #####################################
                xyz_densify=densify_gau_mean,
                opacity_densify=gs_densify_attr.opacity[densify_mask, :],
                rotation_densify=densify_gau_rot[densify_mask, :],
                scaling_densify=densify_gau_scaling,
                shs_densify=gs_densify_attr.shs[densify_mask, :],
                labels_densify=densify_labels,
                #####################################
                skip_training_setup=True,
            )  # [N, 3]

            if i == num_view - 2:  # num_view is 2
                cano_gs_list.append(gs_copy)  # 当 i=1 时，作为 cano，而处理的 smplx 并起来的 2个参数中，前者是 posed参数，后者是 cano 无参数
            else:
                gs_list.append(gs_copy)

        return gs_list, cano_gs_list


    def forward_gs_attr(self, x, query_points, global_feature=None, 
                        batch=None, verts_cam=None, nail_image=None, debug=False, x_fine=None, vis_prob=None):
        device = x.device

        # 接收 vis_prob 并存储到 renderer state（用于软可见性融合）
        if vis_prob is not None:
            self.edit_vis_prob = vis_prob

        # reset cache when masking is disabled to avoid leaking stale values
        if not getattr(self, "edit_mask_mode", False) or getattr(self, "edit_vis_mask", None) is None:
            self.edit_invisible_cache = None

        # detect batched inputs (B, N, C) vs per-sample (N, C)
        batched = (x.dim() == 3) or (query_points is not None and query_points.dim() == 3)

        if self.mlp_network_config is not None and self.mlp_net is not None:
            if batched:
                B = x.shape[0]
                x_flat = x.reshape(-1, x.shape[-1])
                x_ori = x_flat
                x_flat = self.mlp_net(x_flat)

                if getattr(self, "edit_mask_mode", False) and torch.is_tensor(getattr(self, "edit_vis_mask", None)):
                    vis_mask = self.edit_vis_mask  # [B, N]
                    mask_flat = (vis_mask > 0.5).reshape(-1, 1).float().to(x_flat.device)

                    # 可选：根据 vis_prob 对可见边界做软权重，减缓更新
                    weight_flat = mask_flat
                    if torch.is_tensor(getattr(self, "edit_vis_prob", None)):
                        vis_prob = self.edit_vis_prob
                        prob_flat = vis_prob.reshape(-1, 1).to(x_flat.device)
                        low, high = float(self.edit_vis_prob_low), float(self.edit_vis_prob_high)
                        soft = ((prob_flat - low) / max(high - low, 1e-6)).clamp(0.0, 1.0)
                        weight_flat = mask_flat * soft  # 不可见仍为 0，可见但低置信度→权重<1

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

                    # invisible_mask = (inv_mask_flat > 0.5).squeeze(-1)
                    # if invisible_mask.any():
                    #     invisible_vals = x_flat[invisible_mask]
                    #     print(f"[DEBUG] Invisible points: {invisible_mask.sum().item()}/{x_flat.shape[0]}, "
                    #           f"mean={invisible_vals.mean().item():.6f}, std={invisible_vals.std().item():.6f}, "
                    #           f"min={invisible_vals.min().item():.6f}, max={invisible_vals.max().item():.6f}")
                else:
                    if getattr(self, "lora_mlp", None) is not None:
                        # x_flat = self.lora_mlp(x_flat)
                        x_lora = self.lora_mlp.try_1(x_ori)
                        x_flat = x_lora + x_flat
                    # x = x_flat.view(B, x.shape[1], -1)

                x = x_flat.view(B, x.shape[1], -1)

                # 保持 x_fine 分支原样（当前流程未使用 fine）
            else:
                x = self.mlp_net(x)

                if getattr(self, "edit_mask_mode", False) and torch.is_tensor(getattr(self, "edit_vis_mask", None)):
                    vis_mask = self.edit_vis_mask.squeeze(0) if self.edit_vis_mask.dim() > 1 else self.edit_vis_mask  # [N]
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
                        vis_mask = self.edit_vis_mask.squeeze(0) if self.edit_vis_mask.dim() > 1 else self.edit_vis_mask  # [N]
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


        # NOTE that gs_attr contains offset xyz
        # z_buf = depth_map[y_valid, x_valid].reshape(-1)
        # z_in = z[mask_inside].reshape(-1)

        # # 深度比较
        # visible_local = z_in <= (z_buf + 5e-2)

        # # 构造 mask
        # visible_mask = np.zeros(batch['world_vertex'][0, ...].shape[1], dtype=bool)
        # visible_mask[mask_inside] = visible_local  
        # visible_mask = torch.from_numpy(visible_mask).to('cuda')             
        # # visible_mask = ~visible_mask
        # # # vis_msk.append(torch.from_numpy(visible_mask).to('cuda'))



        constrain_dict, constrain_head = None, None
        film_param = None
        visible_mask = None

        # === LoRA on gs_net 输入特征（非 batched 情况）===
        if not batched:
            if getattr(self, "lora_gs", None) is not None:
                x = self.lora_gs(x)
                if x_fine is not None:
                    x_fine = self.lora_gs(x_fine)

            gs_attr, gs_density_attr = self.gs_net(
                x, query_points, x_fine, constrain_dict, constrain_head, global_feature, film_param, visible_mask
            )
            return gs_attr, gs_density_attr, visible_mask

        # Batched path: flatten batch and call gs_net once on flattened points
        B, N, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        pts_flat = query_points.reshape(-1, query_points.shape[-1])
        x_fine_flat = None
        if x_fine is not None:
            x_fine_flat = x_fine.reshape(-1, x_fine.shape[-1])

        # # === LoRA on gs_net 输入特征（batched 情况，flatten 之后）===
        if getattr(self, "lora_gs", None) is not None:
            x_flat = self.lora_gs(x_flat)
            if x_fine_flat is not None:
                x_fine_flat = self.lora_gs(x_fine_flat)

        # expand global_feature per point if provided
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

        # reshape outputs back to (B, N, ...)
        batched_gs = {}
        for name, val in gs_attr_flat.__dict__.items():
            if torch.is_tensor(val) and val.shape[0] == B * N:
                new_shape = (B, N) + tuple(val.shape[1:])
                batched_gs[name] = val.view(*new_shape)
            else:
                # broadcast non-point tensors across batch
                batched_gs[name] = val

        batched_densify = {}
        for name, val in gs_density_flat.__dict__.items():
            if torch.is_tensor(val) and hasattr(self.gs_net, 'mano_face') and val.shape[0] == B * (self.gs_net.mano_face.shape[0]):
                # densify outputs are per-face; reshape to (B, F, ...)
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
                # print(smplx_data["betas"].shape, smplx_data["face_offset"].shape, smplx_data["joint_offset"].shape)
                positions, _, transform_mat_neutral_pose = (
                    self.mano_model.get_query_points(smplx_data, device=device)
                )  # [B, N, 3]
        smplx_data["transform_mat_neutral_pose"] = (
            transform_mat_neutral_pose  # [B, 55, 4, 4]
        )
        return positions, smplx_data

    def decoder_cross_attn_wrapper(self, pcl_embed, latent_feat, extra_info):
        # if self.training and self.gradient_checkpointing:
        #     def create_custom_forward(module):
        #         def custom_forward(*inputs):
        #             return module(*inputs)
        #         return custom_forward
        #     ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        #     gs_feats = torch.utils.checkpoint.checkpoint(
        #         create_custom_forward(self.decoder_cross_attn),
        #         pcl_embed.to(dtype=latent_feat.dtype),
        #         latent_feat,
        #         extra_info,
        #         **ckpt_kwargs,
        #     )
        # else:
        gs_feats = self.decoder_cross_attn(
            pcl_embed.to(dtype=latent_feat.dtype), latent_feat, extra_info
        )
        return gs_feats

    def query_latent_feat(
            self,
            positions: Float[Tensor, "*B N1 3"],
            # smplx_data,
            latent_feat: Float[Tensor, "*B N2 C"],
            extra_info,
    ):
        device = latent_feat.device
        if self.skip_decoder:  # this way !!!!!!
            gs_feats = latent_feat  # 1,40000,1024
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

        return gs_feats, positions, # smplx_data

    def forward_single_batch(
            self,
            gs_list: list[GaussianModel32],
            batch,
            # c2ws: Float[Tensor, "Nv 4 4"],
            # intrinsics: Float[Tensor, "Nv 4 4"],
            # trans: Optional[Float[Tensor, "Nv 3"]],
            height: int,
            width: int,
            background_color: Optional[Float[Tensor, "Nv 3"]],
            offset_xyz: Optional[Float[Tensor, "Nv 3"]] = None,
            debug: bool = False,
    ):
        out_list = []
        self.device = gs_list[0].xyz.device

        # intrinsic = intrinsics
        # N_view = batch["poses"].shape[0]

        # for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
        # for v_idx in range(len(batch['original_image'])):
        render_elapsed_time = 0.0  # Only measure forward_single_view time
        for v_idx in range(len(gs_list)):
            # Sync and start timer for render only
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            render_start = time.perf_counter()
            
            out_list.append(
                self.forward_single_view(
                # self.forward_single_view_32(
                    gs_list[0], #[v_idx],
                    batch, # batch[v_idx],
                    # self.get_single_view_cam(batch, v_idx),
                    # Camera.from_c2w(c2w, intrinsic, height, width),
                    background_color, # [v_idx],
                )
            )
            
            # Sync and end timer for render only
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            render_elapsed_time += time.perf_counter() - render_start

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        # out["3dgs"] = gs_list
        out["scaling"] = gs_list[0].scaling    # gs_list
        out['offset'] = offset_xyz
        out['shs'] = gs_list[0].shs
        out['nail_3d_mask'] = gs_list[0].nail_mask
        
        # Store render timing info
        out['render_time'] = render_elapsed_time
        out['render_views'] = len(gs_list)

        # debug = True
        if debug:
            import cv2

            cv2.imwrite(
                "fuck.png",
                (out["comp_rgb"].detach().cpu().numpy()[0, ..., ::-1] * 255).astype(
                    np.uint8
                ),
            )

        return out

    @torch.no_grad()
    def forward_cano_batch(
            self,
            gs_list: list[GaussianModel],
            c2ws: Float[Tensor, "Nv 4 4"],
            intrinsics: Float[Tensor, "Nv 4 4"],
            background_color: Optional[Float[Tensor, "Nv 3"]],
            height: int = 512,
            width: int = 512,
            debug: bool = False,
    ):
        """using to visualization."""
        degree_list = [0, 90, 180, 270]
        out_list = []
        self.device = gs_list[0].xyz.device

        gs_list_copy = [gs_list[0].clone() for _ in range(len(degree_list))]

        rotation_gs_list = []

        for rotation_degree, gs in zip(degree_list, gs_list_copy):
            _R = torch.eye(3).to(gs.xyz)
            _R[-1, -1] *= -1
            _R[1, 1] *= -1

            self_R = torch.from_numpy(generate_rotation_matrix_y(rotation_degree)).to(
                _R
            )
            _R = self_R @ _R

            gs.xyz = (_R @ gs.xyz.T).T

            _min, _max = aabb(gs.xyz)
            center = (_min + _max) / 2
            gs.xyz -= center.unsqueeze(0)

            _R_quaternion = matrix_to_quaternion(_R)
            gs.rotation = quaternion_multiply(_R_quaternion, gs.rotation)

            gs.xyz[..., -1] += 2.5  # move to (0, 0, 3)
            rotation_gs_list.append(gs)

        intrinsics = torch.eye(4).to(intrinsics).unsqueeze(0)
        intrinsics[0, 0, 0] = width
        intrinsics[0, 1, 1] = height
        intrinsics[0, 0, 2] = width / 2
        intrinsics[0, 1, 2] = height / 2

        for v_idx, gs in enumerate(rotation_gs_list):
            out_list.append(
                self.forward_single_view(
                    rotation_gs_list[v_idx],
                    Camera.from_c2w(c2ws[0], intrinsics[0], height, width),
                    torch.ones_like(background_color[0]),
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = rotation_gs_list

        if debug:
            import cv2

            for i in range(4):
                cv2.imwrite(
                    f"fuck_{i}.png",
                    (out["comp_rgb"].detach().cpu().numpy()[i, ..., ::-1] * 255).astype(
                        np.uint8
                    ),
                )

        return out

    def get_single_batch_smpl_data(self, smpl_data, bidx):
        smpl_data_single_batch = {}
        for k, v in smpl_data.items():
            smpl_data_single_batch[k] = v[
                bidx
            ]  # e.g. body_pose: [B, N_v, 21, 3] -> [N_v, 21, 3]
            if k == "betas" or (k == "joint_offset") or (k == "face_offset"):
                smpl_data_single_batch[k] = v[
                                            bidx: bidx + 1
                                            ]  # e.g. betas: [B, 100] -> [1, 100]
        return smpl_data_single_batch


    # def get_single_view_cam(self, cam_data, bidx):
    #     cam_data_single_view = {}
    #     for k, v in cam_data.items():
    #         # if isinstance(v, (torch.Tensor, list)):
    #         if isinstance(v, torch.Tensor):
    #             cam_data_single_view[k] = v[bidx].to('cuda')
    #         elif isinstance(v, list):
    #             cam_data_single_view[k] = v[bidx]
    #         else:
    #             # 跳过 dict 或保留原样
    #             # 你也可以选择打印出来调试
    #             # print(f"Skipping key {k} of type {type(v)}")
    #             # cam_data_single_view[k] = v  # 或者不加这行就彻底跳过
    #             continue
    #     return cam_data_single_view


    def get_single_view_cam(self, cam_data, bidx):
        cam_data_single_view = {}

        for k, v in cam_data.items():
            # Tensor 情况：取第 bidx 个，再搬到 cuda
            if isinstance(v, torch.Tensor):
                # 只做非常轻量的检查
                if v.dim() == 0:
                    raise ValueError(f"[{k}] got scalar tensor, expect batch dim, v={v}")

                if not (0 <= bidx < v.size(0)):
                    raise IndexError(
                        f"[{k}] bidx={bidx} out of range for v.shape={tuple(v.shape)}"
                    )

                try:
                    cam_data_single_view[k] = v[bidx].to('cuda')
                except RuntimeError as e:
                    # 只在真的出错时打印详细信息
                    print("\n[CAM ERROR] key:", k)
                    print("  type:", type(v))
                    print("  shape:", v.shape)
                    print("  device:", v.device)
                    print("  bidx:", bidx)
                    print("  exception:", e)
                    print()
                    raise

            # list 情况：保持原逻辑，只做越界检查
            elif isinstance(v, list):
                if not (0 <= bidx < len(v)):
                    raise IndexError(
                        f"[{k}] list index out of range: bidx={bidx}, len={len(v)}"
                    )
                cam_data_single_view[k] = v[bidx]

            # 其他类型：安静跳过（和你注释里的意图一致）
            else:
                # include certain scalar metadata fields (e.g., dataset_id)
                if k in ("dataset_id", "dataset_name"):
                    cam_data_single_view[k] = v
                else:
                    # 如果想调试再打开
                    # print(f"[INFO] Skipping key {k} with type {type(v)}")
                    continue

        return cam_data_single_view



    def get_single_view_smpl_data(self, smpl_data, vidx):
        smpl_data_single_view = {}
        for k, v in smpl_data.items():
            smpl_data_single_view[k] = v[vidx: vidx + 1, ...]  # e.g. body_pose: [1, N_v, 21, 3] -> [1, 1, 21, 3]
        
        return smpl_data_single_view


    def forward_gs(
            self,
            gs_hidden_features: Float[Tensor, "B Np Cp"],
            query_points: Float[Tensor, "B Np_q 3"],
            global_feature: Float[Tensor, "B C"],
            batches,
            verts_cam, 
            nail_image,
            # nail_mask: Optional[Float[Tensor, "B Np_q"]] = None,
            # smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
            additional_features: Optional[dict] = None,
            color_bias: Optional[Float[Tensor, "B N 3"]] = None,
            opacity_bias: Optional[Float[Tensor, "B N 1"]] = None,
            vis_prob: Optional[Float[Tensor, "B Np_q"]] = None,
            debug: bool = False,
            **kwargs,
    ):

        batch_size = gs_hidden_features.shape[0]

        # obtain gs_features embedding, cur points position, and also smplx params
        # query_gs_features, query_points, smplx_data = self.query_latent_feat(
        #     query_points, smplx_data, gs_hidden_features, additional_features
        # )
        query_gs_features, query_points = self.query_latent_feat(
            query_points, gs_hidden_features, additional_features
        )

        gs_attr_list = []
        densify_gs_attr_list = []
        # Try a batched call to forward_gs_attr to avoid Python-level loop overhead.
        try:
            batched_out = self.forward_gs_attr(
                query_gs_features, query_points, global_feature, batches, verts_cam, nail_image, debug, vis_prob=vis_prob
            )
            # Expect batched_out to be tuple (gs_attr_batch, densify_batch, vis_mask_batch)
            if isinstance(batched_out, tuple) and len(batched_out) >= 2:
                gs_attr_batch, densify_batch = batched_out[0], batched_out[1]
                # If outputs are dataclass-like with batched tensor fields, split into per-sample dataclasses
                if hasattr(gs_attr_batch, "__dict__"):
                    # split GaussianAppOutput-like object
                    for b in range(batch_size):
                        kwargs = {}
                        for name, val in gs_attr_batch.__dict__.items():
                            if torch.is_tensor(val) and val.shape[0] == batch_size:
                                kwargs[name] = val[b]
                            else:
                                # broadcast non-batched fields
                                try:
                                    kwargs[name] = val[b]
                                except Exception:
                                    kwargs[name] = val
                        gs_attr_list.append(GaussianAppOutput(**kwargs))
                else:
                    # If outputs are tensors or lists with batch dim, split into per-sample entries
                    try:
                        for b in range(batch_size):
                            gs_attr_list.append(gs_attr_batch[b])
                    except Exception:
                        raise RuntimeError("Batched forward_gs_attr returned unexpected format")

                # densify outputs
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
            # Fallback: original per-sample loop
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

        # Apply per-point color/opacity bias if provided
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
                    if cb.dim() == 2:  # (N,3)
                        cb = cb.unsqueeze(1)  # (N,1,3)
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

        # Apply global color shift/scale (from inversion) directly on Gaussian colors.
        # Reset the flag first — GS attrs are freshly created so we must re-apply.
        self._color_applied_in_gs = False
        color_shift = getattr(self, "color_shift", None)
        color_scale = getattr(self, "color_scale", None)
        # if os.getenv("DEBUG_GS_SHAPE", "0") == "1":
        #     try:
        #         base_n = int(gs_attr_list[0].shs.shape[0]) if gs_attr_list else -1
        #         dens_n = int(densify_gs_attr_list[0].shs.shape[0]) if densify_gs_attr_list else -1
        #         print(f"[DEBUG_GS_SHAPE] base_shs={base_n}, densify_shs={dens_n}")
        #     except Exception as e:
        #         print(f"[DEBUG_GS_SHAPE] failed: {e}")

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

            # mark applied to avoid double scaling in rasterization
            self._color_applied_in_gs = True

        # Apply a smooth canonical 3D color field after the global affine so
        # Stage1 can correct subject-level appearance without learning high-frequency texture.
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

        return gs_attr_list, densify_gs_attr_list, query_points#, vis_mask#, smplx_data


    def forward_animate_gs(
            self,
            gs_attr_list,
            gs_densify_attr_list, 
            query_points,
            # vis_mask,
            # dataset_id,
            batch,
            smplx_data,
            # c2w,
            # intrinsic,
            # trans,
            height,
            width,
            background_color,
            # model,
            debug=False,
            df_data=None,  # deepfashion-style dataset
    ):
        batch_size = len(gs_attr_list)
        out_list = []
        cano_out_list = []  # inference DO NOT use

        N_view = smplx_data["poses"].shape[0]

        # for b in range(batch_size):
        #     gs_attr = gs_attr_list[b]
        #     query_pt = query_points[b]#.unsqueeze(0)  # [1, N, 3]
        #     # len(animatable_gs_model_list) = num_view
        #     merge_animatable_gs_model_list, cano_gs_model_list = self.animate_gs_model(
        #         gs_attr,
        #         query_pt,
        #         smplx_data,
        #         # self.get_single_batch_smpl_data(smplx_data, b),
        #         debug=debug,
        #     )

        #     animatable_gs_model_list = merge_animatable_gs_model_list[:N_view]

        #     # assert len(animatable_gs_model_list) == c2w.unsqueeze(0).shape[0]

        #     # gs render animated gs model.
        #     out_list.append(
        #         self.forward_single_batch(
        #             animatable_gs_model_list,
        #             batch,
        #             # c2w, # c2w[b],
        #             # intrinsic, # intrinsic[b],
        #             height,
        #             width,
        #             background_color, #background_color[b] if background_color is not None else None,
        #             gs_attr.offset_xyz,
        #             debug=debug,
        #         )
        #     )

        gs_attr = gs_attr_list
        query_pt = query_points
        # len(animatable_gs_model_list) = num_view
        # try to extract dataset_id from provided `batch` so we can select dataset-specific MANO behavior
        dataset_id = None
        try:
            if isinstance(batch, dict) and "dataset_id" in batch:
                val = batch["dataset_id"]
                # tensor scalar
                if isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        dataset_id = int(val.item())
                    else:
                        # take first element of batch
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

        # assert len(animatable_gs_model_list) == c2w.unsqueeze(0).shape[0]

        # gs render animated gs model.
        out_list.append(
            self.forward_single_batch(
                animatable_gs_model_list,
                batch,
                # c2w, # c2w[b],
                # intrinsic, # intrinsic[b],
                height,
                width,
                background_color, #background_color[b] if background_color is not None else None,
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
                # For scalar values (like render_time, render_views), extract single value
                out[k] = v[0] if len(v) > 0 else v

        out["comp_rgb"] = out["comp_rgb"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 3, H, W]
        # out["comp_obj"] = out["comp_obj"].permute(
        #     0, 1, 4, 2, 3
        # ) 
        # out["comp_mask"] = out["comp_mask"].permute(
        #     0, 1, 4, 2, 3
        # )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        out["comp_depth"] = out["comp_depth"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        # NOTE: Do NOT reset _color_applied_in_gs here. The color_shift/scale
        # was already baked into the GS SHS by forward_gs(). The flag must stay
        # True so that subsequent forward_animate_gs calls (e.g. HandAvatar
        # animation followed by test evaluation) do not double-apply the colour
        # transform in forward_single_batch's rasterizer path.
        # The flag is implicitly reset when forward_gs() is called for the next
        # forward pass (it always applies & re-sets the flag).
        return out


    def forward(
            self,
            gs_hidden_features: Float[Tensor, "B Np Cp"],
            query_points: Float[Tensor, "B Np 3"],
            smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
            c2w: Float[Tensor, "B Nv 4 4"],
            intrinsic: Float[Tensor, "B Nv 4 4"],
            height,
            width,
            additional_features: Optional[Float[Tensor, "B C H W"]] = None,
            background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
            debug: bool = False,
            **kwargs,
    ):

        # need shape_params of smplx_data to get querty points and get "transform_mat_neutral_pose"
        # only forward gs params
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



def test():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    smplx_data_root = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/smplx_params_smoothed"
    shape_param_file = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/shape_param.json"

    batch_size = 1
    device = "cuda"
    smplx_data, cam_param_list, ori_image_list = read_smplx_param(
        smplx_data_root=smplx_data_root, shape_param_file=shape_param_file, batch_size=2
    )
    smplx_data_tmp = smplx_data
    for k, v in smplx_data.items():
        smplx_data_tmp[k] = v.unsqueeze(0)
        if (k == "betas") or (k == "face_offset") or (k == "joint_offset"):
            smplx_data_tmp[k] = v[0].unsqueeze(0)
    smplx_data = smplx_data_tmp

    gs_render = GS3DRenderer(
        human_model_path=human_model_path,
        subdivide_num=2,
        smpl_type="smplx",
        feat_dim=64,
        query_dim=64,
        use_rgb=False,
        sh_degree=3,
        mlp_network_config=None,
        xyz_offset_max_step=1.8 / 32,
    )

    gs_render.to(device)
    # print(cam_param_list[0])

    c2w_list = []
    intr_list = []
    for cam_param in cam_param_list:
        c2w = torch.eye(4).to(device)
        c2w[:3, :3] = cam_param["R"]
        c2w[:3, 3] = cam_param["t"]
        c2w_list.append(c2w)
        intr = torch.eye(4).to(device)
        intr[0, 0] = cam_param["focal"][0]
        intr[1, 1] = cam_param["focal"][1]
        intr[0, 2] = cam_param["princpt"][0]
        intr[1, 2] = cam_param["princpt"][1]
        intr_list.append(intr)

    c2w = torch.stack(c2w_list).unsqueeze(0)
    intrinsic = torch.stack(intr_list).unsqueeze(0)

    out = gs_render.forward(
        gs_hidden_features=torch.zeros((batch_size, 2048, 64)).float().to(device),
        query_points=None,
        smplx_data=smplx_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=int(cam_param_list[0]["princpt"][1]) * 2,
        width=int(cam_param_list[0]["princpt"][0]) * 2,
        background_color=torch.tensor([1.0, 1.0, 1.0])
        .float()
        .view(1, 1, 3)
        .repeat(batch_size, 2, 1)
        .to(device),
        debug=False,
    )

    for k, v in out.items():
        if k == "comp_rgb_bg":
            print("comp_rgb_bg", v)
            continue
        for b_idx in range(len(v)):
            if k == "3dgs":
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(f"./debug_vis/{b_idx}_{v_idx}.ply")
                continue
            for v_idx in range(v.shape[1]):
                save_path = os.path.join("./debug_vis", f"{b_idx}_{v_idx}_{k}.jpg")
                cv2.imwrite(
                    save_path,
                    (v[b_idx, v_idx].detach().cpu().numpy() * 255).astype(np.uint8),
                )


def test1():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    device = "cuda"

    # root_dir = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data"
    # meta_path = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/data_list.json"
    # dataset = ExAvatarDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=3,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(224, 224), source_image_res=384)

    # root_dir = "/data1/datasets1/3d_human_data/humman/humman_compressed"
    # meta_path = "/data1/datasets1/3d_human_data/humman/humman_id_debug_list.json"
    # dataset = HuMManDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=3,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384)

    # from openlrm.datasets.static_human import StaticHumanDataset
    # root_dir = "./train_data/static_human_data"
    # meta_path = "./train_data/static_human_data/data_id_list.json"
    # dataset = StaticHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=7,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384,
    #                 debug=False)

    # from openlrm.datasets.singleview_human import SingleViewHumanDataset
    # root_dir = "./train_data/single_view"
    # meta_path = "./train_data/single_view/data_list.json"
    # dataset = SingleViewHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=0,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384,
    #                 debug=False)

    from accelerate.utils import set_seed

    set_seed(1234)
    from LHM.datasets.video_human import VideoHumanDataset

    root_dir = "./train_data/ClothVideo"
    meta_path = "./train_data/ClothVideo/label/valid_id_with_img_list.json"
    dataset = VideoHumanDataset(
        root_dirs=root_dir,
        meta_path=meta_path,
        sample_side_views=7,
        render_image_res_low=384,
        render_image_res_high=384,
        render_region_size=(682, 384),
        source_image_res=384,
        enlarge_ratio=[0.85, 1.2],
        debug=False,
    )

    data = dataset[0]

    def get_smplx_params(data):
        smplx_params = {}
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
            "betas",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                smplx_params[k] = data[k]
        return smplx_params

    smplx_data = get_smplx_params(data)

    smplx_data_tmp = {}
    for k, v in smplx_data.items():
        smplx_data_tmp[k] = v.unsqueeze(0).to(device)
        print(k, v.shape)
    smplx_data = smplx_data_tmp

    c2ws = data["c2ws"].unsqueeze(0).to(device)
    intrs = data["intrs"].unsqueeze(0).to(device)
    render_images = data["render_image"].numpy()
    render_h = data["render_full_resolutions"][0, 0]
    render_w = data["render_full_resolutions"][0, 1]
    render_bg_colors = data["render_bg_colors"].unsqueeze(0).to(device)
    print("c2ws", c2ws.shape, "intrs", intrs.shape, intrs)

    gs_render = GS3DRenderer(
        human_model_path=human_model_path,
        subdivide_num=2,
        smpl_type="smplx",
        feat_dim=64,
        query_dim=64,
        use_rgb=False,
        sh_degree=3,
        mlp_network_config=None,
        xyz_offset_max_step=1.8 / 32,
        expr_param_dim=10,
        shape_param_dim=10,
        fix_opacity=True,
        fix_rotation=True,
    )
    gs_render.to(device)

    out = gs_render.forward(
        gs_hidden_features=torch.zeros((1, 2048, 64)).float().to(device),
        query_points=None,
        smplx_data=smplx_data,
        c2w=c2ws,
        intrinsic=intrs,
        height=render_h,
        width=render_w,
        background_color=render_bg_colors,
        debug=False,
    )
    os.makedirs("./debug_vis/gs_render", exist_ok=True)
    for k, v in out.items():
        if k == "comp_rgb_bg":
            print("comp_rgb_bg", v)
            continue
        for b_idx in range(len(v)):
            if k == "3dgs":
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(
                        f"./debug_vis/gs_render/{b_idx}_{v_idx}.ply"
                    )
                continue
            for v_idx in range(v.shape[1]):
                save_path = os.path.join(
                    "./debug_vis/gs_render", f"{b_idx}_{v_idx}_{k}.jpg"
                )
                img = (
                    v[b_idx, v_idx].permute(1, 2, 0).detach().cpu().numpy() * 255
                ).astype(np.uint8)
                print(img.shape, save_path)
                if "mask" in k:
                    render_img = render_images[v_idx].transpose(1, 2, 0) * 255
                    cv2.imwrite(
                        save_path,
                        np.hstack(
                            [np.tile(img, (1, 1, 3)), render_img.astype(np.uint8)]
                        ),
                    )
                else:
                    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    # test1()
    test()
    test()
    test()
