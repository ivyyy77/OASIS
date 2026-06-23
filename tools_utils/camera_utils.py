#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from torch import nn
# from tools.camera_utils import Camera
import numpy as np
from tools_utils.general_util import PILtoTorch
from tools_utils.graphics_utils import fov2focal, getWorld2View2, getProjectionMatrix, getProjectionMatrix_refine


WARNED = False

def loadCam(args, id, cam_info, resolution_scale):

    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 3200:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    from PIL import Image
    # with Image.open(cam_info.image) as img:
    with Image.open(cam_info.image_path) as img:
        resized_image_rgb = PILtoTorch(img, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.bound_mask is not None:
        resized_bound_mask = PILtoTorch(cam_info.bound_mask, resolution)
    else:
        resized_bound_mask = None

    if cam_info.bkgd_mask is not None:
        # resized_bkgd_mask = PILtoTorch(cam_info.bkgd_mask, resolution)

        with Image.open(cam_info.mask_path) as mask:
            resized_bkgd_mask = PILtoTorch(mask, resolution)

    else:
        resized_bkgd_mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, K=cam_info.K,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, bkgd_mask=resized_bkgd_mask,
                  bound_mask=resized_bound_mask, smpl_param=cam_info.smpl_param,
                  world_vertex=cam_info.world_vertex, world_bound=cam_info.world_bound,
                  big_pose_smpl_param=cam_info.big_pose_smpl_param,
                  big_pose_world_vertex=cam_info.big_pose_world_vertex,
                  big_pose_world_bound=cam_info.big_pose_world_bound,
                  data_device=args.data_device)

def loadCam_aug(args, id, cam_info, resolution_scale):

    gt_image = torch.from_numpy(cam_info.image).permute(2,0,1)
    # gt_image = torch.from_numpy(cam_info.image)
    # from model import libcore
    # import os
    # libcore.write_tensor_image(os.path.join('/data/hand3d/HandAvatar-main/data/InterHand/5/InterHand2.6M_5fps_batch1/gauhand/test/debug_3.jpg'), gt_image, rgb2bgr=True)
    # # mask = torch.from_numpy(cam_info.bkgd_mask)[:,:,0]
    mask = torch.from_numpy(cam_info.bkgd_mask)
    bound_mask = torch.from_numpy(cam_info.bound_mask)
    loaded_mask = None


    # orig_w = cam_info.width
    # orig_h = cam_info.height
    #
    # if args.resolution in [1, 2, 4, 8]:
    #     resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    # else:  # should be a type that converts to float
    #     if args.resolution == -1:
    #         if orig_w > 3200:
    #             global WARNED
    #             if not WARNED:
    #                 print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
    #                     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
    #                 WARNED = True
    #             global_down = orig_w / 1600
    #         else:
    #             global_down = 1
    #     else:
    #         global_down = orig_w / args.resolution
    #
    #     scale = float(global_down) * float(resolution_scale)
    #     resolution = (int(orig_w / scale), int(orig_h / scale))
    #
    #
    # from PIL import Image
    # # with Image.open(cam_info.image) as img:
    # with Image.open(cam_info.image_path) as img:
    #     resized_image_rgb = PILtoTorch(img, resolution)
    #
    # gt_image = resized_image_rgb[:3, ...]
    # loaded_mask = None
    #
    # if resized_image_rgb.shape[1] == 4:
    #     loaded_mask = resized_image_rgb[3:4, ...]
    #
    # if cam_info.bound_mask is not None:
    #     resized_bound_mask = PILtoTorch(cam_info.bound_mask, resolution)
    # else:
    #     resized_bound_mask = None
    #
    # if cam_info.bkgd_mask is not None:
    #     resized_bkgd_mask = cam_info.bkgd_mask
    #
    #     with Image.open(cam_info.mask_path) as mask:
    #         resized_bkgd_mask = PILtoTorch(mask, resolution)
    #
    # else:
    #     resized_bkgd_mask = None


    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, K=cam_info.K,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, extrinsics = cam_info.extrinsics,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, bkgd_mask=mask,
                  bound_mask=bound_mask, smpl_param=cam_info.smpl_param,
                  world_vertex=cam_info.world_vertex, world_bound=cam_info.world_bound,
                  big_pose_smpl_param=cam_info.big_pose_smpl_param,
                  big_pose_world_vertex=cam_info.big_pose_world_vertex,
                  big_pose_world_bound=cam_info.big_pose_world_bound,
                  data_device=args.data_device)


def loadCam_aug_bs(id, cam_info):

    gt_image = torch.from_numpy(cam_info.image).permute(2,0,1)
    # nail_image = torch.from_numpy(cam_info.nail_image).permute(2,0,1)
    mask = torch.from_numpy(cam_info.bkgd_mask)
    nail_mask = torch.from_numpy(cam_info.nail_mask)
    bound_mask = torch.from_numpy(cam_info.bound_mask)
    loaded_mask = None
    # loaded_mask = mask

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, K=cam_info.K,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, # extrinsic = cam_info.extrinsics,
                  image=gt_image, nail_image=cam_info.nail_image, nail_mask=nail_mask,
                  verts_cam=cam_info.verts_cam,
                  gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, bkgd_mask=mask,
                  bound_mask=bound_mask, 
                  smpl_param=cam_info.smpl_param,
                  world_vertex=cam_info.world_vertex, world_bound=cam_info.world_bound,
                  big_pose_smpl_param=cam_info.big_pose_smpl_param,
                  big_pose_world_vertex=cam_info.big_pose_world_vertex,
                  big_pose_world_bound=cam_info.big_pose_world_bound,
                  data_device='cuda')

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        # camera_list.append(loadCam(args, id, c, resolution_scale))
        # yield loadCam(args, id, c, resolution_scale)
        yield loadCam_aug(args, id, c, resolution_scale)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, K, FoVx, FoVy, image, nail_image, nail_mask, verts_cam, gt_alpha_mask, # pose_id,
                 image_name, uid, # extrinsic,
                 bkgd_mask=None, bound_mask=None, 
                 smpl_param=None,
                 world_vertex=None, world_bound=None, big_pose_smpl_param=None,
                 big_pose_world_vertex=None, big_pose_world_bound=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        # self.pose_id = pose_id
        # self.extrinsic = extrinsic
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.bkgd_mask = bkgd_mask
        self.nail_mask = nail_mask
        self.bound_mask = bound_mask

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)#.to(self.data_device)
        self.nail_image = torch.tensor(nail_image) #.clamp(0.0, 1.0)
        self.verts_cam = torch.tensor(verts_cam)
        self.width = self.original_image.shape[2]
        self.height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask#.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.height, self.width)) #torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 1000 #100.0
        self.znear = 0.001 #0.01

        # self.zfar = 100.0
        # self.znear = 0.01

        self.trans = trans
        self.scale = scale
        # # flip z axis in camera coordinates
        # F = np.diag([1, 1, -1]).astype(np.float32)
        # R_gl = F @ R
        # T_gl = F @ T

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)#.cuda()
        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)#.cuda()
        self.projection_matrix = getProjectionMatrix_refine(torch.Tensor(K), self.height, self.width, self.znear, self.zfar).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)#.T
        # self.full_proj_transform = self.projection_matrix @ self.world_view_transform
        self.camera_center = self.world_view_transform.inverse()[3, :3]   # 相机在世界坐标系下的真实 3D 位置 
        # if torch.all(self.camera_center == 0):
        #     self.camera_center = 
        # self.camera_center = self.world_view_transform.inverse()[:3, 3]   # 相机在世界坐标系下的真实 3D 位置 
        # 内参 矩阵 K 里的主点 (principal point) cx, cy，用来告诉渲染器“像素坐标” 的原点在哪。

        self.smpl_param = smpl_to_cuda(smpl_param, self.data_device)
        self.world_vertex = torch.tensor(world_vertex)#.to(self.data_device)
        # hand_center = np.mean(world_vertex, axis=0, keepdims=True).T  # (3,1)       
        # self.camera_center[0] = torch.tensor(hand_center).to(self.data_device)[0] + self.camera_center[0]   # 相机在世界坐标系下的位置
        # self.camera_center = torch.tensor(hand_center).to(self.data_device) - self.camera_center.reshape(3,1)   # 相机在世界坐标系下的位置
        # self.world_bound = torch.tensor(world_bound).to(self.data_device)
        # self.big_pose_smpl_param = smpl_to_cuda(big_pose_smpl_param, self.data_device)
        self.big_pose_world_vertex = torch.tensor(big_pose_world_vertex) #.to(self.data_device)
        self.big_pose_world_bound = torch.tensor(big_pose_world_bound) #.to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def smpl_to_cuda(param, device):
    for key in param:
        if torch.is_tensor(param[key]):
            param[key] = param[key]#.to(device)
        else:
            param[key] = torch.Tensor(param[key])#.to(device)
    return param



def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T.squeeze()
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry



def get_rays_from_KRT(H, W, K, R, T):
    r""" Sample rays on an image based on camera matrices (K, R and T)

    Args:
        - H: Integer
        - W: Integer
        - K: Array (3, 3)
        - R: Array (3, 3) R_CW
        - T: Array (3, ) T_CW
        
    Returns:
        - rays_o: Array (H, W, 3)
        - rays_d: Array (H, W, 3)
    """

    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d



def rays_intersect_3d_bbox(bounds, ray_o, ray_d):
    r"""calculate intersections with 3d bounding box
        Args:
            - bounds: dictionary or list
            - ray_o: (N_rays, 3)
            - ray_d, (N_rays, 3)
        Output:
            - near: (N_VALID_RAYS, )
            - far: (N_VALID_RAYS, )
            - mask_at_box: (N_RAYS, )
    """

    if isinstance(bounds, dict):
        bounds = np.stack([bounds['min_xyz'], bounds['max_xyz']], axis=0)
    assert bounds.shape == (2,3)

    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None] # (N_rays, 2, 3)
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-5] = 1e-5
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6) # (N_rays, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None] # (N_rays, 6, 3)
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))  # (N_rays, 6)
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2  #(N_rays, )
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3) # (N_VALID_rays, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box




