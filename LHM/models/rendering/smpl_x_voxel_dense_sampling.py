# -*- coding: utf-8 -*-# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-01-08 21:42:24, Version 0.0, SMPLX + FLAME2019 + Voxel-Based Queries.
# @Function      : SMPLX-related functions
# @Description   : 1.canonical query, 2.offset, 3.blendshape -> 4.posed-view

import copy
import math
import os
import os.path as osp
import pdb
import pickle
import sys

sys.path.append("./")
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.io import load_ply, save_ply
from pytorch3d.ops import SubdivideMeshes, knn_points
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from smplx.lbs import batch_rigid_transform#,  get_rigid_transformation_torch
from torch.nn import functional as F

from LHM.models.rendering.mesh_utils import Mesh
from LHM.models.rendering.smplx import smplx
from LHM.models.rendering.smplx.smplx.lbs import blend_shapes
from LHM.models.rendering.smplx.vis_utils import render_mesh

from tools.model.smplx.manohd.subdivide import sub_mano
from tools.model.handavatar.configs import cfg
import tools.model.smplx

"""
Subdivide a triangle mesh by adding a new vertex at the center of each edge and dividing each face into four new faces.
Vectors of vertex attributes can also be subdivided by averaging the values of the attributes at the two vertices which form each edge. 
This implementation preserves face orientation - if the vertices of a face are all ordered counter-clockwise, 
then the faces in the subdivided meshes will also have their vertices ordered counter-clockwise.
If meshes is provided as an input, the initializer performs the relatively expensive computation of determining the new face indices. 
This one-time computation can be reused for all meshes with the same face topology but different vertex positions.
"""


def avaliable_device():

    import torch

    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


class SMPLX_Mesh(object):
    def __init__(
        self,
        human_model_path,
        shape_param_dim=100,
        expr_param_dim=50,
        subdivide_num=2,
        cano_pose_type=0,
    ):
        """SMPLX using dense sampling"""
        super().__init__()
        self.human_model_path = human_model_path
        self.shape_param_dim = shape_param_dim
        self.expr_param_dim = expr_param_dim
        if shape_param_dim == 10 and expr_param_dim == 10:
            self.layer_arg = {
                "create_global_orient": False,
                "create_body_pose": False,
                "create_left_hand_pose": False,
                "create_right_hand_pose": False,
                "create_jaw_pose": False,
                "create_leye_pose": False,
                "create_reye_pose": False,
                "create_betas": False,
                "create_expression": False,
                "create_transl": False,
            }
            self.layer = {
                gender: smplx.create(
                    human_model_path,
                    "smplx",
                    gender=gender,
                    num_betas=self.shape_param_dim,
                    num_expression_coeffs=self.expr_param_dim,
                    use_pca=False,
                    use_face_contour=False,
                    flat_hand_mean=False,
                    # flat_hand_mean=True,
                    **self.layer_arg,
                )
                for gender in ["neutral", "male", "female"]
            }
        else:
            self.layer_arg = {
                "create_global_orient": False,
                "create_body_pose": False,
                "create_left_hand_pose": False,
                "create_right_hand_pose": False,
                "create_jaw_pose": False,
                "create_leye_pose": False,
                "create_reye_pose": False,
                "create_betas": False,
                "create_expression": False,
                "create_transl": False,
            }
            self.layer = {
                gender: smplx.create(
                    human_model_path,
                    "smplx",
                    gender=gender,
                    num_betas=self.shape_param_dim,
                    num_expression_coeffs=self.expr_param_dim,
                    use_pca=False,
                    use_face_contour=True,
                    flat_hand_mean=False,
                    # flat_hand_mean=True,
                    **self.layer_arg,
                )
                for gender in ["neutral", "male", "female"]
            }

        self.face_vertex_idx = np.load(
            osp.join(human_model_path, "smplx", "SMPL-X__FLAME_vertex_ids.npy")
        )
        if shape_param_dim == 10 and expr_param_dim == 10:
            print("not using flame expr")
        else:
            self.layer = {
                gender: self.get_expr_from_flame(self.layer[gender])
                for gender in ["neutral", "male", "female"]
            }
        self.vertex_num = 10475
        self.face_orig = self.layer["neutral"].faces.astype(np.int64)
        self.is_cavity, self.face = self.add_cavity()
        with open(
            osp.join(human_model_path, "smplx", "MANO_SMPLX_vertex_ids.pkl"), "rb"
        ) as f:
            hand_vertex_idx = pickle.load(f, encoding="latin1")

        with open(
                osp.join("./data", "FLAME_masks.pkl"), "rb"
        ) as f:
            face_vertex_idx = pickle.load(f, encoding="latin1")

        # self.debug_1 = np.load(osp.join('./data/debug_face/face_model.npy'))
        # self.debug_11 = np.load('./data/debug_face/face_model.npy',allow_pickle=True).item()
        # self.debug_2 = np.load(osp.join('./data/debug_face/indices_38365_35709.npy'))
        self.rhand_vertex_idx = hand_vertex_idx["right_hand"]
        self.lhand_vertex_idx = hand_vertex_idx["left_hand"]
        self.expr_vertex_idx = self.get_expr_vertex_idx()

        # SMPLX joint set
        self.joint_num = (
            55  # 22 (body joints: 21 + 1) + 3 (face joints) + 30 (hand joints)
        )
        self.joints_name = (
            "Pelvis",
            "L_Hip",
            "R_Hip",
            "Spine_1",
            "L_Knee",
            "R_Knee",
            "Spine_2",
            "L_Ankle",
            "R_Ankle",
            "Spine_3",
            "L_Foot",
            "R_Foot",
            "Neck",
            "L_Collar",
            "R_Collar",
            "Head",
            "L_Shoulder",  # 16
            "R_Shoulder",  # 17
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",  # body joints
            "Jaw",
            "L_Eye",
            "R_Eye",  # face joints
            "L_Index_1",
            "L_Index_2",
            "L_Index_3",
            "L_Middle_1",
            "L_Middle_2",
            "L_Middle_3",
            "L_Pinky_1",
            "L_Pinky_2",
            "L_Pinky_3",
            "L_Ring_1",
            "L_Ring_2",
            "L_Ring_3",
            "L_Thumb_1",
            "L_Thumb_2",
            "L_Thumb_3",  # left hand joints
            "R_Index_1",
            "R_Index_2",
            "R_Index_3",
            "R_Middle_1",
            "R_Middle_2",
            "R_Middle_3",
            "R_Pinky_1",
            "R_Pinky_2",
            "R_Pinky_3",
            "R_Ring_1",
            "R_Ring_2",
            "R_Ring_3",
            "R_Thumb_1",
            "R_Thumb_2",
            "R_Thumb_3",  # right hand joints
        )
        self.root_joint_idx = self.joints_name.index("Pelvis")
        self.joint_part = {
            "body": range(
                self.joints_name.index("Pelvis"), self.joints_name.index("R_Wrist") + 1
            ),
            "face": range(
                self.joints_name.index("Jaw"), self.joints_name.index("R_Eye") + 1
            ),
            "lhand": range(
                self.joints_name.index("L_Index_1"),
                self.joints_name.index("L_Thumb_3") + 1,
            ),
            "rhand": range(
                self.joints_name.index("R_Index_1"),
                self.joints_name.index("R_Thumb_3") + 1,
            ),
            "lower_body": [
                self.joints_name.index("Pelvis"),
                self.joints_name.index("R_Hip"),
                self.joints_name.index("L_Hip"),
                self.joints_name.index("R_Knee"),
                self.joints_name.index("L_Knee"),
                self.joints_name.index("R_Ankle"),
                self.joints_name.index("L_Ankle"),
                self.joints_name.index("R_Foot"),
                self.joints_name.index("L_Foot"),
            ],
        }

        self.joint_part['upper_body']= self.upper_body_label()

        self.lower_body_vertex_idx = self.get_body("lower_body")
        self.upper_body_vertex_idx = self.get_body("upper_body")

        self.neutral_body_pose = torch.zeros(  # 21, 3
            (len(self.joint_part["body"]) - 1, 3)
        )  # 大 pose in axis-angle representation (body pose without root joint)

        # cano_pose_type=0
        if cano_pose_type == 0:  # exavatar-cano-pose    腿全张开的 大，与原来的 ZJU 数据集处理一样的cano
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1])
        else:  # this way                                腿半张开 ！！！ 用的是这个
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, math.pi / 9])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -math.pi / 9])

        self.neutral_jaw_pose = torch.FloatTensor([1 / 3, 0, 0])

        # subdivider
        self.body_head_mapping = self.get_body_face_mapping()

        self.register_constrain_prior()
    
    def upper_body_label(self):

        upper_body_name = [
            "Pelvis",
            "Spine_1",
            "Spine_2",
            "Spine_3",
            "L_Collar",
            "R_Collar",
            "L_Shoulder",  # 16
            "R_Shoulder",  # 17
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",  # body joints
            "L_Index_1",
            "L_Index_2",
            "L_Index_3",
            "L_Middle_1",
            "L_Middle_2",
            "L_Middle_3",
            "L_Pinky_1",
            "L_Pinky_2",
            "L_Pinky_3",
            "L_Ring_1",
            "L_Ring_2",
            "L_Ring_3",
            "L_Thumb_1",
            "L_Thumb_2",
            "L_Thumb_3",  # left hand joints
            "R_Index_1",
            "R_Index_2",
            "R_Index_3",
            "R_Middle_1",
            "R_Middle_2",
            "R_Middle_3",
            "R_Pinky_1",
            "R_Pinky_2",
            "R_Pinky_3",
            "R_Ring_1",
            "R_Ring_2",
            "R_Ring_3",
            "R_Thumb_1",
            "R_Thumb_2",
            "R_Thumb_3",  # right hand joints
        ]

        upper_body_idx_list  = []
        for upper_name in upper_body_name:
            upper_idx = self.joints_name.index(upper_name)
            upper_body_idx_list.append(upper_idx)


        return upper_body_idx_list 

    def register_constrain_prior(self):
        """As video cannot provide insufficient supervision for the canonical space, we add some human prior to constrain the rotation. Although it is a trick, it is very effective."""
        constrain_body = np.load(
            "./pretrained_models/voxel_grid/human_prior_constrain.npz"
        )["masks"]

        self.constrain_body_vertex_idx = np.where(constrain_body > 0)[0]

    def get_body(self, name):
        """using skinning to find lower body vertices."""
        lower_body_skinning_index = set(self.joint_part[name])
        skinning_weight = self.layer["neutral"].lbs_weights.float()
        skinning_part = skinning_weight.argmax(1)
        skinning_part = skinning_part.cpu().numpy()
        lower_body_vertice_idx = []
        for v_id, v_s in enumerate(skinning_part):
            if v_s in lower_body_skinning_index:
                lower_body_vertice_idx.append(v_id)

        lower_body_vertice_idx = np.asarray(lower_body_vertice_idx)

        return lower_body_vertice_idx

    def get_expr_from_flame(self, smplx_layer):
        flame_layer = smplx.create(
            self.human_model_path,
            "flame",
            gender="neutral",
            num_betas=self.shape_param_dim,
            num_expression_coeffs=self.expr_param_dim,
        )
        smplx_layer.expr_dirs[self.face_vertex_idx, :, :] = flame_layer.expr_dirs
        return smplx_layer

    def set_id_info(self, shape_param, face_offset, joint_offset, locator_offset):
        self.shape_param = shape_param
        self.face_offset = face_offset
        self.joint_offset = joint_offset
        self.locator_offset = locator_offset

    def get_joint_offset(self, joint_offset):
        device = joint_offset.device
        batch_size = joint_offset.shape[0]
        weight = torch.ones((batch_size, self.joint_num, 1)).float().to(device)
        weight[:, self.root_joint_idx, :] = 0
        joint_offset = joint_offset * weight
        return joint_offset

    def get_subdivider(self, subdivide_num):
        vert = self.layer["neutral"].v_template.float().cuda()
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None, :, :], face[None, :, :])

        if subdivide_num > 0:
            subdivider_list = [SubdivideMeshes(mesh)]
            for i in range(subdivide_num - 1):
                mesh = subdivider_list[-1](mesh)
                subdivider_list.append(SubdivideMeshes(mesh))
        else:
            subdivider_list = [mesh]
        return subdivider_list

    def get_body_face_mapping(self):
        face_vertex_idx = self.face_vertex_idx
        face_vertex_set = set(face_vertex_idx)
        face = self.face.reshape(-1).tolist()
        face_label = [f in face_vertex_set for f in face]
        face_label = np.asarray(face_label).reshape(-1, 3)
        face_label = face_label.sum(-1)
        face_id = np.where(face_label == 3)[0]

        head_face = self.face[face_id]

        body_set = set(np.arange(self.vertex_num))
        body_v_id = body_set - face_vertex_set
        body_v_id = np.array(list(body_v_id))

        body_face_id = np.where(face_label == 0)[0]
        body_face = self.face[body_face_id]

        ret_dict = dict(
            head=dict(face=head_face, vert=face_vertex_idx),
            body=dict(face=body_face, vert=body_v_id),
        )

        return ret_dict

    def add_cavity(self):
        lip_vertex_idx = [2844, 2855, 8977, 1740, 1730, 1789, 8953, 2892]
        is_cavity = np.zeros((self.vertex_num), dtype=np.float32)
        is_cavity[lip_vertex_idx] = 1.0

        cavity_face = [[0, 1, 7], [1, 2, 7], [2, 3, 5], [3, 4, 5], [2, 5, 6], [2, 6, 7]]
        face_new = list(self.face_orig)
        for face in cavity_face:
            v1, v2, v3 = face
            face_new.append(
                [lip_vertex_idx[v1], lip_vertex_idx[v2], lip_vertex_idx[v3]]
            )
        face_new = np.array(face_new, dtype=np.int64)
        return is_cavity, face_new

    def get_expr_vertex_idx(self):
        # FLAME 2020 has all vertices of expr_vertex_idx. use FLAME 2019
        """
        SMPLX + FLAME2019 Version
        according to LBS weights to search related vertices ID
        """

        with open(
            osp.join(self.human_model_path, "flame", "2019", "generic_model.pkl"), "rb"
        ) as f:
            flame_2019 = pickle.load(f, encoding="latin1")
        vertex_idxs = np.where(     # 从 FLAME 模型中找出哪些顶点受到表情参数（expression parameters）影响     (3360,)
            (flame_2019["shapedirs"][:, :, 300 : 300 + self.expr_param_dim] != 0).sum(
                (1, 2)
            )
            > 0
        )[
            0
        ]  # FLAME.SHAPE_SPACE_DIM == 300

        # exclude neck and eyeball regions
        flame_joints_name = ("Neck", "Head", "Jaw", "L_Eye", "R_Eye")
        expr_vertex_idx = []
        flame_vertex_num = flame_2019["v_template"].shape[0]
        is_neck_eye = torch.zeros((flame_vertex_num)).float()
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("Neck")
        ] = 1
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("L_Eye")
        ] = 1
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("R_Eye")
        ] = 1
        for idx in vertex_idxs:
            if is_neck_eye[idx]:
                continue
            expr_vertex_idx.append(idx)

        expr_vertex_idx = np.array(expr_vertex_idx)
        expr_vertex_idx = self.face_vertex_idx[expr_vertex_idx]

        return expr_vertex_idx

    def get_arm(self, mesh_neutral_pose, skinning_weight):
        normal = (
            Meshes(
                verts=mesh_neutral_pose[None, :, :],
                faces=torch.LongTensor(self.face_upsampled).cuda()[None, :, :],
            )
            .verts_normals_packed()
            .reshape(self.vertex_num_upsampled, 3)
            .detach()
        )
        part_label = skinning_weight.argmax(1)
        is_arm = 0
        for name in ("R_Shoulder", "R_Elbow", "L_Shoulder", "L_Elbow"):
            is_arm = is_arm + (part_label == self.joints_name.index(name))
        is_arm = is_arm > 0
        is_upper_arm = is_arm * (normal[:, 1] > math.cos(math.pi / 3))
        is_lower_arm = is_arm * (normal[:, 1] <= math.cos(math.pi / 3))
        return is_upper_arm, is_lower_arm


class SMPLXVoxelMeshModel(nn.Module):
    def __init__(
        self,
        human_model_path,
        gender,
        subdivide_num,
        expr_param_dim=50,
        shape_param_dim=100,
        cano_pose_type=0,
        body_face_ratio=3,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
    ) -> None:
        super().__init__()

        # expr_param_dim = 10

        self.smpl_x = SMPLX_Mesh(
            human_model_path=human_model_path,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            subdivide_num=subdivide_num,
            cano_pose_type=cano_pose_type,
        )
        self.smplx_layer = copy.deepcopy(self.smpl_x.layer[gender])

        # # DEBUG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # from LHM.models.rendering.smplx import smplx
        # # from LHM.models.rendering.smplx.vis_utils import render_mesh
        #
        # layer_arg = {
        #     "create_global_orient": False,
        #     "create_body_pose": False,
        #     "create_left_hand_pose": False,
        #     "create_right_hand_pose": False,
        #     "create_jaw_pose": False,
        #     "create_leye_pose": False,
        #     "create_reye_pose": False,
        #     "create_betas": False,
        #     "create_expression": False,
        #     "create_transl": False,
        # }
        #
        # self.smplx_layer = smplx.create(
        #     human_model_path,
        #     "smplx",
        #     gender="neutral",
        #     num_betas=10,
        #     num_expression_coeffs=100,
        #     use_pca=False,
        #     use_face_contour=False,
        #     flat_hand_mean=False,
        #     **layer_arg,
        # )
        # # DEBUG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.pose_mean__ = self.smplx_layer.pose_mean
        self.pose_mean_ = torch.zeros_like(self.pose_mean__).to('cuda')
        self.pose_mean_[75:120] = self.smplx_layer.left_hand_mean
        self.pose_mean_[120:165] = self.smplx_layer.right_hand_mean
        # self.pose_mean_[66:111] = self.smplx_layer.left_hand_mean
        # self.pose_mean_[111:156] = self.smplx_layer.right_hand_mean
        
        # register
        self.apply_pose_blendshape = apply_pose_blendshape
        self.cano_pose_type = cano_pose_type
        self.dense_sample(body_face_ratio, dense_sample_points)
        self.smplx_init()

    def rebuild_mesh(self, v, vertices_id, faces_id, num_dense_samples):
        choice_vertices = v[vertices_id]

        new_mapping = dict()

        for new_id, vertice_id in enumerate(vertices_id):
            new_mapping[vertice_id] = new_id

        faces_id_list = faces_id.reshape(-1).tolist()

        new_faces_id = []
        for face_id in faces_id_list:
            new_faces_id.append(new_mapping[face_id])
        new_faces_id = torch.from_numpy(np.array(new_faces_id).reshape(faces_id.shape))

        mymesh = Mesh(v=choice_vertices, f=new_faces_id)

        dense_sample_pts = mymesh.sample_surface(num_dense_samples).detach().cpu()

        return dense_sample_pts

    def dense_sample(self, body_face_ratio, dense_sample_points):

        buff_path = f"./pretrained_models/dense_sample_points/{self.cano_pose_type}_{dense_sample_points}.ply"  # 1_40000.ply

        # debug_hand = f"./pretrained_models/dense_sample_points/manohd_semantic.ply"  # 1_40000.ply

        if os.path.exists(buff_path):
            dense_sample_pts, _ = load_ply(buff_path)
            # manohd_pts, _ = load_ply(debug_hand)
            
            _bin = dense_sample_points // (body_face_ratio + 1)
            body_pts = int(_bin * body_face_ratio)
            self.is_body = torch.arange(dense_sample_pts.shape[0])
            self.is_body[:body_pts] = 1 # first 30000 are 1  # 前3w个是body，后1w个是face？
            self.is_body[body_pts:] = 0 # later 10000 are 0
            self.dense_pts = dense_sample_pts
            # self.hand_pts = manohd_pts
        else:
            smpl_x = self.smpl_x
            body_face_mapping = smpl_x.get_body_face_mapping()
            face = smpl_x.face
            template_verts = self.smplx_layer.v_template

            _bin = dense_sample_points // (body_face_ratio + 1)

            # build body mesh
            body_pts = int(_bin * body_face_ratio)
            body_dict = body_face_mapping["body"]
            face = body_dict["face"]
            verts = body_dict["vert"]

            dense_body_pts = self.rebuild_mesh(template_verts, verts, face, body_pts)

            # build face mesh
            head_pts = int(_bin)
            head_dict = body_face_mapping["head"]
            head_face = head_dict["face"]
            head_verts = head_dict["vert"]
            dense_head_pts = self.rebuild_mesh(
                template_verts, head_verts, head_face, head_pts
            )

            self.dense_pts = torch.cat([dense_body_pts, dense_head_pts], dim=0)
            self.is_body = torch.arange(self.dense_pts.shape[0])
            self.is_body[:body_pts] = 1
            self.is_body[body_pts:] = 0

            save_ply(buff_path, self.dense_pts)

    @torch.no_grad()
    def voxel_smooth_register(
        self, voxel_v, template_v, lbs_weights, k=3, smooth_k=30, smooth_n=3000
    ):
        """Smooth KNN to handle skirt deformation."""

        lbs_weights = lbs_weights.cuda()

        dist = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            template_v.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        )
        mesh_dis = torch.sqrt(dist.dists)
        mesh_indices = dist.idx.squeeze(0, -1)
        knn_lbs_weights = lbs_weights[mesh_indices]

        mesh_dis = mesh_dis.squeeze()

        print(f"Using k = {smooth_k}, N={smooth_n} for LBS smoothing")
        # Smooth Skinning

        knn_dis = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            voxel_v.unsqueeze(0).cuda(),
            K=smooth_k + 1,
            return_nn=True,
        )
        voxel_dis = torch.sqrt(knn_dis.dists)
        voxel_indices = knn_dis.idx
        voxel_indices = voxel_indices.squeeze()[:, 1:]
        voxel_dis = voxel_dis.squeeze()[:, 1:]

        knn_weights = 1.0 / (mesh_dis[voxel_indices] * voxel_dis)
        knn_weights = knn_weights / knn_weights.sum(-1, keepdim=True)  # [N, K]

        def dists_to_weights(
            dists: torch.Tensor, low: float = None, high: float = None
        ):
            if low is None:
                low = high
            if high is None:
                high = low
            assert high >= low
            weights = dists.clone()
            weights[dists <= low] = 0.0
            weights[dists >= high] = 1.0
            indices = (dists > low) & (dists < high)
            weights[indices] = (dists[indices] - low) / (high - low)
            return weights

        update_weights = dists_to_weights(mesh_dis, low=0.01).unsqueeze(-1)  # [N, 1]

        from tqdm import tqdm

        for _ in tqdm(range(smooth_n)):
            N, _ = update_weights.shape
            new_lbs_weights_chunk_list = []
            for chunk_i in range(0, N, 1000000):

                knn_weights_chunk = knn_weights[chunk_i : chunk_i + 1000000]
                voxel_indices_chunk = voxel_indices[chunk_i : chunk_i + 1000000]

                new_lbs_weights_chunk = torch.einsum(
                    "nk,nkj->nj",
                    knn_weights_chunk,
                    knn_lbs_weights[voxel_indices_chunk],
                )
                new_lbs_weights_chunk_list.append(new_lbs_weights_chunk)
            new_lbs_weights = torch.cat(new_lbs_weights_chunk_list, dim=0)
            if update_weights is None:
                knn_lbs_weights = new_lbs_weights
            else:
                knn_lbs_weights = (
                    1.0 - update_weights
                ) * knn_lbs_weights + update_weights * new_lbs_weights

        return knn_lbs_weights

    def voxel_skinning_init(self, scale_ratio=1.05, voxel_size=256):

        skinning_weight = self.smplx_layer.lbs_weights.float()   # 10475,55

        smplx_data = {"betas": torch.zeros(1, self.smpl_x.shape_param_dim)}
        device = skinning_weight.device

        _, mesh_neutral_pose_wo_upsample, _ = self.get_neutral_pose_human(    # 1,10475,3
            jaw_zero_pose=True,
            use_id_info=True,
            shape_param=smplx_data["betas"],
            device=device,
            face_offset=smplx_data.get("face_offset", None),
            joint_offset=smplx_data.get("joint_offset", None),
        )

        template_verts = mesh_neutral_pose_wo_upsample.squeeze(0)

        def scale_voxel_size(template_verts, scale_ratio=1.0):
            min_values, _ = torch.min(template_verts, dim=0)
            max_values, _ = torch.max(template_verts, dim=0)

            center = (min_values + max_values) / 2
            size = max_values - min_values

            scale_size = size * scale_ratio

            upper = center + scale_size / 2
            bottom = center - scale_size / 2

            return torch.cat([bottom[:, None], upper[:, None]], dim=1)

        mini_size_bbox = scale_voxel_size(template_verts, scale_ratio)
        z_voxel_size = voxel_size // 2

        # build coordinate
        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
            voxel_size - 1
        )  # from 0 to 255，
        y_range = np.linspace(0, voxel_size - 1, voxel_size) / (voxel_size - 1)
        z_range = np.linspace(0, z_voxel_size - 1, z_voxel_size) / (z_voxel_size - 1)

        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        coordinates = torch.from_numpy(np.stack([x, y, z], axis=-1))

        coordinates[..., 0] = mini_size_bbox[0, 0] + coordinates[..., 0] * (
            mini_size_bbox[0, 1] - mini_size_bbox[0, 0]
        )
        coordinates[..., 1] = mini_size_bbox[1, 0] + coordinates[..., 1] * (
            mini_size_bbox[1, 1] - mini_size_bbox[1, 0]
        )
        coordinates[..., 2] = mini_size_bbox[2, 0] + coordinates[..., 2] * (
            mini_size_bbox[2, 1] - mini_size_bbox[2, 0]
        )

        coordinates = coordinates.view(-1, 3).float()
        coordinates = coordinates.cuda()

        if os.path.exists(f"./pretrained_models/voxel_grid/voxel_{voxel_size}.pth"):
            print(f"load voxel_grid voxel_{voxel_size}.pth")
            voxel_flat = torch.load(
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
                map_location=avaliable_device(),
            )
        else:
            voxel_flat = self.voxel_smooth_register(
                coordinates, template_verts, skinning_weight, k=1, smooth_n=3000
            )

            torch.save(
                voxel_flat,
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
            )

        N, LBS_F = voxel_flat.shape

        # x, y, z, C
        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

        # [W H D 55]->[55, D, H, W]
        voxel_grid = voxel_grid_original.permute(3, 2, 1, 0)

        return voxel_grid, mini_size_bbox

    def smplx_init(self):
        """
        Initialize the sub-devided smplx model by registering buffers for various attributes
        This method performs the following steps:
        1. Upsamples the mesh and other assets.
        2. Computes skinning weights, pose directions, expression directions, and various flags for different body parts.
        3. Reshapes and permutes the pose and expression directions.
        4. Converts the flags to boolean values.
        5. Registers buffers for the computed attributes.
        Args:
            self: The object instance.
        Returns:
            None
        """

        def _query(weights, indx):

            weights = weights.squeeze(0)
            assert weights.dim() == 2

            return weights[indx]

        smpl_x = self.smpl_x

        # using KNN to query subdivided mesh
        dense_pts = self.dense_pts.cuda()   # 40000,3
        template_verts = self.smplx_layer.v_template    # 10475, 3

        nn_vertex_idxs = knn_points(  # 查找每个gaussian点最近的mesh顶点的索引   1, 40000, 1
            dense_pts.unsqueeze(0).cuda(),
            template_verts.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        ).idx
        query_indx = nn_vertex_idxs.squeeze(0, -1).detach().cpu()    # (40000,)

        skinning_weight = self.smplx_layer.lbs_weights.float()     # 10475,55

        """ PCA regression function w.r.t vertices offset
        """
        pose_dirs = self.smplx_layer.posedirs.permute(1, 0).reshape(    # 10475,1458
            smpl_x.vertex_num, 3 * (smpl_x.joint_num - 1) * 9
        )
        expr_dirs = self.smplx_layer.expr_dirs.view( # 10475, 300  
            smpl_x.vertex_num, 3 * smpl_x.expr_param_dim
        )
        shape_dirs = self.smplx_layer.shapedirs.view( # 10475, 30
            smpl_x.vertex_num, 3 * smpl_x.shape_param_dim
        )

        (
            is_rhand,
            is_lhand,
            is_face,
            is_face_expr,
            is_lower_body,
            is_upper_body,
            is_constrain_body,
        ) = (
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
        )
        (
            is_rhand[smpl_x.rhand_vertex_idx],
            is_lhand[smpl_x.lhand_vertex_idx],
            is_face[smpl_x.face_vertex_idx],
            is_face_expr[smpl_x.expr_vertex_idx],
            is_lower_body[smpl_x.lower_body_vertex_idx],
            is_upper_body[smpl_x.upper_body_vertex_idx],
            is_constrain_body[smpl_x.constrain_body_vertex_idx],
        ) = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        is_cavity = torch.FloatTensor(smpl_x.is_cavity)[:, None]

        skinning_weight = _query(skinning_weight, query_indx)    # 40000,55
        pose_dirs = _query(pose_dirs, query_indx)
        shape_dirs = _query(shape_dirs, query_indx)
        expr_dirs = _query(expr_dirs, query_indx)
        is_rhand = _query(is_rhand, query_indx)
        is_lhand = _query(is_lhand, query_indx)
        is_face = _query(is_face, query_indx)
        is_face_expr = _query(is_face_expr, query_indx)
        is_lower_body = _query(is_lower_body, query_indx)
        is_upper_body = _query(is_upper_body, query_indx)
        is_constrain_body = _query(is_constrain_body, query_indx)
        is_cavity = _query(is_cavity, query_indx)

        vertex_num_upsampled = self.dense_pts.shape[0]

        pose_dirs = pose_dirs.reshape(  # 486, 120000
            vertex_num_upsampled * 3, (smpl_x.joint_num - 1) * 9
        ).permute(1, 0)
        expr_dirs = expr_dirs.view(vertex_num_upsampled, 3, smpl_x.expr_param_dim) # 40000,3,100
        shape_dirs = shape_dirs.view(vertex_num_upsampled, 3, smpl_x.shape_param_dim) # 40000,3,10

        (
            is_rhand,
            is_lhand,
            is_face,
            is_face_expr,
            is_lower_body,
            is_upper_body,
            is_constrain_body,
        ) = ( # 把 0,1 换成 T/F
            is_rhand[:, 0] > 0,
            is_lhand[:, 0] > 0,
            is_face[:, 0] > 0,
            is_face_expr[:, 0] > 0,
            is_lower_body[:, 0] > 0,
            is_upper_body[:, 0] > 0,
            is_constrain_body[:, 0] > 0,
        )
        is_cavity = is_cavity[:, 0] > 0

        # self.register_buffer('pos_enc_mesh', xyz)
        self.register_buffer("skinning_weight", skinning_weight.contiguous())
        self.register_buffer("pose_dirs", pose_dirs.contiguous())
        self.register_buffer("expr_dirs", expr_dirs.contiguous())
        self.register_buffer("shape_dirs", shape_dirs.contiguous())
        self.register_buffer("is_rhand", is_rhand.contiguous())
        self.register_buffer("is_lhand", is_lhand.contiguous())
        self.register_buffer("is_face", is_face.contiguous())
        self.register_buffer("is_face_expr", is_face_expr.contiguous())
        self.register_buffer("is_lower_body", is_lower_body.contiguous())
        self.register_buffer("is_upper_body", is_upper_body.contiguous())
        self.register_buffer("is_constrain_body", is_constrain_body.contiguous())
        self.register_buffer("is_cavity", is_cavity.contiguous())

        self.vertex_num_upsampled = vertex_num_upsampled   # 40000
        self.smpl_x.vertex_num_upsampled = vertex_num_upsampled  # compatible with SMPLX

        voxel_skinning_weight, voxel_bbox = self.voxel_skinning_init(voxel_size=192)
        # voxel_skinning_weight: 55,96,192,192      voxel_bbox: 3,2
        self.register_buffer("voxel_ws", voxel_skinning_weight)
        self.register_buffer("voxel_bbox", voxel_bbox)

        # self.query_voxel_debug()
    

    def get_body_infos(self):

        head_id = torch.where(self.is_face == True)[0]
        body_id = torch.where(self.is_face == False)[0]

        is_lower_body = torch.where(self.is_lower_body == True)[0]
        is_upper_body = torch.where(self.is_upper_body == True)[0]
        is_rhand = torch.where(self.is_rhand == True)[0]
        is_lhand = torch.where(self.is_lhand == True)[0]

        is_hand = torch.cat([is_rhand, is_lhand])

        return dict(
            head=head_id,
            body=body_id,
            lower_body=is_lower_body,
            upper_body=is_upper_body,
            hands=is_hand,
        )

    def query_voxel_debug(self):

        skinning_weight = self.smplx_layer.lbs_weights.float()
        smplx_data = {"betas": torch.zeros(1, self.smpl_x.shape_param_dim)}
        device = skinning_weight.device

        _, mesh_neutral_pose_wo_upsample, _ = self.get_neutral_pose_human(
            jaw_zero_pose=True,
            use_id_info=True,
            shape_param=smplx_data["betas"],
            device=device,
            face_offset=smplx_data.get("face_offset", None),
            joint_offset=smplx_data.get("joint_offset", None),
        )

        template_verts = mesh_neutral_pose_wo_upsample

        query_skinning = (
            self.query_voxel_skinning_weights(template_verts).squeeze(0).detach().cpu()
        )
        skinning_weight = self.smplx_layer.lbs_weights.float()

        diff = torch.abs(query_skinning - skinning_weight)

        print(diff.sum())

    def query_voxel_skinning_weights(self, vs):
        """using voxel-based skinning method
        vs: [B n c]
        """
        voxel_bbox = self.voxel_bbox

        scale = voxel_bbox[..., 1] - voxel_bbox[..., 0]
        center = voxel_bbox.mean(dim=1)
        normalized_vs = (vs - center[None, None, :]) / scale[None, None]
        # mapping to [-1, 1] **3
        normalized_vs = normalized_vs * 2
        normalized_vs.to(self.voxel_ws)

        B, N, _ = normalized_vs.shape

        query_ws = F.grid_sample(
            self.voxel_ws.unsqueeze(0),  # 1 C D H W
            normalized_vs.reshape(1, 1, 1, -1, 3).to(self.voxel_ws),
            align_corners=True,
            padding_mode="border",
        )
        query_ws = query_ws.view(B, -1, N)
        query_ws = query_ws.permute(0, 2, 1)

        return query_ws  # [B N C]

    def get_zero_pose_human(
        self, shape_param, device, face_offset, joint_offset, return_mesh=False
    ):
        smpl_x = self.smpl_x
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_body_pose = (
            torch.zeros((batch_size, (len(smpl_x.joint_part["body"]) - 1) * 3))
            .float()
            .to(device)
        )
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

        face_offset = face_offset
        joint_offset = (
            smpl_x.get_joint_offset(joint_offset) if joint_offset is not None else None
        )
        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=zero_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=zero_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )
        joint_zero_pose = output.joints[:, : smpl_x.joint_num, :]  # zero pose human

        if not return_mesh:
            return joint_zero_pose
        else:
            raise NotImplementedError

    def get_transform_mat_joint(
        self, transform_mat_neutral_pose, joint_zero_pose, smplx_param
    ):
        """_summary_
        Args:
            transform_mat_neutral_pose (_type_): [B, 55, 4, 4]
            joint_zero_pose (_type_): [B, 55, 3]
            smplx_param (_type_): dict
        Returns:
            _type_: _description_
        """

        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose # None

        # 2. zero pose -> image pose
        root_pose = smplx_param["root_pose"]
        body_pose = smplx_param["body_pose"]
        jaw_pose = smplx_param["jaw_pose"]
        leye_pose = smplx_param["leye_pose"]
        reye_pose = smplx_param["reye_pose"]
        lhand_pose = smplx_param["lhand_pose"]
        rhand_pose = smplx_param["rhand_pose"]
        # trans = smplx_param['trans']

        # forward kinematics

        pose = torch.cat(
            (
                root_pose.unsqueeze(1),
                body_pose,
                jaw_pose.unsqueeze(1),
                leye_pose.unsqueeze(1),
                reye_pose.unsqueeze(1),
                lhand_pose,
                rhand_pose,
            ),
            dim=1,
        )  # [B, 55, 3]
        pose[0, :, :] = pose[0, :, :] + self.pose_mean_.reshape(-1, 3)
        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]
        # pose = self.batch_rodrigues_torch(pose)
        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )
        transform_mat_joint_2 = transform_mat_joint_2  # [B, 55, 4, 4]

        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )  # [B, 55, 4, 4]
        else:
            transform_mat_joint = transform_mat_joint_2 # this way !!!

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, query_points, fix_mask):
        # 这个函数：joint的变形矩阵A*lbs权重，得到最终每个vertex的变形矩阵
        batch_size = transform_mat_joint.shape[0]

        query_skinning = self.query_voxel_skinning_weights(query_points)   # 2,40000,55
        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)   # 2,40000,55
        query_skinning[fix_mask] = skinning_weight[fix_mask]

        transform_mat_vertex = torch.matmul(     # 2,40000,4,4
            skinning_weight,
            transform_mat_joint.view(batch_size, self.smpl_x.joint_num, 16),   # ori: 2,55,4,4
        ).view(batch_size, self.smpl_x.vertex_num_upsampled, 4, 4)
        return transform_mat_vertex

    def get_posed_blendshape(self, smplx_param):
        # posed_blendshape is only applied on hand and face, which parts are closed to smplx model
        root_pose = smplx_param["root_pose"]
        body_pose = smplx_param["body_pose"]
        jaw_pose = smplx_param["jaw_pose"]
        leye_pose = smplx_param["leye_pose"]
        reye_pose = smplx_param["reye_pose"]
        lhand_pose = smplx_param["lhand_pose"]
        rhand_pose = smplx_param["rhand_pose"]
        batch_size = root_pose.shape[0]

        pose = torch.cat(
            (
                body_pose,
                jaw_pose.unsqueeze(1),
                leye_pose.unsqueeze(1),
                reye_pose.unsqueeze(1),
                lhand_pose,
                rhand_pose,
            ),
            dim=1,
        )  # [B, 54, 3]
        pose += self.pose_mean_.reshape(-1, 3)[1:, :]
        # smplx pose-dependent vertex offset
        pose = (
            axis_angle_to_matrix(pose) - torch.eye(3)[None, None, :, :].float().cuda()
        ).view(batch_size, (self.smpl_x.joint_num - 1) * 9)
        # (B, 54 * 9) x (54*9, V)

        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(
            batch_size, self.smpl_x.vertex_num_upsampled, 3
        )
        return smplx_pose_offset

    def lbs(self, xyz, transform_mat_vertex, trans):
        batch_size = xyz.shape[0]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )  # 大 pose. xyz1 [B, N, 4]
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, self.vertex_num_upsampled, 4
        )[
            :, :, :3
        ]  # [B, N, 3]
        # trans = None
        if trans is not None:
            xyz = xyz + trans.unsqueeze(1)
        return xyz

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx

    def transform_to_posed_verts_from_neutral_pose(
        self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["betas"]
        # shape_param = torch.tensor([-0.1728,  1.6386, -0.3575,  1.4468,  1.1971,  1.3119,  0.1308, -0.2046, 0.6539, -0.2061]).unsqueeze(0).to('cuda')
        face_offset = smplx_data.get("face_offset", None)
        joint_offset = smplx_data.get("joint_offset", None)

        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )
            if face_offset is not None:
                face_offset = (
                    face_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *face_offset.shape[1:])
                )
            if joint_offset is not None:
                joint_offset = (
                    joint_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *joint_offset.shape[1:])
                )

        # smplx facial expression offset

        try:
            smplx_expr_offset = (
                smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
            ).sum(
                -1
            )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        except:
            print("no use flame params")
            smplx_expr_offset = 0.0

        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

        # get nearest vertex

        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        mask = (
            # ((self.is_rhand + self.is_lhand + self.is_face) > 0)
            ((self.is_rhand + self.is_lhand) > 0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # compute vertices-LBS function  这个函数：joint的变形矩阵A*lbs权重，得到最终每个vertex的变形矩阵
        transform_mat_null_vertex = self.get_transform_mat_vertex(   # 2,40000,4,4
            transform_mat_neutral_pose, mean_3d, mask
        ) # transformation matrix, 用于从 大-pose 到 zero-pose !!!!!

        null_mean_3d = self.lbs(
            mean_3d, transform_mat_null_vertex, torch.zeros_like(smplx_data["trans"])
        )  # posed with smplx_param

        # blend_shape offset
        blend_shape_offset = blend_shapes(shape_param, self.shape_dirs)
        null_mean3d_blendshape = null_mean_3d + blend_shape_offset
        # null_mean3d_blendshape = null_mean_3d

        # get transformation matrix of the nearest vertex and perform lbs
        joint_null_pose = self.get_zero_pose_human(   # 手是正常的，不是下垂的
            shape_param=shape_param,  # target shape
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )  # 2,55,3

        # mesh = trimesh.Trimesh(joint_null_pose[0, ...].squeeze().detach().cpu().numpy())
        # save_path = './data/joint_cano_new_高斯.obj'
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # mesh.export(save_path)
        # print(f"[INFO] Mesh saved to {save_path}")

        # NOTE that the question "joint_zero_pose" is different with (transform_mat_neutral_pose)'s joints.
        transform_mat_joint, j3d = self.get_transform_mat_joint(
            # transform_mat_neutral_pose, joint_null_pose, smplx_data
            None, joint_null_pose, smplx_data
        )

        pose_offsets = self.get_posed_blendshape(smplx_data)

        null_mean3d_blendshape = null_mean3d_blendshape + pose_offsets

        # compute vertices-LBS function
        transform_mat_vertex = self.get_transform_mat_vertex(
            # transform_mat_joint, null_mean3d_blendshape, mask   # DEBUG
            transform_mat_joint, mean_3d, mask
        )

        posed_mean_3d = self.lbs(
            null_mean3d_blendshape, transform_mat_vertex, smplx_data["trans"]
        )  # posed with smplx_param

        # mesh = trimesh.Trimesh(posed_mean_3d[0].squeeze().detach().cpu().numpy())
        # save_path = './data/尝试把xyz位移0.5.obj'
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # mesh.export(save_path)
        # print(f"[INFO] Mesh saved to {save_path}")

        # as we do not use transform port [...,:,3],so we simply compute chain matrix
        neutral_to_posed_vertex = torch.matmul(
            transform_mat_vertex, transform_mat_null_vertex
        )  # [B, N, 4, 4]

        return posed_mean_3d, neutral_to_posed_vertex

    def get_query_points(self, smplx_data, device):
        """transform_mat_neutral_pose is function to warp pre-defined posed to zero-pose"""

        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose_human(
                jaw_zero_pose=True,
                use_id_info=False,  # we blendshape at zero-pose
                shape_param=smplx_data["betas"],
                device=device,
                face_offset=smplx_data.get("face_offset", None),
                joint_offset=smplx_data.get("joint_offset", None),
            )
        )

        return (
            mesh_neutral_pose,
            mesh_neutral_pose_wo_upsample,
            transform_mat_neutral_pose,
        )

    def transform_to_posed_verts(self, smplx_data, device):
        """_summary_
        Args:
            smplx_data (_type_): e.g., body_pose:[B*Nv, 21, 3], betas:[B*Nv, 100]
        """

        # neutral posed verts
        mesh_neutral_pose, _, transform_mat_neutral_pose = self.get_query_points(
            smplx_data, device
        )

        # print(mesh_neutral_pose.shape, transform_mat_neutral_pose.shape, mesh_neutral_pose.shape, smplx_data["body_pose"].shape)
        mean_3d, transform_matrix = self.transform_to_posed_verts_from_neutral_pose(
            mesh_neutral_pose,
            smplx_data,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
            device,
        )

        return mean_3d, transform_matrix

    def upsample_mesh_batch(
        self,
        smpl_x,
        shape_param,
        neutral_body_pose,
        jaw_pose,
        expression,
        betas,
        face_offset=None,
        joint_offset=None,
        device=None,
    ):
        """using blendshape to offset pts"""

        device = device if device is not None else avaliable_device()

        batch_size = shape_param.shape[0]
        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )

        dense_pts = self.dense_pts.to(device)
        dense_pts = dense_pts.unsqueeze(0).repeat(expression.shape[0], 1, 1)

        blend_shape_offset = blend_shapes(betas, self.shape_dirs)

        dense_pts = dense_pts + blend_shape_offset

        joint_zero_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        neutral_pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose,
                jaw_pose,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )  # [B, 55, 3]

        neutral_pose = axis_angle_to_matrix(
            neutral_pose.view(-1, 55, 3)
        )  # [B, 55, 3, 3]
        posed_joints, transform_mat_joint = batch_rigid_transform(
            neutral_pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )

        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # B 55 4,4, B N 55 -> B N 4 4
        transform_mat_vertex = torch.einsum(
            "blij,bnl->bnij", transform_mat_joint, skinning_weight
        )
        mesh_neutral_pose_upsampled = self.lbs(dense_pts, transform_mat_vertex, None)

        return mesh_neutral_pose_upsampled

    def transform_to_neutral_pose(
        self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["betas"]
        face_offset = smplx_data.get("face_offset", None)
        joint_offset = smplx_data.get("joint_offset", None)
        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )
            if face_offset is not None:
                face_offset = (
                    face_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *face_offset.shape[1:])
                )
            if joint_offset is not None:
                joint_offset = (
                    joint_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *joint_offset.shape[1:])
                )

        # smplx facial expression offset
        smplx_expr_offset = (
            smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
        ).sum(
            -1
        )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

    def get_neutral_pose_human(
        self, jaw_zero_pose, use_id_info, shape_param, device, face_offset, joint_offset
    ):

        smpl_x = self.smpl_x
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        neutral_body_pose = (
            smpl_x.neutral_body_pose.view(1, -1).repeat(batch_size, 1).to(device)
        )  # 大 pose
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

        if jaw_zero_pose:
            jaw_pose = torch.zeros((batch_size, 3)).float().to(device)
        else:
            jaw_pose = (
                smpl_x.neutral_jaw_pose.view(1, 3).repeat(batch_size, 1).to(device)
            )  # open mouth

        if use_id_info:
            shape_param = shape_param
            # face_offset = smpl_x.face_offset[None,:,:].float().to(device)
            # joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
            face_offset = face_offset
            joint_offset = (
                smpl_x.get_joint_offset(joint_offset)
                if joint_offset is not None
                else None
            )

        else:
            shape_param = (
                torch.zeros((batch_size, smpl_x.shape_param_dim)).float().to(device)
            )
            face_offset = None
            joint_offset = None

        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=neutral_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        # using dense sample strategy, and warp to neutral pose
        mesh_neutral_pose_upsampled = self.upsample_mesh_batch(
            smpl_x,
            shape_param=shape_param,
            neutral_body_pose=neutral_body_pose,
            jaw_pose=jaw_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
            device=device,
        )

        mesh_neutral_pose = output.vertices
        joint_neutral_pose = output.joints[
            :, : smpl_x.joint_num, :
        ]  # 大 pose human  [B, 55, 3]

        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(
            batch_size, len(smpl_x.joint_part["body"]) - 1, 3
        )
        zero_hand_pose = zero_hand_pose.view(
            batch_size, len(smpl_x.joint_part["lhand"]), 3
        )

        neutral_body_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(neutral_body_pose))
        )
        jaw_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(jaw_pose))
        )

        zero_pose = zero_pose.unsqueeze(1)
        jaw_pose_inv = jaw_pose_inv.unsqueeze(1)

        pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose_inv,
                jaw_pose_inv,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )

        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]

        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.smplx_layer.parents
        )  # [B, 55, 4, 4]

        return (
            mesh_neutral_pose_upsampled,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
        )

    def batch_rodrigues_torch(self, poses):
        """
        poses: B x N x 3 (axis-angle representation)
        return: B x N x 3 x 3 rotation matrices
        """
        B, N, _ = poses.shape
        poses = poses.view(-1, 3)  # (B*N, 3)

        angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)  # (B*N, 1)
        rot_dir = poses / angle  # (B*N, 3)

        cos = torch.cos(angle)[:, None]  # (B*N, 1, 1)
        sin = torch.sin(angle)[:, None]  # (B*N, 1, 1)

        rx, ry, rz = rot_dir[:, 0:1], rot_dir[:, 1:2], rot_dir[:, 2:3]
        zeros = torch.zeros_like(rx)

        K = torch.cat([
            zeros, -rz, ry,
            rz, zeros, -rx,
            -ry, rx, zeros
        ], dim=1).reshape(-1, 3, 3)  # (B*N, 3, 3)

        ident = torch.eye(3, device=poses.device).unsqueeze(0)  # (1, 3, 3)
        rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)  # (B*N, 3, 3)

        return rot_mat.view(B, N, 3, 3)


class MANOVoxelMeshModel(nn.Module):
    def __init__(self, hanco=False, center_add=True) -> None:
        super().__init__()

        # if hanco is True and cfg.smpl_cfg['flat_hand_mean'] == True:
        #     cfg.smpl_cfg['flat_hand_mean'] = False
        #     # cfg.smpl_cfg['scale'] = None
        #     # cfg.smpl_cfg['center_id'] = 4 
        
        # cfg.smpl_cfg['flat_hand_mean'] = False
        # cfg.smpl_cfg['scale'] = None
        # cfg.smpl_cfg['center_idx'] = None   

        self.mano = tools.model.smplx.create(**cfg.smpl_cfg)
        self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights
        self.pose_mean = self.mano.pose_mean.to('cuda')
        
        # control whether to subtract the joint center from posed vertices
        # interhand and hanco expect subtraction; hands11k should disable it
        self.center_add = center_add


        # register
        self.apply_pose_blendshape = True
        self.dense_sample()
        self.manohd_init()

        # get transformation matrix of the nearest vertex and perform lbs
        self.joint_null_pose = self.get_zero_pose_hand(  # 手是正常的，不是下垂的
            shape_param=torch.zeros(1,10).float()).to('cuda')  # 2,55,3
        # self.joint_null_pose = torch.zeros(1, 16, 3).float().to('cuda')  # 2,55,3
        self.center_id = self.mano.center_id
        self.scale = self.mano.scale
        self.mano.lbs_weights.to('cuda')
        # self.center_id, self.scale = self.center_id.to('cuda'), self.scale.to('cuda')

    def rebuild_mesh(self, v, vertices_id, faces_id, num_dense_samples):
        choice_vertices = v[vertices_id]

        new_mapping = dict()

        for new_id, vertice_id in enumerate(vertices_id):
            new_mapping[vertice_id] = new_id

        faces_id_list = faces_id.reshape(-1).tolist()

        new_faces_id = []
        for face_id in faces_id_list:
            new_faces_id.append(new_mapping[face_id])
        new_faces_id = torch.from_numpy(np.array(new_faces_id).reshape(faces_id.shape))

        mymesh = Mesh(v=choice_vertices, f=new_faces_id)

        dense_sample_pts = mymesh.sample_surface(num_dense_samples).detach().cpu()

        return dense_sample_pts

    def dense_sample(self):

        debug_hand = f"./pretrained_models/dense_sample_points/manohd_semantic.ply"  # 1_40000.ply

        manohd_pts, _ = load_ply(debug_hand)
        self.hand_pts = manohd_pts / 10.0

    @torch.no_grad()
    def voxel_smooth_register(
            self, voxel_v, template_v, lbs_weights, k=3, smooth_k=30, smooth_n=3000
    ):
        """Smooth KNN to handle skirt deformation."""

        lbs_weights = lbs_weights.cuda()

        dist = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            template_v.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        )
        mesh_dis = torch.sqrt(dist.dists)
        mesh_indices = dist.idx.squeeze(0, -1)
        knn_lbs_weights = lbs_weights[mesh_indices]

        mesh_dis = mesh_dis.squeeze()

        print(f"Using k = {smooth_k}, N={smooth_n} for LBS smoothing")
        # Smooth Skinning

        knn_dis = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            voxel_v.unsqueeze(0).cuda(),
            K=smooth_k + 1,
            return_nn=True,
        )
        voxel_dis = torch.sqrt(knn_dis.dists)
        voxel_indices = knn_dis.idx
        voxel_indices = voxel_indices.squeeze()[:, 1:]
        voxel_dis = voxel_dis.squeeze()[:, 1:]

        knn_weights = 1.0 / (mesh_dis[voxel_indices] * voxel_dis)
        knn_weights = knn_weights / knn_weights.sum(-1, keepdim=True)  # [N, K]

        def dists_to_weights(
                dists: torch.Tensor, low: float = None, high: float = None
        ):
            if low is None:
                low = high
            if high is None:
                high = low
            assert high >= low
            weights = dists.clone()
            weights[dists <= low] = 0.0
            weights[dists >= high] = 1.0
            indices = (dists > low) & (dists < high)
            weights[indices] = (dists[indices] - low) / (high - low)
            return weights

        update_weights = dists_to_weights(mesh_dis, low=0.01).unsqueeze(-1)  # [N, 1]

        from tqdm import tqdm

        for _ in tqdm(range(smooth_n)):
            N, _ = update_weights.shape
            new_lbs_weights_chunk_list = []
            for chunk_i in range(0, N, 1000000):
                knn_weights_chunk = knn_weights[chunk_i: chunk_i + 1000000]
                voxel_indices_chunk = voxel_indices[chunk_i: chunk_i + 1000000]

                new_lbs_weights_chunk = torch.einsum(
                    "nk,nkj->nj",
                    knn_weights_chunk,
                    knn_lbs_weights[voxel_indices_chunk],
                )
                new_lbs_weights_chunk_list.append(new_lbs_weights_chunk)
            new_lbs_weights = torch.cat(new_lbs_weights_chunk_list, dim=0)
            if update_weights is None:
                knn_lbs_weights = new_lbs_weights
            else:
                knn_lbs_weights = (
                                          1.0 - update_weights
                                  ) * knn_lbs_weights + update_weights * new_lbs_weights

        return knn_lbs_weights

    def voxel_skinning_init(self, scale_ratio=1.05, voxel_size=256):

        skinning_weight = self.smplx_layer.lbs_weights.float()  # 10475,55

        smplx_data = {"betas": torch.zeros(1, self.smpl_x.shape_param_dim)}
        device = skinning_weight.device

        _, mesh_neutral_pose_wo_upsample, _ = self.get_neutral_pose_human(  # 1,10475,3
            jaw_zero_pose=True,
            use_id_info=True,
            shape_param=smplx_data["betas"],
            device=device,
            face_offset=smplx_data.get("face_offset", None),
            joint_offset=smplx_data.get("joint_offset", None),
        )

        template_verts = mesh_neutral_pose_wo_upsample.squeeze(0)

        def scale_voxel_size(template_verts, scale_ratio=1.0):
            min_values, _ = torch.min(template_verts, dim=0)
            max_values, _ = torch.max(template_verts, dim=0)

            center = (min_values + max_values) / 2
            size = max_values - min_values

            scale_size = size * scale_ratio

            upper = center + scale_size / 2
            bottom = center - scale_size / 2

            return torch.cat([bottom[:, None], upper[:, None]], dim=1)

        mini_size_bbox = scale_voxel_size(template_verts, scale_ratio)
        z_voxel_size = voxel_size // 2

        # build coordinate
        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
                voxel_size - 1
        )  # from 0 to 255，
        y_range = np.linspace(0, voxel_size - 1, voxel_size) / (voxel_size - 1)
        z_range = np.linspace(0, z_voxel_size - 1, z_voxel_size) / (z_voxel_size - 1)

        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        coordinates = torch.from_numpy(np.stack([x, y, z], axis=-1))

        coordinates[..., 0] = mini_size_bbox[0, 0] + coordinates[..., 0] * (
                mini_size_bbox[0, 1] - mini_size_bbox[0, 0]
        )
        coordinates[..., 1] = mini_size_bbox[1, 0] + coordinates[..., 1] * (
                mini_size_bbox[1, 1] - mini_size_bbox[1, 0]
        )
        coordinates[..., 2] = mini_size_bbox[2, 0] + coordinates[..., 2] * (
                mini_size_bbox[2, 1] - mini_size_bbox[2, 0]
        )

        coordinates = coordinates.view(-1, 3).float()
        coordinates = coordinates.cuda()

        if os.path.exists(f"./pretrained_models/voxel_grid/voxel_{voxel_size}.pth"):
            print(f"load voxel_grid voxel_{voxel_size}.pth")
            voxel_flat = torch.load(
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
                map_location=avaliable_device(),
            )
        else:
            voxel_flat = self.voxel_smooth_register(
                coordinates, template_verts, skinning_weight, k=1, smooth_n=3000
            )

            torch.save(
                voxel_flat,
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
            )

        N, LBS_F = voxel_flat.shape

        # x, y, z, C
        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

        # [W H D 55]->[55, D, H, W]
        voxel_grid = voxel_grid_original.permute(3, 2, 1, 0)

        return voxel_grid, mini_size_bbox


    def manohd_init(self):
        """
        Initialize the sub-devided smplx model by registering buffers for various attributes
        This method performs the following steps:
        1. Upsamples the mesh and other assets.
        2. Computes skinning weights, pose directions, expression directions, and various flags for different body parts.
        3. Reshapes and permutes the pose and expression directions.
        4. Converts the flags to boolean values.
        5. Registers buffers for the computed attributes.
        Args:
            self: The object instance.
        Returns:
            None
        """

        # mano = self.mano
        # lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        # self.mano.lbs_weights = lbs_weights
        # self.mano.to('cuda')

        # using KNN to query subdivided mesh
        dense_pts = self.hand_pts.cuda()
        template_verts = self.mano.v_template

        nn_vertex_idxs = knn_points(  # 查找每个gaussian点最近的mesh顶点的索引
            dense_pts.unsqueeze(0).cuda(),
            template_verts.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        ).idx
        # query_indx = nn_vertex_idxs.squeeze(0, -1).detach().cpu()

        # skinning_weight = self.smplx_layer.lbs_weights.float()

        """ PCA regression function w.r.t vertices offset
        """
        # pose_dirs = self.mano.posedirs.permute(1, 0).reshape(12337, 3 * (mano.NUM_JOINTS - 1) * 9)
        # shape_dirs = self.mano.shapedirs.view(12337, 30)

        self.mano.vertex_num_upsampled = self.hand_pts  # compatible with SMPLX

        voxel_skinning_weight, voxel_bbox = self.hand_voxel_skinning_init(voxel_size=192)
        # voxel_skinning_weight: 55,96,192,192      voxel_bbox: 3,2
        self.register_buffer("hand_voxel_ws", voxel_skinning_weight)
        self.register_buffer("hand_voxel_bbox", voxel_bbox)

        # self.query_voxel_debug()

    def hand_voxel_skinning_init(self, scale_ratio=1.05, voxel_size=256):
        skinning_weight = self.mano.lbs_weights.float()
        template_verts = self.hand_pts

        def scale_voxel_size(template_verts, scale_ratio=1.0):
            min_values, _ = torch.min(template_verts, dim=0)
            max_values, _ = torch.max(template_verts, dim=0)

            center = (min_values + max_values) / 2
            size = max_values - min_values

            scale_size = size * scale_ratio

            upper = center + scale_size / 2
            bottom = center - scale_size / 2

            return torch.cat([bottom[:, None], upper[:, None]], dim=1)

        mini_size_bbox = scale_voxel_size(template_verts, scale_ratio)
        z_voxel_size = voxel_size // 2

        # build coordinate
        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
                voxel_size - 1
        )  # from 0 to 255，
        y_range = np.linspace(0, voxel_size - 1, voxel_size) / (voxel_size - 1)
        z_range = np.linspace(0, z_voxel_size - 1, z_voxel_size) / (z_voxel_size - 1)

        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        coordinates = torch.from_numpy(np.stack([x, y, z], axis=-1))

        coordinates[..., 0] = mini_size_bbox[0, 0] + coordinates[..., 0] * (
                mini_size_bbox[0, 1] - mini_size_bbox[0, 0]
        )
        coordinates[..., 1] = mini_size_bbox[1, 0] + coordinates[..., 1] * (
                mini_size_bbox[1, 1] - mini_size_bbox[1, 0]
        )
        coordinates[..., 2] = mini_size_bbox[2, 0] + coordinates[..., 2] * (
                mini_size_bbox[2, 1] - mini_size_bbox[2, 0]
        )

        coordinates = coordinates.view(-1, 3).float()
        coordinates = coordinates.cuda()

        if os.path.exists(f"./pretrained_models/voxel_grid/hand_voxel_{voxel_size}.pth"):
            print(f"load voxel_grid voxel_{voxel_size}.pth")
            voxel_flat = torch.load(
                os.path.join(f"pretrained_models/voxel_grid/hand_voxel_{voxel_size}.pth"),
                map_location=avaliable_device(),
            )
        else:
            voxel_flat = self.voxel_smooth_register(
                coordinates, template_verts, skinning_weight, k=1, smooth_n=3000
            )

            torch.save(
                voxel_flat,
                os.path.join(f"pretrained_models/voxel_grid/hand_voxel_{voxel_size}.pth"),
            )

        N, LBS_F = voxel_flat.shape

        # x, y, z, C
        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

        # [W H D 55]->[55, D, H, W]
        voxel_grid = voxel_grid_original.permute(3, 2, 1, 0)

        return voxel_grid, mini_size_bbox


    def query_voxel_skinning_weights(self, vs):
        """using voxel-based skinning method
        vs: [B n c]
        """
        voxel_bbox = self.voxel_bbox

        scale = voxel_bbox[..., 1] - voxel_bbox[..., 0]
        center = voxel_bbox.mean(dim=1)
        normalized_vs = (vs - center[None, None, :]) / scale[None, None]
        # mapping to [-1, 1] **3
        normalized_vs = normalized_vs * 2
        normalized_vs.to(self.voxel_ws)

        B, N, _ = normalized_vs.shape

        query_ws = F.grid_sample(
            self.voxel_ws.unsqueeze(0),  # 1 C D H W
            normalized_vs.reshape(1, 1, 1, -1, 3).to(self.voxel_ws),
            align_corners=True,
            padding_mode="border",
        )
        query_ws = query_ws.view(B, -1, N)
        query_ws = query_ws.permute(0, 2, 1)

        return query_ws  # [B N C]

    def get_zero_pose_hand(self, shape_param, return_mesh=False):

        output = self.mano(shape_param,
                           torch.zeros(1,3).float(),
                           torch.zeros(1,45).float(), return_verts=True)
        self.center = output.center.squeeze().to('cuda')
        # joints = torch.matmul(self.mano.J_regressor[None].cuda(), output.vertices.cuda())

        joint_zero_pose = output.joints[:, : self.mano.NUM_JOINTS, :]  # zero pose human

        return joint_zero_pose


    def get_transform_mat_joint(
            self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        """_summary_
        Args:
            transform_mat_neutral_pose (_type_): [B, 55, 4, 4]
            joint_zero_pose (_type_): [B, 55, 3]
            smplx_param (_type_): dict
        Returns:
            _type_: _description_
        """

        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose  # None

        # 2. zero pose -> image pose
        poses_tensor = smplx_param["poses"]
        if poses_tensor.device.type == 'cpu':
            poses_tensor = poses_tensor.to('cuda')
        root_pose = poses_tensor[0, :3]
        hand_pose = poses_tensor[0, 3:]
        # trans = smplx_param['trans']

        # forward kinematics

        pose = torch.cat(
            (
                root_pose.unsqueeze(0),
                hand_pose.unsqueeze(0),
            ),
            dim=1,
        ).reshape(-1, 3).unsqueeze(0)  # [B, 55, 3]
        
        pose = pose + self.pose_mean.reshape(-1, 3)
        
        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]
        # pose = self.batch_rodrigues_torch(pose)
        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.mano.parents.to('cuda')
        )
        # posed_joints, transform_mat_joint_2 = get_rigid_transformation_torch(pose[:, :, :, :], joint_zero_pose[:, :, :], self.mano.parents.to('cuda'))
        transform_mat_joint_2 = transform_mat_joint_2  # [B, 55, 4, 4]

        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )  # [B, 55, 4, 4]
        else:
            transform_mat_joint = transform_mat_joint_2  # this way !!!

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, query_points):
        # 这个函数：joint的变形矩阵A*lbs权重，得到最终每个vertex的变形矩阵
        batch_size = transform_mat_joint.shape[0]

        # query_skinning = self.query_voxel_skinning_weights(query_points)  # 2,40000,55
        skinning_weight = self.mano.lbs_weights.repeat(batch_size, 1, 1).to('cuda')  # 2,40000,55
        # query_skinning[fix_mask] = skinning_weight[fix_mask]
        skinning_weight = torch.log(skinning_weight + 1e-9) + skinning_weight
        skinning_weight = F.softmax(skinning_weight, dim=-1)

        transform_mat_vertex = torch.matmul(  # 2,40000,4,4
            skinning_weight,
            transform_mat_joint.view(batch_size, self.mano.NUM_JOINTS, 16),  # ori: 2,55,4,4
        ).view(batch_size, self.mano.vertex_num_upsampled.size(0), 4, 4)
        return transform_mat_vertex

    def get_posed_blendshape(self, smplx_param):
        # posed_blendshape is only applied on hand and face, which parts are closed to smplx model
        poses_tensor = smplx_param["poses"]
        if poses_tensor.device.type == 'cpu':
            poses_tensor = poses_tensor.to('cuda')
        pose_full = poses_tensor[0,:].unsqueeze(0)
        # root_pose = pose_full[:,:3]
        hand_pose = pose_full[:, 3:]
        batch_size = 1#root_pose.shape[0]
        pose = hand_pose.reshape(batch_size, -1, 3)

        pose = pose + self.pose_mean.reshape(-1, 3)[1:, :]
        # smplx pose-dependent vertex offset
        pose = (
                axis_angle_to_matrix(pose) - torch.eye(3)[None, None, :, :].float().cuda()
        ).view(batch_size, (self.mano.NUM_JOINTS - 1) * 9)
        # (B, 54 * 9) x (54*9, V)

        smplx_pose_offset = torch.matmul(pose.unsqueeze(1), self.mano.posedirs.to('cuda').view(
            self.mano.vertex_num_upsampled.size(0)*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, self.mano.vertex_num_upsampled.size(0), 3)
        # smplx_pose_offset = torch.matmul(pose, self.mano.posedirs.to('cuda')).view(
        #     batch_size, self.mano.vertex_num_upsampled, 3)
        
        return smplx_pose_offset

    def lbs(self, xyz, transform_mat_vertex, trans):
        batch_size = xyz.shape[0]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )  # 大 pose. xyz1 [B, N, 4]
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, self.mano.vertex_num_upsampled.size(0), 4
        )[:, :, :3]  # [B, N, 3]
        # trans = None
        if trans is not None:
            xyz = xyz + trans.unsqueeze(1)
        return xyz

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx

    def transform_to_posed_verts_from_neutral_pose(
            self, mean_3d, smplx_data, # mesh_neutral_pose, #transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["shape"]#.unsqueeze(0)
        # Ensure shape_param is on the correct device
        if shape_param.device.type == 'cpu':
            shape_param = shape_param.to('cuda')
        merge_mean_3d = mean_3d#[0, ...]
        mean_3d = mean_3d[0, ...]

        # blend_shape offset
        # mean_3d = mean_3d / self.scale
        # mean_3d = mean_3d + self.center
        blend_shape_offset = blend_shapes(shape_param, self.mano.shapedirs.to('cuda'))
        null_mean3d_blendshape = mean_3d# + blend_shape_offset
        merge_mean_3d[0, ...] = null_mean3d_blendshape
        # null_mean3d_blendshape = null_mean_3d

        # # mesh = trimesh.Trimesh(self.joint_null_pose[0, ...].squeeze().detach().cpu().numpy())
        # mesh = trimesh.Trimesh(mean_3d.squeeze().detach().cpu().numpy())
        # save_path = './data/hand/00-hand_高斯.obj'
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # mesh.export(save_path)
        # print(f"[INFO] Mesh saved to {save_path}")

        # NOTE that the question "joint_zero_pose" is different with (transform_mat_neutral_pose)'s joints.

        # A, joints, posed_joints = self.get_transform_params_torch(smplx_data, rot_mats=None)
        transform_mat_joint, j3d = self.get_transform_mat_joint(
            # transform_mat_neutral_pose, joint_null_pose, smplx_data
            None, self.joint_null_pose, smplx_data)    # 1,16,4,4   transform_mat_joint

        pose_offsets = self.get_posed_blendshape(smplx_data)

        null_mean3d_blendshape = null_mean3d_blendshape + pose_offsets

        # compute vertices-LBS function
        transform_mat_vertex = self.get_transform_mat_vertex(
            # transform_mat_joint, null_mean3d_blendshape, mask   # DEBUG
            transform_mat_joint, mean_3d # , mask
        )
        # transform_mat_vertex = transform_mat_vertex / self.scale

        posed_mean_3d = self.lbs(
            null_mean3d_blendshape, transform_mat_vertex, None, # smplx_data['trans'] #None
            # null_mean3d_blendshape, transform_mat_vertex, smplx_data['trans'] #None
        )  # posed with smplx_param
        
        # posed_mean_3d = self.lbs(
        #     null_mean3d_blendshape, transform_mat_vertex, smplx_data['trans'], #None
        # )  # posed with smplx_param

        center = j3d[:, self.center_id:self.center_id+1].clone()

        # Subtract center if configured. For Hands11k we disable this to keep
        # the original pose location (it would otherwise render incorrectly).
        if getattr(self, 'center_add', True):
            posed_mean_3d = posed_mean_3d - center

        # import trimesh
        # mesh2 = trimesh.Trimesh(posed_mean_3d[0].squeeze().detach().cpu().numpy(), self.mano.faces_tensor)
        # mesh2.export('./hand/interhand-w-center.obj')
        # exit()
        # posed_joints = j3d - center
        # posed_mean_3d = posed_mean_3d / self.scale
        # posed_joints = posed_joints * self.scale

        # transforms = transforms * scale

        # mesh = trimesh.Trimesh(posed_mean_3d[0].squeeze().detach().cpu().numpy())
        # save_path = './data/hand/00-hand_posed.obj'
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # mesh.export(save_path)
        # print(f"[INFO] Mesh saved to {save_path}")

        # # as we do not use transform port [...,:,3],so we simply compute chain matrix
        # neutral_to_posed_vertex = torch.matmul(
        #     transform_mat_vertex, transform_mat_null_vertex
        # )  # [B, N, 4, 4]
        #
        # return posed_mean_3d, neutral_to_posed_vertex
        merge_mean_3d[1, ...] = posed_mean_3d
        return merge_mean_3d, transform_mat_vertex


    def get_transform_params_torch(self, params, rot_mats=None, correct_Rs=None):
        """ obtain the transformation parameters for linear blend skinning
        """
        v_template = self.mano.v_template.cuda()  # mano-hd
        #
        # # add shape blend shapes
        shapedirs = self.mano.shapedirs.cuda()  # mano-hd

        # v_template = smpl['v_template']

        # add shape blend shapes
        # shapedirs = smpl['shapedirs']

        betas = params['shape'].unsqueeze(0)
        # v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()
        v_shaped = v_template[None] + torch.sum(shapedirs[None][..., :betas.shape[-1]] * betas[:, None],
                                                axis=-1).float()

        # obtain the joints
        joints = torch.matmul(self.mano.J_regressor[None].cuda(),
                              v_shaped)  # [bs, 24 ,3] # SMPL文章公式 10，得到的是posed关节的位置  # mano-hd
        # joints = torch.matmul(smpl['J_regressor'][None], v_shaped)  # [bs, 24 ,3]

        if rot_mats is None:
            # add pose blend shapes
            poses = params['poses'][0, :].reshape(-1, 3)
            # bs x 24 x 3 x 3
            # rot_mats = self.batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)
            rot_mats = self.batch_rodrigues_torch_new(poses).view(-1, poses.shape[0], 3, 3)
        
        # obtain the rigid transformation
        parents = self.mano.parents.cuda()  # mano-hd
        # parents = smpl['kintree_table'][0]
        A, posed_joints = self.get_rigid_transformation_torch(rot_mats, joints, parents)

        # # apply global transformation
        # R = params['R']
        # Th = params['Th']

        return A, joints, posed_joints

    def batch_rodrigues_torch_new(self, poses):
        """ poses: N x 3
        """
        batch_size = poses.shape[0]  # θ，72个参数。
        angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)  # norm
        rot_dir = poses / angle  # SMPL第 4页：表示 unit norm axis of rotation

        cos = torch.cos(angle)[:, None]
        sin = torch.sin(angle)[:, None]

        rx, ry, rz = torch.split(rot_dir, 1, dim=1)  # 结合文章第4页，这里说的是从 rot_dir中提取 斜对称矩阵
        zeros = torch.zeros((batch_size, 1), device=poses.device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
        K = K.reshape([batch_size, 3, 3])

        ident = torch.eye(3)[None].to(poses.device)
        rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)  # SMPL 公式 1

        return rot_mat  # 返回的就是公式 1，后面会用来计算每个关节的 world transformation
    

    def get_rigid_transformation_torch(self, rot_mats, joints, parents):  # 获得 G
        """
        rot_mats: bs x 24 x 3 x 3
        joints: bs x 24 x 3
        parents: 24
        """
        # obtain the relative joints
        bs, joints_num = joints.shape[0:2]
        rel_joints = joints.clone()  # 对应公式 4中右上角，表示每个关节的位置
        rel_joints[:, 1:] -= joints[:, parents[1:]]  # joints[:, parents[1:]] 后半部分 获取所有非根关节的父关节位置
        # 整体：将每个非根关节的位置减去其父关节的位置，得到相对位置

        # create the transformation matrix
        transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)  # 将旋转矩阵和相对关节位置组合成变换矩阵（4x4）
        padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)  # .to(rot_mats.device) # 构造指定大小的 0矩阵
        padding[..., 3] = 1  # 这两行：添加第四行 [0, 0, 0, 1] 以完成齐次变换矩阵
        transforms_mat = torch.cat([transforms_mat, padding], dim=-2)  # 把(这个01填充的矩阵)和(计算的公式 4中的第一行)concatenate起来
        # 得到最终想要的公式 4

        # rotate each part
        transform_chain = [transforms_mat[:, 0]]  # 是根关节的变换矩阵。由于根关节没有父关节，所以直接使用其变换矩阵作为链的起点
        for i in range(1, parents.shape[0]):  # 对于每个非根关节
            curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:,
                                                                 i])  # 使用其父关节的全局变换矩阵（在 transform_chain 中）与当前关节的局部变换矩阵相乘，得到当前关节的全局变换矩阵
            transform_chain.append(curr_res)
        transforms = torch.stack(transform_chain, dim=1)

        posed_joints = transforms[:, :, :3, 3]

        # obtain the rigid transformation
        padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)  # .to(rot_mats.device)
        joints_homogen = torch.cat([joints, padding], dim=-1)
        rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
        transforms[..., 3] = transforms[..., 3] - rel_joints

        return transforms, posed_joints
    
    

    def get_query_points(self, smplx_data, device):
        """transform_mat_neutral_pose is function to warp pre-defined posed to zero-pose"""

        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose_human(
                jaw_zero_pose=True,
                use_id_info=False,  # we blendshape at zero-pose
                shape_param=smplx_data["shape"],
                device=device,
                # face_offset=smplx_data.get("face_offset", None),
                # joint_offset=smplx_data.get("joint_offset", None),
            )
        )

        return (
            mesh_neutral_pose,
            mesh_neutral_pose_wo_upsample,
            transform_mat_neutral_pose,
        )

    def transform_to_posed_verts(self, smplx_data, device):
        """_summary_
        Args:
            smplx_data (_type_): e.g., body_pose:[B*Nv, 21, 3], betas:[B*Nv, 100]
        """

        # neutral posed verts
        mesh_neutral_pose, _, transform_mat_neutral_pose = self.get_query_points(
            smplx_data, device
        )

        # print(mesh_neutral_pose.shape, transform_mat_neutral_pose.shape, mesh_neutral_pose.shape, smplx_data["body_pose"].shape)
        mean_3d, transform_matrix = self.transform_to_posed_verts_from_neutral_pose(
            mesh_neutral_pose,
            smplx_data,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
            device,
        )

        return mean_3d, transform_matrix

    def upsample_mesh_batch(
            self,
            smpl_x,
            shape_param,
            neutral_body_pose,
            jaw_pose,
            expression,
            betas,
            face_offset=None,
            joint_offset=None,
            device=None,
    ):
        """using blendshape to offset pts"""

        device = device if device is not None else avaliable_device()

        batch_size = shape_param.shape[0]
        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )

        dense_pts = self.dense_pts.to(device)
        dense_pts = dense_pts.unsqueeze(0).repeat(expression.shape[0], 1, 1)

        blend_shape_offset = blend_shapes(betas, self.shape_dirs)

        dense_pts = dense_pts + blend_shape_offset

        joint_zero_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        neutral_pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose,
                jaw_pose,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )  # [B, 55, 3]

        neutral_pose = axis_angle_to_matrix(
            neutral_pose.view(-1, 55, 3)
        )  # [B, 55, 3, 3]
        posed_jints, transform_mat_joint = batch_rigid_transform(
            neutral_pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )

        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # B 55 4,4, B N 55 -> B N 4 4
        transform_mat_vertex = torch.einsum(
            "blij,bnl->bnij", transform_mat_joint, skinning_weight
        )
        mesh_neutral_pose_upsampled = self.lbs(dense_pts, transform_mat_vertex, None)

        return mesh_neutral_pose_upsampled

    def transform_to_neutral_pose(
            self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["betas"]
        face_offset = smplx_data.get("face_offset", None)
        joint_offset = smplx_data.get("joint_offset", None)
        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )
            if face_offset is not None:
                face_offset = (
                    face_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *face_offset.shape[1:])
                )
            if joint_offset is not None:
                joint_offset = (
                    joint_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *joint_offset.shape[1:])
                )

        # smplx facial expression offset
        smplx_expr_offset = (
                smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
        ).sum(
            -1
        )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

    def get_neutral_pose_human(
            self, jaw_zero_pose, use_id_info, shape_param, device, face_offset, joint_offset
    ):

        smpl_x = self.smpl_x
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        neutral_body_pose = (
            smpl_x.neutral_body_pose.view(1, -1).repeat(batch_size, 1).to(device)
        )  # 大 pose
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

        if jaw_zero_pose:
            jaw_pose = torch.zeros((batch_size, 3)).float().to(device)
        else:
            jaw_pose = (
                smpl_x.neutral_jaw_pose.view(1, 3).repeat(batch_size, 1).to(device)
            )  # open mouth

        if use_id_info:
            shape_param = shape_param
            # face_offset = smpl_x.face_offset[None,:,:].float().to(device)
            # joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
            face_offset = face_offset
            joint_offset = (
                smpl_x.get_joint_offset(joint_offset)
                if joint_offset is not None
                else None
            )

        else:
            shape_param = (
                torch.zeros((batch_size, smpl_x.shape_param_dim)).float().to(device)
            )
            face_offset = None
            joint_offset = None

        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=neutral_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        # using dense sample strategy, and warp to neutral pose
        mesh_neutral_pose_upsampled = self.upsample_mesh_batch(
            smpl_x,
            shape_param=shape_param,
            neutral_body_pose=neutral_body_pose,
            jaw_pose=jaw_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
            device=device,
        )

        mesh_neutral_pose = output.vertices
        joint_neutral_pose = output.joints[
                             :, : smpl_x.joint_num, :
                             ]  # 大 pose human  [B, 55, 3]

        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(
            batch_size, len(smpl_x.joint_part["body"]) - 1, 3
        )
        zero_hand_pose = zero_hand_pose.view(
            batch_size, len(smpl_x.joint_part["lhand"]), 3
        )

        neutral_body_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(neutral_body_pose))
        )
        jaw_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(jaw_pose))
        )

        zero_pose = zero_pose.unsqueeze(1)
        jaw_pose_inv = jaw_pose_inv.unsqueeze(1)

        pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose_inv,
                jaw_pose_inv,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )

        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]

        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.smplx_layer.parents
        )  # [B, 55, 4, 4]

        return (
            mesh_neutral_pose_upsampled,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
        )

    def batch_rodrigues_torch(self, poses):
        """
        poses: B x N x 3 (axis-angle representation)
        return: B x N x 3 x 3 rotation matrices
        """
        B, N, _ = poses.shape
        poses = poses.view(-1, 3)  # (B*N, 3)

        angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)  # (B*N, 1)
        rot_dir = poses / angle  # (B*N, 3)

        cos = torch.cos(angle)[:, None]  # (B*N, 1, 1)
        sin = torch.sin(angle)[:, None]  # (B*N, 1, 1)

        rx, ry, rz = rot_dir[:, 0:1], rot_dir[:, 1:2], rot_dir[:, 2:3]
        zeros = torch.zeros_like(rx)

        K = torch.cat([
            zeros, -rz, ry,
            rz, zeros, -rx,
            -ry, rx, zeros
        ], dim=1).reshape(-1, 3, 3)  # (B*N, 3, 3)

        ident = torch.eye(3, device=poses.device).unsqueeze(0)  # (1, 3, 3)
        rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)  # (B*N, 3, 3)

        return rot_mat.view(B, N, 3, 3)


def read_smplx_param(smplx_data_root, shape_param_file, batch_size=1, device="cuda"):
    import json
    from glob import glob

    import cv2

    data_root_path = osp.dirname(osp.dirname(smplx_data_root))

    # load smplx parameters
    smplx_param_path_list = sorted(glob(osp.join(smplx_data_root, "*.json")))
    print(smplx_param_path_list[:3])

    smplx_params_all_frames = {}
    for smplx_param_path in smplx_param_path_list:
        frame_idx = int(smplx_param_path.split("/")[-1][:-5])
        with open(smplx_param_path) as f:
            smplx_params_all_frames[frame_idx] = {
                k: torch.FloatTensor(v) for k, v in json.load(f).items()
            }

    with open(shape_param_file) as f:
        shape_param = torch.FloatTensor(json.load(f))

    smplx_params = {}
    smplx_params["betas"] = shape_param.unsqueeze(0).repeat(batch_size, 1)

    select_frame_idx = [200, 400, 600]
    smplx_params_tmp = defaultdict(list)
    cam_param_list = []
    ori_image_list = []
    for b_idx in range(batch_size):
        frame_idx = select_frame_idx[b_idx]

        for k, v in smplx_params_all_frames[frame_idx].items():
            smplx_params_tmp[k].append(v)

        with open(
            osp.join(data_root_path, "cam_params", str(frame_idx) + ".json")
        ) as f:
            cam_param = {
                k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()
            }
            cam_param_list.append(cam_param)

        img = cv2.imread(osp.join(data_root_path, "frames", str(frame_idx) + ".png"))
        ori_image_list.append(img)

    for k, v in smplx_params_tmp.items():
        smplx_params[k] = torch.stack(smplx_params_tmp[k])

    root_path = osp.dirname(smplx_data_root)
    with open(osp.join(root_path, "face_offset.json")) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, "joint_offset.json")) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, "locator_offset.json")) as f:
        locator_offset = torch.FloatTensor(json.load(f))

    smplx_params["locator_offset"] = locator_offset.unsqueeze(0).repeat(
        batch_size, 1, 1
    )
    smplx_params["joint_offset"] = joint_offset.unsqueeze(0).repeat(batch_size, 1, 1)
    smplx_params["face_offset"] = face_offset.unsqueeze(0).repeat(batch_size, 1, 1)

    for k, v in smplx_params.items():
        print(k, v.shape)
        smplx_params[k] = v.to(device)

    return smplx_params, cam_param_list, ori_image_list


def test():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    gender = "male"
    # gender = "neutral"

    smplx_model = SMPLXMesh_Model(human_model_path, gender, subdivide_num=2)
    smplx_model.to("cuda")

    smplx_data_root = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/smplx_params_smoothed"
    shape_param_file = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/shape_param.json"
    smplx_data, cam_param_list, ori_image_list = read_smplx_param(
        smplx_data_root=smplx_data_root, shape_param_file=shape_param_file, batch_size=2
    )
    posed_verts = smplx_model.transform_to_posed_verts(
        smplx_data=smplx_data, device="cuda"
    )

    smplx_face = smplx_model.smpl_x.face_upsampled
    trimesh.Trimesh(
        vertices=posed_verts[0].detach().cpu().numpy(), faces=smplx_face
    ).export("./posed_obj1.obj")
    trimesh.Trimesh(
        vertices=posed_verts[1].detach().cpu().numpy(), faces=smplx_face
    ).export("./posed_obj2.obj")

    neutral_posed_verts, _, _ = smplx_model.get_query_points(
        smplx_data=smplx_data, device="cuda"
    )
    smplx_face = smplx_model.smpl_x.face
    trimesh.Trimesh(
        vertices=neutral_posed_verts[0].detach().cpu().numpy(), faces=smplx_face
    ).export("./neutral_posed_obj1.obj")
    trimesh.Trimesh(
        vertices=neutral_posed_verts[1].detach().cpu().numpy(), faces=smplx_face
    ).export("./neutral_posed_obj2.obj")

    for idx, (cam_param, img) in enumerate(zip(cam_param_list, ori_image_list)):
        render_shape = img.shape[:2]
        mesh_render, is_bkg = render_mesh(
            posed_verts[idx],
            smplx_face,
            cam_param,
            np.ones((render_shape[0], render_shape[1], 3), dtype=np.float32) * 255,
            return_bg_mask=True,
        )
        mesh_render = mesh_render.astype(np.uint8)
        cv2.imwrite(
            f"./debug_render_{idx}.jpg",
            np.clip(
                (0.9 * mesh_render + 0.1 * img) * (1 - is_bkg) + is_bkg * img, 0, 255
            ).astype(np.uint8),
        )
        # cv2.imwrite(f"./debug_render_{idx}_img.jpg", np.clip(img, 0, 255).astype(np.uint8))
        # cv2.imwrite(f"./debug_render_{idx}_mesh.jpg", np.clip(mesh_render, 0, 255).astype(np.uint8))


def read_smplx_param_humman(
    imgs_root, smplx_params_root, img_size=896, batch_size=1, device="cuda"
):
    import json
    import os
    from glob import glob

    import cv2
    from PIL import Image, ImageOps

    # Input images
    suffixes = (".jpg", ".jpeg", ".png", ".webp")
    img_path_list = [
        os.path.join(imgs_root, file)
        for file in os.listdir(imgs_root)
        if file.endswith(suffixes) and file[0] != "."
    ]

    ori_image_list = []
    smplx_params_tmp = defaultdict(list)

    for img_path in img_path_list:
        smplx_path = os.path.join(
            smplx_params_root, os.path.splitext(os.path.basename(img_path))[0] + ".json"
        )

        # Open and reshape
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = ImageOps.contain(
            img_pil, (img_size, img_size)
        )  # keep the same aspect ratio
        # ori_w, ori_h = img_pil.size
        # img_pil_pad = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side
        # offset_w, offset_h = (img_size - ori_w) // 2, (img_size - ori_h) // 2

        # img = np.array(img_pil_pad)[:, :, (2, 1, 0)]
        img = np.array(img_pil)[:, :, (2, 1, 0)]
        ori_image_list.append(img)

        with open(smplx_path) as f:
            smplx_param = {k: torch.FloatTensor(v) for k, v in json.load(f).items()}

        for k, v in smplx_param.items():
            smplx_params_tmp[k].append(v)

    smplx_params = {}
    for k, v in smplx_params_tmp.items():
        smplx_params[k] = torch.stack(smplx_params_tmp[k])

    for k, v in smplx_params.items():
        print(k, v.shape)
        smplx_params[k] = v.to(device)

    cam_param_list = []
    for i in range(smplx_params["focal"].shape[0]):
        princpt = smplx_params["princpt"][i]
        cam_param = {"focal": smplx_params["focal"][i], "princpt": princpt}
        cam_param_list.append(cam_param)
    return smplx_params, cam_param_list, ori_image_list


def generate_smplx_point():

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
                smplx_params[k] = data[k].unsqueeze(0).cuda()
        return smplx_params

    def sample_one(data):
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "trans",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                data[k] = data[k][:, 0]
        return data

    human_model_path = "./pretrained_models/human_model_files"
    gender = "neutral"
    subdivide_num = 1
    smplx_model = SMPLXVoxelMeshModel(
        human_model_path,
        gender,
        shape_param_dim=10,
        expr_param_dim=100,
        subdivide_num=subdivide_num,
        dense_sample_points=40000,
        cano_pose_type=1,
    )
    smplx_model.to("cuda")

    # save_file = f"pretrained_models/human_model_files/smplx_points/smplx_subdivide{subdivide_num}.npy"
    save_file = f"debug/smplx_points/smplx_subdivide{subdivide_num}.npy"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    smplx_data = {}
    smplx_data["betas"] = torch.zeros((1, 10)).to(device="cuda")
    mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
        smplx_model.get_query_points(smplx_data=smplx_data, device="cuda")
    )

    debug_pose = torch.load("./debug/pose_example.pth")
    debug_pose["expr"] = torch.FloatTensor([0.0] * 100)

    smplx_data = get_smplx_params(debug_pose)
    smplx_data = sample_one(smplx_data)
    smplx_data["betas"] = torch.ones_like(smplx_data["betas"])

    warp_posed, _ = smplx_model.transform_to_posed_verts_from_neutral_pose(
        mesh_neutral_pose,
        smplx_data,
        mesh_neutral_pose,
        transform_mat_neutral_pose,
        "cuda",
    )

    # save_ply("warp_posed.ply", warp_posed[0].detach().cpu())
    save_ply(
        "is_upper_body_posed.ply",
        warp_posed[0, smplx_model.is_upper_body].detach().cpu(),
    )



def visualize_3d_points(points1, points2=None, labels=("Points1", "Points2")):
    """
    Visualize one or two sets of 3D points using matplotlib.

    Args:
        points1 (Tensor or ndarray): [N, 3]
        points2 (Tensor or ndarray): [N, 3] (optional)
        labels (tuple): Labels for the two point sets
    """
    if isinstance(points1, torch.Tensor):
        points1 = points1.cpu().numpy()
    if points2 is not None and isinstance(points2, torch.Tensor):
        points2 = points2.cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='b', label=labels[0], s=10)
    if points2 is not None:
        ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='r', label=labels[1], s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Visualization")
    ax.legend()
    ax.view_init(elev=20, azim=45)
    ax.grid(True)

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    generate_smplx_point()
