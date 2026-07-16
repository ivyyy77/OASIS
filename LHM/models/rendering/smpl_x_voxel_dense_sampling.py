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
from smplx.lbs import batch_rigid_transform
from torch.nn import functional as F

from LHM.models.rendering.mesh_utils import Mesh
from LHM.models.rendering.smplx import smplx
from LHM.models.rendering.smplx.smplx.lbs import blend_shapes

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

        self.rhand_vertex_idx = hand_vertex_idx["right_hand"]
        self.lhand_vertex_idx = hand_vertex_idx["left_hand"]
        self.expr_vertex_idx = self.get_expr_vertex_idx()

        self.joint_num = (
            55
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
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
            "Jaw",
            "L_Eye",
            "R_Eye",
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
            "L_Thumb_3",
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
            "R_Thumb_3",
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

        self.neutral_body_pose = torch.zeros(
            (len(self.joint_part["body"]) - 1, 3)
        )

        if cano_pose_type == 0:
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1])
        else:
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, math.pi / 9])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -math.pi / 9])

        self.neutral_jaw_pose = torch.FloatTensor([1 / 3, 0, 0])

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
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
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
            "L_Thumb_3",
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
            "R_Thumb_3",
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
        """
        SMPLX + FLAME2019 Version
        according to LBS weights to search related vertices ID
        """

        with open(
            osp.join(self.human_model_path, "flame", "2019", "generic_model.pkl"), "rb"
        ) as f:
            flame_2019 = pickle.load(f, encoding="latin1")
        vertex_idxs = np.where(
            (flame_2019["shapedirs"][:, :, 300 : 300 + self.expr_param_dim] != 0).sum(
                (1, 2)
            )
            > 0
        )[
            0
        ]

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

        self.smpl_x = SMPLX_Mesh(
            human_model_path=human_model_path,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            subdivide_num=subdivide_num,
            cano_pose_type=cano_pose_type,
        )
        self.smplx_layer = copy.deepcopy(self.smpl_x.layer[gender])

        self.pose_mean__ = self.smplx_layer.pose_mean
        self.pose_mean_ = torch.zeros_like(self.pose_mean__).to('cuda')
        self.pose_mean_[75:120] = self.smplx_layer.left_hand_mean
        self.pose_mean_[120:165] = self.smplx_layer.right_hand_mean

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
        buff_path = f"./pretrained_models/dense_sample_points/{self.cano_pose_type}_{dense_sample_points}.ply"

        if os.path.exists(buff_path):
            dense_sample_pts, _ = load_ply(buff_path)

            _bin = dense_sample_points // (body_face_ratio + 1)
            body_pts = int(_bin * body_face_ratio)
            self.is_body = torch.arange(dense_sample_pts.shape[0])
            self.is_body[:body_pts] = 1
            self.is_body[body_pts:] = 0
            self.dense_pts = dense_sample_pts
        else:
            smpl_x = self.smpl_x
            body_face_mapping = smpl_x.get_body_face_mapping()
            face = smpl_x.face
            template_verts = self.smplx_layer.v_template

            _bin = dense_sample_points // (body_face_ratio + 1)

            body_pts = int(_bin * body_face_ratio)
            body_dict = body_face_mapping["body"]
            face = body_dict["face"]
            verts = body_dict["vert"]

            dense_body_pts = self.rebuild_mesh(template_verts, verts, face, body_pts)

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
        knn_weights = knn_weights / knn_weights.sum(-1, keepdim=True)

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

        update_weights = dists_to_weights(mesh_dis, low=0.01).unsqueeze(-1)

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

        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
            voxel_size - 1
        )
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

        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

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

        dense_pts = self.dense_pts.cuda()
        template_verts = self.smplx_layer.v_template

        nn_vertex_idxs = knn_points(
            dense_pts.unsqueeze(0).cuda(),
            template_verts.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        ).idx
        query_indx = nn_vertex_idxs.squeeze(0, -1).detach().cpu()

        skinning_weight = self.smplx_layer.lbs_weights.float()

        """ PCA regression function w.r.t vertices offset
        """
        pose_dirs = self.smplx_layer.posedirs.permute(1, 0).reshape(
            smpl_x.vertex_num, 3 * (smpl_x.joint_num - 1) * 9
        )
        expr_dirs = self.smplx_layer.expr_dirs.view(
            smpl_x.vertex_num, 3 * smpl_x.expr_param_dim
        )
        shape_dirs = self.smplx_layer.shapedirs.view(
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

        skinning_weight = _query(skinning_weight, query_indx)
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

        pose_dirs = pose_dirs.reshape(
            vertex_num_upsampled * 3, (smpl_x.joint_num - 1) * 9
        ).permute(1, 0)
        expr_dirs = expr_dirs.view(vertex_num_upsampled, 3, smpl_x.expr_param_dim)
        shape_dirs = shape_dirs.view(vertex_num_upsampled, 3, smpl_x.shape_param_dim)

        (
            is_rhand,
            is_lhand,
            is_face,
            is_face_expr,
            is_lower_body,
            is_upper_body,
            is_constrain_body,
        ) = (
            is_rhand[:, 0] > 0,
            is_lhand[:, 0] > 0,
            is_face[:, 0] > 0,
            is_face_expr[:, 0] > 0,
            is_lower_body[:, 0] > 0,
            is_upper_body[:, 0] > 0,
            is_constrain_body[:, 0] > 0,
        )
        is_cavity = is_cavity[:, 0] > 0

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

        self.vertex_num_upsampled = vertex_num_upsampled
        self.smpl_x.vertex_num_upsampled = vertex_num_upsampled

        voxel_skinning_weight, voxel_bbox = self.voxel_skinning_init(voxel_size=192)

        self.register_buffer("voxel_ws", voxel_skinning_weight)
        self.register_buffer("voxel_bbox", voxel_bbox)

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

    def query_voxel_skinning_weights(self, vs):
        """using voxel-based skinning method
        vs: [B n c]
        """
        voxel_bbox = self.voxel_bbox

        scale = voxel_bbox[..., 1] - voxel_bbox[..., 0]
        center = voxel_bbox.mean(dim=1)
        normalized_vs = (vs - center[None, None, :]) / scale[None, None]

        normalized_vs = normalized_vs * 2
        normalized_vs.to(self.voxel_ws)

        B, N, _ = normalized_vs.shape

        query_ws = F.grid_sample(
            self.voxel_ws.unsqueeze(0),
            normalized_vs.reshape(1, 1, 1, -1, 3).to(self.voxel_ws),
            align_corners=True,
            padding_mode="border",
        )
        query_ws = query_ws.view(B, -1, N)
        query_ws = query_ws.permute(0, 2, 1)

        return query_ws

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
        joint_zero_pose = output.joints[:, : smpl_x.joint_num, :]

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

        transform_mat_joint_1 = transform_mat_neutral_pose

        root_pose = smplx_param["root_pose"]
        body_pose = smplx_param["body_pose"]
        jaw_pose = smplx_param["jaw_pose"]
        leye_pose = smplx_param["leye_pose"]
        reye_pose = smplx_param["reye_pose"]
        lhand_pose = smplx_param["lhand_pose"]
        rhand_pose = smplx_param["rhand_pose"]

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
        )
        pose[0, :, :] = pose[0, :, :] + self.pose_mean_.reshape(-1, 3)
        pose = axis_angle_to_matrix(pose)

        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )
        transform_mat_joint_2 = transform_mat_joint_2

        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )
        else:
            transform_mat_joint = transform_mat_joint_2

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, query_points, fix_mask):
        batch_size = transform_mat_joint.shape[0]

        query_skinning = self.query_voxel_skinning_weights(query_points)
        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_skinning[fix_mask] = skinning_weight[fix_mask]

        transform_mat_vertex = torch.matmul(
            skinning_weight,
            transform_mat_joint.view(batch_size, self.smpl_x.joint_num, 16),
        ).view(batch_size, self.smpl_x.vertex_num_upsampled, 4, 4)
        return transform_mat_vertex

    def get_posed_blendshape(self, smplx_param):
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
        )
        pose += self.pose_mean_.reshape(-1, 3)[1:, :]

        pose = (
            axis_angle_to_matrix(pose) - torch.eye(3)[None, None, :, :].float().cuda()
        ).view(batch_size, (self.smpl_x.joint_num - 1) * 9)

        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(
            batch_size, self.smpl_x.vertex_num_upsampled, 3
        )
        return smplx_pose_offset

    def lbs(self, xyz, transform_mat_vertex, trans):
        batch_size = xyz.shape[0]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, self.vertex_num_upsampled, 4
        )[
            :, :, :3
        ]

        if trans is not None:
            xyz = xyz + trans.unsqueeze(1)
        return xyz

    def lr_idx_to_hr_idx(self, idx):
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

        face_offset = smplx_data.get("face_offset", None)
        joint_offset = smplx_data.get("joint_offset", None)

        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]

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

        try:
            smplx_expr_offset = (
                smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
            ).sum(
                -1
            )
        except:
            print("no use flame params")
            smplx_expr_offset = 0.0

        mean_3d = mean_3d + smplx_expr_offset

        mask = (

            ((self.is_rhand + self.is_lhand) > 0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        transform_mat_null_vertex = self.get_transform_mat_vertex(
            transform_mat_neutral_pose, mean_3d, mask
        )

        null_mean_3d = self.lbs(
            mean_3d, transform_mat_null_vertex, torch.zeros_like(smplx_data["trans"])
        )

        blend_shape_offset = blend_shapes(shape_param, self.shape_dirs)
        null_mean3d_blendshape = null_mean_3d + blend_shape_offset

        joint_null_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        transform_mat_joint, j3d = self.get_transform_mat_joint(

            None, joint_null_pose, smplx_data
        )

        pose_offsets = self.get_posed_blendshape(smplx_data)

        null_mean3d_blendshape = null_mean3d_blendshape + pose_offsets

        transform_mat_vertex = self.get_transform_mat_vertex(

            transform_mat_joint, mean_3d, mask
        )

        posed_mean_3d = self.lbs(
            null_mean3d_blendshape, transform_mat_vertex, smplx_data["trans"]
        )

        neutral_to_posed_vertex = torch.matmul(
            transform_mat_vertex, transform_mat_null_vertex
        )

        return posed_mean_3d, neutral_to_posed_vertex

    def get_query_points(self, smplx_data, device):
        """transform_mat_neutral_pose is function to warp pre-defined posed to zero-pose"""

        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose_human(
                jaw_zero_pose=True,
                use_id_info=False,
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

        mesh_neutral_pose, _, transform_mat_neutral_pose = self.get_query_points(
            smplx_data, device
        )

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
        )

        neutral_pose = axis_angle_to_matrix(
            neutral_pose.view(-1, 55, 3)
        )
        posed_joints, transform_mat_joint = batch_rigid_transform(
            neutral_pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )

        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)

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

        smplx_expr_offset = (
            smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
        ).sum(
            -1
        )
        mean_3d = mean_3d + smplx_expr_offset

    def get_neutral_pose_human(
        self, jaw_zero_pose, use_id_info, shape_param, device, face_offset, joint_offset
    ):
        smpl_x = self.smpl_x
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        neutral_body_pose = (
            smpl_x.neutral_body_pose.view(1, -1).repeat(batch_size, 1).to(device)
        )
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
            )

        if use_id_info:
            shape_param = shape_param

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
        ]

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

        pose = axis_angle_to_matrix(pose)

        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.smplx_layer.parents
        )

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
        poses = poses.view(-1, 3)

        angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
        rot_dir = poses / angle

        cos = torch.cos(angle)[:, None]
        sin = torch.sin(angle)[:, None]

        rx, ry, rz = rot_dir[:, 0:1], rot_dir[:, 1:2], rot_dir[:, 2:3]
        zeros = torch.zeros_like(rx)

        K = torch.cat([
            zeros, -rz, ry,
            rz, zeros, -rx,
            -ry, rx, zeros
        ], dim=1).reshape(-1, 3, 3)

        ident = torch.eye(3, device=poses.device).unsqueeze(0)
        rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

        return rot_mat.view(B, N, 3, 3)

class MANOVoxelMeshModel(nn.Module):
    def __init__(self, hanco=False, center_add=True) -> None:
        super().__init__()

        self.mano = tools.model.smplx.create(**cfg.smpl_cfg)
        self.mano, _, _ = sub_mano(self.mano, cfg.smpl_cfg['manohd'])
        lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
        self.mano.lbs_weights = lbs_weights
        self.pose_mean = self.mano.pose_mean.to('cuda')

        self.center_add = center_add

        self.apply_pose_blendshape = True
        self.dense_sample()
        self.manohd_init()

        self.joint_null_pose = self.get_zero_pose_hand(
            shape_param=torch.zeros(1,10).float()).to('cuda')

        self.center_id = self.mano.center_id
        self.scale = self.mano.scale
        self.mano.lbs_weights.to('cuda')

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
        semantic_path = "./pretrained_models/dense_sample_points/manohd_semantic.ply"

        manohd_pts, _ = load_ply(semantic_path)
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
        knn_weights = knn_weights / knn_weights.sum(-1, keepdim=True)

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

        update_weights = dists_to_weights(mesh_dis, low=0.01).unsqueeze(-1)

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

        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
                voxel_size - 1
        )
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

        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

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

        dense_pts = self.hand_pts.cuda()
        template_verts = self.mano.v_template

        nn_vertex_idxs = knn_points(
            dense_pts.unsqueeze(0).cuda(),
            template_verts.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        ).idx

        """ PCA regression function w.r.t vertices offset
        """

        self.mano.vertex_num_upsampled = self.hand_pts

        voxel_skinning_weight, voxel_bbox = self.hand_voxel_skinning_init(voxel_size=192)

        self.register_buffer("hand_voxel_ws", voxel_skinning_weight)
        self.register_buffer("hand_voxel_bbox", voxel_bbox)

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

        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
                voxel_size - 1
        )
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

        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

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

        normalized_vs = normalized_vs * 2
        normalized_vs.to(self.voxel_ws)

        B, N, _ = normalized_vs.shape

        query_ws = F.grid_sample(
            self.voxel_ws.unsqueeze(0),
            normalized_vs.reshape(1, 1, 1, -1, 3).to(self.voxel_ws),
            align_corners=True,
            padding_mode="border",
        )
        query_ws = query_ws.view(B, -1, N)
        query_ws = query_ws.permute(0, 2, 1)

        return query_ws

    def get_zero_pose_hand(self, shape_param, return_mesh=False):
        output = self.mano(shape_param,
                           torch.zeros(1,3).float(),
                           torch.zeros(1,45).float(), return_verts=True)
        self.center = output.center.squeeze().to('cuda')

        joint_zero_pose = output.joints[:, : self.mano.NUM_JOINTS, :]

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

        transform_mat_joint_1 = transform_mat_neutral_pose

        poses_tensor = smplx_param["poses"]
        if poses_tensor.device.type == 'cpu':
            poses_tensor = poses_tensor.to('cuda')
        root_pose = poses_tensor[0, :3]
        hand_pose = poses_tensor[0, 3:]

        pose = torch.cat(
            (
                root_pose.unsqueeze(0),
                hand_pose.unsqueeze(0),
            ),
            dim=1,
        ).reshape(-1, 3).unsqueeze(0)

        pose = pose + self.pose_mean.reshape(-1, 3)

        pose = axis_angle_to_matrix(pose)

        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.mano.parents.to('cuda')
        )

        transform_mat_joint_2 = transform_mat_joint_2

        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )
        else:
            transform_mat_joint = transform_mat_joint_2

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, query_points):
        batch_size = transform_mat_joint.shape[0]

        skinning_weight = self.mano.lbs_weights.repeat(batch_size, 1, 1).to('cuda')

        skinning_weight = torch.log(skinning_weight + 1e-9) + skinning_weight
        skinning_weight = F.softmax(skinning_weight, dim=-1)

        transform_mat_vertex = torch.matmul(
            skinning_weight,
            transform_mat_joint.view(batch_size, self.mano.NUM_JOINTS, 16),
        ).view(batch_size, self.mano.vertex_num_upsampled.size(0), 4, 4)
        return transform_mat_vertex

    def get_posed_blendshape(self, smplx_param):
        poses_tensor = smplx_param["poses"]
        if poses_tensor.device.type == 'cpu':
            poses_tensor = poses_tensor.to('cuda')
        pose_full = poses_tensor[0,:].unsqueeze(0)

        hand_pose = pose_full[:, 3:]
        batch_size = 1
        pose = hand_pose.reshape(batch_size, -1, 3)

        pose = pose + self.pose_mean.reshape(-1, 3)[1:, :]

        pose = (
                axis_angle_to_matrix(pose) - torch.eye(3)[None, None, :, :].float().cuda()
        ).view(batch_size, (self.mano.NUM_JOINTS - 1) * 9)

        smplx_pose_offset = torch.matmul(pose.unsqueeze(1), self.mano.posedirs.to('cuda').view(
            self.mano.vertex_num_upsampled.size(0)*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, self.mano.vertex_num_upsampled.size(0), 3)

        return smplx_pose_offset

    def lbs(self, xyz, transform_mat_vertex, trans):
        batch_size = xyz.shape[0]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, self.mano.vertex_num_upsampled.size(0), 4
        )[:, :, :3]

        if trans is not None:
            xyz = xyz + trans.unsqueeze(1)
        return xyz

    def lr_idx_to_hr_idx(self, idx):
        return idx

    def transform_to_posed_verts_from_neutral_pose(
            self, mean_3d, smplx_data,
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
        shape_param = smplx_data["shape"]

        if shape_param.device.type == 'cpu':
            shape_param = shape_param.to('cuda')
        merge_mean_3d = mean_3d
        mean_3d = mean_3d[0, ...]

        blend_shape_offset = blend_shapes(shape_param, self.mano.shapedirs.to('cuda'))
        null_mean3d_blendshape = mean_3d
        merge_mean_3d[0, ...] = null_mean3d_blendshape

        transform_mat_joint, j3d = self.get_transform_mat_joint(

            None, self.joint_null_pose, smplx_data)

        pose_offsets = self.get_posed_blendshape(smplx_data)

        null_mean3d_blendshape = null_mean3d_blendshape + pose_offsets

        transform_mat_vertex = self.get_transform_mat_vertex(

            transform_mat_joint, mean_3d
        )

        posed_mean_3d = self.lbs(
            null_mean3d_blendshape, transform_mat_vertex, None,

        )

        center = j3d[:, self.center_id:self.center_id+1].clone()

        if getattr(self, 'center_add', True):
            posed_mean_3d = posed_mean_3d - center

        merge_mean_3d[1, ...] = posed_mean_3d
        return merge_mean_3d, transform_mat_vertex

    def get_transform_params_torch(self, params, rot_mats=None, correct_Rs=None):
        """ obtain the transformation parameters for linear blend skinning
        """
        v_template = self.mano.v_template.cuda()

        shapedirs = self.mano.shapedirs.cuda()

        betas = params['shape'].unsqueeze(0)

        v_shaped = v_template[None] + torch.sum(shapedirs[None][..., :betas.shape[-1]] * betas[:, None],
                                                axis=-1).float()

        joints = torch.matmul(self.mano.J_regressor[None].cuda(),
                              v_shaped)

        if rot_mats is None:
            poses = params['poses'][0, :].reshape(-1, 3)

            rot_mats = self.batch_rodrigues_torch_new(poses).view(-1, poses.shape[0], 3, 3)

        parents = self.mano.parents.cuda()

        A, posed_joints = self.get_rigid_transformation_torch(rot_mats, joints, parents)

        return A, joints, posed_joints

    def batch_rodrigues_torch_new(self, poses):
        """ poses: N x 3
        """
        batch_size = poses.shape[0]
        angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
        rot_dir = poses / angle

        cos = torch.cos(angle)[:, None]
        sin = torch.sin(angle)[:, None]

        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        zeros = torch.zeros((batch_size, 1), device=poses.device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
        K = K.reshape([batch_size, 3, 3])

        ident = torch.eye(3)[None].to(poses.device)
        rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

        return rot_mat

    def get_rigid_transformation_torch(self, rot_mats, joints, parents):
        """
        rot_mats: bs x 24 x 3 x 3
        joints: bs x 24 x 3
        parents: 24
        """

        bs, joints_num = joints.shape[0:2]
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
        padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)
        padding[..., 3] = 1
        transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:,
                                                                 i])
            transform_chain.append(curr_res)
        transforms = torch.stack(transform_chain, dim=1)

        posed_joints = transforms[:, :, :3, 3]

        padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)
        joints_homogen = torch.cat([joints, padding], dim=-1)
        rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
        transforms[..., 3] = transforms[..., 3] - rel_joints

        return transforms, posed_joints

    def get_query_points(self, smplx_data, device):
        """transform_mat_neutral_pose is function to warp pre-defined posed to zero-pose"""

        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose_human(
                jaw_zero_pose=True,
                use_id_info=False,
                shape_param=smplx_data["shape"],
                device=device,

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

        mesh_neutral_pose, _, transform_mat_neutral_pose = self.get_query_points(
            smplx_data, device
        )

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
        )

        neutral_pose = axis_angle_to_matrix(
            neutral_pose.view(-1, 55, 3)
        )
        posed_jints, transform_mat_joint = batch_rigid_transform(
            neutral_pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )

        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)

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

        smplx_expr_offset = (
                smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
        ).sum(
            -1
        )
        mean_3d = mean_3d + smplx_expr_offset

    def get_neutral_pose_human(
            self, jaw_zero_pose, use_id_info, shape_param, device, face_offset, joint_offset
    ):
        smpl_x = self.smpl_x
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        neutral_body_pose = (
            smpl_x.neutral_body_pose.view(1, -1).repeat(batch_size, 1).to(device)
        )
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
            )

        if use_id_info:
            shape_param = shape_param

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
                             ]

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

        pose = axis_angle_to_matrix(pose)

        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.smplx_layer.parents
        )

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
        poses = poses.view(-1, 3)

        angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
        rot_dir = poses / angle

        cos = torch.cos(angle)[:, None]
        sin = torch.sin(angle)[:, None]

        rx, ry, rz = rot_dir[:, 0:1], rot_dir[:, 1:2], rot_dir[:, 2:3]
        zeros = torch.zeros_like(rx)

        K = torch.cat([
            zeros, -rz, ry,
            rz, zeros, -rx,
            -ry, rx, zeros
        ], dim=1).reshape(-1, 3, 3)

        ident = torch.eye(3, device=poses.device).unsqueeze(0)
        rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

        return rot_mat.view(B, N, 3, 3)
