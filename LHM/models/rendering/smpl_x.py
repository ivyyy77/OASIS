import copy
import math
import os
import os.path as osp
import pickle
import sys

sys.path.append("./")
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.ops import SubdivideMeshes, knn_points
from pytorch3d.structures import Meshes
from smplx.lbs import batch_rigid_transform
from torch.nn import functional as F

from LHM.models.rendering.smplx import smplx

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

class SMPLX(object):
    def __init__(
        self,
        human_model_path,
        shape_param_dim=100,
        expr_param_dim=50,
        subdivide_num=2,
        cano_pose_type=0,
    ):
        """SMPLX using pytorch3d subdivsion"""
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

        self.lower_body_vertex_idx = self.get_lower_body()

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

        self.subdivide_num = subdivide_num
        self.subdivider_list = self.get_subdivider(subdivide_num)
        self.subdivider_cpu_list = self.get_subdivider_cpu(subdivide_num)
        self.face_upsampled = (
            self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
            if self.subdivide_num > 0
            else self.face
        )
        print("face_upsampled:", self.face_upsampled.shape)
        self.vertex_num_upsampled = int(np.max(self.face_upsampled) + 1)

    def get_lower_body(self):
        """using skinning to find lower body vertices."""
        lower_body_skinning_index = set(self.joint_part["lower_body"])
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

    def get_subdivider_cpu(self, subdivide_num):
        vert = self.layer["neutral"].v_template.float()
        face = torch.LongTensor(self.face)
        mesh = Meshes(vert[None, :, :], face[None, :, :])

        if subdivide_num > 0:
            subdivider_list = [SubdivideMeshes(mesh)]
            for i in range(subdivide_num - 1):
                mesh = subdivider_list[-1](mesh)
                subdivider_list.append(SubdivideMeshes(mesh))
        else:
            subdivider_list = [mesh]
        return subdivider_list

    def upsample_mesh_cpu(self, vert, feat_list=None):
        face = torch.LongTensor(self.face)
        mesh = Meshes(vert[None, :, :], face[None, :, :])
        if self.subdivide_num > 0:
            if feat_list is None:
                for subdivider in self.subdivider_cpu_list:
                    mesh = subdivider(mesh)
                vert = mesh.verts_list()[0]
                return vert
            else:
                feat_dims = [x.shape[1] for x in feat_list]
                feats = torch.cat(feat_list, 1)
                for subdivider in self.subdivider_cpu_list:
                    mesh, feats = subdivider(mesh, feats)
                vert = mesh.verts_list()[0]
                feats = feats[0]
                feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list
        else:
            if feat_list is None:
                return vert
            else:
                return vert, *feat_list

    def upsample_mesh(self, vert, feat_list=None, device="cuda"):
        face = torch.LongTensor(self.face).to(device)
        mesh = Meshes(vert[None, :, :], face[None, :, :])
        if self.subdivide_num > 0:
            if feat_list is None:
                for subdivider in self.subdivider_list:
                    mesh = subdivider(mesh)
                vert = mesh.verts_list()[0]
                return vert
            else:
                feat_dims = [x.shape[1] for x in feat_list]
                feats = torch.cat(feat_list, 1)
                for subdivider in self.subdivider_list:
                    mesh, feats = subdivider(mesh, feats)
                vert = mesh.verts_list()[0]
                feats = feats[0]
                feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list
        else:
            if feat_list is None:
                return vert
            else:
                return vert, *feat_list

    def upsample_mesh_batch(self, vert, device="cuda"):
        if self.subdivide_num > 0:
            face = (
                torch.LongTensor(self.face)
                .to(device)
                .unsqueeze(0)
                .repeat(vert.shape[0], 1, 1)
            )
            mesh = Meshes(vert, face)
            for subdivider in self.subdivider_list:
                mesh = subdivider(mesh)
            vert = torch.stack(mesh.verts_list(), dim=0)
        else:
            pass
        return vert

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

class SMPLXModel(nn.Module):
    def __init__(
        self,
        human_model_path,
        gender,
        subdivide_num,
        expr_param_dim=50,
        shape_param_dim=100,
        cano_pose_type=0,
        apply_pose_blendshape=False,
    ) -> None:
        super().__init__()

        self.smpl_x = SMPLX(
            human_model_path=human_model_path,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            subdivide_num=subdivide_num,
            cano_pose_type=cano_pose_type,
        )
        self.smplx_layer = copy.deepcopy(self.smpl_x.layer[gender])

        self.apply_pose_blendshape = apply_pose_blendshape

        self.smplx_init()

    def get_body_infos(self):
        head_id = torch.where(self.is_face == True)[0]
        body_id = torch.where(self.is_face == False)[0]
        return dict(head=head_id, body=body_id)

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

        smpl_x = self.smpl_x

        skinning_weight = self.smplx_layer.lbs_weights.float()

        """ PCA regression function w.r.t vertices offset
        """
        pose_dirs = self.smplx_layer.posedirs.permute(1, 0).reshape(
            smpl_x.vertex_num, 3 * (smpl_x.joint_num - 1) * 9
        )
        expr_dirs = self.smplx_layer.expr_dirs.view(
            smpl_x.vertex_num, 3 * smpl_x.expr_param_dim
        )

        is_rhand, is_lhand, is_face, is_face_expr, is_lower_body = (
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
        ) = (1.0, 1.0, 1.0, 1.0, 1.0)
        is_cavity = torch.FloatTensor(smpl_x.is_cavity)[:, None]

        (
            _,
            skinning_weight,
            pose_dirs,
            expr_dirs,
            is_rhand,
            is_lhand,
            is_face,
            is_face_expr,
            is_lower_body,
            is_cavity,
        ) = smpl_x.upsample_mesh_cpu(
            torch.ones((smpl_x.vertex_num, 3)).float(),
            [
                skinning_weight,
                pose_dirs,
                expr_dirs,
                is_rhand,
                is_lhand,
                is_face,
                is_face_expr,
                is_lower_body,
                is_cavity,
            ],
        )

        pose_dirs = pose_dirs.reshape(
            smpl_x.vertex_num_upsampled * 3, (smpl_x.joint_num - 1) * 9
        ).permute(
            1, 0
        )
        expr_dirs = expr_dirs.view(
            smpl_x.vertex_num_upsampled, 3, smpl_x.expr_param_dim
        )
        is_rhand, is_lhand, is_face, is_face_expr, is_lower_body = (
            is_rhand[:, 0] > 0,
            is_lhand[:, 0] > 0,
            is_face[:, 0] > 0,
            is_face_expr[:, 0] > 0,
            is_lower_body[:, 0] > 0,
        )
        is_cavity = is_cavity[:, 0] > 0

        self.register_buffer("skinning_weight", skinning_weight.contiguous())
        self.register_buffer("pose_dirs", pose_dirs.contiguous())
        self.register_buffer("expr_dirs", expr_dirs.contiguous())
        self.register_buffer("is_rhand", is_rhand.contiguous())
        self.register_buffer("is_lhand", is_lhand.contiguous())
        self.register_buffer("is_face", is_face.contiguous())
        self.register_buffer("is_lower_body", is_lower_body.contiguous())
        self.register_buffer("is_face_expr", is_face_expr.contiguous())
        self.register_buffer("is_cavity", is_cavity.contiguous())

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

        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh_batch(
            output.vertices, device=device
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
            mesh_zero_pose = output.vertices[0]
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(
                mesh_zero_pose
            )
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

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
        pose = axis_angle_to_matrix(pose)
        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )
        transform_mat_joint_2 = transform_mat_joint_2

        transform_mat_joint = torch.matmul(
            transform_mat_joint_2, transform_mat_joint_1
        )

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        batch_size = transform_mat_joint.shape[0]
        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        skinning_weight = skinning_weight.view(-1, skinning_weight.shape[-1])[
            nn_vertex_idxs.view(-1)
        ].view(
            nn_vertex_idxs.shape[0], nn_vertex_idxs.shape[1], skinning_weight.shape[-1]
        )
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
            batch_size, self.smpl_x.vertex_num_upsampled, 4
        )[
            :, :, :3
        ]
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
            smplx_expr_offset = 0.0

        if self.apply_pose_blendshape:
            smplx_pose_offset = self.get_posed_blendshape(smplx_data)
            mask = (
                ((self.is_rhand + self.is_lhand + self.is_face_expr) > 0)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            mean_3d[mask] += smplx_pose_offset[mask]

        nn_vertex_idxs = knn_points(
            mean_3d[:, :, :], mesh_neutral_pose[:, :, :], K=1, return_nn=True
        ).idx[
            :, :, 0
        ]

        mask = (
            ((self.is_rhand + self.is_lhand + self.is_face) > 0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        nn_vertex_idxs[mask] = (
            torch.arange(self.smpl_x.vertex_num_upsampled)
            .to(device)
            .unsqueeze(0)
            .repeat(batch_size, 1)[mask]
        )

        joint_zero_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        transform_mat_joint, j3d = self.get_transform_mat_joint(
            transform_mat_neutral_pose, joint_zero_pose, smplx_data
        )

        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, nn_vertex_idxs
        )

        mean_3d = self.lbs(
            mean_3d, transform_mat_vertex, smplx_data["trans"]
        )

        return mean_3d, transform_mat_vertex

    def get_query_points(self, smplx_data, device):
        """transform_mat_neutral_pose is function to warp pre-defined posed to zero-pose"""
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose_human(
                jaw_zero_pose=True,
                use_id_info=True,
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
