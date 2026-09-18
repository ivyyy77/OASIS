import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import math
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import logging
from typing import Optional

from LHM.models.rendering.gs_renderer import PointEmbed, GS_Hand_3DRenderer

from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding


from LHM.models.utils import linear


from .embedder import CameraEmbedder
from .transformer import TransformerDecoder

from data.interhand.train import Renderer_mesh

logger = logging.getLogger(__name__)


_GAUSSIANHAND_ROOT = Path(__file__).resolve().parents[2].parent / 'GuassianHand'
if _GAUSSIANHAND_ROOT.exists():
    if str(_GAUSSIANHAND_ROOT) not in sys.path:
        sys.path.append(str(_GAUSSIANHAND_ROOT))
    try:
        from livehand.input_encoder import get_uvd as gaussianhand_get_uvd
    except Exception:
        gaussianhand_get_uvd = None
else:
    gaussianhand_get_uvd = None

class ModelHandLRM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """

    def __init__(
        self,
        transformer_dim=1024,
        transformer_layers=8,
        transformer_heads=16,
        transformer_type="sd3_mm_cond",
        tf_grad_ckpt=True,
        encoder_grad_ckpt=True,
        encoder_freeze: bool = False,
        encoder_type: str = "dinov2_fusion",
        encoder_model_name: str = "dinov2_vitl14_reg",
        encoder_feat_dim: int = 1024,
        num_pcl: int = 2048,
        pcl_dim: int = 1024,
        human_model_path='./pretrained_models/human_model_files',
        smplx_subdivide_num=1,
        smplx_type="smplx",
        gs_query_dim=1024,

        gs_use_rgb=True,
        gs_sh=3,
        gs_mlp_network_config={'activation': 'silu', 'n_hidden_layers': 2, 'n_neurons': 512},
        gs_xyz_offset_max_step=1.0,
        gs_clip_scaling=[100, 0.01, 0.05, 3000],
        shape_param_dim=10,
        expr_param_dim=100,

        fix_opacity=True,
        fix_rotation=False,
    ):
        super(ModelHandLRM, self).__init__()

        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt

        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=encoder_feat_dim,
        )

        self.uv_texture_hw = (256, 256)
        self._warned_missing_uv_mapper = False

        pcl_dim = 1024

        input_dim = 1024

        input_dim_ = 1024

        mid_dim = input_dim // 2
        self.motion_embed_mlp = nn.Sequential(

            linear(input_dim, mid_dim),
            nn.SiLU(),
            linear(mid_dim, pcl_dim),

        ).to('cuda')

        skip_decoder = False
        self.latent_query_points_type = 'e2e_smplx_sub1'
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl
            self.pcl_embeddings = nn.Embedding(num_pcl, pcl_dim)
        elif self.latent_query_points_type.startswith("smplx"):
            latent_query_points_file = os.path.join(
                human_model_path, "smplx_points", f"{self.latent_query_points_type}.npy"
            )
            pcl_embeddings = torch.from_numpy(np.load(latent_query_points_file)).float()
            print(
                f"==========load smplx points:{latent_query_points_file}, shape:{pcl_embeddings.shape}"
            )
            self.register_buffer("pcl_embeddings", pcl_embeddings)
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        elif self.latent_query_points_type.startswith("e2e_smplx"):
            skip_decoder = True
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        else:
            raise NotImplementedError
        print(f"==========skip_decoder:{skip_decoder}")

        self.transformer = self.build_transformer(
            transformer_type,
            transformer_layers,
            transformer_heads,
            transformer_dim,

            encoder_feat_dim,
        )
        for blk in self.transformer.layers:
            if hasattr(blk, "part_aware_point"):
                blk.part_aware_point.to('cuda:0')

        self.uvmap_size = 1024

        self.uv_map_bias = None

        self.color_b_map = None
        self.opacity_b_map = None

        self.adapter = QFormerWithSelf(dim=input_dim_).cuda()

        self.aggregator = PatchAggregatorWithPosWeightedPool(
            C_in=input_dim_, D_hidden=128, out_dim=input_dim_).cuda()

        cano_pose_type = 1
        dense_sample_pts = 12337

        self.renderer = GS_Hand_3DRenderer(

            pcl_embed=None,
            feat_dim=transformer_dim,
            query_dim=gs_query_dim,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            mlp_network_config=gs_mlp_network_config,
            xyz_offset_max_step=gs_xyz_offset_max_step,
            clip_scaling=gs_clip_scaling,

            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            decoder_mlp=False,
            skip_decoder=skip_decoder,
            decode_with_extra_info=None,
            gradient_checkpointing=self.gradient_checkpointing,
            apply_pose_blendshape=False,

            feature_map=False,

            dataset_center_add_map={0: True, 1: True, 2: False},
        )

        self._init_uvmap_assets()

        params = self.obtain_params()

        self.optimizer = torch.optim.AdamW(params)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=5000,
                    eta_min=1e-6
                )

        self.checkpoint_path = os.path.join('./checkpoint/exp-mix3-1')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.uv_feat_fusion_weight = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _init_uvmap_assets(self):
        """Initialize UV map assets from MANO model for single right hand (256x256 resolution)."""
        import cv2

        mano = self.renderer.mano_model.mano
        uvmap_size = 256

        faces = mano.faces.numpy() if isinstance(mano.faces, torch.Tensor) else mano.faces

        if hasattr(mano, 'faces_uv_idx') and hasattr(mano, 'texcoords'):
            faces_uv_idx = mano.faces_uv_idx.numpy() if isinstance(mano.faces_uv_idx, torch.Tensor) else mano.faces_uv_idx
            texcoords = mano.texcoords.numpy() if isinstance(mano.texcoords, torch.Tensor) else mano.texcoords
        else:
            logger.warning("MANO model doesn't have UV attributes. Attempting to load from assets...")

            num_verts = mano.v_template.shape[0]
            faces_uv_idx = faces.copy()

            v_template_np = mano.v_template.numpy() if isinstance(mano.v_template, torch.Tensor) else mano.v_template
            v_min = v_template_np.min(axis=0)
            v_max = v_template_np.max(axis=0)
            texcoords = (v_template_np[:, :2] - v_min[:2]) / (v_max[:2] - v_min[:2] + 1e-8)

        uvmap_f_idx = self._get_uvmap_faces_index(faces_uv_idx, texcoords, uv_size=uvmap_size)

        uvmap_f_bary = self._get_uvmap_faces_barycoord(uvmap_f_idx, faces_uv_idx, texcoords, uv_size=uvmap_size)

        uvmap_mask = (uvmap_f_idx != -1)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.register_buffer('uvmap_f_idx', torch.tensor(uvmap_f_idx, dtype=torch.int32, device=device))
        self.register_buffer('uvmap_f_bary', torch.tensor(uvmap_f_bary, dtype=torch.float32, device=device))
        self.register_buffer('uvmap_mask', torch.tensor(uvmap_mask, dtype=torch.bool, device=device))

        self.uv_mask_flat = uvmap_mask.flatten()
        self.uvmap_size = uvmap_size

        logger.info(f"UV map assets initialized: size={uvmap_size}x{uvmap_size}, valid_pixels={uvmap_mask.sum().item()}")

    @staticmethod
    def _get_uvmap_faces_index(faces_uv, uv_coords, uv_size=256):
        """
        For each pixel in UV space, find which face it belongs to.
        Args:
            faces_uv: [num_faces, 3] face vertex indices in UV space
            uv_coords: [num_verts, 2] UV coordinates (normalized to [0, 1])
            uv_size: resolution of UV map
        Returns:
            uvmap_faces_idx: [uv_size, uv_size] face index for each pixel (-1 if background)
        """
        uv_coords_px = np.round(uv_coords * uv_size).astype(np.int32)
        faces_uv = faces_uv.astype(np.int32)
        uvmap_faces_idx = np.ones((uv_size, uv_size), dtype=np.int32) * -1

        for f_idx in range(len(faces_uv)):
            cv2.drawContours(uvmap_faces_idx, [uv_coords_px[faces_uv[f_idx]]], 0, int(f_idx), -1)

        return uvmap_faces_idx

    @staticmethod
    def _get_uvmap_faces_barycoord(uvmap_faces_idx, faces_uv, uv_coords, uv_size=256):
        """
        For each valid UV pixel, compute barycentric coordinates within its face.
        Args:
            uvmap_faces_idx: [uv_size, uv_size] face index
            faces_uv: [num_faces, 3] face vertex indices
            uv_coords: [num_verts, 2] UV coordinates (normalized to [0, 1])
            uv_size: resolution
        Returns:
            uvmap_faces_barycoord: [uv_size, uv_size, 3] barycentric coordinates
        """
        uv_coords_px = np.round(uv_coords * uv_size).astype(np.int32)
        uvmap_faces_barycoord = np.zeros((uv_size, uv_size, 3), dtype=np.float32)

        for u_idx in range(uv_size):
            for v_idx in range(uv_size):
                f_idx = uvmap_faces_idx[v_idx, u_idx]
                if f_idx == -1:
                    continue

                v_uvs = uv_coords_px[faces_uv[f_idx]]
                v_uv0, v_uv1, v_uv2 = v_uvs[0], v_uvs[1], v_uvs[2]
                c_uv = np.array([u_idx, v_idx])

                c_0 = c_uv - v_uv0
                c_1 = c_uv - v_uv1
                c_2 = c_uv - v_uv2

                area0 = 0.5 * np.abs(np.cross(c_1, c_2))
                area1 = 0.5 * np.abs(np.cross(c_0, c_2))
                area2 = 0.5 * np.abs(np.cross(c_0, c_1))
                total_area = area0 + area1 + area2 + 1e-6

                uvmap_faces_barycoord[v_idx, u_idx] = np.array([area0, area1, area2]) / total_area

        return uvmap_faces_barycoord

    def convert_pixel_feature_to_uv(self, img_features, deformed_vertices, w2c_cam, img_size=256):
        """
        Warp image features to UV map space using geometry.

        Args:
            img_features: [B, C, H_img, W_img] image-space feature map (H_img=W_img=256)
            deformed_vertices: [B, num_verts, 3] deformed mesh vertices in world space
            w2c_cam: [B, 4, 4] world-to-camera transformation matrix
            img_size: image resolution (256)

        Returns:
            uv_features: [B, C, uvmap_size, uvmap_size] features in UV space
        """
        batch_size, feature_dim = img_features.shape[0], img_features.shape[1]
        device = img_features.device

        uv_features = torch.zeros(
            (batch_size, feature_dim, self.uvmap_size, self.uvmap_size),
            device=device, dtype=torch.float32
        )

        uvmap_f_idx = self.uvmap_f_idx.to(device)
        uvmap_f_bary = self.uvmap_f_bary.to(device)
        uvmap_mask = self.uvmap_mask.to(device)

        mano = self.renderer.mano_model.mano
        faces = torch.tensor(mano.faces, device=device, dtype=torch.long)

        uv_vertex_id = faces[uvmap_f_idx]

        uv_vertex = torch.zeros(
            (batch_size, self.uvmap_size, self.uvmap_size, 3, 3),
            device=device, dtype=torch.float32
        )
        for k in range(3):
            uv_vertex[:, :, :, k, :] = deformed_vertices[:, uv_vertex_id[:, :, k], :]

        uv_vertex_interp = torch.einsum('hwk,bhwkn->bhwn', uvmap_f_bary, uv_vertex)

        uv_vertex_homo = torch.cat([uv_vertex_interp, torch.ones_like(uv_vertex_interp[:, :, :, :1])], dim=-1)

        uv_vertex_cam = torch.einsum('bij,bhwj->bhwi', w2c_cam, uv_vertex_homo)[:, :, :, :3]

        focal = img_size / 2.0
        vertices_img_x = focal * uv_vertex_cam[:, :, :, 0] / (uv_vertex_cam[:, :, :, 2] + 1e-8)
        vertices_img_y = focal * uv_vertex_cam[:, :, :, 1] / (uv_vertex_cam[:, :, :, 2] + 1e-8)

        vertices_img_x_norm = 2.0 * (vertices_img_x / img_size) - 1.0
        vertices_img_y_norm = 2.0 * (vertices_img_y / img_size) - 1.0
        vertices_img_norm = torch.stack([vertices_img_x_norm, vertices_img_y_norm], dim=-1)

        uv_features_sampled = torch.nn.functional.grid_sample(
            img_features,
            vertices_img_norm,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        mask = uvmap_mask.clone()[None, None, :, :].repeat(batch_size, 1, 1, 1).float()
        uv_features = uv_features_sampled * mask

        return uv_features

    def sample_uv_feature_at_points(self, uvmap_features, vert_uv):
        """
        Sample UV features at specified UV coordinates (for each point).

        Args:
            uvmap_features: [B, C, uvmap_size, uvmap_size] UV feature map
            vert_uv: [B, N, 2] UV coordinates for N points (normalized to [0, 1])

        Returns:
            point_features: [B, N, C] sampled features
        """

        batch_size, n_points, _ = vert_uv.shape
        device = vert_uv.device

        grid = vert_uv.clone()
        grid[..., 0] = 2.0 * vert_uv[..., 0] - 1.0
        grid[..., 1] = 2.0 * vert_uv[..., 1] - 1.0

        grid = grid[:, None, :, :].contiguous()

        sampled = torch.nn.functional.grid_sample(
            uvmap_features, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        sampled = sampled.squeeze(2).permute(0, 2, 1).contiguous()

        return sampled

    def get_uvmap_features_from_image(self, combined_feats, uv_map_dict):
        """
        Complete pipeline: image -> UV features -> per-point UV features.
        Process each batch sample separately since they may have different subjects/poses.

        IMPORTANT: This function expects combined_feats from the FIRST view only.
                   Multi-view handling (selecting first view) should be done in forward_latent_points.

        Args:
            combined_feats: [B, 3+C_dino, H, W] concatenated RGB+DINO features from FIRST view only
            uv_map_dict: dict with keys:
                - 'vert_uv': [N, 3] or list of [N, 3] - canonical 3D vertices forming UV mesh
                - 'face_uv': [M, 3] or list of [M, 3] - face indices
                - 'face_uv_xy': [M, 3, 2] or list of [M, 3, 2] - UV coordinates of face vertices
                - 'posed_points': [B, N, 3] or list of [N, 3] - posed vertices
                - 'cam': [4, 4] or [B, 4, 4] or list of [4, 4] - camera matrices

        Returns:
            dict with keys:
                'uvmap_features': [B, C, 256, 256] UV feature map (stacked across batch)
                'point_uv_features': [B, N_pts, C] sampled per-point features
        """
        batch_size = combined_feats.shape[0]
        device = combined_feats.device

        if uv_map_dict is None or not isinstance(uv_map_dict, dict):
            return {'uvmap_features': None, 'point_uv_features': None}

        vert_uv = uv_map_dict.get('vert_uv')
        face_uv = uv_map_dict.get('face_uv')
        face_uv_xy = uv_map_dict.get('face_uv_xy')
        posed_points = uv_map_dict.get('posed_points')
        cam = uv_map_dict.get('cam')

        if vert_uv is None or face_uv is None or face_uv_xy is None:
            return {'uvmap_features': None, 'point_uv_features': None}

        def to_batch_list(data, batch_size):
            """Convert various input formats to list of per-batch tensors"""
            if isinstance(data, list):
                return data
            elif isinstance(data, torch.Tensor):
                if data.ndim >= 2 and data.shape[0] == batch_size:
                    return [data[i] for i in range(batch_size)]
                else:
                    return [data for _ in range(batch_size)]
            else:
                return [data for _ in range(batch_size)]

        vert_uv_list = to_batch_list(vert_uv, batch_size)
        face_uv_list = to_batch_list(face_uv, batch_size)
        face_uv_xy_list = to_batch_list(face_uv_xy, batch_size)
        posed_points_list = to_batch_list(posed_points, batch_size)

        if isinstance(cam, list):
            cam_list = cam
        elif isinstance(cam, torch.Tensor):
            if cam.shape[0] == batch_size and cam.ndim == 3:
                cam_list = [cam[i] for i in range(batch_size)]
            else:
                cam_list = [cam for _ in range(batch_size)]
        else:
            cam_list = [torch.eye(4, device=device, dtype=torch.float32) for _ in range(batch_size)]

        uvmap_feats_list = []
        point_uv_feats_list = []
        point_uv_coords_list = []

        for b in range(batch_size):
            vert_uv_b = vert_uv_list[b].to(device)
            face_uv_b = face_uv_list[b].to(device).long()
            face_uv_xy_b = face_uv_xy_list[b].to(device)
            posed_points_b = posed_points_list[b].to(device)
            cam_b = cam_list[b].to(device)

            if posed_points_b.ndim == 2:
                posed_points_b = posed_points_b.unsqueeze(0)

            if cam_b.ndim == 2:
                cam_b = cam_b.unsqueeze(0)

            combined_feats_b = combined_feats[b:b+1]
            if self.uv_map_bias is not None:
                combined_feats_b = combined_feats_b + self.uv_map_bias

            uvmap_feats_b = self.convert_pixel_feature_to_uv(
                combined_feats_b, posed_points_b, cam_b, img_size=256
            )

            uvmap_feats_list.append(uvmap_feats_b.squeeze(0))

            pts_b = posed_points_b.squeeze(0)

            try:
                from livehand.input_encoder import get_uvd as gaussianhand_get_uvd
                point_uv_b, _, _ = gaussianhand_get_uvd(pts_b, vert_uv_b, face_uv_b, face_uv_xy_b)

                point_uv_b_expanded = point_uv_b.unsqueeze(0)
                point_features_b = self.sample_uv_feature_at_points(
                    uvmap_feats_b, point_uv_b_expanded
                )

                point_uv_feats_list.append(point_features_b.squeeze(0))
                point_uv_coords_list.append(point_uv_b)
            except Exception as e:
                print(f"Warning: get_uvd failed for batch {b}: {e}")
                point_uv_feats_list.append(None)

        uvmap_features = torch.stack(uvmap_feats_list, dim=0)

        if all(f is not None for f in point_uv_feats_list):
            point_uv_features = torch.stack(point_uv_feats_list, dim=0)
            point_uv_coords = torch.stack(point_uv_coords_list, dim=0) if len(point_uv_coords_list) == batch_size else None
        else:
            point_uv_features = None
            point_uv_coords = None

        return {
            'uvmap_features': uvmap_features,
            'point_uv_features': point_uv_features,
            'point_uv_coords': point_uv_coords,
        }

    def set_grad(self, model, iter, warmup_iter=1000):
        if iter < warmup_iter:
            for n, p in model.named_parameters():
                if 'part_path' in n or 'joint_path' in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

    def set_grad_iter(self, model, iteration, warmup_iter=1000):
        if iteration == 0:
            print(f"[Init] Warmup for {warmup_iter} iterations.")
        if iteration == warmup_iter:
            print(f"[Iter {iteration}] Unfreezing all modules...")

        if iteration == 0:
            for n, p in self.pcl_embed.named_parameters():
                p.requires_grad = n in self.base_params
        elif iteration == warmup_iter:
            for p in self.pcl_embed.parameters():
                p.requires_grad = True

    def obtain_params(self):
        try:
            if hasattr(self.renderer, 'lora_mlp') and self.renderer.lora_mlp is not None:
                for p in self.renderer.lora_mlp.parameters():
                    p.requires_grad = False
            if hasattr(self.renderer, 'lora_gs') and self.renderer.lora_gs is not None:
                for p in self.renderer.lora_gs.parameters():
                    p.requires_grad = False
        except Exception:
            pass

        no_decay_params = []
        ln_param_ids = set()
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    if p.requires_grad and id(p) not in ln_param_ids:
                        no_decay_params.append(p)
                        ln_param_ids.add(id(p))

        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith("bias") and id(param) not in ln_param_ids:
                no_decay_params.append(param)
                ln_param_ids.add(id(param))

        all_params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in all_params if id(p) not in ln_param_ids]

        opt_groups = [
            {
                "params": decay_params,
                "weight_decay": 0.05,
                "lr": 4e-4,
                "name": "decay",
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": 4e-4,
                "name": "no_decay",
            },
        ]

        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")
        print(f"Total Params: {len(no_decay_params) + len(decay_params)}")

        return opt_groups

    def save(self, iteration):
        print(f'============================= Save checkpoint of iter {iteration} to {self.checkpoint_path} =============================')
        pcl_embed_path = os.path.join(self.checkpoint_path, 'pcl_embed', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(pcl_embed_path), exist_ok=True)

        torch.save({

            'pcl_embed': self.pcl_embed.state_dict(),
        }, pcl_embed_path)

        motion_embed_path = os.path.join(self.checkpoint_path, 'motion_embed', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(motion_embed_path), exist_ok=True)

        torch.save({

            'motion_embed': self.motion_embed_mlp.state_dict(),
        }, motion_embed_path)

        transformer_path = os.path.join(self.checkpoint_path, 'transformer', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(transformer_path), exist_ok=True)

        torch.save({'transformer': self.transformer}, transformer_path, _use_new_zipfile_serialization=False)

        encoder_path = os.path.join(self.checkpoint_path, 'encoder', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

        torch.save({'encoder': self.encoder}, encoder_path, _use_new_zipfile_serialization=False)

        renderer_path = os.path.join(self.checkpoint_path, 'renderer', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(renderer_path), exist_ok=True)
        torch.save({

            'renderer': self.renderer.state_dict(),

        }, renderer_path)

        if self.renderer.neural_refiner is not None:
            refiner_path = os.path.join(self.checkpoint_path, 'refiner', 'iteration_' + str(iteration), 'ckpt.pth')
            os.makedirs(os.path.dirname(refiner_path), exist_ok=True)
            torch.save({

                'neural_refiner': self.renderer.neural_refiner.state_dict(),
            }, refiner_path)

    def save_checkpoint(self, iteration=None, is_latest=False):
        """
        保存模型检查点。
        - iteration: 迭代次数（int）
        - is_latest: 是否保存为 latest.ckpt
        """
        os.makedirs(self.checkpoint_path, exist_ok=True)

        ckpt = {
            "iteration": iteration,
            "pcl_embed": self.pcl_embed.state_dict(),

            "adapter": self.adapter.state_dict(),
            "aggregator": self.aggregator.state_dict(),
            "motion_embed_mlp": self.motion_embed_mlp.state_dict(),
            "transformer": self.transformer,

            "encoder": self.encoder,
            "renderer": self.renderer.state_dict(),
        }

        if hasattr(self, "vertex_global_mapping") and self.vertex_global_mapping is not None:
            ckpt["vertex_global_mapping"] = self.vertex_global_mapping.state_dict()

        if hasattr(self, "color_shift") and self.color_shift is not None:
            ckpt["color_shift"] = self.color_shift.data.detach().cpu()
        if hasattr(self, "color_scale") and self.color_scale is not None:
            ckpt["color_scale"] = self.color_scale.data.detach().cpu()
        if hasattr(self, "color_bias_lowres") and self.color_bias_lowres is not None:
            ckpt["color_bias_lowres"] = self.color_bias_lowres.data.detach().cpu()

        try:
            ckpt["model_state"] = self.state_dict()
        except Exception:
            pass

        if self.renderer.neural_refiner is not None:
            ckpt["neural_refiner"] = self.renderer.neural_refiner.state_dict()

        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                ckpt["optimizer"] = self.optimizer.state_dict()
        except Exception:
            pass

        try:
            if hasattr(self, "scheduler") and self.scheduler is not None:
                ckpt["scheduler"] = self.scheduler.state_dict()
        except Exception:
            pass

        filename = "latest.ckpt" if is_latest else f"iteration_{iteration}.ckpt"
        path = os.path.join(self.checkpoint_path, filename)
        torch.save(ckpt, path)

        print(f"\033[92m[Checkpoint saved → {path}]\033[0m")

    def build_transformer(
        self,
        transformer_type,
        transformer_layers,
        transformer_heads,
        transformer_dim,
        encoder_feat_dim,
    ):
        return TransformerDecoder(
            block_type=transformer_type,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=None,

        )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight

    def hyper_step(self, step):
        pass

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        if encoder_type != "dinov2_fusion":
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper

        logger.info("Using Dinov2FusionWrapper as the encoder")
        return Dinov2FusionWrapper

    @torch.no_grad()
    def uv_pixels_to_feat_norm(self, uv_pixels, W_img, H_img, W_feat, H_feat):
        x = uv_pixels[..., 0]
        y = uv_pixels[..., 1]
        x_feat = x * (W_feat / float(W_img))
        y_feat = y * (H_feat / float(H_img))
        x_norm = (x_feat / (W_feat - 1.0)) * 2.0 - 1.0
        y_norm = (y_feat / (H_feat - 1.0)) * 2.0 - 1.0
        return torch.stack([x_norm, y_norm], dim=-1)

    def make_patch_grid(self, patch_size, device):
        P = patch_size
        coords = np.linspace(-(P-1)/2.0, (P-1)/2.0, P)
        gx, gy = np.meshgrid(coords, coords)
        grid = np.stack([gx, gy], axis=-1).reshape(-1, 2)
        return torch.tensor(grid, dtype=torch.float32, device=device)

    def sample_local_patches_light(self, feat_map, uv_pixels, W_img, H_img,
                                patch_size=8, proj=None,
                                mask_img=None, batch_nv=512):
        """
        显存友好的局部采样实现。
        不会展开 feat_map 到 (B*Nv,...)
        """
        B, C, Hf, Wf = feat_map.shape
        device = feat_map.device
        Nv = uv_pixels.shape[1]
        P = patch_size
        K = P * P

        base_grid_px = self.make_patch_grid(patch_size, device)
        x_offsets_norm = base_grid_px[:, 0] / (Wf - 1.0) * 2.0
        y_offsets_norm = base_grid_px[:, 1] / (Hf - 1.0) * 2.0
        offsets_norm = torch.stack([x_offsets_norm, y_offsets_norm], dim=-1)

        patches_list, patches_proj_list, mask_conf_list = [], [], []

        for start in range(0, Nv, batch_nv):
            end = min(start + batch_nv, Nv)
            uv_batch = uv_pixels[:, start:end, :]
            nb = end - start

            uv_feat_norm = self.uv_pixels_to_feat_norm(uv_batch, W_img, H_img, Wf, Hf)
            grids = uv_feat_norm.unsqueeze(2) + offsets_norm.view(1, 1, K, 2)
            grids = grids.view(B, nb, P, P, 2)

            batch_patches = []
            for b in range(B):
                grid_b = grids[b]
                feat_b = feat_map[b].unsqueeze(0)
                patches_b = F.grid_sample(
                    feat_b.expand(nb, -1, -1, -1),
                    grid_b,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                )
                batch_patches.append(patches_b)
            patches = torch.stack(batch_patches, dim=0)

            patches_flat = patches.view(B, nb, -1)
            patches_proj = self.proj(patches_flat.view(B*nb, -1)).view(B, nb, -1)

            mask_conf = None
            if mask_img is not None:
                mask_feat = F.interpolate(mask_img, size=(Hf, Wf), mode='nearest')
                mask_batch = []
                for b in range(B):
                    grid_b = grids[b]
                    mask_b = mask_feat[b].unsqueeze(0).expand(nb, -1, -1, -1)
                    mask_p = F.grid_sample(
                        mask_b.float(), grid_b,
                        mode='bilinear', padding_mode='zeros',
                        align_corners=True
                    )
                    mask_batch.append(mask_p)
                mask_conf = torch.stack(mask_batch, dim=0).view(B, nb, -1)

            patches_list.append(patches)
            patches_proj_list.append(patches_proj)
            if mask_conf is not None:
                mask_conf_list.append(mask_conf)

            torch.cuda.empty_cache()

        patches = torch.cat(patches_list, dim=1)
        patches_proj = torch.cat(patches_proj_list, dim=1)
        mask_conf = torch.cat(mask_conf_list, dim=1) if mask_conf_list else None
        return patches, patches_proj, mask_conf

    def _attach_uv_feature_maps(self, batches, query_points, point_features, texture_hw=(256, 512)):
        if query_points is None or point_features is None or not batches:
            return
        if gaussianhand_get_uvd is None:
            if not self._warned_missing_uv_mapper:
                logger.warning(
                    "UV feature projection skipped: unable to import livehand.input_encoder.get_uvd"
                )
                self._warned_missing_uv_mapper = True
            for batch in batches:
                if isinstance(batch, dict):
                    batch['uv_feature_map'] = None
            return

        tex_h, tex_w = texture_hw
        max_samples = min(len(batches), query_points.shape[0], point_features.shape[0])
        for idx, batch in enumerate(batches):
            if not isinstance(batch, dict):
                continue
            if idx >= max_samples:
                batch['uv_feature_map'] = None
                continue
            uv_template = batch.get('uv_map')
            if not (isinstance(uv_template, dict) and uv_template):
                batch['uv_feature_map'] = None
                continue

            feats = point_features[idx]
            pts = query_points[idx].to(feats.device)
            vert_uv = uv_template['vert_uv'].to(feats.device)
            face_uv = uv_template['face_uv'].to(feats.device).long()
            face_uv_xy = uv_template['face_uv_xy'].to(feats.device)

            sampled_uv, signed_dist, _ = gaussianhand_get_uvd(pts, vert_uv, face_uv, face_uv_xy)
            uv_norm = sampled_uv.clone()
            uv_norm[..., 0] = 2.0 * (uv_norm[..., 0] / 1.0) - 1.0
            uv_norm[..., 1] = 2.0 * (uv_norm[..., 1] / 0.5) - 1.0

            uv_texture = self._scatter_uv_features(feats, uv_norm, tex_h, tex_w)
            batch['uv_feature_map'] = {
                'coords': uv_norm.detach(),
                'signed_dist': signed_dist.detach(),
                'feature': uv_texture.detach(),
            }

    def _sample_uv_features_from_token_maps(self, batches, query_points, token_feature_maps):
        """For each batch element, sample per-point features from a UV token map.

        Args:
            batches: list[dict], each may contain 'uv_map' built in infer_single_view.
            query_points: [B, N, 3] points in canonical space.
            token_feature_maps: [B, C, H, W] UV token feature maps (e.g., 2,1024,32,32).

        Side effect:
            For each valid batch i, writes a tensor of shape [N, C] to
            batch['uv_point_features'] containing per-point features sampled
            from the corresponding UV token map via bilinear interpolation.
        """
        if query_points is None or token_feature_maps is None or not batches:
            return None

        if gaussianhand_get_uvd is None:
            if not self._warned_missing_uv_mapper:
                logger.warning(
                    "UV point sampling skipped: unable to import livehand.input_encoder.get_uvd"
                )
                self._warned_missing_uv_mapper = True
            for batch in batches:
                if isinstance(batch, dict):
                    batch['uv_point_features'] = None
            return None

        B, C, H, W = token_feature_maps.shape
        max_samples = min(len(batches), query_points.shape[0], B)
        per_batch_feats = []

        for idx, batch in enumerate(batches):
            if not isinstance(batch, dict) or idx >= max_samples:
                if isinstance(batch, dict):
                    batch['uv_point_features'] = None
                continue

            uv_template = batch.get('uv_map')
            if not (isinstance(uv_template, dict) and uv_template):
                batch['uv_point_features'] = None
                continue

            pts = query_points[idx].to(token_feature_maps.device)
            vert_uv = uv_template['vert_uv'].to(token_feature_maps.device)
            face_uv = uv_template['face_uv'].to(token_feature_maps.device).long()
            face_uv_xy = uv_template['face_uv_xy'].to(token_feature_maps.device)

            sampled_uv, signed_dist, _ = gaussianhand_get_uvd(pts, vert_uv, face_uv, face_uv_xy)

            uv_norm = sampled_uv.clone()
            uv_norm[..., 0] = 2.0 * (uv_norm[..., 0] / 1.0) - 1.0
            uv_norm[..., 1] = 2.0 * (uv_norm[..., 1] / 0.5) - 1.0

            N = uv_norm.shape[0]
            grid = uv_norm.view(1, N, 1, 2)

            token_map_b = token_feature_maps[idx:idx + 1]

            sampled = F.grid_sample(
                token_map_b,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )

            feats = sampled.view(C, N).transpose(0, 1).contiguous()

            batch['uv_point_features'] = feats.detach()
            per_batch_feats.append(feats)

        per_batch_feats = torch.stack(per_batch_feats, dim=0)

        return per_batch_feats

    @staticmethod
    def _scatter_uv_features(point_features, uv_coords_norm, tex_h, tex_w):
        if point_features.numel() == 0:
            return torch.zeros(0, tex_h, tex_w, device=point_features.device)

        coords = uv_coords_norm.to(point_features.device).to(point_features.dtype)
        coords = coords.clamp_(-1.0, 1.0)
        u = (coords[:, 0] + 1.0) * 0.5 * (tex_w - 1)
        v = (coords[:, 1] + 1.0) * 0.5 * (tex_h - 1)
        valid = torch.isfinite(u) & torch.isfinite(v)
        if not torch.any(valid):
            return torch.zeros(point_features.shape[1], tex_h, tex_w, device=point_features.device)

        u_idx = u[valid].round().long().clamp_(0, tex_w - 1)
        v_idx = v[valid].round().long().clamp_(0, tex_h - 1)
        feats = point_features[valid]

        flat_idx = v_idx * tex_w + u_idx
        feat_dim = feats.shape[1]
        feat_map = torch.zeros(feat_dim, tex_h * tex_w, device=point_features.device)
        feat_map.index_add_(1, flat_idx, feats.t())

        weight_map = torch.zeros(1, tex_h * tex_w, device=point_features.device, dtype=point_features.dtype)
        weight_src = torch.ones(1, flat_idx.shape[0], device=point_features.device, dtype=point_features.dtype)
        weight_map.index_add_(1, flat_idx, weight_src)

        feat_map = feat_map / weight_map.clamp_min(1.0)
        return feat_map.view(feat_dim, tex_h, tex_w)

    def sample_local_patches_light_new(feat_pyramid,
                                proj_uv,
                                proj_z=None,
                                patch_size=5,
                                scales=None,
                                device='cuda'):
        """
        Return: patches [B, N, k, C]  (k = patch_size^2 * len(scales))
        feat_pyramid: list of feature maps from FPN (B,C,Hi,Wi)
        proj_uv: pixel coordinates (not normalized), shape [B,N,2]
        proj_z: optional depth for masking [B,N]
        """
        B, N, _ = proj_uv.shape
        if scales is None:
            scales = list(range(len(feat_pyramid)))

        patches_per_scale = []
        for s in scales:
            feat = feat_pyramid[s]
            Bf, C, H, W = feat.shape

            u = proj_uv[..., 0] / (W - 1) * 2 - 1
            v = proj_uv[..., 1] / (H - 1) * 2 - 1
            grid = torch.stack([u, v], dim=-1)

            half = (patch_size // 2)
            xs = torch.linspace(-half, half, steps=patch_size, device=device) / (W - 1) * 2
            ys = torch.linspace(-half, half, steps=patch_size, device=device) / (H - 1) * 2
            delta = torch.stack(torch.meshgrid(xs, ys), dim=-1).reshape(-1,2)

            grid_exp = grid.unsqueeze(2) + delta.unsqueeze(0).unsqueeze(0)
            grid_exp = grid_exp.view(B, N*patch_size*patch_size, 2)

            grid_for_sample = grid_exp.view(B, N*patch_size*patch_size, 1, 2)

            sampled = F.grid_sample(feat, grid_for_sample, mode='bilinear', padding_mode='zeros', align_corners=True)

            sampled = sampled.view(B, C, N, patch_size*patch_size).permute(0,2,3,1)
            patches_per_scale.append(sampled)

        patches = torch.cat(patches_per_scale, dim=2)

        return patches

    def sample_local_patches_from_pyramid_ori(self, feat_pyramid, proj_uv_full, patch_sizes, full_image_wh=(256,256)):
        """
        feat_pyramid: list of [B, C, Hs, Ws] (e.g., P2..P5), order corresponds to patch_sizes list
        proj_uv_full: [B, N, 2] pixel coords in full-image space (x,y)
        patch_sizes: list of int per pyramid level (odd), e.g., [3,3,5,5]
        full_image_wh: tuple (W_full, H_full)
        Return: patches [B, N, K_total, C] and coords tensor [B, N, K_total, 2] (normalized per-level used)
        """
        B, N, _ = proj_uv_full.shape
        device = proj_uv_full.device
        patches_scales = []
        coords_scales = []

        for P, psize in zip(feat_pyramid, patch_sizes):
            Bf, C, H, W = P.shape

            W_full, H_full = full_image_wh

            u_feat_x = proj_uv_full[..., 0] * (W / W_full)
            u_feat_y = proj_uv_full[..., 1] * (H / H_full)

            nx = (u_feat_x / (W - 1)) * 2 - 1
            ny = (u_feat_y / (H - 1)) * 2 - 1
            center = torch.stack([nx, ny], dim=-1)

            half = psize // 2
            xs = (torch.arange(-half, half+1, device=device).float() / (W - 1)) * 2
            ys = (torch.arange(-half, half+1, device=device).float() / (H - 1)) * 2

            dx, dy = torch.meshgrid(xs, ys, indexing='xy')
            delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)

            grid = (center.unsqueeze(2) + delta.view(1,1,-1,2))

            grid_rs = grid.view(B, N * grid.shape[2], 2).unsqueeze(2)

            sampled = F.grid_sample(P, grid_rs, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled = sampled.view(B, C, N, grid.shape[2]).permute(0,2,3,1)
            patches_scales.append(sampled)
            coords_scales.append(grid)

        patches = torch.cat(patches_scales, dim=2)
        coords = torch.cat(coords_scales, dim=2)
        return patches, coords

    def sample_local_patches_from_pyramid(self,
        feat_pyramid,
        proj_uv_full,
        patch_sizes,
        full_image_wh=(256, 256),
        align_corners=True,
    ):
        """
        Returns:
        patches: [B, N, K_total, C]   (C is channel of each level — assumes same C across levels, or you handle projection later)
        coords_norm: [B, N, K_total, 2] normalized coords in [-1,1]
        valid_mask: [B, N, K_total]  boolean mask (True means sample was inside feature map before clamping)
        Notes:
        - proj_uv_full should be float (don't round) for subpixel sampling.
        - If feat_pyramid is a single [B,C,H,W] tensor, it will be treated as a single level.
        - patch_sizes can be single int (applied to all levels) or list with same len as levels.
        - This function **does not** reshape tokens; it expects feature maps [B,C,H,W].
        """

        if isinstance(feat_pyramid, (list, tuple)):
            feat_list = list(feat_pyramid)
        else:
            feat_list = [feat_pyramid]

        B, N, _ = proj_uv_full.shape
        W_full, H_full = full_image_wh

        if isinstance(patch_sizes, int):
            patch_sizes = [patch_sizes] * len(feat_list)
        assert len(patch_sizes) == len(feat_list)

        device = proj_uv_full.device
        patches_per_level = []
        coords_per_level = []
        masks_per_level = []
        C_out = feat_list[0].shape[1]

        for lvl, (P, psize) in enumerate(zip(feat_list, patch_sizes)):
            assert P.dim() == 4, "Each level must be [B,C,H,W]"
            Bf, C, Hf, Wf = P.shape
            assert Bf == B, "Batch mismatch between features and proj_uv"

            u_feat_x = proj_uv_full[..., 0] * (Wf / float(W_full))
            u_feat_y = proj_uv_full[..., 1] * (Hf / float(H_full))

            if align_corners:
                nx = (u_feat_x / (Wf - 1)) * 2.0 - 1.0
                ny = (u_feat_y / (Hf - 1)) * 2.0 - 1.0

                xs = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() / (Wf - 1)) * 2.0
                ys = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() / (Hf - 1)) * 2.0
            else:
                nx = ((u_feat_x + 0.5) / Wf) * 2.0 - 1.0
                ny = ((u_feat_y + 0.5) / Hf) * 2.0 - 1.0
                xs = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() + 0.0) / Wf * 2.0
                ys = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() + 0.0) / Hf * 2.0

            dx, dy = torch.meshgrid(xs, ys, indexing='xy')
            delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)
            k = delta.shape[0]

            center = torch.stack([nx, ny], dim=-1)

            grid = center.unsqueeze(2) + delta.view(1, 1, k, 2)

            valid_mask_lvl = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)

            grid_clamped = grid.clamp(-1.0, 1.0)

            grid_rs = grid_clamped.view(B, N * k, 2).unsqueeze(2)

            sampled = F.grid_sample(P, grid_rs, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
            sampled = sampled.view(B, C, N, k).permute(0, 2, 3, 1).contiguous()
            patches_per_level.append(sampled)
            coords_per_level.append(grid)
            masks_per_level.append(valid_mask_lvl)

        patches = torch.cat(patches_per_level, dim=2)
        coords_norm = torch.cat(coords_per_level, dim=2)
        valid_mask = torch.cat(masks_per_level, dim=2)

        return patches, coords_norm, valid_mask

    def compute_proj_uv_feat_from_original(self, verts_xy_orig, H0, W0, resolution, patch_size):
        """
        verts_xy_orig: tensor [B, N, 2] in original image pixel coords (x horizontal, y vertical) with x∈[0,W0-1], y∈[0,H0-1]
        returns: proj_uv_feat: tensor [B, N, 2] in feature-grid coords (float) where x∈[0, Wf-1], y∈[0, Hf-1]
        """
        B, N, _ = verts_xy_orig.shape
        device = verts_xy_orig.device

        max_size = max(H0, W0)
        pad_left = (max_size - W0) // 2
        pad_top  = (max_size - H0) // 2

        x0 = verts_xy_orig[..., 0]
        y0 = verts_xy_orig[..., 1]
        x_pad = x0 + pad_left
        y_pad = y0 + pad_top

        scale = float(resolution) / float(max_size)
        x_resized = x_pad * scale
        y_resized = y_pad * scale

        Wf = resolution // patch_size
        Hf = resolution // patch_size

        u_feat_x = x_resized / float(patch_size)
        u_feat_y = y_resized / float(patch_size)

        proj_uv_feat = torch.stack([u_feat_x, u_feat_y], dim=-1)
        return proj_uv_feat

    def sample_feats_from_feat32_to_256(self,
        feat32,
        nail_img,
        psize=1,
        chunk_size=2048,
        align_corners=True,
        padding_mode='zeros',
        device=None
    ):
        """
        Upsample feat32 -> 256 and sample per-point features at nail_img coordinates.
        Returns:
            sampled_feats: [B, N, C] if psize==1, else [B, N, K, C] where K = psize*psize
            feat256: upsampled feature map [B, C, 256, 256]
        Notes:
            - nail_img can be numpy or tensor. If numpy with shape [N,2], it will be treated as single batch.
            - nail_img coords must be float, not rounded to ints (to preserve subpixel sampling).
            - This does NOT perform any Dinov2 pad/resize mapping — it assumes coords are in 0..255 space.
        """
        assert feat32.dim() == 4 and feat32.shape[2] == 32 and feat32.shape[3] == 32, "feat32 must be [B,C,32,32]"
        if device is None:
            device = feat32.device

        if isinstance(nail_img, np.ndarray):
            nail_t = torch.from_numpy(nail_img).float().to(device)
        else:
            nail_t = nail_img.to(device)
        if nail_t.dim() == 2:
            nail_t = nail_t.unsqueeze(0)
        B = feat32.shape[0]
        C = feat32.shape[1]
        Bn = nail_t.shape[0]
        if Bn != B:
            if Bn == 1 and B > 1:
                nail_t = nail_t.expand(B, -1, -1).contiguous()
            else:
                raise ValueError(f"batch mismatch: feat32 batch {B} vs nail_img batch {Bn}")

        feat256 = F.interpolate(feat32, size=(256,256), mode='bilinear', align_corners=align_corners)

        Wt, Ht = 256, 256

        if align_corners:
            nx = (nail_t[..., 0] / (Wt - 1)) * 2.0 - 1.0
            ny = (nail_t[..., 1] / (Ht - 1)) * 2.0 - 1.0
        else:
            nx = ((nail_t[..., 0] + 0.5) / Wt) * 2.0 - 1.0
            ny = ((nail_t[..., 1] + 0.5) / Ht) * 2.0 - 1.0
        centers = torch.stack([nx, ny], dim=-1)

        if psize <= 1:
            K = 1
            delta = None
        else:
            assert psize % 2 == 1, "psize must be odd"
            half = psize // 2
            if align_corners:
                xs = (torch.arange(-half, half+1, device=device).float() / (Wt - 1)) * 2.0
                ys = (torch.arange(-half, half+1, device=device).float() / (Ht - 1)) * 2.0
            else:
                xs = (torch.arange(-half, half+1, device=device).float() / Wt) * 2.0
                ys = (torch.arange(-half, half+1, device=device).float() / Ht) * 2.0
            dx, dy = torch.meshgrid(xs, ys, indexing='xy')
            delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)
            K = delta.shape[0]

        sampled_per_batch = []
        for b in range(B):
            feat_b = feat256[b:b+1]
            centers_b = centers[b]
            N = centers_b.shape[0]
            out_chunks = []
            for i in range(0, N, chunk_size):
                ci = centers_b[i:i+chunk_size]
                M = ci.shape[0]
                if psize <= 1:
                    grid = ci.view(1, M, 1, 2)
                    sampled = F.grid_sample(feat_b, grid, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)

                    sampled = sampled.view(1, C, M).permute(0,2,1).contiguous()
                    out_chunks.append(sampled)
                else:
                    grid_pts = (ci.unsqueeze(1) + delta.unsqueeze(0).to(ci.device))

                    grid_rs = grid_pts.clamp(-1.0, 1.0).view(1, M*K, 1, 2)
                    sampled = F.grid_sample(feat_b, grid_rs, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)

                    sampled = sampled.view(1, C, M, K).permute(0,2,3,1).contiguous()
                    out_chunks.append(sampled)

            if len(out_chunks) == 0:
                if psize <= 1:
                    out_b = torch.zeros((1, 0, C), device=feat32.device)
                else:
                    out_b = torch.zeros((1, 0, K, C), device=feat32.device)
            else:
                out_b = torch.cat(out_chunks, dim=1)
            sampled_per_batch.append(out_b)

        sampled_feats = torch.cat(sampled_per_batch, dim=0)

        return sampled_feats, feat256

    def forward_transformer(
        self, image_feats, camera_embeddings, query_points, nail_mask=None, nail_mask_3d=None,
        proj_xy=None, p_vis=None, feat_hw=None, motion_embed=None, global_four=None, point_pos=None, posed_pos=None,
    ):
        """
        Applies forward transformation to the input features.
        Args:
            image_feats (torch.Tensor): Input image features. Shape [B, C, H, W].
            camera_embeddings (torch.Tensor): Camera embeddings. Shape [B, D].
            query_points (torch.Tensor): Query points. Shape [B, L, D].
            motion embed(torch.Tensor): Query points. Shape [B, L, D].
        Returns:
            torch.Tensor: Transformed features. Shape [B, L, D].
        """

        B = image_feats.shape[0]

        x = self.transformer(
            query_points,
            cond=image_feats,
            mod=camera_embeddings,
            temb=motion_embed,
            global_four=global_four,
            nail_mask=None,
            nail_mask_3d=None,
            proj_xy=proj_xy,
            p_vis=p_vis,
            point_pos=point_pos,
            posed_pos=posed_pos,

        )
        return x

    def forward_moitonembed(self, motion_tokens):
        motion_tokens = self.motion_embed_mlp(motion_tokens).squeeze(1)

        return motion_tokens

    def forward_encode_image(self, image):
        """
        Encode image and construct combined feature map (RGB + DINO) for UV mapping.
        Returns: image_feats (feature map), feature (intermediate), cls_token, combined_feats
        """
        encoder_out = self.encoder(image)
        return encoder_out

    @torch.compile
    def forward_latent_points(self, vis_mask, nail_image, uv_map_dict, image, vis_msk=None, camera=None, query_points=None, posed_points=None):
        """
        Forward pass of the latent points generation.
        Args:
            image (torch.Tensor): Input image tensor of shape [B, C_img, H_img, W_img] or [B, N_views, C_img, H_img, W_img].
                                 If multiple views provided, only the first view will be used.
            camera (torch.Tensor): Camera tensor of shape [B, D_cam_raw].
            query_points (torch.Tensor, optional): Query points tensor. for example, smplx surface points, Defaults to None.
        Returns:
            torch.Tensor: Generated tokens tensor.
            torch.Tensor: Encoded image features tensor.
        """

        if image.ndim == 5:
            image = image[:, 0]
        elif image.ndim == 4 and image.shape[-1] == 3 and image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        elif image.ndim == 3:
            image = image.unsqueeze(1)

        B = image.shape[0]

        image_fine_feats, feature, cls_tokens =self.forward_encode_image(image)
        motion_tokens = cls_tokens
        cls_four = cls_tokens
        motion_tokens = self.forward_moitonembed(cls_tokens.to('cuda:0'))

        merge_tokens = image_fine_feats

        query_point = query_points
        query_points = self.pcl_embed(query_points)

        patches, coords_norm, valid_mask = self.sample_local_patches_from_pyramid(feature, nail_image, patch_sizes=5)

        point_feats = self.aggregator(patches, coords_norm, valid_mask)

        merge_tokens = self.adapter(point_feats)

        tokens = self.forward_transformer(

            merge_tokens, camera_embeddings=None, query_points=query_points, nail_mask=None, nail_mask_3d=None,
            proj_xy=nail_image, p_vis=vis_mask, feat_hw=None, motion_embed=motion_tokens, global_four=cls_four, point_pos=query_point, posed_pos=posed_points

        ).to('cuda')

        point_uv_feats = {}

        if point_uv_feats.get('point_uv_features') is not None:
            uv_feats = point_uv_feats['point_uv_features']

            uv_feats_norm = uv_feats / (torch.norm(uv_feats, dim=-1, keepdim=True) + 1e-8)

            tokens = tokens + uv_feats

        color_bias, opacity_bias = None, None
        if point_uv_feats.get('point_uv_coords') is not None:
            uv_coords = point_uv_feats['point_uv_coords']

            def _sample_bias_map(bias_map, coords):
                grid = coords.clone()
                grid[..., 0] = grid[..., 0] * 2.0 - 1.0
                grid[..., 1] = grid[..., 1] * 2.0 - 1.0
                grid = grid[:, None, :, :].contiguous()
                sampled = torch.nn.functional.grid_sample(
                    bias_map, grid,
                    mode='bilinear', padding_mode='border', align_corners=False
                )
                sampled = sampled.squeeze(2).permute(0, 2, 1).contiguous()
                return sampled

            color_bias_base, opacity_bias_base = None, None
            if self.color_b_map is not None:
                color_bias_base = _sample_bias_map(self.color_b_map, uv_coords)
            if self.opacity_b_map is not None:
                opacity_bias_base = _sample_bias_map(self.opacity_b_map, uv_coords)

            color_bias_refined, opacity_bias_refined = None, None

            if (
                hasattr(self, "bias_style_unet")
                and self.bias_style_unet is not None
                and self.color_b_map is not None
                and self.opacity_b_map is not None
            ):
                bias_input = torch.cat([self.color_b_map, self.opacity_b_map], dim=1).to(tokens.device)

                if bias_input.shape[-1] != self.uvmap_size or bias_input.shape[-2] != self.uvmap_size:
                    bias_input = torch.nn.functional.interpolate(
                        bias_input,
                        size=(self.uvmap_size, self.uvmap_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                style_code = motion_tokens

                bias_out = self.bias_style_unet(bias_input, extra_style=style_code)
                color_b_map_refined = bias_out[:, :3]
                opacity_b_map_refined = bias_out[:, 3:4]

                color_bias_refined = _sample_bias_map(color_b_map_refined, uv_coords)
                opacity_bias_refined = _sample_bias_map(opacity_b_map_refined, uv_coords)

            color_bias_refined, opacity_bias_refined = None, None
            if color_bias_base is not None:
                if color_bias_refined is None:
                    color_bias = color_bias_base
                else:
                    if vis_mask is not None:
                        vis = vis_mask.float()
                        if vis.dim() == 1:
                            vis = vis.unsqueeze(0)
                        occ_w = (1.0 - vis).unsqueeze(-1)

                        occ_w = occ_w.to(color_bias_base.device)
                        color_bias = color_bias_base * (1.0 - occ_w) + color_bias_refined * occ_w
                    else:
                        color_bias = color_bias_refined

            if opacity_bias_base is not None:
                if opacity_bias_refined is None:
                    opacity_bias = opacity_bias_base
                else:
                    if vis_mask is not None:
                        vis = vis_mask.float()
                        if vis.dim() == 1:
                            vis = vis.unsqueeze(0)
                        occ_w = (1.0 - vis).unsqueeze(-1)
                        occ_w = occ_w.to(opacity_bias_base.device)
                        opacity_bias = opacity_bias_base * (1.0 - occ_w) + opacity_bias_refined * occ_w
                    else:
                        opacity_bias = opacity_bias_refined

        return tokens, motion_tokens, {"color_bias": color_bias, "opacity_bias": opacity_bias}

    def forward(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        smplx_params,
        **kwargs,
    ):
        assert (
            image.shape[0] == render_c2ws.shape[0]
        ), "Batch size mismatch for image and render_c2ws"
        assert (
            image.shape[0] == render_bg_colors.shape[0]
        ), "Batch size mismatch for image and render_bg_colors"
        assert (
            image.shape[0] == smplx_params["betas"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert (
            image.shape[0] == smplx_params["body_pose"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert len(smplx_params["betas"].shape) == 2

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )
        query_points = None
        if self.latent_query_points_type.startswith("e2e_smplx"):
            query_points, smplx_params = self.renderer.get_query_points(
                smplx_params, device=image.device
            )

        latent_points, image_feats, bias_dict = self.forward_latent_points(
            image[:, 0], camera=None, query_points=query_points
        )

        render_results = self.renderer(
            gs_hidden_features=latent_points,
            query_points=query_points,
            smplx_data=smplx_params,
            c2w=render_c2ws,
            intrinsic=render_intrs,
            height=render_h,
            width=render_w,
            background_color=render_bg_colors,
            additional_features={"image_feats": image_feats, "image": image[:, 0]},
            color_bias=bias_dict.get("color_bias"),
            opacity_bias=bias_dict.get("opacity_bias"),
            df_data=kwargs["df_data"],
        )

        N, M = render_c2ws.shape[:2]
        assert (
            render_results["comp_rgb"].shape[0] == N
        ), "Batch size mismatch for render_results"
        assert (
            render_results["comp_rgb"].shape[1] == M
        ), "Number of rendered views should be consistent with render_cameras"

        gs_attrs_list = render_results.pop("gs_attr")

        offset_list = []
        scaling_list = []
        for gs_attrs in gs_attrs_list:
            offset_list.append(gs_attrs.offset_xyz)
            scaling_list.append(gs_attrs.scaling)
        offset_output = torch.stack(offset_list)
        scaling_output = torch.stack(scaling_list)

        return {
            "latent_points": latent_points,
            "offset_output": offset_output,
            "scaling_output": scaling_output,
            **render_results,
        }

    def hyper_step(self, step):
        self.renderer.hyper_step(step)

    def infer_single_view(
        self,
        vis_prob,
        vis_masks,
        nail_image,
        nail_mask,
        verts_cam,
        image,

        batches,
        cano_pts,
        render_bg_colors,

    ):
        query_points = cano_pts

        def _select_first_view(tensor: Optional[torch.Tensor]):
            if torch.is_tensor(tensor) and tensor.ndim > 2:
                return tensor[0]
            return tensor

        for batch in batches:
            vert_uv = batch.get('vert_uv') if isinstance(batch, dict) else None
            face_uv = batch.get('face_uv') if isinstance(batch, dict) else None
            face_uv_xy = batch.get('face_uv_xy') if isinstance(batch, dict) else None
            posed_points = batch.get('world_vertex') if isinstance(batch, dict) else None
            cam = batch.get('full_proj_transform') if isinstance(batch, dict) else None
            if not (torch.is_tensor(vert_uv) and torch.is_tensor(face_uv) and torch.is_tensor(face_uv_xy)):
                if isinstance(batch, dict):
                    batch['uv_map'] = None
                continue
            if isinstance(batch, dict):
                batch['uv_map'] = {
                    'vert_uv': _select_first_view(vert_uv),
                    'face_uv': _select_first_view(face_uv),
                    'face_uv_xy': _select_first_view(face_uv_xy),
                    'posed_points': _select_first_view(posed_points),
                    'cam': _select_first_view(cam),
                }

        vis_msk = None

        latent_points, global_texture_feature, bias_dict = self.forward_latent_points(
            vis_prob, nail_image, batch['uv_map'], image.permute(0, 3, 1, 2), vis_msk=vis_masks, camera=None, query_points=query_points)

        gs_model_list, gs_densify_list, query_points = self.renderer.forward_gs(
            gs_hidden_features=latent_points.to('cuda'),
            query_points=query_points,
            global_feature=global_texture_feature,
            batches=batches,
            verts_cam=verts_cam,
            nail_image=nail_image,
            color_bias=bias_dict.get("color_bias"),
            opacity_bias=bias_dict.get("opacity_bias"),

        )

        render_res_list = []
        for i in range(len(batches)):
            num_views = batches[i]['original_image'].shape[0]
            res = batches[i]['original_image'].shape[1]
            smplx_params={
                k: v.to('cuda') for k, v in batches[i]['smpl_param'].items()
            }

            for view_idx in range(num_views):
                render_res = self.renderer.forward_animate_gs(
                    gs_model_list[i],
                    gs_densify_list[i],
                    query_points[i],

                    self.renderer.get_single_view_cam(batches[i], view_idx),
                    self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                    batches[i]['width'][0],
                    batches[i]['height'][0],
                    render_bg_colors,
                )
                render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                if k == 'offset' or k == 'scaling' or k == 'shs':
                    selected = [v[0]]
                    out[k] = torch.cat(selected, dim=0)
                else:
                    out[k] = torch.concat(v, dim=1)
                    if k in ["comp_rgb", "comp_mask", "comp_depth", "comp_obj"]:
                        out[k] = out[k][0].permute(

                            0, 2, 3, 1
                        )
            else:
                out[k] = v

        out['vis_masks'] = vis_masks

        return out

class VisibilityBiasNet(nn.Module):
    """
    Learnable visibility-to-bias module.

    输入（per-point）:
      - p_vis: 可见度 [0,1]
      - depth_res (可选): 归一化的深度残差 |z_buf - z_in|
    输出（per-point）:
      - b_local, b_global: 对 local / global token 的 additive bias
    """

    def __init__(self, use_depth_res: bool = True, hidden_dim: int = 32):
        super().__init__()
        self.use_depth_res = use_depth_res

        in_dim = 2
        if use_depth_res:
            in_dim += 1

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_vis: torch.Tensor, depth_res: torch.Tensor | None = None):
        """
        p_vis: [B, L] in [0,1]
        depth_res: [B, L] >= 0 (可选, 若 use_depth_res=False 则传 None)

        返回:
          b_local:  [B, L]
          b_global: [B, L]
        """
        B, L = p_vis.shape
        device, dtype = p_vis.device, p_vis.dtype

        vis = p_vis.clamp(0.0, 1.0)

        uncert = 4.0 * vis * (1.0 - vis)
        conf = 1.0 - uncert

        feats = [vis.unsqueeze(-1), conf.unsqueeze(-1)]

        if self.use_depth_res and depth_res is not None:
            dr = depth_res
            dr = dr / (dr.mean() + 1e-6)
            dr = torch.clamp(dr, 0.0, 3.0)
            feats.append(dr.unsqueeze(-1))

        x = torch.cat(feats, dim=-1)

        bias_raw = self.mlp(x)

        bias_normed = torch.tanh(bias_raw) * self.bias_scale

        b_local = bias_normed[..., 0]
        b_global = bias_normed[..., 1]

        return b_local, b_global

class ProjCrossAttn(nn.Module):
    def __init__(self, gauss_dim, img_c, out_dim=None, num_heads=8, d_model=1024):
        """
        gauss_dim: Dg (input gaussian feature dim)
        img_c: channel dim of sampled image tokens (C)
        d_model: internal head dimension total (should be divisible by num_heads)
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(gauss_dim, d_model)
        self.k_proj = nn.Linear(img_c, d_model)
        self.v_proj = nn.Linear(img_c, d_model)
        self.out = nn.Linear(d_model, out_dim if out_dim is not None else gauss_dim)

        self.q_ln = nn.LayerNorm(d_model)
        self.k_ln = nn.LayerNorm(d_model)
        self.v_ln = nn.LayerNorm(d_model)

        self.nail_delta = nn.Sequential(
            nn.Linear(gauss_dim, gauss_dim//4),
            nn.GELU(),
            nn.Linear(gauss_dim//4, out_dim if out_dim is not None else gauss_dim)
        )
        self.nail_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, gauss_feat, patches, coords=None, mask=None, nail_mask=None):
        """
        gauss_feat: [B, N, Dg]
        patches:    [B, N, K, C_img]
        coords:     [B, N, K, 2] normalized coords (optional, for pos bias)
        mask:       [B, N, K] bool, True means valid (optional)
        returns: out [B,N,out_dim]
        """

        B, N, K, C = patches.shape
        q = self.q_proj(gauss_feat)
        k = self.k_proj(patches)
        v = self.v_proj(patches)

        q = self.q_ln(q)
        k = self.k_ln(k)
        v = self.v_ln(v)

        qh = q.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        kh = k.view(B, N, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)
        vh = v.view(B, N, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)

        logits = torch.einsum('bhnd,bhnkd->bh nk', qh, kh)
        logits = logits / math.sqrt(self.head_dim)

        if coords is not None:
            rel = coords.view(B*N*K, 2)
            bias = self.pos_mlp(rel)
            bias = bias.view(B, N, K, self.num_heads).permute(0,3,1,2)
            logits = logits + bias

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum('bh nk, bhnkd->bhnd', attn, vh)
        out = out.permute(0,2,1,3).contiguous().view(B, N, -1)
        out = self.out(out)

        if nail_mask is not None:
            delta = self.nail_delta(gauss_feat)

            mask_f = nail_mask.float().unsqueeze(-1)
            out = out + mask_f * (self.nail_alpha * delta)
        else:
            out = out

        return out

class PointToTokenAdapter(nn.Module):
    """
    将 per-point features (B, N, D) 聚合成固定数量 L tokens (B, L, D)
    使用 learnable query tokens + multi-head cross-attention (MHA).
    推荐用法：把 adapter 的输出作为 SD3MMJointTransformerBlock 的 encoder_hidden_states。

    Args:
        dim: feature dim (D) — must match transformer dim (e.g., 1024)
        num_tokens: L_ctx (e.g., 1024)
        num_heads: MHA heads (建议与 transformer 保持一致)
        dropout: dropout in MHA (optional)
        use_ln: 在输出前使用 LayerNorm（可选）
    """
    def __init__(self, dim: int, num_tokens: int = 1024, num_heads: int = 8, dropout: float = 0.0, use_ln: bool = True):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads

        self.query_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False)

        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.mlp_ln = nn.LayerNorm(dim)

    def forward(self, point_feats: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            point_feats: (B, N, D)
            valid_mask: optional (B, N) boolean where True = valid. If given, will be converted to attn_mask expected by MHA.

        Returns:
            tokens: (B, L, D)
        """

        B, N, D = point_feats.shape
        assert D == self.dim, f"dim mismatch: {D} vs adapter {self.dim}"

        q = self.query_tokens.unsqueeze(1).expand(-1, B, -1).contiguous()

        kv = point_feats.permute(1, 0, 2).contiguous()

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask

            key_padding_mask = key_padding_mask.bool()

        attn_output, _ = self.mha(q, kv, kv, key_padding_mask=key_padding_mask)

        tokens = attn_output.permute(1, 0, 2).contiguous()

        if self.use_ln:
            tokens = self.ln(tokens)

        mlp_out = self.mlp(tokens)
        tokens = tokens + self.mlp_ln(mlp_out)

        return tokens

class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask: Optional[torch.Tensor] = None, pos_bias: Optional[torch.Tensor] = None):
        """
        q: [B, Lq, D]
        k: [B, Lk, D]
        v: [B, Lk, D]
        key_padding_mask: [B, Lk] boolean True = valid (if follows previous semantics), we'll expect mask where True=valid; we'll convert below
        pos_bias: optional [B, nhead, Lq, Lk] added to logits before softmax
        returns: out [B, Lq, D], attn_weights [B, nhead, Lq, Lk]
        """
        B, Lq, D = q.shape
        _, Lk, _ = k.shape

        q_lin = self.q_proj(q).view(B, Lq, self.nhead, self.head_dim).permute(0,2,1,3)
        k_lin = self.k_proj(k).view(B, Lk, self.nhead, self.head_dim).permute(0,2,1,3)
        v_lin = self.v_proj(v).view(B, Lk, self.nhead, self.head_dim).permute(0,2,1,3)

        logits = torch.einsum('bhqd,bhkd->bhqk', q_lin, k_lin) / (self.head_dim ** 0.5)

        if pos_bias is not None:
            logits = logits + pos_bias

        if key_padding_mask is not None:
            kp = key_padding_mask
            if kp.dtype != torch.bool:
                kp = kp.bool()

            mask = (~kp).unsqueeze(1).unsqueeze(1)
            logits = logits.masked_fill(mask, float('-1e9'))

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out_h = torch.einsum('bhqk,bhkd->bhqd', attn, v_lin)
        out = out_h.permute(0,2,1,3).contiguous().view(B, Lq, D)
        return self.out_proj(out)

class QFormerWithSelf(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_tokens: int = 1024,
        num_heads: int = 16,
        num_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        use_ln: bool = True,
        use_pos_bias: bool = True,
        pos_hidden: int = 128,
    ):
        """
        dim: embedding dim (D)
        num_tokens: L (number of learnable query tokens)
        num_heads: attention heads
        num_layers: number of stacked Q-former blocks
        use_pos_bias: if True, compute pos-bias from (query_pos - point_coord) via small MLP -> per-head bias
        """
        super().__init__()
        self.dim = dim
        self.L = num_tokens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_ln = use_ln
        self.use_pos_bias = use_pos_bias

        self.query_tokens = nn.Parameter(torch.randn(self.L, dim) * 0.02)

        if use_pos_bias:
            self.query_pos = nn.Parameter(torch.randn(self.L, 2) * 0.01)

            self.pos_mlp = nn.Sequential(
                nn.Linear(2, pos_hidden),
                nn.GELU(),
                nn.Linear(pos_hidden, num_heads)
            )

        self.cross_attns = nn.ModuleList([ MultiHeadAttentionCustom(dim, num_heads, dropout=dropout) for _ in range(num_layers) ])
        self.self_attns  = nn.ModuleList([ MultiHeadAttentionCustom(dim, num_heads, dropout=dropout) for _ in range(num_layers) ])
        self.ffns        = nn.ModuleList([ nn.Sequential(
                                            nn.Linear(dim, dim * ffn_mult),
                                            nn.GELU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(dim * ffn_mult, dim),
                                            nn.Dropout(dropout)
                                          ) for _ in range(num_layers) ])

        self.cross_ln = nn.ModuleList([ nn.LayerNorm(dim) for _ in range(num_layers) ])
        self.self_ln  = nn.ModuleList([ nn.LayerNorm(dim) for _ in range(num_layers) ])
        self.ffn_ln   = nn.ModuleList([ nn.LayerNorm(dim) for _ in range(num_layers) ])

        self.out_ln = nn.LayerNorm(dim) if use_ln else nn.Identity()

    def forward(self, point_feats: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, point_coords: Optional[torch.Tensor] = None):
        """
        point_feats: [B, N, D]
        valid_mask: optional [B, N] boolean True = valid
        point_coords: optional [B, N, 2] (x,y normalized or pixel coords) used to compute relative pos bias
                      If provided and use_pos_bias=True, we'll compute bias = pos_mlp(query_pos - point_coords)
        returns: tokens [B, L, D]
        """
        B, N, D = point_feats.shape
        assert D == self.dim, f"dim mismatch {D} vs {self.dim}"

        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()

        kp = None
        if valid_mask is not None:
            kp = valid_mask.bool()

        pos_bias = None
        if self.use_pos_bias and point_coords is not None:
            qp = self.query_pos.unsqueeze(0).expand(B, -1, -1)
            pc = point_coords.unsqueeze(1)
            delta = qp.unsqueeze(2) - pc

            delta_flat = delta.view(B * self.L * N, 2)
            bias_flat = self.pos_mlp(delta_flat)
            bias = bias_flat.view(B, self.L, N, self.num_heads)

            pos_bias = bias.permute(0,3,1,2).contiguous()

        for i in range(self.num_layers):
            cross_attn = self.cross_attns[i]
            out_cross = cross_attn(q, point_feats, point_feats, key_padding_mask=kp, pos_bias=pos_bias)

            q = q + out_cross
            q = self.cross_ln[i](q)

            self_attn = self.self_attns[i]
            out_self = self_attn(q, q, q, key_padding_mask=None, pos_bias=None)
            q = q + out_self
            q = self.self_ln[i](q)

            ffn = self.ffns[i]
            ffn_out = ffn(q)
            q = q + ffn_out
            q = self.ffn_ln[i](q)

        tokens = self.out_ln(q)
        return tokens

class PatchAggregatorWithPosWeightedPool(nn.Module):
    def __init__(self, C_in, D_hidden=256, out_dim=None):
        super().__init__()

        self.out_proj = None
        if out_dim is not None:
            self.out_proj = nn.Linear(C_in, out_dim)

    def forward(self, patches, coords_norm, valid_mask):
        B, N, K, C = patches.shape
        device = patches.device

        center_norm = coords_norm.mean(dim=2)
        point_coords_norm = center_norm.unsqueeze(2).expand(-1, -1, K, -1)

        dist = torch.norm(coords_norm - point_coords_norm, dim=-1)

        pos_logits = - dist / 0.1

        pos_logits = pos_logits.masked_fill(~valid_mask, float('-1e9'))
        weights = torch.softmax(pos_logits, dim=2)

        weighted = (patches * weights.unsqueeze(-1)).sum(dim=2)
        if self.out_proj is not None:
            BN = B*N
            return self.out_proj(weighted.view(BN, C)).view(B, N, -1)
        return weighted

class ProjCrossAttnMatch(nn.Module):
    def __init__(self, gauss_dim, img_c, out_dim, num_heads=8, use_running_stats=True, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim

        self.q_proj = nn.Linear(gauss_dim, out_dim)
        self.k_proj = nn.Linear(img_c, out_dim)
        self.v_proj = nn.Linear(img_c, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.out_ln = nn.LayerNorm(out_dim, eps=1e-6)

        self.alpha = nn.Parameter(torch.tensor(0.1))

        nn.init.constant_(self.out_proj.weight, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        self.use_running_stats = use_running_stats
        self.register_buffer('running_mean', torch.zeros(out_dim))
        self.register_buffer('running_std', torch.ones(out_dim))
        self.register_buffer('running_count', torch.tensor(0, dtype=torch.long))
        self.momentum = 0.01
        self.eps = eps

    def _compute_stats(self, tensor):
        mean = tensor.mean(dim=(0,1), keepdim=True)
        std = tensor.std(dim=(0,1), unbiased=False, keepdim=True)
        return mean, std

    def _update_running(self, batch_mean, batch_std):
        bm = batch_mean.view(-1)
        bs = batch_std.view(-1)
        if self.running_count == 0:
            self.running_mean.copy_(bm.detach())
            self.running_std.copy_(bs.detach())
            self.running_count += 1
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * bm.detach()
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * bs.detach()
            self.running_count += 1

    def forward(self, gauss_feat, patches, ref_tokens=None, mask=None):
        """
        gauss_feat: [B, M, Dg]  (query features)
        patches:    [B, M, K, C_img]  (local image patches)
        ref_tokens: optional [B, N_ref, Dout] (transformer tokens used as reference)
        mask: optional [B, M, K] boolean for invalid patch locations
        returns: updated features [B, M, Dout]
        """
        B, M, K, C = patches.shape

        q = self.q_proj(gauss_feat)
        k = self.k_proj(patches)
        v = self.v_proj(patches)

        qh = q.view(B, M, self.num_heads, self.head_dim).permute(0,2,1,3)
        kh = k.view(B, M, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)
        vh = v.view(B, M, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)

        logits = torch.einsum('bhmd,bhmkd->bhmk', qh, kh) / math.sqrt(self.head_dim)

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = torch.softmax(logits, dim=-1)
        out_h = torch.einsum('bhmk,bhmkd->bhmd', attn, vh)
        out = out_h.permute(0,2,1,3).contiguous().view(B, M, -1)

        out = self.out_proj(out)
        out = self.out_ln(out)

        if ref_tokens is not None:
            assert ref_tokens.shape[-1] == out.shape[-1], "ref token dim mismatch"

            ref_mean, ref_std = self._compute_stats(ref_tokens)
            out_mean, out_std = self._compute_stats(out)

            if self.use_running_stats and self.training:
                self._update_running(ref_mean, ref_std)

            out_normed = (out - out_mean) / (out_std + self.eps)
            matched = out_normed * (ref_std + self.eps) + ref_mean
            final = gauss_feat + self.alpha * matched
            return final
        else:
            if self.use_running_stats and self.running_count > 0:
                running_mean = self.running_mean.view(1,1,-1)
                running_std = self.running_std.view(1,1,-1)
                out_normed = (out - out.mean(dim=(0,1), keepdim=True)) / (out.std(dim=(0,1), keepdim=True) + self.eps)
                matched = out_normed * (running_std + self.eps) + running_mean
                final = gauss_feat + self.alpha * matched
                return final
            else:
                return gauss_feat + self.alpha * out
