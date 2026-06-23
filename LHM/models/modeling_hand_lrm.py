# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  && Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:40:57
# @Function      : Main codes for LHM
import os
import pdb
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

import math
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2  # For UV map rasterization
from accelerate.logging import get_logger
from diffusers.utils import is_torch_version
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional

from LHM.models.arcface_utils import ResNetArcFace
try:
    from LHM.models.ESRGANer_utils import ESRGANEasyModel
except ImportError:
    ESRGANEasyModel = None
from LHM.models.rendering.gs_renderer import PointEmbed, GS_Hand_3DRenderer, HierarchicalPointEmbed
from LHM.models.rendering.gsplat_renderer import GSPlatRenderer

from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding

# from openlrm.models.stylegan2_utils import EasyStyleGAN_series_model
from LHM.models.utils import linear
# from tools_utils import libcore

from .embedder import CameraEmbedder
from .rendering.synthesizer import TriplaneSynthesizer
from .transformer import TransformerDecoder

from splatformer.utils import loss_utils
from splatformer.utils.metrics import psnr, ssim

from data.interhand.train import Renderer_mesh

from LHM.models.styleunet import StyleUNet

logger = get_logger(__name__)

# from splatformer.utils.optimizers import build_optimizer, build_scheduler

_GAUSSIANHAND_ROOT = Path(__file__).resolve().parents[2].parent / 'GuassianHand'
if _GAUSSIANHAND_ROOT.exists():
    if str(_GAUSSIANHAND_ROOT) not in sys.path:
        sys.path.append(str(_GAUSSIANHAND_ROOT))
    try:
        from livehand.input_encoder import get_uvd as gaussianhand_get_uvd  # type: ignore
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
        transformer_layers=8, # 15,
        transformer_heads=16,
        transformer_type="sd3_mm_cond",   # cond
        tf_grad_ckpt=True,
        encoder_grad_ckpt=True,
        encoder_freeze: bool = False,
        encoder_type: str = "dinov2_fusion",     # dino
        encoder_model_name: str = "dinov2_vitl14_reg",    # facebook/dino-vitb16
        encoder_feat_dim: int = 1024,    # 768
        num_pcl: int = 2048,   # 2048
        pcl_dim: int = 1024,    # 512
        human_model_path='./pretrained_models/human_model_files',
        smplx_subdivide_num=1,
        smplx_type="smplx",
        gs_query_dim=1024,
        # gs_use_rgb=False,     # False
        gs_use_rgb=True,     # False
        gs_sh=3,
        gs_mlp_network_config={'activation': 'silu', 'n_hidden_layers': 2, 'n_neurons': 512},
        gs_xyz_offset_max_step=1.0,
        gs_clip_scaling=[100, 0.01, 0.05, 3000],
        shape_param_dim=10, 
        expr_param_dim=100,
        # fix_opacity=False,
        fix_opacity=True,
        fix_rotation=False,
        fine_encoder_type='sapiens',
        fine_encoder_model_name='./pretrained_models/sapiens/pretrained/checkpoints/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2',
        fine_encoder_feat_dim=1536,
        fine_encoder_freeze=True, #False,
    ):
        # super().__init__()
        super(ModelHandLRM, self).__init__()

        # self.renderer_mesh = Renderer_mesh()

        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt

        # # attributes
        # self.encoder_feat_dim = encoder_feat_dim

        # # modules
        # image encoder  default dino-v2
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=encoder_feat_dim,
        )

        self.fine_encoder_feat_dim = fine_encoder_feat_dim
        self.uv_texture_hw = (256, 256)
        self._warned_missing_uv_mapper = False

        # self.hand_encoder = self._encoder_fn('hamba')(
        #     model_name='/data/hand3d/zh174/pretrained_models/hamba/checkpoints/hamba.ckpt',
        #     freeze=encoder_freeze,
        #     encoder_feat_dim=encoder_feat_dim, )

        pcl_dim = 1024
        # pcl_dim=768

        # self.fine_encoder = self._encoder_fn(fine_encoder_type)(
        #     model_name=fine_encoder_model_name,
        #     freeze=fine_encoder_freeze,
        #     encoder_feat_dim=fine_encoder_feat_dim,
        # ).to('cuda')
        # input_dim = 1536
        input_dim = 1024
        # input_dim=768

        input_dim_ = 1024 # 1536 for sapien; 1024 for dinov2
        # input_dim_=768
        mid_dim = input_dim // 2
        self.motion_embed_mlp = nn.Sequential(
            # nn.MaxPool1d(kernel_size=input_dim,),
            linear(input_dim, mid_dim),
            nn.SiLU(),
            linear(mid_dim, pcl_dim),
            # linear(mid_dim, pcl_dim * 2),
        ).to('cuda')
        # self.proj = nn.Linear(input_dim * 8 * 8, input_dim_).to('cuda')

        # learnable points embedding
        skip_decoder = False
        self.latent_query_points_type = 'e2e_smplx_sub1'
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl  # 2048
            self.pcl_embeddings = nn.Embedding(num_pcl, pcl_dim)  # 1024
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
            self.pcl_embed = PointEmbed(dim=pcl_dim)  # pcl dim 1024
            # self.pcl_embed = HierarchicalPointEmbed().cuda()
        else:
            raise NotImplementedError
        print(f"==========skip_decoder:{skip_decoder}")

        # self.vertex_global_mapping = nn.Sequential(nn.Linear(1024, 512),nn.LeakyReLU(inplace=True),
        #                                           nn.Linear(512, 512),nn.LeakyReLU(inplace=True),
        #                                           nn.Linear(512, 1024)).to('cuda')

        # transformer
        self.transformer = self.build_transformer(
            transformer_type,     # sd3_mm_cond
            transformer_layers,
            transformer_heads,
            transformer_dim,
            # 13361, # 12337 + 1024
            encoder_feat_dim,
        )#.to('cuda:1')
        for blk in self.transformer.layers:
            if hasattr(blk, "part_aware_point"):
                blk.part_aware_point.to('cuda:0')

        
        
        # self.uv_style_mapping=nn.Sequential(
        #     nn.Linear(1024,512),nn.LeakyReLU(inplace=True),
        #     nn.Linear(512,512),nn.LeakyReLU(inplace=True),
        #     nn.Linear(512,512))
        
        self.uvmap_size = 1024 # 256
        # self.uv_feature_decoder = StyleUNet(
        #     in_size=self.uvmap_size, 
        #     out_size=self.uvmap_size,
        #     activation=False,
        #     in_dim=input_dim+3, 
        #     out_dim=self.uvmap_size,
        #     extra_style_dim=input_dim).cuda()

        # UV map bias for finetuning (initialized in finetune mode)
        self.uv_map_bias = None  # Will be [1, C, uvmap_size, uvmap_size] when activated
        # Per-UV color / opacity bias (added for fine-tuning; initialized lazily in runner)
        self.color_b_map = None   # expected shape: [1, 3, H, W]
        self.opacity_b_map = None # expected shape: [1, 1, H, W]

        # Style-based refinement network for bias maps (color + opacity).
        # It takes a 4-channel UV bias map (RGB color bias + 1-channel opacity bias)
        # and refines it using a global style code (motion_tokens).
        # This is mainly used during finetuning for in-the-wild images.
        # self.bias_style_unet = StyleUNet(
        #     in_size=self.uvmap_size,
        #     out_size=self.uvmap_size,
        #     in_dim=4,
        #     out_dim=4,
        #     activation=False,
        #     extra_style_dim=transformer_dim,
        # ).to("cuda")
        # self.bias_style_unet = None

        # # Initialize transformer parameters
        # self.intermediate_layer_idx = ([2,5,8])
        # self.transformer_1 = AlternatingCrossAttn(
        #     num_layers=transformer_layers,
        #     num_heads=transformer_heads,
        #     inner_dim=transformer_dim,
        #     cond_dim=transformer_dim,
        #     gradient_checkpointing=self.gradient_checkpointing,
        #     aa_order=aa_order,
        #     aa_block_size=aa_block_size,
        #     intermediate_layer_idx=self.intermediate_layer_idx,
        #     use_flame_tokens=use_flame_tokens,
        #     flame_encoder_config=flame_encoder_config,
        #     use_camera_tokens=use_camera_tokens,
        #     camera_encoder_config=camera_encoder_config,
        #     patch_start_idx=self.patch_start_idx
        # )
        # self.local_attn = ProjCrossAttn(
        #     img_c=input_dim_,
        #     gauss_dim=input_dim_,
        # ).cuda()
        
        # self.adapter = PointToTokenAdapter(dim=input_dim_).cuda()
        self.adapter = QFormerWithSelf(dim=input_dim_).cuda()
        # # ---- Ablation: replace QFormer adapter with a simple MLP ----
        # # Transposes sequence dim, projects 12337 -> 1024, then transposes back.
        # self.adapter = nn.Sequential(
        #     nn.Linear(12337, 1024),   # (B, 1024, 12337) -> (B, 1024, 1024)
        #     nn.GELU(),
        #     nn.Linear(1024, 1024),
        # ).cuda()
        # # Wrap so call convention matches: input (B,12337,1024) -> output (B,1024,1024)
        # _mlp_adapter = self.adapter
        # class _MLPAdapterWrapper(nn.Module):
        #     def __init__(self, mlp):
        #         super().__init__()
        #         self.mlp = mlp
        #     def forward(self, x, **kwargs):
        #         # x: (B, 12337, 1024) -> transpose -> (B, 1024, 12337) -> mlp -> (B, 1024, 1024)
        #         return self.mlp(x.transpose(1, 2))
        # self.adapter = _MLPAdapterWrapper(_mlp_adapter).cuda()
        # # ---- End Ablation ----
        self.aggregator = PatchAggregatorWithPosWeightedPool(
            C_in=input_dim_, D_hidden=128, out_dim=input_dim_).cuda()
            # C_in=1024, D_hidden=128, out_dim=input_dim_).cuda()

        # renderer
        cano_pose_type = 1
        dense_sample_pts = 12337
        #
        # expr_param_dim = 10
        #
        # original 3DGS Raster
        self.renderer = GS_Hand_3DRenderer(
            # human_model_path=human_model_path,
            # subdivide_num=smplx_subdivide_num,
            # smpl_type=smplx_type,
            pcl_embed=None, #self.pcl_embed.point_pos_embed,
            feat_dim=transformer_dim,
            query_dim=gs_query_dim,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            mlp_network_config=gs_mlp_network_config,
            xyz_offset_max_step=gs_xyz_offset_max_step,
            clip_scaling=gs_clip_scaling,
            # shape_param_dim=shape_param_dim,
            # expr_param_dim=expr_param_dim,
            # cano_pose_type=cano_pose_type,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            decoder_mlp=False,
            skip_decoder=skip_decoder,
            decode_with_extra_info=None,
            gradient_checkpointing=self.gradient_checkpointing,
            apply_pose_blendshape=False,
            # dense_sample_pts=dense_sample_pts,
            # feature_map=True,
            feature_map=False,
            # dataset_center_add_map: 0=InterHand True, 1=HanCo True, 2=Hands11k False
            dataset_center_add_map={0: True, 1: True, 2: False},
        )
        
        # ========== Initialize UV map assets ==========
        self._init_uvmap_assets()
        
        params = self.obtain_params()
        
        # self.warmup_param = [n for n, _ in self.pcl_embed.named_parameters() 
        #                      if not ('part_path' in n or 'joint_path' in n)]
        
        self.optimizer = torch.optim.AdamW(params)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=5000,     # 比如你训练 5000 iteration 完成一个周期
                    eta_min=1e-6
                )


        # Default checkpoint path (can be overridden by runner at runtime)
        self.checkpoint_path = os.path.join('./checkpoint/exp-mix3-1')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # ========== UV Feature Fusion Control ==========
        # Learnable weight for controlling UV feature contribution
        # Start with 0.1, can be learned during training
        self.uv_feat_fusion_weight = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        

    def _init_uvmap_assets(self):
        """Initialize UV map assets from MANO model for single right hand (256x256 resolution)."""
        import cv2
        
        mano = self.renderer.mano_model.mano
        uvmap_size = 256  # Single hand, square UV map
        
        # Get MANO mesh info
        faces = mano.faces.numpy() if isinstance(mano.faces, torch.Tensor) else mano.faces
        
        # Try to load UV information from MANO or create from scratch
        # For MANO right hand, we need faces_uv_idx and texcoords
        if hasattr(mano, 'faces_uv_idx') and hasattr(mano, 'texcoords'):
            faces_uv_idx = mano.faces_uv_idx.numpy() if isinstance(mano.faces_uv_idx, torch.Tensor) else mano.faces_uv_idx
            texcoords = mano.texcoords.numpy() if isinstance(mano.texcoords, torch.Tensor) else mano.texcoords
        else:
            # Fallback: try to load from file or use simple parametrization
            logger.warning("MANO model doesn't have UV attributes. Attempting to load from assets...")
            # For now, create a simple planar UV parametrization
            num_verts = mano.v_template.shape[0]
            faces_uv_idx = faces.copy()  # For simple planar, UV vertex = 3D vertex
            # Create simple 2D parametrization (normalize to [0,1])
            v_template_np = mano.v_template.numpy() if isinstance(mano.v_template, torch.Tensor) else mano.v_template
            v_min = v_template_np.min(axis=0)
            v_max = v_template_np.max(axis=0)
            texcoords = (v_template_np[:, :2] - v_min[:2]) / (v_max[:2] - v_min[:2] + 1e-8)
        
        # Compute uvmap_f_idx: for each UV pixel, which face does it belong to?
        uvmap_f_idx = self._get_uvmap_faces_index(faces_uv_idx, texcoords, uv_size=uvmap_size)    # 256,256
        
        # Compute uvmap_f_bary: barycentric coordinates of each UV pixel within its face
        uvmap_f_bary = self._get_uvmap_faces_barycoord(uvmap_f_idx, faces_uv_idx, texcoords, uv_size=uvmap_size)    # 256,256,3
        
        # Create UV validity mask
        uvmap_mask = (uvmap_f_idx != -1)    # 256,256

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Register as buffers (non-trainable)
        self.register_buffer('uvmap_f_idx', torch.tensor(uvmap_f_idx, dtype=torch.int32, device=device))
        self.register_buffer('uvmap_f_bary', torch.tensor(uvmap_f_bary, dtype=torch.float32, device=device))
        self.register_buffer('uvmap_mask', torch.tensor(uvmap_mask, dtype=torch.bool, device=device))
        
        # Flatten mask for later indexing
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
            # Draw filled triangle for each face
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
                
                # Get triangle vertices in UV space
                v_uvs = uv_coords_px[faces_uv[f_idx]]
                v_uv0, v_uv1, v_uv2 = v_uvs[0], v_uvs[1], v_uvs[2]
                c_uv = np.array([u_idx, v_idx])
                
                # Compute barycentric coordinates using cross product
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
        
        # Initialize UV feature map
        uv_features = torch.zeros(
            (batch_size, feature_dim, self.uvmap_size, self.uvmap_size),
            device=device, dtype=torch.float32
        )
        
        # Get pre-computed UV-to-mesh mapping
        uvmap_f_idx = self.uvmap_f_idx.to(device)  # [H, W]
        uvmap_f_bary = self.uvmap_f_bary.to(device)  # [H, W, 3]
        uvmap_mask = self.uvmap_mask.to(device)  # [H, W]
        
        # Get MANO mesh faces
        mano = self.renderer.mano_model.mano
        faces = torch.tensor(mano.faces, device=device, dtype=torch.long)
        
        # For each UV pixel, find corresponding 3D point on mesh
        uv_vertex_id = faces[uvmap_f_idx]  # [H, W, 3]
        
        # Gather 3D vertices: [B, num_verts, 3] -> [B, H, W, 3, 3]
        # uv_vertex[b, h, w, k] = deformed_vertices[b, uv_vertex_id[h, w, k]]
        uv_vertex = torch.zeros(
            (batch_size, self.uvmap_size, self.uvmap_size, 3, 3),
            device=device, dtype=torch.float32
        )
        for k in range(3):
            uv_vertex[:, :, :, k, :] = deformed_vertices[:, uv_vertex_id[:, :, k], :]  # [B, H, W, 3]
        
        # Barycentric interpolation: blend 3 vertices using bary weights
        # uvmap_f_bary: [H, W, 3]
        uv_vertex_interp = torch.einsum('hwk,bhwkn->bhwn', uvmap_f_bary, uv_vertex)  # [B, H, W, 3]
        
        # Project 3D points to image space
        uv_vertex_homo = torch.cat([uv_vertex_interp, torch.ones_like(uv_vertex_interp[:, :, :, :1])], dim=-1)  # [B, H, W, 4]
        
        # Apply world-to-camera transform
        uv_vertex_cam = torch.einsum('bij,bhwj->bhwi', w2c_cam, uv_vertex_homo)[:, :, :, :3]  # [B, H, W, 3]
        
        # Project to image (perspective): (X, Y, Z) -> (u, v) = focal * (X/Z, Y/Z) + principal_point
        # For simplicity, assume camera intrinsics: focal_x = focal_y = 256 (half of image width, typical for normalized coords)
        focal = img_size / 2.0  # Approximate focal length
        vertices_img_x = focal * uv_vertex_cam[:, :, :, 0] / (uv_vertex_cam[:, :, :, 2] + 1e-8)
        vertices_img_y = focal * uv_vertex_cam[:, :, :, 1] / (uv_vertex_cam[:, :, :, 2] + 1e-8)
        
        # Normalize to [-1, 1] for grid_sample (PyTorch convention)
        vertices_img_x_norm = 2.0 * (vertices_img_x / img_size) - 1.0
        vertices_img_y_norm = 2.0 * (vertices_img_y / img_size) - 1.0
        vertices_img_norm = torch.stack([vertices_img_x_norm, vertices_img_y_norm], dim=-1)  # [B, H, W, 2]
        
        # Sample features from image using grid_sample
        # Reshape for grid_sample: [B, C, H, W] and grid [B, H, W, 2]
        uv_features_sampled = torch.nn.functional.grid_sample(
            img_features,
            vertices_img_norm,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, C, H, W]
        
        # Apply mask: keep only valid UV pixels
        mask = uvmap_mask.clone()[None, None, :, :].repeat(batch_size, 1, 1, 1).float()  # [B, 1, H, W]
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
        # vert_uv = vert_uv.unsqueeze(0)
        batch_size, n_points, _ = vert_uv.shape
        device = vert_uv.device
        
        # Normalize UV to [-1, 1] for grid_sample
        grid = vert_uv.clone()
        grid[..., 0] = 2.0 * vert_uv[..., 0] - 1.0  # u
        grid[..., 1] = 2.0 * vert_uv[..., 1] - 1.0  # v
        
        # Add spatial dimension for grid_sample: [B, N, 2] -> [B, 1, N, 2]
        grid = grid[:, None, :, :].contiguous()  # [B, 1, N, 2]
        
        # Sample: [B, C, H, W] and grid [B, 1, N, 2]
        sampled = torch.nn.functional.grid_sample(
            uvmap_features, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [B, C, 1, N]
        
        # Reshape to [B, N, C]
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
        
        # Check if uv_map_dict contains batched data or needs processing
        if uv_map_dict is None or not isinstance(uv_map_dict, dict):
            return {'uvmap_features': None, 'point_uv_features': None}
        
        # Handle different input formats
        # Case 1: All data is already stacked as tensors
        vert_uv = uv_map_dict.get('vert_uv')
        face_uv = uv_map_dict.get('face_uv')
        face_uv_xy = uv_map_dict.get('face_uv_xy')
        posed_points = uv_map_dict.get('posed_points')
        cam = uv_map_dict.get('cam')
        
        if vert_uv is None or face_uv is None or face_uv_xy is None:
            return {'uvmap_features': None, 'point_uv_features': None}
        
        # Normalize data format: convert to lists for per-batch processing
        def to_batch_list(data, batch_size):
            """Convert various input formats to list of per-batch tensors"""
            if isinstance(data, list):
                return data
            elif isinstance(data, torch.Tensor):
                if data.ndim >= 2 and data.shape[0] == batch_size:
                    # Already batched: [B, ...]
                    return [data[i] for i in range(batch_size)]
                else:
                    # Shared across batch: [...] -> repeat for each batch
                    return [data for _ in range(batch_size)]
            else:
                # Fallback: assume shared
                return [data for _ in range(batch_size)]
        
        vert_uv_list = to_batch_list(vert_uv, batch_size)
        face_uv_list = to_batch_list(face_uv, batch_size)
        face_uv_xy_list = to_batch_list(face_uv_xy, batch_size)
        posed_points_list = to_batch_list(posed_points, batch_size)
        
        # Handle camera: can be [4, 4] (shared), [B, 4, 4] (batched), or list
        if isinstance(cam, list):
            cam_list = cam
        elif isinstance(cam, torch.Tensor):
            if cam.shape[0] == batch_size and cam.ndim == 3:
                cam_list = [cam[i] for i in range(batch_size)]
            else:
                cam_list = [cam for _ in range(batch_size)]
        else:
            # Create identity camera if missing
            cam_list = [torch.eye(4, device=device, dtype=torch.float32) for _ in range(batch_size)]
        
        # Process each batch separately
        uvmap_feats_list = []
        point_uv_feats_list = []
        point_uv_coords_list = []  # store per-point UV coords for later bias sampling
        
        for b in range(batch_size):
            # Get data for this batch
            vert_uv_b = vert_uv_list[b].to(device)
            face_uv_b = face_uv_list[b].to(device).long()
            face_uv_xy_b = face_uv_xy_list[b].to(device)
            posed_points_b = posed_points_list[b].to(device)
            cam_b = cam_list[b].to(device)
            
            # Ensure posed_points has batch dimension for convert_pixel_feature_to_uv
            if posed_points_b.ndim == 2:
                posed_points_b = posed_points_b.unsqueeze(0)  # [1, N, 3]
            
            # Ensure camera has batch dimension
            if cam_b.ndim == 2:
                cam_b = cam_b.unsqueeze(0)  # [1, 4, 4]
            
            # Step 1: Warp image features to UV space for this batch
            combined_feats_b = combined_feats[b:b+1]  # [1, C, H, W]
            if self.uv_map_bias is not None:
                # During finetuning, add bias to image features before warping
                combined_feats_b = combined_feats_b + self.uv_map_bias

            uvmap_feats_b = self.convert_pixel_feature_to_uv(
                combined_feats_b, posed_points_b, cam_b, img_size=256
            )  # [1, C, 256, 256]
            
            uvmap_feats_list.append(uvmap_feats_b.squeeze(0))  # [C, 256, 256]
            
            # Step 2: Sample UV features at deformed point locations
            # Use get_uvd to compute UV coordinates of posed points on the canonical UV mesh
            pts_b = posed_points_b.squeeze(0)  # [N_pts, 3]
            
            try:
                from livehand.input_encoder import get_uvd as gaussianhand_get_uvd
                point_uv_b, _, _ = gaussianhand_get_uvd(pts_b, vert_uv_b, face_uv_b, face_uv_xy_b)
                # point_uv_b: [N_pts, 2] UV coordinates
                
                # Sample features from UV map at these coordinates
                point_uv_b_expanded = point_uv_b.unsqueeze(0)  # [1, N_pts, 2]
                point_features_b = self.sample_uv_feature_at_points(
                    uvmap_feats_b, point_uv_b_expanded
                )  # [1, N_pts, C]
                
                point_uv_feats_list.append(point_features_b.squeeze(0))  # [N_pts, C]
                point_uv_coords_list.append(point_uv_b)  # [N_pts, 2] in [0,1]
                
            except Exception as e:
                # If get_uvd fails, return None for this batch
                print(f"Warning: get_uvd failed for batch {b}: {e}")
                point_uv_feats_list.append(None)
        
        # Stack results across batch
        uvmap_features = torch.stack(uvmap_feats_list, dim=0)  # [B, C, 256, 256]
        
        # # ===== Apply UV Map Bias (for finetuning) =====
        # if self.uv_map_bias is not None:
        #     # Add learnable bias to UV features
        #     # uv_map_bias: [1, C, 256, 256], uvmap_features: [B, C, 256, 256]
        #     uvmap_features = uvmap_features + self.uv_map_bias
        # # ===== End UV Map Bias =====
        
        # Handle point features: only stack if all are valid
        if all(f is not None for f in point_uv_feats_list):
            point_uv_features = torch.stack(point_uv_feats_list, dim=0)  # [B, N_pts, C]
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
            # only train base & neighbor
            for n, p in model.named_parameters():
                if 'part_path' in n or 'joint_path' in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            # train all
            for p in model.parameters():
                p.requires_grad = True
        
        


# 提前缓存参数名字
# base_params = [n for n, _ in model.pcl_embed.named_parameters() if not ('part_path' in n or 'joint_path' in n)]
# part_params = [n for n, _ in model.pcl_embed.named_parameters() if 'part_path' in n or 'joint_path' in n]

    def set_grad_iter(self, model, iteration, warmup_iter=1000):
        if iteration == 0:
            print(f"[Init] Warmup for {warmup_iter} iterations.")
        if iteration == warmup_iter:
            print(f"[Iter {iteration}] Unfreezing all modules...")

        # 只在关键点改变 requires_grad
        if iteration == 0:
            for n, p in self.pcl_embed.named_parameters():
                p.requires_grad = n in self.base_params
        elif iteration == warmup_iter:
            for p in self.pcl_embed.parameters():
                p.requires_grad = True


    def obtain_params(self):
        # 禁用 LoRA 参数在常规训练中的梯度（仅在微调脚本中启用）
        try:
            if hasattr(self.renderer, 'lora_mlp') and self.renderer.lora_mlp is not None:
                for p in self.renderer.lora_mlp.parameters():
                    p.requires_grad = False
            if hasattr(self.renderer, 'lora_gs') and self.renderer.lora_gs is not None:
                for p in self.renderer.lora_gs.parameters():
                    p.requires_grad = False
        except Exception:
            pass

        # 自动收集 LayerNorm 参数和 bias 参数，不再手动维护模块列表
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
                "lr": 4e-4, # interhand
                "name": "decay",
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": 4e-4,  # interhand
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
        # torch.save(self.pcl_embed, pcl_embed_path)
        torch.save({
            # 'iter': iteration,
            'pcl_embed': self.pcl_embed.state_dict(),
        }, pcl_embed_path)
        
        motion_embed_path = os.path.join(self.checkpoint_path, 'motion_embed', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(motion_embed_path), exist_ok=True)
        # torch.save(self.pcl_embed, pcl_embed_path)
        torch.save({
            # 'iter': iteration,
            'motion_embed': self.motion_embed_mlp.state_dict(),
        }, motion_embed_path)

        transformer_path = os.path.join(self.checkpoint_path, 'transformer', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(transformer_path), exist_ok=True)
        # torch.save(self.transformer.state_dict(), transformer_path)
        torch.save({'transformer': self.transformer}, transformer_path, _use_new_zipfile_serialization=False)
        
        encoder_path = os.path.join(self.checkpoint_path, 'encoder', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        # torch.save(self.transformer.state_dict(), transformer_path)
        # torch.save({'encoder': self.encoder.fusion_head}, encoder_path, _use_new_zipfile_serialization=False)
        torch.save({'encoder': self.encoder}, encoder_path, _use_new_zipfile_serialization=False)
        # torch.save({
        #     # 'iter': iteration,
        #     'transformer': self.transformer.state_dict(),
        # }, transformer_path)

        renderer_path = os.path.join(self.checkpoint_path, 'renderer', 'iteration_' + str(iteration), 'ckpt.pth')
        os.makedirs(os.path.dirname(renderer_path), exist_ok=True)
        torch.save({
            # 'iter': iteration,
            'renderer': self.renderer.state_dict(),
            # 'mlp_net': self.renderer.mlp_net.state_dict(),
            # 'gs_net': self.renderer.gs_net.state_dict(),
        }, renderer_path)



        if self.renderer.neural_refiner is not None:
            refiner_path = os.path.join(self.checkpoint_path, 'refiner', 'iteration_' + str(iteration), 'ckpt.pth')
            os.makedirs(os.path.dirname(refiner_path), exist_ok=True)
            torch.save({
                # 'iter': iteration,
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
            # 'proj': self.proj.state_dict(),
            # 'local_attn': self.local_attn.state_dict(),
            "adapter": self.adapter.state_dict(),
            "aggregator": self.aggregator.state_dict(),
            "motion_embed_mlp": self.motion_embed_mlp.state_dict(),
            "transformer": self.transformer,
            # "encoder": self.encoder.fusion_head,
            "encoder": self.encoder,
            "renderer": self.renderer.state_dict(),
        }

        # 额外存储新引入的模块，避免手动补充键
        if hasattr(self, "vertex_global_mapping") and self.vertex_global_mapping is not None:
            ckpt["vertex_global_mapping"] = self.vertex_global_mapping.state_dict()

        # Save inversion color parameters (color_shift/color_scale) if present
        if hasattr(self, "color_shift") and self.color_shift is not None:
            ckpt["color_shift"] = self.color_shift.data.detach().cpu()
        if hasattr(self, "color_scale") and self.color_scale is not None:
            ckpt["color_scale"] = self.color_scale.data.detach().cpu()
        if hasattr(self, "color_bias_lowres") and self.color_bias_lowres is not None:
            ckpt["color_bias_lowres"] = self.color_bias_lowres.data.detach().cpu()

        # 完整模型 state_dict，方便未来自动加载（strict=False 可兼容缺失/新增参数）
        try:
            ckpt["model_state"] = self.state_dict()
        except Exception:
            pass

        if self.renderer.neural_refiner is not None:
            ckpt["neural_refiner"] = self.renderer.neural_refiner.state_dict()

        # save optimizer / scheduler states if available
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                ckpt["optimizer"] = self.optimizer.state_dict()
        except Exception:
            pass

        try:
            if hasattr(self, "scheduler") and self.scheduler is not None:
                # schedulers may not be picklable fully; save state_dict
                ckpt["scheduler"] = self.scheduler.state_dict()
        except Exception:
            pass

        filename = "latest.ckpt" if is_latest else f"iteration_{iteration}.ckpt"
        path = os.path.join(self.checkpoint_path, filename)
        torch.save(ckpt, path)

        print(f"\033[92m[Checkpoint saved → {path}]\033[0m")


    def compute_discriminator_loss(self, data):
        return -F.softplus(self.stylegan2_prior(data)).mean()  # StyleGAN2

    # def train(self, mode=True):
    #     super().train(mode)
    #     if self.use_face_id:
    #         # setting id_face_net to evaluation
    #         self.id_face_net.eval()

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
            cond_dim=encoder_feat_dim, #1536, #encoder_feat_dim,
            mod_dim=None,
            # gradient_checkpointing=self.gradient_checkpointing,
        )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight

    def hyper_step(self, step):
        pass

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in [
            "dino",
            "dinov2",
            "dinov2_unet",
            "resunet",
            "dinov2_featup",
            "dinov2_dpt",
            "dinov2_fusion",
            "sapiens",
            "hamba"
        ], "Unsupported encoder type"
        if encoder_type == "dino":
            from .encoders.dino_wrapper import DinoWrapper

            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == "dinov2":
            from .encoders.dinov2_wrapper import Dinov2Wrapper

            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper
        elif encoder_type == "dinov2_unet":
            from .encoders.dinov2_unet_wrapper import Dinov2UnetWrapper

            logger.info("Using Dinov2Unet as the encoder")
            return Dinov2UnetWrapper
        elif encoder_type == "resunet":
            from .encoders.xunet_wrapper import XnetWrapper

            logger.info("Using XnetWrapper as the encoder")
            return XnetWrapper
        elif encoder_type == "dinov2_featup":
            from .encoders.dinov2_featup_wrapper import Dinov2FeatUpWrapper

            logger.info("Using Dinov2FeatUpWrapper as the encoder")
            return Dinov2FeatUpWrapper
        elif encoder_type == "dinov2_dpt":
            from .encoders.dinov2_dpt_wrapper import Dinov2DPTWrapper

            logger.info("Using Dinov2DPTWrapper as the encoder")
            return Dinov2DPTWrapper
        elif encoder_type == "dinov2_fusion":
            from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper

            logger.info("Using Dinov2FusionWrapper as the encoder")
            return Dinov2FusionWrapper
        elif encoder_type == "sapiens":
            from .encoders.sapiens_warpper import SapiensWrapper

            logger.info("Using Sapiens as the encoder")
            return SapiensWrapper
        elif encoder_type == 'hamba':
            from .encoders.hamba_wrapper import HambaWrapper

            logger.info("Using Hamba as the hand encoder")
            return HambaWrapper

    # def forward_transformer(self, image_feats, camera_embeddings, query_points):
    #     """
    #     Applies forward transformation to the input features.
    #     Args:
    #         image_feats (torch.Tensor): Input image features. Shape [B, C, H, W].
    #         camera_embeddings (torch.Tensor): Camera embeddings. Shape [B, D].
    #         query_points (torch.Tensor): Query points. Shape [B, L, D].
    #     Returns:
    #         torch.Tensor: Transformed features. Shape [B, L, D].
    #     """
    # 
    #     B = image_feats.shape[0]
    # 
    #     if self.latent_query_points_type == "embedding":
    #         range_ = torch.arange(self.num_pcl, device=image_feats.device)
    #         x = self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1))  # [B, L, D]
    # 
    #     elif self.latent_query_points_type.startswith("smplx"):
    #         x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat(
    #             (B, 1, 1)
    #         )  # [B, L, D]
    # 
    #     elif self.latent_query_points_type.startswith("e2e_smplx"):   # this way
    #         # Linear warp -> MLP + LayerNorm
    #         x = self.pcl_embed(query_points)  # [B, L, D]
    # 
    #     x = self.transformer(
    #         x,
    #         # temb=image_feats,
    #         # cond=None,
    #         cond=image_feats,
    #         mod=camera_embeddings,
    #     )  # [B, L, D]
    #     return x


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

            uv_feat_norm = self.uv_pixels_to_feat_norm(uv_batch, W_img, H_img, Wf, Hf)  # (B, nb, 2)
            grids = uv_feat_norm.unsqueeze(2) + offsets_norm.view(1, 1, K, 2)
            grids = grids.view(B, nb, P, P, 2)

            batch_patches = []
            for b in range(B):
                grid_b = grids[b]  # (nb, P, P, 2)
                feat_b = feat_map[b].unsqueeze(0)  # (1,C,Hf,Wf)
                patches_b = F.grid_sample(
                    feat_b.expand(nb, -1, -1, -1),  # expand nb only
                    grid_b,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                )  # (nb,C,P,P)
                batch_patches.append(patches_b)
            patches = torch.stack(batch_patches, dim=0)  # (B, nb, C, P, P)

            # flatten + project
            patches_flat = patches.view(B, nb, -1)
            patches_proj = self.proj(patches_flat.view(B*nb, -1)).view(B, nb, -1)
            # if proj is not None:
            #     patches_proj = self.proj(patches_flat.view(B*nb, -1)).view(B, nb, -1)
            # else:
            #     patches_proj = patches_flat

            # sample mask (optional)
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

        # token_feature_maps = token_feature_maps.reshape()
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

            pts = query_points[idx].to(token_feature_maps.device)  # [N,3]
            vert_uv = uv_template['vert_uv'].to(token_feature_maps.device)
            face_uv = uv_template['face_uv'].to(token_feature_maps.device).long()
            face_uv_xy = uv_template['face_uv_xy'].to(token_feature_maps.device)

            # Use GaussianHand get_uvd to obtain per-point UV coordinates on the template.
            sampled_uv, signed_dist, _ = gaussianhand_get_uvd(pts, vert_uv, face_uv, face_uv_xy)

            # Normalize UV to [-1, 1] range for grid_sample, following GaussianHand's convention:
            #   U in [0,1]  -> x in [-1,1]
            #   V in [0,0.5] -> y in [-1,1]
            uv_norm = sampled_uv.clone()
            uv_norm[..., 0] = 2.0 * (uv_norm[..., 0] / 1.0) - 1.0
            uv_norm[..., 1] = 2.0 * (uv_norm[..., 1] / 0.5) - 1.0

            # Prepare grid for grid_sample: [1, N, 1, 2]
            N = uv_norm.shape[0]
            grid = uv_norm.view(1, N, 1, 2)

            # Token map for this batch: [1, C, H, W]
            token_map_b = token_feature_maps[idx:idx + 1]  # keep batch dim

            sampled = F.grid_sample(
                token_map_b,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )  # [1, C, N, 1]

            # -> [N, C]
            feats = sampled.view(C, N).transpose(0, 1).contiguous()

            batch['uv_point_features'] = feats.detach()
            per_batch_feats.append(feats)

        per_batch_feats = torch.stack(per_batch_feats, dim=0) # only the valid ones
        # if not per_batch_feats:
        #     return None

        # Pad or stack only the ones we filled; caller can handle None for missing batches.
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



    def sample_local_patches_light_new(feat_pyramid,  # list of [B,C,Hi,Wi]
                                proj_uv,        # [B, N, 2] in pixel coords (x,y)
                                proj_z=None,    # [B, N] optional depth in cam space
                                patch_size=5,   # odd integer
                                scales=None,    # list of scale factors relative to feat map, or indices
                                device='cuda'):
        """
        Return: patches [B, N, k, C]  (k = patch_size^2 * len(scales))
        feat_pyramid: list of feature maps from FPN (B,C,Hi,Wi)
        proj_uv: pixel coordinates (not normalized), shape [B,N,2]
        proj_z: optional depth for masking [B,N]
        """
        B, N, _ = proj_uv.shape
        if scales is None:
            scales = list(range(len(feat_pyramid)))  # use all pyramid levels

        patches_per_scale = []
        for s in scales:
            feat = feat_pyramid[s]           # [B, C, H, W]
            Bf, C, H, W = feat.shape

            # normalize pixel coords to [-1,1] for this resolution
            # proj_uv are in pixel coordinates relative to full-res image; need to scale to feat map
            # Suppose proj_uv are in full-image coords, and we know full image size (W_img,H_img).
            # Here assume proj_uv already pre-scaled to current feat resolution: u_feat = proj_uv * (W / W_img)
            # For robustness, user should pass proj_uv_feat per scale; for demo we assume already scaled.

            # convert to normalized grid coords
            u = proj_uv[..., 0] / (W - 1) * 2 - 1   # [B,N]
            v = proj_uv[..., 1] / (H - 1) * 2 - 1   # [B,N]
            grid = torch.stack([u, v], dim=-1)      # [B,N,2] normalized

            # build local patch grid offsets in normalized coordinates
            half = (patch_size // 2)
            xs = torch.linspace(-half, half, steps=patch_size, device=device) / (W - 1) * 2
            ys = torch.linspace(-half, half, steps=patch_size, device=device) / (H - 1) * 2
            delta = torch.stack(torch.meshgrid(xs, ys), dim=-1).reshape(-1,2)  # [k,2]

            # expand base grid and add offsets -> [B, N, k, 2]
            grid_exp = grid.unsqueeze(2) + delta.unsqueeze(0).unsqueeze(0)  # broadcast
            grid_exp = grid_exp.view(B, N*patch_size*patch_size, 2)

            # grid_sample expects [B, C, H, W] and grid [B, out_h, out_w, 2] -> we'll reshape
            # We'll sample as (B, N*k, 1) "pixels": use grid_sample by reshaping grid into (B, N*k, 1, 2)
            grid_for_sample = grid_exp.view(B, N*patch_size*patch_size, 1, 2)

            sampled = F.grid_sample(feat, grid_for_sample, mode='bilinear', padding_mode='zeros', align_corners=True)
            # sampled: [B, C, N*k, 1] -> squeeze -> [B, C, N*k]
            sampled = sampled.view(B, C, N, patch_size*patch_size).permute(0,2,3,1)  # [B,N,k,C]
            patches_per_scale.append(sampled)

        # concat across scales into [B,N,k_total,C]
        patches = torch.cat(patches_per_scale, dim=2)  # k_total = patch_size^2 * num_scales
        # optionally flatten k and C for linear: out [B, N, k_total, C]
        return patches  # shape [B, N, k_total, C]



    # ---------- helper: sample local patches across pyramid -----------
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
            # normalize proj_uv to this feat resolution
            # assume proj_uv_full in pixel coords (0..W_full-1, 0..H_full-1)
            W_full, H_full = full_image_wh
            # scale to feat map pixels
            u_feat_x = proj_uv_full[..., 0] * (W / W_full)
            u_feat_y = proj_uv_full[..., 1] * (H / H_full)
            # normalized coords for grid_sample in [-1,1]
            nx = (u_feat_x / (W - 1)) * 2 - 1  # [B,N]
            ny = (u_feat_y / (H - 1)) * 2 - 1
            center = torch.stack([nx, ny], dim=-1)  # [B,N,2]

            half = psize // 2
            xs = (torch.arange(-half, half+1, device=device).float() / (W - 1)) * 2
            ys = (torch.arange(-half, half+1, device=device).float() / (H - 1)) * 2
            # meshgrid offsets in normalized coords
            dx, dy = torch.meshgrid(xs, ys, indexing='xy')  # shape psize x psize
            delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)  # [k,2] normalized offsets

            # expand and add: center [B,N,1,2] + delta[1,k,2] -> [B,N,k,2]
            grid = (center.unsqueeze(2) + delta.view(1,1,-1,2))  # [B,N,k,2]
            # reshape to grid_sample shape [B, N*k, 1, 2]
            grid_rs = grid.view(B, N * grid.shape[2], 2).unsqueeze(2)  # [B, N*k, 1, 2]
            # sample P with grid_sample -> output shape [B, C, N*k, 1]
            sampled = F.grid_sample(P, grid_rs, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled = sampled.view(B, C, N, grid.shape[2]).permute(0,2,3,1)  # [B,N,k,C]
            patches_scales.append(sampled)
            coords_scales.append(grid)  # per-scale normalized coords

        patches = torch.cat(patches_scales, dim=2)  # [B,N,K_total,C]
        coords = torch.cat(coords_scales, dim=2)    # [B,N,K_total,2]
        return patches, coords



    def sample_local_patches_from_pyramid(self,
        feat_pyramid,        # tensor [B,C,H,W] or list of such tensors
        proj_uv_full,        # tensor [B, N, 2]  (pixel coords, floats; x in [0,W_full-1], y in [0,H_full-1])
        patch_sizes,         # int or list (one per level)
        full_image_wh=(256, 256),       # tuple (W_full, H_full)
        align_corners=True,  # grid_sample align_corners consistency
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
        # ensure list
        if isinstance(feat_pyramid, (list, tuple)):
            feat_list = list(feat_pyramid)
        else:
            feat_list = [feat_pyramid]

        B, N, _ = proj_uv_full.shape
        W_full, H_full = full_image_wh

        # normalize patch_sizes to list
        if isinstance(patch_sizes, int):
            patch_sizes = [patch_sizes] * len(feat_list)
        assert len(patch_sizes) == len(feat_list)

        device = proj_uv_full.device
        patches_per_level = []
        coords_per_level = []
        masks_per_level = []
        C_out = feat_list[0].shape[1]  # assume channels same across levels; if not, you'll need to project later

        for lvl, (P, psize) in enumerate(zip(feat_list, patch_sizes)):
            # P: [B, C, Hf, Wf]
            assert P.dim() == 4, "Each level must be [B,C,H,W]"
            Bf, C, Hf, Wf = P.shape
            assert Bf == B, "Batch mismatch between features and proj_uv"

            # scale proj_uv_full (pixel coords) to this level's pixel coords (float)
            # u_feat_x in [0, Wf-1], u_feat_y in [0, Hf-1]
            u_feat_x = proj_uv_full[..., 0] * (Wf / float(W_full))
            u_feat_y = proj_uv_full[..., 1] * (Hf / float(H_full))

            # normalized centers in [-1,1] for grid_sample (align_corners True formula)
            if align_corners:
                nx = (u_feat_x / (Wf - 1)) * 2.0 - 1.0
                ny = (u_feat_y / (Hf - 1)) * 2.0 - 1.0
                # offsets in normalized coords per pixel delta:
                xs = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() / (Wf - 1)) * 2.0
                ys = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() / (Hf - 1)) * 2.0
            else:
                # align_corners False: map pixel centers; use (coord + 0.5)/Wf
                nx = ((u_feat_x + 0.5) / Wf) * 2.0 - 1.0
                ny = ((u_feat_y + 0.5) / Hf) * 2.0 - 1.0
                xs = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() + 0.0) / Wf * 2.0
                ys = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() + 0.0) / Hf * 2.0

            dx, dy = torch.meshgrid(xs, ys, indexing='xy')  # psize x psize
            delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)  # [k, 2]
            k = delta.shape[0]

            # center: [B, N, 2]
            center = torch.stack([nx, ny], dim=-1)  # [B, N, 2]

            # grid before clamping: [B, N, k, 2]
            grid = center.unsqueeze(2) + delta.view(1, 1, k, 2)

            # valid mask per sample (True if inside [-1,1] before clamp)
            valid_mask_lvl = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)  # [B, N, k]

            # clamp grid to [-1,1] so grid_sample won't error
            grid_clamped = grid.clamp(-1.0, 1.0)
            # reshape for grid_sample: [B, N*k, 1, 2]
            grid_rs = grid_clamped.view(B, N * k, 2).unsqueeze(2)  # [B, N*k, 1, 2]

            # sample -> out [B, C, N*k, 1]  then reshape to [B, N, k, C]
            sampled = F.grid_sample(P, grid_rs, mode='bilinear', padding_mode='zeros', align_corners=align_corners)   #为了什么？
            sampled = sampled.view(B, C, N, k).permute(0, 2, 3, 1).contiguous()  # [B, N, k, C]
            patches_per_level.append(sampled)
            coords_per_level.append(grid)         # un-clamped normalized coords (useful for pos bias)
            masks_per_level.append(valid_mask_lvl)

        # concat levels
        patches = torch.cat(patches_per_level, dim=2)  # [B, N, K_total, C]
        coords_norm = torch.cat(coords_per_level, dim=2)  # [B, N, K_total, 2]
        valid_mask = torch.cat(masks_per_level, dim=2)    # [B, N, K_total]

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

        # Step 1: map original pixel -> padded image pixel coords
        x0 = verts_xy_orig[..., 0]
        y0 = verts_xy_orig[..., 1]
        x_pad = x0 + pad_left
        y_pad = y0 + pad_top

        # Step 2: scale padded image to resolution (kornia.resize used align_corners=True)
        scale = float(resolution) / float(max_size)
        x_resized = x_pad * scale
        y_resized = y_pad * scale
        # now x_resized ∈ [0, resolution-1] (float)

        # Step 3: map resized pixel coords -> feature-grid coords (float indices)
        # feature grid size:
        Wf = resolution // patch_size
        Hf = resolution // patch_size
        # Each feature-grid cell corresponds to patch_size pixels. So:
        u_feat_x = x_resized / float(patch_size)   # in [0, Wf-1]
        u_feat_y = y_resized / float(patch_size)

        proj_uv_feat = torch.stack([u_feat_x, u_feat_y], dim=-1)  # [B,N,2]
        return proj_uv_feat


    def sample_feats_from_feat32_to_256(self, 
        feat32,                # torch.Tensor [B, C, 32, 32]
        nail_img,              # torch.Tensor or np.ndarray [B, N, 2] (x,y) in original 256-pixel coords (float)
        psize=1,               # patch size: 1 -> single token per point; odd int >1 -> psize x psize patch
        chunk_size=2048,       # process N in chunks to save memory
        align_corners=True,    # consistent coord transform for grid_sample / interpolate
        padding_mode='zeros',  # grid_sample padding_mode
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

        # normalize nail_img to tensor [B,N,2]
        if isinstance(nail_img, np.ndarray):
            nail_t = torch.from_numpy(nail_img).float().to(device)
        else:
            nail_t = nail_img.to(device)
        if nail_t.dim() == 2:
            nail_t = nail_t.unsqueeze(0)  # [1,N,2]
        B = feat32.shape[0]
        C = feat32.shape[1]
        Bn = nail_t.shape[0]
        if Bn != B:
            # allow user to pass nail_img per-batch or single-batch
            if Bn == 1 and B > 1:
                # broadcast
                nail_t = nail_t.expand(B, -1, -1).contiguous()
            else:
                raise ValueError(f"batch mismatch: feat32 batch {B} vs nail_img batch {Bn}")

        # 1) upsample feat32 -> feat256
        feat256 = F.interpolate(feat32, size=(256,256), mode='bilinear', align_corners=align_corners)  # [B,C,256,256]

        # 2) prepare normalized centers for grid_sample
        # nail_t: [B,N,2] ; expected in pixel coords where 0..255 maps to left..right/top..bottom
        Wt, Ht = 256, 256  # target dims
        # compute normalized coordinates in [-1,1] consistent with align_corners
        if align_corners:
            nx = (nail_t[..., 0] / (Wt - 1)) * 2.0 - 1.0
            ny = (nail_t[..., 1] / (Ht - 1)) * 2.0 - 1.0
        else:
            nx = ((nail_t[..., 0] + 0.5) / Wt) * 2.0 - 1.0
            ny = ((nail_t[..., 1] + 0.5) / Ht) * 2.0 - 1.0
        centers = torch.stack([nx, ny], dim=-1)  # [B,N,2] as (x_norm, y_norm)

        # 3) build delta if patch sampling
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
            delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)  # [K,2]
            K = delta.shape[0]

        # 4) chunked sampling per batch
        sampled_per_batch = []
        for b in range(B):
            feat_b = feat256[b:b+1]  # [1, C, Ht, Wt]
            centers_b = centers[b]   # [N,2]
            N = centers_b.shape[0]
            out_chunks = []
            for i in range(0, N, chunk_size):
                ci = centers_b[i:i+chunk_size]  # [M,2]
                M = ci.shape[0]
                if psize <= 1:
                    grid = ci.view(1, M, 1, 2)  # [1, M, 1, 2]
                    sampled = F.grid_sample(feat_b, grid, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)
                    # sampled: [1, C, M, 1] -> reshape
                    sampled = sampled.view(1, C, M).permute(0,2,1).contiguous()  # [1, M, C]
                    out_chunks.append(sampled)
                else:
                    # M x K grid (normalized coords)
                    grid_pts = (ci.unsqueeze(1) + delta.unsqueeze(0).to(ci.device))  # [M, K, 2]
                    # clamp to [-1,1] to avoid grid_sample error; padding_mode handles out-of-range semantics
                    grid_rs = grid_pts.clamp(-1.0, 1.0).view(1, M*K, 1, 2)  # [1, M*K, 1, 2]
                    sampled = F.grid_sample(feat_b, grid_rs, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)
                    # sampled: [1, C, M*K, 1] -> reshape to [1, C, M, K] -> permute to [1, M, K, C]
                    sampled = sampled.view(1, C, M, K).permute(0,2,3,1).contiguous()  # [1, M, K, C]
                    out_chunks.append(sampled)

            # concat chunks
            if len(out_chunks) == 0:
                # no points
                if psize <= 1:
                    out_b = torch.zeros((1, 0, C), device=feat32.device)
                else:
                    out_b = torch.zeros((1, 0, K, C), device=feat32.device)
            else:
                out_b = torch.cat(out_chunks, dim=1)  # concat along N dimension: [1, N, C] or [1, N, K, C]
            sampled_per_batch.append(out_b)

        # stack batches
        sampled_feats = torch.cat(sampled_per_batch, dim=0)  # [B, N, C] or [B, N, K, C]

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

        # if self.latent_query_points_type == "embedding":
        #     range_ = torch.arange(self.num_pcl, device=image_feats.device)
        #     x = self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1))  # [B, L, D]

        # elif self.latent_query_points_type.startswith("smplx"):
        #     x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat(
        #         (B, 1, 1)
        #     )  # [B, L, D]

        # elif self.latent_query_points_type.startswith("e2e_smplx"):
        #     # Linear warp -> MLP + LayerNorm

        #     x = self.pcl_embed(query_points)  # [B, L, D]

        x = self.transformer(
            query_points, #.to('cuda:1'), # 2,12337,1024
            cond=image_feats, #.to('cuda:1'), # 2,1024,1024
            mod=camera_embeddings,
            temb=motion_embed, #.to('cuda:1'), # 2,1024
            global_four=global_four, #.to('cuda:1'),
            nail_mask=None,
            nail_mask_3d=None,
            proj_xy=proj_xy,
            p_vis=p_vis,
            point_pos=point_pos,
            posed_pos=posed_pos,
            # nail_mask=nail_mask.to('cuda:1'),
            # nail_mask_3d=nail_mask_3d.to('cuda:1')

            # proj_xy=proj_xy.to('cuda:1'), p_vis=p_vis.to('cuda:1'), feat_hw=feat_hw,
            # is_hand=True,
        )  # [B, L, D]
        return x 

    

    def forward_moitonembed(self, motion_tokens):

        # motion_tokens = motion_tokens.mean(dim=1, keepdim=True)
        # motion_tokens, _ = motion_tokens.max(dim=1, keepdim=True)     # 2,4096,1536 --> 2,1,1536

        motion_tokens = self.motion_embed_mlp(motion_tokens).squeeze(1)  # [B, 2*D]  # one for head, one for body

        return motion_tokens


    def forward_encode_image(self, image):
        """
        Encode image and construct combined feature map (RGB + DINO) for UV mapping.
        Returns: image_feats (feature map), feature (intermediate), cls_token, combined_feats
        """
        encoder_out = self.encoder(image)   # Dinov2FusionWrapper returns tuple
        return encoder_out
        # # Handle encoder output: could be tuple (out_local, out_, out_global, out_global_four, out_upsampled)
        # # or dict for other encoders
        # if isinstance(encoder_out, tuple):
        #     # Dinov2FusionWrapper returns: (out_local, out_, out_global, out_global_four, out_upsampled)
        #     if len(encoder_out) == 5:
        #         out_local, out_, out_global, out_global_four, out_upsampled = encoder_out
        #         # out_upsampled: [B, 1024, 256, 256] — already at target resolution!
        #         image_feats = out_upsampled
        #         cls_token = out_global  # [B, D]
        #         feature = out_  # intermediate feature map
        #     else:
        #         # Fallback for other tuple returns
        #         image_feats = encoder_out[0] if isinstance(encoder_out[0], torch.Tensor) else encoder_out
        #         cls_token = encoder_out[1] if len(encoder_out) > 1 and isinstance(encoder_out[1], torch.Tensor) else None
        #         feature = image_feats
        # elif isinstance(encoder_out, dict):
        #     # Dict-based encoder (backward compatibility)
        #     image_feats = encoder_out.get('features', encoder_out.get('f_map', None))  # [B, C, H, W]
        #     cls_token = encoder_out.get('cls_token', None)  # [B, D]
        #     feature = encoder_out.get('feature', image_feats)  # intermediate pyramid feature
        # else:
        #     # Fallback: encoder_out is already the feature
        #     image_feats = encoder_out
        #     cls_token = None
        #     feature = image_feats
        
        # # Construct combined feature map: concat RGB image + DINO features
        # # image shape: [B, 3, 256, 256], image_feats shape: [B, C_dino, 256, 256]
        # if image.shape[1] == 3 and image_feats.shape[1] > 3:  # RGB image + DINO
        #     combined_feats = torch.cat([image, image_feats], dim=1)  # [B, 3+C_dino, 256, 256]
        # else:
        #     combined_feats = image_feats
        
        # # For backward compatibility, try to extract cls token if not available
        # if cls_token is None and hasattr(self.encoder, 'cls_tokens'):
        #     # Encoder might have cls_tokens as learnable parameter
        #     B = image.shape[0]
        #     cls_token = self.encoder.cls_tokens.expand(B, -1).to(image.device)  # [B, 768]
        
        # return image_feats, feature, cls_token, combined_feats


    def forward_fine_encode_image(self, image):
        image_feats = self.fine_encoder(image)   # 2,4096,1536
        return image_feats


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
        
        # Handle multi-view input: use only the first view for UV mapping
        if image.ndim == 5:  # [B, N_views, C, H, W]
            image = image[:, 0]  # Take first view: [B, C, H, W]
        elif image.ndim == 4 and image.shape[-1] == 3 and image.shape[1] != 3:
            # HWC -> CHW
            image = image.permute(0, 3, 1, 2)
        elif image.ndim == 3:  # [B, H, W] - add channel dimension
            image = image.unsqueeze(1)  # [B, 1, H, W]

        B = image.shape[0]

        # encode image
        # image_feats is cond texture
        # image_feats = self.forward_encode_image(image)  # 1,1024,1024

        # global_image_feature = self.forward_fine_encode_image(image.to('cuda'))   # bs,4096,1536
        
        # image_fine_feats, feature =self.forward_encode_image(image)    # bs,1024,1024        bs, 1024
        # image_fine_feats, feature, cls_tokens, upsample_img_feature =self.forward_encode_image(image)
        image_fine_feats, feature, cls_tokens =self.forward_encode_image(image)    # bs,1024,1024        bs, 1024
        motion_tokens = cls_tokens
        cls_four = cls_tokens
        motion_tokens = self.forward_moitonembed(cls_tokens.to('cuda:0'))    # 2,1024
        # global_vertex_feature = self.vertex_global_mapping(motion_tokens)    # 2,1024
        # global_vertex_feature = global_vertex_feature[:,None,:].expand(-1, query_points.shape[1], -1) # .unsqueeze(1).repeat(1, query_points.shape[1], 1)  # 2,12337,1024
        # global_texture_feature = feature

        # image_feats = F.pad(image_feats, (0, image_fine_feats.shape[-1] - image_feats.shape[-1], 0, 0, 0, 0))   # 1,1024,1536
        
        # merge_tokens = torch.cat((image_fine_feats, image_feats), dim=1)    # 1,5120,1536
        # motion_tokens_wrong = self.forward_moitonembed(image_fine_feats)    # 1,2048   global feature

        # motion_tokens = self.forward_moitonembed(global_image_feature.to('cuda:0'))    # 2,1024

        merge_tokens = image_fine_feats

        # _, N, C = global_image_feature.shape
        # Hf = Wf = int(N ** 0.5)
        # feat_map = global_image_feature.permute(0, 2, 1).reshape(B, C, Hf, Wf)
        # proj = nn.Linear(C * 8 * 8, 1024).to('cuda')

        # patches, patches_proj, mask_conf = self.sample_local_patches_light(
        #                                                             feat_map.to('cuda'),
        #                                                             nail_image.to('cuda'),
        #                                                             256,
        #                                                             256,
        #                                                             8,
        #                                                             proj,
        #                                                             # mask_img=mask_img
        #                                                             )  # (B,Nv,C,P,P), (B,Nv,D), (B,Nv,K)
        # # 2, 4301, 1536, 8, 8           2, 4301, 256            
        # # motion_tokens = self.forward_moitonembed(patches_proj.to('cuda:0')) 
        # # merge_tokens = patches_proj.mean(dim=1, keepdim=False).to('cuda:0')

        # assert (image_feats.shape[-1] == self.encoder_feat_dim), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"
        # assert (image_feats.shape[-1] == self.fine_encoder_feat_dim), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.fine_encoder_feat_dim}"

        # # embed camera
        # camera_embeddings = self.camera_embedder(camera)
        # assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
        #     f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        query_point = query_points
        query_points = self.pcl_embed(query_points) 

        # proj_uv_feat = self.compute_proj_uv_feat_from_original(
        #     nail_image, 256, 256, 448, 14   
        # )  # [B,N,2] in feat map pixel coords   

        patches, coords_norm, valid_mask = self.sample_local_patches_from_pyramid(feature, nail_image, patch_sizes=5)
        # point_feats, _ = self.sample_feats_from_feat32_to_256(feature, nail_image, psize=1, align_corners=True)  # 2,12337,1,1024 
        # 尝试使用 GaussianLens 的操作，把 feature map 插值到256，再采样
        # 采样的原因是：直接整数索引 — 适用于 2d像素索引 已经是整数像素且你不需要亚像素精度、也不需要 patch（或 patch 用整像素）的情况。

        # 2,12337,25,1024    表示在该点投影的像素邻域内取了 25 个子采样位置（中心 + 周围像素）
        # patches: [B, N, K_total, C]   (C is channel of each level — assumes same C across levels, or you handle projection later)
        # coords_norm: [B, N, K_total, 2] normalized coords in [-1,1]
        # valid_mask: [B, N, K_total]  boolean mask (True means sample was inside feature map before clamping)
        
        point_feats = self.aggregator(patches, coords_norm, valid_mask)  # 2,12337,1024
        # query_points = point_feats
        # query_points = self.pcl_embed(point_feats)  # 2,12337,1024
        # point_feats = patches.squeeze(2) # 做实验,尝试 patch_size=1 时，去掉多余的维度

        
        merge_tokens = self.adapter(point_feats)  # 2,12337,1024 --> 2,1024,1024
        # merge_tokens = torch.concat((point_feats, merge_tokens), dim=1)    # 2,2048,1536
        
        
        # 下面这一行是原来的对 点云的处理（映射对齐的图片做V，原始点云特征做Q，然后输出的点云特征再输入transformer）
        # query_points = self.local_attn(query_points, patches, nail_mask=self.renderer.gs_net.nail_mask) # 2,12337,1024


        # patches, _, _ = sample_local_patches_from_pyramid_debug(feature, nail_image, patch_sizes=5,
        #                  full_image_wh=(256,256), vis_image=image, vis_prefix='./hand/proj', visualize=True)
        # visualize_patches_and_attn(image, feature, proj_uv_feat, self.sample_local_patches_from_pyramid)
        # visualize_2d_projection_alignment(image[0], nail_image[0])


        # transformer generating latent points
        tokens = self.forward_transformer(
            # tokens, moe_loss = self.forward_transformer(
            merge_tokens, camera_embeddings=None, query_points=query_points, nail_mask=None, nail_mask_3d=None,
            proj_xy=nail_image, p_vis=vis_mask, feat_hw=None, motion_embed=motion_tokens, global_four=cls_four, point_pos=query_point, posed_pos=posed_points
            # proj_xy=nail_image, p_vis=vis_msk, feat_hw=(32,32), motion_embed=motion_tokens
            # image_feats, image_fine_feats, camera_embeddings=None, query_points=query_points
        ).to('cuda')

        # tokens = tokens + global_vertex_feature

        # img_combined_feats = torch.concat([image, upsample_img_feature], dim=1)  # [B, 3+C_dino, 256, 256]
        
        # img_combined_feats = upsample_img_feature
        # point_uv_feats = self.get_uvmap_features_from_image(img_combined_feats, uv_map_dict)
        point_uv_feats = {}
        
        # ===== UV Feature Fusion with Normalization & Learnable Weight =====
        if point_uv_feats.get('point_uv_features') is not None:
            uv_feats = point_uv_feats['point_uv_features']  # [B, N_pts, C]
            
            # L2 normalize UV features to prevent scale explosion
            uv_feats_norm = uv_feats / (torch.norm(uv_feats, dim=-1, keepdim=True) + 1e-8)
            
            # Fuse with learnable weight control
            # This prevents the UV features from dominating the tokens
            # tokens = self.uv_feat_fusion_weight * uv_feats_norm
            tokens = tokens + uv_feats
            # tokens = tokens + self.uv_feat_fusion_weight * uv_feats_norm
        # ===== End UV Feature Fusion =====

        # ===== Sample learnable color/opacity bias at per-point UV coords =====
        color_bias, opacity_bias = None, None
        if point_uv_feats.get('point_uv_coords') is not None:
            uv_coords = point_uv_feats['point_uv_coords']  # [B, N_pts, 2] in [0,1]

            def _sample_bias_map(bias_map, coords):
                # coords: [B, N, 2] in [0,1]; bias_map: [1, C, H, W]
                grid = coords.clone()
                grid[..., 0] = grid[..., 0] * 2.0 - 1.0
                grid[..., 1] = grid[..., 1] * 2.0 - 1.0
                grid = grid[:, None, :, :].contiguous()
                sampled = torch.nn.functional.grid_sample(
                    bias_map, grid,
                    mode='bilinear', padding_mode='border', align_corners=False
                )  # [1, C, 1, N]
                sampled = sampled.squeeze(2).permute(0, 2, 1).contiguous()  # [1, N, C]
                return sampled

            # Base learnable bias maps (initialized in finetune_model).
            color_bias_base, opacity_bias_base = None, None
            if self.color_b_map is not None:
                color_bias_base = _sample_bias_map(self.color_b_map, uv_coords)  # [1, N, 3]
            if self.opacity_b_map is not None:
                opacity_bias_base = _sample_bias_map(self.opacity_b_map, uv_coords)  # [1, N, 1]

            # Optionally refine bias maps using StyleUNet and global style (motion_tokens).
            color_bias_refined, opacity_bias_refined = None, None
            
            if (
                hasattr(self, "bias_style_unet")
                and self.bias_style_unet is not None
                and self.color_b_map is not None
                and self.opacity_b_map is not None
            ):
                # Build 4-channel bias map: [1, 4, H, W]
                bias_input = torch.cat([self.color_b_map, self.opacity_b_map], dim=1).to(tokens.device)

                # Downsample / upsample bias map to the StyleUNet resolution (uvmap_size).
                # The bias maps are initialized at 1024x1024, while StyleUNet expects
                # a fixed spatial size (256x256); without this, the internal linear
                # layer sees mismatched flattened feature size.
                if bias_input.shape[-1] != self.uvmap_size or bias_input.shape[-2] != self.uvmap_size:
                    bias_input = torch.nn.functional.interpolate(
                        bias_input,
                        size=(self.uvmap_size, self.uvmap_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Global style code from motion_tokens: [1, D]
                # Aggregate over batch to get a single style vector.
                style_code = motion_tokens#.mean(dim=0, keepdim=True).to(tokens.device)

                bias_out = self.bias_style_unet(bias_input, extra_style=style_code)  # [1, 4, H, W]
                color_b_map_refined = bias_out[:, :3]
                opacity_b_map_refined = bias_out[:, 3:4]

                color_bias_refined = _sample_bias_map(color_b_map_refined, uv_coords)
                opacity_bias_refined = _sample_bias_map(opacity_b_map_refined, uv_coords)

            # Combine base and refined biases with visibility-aware blending:
            #   - For visible points, stay closer to base biases.
            #   - For occluded points, allow refined biases (driven by global style) to dominate.
            
            color_bias_refined, opacity_bias_refined = None, None
            if color_bias_base is not None:
                if color_bias_refined is None:
                    color_bias = color_bias_base
                else:
                    if vis_mask is not None:
                        # vis_mask: [B, N]; convert to occlusion weight [1, N, 1]
                        # assume vis_mask in {0,1} or [0,1]
                        vis = vis_mask.float()
                        if vis.dim() == 1:
                            vis = vis.unsqueeze(0)
                        occ_w = (1.0 - vis).unsqueeze(-1)  # [B, N, 1]
                        # broadcast occ_w to match bias shape [1, N, 3]
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
                        occ_w = (1.0 - vis).unsqueeze(-1)  # [B, N, 1]
                        occ_w = occ_w.to(opacity_bias_base.device)
                        opacity_bias = opacity_bias_base * (1.0 - occ_w) + opacity_bias_refined * occ_w
                    else:
                        opacity_bias = opacity_bias_refined
        # ===== End bias sampling =====

        # ===== Optional UV-space visualization of occluded points on color bias map =====
        # When DEBUG_UV_OCC=1, save a PNG where occluded points (vis_mask==0) are drawn
        # on top of the current color_b_map UV texture (first batch only).
        if os.getenv("DEBUG_UV_OCC", "0") == "1":
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                if (
                    self.color_b_map is not None
                    and point_uv_feats.get("point_uv_coords") is not None
                    and vis_mask is not None
                ):
                    uv = point_uv_feats["point_uv_coords"][0].detach().cpu().numpy()  # [N,2]
                    vis_np = vis_mask[0].detach().cpu().numpy()  # [N]

                    occ_mask = vis_np < 0.5
                    if occ_mask.any():
                        occ_uv = uv[occ_mask]

                        cb = self.color_b_map[0].detach().cpu()  # [3,H,W]
                        cb_img = cb.permute(1, 2, 0).numpy()  # [H,W,3]

                        H, W, _ = cb_img.shape
                        xs = occ_uv[:, 0] * (W - 1)
                        ys = occ_uv[:, 1] * (H - 1)

                        import os as _os
                        _os.makedirs("./uv_debug", exist_ok=True)
                        save_path = _os.path.join("./uv_debug", "occ_on_color_bias.png")

                        plt.figure(figsize=(6, 6))
                        plt.imshow(np.clip(cb_img, 0.0, 1.0))
                        plt.scatter(xs, ys, s=3, c="red", alpha=0.8)
                        plt.title("Occluded points on color_b_map (UV)")
                        plt.axis("off")
                        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                        plt.close()
            except Exception as e:
                print(f"[DEBUG_UV_OCC] Failed to visualize occluded UV points: {e}")
        
        # if point_uv_feats.get('uvmap_features') is not None:
        #     point_uv_feats['uvmap_features'] = self.uv_feature_decoder(
        #         point_uv_feats['uvmap_features'], extra_style=motion_tokens
        #     )
        # tokens = tokens + point_uv_feats['point_uv_features']

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

        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # smplx_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        # kwargs: Dict, e.g., src_head_imgs

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
        )  # [B, N, C]

        # render target views

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

    # @torch.no_grad()
    def infer_single_view(
        self,
        vis_prob, 
        vis_masks,
        nail_image,
        nail_mask,
        verts_cam,
        image,
        # source_c2ws,
        # source_intrs,
        # render_c2ws,
        # render_intrs,
        batches,
        cano_pts,
        render_bg_colors,
        # smplx_params,
    ):

        # assert image.shape[0] == 1
        # num_views = 1
        
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

        # vis_msk = []
        # for i in range(len(batches)):
        #     ref_view = 0
        #     E = torch.eye(4).to('cuda')
        #     E[:3, :3] = torch.tensor(batches[i]['R'][ref_view, ...])
        #     E[:3, 3] = torch.tensor(batches[i]['T'][ref_view, ...])
        #     depth_map = self.renderer_mesh(batches[i]['world_vertex'][ref_view, ...].cuda(), 
        #                              torch.tensor(batches[i]['K'][ref_view, ...]), E, self.renderer.mano_model.mano.faces)
        #     z = verts_cam[i,:,2].reshape(-1).cpu().numpy() 

        #     # depth_map 处理
        #     depth_map = np.asarray(depth_map.cpu())
        #     H, W = depth_map.shape
        #     depth_map[np.isnan(depth_map)] = np.inf

        #     # 计算在图像范围内的点
        #     mask_inside = (z > 0)       
        #     # 有效像素索引
        #     x = nail_image[i,:,0].cpu().numpy()
        #     y = nail_image[i,:,1].cpu().numpy()
        #     x_valid = np.clip(x[mask_inside].astype(np.int32), 0, W - 1)
        #     y_valid = np.clip(y[mask_inside].astype(np.int32), 0, H - 1)

        #     # 深度采样
        #     z_buf = depth_map[y_valid, x_valid].reshape(-1)
        #     z_in = z[mask_inside].reshape(-1)

        #     # 深度比较
        #     visible_local = z_in <= (z_buf + 1e-3)

        #     # 构造 mask
        #     visible_mask = np.zeros(batches[i]['world_vertex'][ref_view, ...].shape[1], dtype=bool)
        #     visible_mask[mask_inside] = visible_local               
        #     vis_msk.append(torch.from_numpy(visible_mask).to('cuda'))
        # vis_msk = torch.stack(vis_msk, dim=0)  # B,N
        vis_msk = None

        # using DiT to predict query points features.
        # latent_points, image_feats = self.forward_latent_points(
        latent_points, global_texture_feature, bias_dict = self.forward_latent_points(
            vis_prob, nail_image, batch['uv_map'], image.permute(0, 3, 1, 2), vis_msk=vis_masks, camera=None, query_points=query_points)  # [B, N, C]      1,12337,1024     1,1024,1536
            # image.permute(0, 3, 1, 2), camera=w2c, query_points=query_points)  # [B, N, C]      1,12337,1024     1,1024,1536


        # if torch.is_tensor(query_points):
        #     qp_for_uv = query_points.detach()
        # else:
        #     qp_for_uv = query_points
        # self._attach_uv_feature_maps(
        #     batches,
        #     qp_for_uv,
        #     latent_points.detach(),
        #     texture_hw=self.uv_texture_hw,
        # )

        # B, C, N = latent_points.shape  # (1, 1024, 1024)
        # H = W = 32

        # latent_points = latent_points.view(B, C, H, W)  # (1, 1024, 32, 32)
        # # 或 latent_points = latent_points.reshape(B, C, H, W)
        # latent_points = self._sample_uv_features_from_token_maps(batches, query_points, latent_points)

        # 下面原始是需要输出 smplx_params，我取消了
        gs_model_list, gs_densify_list, query_points = self.renderer.forward_gs(
            gs_hidden_features=latent_points.to('cuda'),
            query_points=query_points,
            global_feature=global_texture_feature,
            batches=batches,
            verts_cam=verts_cam,
            nail_image=nail_image,
            color_bias=bias_dict.get("color_bias"),
            opacity_bias=bias_dict.get("opacity_bias"),
            # smplx_data=smplx_params,
            # additional_features={"image_feats": image_feats, "image": image[:, 0]},
        )

        # # render target views
        render_res_list = []
        for i in range(len(batches)):
            num_views = batches[i]['original_image'].shape[0]
            res = batches[i]['original_image'].shape[1]
            smplx_params={
                k: v.to('cuda') for k, v in batches[i]['smpl_param'].items()  # concat all smplx params
            }
            # print('num_views:', num_views)
            for view_idx in range(num_views):
                render_res = self.renderer.forward_animate_gs(
                    gs_model_list[i], 
                    gs_densify_list[i], 
                    query_points[i], 
                    # vis_mask,
                    # batches[i]['dataset_id'],
                    self.renderer.get_single_view_cam(batches[i], view_idx),
                    self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                    batches[i]['width'][0],
                    batches[i]['height'][0],
                    render_bg_colors,#[:, view_idx : view_idx + 1],
                )
                render_res_list.append(render_res)

        # for i, batch_i in enumerate(batches):
        #     smplx_params = {k: v.to('cuda') for k, v in batch_i['smpl_param'].items()}
        #     render_res = self.renderer.forward_animate_gs(
        #         gs_model_list[batch_i],
        #         gs_densify_list[batch_i],
        #         query_points[batch_i],
        #         self.renderer.get_single_view_cam(batch_i, i),          # 不切视角
        #         self.renderer.get_single_view_smpl_data(smplx_params, i),     # 含所有 view 的 poses/trans 等
        #         height=256,
        #         width=256,
        #         background_color=render_bg_colors,
        #     )        
        #     render_res_list.append(render_res)

        # ################################################################
        # for view_idx in range(num_views):
        #     render_res = self.renderer.forward_animate_gs(
        #     gs_model_list,
        #     query_points.unsqueeze(0),
        #     self.renderer.get_single_view_cam(batches, view_idx),
        #     self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
        #     256,
        #     256,
        #     render_bg_colors,#[:, view_idx : view_idx + 1],
        #     )
        #     render_res_list.append(render_res)
        # ################################################################
        # print('========', len(render_res_list)) # 2，1，3，256，256
        # print('========', render_res_list[0]['comp_mask'].shape) # 2，1，1，256，256
        # exit(0)
        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            # print(f"out key:{k}")
            if isinstance(v[0], torch.Tensor):
                if k == 'offset' or k == 'scaling' or k == 'shs':
                    # if len(v) < 8:
                    #     print(f"Warning: only {len(v)} views for {k}, less than 8")
                    # selected = [v[0], v[num_views]]  # only keep the first and last views
                    selected = [v[0]]
                    out[k] = torch.cat(selected, dim=0)
                    # out[k] = torch.concat(v, dim=0)
                    # out[k] = v
                else:
                    out[k] = torch.concat(v, dim=1)
                    if k in ["comp_rgb", "comp_mask", "comp_depth", "comp_obj"]:
                        out[k] = out[k][0].permute(
                            # 0, 1, 3, 4, 2)  # [bs, Nv, 3, H, W] -> [bs, Nv, H, W, 3]
                            0, 2, 3, 1
                        )  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
            else:
                out[k] = v
        
        # CRITICAL: Add visibility masks to output for stage2 visibility-guided training
        out['vis_masks'] = vis_masks  # [B, N] visibility mask for each point
        
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

        in_dim = 2  # p_vis, conf
        if use_depth_res:
            in_dim += 1  # depth_res

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # [b_local_raw, b_global_raw]
        )
        # 控制整体 bias 幅度的可学习系数
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

        vis = p_vis.clamp(0.0, 1.0)              # [B,L]
        # 置信度: p≈0/1 → conf≈1, p≈0.5 → conf≈0
        uncert = 4.0 * vis * (1.0 - vis)         # [B,L]
        conf = 1.0 - uncert                      # [B,L]

        feats = [vis.unsqueeze(-1), conf.unsqueeze(-1)]  # [B,L,2]

        if self.use_depth_res and depth_res is not None:
            # 简单归一化一下深度残差
            dr = depth_res
            dr = dr / (dr.mean() + 1e-6)
            dr = torch.clamp(dr, 0.0, 3.0)       # 限制到一个合理范围
            feats.append(dr.unsqueeze(-1))       # [B,L,1]

        x = torch.cat(feats, dim=-1)             # [B,L,in_dim]

        bias_raw = self.mlp(x)                   # [B,L,2]
        # 通过 tanh 限幅，再乘一个 learnable scale
        bias_normed = torch.tanh(bias_raw) * self.bias_scale  # [B,L,2]

        b_local = bias_normed[..., 0]            # [B,L]
        b_global = bias_normed[..., 1]           # [B,L]

        return b_local, b_global




# ---------- ProjCrossAttn module (multi-head, per-gauss local attn) ----------
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

        # optional small LN for stability
        self.q_ln = nn.LayerNorm(d_model)
        self.k_ln = nn.LayerNorm(d_model)
        self.v_ln = nn.LayerNorm(d_model)

        # 构造时加入（一次）
        self.nail_delta = nn.Sequential(
            nn.Linear(gauss_dim, gauss_dim//4),
            nn.GELU(),
            nn.Linear(gauss_dim//4, out_dim if out_dim is not None else gauss_dim)
        )
        self.nail_alpha = nn.Parameter(torch.tensor(0.1))  # 控制强度，可以学习


        # # optional learned small MLP for relative positional bias
        # self.pos_mlp = nn.Sequential(nn.Linear(2, self.head_dim), nn.ReLU(), nn.Linear(self.head_dim, num_heads))

    def forward(self, gauss_feat, patches, coords=None, mask=None, nail_mask=None):
        
        """
        gauss_feat: [B, N, Dg]
        patches:    [B, N, K, C_img]
        coords:     [B, N, K, 2] normalized coords (optional, for pos bias)
        mask:       [B, N, K] bool, True means valid (optional)
        returns: out [B,N,out_dim]
        """
        
        B, N, K, C = patches.shape
        q = self.q_proj(gauss_feat)  # [B,N,d_model]
        k = self.k_proj(patches)     # [B,N,K,d_model]
        v = self.v_proj(patches)     # [B,N,K,d_model]

        q = self.q_ln(q)
        k = self.k_ln(k)
        v = self.v_ln(v)

        # reshape for heads: q->[B,heads,N,hd], k->[B,heads,N,K,hd]
        qh = q.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B,heads,N,hd]
        kh = k.view(B, N, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)  # [B,heads,N,K,hd]
        vh = v.view(B, N, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)  # [B,heads,N,K,hd]

        # attention logits: batch matmul across head/gauss
        logits = torch.einsum('bhnd,bhnkd->bh nk', qh, kh)  # [B,heads,N,K]
        logits = logits / math.sqrt(self.head_dim)

        if coords is not None:
            # coords normalized in [-1,1] differences; compute relative pos bias per token
            # coords: [B,N,K,2] -> feed to pos_mlp (flatten)
            rel = coords.view(B*N*K, 2)
            bias = self.pos_mlp(rel)  # [B*N*K, heads]
            bias = bias.view(B, N, K, self.num_heads).permute(0,3,1,2)  # [B,heads,N,K]
            logits = logits + bias

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = torch.softmax(logits, dim=-1)  # [B,heads,N,K]
        out = torch.einsum('bh nk, bhnkd->bhnd', attn, vh)  # [B,heads,N,hd]
        out = out.permute(0,2,1,3).contiguous().view(B, N, -1)  # [B,N,d_model]
        out = self.out(out)  # [B,N,out_dim]

        if nail_mask is not None:
            # nail_mask: [B,N] bool
            delta = self.nail_delta(gauss_feat)    # [B,N,out_dim]
            # 仅对 nail 点生效
            mask_f = nail_mask.float().unsqueeze(-1)  # [B,N,1]
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

        # learnable queries (L, D)
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)   # 1024，1024

        # multi-head attention: queries shape (L, B, D), keys/vals (N, B, D)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False)

        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(dim)

        # optional small MLP to refine tokens
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

        # prepare queries (L, B, D)
        q = self.query_tokens.unsqueeze(1).expand(-1, B, -1).contiguous()  # (L, B, D)

        # keys/values must be (N, B, D) for nn.MultiheadAttention (batch_first=False)
        kv = point_feats.permute(1, 0, 2).contiguous()  # (N, B, D)

        # build attn mask if provided: nn.MultiheadAttention accepts key_padding_mask of shape (B, N) with True for positions that should be ignored.
        key_padding_mask = None
        if valid_mask is not None:
            # valid_mask: (B, N) True = valid -> key_padding_mask expects True for positions to be ignored,
            # so we invert
            key_padding_mask = ~valid_mask  # type: ignore
            # ensure dtype=bool
            key_padding_mask = key_padding_mask.bool()

        # run cross-attention: queries attend to keys/vals
        # attn_output: (L, B, D)
        attn_output, _ = self.mha(q, kv, kv, key_padding_mask=key_padding_mask)

        # transpose to (B, L, D)
        tokens = attn_output.permute(1, 0, 2).contiguous()

        if self.use_ln:
            tokens = self.ln(tokens)

        # small MLP residual
        mlp_out = self.mlp(tokens)
        tokens = tokens + self.mlp_ln(mlp_out)

        return tokens



# ---- helper: multi-head attention implemented manually so we can add pos-bias ----
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

        q_lin = self.q_proj(q).view(B, Lq, self.nhead, self.head_dim).permute(0,2,1,3)  # [B, H, Lq, Hd]
        k_lin = self.k_proj(k).view(B, Lk, self.nhead, self.head_dim).permute(0,2,1,3)  # [B, H, Lk, Hd]
        v_lin = self.v_proj(v).view(B, Lk, self.nhead, self.head_dim).permute(0,2,1,3)  # [B, H, Lk, Hd]

        # scaled dot-product
        # logits: [B, H, Lq, Lk]
        logits = torch.einsum('bhqd,bhkd->bhqk', q_lin, k_lin) / (self.head_dim ** 0.5)

        # add pos bias if provided
        if pos_bias is not None:
            # pos_bias expected [B, H, Lq, Lk] or broadcastable
            logits = logits + pos_bias

        # handle key padding mask: expect key_padding_mask shape [B, Lk] boolean (True = valid)
        if key_padding_mask is not None:
            # convert to mask where True = keep, False = mask -> we want to set masked positions to -inf
            # key_padding_mask might be bool with True = valid; ensure that
            kp = key_padding_mask
            if kp.dtype != torch.bool:
                kp = kp.bool()
            # invert: False means masked -> set -inf
            # broadcast to [B,1,1,Lk]
            mask = (~kp).unsqueeze(1).unsqueeze(1)  # True where to mask
            logits = logits.masked_fill(mask, float('-1e9'))

        attn = torch.softmax(logits, dim=-1)  # [B,H,Lq,Lk]
        attn = self.dropout(attn)

        out_h = torch.einsum('bhqk,bhkd->bhqd', attn, v_lin)  # [B,H,Lq,Hd]
        out = out_h.permute(0,2,1,3).contiguous().view(B, Lq, D)  # [B,Lq,D]
        out = self.out_proj(out)
        return out, attn


# ---- Q-Former with stacked blocks: cross-attn -> self-attn -> FFN ----
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
        use_pos_bias: bool = True,   # whether to use relative pos bias between queries and points
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

        # learnable query tokens (L, D)
        self.query_tokens = nn.Parameter(torch.randn(self.L, dim) * 0.02)

        # optional learnable 2D positions for queries (used to compute relative pos w.r.t. points)
        if use_pos_bias:
            self.query_pos = nn.Parameter(torch.randn(self.L, 2) * 0.01)  # small init
            # pos_mlp maps delta (dx,dy) -> per-head bias values
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, pos_hidden),
                nn.GELU(),
                nn.Linear(pos_hidden, num_heads)
            )

        # build stacks of blocks
        self.cross_attns = nn.ModuleList([ MultiHeadAttentionCustom(dim, num_heads, dropout=dropout) for _ in range(num_layers) ])
        self.self_attns  = nn.ModuleList([ MultiHeadAttentionCustom(dim, num_heads, dropout=dropout) for _ in range(num_layers) ])
        self.ffns        = nn.ModuleList([ nn.Sequential(
                                            nn.Linear(dim, dim * ffn_mult),
                                            nn.GELU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(dim * ffn_mult, dim),
                                            nn.Dropout(dropout)
                                          ) for _ in range(num_layers) ])

        # LayerNorms
        self.cross_ln = nn.ModuleList([ nn.LayerNorm(dim) for _ in range(num_layers) ])
        self.self_ln  = nn.ModuleList([ nn.LayerNorm(dim) for _ in range(num_layers) ])
        self.ffn_ln   = nn.ModuleList([ nn.LayerNorm(dim) for _ in range(num_layers) ])

        # small residual MLP at the end optionally (compatibility with previous adapter)
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

        # prepare queries: [B, L, D]
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B,L,D]

        # build key_padding_mask expected as boolean where True = valid -> our MultiHead uses True = valid
        kp = None
        if valid_mask is not None:
            kp = valid_mask.bool()  # [B,N]

        # precompute pos_bias if requested and point_coords given
        # pos_bias: [B, num_heads, L, N] or None
        pos_bias = None
        if self.use_pos_bias and point_coords is not None:
            # query_pos: [L,2] -> expand to [B,L,2]; point_coords: [B,N,2]
            # compute delta = query_pos - point_coords for each pair -> [B,L,N,2]
            qp = self.query_pos.unsqueeze(0).expand(B, -1, -1)  # [B,L,2]
            pc = point_coords.unsqueeze(1)                      # [B,1,N,2]
            delta = qp.unsqueeze(2) - pc                       # [B,L,N,2]
            # feed delta to pos_mlp in flattened form
            delta_flat = delta.view(B * self.L * N, 2)
            bias_flat = self.pos_mlp(delta_flat)                # [B*L*N, num_heads]
            bias = bias_flat.view(B, self.L, N, self.num_heads) # [B,L,N,H]
            # permute to [B,H,L,N]
            pos_bias = bias.permute(0,3,1,2).contiguous()

        # stacked blocks
        for i in range(self.num_layers):


            # --- cross-attn: queries attend to point_feats (keys/vals)
            cross_attn = self.cross_attns[i]
            out_cross, attn_map = cross_attn(q, point_feats, point_feats, key_padding_mask=kp, pos_bias=pos_bias)
            # residual + ln
            q = q + out_cross
            q = self.cross_ln[i](q)

            # --- self-attn on queries
            self_attn = self.self_attns[i]
            out_self, self_attn_map = self_attn(q, q, q, key_padding_mask=None, pos_bias=None)
            q = q + out_self
            q = self.self_ln[i](q)

            # --- FFN
            ffn = self.ffns[i]
            ffn_out = ffn(q)
            q = q + ffn_out
            q = self.ffn_ln[i](q)


            # # --- self-attn on queries
            # self_attn = self.self_attns[i]
            # out_self, self_attn_map = self_attn(q, q, q, key_padding_mask=None, pos_bias=None)
            # q = q + out_self
            # q = self.self_ln[i](q)
                    
            # # --- cross-attn: queries attend to point_feats (keys/vals)
            # cross_attn = self.cross_attns[i]
            # out_cross, attn_map = cross_attn(q, point_feats, point_feats, key_padding_mask=kp, pos_bias=pos_bias)
            # # residual + ln
            # q = q + out_cross
            # q = self.cross_ln[i](q)

            # # --- FFN
            # ffn = self.ffns[i]
            # ffn_out = ffn(q)
            # q = q + ffn_out
            # q = self.ffn_ln[i](q)


        # final normalization
        tokens = self.out_ln(q)  # [B,L,D]
        return tokens



class PatchAggregatorWithPosWeightedPool(nn.Module):
    def __init__(self, C_in, D_hidden=256, out_dim=None):
        super().__init__()
        # self.pos_mlp = nn.Sequential(
        #     nn.Linear(2, D_hidden),
        #     nn.ReLU(),
        #     nn.Linear(D_hidden, 1)  # scalar logit per patch
        # )
        self.out_proj = None
        if out_dim is not None:
            self.out_proj = nn.Linear(C_in, out_dim)

    def forward(self, patches, coords_norm, valid_mask):
        # patches: (B,N,K,C), coords_norm: (B,N,K
        # \,2), valid_mask: (B,N,K)
        B, N, K, C = patches.shape
        device = patches.device

        # pos_logits = self.pos_mlp(coords_norm.view(B*N*K, 2)).view(B, N, K)  # (B,N,K)

        ####################################################################3
        # 1) 从 coords_norm 恢复每个点的“几何中心”（归一化坐标）
        center_norm = coords_norm.mean(dim=2)                     # [B, N, 2]
        point_coords_norm = center_norm.unsqueeze(2).expand(-1, -1, K, -1)  # [B, N, K, 2]
        
        # 2) 计算每个 patch 与该点中心的归一化距离
        dist = torch.norm(coords_norm - point_coords_norm, dim=-1)  # [B, N, K]

        # 3) 距离 → logit（RBF 风格的几何权重），掩码无效 patch
        pos_logits = - dist / 0.1  # 控制距离影响范围的超参数 sigma=0.1       
        ####################################################################

        # mask out invalid by large negative
        pos_logits = pos_logits.masked_fill(~valid_mask, float('-1e9'))
        weights = torch.softmax(pos_logits, dim=2)  # (B,N,K)

        # weighted sum
        weighted = (patches * weights.unsqueeze(-1)).sum(dim=2)  # (B,N,C)
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

        # stable normalization + small learnable residual scale
        self.out_ln = nn.LayerNorm(out_dim, eps=1e-6)
        # 儒学初始为小值，便于 warmup
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # zero-init out_proj so module starts near-zero (safe)
        nn.init.constant_(self.out_proj.weight, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        # optional running target stats (per-dim)
        self.use_running_stats = use_running_stats
        self.register_buffer('running_mean', torch.zeros(out_dim))
        self.register_buffer('running_std', torch.ones(out_dim))
        self.register_buffer('running_count', torch.tensor(0, dtype=torch.long))
        self.momentum = 0.01
        self.eps = eps

    def _compute_stats(self, tensor):
        # tensor: [B, N, D]
        # compute per-dim mean/std over batch and tokens -> shape [1,1,D]
        # use unbiased=False for stability
        mean = tensor.mean(dim=(0,1), keepdim=True)   # [1,1,D]
        std = tensor.std(dim=(0,1), unbiased=False, keepdim=True)  # [1,1,D]
        return mean, std

    def _update_running(self, batch_mean, batch_std):
        # batch_mean, batch_std: [1,1,D] -> squeeze to [D]
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
        # linear proj
        q = self.q_proj(gauss_feat)           # [B, M, Dout]
        k = self.k_proj(patches)              # [B, M, K, Dout]
        v = self.v_proj(patches)

        # reshape to heads
        qh = q.view(B, M, self.num_heads, self.head_dim).permute(0,2,1,3)         # [B, H, M, Hd]
        kh = k.view(B, M, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)    # [B, H, M, K, Hd]
        vh = v.view(B, M, K, self.num_heads, self.head_dim).permute(0,3,1,2,4)    # [B, H, M, K, Hd]

        logits = torch.einsum('bhmd,bhmkd->bhmk', qh, kh) / math.sqrt(self.head_dim)  # [B,H,M,K]

        if mask is not None:
            # mask: [B, M, K], True means valid -> we mask invalid by -inf
            logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = torch.softmax(logits, dim=-1)  # per-point local softmax
        out_h = torch.einsum('bhmk,bhmkd->bhmd', attn, vh)  # [B,H,M,Hd]
        out = out_h.permute(0,2,1,3).contiguous().view(B, M, -1)  # [B, M, Dout]

        out = self.out_proj(out)   # zero-init => initially ~0
        out = self.out_ln(out)     # normalize per-dim

        # === Moment matching: make out have similar mean/std to ref_tokens if provided ===
        if ref_tokens is not None:
            # ref_tokens: [B, N_ref, Dout] (Dout must equal self.out_dim)
            assert ref_tokens.shape[-1] == out.shape[-1], "ref token dim mismatch"

            # compute batch stats for reference and for out
            ref_mean, ref_std = self._compute_stats(ref_tokens)  # [1,1,D]
            out_mean, out_std = self._compute_stats(out)

            # optionally update running stats (for inference smoothing)
            if self.use_running_stats and self.training:
                self._update_running(ref_mean, ref_std)

            # normalize out and re-scale to match ref
            out_normed = (out - out_mean) / (out_std + self.eps)
            matched = out_normed * (ref_std + self.eps) + ref_mean
            final = gauss_feat + self.alpha * matched
            return final
        else:
            # fallback: use running stats if available to scale, else just residual
            if self.use_running_stats and self.running_count > 0:
                running_mean = self.running_mean.view(1,1,-1)
                running_std = self.running_std.view(1,1,-1)
                out_normed = (out - out.mean(dim=(0,1), keepdim=True)) / (out.std(dim=(0,1), keepdim=True) + self.eps)
                matched = out_normed * (running_std + self.eps) + running_mean
                final = gauss_feat + self.alpha * matched
                return final
            else:
                return gauss_feat + self.alpha * out
            



# ---------- helper: map original pixel coords -> feature-grid coords ----------
def compute_proj_uv_feat_from_original(verts_xy_orig, H0, W0, resolution, patch_size):
    """
    verts_xy_orig: [B, N, 2] float (x horizontal, y vertical) in original image pixels (0..W0-1, 0..H0-1)
    returns: proj_uv_feat: [B, N, 2] floats in feature-grid coords (x in [0, Wf-1], y in [0, Hf-1])
    NOTE: This follows Dinov2FusionWrapper._preprocess_image pad->resize (align_corners=True in your code).
    """
    device = verts_xy_orig.device
    max_size = max(H0, W0)
    pad_left = (max_size - W0) // 2
    pad_top  = (max_size - H0) // 2

    x0 = verts_xy_orig[..., 0]
    y0 = verts_xy_orig[..., 1]
    x_pad = x0 + pad_left
    y_pad = y0 + pad_top

    scale = float(resolution) / float(max_size)  # resize scale
    x_resized = x_pad * scale
    y_resized = y_pad * scale

    u_feat_x = x_resized / float(patch_size)
    u_feat_y = y_resized / float(patch_size)
    proj_uv_feat = torch.stack([u_feat_x, u_feat_y], dim=-1)
    return proj_uv_feat  # [B,N,2]

# ---------- helper: convert possibly-flattened token -> [B,C,Hf,Wf] safely ----------
def ensure_feature_map(feats, B, C_target=None, Hf=None, Wf=None):
    """
    feats: possibly one of
        - [B, C, Hf, Wf]  (already correct)
        - [B, N, C]       (flatten tokens N=Hf*Wf)
        - [B, C, N]       (flatten channels last)
        - [B, C*Hf*Wf]    (rare)
    You must provide either (Hf,Wf) or let function infer from N if possible.
    Returns: feature_map [B, C, Hf, Wf]
    """
    t = feats
    if isinstance(t, torch.Tensor):
        shape = list(t.shape)
    else:
        raise ValueError("feats must be a torch.Tensor")
    if len(shape) == 4:
        # assume [B,C,Hf,Wf]
        return t
    B_in = shape[0]
    if len(shape) == 3:
        # could be [B,N,C] or [B,C,N]
        if shape[1] == (Hf * Wf if Hf is not None else None):
            # ambiguous; prefer [B, N, C]
            if Hf is None or Wf is None:
                # try square N
                N = shape[1]
                s = int(math.sqrt(N))
                Hf = Hf or s
                Wf = Wf or s
            C = shape[2]
            return t.permute(0,2,1).reshape(B_in, C, Hf, Wf)
        else:
            # assume [B, N, C]
            N = shape[1]
            if Hf is None or Wf is None:
                s = int(math.sqrt(N))
                Hf = Hf or s
                Wf = Wf or s
            C = shape[2]
            return t.permute(0,2,1).reshape(B_in, C, Hf, Wf)
    elif len(shape) == 2:
        # [B, C*N] or [B, (C*H*W)] - need Hf,Wf
        if Hf is None or Wf is None or C_target is None:
            raise ValueError("Provide Hf,Wf and C_target for 2D flattened features")
        return t.reshape(B_in, C_target, Hf, Wf)
    else:
        raise ValueError(f"Unsupported feats shape {shape}")
    

def visualize_patches_and_attn(img, feat_pyramid, proj_uv_full, sample_fn,
                               module=None, gauss_feats=None, selected_idx=None,
                               patch_sizes=5, full_image_wh=(256,256), max_show=8, device='cuda'):
    """
    img: numpy array HxWx3 (RGB)
    feat_pyramid: list of torch tensors [B,C,Hf,Wf] or single tensor
    proj_uv_full: torch tensor [B, N, 2] pixel coords (float)
    sample_fn: your sample_local_patches_from_pyramid function
    module: if provided, should be ProjCrossAttn instance so we can call it to get attn
    gauss_feats: [B,N,D] gauss features (optional), used with module to compute attn
    selected_idx: list of indices (N indices) to visualize; if None choose random
    """
    img = img[0].permute(1,2,0).cpu().numpy()
    proj_uv_full = proj_uv_full[0].unsqueeze(0)
    feat_pyramid = feat_pyramid[0].unsqueeze(0)

    B, N, _ = proj_uv_full.shape
    assert B == 1, "visualize single batch for now"
    proj_uv = proj_uv_full.to(device)

    patches, coords_norm, valid_mask = sample_fn(feat_pyramid, proj_uv, patch_sizes=patch_sizes, full_image_wh=full_image_wh)

    # patches: [B,N,K,C]
    patches = patches.detach().cpu().numpy()[0]   # [N, K, C]
    valid_mask = valid_mask.cpu().numpy()[0]
    coords_norm = coords_norm.cpu().numpy()[0]  # [N, K, 2]

    # choose indices
    if selected_idx is None:
        rng = np.random.default_rng(seed=42)
        selected_idx = rng.choice(N, size=min(max_show, N), replace=False)
    else:
        selected_idx = selected_idx[:max_show]

    # if module and gauss_feats given, get attn
    attn_vis = None
    if module is not None and gauss_feats is not None:
        module.eval()
        with torch.no_grad():
            out, attn = module(gauss_feats.to(device), torch.from_numpy(patches).unsqueeze(0).to(device).float(), return_attn=True)
            # Note: above expects patches shape [B,N,K,C] -> our patches is [N,K,C]
            # you may need to adapt if shapes differ
            attn_vis = attn.cpu().numpy()[0]  # [N,K]

    H, W, _ = img.shape
    fig = plt.figure(figsize=(12, 4 * len(selected_idx)))
    for i, idx in enumerate(selected_idx):
        ax = fig.add_subplot(len(selected_idx), 3, i*3 + 1)
        # show original image with projected point
        ax.imshow(img)
        x, y = proj_uv_full[0, idx].cpu().numpy()
        ax.scatter([x], [y], c='r', s=40)
        ax.set_title(f'proj idx {idx}')
        ax.axis('off')

        # show center patch preview: pick K center element index (K_total = psize*psize*levels)
        patch = patches[idx]  # [K, C]
        K_total, C = patch.shape
        # reshape each patch token into small image if C fits (e.g. sampled from conv features channels)
        # often C>3; for visualization, if C>=3 take first 3 dims and normalize
        # here we'll show the central token (middle of K)
        center_token = patch[K_total//2]
        if center_token.shape[0] >= 3:
            token_img = center_token[:3]
            # normalize for display
            token_img = (token_img - token_img.min()) / (token_img.max() - token_img.min() + 1e-8)
            # make small color patch
            token_img_disp = np.tile(token_img.reshape(1,1,3), (32,32,1))
            ax2 = fig.add_subplot(len(selected_idx), 3, i*3 + 2)
            ax2.imshow(token_img_disp)
            ax2.set_title('center token approx')
            ax2.axis('off')
        else:
            ax2 = fig.add_subplot(len(selected_idx), 3, i*3 + 2)
            ax2.text(0.1, 0.5, "C<3, can't display", fontsize=12)
            ax2.axis('off')

        # show a grid of first up to 9 sampled tokens as color proxies
        ax3 = fig.add_subplot(len(selected_idx), 3, i*3 + 3)
        show_n = min(9, K_total)
        grid = np.zeros((32*3, 32*3, 3), dtype=np.float32)
        for j in range(show_n):
            r = j // 3
            c = j % 3
            t = patch[j]
            if t.shape[0] >= 3:
                timg = (t[:3] - t[:3].min()) / (t[:3].max() - t[:3].min() + 1e-8)
                grid[r*32:(r+1)*32, c*32:(c+1)*32, :] = np.tile(timg.reshape(1,1,3), (32,32,1))
        ax3.imshow(grid)
        if attn_vis is not None:
            # overlay attention: reshape attn per token into small heatmap (if tokens came from psize*psize)
            # here we simply show the K attention values as a bar
            att_vec = attn_vis[idx]  # [K]
            ax3.set_title(f'patch tokens (attn sum {att_vec.sum():.3f})')
        else:
            ax3.set_title('patch tokens (no attn)')
        ax3.axis('off')

    plt.tight_layout()
    plt.show()
    print('1')


# ---------- helper: overlay attention heatmap on resized image ----------
def overlay_attn_on_image(img_resized_uint8, x_resized, y_resized, attn_vec, patch_grid_size, up=16, alpha=0.6, cmap='jet'):
    """
    img_resized_uint8: HxWx3 uint8 (Dinov2 preprocessed image)
    x_resized,y_resized: floats (pixel coords on resized image)
    attn_vec: 1D numpy array length K == patch_grid_size*patch_grid_size
    patch_grid_size: int, e.g., 5
    up: upsample factor per token -> pixels for display
    returns: overlay image uint8
    """
    H, W, _ = img_resized_uint8.shape
    K = len(attn_vec)
    p = patch_grid_size
    assert K == p*p, "K must be p*p"

    # build small grid
    attn_grid = attn_vec.reshape(p, p)
    # normalize
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    # upsample
    heat = np.kron(attn_grid, np.ones((up, up)))
    # colormap
    cmap_obj = plt.get_cmap(cmap)
    heat_color = cmap_obj(heat)[:, :, :3]  # HxW x 3
    hx, hy = heat_color.shape[1], heat_color.shape[0]
    x0 = int(round(x_resized)) - hx // 2
    y0 = int(round(y_resized)) - hy // 2

    canvas = img_resized_uint8.astype(np.float32) / 255.0
    x1, x2 = max(0, x0), min(W, x0 + hx)
    y1, y2 = max(0, y0), min(H, y0 + hy)
    hx1, hx2 = x1 - x0, x2 - x0
    hy1, hy2 = y1 - y0, y2 - y0
    if x1 >= x2 or y1 >= y2:
        return img_resized_uint8  # nothing to overlay (out of image)
    canvas[y1:y2, x1:x2] = (1-alpha) * canvas[y1:y2, x1:x2] + alpha * heat_color[hy1:hy2, hx1:hx2]
    out = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    return out

# ---------- main visualize function ----------
def visualize_alignment(
    image,                      # torch tensor [B,3,H0,W0], range [0,1]
    verts_xy_orig,              # torch tensor [B,N,2] original pixel coords (x,y)
    dinov2_wrapper,             # your Dinov2FusionWrapper instance (has .resolution and .model.patch_size and _preprocess_image)
    out_local_or_flat,          # feature map OR flattened tokens (see ensure_feature_map)
    sample_local_patches_fn,    # your sample_local_patches_from_pyramid function
    proj_cross_attn_module,     # your ProjCrossAttn that supports return_attn=True
    gauss_feat,                 # [B,N,Dg]
    nail_mask=None,
    select_indices=None,        # list of gauss indices to visualize (per batch element)
    patch_sizes=5,              # patch size in feature-grid (psize)
    save_prefix='vis',
    device='cpu'
):
    """
    Produces visualizations: for each selected gauss -> overlay attn on resized image, show patch tokens grid.
    Saves files: {save_prefix}_batch{b}_gauss{i}_overlay.png and a figure combining patches.
    """
    image = image.to(device)
    verts_xy_orig = verts_xy_orig.to(device)
    gauss_feat = gauss_feat.to(device)
    dinov2_wrapper_device = next(dinov2_wrapper.parameters()).device if any(True for _ in dinov2_wrapper.parameters()) else device

    B = image.shape[0]
    H0, W0 = image.shape[2], image.shape[3]
    resolution = dinov2_wrapper.resolution
    patch_size = dinov2_wrapper.model.patch_size
    Wf = resolution // patch_size
    Hf = resolution // patch_size

    # ensure out_local is [B,C,Hf,Wf]
    out_local = ensure_feature_map(out_local_or_flat, B, C_target=None, Hf=Hf, Wf=Wf)
    out_local = out_local.to(device)

    # 1) compute proj uv (feature-grid coords)
    proj_uv_feat = compute_proj_uv_feat_from_original(verts_xy_orig, H0, W0, resolution, patch_size)

    # 2) sample patches
    patches, coords_norm, valid_mask = sample_local_patches_fn(
        out_local, proj_uv_feat, patch_sizes=patch_sizes, full_image_wh=(Wf, Hf), align_corners=True
    )
    # shapes: patches [B,N,K,C], coords_norm [B,N,K,2], valid_mask [B,N,K]
    print("patches.shape:", patches.shape, "coords_norm.shape:", coords_norm.shape)

    # 3) run proj_cross_attn to get attn
    proj_cross_attn_module.eval()
    with torch.no_grad():
        out_gauss, attn = proj_cross_attn_module(gauss_feat.to(device), patches.to(device), coords=coords_norm.to(device), mask=valid_mask.to(device), nail_mask=nail_mask, return_attn=True)
    # attn: [B, N, K]
    print("attn.shape:", attn.shape)

    # 4) get resized image (Dinov2 sees) for overlay
    with torch.no_grad():
        img_resized = dinov2_wrapper._preprocess_image(image.clone(), dinov2_wrapper.resolution)  # [B,3,res,res]
    # convert first batch image to uint8
    for b in range(B):
        img_r = img_resized[b].cpu().permute(1,2,0).numpy()  # HxWx3, floats [0,1]
        img_r_uint8 = (np.clip(img_r,0,1) * 255).astype(np.uint8)
        N = patches.shape[1]
        K = patches.shape[2]
        pgrid = int(math.sqrt(K))
        if select_indices is None:
            sel = list(range(min(6, N)))
        else:
            sel = select_indices if isinstance(select_indices, (list,tuple)) else [select_indices]

        for i_idx, i in enumerate(sel):
            # compute resized pixel coords for overlay center
            u_feat_x = proj_uv_feat[b, i, 0].item()
            u_feat_y = proj_uv_feat[b, i, 1].item()
            x_resized = u_feat_x * patch_size
            y_resized = u_feat_y * patch_size

            attn_vec = attn[b, i].cpu().numpy()  # length K
            overlay_img = overlay_attn_on_image(img_r_uint8, x_resized, y_resized, attn_vec, pgrid, up=16, alpha=0.6)
            fname = f"{save_prefix}_batch{b}_gauss{i}_overlay.png"
            plt.imsave(fname, overlay_img)
            print("saved overlay:", fname)

            # plot patch token grid (first up to 9 tokens shown as color proxies)
            patch_tokens = patches[b, i].cpu().numpy()  # [K, C]
            C = patch_tokens.shape[1]
            show_n = min(9, K)
            grid_img = np.zeros((32*3, 32*3, 3), dtype=np.float32)
            for j in range(show_n):
                r = j // 3
                c = j % 3
                tok = patch_tokens[j]
                if C >= 3:
                    timg = tok[:3]
                    timg = (timg - timg.min()) / (timg.max() - timg.min() + 1e-8)
                    grid_img[r*32:(r+1)*32, c*32:(c+1)*32, :] = np.tile(timg.reshape(1,1,3), (32,32,1))
            fname2 = f"{save_prefix}_batch{b}_gauss{i}_patchgrid.png"
            plt.imsave(fname2, np.clip(grid_img,0,1))
            print("saved patch grid:", fname2)

    print("visualize_alignment done. Inspect saved images.")


def visualize_2d_projection_alignment(
    image,              # [H,W,3] numpy or [3,H,W] tensor
    verts_img,          # [N,2]  np or tensor，像素坐标 (u,v)
    mask=None,          # [N] bool，可选（比如 nail mask）
    save_path='./hand/proj.jpg',     # str，可选
    title="Projection Alignment Check",
    point_size=5,
):
    """
    可视化 3D->2D 投影的像素坐标是否与图像对齐
    """

    # 1. 转换格式
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1,2,0).detach().cpu().numpy()
        image = np.clip(image, 0, 1)

    if isinstance(verts_img, torch.Tensor):
        verts_img = verts_img.detach().cpu().numpy()

    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)

    H, W = image.shape[:2]

    # 2. 绘图
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.title(title)

    if mask is None:
        plt.scatter(verts_img[:,0], verts_img[:,1], s=point_size, c='r', alpha=0.6, label="projected pts")
    else:
        plt.scatter(verts_img[~mask,0], verts_img[~mask,1], s=point_size, c='b', alpha=0.3, label="non-nail")
        plt.scatter(verts_img[mask,0], verts_img[mask,1], s=point_size, c='r', alpha=0.8, label="nail")

    plt.legend()
    plt.xlim([0, W])
    plt.ylim([H, 0])   # y轴反转对齐图像坐标
    plt.tight_layout()

    # 3. 保存 or 显示
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"✅ Saved projection visualization to: {save_path}")
    else:
        plt.show()

    plt.close()




def sample_local_patches_from_pyramid_debug(
    feat_pyramid,
    proj_uv_full,
    patch_sizes,
    full_image_wh=(256, 256),
    align_corners=True,
    vis_image=None,         # ✅ 原始输入图像 [H,W,3]，用于可视化
    vis_prefix=None,        # ✅ 保存路径前缀，如 "debug_vis/sample"
    visualize=False,        # ✅ 是否启用可视化
):
    """
    与原始 sample_local_patches_from_pyramid 功能完全相同，
    但增加了 2D 对齐可视化。
    """
    # ----------- 1. 预处理输入 ------------
    if isinstance(feat_pyramid, (list, tuple)):
        feat_list = list(feat_pyramid)
    else:
        feat_list = [feat_pyramid]

    B, N, _ = proj_uv_full.shape
    W_full, H_full = full_image_wh

    if isinstance(patch_sizes, int):
        patch_sizes = [patch_sizes] * len(feat_list)

    device = proj_uv_full.device

    patches_per_level = []
    coords_per_level = []
    masks_per_level = []

    # ----------- 2. 遍历每个特征层 ------------
    for lvl, (P, psize) in enumerate(zip(feat_list, patch_sizes)):
        Bf, C, Hf, Wf = P.shape
        assert Bf == B, "Batch mismatch"

        # === scale 投影坐标到该 level ===
        u_feat_x = proj_uv_full[..., 0] * (Wf / float(W_full))
        u_feat_y = proj_uv_full[..., 1] * (Hf / float(H_full))

        if align_corners:
            nx = (u_feat_x / (Wf - 1)) * 2.0 - 1.0
            ny = (u_feat_y / (Hf - 1)) * 2.0 - 1.0
        else:
            nx = ((u_feat_x + 0.5) / Wf) * 2.0 - 1.0
            ny = ((u_feat_y + 0.5) / Hf) * 2.0 - 1.0

        xs = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() / (Wf - 1)) * 2.0
        ys = (torch.arange(-(psize//2), psize//2 + 1, device=device).float() / (Hf - 1)) * 2.0
        dx, dy = torch.meshgrid(xs, ys, indexing='xy')
        delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)
        k = delta.shape[0]

        center = torch.stack([nx, ny], dim=-1)
        grid = center.unsqueeze(2) + delta.view(1, 1, k, 2)
        valid_mask_lvl = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & \
                         (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
        grid_clamped = grid.clamp(-1.0, 1.0)
        grid_rs = grid_clamped.view(B, N * k, 2).unsqueeze(2)

        sampled = F.grid_sample(P, grid_rs, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
        sampled = sampled.view(B, C, N, k).permute(0, 2, 3, 1).contiguous()

        patches_per_level.append(sampled)
        coords_per_level.append(grid)
        masks_per_level.append(valid_mask_lvl)

        # ----------- 3. ✅ 可视化部分（新增）------------
        if visualize and vis_image is not None:
            with torch.no_grad():
                B = vis_image.shape[0]
                centers_lvl = center.detach().cpu().numpy()  # [B, N, 2] in [-1,1]
                H_full, W_full = vis_image.shape[-2], vis_image.shape[-1]

                for b in range(B):
                    centers_b = centers_lvl[b]  # [N,2]
                    
                    # ✅ 直接映射到原图像素坐标，而不是特征图
                    u_pix_full = ((centers_b[:,0] + 1)/2) * (W_full - 1)
                    v_pix_full = ((centers_b[:,1] + 1)/2) * (H_full - 1)

                    img_b = vis_image[b]
                    if torch.is_tensor(img_b):
                        vis = img_b.detach().cpu().permute(1, 2, 0).numpy()
                    else:
                        vis = img_b.copy()
                    vis = np.clip(vis, 0, 1)

                    plt.figure(figsize=(6, 6))
                    plt.imshow(vis)
                    plt.scatter(u_pix_full, v_pix_full, s=5, c='r', alpha=0.6)
                    plt.title(f"Batch {b} | Level {lvl} centers projected to full image")
                    plt.xlim([0, W_full])
                    plt.ylim([H_full, 0])
                    plt.tight_layout()

                    if vis_prefix is not None:
                        os.makedirs(os.path.dirname(vis_prefix), exist_ok=True)
                        fname = f"{vis_prefix}_level{lvl}_batch{b}.png"
                        plt.savefig(fname, dpi=200)
                        print(f"✅ Saved visualization: {fname}")
                    else:
                        plt.show()
                    plt.close()

    # ----------- 4. 输出结果 ------------
    patches = torch.cat(patches_per_level, dim=2)
    coords_norm = torch.cat(coords_per_level, dim=2)
    valid_mask = torch.cat(masks_per_level, dim=2)

    return patches, coords_norm, valid_mask




def debug_sampling_alignment(
    image,                 # torch tensor [B,3,H0,W0], values in [0,1] (原始图片）
    verts_xy_orig,         # torch tensor [B,N,2] 原始像素坐标 (x,y)
    dinov2_wrapper,        # wrapper，需有 .resolution 和 .model.patch_size，且 _preprocess_image 可重现 resize
    feat_pyramid,          # either single tensor [B,C,Hf,Wf] or list of such tensors
    proj_uv_feat=None,     # optional: [B,N,2] feature-grid coords (float) in [0,Wf-1],[0,Hf-1]; 如果 None，会自动计算（用下面 map）
    full_image_wh=None,    # int tuple used in sample function; if None inferred from feat size
    patch_sizes=5,
    align_corners=True,
    vis_prefix="./hand/debug_align",
    max_points=5
):
    """
    会打印详细数值并保存若干可视化图像到 vis_prefix_* 文件。
    返回 dict 便于程序化检查。
    """
    os.makedirs(os.path.dirname(vis_prefix) or ".", exist_ok=True)
    device = image.device
    B, _, H0, W0 = image.shape

    # ---------- 0. get resolution + patch_size ----------
    resolution = dinov2_wrapper.resolution   # 448
    psize_model = dinov2_wrapper.model.patch_size  # 14
    # ---------- 1. compute proj_uv_feat if not provided ----------
    def compute_proj_uv(verts):
        # verts: [B,N,2] original pix coords x in [0,W0-1], y in [0,H0-1]
        max_size = max(H0, W0)
        pad_left = (max_size - W0) // 2
        pad_top  = (max_size - H0) // 2
        scale = float(resolution) / float(max_size)
        x0 = verts[...,0]
        y0 = verts[...,1]
        x_pad = x0 + pad_left
        y_pad = y0 + pad_top
        x_resized = x_pad * scale
        y_resized = y_pad * scale
        u_feat_x = x_resized / float(psize_model)
        u_feat_y = y_resized / float(psize_model)
        return x_pad, y_pad, x_resized, y_resized, u_feat_x, u_feat_y

    if proj_uv_feat is None:
        x_pad, y_pad, x_resized, y_resized, u_feat_x, u_feat_y = compute_proj_uv(verts_xy_orig)
        proj_uv_feat = torch.stack([u_feat_x, u_feat_y], dim=-1)  # [B,N,2]
    else:
        # also compute resized coords for comparison
        u_feat_x = proj_uv_feat[...,0]
        u_feat_y = proj_uv_feat[...,1]
        x_resized = u_feat_x * float(psize_model)
        y_resized = u_feat_y * float(psize_model)
        max_size = max(H0, W0)
        pad_left = (max_size - W0) // 2
        pad_top  = (max_size - H0) // 2
        scale = float(resolution) / float(max_size)
        x_pad = x_resized / scale
        y_pad = y_resized / scale
        x0 = x_pad - pad_left
        y0 = y_pad - pad_top

    # ---------- 2. ensure feat_pyramid list and print shapes ----------
    if isinstance(feat_pyramid, (list, tuple)):
        feat_list = list(feat_pyramid)
    else:
        feat_list = [feat_pyramid]
    print("=== feature pyramid layers ===")
    for i,P in enumerate(feat_list):
        print(f" layer {i}: tensor shape {tuple(P.shape)} dtype {P.dtype} device {P.device}")

    # ---------- 3. determine inferred full_image_wh (for sample fn) ----------
    # If caller gave full_image_wh, we print warning if inconsistent.
    # full_image_wh is expected to be (W_full,H_full) in the earlier function usage; but we will treat it carefully.
    # We'll infer Wf,Hf from first level
    P0 = feat_list[0]
    Bf, C, Hf, Wf = P0.shape
    inferred_full_wh = (Wf, Hf)
    print("inferred feature-grid size (Wf,Hf):", (Wf,Hf), "model.patch_size:", psize_model, "dinov2.resolution:", resolution)
    if full_image_wh is not None:
        print("provided full_image_wh:", full_image_wh, " (should match feature-grid coords basis)")
    else:
        print("no full_image_wh provided; will assume full_image_wh = (Wf,Hf) when calling sample function.")

    # ---------- 4. print sample of coordinate transforms for first few points ----------
    N = proj_uv_feat.shape[1]
    nshow = min(max_points, N)
    print("\n=== example coordinate transforms (first {} points per batch) ===".format(nshow))
    for b in range(B):
        print(f"\n batch {b}:")
        for i in range(nshow):
            x0_i = float(verts_xy_orig[b,i,0].cpu().item())
            y0_i = float(verts_xy_orig[b,i,1].cpu().item())
            # pad/resize etc
            # recompute using same code for clarity
            max_size = max(H0, W0)
            pad_left = (max_size - W0) // 2
            pad_top  = (max_size - H0) // 2
            scale = float(resolution) / float(max_size)
            x_pad_i = x0_i + pad_left
            y_pad_i = y0_i + pad_top
            x_resized_i = x_pad_i * scale
            y_resized_i = y_pad_i * scale
            u_fx = float(proj_uv_feat[b,i,0].cpu().item())
            u_fy = float(proj_uv_feat[b,i,1].cpu().item())
            # normalized coords for grid_sample at level 0 (align_corners handling)
            if align_corners:
                nx = (u_fx / (Wf - 1)) * 2.0 - 1.0
                ny = (u_fy / (Hf - 1)) * 2.0 - 1.0
            else:
                nx = ((u_fx + 0.5) / Wf) * 2.0 - 1.0
                ny = ((u_fy + 0.5) / Hf) * 2.0 - 1.0

            # map this normalized coord back to resized-pixel coordinates (sanity)
            if align_corners:
                u_back = ( (nx + 1.0) / 2.0 ) * (Wf - 1)
                v_back = ( (ny + 1.0) / 2.0 ) * (Hf - 1)
            else:
                u_back = ( (nx + 1.0) / 2.0 ) * Wf - 0.5
                v_back = ( (ny + 1.0) / 2.0 ) * Hf - 0.5
            # resized pixels coordinates
            x_resized_from_u = u_fx * psize_model
            y_resized_from_u = u_fy * psize_model

            print(f"  pt{i}: orig ({x0_i:.2f},{y0_i:.2f}) pad ({x_pad_i:.2f},{y_pad_i:.2f}) resized ({x_resized_i:.2f},{y_resized_i:.2f}) "
                  f"-> u_feat ({u_fx:.3f},{u_fy:.3f}) nx/ny ({nx:.4f},{ny:.4f}) back_to_u ({u_back:.3f},{v_back:.3f}) resized_from_u ({x_resized_from_u:.2f},{y_resized_from_u:.2f})")

    # ---------- 5. visualize mean feature map upsampled to resized image and overlay points ----------
    # compute mean feature across channels for layer0, normalize and upsample to dinov2 resolution for display
    feat0 = feat_list[0]  # [B,C,Hf,Wf]
    feat0_mean = feat0.mean(dim=1, keepdim=True)  # [B,1,Hf,Wf]
    # upsample to resolution pixels
    feat0_up = F.interpolate(feat0_mean, size=(resolution, resolution), mode='bilinear', align_corners=align_corners)  # [B,1,res,res]
    feat0_up_np = feat0_up.detach().cpu().numpy()  # B,1,res,res

    # get resized image (Dinov2 sees)
    with torch.no_grad():
        img_resized = dinov2_wrapper._preprocess_image(image.clone() * 255, dinov2_wrapper.resolution)  # [B,3,res,res]

    for b in range(B):
        vis_img = img_resized[b].cpu().permute(1,2,0).numpy()
        feat_img = feat0_up_np[b,0]
        # normalize feature for display
        feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min() + 1e-8)
        Hres, Wres = feat_img.shape
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(vis_img)
        # overlay first few projected points as red
        for i in range(min(nshow, N)):
            x_res = float(x_resized[b,i].cpu().item())
            y_res = float(y_resized[b,i].cpu().item())
            ax[0].scatter([x_res],[y_res], c='r', s=10)
        ax[0].set_title("resized image with proj points")

        ax[1].imshow(feat_img, cmap='gray')
        # overlay same points on feat_img (resized coords)
        for i in range(min(nshow, N)):
            x_res = float(x_resized[b,i].cpu().item())
            y_res = float(y_resized[b,i].cpu().item())
            # map to feat_img pixel coords: since feat_img is resolution×resolution, these are direct
            ax[1].scatter([x_res],[y_res], c='r', s=10)
        ax[1].set_title("mean feature upsampled to resized")
        plt.tight_layout()
        fpath = f"{vis_prefix}_batch{b}_resized_and_feat.png"
        plt.savefig(fpath, dpi=200)
        plt.close()
        print("Saved resized vs feat overlay:", fpath)

    # ---------- 6. manual single-point sampling check (grid_sample) ----------
    # take first batch, first point; compute grid for psize and sample raw feature map P0
    b0 = 0
    i0 = 0
    u_fx0 = proj_uv_feat[b0, i0, 0].to(device)
    u_fy0 = proj_uv_feat[b0, i0, 1].to(device)
    # build normalized center as in code
    if align_corners:
        nx0 = (u_fx0 / (Wf - 1)) * 2.0 - 1.0
        ny0 = (u_fy0 / (Hf - 1)) * 2.0 - 1.0
    else:
        nx0 = ((u_fx0 + 0.5) / Wf) * 2.0 - 1.0
        ny0 = ((u_fy0 + 0.5) / Hf) * 2.0 - 1.0
    xs = (torch.arange(-(patch_sizes//2), patch_sizes//2 + 1, device=device).float() / (Wf - 1)) * 2.0 if align_corners else (torch.arange(-(patch_sizes//2), patch_sizes//2 + 1, device=device).float() / Wf) * 2.0
    ys = (torch.arange(-(patch_sizes//2), patch_sizes//2 + 1, device=device).float() / (Hf - 1)) * 2.0 if align_corners else (torch.arange(-(patch_sizes//2), patch_sizes//2 + 1, device=device).float() / Hf) * 2.0
    dx, dy = torch.meshgrid(xs, ys, indexing='xy')
    delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)  # [K,2]
    k = delta.shape[0]
    center_norm = torch.tensor([nx0, ny0], device=device).view(1,1,2)
    grid = center_norm + delta.view(1,1,k,2)
    grid_clamped = grid.clamp(-1.0,1.0)
    grid_rs = grid_clamped.view(1, k, 1, 2)
    P = P0[b0:b0+1]  # [1,C,Hf,Wf]
    sampled = F.grid_sample(P, grid_rs, mode='bilinear', padding_mode='zeros', align_corners=align_corners)  # [1,C,k,1]
    sampled = sampled.view(1, C, k).permute(0,2,1).contiguous()  # [1,k,C]
    sampled_np = sampled.detach().cpu().numpy()[0]  # [K,C]
    print("Manual sampled tokens shape:", sampled_np.shape, "K:", k, "C:", sampled_np.shape[1])

    # Save a visualization of these K tokens as small color proxies
    show_n = min(9, k)
    grid_img = np.zeros((32*3, 32*3, 3), dtype=np.float32)
    for j in range(show_n):
        t = sampled_np[j]
        if t.shape[0] >= 3:
            timg = t[:3]
            timg = (timg - timg.min()) / (timg.max() - timg.min() + 1e-8)
            r = j // 3
            c = j % 3
            grid_img[r*32:(r+1)*32, c*32:(c+1)*32, :] = np.tile(timg.reshape(1,1,3), (32,32,1))
    fpath2 = f"{vis_prefix}_batch{b0}_pt{i0}_sampled_patchgrid.png"
    plt.imsave(fpath2, np.clip(grid_img,0,1))
    print("Saved manual sampled patch tokens (color proxies):", fpath2)

    # ---------- return a diagnostic dict ----------
    diag = {
        "resolution": resolution,
        "patch_size_model": psize_model,
        "feat_shapes": [tuple(p.shape) for p in feat_list],
        "proj_uv_feat_sample": proj_uv_feat.detach().cpu().numpy()[:, :nshow, :],
        "x_resized_sample": x_resized.detach().cpu().numpy()[:, :nshow],
        "coords_summary": None
    }
    return diag














