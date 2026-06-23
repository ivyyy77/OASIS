# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pdb

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger

logger = get_logger(__name__)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        use_clstoken=False,
        out_channel=1024,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in inner_channels
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.output_conv = nn.Conv2d(
            sum(inner_channels), out_channel, kernel_size=1, stride=1, padding=0
        )
        self.to('cuda')

    def forward(self, out_features, patch_h, patch_w):

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)

            out.append(x)

        fusion_feats = torch.cat(out, dim=1)

        fusion_feats = self.output_conv(fusion_feats)

        return fusion_feats


class Dinov2FusionWrapper(nn.Module):
    """
    Dinov2FusionWrapper using original implementation, hacked with modulation.
    """

    def __init__(
        self,
        model_name: str,
        modulation_dim: int = None,
        freeze: bool = True,
        encoder_feat_dim: int = 384,
        resolution=448,  # DINOV2 default resolution
        antialias=True,
    ):
        super().__init__()
        self.modulation_dim = modulation_dim
        self.model = self._build_dinov2(model_name, modulation_dim=modulation_dim)

        self.intermediate_layer_idx_info = {
            "dinov2_vits14_reg": [2, 5, 8, 11],
            "dinov2_vitb14_reg": [2, 5, 8, 11],
            "dinov2_vitl14_reg": [4, 11, 17, 23],
            "dinov2_vitg14_reg": [9, 19, 29, 39],
        }

        self.intermediate_layer_idx = self.intermediate_layer_idx_info[model_name]
        self.fusion_head = DPTHead(
            in_channels=self.model.embed_dim,
            inner_channels=[self.model.embed_dim] * 4,
            out_channel=encoder_feat_dim,
        )

        # ====== Upsampling to target resolution (256x256 for hand) ======
        # 从 fusion_head 输出 (e.g., [B, 1024, 32, 32]) 上采样到 256x256
        # 参照 GUAVA dino_encoder.py 的处理方式
        self.upsample_to_256 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(encoder_feat_dim, encoder_feat_dim, kernel_size=3, stride=1, padding=1, bias=False),
        ).cuda()

        self.resolution = resolution
        self.antialias = antialias

        # # M = 4
        # # self.num_style_tokens = M
        # # # self.cls_fuse = nn.Linear(in_features=self.model.embed_dim*len(self.intermediate_layer_idx),
        # # #                           out_features=self.model.embed_dim*self.num_style_tokens).cuda()
        # self.cls_fuse = nn.Sequential(  
        #     nn.Linear(self.model.embed_dim * len(self.intermediate_layer_idx), self.model.embed_dim),
        #     nn.GELU(),
        #     # nn.Linear(self.model.embed_dim * len(self.intermediate_layer_idx), self.model.embed_dim * len(self.intermediate_layer_idx)),).cuda()
        #     nn.Linear(self.model.embed_dim, self.model.embed_dim * len(self.intermediate_layer_idx)),).cuda()
        
        #     # nn.Linear(self.model.embed_dim * len(self.intermediate_layer_idx), self.model.embed_dim),
        #     # # nn.GELU(),
        #     # nn.SiLU(),
        #     # # nn.Linear(self.model.embed_dim * len(self.intermediate_layer_idx), self.model.embed_dim * len(self.intermediate_layer_idx)),).cuda()
        #     # nn.Linear(self.model.embed_dim, self.model.embed_dim)).cuda()


        if freeze:
            if modulation_dim is not None:
                raise ValueError(
                    "Modulated Dinov2 requires training, freezing is not allowed."
                )
            self._freeze()

    def _freeze(self):
        logger.warning(f"======== Freezing Dinov2FusionWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def _preprocess_image(
        self, image: torch.tensor, resolution: int = 1024
    ) -> torch.Tensor:

        _, __, H, W = image.shape
        max_size = max(H, W)
        H_pad = max_size - H
        W_pad = max_size - W
        pad_size = (
            W_pad // 2,
            max_size - (W + W_pad // 2),
            H_pad // 2,
            max_size - (H + H_pad // 2),
            0,
            0,
            0,
            0,
        )

        image = F.pad(image, pad_size, value=1)

        image = kornia.geometry.resize(
            image,
            (resolution, resolution),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )

        return image

    @staticmethod
    def _build_dinov2(
        model_name: str, modulation_dim: int = None, pretrained: bool = True
    ):
        from importlib import import_module

        dinov2_hub = import_module(".dinov2.hub.backbones", package=__package__)
        model_fn = getattr(dinov2_hub, model_name)
        logger.debug(f"Modulation dim for Dinov2 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained)
        return model

    @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized

        image = self._preprocess_image(image, self.resolution)     # (1,3,256,256) ----> (1,3,448,448)

        patch_h, patch_w =(     # 32, 32
            image.shape[-2] // self.model.patch_size,
            image.shape[-1] // self.model.patch_size,
        )

        features = self.model.get_intermediate_layers(
            image, self.intermediate_layer_idx, return_class_token=True
        )   # 每一个 features，第一个是latent (2,1024,1024)，第二个是cls_token (2,1024)

        # out_local = self.fusion_head(features, patch_h, patch_w).permute(0, 2, 3, 1).flatten(1, 2)  # 2.1024.1024
        out_ = self.fusion_head(features, patch_h, patch_w) # 2,1024,32,32
        out_local = out_.permute(0, 2, 3, 1).flatten(1, 2)  # 2.1024.1024

        # # ====== 新增：从 4 个层的 CLS 聚合 style tokens ======

        # # 1) 收集多层 CLS: 每个 features[i][1] 是 [B, D]
        # cls_list = [feat[1] for feat in features]         # list of 4 x [B, D]
        # # 2) 在通道维度 concat: [B, 4D]
        # cls_cat = torch.cat(cls_list, dim=-1)             # [B, 4D]

        # # 3) MLP 融合成单一 global feature → [B, D]
        # out_global_four = self.cls_fuse(cls_cat)

        # # ==================================================
        # # 3) MLP 映射成 M*D，然后 reshape 成 [B, M, D]
        # style_flat = self.cls_fuse(cls_cat)           # [B, M*D]
        # # B, _ = style_flat.shape
        # # D = self.model.embed_dim
        # # M = self.num_style_tokens
        # # out_global = style_flat.view(B, M, D)           # [B, M, D]
        # out_global = style_flat
        
        out_global = features[-1][1]
        out_global_four = out_global
        # return out_local, out_global
        
        # out_global = None
        # if out_global is not None:
        #     ret = torch.cat(
        #         [out_local.permute(0, 2, 3, 1).flatten(1, 2), out_global.unsqueeze(1)],
        #         dim=1,
        #     )
        # else:
        #     ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)

        # ====== Upsample fusion output to 256x256 (参照 GUAVA) ======
        # out_: [B, 1024, 32, 32] → [B, 1024, 256, 256]
        # out_upsampled = self.upsample_to_256(out_)  # [B, encoder_feat_dim, 256, 256]

    
        return out_local, out_, out_global #, out_upsampled
        # return ret
