# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Xiaodong Gu & Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:49:25
# @Function      : transformer_block

import pdb
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from diffusers.utils import is_torch_version
try:
    from LHM.models.transformer_mmeff import MemEffBasicBlock
except ImportError:
    MemEffBasicBlock = None

logger = get_logger(__name__)


class TransformerDecoder(nn.Module):
    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """

    motion_embed_type = ["sd3_mm_cond", "sd3_mm_bh_cond"]

    def __init__(
        self,
        block_type: str,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        cond_dim: int = None,
        mod_dim: int = None,
        gradient_checkpointing=False,
        eps: float = 1e-6,
        use_mm_attn: bool = True,
        use_self_attn: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.use_mm_attn = use_mm_attn
        self.use_self_attn = use_self_attn
        self.block_type = block_type

        if (
            block_type == "sd3_cond"
            or block_type == "sd3_mm_cond"
            or block_type == "sd3_mm_bh_cond"   # this way
        ):
            # dual_attention_layers = list(range(num_layers//2))
            dual_attention_layers = []
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim)(
                        num_heads=num_heads,
                        eps=eps,
                        context_pre_only=i == num_layers - 1,
                        use_dual_attention=(
                            True if i in dual_attention_layers else False
                        ),
                        layer_idx=i, 
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim)(
                        num_heads=num_heads,
                        eps=eps,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.norm = nn.LayerNorm(inner_dim, eps=eps)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
        ]:
            self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)
            # Learnable positional encoding for image condition tokens
            self.cond_pos_embed = nn.Parameter(torch.zeros(1, 1024, inner_dim))
            nn.init.trunc_normal_(self.cond_pos_embed, std=0.02)


        # self.local_attn_layers = nn.ModuleList([
        #     MemEffBasicBlock(
        #         inner_dim=inner_dim,
        #         num_heads=num_heads,
        #         eps=eps,
        #         rope=self.rope2d
        #     )
        #     for _ in range(self.num_layers)
        # ])
        

        # self.local_attn_layers = nn.ModuleList([
        #     ProjCrossAttn(
        #         gauss_dim=12337,
        #         img_c=1024,
        #         # eps=eps,
        #         # rope=self.rope2d
        #     )
        #     for _ in range(num_layers)
        # ])



    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in [
            "basic",
            "cond",
            "mod",
            "cond_mod",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
            "cogvideo_cond",
        ], f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim, mod_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == "basic":
            assert (
                cond_dim is None and mod_dim is None
            ), f"Condition and modulation are not supported for BasicBlock"
            from .block import BasicBlock

            logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == "cond":
            assert (
                cond_dim is not None
            ), f"Condition dimension must be specified for ConditionBlock"
            assert (
                mod_dim is None
            ), f"Modulation dimension is not supported for ConditionBlock"
            from .block import ConditionBlock

            logger.debug(f"Using ConditionBlock")
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        elif self.block_type == "mod":
            logger.error(f"modulation without condition is not implemented")
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        elif self.block_type == "cond_mod":
            assert (
                cond_dim is not None and mod_dim is not None
            ), f"Condition and modulation dimensions must be specified for ConditionModulationBlock"
            from .block import ConditionModulationBlock

            logger.debug(f"Using ConditionModulationBlock")
            return partial(
                ConditionModulationBlock,
                inner_dim=inner_dim,
                cond_dim=cond_dim,
                mod_dim=mod_dim,
            )
        elif self.block_type == "cogvideo_cond":
            logger.debug(f"Using CogVideoXBlock")
            from LHM.models.transformer_dit import CogVideoXBlock

            # assert inner_dim == cond_dim, f"inner_dim:{inner_dim}, cond_dim:{cond_dim}"
            return partial(CogVideoXBlock, dim=inner_dim, attention_bias=True)
        elif self.block_type == "sd3_cond":
            logger.debug(f"Using SD3JointTransformerBlock")
            from LHM.models.transformer_dit import SD3JointTransformerBlock

            return partial(SD3JointTransformerBlock, dim=inner_dim, qk_norm="rms_norm")
        elif self.block_type == "sd3_mm_cond":
            logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit import SD3MMJointTransformerBlock

            return partial(
                SD3MMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm",
                use_mm_attn=self.use_mm_attn,
                use_self_attn=self.use_self_attn,
            )
        elif self.block_type == "sd3_mm_bh_cond":
            logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit import SD3BodyHeadMMJointTransformerBlock

            return partial(
                SD3BodyHeadMMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        else:
            raise ValueError(
                f"Unsupported block type during runtime: {self.block_type}"
            )

    def assert_runtime_integrity(
        self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        assert x is not None, f"Input tensor must be specified"
        if self.block_type == "basic":
            assert (
                cond is None and mod is None
            ), f"Condition and modulation are not supported for BasicBlock"
        elif "cond" in self.block_type:
            assert (
                cond is not None and mod is None
            ), f"Condition must be specified and modulation is not supported for ConditionBlock"
        elif self.block_type == "mod":
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        else:
            assert (
                cond is not None and mod is not None
            ), f"Condition and modulation must be specified for ConditionModulationBlock"

    def forward_layer(
        self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        if self.block_type == "basic":
            return layer(x)
        elif self.block_type == "cond":
            return layer(x, cond)
        elif self.block_type == "mod":
            return layer(x, mod)
        else:
            return layer(x, cond, mod)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor = None,
        mod: torch.Tensor = None,
        temb: torch.Tensor = None,
        global_four: torch.Tensor = None,
        nail_mask: Optional[torch.FloatTensor] = None,  # (B, 1, H, W)  feature-map coords (u,v) float
        nail_mask_3d: Optional[torch.FloatTensor] = None,  # (B, 1, D, H, W)  feature-map coords (u,v) float
        proj_xy: Optional[torch.FloatTensor] = None,   # (B, L, 2)  feature-map coords (u,v) float
        p_vis: Optional[torch.FloatTensor] = None,     # (B, L)    visibility score in [0,1]
        point_pos: Optional[torch.FloatTensor] = None, 
        posed_pos: Optional[torch.FloatTensor] = None,
        feat_hw: Optional[Tuple[int,int]] = None,      # (Hf, Wf)  feature-map spatial size (if proj_xy provided)
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, D].
            cond (torch.Tensor, optional): Conditional tensor of shape [N, L_cond, D_cond] or None. Defaults to None.
            mod (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.
            temb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """

        # x: [N, L, D]
        # cond: [N, L_cond, D_cond] or None
        # mod: [N, D_mod] or None
        self.assert_runtime_integrity(x, cond, mod)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
        ]:
            cond = self.linear_cond_proj(cond)
            cond = cond + self.cond_pos_embed[:, :cond.shape[1], :]
            # for layer, layer_local in zip(self.layers, self.local_attn_layers):
            for i, layer in enumerate(self.layers):
                x, cond = layer(
                # x, cond, moe_loss = layer(
                        hidden_states=x,
                        encoder_hidden_states=cond,
                        temb=temb,
                        global_four=global_four,
                        nail_mask=nail_mask,
                        nail_mask_3d=nail_mask_3d,
                        proj_xy=proj_xy,
                        p_vis=p_vis,
                        point_pos=point_pos,
                        posed_pos=posed_pos,
                        feat_hw=feat_hw,
                        layer_idx=i,
                        # image_rotary_emb=None,
                    )
                # x = layer_local(x, nail_img_feats)
            x = self.norm(x)
            # x = self.norm(cond)

        else:
            for layer in self.layers:
                x = self.forward_layer(layer, x, cond, mod)
            x = self.norm(x)

        return x#, moe_loss

