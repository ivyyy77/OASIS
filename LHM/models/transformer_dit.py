# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu & Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:41:38
# @Function      : Transformer Block  


from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

from plyfile import PlyData 
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from diffusers.models.attention import DropPath
from LHM.models.modeling_hand_lrm import VisibilityBiasNet

# ==== Attention Visualization (Ablation Study) ====
# Global flag to enable/disable attention capture
ATTN_VIZ_ENABLED = False
_ATTN_VISUALIZER = None
VIS_BIAS_ENABLED = False

def set_attn_viz_enabled(enabled: bool, save_dir: str = 'output/attn_debug_v2', **kwargs):
    """Enable or disable attention visualization globally."""
    global ATTN_VIZ_ENABLED, _ATTN_VISUALIZER
    ATTN_VIZ_ENABLED = enabled
    if enabled:
        from LHM.utils.attn_visualizer import AttnVisualizer
        _ATTN_VISUALIZER = AttnVisualizer(save_dir=save_dir, enabled=True, **kwargs)
        print(f"[AttnViz] Enabled, saving to {save_dir}")
    else:
        _ATTN_VISUALIZER = None
        print("[AttnViz] Disabled")

def get_attn_visualizer():
    """Get the global attention visualizer instance."""
    return _ATTN_VISUALIZER


def set_vis_bias_enabled(enabled: bool):
    """Enable or disable visibility bias in pc2img cross-attention."""
    global VIS_BIAS_ENABLED
    VIS_BIAS_ENABLED = enabled
    print(f"[VisBias] {'Enabled' if enabled else 'Disabled'}")


def is_vis_bias_enabled() -> bool:
    """Return whether visibility bias is currently enabled."""
    return VIS_BIAS_ENABLED


def _reduce_attn_heads(attn_weights: torch.Tensor, head_mode: str) -> torch.Tensor:
    if attn_weights.ndim != 4:
        raise ValueError(f"Expected attention weights [B, H, L, M], got {tuple(attn_weights.shape)}")
    if head_mode == "last":
        return attn_weights[:, -1, :, :]
    if head_mode == "mean":
        return attn_weights.mean(dim=1)
    raise ValueError(f"Unsupported head_mode: {head_mode}")
# ===================================================

assert hasattr(F, "scaled_dot_product_attention"), print(
    "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
)
import pdb

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    CogVideoXAttnProcessor2_0,
    JointAttnProcessor2_0,
)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero

from LHM.models.rendering.gs_renderer import HierarchicalPointEmbedTransformer


class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        # num_attention_heads: int,
        # attention_head_dim: int,
        # time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        eps: float = 1e-5,
        # norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        norm_eps = eps
        num_attention_heads = num_heads
        attention_head_dim = dim // num_attention_heads
        assert attention_head_dim * num_attention_heads == dim

        # 1. Self Attention
        self.norm1 = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps, bias=True
        )
        self.norm1_context = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps, bias=True
        )
        self.norm2_context = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        # norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
        #     hidden_states, encoder_hidden_states, temb
        # )
        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states

        # norm & modulate
        # norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
        #     hidden_states, encoder_hidden_states, temb
        # )
        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x / keep_prob * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class SwiGLU_FF(nn.Module):
    def __init__(self, dim, inner_dim, dropout=0.0):
        super().__init__()
        self.proj_in = nn.Linear(dim, inner_dim * 2)  # split for gate
        self.proj_out = nn.Linear(inner_dim, dim)
        self.act = nn.SiLU()   # 或 torch.nn.functional.silu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.proj_in(x).chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.proj_out(x)
        return self.dropout(x)




def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class QKNormJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states, encoder_hidden_states


class SD3JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float,
        # num_attention_heads: int,
        # attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
    ):
        super().__init__()
        num_attention_heads = num_heads
        attention_head_dim = dim // num_attention_heads
        assert attention_head_dim * num_attention_heads == dim

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        self.norm1 = nn.LayerNorm(dim)

        self.norm1_context = nn.LayerNorm(dim)

        """Attention processor used typically in processing 
        the SD3-like self-attention projections."""

        processor = JointAttnProcessor2_0()

        # qk_norm rms_norm
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        # SD-3.5
        if use_dual_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=eps,
            )
        else:
            self.attn2 = None

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            self.ff_context = FeedForward(
                dim=dim, dim_out=dim, activation_fn="gelu-approximate"
            )
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor = None,
    ):
        """
        Forward pass of the transformer_dit model.
        Args:
            hidden_states (torch.FloatTensor): Input hidden states. Query Points features
            encoder_hidden_states (torch.FloatTensor): Encoder hidden states. Context features
            temb (torch.FloatTensor, optional): Optional tensor for embedding. Defaults to None.
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Tuple containing the updated hidden states and encoder hidden states.
        """


        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)

        # Attention.
        # norma hidden states [B, L, D] - > multi-head atten [B, num_head, L, D/num_head]
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Process attention outputs for the `hidden_states`.
        # attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # ffd

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states)
            # attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        # norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)
        # ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            # context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            # norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            # encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            encoder_hidden_states = encoder_hidden_states + context_ff_output

        return hidden_states, encoder_hidden_states


class SD3MMJointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float,
        # num_attention_heads: int,
        # attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        layer_idx: Optional[int] = None,
        use_mm_attn: bool = False,
        use_self_attn: bool = True,
    ):
        super().__init__()
        num_attention_heads = num_heads
        attention_head_dim = dim // num_attention_heads
        assert attention_head_dim * num_attention_heads == dim

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        self.use_mm_attn = use_mm_attn
        self.use_self_attn = use_self_attn

        context_norm_type = (
            "ada_norm_continous" if context_pre_only else "ada_norm_zero"
        )

        self.norm1 = AdaLayerNormZero(dim)
        # self.norm_occl = AdaLayerNormZero(dim)
        self.dim = dim

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim,
                dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        processor = JointAttnProcessor2_0()

        self.drop_path = DropPath(drop_prob=0.2) 

        # qk_norm rms_norm
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,            
            bias=True,
        )

        self.pc2img_attn = Attention(
            query_dim=dim,
            cross_attention_dim=1024,
            heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            out_dim=dim,
            out_context_dim=dim,
            )
        
        self.gamma_global_boost = 0.0
        # self.vis_bias_net = VisibilityBiasNet(
        #     use_depth_res = False,
        #     hidden_dim = 32,
        # )


        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm_global = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # self.norm_global = nn.LayerNorm(dim*4, elementwise_affine=False, eps=eps)
        self.beta_local = 2.0
        self.beta_global = 2.0
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        # self.ff = SwiGLU_FF(dim=dim, inner_dim=dim*4, dropout=0.0)
        # self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="geglu")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            self.ff_context = FeedForward(
                dim=dim, dim_out=dim, activation_fn="gelu-approximate"
                # dim=dim, dim_out=dim, activation_fn="geglu"
            )
            # self.ff_context = SwiGLU_FF(dim=dim, inner_dim=dim*4, dropout=0.0)  
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # if layer_idx == 0 or layer_idx == 3:
        self.part_aware_point = HierarchicalPointEmbedTransformer(
            in_feat_dim=dim, out_dim=dim, part_dim=dim
        )#.cuda()
        self.pointwise_attn_heatmap_enabled = False
        self.pointwise_attn_heatmap_metric = "global_reliance"
        self.pointwise_attn_heatmap_head_mode = "last"
        self.latest_pointwise_attn_scores = None
        self.latest_pointwise_attn_score_maps = {}
        self.latest_pointwise_attn_meta = {}


    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def set_pointwise_attn_heatmap_enabled(
        self,
        enabled: bool,
        metric: str = "global_reliance",
        head_mode: str = "last",
    ):
        self.pointwise_attn_heatmap_enabled = enabled
        self.pointwise_attn_heatmap_metric = metric
        self.pointwise_attn_heatmap_head_mode = head_mode
        if not enabled:
            self.latest_pointwise_attn_scores = None
            self.latest_pointwise_attn_score_maps = {}
            self.latest_pointwise_attn_meta = {}

    def _compute_pointwise_attn_score_maps(self, attn_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        reduced_attn = _reduce_attn_heads(attn_weights, self.pointwise_attn_heatmap_head_mode)
        n_local = reduced_attn.shape[-1] - 1
        local_attn = reduced_attn[..., :n_local]
        local_mass = local_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        local_attn = local_attn / local_mass
        return {
            "global_reliance": reduced_attn[..., n_local],
            "local_reliance": reduced_attn[..., :n_local].sum(dim=-1),
            "local_entropy": -(local_attn * torch.log(local_attn.clamp_min(1e-10))).sum(dim=-1),
        }

    def _compute_pointwise_attn_scores(self, attn_weights: torch.Tensor) -> torch.Tensor:
        score_maps = self._compute_pointwise_attn_score_maps(attn_weights)
        if self.pointwise_attn_heatmap_metric in score_maps:
            return score_maps[self.pointwise_attn_heatmap_metric]
        raise ValueError(f"Unsupported pointwise attention metric: {self.pointwise_attn_heatmap_metric}")


    def apply_visibility_bias(self, scores, vis_soft, beta_local=2.0, beta_global=2.0):
        """
        scores:    [B, H, N_p, 1025]
        vis_soft:  [B, N_p]  in [0,1]
        """

        B, H, N_p, N_k = scores.shape
        N_local = N_k - 1    # 1024
        N_global = 1         # 最后一个 token

        # ---- 1. 转换到 s ∈ [-1, 1] ----
        vis = vis_soft.clamp(0, 1)           # [B, N_p]
        s = 2.0 * vis - 1.0                  # visible=+1, invisible=-1

        # ---- 2. local bias（对前1024个token）----
        # visible → +β_local
        # invisible → -β_local
        local_bias_row = s * beta_local      # [B, N_p]
        local_bias = local_bias_row.unsqueeze(-1).expand(-1, -1, N_local)

        # ---- 3. global bias（对第1025个token）----
        # visible → -β_global
        # invisible → +β_global
        global_bias_row = -s * beta_global   # [B, N_p]
        global_bias = global_bias_row.unsqueeze(-1)  # [B, N_p, 1]

        # ---- 4. 合并 ----
        bias_2d = torch.cat([local_bias, global_bias], dim=-1)   # [B, N_p, 1025]

        # ---- 5. 扩展到 multi-head ----
        bias = bias_2d.unsqueeze(1).to('cuda:1')          # [B, 1, N_p, 1025]

        # ---- 6. 加到 logits ----
        scores = scores + bias
        return scores



    def build_visibility_attn_mask(self, p_vis, M, beta_local=2.0, beta_global=2.0):
        """
        p_vis: [B, L] in [0,1]  soft visibility
        M: int, 总 token 数（你的情况是 1025 = 1024 local + 1 global）

        返回: attn_mask [B, 1, L, M] 作为 additive bias
        - 对可见点: local logits ↑, global logits ↓
        - 对不可见: local logits ↓, global logits ↑
        """
        B, L = p_vis.shape
        N_local = M - 1   # 1024
        N_global = 1

        vis = p_vis.clamp(0.0, 1.0)           # [B, L]
        s = 2.0 * vis - 1.0                   # [B, L], vis=1→+1, vis=0→-1

        # local bias: visible → +β_local, invisible → -β_local
        local_bias_row = s * beta_local                      # [B, L]
        local_bias = local_bias_row.unsqueeze(-1).expand(-1, -1, N_local)   # [B, L, N_local] 2,12337,1024
        # 因为 local tokens 有 1024 个，所以对每个点 i，你需要对 local 的所有 1024 列都加相同的偏置。
        
        # global bias: visible → -β_global, invisible → +β_global
        global_bias_row = -s * beta_global                   # [B, L]
        global_bias = global_bias_row.unsqueeze(-1).expand(-1, -1, N_global) # [B, L, 1] 2,12337,1

        # 拼成 [B, L, M]
        bias_2d = torch.cat([local_bias, global_bias], dim=-1)  # [B, L, M]

        # 升维到 [B, 1, L, M]，在 head 维度 broadcast
        attn_mask = bias_2d.unsqueeze(1)                        # [B, 1, L, M]
        return attn_mask


    def build_learned_visibility_attn_mask(
        self, p_vis: torch.Tensor, depth_res: torch.Tensor | None = None
    ):
        """
        p_vis: [B, L]
        depth_res: [B, L] or None

        返回:
          attn_mask: [B, 1, L, M] additive bias，用于 SDPA
        """
        B, L = p_vis.shape
        M = 1025
        N_local = M - 1
        device, dtype = p_vis.device, p_vis.dtype

        # 1) 用 meta-net 预测每个点的 [b_local, b_global]
        b_local, b_global = self.vis_bias_net(p_vis, depth_res=depth_res)  # [B,L],[B,L]

        # 2) 展开到 [B,L,M]
        local_bias = b_local.unsqueeze(-1).expand(-1, -1, N_local)       # [B,L,N_local]
        global_bias = b_global.unsqueeze(-1).expand(-1, -1, M-N_local)     # [B,L,1]

        bias_2d = torch.cat([local_bias, global_bias], dim=-1)       # [B,L,M]

        # 3) 升维到 [B,1,L,M]，head 维 broadcast
        attn_mask = bias_2d.unsqueeze(1)                             # [B,1,L,M]
        return attn_mask


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor = None,
        global_four: torch.FloatTensor = None,
        nail_mask: Optional[torch.FloatTensor] = None,  # (B, 1, H, W)  1 for valid points, 0 for nails
        nail_mask_3d: Optional[torch.FloatTensor] = None,  # (B, L, 2)  nail (u,v) coords
        proj_xy: Optional[torch.FloatTensor] = None,   # (B, L, 2)  feature-map coords (u,v) float
        p_vis: Optional[torch.FloatTensor] = None,     # (B, L)    visibility score in [0,1]
        point_pos: Optional[torch.FloatTensor] = None,  
        posed_pos: Optional[torch.FloatTensor] = None,
        feat_hw: Optional[Tuple[int,int]] = None,      # (Hf, Wf)  feature-map spatial size (if proj_xy provided)
        layer_idx: int = 0,    # ⭐ 新增：当前 block 的 index
    ):
        """
        Forward pass of the transformer_dit model.
        Args:
            hidden_states (torch.FloatTensor): Input hidden states. Query Points features
            encoder_hidden_states (torch.FloatTensor): Encoder hidden states. Context features
            motion embed:(torch.FloatTensor, optional): Optional tensor for embedding. Defaults to None.
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Tuple containing the updated hidden states and encoder hidden states.
        """

        if temb is None:
            pdb.set_trace()

        pointwise_attn_heatmap_enabled = getattr(self, "pointwise_attn_heatmap_enabled", False)
        pointwise_attn_heatmap_metric = getattr(self, "pointwise_attn_heatmap_metric", "global_reliance")
        pointwise_attn_heatmap_head_mode = getattr(self, "pointwise_attn_heatmap_head_mode", "last")
        use_self_attn = getattr(self, "use_self_attn", True)
        use_mm_attn = getattr(self, "use_mm_attn", False)
        beta_local = getattr(self, "beta_local", 2.0)
        beta_global = getattr(self, "beta_global", 2.0)

        if pointwise_attn_heatmap_enabled:
            self.latest_pointwise_attn_scores = None
            self.latest_pointwise_attn_meta = {}

        # norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        #     hidden_states, out_msa=True, emb=temb     # hidden_states: 2,12337,1024       emb: 2,1024
        # )

        # # if (layer_idx == 0 or layer_idx == 3):
        # hidden_states = self.part_aware_point(
        #     point_feat=hidden_states.cuda(),
        #     verts_pos=point_pos,
        #     # verts_pos=point_pos.to('cuda:1'),
        # ).to('cuda:1')

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb     # hidden_states: 2,12337,1024       emb: 2,1024
        )

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:   # this way
            (
                norm_encoder_hidden_states,    # 2,1024,1024
                c_gate_msa,   # all of the followings are (2,1024)
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # # Attention.
        # # norma hidden states [B, L, D] - > multi-head atten [B, num_head, L, D/num_head]
        # attn_output, context_attn_output = self.attn(
        #     hidden_states=norm_hidden_states,
        #     encoder_hidden_states=norm_encoder_hidden_states,
        # )   # attn_output: 2,12337,1024         context_attn_output: 2,1024,1024




        # # ============================================================
        # # ============= B1: Visibility-Aware Attention Bias ==========
        # # ============================================================
        # S_bias = None
        # if proj_xy is not None and p_vis is not None and feat_hw is not None:
        #     B, L, _ = proj_xy.shape
        #     Hf, Wf = feat_hw
        #     device = proj_xy.device
        #     dtype = proj_xy.dtype

        # # === 坐标映射：从原图像 → feature map 坐标系 ===
        # scale_x = Wf / 256
        # scale_y = Hf / 256
        # proj_xy_feat = torch.empty_like(proj_xy)
        # proj_xy_feat[..., 0] = proj_xy[..., 0] * scale_x   # u
        # proj_xy_feat[..., 1] = proj_xy[..., 1] * scale_y   # v

        # # === 生成 encoder 特征图坐标 grid ===
        # ys = torch.arange(Hf, device=device, dtype=dtype)
        # xs = torch.arange(Wf, device=device, dtype=dtype)
        # grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # key_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (M,2)
        # M = key_coords.shape[0]

        # # # === 计算高斯 bias ===
        # # diff = proj_xy_feat.unsqueeze(2) - key_coords.view(1, 1, M, 2)
        # # d2 = (diff ** 2).sum(dim=-1)  # (B, L, M)
        # # B_mag = getattr(self, "vis_bias_mag", 3.0)
        # # sigma = getattr(self, "vis_bias_sigma", 4.0)
        # # G = torch.exp(-d2 / (2.0 * sigma ** 2))
        # # p_vis_exp = p_vis.unsqueeze(-1)
        # # S_bias = p_vis_exp * (B_mag * G)


        # chunk = 1024  # 每次最多处理2048个点
        # S_bias_chunks = []
        # for start in range(0, L, chunk):
        #     end = min(start + chunk, L)
        #     diff = proj_xy_feat[:, start:end, None, :] - key_coords[None, None, :, :]
        #     d2 = (diff ** 2).sum(dim=-1)
        #     B_mag = getattr(self, "vis_bias_mag", 3.0)
        #     sigma = getattr(self, "vis_bias_sigma", 4.0)
        #     G = torch.exp(-d2 / (2.0 * sigma ** 2))
        #     S_bias_chunk = p_vis[:, start:end, None] * (B_mag * G)
        #     S_bias_chunks.append(S_bias_chunk)
        # S_bias = torch.cat(S_bias_chunks, dim=1)


        # # NOTE: 此处不归一化，直接加到attention logits

        # # ============================================================
        # # ============ 调用attention并加入bias ======================
        # # ============================================================
        # if S_bias is not None:
        #     # 需要从attn模块中取得 logits，加bias，再softmax
        #     # 典型attention接口不支持bias → 在这里直接修改
        #     q = self.attn.to_q(norm_hidden_states)       # (B, L, D)
        #     k = self.attn.to_k(norm_encoder_hidden_states)
        #     v = self.attn.to_v(norm_encoder_hidden_states)
        #     B, L, D = q.shape
        #     _, M, _ = k.shape
        #     H = 16 # self.attn.num_heads
        #     head_dim = D // H
        #     scale = 1.0 / (head_dim ** 0.5)

        #     # 分多头
        #     q = q.view(B, L, H, head_dim).transpose(1, 2)   # (B,H,L,d)
        #     k = k.view(B, M, H, head_dim).transpose(1, 2)   # (B,H,M,d)
        #     v = v.view(B, M, H, head_dim).transpose(1, 2)

        #     # logits
        #     logits = torch.einsum('bhld,bhmd->bhlm', q, k) * scale   # (B,H,L,M)
        #     if head_idx < 8:
        #         logits = logits + S_bias.unsqueeze(1)
        #     # logits = logits + S_bias.unsqueeze(1)                    # 加bias (B,1,L,M)
        #     attn = torch.softmax(logits, dim=-1)
        #     out = torch.einsum('bhlm,bhmd->bhld', attn, v)
        #     out = out.transpose(1,2).contiguous().view(B,L,D)
        #     # attn_output = self.attn.to_out(out)                      # (B,L,D)
        #     # 手动调用 ModuleList 内的每个模块
        #     attn_output = out
        #     if isinstance(self.attn.to_out, torch.nn.ModuleList):
        #         for m in self.attn.to_out:
        #             attn_output = m(attn_output)
        #     else:
        #         attn_output = self.attn.to_out(attn_output)

        #     # # 如果你保留context的输出
        #     # context_attn_output = None
        #     # ====== Context -> Query（图像特征更新，不加bias）======
        #     # 如果需要让 encoder 侧也同步更新（如原LHM）
        #     q_c = self.attn.to_q(norm_encoder_hidden_states)
        #     k_c = self.attn.to_k(norm_hidden_states)
        #     v_c = self.attn.to_v(norm_hidden_states)

        #     B, Mc, D = q_c.shape
        #     _, Lc, _ = k_c.shape
        #     H = 16 # self.attn.num_heads
        #     head_dim = D // H
        #     scale = 1.0 / (head_dim ** 0.5)

        #     q_c = q_c.view(B, Mc, H, head_dim).transpose(1, 2)  # (B,H,M,d)
        #     k_c = k_c.view(B, Lc, H, head_dim).transpose(1, 2)  # (B,H,L,d)
        #     v_c = v_c.view(B, Lc, H, head_dim).transpose(1, 2)

        #     logits_c = torch.einsum('bhmd,bhld->bhml', q_c, k_c) * scale
        #     attn_c = torch.softmax(logits_c, dim=-1)
        #     out_c = torch.einsum('bhml,bhld->bhmd', attn_c, v_c)
        #     out_c = out_c.transpose(1, 2).contiguous().view(B, Mc, D)

        #     # 同样兼容 ModuleList 的 to_out
        #     context_attn_output = out_c
        #     if isinstance(self.attn.to_out, torch.nn.ModuleList):
        #         for m in self.attn.to_out:
        #             context_attn_output = m(context_attn_output)
        #     else:
        #         context_attn_output = self.attn.to_out(context_attn_output)
        # else:
        #     # 原始方式
        #     attn_output, context_attn_output = self.attn(
        #         hidden_states=norm_hidden_states,
        #         encoder_hidden_states=norm_encoder_hidden_states,
        #     )


        # #############################################################################
        # #############################################################################
        # 加入的 交叉注意力
        norm_global = self.norm_global(global_four)    # 2,1024
        norm_global = norm_global.reshape(norm_global.size(0), -1, 1024)
        local_global = torch.cat((norm_encoder_hidden_states, norm_global), dim=1)  # 2,1025,1024

        Q = self.pc2img_attn.to_q(norm_hidden_states)       # point queries
        K = self.pc2img_attn.to_k(local_global)  # image keys/values
        V = self.pc2img_attn.to_v(local_global)

        B, L, D = Q.shape
        _, M, _ = K.shape
        H = self.pc2img_attn.heads
        head_dim = D // H
        scale = 1.0 / (head_dim ** 0.5)

        # 分多头
        Q = Q.view(B, L, H, head_dim).transpose(1, 2)   # (B,H,L,d)
        K = K.view(B, M, H, head_dim).transpose(1, 2)   # (B,H,M,d)
        V = V.view(B, M, H, head_dim).transpose(1, 2)  

        Q = self.pc2img_attn.norm_q(Q)
        K = self.pc2img_attn.norm_k(K)  
        
        attn_mask = None
        if p_vis is not None:
            attn_mask = self.build_visibility_attn_mask(
                p_vis,
                M,
                beta_local,
                beta_global,
            ).to(Q.device)
            # attn_mask = self.build_learned_visibility_attn_mask(p_vis=p_vis, depth_res=None).to(Q.device)
            # attn_mask = self.build_global_boost_mask(p_vis, M, gamma=self.gamma_global_boost).to(Q.device)
            if not VIS_BIAS_ENABLED:
                attn_mask = torch.zeros_like(attn_mask)
        
        # ==== Attention Visualization / Pointwise Score Capture ====
        use_explicit_cross_attn = (
            (ATTN_VIZ_ENABLED and _ATTN_VISUALIZER is not None)
            or pointwise_attn_heatmap_enabled
        )
        if use_explicit_cross_attn:
            # Compute attention scores manually for visualization
            attn_logits = torch.einsum('bhld,bhmd->bhlm', Q, K) * scale  # [B, H, L, M]
            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask
            attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, L, M]

            if pointwise_attn_heatmap_enabled:
                pointwise_score_maps = self._compute_pointwise_attn_score_maps(attn_weights)
                self.latest_pointwise_attn_score_maps = {
                    name: score.detach() for name, score in pointwise_score_maps.items()
                }
                self.latest_pointwise_attn_scores = self.latest_pointwise_attn_score_maps.get(
                    pointwise_attn_heatmap_metric
                )
                self.latest_pointwise_attn_meta = {
                    'metric': pointwise_attn_heatmap_metric,
                    'head_mode': pointwise_attn_heatmap_head_mode,
                    'layer_idx': layer_idx,
                    'n_local_tokens': M - 1,
                }
            
            # Capture attention for visualization (only sample every N steps to save memory)
            if ATTN_VIZ_ENABLED and _ATTN_VISUALIZER is not None:
                _ATTN_VISUALIZER.capture(
                    attn_scores=attn_weights,
                    layer_idx=layer_idx,
                    attn_type='cross',
                    p_vis=p_vis,
                    proj_xy=proj_xy,
                    point_pos=point_pos,
                    posed_pos=posed_pos,
                    extra_info={
                        'attn_mask_used': attn_mask is not None,
                        'vis_bias_enabled': VIS_BIAS_ENABLED,
                        'beta_local': beta_local,
                        'beta_global': beta_global,
                    }
                )
            
            cross_res = torch.einsum('bhlm,bhmd->bhld', attn_weights, V)
            cross_res = cross_res.transpose(1, 2).contiguous().view(B, L, D)
        else:
            # Use efficient SDPA when not visualizing
            cross_res = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            cross_res = cross_res.transpose(1,2).contiguous().view(B,L,D)
        # ========================================================================

        # linear proj
        cross_res = self.pc2img_attn.to_out[0](cross_res)
        # dropout
        cross_res = self.pc2img_attn.to_out[1](cross_res)
        # 手动调用 ModuleList 内的每个模块      
        #############################################################################
        #############################################################################
    
        #############################################################################

        cross_out = gate_msa.unsqueeze(1) * cross_res
        # # scale cross-attn contribution (可调)，此处示例不缩放额外系数
        # cross_out = self.drop_path(cross_res)
        # 把 cross-attn 的残差也加回 hidden_states（保持信息融合）
        norm_hidden_states = hidden_states + cross_out

        # # if (layer_idx == 0 or layer_idx == 3):
        # norm_hidden_states = self.part_aware_point(
        #     # point_feat=norm_hidden_states.to('cuda:2'),
        #     # verts_pos=point_pos.to('cuda:2'),

        #     point_feat=norm_hidden_states.cuda(),
        #     verts_pos=point_pos,
        #     # verts_pos=point_pos.to('cuda:1'),
        # ).to('cuda:1')

        # ###########===========================================  ??
        # # un_vis_pts = norm_hidden_states[~p_vis].unsqueeze(0)
        # # norm_hidden_states_unvis = self.norm_global(un_vis_pts) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        # # norm_hidden_states[~p_vis] = norm_hidden_states_unvis


        # # 2) 构造 mask: 不可见点 = True
        # unvis_mask = (~p_vis.bool())          # [B, N]

        # # 3) 先在整张图上算一版“unvis 变换后的特征”
        # #    norm_global 是对最后一维做的 MLP/LayerNorm 类操作: [*, C] → [*, C]
        # unvis_all = self.norm_global(norm_hidden_states)   # [B, N, C]
        # norm_hidden_states = unvis_all

        # # 4) 加上 scale / shift（按 batch 广播到 N）
        # #    scale_msa, shift_msa: [B, C]
        # scale = 1 + scale_msa[:, None, :]    # [B, 1, C] → broadcast 到 [B, N, C]
        # shift = shift_msa[:, None, :]        # [B, 1, C] → broadcast 到 [B, N, C]
        # unvis_all = unvis_all * scale + shift   # [B, N, C]

        # # 5) 只在不可见点位置替换
        # #    扩展 mask 到 [B, N, 1]，以便在特征维广播
        # mask_exp = unvis_mask.unsqueeze(-1)        # [B, N, 1]

        # norm_hidden_states = torch.where(
        #     mask_exp,          # True 的位置用 unvis_all，False 保留原值
        #     unvis_all,
        #     norm_hidden_states
        # )
        # ###########===========================================  ??

        # norm_hidden_states2 = self.norm2(hidden_states)
        #############################################################################
        # ⭐ use_mm_attn / use_self_attn 开关
        #   use_self_attn=True  → 纯 hidden_states 自注意力（复用 self.attn 的 Q/K/V 权重，跳过 Joint 拼接）
        #   use_mm_attn=True    → 原始 joint self-attention（hidden + encoder 拼接）
        #   否则                → 跳过 attn，直接用 cross-attn 残差
        #############################################################################
        if use_self_attn:
            # 纯 hidden_states 自注意力：复用 self.attn 的 to_q/to_k/to_v 权重
            Q_sa = self.attn.to_q(norm_hidden_states)
            K_sa = self.attn.to_k(norm_hidden_states)
            V_sa = self.attn.to_v(norm_hidden_states)

            B_sa, L_sa, D_sa = Q_sa.shape
            H_sa = self.attn.heads
            head_dim_sa = D_sa // H_sa
            Q_sa = Q_sa.view(B_sa, L_sa, H_sa, head_dim_sa).transpose(1, 2)
            K_sa = K_sa.view(B_sa, L_sa, H_sa, head_dim_sa).transpose(1, 2)
            V_sa = V_sa.view(B_sa, L_sa, H_sa, head_dim_sa).transpose(1, 2)

            # qk_norm
            if hasattr(self.attn, 'norm_q') and self.attn.norm_q is not None:
                Q_sa = self.attn.norm_q(Q_sa)
                K_sa = self.attn.norm_k(K_sa)

            # ==== Self-Attention Visualization ====
            # Skip explicit self-attention capture for long sequences to avoid
            # materializing a prohibitively large L x L attention matrix.
            if ATTN_VIZ_ENABLED and _ATTN_VISUALIZER is not None and L_sa <= 2000:
                scale_sa = 1.0 / (head_dim_sa ** 0.5)
                sa_logits = torch.einsum('bhld,bhmd->bhlm', Q_sa, K_sa) * scale_sa
                sa_weights = F.softmax(sa_logits, dim=-1)
                
                _ATTN_VISUALIZER.capture(
                    attn_scores=sa_weights,
                    layer_idx=layer_idx,
                    attn_type='self',
                    p_vis=p_vis,
                    proj_xy=proj_xy,
                    extra_info={'sampled': False, 'original_L': L_sa}
                )
                
                sa_output = torch.einsum('bhlm,bhmd->bhld', sa_weights, V_sa)
                sa_output = sa_output.transpose(1, 2).contiguous().view(B_sa, L_sa, D_sa)
            else:
                sa_output = F.scaled_dot_product_attention(Q_sa, K_sa, V_sa, dropout_p=0.0, is_causal=False)
                sa_output = sa_output.transpose(1, 2).contiguous().view(B_sa, L_sa, D_sa)
            # =======================================
            
            sa_output = self.attn.to_out[0](sa_output)
            sa_output = self.attn.to_out[1](sa_output)
            attn_output = gate_msa.unsqueeze(1) * sa_output
            hidden_states = hidden_states + attn_output
            context_attn_output = None
        elif use_mm_attn:
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output   # 2,12337,1024
            hidden_states = hidden_states + attn_output
        else:
            # 不使用 joint self-attn，直接用 cross-attn 残差
            hidden_states = norm_hidden_states
            context_attn_output = None

        # #############################################################################
        # # 加入的 交叉注意力
        # # norm_hidden_states2 = self.norm2(hidden_states)
        # # norm_encoder_hidden_states2 = self.norm2_context(encoder_hidden_states)
        # cross_res = self.pc2img_attn(
        #         # hidden_states=norm_hidden_states,            # 一开始的操作
        #         hidden_states=hidden_states,            # point queries
        #         encoder_hidden_states=norm_encoder_hidden_states,  # image keys/values
        #     )
        # cross_out = gate_msa.unsqueeze(1) * cross_res
        # # scale cross-attn contribution (可调)，此处示例不缩放额外系数
        # # cross_out = self.drop_path(cross_out)
        # # 把 cross-attn 的残差也加回 hidden_states（保持信息融合）
        # hidden_states = hidden_states + cross_out
        # #############################################################################


        # ffd

        norm_hidden_states = self.norm2(hidden_states)   # 2,12337,1024
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:   # this way
            ff_output = self.ff(norm_hidden_states)   # 2,12337,1024
            # moe_out, counts_or_loadloss = self.moe_ff(norm_hidden_states)
            
        # ff_output = gate_mlp.unsqueeze(1) * moe_out
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        # ff_output = self.drop_path(ff_output)
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        elif use_mm_attn and context_attn_output is not None:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output   # 2,1024,1024
            # context_attn_output = self.drop_path(context_attn_output)
            encoder_hidden_states = encoder_hidden_states + context_attn_output   # 2,1024,1024

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)   # 2,1024,1024
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
                + c_shift_mlp[:, None]
            )
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:    # this way
                context_ff_output = self.ff_context(norm_encoder_hidden_states)   # 2,1024,1024
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            )
            # encoder_hidden_states = self.drop_path(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + context_ff_output   # 2,1024,1024
        # else: use_mm_attn=False, encoder_hidden_states 直接透传不更新

        return hidden_states, encoder_hidden_states#, counts_or_loadloss




class SD3BodyHeadMMJointTransformerBlock(nn.Module):
    r"""
    BodyHead Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float,
        # num_attention_heads: int,
        # attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
    ):
        super().__init__()

        self.head_dit = SD3MMJointTransformerBlock(
            dim,  # 1024
            num_heads,   # 16
            eps,   # 1e-06
            context_pre_only=context_pre_only,   # False
            qk_norm=qk_norm,    # rms_norm
            use_dual_attention=use_dual_attention,    # False
        )
        self.body_dit = SD3MMJointTransformerBlock(
            dim,
            num_heads,
            eps,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
            use_dual_attention=use_dual_attention,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor = None,
        # is_hand: bool = False,
        **kwargs,
):
        """Default, last 1 / 4 is head"""

        _, N, _ = hidden_states.shape
        body_size = int(N * 0.75)

        body_hidden_states, head_hidden_states = (
            hidden_states[:, :body_size],
            hidden_states[:, body_size:],
        )

        if temb is not None:
            _, temb_N = temb.shape
            temb_size = temb_N // 2
            body_temb, head_temb = temb[:, :temb_size], temb[:, temb_size:]
            cond_dim=1536, #encoder_feat_dim,

        # body: 4096, head 1024, Sapiens & DINO
        body_encoder_hidden_states, head_encoder_hidden_states = (
            encoder_hidden_states[:, :4096],
            encoder_hidden_states[:, 4096:],
        )

        head_states, head_encoder_hidden_states = self.head_dit(
            head_hidden_states, head_encoder_hidden_states, head_temb
        )
        hidden_states = torch.cat([body_hidden_states, head_states], dim=1)
        hidden_states, body_encoder_hidden_states = self.body_dit(
            hidden_states, body_encoder_hidden_states, body_temb
        )

        if body_encoder_hidden_states is not None:
            encoder_hidden_states = torch.cat(
                [body_encoder_hidden_states, head_encoder_hidden_states], dim=1
            )
        else:
            encoder_hidden_states = None

        return hidden_states, encoder_hidden_states



    def forward_hand(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor = None,
    ):

        hand_temb = temb  # 1,1024
        hand_hidden_states = hidden_states  # 1,12337, 1024
        hand_encoder_hidden_states = encoder_hidden_states  # 1,4096,1024
        hand_states, hand_encoder_hidden_states = self.head_dit(
            hand_hidden_states, hand_encoder_hidden_states, hand_temb
        )

        return hand_states, hand_encoder_hidden_states
