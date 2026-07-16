from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

from plyfile import PlyData
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from LHM.models.modeling_hand_lrm import VisibilityBiasNet

VIS_BIAS_ENABLED = False

def set_vis_bias_enabled(enabled: bool):
    """Enable or disable visibility bias in pc2img cross-attention."""
    global VIS_BIAS_ENABLED
    VIS_BIAS_ENABLED = enabled
    print(f"[VisBias] {'Enabled' if enabled else 'Disabled'}")

def is_vis_bias_enabled() -> bool:
    """Return whether visibility bias is currently enabled."""
    return VIS_BIAS_ENABLED

assert hasattr(F, "scaled_dot_product_attention"), print(
    "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
)

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

        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        eps: float = 1e-5,

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

        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

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
        self.proj_in = nn.Linear(dim, inner_dim * 2)
        self.proj_out = nn.Linear(inner_dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.proj_in(x).chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.proj_out(x)
        return self.dropout(x)

def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
):
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

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

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

        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        hidden_states = attn.to_out[0](hidden_states)

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

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
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

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states)

            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + ff_output

        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

            if self._chunk_size is not None:
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)

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

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm_global = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.beta_local = 2.0
        self.beta_global = 2.0
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            self.ff_context = FeedForward(
                dim=dim, dim_out=dim, activation_fn="gelu-approximate"

            )
        else:
            self.norm2_context = None
            self.ff_context = None

        self._chunk_size = None
        self._chunk_dim = 0

        self.part_aware_point = HierarchicalPointEmbedTransformer(
            in_feat_dim=dim, out_dim=dim, part_dim=dim
        )

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def apply_visibility_bias(self, scores, vis_soft, beta_local=2.0, beta_global=2.0):
        """
        scores:    [B, H, N_p, 1025]
        vis_soft:  [B, N_p]  in [0,1]
        """

        B, H, N_p, N_k = scores.shape
        N_local = N_k - 1
        N_global = 1

        vis = vis_soft.clamp(0, 1)
        s = 2.0 * vis - 1.0

        local_bias_row = s * beta_local
        local_bias = local_bias_row.unsqueeze(-1).expand(-1, -1, N_local)

        global_bias_row = -s * beta_global
        global_bias = global_bias_row.unsqueeze(-1)

        bias_2d = torch.cat([local_bias, global_bias], dim=-1)

        bias = bias_2d.unsqueeze(1).to('cuda:1')

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
        N_local = M - 1
        N_global = 1

        vis = p_vis.clamp(0.0, 1.0)
        s = 2.0 * vis - 1.0

        local_bias_row = s * beta_local
        local_bias = local_bias_row.unsqueeze(-1).expand(-1, -1, N_local)

        global_bias_row = -s * beta_global
        global_bias = global_bias_row.unsqueeze(-1).expand(-1, -1, N_global)

        bias_2d = torch.cat([local_bias, global_bias], dim=-1)

        attn_mask = bias_2d.unsqueeze(1)
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

        b_local, b_global = self.vis_bias_net(p_vis, depth_res=depth_res)

        local_bias = b_local.unsqueeze(-1).expand(-1, -1, N_local)
        global_bias = b_global.unsqueeze(-1).expand(-1, -1, M-N_local)

        bias_2d = torch.cat([local_bias, global_bias], dim=-1)

        attn_mask = bias_2d.unsqueeze(1)
        return attn_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor = None,
        global_four: torch.FloatTensor = None,
        nail_mask: Optional[torch.FloatTensor] = None,
        nail_mask_3d: Optional[torch.FloatTensor] = None,
        proj_xy: Optional[torch.FloatTensor] = None,
        p_vis: Optional[torch.FloatTensor] = None,
        point_pos: Optional[torch.FloatTensor] = None,
        posed_pos: Optional[torch.FloatTensor] = None,
        feat_hw: Optional[Tuple[int,int]] = None,
        layer_idx: int = 0,
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
            raise ValueError("temb is required")

        use_self_attn = getattr(self, "use_self_attn", True)
        use_mm_attn = getattr(self, "use_mm_attn", False)
        beta_local = getattr(self, "beta_local", 2.0)
        beta_global = getattr(self, "beta_global", 2.0)

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        norm_global = self.norm_global(global_four)
        norm_global = norm_global.reshape(norm_global.size(0), -1, 1024)
        local_global = torch.cat((norm_encoder_hidden_states, norm_global), dim=1)

        Q = self.pc2img_attn.to_q(norm_hidden_states)
        K = self.pc2img_attn.to_k(local_global)
        V = self.pc2img_attn.to_v(local_global)

        B, L, D = Q.shape
        _, M, _ = K.shape
        H = self.pc2img_attn.heads
        head_dim = D // H
        scale = 1.0 / (head_dim ** 0.5)

        Q = Q.view(B, L, H, head_dim).transpose(1, 2)
        K = K.view(B, M, H, head_dim).transpose(1, 2)
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

            if not VIS_BIAS_ENABLED:
                attn_mask = torch.zeros_like(attn_mask)

        cross_res = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        cross_res = cross_res.transpose(1, 2).contiguous().view(B, L, D)

        cross_res = self.pc2img_attn.to_out[0](cross_res)

        cross_res = self.pc2img_attn.to_out[1](cross_res)

        cross_out = gate_msa.unsqueeze(1) * cross_res

        norm_hidden_states = hidden_states + cross_out

        if use_self_attn:
            Q_sa = self.attn.to_q(norm_hidden_states)
            K_sa = self.attn.to_k(norm_hidden_states)
            V_sa = self.attn.to_v(norm_hidden_states)

            B_sa, L_sa, D_sa = Q_sa.shape
            H_sa = self.attn.heads
            head_dim_sa = D_sa // H_sa
            Q_sa = Q_sa.view(B_sa, L_sa, H_sa, head_dim_sa).transpose(1, 2)
            K_sa = K_sa.view(B_sa, L_sa, H_sa, head_dim_sa).transpose(1, 2)
            V_sa = V_sa.view(B_sa, L_sa, H_sa, head_dim_sa).transpose(1, 2)

            if hasattr(self.attn, 'norm_q') and self.attn.norm_q is not None:
                Q_sa = self.attn.norm_q(Q_sa)
                K_sa = self.attn.norm_k(K_sa)

            sa_output = F.scaled_dot_product_attention(
                Q_sa,
                K_sa,
                V_sa,
                dropout_p=0.0,
                is_causal=False,
            )
            sa_output = sa_output.transpose(1, 2).contiguous().view(B_sa, L_sa, D_sa)

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

            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output
        else:
            hidden_states = norm_hidden_states
            context_attn_output = None

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        if self.context_pre_only:
            encoder_hidden_states = None
        elif use_mm_attn and context_attn_output is not None:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output

            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
                + c_shift_mlp[:, None]
            )
            if self._chunk_size is not None:
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            )

            encoder_hidden_states = encoder_hidden_states + context_ff_output

        return hidden_states, encoder_hidden_states

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

        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
    ):
        super().__init__()

        self.head_dit = SD3MMJointTransformerBlock(
            dim,
            num_heads,
            eps,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
            use_dual_attention=use_dual_attention,
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
            cond_dim=1536,

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
        hand_temb = temb
        hand_hidden_states = hidden_states
        hand_encoder_hidden_states = encoder_hidden_states
        hand_states, hand_encoder_hidden_states = self.head_dit(
            hand_hidden_states, hand_encoder_hidden_states, hand_temb
        )

        return hand_states, hand_encoder_hidden_states
