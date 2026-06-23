# -*- coding: utf-8 -*-
"""
Attention Score Visualization Utility for Ablation Study
用于消融实验的注意力分数可视化工具

Usage:
    from LHM.utils.attn_visualizer import AttnVisualizer
    
    visualizer = AttnVisualizer(save_dir='output/attn_viz')
    
    # In transformer forward pass, call:
    visualizer.capture(
        attn_scores=attn_scores,  # [B, H, L, M] attention weights after softmax
        layer_idx=layer_idx,
        attn_type='cross',  # 'cross' or 'self'
        p_vis=p_vis,  # [B, L] visibility scores
        proj_xy=proj_xy,  # [B, L, 2] point projections
    )
    
    # After inference, call:
    visualizer.visualize_all()
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
import csv


class AttnVisualizer:
    """Attention score visualizer for transformer ablation study."""
    
    def __init__(
        self,
        save_dir: str = 'output/attn_debug_v2',
        enabled: bool = True,
        capture_layers: Optional[List[int]] = None,  # None = all layers
        capture_heads: Optional[List[int]] = None,   # None = all heads
        max_points_to_viz: int = 1000,  # Subsample for efficiency
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            save_dir: Directory to save visualization outputs
            enabled: Whether to capture attention scores
            capture_layers: List of layer indices to capture (None = all)
            capture_heads: List of head indices to capture (None = all)
            max_points_to_viz: Max number of points to visualize
            image_size: Input image size for projection mapping
        """
        self.save_dir = save_dir
        self.enabled = enabled
        self.capture_layers = capture_layers
        self.capture_heads = capture_heads
        self.max_points_to_viz = max_points_to_viz
        self.image_size = image_size
        
        # Storage for captured attention data
        self.captured_data: List[Dict] = []
        self.metadata: Dict = {}

        # Optional per-image input data (RGB + mesh renderer) set via set_input_data()
        self._input_rgb = None        # np.ndarray [H,W,3] uint8
        self._mesh_render_fn = None   # callable () -> np.ndarray [H,W,3] uint8

        # Create save directory
        if enabled:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_dir = os.path.join(save_dir, f'session_{timestamp}')
            os.makedirs(self.session_dir, exist_ok=True)
    
    def set_input_data(self, input_rgb=None, mesh_render_fn=None):
        """Register per-image input data to be saved alongside attention visualizations.

        Args:
            input_rgb: np.ndarray [H, W, 3] uint8 — the original input RGB image.
            mesh_render_fn: callable () -> np.ndarray [H, W, 3] uint8 — renders posed mesh.
        """
        self._input_rgb = input_rgb
        self._mesh_render_fn = mesh_render_fn

    def should_capture(self, layer_idx: int, head_idx: Optional[int] = None) -> bool:
        """Check if this layer/head should be captured."""
        if not self.enabled:
            return False
        if self.capture_layers is not None and layer_idx not in self.capture_layers:
            return False
        if head_idx is not None and self.capture_heads is not None:
            if head_idx not in self.capture_heads:
                return False
        return True
    
    def capture(
        self,
        attn_scores: torch.Tensor,  # [B, H, L, M] or [B, L, M]
        layer_idx: int,
        attn_type: str = 'cross',  # 'cross' or 'self'
        p_vis: Optional[torch.Tensor] = None,  # [B, L]
        proj_xy: Optional[torch.Tensor] = None,  # [B, L, 2]
        point_pos: Optional[torch.Tensor] = None,  # [B, L, 3]
        posed_pos: Optional[torch.Tensor] = None,  # [B, L, 3]
        step: int = 0,
        extra_info: Optional[Dict] = None,
    ):
        """
        Capture attention scores for visualization.
        
        Args:
            attn_scores: Attention weights [B, H, L, M] or [B, L, M]
            layer_idx: Index of the transformer layer
            attn_type: Type of attention ('cross' for pc2img, 'self' for self-attn)
            p_vis: Point visibility scores [B, L] in [0,1]
            proj_xy: 2D projections of points [B, L, 2]
            point_pos: 3D positions of points [B, L, 3]
            step: Training/inference step
            extra_info: Additional info to store
        """
        if not self.should_capture(layer_idx):
            return
        
        # Detach and move to CPU
        attn_np = attn_scores.detach().cpu().numpy()
        
        data = {
            'layer_idx': layer_idx,
            'attn_type': attn_type,
            'step': step,
            'attn_shape': list(attn_scores.shape),
            'attn_scores': attn_np,
        }
        
        if p_vis is not None:
            data['p_vis'] = p_vis.detach().cpu().numpy()
        if proj_xy is not None:
            data['proj_xy'] = proj_xy.detach().cpu().numpy()
        if point_pos is not None:
            data['point_pos'] = point_pos.detach().cpu().numpy()
        if posed_pos is not None:
            data['posed_pos'] = posed_pos.detach().cpu().numpy()
        if extra_info is not None:
            data['extra_info'] = extra_info
        
        self.captured_data.append(data)
    
    def compute_attn_entropy(self, attn: np.ndarray) -> np.ndarray:
        """
        Compute entropy of attention distribution.
        High entropy = attention is spread across many tokens
        Low entropy = attention is focused on few tokens
        """
        # attn: [..., M] where M is number of keys
        eps = 1e-10
        log_attn = np.log(attn + eps)
        entropy = -np.sum(attn * log_attn, axis=-1)
        return entropy
    
    def compute_local_vs_global_ratio(
        self,
        attn: np.ndarray,
        n_local: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ratio of attention to local vs global tokens.
        
        Returns:
            local_attn_sum: Sum of attention to local tokens (first n_local)
            global_attn_sum: Sum of attention to global token (last token)
        """
        # attn: [..., M] where M = n_local + 1
        local_attn_sum = np.sum(attn[..., :n_local], axis=-1)
        global_attn_sum = np.sum(attn[..., n_local:], axis=-1)
        return local_attn_sum, global_attn_sum

    def _safe_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            return float("nan")
        x = x[valid]
        y = y[valid]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def compute_local_global_ratio_maps(
        self,
        attn: np.ndarray,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pointwise local/global relative ratios from post-softmax attention.

        attn: [B, H, L, M], where keys are [local tokens..., global token].
        Returns per-point maps averaged over heads: [B, L].
        """
        if attn.ndim == 3:
            attn = attn[:, np.newaxis, :, :]
        n_local = attn.shape[-1] - 1
        local_attn = attn[..., :n_local].sum(axis=-1)       # [B, H, L]
        global_attn = attn[..., n_local:].sum(axis=-1)      # [B, H, L]
        denom = local_attn + global_attn + eps
        local_ratio = (local_attn / denom).mean(axis=1)     # [B, L]
        global_ratio = (global_attn / denom).mean(axis=1)   # [B, L]
        local_attn_mean = local_attn.mean(axis=1)           # [B, L]
        global_attn_mean = global_attn.mean(axis=1)         # [B, L]
        return local_ratio, global_ratio, local_attn_mean, global_attn_mean

    def compute_visibility_ratio_stats(
        self,
        p_vis: np.ndarray,
        local_ratio: np.ndarray,
        global_ratio: np.ndarray,
    ) -> Dict:
        p_vis = np.asarray(p_vis, dtype=np.float64)
        if p_vis.ndim == 3 and p_vis.shape[-1] == 1:
            p_vis = p_vis[..., 0]
        visible = p_vis > 0.7
        invisible = p_vis < 0.3

        def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
            if not np.any(mask):
                return float("nan")
            return float(np.asarray(values)[mask].mean())

        return {
            "p_vis_min": float(np.nanmin(p_vis)),
            "p_vis_max": float(np.nanmax(p_vis)),
            "p_vis_mean": float(np.nanmean(p_vis)),
            "visible_points_count_p_vis_gt_0.7": int(visible.sum()),
            "invisible_points_count_p_vis_lt_0.3": int(invisible.sum()),
            "visible_global_ratio_mean": masked_mean(global_ratio, visible),
            "visible_local_ratio_mean": masked_mean(local_ratio, visible),
            "invisible_global_ratio_mean": masked_mean(global_ratio, invisible),
            "invisible_local_ratio_mean": masked_mean(local_ratio, invisible),
            "correlation_p_vis_local_ratio": self._safe_corr(p_vis, local_ratio),
            "correlation_p_vis_global_ratio": self._safe_corr(p_vis, global_ratio),
        }

    def _plot_projected_points(
        self,
        values: np.ndarray,
        title: str,
        out_path: str,
        proj_xy: Optional[np.ndarray] = None,
        value_label: str = "value",
        vmin: float = 0.0,
        vmax: float = 1.0,
        cmap: str = "viridis",
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        if proj_xy is not None:
            sc = ax.scatter(
                proj_xy[:, 0],
                proj_xy[:, 1],
                c=values,
                s=2,
                alpha=0.75,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
            )
            ax.set_xlim(0, self.image_size[0])
            ax.set_ylim(self.image_size[1], 0)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            sc = ax.scatter(
                np.arange(values.shape[0]),
                values,
                c=values,
                s=2,
                alpha=0.75,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
            )
            ax.set_xlabel("point index")
            ax.set_ylabel(value_label)
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label=value_label)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close(fig)

    def _project_3d_points(self, points: np.ndarray, view: str) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[-1] < 3:
            return np.empty((0, 2), dtype=np.float64)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if view == "canonical_back":
            xy = np.stack([x, -z], axis=-1)
        elif view == "canonical_palm":
            xy = np.stack([-x, -z], axis=-1)
        elif view == "posed_rotx-90":
            xy = np.stack([x, -z], axis=-1)
        else:
            xy = np.stack([x, -y], axis=-1)

        valid = np.isfinite(xy).all(axis=1)
        if valid.sum() < 2:
            return xy

        xy_valid = xy[valid]
        min_xy = xy_valid.min(axis=0)
        max_xy = xy_valid.max(axis=0)
        center = (min_xy + max_xy) * 0.5
        extent = float(np.max(max_xy - min_xy))
        if extent < 1e-12:
            extent = 1.0

        width, height = self.image_size
        scale = 0.88 * min(width, height) / extent
        out = (xy - center) * scale
        out[:, 0] += width * 0.5
        out[:, 1] += height * 0.5
        return out

    def _project_3d_view(self, points: np.ndarray, view: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[-1] < 3:
            return np.empty((0, 2), dtype=np.float64), None, "max"

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if view == "canonical_back":
            coords_raw = np.stack([x, -z], axis=-1)
            depth = y
        elif view == "canonical_palm":
            coords_raw = np.stack([-x, -z], axis=-1)
            depth = -y
        elif view == "posed_rotx-90":
            coords_raw = np.stack([x, -z], axis=-1)
            depth = y
        else:
            coords_raw = np.stack([x, -y], axis=-1)
            depth = z

        valid = np.isfinite(coords_raw).all(axis=1)
        if valid.sum() < 2:
            return coords_raw, depth, "max"

        xy_valid = coords_raw[valid]
        min_xy = xy_valid.min(axis=0)
        max_xy = xy_valid.max(axis=0)
        center = (min_xy + max_xy) * 0.5
        extent = float(np.max(max_xy - min_xy))
        if extent < 1e-12:
            extent = 1.0

        width, height = self.image_size
        scale = 0.88 * min(width, height) / extent
        coords = (coords_raw - center) * scale
        coords[:, 0] += width * 0.5
        coords[:, 1] += height * 0.5
        return coords, depth, "max"

    def _plot_heatmap_map(
        self,
        values: np.ndarray,
        out_path: str,
        coords: np.ndarray,
        depth: Optional[np.ndarray] = None,
        near: str = "max",
        cmap: str = "jet",
        radius: int = 4,
        sigma: float = 1.8,
    ):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] != values.shape[0]:
            return

        valid = np.isfinite(values) & np.isfinite(coords).all(axis=1)
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float64).reshape(-1)
            valid &= np.isfinite(depth)
        values = np.clip(values[valid], 0.0, 1.0)
        coords = coords[valid]
        depth_valid = depth[valid] if depth is not None else None

        width, height = self.image_size
        in_frame = (
            (coords[:, 0] >= -radius)
            & (coords[:, 0] < width + radius)
            & (coords[:, 1] >= -radius)
            & (coords[:, 1] < height + radius)
        )
        values = values[in_frame]
        coords = coords[in_frame]
        if depth_valid is not None:
            depth_valid = depth_valid[in_frame]

        num = np.zeros((height, width), dtype=np.float64)
        den = np.zeros((height, width), dtype=np.float64)
        if values.size:
            xi = np.rint(coords[:, 0]).astype(np.int64)
            yi = np.rint(coords[:, 1]).astype(np.int64)
            offsets = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    d2 = dx * dx + dy * dy
                    if d2 <= radius * radius:
                        offsets.append((dx, dy, np.exp(-0.5 * d2 / (sigma * sigma))))

            best_depth = None
            depth_tol = 0.0
            if depth_valid is not None and depth_valid.size:
                if near == "min":
                    best_depth = np.full((height, width), np.inf, dtype=np.float64)
                    reducer = np.minimum.at
                else:
                    best_depth = np.full((height, width), -np.inf, dtype=np.float64)
                    reducer = np.maximum.at
                for dx, dy, _ in offsets:
                    xs = xi + dx
                    ys = yi + dy
                    ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                    if np.any(ok):
                        reducer(best_depth, (ys[ok], xs[ok]), depth_valid[ok])
                depth_range = float(np.nanmax(depth_valid) - np.nanmin(depth_valid))
                depth_tol = max(depth_range * 0.025, 1e-6)

            for dx, dy, w in offsets:
                xs = xi + dx
                ys = yi + dy
                ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                if not np.any(ok):
                    continue
                if best_depth is not None:
                    pixel_best = best_depth[ys[ok], xs[ok]]
                    if near == "min":
                        z_ok = depth_valid[ok] <= pixel_best + depth_tol
                    else:
                        z_ok = depth_valid[ok] >= pixel_best - depth_tol
                    ok_indices = np.where(ok)[0][z_ok]
                    if ok_indices.size == 0:
                        continue
                    xs_ok = xs[ok_indices]
                    ys_ok = ys[ok_indices]
                    values_ok = values[ok_indices]
                else:
                    xs_ok = xs[ok]
                    ys_ok = ys[ok]
                    values_ok = values[ok]
                np.add.at(num, (ys_ok, xs_ok), values_ok * w)
                np.add.at(den, (ys_ok, xs_ok), w)

        heat = np.zeros((height, width), dtype=np.float64)
        mask = den > 1e-8
        heat[mask] = num[mask] / den[mask]

        cmap_fn = plt.get_cmap(cmap)
        rgb = (cmap_fn(np.clip(heat, 0.0, 1.0))[..., :3] * 255.0).astype(np.uint8)
        rgb[~mask] = 0
        plt.imsave(out_path, rgb)

    def _plot_point_map(
        self,
        values: np.ndarray,
        out_path: str,
        coords: np.ndarray,
        title: str,
        value_label: str,
        cmap: str = "viridis",
    ):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        valid = np.isfinite(coords).all(axis=1) & np.isfinite(values)
        if np.any(valid):
            sc = ax.scatter(
                coords[valid, 0],
                coords[valid, 1],
                c=values[valid],
                s=2,
                alpha=0.8,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                linewidths=0,
            )
            plt.colorbar(sc, ax=ax, label=value_label)
        ax.set_title(title)
        ax.set_xlim(0, self.image_size[0])
        ax.set_ylim(self.image_size[1], 0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        plt.tight_layout(pad=0.1)
        plt.savefig(out_path, dpi=220)
        plt.close(fig)

    def _stats_for_batch(
        self,
        p_vis: np.ndarray,
        local_ratio: np.ndarray,
        global_ratio: np.ndarray,
        layer_idx: int,
        image_name: str,
    ) -> Dict:
        stats = self.compute_visibility_ratio_stats(p_vis, local_ratio, global_ratio)
        stats.update({
            "image_name": image_name,
            "layer_idx": int(layer_idx),
            "num_visible_points": stats["visible_points_count_p_vis_gt_0.7"],
            "num_invisible_points": stats["invisible_points_count_p_vis_lt_0.3"],
            "corr_pvis_local_ratio": stats["correlation_p_vis_local_ratio"],
            "corr_pvis_global_ratio": stats["correlation_p_vis_global_ratio"],
            "ratio_definition": "global_ratio=global_attention/(global_attention+local_attention+eps), local_ratio=local_attention/(global_attention+local_attention+eps)",
        })
        return stats

    def visualize_canonical_outputs(
        self,
        data: Dict,
        root_dir: str,
        image_name: str,
        layer_subdir: Optional[str] = None,
        file_prefix: Optional[str] = None,
    ) -> List[Dict]:
        if data.get("attn_type") != "cross" or "p_vis" not in data:
            return []

        attn = data["attn_scores"]
        if attn.ndim == 3:
            attn = attn[:, np.newaxis, :, :]
        if attn.shape[-1] < 2:
            return []

        p_vis = data["p_vis"]
        if p_vis.ndim == 3 and p_vis.shape[-1] == 1:
            p_vis = p_vis[..., 0]
        local_ratio, global_ratio, _, _ = self.compute_local_global_ratio_maps(attn)

        proj_xy = data.get("proj_xy")
        point_pos = data.get("point_pos")
        posed_pos = data.get("posed_pos")
        if posed_pos is None:
            posed_pos = point_pos

        rows = []
        batch_count = min(p_vis.shape[0], local_ratio.shape[0], global_ratio.shape[0])
        for b in range(batch_count):
            sample_name = image_name if batch_count == 1 else f"{image_name}_b{b}"
            out_dir = os.path.join(root_dir, sample_name)
            if layer_subdir:
                out_dir = os.path.join(out_dir, layer_subdir)
            posed_dir = os.path.join(out_dir, "posed")
            os.makedirs(posed_dir, exist_ok=True)

            coords_orig = proj_xy[b] if proj_xy is not None else None
            if coords_orig is None:
                coords_orig = self._project_3d_points(posed_pos[b], "canonical_front")
            if posed_pos is not None:
                coords_rotx, depth_rotx, near_rotx = self._project_3d_view(posed_pos[b], "posed_rotx-90")
            else:
                coords_rotx, depth_rotx, near_rotx = coords_orig, None, "max"
            if point_pos is not None:
                coords_back, depth_back, near_back = self._project_3d_view(point_pos[b], "canonical_back")
                coords_palm, depth_palm, near_palm = self._project_3d_view(point_pos[b], "canonical_palm")
            else:
                coords_back, depth_back, near_back = coords_orig, None, "max"
                coords_palm, depth_palm, near_palm = coords_orig, None, "max"

            maps = {
                "global_ratio": (global_ratio[b], "global_ratio"),
                "local_ratio": (local_ratio[b], "local_ratio"),
                "pvis": (p_vis[b], "p_vis"),
            }
            prefix = file_prefix or sample_name
            for key, (values, label) in maps.items():
                self._plot_heatmap_map(values, os.path.join(posed_dir, f"orig_{key}.jpg"), coords_orig)
                self._plot_heatmap_map(values, os.path.join(posed_dir, f"rotx-90_{key}.jpg"), coords_rotx, depth=depth_rotx, near=near_rotx)
                self._plot_heatmap_map(values, os.path.join(out_dir, f"canonical_back_{key}.jpg"), coords_back, depth=depth_back, near=near_back)
                self._plot_heatmap_map(values, os.path.join(out_dir, f"canonical_palm_{key}.jpg"), coords_palm, depth=depth_palm, near=near_palm)
                self._plot_heatmap_map(values, os.path.join(out_dir, f"{prefix}-{key}-orig.jpg"), coords_orig)
                self._plot_heatmap_map(values, os.path.join(out_dir, f"{prefix}-{key}-rotx-90.jpg"), coords_rotx, depth=depth_rotx, near=near_rotx)

            stats = self._stats_for_batch(p_vis[b], local_ratio[b], global_ratio[b], data["layer_idx"], sample_name)
            stats["attn_shape"] = list(attn.shape)
            stats["n_local_tokens"] = int(attn.shape[-1] - 1)
            stats["has_posed_pos"] = posed_pos is not None
            stats["has_point_pos"] = point_pos is not None
            with open(os.path.join(out_dir, "stats.json"), "w") as f:
                json.dump(stats, f, indent=2)
            rows.append(stats)

            # --- Save input RGB and posed mesh alongside attention maps ---
            if self._input_rgb is not None:
                try:
                    from PIL import Image as _PIL_Image
                    rgb_save_path = os.path.join(out_dir, f"{prefix}_input_rgb.png")
                    if not os.path.exists(rgb_save_path):
                        _PIL_Image.fromarray(self._input_rgb).save(rgb_save_path)
                except Exception as _e:
                    print(f"[AttnViz] input_rgb save failed: {_e}")
            if self._mesh_render_fn is not None:
                try:
                    from PIL import Image as _PIL_Image
                    mesh_save_path = os.path.join(out_dir, f"{prefix}_posed_mesh.png")
                    if not os.path.exists(mesh_save_path):
                        _mesh_img = self._mesh_render_fn()
                        _PIL_Image.fromarray(_mesh_img).save(mesh_save_path)
                except Exception as _e:
                    print(f"[AttnViz] posed_mesh save failed: {_e}")
            # ---------------------------------------------------------------
        return rows

    def append_summary_csv(self, csv_path: str, rows: List[Dict]):
        if not rows:
            return
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = [
            "image_name",
            "layer_idx",
            "visible_global_ratio_mean",
            "visible_local_ratio_mean",
            "invisible_global_ratio_mean",
            "invisible_local_ratio_mean",
            "corr_pvis_local_ratio",
            "corr_pvis_global_ratio",
            "num_visible_points",
            "num_invisible_points",
            "p_vis_min",
            "p_vis_max",
            "p_vis_mean",
        ]
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def visualize_visibility_ratio_v2(
        self,
        data: Dict,
        output_prefix: str,
    ) -> Optional[Dict]:
        """Save physically comparable local/global ratio maps and visibility stats."""
        if data.get("attn_type") != "cross":
            return None
        if "p_vis" not in data or data["p_vis"] is None:
            return None

        attn = data["attn_scores"]
        if attn.ndim == 3:
            attn = attn[:, np.newaxis, :, :]
        if attn.shape[-1] < 2:
            return None

        p_vis = data["p_vis"]
        if p_vis.ndim == 3 and p_vis.shape[-1] == 1:
            p_vis = p_vis[..., 0]

        local_ratio, global_ratio, local_attn, global_attn = self.compute_local_global_ratio_maps(attn)
        stats = self.compute_visibility_ratio_stats(p_vis, local_ratio, global_ratio)
        stats.update({
            "layer_idx": int(data["layer_idx"]),
            "attn_type": data["attn_type"],
            "attn_shape": list(attn.shape),
            "n_local_tokens": int(attn.shape[-1] - 1),
            "ratio_definition": "global_ratio=global_attention/(global_attention+local_attention+eps), local_ratio=local_attention/(global_attention+local_attention+eps)",
            "head_reduction": "ratio computed per head, then averaged over heads",
        })

        proj_xy = data.get("proj_xy")
        batch_count = min(p_vis.shape[0], local_ratio.shape[0], global_ratio.shape[0])
        for b in range(batch_count):
            proj_b = None
            if proj_xy is not None:
                proj_b = proj_xy[b]

            batch_prefix = f"{output_prefix}_layer{data['layer_idx']}_b{b}"
            self._plot_projected_points(
                p_vis[b],
                f"Layer {data['layer_idx']} p_vis",
                f"{batch_prefix}_p_vis.png",
                proj_xy=proj_b,
                value_label="p_vis",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            self._plot_projected_points(
                global_ratio[b],
                f"Layer {data['layer_idx']} global_ratio",
                f"{batch_prefix}_global_ratio.png",
                proj_xy=proj_b,
                value_label="global_ratio",
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
            )
            self._plot_projected_points(
                local_ratio[b],
                f"Layer {data['layer_idx']} local_ratio",
                f"{batch_prefix}_local_ratio.png",
                proj_xy=proj_b,
                value_label="local_ratio",
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(p_vis[b].reshape(-1), bins=50, range=(0, 1), color="tab:green")
            axes[0].set_title("p_vis")
            axes[0].set_xlim(0, 1)
            axes[1].hist(global_ratio[b].reshape(-1), bins=50, range=(0, 1), color="tab:red")
            axes[1].set_title("global_ratio")
            axes[1].set_xlim(0, 1)
            axes[2].scatter(p_vis[b].reshape(-1), local_ratio[b].reshape(-1), s=2, alpha=0.35)
            axes[2].set_xlabel("p_vis")
            axes[2].set_ylabel("local_ratio")
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
            axes[2].set_title("p_vis vs local_ratio")
            plt.tight_layout()
            plt.savefig(f"{batch_prefix}_stats_panel.png", dpi=180)
            plt.close(fig)

        np.savez_compressed(
            f"{output_prefix}_layer{data['layer_idx']}_ratio_maps.npz",
            p_vis=p_vis,
            local_ratio=local_ratio,
            global_ratio=global_ratio,
            local_attention=local_attn,
            global_attention=global_attn,
            proj_xy=proj_xy if proj_xy is not None else np.array([]),
            point_pos=data.get("point_pos", np.array([])),
            posed_pos=data.get("posed_pos", np.array([])),
        )

        with open(f"{output_prefix}_layer{data['layer_idx']}_ratio_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        return stats
    
    def visualize_layer(
        self,
        data: Dict,
        output_prefix: str,
    ):
        """Visualize attention for a single layer."""
        attn = data['attn_scores']  # [B, H, L, M] or [B, L, M]
        layer_idx = data['layer_idx']
        attn_type = data['attn_type']
        
        # Handle different shapes
        if attn.ndim == 3:
            # [B, L, M] -> add head dim
            attn = attn[:, np.newaxis, :, :]
        
        B, H, L, M = attn.shape
        n_local = M - 1  # Last token is global
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Layer {layer_idx} - {attn_type.upper()} Attention Analysis', fontsize=16)
        
        # ========== 1. Attention entropy per head ==========
        ax = axes[0, 0]
        entropy = self.compute_attn_entropy(attn)  # [B, H, L]
        entropy_mean = entropy.mean(axis=(0, 2))  # [H]
        ax.bar(range(H), entropy_mean)
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Mean Entropy')
        ax.set_title('Attention Entropy per Head')
        ax.axhline(y=np.log(M), color='r', linestyle='--', label=f'Max Entropy (log({M}))')
        ax.legend()
        
        # ========== 2. Local vs Global attention ratio ==========
        ax = axes[0, 1]
        local_attn, global_attn = self.compute_local_vs_global_ratio(attn, n_local)
        # Average over batch and query tokens
        local_mean = local_attn.mean(axis=(0, 2))  # [H]
        global_mean = global_attn.mean(axis=(0, 2))  # [H]
        
        x = np.arange(H)
        width = 0.35
        ax.bar(x - width/2, local_mean, width, label='Local (1-1024)')
        ax.bar(x + width/2, global_mean, width, label='Global (1025)')
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Mean Attention Weight')
        ax.set_title('Local vs Global Attention per Head')
        ax.legend()
        
        # ========== 3. If visibility info available, stratify by visibility ==========
        ax = axes[0, 2]
        if 'p_vis' in data and data['p_vis'] is not None:
            p_vis = data['p_vis']  # [B, L]
            vis_threshold = 0.5
            
            # Compute local/global ratio stratified by visibility
            vis_mask = p_vis > vis_threshold  # [B, L]
            
            vis_local = []
            vis_global = []
            invis_local = []
            invis_global = []
            
            for b in range(B):
                for h in range(H):
                    vis_pts = vis_mask[b]
                    local_sum = attn[b, h, :, :n_local].sum(axis=-1)  # [L]
                    global_sum = attn[b, h, :, n_local:].sum(axis=-1)  # [L]
                    
                    vis_local.append(local_sum[vis_pts].mean() if vis_pts.any() else 0)
                    vis_global.append(global_sum[vis_pts].mean() if vis_pts.any() else 0)
                    invis_local.append(local_sum[~vis_pts].mean() if (~vis_pts).any() else 0)
                    invis_global.append(global_sum[~vis_pts].mean() if (~vis_pts).any() else 0)
            
            vis_local = np.array(vis_local).reshape(B, H).mean(axis=0)
            vis_global = np.array(vis_global).reshape(B, H).mean(axis=0)
            invis_local = np.array(invis_local).reshape(B, H).mean(axis=0)
            invis_global = np.array(invis_global).reshape(B, H).mean(axis=0)
            
            x = np.arange(H)
            width = 0.2
            ax.bar(x - 1.5*width, vis_local, width, label='Visible→Local', color='blue')
            ax.bar(x - 0.5*width, vis_global, width, label='Visible→Global', color='lightblue')
            ax.bar(x + 0.5*width, invis_local, width, label='Invisible→Local', color='red')
            ax.bar(x + 1.5*width, invis_global, width, label='Invisible→Global', color='lightcoral')
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Mean Attention Weight')
            ax.set_title('Attention Stratified by Visibility')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No visibility data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Visibility Stratification (N/A)')
        
        # ========== 4. Attention heatmap (averaged over batch, sample heads) ==========
        ax = axes[1, 0]
        # Sample points for visualization
        sample_pts = min(self.max_points_to_viz, L)
        pt_indices = np.linspace(0, L-1, sample_pts, dtype=int)
        
        # Average attention over batch, show first head
        attn_sample = attn[0, 0, pt_indices, :]  # [sample_pts, M]
        im = ax.imshow(attn_sample, aspect='auto', cmap='viridis')
        ax.set_xlabel('Key Token Index')
        ax.set_ylabel('Query Point (sampled)')
        ax.set_title(f'Attention Heatmap (Head 0)')
        plt.colorbar(im, ax=ax)
        
        # ========== 5. Top-k attention visualization ==========
        ax = axes[1, 1]
        top_k = 10
        attn_flat = attn[0, 0, :, :n_local]  # [L, n_local] - only local tokens
        top_attn_indices = np.argpartition(attn_flat, -top_k, axis=-1)[:, -top_k:]  # [L, top_k]
        
        # Histogram of top-k token indices
        ax.hist(top_attn_indices.flatten(), bins=min(50, n_local), density=True)
        ax.set_xlabel('Local Token Index')
        ax.set_ylabel('Frequency in Top-k')
        ax.set_title(f'Top-{top_k} Attended Local Tokens Distribution')
        
        # ========== 6. 2D projection with attention overlay ==========
        ax = axes[1, 2]
        if 'proj_xy' in data and data['proj_xy'] is not None:
            proj_xy = data['proj_xy'][0]  # [L, 2]
            # Color by global attention ratio
            global_ratio = attn[0, 0, :, n_local:].sum(axis=-1)  # [L]
            
            sc = ax.scatter(
                proj_xy[:, 0], proj_xy[:, 1],
                c=global_ratio, cmap='coolwarm',
                s=1, alpha=0.5
            )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Global Attention Ratio in 2D Projection')
            ax.set_xlim(0, self.image_size[0])
            ax.set_ylim(self.image_size[1], 0)  # Flip y-axis
            plt.colorbar(sc, ax=ax, label='Global Attn')
        else:
            ax.text(0.5, 0.5, 'No projection data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('2D Projection (N/A)')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_layer{layer_idx}.png', dpi=150)
        plt.close(fig)
        
        # Save numerical statistics
        stats = {
            'layer_idx': layer_idx,
            'attn_type': attn_type,
            'shape': list(attn.shape),
            'entropy_per_head': entropy_mean.tolist(),
            'local_attn_per_head': local_mean.tolist(),
            'global_attn_per_head': global_mean.tolist(),
            'step': data.get('step', 0),
        }
        
        with open(f'{output_prefix}_layer{layer_idx}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def visualize_spatial_attention(
        self,
        data: Dict,
        output_prefix: str,
        input_image: Optional[np.ndarray] = None,
    ):
        """
        Visualize attention as spatial heatmap on the point cloud projection.
        
        This shows WHERE points are attending to in the 2D image.
        """
        attn = data['attn_scores']  # [B, H, L, M]
        if attn.ndim == 3:
            attn = attn[:, np.newaxis, :, :]
        
        B, H, L, M = attn.shape
        n_local = M - 1
        
        # Reshape local attention to spatial form (32x32 feature map)
        feat_h, feat_w = 32, 32
        assert n_local == feat_h * feat_w, f"Expected {feat_h*feat_w} local tokens, got {n_local}"
        
        # Sample some points to visualize
        n_sample = 8
        sample_indices = np.linspace(0, L-1, n_sample, dtype=int)
        
        fig, axes = plt.subplots(2, n_sample, figsize=(n_sample * 3, 6))
        fig.suptitle(f'Spatial Attention Maps (Layer {data["layer_idx"]})', fontsize=14)
        
        for i, pt_idx in enumerate(sample_indices):
            # Row 1: Attention heatmap on feature map grid
            ax = axes[0, i]
            attn_map = attn[0, 0, pt_idx, :n_local].reshape(feat_h, feat_w)
            im = ax.imshow(attn_map, cmap='hot')
            ax.set_title(f'Pt {pt_idx}')
            ax.axis('off')
            
            # Row 2: If we have visibility info, show it
            if 'p_vis' in data and data['p_vis'] is not None:
                ax = axes[1, i]
                p_vis_pt = data['p_vis'][0, pt_idx]
                global_attn = attn[0, 0, pt_idx, n_local:].sum()
                ax.text(0.5, 0.5, f'vis={p_vis_pt:.2f}\nglobal={global_attn:.3f}',
                       ha='center', va='center', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_spatial_attn.png', dpi=150)
        plt.close(fig)
    
    def visualize_all(
        self,
        input_image: Optional[np.ndarray] = None,
        canonical_root_dir: Optional[str] = None,
        image_name: Optional[str] = None,
        layer_subdir: Optional[str] = None,
        file_prefix: Optional[str] = None,
    ):
        """Visualize all captured attention data."""
        if not self.captured_data:
            print("[AttnVisualizer] No data captured!")
            return
        
        print(f"[AttnVisualizer] Visualizing {len(self.captured_data)} captured attention maps...")
        
        all_stats = []
        canonical_rows = []
        for i, data in enumerate(self.captured_data):
            output_prefix = os.path.join(self.session_dir, f'step{data.get("step", 0)}')
            
            # Main analysis visualization
            stats = self.visualize_layer(data, output_prefix)
            ratio_stats = self.visualize_visibility_ratio_v2(data, output_prefix)
            if ratio_stats is not None:
                stats["visibility_ratio_v2"] = ratio_stats
            all_stats.append(stats)

            if canonical_root_dir is not None and image_name is not None:
                canonical_rows.extend(
                    self.visualize_canonical_outputs(
                        data,
                        root_dir=canonical_root_dir,
                        image_name=image_name,
                        layer_subdir=layer_subdir,
                        file_prefix=file_prefix,
                    )
                )
            
            # Spatial attention visualization
            self.visualize_spatial_attention(data, output_prefix, input_image)
        
        # Save summary
        summary = {
            'n_captures': len(self.captured_data),
            'session_dir': self.session_dir,
            'all_stats': all_stats,
            'canonical_rows': canonical_rows,
        }
        with open(os.path.join(self.session_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[AttnVisualizer] Saved visualizations to: {self.session_dir}")

        # --- Save input RGB and posed mesh to canonical_root_dir (always, if set) ---
        if canonical_root_dir is not None and image_name is not None and (
            self._input_rgb is not None or self._mesh_render_fn is not None
        ):
            _prefix = file_prefix or image_name
            _out_dir = os.path.join(canonical_root_dir, image_name)
            os.makedirs(_out_dir, exist_ok=True)
            if self._input_rgb is not None:
                try:
                    from PIL import Image as _PIL_Image
                    _rgb_path = os.path.join(_out_dir, f"{_prefix}_input_rgb.png")
                    _PIL_Image.fromarray(self._input_rgb).save(_rgb_path)
                    print(f"[AttnViz] Saved input RGB → {_rgb_path}")
                except Exception as _e:
                    print(f"[AttnViz] input_rgb save failed: {_e}")
            if self._mesh_render_fn is not None:
                try:
                    from PIL import Image as _PIL_Image
                    _mesh_path = os.path.join(_out_dir, f"{_prefix}_posed_mesh.png")
                    _PIL_Image.fromarray(self._mesh_render_fn()).save(_mesh_path)
                    print(f"[AttnViz] Saved posed mesh → {_mesh_path}")
                except Exception as _e:
                    print(f"[AttnViz] posed_mesh save failed: {_e}")
        # -------------------------------------------------------------------------

        return summary
    
    def clear(self):
        """Clear captured data."""
        self.captured_data = []
        self.metadata = {}
    
    def save_raw_attention(self, filename: str = 'raw_attention.npz'):
        """Save raw attention data for offline analysis."""
        if not self.captured_data:
            return
        
        save_dict = {}
        for i, data in enumerate(self.captured_data):
            prefix = f'cap{i}'
            save_dict[f'{prefix}_attn'] = data['attn_scores']
            save_dict[f'{prefix}_layer'] = data['layer_idx']
            if 'p_vis' in data:
                save_dict[f'{prefix}_pvis'] = data['p_vis']
            if 'proj_xy' in data:
                save_dict[f'{prefix}_proj'] = data['proj_xy']
            if 'point_pos' in data:
                save_dict[f'{prefix}_point_pos'] = data['point_pos']
            if 'posed_pos' in data:
                save_dict[f'{prefix}_posed_pos'] = data['posed_pos']
        
        save_path = os.path.join(self.session_dir, filename)
        np.savez_compressed(save_path, **save_dict)
        print(f"[AttnVisualizer] Saved raw attention to: {save_path}")


# Global visualizer instance (optional singleton pattern)
_global_visualizer: Optional[AttnVisualizer] = None


def get_visualizer(
    save_dir: str = 'output/attn_debug_v2',
    enabled: bool = True,
    **kwargs
) -> AttnVisualizer:
    """Get or create global visualizer instance."""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = AttnVisualizer(save_dir=save_dir, enabled=enabled, **kwargs)
    return _global_visualizer


def reset_visualizer():
    """Reset global visualizer."""
    global _global_visualizer
    _global_visualizer = None
