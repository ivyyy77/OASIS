# -*- coding: utf-8 -*-
"""Utility for rendering posed MANO mesh overlays using pyrender."""

import numpy as np
import torch


def render_mano_mesh_overlay(world_vertex_np, K_np, R_np, T_np, faces, H, W, bg_rgb_np=None):
    """Render posed MANO mesh overlay using pyrender with actual camera intrinsics.

    Args:
        world_vertex_np: (N, 3) numpy array of MANO vertices in world space.
        K_np: (3, 3) camera intrinsic matrix.
        R_np: (3, 3) world-to-camera rotation.
        T_np: (3,) world-to-camera translation.
        faces: (F, 3) face indices (numpy or tensor).
        H, W: image height / width.
        bg_rgb_np: optional (H, W, 3) uint8 RGB background image.

    Returns:
        (H, W, 3) uint8 RGB image with mesh overlay.
    """
    import pyrender as _pyrender
    import trimesh as _trimesh

    # Convert faces to numpy if needed
    if isinstance(faces, torch.Tensor):
        faces_np = faces.cpu().numpy()
    else:
        faces_np = np.asarray(faces)

    # World → camera space (OpenCV convention: +Z towards scene)
    v_cam = (R_np @ world_vertex_np.T).T + T_np.reshape(1, 3)  # (N, 3)

    # Build trimesh in camera space then convert OpenCV → OpenGL (flip Y and Z)
    mesh = _trimesh.Trimesh(v_cam.copy(), faces_np.copy())
    rot = _trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    material = _pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.40, 0.55, 0.85, 1.0),
    )
    mesh_pr = _pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = _pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4))
    scene.add(mesh_pr, 'mesh')

    fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
    cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
    camera = _pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
    scene.add(camera, pose=np.eye(4))

    # Two directional lights for decent shading
    light = _pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))
    lpose = np.eye(4)
    lpose[:3, 3] = [0, -1, 1]
    scene.add(light, pose=lpose)

    renderer = _pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    try:
        color, _ = renderer.render(scene, flags=_pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    color_f = color.astype(np.float32) / 255.0
    valid_mask = color_f[:, :, 3:4]
    if bg_rgb_np is not None:
        bg = np.clip(bg_rgb_np.astype(np.float32) / 255.0, 0, 1)
        out = color_f[:, :, :3] * valid_mask + bg * (1.0 - valid_mask)
    else:
        out = color_f[:, :, :3] * valid_mask + np.ones((H, W, 3), dtype=np.float32) * (1.0 - valid_mask)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
