from typing import Optional

import torch
import trimesh
from torch import Tensor


class Mesh:
    def __init__(
        self,
        v: Optional[Tensor] = None,
        f: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (v.device if torch.is_tensor(v) else torch.device("cpu"))
        self.v = v
        self.f = f

    def sample_surface(self, count: int):
        mesh = trimesh.Trimesh(
            vertices=self.v.detach().cpu().numpy(),
            faces=self.f.detach().cpu().numpy(),
            process=False,
        )
        points, _ = trimesh.sample.sample_surface(mesh, count)
        return torch.from_numpy(points).float().to(self.device)
