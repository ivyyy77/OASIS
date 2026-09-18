"""Typed output containers returned by the Gaussian renderer."""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from .base import BaseOutput


@dataclass
class GaussianAppOutput(BaseOutput):
    """Attributes of one Gaussian per query point."""

    offset_xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor
    use_rgb: bool


@dataclass
class GaussianDensifyOutput(BaseOutput):
    """Attributes of optional face-interior Gaussians."""

    activation: Tensor
    shs: Tensor
    bary: Tensor
    opacity: Optional[Tensor] = None
    rotation: Optional[Tensor] = None
    scaling: Optional[Tensor] = None
