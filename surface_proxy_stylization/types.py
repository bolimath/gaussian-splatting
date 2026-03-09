from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class SurfaceProxy:
    """Surface proxy container.

    Shapes:
      positions: (M, 3)
      normals: (M, 3)
      frames: (M, 3, 3)   columns: tangent_x, tangent_y, normal
      edges: (E, 2)
      anchor_mask: (M,) bool
    """
    positions: torch.Tensor
    normals: torch.Tensor
    frames: torch.Tensor
    edges: torch.Tensor
    anchor_mask: Optional[torch.Tensor] = None


@dataclass
class GaussianBinding:
    """Binding from each gaussian to K proxy nodes.

    Shapes:
      indices: (N, K) long
      weights: (N, K)
      local_coords: (N, K, 3)
      rest_positions: (N, 3)
      rest_covariances: (N, 3, 3)
    """
    indices: torch.Tensor
    weights: torch.Tensor
    local_coords: torch.Tensor
    rest_positions: torch.Tensor
    rest_covariances: torch.Tensor
