from typing import Dict
import torch
import torch.nn.functional as F
from gaussian_renderer import render
from .math_utils import normalize


def render_structural_fields(cam, gaussians, pipe, bg_color) -> Dict[str, torch.Tensor]:
    pkg = render(cam, gaussians, pipe, bg_color)
    depth = pkg["depth"]
    n = depth_to_normal(depth)
    return {
        "rgb": pkg["render"],
        "depth": depth,
        "normal": n,
    }


def depth_to_normal(depth: torch.Tensor) -> torch.Tensor:
    """Estimate normal from inverse-depth image via image gradients.
    depth: (H, W)
    returns: (3, H, W)
    """
    d = depth[None, None]
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=depth.device, dtype=depth.dtype)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=depth.device, dtype=depth.dtype)
    gx = F.conv2d(d, kx, padding=1)[0, 0]
    gy = F.conv2d(d, ky, padding=1)[0, 0]
    nz = torch.ones_like(gx)
    n = torch.stack([-gx, -gy, nz], dim=0)
    n = normalize(n.permute(1, 2, 0)).permute(2, 0, 1)
    return n
