import torch
from .types import SurfaceProxy


def depth_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def normal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos = (pred * target).sum(dim=0).clamp(-1, 1)
    return (1.0 - cos).mean()


def arap_loss(proxy_deformed: SurfaceProxy, proxy_rest: SurfaceProxy) -> torch.Tensor:
    e = proxy_rest.edges
    p0 = proxy_rest.positions[e[:, 0]]
    p1 = proxy_rest.positions[e[:, 1]]
    q0 = proxy_deformed.positions[e[:, 0]]
    q1 = proxy_deformed.positions[e[:, 1]]
    return ((q0 - q1).norm(dim=-1) - (p0 - p1).norm(dim=-1)).pow(2).mean()


def anchor_loss(proxy_deformed: SurfaceProxy, proxy_rest: SurfaceProxy) -> torch.Tensor:
    if proxy_rest.anchor_mask is None:
        return torch.zeros([], device=proxy_rest.positions.device)
    m = proxy_rest.anchor_mask
    return (proxy_deformed.positions[m] - proxy_rest.positions[m]).pow(2).sum(dim=-1).mean()
