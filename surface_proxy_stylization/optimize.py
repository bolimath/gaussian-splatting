from dataclasses import dataclass
from typing import List, Dict
import torch
from .types import SurfaceProxy
from .losses import depth_loss, normal_loss, arap_loss, anchor_loss


@dataclass
class GeometryOptConfig:
    iters: int = 200
    lr: float = 1e-3
    w_depth: float = 1.0
    w_normal: float = 0.5
    w_arap: float = 1.0
    w_anchor: float = 0.2


def optimize_proxy_translations(
    proxy_rest: SurfaceProxy,
    render_and_compare_fn,
    config: GeometryOptConfig,
) -> (SurfaceProxy, List[Dict[str, float]]):
    delta = torch.zeros_like(proxy_rest.positions, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=config.lr)
    logs: List[Dict[str, float]] = []

    for it in range(config.iters):
        opt.zero_grad()
        proxy_def = SurfaceProxy(
            positions=proxy_rest.positions + delta,
            normals=proxy_rest.normals,
            frames=proxy_rest.frames,
            edges=proxy_rest.edges,
            anchor_mask=proxy_rest.anchor_mask,
        )

        l_depth, l_normal = render_and_compare_fn(proxy_def)
        l_arap = arap_loss(proxy_def, proxy_rest)
        l_anchor = anchor_loss(proxy_def, proxy_rest)
        loss = (
            config.w_depth * l_depth
            + config.w_normal * l_normal
            + config.w_arap * l_arap
            + config.w_anchor * l_anchor
        )
        loss.backward()
        opt.step()

        logs.append(
            {
                "iter": it,
                "loss": float(loss.item()),
                "depth": float(l_depth.item()),
                "normal": float(l_normal.item()),
                "arap": float(l_arap.item()),
                "anchor": float(l_anchor.item()),
            }
        )

    proxy_final = SurfaceProxy(
        positions=proxy_rest.positions + delta.detach(),
        normals=proxy_rest.normals,
        frames=proxy_rest.frames,
        edges=proxy_rest.edges,
        anchor_mask=proxy_rest.anchor_mask,
    )
    return proxy_final, logs
