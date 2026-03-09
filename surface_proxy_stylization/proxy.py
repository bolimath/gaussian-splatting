from dataclasses import dataclass
from typing import List
import torch
from .types import SurfaceProxy
from .math_utils import normalize, orthonormal_frame_from_normal


@dataclass
class ProxyBuildConfig:
    num_views: int = 8
    max_points_per_view: int = 6000
    voxel_size: float = 0.02
    knn_k: int = 8
    normal_knn: int = 16


def invdepth_to_world_points(invdepth: torch.Tensor, cam) -> torch.Tensor:
    """Back-project inverse depth map to world points.

    invdepth: (H, W)
    returns world points: (H, W, 3)
    """
    device = invdepth.device
    h, w = invdepth.shape
    ys = torch.linspace(0, h - 1, h, device=device)
    xs = torch.linspace(0, w - 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x_ndc = (xx / (w - 1)) * 2 - 1
    y_ndc = (yy / (h - 1)) * 2 - 1

    full_proj = cam.full_proj_transform.transpose(0, 1)
    inv_proj = torch.inverse(full_proj)
    clip_near = torch.stack([x_ndc, y_ndc, torch.zeros_like(x_ndc), torch.ones_like(x_ndc)], dim=-1)
    clip_far = torch.stack([x_ndc, y_ndc, torch.ones_like(x_ndc), torch.ones_like(x_ndc)], dim=-1)

    near = (inv_proj @ clip_near.reshape(-1, 4).T).T
    far = (inv_proj @ clip_far.reshape(-1, 4).T).T
    near = near[:, :3] / near[:, 3:4].clamp_min(1e-6)
    far = far[:, :3] / far[:, 3:4].clamp_min(1e-6)

    ray = normalize(far - near)
    depth = (1.0 / invdepth.reshape(-1).clamp_min(1e-6)).unsqueeze(-1)
    world = near + ray * depth
    return world.reshape(h, w, 3)


def _voxel_downsample(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    keys = torch.floor(points / voxel_size).to(torch.int64)
    unique_keys, inv = torch.unique(keys, dim=0, return_inverse=True)
    counts = torch.bincount(inv)
    out = torch.zeros((unique_keys.shape[0], 3), device=points.device, dtype=points.dtype)
    out.index_add_(0, inv, points)
    out = out / counts[:, None].to(points.dtype)
    return out


def _estimate_normals(points: torch.Tensor, knn: int) -> torch.Tensor:
    d = torch.cdist(points, points)
    idx = torch.topk(d, k=min(knn + 1, points.shape[0]), largest=False).indices[:, 1:]
    neigh = points[idx] - points[:, None, :]
    cov = torch.einsum("nki,nkj->nij", neigh, neigh) / float(idx.shape[1])
    _, eigvec = torch.linalg.eigh(cov)
    normals = eigvec[:, :, 0]
    normals = normalize(normals)
    return normals


def _build_graph(points: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(points, points)
    idx = torch.topk(d, k=min(k + 1, points.shape[0]), largest=False).indices[:, 1:]
    src = torch.arange(points.shape[0], device=points.device)[:, None].expand_as(idx)
    edges = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=-1)
    edges = torch.sort(edges, dim=-1).values
    edges = torch.unique(edges, dim=0)
    return edges


def build_surface_proxy(depth_maps: List[torch.Tensor], cameras: List, config: ProxyBuildConfig) -> SurfaceProxy:
    all_points = []
    for invd, cam in zip(depth_maps, cameras):
        pts = invdepth_to_world_points(invd, cam).reshape(-1, 3)
        valid = torch.isfinite(pts).all(dim=-1)
        pts = pts[valid]
        if pts.shape[0] > config.max_points_per_view:
            perm = torch.randperm(pts.shape[0], device=pts.device)[: config.max_points_per_view]
            pts = pts[perm]
        all_points.append(pts)

    fused = torch.cat(all_points, dim=0)
    fused = _voxel_downsample(fused, config.voxel_size)
    normals = _estimate_normals(fused, config.normal_knn)
    frames = orthonormal_frame_from_normal(normals)
    edges = _build_graph(fused, config.knn_k)

    center = fused.mean(dim=0)
    anchor_mask = (fused - center).norm(dim=-1) > (fused - center).norm(dim=-1).median()
    return SurfaceProxy(positions=fused, normals=normals, frames=frames, edges=edges, anchor_mask=anchor_mask)
