import torch
from utils.general_utils import build_rotation
from .types import SurfaceProxy, GaussianBinding
from .math_utils import normalize


def gaussian_reference_normals(gaussian_model) -> torch.Tensor:
    """Approximate gaussian normal as smallest-variance principal axis."""
    scales = gaussian_model.get_scaling
    rot = build_rotation(gaussian_model.get_rotation)
    min_axis = torch.argmin(scales, dim=-1)
    gather_idx = min_axis[:, None, None].expand(-1, 3, 1)
    normals = torch.gather(rot, 2, gather_idx).squeeze(-1)
    return normalize(normals)


def build_gaussian_proxy_binding(gaussian_model, proxy: SurfaceProxy, k: int = 4, tau_p: float = 0.05, tau_n: float = 0.2) -> GaussianBinding:
    mu = gaussian_model.get_xyz
    g_n = gaussian_reference_normals(gaussian_model)
    p = proxy.positions
    n = proxy.normals

    d = torch.cdist(mu, p)
    knn = torch.topk(d, k=min(k, p.shape[0]), largest=False)
    idx = knn.indices
    d_near = knn.values

    p_near = p[idx]
    n_near = n[idx]
    g_n_rep = g_n[:, None, :].expand_as(n_near)
    n_term = (1.0 - (g_n_rep * n_near).sum(dim=-1).clamp(-1, 1))
    logits = -d_near.pow(2) / tau_p - n_term / tau_n
    w = torch.softmax(logits, dim=-1)

    U = proxy.frames[idx]
    rel = mu[:, None, :] - p_near
    r = torch.einsum("nkij,nkj->nki", U.transpose(-1, -2), rel)

    rot = build_rotation(gaussian_model.get_rotation)
    sc = gaussian_model.get_scaling
    cov = rot @ torch.diag_embed(sc.pow(2)) @ rot.transpose(-1, -2)

    return GaussianBinding(indices=idx, weights=w, local_coords=r, rest_positions=mu.detach().clone(), rest_covariances=cov.detach().clone())
