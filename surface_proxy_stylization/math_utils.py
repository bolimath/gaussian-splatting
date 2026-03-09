import torch


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


def orthonormal_frame_from_normal(normals: torch.Tensor) -> torch.Tensor:
    n = normalize(normals)
    ref = torch.tensor([1.0, 0.0, 0.0], device=n.device).expand_as(n)
    alt = torch.tensor([0.0, 1.0, 0.0], device=n.device).expand_as(n)
    use_alt = (n[:, 0].abs() > 0.9).unsqueeze(-1)
    t0 = torch.where(use_alt, alt, ref)
    t1 = normalize(torch.cross(n, t0, dim=-1))
    t2 = normalize(torch.cross(n, t1, dim=-1))
    return torch.stack([t1, t2, n], dim=-1)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """Convert (N,3,3) rotation matrices to (N,4) quaternions in (w,x,y,z)."""
    n = m.shape[0]
    q = torch.zeros((n, 4), dtype=m.dtype, device=m.device)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    mask = tr > 0.0
    s = torch.sqrt((tr[mask] + 1.0).clamp_min(1e-8)) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    q[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    q[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s

    mask0 = (~mask) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = torch.sqrt((1.0 + m[mask0, 0, 0] - m[mask0, 1, 1] - m[mask0, 2, 2]).clamp_min(1e-8)) * 2.0
    q[mask0, 0] = (m[mask0, 2, 1] - m[mask0, 1, 2]) / s
    q[mask0, 1] = 0.25 * s
    q[mask0, 2] = (m[mask0, 0, 1] + m[mask0, 1, 0]) / s
    q[mask0, 3] = (m[mask0, 0, 2] + m[mask0, 2, 0]) / s

    mask1 = (~mask) & (~mask0) & (m[:, 1, 1] > m[:, 2, 2])
    s = torch.sqrt((1.0 + m[mask1, 1, 1] - m[mask1, 0, 0] - m[mask1, 2, 2]).clamp_min(1e-8)) * 2.0
    q[mask1, 0] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s
    q[mask1, 1] = (m[mask1, 0, 1] + m[mask1, 1, 0]) / s
    q[mask1, 2] = 0.25 * s
    q[mask1, 3] = (m[mask1, 1, 2] + m[mask1, 2, 1]) / s

    mask2 = (~mask) & (~mask0) & (~mask1)
    s = torch.sqrt((1.0 + m[mask2, 2, 2] - m[mask2, 0, 0] - m[mask2, 1, 1]).clamp_min(1e-8)) * 2.0
    q[mask2, 0] = (m[mask2, 1, 0] - m[mask2, 0, 1]) / s
    q[mask2, 1] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s
    q[mask2, 2] = (m[mask2, 1, 2] + m[mask2, 2, 1]) / s
    q[mask2, 3] = 0.25 * s
    return normalize(q)
