import torch
from .types import SurfaceProxy, GaussianBinding
from .math_utils import matrix_to_quaternion


def update_gaussians_from_proxy(proxy_rest: SurfaceProxy, proxy_deformed: SurfaceProxy, binding: GaussianBinding):
    idx = binding.indices
    w = binding.weights
    r = binding.local_coords

    p_new = proxy_deformed.positions[idx]
    U_new = proxy_deformed.frames[idx]

    mu_k = p_new + torch.einsum("nkij,nkj->nki", U_new, r)
    mu_new = (w[..., None] * mu_k).sum(dim=1)

    p_old = proxy_rest.positions[idx]
    U_old = proxy_rest.frames[idx]
    J = U_new @ U_old.transpose(-1, -2)
    J_bar = (w[..., None, None] * J).sum(dim=1)

    cov_new = J_bar @ binding.rest_covariances @ J_bar.transpose(-1, -2)
    eigvals, eigvecs = torch.linalg.eigh(cov_new)
    scales_new = torch.sqrt(eigvals.clamp_min(1e-10))
    rot_new = matrix_to_quaternion(eigvecs)
    return mu_new, scales_new, rot_new, cov_new


def apply_updates_to_gaussian_model(gaussian_model, mu_new: torch.Tensor, scales_new: torch.Tensor, rot_new: torch.Tensor):
    gaussian_model._xyz.data = mu_new
    gaussian_model._scaling.data = torch.log(scales_new.clamp_min(1e-10))
    gaussian_model._rotation.data = rot_new
