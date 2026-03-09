import torch
from surface_proxy_stylization.types import SurfaceProxy, GaussianBinding
from surface_proxy_stylization.deformation import update_gaussians_from_proxy


def test_identity_deformation_keeps_mean():
    p = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    n = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    f = torch.eye(3).repeat(2, 1, 1)
    e = torch.tensor([[0, 1]])
    proxy = SurfaceProxy(positions=p, normals=n, frames=f, edges=e)

    indices = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.25, 0.75]])
    local = torch.zeros(1, 2, 3)
    rest_pos = torch.tensor([[0.75, 0.0, 0.0]])
    rest_cov = torch.eye(3).unsqueeze(0)
    binding = GaussianBinding(indices, weights, local, rest_pos, rest_cov)

    mu, sc, rot, cov = update_gaussians_from_proxy(proxy, proxy, binding)
    assert torch.allclose(mu, rest_pos, atol=1e-6)
    assert mu.shape == (1, 3)
    assert sc.shape == (1, 3)
    assert rot.shape == (1, 4)
    assert cov.shape == (1, 3, 3)
