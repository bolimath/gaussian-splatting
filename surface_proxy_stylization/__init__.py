from .types import SurfaceProxy, GaussianBinding
from .proxy import ProxyBuildConfig, build_surface_proxy
from .binding import build_gaussian_proxy_binding
from .deformation import update_gaussians_from_proxy, apply_updates_to_gaussian_model
from .optimize import GeometryOptConfig, optimize_proxy_translations
from .structural_render import render_structural_fields
from .appearance import refine_appearance_placeholder
