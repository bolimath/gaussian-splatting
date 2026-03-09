import os
import json
import argparse
import random
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from surface_proxy_stylization import (
    ProxyBuildConfig,
    build_surface_proxy,
    build_gaussian_proxy_binding,
    GeometryOptConfig,
    optimize_proxy_translations,
    update_gaussians_from_proxy,
    apply_updates_to_gaussian_model,
    render_structural_fields,
    refine_appearance_placeholder,
)
from surface_proxy_stylization.io_utils import ensure_dir, write_ply_points, write_obj_edges, save_npz


def synthetic_targets(depth: torch.Tensor, normal: torch.Tensor, strength: float = 0.05):
    h, w = depth.shape
    ys = torch.linspace(0, h - 1, h, device=depth.device)
    xs = torch.linspace(0, w - 1, w, device=depth.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = 0.55 * w, 0.55 * h
    r = min(h, w) * 0.18
    m = (((xx - cx) ** 2 + (yy - cy) ** 2) < r * r).float()
    depth_t = depth + strength * m
    normal_t = normal.clone()
    normal_t[0] = (normal_t[0] + 0.3 * m).clamp(-1, 1)
    normal_t = normal_t / normal_t.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return depth_t, normal_t


def main():
    parser = argparse.ArgumentParser(description="Surface proxy stylization MVP")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, default="./surface_proxy_outputs")
    parser.add_argument("--proxy_views", type=int, default=6)
    parser.add_argument("--proxy_voxel", type=float, default=0.03)
    parser.add_argument("--proxy_knn", type=int, default=8)
    parser.add_argument("--bind_k", type=int, default=4)
    parser.add_argument("--opt_iters", type=int, default=100)
    parser.add_argument("--opt_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = get_combined_args(parser)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_dir(args.out_dir)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    pipe = pipeline.extract(args)
    bg_color = torch.zeros(3, device="cuda")

    cams = scene.getTrainCameras()
    chosen = [cams[i] for i in torch.linspace(0, len(cams) - 1, args.proxy_views).long().tolist()]

    with torch.no_grad():
        depth_maps = [render_structural_fields(c, gaussians, pipe, bg_color)["depth"] for c in chosen]

    proxy_cfg = ProxyBuildConfig(num_views=args.proxy_views, voxel_size=args.proxy_voxel, knn_k=args.proxy_knn)
    proxy_rest = build_surface_proxy(depth_maps, chosen, proxy_cfg)

    write_ply_points(os.path.join(args.out_dir, "proxy_rest.ply"), proxy_rest.positions, proxy_rest.normals)
    write_obj_edges(os.path.join(args.out_dir, "proxy_graph.obj"), proxy_rest.positions, proxy_rest.edges)
    save_npz(
        os.path.join(args.out_dir, "proxy_rest.npz"),
        positions=proxy_rest.positions,
        normals=proxy_rest.normals,
        frames=proxy_rest.frames,
        edges=proxy_rest.edges,
        anchor_mask=proxy_rest.anchor_mask,
    )

    binding = build_gaussian_proxy_binding(gaussians, proxy_rest, k=args.bind_k)
    save_npz(
        os.path.join(args.out_dir, "binding.npz"),
        indices=binding.indices,
        weights=binding.weights,
        local_coords=binding.local_coords,
    )

    supervise_cam = cams[0]
    with torch.no_grad():
        base_fields = render_structural_fields(supervise_cam, gaussians, pipe, bg_color)
        target_depth, target_normal = synthetic_targets(base_fields["depth"], base_fields["normal"])

    def render_and_compare(proxy_deformed):
        mu_new, sc_new, rot_new, _ = update_gaussians_from_proxy(proxy_rest, proxy_deformed, binding)
        apply_updates_to_gaussian_model(gaussians, mu_new, sc_new, rot_new)
        fields = render_structural_fields(supervise_cam, gaussians, pipe, bg_color)
        l_depth = (fields["depth"] - target_depth).abs().mean()
        l_normal = (1.0 - (fields["normal"] * target_normal).sum(dim=0).clamp(-1, 1)).mean()
        return l_depth, l_normal

    opt_cfg = GeometryOptConfig(iters=args.opt_iters, lr=args.opt_lr)
    proxy_final, logs = optimize_proxy_translations(proxy_rest, render_and_compare, opt_cfg)

    mu_new, sc_new, rot_new, cov_new = update_gaussians_from_proxy(proxy_rest, proxy_final, binding)
    apply_updates_to_gaussian_model(gaussians, mu_new, sc_new, rot_new)

    write_ply_points(os.path.join(args.out_dir, "proxy_deformed.ply"), proxy_final.positions, proxy_final.normals)
    save_npz(os.path.join(args.out_dir, "gaussian_update.npz"), mu=mu_new, scaling=sc_new, rotation=rot_new, covariance=cov_new)

    with open(os.path.join(args.out_dir, "loss_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    refine_appearance_placeholder(gaussians=gaussians, scene=scene)
    print(f"Done. Artifacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
