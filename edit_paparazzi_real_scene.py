import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Inria 3DGS imports
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state


def to_uint8(img_3hw):
    x = img_3hw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_img(img_3hw, path):
    Image.fromarray(to_uint8(img_3hw)).save(path)


def make_disk_flow(H, W, cx, cy, r, dx, dy, device):
    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2).float()
    flow = torch.zeros(H, W, 2, device=device)
    flow[..., 0] = dx * mask
    flow[..., 1] = dy * mask
    return flow, mask


def warp_image(img_3hw, flow_hw2):
    """
    Backward warp: I'(p) = I(p - flow(p))
    img: (3,H,W)
    flow: (H,W,2) in pixels
    """
    device = img_3hw.device
    _, H, W = img_3hw.shape
    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    src_x = (xx - flow_hw2[..., 0]).clamp(0, W - 1)
    src_y = (yy - flow_hw2[..., 1]).clamp(0, H - 1)

    nx = (src_x / (W - 1)) * 2 - 1
    ny = (src_y / (H - 1)) * 2 - 1
    grid = torch.stack([nx, ny], dim=-1)[None, ...]  # (1,H,W,2)

    out = F.grid_sample(
        img_3hw[None, ...],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out[0]


def psnr(a, b, eps=1e-8):
    mse = (a - b).pow(2).mean().clamp_min(eps)
    return float(10.0 * torch.log10(1.0 / mse))


def get_cam_mats(cam):
    """
    Extract camera matrices from Inria Camera object.
    """
    W2V = cam.world_view_transform.transpose(0, 1)
    P = cam.full_proj_transform.transpose(0, 1)
    C = cam.camera_center
    return W2V, P, C


def ndc_to_pix(ndc_xy, H, W):
    x = (ndc_xy[..., 0] * 0.5 + 0.5) * (W - 1)
    y = (ndc_xy[..., 1] * 0.5 + 0.5) * (H - 1)
    return torch.stack([x, y], dim=-1)


def pix_to_ndc(pix_xy, H, W):
    x = (pix_xy[..., 0] / (W - 1)) * 2 - 1
    y = (pix_xy[..., 1] / (H - 1)) * 2 - 1
    return torch.stack([x, y], dim=-1)


@torch.no_grad()
def forward_splat_ref_to_tgt(I_ref_3hw, depth_ref_hw, cam_ref, cam_tgt, H, W, device):
    """
    Forward-splat ref image into target view using ref depth (single-layer z-buffer).
    Produces a synthesized target image for tgt view consistent with geometry.
    """
    # Camera matrices
    _, P_ref, _ = get_cam_mats(cam_ref)
    _, P_tgt, _ = get_cam_mats(cam_tgt)

    # Build pixel grid in ref
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([xx, yy], dim=-1).float()  # (H,W,2)

    ndc = pix_to_ndc(pix, H, W)  # (H,W,2)

    # Inverse transforms
    P_ref_inv = torch.inverse(P_ref)

    # Construct clip coords at z=0 and z=1 to build a world-space ray
    ones = torch.ones(H, W, 1, device=device)
    clip_near = torch.cat([ndc, torch.zeros(H, W, 1, device=device), ones], dim=-1)
    clip_far = torch.cat([ndc, torch.ones(H, W, 1, device=device), ones], dim=-1)

    def clip_to_world(clip):
        x = clip.view(-1, 4).T
        w = (P_ref_inv @ x).T
        w = w[:, :3] / w[:, 3:4].clamp_min(1e-8)
        return w.view(H, W, 3)

    w_near = clip_to_world(clip_near)
    w_far = clip_to_world(clip_far)
    ray_dir = w_far - w_near
    ray_dir = ray_dir / (torch.norm(ray_dir, dim=-1, keepdim=True) + 1e-8)

    # Approximate world point from ray and rendered depth
    world_pts = w_near + ray_dir * depth_ref_hw[..., None]

    # Project into target clip
    X = torch.cat([world_pts, torch.ones(H, W, 1, device=device)], dim=-1)
    Xf = X.view(-1, 4).T
    clip_t = (P_tgt @ Xf).T
    ndc_t = clip_t[:, :3] / clip_t[:, 3:4].clamp_min(1e-8)
    ndc_xy = ndc_t[:, :2].view(H, W, 2)
    ndc_z = ndc_t[:, 2].view(H, W)

    pix_t = ndc_to_pix(ndc_xy, H, W)

    # Forward splat with z-buffer
    tgt = torch.zeros(3, H, W, device=device)
    zbuf = torch.full((H, W), 1e9, device=device)
    valid = torch.zeros(H, W, device=device)

    px = pix_t[..., 0].round().long().clamp(0, W - 1).view(-1)
    py = pix_t[..., 1].round().long().clamp(0, H - 1).view(-1)
    z = ndc_z.view(-1)
    col = I_ref_3hw.permute(1, 2, 0).reshape(-1, 3)

    inside = (
        (pix_t[..., 0] >= 0)
        & (pix_t[..., 0] <= W - 1)
        & (pix_t[..., 1] >= 0)
        & (pix_t[..., 1] <= H - 1)
    ).view(-1)

    px = px[inside]
    py = py[inside]
    z = z[inside]
    col = col[inside]

    for i in range(px.numel()):
        x = px[i].item()
        y = py[i].item()
        if z[i] < zbuf[y, x]:
            zbuf[y, x] = z[i]
            tgt[:, y, x] = col[i]
            valid[y, x] = 1.0

    return tgt, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="trained 3DGS model folder")
    parser.add_argument("--out_dir", default="./paparazzi_real_outputs", type=str)
    parser.add_argument("--ref_view", default=0, type=int, help="reference view index")
    parser.add_argument("--views", default="0,1,2", type=str, help="comma list of view indices to supervise")
    parser.add_argument("--iters", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--down", default=2, type=int, help="downsample factor for speed (>=1)")
    parser.add_argument("--dx", default=40.0, type=float)
    parser.add_argument("--dy", default=-25.0, type=float)
    parser.add_argument("--radius", default=120.0, type=float)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    safe_state(True)

    # Load scene + gaussians using Inria pipeline
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    dummy = get_combined_args(parser)
    dummy.model_path = args.model_path

    scene = Scene(model.extract(dummy), shuffle=False)
    gaussians = scene.gaussians

    # Choose views
    view_ids = [int(x) for x in args.views.split(",")]
    cams = scene.getTrainCameras()

    def render_view(cam):
        pkg = render(cam, gaussians, pipeline.extract(dummy), bg_color=torch.zeros(3, device=device))
        img = pkg["render"]
        depth = pkg.get("depth", None)
        alpha = pkg.get("alpha", None)
        return img, depth, alpha, pkg

    # Render original for reference view
    with torch.no_grad():
        img_ref, depth_ref, _, _ = render_view(cams[args.ref_view])
        if args.down > 1:
            img_ref_d = F.interpolate(
                img_ref[None],
                scale_factor=1 / args.down,
                mode="bilinear",
                align_corners=False,
            )[0]
            if depth_ref is not None:
                depth_ref_d = F.interpolate(
                    depth_ref[None, None],
                    scale_factor=1 / args.down,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            else:
                raise RuntimeError(
                    "Renderer did not return depth. Please use a 3DGS repo version that outputs "
                    "depth, or add depth output in gaussian_renderer."
                )
        else:
            img_ref_d = img_ref
            depth_ref_d = depth_ref

    _, H, W = img_ref_d.shape

    # Known 2D edit on ref: disk translation
    cx, cy = W * 0.55, H * 0.55
    flow_ref, _ = make_disk_flow(
        H,
        W,
        cx,
        cy,
        args.radius / args.down,
        args.dx / args.down,
        args.dy / args.down,
        device,
    )
    img_ref_edit = warp_image(img_ref_d, flow_ref)

    save_img(img_ref_d, os.path.join(args.out_dir, "ref_original.png"))
    save_img(img_ref_edit, os.path.join(args.out_dir, "ref_edited2d.png"))

    # Build targets for supervised views via forward-splatting
    targets = {}
    valids = {}
    for vid in view_ids:
        with torch.no_grad():
            img_v, _, _, _ = render_view(cams[vid])
            if args.down > 1:
                img_v = F.interpolate(
                    img_v[None],
                    scale_factor=1 / args.down,
                    mode="bilinear",
                    align_corners=False,
                )[0]
            save_img(img_v, os.path.join(args.out_dir, f"v{vid}_original.png"))

            tgt_v, valid_v = forward_splat_ref_to_tgt(
                img_ref_edit,
                depth_ref_d,
                cams[args.ref_view],
                cams[vid],
                H,
                W,
                device,
            )
            targets[vid] = tgt_v
            valids[vid] = valid_v

            save_img(tgt_v, os.path.join(args.out_dir, f"v{vid}_target.png"))
            Image.fromarray((valid_v.detach().cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"v{vid}_valid.png")
            )

    # Optimization vars: gate + delta_mu
    xyz = gaussians.get_xyz
    N = xyz.shape[0]
    delta = torch.zeros_like(xyz, requires_grad=True)
    eta = torch.zeros(N, device=device, requires_grad=True)

    with torch.no_grad():
        base_xyz = xyz.detach()
        if base_xyz.shape[0] > 80000:
            idx = torch.randperm(base_xyz.shape[0], device=device)[:80000]
            base_xyz_knn = base_xyz[idx]
        else:
            idx = None
            base_xyz_knn = base_xyz

        dist = torch.cdist(base_xyz_knn, base_xyz_knn)
        knn = dist.topk(8, largest=False).indices[:, 1:]

    def smooth_l2(d):
        nbr = d[knn]
        return ((d[:, None, :] - nbr) ** 2).mean()

    optim = torch.optim.Adam([delta, eta], lr=args.lr)

    xyz0 = xyz.detach().clone()

    for it in range(args.iters):
        optim.zero_grad()
        gate = torch.sigmoid(eta)

        with torch.no_grad():
            xyz.copy_(xyz0 + gate[:, None] * delta)

        loss_img = 0.0

        for vid in view_ids:
            pkg = render(cams[vid], gaussians, pipeline.extract(dummy), bg_color=torch.zeros(3, device=device))
            img = pkg["render"]
            if args.down > 1:
                img = F.interpolate(img[None], size=(H, W), mode="bilinear", align_corners=False)[0]

            tgt = targets[vid]
            valid = valids[vid]
            diff = (img - tgt).abs().mean(dim=0)
            loss_img = loss_img + (diff * valid).sum() / (valid.sum().clamp_min(1.0))

        gate_sparse = gate.mean()
        if idx is not None:
            d_sub = gate[idx][:, None] * delta[idx]
            delta_smooth = smooth_l2(d_sub)
        else:
            delta_smooth = smooth_l2(gate[:, None] * delta)

        mag = (gate[:, None] * delta).pow(2).mean()

        loss = loss_img + 0.05 * gate_sparse + 0.25 * delta_smooth + 0.02 * mag
        loss.backward()
        optim.step()

        if it % 20 == 0 or it == args.iters - 1:
            print(
                f"[{it:04d}] loss={loss.item():.4f} | img={loss_img.item():.4f} | "
                f"gate_mean={gate.mean().item():.4f}"
            )

    with torch.no_grad():
        gate = torch.sigmoid(eta)
        xyz.copy_(xyz0 + gate[:, None] * delta)

    for vid in view_ids:
        pkg = render(cams[vid], gaussians, pipeline.extract(dummy), bg_color=torch.zeros(3, device=device))
        img = pkg["render"]
        if args.down > 1:
            img = F.interpolate(img[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        save_img(img, os.path.join(args.out_dir, f"v{vid}_edited.png"))

        l1 = float((img - targets[vid]).abs().mean())
        p = psnr(img, targets[vid])
        print(f"View {vid}: L1={l1:.4f} PSNR={p:.2f}")

    print("\nDone. Outputs in:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
