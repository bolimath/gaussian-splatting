from __future__ import annotations

import argparse
import json
import math
import os
import random


def v_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def v_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def v_scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_norm(a):
    return math.sqrt(max(v_dot(a, a), 1e-12))


def normalize(a):
    n = v_norm(a)
    return [a[0] / n, a[1] / n, a[2] / n]


def make_plane_proxy(nx=5, ny=5, spacing=0.2, n_gauss=80):
    xs = [(i - (nx - 1) / 2.0) * spacing for i in range(nx)]
    ys = [(j - (ny - 1) / 2.0) * spacing for j in range(ny)]

    proxy_pos = []
    for y in ys:
        for x in xs:
            proxy_pos.append([x, y, 0.0])

    proxy_nrm = [[0.0, 0.0, 1.0] for _ in proxy_pos]
    # tangent_x, tangent_y, normal
    frame = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for _ in proxy_pos
    ]

    random.seed(0)
    gauss_mu, gauss_nrm = [], []
    for _ in range(n_gauss):
        x = random.uniform(min(xs), max(xs))
        y = random.uniform(min(ys), max(ys))
        z = random.gauss(0.03, 0.01)
        gauss_mu.append([x, y, z])
        g = normalize([random.gauss(0, 0.05), random.gauss(0, 0.05), 1.0])
        gauss_nrm.append(g)

    return proxy_pos, proxy_nrm, frame, gauss_mu, gauss_nrm


def local_coords(U, rel):
    # U columns are frame axes. r = U^T rel
    return [v_dot(U[0], rel), v_dot(U[1], rel), v_dot(U[2], rel)]


def frame_apply(U, r):
    return [
        U[0][0] * r[0] + U[1][0] * r[1] + U[2][0] * r[2],
        U[0][1] * r[0] + U[1][1] * r[1] + U[2][1] * r[2],
        U[0][2] * r[0] + U[1][2] * r[1] + U[2][2] * r[2],
    ]


def knn_binding(gauss_mu, gauss_nrm, proxy_pos, proxy_nrm, proxy_frame, k=4, tau_p=0.08, tau_n=0.3):
    n = len(gauss_mu)
    bind_idx, bind_w, bind_r = [], [], []
    for i in range(n):
        pairs = []
        for j, p in enumerate(proxy_pos):
            d = v_norm(v_sub(gauss_mu[i], p))
            pairs.append((d, j))
        pairs.sort(key=lambda x: x[0])
        top = [j for _, j in pairs[:k]]

        logits = []
        for j in top:
            d = v_norm(v_sub(gauss_mu[i], proxy_pos[j]))
            nterm = 1.0 - max(-1.0, min(1.0, v_dot(gauss_nrm[i], proxy_nrm[j])))
            logits.append(-(d * d) / tau_p - nterm / tau_n)
        m = max(logits)
        ex = [math.exp(x - m) for x in logits]
        s = sum(ex)
        w = [x / max(s, 1e-12) for x in ex]

        r = []
        for j in top:
            rel = v_sub(gauss_mu[i], proxy_pos[j])
            r.append(local_coords(proxy_frame[j], rel))

        bind_idx.append(top)
        bind_w.append(w)
        bind_r.append(r)
    return bind_idx, bind_w, bind_r


def apply_mapping(proxy_pos_new, proxy_frame_new, bind_idx, bind_w, bind_r):
    out = []
    for i in range(len(bind_idx)):
        acc = [0.0, 0.0, 0.0]
        for kk, j in enumerate(bind_idx[i]):
            term = v_add(proxy_pos_new[j], frame_apply(proxy_frame_new[j], bind_r[i][kk]))
            acc = v_add(acc, v_scale(term, bind_w[i][kk]))
        out.append(acc)
    return out


def radial_bump(proxy_pos, amp=0.06, sigma=0.35):
    out = []
    for p in proxy_pos:
        rr = math.sqrt(p[0] * p[0] + p[1] * p[1])
        dz = amp * math.exp(-(rr * rr) / (2.0 * sigma * sigma))
        out.append([p[0], p[1], p[2] + dz])
    return out


def rmse(a, b):
    s = 0.0
    c = 0
    for va, vb in zip(a, b):
        d = v_sub(va, vb)
        s += v_dot(d, d)
        c += 3
    return math.sqrt(s / max(c, 1))


def _color(v, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    # blue -> red
    r = int(255 * t)
    g = int(64 * (1.0 - abs(t - 0.5) * 2.0))
    b = int(255 * (1.0 - t))
    return f"rgb({r},{g},{b})"


def save_svg_topdown(path, proxy_rest, proxy_def, gauss_rest, gauss_def, width=900, height=900):
    all_pts = proxy_rest + proxy_def + gauss_rest + gauss_def
    minx = min(p[0] for p in all_pts)
    maxx = max(p[0] for p in all_pts)
    miny = min(p[1] for p in all_pts)
    maxy = max(p[1] for p in all_pts)

    pad = 40.0
    sx = (width - 2 * pad) / max(maxx - minx, 1e-6)
    sy = (height - 2 * pad) / max(maxy - miny, 1e-6)
    s = min(sx, sy)

    def to_px(p):
        x = pad + (p[0] - minx) * s
        y = height - (pad + (p[1] - miny) * s)
        return x, y

    zvals = [p[2] for p in proxy_def]
    zmin, zmax = min(zvals), max(zvals)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        f.write('<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n')
        f.write('<text x="20" y="30" font-size="20" font-family="monospace">Toy Surface Proxy: top-down view</text>\n')

        # proxy rest: light gray
        for p in proxy_rest:
            x, y = to_px(p)
            f.write(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="rgb(210,210,210)"/>\n')

        # proxy deformed: colored by z
        for p in proxy_def:
            x, y = to_px(p)
            c = _color(p[2], zmin, zmax)
            f.write(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{c}" stroke="black" stroke-width="0.4"/>\n')

        # gaussian displacement arrows (subsample for readability)
        step = max(1, len(gauss_rest) // 40)
        for i in range(0, len(gauss_rest), step):
            x0, y0 = to_px(gauss_rest[i])
            x1, y1 = to_px(gauss_def[i])
            f.write(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="rgb(20,90,200)" stroke-width="1.1"/>\n')
            f.write(f'<circle cx="{x1:.2f}" cy="{y1:.2f}" r="2.2" fill="rgb(20,90,200)"/>\n')

        # legend
        f.write('<rect x="20" y="50" width="320" height="100" fill="white" stroke="black"/>\n')
        f.write('<circle cx="40" cy="75" r="4" fill="rgb(210,210,210)"/><text x="55" y="80" font-size="14">proxy rest</text>\n')
        f.write('<circle cx="40" cy="102" r="5" fill="rgb(220,20,60)" stroke="black" stroke-width="0.4"/><text x="55" y="107" font-size="14">proxy deformed (z-colored)</text>\n')
        f.write('<line x1="33" y1="129" x2="47" y2="129" stroke="rgb(20,90,200)" stroke-width="1.2"/><text x="55" y="134" font-size="14">gaussian displacement</text>\n')
        f.write('</svg>\n')


def save_svg_z_curve(path, proxy_rest, proxy_def, width=1000, height=360):
    # sort by radial distance to show bump profile
    rows = []
    for pr, pd in zip(proxy_rest, proxy_def):
        rr = math.sqrt(pr[0] * pr[0] + pr[1] * pr[1])
        rows.append((rr, pr[2], pd[2]))
    rows.sort(key=lambda x: x[0])

    rr_min, rr_max = rows[0][0], rows[-1][0]
    z_min = min(min(r[1], r[2]) for r in rows)
    z_max = max(max(r[1], r[2]) for r in rows)

    pad = 35.0

    def map_x(rr):
        return pad + (rr - rr_min) / max(rr_max - rr_min, 1e-6) * (width - 2 * pad)

    def map_y(zz):
        return height - (pad + (zz - z_min) / max(z_max - z_min, 1e-6) * (height - 2 * pad))

    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        f.write('<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n')
        f.write('<text x="20" y="25" font-size="18" font-family="monospace">Proxy z profile vs radius</text>\n')
        f.write(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>\n')
        f.write(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>\n')

        rest_path = []
        def_path = []
        for rr, z0, z1 in rows:
            rest_path.append(f"{map_x(rr):.2f},{map_y(z0):.2f}")
            def_path.append(f"{map_x(rr):.2f},{map_y(z1):.2f}")
        f.write(f'<polyline fill="none" stroke="rgb(140,140,140)" stroke-width="2" points="{" ".join(rest_path)}"/>\n')
        f.write(f'<polyline fill="none" stroke="rgb(220,20,60)" stroke-width="2" points="{" ".join(def_path)}"/>\n')
        f.write('<text x="45" y="45" font-size="14" fill="rgb(140,140,140)">rest z</text>\n')
        f.write('<text x="45" y="65" font-size="14" fill="rgb(220,20,60)">deformed z</text>\n')
        f.write('</svg>\n')


def run(out_dir: str, k: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    proxy_pos, proxy_nrm, proxy_frame, gauss_mu, gauss_nrm = make_plane_proxy()
    bind_idx, bind_w, bind_r = knn_binding(gauss_mu, gauss_nrm, proxy_pos, proxy_nrm, proxy_frame, k=k)

    proxy_gt = radial_bump(proxy_pos)
    gauss_gt = apply_mapping(proxy_gt, proxy_frame, bind_idx, bind_w, bind_r)

    # Verification: forward mapping should differ from rest (non-trivial deformation)
    gauss_rest = apply_mapping(proxy_pos, proxy_frame, bind_idx, bind_w, bind_r)

    metrics = {
        "num_proxy_nodes": len(proxy_pos),
        "num_gaussians": len(gauss_mu),
        "bind_k": k,
        "deformation_nontrivial_rmse": rmse(gauss_rest, gauss_gt),
        "self_consistency_rmse": rmse(gauss_gt, apply_mapping(proxy_gt, proxy_frame, bind_idx, bind_w, bind_r)),
    }

    with open(os.path.join(out_dir, "toy_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "toy_preview.json"), "w", encoding="utf-8") as f:
        json.dump({"first_gaussian_before": gauss_rest[0], "first_gaussian_after": gauss_gt[0]}, f, indent=2)

    save_svg_topdown(
        os.path.join(out_dir, "toy_topdown.svg"),
        proxy_pos,
        proxy_gt,
        gauss_rest,
        gauss_gt,
    )
    save_svg_z_curve(
        os.path.join(out_dir, "toy_profile.svg"),
        proxy_pos,
        proxy_gt,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./surface_proxy_outputs/toy")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    run(args.out_dir, args.k)
