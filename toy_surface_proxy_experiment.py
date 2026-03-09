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

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./surface_proxy_outputs/toy")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    run(args.out_dir, args.k)
