import os
import numpy as np
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_npz(path: str, **arrays):
    payload = {}
    for k, v in arrays.items():
        if isinstance(v, torch.Tensor):
            payload[k] = v.detach().cpu().numpy()
        else:
            payload[k] = v
    np.savez(path, **payload)


def write_ply_points(path: str, points: torch.Tensor, normals: torch.Tensor = None):
    p = points.detach().cpu().numpy()
    n = normals.detach().cpu().numpy() if normals is not None else None
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {p.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if n is not None:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        if n is None:
            for x, y, z in p:
                f.write(f"{x} {y} {z}\n")
        else:
            for (x, y, z), (nx, ny, nz) in zip(p, n):
                f.write(f"{x} {y} {z} {nx} {ny} {nz}\n")


def write_obj_edges(path: str, points: torch.Tensor, edges: torch.Tensor):
    p = points.detach().cpu().numpy()
    e = edges.detach().cpu().numpy()
    with open(path, "w", encoding="utf-8") as f:
        for x, y, z in p:
            f.write(f"v {x} {y} {z}\n")
        for a, b in e:
            f.write(f"l {a + 1} {b + 1}\n")
