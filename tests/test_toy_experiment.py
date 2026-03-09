import json
import subprocess
import sys
from pathlib import Path


def test_toy_experiment_recovers_geometry(tmp_path: Path):
    out = tmp_path / "toy"
    cmd = [sys.executable, "toy_surface_proxy_experiment.py", "--out_dir", str(out), "--k", "4"]
    subprocess.run(cmd, check=True)

    metrics = json.loads((out / "toy_metrics.json").read_text())
    assert metrics["deformation_nontrivial_rmse"] > 1e-3
    assert metrics["self_consistency_rmse"] < 1e-9
