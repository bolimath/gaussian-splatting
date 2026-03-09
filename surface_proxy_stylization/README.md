# Surface Proxy Stylization MVP

This module implements a geometry-first stylization backend prototype for a pretrained 3DGS scene.

## Pipeline stages
- **A: surface proxy construction** from multi-view inverse depth renders.
- **B: gaussian-to-proxy binding** with distance + normal-aware weights.
- **C: structural rendering** (RGB/depth/normal).
- **D: geometry optimization** over per-node proxy translations with depth+normal+ARAP+anchor losses.
- **E: appearance refinement** placeholder interface.

## Run
```bash
python surface_proxy_stylization_mvp.py \
  -m <trained_model_path> \
  -s <source_scene_path> \
  --iteration -1 \
  --out_dir ./surface_proxy_outputs
```

## Main outputs
- `proxy_rest.ply`, `proxy_deformed.ply`
- `proxy_graph.obj`
- `proxy_rest.npz`, `binding.npz`, `gaussian_update.npz`
- `loss_log.json`


## Toy verification experiment
```bash
python toy_surface_proxy_experiment.py --out_dir ./surface_proxy_outputs/toy
```
Outputs:
- `toy_metrics.json`: non-trivial deformation and self-consistency metrics.
- `toy_preview.json`: before/after sample gaussian center values.
- `toy_topdown.svg`: 俯视图，可视化 proxy 形变与 gaussian 位移。
- `toy_profile.svg`: 半径-高度曲线，展示 bump 形变趋势。
