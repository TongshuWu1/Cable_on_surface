# Particle Filter Cable Project

Particle-filter 3D cable reconstruction and tracking from a PIDNet cable mask
and ZED point cloud.

Run the live tracker with no arguments:

```bash
../.venv/bin/python main.py
```

Runtime tuning lives in `config.toml`. Command-line flags are still available
for temporary overrides, but normal tuning should go into that file so each run
starts from the same known settings.

The cable model intentionally ignores the physical outline of the cable. The
particle filter represents it as connected fixed-length 3D segments:

```text
P0 -- P1 -- P2 -- ... -- PN
```

`cable.segments = N` means each estimate has `N + 1` connected control points.
Each cable has its own fixed segment length derived from
`cable.lengths_m[cable_index] / cable.segments`.

Current pipeline:

1. Segment RGB with the PIDNet-S checkpoint trained for one generic cable mask
   plus one endpoint channel per cable.
2. For each cable, use its endpoint channel to build fixed endpoint anchors.
3. Lift the shared cable mask into ZED 3D support points.
4. Each cable has its own particle filter, fixed physical length, endpoint tape
   length, and endpoint-constrained particle set.
5. Before scoring, each PF keeps only support points physically reachable from
   that cable's endpoints.
6. Use endpoint-constrained RANSAC to select the inlier support subset for each
   cable.
7. Score particles by point-to-polyline distance using CUDA tensor scoring when
   available.
8. Track through occlusion with velocity prediction, visible-segment assignment,
   and resampling.

Label masks and train the PIDNet-S detector on CUDA:

```bash
../.venv/bin/python tools/pidnet_training_gui.py --device cuda
```

The GUI workflow is:

1. Open/capture RGB frames.
2. Paint the generic cable body and the endpoint class for each cable. Every
   unpainted pixel is background.
3. Use `Label Open` / `Label Close` only when cleaning a hand-painted training
   mask.
4. Save labels to `datasets/cable_pidnet/images/...` and `datasets/cable_pidnet/masks/...`.
5. Use `Live PIDNet cleanup` to tune the same threshold, open kernel, close
   kernel, and min-area cleanup used by `main.py`, then save those values to
   `config.toml`.
6. Use `Check Dataset`, tune the PIDNet training values, and save/load those
   values with `Save Params` / `Load Params`.
7. Click `Train PIDNet-S on CUDA`, then test the checkpoint on current, train,
   and validation frames.
8. Run live tracking with the saved checkpoint.

Live ZED split view:

```bash
../.venv/bin/python main.py
```

By default, live mode reads `config.toml` and expects the checkpoint named by
`pidnet.checkpoint`. The checkpoint must output one generic cable channel plus
one endpoint channel per cable. To use another config:

```bash
../.venv/bin/python main.py --config /path/to/config.toml
```

Headless PIDNet-S training:

```bash
../.venv/bin/python tools/train_pidnet_cable.py --dataset /path/to/cable_dataset --output models/pidnet_cable_best.pt --epochs 80 --batch-size 8 --imgsz 1280x720 --device cuda
```

Dataset layout:

```text
cable_dataset/
  images/train/frame_0001.png
  masks/train/frame_0001.png
  images/val/frame_0101.png
  masks/val/frame_0101.png
```

The `val` split is optional. A flat `images/` and `masks/` layout also works,
and the trainer will create a validation split by filename stem. Masks use one
generic cable-body class plus one endpoint class per cable. You do not paint
background explicitly. The PIDNet path feeds the downstream tracker as endpoint
anchors, masked ZED cable points, then particle-filter segment scoring.

The live UI shows RGB on the left and the ZED point cloud on the right. The
cable model is controlled by `cable.segments`; `N` segments means `N + 1`
connected 3D nodes. Important tuning sections:

- `pidnet`: model checkpoint, CUDA device, probability threshold.
- `detector`: PIDNet inference scale, mask cleanup, and update cadence.
- `measurement`: masked ZED point selection, depth rejection, prediction gating.
- `particle_filter`: particle count, process noise, scoring, occlusion.
- `point_cloud`: viewer point-cloud sampling and confidence-map cadence.
