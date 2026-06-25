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

`cable.segments = N` means the estimate has `N + 1` connected control points.
All segments share one fixed length. Set `cable.length_m` to the measured
physical cable length; the filter uses `length_m * length_scale / segments` as
the fixed segment length. Keep `length_scale` slightly below `1.0` when noisy
endpoints make the chain spend excess length as zigzag. Leave `length_m = 0` to
estimate the segment length once from the first measurement.

Current research pipeline:

1. Segment the cable in RGB using the PIDNet-S binary cable segmenter.
2. Clean the mask while preserving multiple visible fragments by default,
   because there is only one physical cable and gaps usually mean occlusion.
3. Use the PIDNet mask to select ZED point-cloud samples that belong to the
   cable.
4. Treat the masked 3D points as the measurement support set for the particle
   filter. The support cap is `particle_filter.measurement_points`.
5. Each particle is one connected fixed-length 3D segment chain.
6. On initialization/reacquisition, estimate a temporary cable path from the
   masked 3D cloud only to seed particles.
7. On normal tracking frames, build temporary ordered cloud fits only to propose
   particles and detect the two cable endpoints; the final estimate still comes
   from the particle filter.
8. Use the detected endpoint pair as a soft anchor. Particles are nudged partway
   toward those endpoints before scoring, and endpoint error is added to the
   likelihood. This helps when the middle is partially occluded, but it is not a
   hard kinematic constraint.
9. Assign visible support points to the closest segment of the current/reference
   cable. When the mask is partially occluded, hidden segments receive no
   negative point-cloud evidence.
10. Score visible support points against only their assigned segment neighborhood
   in each particle:

   ```text
   E(x) = mean_j min_{i in assigned_window(j)} distance(point_j, segment_i(x))^2
   ```

   CUDA tensor scoring is used automatically when available.
11. Use conservative per-node velocity prediction to carry hidden cable portions
    through short occlusions. Velocities are updated from visible nodes, ignored
    below a deadband, and decay while hidden.
12. Use a mixed proposal distribution: normal proposals near the temporary
    masked-cloud fit, broad proposals for recovery, and conservative proposals
    near the current displayed chain.
13. Output the weighted average of top particles near the MAP particle after
     simple point-to-cable scoring. This avoids averaging separate shape modes
     and reduces frame-to-frame jitter from switching MAP particles.
14. Draw the displayed chain plus the top weighted particles in the 3D viewer.
15. Track the fixed-length connected segment chain with the particle filter.

Label masks and train the PIDNet-S detector on CUDA:

```bash
../.venv/bin/python tools/pidnet_training_gui.py --device cuda
```

The GUI workflow is:

1. Open/capture RGB frames.
2. Paint only the cable pixels in the binary mask. Every unpainted pixel is
   background.
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

By default, live mode reads `config.toml` and expects
`models/pidnet_cable_best.pt`. Train that checkpoint first, or edit
`pidnet.checkpoint` in the config. To use another config:

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
and the trainer will create a validation split by filename stem. Masks are
binary cable masks where nonzero pixels are cable and zero pixels are
background. You do not paint background explicitly. The PIDNet path feeds the
downstream tracker as binary mask, masked ZED cable points, then particle-filter
segment scoring.

The live UI shows RGB on the left and the ZED point cloud on the right. The
cable model is controlled by `cable.segments`; `N` segments means `N + 1`
connected 3D nodes. The displayed cable is a clustered weighted average of the
top particles after simple point-to-cable scoring, not a pre-fitted measurement
curve.
Important tuning sections:

- `pidnet`: model checkpoint, CUDA device, probability threshold.
- `detector`: resized PIDNet inference, ROI reacquisition, mask cleanup.
- `measurement`: masked ZED point selection, depth rejection, prediction gating.
- `cable`: number of segments and physical cable length.
- `particle_filter`: particle count, process noise, proposal injection, scoring.
- `point_cloud`: viewer point-cloud sampling and confidence-map cadence.
