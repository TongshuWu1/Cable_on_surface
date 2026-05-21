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
All segments share one fixed length. Use `cable.segment_length_m = 0` to
estimate that length once from the first measurement, or set a positive value in
meters.

Current pipeline:

1. Segment the cable in RGB using the PIDNet-S binary cable segmenter.
2. Clean the mask while preserving multiple visible fragments by default,
   because there is only one physical cable and gaps usually mean occlusion.
3. Use the PIDNet mask to select ZED point-cloud samples that belong to the
   cable.
4. On initialization/reacquisition, estimate cable start/end from the masked 3D
   cloud with a k-nearest-neighbor graph, then resample the endpoint-to-endpoint
   path into ordered cable nodes.
5. On normal tracking frames, order the masked 3D points by projection onto the
   current filtered chain. This is cheaper than rebuilding the graph every frame
   and keeps start/end identity stable.
6. Treat the same masked 3D points as the measurement support set for the particle
   filter. The support cap is `particle_filter.measurement_points`.
7. Each particle is one connected fixed-length 3D segment chain.
8. Score each particle by the distance from every masked cable point to the
   closest segment in that particle. CUDA tensor scoring is used automatically
   when available.
9. Add an endpoint penalty so the particle start/end stay close to the measured
   cable start/end instead of flipping segment order.
10. Keep the best `particle_filter.score_keep_fraction` point distances, so
   sparse bad depth points do not dominate the particle weight.
11. During occlusion, assign visible points to the closest predicted segment and
   only mark those segments as observed; hidden segments continue by prediction.
12. Add a coverage penalty when expected visible segments do not own enough
   support points.
13. Regenerate a fraction of particles around the current ordered masked-cloud
   fit, then score the mixed particle set with the same point-to-segment
   distance.
14. Track the fixed-length connected segment chain with the particle filter.

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

To compare HSV segmentation against PIDNet, set `detector.backend = "hsv"` in
`config.toml`. HSV supports a simple range threshold and a Gaussian color model
learned from labeled masks. You can also switch for one run:

```bash
../.venv/bin/python main.py --detector-backend hsv
```

In HSV mode, `hsv.fast_mask_only = true` skips skeleton/centerline extraction on
normal tracking frames. It samples mask pixels, lifts them to 3D, and scores the
particle filter directly by point-to-segment distance. Skeleton geometry is
still refreshed for initialization and reacquisition.

To tune HSV interactively against your existing labeled training masks:

```bash
../.venv/bin/python tools/hsv_tuning_gui.py
```

To fit and save HSV settings from labels without the GUI:

```bash
../.venv/bin/python tools/tune_hsv_from_masks.py --method gaussian --write-config
```

That scans `datasets/cable_pidnet/images/...` and `masks/...`, estimates a
robust HSV profile from pixels labeled as cable plus background negatives,
prints IoU/Dice on the labeled set, and updates only the `[hsv]` values in
`config.toml`.

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
connected 3D nodes. Important tuning sections:

- `pidnet`: model checkpoint, CUDA device, probability threshold.
- `detector`: PIDNet/HSV backend, resized inference, ROI reacquisition.
- `hsv`: range/Gaussian HSV baseline and fast mask-only tracking path.
- `measurement`: masked ZED point selection, depth rejection, prediction gating.
- `particle_filter`: particle count, process noise, scoring, occlusion.
- `point_cloud`: viewer point-cloud sampling and confidence-map cadence.
