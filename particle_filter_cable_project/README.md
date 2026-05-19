# Particle Filter Cable Project

Starter project for particle-filter cable skeleton reconstruction and tracking.

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
3. Skeletonize each visible fragment, then stitch the fragment centerlines into
   one ordered cable path for initialization.
4. Use the ordered RGB centerline to sample the ZED point cloud.
5. Use those ordered 3D points only to initialize the chain.
6. Score each particle by distance from a small support set of lifted skeleton
   centerline points to the closest segment in that particle. The support cap is
   `particle_filter.measurement_points`; `measurement.source_points = "mask"`
   can switch back to sampled masked RGB points for comparison. The score keeps
   the best `particle_filter.score_keep_fraction` distances, so sparse bad depth
   points do not dominate the particle weight.
7. During occlusion, assign visible points to the closest predicted segment and
   only mark those segments as observed; hidden segments continue by prediction.
8. Add a coverage penalty when expected visible segments do not own enough
   support points.
9. Regenerate a fraction of particles around the current measured 3D cable fit,
   then score the mixed particle set with the same point-to-segment distance.
10. Track the fixed-length connected segment chain with the particle filter.

Label masks and train the PIDNet-S detector on CUDA:

```bash
../.venv/bin/python tools/pidnet_training_gui.py --device cuda
```

The GUI workflow is:

1. Open/capture RGB frames.
2. Paint only the cable pixels in the binary mask. Every unpainted pixel is
   background.
3. Save labels to `datasets/cable_pidnet/images/...` and `datasets/cable_pidnet/masks/...`.
4. Click `Train PIDNet-S on CUDA`.
5. Run live tracking with the saved checkpoint.

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
../.venv/bin/python tools/train_pidnet_cable.py --dataset /path/to/cable_dataset --output models/pidnet_cable_best.pt --epochs 80 --batch-size 8 --imgsz 512 --device cuda
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
same downstream tracking output: binary mask, skeleton, ordered centerline,
lifted 3D points, then particle-filter segment scoring.

The live UI shows RGB on the left and the ZED point cloud on the right. The
default cable model is `cable.segments = 2`, so the tracked skeleton has 3
nodes. Important tuning sections:

- `pidnet`: model checkpoint, CUDA device, probability threshold.
- `detector`: PIDNet/HSV backend, resized inference, ROI reacquisition.
- `hsv`: range/Gaussian HSV baseline and fast mask-only tracking path.
- `measurement`: RGB centerline lifting, depth rejection, prediction gating.
- `particle_filter`: particle count, process noise, scoring, occlusion.
- `point_cloud`: viewer point-cloud sampling and confidence-map cadence.
