# Two-Cable Particle Filter Tracker

Real-time 3D reconstruction of two cables from a ZED camera, a PIDNet-S
segmentation model, and one particle filter per cable.

## Run

Start the ZED 3D viewer and tracker:

```powershell
..\.venv\Scripts\python.exe main.py
```

`main.py` is only the live tracking application. Runtime settings live in
`config.toml`; a missing setting is an error.

Start the labeling and training GUI separately:

```powershell
..\.venv\Scripts\python.exe tools\pidnet_training_gui.py
```

The active experiment uses:

- Dataset: `datasets/two_cable_pidnet`
- Checkpoint: `models/pidnet_two_cable_best.pt`
- Training parameters: `pidnet_two_cable_training_params.json`

## Measurement Model

PIDNet outputs four independent channels in this exact order:

1. Generic cable body
2. `endpoints_cable1`, containing both ends of Cable 1
3. `endpoints_cable2`, containing both ends of Cable 2
4. Cable crossing

The cable body is shared, but endpoint-channel identity is physical cable
identity: `endpoints_cable1` always anchors PF1 and `endpoints_cable2` always
anchors PF2. The two components within a channel are the two ends of that same
cable; their component order is aligned to the previous PF chain so a connected
component reorder cannot flip the chain. Crossing is an independent multilabel
channel, so marking a crossing does not erase cable or endpoint annotations.

The crossing head is only an RGB proposal. Each connected proposal is matched
to one segment on each PF chain, producing arc lengths, closest 3D points,
centerline distance, cable-surface gap, depth order, covariance, and confidence.
The viewer labels proposals as `RGB CROSSING` until both physical cable
diameters are configured and the 3D gap is within the contact tolerance. This
observation is diagnostic only; it does not impose a PF contact constraint.

Deterministic mask indices are sampled directly from the ZED GPU depth buffer
into one compact shared 3D support cloud. The cloud is never hard-split or
deleted before filtering. A fused CUDA kernel evaluates every shared point
against both predicted particle populations. The posterior populations define
soft `cable1`, `cable2`, and `outlier` responsibilities, which remain ambiguous
at genuine image crossings instead of forcing a premature ownership decision.

For point `p_n` and predicted particle `Q_i^k`, the cable compatibility is

```text
L_i,n = sum_k w_i,k exp(-d(p_n, Q_i^k)^2 / (2 sigma^2)).
```

Normalizing the two cable compatibilities together with an explicit outlier
component gives the responsibility used by the robust, bounded measurement
cost. Each PF is weighted once from its predicted weights; there is no
preliminary likelihood multiplication and no RANSAC-selected support subset.

## Cable Model

Each cable is a fixed-length polygonal chain:

```text
P0 -- P1 -- P2 -- ... -- PN
```

`cable.segments = N` creates `N + 1` nodes. Cable `i` uses the physical length
in `cable.lengths_m[i]`, so every segment has length
`cable.lengths_m[i] / N`. Both endpoints are fixed to the detected 3D endpoint
positions to the configured tolerance. The PF estimates the middle-node
configuration and velocity while preserving segment lengths. The live estimate
is the arithmetic mean of the configured number of highest-weight particles;
that mean is projected back onto the fixed segment lengths and detected
endpoints before it is displayed or used by downstream geometry.

The 3D viewer exposes estimator diagnostics without changing the PF state:
translucent magenta/cyan lines are the exact particles included in the
arithmetic mean, the white chain is the single MAP particle, the normal
green/yellow/red chain is the constrained top-particle average, and purple
two-standard-deviation bars show node-wise spread along each node's principal
uncertainty axis. `MAP-AVG`, mean/max spread, and start/end MAP-to-average
direction disagreement are reported per PF. Press `P` to toggle this diagnostic
layer; the fixed 10% global recovery population and all PF calculations remain
unchanged.

The same layer draws the two measured inward endpoint tangents. Its label also
reports tangent confidence/support, mean cable ownership, ownership entropy,
visible segment fraction, and the active endpoint-conditioned proposal ratio.

On measurement frames the transition is the explicit mixture

```text
0.65 posterior-local motion
+ 0.25 endpoint/tangent-conditioned deformation
+ 0.10 globally random endpoint-constrained recovery.
```

The conditioned component perturbs a few low-frequency sine deformation modes,
not every node independently. Its scale increases with cable slack, so a loose
loop receives broader shape proposals than a nearly taut cable. Endpoint-only
frames constrain and predict the cable while correctly reporting that the
interior shape is occluded. Complete occlusion cannot uniquely determine the
hidden shape; particle spread is the corresponding uncertainty diagnostic.

## Runtime Pipeline

1. A dedicated capture thread acquires RGB plus rotating ZED GPU depth buffers.
2. The detector stage runs PIDNet once for cable, endpoints_cable1,
   endpoints_cable2, and crossing masks.
3. The tracker lifts both endpoint groups and shared cable support in parallel.
4. Each PF predicts its endpoint-constrained transition population. One fused
   CUDA distance pass then computes shared ownership, robust costs, coverage,
   likelihoods, visibility, and posterior diagnostics for both cables.
5. The main thread renders the latest complete result without blocking capture
   or estimation.

Detection for frame `t + 1` overlaps tracking for frame `t`. Strict
backpressure keeps at most one detected frame waiting, so GPU time is not spent
on stale frames. Runtime output reports GUI, camera capture, and completed
tracking FPS separately.

## Label And Train

In the training GUI, paint the generic cable body, the two endpoint classes,
and the small crossing region. Save overlap-preserving annotations in
`masks_layers`; the rendered PNG masks are previews, not the source of truth for
multilabel crossings.

The GUI can browse train, validation, and capture folders, identify missing
masks, delete bad frames, check dataset integrity, train on CUDA, and preview
the current model.

Headless training is also available:

```powershell
..\.venv\Scripts\python.exe tools\train_pidnet_cable.py --dataset datasets\two_cable_pidnet --output models\pidnet_two_cable_best.pt --device cuda
```

The required split layout is:

```text
datasets/two_cable_pidnet/
  images/train/*.png
  images/val/*.png
  masks/train/*.png
  masks/val/*.png
  masks_layers/train/*.npz
  masks_layers/val/*.npz
```

## Code Layout

- `main.py`: independent ZED capture, detector, tracker, and live viewer stages
- `cable_cuda.py`: strict CUDA runtime and fused-kernel launch interface
- `cable_cuda_kernels.cu`: fused constraints, two-cable distances, and ZED sampling
- `cable_crossing.py`: RGB crossing proposals and continuous 3D contact observations
- `cable_pidnet.py`: strict four-channel PIDNet inference
- `cable_detection.py`: 2D mask cleanup and 3D measurement construction
- `cable_particle_filter.py`: endpoint-constrained two-cable PF
- `zed_spatial.py`: ZED point-cloud conversion
- `zed_split_viewer.py`: RGB and 3D rendering
- `tools/pidnet_training_gui.py`: labeling, dataset inspection, and training UI
- `tools/train_pidnet_cable.py`: PIDNet training and evaluation

There are no legacy single-cable model paths or automatic model/config
fallbacks. Incompatible checkpoints, missing configuration, unavailable CUDA,
or invalid measurements fail explicitly.
