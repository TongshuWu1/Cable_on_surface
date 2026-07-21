# Two-Cable Particle Filter Tracker

Real-time 3D reconstruction of one or two cables from a ZED camera, a PIDNet-S
segmentation model, and one independent particle filter per cable.

## Run

Start the ZED 3D viewer and tracker:

```powershell
..\.venv\Scripts\python.exe main.py
```

`main.py` is only the live tracking application. Runtime settings live in
`config.toml`; a missing setting is an error.

## Feature Ablations

Every optional estimator stage has one boolean in `[features]` in
`config.toml`. Numeric values stay fixed when a stage is disabled. The separate
**Feature Controls** window contains only feature switches and one **Apply**
button. Enabling a feature automatically enables its required parents;
disabling a parent visibly disables its dependents. Applying a revision resets
both PFs so the previous posterior cannot contaminate the comparison.

The switches cover mask morphology and component rejection, ZED confidence,
endpoint-pair support, velocity and adaptive motion, occlusion prediction,
direction smoothing, posterior medoid selection, endpoint tangent/PCA/RANSAC,
tangent-conditioned proposals, the fixed global-random population, robust
dense path support, union final-representative selection, bend regularization,
RGB crossing proposals, the RGB crossing likelihood, and the two diagnostic
renderers. Dependencies are strict: an invalid combination is rejected rather
than silently changing another method.

Each PF state always retains fixed segment lengths, connected node order,
same-channel endpoint identity, normalized weights, and the measured endpoints.
PF1 and PF2 do not share particle indices, ancestors, random draws, weights, or
estimates. These are model invariants, not optional heuristics.

The experiment recorder remains available programmatically to store
configuration and checkpoint hashes, seeds, physical values, exact feature
state, per-frame diagnostics, and stage timings. Summarize a recorded run with:

```powershell
..\.venv\Scripts\python.exe tools\summarize_ablation.py experiments\pf_ablation_20260721T120000.000000Z.jsonl --warmup-frames 10
```

Warm-up is discarded independently after every PF-reset revision. Report both
feature-only and leave-one-out experiments: the former measures what a method
can add to the baseline, while the latter measures what the complete configured
system loses without it.

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

The crossing head predicts only an RGB crossing probability; it never predicts
cable ownership, depth order, or physical contact. Cable-mask pixels in an
annulus around each proposal estimate two unoriented image axes. The endpoint
channels identify PF1/PF2 but do not label the crossing axes. Each independent
PF therefore evaluates both local axes and uses the better axis hypothesis for
each crossing. Neither a straight endpoint chord nor the previous PF estimate
is allowed to label an axis: the chord is not a local tangent for a curved
cable, while the previous PF would make a wrong branch self-reinforcing. A PF
gains a bounded log-weight reward only when its projected particle passes
through the RGB crossing region and continues along one observed axis on both
sides. No 3D gap, radius, contact, or depth-order term is present.

For particle `k`, the crossing reward is

```text
R_cross(k) = gamma exp(-d_rgb(k)^2 / (2 sigma_p^2)
                       -d_angle(k)^2 / (2 sigma_angle^2)
                       -d_two_side(k)^2 / (2 sigma_side^2)).
log w(k) <- log w(k) + lambda_cross R_cross(k).
```

`d_two_side` is the larger of the projected-polyline distances to two probe
points placed on opposite sides of the crossing centroid. A path that touches
the centroid but turns onto the other cable's outgoing branch therefore cannot
receive the maximum crossing reward.

The two PFs apply this equation separately. Crossing evidence changes ranking
only; it does not create a joint particle state or crossing-conditioned proposal.

Deterministic mask indices are sampled directly from the ZED GPU depth buffer
into one explicit, shared, unlabeled 3D observation. Observation acceptance is
independent of both PFs: the only rejection reasons are invalid/range depth,
poor ZED confidence, configured 2D mask cleanup, or insufficient local 3D
radius support. The spatial filter has its own ablation switch. A PF estimate
is never used to remove or relabel an observation point. For PF `a`, segment
`j`, and dense sample `m`, let
`x_a,k,j,m` be a point on particle `k`. Its measurement cost is

```text
C_path,a(k) = (1/JM) sum_j sum_m rho(
    min_n ||x_a,k,j,m - p_n||^2
).
```

`rho` is the configured truncated quadratic. This particle-to-cloud direction
is deliberate: there is no cloud-to-particle coverage term in either PF's
likelihood, so observations on the other cable cannot penalize PF1 for failing
to explain them or pull two cables toward an average. The dense path term is
the dominant likelihood; endpoint-tangent and optional bend terms are added
continuously, while endpoint positions and per-segment length remain hard
constraints. Each PF combines this likelihood only with its own temporal prior.

After those independent posteriors are scored, a separate final-estimate layer
compares the top `K` particles from each PF. It selects the pair whose *union*
best explains the accepted observation cloud while retaining the independent posterior
probabilities:

```text
score(k,l) = log w1(k) + log w2(l)
             - lambda_union/sigma^2 mean_y rho_H(
                   min(d(y, X1(k)), d(y, X2(l)))).

rho_H(r) = 0.5 r^2                         if r <= delta
           delta (r - 0.5 delta)           otherwise.
```

For one configured cable the same equation has one candidate index. This stage
does not change weights, resample particles, couple motion models, or assign
cloud points permanently. It only chooses the final representative(s) from
already plausible particles. Unlike the per-PF truncated path loss, Huber loss
does not become flat: a legitimate distant loop continues to influence final
selection, while one extreme point grows only linearly. The runtime reports
each selected rank, unexplained-point RMS and maximum distance, Huber cost,
covered fraction, and signed gain relative to independent rank-1 selection.

The UI has fixed semantic-overlay colors that do not depend on PF proximity:
accepted NN cable observations are orange, observation-quality rejections are
red, and crossing-head observations are blue. Ordinary scene points retain
their original ZED RGB values.
Invalid-depth rejected pixels can appear red in RGB but have no finite 3D point
to draw. The PF cannot turn an accepted orange point gray.

This formulation has an explicit identifiability limit: if two routes between
the same endpoints have equal fixed length, equal endpoint tangents, and equal
dense cloud support (for example, a hybrid route between two crossings), the
instantaneous point-set likelihood cannot distinguish them. The temporal prior
preserves the previously supported mode but does not create new topological
evidence. `test_double_crossing_geometry_is_ambiguous_without_temporal_prior`
keeps that limitation visible for the later graph/topology experiment.

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
is a posterior medoid selected from the configured highest-weight particles.
It is therefore an actual scored particle, not a pointwise average that could
invent an unobserved cable shape between two distinct modes.

The 3D viewer exposes estimator diagnostics without changing the PF state:
faint magenta/cyan lines are the exact top particles considered by the medoid
calculation, the white chain is the single MAP particle, the normal
green/yellow/red chain is the selected posterior medoid, and purple two-
standard-deviation bars show node-wise spread along each node's principal
uncertainty axis. `MAP-MED`, spread, direction disagreement, and the exact
local/endpoint/global proposal fractions are reported per PF. Press `P` to
toggle this layer.
Blue samples are lifted independently from the NN crossing mask. Post-estimate
contact geometry is intentionally not computed. The RGB panel draws the two
observed axis hypotheses, each PF's selected axis, projected closest point, centroid
distance, angle error, two-sided continuation error, and crossing reward.

The same layer draws the two measured inward endpoint tangents. Local RANSAC
selects a narrow endpoint-connected directional consensus, PCA/eigendecomposition
refines that direction, and the previous posterior tangent is only a bounded
tie-breaker. Tangent alignment is also a continuous particle likelihood. The
label reports tangent confidence/support, mean support affinity, dense-sample
coverage, visibility, and proposal ratios.

On measurement frames the transition is the explicit mixture

```text
0.65 posterior-local motion
+ 0.25 endpoint-tangent-conditioned deformation
+ 0.10 globally random endpoint-constrained recovery.
```

Conditioned particles use a few low-frequency deformation modes projected into
the chain's local normal space, not independent XYZ kicks. Slack increases the
mode scale, but displacement is capped at roughly one segment. Endpoint speed
and constant-velocity innovation continuously enlarge transition covariance;
the scale is smoothed and capped. Endpoint-only frames constrain and predict
the cable while correctly reporting that the interior shape is occluded.

Complete interior occlusion remains non-identifiable. The PF propagates its
previous posterior for the configured bounded interval, while the independent
10% global population supports reacquisition when measurements return. Particle
spread reports the resulting uncertainty rather than hiding it with a fitted
curve.

## Runtime Pipeline

1. A dedicated capture thread acquires RGB plus rotating ZED GPU depth buffers.
2. The detector stage runs PIDNet once for cable, endpoints_cable1,
   endpoints_cable2, and crossing masks.
3. The tracker lifts both endpoint groups and one shared cable-support cloud.
4. Each PF independently resamples, predicts, constrains, and scores on its own
   CUDA stream. Both read the same unlabeled cloud; neither writes the other's
   particle state or weights.
5. A bounded top-`K` CUDA selector chooses the one/two final representatives
   whose union explains the shared cloud, without creating a joint PF state.
6. RGB crossing proposals estimate two image axes and independently reward each
   PF's calibrated projection; the RGB panel renders the resulting diagnostics.
7. The main thread renders the latest complete result without blocking capture
   or estimation.

Detection for frame `t + 1` overlaps tracking for frame `t`. Strict
backpressure keeps at most one detected frame waiting, so GPU time is not spent
on stale frames. Runtime output reports GUI, camera capture, and completed
tracking FPS separately.

## Label And Train

The editable annotation schema maps one-to-one onto the four neural outputs:

1. generic cable body;
2. both endpoints of Cable 1;
3. both endpoints of Cable 2;
4. RGB crossing proposal.

Endpoint and crossing strokes also retain the generic-cable bit, so the cable
target remains continuous while the attributes overlap it. The source of truth
is the versioned bit mask in `masks_layers`; `masks/*.png` is only a flat visual
preview. There are no cable1/cable2 neural body classes.

Every captured frame belongs to a named collection session, and an entire
session belongs to exactly one of `train`, `val`, or locked `test`. The dataset
manifest records the split, notes, manual verification state, and explicit
negative-frame state. Validation threshold calibration never uses
the locked test set. The application deliberately performs no automatic label
validation and never marks an annotation correct on the user's behalf.

The GUI provides undo/redo, unique capture identifiers, non-blocking
near-duplicate warnings, an explicit verified-negative action, channel
visibility controls, per-channel probability/TP/FP/FN views, validation-only
threshold calibration, prediction-as-unverified-draft, session coverage, and
CUDA training controls. **Predict Current Capture as Draft** freezes or selects
the current capture, runs one inference, converts all four heads into the same
editable bit layers used by manual painting, and returns to the Annotate tab.
The draft is always unverified; pixel edits invalidate prior verification, and
the manifest records the checkpoint hash, thresholds, cleanup settings, and
human-review provenance. For the physical markers, use a matte saturated
magenta/blue marker for Cable 1 and the stable green marker for Cable 2; both
ends of a cable use the same marker appearance.

Training optimizes the four heads independently:

```text
L = L_body + lambda_e mean(L_endpoint1, L_endpoint2)
    + lambda_x L_crossing + lambda_boundary L_body_boundary,
L_head = focal_BCE_head + lambda_dice soft_Dice_head.
```

The focal balance is derived from the training-set prevalence of each channel;
its weights are normalized so rare endpoint and crossing heads remain on the
same loss scale as the body head. Endpoint and crossing validation quality is
the mean of region IoU and component F1. The checkpoint score is a softened
weighted harmonic mean of body and head quality: it responds to partial early
progress, while a failed head still caps the score. Thresholds are calibrated independently on
validation data, while component metrics report precision, recall, F1, and
centroid error. Component assignment requires both centroid proximity and
reasonable region-size agreement, so a large centered blob cannot count as a
correct endpoint or crossing. Locked-test evaluation is opt-in and uses the already selected
validation thresholds. The trainer also uses update-count-warmed EMA weights,
cosine learning-rate decay, gradient clipping, early stopping, AMP, TF32,
channels-last tensors, optional GPU photometric augmentation, resumable
checkpoints, and optional deterministic execution. The EMA warm-up prevents a
short, small-dataset run from retaining a large contribution from the randomly
initialized model. Images and masks remain compact `uint8` tensors in loader
workers and are converted and normalized after transfer to CUDA, reducing
Windows worker memory and host-to-device traffic.
Compatible checkpoints can optionally initialize matching tensors for a
pretraining-versus-random-initialization ablation without treating initialization
as a resumed experiment.

Each run writes the best and last checkpoints, epoch CSV history, environment
and Git metadata, and a content-hashed dataset snapshot next to the model.
Exact resume requires the checkpoint's dataset hash and optimization settings
to match the current run; use initialization instead when intentionally starting
a new experiment. The annotation UI blocks dataset mutations during training,
uses collision-safe names for imported images, and commits image, mask, and
layered annotations atomically.
Before optimization starts, the trainer reports independent-session counts,
positive-frame coverage for every output head, frames containing both endpoint
groups, and optimizer updates per epoch.  Sparse or distribution-shifted splits
produce visible warnings but are never modified automatically.
If early stopping triggers, the log reports the exact no-improvement epoch
window, patience, best and current scores, required `best + min_delta` score,
and current per-head validation quality.  A patience of zero disables it.

Headless training is also available:

```powershell
..\.venv\Scripts\python.exe tools\train_pidnet_cable.py --dataset datasets\two_cable_pidnet --output models\pidnet_two_cable_best.pt --device cuda
```

The required split layout is:

```text
datasets/two_cable_pidnet/
  dataset_manifest.json
  images/train/*.png
  images/val/*.png
  images/test/*.png
  masks/train/*.png
  masks/val/*.png
  masks/test/*.png
  masks_layers/train/*.npz
  masks_layers/val/*.npz
  masks_layers/test/*.npz
```

## Code Layout

- `main.py`: independent ZED capture, detector, tracker, and live viewer stages
- `cable_cuda.py`: strict CUDA runtime and fused-kernel launch interface
- `cable_cuda_kernels.cu`: fused constraints, dense path distances, and ZED sampling
- `cable_crossing.py`: RGB crossing proposals, image-axis estimation, and projection rewards
- `cable_pidnet.py`: strict four-channel PIDNet inference
- `cable_detection.py`: 2D mask cleanup and 3D measurement construction
- `cable_particle_filter.py`: independent endpoint-constrained cable PFs
- `pf_ablation.py`: strict feature registry, live control UI, and experiment records
- `pidnet_dataset.py`: versioned session metadata, split isolation, and dataset snapshots
- `zed_spatial.py`: ZED point-cloud conversion
- `zed_split_viewer.py`: RGB and 3D rendering
- `tools/pidnet_training_gui.py`: labeling, dataset inspection, and training UI
- `tools/train_pidnet_cable.py`: PIDNet training and evaluation
- `tools/summarize_live_log.py`: reproducible steady-state live-log statistics
- `tools/summarize_ablation.py`: per-revision accuracy/stability/runtime summaries
- `CONFIG_TUNING.md`: calibration data, live sweeps, selected values, and limits

There are no legacy single-cable model paths or automatic model/config
fallbacks. Incompatible checkpoints, missing configuration, unavailable CUDA,
or invalid measurements fail explicitly.
