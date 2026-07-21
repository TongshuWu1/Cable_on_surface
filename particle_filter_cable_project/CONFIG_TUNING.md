# Runtime Configuration Calibration

> The dataset and checkpoint used for the threshold table below were retired on
> 2026-07-20. These values are historical evidence only and are not active for
> the replacement model. The active configuration contains neutral `0.50`
> placeholders until validation-only calibration is completed again.

This file records why the values in `config.toml` were selected. It separates
parameters that can be calibrated from data from physical quantities that must
be measured. The current profile was calibrated on 2026-07-15 for the local
ZED2i and RTX 4080 system.

## Feature-ablation protocol

Use only the centralized `[features]` booleans for runtime ablations. Keep all
numeric parameters fixed, replay the same SVO, discard the same warm-up count,
and change one switch at a time. Preserve the startup `FEATURES ...` line with
the timing summary so every result records its effective estimator. For feature
interactions, test the full system, each single-feature removal, and then only
the small number of predeclared pairs; searching arbitrary combinations on the
test sequence would turn the test set into tuning data.

## Neural observation calibration

All 13 held-out images in `datasets/two_cable_pidnet/images/val` were evaluated
at their native 1280 x 720 resolution. Thresholds were selected per output head
rather than sharing one value because the heads have different probability
calibration.

| Observation | Selected threshold | Validation result |
| --- | ---: | --- |
| Generic cable body | 0.95 | IoU 0.790 with open 1, close 5 |
| `endpoints_cable1` | 0.50 | 12/13 two-endpoint frames, 2.51 px mean center error |
| `endpoints_cable2` | 0.98 | 13/13 frames, 1.39 px mean center error |
| Crossing | 0.77 | component F1 0.976, recall 1.000, precision 0.952 |

Full-scale inference was both more accurate and faster than 0.875, 0.75, and
0.625 scale on this GPU. CUDA AMP plus channels-last had a 4.82 ms median model
forward time in a 30-run isolated benchmark, compared with 7.67--10.68 ms for
the other memory-format/precision combinations.

The 21 predicted validation crossing regions had a median learned covariance
trace standard deviation of 6.59 px. Matched genuine centroids were normally
within 5 px of the labels. The validation set contains at most three labeled
crossing components per frame, so four live proposals retain one spare slot.
These proposals enter only the explicitly gated RGB crossing likelihood; they
never declare physical contact or modify the motion model.

## Live estimator sweeps

The earlier 28--50 second sweeps used the retired joint-coverage likelihood and
must not be cited as calibration evidence for the independent formulation.
Values retained for the first controlled independent-PF experiment are:

| Group | Selected value | Reason |
| --- | --- | --- |
| Particles | 800 | 600 increased residual; 1000 gave no accuracy gain |
| Cable segments | 14 | 16 increased residual and PF time |
| Measurement points | 1000 | 1500 increased residual and spread |
| Medoid set | 30 | 80 was slower and more diffuse |
| Node/direction noise | 0.010 m / 0.025 | best residual/spread compromise |
| Dense path support | weight 1.00, 5 samples/segment | dominant per-PF particle-to-cloud term; requires new ablation |
| Union final selector | top 30, weight 1.00 | bounded final representative search; weights and motion remain independent |
| Depth mode | NEURAL with fill | NEURAL_PLUS was slower and less accurate; no-fill was worse |
| Measurement confidence | 90, map every 2 frames | stricter 80/every-frame was slower and worse |
| Display cloud | 20,000 points every 4 frames | reduced capture cost without changing PF measurements |

No live accuracy claim is recorded for the new formulation until a fixed SVO
suite has been replayed. CPU/CUDA equivalence, independent random streams,
absence of global coverage from the per-PF likelihood, union-selector behavior,
and synthetic branch ambiguity are covered by automated tests. The runtime
fields `union-rank`, `rms`, `cov`, and `gain` expose the selector directly for
controlled feature ablations.

Reproduce the log summary with:

```powershell
..\.venv\Scripts\python.exe tools\summarize_live_log.py <stdout-log> --warmup-results 5
```

## Parameters deliberately not optimized from one scene

Camera source, resolution, working depth range, cable count, cable lengths,
and tape lengths describe the hardware or experiment. They must not be changed
to reduce an image residual. The current crossing experiment is image-space
only and contains no cable-diameter or physical-contact parameter.

Velocity damping/noise, endpoint-conditioned deformation, the required 10%
global random population, occlusion timeouts, and endpoint ambiguity safeguards
were retained because a static live scene cannot identify their fast-motion or
occlusion behavior. A defensible second-stage calibration requires a recorded
SVO suite with normal motion, fast endpoint motion, loss/reacquisition,
occlusion, image-only crossings at different depths, physical crossings,
single-cable operation, and double-crossing branch ambiguities. The present
values are an explicit starting profile, not a claim of global optimality.
