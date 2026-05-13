# Particle Filter Project

Starter project for particle-filter 3D reconstruction and tracking.

The first app uses the same UI pattern as the cable project:

- left panel: live RGB image
- right panel: live ZED point cloud
- resizing the window keeps both panels scaled proportionally
- drag the right panel to orbit the 3D scene; use the mouse wheel to zoom

Run from this folder:

```bash
../.venv/bin/python main.py
```

The default ZED input mode is `HD720` at `60` FPS for both tracking and color
calibration.

The MuJoCo sphere reconstruction opens in its own window by default and shows the primary tracked sphere on a table. By default, `--sphere-radius 0` estimates the ball radius in meters from the first valid stereo point cloud/RGB blob measurement, then the particle filter locks that radius and does not change it while tracking. Use `--sphere-radius` when you want to force a measured physical radius.

The 3D estimate runs a lightweight scene split before tracking:

- ball cap: points inside the RGB ball mask
- table: dominant flat background plane
- other: sampled non-table clutter/background

The table plane is cached and refreshed every few frames, so the tracker does
not run table RANSAC every frame. Table-like points that leak into the ball mask
are removed before fitting. Since the depth mask usually sees only a partial cap
of the ball, a known-radius depth-cap estimate uses the masked 3D points plus
the RGB circle center to estimate the full ball center. After the first valid
radius lock, that same locked radius is fed back into reconstruction so lifting
the ball does not make the detector resize the sphere. The particle filter
target is the ball center only, with the ball radius locked by default. Use
`--no-particle-filter` for raw per-frame measurements.

The live UI is intentionally simple: RGB on the left, sampled ZED point cloud on
the right, and the tracked ball overlay. Segmentation rendering is off by
default; use `--segmentation-visualization points` only when debugging the
ball/table/other split.

```bash
../.venv/bin/python main.py
```

Run without MuJoCo:

```bash
../.venv/bin/python main.py --no-mujoco
```

Calibrate sphere color detection:

```bash
../.venv/bin/python tools/sphere_threshold_setup.py
```

In the threshold tool, press `p` to capture frames, `i` to select the sphere ROI from a captured frame, tune the HSV/blob trackbars until only the ball is detected, then press `s` to save `tools/sphere_color_profile.json`.

The app can track multiple balls that match the same color profile. RGB candidates
are reconstructed independently and associated to one particle filter per ball;
the OpenGL view shows every active tracked ball, while MuJoCo follows the primary
track.

Occlusion recovery is on by default. When an existing track loses the normal
circular RGB blob, the tracker projects that ball into the RGB image and searches
a small local ROI with looser shape rules. If a partial colored patch remains,
that patch is used with the locked physical radius to update the track instead of
going prediction-only. If the RGB patch is still not enough for a normal 3D
measurement, the unmatched track also checks the organized ZED point cloud inside
the projected locked-radius sphere footprint. Points that are clearly in front of
the predicted sphere are treated as occluders and ignored; points that fit the
sphere surface update the particle filter as an occlusion-aware surface
measurement.

The particle filter state is shown in the overlay and sent to the MuJoCo window.

The default particle filter is tuned for the ball moving on or above the table:
it uses 400 particles over only the ball center, estimates radius in real-world
meters from the first valid stereo frame, and locks that physical radius. Each
frame adds random position noise, scores particles against the measured center,
fixed-radius surface fit, and RGB projection, then resamples the best particles.
When the mask drops out, the filter keeps a short noisy position-only prediction.
Surface-normal likelihood is available as an optional experiment, but it is off
by default because the live path prioritizes `HD720` at `60` FPS. On each valid
measurement frame it also injects a fraction of particles around the current
segmented ball-cap measurement before weighting; this measurement-proposal step
reduces lag when the ball is lifted quickly. Full scene/table segmentation is
skipped during normal tracking unless segmentation visualization or debug
recording is enabled; the table plane is refreshed with a lighter cached update.

Useful tuning/debug flags:

```bash
../.venv/bin/python main.py --debug-record-dir debug_runs/latest
../.venv/bin/python main.py --max-balls 4
../.venv/bin/python main.py --no-occlusion-recovery
../.venv/bin/python main.py --occlusion-surface-std 0.02 --occlusion-min-visible-points 24
../.venv/bin/python main.py --sphere-confidence-max 80
../.venv/bin/python main.py --segmentation-visualization points
../.venv/bin/python main.py --table-update-interval 6 --table-max-points 3500
../.venv/bin/python main.py --pf-particles 600 --pf-surface-points 256
../.venv/bin/python main.py --pf-proposal-ratio 0.45 --pf-proposal-depth-std 0.05
../.venv/bin/python main.py --pf-surface-weight 0.5 --pf-projection-weight 0.8
../.venv/bin/python main.py --pf-surface-normal-likelihood --pf-surface-normal-weight 0.5
```

If the ball still trails your hand, increase `--pf-proposal-ratio` toward
`0.60`, increase `--pf-position-blend` toward `0.55`, or increase
`--pf-process-std` toward `0.03`. If the track is too jumpy, reduce
`--pf-proposal-ratio` toward `0.25`, reduce `--pf-position-blend` toward
`0.20`, increase `--pf-measurement-std` toward `0.04`, or reduce
`--pf-projection-weight` / `--pf-surface-weight`.
