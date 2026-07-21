from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import math
import statistics
import threading


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    group: str
    requires: tuple = ()
    algorithmic: bool = True


FEATURE_SPECS = (
    FeatureSpec("particle_filter", "Independent particle filters", "Core"),
    FeatureSpec("cable_mask_morphology", "Cable-mask morphology", "Preprocess"),
    FeatureSpec("cable_component_filter", "Cable component filter", "Preprocess"),
    FeatureSpec("endpoint_mask_morphology", "Endpoint-mask morphology", "Preprocess"),
    FeatureSpec("endpoint_component_filter", "Endpoint component filter", "Preprocess"),
    FeatureSpec("depth_confidence_filter", "ZED confidence filter", "Preprocess"),
    FeatureSpec("endpoint_association_support", "Endpoint support cost", "Observation"),
    FeatureSpec("pf_velocity", "Velocity transition", "Motion", ("particle_filter",)),
    FeatureSpec("pf_adaptive_motion", "Adaptive motion noise", "Motion", ("particle_filter",)),
    FeatureSpec("pf_occlusion_prediction", "Occlusion prediction", "Motion", ("particle_filter",)),
    FeatureSpec("pf_direction_smoothing", "Direction smoothing", "Motion", ("particle_filter",)),
    FeatureSpec("pf_posterior_medoid", "Posterior medoid", "Estimate", ("particle_filter",)),
    FeatureSpec("pf_endpoint_tangent", "Endpoint tangent", "Endpoint", ("particle_filter",)),
    FeatureSpec(
        "pf_endpoint_tangent_ransac",
        "Tangent RANSAC",
        "Endpoint",
        ("particle_filter", "pf_endpoint_tangent"),
    ),
    FeatureSpec(
        "pf_endpoint_tangent_likelihood",
        "Tangent likelihood",
        "Likelihood",
        ("particle_filter", "pf_endpoint_tangent"),
    ),
    FeatureSpec(
        "pf_conditioned_proposals",
        "Endpoint proposals",
        "Proposal",
        ("particle_filter", "pf_endpoint_tangent"),
    ),
    FeatureSpec("pf_global_random_particles", "10% random proposals", "Proposal", ("particle_filter",)),
    FeatureSpec("pf_robust_measurement", "Robust body loss", "Likelihood", ("particle_filter",)),
    FeatureSpec("pf_dense_path_support", "Dense path support", "Likelihood", ("particle_filter",)),
    FeatureSpec("pf_union_coverage", "Union coverage selection", "Estimate", ("particle_filter",)),
    FeatureSpec("pf_bend_regularization", "Bend regularization", "Likelihood", ("particle_filter",)),
    FeatureSpec("crossing_proposals", "RGB crossing proposals", "Crossing"),
    FeatureSpec(
        "crossing_likelihood",
        "RGB crossing likelihood",
        "Crossing",
        ("particle_filter", "crossing_proposals"),
    ),
    FeatureSpec(
        "point_support_coloring",
        "Point support colors",
        "Visualization",
        ("particle_filter",),
        algorithmic=False,
    ),
    FeatureSpec(
        "particle_diagnostics_overlay",
        "Particle diagnostics",
        "Visualization",
        ("particle_filter",),
        algorithmic=False,
    ),
)

FEATURE_BY_KEY = {spec.key: spec for spec in FEATURE_SPECS}


def minimal_baseline_state(initial_state):
    state = {key: False for key in FEATURE_BY_KEY}
    state["particle_filter"] = True
    state["point_support_coloring"] = bool(
        initial_state.get("point_support_coloring", True)
    )
    state["particle_diagnostics_overlay"] = bool(
        initial_state.get("particle_diagnostics_overlay", True)
    )
    return state


def isolated_feature_state(initial_state, feature_key):
    if feature_key not in FEATURE_BY_KEY:
        raise KeyError(f"Unknown ablation feature: {feature_key}")
    state = minimal_baseline_state(initial_state)
    pending = [feature_key]
    while pending:
        key = pending.pop()
        state[key] = True
        pending.extend(FEATURE_BY_KEY[key].requires)
    return state


def leave_one_out_state(initial_state, feature_key):
    """Disable one configured algorithm and any feature that requires it.

    A dependent likelihood cannot remain active after its observation stage is
    removed.  The returned state therefore represents the smallest valid
    removal bundle and makes that dependency explicit in the recorded state.
    """
    if feature_key not in FEATURE_BY_KEY:
        raise KeyError(f"Unknown ablation feature: {feature_key}")
    state = {key: bool(initial_state.get(key, False)) for key in FEATURE_BY_KEY}
    state[feature_key] = False
    changed = True
    while changed:
        changed = False
        for spec in FEATURE_SPECS:
            if state[spec.key] and any(not state[dependency] for dependency in spec.requires):
                state[spec.key] = False
                changed = True
    validate_feature_state(state)
    return state


def validate_feature_state(state):
    missing = []
    for spec in FEATURE_SPECS:
        if not bool(state.get(spec.key, False)):
            continue
        for dependency in spec.requires:
            if not bool(state.get(dependency, False)):
                missing.append(f"{spec.key} requires {dependency}")
    if missing:
        raise ValueError("Invalid ablation state: " + "; ".join(missing))


class RuntimeFeatureController:
    def __init__(self, initial_state, validator=None):
        self._lock = threading.Lock()
        self._initial_state = {
            key: bool(initial_state.get(key, False)) for key in FEATURE_BY_KEY
        }
        validate_feature_state(self._initial_state)
        self._state = dict(self._initial_state)
        self._history = {0: dict(self._initial_state)}
        self._validator = validator
        self._revision = 0
        self._reset_revision = 0
        self._recording = False
        self._metrics = {}

    @property
    def initial_state(self):
        return dict(self._initial_state)

    def snapshot(self):
        with self._lock:
            return {
                "revision": int(self._revision),
                "reset_revision": int(self._reset_revision),
                "features": dict(self._state),
                "recording": bool(self._recording),
                "metrics": dict(self._metrics),
            }

    def apply(self, state, *, reset_filter=True):
        candidate = {key: bool(state.get(key, False)) for key in FEATURE_BY_KEY}
        validate_feature_state(candidate)
        if self._validator is not None:
            self._validator(candidate)
        with self._lock:
            self._state = candidate
            self._revision += 1
            self._history[self._revision] = dict(candidate)
            if reset_filter:
                self._reset_revision = self._revision
            return int(self._revision)

    def snapshot_for_revision(self, revision):
        with self._lock:
            revision = int(revision)
            state = self._history.get(revision)
            if state is None:
                raise KeyError(f"Unknown feature revision {revision}.")
            return {
                "revision": revision,
                "features": dict(state),
            }

    def set_recording(self, enabled):
        with self._lock:
            self._recording = bool(enabled)

    def publish_metrics(self, metrics):
        with self._lock:
            self._metrics = dict(metrics or {})


def feature_state_from_args(args):
    return {spec.key: bool(getattr(args, spec.key)) for spec in FEATURE_SPECS}


def apply_feature_state_to_args(args, state):
    for key in FEATURE_BY_KEY:
        setattr(args, key, bool(state[key]))


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ExperimentRecorder:
    """Write reproducible live ablation records without retaining frame images."""

    def __init__(self, output_directory, metadata):
        self.output_directory = Path(output_directory)
        self.metadata = dict(metadata)
        self._stream = None
        self.path = None

    def _open(self):
        if self._stream is not None:
            return
        self.output_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = self.output_directory / f"pf_ablation_{stamp}.jsonl"
        self._stream = open(self.path, "w", encoding="utf-8", buffering=1)
        self._write({"type": "metadata", **self.metadata})

    def record(
        self,
        frame_index,
        feature_snapshot,
        diagnostics,
        timing=None,
        frame_timestamp_s=None,
    ):
        self._open()
        self._write({
            "type": "frame",
            "frame_index": int(frame_index),
            "feature_revision": int(feature_snapshot["revision"]),
            "features": dict(feature_snapshot["features"]),
            "diagnostics": diagnostics,
            "timing": dict(timing or {}),
            "frame_timestamp_s": _finite_number(frame_timestamp_s),
        })

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _write(self, value):
        self._stream.write(json.dumps(value, default=_json_default, sort_keys=True) + "\n")


ABLATION_METRICS = (
    "effective_sample_size",
    "path_support_rms_m",
    "mean_node_spread_m",
    "max_node_spread_m",
    "mean_node_speed_mps",
    "endpoint_speed_mps",
    "endpoint_motion_innovation_m",
    "measurement_to_filter_m",
    "estimate_temporal_delta_m",
    "support_affinity",
    "supported_sample_fraction",
    "crossing_reward",
    "crossing_distance_px",
    "crossing_angle_error_deg",
)


def read_ablation_jsonl(path):
    """Read one recorder output and reject malformed experiment records."""
    path = Path(path)
    metadata = None
    frames = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            record_type = record.get("type")
            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError(f"{path} contains more than one metadata record.")
                metadata = record
            elif record_type == "frame":
                frames.append(record)
            else:
                raise ValueError(f"Unknown record type at {path}:{line_number}: {record_type!r}")
    if metadata is None:
        raise ValueError(f"{path} has no metadata record.")
    return metadata, frames


def summarize_ablation_records(metadata, frames, *, warmup_frames=0):
    """Aggregate comparable per-revision metrics from a recorded live run."""
    grouped = {}
    for frame in frames:
        revision = int(frame.get("feature_revision", 0))
        grouped.setdefault(revision, []).append(frame)
    initial_features = dict(metadata.get("initial_features", {}))
    summaries = []
    for revision in sorted(grouped):
        ordered = sorted(grouped[revision], key=lambda item: int(item.get("frame_index", 0)))
        retained = ordered[max(0, int(warmup_frames)):]
        features = dict((ordered[-1] if ordered else {}).get("features", {}))
        changes = {
            key: value
            for key, value in sorted(features.items())
            if bool(initial_features.get(key, False)) != bool(value)
        }
        metric_values = {name: [] for name in ABLATION_METRICS}
        metric_values["filter_ms"] = []
        metric_values["worker_ms"] = []
        for frame in retained:
            diagnostics = frame.get("diagnostics") or {}
            timing = frame.get("timing") or {}
            for name in ABLATION_METRICS:
                _append_finite(metric_values[name], diagnostics.get(name))
            for cable_index, cable_diagnostics in enumerate(diagnostics.get("per_cable") or ()):
                if not isinstance(cable_diagnostics, dict):
                    continue
                for name in ABLATION_METRICS:
                    key = f"pf{cable_index + 1}.{name}"
                    values = metric_values.setdefault(key, [])
                    _append_finite(values, cable_diagnostics.get(name))
            _append_finite(metric_values["filter_ms"], _seconds_to_ms(timing.get("filter")))
            worker_seconds = sum(
                value for value in (_finite_number(item) for item in timing.values())
                if value is not None
            )
            if timing:
                _append_finite(metric_values["worker_ms"], 1000.0 * worker_seconds)
        summaries.append({
            "revision": revision,
            "frame_count": len(ordered),
            "retained_frame_count": len(retained),
            "features": features,
            "changes_from_configured": changes,
            "metrics": {
                name: _distribution_summary(values)
                for name, values in metric_values.items()
                if values
            },
        })
    baseline = next(
        (item for item in summaries if not item["changes_from_configured"]),
        summaries[0] if summaries else None,
    )
    if baseline is not None:
        baseline_metrics = baseline["metrics"]
        for item in summaries:
            comparisons = {}
            for name, values in item["metrics"].items():
                reference = baseline_metrics.get(name)
                if reference is None:
                    continue
                delta = float(values["median"] - reference["median"])
                reference_median = float(reference["median"])
                comparisons[name] = {
                    "median_delta": delta,
                    "median_percent": (
                        100.0 * delta / abs(reference_median)
                        if abs(reference_median) > 1e-20
                        else None
                    ),
                }
            item["baseline_revision"] = int(baseline["revision"])
            item["comparison_to_baseline"] = comparisons
    return {
        "schema_version": 1,
        "source_metadata": metadata,
        "warmup_frames_per_revision": max(0, int(warmup_frames)),
        "revisions": summaries,
    }


def _seconds_to_ms(value):
    value = _finite_number(value)
    return None if value is None else 1000.0 * value


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _append_finite(output, value):
    number = _finite_number(value)
    if number is not None:
        output.append(number)


def _distribution_summary(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}
    percentile_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "std": statistics.pstdev(ordered),
        "p95": ordered[percentile_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class AblationControlPanel:
    """Small independent Tk window for explicit, resettable live ablations."""

    def __init__(self, controller):
        self.controller = controller
        self._thread = threading.Thread(target=self._run, name="pf-ablation-ui", daemon=True)
        self._stop = threading.Event()
        self._root = None

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        import tkinter as tk
        from tkinter import messagebox, ttk

        root = tk.Tk()
        self._root = root
        root.title("Cable PF Ablation Control")
        root.geometry("620x820")
        root.minsize(560, 620)

        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="Research Ablation Control",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Changes apply together and reset the PF so posterior history does not contaminate comparisons.",
            wraplength=570,
        ).pack(anchor="w", pady=(2, 10))

        variables = {
            key: tk.BooleanVar(value=value)
            for key, value in self.controller.snapshot()["features"].items()
        }
        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        groups = {}
        for spec in FEATURE_SPECS:
            frame = groups.get(spec.group)
            if frame is None:
                frame = ttk.Frame(notebook, padding=10)
                notebook.add(frame, text=spec.group)
                groups[spec.group] = frame
            text = spec.label
            if spec.requires:
                text += "  [requires " + ", ".join(
                    FEATURE_BY_KEY[key].label for key in spec.requires
                ) + "]"
            ttk.Checkbutton(frame, text=text, variable=variables[spec.key]).pack(anchor="w", pady=3)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(10, 4))
        status_var = tk.StringVar(value="Configured state loaded")

        def load_state(state, message):
            for key, value in state.items():
                variables[key].set(bool(value))
            status_var.set(message)

        def apply_state():
            state = {key: variable.get() for key, variable in variables.items()}
            try:
                revision = self.controller.apply(state, reset_filter=True)
            except Exception as exc:
                messagebox.showerror("Invalid ablation", str(exc), parent=root)
                return
            status_var.set(f"Applied revision {revision}; PF reset requested")

        ttk.Button(controls, text="Apply + reset PF", command=apply_state).pack(side="left")
        ttk.Button(
            controls,
            text="Configured",
            command=lambda: load_state(self.controller.initial_state, "Configured preset loaded; press Apply"),
        ).pack(side="left", padx=6)
        ttk.Button(
            controls,
            text="Minimal baseline",
            command=lambda: load_state(
                minimal_baseline_state(self.controller.initial_state),
                "Minimal baseline loaded; press Apply",
            ),
        ).pack(side="left")

        isolation = ttk.Frame(outer)
        isolation.pack(fill="x", pady=4)
        algorithm_specs = [
            spec for spec in FEATURE_SPECS
            if spec.algorithmic and spec.key != "particle_filter"
        ]
        algorithm_labels = [spec.label for spec in algorithm_specs]
        key_by_label = {spec.label: spec.key for spec in algorithm_specs}
        isolated_var = tk.StringVar(value=algorithm_labels[0])
        ttk.Label(isolation, text="Isolate:").pack(side="left")
        ttk.Combobox(
            isolation,
            textvariable=isolated_var,
            values=algorithm_labels,
            state="readonly",
            width=31,
        ).pack(side="left", padx=6)
        ttk.Button(
            isolation,
            text="Load isolated preset",
            command=lambda: load_state(
                isolated_feature_state(
                    self.controller.initial_state,
                    key_by_label[isolated_var.get()],
                ),
                f"Isolated {isolated_var.get()} with explicit dependencies; press Apply",
            ),
        ).pack(side="left")

        leave_out = ttk.Frame(outer)
        leave_out.pack(fill="x", pady=4)
        leave_out_var = tk.StringVar(value=algorithm_labels[0])
        ttk.Label(leave_out, text="Remove:").pack(side="left")
        ttk.Combobox(
            leave_out,
            textvariable=leave_out_var,
            values=algorithm_labels,
            state="readonly",
            width=31,
        ).pack(side="left", padx=6)
        ttk.Button(
            leave_out,
            text="Load leave-one-out",
            command=lambda: load_state(
                leave_one_out_state(
                    self.controller.initial_state,
                    key_by_label[leave_out_var.get()],
                ),
                f"Removed {leave_out_var.get()} and required dependents; press Apply",
            ),
        ).pack(side="left")

        recording_var = tk.BooleanVar(value=False)

        def toggle_recording():
            self.controller.set_recording(recording_var.get())
            status_var.set("Recording enabled" if recording_var.get() else "Recording stopped")

        ttk.Checkbutton(
            outer,
            text="Record self-describing JSONL experiment",
            variable=recording_var,
            command=toggle_recording,
        ).pack(anchor="w", pady=(8, 2))
        metrics_var = tk.StringVar(value="Waiting for live diagnostics")
        ttk.Label(outer, textvariable=metrics_var, justify="left", font=("Consolas", 10)).pack(
            fill="x", pady=(8, 2)
        )
        ttk.Label(outer, textvariable=status_var, foreground="#2060a0", wraplength=570).pack(
            fill="x", pady=(4, 0)
        )

        def refresh():
            if self._stop.is_set():
                root.destroy()
                return
            snapshot = self.controller.snapshot()
            metrics = snapshot["metrics"]
            lines = [f"revision={snapshot['revision']}  recording={int(snapshot['recording'])}"]
            for key in (
                "tracking_fps",
                "filter_ms",
                "effective_sample_size",
                "path_support_rms_m",
                "estimate_temporal_delta_m",
                "mean_node_spread_m",
                "crossing_reward",
                "crossing_distance_px",
                "crossing_angle_error_deg",
            ):
                if key in metrics:
                    lines.append(f"{key}={metrics[key]}")
            metrics_var.set("\n".join(lines))
            root.after(250, refresh)

        root.protocol("WM_DELETE_WINDOW", root.withdraw)
        root.after(250, refresh)
        root.mainloop()
        self._root = None
