from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import math
import multiprocessing
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
    FeatureSpec("cable_spatial_outlier_filter", "3D spatial outlier filter", "Preprocess"),
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


def dependency_safe_feature_toggle(state, feature_key, enabled):
    """Return the valid feature state produced by one explicit UI toggle.

    Enabling a feature enables its required parents. Disabling a feature
    disables every active dependent. This keeps dependencies visible in the
    checkboxes while ensuring the Apply button can never submit a structurally
    invalid feature combination.
    """
    if feature_key not in FEATURE_BY_KEY:
        raise KeyError(f"Unknown ablation feature: {feature_key}")
    candidate = {
        key: bool(state.get(key, False))
        for key in FEATURE_BY_KEY
    }
    candidate[feature_key] = bool(enabled)
    if enabled:
        pending = list(FEATURE_BY_KEY[feature_key].requires)
        while pending:
            dependency = pending.pop()
            if candidate[dependency]:
                continue
            candidate[dependency] = True
            pending.extend(FEATURE_BY_KEY[dependency].requires)
    else:
        changed = True
        while changed:
            changed = False
            for spec in FEATURE_SPECS:
                if candidate[spec.key] and any(
                    not candidate[dependency]
                    for dependency in spec.requires
                ):
                    candidate[spec.key] = False
                    changed = True
    validate_feature_state(candidate)
    return candidate


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


def _run_feature_control_window(connection, initial_state):
    """Run Tk in a dedicated process where it owns the process main thread."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Cable Tracker Features")
    root.geometry("500x760")
    root.minsize(440, 560)

    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass

    outer = ttk.Frame(root, padding=16)
    outer.pack(fill="both", expand=True)
    ttk.Label(
        outer,
        text="Feature Controls",
        font=("Segoe UI", 17, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        outer,
        text="Turn features on or off, then click Apply. Applying resets both particle filters.",
        wraplength=450,
    ).pack(anchor="w", pady=(3, 12))

    list_border = ttk.Frame(outer)
    list_border.pack(fill="both", expand=True)
    canvas = tk.Canvas(list_border, highlightthickness=0, borderwidth=0)
    scrollbar = ttk.Scrollbar(list_border, orient="vertical", command=canvas.yview)
    feature_frame = ttk.Frame(canvas, padding=(4, 2, 10, 10))
    feature_window = canvas.create_window((0, 0), window=feature_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    feature_frame.bind(
        "<Configure>",
        lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.bind(
        "<Configure>",
        lambda event: canvas.itemconfigure(feature_window, width=event.width),
    )
    canvas.bind_all(
        "<MouseWheel>",
        lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"),
    )

    current_state = {
        key: bool(initial_state.get(key, False))
        for key in FEATURE_BY_KEY
    }
    applied_state = dict(current_state)
    variables = {
        key: tk.BooleanVar(root, value=value)
        for key, value in current_state.items()
    }
    checkbuttons = []
    status_var = tk.StringVar(root, value="No unapplied changes")

    footer = ttk.Frame(outer)
    footer.pack(fill="x", pady=(12, 0))
    ttk.Separator(footer).pack(fill="x", pady=(0, 10))
    status_label = ttk.Label(footer, textvariable=status_var, wraplength=330)
    status_label.pack(side="left", fill="x", expand=True, padx=(0, 10))
    apply_button = ttk.Button(footer, text="Apply")
    apply_button.pack(side="right", ipadx=18, ipady=4)

    def set_controls_enabled(enabled):
        widget_state = "normal" if enabled else "disabled"
        for widget in checkbuttons:
            widget.configure(state=widget_state)
        apply_button.configure(state=widget_state)

    def show_pending_status(automatic_changes=0):
        if current_state == applied_state:
            status_var.set("No unapplied changes")
            apply_button.configure(state="disabled")
        else:
            suffix = ""
            if automatic_changes:
                suffix = f" ({automatic_changes} dependencies adjusted)"
            status_var.set("Changes ready" + suffix)
            apply_button.configure(state="normal")

    def toggle_feature(feature_key):
        nonlocal current_state
        requested = bool(variables[feature_key].get())
        previous = dict(current_state)
        current_state = dependency_safe_feature_toggle(
            current_state,
            feature_key,
            requested,
        )
        for key, value in current_state.items():
            variables[key].set(value)
        automatic_changes = sum(
            previous[key] != current_state[key]
            for key in current_state
            if key != feature_key
        )
        show_pending_status(automatic_changes)

    last_group = None
    for spec in FEATURE_SPECS:
        if spec.group != last_group:
            ttk.Label(
                feature_frame,
                text=spec.group,
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w", pady=(12 if last_group is not None else 4, 3))
            last_group = spec.group
        checkbox = ttk.Checkbutton(
            feature_frame,
            text=spec.label,
            variable=variables[spec.key],
            command=lambda key=spec.key: toggle_feature(key),
        )
        checkbox.pack(anchor="w", fill="x", pady=2)
        checkbuttons.append(checkbox)

    def apply_state():
        set_controls_enabled(False)
        status_var.set("Applying...")
        try:
            connection.send({"type": "apply", "features": dict(current_state)})
        except (BrokenPipeError, EOFError, OSError) as exc:
            set_controls_enabled(True)
            status_var.set(f"Tracker connection lost: {exc}")

    apply_button.configure(command=apply_state, state="disabled")

    def poll_connection():
        nonlocal applied_state
        try:
            while connection.poll():
                message = connection.recv()
                message_type = message.get("type")
                if message_type == "shutdown":
                    root.destroy()
                    return
                if message_type != "apply_result":
                    continue
                set_controls_enabled(True)
                if message.get("ok"):
                    applied_state = dict(message["features"])
                    status_var.set(
                        f"Applied revision {message['revision']} - particle filters reset"
                    )
                    apply_button.configure(state="disabled")
                else:
                    status_var.set(f"Could not apply: {message.get('error', 'unknown error')}")
        except (BrokenPipeError, EOFError, OSError):
            root.destroy()
            return
        root.after(50, poll_connection)

    def close_window():
        try:
            connection.close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(50, poll_connection)
    root.mainloop()


class AblationControlPanel:
    """Process-isolated feature switches connected to a runtime controller."""

    def __init__(self, controller):
        self.controller = controller
        self._process = None
        self._connection = None

    def start(self):
        if self._process is not None and self._process.is_alive():
            return
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_run_feature_control_window,
            args=(child_connection, self.controller.snapshot()["features"]),
            name="pf-feature-ui",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def poll(self):
        connection = self._connection
        if connection is None:
            return
        try:
            while connection.poll():
                message = connection.recv()
                if message.get("type") != "apply":
                    continue
                state = message.get("features", {})
                try:
                    revision = self.controller.apply(state, reset_filter=True)
                    response = {
                        "type": "apply_result",
                        "ok": True,
                        "revision": revision,
                        "features": dict(state),
                    }
                except Exception as exc:
                    response = {
                        "type": "apply_result",
                        "ok": False,
                        "error": str(exc),
                    }
                connection.send(response)
        except (BrokenPipeError, EOFError, OSError):
            connection.close()
            self._connection = None

    def close(self):
        connection = self._connection
        process = self._process
        if connection is not None:
            try:
                connection.send({"type": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if connection is not None:
            connection.close()
        self._connection = None
        self._process = None
