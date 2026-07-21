"""Summarize the per-second diagnostics emitted by ``main.py``.

This keeps live configuration comparisons reproducible.  It deliberately uses
only the completed tracking-result lines, and can discard the first few results
to remove PIDNet/CUDA warm-up from the steady-state statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


FLOAT = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def _number(text: str, expression: str) -> float | None:
    match = re.search(expression, text)
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def parse_tracking_line(line: str) -> dict | None:
    if not line.startswith("Frame ") or "| TRACK " not in line or "cables:" not in line:
        return None
    values = {
        "frame": _number(line, rf"^Frame {FLOAT}"),
        "gui_fps": _number(line, rf"\| GUI {FLOAT} FPS"),
        "capture_fps": _number(line, rf"\| CAPTURE {FLOAT} FPS"),
        "tracking_fps": _number(line, rf"\| TRACK {FLOAT} FPS"),
        "lag_frames": _number(line, rf"\| lag {FLOAT}f/"),
        "lag_ms": _number(line, rf"\| lag \d+f/{FLOAT}ms"),
        "capture_ms": _number(line, rf"\| capture {FLOAT}ms"),
        "ui_ms": _number(line, rf"\| ui {FLOAT}ms/update"),
        "worker_ms": _number(line, rf"\| worker {FLOAT}ms"),
        "detector_ms": _number(line, rf"\bdet={FLOAT}"),
        "measurement_fit_ms": _number(line, rf"\bfit={FLOAT}"),
        "particle_filter_ms": _number(line, rf"\bpf={FLOAT}\s*\| async"),
        "raw_residual_mm": _number(line, rf"\bres={FLOAT}mm"),
        "support_to_prior_mm": _number(line, rf"\bsupport2prior={FLOAT}mm"),
        "map_to_medoid_mm": _number(line, rf"\bMAP-MED={FLOAT}mm"),
        "mean_spread_mm": _number(line, rf"\bspread={FLOAT}mm"),
        "max_spread_mm": _number(line, rf"\bspread=[^\s]+/{FLOAT}mmmax"),
        "assignment": _number(line, rf"\bassign={FLOAT}"),
        "assignment_entropy": _number(line, rf"\bH={FLOAT}"),
        "endpoint_tangent_confidence": _number(line, rf"\btangent={FLOAT}/"),
        "mean_node_speed_mps": _number(line, rf"\bv={FLOAT}m/s"),
        "effective_sample_size": _number(line, rf"\bESS={FLOAT}"),
        "pf_prepare_ms": _number(line, rf"\bpfms:prep={FLOAT}"),
        "pf_tangent_ms": _number(line, rf"\bpfms:[^\n]*?tan={FLOAT}"),
        "pf_predict_ms": _number(line, rf"\bpfms:[^\n]*?pred={FLOAT}"),
        # LEGACY LOG COMPATIBILITY: runs recorded before the joint raw-mask
        # likelihood was correctly named used ``cons=`` for this same stage.
        "pf_measurement_ms": _number(
            line,
            rf"\bpfms:[^\n]*?(?:meas|cons)={FLOAT}",
        ),
        "pf_crossing_ms": _number(line, rf"\bpfms:[^\n]*?cross={FLOAT}"),
        "pf_union_ms": _number(line, rf"\bpfms:[^\n]*?union={FLOAT}"),
        "pf_estimate_ms": _number(line, rf"\bpfms:[^\n]*?est={FLOAT}"),
        "crossing_proposals": _number(line, rf"\bcross={FLOAT}\s+axes="),
        "crossing_axes": _number(line, rf"\bcross=[^\s]+\s+axes={FLOAT}"),
        "crossing_targets": _number(line, rf"\bcross=[^\s]+\s+axes=[^\s]+\s+targets={FLOAT}"),
        "crossing_reward": _number(line, rf"\bcross=[^|]*?\sR={FLOAT}"),
        "crossing_distance_px": _number(line, rf"\bcross=[^|]*?\sd={FLOAT}px"),
        "crossing_angle_error_deg": _number(line, rf"\bcross=[^|]*?\sa={FLOAT}deg"),
        "crossing_continuation_error_px": _number(
            line,
            rf"\bcross=[^|]*?\stwo-side={FLOAT}px",
        ),
        "observation_accepted_count": _number(line, rf"\brawobs={FLOAT}accepted/"),
        "observation_rejected_count": _number(line, rf"\brawobs=[^/]+/{FLOAT}rejected"),
        "observation_rejected_invalid": _number(line, rf"\binvalid:{FLOAT}"),
        "observation_rejected_depth_range": _number(line, rf"\brange:{FLOAT}"),
        "observation_rejected_confidence": _number(line, rf"\bconf:{FLOAT}"),
        "observation_rejected_component": _number(line, rf"\bcomponent:{FLOAT}"),
        "observation_rejected_morphology": _number(line, rf"\bmorph:{FLOAT}"),
        "observation_rejected_isolation": _number(line, rf"\bisolation:{FLOAT}"),
        "union_coverage_rms_mm": _number(line, rf"\bunion-rank=[^|]*?\srms={FLOAT}mm"),
        "union_coverage_max_mm": _number(line, rf"\bunion-rank=[^|]*?\smax={FLOAT}mm"),
        "union_huber_cost_mm2": _number(line, rf"\bunion-rank=[^|]*?\shuber={FLOAT}mm2"),
        "union_covered_fraction": _number(line, rf"\bunion-rank=[^|]*?\scov={FLOAT}"),
        "union_rms_gain_mm": _number(line, rf"\bunion-rank=[^|]*?\sgain={FLOAT}mm"),
        "union_coverage_fraction_gain": _number(
            line,
            rf"\bunion-rank=[^|]*?\sgain=[^\s]+/{FLOAT}cov",
        ),
    }
    visibility = re.search(r"\bvis=(\d+)/(\d+)", line)
    if visibility is not None and int(visibility.group(2)) > 0:
        values["visible_segment_fraction"] = (
            int(visibility.group(1)) / int(visibility.group(2))
        )
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def summarize_records(records: list[dict]) -> dict:
    summary = {"samples": len(records)}
    numeric_keys = sorted({
        key
        for record in records
        for key, value in record.items()
        if isinstance(value, (int, float)) and value is not None
    })
    for key in numeric_keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": sum(values) / len(values),
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--warmup-results", type=int, default=3)
    args = parser.parse_args()
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    records = [record for line in lines if (record := parse_tracking_line(line)) is not None]
    warmup = max(0, int(args.warmup_results))
    steady = records[warmup:]
    output = {
        "log": str(args.log.resolve()),
        "all_tracking_results": len(records),
        "discarded_warmup_results": min(warmup, len(records)),
        "steady_state": summarize_records(steady),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
