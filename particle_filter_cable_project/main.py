import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import time
import tomllib
import traceback

import cv2
import numpy as np
import pyzed.sl as sl

from cable_detection import (
    CableEstimate3D,
    attach_endpoint_markers_to_measurement,
    cable_measurement_from_mask_points,
    cleanup_marker_mask,
    endpoint_markers_from_mask,
    polyline_residual,
)
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    filtered_cable_estimate,
)
from zed_spatial import (
    DEPTH_MODES,
    RESOLUTIONS,
    configure_input_source,
    configure_viewer_from_zed,
    live_point_cloud_to_vertices,
)
from zed_split_viewer import ZedDepthGLViewer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIDNET_CHECKPOINT = PROJECT_DIR / "models/pidnet_cable_best.pt"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"


def load_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def config_value(config, section, key, default, base_dir=None):
    value = config.get(section, {}).get(key, default)
    if isinstance(default, Path):
        return path_from_config(value, base_dir or DEFAULT_CONFIG_PATH.parent)
    return value


def required_per_cable_float_values(value, cable_count, name, *, minimum=0.0):
    cable_count = max(1, int(cable_count))
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list with exactly {cable_count} values.")
    if len(value) != cable_count:
        raise ValueError(f"{name} must contain exactly {cable_count} values.")
    try:
        return [max(float(minimum), float(item)) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values.") from exc


def per_cable_value(values, cable_index, name):
    index = int(cable_index)
    if index < 0 or index >= len(values):
        raise IndexError(f"{name} is missing value for cable index {index}.")
    return float(values[index])


def cable_length_m(args, cable_index):
    return per_cable_value(args.cable_lengths_m, cable_index, "cable.lengths_m")


def cable_segment_length_m(args, cable_index):
    return per_cable_value(args.derived_segment_lengths_m, cable_index, "derived cable segment lengths")


def endpoint_marker_tape_length_m(args, cable_index):
    return per_cable_value(args.endpoint_marker_tape_lengths_m, cable_index, "endpoint_markers.tape_lengths_m")


def path_from_config(value, base_dir):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def parse_config_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args, _remaining = parser.parse_known_args()
    return args.config


def parse_args():
    pf_defaults = CableParticleFilterConfig()
    config_path = parse_config_path()
    config_base_dir = Path(config_path).resolve().parent
    config = load_config(config_path)
    parser = argparse.ArgumentParser(
        description="Cable particle-filter UI: RGB left, point cloud right.",
    )
    parser.add_argument("--config", type=Path, default=config_path, help="TOML config file. CLI flags override config values.")
    parser.add_argument("--input-svo-file", default=config_value(config, "camera", "input_svo_file", ""))
    parser.add_argument("--ip-address", default=config_value(config, "camera", "ip_address", ""))
    parser.add_argument("--resolution", choices=RESOLUTIONS, default=config_value(config, "camera", "resolution", "HD720"))
    parser.add_argument("--fps", type=int, default=config_value(config, "camera", "fps", 60))
    parser.add_argument("--depth-mode", choices=DEPTH_MODES, default=config_value(config, "camera", "depth_mode", "NEURAL_PLUS"))
    parser.add_argument("--depth-min", type=float, default=config_value(config, "depth", "min_m", 0.01))
    parser.add_argument("--depth-max", type=float, default=config_value(config, "depth", "max_m", 1.0))
    parser.add_argument("--confidence", type=int, default=config_value(config, "depth", "confidence", 95))
    parser.add_argument("--texture-confidence", type=int, default=config_value(config, "depth", "texture_confidence", 100))
    parser.add_argument("--depth-fill", action=argparse.BooleanOptionalAction, default=config_value(config, "depth", "fill", True))
    parser.add_argument("--live-stride", type=int, default=config_value(config, "point_cloud", "stride", 4))
    parser.add_argument("--live-max-points", type=int, default=config_value(config, "point_cloud", "max_points", 100000), help="0 keeps every sampled point.")
    parser.add_argument("--cloud-update-every", type=int, default=config_value(config, "point_cloud", "cloud_update_every", 1), help="Update the visual point cloud every N frames; tracking still runs every frame.")
    parser.add_argument("--confidence-update-every", type=int, default=config_value(config, "point_cloud", "confidence_update_every", 1), help="Retrieve ZED confidence every N frames; 0 disables confidence filtering.")
    parser.add_argument("--detector-scale", type=float, default=config_value(config, "detector", "scale", 0.50), help="Run RGB cable detection at this image scale, then lift coordinates back to full resolution.")
    parser.add_argument("--detector-update-every", type=int, default=config_value(config, "detector", "update_every", 1), help="Run RGB detection + 3D measurement every N frames; skipped frames use PF prediction.")
    parser.add_argument("--point-size", type=float, default=config_value(config, "viewer", "point_size", 2.0))
    parser.add_argument("--rgb-width", type=int, default=config_value(config, "viewer", "rgb_width", 620))
    parser.add_argument("--cloud-width", type=int, default=config_value(config, "viewer", "cloud_width", 1180))
    parser.add_argument("--height", type=int, default=config_value(config, "viewer", "height", 900))
    parser.add_argument(
        "--rgb-view",
        choices=("segmentation", "tracking", "mask"),
        default=config_value(config, "viewer", "rgb_view", "segmentation"),
        help="Left RGB panel visualization. Segmentation shows the live PIDNet mask clearly.",
    )
    parser.add_argument("--neural-detector-checkpoint", type=Path, default=config_value(config, "pidnet", "checkpoint", DEFAULT_PIDNET_CHECKPOINT, config_base_dir), help="PIDNet cable segmentation checkpoint.")
    parser.add_argument("--neural-detector-device", default=config_value(config, "pidnet", "device", "cuda"), help="PyTorch device for PIDNet detector. Default: cuda.")
    parser.add_argument("--neural-detector-threshold", type=float, default=config_value(config, "pidnet", "threshold", 0.50), help="PIDNet probability threshold for the binary cable mask.")
    parser.add_argument("--neural-detector-base-channels", type=int, default=config_value(config, "pidnet", "base_channels", 24))
    parser.add_argument("--neural-detector-amp", action=argparse.BooleanOptionalAction, default=config_value(config, "pidnet", "amp", True), help="Use CUDA automatic mixed precision for PIDNet inference.")
    parser.add_argument("--neural-detector-channels-last", action=argparse.BooleanOptionalAction, default=config_value(config, "pidnet", "channels_last", True), help="Use channels-last CUDA tensors for PIDNet inference.")
    parser.add_argument("--endpoint-marker-min-area", type=int, default=config_value(config, "endpoint_markers", "min_area_px", 50))
    parser.add_argument("--endpoint-marker-min-points", type=int, default=config_value(config, "endpoint_markers", "min_points", 8))
    parser.add_argument("--endpoint-marker-open-kernel", type=int, default=config_value(config, "endpoint_markers", "open_kernel", 3))
    parser.add_argument("--endpoint-marker-close-kernel", type=int, default=config_value(config, "endpoint_markers", "close_kernel", 5))
    parser.add_argument("--endpoint-marker-points", type=int, default=config_value(config, "endpoint_markers", "max_points_per_marker", 256))
    parser.add_argument("--endpoint-marker-max-count", type=int, default=config_value(config, "endpoint_markers", "max_count", 2))
    parser.add_argument("--endpoint-marker-tape-lengths", type=float, nargs="+", default=config_value(config, "endpoint_markers", "tape_lengths_m", None))
    parser.add_argument("--endpoint-marker-offset-to-tips", action=argparse.BooleanOptionalAction, default=config_value(config, "endpoint_markers", "offset_to_tips", False))
    parser.add_argument("--detector-min-area", type=int, default=config_value(config, "detector", "min_area_px", 80))
    parser.add_argument("--detector-open-kernel", type=int, default=config_value(config, "detector", "open_kernel", 3))
    parser.add_argument("--detector-close-kernel", type=int, default=config_value(config, "detector", "close_kernel", 5))
    parser.add_argument("--cable-segments", type=int, default=config_value(config, "cable", "segments", 2))
    parser.add_argument("--cable-count", type=int, default=config_value(config, "cable", "count", 1), help="Number of separate cables to reconstruct.")
    parser.add_argument("--cable-lengths", type=float, nargs="+", default=config_value(config, "cable", "lengths_m", None), help="Physical cable lengths in meters, one value per cable.")
    parser.add_argument("--cable-max-points", type=int, default=config_value(config, "cable", "max_visual_points", 1000))
    parser.add_argument("--measurement-smoothing", action=argparse.BooleanOptionalAction, default=config_value(config, "measurement", "smoothing", True), help="Smooth fitted 3D measurement nodes before particle-filter proposal injection.")
    parser.add_argument("--measurement-smoothing-alpha", type=float, default=config_value(config, "measurement", "smoothing_alpha", 0.30), help="Current-frame weight for measurement node smoothing.")
    parser.add_argument("--measurement-smoothing-gate", type=float, default=config_value(config, "measurement", "smoothing_gate_m", 0.040), help="Do not smooth when raw measurement jumps farther than this mean node distance. Use 0 to always smooth.")
    parser.add_argument(
        "--measurement-prediction-gate",
        type=float,
        default=config_value(config, "measurement", "prediction_gate_m", 0.12),
        help="Before fitting, keep lifted 3D points within this distance of the previous filtered cable. Use 0 to disable.",
    )
    parser.add_argument(
        "--measurement-gate-reacquire-after",
        type=int,
        default=config_value(config, "measurement", "gate_reacquire_after", 4),
        help="Disable prediction gating after this many prediction-only frames so the tracker can reacquire.",
    )
    parser.add_argument("--cable-confidence-max", type=float, default=config_value(config, "measurement", "confidence_max", 85.0), help="Use <0 to disable.")
    parser.add_argument("--particle-filter", action=argparse.BooleanOptionalAction, default=config_value(config, "particle_filter", "enabled", True))
    parser.add_argument("--pf-particles", type=int, default=config_value(config, "particle_filter", "particles", pf_defaults.particle_count))
    parser.add_argument("--pf-initial-node-std", type=float, default=config_value(config, "particle_filter", "initial_node_std_m", pf_defaults.initial_node_std_m))
    parser.add_argument("--pf-initial-direction-std", type=float, default=config_value(config, "particle_filter", "initial_direction_std", pf_defaults.initial_direction_std))
    parser.add_argument("--pf-process-std", type=float, default=config_value(config, "particle_filter", "process_node_std_m", pf_defaults.process_node_std_m))
    parser.add_argument("--pf-process-direction-std", type=float, default=config_value(config, "particle_filter", "process_direction_std", pf_defaults.process_direction_std))
    parser.add_argument("--pf-velocity", action=argparse.BooleanOptionalAction, default=config_value(config, "particle_filter", "velocity", pf_defaults.velocity_enabled))
    parser.add_argument("--pf-velocity-damping", type=float, default=config_value(config, "particle_filter", "velocity_damping", pf_defaults.velocity_damping))
    parser.add_argument("--pf-velocity-measurement-blend", type=float, default=config_value(config, "particle_filter", "velocity_measurement_blend", pf_defaults.velocity_measurement_blend))
    parser.add_argument("--pf-velocity-process-std", type=float, default=config_value(config, "particle_filter", "velocity_process_std_mps", pf_defaults.velocity_process_std_mps))
    parser.add_argument("--pf-max-node-speed", type=float, default=config_value(config, "particle_filter", "max_node_speed_mps", pf_defaults.max_node_speed_mps))
    parser.add_argument("--pf-direction-smooth-passes", type=int, default=config_value(config, "particle_filter", "direction_smooth_passes", pf_defaults.direction_smooth_passes))
    parser.add_argument("--pf-measurement-std", type=float, default=config_value(config, "particle_filter", "measurement_node_std_m", pf_defaults.measurement_node_std_m))
    parser.add_argument(
        "--pf-measurement-points",
        type=int,
        default=config_value(config, "particle_filter", "measurement_points", pf_defaults.measurement_max_points),
        help="Maximum masked ZED cable points used to score particles.",
    )
    parser.add_argument(
        "--pf-scoring-backend",
        choices=("auto", "cuda", "cpu"),
        default=config_value(config, "particle_filter", "scoring_backend", pf_defaults.scoring_backend),
        help="Particle scoring backend. auto uses CUDA tensors when available.",
    )
    parser.add_argument(
        "--pf-score-chunk-points",
        type=int,
        default=config_value(config, "particle_filter", "score_chunk_points", pf_defaults.score_chunk_points),
        help="Point chunk size for tensor particle scoring.",
    )
    parser.add_argument(
        "--pf-endpoint-ordering",
        action=argparse.BooleanOptionalAction,
        default=config_value(config, "particle_filter", "endpoint_ordering", pf_defaults.endpoint_ordering),
        help="Order masked 3D cable points by graph endpoints for PF initialization/reacquisition.",
    )
    parser.add_argument(
        "--pf-ordering-max-points",
        type=int,
        default=config_value(config, "particle_filter", "ordering_max_points", pf_defaults.ordering_max_points),
        help="Maximum masked cable points used for endpoint/geodesic ordering.",
    )
    parser.add_argument(
        "--pf-ordering-knn",
        type=int,
        default=config_value(config, "particle_filter", "ordering_knn", pf_defaults.ordering_knn),
        help="K-nearest neighbors for endpoint/geodesic ordering graph.",
    )
    parser.add_argument(
        "--pf-endpoint-refresh-interval",
        type=int,
        default=config_value(config, "particle_filter", "endpoint_refresh_interval", pf_defaults.endpoint_refresh_interval),
        help="Run the endpoint graph ordering every N measurement updates to correct reference-order drift. Use 0 to disable.",
    )
    parser.add_argument(
        "--pf-reference-ordering",
        action=argparse.BooleanOptionalAction,
        default=config_value(config, "particle_filter", "reference_ordering", pf_defaults.reference_ordering),
        help="After initialization, order masked points against the current filtered chain instead of rebuilding the endpoint graph.",
    )
    parser.add_argument(
        "--pf-reference-ordering-gate",
        type=float,
        default=config_value(config, "particle_filter", "reference_ordering_gate_m", pf_defaults.reference_ordering_gate_m),
        help="Max distance from the current chain for points used to build the ordered measurement fit.",
    )
    parser.add_argument(
        "--pf-measurement-fit-interval",
        type=int,
        default=config_value(config, "particle_filter", "measurement_fit_interval", pf_defaults.measurement_fit_interval),
        help="Fit an ordered measurement chain every N healthy PF updates. Skipped updates still score raw mask points.",
    )
    parser.add_argument(
        "--pf-endpoint-penalty-weight",
        type=float,
        default=config_value(config, "particle_filter", "endpoint_penalty_weight", pf_defaults.endpoint_penalty_weight),
        help="Extra score weight that keeps particle start/end near measured cable endpoints.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-ratio",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_ratio", pf_defaults.measurement_proposal_ratio),
        help="Maximum fraction of particles regenerated around the current measured cable fit.",
    )
    parser.add_argument(
        "--pf-measurement-reset-error",
        type=float,
        default=config_value(config, "particle_filter", "measurement_reset_error_m", pf_defaults.measurement_reset_error_m),
        help="Reset the PF from a valid measured cable fit when mean node disagreement exceeds this many meters. Use 0 to disable.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-stable-ratio",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_stable_ratio", pf_defaults.measurement_proposal_stable_ratio),
        help="Proposal fraction used when the measured cable is close to the current estimate.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-start-error",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_start_error_m", pf_defaults.measurement_proposal_start_error_m),
        help="Mean node error below which proposal injection stays at the stable ratio.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-full-error",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_full_error_m", pf_defaults.measurement_proposal_full_error_m),
        help="Mean node error where proposal injection reaches the maximum ratio.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-std",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_node_std_m", pf_defaults.measurement_proposal_node_std_m),
        help="3D start-node noise in meters for measurement-proposal particles.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-direction-std",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_direction_std", pf_defaults.measurement_proposal_direction_std),
        help="Direction noise for measurement-proposal particles.",
    )
    parser.add_argument("--pf-ransac-inlier-selection", action=argparse.BooleanOptionalAction, default=config_value(config, "particle_filter", "ransac_inlier_selection", pf_defaults.ransac_inlier_selection_enabled))
    parser.add_argument("--pf-ransac-hypotheses", type=int, default=config_value(config, "particle_filter", "ransac_hypotheses", pf_defaults.ransac_hypotheses))
    parser.add_argument("--pf-ransac-subset-points", type=int, default=config_value(config, "particle_filter", "ransac_subset_points", pf_defaults.ransac_subset_points))
    parser.add_argument("--pf-ransac-inlier-distance", type=float, default=config_value(config, "particle_filter", "ransac_inlier_distance_m", pf_defaults.ransac_inlier_distance_m))
    parser.add_argument("--pf-ransac-min-points", type=int, default=config_value(config, "particle_filter", "ransac_min_points", pf_defaults.ransac_min_points))
    parser.add_argument(
        "--pf-score-keep-fraction",
        type=float,
        default=config_value(config, "particle_filter", "score_keep_fraction", pf_defaults.score_keep_fraction),
        help="Robust score keeps this fraction of lowest point distances per particle.",
    )
    parser.add_argument(
        "--pf-coverage-penalty",
        type=float,
        default=config_value(config, "particle_filter", "coverage_penalty_m", pf_defaults.coverage_penalty_m),
        help="Penalty scale in meters when expected visible segments do not own enough points. Use 0 to disable.",
    )
    parser.add_argument(
        "--pf-coverage-min-fraction",
        type=float,
        default=config_value(config, "particle_filter", "coverage_min_fraction", pf_defaults.coverage_min_fraction),
        help="Minimum fraction of support points each expected visible segment should own.",
    )
    parser.add_argument("--pf-bend-penalty", type=float, default=config_value(config, "particle_filter", "bend_penalty_m", pf_defaults.bend_penalty_m))
    parser.add_argument("--pf-coarse-score-points", type=int, default=config_value(config, "particle_filter", "coarse_score_points", pf_defaults.coarse_score_points))
    parser.add_argument("--pf-coarse-score-full-fraction", type=float, default=config_value(config, "particle_filter", "coarse_score_full_fraction", pf_defaults.coarse_score_full_fraction))
    parser.add_argument("--pf-coarse-score-min-particles", type=int, default=config_value(config, "particle_filter", "coarse_score_min_particles", pf_defaults.coarse_score_min_particles))
    parser.add_argument("--pf-top-particles", type=int, default=config_value(config, "particle_filter", "top_particles", pf_defaults.top_particle_count))
    parser.add_argument("--pf-global-random-ratio", type=float, default=config_value(config, "particle_filter", "global_random_particle_ratio", pf_defaults.global_random_particle_ratio))
    parser.add_argument("--pf-global-random-effective-ratio", type=float, default=config_value(config, "particle_filter", "global_random_effective_ratio", pf_defaults.global_random_effective_ratio))
    parser.add_argument("--pf-global-random-bounds-padding", type=float, default=config_value(config, "particle_filter", "global_random_bounds_padding_m", pf_defaults.global_random_bounds_padding_m))
    parser.add_argument("--pf-endpoint-constraint-iterations", type=int, default=config_value(config, "particle_filter", "endpoint_constraint_iterations", pf_defaults.endpoint_constraint_iterations))
    parser.add_argument("--pf-endpoint-constraint-tolerance", type=float, default=config_value(config, "particle_filter", "endpoint_constraint_tolerance_m", pf_defaults.endpoint_constraint_tolerance_m))
    parser.add_argument("--pf-min-measurement-points", type=int, default=config_value(config, "particle_filter", "min_measurement_points", pf_defaults.min_measurement_points))
    parser.add_argument("--pf-min-segment-points", type=int, default=config_value(config, "particle_filter", "min_segment_points", pf_defaults.min_segment_points))
    parser.add_argument("--pf-occlusion-gate", type=float, default=config_value(config, "particle_filter", "occlusion_gate_m", pf_defaults.occlusion_assignment_max_distance_m))
    parser.add_argument("--pf-outlier-distance", type=float, default=config_value(config, "particle_filter", "outlier_distance_m", pf_defaults.outlier_distance_m))
    parser.add_argument("--pf-resample-effective-ratio", type=float, default=config_value(config, "particle_filter", "resample_effective_ratio", pf_defaults.resample_effective_ratio))
    parser.add_argument("--pf-max-prediction-frames", type=int, default=config_value(config, "particle_filter", "max_prediction_frames", pf_defaults.max_prediction_frames))
    parser.add_argument("--pf-max-motion-noise-scale", type=float, default=config_value(config, "particle_filter", "max_motion_noise_scale", pf_defaults.max_motion_noise_scale))
    args = parser.parse_args()
    if args.input_svo_file and args.ip_address:
        raise ValueError("Specify only one input source: --input-svo-file or --ip-address.")
    args.cable_segments = max(1, int(args.cable_segments))
    args.cable_count = max(1, int(args.cable_count))
    args.cable_lengths_m = required_per_cable_float_values(
        args.cable_lengths,
        args.cable_count,
        "cable.lengths_m",
        minimum=0.0,
    )
    args.cloud_update_every = max(1, int(args.cloud_update_every))
    args.confidence_update_every = max(0, int(args.confidence_update_every))
    args.detector_scale = float(np.clip(args.detector_scale, 0.10, 1.0))
    args.detector_update_every = max(1, int(args.detector_update_every))
    args.endpoint_marker_min_area = max(1, int(args.endpoint_marker_min_area))
    args.endpoint_marker_min_points = max(1, int(args.endpoint_marker_min_points))
    args.endpoint_marker_open_kernel = max(0, int(args.endpoint_marker_open_kernel))
    args.endpoint_marker_close_kernel = max(0, int(args.endpoint_marker_close_kernel))
    args.endpoint_marker_points = max(1, int(args.endpoint_marker_points))
    args.endpoint_marker_max_count = max(1, int(args.endpoint_marker_max_count))
    if args.cable_count > 1:
        args.endpoint_marker_max_count = max(args.endpoint_marker_max_count, 2 * args.cable_count)
    args.endpoint_marker_tape_lengths_m = required_per_cable_float_values(
        args.endpoint_marker_tape_lengths,
        args.cable_count,
        "endpoint_markers.tape_lengths_m",
        minimum=0.0,
    )
    args.measurement_smoothing_alpha = float(np.clip(args.measurement_smoothing_alpha, 0.0, 1.0))
    args.measurement_smoothing_gate = max(0.0, float(args.measurement_smoothing_gate))
    args.measurement_gate_reacquire_after = max(0, int(args.measurement_gate_reacquire_after))
    args.derived_segment_lengths_m = [
        length_m / max(args.cable_segments, 1)
        for length_m in args.cable_lengths_m
    ]
    args.pf_score_chunk_points = max(1, int(args.pf_score_chunk_points))
    args.pf_velocity_damping = float(np.clip(args.pf_velocity_damping, 0.0, 1.0))
    args.pf_velocity_measurement_blend = float(np.clip(args.pf_velocity_measurement_blend, 0.0, 1.0))
    args.pf_velocity_process_std = max(0.0, float(args.pf_velocity_process_std))
    args.pf_max_node_speed = max(0.0, float(args.pf_max_node_speed))
    args.pf_endpoint_refresh_interval = max(0, int(args.pf_endpoint_refresh_interval))
    args.pf_reference_ordering_gate = max(0.0, float(args.pf_reference_ordering_gate))
    args.pf_measurement_fit_interval = max(1, int(args.pf_measurement_fit_interval))
    args.pf_ransac_hypotheses = max(0, int(args.pf_ransac_hypotheses))
    args.pf_ransac_subset_points = max(1, int(args.pf_ransac_subset_points))
    args.pf_ransac_inlier_distance = max(1e-4, float(args.pf_ransac_inlier_distance))
    args.pf_ransac_min_points = max(2, int(args.pf_ransac_min_points))
    args.pf_coarse_score_points = max(0, int(args.pf_coarse_score_points))
    args.pf_coarse_score_full_fraction = float(np.clip(args.pf_coarse_score_full_fraction, 0.0, 1.0))
    args.pf_coarse_score_min_particles = max(1, int(args.pf_coarse_score_min_particles))
    args.pf_top_particles = max(1, int(args.pf_top_particles))
    args.pf_global_random_ratio = float(np.clip(args.pf_global_random_ratio, 0.0, 1.0))
    args.pf_global_random_effective_ratio = float(np.clip(args.pf_global_random_effective_ratio, 0.0, 1.0))
    args.pf_global_random_bounds_padding = max(0.0, float(args.pf_global_random_bounds_padding))
    args.pf_endpoint_constraint_iterations = max(1, int(args.pf_endpoint_constraint_iterations))
    args.pf_endpoint_constraint_tolerance = max(0.0, float(args.pf_endpoint_constraint_tolerance))
    return args


def open_zed(args):
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[args.resolution]
    init.camera_fps = args.fps
    init.depth_mode = DEPTH_MODES[args.depth_mode]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_minimum_distance = args.depth_min
    init.depth_maximum_distance = args.depth_max
    configure_input_source(init, args)

    zed = sl.Camera()
    status = zed.open(init)
    print("Open status:", status)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")
    return zed


def make_runtime_parameters(args):
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = args.confidence
    runtime.texture_confidence_threshold = args.texture_confidence
    runtime.remove_saturated_areas = False
    if hasattr(runtime, "enable_fill_mode"):
        runtime.enable_fill_mode = args.depth_fill
    return runtime


def load_cable_detector(args):
    checkpoint_path = Path(args.neural_detector_checkpoint)
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"No PIDNet checkpoint found at {checkpoint_path}. "
            "Train a cable+endpoint checkpoint before running live tracking."
        )

    from cable_pidnet import PidNetCableDetector

    detector = PidNetCableDetector(
        checkpoint_path,
        device=args.neural_detector_device,
        threshold=float(args.neural_detector_threshold),
        base_channels=int(args.neural_detector_base_channels),
        min_area=int(args.detector_min_area),
        open_kernel=int(args.detector_open_kernel),
        close_kernel=int(args.detector_close_kernel),
        skeleton_prune_px=0,
        skeleton_prune_passes=0,
        centerline_smooth_window=1,
        amp=bool(args.neural_detector_amp),
        channels_last=bool(args.neural_detector_channels_last),
    )
    if not detector.can_use_instance_channels(int(args.cable_count), force=False):
        raise RuntimeError(
            "Live tracking requires one generic cable channel plus one endpoint channel per cable. "
            f"checkpoint outputs={getattr(detector, 'output_channels', '?')} "
            f"label_mode={getattr(detector, 'label_mode', '?')} cable_count={int(args.cable_count)}"
        )
    if bool(getattr(detector, "requires_endpoint_prompt", False)):
        raise RuntimeError("Endpoint-prompt PIDNet checkpoints are not supported by the cleaned live tracker.")
    print(
        f"Loaded PIDNet cable detector: {checkpoint_path} on {args.neural_detector_device} "
        f"(amp={bool(args.neural_detector_amp)} channels_last={bool(args.neural_detector_channels_last)} "
        f"outputs={getattr(detector, 'output_channels', '?')} labels={getattr(detector, 'label_mode', 'binary')})"
    )
    return detector


def should_use_pidnet_instance_channels(args, detector, cable_count):
    available = bool(
        hasattr(detector, "can_use_instance_channels")
        and detector.can_use_instance_channels(int(cable_count), force=False)
    )
    if not available:
        raise RuntimeError(
            f"PIDNet checkpoint does not provide the required cable/endpoints channels for {int(cable_count)} cables."
        )
    return True


def detector_description(args):
    return f"PIDNet mask threshold {float(args.neural_detector_threshold):.2f}"


def detect_endpoint_markers(
    cable_detector,
    bgr,
    point_cloud,
    args,
    confidence_measure=None,
    reference_nodes=None,
    endpoint_mask=None,
    max_markers=None,
    cable_index=0,
):
    if endpoint_mask is None:
        raise RuntimeError(
            "Internal error: neural endpoint detection requires the per-cable endpoint mask "
            "returned by PIDNet detect_instance_channel_masks()."
        )
    endpoint_mask = cleanup_marker_mask(
        endpoint_mask,
        open_kernel=args.endpoint_marker_open_kernel,
        close_kernel=args.endpoint_marker_close_kernel,
    )
    return endpoint_markers_from_mask(
        endpoint_mask,
        point_cloud,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        confidence_map=confidence_measure,
        max_confidence=args.cable_confidence_max if args.cable_confidence_max >= 0.0 else None,
        reference_nodes=reference_nodes,
        min_area_px=args.endpoint_marker_min_area,
        min_points_per_marker=args.endpoint_marker_min_points,
        max_points_per_marker=args.endpoint_marker_points,
        tape_length_m=endpoint_marker_tape_length_m(args, cable_index),
        offset_to_tips=bool(args.endpoint_marker_offset_to_tips),
        max_markers=args.endpoint_marker_max_count if max_markers is None else max(1, int(max_markers)),
    )


def anchor_measurement_to_endpoint_markers(measurement):
    if measurement is None:
        return None
    endpoint_nodes = np.asarray(getattr(measurement, "endpoint_nodes", None), dtype=np.float32)
    if endpoint_nodes.ndim != 2 or endpoint_nodes.shape[0] < 2 or endpoint_nodes.shape[1] < 3:
        return measurement
    nodes = np.asarray(getattr(measurement, "points_xyz", None), dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[0] < 2 or nodes.shape[1] < 3:
        return measurement
    if not endpoint_pair_reachable_for_nodes(endpoint_nodes, nodes):
        return measurement
    anchored = nodes.copy()
    if np.all(np.isfinite(endpoint_nodes[0, :3])):
        anchored[0, :3] = endpoint_nodes[0, :3]
    if np.all(np.isfinite(endpoint_nodes[-1, :3])):
        anchored[-1, :3] = endpoint_nodes[-1, :3]
    residual = polyline_residual(getattr(measurement, "source_points", np.empty((0, 3), dtype=np.float32)), anchored)
    return replace(measurement, points_xyz=np.ascontiguousarray(anchored, dtype=np.float32), residual_m=float(residual))


def anchor_estimate_to_known_endpoints(estimate, endpoint_source):
    if estimate is None or endpoint_source is None:
        return estimate
    endpoint_nodes = np.asarray(getattr(endpoint_source, "endpoint_nodes", None), dtype=np.float32)
    if endpoint_nodes.ndim != 2 or endpoint_nodes.shape[0] < 2 or endpoint_nodes.shape[1] < 3:
        return estimate
    nodes = np.asarray(getattr(estimate, "points_xyz", None), dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[0] < 2 or nodes.shape[1] < 3:
        return estimate
    if not endpoint_pair_reachable_for_nodes(endpoint_nodes, nodes):
        return estimate

    anchored = nodes.copy()
    if np.all(np.isfinite(endpoint_nodes[0, :3])):
        anchored[0, :3] = endpoint_nodes[0, :3]
    if np.all(np.isfinite(endpoint_nodes[-1, :3])):
        anchored[-1, :3] = endpoint_nodes[-1, :3]

    source_points = np.asarray(getattr(estimate, "source_points", np.empty((0, 3))), dtype=np.float32)
    residual = polyline_residual(source_points, anchored) if len(source_points) else float(getattr(estimate, "residual_m", 0.0))
    return replace(
        estimate,
        points_xyz=np.ascontiguousarray(anchored, dtype=np.float32),
        residual_m=float(residual),
        endpoint_nodes=np.ascontiguousarray(endpoint_nodes, dtype=np.float32),
        endpoint_marker_centers_xyz=getattr(endpoint_source, "endpoint_marker_centers_xyz", None),
        endpoint_marker_centers_xy=getattr(endpoint_source, "endpoint_marker_centers_xy", None),
        endpoint_marker_mask=getattr(endpoint_source, "endpoint_marker_mask", None),
        endpoint_marker_count=int(getattr(endpoint_source, "endpoint_marker_count", 0)),
    )


def endpoint_pair_reachable_for_nodes(endpoint_nodes, nodes, tolerance_m=0.005):
    endpoints = np.asarray(endpoint_nodes, dtype=np.float32)
    nodes = np.asarray(nodes, dtype=np.float32)
    if endpoints.ndim != 2 or endpoints.shape[0] < 2 or endpoints.shape[1] < 3:
        return False
    if nodes.ndim != 2 or nodes.shape[0] < 2 or nodes.shape[1] < 3:
        return False
    start = endpoints[0, :3]
    end = endpoints[-1, :3]
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
        return False
    finite_nodes = nodes[:, :3]
    finite = np.all(np.isfinite(finite_nodes), axis=1)
    if int(np.count_nonzero(finite)) < 2:
        return False
    endpoint_distance = float(np.linalg.norm(end - start))
    total_length = 0.0
    previous = None
    for point, is_finite in zip(finite_nodes, finite):
        if not is_finite:
            previous = None
            continue
        if previous is not None:
            total_length += float(np.linalg.norm(point - previous))
        previous = point
    return bool(endpoint_distance <= total_length + max(0.0, float(tolerance_m)))


def endpoint_anchor_diagnostics(markers, args, cable_index=0):
    diagnostics = {
        "endpoint_marker_count": 0,
        "endpoint_required_count": 2,
        "endpoint_distance_m": np.nan,
        "endpoint_model_length_m": np.nan,
        "endpoint_fixed": False,
        "endpoint_unreachable": False,
    }
    if markers is None:
        return diagnostics

    diagnostics["endpoint_marker_count"] = int(getattr(markers, "component_count", 0) or 0)
    segment_length = cable_segment_length_m(args, cable_index)
    total_length = segment_length * max(1, int(getattr(args, "cable_segments", 1)))
    if np.isfinite(total_length) and total_length > 0.0:
        diagnostics["endpoint_model_length_m"] = float(total_length)

    endpoint_nodes = np.asarray(getattr(markers, "endpoint_nodes", None), dtype=np.float32)
    if endpoint_nodes.ndim != 2 or endpoint_nodes.shape[0] < 2 or endpoint_nodes.shape[1] < 3:
        return diagnostics
    start = endpoint_nodes[0, :3]
    end = endpoint_nodes[-1, :3]
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
        return diagnostics

    endpoint_distance = float(np.linalg.norm(end - start))
    diagnostics["endpoint_distance_m"] = endpoint_distance
    if total_length > 0.0 and np.isfinite(total_length):
        tolerance = max(0.0, float(getattr(args, "pf_endpoint_constraint_tolerance", 0.0)))
        reachable = bool(endpoint_distance <= total_length + tolerance)
        diagnostics["endpoint_fixed"] = bool(diagnostics["endpoint_marker_count"] >= 2 and reachable)
        diagnostics["endpoint_unreachable"] = bool(diagnostics["endpoint_marker_count"] >= 2 and not reachable)
    return diagnostics


def make_particle_filter_config(args, cable_index=0):
    return CableParticleFilterConfig(
        particle_count=int(args.pf_particles),
        segment_length_m=cable_segment_length_m(args, cable_index),
        initial_node_std_m=float(args.pf_initial_node_std),
        initial_direction_std=float(args.pf_initial_direction_std),
        process_node_std_m=float(args.pf_process_std),
        process_direction_std=float(args.pf_process_direction_std),
        velocity_enabled=bool(args.pf_velocity),
        velocity_damping=float(args.pf_velocity_damping),
        velocity_measurement_blend=float(args.pf_velocity_measurement_blend),
        velocity_process_std_mps=float(args.pf_velocity_process_std),
        max_node_speed_mps=float(args.pf_max_node_speed),
        direction_smooth_passes=int(args.pf_direction_smooth_passes),
        measurement_node_std_m=float(args.pf_measurement_std),
        measurement_max_points=int(args.pf_measurement_points),
        scoring_backend=str(args.pf_scoring_backend),
        score_chunk_points=int(args.pf_score_chunk_points),
        endpoint_ordering=bool(args.pf_endpoint_ordering),
        ordering_max_points=int(args.pf_ordering_max_points),
        ordering_knn=int(args.pf_ordering_knn),
        endpoint_refresh_interval=int(args.pf_endpoint_refresh_interval),
        reference_ordering=bool(args.pf_reference_ordering),
        reference_ordering_gate_m=float(args.pf_reference_ordering_gate),
        measurement_fit_interval=int(args.pf_measurement_fit_interval),
        endpoint_penalty_weight=float(args.pf_endpoint_penalty_weight),
        measurement_reset_error_m=float(args.pf_measurement_reset_error),
        measurement_proposal_ratio=float(args.pf_measurement_proposal_ratio),
        measurement_proposal_stable_ratio=float(args.pf_measurement_proposal_stable_ratio),
        measurement_proposal_start_error_m=float(args.pf_measurement_proposal_start_error),
        measurement_proposal_full_error_m=float(args.pf_measurement_proposal_full_error),
        measurement_proposal_node_std_m=float(args.pf_measurement_proposal_std),
        measurement_proposal_direction_std=float(args.pf_measurement_proposal_direction_std),
        ransac_inlier_selection_enabled=bool(args.pf_ransac_inlier_selection),
        ransac_hypotheses=int(args.pf_ransac_hypotheses),
        ransac_subset_points=int(args.pf_ransac_subset_points),
        ransac_inlier_distance_m=float(args.pf_ransac_inlier_distance),
        ransac_min_points=int(args.pf_ransac_min_points),
        score_keep_fraction=float(args.pf_score_keep_fraction),
        coverage_penalty_m=float(args.pf_coverage_penalty),
        coverage_min_fraction=float(args.pf_coverage_min_fraction),
        bend_penalty_m=float(args.pf_bend_penalty),
        coarse_score_points=int(args.pf_coarse_score_points),
        coarse_score_full_fraction=float(args.pf_coarse_score_full_fraction),
        coarse_score_min_particles=int(args.pf_coarse_score_min_particles),
        top_particle_count=int(args.pf_top_particles),
        global_random_particle_ratio=float(args.pf_global_random_ratio),
        global_random_effective_ratio=float(args.pf_global_random_effective_ratio),
        global_random_bounds_padding_m=float(args.pf_global_random_bounds_padding),
        endpoint_constraint_iterations=int(args.pf_endpoint_constraint_iterations),
        endpoint_constraint_tolerance_m=float(args.pf_endpoint_constraint_tolerance),
        min_measurement_points=int(args.pf_min_measurement_points),
        min_segment_points=int(args.pf_min_segment_points),
        occlusion_assignment_max_distance_m=float(args.pf_occlusion_gate),
        outlier_distance_m=float(args.pf_outlier_distance),
        resample_effective_ratio=float(args.pf_resample_effective_ratio),
        max_prediction_frames=int(args.pf_max_prediction_frames),
        max_motion_noise_scale=float(args.pf_max_motion_noise_scale),
    )


def main():
    args = parse_args()
    run_live(args)


@dataclass
class AsyncTrackingFrame:
    frame_index: int
    timestamp_s: float
    bgr: np.ndarray
    point_cloud: np.ndarray
    confidence_measure: np.ndarray | None


@dataclass
class AsyncTrackingResult:
    frame_index: int
    frame_timestamp_s: float
    detection: object | None
    endpoint_mask: np.ndarray | None
    detection_label_mask: np.ndarray | None
    endpoint_label_mask: np.ndarray | None
    measurement: object | None
    estimate: object | None
    filter_result: object | None
    tracking_diagnostics: dict
    worker_stage_seconds: dict
    worker_seconds: float
    last_filter_lost_frames: int


def label_mask_from_binary_masks(masks, output_shape, first_label=1):
    output_h, output_w = int(output_shape[0]), int(output_shape[1])
    label_mask = np.zeros((output_h, output_w), dtype=np.uint8)
    for index, mask in enumerate(masks or []):
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim != 2 or not np.any(mask):
            continue
        if mask.shape[:2] != (output_h, output_w):
            mask = cv2.resize(mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
        label_mask[mask > 0] = int(first_label) + int(index)
    return None if not np.any(label_mask) else np.ascontiguousarray(label_mask, dtype=np.uint8)


def label_mask_from_detections(detections, output_shape, first_label=1):
    masks = []
    for detection in detections or []:
        masks.append(None if detection is None else getattr(detection, "mask", None))
    return label_mask_from_binary_masks(masks, output_shape, first_label=first_label)


class AsyncTrackingWorker:
    def __init__(self, args, cable_detector, particle_filters):
        self.args = args
        self.cable_detector = cable_detector
        if particle_filters is None:
            particle_filters = []
        elif isinstance(particle_filters, CableParticleFilter):
            particle_filters = [particle_filters]
        self.particle_filters = list(particle_filters)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cable-tracking")
        self.future = None
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.busy_frames = 0
        self.last_smoothed_measurement_nodes = None
        self.last_smoothed_measurement_nodes_by_cable = [None for _ in range(max(1, int(args.cable_count)))]
        self.last_detection_hint_xy = None
        self.last_filter_nodes = None
        self.last_filter_nodes_by_cable = [None for _ in range(max(1, int(args.cable_count)))]
        self.last_filter_lost_frames = 0
        self.last_filter_lost_frames_by_cable = [0 for _ in range(max(1, int(args.cable_count)))]
        self.last_filter_time = None

    def is_busy(self):
        return self.future is not None and not self.future.done()

    def can_submit(self):
        return self.future is None or self.future.done()

    def submit(self, frame):
        if self.is_busy():
            self.dropped += 1
            return False
        self.future = self.executor.submit(self._process_frame, frame)
        self.submitted += 1
        return True

    def drain_latest(self):
        if self.future is None or not self.future.done():
            return None
        future = self.future
        self.future = None
        try:
            result = future.result()
        except Exception as exc:
            formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            raise RuntimeError(f"Async tracking worker failed:\n{formatted}") from exc
        self.completed += 1
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _process_frame(self, frame):
        worker_start = time.monotonic()
        stage_seconds = {
            "detect": 0.0,
            "fit": 0.0,
            "filter": 0.0,
        }
        detection = None
        endpoint_mask = None
        detection_label_mask = None
        endpoint_label_mask = None
        measurement = None
        raw_measurement = None
        estimate = None
        filter_result = None
        smoothing_diagnostics = {}
        instance_detections = None
        instance_endpoint_masks = None
        args = self.args

        stage_start = time.monotonic()
        should_use_pidnet_instance_channels(args, self.cable_detector, args.cable_count)
        if not hasattr(self.cable_detector, "detect_instance_channel_masks"):
            raise RuntimeError("Loaded detector does not implement detect_instance_channel_masks().")
        instance_detections, detection, endpoint_mask, instance_endpoint_masks = self.cable_detector.detect_instance_channel_masks(
            frame.bgr,
            cable_count=args.cable_count,
            scale=args.detector_scale,
            extract_geometry=False,
            include_endpoint_mask=True,
            force=False,
        )
        cable_count = max(1, int(args.cable_count))
        if len(instance_detections or []) < cable_count or detection is None:
            raise RuntimeError("PIDNet did not return the required cable mask channels.")
        if len(instance_endpoint_masks or []) < cable_count or any(mask is None for mask in instance_endpoint_masks[:cable_count]):
            raise RuntimeError("PIDNet did not return one endpoint mask per cable.")
        detection_label_mask = label_mask_from_detections(
            list(instance_detections)[:cable_count],
            frame.bgr.shape[:2],
            first_label=1,
        )
        endpoint_label_mask = label_mask_from_binary_masks(
            list(instance_endpoint_masks)[:cable_count],
            frame.bgr.shape[:2],
            first_label=cable_count + 1,
        )
        stage_seconds["detect"] += time.monotonic() - stage_start

        return self._process_frame_multi(
            frame,
            detection,
            endpoint_mask,
            detection_label_mask,
            endpoint_label_mask,
            stage_seconds,
            worker_start,
            instance_detections=instance_detections,
            instance_endpoint_masks=instance_endpoint_masks,
        )

    def _process_frame_multi(
        self,
        frame,
        detection,
        endpoint_mask,
        detection_label_mask,
        endpoint_label_mask,
        stage_seconds,
        worker_start,
        instance_detections=None,
        instance_endpoint_masks=None,
    ):
        args = self.args
        cable_count = max(1, int(args.cable_count))
        instance_detections = list(instance_detections or [])
        instance_endpoint_masks = list(instance_endpoint_masks or [])
        if len(instance_detections) < cable_count:
            raise RuntimeError(f"Expected {cable_count} cable detections, got {len(instance_detections)}.")
        if len(instance_endpoint_masks) < cable_count or any(mask is None for mask in instance_endpoint_masks[:cable_count]):
            raise RuntimeError(f"Expected {cable_count} endpoint masks, got {len(instance_endpoint_masks)}.")
        measurements = [None for _ in range(cable_count)]
        raw_measurements = [None for _ in range(cable_count)]
        estimates = [None for _ in range(cable_count)]
        filter_results = [None for _ in range(cable_count)]
        diagnostics = [{} for _ in range(cable_count)]

        stage_start = time.monotonic()
        for cable_index in range(cable_count):
            instance_detection = instance_detections[cable_index]
            cable_endpoint_mask = instance_endpoint_masks[cable_index]
            previous_nodes = self.last_filter_nodes_by_cable[cable_index]
            lost_frames = self.last_filter_lost_frames_by_cable[cable_index]
            reference_nodes = measurement_reference_nodes(
                previous_nodes,
                lost_frames,
                reacquire_after=args.measurement_gate_reacquire_after,
            )
            candidate_markers = detect_endpoint_markers(
                self.cable_detector,
                frame.bgr,
                frame.point_cloud,
                args,
                confidence_measure=frame.confidence_measure,
                reference_nodes=reference_nodes,
                endpoint_mask=cable_endpoint_mask,
                max_markers=2,
                cable_index=cable_index,
            )
            gated_reference_nodes = endpoint_aligned_reference_nodes(reference_nodes, candidate_markers)
            reference_gate = measurement_reference_gate(
                args.measurement_prediction_gate,
                lost_frames,
            )
            prior_diag = {
                "prior_gate_m": float(reference_gate) if gated_reference_nodes is not None else np.nan,
                "prior_active": bool(gated_reference_nodes is not None and reference_gate > 0.0),
            }
            measurement = self._measurement_from_detection(
                frame,
                instance_detection,
                gated_reference_nodes,
                reference_gate,
            )
            endpoint_diag = endpoint_anchor_diagnostics(candidate_markers, args, cable_index=cable_index)
            if not bool(endpoint_diag.get("endpoint_fixed", False)):
                measurement = None
            measurement = attach_endpoint_markers_to_measurement(measurement, candidate_markers)
            measurement = anchor_measurement_to_endpoint_markers(measurement)
            raw_measurements[cable_index] = measurement
            measurement, self.last_smoothed_measurement_nodes_by_cable[cable_index], smooth_diag = smooth_measurement_nodes(
                measurement,
                self.last_smoothed_measurement_nodes_by_cable[cable_index],
                args,
                lost_frames,
            )
            measurement = anchor_measurement_to_endpoint_markers(measurement)
            if measurement is not None:
                self.last_smoothed_measurement_nodes_by_cable[cable_index] = measurement_nodes_array(measurement)
            measurements[cable_index] = measurement
            smooth_diag.update(prior_diag)
            smooth_diag.update(endpoint_diag)
            diagnostics[cable_index] = smooth_diag
        stage_seconds["fit"] += time.monotonic() - stage_start

        stage_start = time.monotonic()
        filter_dt = (
            1.0 / max(float(args.fps), 1.0)
            if self.last_filter_time is None
            else max(1e-3, float(frame.timestamp_s - self.last_filter_time))
        )
        self.last_filter_time = frame.timestamp_s

        for cable_index in range(cable_count):
            measurement = measurements[cable_index]
            particle_filter = self.particle_filters[cable_index] if cable_index < len(self.particle_filters) else None
            if particle_filter is not None:
                filter_result = particle_filter.step(measurement, filter_dt)
                estimate = filtered_cable_estimate(measurement, filter_result)
                estimate = anchor_estimate_to_known_endpoints(estimate, measurement)
            else:
                filter_result = None
                estimate = measurement
            filter_results[cable_index] = filter_result
            estimates[cable_index] = estimate

            diagnostics[cable_index] = cable_tracking_diagnostics(
                self.last_filter_nodes_by_cable[cable_index],
                raw_measurements[cable_index],
                measurement,
                estimate,
                filter_result,
                diagnostics[cable_index],
            )
            if estimate is not None:
                self.last_filter_nodes_by_cable[cable_index] = np.asarray(estimate.points_xyz, dtype=np.float32)
            elif particle_filter is None:
                self.last_filter_nodes_by_cable[cable_index] = None
            if filter_result is not None:
                self.last_filter_lost_frames_by_cable[cable_index] = int(filter_result.lost_frames)
            elif estimate is not None:
                self.last_filter_lost_frames_by_cable[cable_index] = 0

        stage_seconds["filter"] += time.monotonic() - stage_start

        measurement = combine_cable_estimates(measurements, method="multi-cable measurement")
        raw_measurement = combine_cable_estimates(raw_measurements, method="multi-cable raw measurement")
        estimate = combine_cable_estimates(estimates, method="multi-cable estimate")
        filter_result = combine_filter_results(filter_results)
        tracking_diagnostics = combine_tracking_diagnostics(
            diagnostics,
            candidate_count=cable_count,
            cable_count=cable_count,
        )
        tracking_diagnostics["association"] = "pidnet_channels"

        self.last_filter_nodes = None if estimate is None else np.asarray(estimate.points_xyz, dtype=np.float32)
        self.last_filter_lost_frames = max(self.last_filter_lost_frames_by_cable) if self.last_filter_lost_frames_by_cable else 0
        self.last_smoothed_measurement_nodes = None if measurement is None else measurement_nodes_array(measurement)

        worker_done = time.monotonic()
        return AsyncTrackingResult(
            frame_index=int(frame.frame_index),
            frame_timestamp_s=float(frame.timestamp_s),
            detection=detection,
            endpoint_mask=endpoint_mask,
            detection_label_mask=detection_label_mask,
            endpoint_label_mask=endpoint_label_mask,
            measurement=measurement,
            estimate=estimate,
            filter_result=filter_result,
            tracking_diagnostics=tracking_diagnostics,
            worker_stage_seconds=stage_seconds,
            worker_seconds=float(worker_done - worker_start),
            last_filter_lost_frames=int(self.last_filter_lost_frames),
        )

    def _measurement_from_detection(
        self,
        frame,
        detection,
        reference_nodes,
        reference_gate,
    ):
        args = self.args
        if detection is None:
            return None
        return cable_measurement_from_mask_points(
            frame.point_cloud,
            detection,
            segment_count=args.cable_segments,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            confidence_map=frame.confidence_measure,
            max_confidence=args.cable_confidence_max if args.cable_confidence_max >= 0.0 else None,
            max_points=args.pf_measurement_points,
            reference_nodes=reference_nodes,
            reference_gate_m=reference_gate,
            reference_min_points=args.pf_min_measurement_points,
        )


def combine_cable_estimates(estimates, method="multi-cable"):
    valid_estimates = [estimate for estimate in estimates if estimate is not None]
    if not valid_estimates:
        return None
    nodes = concatenate_node_chains([getattr(estimate, "points_xyz", None) for estimate in valid_estimates])
    source_points = concatenate_point_sets([getattr(estimate, "source_points", None) for estimate in valid_estimates])
    residuals = [float(getattr(estimate, "residual_m", np.nan)) for estimate in valid_estimates]
    finite_residuals = [value for value in residuals if np.isfinite(value)]
    residual = float(np.mean(finite_residuals)) if finite_residuals else 0.0
    centers_xyz = concatenate_point_sets([getattr(estimate, "endpoint_marker_centers_xyz", None) for estimate in valid_estimates])
    centers_xy = concatenate_xy_sets([getattr(estimate, "endpoint_marker_centers_xy", None) for estimate in valid_estimates])
    endpoint_nodes = concatenate_node_chains([getattr(estimate, "endpoint_nodes", None) for estimate in valid_estimates])
    return CableEstimate3D(
        points_xyz=nodes,
        source_points=source_points,
        residual_m=residual,
        method=f"{method} | active={len(valid_estimates)}",
        endpoint_nodes=endpoint_nodes,
        endpoint_marker_centers_xyz=centers_xyz if len(centers_xyz) else None,
        endpoint_marker_centers_xy=centers_xy if len(centers_xy) else None,
        endpoint_marker_mask=None,
        endpoint_marker_count=int(len(centers_xy)),
    )


def concatenate_node_chains(chains):
    output = []
    for chain in chains:
        points = np.asarray(chain, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            continue
        if output:
            output.append(np.full((1, 3), np.nan, dtype=np.float32))
        output.append(np.ascontiguousarray(points[:, :3], dtype=np.float32))
    if not output:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


def concatenate_point_sets(point_sets):
    output = []
    for point_set in point_sets:
        points = np.asarray(point_set, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] >= 3 and len(points):
            output.append(points[:, :3])
    if not output:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


def concatenate_xy_sets(point_sets):
    output = []
    for point_set in point_sets:
        points = np.asarray(point_set, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] >= 2 and len(points):
            output.append(points[:, :2])
    if not output:
        return np.empty((0, 2), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32)


def combine_filter_results(filter_results):
    valid_results = [result for result in filter_results if result is not None]
    if not valid_results:
        return None
    ransac_error_values = [
        float(getattr(result, "ransac_error_m", np.nan))
        for result in valid_results
        if np.isfinite(float(getattr(result, "ransac_error_m", np.nan)))
    ]
    visible_nodes = concatenate_bool_chains([getattr(result, "visible_nodes", None) for result in valid_results])
    extended_nodes = concatenate_bool_chains([
        np.ones_like(np.asarray(getattr(result, "visible_nodes", []), dtype=bool))
        for result in valid_results
    ])
    visible_segments = np.concatenate([
        np.asarray(getattr(result, "visible_segments", np.empty(0, dtype=bool)), dtype=bool).reshape(-1)
        for result in valid_results
    ])
    stage_seconds = {}
    for result in valid_results:
        for key, value in dict(getattr(result, "stage_seconds", {}) or {}).items():
            stage_seconds[key] = stage_seconds.get(key, 0.0) + float(value)
    return SimpleNamespace(
        visible_nodes=visible_nodes,
        extended_visible_nodes=extended_nodes,
        visible_segments=visible_segments,
        measurement_used=any(bool(getattr(result, "measurement_used", False)) for result in valid_results),
        prediction_only=all(bool(getattr(result, "prediction_only", False)) for result in valid_results),
        lost_frames=max(int(getattr(result, "lost_frames", 0)) for result in valid_results),
        measurement_point_count=sum(int(getattr(result, "measurement_point_count", 0)) for result in valid_results),
        segment_length_m=float(np.nanmean([float(getattr(result, "segment_length_m", np.nan)) for result in valid_results])),
        measurement_proposal_ratio=float(np.nanmean([float(getattr(result, "measurement_proposal_ratio", np.nan)) for result in valid_results])),
        global_random_particle_ratio=float(np.nanmean([float(getattr(result, "global_random_particle_ratio", np.nan)) for result in valid_results])),
        ransac_inlier_ratio=float(np.nanmean([float(getattr(result, "ransac_inlier_ratio", np.nan)) for result in valid_results])),
        ransac_inlier_count=sum(int(getattr(result, "ransac_inlier_count", 0)) for result in valid_results),
        ransac_error_m=float(np.mean(ransac_error_values)) if ransac_error_values else np.nan,
        ransac_hypothesis_count=sum(int(getattr(result, "ransac_hypothesis_count", 0)) for result in valid_results),
        coarse_score_point_count=sum(int(getattr(result, "coarse_score_point_count", 0)) for result in valid_results),
        full_score_particle_count=sum(int(getattr(result, "full_score_particle_count", 0)) for result in valid_results),
        mean_node_speed_mps=float(np.nanmean([float(getattr(result, "mean_node_speed_mps", np.nan)) for result in valid_results])),
        stage_seconds=stage_seconds,
    )


def concatenate_bool_chains(chains):
    output = []
    for chain in chains:
        values = np.asarray(chain, dtype=bool).reshape(-1)
        if len(values) == 0:
            continue
        if output:
            output.append(np.zeros(1, dtype=bool))
        output.append(values)
    if not output:
        return np.empty(0, dtype=bool)
    return np.concatenate(output)


def combine_tracking_diagnostics(diagnostics, candidate_count=0, cable_count=1):
    valid = [dict(item) for item in diagnostics if isinstance(item, dict)]
    combined = {
        "cable_count": int(cable_count),
        "candidate_count": int(candidate_count),
        "active_cables": int(sum(1 for item in valid if np.isfinite(float(item.get("filtered_residual_m", np.nan))))),
    }
    if int(cable_count) > 1 and valid:
        endpoint_counts = [int(item.get("endpoint_marker_count", 0) or 0) for item in valid[: int(cable_count)]]
        endpoint_fixed = [bool(item.get("endpoint_fixed", False)) for item in valid[: int(cable_count)]]
        endpoint_unreachable = [bool(item.get("endpoint_unreachable", False)) for item in valid[: int(cable_count)]]
        endpoint_distances = [
            float(item.get("endpoint_distance_m", np.nan))
            for item in valid[: int(cable_count)]
        ]
        endpoint_lengths = [
            float(item.get("endpoint_model_length_m", np.nan))
            for item in valid[: int(cable_count)]
        ]
        combined["endpoint_counts_text"] = ",".join(f"{count}/2" for count in endpoint_counts)
        combined["endpoint_fixed_count"] = int(sum(endpoint_fixed))
        combined["endpoint_unreachable_count"] = int(sum(endpoint_unreachable))
        finite_distances = [value for value in endpoint_distances if np.isfinite(value)]
        if finite_distances:
            combined["endpoint_distance_text"] = ",".join(
                format_mm(value) for value in endpoint_distances if np.isfinite(value)
            )
        finite_lengths = [value for value in endpoint_lengths if np.isfinite(value)]
        if finite_lengths:
            combined["endpoint_model_length_text"] = ",".join(
                format_mm(value) for value in endpoint_lengths if np.isfinite(value)
            )
    for key in (
        "raw_to_filter_m",
        "smooth_delta_m",
        "prior_gate_m",
        "proposal_ratio",
        "global_random_ratio",
        "ransac_inlier_ratio",
        "ransac_error_m",
        "mean_node_speed_mps",
    ):
        values = [float(item.get(key, np.nan)) for item in valid]
        finite = [value for value in values if np.isfinite(value)]
        if finite:
            combined[key] = float(np.mean(finite))
    ransac_hypotheses = [int(item.get("ransac_hypotheses", 0) or 0) for item in valid]
    if ransac_hypotheses:
        combined["ransac_hypotheses"] = int(sum(ransac_hypotheses))
    ransac_inliers = [int(item.get("ransac_inlier_count", 0) or 0) for item in valid]
    if ransac_inliers:
        combined["ransac_inlier_count"] = int(sum(ransac_inliers))
    prior_active = [bool(item.get("prior_active", False)) for item in valid]
    if prior_active:
        combined["prior_active_count"] = int(sum(1 for value in prior_active if value))
    pf_stage = {}
    for item in valid:
        for key, value in dict(item.get("pf_stage_ms", {}) or {}).items():
            pf_stage[key] = pf_stage.get(key, 0.0) + float(value)
    if pf_stage:
        combined["pf_stage_ms"] = pf_stage
    return combined


def should_submit_async_frame(args, frame_count, latest_result):
    if latest_result is None:
        return True
    if int(getattr(latest_result, "last_filter_lost_frames", 0)) > 0:
        return True
    update_every = max(1, int(args.detector_update_every))
    return int(frame_count) % update_every == 0


def format_runtime_status(
    frame_count,
    render_fps,
    compute_fps,
    submit_fps,
    main_ms,
    worker_ms,
    result_age_frames,
    result_age_ms,
    latest_stats,
    main_stage_ms,
    worker_stage_ms,
    async_worker,
    latest_cable_status,
):
    lag_text = "no-result" if result_age_frames < 0 else f"{result_age_frames}f/{result_age_ms:.0f}ms"
    cloud_text = f"{int(latest_stats['returned'])}/{int(latest_stats['sampled'])}"
    return (
        f"Frame {frame_count} | GUI {render_fps:.1f}Hz | TRACK {compute_fps:.1f}Hz "
        f"(submit {submit_fps:.1f}Hz) | lag {lag_text} | "
        f"main {main_ms:.1f}ms cap={main_stage_ms['capture']:.1f} "
        f"cloud={main_stage_ms['cloud']:.1f} copy={main_stage_ms['copy']:.1f} "
        f"ui={main_stage_ms['ui']:.1f} | "
        f"worker {worker_ms:.1f}ms det={worker_stage_ms['detect']:.1f} "
        f"fit={worker_stage_ms['fit']:.1f} pf={worker_stage_ms['filter']:.1f} | "
        f"async sub/ok/drop={async_worker.submitted}/{async_worker.completed}/{async_worker.dropped} "
        f"busy={1 if async_worker.is_busy() else 0} | cloud {cloud_text} pts | "
        f"{latest_cable_status}"
    )


def run_live(args):
    cable_detector = load_cable_detector(args)
    particle_filters = (
        [
            CableParticleFilter(node_count=args.cable_segments + 1, config=make_particle_filter_config(args, cable_index))
            for cable_index in range(args.cable_count)
        ]
        if args.particle_filter
        else []
    )
    async_worker = AsyncTrackingWorker(args, cable_detector, particle_filters)

    zed = open_zed(args)
    runtime = make_runtime_parameters(args)
    image = sl.Mat()
    point_cloud = sl.Mat()
    confidence_map = sl.Mat()

    viewer = ZedDepthGLViewer(
        args.rgb_width + args.cloud_width,
        args.height,
        "Particle Filter Cable",
        window_x=40,
        window_y=40,
        left_panel_width=args.rgb_width,
    )
    viewer.init()
    viewer.view_mode = "orbit"
    viewer.yaw_deg = -35.0
    viewer.pitch_deg = 22.0
    viewer.point_size = args.point_size
    viewer.set_depth_max(args.depth_max)
    configure_viewer_from_zed(zed, viewer)
    print(
        "Async GUI pipeline enabled: main thread handles ZED capture + OpenGL render; "
        "worker thread handles detection + 3D construction + particle filter. "
        "Stats report GUI render FPS and COMPUTE detect/track FPS separately."
    )

    frame_count = 0
    last_stats_time = time.monotonic()
    last_stats_frame_count = 0
    last_stats_completed_count = 0
    last_stats_submitted_count = 0
    stats_main_seconds = 0.0
    stats_main_frames = 0
    stats_worker_seconds = 0.0
    stats_worker_frames = 0
    stats_worker_stage_seconds = {
        "detect": 0.0,
        "fit": 0.0,
        "filter": 0.0,
    }
    latest_stats = empty_point_cloud_stats()
    latest_vertices = np.empty((0, 6), dtype=np.float32)
    latest_confidence_measure = None
    latest_detection = None
    latest_endpoint_mask = None
    latest_detection_label_mask = None
    latest_endpoint_label_mask = None
    last_visual_measurement = None
    last_visual_estimate = None
    last_visual_filter_result = None
    latest_tracking_diagnostics = {}
    latest_result = None
    latest_cable_status = "waiting for cable"
    main_stage_seconds = {
        "capture": 0.0,
        "cloud": 0.0,
        "copy": 0.0,
        "ui": 0.0,
    }

    try:
        while viewer.is_available():
            if zed.grab(runtime) <= sl.ERROR_CODE.SUCCESS:
                frame_start_time = time.monotonic()
                completed_result = async_worker.drain_latest()
                if completed_result is not None:
                    latest_result = completed_result
                    latest_detection = completed_result.detection
                    latest_endpoint_mask = completed_result.endpoint_mask
                    latest_detection_label_mask = completed_result.detection_label_mask
                    latest_endpoint_label_mask = completed_result.endpoint_label_mask
                    latest_tracking_diagnostics = completed_result.tracking_diagnostics
                    if completed_result.measurement is not None:
                        last_visual_measurement = completed_result.measurement
                    if completed_result.estimate is not None:
                        last_visual_estimate = completed_result.estimate
                        last_visual_filter_result = completed_result.filter_result
                    stats_worker_frames += 1
                    stats_worker_seconds += float(completed_result.worker_seconds)
                    for key in stats_worker_stage_seconds:
                        stats_worker_stage_seconds[key] += float(completed_result.worker_stage_seconds.get(key, 0.0))

                want_worker_submit = should_submit_async_frame(args, frame_count, latest_result)
                submit_worker_frame = bool(want_worker_submit and async_worker.can_submit())
                if want_worker_submit and async_worker.is_busy():
                    async_worker.busy_frames += 1
                    async_worker.dropped += 1

                stage_start = frame_start_time
                zed.retrieve_image(image, sl.VIEW.LEFT)
                update_cloud_view = frame_count % args.cloud_update_every == 0
                update_point_cloud = update_cloud_view or submit_worker_frame
                if update_point_cloud:
                    point_measure = sl.MEASURE.XYZRGBA if update_cloud_view else getattr(sl.MEASURE, "XYZ", sl.MEASURE.XYZRGBA)
                    zed.retrieve_measure(point_cloud, point_measure)
                if args.confidence_update_every == 0:
                    confidence_measure = None
                elif submit_worker_frame and frame_count % args.confidence_update_every == 0:
                    confidence_mat = retrieve_confidence_measure(zed, confidence_map)
                    latest_confidence_measure = (
                        None
                        if confidence_mat is None
                        else np.ascontiguousarray(np.asarray(confidence_mat.get_data()).copy())
                    )
                    confidence_measure = latest_confidence_measure
                else:
                    confidence_measure = latest_confidence_measure
                main_stage_seconds["capture"] += time.monotonic() - stage_start

                stage_start = time.monotonic()
                bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                if update_cloud_view:
                    latest_vertices, latest_stats = live_point_cloud_to_vertices(
                        point_cloud,
                        stride=args.live_stride,
                        max_points=args.live_max_points,
                        depth_min=args.depth_min,
                        depth_max=args.depth_max,
                        return_stats=True,
                    )
                main_stage_seconds["cloud"] += time.monotonic() - stage_start

                stage_start = time.monotonic()
                if submit_worker_frame:
                    point_cloud_snapshot = np.ascontiguousarray(np.asarray(point_cloud.get_data()).copy())
                    bgr_snapshot = np.ascontiguousarray(bgr.copy())
                    confidence_snapshot = (
                        None
                        if confidence_measure is None
                        else np.ascontiguousarray(np.asarray(confidence_measure).copy())
                    )
                    async_worker.submit(
                        AsyncTrackingFrame(
                            frame_index=int(frame_count),
                            timestamp_s=float(frame_start_time),
                            bgr=bgr_snapshot,
                            point_cloud=point_cloud_snapshot,
                            confidence_measure=confidence_snapshot,
                        )
                    )
                main_stage_seconds["copy"] += time.monotonic() - stage_start

                display_measurement = last_visual_measurement
                display_estimate = last_visual_estimate
                display_filter_result = last_visual_filter_result
                detection = latest_detection
                endpoint_mask = latest_endpoint_mask
                detection_label_mask = latest_detection_label_mask
                endpoint_label_mask = latest_endpoint_label_mask
                if display_measurement is None and display_estimate is not None:
                    display_measurement = last_visual_measurement

                stage_start = time.monotonic()
                debug_bgr = draw_cable_rgb_panel(
                    bgr,
                    detection=detection,
                    endpoint_mask=endpoint_mask,
                    detection_label_mask=detection_label_mask,
                    endpoint_label_mask=endpoint_label_mask,
                    measurement=display_measurement,
                    estimate=display_estimate,
                    segment_count=args.cable_segments,
                    cable_count=args.cable_count,
                    mode=args.rgb_view,
                    detector_description=detector_description(args),
                )
                update_viewer_cable(viewer, display_measurement, display_estimate, display_filter_result, args.cable_max_points)
                latest_cable_status = cable_status(
                    detection,
                    display_measurement,
                display_estimate,
                display_filter_result,
                args.cable_segments,
                detector_name="pidnet",
                diagnostics=latest_tracking_diagnostics,
            )

                frame_count += 1
                viewer.update_rgb_image(cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB))
                if update_cloud_view:
                    viewer.update_vertices(
                        latest_vertices,
                        f"live ZED point cloud | frame {frame_count} | {len(latest_vertices)} points | {latest_cable_status}",
                    )
                main_stage_seconds["ui"] += time.monotonic() - stage_start

                now = time.monotonic()
                stats_main_seconds += max(0.0, now - frame_start_time)
                stats_main_frames += 1
                if now - last_stats_time >= 1.0:
                    elapsed = max(now - last_stats_time, 1e-6)
                    render_fps = (frame_count - last_stats_frame_count) / elapsed
                    compute_fps = (async_worker.completed - last_stats_completed_count) / elapsed
                    submit_fps = (async_worker.submitted - last_stats_submitted_count) / elapsed
                    main_ms = 1000.0 * stats_main_seconds / max(stats_main_frames, 1)
                    worker_ms = 1000.0 * stats_worker_seconds / max(stats_worker_frames, 1)
                    main_stage_ms = {
                        name: 1000.0 * value / max(stats_main_frames, 1)
                        for name, value in main_stage_seconds.items()
                    }
                    worker_stage_ms = {
                        name: 1000.0 * value / max(stats_worker_frames, 1)
                        for name, value in stats_worker_stage_seconds.items()
                    }
                    result_age_frames = -1 if latest_result is None else max(0, frame_count - int(latest_result.frame_index))
                    result_age_ms = float("nan") if latest_result is None else 1000.0 * max(0.0, now - latest_result.frame_timestamp_s)
                    print(
                        format_runtime_status(
                            frame_count,
                            render_fps,
                            compute_fps,
                            submit_fps,
                            main_ms,
                            worker_ms,
                            result_age_frames,
                            result_age_ms,
                            latest_stats,
                            main_stage_ms,
                            worker_stage_ms,
                            async_worker,
                            latest_cable_status,
                        )
                    )
                    last_stats_time = now
                    last_stats_frame_count = frame_count
                    last_stats_completed_count = async_worker.completed
                    last_stats_submitted_count = async_worker.submitted
                    stats_main_seconds = 0.0
                    stats_main_frames = 0
                    stats_worker_seconds = 0.0
                    stats_worker_frames = 0
                    for key in main_stage_seconds:
                        main_stage_seconds[key] = 0.0
                    for key in stats_worker_stage_seconds:
                        stats_worker_stage_seconds[key] = 0.0

            viewer.poll()

    finally:
        async_worker.close()
        viewer.close()
        image.free()
        point_cloud.free()
        confidence_map.free()
        zed.close()


def retrieve_confidence_measure(zed, confidence_map):
    try:
        if zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE) <= sl.ERROR_CODE.SUCCESS:
            return confidence_map
    except Exception:
        return None
    return None


def measurement_reference_nodes(last_filter_nodes, lost_frames, reacquire_after=4):
    if last_filter_nodes is None:
        return None
    reacquire_after = int(reacquire_after)
    if reacquire_after > 0 and int(lost_frames) >= reacquire_after:
        return None
    nodes = np.asarray(last_filter_nodes, dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) < 2:
        return None
    if not np.all(np.isfinite(nodes[:, :3])):
        return None
    return nodes[:, :3]


def endpoint_aligned_reference_nodes(reference_nodes, endpoint_markers):
    if reference_nodes is None or endpoint_markers is None:
        return reference_nodes
    endpoint_nodes = np.asarray(getattr(endpoint_markers, "endpoint_nodes", None), dtype=np.float32)
    if endpoint_nodes.ndim != 2 or endpoint_nodes.shape[0] < 2 or endpoint_nodes.shape[1] < 3:
        return reference_nodes
    endpoint_nodes = endpoint_nodes[:2, :3]
    if not np.all(np.isfinite(endpoint_nodes)):
        return reference_nodes

    nodes = np.asarray(reference_nodes, dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[0] < 2 or nodes.shape[1] < 3:
        return reference_nodes
    nodes = nodes[:, :3]
    if not np.all(np.isfinite(nodes)):
        return reference_nodes

    start_delta = endpoint_nodes[0] - nodes[0]
    end_delta = endpoint_nodes[1] - nodes[-1]
    t = np.linspace(0.0, 1.0, len(nodes), dtype=np.float32)[:, None]
    aligned = nodes + (1.0 - t) * start_delta[None, :] + t * end_delta[None, :]
    return np.ascontiguousarray(aligned, dtype=np.float32)


def measurement_reference_gate(base_gate_m, lost_frames):
    base_gate_m = float(base_gate_m)
    if base_gate_m <= 0.0:
        return 0.0
    scale = min(3.0, 1.0 + 0.25 * max(0, int(lost_frames)))
    return base_gate_m * scale


def smooth_measurement_nodes(measurement, previous_nodes, args, lost_frames):
    diagnostics = {
        "smoothing": "off",
        "smooth_delta_m": np.nan,
        "raw_to_previous_m": np.nan,
    }
    nodes = measurement_nodes_array(measurement)
    if measurement is None or nodes is None:
        return measurement, previous_nodes, diagnostics

    diagnostics["smoothing"] = "raw"
    if not bool(args.measurement_smoothing) or int(lost_frames) > 0:
        return measurement, nodes, diagnostics

    previous = measurement_nodes_array(previous_nodes)
    if previous is None or previous.shape != nodes.shape:
        return measurement, nodes, diagnostics

    nodes = align_node_orientation(previous, nodes)
    raw_to_previous = mean_node_error(previous, nodes)
    diagnostics["raw_to_previous_m"] = raw_to_previous
    gate_m = float(args.measurement_smoothing_gate)
    if gate_m > 0.0 and np.isfinite(raw_to_previous) and raw_to_previous > gate_m:
        diagnostics["smoothing"] = "reset"
        return measurement, nodes, diagnostics

    alpha = float(args.measurement_smoothing_alpha)
    smoothed = (1.0 - alpha) * previous + alpha * nodes
    smoothed = np.ascontiguousarray(smoothed, dtype=np.float32)
    diagnostics["smoothing"] = "ema"
    diagnostics["smooth_delta_m"] = mean_node_error(nodes, smoothed)
    source_points = np.asarray(getattr(measurement, "source_points", np.empty((0, 3))), dtype=np.float32)
    residual = polyline_residual(source_points, smoothed) if len(source_points) else float(getattr(measurement, "residual_m", 0.0))
    method = f"{getattr(measurement, 'method', 'cable measurement')} | smoothed alpha={alpha:.2f}"
    return replace(measurement, points_xyz=smoothed, residual_m=float(residual), method=method), smoothed, diagnostics


def cable_tracking_diagnostics(previous_filter_nodes, raw_measurement, measurement, estimate, filter_result, smoothing_diagnostics=None):
    diagnostics = dict(smoothing_diagnostics or {})
    raw_nodes = measurement_nodes_array(raw_measurement)
    measured_nodes = measurement_nodes_array(measurement)
    estimate_nodes = measurement_nodes_array(estimate)
    previous_nodes = measurement_nodes_array(previous_filter_nodes)
    diagnostics["raw_to_filter_m"] = mean_node_error(previous_nodes, raw_nodes)
    diagnostics["measurement_to_filter_m"] = mean_node_error(previous_nodes, measured_nodes)
    diagnostics["estimate_to_raw_m"] = mean_node_error(estimate_nodes, raw_nodes)
    diagnostics["raw_residual_m"] = float(getattr(raw_measurement, "residual_m", np.nan)) if raw_measurement is not None else np.nan
    diagnostics["filtered_residual_m"] = float(getattr(estimate, "residual_m", np.nan)) if estimate is not None else np.nan
    diagnostics["proposal_ratio"] = (
        float(getattr(filter_result, "measurement_proposal_ratio", np.nan))
        if filter_result is not None
        else np.nan
    )
    diagnostics["global_random_ratio"] = (
        float(getattr(filter_result, "global_random_particle_ratio", np.nan))
        if filter_result is not None
        else np.nan
    )
    diagnostics["ransac_inlier_ratio"] = (
        float(getattr(filter_result, "ransac_inlier_ratio", np.nan))
        if filter_result is not None
        else np.nan
    )
    diagnostics["ransac_inlier_count"] = (
        int(getattr(filter_result, "ransac_inlier_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["ransac_error_m"] = (
        float(getattr(filter_result, "ransac_error_m", np.nan))
        if filter_result is not None
        else np.nan
    )
    diagnostics["ransac_hypotheses"] = (
        int(getattr(filter_result, "ransac_hypothesis_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["coarse_score_points"] = (
        int(getattr(filter_result, "coarse_score_point_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["full_score_particles"] = (
        int(getattr(filter_result, "full_score_particle_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["mean_node_speed_mps"] = (
        float(getattr(filter_result, "mean_node_speed_mps", np.nan))
        if filter_result is not None
        else np.nan
    )
    stage_seconds = getattr(filter_result, "stage_seconds", None) if filter_result is not None else None
    if isinstance(stage_seconds, dict):
        diagnostics["pf_stage_ms"] = {
            str(name): 1000.0 * float(value)
            for name, value in stage_seconds.items()
            if np.isfinite(float(value)) and float(value) > 0.0
        }
    return diagnostics


def measurement_nodes_array(value):
    if value is None:
        return None
    nodes = getattr(value, "points_xyz", value)
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) < 2:
        return None
    nodes = nodes[:, :3]
    if not np.all(np.isfinite(nodes)):
        return None
    return np.ascontiguousarray(nodes, dtype=np.float32)


def align_node_orientation(reference_nodes, candidate_nodes):
    reference = measurement_nodes_array(reference_nodes)
    candidate = measurement_nodes_array(candidate_nodes)
    if reference is None or candidate is None or reference.shape != candidate.shape:
        return candidate_nodes
    direct = float(np.mean(np.sum((reference - candidate) ** 2, axis=1)))
    reversed_error = float(np.mean(np.sum((reference - candidate[::-1]) ** 2, axis=1)))
    if reversed_error < direct:
        return candidate[::-1].copy()
    return candidate


def mean_node_error(reference_nodes, candidate_nodes):
    reference = measurement_nodes_array(reference_nodes)
    candidate = measurement_nodes_array(candidate_nodes)
    if reference is None or candidate is None or reference.shape != candidate.shape:
        return np.nan
    return float(np.mean(np.linalg.norm(reference - candidate, axis=1)))


def format_mm(value):
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{1000.0 * value:.1f}mm"


def update_viewer_cable(viewer, measurement, estimate, filter_result, max_points):
    if estimate is None:
        viewer.update_cable(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
        )
        return

    nodes = np.asarray(estimate.points_xyz, dtype=np.float32)
    valid_nodes = np.all(np.isfinite(nodes), axis=1)
    if filter_result is not None and getattr(filter_result, "visible_nodes", None) is not None:
        visible_nodes = np.asarray(filter_result.visible_nodes, dtype=bool)
        if len(visible_nodes) != len(nodes):
            visible_nodes = np.zeros(len(nodes), dtype=bool)
        visible_nodes = visible_nodes & valid_nodes
    else:
        measurement_used = filter_result is None or bool(getattr(filter_result, "measurement_used", measurement is not None))
        visible_nodes = valid_nodes if measurement_used else np.zeros(len(nodes), dtype=bool)
    extended_visible_nodes = valid_nodes
    source_points = np.empty((0, 3), dtype=np.float32)
    if measurement is not None:
        source_points = np.asarray(measurement.source_points, dtype=np.float32)

    viewer.update_cable(
        sample_points(source_points, max_points),
        nodes,
        valid_nodes,
        visible_nodes=visible_nodes,
        extended_visible_nodes=extended_visible_nodes,
    )


def cable_status(detection, measurement, estimate, filter_result, segment_count, detector_name="detector", diagnostics=None):
    if detection is None:
        return "cable: no detector result"

    diagnostics = dict(diagnostics or {})
    prefix = f"cable:{detector_name}"
    cable_count = int(diagnostics.get("cable_count", 1) or 1)
    if cable_count > 1:
        prefix = f"cables:{detector_name}"
    if estimate is not None:
        residual = float(getattr(estimate, "residual_m", 0.0))
        source_count = 0 if measurement is None else len(getattr(measurement, "source_points", ()))
        marker_count = int(getattr(measurement, "endpoint_marker_count", 0)) if measurement is not None else 0
        mode = "measurement"
        details = []
        if filter_result is not None:
            mode = "filtered" if filter_result.measurement_used else f"prediction lost={filter_result.lost_frames}"
            visible_segments = getattr(filter_result, "visible_segments", None)
            if visible_segments is not None:
                details.append(f"vis={int(np.count_nonzero(visible_segments))}/{len(visible_segments)}")
            details.append(f"pts={int(getattr(filter_result, 'measurement_point_count', 0))}")
            segment_length = float(getattr(filter_result, "segment_length_m", np.nan))
            if np.isfinite(segment_length):
                details.append(f"seg={format_mm(segment_length)}")
        details.append(f"src={source_count}")
        details.append(f"res={format_mm(residual)}")
        if cable_count > 1:
            active = int(diagnostics.get("active_cables", 0) or 0)
            candidates = int(diagnostics.get("candidate_count", 0) or 0)
            details.append(f"active={active}/{cable_count}")
            details.append(f"cand={candidates}")
        if marker_count > 0:
            details.append(f"end={marker_count}/{2 * cable_count if cable_count > 1 else 2}")
        diag_text = format_tracking_diagnostics(diagnostics)
        return f"{prefix} {mode} nseg={segment_count} " + " ".join(details) + diag_text
    if measurement is None:
        return f"{prefix} waiting for 3D support mask={detection.component_count}"
    return f"{prefix} waiting for filter nseg={segment_count}"


def format_tracking_diagnostics(diagnostics):
    if not diagnostics:
        return ""
    parts = []
    cable_count = int(diagnostics.get("cable_count", 1) or 1)
    if cable_count > 1:
        active = int(diagnostics.get("active_cables", 0) or 0)
        candidates = int(diagnostics.get("candidate_count", 0) or 0)
        parts.append(f"cables={active}/{cable_count} cand={candidates}")
        association = str(diagnostics.get("association", "") or "")
        if association:
            parts.append(f"assoc={association}")
        endpoint_counts = str(diagnostics.get("endpoint_counts_text", "") or "")
        if endpoint_counts:
            parts.append(f"end={endpoint_counts}")
        endpoint_fixed_count = diagnostics.get("endpoint_fixed_count", None)
        if endpoint_fixed_count is not None:
            parts.append(f"fixed={int(endpoint_fixed_count)}/{cable_count}")
        endpoint_distance_text = str(diagnostics.get("endpoint_distance_text", "") or "")
        if endpoint_distance_text:
            parts.append(f"edist={endpoint_distance_text}")
        endpoint_length_text = str(diagnostics.get("endpoint_model_length_text", "") or "")
        if endpoint_length_text:
            parts.append(f"elen={endpoint_length_text}")
        endpoint_unreachable = int(diagnostics.get("endpoint_unreachable_count", 0) or 0)
        if endpoint_unreachable > 0:
            parts.append(f"unreach={endpoint_unreachable}")
    raw_to_filter = diagnostics.get("raw_to_filter_m", np.nan)
    if np.isfinite(raw_to_filter):
        parts.append(f"raw2f={format_mm(raw_to_filter)}")
    prior_active_count = int(diagnostics.get("prior_active_count", 0) or 0)
    prior_gate = diagnostics.get("prior_gate_m", np.nan)
    if prior_active_count > 0 and np.isfinite(prior_gate):
        parts.append(f"prior={prior_active_count}@{format_mm(prior_gate)}")
    smooth_delta = diagnostics.get("smooth_delta_m", np.nan)
    smoothing = diagnostics.get("smoothing", "")
    if np.isfinite(smooth_delta):
        parts.append(f"smooth={format_mm(smooth_delta)}")
    elif smoothing == "reset":
        parts.append(f"smooth={smoothing}")
    proposal_ratio = diagnostics.get("proposal_ratio", np.nan)
    if np.isfinite(proposal_ratio):
        parts.append(f"prop={proposal_ratio:.2f}")
    random_ratio = diagnostics.get("global_random_ratio", np.nan)
    if np.isfinite(random_ratio):
        parts.append(f"rand={random_ratio:.2f}")
    ransac_inlier_ratio = diagnostics.get("ransac_inlier_ratio", np.nan)
    ransac_inlier_count = int(diagnostics.get("ransac_inlier_count", 0) or 0)
    ransac_error = diagnostics.get("ransac_error_m", np.nan)
    ransac_hypotheses = int(diagnostics.get("ransac_hypotheses", 0) or 0)
    if ransac_inlier_count > 0 or ransac_hypotheses > 0:
        text = f"ransac={ransac_inlier_count}pts"
        if np.isfinite(ransac_inlier_ratio):
            text += f"/{ransac_inlier_ratio:.2f}"
        if ransac_hypotheses > 0:
            text += f" hyp={ransac_hypotheses}"
        if np.isfinite(ransac_error):
            text += f" err={format_mm(ransac_error)}"
        parts.append(text)
    mean_node_speed = diagnostics.get("mean_node_speed_mps", np.nan)
    if np.isfinite(mean_node_speed):
        parts.append(f"v={mean_node_speed:.2f}m/s")
    coarse_points = int(diagnostics.get("coarse_score_points", 0) or 0)
    full_particles = int(diagnostics.get("full_score_particles", 0) or 0)
    if coarse_points > 0 and full_particles > 0:
        parts.append(f"score={coarse_points}c/{full_particles}f")
    pf_stage_ms = diagnostics.get("pf_stage_ms", None)
    if isinstance(pf_stage_ms, dict):
        stage_text = format_pf_stage_ms(pf_stage_ms)
        if stage_text:
            parts.append(stage_text)
    return "" if not parts else " | " + " ".join(parts)


def format_pf_stage_ms(stage_ms):
    labels = (
        ("prepare", "prep"),
        ("initialize", "init"),
        ("predict", "pred"),
        ("measurement", "meas"),
        ("velocity", "vel"),
        ("proposal", "propms"),
        ("associate", "assoc"),
        ("ransac", "ransacms"),
        ("random", "randms"),
        ("score", "scorems"),
        ("weight", "w"),
        ("estimate", "est"),
        ("resample", "resamp"),
    )
    parts = []
    for key, label in labels:
        value = float(stage_ms.get(key, 0.0))
        if np.isfinite(value) and value >= 0.05:
            parts.append(f"{label}={value:.1f}")
    return "" if not parts else "pfms:" + ",".join(parts)


def draw_cable_rgb_panel(
    bgr,
    detection=None,
    endpoint_mask=None,
    detection_label_mask=None,
    endpoint_label_mask=None,
    measurement=None,
    estimate=None,
    segment_count=2,
    cable_count=1,
    mode="segmentation",
    detector_description="PIDNet mask threshold 0.50",
):
    if mode == "mask":
        return draw_cable_mask_view(
            bgr,
            detection=detection,
            endpoint_mask=endpoint_mask,
            detection_label_mask=detection_label_mask,
            endpoint_label_mask=endpoint_label_mask,
            measurement=measurement,
            estimate=estimate,
            segment_count=segment_count,
            cable_count=cable_count,
            detector_description=detector_description,
        )
    if mode == "tracking":
        return draw_cable_debug_overlay(
            bgr,
            detection=detection,
            endpoint_mask=endpoint_mask,
            detection_label_mask=detection_label_mask,
            endpoint_label_mask=endpoint_label_mask,
            measurement=measurement,
            estimate=estimate,
            segment_count=segment_count,
            cable_count=cable_count,
        )
    return draw_cable_segmentation_view(
        bgr,
        detection=detection,
        endpoint_mask=endpoint_mask,
        detection_label_mask=detection_label_mask,
        endpoint_label_mask=endpoint_label_mask,
        measurement=measurement,
        estimate=estimate,
        segment_count=segment_count,
        cable_count=cable_count,
        detector_description=detector_description,
    )


SEGMENTATION_LABEL_COLORS_BGR = (
    (40, 255, 80),     # cable body
    (40, 255, 80),     # reserved cable body class
    (40, 255, 80),     # reserved cable body class
    (40, 255, 80),     # reserved cable body class
    (255, 0, 255),     # endpoints_1
    (255, 220, 0),     # endpoints_2
    (80, 80, 255),     # endpoints_3
    (0, 220, 255),     # endpoints_4
)


def segmentation_label_color(label, cable_count=None):
    label = max(1, int(label))
    if cable_count is None:
        return SEGMENTATION_LABEL_COLORS_BGR[(label - 1) % len(SEGMENTATION_LABEL_COLORS_BGR)]
    cable_count = max(1, int(cable_count))
    if label <= cable_count:
        return SEGMENTATION_LABEL_COLORS_BGR[(label - 1) % 4]
    endpoint_index = label - cable_count - 1
    return SEGMENTATION_LABEL_COLORS_BGR[4 + (endpoint_index % 4)]


def draw_labeled_mask_overlay(panel, label_mask, alpha=0.70, contour_thickness=1, cable_count=None):
    if label_mask is None:
        return False
    labels = np.asarray(label_mask, dtype=np.uint8)
    if labels.ndim != 2 or not np.any(labels):
        return False
    if labels.shape[:2] != panel.shape[:2]:
        labels = cv2.resize(labels, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        pixels = labels == label
        color = segmentation_label_color(label, cable_count=cable_count)
        tint = np.zeros_like(panel)
        tint[:, :] = np.array(color, dtype=np.uint8)
        panel[pixels] = cv2.addWeighted(panel[pixels], 1.0 - float(alpha), tint[pixels], float(alpha), 0.0)
        if int(contour_thickness) > 0:
            contour_mask = (pixels.astype(np.uint8) * 255)
            contours, _hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, color, int(contour_thickness), cv2.LINE_AA)
    return True


def draw_cable_debug_overlay(
    bgr,
    detection=None,
    endpoint_mask=None,
    detection_label_mask=None,
    endpoint_label_mask=None,
    measurement=None,
    estimate=None,
    segment_count=2,
    cable_count=1,
):
    panel = np.asarray(bgr, dtype=np.uint8).copy()
    if detection is not None:
        drew_label_mask = draw_labeled_mask_overlay(panel, detection_label_mask, alpha=0.42, contour_thickness=1, cable_count=cable_count)
        if not drew_label_mask and detection.mask is not None and np.any(detection.mask):
            pixels = detection.mask > 0
            tint = np.zeros_like(panel)
            tint[:, :, 1] = 190
            tint[:, :, 2] = 80
            panel[pixels] = cv2.addWeighted(panel[pixels], 0.58, tint[pixels], 0.42, 0.0)

        if detection.skeleton is not None and np.any(detection.skeleton):
            ys, xs = np.nonzero(detection.skeleton)
            panel[ys, xs] = (255, 180, 40)

        centerline = np.asarray(detection.centerline_xy, dtype=np.float32)
        if len(centerline) >= 2:
            pts = np.round(centerline).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(panel, [pts], isClosed=False, color=(80, 255, 120), thickness=2, lineType=cv2.LINE_AA)
            draw_2d_segment_nodes(panel, centerline, segment_count + 1)
    if not draw_labeled_mask_overlay(panel, endpoint_label_mask, alpha=0.78, contour_thickness=2, cable_count=cable_count):
        draw_endpoint_mask_overlay(panel, endpoint_mask)
    draw_endpoint_marker_overlay(panel, measurement)

    if measurement is not None:
        measured_segments = max(0, len(measurement.points_xyz) - 1)
        if measured_segments > 0:
            text = f"measurement: {measured_segments} segments residual {measurement.residual_m:.4f}m"
        else:
            text = f"measurement: mask-only support {len(measurement.source_points)} points"
    elif estimate is not None:
        text = f"prediction: {len(estimate.points_xyz) - 1} segments"
    else:
        text = f"cable: {segment_count} segments"
    cv2.putText(panel, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def draw_cable_segmentation_view(
    bgr,
    detection=None,
    endpoint_mask=None,
    detection_label_mask=None,
    endpoint_label_mask=None,
    measurement=None,
    estimate=None,
    segment_count=2,
    cable_count=1,
    detector_description="PIDNet mask threshold 0.50",
):
    bgr = np.asarray(bgr, dtype=np.uint8)
    panel = cv2.addWeighted(bgr, 0.50, np.zeros_like(bgr), 0.50, 0.0)
    if detection is not None and detection.mask is not None and np.any(detection.mask):
        drew_label_mask = draw_labeled_mask_overlay(panel, detection_label_mask, alpha=0.72, contour_thickness=1, cable_count=cable_count)
        if not drew_label_mask:
            mask = detection.mask > 0
            tint = np.zeros_like(panel)
            tint[:, :, 1] = 230
            tint[:, :, 2] = 130
            panel[mask] = cv2.addWeighted(panel[mask], 0.28, tint[mask], 0.72, 0.0)

        contours, _hierarchy = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    if not draw_labeled_mask_overlay(panel, endpoint_label_mask, alpha=0.78, contour_thickness=2, cable_count=cable_count):
        draw_endpoint_mask_overlay(panel, endpoint_mask)
    draw_detection_geometry(panel, detection, segment_count)
    draw_endpoint_marker_overlay(panel, measurement)
    draw_cable_status_text(panel, detection, measurement, estimate, segment_count, mode_text=detector_description)
    return panel


def draw_cable_mask_view(
    bgr,
    detection=None,
    endpoint_mask=None,
    detection_label_mask=None,
    endpoint_label_mask=None,
    measurement=None,
    estimate=None,
    segment_count=2,
    cable_count=1,
    detector_description="PIDNet mask threshold 0.50",
):
    shape = np.asarray(bgr, dtype=np.uint8).shape[:2]
    panel = np.full((shape[0], shape[1], 3), 18, dtype=np.uint8)
    if detection is not None and detection.mask is not None and np.any(detection.mask):
        if not draw_labeled_mask_overlay(panel, detection_label_mask, alpha=1.0, contour_thickness=1, cable_count=cable_count):
            panel[detection.mask > 0] = (255, 255, 255)
    if not draw_labeled_mask_overlay(panel, endpoint_label_mask, alpha=1.0, contour_thickness=2, cable_count=cable_count):
        draw_endpoint_mask_overlay(panel, endpoint_mask, alpha=1.0)
    if detection is not None and detection.skeleton is not None and np.any(detection.skeleton):
        ys, xs = np.nonzero(detection.skeleton)
        panel[ys, xs] = (0, 180, 255)
    draw_detection_geometry(panel, detection, segment_count)
    draw_endpoint_marker_overlay(panel, measurement)
    draw_cable_status_text(panel, detection, measurement, estimate, segment_count, mode_text=detector_description)
    return panel


def draw_endpoint_mask_overlay(panel, endpoint_mask, alpha=0.75):
    if endpoint_mask is None:
        return
    mask = np.asarray(endpoint_mask, dtype=np.uint8)
    if mask.ndim != 2 or not np.any(mask):
        return
    if mask.shape[:2] != panel.shape[:2]:
        mask = cv2.resize(mask, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
    pixels = mask > 0
    tint = np.zeros_like(panel)
    tint[:, :, 0] = 255
    tint[:, :, 2] = 255
    panel[pixels] = cv2.addWeighted(panel[pixels], 1.0 - float(alpha), tint[pixels], float(alpha), 0.0)
    contours, _hierarchy = cv2.findContours((pixels.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (255, 180, 255), 2, cv2.LINE_AA)


def draw_endpoint_marker_overlay(panel, measurement):
    if measurement is None:
        return
    centers = getattr(measurement, "endpoint_marker_centers_xy", None)
    if centers is None:
        return
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    for index, center in enumerate(centers):
        if not np.all(np.isfinite(center)):
            continue
        point = tuple(np.round(center).astype(np.int32))
        endpoint_group = index // 2
        color = SEGMENTATION_LABEL_COLORS_BGR[4 + (endpoint_group % 4)]
        cv2.circle(panel, point, 9, color, -1, cv2.LINE_AA)
        cv2.circle(panel, point, 12, (255, 255, 255), 2, cv2.LINE_AA)


def draw_detection_geometry(panel, detection, segment_count):
    if detection is None:
        return
    if detection.skeleton is not None and np.any(detection.skeleton):
        ys, xs = np.nonzero(detection.skeleton)
        panel[ys, xs] = (255, 180, 40)

    centerline = np.asarray(detection.centerline_xy, dtype=np.float32)
    if len(centerline) >= 2:
        pts = np.round(centerline).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], isClosed=False, color=(80, 255, 120), thickness=2, lineType=cv2.LINE_AA)
        draw_2d_segment_nodes(panel, centerline, segment_count + 1)


def draw_cable_status_text(panel, detection, measurement, estimate, segment_count, mode_text):
    cv2.putText(panel, mode_text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if detection is None:
        text = "waiting for cable detection"
    elif measurement is not None:
        text = f"{segment_count} segments | centerline {len(detection.centerline_xy)} | residual {measurement.residual_m:.4f}m"
    elif estimate is not None:
        text = f"{segment_count} segments | prediction only"
    else:
        text = f"{segment_count} segments | mask components {detection.component_count} | centerline {len(detection.centerline_xy)}"
    cv2.putText(panel, text, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 245, 255), 2, cv2.LINE_AA)


def draw_2d_segment_nodes(panel, centerline_xy, node_count):
    nodes = resample_xy(centerline_xy, node_count)
    if len(nodes) < 2:
        return
    pts = np.round(nodes).astype(np.int32)
    for idx in range(len(pts) - 1):
        cv2.line(panel, tuple(pts[idx]), tuple(pts[idx + 1]), (0, 255, 255), 3, cv2.LINE_AA)
    for idx, point in enumerate(pts):
        color = (255, 220, 0)
        if idx == 0:
            color = (255, 210, 40)
        elif idx == len(pts) - 1:
            color = (255, 80, 220)
        cv2.circle(panel, tuple(point), 6, color, -1, cv2.LINE_AA)


def resample_xy(points_xy, output_count):
    points = np.asarray(points_xy, dtype=np.float64)
    output_count = max(2, int(output_count))
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:, :2], output_count, axis=0).astype(np.float32)

    deltas = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 1e-9:
        return np.repeat(points[:1, :2], output_count, axis=0).astype(np.float32)

    target = np.linspace(0.0, total, output_count)
    output = np.empty((output_count, 2), dtype=np.float64)
    for axis in range(2):
        output[:, axis] = np.interp(target, cumulative, points[:, axis])
    return output.astype(np.float32)


def sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    points = points[:, :3]
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]
    max_points = max(0, int(max_points))
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return np.ascontiguousarray(points, dtype=np.float32)


def empty_point_cloud_stats():
    return {
        "shape": None,
        "sampled": 0,
        "finite": 0,
        "in_range": 0,
        "returned": 0,
        "capped": False,
    }


if __name__ == "__main__":
    main()
