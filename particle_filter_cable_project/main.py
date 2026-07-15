import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from types import SimpleNamespace
import time
import tomllib
import traceback

import cv2
import numpy as np
import pyzed.sl as sl

from cable_cuda import CudaPointCloudView
from cable_detection import (
    CableEstimate3D,
    attach_endpoint_markers_to_measurement,
    cable_measurement_from_mask_points,
    cable_measurement_from_support_points,
    endpoint_group_observations_from_mask,
    polyline_residual,
)
from cable_crossing import (
    CameraIntrinsics,
    extract_crossing_proposals,
    verify_crossing_proposals,
)
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    PerCableEndpointAssociationConfig,
    PerCableEndpointAssociator,
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
from pidnet_schema import CROSSING_CHANNEL, OUTPUT_CHANNEL_COUNT


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIDNET_CHECKPOINT = PROJECT_DIR / "models/pidnet_two_cable_best.pt"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def config_value(config, section, key, default, base_dir=None):
    section_values = config.get(section)
    if not isinstance(section_values, dict):
        raise KeyError(f"Missing configuration section [{section}].")
    if key not in section_values:
        raise KeyError(f"Missing configuration value {section}.{key}.")
    value = section_values[key]
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


def cable_segment_length_m(args, cable_index):
    return per_cable_value(args.derived_segment_lengths_m, cable_index, "derived cable segment lengths")


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
    parser.add_argument("--viewer-update-every", type=int, default=config_value(config, "viewer", "update_every", 2))
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
    parser.add_argument("--neural-detector-amp", action=argparse.BooleanOptionalAction, default=config_value(config, "pidnet", "amp", True), help="Use CUDA automatic mixed precision for PIDNet inference.")
    parser.add_argument("--neural-detector-channels-last", action=argparse.BooleanOptionalAction, default=config_value(config, "pidnet", "channels_last", True), help="Use channels-last CUDA tensors for PIDNet inference.")
    parser.add_argument("--endpoint-marker-min-area", type=int, default=config_value(config, "endpoint_markers", "min_area_px", 50))
    parser.add_argument("--endpoint-marker-min-points", type=int, default=config_value(config, "endpoint_markers", "min_points", 8))
    parser.add_argument("--endpoint-marker-open-kernel", type=int, default=config_value(config, "endpoint_markers", "open_kernel", 3))
    parser.add_argument("--endpoint-marker-close-kernel", type=int, default=config_value(config, "endpoint_markers", "close_kernel", 5))
    parser.add_argument("--endpoint-marker-points", type=int, default=config_value(config, "endpoint_markers", "max_points_per_marker", 256))
    parser.add_argument("--endpoint-marker-tape-lengths", type=float, nargs="+", default=config_value(config, "endpoint_markers", "tape_lengths_m", None))
    parser.add_argument("--endpoint-marker-offset-to-tips", action=argparse.BooleanOptionalAction, default=config_value(config, "endpoint_markers", "offset_to_tips", False))
    parser.add_argument("--endpoint-association-ambiguity-margin", type=float, default=config_value(config, "endpoint_association", "ambiguity_margin_m", 0.015))
    parser.add_argument("--endpoint-association-support-weight", type=float, default=config_value(config, "endpoint_association", "support_weight", 0.35))
    parser.add_argument("--endpoint-association-support-clip", type=float, default=config_value(config, "endpoint_association", "support_clip_m", 0.080))
    parser.add_argument("--crossing-threshold", type=float, default=config_value(config, "crossing", "threshold", 0.50))
    parser.add_argument("--crossing-min-area", type=int, default=config_value(config, "crossing", "min_area_px", 12))
    parser.add_argument("--crossing-max-proposals", type=int, default=config_value(config, "crossing", "max_proposals", 8))
    parser.add_argument("--cable-diameters", type=float, nargs="+", default=config_value(config, "crossing", "cable_diameters_m", None))
    parser.add_argument("--crossing-contact-tolerance", type=float, default=config_value(config, "crossing", "contact_tolerance_m", 0.002))
    parser.add_argument("--crossing-association-sigma", type=float, default=config_value(config, "crossing", "association_sigma_px", 18.0))
    parser.add_argument("--detector-min-area", type=int, default=config_value(config, "detector", "min_area_px", 80))
    parser.add_argument("--detector-open-kernel", type=int, default=config_value(config, "detector", "open_kernel", 3))
    parser.add_argument("--detector-close-kernel", type=int, default=config_value(config, "detector", "close_kernel", 5))
    parser.add_argument("--cable-segments", type=int, default=config_value(config, "cable", "segments", 2))
    parser.add_argument("--cable-count", type=int, default=config_value(config, "cable", "count", 1), help="Number of separate cables to reconstruct.")
    parser.add_argument("--cable-lengths", type=float, nargs="+", default=config_value(config, "cable", "lengths_m", None), help="Physical cable lengths in meters, one value per cable.")
    parser.add_argument("--cable-max-points", type=int, default=config_value(config, "cable", "max_visual_points", 1000))
    parser.add_argument("--cable-confidence-max", type=float, default=config_value(config, "measurement", "confidence_max", 85.0), help="Use <0 to disable.")
    parser.add_argument("--particle-filter", action=argparse.BooleanOptionalAction, default=config_value(config, "particle_filter", "enabled", True))
    parser.add_argument("--pf-particles", type=int, default=config_value(config, "particle_filter", "particles", pf_defaults.particle_count))
    parser.add_argument(
        "--pf-estimate-top-particles",
        type=int,
        default=config_value(
            config,
            "particle_filter",
            "estimate_top_particles",
            pf_defaults.estimate_top_particle_count,
        ),
        help="Display and propagate the constrained arithmetic mean of the highest-weight N particles.",
    )
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
    parser.add_argument("--pf-global-random-ratio", type=float, default=config_value(config, "particle_filter", "global_random_particle_ratio", pf_defaults.global_random_particle_ratio))
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
    args.cable_count = int(args.cable_count)
    if args.cable_count != 2:
        raise ValueError(f"The live endpoint associator requires exactly two physical cables; got {args.cable_count}.")
    args.cable_lengths_m = required_per_cable_float_values(
        args.cable_lengths,
        args.cable_count,
        "cable.lengths_m",
        minimum=0.0,
    )
    args.cloud_update_every = max(1, int(args.cloud_update_every))
    args.viewer_update_every = max(1, int(args.viewer_update_every))
    args.confidence_update_every = max(0, int(args.confidence_update_every))
    args.detector_scale = float(np.clip(args.detector_scale, 0.10, 1.0))
    args.detector_update_every = max(1, int(args.detector_update_every))
    args.endpoint_marker_min_area = max(1, int(args.endpoint_marker_min_area))
    args.endpoint_marker_min_points = max(1, int(args.endpoint_marker_min_points))
    args.endpoint_marker_open_kernel = max(0, int(args.endpoint_marker_open_kernel))
    args.endpoint_marker_close_kernel = max(0, int(args.endpoint_marker_close_kernel))
    args.endpoint_marker_points = max(1, int(args.endpoint_marker_points))
    args.endpoint_association_ambiguity_margin = max(0.0, float(args.endpoint_association_ambiguity_margin))
    args.endpoint_association_support_weight = max(0.0, float(args.endpoint_association_support_weight))
    args.endpoint_association_support_clip = max(1e-4, float(args.endpoint_association_support_clip))
    args.endpoint_marker_tape_lengths_m = required_per_cable_float_values(
        args.endpoint_marker_tape_lengths,
        args.cable_count,
        "endpoint_markers.tape_lengths_m",
        minimum=0.0,
    )
    args.crossing_threshold = float(np.clip(args.crossing_threshold, 0.0, 1.0))
    args.crossing_min_area = max(1, int(args.crossing_min_area))
    args.crossing_max_proposals = max(1, int(args.crossing_max_proposals))
    args.cable_diameters_m = required_per_cable_float_values(
        args.cable_diameters,
        args.cable_count,
        "crossing.cable_diameters_m",
        minimum=0.0,
    )
    args.crossing_contact_tolerance = max(0.0, float(args.crossing_contact_tolerance))
    args.crossing_association_sigma = max(1e-3, float(args.crossing_association_sigma))
    args.derived_segment_lengths_m = [
        length_m / max(args.cable_segments, 1)
        for length_m in args.cable_lengths_m
    ]
    args.pf_particles = max(32, int(args.pf_particles))
    args.pf_estimate_top_particles = int(np.clip(
        args.pf_estimate_top_particles,
        1,
        args.pf_particles,
    ))
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
    args.pf_global_random_ratio = float(np.clip(args.pf_global_random_ratio, 0.0, 1.0))
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


def camera_intrinsics_from_zed(zed):
    information = zed.get_camera_information()
    left = information.camera_configuration.calibration_parameters.left_cam
    intrinsics = CameraIntrinsics(
        fx=float(left.fx),
        fy=float(left.fy),
        cx=float(left.cx),
        cy=float(left.cy),
        y_axis_up=True,
    )
    if not all(np.isfinite((intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy))):
        raise RuntimeError("ZED returned non-finite left-camera intrinsics.")
    return intrinsics


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
        min_area=int(args.detector_min_area),
        open_kernel=int(args.detector_open_kernel),
        close_kernel=int(args.detector_close_kernel),
        amp=bool(args.neural_detector_amp),
        channels_last=bool(args.neural_detector_channels_last),
    )
    endpoint_channel_count = 2
    expected_channels = OUTPUT_CHANNEL_COUNT
    expected_crossing_channel = CROSSING_CHANNEL
    if detector.trained_endpoint_channel_count != endpoint_channel_count:
        raise RuntimeError(
            f"Checkpoint endpoint_channel_count={detector.trained_endpoint_channel_count}; "
            f"expected endpoints_cable1 and endpoints_cable2 ({endpoint_channel_count} heads)."
        )
    if detector.output_channels != expected_channels:
        raise RuntimeError(f"Checkpoint must output {expected_channels} channels; got {detector.output_channels}.")
    if not detector.has_crossing_channel or detector.crossing_channel != expected_crossing_channel:
        raise RuntimeError(
            f"Checkpoint crossing channel must be {expected_crossing_channel}; got {detector.crossing_channel}."
        )
    print(
        f"Loaded PIDNet cable detector: {checkpoint_path} on {args.neural_detector_device} "
        f"(amp={bool(args.neural_detector_amp)} channels_last={bool(args.neural_detector_channels_last)} "
        f"outputs={detector.output_channels} labels={detector.label_mode})"
    )
    return detector


def detector_description(args):
    return f"PIDNet mask threshold {float(args.neural_detector_threshold):.2f}"


def detect_cable_endpoint_markers(
    point_cloud,
    args,
    confidence_measure=None,
    endpoint_mask=None,
):
    if endpoint_mask is None:
        raise RuntimeError(
            "Internal error: endpoint detection requires the corresponding cable endpoint mask from PIDNet."
        )
    return endpoint_group_observations_from_mask(
        endpoint_mask,
        point_cloud,
        open_kernel=args.endpoint_marker_open_kernel,
        close_kernel=args.endpoint_marker_close_kernel,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        confidence_map=confidence_measure,
        max_confidence=args.cable_confidence_max if args.cable_confidence_max >= 0.0 else None,
        min_area_px=args.endpoint_marker_min_area,
        min_points_per_component=args.endpoint_marker_min_points,
        max_points_per_component=args.endpoint_marker_points,
        max_components=2,
    )


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
        estimate_top_particle_count=int(args.pf_estimate_top_particles),
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
        global_random_particle_ratio=float(args.pf_global_random_ratio),
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
    run_live_parallel(args)


@dataclass
class AsyncTrackingFrame:
    frame_index: int
    timestamp_s: float
    bgr: np.ndarray
    point_cloud: object
    confidence_measure: np.ndarray | None
    release_callback: object | None = None


@dataclass
class DetectedTrackingFrame:
    frame: AsyncTrackingFrame
    detection: object
    endpoint_mask: np.ndarray
    detection_label_mask: np.ndarray | None
    endpoint_label_mask: np.ndarray | None
    endpoint_channel_masks: list
    crossing_mask: np.ndarray
    crossing_proposals: tuple
    detect_seconds: float


@dataclass
class AsyncTrackingResult:
    frame_index: int
    frame_timestamp_s: float
    detection: object | None
    endpoint_mask: np.ndarray | None
    detection_label_mask: np.ndarray | None
    endpoint_label_mask: np.ndarray | None
    endpoint_overlap_mask: np.ndarray | None
    endpoint_observations: tuple | None
    crossing_mask: np.ndarray | None
    crossing_proposals: tuple
    contact_observations: tuple
    measurement: object | None
    estimate: object | None
    filter_result: object | None
    tracking_diagnostics: dict
    worker_stage_seconds: dict
    worker_seconds: float
    last_filter_lost_frames: int


def label_mask_from_binary_masks(masks, output_shape, first_label=1, reject_overlaps=False):
    output_h, output_w = int(output_shape[0]), int(output_shape[1])
    label_mask = np.zeros((output_h, output_w), dtype=np.uint8)
    overlap_count = np.zeros((output_h, output_w), dtype=np.uint8) if bool(reject_overlaps) else None
    for index, mask in enumerate(masks or []):
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim != 2 or not np.any(mask):
            continue
        if mask.shape[:2] != (output_h, output_w):
            mask = cv2.resize(mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
        pixels = mask > 0
        label_mask[pixels] = int(first_label) + int(index)
        if overlap_count is not None:
            overlap_count[pixels] = np.minimum(overlap_count[pixels] + 1, 255)
    if overlap_count is not None:
        label_mask[overlap_count > 1] = 0
    return None if not np.any(label_mask) else np.ascontiguousarray(label_mask, dtype=np.uint8)


def overlap_mask_from_binary_masks(masks, output_shape):
    output_h, output_w = int(output_shape[0]), int(output_shape[1])
    overlap_count = np.zeros((output_h, output_w), dtype=np.uint8)
    for mask in masks or []:
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim != 2:
            continue
        if mask.shape[:2] != (output_h, output_w):
            mask = cv2.resize(mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
        pixels = mask > 0
        overlap_count[pixels] = np.minimum(overlap_count[pixels] + 1, 255)
    overlap = (overlap_count > 1).astype(np.uint8) * 255
    return None if not np.any(overlap) else np.ascontiguousarray(overlap, dtype=np.uint8)


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
        self.detector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cable-detector")
        self.tracking_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cable-tracking")
        self.branch_executor = ThreadPoolExecutor(
            max_workers=max(1, int(args.cable_count)),
            thread_name_prefix="cable-branch",
        )
        self.detector_future = None
        self.detector_frame = None
        self.tracking_future = None
        self.tracking_frame = None
        self.pending_detected = None
        self.completed_results = []
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.busy_frames = 0
        self.last_filter_nodes_by_cable = [None for _ in range(max(1, int(args.cable_count)))]
        self.last_filter_lost_frames = 0
        self.last_filter_lost_frames_by_cable = [0 for _ in range(max(1, int(args.cable_count)))]
        self.endpoint_associator = PerCableEndpointAssociator(
            cable_count=max(1, int(args.cable_count)),
            config=PerCableEndpointAssociationConfig(
                ambiguity_margin_m=float(args.endpoint_association_ambiguity_margin),
                support_weight=float(args.endpoint_association_support_weight),
                support_clip_m=float(args.endpoint_association_support_clip),
                endpoint_constraint_tolerance_m=float(args.pf_endpoint_constraint_tolerance),
                node_count=int(args.cable_segments) + 1,
                max_stale_frames=int(args.pf_max_prediction_frames),
            )
        )
        self.last_filter_time = None

    def busy_flag(self):
        return bool(
            self.detector_future is not None
            or self.tracking_future is not None
            or self.pending_detected is not None
        )

    def can_submit(self):
        self._advance()
        return self.detector_future is None and self.pending_detected is None

    def submit(self, frame):
        self._advance()
        if self.detector_future is not None:
            self.dropped += 1
            return False
        self.detector_frame = frame
        self.detector_future = self.detector_executor.submit(self._detect_frame, frame)
        self.submitted += 1
        return True

    def drain_latest(self):
        self._advance()
        if not self.completed_results:
            return None
        result = self.completed_results[-1]
        self.completed_results.clear()
        return result

    def close(self):
        try:
            while (
                self.detector_future is not None
                or self.tracking_future is not None
                or self.pending_detected is not None
            ):
                self._advance()
                time.sleep(0.001)
        finally:
            self.detector_executor.shutdown(wait=True, cancel_futures=True)
            self.tracking_executor.shutdown(wait=True, cancel_futures=True)
            self.branch_executor.shutdown(wait=True, cancel_futures=True)

    def _advance(self):
        if self.tracking_future is not None and self.tracking_future.done():
            future = self.tracking_future
            frame = self.tracking_frame
            self.tracking_future = None
            self.tracking_frame = None
            try:
                result = future.result()
            except Exception as exc:
                self._release_frame(frame)
                formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                raise RuntimeError(f"Async tracking stage failed:\n{formatted}") from exc
            self._release_frame(frame)
            self.completed += 1
            self.completed_results.append(result)

        if self.detector_future is not None and self.detector_future.done():
            future = self.detector_future
            frame = self.detector_frame
            self.detector_future = None
            self.detector_frame = None
            try:
                detected = future.result()
            except Exception as exc:
                self._release_frame(frame)
                formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                raise RuntimeError(f"Async detector stage failed:\n{formatted}") from exc
            if self.tracking_future is None:
                self._submit_tracking(detected)
            else:
                if self.pending_detected is not None:
                    self._release_frame(self.pending_detected.frame)
                    self.dropped += 1
                self.pending_detected = detected

        if self.tracking_future is None and self.pending_detected is not None:
            detected = self.pending_detected
            self.pending_detected = None
            self._submit_tracking(detected)

    def _submit_tracking(self, detected):
        self.tracking_frame = detected.frame
        self.tracking_future = self.tracking_executor.submit(self._process_detected_frame, detected)

    @staticmethod
    def _release_frame(frame):
        if frame is not None and callable(frame.release_callback):
            frame.release_callback()

    def _detect_frame(self, frame):
        args = self.args
        stage_start = time.monotonic()
        observation = self.cable_detector.detect_observation_masks(
            frame.bgr,
            endpoint_channel_count=2,
            scale=args.detector_scale,
            crossing_threshold=args.crossing_threshold,
            include_endpoint_mask=True,
        )
        detection = observation.cable_detection
        endpoint_mask = observation.endpoint_mask
        endpoint_channel_masks = list(observation.endpoint_masks_by_cable)
        if detection is None:
            raise RuntimeError("PIDNet did not return the shared cable mask.")
        if len(endpoint_channel_masks or []) != 2 or any(mask is None for mask in endpoint_channel_masks):
            raise RuntimeError("PIDNet must return endpoints_cable1 and endpoints_cable2 masks.")
        detection_label_mask = label_mask_from_detections(
            [detection],
            frame.bgr.shape[:2],
            first_label=1,
        )
        endpoint_label_mask = label_mask_from_binary_masks(
            endpoint_channel_masks,
            frame.bgr.shape[:2],
            first_label=2,
            reject_overlaps=True,
        )
        crossing_proposals = extract_crossing_proposals(
            observation.crossing_mask,
            observation.crossing_probability,
            min_area_px=args.crossing_min_area,
            max_proposals=args.crossing_max_proposals,
        )
        return DetectedTrackingFrame(
            frame=frame,
            detection=detection,
            endpoint_mask=endpoint_mask,
            detection_label_mask=detection_label_mask,
            endpoint_label_mask=endpoint_label_mask,
            endpoint_channel_masks=list(endpoint_channel_masks),
            crossing_mask=observation.crossing_mask,
            crossing_proposals=crossing_proposals,
            detect_seconds=time.monotonic() - stage_start,
        )

    def _process_detected_frame(self, detected):
        stage_seconds = {"detect": float(detected.detect_seconds), "fit": 0.0, "filter": 0.0}
        return self._process_frame_multi(
            detected.frame,
            detected.detection,
            detected.endpoint_mask,
            detected.detection_label_mask,
            detected.endpoint_label_mask,
            stage_seconds,
            time.monotonic() - float(detected.detect_seconds),
            endpoint_channel_masks=detected.endpoint_channel_masks,
            crossing_mask=detected.crossing_mask,
            crossing_proposals=detected.crossing_proposals,
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
        endpoint_channel_masks=None,
        crossing_mask=None,
        crossing_proposals=None,
    ):
        args = self.args
        cable_count = max(1, int(args.cable_count))
        endpoint_channel_masks = list(endpoint_channel_masks or [])
        if len(endpoint_channel_masks) != 2 or any(mask is None for mask in endpoint_channel_masks):
            raise RuntimeError("Expected exactly two endpoint masks: endpoints_cable1 and endpoints_cable2.")
        measurements = [None for _ in range(cable_count)]
        estimates = [None for _ in range(cable_count)]
        filter_results = [None for _ in range(cable_count)]
        diagnostics = [{} for _ in range(cable_count)]

        stage_start = time.monotonic()
        shared_support_future = self.branch_executor.submit(
            self._shared_cable_support,
            frame,
            detection,
        )
        endpoint_futures = [
            self.branch_executor.submit(
                self._detect_endpoint_branch,
                frame,
                endpoint_channel_masks[cable_index],
            )
            for cable_index in range(2)
        ]
        shared_support = shared_support_future.result()
        endpoint_markers_by_cable = [future.result() for future in endpoint_futures]
        association_result = self.endpoint_associator.associate(
            endpoint_markers_by_cable,
            self.last_filter_nodes_by_cable,
            shared_support,
            args.cable_lengths_m,
            args.endpoint_marker_tape_lengths_m,
            offset_to_tips=args.endpoint_marker_offset_to_tips,
        )
        candidate_markers = association_result.markers_by_cable
        endpoint_observations = endpoint_group_observations_xy(endpoint_markers_by_cable)
        endpoint_overlap_mask = overlap_mask_from_binary_masks(
            endpoint_channel_masks,
            frame.bgr.shape[:2],
        )
        association_result.diagnostics["endpoint_overlap_px"] = int(
            0 if endpoint_overlap_mask is None else np.count_nonzero(endpoint_overlap_mask)
        )
        measurement_futures = [
            self.branch_executor.submit(
                self._process_measurement_branch,
                cable_index,
                candidate_markers[cable_index],
                shared_support,
                association_result.per_cable_diagnostics[cable_index],
            )
            for cable_index in range(cable_count)
        ]
        for cable_index, future in enumerate(measurement_futures):
            measurements[cable_index], diagnostics[cable_index] = future.result()
        stage_seconds["fit"] += time.monotonic() - stage_start

        stage_start = time.monotonic()
        filter_dt = (
            1.0 / max(float(args.fps), 1.0)
            if self.last_filter_time is None
            else max(1e-3, float(frame.timestamp_s - self.last_filter_time))
        )
        self.last_filter_time = frame.timestamp_s

        filter_futures = [
            self.branch_executor.submit(
                self._process_filter_branch,
                cable_index,
                measurements[cable_index],
                diagnostics[cable_index],
                filter_dt,
            )
            for cable_index in range(cable_count)
        ]
        for cable_index, future in enumerate(filter_futures):
            filter_results[cable_index], estimates[cable_index], diagnostics[cable_index] = future.result()

        contact_observations = verify_crossing_proposals(
            crossing_proposals,
            [None if estimate is None else estimate.points_xyz for estimate in estimates],
            getattr(args, "camera_intrinsics", None),
            args.cable_diameters_m,
            contact_tolerance_m=args.crossing_contact_tolerance,
            association_sigma_px=args.crossing_association_sigma,
        )

        stage_seconds["filter"] += time.monotonic() - stage_start

        measurement = combine_cable_estimates(measurements, method="multi-cable measurement")
        estimate = combine_cable_estimates(estimates, method="multi-cable estimate")
        filter_result = combine_filter_results(filter_results)
        tracking_diagnostics = combine_tracking_diagnostics(
            diagnostics,
            candidate_count=cable_count,
            cable_count=cable_count,
        )
        tracking_diagnostics.update(association_result.diagnostics)
        tracking_diagnostics["association"] = "fixed_endpoint_channel_identity"
        tracking_diagnostics["crossing_proposal_count"] = len(tuple(crossing_proposals or ()))
        tracking_diagnostics["crossing_verified_count"] = sum(
            int(observation.verified_contact) for observation in contact_observations
        )
        tracking_diagnostics["crossing_diameter_calibrated"] = bool(
            np.all(np.asarray(args.cable_diameters_m, dtype=np.float64) > 0.0)
        )
        if contact_observations:
            best_contact = max(contact_observations, key=lambda item: item.confidence)
            tracking_diagnostics["crossing_gap_m"] = float(best_contact.gap_m)
            tracking_diagnostics["crossing_confidence"] = float(best_contact.confidence)
            tracking_diagnostics["crossing_depth_order"] = str(best_contact.depth_order)

        self.last_filter_lost_frames = max(self.last_filter_lost_frames_by_cable) if self.last_filter_lost_frames_by_cable else 0

        worker_done = time.monotonic()
        return AsyncTrackingResult(
            frame_index=int(frame.frame_index),
            frame_timestamp_s=float(frame.timestamp_s),
            detection=detection,
            endpoint_mask=endpoint_mask,
            detection_label_mask=detection_label_mask,
            endpoint_label_mask=endpoint_label_mask,
            endpoint_overlap_mask=endpoint_overlap_mask,
            endpoint_observations=endpoint_observations,
            crossing_mask=crossing_mask,
            crossing_proposals=tuple(crossing_proposals or ()),
            contact_observations=contact_observations,
            measurement=measurement,
            estimate=estimate,
            filter_result=filter_result,
            tracking_diagnostics=tracking_diagnostics,
            worker_stage_seconds=stage_seconds,
            worker_seconds=float(worker_done - worker_start),
            last_filter_lost_frames=int(self.last_filter_lost_frames),
        )

    def _detect_endpoint_branch(self, frame, endpoint_group_mask):
        args = self.args
        return detect_cable_endpoint_markers(
            frame.point_cloud,
            args,
            confidence_measure=frame.confidence_measure,
            endpoint_mask=endpoint_group_mask,
        )

    def _shared_cable_support(self, frame, detection):
        measurement = self._measurement_from_detection(
            frame,
            detection,
        )
        if measurement is None:
            return np.empty((0, 3), dtype=np.float32)
        return np.ascontiguousarray(measurement.source_points, dtype=np.float32)

    def _process_measurement_branch(
        self,
        cable_index,
        candidate_markers,
        shared_support,
        association_diagnostics,
    ):
        args = self.args
        measurement = cable_measurement_from_support_points(
            shared_support,
            segment_count=args.cable_segments,
        )
        endpoint_diag = endpoint_anchor_diagnostics(candidate_markers, args, cable_index=cable_index)
        if not bool(endpoint_diag.get("endpoint_fixed", False)):
            measurement = None
        measurement = attach_endpoint_markers_to_measurement(measurement, candidate_markers)
        measurement_diagnostics = {}
        measurement_diagnostics.update(endpoint_diag)
        measurement_diagnostics.update(dict(association_diagnostics or {}))
        return measurement, measurement_diagnostics

    def _process_filter_branch(self, cable_index, measurement, diagnostics, filter_dt):
        particle_filter = self.particle_filters[cable_index] if cable_index < len(self.particle_filters) else None
        if particle_filter is not None:
            filter_result = particle_filter.step(measurement, filter_dt)
            estimate = filtered_cable_estimate(measurement, filter_result)
        else:
            filter_result = None
            estimate = measurement
        diagnostics = cable_tracking_diagnostics(
            self.last_filter_nodes_by_cable[cable_index],
            measurement,
            estimate,
            filter_result,
            diagnostics,
        )
        if estimate is not None:
            self.last_filter_nodes_by_cable[cable_index] = np.asarray(estimate.points_xyz, dtype=np.float32)
        elif particle_filter is None or not bool(getattr(particle_filter, "initialized", False)):
            self.last_filter_nodes_by_cable[cable_index] = None
        if filter_result is not None:
            self.last_filter_lost_frames_by_cable[cable_index] = int(filter_result.lost_frames)
        elif estimate is not None:
            self.last_filter_lost_frames_by_cable[cable_index] = 0
        return filter_result, estimate, diagnostics

    def _measurement_from_detection(
        self,
        frame,
        detection,
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
        )


def combine_cable_estimates(estimates, method="multi-cable"):
    indexed_estimates = [
        (int(cable_index), estimate)
        for cable_index, estimate in enumerate(estimates)
        if estimate is not None
    ]
    if not indexed_estimates:
        return None
    valid_estimates = [estimate for _cable_index, estimate in indexed_estimates]
    nodes, pf_node_runs = concatenate_indexed_node_chains([
        (cable_index, getattr(estimate, "points_xyz", None))
        for cable_index, estimate in indexed_estimates
    ])
    source_points = concatenate_point_sets([getattr(estimate, "source_points", None) for estimate in valid_estimates])
    residuals = [float(getattr(estimate, "residual_m", np.nan)) for estimate in valid_estimates]
    finite_residuals = [value for value in residuals if np.isfinite(value)]
    residual = float(np.mean(finite_residuals)) if finite_residuals else 0.0
    centers_xyz = concatenate_point_sets([getattr(estimate, "endpoint_marker_centers_xyz", None) for estimate in valid_estimates])
    centers_xy = concatenate_xy_sets([getattr(estimate, "endpoint_marker_centers_xy", None) for estimate in valid_estimates])
    endpoint_nodes = concatenate_node_chains([getattr(estimate, "endpoint_nodes", None) for estimate in valid_estimates])
    endpoint_marker_pf_ids = []
    endpoint_marker_end_indices = []
    for cable_index, estimate in indexed_estimates:
        marker_centers = np.asarray(
            getattr(estimate, "endpoint_marker_centers_xy", None),
            dtype=np.float32,
        )
        marker_count = len(marker_centers) if marker_centers.ndim == 2 and marker_centers.shape[1] >= 2 else 0
        endpoint_marker_pf_ids.extend([int(cable_index)] * marker_count)
        endpoint_marker_end_indices.extend([0 if index == 0 else 1 if index == 1 else -1 for index in range(marker_count)])
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
        pf_node_runs=pf_node_runs,
        endpoint_marker_pf_ids=np.asarray(endpoint_marker_pf_ids, dtype=np.int16),
        endpoint_marker_end_indices=np.asarray(endpoint_marker_end_indices, dtype=np.int8),
    )


def endpoint_group_observations_xy(endpoint_markers_by_cable):
    observations = []
    for cable_index, marker in enumerate(endpoint_markers_by_cable or []):
        centers = np.asarray(getattr(marker, "centers_xy", None), dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] < 2:
            continue
        for candidate_index, center in enumerate(centers[:, :2]):
            if np.all(np.isfinite(center)):
                observations.append((int(cable_index), int(candidate_index), float(center[0]), float(center[1])))
    return tuple(observations)


def concatenate_indexed_node_chains(indexed_chains):
    output = []
    runs = []
    cursor = 0
    for cable_index, chain in indexed_chains:
        points = np.asarray(chain, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            continue
        if output:
            output.append(np.full((1, 3), np.nan, dtype=np.float32))
            cursor += 1
        points = np.ascontiguousarray(points[:, :3], dtype=np.float32)
        start = cursor
        output.append(points)
        cursor += len(points)
        runs.append((int(cable_index), int(start), int(cursor - 1)))
    if not output:
        return np.empty((0, 3), dtype=np.float32), tuple()
    return np.ascontiguousarray(np.vstack(output), dtype=np.float32), tuple(runs)


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


def finite_result_mean(results, attribute):
    values = [
        float(getattr(result, attribute, np.nan))
        for result in results
        if np.isfinite(float(getattr(result, attribute, np.nan)))
    ]
    return float(np.mean(values)) if values else np.nan


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
    indexed_results = [
        (cable_index, result)
        for cable_index, result in enumerate(filter_results)
        if result is not None
    ]
    valid_results = [result for _cable_index, result in indexed_results]
    if not valid_results:
        return None
    updated_results = [
        result for result in valid_results
        if bool(getattr(result, "measurement_used", False))
    ]
    ransac_results = [
        result for result in valid_results
        if int(getattr(result, "ransac_hypothesis_count", 0) or 0) > 0
    ]
    ransac_error_values = [
        float(getattr(result, "ransac_error_m", np.nan))
        for result in valid_results
        if np.isfinite(float(getattr(result, "ransac_error_m", np.nan)))
    ]
    estimate_weight_values = [
        float(getattr(result, "estimate_weight_mass", np.nan))
        for result in valid_results
        if np.isfinite(float(getattr(result, "estimate_weight_mass", np.nan)))
    ]
    particle_diagnostics = tuple(
        (int(cable_index), diagnostics)
        for cable_index, result in indexed_results
        for diagnostics in (getattr(result, "particle_diagnostics", None),)
        if diagnostics is not None
    )
    map_average_errors = [
        float(getattr(diagnostics, "map_to_average_node_error_m", np.nan))
        for _cable_index, diagnostics in particle_diagnostics
        if np.isfinite(float(getattr(diagnostics, "map_to_average_node_error_m", np.nan)))
    ]
    mean_spreads = [
        float(getattr(diagnostics, "mean_node_spread_m", np.nan))
        for _cable_index, diagnostics in particle_diagnostics
        if np.isfinite(float(getattr(diagnostics, "mean_node_spread_m", np.nan)))
    ]
    max_spreads = [
        float(getattr(diagnostics, "max_node_spread_m", np.nan))
        for _cable_index, diagnostics in particle_diagnostics
        if np.isfinite(float(getattr(diagnostics, "max_node_spread_m", np.nan)))
    ]
    endpoint_direction_deltas = [
        np.asarray(getattr(diagnostics, "endpoint_direction_delta_deg", (np.nan, np.nan)), dtype=np.float32)
        for _cable_index, diagnostics in particle_diagnostics
    ]
    endpoint_direction_mean = np.full(2, np.nan, dtype=np.float32)
    if endpoint_direction_deltas:
        direction_stack = np.stack(endpoint_direction_deltas, axis=0)
        finite_direction = np.isfinite(direction_stack)
        finite_count = np.count_nonzero(finite_direction, axis=0)
        finite_sum = np.sum(np.where(finite_direction, direction_stack, 0.0), axis=0)
        np.divide(finite_sum, finite_count, out=endpoint_direction_mean, where=finite_count > 0)
    visible_nodes = concatenate_bool_chains([getattr(result, "visible_nodes", None) for result in valid_results])
    extended_nodes = concatenate_bool_chains([
        extend_node_visibility(getattr(result, "visible_nodes", None))
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
        segment_length_m=finite_result_mean(valid_results, "segment_length_m"),
        measurement_proposal_ratio=finite_result_mean(updated_results, "measurement_proposal_ratio"),
        global_random_particle_ratio=finite_result_mean(updated_results, "global_random_particle_ratio"),
        ransac_inlier_ratio=finite_result_mean(ransac_results, "ransac_inlier_ratio"),
        ransac_inlier_count=sum(int(getattr(result, "ransac_inlier_count", 0)) for result in valid_results),
        ransac_error_m=float(np.mean(ransac_error_values)) if ransac_error_values else np.nan,
        ransac_hypothesis_count=sum(int(getattr(result, "ransac_hypothesis_count", 0)) for result in valid_results),
        coarse_score_point_count=sum(int(getattr(result, "coarse_score_point_count", 0)) for result in valid_results),
        full_score_particle_count=sum(int(getattr(result, "full_score_particle_count", 0)) for result in valid_results),
        estimate_particle_count=int(round(np.mean([
            int(getattr(result, "estimate_particle_count", 1))
            for result in valid_results
        ]))),
        estimate_weight_mass=(
            float(np.mean(estimate_weight_values))
            if estimate_weight_values else np.nan
        ),
        map_to_average_node_error_m=float(np.mean(map_average_errors)) if map_average_errors else np.nan,
        mean_node_spread_m=float(np.mean(mean_spreads)) if mean_spreads else np.nan,
        max_node_spread_m=float(np.max(max_spreads)) if max_spreads else np.nan,
        endpoint_direction_delta_deg=endpoint_direction_mean,
        particle_diagnostics=particle_diagnostics,
        mean_node_speed_mps=finite_result_mean(valid_results, "mean_node_speed_mps"),
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


def extend_node_visibility(values):
    values = np.asarray(values, dtype=bool).reshape(-1)
    if len(values) == 0:
        return values
    extended = values.copy()
    extended[:-1] |= values[1:]
    extended[1:] |= values[:-1]
    return extended


def combine_tracking_diagnostics(diagnostics, candidate_count=0, cable_count=1):
    valid = [dict(item) for item in diagnostics if isinstance(item, dict)]
    combined = {
        "cable_count": int(cable_count),
        "candidate_count": int(candidate_count),
        "active_cables": int(sum(bool(item.get("measurement_used", False)) for item in valid)),
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
        "support_to_prior_m",
        "proposal_ratio",
        "global_random_ratio",
        "ransac_inlier_ratio",
        "ransac_error_m",
        "estimate_weight_mass",
        "map_to_average_node_error_m",
        "mean_node_spread_m",
        "estimate_start_direction_delta_deg",
        "estimate_end_direction_delta_deg",
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
    estimate_counts = [
        int(item.get("estimate_particle_count", 0) or 0)
        for item in valid
        if int(item.get("estimate_particle_count", 0) or 0) > 0
    ]
    if estimate_counts:
        combined["estimate_particle_count"] = int(round(np.mean(estimate_counts)))
    max_spreads = [
        float(item.get("max_node_spread_m", np.nan))
        for item in valid
        if np.isfinite(float(item.get("max_node_spread_m", np.nan)))
    ]
    if max_spreads:
        combined["max_node_spread_m"] = float(np.max(max_spreads))
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
    capture_fps,
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
        f"Frame {frame_count} | GUI {render_fps:.1f} FPS | CAPTURE {capture_fps:.1f} FPS | "
        f"TRACK {compute_fps:.1f} FPS (submit {submit_fps:.1f} FPS) | lag {lag_text} | "
        f"capture {main_ms:.1f}ms grab={main_stage_ms['capture']:.1f} "
        f"cloud={main_stage_ms['cloud']:.1f} submit={main_stage_ms['copy']:.1f} | "
        f"ui {main_stage_ms['ui']:.1f}ms/update | "
        f"worker {worker_ms:.1f}ms det={worker_stage_ms['detect']:.1f} "
        f"fit={worker_stage_ms['fit']:.1f} pf={worker_stage_ms['filter']:.1f} | "
        f"async sub/ok/drop={async_worker.submitted}/{async_worker.completed}/{async_worker.dropped} "
        f"busy={1 if async_worker.busy_flag() else 0} | cloud {cloud_text} pts | "
        f"{latest_cable_status}"
    )


class TrackingGpuBuffer:
    def __init__(self):
        self.point_cloud = sl.Mat()
        self.confidence_map = sl.Mat()
        self.in_use = False
        self.last_confidence_frame = -1


class LiveCaptureWorker:
    def __init__(self, args, zed, runtime, async_worker):
        self.args = args
        self.zed = zed
        self.runtime = runtime
        self.async_worker = async_worker
        self.image = sl.Mat()
        self.visual_point_cloud = sl.Mat()
        self.tracking_buffers = [TrackingGpuBuffer() for _ in range(3)]
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="zed-capture", daemon=True)
        self.error = None
        self.frame_count = 0
        self.latest_bgr = None
        self.latest_vertices = np.empty((0, 6), dtype=np.float32)
        self.latest_stats = empty_point_cloud_stats()
        self.latest_result = None
        self.capture_loop_seconds = 0.0
        self.capture_stage_seconds = {"capture": 0.0, "cloud": 0.0, "copy": 0.0}
        self.worker_seconds = 0.0
        self.worker_frames = 0
        self.worker_stage_seconds = {"detect": 0.0, "fit": 0.0, "filter": 0.0}

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise RuntimeError("ZED capture thread did not stop within five seconds.")

    def free(self):
        self.image.free()
        self.visual_point_cloud.free()
        for buffer in self.tracking_buffers:
            buffer.point_cloud.free(sl.MEM.GPU)
            if buffer.last_confidence_frame >= 0:
                buffer.confidence_map.free(sl.MEM.GPU)

    def snapshot(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"ZED capture pipeline failed:\n{self.error}")
            return SimpleNamespace(
                frame_count=int(self.frame_count),
                bgr=self.latest_bgr,
                vertices=self.latest_vertices,
                point_stats=dict(self.latest_stats),
                result=self.latest_result,
                capture_loop_seconds=float(self.capture_loop_seconds),
                capture_stage_seconds=dict(self.capture_stage_seconds),
                worker_seconds=float(self.worker_seconds),
                worker_frames=int(self.worker_frames),
                worker_stage_seconds=dict(self.worker_stage_seconds),
                submitted=int(self.async_worker.submitted),
                completed=int(self.async_worker.completed),
            )

    def _run(self):
        latest_result = None
        try:
            while not self.stop_event.is_set():
                if self.zed.grab(self.runtime) > sl.ERROR_CODE.SUCCESS:
                    continue
                loop_start = time.monotonic()
                completed_result = self.async_worker.drain_latest()
                if completed_result is not None:
                    latest_result = completed_result

                frame_index = self.frame_count
                want_submit = should_submit_async_frame(self.args, frame_index, latest_result)
                can_submit = self.async_worker.can_submit()
                tracking_buffer = next(
                    (buffer for buffer in self.tracking_buffers if not buffer.in_use),
                    None,
                )
                submit_frame = bool(want_submit and can_submit and tracking_buffer is not None)
                if want_submit and not submit_frame:
                    self.async_worker.busy_frames += 1
                    self.async_worker.dropped += 1

                stage_start = time.monotonic()
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU)
                update_cloud = frame_index % self.args.cloud_update_every == 0
                if update_cloud:
                    self.zed.retrieve_measure(self.visual_point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                if submit_frame:
                    error = self.zed.retrieve_measure(
                        tracking_buffer.point_cloud,
                        getattr(sl.MEASURE, "XYZ", sl.MEASURE.XYZRGBA),
                        sl.MEM.GPU,
                    )
                    if error != sl.ERROR_CODE.SUCCESS:
                        raise RuntimeError(f"ZED GPU point-cloud retrieval failed: {error}")
                    if self.args.confidence_update_every > 0 and (
                        tracking_buffer.last_confidence_frame < 0
                        or frame_index - tracking_buffer.last_confidence_frame
                        >= self.args.confidence_update_every
                    ):
                        error = self.zed.retrieve_measure(
                            tracking_buffer.confidence_map,
                            sl.MEASURE.CONFIDENCE,
                            sl.MEM.GPU,
                        )
                        if error != sl.ERROR_CODE.SUCCESS:
                            raise RuntimeError(f"ZED GPU confidence retrieval failed: {error}")
                        tracking_buffer.last_confidence_frame = int(frame_index)
                capture_seconds = time.monotonic() - stage_start

                stage_start = time.monotonic()
                bgr = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR)
                vertices = self.latest_vertices
                point_stats = self.latest_stats
                if update_cloud:
                    vertices, point_stats = live_point_cloud_to_vertices(
                        self.visual_point_cloud,
                        stride=self.args.live_stride,
                        max_points=self.args.live_max_points,
                        depth_min=self.args.depth_min,
                        depth_max=self.args.depth_max,
                        return_stats=True,
                    )
                cloud_seconds = time.monotonic() - stage_start

                stage_start = time.monotonic()
                if submit_frame:
                    confidence_enabled = bool(
                        self.args.confidence_update_every > 0
                        and self.args.cable_confidence_max >= 0.0
                        and tracking_buffer.last_confidence_frame >= 0
                    )
                    point_cloud_view = CudaPointCloudView(
                        pointer=int(tracking_buffer.point_cloud.get_pointer(sl.MEM.GPU)),
                        width=int(tracking_buffer.point_cloud.get_width()),
                        height=int(tracking_buffer.point_cloud.get_height()),
                        step_bytes=int(tracking_buffer.point_cloud.get_step_bytes(sl.MEM.GPU)),
                        confidence_pointer=(
                            int(tracking_buffer.confidence_map.get_pointer(sl.MEM.GPU))
                            if confidence_enabled
                            else 0
                        ),
                        confidence_step_bytes=(
                            int(tracking_buffer.confidence_map.get_step_bytes(sl.MEM.GPU))
                            if confidence_enabled
                            else 0
                        ),
                        owner=tracking_buffer.point_cloud,
                        confidence_owner=tracking_buffer.confidence_map if confidence_enabled else None,
                    )
                    tracking_buffer.in_use = True
                    submitted = self.async_worker.submit(
                        AsyncTrackingFrame(
                            frame_index=int(frame_index),
                            timestamp_s=float(loop_start),
                            bgr=np.ascontiguousarray(bgr.copy()),
                            point_cloud=point_cloud_view,
                            confidence_measure=None,
                            release_callback=lambda buffer=tracking_buffer: setattr(buffer, "in_use", False),
                        )
                    )
                    if not submitted:
                        tracking_buffer.in_use = False
                copy_seconds = time.monotonic() - stage_start
                loop_seconds = time.monotonic() - loop_start

                with self.lock:
                    self.frame_count += 1
                    self.latest_bgr = bgr
                    if update_cloud:
                        self.latest_vertices = vertices
                        self.latest_stats = point_stats
                    if completed_result is not None:
                        self.latest_result = completed_result
                        self.worker_seconds += float(completed_result.worker_seconds)
                        self.worker_frames += 1
                        for key in self.worker_stage_seconds:
                            self.worker_stage_seconds[key] += float(
                                completed_result.worker_stage_seconds.get(key, 0.0)
                            )
                    self.capture_loop_seconds += loop_seconds
                    self.capture_stage_seconds["capture"] += capture_seconds
                    self.capture_stage_seconds["cloud"] += cloud_seconds
                    self.capture_stage_seconds["copy"] += copy_seconds
        except Exception as exc:
            formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            with self.lock:
                self.error = formatted


def show_viewer_startup_status(viewer, status):
    viewer.update_vertices(np.empty((0, 6), dtype=np.float32), str(status))
    return viewer.poll()


def run_live_parallel(args):
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
    show_viewer_startup_status(viewer, "Starting particle-filter viewer | loading PIDNet on CUDA...")

    zed = None
    async_worker = None
    capture = None
    try:
        cable_detector = load_cable_detector(args)
        show_viewer_startup_status(viewer, "PIDNet ready | creating particle filters...")
        particle_filters = (
            [
                CableParticleFilter(
                    node_count=args.cable_segments + 1,
                    config=make_particle_filter_config(args, cable_index),
                    seed=17 + cable_index,
                )
                for cable_index in range(args.cable_count)
            ]
            if args.particle_filter
            else []
        )
        show_viewer_startup_status(viewer, "PIDNet ready | opening ZED camera...")
        zed = open_zed(args)
        args.camera_intrinsics = camera_intrinsics_from_zed(zed)
        runtime = make_runtime_parameters(args)
        configure_viewer_from_zed(zed, viewer)
        show_viewer_startup_status(viewer, "ZED ready | starting tracking workers...")
        async_worker = AsyncTrackingWorker(args, cable_detector, particle_filters)
        capture = LiveCaptureWorker(args, zed, runtime, async_worker)
        capture.start()
    except Exception as exc:
        message = f"STARTUP FAILED: {type(exc).__name__}: {exc}"
        print(message)
        show_viewer_startup_status(viewer, message)
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and viewer.poll():
            time.sleep(0.02)
        if async_worker is not None:
            async_worker.close()
        viewer.close()
        if zed is not None:
            zed.close()
        raise

    print(
        "Parallel pipeline enabled: ZED capture/submission, tracking, and OpenGL UI run independently. "
        "Stats report GUI, CAPTURE, and TRACK FPS separately."
    )

    latest_detection = None
    latest_endpoint_mask = None
    latest_detection_label_mask = None
    latest_endpoint_label_mask = None
    latest_endpoint_overlap_mask = None
    latest_endpoint_observations = None
    latest_crossing_mask = None
    latest_crossing_proposals = tuple()
    latest_contact_observations = tuple()
    last_visual_measurement = None
    last_visual_estimate = None
    last_visual_filter_result = None
    latest_tracking_diagnostics = {}
    latest_cable_status = "waiting for cable"
    last_processed_completed = 0
    last_ui_capture_frame = -int(args.viewer_update_every)
    visual_update_count = 0
    ui_seconds = 0.0
    last_stats_time = time.monotonic()
    previous = capture.snapshot()
    previous_visual_updates = 0
    previous_ui_seconds = 0.0

    try:
        while viewer.is_available():
            snapshot = capture.snapshot()
            if snapshot.completed > last_processed_completed and snapshot.result is not None:
                result = snapshot.result
                latest_detection = result.detection
                latest_endpoint_mask = result.endpoint_mask
                latest_detection_label_mask = result.detection_label_mask
                latest_endpoint_label_mask = result.endpoint_label_mask
                latest_endpoint_overlap_mask = result.endpoint_overlap_mask
                latest_endpoint_observations = result.endpoint_observations
                latest_crossing_mask = result.crossing_mask
                latest_crossing_proposals = result.crossing_proposals
                latest_contact_observations = result.contact_observations
                latest_tracking_diagnostics = result.tracking_diagnostics
                last_visual_measurement = result.measurement
                last_visual_estimate = result.estimate
                last_visual_filter_result = result.filter_result
                last_processed_completed = snapshot.completed

            if (
                snapshot.bgr is not None
                and snapshot.frame_count - last_ui_capture_frame >= int(args.viewer_update_every)
            ):
                ui_start = time.monotonic()
                debug_bgr = draw_cable_rgb_panel(
                    snapshot.bgr,
                    detection=latest_detection,
                    endpoint_mask=latest_endpoint_mask,
                    detection_label_mask=latest_detection_label_mask,
                    endpoint_label_mask=latest_endpoint_label_mask,
                    measurement=last_visual_measurement,
                    estimate=last_visual_estimate,
                    segment_count=args.cable_segments,
                    cable_count=args.cable_count,
                    mode=args.rgb_view,
                    detector_description=detector_description(args),
                    endpoint_overlap_mask=latest_endpoint_overlap_mask,
                    tracking_diagnostics=latest_tracking_diagnostics,
                    endpoint_observations=latest_endpoint_observations,
                    crossing_mask=latest_crossing_mask,
                    crossing_proposals=latest_crossing_proposals,
                    contact_observations=latest_contact_observations,
                )
                update_viewer_cable(
                    viewer,
                    last_visual_measurement,
                    last_visual_estimate,
                    last_visual_filter_result,
                    args.cable_max_points,
                    contact_observations=latest_contact_observations,
                )
                latest_cable_status = cable_status(
                    latest_detection,
                    last_visual_measurement,
                    last_visual_estimate,
                    last_visual_filter_result,
                    args.cable_segments,
                    detector_name="pidnet",
                    diagnostics=latest_tracking_diagnostics,
                )
                viewer.update_rgb_image(cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB))
                viewer.update_vertices(
                    snapshot.vertices,
                    f"live ZED point cloud | frame {snapshot.frame_count} | "
                    f"{len(snapshot.vertices)} points | {latest_cable_status}",
                )
                last_ui_capture_frame = snapshot.frame_count
                visual_update_count += 1
                ui_seconds += time.monotonic() - ui_start

            now = time.monotonic()
            if now - last_stats_time >= 1.0:
                elapsed = max(now - last_stats_time, 1e-6)
                capture_frames = snapshot.frame_count - previous.frame_count
                worker_frames = snapshot.worker_frames - previous.worker_frames
                visual_updates = visual_update_count - previous_visual_updates
                main_stage_ms = {
                    key: 1000.0
                    * (snapshot.capture_stage_seconds[key] - previous.capture_stage_seconds[key])
                    / max(capture_frames, 1)
                    for key in ("capture", "cloud", "copy")
                }
                main_stage_ms["ui"] = (
                    1000.0 * (ui_seconds - previous_ui_seconds) / max(visual_updates, 1)
                )
                worker_stage_ms = {
                    key: 1000.0
                    * (snapshot.worker_stage_seconds[key] - previous.worker_stage_seconds[key])
                    / max(worker_frames, 1)
                    for key in ("detect", "fit", "filter")
                }
                capture_ms = (
                    1000.0
                    * (snapshot.capture_loop_seconds - previous.capture_loop_seconds)
                    / max(capture_frames, 1)
                )
                worker_ms = (
                    1000.0 * (snapshot.worker_seconds - previous.worker_seconds) / max(worker_frames, 1)
                )
                result_age_frames = (
                    -1
                    if snapshot.result is None
                    else max(0, snapshot.frame_count - int(snapshot.result.frame_index))
                )
                result_age_ms = (
                    float("nan")
                    if snapshot.result is None
                    else 1000.0 * max(0.0, now - snapshot.result.frame_timestamp_s)
                )
                print(
                    format_runtime_status(
                        snapshot.frame_count,
                        visual_updates / elapsed,
                        capture_frames / elapsed,
                        (snapshot.completed - previous.completed) / elapsed,
                        (snapshot.submitted - previous.submitted) / elapsed,
                        capture_ms,
                        worker_ms,
                        result_age_frames,
                        result_age_ms,
                        snapshot.point_stats,
                        main_stage_ms,
                        worker_stage_ms,
                        async_worker,
                        latest_cable_status,
                    )
                )
                previous = snapshot
                previous_visual_updates = visual_update_count
                previous_ui_seconds = ui_seconds
                last_stats_time = now
            viewer.poll()
    finally:
        capture.stop()
        async_worker.close()
        viewer.close()
        capture.free()
        zed.close()


def cable_tracking_diagnostics(previous_filter_nodes, measurement, estimate, filter_result, base_diagnostics=None):
    diagnostics = dict(base_diagnostics or {})
    measurement_used = (
        bool(getattr(filter_result, "measurement_used", False))
        if filter_result is not None
        else measurement is not None
    )
    diagnostics["measurement_used"] = measurement_used
    measured_nodes = measurement_nodes_array(measurement)
    estimate_nodes = measurement_nodes_array(estimate)
    previous_nodes = measurement_nodes_array(previous_filter_nodes)
    support_owner = estimate if estimate is not None else measurement
    support_points = np.asarray(
        getattr(support_owner, "source_points", np.empty((0, 3))), dtype=np.float32
    ) if support_owner is not None else np.empty((0, 3), dtype=np.float32)
    diagnostics["support_to_prior_m"] = (
        polyline_residual(support_points, previous_nodes)
        if len(support_points) and previous_nodes is not None
        else np.nan
    )
    diagnostics["measurement_to_filter_m"] = mean_node_error(previous_nodes, measured_nodes)
    diagnostics["estimate_to_raw_m"] = (
        polyline_residual(support_points, estimate_nodes)
        if len(support_points) and estimate_nodes is not None
        else np.nan
    )
    diagnostics["raw_residual_m"] = float(getattr(measurement, "residual_m", np.nan)) if measurement is not None else np.nan
    diagnostics["filtered_residual_m"] = float(getattr(estimate, "residual_m", np.nan)) if estimate is not None else np.nan
    diagnostics["proposal_ratio"] = (
        float(getattr(filter_result, "measurement_proposal_ratio", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    diagnostics["global_random_ratio"] = (
        float(getattr(filter_result, "global_random_particle_ratio", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    ransac_hypothesis_count = (
        int(getattr(filter_result, "ransac_hypothesis_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["ransac_inlier_ratio"] = (
        float(getattr(filter_result, "ransac_inlier_ratio", np.nan))
        if filter_result is not None and ransac_hypothesis_count > 0
        else np.nan
    )
    diagnostics["ransac_inlier_count"] = (
        int(getattr(filter_result, "ransac_inlier_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["ransac_error_m"] = (
        float(getattr(filter_result, "ransac_error_m", np.nan))
        if filter_result is not None and ransac_hypothesis_count > 0
        else np.nan
    )
    diagnostics["ransac_hypotheses"] = ransac_hypothesis_count
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
    diagnostics["estimate_particle_count"] = (
        int(getattr(filter_result, "estimate_particle_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["estimate_weight_mass"] = (
        float(getattr(filter_result, "estimate_weight_mass", np.nan))
        if filter_result is not None
        else np.nan
    )
    particle_diagnostics = (
        getattr(filter_result, "particle_diagnostics", None)
        if filter_result is not None
        else None
    )
    diagnostics["map_to_average_node_error_m"] = (
        float(getattr(particle_diagnostics, "map_to_average_node_error_m", np.nan))
        if particle_diagnostics is not None
        else np.nan
    )
    diagnostics["mean_node_spread_m"] = (
        float(getattr(particle_diagnostics, "mean_node_spread_m", np.nan))
        if particle_diagnostics is not None
        else np.nan
    )
    diagnostics["max_node_spread_m"] = (
        float(getattr(particle_diagnostics, "max_node_spread_m", np.nan))
        if particle_diagnostics is not None
        else np.nan
    )
    endpoint_direction_delta = np.asarray(
        getattr(particle_diagnostics, "endpoint_direction_delta_deg", (np.nan, np.nan)),
        dtype=np.float32,
    ).reshape(-1)
    diagnostics["estimate_start_direction_delta_deg"] = (
        float(endpoint_direction_delta[0]) if len(endpoint_direction_delta) > 0 else np.nan
    )
    diagnostics["estimate_end_direction_delta_deg"] = (
        float(endpoint_direction_delta[1]) if len(endpoint_direction_delta) > 1 else np.nan
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


def update_viewer_cable(
    viewer,
    measurement,
    estimate,
    filter_result,
    max_points,
    contact_observations=(),
):
    if estimate is None:
        viewer.update_cable(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
            cable_runs=(),
            contact_observations=contact_observations,
            particle_diagnostics=(),
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
    if filter_result is not None and getattr(filter_result, "extended_visible_nodes", None) is not None:
        extended_visible_nodes = np.asarray(filter_result.extended_visible_nodes, dtype=bool)
        if len(extended_visible_nodes) != len(nodes):
            extended_visible_nodes = visible_nodes.copy()
        extended_visible_nodes &= valid_nodes
    else:
        extended_visible_nodes = visible_nodes.copy()
    source_points = np.empty((0, 3), dtype=np.float32)
    if measurement is not None:
        source_points = np.asarray(measurement.source_points, dtype=np.float32)

    viewer.update_cable(
        sample_points(source_points, max_points),
        nodes,
        valid_nodes,
        visible_nodes=visible_nodes,
        extended_visible_nodes=extended_visible_nodes,
        cable_runs=getattr(estimate, "pf_node_runs", None),
        contact_observations=contact_observations,
        particle_diagnostics=getattr(filter_result, "particle_diagnostics", ()),
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
        details.append(f"raw={source_count}")
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
        observation_model = str(diagnostics.get("endpoint_observation_model", "") or "")
        if observation_model:
            parts.append(f"obs={observation_model}")
        group_counts = str(diagnostics.get("endpoint_group_counts_text", "") or "")
        if group_counts:
            parts.append(f"endpoint_groups={group_counts}")
        association_status = str(diagnostics.get("endpoint_association_status", "") or "")
        if association_status:
            parts.append(f"pfassoc={association_status}")
        assignment = str(diagnostics.get("endpoint_assignment_text", "") or "")
        if assignment:
            parts.append(f"assignment={assignment}")
        association_margin = float(diagnostics.get("endpoint_association_margin_m", np.nan))
        if np.isfinite(association_margin):
            parts.append(f"pairmargin={format_mm(association_margin)}")
        overlap_px = int(diagnostics.get("endpoint_overlap_px", 0) or 0)
        if overlap_px > 0:
            parts.append(f"endoverlap={overlap_px}px")
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
    support_to_prior = diagnostics.get("support_to_prior_m", np.nan)
    if np.isfinite(support_to_prior):
        parts.append(f"support2prior={format_mm(support_to_prior)}")
    crossing_count = int(diagnostics.get("crossing_proposal_count", 0) or 0)
    verified_count = int(diagnostics.get("crossing_verified_count", 0) or 0)
    if crossing_count > 0:
        crossing_text = f"cross={crossing_count} contact={verified_count}"
        gap = float(diagnostics.get("crossing_gap_m", np.nan))
        confidence = float(diagnostics.get("crossing_confidence", np.nan))
        calibrated = bool(diagnostics.get("crossing_diameter_calibrated", False))
        if np.isfinite(gap):
            crossing_text += f" {'gap' if calibrated else 'center'}={format_mm(gap)}"
        if np.isfinite(confidence):
            crossing_text += f" conf={confidence:.2f}"
        if not calibrated:
            crossing_text += " diameter=UNSET"
        depth_order = str(diagnostics.get("crossing_depth_order", "") or "")
        if depth_order:
            crossing_text += f" {depth_order.replace(' ', '_')}"
        parts.append(crossing_text)
    proposal_ratio = diagnostics.get("proposal_ratio", np.nan)
    if np.isfinite(proposal_ratio):
        parts.append(f"prop={proposal_ratio:.2f}")
    random_ratio = diagnostics.get("global_random_ratio", np.nan)
    if np.isfinite(random_ratio):
        parts.append(f"rand={random_ratio:.2f}")
    estimate_count = int(diagnostics.get("estimate_particle_count", 0) or 0)
    estimate_mass = float(diagnostics.get("estimate_weight_mass", np.nan))
    if estimate_count > 0:
        estimate_text = f"topavg={estimate_count}"
        if np.isfinite(estimate_mass):
            estimate_text += f" mass={estimate_mass:.2f}"
        parts.append(estimate_text)
    map_average_error = float(diagnostics.get("map_to_average_node_error_m", np.nan))
    mean_spread = float(diagnostics.get("mean_node_spread_m", np.nan))
    max_spread = float(diagnostics.get("max_node_spread_m", np.nan))
    if np.isfinite(map_average_error):
        parts.append(f"MAP-AVG={format_mm(map_average_error)}")
    if np.isfinite(mean_spread):
        spread_text = f"spread={format_mm(mean_spread)}"
        if np.isfinite(max_spread):
            spread_text += f"/{format_mm(max_spread)}max"
        parts.append(spread_text)
    start_direction_delta = float(diagnostics.get("estimate_start_direction_delta_deg", np.nan))
    end_direction_delta = float(diagnostics.get("estimate_end_direction_delta_deg", np.nan))
    if np.isfinite(start_direction_delta) or np.isfinite(end_direction_delta):
        start_text = f"{start_direction_delta:.1f}" if np.isfinite(start_direction_delta) else "nan"
        end_text = f"{end_direction_delta:.1f}" if np.isfinite(end_direction_delta) else "nan"
        parts.append(f"MAP-AVG-dir={start_text}/{end_text}deg")
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
    endpoint_overlap_mask=None,
    tracking_diagnostics=None,
    endpoint_observations=None,
    crossing_mask=None,
    crossing_proposals=None,
    contact_observations=None,
):
    if mode == "mask":
        panel = draw_cable_mask_view(
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
    elif mode == "tracking":
        panel = draw_cable_debug_overlay(
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
    else:
        panel = draw_cable_segmentation_view(
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
    draw_raw_endpoint_observations(panel, endpoint_observations)
    draw_endpoint_overlap_overlay(panel, endpoint_overlap_mask)
    draw_crossing_observations(
        panel,
        crossing_mask,
        crossing_proposals,
        contact_observations,
    )
    draw_endpoint_association_status(panel, tracking_diagnostics)
    return panel


SEGMENTATION_LABEL_COLORS_BGR = (
    (40, 255, 80),     # cable body
    (255, 0, 255),     # endpoints_cable1
    (255, 220, 0),     # endpoints_cable2
)


def segmentation_label_color(label, cable_count=None):
    label = max(1, int(label))
    return SEGMENTATION_LABEL_COLORS_BGR[(label - 1) % len(SEGMENTATION_LABEL_COLORS_BGR)]


def draw_labeled_mask_overlay(panel, label_mask, alpha=0.70, contour_thickness=1, cable_count=None):
    if label_mask is None:
        return False
    labels = np.asarray(label_mask, dtype=np.uint8)
    if labels.ndim != 2 or not np.any(labels):
        return False
    if labels.shape[:2] != panel.shape[:2]:
        labels = cv2.resize(labels, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        binary = (labels == label).astype(np.uint8) * 255
        x, y, width, height = cv2.boundingRect(binary)
        if width <= 0 or height <= 0:
            continue
        color = segmentation_label_color(label, cable_count=cable_count)
        roi = panel[y:y + height, x:x + width]
        mask_roi = binary[y:y + height, x:x + width]
        tint = np.empty_like(roi)
        tint[:] = color
        blended = cv2.addWeighted(roi, 1.0 - float(alpha), tint, float(alpha), 0.0)
        cv2.copyTo(blended, mask_roi, roi)
        if int(contour_thickness) > 0:
            contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    pf_ids = np.asarray(
        getattr(measurement, "endpoint_marker_pf_ids", np.empty(0, dtype=np.int16)),
        dtype=np.int16,
    ).reshape(-1)
    end_indices = np.asarray(
        getattr(measurement, "endpoint_marker_end_indices", np.empty(0, dtype=np.int8)),
        dtype=np.int8,
    ).reshape(-1)
    for index, center in enumerate(centers):
        if not np.all(np.isfinite(center)):
            continue
        point = tuple(np.round(center).astype(np.int32))
        pf_id = int(pf_ids[index]) if index < len(pf_ids) else index // 2
        end_index = int(end_indices[index]) if index < len(end_indices) else index % 2
        color = SEGMENTATION_LABEL_COLORS_BGR[1 + (pf_id % 2)]
        outline = (255, 255, 255)
        cv2.circle(panel, point, 9, color, -1, cv2.LINE_AA)
        cv2.circle(panel, point, 12, outline, 2, cv2.LINE_AA)
        end_text = "start" if end_index == 0 else "end" if end_index == 1 else "endpoint"
        label = f"PF{pf_id + 1} {end_text}"
        label_point = (int(point[0] + 15), int(point[1] - 10 if end_index == 0 else point[1] + 20))
        cv2.putText(panel, label, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(panel, label, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def draw_endpoint_overlap_overlay(panel, endpoint_overlap_mask):
    if endpoint_overlap_mask is None:
        return
    mask = np.asarray(endpoint_overlap_mask, dtype=np.uint8)
    if mask.ndim != 2 or not np.any(mask):
        return
    if mask.shape[:2] != panel.shape[:2]:
        mask = cv2.resize(mask, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (0, 255, 255), 3, cv2.LINE_AA)
    for contour in contours:
        x, y, _width, _height = cv2.boundingRect(contour)
        cv2.putText(
            panel,
            "ENDPOINT CLASS OVERLAP",
            (max(4, int(x)), max(18, int(y) - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_raw_endpoint_observations(panel, endpoint_observations):
    for cable_index, candidate_index, x, y in endpoint_observations or ():
        point = (int(round(x)), int(round(y)))
        color = SEGMENTATION_LABEL_COLORS_BGR[1 + (int(cable_index) % 2)]
        cv2.drawMarker(panel, point, color, cv2.MARKER_DIAMOND, 18, 2, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"NN endpoints_{int(cable_index) + 1}.{int(candidate_index) + 1}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_crossing_observations(panel, crossing_mask, crossing_proposals, contact_observations):
    """Draw NN proposals and their PF/3D verification without conflating them."""

    if crossing_mask is not None:
        mask = np.asarray(crossing_mask, dtype=np.uint8)
        if mask.ndim == 2 and np.any(mask):
            if mask.shape != panel.shape[:2]:
                mask = cv2.resize(mask, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
            x, y, width, height = cv2.boundingRect(mask)
            if width > 0 and height > 0:
                roi = panel[y:y + height, x:x + width]
                mask_roi = mask[y:y + height, x:x + width]
                tint = np.empty_like(roi)
                tint[:] = (0, 210, 255)
                blended = cv2.addWeighted(roi, 0.58, tint, 0.42, 0.0)
                cv2.copyTo(blended, mask_roi, roi)

    contacts = {
        int(observation.proposal.proposal_id): observation
        for observation in tuple(contact_observations or ())
    }
    for proposal in tuple(crossing_proposals or ()):
        observation = contacts.get(int(proposal.proposal_id))
        verified = bool(observation is not None and observation.verified_contact)
        color = (40, 255, 80) if verified else (255, 210, 30)
        x, y, width, height = [int(value) for value in proposal.bbox_xywh]
        centroid = tuple(np.round(proposal.centroid_xy).astype(np.int32))
        cv2.rectangle(panel, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
        cv2.drawMarker(panel, centroid, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        state = "3D CONTACT" if verified else "RGB CROSSING"
        details = f"{state} p={proposal.mean_probability:.2f}"
        if observation is not None:
            for cable_index, segment_points in enumerate(observation.segment_image_points):
                points = np.round(segment_points).astype(np.int32)
                if points.shape == (2, 2) and np.all(np.isfinite(segment_points)):
                    segment_color = (255, 0, 255) if cable_index == 0 else (255, 220, 0)
                    cv2.line(panel, tuple(points[0]), tuple(points[1]), segment_color, 4, cv2.LINE_AA)
            if observation.diameter_calibrated:
                details += f" gap={1000.0 * observation.gap_m:.1f}mm"
            else:
                details += f" center={1000.0 * observation.centerline_distance_m:.1f}mm DIAMETER UNSET"
            details += (
                f" s=({observation.s1_m:.3f},{observation.s2_m:.3f})m "
                f"g={observation.confidence:.2f} {observation.depth_order}"
            )
        text_y = min(panel.shape[0] - 8, max(18, y + height + 18))
        cv2.putText(panel, details, (max(4, x), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(panel, details, (max(4, x), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_endpoint_association_status(panel, diagnostics):
    diagnostics = dict(diagnostics or {})
    group_counts = str(diagnostics.get("endpoint_group_counts_text", "") or "")
    status = str(diagnostics.get("endpoint_association_status", "") or "")
    assignment = str(diagnostics.get("endpoint_assignment_text", "") or "")
    if not (group_counts or status or assignment):
        return
    text = f"NN endpoints {group_counts} | fixed cable identity:{status}"
    if assignment:
        text += f" | {assignment}"
    unhealthy = any(
        token in text
        for token in ("ambiguous", "incomplete", "unreachable")
    )
    color = (0, 215, 255) if unhealthy else (80, 255, 120)
    baseline_y = min(panel.shape[0] - 12, 102)
    text_width = min(panel.shape[1] - 16, max(220, 9 * len(text)))
    cv2.rectangle(panel, (8, baseline_y - 19), (8 + text_width, baseline_y + 7), (15, 18, 22), -1)
    cv2.putText(panel, text, (16, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def draw_cable_status_text(panel, detection, measurement, estimate, segment_count, mode_text):
    cv2.putText(panel, mode_text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if detection is None:
        text = "waiting for cable detection"
    elif measurement is not None:
        text = f"{segment_count} segments | residual {measurement.residual_m:.4f}m"
    elif estimate is not None:
        text = f"{segment_count} segments | prediction only"
    else:
        text = f"{segment_count} segments | mask components {detection.component_count}"
    cv2.putText(panel, text, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 245, 255), 2, cv2.LINE_AA)


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
