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
    cable_measurement_from_support_points,
    endpoint_group_observations_from_mask,
    polyline_residual,
    sampled_masked_point_cloud_point_sets,
)
from cable_crossing import (
    CameraIntrinsics,
    assign_crossing_targets,
    estimate_crossing_axes,
    extract_crossing_proposals,
)
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    PerCableEndpointAssociationConfig,
    PerCableEndpointAssociator,
    filtered_cable_estimate,
    update_cable_particle_filters,
)
from zed_spatial import (
    DEPTH_MODES,
    RESOLUTIONS,
    configure_input_source,
    configure_viewer_from_zed,
    live_point_cloud_to_vertices,
)
from zed_split_viewer import ZedDepthGLViewer
from pidnet_schema import CROSSING_CHANNEL, ENDPOINT_CHANNELS, OUTPUT_CHANNEL_COUNT
from pf_ablation import (
    AblationControlPanel,
    ExperimentRecorder,
    RuntimeFeatureController,
    apply_feature_state_to_args,
    feature_state_from_args,
    file_sha256,
    validate_feature_state,
)


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
    parser.add_argument(
        "--ablation-control-ui",
        action=argparse.BooleanOptionalAction,
        default=config_value(config, "viewer", "ablation_control_ui", True),
    )
    parser.add_argument(
        "--experiment-output-directory",
        type=Path,
        default=config_value(
            config,
            "experiment",
            "output_directory",
            PROJECT_DIR / "experiments",
            config_base_dir,
        ),
    )
    parser.add_argument("--neural-detector-checkpoint", type=Path, default=config_value(config, "pidnet", "checkpoint", DEFAULT_PIDNET_CHECKPOINT, config_base_dir), help="PIDNet cable segmentation checkpoint.")
    parser.add_argument("--neural-detector-device", default=config_value(config, "pidnet", "device", "cuda"), help="PyTorch device for PIDNet detector. Default: cuda.")
    parser.add_argument("--neural-detector-threshold", type=float, default=config_value(config, "pidnet", "threshold", 0.50), help="PIDNet probability threshold for the binary cable mask.")
    parser.add_argument(
        "--neural-detector-endpoint-thresholds",
        type=float,
        nargs="+",
        default=config_value(config, "pidnet", "endpoint_thresholds", None),
        help="Per-cable thresholds for the endpoints_cable1 and endpoints_cable2 heads.",
    )
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
    parser.add_argument("--endpoint-association-support", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "endpoint_association_support", True))
    parser.add_argument("--cable-mask-morphology", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "cable_mask_morphology", True))
    parser.add_argument("--cable-component-filter", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "cable_component_filter", True))
    parser.add_argument("--endpoint-mask-morphology", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "endpoint_mask_morphology", True))
    parser.add_argument("--endpoint-component-filter", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "endpoint_component_filter", True))
    parser.add_argument("--depth-confidence-filter", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "depth_confidence_filter", True))
    parser.add_argument("--crossing-threshold", type=float, default=config_value(config, "crossing", "threshold", 0.50))
    parser.add_argument("--crossing-min-area", type=int, default=config_value(config, "crossing", "min_area_px", 12))
    parser.add_argument("--crossing-max-proposals", type=int, default=config_value(config, "crossing", "max_proposals", 4))
    parser.add_argument("--crossing-max-visual-points", type=int, default=config_value(config, "crossing", "max_visual_points", 512))
    parser.add_argument("--crossing-axis-radius", type=float, default=config_value(config, "crossing", "axis_radius_px", 36.0))
    parser.add_argument("--crossing-min-axis-separation", type=float, default=config_value(config, "crossing", "min_axis_separation_deg", 25.0))
    parser.add_argument("--crossing-min-axis-support", type=int, default=config_value(config, "crossing", "min_axis_support_px", 12))
    parser.add_argument("--crossing-position-sigma", type=float, default=config_value(config, "crossing", "position_sigma_px", 14.0))
    parser.add_argument("--crossing-angle-sigma", type=float, default=config_value(config, "crossing", "angle_sigma_deg", 20.0))
    parser.add_argument("--crossing-log-reward", type=float, default=config_value(config, "crossing", "log_reward", 4.0))
    parser.add_argument("--crossing-proposals", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "crossing_proposals", True))
    parser.add_argument("--crossing-likelihood", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "crossing_likelihood", True))
    parser.add_argument("--detector-min-area", type=int, default=config_value(config, "detector", "min_area_px", 80))
    parser.add_argument("--detector-open-kernel", type=int, default=config_value(config, "detector", "open_kernel", 3))
    parser.add_argument("--detector-close-kernel", type=int, default=config_value(config, "detector", "close_kernel", 5))
    parser.add_argument("--cable-segments", type=int, default=config_value(config, "cable", "segments", 2))
    parser.add_argument("--cable-count", type=int, default=config_value(config, "cable", "count", 1), help="Number of separate cables to reconstruct.")
    parser.add_argument("--cable-lengths", type=float, nargs="+", default=config_value(config, "cable", "lengths_m", None), help="Physical cable lengths in meters, one value per cable.")
    parser.add_argument("--cable-max-points", type=int, default=config_value(config, "cable", "max_visual_points", 1000))
    parser.add_argument("--cable-confidence-max", type=float, default=config_value(config, "measurement", "confidence_max", 85.0), help="Use <0 to disable.")
    parser.add_argument("--particle-filter", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "particle_filter", True))
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
        help="Choose a posterior-medoid representative from the highest-weight N particles.",
    )
    parser.add_argument("--pf-process-std", type=float, default=config_value(config, "particle_filter", "process_node_std_m", pf_defaults.process_node_std_m))
    parser.add_argument("--pf-process-direction-std", type=float, default=config_value(config, "particle_filter", "process_direction_std", pf_defaults.process_direction_std))
    parser.add_argument("--pf-velocity", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "velocity_prediction", pf_defaults.velocity_enabled))
    parser.add_argument("--pf-adaptive-motion", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "adaptive_motion_noise", True))
    parser.add_argument("--pf-occlusion-prediction", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "occlusion_prediction", True))
    parser.add_argument("--pf-direction-smoothing", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "direction_smoothing", False))
    parser.add_argument("--pf-posterior-medoid", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "posterior_medoid_estimate", True))
    parser.add_argument("--pf-endpoint-tangent", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "endpoint_tangent_estimation", True))
    parser.add_argument("--pf-endpoint-tangent-ransac", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "endpoint_tangent_ransac", True))
    parser.add_argument("--pf-endpoint-tangent-likelihood", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "endpoint_tangent_likelihood", True))
    parser.add_argument("--pf-conditioned-proposals", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "endpoint_conditioned_proposals", True))
    parser.add_argument("--pf-global-random-particles", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "global_random_particles", True))
    parser.add_argument("--pf-robust-measurement", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "robust_measurement_likelihood", True))
    parser.add_argument("--pf-dense-path-support", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "dense_path_support", True))
    parser.add_argument("--pf-union-coverage", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "union_coverage_selection", True))
    parser.add_argument("--pf-bend-regularization", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "bend_regularization", False))
    parser.add_argument("--point-support-coloring", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "point_support_coloring", True))
    parser.add_argument("--particle-diagnostics-overlay", action=argparse.BooleanOptionalAction, default=config_value(config, "features", "particle_diagnostics_overlay", True))
    parser.add_argument("--pf-velocity-damping", type=float, default=config_value(config, "particle_filter", "velocity_damping", pf_defaults.velocity_damping))
    parser.add_argument("--pf-velocity-measurement-blend", type=float, default=config_value(config, "particle_filter", "velocity_measurement_blend", pf_defaults.velocity_measurement_blend))
    parser.add_argument("--pf-velocity-process-std", type=float, default=config_value(config, "particle_filter", "velocity_process_std_mps", pf_defaults.velocity_process_std_mps))
    parser.add_argument("--pf-max-node-speed", type=float, default=config_value(config, "particle_filter", "max_node_speed_mps", pf_defaults.max_node_speed_mps))
    parser.add_argument("--pf-motion-speed-reference", type=float, default=config_value(config, "particle_filter", "motion_speed_reference_mps", pf_defaults.motion_speed_reference_mps))
    parser.add_argument("--pf-motion-innovation-reference", type=float, default=config_value(config, "particle_filter", "motion_innovation_reference_m", pf_defaults.motion_innovation_reference_m))
    parser.add_argument("--pf-motion-noise-adaptation", type=float, default=config_value(config, "particle_filter", "motion_noise_adaptation", pf_defaults.motion_noise_adaptation))
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
    parser.add_argument("--pf-tangent-min-radius", type=float, default=config_value(config, "particle_filter", "endpoint_tangent_min_radius_m", pf_defaults.endpoint_tangent_min_radius_m))
    parser.add_argument("--pf-tangent-radius", type=float, default=config_value(config, "particle_filter", "endpoint_tangent_radius_m", pf_defaults.endpoint_tangent_radius_m))
    parser.add_argument("--pf-tangent-sigma", type=float, default=config_value(config, "particle_filter", "endpoint_tangent_sigma_m", pf_defaults.endpoint_tangent_sigma_m))
    parser.add_argument("--pf-tangent-min-points", type=int, default=config_value(config, "particle_filter", "endpoint_tangent_min_points", pf_defaults.endpoint_tangent_min_points))
    parser.add_argument("--pf-tangent-min-confidence", type=float, default=config_value(config, "particle_filter", "endpoint_tangent_min_confidence", pf_defaults.endpoint_tangent_min_confidence))
    parser.add_argument("--pf-tangent-ransac-inlier", type=float, default=config_value(config, "particle_filter", "endpoint_tangent_ransac_inlier_m", pf_defaults.endpoint_tangent_ransac_inlier_m))
    parser.add_argument("--pf-tangent-max-hypotheses", type=int, default=config_value(config, "particle_filter", "endpoint_tangent_max_hypotheses", pf_defaults.endpoint_tangent_max_hypotheses))
    parser.add_argument("--pf-tangent-reference-weight", type=float, default=config_value(config, "particle_filter", "endpoint_tangent_reference_weight", pf_defaults.endpoint_tangent_reference_weight))
    parser.add_argument("--pf-tangent-likelihood-scale", type=float, default=config_value(config, "particle_filter", "endpoint_tangent_likelihood_scale_m", pf_defaults.endpoint_tangent_likelihood_scale_m))
    parser.add_argument("--pf-local-proposal-ratio", type=float, default=config_value(config, "particle_filter", "local_proposal_ratio", pf_defaults.local_proposal_ratio))
    parser.add_argument("--pf-conditioned-proposal-ratio", type=float, default=config_value(config, "particle_filter", "endpoint_conditioned_proposal_ratio", pf_defaults.endpoint_conditioned_proposal_ratio))
    parser.add_argument("--pf-conditioned-direction-std", type=float, default=config_value(config, "particle_filter", "endpoint_conditioned_direction_std", pf_defaults.endpoint_conditioned_direction_std))
    parser.add_argument("--pf-conditioned-deformation-std", type=float, default=config_value(config, "particle_filter", "endpoint_conditioned_deformation_std_m", pf_defaults.endpoint_conditioned_deformation_std_m))
    parser.add_argument("--pf-conditioned-slack-gain", type=float, default=config_value(config, "particle_filter", "endpoint_conditioned_slack_gain", pf_defaults.endpoint_conditioned_slack_gain))
    parser.add_argument("--pf-conditioned-max-deformation", type=float, default=config_value(config, "particle_filter", "endpoint_conditioned_max_deformation_m", pf_defaults.endpoint_conditioned_max_deformation_m))
    parser.add_argument("--pf-conditioned-deformation-modes", type=int, default=config_value(config, "particle_filter", "endpoint_conditioned_deformation_modes", pf_defaults.endpoint_conditioned_deformation_modes))
    parser.add_argument("--pf-robust-distance", type=float, default=config_value(config, "particle_filter", "robust_distance_m", pf_defaults.robust_distance_m))
    parser.add_argument("--pf-path-support-weight", type=float, default=config_value(config, "particle_filter", "path_support_weight", pf_defaults.path_support_weight))
    parser.add_argument("--pf-path-support-samples", type=int, default=config_value(config, "particle_filter", "path_support_samples_per_segment", pf_defaults.path_support_samples_per_segment))
    parser.add_argument("--pf-union-coverage-top-particles", type=int, default=config_value(config, "particle_filter", "union_coverage_top_particles", pf_defaults.union_coverage_top_particle_count))
    parser.add_argument("--pf-union-coverage-weight", type=float, default=config_value(config, "particle_filter", "union_coverage_weight", pf_defaults.union_coverage_weight))
    parser.add_argument("--pf-bend-penalty", type=float, default=config_value(config, "particle_filter", "bend_penalty_m", pf_defaults.bend_penalty_m))
    parser.add_argument("--pf-global-random-ratio", type=float, default=config_value(config, "particle_filter", "global_random_particle_ratio", pf_defaults.global_random_particle_ratio))
    parser.add_argument("--pf-endpoint-constraint-iterations", type=int, default=config_value(config, "particle_filter", "endpoint_constraint_iterations", pf_defaults.endpoint_constraint_iterations))
    parser.add_argument("--pf-endpoint-constraint-tolerance", type=float, default=config_value(config, "particle_filter", "endpoint_constraint_tolerance_m", pf_defaults.endpoint_constraint_tolerance_m))
    parser.add_argument("--pf-min-measurement-points", type=int, default=config_value(config, "particle_filter", "min_measurement_points", pf_defaults.min_measurement_points))
    parser.add_argument("--pf-min-segment-support-samples", type=int, default=config_value(config, "particle_filter", "min_segment_support_samples", pf_defaults.min_segment_support_samples))
    parser.add_argument("--pf-support-visibility-distance", type=float, default=config_value(config, "particle_filter", "support_visibility_distance_m", pf_defaults.support_visibility_distance_m))
    parser.add_argument("--pf-max-prediction-frames", type=int, default=config_value(config, "particle_filter", "max_prediction_frames", pf_defaults.max_prediction_frames))
    parser.add_argument("--pf-max-motion-noise-scale", type=float, default=config_value(config, "particle_filter", "max_motion_noise_scale", pf_defaults.max_motion_noise_scale))
    args = parser.parse_args()
    if args.input_svo_file and args.ip_address:
        raise ValueError("Specify only one input source: --input-svo-file or --ip-address.")
    args.cable_segments = max(1, int(args.cable_segments))
    args.cable_count = int(args.cable_count)
    if args.cable_count not in (1, 2):
        raise ValueError(f"The live tracker supports one or two physical cables; got {args.cable_count}.")
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
    args.neural_detector_threshold = float(np.clip(args.neural_detector_threshold, 0.0, 1.0))
    args.neural_detector_endpoint_thresholds = required_per_cable_float_values(
        args.neural_detector_endpoint_thresholds,
        len(ENDPOINT_CHANNELS),
        "pidnet.endpoint_thresholds",
        minimum=0.0,
    )
    args.neural_detector_endpoint_thresholds = [
        float(np.clip(value, 0.0, 1.0))
        for value in args.neural_detector_endpoint_thresholds
    ]
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
    args.crossing_max_visual_points = max(1, int(args.crossing_max_visual_points))
    args.crossing_axis_radius = max(4.0, float(args.crossing_axis_radius))
    args.crossing_min_axis_separation = float(np.clip(args.crossing_min_axis_separation, 1.0, 89.0))
    args.crossing_min_axis_support = max(2, int(args.crossing_min_axis_support))
    args.crossing_position_sigma = max(1e-3, float(args.crossing_position_sigma))
    args.crossing_angle_sigma = max(1e-3, float(args.crossing_angle_sigma))
    args.crossing_log_reward = max(0.0, float(args.crossing_log_reward))
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
    args.pf_velocity_damping = float(np.clip(args.pf_velocity_damping, 0.0, 1.0))
    args.pf_velocity_measurement_blend = float(np.clip(args.pf_velocity_measurement_blend, 0.0, 1.0))
    args.pf_velocity_process_std = max(0.0, float(args.pf_velocity_process_std))
    args.pf_max_node_speed = max(0.0, float(args.pf_max_node_speed))
    args.pf_motion_speed_reference = max(1e-4, float(args.pf_motion_speed_reference))
    args.pf_motion_innovation_reference = max(1e-5, float(args.pf_motion_innovation_reference))
    args.pf_motion_noise_adaptation = float(np.clip(args.pf_motion_noise_adaptation, 0.0, 1.0))
    args.pf_tangent_min_radius = max(0.0, float(args.pf_tangent_min_radius))
    args.pf_tangent_radius = max(args.pf_tangent_min_radius + 1e-4, float(args.pf_tangent_radius))
    args.pf_tangent_sigma = max(1e-4, float(args.pf_tangent_sigma))
    args.pf_tangent_min_points = max(2, int(args.pf_tangent_min_points))
    args.pf_tangent_min_confidence = float(np.clip(args.pf_tangent_min_confidence, 0.0, 1.0))
    args.pf_tangent_ransac_inlier = max(1e-4, float(args.pf_tangent_ransac_inlier))
    args.pf_tangent_max_hypotheses = max(4, int(args.pf_tangent_max_hypotheses))
    args.pf_tangent_reference_weight = max(0.0, float(args.pf_tangent_reference_weight))
    args.pf_tangent_likelihood_scale = max(0.0, float(args.pf_tangent_likelihood_scale))
    proposal_ratios = np.asarray((
        args.pf_local_proposal_ratio,
        args.pf_conditioned_proposal_ratio,
        args.pf_global_random_ratio,
    ), dtype=np.float64)
    if not np.all(np.isfinite(proposal_ratios)) or np.any(proposal_ratios < 0.0):
        raise ValueError("PF proposal ratios must be finite and non-negative.")
    if not np.isclose(float(np.sum(proposal_ratios)), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            "particle_filter local_proposal_ratio, endpoint_conditioned_proposal_ratio, "
            "and global_random_particle_ratio must sum to 1."
        )
    args.pf_local_proposal_ratio = float(proposal_ratios[0])
    args.pf_conditioned_proposal_ratio = float(proposal_ratios[1])
    args.pf_global_random_ratio = float(proposal_ratios[2])
    args.pf_conditioned_direction_std = max(0.0, float(args.pf_conditioned_direction_std))
    args.pf_conditioned_deformation_std = max(0.0, float(args.pf_conditioned_deformation_std))
    args.pf_conditioned_slack_gain = max(0.0, float(args.pf_conditioned_slack_gain))
    args.pf_conditioned_max_deformation = max(1e-4, float(args.pf_conditioned_max_deformation))
    args.pf_conditioned_deformation_modes = max(1, int(args.pf_conditioned_deformation_modes))
    args.pf_robust_distance = max(1e-4, float(args.pf_robust_distance))
    args.pf_path_support_weight = max(0.0, float(args.pf_path_support_weight))
    args.pf_path_support_samples = max(1, int(args.pf_path_support_samples))
    args.pf_union_coverage_top_particles = int(np.clip(
        args.pf_union_coverage_top_particles,
        1,
        args.pf_particles,
    ))
    args.pf_union_coverage_weight = max(0.0, float(args.pf_union_coverage_weight))
    args.pf_min_segment_support_samples = max(
        1,
        min(int(args.pf_min_segment_support_samples), args.pf_path_support_samples),
    )
    args.pf_support_visibility_distance = max(
        1e-4,
        float(args.pf_support_visibility_distance),
    )
    args.pf_endpoint_constraint_iterations = max(1, int(args.pf_endpoint_constraint_iterations))
    args.pf_endpoint_constraint_tolerance = max(0.0, float(args.pf_endpoint_constraint_tolerance))
    validate_feature_contract(args)
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
        z_axis_forward=False,
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
        min_area=int(args.detector_min_area) if bool(args.cable_component_filter) else 1,
        open_kernel=int(args.detector_open_kernel) if bool(args.cable_mask_morphology) else 1,
        close_kernel=int(args.detector_close_kernel) if bool(args.cable_mask_morphology) else 1,
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
    endpoint_text = ",".join(
        f"{float(value):.2f}" for value in args.neural_detector_endpoint_thresholds
    )
    return (
        f"PIDNet thresholds body={float(args.neural_detector_threshold):.2f} "
        f"endpoints={endpoint_text} crossing={float(args.crossing_threshold):.2f}"
    )


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
        open_kernel=(args.endpoint_marker_open_kernel if bool(args.endpoint_mask_morphology) else 0),
        close_kernel=(args.endpoint_marker_close_kernel if bool(args.endpoint_mask_morphology) else 0),
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        confidence_map=confidence_measure,
        max_confidence=effective_cable_confidence_max(args),
        min_area_px=(args.endpoint_marker_min_area if bool(args.endpoint_component_filter) else 1),
        min_points_per_component=(args.endpoint_marker_min_points if bool(args.endpoint_component_filter) else 1),
        max_points_per_component=args.endpoint_marker_points,
        max_components=2,
    )


def effective_cable_confidence_max(args):
    if not bool(args.depth_confidence_filter) or float(args.cable_confidence_max) < 0.0:
        return None
    return float(args.cable_confidence_max)


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
    endpoint_ratio = (
        float(args.pf_conditioned_proposal_ratio)
        if bool(args.pf_conditioned_proposals)
        else 0.0
    )
    random_ratio = (
        float(args.pf_global_random_ratio)
        if bool(args.pf_global_random_particles)
        else 0.0
    )
    # A disabled proposal component transfers only its mass to the local
    # posterior transition.  The other optional component keeps its configured
    # mass, so a single-feature ablation has one documented effect.
    local_ratio = 1.0 - endpoint_ratio - random_ratio
    return CableParticleFilterConfig(
        particle_count=int(args.pf_particles),
        estimate_top_particle_count=(
            int(args.pf_estimate_top_particles)
            if bool(args.pf_posterior_medoid)
            else 1
        ),
        segment_length_m=cable_segment_length_m(args, cable_index),
        process_node_std_m=float(args.pf_process_std),
        process_direction_std=float(args.pf_process_direction_std),
        velocity_enabled=bool(args.pf_velocity),
        velocity_damping=float(args.pf_velocity_damping),
        velocity_measurement_blend=float(args.pf_velocity_measurement_blend),
        velocity_process_std_mps=float(args.pf_velocity_process_std),
        max_node_speed_mps=float(args.pf_max_node_speed),
        adaptive_motion_noise_enabled=bool(args.pf_adaptive_motion),
        motion_speed_reference_mps=float(args.pf_motion_speed_reference),
        motion_innovation_reference_m=float(args.pf_motion_innovation_reference),
        motion_noise_adaptation=float(args.pf_motion_noise_adaptation),
        direction_smooth_passes=(
            int(args.pf_direction_smooth_passes)
            if bool(args.pf_direction_smoothing)
            else 0
        ),
        measurement_node_std_m=float(args.pf_measurement_std),
        measurement_max_points=int(args.pf_measurement_points),
        scoring_backend=str(args.pf_scoring_backend),
        endpoint_tangent_min_radius_m=float(args.pf_tangent_min_radius),
        endpoint_tangent_radius_m=float(args.pf_tangent_radius),
        endpoint_tangent_sigma_m=float(args.pf_tangent_sigma),
        endpoint_tangent_min_points=int(args.pf_tangent_min_points),
        endpoint_tangent_min_confidence=float(args.pf_tangent_min_confidence),
        endpoint_tangent_ransac_inlier_m=float(args.pf_tangent_ransac_inlier),
        endpoint_tangent_max_hypotheses=int(args.pf_tangent_max_hypotheses),
        endpoint_tangent_reference_weight=float(args.pf_tangent_reference_weight),
        endpoint_tangent_estimation_enabled=bool(args.pf_endpoint_tangent),
        endpoint_tangent_ransac_enabled=bool(args.pf_endpoint_tangent_ransac),
        endpoint_tangent_likelihood_scale_m=(
            float(args.pf_tangent_likelihood_scale)
            if bool(args.pf_endpoint_tangent and args.pf_endpoint_tangent_likelihood)
            else 0.0
        ),
        local_proposal_ratio=local_ratio,
        endpoint_conditioned_proposal_ratio=endpoint_ratio,
        endpoint_conditioned_direction_std=float(args.pf_conditioned_direction_std),
        endpoint_conditioned_deformation_std_m=float(args.pf_conditioned_deformation_std),
        endpoint_conditioned_slack_gain=float(args.pf_conditioned_slack_gain),
        endpoint_conditioned_max_deformation_m=float(args.pf_conditioned_max_deformation),
        endpoint_conditioned_deformation_modes=int(args.pf_conditioned_deformation_modes),
        robust_measurement_enabled=bool(args.pf_robust_measurement),
        robust_distance_m=float(args.pf_robust_distance),
        path_support_weight=(
            float(args.pf_path_support_weight)
            if bool(args.pf_dense_path_support)
            else 0.0
        ),
        path_support_samples_per_segment=int(args.pf_path_support_samples),
        union_coverage_top_particle_count=int(args.pf_union_coverage_top_particles),
        union_coverage_weight=(
            float(args.pf_union_coverage_weight)
            if bool(args.pf_union_coverage)
            else 0.0
        ),
        bend_penalty_m=(
            float(args.pf_bend_penalty)
            if bool(args.pf_bend_regularization)
            else 0.0
        ),
        global_random_particle_ratio=random_ratio,
        endpoint_constraint_iterations=int(args.pf_endpoint_constraint_iterations),
        endpoint_constraint_tolerance_m=float(args.pf_endpoint_constraint_tolerance),
        min_measurement_points=int(args.pf_min_measurement_points),
        min_segment_support_samples=int(args.pf_min_segment_support_samples),
        support_visibility_distance_m=float(args.pf_support_visibility_distance),
        crossing_position_sigma_px=float(args.crossing_position_sigma),
        crossing_angle_sigma_deg=float(args.crossing_angle_sigma),
        crossing_log_reward=(
            float(args.crossing_log_reward)
            if bool(args.crossing_likelihood)
            else 0.0
        ),
        max_prediction_frames=(
            int(args.pf_max_prediction_frames)
            if bool(args.pf_occlusion_prediction)
            else 0
        ),
        max_motion_noise_scale=float(args.pf_max_motion_noise_scale),
        point_support_diagnostics_enabled=bool(args.point_support_coloring),
        particle_diagnostics_enabled=bool(args.particle_diagnostics_overlay),
    )


def validate_feature_contract(args):
    """Reject feature combinations that would otherwise be silently inert."""

    validate_feature_state(feature_state_from_args(args))


def feature_gate_status(args):
    """Compact, stable feature summary for ablation logs."""

    gates = (
        ("PF", args.particle_filter),
        ("body-morph", args.cable_mask_morphology),
        ("body-components", args.cable_component_filter),
        ("endpoint-morph", args.endpoint_mask_morphology),
        ("endpoint-components", args.endpoint_component_filter),
        ("depth-confidence", args.depth_confidence_filter),
        ("endpoint-support", args.endpoint_association_support),
        ("velocity", args.pf_velocity),
        ("adaptive-motion", args.pf_adaptive_motion),
        ("occlusion", args.pf_occlusion_prediction),
        ("direction-smooth", args.pf_direction_smoothing),
        ("medoid", args.pf_posterior_medoid),
        ("tangent", args.pf_endpoint_tangent),
        ("tangent-RANSAC", args.pf_endpoint_tangent_ransac),
        ("tangent-L", args.pf_endpoint_tangent_likelihood),
        ("conditioned", args.pf_conditioned_proposals),
        ("random", args.pf_global_random_particles),
        ("robust-L", args.pf_robust_measurement),
        ("path-support", args.pf_dense_path_support),
        ("union", args.pf_union_coverage),
        ("bend", args.pf_bend_regularization),
        ("crossing", args.crossing_proposals),
        ("crossing-L", args.crossing_likelihood),
        ("point-colors", args.point_support_coloring),
        ("PF-overlay", args.particle_diagnostics_overlay),
    )
    return "FEATURES " + " ".join(
        f"{name}={int(bool(enabled))}" for name, enabled in gates
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
    feature_snapshot: dict | None = None


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
    feature_snapshot: dict


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
    crossing_points: np.ndarray
    crossing_proposals: tuple
    crossing_targets_by_cable: tuple
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
    def __init__(self, args, cable_detector, particle_filters, runtime_features=None):
        self.args = args
        self.cable_detector = cable_detector
        if particle_filters is None:
            particle_filters = []
        elif isinstance(particle_filters, CableParticleFilter):
            particle_filters = [particle_filters]
        self.particle_filters = list(particle_filters)
        self.runtime_features = runtime_features
        self.runtime_feature_revision = 0
        self.runtime_reset_revision = 0
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
                support_weight=(
                    float(args.endpoint_association_support_weight)
                    if bool(args.endpoint_association_support)
                    else 0.0
                ),
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
        feature_snapshot = frame.feature_snapshot or (
            self.runtime_features.snapshot()
            if self.runtime_features is not None
            else {
                "revision": int(self.runtime_feature_revision),
                "reset_revision": int(self.runtime_reset_revision),
                "features": feature_state_from_args(args),
            }
        )
        feature_state = feature_snapshot["features"]
        # The detector executor is single-threaded. Configure preprocessing
        # from the snapshot belonging to this exact frame, without mutating
        # the shared runtime args while the previous frame is being tracked.
        self.cable_detector.min_area = (
            int(args.detector_min_area)
            if bool(feature_state["cable_component_filter"])
            else 1
        )
        self.cable_detector.open_kernel = (
            max(1, int(args.detector_open_kernel)) | 1
            if bool(feature_state["cable_mask_morphology"])
            else 1
        )
        self.cable_detector.close_kernel = (
            max(1, int(args.detector_close_kernel)) | 1
            if bool(feature_state["cable_mask_morphology"])
            else 1
        )
        observation = self.cable_detector.detect_observation_masks(
            frame.bgr,
            endpoint_channel_count=2,
            scale=args.detector_scale,
            endpoint_thresholds=args.neural_detector_endpoint_thresholds,
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
        crossing_mask = (
            observation.crossing_mask
            if bool(feature_state["crossing_proposals"])
            else None
        )
        crossing_proposals = (
            extract_crossing_proposals(
                crossing_mask,
                observation.crossing_probability,
                min_area_px=args.crossing_min_area,
                max_proposals=args.crossing_max_proposals,
            )
            if crossing_mask is not None
            else tuple()
        )
        return DetectedTrackingFrame(
            frame=frame,
            detection=detection,
            endpoint_mask=endpoint_mask,
            detection_label_mask=detection_label_mask,
            endpoint_label_mask=endpoint_label_mask,
            endpoint_channel_masks=list(endpoint_channel_masks),
            crossing_mask=crossing_mask,
            crossing_proposals=crossing_proposals,
            detect_seconds=time.monotonic() - stage_start,
            feature_snapshot=feature_snapshot,
        )

    def _apply_runtime_features(self, feature_snapshot=None):
        if self.runtime_features is None:
            return
        snapshot = feature_snapshot or self.runtime_features.snapshot()
        revision = int(snapshot["revision"])
        if revision == self.runtime_feature_revision:
            return
        apply_feature_state_to_args(self.args, snapshot["features"])
        validate_feature_contract(self.args)
        self.endpoint_associator.config.support_weight = (
            float(self.args.endpoint_association_support_weight)
            if bool(self.args.endpoint_association_support)
            else 0.0
        )
        reset_revision = int(snapshot["reset_revision"])
        if reset_revision > self.runtime_reset_revision:
            self.particle_filters = [
                CableParticleFilter(
                    node_count=self.args.cable_segments + 1,
                    config=make_particle_filter_config(self.args, cable_index),
                    seed=17 + cable_index,
                )
                for cable_index in range(self.args.cable_count)
            ]
            self.last_filter_nodes_by_cable = [None for _ in range(self.args.cable_count)]
            self.last_filter_lost_frames_by_cable = [0 for _ in range(self.args.cable_count)]
            self.last_filter_lost_frames = 0
            self.last_filter_time = None
            self.endpoint_associator.last_endpoint_nodes_by_cable = [
                None for _ in range(self.args.cable_count)
            ]
            self.endpoint_associator.endpoint_stale_frames_by_cable = [
                0 for _ in range(self.args.cable_count)
            ]
            self.runtime_reset_revision = reset_revision
        self.runtime_feature_revision = revision
        print(f"Applied ablation revision {revision}: {feature_gate_status(self.args)}")

    def _process_detected_frame(self, detected):
        self._apply_runtime_features(detected.feature_snapshot)
        stage_seconds = {
            "detect": float(detected.detect_seconds),
            "fit": 0.0,
            "filter": 0.0,
        }
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
            self._shared_cable_and_crossing_support,
            frame,
            detection,
            crossing_mask,
        )
        endpoint_futures = [
            self.branch_executor.submit(
                self._detect_endpoint_branch,
                frame,
                endpoint_channel_masks[cable_index],
            )
            for cable_index in range(cable_count)
        ]
        shared_support, crossing_points = shared_support_future.result()
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
        crossing_proposals = estimate_crossing_axes(
            crossing_proposals,
            None if detection is None else detection.mask,
            outer_radius_px=args.crossing_axis_radius,
            min_axis_separation_deg=args.crossing_min_axis_separation,
            min_axis_support_px=args.crossing_min_axis_support,
        )
        endpoint_nodes_by_cable = [
            getattr(marker, "endpoint_nodes", None) if marker is not None else None
            for marker in candidate_markers
        ]
        crossing_targets_by_cable = (
            assign_crossing_targets(
                crossing_proposals,
                self.last_filter_nodes_by_cable,
                endpoint_nodes_by_cable,
                getattr(args, "camera_intrinsics", None),
            )
            if bool(args.crossing_likelihood)
            else tuple(tuple() for _ in range(cable_count))
        )
        endpoint_observations = endpoint_group_observations_xy(endpoint_markers_by_cable)
        endpoint_overlap_mask = overlap_mask_from_binary_masks(
            endpoint_channel_masks[:cable_count],
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
        stage_seconds["fit"] += max(0.0, time.monotonic() - stage_start)

        stage_start = time.monotonic()
        filter_dt = (
            1.0 / max(float(args.fps), 1.0)
            if self.last_filter_time is None
            else max(1e-3, float(frame.timestamp_s - self.last_filter_time))
        )
        self.last_filter_time = frame.timestamp_s

        active_filters = (
            self.particle_filters[:cable_count]
            if bool(args.particle_filter)
            else []
        )
        filter_results = update_cable_particle_filters(
            active_filters,
            measurements[:len(active_filters)],
            dt=filter_dt,
            crossing_targets_by_filter=crossing_targets_by_cable[:len(active_filters)],
            camera_intrinsics=getattr(args, "camera_intrinsics", None),
            union_coverage_enabled=bool(args.pf_union_coverage),
        )
        if len(filter_results) < cable_count:
            filter_results.extend([None] * (cable_count - len(filter_results)))
        for cable_index in range(cable_count):
            filter_result = filter_results[cable_index]
            measurement_for_cable = measurements[cable_index]
            estimate = filtered_cable_estimate(measurement_for_cable, filter_result)
            if estimate is None and cable_index >= len(active_filters):
                estimate = measurement_for_cable
            estimates[cable_index] = estimate
            diagnostics[cable_index] = cable_tracking_diagnostics(
                self.last_filter_nodes_by_cable[cable_index],
                measurement_for_cable,
                estimate,
                filter_result,
                diagnostics[cable_index],
            )
            if estimate is not None:
                self.last_filter_nodes_by_cable[cable_index] = np.asarray(estimate.points_xyz, dtype=np.float32)
            elif cable_index >= len(active_filters) or not bool(active_filters[cable_index].initialized):
                self.last_filter_nodes_by_cable[cable_index] = None
            if filter_result is not None:
                self.last_filter_lost_frames_by_cable[cable_index] = int(filter_result.lost_frames)

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
        tracking_diagnostics["ablation_revision"] = int(self.runtime_feature_revision)
        tracking_diagnostics["association"] = "fixed_endpoint_channel_identity"
        tracking_diagnostics["crossing_proposal_count"] = len(tuple(crossing_proposals or ()))
        tracking_diagnostics["crossing_point_count"] = int(len(crossing_points))
        tracking_diagnostics["crossing_axis_count"] = sum(
            int(np.asarray(proposal.axes_xy).shape == (2, 2))
            for proposal in tuple(crossing_proposals or ())
        )
        tracking_diagnostics["crossing_target_count"] = sum(
            len(targets) for targets in crossing_targets_by_cable
        )

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
            crossing_points=np.ascontiguousarray(crossing_points, dtype=np.float32),
            crossing_proposals=tuple(crossing_proposals or ()),
            crossing_targets_by_cable=tuple(crossing_targets_by_cable),
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

    def _shared_cable_and_crossing_support(self, frame, detection, crossing_mask):
        if detection is None or detection.mask is None:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty.copy()
        args = self.args
        return sampled_masked_point_cloud_point_sets(
            frame.point_cloud,
            (detection.mask, crossing_mask),
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            confidence_map=frame.confidence_measure,
            max_confidence=effective_cable_confidence_max(args),
            max_points_by_mask=(
                args.pf_measurement_points,
                args.crossing_max_visual_points,
            ),
            oversample=4,
        )

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
        elif measurement is None:
            measurement = CableEstimate3D(
                points_xyz=np.empty((0, 3), dtype=np.float32),
                source_points=np.empty((0, 3), dtype=np.float32),
                residual_m=np.nan,
                method="endpoint-only cable observation",
            )
        measurement = attach_endpoint_markers_to_measurement(measurement, candidate_markers)
        measurement_diagnostics = {}
        measurement_diagnostics.update(endpoint_diag)
        measurement_diagnostics.update(dict(association_diagnostics or {}))
        return measurement, measurement_diagnostics


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
    # Both PFs consume the same semantic cable cloud. Keep one copy for the
    # viewer/residual path instead of duplicating every RGB-D support point.
    source_points = concatenate_point_sets([getattr(valid_estimates[0], "source_points", None)])
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
    map_representative_errors = [
        float(getattr(diagnostics, "map_to_representative_node_error_m", np.nan))
        for _cable_index, diagnostics in particle_diagnostics
        if np.isfinite(float(getattr(diagnostics, "map_to_representative_node_error_m", np.nan)))
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
            if key in ("measurement", "crossing", "union"):
                stage_seconds[key] = max(stage_seconds.get(key, 0.0), float(value))
            else:
                stage_seconds[key] = stage_seconds.get(key, 0.0) + float(value)
    support_points = np.asarray(
        getattr(valid_results[0], "support_points_xyz", np.empty((0, 3))),
        dtype=np.float32,
    )
    affinity_rows = []
    affinities_valid = support_points.ndim == 2 and support_points.shape[1] >= 3
    if affinities_valid:
        affinities_by_index = {
            int(cable_index): np.asarray(
                getattr(result, "support_point_affinities", np.empty(0)),
                dtype=np.float32,
            ).reshape(-1)
            for cable_index, result in indexed_results
        }
        row_count = max(affinities_by_index, default=-1) + 1
        for cable_index in range(row_count):
            row = affinities_by_index.get(cable_index)
            if row is None or len(row) != len(support_points):
                affinities_valid = False
                break
            affinity_rows.append(row)
    support_affinities = (
        np.ascontiguousarray(np.stack(affinity_rows, axis=0), dtype=np.float32)
        if affinities_valid and affinity_rows
        else np.empty((0, 0), dtype=np.float32)
    )
    return SimpleNamespace(
        visible_nodes=visible_nodes,
        extended_visible_nodes=extended_nodes,
        visible_segments=visible_segments,
        measurement_used=any(bool(getattr(result, "measurement_used", False)) for result in valid_results),
        prediction_only=all(bool(getattr(result, "prediction_only", False)) for result in valid_results),
        lost_frames=max(int(getattr(result, "lost_frames", 0)) for result in valid_results),
        measurement_point_count=max(int(getattr(result, "measurement_point_count", 0)) for result in valid_results),
        segment_length_m=finite_result_mean(valid_results, "segment_length_m"),
        local_proposal_ratio=finite_result_mean(updated_results, "local_proposal_ratio"),
        endpoint_conditioned_proposal_ratio=finite_result_mean(
            updated_results,
            "endpoint_conditioned_proposal_ratio",
        ),
        global_random_particle_ratio=finite_result_mean(updated_results, "global_random_particle_ratio"),
        effective_sample_size=finite_result_mean(updated_results, "effective_sample_size"),
        path_support_rms_m=finite_result_mean(updated_results, "path_support_rms_m"),
        mean_support_affinity=finite_result_mean(updated_results, "mean_support_affinity"),
        supported_sample_fraction=finite_result_mean(
            updated_results,
            "supported_sample_fraction",
        ),
        estimate_particle_count=int(round(np.mean([
            int(getattr(result, "estimate_particle_count", 1))
            for result in valid_results
        ]))),
        estimate_weight_mass=(
            float(np.mean(estimate_weight_values))
            if estimate_weight_values else np.nan
        ),
        map_to_representative_node_error_m=(
            float(np.mean(map_representative_errors)) if map_representative_errors else np.nan
        ),
        mean_node_spread_m=float(np.mean(mean_spreads)) if mean_spreads else np.nan,
        max_node_spread_m=float(np.max(max_spreads)) if max_spreads else np.nan,
        endpoint_direction_delta_deg=endpoint_direction_mean,
        particle_diagnostics=particle_diagnostics,
        support_points_xyz=np.ascontiguousarray(support_points[:, :3], dtype=np.float32),
        support_point_affinities=support_affinities,
        mean_node_speed_mps=finite_result_mean(valid_results, "mean_node_speed_mps"),
        endpoint_speed_mps=finite_result_mean(valid_results, "endpoint_speed_mps"),
        endpoint_motion_innovation_m=finite_result_mean(
            valid_results,
            "endpoint_motion_innovation_m",
        ),
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
        "per_cable": valid[:max(0, int(cable_count))],
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
        "measurement_to_filter_m",
        "path_support_rms_m",
        "estimate_temporal_delta_m",
        "raw_residual_m",
        "filtered_residual_m",
        "local_ratio",
        "conditioned_ratio",
        "global_random_ratio",
        "motion_noise_scale",
        "endpoint_speed_mps",
        "endpoint_motion_innovation_m",
        "effective_sample_size",
        "support_affinity",
        "supported_sample_fraction",
        "tangent_confidence",
        "estimate_weight_mass",
        "map_to_representative_node_error_m",
        "mean_node_spread_m",
        "estimate_start_direction_delta_deg",
        "estimate_end_direction_delta_deg",
        "mean_node_speed_mps",
        "crossing_reward",
        "crossing_distance_px",
        "crossing_angle_error_deg",
        "union_coverage_rms_m",
        "union_coverage_fraction",
        "union_coverage_rms_gain_m",
        "union_coverage_fraction_gain",
    ):
        values = [float(item.get(key, np.nan)) for item in valid]
        finite = [value for value in values if np.isfinite(value)]
        if finite:
            combined[key] = float(np.mean(finite))
    combined["assigned_support_points"] = float(sum(
        max(0.0, float(item.get("assigned_support_points", 0.0) or 0.0))
        for item in valid
    ))
    combined["tangent_support_count"] = int(sum(
        max(0, int(item.get("tangent_support_count", 0) or 0))
        for item in valid
    ))
    estimate_counts = [
        int(item.get("estimate_particle_count", 0) or 0)
        for item in valid
        if int(item.get("estimate_particle_count", 0) or 0) > 0
    ]
    if estimate_counts:
        combined["estimate_particle_count"] = int(round(np.mean(estimate_counts)))
    union_ranks = [
        int(item.get("union_coverage_rank", 0) or 0)
        for item in valid[:max(0, int(cable_count))]
    ]
    if union_ranks and all(rank > 0 for rank in union_ranks):
        combined["union_coverage_rank_text"] = ",".join(str(rank) for rank in union_ranks)
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
        f"fit={worker_stage_ms['fit']:.1f} "
        f"pf={worker_stage_ms['filter']:.1f} | "
        f"async sub/ok/drop={async_worker.submitted}/{async_worker.completed}/{async_worker.dropped} "
        f"busy={1 if async_worker.busy_flag() else 0} | cloud {cloud_text} pts | "
        f"{latest_cable_status}"
    )


def make_runtime_feature_controller(args):
    initial = feature_state_from_args(args)

    def validate_runtime_state(state):
        candidate = SimpleNamespace(**vars(args))
        apply_feature_state_to_args(candidate, state)
        validate_feature_contract(candidate)

    return RuntimeFeatureController(initial, validator=validate_runtime_state)


def experiment_metadata(args, runtime_features):
    source_files = (
        "main.py",
        "cable_particle_filter.py",
        "cable_crossing.py",
        "cable_cuda.py",
        "cable_cuda_kernels.cu",
        "cable_detection.py",
        "cable_pidnet.py",
        "pf_ablation.py",
        "zed_spatial.py",
        "zed_split_viewer.py",
    )
    return {
        "schema_version": 2,
        "objective": "independent endpoint-constrained cable particle-filter ablation",
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "checkpoint_path": str(Path(args.neural_detector_checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.neural_detector_checkpoint),
        "source_sha256": {
            name: file_sha256(PROJECT_DIR / name)
            for name in source_files
        },
        "camera_source": (
            str(Path(args.input_svo_file).resolve())
            if args.input_svo_file
            else (str(args.ip_address) if args.ip_address else "local_zed")
        ),
        "scoring_backend": str(args.pf_scoring_backend),
        "particle_count": int(args.pf_particles),
        "node_count": int(args.cable_segments) + 1,
        "cable_lengths_m": list(args.cable_lengths_m),
        "crossing_likelihood": {
            "position_sigma_px": float(args.crossing_position_sigma),
            "angle_sigma_deg": float(args.crossing_angle_sigma),
            "log_reward": float(args.crossing_log_reward),
        },
        "proposal_ratios": {
            "local": float(args.pf_local_proposal_ratio),
            "endpoint": float(args.pf_conditioned_proposal_ratio),
            "global_random": float(args.pf_global_random_ratio),
        },
        "seeds": [17 + index for index in range(int(args.cable_count))],
        "initial_features": runtime_features.snapshot()["features"],
    }


class TrackingGpuBuffer:
    def __init__(self):
        self.point_cloud = sl.Mat()
        self.confidence_map = sl.Mat()
        self.in_use = False
        self.last_confidence_frame = -1


class LiveCaptureWorker:
    def __init__(self, args, zed, runtime, async_worker, runtime_features=None):
        self.args = args
        self.zed = zed
        self.runtime = runtime
        self.async_worker = async_worker
        self.runtime_features = runtime_features
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
                feature_snapshot = (
                    self.runtime_features.snapshot()
                    if self.runtime_features is not None
                    else {
                        "revision": 0,
                        "reset_revision": 0,
                        "features": feature_state_from_args(self.args),
                    }
                )
                feature_state = feature_snapshot["features"]
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
                    if bool(feature_state["depth_confidence_filter"]) and self.args.confidence_update_every > 0 and (
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
                        and bool(feature_state["depth_confidence_filter"])
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
                            feature_snapshot=feature_snapshot,
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
    runtime_features = make_runtime_feature_controller(args)
    control_panel = None
    recorder = ExperimentRecorder(
        args.experiment_output_directory,
        experiment_metadata(args, runtime_features),
    )
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
        particle_filters = [
            CableParticleFilter(
                node_count=args.cable_segments + 1,
                config=make_particle_filter_config(args, cable_index),
                seed=17 + cable_index,
            )
            for cable_index in range(args.cable_count)
        ]
        show_viewer_startup_status(viewer, "PIDNet ready | opening ZED camera...")
        zed = open_zed(args)
        args.camera_intrinsics = camera_intrinsics_from_zed(zed)
        recorder.metadata["camera_intrinsics"] = {
            "fx": float(args.camera_intrinsics.fx),
            "fy": float(args.camera_intrinsics.fy),
            "cx": float(args.camera_intrinsics.cx),
            "cy": float(args.camera_intrinsics.cy),
            "y_axis_up": bool(args.camera_intrinsics.y_axis_up),
            "z_axis_forward": bool(args.camera_intrinsics.z_axis_forward),
        }
        runtime = make_runtime_parameters(args)
        configure_viewer_from_zed(zed, viewer)
        show_viewer_startup_status(viewer, "ZED ready | starting tracking workers...")
        async_worker = AsyncTrackingWorker(
            args,
            cable_detector,
            particle_filters,
            runtime_features=runtime_features,
        )
        capture = LiveCaptureWorker(
            args,
            zed,
            runtime,
            async_worker,
            runtime_features=runtime_features,
        )
        capture.start()
        if bool(args.ablation_control_ui):
            control_panel = AblationControlPanel(runtime_features)
            control_panel.start()
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
    print(feature_gate_status(args))

    latest_detection = None
    latest_endpoint_mask = None
    latest_detection_label_mask = None
    latest_endpoint_label_mask = None
    latest_endpoint_overlap_mask = None
    latest_endpoint_observations = None
    latest_crossing_mask = None
    latest_crossing_points = np.empty((0, 3), dtype=np.float32)
    latest_crossing_proposals = tuple()
    latest_crossing_targets_by_cable = tuple()
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
                latest_crossing_points = result.crossing_points
                latest_crossing_proposals = result.crossing_proposals
                latest_crossing_targets_by_cable = result.crossing_targets_by_cable
                latest_tracking_diagnostics = result.tracking_diagnostics
                last_visual_measurement = result.measurement
                last_visual_estimate = result.estimate
                last_visual_filter_result = result.filter_result
                last_processed_completed = snapshot.completed
                feature_revision = int(latest_tracking_diagnostics.get("ablation_revision", 0) or 0)
                feature_snapshot = runtime_features.snapshot_for_revision(feature_revision)
                if runtime_features.snapshot()["recording"]:
                    recorder.record(
                        result.frame_index,
                        feature_snapshot,
                        latest_tracking_diagnostics,
                        timing=result.worker_stage_seconds,
                        frame_timestamp_s=result.frame_timestamp_s,
                    )

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
                    crossing_targets_by_cable=latest_crossing_targets_by_cable,
                )
                update_viewer_cable(
                    viewer,
                    last_visual_measurement,
                    last_visual_estimate,
                    last_visual_filter_result,
                    args.cable_max_points,
                    crossing_points=latest_crossing_points,
                    crossing_proposal_count=len(latest_crossing_proposals),
                    point_support_coloring=bool(args.point_support_coloring),
                    particle_diagnostics_overlay=bool(args.particle_diagnostics_overlay),
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
                panel_metrics = {
                    "tracking_fps": round((snapshot.completed - previous.completed) / elapsed, 2),
                    "filter_ms": round(worker_stage_ms["filter"], 3),
                }
                for key in (
                    "effective_sample_size",
                    "path_support_rms_m",
                    "mean_node_spread_m",
                    "crossing_reward",
                    "crossing_distance_px",
                    "crossing_angle_error_deg",
                ):
                    value = latest_tracking_diagnostics.get(key)
                    if value is not None and np.isscalar(value):
                        panel_metrics[key] = float(value)
                runtime_features.publish_metrics(panel_metrics)
                previous = snapshot
                previous_visual_updates = visual_update_count
                previous_ui_seconds = ui_seconds
                last_stats_time = now
            viewer.poll()
    finally:
        recorder.close()
        if control_panel is not None:
            control_panel.close()
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
    diagnostics["estimate_temporal_delta_m"] = mean_node_error(previous_nodes, estimate_nodes)
    diagnostics["path_support_rms_m"] = (
        float(getattr(filter_result, "path_support_rms_m", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    diagnostics["raw_residual_m"] = float(getattr(measurement, "residual_m", np.nan)) if measurement is not None else np.nan
    diagnostics["filtered_residual_m"] = float(getattr(estimate, "residual_m", np.nan)) if estimate is not None else np.nan
    diagnostics["local_ratio"] = (
        float(getattr(filter_result, "local_proposal_ratio", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    diagnostics["conditioned_ratio"] = (
        float(getattr(filter_result, "endpoint_conditioned_proposal_ratio", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    diagnostics["global_random_ratio"] = (
        float(getattr(filter_result, "global_random_particle_ratio", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    diagnostics["motion_noise_scale"] = (
        float(getattr(filter_result, "motion_noise_scale", np.nan))
        if filter_result is not None else np.nan
    )
    diagnostics["endpoint_speed_mps"] = (
        float(getattr(filter_result, "endpoint_speed_mps", np.nan))
        if filter_result is not None else np.nan
    )
    diagnostics["endpoint_motion_innovation_m"] = (
        float(getattr(filter_result, "endpoint_motion_innovation_m", np.nan))
        if filter_result is not None else np.nan
    )
    diagnostics["effective_sample_size"] = (
        float(getattr(filter_result, "effective_sample_size", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    diagnostics["support_affinity"] = (
        float(getattr(filter_result, "mean_support_affinity", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    diagnostics["supported_sample_fraction"] = (
        float(getattr(filter_result, "supported_sample_fraction", np.nan))
        if filter_result is not None and measurement_used
        else np.nan
    )
    tangent_confidence = np.asarray(
        getattr(filter_result, "endpoint_tangent_confidence", (np.nan, np.nan)),
        dtype=np.float32,
    ).reshape(-1)
    diagnostics["tangent_confidence"] = (
        float(np.mean(tangent_confidence[np.isfinite(tangent_confidence)]))
        if np.any(np.isfinite(tangent_confidence))
        else np.nan
    )
    tangent_support = np.asarray(
        getattr(filter_result, "endpoint_tangent_support_count", (0, 0)),
        dtype=np.int32,
    ).reshape(-1)
    diagnostics["tangent_support_count"] = int(np.sum(tangent_support))
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
    diagnostics["map_to_representative_node_error_m"] = (
        float(getattr(particle_diagnostics, "map_to_representative_node_error_m", np.nan))
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
    diagnostics["crossing_target_count"] = (
        int(getattr(filter_result, "crossing_target_count", 0))
        if filter_result is not None
        else 0
    )
    diagnostics["crossing_reward"] = (
        float(getattr(filter_result, "crossing_reward", np.nan))
        if filter_result is not None
        else np.nan
    )
    diagnostics["crossing_distance_px"] = (
        float(getattr(filter_result, "crossing_distance_px", np.nan))
        if filter_result is not None
        else np.nan
    )
    diagnostics["crossing_angle_error_deg"] = (
        float(getattr(filter_result, "crossing_angle_error_deg", np.nan))
        if filter_result is not None
        else np.nan
    )
    diagnostics["crossing_closest_xy"] = np.asarray(
        getattr(filter_result, "crossing_closest_xy", (np.nan, np.nan)),
        dtype=np.float32,
    ).reshape(-1)[:2]
    diagnostics["crossing_target_xy"] = np.asarray(
        getattr(filter_result, "crossing_target_xy", (np.nan, np.nan)),
        dtype=np.float32,
    ).reshape(-1)[:2]
    diagnostics["crossing_axis_xy"] = np.asarray(
        getattr(filter_result, "crossing_axis_xy", (np.nan, np.nan)),
        dtype=np.float32,
    ).reshape(-1)[:2]
    diagnostics["union_coverage_selected"] = bool(
        getattr(filter_result, "union_coverage_selected", False)
    ) if filter_result is not None else False
    diagnostics["union_coverage_rank"] = int(
        getattr(filter_result, "union_coverage_rank", 0)
    ) if filter_result is not None else 0
    diagnostics["union_coverage_rms_m"] = float(
        getattr(filter_result, "union_coverage_rms_m", np.nan)
    ) if filter_result is not None else np.nan
    diagnostics["union_coverage_fraction"] = float(
        getattr(filter_result, "union_coverage_fraction", np.nan)
    ) if filter_result is not None else np.nan
    diagnostics["union_coverage_rms_gain_m"] = float(
        getattr(filter_result, "union_coverage_rms_gain_m", np.nan)
    ) if filter_result is not None else np.nan
    diagnostics["union_coverage_fraction_gain"] = float(
        getattr(filter_result, "union_coverage_fraction_gain", np.nan)
    ) if filter_result is not None else np.nan
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
    crossing_points=(),
    crossing_proposal_count=0,
    point_support_coloring=True,
    particle_diagnostics_overlay=True,
):
    if estimate is None:
        viewer.update_cable(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
            crossing_points=crossing_points,
            crossing_proposal_count=crossing_proposal_count,
            cable_runs=(),
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
    source_points = np.asarray(
        getattr(filter_result, "support_points_xyz", np.empty((0, 3))),
        dtype=np.float32,
    )
    if len(source_points) == 0 and measurement is not None:
        source_points = np.asarray(measurement.source_points, dtype=np.float32)
    support_affinities = np.asarray(
        getattr(filter_result, "support_point_affinities", np.empty((0, 0))),
        dtype=np.float32,
    )
    if bool(point_support_coloring):
        sampled_points, sampled_affinities = sample_points_with_rows(
            source_points,
            support_affinities,
            max_points,
        )
        cable_point_colors = cable_support_colors(sampled_affinities)
    else:
        sampled_points = sample_points(source_points, max_points)
        cable_point_colors = None

    viewer.update_cable(
        sampled_points,
        nodes,
        valid_nodes,
        cable_point_colors=cable_point_colors,
        crossing_points=crossing_points,
        crossing_proposal_count=crossing_proposal_count,
        visible_nodes=visible_nodes,
        extended_visible_nodes=extended_visible_nodes,
        cable_runs=getattr(estimate, "pf_node_runs", None),
        particle_diagnostics=(
            getattr(filter_result, "particle_diagnostics", ())
            if bool(particle_diagnostics_overlay)
            else ()
        ),
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
    if crossing_count > 0:
        axis_count = int(diagnostics.get("crossing_axis_count", 0) or 0)
        target_count = int(diagnostics.get("crossing_target_count", 0) or 0)
        crossing_text = f"cross={crossing_count} axes={axis_count} targets={target_count}"
        crossing_point_count = int(diagnostics.get("crossing_point_count", 0) or 0)
        crossing_text += f" blue3d={crossing_point_count}"
        reward = float(diagnostics.get("crossing_reward", np.nan))
        distance = float(diagnostics.get("crossing_distance_px", np.nan))
        angle = float(diagnostics.get("crossing_angle_error_deg", np.nan))
        if np.isfinite(reward):
            crossing_text += f" R={reward:.2f}"
        if np.isfinite(distance):
            crossing_text += f" d={distance:.1f}px"
        if np.isfinite(angle):
            crossing_text += f" a={angle:.1f}deg"
        parts.append(crossing_text)
    union_rank_text = str(diagnostics.get("union_coverage_rank_text", "") or "")
    union_rms = float(diagnostics.get("union_coverage_rms_m", np.nan))
    union_fraction = float(diagnostics.get("union_coverage_fraction", np.nan))
    union_rms_gain = float(diagnostics.get("union_coverage_rms_gain_m", np.nan))
    union_fraction_gain = float(diagnostics.get("union_coverage_fraction_gain", np.nan))
    if union_rank_text:
        union_text = f"union-rank={union_rank_text}"
        if np.isfinite(union_rms):
            union_text += f" rms={format_mm(union_rms)}"
        if np.isfinite(union_fraction):
            union_text += f" cov={union_fraction:.2f}"
        if np.isfinite(union_rms_gain):
            union_text += f" gain={1000.0 * union_rms_gain:+.1f}mm"
        if np.isfinite(union_fraction_gain):
            union_text += f"/{union_fraction_gain:+.2f}cov"
        parts.append(union_text)
    local_ratio = diagnostics.get("local_ratio", np.nan)
    if np.isfinite(local_ratio):
        parts.append(f"local={local_ratio:.2f}")
    conditioned_ratio = diagnostics.get("conditioned_ratio", np.nan)
    if np.isfinite(conditioned_ratio):
        parts.append(f"conditioned={conditioned_ratio:.2f}")
    random_ratio = diagnostics.get("global_random_ratio", np.nan)
    if np.isfinite(random_ratio):
        parts.append(f"rand={random_ratio:.2f}")
    motion_scale = float(diagnostics.get("motion_noise_scale", np.nan))
    endpoint_speed = float(diagnostics.get("endpoint_speed_mps", np.nan))
    motion_innovation = float(diagnostics.get("endpoint_motion_innovation_m", np.nan))
    if np.isfinite(motion_scale):
        motion_text = f"motion={motion_scale:.2f}x"
        if np.isfinite(endpoint_speed):
            motion_text += f"/{endpoint_speed:.2f}mps"
        if np.isfinite(motion_innovation):
            motion_text += f"/{format_mm(motion_innovation)}innov"
        parts.append(motion_text)
    estimate_count = int(diagnostics.get("estimate_particle_count", 0) or 0)
    estimate_mass = float(diagnostics.get("estimate_weight_mass", np.nan))
    if estimate_count > 0:
        estimate_text = f"medoid-set={estimate_count}"
        if np.isfinite(estimate_mass):
            estimate_text += f" mass={estimate_mass:.2f}"
        parts.append(estimate_text)
    map_representative_error = float(diagnostics.get("map_to_representative_node_error_m", np.nan))
    mean_spread = float(diagnostics.get("mean_node_spread_m", np.nan))
    max_spread = float(diagnostics.get("max_node_spread_m", np.nan))
    if np.isfinite(map_representative_error):
        parts.append(f"MAP-MED={format_mm(map_representative_error)}")
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
        parts.append(f"MAP-MED-dir={start_text}/{end_text}deg")
    support_affinity = float(diagnostics.get("support_affinity", np.nan))
    supported_fraction = float(diagnostics.get("supported_sample_fraction", np.nan))
    path_support_rms = float(diagnostics.get("path_support_rms_m", np.nan))
    if np.isfinite(support_affinity) or np.isfinite(supported_fraction) or np.isfinite(path_support_rms):
        text = (
            f"support={support_affinity:.2f}"
            if np.isfinite(support_affinity)
            else "support=off"
        )
        if np.isfinite(supported_fraction):
            text += f" coverage={supported_fraction:.2f}"
        if np.isfinite(path_support_rms):
            text += f" rms={format_mm(path_support_rms)}"
        parts.append(text)
    effective_sample_size = float(diagnostics.get("effective_sample_size", np.nan))
    if np.isfinite(effective_sample_size):
        parts.append(f"ESS={effective_sample_size:.0f}")
    tangent_confidence = float(diagnostics.get("tangent_confidence", np.nan))
    tangent_support = int(diagnostics.get("tangent_support_count", 0) or 0)
    if np.isfinite(tangent_confidence):
        parts.append(f"tangent={tangent_confidence:.2f}/{tangent_support}pts")
    mean_node_speed = diagnostics.get("mean_node_speed_mps", np.nan)
    if np.isfinite(mean_node_speed):
        parts.append(f"v={mean_node_speed:.2f}m/s")
    pf_stage_ms = diagnostics.get("pf_stage_ms", None)
    if isinstance(pf_stage_ms, dict):
        stage_text = format_pf_stage_ms(pf_stage_ms)
        if stage_text:
            parts.append(stage_text)
    return "" if not parts else " | " + " ".join(parts)


def format_pf_stage_ms(stage_ms):
    labels = (
        ("prepare", "prep"),
        ("tangent", "tan"),
        ("initialize", "init"),
        ("predict", "pred"),
        ("measurement", "meas"),
        ("crossing", "cross"),
        ("union", "union"),
        ("estimate", "est"),
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
    crossing_targets_by_cable=None,
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
        crossing_targets_by_cable,
        tracking_diagnostics,
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


def draw_crossing_observations(
    panel,
    crossing_mask,
    crossing_proposals,
    crossing_targets_by_cable,
    tracking_diagnostics=None,
):
    """Draw the RGB-only crossing observation and PF likelihood diagnostics."""

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
                tint[:] = (255, 90, 10)
                blended = cv2.addWeighted(roi, 0.58, tint, 0.42, 0.0)
                cv2.copyTo(blended, mask_roi, roi)

    targets_by_proposal = {}
    for cable_index, targets in enumerate(tuple(crossing_targets_by_cable or ())):
        for target in tuple(targets or ()):
            targets_by_proposal.setdefault(int(target.proposal_id), []).append((cable_index, target))
    for proposal in tuple(crossing_proposals or ()):
        centroid_xy = np.asarray(getattr(proposal, "centroid_xy", ()), dtype=np.float64).reshape(-1)
        if len(centroid_xy) < 2 or not np.all(np.isfinite(centroid_xy[:2])):
            continue
        centroid_xy = centroid_xy[:2]
        color = (255, 170, 30)
        x, y, width, height = [int(value) for value in proposal.bbox_xywh]
        centroid = bounded_panel_point(centroid_xy, panel.shape)
        if centroid is None:
            continue
        cv2.rectangle(panel, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
        cv2.drawMarker(panel, centroid, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        axes = np.asarray(getattr(proposal, "axes_xy", ()), dtype=np.float64)
        axis_length = max(24, int(round(1.4 * max(width, height))))
        if axes.shape == (2, 2) and np.all(np.isfinite(axes)):
            for axis in axes:
                start = bounded_panel_point(centroid_xy - axis_length * axis, panel.shape)
                end = bounded_panel_point(centroid_xy + axis_length * axis, panel.shape)
                if start is not None and end is not None:
                    cv2.line(panel, start, end, (210, 210, 210), 2, cv2.LINE_AA)
        for cable_index, target in targets_by_proposal.get(int(proposal.proposal_id), ()):
            target_color = SEGMENTATION_LABEL_COLORS_BGR[1 + (int(cable_index) % 2)]
            axis = np.asarray(target.axis_xy, dtype=np.float64).reshape(-1)
            if len(axis) < 2 or not np.all(np.isfinite(axis[:2])):
                continue
            axis = axis[:2]
            start = bounded_panel_point(centroid_xy - axis_length * axis, panel.shape)
            end = bounded_panel_point(centroid_xy + axis_length * axis, panel.shape)
            if start is not None and end is not None:
                cv2.line(panel, start, end, target_color, 3, cv2.LINE_AA)
        details = (
            f"RGB CROSSING p={proposal.mean_probability:.2f} "
            f"axes={2 if axes.shape == (2, 2) else 0}"
        )
        text_y = min(panel.shape[0] - 8, max(18, y + height + 18))
        cv2.putText(panel, details, (max(4, x), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(panel, details, (max(4, x), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    per_cable = list(dict(tracking_diagnostics or {}).get("per_cable", ()) or ())
    for cable_index, diagnostics in enumerate(per_cable):
        closest = np.asarray(diagnostics.get("crossing_closest_xy", ()), dtype=np.float32).reshape(-1)
        target = np.asarray(diagnostics.get("crossing_target_xy", ()), dtype=np.float32).reshape(-1)
        if len(closest) < 2 or len(target) < 2 or not np.all(np.isfinite((closest[:2], target[:2]))):
            continue
        closest_point = tuple(np.round(closest[:2]).astype(np.int32))
        target_point = tuple(np.round(target[:2]).astype(np.int32))
        cable_color = SEGMENTATION_LABEL_COLORS_BGR[1 + (int(cable_index) % 2)]
        cv2.line(panel, closest_point, target_point, cable_color, 2, cv2.LINE_AA)
        cv2.drawMarker(panel, closest_point, cable_color, cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)
        reward = float(diagnostics.get("crossing_reward", np.nan))
        distance = float(diagnostics.get("crossing_distance_px", np.nan))
        angle = float(diagnostics.get("crossing_angle_error_deg", np.nan))
        text = f"PF{cable_index + 1} cross R={reward:.2f} d={distance:.1f}px a={angle:.1f}deg"
        text_origin = (min(panel.shape[1] - 280, max(4, closest_point[0] + 10)), max(18, closest_point[1] - 10))
        cv2.putText(panel, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(panel, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.40, cable_color, 1, cv2.LINE_AA)


def bounded_panel_point(values, panel_shape):
    """Convert finite floating-point image coordinates to safe OpenCV integers."""

    point = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        return None
    height, width = int(panel_shape[0]), int(panel_shape[1])
    limits = np.asarray((max(width, 1), max(height, 1)), dtype=np.float64)
    point = np.clip(point[:2], -2.0 * limits, 3.0 * limits)
    return int(round(float(point[0]))), int(round(float(point[1])))


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


def sample_points_with_rows(points, rows, max_points):
    points = np.asarray(points, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    points = points[:, :3]
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]
    if rows.ndim != 2 or rows.shape[1] != len(valid):
        rows = np.empty((0, len(points)), dtype=np.float32)
    else:
        rows = rows[:, valid]
    max_points = max(0, int(max_points))
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
        rows = rows[:, indices]
    return (
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(rows, dtype=np.float32),
    )


def cable_support_colors(affinities):
    """Color shared cloud points by independent proximity to each PF estimate."""

    affinities = np.asarray(affinities, dtype=np.float32)
    if affinities.ndim != 2 or affinities.shape[0] < 1:
        return np.empty((0, 3), dtype=np.float32)
    first = np.clip(affinities[0], 0.0, 1.0)
    second = (
        np.clip(affinities[1], 0.0, 1.0)
        if affinities.shape[0] >= 2
        else np.zeros_like(first)
    )
    explained = np.maximum(first, second)
    certainty = np.abs(first - second) / np.maximum(first + second, 1e-6)
    winner = first >= second
    pf1 = np.asarray((1.00, 0.08, 0.88), dtype=np.float32)
    pf2 = np.asarray((0.00, 0.86, 1.00), dtype=np.float32)
    ambiguous = np.asarray((1.00, 0.58, 0.08), dtype=np.float32)
    outlier = np.asarray((0.36, 0.39, 0.42), dtype=np.float32)
    selected = np.where(winner[:, None], pf1[None, :], pf2[None, :])
    explained_color = ambiguous[None, :] * (1.0 - certainty[:, None]) + selected * certainty[:, None]
    colors = outlier[None, :] * (1.0 - explained[:, None]) + explained_color * explained[:, None]
    return np.ascontiguousarray(colors, dtype=np.float32)


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
