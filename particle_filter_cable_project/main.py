import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import queue
import threading
import time
import tomllib

import cv2
import numpy as np
import pyzed.sl as sl

from cable_detection import (
    HsvCableDetector,
    cable_measurement_from_mask_points,
    fit_cable_segments_from_zed_point_cloud,
    polyline_residual,
)
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    fit_reference_ordered_point_cloud_chain,
    fit_unordered_point_cloud_chain,
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
    parser.add_argument("--sdk-gpu-id", type=int, default=config_value(config, "camera", "sdk_gpu_id", 0), help="ZED SDK CUDA device id. Use -1 for SDK auto selection.")
    parser.add_argument("--zed-async-image-retrieval", action=argparse.BooleanOptionalAction, default=config_value(config, "camera", "async_image_retrieval", True), help="Allow ZED image retrieval to run asynchronously when supported by the SDK.")
    parser.add_argument("--zed-async-recovery", action=argparse.BooleanOptionalAction, default=config_value(config, "camera", "async_grab_camera_recovery", True), help="Recover camera communication issues in the background instead of blocking grab().")
    parser.add_argument("--depth-mode", choices=DEPTH_MODES, default=config_value(config, "camera", "depth_mode", "NEURAL_PLUS"))
    parser.add_argument("--depth-min", type=float, default=config_value(config, "depth", "min_m", 0.01))
    parser.add_argument("--depth-max", type=float, default=config_value(config, "depth", "max_m", 1.0))
    parser.add_argument("--depth-stabilization", type=int, default=config_value(config, "depth", "stabilization", 0), help="ZED depth stabilization strength. 0 disables hidden positional tracking overhead.")
    parser.add_argument("--image-enhancement", action=argparse.BooleanOptionalAction, default=config_value(config, "camera", "image_enhancement", True), help="Use ZED SDK image enhancement for the left RGB image.")
    parser.add_argument("--confidence", type=int, default=config_value(config, "depth", "confidence", 95))
    parser.add_argument("--texture-confidence", type=int, default=config_value(config, "depth", "texture_confidence", 100))
    parser.add_argument("--depth-fill", action=argparse.BooleanOptionalAction, default=config_value(config, "depth", "fill", True))
    parser.add_argument("--live-stride", type=int, default=config_value(config, "point_cloud", "stride", 4))
    parser.add_argument("--live-max-points", type=int, default=config_value(config, "point_cloud", "max_points", 100000), help="0 keeps every sampled point.")
    parser.add_argument("--cloud-update-every", type=int, default=config_value(config, "point_cloud", "cloud_update_every", 1), help="Update the visual point cloud every N frames; tracking still runs every frame.")
    parser.add_argument("--confidence-update-every", type=int, default=config_value(config, "point_cloud", "confidence_update_every", 1), help="Retrieve ZED confidence every N frames; 0 disables confidence filtering.")
    parser.add_argument("--async-measurement", action=argparse.BooleanOptionalAction, default=config_value(config, "processing", "async_measurement", True), help="Run RGB detection, 3D fitting, and particle-filter update in a worker thread.")
    parser.add_argument("--detector-backend", choices=("pidnet", "hsv"), default=config_value(config, "detector", "backend", "pidnet"), help="RGB segmentation backend for cable mask comparison.")
    parser.add_argument("--detector-scale", type=float, default=config_value(config, "detector", "scale", 0.50), help="Run RGB cable detection at this image scale, then lift coordinates back to full resolution.")
    parser.add_argument("--detector-roi", action=argparse.BooleanOptionalAction, default=config_value(config, "detector", "roi", True), help="Use the previous 2D cable path as a gated detector search region.")
    parser.add_argument("--detector-roi-padding", type=int, default=config_value(config, "detector", "roi_padding_px", 96), help="Pixels of padding around the previous cable path for ROI detection.")
    parser.add_argument("--detector-full-every", type=int, default=config_value(config, "detector", "full_frame_every", 12), help="Run full-frame detection every N frames for reacquisition; 0 disables periodic full detection.")
    parser.add_argument("--detector-update-every", type=int, default=config_value(config, "detector", "update_every", 1), help="Run RGB detection + 3D measurement every N frames; skipped frames use PF prediction.")
    parser.add_argument("--point-size", type=float, default=config_value(config, "viewer", "point_size", 2.0))
    parser.add_argument("--rgb-width", type=int, default=config_value(config, "viewer", "rgb_width", 620))
    parser.add_argument("--cloud-width", type=int, default=config_value(config, "viewer", "cloud_width", 1180))
    parser.add_argument("--height", type=int, default=config_value(config, "viewer", "height", 900))
    parser.add_argument("--viewer-hold-frames", type=int, default=config_value(config, "viewer", "hold_frames", 8), help="Keep the last drawn cable overlay for this many frames when a measurement frame is skipped or briefly invalid.")
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
    parser.add_argument("--neural-detector-amp", action=argparse.BooleanOptionalAction, default=config_value(config, "pidnet", "amp", True), help="Use CUDA autocast for PIDNet inference.")
    parser.add_argument("--detector-min-area", type=int, default=config_value(config, "detector", "min_area_px", 80))
    parser.add_argument("--detector-open-kernel", type=int, default=config_value(config, "detector", "open_kernel", 3))
    parser.add_argument("--detector-close-kernel", type=int, default=config_value(config, "detector", "close_kernel", 5))
    parser.add_argument("--detector-skeleton-prune-px", type=int, default=config_value(config, "detector", "skeleton_prune_px", 10))
    parser.add_argument("--detector-skeleton-prune-passes", type=int, default=config_value(config, "detector", "skeleton_prune_passes", 2))
    parser.add_argument("--detector-centerline-smooth-window", type=int, default=config_value(config, "detector", "centerline_smooth_window", 9))
    parser.add_argument("--hsv-h-min", type=int, default=config_value(config, "hsv", "h_min", 50))
    parser.add_argument("--hsv-h-max", type=int, default=config_value(config, "hsv", "h_max", 80))
    parser.add_argument("--hsv-s-min", type=int, default=config_value(config, "hsv", "s_min", 80))
    parser.add_argument("--hsv-s-max", type=int, default=config_value(config, "hsv", "s_max", 255))
    parser.add_argument("--hsv-v-min", type=int, default=config_value(config, "hsv", "v_min", 80))
    parser.add_argument("--hsv-v-max", type=int, default=config_value(config, "hsv", "v_max", 255))
    parser.add_argument("--hsv-mode", choices=("range", "gaussian"), default=config_value(config, "hsv", "mode", "range"))
    parser.add_argument("--hsv-fast-mask-only", action=argparse.BooleanOptionalAction, default=config_value(config, "hsv", "fast_mask_only", True))
    parser.add_argument("--hsv-geometry-every", type=int, default=config_value(config, "hsv", "geometry_every", 12))
    parser.add_argument("--hsv-mask-points", type=int, default=config_value(config, "hsv", "mask_points", 512))
    parser.add_argument("--hsv-gaussian-threshold", type=float, default=config_value(config, "hsv", "gaussian_threshold", 0.0))
    parser.add_argument("--hsv-gaussian-positive-mean", type=float, nargs=4, default=config_value(config, "hsv", "gaussian_positive_mean", None))
    parser.add_argument("--hsv-gaussian-positive-std", type=float, nargs=4, default=config_value(config, "hsv", "gaussian_positive_std", None))
    parser.add_argument("--hsv-gaussian-negative-mean", type=float, nargs=4, default=config_value(config, "hsv", "gaussian_negative_mean", None))
    parser.add_argument("--hsv-gaussian-negative-std", type=float, nargs=4, default=config_value(config, "hsv", "gaussian_negative_std", None))
    parser.add_argument("--cable-segments", type=int, default=config_value(config, "cable", "segments", 2))
    parser.add_argument("--cable-max-points", type=int, default=config_value(config, "cable", "max_visual_points", 1000))
    parser.add_argument("--node-search-px", type=int, default=config_value(config, "measurement", "node_search_px", 2))
    parser.add_argument(
        "--measurement-centerline-points",
        type=int,
        default=config_value(config, "measurement", "centerline_points", 128),
        help="Maximum ordered 2D centerline pixels lifted to 3D before fitting. Keeps measurement cost bounded.",
    )
    parser.add_argument(
        "--measurement-local-depth-spread",
        type=float,
        default=config_value(config, "measurement", "local_depth_spread_m", 0.06),
        help="Reject a local ZED lookup window when 75%% of 3D samples spread farther than this many meters. Use 0 to disable.",
    )
    parser.add_argument("--measurement-smoothing", action=argparse.BooleanOptionalAction, default=config_value(config, "measurement", "smoothing", True), help="Smooth fitted 3D measurement nodes before particle-filter proposal injection.")
    parser.add_argument("--measurement-smoothing-alpha", type=float, default=config_value(config, "measurement", "smoothing_alpha", 0.30), help="Current-frame weight for measurement node smoothing.")
    parser.add_argument("--measurement-smoothing-gate", type=float, default=config_value(config, "measurement", "smoothing_gate_m", 0.040), help="Do not smooth when raw measurement jumps farther than this mean node distance. Use 0 to always smooth.")
    parser.add_argument(
        "--measurement-mask-centerline",
        action=argparse.BooleanOptionalAction,
        default=config_value(config, "measurement", "mask_centerline", False),
        help="For PIDNet masks, build an ordered 3D centerline before the particle-filter update. Disable for full-PF raw cloud scoring.",
    )
    parser.add_argument(
        "--measurement-mask-centerline-gate",
        type=float,
        default=config_value(config, "measurement", "mask_centerline_reference_gate_m", 0.15),
        help="Reference-projection gate in meters when ordering PIDNet mask points by the previous filtered cable. Use 0 for ungated projection.",
    )
    parser.add_argument(
        "--measurement-mask-centerline-max-residual",
        type=float,
        default=config_value(config, "measurement", "mask_centerline_max_residual_m", 0.08),
        help="Reject an ordered mask-cloud centerline when its median support residual is above this many meters. Use 0 to disable.",
    )
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
    parser.add_argument("--cable-segment-length", type=float, default=config_value(config, "cable", "segment_length_m", 0.0), help="Fixed segment length in meters. 0 estimates once from the first measurement.")
    parser.add_argument("--particle-filter", action=argparse.BooleanOptionalAction, default=config_value(config, "particle_filter", "enabled", True))
    parser.add_argument("--pf-particles", type=int, default=config_value(config, "particle_filter", "particles", pf_defaults.particle_count))
    parser.add_argument("--pf-initial-node-std", type=float, default=config_value(config, "particle_filter", "initial_node_std_m", pf_defaults.initial_node_std_m))
    parser.add_argument("--pf-initial-direction-std", type=float, default=config_value(config, "particle_filter", "initial_direction_std", pf_defaults.initial_direction_std))
    parser.add_argument("--pf-process-std", type=float, default=config_value(config, "particle_filter", "process_node_std_m", pf_defaults.process_node_std_m))
    parser.add_argument("--pf-process-direction-std", type=float, default=config_value(config, "particle_filter", "process_direction_std", pf_defaults.process_direction_std))
    parser.add_argument("--pf-direction-smooth-passes", type=int, default=config_value(config, "particle_filter", "direction_smooth_passes", pf_defaults.direction_smooth_passes))
    parser.add_argument("--pf-temporal-prediction", action=argparse.BooleanOptionalAction, default=config_value(config, "particle_filter", "temporal_prediction", pf_defaults.temporal_prediction), help="Use TrackDLO-style per-node velocity when predicting particles.")
    parser.add_argument("--pf-prediction-gain", type=float, default=config_value(config, "particle_filter", "prediction_gain", pf_defaults.prediction_gain), help="Scale applied to learned per-node velocity before prediction.")
    parser.add_argument("--pf-prediction-velocity-alpha", type=float, default=config_value(config, "particle_filter", "prediction_velocity_alpha", pf_defaults.prediction_velocity_alpha), help="EMA update weight for measured per-node velocity.")
    parser.add_argument("--pf-prediction-velocity-decay", type=float, default=config_value(config, "particle_filter", "prediction_velocity_decay", pf_defaults.prediction_velocity_decay), help="Velocity decay applied during prediction-only frames.")
    parser.add_argument("--pf-max-prediction-step", type=float, default=config_value(config, "particle_filter", "max_prediction_step_m", pf_defaults.max_prediction_step_m), help="Maximum temporal-prediction displacement per node per update, in meters.")
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
    parser.add_argument(
        "--pf-measurement-proposal-wide-fraction",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_wide_fraction", pf_defaults.measurement_proposal_wide_fraction),
        help="Fraction of injected proposals sampled broadly around the temporary measured chain.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-current-fraction",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_current_fraction", pf_defaults.measurement_proposal_current_fraction),
        help="Fraction of injected proposals sampled near the current MAP/filtered chain.",
    )
    parser.add_argument(
        "--pf-measurement-proposal-wide-std-multiplier",
        type=float,
        default=config_value(config, "particle_filter", "measurement_proposal_wide_std_multiplier", pf_defaults.measurement_proposal_wide_std_multiplier),
        help="Noise multiplier for broad measurement proposals.",
    )
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
    parser.add_argument(
        "--pf-two-sided-support-weight",
        type=float,
        default=config_value(config, "particle_filter", "two_sided_support_weight", pf_defaults.two_sided_support_weight),
        help="Weight for particle-to-cloud support scoring. 0 disables the reverse likelihood.",
    )
    parser.add_argument(
        "--pf-two-sided-support-samples",
        type=int,
        default=config_value(config, "particle_filter", "two_sided_support_samples_per_segment", pf_defaults.two_sided_support_samples_per_segment),
        help="Number of support samples per visible particle segment for reverse scoring.",
    )
    parser.add_argument(
        "--pf-two-sided-support-distance",
        type=float,
        default=config_value(config, "particle_filter", "two_sided_support_distance_m", pf_defaults.two_sided_support_distance_m),
        help="Clamp distance for particle-to-cloud support scoring, in meters.",
    )
    parser.add_argument("--pf-bend-penalty", type=float, default=config_value(config, "particle_filter", "bend_penalty_m", pf_defaults.bend_penalty_m))
    parser.add_argument(
        "--pf-estimate-mode",
        choices=("map", "weighted", "auto"),
        default=config_value(config, "particle_filter", "estimate_mode", pf_defaults.estimate_mode),
        help="map returns the highest-probability particle; weighted averages particles; auto uses MAP only after weight collapse.",
    )
    parser.add_argument("--pf-map-estimate-effective-ratio", type=float, default=config_value(config, "particle_filter", "map_estimate_effective_ratio", pf_defaults.map_estimate_effective_ratio))
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
    args.cloud_update_every = max(1, int(args.cloud_update_every))
    args.confidence_update_every = max(0, int(args.confidence_update_every))
    args.detector_scale = float(np.clip(args.detector_scale, 0.10, 1.0))
    args.detector_roi_padding = max(0, int(args.detector_roi_padding))
    args.detector_full_every = max(0, int(args.detector_full_every))
    args.detector_update_every = max(1, int(args.detector_update_every))
    args.detector_skeleton_prune_px = max(0, int(args.detector_skeleton_prune_px))
    args.detector_skeleton_prune_passes = max(0, int(args.detector_skeleton_prune_passes))
    args.detector_centerline_smooth_window = max(1, int(args.detector_centerline_smooth_window))
    args.viewer_hold_frames = max(0, int(args.viewer_hold_frames))
    args.hsv_geometry_every = max(0, int(args.hsv_geometry_every))
    args.hsv_mask_points = max(2, int(args.hsv_mask_points))
    args.measurement_centerline_points = max(2, int(args.measurement_centerline_points))
    args.measurement_smoothing_alpha = float(np.clip(args.measurement_smoothing_alpha, 0.0, 1.0))
    args.measurement_smoothing_gate = max(0.0, float(args.measurement_smoothing_gate))
    args.measurement_mask_centerline_gate = max(0.0, float(args.measurement_mask_centerline_gate))
    args.measurement_mask_centerline_max_residual = max(0.0, float(args.measurement_mask_centerline_max_residual))
    args.measurement_gate_reacquire_after = max(0, int(args.measurement_gate_reacquire_after))
    args.pf_score_chunk_points = max(1, int(args.pf_score_chunk_points))
    args.pf_endpoint_refresh_interval = max(0, int(args.pf_endpoint_refresh_interval))
    args.pf_reference_ordering_gate = max(0.0, float(args.pf_reference_ordering_gate))
    args.pf_prediction_gain = max(0.0, float(args.pf_prediction_gain))
    args.pf_prediction_velocity_alpha = float(np.clip(args.pf_prediction_velocity_alpha, 0.0, 1.0))
    args.pf_prediction_velocity_decay = float(np.clip(args.pf_prediction_velocity_decay, 0.0, 1.0))
    args.pf_max_prediction_step = max(0.0, float(args.pf_max_prediction_step))
    args.pf_measurement_proposal_wide_fraction = float(np.clip(args.pf_measurement_proposal_wide_fraction, 0.0, 0.8))
    args.pf_measurement_proposal_current_fraction = float(np.clip(args.pf_measurement_proposal_current_fraction, 0.0, 0.8))
    args.pf_measurement_proposal_wide_std_multiplier = max(1.0, float(args.pf_measurement_proposal_wide_std_multiplier))
    args.pf_two_sided_support_weight = max(0.0, float(args.pf_two_sided_support_weight))
    args.pf_two_sided_support_samples = max(1, int(args.pf_two_sided_support_samples))
    args.pf_two_sided_support_distance = max(0.0, float(args.pf_two_sided_support_distance))
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
    if hasattr(init, "sdk_gpu_id"):
        init.sdk_gpu_id = int(args.sdk_gpu_id)
    if hasattr(init, "async_image_retrieval"):
        init.async_image_retrieval = bool(args.zed_async_image_retrieval)
    if hasattr(init, "async_grab_camera_recovery"):
        init.async_grab_camera_recovery = bool(args.zed_async_recovery)
    if hasattr(init, "depth_stabilization"):
        init.depth_stabilization = int(args.depth_stabilization)
    if hasattr(init, "enable_image_enhancement"):
        init.enable_image_enhancement = bool(args.image_enhancement)
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
    if args.detector_backend == "hsv":
        detector = HsvCableDetector(
            h_min=args.hsv_h_min,
            h_max=args.hsv_h_max,
            s_min=args.hsv_s_min,
            s_max=args.hsv_s_max,
            v_min=args.hsv_v_min,
            v_max=args.hsv_v_max,
            mode=args.hsv_mode,
            gaussian_threshold=args.hsv_gaussian_threshold,
            gaussian_positive_mean=args.hsv_gaussian_positive_mean,
            gaussian_positive_std=args.hsv_gaussian_positive_std,
            gaussian_negative_mean=args.hsv_gaussian_negative_mean,
            gaussian_negative_std=args.hsv_gaussian_negative_std,
            min_area=int(args.detector_min_area),
            open_kernel=int(args.detector_open_kernel),
            close_kernel=int(args.detector_close_kernel),
            skeleton_prune_px=int(args.detector_skeleton_prune_px),
            skeleton_prune_passes=int(args.detector_skeleton_prune_passes),
            centerline_smooth_window=int(args.detector_centerline_smooth_window),
        )
        print(f"Loaded HSV cable detector: {detector.description()}")
        return detector

    checkpoint_path = Path(args.neural_detector_checkpoint)
    if not checkpoint_path.exists():
        print(
            f"No PIDNet checkpoint found at {checkpoint_path}. "
            "Run tools/pidnet_training_gui.py or tools/train_pidnet_cable.py first."
        )
        return None
    try:
        from cable_pidnet import PidNetCableDetector

        detector = PidNetCableDetector(
            checkpoint_path,
            device=args.neural_detector_device,
            threshold=float(args.neural_detector_threshold),
            base_channels=int(args.neural_detector_base_channels),
            amp=bool(args.neural_detector_amp),
            min_area=int(args.detector_min_area),
            open_kernel=int(args.detector_open_kernel),
            close_kernel=int(args.detector_close_kernel),
            skeleton_prune_px=int(args.detector_skeleton_prune_px),
            skeleton_prune_passes=int(args.detector_skeleton_prune_passes),
            centerline_smooth_window=int(args.detector_centerline_smooth_window),
        )
        print(f"Loaded PIDNet cable detector: {checkpoint_path} on {args.neural_detector_device}")
    except Exception as exc:
        print(f"Could not load cable detector: {exc}")
        return None
    return detector


def detector_description(args):
    if args.detector_backend == "hsv":
        if args.hsv_mode == "gaussian":
            return f"HSV Gaussian threshold {float(args.hsv_gaussian_threshold):.3f}"
        return (
            f"HSV H {int(args.hsv_h_min)}-{int(args.hsv_h_max)} "
            f"S {int(args.hsv_s_min)}-{int(args.hsv_s_max)} "
            f"V {int(args.hsv_v_min)}-{int(args.hsv_v_max)}"
        )
    return f"PIDNet mask threshold {float(args.neural_detector_threshold):.2f}"


def make_particle_filter_config(args):
    return CableParticleFilterConfig(
        particle_count=int(args.pf_particles),
        segment_length_m=float(args.cable_segment_length),
        initial_node_std_m=float(args.pf_initial_node_std),
        initial_direction_std=float(args.pf_initial_direction_std),
        process_node_std_m=float(args.pf_process_std),
        process_direction_std=float(args.pf_process_direction_std),
        direction_smooth_passes=int(args.pf_direction_smooth_passes),
        temporal_prediction=bool(args.pf_temporal_prediction),
        prediction_gain=float(args.pf_prediction_gain),
        prediction_velocity_alpha=float(args.pf_prediction_velocity_alpha),
        prediction_velocity_decay=float(args.pf_prediction_velocity_decay),
        max_prediction_step_m=float(args.pf_max_prediction_step),
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
        endpoint_penalty_weight=float(args.pf_endpoint_penalty_weight),
        measurement_reset_error_m=float(args.pf_measurement_reset_error),
        measurement_proposal_ratio=float(args.pf_measurement_proposal_ratio),
        measurement_proposal_stable_ratio=float(args.pf_measurement_proposal_stable_ratio),
        measurement_proposal_start_error_m=float(args.pf_measurement_proposal_start_error),
        measurement_proposal_full_error_m=float(args.pf_measurement_proposal_full_error),
        measurement_proposal_node_std_m=float(args.pf_measurement_proposal_std),
        measurement_proposal_direction_std=float(args.pf_measurement_proposal_direction_std),
        measurement_proposal_wide_fraction=float(args.pf_measurement_proposal_wide_fraction),
        measurement_proposal_current_fraction=float(args.pf_measurement_proposal_current_fraction),
        measurement_proposal_wide_std_multiplier=float(args.pf_measurement_proposal_wide_std_multiplier),
        score_keep_fraction=float(args.pf_score_keep_fraction),
        coverage_penalty_m=float(args.pf_coverage_penalty),
        coverage_min_fraction=float(args.pf_coverage_min_fraction),
        two_sided_support_weight=float(args.pf_two_sided_support_weight),
        two_sided_support_samples_per_segment=int(args.pf_two_sided_support_samples),
        two_sided_support_distance_m=float(args.pf_two_sided_support_distance),
        bend_penalty_m=float(args.pf_bend_penalty),
        estimate_mode=str(args.pf_estimate_mode),
        map_estimate_effective_ratio=float(args.pf_map_estimate_effective_ratio),
        min_measurement_points=int(args.pf_min_measurement_points),
        min_segment_points=int(args.pf_min_segment_points),
        occlusion_assignment_max_distance_m=float(args.pf_occlusion_gate),
        outlier_distance_m=float(args.pf_outlier_distance),
        resample_effective_ratio=float(args.pf_resample_effective_ratio),
        max_prediction_frames=int(args.pf_max_prediction_frames),
        max_motion_noise_scale=float(args.pf_max_motion_noise_scale),
    )


@dataclass
class MeasurementProcessingState:
    last_filter_time: float | None = None
    last_smoothed_measurement_nodes: np.ndarray | None = None
    last_detection_hint_xy: np.ndarray | None = None
    last_filter_nodes: np.ndarray | None = None
    last_filter_estimate: object | None = None
    last_filter_result: object | None = None
    last_filter_lost_frames: int = 0


@dataclass
class MeasurementTask:
    frame_count: int
    bgr: np.ndarray
    point_cloud: np.ndarray
    confidence_measure: np.ndarray | None


@dataclass
class MeasurementProcessingResult:
    frame_count: int
    detection: object | None
    measurement: object | None
    raw_measurement: object | None
    estimate: object | None
    filter_result: object | None
    tracking_diagnostics: dict
    stage_seconds: dict
    state: MeasurementProcessingState


class AsyncMeasurementWorker:
    def __init__(self, args, cable_detector, particle_filter):
        self.args = args
        self.cable_detector = cable_detector
        self.particle_filter = particle_filter
        self.state = MeasurementProcessingState()
        self.tasks = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_result = None
        self.in_flight = False
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, name="cable-measurement-worker", daemon=True)
        self.thread.start()

    def busy(self):
        with self.lock:
            return self.in_flight or not self.tasks.empty()

    def submit(self, task):
        with self.lock:
            if self.in_flight or not self.tasks.empty():
                self.dropped += 1
                return False
            self.in_flight = True
            self.submitted += 1
        try:
            self.tasks.put_nowait(task)
            return True
        except queue.Full:
            with self.lock:
                self.in_flight = False
                self.dropped += 1
            return False

    def drain_latest(self):
        with self.lock:
            result = self.latest_result
            self.latest_result = None
            return result

    def counters(self):
        with self.lock:
            return {
                "submitted": int(self.submitted),
                "completed": int(self.completed),
                "dropped": int(self.dropped),
                "busy": bool(self.in_flight or not self.tasks.empty()),
            }

    def stop(self):
        try:
            self.tasks.put(None, timeout=2.0)
        except queue.Full:
            return
        self.thread.join(timeout=2.0)

    def _run(self):
        while True:
            task = self.tasks.get()
            if task is None:
                return
            try:
                result = process_measurement_frame(
                    self.args,
                    self.cable_detector,
                    self.particle_filter,
                    self.state,
                    task.frame_count,
                    task.bgr,
                    task.point_cloud,
                    task.confidence_measure,
                )
                with self.lock:
                    self.latest_result = result
                    self.completed += 1
            except Exception as exc:
                print(f"Async measurement worker error: {exc}")
            finally:
                with self.lock:
                    self.in_flight = False


def process_measurement_frame(args, cable_detector, particle_filter, state, frame_count, bgr, point_cloud, confidence_measure):
    stage_seconds = {"detect": 0.0, "fit": 0.0, "filter": 0.0}
    detection = None
    measurement = None
    raw_measurement = None
    estimate = None
    filter_result = None
    tracking_diagnostics = {}
    smoothing_diagnostics = {}

    stage_start = time.monotonic()
    roi_bbox = None
    use_roi = (
        bool(args.detector_roi)
        and state.last_detection_hint_xy is not None
        and (args.detector_full_every == 0 or int(frame_count) % int(args.detector_full_every) != 0)
    )
    if use_roi:
        roi_bbox = centerline_roi_bbox(state.last_detection_hint_xy, bgr.shape[:2], args.detector_roi_padding)
    mask_only_detection = should_use_hsv_mask_only(
        args,
        frame_count,
        state.last_filter_nodes,
        state.last_filter_lost_frames,
    )
    pidnet_mask_point_measurement = args.detector_backend == "pidnet"
    extract_geometry = not (mask_only_detection or pidnet_mask_point_measurement)
    detection = cable_detector.detect(
        bgr,
        roi_bbox=roi_bbox,
        scale=args.detector_scale,
        extract_geometry=extract_geometry,
    )
    if roi_bbox is not None and extract_geometry and len(detection.centerline_xy) < 2:
        detection = cable_detector.detect(bgr, scale=args.detector_scale)
    if detection is not None and (pidnet_mask_point_measurement or len(detection.centerline_xy) >= 2):
        if len(detection.centerline_xy) >= 2:
            state.last_detection_hint_xy = detection.centerline_xy
    stage_seconds["detect"] += time.monotonic() - stage_start

    stage_start = time.monotonic()
    reference_nodes = measurement_reference_nodes(
        state.last_filter_nodes,
        state.last_filter_lost_frames,
        reacquire_after=args.measurement_gate_reacquire_after,
    )
    reference_gate = measurement_reference_gate(
        args.measurement_prediction_gate,
        state.last_filter_lost_frames,
    )
    if pidnet_mask_point_measurement and bool(args.measurement_mask_centerline):
        measurement = mask_cloud_centerline_measurement(
            point_cloud,
            detection,
            args,
            reference_nodes=reference_nodes,
            confidence_measure=confidence_measure,
        )
    elif mask_only_detection or pidnet_mask_point_measurement:
        support_reference_nodes = None if pidnet_mask_point_measurement else reference_nodes
        support_reference_gate = 0.0 if pidnet_mask_point_measurement else reference_gate
        measurement = cable_measurement_from_mask_points(
            point_cloud,
            detection,
            segment_count=args.cable_segments,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            confidence_map=confidence_measure,
            max_confidence=args.cable_confidence_max if args.cable_confidence_max >= 0.0 else None,
            max_points=args.hsv_mask_points if mask_only_detection else args.pf_measurement_points,
            reference_nodes=support_reference_nodes,
            reference_gate_m=support_reference_gate,
            reference_min_points=args.pf_min_measurement_points,
        )
    else:
        measurement = fit_cable_segments_from_zed_point_cloud(
            point_cloud,
            detection,
            segment_count=args.cable_segments,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            confidence_map=confidence_measure,
            max_confidence=args.cable_confidence_max if args.cable_confidence_max >= 0.0 else None,
            node_search_px=args.node_search_px,
            source_point_mode="centerline",
            max_centerline_points=args.measurement_centerline_points,
            max_local_depth_std_m=args.measurement_local_depth_spread,
            reference_nodes=reference_nodes,
            reference_gate_m=reference_gate,
            reference_min_points=args.pf_min_measurement_points,
        )
    raw_measurement = measurement
    measurement, state.last_smoothed_measurement_nodes, smoothing_diagnostics = smooth_measurement_nodes(
        measurement,
        state.last_smoothed_measurement_nodes,
        args,
        state.last_filter_lost_frames,
    )
    stage_seconds["fit"] += time.monotonic() - stage_start

    now = time.monotonic()
    filter_dt = 1.0 / max(float(args.fps), 1.0) if state.last_filter_time is None else now - state.last_filter_time
    state.last_filter_time = now
    if particle_filter is not None:
        stage_start = time.monotonic()
        filter_result = particle_filter.step(measurement, filter_dt)
        estimate = filtered_cable_estimate(measurement, filter_result)
        stage_seconds["filter"] += time.monotonic() - stage_start
    else:
        estimate = measurement
    tracking_diagnostics = cable_tracking_diagnostics(
        state.last_filter_nodes,
        raw_measurement,
        measurement,
        estimate,
        filter_result,
        smoothing_diagnostics,
    )
    if estimate is not None:
        state.last_filter_nodes = np.asarray(estimate.points_xyz, dtype=np.float32)
        state.last_filter_estimate = estimate
        state.last_filter_result = filter_result
    elif particle_filter is None:
        state.last_filter_nodes = None
        state.last_filter_estimate = None
        state.last_filter_result = None
    if filter_result is not None:
        state.last_filter_lost_frames = int(filter_result.lost_frames)
    elif estimate is not None:
        state.last_filter_lost_frames = 0

    return MeasurementProcessingResult(
        frame_count=int(frame_count),
        detection=detection,
        measurement=measurement,
        raw_measurement=raw_measurement,
        estimate=estimate,
        filter_result=filter_result,
        tracking_diagnostics=tracking_diagnostics,
        stage_seconds=stage_seconds,
        state=replace(state),
    )


def snapshot_mat_data(mat):
    if mat is None:
        return None
    try:
        data = np.asarray(mat.get_data())
    except Exception:
        data = np.asarray(mat)
    if data.size == 0:
        return None
    return np.ascontiguousarray(data).copy()


def main():
    args = parse_args()
    run_live(args)


def run_live(args):
    cable_detector = load_cable_detector(args)
    if cable_detector is None:
        raise RuntimeError("Live cable tracking requires a trained PIDNet checkpoint.")
    particle_filter = (
        CableParticleFilter(node_count=args.cable_segments + 1, config=make_particle_filter_config(args))
        if args.particle_filter
        else None
    )

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

    frame_count = 0
    last_stats_time = time.monotonic()
    last_stats_frame_count = 0
    stats_compute_seconds = 0.0
    stats_compute_frames = 0
    latest_stats = empty_point_cloud_stats()
    latest_vertices = np.empty((0, 6), dtype=np.float32)
    latest_confidence_data = None
    latest_detection = None
    last_filter_nodes = None
    last_filter_estimate = None
    last_filter_result = None
    last_filter_lost_frames = 0
    processing_state = MeasurementProcessingState()
    async_worker = (
        AsyncMeasurementWorker(args, cable_detector, particle_filter)
        if bool(args.async_measurement)
        else None
    )
    last_visual_measurement = None
    last_visual_estimate = None
    last_visual_filter_result = None
    last_visual_age = 0
    latest_cable_status = "cable detector unavailable" if cable_detector is None else "waiting for cable"
    stage_seconds = {
        "capture": 0.0,
        "cloud": 0.0,
        "copy": 0.0,
        "detect": 0.0,
        "fit": 0.0,
        "filter": 0.0,
        "ui": 0.0,
    }
    async_busy_frames = 0

    try:
        while viewer.is_available():
            if zed.grab(runtime) <= sl.ERROR_CODE.SUCCESS:
                frame_start_time = time.monotonic()
                completed_result = async_worker.drain_latest() if async_worker is not None else None
                if completed_result is not None:
                    latest_detection = completed_result.detection or latest_detection
                    last_filter_nodes = completed_result.state.last_filter_nodes
                    last_filter_estimate = completed_result.state.last_filter_estimate
                    last_filter_result = completed_result.state.last_filter_result
                    last_filter_lost_frames = int(completed_result.state.last_filter_lost_frames)
                    processing_state = completed_result.state
                    for key, value in completed_result.stage_seconds.items():
                        if key in stage_seconds:
                            stage_seconds[key] += float(value)

                requested_measurement = should_update_measurement(args, frame_count, last_filter_nodes, last_filter_lost_frames)
                worker_busy = async_worker.busy() if async_worker is not None else False
                update_measurement = requested_measurement and not worker_busy
                if requested_measurement and worker_busy:
                    async_busy_frames += 1
                stage_start = frame_start_time
                zed.retrieve_image(image, sl.VIEW.LEFT)
                update_cloud_view = frame_count % args.cloud_update_every == 0
                update_point_cloud = update_cloud_view or update_measurement
                if update_point_cloud:
                    point_measure = sl.MEASURE.XYZRGBA if update_cloud_view else getattr(sl.MEASURE, "XYZ", sl.MEASURE.XYZRGBA)
                    zed.retrieve_measure(point_cloud, point_measure)
                if args.confidence_update_every == 0:
                    confidence_measure = None
                    latest_confidence_data = None
                elif update_measurement and frame_count % args.confidence_update_every == 0:
                    confidence_mat = retrieve_confidence_measure(zed, confidence_map)
                    latest_confidence_data = snapshot_mat_data(confidence_mat)
                    confidence_measure = latest_confidence_data
                else:
                    confidence_measure = latest_confidence_data
                stage_seconds["capture"] += time.monotonic() - stage_start

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
                stage_seconds["cloud"] += time.monotonic() - stage_start

                detection = latest_detection
                measurement = None
                raw_measurement = None
                tracking_diagnostics = {}
                estimate = last_filter_estimate
                filter_result = last_filter_result
                if completed_result is not None:
                    detection = completed_result.detection or latest_detection
                    measurement = completed_result.measurement
                    raw_measurement = completed_result.raw_measurement
                    estimate = completed_result.estimate
                    filter_result = completed_result.filter_result
                    tracking_diagnostics = completed_result.tracking_diagnostics

                if cable_detector is not None and update_measurement:
                    if async_worker is not None:
                        stage_start = time.monotonic()
                        point_cloud_data = snapshot_mat_data(point_cloud)
                        confidence_data = None if confidence_measure is None else np.ascontiguousarray(confidence_measure).copy()
                        if point_cloud_data is not None:
                            async_worker.submit(
                                MeasurementTask(
                                    frame_count=int(frame_count),
                                    bgr=np.ascontiguousarray(bgr).copy(),
                                    point_cloud=point_cloud_data,
                                    confidence_measure=confidence_data,
                                )
                            )
                        stage_seconds["copy"] += time.monotonic() - stage_start
                    else:
                        result = process_measurement_frame(
                            args,
                            cable_detector,
                            particle_filter,
                            processing_state,
                            frame_count,
                            bgr,
                            point_cloud,
                            confidence_measure,
                        )
                        latest_detection = result.detection or latest_detection
                        last_filter_nodes = result.state.last_filter_nodes
                        last_filter_estimate = result.state.last_filter_estimate
                        last_filter_result = result.state.last_filter_result
                        last_filter_lost_frames = int(result.state.last_filter_lost_frames)
                        processing_state = result.state
                        detection = result.detection or latest_detection
                        measurement = result.measurement
                        raw_measurement = result.raw_measurement
                        estimate = result.estimate
                        filter_result = result.filter_result
                        tracking_diagnostics = result.tracking_diagnostics
                        for key, value in result.stage_seconds.items():
                            if key in stage_seconds:
                                stage_seconds[key] += float(value)

                stage_start = time.monotonic()
                if measurement is not None:
                    last_visual_measurement = measurement
                if estimate is not None:
                    last_visual_estimate = estimate
                    last_visual_filter_result = filter_result
                    last_visual_age = 0
                else:
                    last_visual_age += 1
                    if last_visual_age > args.viewer_hold_frames:
                        last_visual_estimate = None
                        last_visual_filter_result = None
                        if measurement is None:
                            last_visual_measurement = None

                display_measurement = measurement
                display_estimate = estimate
                display_filter_result = filter_result
                if display_measurement is None and display_estimate is not None:
                    display_measurement = last_visual_measurement
                if (
                    display_estimate is None
                    and last_visual_estimate is not None
                    and last_visual_age <= args.viewer_hold_frames
                ):
                    display_measurement = last_visual_measurement
                    display_estimate = last_visual_estimate
                    display_filter_result = last_visual_filter_result

                debug_bgr = draw_cable_rgb_panel(
                    bgr,
                    detection=detection,
                    measurement=display_measurement,
                    estimate=display_estimate,
                    segment_count=args.cable_segments,
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
                    detector_name=args.detector_backend,
                    diagnostics=tracking_diagnostics,
                )

                frame_count += 1
                viewer.update_rgb_image(cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB))
                if update_cloud_view:
                    viewer.update_vertices(
                        latest_vertices,
                        f"live ZED point cloud | frame {frame_count} | {len(latest_vertices)} points | {latest_cable_status}",
                    )
                stage_seconds["ui"] += time.monotonic() - stage_start

                now = time.monotonic()
                stats_compute_seconds += max(0.0, now - frame_start_time)
                stats_compute_frames += 1
                if now - last_stats_time >= 1.0:
                    elapsed = max(now - last_stats_time, 1e-6)
                    render_fps = (frame_count - last_stats_frame_count) / elapsed
                    compute_ms = 1000.0 * stats_compute_seconds / max(stats_compute_frames, 1)
                    stage_ms = {
                        name: 1000.0 * value / max(stats_compute_frames, 1)
                        for name, value in stage_seconds.items()
                    }
                    async_text = ""
                    if async_worker is not None:
                        counters = async_worker.counters()
                        async_text = (
                            f" | async submitted {counters['submitted']} completed {counters['completed']} "
                            f"dropped {counters['dropped']} busy_frames {async_busy_frames} busy {int(counters['busy'])}"
                        )
                    print(
                        f"Frame {frame_count}: point cloud shape {latest_stats['shape']} | "
                        f"render {render_fps:.1f} fps compute {compute_ms:.1f} ms | "
                        f"sampled {latest_stats['sampled']} finite {latest_stats['finite']} "
                        f"in range {latest_stats['in_range']} returned {latest_stats['returned']} "
                        f"capped {latest_stats['capped']} | "
                        f"ms capture {stage_ms['capture']:.1f} cloud {stage_ms['cloud']:.1f} copy {stage_ms['copy']:.1f} "
                        f"detect {stage_ms['detect']:.1f} fit {stage_ms['fit']:.1f} "
                        f"filter {stage_ms['filter']:.1f} ui {stage_ms['ui']:.1f} | "
                        f"{latest_cable_status}{async_text}"
                    )
                    last_stats_time = now
                    last_stats_frame_count = frame_count
                    stats_compute_seconds = 0.0
                    stats_compute_frames = 0
                    for key in stage_seconds:
                        stage_seconds[key] = 0.0
                    async_busy_frames = 0

            viewer.poll()

    finally:
        if async_worker is not None:
            async_worker.stop()
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


def mask_cloud_centerline_measurement(point_cloud, detection, args, reference_nodes=None, confidence_measure=None):
    measurement = cable_measurement_from_mask_points(
        point_cloud,
        detection,
        segment_count=args.cable_segments,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        confidence_map=confidence_measure,
        max_confidence=args.cable_confidence_max if args.cable_confidence_max >= 0.0 else None,
        max_points=args.pf_measurement_points,
        reference_nodes=None,
        reference_gate_m=0.0,
        reference_min_points=args.pf_min_measurement_points,
    )
    if measurement is None:
        return None

    source_points = np.asarray(getattr(measurement, "source_points", np.empty((0, 3))), dtype=np.float32)
    nodes, method = ordered_centerline_from_mask_cloud(
        source_points,
        args,
        reference_nodes=reference_nodes,
    )
    if nodes is None:
        return measurement

    nodes = np.ascontiguousarray(nodes, dtype=np.float32)
    residual = polyline_residual(source_points, nodes)
    return replace(
        measurement,
        points_xyz=nodes,
        residual_m=float(residual),
        method=f"ordered mask-cloud 3D centerline ({method}) | {measurement.method}",
    )


def ordered_centerline_from_mask_cloud(source_points, args, reference_nodes=None):
    points = np.asarray(source_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return None, "none"
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < int(args.pf_min_measurement_points):
        return None, "too few support points"

    if bool(args.pf_reference_ordering) and reference_nodes is not None:
        nodes = fit_reference_ordered_point_cloud_chain(
            points,
            reference_nodes=reference_nodes,
            segment_count=args.cable_segments,
            gate_m=float(args.measurement_mask_centerline_gate),
            min_points=int(args.pf_min_measurement_points),
        )
        if mask_cloud_centerline_is_supported(points, nodes, args):
            return nodes, "reference projection"

    nodes = fit_unordered_point_cloud_chain(
        points,
        segment_count=args.cable_segments,
        endpoint_ordering=bool(args.pf_endpoint_ordering),
        max_points=int(args.pf_ordering_max_points),
        knn=int(args.pf_ordering_knn),
    )
    if mask_cloud_centerline_is_supported(points, nodes, args):
        return nodes, "endpoint graph"
    return None, "unsupported"


def mask_cloud_centerline_is_supported(source_points, nodes, args):
    if nodes is None:
        return False
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) < 2:
        return False
    if not np.all(np.isfinite(nodes[:, :3])):
        return False
    max_residual = float(args.measurement_mask_centerline_max_residual)
    if max_residual <= 0.0:
        return True
    residual = polyline_residual(source_points, nodes)
    return bool(np.isfinite(residual) and residual <= max_residual)


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


def measurement_reference_gate(base_gate_m, lost_frames):
    base_gate_m = float(base_gate_m)
    if base_gate_m <= 0.0:
        return 0.0
    scale = min(3.0, 1.0 + 0.25 * max(0, int(lost_frames)))
    return base_gate_m * scale


def should_use_hsv_mask_only(args, frame_count, last_filter_nodes, lost_frames):
    if args.detector_backend != "hsv" or not bool(args.hsv_fast_mask_only):
        return False
    if not bool(args.particle_filter):
        return False
    if last_filter_nodes is None:
        return False
    reacquire_after = int(args.measurement_gate_reacquire_after)
    if reacquire_after > 0 and int(lost_frames) >= reacquire_after:
        return False
    geometry_every = int(args.hsv_geometry_every)
    if geometry_every > 0 and int(frame_count) % geometry_every == 0:
        return False
    return True


def should_update_measurement(args, frame_count, last_filter_nodes, lost_frames):
    if last_filter_nodes is None:
        return True
    if int(lost_frames) > 0:
        return True
    if args.detector_full_every > 0 and int(frame_count) % int(args.detector_full_every) == 0:
        return True
    update_every = max(1, int(args.detector_update_every))
    return int(frame_count) % update_every == 0


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


def centerline_roi_bbox(centerline_xy, image_shape, padding):
    points = np.asarray(centerline_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        return None
    valid = np.all(np.isfinite(points[:, :2]), axis=1)
    points = points[valid, :2]
    if len(points) < 2:
        return None

    height, width = [int(v) for v in image_shape[:2]]
    padding = max(0, int(padding))
    x0 = int(np.floor(np.min(points[:, 0]))) - padding
    y0 = int(np.floor(np.min(points[:, 1]))) - padding
    x1 = int(np.ceil(np.max(points[:, 0]))) + padding + 1
    y1 = int(np.ceil(np.max(points[:, 1]))) + padding + 1
    return (
        int(np.clip(x0, 0, width)),
        int(np.clip(y0, 0, height)),
        int(np.clip(x1, 0, width)),
        int(np.clip(y1, 0, height)),
    )


def cable_status(detection, measurement, estimate, filter_result, segment_count, detector_name="detector", diagnostics=None):
    if detection is None:
        return "cable detector unavailable"

    diagnostics = dict(diagnostics or {})
    centerline_count = len(detection.centerline_xy)
    prefix = str(detector_name)
    if estimate is not None:
        residual = float(getattr(estimate, "residual_m", 0.0))
        source_count = 0 if measurement is None else len(getattr(measurement, "source_points", ()))
        mode = "measurement"
        if filter_result is not None:
            mode = "filtered" if filter_result.measurement_used else f"prediction lost={filter_result.lost_frames}"
            visible_segments = getattr(filter_result, "visible_segments", None)
            if visible_segments is not None:
                mode += f" visible={int(np.count_nonzero(visible_segments))}/{len(visible_segments)}"
            mode += f" support={int(getattr(filter_result, 'measurement_point_count', 0))}"
            segment_length = float(getattr(filter_result, "segment_length_m", np.nan))
            if np.isfinite(segment_length):
                mode += f" seglen={format_mm(segment_length)}"
            prediction_step = float(getattr(filter_result, "prediction_step_m", 0.0))
            if np.isfinite(prediction_step) and prediction_step > 0.0:
                mode += f" pred={format_mm(prediction_step)}"
        diag_text = format_tracking_diagnostics(diagnostics)
        return (
            f"{prefix} | {segment_count} segments | {mode} | residual {residual:.4f}m | "
            f"source={source_count} mask components {detection.component_count} "
            f"centerline {centerline_count} branches {detection.branch_count}{diag_text}"
        )
    if measurement is None:
        return (
            f"{prefix} | {segment_count} segments | waiting for valid 3D cable points | "
            f"mask components {detection.component_count} centerline {centerline_count} branches {detection.branch_count}"
        )
    return f"{prefix} | {segment_count} segments | waiting for filter | centerline {centerline_count}"


def format_tracking_diagnostics(diagnostics):
    if not diagnostics:
        return ""
    parts = []
    raw_to_filter = diagnostics.get("raw_to_filter_m", np.nan)
    if np.isfinite(raw_to_filter):
        parts.append(f"raw2f={format_mm(raw_to_filter)}")
    smooth_delta = diagnostics.get("smooth_delta_m", np.nan)
    smoothing = diagnostics.get("smoothing", "")
    if np.isfinite(smooth_delta):
        parts.append(f"smooth={format_mm(smooth_delta)}")
    elif smoothing in {"raw", "reset"}:
        parts.append(f"smooth={smoothing}")
    raw_residual = diagnostics.get("raw_residual_m", np.nan)
    if np.isfinite(raw_residual):
        parts.append(f"rawres={format_mm(raw_residual)}")
    filtered_residual = diagnostics.get("filtered_residual_m", np.nan)
    if np.isfinite(filtered_residual):
        parts.append(f"filtres={format_mm(filtered_residual)}")
    proposal_ratio = diagnostics.get("proposal_ratio", np.nan)
    if np.isfinite(proposal_ratio):
        parts.append(f"prop={proposal_ratio:.2f}")
    return "" if not parts else " | " + " ".join(parts)


def draw_cable_rgb_panel(
    bgr,
    detection=None,
    measurement=None,
    estimate=None,
    segment_count=2,
    mode="segmentation",
    detector_description="PIDNet mask threshold 0.50",
):
    if mode == "mask":
        return draw_cable_mask_view(
            bgr,
            detection=detection,
            measurement=measurement,
            estimate=estimate,
            segment_count=segment_count,
            detector_description=detector_description,
        )
    if mode == "tracking":
        return draw_cable_debug_overlay(
            bgr,
            detection=detection,
            measurement=measurement,
            estimate=estimate,
            segment_count=segment_count,
        )
    return draw_cable_segmentation_view(
        bgr,
        detection=detection,
        measurement=measurement,
        estimate=estimate,
        segment_count=segment_count,
        detector_description=detector_description,
    )


def draw_cable_debug_overlay(bgr, detection=None, measurement=None, estimate=None, segment_count=2):
    panel = np.asarray(bgr, dtype=np.uint8).copy()
    if detection is not None:
        if detection.mask is not None and np.any(detection.mask):
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
    measurement=None,
    estimate=None,
    segment_count=2,
    detector_description="PIDNet mask threshold 0.50",
):
    bgr = np.asarray(bgr, dtype=np.uint8)
    panel = cv2.addWeighted(bgr, 0.50, np.zeros_like(bgr), 0.50, 0.0)
    if detection is not None and detection.mask is not None and np.any(detection.mask):
        mask = detection.mask > 0
        tint = np.zeros_like(panel)
        tint[:, :, 1] = 230
        tint[:, :, 2] = 130
        panel[mask] = cv2.addWeighted(panel[mask], 0.28, tint[mask], 0.72, 0.0)

        contours, _hierarchy = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    draw_detection_geometry(panel, detection, segment_count)
    draw_cable_status_text(panel, detection, measurement, estimate, segment_count, mode_text=detector_description)
    return panel


def draw_cable_mask_view(
    bgr,
    detection=None,
    measurement=None,
    estimate=None,
    segment_count=2,
    detector_description="PIDNet mask threshold 0.50",
):
    shape = np.asarray(bgr, dtype=np.uint8).shape[:2]
    panel = np.full((shape[0], shape[1], 3), 18, dtype=np.uint8)
    if detection is not None and detection.mask is not None and np.any(detection.mask):
        panel[detection.mask > 0] = (255, 255, 255)
    if detection is not None and detection.skeleton is not None and np.any(detection.skeleton):
        ys, xs = np.nonzero(detection.skeleton)
        panel[ys, xs] = (0, 180, 255)
    draw_detection_geometry(panel, detection, segment_count)
    draw_cable_status_text(panel, detection, measurement, estimate, segment_count, mode_text=detector_description)
    return panel


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
