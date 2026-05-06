import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

try:
    from .trackdlo_tracker import (
        TrackDLOParams,
        TrackDLOTracker,
        choose_projection_convention,
        project_points_to_image,
    )
    from .cable_vision import (
        DEFAULT_PROFILE_PATH,
        MIN_COMPONENT_AREA,
        MIN_NODE_XYZ_SAMPLES,
        NODE_COUNT,
        NODE_XYZ_FALLBACK_RADIUS,
        NODE_XYZ_RADIUS,
        REINIT_THRESHOLD_PX,
        SMOOTH_ALPHA,
        attach_3d_observations,
        clean_cable_mask,
        create_raw_mask,
        extract_centerline,
        load_hsv_ranges,
        NodeSmoother,
    )
    from .zed_depth_gl_viewer import ZedDepthGLViewer
    from .zed_spatial_mapping import (
        DEPTH_MODES,
        MAPPING_RANGES,
        MAPPING_RESOLUTIONS,
        RESOLUTIONS,
        configure_input_source,
        configure_viewer_from_zed,
        fused_point_cloud_to_vertices,
        get_left_camera_intrinsics,
        live_point_cloud_to_vertices,
        make_spatial_mapping_parameters,
    )
except ImportError:
    from trackdlo_tracker import (
        TrackDLOParams,
        TrackDLOTracker,
        choose_projection_convention,
        project_points_to_image,
    )
    from cable_vision import (
        DEFAULT_PROFILE_PATH,
        MIN_COMPONENT_AREA,
        MIN_NODE_XYZ_SAMPLES,
        NODE_COUNT,
        NODE_XYZ_FALLBACK_RADIUS,
        NODE_XYZ_RADIUS,
        REINIT_THRESHOLD_PX,
        SMOOTH_ALPHA,
        attach_3d_observations,
        clean_cable_mask,
        create_raw_mask,
        extract_centerline,
        load_hsv_ranges,
        NodeSmoother,
    )
    from zed_depth_gl_viewer import ZedDepthGLViewer
    from zed_spatial_mapping import (
        DEPTH_MODES,
        MAPPING_RANGES,
        MAPPING_RESOLUTIONS,
        RESOLUTIONS,
        configure_input_source,
        configure_viewer_from_zed,
        fused_point_cloud_to_vertices,
        get_left_camera_intrinsics,
        live_point_cloud_to_vertices,
        make_spatial_mapping_parameters,
    )


MAX_MASK_POINT_SAMPLES = 1500
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "trackdlo_config.json"

DEFAULT_CONFIG = {
    "profile": str(DEFAULT_PROFILE_PATH),
    "input_svo_file": "",
    "ip_address": "",
    "resolution": "HD720",
    "fps": 30,
    "depth_mode": "NEURAL_PLUS",
    "depth_min": 0.1,
    "depth_max": 1.8,
    "confidence": 100,
    "texture_confidence": 100,
    "auto_camera_controls": True,
    "depth_fill": True,
    "imu_required": True,
    "mapping_resolution": "MEDIUM",
    "mapping_range": "MEDIUM",
    "max_memory_mb": 2048,
    "update_period": 0.5,
    "max_map_points": 0,
    "min_fused_display_vertices": 5000,
    "live_fallback": True,
    "live_stride": 1,
    "live_max_points": 0,
    "nodes": NODE_COUNT,
    "smooth_alpha": SMOOTH_ALPHA,
    "reinit_threshold": REINIT_THRESHOLD_PX,
    "node_xyz_radius": NODE_XYZ_RADIUS,
    "node_xyz_fallback_radius": NODE_XYZ_FALLBACK_RADIUS,
    "min_node_xyz_samples": MIN_NODE_XYZ_SAMPLES,
    "max_mask_points": MAX_MASK_POINT_SAMPLES,
    "min_component_area": MIN_COMPONENT_AREA,
    "keep_largest_component": False,
    "max_components": 0,
    "trackdlo": True,
    "trackdlo_beta": 0.35,
    "trackdlo_lambda": 50000.0,
    "trackdlo_visibility_threshold": 0.008,
    "temporal_prediction": True,
    "prediction_gain": 0.75,
    "prediction_velocity_alpha": 0.5,
    "prediction_velocity_decay": 0.85,
    "max_prediction_step_m": 0.035,
    "crossing_lock": True,
    "crossing_lock_distance_m": 0.025,
    "crossing_lock_projection_px": 12.0,
    "crossing_lock_depth_margin_m": 0.06,
    "crossing_lock_frames": 10,
    "crossing_lock_window_nodes": 2,
    "crossing_lock_prior_weight": 4.0,
    "crossing_lock_min_edge_gap": 4,
    "crossing_lock_max_pairs": 6,
    "dlo_pixel_width": 5,
    "cable_diameter_m": 0.005,
    "fixed_cable_length_m": 0.0,
    "enforce_cable_length": True,
    "init_min_valid_ratio": 0.9,
    "init_require_endpoints": True,
    "init_stable_frames": 12,
    "init_length_std_m": 0.015,
    "min_projected_dlo_width_px": 3.0,
    "max_projected_dlo_width_px": 18.0,
    "self_occlusion_depth_margin": 0.02,
    "self_occlusion_max_projection_error": 18.0,
    "point_size": 3.0,
    "debug_width": 620,
    "gl_width": 1180,
    "gl_height": 900,
}


def load_config(config_path: Path):
    config_path = Path(config_path)
    config = DEFAULT_CONFIG.copy()
    if not config_path.exists():
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown)}")

    config.update(loaded)
    profile_path = Path(config["profile"])
    if not profile_path.is_absolute():
        config["profile"] = str(SCRIPT_DIR / profile_path)
    return config


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, remaining_argv = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="TrackDLO cable reconstruction inside a ZED spatial map.",
        parents=[config_parser],
    )
    parser.set_defaults(config=config_args.config)
    parser.add_argument("--profile", type=Path, default=Path(config["profile"]))
    parser.add_argument("--input-svo-file", default=config["input_svo_file"])
    parser.add_argument("--ip-address", default=config["ip_address"])
    parser.add_argument("--resolution", choices=RESOLUTIONS, default=config["resolution"])
    parser.add_argument("--fps", type=int, default=config["fps"])
    parser.add_argument("--depth-mode", choices=DEPTH_MODES, default=config["depth_mode"])
    parser.add_argument("--depth-min", type=float, default=config["depth_min"])
    parser.add_argument("--depth-max", type=float, default=config["depth_max"])
    parser.add_argument("--confidence", type=int, default=config["confidence"])
    parser.add_argument("--texture-confidence", type=int, default=config["texture_confidence"])
    parser.add_argument(
        "--auto-camera-controls",
        action=argparse.BooleanOptionalAction,
        default=config["auto_camera_controls"],
        help="Enable all available ZED auto camera controls: exposure/gain and white balance.",
    )
    parser.add_argument(
        "--depth-fill",
        action=argparse.BooleanOptionalAction,
        default=config["depth_fill"],
        help="Enable ZED depth fill mode to keep the live cloud dense.",
    )
    parser.add_argument(
        "--imu-required",
        action=argparse.BooleanOptionalAction,
        default=config["imu_required"],
        help="Require ZED sensors and use IMU orientation fusion.",
    )
    parser.add_argument(
        "--disable-depth-fill",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-no-imu",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--mapping-resolution", choices=MAPPING_RESOLUTIONS, default=config["mapping_resolution"])
    parser.add_argument("--mapping-range", choices=MAPPING_RANGES, default=config["mapping_range"])
    parser.add_argument("--max-memory-mb", type=int, default=config["max_memory_mb"])
    parser.add_argument("--update-period", type=float, default=config["update_period"])
    parser.add_argument("--max-map-points", type=int, default=config["max_map_points"], help="0 keeps every fused map point.")
    parser.add_argument(
        "--min-fused-display-vertices",
        type=int,
        default=config["min_fused_display_vertices"],
        help="Keep showing live depth until the fused spatial map has at least this many vertices.",
    )
    parser.add_argument(
        "--live-fallback",
        action=argparse.BooleanOptionalAction,
        default=config["live_fallback"],
    )
    parser.add_argument("--disable-live-fallback", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--live-stride", type=int, default=config["live_stride"])
    parser.add_argument("--live-max-points", type=int, default=config["live_max_points"], help="0 keeps every finite live depth point.")

    parser.add_argument("--nodes", type=int, default=config["nodes"])
    parser.add_argument("--smooth-alpha", type=float, default=config["smooth_alpha"])
    parser.add_argument("--reinit-threshold", type=float, default=config["reinit_threshold"])
    parser.add_argument("--node-xyz-radius", type=int, default=config["node_xyz_radius"])
    parser.add_argument("--node-xyz-fallback-radius", type=int, default=config["node_xyz_fallback_radius"])
    parser.add_argument("--min-node-xyz-samples", type=int, default=config["min_node_xyz_samples"])
    parser.add_argument("--max-mask-points", type=int, default=config["max_mask_points"])
    parser.add_argument("--min-component-area", type=int, default=config["min_component_area"])
    parser.add_argument(
        "--keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=config["keep_largest_component"],
        help="Keep only the largest segmented cable component. Disable this for TrackDLO mid-section occlusion.",
    )
    parser.add_argument("--max-components", type=int, default=config["max_components"], help="0 keeps every component above min area.")

    parser.add_argument(
        "--trackdlo",
        action=argparse.BooleanOptionalAction,
        default=config["trackdlo"],
    )
    parser.add_argument("--disable-trackdlo", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trackdlo-beta", type=float, default=config["trackdlo_beta"])
    parser.add_argument("--trackdlo-lambda", type=float, default=config["trackdlo_lambda"])
    parser.add_argument("--trackdlo-visibility-threshold", type=float, default=config["trackdlo_visibility_threshold"])
    parser.add_argument("--temporal-prediction", action=argparse.BooleanOptionalAction, default=config["temporal_prediction"])
    parser.add_argument("--prediction-gain", type=float, default=config["prediction_gain"])
    parser.add_argument("--prediction-velocity-alpha", type=float, default=config["prediction_velocity_alpha"])
    parser.add_argument("--prediction-velocity-decay", type=float, default=config["prediction_velocity_decay"])
    parser.add_argument("--max-prediction-step-m", type=float, default=config["max_prediction_step_m"])
    parser.add_argument("--crossing-lock", action=argparse.BooleanOptionalAction, default=config["crossing_lock"])
    parser.add_argument("--crossing-lock-distance-m", type=float, default=config["crossing_lock_distance_m"])
    parser.add_argument("--crossing-lock-projection-px", type=float, default=config["crossing_lock_projection_px"])
    parser.add_argument("--crossing-lock-depth-margin-m", type=float, default=config["crossing_lock_depth_margin_m"])
    parser.add_argument("--crossing-lock-frames", type=int, default=config["crossing_lock_frames"])
    parser.add_argument("--crossing-lock-window-nodes", type=int, default=config["crossing_lock_window_nodes"])
    parser.add_argument("--crossing-lock-prior-weight", type=float, default=config["crossing_lock_prior_weight"])
    parser.add_argument("--crossing-lock-min-edge-gap", type=int, default=config["crossing_lock_min_edge_gap"])
    parser.add_argument("--crossing-lock-max-pairs", type=int, default=config["crossing_lock_max_pairs"])
    parser.add_argument("--dlo-pixel-width", type=int, default=config["dlo_pixel_width"])
    parser.add_argument("--cable-diameter-m", type=float, default=config["cable_diameter_m"])
    parser.add_argument(
        "--fixed-cable-length-m",
        type=float,
        default=config["fixed_cable_length_m"],
        help="Known total cable length in meters. Use 0 to learn the fixed length at initialization.",
    )
    parser.add_argument(
        "--enforce-cable-length",
        action=argparse.BooleanOptionalAction,
        default=config["enforce_cable_length"],
    )
    parser.add_argument(
        "--init-min-valid-ratio",
        type=float,
        default=config["init_min_valid_ratio"],
        help="Fraction of 3D cable nodes required before learning startup cable length.",
    )
    parser.add_argument(
        "--init-require-endpoints",
        action=argparse.BooleanOptionalAction,
        default=config["init_require_endpoints"],
        help="Require both sampled cable endpoints before startup length calibration can finish.",
    )
    parser.add_argument("--init-stable-frames", type=int, default=config["init_stable_frames"])
    parser.add_argument("--init-length-std-m", type=float, default=config["init_length_std_m"])
    parser.add_argument("--min-projected-dlo-width-px", type=float, default=config["min_projected_dlo_width_px"])
    parser.add_argument("--max-projected-dlo-width-px", type=float, default=config["max_projected_dlo_width_px"])
    parser.add_argument("--self-occlusion-depth-margin", type=float, default=config["self_occlusion_depth_margin"])
    parser.add_argument("--self-occlusion-max-projection-error", type=float, default=config["self_occlusion_max_projection_error"])

    parser.add_argument("--point-size", type=float, default=config["point_size"])
    parser.add_argument("--debug-width", type=int, default=config["debug_width"], help="Left RGB/debug panel width in the combined UI.")
    parser.add_argument("--gl-width", type=int, default=config["gl_width"], help="Right 3D point-cloud panel width in the combined UI.")
    parser.add_argument("--gl-height", type=int, default=config["gl_height"], help="Combined UI window height.")

    args = parser.parse_args(remaining_argv)
    if args.disable_depth_fill:
        args.depth_fill = False
    if args.allow_no_imu:
        args.imu_required = False
    if args.disable_live_fallback:
        args.live_fallback = False
    if args.disable_trackdlo:
        args.trackdlo = False

    if args.input_svo_file and args.ip_address:
        raise ValueError("Specify only one input source: --input-svo-file or --ip-address.")
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
    if hasattr(init, "sensors_required"):
        init.sensors_required = args.imu_required
    configure_input_source(init, args)

    zed = sl.Camera()
    status = zed.open(init)
    print("Open status:", status)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")
    if args.auto_camera_controls:
        apply_auto_camera_controls(zed)
    return zed


def apply_auto_camera_controls(zed):
    def safe_set(setting_name, value):
        if not hasattr(sl.VIDEO_SETTINGS, setting_name):
            print(f"Auto camera setting unavailable: {setting_name}")
            return

        setting = getattr(sl.VIDEO_SETTINGS, setting_name)
        try:
            status = zed.set_camera_settings(setting, value)
            print(f"Auto camera setting {setting_name}={value}: {status}")
        except Exception as exc:
            print(f"Could not set auto camera setting {setting_name}: {exc}")

    print("Applying ZED auto camera controls...")
    safe_set("AEC_AGC", 1)
    safe_set("EXPOSURE", -1)
    safe_set("GAIN", -1)
    safe_set("WHITEBALANCE_AUTO", 1)


def make_runtime_parameters(args):
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = args.confidence
    runtime.texture_confidence_threshold = args.texture_confidence
    runtime.remove_saturated_areas = False
    if hasattr(runtime, "enable_fill_mode"):
        runtime.enable_fill_mode = args.depth_fill
    return runtime


def transform_points(transform, points, preserve_shape=False):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)

    xyz = points[:, :3]
    valid = np.all(np.isfinite(xyz), axis=1)
    if preserve_shape:
        output = np.full((len(xyz), 3), np.nan, dtype=np.float32)
    else:
        output = np.empty((np.count_nonzero(valid), 3), dtype=np.float32)

    if not np.any(valid):
        return output

    matrix = np.asarray(transform.m, dtype=np.float32)
    homogeneous = np.ones((np.count_nonzero(valid), 4), dtype=np.float32)
    homogeneous[:, :3] = xyz[valid]
    transformed = (matrix @ homogeneous.T).T[:, :3].astype(np.float32)

    if preserve_shape:
        output[valid] = transformed
    else:
        output[:] = transformed
    return output


def update_spatial_map(
    zed,
    fused_cloud,
    request_pending,
    last_request_time,
    update_period,
):
    now = time.monotonic()
    request_status = sl.ERROR_CODE.SUCCESS
    retrieve_status = sl.ERROR_CODE.SUCCESS
    did_update = False

    if not request_pending and now - last_request_time >= update_period:
        zed.request_spatial_map_async()
        request_pending = True
        last_request_time = now

    if request_pending:
        request_status = zed.get_spatial_map_request_status_async()
        if request_status == sl.ERROR_CODE.SUCCESS:
            retrieve_status = zed.retrieve_spatial_map_async(fused_cloud)
            request_pending = False
            did_update = True

    return request_pending, last_request_time, request_status, retrieve_status, did_update


def fused_point_count(fused_cloud):
    try:
        return int(fused_cloud.get_number_of_points())
    except Exception:
        return 0


def camera_position_from_pose(transform):
    matrix = np.asarray(transform.m, dtype=np.float32)
    if matrix.shape != (4, 4):
        return np.zeros(3, dtype=np.float32)
    return matrix[:3, 3].astype(np.float32)


def clip_vertices_by_distance(vertices, origin, depth_min, depth_max):
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 6 or len(vertices) == 0:
        return np.empty((0, 6), dtype=np.float32)

    origin = np.asarray(origin, dtype=np.float32).reshape(1, 3)
    xyz = vertices[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    distances = np.linalg.norm(xyz - origin, axis=1)
    valid = finite & (distances >= float(depth_min)) & (distances <= float(depth_max))
    return np.ascontiguousarray(vertices[valid], dtype=np.float32)


def project_tracked_nodes_to_rgb(
    nodes_xyz,
    camera_intrinsics,
    image_shape,
    observed_nodes_xyz,
    observed_valid_nodes,
    observed_nodes_xy,
):
    nodes_xyz = np.asarray(nodes_xyz, dtype=np.float64)
    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] < 3 or len(nodes_xyz) == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    convention = choose_projection_convention(
        camera_intrinsics,
        image_shape,
        observed_nodes_xyz,
        observed_valid_nodes,
        observed_nodes_xy,
    )
    pixels, valid, _depth = project_points_to_image(
        nodes_xyz,
        camera_intrinsics,
        image_shape,
        convention,
    )
    return pixels.astype(np.float32), valid


def make_rgb_debug_overlay(
    bgr,
    mask,
    centerline,
    observations,
    component_count,
    frame_count,
    tracker_status,
    tracked_nodes_xy=None,
    tracked_nodes_valid=None,
    tracked_visible=None,
    tracked_extended_visible=None,
):
    image = bgr.copy()
    mask_pixels = int(np.count_nonzero(mask))

    if mask_pixels > 0:
        tint = np.zeros_like(image)
        tint[:, :, 1] = 70
        tint[:, :, 2] = 255
        tinted = cv2.addWeighted(image, 0.68, tint, 0.32, 0.0)
        image[mask > 0] = tinted[mask > 0]

    skeleton = centerline.get("skeleton")
    if skeleton is not None and np.any(skeleton):
        image[skeleton > 0] = (245, 245, 245)

    path_xy = centerline.get("path_xy", np.empty((0, 2), dtype=np.float32))
    if len(path_xy) >= 2:
        path = np.round(path_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [path], False, (0, 255, 120), 2, cv2.LINE_AA)

    raw_nodes = centerline.get("raw_nodes_xy", np.empty((0, 2), dtype=np.float32))
    if len(raw_nodes) >= 2:
        raw_path = np.round(raw_nodes).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [raw_path], False, (60, 170, 255), 1, cv2.LINE_AA)

    nodes_xy = centerline.get("nodes_xy", np.empty((0, 2), dtype=np.float32))
    valid_nodes = observations.get("valid_nodes", np.zeros(len(nodes_xy), dtype=bool))
    for idx, node in enumerate(nodes_xy):
        if not np.all(np.isfinite(node)):
            continue
        x, y = int(round(node[0])), int(round(node[1]))
        valid = idx < len(valid_nodes) and bool(valid_nodes[idx])
        color = (20, 240, 255) if valid else (80, 90, 105)
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 6, (8, 14, 20), 1, cv2.LINE_AA)

    draw_tracked_model_overlay(
        image,
        tracked_nodes_xy,
        tracked_nodes_valid,
        tracked_visible,
        tracked_extended_visible,
    )

    text_lines = [
        f"Frame {frame_count}   Mask {mask_pixels}   Visible pieces {component_count}",
        f"Tracked cable 1   Init nodes {observations['valid_node_count']}/{len(nodes_xy)}   X_t {observations['point_count']}",
        tracker_status,
    ]
    _draw_status_strip(image, text_lines)
    return image


def draw_tracked_model_overlay(
    image,
    tracked_nodes_xy,
    tracked_nodes_valid,
    tracked_visible,
    tracked_extended_visible,
):
    if tracked_nodes_xy is None or tracked_nodes_valid is None:
        return

    tracked_nodes_xy = np.asarray(tracked_nodes_xy, dtype=np.float32)
    tracked_nodes_valid = np.asarray(tracked_nodes_valid, dtype=bool).reshape(-1)
    if tracked_nodes_xy.ndim != 2 or tracked_nodes_xy.shape[1] < 2:
        return
    if len(tracked_nodes_xy) != len(tracked_nodes_valid):
        return

    tracked_visible = _mask_or_default(tracked_visible, len(tracked_nodes_xy), tracked_nodes_valid)
    tracked_extended_visible = _mask_or_default(tracked_extended_visible, len(tracked_nodes_xy), tracked_visible)
    tracked_extended_visible = tracked_extended_visible | tracked_visible

    for idx in range(len(tracked_nodes_xy) - 1):
        if not (tracked_nodes_valid[idx] and tracked_nodes_valid[idx + 1]):
            continue
        p0 = tracked_nodes_xy[idx]
        p1 = tracked_nodes_xy[idx + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        color = _tracked_segment_bgr(idx, tracked_visible, tracked_extended_visible)
        cv2.line(
            image,
            tuple(np.round(p0).astype(int)),
            tuple(np.round(p1).astype(int)),
            color,
            4,
            cv2.LINE_AA,
        )

    for idx, point in enumerate(tracked_nodes_xy):
        if not (tracked_nodes_valid[idx] and np.all(np.isfinite(point))):
            continue
        x, y = tuple(np.round(point).astype(int))
        color = _tracked_node_bgr(idx, tracked_visible, tracked_extended_visible)
        cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 7, (6, 10, 14), 1, cv2.LINE_AA)


def _mask_or_default(mask, size, default):
    if mask is None:
        if isinstance(default, np.ndarray):
            return np.asarray(default, dtype=bool).copy()
        return np.full(size, bool(default), dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(mask) != size:
        return np.zeros(size, dtype=bool)
    return mask


def _tracked_node_bgr(idx, visible, extended_visible):
    if visible[idx]:
        return (80, 255, 120)
    if extended_visible[idx]:
        return (35, 210, 255)
    return (40, 55, 255)


def _tracked_segment_bgr(idx, visible, extended_visible):
    if visible[idx] and visible[idx + 1]:
        return (60, 235, 110)
    if extended_visible[idx] and extended_visible[idx + 1]:
        return (30, 190, 255)
    return (40, 55, 255)


def _draw_status_strip(image, lines):
    height, width = image.shape[:2]
    strip_h = min(height, 18 + 20 * len(lines))
    y0 = height - strip_h
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y0), (width, height), (10, 14, 18), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)
    cv2.line(image, (0, y0), (width, y0), (54, 66, 76), 1, cv2.LINE_AA)

    y = y0 + 20
    for idx, line in enumerate(lines):
        color = (235, 241, 246) if idx == 0 else (178, 190, 202)
        cv2.putText(
            image,
            str(line)[:110],
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20


def main():
    args = parse_args()
    hsv_ranges = load_hsv_ranges(args.profile)
    print(f"Loaded {len(hsv_ranges)} HSV ranges from {args.profile}")
    print(
        f"ZED settings: {args.resolution}@{args.fps}, depth {args.depth_mode}, "
        f"view {args.depth_min:.2f}-{args.depth_max:.2f} m, "
        f"confidence {args.confidence}, texture confidence {args.texture_confidence}, "
        f"auto camera controls {args.auto_camera_controls}, IMU required {args.imu_required}"
    )
    length_mode = (
        f"{args.fixed_cable_length_m:.3f} m fixed"
        if args.fixed_cable_length_m > 0.0
        else "learned during startup calibration"
    )
    print(
        f"Cable model: diameter {args.cable_diameter_m * 1000.0:.1f} mm, "
        f"length {length_mode}, "
        f"length constraint {args.enforce_cable_length}, "
        f"init {args.init_min_valid_ratio:.0%} nodes, endpoints {args.init_require_endpoints}, "
        f"{args.init_stable_frames} stable frames, "
        f"prediction {args.temporal_prediction} gain {args.prediction_gain:.2f} "
        f"max {args.max_prediction_step_m * 1000.0:.0f}mm, "
        f"crossing lock {args.crossing_lock} {args.crossing_lock_distance_m * 1000.0:.0f}mm/"
        f"{args.crossing_lock_projection_px:.0f}px, "
        f"projected width {args.min_projected_dlo_width_px:.1f}-{args.max_projected_dlo_width_px:.1f}px "
        f"(fallback {args.dlo_pixel_width}px)"
    )

    zed = open_zed(args)
    runtime = make_runtime_parameters(args)
    image = sl.Mat()
    point_cloud = sl.Mat()
    pose = sl.Pose()
    fused_cloud = sl.FusedPointCloud()

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_imu_fusion = True
    status = zed.enable_positional_tracking(tracking_params)
    print("Enable positional tracking:", status)
    print("IMU orientation fusion:", tracking_params.enable_imu_fusion)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not enable positional tracking: {status}")
    zed.reset_positional_tracking(sl.Transform())

    mapping_params = make_spatial_mapping_parameters(args)
    status = zed.enable_spatial_mapping(mapping_params)
    print("Enable spatial mapping:", status)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not enable spatial mapping: {status}")

    tracker = None
    if args.trackdlo:
        tracker = TrackDLOTracker(
            args.nodes,
            TrackDLOParams(
                beta=args.trackdlo_beta,
                lam=args.trackdlo_lambda,
                visibility_threshold=args.trackdlo_visibility_threshold,
                temporal_prediction=args.temporal_prediction,
                prediction_gain=args.prediction_gain,
                prediction_velocity_alpha=args.prediction_velocity_alpha,
                prediction_velocity_decay=args.prediction_velocity_decay,
                max_prediction_step_m=args.max_prediction_step_m,
                crossing_lock=args.crossing_lock,
                crossing_lock_distance_m=args.crossing_lock_distance_m,
                crossing_lock_projection_px=args.crossing_lock_projection_px,
                crossing_lock_depth_margin_m=args.crossing_lock_depth_margin_m,
                crossing_lock_frames=args.crossing_lock_frames,
                crossing_lock_window_nodes=args.crossing_lock_window_nodes,
                crossing_lock_prior_weight=args.crossing_lock_prior_weight,
                crossing_lock_min_edge_gap=args.crossing_lock_min_edge_gap,
                crossing_lock_max_pairs=args.crossing_lock_max_pairs,
                dlo_pixel_width=args.dlo_pixel_width,
                cable_diameter_m=args.cable_diameter_m,
                fixed_cable_length_m=args.fixed_cable_length_m,
                enforce_cable_length=args.enforce_cable_length,
                min_init_valid_ratio=args.init_min_valid_ratio,
                init_require_endpoints=args.init_require_endpoints,
                init_stable_frames=args.init_stable_frames,
                init_length_std_m=args.init_length_std_m,
                min_projected_dlo_width_px=args.min_projected_dlo_width_px,
                max_projected_dlo_width_px=args.max_projected_dlo_width_px,
                self_occlusion_depth_margin=args.self_occlusion_depth_margin,
                self_occlusion_max_projection_error=args.self_occlusion_max_projection_error,
            ),
        )
    smoother = NodeSmoother(args.smooth_alpha, args.reinit_threshold)

    viewer = ZedDepthGLViewer(
        args.debug_width + args.gl_width,
        args.gl_height,
        "TrackDLO in ZED Spatial Map",
        window_x=40,
        window_y=40,
        left_panel_width=args.debug_width,
    )
    viewer.init()
    viewer.view_mode = "orbit"
    viewer.point_size = args.point_size
    viewer.set_depth_max(args.depth_max)
    configure_viewer_from_zed(zed, viewer)
    camera_intrinsics = get_left_camera_intrinsics(zed)
    print(
        "Using ZED intrinsics: "
        f"fx {camera_intrinsics['fx']:.1f}, fy {camera_intrinsics['fy']:.1f}, "
        f"cx {camera_intrinsics['cx']:.1f}, cy {camera_intrinsics['cy']:.1f}"
    )

    print("Running TrackDLO + ZED spatial mapping.")
    print("Move the camera slowly. The environment is fused by ZED; the cable is TrackDLO in the same world frame.")

    request_pending = False
    last_request_time = 0.0
    last_stats_time = 0.0
    frame_count = 0
    fused_vertices = np.empty((0, 6), dtype=np.float32)
    latest_fused_points = 0
    latest_fused_vertices = 0
    latest_live_vertices = 0
    latest_live_stats = {
        "shape": None,
        "sampled": 0,
        "finite": 0,
        "in_range": 0,
        "returned": 0,
        "capped": False,
    }
    latest_request_status = sl.ERROR_CODE.SUCCESS
    latest_retrieve_status = sl.ERROR_CODE.SUCCESS
    latest_tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    latest_mapping_state = sl.SPATIAL_MAPPING_STATE.INITIALIZING

    try:
        while viewer.is_available():
            if zed.grab(runtime) <= sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                latest_tracking_state = zed.get_position(pose)
                latest_mapping_state = zed.get_spatial_mapping_state()
                pose_transform = pose.pose_data()
                camera_position = camera_position_from_pose(pose_transform)

                request_pending, last_request_time, latest_request_status, latest_retrieve_status, did_update = (
                    update_spatial_map(
                        zed,
                        fused_cloud,
                        request_pending,
                        last_request_time,
                        args.update_period,
                    )
                )
                if did_update:
                    latest_fused_points = fused_point_count(fused_cloud)
                    fused_vertices = fused_point_cloud_to_vertices(fused_cloud, args.max_map_points)
                    latest_fused_vertices = len(fused_vertices)

                display_fused_vertices = clip_vertices_by_distance(
                    fused_vertices,
                    camera_position,
                    args.depth_min,
                    args.depth_max,
                )
                should_show_fused = len(display_fused_vertices) >= args.min_fused_display_vertices

                if should_show_fused:
                    env_vertices = display_fused_vertices
                    env_label = "FUSED spatial map"
                    env_frame = "world"
                    latest_live_vertices = 0
                elif args.live_fallback:
                    env_vertices, latest_live_stats = live_point_cloud_to_vertices(
                        point_cloud,
                        args.live_stride,
                        args.live_max_points,
                        args.depth_min,
                        args.depth_max,
                        return_stats=True,
                    )
                    latest_live_vertices = len(env_vertices)
                    if len(fused_vertices) > 0:
                        env_label = (
                            "LIVE depth fallback while fused map is sparse "
                            f"({len(display_fused_vertices)}/{args.min_fused_display_vertices})"
                        )
                    else:
                        env_label = "LIVE depth fallback while fused spatial map is empty"
                    env_frame = "camera"
                else:
                    env_vertices = np.empty((0, 6), dtype=np.float32)
                    latest_live_vertices = 0
                    latest_live_stats = {
                        "shape": None,
                        "sampled": 0,
                        "finite": 0,
                        "in_range": 0,
                        "returned": 0,
                        "capped": False,
                    }
                    env_label = (
                        f"waiting for fused spatial map "
                        f"({len(display_fused_vertices)}/{args.min_fused_display_vertices})"
                    )
                    env_frame = "none"

                bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                raw_mask = create_raw_mask(bgr, hsv_ranges)
                mask, component_count, _components = clean_cable_mask(
                    raw_mask,
                    args.min_component_area,
                    args.keep_largest_component,
                    args.max_components,
                )
                centerline_mask = mask
                if component_count > 1 and not args.keep_largest_component:
                    centerline_mask, _centerline_component_count, _centerline_components = clean_cable_mask(
                        raw_mask,
                        args.min_component_area,
                        True,
                        1,
                    )
                centerline = smoother.update(extract_centerline(centerline_mask, args.nodes))
                observations = attach_3d_observations(
                    point_cloud,
                    mask,
                    centerline,
                    args.node_xyz_radius,
                    args.node_xyz_fallback_radius,
                    args.min_node_xyz_samples,
                    args.max_mask_points,
                )
                if tracker is not None:
                    observations.update(
                        tracker.step(
                            observations["X_t"],
                            observations["Y_observed_t"],
                            observations["valid_nodes"],
                            centerline.get("nodes_xy"),
                            camera_intrinsics,
                            bgr.shape[:2],
                        )
                    )

                cable_valid = observations.get("Y_t_valid", observations["valid_nodes"])
                cable_visible = observations.get("visible_node_mask", cable_valid)
                cable_extended_visible = observations.get("extended_visible_node_mask", cable_visible)
                cable_points_view = np.empty((0, 3), dtype=np.float32)
                cable_nodes_view = np.empty((0, 3), dtype=np.float32)
                if env_frame == "world" and latest_tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    cable_points_view = transform_points(pose_transform, observations["X_t"])
                    cable_nodes_view = transform_points(
                        pose_transform,
                        observations.get("Y_t", observations["Y_observed_t"]),
                        preserve_shape=True,
                    )
                elif env_frame == "camera":
                    cable_points_view = observations["X_t"]
                    cable_nodes_view = observations.get("Y_t", observations["Y_observed_t"])

                finite_cable_nodes_view = (
                    np.all(np.isfinite(cable_nodes_view), axis=1)
                    if len(cable_nodes_view) > 0
                    else np.zeros(0, dtype=bool)
                )
                visible_count = int(np.count_nonzero(cable_valid & cable_visible))
                extended_count = int(np.count_nonzero(cable_valid & cable_extended_visible & ~cable_visible))
                occluded_count = int(np.count_nonzero(cable_valid & ~cable_extended_visible))
                self_occluded_count = int(np.count_nonzero(observations.get("self_occluded_node_mask", np.zeros_like(cable_valid))))
                projection_convention = observations.get("projection_convention", "unavailable")
                tracked_nodes_xy, tracked_nodes_projected = project_tracked_nodes_to_rgb(
                    observations.get("Y_t", observations["Y_observed_t"]),
                    camera_intrinsics,
                    bgr.shape[:2],
                    observations["Y_observed_t"],
                    observations["valid_nodes"],
                    centerline.get("nodes_xy"),
                )
                if len(tracked_nodes_projected) == len(cable_valid):
                    tracked_nodes_projected = tracked_nodes_projected & cable_valid

                frame_count += 1
                tracker_status = observations.get("tracker_status", "raw observed cable")
                debug_bgr = make_rgb_debug_overlay(
                    bgr,
                    mask,
                    centerline,
                    observations,
                    component_count,
                    frame_count,
                    tracker_status,
                    tracked_nodes_xy,
                    tracked_nodes_projected,
                    cable_visible,
                    cable_extended_visible,
                )
                viewer.update_rgb_image(cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB))
                viewer.update_vertices(
                    env_vertices,
                    f"{env_label} | {tracker_status}",
                )
                viewer.update_cable(
                    cable_points_view,
                    cable_nodes_view,
                    cable_valid,
                    cable_visible,
                    cable_extended_visible,
                )

                now = time.monotonic()
                if now - last_stats_time >= 1.0:
                    print(
                        f"Frame {frame_count}: tracking {latest_tracking_state} | "
                        f"mapping {latest_mapping_state} | request {latest_request_status} | "
                        f"retrieve {latest_retrieve_status} | fused points {latest_fused_points} | "
                        f"fused vertices {latest_fused_vertices} | live vertices {latest_live_vertices} | "
                        f"live shape {latest_live_stats['shape']} sampled {latest_live_stats['sampled']} "
                        f"finite {latest_live_stats['finite']} in range {latest_live_stats['in_range']} "
                        f"capped {latest_live_stats['capped']} | "
                        f"shown {env_label} {len(env_vertices)} | "
                        f"visible pieces {component_count} tracked cable 1 | "
                        f"X_t {observations['point_count']} | init nodes {observations['valid_node_count']}/{args.nodes} | "
                        f"3D cable pts {len(cable_points_view)} nodes {int(np.count_nonzero(cable_valid & finite_cable_nodes_view))} "
                        f"vis/ext/occ {visible_count}/{extended_count}/{occluded_count} "
                        f"self-occ {self_occluded_count} proj {projection_convention} | "
                        f"{tracker_status}"
                    )
                    last_stats_time = now

            viewer.poll()

    finally:
        viewer.close()
        image.free()
        point_cloud.free()
        fused_cloud.clear()
        zed.disable_spatial_mapping()
        zed.disable_positional_tracking()
        zed.close()
