import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyzed.sl as sl

from mujoco_sphere import MujocoSphereViewer
from sphere_detection import (
    SceneSegmentation,
    SphereDetection,
    SphereDetector,
    SphereEstimate3D,
    confidence_array,
    draw_spheres_debug_overlay,
    estimate_table_plane,
    estimate_known_radius_from_depth_cap,
    reconstruct_sphere_from_zed_point_cloud,
    segment_scene_from_zed_point_cloud,
    table_candidate_point_cloud_points,
)
from sphere_particle_filter import (
    ParticleFilterConfig,
    SphereParticleFilter,
    filtered_sphere_estimate,
)

from zed_spatial import (
    DEPTH_MODES,
    RESOLUTIONS,
    configure_input_source,
    configure_viewer_from_zed,
    get_left_camera_intrinsics,
    live_point_cloud_to_vertices,
)
from zed_split_viewer import ZedDepthGLViewer


DEFAULT_SPHERE_RADIUS_M = 0.0
MUJOCO_PLACEHOLDER_RADIUS_M = 0.05


@dataclass
class BallTrack:
    track_id: int
    particle_filter: SphereParticleFilter | None
    estimate: object | None = None
    measurement: object | None = None
    result: object | None = None
    missed_frames: int = 0


def parse_args():
    pf_defaults = ParticleFilterConfig()
    parser = argparse.ArgumentParser(
        description="Particle-filter 3D reconstruction starter UI: RGB left, point cloud right.",
    )
    parser.add_argument("--input-svo-file", default="")
    parser.add_argument("--ip-address", default="")
    parser.add_argument("--resolution", choices=RESOLUTIONS, default="HD720")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--depth-mode", choices=DEPTH_MODES, default="NEURAL_PLUS")
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=2.0)
    parser.add_argument("--confidence", type=int, default=95)
    parser.add_argument("--texture-confidence", type=int, default=100)
    parser.add_argument("--depth-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-stride", type=int, default=4)
    parser.add_argument("--live-max-points", type=int, default=120000, help="0 keeps every sampled point.")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--segmentation-visualization",
        choices=("points", "off"),
        default="off",
        help="Optionally draw ball/table/other segmentation points in the 3D view.",
    )
    parser.add_argument("--rgb-width", type=int, default=620)
    parser.add_argument("--cloud-width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--sphere-profile", type=Path, default=Path("tools/sphere_color_profile.json"))
    parser.add_argument("--max-balls", type=int, default=4, help="Maximum number of detected ball blobs to track.")
    parser.add_argument(
        "--multi-ball-association-distance",
        type=float,
        default=0.30,
        help="Maximum 3D distance in meters for matching a measurement to an existing ball track.",
    )
    parser.add_argument(
        "--occlusion-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recover partial ball blobs near predicted track projections when normal blob detection fails.",
    )
    parser.add_argument("--occlusion-search-scale", type=float, default=2.4)
    parser.add_argument("--occlusion-min-area-ratio", type=float, default=0.06)
    parser.add_argument("--occlusion-min-circularity", type=float, default=0.04)
    parser.add_argument("--occlusion-max-aspect", type=float, default=6.0)
    parser.add_argument("--occlusion-detection-gate-px", type=float, default=48.0)
    parser.add_argument(
        "--occlusion-depth-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When RGB blob recovery is weak, use the projected locked-radius sphere surface in the ZED point cloud.",
    )
    parser.add_argument("--occlusion-depth-search-scale", type=float, default=1.45)
    parser.add_argument("--occlusion-surface-std", type=float, default=0.025)
    parser.add_argument("--occlusion-foreground-margin", type=float, default=0.025)
    parser.add_argument("--occlusion-min-visible-points", type=int, default=18)
    parser.add_argument("--occlusion-max-surface-points", type=int, default=600)
    parser.add_argument("--occlusion-depth-track-gate", type=float, default=0.30)
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=DEFAULT_SPHERE_RADIUS_M,
        help="Known sphere radius in meters. Default 0 estimates metric radius from stereo, then locks it.",
    )
    parser.add_argument("--sphere-max-points", type=int, default=900)
    parser.add_argument(
        "--sphere-confidence-max",
        type=float,
        default=85.0,
        help="Keep masked ZED points with confidence values at or below this threshold. Use <0 to disable.",
    )
    parser.add_argument(
        "--table-update-interval",
        type=int,
        default=30,
        help="Re-estimate the table plane every N frames. Cached table segmentation is used between updates.",
    )
    parser.add_argument(
        "--table-max-points",
        type=int,
        default=1200,
        help="Maximum background samples used when estimating/classifying the table plane.",
    )
    parser.add_argument(
        "--particle-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track the ball over time with a particle filter.",
    )
    parser.add_argument("--pf-particles", type=int, default=pf_defaults.particle_count)
    parser.add_argument("--pf-measurement-std", type=float, default=pf_defaults.measurement_position_std_m)
    parser.add_argument("--pf-radius-std", type=float, default=pf_defaults.measurement_radius_std_m)
    parser.add_argument(
        "--pf-lock-radius",
        action=argparse.BooleanOptionalAction,
        default=pf_defaults.lock_radius,
        help="Keep the particle filter ball radius fixed. Uses --sphere-radius when it is greater than 0.",
    )
    parser.add_argument(
        "--pf-surface-likelihood",
        action=argparse.BooleanOptionalAction,
        default=pf_defaults.surface_likelihood,
        help="Weight particles by fixed-radius sphere fit to the masked 3D surface points.",
    )
    parser.add_argument("--pf-surface-points", type=int, default=pf_defaults.surface_likelihood_points)
    parser.add_argument("--pf-surface-std", type=float, default=pf_defaults.surface_distance_std_m)
    parser.add_argument("--pf-surface-weight", type=float, default=pf_defaults.surface_likelihood_weight)
    parser.add_argument(
        "--pf-surface-normal-likelihood",
        action=argparse.BooleanOptionalAction,
        default=pf_defaults.surface_normal_likelihood,
        help="Weight particles by local surface patch normal agreement with the sphere model.",
    )
    parser.add_argument("--pf-surface-normal-std", type=float, default=pf_defaults.surface_normal_std)
    parser.add_argument("--pf-surface-normal-weight", type=float, default=pf_defaults.surface_normal_weight)
    parser.add_argument("--pf-radius-calibration-frames", type=int, default=pf_defaults.radius_calibration_frames)
    parser.add_argument("--pf-radius-calibration-min-frames", type=int, default=pf_defaults.radius_calibration_min_frames)
    parser.add_argument(
        "--pf-projection-likelihood",
        action=argparse.BooleanOptionalAction,
        default=pf_defaults.projection_likelihood,
        help="Weight particles by projecting the sphere into RGB and comparing with the detected circle.",
    )
    parser.add_argument("--pf-projection-center-std", type=float, default=pf_defaults.projection_center_std_px)
    parser.add_argument("--pf-projection-radius-std", type=float, default=pf_defaults.projection_radius_std_px)
    parser.add_argument("--pf-projection-weight", type=float, default=pf_defaults.projection_likelihood_weight)
    parser.add_argument(
        "--pf-adaptive-noise",
        action=argparse.BooleanOptionalAction,
        default=pf_defaults.adaptive_noise,
        help="Increase particle motion noise during high residuals or short dropouts.",
    )
    parser.add_argument("--pf-process-std", type=float, default=pf_defaults.process_position_std_m)
    parser.add_argument("--pf-reinit-distance", type=float, default=pf_defaults.reinitialize_distance_m)
    parser.add_argument("--pf-position-blend", type=float, default=pf_defaults.position_measurement_blend)
    parser.add_argument(
        "--pf-measurement-proposal",
        action=argparse.BooleanOptionalAction,
        default=pf_defaults.measurement_proposal,
        help="Inject particles around the current segmented ball measurement before weighting.",
    )
    parser.add_argument("--pf-proposal-ratio", type=float, default=pf_defaults.measurement_proposal_ratio)
    parser.add_argument("--pf-proposal-lateral-std", type=float, default=pf_defaults.proposal_lateral_std_m)
    parser.add_argument("--pf-proposal-depth-std", type=float, default=pf_defaults.proposal_depth_std_m)
    parser.add_argument("--pf-max-lost-frames", type=int, default=pf_defaults.max_prediction_frames)
    parser.add_argument("--debug-record-dir", type=Path, default=None)
    parser.add_argument("--debug-record-every", type=int, default=1)
    parser.add_argument("--debug-record-max-points", type=int, default=2000)
    parser.add_argument(
        "--mujoco",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open a separate MuJoCo viewer and update a sphere model from the 3D estimate.",
    )
    args = parser.parse_args()
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


def load_sphere_detector(profile_path):
    profile_path = Path(profile_path)
    if not profile_path.exists():
        print(f"No sphere profile found at {profile_path}. Run tools/sphere_threshold_setup.py first.")
        return None
    try:
        detector = SphereDetector.from_profile_path(profile_path)
    except Exception as exc:
        print(f"Could not load sphere profile {profile_path}: {exc}")
        return None
    print(f"Loaded sphere color profile: {profile_path}")
    print(f"  HSV ranges: {len(detector.hsv_ranges)}")
    return detector


def make_particle_filter_config(args):
    return ParticleFilterConfig(
        particle_count=args.pf_particles,
        measurement_position_std_m=args.pf_measurement_std,
        measurement_radius_std_m=args.pf_radius_std,
        lock_radius=args.pf_lock_radius,
        fixed_radius_m=args.sphere_radius if args.sphere_radius > 0.0 else 0.0,
        surface_likelihood=args.pf_surface_likelihood,
        surface_likelihood_points=args.pf_surface_points,
        surface_distance_std_m=args.pf_surface_std,
        surface_likelihood_weight=args.pf_surface_weight,
        surface_normal_likelihood=args.pf_surface_normal_likelihood,
        surface_normal_std=args.pf_surface_normal_std,
        surface_normal_weight=args.pf_surface_normal_weight,
        radius_calibration_frames=args.pf_radius_calibration_frames,
        radius_calibration_min_frames=args.pf_radius_calibration_min_frames,
        projection_likelihood=args.pf_projection_likelihood,
        projection_center_std_px=args.pf_projection_center_std,
        projection_radius_std_px=args.pf_projection_radius_std,
        projection_likelihood_weight=args.pf_projection_weight,
        adaptive_noise=args.pf_adaptive_noise,
        process_position_std_m=args.pf_process_std,
        reinitialize_distance_m=args.pf_reinit_distance,
        position_measurement_blend=args.pf_position_blend,
        measurement_proposal=args.pf_measurement_proposal,
        measurement_proposal_ratio=args.pf_proposal_ratio,
        proposal_lateral_std_m=args.pf_proposal_lateral_std,
        proposal_depth_std_m=args.pf_proposal_depth_std,
        max_prediction_frames=args.pf_max_lost_frames,
    )


def main():
    args = parse_args()
    sphere_detector = load_sphere_detector(args.sphere_profile)
    pf_config = make_particle_filter_config(args)
    tracks = []
    next_track_id = 1
    mujoco_viewer = None
    if args.mujoco:
        initial_radius = args.sphere_radius if args.sphere_radius > 0.0 else MUJOCO_PLACEHOLDER_RADIUS_M
        mujoco_viewer = MujocoSphereViewer(initial_radius, max_spheres=max(1, int(args.max_balls)))
        mujoco_viewer.start()

    zed = open_zed(args)
    runtime = make_runtime_parameters(args)
    image = sl.Mat()
    point_cloud = sl.Mat()
    confidence_map = sl.Mat()
    debug_record_dir = args.debug_record_dir
    if debug_record_dir is not None:
        debug_record_dir.mkdir(parents=True, exist_ok=True)

    viewer = ZedDepthGLViewer(
        args.rgb_width + args.cloud_width,
        args.height,
        "Particle Filter 3D Reconstruction",
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
    camera_intrinsics = None
    try:
        camera_intrinsics = get_left_camera_intrinsics(zed)
    except Exception as exc:
        print(f"Could not read ZED intrinsics for sphere estimate: {exc}")

    frame_count = 0
    last_stats_time = time.monotonic()
    last_stats_frame_count = 0
    stats_compute_seconds = 0.0
    stats_compute_frames = 0
    latest_stats = {
        "shape": None,
        "sampled": 0,
        "finite": 0,
        "in_range": 0,
        "returned": 0,
        "capped": False,
    }
    latest_sphere_status = "sphere detector unavailable" if sphere_detector is None else "waiting for sphere"
    last_filter_time = None
    cached_table_plane = None
    last_table_update_frame = -10**9
    scene_segmentation = None
    reconstruction_radius_m = args.sphere_radius if args.sphere_radius > 0.0 else 0.0

    try:
        while viewer.is_available():
            if zed.grab(runtime) <= sl.ERROR_CODE.SUCCESS:
                frame_start_time = time.monotonic()
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                confidence_measure = None
                try:
                    if zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE) <= sl.ERROR_CODE.SUCCESS:
                        confidence_measure = confidence_map
                except Exception:
                    confidence_measure = None

                bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                vertices, latest_stats = live_point_cloud_to_vertices(
                    point_cloud,
                    stride=args.live_stride,
                    max_points=args.live_max_points,
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                    return_stats=True,
                )

                sphere_estimates = []
                sphere_measurements = []
                filter_results = []
                sphere_mask = None
                sphere_detection = None
                sphere_detections = []
                measurement_estimate = None
                sphere_estimate = None
                filter_result = None
                scene_segmentation = None
                if sphere_detector is not None:
                    sphere_mask, sphere_detection, candidates = sphere_detector.detect(bgr)
                    max_balls = max(1, int(args.max_balls))
                    sphere_detections = list(candidates[:max_balls])
                    recovered_detections = recover_occluded_detections(
                        detector=sphere_detector,
                        mask=sphere_mask,
                        tracks=tracks,
                        detections=sphere_detections,
                        camera_intrinsics=camera_intrinsics,
                        max_balls=max_balls,
                        args=args,
                    )
                    if recovered_detections:
                        sphere_detections = sphere_detections + recovered_detections
                    union_mask = detection_union_mask(sphere_mask, sphere_detections)
                    table_interval = max(1, int(args.table_update_interval))
                    update_table = (
                        cached_table_plane is None
                        or (frame_count - last_table_update_frame) >= table_interval
                    )
                    max_confidence = args.sphere_confidence_max if args.sphere_confidence_max >= 0.0 else None
                    need_full_segmentation = (
                        args.segmentation_visualization == "points"
                        or should_write_debug_record(debug_record_dir, frame_count + 1, args.debug_record_every)
                    )
                    if need_full_segmentation:
                        table_segmentation = segment_scene_from_zed_point_cloud(
                            point_cloud,
                            union_mask,
                            radius_m=reconstruction_radius_m,
                            depth_min=args.depth_min,
                            depth_max=args.depth_max,
                            max_ball_points=max(args.sphere_max_points, args.sphere_max_points * max(1, len(sphere_detections))),
                            max_table_points=args.table_max_points,
                            confidence_map=confidence_measure,
                            max_confidence=max_confidence,
                            cached_table_plane=cached_table_plane,
                            update_table=update_table,
                        )
                    else:
                        table_segmentation = update_cached_table_plane_fast(
                            point_cloud=point_cloud,
                            exclude_mask=union_mask,
                            radius_m=reconstruction_radius_m,
                            depth_min=args.depth_min,
                            depth_max=args.depth_max,
                            max_points=args.table_max_points,
                            confidence_map=confidence_measure,
                            max_confidence=max_confidence,
                            cached_table_plane=cached_table_plane,
                            update_table=update_table,
                        )
                    if table_segmentation.table_plane is not None:
                        cached_table_plane = table_segmentation.table_plane
                        if table_segmentation.table_updated:
                            last_table_update_frame = frame_count

                    sphere_measurements, ball_segmentations = reconstruct_ball_measurements(
                        point_cloud=point_cloud,
                        sphere_mask=sphere_mask,
                        detections=sphere_detections,
                        detector=sphere_detector,
                        radius_m=reconstruction_radius_m,
                        depth_min=args.depth_min,
                        depth_max=args.depth_max,
                        max_points=args.sphere_max_points,
                        camera_intrinsics=camera_intrinsics,
                        confidence_map=confidence_measure,
                        max_confidence=max_confidence,
                        cached_table_plane=cached_table_plane,
                        table_max_points=args.table_max_points,
                    )
                    scene_segmentation = aggregate_scene_segmentation(table_segmentation, ball_segmentations)
                    now = time.monotonic()
                    if last_filter_time is None:
                        filter_dt = 1.0 / max(float(args.fps), 1.0)
                    else:
                        filter_dt = now - last_filter_time
                    last_filter_time = now

                    if args.particle_filter:
                        tracks, next_track_id = update_ball_tracks(
                            tracks,
                            sphere_measurements,
                            filter_dt,
                            camera_intrinsics,
                            pf_config,
                            next_track_id,
                            max_tracks=max_balls,
                            association_distance=args.multi_ball_association_distance,
                            point_cloud=point_cloud,
                            confidence_map=confidence_measure,
                            max_confidence=max_confidence,
                            args=args,
                        )
                        sphere_estimates = [track.estimate for track in tracks if track.estimate is not None]
                        filter_results = [track.result for track in tracks if track.result is not None]
                    else:
                        tracks = [
                            BallTrack(index + 1, None, estimate=estimate, measurement=estimate)
                            for index, estimate in enumerate(sphere_measurements[:max_balls])
                        ]
                        sphere_estimates = [track.estimate for track in tracks if track.estimate is not None]

                    reconstruction_radius_m = maybe_update_reconstruction_radius(
                        reconstruction_radius_m,
                        args,
                        sphere_measurements,
                        filter_results,
                    )
                    measurement_estimate = sphere_measurements[0] if sphere_measurements else None
                    sphere_estimate = sphere_estimates[0] if sphere_estimates else None
                    filter_result = filter_results[0] if filter_results else None

                    if sphere_estimates:
                        latest_sphere_status = multi_sphere_status(
                            sphere_estimates,
                            sphere_detections,
                            scene_segmentation,
                        )
                        viewer.update_spheres(make_viewer_spheres(tracks))
                        if mujoco_viewer is not None and mujoco_viewer.available:
                            mujoco_viewer.update_spheres(make_viewer_spheres(tracks))
                    elif sphere_detections:
                        latest_sphere_status = (
                            f"{len(sphere_detections)} sphere RGB detections, waiting for valid 3D points | "
                            f"{scene_segmentation_status(scene_segmentation)}"
                        )
                        viewer.update_spheres([])
                        if mujoco_viewer is not None and mujoco_viewer.available:
                            mujoco_viewer.update_spheres([])
                    else:
                        latest_sphere_status = "spheres not detected"
                        viewer.update_spheres([])
                        if mujoco_viewer is not None and mujoco_viewer.available:
                            mujoco_viewer.update_spheres([])

                debug_bgr = draw_spheres_debug_overlay(
                    bgr,
                    sphere_mask,
                    sphere_detections,
                    sphere_estimates,
                )
                if should_write_debug_record(debug_record_dir, frame_count + 1, args.debug_record_every):
                    write_debug_record(
                        debug_record_dir,
                        frame_count + 1,
                        debug_bgr,
                        sphere_mask,
                        measurement_estimate,
                        sphere_estimate,
                        filter_result,
                        scene_segmentation,
                        max_points=args.debug_record_max_points,
                    )
                frame_count += 1
                viewer.update_rgb_image(cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB))
                if scene_segmentation is not None and args.segmentation_visualization == "points":
                    viewer.update_segmentation(
                        scene_segmentation.ball_points,
                        scene_segmentation.table_points,
                        scene_segmentation.other_points,
                    )
                else:
                    viewer.update_segmentation(None, None, None)
                viewer.update_vertices(
                    vertices,
                    f"live ZED point cloud | frame {frame_count} | {len(vertices)} points | {latest_sphere_status}",
                )

                now = time.monotonic()
                stats_compute_seconds += max(0.0, now - frame_start_time)
                stats_compute_frames += 1
                if now - last_stats_time >= 1.0:
                    elapsed = max(now - last_stats_time, 1e-6)
                    render_fps = (frame_count - last_stats_frame_count) / elapsed
                    compute_ms = 1000.0 * stats_compute_seconds / max(stats_compute_frames, 1)
                    print(
                        f"Frame {frame_count}: point cloud shape {latest_stats['shape']} | "
                        f"render {render_fps:.1f} fps compute {compute_ms:.1f} ms | "
                        f"sampled {latest_stats['sampled']} finite {latest_stats['finite']} "
                        f"in range {latest_stats['in_range']} returned {latest_stats['returned']} "
                        f"capped {latest_stats['capped']} | {latest_sphere_status}"
                    )
                    last_stats_time = now
                    last_stats_frame_count = frame_count
                    stats_compute_seconds = 0.0
                    stats_compute_frames = 0

            viewer.poll()

    finally:
        viewer.close()
        if mujoco_viewer is not None:
            mujoco_viewer.close()
        image.free()
        point_cloud.free()
        confidence_map.free()
        zed.close()


def detection_union_mask(mask, detections):
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    union = np.zeros_like(mask, dtype=np.uint8)
    for detection in detections:
        if detection is not None:
            cv2.drawContours(union, [detection.contour], -1, 255, -1)
    return union


def update_cached_table_plane_fast(
    point_cloud,
    exclude_mask,
    radius_m,
    depth_min,
    depth_max,
    max_points,
    confidence_map=None,
    max_confidence=None,
    cached_table_plane=None,
    update_table=False,
):
    if not bool(update_table) and cached_table_plane is not None:
        return empty_scene_segmentation(
            table_plane=cached_table_plane,
            table_updated=False,
            method="scene segmentation: table cached fast",
        )

    candidate_points = table_candidate_point_cloud_points(
        point_cloud,
        exclude_mask,
        depth_min=depth_min,
        depth_max=depth_max,
        max_points=max_points,
        confidence_map=confidence_map,
        max_confidence=max_confidence,
    )
    table_plane = estimate_table_plane(
        candidate_points,
        sphere_points=None,
        radius_m=radius_m,
        max_iterations=48,
    )
    if table_plane is None:
        return empty_scene_segmentation(
            table_plane=cached_table_plane,
            table_updated=False,
            method="scene segmentation: table update skipped",
        )
    return empty_scene_segmentation(
        table_plane=table_plane,
        table_updated=True,
        method="scene segmentation: table updated fast",
    )


def empty_scene_segmentation(table_plane=None, table_updated=False, method="scene segmentation: fast"):
    empty = np.empty((0, 3), dtype=np.float32)
    return SceneSegmentation(
        ball_points=empty,
        raw_ball_points=empty,
        table_points=empty,
        other_points=empty,
        table_plane=table_plane,
        table_updated=bool(table_updated),
        method=method,
    )


def recover_occluded_detections(detector, mask, tracks, detections, camera_intrinsics, max_balls, args):
    if not bool(args.occlusion_recovery):
        return []
    if detector is None or mask is None or camera_intrinsics is None:
        return []
    recovered = []
    existing = list(detections)
    for track in tracks:
        if len(existing) + len(recovered) >= int(max_balls):
            break
        projection = project_track_to_image(track, camera_intrinsics)
        if projection is None:
            continue
        center_xy, radius_px = projection
        if projected_track_has_detection(center_xy, radius_px, existing + recovered, args.occlusion_detection_gate_px):
            continue
        detection = detector.detect_partial_blob(
            mask,
            center_xy,
            radius_px,
            search_scale=args.occlusion_search_scale,
            min_area_ratio=args.occlusion_min_area_ratio,
            min_circularity=args.occlusion_min_circularity,
            max_aspect_ratio=args.occlusion_max_aspect,
        )
        if detection is not None:
            recovered.append(detection)
    return recovered


def project_track_to_image(track, camera_intrinsics):
    estimate = getattr(track, "estimate", None)
    if estimate is None:
        return None
    return project_sphere_to_image(estimate.center_xyz, estimate.radius_m, camera_intrinsics)


def project_sphere_to_image(center_xyz, radius_m, camera_intrinsics):
    try:
        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])
    except Exception:
        return None
    center = np.asarray(center_xyz, dtype=np.float64).reshape(-1)
    if len(center) < 3 or not np.all(np.isfinite(center[:3])):
        return None
    z = float(center[2])
    radius_m = float(radius_m)
    if not (np.isfinite(z) and z > 1e-5 and np.isfinite(radius_m) and radius_m > 0.0):
        return None
    if not all(np.isfinite(value) and abs(value) > 1e-6 for value in (fx, fy)):
        return None
    u = cx + fx * center[0] / z
    v = cy - fy * center[1] / z
    radius_px = 0.5 * (abs(fx) + abs(fy)) * radius_m / z
    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(radius_px) and radius_px > 1.0):
        return None
    return (float(u), float(v)), float(radius_px)


def projected_track_has_detection(center_xy, radius_px, detections, gate_px):
    cx, cy = center_xy
    gate = max(float(gate_px), 0.75 * float(radius_px))
    for detection in detections:
        dx = float(detection.center_xy[0]) - cx
        dy = float(detection.center_xy[1]) - cy
        if np.hypot(dx, dy) <= gate:
            return True
    return False


def recover_occluded_track_measurement(
    track,
    point_cloud,
    camera_intrinsics,
    args,
    confidence_map=None,
    max_confidence=None,
):
    if args is None or not bool(getattr(args, "occlusion_recovery", False)):
        return None
    if not bool(getattr(args, "occlusion_depth_recovery", True)):
        return None
    if point_cloud is None or camera_intrinsics is None:
        return None

    previous = getattr(track, "estimate", None)
    if previous is None:
        return None

    center = np.asarray(previous.center_xyz, dtype=np.float64).reshape(-1)
    if len(center) < 3 or not np.all(np.isfinite(center[:3])):
        return None
    center = center[:3]

    radius = float(getattr(previous, "radius_m", 0.0))
    if not (np.isfinite(radius) and radius > 0.0):
        return None

    projection = project_sphere_to_image(center, radius, camera_intrinsics)
    if projection is None:
        return None
    image_center_xy, image_radius_px = projection

    try:
        point_data = np.asarray(point_cloud.get_data())
    except Exception:
        return None
    if point_data.ndim != 3 or point_data.shape[2] < 3:
        return None

    candidate_points = projected_sphere_candidate_points(
        point_data,
        image_center_xy,
        image_radius_px,
        args,
        confidence_map=confidence_map,
        max_confidence=max_confidence,
    )
    min_points = max(3, int(getattr(args, "occlusion_min_visible_points", 18)))
    if len(candidate_points) < min_points:
        return None

    candidate_points = reject_foreground_occluders(candidate_points, center, radius, args)
    candidate_points = reject_track_table_points(candidate_points, previous, radius)
    if len(candidate_points) < min_points:
        return None

    surface_gate = occlusion_surface_gate(radius, args)
    predicted_residuals = np.abs(np.linalg.norm(candidate_points - center[None, :], axis=1) - radius)
    surface_points = candidate_points[predicted_residuals <= surface_gate]
    if len(surface_points) < min_points:
        return None

    surface_points = cap_occlusion_surface_points(
        surface_points,
        max_points=getattr(args, "occlusion_max_surface_points", 600),
    )
    detection = SphereDetection(
        center_xy=(float(image_center_xy[0]), float(image_center_xy[1])),
        radius_px=float(image_radius_px),
        area=float(np.pi * image_radius_px * image_radius_px),
        circularity=1.0,
        fill_ratio=1.0,
        aspect=1.0,
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )

    cap_estimate = estimate_known_radius_from_depth_cap(
        surface_points,
        radius,
        detection=detection,
        camera_intrinsics=camera_intrinsics,
    )
    if cap_estimate is None:
        recovered_center = center.astype(np.float32)
    else:
        recovered_center = np.asarray(cap_estimate.center_xyz, dtype=np.float32).reshape(3)

    if not occlusion_center_is_consistent(recovered_center, center, radius, image_center_xy, image_radius_px, camera_intrinsics, args):
        return None

    refined_residuals = np.abs(np.linalg.norm(candidate_points - recovered_center[None, :], axis=1) - radius)
    refined_points = candidate_points[refined_residuals <= surface_gate]
    if len(refined_points) >= min_points:
        surface_points = cap_occlusion_surface_points(
            refined_points,
            max_points=getattr(args, "occlusion_max_surface_points", 600),
        )
        refined_residuals = np.abs(np.linalg.norm(surface_points - recovered_center[None, :], axis=1) - radius)
    else:
        refined_residuals = np.abs(np.linalg.norm(surface_points - recovered_center[None, :], axis=1) - radius)

    residual_m = float(np.median(refined_residuals)) if len(refined_residuals) else np.inf
    residual_gate = max(float(getattr(args, "occlusion_surface_std", 0.025)) * 2.5, 0.012, 0.04 * radius)
    if not np.isfinite(residual_m) or residual_m > residual_gate:
        return None

    return SphereEstimate3D(
        center_xyz=np.asarray(recovered_center, dtype=np.float32),
        radius_m=float(radius),
        surface_points=np.ascontiguousarray(surface_points[:, :3], dtype=np.float32),
        residual_m=residual_m,
        method="occlusion-aware projected sphere surface",
        table_normal=getattr(previous, "table_normal", None),
        table_offset=getattr(previous, "table_offset", None),
        image_center_xy=(float(image_center_xy[0]), float(image_center_xy[1])),
        image_radius_px=float(image_radius_px),
    )


def projected_sphere_candidate_points(
    point_data,
    image_center_xy,
    image_radius_px,
    args,
    confidence_map=None,
    max_confidence=None,
):
    height, width = point_data.shape[:2]
    cx, cy = image_center_xy
    search_scale = max(1.0, float(getattr(args, "occlusion_depth_search_scale", 1.45)))
    search_radius = max(8.0, float(image_radius_px) * search_scale)

    x0 = int(max(0, np.floor(float(cx) - search_radius)))
    x1 = int(min(width, np.ceil(float(cx) + search_radius + 1.0)))
    y0 = int(max(0, np.floor(float(cy) - search_radius)))
    y1 = int(min(height, np.ceil(float(cy) + search_radius + 1.0)))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 3), dtype=np.float32)

    stride = projected_sphere_sample_stride(search_radius, args)
    rows = np.arange(y0, y1, stride, dtype=np.int32)
    cols = np.arange(x0, x1, stride, dtype=np.int32)
    if len(rows) == 0 or len(cols) == 0:
        return np.empty((0, 3), dtype=np.float32)

    xyz_grid = point_data[y0:y1:stride, x0:x1:stride, :3].astype(np.float32)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    selected = (xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2
    selected = selected <= float(search_radius * search_radius)
    selected &= np.all(np.isfinite(xyz_grid), axis=2)

    distances = np.linalg.norm(xyz_grid, axis=2)
    depth_min = getattr(args, "depth_min", None)
    depth_max = getattr(args, "depth_max", None)
    if depth_min is not None:
        selected &= distances >= float(depth_min)
    if depth_max is not None:
        selected &= distances <= float(depth_max)

    if confidence_map is not None and max_confidence is not None:
        confidence = confidence_array(confidence_map, (height, width))
        if confidence is not None:
            selected &= confidence[y0:y1:stride, x0:x1:stride] <= float(max_confidence)

    points = xyz_grid[selected]
    return np.ascontiguousarray(points[:, :3], dtype=np.float32)


def projected_sphere_sample_stride(search_radius_px, args):
    max_points = max(1, int(getattr(args, "occlusion_max_surface_points", 600)))
    target_samples = max(4 * max_points, 8 * max(1, int(getattr(args, "occlusion_min_visible_points", 18))))
    disk_area = np.pi * float(search_radius_px) * float(search_radius_px)
    return max(1, int(np.floor(np.sqrt(max(disk_area / float(target_samples), 1.0)))))


def reject_foreground_occluders(points, center, radius, args):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    center_distance = float(np.linalg.norm(np.asarray(center, dtype=np.float64).reshape(3)))
    if not np.isfinite(center_distance) or center_distance <= 1e-6:
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)

    margin = max(0.0, float(getattr(args, "occlusion_foreground_margin", 0.025)))
    front_distance = max(center_distance - float(radius), 0.0)
    point_distances = np.linalg.norm(points[:, :3], axis=1)
    keep = np.isfinite(point_distances) & (point_distances >= front_distance - margin)
    return np.ascontiguousarray(points[keep, :3], dtype=np.float32)


def reject_track_table_points(points, estimate, radius):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    normal = getattr(estimate, "table_normal", None)
    offset = getattr(estimate, "table_offset", None)
    if normal is None or offset is None:
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)

    normal = np.asarray(normal, dtype=np.float64).reshape(-1)
    if len(normal) < 3:
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)
    normal = normal[:3]
    normal_norm = np.linalg.norm(normal)
    if not np.isfinite(normal_norm) or normal_norm < 1e-8:
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)
    normal = normal / normal_norm

    heights = points[:, :3].astype(np.float64) @ normal + float(offset)
    min_height = float(np.clip(0.12 * float(radius), 0.006, 0.045))
    keep = np.isfinite(heights) & (heights > min_height)
    if np.count_nonzero(keep) < 6:
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)
    return np.ascontiguousarray(points[keep, :3], dtype=np.float32)


def occlusion_surface_gate(radius, args):
    configured = max(0.0, float(getattr(args, "occlusion_surface_std", 0.025)))
    radius_scaled = float(np.clip(0.10 * float(radius), 0.010, 0.080))
    return max(2.5 * configured, radius_scaled)


def cap_occlusion_surface_points(points, max_points=600):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    max_points = max(0, int(max_points))
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return np.ascontiguousarray(points[:, :3], dtype=np.float32)


def occlusion_center_is_consistent(
    recovered_center,
    predicted_center,
    radius,
    predicted_image_center_xy,
    predicted_image_radius_px,
    camera_intrinsics,
    args,
):
    recovered_center = np.asarray(recovered_center, dtype=np.float64).reshape(-1)
    predicted_center = np.asarray(predicted_center, dtype=np.float64).reshape(-1)
    if len(recovered_center) < 3 or len(predicted_center) < 3:
        return False
    if not (np.all(np.isfinite(recovered_center[:3])) and np.all(np.isfinite(predicted_center[:3]))):
        return False

    displacement = float(np.linalg.norm(recovered_center[:3] - predicted_center[:3]))
    track_gate = max(float(getattr(args, "occlusion_depth_track_gate", 0.30)), 0.50 * float(radius), 0.04)
    if displacement > track_gate:
        return False

    reprojection = project_sphere_to_image(recovered_center[:3], radius, camera_intrinsics)
    if reprojection is None:
        return False
    center_xy, _radius_px = reprojection
    image_error = float(np.hypot(
        float(center_xy[0]) - float(predicted_image_center_xy[0]),
        float(center_xy[1]) - float(predicted_image_center_xy[1]),
    ))
    image_gate = max(
        float(getattr(args, "occlusion_depth_search_scale", 1.45)) * float(predicted_image_radius_px),
        float(getattr(args, "occlusion_detection_gate_px", 48.0)),
    )
    return image_error <= image_gate


def reconstruct_ball_measurements(
    point_cloud,
    sphere_mask,
    detections,
    detector,
    radius_m,
    depth_min,
    depth_max,
    max_points,
    camera_intrinsics,
    confidence_map,
    max_confidence,
    cached_table_plane,
    table_max_points,
):
    measurements = []
    segmentations = []
    for detection in detections:
        selected_mask = detector.detection_mask(sphere_mask, detection)
        segmentation = segment_scene_from_zed_point_cloud(
            point_cloud,
            selected_mask,
            radius_m=radius_m,
            depth_min=depth_min,
            depth_max=depth_max,
            max_ball_points=max_points,
            max_table_points=table_max_points,
            confidence_map=confidence_map,
            max_confidence=max_confidence,
            cached_table_plane=cached_table_plane,
            update_table=False,
        )
        segmentations.append(segmentation)
        estimate = reconstruct_sphere_from_zed_point_cloud(
            point_cloud,
            selected_mask,
            detection,
            radius_m=radius_m,
            depth_min=depth_min,
            depth_max=depth_max,
            max_points=max_points,
            camera_intrinsics=camera_intrinsics,
            confidence_map=confidence_map,
            max_confidence=max_confidence,
            scene_segmentation=segmentation,
            cached_table_plane=cached_table_plane,
            update_table=False,
            table_max_points=table_max_points,
        )
        if estimate is not None:
            measurements.append(estimate)
    return measurements, segmentations


def update_ball_tracks(
    tracks,
    measurements,
    dt,
    camera_intrinsics,
    pf_config,
    next_track_id,
    max_tracks=4,
    association_distance=0.30,
    point_cloud=None,
    confidence_map=None,
    max_confidence=None,
    args=None,
):
    matches, unmatched_tracks, unmatched_measurements = associate_measurements_to_tracks(
        tracks,
        measurements,
        max_distance=association_distance,
    )

    updated_tracks = []
    for track_index, measurement_index in matches:
        track = tracks[track_index]
        measurement = measurements[measurement_index]
        result = track.particle_filter.step(measurement, dt, camera_intrinsics=camera_intrinsics)
        estimate = filtered_sphere_estimate(measurement, result)
        if estimate is not None:
            track.measurement = measurement
            track.result = result
            track.estimate = estimate
            track.missed_frames = 0
            updated_tracks.append(track)

    for track_index in unmatched_tracks:
        track = tracks[track_index]
        if track.particle_filter is None:
            continue
        previous_estimate = track.estimate
        recovered_measurement = recover_occluded_track_measurement(
            track,
            point_cloud,
            camera_intrinsics,
            args,
            confidence_map=confidence_map,
            max_confidence=max_confidence,
        )
        result = track.particle_filter.step(recovered_measurement, dt, camera_intrinsics=camera_intrinsics)
        estimate = filtered_sphere_estimate(recovered_measurement, result)
        if estimate is not None:
            if recovered_measurement is None and previous_estimate is not None:
                estimate.table_normal = getattr(previous_estimate, "table_normal", None)
                estimate.table_offset = getattr(previous_estimate, "table_offset", None)
            track.measurement = recovered_measurement
            track.result = result
            track.estimate = estimate
            track.missed_frames = 0 if recovered_measurement is not None else track.missed_frames + 1
            updated_tracks.append(track)

    max_tracks = max(1, int(max_tracks))
    for measurement_index in unmatched_measurements:
        if len(updated_tracks) >= max_tracks:
            break
        measurement = measurements[measurement_index]
        particle_filter = SphereParticleFilter(pf_config, seed=1000 + int(next_track_id))
        result = particle_filter.step(measurement, dt, camera_intrinsics=camera_intrinsics)
        estimate = filtered_sphere_estimate(measurement, result)
        if estimate is None:
            continue
        updated_tracks.append(
            BallTrack(
                track_id=int(next_track_id),
                particle_filter=particle_filter,
                estimate=estimate,
                measurement=measurement,
                result=result,
                missed_frames=0,
            )
        )
        next_track_id += 1

    updated_tracks.sort(key=lambda item: item.track_id)
    return updated_tracks[:max_tracks], next_track_id


def associate_measurements_to_tracks(tracks, measurements, max_distance=0.30):
    if not tracks:
        return [], [], list(range(len(measurements)))
    if not measurements:
        return [], list(range(len(tracks))), []

    pairs = []
    for track_index, track in enumerate(tracks):
        track_center = track_reference_center(track)
        if track_center is None:
            continue
        for measurement_index, measurement in enumerate(measurements):
            center = np.asarray(measurement.center_xyz, dtype=np.float64).reshape(3)
            distance = float(np.linalg.norm(center - track_center))
            if np.isfinite(distance):
                pairs.append((distance, track_index, measurement_index))

    pairs.sort(key=lambda item: item[0])
    matched_tracks = set()
    matched_measurements = set()
    matches = []
    threshold = max(float(max_distance), 1e-6)
    for distance, track_index, measurement_index in pairs:
        if distance > threshold:
            break
        if track_index in matched_tracks or measurement_index in matched_measurements:
            continue
        matched_tracks.add(track_index)
        matched_measurements.add(measurement_index)
        matches.append((track_index, measurement_index))

    unmatched_tracks = [index for index in range(len(tracks)) if index not in matched_tracks]
    unmatched_measurements = [index for index in range(len(measurements)) if index not in matched_measurements]
    return matches, unmatched_tracks, unmatched_measurements


def track_reference_center(track):
    estimate = getattr(track, "estimate", None)
    if estimate is None:
        return None
    center = np.asarray(estimate.center_xyz, dtype=np.float64).reshape(-1)
    if len(center) < 3 or not np.all(np.isfinite(center[:3])):
        return None
    return center[:3]


def maybe_update_reconstruction_radius(current_radius, args, measurements, filter_results):
    if args.sphere_radius > 0.0:
        return float(args.sphere_radius)
    if current_radius > 0.0:
        return float(current_radius)
    for result in filter_results:
        if result is not None and result.radius_calibrated and result.radius_m > 0.0:
            return float(result.radius_m)
    for measurement in measurements:
        if measurement is not None and measurement.radius_m > 0.0:
            return float(measurement.radius_m)
    return float(current_radius)


def make_viewer_spheres(tracks):
    spheres = []
    for track in tracks:
        estimate = track.estimate
        if estimate is None:
            continue
        spheres.append(
            {
                "track_id": track.track_id,
                "center": estimate.center_xyz,
                "radius": estimate.radius_m,
                "points": estimate.surface_points,
            }
        )
    return spheres


def aggregate_scene_segmentation(table_segmentation, ball_segmentations):
    if table_segmentation is None and not ball_segmentations:
        return None
    ball_points = stack_point_sets([getattr(item, "ball_points", None) for item in ball_segmentations])
    raw_ball_points = stack_point_sets([getattr(item, "raw_ball_points", None) for item in ball_segmentations])
    if table_segmentation is None:
        table_points = np.empty((0, 3), dtype=np.float32)
        other_points = np.empty((0, 3), dtype=np.float32)
        table_plane = None
        table_updated = False
        method = "scene segmentation: multi-ball"
    else:
        table_points = np.asarray(table_segmentation.table_points, dtype=np.float32)
        other_points = np.asarray(table_segmentation.other_points, dtype=np.float32)
        table_plane = table_segmentation.table_plane
        table_updated = table_segmentation.table_updated
        method = f"{table_segmentation.method} | multi-ball {len(ball_segmentations)}"
    return SceneSegmentation(
        ball_points=ball_points,
        raw_ball_points=raw_ball_points,
        table_points=np.ascontiguousarray(table_points[:, :3], dtype=np.float32) if table_points.ndim == 2 and table_points.shape[1] >= 3 else np.empty((0, 3), dtype=np.float32),
        other_points=np.ascontiguousarray(other_points[:, :3], dtype=np.float32) if other_points.ndim == 2 and other_points.shape[1] >= 3 else np.empty((0, 3), dtype=np.float32),
        table_plane=table_plane,
        table_updated=bool(table_updated),
        method=method,
    )


def stack_point_sets(point_sets):
    arrays = []
    for points in point_sets:
        if points is None:
            continue
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] >= 3 and len(points) > 0:
            arrays.append(points[:, :3])
    if not arrays:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(arrays), dtype=np.float32)


def multi_sphere_status(estimates, detections, segmentation):
    primary = estimates[0]
    center = primary.center_xyz
    return (
        f"balls {len(estimates)}/{len(detections)} "
        f"primary ({center[0]:+.3f},{center[1]:+.3f},{center[2]:+.3f})m "
        f"r {primary.radius_m:.3f}m pts {len(primary.surface_points)} "
        f"{scene_segmentation_status(segmentation)}"
    )


def should_write_debug_record(record_dir, frame_index, every):
    if record_dir is None:
        return False
    every = max(1, int(every))
    return int(frame_index) % every == 0


def write_debug_record(
    record_dir,
    frame_index,
    debug_bgr,
    sphere_mask,
    measurement_estimate,
    sphere_estimate,
    filter_result,
    scene_segmentation,
    max_points=2000,
):
    frame_dir = Path(record_dir) / f"frame_{int(frame_index):06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(frame_dir / "debug_rgb.png"), debug_bgr)
    if sphere_mask is not None:
        cv2.imwrite(str(frame_dir / "mask.png"), sphere_mask)

    arrays = {}
    if measurement_estimate is not None:
        add_surface_debug_arrays(arrays, "measurement", measurement_estimate, max_points=max_points)
    if sphere_estimate is not None:
        add_surface_debug_arrays(arrays, "estimate", sphere_estimate, max_points=max_points)
    if scene_segmentation is not None:
        add_segmentation_debug_arrays(arrays, scene_segmentation, max_points=max_points)
    if arrays:
        np.savez_compressed(frame_dir / "points.npz", **arrays)

    meta = {
        "frame": int(frame_index),
        "measurement": estimate_to_metadata(measurement_estimate),
        "estimate": estimate_to_metadata(sphere_estimate),
        "filter": filter_result_to_metadata(filter_result),
        "segmentation": scene_segmentation_to_metadata(scene_segmentation),
    }
    with open(frame_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def add_surface_debug_arrays(arrays, prefix, estimate, max_points=2000):
    points = np.asarray(getattr(estimate, "surface_points", np.empty((0, 3))), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return

    indices = sample_row_indices(len(points), max_points=max_points)
    arrays[f"{prefix}_surface_points"] = np.ascontiguousarray(points[indices, :3], dtype=np.float32)

    normals = getattr(estimate, "surface_normals", None)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.ndim == 2 and normals.shape[1] >= 3 and len(normals) == len(points):
            arrays[f"{prefix}_surface_normals"] = np.ascontiguousarray(normals[indices, :3], dtype=np.float32)

    weights = getattr(estimate, "surface_weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(weights) == len(points):
            arrays[f"{prefix}_surface_weights"] = np.ascontiguousarray(weights[indices], dtype=np.float32)


def add_segmentation_debug_arrays(arrays, segmentation, max_points=2000):
    arrays["segmentation_ball_points"] = sample_points(
        getattr(segmentation, "ball_points", np.empty((0, 3))),
        max_points=max_points,
    )
    arrays["segmentation_raw_ball_points"] = sample_points(
        getattr(segmentation, "raw_ball_points", np.empty((0, 3))),
        max_points=max_points,
    )
    arrays["segmentation_table_points"] = sample_points(
        getattr(segmentation, "table_points", np.empty((0, 3))),
        max_points=max_points,
    )
    arrays["segmentation_other_points"] = sample_points(
        getattr(segmentation, "other_points", np.empty((0, 3))),
        max_points=max_points,
    )


def scene_segmentation_status(segmentation):
    if segmentation is None:
        return "seg none"
    table_mode = "table update" if getattr(segmentation, "table_updated", False) else "table cached"
    if getattr(segmentation, "table_plane", None) is None:
        table_mode = "table unknown"
    return (
        f"seg ball {getattr(segmentation, 'ball_count', 0)}/"
        f"{getattr(segmentation, 'raw_ball_count', 0)} "
        f"{table_mode} tbl {getattr(segmentation, 'table_count', 0)} "
        f"other {getattr(segmentation, 'other_count', 0)}"
    )


def sample_row_indices(row_count, max_points=2000):
    row_count = max(0, int(row_count))
    if row_count == 0:
        return np.empty((0,), dtype=np.int64)
    max_points = max(0, int(max_points))
    if max_points > 0 and row_count > max_points:
        return np.linspace(0, row_count - 1, max_points, dtype=np.int64)
    return np.arange(row_count, dtype=np.int64)


def sample_points(points, max_points=2000):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    points = points[sample_row_indices(len(points), max_points=max_points)]
    return np.ascontiguousarray(points[:, :3], dtype=np.float32)


def estimate_to_metadata(estimate):
    if estimate is None:
        return None
    center = np.asarray(estimate.center_xyz, dtype=np.float64).reshape(3)
    return {
        "center_xyz": [float(value) for value in center],
        "radius_m": float(estimate.radius_m),
        "residual_m": float(estimate.residual_m),
        "surface_point_count": int(len(estimate.surface_points)),
        "surface_normal_count": surface_normal_count(estimate),
        "method": str(estimate.method),
        "image_center_xy": (
            None
            if getattr(estimate, "image_center_xy", None) is None
            else [float(value) for value in estimate.image_center_xy]
        ),
        "image_radius_px": (
            None
            if getattr(estimate, "image_radius_px", None) is None
            else float(estimate.image_radius_px)
        ),
    }


def scene_segmentation_to_metadata(segmentation):
    if segmentation is None:
        return None

    table_plane = getattr(segmentation, "table_plane", None)
    if table_plane is None:
        table_meta = None
    else:
        table_meta = {
            "normal": [float(value) for value in np.asarray(table_plane.normal).reshape(3)],
            "offset": float(table_plane.offset),
            "inlier_count": int(table_plane.inlier_count),
            "residual_m": float(table_plane.residual_m),
        }

    return {
        "method": str(getattr(segmentation, "method", "")),
        "ball_count": int(getattr(segmentation, "ball_count", 0)),
        "raw_ball_count": int(getattr(segmentation, "raw_ball_count", 0)),
        "table_count": int(getattr(segmentation, "table_count", 0)),
        "other_count": int(getattr(segmentation, "other_count", 0)),
        "table_updated": bool(getattr(segmentation, "table_updated", False)),
        "table_plane": table_meta,
    }


def surface_normal_count(estimate):
    normals = getattr(estimate, "surface_normals", None)
    if normals is None:
        return 0
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 2 or normals.shape[1] < 3:
        return 0
    lengths = np.linalg.norm(normals[:, :3], axis=1)
    return int(np.count_nonzero(np.isfinite(lengths) & (lengths > 0.5)))


def filter_result_to_metadata(result):
    if result is None:
        return None
    return {
        "velocity_xyz": [float(value) for value in np.asarray(result.velocity_xyz).reshape(3)],
        "effective_sample_size": float(result.effective_sample_size),
        "measurement_used": bool(result.measurement_used),
        "prediction_only": bool(result.prediction_only),
        "lost_frames": int(result.lost_frames),
        "model_likelihood_used": bool(result.model_likelihood_used),
        "projection_likelihood_used": bool(result.projection_likelihood_used),
        "measurement_proposal_used": bool(getattr(result, "measurement_proposal_used", False)),
        "radius_calibrated": bool(result.radius_calibrated),
        "radius_sample_count": int(result.radius_sample_count),
        "motion_noise_scale": float(result.motion_noise_scale),
    }


if __name__ == "__main__":
    main()
