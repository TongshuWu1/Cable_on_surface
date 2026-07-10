from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CableDetection2D:
    mask: np.ndarray
    skeleton: np.ndarray
    centerline_xy: np.ndarray
    component_count: int
    branch_count: int
    centerline_paths_xy: tuple[np.ndarray, ...] = ()


@dataclass
class CableEstimate3D:
    points_xyz: np.ndarray
    source_points: np.ndarray
    residual_m: float
    method: str
    centerline_xy: np.ndarray | None = None
    valid_centerline_mask: np.ndarray | None = None
    endpoint_nodes: np.ndarray | None = None
    endpoint_marker_centers_xyz: np.ndarray | None = None
    endpoint_marker_centers_xy: np.ndarray | None = None
    endpoint_marker_mask: np.ndarray | None = None
    endpoint_marker_count: int = 0


@dataclass
class EndpointMarker3D:
    mask: np.ndarray
    centers_xyz: np.ndarray
    centers_xy: np.ndarray
    endpoint_nodes: np.ndarray | None
    component_count: int
    points_xyz: np.ndarray


class CableMaskDetector:
    """Base class for cable mask detectors.

    Subclasses provide ``create_mask``. This class only handles ROI/scale,
    cleanup, skeletonization, and ordered centerline extraction.
    """

    def __init__(
        self,
        min_area=80,
        keep_largest_component=False,
        max_components=0,
        allow_occluded_fragments=True,
        open_kernel=3,
        close_kernel=5,
        skeleton_prune_px=10,
        skeleton_prune_passes=2,
        centerline_smooth_window=9,
    ):
        self.min_area = int(min_area)
        self.keep_largest_component = bool(keep_largest_component)
        self.max_components = int(max_components)
        self.allow_occluded_fragments = bool(allow_occluded_fragments)
        self.open_kernel = odd_kernel_size(open_kernel)
        self.close_kernel = odd_kernel_size(close_kernel)
        self.skeleton_prune_px = max(0, int(skeleton_prune_px))
        self.skeleton_prune_passes = max(0, int(skeleton_prune_passes))
        self.centerline_smooth_window = max(1, int(centerline_smooth_window))

    def create_mask(self, bgr):
        raise NotImplementedError("Subclasses must return a binary cable mask.")

    def clean_mask(self, raw_mask):
        mask = np.asarray(raw_mask, dtype=np.uint8)
        if self.open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return remove_small_components(
            mask,
            min_area=self.min_area,
            keep_largest_component=self.keep_largest_component and not self.allow_occluded_fragments,
            max_components=0 if self.allow_occluded_fragments else self.max_components,
        )

    def detect(self, bgr, roi_bbox=None, scale=1.0, extract_geometry=True):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if roi_bbox is None and float(scale) >= 0.999:
            return self._detect_crop(bgr, extract_geometry=extract_geometry)

        crop, bbox = crop_bgr_to_bbox(bgr, roi_bbox)
        if crop.size == 0:
            return empty_detection(bgr.shape[:2])

        original_h, original_w = crop.shape[:2]
        scale = float(np.clip(scale, 0.10, 1.0))
        if scale < 0.999:
            scaled_w = max(2, int(round(original_w * scale)))
            scaled_h = max(2, int(round(original_h * scale)))
            detector_input = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            detection = self._detect_crop(detector_input, extract_geometry=extract_geometry)
            detection = resize_detection(detection, (original_h, original_w))
        else:
            detection = self._detect_crop(crop, extract_geometry=extract_geometry)

        return offset_detection(detection, bgr.shape[:2], bbox)

    def _detect_crop(self, bgr, extract_geometry=True):
        raw_mask = self.create_mask(bgr)
        mask, component_count = self.clean_mask(raw_mask)
        if not extract_geometry:
            skeleton = np.zeros_like(mask, dtype=np.uint8)
            return CableDetection2D(
                mask=mask,
                skeleton=skeleton,
                centerline_xy=np.empty((0, 2), dtype=np.float32),
                component_count=component_count,
                branch_count=0,
                centerline_paths_xy=(),
            )
        skeleton = skeletonize_mask(mask)
        skeleton = prune_short_skeleton_branches(
            skeleton,
            max_branch_length=self.skeleton_prune_px,
            max_passes=self.skeleton_prune_passes,
        )
        centerline_paths_xy, branch_count = skeleton_centerline_paths(skeleton)
        centerline_xy = stitch_centerline_paths(centerline_paths_xy)
        centerline_xy = smooth_polyline_xy(centerline_xy, self.centerline_smooth_window)
        return CableDetection2D(
            mask=mask,
            skeleton=skeleton,
            centerline_xy=centerline_xy,
            component_count=component_count,
            branch_count=branch_count,
            centerline_paths_xy=tuple(centerline_paths_xy),
        )


def crop_bgr_to_bbox(bgr, roi_bbox=None):
    bgr = np.asarray(bgr, dtype=np.uint8)
    height, width = bgr.shape[:2]
    if roi_bbox is None:
        return bgr, (0, 0, width, height)

    x0, y0, x1, y1 = [int(round(float(v))) for v in roi_bbox]
    x0 = int(np.clip(x0, 0, width))
    x1 = int(np.clip(x1, 0, width))
    y0 = int(np.clip(y0, 0, height))
    y1 = int(np.clip(y1, 0, height))
    if x1 <= x0 or y1 <= y0:
        return bgr[:0, :0], (0, 0, 0, 0)
    return bgr[y0:y1, x0:x1], (x0, y0, x1, y1)


def empty_detection(image_shape):
    height, width = [int(v) for v in image_shape[:2]]
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    return CableDetection2D(
        mask=empty_mask,
        skeleton=empty_mask.copy(),
        centerline_xy=np.empty((0, 2), dtype=np.float32),
        component_count=0,
        branch_count=0,
        centerline_paths_xy=(),
    )


def resize_detection(detection, output_shape):
    output_h, output_w = [int(v) for v in output_shape[:2]]
    input_h, input_w = detection.mask.shape[:2]
    if input_h == output_h and input_w == output_w:
        return detection

    scale_x = output_w / max(float(input_w), 1.0)
    scale_y = output_h / max(float(input_h), 1.0)
    mask = cv2.resize(detection.mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    skeleton = cv2.resize(detection.skeleton, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    centerline_xy = scale_xy_points(detection.centerline_xy, scale_x, scale_y)
    paths = tuple(scale_xy_points(path, scale_x, scale_y) for path in detection.centerline_paths_xy)
    return CableDetection2D(
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
        skeleton=np.ascontiguousarray(skeleton, dtype=np.uint8),
        centerline_xy=centerline_xy,
        component_count=int(detection.component_count),
        branch_count=int(detection.branch_count),
        centerline_paths_xy=paths,
    )


def offset_detection(detection, image_shape, bbox):
    height, width = [int(v) for v in image_shape[:2]]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if x0 == 0 and y0 == 0 and x1 == width and y1 == height:
        return detection

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_skeleton = np.zeros((height, width), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        crop_h = y1 - y0
        crop_w = x1 - x0
        mask = detection.mask
        skeleton = detection.skeleton
        if mask.shape[:2] != (crop_h, crop_w):
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        if skeleton.shape[:2] != (crop_h, crop_w):
            skeleton = cv2.resize(skeleton, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        full_mask[y0:y1, x0:x1] = mask
        full_skeleton[y0:y1, x0:x1] = skeleton

    centerline_xy = translate_xy_points(detection.centerline_xy, x0, y0)
    paths = tuple(translate_xy_points(path, x0, y0) for path in detection.centerline_paths_xy)
    return CableDetection2D(
        mask=full_mask,
        skeleton=full_skeleton,
        centerline_xy=centerline_xy,
        component_count=int(detection.component_count),
        branch_count=int(detection.branch_count),
        centerline_paths_xy=paths,
    )


def scale_xy_points(points_xy, scale_x, scale_y):
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    output = points[:, :2].copy()
    output[:, 0] *= float(scale_x)
    output[:, 1] *= float(scale_y)
    return np.ascontiguousarray(output, dtype=np.float32)


def translate_xy_points(points_xy, offset_x, offset_y):
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    output = points[:, :2].copy()
    output[:, 0] += float(offset_x)
    output[:, 1] += float(offset_y)
    return np.ascontiguousarray(output, dtype=np.float32)


def cable_measurement_from_mask_points(
    point_cloud,
    detection,
    segment_count=12,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    max_points=512,
    reference_nodes=None,
    reference_gate_m=0.0,
    reference_min_points=0,
):
    if detection is None or detection.mask is None:
        return None

    source_points = sampled_masked_point_cloud_points(
        point_cloud,
        detection.mask,
        depth_min=depth_min,
        depth_max=depth_max,
        confidence_map=confidence_map,
        max_confidence=max_confidence,
        max_points=max_points,
    )
    source_points = gate_points_to_reference(
        source_points,
        reference_nodes,
        gate_m=reference_gate_m,
        min_points=reference_min_points,
    )
    if len(source_points) < max(2, int(reference_min_points)):
        return None

    residual = 0.0
    if reference_nodes is not None:
        residual = polyline_residual(source_points, reference_nodes)
    return CableEstimate3D(
        points_xyz=np.empty((0, 3), dtype=np.float32),
        source_points=np.ascontiguousarray(source_points, dtype=np.float32),
        residual_m=float(residual),
        method=f"mask-only cable point support from RGB mask + ZED points | segments={int(segment_count)}",
        centerline_xy=np.empty((0, 2), dtype=np.float32),
        valid_centerline_mask=np.zeros(0, dtype=bool),
    )


def endpoint_markers_from_mask(
    mask,
    point_cloud,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    reference_nodes=None,
    min_area_px=50,
    min_points_per_marker=8,
    max_points_per_marker=256,
    tape_length_m=0.035,
    offset_to_tips=False,
    max_markers=2,
):
    mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    markers = []
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        component_mask = (labels == label).astype(np.uint8) * 255
        points = sampled_masked_point_cloud_points(
            point_cloud,
            component_mask,
            depth_min=depth_min,
            depth_max=depth_max,
            confidence_map=confidence_map,
            max_confidence=max_confidence,
            max_points=max_points_per_marker,
        )
        if len(points) < int(min_points_per_marker):
            continue
        center_xyz = np.nanmedian(points[:, :3], axis=0).astype(np.float32)
        if not np.all(np.isfinite(center_xyz)):
            continue
        markers.append(
            {
                "area": area,
                "point_count": int(len(points)),
                "center_xyz": center_xyz,
                "center_xy": np.asarray(centroids[label], dtype=np.float32),
                "points": points,
                "label": label,
            }
        )

    if not markers:
        return None

    markers.sort(key=lambda item: (item["point_count"], item["area"]), reverse=True)
    max_markers = max(1, int(max_markers))
    markers = markers[:max_markers]
    markers = order_marker_records(markers, reference_nodes)
    for marker in markers:
        cleaned[labels == marker["label"]] = 255

    centers_xyz = np.ascontiguousarray([marker["center_xyz"] for marker in markers], dtype=np.float32)
    centers_xy = np.ascontiguousarray([marker["center_xy"] for marker in markers], dtype=np.float32)
    points_xyz = np.concatenate([marker["points"] for marker in markers], axis=0).astype(np.float32, copy=False)
    endpoint_nodes = endpoint_nodes_from_marker_centers(
        centers_xyz,
        reference_nodes=reference_nodes,
        tape_length_m=tape_length_m,
        offset_to_tips=offset_to_tips,
    )
    return EndpointMarker3D(
        mask=np.ascontiguousarray(cleaned, dtype=np.uint8),
        centers_xyz=centers_xyz,
        centers_xy=centers_xy,
        endpoint_nodes=endpoint_nodes,
        component_count=len(markers),
        points_xyz=np.ascontiguousarray(points_xyz, dtype=np.float32),
    )


def attach_endpoint_markers_to_measurement(measurement, markers):
    if measurement is None or markers is None:
        return measurement
    measurement.endpoint_nodes = markers.endpoint_nodes
    measurement.endpoint_marker_centers_xyz = markers.centers_xyz
    measurement.endpoint_marker_centers_xy = markers.centers_xy
    measurement.endpoint_marker_mask = markers.mask
    measurement.endpoint_marker_count = int(markers.component_count)
    return measurement


def cleanup_marker_mask(mask, open_kernel=3, close_kernel=5):
    mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    open_kernel = odd_kernel_size(open_kernel)
    close_kernel = odd_kernel_size(close_kernel)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return np.ascontiguousarray(mask, dtype=np.uint8)


def order_marker_records(markers, reference_nodes=None):
    markers = list(markers)
    if len(markers) != 2:
        return markers
    reference = valid_xyz(reference_nodes)
    if len(reference) < 2:
        centers = np.asarray([marker["center_xyz"] for marker in markers], dtype=np.float64)
        return [markers[int(index)] for index in np.argsort(centers[:, 0])]
    centers = np.asarray([marker["center_xyz"] for marker in markers], dtype=np.float64)
    direct = float(np.linalg.norm(centers[0] - reference[0]) + np.linalg.norm(centers[1] - reference[-1]))
    reverse = float(np.linalg.norm(centers[1] - reference[0]) + np.linalg.norm(centers[0] - reference[-1]))
    return markers if direct <= reverse else [markers[1], markers[0]]


def endpoint_nodes_from_marker_centers(
    centers_xyz,
    reference_nodes=None,
    tape_length_m=0.035,
    offset_to_tips=False,
):
    centers = valid_xyz(centers_xyz)
    if len(centers) == 0:
        return None
    half_tape = 0.5 * max(0.0, float(tape_length_m)) if bool(offset_to_tips) else 0.0
    if len(centers) >= 2:
        centers = centers[:2]
        reference = valid_xyz(reference_nodes)
        if len(reference) >= 2:
            centers = order_endpoint_pair_to_reference(centers, reference)
        direction = centers[-1] - centers[0]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction /= norm
            endpoints = centers.copy()
            endpoints[0] -= half_tape * direction
            endpoints[-1] += half_tape * direction
        else:
            endpoints = centers.copy()
        return np.ascontiguousarray(endpoints, dtype=np.float32)

    reference = valid_xyz(reference_nodes)
    if len(reference) < 2:
        return None
    endpoints = np.full((2, 3), np.nan, dtype=np.float32)
    center = centers[0]
    if float(np.linalg.norm(center - reference[0])) <= float(np.linalg.norm(center - reference[-1])):
        endpoints[0] = center - half_tape * unit_vector(reference[1] - reference[0])
    else:
        endpoints[-1] = center + half_tape * unit_vector(reference[-1] - reference[-2])
    return np.ascontiguousarray(endpoints, dtype=np.float32)


def order_endpoint_pair_to_reference(endpoints, reference_nodes):
    endpoints = valid_xyz(endpoints)
    reference = valid_xyz(reference_nodes)
    if len(endpoints) < 2 or len(reference) < 2:
        return endpoints
    direct = float(np.linalg.norm(endpoints[0] - reference[0]) + np.linalg.norm(endpoints[-1] - reference[-1]))
    reverse = float(np.linalg.norm(endpoints[-1] - reference[0]) + np.linalg.norm(endpoints[0] - reference[-1]))
    return endpoints if direct <= reverse else endpoints[::-1].copy()


def valid_xyz(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    points = points[:, :3]
    return np.ascontiguousarray(points[np.all(np.isfinite(points), axis=1)], dtype=np.float64)


def unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def gate_points_to_reference(points_xyz, reference_nodes, gate_m=0.0, min_points=0):
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    gate_m = float(gate_m)
    if gate_m <= 0.0 or reference_nodes is None:
        return np.ascontiguousarray(points, dtype=np.float32)

    reference = np.asarray(reference_nodes, dtype=np.float32)
    if reference.ndim != 2 or reference.shape[1] < 3 or len(reference) < 2:
        return np.ascontiguousarray(points, dtype=np.float32)
    distances = point_to_polyline_distances(points, reference)
    if len(distances) != len(points):
        return np.empty((0, 3), dtype=np.float32)
    keep = distances <= gate_m
    if int(np.count_nonzero(keep)) < max(2, int(min_points)):
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(points[keep], dtype=np.float32)


def masked_point_cloud_points(
    point_cloud,
    mask,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    max_points=0,
):
    try:
        point_data = np.asarray(point_cloud.get_data())
    except Exception:
        point_data = np.asarray(point_cloud)

    if point_data.ndim != 3 or point_data.shape[2] < 3:
        return np.empty((0, 3), dtype=np.float32)

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[:2] != point_data.shape[:2]:
        mask = cv2.resize(mask, (point_data.shape[1], point_data.shape[0]), interpolation=cv2.INTER_NEAREST)

    selected = mask > 0
    xyz = point_data[:, :, :3].astype(np.float32)
    finite = np.all(np.isfinite(xyz), axis=2)
    selected &= finite
    distances = np.linalg.norm(xyz, axis=2)
    if depth_min is not None:
        selected &= distances >= float(depth_min)
    if depth_max is not None:
        selected &= distances <= float(depth_max)
    confidence = confidence_array(confidence_map, point_data.shape[:2])
    if confidence is not None and max_confidence is not None:
        selected &= confidence <= float(max_confidence)

    points = xyz[selected]
    max_points = max(0, int(max_points))
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return np.ascontiguousarray(points, dtype=np.float32)


def sampled_masked_point_cloud_points(
    point_cloud,
    mask,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    max_points=512,
    oversample=4,
):
    try:
        point_data = np.asarray(point_cloud.get_data())
    except Exception:
        point_data = np.asarray(point_cloud)

    if point_data.ndim != 3 or point_data.shape[2] < 3:
        return np.empty((0, 3), dtype=np.float32)

    max_points = max(0, int(max_points))
    if max_points <= 0:
        return masked_point_cloud_points(
            point_cloud,
            mask,
            depth_min=depth_min,
            depth_max=depth_max,
            confidence_map=confidence_map,
            max_confidence=max_confidence,
            max_points=0,
        )

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[:2] != point_data.shape[:2]:
        mask = cv2.resize(mask, (point_data.shape[1], point_data.shape[0]), interpolation=cv2.INTER_NEAREST)

    height, width = point_data.shape[:2]
    flat_indices = np.flatnonzero(mask.reshape(-1) > 0)
    if len(flat_indices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    candidate_count = min(len(flat_indices), max(max_points, int(max_points) * max(1, int(oversample))))
    candidate_indices = evenly_sample_indices(flat_indices, candidate_count)
    points = filtered_points_at_flat_indices(
        point_data,
        candidate_indices,
        width,
        depth_min=depth_min,
        depth_max=depth_max,
        confidence_map=confidence_map,
        max_confidence=max_confidence,
    )

    if len(points) < max_points and candidate_count < len(flat_indices):
        retry_count = min(len(flat_indices), max(max_points * 12, candidate_count * 3))
        if retry_count > candidate_count:
            candidate_indices = evenly_sample_indices(flat_indices, retry_count)
            points = filtered_points_at_flat_indices(
                point_data,
                candidate_indices,
                width,
                depth_min=depth_min,
                depth_max=depth_max,
                confidence_map=confidence_map,
                max_confidence=max_confidence,
            )

    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return np.ascontiguousarray(points, dtype=np.float32)


def evenly_sample_indices(indices, count):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    count = max(0, int(count))
    if count <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    if count >= len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return indices[positions]


def filtered_points_at_flat_indices(
    point_data,
    flat_indices,
    width,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
):
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    if len(flat_indices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    ys = flat_indices // int(width)
    xs = flat_indices - ys * int(width)
    xyz = point_data[ys, xs, :3].astype(np.float32, copy=False)
    keep = np.all(np.isfinite(xyz), axis=1)
    distances = np.linalg.norm(xyz, axis=1)
    if depth_min is not None:
        keep &= distances >= float(depth_min)
    if depth_max is not None:
        keep &= distances <= float(depth_max)
    if confidence_map is not None and max_confidence is not None:
        confidence = confidence_values_at_pixels(confidence_map, point_data.shape[:2], ys, xs)
        if confidence is not None and len(confidence) == len(keep):
            keep &= confidence <= float(max_confidence)
    return np.ascontiguousarray(xyz[keep], dtype=np.float32)


def confidence_values_at_pixels(confidence_map, target_shape, ys, xs):
    try:
        data = np.asarray(confidence_map.get_data())
    except Exception:
        data = np.asarray(confidence_map)
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.ndim != 2:
        return None
    if data.shape[:2] != tuple(target_shape):
        data = cv2.resize(
            data.astype(np.float32),
            (int(target_shape[1]), int(target_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(data[ys, xs], dtype=np.float32)


def fit_polyline_segments(points_xyz, segment_count=12):
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return None
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 2:
        return None
    return resample_polyline(points, max(2, int(segment_count) + 1)).astype(np.float32)


def resample_polyline(points_xyz, output_count):
    points = np.asarray(points_xyz, dtype=np.float64)
    output_count = max(2, int(output_count))
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:, :3], output_count, axis=0).astype(np.float32)

    deltas = np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 1e-9:
        return np.repeat(points[:1, :3], output_count, axis=0).astype(np.float32)

    target = np.linspace(0.0, total, output_count)
    output = np.empty((output_count, 3), dtype=np.float64)
    for axis in range(3):
        output[:, axis] = np.interp(target, cumulative, points[:, axis])
    return output.astype(np.float32)


def polyline_residual(points, nodes):
    distances = point_to_polyline_distances(points, nodes)
    if len(distances) == 0:
        return 0.0
    return float(np.median(distances))


def point_to_polyline_distances(points, nodes):
    points = np.asarray(points, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) < 2:
        return np.empty((0,), dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    nodes = nodes[np.all(np.isfinite(nodes[:, :3]), axis=1), :3]
    if len(points) == 0 or len(nodes) < 2:
        return np.empty((0,), dtype=np.float64)

    best = np.full(len(points), np.inf, dtype=np.float64)
    for start, end in zip(nodes[:-1], nodes[1:]):
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            candidate = np.linalg.norm(points - start[None, :], axis=1)
        else:
            t = np.clip(((points - start[None, :]) @ segment) / length_sq, 0.0, 1.0)
            projection = start[None, :] + t[:, None] * segment[None, :]
            candidate = np.linalg.norm(points - projection, axis=1)
        best = np.minimum(best, candidate)
    return best[np.isfinite(best)]


def remove_small_components(mask, min_area=80, keep_largest_component=False, max_components=0):
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask), 0

    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            components.append((area, label))

    if not components:
        return np.zeros_like(mask), 0

    components.sort(reverse=True)
    if keep_largest_component:
        components = components[:1]
    elif int(max_components) > 0:
        components = components[: int(max_components)]

    cleaned = np.zeros_like(mask)
    for _area, label in components:
        cleaned[labels == label] = 255
    return cleaned, len(components)


def skeletonize_mask(mask, max_iterations=200):
    image = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(image):
        return np.zeros_like(mask, dtype=np.uint8)

    image = np.pad(image, 1, mode="constant", constant_values=0)
    for _ in range(int(max_iterations)):
        changed = False
        for subiteration in (0, 1):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            p1 = image[1:-1, 1:-1]

            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transition_count = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if subiteration == 0:
                marker = (
                    (p1 == 1)
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transition_count == 1)
                    & ((p2 * p4 * p6) == 0)
                    & ((p4 * p6 * p8) == 0)
                )
            else:
                marker = (
                    (p1 == 1)
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transition_count == 1)
                    & ((p2 * p4 * p8) == 0)
                    & ((p2 * p6 * p8) == 0)
                )

            if np.any(marker):
                p1[marker] = 0
                changed = True
        if not changed:
            break

    return (image[1:-1, 1:-1] * 255).astype(np.uint8)


def build_skeleton_graph(skeleton):
    coords_yx = np.argwhere(np.asarray(skeleton) > 0)
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_yx)}
    neighbors = [[] for _ in range(len(coords_yx))]
    for idx, (y, x) in enumerate(coords_yx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_idx = coord_to_idx.get((y + dy, x + dx))
                if neighbor_idx is not None:
                    neighbors[idx].append(neighbor_idx)
    return coords_yx, neighbors


def prune_short_skeleton_branches(skeleton, max_branch_length=10, max_passes=2):
    pruned = (np.asarray(skeleton, dtype=np.uint8) > 0).astype(np.uint8)
    max_branch_length = max(0, int(max_branch_length))
    max_passes = max(0, int(max_passes))
    if max_branch_length <= 0 or max_passes <= 0 or not np.any(pruned):
        return (pruned * 255).astype(np.uint8)

    for _ in range(max_passes):
        coords_yx, neighbors = build_skeleton_graph(pruned)
        if len(coords_yx) == 0:
            break
        degrees = np.array([len(item) for item in neighbors], dtype=np.int32)
        endpoints = np.flatnonzero(degrees == 1)
        remove_indices = set()

        for endpoint in endpoints:
            endpoint = int(endpoint)
            path = [endpoint]
            previous = -1
            current = endpoint

            while True:
                next_nodes = [idx for idx in neighbors[current] if idx != previous]
                if not next_nodes:
                    break
                if len(next_nodes) > 1:
                    break
                previous, current = current, int(next_nodes[0])
                path.append(current)
                if degrees[current] != 2:
                    break
                if len(path) > max_branch_length + 1:
                    break

            if len(path) <= max_branch_length + 1 and degrees[current] > 2:
                remove_indices.update(path[:-1])

        if not remove_indices:
            break
        remove_yx = coords_yx[np.fromiter(remove_indices, dtype=np.int64)]
        pruned[remove_yx[:, 0], remove_yx[:, 1]] = 0

    return (pruned * 255).astype(np.uint8)


def skeleton_centerline_paths(skeleton):
    skeleton = np.asarray(skeleton, dtype=np.uint8)
    if not np.any(skeleton):
        return [], 0

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((skeleton > 0).astype(np.uint8), connectivity=8)
    paths = []
    total_branch_count = 0
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > 0:
            components.append((area, label))
    components.sort(reverse=True)

    for _area, label in components:
        component_skeleton = np.zeros_like(skeleton, dtype=np.uint8)
        component_skeleton[labels == label] = 255
        coords_yx, neighbors = build_skeleton_graph(component_skeleton)
        path_xy, branch_count = longest_skeleton_path(coords_yx, neighbors)
        total_branch_count += branch_count
        if len(path_xy) >= 2:
            paths.append(path_xy)

    return paths, int(total_branch_count)


def stitch_centerline_paths(paths):
    remaining = [
        np.asarray(path, dtype=np.float32)[:, :2]
        for path in paths
        if np.asarray(path).ndim == 2 and np.asarray(path).shape[1] >= 2 and len(path) >= 2
    ]
    if not remaining:
        return np.empty((0, 2), dtype=np.float32)

    start_index = int(np.argmax([polyline_length(path) for path in remaining]))
    stitched = remaining.pop(start_index).copy()

    while remaining:
        best = None
        for index, path in enumerate(remaining):
            candidates = (
                (np.linalg.norm(stitched[-1] - path[0]), "append", False),
                (np.linalg.norm(stitched[-1] - path[-1]), "append", True),
                (np.linalg.norm(stitched[0] - path[-1]), "prepend", False),
                (np.linalg.norm(stitched[0] - path[0]), "prepend", True),
            )
            distance, side, reverse = min(candidates, key=lambda item: item[0])
            if best is None or distance < best[0]:
                best = (distance, index, side, reverse)

        _distance, index, side, reverse = best
        path = remaining.pop(index)
        if reverse:
            path = path[::-1].copy()
        if side == "append":
            stitched = np.vstack([stitched, path])
        else:
            stitched = np.vstack([path, stitched])

    return np.ascontiguousarray(stitched, dtype=np.float32)


def smooth_polyline_xy(points_xy, window=9):
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 3:
        return np.empty((0, 2), dtype=np.float32) if len(points) == 0 else np.ascontiguousarray(points[:, :2], dtype=np.float32)

    window = max(1, int(window))
    if window <= 1:
        return np.ascontiguousarray(points[:, :2], dtype=np.float32)
    if window % 2 == 0:
        window += 1
    window = min(window, len(points) if len(points) % 2 == 1 else len(points) - 1)
    if window <= 1:
        return np.ascontiguousarray(points[:, :2], dtype=np.float32)

    radius = window // 2
    padded = np.pad(points[:, :2], ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    smoothed = np.empty((len(points), 2), dtype=np.float32)
    for axis in range(2):
        smoothed[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    smoothed[0] = points[0, :2]
    smoothed[-1] = points[-1, :2]
    return np.ascontiguousarray(smoothed, dtype=np.float32)


def polyline_length(points_xy):
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))


def longest_skeleton_path(coords_yx, neighbors):
    if len(coords_yx) == 0:
        return np.empty((0, 2), dtype=np.float32), 0

    degrees = np.array([len(item) for item in neighbors], dtype=np.int32)
    endpoint_indices = np.flatnonzero(degrees == 1)
    branch_count = int(np.count_nonzero(degrees > 2))

    starts = endpoint_indices if len(endpoint_indices) >= 2 else np.arange(len(coords_yx))
    best_path = []
    best_distance = -1
    for start in starts:
        distances, parents = bfs_path(neighbors, int(start))
        candidate_ends = endpoint_indices if len(endpoint_indices) >= 2 else np.arange(len(coords_yx))
        for end in candidate_ends:
            end = int(end)
            if end == int(start) or distances[end] <= best_distance:
                continue
            path = reconstruct_path(parents, int(start), end)
            if path:
                best_path = path
                best_distance = distances[end]

    if not best_path:
        best_path = list(range(len(coords_yx)))

    path_yx = coords_yx[np.asarray(best_path, dtype=np.int64)]
    path_xy = np.column_stack([path_yx[:, 1], path_yx[:, 0]])
    return np.ascontiguousarray(path_xy, dtype=np.float32), branch_count


def bfs_path(neighbors, start_idx):
    distances = [-1] * len(neighbors)
    parents = [-1] * len(neighbors)
    queue = deque([start_idx])
    distances[start_idx] = 0
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            if distances[neighbor] != -1:
                continue
            distances[neighbor] = distances[current] + 1
            parents[neighbor] = current
            queue.append(neighbor)
    return distances, parents


def reconstruct_path(parents, start_idx, end_idx):
    path = [end_idx]
    current = end_idx
    while current != start_idx:
        current = parents[current]
        if current == -1:
            return []
        path.append(current)
    path.reverse()
    return path


def confidence_array(confidence_map, target_shape):
    if confidence_map is None:
        return None
    try:
        data = np.asarray(confidence_map.get_data())
    except Exception:
        data = np.asarray(confidence_map)
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.ndim != 2:
        return None
    if data.shape[:2] != tuple(target_shape):
        data = cv2.resize(
            data.astype(np.float32),
            (int(target_shape[1]), int(target_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(data, dtype=np.float32)


def odd_kernel_size(value):
    value = int(max(0, value))
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1
