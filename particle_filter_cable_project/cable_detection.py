from dataclasses import dataclass

import cv2
import numpy as np

from cable_cuda import CudaPointCloudView, sample_indexed_points, sample_masked_points


@dataclass
class CableDetection2D:
    mask: np.ndarray
    component_count: int


@dataclass
class CableEstimate3D:
    points_xyz: np.ndarray
    source_points: np.ndarray
    residual_m: float
    method: str
    endpoint_nodes: np.ndarray | None = None
    endpoint_marker_centers_xyz: np.ndarray | None = None
    endpoint_marker_centers_xy: np.ndarray | None = None
    endpoint_marker_mask: np.ndarray | None = None
    endpoint_marker_count: int = 0
    pf_node_runs: tuple | None = None
    endpoint_marker_pf_ids: np.ndarray | None = None
    endpoint_marker_end_indices: np.ndarray | None = None


@dataclass
class EndpointMarker3D:
    mask: np.ndarray
    centers_xyz: np.ndarray
    centers_xy: np.ndarray
    endpoint_nodes: np.ndarray | None
    component_count: int
    points_xyz: np.ndarray


@dataclass
class EndpointGroupObservations3D:
    """3D endpoint candidates belonging to one physical cable."""

    mask: np.ndarray
    centers_xyz: np.ndarray
    centers_xy: np.ndarray
    component_count: int
    point_counts: np.ndarray
    areas_px: np.ndarray


class CableMaskDetector:
    """Shared morphological cleanup for neural cable masks."""

    def __init__(
        self,
        min_area=80,
        open_kernel=3,
        close_kernel=5,
    ):
        self.min_area = int(min_area)
        self.open_kernel = odd_kernel_size(open_kernel)
        self.close_kernel = odd_kernel_size(close_kernel)

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
            keep_largest_component=False,
            max_components=0,
        )


def resize_detection(detection, output_shape):
    output_h, output_w = [int(v) for v in output_shape[:2]]
    input_h, input_w = detection.mask.shape[:2]
    if input_h == output_h and input_w == output_w:
        return detection

    mask = cv2.resize(detection.mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    return CableDetection2D(
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
        component_count=int(detection.component_count),
    )


def cable_measurement_from_mask_points(
    point_cloud,
    detection,
    segment_count=12,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    max_points=512,
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
    return cable_measurement_from_support_points(
        source_points,
        segment_count=segment_count,
    )


def cable_measurement_from_support_points(
    source_points,
    segment_count=12,
):
    source_points = np.asarray(source_points, dtype=np.float32)
    if source_points.ndim != 2 or source_points.shape[1] < 3:
        return None
    source_points = source_points[np.all(np.isfinite(source_points[:, :3]), axis=1), :3]
    if len(source_points) < 2:
        return None

    return CableEstimate3D(
        points_xyz=np.empty((0, 3), dtype=np.float32),
        source_points=np.ascontiguousarray(source_points, dtype=np.float32),
        residual_m=0.0,
        method=f"shared cable point support from RGB mask + ZED points | segments={int(segment_count)}",
    )


def endpoint_group_observations_from_mask(
    mask,
    point_cloud,
    *,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    min_area_px=50,
    min_points_per_component=8,
    max_points_per_component=256,
    max_components=2,
    open_kernel=3,
    close_kernel=5,
    oversample=4,
):
    """Lift both endpoint components for one cable without full component masks."""

    cleaned_input = cleanup_marker_mask(mask, open_kernel=open_kernel, close_kernel=close_kernel)
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_input, connectivity=8)
    if label_count <= 1:
        return None
    candidates = []
    width = int(cleaned_input.shape[1])
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(1, int(min_area_px)):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        local_y, local_x = np.nonzero(labels[y:y + component_height, x:x + component_width] == label)
        flat_indices = (local_y.astype(np.int64) + y) * width + local_x.astype(np.int64) + x
        candidate_limit = min(
            len(flat_indices),
            max(1, int(max_points_per_component)) * max(1, int(oversample)),
        )
        flat_indices = evenly_sample_indices(flat_indices, candidate_limit)
        candidates.append(
            {
                "label": label,
                "area": area,
                "center_xy": np.asarray(centroids[label], dtype=np.float32),
                "flat_indices": flat_indices,
            }
        )
    if not candidates:
        return None

    candidates.sort(key=lambda item: item["area"], reverse=True)
    candidates = candidates[:max(2, int(max_components) * 3)]
    if isinstance(point_cloud, CudaPointCloudView):
        all_indices = np.concatenate([item["flat_indices"] for item in candidates])
        gathered = sample_indexed_points(
            point_cloud,
            all_indices,
            depth_min=depth_min,
            depth_max=depth_max,
            max_confidence=max_confidence,
        ).cpu().numpy()
        cursor = 0
        for item in candidates:
            count = len(item["flat_indices"])
            points = np.asarray(gathered[cursor:cursor + count], dtype=np.float32)
            cursor += count
            item["points"] = points[np.all(np.isfinite(points), axis=1)]
    else:
        try:
            point_data = np.asarray(point_cloud.get_data())
        except Exception:
            point_data = np.asarray(point_cloud)
        if point_data.ndim != 3 or point_data.shape[2] < 3:
            return None
        for item in candidates:
            item["points"] = filtered_points_at_flat_indices(
                point_data,
                item["flat_indices"],
                point_data.shape[1],
                depth_min=depth_min,
                depth_max=depth_max,
                confidence_map=confidence_map,
                max_confidence=max_confidence,
            )

    accepted = []
    for item in candidates:
        points = np.asarray(item["points"], dtype=np.float32)
        if len(points) < max(1, int(min_points_per_component)):
            continue
        if len(points) > max(1, int(max_points_per_component)):
            points = points[np.linspace(0, len(points) - 1, int(max_points_per_component), dtype=np.int64)]
        center_xyz = np.nanmedian(points[:, :3], axis=0).astype(np.float32)
        if not np.all(np.isfinite(center_xyz)):
            continue
        item["center_xyz"] = center_xyz
        item["point_count"] = int(len(points))
        accepted.append(item)
    accepted.sort(key=lambda item: (item["point_count"], item["area"]), reverse=True)
    accepted = accepted[:max(1, int(max_components))]
    if not accepted:
        return None

    output_mask = np.zeros_like(cleaned_input, dtype=np.uint8)
    for item in accepted:
        output_mask[labels == item["label"]] = 255
    return EndpointGroupObservations3D(
        mask=np.ascontiguousarray(output_mask, dtype=np.uint8),
        centers_xyz=np.ascontiguousarray([item["center_xyz"] for item in accepted], dtype=np.float32),
        centers_xy=np.ascontiguousarray([item["center_xy"] for item in accepted], dtype=np.float32),
        component_count=len(accepted),
        point_counts=np.asarray([item["point_count"] for item in accepted], dtype=np.int32),
        areas_px=np.asarray([item["area"] for item in accepted], dtype=np.int32),
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
    if isinstance(point_cloud, CudaPointCloudView):
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape[:2] != (point_cloud.height, point_cloud.width):
            mask = cv2.resize(
                mask,
                (point_cloud.width, point_cloud.height),
                interpolation=cv2.INTER_NEAREST,
            )
        points_t = sample_masked_points(
            point_cloud,
            np.ascontiguousarray(mask, dtype=np.uint8),
            depth_min=depth_min,
            depth_max=depth_max,
            max_confidence=max_confidence,
            max_points=max_points,
            oversample=oversample,
        )
        return np.ascontiguousarray(points_t.cpu().numpy(), dtype=np.float32)

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
