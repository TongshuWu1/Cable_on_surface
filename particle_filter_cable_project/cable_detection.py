from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from cable_cuda import (
    CudaPointCloudView,
    OBSERVATION_ACCEPTED,
    OBSERVATION_INVALID_DEPTH,
    OBSERVATION_OUTSIDE_DEPTH_RANGE,
    OBSERVATION_POOR_CONFIDENCE,
    radius_neighbor_inlier_mask,
    sample_indexed_observations,
    sample_indexed_points,
    sample_masked_points,
)


@dataclass
class CableDetection2D:
    mask: np.ndarray
    component_count: int
    component_rejected_mask: np.ndarray | None = None
    morphology_rejected_mask: np.ndarray | None = None


OBSERVATION_REJECTION_KEYS = (
    "invalid_depth",
    "depth_range",
    "depth_confidence",
    "small_component",
    "mask_morphology",
    "spatial_isolation",
)


@dataclass
class CableObservation3D:
    """One explicit, PF-independent sampled cable observation."""

    accepted_points_xyz: np.ndarray
    rejected_points_xyz: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    rejected_pixels_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    rejection_counts: dict = field(default_factory=dict)
    spatial_filter_applied: bool = False

    def __post_init__(self):
        self.accepted_points_xyz = normalized_xyz(self.accepted_points_xyz)
        self.rejected_points_xyz = normalized_xyz(self.rejected_points_xyz)
        pixels = np.asarray(self.rejected_pixels_xy, dtype=np.int32)
        self.rejected_pixels_xy = (
            np.ascontiguousarray(pixels[:, :2], dtype=np.int32)
            if pixels.ndim == 2 and pixels.shape[1] >= 2
            else np.empty((0, 2), dtype=np.int32)
        )
        supplied = dict(self.rejection_counts or {})
        self.rejection_counts = {
            key: max(0, int(supplied.get(key, 0)))
            for key in OBSERVATION_REJECTION_KEYS
        }

    @property
    def accepted_count(self):
        return int(len(self.accepted_points_xyz))

    @property
    def rejected_count(self):
        return int(sum(self.rejection_counts.values()))


@dataclass
class CableEstimate3D:
    points_xyz: np.ndarray
    source_points: np.ndarray
    residual_m: float
    method: str
    observation: CableObservation3D | None = None
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


def normalized_xyz(points):
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    values = values[:, :3]
    return np.ascontiguousarray(values[np.all(np.isfinite(values), axis=1)], dtype=np.float32)


def resize_optional_mask(mask, output_shape):
    if mask is None:
        return np.zeros(tuple(output_shape[:2]), dtype=np.uint8)
    values = np.asarray(mask, dtype=np.uint8)
    output_h, output_w = int(output_shape[0]), int(output_shape[1])
    if values.shape[:2] != (output_h, output_w):
        values = cv2.resize(values, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(np.where(values > 0, 255, 0), dtype=np.uint8)


def empty_cable_observation(spatial_filter_applied=False):
    return CableObservation3D(
        accepted_points_xyz=np.empty((0, 3), dtype=np.float32),
        rejected_points_xyz=np.empty((0, 3), dtype=np.float32),
        rejected_pixels_xy=np.empty((0, 2), dtype=np.int32),
        rejection_counts={},
        spatial_filter_applied=bool(spatial_filter_applied),
    )


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
        raw = np.where(np.asarray(raw_mask, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
        mask = raw.copy()
        if self.open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        morphology_rejected = np.where((raw > 0) & (mask == 0), 255, 0).astype(np.uint8)
        cleaned, component_count = remove_small_components(
            mask,
            min_area=self.min_area,
            keep_largest_component=False,
            max_components=0,
        )
        component_rejected = np.where((mask > 0) & (cleaned == 0), 255, 0).astype(np.uint8)
        return cleaned, component_count, component_rejected, morphology_rejected


def resize_detection(detection, output_shape):
    output_h, output_w = [int(v) for v in output_shape[:2]]
    input_h, input_w = detection.mask.shape[:2]
    if input_h == output_h and input_w == output_w:
        return detection

    mask = cv2.resize(detection.mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    component_rejected = resize_optional_mask(
        detection.component_rejected_mask,
        (output_h, output_w),
    )
    morphology_rejected = resize_optional_mask(
        detection.morphology_rejected_mask,
        (output_h, output_w),
    )
    return CableDetection2D(
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
        component_count=int(detection.component_count),
        component_rejected_mask=component_rejected,
        morphology_rejected_mask=morphology_rejected,
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
    observation=None,
):
    if observation is not None and not isinstance(observation, CableObservation3D):
        raise TypeError("observation must be a CableObservation3D instance.")
    source_points = (
        observation.accepted_points_xyz
        if observation is not None
        else np.asarray(source_points, dtype=np.float32)
    )
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
        observation=observation,
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


def sampled_cable_observation(
    point_cloud,
    detection,
    *,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    max_points=1000,
    spatial_filter_enabled=False,
    spatial_radius_m=0.015,
    spatial_min_neighbors=2,
):
    """Lift one shared cable observation and retain every rejection decision.

    All decisions use the NN masks, ZED quality values, and the sampled 3D
    cloud only.  No particle-filter state or estimate is accepted by this API.
    """

    if detection is None or getattr(detection, "mask", None) is None:
        return empty_cable_observation(spatial_filter_enabled)
    max_points = max(1, int(max_points))
    spatial_radius_m = max(1e-6, float(spatial_radius_m))
    spatial_min_neighbors = max(0, int(spatial_min_neighbors))
    if isinstance(point_cloud, CudaPointCloudView):
        target_shape = (int(point_cloud.height), int(point_cloud.width))
        point_data = None
    else:
        try:
            point_data = np.asarray(point_cloud.get_data())
        except Exception:
            point_data = np.asarray(point_cloud)
        if point_data.ndim != 3 or point_data.shape[2] < 3:
            return empty_cable_observation(spatial_filter_enabled)
        target_shape = point_data.shape[:2]

    accepted_mask = resize_optional_mask(detection.mask, target_shape)
    component_mask = resize_optional_mask(
        getattr(detection, "component_rejected_mask", None),
        target_shape,
    )
    morphology_mask = resize_optional_mask(
        getattr(detection, "morphology_rejected_mask", None),
        target_shape,
    )
    clean_indices = evenly_sample_indices(
        np.flatnonzero(accepted_mask.reshape(-1)),
        max_points,
    )
    component_indices = evenly_sample_indices(
        np.flatnonzero(component_mask.reshape(-1)),
        max_points,
    )
    morphology_indices = evenly_sample_indices(
        np.flatnonzero(morphology_mask.reshape(-1)),
        max_points,
    )
    index_groups = (clean_indices, component_indices, morphology_indices)
    counts = tuple(len(indices) for indices in index_groups)
    if not any(counts):
        return empty_cable_observation(spatial_filter_enabled)
    combined_indices = np.ascontiguousarray(np.concatenate(index_groups), dtype=np.int64)

    if isinstance(point_cloud, CudaPointCloudView):
        raw_points_t, status_t = sample_indexed_observations(
            point_cloud,
            combined_indices,
            depth_min=depth_min,
            depth_max=depth_max,
            max_confidence=max_confidence,
        )
        return _build_cable_observation_cuda(
            raw_points_t,
            status_t,
            index_groups,
            target_shape[1],
            spatial_filter_enabled,
            spatial_radius_m,
            spatial_min_neighbors,
        )

    raw_points, status = observation_points_and_status_at_indices(
        point_data,
        combined_indices,
        target_shape[1],
        depth_min=depth_min,
        depth_max=depth_max,
        confidence_map=confidence_map,
        max_confidence=max_confidence,
    )
    return _build_cable_observation_cpu(
        raw_points,
        status,
        index_groups,
        target_shape[1],
        spatial_filter_enabled,
        spatial_radius_m,
        spatial_min_neighbors,
    )


def _build_cable_observation_cpu(
    raw_points,
    status,
    index_groups,
    width,
    spatial_filter_enabled,
    spatial_radius_m,
    spatial_min_neighbors,
):
    clean_indices, component_indices, morphology_indices = index_groups
    clean_count, component_count, morphology_count = [len(values) for values in index_groups]
    clean_points = np.asarray(raw_points[:clean_count], dtype=np.float32)
    clean_status = np.asarray(status[:clean_count], dtype=np.int32)
    cursor = clean_count
    component_points = np.asarray(raw_points[cursor:cursor + component_count], dtype=np.float32)
    cursor += component_count
    morphology_points = np.asarray(raw_points[cursor:cursor + morphology_count], dtype=np.float32)

    quality_mask = clean_status == OBSERVATION_ACCEPTED
    quality_points = np.ascontiguousarray(clean_points[quality_mask], dtype=np.float32)
    spatial_inliers = (
        radius_neighbor_inlier_mask_cpu(
            quality_points,
            spatial_radius_m,
            spatial_min_neighbors,
        )
        if spatial_filter_enabled
        else np.ones(len(quality_points), dtype=bool)
    )
    accepted_points = quality_points[spatial_inliers]
    isolated_points = quality_points[~spatial_inliers]
    quality_indices = clean_indices[quality_mask]

    rejected_finite = [
        clean_points[(clean_status != OBSERVATION_ACCEPTED) & np.all(np.isfinite(clean_points), axis=1)],
        isolated_points,
        component_points[np.all(np.isfinite(component_points), axis=1)],
        morphology_points[np.all(np.isfinite(morphology_points), axis=1)],
    ]
    rejected_points = normalized_xyz(np.vstack([values for values in rejected_finite if len(values)])) if any(
        len(values) for values in rejected_finite
    ) else np.empty((0, 3), dtype=np.float32)
    rejected_indices = np.concatenate((
        clean_indices[clean_status != OBSERVATION_ACCEPTED],
        quality_indices[~spatial_inliers],
        component_indices,
        morphology_indices,
    ))
    return CableObservation3D(
        accepted_points_xyz=accepted_points,
        rejected_points_xyz=rejected_points,
        rejected_pixels_xy=flat_indices_to_xy(rejected_indices, width),
        rejection_counts=observation_rejection_counts(
            clean_status,
            int(np.count_nonzero(~spatial_inliers)),
            component_count,
            morphology_count,
        ),
        spatial_filter_applied=bool(spatial_filter_enabled),
    )


def _build_cable_observation_cuda(
    raw_points_t,
    status_t,
    index_groups,
    width,
    spatial_filter_enabled,
    spatial_radius_m,
    spatial_min_neighbors,
):
    clean_indices, component_indices, morphology_indices = index_groups
    clean_count, component_count, morphology_count = [len(values) for values in index_groups]
    clean_points_t = raw_points_t[:clean_count]
    clean_status_t = status_t[:clean_count]
    cursor = clean_count
    component_points_t = raw_points_t[cursor:cursor + component_count]
    cursor += component_count
    morphology_points_t = raw_points_t[cursor:cursor + morphology_count]
    quality_mask_t = clean_status_t == OBSERVATION_ACCEPTED
    quality_points_t = clean_points_t[quality_mask_t].contiguous()
    spatial_inliers_t = (
        radius_neighbor_inlier_mask(
            quality_points_t,
            spatial_radius_m,
            spatial_min_neighbors,
        )
        if spatial_filter_enabled
        else quality_mask_t.new_ones(len(quality_points_t), dtype=torch.bool)
    )
    accepted_points_t = quality_points_t[spatial_inliers_t]
    rejected_parts_t = []
    clean_rejected_t = clean_points_t[~quality_mask_t]
    clean_rejected_t = clean_rejected_t[torch.isfinite(clean_rejected_t).all(dim=1)]
    if len(clean_rejected_t):
        rejected_parts_t.append(clean_rejected_t)
    isolated_t = quality_points_t[~spatial_inliers_t]
    if len(isolated_t):
        rejected_parts_t.append(isolated_t)
    for values_t in (component_points_t, morphology_points_t):
        finite_t = values_t[torch.isfinite(values_t).all(dim=1)]
        if len(finite_t):
            rejected_parts_t.append(finite_t)
    rejected_points_t = (
        torch.cat(rejected_parts_t, dim=0)
        if rejected_parts_t
        else raw_points_t.new_empty((0, 3))
    )

    clean_status = clean_status_t.cpu().numpy().astype(np.int32, copy=False)
    quality_mask = clean_status == OBSERVATION_ACCEPTED
    spatial_inliers = spatial_inliers_t.cpu().numpy().astype(bool, copy=False)
    quality_indices = clean_indices[quality_mask]
    rejected_indices = np.concatenate((
        clean_indices[~quality_mask],
        quality_indices[~spatial_inliers],
        component_indices,
        morphology_indices,
    ))
    return CableObservation3D(
        accepted_points_xyz=accepted_points_t.cpu().numpy(),
        rejected_points_xyz=rejected_points_t.cpu().numpy(),
        rejected_pixels_xy=flat_indices_to_xy(rejected_indices, width),
        rejection_counts=observation_rejection_counts(
            clean_status,
            int(np.count_nonzero(~spatial_inliers)),
            component_count,
            morphology_count,
        ),
        spatial_filter_applied=bool(spatial_filter_enabled),
    )


def observation_points_and_status_at_indices(
    point_data,
    flat_indices,
    width,
    *,
    depth_min,
    depth_max,
    confidence_map,
    max_confidence,
):
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    ys = flat_indices // int(width)
    xs = flat_indices - ys * int(width)
    xyz = np.ascontiguousarray(point_data[ys, xs, :3], dtype=np.float32)
    status = np.full(len(xyz), OBSERVATION_ACCEPTED, dtype=np.int32)
    finite = np.all(np.isfinite(xyz), axis=1)
    status[~finite] = OBSERVATION_INVALID_DEPTH
    distances = np.linalg.norm(np.where(finite[:, None], xyz, 0.0), axis=1)
    in_range = np.ones(len(xyz), dtype=bool)
    if depth_min is not None:
        in_range &= distances >= float(depth_min)
    if depth_max is not None:
        in_range &= distances <= float(depth_max)
    status[finite & ~in_range] = OBSERVATION_OUTSIDE_DEPTH_RANGE
    confidence = confidence_values_at_pixels(confidence_map, point_data.shape[:2], ys, xs)
    if confidence is not None and max_confidence is not None:
        confidence_ok = np.isfinite(confidence) & (confidence <= float(max_confidence))
        status[(status == OBSERVATION_ACCEPTED) & ~confidence_ok] = OBSERVATION_POOR_CONFIDENCE
    return xyz, status


def radius_neighbor_inlier_mask_cpu(points, radius_m, minimum_neighbors, chunk_size=512):
    points = normalized_xyz(points)
    minimum_neighbors = max(0, int(minimum_neighbors))
    if minimum_neighbors == 0:
        return np.ones(len(points), dtype=bool)
    if len(points) == 0:
        return np.empty(0, dtype=bool)
    radius_squared = float(radius_m) ** 2
    output = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), max(1, int(chunk_size))):
        stop = min(len(points), start + max(1, int(chunk_size)))
        delta = points[start:stop, None, :] - points[None, :, :]
        counts = np.sum(np.sum(delta * delta, axis=2) <= radius_squared, axis=1) - 1
        output[start:stop] = counts >= minimum_neighbors
    return output


def observation_rejection_counts(clean_status, isolation_count, component_count, morphology_count):
    clean_status = np.asarray(clean_status, dtype=np.int32)
    return {
        "invalid_depth": int(np.count_nonzero(clean_status == OBSERVATION_INVALID_DEPTH)),
        "depth_range": int(np.count_nonzero(clean_status == OBSERVATION_OUTSIDE_DEPTH_RANGE)),
        "depth_confidence": int(np.count_nonzero(clean_status == OBSERVATION_POOR_CONFIDENCE)),
        "small_component": int(component_count),
        "mask_morphology": int(morphology_count),
        "spatial_isolation": int(isolation_count),
    }


def flat_indices_to_xy(flat_indices, width):
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    if len(flat_indices) == 0:
        return np.empty((0, 2), dtype=np.int32)
    ys = flat_indices // int(width)
    xs = flat_indices - ys * int(width)
    return np.ascontiguousarray(np.stack((xs, ys), axis=1), dtype=np.int32)


def sampled_masked_point_cloud_point_sets(
    point_cloud,
    masks,
    *,
    depth_min=0.05,
    depth_max=None,
    confidence_map=None,
    max_confidence=None,
    max_points_by_mask=(),
    oversample=4,
):
    """Lift several independent masks with one CUDA gather and one CPU transfer."""

    masks = tuple(masks or ())
    limits = tuple(int(value) for value in max_points_by_mask)
    if len(limits) != len(masks):
        raise ValueError("max_points_by_mask must contain one limit for each mask.")
    if not isinstance(point_cloud, CudaPointCloudView):
        return tuple(
            sampled_masked_point_cloud_points(
                point_cloud,
                mask,
                depth_min=depth_min,
                depth_max=depth_max,
                confidence_map=confidence_map,
                max_confidence=max_confidence,
                max_points=max(0, limit),
                oversample=oversample,
            )
            if mask is not None
            else np.empty((0, 3), dtype=np.float32)
            for mask, limit in zip(masks, limits)
        )

    sampled_indices = []
    for mask, limit in zip(masks, limits):
        if mask is None:
            sampled_indices.append(np.empty(0, dtype=np.int64))
            continue
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.shape[:2] != (point_cloud.height, point_cloud.width):
            mask_array = cv2.resize(
                mask_array,
                (point_cloud.width, point_cloud.height),
                interpolation=cv2.INTER_NEAREST,
            )
        flat_indices = np.flatnonzero(mask_array.reshape(-1))
        limit = max(0, int(limit))
        if limit > 0 and len(flat_indices) > limit * max(1, int(oversample)):
            flat_indices = evenly_sample_indices(
                flat_indices,
                limit * max(1, int(oversample)),
            )
        sampled_indices.append(np.ascontiguousarray(flat_indices, dtype=np.int64))

    counts = [len(value) for value in sampled_indices]
    if not any(counts):
        return tuple(np.empty((0, 3), dtype=np.float32) for _ in masks)
    combined_indices = np.concatenate(sampled_indices)
    combined_points = sample_indexed_points(
        point_cloud,
        combined_indices,
        depth_min=depth_min,
        depth_max=depth_max,
        max_confidence=max_confidence,
    ).cpu().numpy()

    outputs = []
    cursor = 0
    for count, limit in zip(counts, limits):
        points = np.asarray(combined_points[cursor:cursor + count], dtype=np.float32)
        cursor += count
        points = points[np.all(np.isfinite(points), axis=1)]
        limit = max(0, int(limit))
        if limit > 0 and len(points) > limit:
            indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
            points = points[indices]
        outputs.append(np.ascontiguousarray(points, dtype=np.float32))
    return tuple(outputs)


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
