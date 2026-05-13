from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SphereDetection:
    center_xy: tuple[float, float]
    radius_px: float
    area: float
    circularity: float
    fill_ratio: float
    aspect: float
    contour: np.ndarray


@dataclass
class SphereEstimate3D:
    center_xyz: np.ndarray
    radius_m: float
    surface_points: np.ndarray
    residual_m: float
    method: str
    table_normal: np.ndarray | None = None
    table_offset: float | None = None
    image_center_xy: tuple[float, float] | None = None
    image_radius_px: float | None = None
    surface_normals: np.ndarray | None = None
    surface_weights: np.ndarray | None = None


@dataclass
class SurfacePatchObservation:
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray

    def __len__(self):
        return len(self.points)


@dataclass
class SceneSegmentation:
    ball_points: np.ndarray
    raw_ball_points: np.ndarray
    table_points: np.ndarray
    other_points: np.ndarray
    table_plane: "TablePlane | None" = None
    table_updated: bool = False
    method: str = "scene segmentation"

    @property
    def raw_ball_count(self):
        return int(len(self.raw_ball_points))

    @property
    def ball_count(self):
        return int(len(self.ball_points))

    @property
    def table_count(self):
        return int(len(self.table_points))

    @property
    def other_count(self):
        return int(len(self.other_points))


@dataclass
class TablePlane:
    normal: np.ndarray
    offset: float
    inlier_count: int
    residual_m: float


class SphereDetector:
    def __init__(self, profile):
        self.profile = profile
        self.hsv_ranges = [
            self._expand_range(item)
            for item in profile.get("hsv_ranges", [])
        ]
        blob_filter = profile.get("blob_filter", {})
        self.min_area = int(blob_filter.get("min_area", 250))
        self.min_circularity = float(blob_filter.get("min_circularity", 0.55))
        self.max_aspect_ratio = float(blob_filter.get("max_aspect_ratio", 1.6))
        self.open_kernel = _odd_kernel_size(blob_filter.get("open_kernel", 3))
        self.close_kernel = _odd_kernel_size(blob_filter.get("close_kernel", 7))

    @classmethod
    def from_profile_path(cls, path):
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def _expand_range(self, hsv_range):
        lower = np.array(hsv_range["lower"], dtype=np.int16)
        upper = np.array(hsv_range["upper"], dtype=np.int16)
        expansion = self.profile.get("range_expansion", {})
        h_extra = int(expansion.get("h_extra", 0))
        s_extra = int(expansion.get("s_extra", 0))
        v_extra = int(expansion.get("v_extra", 0))
        lower[0] = max(0, lower[0] - h_extra)
        lower[1] = max(0, lower[1] - s_extra)
        lower[2] = max(0, lower[2] - v_extra)
        upper[0] = min(179, upper[0] + h_extra)
        upper[1] = min(255, upper[1] + s_extra)
        upper[2] = min(255, upper[2] + v_extra)
        return {
            "lower": lower.astype(np.uint8),
            "upper": upper.astype(np.uint8),
        }

    def detect(self, bgr):
        mask = self.create_mask(bgr)
        detection, candidates = self.detect_blob(mask)
        return mask, detection, candidates

    def create_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for item in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, item["lower"], item["upper"]))

        if self.open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def detect_blob(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue

            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.min_circularity:
                continue

            _x, _y, width, height = cv2.boundingRect(contour)
            aspect = max(width, height) / max(1.0, min(width, height))
            if aspect > self.max_aspect_ratio:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            fill_ratio = area / max(np.pi * radius * radius, 1.0)
            score = area * circularity * np.clip(fill_ratio, 0.35, 1.2)
            candidates.append(
                (
                    score,
                    SphereDetection(
                        center_xy=(float(cx), float(cy)),
                        radius_px=float(radius),
                        area=area,
                        circularity=circularity,
                        fill_ratio=float(fill_ratio),
                        aspect=float(aspect),
                        contour=contour,
                    ),
                )
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        detections = [item[1] for item in candidates]
        return (detections[0] if detections else None), detections

    def detect_partial_blob(
        self,
        mask,
        predicted_center_xy,
        predicted_radius_px,
        search_scale=2.4,
        min_area_ratio=0.06,
        min_circularity=0.04,
        max_aspect_ratio=6.0,
    ):
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim != 2:
            return None
        try:
            cx, cy = predicted_center_xy
            cx = float(cx)
            cy = float(cy)
            predicted_radius_px = float(predicted_radius_px)
        except Exception:
            return None
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(predicted_radius_px) and predicted_radius_px > 1.0):
            return None

        height, width = mask.shape[:2]
        search_radius = max(12.0, float(search_scale) * predicted_radius_px)
        x0 = int(max(0, np.floor(cx - search_radius)))
        x1 = int(min(width, np.ceil(cx + search_radius + 1)))
        y0 = int(max(0, np.floor(cy - search_radius)))
        y1 = int(min(height, np.ceil(cy + search_radius + 1)))
        if x1 <= x0 or y1 <= y0:
            return None

        roi = mask[y0:y1, x0:x1]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(12.0, np.pi * predicted_radius_px * predicted_radius_px * float(min_area_ratio), 0.08 * self.min_area)
        best = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < float(min_circularity):
                continue

            _x, _y, contour_width, contour_height = cv2.boundingRect(contour)
            aspect = max(contour_width, contour_height) / max(1.0, min(contour_width, contour_height))
            if aspect > float(max_aspect_ratio):
                continue

            full_contour = contour.copy()
            full_contour[:, 0, 0] += x0
            full_contour[:, 0, 1] += y0
            (observed_cx, observed_cy), observed_radius = cv2.minEnclosingCircle(full_contour)
            center_error = float(np.hypot(observed_cx - cx, observed_cy - cy))
            if center_error > search_radius:
                continue

            recovered_center = (
                float(0.78 * cx + 0.22 * observed_cx),
                float(0.78 * cy + 0.22 * observed_cy),
            )
            fill_ratio = area / max(np.pi * predicted_radius_px * predicted_radius_px, 1.0)
            score = area * max(circularity, 0.05) * np.exp(
                -0.5 * (center_error / max(predicted_radius_px, 1.0)) ** 2
            )
            detection = SphereDetection(
                center_xy=recovered_center,
                radius_px=float(predicted_radius_px),
                area=area,
                circularity=circularity,
                fill_ratio=float(fill_ratio),
                aspect=float(aspect),
                contour=full_contour,
            )
            if best is None or score > best[0]:
                best = (score, detection)

        return None if best is None else best[1]

    def detection_mask(self, mask, detection):
        if detection is None:
            return np.zeros_like(mask)
        selected = np.zeros_like(mask)
        cv2.drawContours(selected, [detection.contour], -1, 255, -1)
        return selected


def reconstruct_sphere_from_zed_point_cloud(
    point_cloud,
    mask,
    detection,
    radius_m=0.0,
    depth_min=0.05,
    depth_max=None,
    max_points=6000,
    camera_intrinsics=None,
    confidence_map=None,
    max_confidence=None,
    scene_segmentation=None,
    cached_table_plane=None,
    update_table=True,
    table_max_points=2500,
):
    if detection is None:
        return None

    if scene_segmentation is None:
        scene_segmentation = segment_scene_from_zed_point_cloud(
            point_cloud,
            mask,
            radius_m=radius_m,
            depth_min=depth_min,
            depth_max=depth_max,
            max_ball_points=max_points,
            max_table_points=table_max_points,
            confidence_map=confidence_map,
            max_confidence=max_confidence,
            cached_table_plane=cached_table_plane,
            update_table=update_table,
        )

    points = np.asarray(scene_segmentation.ball_points, dtype=np.float32)
    if len(points) < 12:
        return None

    configured_radius = positive_radius(radius_m)
    working_radius = configured_radius
    radius_source = "configured"

    table_plane = scene_segmentation.table_plane
    if table_plane is not None:
        sphere_points = points

        if working_radius <= 0.0:
            working_radius = estimate_metric_radius_from_stereo(sphere_points, detection, camera_intrinsics)
            if working_radius > 0.0:
                radius_source = "stereo metric"

        if points_are_table_supported(sphere_points, table_plane, radius_m=working_radius):
            table_estimate = estimate_table_supported_sphere(
                sphere_points,
                table_plane,
                radius_m=working_radius,
                radius_source=radius_source,
            )
            if table_estimate is not None:
                return attach_image_measurement(table_estimate, detection)

        free_estimate = estimate_sphere_from_points(
            sphere_points,
            radius_m=working_radius,
            detection=detection,
            camera_intrinsics=camera_intrinsics,
            radius_source=radius_source,
        )
        if free_estimate is not None:
            free_estimate.table_normal = np.asarray(table_plane.normal, dtype=np.float32)
            free_estimate.table_offset = float(table_plane.offset)
            free_estimate.method = f"{free_estimate.method} + segmented table reference"
            return attach_image_measurement(free_estimate, detection)

    if working_radius <= 0.0:
        working_radius = estimate_metric_radius_from_stereo(points, detection, camera_intrinsics)
        if working_radius > 0.0:
            radius_source = "stereo metric"

    estimate = estimate_sphere_from_points(
        points,
        radius_m=working_radius,
        detection=detection,
        camera_intrinsics=camera_intrinsics,
        radius_source=radius_source,
    )
    return attach_image_measurement(estimate, detection)


def masked_point_cloud_points(
    point_cloud,
    mask,
    depth_min=0.05,
    depth_max=None,
    max_points=6000,
    confidence_map=None,
    max_confidence=None,
):
    try:
        point_data = np.asarray(point_cloud.get_data())
    except Exception:
        return np.empty((0, 3), dtype=np.float32)

    if point_data.ndim != 3 or point_data.shape[2] < 3:
        return np.empty((0, 3), dtype=np.float32)

    if mask.shape[:2] != point_data.shape[:2]:
        mask = cv2.resize(
            mask,
            (point_data.shape[1], point_data.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    bounds = nonzero_mask_bounds(mask)
    if bounds is None:
        return np.empty((0, 3), dtype=np.float32)

    x0, y0, x1, y1 = bounds
    mask_roi = mask[y0:y1, x0:x1] > 0
    xyz = point_data[y0:y1, x0:x1, :3][mask_roi].astype(np.float32)
    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    finite = np.all(np.isfinite(xyz), axis=1)
    valid = finite.copy()
    distances = np.linalg.norm(xyz, axis=1)
    if depth_min is not None:
        valid &= distances >= float(depth_min)
    if depth_max is not None:
        valid &= distances <= float(depth_max)
    if confidence_map is not None and max_confidence is not None:
        confidence = confidence_array(confidence_map, point_data.shape[:2])
        if confidence is not None:
            confidence_values = np.asarray(confidence[y0:y1, x0:x1][mask_roi], dtype=np.float32).reshape(-1)
            if len(confidence_values) == len(valid):
                valid &= confidence_values <= float(max_confidence)
    xyz = xyz[valid]

    max_points = max(0, int(max_points))
    if max_points > 0 and len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[indices]

    return np.ascontiguousarray(xyz, dtype=np.float32)


def nonzero_mask_bounds(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        return None
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return int(x), int(y), int(x + width), int(y + height)


def segment_scene_from_zed_point_cloud(
    point_cloud,
    ball_mask,
    radius_m=0.0,
    depth_min=0.05,
    depth_max=None,
    max_ball_points=2000,
    max_table_points=2500,
    confidence_map=None,
    max_confidence=None,
    cached_table_plane=None,
    update_table=True,
):
    raw_ball_points = masked_point_cloud_points(
        point_cloud,
        ball_mask,
        depth_min=depth_min,
        depth_max=depth_max,
        max_points=max_ball_points,
        confidence_map=confidence_map,
        max_confidence=max_confidence,
    )

    table_plane = cached_table_plane
    table_updated = False
    table_points = np.empty((0, 3), dtype=np.float32)
    other_points = np.empty((0, 3), dtype=np.float32)

    if bool(update_table) or table_plane is None:
        candidate_points = table_candidate_point_cloud_points(
            point_cloud,
            ball_mask,
            depth_min=depth_min,
            depth_max=depth_max,
            max_points=max_table_points,
            confidence_map=confidence_map,
            max_confidence=max_confidence,
        )
        estimated = estimate_table_plane(candidate_points, raw_ball_points, radius_m=radius_m)
        if estimated is not None:
            table_plane = estimated
            table_updated = True
        if table_plane is not None and len(candidate_points) > 0:
            table_points, other_points = split_points_by_table(candidate_points, table_plane)

    ball_points = raw_ball_points
    if table_plane is not None and len(raw_ball_points) > 0:
        cleaned = remove_table_background_points(raw_ball_points, table_plane, radius_m=radius_m)
        if len(cleaned) >= 12:
            ball_points = cleaned

    if table_plane is None:
        method = "scene segmentation: ball + unknown table"
    elif table_updated:
        method = "scene segmentation: ball/table/other updated"
    else:
        method = "scene segmentation: ball/table cached"

    return SceneSegmentation(
        ball_points=np.ascontiguousarray(ball_points, dtype=np.float32),
        raw_ball_points=np.ascontiguousarray(raw_ball_points, dtype=np.float32),
        table_points=np.ascontiguousarray(table_points, dtype=np.float32),
        other_points=np.ascontiguousarray(other_points, dtype=np.float32),
        table_plane=table_plane,
        table_updated=bool(table_updated),
        method=method,
    )


def split_points_by_table(points, table_plane, distance_threshold=0.015):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty

    normal = np.asarray(table_plane.normal, dtype=np.float64).reshape(3)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, np.ascontiguousarray(points[:, :3], dtype=np.float32)

    normal = normal / normal_norm
    signed = points[:, :3].astype(np.float64) @ normal + float(table_plane.offset)
    valid = np.isfinite(signed)
    table = valid & (np.abs(signed) <= float(distance_threshold))
    other = valid & ~table
    return (
        np.ascontiguousarray(points[table, :3], dtype=np.float32),
        np.ascontiguousarray(points[other, :3], dtype=np.float32),
    )


def masked_surface_patches(
    point_cloud,
    mask,
    depth_min=0.05,
    depth_max=None,
    max_points=6000,
    confidence_map=None,
    max_confidence=None,
):
    try:
        point_data = np.asarray(point_cloud.get_data())
    except Exception:
        return empty_surface_patches()

    if point_data.ndim != 3 or point_data.shape[2] < 3:
        return empty_surface_patches()

    if mask.shape[:2] != point_data.shape[:2]:
        mask = cv2.resize(
            mask,
            (point_data.shape[1], point_data.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    xyz_grid = point_data[:, :, :3].astype(np.float32)
    selected_pixels = mask > 0
    finite_grid = np.all(np.isfinite(xyz_grid), axis=2)
    selected_pixels &= finite_grid
    distances_grid = np.linalg.norm(xyz_grid, axis=2)
    if depth_min is not None:
        selected_pixels &= distances_grid >= float(depth_min)
    if depth_max is not None:
        selected_pixels &= distances_grid <= float(depth_max)
    confidence = confidence_array(confidence_map, point_data.shape[:2])
    if confidence is not None and max_confidence is not None:
        selected_pixels &= confidence <= float(max_confidence)

    normals_grid, weights_grid = estimate_surface_patch_normals_and_weights(xyz_grid, selected_pixels)
    xyz = xyz_grid[selected_pixels].astype(np.float32)
    if len(xyz) == 0:
        return empty_surface_patches()

    normals = normalize_surface_normals(normals_grid[selected_pixels].astype(np.float32))
    weights = normalize_surface_weights(weights_grid[selected_pixels].astype(np.float32))

    max_points = max(0, int(max_points))
    if max_points > 0 and len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[indices]
        normals = normals[indices]
        weights = weights[indices]

    return SurfacePatchObservation(
        points=np.ascontiguousarray(xyz, dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        weights=np.ascontiguousarray(weights, dtype=np.float32),
    )


def empty_surface_patches():
    return SurfacePatchObservation(
        points=np.empty((0, 3), dtype=np.float32),
        normals=np.empty((0, 3), dtype=np.float32),
        weights=np.empty((0,), dtype=np.float32),
    )


def estimate_surface_patch_normals_and_weights(point_data, selected_pixels, max_neighbor_distance_m=0.08):
    xyz = np.asarray(point_data, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] < 3:
        return (
            np.zeros((*selected_pixels.shape, 3), dtype=np.float32),
            np.zeros(selected_pixels.shape, dtype=np.float32),
        )

    height, width = xyz.shape[:2]
    normals = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    if height < 3 or width < 3:
        return normals, weights

    selected = np.asarray(selected_pixels, dtype=bool)
    center = xyz[1:-1, 1:-1, :3]
    left = xyz[1:-1, :-2, :3]
    right = xyz[1:-1, 2:, :3]
    up = xyz[:-2, 1:-1, :3]
    down = xyz[2:, 1:-1, :3]

    selected_patch = (
        selected[1:-1, 1:-1]
        & selected[1:-1, :-2]
        & selected[1:-1, 2:]
        & selected[:-2, 1:-1]
        & selected[2:, 1:-1]
    )
    finite_patch = (
        np.all(np.isfinite(center), axis=2)
        & np.all(np.isfinite(left), axis=2)
        & np.all(np.isfinite(right), axis=2)
        & np.all(np.isfinite(up), axis=2)
        & np.all(np.isfinite(down), axis=2)
    )

    max_edge = np.maximum.reduce(
        [
            np.linalg.norm(left - center, axis=2),
            np.linalg.norm(right - center, axis=2),
            np.linalg.norm(up - center, axis=2),
            np.linalg.norm(down - center, axis=2),
        ]
    )
    if max_neighbor_distance_m is not None and float(max_neighbor_distance_m) > 0.0:
        nearby = max_edge <= float(max_neighbor_distance_m)
    else:
        nearby = np.ones_like(selected_patch, dtype=bool)

    dx = right - left
    dy = down - up
    patch_normals = np.cross(dx, dy)
    patch_areas = 0.25 * np.linalg.norm(patch_normals, axis=2)
    valid = selected_patch & finite_patch & nearby & np.isfinite(patch_areas) & (patch_areas > 1e-10)
    if not np.any(valid):
        return normals, weights

    patch_normals[valid] /= np.linalg.norm(patch_normals[valid], axis=1)[:, None]
    facing_away = np.sum(patch_normals * center, axis=2) > 0.0
    patch_normals[valid & facing_away] *= -1.0

    valid_areas = patch_areas[valid]
    if len(valid_areas) >= 8:
        low, high = np.percentile(valid_areas, [5.0, 95.0])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            patch_areas = np.clip(patch_areas, low, high)

    inner_normals = normals[1:-1, 1:-1]
    inner_weights = weights[1:-1, 1:-1]
    inner_normals[valid] = patch_normals[valid].astype(np.float32)
    inner_weights[valid] = patch_areas[valid].astype(np.float32)
    return normals, weights


def normalize_surface_normals(normals):
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 2 or normals.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    normals = normals[:, :3].copy()
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-6)
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0
    return np.ascontiguousarray(normals, dtype=np.float32)


def normalize_surface_weights(weights):
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if len(weights) == 0:
        return weights
    finite_positive = np.isfinite(weights) & (weights > 0.0)
    fallback = float(np.median(weights[finite_positive])) if np.any(finite_positive) else 1.0
    weights = weights.copy()
    weights[~finite_positive] = fallback
    return np.ascontiguousarray(weights, dtype=np.float32)


def table_candidate_point_cloud_points(
    point_cloud,
    exclude_mask,
    depth_min=0.05,
    depth_max=None,
    max_points=10000,
    confidence_map=None,
    max_confidence=None,
):
    try:
        point_data = np.asarray(point_cloud.get_data())
    except Exception:
        return np.empty((0, 3), dtype=np.float32)

    if point_data.ndim != 3 or point_data.shape[2] < 3:
        return np.empty((0, 3), dtype=np.float32)

    height, width = point_data.shape[:2]
    mask = np.asarray(exclude_mask, dtype=np.uint8)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    kernel_size = int(np.clip(round(min(height, width) * 0.035), 17, 51))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    excluded = cv2.dilate(mask, kernel, iterations=1) > 0

    max_points = max(0, int(max_points))
    target_pixels = max(20000, max_points * 4)
    stride = max(1, int(np.sqrt(max(height * width / target_pixels, 1.0))))

    sampled_xyz = point_data[::stride, ::stride, :3].reshape(-1, 3).astype(np.float32)
    sampled_excluded = excluded[::stride, ::stride].reshape(-1)
    sampled_confidence = confidence_sample_values(confidence_map, (height, width), stride)

    finite = np.all(np.isfinite(sampled_xyz), axis=1)
    valid = finite & ~sampled_excluded
    distances = np.linalg.norm(sampled_xyz, axis=1)
    if depth_min is not None:
        valid &= distances >= float(depth_min)
    if depth_max is not None:
        valid &= distances <= float(depth_max)
    if sampled_confidence is not None and max_confidence is not None:
        valid &= sampled_confidence <= float(max_confidence)

    points = sampled_xyz[valid]
    if max_points > 0 and len(points) > max_points:
        selected = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[selected]

    return np.ascontiguousarray(points, dtype=np.float32)


def estimate_table_plane(table_points, sphere_points=None, radius_m=0.0, max_iterations=96, distance_threshold=0.012):
    points = np.asarray(table_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return None

    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 80:
        return None

    sphere_points = np.asarray(sphere_points, dtype=np.float64)
    if sphere_points.ndim != 2 or sphere_points.shape[1] < 3:
        sphere_points = np.empty((0, 3), dtype=np.float64)
    else:
        sphere_points = sphere_points[np.all(np.isfinite(sphere_points[:, :3]), axis=1), :3]

    rng = np.random.default_rng(7)
    best = None
    best_score = -np.inf
    min_inliers = max(60, int(0.04 * len(points)))

    for _ in range(int(max_iterations)):
        ids = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            continue
        normal = normal / normal_norm
        offset = -float(np.dot(normal, p0))

        normal, offset, _sphere_heights = orient_plane_toward_sphere(normal, offset, sphere_points)

        distances = np.abs(points @ normal + offset)
        inliers = distances <= float(distance_threshold)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < min_inliers:
            continue

        residual = float(np.median(distances[inliers]))
        verticality = abs(float(normal[1]))
        score = inlier_count - 350.0 * residual + 12.0 * verticality
        if score > best_score:
            best_score = score
            best = (normal, offset, inliers, inlier_count, residual)

    if best is None:
        return None

    normal, offset, inliers, _inlier_count, _residual = best
    refined = fit_plane_svd(points[inliers])
    if refined is not None:
        normal, offset = refined
        normal, offset, _sphere_heights = orient_plane_toward_sphere(normal, offset, sphere_points)
        distances = np.abs(points @ normal + offset)
        inliers = distances <= float(distance_threshold)

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < min_inliers:
        return None

    residual = float(np.median(np.abs(points[inliers] @ normal + offset)))
    return TablePlane(
        normal=normal.astype(np.float64),
        offset=float(offset),
        inlier_count=inlier_count,
        residual_m=residual,
    )


def points_are_table_supported(points, table_plane, radius_m=0.0):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 12:
        return False

    normal = np.asarray(table_plane.normal, dtype=np.float64).reshape(3)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return False
    normal = normal / normal_norm
    heights = points @ normal + float(table_plane.offset)
    heights = heights[np.isfinite(heights)]
    if len(heights) < 12:
        return False

    high = float(np.percentile(heights, 95))
    median = float(np.median(heights))
    if high <= 0.008 or median <= -0.01:
        return False

    known_radius = float(radius_m)
    if known_radius > 0.0:
        return high <= 2.25 * known_radius + 0.015

    lateral_radius = estimate_lateral_radius(points, normal, float(table_plane.offset))
    inferred_radius = max(0.5 * high, lateral_radius, 1e-4)
    return high <= 2.35 * inferred_radius + 0.015


def remove_table_background_points(points, table_plane, radius_m=0.0):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)

    normal = np.asarray(table_plane.normal, dtype=np.float64).reshape(3)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)

    normal = normal / normal_norm
    heights = points[:, :3].astype(np.float64) @ normal + float(table_plane.offset)
    known_radius = float(radius_m)
    min_height = 0.008 if known_radius <= 0.0 else float(np.clip(0.18 * known_radius, 0.006, 0.018))
    keep = np.isfinite(heights) & (heights > min_height)
    if np.count_nonzero(keep) < 12:
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)
    return np.ascontiguousarray(points[keep, :3], dtype=np.float32)


def estimate_table_supported_sphere(points, table_plane, radius_m=0.0, radius_source="configured"):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 12:
        return None

    normal = np.asarray(table_plane.normal, dtype=np.float64).reshape(3)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return None
    normal = normal / normal_norm
    offset = float(table_plane.offset)

    heights = points @ normal + offset
    positive = heights > -0.01
    if np.count_nonzero(positive) < 12:
        return None

    points = points[positive]
    heights = np.maximum(heights[positive], 0.0)
    radius = float(radius_m)
    if radius <= 0.0:
        height_radius = 0.5 * float(np.percentile(heights, 95))
        lateral_radius = estimate_lateral_radius(points, normal, offset)
        radius = max(height_radius, lateral_radius, 1e-4)

    if not np.isfinite(radius) or radius <= 0.0:
        return None

    center = fit_table_supported_center(points, heights, normal, offset, radius)
    if center is None:
        return None

    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    method = (
        f"table-reference center + {radius_source} radius"
        if float(radius_m) > 0.0
        else "table-reference sphere"
    )
    return SphereEstimate3D(
        center_xyz=center.astype(np.float32),
        radius_m=float(radius),
        surface_points=points.astype(np.float32),
        residual_m=float(np.median(residuals)),
        method=method,
        table_normal=normal.astype(np.float32),
        table_offset=float(offset),
    )


def fit_table_supported_center(points, heights, normal, offset, radius):
    origin = -float(offset) * normal
    basis_x, basis_y = plane_basis(normal)
    lateral = np.column_stack([(points - origin) @ basis_x, (points - origin) @ basis_y])
    heights = np.asarray(heights, dtype=np.float64)
    radius = float(radius)

    valid = (heights >= -0.02) & (heights <= 2.25 * radius)
    if np.count_nonzero(valid) < 8:
        valid = np.ones(len(points), dtype=bool)

    h = np.clip(heights[valid], 0.0, 2.0 * radius)
    q = lateral[valid]
    radial_sq = np.maximum(0.0, 2.0 * radius * h - h * h)
    A = np.column_stack([2.0 * q[:, 0], 2.0 * q[:, 1], np.ones(len(q))])
    b = q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] - radial_sq

    try:
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        center_lateral = solution[:2]
    except np.linalg.LinAlgError:
        center_lateral = np.median(q, axis=0)

    if not np.all(np.isfinite(center_lateral)):
        center_lateral = np.median(q, axis=0)
    if not np.all(np.isfinite(center_lateral)):
        return None

    return origin + center_lateral[0] * basis_x + center_lateral[1] * basis_y + radius * normal


def estimate_lateral_radius(points, normal, offset):
    origin = -float(offset) * normal
    basis_x, basis_y = plane_basis(normal)
    lateral = np.column_stack([(points - origin) @ basis_x, (points - origin) @ basis_y])
    center = np.median(lateral, axis=0)
    distances = np.linalg.norm(lateral - center, axis=1)
    if len(distances) == 0:
        return 0.0
    return float(np.percentile(distances, 75))


def estimate_metric_radius_from_stereo(points, detection, camera_intrinsics, min_radius_m=0.005, max_radius_m=1.0):
    if detection is None or camera_intrinsics is None:
        return 0.0

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return 0.0
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 12:
        return 0.0

    try:
        fx = abs(float(camera_intrinsics["fx"]))
        fy = abs(float(camera_intrinsics["fy"]))
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])
    except Exception:
        return 0.0

    radius_px = float(getattr(detection, "radius_px", 0.0))
    if not all(np.isfinite(value) and value > 1e-6 for value in (fx, fy, radius_px)):
        return 0.0

    u, v = detection.center_xy
    ray = np.array([(float(u) - cx) / fx, -(float(v) - cy) / fy, 1.0], dtype=np.float64)
    ray_norm = np.linalg.norm(ray)
    if ray_norm < 1e-8:
        return 0.0
    ray /= ray_norm

    depths = points @ ray
    valid = np.isfinite(depths) & (depths > 1e-4)
    if np.count_nonzero(valid) < 12:
        return 0.0
    points = points[valid]
    depths = depths[valid]

    projected = depths[:, None] * ray
    lateral_distances = np.linalg.norm(points - projected, axis=1)
    lateral_distances = lateral_distances[np.isfinite(lateral_distances)]

    candidates = []
    f_eff = 0.5 * (fx + fy)
    image_radius = radius_px / f_eff
    if np.isfinite(image_radius) and 1e-5 < image_radius < 1.5:
        # Use the closest visible surface depth from stereo, then convert the
        # RGB angular radius into a metric sphere radius.
        front_depth = float(np.percentile(depths, 15))
        angular_scale = image_radius / np.sqrt(1.0 + image_radius * image_radius)
        denom = max(1.0 - angular_scale, 1e-6)
        depth_radius = angular_scale * front_depth / denom
        if min_radius_m <= depth_radius <= max_radius_m:
            candidates.append(float(depth_radius))

    if len(lateral_distances) >= 12:
        lateral_radius = float(np.percentile(lateral_distances, 90))
        if min_radius_m <= lateral_radius <= max_radius_m:
            candidates.append(lateral_radius)

    if len(candidates) == 0:
        return 0.0

    if len(candidates) == 1:
        return float(candidates[0])

    depth_radius, lateral_radius = candidates[0], candidates[1]
    if 0.45 * depth_radius <= lateral_radius <= 1.8 * depth_radius:
        return float(np.median(candidates))
    return float(depth_radius)


def fit_plane_svd(points):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 3:
        return None

    center = np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return None

    normal = normal / normal_norm
    offset = -float(np.dot(normal, center))
    return normal, offset


def orient_plane_toward_sphere(normal, offset, sphere_points):
    if sphere_points is None or len(sphere_points) == 0:
        if normal[1] < 0.0:
            return -normal, -offset, None
        return normal, offset, None

    heights = sphere_points @ normal + offset
    if np.median(heights) < 0.0:
        normal = -normal
        offset = -offset
        heights = -heights
    return normal, offset, heights


def plane_basis(normal):
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(seed, normal))) > 0.92:
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_x = np.cross(seed, normal)
    basis_x /= max(np.linalg.norm(basis_x), 1e-9)
    basis_y = np.cross(normal, basis_x)
    basis_y /= max(np.linalg.norm(basis_y), 1e-9)
    return basis_x, basis_y


def estimate_sphere_from_points(points, radius_m=0.0, detection=None, camera_intrinsics=None, radius_source="configured"):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 12:
        return None

    known_radius = float(radius_m)
    if known_radius > 0.0:
        cap_fit = estimate_known_radius_from_depth_cap(
            points,
            known_radius,
            detection=detection,
            camera_intrinsics=camera_intrinsics,
        )
        if cap_fit is not None:
            if radius_source != "configured":
                cap_fit.method = f"{radius_source} radius depth-cap sphere"
            return cap_fit

    fit = fit_sphere_least_squares(points)
    if fit is not None:
        center, fitted_radius, residual = fit
        if known_radius > 0.0:
            radius = known_radius
            method = f"least-squares center + {radius_source} radius"
        else:
            radius = fitted_radius
            method = "least-squares sphere"

        if np.all(np.isfinite(center)) and np.isfinite(radius) and radius > 0.0:
            if known_radius <= 0.0 or 0.20 * known_radius <= fitted_radius <= 5.0 * known_radius:
                return SphereEstimate3D(
                    center_xyz=center.astype(np.float32),
                    radius_m=float(radius),
                    surface_points=points.astype(np.float32),
                    residual_m=float(residual),
                    method=method,
                )

    fallback = fallback_sphere_estimate(points, known_radius, radius_source=radius_source)
    if fallback is None:
        return None
    return fallback


def fit_sphere_least_squares(points):
    points = np.asarray(points, dtype=np.float64)
    A = np.column_stack([2.0 * points[:, 0], 2.0 * points[:, 1], 2.0 * points[:, 2], np.ones(len(points))])
    b = np.sum(points * points, axis=1)
    try:
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    center = solution[:3]
    radius_sq = float(np.dot(center, center) + solution[3])
    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        return None

    radius = float(np.sqrt(radius_sq))
    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    trimmed = residuals <= np.percentile(residuals, 85)
    if np.count_nonzero(trimmed) >= 12 and np.count_nonzero(trimmed) < len(points):
        return fit_sphere_least_squares(points[trimmed])

    return center, radius, float(np.median(residuals))


def fallback_sphere_estimate(points, known_radius, radius_source="configured"):
    points = np.asarray(points, dtype=np.float64)
    surface_center = np.median(points, axis=0)
    if not np.all(np.isfinite(surface_center)):
        return None

    known_radius = float(known_radius)
    if known_radius > 0.0:
        ray = surface_center / max(np.linalg.norm(surface_center), 1e-9)
        center = surface_center + known_radius * ray
        radius = known_radius
        method = f"median surface + {radius_source} radius"
    else:
        distances = np.linalg.norm(points - surface_center, axis=1)
        radius = float(np.percentile(distances, 75))
        center = surface_center
        method = "median surface estimate"

    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    return SphereEstimate3D(
        center_xyz=center.astype(np.float32),
        radius_m=float(radius),
        surface_points=points.astype(np.float32),
        residual_m=float(np.median(residuals)),
        method=method,
    )


def estimate_known_radius_from_depth_cap(points, radius_m, detection=None, camera_intrinsics=None):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    radius = float(radius_m)
    if len(points) < 12 or radius <= 0.0:
        return None

    distances = np.linalg.norm(points, axis=1)
    valid = np.isfinite(distances) & (distances > 1e-6)
    points = points[valid]
    distances = distances[valid]
    if len(points) < 12:
        return None

    ray_dirs = points / distances[:, None]
    center_candidates = points + radius * ray_dirs
    center = np.median(center_candidates, axis=0)
    center_from_rgb_ray = False

    optimized_center = rgb_ray_known_radius_center_estimate(points, radius, detection, camera_intrinsics)
    if optimized_center is not None:
        optimized_residual = sphere_residual(points, optimized_center, radius)
        candidate_residual = sphere_residual(points, center, radius)
        if optimized_residual <= candidate_residual * 2.0 + 0.012:
            center = optimized_center
            center_from_rgb_ray = True

    if not center_from_rgb_ray:
        ray_center = camera_ray_center_estimate(points, center_candidates, detection, camera_intrinsics)
        if ray_center is not None:
            candidate_residual = sphere_residual(points, center, radius)
            ray_residual = sphere_residual(points, ray_center, radius)
            if ray_residual <= candidate_residual * 1.35 + 0.006:
                center = 0.55 * center + 0.45 * ray_center

    if not np.all(np.isfinite(center)):
        return None

    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    trimmed = residuals <= np.percentile(residuals, 85)
    if np.count_nonzero(trimmed) >= 12 and np.count_nonzero(trimmed) < len(points):
        trimmed_center = np.median(center_candidates[trimmed], axis=0)
        trimmed_from_rgb_ray = False
        trimmed_optimized_center = rgb_ray_known_radius_center_estimate(
            points[trimmed],
            radius,
            detection,
            camera_intrinsics,
        )
        if trimmed_optimized_center is not None:
            optimized_residual = sphere_residual(points[trimmed], trimmed_optimized_center, radius)
            candidate_residual = sphere_residual(points[trimmed], trimmed_center, radius)
            if optimized_residual <= candidate_residual * 2.0 + 0.012:
                trimmed_center = trimmed_optimized_center
                trimmed_from_rgb_ray = True
        if not trimmed_from_rgb_ray:
            trimmed_ray_center = camera_ray_center_estimate(points[trimmed], center_candidates[trimmed], detection, camera_intrinsics)
            if trimmed_ray_center is not None:
                trimmed_center = 0.55 * trimmed_center + 0.45 * trimmed_ray_center
        if np.all(np.isfinite(trimmed_center)):
            center = trimmed_center
            points = points[trimmed]
            residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)

    return SphereEstimate3D(
        center_xyz=center.astype(np.float32),
        radius_m=radius,
        surface_points=points.astype(np.float32),
        residual_m=float(np.median(residuals)),
        method="known-radius depth-cap sphere",
    )


def rgb_ray_known_radius_center_estimate(points, radius, detection, camera_intrinsics):
    ray = detection_center_ray(detection, camera_intrinsics)
    if ray is None:
        return None

    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    if len(points) < 12:
        return None

    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        return None

    depths = points @ ray
    depths = depths[np.isfinite(depths) & (depths > 1e-5)]
    if len(depths) < 12:
        return None

    point_distances = np.linalg.norm(points, axis=1)
    valid_distances = np.isfinite(point_distances) & (point_distances > 1e-6)
    if np.count_nonzero(valid_distances) >= 12:
        point_ray_dirs = points[valid_distances] / point_distances[valid_distances, None]
        old_center_depths = (points[valid_distances] + radius * point_ray_dirs) @ ray
        old_center_depths = old_center_depths[np.isfinite(old_center_depths) & (old_center_depths > 1e-5)]
    else:
        old_center_depths = np.empty((0,), dtype=np.float64)

    depth_candidates = [depths]
    if len(old_center_depths) > 0:
        depth_candidates.append(old_center_depths)
    all_depths = np.concatenate(depth_candidates)

    low = min(
        float(np.percentile(depths, 5.0)) - 2.25 * radius,
        float(np.percentile(all_depths, 5.0)) - 1.25 * radius,
    )
    high = max(
        float(np.percentile(depths, 95.0)) + 2.25 * radius,
        float(np.percentile(all_depths, 95.0)) + 1.25 * radius,
    )
    low = max(low, 1e-4)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    center_depth = optimize_center_depth_along_ray(points, ray, radius, low, high)
    if center_depth is None:
        return None
    center = ray * center_depth
    if not np.all(np.isfinite(center)):
        return None
    return center


def optimize_center_depth_along_ray(points, ray, radius, low, high):
    low = float(low)
    high = float(high)
    best_depth = None
    best_error = np.inf

    for sample_count in (96, 48, 32):
        depths = np.linspace(low, high, int(sample_count), dtype=np.float64)
        centers = depths[:, None] * ray[None, :]
        distances = np.linalg.norm(points[None, :, :] - centers[:, None, :], axis=2)
        residuals = np.abs(distances - float(radius))
        errors = trimmed_row_mean(residuals, trim_fraction=0.75)
        best_index = int(np.argmin(errors))
        if float(errors[best_index]) < best_error:
            best_error = float(errors[best_index])
            best_depth = float(depths[best_index])
        step = max((high - low) / max(int(sample_count) - 1, 1), 1e-4)
        low = max(1e-4, best_depth - 4.0 * step)
        high = best_depth + 4.0 * step

    if best_depth is None or not np.isfinite(best_depth):
        return None
    return best_depth


def trimmed_row_mean(values, trim_fraction=0.75):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.full(values.shape[0], np.inf, dtype=np.float64)
    keep_count = int(np.ceil(float(np.clip(trim_fraction, 0.1, 1.0)) * values.shape[1]))
    keep_count = int(np.clip(keep_count, 1, values.shape[1]))
    if keep_count < values.shape[1]:
        values = np.partition(values, keep_count - 1, axis=1)[:, :keep_count]
    return np.mean(values, axis=1)


def detection_center_ray(detection, camera_intrinsics):
    if detection is None or camera_intrinsics is None:
        return None

    try:
        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])
    except Exception:
        return None

    if not all(np.isfinite(v) and abs(v) > 1e-6 for v in (fx, fy)):
        return None

    u, v = detection.center_xy
    ray = np.array([(float(u) - cx) / fx, -(float(v) - cy) / fy, 1.0], dtype=np.float64)
    ray_norm = np.linalg.norm(ray)
    if ray_norm < 1e-8:
        return None
    return ray / ray_norm


def camera_ray_center_estimate(points, center_candidates, detection, camera_intrinsics):
    ray = detection_center_ray(detection, camera_intrinsics)
    if ray is None:
        return None

    depths = center_candidates @ ray
    depths = depths[np.isfinite(depths)]
    if len(depths) == 0:
        return None

    depth = float(np.median(depths))
    if not np.isfinite(depth) or depth <= 0.0:
        return None
    return ray * depth


def sphere_residual(points, center, radius):
    residuals = np.abs(np.linalg.norm(points - center, axis=1) - float(radius))
    if len(residuals) == 0:
        return np.inf
    return float(np.median(residuals))


def draw_sphere_debug_overlay(bgr, mask, detection, estimate=None):
    detections = [] if detection is None else [detection]
    estimates = [] if estimate is None else [estimate]
    return draw_spheres_debug_overlay(bgr, mask, detections, estimates)


def draw_spheres_debug_overlay(bgr, mask, detections=None, estimates=None):
    output = bgr.copy()
    if mask is not None and np.any(mask):
        pixels = mask > 0
        tint = np.zeros_like(output)
        tint[:, :, 1] = 180
        tint[:, :, 2] = 255
        blended = ((0.62 * output[pixels].astype(np.float32)) + (0.38 * tint[pixels].astype(np.float32)))
        output[pixels] = np.clip(blended, 0, 255).astype(np.uint8)

    detections = [] if detections is None else list(detections)
    estimates = [] if estimates is None else list(estimates)
    colors = [
        (80, 255, 120),
        (40, 190, 255),
        (255, 140, 70),
        (230, 80, 255),
        (180, 150, 255),
        (80, 255, 230),
    ]

    for index, detection in enumerate(detections):
        color = colors[index % len(colors)]
        cv2.drawContours(output, [detection.contour], -1, (80, 180, 255), 2, cv2.LINE_AA)
        center = tuple(np.round(detection.center_xy).astype(int))
        cv2.circle(output, center, int(round(detection.radius_px)), color, 2, cv2.LINE_AA)
        cv2.circle(output, center, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            f"B{index + 1}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    lines = ["Spheres: no detection"]
    if detections:
        lines = [f"Spheres RGB detected: {len(detections)}"]
    if estimates:
        lines.append(f"Tracked 3D balls: {len(estimates)}")
    for index, estimate in enumerate(estimates[:4]):
        c = estimate.center_xyz
        lines.append(
            f"B{index + 1} center ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) m | "
            f"r={estimate.radius_m:.3f} m | pts={len(estimate.surface_points)}"
        )
    if len(estimates) == 1:
        lines.append(f"{estimates[0].method} | residual={estimates[0].residual_m:.3f} m")

    for idx, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (24, 38 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def attach_image_measurement(estimate, detection):
    if estimate is None:
        return None
    if detection is None:
        return estimate
    estimate.image_center_xy = (float(detection.center_xy[0]), float(detection.center_xy[1]))
    estimate.image_radius_px = float(detection.radius_px)
    return estimate


def confidence_values_for_pixels(confidence_map, target_shape, selected_pixels):
    confidence = confidence_array(confidence_map, target_shape)
    if confidence is None:
        return None
    try:
        return np.asarray(confidence[selected_pixels], dtype=np.float32).reshape(-1)
    except Exception:
        return None


def confidence_sample_values(confidence_map, target_shape, stride):
    confidence = confidence_array(confidence_map, target_shape)
    if confidence is None:
        return None
    try:
        return np.asarray(confidence[::stride, ::stride], dtype=np.float32).reshape(-1)
    except Exception:
        return None


def confidence_array(confidence_map, target_shape):
    if confidence_map is None:
        return None
    try:
        data = np.asarray(confidence_map.get_data())
    except Exception:
        try:
            data = np.asarray(confidence_map)
        except Exception:
            return None
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


def _odd_kernel_size(value):
    value = int(max(0, value))
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def positive_radius(radius_m):
    try:
        radius = float(radius_m)
    except Exception:
        return 0.0
    if np.isfinite(radius) and radius > 0.0:
        return radius
    return 0.0
