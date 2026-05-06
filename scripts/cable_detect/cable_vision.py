from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = SCRIPT_DIR / "tools" / "cable_color_profile.json"

MIN_COMPONENT_AREA = 80
KEEP_LARGEST_COMPONENT = False
MAX_COMPONENTS = 0
NODE_COUNT = 60
SMOOTH_ALPHA = 0.45
REINIT_THRESHOLD_PX = 80.0
NODE_XYZ_RADIUS = 6
NODE_XYZ_FALLBACK_RADIUS = 14
MIN_NODE_XYZ_SAMPLES = 3


def load_hsv_ranges(profile_path: Path):
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    hsv_ranges = profile.get("hsv_ranges", [])
    if not hsv_ranges:
        raise ValueError(f"No hsv_ranges found in {profile_path}")

    ranges = []
    for idx, item in enumerate(hsv_ranges):
        try:
            lower = np.array(item["lower"], dtype=np.uint8)
            upper = np.array(item["upper"], dtype=np.uint8)
        except KeyError as exc:
            raise ValueError(f"Invalid HSV range at index {idx}: {item}") from exc

        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError(f"Invalid HSV range shape at index {idx}: {item}")

        ranges.append((lower, upper))

    return ranges


def create_raw_mask(bgr: np.ndarray, hsv_ranges):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    return mask


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
    keep_largest_component: bool = KEEP_LARGEST_COMPONENT,
    max_components: int = MAX_COMPONENTS,
):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask, 0, []

    cleaned = np.zeros_like(mask)
    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append({
            "label": label,
            "area": area,
            "bbox": (x, y, w, h),
        })

    if not components:
        return cleaned, 0, []

    components = sorted(components, key=lambda c: c["area"], reverse=True)
    if keep_largest_component:
        components = components[:1]
    elif max_components > 0:
        components = components[: int(max_components)]

    for component in components:
        cleaned[labels == component["label"]] = 255

    return cleaned, len(components), components


def clean_cable_mask(
    raw_mask: np.ndarray,
    min_area: int = MIN_COMPONENT_AREA,
    keep_largest_component: bool = KEEP_LARGEST_COMPONENT,
    max_components: int = MAX_COMPONENTS,
):
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)

    return remove_small_components(
        mask,
        min_area,
        keep_largest_component,
        max_components,
    )


def skeletonize_mask(mask: np.ndarray, max_iterations: int = 200):
    image = (mask > 0).astype(np.uint8)
    if not np.any(image):
        return np.zeros_like(mask)

    image = np.pad(image, 1, mode="constant", constant_values=0)

    for _ in range(max_iterations):
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


def build_skeleton_graph(skeleton: np.ndarray):
    coords_yx = np.argwhere(skeleton > 0)
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


def bfs_path(neighbors, start_idx: int):
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


def reconstruct_path(parents, start_idx: int, end_idx: int):
    path = [end_idx]
    current = end_idx

    while current != start_idx:
        current = parents[current]
        if current == -1:
            return []
        path.append(current)

    path.reverse()
    return path


def longest_skeleton_path(coords_yx: np.ndarray, neighbors):
    if len(coords_yx) == 0:
        return np.empty((0, 2), dtype=np.float32), 0, 0

    degrees = np.array([len(item) for item in neighbors], dtype=np.int32)
    endpoint_indices = np.flatnonzero(degrees == 1)
    branch_count = int(np.count_nonzero(degrees > 2))

    best_start = None
    best_end = None
    best_distance = -1
    best_parents = None

    if len(endpoint_indices) >= 2:
        for start in endpoint_indices:
            distances, parents = bfs_path(neighbors, int(start))

            for end in endpoint_indices:
                end = int(end)
                if end == start or distances[end] <= best_distance:
                    continue

                best_start = int(start)
                best_end = end
                best_distance = distances[end]
                best_parents = parents
    else:
        first_distances, _ = bfs_path(neighbors, 0)
        best_start = int(np.argmax(first_distances))
        second_distances, best_parents = bfs_path(neighbors, best_start)
        best_end = int(np.argmax(second_distances))

    if best_start is None or best_end is None or best_parents is None:
        return np.empty((0, 2), dtype=np.float32), int(len(endpoint_indices)), branch_count

    path_indices = reconstruct_path(best_parents, best_start, best_end)
    if not path_indices:
        return np.empty((0, 2), dtype=np.float32), int(len(endpoint_indices)), branch_count

    path_yx = coords_yx[path_indices]
    path_xy = path_yx[:, ::-1].astype(np.float32)

    return path_xy, int(len(endpoint_indices)), branch_count


def sample_path_nodes(path_xy: np.ndarray, node_count: int):
    if len(path_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)

    if len(path_xy) == 1 or node_count <= 1:
        return path_xy[:1].copy()

    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))

    if total_length <= 1e-6:
        return path_xy[:1].copy()

    distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    sample_distances = np.linspace(0.0, total_length, node_count)

    sampled_x = np.interp(sample_distances, distances, path_xy[:, 0])
    sampled_y = np.interp(sample_distances, distances, path_xy[:, 1])

    return np.column_stack([sampled_x, sampled_y]).astype(np.float32)


def extract_centerline(mask: np.ndarray, node_count: int):
    skeleton = skeletonize_mask(mask)
    skeleton, _, _ = remove_small_components(skeleton, min_area=8)

    coords_yx, neighbors = build_skeleton_graph(skeleton)
    path_xy, endpoint_count, branch_count = longest_skeleton_path(coords_yx, neighbors)
    nodes_xy = sample_path_nodes(path_xy, node_count)

    return {
        "skeleton": skeleton,
        "path_xy": path_xy,
        "nodes_xy": nodes_xy,
        "raw_nodes_xy": nodes_xy.copy(),
        "endpoint_count": endpoint_count,
        "branch_count": branch_count,
        "mean_node_delta_px": 0.0,
        "reinitialized": False,
        "reversed_order": False,
    }


class NodeSmoother:
    def __init__(self, alpha: float, reinit_threshold_px: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.reinit_threshold_px = float(reinit_threshold_px)
        self.previous_nodes = None

    def update(self, centerline):
        nodes = centerline["nodes_xy"]

        if len(nodes) == 0:
            return centerline

        if self.previous_nodes is None or self.previous_nodes.shape != nodes.shape:
            self.previous_nodes = nodes.copy()
            centerline["raw_nodes_xy"] = nodes.copy()
            centerline["mean_node_delta_px"] = 0.0
            centerline["reinitialized"] = True
            return centerline

        direct_delta = np.linalg.norm(nodes - self.previous_nodes, axis=1)
        reversed_nodes = nodes[::-1].copy()
        reversed_delta = np.linalg.norm(reversed_nodes - self.previous_nodes, axis=1)

        direct_mean = float(np.mean(direct_delta))
        reversed_mean = float(np.mean(reversed_delta))

        if reversed_mean < direct_mean:
            nodes = reversed_nodes
            best_mean = reversed_mean
            reversed_order = True
        else:
            best_mean = direct_mean
            reversed_order = False

        if best_mean > self.reinit_threshold_px:
            self.previous_nodes = nodes.copy()
            centerline["nodes_xy"] = nodes
            centerline["raw_nodes_xy"] = nodes.copy()
            centerline["mean_node_delta_px"] = best_mean
            centerline["reinitialized"] = True
            centerline["reversed_order"] = reversed_order
            return centerline

        smoothed_nodes = (
            self.alpha * nodes + (1.0 - self.alpha) * self.previous_nodes
        ).astype(np.float32)

        self.previous_nodes = smoothed_nodes.copy()
        centerline["nodes_xy"] = smoothed_nodes
        centerline["raw_nodes_xy"] = nodes.copy()
        centerline["mean_node_delta_px"] = best_mean
        centerline["reinitialized"] = False
        centerline["reversed_order"] = reversed_order

        return centerline


def is_valid_xyz(xyz):
    if xyz is None or not np.all(np.isfinite(xyz)):
        return False

    distance = float(np.linalg.norm(xyz))
    return (
        distance > 1e-6
        and distance < 100.0
    )


def xyz_samples_from_masked_window(
    point_cloud: sl.Mat,
    mask: np.ndarray,
    x: int,
    y: int,
    radius: int,
):
    samples = []
    point_cloud_width = point_cloud.get_width()
    point_cloud_height = point_cloud.get_height()
    mask_height, mask_width = mask.shape[:2]

    width = min(point_cloud_width, mask_width)
    height = min(point_cloud_height, mask_height)

    x0 = max(0, int(round(x)) - radius)
    x1 = min(width - 1, int(round(x)) + radius)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height - 1, int(round(y)) + radius)

    local_mask = mask[y0:y1 + 1, x0:x1 + 1] > 0
    if not np.any(local_mask):
        return samples

    local_ys, local_xs = np.nonzero(local_mask)
    xs = local_xs + x0
    ys = local_ys + y0

    order = np.argsort((xs - x) ** 2 + (ys - y) ** 2)

    for sample_idx in order:
        xx = int(xs[sample_idx])
        yy = int(ys[sample_idx])
        err, value = point_cloud.get_value(xx, yy)
        if err != sl.ERROR_CODE.SUCCESS:
            continue

        xyz = np.array([value[0], value[1], value[2]], dtype=np.float32)
        if is_valid_xyz(xyz):
            samples.append(xyz)

    return samples


def robust_xyz_from_masked_point_cloud(
    point_cloud: sl.Mat,
    mask: np.ndarray,
    x: int,
    y: int,
    radius: int,
    fallback_radius: int,
    min_samples: int,
):
    samples = xyz_samples_from_masked_window(point_cloud, mask, x, y, radius)

    if len(samples) < min_samples and fallback_radius > radius:
        samples = xyz_samples_from_masked_window(
            point_cloud,
            mask,
            x,
            y,
            fallback_radius,
        )

    if not samples:
        return None, 0

    sample_array = np.stack(samples, axis=0).astype(np.float32)
    median_xyz = np.median(sample_array, axis=0)

    distances = np.linalg.norm(sample_array - median_xyz, axis=1)
    median_distance = float(np.median(distances))
    inlier_threshold = max(0.015, 3.0 * median_distance)
    inliers = sample_array[distances <= inlier_threshold]

    if len(inliers) >= min_samples:
        sample_array = inliers

    return np.median(sample_array, axis=0).astype(np.float32), len(sample_array)


def extract_node_xyz(
    point_cloud: sl.Mat,
    mask: np.ndarray,
    nodes_xy: np.ndarray,
    radius: int,
    fallback_radius: int,
    min_samples: int,
):
    nodes_xyz = np.full((len(nodes_xy), 3), np.nan, dtype=np.float32)
    valid_nodes = np.zeros(len(nodes_xy), dtype=bool)
    sample_counts = np.zeros(len(nodes_xy), dtype=np.int32)

    for idx, point in enumerate(nodes_xy):
        xyz, sample_count = robust_xyz_from_masked_point_cloud(
            point_cloud,
            mask,
            point[0],
            point[1],
            radius,
            fallback_radius,
            min_samples,
        )
        sample_counts[idx] = sample_count

        if is_valid_xyz(xyz):
            nodes_xyz[idx] = xyz
            valid_nodes[idx] = True

    return nodes_xyz, valid_nodes, sample_counts


def extract_mask_point_cloud(
    point_cloud: sl.Mat,
    mask: np.ndarray,
    max_points: int,
):
    ys, xs = np.nonzero(mask)
    pixel_count = len(xs)

    if pixel_count == 0 or max_points <= 0:
        return np.empty((0, 3), dtype=np.float32), pixel_count

    if pixel_count > max_points:
        selected = np.linspace(0, pixel_count - 1, max_points, dtype=np.int64)
        xs = xs[selected]
        ys = ys[selected]

    points = []
    width = point_cloud.get_width()
    height = point_cloud.get_height()

    for x, y in zip(xs, ys):
        x = int(x)
        y = int(y)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        err, value = point_cloud.get_value(x, y)
        if err != sl.ERROR_CODE.SUCCESS:
            continue

        xyz = np.array([value[0], value[1], value[2]], dtype=np.float32)
        if is_valid_xyz(xyz):
            points.append(xyz)

    if not points:
        return np.empty((0, 3), dtype=np.float32), pixel_count

    return np.stack(points, axis=0).astype(np.float32), pixel_count


def compute_valid_polyline_length(nodes_xyz: np.ndarray, valid_nodes: np.ndarray):
    length_m = 0.0
    valid_segments = 0

    if len(nodes_xyz) < 2:
        return 0.0, 0

    for idx in range(len(nodes_xyz) - 1):
        if not (valid_nodes[idx] and valid_nodes[idx + 1]):
            continue

        length_m += float(np.linalg.norm(nodes_xyz[idx + 1] - nodes_xyz[idx]))
        valid_segments += 1

    return length_m, valid_segments


def attach_3d_observations(
    point_cloud: sl.Mat,
    mask: np.ndarray,
    centerline,
    node_radius: int,
    node_fallback_radius: int,
    min_node_samples: int,
    max_mask_points: int,
):
    nodes_xyz, valid_nodes, sample_counts = extract_node_xyz(
        point_cloud,
        mask,
        centerline["nodes_xy"],
        node_radius,
        node_fallback_radius,
        min_node_samples,
    )
    cable_points_xyz, mask_pixel_count = extract_mask_point_cloud(
        point_cloud,
        mask,
        max_mask_points,
    )
    length_m, valid_segments = compute_valid_polyline_length(nodes_xyz, valid_nodes)

    centerline["nodes_xyz"] = nodes_xyz
    centerline["nodes_valid"] = valid_nodes
    centerline["node_xyz_sample_counts"] = sample_counts

    return {
        "X_t": cable_points_xyz,
        "Y_observed_t": nodes_xyz,
        "valid_nodes": valid_nodes,
        "node_sample_counts": sample_counts,
        "valid_node_count": int(np.count_nonzero(valid_nodes)),
        "mask_pixel_count": int(mask_pixel_count),
        "point_count": int(len(cable_points_xyz)),
        "length_m": float(length_m),
        "valid_segments": int(valid_segments),
    }
