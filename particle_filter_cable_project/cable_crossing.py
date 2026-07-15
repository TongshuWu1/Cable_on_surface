from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    y_axis_up: bool = True


@dataclass(frozen=True)
class CrossingProposal2D:
    proposal_id: int
    centroid_xy: np.ndarray
    bbox_xywh: tuple
    extent_xy: np.ndarray
    area_px: int
    mean_probability: float
    uncertainty_px: np.ndarray


@dataclass(frozen=True)
class CableContactObservation:
    proposal: CrossingProposal2D
    segment_indices: tuple
    segment_image_points: tuple
    s1_m: float
    s2_m: float
    q1_xyz: np.ndarray
    q2_xyz: np.ndarray
    centerline_distance_m: float
    gap_m: float
    depth_order: str
    covariance_xyz: np.ndarray
    confidence: float
    diameter_calibrated: bool
    verified_contact: bool
    image_association_error_px: float


def extract_crossing_proposals(
    crossing_mask,
    crossing_probability,
    *,
    min_area_px=12,
    max_proposals=8,
):
    """Return connected crossing regions without assigning physical contact."""

    if crossing_mask is None:
        return tuple()
    mask = (np.asarray(crossing_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if mask.ndim != 2 or not np.any(mask):
        return tuple()
    probability = np.asarray(crossing_probability, dtype=np.float32)
    if probability.shape != mask.shape:
        raise ValueError(
            f"Crossing probability shape {probability.shape} does not match mask {mask.shape}."
        )

    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    records = []
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(1, int(min_area_px)):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        roi_labels = labels[y:y + height, x:x + width]
        component = roi_labels == label
        values = probability[y:y + height, x:x + width][component]
        yy, xx = np.nonzero(component)
        if len(values) == 0:
            continue
        coords = np.column_stack((xx.astype(np.float64) + x, yy.astype(np.float64) + y))
        centroid = np.asarray(centroids[label], dtype=np.float64)
        centered = coords - centroid[None, :]
        covariance = (centered.T @ centered) / max(len(coords) - 1, 1)
        records.append(
            CrossingProposal2D(
                proposal_id=int(label),
                centroid_xy=np.ascontiguousarray(centroid, dtype=np.float32),
                bbox_xywh=(x, y, width, height),
                extent_xy=np.asarray((width, height), dtype=np.float32),
                area_px=area,
                mean_probability=float(np.mean(values)),
                uncertainty_px=np.ascontiguousarray(covariance, dtype=np.float32),
            )
        )
    records.sort(key=lambda item: (item.mean_probability, item.area_px), reverse=True)
    return tuple(records[:max(1, int(max_proposals))])


def verify_crossing_proposals(
    proposals,
    cable_nodes,
    intrinsics,
    cable_diameters_m,
    *,
    contact_tolerance_m=0.002,
    association_sigma_px=18.0,
):
    """Associate RGB proposals with both PF chains and evaluate 3D contact."""

    proposals = tuple(proposals or ())
    chains = list(cable_nodes or [])
    if len(chains) != 2 or intrinsics is None:
        return tuple()
    chains = [_valid_polyline(chain) for chain in chains]
    if any(len(chain) < 2 for chain in chains):
        return tuple()
    projected = [project_points(chain, intrinsics) for chain in chains]
    if any(np.count_nonzero(np.all(np.isfinite(points), axis=1)) < 2 for points in projected):
        return tuple()

    diameters = np.asarray(cable_diameters_m, dtype=np.float64).reshape(-1)
    if len(diameters) != 2:
        raise ValueError("cable_diameters_m must contain exactly two values.")
    calibrated = bool(np.all(np.isfinite(diameters)) and np.all(diameters > 0.0))
    radii = 0.5 * np.maximum(diameters, 0.0)
    arc_prefix = [_arc_length_prefix(chain) for chain in chains]
    observations = []
    sigma_px = max(float(association_sigma_px), 1e-3)

    for proposal in proposals:
        association = _best_segment_pair(projected[0], projected[1], proposal.centroid_xy)
        if association is None:
            continue
        index1, index2, parameter1_image, parameter2_image, image_error = association
        q1, q2, parameter1, parameter2 = closest_points_on_segments(
            chains[0][index1],
            chains[0][index1 + 1],
            chains[1][index2],
            chains[1][index2 + 1],
        )
        distance = float(np.linalg.norm(q1 - q2))
        gap = distance - float(radii[0] + radii[1]) if calibrated else distance
        s1 = float(arc_prefix[0][index1] + parameter1 * np.linalg.norm(chains[0][index1 + 1] - chains[0][index1]))
        s2 = float(arc_prefix[1][index2] + parameter2 * np.linalg.norm(chains[1][index2 + 1] - chains[1][index2]))
        if abs(float(q1[2] - q2[2])) <= max(float(contact_tolerance_m), 1e-6):
            depth_order = "same depth"
        elif q1[2] < q2[2]:
            depth_order = "PF1 above PF2"
        else:
            depth_order = "PF2 above PF1"

        mean_depth = max(0.5 * float(q1[2] + q2[2]), 1e-4)
        proposal_sigma_px = float(np.sqrt(max(np.trace(proposal.uncertainty_px), 1e-6)))
        lateral_sigma = mean_depth * proposal_sigma_px / max(0.5 * (intrinsics.fx + intrinsics.fy), 1e-6)
        depth_sigma = max(2.0 * lateral_sigma, 1e-4)
        covariance = np.diag((lateral_sigma ** 2, lateral_sigma ** 2, depth_sigma ** 2)).astype(np.float32)
        association_confidence = np.exp(-0.5 * (float(image_error) / sigma_px) ** 2)
        confidence = float(np.clip(proposal.mean_probability * association_confidence, 0.0, 1.0))
        verified = bool(calibrated and abs(gap) <= max(float(contact_tolerance_m), 0.0))
        segment_image_points = (
            np.ascontiguousarray(projected[0][index1:index1 + 2], dtype=np.float32),
            np.ascontiguousarray(projected[1][index2:index2 + 2], dtype=np.float32),
        )
        observations.append(
            CableContactObservation(
                proposal=proposal,
                segment_indices=(int(index1), int(index2)),
                segment_image_points=segment_image_points,
                s1_m=s1,
                s2_m=s2,
                q1_xyz=np.ascontiguousarray(q1, dtype=np.float32),
                q2_xyz=np.ascontiguousarray(q2, dtype=np.float32),
                centerline_distance_m=distance,
                gap_m=float(gap),
                depth_order=depth_order,
                covariance_xyz=covariance,
                confidence=confidence,
                diameter_calibrated=calibrated,
                verified_contact=verified,
                image_association_error_px=float(image_error),
            )
        )
    return tuple(observations)


def project_points(points_xyz, intrinsics):
    points = np.asarray(points_xyz, dtype=np.float64)
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 1e-6)
    if not np.any(valid):
        return result.astype(np.float32)
    xyz = points[valid]
    result[valid, 0] = float(intrinsics.fx) * xyz[:, 0] / xyz[:, 2] + float(intrinsics.cx)
    vertical = float(intrinsics.fy) * xyz[:, 1] / xyz[:, 2]
    result[valid, 1] = float(intrinsics.cy) - vertical if intrinsics.y_axis_up else float(intrinsics.cy) + vertical
    return np.ascontiguousarray(result, dtype=np.float32)


def closest_points_on_segments(p1, p2, q1, q2):
    """Return closest points and segment parameters for two finite 3D segments."""

    p1, p2, q1, q2 = [np.asarray(value, dtype=np.float64) for value in (p1, p2, q1, q2)]
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    small = 1e-12
    if a <= small and c <= small:
        return p1, q1, 0.0, 0.0
    if a <= small:
        s, t = 0.0, np.clip(e / max(c, small), 0.0, 1.0)
    elif c <= small:
        s, t = np.clip(-d / max(a, small), 0.0, 1.0), 0.0
    else:
        s = np.clip((b * e - c * d) / denominator, 0.0, 1.0) if abs(denominator) > small else 0.0
        t = (b * s + e) / c
        if t < 0.0:
            t = 0.0
            s = np.clip(-d / a, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = np.clip((b - d) / a, 0.0, 1.0)
    return p1 + s * u, q1 + t * v, float(s), float(t)


def _best_segment_pair(projected1, projected2, centroid):
    centroid = np.asarray(centroid, dtype=np.float64)
    best = None
    for index1 in range(len(projected1) - 1):
        p0, p1 = projected1[index1:index1 + 2]
        if not np.all(np.isfinite((p0, p1))):
            continue
        d1, t1 = _point_segment_distance(centroid, p0, p1)
        for index2 in range(len(projected2) - 1):
            q0, q1 = projected2[index2:index2 + 2]
            if not np.all(np.isfinite((q0, q1))):
                continue
            d2, t2 = _point_segment_distance(centroid, q0, q1)
            segment_distance = _segment_segment_distance_2d(p0, p1, q0, q1)
            score = float(np.hypot(d1, d2) + 0.25 * segment_distance)
            candidate = (score, index1, index2, t1, t2)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    return int(best[1]), int(best[2]), float(best[3]), float(best[4]), float(best[0])


def _point_segment_distance(point, start, end):
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm2 = float(np.dot(direction, direction))
    parameter = 0.0 if norm2 <= 1e-12 else float(np.clip(np.dot(point - start, direction) / norm2, 0.0, 1.0))
    closest = np.asarray(start, dtype=np.float64) + parameter * direction
    return float(np.linalg.norm(point - closest)), parameter


def _segment_segment_distance_2d(p0, p1, q0, q1):
    if _segments_intersect_2d(p0, p1, q0, q1):
        return 0.0
    return min(
        _point_segment_distance(p0, q0, q1)[0],
        _point_segment_distance(p1, q0, q1)[0],
        _point_segment_distance(q0, p0, p1)[0],
        _point_segment_distance(q1, p0, p1)[0],
    )


def _segments_intersect_2d(p0, p1, q0, q1, epsilon=1e-8):
    p0, p1, q0, q1 = [np.asarray(value, dtype=np.float64) for value in (p0, p1, q0, q1)]

    def cross(a, b, c):
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def inside(a, b, point):
        return bool(
            min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
            and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
        )

    c1, c2 = cross(p0, p1, q0), cross(p0, p1, q1)
    c3, c4 = cross(q0, q1, p0), cross(q0, q1, p1)
    if ((c1 > epsilon and c2 < -epsilon) or (c1 < -epsilon and c2 > epsilon)) and (
        (c3 > epsilon and c4 < -epsilon) or (c3 < -epsilon and c4 > epsilon)
    ):
        return True
    return bool(
        (abs(c1) <= epsilon and inside(p0, p1, q0))
        or (abs(c2) <= epsilon and inside(p0, p1, q1))
        or (abs(c3) <= epsilon and inside(q0, q1, p0))
        or (abs(c4) <= epsilon and inside(q0, q1, p1))
    )


def _arc_length_prefix(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def _valid_polyline(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    points = points[:, :3]
    return points if np.all(np.isfinite(points)) else np.empty((0, 3), dtype=np.float64)
