from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    y_axis_up: bool = True
    z_axis_forward: bool = True


@dataclass(frozen=True)
class CrossingProposal2D:
    proposal_id: int
    centroid_xy: np.ndarray
    bbox_xywh: tuple
    extent_xy: np.ndarray
    area_px: int
    mean_probability: float
    uncertainty_px: np.ndarray
    axes_xy: np.ndarray | None = None
    axis_confidence: float = 0.0


@dataclass(frozen=True)
class CrossingProjectionTarget:
    proposal_id: int
    centroid_xy: np.ndarray
    axis_xy: np.ndarray
    confidence: float
    continuation_offset_px: float = 18.0


def extract_crossing_proposals(
    crossing_mask,
    crossing_probability,
    *,
    min_area_px=12,
    max_proposals=4,
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


def estimate_crossing_axes(
    proposals,
    cable_mask,
    *,
    outer_radius_px=36.0,
    min_axis_separation_deg=25.0,
    min_axis_support_px=12,
):
    """Estimate the two unoriented cable axes around each RGB crossing region.

    The central crossing pixels are excluded.  Directions are estimated from
    cable-mask pixels in the surrounding annulus, where the four cable arms
    provide two peaks modulo pi.  Proposals without two supported, separated
    axes remain valid RGB diagnostics but cannot influence the PF.
    """
    proposals = tuple(proposals or ())
    if cable_mask is None:
        return tuple(replace(proposal, axes_xy=None, axis_confidence=0.0) for proposal in proposals)
    mask = np.asarray(cable_mask, dtype=np.uint8)
    if not proposals or mask.ndim != 2 or not np.any(mask):
        return tuple()
    outer_radius = max(float(outer_radius_px), 4.0)
    minimum_separation = np.deg2rad(max(float(min_axis_separation_deg), 1.0))
    minimum_support = max(2, int(min_axis_support_px))
    output = []
    for proposal in proposals:
        center = np.asarray(proposal.centroid_xy, dtype=np.float64)
        inner_radius = max(4.0, 0.60 * float(np.max(proposal.extent_xy)))
        x0 = max(0, int(np.floor(center[0] - outer_radius)))
        x1 = min(mask.shape[1], int(np.ceil(center[0] + outer_radius + 1.0)))
        y0 = max(0, int(np.floor(center[1] - outer_radius)))
        y1 = min(mask.shape[0], int(np.ceil(center[1] + outer_radius + 1.0)))
        yy, xx = np.nonzero(mask[y0:y1, x0:x1] > 0)
        dx = xx.astype(np.float64) + x0 - center[0]
        dy = yy.astype(np.float64) + y0 - center[1]
        radius = np.hypot(dx, dy)
        keep = (radius >= inner_radius) & (radius <= outer_radius)
        if np.count_nonzero(keep) < 2 * minimum_support:
            output.append(replace(proposal, axes_xy=None, axis_confidence=0.0))
            continue
        angles = np.mod(np.arctan2(dy[keep], dx[keep]), np.pi)
        axes, support = _two_axis_histogram(
            angles,
            minimum_separation,
            minimum_support,
        )
        if axes is None:
            output.append(replace(proposal, axes_xy=None, axis_confidence=0.0))
            continue
        separation = _axis_angle_difference(axes[0], axes[1])
        support_confidence = min(1.0, float(min(support)) / float(3 * minimum_support))
        separation_confidence = min(1.0, separation / max(np.deg2rad(45.0), 1e-6))
        confidence = float(np.clip(support_confidence * separation_confidence, 0.0, 1.0))
        axis_vectors = np.asarray(
            ((np.cos(axes[0]), np.sin(axes[0])), (np.cos(axes[1]), np.sin(axes[1]))),
            dtype=np.float32,
        )
        output.append(
            replace(
                proposal,
                axes_xy=np.ascontiguousarray(axis_vectors),
                axis_confidence=confidence,
            )
        )
    return tuple(output)


def assign_crossing_targets(
    proposals,
    cable_endpoint_nodes,
    intrinsics,
    *,
    continuation_offset_px=18.0,
):
    """Build the two latent axis hypotheses for every active cable and crossing.

    The crossing channel observes a crossing and its two local image axes; it
    does not identify which cable owns either axis.  A straight endpoint chord
    is not a valid local-axis label for a curved cable or hitch, and a previous
    PF path would make a wrong assignment self-reinforcing.  Therefore every
    independent PF receives both axes.  The likelihood below takes the best
    axis *per proposal* rather than rewarding both alternatives.
    """

    endpoints = list(cable_endpoint_nodes or ())
    cable_count = len(endpoints)
    targets = [[] for _ in range(cable_count)]
    if cable_count == 0 or intrinsics is None:
        return tuple(tuple(items) for items in targets)

    for proposal in tuple(proposals or ()):
        axes = np.asarray(proposal.axes_xy, dtype=np.float64)
        if axes.shape != (2, 2) or not np.all(np.isfinite(axes)):
            continue
        confidence = float(np.clip(proposal.mean_probability * proposal.axis_confidence, 0.0, 1.0))
        for cable_index, endpoint_nodes in enumerate(endpoints):
            projected_endpoints = project_points(endpoint_nodes, intrinsics)
            if projected_endpoints.shape != (2, 2) or not np.all(np.isfinite(projected_endpoints)):
                continue
            for axis in axes:
                targets[cable_index].append(
                    CrossingProjectionTarget(
                        proposal_id=int(proposal.proposal_id),
                        centroid_xy=np.ascontiguousarray(proposal.centroid_xy, dtype=np.float32),
                        axis_xy=np.ascontiguousarray(axis, dtype=np.float32),
                        confidence=confidence,
                        continuation_offset_px=max(1.0, float(continuation_offset_px)),
                    )
                )
    return tuple(tuple(items) for items in targets)


def particle_crossing_rewards(
    particles,
    targets,
    intrinsics,
    *,
    position_sigma_px,
    angle_sigma_deg,
    continuation_sigma_px,
):
    """Return a two-sided RGB crossing reward for every NumPy particle.

    A valid path must pass the crossing centroid with the assigned tangent and
    remain close to that same observed axis on both sides of the crossing.
    The two probe points prevent a single locally aligned segment from earning
    a high reward before the polyline turns onto the wrong outgoing branch.
    """

    particles = np.asarray(particles, dtype=np.float64)
    targets = tuple(targets or ())
    count = len(particles)
    total = np.zeros(count, dtype=np.float64)
    best_reward = np.full(count, -np.inf, dtype=np.float64)
    best_distance = np.full(count, np.nan, dtype=np.float64)
    best_angle = np.full(count, np.nan, dtype=np.float64)
    best_continuation = np.full(count, np.nan, dtype=np.float64)
    best_closest = np.full((count, 2), np.nan, dtype=np.float64)
    best_target = np.full(count, -1, dtype=np.int64)
    if count == 0 or not targets:
        return total, best_distance, best_angle, best_continuation, best_closest, best_target
    projected = _project_particle_nodes(particles, intrinsics)
    position_sigma = max(float(position_sigma_px), 1e-3)
    angle_sigma = np.deg2rad(max(float(angle_sigma_deg), 1e-3))
    continuation_sigma = max(float(continuation_sigma_px), 1e-3)
    reward_columns = []
    for target_index, target in enumerate(targets):
        distance, angle, continuation, closest = _particle_target_geometry(
            projected,
            target.centroid_xy,
            target.axis_xy,
            target.continuation_offset_px,
        )
        reward = float(target.confidence) * np.exp(
            -0.5 * (distance / position_sigma) ** 2
            -0.5 * (angle / angle_sigma) ** 2
            -0.5 * (continuation / continuation_sigma) ** 2
        )
        reward = np.where(np.isfinite(reward), reward, 0.0)
        reward_columns.append(reward)
        selected = reward > best_reward
        best_reward[selected] = reward[selected]
        best_distance[selected] = distance[selected]
        best_angle[selected] = np.rad2deg(angle[selected])
        best_continuation[selected] = continuation[selected]
        best_closest[selected] = closest[selected]
        best_target[selected] = int(target_index)
    reward_matrix = np.column_stack(reward_columns)
    for proposal_id in dict.fromkeys(target.proposal_id for target in targets):
        indices = [
            index for index, target in enumerate(targets)
            if target.proposal_id == proposal_id
        ]
        total += np.max(reward_matrix[:, indices], axis=1)
    return total, best_distance, best_angle, best_continuation, best_closest, best_target


def particle_crossing_rewards_torch(
    particles_t,
    targets,
    intrinsics,
    *,
    position_sigma_px,
    angle_sigma_deg,
    continuation_sigma_px,
):
    """CUDA/torch equivalent of :func:`particle_crossing_rewards`."""

    import torch

    targets = tuple(targets or ())
    count = len(particles_t)
    total_t = torch.zeros(count, dtype=particles_t.dtype, device=particles_t.device)
    best_reward_t = torch.full_like(total_t, -torch.inf)
    best_distance_t = torch.full_like(total_t, torch.nan)
    best_angle_t = torch.full_like(total_t, torch.nan)
    best_continuation_t = torch.full_like(total_t, torch.nan)
    best_closest_t = torch.full((count, 2), torch.nan, dtype=particles_t.dtype, device=particles_t.device)
    best_target_t = torch.full((count,), -1, dtype=torch.int64, device=particles_t.device)
    if count == 0 or not targets:
        return total_t, best_distance_t, best_angle_t, best_continuation_t, best_closest_t, best_target_t
    projected_t = _project_particle_nodes_torch(particles_t, intrinsics)
    position_sigma = max(float(position_sigma_px), 1e-3)
    angle_sigma = np.deg2rad(max(float(angle_sigma_deg), 1e-3))
    continuation_sigma = max(float(continuation_sigma_px), 1e-3)
    centroids_t = torch.as_tensor(
        np.stack([target.centroid_xy for target in targets]),
        dtype=particles_t.dtype,
        device=particles_t.device,
    )
    axes_t = torch.as_tensor(
        np.stack([target.axis_xy for target in targets]),
        dtype=particles_t.dtype,
        device=particles_t.device,
    )
    confidence_t = torch.as_tensor(
        [target.confidence for target in targets],
        dtype=particles_t.dtype,
        device=particles_t.device,
    )
    continuation_offset_t = torch.as_tensor(
        [target.continuation_offset_px for target in targets],
        dtype=particles_t.dtype,
        device=particles_t.device,
    )
    start_t = projected_t[:, :-1, :]
    direction_t = projected_t[:, 1:, :] - start_t
    norm2_t = torch.sum(direction_t * direction_t, dim=2)
    offset_t = centroids_t[None, :, None, :] - start_t[:, None, :, :]
    parameter_t = torch.sum(offset_t * direction_t[:, None, :, :], dim=3)
    parameter_t = torch.clamp(parameter_t / norm2_t[:, None, :].clamp_min(1e-12), 0.0, 1.0)
    closest_t = start_t[:, None, :, :] + parameter_t[:, :, :, None] * direction_t[:, None, :, :]
    distance2_t = torch.sum((closest_t - centroids_t[None, :, None, :]) ** 2, dim=3)
    valid_t = torch.isfinite(start_t).all(dim=2) & torch.isfinite(direction_t).all(dim=2) & (norm2_t > 1e-12)
    distance2_t = torch.where(valid_t[:, None, :], distance2_t, torch.full_like(distance2_t, torch.inf))
    best_distance2_by_target_t, segment_index_t = torch.min(distance2_t, dim=2)
    gather_xy_t = segment_index_t[:, :, None, None].expand(-1, -1, 1, 2)
    expanded_direction_t = direction_t[:, None, :, :].expand(-1, len(targets), -1, -1)
    selected_direction_t = torch.gather(expanded_direction_t, 2, gather_xy_t)[:, :, 0, :]
    selected_closest_by_target_t = torch.gather(closest_t, 2, gather_xy_t)[:, :, 0, :]
    selected_direction_t = selected_direction_t / torch.linalg.vector_norm(
        selected_direction_t, dim=2, keepdim=True
    ).clamp_min(1e-12)
    alignment_t = torch.abs(torch.sum(selected_direction_t * axes_t[None, :, :], dim=2)).clamp(0.0, 1.0)
    angle_by_target_t = torch.acos(alignment_t)
    distance_by_target_t = torch.sqrt(best_distance2_by_target_t)
    probe_t = centroids_t[:, None, :] + (
        continuation_offset_t[:, None, None]
        * torch.as_tensor((-1.0, 1.0), dtype=particles_t.dtype, device=particles_t.device)[None, :, None]
        * axes_t[:, None, :]
    )
    probe_offset_t = probe_t[None, :, :, None, :] - start_t[:, None, None, :, :]
    probe_parameter_t = torch.sum(
        probe_offset_t * direction_t[:, None, None, :, :],
        dim=4,
    )
    probe_parameter_t = torch.clamp(
        probe_parameter_t / norm2_t[:, None, None, :].clamp_min(1e-12),
        0.0,
        1.0,
    )
    probe_closest_t = (
        start_t[:, None, None, :, :]
        + probe_parameter_t[:, :, :, :, None] * direction_t[:, None, None, :, :]
    )
    probe_distance2_t = torch.sum(
        (probe_closest_t - probe_t[None, :, :, None, :]) ** 2,
        dim=4,
    )
    probe_distance2_t = torch.where(
        valid_t[:, None, None, :],
        probe_distance2_t,
        torch.full_like(probe_distance2_t, torch.inf),
    )
    # The worse side controls the continuation error.  Averaging the two sides
    # lets a particle hide a wrong outgoing branch behind one perfect incoming
    # arm, which is exactly the branch-switching failure this term prevents.
    continuation_by_target_t = torch.sqrt(
        torch.max(torch.min(probe_distance2_t, dim=3).values, dim=2).values
    )
    reward_by_target_t = confidence_t[None, :] * torch.exp(
        -0.5 * (distance_by_target_t / position_sigma) ** 2
        -0.5 * (angle_by_target_t / angle_sigma) ** 2
        -0.5 * (continuation_by_target_t / continuation_sigma) ** 2
    )
    reward_by_target_t = torch.where(
        torch.isfinite(reward_by_target_t),
        reward_by_target_t,
        torch.zeros_like(reward_by_target_t),
    )
    # Axis identity is latent.  Marginalize with a max-mixture approximation:
    # one reward per crossing proposal, never one reward per alternative axis.
    total_t.zero_()
    for proposal_id in dict.fromkeys(target.proposal_id for target in targets):
        indices = [
            index for index, target in enumerate(targets)
            if target.proposal_id == proposal_id
        ]
        total_t.add_(torch.max(reward_by_target_t[:, indices], dim=1).values)
    best_reward_t, best_target_t = torch.max(reward_by_target_t, dim=1)
    row_t = torch.arange(count, device=particles_t.device)
    best_distance_t = distance_by_target_t[row_t, best_target_t]
    best_angle_t = torch.rad2deg(angle_by_target_t[row_t, best_target_t])
    best_continuation_t = continuation_by_target_t[row_t, best_target_t]
    best_closest_t = selected_closest_by_target_t[row_t, best_target_t]
    return total_t, best_distance_t, best_angle_t, best_continuation_t, best_closest_t, best_target_t


def project_points(points_xyz, intrinsics):
    points = np.asarray(points_xyz, dtype=np.float64)
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    depth = points[:, 2] if bool(intrinsics.z_axis_forward) else -points[:, 2]
    valid = np.all(np.isfinite(points), axis=1) & (depth > 1e-6)
    if not np.any(valid):
        return result.astype(np.float32)
    xyz = points[valid]
    valid_depth = depth[valid]
    result[valid, 0] = float(intrinsics.fx) * xyz[:, 0] / valid_depth + float(intrinsics.cx)
    vertical = float(intrinsics.fy) * xyz[:, 1] / valid_depth
    result[valid, 1] = (
        float(intrinsics.cy) - vertical
        if intrinsics.y_axis_up
        else float(intrinsics.cy) + vertical
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def _two_axis_histogram(angles, minimum_separation, minimum_support):
    bin_count = 36
    histogram, _edges = np.histogram(angles, bins=bin_count, range=(0.0, np.pi))
    smooth = (
        np.roll(histogram, -2)
        + 2 * np.roll(histogram, -1)
        + 3 * histogram
        + 2 * np.roll(histogram, 1)
        + np.roll(histogram, 2)
    ).astype(np.float64)
    first_bin = int(np.argmax(smooth))
    bin_angles = (np.arange(bin_count, dtype=np.float64) + 0.5) * np.pi / bin_count
    eligible = np.asarray([
        _axis_angle_difference(angle, bin_angles[first_bin]) >= minimum_separation
        for angle in bin_angles
    ])
    if not np.any(eligible):
        return None, None
    second_scores = np.where(eligible, smooth, -np.inf)
    second_bin = int(np.argmax(second_scores))
    tolerance = max(np.deg2rad(18.0), 1.5 * np.pi / bin_count)
    refined = []
    support = []
    for peak in (bin_angles[first_bin], bin_angles[second_bin]):
        selected = np.asarray([
            _axis_angle_difference(angle, peak) <= tolerance
            for angle in angles
        ])
        count = int(np.count_nonzero(selected))
        if count < minimum_support:
            return None, None
        doubled = 2.0 * angles[selected]
        angle = 0.5 * np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled)))
        refined.append(float(np.mod(angle, np.pi)))
        support.append(count)
    if _axis_angle_difference(refined[0], refined[1]) < minimum_separation:
        return None, None
    return tuple(refined), tuple(support)


def _axis_angle_difference(first, second):
    return float(abs((float(first) - float(second) + 0.5 * np.pi) % np.pi - 0.5 * np.pi))


def _project_particle_nodes(particles, intrinsics):
    result = np.full((*particles.shape[:2], 2), np.nan, dtype=np.float64)
    z = particles[:, :, 2]
    depth = z if bool(intrinsics.z_axis_forward) else -z
    valid = np.all(np.isfinite(particles[:, :, :3]), axis=2) & (depth > 1e-6)
    result[:, :, 0] = np.where(
        valid,
        float(intrinsics.fx) * particles[:, :, 0] / np.maximum(depth, 1e-12) + float(intrinsics.cx),
        np.nan,
    )
    vertical = float(intrinsics.fy) * particles[:, :, 1] / np.maximum(depth, 1e-12)
    result[:, :, 1] = np.where(
        valid,
        float(intrinsics.cy) - vertical if bool(intrinsics.y_axis_up) else float(intrinsics.cy) + vertical,
        np.nan,
    )
    return result


def _project_particle_nodes_torch(particles_t, intrinsics):
    import torch

    z_t = particles_t[:, :, 2]
    depth_t = z_t if bool(intrinsics.z_axis_forward) else -z_t
    valid_t = torch.isfinite(particles_t[:, :, :3]).all(dim=2) & (depth_t > 1e-6)
    u_t = float(intrinsics.fx) * particles_t[:, :, 0] / depth_t.clamp_min(1e-12) + float(intrinsics.cx)
    vertical_t = float(intrinsics.fy) * particles_t[:, :, 1] / depth_t.clamp_min(1e-12)
    v_t = float(intrinsics.cy) - vertical_t if bool(intrinsics.y_axis_up) else float(intrinsics.cy) + vertical_t
    projected_t = torch.stack((u_t, v_t), dim=2)
    return torch.where(valid_t[:, :, None], projected_t, torch.full_like(projected_t, torch.nan))


def _particle_target_geometry(projected, centroid_xy, axis_xy, continuation_offset_px):
    centroid = np.asarray(centroid_xy, dtype=np.float64).reshape(2)
    axis = np.asarray(axis_xy, dtype=np.float64).reshape(2)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    start = projected[:, :-1, :]
    direction = projected[:, 1:, :] - start
    norm2 = np.sum(direction * direction, axis=2)
    parameter = np.sum((centroid[None, None, :] - start) * direction, axis=2)
    parameter = np.clip(parameter / np.maximum(norm2, 1e-12), 0.0, 1.0)
    closest = start + parameter[:, :, None] * direction
    distance2 = np.sum((closest - centroid[None, None, :]) ** 2, axis=2)
    valid = np.all(np.isfinite(start), axis=2) & np.all(np.isfinite(direction), axis=2) & (norm2 > 1e-12)
    distance2[~valid] = np.inf
    segment_index = np.argmin(distance2, axis=1)
    row_index = np.arange(len(projected))
    selected_direction = direction[row_index, segment_index]
    selected_norm = np.linalg.norm(selected_direction, axis=1)
    selected_direction /= np.maximum(selected_norm[:, None], 1e-12)
    alignment = np.clip(np.abs(selected_direction @ axis), 0.0, 1.0)
    angle = np.arccos(alignment)
    distance = np.sqrt(distance2[row_index, segment_index])
    selected_closest = closest[row_index, segment_index]
    probes = centroid[None, :] + (
        np.asarray((-1.0, 1.0), dtype=np.float64)[:, None]
        * max(float(continuation_offset_px), 1.0)
        * axis[None, :]
    )
    probe_offset = probes[None, :, None, :] - start[:, None, :, :]
    probe_parameter = np.sum(
        probe_offset * direction[:, None, :, :],
        axis=3,
    ) / np.maximum(norm2[:, None, :], 1e-12)
    probe_parameter = np.clip(probe_parameter, 0.0, 1.0)
    probe_closest = (
        start[:, None, :, :]
        + probe_parameter[:, :, :, None] * direction[:, None, :, :]
    )
    probe_distance2 = np.sum(
        (probe_closest - probes[None, :, None, :]) ** 2,
        axis=3,
    )
    probe_distance2[~valid[:, None, :].repeat(2, axis=1)] = np.inf
    continuation = np.sqrt(np.max(np.min(probe_distance2, axis=2), axis=1))
    invalid = ~np.isfinite(distance)
    angle[invalid] = np.nan
    continuation[invalid] = np.nan
    selected_closest[invalid] = np.nan
    return distance, angle, continuation, selected_closest
