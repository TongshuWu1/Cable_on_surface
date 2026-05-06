from dataclasses import dataclass

import numpy as np


@dataclass
class TrackDLOParams:
    beta: float = 0.35
    lam: float = 50000.0
    alpha: float = 3.0
    mu: float = 0.1
    max_iter: int = 50
    tol: float = 0.0002
    k_vis: float = 50.0
    d_vis: float = 0.06
    visibility_threshold: float = 0.008
    dlo_pixel_width: int = 5
    cable_diameter_m: float = 0.005
    fixed_cable_length_m: float = 0.0
    enforce_cable_length: bool = True
    min_projected_dlo_width_px: float = 3.0
    max_projected_dlo_width_px: float = 18.0
    self_occlusion_depth_margin: float = 0.02
    self_occlusion_max_projection_error: float = 18.0
    beta_pre_proc: float = 3.0
    lambda_pre_proc: float = 1.0
    prune_distance: float = 0.1
    temporal_prediction: bool = True
    prediction_gain: float = 0.75
    prediction_velocity_alpha: float = 0.5
    prediction_velocity_decay: float = 0.85
    max_prediction_step_m: float = 0.035
    crossing_lock: bool = True
    crossing_lock_distance_m: float = 0.025
    crossing_lock_projection_px: float = 12.0
    crossing_lock_depth_margin_m: float = 0.06
    crossing_lock_frames: int = 10
    crossing_lock_window_nodes: int = 2
    crossing_lock_prior_weight: float = 4.0
    crossing_lock_min_edge_gap: int = 4
    crossing_lock_max_pairs: int = 6
    min_init_valid_ratio: float = 0.9
    init_require_endpoints: bool = True
    init_stable_frames: int = 12
    init_length_std_m: float = 0.015
    min_visible_nodes: int = 3
    sigma2_min: float = 1e-8


class TrackDLOTracker:
    def __init__(self, node_count: int, params: TrackDLOParams | None = None):
        self.node_count = int(node_count)
        self.params = params or TrackDLOParams()
        self.Y = None
        self.sigma2 = 0.0
        self.geodesic_coord = None
        self.rest_geodesic_coord = None
        self.node_velocity = None
        self.last_prediction_step_m = 0.0
        self.crossing_locks = {}
        self.last_crossing_pairs = []
        self.last_visible_nodes = []
        self.last_visible_nodes_extended = []
        self.last_self_occluded_nodes = []
        self.last_projection_convention = "unavailable"
        self.last_correspondence_priors = []
        self.last_visibility_state = "uninitialized"
        self.init_candidate_nodes = []
        self.init_candidate_lengths = []
        self.initialized = False
        self.status = "uninitialized"

    def initialize_from_observation(self, nodes_xyz, valid_nodes):
        nodes_xyz = np.asarray(nodes_xyz, dtype=np.float64)
        valid_nodes = np.asarray(valid_nodes, dtype=bool)

        if len(nodes_xyz) != self.node_count:
            self.status = "init skipped: node count mismatch"
            return False

        valid_ratio = float(np.count_nonzero(valid_nodes)) / max(len(valid_nodes), 1)
        if valid_ratio < self.params.min_init_valid_ratio:
            self.status = f"init waiting: valid nodes {valid_ratio:.0%}"
            self.reset_initialization_candidates()
            return False

        if self.params.init_require_endpoints and not (valid_nodes[0] and valid_nodes[-1]):
            self.status = "init waiting: both cable endpoints must be visible"
            self.reset_initialization_candidates()
            return False

        filled = fill_missing_nodes(nodes_xyz, valid_nodes)
        if filled is None:
            self.status = "init skipped: could not fill missing nodes"
            self.reset_initialization_candidates()
            return False

        candidate_length, _valid_segments = compute_polyline_length(
            filled,
            np.ones(len(filled), dtype=bool),
        )
        filled = self.stabilize_initialization_candidate(filled, candidate_length)
        if filled is None:
            return False

        self.Y = filled.astype(np.float64)
        self.rest_geodesic_coord = make_rest_geodesic_coord(
            self.Y,
            self.params.fixed_cable_length_m,
        )
        if self.params.enforce_cable_length:
            self.Y = enforce_rest_geodesic_spacing(self.Y, self.rest_geodesic_coord)
        self.geodesic_coord = self.rest_geodesic_coord.copy()
        self.last_visible_nodes = [int(idx) for idx in np.flatnonzero(valid_nodes)]
        self.last_visible_nodes_extended = extend_visible_nodes(
            self.last_visible_nodes,
            self.geodesic_coord,
            self.params.d_vis,
        )
        self.last_self_occluded_nodes = []
        self.last_projection_convention = "initialized"
        self.last_visibility_state = "initialized"
        self.sigma2 = 0.0
        self.node_velocity = np.zeros_like(self.Y, dtype=np.float64)
        self.last_prediction_step_m = 0.0
        self.crossing_locks = {}
        self.last_crossing_pairs = []
        self.initialized = True
        self.init_candidate_nodes = []
        self.init_candidate_lengths = []
        self.status = f"initialized: learned length {rest_length_from_geodesic(self.rest_geodesic_coord):.3f}m"
        return True

    def reset_initialization_candidates(self):
        self.init_candidate_nodes = []
        self.init_candidate_lengths = []

    def stabilize_initialization_candidate(self, candidate_nodes, candidate_length):
        stable_frames = max(1, int(self.params.init_stable_frames))
        length_std_limit = max(0.0, float(self.params.init_length_std_m))

        candidate_nodes = np.asarray(candidate_nodes, dtype=np.float64)
        if self.init_candidate_nodes:
            reference = self.init_candidate_nodes[0]
            direct_error = mean_finite_node_distance(candidate_nodes, reference)
            reversed_error = mean_finite_node_distance(candidate_nodes[::-1], reference)
            if reversed_error < direct_error:
                candidate_nodes = candidate_nodes[::-1].copy()

        self.init_candidate_nodes.append(candidate_nodes.copy())
        self.init_candidate_lengths.append(float(candidate_length))
        if len(self.init_candidate_nodes) > stable_frames:
            self.init_candidate_nodes = self.init_candidate_nodes[-stable_frames:]
            self.init_candidate_lengths = self.init_candidate_lengths[-stable_frames:]

        if len(self.init_candidate_nodes) < stable_frames:
            self.status = (
                f"init calibrating: stable full-cable frames "
                f"{len(self.init_candidate_nodes)}/{stable_frames}, length {candidate_length:.3f}m"
            )
            return None

        length_std = float(np.std(self.init_candidate_lengths))
        if length_std > length_std_limit:
            self.status = (
                f"init waiting: length stabilizing std {length_std * 1000.0:.0f}mm/"
                f"{length_std_limit * 1000.0:.0f}mm"
            )
            return None

        return np.median(np.stack(self.init_candidate_nodes, axis=0), axis=0)

    def step(
        self,
        X_t,
        observed_nodes_xyz=None,
        observed_valid_nodes=None,
        observed_nodes_xy=None,
        camera_intrinsics=None,
        image_shape=None,
    ):
        X_t = finite_points(np.asarray(X_t, dtype=np.float64))

        if not self.initialized:
            if observed_nodes_xyz is None or observed_valid_nodes is None:
                return self._result(X_t, None, np.zeros(self.node_count, dtype=bool))
            self.initialize_from_observation(observed_nodes_xyz, observed_valid_nodes)
            return self._result(X_t, observed_nodes_xyz, observed_valid_nodes)

        if len(X_t) < 5:
            self.status = "tracking skipped: too few X_t points"
            self.decay_node_velocity()
            return self._result(X_t, observed_nodes_xyz, observed_valid_nodes)

        previous_Y = self.Y.copy()
        Y_prior, prediction_step_m = self.predict_next_nodes()
        self.last_prediction_step_m = prediction_step_m

        visible_nodes_initial = estimate_visible_nodes(
            Y_prior,
            X_t,
            self.params.visibility_threshold,
        )
        visible_nodes, self_occluded_nodes, projection_convention = prune_self_occluded_visible_nodes(
            Y_prior,
            visible_nodes_initial,
            camera_intrinsics,
            image_shape,
            self.params.dlo_pixel_width,
            self.params.cable_diameter_m,
            self.params.min_projected_dlo_width_px,
            self.params.max_projected_dlo_width_px,
            self.params.self_occlusion_depth_margin,
            self.params.self_occlusion_max_projection_error,
            observed_nodes_xyz,
            observed_valid_nodes,
            observed_nodes_xy,
        )
        visible_nodes_extended = extend_visible_nodes(
            visible_nodes,
            self.geodesic_coord,
            self.params.d_vis,
        )
        crossing_pairs = self.update_crossing_locks(
            Y_prior,
            camera_intrinsics,
            image_shape,
            observed_nodes_xyz,
            observed_valid_nodes,
            observed_nodes_xy,
        )
        visibility_state = classify_visibility_state(
            visible_nodes,
            visible_nodes_extended,
            self.node_count,
        )

        self.last_visible_nodes = visible_nodes
        self.last_visible_nodes_extended = visible_nodes_extended
        self.last_self_occluded_nodes = self_occluded_nodes
        self.last_projection_convention = projection_convention
        self.last_visibility_state = visibility_state

        if len(visible_nodes_extended) < self.params.min_visible_nodes:
            self.status = (
                f"tracking held: {visibility_state}, visible nodes {len(visible_nodes_extended)}, "
                f"self-occ {len(self_occluded_nodes)}, cross {len(crossing_pairs)}, "
                f"pred {prediction_step_m * 1000.0:.0f}mm"
            )
            self.decay_node_velocity()
            return self._result(X_t, observed_nodes_xyz, observed_valid_nodes)

        guide_nodes = Y_prior[visible_nodes_extended].copy()
        sigma2_pre = self.sigma2
        guide_nodes, sigma2_pre, _ = cpd_mct_registration(
            X_t,
            guide_nodes,
            sigma2_pre,
            beta=self.params.beta_pre_proc,
            lam=self.params.lambda_pre_proc,
            mu=self.params.mu,
            max_iter=self.params.max_iter,
            tol=self.params.tol,
            prune_distance=self.params.prune_distance,
            sigma2_min=self.params.sigma2_min,
        )

        correspondence_priors, visibility_state = build_trackdlo_length_priors(
            previous_Y,
            guide_nodes,
            visible_nodes,
            visible_nodes_extended,
            self.geodesic_coord,
        )
        crossing_priors = self.build_crossing_lock_priors(Y_prior)
        correspondence_priors = correspondence_priors + crossing_priors
        self.last_correspondence_priors = correspondence_priors
        self.last_visibility_state = visibility_state

        self.Y, self.sigma2, converged = cpd_mct_registration(
            X_t,
            Y_prior,
            self.sigma2,
            beta=self.params.beta,
            lam=self.params.lam,
            mu=self.params.mu,
            max_iter=self.params.max_iter,
            tol=self.params.tol,
            alpha=self.params.alpha,
            correspondence_priors=correspondence_priors,
            visible_nodes=visible_nodes_extended,
            k_vis=self.params.k_vis,
            visibility_threshold=self.params.visibility_threshold,
            prune_distance=self.params.prune_distance,
            sigma2_min=self.params.sigma2_min,
        )
        if self.params.enforce_cable_length and self.rest_geodesic_coord is not None:
            self.Y = enforce_rest_geodesic_spacing(self.Y, self.rest_geodesic_coord)
            self.geodesic_coord = self.rest_geodesic_coord.copy()
        else:
            self.geodesic_coord = cumulative_node_distances(self.Y)

        self.update_node_velocity(previous_Y, self.Y)
        current_length, _valid_segments = compute_polyline_length(
            self.Y,
            np.ones(self.node_count, dtype=bool),
        )
        rest_length = rest_length_from_geodesic(self.rest_geodesic_coord)

        self.status = (
            f"trackdlo {'converged' if converged else 'max iter'}: "
            f"{visibility_state}, "
            f"visible {len(visible_nodes)}/{self.node_count}, "
            f"extended {len(visible_nodes_extended)}, "
            f"self-occ {len(self_occluded_nodes)}, "
            f"cross {len(crossing_pairs)}, "
            f"pred {prediction_step_m * 1000.0:.0f}mm, "
            f"length {current_length:.3f}/{rest_length:.3f}m"
        )
        return self._result(X_t, observed_nodes_xyz, observed_valid_nodes)

    def predict_next_nodes(self):
        if (
            not self.params.temporal_prediction
            or self.node_velocity is None
            or self.Y is None
            or self.node_velocity.shape != self.Y.shape
        ):
            return self.Y.copy(), 0.0

        predicted_delta = float(self.params.prediction_gain) * self.node_velocity
        predicted_delta = clamp_node_displacements(
            predicted_delta,
            self.params.max_prediction_step_m,
        )
        predicted = self.Y + predicted_delta
        if self.params.enforce_cable_length and self.rest_geodesic_coord is not None:
            predicted = enforce_rest_geodesic_spacing(predicted, self.rest_geodesic_coord)
        prediction_step_m = max_node_step(predicted - self.Y)
        return predicted.astype(np.float64), prediction_step_m

    def update_node_velocity(self, previous_nodes, current_nodes):
        previous_nodes = np.asarray(previous_nodes, dtype=np.float64)
        current_nodes = np.asarray(current_nodes, dtype=np.float64)
        if previous_nodes.shape != current_nodes.shape:
            return

        measured_velocity = current_nodes - previous_nodes
        measured_velocity = clamp_node_displacements(
            measured_velocity,
            max(1e-6, 2.0 * float(self.params.max_prediction_step_m)),
        )
        if self.node_velocity is None or self.node_velocity.shape != measured_velocity.shape:
            self.node_velocity = np.zeros_like(measured_velocity, dtype=np.float64)

        alpha = float(np.clip(self.params.prediction_velocity_alpha, 0.0, 1.0))
        self.node_velocity = (
            (1.0 - alpha) * self.node_velocity
            + alpha * measured_velocity
        )

    def decay_node_velocity(self):
        if self.node_velocity is None:
            return
        decay = float(np.clip(self.params.prediction_velocity_decay, 0.0, 1.0))
        self.node_velocity *= decay

    def update_crossing_locks(
        self,
        Y_prior,
        camera_intrinsics,
        image_shape,
        observed_nodes_xyz,
        observed_valid_nodes,
        observed_nodes_xy,
    ):
        if not self.params.crossing_lock:
            self.crossing_locks = {}
            self.last_crossing_pairs = []
            return []

        for key in list(self.crossing_locks):
            self.crossing_locks[key]["ttl"] -= 1
            if self.crossing_locks[key]["ttl"] <= 0:
                del self.crossing_locks[key]

        detected = detect_crossing_segment_pairs(
            Y_prior,
            camera_intrinsics,
            image_shape,
            observed_nodes_xyz,
            observed_valid_nodes,
            observed_nodes_xy,
            self.params.crossing_lock_distance_m,
            self.params.crossing_lock_projection_px,
            self.params.crossing_lock_depth_margin_m,
            self.params.crossing_lock_min_edge_gap,
            self.params.crossing_lock_max_pairs,
        )
        for pair in detected:
            key = tuple(sorted((int(pair["edge_a"]), int(pair["edge_b"]))))
            self.crossing_locks[key] = {
                "ttl": max(1, int(self.params.crossing_lock_frames)),
                "distance_m": float(pair["distance_m"]),
                "projection_px": float(pair["projection_px"]),
            }

        self.last_crossing_pairs = [
            {
                "edge_a": int(key[0]),
                "edge_b": int(key[1]),
                "ttl": int(item["ttl"]),
                "distance_m": float(item["distance_m"]),
                "projection_px": float(item["projection_px"]),
            }
            for key, item in sorted(self.crossing_locks.items())
        ]
        return self.last_crossing_pairs

    def build_crossing_lock_priors(self, Y_prior):
        if not self.crossing_locks:
            return []

        Y_prior = np.asarray(Y_prior, dtype=np.float64)
        window = max(0, int(self.params.crossing_lock_window_nodes))
        weight = max(0.0, float(self.params.crossing_lock_prior_weight))
        if weight <= 0.0:
            return []

        priors = []
        for edge_a, edge_b in self.crossing_locks:
            for edge_idx in (edge_a, edge_b):
                start = max(0, int(edge_idx) - window)
                stop = min(len(Y_prior), int(edge_idx) + 2 + window)
                for node_idx in range(start, stop):
                    xyz = Y_prior[node_idx]
                    if np.all(np.isfinite(xyz)):
                        priors.append((int(node_idx), xyz.copy(), weight))

        return priors

    def _result(self, X_t, observed_nodes_xyz, observed_valid_nodes):
        if self.Y is None:
            Y_t = observed_nodes_xyz
            valid = observed_valid_nodes
            if Y_t is None:
                Y_t = np.full((self.node_count, 3), np.nan, dtype=np.float64)
                valid = np.zeros(self.node_count, dtype=bool)
        else:
            Y_t = self.Y
            valid = np.ones(self.node_count, dtype=bool)

        valid = np.asarray(valid, dtype=bool)
        visible_mask = index_mask(self.last_visible_nodes, self.node_count)
        extended_visible_mask = index_mask(self.last_visible_nodes_extended, self.node_count)
        self_occluded_mask = index_mask(self.last_self_occluded_nodes, self.node_count)
        if not self.initialized:
            visible_mask = valid.copy()
            extended_visible_mask = valid.copy()
            occluded_mask = np.zeros(self.node_count, dtype=bool)
            self_occluded_mask = np.zeros(self.node_count, dtype=bool)
        else:
            occluded_mask = valid & ~extended_visible_mask

        length_m, valid_segments = compute_polyline_length(Y_t, valid)
        rest_length_m = rest_length_from_geodesic(self.rest_geodesic_coord)
        return {
            "Y_t": np.asarray(Y_t, dtype=np.float32),
            "Y_t_valid": np.asarray(valid, dtype=bool),
            "tracked_length_m": float(length_m),
            "rest_length_m": float(rest_length_m),
            "length_error_m": float(length_m - rest_length_m),
            "cable_diameter_m": float(self.params.cable_diameter_m),
            "tracked_valid_segments": int(valid_segments),
            "tracker_initialized": bool(self.initialized),
            "tracker_status": self.status,
            "visible_nodes": list(self.last_visible_nodes),
            "visible_nodes_extended": list(self.last_visible_nodes_extended),
            "visible_node_mask": visible_mask,
            "extended_visible_node_mask": extended_visible_mask,
            "occluded_node_mask": occluded_mask,
            "self_occluded_nodes": list(self.last_self_occluded_nodes),
            "self_occluded_node_mask": self_occluded_mask,
            "projection_convention": self.last_projection_convention,
            "visibility_state": self.last_visibility_state,
            "crossing_locks": list(self.last_crossing_pairs),
            "crossing_lock_count": int(len(self.last_crossing_pairs)),
            "temporal_prediction": bool(self.params.temporal_prediction),
            "prediction_step_m": float(self.last_prediction_step_m),
            "sigma2": float(self.sigma2),
        }


def finite_points(points):
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    finite = np.all(np.isfinite(points[:, :3]), axis=1)
    return points[finite, :3]


def index_mask(indices, size):
    mask = np.zeros(int(size), dtype=bool)
    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(mask):
            mask[idx] = True
    return mask


def mean_finite_node_distance(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2:
        return np.inf

    finite = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    if not np.any(finite):
        return np.inf

    return float(np.mean(np.linalg.norm(A[finite] - B[finite], axis=1)))


def max_node_step(displacements):
    displacements = np.asarray(displacements, dtype=np.float64)
    if displacements.ndim != 2 or displacements.shape[1] < 3 or len(displacements) == 0:
        return 0.0
    finite = np.all(np.isfinite(displacements[:, :3]), axis=1)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.linalg.norm(displacements[finite, :3], axis=1)))


def clamp_node_displacements(displacements, max_step_m):
    displacements = np.asarray(displacements, dtype=np.float64).copy()
    if displacements.ndim != 2 or displacements.shape[1] < 3:
        return displacements

    max_step_m = max(0.0, float(max_step_m))
    if max_step_m <= 0.0:
        return np.zeros_like(displacements, dtype=np.float64)

    finite = np.all(np.isfinite(displacements[:, :3]), axis=1)
    if not np.any(finite):
        return np.zeros_like(displacements, dtype=np.float64)

    norms = np.linalg.norm(displacements[:, :3], axis=1)
    scale = np.ones(len(displacements), dtype=np.float64)
    too_large = finite & (norms > max_step_m)
    scale[too_large] = max_step_m / np.maximum(norms[too_large], 1e-12)
    displacements[:, :3] *= scale[:, None]
    displacements[~finite, :3] = 0.0
    return displacements


def detect_crossing_segment_pairs(
    Y,
    camera_intrinsics,
    image_shape,
    observed_nodes_xyz,
    observed_valid_nodes,
    observed_nodes_xy,
    distance_threshold_m,
    projection_threshold_px,
    projection_depth_margin_m,
    min_edge_gap,
    max_pairs,
):
    Y = np.asarray(Y, dtype=np.float64)
    if len(Y) < 4:
        return []

    distance_threshold_m = max(0.0, float(distance_threshold_m))
    projection_threshold_px = max(0.0, float(projection_threshold_px))
    projection_depth_margin_m = max(0.0, float(projection_depth_margin_m))
    min_edge_gap = max(1, int(min_edge_gap))
    max_pairs = max(0, int(max_pairs))
    if max_pairs == 0:
        return []

    projected_pixels = None
    projected_valid = None
    projected_depth = None
    if valid_projection_inputs(camera_intrinsics, image_shape):
        convention = choose_projection_convention(
            camera_intrinsics,
            image_shape,
            observed_nodes_xyz,
            observed_valid_nodes,
            observed_nodes_xy,
        )
        projected_pixels, projected_valid, projected_depth = project_points_to_image(
            Y,
            camera_intrinsics,
            image_shape,
            convention,
        )

    pairs = []
    for edge_a in range(len(Y) - 1):
        a0 = Y[edge_a]
        a1 = Y[edge_a + 1]
        if not (np.all(np.isfinite(a0)) and np.all(np.isfinite(a1))):
            continue

        for edge_b in range(edge_a + min_edge_gap + 1, len(Y) - 1):
            b0 = Y[edge_b]
            b1 = Y[edge_b + 1]
            if not (np.all(np.isfinite(b0)) and np.all(np.isfinite(b1))):
                continue

            distance_m = segment_segment_distance_3d(a0, a1, b0, b1)
            projection_px = np.inf
            projected_close = False
            if projected_pixels is not None and np.all(projected_valid[[edge_a, edge_a + 1, edge_b, edge_b + 1]]):
                pa0 = projected_pixels[edge_a]
                pa1 = projected_pixels[edge_a + 1]
                pb0 = projected_pixels[edge_b]
                pb1 = projected_pixels[edge_b + 1]
                projection_px = segment_segment_distance_2d(pa0, pa1, pb0, pb1)
                depth_a = float(np.mean(projected_depth[[edge_a, edge_a + 1]]))
                depth_b = float(np.mean(projected_depth[[edge_b, edge_b + 1]]))
                projected_close = (
                    projection_px <= projection_threshold_px
                    and abs(depth_a - depth_b) <= projection_depth_margin_m
                )

            if distance_m <= distance_threshold_m or projected_close:
                pairs.append(
                    {
                        "edge_a": int(edge_a),
                        "edge_b": int(edge_b),
                        "distance_m": float(distance_m),
                        "projection_px": float(projection_px),
                    }
                )

    pairs.sort(key=lambda item: (item["distance_m"], item["projection_px"]))
    return pairs[:max_pairs]


def segment_segment_distance_3d(a0, a1, b0, b1):
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)

    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b

    if a <= 1e-12 and c <= 1e-12:
        return float(np.linalg.norm(a0 - b0))
    if a <= 1e-12:
        return point_to_segment_distance_3d(a0, b0, b1)
    if c <= 1e-12:
        return point_to_segment_distance_3d(b0, a0, a1)

    if denom <= 1e-12:
        candidates = [
            point_to_segment_distance_3d(a0, b0, b1),
            point_to_segment_distance_3d(a1, b0, b1),
            point_to_segment_distance_3d(b0, a0, a1),
            point_to_segment_distance_3d(b1, a0, a1),
        ]
        return float(min(candidates))

    s = float(np.clip((b * e - c * d) / denom, 0.0, 1.0))
    t = float(np.clip((a * e - b * d) / denom, 0.0, 1.0))

    for _iteration in range(2):
        s = float(np.clip((b * t - d) / max(a, 1e-12), 0.0, 1.0))
        t = float(np.clip((b * s + e) / max(c, 1e-12), 0.0, 1.0))

    closest_a = a0 + s * u
    closest_b = b0 + t * v
    return float(np.linalg.norm(closest_a - closest_b))


def point_to_segment_distance_3d(point, segment_start, segment_end):
    point = np.asarray(point, dtype=np.float64)
    segment_start = np.asarray(segment_start, dtype=np.float64)
    segment_end = np.asarray(segment_end, dtype=np.float64)
    segment = segment_end - segment_start
    segment_norm = float(np.dot(segment, segment))
    if segment_norm <= 1e-12:
        return float(np.linalg.norm(point - segment_start))

    t = float(np.dot(point - segment_start, segment) / segment_norm)
    t = float(np.clip(t, 0.0, 1.0))
    closest = segment_start + t * segment
    return float(np.linalg.norm(point - closest))


def segment_segment_distance_2d(a0, a1, b0, b1):
    if segments_intersect_2d(a0, a1, b0, b1):
        return 0.0
    return float(
        min(
            point_to_segment_distance_2d(a0, b0, b1),
            point_to_segment_distance_2d(a1, b0, b1),
            point_to_segment_distance_2d(b0, a0, a1),
            point_to_segment_distance_2d(b1, a0, a1),
        )
    )


def segments_intersect_2d(a0, a1, b0, b1):
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)

    def orient(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)

    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= 1e-9 and on_segment(a0, b0, a1):
        return True
    if abs(o2) <= 1e-9 and on_segment(a0, b1, a1):
        return True
    if abs(o3) <= 1e-9 and on_segment(b0, a0, b1):
        return True
    if abs(o4) <= 1e-9 and on_segment(b0, a1, b1):
        return True
    return False


def fill_missing_nodes(nodes_xyz, valid_nodes):
    valid_indices = np.flatnonzero(valid_nodes & np.all(np.isfinite(nodes_xyz), axis=1))
    if len(valid_indices) < 2:
        return None

    filled = nodes_xyz.copy().astype(np.float64)
    all_indices = np.arange(len(nodes_xyz), dtype=np.float64)

    for dim in range(3):
        filled[:, dim] = np.interp(
            all_indices,
            valid_indices.astype(np.float64),
            nodes_xyz[valid_indices, dim].astype(np.float64),
        )

    return filled


def cumulative_node_distances(Y):
    if len(Y) == 0:
        return np.empty(0, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def rest_length_from_geodesic(geodesic_coord):
    if geodesic_coord is None:
        return 0.0
    geodesic_coord = np.asarray(geodesic_coord, dtype=np.float64).reshape(-1)
    if len(geodesic_coord) == 0:
        return 0.0
    return float(max(0.0, geodesic_coord[-1] - geodesic_coord[0]))


def make_rest_geodesic_coord(nodes_xyz, fixed_cable_length_m):
    nodes_xyz = np.asarray(nodes_xyz, dtype=np.float64)
    measured = cumulative_node_distances(nodes_xyz)
    if len(measured) == 0:
        return measured

    measured_length = rest_length_from_geodesic(measured)
    fixed_length = max(0.0, float(fixed_cable_length_m))
    rest_length = fixed_length if fixed_length > 1e-9 else measured_length
    if rest_length <= 1e-9:
        return measured

    return np.linspace(0.0, rest_length, len(measured), dtype=np.float64)


def enforce_rest_geodesic_spacing(nodes_xyz, rest_geodesic_coord, iterations=12):
    nodes = np.asarray(nodes_xyz, dtype=np.float64).copy()
    rest_geodesic_coord = np.asarray(rest_geodesic_coord, dtype=np.float64).reshape(-1)
    if len(nodes) < 2 or len(rest_geodesic_coord) != len(nodes):
        return nodes

    finite = np.all(np.isfinite(nodes), axis=1)
    if not np.all(finite):
        return nodes

    target_lengths = np.diff(rest_geodesic_coord)
    if len(target_lengths) != len(nodes) - 1 or not np.all(np.isfinite(target_lengths)):
        return nodes
    target_lengths = np.maximum(target_lengths.astype(np.float64), 0.0)
    target_total = float(np.sum(target_lengths))
    if target_total <= 1e-9:
        return nodes

    original_center = np.mean(nodes, axis=0)
    current_offsets = cumulative_node_distances(nodes)
    current_total = rest_length_from_geodesic(current_offsets)
    if current_total > 1e-9:
        scale = target_total / current_total
        nodes = original_center + (nodes - original_center) * scale
        scaled_offsets = cumulative_node_distances(nodes)
        rest_offsets = rest_geodesic_coord - rest_geodesic_coord[0]
        nodes = np.stack(
            [
                np.interp(rest_offsets, scaled_offsets, nodes[:, dim])
                for dim in range(3)
            ],
            axis=1,
        )
        nodes += original_center - np.mean(nodes, axis=0)

    previous_directions = segment_directions(nodes)

    for _iteration in range(max(1, int(iterations))):
        for idx, target_length in enumerate(target_lengths):
            delta = nodes[idx + 1] - nodes[idx]
            current_length = float(np.linalg.norm(delta))
            if current_length <= 1e-9:
                direction = previous_directions[idx]
            else:
                direction = delta / current_length
                previous_directions[idx] = direction

            correction = 0.5 * (current_length - float(target_length)) * direction
            nodes[idx] += correction
            nodes[idx + 1] -= correction

        nodes += original_center - np.mean(nodes, axis=0)

    return nodes


def segment_directions(nodes_xyz):
    nodes_xyz = np.asarray(nodes_xyz, dtype=np.float64)
    directions = np.zeros((max(len(nodes_xyz) - 1, 0), 3), dtype=np.float64)
    for idx in range(len(directions)):
        delta = nodes_xyz[idx + 1] - nodes_xyz[idx]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            directions[idx] = delta / norm
        elif idx > 0:
            directions[idx] = directions[idx - 1]
        else:
            directions[idx] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return directions


def pairwise_squared_distances(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sum(diff * diff, axis=2)


def second_order_kernel_from_chain(Y, beta):
    geodesic = cumulative_node_distances(Y)
    distances = np.abs(geodesic[:, None] - geodesic[None, :])
    beta = max(float(beta), 1e-6)
    return (
        1.0
        / (4.0 * beta * beta)
        * np.exp(-np.sqrt(2.0) * distances / beta)
        * (2.0 * distances + np.sqrt(2.0) * beta)
    )


def geodesic_node_point_squared_distances(Y, X, base_P):
    M, N = base_P.shape
    if M < 2 or N == 0:
        return pairwise_squared_distances(Y, X)

    geodesic = cumulative_node_distances(Y)
    geodesic_between_nodes = np.abs(geodesic[:, None] - geodesic[None, :])
    euclidean_distances = np.sqrt(pairwise_squared_distances(Y, X))
    output = np.zeros((M, N), dtype=np.float64)

    for n in range(N):
        finite = np.isfinite(euclidean_distances[:, n])
        if np.count_nonzero(finite) < 2:
            output[:, n] = euclidean_distances[:, n] ** 2
            continue

        candidates = np.flatnonzero(finite)
        local_order = np.argsort(euclidean_distances[candidates, n])
        c1 = int(candidates[local_order[0]])
        c2 = int(candidates[local_order[1]])
        d1 = float(euclidean_distances[c1, n])
        d2 = float(euclidean_distances[c2, n])

        # TrackDLO Eq. 18: near crossings, use the nearest of two Euclidean anchors
        # plus topological distance along the cable, not only Euclidean proximity.
        to_c1 = geodesic_between_nodes[:, c1]
        to_c2 = geodesic_between_nodes[:, c2]
        use_c1 = to_c1 <= to_c2
        distances = np.where(use_c1, d1 + to_c1, d2 + to_c2)
        distances[c1] = d1
        distances[c2] = d2
        output[:, n] = distances * distances

    return output


def cpd_mct_registration(
    X_orig,
    Y_init,
    sigma2,
    beta,
    lam,
    mu,
    max_iter,
    tol,
    alpha=0.0,
    correspondence_priors=None,
    visible_nodes=None,
    k_vis=0.0,
    visibility_threshold=0.01,
    prune_distance=0.1,
    sigma2_min=1e-8,
):
    Y = np.asarray(Y_init, dtype=np.float64).copy()
    X_orig = finite_points(np.asarray(X_orig, dtype=np.float64))
    M = len(Y)
    D = 3

    if len(X_orig) == 0 or M == 0:
        return Y, max(float(sigma2), sigma2_min), False

    nearest_to_Y = np.min(pairwise_squared_distances(X_orig, Y), axis=1) ** 0.5
    X = X_orig[nearest_to_Y < prune_distance]
    if len(X) == 0:
        X = X_orig

    N = len(X)
    Y0 = Y.copy()
    G = second_order_kernel_from_chain(Y0, beta)

    prior_points = {}
    prior_weights = {}
    if correspondence_priors:
        for prior in correspondence_priors:
            if len(prior) == 3:
                idx, xyz, weight = prior
            else:
                idx, xyz = prior
                weight = 1.0
            if idx < 0 or idx >= M:
                continue
            if np.all(np.isfinite(xyz)):
                idx = int(idx)
                weight = max(0.0, float(weight))
                if weight <= 0.0:
                    continue
                prior_points.setdefault(idx, []).append(weight * np.asarray(xyz, dtype=np.float64))
                prior_weights[idx] = prior_weights.get(idx, 0.0) + weight

    J = np.zeros((M, M), dtype=np.float64)
    Y_extended = Y0.copy()
    for idx, weighted_points in prior_points.items():
        weight = max(float(prior_weights[idx]), 1e-12)
        J[idx, idx] = weight
        Y_extended[idx] = np.sum(np.stack(weighted_points, axis=0), axis=0) / weight

    diff_xy = pairwise_squared_distances(Y0, X)
    if sigma2 <= sigma2_min:
        sigma2 = float(np.sum(diff_xy) / max(D * M * N, 1))
    sigma2 = max(float(sigma2), sigma2_min)

    converged = False
    identity = np.eye(M, dtype=np.float64)

    for _iteration in range(max_iter):
        diff_xy = pairwise_squared_distances(Y, X)
        base_P = np.exp(-0.5 * diff_xy / sigma2)
        geo_diff_xy = geodesic_node_point_squared_distances(Y, X, base_P)
        P = np.exp(-0.5 * geo_diff_xy / sigma2)

        c = ((2.0 * np.pi * sigma2) ** (D / 2.0)) * mu / max(1.0 - mu, 1e-6) * M / N

        if visible_nodes is not None and len(visible_nodes) != M and len(visible_nodes) > 0 and k_vis != 0:
            shortest = np.sqrt(np.min(diff_xy, axis=1))
            shortest = np.where(shortest <= visibility_threshold, 0.0, shortest)
            P_vis = np.exp(-float(k_vis) * shortest)
            P_vis = P_vis / max(float(np.sum(P_vis)), 1e-12)
            P *= P_vis[:, None]
            c = ((2.0 * np.pi * sigma2) ** (D / 2.0)) * mu / max(1.0 - mu, 1e-6) / N

        denom = np.sum(P, axis=0, keepdims=True) + c
        P = P / np.maximum(denom, 1e-12)

        P1 = np.sum(P, axis=1)
        Pt1 = np.sum(P, axis=0)
        Np = float(np.sum(P1))
        if Np <= 1e-12:
            break

        PX = P @ X
        A = np.diag(P1) @ G + float(lam) * sigma2 * identity
        B = PX - np.diag(P1) @ Y0

        if prior_points:
            A = A + float(alpha) * J @ G
            B = B + float(alpha) * J @ (Y_extended - Y0)

        try:
            W = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(A, B, rcond=None)[0]

        T = Y0 + G @ W
        tr_x = float(np.sum((X * X) * Pt1[:, None]))
        tr_px = float(np.trace(PX.T @ T))
        tr_t = float(np.sum((T * T) * P1[:, None]))
        sigma2_new = (tr_x - 2.0 * tr_px + tr_t) / max(Np * D, 1e-12)
        sigma2_new = max(float(sigma2_new), sigma2_min)

        mean_delta = float(np.linalg.norm(T - Y) / max(M, 1))
        Y = T
        sigma2 = sigma2_new
        if mean_delta < tol:
            converged = True
            break

    return Y.astype(np.float64), float(sigma2), converged


def estimate_visible_nodes(Y, X, visibility_threshold):
    X = finite_points(X)
    if len(X) == 0 or len(Y) == 0:
        return []
    nearest = np.sqrt(np.min(pairwise_squared_distances(Y, X), axis=1))
    return [int(idx) for idx in np.flatnonzero(nearest <= visibility_threshold)]


def prune_self_occluded_visible_nodes(
    Y,
    visible_nodes,
    camera_intrinsics,
    image_shape,
    dlo_pixel_width,
    cable_diameter_m,
    min_projected_dlo_width_px,
    max_projected_dlo_width_px,
    depth_margin,
    max_projection_error,
    observed_nodes_xyz=None,
    observed_valid_nodes=None,
    observed_nodes_xy=None,
):
    visible_set = set(int(idx) for idx in visible_nodes)
    if len(Y) < 2 or not visible_set:
        return sorted(visible_set), [], "unavailable"

    if not valid_projection_inputs(camera_intrinsics, image_shape):
        return sorted(visible_set), [], "unavailable"

    convention = choose_projection_convention(
        camera_intrinsics,
        image_shape,
        observed_nodes_xyz,
        observed_valid_nodes,
        observed_nodes_xy,
    )
    projection_error = float(convention.get("score", np.nan))
    if np.isfinite(projection_error) and projection_error > float(max_projection_error):
        return sorted(visible_set), [], projection_convention_label(convention) + " skip"

    pixels, projected_valid, depth = project_points_to_image(
        Y,
        camera_intrinsics,
        image_shape,
        convention,
    )

    if not np.any(projected_valid):
        return sorted(visible_set), [], projection_convention_label(convention)

    rows, cols = int(image_shape[0]), int(image_shape[1])
    accepted_visible = set(visible_set)
    self_occluded = set()

    edge_distances = []
    for idx in range(len(Y) - 1):
        p0 = Y[idx]
        p1 = Y[idx + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        midpoint = 0.5 * (p0 + p1)
        edge_depth = edge_depth_m(depth, midpoint, idx)
        edge_distances.append((edge_depth, idx))

    edge_distances_by_idx = {idx: distance for distance, idx in edge_distances}
    edge_indices = [idx for _distance, idx in sorted(edge_distances)]
    depth_margin = max(0.0, float(depth_margin))
    drawn_edges = []

    for edge_idx in edge_indices:
        for node_idx in (edge_idx, edge_idx + 1):
            if node_idx not in visible_set:
                continue
            if node_idx in self_occluded:
                continue
            if not projected_valid[node_idx]:
                continue

            col, row = projected_node_to_pixel(pixels[node_idx], rows, cols)
            if col is None:
                continue

            node_pixel = np.array([col, row], dtype=np.float64)
            node_distance = node_depth_m(depth, Y[node_idx], node_idx)
            node_width = projected_cable_width_px(
                node_distance,
                camera_intrinsics,
                dlo_pixel_width,
                cable_diameter_m,
                min_projected_dlo_width_px,
                max_projected_dlo_width_px,
            )
            node_radius = max(1.0, 0.5 * node_width)
            if has_non_adjacent_front_edge_overlap(
                node_idx,
                node_pixel,
                node_distance,
                drawn_edges,
                node_radius,
                depth_margin,
            ):
                accepted_visible.discard(node_idx)
                self_occluded.add(node_idx)

        if projected_valid[edge_idx] and projected_valid[edge_idx + 1]:
            p0 = projected_node_to_pixel(pixels[edge_idx], rows, cols, clamp=True)
            p1 = projected_node_to_pixel(pixels[edge_idx + 1], rows, cols, clamp=True)
            if p0[0] is not None and p1[0] is not None:
                edge_width = projected_cable_width_px(
                    edge_distances_by_idx[edge_idx],
                    camera_intrinsics,
                    dlo_pixel_width,
                    cable_diameter_m,
                    min_projected_dlo_width_px,
                    max_projected_dlo_width_px,
                )
                drawn_edges.append(
                    {
                        "idx": int(edge_idx),
                        "distance": float(edge_distances_by_idx[edge_idx]),
                        "radius": max(1.0, 0.5 * edge_width),
                        "p0": np.array(p0, dtype=np.float64),
                        "p1": np.array(p1, dtype=np.float64),
                    }
                )

    return sorted(accepted_visible), sorted(self_occluded), projection_convention_label(convention)


def has_non_adjacent_front_edge_overlap(
    node_idx,
    node_pixel,
    node_distance,
    drawn_edges,
    node_radius,
    depth_margin,
):
    for edge in drawn_edges:
        edge_idx = int(edge["idx"])
        if edge_idx in (node_idx - 1, node_idx):
            continue
        if node_distance - float(edge["distance"]) < depth_margin:
            continue
        distance_px = point_to_segment_distance_2d(node_pixel, edge["p0"], edge["p1"])
        overlap_radius = max(float(node_radius), float(edge.get("radius", 1.0)))
        if distance_px <= overlap_radius:
            return True
    return False


def edge_depth_m(depth, midpoint, edge_idx):
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    local_depth = depth[edge_idx : edge_idx + 2]
    finite_depth = local_depth[np.isfinite(local_depth) & (local_depth > 1e-6)]
    if len(finite_depth) > 0:
        return float(np.mean(finite_depth))
    return float(max(np.linalg.norm(midpoint), 1e-6))


def node_depth_m(depth, node_xyz, node_idx):
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    if 0 <= int(node_idx) < len(depth):
        value = float(depth[int(node_idx)])
        if np.isfinite(value) and value > 1e-6:
            return value
    return float(max(np.linalg.norm(node_xyz), 1e-6))


def projected_cable_width_px(
    depth_m,
    camera_intrinsics,
    fallback_width_px,
    cable_diameter_m,
    min_width_px,
    max_width_px,
):
    fallback = max(1.0, float(fallback_width_px))
    min_width = max(1.0, float(min_width_px))
    max_width = max(min_width, float(max_width_px))
    depth_m = float(depth_m)
    cable_diameter_m = float(cable_diameter_m)

    width = fallback
    if (
        camera_intrinsics is not None
        and cable_diameter_m > 0.0
        and np.isfinite(depth_m)
        and depth_m > 1e-6
    ):
        fx = float(camera_intrinsics.get("fx", np.nan))
        fy = float(camera_intrinsics.get("fy", np.nan))
        if np.isfinite(fx) and np.isfinite(fy) and fx > 0.0 and fy > 0.0:
            focal_px = 0.5 * (fx + fy)
            width = focal_px * cable_diameter_m / depth_m

    return float(np.clip(width, min_width, max_width))


def point_to_segment_distance_2d(point, segment_start, segment_end):
    point = np.asarray(point, dtype=np.float64)
    segment_start = np.asarray(segment_start, dtype=np.float64)
    segment_end = np.asarray(segment_end, dtype=np.float64)
    segment = segment_end - segment_start
    segment_norm = float(np.dot(segment, segment))
    if segment_norm <= 1e-12:
        return float(np.linalg.norm(point - segment_start))

    t = float(np.dot(point - segment_start, segment) / segment_norm)
    t = float(np.clip(t, 0.0, 1.0))
    closest = segment_start + t * segment
    return float(np.linalg.norm(point - closest))


def valid_projection_inputs(camera_intrinsics, image_shape):
    if camera_intrinsics is None or image_shape is None:
        return False
    if len(image_shape) < 2:
        return False
    if int(image_shape[0]) <= 0 or int(image_shape[1]) <= 0:
        return False
    for key in ("fx", "fy", "cx", "cy"):
        if key not in camera_intrinsics:
            return False
        if not np.isfinite(float(camera_intrinsics[key])):
            return False
    return True


def choose_projection_convention(
    camera_intrinsics,
    image_shape,
    observed_nodes_xyz,
    observed_valid_nodes,
    observed_nodes_xy,
):
    conventions = [
        {"depth_sign": 1.0, "image_y_sign": 1.0, "name": "z+ y-down", "score": np.nan},
        {"depth_sign": 1.0, "image_y_sign": -1.0, "name": "z+ y-up", "score": np.nan},
        {"depth_sign": -1.0, "image_y_sign": 1.0, "name": "z- y-down", "score": np.nan},
        {"depth_sign": -1.0, "image_y_sign": -1.0, "name": "z- y-up", "score": np.nan},
    ]

    if observed_nodes_xyz is None:
        observed_nodes_xyz = np.empty((0, 3), dtype=np.float64)
    else:
        observed_nodes_xyz = np.asarray(observed_nodes_xyz, dtype=np.float64)

    if observed_nodes_xy is None:
        observed_nodes_xy = np.empty((0, 2), dtype=np.float64)
    else:
        observed_nodes_xy = np.asarray(observed_nodes_xy, dtype=np.float64)

    if observed_valid_nodes is None:
        observed_valid_nodes = np.zeros(len(observed_nodes_xy), dtype=bool)
    else:
        observed_valid_nodes = np.asarray(observed_valid_nodes, dtype=bool).reshape(-1)

    can_score = (
        observed_nodes_xyz.ndim == 2
        and observed_nodes_xyz.shape[1] >= 3
        and observed_nodes_xy.ndim == 2
        and observed_nodes_xy.shape[1] >= 2
        and len(observed_nodes_xyz) == len(observed_nodes_xy)
        and len(observed_valid_nodes) == len(observed_nodes_xy)
        and np.count_nonzero(observed_valid_nodes) >= 3
    )

    if can_score:
        reference_xyz = observed_nodes_xyz[:, :3]
        reference_xy = observed_nodes_xy[:, :2]
        best_convention = conventions[0]
        best_score = np.inf

        for convention in conventions:
            pixels, valid, _depth = project_points_to_image(
                reference_xyz,
                camera_intrinsics,
                image_shape,
                convention,
            )
            usable = valid & observed_valid_nodes & np.all(np.isfinite(reference_xy), axis=1)
            if np.count_nonzero(usable) < 3:
                continue

            errors = np.linalg.norm(pixels[usable] - reference_xy[usable], axis=1)
            score = float(np.median(errors))
            if score < best_score:
                best_score = score
                best_convention = convention.copy()
                best_convention["score"] = score

        if np.isfinite(best_score):
            return best_convention

    if observed_nodes_xyz.ndim == 2 and observed_nodes_xyz.shape[1] >= 3:
        finite_z = observed_nodes_xyz[np.all(np.isfinite(observed_nodes_xyz[:, :3]), axis=1), 2]
        if len(finite_z) > 0 and float(np.median(finite_z)) < 0.0:
            return conventions[3].copy()

    return conventions[0].copy()


def projection_convention_label(convention):
    name = str(convention.get("name", "unavailable"))
    score = float(convention.get("score", np.nan))
    if np.isfinite(score):
        return f"{name} err {score:.1f}px"
    return name


def project_points_to_image(points, camera_intrinsics, image_shape, convention):
    points = np.asarray(points, dtype=np.float64)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    depth = np.full(len(points), np.nan, dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)

    if len(points) == 0:
        return pixels, valid, depth

    xyz = points[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    depth = float(convention["depth_sign"]) * xyz[:, 2]
    valid = finite & (depth > 1e-6)

    if not np.any(valid):
        return pixels, valid, depth

    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    y_sign = float(convention["image_y_sign"])

    pixels[valid, 0] = fx * xyz[valid, 0] / depth[valid] + cx
    pixels[valid, 1] = y_sign * fy * xyz[valid, 1] / depth[valid] + cy

    rows, cols = int(image_shape[0]), int(image_shape[1])
    in_frame = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < cols)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < rows)
    )
    valid &= in_frame
    return pixels, valid, depth


def projected_node_to_pixel(pixel, rows, cols, clamp=False):
    if not np.all(np.isfinite(pixel)):
        return None, None

    col = int(round(float(pixel[0])))
    row = int(round(float(pixel[1])))

    if clamp:
        col = int(np.clip(col, 0, cols - 1))
        row = int(np.clip(row, 0, rows - 1))
        return col, row

    if col < 0 or col >= cols or row < 0 or row >= rows:
        return None, None
    return col, row


def extend_visible_nodes(visible_nodes, geodesic_coord, d_vis):
    if not visible_nodes:
        return []

    visible_nodes = sorted(set(int(idx) for idx in visible_nodes))
    extended = []

    for left, right in zip(visible_nodes[:-1], visible_nodes[1:]):
        extended.append(left)
        if abs(float(geodesic_coord[right] - geodesic_coord[left])) <= d_vis:
            extended.extend(range(left + 1, right))

    extended.append(visible_nodes[-1])
    return sorted(set(extended))


def contiguous_runs(indices):
    if not indices:
        return []
    indices = sorted(set(int(idx) for idx in indices))
    runs = [[indices[0]]]
    for idx in indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def build_trackdlo_length_priors(previous_nodes, guide_nodes, visible_nodes, visible_nodes_extended, geodesic_coord):
    previous_nodes = np.asarray(previous_nodes, dtype=np.float64)
    guide_nodes = np.asarray(guide_nodes, dtype=np.float64)
    visible_nodes = sorted(set(int(idx) for idx in visible_nodes))
    visible_nodes_extended = sorted(set(int(idx) for idx in visible_nodes_extended))
    geodesic_coord = np.asarray(geodesic_coord, dtype=np.float64)

    if len(guide_nodes) == 0 or not visible_nodes_extended:
        return [], "no visible nodes"

    node_count = len(previous_nodes)
    guide_by_index = {
        int(node_idx): np.asarray(guide_nodes[row], dtype=np.float64)
        for row, node_idx in enumerate(visible_nodes_extended)
        if row < len(guide_nodes)
    }
    runs = contiguous_runs(visible_nodes_extended)

    state = classify_visibility_state(visible_nodes, visible_nodes_extended, node_count)

    if len(visible_nodes_extended) == node_count:
        head_priors = traverse_visible_run(runs[0], guide_by_index, geodesic_coord, "head")
        tail_priors = traverse_visible_run(runs[-1], guide_by_index, geodesic_coord, "tail")
        return merge_prior_sets([head_priors, tail_priors]), state

    if visible_nodes_extended[0] == 0 and visible_nodes_extended[-1] == node_count - 1:
        head_run = runs[0]
        tail_run = runs[-1]
        head_priors = traverse_visible_run(head_run, guide_by_index, geodesic_coord, "head")
        tail_priors = traverse_visible_run(tail_run, guide_by_index, geodesic_coord, "tail")
        return merge_prior_sets([head_priors, tail_priors]), state

    if visible_nodes_extended[0] == 0:
        return traverse_visible_run(runs[0], guide_by_index, geodesic_coord, "head"), state

    if visible_nodes_extended[-1] == node_count - 1:
        return traverse_visible_run(runs[-1], guide_by_index, geodesic_coord, "tail"), state

    anchor = choose_least_moving_visible_anchor(previous_nodes, guide_by_index, visible_nodes, visible_nodes_extended)
    if anchor is None:
        return [], state

    anchor_run = next((run for run in runs if run[0] <= anchor <= run[-1]), [anchor])
    anchor_pos = anchor_run.index(anchor)
    backward_run = anchor_run[: anchor_pos + 1]
    forward_run = anchor_run[anchor_pos:]
    backward_priors = traverse_visible_run(backward_run, guide_by_index, geodesic_coord, "tail")
    forward_priors = traverse_visible_run(forward_run, guide_by_index, geodesic_coord, "head")
    return merge_prior_sets([backward_priors, forward_priors]), state


def classify_visibility_state(visible_nodes, visible_nodes_extended, node_count):
    visible_nodes = sorted(set(int(idx) for idx in visible_nodes))
    visible_nodes_extended = sorted(set(int(idx) for idx in visible_nodes_extended))
    if not visible_nodes_extended:
        return "no visible nodes"
    if len(visible_nodes_extended) == int(node_count):
        return "all visible" if len(visible_nodes) == len(visible_nodes_extended) else "minor occlusion"
    if visible_nodes_extended[0] == 0 and visible_nodes_extended[-1] == int(node_count) - 1:
        return "mid-section occluded"
    if visible_nodes_extended[0] == 0:
        return "tail occluded"
    if visible_nodes_extended[-1] == int(node_count) - 1:
        return "head occluded"
    return "both ends occluded"


def traverse_visible_run(run, guide_by_index, geodesic_coord, direction):
    run = [int(idx) for idx in run if int(idx) in guide_by_index]
    if not run:
        return []
    if len(run) == 1:
        idx = run[0]
        return [(idx, guide_by_index[idx].copy())]

    if direction == "tail":
        ordered_indices = list(reversed(run))
    else:
        ordered_indices = list(run)

    ordered_points = np.stack([guide_by_index[idx] for idx in ordered_indices], axis=0)
    path_offsets = cumulative_node_distances(ordered_points)
    if path_offsets[-1] <= 1e-9:
        return [(ordered_indices[0], ordered_points[0].copy())]

    start_idx = ordered_indices[0]
    priors = []
    for idx in ordered_indices:
        if direction == "tail":
            target_offset = float(geodesic_coord[start_idx] - geodesic_coord[idx])
        else:
            target_offset = float(geodesic_coord[idx] - geodesic_coord[start_idx])

        if target_offset < -1e-9:
            continue
        if target_offset > path_offsets[-1] + 1e-9:
            break

        priors.append((idx, interpolate_polyline(ordered_points, path_offsets, max(0.0, target_offset))))

    return priors


def interpolate_polyline(points, path_offsets, target_offset):
    target_offset = float(np.clip(target_offset, 0.0, path_offsets[-1]))
    return np.array(
        [
            np.interp(target_offset, path_offsets, points[:, dim])
            for dim in range(3)
        ],
        dtype=np.float64,
    )


def merge_prior_sets(prior_sets):
    merged = {}
    for priors in prior_sets:
        for idx, xyz in priors:
            idx = int(idx)
            xyz = np.asarray(xyz, dtype=np.float64)
            if not np.all(np.isfinite(xyz)):
                continue
            merged.setdefault(idx, []).append(xyz)

    return [
        (idx, np.mean(np.stack(points, axis=0), axis=0))
        for idx, points in sorted(merged.items())
    ]


def choose_least_moving_visible_anchor(previous_nodes, guide_by_index, visible_nodes, visible_nodes_extended):
    candidates = [idx for idx in visible_nodes if idx in guide_by_index]
    if not candidates:
        candidates = [idx for idx in visible_nodes_extended if idx in guide_by_index]
    if not candidates:
        return None

    best_idx = None
    best_distance = np.inf
    for idx in candidates:
        if idx >= len(previous_nodes) or not np.all(np.isfinite(previous_nodes[idx])):
            continue
        distance = float(np.linalg.norm(guide_by_index[idx] - previous_nodes[idx]))
        if distance < best_distance:
            best_idx = idx
            best_distance = distance

    if best_idx is not None:
        return int(best_idx)
    return int(candidates[len(candidates) // 2])


def compute_polyline_length(nodes_xyz, valid_nodes):
    nodes_xyz = np.asarray(nodes_xyz, dtype=np.float64)
    valid_nodes = np.asarray(valid_nodes, dtype=bool)
    length = 0.0
    valid_segments = 0

    for idx in range(len(nodes_xyz) - 1):
        if not (valid_nodes[idx] and valid_nodes[idx + 1]):
            continue
        if not (np.all(np.isfinite(nodes_xyz[idx])) and np.all(np.isfinite(nodes_xyz[idx + 1]))):
            continue
        length += float(np.linalg.norm(nodes_xyz[idx + 1] - nodes_xyz[idx]))
        valid_segments += 1

    return length, valid_segments
