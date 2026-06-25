from dataclasses import dataclass
import heapq

import numpy as np

from cable_detection import (
    CableEstimate3D,
    fit_polyline_segments,
    polyline_residual,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for CPU-only use
    torch = None


@dataclass
class CableParticleFilterConfig:
    particle_count: int = 200
    cable_length_m: float = 0.0
    cable_length_scale: float = 1.0
    segment_length_m: float = 0.0
    initial_node_std_m: float = 0.025
    initial_direction_std: float = 0.10
    process_node_std_m: float = 0.006
    process_direction_std: float = 0.020
    velocity_prediction: bool = True
    velocity_alpha: float = 0.35
    velocity_decay: float = 0.90
    velocity_deadband_mps: float = 0.06
    max_velocity_mps: float = 0.40
    direction_smooth_passes: int = 1
    measurement_node_std_m: float = 0.030
    measurement_max_points: int = 1024
    scoring_backend: str = "auto"
    score_chunk_points: int = 512
    endpoint_ordering: bool = True
    ordering_max_points: int = 384
    ordering_knn: int = 8
    reference_ordering: bool = True
    reference_ordering_gate_m: float = 0.08
    measurement_proposal_ratio: float = 0.30
    measurement_proposal_stable_ratio: float = 0.04
    measurement_proposal_start_error_m: float = 0.015
    measurement_proposal_full_error_m: float = 0.060
    measurement_proposal_node_std_m: float = 0.012
    measurement_proposal_direction_std: float = 0.025
    measurement_proposal_wide_fraction: float = 0.20
    measurement_proposal_current_fraction: float = 0.20
    measurement_proposal_wide_std_multiplier: float = 3.0
    estimate_mode: str = "top_clustered"
    top_particle_count: int = 10
    top_particle_cluster_gate_m: float = 0.030
    endpoint_anchor_weight: float = 0.0
    endpoint_anchor_projection: bool = False
    endpoint_anchor_projection_gain: float = 0.0
    endpoint_anchor_projection_passes: int = 1
    map_estimate_effective_ratio: float = 0.35
    min_measurement_points: int = 12
    min_segment_points: int = 4
    occlusion_assignment_max_distance_m: float = 0.12
    occlusion_segment_scoring: bool = True
    occlusion_segment_window: int = 1
    outlier_distance_m: float = 0.08
    resample_effective_ratio: float = 0.50
    max_prediction_frames: int = 12
    max_motion_noise_scale: float = 4.0


@dataclass
class CableParticleFilterResult:
    points_xyz: np.ndarray
    effective_sample_size: float
    measurement_used: bool
    prediction_only: bool
    lost_frames: int
    motion_noise_scale: float
    segment_length_m: float
    prediction_step_m: float = 0.0
    measurement_point_count: int = 0
    visible_segments: np.ndarray | None = None
    visible_nodes: np.ndarray | None = None
    measurement_proposal_ratio: float = 0.0
    estimate_mode: str = "top_clustered"
    top_particles: np.ndarray | None = None
    estimate_particle_count: int = 0


class CableParticleFilter:
    """Particle filter over connected fixed-length cable segments.

    A particle is stored as connected nodes, but prediction only moves the chain
    start and segment directions. Segment roll is not represented, and every
    segment is projected back to one shared fixed length.
    """

    def __init__(self, node_count, config=None, seed=17):
        self.node_count = max(2, int(node_count))
        self.segment_count = self.node_count - 1
        self.config = config if config is not None else CableParticleFilterConfig()
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.weights = None
        self.initialized = False
        self.lost_frames = 0
        self.segment_length_m = fixed_segment_length_from_config(self.config, self.segment_count)
        self.last_motion_noise_scale = 1.0
        self.last_visible_segments = np.zeros(self.segment_count, dtype=bool)
        self.last_visible_nodes = np.zeros(self.node_count, dtype=bool)
        self.last_measurement_point_count = 0
        self.last_measurement_proposal_ratio = 0.0
        self.last_estimate_mode = str(getattr(self.config, "estimate_mode", "top_clustered"))
        self.last_estimate_particle_count = 0
        self.last_ordered_measurement_nodes = None
        self.last_endpoint_anchor_nodes = None
        self.measurement_update_count = 0
        self.previous_estimate_nodes = None
        self.node_velocity_mps = np.zeros((self.node_count, 3), dtype=np.float64)
        self.last_prediction_step_m = 0.0

    def step(self, measurement, dt=1.0 / 30.0, count_lost=True):
        dt = float(np.clip(dt, 1e-3, 0.20))
        measurement_points = self._measurement_points(measurement)
        if measurement_points is not None:
            self.last_measurement_point_count = int(len(measurement_points))
        elif count_lost:
            self.last_measurement_point_count = 0

        if measurement_points is not None:
            if not self.initialized:
                self._initialize(measurement, measurement_points)
            else:
                self._predict(dt, noise_scale=1.0)
                reference_nodes = self._estimate_nodes()
                measurement_nodes = self._measurement_nodes(
                    measurement,
                    measurement_points,
                    reference_nodes=reference_nodes,
                )
                associated_points, segment_indices, visible_segments = self._associate_visible_points(
                    measurement_points,
                    candidate_nodes=measurement_nodes,
                    reference_nodes=reference_nodes,
                )
                if associated_points is None:
                    visible_segments = np.zeros(self.segment_count, dtype=bool)
                    support_points = measurement_points
                    support_segment_indices = None
                else:
                    support_points = associated_points
                    support_segment_indices = segment_indices
                endpoint_nodes = self._current_endpoint_anchor_nodes(measurement_nodes)
                endpoint_nodes = mask_endpoint_nodes_by_visibility(endpoint_nodes, visible_segments, self.segment_count)
                proposal_nodes = merge_visible_measurement_nodes(
                    measurement_nodes,
                    reference_nodes,
                    visible_segments,
                )
                self._inject_measurement_proposals(proposal_nodes, current_nodes=reference_nodes)
                self._anchor_particles_to_endpoints(endpoint_nodes)
                if support_points is None:
                    self.last_measurement_point_count = 0
                    result = self._prediction_only_after_update_drop()
                    return self._finalize_result(result, dt)
                self.last_measurement_point_count = int(len(support_points))
                self._weight(
                    support_points,
                    visible_segments=visible_segments,
                    endpoint_nodes=endpoint_nodes,
                    segment_indices=support_segment_indices,
                )
                self._set_visibility(visible_segments)
                result = self._estimate(measurement_used=True, prediction_only=False)
                self._maybe_resample(noise_scale=self.last_motion_noise_scale)
                self.lost_frames = 0
                return self._finalize_result(result, dt)

            self.lost_frames = 0
            result = self._estimate(measurement_used=True, prediction_only=False)
            return self._finalize_result(result, dt)

        if not self.initialized:
            return None

        self.last_measurement_point_count = 0
        if count_lost:
            self.lost_frames += 1
        if self.lost_frames > int(self.config.max_prediction_frames):
            self.initialized = False
            self._reset_motion_state()
            return None

        self.last_motion_noise_scale = min(
            1.0 + 0.25 * self.lost_frames,
            float(self.config.max_motion_noise_scale),
        )
        self._predict(dt, self.last_motion_noise_scale)
        if count_lost:
            self._set_visibility(np.zeros(self.segment_count, dtype=bool))
        result = self._estimate(measurement_used=False, prediction_only=True)
        self._maybe_resample(force=self.lost_frames > 1, noise_scale=self.last_motion_noise_scale)
        return self._finalize_result(result, dt)

    def _initialize(self, measurement, measurement_points):
        nodes = self._initial_nodes(measurement, measurement_points)
        if nodes is None:
            return

        endpoint_nodes = endpoint_nodes_from_chain(nodes)
        if self.segment_length_m is None:
            self.segment_length_m = estimate_equal_segment_length(nodes, self.segment_count)
        nodes = project_equal_length_chain(nodes, self.segment_length_m, self.node_count)
        if nodes is None:
            return
        if endpoint_nodes is None:
            endpoint_nodes = endpoint_nodes_from_chain(nodes)
        self.last_endpoint_anchor_nodes = endpoint_nodes
        self.last_ordered_measurement_nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        directions = chain_directions(nodes)
        count = max(32, int(self.config.particle_count))

        starts = nodes[0][None, :] + self.rng.normal(
            0.0,
            float(self.config.initial_node_std_m),
            (count, 3),
        )
        direction_noise = self.rng.normal(
            0.0,
            float(self.config.initial_direction_std),
            (count, self.segment_count, 3),
        )
        particle_directions = smooth_particle_directions(
            normalize_vectors(directions[None, :, :] + direction_noise),
            passes=int(getattr(self.config, "direction_smooth_passes", 1)),
        )
        self.particles = build_chains(starts, particle_directions, self.segment_length_m)
        self.weights = np.full(count, 1.0 / count, dtype=np.float64)
        self._anchor_particles_to_endpoints(endpoint_nodes)
        self.initialized = True
        self.last_motion_noise_scale = 1.0
        self.last_measurement_proposal_ratio = 0.0
        self.measurement_update_count = 1
        _points, _indices, visible_segments = self._associate_visible_points(measurement_points)
        self.last_measurement_point_count = int(len(measurement_points))
        self._weight(measurement_points, visible_segments=visible_segments, endpoint_nodes=endpoint_nodes)
        self._set_visibility(visible_segments)
        self._reset_motion_state(nodes)

    def _predict(self, dt, noise_scale=1.0):
        if self.particles is None or self.segment_length_m is None:
            return
        noise_scale = float(np.clip(noise_scale, 0.25, self.config.max_motion_noise_scale))
        self.last_motion_noise_scale = noise_scale
        predicted_particles = self.particles
        prediction_delta = self._prediction_delta(dt)
        if prediction_delta is not None:
            predicted_particles = self.particles + prediction_delta[None, :, :]

        starts = predicted_particles[:, 0, :] + self.rng.normal(
            0.0,
            float(self.config.process_node_std_m) * noise_scale,
            (len(self.particles), 3),
        )
        directions = particle_directions(predicted_particles)
        directions = normalize_vectors(
            directions
            + self.rng.normal(
                0.0,
                float(self.config.process_direction_std) * noise_scale,
                directions.shape,
            )
        )
        directions = smooth_particle_directions(
            directions,
            passes=int(getattr(self.config, "direction_smooth_passes", 1)),
        )
        self.particles = build_chains(starts, directions, self.segment_length_m)

    def _prediction_delta(self, dt):
        self.last_prediction_step_m = 0.0
        if not bool(getattr(self.config, "velocity_prediction", True)):
            return None

        velocity = np.asarray(self.node_velocity_mps, dtype=np.float64)
        if velocity.shape != (self.node_count, 3) or not np.all(np.isfinite(velocity)):
            self.node_velocity_mps = np.zeros((self.node_count, 3), dtype=np.float64)
            return None

        delta = velocity * float(np.clip(dt, 1e-3, 0.20))
        max_step = max(0.0, float(getattr(self.config, "max_velocity_mps", 0.40))) * float(np.clip(dt, 1e-3, 0.20))
        if max_step > 0.0:
            delta = clamp_node_vectors(delta, max_norm=max_step)
        step_m = max_node_vector_norm(delta)
        self.last_prediction_step_m = float(step_m)
        if step_m <= 1e-9:
            return None
        return np.ascontiguousarray(delta, dtype=np.float64)

    def _weight(self, measurement_points, visible_segments=None, endpoint_nodes=None, segment_indices=None):
        sigma = max(float(self.config.measurement_node_std_m), 1e-5)
        outlier_distance = max(float(self.config.outlier_distance_m), sigma)
        scoring_segment_indices = None
        if bool(getattr(self.config, "occlusion_segment_scoring", True)):
            scoring_segment_indices = segment_indices
        scores = particle_distance_scores(
            measurement_points,
            self.particles,
            outlier_distance=outlier_distance,
            segment_indices=scoring_segment_indices,
            segment_window=int(getattr(self.config, "occlusion_segment_window", 1)),
            backend=str(getattr(self.config, "scoring_backend", "auto")),
            chunk_points=int(getattr(self.config, "score_chunk_points", 512)),
        )
        endpoint_weight = max(0.0, float(getattr(self.config, "endpoint_anchor_weight", 0.0)))
        if endpoint_weight > 0.0:
            scores = scores + endpoint_weight * endpoint_anchor_scores(self.particles, endpoint_nodes)

        if not np.any(np.isfinite(scores)):
            return

        finite_scores = np.where(np.isfinite(scores), scores, np.nanmax(scores[np.isfinite(scores)]) + outlier_distance**2)
        log_likelihood = -0.5 * finite_scores / (sigma * sigma)
        log_likelihood -= float(np.max(log_likelihood))
        likelihood = np.exp(log_likelihood)
        weighted = self.weights * likelihood
        total = float(np.sum(weighted))
        if not np.isfinite(total) or total <= 1e-12:
            self.weights.fill(1.0 / len(self.weights))
            return
        self.weights = weighted / total

    def _finalize_result(self, result, dt):
        if result is None:
            return None
        self._update_motion_state(
            result.points_xyz,
            dt=dt,
            measurement_used=bool(result.measurement_used),
            prediction_only=bool(result.prediction_only),
            visible_nodes=result.visible_nodes,
        )
        return result

    def _reset_motion_state(self, nodes=None):
        self.node_velocity_mps = np.zeros((self.node_count, 3), dtype=np.float64)
        node_array = finite_node_chain(nodes, self.node_count)
        self.previous_estimate_nodes = None if node_array is None else node_array.copy()
        self.last_prediction_step_m = 0.0

    def _update_motion_state(self, nodes, dt, measurement_used, prediction_only, visible_nodes=None):
        node_array = finite_node_chain(nodes, self.node_count)
        if node_array is None:
            return
        if self.previous_estimate_nodes is None:
            self.previous_estimate_nodes = node_array.copy()
            self.node_velocity_mps = np.zeros((self.node_count, 3), dtype=np.float64)
            return

        current_velocity = np.asarray(self.node_velocity_mps, dtype=np.float64)
        if current_velocity.shape != (self.node_count, 3) or not np.all(np.isfinite(current_velocity)):
            current_velocity = np.zeros((self.node_count, 3), dtype=np.float64)

        if not bool(getattr(self.config, "velocity_prediction", True)):
            self.node_velocity_mps = np.zeros((self.node_count, 3), dtype=np.float64)
            self.previous_estimate_nodes = node_array.copy()
            return

        decay = float(np.clip(getattr(self.config, "velocity_decay", 0.90), 0.0, 1.0))
        if prediction_only or not measurement_used:
            self.node_velocity_mps = current_velocity * decay
            self.previous_estimate_nodes = node_array.copy()
            return

        visible = np.asarray(visible_nodes, dtype=bool).reshape(-1) if visible_nodes is not None else np.ones(self.node_count, dtype=bool)
        if len(visible) != self.node_count:
            visible = np.ones(self.node_count, dtype=bool)
        if not np.any(visible):
            self.node_velocity_mps = current_velocity * decay
            self.previous_estimate_nodes = node_array.copy()
            return

        previous = finite_node_chain(self.previous_estimate_nodes, self.node_count)
        if previous is None:
            self.previous_estimate_nodes = node_array.copy()
            self.node_velocity_mps = np.zeros((self.node_count, 3), dtype=np.float64)
            return

        measured_velocity = (node_array - previous) / max(float(dt), 1e-3)
        max_velocity = max(0.0, float(getattr(self.config, "max_velocity_mps", 0.40)))
        if max_velocity > 0.0:
            measured_velocity = clamp_node_vectors(measured_velocity, max_norm=max_velocity)
            current_velocity = clamp_node_vectors(current_velocity, max_norm=max_velocity)
        deadband = max(0.0, float(getattr(self.config, "velocity_deadband_mps", 0.0)))
        if deadband > 0.0:
            measured_velocity = zero_small_node_vectors(measured_velocity, min_norm=deadband)
            current_velocity = zero_small_node_vectors(current_velocity, min_norm=deadband)

        alpha = float(np.clip(getattr(self.config, "velocity_alpha", 0.35), 0.0, 1.0))
        updated_velocity = current_velocity * decay
        updated_velocity[visible] = (1.0 - alpha) * current_velocity[visible] + alpha * measured_velocity[visible]
        if max_velocity > 0.0:
            updated_velocity = clamp_node_vectors(updated_velocity, max_norm=max_velocity)
        if deadband > 0.0:
            updated_velocity = zero_small_node_vectors(updated_velocity, min_norm=deadband)
        self.node_velocity_mps = np.ascontiguousarray(updated_velocity, dtype=np.float64)
        self.previous_estimate_nodes = node_array.copy()

    def _maybe_resample(self, force=False, noise_scale=1.0):
        if self.weights is None or self.particles is None:
            return
        effective = self._effective_sample_size()
        threshold = float(self.config.resample_effective_ratio) * len(self.weights)
        if not force and effective >= threshold:
            return

        indices = self._systematic_resample()
        self.particles = self.particles[indices].copy()
        self.weights.fill(1.0 / len(self.weights))
        self._roughen(noise_scale=float(noise_scale))

    def _roughen(self, noise_scale=1.0):
        if self.particles is None or self.segment_length_m is None:
            return
        starts = self.particles[:, 0, :] + self.rng.normal(
            0.0,
            float(self.config.process_node_std_m) * 0.5 * noise_scale,
            (len(self.particles), 3),
        )
        directions = normalize_vectors(
            particle_directions(self.particles)
            + self.rng.normal(
                0.0,
                float(self.config.process_direction_std) * 0.5 * noise_scale,
                (len(self.particles), self.segment_count, 3),
            )
        )
        directions = smooth_particle_directions(
            directions,
            passes=int(getattr(self.config, "direction_smooth_passes", 1)),
        )
        self.particles = build_chains(starts, directions, self.segment_length_m)

    def _estimate(self, measurement_used, prediction_only):
        if self.particles is None or self.weights is None or self.segment_length_m is None:
            return None
        nodes = self._estimate_nodes()
        return CableParticleFilterResult(
            points_xyz=np.ascontiguousarray(nodes, dtype=np.float32),
            effective_sample_size=self._effective_sample_size(),
            measurement_used=bool(measurement_used),
            prediction_only=bool(prediction_only),
            lost_frames=int(self.lost_frames),
            motion_noise_scale=float(self.last_motion_noise_scale),
            segment_length_m=float(self.segment_length_m),
            prediction_step_m=float(self.last_prediction_step_m),
            measurement_point_count=int(self.last_measurement_point_count),
            visible_segments=self.last_visible_segments.copy(),
            visible_nodes=self.last_visible_nodes.copy(),
            measurement_proposal_ratio=float(self.last_measurement_proposal_ratio),
            estimate_mode=str(self.last_estimate_mode),
            top_particles=self._top_particles(),
            estimate_particle_count=int(self.last_estimate_particle_count),
        )

    def _estimate_nodes(self):
        estimate_mode = str(getattr(self.config, "estimate_mode", "top_clustered")).strip().lower()
        if estimate_mode not in {"auto", "map", "weighted", "top_weighted", "top_clustered"}:
            estimate_mode = "top_clustered"
        self.last_estimate_mode = estimate_mode

        if estimate_mode == "map":
            self.last_estimate_mode = "map"
            self.last_estimate_particle_count = 1
            return np.asarray(self.particles[int(np.argmax(self.weights))], dtype=np.float32)

        if estimate_mode == "top_clustered":
            return self._top_clustered_nodes()

        if estimate_mode == "top_weighted":
            return self._top_weighted_nodes()

        if estimate_mode == "auto":
            map_ratio = float(getattr(self.config, "map_estimate_effective_ratio", 0.0))
            if map_ratio > 0.0 and self.particles is not None and self.weights is not None and len(self.weights):
                effective_ratio = self._effective_sample_size() / max(len(self.weights), 1)
                if effective_ratio <= map_ratio:
                    self.last_estimate_mode = "map"
                    self.last_estimate_particle_count = 1
                    return np.asarray(self.particles[int(np.argmax(self.weights))], dtype=np.float32)
            self.last_estimate_mode = "weighted"

        starts = self.particles[:, 0, :]
        start = np.average(starts, axis=0, weights=self.weights)
        directions = np.average(particle_directions(self.particles), axis=0, weights=self.weights)
        directions = normalize_vectors(directions)
        self.last_estimate_particle_count = len(self.weights)
        return build_chain(start, directions, self.segment_length_m)

    def _top_weighted_nodes(self):
        if self.particles is None or self.weights is None or self.segment_length_m is None:
            return None
        indices = self._top_particle_indices(min_count=1)
        return self._weighted_nodes_from_indices(indices, estimate_mode="top_weighted")

    def _top_clustered_nodes(self):
        if self.particles is None or self.weights is None or self.segment_length_m is None:
            return None
        indices = self._top_particle_indices(min_count=1)
        if len(indices) == 0:
            return None

        reference = self.particles[indices[0], :, :3]
        candidates = self.particles[indices, :, :3]
        distances = np.mean(np.linalg.norm(candidates - reference[None, :, :], axis=2), axis=1)
        gate_m = max(0.0, float(getattr(self.config, "top_particle_cluster_gate_m", 0.030)))
        if gate_m > 0.0:
            clustered = indices[distances <= gate_m]
        else:
            clustered = indices
        if len(clustered) == 0:
            clustered = indices[:1]
        return self._weighted_nodes_from_indices(clustered, estimate_mode="top_clustered")

    def _weighted_nodes_from_indices(self, indices, estimate_mode):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(indices) == 0:
            return None
        top_weights = np.asarray(self.weights[indices], dtype=np.float64)
        total = float(np.sum(top_weights))
        if not np.isfinite(total) or total <= 1e-12:
            top_weights = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        else:
            top_weights = top_weights / total
        top_particles = self.particles[indices]
        start = np.average(top_particles[:, 0, :], axis=0, weights=top_weights)
        directions = np.average(particle_directions(top_particles), axis=0, weights=top_weights)
        directions = normalize_vectors(directions)
        self.last_estimate_mode = str(estimate_mode)
        self.last_estimate_particle_count = int(len(indices))
        return build_chain(start, directions, self.segment_length_m)

    def _top_particles(self):
        if self.particles is None or self.weights is None:
            return np.empty((0, self.node_count, 3), dtype=np.float32)
        indices = self._top_particle_indices(min_count=0)
        if len(indices) == 0:
            return np.empty((0, self.node_count, 3), dtype=np.float32)
        return np.ascontiguousarray(self.particles[indices, :, :3], dtype=np.float32)

    def _top_particle_indices(self, min_count=0):
        if self.weights is None:
            return np.empty(0, dtype=np.int64)
        count = max(0, int(getattr(self.config, "top_particle_count", 10)))
        count = max(int(min_count), count)
        if count == 0:
            return np.empty(0, dtype=np.int64)
        count = min(count, len(self.weights))
        if count >= len(self.weights):
            return np.argsort(self.weights)[::-1]
        candidates = np.argpartition(self.weights, -count)[-count:]
        return candidates[np.argsort(self.weights[candidates])[::-1]]

    def _measurement_points(self, measurement):
        if measurement is None:
            return None

        source_points = getattr(measurement, "source_points", None)
        if source_points is not None:
            points = valid_points(source_points)
            if len(points) >= int(self.config.min_measurement_points):
                return sample_points(points, int(self.config.measurement_max_points))

        points = valid_points(getattr(measurement, "points_xyz", measurement))
        if len(points) >= int(self.config.min_measurement_points):
            return sample_points(points, int(self.config.measurement_max_points))
        return None

    def _initial_nodes(self, measurement, measurement_points):
        candidate = getattr(measurement, "points_xyz", None)
        nodes = fit_polyline_segments(candidate, segment_count=self.segment_count) if candidate is not None else None
        if nodes is None:
            nodes = self._fit_measurement_point_chain(measurement_points)
        return nodes

    def _measurement_nodes(self, measurement, measurement_points, reference_nodes=None):
        self.last_endpoint_anchor_nodes = None
        candidate = valid_points(getattr(measurement, "points_xyz", None))
        if len(candidate) >= 2:
            nodes = fit_polyline_segments(candidate, segment_count=self.segment_count)
        else:
            nodes = self._fit_measurement_point_chain(
                measurement_points,
                reference_nodes=reference_nodes,
                allow_endpoint_fallback=self.lost_frames > 0,
            )
            self.measurement_update_count += 1
        if nodes is None or self.segment_length_m is None:
            return None
        if reference_nodes is not None:
            nodes = align_polyline_orientation(reference_nodes, nodes)
        endpoint_nodes = endpoint_nodes_from_chain(nodes)
        nodes = project_equal_length_chain(nodes, self.segment_length_m, self.node_count)
        if nodes is None:
            return None
        self.last_endpoint_anchor_nodes = endpoint_nodes
        self.last_ordered_measurement_nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        return np.ascontiguousarray(nodes, dtype=np.float64)

    def _current_endpoint_anchor_nodes(self, measurement_nodes=None):
        endpoint_nodes = endpoint_nodes_from_chain(self.last_endpoint_anchor_nodes)
        if endpoint_nodes is not None:
            return endpoint_nodes
        return endpoint_nodes_from_chain(measurement_nodes)

    def _anchor_particles_to_endpoints(self, endpoint_nodes):
        if (
            not bool(getattr(self.config, "endpoint_anchor_projection", True))
            or self.particles is None
            or self.segment_length_m is None
        ):
            return
        endpoint_nodes = endpoint_nodes_from_chain(endpoint_nodes)
        if endpoint_nodes is None:
            return
        passes = max(1, int(getattr(self.config, "endpoint_anchor_projection_passes", 1)))
        for _ in range(passes):
            anchored = anchor_particle_chains_to_endpoints(
                self.particles,
                endpoint_nodes,
                self.segment_length_m,
                gain=float(getattr(self.config, "endpoint_anchor_projection_gain", 1.0)),
            )
            if anchored.shape != self.particles.shape:
                return
            self.particles = anchored

    def _fit_measurement_point_chain(
        self,
        measurement_points,
        reference_nodes=None,
        allow_endpoint_fallback=True,
    ):
        if bool(getattr(self.config, "reference_ordering", True)) and reference_nodes is not None:
            nodes = fit_reference_ordered_point_cloud_chain(
                measurement_points,
                reference_nodes=reference_nodes,
                segment_count=self.segment_count,
                gate_m=float(getattr(self.config, "reference_ordering_gate_m", 0.08)),
                min_points=int(getattr(self.config, "min_measurement_points", 12)),
            )
            if nodes is not None:
                return nodes
            if not bool(allow_endpoint_fallback):
                return None

        nodes = fit_unordered_point_cloud_chain(
            measurement_points,
            segment_count=self.segment_count,
            endpoint_ordering=bool(getattr(self.config, "endpoint_ordering", True)),
            max_points=int(getattr(self.config, "ordering_max_points", 768)),
            knn=int(getattr(self.config, "ordering_knn", 10)),
        )
        if nodes is not None:
            return nodes
        if reference_nodes is not None:
            cached = valid_points(self.last_ordered_measurement_nodes)
            if len(cached) == self.node_count:
                return cached.copy()
        return None

    def _inject_measurement_proposals(self, measurement_nodes, current_nodes=None):
        if measurement_nodes is None or self.particles is None or self.weights is None or self.segment_length_m is None:
            self.last_measurement_proposal_ratio = 0.0
            return

        ratio = self._adaptive_measurement_proposal_ratio(measurement_nodes, current_nodes=current_nodes)
        self.last_measurement_proposal_ratio = float(ratio)
        if ratio <= 0.0:
            return
        total_count = len(self.particles)
        proposal_count = int(round(ratio * total_count))
        proposal_count = int(np.clip(proposal_count, 1, total_count))

        proposals = self._mixed_measurement_proposals(measurement_nodes, proposal_count, current_nodes=current_nodes)
        if len(proposals) == 0:
            self.last_measurement_proposal_ratio = 0.0
            return
        proposal_count = min(proposal_count, len(proposals))
        proposals = proposals[:proposal_count]
        replace_indices = self.rng.choice(total_count, size=proposal_count, replace=False)
        self.particles[replace_indices] = proposals

        self.weights *= 1.0 - ratio
        self.weights[replace_indices] = ratio / proposal_count
        total = float(np.sum(self.weights))
        if np.isfinite(total) and total > 1e-12:
            self.weights /= total
        else:
            self.weights.fill(1.0 / total_count)

    def _mixed_measurement_proposals(self, measurement_nodes, proposal_count, current_nodes=None):
        proposal_count = max(0, int(proposal_count))
        if proposal_count == 0 or self.segment_length_m is None:
            return np.empty((0, self.node_count, 3), dtype=np.float64)

        base_node_std = max(0.0, float(self.config.measurement_proposal_node_std_m))
        base_direction_std = max(0.0, float(self.config.measurement_proposal_direction_std))
        smooth_passes = int(getattr(self.config, "direction_smooth_passes", 1))
        current_nodes = finite_node_chain(current_nodes, self.node_count)

        current_fraction = float(np.clip(getattr(self.config, "measurement_proposal_current_fraction", 0.0), 0.0, 0.8))
        wide_fraction = float(np.clip(getattr(self.config, "measurement_proposal_wide_fraction", 0.0), 0.0, 0.8))
        if current_nodes is None:
            current_fraction = 0.0
        if current_fraction + wide_fraction > 0.9:
            scale = 0.9 / (current_fraction + wide_fraction)
            current_fraction *= scale
            wide_fraction *= scale

        current_count = int(round(proposal_count * current_fraction))
        wide_count = int(round(proposal_count * wide_fraction))
        if current_count + wide_count > proposal_count:
            overflow = current_count + wide_count - proposal_count
            wide_drop = min(wide_count, overflow)
            wide_count -= wide_drop
            current_count -= max(0, overflow - wide_drop)
        measurement_count = proposal_count - current_count - wide_count

        proposal_sets = []
        if measurement_count > 0:
            proposal_sets.append(
                sample_noisy_chains_around_nodes(
                    measurement_nodes,
                    measurement_count,
                    self.segment_length_m,
                    self.rng,
                    node_std_m=base_node_std,
                    direction_std=base_direction_std,
                    direction_smooth_passes=smooth_passes,
                )
            )
        if wide_count > 0:
            multiplier = max(1.0, float(getattr(self.config, "measurement_proposal_wide_std_multiplier", 3.0)))
            proposal_sets.append(
                sample_noisy_chains_around_nodes(
                    measurement_nodes,
                    wide_count,
                    self.segment_length_m,
                    self.rng,
                    node_std_m=base_node_std * multiplier,
                    direction_std=base_direction_std * multiplier,
                    direction_smooth_passes=smooth_passes,
                )
            )
        if current_count > 0 and current_nodes is not None:
            proposal_sets.append(
                sample_noisy_chains_around_nodes(
                    current_nodes,
                    current_count,
                    self.segment_length_m,
                    self.rng,
                    node_std_m=base_node_std * 0.75,
                    direction_std=base_direction_std * 0.75,
                    direction_smooth_passes=smooth_passes,
                )
            )

        proposal_sets = [item for item in proposal_sets if item.ndim == 3 and len(item)]
        if not proposal_sets:
            return np.empty((0, self.node_count, 3), dtype=np.float64)
        proposals = np.concatenate(proposal_sets, axis=0)
        if len(proposals) > proposal_count:
            proposals = proposals[:proposal_count]
        if len(proposals) < proposal_count:
            extra = sample_noisy_chains_around_nodes(
                measurement_nodes,
                proposal_count - len(proposals),
                self.segment_length_m,
                self.rng,
                node_std_m=base_node_std,
                direction_std=base_direction_std,
                direction_smooth_passes=smooth_passes,
            )
            if len(extra):
                proposals = np.concatenate([proposals, extra], axis=0)
        return np.ascontiguousarray(proposals, dtype=np.float64)

    def _adaptive_measurement_proposal_ratio(self, measurement_nodes, current_nodes=None):
        max_ratio = float(np.clip(self.config.measurement_proposal_ratio, 0.0, 1.0))
        if max_ratio <= 0.0 or measurement_nodes is None:
            return 0.0
        if self.lost_frames > 0:
            return max_ratio

        stable_ratio = float(np.clip(self.config.measurement_proposal_stable_ratio, 0.0, max_ratio))
        error_m = mean_node_distance(current_nodes, measurement_nodes)
        if not np.isfinite(error_m):
            return max_ratio

        start_error = max(0.0, float(self.config.measurement_proposal_start_error_m))
        full_error = max(start_error + 1e-6, float(self.config.measurement_proposal_full_error_m))
        if error_m <= start_error:
            return stable_ratio
        if error_m >= full_error:
            return max_ratio

        blend = (error_m - start_error) / (full_error - start_error)
        return stable_ratio + blend * (max_ratio - stable_ratio)

    def _associate_visible_points(self, measurement_points, candidate_nodes=None, reference_nodes=None):
        if reference_nodes is None and self.initialized:
            reference_nodes = self._estimate_nodes()
        if reference_nodes is None:
            visible_segments = np.ones(self.segment_count, dtype=bool)
            return measurement_points, None, visible_segments

        associated = associate_points_to_reference(
            measurement_points,
            reference_nodes,
            self.segment_count,
            gate=float(self.config.occlusion_assignment_max_distance_m),
            min_measurement_points=int(self.config.min_measurement_points),
            min_segment_points=int(self.config.min_segment_points),
        )
        if associated[0] is not None:
            return associated

        if candidate_nodes is not None:
            associated = associate_points_to_reference(
                measurement_points,
                candidate_nodes,
                self.segment_count,
                gate=float(self.config.occlusion_assignment_max_distance_m),
                min_measurement_points=int(self.config.min_measurement_points),
                min_segment_points=int(self.config.min_segment_points),
            )
            if associated[0] is not None:
                return associated

        visible_segments = np.zeros(self.segment_count, dtype=bool)
        return None, None, visible_segments

    def _set_visibility(self, visible_segments):
        visible_segments = np.asarray(visible_segments, dtype=bool).reshape(-1)
        if len(visible_segments) != self.segment_count:
            visible_segments = np.zeros(self.segment_count, dtype=bool)
        self.last_visible_segments = visible_segments.copy()
        self.last_visible_nodes = node_visibility_from_segments(visible_segments, self.node_count)

    def _prediction_only_after_update_drop(self):
        self.lost_frames += 1
        self._set_visibility(np.zeros(self.segment_count, dtype=bool))
        if self.lost_frames > int(self.config.max_prediction_frames):
            self.initialized = False
            return None
        result = self._estimate(measurement_used=False, prediction_only=True)
        self._maybe_resample(force=self.lost_frames > 1, noise_scale=self.last_motion_noise_scale)
        return result

    def _systematic_resample(self):
        count = len(self.weights)
        positions = (self.rng.random() + np.arange(count)) / count
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions, side="left")

    def _effective_sample_size(self):
        return float(1.0 / max(np.sum(self.weights * self.weights), 1e-12))


def associate_points_to_reference(
    measurement_points,
    reference_nodes,
    segment_count,
    gate,
    min_measurement_points,
    min_segment_points,
):
    reference_nodes = valid_points(reference_nodes)
    if len(reference_nodes) < 2:
        visible_segments = np.zeros(int(segment_count), dtype=bool)
        return None, None, visible_segments

    measurement_points = valid_points(measurement_points)
    if len(measurement_points) == 0:
        visible_segments = np.zeros(int(segment_count), dtype=bool)
        return None, None, visible_segments

    segment_indices, distances = assign_points_to_nearest_segments(measurement_points, reference_nodes)
    gate = float(gate)
    if gate > 0.0:
        keep = distances <= gate
    else:
        keep = np.ones(len(distances), dtype=bool)

    if int(np.count_nonzero(keep)) < int(min_measurement_points):
        visible_segments = np.zeros(int(segment_count), dtype=bool)
        return None, None, visible_segments

    segment_indices = segment_indices[keep]
    measurement_points = measurement_points[keep]
    visible_segments = segment_visibility(
        segment_indices,
        int(segment_count),
        min_points=int(min_segment_points),
    )
    return measurement_points, segment_indices, visible_segments


def filtered_cable_estimate(measurement, result):
    if result is None:
        return None
    source_points = (
        np.empty((0, 3), dtype=np.float32)
        if measurement is None
        else np.asarray(getattr(measurement, "source_points", np.empty((0, 3))), dtype=np.float32)
    )
    base_method = "no measurement" if measurement is None else str(getattr(measurement, "method", "cable measurement"))
    mode = "prediction" if result.prediction_only else "update"
    return CableEstimate3D(
        points_xyz=np.asarray(result.points_xyz, dtype=np.float32),
        source_points=source_points,
        residual_m=polyline_residual(source_points, result.points_xyz) if len(source_points) else 0.0,
        method=(
            f"raw fixed-length segment particle filter {mode} {result.estimate_mode} estimate | {base_method} | "
            f"segment_length={result.segment_length_m:.4f}m ess={result.effective_sample_size:.0f} "
            f"support={result.measurement_point_count} lost={result.lost_frames}"
        ),
    )


def fixed_segment_length_from_config(config, segment_count=None):
    cable_length = float(getattr(config, "cable_length_m", 0.0))
    if cable_length > 0.0 and segment_count is not None:
        length_scale = float(np.clip(getattr(config, "cable_length_scale", 1.0), 0.50, 1.20))
        return (cable_length * length_scale) / max(1, int(segment_count))
    length = float(getattr(config, "segment_length_m", 0.0))
    return length if length > 0.0 else None


def fit_reference_ordered_point_cloud_chain(
    points_xyz,
    reference_nodes,
    segment_count=12,
    gate_m=0.08,
    min_points=12,
):
    points = valid_points(points_xyz)
    reference = valid_points(reference_nodes)
    if len(points) < 2 or len(reference) < 2:
        return None

    _segment_indices, distances, arclength = assign_points_to_reference_arclength(points, reference)
    if len(distances) != len(points):
        return None

    keep = np.isfinite(distances) & np.isfinite(arclength)
    gate_m = float(gate_m)
    if gate_m > 0.0:
        keep &= distances <= gate_m
    if int(np.count_nonzero(keep)) < max(2, int(min_points)):
        return None

    centerline = binned_centerline_from_ordered_points(
        points[keep],
        arclength[keep],
        segment_count=segment_count,
    )
    if centerline is None or len(centerline) < 2:
        return None

    nodes = fit_polyline_segments(centerline, segment_count=segment_count)
    if nodes is None:
        return None
    return align_polyline_orientation(reference, nodes)


def fit_unordered_point_cloud_chain(
    points_xyz,
    segment_count=12,
    endpoint_ordering=True,
    max_points=768,
    knn=10,
):
    points = valid_points(points_xyz)
    if len(points) < 2:
        return None

    if bool(endpoint_ordering):
        nodes = fit_endpoint_ordered_point_cloud_chain(
            points,
            segment_count=segment_count,
            max_points=max_points,
            knn=knn,
        )
        if nodes is not None:
            return nodes

    return fit_pca_binned_point_cloud_chain(points, segment_count=segment_count)


def fit_endpoint_ordered_point_cloud_chain(points_xyz, segment_count=12, max_points=768, knn=10):
    points = valid_points(points_xyz)
    if len(points) < 2:
        return None

    max_points = max(max(64, int(segment_count) * 16), int(max_points))
    points = sample_points(points, max_points)
    if len(points) < 2:
        return None

    neighbors, weights = knn_graph(points, k=knn)
    if not neighbors:
        return None

    start_hint, end_hint = pca_endpoint_hints(points)
    dist0, _parents0 = dijkstra_graph(neighbors, weights, start_hint)
    start = farthest_reachable_index(dist0)
    if start < 0:
        start = int(start_hint)
    dist1, parents1 = dijkstra_graph(neighbors, weights, start)
    end = farthest_reachable_index(dist1)
    if end < 0 or end == start:
        end = int(end_hint)

    path_indices = reconstruct_index_path(parents1, start, end)
    if len(path_indices) < 2:
        return None

    path_points = points[np.asarray(path_indices, dtype=np.int64)]
    path_points = remove_large_centerline_jumps(path_points)
    path_points = smooth_centerline_points(path_points)
    if len(path_points) < 2:
        return None
    return fit_polyline_segments(path_points, segment_count=segment_count)


def fit_pca_binned_point_cloud_chain(points_xyz, segment_count=12):
    points = valid_points(points_xyz)
    if len(points) < 2:
        return None

    points = sample_points(points, max(256, int(segment_count) * 96))
    center = np.mean(points, axis=0)
    centered = points - center[None, :]
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / axis_norm

    projection = centered @ axis
    order = np.argsort(projection)
    ordered_points = points[order]
    ordered_projection = projection[order]
    centerline = binned_centerline_from_ordered_points(
        ordered_points,
        ordered_projection,
        segment_count=segment_count,
    )
    if centerline is None or len(centerline) < 2:
        return None
    return fit_polyline_segments(centerline, segment_count=segment_count)


def pca_endpoint_hints(points):
    points = valid_points(points)
    if len(points) < 2:
        return 0, 0
    centered = points - np.mean(points, axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    projection = centered @ axis
    return int(np.argmin(projection)), int(np.argmax(projection))


def knn_graph(points, k=10):
    points = valid_points(points)
    count = len(points)
    if count < 2:
        return [], []

    k = int(np.clip(int(k), 1, max(1, count - 1)))
    deltas = points[:, None, :] - points[None, :, :]
    squared = np.sum(deltas * deltas, axis=2)
    np.fill_diagonal(squared, np.inf)
    neighbor_idx = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]

    adjacency = [dict() for _ in range(count)]
    for index in range(count):
        for neighbor in neighbor_idx[index]:
            neighbor = int(neighbor)
            weight = float(np.sqrt(squared[index, neighbor]))
            if not np.isfinite(weight):
                continue
            current = adjacency[index].get(neighbor)
            if current is None or weight < current:
                adjacency[index][neighbor] = weight
                adjacency[neighbor][index] = weight

    neighbors = []
    weights = []
    for item in adjacency:
        ordered = sorted(item.items(), key=lambda pair: pair[1])
        neighbors.append([int(index) for index, _weight in ordered])
        weights.append([float(weight) for _index, weight in ordered])
    return neighbors, weights


def dijkstra_graph(neighbors, weights, start):
    count = len(neighbors)
    start = int(start)
    distances = np.full(count, np.inf, dtype=np.float64)
    parents = np.full(count, -1, dtype=np.int64)
    if count == 0 or start < 0 or start >= count:
        return distances, parents

    distances[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for neighbor, weight in zip(neighbors[node], weights[node]):
            candidate = distance + float(weight)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(queue, (candidate, int(neighbor)))
    return distances, parents


def farthest_reachable_index(distances):
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    finite = np.flatnonzero(np.isfinite(distances))
    if len(finite) == 0:
        return -1
    return int(finite[np.argmax(distances[finite])])


def reconstruct_index_path(parents, start, end):
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    start = int(start)
    end = int(end)
    if start < 0 or end < 0 or start >= len(parents) or end >= len(parents):
        return []
    path = [end]
    current = end
    seen = {end}
    while current != start:
        current = int(parents[current])
        if current < 0 or current in seen:
            return []
        seen.add(current)
        path.append(current)
    path.reverse()
    return path


def binned_centerline_from_ordered_points(points, projection, segment_count=12):
    points = valid_points(points)
    projection = np.asarray(projection, dtype=np.float64).reshape(-1)
    if len(points) < 2 or len(projection) != len(points):
        return None

    order = np.argsort(projection)
    points = points[order]
    projection = projection[order]

    # Collapse the thick unordered cable cloud into robust slice medians before
    # resampling. This avoids fitting a long zigzag through cable-mask thickness
    # and ZED depth jitter.
    bin_count = int(np.clip(max(int(segment_count) * 6, int(segment_count) + 1), 4, len(points)))
    edges = np.linspace(0, len(points), bin_count + 1, dtype=np.int64)
    centers = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        chunk = points[start:end]
        if len(chunk) == 0:
            continue
        centers.append(np.median(chunk, axis=0))

    if len(centers) < 2:
        return None

    centers = np.asarray(centers, dtype=np.float64)
    centers = remove_large_centerline_jumps(centers)
    centers = smooth_centerline_points(centers)
    if len(centers) < 2:
        return None
    return np.ascontiguousarray(centers, dtype=np.float32)


def remove_large_centerline_jumps(points):
    points = valid_points(points)
    if len(points) < 3:
        return points
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    finite = deltas[np.isfinite(deltas) & (deltas > 1e-9)]
    if len(finite) == 0:
        return points
    median = float(np.median(finite))
    if median <= 1e-9:
        return points
    keep = np.ones(len(points), dtype=bool)
    # A single bad depth slice should not stretch the whole fixed-length chain.
    for index, distance in enumerate(deltas):
        if distance > 6.0 * median and 0 < index + 1 < len(points) - 1:
            keep[index + 1] = False
    cleaned = points[keep]
    return cleaned if len(cleaned) >= 2 else points


def smooth_centerline_points(points):
    points = valid_points(points)
    if len(points) < 5:
        return points
    smoothed = points.copy()
    smoothed[1:-1] = 0.25 * points[:-2] + 0.50 * points[1:-1] + 0.25 * points[2:]
    return smoothed


def estimate_equal_segment_length(nodes, segment_count):
    lengths = segment_lengths(nodes)
    lengths = lengths[np.isfinite(lengths)]
    if len(lengths) == 0:
        return 0.05
    total = float(np.sum(lengths))
    if not np.isfinite(total) or total <= 1e-9:
        return 0.05
    node_points = valid_points(nodes)
    if len(node_points) >= 2:
        extent = float(np.linalg.norm(np.max(node_points, axis=0) - np.min(node_points, axis=0)))
        if np.isfinite(extent) and extent > 1e-6 and total > 3.0 * extent:
            total = 3.0 * extent
    return total / max(1, int(segment_count))


def project_equal_length_chain(nodes, segment_length_m, node_count=None):
    nodes = valid_points(nodes)
    if len(nodes) < 2:
        return None
    node_count = len(nodes) if node_count is None else max(2, int(node_count))
    nodes = fit_polyline_segments(nodes, segment_count=node_count - 1)
    directions = chain_directions(nodes)
    return build_chain(nodes[0], directions, float(segment_length_m))


def endpoint_nodes_from_chain(nodes):
    return endpoint_pair_array(nodes, allow_partial=False)


def endpoint_pair_array(nodes, allow_partial=False):
    if nodes is None:
        return None
    points = np.asarray(nodes, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 2:
        return None
    if len(points) == 2:
        endpoints = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    else:
        endpoints = np.ascontiguousarray([points[0, :3], points[-1, :3]], dtype=np.float64)
    finite = np.all(np.isfinite(endpoints), axis=1)
    if allow_partial:
        return endpoints if np.any(finite) else None
    return endpoints if np.all(finite) else None


def mask_endpoint_nodes_by_visibility(endpoint_nodes, visible_segments, segment_count):
    endpoints = endpoint_pair_array(endpoint_nodes, allow_partial=False)
    if endpoints is None:
        return None
    visible = np.asarray(visible_segments, dtype=bool).reshape(-1)
    if len(visible) != int(segment_count):
        return endpoints
    masked = endpoints.copy()
    if len(visible) and not bool(visible[0]):
        masked[0, :] = np.nan
    if len(visible) and not bool(visible[-1]):
        masked[-1, :] = np.nan
    return masked if np.any(np.all(np.isfinite(masked), axis=1)) else None


def endpoint_anchor_scores(particles, endpoint_nodes):
    particles = valid_particles(particles)
    endpoint_nodes = endpoint_pair_array(endpoint_nodes, allow_partial=True)
    if len(particles) == 0:
        return np.empty(0, dtype=np.float64)
    if endpoint_nodes is None:
        return np.zeros(len(particles), dtype=np.float64)

    scores = np.zeros(len(particles), dtype=np.float64)
    count = np.zeros(len(particles), dtype=np.float64)
    if np.all(np.isfinite(endpoint_nodes[0])):
        start_delta = particles[:, 0, :3] - endpoint_nodes[0][None, :]
        scores += np.sum(start_delta * start_delta, axis=1)
        count += 1.0
    if np.all(np.isfinite(endpoint_nodes[-1])):
        end_delta = particles[:, -1, :3] - endpoint_nodes[-1][None, :]
        scores += np.sum(end_delta * end_delta, axis=1)
        count += 1.0
    valid = count > 0.0
    scores[valid] /= count[valid]
    return scores


def anchor_particle_chains_to_endpoints(particles, endpoint_nodes, segment_length_m, gain=1.0):
    original = np.asarray(particles, dtype=np.float64)
    particles = valid_particles(original)
    endpoint_nodes = endpoint_pair_array(endpoint_nodes, allow_partial=True)
    if len(particles) == 0 or endpoint_nodes is None:
        return original

    gain = float(np.clip(gain, 0.0, 1.0))
    if gain <= 0.0:
        return particles
    node_count = particles.shape[1]
    progress = np.linspace(0.0, 1.0, node_count, dtype=np.float64)
    start_delta = np.zeros((len(particles), 3), dtype=np.float64)
    end_delta = np.zeros((len(particles), 3), dtype=np.float64)
    if np.all(np.isfinite(endpoint_nodes[0])):
        start_delta = gain * (endpoint_nodes[0][None, :] - particles[:, 0, :3])
    if np.all(np.isfinite(endpoint_nodes[-1])):
        end_delta = gain * (endpoint_nodes[-1][None, :] - particles[:, -1, :3])
    warped = (
        particles
        + (1.0 - progress)[None, :, None] * start_delta[:, None, :]
        + progress[None, :, None] * end_delta[:, None, :]
    )
    directions = particle_directions(warped)
    starts = warped[:, 0, :3]
    return build_chains(starts, directions, float(segment_length_m))


def build_chain(start, directions, segment_length_m):
    start = np.asarray(start, dtype=np.float64).reshape(3)
    directions = normalize_vectors(directions)
    nodes = np.empty((len(directions) + 1, 3), dtype=np.float64)
    nodes[0] = start
    if len(directions):
        nodes[1:] = start[None, :] + np.cumsum(float(segment_length_m) * directions, axis=0)
    return nodes.astype(np.float32)


def build_chains(starts, directions, segment_length_m):
    starts = np.asarray(starts, dtype=np.float64)
    directions = normalize_vectors(directions)
    output = np.empty((len(starts), directions.shape[1] + 1, 3), dtype=np.float64)
    output[:, 0, :] = starts
    if directions.shape[1]:
        output[:, 1:, :] = starts[:, None, :] + np.cumsum(float(segment_length_m) * directions, axis=1)
    return np.ascontiguousarray(output, dtype=np.float64)


def sample_noisy_chains_around_nodes(
    nodes,
    count,
    segment_length_m,
    rng,
    node_std_m=0.020,
    direction_std=0.080,
    direction_smooth_passes=1,
):
    nodes = valid_points(nodes)
    count = max(0, int(count))
    if count == 0 or len(nodes) < 2:
        return np.empty((0, 0, 3), dtype=np.float64)

    directions = chain_directions(nodes)
    starts = nodes[0][None, :] + rng.normal(0.0, max(0.0, float(node_std_m)), (count, 3))
    noisy_directions = directions[None, :, :] + rng.normal(
        0.0,
        max(0.0, float(direction_std)),
        (count, len(directions), 3),
    )
    noisy_directions = smooth_particle_directions(
        normalize_vectors(noisy_directions),
        passes=int(direction_smooth_passes),
    )
    return build_chains(starts, noisy_directions, float(segment_length_m))


def chain_directions(nodes):
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) < 2:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    directions = np.diff(nodes[:, :3], axis=0)
    return normalize_vectors(directions)


def particle_directions(particles):
    return normalize_vectors(np.diff(np.asarray(particles, dtype=np.float64), axis=1))


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    output = vectors.copy()
    norms = np.linalg.norm(output, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-12
    output[valid] = output[valid] / norms[valid]
    if np.any(~valid):
        output[~valid] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return output


def smooth_particle_directions(directions, passes=1):
    directions = normalize_vectors(directions)
    passes = max(0, int(passes))
    if directions.ndim != 3 or directions.shape[1] < 3 or passes == 0:
        return directions
    smoothed = directions.copy()
    for _ in range(passes):
        updated = smoothed.copy()
        updated[:, 1:-1, :] = 0.25 * smoothed[:, :-2, :] + 0.50 * smoothed[:, 1:-1, :] + 0.25 * smoothed[:, 2:, :]
        smoothed = normalize_vectors(updated)
    return smoothed


def valid_points(points, dtype=np.float64):
    dtype = np.dtype(dtype)
    points = np.asarray(points, dtype=dtype)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=dtype)
    points = points[:, :3]
    return np.ascontiguousarray(points[np.all(np.isfinite(points), axis=1)], dtype=dtype)


def sample_points(points, max_points, dtype=np.float64):
    points = valid_points(points, dtype=dtype)
    max_points = max(0, int(max_points))
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return np.ascontiguousarray(points, dtype=np.dtype(dtype))


def assign_points_to_nearest_segments(points, nodes):
    distances = point_to_all_segment_distances(points, nodes)
    if distances.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    segment_indices = np.argmin(distances, axis=1).astype(np.int64)
    best_distances = distances[np.arange(len(segment_indices)), segment_indices]
    return segment_indices, best_distances


def assign_points_to_reference_arclength(points, nodes):
    points = valid_points(points)
    nodes = valid_points(nodes)
    if len(points) == 0 or len(nodes) < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    segment_vectors = nodes[1:] - nodes[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    length_sq = np.sum(segment_vectors * segment_vectors, axis=1)
    safe_length_sq = np.maximum(length_sq, 1e-12)
    point_delta = points[:, None, :] - nodes[:-1][None, :, :]
    projection_fraction = np.sum(point_delta * segment_vectors[None, :, :], axis=2) / safe_length_sq[None, :]
    projection_fraction = np.clip(projection_fraction, 0.0, 1.0)
    projection_fraction[:, length_sq <= 1e-12] = 0.0
    projected = nodes[:-1][None, :, :] + projection_fraction[:, :, None] * segment_vectors[None, :, :]
    squared = np.sum((points[:, None, :] - projected) ** 2, axis=2)

    best_segments = np.argmin(squared, axis=1).astype(np.int64)
    rows = np.arange(len(points))
    best_squared = squared[rows, best_segments]
    best_arclength = cumulative[best_segments] + projection_fraction[rows, best_segments] * segment_lengths[best_segments]
    return best_segments, np.sqrt(best_squared), best_arclength


def particle_distance_scores(
    points,
    particles,
    outlier_distance=np.inf,
    segment_indices=None,
    segment_window=0,
    backend="auto",
    chunk_points=512,
):
    device = torch_scoring_device(backend)
    scoring_dtype = np.float32 if device is not None else np.float64
    points = valid_points(points, dtype=scoring_dtype)
    particles = valid_particles(particles, dtype=scoring_dtype)
    if len(points) == 0 or len(particles) == 0 or particles.shape[1] < 2:
        return np.full(len(particles), np.inf, dtype=np.float64)

    segment_indices = valid_segment_indices(segment_indices, len(points), particles.shape[1] - 1)
    if segment_indices is not None:
        if device is not None:
            torch_scores = particle_assigned_segment_distance_scores_torch(
                points,
                particles,
                segment_indices,
                outlier_distance=outlier_distance,
                segment_window=segment_window,
                chunk_points=chunk_points,
                device=device,
            )
            if torch_scores is not None:
                return torch_scores
        return particle_assigned_segment_distance_scores(
            points,
            particles,
            segment_indices,
            outlier_distance=outlier_distance,
            segment_window=segment_window,
            chunk_points=chunk_points,
        )

    if device is not None:
        torch_scores = particle_distance_scores_torch(
            points,
            particles,
            outlier_distance=outlier_distance,
            backend=backend,
            chunk_points=chunk_points,
            device=device,
        )
        if torch_scores is not None:
            return torch_scores

    all_squared_distances = point_to_particle_segment_squared_distances(points, particles[:, :-1, :3], particles[:, 1:, :3])
    if all_squared_distances.size:
        squared_distances = np.min(all_squared_distances, axis=1)
    else:
        squared_distances = np.empty((len(particles), 0), dtype=np.float64)
    if squared_distances.size == 0:
        return np.full(len(particles), np.inf, dtype=np.float64)
    if np.isfinite(float(outlier_distance)):
        squared_distances = np.minimum(squared_distances, float(outlier_distance) * float(outlier_distance))
    return np.mean(squared_distances, axis=1)


def valid_segment_indices(segment_indices, point_count, segment_count):
    if segment_indices is None:
        return None
    indices = np.asarray(segment_indices, dtype=np.int64).reshape(-1)
    if len(indices) != int(point_count):
        return None
    if int(segment_count) <= 0:
        return None
    return np.ascontiguousarray(np.clip(indices, 0, int(segment_count) - 1), dtype=np.int64)


def segment_index_windows(segment_indices, segment_count, segment_window=0):
    indices = np.asarray(segment_indices, dtype=np.int64).reshape(-1)
    segment_count = max(1, int(segment_count))
    window = max(0, int(segment_window))
    offsets = np.arange(-window, window + 1, dtype=np.int64)
    windows = indices[:, None] + offsets[None, :]
    return np.ascontiguousarray(np.clip(windows, 0, segment_count - 1), dtype=np.int64)


def particle_assigned_segment_distance_scores(
    points,
    particles,
    segment_indices,
    outlier_distance=np.inf,
    segment_window=0,
    chunk_points=512,
):
    points = valid_points(points)
    particles = valid_particles(particles)
    segment_indices = valid_segment_indices(segment_indices, len(points), particles.shape[1] - 1)
    if len(points) == 0 or len(particles) == 0 or segment_indices is None:
        return np.full(len(particles), np.inf, dtype=np.float64)

    segment_count = particles.shape[1] - 1
    score_sum = np.zeros(len(particles), dtype=np.float64)
    total_points = 0
    chunk_points = max(1, int(chunk_points))
    for start_index in range(0, len(points), chunk_points):
        chunk = points[start_index : start_index + chunk_points, :3]
        index_window = segment_index_windows(
            segment_indices[start_index : start_index + len(chunk)],
            segment_count,
            segment_window=segment_window,
        )
        starts = particles[:, index_window, :3]
        ends = particles[:, index_window + 1, :3]
        segment = ends - starts
        length_sq = np.sum(segment * segment, axis=3)
        point_delta = chunk[None, :, None, :] - starts
        t = np.sum(point_delta * segment, axis=3) / np.maximum(length_sq, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        projection = starts + t[:, :, :, None] * segment
        delta = chunk[None, :, None, :] - projection
        squared = np.sum(delta * delta, axis=3)
        best = np.min(squared, axis=2)
        if np.isfinite(float(outlier_distance)):
            best = np.minimum(best, float(outlier_distance) * float(outlier_distance))
        score_sum += np.sum(best, axis=1)
        total_points += len(chunk)

    if total_points <= 0:
        return np.full(len(particles), np.inf, dtype=np.float64)
    return score_sum / float(total_points)


def particle_assigned_segment_distance_scores_torch(
    points,
    particles,
    segment_indices,
    outlier_distance=np.inf,
    segment_window=0,
    chunk_points=512,
    device=None,
):
    device = torch_scoring_device("auto") if device is None else device
    if device is None:
        return None

    try:
        with torch.inference_mode():
            points_t = torch.as_tensor(points[:, :3], dtype=torch.float32, device=device)
            particles_t = torch.as_tensor(particles[:, :, :3], dtype=torch.float32, device=device)
            indices_t = torch.as_tensor(segment_indices, dtype=torch.long, device=device)
            starts_all = particles_t[:, :-1, :]
            ends_all = particles_t[:, 1:, :]
            segment_count = starts_all.shape[1]
            window = max(0, int(segment_window))
            offsets = torch.arange(-window, window + 1, dtype=torch.long, device=device)

            best_chunks = []
            chunk_points = max(1, int(chunk_points))
            particle_count = int(particles_t.shape[0])
            for start_index in range(0, points_t.shape[0], chunk_points):
                chunk = points_t[start_index : start_index + chunk_points]
                chunk_indices = indices_t[start_index : start_index + chunk.shape[0]]
                index_window = torch.clamp(chunk_indices[:, None] + offsets[None, :], 0, segment_count - 1)
                gather_index = index_window[None, :, :, None].expand(particle_count, -1, -1, 3)
                starts = torch.gather(
                    starts_all[:, None, :, :].expand(-1, chunk.shape[0], -1, -1),
                    2,
                    gather_index,
                )
                ends = torch.gather(
                    ends_all[:, None, :, :].expand(-1, chunk.shape[0], -1, -1),
                    2,
                    gather_index,
                )
                segment = ends - starts
                length_sq = torch.sum(segment * segment, dim=3).clamp_min(1e-12)
                point_delta = chunk[None, :, None, :] - starts
                t = torch.sum(point_delta * segment, dim=3) / length_sq
                t = torch.clamp(t, 0.0, 1.0)
                projection = starts + t[:, :, :, None] * segment
                delta = chunk[None, :, None, :] - projection
                squared = torch.sum(delta * delta, dim=3)
                best = torch.min(squared, dim=2).values
                best_chunks.append(best)

            if not best_chunks:
                return np.full(len(particles), np.inf, dtype=np.float64)
            squared_distances = torch.cat(best_chunks, dim=1)
            outlier_distance = float(outlier_distance)
            if np.isfinite(outlier_distance):
                squared_distances = torch.clamp(squared_distances, max=outlier_distance * outlier_distance)
            scores = torch.mean(squared_distances, dim=1)
            return scores.detach().cpu().numpy().astype(np.float64, copy=False)
    except RuntimeError:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return None


def particle_distance_scores_torch(
    points,
    particles,
    outlier_distance=np.inf,
    backend="auto",
    chunk_points=512,
    device=None,
):
    device = torch_scoring_device(backend) if device is None else device
    if device is None:
        return None

    try:
        with torch.inference_mode():
            points_t = torch.as_tensor(points[:, :3], dtype=torch.float32, device=device)
            particles_t = torch.as_tensor(particles[:, :, :3], dtype=torch.float32, device=device)
            starts = particles_t[:, :-1, :]
            ends = particles_t[:, 1:, :]
            segment = ends - starts
            length_sq = torch.sum(segment * segment, dim=2).clamp_min(1e-12)

            best_chunks = []
            chunk_points = max(1, int(chunk_points))
            for start_index in range(0, points_t.shape[0], chunk_points):
                chunk = points_t[start_index : start_index + chunk_points]
                point_delta = chunk[None, None, :, :] - starts[:, :, None, :]
                t = torch.sum(point_delta * segment[:, :, None, :], dim=3) / length_sq[:, :, None]
                t = torch.clamp(t, 0.0, 1.0)
                projection = starts[:, :, None, :] + t[:, :, :, None] * segment[:, :, None, :]
                delta = chunk[None, None, :, :] - projection
                squared = torch.sum(delta * delta, dim=3)
                best, nearest = torch.min(squared, dim=1)
                best_chunks.append(best)

            if not best_chunks:
                return np.full(len(particles), np.inf, dtype=np.float64)
            squared_distances = torch.cat(best_chunks, dim=1)

            outlier_distance = float(outlier_distance)
            if np.isfinite(outlier_distance):
                squared_distances = torch.clamp(squared_distances, max=outlier_distance * outlier_distance)
            scores = torch.mean(squared_distances, dim=1)

            return scores.detach().cpu().numpy().astype(np.float64, copy=False)
    except RuntimeError:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return None


def torch_scoring_device(backend):
    backend = str(backend or "auto").lower()
    if backend == "cpu":
        return None
    if torch is None:
        return None
    if backend in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return None


def point_to_particle_segment_squared_distances(points, starts, ends):
    points = np.asarray(points, dtype=np.float64)[:, :3]
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.ndim != 3 or ends.shape != starts.shape or starts.shape[2] < 3:
        return np.empty((0, 0, 0), dtype=np.float64)

    segment = ends[:, :, :3] - starts[:, :, :3]
    length_sq = np.sum(segment * segment, axis=2)
    point_delta = points[None, None, :, :] - starts[:, :, None, :3]
    denom = np.maximum(length_sq[:, :, None], 1e-12)
    t = np.clip(np.sum(point_delta * segment[:, :, None, :], axis=3) / denom, 0.0, 1.0)
    projection = starts[:, :, None, :3] + t[:, :, :, None] * segment[:, :, None, :]
    delta = points[None, None, :, :] - projection
    return np.sum(delta * delta, axis=3)


def point_to_all_segment_distances(points, nodes):
    points = valid_points(points)
    nodes = valid_points(nodes)
    if len(points) == 0 or len(nodes) < 2:
        return np.empty((0, 0), dtype=np.float64)

    starts = nodes[:-1]
    segment = nodes[1:] - starts
    length_sq = np.sum(segment * segment, axis=1)
    safe_length_sq = np.maximum(length_sq, 1e-12)
    point_delta = points[:, None, :] - starts[None, :, :]
    t = np.sum(point_delta * segment[None, :, :], axis=2) / safe_length_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    t[:, length_sq <= 1e-12] = 0.0
    projection = starts[None, :, :] + t[:, :, None] * segment[None, :, :]
    return np.linalg.norm(points[:, None, :] - projection, axis=2)


def valid_particles(particles, dtype=np.float64):
    dtype = np.dtype(dtype)
    particles = np.asarray(particles, dtype=dtype)
    if particles.ndim != 3 or particles.shape[1] < 2 or particles.shape[2] < 3:
        return np.empty((0, 0, 3), dtype=dtype)
    return np.ascontiguousarray(particles[:, :, :3], dtype=dtype)


def point_to_segment_distances(points, start, end):
    points = valid_points(points)
    start = np.asarray(start, dtype=np.float64).reshape(1, 3)
    end = np.asarray(end, dtype=np.float64).reshape(1, 3)
    segment = end - start
    length_sq = float(np.sum(segment * segment))
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    if length_sq <= 1e-12:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(((points - start) @ segment.reshape(3)) / length_sq, 0.0, 1.0)
    projection = start + t[:, None] * segment
    return np.linalg.norm(points - projection, axis=1)


def segment_visibility(segment_indices, segment_count, min_points=4):
    segment_indices = np.asarray(segment_indices, dtype=np.int64).reshape(-1)
    visible = np.zeros(int(segment_count), dtype=bool)
    if len(segment_indices) == 0 or segment_count <= 0:
        return visible
    counts = np.bincount(np.clip(segment_indices, 0, int(segment_count) - 1), minlength=int(segment_count))
    visible[:] = counts[: int(segment_count)] >= max(1, int(min_points))
    return visible


def node_visibility_from_segments(visible_segments, node_count):
    visible_segments = np.asarray(visible_segments, dtype=bool).reshape(-1)
    visible_nodes = np.zeros(int(node_count), dtype=bool)
    if len(visible_segments) == 0 or node_count <= 0:
        return visible_nodes
    for index, visible in enumerate(visible_segments[: max(0, int(node_count) - 1)]):
        if visible:
            visible_nodes[index] = True
            visible_nodes[index + 1] = True
    return visible_nodes


def merge_visible_measurement_nodes(measurement_nodes, reference_nodes, visible_segments):
    measurement = finite_node_chain(measurement_nodes, len(measurement_nodes) if measurement_nodes is not None else 0)
    reference = finite_node_chain(reference_nodes, len(reference_nodes) if reference_nodes is not None else 0)
    if measurement is None:
        return reference
    if reference is None or reference.shape != measurement.shape:
        return measurement
    visible_nodes = node_visibility_from_segments(visible_segments, len(measurement))
    if len(visible_nodes) != len(measurement):
        return measurement
    merged = reference.copy()
    merged[visible_nodes] = measurement[visible_nodes]
    return np.ascontiguousarray(merged, dtype=np.float64)


def align_polyline_orientation(reference_nodes, candidate_nodes):
    reference = np.asarray(reference_nodes, dtype=np.float64)
    candidate = np.asarray(candidate_nodes, dtype=np.float64)
    if reference.shape != candidate.shape:
        return candidate
    direct = float(np.mean(np.sum((reference - candidate) ** 2, axis=1)))
    reversed_error = float(np.mean(np.sum((reference - candidate[::-1]) ** 2, axis=1)))
    if reversed_error < direct:
        return candidate[::-1].copy()
    return candidate


def mean_node_distance(reference_nodes, candidate_nodes):
    reference = valid_points(reference_nodes)
    candidate = valid_points(candidate_nodes)
    if reference.shape != candidate.shape or len(reference) == 0:
        return np.inf
    distances = np.linalg.norm(reference - candidate, axis=1)
    if len(distances) == 0:
        return np.inf
    return float(np.mean(distances))


def finite_node_chain(nodes, node_count):
    if nodes is None:
        return None
    nodes = np.asarray(nodes, dtype=np.float64)
    expected_shape = (int(node_count), 3)
    if nodes.shape != expected_shape:
        return None
    if not np.all(np.isfinite(nodes)):
        return None
    return np.ascontiguousarray(nodes, dtype=np.float64)



def max_node_vector_norm(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] < 3 or len(vectors) == 0:
        return 0.0
    finite = np.all(np.isfinite(vectors[:, :3]), axis=1)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.linalg.norm(vectors[finite, :3], axis=1)))


def clamp_node_vectors(vectors, max_norm):
    vectors = np.asarray(vectors, dtype=np.float64).copy()
    if vectors.ndim != 2 or vectors.shape[1] < 3:
        return vectors
    max_norm = max(0.0, float(max_norm))
    if max_norm <= 0.0:
        return np.zeros_like(vectors, dtype=np.float64)
    finite = np.all(np.isfinite(vectors[:, :3]), axis=1)
    norms = np.linalg.norm(vectors[:, :3], axis=1)
    scale = np.ones(len(vectors), dtype=np.float64)
    too_large = finite & (norms > max_norm)
    scale[too_large] = max_norm / np.maximum(norms[too_large], 1e-12)
    vectors[:, :3] *= scale[:, None]
    vectors[~finite, :3] = 0.0
    return np.ascontiguousarray(vectors[:, :3], dtype=np.float64)


def zero_small_node_vectors(vectors, min_norm):
    vectors = np.asarray(vectors, dtype=np.float64).copy()
    if vectors.ndim != 2 or vectors.shape[1] < 3:
        return vectors
    min_norm = max(0.0, float(min_norm))
    if min_norm <= 0.0:
        return np.ascontiguousarray(vectors[:, :3], dtype=np.float64)
    finite = np.all(np.isfinite(vectors[:, :3]), axis=1)
    norms = np.linalg.norm(vectors[:, :3], axis=1)
    vectors[(~finite) | (norms < min_norm), :3] = 0.0
    return np.ascontiguousarray(vectors[:, :3], dtype=np.float64)


def segment_lengths(nodes):
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim == 3:
        return np.linalg.norm(np.diff(nodes[:, :, :3], axis=1), axis=2)
    return np.linalg.norm(np.diff(nodes[:, :3], axis=0), axis=1)
