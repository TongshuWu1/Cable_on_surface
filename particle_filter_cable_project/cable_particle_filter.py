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
    segment_length_m: float = 0.0
    initial_node_std_m: float = 0.025
    initial_direction_std: float = 0.10
    process_node_std_m: float = 0.006
    process_direction_std: float = 0.020
    direction_smooth_passes: int = 1
    measurement_node_std_m: float = 0.030
    measurement_max_points: int = 1024
    scoring_backend: str = "auto"
    score_chunk_points: int = 512
    endpoint_ordering: bool = True
    ordering_max_points: int = 384
    ordering_knn: int = 8
    endpoint_refresh_interval: int = 0
    reference_ordering: bool = True
    reference_ordering_gate_m: float = 0.08
    endpoint_penalty_weight: float = 1.0
    measurement_reset_error_m: float = 0.25
    measurement_proposal_ratio: float = 0.30
    measurement_proposal_stable_ratio: float = 0.04
    measurement_proposal_start_error_m: float = 0.015
    measurement_proposal_full_error_m: float = 0.060
    measurement_proposal_node_std_m: float = 0.012
    measurement_proposal_direction_std: float = 0.025
    score_keep_fraction: float = 0.80
    coverage_penalty_m: float = 0.025
    coverage_min_fraction: float = 0.05
    bend_penalty_m: float = 0.030
    map_estimate_effective_ratio: float = 0.35
    top_particle_count: int = 50
    global_random_particle_ratio: float = 0.10
    global_random_bounds_padding_m: float = 0.10
    min_measurement_points: int = 12
    min_segment_points: int = 4
    occlusion_assignment_max_distance_m: float = 0.12
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
    measurement_point_count: int = 0
    visible_segments: np.ndarray | None = None
    visible_nodes: np.ndarray | None = None
    measurement_proposal_ratio: float = 0.0
    global_random_particle_ratio: float = 0.0


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
        self.segment_length_m = fixed_segment_length_from_config(self.config)
        self.last_motion_noise_scale = 1.0
        self.last_visible_segments = np.zeros(self.segment_count, dtype=bool)
        self.last_visible_nodes = np.zeros(self.node_count, dtype=bool)
        self.last_measurement_point_count = 0
        self.last_measurement_proposal_ratio = 0.0
        self.last_global_random_particle_ratio = 0.0
        self.last_ordered_measurement_nodes = None
        self.measurement_update_count = 0

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
                measurement_nodes = self._measurement_nodes(measurement, measurement_points)
                if self._should_reset_to_measurement(measurement_nodes):
                    self.initialized = False
                    self._initialize(measurement, measurement_points)
                    self.lost_frames = 0
                    return self._estimate(measurement_used=True, prediction_only=False)
                self._inject_measurement_proposals(measurement_nodes)
                measurement_points, _segment_indices, visible_segments = self._associate_visible_points(
                    measurement_points,
                    candidate_nodes=measurement_nodes,
                )
                if measurement_points is None:
                    self.last_measurement_point_count = 0
                    return self._prediction_only_after_update_drop()
                self.last_measurement_point_count = int(len(measurement_points))
                self._inject_global_random_particles(measurement_points)
                self._weight(
                    measurement_points,
                    visible_segments=visible_segments,
                    measurement_nodes=measurement_nodes,
                )
                self._set_visibility(visible_segments)
                result = self._estimate(measurement_used=True, prediction_only=False)
                self._maybe_resample(noise_scale=self.last_motion_noise_scale)
                self.lost_frames = 0
                return result

            self.lost_frames = 0
            return self._estimate(measurement_used=True, prediction_only=False)

        if not self.initialized:
            return None

        self.last_measurement_point_count = 0
        if count_lost:
            self.lost_frames += 1
        if self.lost_frames > int(self.config.max_prediction_frames):
            self.initialized = False
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
        return result

    def _initialize(self, measurement, measurement_points):
        nodes = self._initial_nodes(measurement, measurement_points)
        if nodes is None:
            return

        if self.segment_length_m is None:
            self.segment_length_m = estimate_equal_segment_length(nodes, self.segment_count)
        nodes = project_equal_length_chain(nodes, self.segment_length_m, self.node_count)
        if nodes is None:
            return
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
        self.initialized = True
        self.last_motion_noise_scale = 1.0
        self.last_measurement_proposal_ratio = 0.0
        self.last_global_random_particle_ratio = 0.0
        self.measurement_update_count = 1
        _points, _indices, visible_segments = self._associate_visible_points(measurement_points)
        self._set_visibility(visible_segments)

    def _predict(self, dt, noise_scale=1.0):
        if self.particles is None or self.segment_length_m is None:
            return
        noise_scale = float(np.clip(noise_scale, 0.25, self.config.max_motion_noise_scale))
        self.last_motion_noise_scale = noise_scale
        starts = self.particles[:, 0, :] + self.rng.normal(
            0.0,
            float(self.config.process_node_std_m) * noise_scale,
            (len(self.particles), 3),
        )
        directions = particle_directions(self.particles)
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

    def _weight(self, measurement_points, visible_segments=None, measurement_nodes=None):
        sigma = max(float(self.config.measurement_node_std_m), 1e-5)
        outlier_distance = max(float(self.config.outlier_distance_m), sigma)
        expected_segments = None
        if visible_segments is not None:
            visible_segments = np.asarray(visible_segments, dtype=bool).reshape(-1)
            expected_segments = np.flatnonzero(visible_segments)
        scores = particle_distance_scores(
            measurement_points,
            self.particles,
            outlier_distance=outlier_distance,
            score_keep_fraction=float(self.config.score_keep_fraction),
            coverage_penalty_m=float(self.config.coverage_penalty_m),
            coverage_min_fraction=float(self.config.coverage_min_fraction),
            bend_penalty_m=float(getattr(self.config, "bend_penalty_m", 0.0)),
            expected_segments=expected_segments,
            backend=str(getattr(self.config, "scoring_backend", "auto")),
            chunk_points=int(getattr(self.config, "score_chunk_points", 512)),
        )

        if not np.any(np.isfinite(scores)):
            return

        endpoint_weight = float(getattr(self.config, "endpoint_penalty_weight", 0.0))
        if endpoint_weight > 0.0 and measurement_nodes is not None:
            scores = scores + endpoint_order_penalty(self.particles, measurement_nodes, weight=endpoint_weight)

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
            measurement_point_count=int(self.last_measurement_point_count),
            visible_segments=self.last_visible_segments.copy(),
            visible_nodes=self.last_visible_nodes.copy(),
            measurement_proposal_ratio=float(self.last_measurement_proposal_ratio),
            global_random_particle_ratio=float(self.last_global_random_particle_ratio),
        )

    def _estimate_nodes(self):
        top_count = int(getattr(self.config, "top_particle_count", 50))
        top_count = int(np.clip(top_count, 1, len(self.weights)))
        top_indices = np.argsort(self.weights)[-top_count:]
        top_weights = np.asarray(self.weights[top_indices], dtype=np.float64)
        total = float(np.sum(top_weights))
        if not np.isfinite(total) or total <= 1e-12:
            top_weights = np.full(top_count, 1.0 / top_count, dtype=np.float64)
        else:
            top_weights = top_weights / total
        top_particles = self.particles[top_indices]
        starts = top_particles[:, 0, :]
        start = np.average(starts, axis=0, weights=top_weights)
        directions = np.average(particle_directions(top_particles), axis=0, weights=top_weights)
        directions = normalize_vectors(directions)
        return build_chain(start, directions, self.segment_length_m)

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

    def _measurement_nodes(self, measurement, measurement_points):
        candidate = valid_points(getattr(measurement, "points_xyz", None))
        if len(candidate) >= 2:
            nodes = fit_polyline_segments(candidate, segment_count=self.segment_count)
        else:
            reference_nodes = self._estimate_nodes() if self.initialized and self.particles is not None else None
            force_endpoint = self._should_refresh_endpoint_ordering()
            nodes = self._fit_measurement_point_chain(
                measurement_points,
                reference_nodes=reference_nodes,
                allow_endpoint_fallback=self.lost_frames > 0 or force_endpoint,
                force_endpoint=force_endpoint,
            )
            self.measurement_update_count += 1
        if nodes is None or self.segment_length_m is None:
            return None
        nodes = project_equal_length_chain(nodes, self.segment_length_m, self.node_count)
        if nodes is None:
            return None
        if self.initialized and self.particles is not None:
            nodes = align_polyline_orientation(self._estimate_nodes(), nodes)
        self.last_ordered_measurement_nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        return np.ascontiguousarray(nodes, dtype=np.float64)

    def _fit_measurement_point_chain(
        self,
        measurement_points,
        reference_nodes=None,
        allow_endpoint_fallback=True,
        force_endpoint=False,
    ):
        if bool(force_endpoint):
            nodes = fit_unordered_point_cloud_chain(
                measurement_points,
                segment_count=self.segment_count,
                endpoint_ordering=bool(getattr(self.config, "endpoint_ordering", True)),
                max_points=int(getattr(self.config, "ordering_max_points", 768)),
                knn=int(getattr(self.config, "ordering_knn", 10)),
            )
            if nodes is not None:
                return nodes

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

    def _should_refresh_endpoint_ordering(self):
        interval = int(getattr(self.config, "endpoint_refresh_interval", 0))
        if interval <= 0 or not bool(getattr(self.config, "endpoint_ordering", True)):
            return False
        if self.lost_frames > 0:
            return True
        return int(self.measurement_update_count) > 0 and int(self.measurement_update_count) % interval == 0

    def _inject_measurement_proposals(self, measurement_nodes):
        if measurement_nodes is None or self.particles is None or self.weights is None or self.segment_length_m is None:
            self.last_measurement_proposal_ratio = 0.0
            return

        ratio = self._adaptive_measurement_proposal_ratio(measurement_nodes)
        self.last_measurement_proposal_ratio = float(ratio)
        if ratio <= 0.0:
            return
        total_count = len(self.particles)
        proposal_count = int(round(ratio * total_count))
        proposal_count = int(np.clip(proposal_count, 1, total_count))

        proposals = sample_noisy_chains_around_nodes(
            measurement_nodes,
            proposal_count,
            self.segment_length_m,
            self.rng,
            node_std_m=float(self.config.measurement_proposal_node_std_m),
            direction_std=float(self.config.measurement_proposal_direction_std),
            direction_smooth_passes=int(getattr(self.config, "direction_smooth_passes", 1)),
        )
        replace_indices = self.rng.choice(total_count, size=proposal_count, replace=False)
        self.particles[replace_indices] = proposals

        self.weights *= 1.0 - ratio
        self.weights[replace_indices] = ratio / proposal_count
        total = float(np.sum(self.weights))
        if np.isfinite(total) and total > 1e-12:
            self.weights /= total
        else:
            self.weights.fill(1.0 / total_count)

    def _inject_global_random_particles(self, measurement_points):
        self.last_global_random_particle_ratio = 0.0
        if self.particles is None or self.weights is None or self.segment_length_m is None:
            return
        ratio = float(np.clip(getattr(self.config, "global_random_particle_ratio", 0.0), 0.0, 1.0))
        if ratio <= 0.0:
            return
        points = valid_points(measurement_points)
        if len(points) < 2:
            return

        total_count = len(self.particles)
        random_count = int(round(ratio * total_count))
        random_count = int(np.clip(random_count, 1, total_count))
        padding = max(0.0, float(getattr(self.config, "global_random_bounds_padding_m", 0.10)))
        random_particles = sample_global_random_chains(
            points,
            random_count,
            self.segment_count,
            self.segment_length_m,
            self.rng,
            padding_m=padding,
            direction_smooth_passes=int(getattr(self.config, "direction_smooth_passes", 1)),
        )
        if len(random_particles) != random_count:
            return

        replace_indices = self.rng.choice(total_count, size=random_count, replace=False)
        self.particles[replace_indices] = random_particles
        self.weights *= 1.0 - ratio
        self.weights[replace_indices] = ratio / random_count
        total = float(np.sum(self.weights))
        if np.isfinite(total) and total > 1e-12:
            self.weights /= total
        else:
            self.weights.fill(1.0 / total_count)
        self.last_global_random_particle_ratio = float(random_count) / max(float(total_count), 1.0)

    def _should_reset_to_measurement(self, measurement_nodes):
        threshold = float(getattr(self.config, "measurement_reset_error_m", 0.0))
        if threshold <= 0.0 or measurement_nodes is None or not self.initialized:
            return False
        current_nodes = self._estimate_nodes()
        error_m = mean_node_distance(current_nodes, measurement_nodes)
        return bool(np.isfinite(error_m) and error_m > threshold)

    def _adaptive_measurement_proposal_ratio(self, measurement_nodes):
        max_ratio = float(np.clip(self.config.measurement_proposal_ratio, 0.0, 1.0))
        if max_ratio <= 0.0 or measurement_nodes is None:
            return 0.0
        if self.lost_frames > 0:
            return max_ratio

        stable_ratio = float(np.clip(self.config.measurement_proposal_stable_ratio, 0.0, max_ratio))
        current_nodes = self._estimate_nodes() if self.initialized else None
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

    def _associate_visible_points(self, measurement_points, candidate_nodes=None):
        reference_nodes = self._estimate_nodes() if self.initialized else None
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
            f"raw fixed-length segment particle filter {mode} | {base_method} | "
            f"segment_length={result.segment_length_m:.4f}m ess={result.effective_sample_size:.0f} "
            f"support={result.measurement_point_count} lost={result.lost_frames}"
        ),
    )


def fixed_segment_length_from_config(config):
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


def build_chain(start, directions, segment_length_m):
    start = np.asarray(start, dtype=np.float64).reshape(3)
    directions = normalize_vectors(directions)
    nodes = np.empty((len(directions) + 1, 3), dtype=np.float64)
    nodes[0] = start
    for index, direction in enumerate(directions):
        nodes[index + 1] = nodes[index] + float(segment_length_m) * direction
    return nodes.astype(np.float32)


def build_chains(starts, directions, segment_length_m):
    starts = np.asarray(starts, dtype=np.float64)
    directions = normalize_vectors(directions)
    output = np.empty((len(starts), directions.shape[1] + 1, 3), dtype=np.float64)
    output[:, 0, :] = starts
    for index in range(directions.shape[1]):
        output[:, index + 1, :] = output[:, index, :] + float(segment_length_m) * directions[:, index, :]
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


def sample_global_random_chains(
    points_xyz,
    count,
    segment_count,
    segment_length_m,
    rng,
    padding_m=0.10,
    direction_smooth_passes=1,
):
    points = valid_points(points_xyz)
    count = max(0, int(count))
    segment_count = max(1, int(segment_count))
    if count == 0 or len(points) < 2:
        return np.empty((0, 0, 3), dtype=np.float64)

    mins = np.min(points, axis=0) - float(padding_m)
    maxs = np.max(points, axis=0) + float(padding_m)
    span = maxs - mins
    min_span = max(float(segment_length_m), 1e-3)
    tiny = span < min_span
    if np.any(tiny):
        center = 0.5 * (mins + maxs)
        mins[tiny] = center[tiny] - 0.5 * min_span
        maxs[tiny] = center[tiny] + 0.5 * min_span

    starts = rng.uniform(mins, maxs, size=(count, 3))
    directions = normalize_vectors(rng.normal(0.0, 1.0, size=(count, segment_count, 3)))
    directions = smooth_particle_directions(
        directions,
        passes=int(direction_smooth_passes),
    )
    return build_chains(starts, directions, float(segment_length_m))


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


def valid_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    points = points[:, :3]
    return np.ascontiguousarray(points[np.all(np.isfinite(points), axis=1)], dtype=np.float64)


def sample_points(points, max_points):
    points = valid_points(points)
    max_points = max(0, int(max_points))
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return np.ascontiguousarray(points, dtype=np.float64)


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
    best_squared = np.full(len(points), np.inf, dtype=np.float64)
    best_segments = np.zeros(len(points), dtype=np.int64)
    best_arclength = np.zeros(len(points), dtype=np.float64)

    for index, (start, vector, length, length_start) in enumerate(
        zip(nodes[:-1], segment_vectors, segment_lengths, cumulative[:-1])
    ):
        length_sq = float(np.dot(vector, vector))
        if length_sq <= 1e-12:
            projection_fraction = np.zeros(len(points), dtype=np.float64)
            projected = start[None, :]
        else:
            projection_fraction = np.clip(((points - start[None, :]) @ vector) / length_sq, 0.0, 1.0)
            projected = start[None, :] + projection_fraction[:, None] * vector[None, :]
        squared = np.sum((points - projected) ** 2, axis=1)
        update = squared < best_squared
        if np.any(update):
            best_squared[update] = squared[update]
            best_segments[update] = int(index)
            best_arclength[update] = float(length_start) + projection_fraction[update] * float(length)

    return best_segments, np.sqrt(best_squared), best_arclength


def particle_distance_scores(
    points,
    particles,
    outlier_distance=np.inf,
    score_keep_fraction=1.0,
    coverage_penalty_m=0.0,
    coverage_min_fraction=0.05,
    bend_penalty_m=0.0,
    expected_segments=None,
    backend="auto",
    chunk_points=512,
):
    points = valid_points(points)
    particles = valid_particles(particles)
    if len(points) == 0 or len(particles) == 0 or particles.shape[1] < 2:
        return np.full(len(particles), np.inf, dtype=np.float64)

    torch_scores = particle_distance_scores_torch(
        points,
        particles,
        outlier_distance=outlier_distance,
        score_keep_fraction=score_keep_fraction,
        coverage_penalty_m=coverage_penalty_m,
        coverage_min_fraction=coverage_min_fraction,
        bend_penalty_m=bend_penalty_m,
        expected_segments=expected_segments,
        backend=backend,
        chunk_points=chunk_points,
    )
    if torch_scores is not None:
        return torch_scores

    all_squared_distances = point_to_particle_segment_squared_distances(points, particles[:, :-1, :3], particles[:, 1:, :3])
    if all_squared_distances.size:
        squared_distances = np.min(all_squared_distances, axis=1)
        nearest_segments = np.argmin(all_squared_distances, axis=1)
    else:
        squared_distances = np.empty((len(particles), 0), dtype=np.float64)
        nearest_segments = np.empty((len(particles), 0), dtype=np.int64)
    expected_segments = expected_segment_indices(expected_segments, particles.shape[1] - 1)
    if squared_distances.size == 0:
        return np.full(len(particles), np.inf, dtype=np.float64)

    scores = trimmed_mean_squared_values(
        squared_distances,
        outlier_distance=float(outlier_distance),
        keep_fraction=float(score_keep_fraction),
    )
    if coverage_penalty_m > 0.0 and len(expected_segments):
        scores = scores + segment_coverage_penalty(
            nearest_segments,
            expected_segments=expected_segments,
            penalty_m=float(coverage_penalty_m),
            min_fraction=float(coverage_min_fraction),
        )
    if float(bend_penalty_m) > 0.0:
        scores = scores + particle_bend_penalty(particles, penalty_m=float(bend_penalty_m))
    return scores


def particle_distance_scores_torch(
    points,
    particles,
    outlier_distance=np.inf,
    score_keep_fraction=1.0,
    coverage_penalty_m=0.0,
    coverage_min_fraction=0.05,
    bend_penalty_m=0.0,
    expected_segments=None,
    backend="auto",
    chunk_points=512,
):
    device = torch_scoring_device(backend)
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
            nearest_chunks = []
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
                nearest_chunks.append(nearest)

            if not best_chunks:
                return np.full(len(particles), np.inf, dtype=np.float64)
            squared_distances = torch.cat(best_chunks, dim=1)
            nearest_segments = torch.cat(nearest_chunks, dim=1)

            outlier_distance = float(outlier_distance)
            if np.isfinite(outlier_distance):
                squared_distances = torch.clamp(squared_distances, max=outlier_distance * outlier_distance)
            keep_fraction = float(np.clip(score_keep_fraction, 1e-6, 1.0))
            keep_count = max(1, int(np.ceil(squared_distances.shape[1] * keep_fraction)))
            if keep_count >= squared_distances.shape[1]:
                scores = torch.mean(squared_distances, dim=1)
            else:
                selected, _indices = torch.topk(squared_distances, keep_count, dim=1, largest=False)
                scores = torch.mean(selected, dim=1)

            expected = expected_segment_indices(expected_segments, particles.shape[1] - 1)
            if float(coverage_penalty_m) > 0.0 and len(expected):
                required = max(1, int(np.ceil(nearest_segments.shape[1] * float(np.clip(coverage_min_fraction, 0.0, 1.0)))))
                missing = torch.zeros(nearest_segments.shape[0], dtype=torch.float32, device=device)
                for segment_index in expected:
                    counts = torch.count_nonzero(nearest_segments == int(segment_index), dim=1)
                    missing = missing + (counts < required).to(torch.float32)
                scores = scores + missing * float(coverage_penalty_m) * float(coverage_penalty_m)

            if float(bend_penalty_m) > 0.0 and particles_t.shape[1] > 2:
                directions = segment / torch.sqrt(length_sq[:, :, None])
                dots = torch.sum(directions[:, :-1, :] * directions[:, 1:, :], dim=2).clamp(-1.0, 1.0)
                bend = torch.mean(torch.clamp(1.0 - dots, min=0.0), dim=1)
                scores = scores + bend * float(bend_penalty_m) * float(bend_penalty_m)

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


def expected_segment_indices(expected_segments, segment_count):
    if expected_segments is None:
        return np.arange(int(segment_count), dtype=np.int64)
    expected = np.asarray(expected_segments, dtype=np.int64).reshape(-1)
    if len(expected) == 0:
        return expected
    return np.unique(np.clip(expected, 0, int(segment_count) - 1))


def trimmed_mean_squared_values(squared_distances, outlier_distance=np.inf, keep_fraction=1.0):
    squared = np.asarray(squared_distances, dtype=np.float64)
    if squared.ndim != 2 or squared.shape[1] == 0:
        return np.full(squared.shape[0] if squared.ndim == 2 else 0, np.inf, dtype=np.float64)

    outlier_distance = float(outlier_distance)
    if np.isfinite(outlier_distance):
        squared = np.minimum(squared, outlier_distance * outlier_distance)
    keep_fraction = float(np.clip(keep_fraction, 1e-6, 1.0))
    keep_count = max(1, int(np.ceil(squared.shape[1] * keep_fraction)))
    if keep_count >= squared.shape[1]:
        return np.mean(squared, axis=1)

    selected = np.partition(squared, keep_count - 1, axis=1)[:, :keep_count]
    return np.mean(selected, axis=1)


def segment_coverage_penalty(nearest_segments, expected_segments, penalty_m=0.025, min_fraction=0.05):
    nearest_segments = np.asarray(nearest_segments, dtype=np.int64)
    if nearest_segments.ndim != 2 or nearest_segments.shape[1] == 0:
        return np.zeros(nearest_segments.shape[0] if nearest_segments.ndim == 2 else 0, dtype=np.float64)

    expected_segments = np.asarray(expected_segments, dtype=np.int64).reshape(-1)
    if len(expected_segments) == 0:
        return np.zeros(nearest_segments.shape[0], dtype=np.float64)

    required = max(1, int(np.ceil(nearest_segments.shape[1] * float(np.clip(min_fraction, 0.0, 1.0)))))
    missing_counts = np.zeros(nearest_segments.shape[0], dtype=np.float64)
    for segment_index in expected_segments:
        counts = np.count_nonzero(nearest_segments == int(segment_index), axis=1)
        missing_counts += counts < required
    return missing_counts * float(penalty_m) * float(penalty_m)


def particle_bend_penalty(particles, penalty_m=0.030):
    particles = valid_particles(particles)
    if len(particles) == 0 or particles.shape[1] < 3:
        return np.zeros(len(particles), dtype=np.float64)
    directions = particle_directions(particles)
    dots = np.sum(directions[:, :-1, :] * directions[:, 1:, :], axis=2)
    bend = np.mean(np.maximum(0.0, 1.0 - np.clip(dots, -1.0, 1.0)), axis=1)
    return bend * float(penalty_m) * float(penalty_m)


def endpoint_order_penalty(particles, measurement_nodes, weight=1.0):
    particles = valid_particles(particles)
    nodes = valid_points(measurement_nodes)
    if len(particles) == 0 or len(nodes) < 2:
        return np.zeros(len(particles), dtype=np.float64)
    start = nodes[0]
    end = nodes[-1]
    start_error = np.sum((particles[:, 0, :3] - start[None, :]) ** 2, axis=1)
    end_error = np.sum((particles[:, -1, :3] - end[None, :]) ** 2, axis=1)
    return 0.5 * float(weight) * (start_error + end_error)


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

    output = np.empty((len(points), len(nodes) - 1), dtype=np.float64)
    for index, (start, end) in enumerate(zip(nodes[:-1], nodes[1:])):
        output[:, index] = point_to_segment_distances(points, start, end)
    return output


def valid_particles(particles):
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 3 or particles.shape[1] < 2 or particles.shape[2] < 3:
        return np.empty((0, 0, 3), dtype=np.float64)
    return np.ascontiguousarray(particles[:, :, :3], dtype=np.float64)


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


def segment_lengths(nodes):
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim == 3:
        return np.linalg.norm(np.diff(nodes[:, :, :3], axis=1), axis=2)
    return np.linalg.norm(np.diff(nodes[:, :3], axis=0), axis=1)
