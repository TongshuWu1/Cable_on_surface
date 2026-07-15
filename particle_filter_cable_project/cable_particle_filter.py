from dataclasses import dataclass
from itertools import combinations
import time

import numpy as np

from cable_cuda import constrain_chains as constrain_chains_cuda
from cable_cuda import particle_point_distances as particle_point_distances_cuda
from cable_detection import (
    CableEstimate3D,
    EndpointMarker3D,
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
    estimate_top_particle_count: int = 32
    segment_length_m: float = 0.0
    process_node_std_m: float = 0.006
    process_direction_std: float = 0.020
    velocity_enabled: bool = True
    velocity_damping: float = 0.85
    velocity_measurement_blend: float = 0.30
    velocity_process_std_mps: float = 0.03
    max_node_speed_mps: float = 1.0
    direction_smooth_passes: int = 1
    measurement_node_std_m: float = 0.030
    measurement_max_points: int = 1024
    scoring_backend: str = "auto"
    endpoint_tangent_min_radius_m: float = 0.008
    endpoint_tangent_radius_m: float = 0.075
    endpoint_tangent_sigma_m: float = 0.035
    endpoint_tangent_min_points: int = 8
    endpoint_tangent_min_confidence: float = 0.20
    endpoint_conditioned_proposal_ratio: float = 0.25
    endpoint_conditioned_direction_std: float = 0.08
    endpoint_conditioned_deformation_std_m: float = 0.025
    endpoint_conditioned_deformation_modes: int = 3
    ownership_outlier_likelihood: float = 0.05
    ownership_outlier_prior: float = 0.10
    ownership_min_responsibility: float = 0.01
    ownership_visibility_threshold: float = 0.15
    robust_distance_m: float = 0.05
    coverage_penalty_m: float = 0.025
    coverage_min_fraction: float = 0.05
    bend_penalty_m: float = 0.030
    global_random_particle_ratio: float = 0.10
    endpoint_constraint_iterations: int = 128
    endpoint_constraint_tolerance_m: float = 1e-4
    min_measurement_points: int = 12
    min_segment_points: int = 4
    occlusion_assignment_max_distance_m: float = 0.12
    max_prediction_frames: int = 12
    max_motion_noise_scale: float = 4.0


@dataclass
class ParticleEstimateDiagnostics:
    average_points_xyz: np.ndarray
    map_points_xyz: np.ndarray
    top_particle_points_xyz: np.ndarray
    node_rms_spread_m: np.ndarray
    node_principal_std_xyz: np.ndarray
    map_to_average_node_error_m: float
    mean_node_spread_m: float
    max_node_spread_m: float
    endpoint_direction_delta_deg: np.ndarray
    endpoint_tangents_xyz: np.ndarray
    endpoint_tangent_confidence: np.ndarray
    endpoint_tangent_support_count: np.ndarray
    mean_ownership_responsibility: float
    ownership_entropy: float
    visible_segment_fraction: float
    endpoint_conditioned_proposal_ratio: float


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
    support_points_xyz: np.ndarray | None = None
    visible_segments: np.ndarray | None = None
    visible_nodes: np.ndarray | None = None
    endpoint_conditioned_proposal_ratio: float = 0.0
    global_random_particle_ratio: float = 0.0
    mean_ownership_responsibility: float = np.nan
    ownership_entropy: float = np.nan
    ownership_effective_point_count: float = 0.0
    endpoint_tangent_confidence: np.ndarray | None = None
    endpoint_tangent_support_count: np.ndarray | None = None
    estimate_particle_count: int = 1
    estimate_weight_mass: float = 1.0
    mean_node_speed_mps: float = 0.0
    stage_seconds: dict | None = None
    particle_diagnostics: ParticleEstimateDiagnostics | None = None


@dataclass
class ParticleFilterUpdateContext:
    measurement: object | None
    measurement_points: np.ndarray | None
    endpoint_nodes: np.ndarray | None
    dt: float
    measurement_used: bool = False
    prediction_only: bool = False
    initialized_this_frame: bool = False


@dataclass(frozen=True)
class PerCableEndpointAssociationConfig:
    ambiguity_margin_m: float = 0.015
    support_weight: float = 0.35
    support_clip_m: float = 0.080
    endpoint_constraint_tolerance_m: float = 0.005
    node_count: int = 13
    max_stale_frames: int = 30


@dataclass
class PerCableEndpointAssociationResult:
    markers_by_cable: list
    per_cable_diagnostics: list
    diagnostics: dict


class PerCableEndpointAssociator:
    """Bind each endpoint channel directly to its corresponding cable PF.

    Channel ``endpoints_cable1`` contains both ends of PF1 and channel
    ``endpoints_cable2`` contains both ends of PF2.  Connected-component order
    has no semantic meaning, so each accepted pair is oriented against that
    cable's previous PF chain before it is used as an ordered constraint.
    """

    def __init__(self, cable_count=2, config=None):
        if int(cable_count) != 2:
            raise ValueError(f"Endpoint association requires exactly two cable PFs; got {cable_count}.")
        self.cable_count = 2
        self.config = config if config is not None else PerCableEndpointAssociationConfig()
        self.last_endpoint_nodes_by_cable = [None for _ in range(self.cable_count)]
        self.endpoint_stale_frames_by_cable = [0 for _ in range(self.cable_count)]

    def associate(
        self,
        endpoint_markers_by_cable,
        filter_reference_nodes,
        shared_support,
        cable_lengths_m,
        tape_lengths_m,
        offset_to_tips=False,
    ):
        groups = _association_values(endpoint_markers_by_cable, self.cable_count, None)
        references = _association_values(filter_reference_nodes, self.cable_count, None)
        cable_lengths = _association_values(cable_lengths_m, self.cable_count, 0.0, float)
        tape_lengths = _association_values(tape_lengths_m, self.cable_count, 0.0, float)
        support = _association_points(shared_support)
        counts = [_association_marker_count(group) for group in groups]
        effective_references = []
        for cable_index in range(self.cable_count):
            reference = _association_reference(references[cable_index])
            if reference is None:
                reference = _association_reference(self.last_endpoint_nodes_by_cable[cable_index])
            effective_references.append(reference)

        markers = [None for _ in range(self.cable_count)]
        per_cable_diagnostics = []
        accepted_cables = set()
        assignment_parts = []
        for cable_index in range(self.cable_count):
            marker, diagnostic = self._associate_cable(
                cable_index,
                groups[cable_index],
                effective_references[cable_index],
                support,
                cable_lengths[cable_index],
                tape_lengths[cable_index],
                bool(offset_to_tips),
            )
            markers[cable_index] = marker
            per_cable_diagnostics.append(diagnostic)
            if marker is None:
                continue
            endpoints = _association_endpoint_nodes(marker)
            self.last_endpoint_nodes_by_cable[cable_index] = endpoints.copy()
            accepted_cables.add(cable_index)
            pair = diagnostic["endpoint_component_pair"]
            assignment_parts.append(
                f"PF{cable_index + 1}<-endpoints_{cable_index + 1}[{pair[0] + 1},{pair[1] + 1}]"
            )

        self._age_endpoint_memory(accepted_cables)
        statuses = [item["endpoint_association_status"] for item in per_cable_diagnostics]
        if len(accepted_cables) == self.cable_count:
            overall_status = "stable" if all(reference is not None for reference in effective_references) else "initialized"
        elif accepted_cables:
            overall_status = "partial"
        elif any(status == "ambiguous" for status in statuses):
            overall_status = "ambiguous"
        elif any(status == "unreachable" for status in statuses):
            overall_status = "unreachable"
        else:
            overall_status = "incomplete"

        finite_costs = [
            float(item["endpoint_association_cost_m"])
            for item in per_cable_diagnostics
            if np.isfinite(float(item["endpoint_association_cost_m"]))
        ]
        finite_margins = [
            float(item["endpoint_association_margin_m"])
            for item in per_cable_diagnostics
            if np.isfinite(float(item["endpoint_association_margin_m"]))
        ]
        diagnostics = {
            "endpoint_observation_model": "NN:cable+per-cable-endpoints PF:fixed-channel-identity",
            "endpoint_group_counts_text": f"C1={counts[0]},C2={counts[1]}",
            "endpoint_association_status": overall_status,
            "endpoint_association_cost_m": float(np.sum(finite_costs)) if finite_costs else np.nan,
            "endpoint_association_margin_m": min(finite_margins) if finite_margins else np.nan,
            "endpoint_assignment_text": ",".join(assignment_parts),
        }
        return PerCableEndpointAssociationResult(markers, per_cable_diagnostics, diagnostics)

    def _associate_cable(
        self,
        cable_index,
        group,
        reference,
        support,
        cable_length_m,
        tape_length_m,
        offset_to_tips,
    ):
        count = _association_marker_count(group)
        base = {
            "endpoint_association_status": "incomplete",
            "endpoint_cable_index": int(cable_index),
            "endpoint_component_count": int(count),
            "endpoint_component_pair": (-1, -1),
            "endpoint_association_cost_m": np.nan,
            "endpoint_association_margin_m": np.nan,
            "endpoint_motion_m": np.nan,
        }
        if count < 2:
            return None, base

        hypotheses = []
        for first_index, second_index in combinations(range(count), 2):
            marker = _same_group_endpoint_marker(group, first_index, second_index)
            marker = _ordered_endpoint_marker(marker, reference)
            marker = _marker_with_endpoint_tip_offset(marker, tape_length_m, offset_to_tips)
            cost = self._candidate_cost(marker, reference, support, cable_length_m)
            if np.isfinite(cost):
                hypotheses.append((float(cost), int(first_index), int(second_index), marker))
        if not hypotheses:
            base["endpoint_association_status"] = "unreachable"
            return None, base

        hypotheses.sort(key=lambda item: item[:3])
        best = hypotheses[0]
        margin = float(hypotheses[1][0] - best[0]) if len(hypotheses) > 1 else np.inf
        if margin < float(self.config.ambiguity_margin_m):
            base.update(
                endpoint_association_status="ambiguous",
                endpoint_association_margin_m=margin,
            )
            return None, base

        endpoints = _association_endpoint_nodes(best[3])
        previous = _association_reference(self.last_endpoint_nodes_by_cable[cable_index])
        motion_m = np.nan
        if previous is not None:
            motion_m = float(max(
                np.linalg.norm(endpoints[0] - previous[0]),
                np.linalg.norm(endpoints[1] - previous[-1]),
            ))
        base.update(
            endpoint_association_status="stable" if reference is not None else "initialized",
            endpoint_component_pair=(best[1], best[2]),
            endpoint_association_cost_m=float(best[0]),
            endpoint_association_margin_m=margin,
            endpoint_motion_m=motion_m,
        )
        return best[3], base

    def _candidate_cost(self, marker, reference, support, cable_length_m):
        endpoints = _association_endpoint_nodes(marker)
        if endpoints is None:
            return np.inf
        chord_m = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        cable_length_m = max(0.0, float(cable_length_m))
        if cable_length_m > 0.0 and chord_m > cable_length_m + float(self.config.endpoint_constraint_tolerance_m):
            return np.inf

        if reference is None:
            node_count = max(2, int(self.config.node_count))
            nodes = np.linspace(endpoints[0], endpoints[1], node_count, dtype=np.float32)
            support_cost = _association_support_cost(nodes, support, float(self.config.support_clip_m))
            length_cost = abs(cable_length_m - chord_m) if cable_length_m > 0.0 else 0.0
            return length_cost + float(self.config.support_weight) * support_cost

        endpoint_cost = 0.5 * (
            float(np.linalg.norm(endpoints[0] - reference[0]))
            + float(np.linalg.norm(endpoints[1] - reference[-1]))
        )
        aligned = _association_align_reference(reference, endpoints)
        support_cost = _association_support_cost(aligned, support, float(self.config.support_clip_m))
        return endpoint_cost + float(self.config.support_weight) * support_cost

    def _age_endpoint_memory(self, accepted_cables):
        max_stale = max(0, int(self.config.max_stale_frames))
        for cable_index in range(self.cable_count):
            if cable_index in accepted_cables:
                self.endpoint_stale_frames_by_cable[cable_index] = 0
                continue
            self.endpoint_stale_frames_by_cable[cable_index] += 1
            if self.endpoint_stale_frames_by_cable[cable_index] > max_stale:
                self.last_endpoint_nodes_by_cable[cable_index] = None


def _association_values(values, count, default, converter=None):
    values = list(values or []) if isinstance(values, (list, tuple)) else []
    output = []
    for index in range(int(count)):
        value = values[index] if index < len(values) else default
        output.append(converter(value) if converter is not None else value)
    return output


def _association_marker_count(marker):
    centers = np.asarray(getattr(marker, "centers_xyz", None), dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] < 3:
        return 0
    return int(np.count_nonzero(np.all(np.isfinite(centers[:, :3]), axis=1)))


def _association_points(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    points = points[:, :3]
    return np.ascontiguousarray(points[np.all(np.isfinite(points), axis=1)], dtype=np.float32)


def _association_reference(nodes):
    points = _association_points(nodes)
    return points if len(points) >= 2 else None


def _association_endpoint_nodes(marker):
    if marker is None:
        return None
    endpoints = np.asarray(getattr(marker, "endpoint_nodes", None), dtype=np.float32)
    if endpoints.ndim != 2 or endpoints.shape[0] < 2 or endpoints.shape[1] < 3:
        return None
    endpoints = endpoints[:2, :3]
    if not np.all(np.isfinite(endpoints)):
        return None
    return np.ascontiguousarray(endpoints, dtype=np.float32)


def _same_group_endpoint_marker(group, first_index, second_index):
    centers_xyz_all = np.asarray(group.centers_xyz, dtype=np.float32)
    centers_xy_all = np.asarray(group.centers_xy, dtype=np.float32)
    indices = [int(first_index), int(second_index)]
    centers_xyz = np.ascontiguousarray(centers_xyz_all[indices, :3], dtype=np.float32)
    centers_xy = np.ascontiguousarray(centers_xy_all[indices, :2], dtype=np.float32)
    return EndpointMarker3D(
        mask=np.empty((0, 0), dtype=np.uint8),
        centers_xyz=centers_xyz,
        centers_xy=centers_xy,
        endpoint_nodes=centers_xyz.copy(),
        component_count=2,
        points_xyz=np.empty((0, 3), dtype=np.float32),
    )


def _ordered_endpoint_marker(marker, reference):
    centers_xyz = np.asarray(marker.centers_xyz, dtype=np.float32)[:2, :3]
    centers_xy = np.asarray(marker.centers_xy, dtype=np.float32)[:2, :2]
    reference = _association_reference(reference)
    if reference is not None:
        direct_cost = float(
            np.linalg.norm(centers_xyz[0] - reference[0])
            + np.linalg.norm(centers_xyz[1] - reference[-1])
        )
        reverse_cost = float(
            np.linalg.norm(centers_xyz[1] - reference[0])
            + np.linalg.norm(centers_xyz[0] - reference[-1])
        )
        reverse = reverse_cost < direct_cost
    else:
        # The endpoint class has no start/end semantics.  Use image position
        # only for deterministic bootstrap, then preserve orientation from the
        # PF reference on later frames.
        reverse = tuple(float(value) for value in centers_xy[1]) < tuple(
            float(value) for value in centers_xy[0]
        )
    if reverse:
        centers_xyz = centers_xyz[::-1]
        centers_xy = centers_xy[::-1]
    return EndpointMarker3D(
        mask=marker.mask,
        centers_xyz=np.ascontiguousarray(centers_xyz, dtype=np.float32),
        centers_xy=np.ascontiguousarray(centers_xy, dtype=np.float32),
        endpoint_nodes=np.ascontiguousarray(centers_xyz, dtype=np.float32),
        component_count=2,
        points_xyz=np.ascontiguousarray(marker.points_xyz, dtype=np.float32),
    )


def _marker_with_endpoint_tip_offset(marker, tape_length_m, offset_to_tips):
    centers = np.asarray(marker.centers_xyz, dtype=np.float32)[:2, :3]
    endpoints = centers.copy()
    if bool(offset_to_tips):
        direction = centers[1] - centers[0]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction /= norm
            half_tape = 0.5 * max(0.0, float(tape_length_m))
            endpoints[0] -= half_tape * direction
            endpoints[1] += half_tape * direction
    return EndpointMarker3D(
        mask=marker.mask,
        centers_xyz=np.ascontiguousarray(centers, dtype=np.float32),
        centers_xy=np.ascontiguousarray(marker.centers_xy, dtype=np.float32),
        endpoint_nodes=np.ascontiguousarray(endpoints, dtype=np.float32),
        component_count=2,
        points_xyz=np.ascontiguousarray(marker.points_xyz, dtype=np.float32),
    )


def _association_align_reference(reference, endpoints):
    nodes = np.asarray(reference, dtype=np.float32)[:, :3]
    start_delta = endpoints[0] - nodes[0]
    end_delta = endpoints[1] - nodes[-1]
    t = np.linspace(0.0, 1.0, len(nodes), dtype=np.float32)[:, None]
    return np.ascontiguousarray(nodes + (1.0 - t) * start_delta + t * end_delta, dtype=np.float32)


def _association_support_cost(nodes, support, clip_m):
    nodes = _association_points(nodes)
    support = _association_points(support)
    clip_m = max(1e-4, float(clip_m))
    if len(nodes) == 0 or len(support) == 0:
        return clip_m
    distances = np.linalg.norm(nodes[:, None, :] - support[None, :, :], axis=2)
    nearest = np.min(distances, axis=1)
    return float(np.mean(np.minimum(nearest, clip_m)))


class CableParticleFilter:
    """Particle filter over connected fixed-length cable segments.

    A particle is stored as connected nodes plus one 3D velocity per node.
    Segment roll is not represented, and every segment is projected back to one
    shared fixed length.
    """

    def __init__(self, node_count, config=None, seed=17):
        self.node_count = max(2, int(node_count))
        self.segment_count = self.node_count - 1
        self.config = config if config is not None else CableParticleFilterConfig()
        self.rng = np.random.default_rng(seed)
        self.device = torch_scoring_device(self.config.scoring_backend)
        self.cuda_state = self.device is not None and str(self.config.scoring_backend).strip().lower() == "cuda"
        self.torch_generator = None
        self.cuda_stream = None
        if self.device is not None:
            self.torch_generator = torch.Generator(device=self.device)
            self.torch_generator.manual_seed(int(seed))
            self.cuda_stream = torch.cuda.Stream(device=self.device)
        self.particles = None
        self.node_velocities = None
        self.weights = None
        self.initialized = False
        self.lost_frames = 0
        self.segment_length_m = fixed_segment_length_from_config(self.config)
        self.last_motion_noise_scale = 1.0
        self.last_visible_segments = np.zeros(self.segment_count, dtype=bool)
        self.last_visible_nodes = np.zeros(self.node_count, dtype=bool)
        self.last_measurement_point_count = 0
        self.last_support_points = np.empty((0, 3), dtype=np.float32)
        self.last_endpoint_conditioned_proposal_ratio = 0.0
        self.last_global_random_particle_ratio = 0.0
        self.last_endpoint_tangents = np.full((2, 3), np.nan, dtype=np.float32)
        self.last_endpoint_tangent_confidence = np.zeros(2, dtype=np.float32)
        self.last_endpoint_tangent_support_count = np.zeros(2, dtype=np.int32)
        self.last_endpoint_tangents_t = None
        self.last_endpoint_tangent_confidence_t = None
        self.last_mean_ownership_responsibility = np.nan
        self.last_ownership_entropy = np.nan
        self.last_ownership_effective_point_count = 0.0
        self.last_ordered_measurement_nodes = None
        self.last_endpoint_nodes = None
        self.last_posterior_nodes = None
        self.last_stage_seconds = {}
        self._cuda_stage_events = []
        self._cuda_previous_event = None
        self.measurement_update_count = 0
        self.last_endpoint_nodes_t = None
        self._estimate_nodes_cache = None
        self._estimate_particle_count_cache = 0
        self._estimate_weight_mass_cache = np.nan
        self._particle_estimate_diagnostics_cache = None

    def _invalidate_estimate_cache(self):
        self._estimate_nodes_cache = None
        self._estimate_particle_count_cache = 0
        self._estimate_weight_mass_cache = np.nan
        self._particle_estimate_diagnostics_cache = None

    def _set_endpoint_nodes(self, endpoint_nodes):
        self.last_endpoint_nodes = np.ascontiguousarray(endpoint_nodes, dtype=np.float64)
        if self.cuda_state:
            with torch.cuda.stream(self.cuda_stream):
                self.last_endpoint_nodes_t = torch.as_tensor(
                    self.last_endpoint_nodes,
                    dtype=torch.float32,
                    device=self.device,
                ).contiguous()

    def step(self, measurement, dt=1.0 / 30.0, count_lost=True):
        return update_cable_particle_filters([self], [measurement], dt=dt, count_lost=count_lost)[0]

    def _prepare_update(self, measurement, dt, count_lost=True):
        self._reset_stage_seconds()
        self.last_support_points = np.empty((0, 3), dtype=np.float32)
        self.last_endpoint_conditioned_proposal_ratio = 0.0
        self.last_global_random_particle_ratio = 0.0
        self.last_mean_ownership_responsibility = np.nan
        self.last_ownership_entropy = np.nan
        self.last_ownership_effective_point_count = 0.0
        prepare_start = time.perf_counter()
        dt = float(np.clip(dt, 1e-3, 0.20))
        measurement_points = self._measurement_points(measurement)
        endpoint_nodes = self._measurement_endpoint_nodes(measurement)
        if endpoint_nodes is not None:
            self._set_endpoint_nodes(endpoint_nodes)
        if measurement_points is not None:
            self.last_support_points = np.ascontiguousarray(measurement_points, dtype=np.float32)
            self.last_measurement_point_count = int(len(measurement_points))
        elif count_lost:
            self.last_measurement_point_count = 0
        self._record_stage("prepare", prepare_start)

        if measurement_points is not None and endpoint_nodes is not None:
            stage_start = time.perf_counter()
            self._estimate_endpoint_tangents(measurement_points, endpoint_nodes)
            self._record_stage("tangent", stage_start)
            initialized_before = self.initialized
            stage_start = time.perf_counter()
            if initialized_before:
                self._predict_transition(dt)
                self._record_stage("predict", stage_start)
            else:
                self._initialize(endpoint_nodes)
                self._record_stage("initialize", stage_start)
            return ParticleFilterUpdateContext(
                measurement=measurement,
                measurement_points=measurement_points,
                endpoint_nodes=endpoint_nodes,
                dt=dt,
                measurement_used=bool(self.initialized),
                initialized_this_frame=bool(self.initialized and not initialized_before),
            )

        if not self.initialized:
            return ParticleFilterUpdateContext(measurement, None, endpoint_nodes, dt)

        if count_lost:
            self.lost_frames += 1
        if self.lost_frames > int(self.config.max_prediction_frames):
            self.initialized = False
            return ParticleFilterUpdateContext(measurement, None, endpoint_nodes, dt)
        self.last_motion_noise_scale = min(
            1.0 + 0.25 * self.lost_frames,
            float(self.config.max_motion_noise_scale),
        )
        stage_start = time.perf_counter()
        self._predict(dt, self.last_motion_noise_scale)
        self._record_stage("predict", stage_start)
        if count_lost:
            self._set_visibility(np.zeros(self.segment_count, dtype=bool))
        return ParticleFilterUpdateContext(
            measurement=measurement,
            measurement_points=None,
            endpoint_nodes=endpoint_nodes,
            dt=dt,
            prediction_only=True,
        )

    def _initialize(self, endpoint_nodes):
        if self.segment_length_m is None or endpoint_nodes is None:
            return
        nodes = endpoint_tangent_bridge_nodes(
            endpoint_nodes,
            self.last_endpoint_tangents,
            node_count=self.node_count,
            segment_length_m=self.segment_length_m,
        )
        if nodes is None:
            return
        self.last_ordered_measurement_nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        count = max(32, int(self.config.particle_count))
        global_count = transition_component_count(
            count,
            float(getattr(self.config, "global_random_particle_ratio", 0.10)),
        )
        conditioned_count = count - global_count
        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                nodes_t = torch.as_tensor(nodes, dtype=torch.float32, device=self.device)
                conditioned = endpoint_conditioned_chains_torch(
                    nodes_t[None, :, :].expand(conditioned_count, -1, -1).clone(),
                    self.last_endpoint_nodes_t,
                    self.last_endpoint_tangents_t,
                    self.last_endpoint_tangent_confidence_t,
                    self.segment_length_m,
                    self.config,
                    self.torch_generator,
                    self.cuda_stream,
                    initialization=True,
                    slack_m=max(
                        0.0,
                        self.segment_length_m * self.segment_count
                        - float(np.linalg.norm(endpoint_nodes[-1] - endpoint_nodes[0])),
                    ),
                )
                global_particles = global_endpoint_chains_torch(
                    global_count,
                    self.node_count,
                    self.last_endpoint_nodes_t,
                    self.segment_length_m,
                    self.config,
                    self.torch_generator,
                    self.cuda_stream,
                )
                self.particles = torch.cat((conditioned, global_particles), dim=0).contiguous()
                self._reset_velocities(count)
                self.weights = transition_component_weights_torch(
                    conditioned_count,
                    global_count,
                    conditioned_mass=1.0 - float(global_count) / float(count),
                    device=self.device,
                )
        else:
            conditioned = endpoint_conditioned_chains(
                np.repeat(nodes[None, :, :], conditioned_count, axis=0),
                endpoint_nodes,
                self.last_endpoint_tangents,
                self.last_endpoint_tangent_confidence,
                self.segment_length_m,
                self.config,
                self.rng,
                initialization=True,
            )
            global_particles = global_endpoint_chains(
                global_count,
                self.node_count,
                endpoint_nodes,
                self.segment_length_m,
                self.config,
                self.rng,
            )
            self.particles = np.ascontiguousarray(np.concatenate((conditioned, global_particles), axis=0))
            self._reset_velocities(count)
            self.weights = transition_component_weights(
                conditioned_count,
                global_count,
                conditioned_mass=1.0 - float(global_count) / float(count),
            )
        self._invalidate_estimate_cache()
        self.initialized = True
        self.last_motion_noise_scale = 1.0
        self.last_endpoint_conditioned_proposal_ratio = float(conditioned_count) / float(count)
        self.last_global_random_particle_ratio = float(global_count) / float(count)
        self.measurement_update_count = 1

    def _estimate_endpoint_tangents(self, measurement_points, endpoint_nodes):
        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                points_t = torch.as_tensor(
                    measurement_points,
                    dtype=torch.float32,
                    device=self.device,
                ).contiguous()
                endpoints_t = self.last_endpoint_nodes_t
                tangents_t, confidence_t, support_count_t = estimate_endpoint_tangents_torch(
                    points_t,
                    endpoints_t,
                    min_radius_m=float(self.config.endpoint_tangent_min_radius_m),
                    radius_m=float(self.config.endpoint_tangent_radius_m),
                    sigma_m=float(self.config.endpoint_tangent_sigma_m),
                    min_points=int(self.config.endpoint_tangent_min_points),
                )
                self.last_endpoint_tangents_t = tangents_t.contiguous()
                self.last_endpoint_tangent_confidence_t = confidence_t.contiguous()
                self.last_endpoint_tangents = tangents_t.cpu().numpy().astype(np.float32, copy=False)
                self.last_endpoint_tangent_confidence = confidence_t.cpu().numpy().astype(np.float32, copy=False)
                self.last_endpoint_tangent_support_count = support_count_t.cpu().numpy().astype(np.int32, copy=False)
            return
        tangents, confidence, support_count = estimate_endpoint_tangents(
            measurement_points,
            endpoint_nodes,
            min_radius_m=float(self.config.endpoint_tangent_min_radius_m),
            radius_m=float(self.config.endpoint_tangent_radius_m),
            sigma_m=float(self.config.endpoint_tangent_sigma_m),
            min_points=int(self.config.endpoint_tangent_min_points),
        )
        self.last_endpoint_tangents = tangents
        self.last_endpoint_tangent_confidence = confidence
        self.last_endpoint_tangent_support_count = support_count

    def _predict_transition(self, dt):
        if self.particles is None or self.weights is None or self.segment_length_m is None:
            return
        total_count = len(self.particles)
        global_ratio = float(np.clip(self.config.global_random_particle_ratio, 0.0, 0.95))
        conditioned_ratio = float(np.clip(
            self.config.endpoint_conditioned_proposal_ratio,
            0.0,
            1.0 - global_ratio,
        ))
        global_count = transition_component_count(total_count, global_ratio)
        conditioned_count = transition_component_count(total_count, conditioned_ratio)
        if global_count + conditioned_count >= total_count:
            conditioned_count = max(0, total_count - global_count - 1)
        local_count = total_count - conditioned_count - global_count
        local_mass = 1.0 - conditioned_ratio - global_ratio
        self.last_motion_noise_scale = 1.0

        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                probabilities = torch.clamp_min(self.weights, 0.0)
                probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
                parent_indices = torch.multinomial(
                    probabilities,
                    local_count + conditioned_count,
                    replacement=True,
                    generator=self.torch_generator,
                )
                self._ensure_velocity_array()
                local_particles = self.particles.index_select(0, parent_indices[:local_count]).contiguous()
                local_velocities = self.node_velocities.index_select(0, parent_indices[:local_count]).contiguous()
                if self._velocity_enabled():
                    damping = float(np.clip(self.config.velocity_damping, 0.0, 1.0))
                    local_velocities.mul_(damping).add_(
                        torch.randn(
                            local_velocities.shape,
                            dtype=torch.float32,
                            device=self.device,
                            generator=self.torch_generator,
                        ) * float(self.config.velocity_process_std_mps)
                    )
                    local_velocities = clip_vector_norms_torch(local_velocities, self.config.max_node_speed_mps)
                    predicted = local_particles + local_velocities * float(dt)
                    local_directions = normalize_vectors_torch(predicted[:, 1:, :] - predicted[:, :-1, :])
                else:
                    local_directions = normalize_vectors_torch(local_particles[:, 1:, :] - local_particles[:, :-1, :])
                    local_directions = normalize_vectors_torch(
                        local_directions
                        + torch.randn(
                            local_directions.shape,
                            dtype=torch.float32,
                            device=self.device,
                            generator=self.torch_generator,
                        ) * float(self.config.process_direction_std)
                    )
                    local_velocities.zero_()
                local_particles = build_chains_torch(
                    self.last_endpoint_nodes_t[0][None, :].expand(local_count, -1),
                    local_directions,
                    self.segment_length_m,
                )
                local_particles = constrain_chains_cuda(
                    local_particles,
                    self.last_endpoint_nodes_t,
                    self.segment_length_m,
                    int(self.config.endpoint_constraint_iterations),
                    float(self.config.endpoint_constraint_tolerance_m),
                    self.cuda_stream,
                )
                conditioned_base = self.particles.index_select(
                    0,
                    parent_indices[local_count:],
                ).contiguous()
                conditioned_particles = endpoint_conditioned_chains_torch(
                    conditioned_base,
                    self.last_endpoint_nodes_t,
                    self.last_endpoint_tangents_t,
                    self.last_endpoint_tangent_confidence_t,
                    self.segment_length_m,
                    self.config,
                    self.torch_generator,
                    self.cuda_stream,
                    slack_m=max(
                        0.0,
                        self.segment_length_m * self.segment_count
                        - float(np.linalg.norm(self.last_endpoint_nodes[-1] - self.last_endpoint_nodes[0])),
                    ),
                )
                global_particles = global_endpoint_chains_torch(
                    global_count,
                    self.node_count,
                    self.last_endpoint_nodes_t,
                    self.segment_length_m,
                    self.config,
                    self.torch_generator,
                    self.cuda_stream,
                )
                self.particles = torch.cat(
                    (local_particles, conditioned_particles, global_particles),
                    dim=0,
                ).contiguous()
                zero_velocities = torch.zeros(
                    (conditioned_count + global_count, self.node_count, 3),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.node_velocities = torch.cat((local_velocities, zero_velocities), dim=0).contiguous()
                self.weights = transition_mixture_weights_torch(
                    local_count,
                    conditioned_count,
                    global_count,
                    local_mass,
                    conditioned_ratio,
                    global_ratio,
                    self.device,
                )
                permutation = torch.randperm(total_count, device=self.device, generator=self.torch_generator)
                self.particles = self.particles.index_select(0, permutation).contiguous()
                self.node_velocities = self.node_velocities.index_select(0, permutation).contiguous()
                self.weights = self.weights.index_select(0, permutation).contiguous()
        else:
            probabilities = np.maximum(np.asarray(self.weights, dtype=np.float64), 0.0)
            probabilities /= max(float(np.sum(probabilities)), 1e-12)
            parent_indices = self.rng.choice(
                total_count,
                size=local_count + conditioned_count,
                replace=True,
                p=probabilities,
            )
            self._ensure_velocity_array()
            local_particles = self.particles[parent_indices[:local_count]].copy()
            local_velocities = self.node_velocities[parent_indices[:local_count]].copy()
            if self._velocity_enabled():
                local_velocities *= float(np.clip(self.config.velocity_damping, 0.0, 1.0))
                local_velocities += self.rng.normal(
                    0.0,
                    float(self.config.velocity_process_std_mps),
                    local_velocities.shape,
                )
                local_velocities = clip_vector_norms(local_velocities, self.config.max_node_speed_mps)
                predicted = local_particles + local_velocities * float(dt)
                local_directions = particle_directions(predicted)
            else:
                local_directions = normalize_vectors(
                    particle_directions(local_particles)
                    + self.rng.normal(
                        0.0,
                        float(self.config.process_direction_std),
                        (local_count, self.segment_count, 3),
                    )
                )
                local_velocities.fill(0.0)
            local_particles = build_chains(
                np.repeat(np.asarray(self.last_endpoint_nodes[0])[None, :], local_count, axis=0),
                local_directions,
                self.segment_length_m,
            )
            local_particles = constrain_chains_to_endpoints(
                local_particles,
                self.last_endpoint_nodes,
                self.segment_length_m,
                iterations=int(self.config.endpoint_constraint_iterations),
                tolerance_m=float(self.config.endpoint_constraint_tolerance_m),
            )
            conditioned_particles = endpoint_conditioned_chains(
                self.particles[parent_indices[local_count:]],
                self.last_endpoint_nodes,
                self.last_endpoint_tangents,
                self.last_endpoint_tangent_confidence,
                self.segment_length_m,
                self.config,
                self.rng,
            )
            global_particles = global_endpoint_chains(
                global_count,
                self.node_count,
                self.last_endpoint_nodes,
                self.segment_length_m,
                self.config,
                self.rng,
            )
            self.particles = np.ascontiguousarray(np.concatenate(
                (local_particles, conditioned_particles, global_particles),
                axis=0,
            ))
            self.node_velocities = np.ascontiguousarray(np.concatenate(
                (
                    local_velocities,
                    np.zeros((conditioned_count + global_count, self.node_count, 3), dtype=np.float64),
                ),
                axis=0,
            ))
            self.weights = transition_mixture_weights(
                local_count,
                conditioned_count,
                global_count,
                local_mass,
                conditioned_ratio,
                global_ratio,
            )
            permutation = self.rng.permutation(total_count)
            self.particles = self.particles[permutation].copy()
            self.node_velocities = self.node_velocities[permutation].copy()
            self.weights = self.weights[permutation].copy()

        self.last_endpoint_conditioned_proposal_ratio = float(conditioned_count) / float(total_count)
        self.last_global_random_particle_ratio = float(global_count) / float(total_count)
        self.measurement_update_count += 1
        self._invalidate_estimate_cache()

    def _predict(self, dt, noise_scale=1.0):
        if self.particles is None or self.segment_length_m is None:
            return
        noise_scale = float(np.clip(noise_scale, 0.25, self.config.max_motion_noise_scale))
        self.last_motion_noise_scale = noise_scale

        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                if bool(getattr(self.config, "velocity_enabled", True)):
                    self._ensure_velocity_array()
                    damping = float(np.clip(getattr(self.config, "velocity_damping", 0.85), 0.0, 1.0))
                    velocity_noise = torch.randn(
                        self.node_velocities.shape,
                        dtype=torch.float32,
                        device=self.device,
                        generator=self.torch_generator,
                    ) * (float(getattr(self.config, "velocity_process_std_mps", 0.03)) * noise_scale)
                    self.node_velocities.mul_(damping).add_(velocity_noise)
                    self._clip_velocities()
                    predicted = self.particles + self.node_velocities * float(dt)
                    starts = predicted[:, 0, :]
                    directions = normalize_vectors_torch(predicted[:, 1:, :] - predicted[:, :-1, :])
                else:
                    starts = self.particles[:, 0, :] + torch.randn(
                        (len(self.particles), 3),
                        dtype=torch.float32,
                        device=self.device,
                        generator=self.torch_generator,
                    ) * (float(self.config.process_node_std_m) * noise_scale)
                    directions = normalize_vectors_torch(self.particles[:, 1:, :] - self.particles[:, :-1, :])
                    directions = normalize_vectors_torch(
                        directions
                        + torch.randn(
                            directions.shape,
                            dtype=torch.float32,
                            device=self.device,
                            generator=self.torch_generator,
                        ) * (float(self.config.process_direction_std) * noise_scale)
                    )
                directions = smooth_particle_directions_torch(
                    directions,
                    passes=int(getattr(self.config, "direction_smooth_passes", 1)),
                )
                self.particles = build_chains_torch(starts, directions, self.segment_length_m)
                self._constrain_particles_to_known_endpoints()
            self._invalidate_estimate_cache()
            return

        if bool(getattr(self.config, "velocity_enabled", True)):
            self._ensure_velocity_array()
            damping = float(np.clip(getattr(self.config, "velocity_damping", 0.85), 0.0, 1.0))
            self.node_velocities *= damping
            self.node_velocities += self.rng.normal(
                0.0,
                float(getattr(self.config, "velocity_process_std_mps", 0.03)) * noise_scale,
                self.node_velocities.shape,
            )
            self._clip_velocities()
            predicted = self.particles + self.node_velocities * float(dt)
            starts = predicted[:, 0, :]
            directions = particle_directions(predicted)
        else:
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
        self._constrain_particles_to_known_endpoints()
        self._invalidate_estimate_cache()

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
            support_points_xyz=self.last_support_points.copy(),
            visible_segments=self.last_visible_segments.copy(),
            visible_nodes=self.last_visible_nodes.copy(),
            endpoint_conditioned_proposal_ratio=float(self.last_endpoint_conditioned_proposal_ratio),
            global_random_particle_ratio=float(self.last_global_random_particle_ratio),
            mean_ownership_responsibility=float(self.last_mean_ownership_responsibility),
            ownership_entropy=float(self.last_ownership_entropy),
            ownership_effective_point_count=float(self.last_ownership_effective_point_count),
            endpoint_tangent_confidence=self.last_endpoint_tangent_confidence.copy(),
            endpoint_tangent_support_count=self.last_endpoint_tangent_support_count.copy(),
            estimate_particle_count=int(self._estimate_particle_count_cache),
            estimate_weight_mass=float(self._estimate_weight_mass_cache),
            mean_node_speed_mps=float(self._weighted_node_speed()),
            stage_seconds=self._resolved_stage_seconds(),
            particle_diagnostics=self._particle_estimate_diagnostics_cache,
        )

    def _estimate_nodes(self):
        if self._estimate_nodes_cache is not None:
            return self._estimate_nodes_cache.copy()
        particle_count = len(self.particles)
        top_count = int(np.clip(
            getattr(self.config, "estimate_top_particle_count", 32),
            1,
            particle_count,
        ))
        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                rank_weights = torch.where(
                    torch.isfinite(self.weights),
                    self.weights,
                    torch.full_like(self.weights, -torch.inf),
                )
                top_weights, _top_indices = torch.topk(
                    rank_weights,
                    k=top_count,
                    largest=True,
                    sorted=False,
                )
                cutoff = torch.min(top_weights)
                selected = torch.isfinite(rank_weights) & (rank_weights >= cutoff)
                selected_float = selected.to(self.particles.dtype)
                selected_count_t = torch.sum(selected_float).clamp_min(1.0)
                selected_particles_t = self.particles[selected]
                map_nodes_t = self.particles[torch.argmax(rank_weights)]
                nodes_t = torch.sum(
                    self.particles * selected_float[:, None, None],
                    dim=0,
                ) / selected_count_t
                projected_on_cuda = self.last_endpoint_nodes_t is not None
                if projected_on_cuda:
                    nodes_t = constrain_chains_cuda(
                        nodes_t[None, :, :].contiguous(),
                        self.last_endpoint_nodes_t,
                        self.segment_length_m,
                        int(getattr(self.config, "endpoint_constraint_iterations", 16)),
                        float(getattr(self.config, "endpoint_constraint_tolerance_m", 1e-4)),
                        self.cuda_stream,
                    )[0]
                weight_mass_t = torch.sum(
                    torch.clamp_min(rank_weights, 0.0) * selected_float
                )
                nodes = nodes_t.cpu().numpy().astype(np.float32, copy=False)
                map_nodes = map_nodes_t.cpu().numpy().astype(np.float32, copy=False)
                selected_particles = selected_particles_t.cpu().numpy().astype(np.float32, copy=False)
                weight_mass = float(weight_mass_t.item())
                selected_count = int(selected_count_t.item())
            self._estimate_nodes_cache = (
                np.ascontiguousarray(nodes, dtype=np.float32)
                if projected_on_cuda
                else self._project_estimate_nodes(nodes)
            )
            self._estimate_particle_count_cache = selected_count
            self._estimate_weight_mass_cache = weight_mass
            self._particle_estimate_diagnostics_cache = particle_estimate_diagnostics(
                self._estimate_nodes_cache,
                map_nodes,
                selected_particles,
                endpoint_tangents=self.last_endpoint_tangents,
                endpoint_tangent_confidence=self.last_endpoint_tangent_confidence,
                endpoint_tangent_support_count=self.last_endpoint_tangent_support_count,
                mean_ownership_responsibility=self.last_mean_ownership_responsibility,
                ownership_entropy=self.last_ownership_entropy,
                visible_segment_fraction=float(np.mean(self.last_visible_segments)),
                endpoint_conditioned_proposal_ratio=self.last_endpoint_conditioned_proposal_ratio,
            )
            return self._estimate_nodes_cache.copy()

        rank_weights = np.where(np.isfinite(self.weights), self.weights, -np.inf)
        if top_count >= particle_count:
            cutoff = float(np.min(rank_weights))
        else:
            cutoff = float(np.partition(rank_weights, particle_count - top_count)[particle_count - top_count])
        selected = np.isfinite(rank_weights) & (rank_weights >= cutoff)
        selected_count = int(np.count_nonzero(selected))
        if selected_count == 0:
            selected = np.ones(particle_count, dtype=bool)
            selected_count = particle_count
        nodes = np.mean(self.particles[selected], axis=0)
        weight_mass = float(np.sum(np.maximum(rank_weights[selected], 0.0)))
        self._estimate_nodes_cache = self._project_estimate_nodes(nodes)
        self._estimate_particle_count_cache = selected_count
        self._estimate_weight_mass_cache = weight_mass
        map_nodes = self.particles[int(np.argmax(rank_weights))]
        self._particle_estimate_diagnostics_cache = particle_estimate_diagnostics(
            self._estimate_nodes_cache,
            map_nodes,
            self.particles[selected],
            endpoint_tangents=self.last_endpoint_tangents,
            endpoint_tangent_confidence=self.last_endpoint_tangent_confidence,
            endpoint_tangent_support_count=self.last_endpoint_tangent_support_count,
            mean_ownership_responsibility=self.last_mean_ownership_responsibility,
            ownership_entropy=self.last_ownership_entropy,
            visible_segment_fraction=float(np.mean(self.last_visible_segments)),
            endpoint_conditioned_proposal_ratio=self.last_endpoint_conditioned_proposal_ratio,
        )
        return self._estimate_nodes_cache.copy()

    def _project_estimate_nodes(self, nodes):
        projected = project_equal_length_chain(
            nodes,
            self.segment_length_m,
            node_count=self.node_count,
        )
        if projected is None:
            projected = np.ascontiguousarray(nodes, dtype=np.float32)
        if self.last_endpoint_nodes is not None:
            projected = constrain_chain_to_endpoints(
                projected,
                self.last_endpoint_nodes,
                self.segment_length_m,
                iterations=int(getattr(self.config, "endpoint_constraint_iterations", 16)),
                tolerance_m=float(getattr(self.config, "endpoint_constraint_tolerance_m", 1e-4)),
            )
        return np.ascontiguousarray(projected, dtype=np.float32)

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

    def _velocity_enabled(self):
        return bool(getattr(self.config, "velocity_enabled", True))

    def _reset_stage_seconds(self):
        self.last_stage_seconds = {}
        self._cuda_stage_events = []
        self._cuda_previous_event = None
        if self.cuda_state:
            event = torch.cuda.Event(enable_timing=True)
            event.record(self.cuda_stream)
            self._cuda_previous_event = event

    def _record_stage(self, name, start_time):
        elapsed = max(0.0, time.perf_counter() - float(start_time))
        self.last_stage_seconds[str(name)] = self.last_stage_seconds.get(str(name), 0.0) + elapsed
        if self.cuda_state and self._cuda_previous_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(self.cuda_stream)
            self._cuda_stage_events.append((str(name), self._cuda_previous_event, end_event))
            self._cuda_previous_event = end_event

    def _resolved_stage_seconds(self):
        resolved = dict(self.last_stage_seconds)
        if not self.cuda_state or not self._cuda_stage_events:
            return resolved
        self._cuda_stage_events[-1][2].synchronize()
        gpu_seconds = {}
        for name, start_event, end_event in self._cuda_stage_events:
            elapsed = max(0.0, float(start_event.elapsed_time(end_event)) / 1000.0)
            gpu_seconds[name] = gpu_seconds.get(name, 0.0) + elapsed
        for name, elapsed in gpu_seconds.items():
            resolved[name] = max(float(resolved.get(name, 0.0)), elapsed)
        return resolved

    def _reset_velocities(self, count):
        if not self._velocity_enabled():
            self.node_velocities = None
            return
        count = max(0, int(count))
        if self.cuda_state:
            self.node_velocities = torch.zeros(
                (count, self.node_count, 3),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.node_velocities = np.zeros((count, self.node_count, 3), dtype=np.float64)

    def _ensure_velocity_array(self):
        if not self._velocity_enabled() or self.particles is None:
            return False
        count = len(self.particles)
        shape = (count, self.node_count, 3)
        if self.node_velocities is None or self.node_velocities.shape != shape:
            if self.cuda_state:
                self.node_velocities = torch.zeros(shape, dtype=torch.float32, device=self.device)
            else:
                self.node_velocities = np.zeros(shape, dtype=np.float64)
        return True

    def _clip_velocities(self, indices=None):
        if self.node_velocities is None:
            return
        if self.cuda_state:
            target = self.node_velocities if indices is None else self.node_velocities[indices]
            maximum = float(getattr(self.config, "max_node_speed_mps", 1.0))
            norms = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
            clipped = target * torch.clamp(maximum / norms.clamp_min(1e-12), max=1.0)
            if indices is None:
                self.node_velocities = clipped
            else:
                self.node_velocities[indices] = clipped
            return
        if indices is None:
            self.node_velocities = clip_vector_norms(
                self.node_velocities,
                float(getattr(self.config, "max_node_speed_mps", 1.0)),
            )
            return
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(indices) == 0:
            return
        self.node_velocities[indices] = clip_vector_norms(
            self.node_velocities[indices],
            float(getattr(self.config, "max_node_speed_mps", 1.0)),
        )

    def _weighted_node_speed(self):
        if self.node_velocities is None or self.weights is None:
            return 0.0
        if self.cuda_state:
            with torch.cuda.stream(self.cuda_stream):
                speeds = torch.mean(torch.linalg.vector_norm(self.node_velocities, dim=2), dim=1)
                value = torch.sum(speeds * self.weights)
                return float(value.item()) if bool(torch.isfinite(value).item()) else 0.0
        speeds = np.mean(np.linalg.norm(self.node_velocities, axis=2), axis=1)
        return weighted_mean_or_zero(speeds, self.weights)

    def _measurement_endpoint_nodes(self, measurement):
        if measurement is None:
            return None
        endpoint_nodes = valid_points(getattr(measurement, "endpoint_nodes", None))
        if len(endpoint_nodes) < 2:
            return None
        endpoint_nodes = np.ascontiguousarray([endpoint_nodes[0], endpoint_nodes[-1]], dtype=np.float64)
        if not np.all(np.isfinite(endpoint_nodes)):
            return None
        return endpoint_nodes

    def _constrain_particles_to_known_endpoints(self, indices=None):
        if self.particles is None or self.segment_length_m is None or self.last_endpoint_nodes is None:
            return
        if self.cuda_state:
            if self.last_endpoint_nodes_t is None:
                raise RuntimeError("CUDA particle constraints require endpoint tensors on the PF device.")
            if indices is None:
                self.particles = constrain_chains_cuda(
                    self.particles,
                    self.last_endpoint_nodes_t,
                    self.segment_length_m,
                    int(getattr(self.config, "endpoint_constraint_iterations", 16)),
                    float(getattr(self.config, "endpoint_constraint_tolerance_m", 1e-4)),
                    self.cuda_stream,
                )
            else:
                if not torch.is_tensor(indices):
                    indices = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
                indices = indices.reshape(-1)
                if len(indices) == 0:
                    return
                selected = self.particles.index_select(0, indices).contiguous()
                selected = constrain_chains_cuda(
                    selected,
                    self.last_endpoint_nodes_t,
                    self.segment_length_m,
                    int(getattr(self.config, "endpoint_constraint_iterations", 16)),
                    float(getattr(self.config, "endpoint_constraint_tolerance_m", 1e-4)),
                    self.cuda_stream,
                )
                self.particles.index_copy_(0, indices, selected)
            self._invalidate_estimate_cache()
            return
        if indices is None:
            self.particles = constrain_chains_to_endpoints(
                self.particles,
                self.last_endpoint_nodes,
                self.segment_length_m,
                iterations=int(getattr(self.config, "endpoint_constraint_iterations", 16)),
                tolerance_m=float(getattr(self.config, "endpoint_constraint_tolerance_m", 1e-4)),
            )
            return
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(indices) == 0:
            return
        self.particles[indices] = constrain_chains_to_endpoints(
            self.particles[indices],
            self.last_endpoint_nodes,
            self.segment_length_m,
            iterations=int(getattr(self.config, "endpoint_constraint_iterations", 16)),
            tolerance_m=float(getattr(self.config, "endpoint_constraint_tolerance_m", 1e-4)),
        )

    def _set_visibility(self, visible_segments):
        visible_segments = np.asarray(visible_segments, dtype=bool).reshape(-1)
        if len(visible_segments) != self.segment_count:
            visible_segments = np.zeros(self.segment_count, dtype=bool)
        self.last_visible_segments = visible_segments.copy()
        self.last_visible_nodes = node_visibility_from_segments(visible_segments, self.node_count)

    def _effective_sample_size(self):
        if self.cuda_state:
            with torch.cuda.stream(self.cuda_stream):
                value = 1.0 / torch.sum(self.weights * self.weights).clamp_min(1e-12)
                return float(value.item())
        return float(1.0 / max(np.sum(self.weights * self.weights), 1e-12))


def update_cable_particle_filters(particle_filters, measurements, dt=1.0 / 30.0, count_lost=True):
    """Advance one or more cable PFs with one shared posterior-consensus update.

    Prediction is performed independently because each cable has its own
    endpoints and dynamics. Measurement ownership is deliberately coupled:
    the shared semantic cable cloud is evaluated against all active predicted
    populations before any filter receives a likelihood update.
    """

    filters = list(particle_filters or ())
    observations = list(measurements or ())
    if len(filters) != len(observations):
        raise ValueError("particle_filters and measurements must have the same length.")
    if not filters:
        return []

    cuda_filters = [particle_filter for particle_filter in filters if particle_filter.cuda_state]
    if cuda_filters:
        if len(cuda_filters) != len(filters):
            raise RuntimeError("A coupled PF update cannot mix CPU and CUDA filter states.")
        shared_stream = cuda_filters[0].cuda_stream
        for particle_filter in cuda_filters[1:]:
            particle_filter.cuda_stream = shared_stream

    contexts = [
        particle_filter._prepare_update(observation, dt=dt, count_lost=count_lost)
        for particle_filter, observation in zip(filters, observations)
    ]
    active_indices = [
        index
        for index, context in enumerate(contexts)
        if context.measurement_used and filters[index].initialized
    ]
    if active_indices:
        shared_points = contexts[active_indices[0]].measurement_points
        ownership_indices = [
            index
            for index, particle_filter in enumerate(filters)
            if particle_filter.initialized
        ]
        ownership_filters = [filters[index] for index in ownership_indices]
        update_mask = [index in active_indices for index in ownership_indices]
        stage_start = time.perf_counter()
        posterior_consensus_measurement_update(
            ownership_filters,
            shared_points,
            update_mask=update_mask,
        )
        for index in active_indices:
            particle_filter = filters[index]
            particle_filter._record_stage("consensus", stage_start)

    results = []
    for particle_filter, context in zip(filters, contexts):
        if not particle_filter.initialized:
            results.append(None)
            continue
        stage_start = time.perf_counter()
        if context.measurement_used:
            nodes = particle_filter._estimate_nodes()
            update_posterior_velocity(particle_filter, nodes, context.dt, measurement_used=True)
            particle_filter.lost_frames = 0
            result = particle_filter._estimate(measurement_used=True, prediction_only=False)
        else:
            result = particle_filter._estimate(measurement_used=False, prediction_only=True)
            if result is not None:
                update_posterior_velocity(
                    particle_filter,
                    result.points_xyz,
                    context.dt,
                    measurement_used=False,
                )
        particle_filter._record_stage("estimate", stage_start)
        if result is not None:
            result.stage_seconds = particle_filter._resolved_stage_seconds()
        results.append(result)
    return results


def posterior_consensus_measurement_update(
    particle_filters,
    measurement_points,
    update_mask=None,
):
    """Partition shared support and update selected PF populations.

    Every supplied population participates in the ownership denominator.  This
    includes prediction-only cables during short occlusions, which prevents a
    visible cable from absorbing support that remains geometrically compatible
    with the occluded cable.  ``update_mask`` controls which populations receive
    a measurement likelihood; omitted means that all populations are updated.
    """
    filters = list(particle_filters or ())
    points = valid_points(measurement_points)
    if not filters or len(points) == 0:
        return
    if update_mask is None:
        update_mask = [True] * len(filters)
    else:
        update_mask = [bool(value) for value in update_mask]
        if len(update_mask) != len(filters):
            raise ValueError("update_mask must match particle_filters.")
    if all(particle_filter.cuda_state for particle_filter in filters):
        _posterior_consensus_measurement_update_cuda(filters, points, update_mask)
        return
    if any(particle_filter.cuda_state for particle_filter in filters):
        raise RuntimeError("Posterior consensus requires all filter states on one backend.")
    _posterior_consensus_measurement_update_cpu(filters, points, update_mask)


def _posterior_consensus_measurement_update_cuda(filters, points, update_mask):
    stream = filters[0].cuda_stream
    device = filters[0].device
    particle_count = len(filters[0].particles)
    node_count = filters[0].node_count
    if any(len(item.particles) != particle_count or item.node_count != node_count for item in filters):
        raise ValueError("Coupled CUDA filters must use equal particle and node counts.")
    with torch.inference_mode(), torch.cuda.stream(stream):
        points_t = torch.as_tensor(points, dtype=torch.float32, device=device).contiguous()
        particles_t = torch.stack([item.particles for item in filters], dim=0).contiguous()
        weights_t = torch.stack([item.weights for item in filters], dim=0).contiguous()
        squared_t, nearest_t = particle_point_distances_cuda(particles_t, points_t, stream)

        ownership_sigma = max(
            1e-5,
            float(np.mean([item.config.measurement_node_std_m for item in filters])),
        )
        compatibility_t = torch.sum(
            weights_t[:, :, None]
            * torch.exp(-0.5 * squared_t / (ownership_sigma * ownership_sigma)),
            dim=1,
        )
        outlier_prior = float(np.clip(
            np.mean([item.config.ownership_outlier_prior for item in filters]),
            1e-4,
            0.95,
        ))
        cable_prior = (1.0 - outlier_prior) / float(len(filters))
        outlier_component = outlier_prior * max(
            1e-6,
            float(np.mean([item.config.ownership_outlier_likelihood for item in filters])),
        )
        cable_components_t = cable_prior * compatibility_t
        denominator_t = torch.sum(cable_components_t, dim=0) + outlier_component
        responsibilities_t = cable_components_t / denominator_t.clamp_min(1e-12)
        responsibility_floor = max(
            0.0,
            float(np.mean([item.config.ownership_min_responsibility for item in filters])),
        )
        if responsibility_floor > 0.0:
            responsibilities_t = torch.clamp_min(responsibilities_t, responsibility_floor)
            normalizer_t = torch.sum(responsibilities_t, dim=0) + outlier_component / denominator_t.clamp_min(1e-12)
            responsibilities_t = responsibilities_t / normalizer_t.clamp_min(1e-12)
        outlier_responsibility_t = torch.clamp(
            1.0 - torch.sum(responsibilities_t, dim=0),
            min=0.0,
            max=1.0,
        )
        assignment_probabilities_t = torch.cat(
            (responsibilities_t, outlier_responsibility_t[None, :]),
            dim=0,
        )
        entropy_t = -torch.sum(
            assignment_probabilities_t * torch.log(assignment_probabilities_t.clamp_min(1e-12)),
            dim=0,
        ) / np.log(float(len(filters) + 1))

        for cable_index, particle_filter in enumerate(filters):
            if not update_mask[cable_index]:
                continue
            responsibility_t = responsibilities_t[cable_index]
            robust_distance = max(
                float(particle_filter.config.robust_distance_m),
                float(particle_filter.config.measurement_node_std_m),
            )
            robust_squared_t = torch.clamp(
                squared_t[cable_index],
                max=robust_distance * robust_distance,
            )
            responsibility_sum_t = torch.sum(responsibility_t).clamp_min(1e-6)
            scores_t = torch.sum(
                robust_squared_t * responsibility_t[None, :],
                dim=1,
            ) / responsibility_sum_t
            scores_t = scores_t + posterior_coverage_penalty_torch(
                nearest_t[cable_index].to(torch.int64),
                responsibility_t,
                particle_filter.segment_count,
                particle_filter.config.coverage_penalty_m,
                particle_filter.config.coverage_min_fraction,
            )
            if float(particle_filter.config.bend_penalty_m) > 0.0:
                segment = particle_filter.particles[:, 1:, :] - particle_filter.particles[:, :-1, :]
                directions = normalize_vectors_torch(segment)
                dots = torch.sum(directions[:, :-1, :] * directions[:, 1:, :], dim=2).clamp(-1.0, 1.0)
                bend = torch.mean(torch.clamp(1.0 - dots, min=0.0), dim=1)
                scores_t = scores_t + bend * float(particle_filter.config.bend_penalty_m) ** 2
            sigma = max(float(particle_filter.config.measurement_node_std_m), 1e-5)
            log_weights_t = torch.log(particle_filter.weights.clamp_min(1e-20)) - 0.5 * scores_t / (sigma * sigma)
            endpoint_residual_t = torch.maximum(
                torch.linalg.vector_norm(
                    particle_filter.particles[:, 0, :] - particle_filter.last_endpoint_nodes_t[0][None, :],
                    dim=1,
                ),
                torch.linalg.vector_norm(
                    particle_filter.particles[:, -1, :] - particle_filter.last_endpoint_nodes_t[-1][None, :],
                    dim=1,
                ),
            )
            valid_t = endpoint_residual_t <= max(
                float(particle_filter.config.endpoint_constraint_tolerance_m),
                1e-7,
            )
            finite_log_weights_t = torch.where(valid_t, log_weights_t, torch.full_like(log_weights_t, -torch.inf))
            maximum_t = torch.max(finite_log_weights_t)
            unnormalized_t = torch.where(
                valid_t,
                torch.exp(finite_log_weights_t - maximum_t),
                torch.zeros_like(finite_log_weights_t),
            )
            total_t = torch.sum(unnormalized_t)
            valid_uniform_t = valid_t.to(torch.float32) / valid_t.to(torch.float32).sum().clamp_min(1.0)
            fallback_t = torch.where(
                torch.any(valid_t),
                valid_uniform_t,
                torch.full_like(valid_uniform_t, 1.0 / len(valid_uniform_t)),
            )
            particle_filter.weights = torch.where(
                torch.isfinite(total_t) & (total_t > 1e-20),
                unnormalized_t / total_t.clamp_min(1e-20),
                fallback_t,
            )
            particle_filter._invalidate_estimate_cache()

            map_index_t = torch.argmax(particle_filter.weights)
            map_nearest_t = nearest_t[cable_index, map_index_t].to(torch.int64)
            map_distance_t = torch.sqrt(torch.clamp_min(squared_t[cable_index, map_index_t], 0.0))
            gate = float(particle_filter.config.occlusion_assignment_max_distance_m)
            support_weights_t = torch.where(
                map_distance_t <= gate,
                responsibility_t,
                torch.zeros_like(responsibility_t),
            )
            segment_support_t = torch.zeros(
                particle_filter.segment_count,
                dtype=torch.float32,
                device=device,
            )
            segment_support_t.scatter_add_(0, map_nearest_t, support_weights_t)
            required = (
                max(1, int(particle_filter.config.min_segment_points))
                * float(particle_filter.config.ownership_visibility_threshold)
            )
            visible_t = segment_support_t >= required
            particle_filter._set_visibility(visible_t.cpu().numpy().astype(bool, copy=False))
            particle_filter.last_mean_ownership_responsibility = float(torch.mean(responsibility_t).item())
            particle_filter.last_ownership_effective_point_count = float(responsibility_sum_t.item())
            particle_filter.last_ownership_entropy = float(torch.mean(entropy_t).item())


def _posterior_consensus_measurement_update_cpu(filters, points, update_mask):
    squared_by_cable = []
    nearest_by_cable = []
    compatibility = []
    for particle_filter in filters:
        all_squared = point_to_particle_segment_squared_distances(
            points,
            particle_filter.particles[:, :-1, :],
            particle_filter.particles[:, 1:, :],
        )
        squared = np.min(all_squared, axis=1)
        nearest = np.argmin(all_squared, axis=1)
        squared_by_cable.append(squared)
        nearest_by_cable.append(nearest)
        sigma = max(float(particle_filter.config.measurement_node_std_m), 1e-5)
        compatibility.append(np.sum(
            particle_filter.weights[:, None] * np.exp(-0.5 * squared / (sigma * sigma)),
            axis=0,
        ))
    compatibility = np.asarray(compatibility, dtype=np.float64)
    outlier_prior = float(np.clip(np.mean([item.config.ownership_outlier_prior for item in filters]), 1e-4, 0.95))
    cable_prior = (1.0 - outlier_prior) / float(len(filters))
    outlier_component = outlier_prior * max(
        1e-6,
        float(np.mean([item.config.ownership_outlier_likelihood for item in filters])),
    )
    components = cable_prior * compatibility
    denominator = np.sum(components, axis=0) + outlier_component
    responsibilities = components / np.maximum(denominator[None, :], 1e-12)
    floor = max(0.0, float(np.mean([item.config.ownership_min_responsibility for item in filters])))
    if floor > 0.0:
        responsibilities = np.maximum(responsibilities, floor)
        outlier_responsibility = outlier_component / np.maximum(denominator, 1e-12)
        responsibilities /= np.maximum(np.sum(responsibilities, axis=0) + outlier_responsibility, 1e-12)
    outlier_responsibility = np.clip(1.0 - np.sum(responsibilities, axis=0), 0.0, 1.0)
    assignment = np.concatenate((responsibilities, outlier_responsibility[None, :]), axis=0)
    entropy = -np.sum(assignment * np.log(np.maximum(assignment, 1e-12)), axis=0) / np.log(float(len(filters) + 1))

    for cable_index, particle_filter in enumerate(filters):
        if not update_mask[cable_index]:
            continue
        responsibility = responsibilities[cable_index]
        robust_distance = max(
            float(particle_filter.config.robust_distance_m),
            float(particle_filter.config.measurement_node_std_m),
        )
        robust_squared = np.minimum(squared_by_cable[cable_index], robust_distance * robust_distance)
        responsibility_sum = max(float(np.sum(responsibility)), 1e-6)
        scores = np.sum(robust_squared * responsibility[None, :], axis=1) / responsibility_sum
        scores += posterior_coverage_penalty(
            nearest_by_cable[cable_index],
            responsibility,
            particle_filter.segment_count,
            particle_filter.config.coverage_penalty_m,
            particle_filter.config.coverage_min_fraction,
        )
        if float(particle_filter.config.bend_penalty_m) > 0.0:
            scores += particle_bend_penalty(
                particle_filter.particles,
                penalty_m=float(particle_filter.config.bend_penalty_m),
            )
        sigma = max(float(particle_filter.config.measurement_node_std_m), 1e-5)
        log_weights = np.log(np.maximum(particle_filter.weights, 1e-20)) - 0.5 * scores / (sigma * sigma)
        endpoint_residual = np.maximum(
            np.linalg.norm(particle_filter.particles[:, 0, :] - particle_filter.last_endpoint_nodes[0][None, :], axis=1),
            np.linalg.norm(particle_filter.particles[:, -1, :] - particle_filter.last_endpoint_nodes[-1][None, :], axis=1),
        )
        valid = endpoint_residual <= max(float(particle_filter.config.endpoint_constraint_tolerance_m), 1e-7)
        if np.any(valid):
            log_weights[~valid] = -np.inf
            log_weights -= float(np.max(log_weights[valid]))
            particle_filter.weights = np.where(valid, np.exp(log_weights), 0.0)
            particle_filter.weights /= max(float(np.sum(particle_filter.weights)), 1e-12)
        else:
            particle_filter.weights.fill(1.0 / len(particle_filter.weights))
        particle_filter._invalidate_estimate_cache()

        map_index = int(np.argmax(particle_filter.weights))
        map_nearest = nearest_by_cable[cable_index][map_index]
        map_distance = np.sqrt(np.maximum(squared_by_cable[cable_index][map_index], 0.0))
        support_weights = np.where(
            map_distance <= float(particle_filter.config.occlusion_assignment_max_distance_m),
            responsibility,
            0.0,
        )
        segment_support = np.bincount(
            map_nearest,
            weights=support_weights,
            minlength=particle_filter.segment_count,
        )
        required = (
            max(1, int(particle_filter.config.min_segment_points))
            * float(particle_filter.config.ownership_visibility_threshold)
        )
        particle_filter._set_visibility(segment_support >= required)
        particle_filter.last_mean_ownership_responsibility = float(np.mean(responsibility))
        particle_filter.last_ownership_effective_point_count = responsibility_sum
        particle_filter.last_ownership_entropy = float(np.mean(entropy))


def posterior_coverage_penalty_torch(
    nearest_segments_t,
    responsibility_t,
    segment_count,
    penalty_m,
    min_fraction,
):
    if float(penalty_m) <= 0.0 or int(segment_count) <= 0:
        return torch.zeros(nearest_segments_t.shape[0], dtype=torch.float32, device=nearest_segments_t.device)
    coverage_t = torch.zeros(
        (nearest_segments_t.shape[0], int(segment_count)),
        dtype=torch.float32,
        device=nearest_segments_t.device,
    )
    coverage_t.scatter_add_(
        1,
        nearest_segments_t,
        responsibility_t[None, :].expand(nearest_segments_t.shape[0], -1),
    )
    required = torch.sum(responsibility_t) * float(np.clip(min_fraction, 0.0, 1.0))
    missing = torch.sum((coverage_t < required).to(torch.float32), dim=1)
    return missing * float(penalty_m) * float(penalty_m)


def posterior_coverage_penalty(nearest_segments, responsibility, segment_count, penalty_m, min_fraction):
    if float(penalty_m) <= 0.0 or int(segment_count) <= 0:
        return np.zeros(nearest_segments.shape[0], dtype=np.float64)
    output = np.zeros(nearest_segments.shape[0], dtype=np.float64)
    required = float(np.sum(responsibility)) * float(np.clip(min_fraction, 0.0, 1.0))
    for particle_index, indices in enumerate(nearest_segments):
        coverage = np.bincount(indices, weights=responsibility, minlength=int(segment_count))
        output[particle_index] = np.count_nonzero(coverage < required) * float(penalty_m) ** 2
    return output


def update_posterior_velocity(particle_filter, nodes, dt, measurement_used):
    nodes = np.asarray(nodes, dtype=np.float32)
    previous = valid_points(particle_filter.last_posterior_nodes)
    if measurement_used and particle_filter._velocity_enabled() and previous.shape == nodes.shape:
        observed_velocity = clip_vector_norms(
            (nodes - previous) / max(float(dt), 1e-3),
            float(particle_filter.config.max_node_speed_mps),
        )
        blend = float(np.clip(particle_filter.config.velocity_measurement_blend, 0.0, 1.0))
        if particle_filter.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(particle_filter.cuda_stream):
                particle_filter._ensure_velocity_array()
                observed_t = torch.as_tensor(observed_velocity, dtype=torch.float32, device=particle_filter.device)
                particle_filter.node_velocities.mul_(1.0 - blend).add_(observed_t[None, :, :] * blend)
                particle_filter.node_velocities = clip_vector_norms_torch(
                    particle_filter.node_velocities,
                    particle_filter.config.max_node_speed_mps,
                )
        else:
            particle_filter._ensure_velocity_array()
            particle_filter.node_velocities *= 1.0 - blend
            particle_filter.node_velocities += blend * observed_velocity[None, :, :]
            particle_filter.node_velocities = clip_vector_norms(
                particle_filter.node_velocities,
                particle_filter.config.max_node_speed_mps,
            )
    particle_filter.last_posterior_nodes = np.ascontiguousarray(nodes, dtype=np.float32)


def filtered_cable_estimate(measurement, result):
    if result is None:
        return None
    source_points = np.asarray(
        getattr(result, "support_points_xyz", np.empty((0, 3))),
        dtype=np.float32,
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


def project_equal_length_chain(nodes, segment_length_m, node_count=None):
    nodes = valid_points(nodes)
    if len(nodes) < 2:
        return None
    node_count = len(nodes) if node_count is None else max(2, int(node_count))
    nodes = fit_polyline_segments(nodes, segment_count=node_count - 1)
    directions = chain_directions(nodes)
    return build_chain(nodes[0], directions, float(segment_length_m))


def constrain_chain_to_endpoints(nodes, endpoint_nodes, segment_length_m, iterations=16, tolerance_m=1e-4):
    chain = valid_points(nodes)
    if len(chain) < 2:
        return None
    constrained = constrain_chains_to_endpoints(
        chain[None, :, :],
        endpoint_nodes,
        segment_length_m,
        iterations=iterations,
        tolerance_m=tolerance_m,
    )
    return np.ascontiguousarray(constrained[0], dtype=np.float32)


def constrain_chains_to_endpoints(chains, endpoint_nodes, segment_length_m, iterations=16, tolerance_m=1e-4):
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 3 or chains.shape[1] < 2 or chains.shape[2] < 3:
        return np.empty((0, 0, 3), dtype=np.float64)
    endpoints = valid_points(endpoint_nodes)
    if len(endpoints) < 2:
        return np.ascontiguousarray(chains[:, :, :3], dtype=np.float64)

    start = endpoints[0, :3]
    end = endpoints[-1, :3]
    segment_length = float(segment_length_m)
    if not np.isfinite(segment_length) or segment_length <= 0.0:
        raise ValueError("Endpoint-constrained cable model requires a positive fixed segment length.")

    segment_count = chains.shape[1] - 1
    total_length = segment_count * segment_length
    endpoint_distance = float(np.linalg.norm(end - start))
    tolerance = max(0.0, float(tolerance_m))
    if endpoint_distance > total_length + tolerance:
        raise ValueError(
            "Known cable endpoints are farther apart than the fixed cable length: "
            f"distance={endpoint_distance:.4f}m total_length={total_length:.4f}m"
        )

    output = np.ascontiguousarray(chains[:, :, :3], dtype=np.float64).copy()
    output[:, 0, :] = start
    output[:, -1, :] = end

    if endpoint_distance > total_length - tolerance:
        direction = normalize_vectors((end - start)[None, :])[0]
        arclength = np.arange(segment_count + 1, dtype=np.float64) * segment_length
        straight = start[None, :] + arclength[:, None] * direction[None, :]
        return np.repeat(straight[None, :, :], len(output), axis=0)

    for _ in range(max(1, int(iterations))):
        output[:, -1, :] = end
        for index in range(segment_count - 1, -1, -1):
            direction = normalize_vectors(output[:, index, :] - output[:, index + 1, :])
            output[:, index, :] = output[:, index + 1, :] + segment_length * direction

        output[:, 0, :] = start
        for index in range(segment_count):
            direction = normalize_vectors(output[:, index + 1, :] - output[:, index, :])
            output[:, index + 1, :] = output[:, index, :] + segment_length * direction

        residual = float(np.max(np.linalg.norm(output[:, -1, :] - end[None, :], axis=1)))
        if residual <= tolerance:
            break

    return np.ascontiguousarray(output, dtype=np.float64)


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


def normalize_vectors_torch(vectors_t):
    norms = torch.linalg.vector_norm(vectors_t, dim=-1, keepdim=True)
    normalized = vectors_t / norms.clamp_min(1e-12)
    fallback = torch.zeros_like(normalized)
    fallback[..., 0] = 1.0
    return torch.where(norms > 1e-12, normalized, fallback)


def smooth_particle_directions_torch(directions_t, passes=1):
    smoothed = normalize_vectors_torch(directions_t)
    for _ in range(max(0, int(passes))):
        if smoothed.shape[1] < 3:
            break
        updated = smoothed.clone()
        updated[:, 1:-1, :] = normalize_vectors_torch(
            smoothed[:, :-2, :] + 2.0 * smoothed[:, 1:-1, :] + smoothed[:, 2:, :]
        )
        smoothed = updated
    return smoothed


def build_chains_torch(starts_t, directions_t, segment_length_m):
    directions_t = normalize_vectors_torch(directions_t)
    offsets = torch.cumsum(directions_t * float(segment_length_m), dim=1)
    zero = torch.zeros((directions_t.shape[0], 1, 3), dtype=directions_t.dtype, device=directions_t.device)
    return starts_t[:, None, :] + torch.cat((zero, offsets), dim=1)


def clip_vector_norms_torch(vectors_t, max_norm):
    max_norm = max(0.0, float(max_norm))
    if max_norm <= 0.0:
        return torch.zeros_like(vectors_t)
    norms = torch.linalg.vector_norm(vectors_t, dim=-1, keepdim=True)
    return vectors_t * torch.clamp(max_norm / norms.clamp_min(1e-12), max=1.0)


def transition_component_count(total_count, ratio):
    total_count = max(0, int(total_count))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if total_count == 0 or ratio <= 0.0:
        return 0
    return int(np.clip(round(total_count * ratio), 1, total_count))


def transition_component_weights(conditioned_count, global_count, conditioned_mass):
    conditioned_count = max(0, int(conditioned_count))
    global_count = max(0, int(global_count))
    conditioned_mass = float(np.clip(conditioned_mass, 0.0, 1.0))
    values = []
    if conditioned_count:
        values.append(np.full(conditioned_count, conditioned_mass / conditioned_count, dtype=np.float64))
    if global_count:
        values.append(np.full(global_count, (1.0 - conditioned_mass) / global_count, dtype=np.float64))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def transition_component_weights_torch(conditioned_count, global_count, conditioned_mass, device):
    values = transition_component_weights(conditioned_count, global_count, conditioned_mass)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def transition_mixture_weights(
    local_count,
    conditioned_count,
    global_count,
    local_mass,
    conditioned_mass,
    global_mass,
):
    components = []
    for count, mass in (
        (local_count, local_mass),
        (conditioned_count, conditioned_mass),
        (global_count, global_mass),
    ):
        count = max(0, int(count))
        if count:
            components.append(np.full(count, max(0.0, float(mass)) / count, dtype=np.float64))
    weights = np.concatenate(components) if components else np.empty(0, dtype=np.float64)
    total = float(np.sum(weights))
    if total > 0.0:
        weights /= total
    return weights


def transition_mixture_weights_torch(
    local_count,
    conditioned_count,
    global_count,
    local_mass,
    conditioned_mass,
    global_mass,
    device,
):
    return torch.as_tensor(
        transition_mixture_weights(
            local_count,
            conditioned_count,
            global_count,
            local_mass,
            conditioned_mass,
            global_mass,
        ),
        dtype=torch.float32,
        device=device,
    )


def estimate_endpoint_tangents(
    points_xyz,
    endpoint_nodes,
    min_radius_m=0.008,
    radius_m=0.075,
    sigma_m=0.035,
    min_points=8,
):
    points = valid_points(points_xyz)
    endpoints = valid_points(endpoint_nodes)
    if len(endpoints) < 2:
        return (
            np.full((2, 3), np.nan, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.int32),
        )
    chord = normalize_vectors((endpoints[-1] - endpoints[0])[None, :])[0]
    fallback = np.stack((chord, -chord), axis=0)
    tangents = fallback.copy()
    confidence = np.zeros(2, dtype=np.float64)
    support_count = np.zeros(2, dtype=np.int32)
    min_radius = max(0.0, float(min_radius_m))
    radius = max(min_radius + 1e-6, float(radius_m))
    sigma = max(1e-6, float(sigma_m))
    for endpoint_index, endpoint in enumerate((endpoints[0], endpoints[-1])):
        delta = points - endpoint[None, :]
        distance = np.linalg.norm(delta, axis=1)
        keep = (distance >= min_radius) & (distance <= radius)
        count = int(np.count_nonzero(keep))
        support_count[endpoint_index] = count
        if count < 2:
            continue
        unit = delta[keep] / np.maximum(distance[keep, None], 1e-12)
        weights = np.exp(-0.5 * (distance[keep] / sigma) ** 2)
        scatter = np.einsum("n,ni,nj->ij", weights, unit, unit) / max(float(np.sum(weights)), 1e-12)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        tangent = eigenvectors[:, -1]
        mean_direction = np.sum(weights[:, None] * unit, axis=0)
        orientation = mean_direction if np.linalg.norm(mean_direction) > 1e-8 else fallback[endpoint_index]
        if float(np.dot(tangent, orientation)) < 0.0:
            tangent = -tangent
        anisotropy = max(0.0, float(eigenvalues[-1] - eigenvalues[-2])) / max(float(eigenvalues[-1]), 1e-12)
        support_factor = min(1.0, count / max(float(min_points), 1.0))
        tangents[endpoint_index] = tangent
        confidence[endpoint_index] = anisotropy * support_factor
    return (
        np.ascontiguousarray(tangents, dtype=np.float32),
        np.ascontiguousarray(confidence, dtype=np.float32),
        support_count,
    )


def estimate_endpoint_tangents_torch(
    points_t,
    endpoints_t,
    min_radius_m=0.008,
    radius_m=0.075,
    sigma_m=0.035,
    min_points=8,
):
    chord_t = normalize_vectors_torch((endpoints_t[-1] - endpoints_t[0])[None, :])[0]
    fallback_t = torch.stack((chord_t, -chord_t), dim=0)
    tangents = []
    confidence = []
    support_counts = []
    min_radius = max(0.0, float(min_radius_m))
    radius = max(min_radius + 1e-6, float(radius_m))
    sigma = max(1e-6, float(sigma_m))
    for endpoint_index in range(2):
        delta_t = points_t - endpoints_t[endpoint_index][None, :]
        distance_t = torch.linalg.vector_norm(delta_t, dim=1)
        keep_t = (distance_t >= min_radius) & (distance_t <= radius)
        support_count_t = torch.count_nonzero(keep_t)
        support_counts.append(support_count_t)
        unit_t = delta_t / distance_t[:, None].clamp_min(1e-12)
        weights_t = torch.exp(-0.5 * (distance_t / sigma) ** 2) * keep_t.to(points_t.dtype)
        scatter_t = torch.einsum("n,ni,nj->ij", weights_t, unit_t, unit_t) / weights_t.sum().clamp_min(1e-12)
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(scatter_t)
        tangent_t = eigenvectors_t[:, -1]
        mean_direction_t = torch.sum(weights_t[:, None] * unit_t, dim=0)
        orientation_t = torch.where(
            torch.linalg.vector_norm(mean_direction_t) > 1e-8,
            mean_direction_t,
            fallback_t[endpoint_index],
        )
        tangent_t = torch.where(
            torch.dot(tangent_t, orientation_t) < 0.0,
            -tangent_t,
            tangent_t,
        )
        anisotropy_t = torch.clamp_min(eigenvalues_t[-1] - eigenvalues_t[-2], 0.0) / eigenvalues_t[-1].clamp_min(1e-12)
        support_factor_t = torch.clamp(support_count_t.to(points_t.dtype) / max(float(min_points), 1.0), max=1.0)
        valid_t = support_count_t >= 2
        tangents.append(torch.where(valid_t, tangent_t, fallback_t[endpoint_index]))
        confidence.append(torch.where(valid_t, anisotropy_t * support_factor_t, torch.zeros_like(anisotropy_t)))
    return (
        torch.stack(tangents, dim=0),
        torch.stack(confidence, dim=0),
        torch.stack(support_counts, dim=0).to(torch.int32),
    )


def endpoint_tangent_bridge_nodes(endpoint_nodes, endpoint_tangents, node_count, segment_length_m):
    endpoints = valid_points(endpoint_nodes)
    if len(endpoints) < 2 or int(node_count) < 2:
        return None
    chord = endpoints[-1] - endpoints[0]
    chord_distance = float(np.linalg.norm(chord))
    total_length = float(segment_length_m) * (int(node_count) - 1)
    if chord_distance > total_length + 1e-6:
        return None
    chord_direction = normalize_vectors(chord[None, :])[0]
    tangents = np.asarray(endpoint_tangents, dtype=np.float64)
    if tangents.shape != (2, 3) or not np.all(np.isfinite(tangents)):
        tangents = np.stack((chord_direction, -chord_direction), axis=0)
    tangents = normalize_vectors(tangents)
    u = np.linspace(0.0, 1.0, int(node_count), dtype=np.float64)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    tangent_scale = min(total_length / 3.0, max(chord_distance / 3.0, 2.0 * float(segment_length_m)))
    chain_end_direction = -tangents[1]
    nodes = (
        h00[:, None] * endpoints[0][None, :]
        + h10[:, None] * tangent_scale * tangents[0][None, :]
        + h01[:, None] * endpoints[-1][None, :]
        + h11[:, None] * tangent_scale * chain_end_direction[None, :]
    )
    nodes = project_equal_length_chain(nodes, segment_length_m, node_count=int(node_count))
    return constrain_chain_to_endpoints(nodes, endpoints, segment_length_m, iterations=128, tolerance_m=1e-4)


def endpoint_conditioned_chains_torch(
    base_particles_t,
    endpoints_t,
    tangents_t,
    confidence_t,
    segment_length_m,
    config,
    generator,
    stream,
    initialization=False,
    slack_m=None,
):
    count, node_count, _ = base_particles_t.shape
    if count == 0:
        return base_particles_t
    modes = max(1, int(config.endpoint_conditioned_deformation_modes))
    u_t = torch.linspace(0.0, 1.0, node_count, dtype=torch.float32, device=base_particles_t.device)
    mode_index_t = torch.arange(1, modes + 1, dtype=torch.float32, device=base_particles_t.device)
    basis_t = torch.sin(np.pi * mode_index_t[:, None] * u_t[None, :]) / torch.sqrt(mode_index_t[:, None])
    if slack_m is None:
        total_length = float(segment_length_m) * (node_count - 1)
        chord_distance = float(torch.linalg.vector_norm(endpoints_t[-1] - endpoints_t[0]).item())
        slack_m = max(0.0, total_length - chord_distance)
    deformation_std = float(config.endpoint_conditioned_deformation_std_m) + 0.35 * max(0.0, float(slack_m))
    if initialization:
        deformation_std *= 1.25
    coefficients_t = torch.randn(
        (count, modes, 3),
        dtype=torch.float32,
        device=base_particles_t.device,
        generator=generator,
    ) * deformation_std
    offsets_t = torch.einsum("cmx,mn->cnx", coefficients_t, basis_t)
    candidate_t = base_particles_t + offsets_t
    candidate_t[:, 0, :] = endpoints_t[0]
    candidate_t[:, -1, :] = endpoints_t[-1]
    directions_t = normalize_vectors_torch(candidate_t[:, 1:, :] - candidate_t[:, :-1, :])
    directions_t = normalize_vectors_torch(
        directions_t
        + torch.randn(
            directions_t.shape,
            dtype=torch.float32,
            device=base_particles_t.device,
            generator=generator,
        ) * float(config.endpoint_conditioned_direction_std)
    )
    confidence_gate_t = torch.clamp(
        (confidence_t - float(config.endpoint_tangent_min_confidence))
        / max(1.0 - float(config.endpoint_tangent_min_confidence), 1e-6),
        min=0.0,
        max=1.0,
    )
    tangent_noise_t = torch.randn(
        (count, 2, 3),
        dtype=torch.float32,
        device=base_particles_t.device,
        generator=generator,
    ) * float(config.endpoint_conditioned_direction_std)
    start_t = normalize_vectors_torch(tangents_t[0][None, :] + tangent_noise_t[:, 0, :])
    end_t = normalize_vectors_torch(-tangents_t[1][None, :] + tangent_noise_t[:, 1, :])
    directions_t[:, 0, :] = normalize_vectors_torch(
        (1.0 - confidence_gate_t[0]) * directions_t[:, 0, :] + confidence_gate_t[0] * start_t
    )
    directions_t[:, -1, :] = normalize_vectors_torch(
        (1.0 - confidence_gate_t[1]) * directions_t[:, -1, :] + confidence_gate_t[1] * end_t
    )
    chains_t = build_chains_torch(
        endpoints_t[0][None, :].expand(count, -1),
        directions_t,
        segment_length_m,
    )
    return constrain_chains_cuda(
        chains_t,
        endpoints_t,
        segment_length_m,
        int(config.endpoint_constraint_iterations),
        float(config.endpoint_constraint_tolerance_m),
        stream,
    )


def endpoint_conditioned_chains(
    base_particles,
    endpoints,
    tangents,
    confidence,
    segment_length_m,
    config,
    rng,
    initialization=False,
):
    base = np.asarray(base_particles, dtype=np.float64)
    count, node_count, _ = base.shape
    if count == 0:
        return base
    modes = max(1, int(config.endpoint_conditioned_deformation_modes))
    u = np.linspace(0.0, 1.0, node_count, dtype=np.float64)
    mode_index = np.arange(1, modes + 1, dtype=np.float64)
    basis = np.sin(np.pi * mode_index[:, None] * u[None, :]) / np.sqrt(mode_index[:, None])
    total_length = float(segment_length_m) * (node_count - 1)
    chord_distance = float(np.linalg.norm(np.asarray(endpoints)[-1] - np.asarray(endpoints)[0]))
    deformation_std = float(config.endpoint_conditioned_deformation_std_m) + 0.35 * max(0.0, total_length - chord_distance)
    if initialization:
        deformation_std *= 1.25
    coefficients = rng.normal(0.0, deformation_std, (count, modes, 3))
    candidate = base + np.einsum("cmx,mn->cnx", coefficients, basis)
    candidate[:, 0, :] = endpoints[0]
    candidate[:, -1, :] = endpoints[-1]
    directions = normalize_vectors(
        candidate[:, 1:, :] - candidate[:, :-1, :]
        + rng.normal(0.0, float(config.endpoint_conditioned_direction_std), (count, node_count - 1, 3))
    )
    gate = np.clip(
        (np.asarray(confidence) - float(config.endpoint_tangent_min_confidence))
        / max(1.0 - float(config.endpoint_tangent_min_confidence), 1e-6),
        0.0,
        1.0,
    )
    tangent_noise = rng.normal(0.0, float(config.endpoint_conditioned_direction_std), (count, 2, 3))
    start = normalize_vectors(np.asarray(tangents)[0][None, :] + tangent_noise[:, 0, :])
    end = normalize_vectors(-np.asarray(tangents)[1][None, :] + tangent_noise[:, 1, :])
    directions[:, 0, :] = normalize_vectors((1.0 - gate[0]) * directions[:, 0, :] + gate[0] * start)
    directions[:, -1, :] = normalize_vectors((1.0 - gate[1]) * directions[:, -1, :] + gate[1] * end)
    chains = build_chains(
        np.repeat(np.asarray(endpoints[0])[None, :], count, axis=0),
        directions,
        segment_length_m,
    )
    return constrain_chains_to_endpoints(
        chains,
        endpoints,
        segment_length_m,
        iterations=int(config.endpoint_constraint_iterations),
        tolerance_m=float(config.endpoint_constraint_tolerance_m),
    )


def global_endpoint_chains_torch(
    count,
    node_count,
    endpoints_t,
    segment_length_m,
    config,
    generator,
    stream,
):
    count = max(0, int(count))
    if count == 0:
        return torch.empty((0, int(node_count), 3), dtype=torch.float32, device=endpoints_t.device)
    directions_t = normalize_vectors_torch(torch.randn(
        (count, int(node_count) - 1, 3),
        dtype=torch.float32,
        device=endpoints_t.device,
        generator=generator,
    ))
    chains_t = build_chains_torch(
        endpoints_t[0][None, :].expand(count, -1),
        directions_t,
        segment_length_m,
    )
    return constrain_chains_cuda(
        chains_t,
        endpoints_t,
        segment_length_m,
        int(config.endpoint_constraint_iterations),
        float(config.endpoint_constraint_tolerance_m),
        stream,
    )


def global_endpoint_chains(count, node_count, endpoints, segment_length_m, config, rng):
    count = max(0, int(count))
    if count == 0:
        return np.empty((0, int(node_count), 3), dtype=np.float64)
    directions = normalize_vectors(rng.normal(0.0, 1.0, (count, int(node_count) - 1, 3)))
    chains = build_chains(
        np.repeat(np.asarray(endpoints[0])[None, :], count, axis=0),
        directions,
        segment_length_m,
    )
    return constrain_chains_to_endpoints(
        chains,
        endpoints,
        segment_length_m,
        iterations=int(config.endpoint_constraint_iterations),
        tolerance_m=float(config.endpoint_constraint_tolerance_m),
    )


def particle_estimate_diagnostics(
    average_nodes,
    map_nodes,
    top_particles,
    *,
    endpoint_tangents=None,
    endpoint_tangent_confidence=None,
    endpoint_tangent_support_count=None,
    mean_ownership_responsibility=np.nan,
    ownership_entropy=np.nan,
    visible_segment_fraction=np.nan,
    endpoint_conditioned_proposal_ratio=0.0,
):
    """Describe estimator spread without changing the particle-filter state."""
    average = np.asarray(average_nodes, dtype=np.float32)
    map_nodes = np.asarray(map_nodes, dtype=np.float32)
    particles = np.asarray(top_particles, dtype=np.float32)
    if average.ndim != 2 or average.shape[1] < 3:
        return None
    average = np.ascontiguousarray(average[:, :3], dtype=np.float32)
    if map_nodes.shape != average.shape or particles.ndim != 3 or particles.shape[1:] != average.shape:
        return None
    finite_particles = np.all(np.isfinite(particles), axis=(1, 2))
    particles = np.ascontiguousarray(particles[finite_particles], dtype=np.float32)
    if len(particles) == 0 or not np.all(np.isfinite(average)) or not np.all(np.isfinite(map_nodes)):
        return None

    particle_mean = np.mean(particles, axis=0, dtype=np.float64)
    deltas = particles.astype(np.float64) - particle_mean[None, :, :]
    squared_radius = np.sum(deltas * deltas, axis=2)
    node_rms = np.sqrt(np.mean(squared_radius, axis=0))

    covariance = np.einsum("knc,knd->ncd", deltas, deltas, optimize=True) / max(len(particles), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_sigma = np.sqrt(np.maximum(eigenvalues[:, -1], 0.0))
    principal_std = eigenvectors[:, :, -1] * principal_sigma[:, None]

    map_error = float(np.mean(np.linalg.norm(map_nodes.astype(np.float64) - average, axis=1)))
    endpoint_delta = endpoint_direction_disagreement_deg(map_nodes, average)
    tangents = np.asarray(
        np.full((2, 3), np.nan, dtype=np.float32) if endpoint_tangents is None else endpoint_tangents,
        dtype=np.float32,
    )
    if tangents.shape != (2, 3):
        tangents = np.full((2, 3), np.nan, dtype=np.float32)
    tangent_confidence = np.asarray(
        np.zeros(2, dtype=np.float32) if endpoint_tangent_confidence is None else endpoint_tangent_confidence,
        dtype=np.float32,
    ).reshape(-1)
    if len(tangent_confidence) != 2:
        tangent_confidence = np.zeros(2, dtype=np.float32)
    tangent_support = np.asarray(
        np.zeros(2, dtype=np.int32) if endpoint_tangent_support_count is None else endpoint_tangent_support_count,
        dtype=np.int32,
    ).reshape(-1)
    if len(tangent_support) != 2:
        tangent_support = np.zeros(2, dtype=np.int32)
    return ParticleEstimateDiagnostics(
        average_points_xyz=average.copy(),
        map_points_xyz=np.ascontiguousarray(map_nodes, dtype=np.float32),
        top_particle_points_xyz=particles,
        node_rms_spread_m=np.ascontiguousarray(node_rms, dtype=np.float32),
        node_principal_std_xyz=np.ascontiguousarray(principal_std, dtype=np.float32),
        map_to_average_node_error_m=map_error,
        mean_node_spread_m=float(np.mean(node_rms)),
        max_node_spread_m=float(np.max(node_rms)),
        endpoint_direction_delta_deg=endpoint_delta,
        endpoint_tangents_xyz=np.ascontiguousarray(tangents, dtype=np.float32),
        endpoint_tangent_confidence=np.ascontiguousarray(tangent_confidence, dtype=np.float32),
        endpoint_tangent_support_count=np.ascontiguousarray(tangent_support, dtype=np.int32),
        mean_ownership_responsibility=float(mean_ownership_responsibility),
        ownership_entropy=float(ownership_entropy),
        visible_segment_fraction=float(visible_segment_fraction),
        endpoint_conditioned_proposal_ratio=float(endpoint_conditioned_proposal_ratio),
    )


def endpoint_direction_disagreement_deg(reference_nodes, candidate_nodes):
    reference = np.asarray(reference_nodes, dtype=np.float64)
    candidate = np.asarray(candidate_nodes, dtype=np.float64)
    if reference.ndim != 2 or candidate.shape != reference.shape or reference.shape[0] < 2 or reference.shape[1] < 3:
        return np.full(2, np.nan, dtype=np.float32)
    reference_directions = np.stack((
        reference[1, :3] - reference[0, :3],
        reference[-2, :3] - reference[-1, :3],
    ))
    candidate_directions = np.stack((
        candidate[1, :3] - candidate[0, :3],
        candidate[-2, :3] - candidate[-1, :3],
    ))
    reference_norm = np.linalg.norm(reference_directions, axis=1)
    candidate_norm = np.linalg.norm(candidate_directions, axis=1)
    valid = (reference_norm > 1e-12) & (candidate_norm > 1e-12)
    output = np.full(2, np.nan, dtype=np.float64)
    if np.any(valid):
        dots = np.sum(reference_directions[valid] * candidate_directions[valid], axis=1)
        dots /= reference_norm[valid] * candidate_norm[valid]
        output[valid] = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return np.ascontiguousarray(output, dtype=np.float32)


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


def torch_scoring_device(backend):
    backend = str(backend or "auto").lower()
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown particle scoring backend: {backend!r}")
    if backend == "cpu":
        return None
    if torch is None:
        if backend == "cuda":
            raise RuntimeError("CUDA particle scoring requires PyTorch.")
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if backend == "cuda":
        raise RuntimeError("CUDA particle scoring was requested, but torch.cuda.is_available() is false.")
    return None


def particle_bend_penalty(particles, penalty_m=0.030):
    particles = valid_particles(particles)
    if len(particles) == 0 or particles.shape[1] < 3:
        return np.zeros(len(particles), dtype=np.float64)
    directions = particle_directions(particles)
    dots = np.sum(directions[:, :-1, :] * directions[:, 1:, :], axis=2)
    bend = np.mean(np.maximum(0.0, 1.0 - np.clip(dots, -1.0, 1.0)), axis=1)
    return bend * float(penalty_m) * float(penalty_m)


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


def valid_particles(particles):
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 3 or particles.shape[1] < 2 or particles.shape[2] < 3:
        return np.empty((0, 0, 3), dtype=np.float64)
    return np.ascontiguousarray(particles[:, :, :3], dtype=np.float64)


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


def clip_vector_norms(vectors, max_norm):
    vectors = np.asarray(vectors, dtype=np.float64)
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        return np.ascontiguousarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.ones_like(norms)
    too_fast = norms[..., 0] > max_norm
    scale[too_fast] = max_norm / np.maximum(norms[too_fast], 1e-12)
    return np.ascontiguousarray(vectors * scale, dtype=np.float64)


def weighted_mean_or_zero(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or len(values) == 0:
        return 0.0
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 1e-12:
        return 0.0
    return float(np.sum(values * weights) / total)
