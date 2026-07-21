from dataclasses import dataclass
from itertools import combinations
import time

import numpy as np

from cable_crossing import particle_crossing_rewards, particle_crossing_rewards_torch
from cable_cuda import constrain_chains as constrain_chains_cuda
from cable_cuda import particle_point_distances as particle_point_distances_cuda
from cable_cuda import particle_support_distances as particle_support_distances_cuda
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


def normalize_particle_weights_torch(weights_t):
    """Return a finite, non-negative, unit-sum CUDA/torch posterior.

    This is a numerical invariant, not a tracking recovery mechanism.  The
    uniform branch is used only when the supplied vector has no finite positive
    mass, which keeps downstream multinomial sampling and ESS well-defined.
    """

    finite_positive_t = torch.where(
        torch.isfinite(weights_t) & (weights_t > 0.0),
        weights_t,
        torch.zeros_like(weights_t),
    )
    total_t = finite_positive_t.sum()
    uniform_t = torch.full_like(
        finite_positive_t,
        1.0 / max(1, int(finite_positive_t.numel())),
    )
    return torch.where(
        torch.isfinite(total_t) & (total_t > 1e-20),
        finite_positive_t / total_t.clamp_min(1e-20),
        uniform_t,
    ).contiguous()


def normalize_particle_weights(weights):
    """NumPy equivalent of :func:`normalize_particle_weights_torch`."""

    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite_positive = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(np.sum(finite_positive))
    if np.isfinite(total) and total > 1e-20:
        return np.ascontiguousarray(finite_positive / total, dtype=np.float64)
    return np.full(len(values), 1.0 / max(1, len(values)), dtype=np.float64)


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
    adaptive_motion_noise_enabled: bool = True
    motion_speed_reference_mps: float = 0.25
    motion_innovation_reference_m: float = 0.010
    motion_noise_adaptation: float = 0.35
    direction_smooth_passes: int = 1
    measurement_node_std_m: float = 0.030
    measurement_max_points: int = 1024
    scoring_backend: str = "auto"
    endpoint_tangent_min_radius_m: float = 0.008
    endpoint_tangent_radius_m: float = 0.075
    endpoint_tangent_sigma_m: float = 0.035
    endpoint_tangent_min_points: int = 8
    endpoint_tangent_min_confidence: float = 0.20
    endpoint_tangent_ransac_inlier_m: float = 0.006
    endpoint_tangent_max_hypotheses: int = 64
    endpoint_tangent_reference_weight: float = 0.25
    endpoint_tangent_estimation_enabled: bool = True
    endpoint_tangent_ransac_enabled: bool = True
    endpoint_tangent_likelihood_scale_m: float = 0.020
    local_proposal_ratio: float = 0.65
    endpoint_conditioned_proposal_ratio: float = 0.25
    endpoint_conditioned_direction_std: float = 0.08
    endpoint_conditioned_deformation_std_m: float = 0.012
    endpoint_conditioned_slack_gain: float = 0.10
    endpoint_conditioned_max_deformation_m: float = 0.040
    endpoint_conditioned_deformation_modes: int = 3
    robust_measurement_enabled: bool = True
    robust_distance_m: float = 0.05
    path_support_weight: float = 1.0
    path_support_samples_per_segment: int = 5
    bend_penalty_m: float = 0.030
    global_random_particle_ratio: float = 0.10
    endpoint_constraint_iterations: int = 128
    endpoint_constraint_tolerance_m: float = 1e-4
    min_measurement_points: int = 12
    min_segment_support_samples: int = 3
    support_visibility_distance_m: float = 0.030
    crossing_position_sigma_px: float = 14.0
    crossing_angle_sigma_deg: float = 20.0
    crossing_log_reward: float = 4.0
    union_coverage_top_particle_count: int = 30
    union_coverage_weight: float = 1.0
    max_prediction_frames: int = 12
    max_motion_noise_scale: float = 4.0
    point_support_diagnostics_enabled: bool = True
    particle_diagnostics_enabled: bool = True


@dataclass
class ParticleEstimateDiagnostics:
    representative_points_xyz: np.ndarray
    map_points_xyz: np.ndarray
    top_particle_points_xyz: np.ndarray
    node_rms_spread_m: np.ndarray
    node_principal_std_xyz: np.ndarray
    map_to_representative_node_error_m: float
    mean_node_spread_m: float
    max_node_spread_m: float
    endpoint_direction_delta_deg: np.ndarray
    endpoint_tangents_xyz: np.ndarray
    endpoint_tangent_confidence: np.ndarray
    endpoint_tangent_support_count: np.ndarray
    mean_support_affinity: float
    supported_sample_fraction: float
    visible_segment_fraction: float
    local_proposal_ratio: float
    endpoint_conditioned_proposal_ratio: float
    global_random_particle_ratio: float


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
    support_point_affinities: np.ndarray | None = None
    visible_segments: np.ndarray | None = None
    visible_nodes: np.ndarray | None = None
    local_proposal_ratio: float = 0.0
    endpoint_conditioned_proposal_ratio: float = 0.0
    global_random_particle_ratio: float = 0.0
    path_support_rms_m: float = np.nan
    mean_support_affinity: float = np.nan
    supported_sample_fraction: float = np.nan
    endpoint_tangent_confidence: np.ndarray | None = None
    endpoint_tangent_support_count: np.ndarray | None = None
    estimate_particle_count: int = 1
    estimate_weight_mass: float = 1.0
    mean_node_speed_mps: float = 0.0
    endpoint_speed_mps: float = 0.0
    endpoint_motion_innovation_m: float = 0.0
    crossing_target_count: int = 0
    crossing_reward: float = 0.0
    crossing_distance_px: float = np.nan
    crossing_angle_error_deg: float = np.nan
    crossing_closest_xy: np.ndarray | None = None
    crossing_target_xy: np.ndarray | None = None
    crossing_axis_xy: np.ndarray | None = None
    union_coverage_selected: bool = False
    union_coverage_rank: int = 0
    union_coverage_rms_m: float = np.nan
    union_coverage_fraction: float = np.nan
    union_coverage_rms_gain_m: float = np.nan
    union_coverage_fraction_gain: float = np.nan
    stage_seconds: dict | None = None
    particle_diagnostics: ParticleEstimateDiagnostics | None = None


@dataclass(frozen=True)
class UnionCoverageSelection:
    selected_particle_indices: tuple
    selected_ranks: tuple
    robust_coverage_rms_m: float
    covered_point_fraction: float
    robust_rms_gain_m: float
    covered_fraction_gain: float
    observation_point_count: int
    stage_seconds: float
    cuda_events: tuple | None = None


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
        self.last_endpoint_speed_mps = 0.0
        self.last_endpoint_motion_innovation_m = 0.0
        self.last_endpoint_velocity_mps = np.zeros((2, 3), dtype=np.float32)
        self.last_visible_segments = np.zeros(self.segment_count, dtype=bool)
        self.last_visible_nodes = np.zeros(self.node_count, dtype=bool)
        self.last_measurement_point_count = 0
        self.last_support_points = np.empty((0, 3), dtype=np.float32)
        self.last_support_point_affinities = np.empty(0, dtype=np.float32)
        self._pending_support_diagnostics = None
        self.last_local_proposal_ratio = 0.0
        self.last_endpoint_conditioned_proposal_ratio = 0.0
        self.last_global_random_particle_ratio = 0.0
        self.last_endpoint_tangents = np.full((2, 3), np.nan, dtype=np.float32)
        self.last_endpoint_tangent_confidence = np.zeros(2, dtype=np.float32)
        self.last_endpoint_tangent_support_count = np.zeros(2, dtype=np.int32)
        self.last_endpoint_tangents_t = None
        self.last_endpoint_tangent_confidence_t = None
        self.last_endpoint_tangent_support_count_t = None
        self.last_mean_support_affinity = np.nan
        self.last_supported_sample_fraction = np.nan
        self.last_path_support_rms_m = np.nan
        self.last_crossing_target_count = 0
        self.last_crossing_reward = 0.0
        self.last_crossing_distance_px = np.nan
        self.last_crossing_angle_error_deg = np.nan
        self.last_crossing_closest_xy = np.full(2, np.nan, dtype=np.float32)
        self.last_crossing_target_xy = np.full(2, np.nan, dtype=np.float32)
        self.last_crossing_axis_xy = np.full(2, np.nan, dtype=np.float32)
        self._pending_crossing_diagnostics = None
        self.last_union_coverage_selected = False
        self.last_union_coverage_rank = 0
        self.last_union_coverage_rms_m = np.nan
        self.last_union_coverage_fraction = np.nan
        self.last_union_coverage_rms_gain_m = np.nan
        self.last_union_coverage_fraction_gain = np.nan
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
        self.last_representative_particle_index = None

    def _invalidate_estimate_cache(self):
        self._estimate_nodes_cache = None
        self._estimate_particle_count_cache = 0
        self._estimate_weight_mass_cache = np.nan
        self._particle_estimate_diagnostics_cache = None
        self.last_representative_particle_index = None

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
        self.last_support_point_affinities = np.empty(0, dtype=np.float32)
        self._pending_support_diagnostics = None
        self.last_local_proposal_ratio = 0.0
        self.last_endpoint_conditioned_proposal_ratio = 0.0
        self.last_global_random_particle_ratio = 0.0
        self.last_mean_support_affinity = np.nan
        self.last_supported_sample_fraction = np.nan
        self.last_path_support_rms_m = np.nan
        self.last_crossing_target_count = 0
        self.last_crossing_reward = 0.0
        self.last_crossing_distance_px = np.nan
        self.last_crossing_angle_error_deg = np.nan
        self.last_crossing_closest_xy.fill(np.nan)
        self.last_crossing_target_xy.fill(np.nan)
        self.last_crossing_axis_xy.fill(np.nan)
        self._pending_crossing_diagnostics = None
        self.last_union_coverage_selected = False
        self.last_union_coverage_rank = 0
        self.last_union_coverage_rms_m = np.nan
        self.last_union_coverage_fraction = np.nan
        self.last_union_coverage_rms_gain_m = np.nan
        self.last_union_coverage_fraction_gain = np.nan
        prepare_start = time.perf_counter()
        dt = float(np.clip(dt, 1e-3, 0.20))
        measurement_points = self._measurement_points(measurement)
        endpoint_nodes = self._measurement_endpoint_nodes(measurement)
        if endpoint_nodes is not None:
            self._update_endpoint_motion_model(endpoint_nodes, dt)
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
        _, _, global_ratio = transition_proposal_ratios(self.config)
        global_count = transition_component_count(count, global_ratio)
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
        self.last_local_proposal_ratio = 0.0
        self.last_endpoint_conditioned_proposal_ratio = float(conditioned_count) / float(count)
        self.last_global_random_particle_ratio = float(global_count) / float(count)
        self.measurement_update_count = 1

    def _estimate_endpoint_tangents(self, measurement_points, endpoint_nodes):
        if not bool(self.config.endpoint_tangent_estimation_enabled):
            chord = normalize_vectors(
                (np.asarray(endpoint_nodes[-1]) - np.asarray(endpoint_nodes[0]))[None, :]
            )[0]
            self.last_endpoint_tangents = np.ascontiguousarray(
                np.stack((chord, -chord), axis=0),
                dtype=np.float32,
            )
            self.last_endpoint_tangent_confidence = np.zeros(2, dtype=np.float32)
            self.last_endpoint_tangent_support_count = np.zeros(2, dtype=np.int32)
            if self.cuda_state:
                # Keep this ablation path on the PF stream as well.  Creating
                # these tensors on the default stream and immediately reading
                # them from ``self.cuda_stream`` is an unsynchronized handoff.
                with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                    self.last_endpoint_tangents_t = torch.as_tensor(
                        self.last_endpoint_tangents,
                        dtype=torch.float32,
                        device=self.device,
                    ).contiguous()
                    self.last_endpoint_tangent_confidence_t = torch.zeros(
                        2,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.last_endpoint_tangent_support_count_t = torch.zeros(
                        2,
                        dtype=torch.int32,
                        device=self.device,
                    )
            return
        reference_tangents = posterior_endpoint_tangents(self.last_posterior_nodes)
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
                    ransac_inlier_m=float(self.config.endpoint_tangent_ransac_inlier_m),
                    max_hypotheses=int(self.config.endpoint_tangent_max_hypotheses),
                    ransac_enabled=bool(self.config.endpoint_tangent_ransac_enabled),
                    reference_tangents=reference_tangents,
                    reference_weight=float(self.config.endpoint_tangent_reference_weight),
                )
                self.last_endpoint_tangents_t = tangents_t.contiguous()
                self.last_endpoint_tangent_confidence_t = confidence_t.contiguous()
                self.last_endpoint_tangent_support_count_t = support_count_t.contiguous()
                # Initialization needs the tangent immediately to build its
                # first bridge. On subsequent frames, keep prediction fully
                # asynchronous and copy UI diagnostics only after all PF work
                # has been enqueued on the independent streams.
                if not self.initialized:
                    self._sync_endpoint_tangent_diagnostics()
            return
        tangents, confidence, support_count = estimate_endpoint_tangents(
            measurement_points,
            endpoint_nodes,
            min_radius_m=float(self.config.endpoint_tangent_min_radius_m),
            radius_m=float(self.config.endpoint_tangent_radius_m),
            sigma_m=float(self.config.endpoint_tangent_sigma_m),
            min_points=int(self.config.endpoint_tangent_min_points),
            ransac_inlier_m=float(self.config.endpoint_tangent_ransac_inlier_m),
            max_hypotheses=int(self.config.endpoint_tangent_max_hypotheses),
            ransac_enabled=bool(self.config.endpoint_tangent_ransac_enabled),
            reference_tangents=reference_tangents,
            reference_weight=float(self.config.endpoint_tangent_reference_weight),
        )
        self.last_endpoint_tangents = tangents
        self.last_endpoint_tangent_confidence = confidence
        self.last_endpoint_tangent_support_count = support_count

    def _sync_endpoint_tangent_diagnostics(self):
        if not self.cuda_state or self.last_endpoint_tangents_t is None:
            return
        with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
            packed_t = torch.cat((
                self.last_endpoint_tangents_t.reshape(-1),
                self.last_endpoint_tangent_confidence_t.reshape(-1),
                self.last_endpoint_tangent_support_count_t.to(torch.float32).reshape(-1),
            ))
            packed = packed_t.cpu().numpy().astype(np.float32, copy=False)
        self.last_endpoint_tangents = np.ascontiguousarray(packed[:6].reshape(2, 3))
        self.last_endpoint_tangent_confidence = np.ascontiguousarray(packed[6:8])
        self.last_endpoint_tangent_support_count = np.ascontiguousarray(
            np.rint(packed[8:10]).astype(np.int32)
        )

    def _sync_support_diagnostics(self):
        pending = self._pending_support_diagnostics
        self._pending_support_diagnostics = None
        if pending is None:
            return
        support_squared, point_affinities = pending
        if self.cuda_state:
            packed_t = torch.cat((
                support_squared.reshape(-1),
                point_affinities.reshape(-1),
            ))
            packed = packed_t.cpu().numpy().astype(np.float32, copy=False)
            support_count = int(support_squared.numel())
            support_squared = packed[:support_count].reshape(
                self.segment_count,
                -1,
            )
            point_affinities = packed[support_count:]
        _apply_support_diagnostics(
            self,
            support_squared,
            point_affinities,
        )

    def _sync_crossing_diagnostics(self):
        pending = self._pending_crossing_diagnostics
        self._pending_crossing_diagnostics = None
        if pending is None or self.last_representative_particle_index is None:
            return
        reward, distance, angle, closest, target_index, targets = pending
        particle_index = int(self.last_representative_particle_index)
        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                packed_t = torch.cat((
                    reward[particle_index:particle_index + 1],
                    distance[particle_index:particle_index + 1],
                    angle[particle_index:particle_index + 1],
                    closest[particle_index].reshape(-1),
                    target_index[particle_index:particle_index + 1].to(torch.float32),
                ))
                packed = packed_t.cpu().numpy().astype(np.float32, copy=False)
        else:
            packed = np.asarray((
                reward[particle_index],
                distance[particle_index],
                angle[particle_index],
                closest[particle_index, 0],
                closest[particle_index, 1],
                target_index[particle_index],
            ), dtype=np.float32)
        self.last_crossing_reward = float(packed[0])
        self.last_crossing_distance_px = float(packed[1])
        self.last_crossing_angle_error_deg = float(packed[2])
        self.last_crossing_closest_xy = np.ascontiguousarray(packed[3:5], dtype=np.float32)
        selected_target = int(round(float(packed[5])))
        if 0 <= selected_target < len(targets):
            target = targets[selected_target]
            self.last_crossing_target_xy = np.ascontiguousarray(target.centroid_xy, dtype=np.float32)
            self.last_crossing_axis_xy = np.ascontiguousarray(target.axis_xy, dtype=np.float32)

    def _predict_transition(self, dt):
        if self.particles is None or self.weights is None or self.segment_length_m is None:
            return
        total_count = len(self.particles)
        local_ratio, conditioned_ratio, global_ratio = transition_proposal_ratios(self.config)
        local_count, conditioned_count, global_count = transition_population_counts(
            total_count,
            (local_ratio, conditioned_ratio, global_ratio),
        )
        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                probabilities = torch.clamp_min(self.weights, 0.0)
                probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
                parent_indices_t = torch.multinomial(
                    probabilities,
                    local_count + conditioned_count,
                    replacement=True,
                    generator=self.torch_generator,
                )
                self._ensure_velocity_array()
                local_particles = self.particles.index_select(0, parent_indices_t[:local_count]).contiguous()
                local_velocities = self.node_velocities.index_select(0, parent_indices_t[:local_count]).contiguous()
                if self._velocity_enabled():
                    damping = float(np.clip(self.config.velocity_damping, 0.0, 1.0))
                    local_velocities.mul_(damping).add_(
                        torch.randn(
                            local_velocities.shape,
                            dtype=torch.float32,
                            device=self.device,
                            generator=self.torch_generator,
                        ) * (
                            float(self.config.velocity_process_std_mps)
                            * float(self.last_motion_noise_scale)
                        )
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
                        ) * (
                            float(self.config.process_direction_std)
                            * float(self.last_motion_noise_scale)
                        )
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
                    parent_indices_t[local_count:local_count + conditioned_count],
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
                    noise_scale=self.last_motion_noise_scale,
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
                    local_ratio,
                    conditioned_ratio,
                    global_ratio,
                    self.device,
                )
                permutation_t = torch.randperm(
                    total_count,
                    device=self.device,
                    generator=self.torch_generator,
                )
                self.particles = self.particles.index_select(0, permutation_t).contiguous()
                self.node_velocities = self.node_velocities.index_select(0, permutation_t).contiguous()
                self.weights = self.weights.index_select(0, permutation_t).contiguous()
        else:
            probabilities = np.maximum(np.asarray(self.weights, dtype=np.float64), 0.0)
            probabilities /= max(float(np.sum(probabilities)), 1e-12)
            parent_indices_array = self.rng.choice(
                total_count,
                size=local_count + conditioned_count,
                replace=True,
                p=probabilities,
            )
            self._ensure_velocity_array()
            local_particles = self.particles[parent_indices_array[:local_count]].copy()
            local_velocities = self.node_velocities[parent_indices_array[:local_count]].copy()
            if self._velocity_enabled():
                local_velocities *= float(np.clip(self.config.velocity_damping, 0.0, 1.0))
                local_velocities += self.rng.normal(
                    0.0,
                    float(self.config.velocity_process_std_mps) * float(self.last_motion_noise_scale),
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
                        float(self.config.process_direction_std) * float(self.last_motion_noise_scale),
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
                self.particles[
                    parent_indices_array[local_count:local_count + conditioned_count]
                ],
                self.last_endpoint_nodes,
                self.last_endpoint_tangents,
                self.last_endpoint_tangent_confidence,
                self.segment_length_m,
                self.config,
                self.rng,
                noise_scale=self.last_motion_noise_scale,
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
                local_ratio,
                conditioned_ratio,
                global_ratio,
            )
            permutation_array = self.rng.permutation(total_count)
            self.particles = self.particles[permutation_array].copy()
            self.node_velocities = self.node_velocities[permutation_array].copy()
            self.weights = self.weights[permutation_array].copy()

        self.last_local_proposal_ratio = float(local_count) / float(total_count)
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
        effective_sample_size, mean_node_speed_mps = self._posterior_scalar_diagnostics()
        nodes = self._estimate_nodes()
        self._sync_crossing_diagnostics()
        return CableParticleFilterResult(
            points_xyz=np.ascontiguousarray(nodes, dtype=np.float32),
            effective_sample_size=effective_sample_size,
            measurement_used=bool(measurement_used),
            prediction_only=bool(prediction_only),
            lost_frames=int(self.lost_frames),
            motion_noise_scale=float(self.last_motion_noise_scale),
            segment_length_m=float(self.segment_length_m),
            measurement_point_count=int(self.last_measurement_point_count),
            support_points_xyz=self.last_support_points.copy(),
            support_point_affinities=self.last_support_point_affinities.copy(),
            visible_segments=self.last_visible_segments.copy(),
            visible_nodes=self.last_visible_nodes.copy(),
            local_proposal_ratio=float(self.last_local_proposal_ratio),
            endpoint_conditioned_proposal_ratio=float(self.last_endpoint_conditioned_proposal_ratio),
            global_random_particle_ratio=float(self.last_global_random_particle_ratio),
            path_support_rms_m=float(self.last_path_support_rms_m),
            mean_support_affinity=float(self.last_mean_support_affinity),
            supported_sample_fraction=float(self.last_supported_sample_fraction),
            endpoint_tangent_confidence=self.last_endpoint_tangent_confidence.copy(),
            endpoint_tangent_support_count=self.last_endpoint_tangent_support_count.copy(),
            estimate_particle_count=int(self._estimate_particle_count_cache),
            estimate_weight_mass=float(self._estimate_weight_mass_cache),
            mean_node_speed_mps=mean_node_speed_mps,
            endpoint_speed_mps=float(self.last_endpoint_speed_mps),
            endpoint_motion_innovation_m=float(self.last_endpoint_motion_innovation_m),
            crossing_target_count=int(self.last_crossing_target_count),
            crossing_reward=float(self.last_crossing_reward),
            crossing_distance_px=float(self.last_crossing_distance_px),
            crossing_angle_error_deg=float(self.last_crossing_angle_error_deg),
            crossing_closest_xy=self.last_crossing_closest_xy.copy(),
            crossing_target_xy=self.last_crossing_target_xy.copy(),
            crossing_axis_xy=self.last_crossing_axis_xy.copy(),
            union_coverage_selected=bool(self.last_union_coverage_selected),
            union_coverage_rank=int(self.last_union_coverage_rank),
            union_coverage_rms_m=float(self.last_union_coverage_rms_m),
            union_coverage_fraction=float(self.last_union_coverage_fraction),
            union_coverage_rms_gain_m=float(self.last_union_coverage_rms_gain_m),
            union_coverage_fraction_gain=float(self.last_union_coverage_fraction_gain),
            # The caller resolves this only after recording the complete
            # estimate stage.  Resolving here caused a redundant CUDA sync and
            # omitted the estimate stage from the first timing snapshot.
            stage_seconds={},
            particle_diagnostics=self._particle_estimate_diagnostics_cache,
        )

    def _estimate_nodes(self):
        if self._estimate_nodes_cache is not None:
            return self._estimate_nodes_cache.copy()
        particle_count = len(self.particles)
        configured_top_count = int(getattr(self.config, "estimate_top_particle_count", 32))
        if self.last_union_coverage_selected:
            configured_top_count = max(
                configured_top_count,
                int(getattr(self.config, "union_coverage_top_particle_count", configured_top_count)),
            )
        top_count = int(np.clip(
            configured_top_count,
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
                top_weights_t, top_indices_t = torch.topk(
                    rank_weights,
                    k=top_count,
                    largest=True,
                    sorted=False,
                )
                representative_index = self.last_representative_particle_index
                if representative_index is None or not (0 <= int(representative_index) < particle_count):
                    representative_index = posterior_medoid_index_torch(
                        self.particles,
                        rank_weights,
                        top_indices_t,
                    )
                representative_index = int(representative_index)
                self.last_representative_particle_index = representative_index
                nodes_t = self.particles[representative_index]
                weight_mass_t = torch.sum(torch.clamp_min(top_weights_t, 0.0))
                nodes = nodes_t.cpu().numpy().astype(np.float32, copy=False)
                diagnostics_enabled = bool(self.config.particle_diagnostics_enabled)
                if diagnostics_enabled:
                    map_nodes = self.particles[torch.argmax(rank_weights)].cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                    selected_particles = self.particles.index_select(
                        0,
                        top_indices_t,
                    ).cpu().numpy().astype(np.float32, copy=False)
                weight_mass = float(weight_mass_t.item())
                selected_count = top_count
            self._estimate_nodes_cache = np.ascontiguousarray(nodes, dtype=np.float32)
            self._estimate_particle_count_cache = selected_count
            self._estimate_weight_mass_cache = weight_mass
            self._particle_estimate_diagnostics_cache = (
                particle_estimate_diagnostics(
                    self._estimate_nodes_cache,
                    map_nodes,
                    selected_particles,
                    endpoint_tangents=self.last_endpoint_tangents,
                    endpoint_tangent_confidence=self.last_endpoint_tangent_confidence,
                    endpoint_tangent_support_count=self.last_endpoint_tangent_support_count,
                    mean_support_affinity=self.last_mean_support_affinity,
                    supported_sample_fraction=self.last_supported_sample_fraction,
                    visible_segment_fraction=float(np.mean(self.last_visible_segments)),
                    local_proposal_ratio=self.last_local_proposal_ratio,
                    endpoint_conditioned_proposal_ratio=self.last_endpoint_conditioned_proposal_ratio,
                    global_random_particle_ratio=self.last_global_random_particle_ratio,
                )
                if diagnostics_enabled
                else None
            )
            return self._estimate_nodes_cache.copy()

        rank_weights = np.where(np.isfinite(self.weights), self.weights, -np.inf)
        top_indices = np.argpartition(rank_weights, particle_count - top_count)[-top_count:]
        representative_index = self.last_representative_particle_index
        if representative_index is None or not (0 <= int(representative_index) < particle_count):
            representative_index = posterior_medoid_index(
                self.particles,
                rank_weights,
                top_indices,
            )
        self.last_representative_particle_index = int(representative_index)
        nodes = self.particles[int(representative_index)]
        weight_mass = float(np.sum(np.maximum(rank_weights[top_indices], 0.0)))
        self._estimate_nodes_cache = np.ascontiguousarray(nodes, dtype=np.float32)
        self._estimate_particle_count_cache = top_count
        self._estimate_weight_mass_cache = weight_mass
        self._particle_estimate_diagnostics_cache = (
            particle_estimate_diagnostics(
                self._estimate_nodes_cache,
                self.particles[int(np.argmax(rank_weights))],
                self.particles[top_indices],
                endpoint_tangents=self.last_endpoint_tangents,
                endpoint_tangent_confidence=self.last_endpoint_tangent_confidence,
                endpoint_tangent_support_count=self.last_endpoint_tangent_support_count,
                mean_support_affinity=self.last_mean_support_affinity,
                supported_sample_fraction=self.last_supported_sample_fraction,
                visible_segment_fraction=float(np.mean(self.last_visible_segments)),
                local_proposal_ratio=self.last_local_proposal_ratio,
                endpoint_conditioned_proposal_ratio=self.last_endpoint_conditioned_proposal_ratio,
                global_random_particle_ratio=self.last_global_random_particle_ratio,
            )
            if bool(self.config.particle_diagnostics_enabled)
            else None
        )
        return self._estimate_nodes_cache.copy()

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

    def _update_endpoint_motion_model(self, endpoint_nodes, dt):
        current = np.asarray(endpoint_nodes, dtype=np.float64)
        previous = np.asarray(self.last_endpoint_nodes, dtype=np.float64)
        if current.shape != (2, 3) or not np.all(np.isfinite(current)):
            return
        if previous.shape != (2, 3) or not np.all(np.isfinite(previous)):
            self.last_endpoint_velocity_mps.fill(0.0)
            self.last_endpoint_speed_mps = 0.0
            self.last_endpoint_motion_innovation_m = 0.0
            self.last_motion_noise_scale = 1.0
            return

        if not bool(self.config.adaptive_motion_noise_enabled):
            dt = max(float(dt), 1e-3)
            observed_velocity = (current - previous) / dt
            self.last_endpoint_velocity_mps = np.ascontiguousarray(
                observed_velocity,
                dtype=np.float32,
            )
            self.last_endpoint_speed_mps = float(
                np.mean(np.linalg.norm(observed_velocity, axis=1))
            )
            self.last_endpoint_motion_innovation_m = 0.0
            self.last_motion_noise_scale = 1.0
            return

        dt = max(float(dt), 1e-3)
        observed_velocity = (current - previous) / dt
        predicted = previous + np.asarray(self.last_endpoint_velocity_mps, dtype=np.float64) * dt
        endpoint_speed = float(np.mean(np.linalg.norm(observed_velocity, axis=1)))
        innovation = float(np.mean(np.linalg.norm(current - predicted, axis=1)))
        speed_reference = max(float(self.config.motion_speed_reference_mps), 1e-4)
        innovation_reference = max(float(self.config.motion_innovation_reference_m), 1e-5)
        target_scale = float(np.sqrt(
            1.0
            + (endpoint_speed / speed_reference) ** 2
            + (innovation / innovation_reference) ** 2
        ))
        target_scale = float(np.clip(target_scale, 1.0, self.config.max_motion_noise_scale))
        adaptation = float(np.clip(self.config.motion_noise_adaptation, 0.0, 1.0))
        self.last_motion_noise_scale = float(np.clip(
            (1.0 - adaptation) * self.last_motion_noise_scale + adaptation * target_scale,
            1.0,
            self.config.max_motion_noise_scale,
        ))
        velocity_blend = min(1.0, max(0.15, adaptation))
        self.last_endpoint_velocity_mps = np.ascontiguousarray(
            (1.0 - velocity_blend) * self.last_endpoint_velocity_mps
            + velocity_blend * observed_velocity,
            dtype=np.float32,
        )
        self.last_endpoint_speed_mps = endpoint_speed
        self.last_endpoint_motion_innovation_m = innovation

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

    def _record_cuda_interval(self, name, start_event, end_event):
        """Record one explicitly bounded CUDA stage shared by coupled filters."""
        if not self.cuda_state:
            raise RuntimeError("CUDA intervals can only be recorded for CUDA filter state.")
        self.last_stage_seconds[str(name)] = self.last_stage_seconds.get(str(name), 0.0)
        self._cuda_stage_events.append((str(name), start_event, end_event))
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

    def _posterior_scalar_diagnostics(self):
        """Normalize once and transfer ESS/node-speed in one synchronization."""

        if self.cuda_state:
            with torch.inference_mode(), torch.cuda.stream(self.cuda_stream):
                self.weights = normalize_particle_weights_torch(self.weights)
                effective_sample_size_t = 1.0 / torch.sum(
                    self.weights * self.weights
                ).clamp_min(1e-20)
                if self.node_velocities is None:
                    mean_node_speed_t = torch.zeros(
                        (), dtype=torch.float32, device=self.device
                    )
                else:
                    speeds_t = torch.mean(
                        torch.linalg.vector_norm(self.node_velocities, dim=2),
                        dim=1,
                    )
                    mean_node_speed_t = torch.sum(speeds_t * self.weights)
                values = torch.stack((effective_sample_size_t, mean_node_speed_t))
                effective_sample_size, mean_node_speed = (
                    values.cpu().numpy().astype(np.float64, copy=False)
                )
            return (
                float(effective_sample_size),
                float(mean_node_speed) if np.isfinite(mean_node_speed) else 0.0,
            )

        self.weights = normalize_particle_weights(self.weights)
        effective_sample_size = 1.0 / max(
            float(np.sum(self.weights * self.weights)),
            1e-20,
        )
        if self.node_velocities is None:
            return float(effective_sample_size), 0.0
        speeds = np.mean(np.linalg.norm(self.node_velocities, axis=2), axis=1)
        return float(effective_sample_size), weighted_mean_or_zero(speeds, self.weights)

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

def select_union_coverage_representatives(particle_filters, observation_points):
    """Select one or two independent PF representatives whose union explains the cloud.

    Independent posterior weights first restrict each PF to its top candidates.
    The additional likelihood is the robust observation-to-union distance, so
    neither PF is individually penalized for observations explained by the
    other cable.  Particle populations and weights are not coupled or changed.
    """

    filters = list(particle_filters or ())
    if len(filters) not in (1, 2):
        raise ValueError("Union coverage selection supports one or two particle filters.")
    if any(
        not particle_filter.initialized
        or particle_filter.particles is None
        or particle_filter.weights is None
        for particle_filter in filters
    ):
        return None
    points = valid_points(observation_points)
    minimum_points = max(int(particle_filter.config.min_measurement_points) for particle_filter in filters)
    if len(points) < minimum_points:
        return None
    weight = float(np.mean([
        max(0.0, float(particle_filter.config.union_coverage_weight))
        for particle_filter in filters
    ]))
    if weight <= 0.0:
        return None
    cuda_flags = [bool(particle_filter.cuda_state) for particle_filter in filters]
    if any(cuda_flags) and not all(cuda_flags):
        raise ValueError("Union coverage selection requires all PFs to use the same backend.")
    settings = {
        "weight": weight,
        "sigma_m": float(np.mean([
            max(1e-5, float(particle_filter.config.measurement_node_std_m))
            for particle_filter in filters
        ])),
        "robust_distance_m": float(np.mean([
            max(1e-5, float(particle_filter.config.robust_distance_m))
            for particle_filter in filters
        ])),
        "visibility_distance_m": float(np.mean([
            max(1e-5, float(particle_filter.config.support_visibility_distance_m))
            for particle_filter in filters
        ])),
    }
    selection = (
        _select_union_coverage_cuda(filters, points, settings)
        if all(cuda_flags)
        else _select_union_coverage_cpu(filters, points, settings)
    )
    for particle_filter, particle_index, rank in zip(
        filters,
        selection.selected_particle_indices,
        selection.selected_ranks,
    ):
        particle_filter.last_representative_particle_index = int(particle_index)
        particle_filter.last_union_coverage_selected = True
        particle_filter.last_union_coverage_rank = int(rank)
        particle_filter.last_union_coverage_rms_m = float(selection.robust_coverage_rms_m)
        particle_filter.last_union_coverage_fraction = float(selection.covered_point_fraction)
        particle_filter.last_union_coverage_rms_gain_m = float(selection.robust_rms_gain_m)
        particle_filter.last_union_coverage_fraction_gain = float(selection.covered_fraction_gain)
    return selection


def _select_union_coverage_cpu(filters, points, settings):
    started = time.perf_counter()
    top_indices = []
    top_particles = []
    top_log_weights = []
    point_squared = []
    for particle_filter in filters:
        weights = normalize_particle_weights(particle_filter.weights)
        top_count = int(np.clip(
            particle_filter.config.union_coverage_top_particle_count,
            1,
            len(weights),
        ))
        indices = np.argsort(weights, kind="stable")[::-1][:top_count]
        particles = np.asarray(particle_filter.particles, dtype=np.float64)[indices]
        distances = point_to_particle_segment_squared_distances(
            points,
            particles[:, :-1, :],
            particles[:, 1:, :],
        )
        top_indices.append(np.ascontiguousarray(indices, dtype=np.int64))
        top_particles.append(particles)
        top_log_weights.append(np.log(np.maximum(weights[indices], 1e-20)))
        point_squared.append(np.min(distances, axis=1))

    robust_squared = float(settings["robust_distance_m"]) ** 2
    sigma_squared = float(settings["sigma_m"]) ** 2
    scale = 0.5 * float(settings["weight"]) / max(sigma_squared, 1e-12)
    if len(filters) == 1:
        union_squared_by_choice = point_squared[0]
        baseline_union_squared = union_squared_by_choice[0]
        coverage_cost = np.mean(np.minimum(union_squared_by_choice, robust_squared), axis=1)
        score = top_log_weights[0] - scale * coverage_cost
        positions = (int(np.argmax(score)),)
        selected_union_squared = union_squared_by_choice[positions[0]]
    else:
        union_squared_by_pair = np.minimum(
            point_squared[0][:, None, :],
            point_squared[1][None, :, :],
        )
        baseline_union_squared = union_squared_by_pair[0, 0]
        coverage_cost = np.mean(np.minimum(union_squared_by_pair, robust_squared), axis=2)
        score = (
            top_log_weights[0][:, None]
            + top_log_weights[1][None, :]
            - scale * coverage_cost
        )
        positions = tuple(int(value) for value in np.unravel_index(int(np.argmax(score)), score.shape))
        selected_union_squared = union_squared_by_pair[positions]

    visibility_squared = float(settings["visibility_distance_m"]) ** 2
    for particle_filter, particles, distances, position in zip(
        filters,
        top_particles,
        point_squared,
        positions,
    ):
        selected_particle = particles[position:position + 1]
        support_squared = particle_chain_support_squared_distances(
            selected_particle,
            points,
            int(particle_filter.config.path_support_samples_per_segment),
        )[0].reshape(
            particle_filter.segment_count,
            int(particle_filter.config.path_support_samples_per_segment),
        )
        selected_point_squared = distances[position]
        affinity = np.exp(-0.5 * selected_point_squared / max(sigma_squared, 1e-12))
        affinity = np.where(selected_point_squared <= visibility_squared, affinity, 0.0)
        particle_filter._pending_support_diagnostics = (
            np.ascontiguousarray(support_squared, dtype=np.float64),
            np.ascontiguousarray(affinity, dtype=np.float32),
        )

    robust_rms = float(np.sqrt(np.mean(np.minimum(selected_union_squared, robust_squared))))
    covered_fraction = float(np.mean(selected_union_squared <= visibility_squared))
    baseline_rms = float(np.sqrt(np.mean(np.minimum(baseline_union_squared, robust_squared))))
    baseline_fraction = float(np.mean(baseline_union_squared <= visibility_squared))
    return UnionCoverageSelection(
        selected_particle_indices=tuple(
            int(indices[position]) for indices, position in zip(top_indices, positions)
        ),
        selected_ranks=tuple(int(position) + 1 for position in positions),
        robust_coverage_rms_m=robust_rms,
        covered_point_fraction=covered_fraction,
        robust_rms_gain_m=baseline_rms - robust_rms,
        covered_fraction_gain=covered_fraction - baseline_fraction,
        observation_point_count=int(len(points)),
        stage_seconds=float(time.perf_counter() - started),
    )


def _select_union_coverage_cuda(filters, points, settings):
    started = time.perf_counter()
    stream = filters[0].cuda_stream
    for particle_filter in filters:
        stream.wait_stream(particle_filter.cuda_stream)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode(), torch.cuda.stream(stream):
        start_event.record(stream)
        points_t = torch.as_tensor(
            np.ascontiguousarray(points, dtype=np.float32),
            dtype=torch.float32,
            device=filters[0].device,
        ).contiguous()
        top_indices_t = []
        top_particles_t = []
        top_log_weights_t = []
        point_squared_t = []
        for particle_filter in filters:
            weights_t = normalize_particle_weights_torch(particle_filter.weights)
            top_count = int(np.clip(
                particle_filter.config.union_coverage_top_particle_count,
                1,
                len(weights_t),
            ))
            top_weights_t, indices_t = torch.topk(weights_t, top_count, largest=True, sorted=True)
            particles_t = particle_filter.particles.index_select(0, indices_t).contiguous()
            distances_t, _nearest_t = particle_point_distances_cuda(
                particles_t.unsqueeze(0),
                points_t,
                stream,
            )
            top_indices_t.append(indices_t)
            top_particles_t.append(particles_t)
            top_log_weights_t.append(torch.log(top_weights_t.clamp_min(1e-20)))
            point_squared_t.append(distances_t[0])

        robust_squared = float(settings["robust_distance_m"]) ** 2
        sigma_squared = float(settings["sigma_m"]) ** 2
        visibility_squared = float(settings["visibility_distance_m"]) ** 2
        scale = 0.5 * float(settings["weight"]) / max(sigma_squared, 1e-12)
        if len(filters) == 1:
            baseline_union_squared_t = point_squared_t[0][0]
            coverage_cost_t = torch.mean(torch.clamp(point_squared_t[0], max=robust_squared), dim=1)
            score_t = top_log_weights_t[0] - scale * coverage_cost_t
            positions_t = (torch.argmax(score_t),)
            selected_union_squared_t = point_squared_t[0][positions_t[0]]
        else:
            union_squared_t = torch.minimum(
                point_squared_t[0][:, None, :],
                point_squared_t[1][None, :, :],
            )
            baseline_union_squared_t = union_squared_t[0, 0]
            coverage_cost_t = torch.mean(torch.clamp(union_squared_t, max=robust_squared), dim=2)
            score_t = (
                top_log_weights_t[0][:, None]
                + top_log_weights_t[1][None, :]
                - scale * coverage_cost_t
            )
            flat_position_t = torch.argmax(score_t)
            positions_t = (
                torch.div(flat_position_t, score_t.shape[1], rounding_mode="floor"),
                torch.remainder(flat_position_t, score_t.shape[1]),
            )
            selected_union_squared_t = union_squared_t[positions_t]

        for particle_filter, particles_t, distances_t, position_t in zip(
            filters,
            top_particles_t,
            point_squared_t,
            positions_t,
        ):
            selected_particle_t = particles_t[position_t].reshape(1, particle_filter.node_count, 3)
            support_squared_t = particle_support_distances_cuda(
                selected_particle_t.unsqueeze(0),
                points_t,
                int(particle_filter.config.path_support_samples_per_segment),
                stream,
            )[0, 0]
            selected_point_squared_t = distances_t[position_t]
            affinity_t = torch.exp(-0.5 * selected_point_squared_t / max(sigma_squared, 1e-12))
            affinity_t = torch.where(
                selected_point_squared_t <= visibility_squared,
                affinity_t,
                torch.zeros_like(affinity_t),
            )
            particle_filter._pending_support_diagnostics = (support_squared_t, affinity_t)

        robust_rms_t = torch.sqrt(torch.mean(torch.clamp(selected_union_squared_t, max=robust_squared)))
        covered_fraction_t = torch.mean((selected_union_squared_t <= visibility_squared).to(torch.float32))
        baseline_rms_t = torch.sqrt(torch.mean(torch.clamp(baseline_union_squared_t, max=robust_squared)))
        baseline_fraction_t = torch.mean((baseline_union_squared_t <= visibility_squared).to(torch.float32))
        packed_t = torch.cat((
            torch.stack([
                top_indices_t[index][positions_t[index]].to(torch.float32)
                for index in range(len(filters))
            ]),
            torch.stack([position.to(torch.float32) + 1.0 for position in positions_t]),
            robust_rms_t.reshape(1),
            covered_fraction_t.reshape(1),
            (baseline_rms_t - robust_rms_t).reshape(1),
            (covered_fraction_t - baseline_fraction_t).reshape(1),
        ))
        end_event.record(stream)
        packed = packed_t.cpu().numpy().astype(np.float32, copy=False)

    count = len(filters)
    return UnionCoverageSelection(
        selected_particle_indices=tuple(int(round(float(value))) for value in packed[:count]),
        selected_ranks=tuple(int(round(float(value))) for value in packed[count:2 * count]),
        robust_coverage_rms_m=float(packed[-4]),
        covered_point_fraction=float(packed[-3]),
        robust_rms_gain_m=float(packed[-2]),
        covered_fraction_gain=float(packed[-1]),
        observation_point_count=int(len(points)),
        stage_seconds=float(time.perf_counter() - started),
        cuda_events=(start_event, end_event),
    )


def update_cable_particle_filters(
    particle_filters,
    measurements,
    dt=1.0 / 30.0,
    count_lost=True,
    crossing_targets_by_filter=None,
    camera_intrinsics=None,
    union_coverage_enabled=False,
):
    """Advance independent endpoint-anchored cable particle filters.

    Every PF owns its particles, weights, transition draws, resampling, and
    posterior estimate.  All PFs may observe the same unlabeled cable cloud,
    but one filter never changes another filter's state.  CUDA filters enqueue
    prediction and measurement work on their own streams, so the two cable
    updates execute concurrently without index locking.
    """

    filters = list(particle_filters or ())
    observations = list(measurements or ())
    if len(filters) != len(observations):
        raise ValueError("particle_filters and measurements must have the same length.")
    if not filters:
        return []
    crossing_targets = (
        [tuple() for _ in filters]
        if crossing_targets_by_filter is None
        else list(crossing_targets_by_filter)
    )
    if len(crossing_targets) != len(filters):
        raise ValueError("crossing_targets_by_filter must match particle_filters length.")

    contexts = [
        particle_filter._prepare_update(
            observation,
            dt=dt,
            count_lost=count_lost,
        )
        for particle_filter, observation in zip(filters, observations)
    ]
    for particle_filter, context in zip(filters, contexts):
        if not (context.measurement_used and particle_filter.initialized):
            continue
        stage_start = time.perf_counter()
        cable_measurement_update(particle_filter, context.measurement_points)
        particle_filter._record_stage("measurement", stage_start)

    for particle_filter, targets in zip(filters, crossing_targets):
        if not particle_filter.initialized or not targets or camera_intrinsics is None:
            continue
        stage_start = time.perf_counter()
        cable_crossing_likelihood_update(
            particle_filter,
            targets,
            camera_intrinsics,
        )
        particle_filter._record_stage("crossing", stage_start)

    if bool(union_coverage_enabled):
        coverage_points = next(
            (
                context.measurement_points
                for context in contexts
                if context.measurement_points is not None and len(context.measurement_points) > 0
            ),
            None,
        )
        if coverage_points is not None:
            selection = select_union_coverage_representatives(filters, coverage_points)
            if selection is not None:
                if selection.cuda_events is not None:
                    start_event, end_event = selection.cuda_events
                    for particle_filter in filters:
                        particle_filter._record_cuda_interval("union", start_event, end_event)
                        boundary_event = torch.cuda.Event(enable_timing=True)
                        boundary_event.record(particle_filter.cuda_stream)
                        particle_filter._cuda_previous_event = boundary_event
                else:
                    for particle_filter in filters:
                        particle_filter.last_stage_seconds["union"] = (
                            particle_filter.last_stage_seconds.get("union", 0.0)
                            + float(selection.stage_seconds)
                        )

    results = []
    for particle_filter, context in zip(filters, contexts):
        if not particle_filter.initialized:
            results.append(None)
            continue
        stage_start = time.perf_counter()
        particle_filter._sync_endpoint_tangent_diagnostics()
        particle_filter._sync_support_diagnostics()
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


def cable_crossing_likelihood_update(particle_filter, targets, intrinsics):
    """Reward one PF when its RGB projection passes through assigned crossings."""

    targets = tuple(targets or ())
    gain = max(0.0, float(particle_filter.config.crossing_log_reward))
    particle_filter.last_crossing_target_count = len(targets)
    if (
        not targets
        or intrinsics is None
        or gain <= 0.0
        or not particle_filter.initialized
        or particle_filter.particles is None
        or particle_filter.weights is None
    ):
        return
    if particle_filter.cuda_state:
        with torch.inference_mode(), torch.cuda.stream(particle_filter.cuda_stream):
            reward_t, distance_t, angle_t, closest_t, target_index_t = particle_crossing_rewards_torch(
                particle_filter.particles,
                targets,
                intrinsics,
                position_sigma_px=float(particle_filter.config.crossing_position_sigma_px),
                angle_sigma_deg=float(particle_filter.config.crossing_angle_sigma_deg),
            )
            prior_t = normalize_particle_weights_torch(particle_filter.weights)
            log_weights_t = torch.log(prior_t.clamp_min(1e-20)) + gain * reward_t
            particle_filter.weights = torch.softmax(log_weights_t, dim=0).contiguous()
            particle_filter._pending_crossing_diagnostics = (
                reward_t,
                distance_t,
                angle_t,
                closest_t,
                target_index_t,
                targets,
            )
    else:
        reward, distance, angle, closest, target_index = particle_crossing_rewards(
            particle_filter.particles,
            targets,
            intrinsics,
            position_sigma_px=float(particle_filter.config.crossing_position_sigma_px),
            angle_sigma_deg=float(particle_filter.config.crossing_angle_sigma_deg),
        )
        prior = normalize_particle_weights(particle_filter.weights)
        log_weights = np.log(np.maximum(prior, 1e-20)) + gain * reward
        maximum = float(np.max(log_weights)) if len(log_weights) else 0.0
        unnormalized = np.exp(log_weights - maximum)
        particle_filter.weights = normalize_particle_weights(unnormalized)
        particle_filter._pending_crossing_diagnostics = (
            reward,
            distance,
            angle,
            closest,
            target_index,
            targets,
        )
    particle_filter._invalidate_estimate_cache()


def cable_measurement_update(particle_filter, measurement_points):
    """Score one cable independently against the unlabeled cable support cloud.

    Dense samples from the proposed path must lie near some segmented cable
    observation. Observations belonging to another cable are never averaged
    into a penalty for this PF. Endpoint positions and segment lengths remain
    hard geometric constraints.
    """

    points = valid_points(measurement_points)
    if (
        not particle_filter.initialized
        or particle_filter.particles is None
        or particle_filter.weights is None
        or len(points) == 0
    ):
        return
    if particle_filter.cuda_state:
        _cable_measurement_update_cuda(particle_filter, points)
    else:
        _cable_measurement_update_cpu(particle_filter, points)
    particle_filter._invalidate_estimate_cache()


def _cable_measurement_update_cuda(particle_filter, points):
    stream = particle_filter.cuda_stream
    config = particle_filter.config
    with torch.inference_mode(), torch.cuda.stream(stream):
        points_t = torch.as_tensor(
            np.ascontiguousarray(points, dtype=np.float32),
            dtype=torch.float32,
            device=particle_filter.device,
        ).contiguous()
        sample_count = max(1, int(config.path_support_samples_per_segment))
        support_squared_t = particle_support_distances_cuda(
            particle_filter.particles.unsqueeze(0),
            points_t,
            sample_count,
            stream,
        )[0]
        sigma = max(1e-5, float(config.measurement_node_std_m))
        robust_distance = max(float(config.robust_distance_m), sigma)
        support_cost_t = (
            torch.clamp(support_squared_t, max=robust_distance * robust_distance)
            if bool(config.robust_measurement_enabled)
            else support_squared_t
        )
        scores_t = (
            max(0.0, float(config.path_support_weight))
            * torch.mean(support_cost_t, dim=(1, 2))
        )
        scores_t = scores_t + particle_endpoint_tangent_penalty_torch(
            particle_filter.particles,
            particle_filter.last_endpoint_tangents_t,
            particle_filter.last_endpoint_tangent_confidence_t,
            float(config.endpoint_tangent_min_confidence),
            float(config.endpoint_tangent_likelihood_scale_m),
        )
        if float(config.bend_penalty_m) > 0.0:
            directions_t = normalize_vectors_torch(
                particle_filter.particles[:, 1:, :]
                - particle_filter.particles[:, :-1, :]
            )
            dots_t = torch.sum(
                directions_t[:, :-1, :] * directions_t[:, 1:, :],
                dim=2,
            ).clamp(-1.0, 1.0)
            scores_t = scores_t + (
                torch.mean(torch.clamp(1.0 - dots_t, min=0.0), dim=1)
                * float(config.bend_penalty_m) ** 2
            )

        prior_t = normalize_particle_weights_torch(particle_filter.weights)
        log_weights_t = (
            torch.log(prior_t.clamp_min(1e-20))
            - 0.5 * scores_t / (sigma * sigma)
        )
        endpoint_residual_t = torch.maximum(
            torch.linalg.vector_norm(
                particle_filter.particles[:, 0, :]
                - particle_filter.last_endpoint_nodes_t[0][None, :],
                dim=1,
            ),
            torch.linalg.vector_norm(
                particle_filter.particles[:, -1, :]
                - particle_filter.last_endpoint_nodes_t[-1][None, :],
                dim=1,
            ),
        )
        valid_t = (
            endpoint_residual_t <= max(
            float(config.endpoint_constraint_tolerance_m),
            1e-7,
            )
        ) & torch.isfinite(log_weights_t)
        finite_log_t = torch.where(
            valid_t,
            log_weights_t,
            torch.full_like(log_weights_t, -torch.inf),
        )
        any_valid_t = torch.any(valid_t)
        maximum_t = torch.where(
            any_valid_t,
            torch.max(finite_log_t),
            torch.zeros((), dtype=finite_log_t.dtype, device=finite_log_t.device),
        )
        unnormalized_t = torch.where(
            valid_t,
            torch.exp(finite_log_t - maximum_t),
            torch.zeros_like(finite_log_t),
        )
        total_t = torch.sum(unnormalized_t)
        fallback_t = valid_t.to(torch.float32)
        fallback_t = torch.where(
            any_valid_t,
            fallback_t / fallback_t.sum().clamp_min(1.0),
            torch.full_like(fallback_t, 1.0 / len(fallback_t)),
        )
        particle_filter.weights = torch.where(
            torch.isfinite(total_t) & (total_t > 1e-20),
            unnormalized_t / total_t.clamp_min(1e-20),
            fallback_t,
        ).contiguous()

        map_index_t = torch.argmax(particle_filter.weights).reshape(1)
        map_particle_t = particle_filter.particles.index_select(0, map_index_t)
        map_support_squared_t = support_squared_t.index_select(0, map_index_t)[0]
        visibility_distance = max(float(config.support_visibility_distance_m), 1e-6)
        if bool(config.point_support_diagnostics_enabled):
            point_squared_t, _nearest_t = particle_point_distances_cuda(
                map_particle_t.unsqueeze(0),
                points_t,
                stream,
            )
            point_affinity_t = torch.exp(
                -0.5 * point_squared_t[0, 0] / (sigma * sigma)
            )
            point_affinity_t = torch.where(
                point_squared_t[0, 0] <= visibility_distance * visibility_distance,
                point_affinity_t,
                torch.zeros_like(point_affinity_t),
            )
        else:
            point_affinity_t = torch.empty(
                0,
                dtype=torch.float32,
                device=particle_filter.device,
            )
        particle_filter._pending_support_diagnostics = (
            map_support_squared_t,
            point_affinity_t,
        )


def _cable_measurement_update_cpu(particle_filter, points):
    config = particle_filter.config
    sample_count = max(1, int(config.path_support_samples_per_segment))
    support_squared = particle_chain_support_squared_distances(
        particle_filter.particles,
        points,
        sample_count,
    ).reshape(
        len(particle_filter.particles),
        particle_filter.segment_count,
        sample_count,
    )
    sigma = max(1e-5, float(config.measurement_node_std_m))
    robust_distance = max(float(config.robust_distance_m), sigma)
    support_cost = (
        np.minimum(support_squared, robust_distance * robust_distance)
        if bool(config.robust_measurement_enabled)
        else support_squared
    )
    scores = (
        max(0.0, float(config.path_support_weight))
        * np.mean(support_cost, axis=(1, 2))
    )
    scores += particle_endpoint_tangent_penalty(
        particle_filter.particles,
        particle_filter.last_endpoint_tangents,
        particle_filter.last_endpoint_tangent_confidence,
        float(config.endpoint_tangent_min_confidence),
        float(config.endpoint_tangent_likelihood_scale_m),
    )
    if float(config.bend_penalty_m) > 0.0:
        scores += particle_bend_penalty(
            particle_filter.particles,
            penalty_m=float(config.bend_penalty_m),
        )

    prior = normalize_particle_weights(particle_filter.weights)
    log_weights = (
        np.log(np.maximum(prior, 1e-20))
        - 0.5 * scores / (sigma * sigma)
    )
    endpoint_residual = np.maximum(
        np.linalg.norm(
            particle_filter.particles[:, 0, :]
            - particle_filter.last_endpoint_nodes[0][None, :],
            axis=1,
        ),
        np.linalg.norm(
            particle_filter.particles[:, -1, :]
            - particle_filter.last_endpoint_nodes[-1][None, :],
            axis=1,
        ),
    )
    valid = (
        endpoint_residual <= max(
            float(config.endpoint_constraint_tolerance_m),
            1e-7,
        )
    ) & np.isfinite(log_weights)
    finite_log = np.where(valid, log_weights, -np.inf)
    maximum = float(np.max(finite_log)) if np.any(valid) else 0.0
    unnormalized = np.zeros_like(log_weights, dtype=np.float64)
    unnormalized[valid] = np.exp(finite_log[valid] - maximum)
    total = float(np.sum(unnormalized))
    if np.isfinite(total) and total > 1e-20:
        particle_filter.weights = np.ascontiguousarray(
            unnormalized / total,
            dtype=np.float64,
        )
    elif np.any(valid):
        particle_filter.weights = np.ascontiguousarray(
            valid.astype(np.float64) / float(np.count_nonzero(valid)),
        )
    else:
        particle_filter.weights = np.full(
            len(particle_filter.particles),
            1.0 / len(particle_filter.particles),
            dtype=np.float64,
        )

    map_index = int(np.argmax(particle_filter.weights))
    map_particle = particle_filter.particles[map_index:map_index + 1]
    visibility_distance = max(float(config.support_visibility_distance_m), 1e-6)
    if bool(config.point_support_diagnostics_enabled):
        point_squared = np.min(
            point_to_particle_segment_squared_distances(
                points,
                map_particle[:, :-1, :],
                map_particle[:, 1:, :],
            ),
            axis=1,
        )[0]
        point_affinities = np.exp(-0.5 * point_squared / (sigma * sigma))
        point_affinities = np.where(
            point_squared <= visibility_distance * visibility_distance,
            point_affinities,
            0.0,
        )
    else:
        point_affinities = np.empty(0, dtype=np.float32)
    _apply_support_diagnostics(
        particle_filter,
        support_squared[map_index],
        point_affinities,
    )


def _apply_support_diagnostics(particle_filter, map_support_squared, point_affinities):
    support_squared = np.asarray(map_support_squared, dtype=np.float64)
    if support_squared.shape[0] != particle_filter.segment_count:
        support_squared = np.empty(
            (particle_filter.segment_count, 0),
            dtype=np.float64,
        )
    visibility_distance = max(
        float(particle_filter.config.support_visibility_distance_m),
        1e-6,
    )
    supported = support_squared <= visibility_distance * visibility_distance
    minimum = max(
        1,
        int(particle_filter.config.min_segment_support_samples),
    )
    visible_segments = (
        np.count_nonzero(supported, axis=1) >= minimum
        if supported.ndim == 2 and support_squared.shape[1] > 0
        else np.zeros(particle_filter.segment_count, dtype=bool)
    )
    particle_filter._set_visibility(visible_segments)
    particle_filter.last_supported_sample_fraction = (
        float(np.mean(supported)) if supported.size else np.nan
    )
    particle_filter.last_path_support_rms_m = (
        float(np.sqrt(np.mean(support_squared))) if support_squared.size else np.nan
    )
    affinities = np.asarray(point_affinities, dtype=np.float32).reshape(-1)
    particle_filter.last_mean_support_affinity = (
        float(np.mean(affinities)) if len(affinities) else np.nan
    )
    particle_filter.last_support_point_affinities = np.ascontiguousarray(
        affinities,
        dtype=np.float32,
    )


def particle_chain_support_squared_distances(particles, points, samples_per_segment, chunk_size=32):
    particles = np.asarray(particles, dtype=np.float64)
    points = valid_points(points)
    samples_per_segment = max(1, int(samples_per_segment))
    fractions = (np.arange(samples_per_segment, dtype=np.float64) + 0.5) / samples_per_segment
    samples = (
        particles[:, :-1, None, :]
        + fractions[None, None, :, None]
        * (particles[:, 1:, None, :] - particles[:, :-1, None, :])
    ).reshape(len(particles), -1, 3)
    output = np.empty(samples.shape[:2], dtype=np.float64)
    for start in range(0, len(samples), max(1, int(chunk_size))):
        batch = samples[start:start + max(1, int(chunk_size))]
        squared = np.sum((batch[:, :, None, :] - points[None, None, :, :]) ** 2, axis=3)
        output[start:start + len(batch)] = np.min(squared, axis=2)
    return output


def particle_endpoint_tangent_penalty(
    particles,
    endpoint_tangents,
    confidence,
    minimum_confidence,
    likelihood_scale_m,
):
    particles = np.asarray(particles, dtype=np.float64)
    tangents = np.asarray(endpoint_tangents, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if (
        particles.ndim != 3
        or particles.shape[1] < 2
        or tangents.shape != (2, 3)
        or confidence.shape != (2,)
        or not np.all(np.isfinite(tangents))
        or float(likelihood_scale_m) <= 0.0
    ):
        return np.zeros(len(particles), dtype=np.float64)
    inward = np.stack((
        particles[:, 1, :] - particles[:, 0, :],
        particles[:, -2, :] - particles[:, -1, :],
    ), axis=1)
    inward = normalize_vectors(inward)
    tangents = normalize_vectors(tangents)
    dots = np.clip(np.sum(inward * tangents[None, :, :], axis=2), -1.0, 1.0)
    gate = np.clip(
        (confidence - float(minimum_confidence))
        / max(1.0 - float(minimum_confidence), 1e-6),
        0.0,
        1.0,
    )
    active = min(1.0, float(np.sum(gate)))
    mismatch = np.sum((1.0 - dots) * gate[None, :], axis=1) / max(float(np.sum(gate)), 1e-12)
    return float(likelihood_scale_m) ** 2 * active * mismatch


def particle_endpoint_tangent_penalty_torch(
    particles_t,
    endpoint_tangents_t,
    confidence_t,
    minimum_confidence,
    likelihood_scale_m,
):
    if (
        endpoint_tangents_t is None
        or confidence_t is None
        or float(likelihood_scale_m) <= 0.0
    ):
        return torch.zeros(len(particles_t), dtype=particles_t.dtype, device=particles_t.device)
    inward_t = torch.stack((
        particles_t[:, 1, :] - particles_t[:, 0, :],
        particles_t[:, -2, :] - particles_t[:, -1, :],
    ), dim=1)
    inward_t = normalize_vectors_torch(inward_t)
    tangents_t = normalize_vectors_torch(endpoint_tangents_t)
    dots_t = torch.sum(inward_t * tangents_t[None, :, :], dim=2).clamp(-1.0, 1.0)
    gate_t = torch.clamp(
        (confidence_t - float(minimum_confidence))
        / max(1.0 - float(minimum_confidence), 1e-6),
        min=0.0,
        max=1.0,
    )
    gate_sum_t = torch.sum(gate_t)
    active_t = torch.clamp(gate_sum_t, max=1.0)
    mismatch_t = torch.sum((1.0 - dots_t) * gate_t[None, :], dim=1) / gate_sum_t.clamp_min(1e-12)
    return float(likelihood_scale_m) ** 2 * active_t * mismatch_t


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
    path_support_rms = float(getattr(result, "path_support_rms_m", np.nan))
    return CableEstimate3D(
        points_xyz=np.asarray(result.points_xyz, dtype=np.float32),
        source_points=source_points,
        residual_m=(
            path_support_rms
            if np.isfinite(path_support_rms)
            else (polyline_residual(source_points, result.points_xyz) if len(source_points) else 0.0)
        ),
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


def transition_proposal_ratios(config):
    """Return the explicit local/endpoint/global proposal distribution.

    These are proposal masses, not tuning hints.  Requiring a normalized
    distribution makes each ablation interpretable and prevents a disabled
    component from silently changing the other non-local component.
    """

    ratios = np.asarray((
        float(config.local_proposal_ratio),
        float(config.endpoint_conditioned_proposal_ratio),
        float(config.global_random_particle_ratio),
    ), dtype=np.float64)
    if not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("Proposal ratios must be finite and non-negative.")
    if not np.isclose(float(np.sum(ratios)), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            "local, endpoint-conditioned, and global-random proposal ratios must sum to 1."
        )
    return tuple(float(value) for value in ratios)


def transition_population_counts(total_count, ratios):
    """Allocate an exact particle count using the largest-remainder method."""

    total_count = max(0, int(total_count))
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.shape != (3,) or not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("Exactly three finite non-negative proposal ratios are required.")
    if not np.isclose(float(np.sum(ratios)), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Proposal ratios must sum to 1 before particle allocation.")
    exact = ratios * total_count
    counts = np.floor(exact).astype(np.int64)
    remaining = total_count - int(np.sum(counts))
    if remaining:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remaining]] += 1
    return tuple(int(value) for value in counts)


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


def posterior_endpoint_tangents(nodes_xyz):
    nodes = valid_points(nodes_xyz)
    if len(nodes) < 2:
        return None
    tangents = np.stack((
        nodes[1] - nodes[0],
        nodes[-2] - nodes[-1],
    ), axis=0)
    if not np.all(np.isfinite(tangents)) or np.any(np.linalg.norm(tangents, axis=1) <= 1e-8):
        return None
    return np.ascontiguousarray(normalize_vectors(tangents), dtype=np.float32)


def estimate_endpoint_tangents(
    points_xyz,
    endpoint_nodes,
    min_radius_m=0.008,
    radius_m=0.075,
    sigma_m=0.035,
    min_points=8,
    ransac_inlier_m=0.006,
    max_hypotheses=64,
    ransac_enabled=True,
    reference_tangents=None,
    reference_weight=0.25,
):
    """Estimate each inward endpoint tangent with line consensus then PCA.

    Every nearby point supplies a one-sided direction hypothesis from the
    endpoint.  RANSAC selects a narrow cylindrical support, which prevents a
    nearby branch or the other cable from rotating the PCA scatter matrix.
    The previous posterior direction is a bounded tie-breaker, never an inlier
    gate, so a genuine fast turn can still replace it.
    """

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
    inlier_radius = max(1e-6, float(ransac_inlier_m))
    hypothesis_limit = max(2, int(max_hypotheses))
    reference = np.asarray(reference_tangents, dtype=np.float64)
    if reference.shape != (2, 3) or not np.all(np.isfinite(reference)):
        reference = np.full((2, 3), np.nan, dtype=np.float64)
    else:
        reference = normalize_vectors(reference)
    reference_prior = max(0.0, float(reference_weight))
    for endpoint_index, endpoint in enumerate((endpoints[0], endpoints[-1])):
        delta = points - endpoint[None, :]
        distance = np.linalg.norm(delta, axis=1)
        keep = (distance >= min_radius) & (distance <= radius)
        count = int(np.count_nonzero(keep))
        if count < 2:
            continue
        local_distance = distance[keep]
        unit = delta[keep] / np.maximum(local_distance[:, None], 1e-12)
        weights = np.exp(-0.5 * (local_distance / sigma) ** 2)
        if bool(ransac_enabled):
            if count > hypothesis_limit:
                hypothesis_indices = np.linspace(0, count - 1, hypothesis_limit, dtype=np.int64)
                hypotheses = unit[hypothesis_indices]
            else:
                hypotheses = unit
            alignment = np.clip(hypotheses @ unit.T, -1.0, 1.0)
            perpendicular_squared = (
                local_distance[None, :] ** 2
                * np.maximum(0.0, 1.0 - alignment * alignment)
            )
            inliers = (alignment > 0.0) & (perpendicular_squared <= inlier_radius * inlier_radius)
            consensus = np.sum(
                inliers * weights[None, :] * np.maximum(alignment, 0.0) ** 2,
                axis=1,
            ) / max(float(np.sum(weights)), 1e-12)
            if np.all(np.isfinite(reference[endpoint_index])):
                consensus += reference_prior * np.maximum(
                    hypotheses @ reference[endpoint_index],
                    0.0,
                ) ** 2
            best = int(np.argmax(consensus))
            selected = inliers[best]
            selected_alignment = np.maximum(alignment[best, selected], 0.0)
            selected_weights = weights[selected] * selected_alignment**2
        else:
            selected = np.ones(count, dtype=bool)
            selected_weights = weights
        selected_count = int(np.count_nonzero(selected))
        support_count[endpoint_index] = selected_count
        if selected_count < 2:
            continue
        selected_unit = unit[selected]
        scatter = np.einsum(
            "n,ni,nj->ij",
            selected_weights,
            selected_unit,
            selected_unit,
        ) / max(float(np.sum(selected_weights)), 1e-12)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        tangent = eigenvectors[:, -1]
        mean_direction = np.sum(selected_weights[:, None] * selected_unit, axis=0)
        orientation = mean_direction if np.linalg.norm(mean_direction) > 1e-8 else fallback[endpoint_index]
        if float(np.dot(tangent, orientation)) < 0.0:
            tangent = -tangent
        anisotropy = max(0.0, float(eigenvalues[-1] - eigenvalues[-2])) / max(float(eigenvalues[-1]), 1e-12)
        support_factor = min(1.0, selected_count / max(float(min_points), 1.0))
        consensus_factor = float(np.sum(weights[selected])) / max(float(np.sum(weights)), 1e-12)
        tangents[endpoint_index] = tangent
        confidence[endpoint_index] = anisotropy * support_factor * consensus_factor
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
    ransac_inlier_m=0.006,
    max_hypotheses=64,
    ransac_enabled=True,
    reference_tangents=None,
    reference_weight=0.25,
):
    chord_t = normalize_vectors_torch((endpoints_t[-1] - endpoints_t[0])[None, :])[0]
    fallback_t = torch.stack((chord_t, -chord_t), dim=0)
    tangents = []
    confidence = []
    support_counts = []
    min_radius = max(0.0, float(min_radius_m))
    radius = max(min_radius + 1e-6, float(radius_m))
    sigma = max(1e-6, float(sigma_m))
    inlier_radius = max(1e-6, float(ransac_inlier_m))
    hypothesis_limit = max(2, int(max_hypotheses))
    if reference_tangents is None:
        reference_t = torch.full(
            (2, 3),
            torch.nan,
            dtype=points_t.dtype,
            device=points_t.device,
        )
    else:
        reference_t = torch.as_tensor(
            reference_tangents,
            dtype=points_t.dtype,
            device=points_t.device,
        )
        if reference_t.shape != (2, 3):
            reference_t = torch.full_like(fallback_t, torch.nan)
        else:
            reference_t = normalize_vectors_torch(reference_t)
    reference_prior = max(0.0, float(reference_weight))
    for endpoint_index in range(2):
        delta_t = points_t - endpoints_t[endpoint_index][None, :]
        distance_t = torch.linalg.vector_norm(delta_t, dim=1)
        keep_t = (distance_t >= min_radius) & (distance_t <= radius)
        local_distance_t = distance_t[keep_t]
        local_delta_t = delta_t[keep_t]
        count = int(local_distance_t.numel())
        if count < 2:
            tangents.append(fallback_t[endpoint_index])
            confidence.append(torch.zeros((), dtype=points_t.dtype, device=points_t.device))
            support_counts.append(torch.zeros((), dtype=torch.int32, device=points_t.device))
            continue
        unit_t = local_delta_t / local_distance_t[:, None].clamp_min(1e-12)
        weights_t = torch.exp(-0.5 * (local_distance_t / sigma) ** 2)
        if bool(ransac_enabled):
            if count > hypothesis_limit:
                hypothesis_indices_t = torch.linspace(
                    0,
                    count - 1,
                    hypothesis_limit,
                    dtype=torch.float32,
                    device=points_t.device,
                ).round().to(torch.int64)
                hypotheses_t = unit_t.index_select(0, hypothesis_indices_t)
            else:
                hypotheses_t = unit_t
            alignment_t = (hypotheses_t @ unit_t.T).clamp(-1.0, 1.0)
            perpendicular_squared_t = (
                local_distance_t[None, :] ** 2
                * (1.0 - alignment_t * alignment_t).clamp_min(0.0)
            )
            inliers_t = (alignment_t > 0.0) & (
                perpendicular_squared_t <= inlier_radius * inlier_radius
            )
            consensus_t = torch.sum(
                inliers_t.to(points_t.dtype)
                * weights_t[None, :]
                * alignment_t.clamp_min(0.0) ** 2,
                dim=1,
            ) / weights_t.sum().clamp_min(1e-12)
            reference_valid_t = torch.all(torch.isfinite(reference_t[endpoint_index]))
            reference_bonus_t = reference_prior * (
                hypotheses_t @ torch.nan_to_num(reference_t[endpoint_index])
            ).clamp_min(0.0) ** 2
            consensus_t = consensus_t + torch.where(
                reference_valid_t,
                reference_bonus_t,
                torch.zeros_like(reference_bonus_t),
            )
            best_t = torch.argmax(consensus_t)
            selected_t = inliers_t[best_t]
            selected_alignment_t = alignment_t[best_t, selected_t].clamp_min(0.0)
            selected_weights_t = weights_t[selected_t] * selected_alignment_t**2
        else:
            selected_t = torch.ones(count, dtype=torch.bool, device=points_t.device)
            selected_weights_t = weights_t
        selected_count_t = torch.count_nonzero(selected_t)
        support_counts.append(selected_count_t.to(torch.int32))
        selected_unit_t = unit_t[selected_t]
        scatter_t = torch.einsum(
            "n,ni,nj->ij",
            selected_weights_t,
            selected_unit_t,
            selected_unit_t,
        ) / selected_weights_t.sum().clamp_min(1e-12)
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(scatter_t)
        tangent_t = eigenvectors_t[:, -1]
        mean_direction_t = torch.sum(selected_weights_t[:, None] * selected_unit_t, dim=0)
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
        support_factor_t = torch.clamp(selected_count_t.to(points_t.dtype) / max(float(min_points), 1.0), max=1.0)
        consensus_factor_t = weights_t[selected_t].sum() / weights_t.sum().clamp_min(1e-12)
        valid_t = selected_count_t >= 2
        tangents.append(torch.where(valid_t, tangent_t, fallback_t[endpoint_index]))
        confidence.append(torch.where(
            valid_t,
            anisotropy_t * support_factor_t * consensus_factor_t,
            torch.zeros_like(anisotropy_t),
        ))
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
    noise_scale=1.0,
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
    maximum_deformation = max(
        float(config.endpoint_conditioned_max_deformation_m),
        1e-6,
    )
    deformation_std = min(
        maximum_deformation,
        (
            float(config.endpoint_conditioned_deformation_std_m)
            + float(config.endpoint_conditioned_slack_gain) * max(0.0, float(slack_m))
        ) * np.sqrt(max(float(noise_scale), 0.25)),
    )
    if initialization:
        deformation_std = min(maximum_deformation, 1.25 * deformation_std)
    coefficients_t = torch.randn(
        (count, modes, 3),
        dtype=torch.float32,
        device=base_particles_t.device,
        generator=generator,
    ) * deformation_std
    offsets_t = torch.einsum("cmx,mn->cnx", coefficients_t, basis_t)
    node_tangents_t = torch.empty_like(base_particles_t)
    node_tangents_t[:, 0, :] = base_particles_t[:, 1, :] - base_particles_t[:, 0, :]
    node_tangents_t[:, -1, :] = base_particles_t[:, -1, :] - base_particles_t[:, -2, :]
    if node_count > 2:
        node_tangents_t[:, 1:-1, :] = base_particles_t[:, 2:, :] - base_particles_t[:, :-2, :]
    node_tangents_t = normalize_vectors_torch(node_tangents_t)
    offsets_t = offsets_t - torch.sum(
        offsets_t * node_tangents_t,
        dim=2,
        keepdim=True,
    ) * node_tangents_t
    offset_norm_t = torch.linalg.vector_norm(offsets_t, dim=2, keepdim=True)
    offsets_t = offsets_t * torch.clamp(
        maximum_deformation / offset_norm_t.clamp_min(1e-12),
        max=1.0,
    )
    candidate_t = base_particles_t + offsets_t
    candidate_t[:, 0, :] = endpoints_t[0]
    candidate_t[:, -1, :] = endpoints_t[-1]
    directions_t = normalize_vectors_torch(candidate_t[:, 1:, :] - candidate_t[:, :-1, :])
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
    noise_scale=1.0,
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
    maximum_deformation = max(
        float(config.endpoint_conditioned_max_deformation_m),
        1e-6,
    )
    deformation_std = min(
        maximum_deformation,
        (
            float(config.endpoint_conditioned_deformation_std_m)
            + float(config.endpoint_conditioned_slack_gain) * max(0.0, total_length - chord_distance)
        ) * np.sqrt(max(float(noise_scale), 0.25)),
    )
    if initialization:
        deformation_std = min(maximum_deformation, 1.25 * deformation_std)
    coefficients = rng.normal(0.0, deformation_std, (count, modes, 3))
    offsets = np.einsum("cmx,mn->cnx", coefficients, basis)
    node_tangents = np.empty_like(base)
    node_tangents[:, 0, :] = base[:, 1, :] - base[:, 0, :]
    node_tangents[:, -1, :] = base[:, -1, :] - base[:, -2, :]
    if node_count > 2:
        node_tangents[:, 1:-1, :] = base[:, 2:, :] - base[:, :-2, :]
    node_tangents = normalize_vectors(node_tangents)
    offsets -= np.sum(offsets * node_tangents, axis=2, keepdims=True) * node_tangents
    offset_norm = np.linalg.norm(offsets, axis=2, keepdims=True)
    offsets *= np.minimum(1.0, maximum_deformation / np.maximum(offset_norm, 1e-12))
    candidate = base + offsets
    candidate[:, 0, :] = endpoints[0]
    candidate[:, -1, :] = endpoints[-1]
    directions = normalize_vectors(candidate[:, 1:, :] - candidate[:, :-1, :])
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


def posterior_medoid_index_torch(particles_t, rank_weights_t, top_indices_t):
    """Bayes representative constrained to an actually sampled cable shape."""

    selected_t = particles_t.index_select(0, top_indices_t)
    delta_t = selected_t[:, None, :, :] - selected_t[None, :, :, :]
    distance_t = torch.mean(torch.sum(delta_t * delta_t, dim=-1), dim=2)
    top_weights_t = torch.clamp_min(rank_weights_t.index_select(0, top_indices_t), 0.0)
    top_weights_t = top_weights_t / top_weights_t.sum().clamp_min(1e-12)
    risk_t = torch.sum(distance_t * top_weights_t[None, :], dim=1)
    return int(top_indices_t[torch.argmin(risk_t)].item())


def posterior_medoid_index(particles, rank_weights, top_indices):
    particles = np.asarray(particles, dtype=np.float64)
    rank_weights = np.asarray(rank_weights, dtype=np.float64)
    top_indices = np.asarray(top_indices, dtype=np.int64)
    selected = particles[top_indices]
    delta = selected[:, None, :, :] - selected[None, :, :, :]
    distance = np.mean(np.sum(delta * delta, axis=-1), axis=2)
    top_weights = np.maximum(rank_weights[top_indices], 0.0)
    top_weights /= max(float(np.sum(top_weights)), 1e-12)
    risk = np.sum(distance * top_weights[None, :], axis=1)
    return int(top_indices[int(np.argmin(risk))])


def particle_estimate_diagnostics(
    representative_nodes,
    map_nodes,
    top_particles,
    *,
    endpoint_tangents=None,
    endpoint_tangent_confidence=None,
    endpoint_tangent_support_count=None,
    mean_support_affinity=np.nan,
    supported_sample_fraction=np.nan,
    visible_segment_fraction=np.nan,
    local_proposal_ratio=0.0,
    endpoint_conditioned_proposal_ratio=0.0,
    global_random_particle_ratio=0.0,
):
    """Describe estimator spread without changing the particle-filter state."""
    representative = np.asarray(representative_nodes, dtype=np.float32)
    map_nodes = np.asarray(map_nodes, dtype=np.float32)
    particles = np.asarray(top_particles, dtype=np.float32)
    if representative.ndim != 2 or representative.shape[1] < 3:
        return None
    representative = np.ascontiguousarray(representative[:, :3], dtype=np.float32)
    if map_nodes.shape != representative.shape or particles.ndim != 3 or particles.shape[1:] != representative.shape:
        return None
    finite_particles = np.all(np.isfinite(particles), axis=(1, 2))
    particles = np.ascontiguousarray(particles[finite_particles], dtype=np.float32)
    if len(particles) == 0 or not np.all(np.isfinite(representative)) or not np.all(np.isfinite(map_nodes)):
        return None

    particle_mean = np.mean(particles, axis=0, dtype=np.float64)
    deltas = particles.astype(np.float64) - particle_mean[None, :, :]
    squared_radius = np.sum(deltas * deltas, axis=2)
    node_rms = np.sqrt(np.mean(squared_radius, axis=0))

    covariance = np.einsum("knc,knd->ncd", deltas, deltas, optimize=True) / max(len(particles), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_sigma = np.sqrt(np.maximum(eigenvalues[:, -1], 0.0))
    principal_std = eigenvectors[:, :, -1] * principal_sigma[:, None]

    map_error = float(np.mean(np.linalg.norm(map_nodes.astype(np.float64) - representative, axis=1)))
    endpoint_delta = endpoint_direction_disagreement_deg(map_nodes, representative)
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
        representative_points_xyz=representative.copy(),
        map_points_xyz=np.ascontiguousarray(map_nodes, dtype=np.float32),
        top_particle_points_xyz=particles,
        node_rms_spread_m=np.ascontiguousarray(node_rms, dtype=np.float32),
        node_principal_std_xyz=np.ascontiguousarray(principal_std, dtype=np.float32),
        map_to_representative_node_error_m=map_error,
        mean_node_spread_m=float(np.mean(node_rms)),
        max_node_spread_m=float(np.max(node_rms)),
        endpoint_direction_delta_deg=endpoint_delta,
        endpoint_tangents_xyz=np.ascontiguousarray(tangents, dtype=np.float32),
        endpoint_tangent_confidence=np.ascontiguousarray(tangent_confidence, dtype=np.float32),
        endpoint_tangent_support_count=np.ascontiguousarray(tangent_support, dtype=np.int32),
        mean_support_affinity=float(mean_support_affinity),
        supported_sample_fraction=float(supported_sample_fraction),
        visible_segment_fraction=float(visible_segment_fraction),
        local_proposal_ratio=float(local_proposal_ratio),
        endpoint_conditioned_proposal_ratio=float(endpoint_conditioned_proposal_ratio),
        global_random_particle_ratio=float(global_random_particle_ratio),
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
