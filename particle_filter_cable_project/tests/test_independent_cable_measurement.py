import unittest

import numpy as np
import torch

from cable_cuda import particle_point_distances as particle_point_distances_cuda
from cable_cuda import particle_support_distances as particle_support_distances_cuda
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    cable_measurement_update,
    estimate_endpoint_tangents,
    particle_chain_support_squared_distances,
    particle_endpoint_tangent_penalty,
    point_to_particle_segment_squared_distances,
    update_cable_particle_filters,
)
from cable_detection import CableEstimate3D


def sampled_chain_points(chain, samples_per_segment=5):
    chain = np.asarray(chain, dtype=np.float64)
    fractions = (
        (np.arange(samples_per_segment, dtype=np.float64) + 0.5)
        / samples_per_segment
    )
    return (
        chain[:-1, None, :] * (1.0 - fractions[None, :, None])
        + chain[1:, None, :] * fractions[None, :, None]
    ).reshape(-1, 3)


def initialized_cpu_filter(particles, *, weights=None, tangents=None):
    particles = np.asarray(particles, dtype=np.float64)
    config = CableParticleFilterConfig(
        particle_count=len(particles),
        segment_length_m=float(np.linalg.norm(particles[0, 1] - particles[0, 0])),
        scoring_backend="cpu",
        measurement_node_std_m=0.015,
        robust_distance_m=0.050,
        path_support_weight=1.0,
        path_support_samples_per_segment=5,
        endpoint_constraint_tolerance_m=1e-7,
        bend_penalty_m=0.0,
    )
    particle_filter = CableParticleFilter(particles.shape[1], config, seed=17)
    particle_filter.particles = particles.copy()
    particle_filter.weights = (
        np.full(len(particles), 1.0 / len(particles), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).copy()
    )
    particle_filter.last_endpoint_nodes = particles[0, [0, -1]].copy()
    particle_filter.last_endpoint_tangents = (
        np.asarray(tangents, dtype=np.float64)
        if tangents is not None
        else np.zeros((2, 3), dtype=np.float64)
    )
    particle_filter.last_endpoint_tangent_confidence = (
        np.ones(2, dtype=np.float64)
        if tangents is not None
        else np.zeros(2, dtype=np.float64)
    )
    particle_filter.initialized = True
    return particle_filter


class IndependentCableMeasurementTests(unittest.TestCase):

    def test_distant_points_from_the_other_cable_do_not_change_path_likelihood(self):
        supported = np.asarray((
            (-0.10, 0.0, 1.0),
            (0.0, 0.08, 1.0),
            (0.10, 0.0, 1.0),
        ))
        unsupported = supported.copy()
        unsupported[1, 1] = -0.08
        particles = np.stack((supported, unsupported))
        own_cloud = sampled_chain_points(supported)
        distractor = np.column_stack((
            np.linspace(-0.3, 0.3, 200),
            np.full(200, 0.4),
            np.full(200, 1.2),
        ))

        own_filter = initialized_cpu_filter(particles)
        union_filter = initialized_cpu_filter(particles)
        cable_measurement_update(own_filter, own_cloud)
        cable_measurement_update(union_filter, np.vstack((own_cloud, distractor)))

        np.testing.assert_allclose(union_filter.weights, own_filter.weights, rtol=0.0, atol=1e-12)
        self.assertGreater(float(own_filter.weights[0]), 0.98)

    def test_updating_pf1_cannot_change_pf2_state(self):
        first = np.asarray((
            (-0.10, 0.0, 1.0),
            (0.0, 0.08, 1.0),
            (0.10, 0.0, 1.0),
        ))
        second = np.asarray((
            (0.0, -0.10, 1.02),
            (0.08, 0.0, 1.02),
            (0.0, 0.10, 1.02),
        ))
        pf1 = initialized_cpu_filter(np.stack((first, first.copy())))
        pf2_particles = np.stack((second, second.copy()))
        pf2 = initialized_cpu_filter(pf2_particles, weights=(0.8, 0.2))
        particles_before = pf2.particles.copy()
        weights_before = pf2.weights.copy()

        cable_measurement_update(pf1, sampled_chain_points(first))

        np.testing.assert_array_equal(pf2.particles, particles_before)
        np.testing.assert_array_equal(pf2.weights, weights_before)

    def test_known_endpoint_gate_rejects_a_path_that_exits_on_another_branch(self):
        correct = np.asarray((
            (-0.10, 0.0, 1.0),
            (0.0, 0.08, 1.0),
            (0.10, 0.0, 1.0),
        ))
        wrong_exit = correct.copy()
        wrong_exit[-1] = (0.0, 0.10, 1.0)
        particle_filter = initialized_cpu_filter(np.stack((correct, wrong_exit)))
        cloud = np.vstack((
            sampled_chain_points(correct),
            sampled_chain_points(wrong_exit),
        ))

        cable_measurement_update(particle_filter, cloud)

        self.assertAlmostEqual(float(particle_filter.weights[0]), 1.0, places=12)
        self.assertAlmostEqual(float(particle_filter.weights[1]), 0.0, places=12)

    def test_independent_prediction_draws_are_not_index_locked(self):
        chain = np.asarray((
            (-0.10, 0.0, 1.0),
            (0.0, 0.08, 1.0),
            (0.10, 0.0, 1.0),
        ))
        population = np.repeat(chain[None, :, :], 64, axis=0)
        config = CableParticleFilterConfig(
            particle_count=64,
            segment_length_m=float(np.linalg.norm(chain[1] - chain[0])),
            scoring_backend="cpu",
            endpoint_constraint_iterations=32,
        )
        filters = [CableParticleFilter(3, config, seed=17), CableParticleFilter(3, config, seed=29)]
        for particle_filter in filters:
            particle_filter.particles = population.copy()
            particle_filter.weights = np.linspace(1.0, 2.0, 64)
            particle_filter.weights /= np.sum(particle_filter.weights)
            particle_filter.last_endpoint_nodes = chain[[0, -1]].copy()
            particle_filter.last_endpoint_tangents = np.zeros((2, 3))
            particle_filter.last_endpoint_tangent_confidence = np.zeros(2)
            particle_filter.node_velocities = np.zeros_like(population)
            particle_filter.initialized = True

        for particle_filter in filters:
            particle_filter._predict_transition(1.0 / 30.0)

        self.assertFalse(np.array_equal(filters[0].particles, filters[1].particles))
        self.assertFalse(np.array_equal(filters[0].weights, filters[1].weights))

    def test_double_crossing_geometry_is_ambiguous_without_temporal_prior(self):
        """Document the exact branch-switching limit of a point-set likelihood.

        The two paths have identical endpoints, tangent directions, total
        length, and dense cloud support. The measurement therefore must not
        invent a distinction; only their existing temporal prior separates
        them. This is the case a later graph/topology term must address.
        """

        correct = np.asarray((
            (-0.26, 0.0, 1.0),
            (-0.16, 0.0, 1.0),
            (-0.06, 0.0, 1.0),
            (0.0, 0.08, 1.0),
            (0.06, 0.0, 1.0),
            (0.16, 0.0, 1.0),
            (0.26, 0.0, 1.0),
        ))
        switched = correct.copy()
        switched[3, 1] = -0.08
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(correct, axis=0), axis=1),
            np.linalg.norm(np.diff(switched, axis=0), axis=1),
            atol=1e-12,
        )
        cloud = np.vstack((sampled_chain_points(correct), sampled_chain_points(switched)))
        prior = np.asarray((0.92, 0.08))
        particle_filter = initialized_cpu_filter(
            np.stack((correct, switched)),
            weights=prior,
            tangents=((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        )

        cable_measurement_update(particle_filter, cloud)

        np.testing.assert_allclose(particle_filter.weights, prior, rtol=0.0, atol=1e-12)

    def test_endpoint_tangent_is_a_continuous_particle_likelihood(self):
        particles = np.asarray((
            ((0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)),
            ((0.0, 0.0, 1.0), (0.0, 0.1, 1.0), (0.2, 0.0, 1.0)),
        ))
        penalty = particle_endpoint_tangent_penalty(
            particles,
            np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            np.asarray((1.0, 1.0)),
            0.20,
            0.020,
        )

        self.assertAlmostEqual(float(penalty[0]), 0.0, places=12)
        self.assertGreater(float(penalty[1]), float(penalty[0]))

    def test_endpoint_tangent_uses_local_directional_consensus(self):
        x = np.linspace(-0.20, 0.20, 160, dtype=np.float32)
        cable = np.column_stack((x, 0.02 * np.sin(3.0 * x), np.ones_like(x)))
        distractor_y = np.linspace(-0.15, 0.15, 100, dtype=np.float32)
        distractor = np.column_stack((np.zeros_like(distractor_y), distractor_y, np.ones_like(distractor_y)))
        endpoints = cable[[0, -1]]

        tangents, confidence, support = estimate_endpoint_tangents(
            np.vstack((cable, distractor)),
            endpoints,
            min_radius_m=0.005,
            radius_m=0.060,
            sigma_m=0.030,
            min_points=8,
        )

        self.assertGreater(float(confidence[0]), 0.9)
        self.assertGreater(float(confidence[1]), 0.9)
        self.assertGreaterEqual(int(np.min(support)), 8)
        self.assertGreater(float(tangents[0, 0]), 0.98)
        self.assertLess(float(tangents[1, 0]), -0.98)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for stream validation")
    def test_two_filters_keep_distinct_cuda_streams(self):
        config = CableParticleFilterConfig(
            particle_count=32,
            segment_length_m=0.05,
            scoring_backend="cuda",
            endpoint_constraint_iterations=32,
        )
        filters = [CableParticleFilter(5, config, seed=17), CableParticleFilter(5, config, seed=29)]
        stream_ids = [item.cuda_stream.cuda_stream for item in filters]
        self.assertNotEqual(stream_ids[0], stream_ids[1])
        measurements = []
        for y in (-0.04, 0.04):
            endpoints = np.asarray(((-0.10, y, 1.0), (0.10, y, 1.0)), dtype=np.float32)
            measurements.append(CableEstimate3D(
                points_xyz=np.empty((0, 3), dtype=np.float32),
                source_points=np.linspace(endpoints[0], endpoints[1], 80, dtype=np.float32),
                residual_m=0.0,
                method="stream test",
                endpoint_nodes=endpoints,
            ))

        update_cable_particle_filters(filters, measurements, dt=1.0 / 30.0)
        update_cable_particle_filters(filters, measurements, dt=1.0 / 30.0)
        for item in filters:
            item.cuda_stream.synchronize()

        self.assertEqual([item.cuda_stream.cuda_stream for item in filters], stream_ids)
        self.assertFalse(torch.equal(filters[0].particles, filters[1].particles))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for fused-kernel validation")
    def test_fused_cuda_distances_match_cpu_geometry(self):
        rng = np.random.default_rng(4)
        particles = np.cumsum(rng.normal(0.0, 0.03, (2, 16, 5, 3)), axis=2).astype(np.float32)
        points = rng.normal(0.0, 0.08, (37, 3)).astype(np.float32)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            squared_t, nearest_t = particle_point_distances_cuda(
                torch.as_tensor(particles, device="cuda"),
                torch.as_tensor(points, device="cuda"),
                stream,
            )
        stream.synchronize()
        squared = squared_t.cpu().numpy()
        nearest = nearest_t.cpu().numpy()
        for cable_index in range(2):
            cpu_all = point_to_particle_segment_squared_distances(
                points,
                particles[cable_index, :, :-1],
                particles[cable_index, :, 1:],
            )
            np.testing.assert_allclose(squared[cable_index], np.min(cpu_all, axis=1), rtol=2e-5, atol=1e-7)
            selected = np.take_along_axis(cpu_all, nearest[cable_index][:, None, :], axis=1)[:, 0, :]
            np.testing.assert_allclose(selected, np.min(cpu_all, axis=1), rtol=2e-5, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for dense-support validation")
    def test_fused_cuda_dense_support_matches_cpu_geometry(self):
        rng = np.random.default_rng(9)
        particles = np.cumsum(rng.normal(0.0, 0.03, (2, 12, 6, 3)), axis=2).astype(np.float32)
        points = rng.normal(0.0, 0.08, (41, 3)).astype(np.float32)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            squared_t = particle_support_distances_cuda(
                torch.as_tensor(particles, device="cuda"),
                torch.as_tensor(points, device="cuda"),
                3,
                stream,
            )
        stream.synchronize()
        squared = squared_t.cpu().numpy().reshape(2, 12, -1)
        for cable_index in range(2):
            expected = particle_chain_support_squared_distances(particles[cable_index], points, 3)
            np.testing.assert_allclose(squared[cable_index], expected, rtol=2e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
