import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cable_detection import CableEstimate3D, polyline_residual
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    constrain_chains_to_endpoints,
    endpoint_direction_disagreement_deg,
    filtered_cable_estimate,
    particle_estimate_diagnostics,
)


class ParticleFilterInvariantTests(unittest.TestCase):
    def test_endpoint_constraint_converges_and_preserves_lengths(self):
        rng = np.random.default_rng(12)
        count, node_count = 100, 13
        segment_length = 0.515 / 12.0
        chains = np.cumsum(rng.normal(size=(count, node_count, 3)), axis=1)
        chains[:, 0] = 0.0
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        chords = rng.uniform(0.03, 0.50, size=count)
        for index in range(count):
            endpoints = np.stack((np.zeros(3), directions[index] * chords[index]))
            result = constrain_chains_to_endpoints(
                chains[index:index + 1], endpoints, segment_length, iterations=128, tolerance_m=1e-4
            )[0]
            self.assertLessEqual(float(np.linalg.norm(result[-1] - endpoints[-1])), 1.0001e-4)
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(result, axis=0), axis=1), segment_length, atol=1e-10
            )

    def test_rejected_support_does_not_commit_candidate_endpoints(self):
        config = CableParticleFilterConfig(
            segment_length_m=0.1,
            scoring_backend="cpu",
            min_measurement_points=2,
        )
        particle_filter = CableParticleFilter(node_count=6, config=config)
        old_endpoints = np.asarray([[0.0, 0.0, 1.0], [0.4, 0.0, 1.0]], dtype=np.float32)
        candidate = np.asarray([[0.0, 0.1, 1.0], [0.4, 0.1, 1.0]], dtype=np.float32)
        particle_filter._set_endpoint_nodes(old_endpoints)
        measurement = CableEstimate3D(
            points_xyz=np.empty((0, 3), dtype=np.float32),
            source_points=np.asarray([[0.1, 0.1, 1.0], [0.2, 0.1, 1.0]], dtype=np.float32),
            residual_m=0.0,
            method="test",
            endpoint_nodes=candidate,
        )
        with patch.object(particle_filter, "_select_ransac_inlier_points", return_value=None):
            particle_filter.step(measurement)
        np.testing.assert_allclose(particle_filter.last_endpoint_nodes, old_endpoints)

    def test_estimate_is_mean_of_top_ranked_particles(self):
        config = CableParticleFilterConfig(
            segment_length_m=0.1,
            scoring_backend="cpu",
            estimate_top_particle_count=2,
        )
        particle_filter = CableParticleFilter(node_count=3, config=config)
        particle_filter.particles = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        particle_filter.weights = np.asarray([0.1, 0.9], dtype=np.float64)
        expected = np.mean(particle_filter.particles, axis=0)
        np.testing.assert_allclose(particle_filter._estimate_nodes(), expected, atol=1e-7)
        self.assertEqual(particle_filter._estimate_particle_count_cache, 2)
        self.assertAlmostEqual(particle_filter._estimate_weight_mass_cache, 1.0)
        diagnostics = particle_filter._particle_estimate_diagnostics_cache
        self.assertIsNotNone(diagnostics)
        np.testing.assert_allclose(diagnostics.average_points_xyz, expected, atol=1e-7)
        np.testing.assert_allclose(diagnostics.map_points_xyz, particle_filter.particles[1], atol=1e-7)
        np.testing.assert_allclose(diagnostics.top_particle_points_xyz, particle_filter.particles, atol=1e-7)

    def test_particle_spread_and_endpoint_direction_diagnostics(self):
        average = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        top_particles = np.stack((
            average + np.asarray([0.0, 0.1, 0.0], dtype=np.float32),
            average - np.asarray([0.0, 0.1, 0.0], dtype=np.float32),
        ))
        diagnostics = particle_estimate_diagnostics(average, top_particles[0], top_particles)

        self.assertIsNotNone(diagnostics)
        np.testing.assert_allclose(diagnostics.node_rms_spread_m, 0.1, atol=1e-7)
        np.testing.assert_allclose(
            np.linalg.norm(diagnostics.node_principal_std_xyz, axis=1),
            0.1,
            atol=1e-7,
        )
        self.assertAlmostEqual(diagnostics.map_to_average_node_error_m, 0.1, places=6)
        self.assertAlmostEqual(diagnostics.mean_node_spread_m, 0.1, places=6)
        self.assertAlmostEqual(diagnostics.max_node_spread_m, 0.1, places=6)

        perpendicular = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            endpoint_direction_disagreement_deg(average, perpendicular),
            [90.0, 90.0],
            atol=1e-5,
        )

    def test_top_particle_mean_preserves_endpoints_and_segment_lengths(self):
        segment_length = 0.1
        endpoints = np.asarray([[0.0, 0.0, 1.0], [0.25, 0.0, 1.0]], dtype=np.float64)
        seeds = np.asarray(
            [
                [[0.0, 0.0, 1.0], [0.06, 0.08, 1.0], [0.13, 0.13, 1.0], [0.21, 0.08, 1.0], [0.25, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [0.04, -0.09, 1.0], [0.12, -0.15, 1.0], [0.20, -0.09, 1.0], [0.25, 0.0, 1.0]],
            ],
            dtype=np.float64,
        )
        particles = constrain_chains_to_endpoints(
            seeds,
            endpoints,
            segment_length,
            iterations=128,
            tolerance_m=1e-5,
        )
        config = CableParticleFilterConfig(
            segment_length_m=segment_length,
            scoring_backend="cpu",
            estimate_top_particle_count=2,
            endpoint_constraint_iterations=128,
            endpoint_constraint_tolerance_m=1e-5,
        )
        particle_filter = CableParticleFilter(node_count=5, config=config)
        particle_filter.particles = particles
        particle_filter.weights = np.asarray([0.55, 0.45], dtype=np.float64)
        particle_filter._set_endpoint_nodes(endpoints)

        estimate = particle_filter._estimate_nodes()

        np.testing.assert_allclose(estimate[[0, -1]], endpoints, atol=1.0001e-5)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(estimate, axis=0), axis=1),
            segment_length,
            atol=2e-6,
        )

    def test_top_particle_mean_reduces_single_map_outlier_error(self):
        base = np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float64,
        )
        particles = np.stack((
            base + np.asarray([0.0, 0.10, 0.0]),
            base,
            base + np.asarray([0.0, -0.30, 0.0]),
        ))
        support = base + np.asarray([0.0, 0.05, 0.0])
        config = CableParticleFilterConfig(
            segment_length_m=0.1,
            scoring_backend="cpu",
            estimate_top_particle_count=2,
        )
        particle_filter = CableParticleFilter(node_count=3, config=config)
        particle_filter.particles = particles
        particle_filter.weights = np.asarray([0.55, 0.40, 0.05], dtype=np.float64)

        estimate = particle_filter._estimate_nodes()
        map_error = polyline_residual(support, particles[0])
        averaged_error = polyline_residual(support, estimate)

        self.assertLess(averaged_error, 1e-7)
        self.assertGreater(map_error, 0.049)

    def test_equal_weight_resampled_particles_are_all_averaged(self):
        base = np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float64,
        )
        particles = np.stack((
            base + np.asarray([0.0, -0.03, 0.0]),
            base,
            base + np.asarray([0.0, 0.03, 0.0]),
        ))
        config = CableParticleFilterConfig(
            segment_length_m=0.1,
            scoring_backend="cpu",
            estimate_top_particle_count=2,
        )
        particle_filter = CableParticleFilter(node_count=3, config=config)
        particle_filter.particles = particles
        particle_filter.weights = np.full(3, 1.0 / 3.0, dtype=np.float64)

        estimate = particle_filter._estimate_nodes()

        np.testing.assert_allclose(estimate, base, atol=1e-7)
        self.assertEqual(particle_filter._estimate_particle_count_cache, 3)

    def test_global_random_particles_are_fixed_ten_percent_with_healthy_ess(self):
        config = CableParticleFilterConfig(
            particle_count=100,
            segment_length_m=0.1,
            scoring_backend="cpu",
            global_random_particle_ratio=0.10,
        )
        particle_filter = CableParticleFilter(node_count=3, config=config, seed=7)
        base_chain = np.asarray(
            [[0.0, 0.0, 1.0], [0.05, 0.0866, 1.0], [0.1, 0.0, 1.0]],
            dtype=np.float64,
        )
        particle_filter.particles = np.repeat(base_chain[None, :, :], 100, axis=0)
        particle_filter.weights = np.full(100, 0.01, dtype=np.float64)
        particle_filter._set_endpoint_nodes(base_chain[[0, -1]])
        support = np.asarray(
            [[-0.05, -0.05, 0.95], [0.15, 0.15, 1.05]],
            dtype=np.float64,
        )

        particle_filter._inject_global_random_particles(support)

        self.assertAlmostEqual(particle_filter.last_global_random_particle_ratio, 0.10)
        highest_weight = float(np.max(particle_filter.weights))
        self.assertEqual(int(np.count_nonzero(np.isclose(particle_filter.weights, highest_weight))), 10)

    def test_filtered_estimate_uses_selected_pf_support_for_residual(self):
        chain = np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        selected_support = np.asarray(
            [[0.05, 0.0, 1.0], [0.15, 0.0, 1.0]],
            dtype=np.float32,
        )
        measurement = CableEstimate3D(
            points_xyz=chain,
            source_points=np.asarray([[10.0, 10.0, 10.0]], dtype=np.float32),
            residual_m=1.0,
            method="shared support",
        )
        result = SimpleNamespace(
            points_xyz=chain,
            support_points_xyz=selected_support,
            prediction_only=False,
            segment_length_m=0.1,
            effective_sample_size=100.0,
            measurement_point_count=len(selected_support),
            lost_frames=0,
        )

        estimate = filtered_cable_estimate(measurement, result)

        np.testing.assert_allclose(estimate.source_points, selected_support)
        self.assertLess(estimate.residual_m, 1e-6)
