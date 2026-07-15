import unittest

import numpy as np
import torch

from cable_cuda import particle_point_distances as particle_point_distances_cuda
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    estimate_endpoint_tangents,
    point_to_particle_segment_squared_distances,
    posterior_consensus_measurement_update,
)


class PosteriorConsensusTests(unittest.TestCase):
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
            selected_cpu_distance = np.take_along_axis(
                cpu_all,
                nearest[cable_index][:, None, :],
                axis=1,
            )[:, 0, :]
            np.testing.assert_allclose(
                selected_cpu_distance,
                np.min(cpu_all, axis=1),
                rtol=2e-5,
                atol=1e-7,
            )

    def test_endpoint_tangent_uses_local_directional_consensus(self):
        x = np.linspace(-0.20, 0.20, 160, dtype=np.float32)
        cable = np.column_stack((x, 0.02 * np.sin(3.0 * x), np.ones_like(x)))
        distractor_y = np.linspace(-0.15, 0.15, 100, dtype=np.float32)
        distractor = np.column_stack((
            np.zeros_like(distractor_y),
            distractor_y,
            np.ones_like(distractor_y),
        ))
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

    def test_two_cable_consensus_partitions_shared_support_and_rejects_outliers(self):
        config = CableParticleFilterConfig(
            particle_count=64,
            segment_length_m=0.10,
            scoring_backend="cpu",
            measurement_node_std_m=0.015,
            robust_distance_m=0.04,
            coverage_penalty_m=0.0,
        )
        horizontal = np.asarray(
            [[-0.10, 0.0, 1.0], [0.0, 0.0, 1.0], [0.10, 0.0, 1.0]],
            dtype=np.float64,
        )
        vertical = np.asarray(
            [[0.0, -0.10, 1.01], [0.0, 0.0, 1.01], [0.0, 0.10, 1.01]],
            dtype=np.float64,
        )
        filters = [CableParticleFilter(3, config, seed=17), CableParticleFilter(3, config, seed=29)]
        for particle_filter, chain in zip(filters, (horizontal, vertical)):
            particle_filter.particles = np.repeat(chain[None, :, :], 64, axis=0)
            particle_filter.weights = np.full(64, 1.0 / 64.0, dtype=np.float64)
            particle_filter.last_endpoint_nodes = chain[[0, -1]].copy()
            particle_filter.initialized = True

        line = np.linspace(-0.10, 0.10, 80, dtype=np.float64)
        horizontal_points = np.column_stack((line, np.zeros_like(line), np.ones_like(line)))
        vertical_points = np.column_stack((np.zeros_like(line), line, np.full_like(line, 1.01)))
        outliers = np.column_stack((
            np.linspace(0.25, 0.35, 40),
            np.linspace(0.25, 0.35, 40),
            np.full(40, 1.25),
        ))
        posterior_consensus_measurement_update(
            filters,
            np.vstack((horizontal_points, vertical_points, outliers)),
        )

        for particle_filter in filters:
            self.assertGreater(particle_filter.last_ownership_effective_point_count, 65.0)
            self.assertLess(particle_filter.last_ownership_effective_point_count, 95.0)
            self.assertGreater(particle_filter.last_mean_ownership_responsibility, 0.32)
            self.assertLess(particle_filter.last_mean_ownership_responsibility, 0.48)
            self.assertTrue(np.all(np.isfinite(particle_filter.weights)))
            self.assertAlmostEqual(float(np.sum(particle_filter.weights)), 1.0)

    def test_prediction_only_cable_participates_in_ownership_without_weight_update(self):
        config = CableParticleFilterConfig(
            particle_count=32,
            segment_length_m=0.10,
            scoring_backend="cpu",
            measurement_node_std_m=0.015,
            robust_distance_m=0.04,
            coverage_penalty_m=0.0,
        )
        chains = (
            np.asarray([[-0.10, 0.0, 1.0], [0.0, 0.0, 1.0], [0.10, 0.0, 1.0]]),
            np.asarray([[0.0, -0.10, 1.01], [0.0, 0.0, 1.01], [0.0, 0.10, 1.01]]),
        )
        filters = [CableParticleFilter(3, config, seed=3), CableParticleFilter(3, config, seed=5)]
        for particle_filter, chain in zip(filters, chains):
            particle_filter.particles = np.repeat(chain[None, :, :], 32, axis=0)
            particle_filter.last_endpoint_nodes = chain[[0, -1]].copy()
            particle_filter.initialized = True
        filters[0].weights = np.full(32, 1.0 / 32.0, dtype=np.float64)
        filters[1].weights = np.linspace(1.0, 2.0, 32, dtype=np.float64)
        filters[1].weights /= np.sum(filters[1].weights)
        prediction_only_weights = filters[1].weights.copy()

        line = np.linspace(-0.10, 0.10, 60)
        points = np.vstack((
            np.column_stack((line, np.zeros_like(line), np.ones_like(line))),
            np.column_stack((np.zeros_like(line), line, np.full_like(line, 1.01))),
        ))
        posterior_consensus_measurement_update(filters, points, update_mask=[True, False])

        self.assertGreater(filters[0].last_ownership_effective_point_count, 45.0)
        self.assertLess(filters[0].last_ownership_effective_point_count, 75.0)
        np.testing.assert_array_equal(filters[1].weights, prediction_only_weights)
        self.assertTrue(np.isnan(filters[1].last_mean_ownership_responsibility))


if __name__ == "__main__":
    unittest.main()
