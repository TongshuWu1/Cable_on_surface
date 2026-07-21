import unittest

import numpy as np

from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    select_union_coverage_representatives,
)

try:
    import torch
except Exception:
    torch = None


def line(start, end, count=5):
    return np.linspace(start, end, count, dtype=np.float64)


class UnionCoverageSelectionTests(unittest.TestCase):
    @staticmethod
    def _scene(backend="cpu"):
        horizontal = line((-0.10, 0.0, 1.0), (0.10, 0.0, 1.0))
        vertical = line((0.0, -0.10, 1.0), (0.0, 0.10, 1.0))
        observations = np.vstack((
            line((-0.10, 0.0, 1.0), (0.10, 0.0, 1.0), 21),
            line((0.0, -0.10, 1.0), (0.0, 0.10, 1.0), 21),
        ))
        config = CableParticleFilterConfig(
            particle_count=2,
            segment_length_m=0.05,
            scoring_backend=backend,
            measurement_node_std_m=0.025,
            robust_distance_m=0.05,
            # Keep the synthetic 10 mm grid away from the decision boundary so
            # float32 CUDA and float64 CPU test the algorithm, not 30 mm ties.
            support_visibility_distance_m=0.033,
            path_support_samples_per_segment=3,
            union_coverage_top_particle_count=2,
            union_coverage_weight=1.0,
            min_measurement_points=4,
        )
        first = CableParticleFilter(5, config=config, seed=1)
        second = CableParticleFilter(5, config=config, seed=2)
        first_particles = np.stack((horizontal, vertical))
        second_particles = horizontal[None, :, :]
        if backend == "cuda":
            first.particles = torch.as_tensor(first_particles, dtype=torch.float32, device="cuda")
            first.weights = torch.as_tensor((0.60, 0.40), dtype=torch.float32, device="cuda")
            second.particles = torch.as_tensor(second_particles, dtype=torch.float32, device="cuda")
            second.weights = torch.ones(1, dtype=torch.float32, device="cuda")
        else:
            first.particles = first_particles
            first.weights = np.asarray((0.60, 0.40), dtype=np.float64)
            second.particles = second_particles
            second.weights = np.ones(1, dtype=np.float64)
        first.initialized = True
        second.initialized = True
        return first, second, observations

    def test_union_coverage_selects_complementary_paths_not_duplicate_best_paths(self):
        first, second, observations = self._scene()
        first_weights = first.weights.copy()
        second_weights = second.weights.copy()

        selection = select_union_coverage_representatives((first, second), observations)

        self.assertEqual(selection.selected_particle_indices, (1, 0))
        self.assertEqual(selection.selected_ranks, (2, 1))
        self.assertAlmostEqual(selection.robust_coverage_rms_m, 0.0, places=7)
        self.assertAlmostEqual(selection.covered_point_fraction, 1.0, places=7)
        self.assertGreater(selection.robust_rms_gain_m, 0.0)
        self.assertGreater(selection.covered_fraction_gain, 0.0)
        np.testing.assert_array_equal(first.weights, first_weights)
        np.testing.assert_array_equal(second.weights, second_weights)
        self.assertEqual(first.last_representative_particle_index, 1)
        self.assertEqual(second.last_representative_particle_index, 0)

    def test_one_cable_mode_uses_the_same_complete_observation_score(self):
        first, _second, observations = self._scene()
        vertical_observations = observations[21:]

        selection = select_union_coverage_representatives((first,), vertical_observations)

        self.assertEqual(selection.selected_particle_indices, (1,))
        self.assertEqual(selection.selected_ranks, (2,))

    def test_selector_does_not_run_with_an_uninitialized_configured_filter(self):
        first, second, observations = self._scene()
        second.initialized = False

        selection = select_union_coverage_representatives((first, second), observations)

        self.assertIsNone(selection)
        self.assertIsNone(first.last_representative_particle_index)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_and_cpu_select_the_same_pair_and_metrics(self):
        cpu_first, cpu_second, observations = self._scene("cpu")
        gpu_first, gpu_second, _ = self._scene("cuda")

        cpu = select_union_coverage_representatives((cpu_first, cpu_second), observations)
        gpu = select_union_coverage_representatives((gpu_first, gpu_second), observations)

        self.assertEqual(gpu.selected_particle_indices, cpu.selected_particle_indices)
        self.assertEqual(gpu.selected_ranks, cpu.selected_ranks)
        self.assertAlmostEqual(gpu.robust_coverage_rms_m, cpu.robust_coverage_rms_m, places=5)
        self.assertAlmostEqual(gpu.covered_point_fraction, cpu.covered_point_fraction, places=5)
        self.assertAlmostEqual(gpu.robust_rms_gain_m, cpu.robust_rms_gain_m, places=5)
        self.assertAlmostEqual(gpu.covered_fraction_gain, cpu.covered_fraction_gain, places=5)


if __name__ == "__main__":
    unittest.main()
