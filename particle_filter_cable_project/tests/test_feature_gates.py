import sys
import unittest
from unittest.mock import patch

import numpy as np

from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    cable_measurement_update,
    transition_population_counts,
    transition_proposal_ratios,
)
from main import (
    feature_gate_status,
    make_particle_filter_config,
    parse_args,
)
from pf_ablation import (
    RuntimeFeatureController,
    feature_state_from_args,
    isolated_feature_state,
    minimal_baseline_state,
)


class FeatureGateTests(unittest.TestCase):
    @staticmethod
    def parse(*arguments):
        with patch.object(sys, "argv", ["main.py", *arguments]):
            return parse_args()

    def test_default_config_exposes_intentional_on_and_off_gates(self):
        args = self.parse()
        status = feature_gate_status(args)
        self.assertTrue(args.particle_filter)
        self.assertTrue(args.crossing_proposals)
        self.assertTrue(args.crossing_likelihood)
        self.assertTrue(args.pf_global_random_particles)
        self.assertFalse(args.pf_bend_regularization)
        self.assertFalse(args.pf_direction_smoothing)
        self.assertTrue(args.pf_dense_path_support)
        self.assertTrue(args.pf_union_coverage)
        self.assertIn("path-support=1", status)
        self.assertIn("union=1", status)

    def test_one_cable_mode_keeps_the_two_head_neural_schema(self):
        args = self.parse(
            "--cable-count", "1",
            "--cable-lengths", "0.515",
            "--endpoint-marker-tape-lengths", "0.03",
        )
        self.assertEqual(args.cable_count, 1)
        self.assertEqual(len(args.neural_detector_endpoint_thresholds), 2)
        self.assertEqual(args.cable_lengths_m, [0.515])

    def test_algorithmic_features_map_to_real_work(self):
        args = self.parse(
            "--no-endpoint-association-support",
            "--no-pf-velocity",
            "--no-pf-adaptive-motion",
            "--no-pf-occlusion-prediction",
            "--no-pf-posterior-medoid",
            "--no-pf-endpoint-tangent",
            "--no-pf-endpoint-tangent-ransac",
            "--no-pf-endpoint-tangent-likelihood",
            "--no-pf-conditioned-proposals",
            "--no-pf-global-random-particles",
            "--no-pf-robust-measurement",
            "--no-pf-dense-path-support",
            "--no-pf-union-coverage",
            "--no-point-support-coloring",
            "--no-particle-diagnostics-overlay",
        )
        config = make_particle_filter_config(args, cable_index=0)
        self.assertFalse(config.velocity_enabled)
        self.assertFalse(config.adaptive_motion_noise_enabled)
        self.assertEqual(config.max_prediction_frames, 0)
        self.assertEqual(config.estimate_top_particle_count, 1)
        self.assertFalse(config.endpoint_tangent_estimation_enabled)
        self.assertFalse(config.endpoint_tangent_ransac_enabled)
        self.assertEqual(config.endpoint_tangent_likelihood_scale_m, 0.0)
        self.assertEqual(config.local_proposal_ratio, 1.0)
        self.assertEqual(config.endpoint_conditioned_proposal_ratio, 0.0)
        self.assertEqual(config.global_random_particle_ratio, 0.0)
        self.assertFalse(config.robust_measurement_enabled)
        self.assertEqual(config.path_support_weight, 0.0)
        self.assertEqual(config.union_coverage_weight, 0.0)
        self.assertFalse(config.point_support_diagnostics_enabled)
        self.assertFalse(config.particle_diagnostics_enabled)

    def test_disabling_one_population_transfers_only_its_mass_to_local(self):
        no_endpoint = make_particle_filter_config(
            self.parse("--no-pf-conditioned-proposals"),
            cable_index=0,
        )
        self.assertAlmostEqual(no_endpoint.local_proposal_ratio, 0.90)
        self.assertAlmostEqual(no_endpoint.endpoint_conditioned_proposal_ratio, 0.0)
        self.assertAlmostEqual(no_endpoint.global_random_particle_ratio, 0.10)

        no_random = make_particle_filter_config(
            self.parse("--no-pf-global-random-particles"),
            cable_index=0,
        )
        self.assertAlmostEqual(no_random.local_proposal_ratio, 0.75)
        self.assertAlmostEqual(no_random.endpoint_conditioned_proposal_ratio, 0.25)
        self.assertAlmostEqual(no_random.global_random_particle_ratio, 0.0)

    def test_exact_population_allocator_preserves_count_and_mass_contract(self):
        config = CableParticleFilterConfig()
        ratios = transition_proposal_ratios(config)
        counts = transition_population_counts(803, ratios)
        self.assertEqual(sum(counts), 803)
        self.assertEqual(counts, (522, 201, 80))

    def test_invalid_dependencies_are_rejected_instead_of_silently_disabled(self):
        with self.assertRaisesRegex(ValueError, "crossing_likelihood"):
            self.parse("--no-crossing-proposals")
        with self.assertRaisesRegex(ValueError, "endpoint_tangent_likelihood"):
            self.parse("--no-pf-endpoint-tangent")

    def test_crossing_diagnostics_can_be_explicitly_ablated(self):
        args = self.parse(
            "--no-crossing-proposals",
            "--no-crossing-likelihood",
        )
        self.assertFalse(args.crossing_proposals)
        self.assertFalse(args.crossing_likelihood)

    def test_isolated_preset_lists_required_parent_algorithms(self):
        args = self.parse()
        initial = feature_state_from_args(args)
        state = isolated_feature_state(initial, "pf_endpoint_tangent_likelihood")
        self.assertTrue(state["particle_filter"])
        self.assertTrue(state["pf_endpoint_tangent"])
        self.assertTrue(state["pf_endpoint_tangent_likelihood"])
        self.assertFalse(state["pf_conditioned_proposals"])
        controller = RuntimeFeatureController(initial)
        revision = controller.apply(state)
        self.assertEqual(revision, 1)
        self.assertEqual(controller.snapshot_for_revision(1)["features"], state)

    def test_minimal_baseline_keeps_only_model_and_visual_diagnostics(self):
        state = minimal_baseline_state(feature_state_from_args(self.parse()))
        self.assertTrue(state["particle_filter"])
        self.assertTrue(state["particle_diagnostics_overlay"])
        self.assertFalse(state["pf_velocity"])
        self.assertFalse(state["crossing_proposals"])

    def test_disabled_tangent_estimator_uses_zero_confidence_chord(self):
        config = CableParticleFilterConfig(
            segment_length_m=0.1,
            scoring_backend="cpu",
            endpoint_tangent_estimation_enabled=False,
        )
        particle_filter = CableParticleFilter(3, config)
        endpoints = np.asarray(((0.0, 0.0, 1.0), (0.2, 0.0, 1.0)))
        distractor = np.asarray(((0.0, 0.05, 1.0), (0.0, 0.10, 1.0)))
        particle_filter._estimate_endpoint_tangents(distractor, endpoints)
        np.testing.assert_allclose(
            particle_filter.last_endpoint_tangents,
            ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            particle_filter.last_endpoint_tangent_confidence,
            np.zeros(2),
        )

    def test_disabled_support_affinity_storage_avoids_point_vector(self):
        config = CableParticleFilterConfig(
            particle_count=8,
            segment_length_m=0.1,
            scoring_backend="cpu",
            path_support_weight=0.0,
            point_support_diagnostics_enabled=False,
        )
        chain = np.asarray(((-0.1, 0.0, 1.0), (0.0, 0.0, 1.0), (0.1, 0.0, 1.0)))
        particle_filter = CableParticleFilter(3, config, seed=1)
        particle_filter.particles = np.repeat(chain[None, :, :], 8, axis=0)
        particle_filter.weights = np.full(8, 1.0 / 8.0)
        particle_filter.last_endpoint_nodes = chain[[0, -1]].copy()
        particle_filter.initialized = True
        cable_measurement_update(particle_filter, chain)
        self.assertEqual(particle_filter.last_support_point_affinities.size, 0)

    def test_disabled_particle_overlay_skips_diagnostic_payload(self):
        config = CableParticleFilterConfig(
            particle_count=8,
            estimate_top_particle_count=4,
            segment_length_m=0.1,
            scoring_backend="cpu",
            particle_diagnostics_enabled=False,
        )
        particle_filter = CableParticleFilter(3, config)
        chain = np.asarray(((-0.1, 0.0, 1.0), (0.0, 0.0, 1.0), (0.1, 0.0, 1.0)))
        particle_filter.particles = np.repeat(chain[None, :, :], 8, axis=0)
        particle_filter.weights = np.full(8, 1.0 / 8.0)
        result = particle_filter._estimate(measurement_used=True, prediction_only=False)
        self.assertIsNone(result.particle_diagnostics)


if __name__ == "__main__":
    unittest.main()
