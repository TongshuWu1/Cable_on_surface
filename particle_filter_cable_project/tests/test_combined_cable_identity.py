import unittest
from types import SimpleNamespace

import numpy as np

from cable_detection import CableEstimate3D
from main import combine_cable_estimates, combine_filter_results, combine_tracking_diagnostics


class CombinedCableIdentityTests(unittest.TestCase):
    def test_missing_cable_one_does_not_renumber_cable_two(self):
        cable_two = CableEstimate3D(
            points_xyz=np.asarray(
                [[2.0, 0.0, 1.0], [2.5, 0.0, 1.0], [3.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            source_points=np.empty((0, 3), dtype=np.float32),
            residual_m=0.001,
            method="test",
            endpoint_nodes=np.asarray([[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]], dtype=np.float32),
            endpoint_marker_centers_xyz=np.asarray(
                [[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            endpoint_marker_centers_xy=np.asarray([[200.0, 100.0], [300.0, 100.0]], dtype=np.float32),
            endpoint_marker_count=2,
        )

        combined = combine_cable_estimates([None, cable_two])

        self.assertEqual(combined.pf_node_runs, ((1, 0, 2),))
        np.testing.assert_array_equal(combined.endpoint_marker_pf_ids, [1, 1])
        np.testing.assert_array_equal(combined.endpoint_marker_end_indices, [0, 1])

    def test_prediction_only_cable_does_not_dilute_active_update_ratios(self):
        common = dict(
            visible_nodes=np.ones(3, dtype=bool),
            visible_segments=np.ones(2, dtype=bool),
            lost_frames=0,
            measurement_point_count=100,
            segment_length_m=0.1,
            mean_node_speed_mps=0.0,
            stage_seconds={},
        )
        updated = SimpleNamespace(
            **common,
            measurement_used=True,
            prediction_only=False,
            endpoint_conditioned_proposal_ratio=0.25,
            global_random_particle_ratio=0.10,
            mean_support_affinity=0.45,
            supported_sample_fraction=0.70,
        )
        predicted = SimpleNamespace(
            **common,
            measurement_used=False,
            prediction_only=True,
            endpoint_conditioned_proposal_ratio=0.0,
            global_random_particle_ratio=0.0,
            mean_support_affinity=np.nan,
            supported_sample_fraction=np.nan,
        )

        combined = combine_filter_results([updated, predicted])
        diagnostics = combine_tracking_diagnostics(
            [
                {"measurement_used": True, "global_random_ratio": 0.10},
                {"measurement_used": False, "global_random_ratio": np.nan},
            ],
            candidate_count=1,
            cable_count=2,
        )

        self.assertAlmostEqual(combined.global_random_particle_ratio, 0.10)
        self.assertAlmostEqual(combined.endpoint_conditioned_proposal_ratio, 0.25)
        self.assertAlmostEqual(combined.mean_support_affinity, 0.45)
        self.assertEqual(diagnostics["active_cables"], 1)
        self.assertAlmostEqual(diagnostics["global_random_ratio"], 0.10)

    def test_particle_diagnostic_group_keeps_physical_pf_index(self):
        estimate_diagnostics = SimpleNamespace(
            map_to_representative_node_error_m=0.012,
            mean_node_spread_m=0.008,
            max_node_spread_m=0.020,
            endpoint_direction_delta_deg=np.asarray([4.0, 7.0], dtype=np.float32),
        )
        cable_two = SimpleNamespace(
            particle_diagnostics=estimate_diagnostics,
            measurement_used=True,
            prediction_only=False,
            visible_nodes=np.ones(3, dtype=bool),
            visible_segments=np.ones(2, dtype=bool),
            segment_length_m=0.1,
            mean_node_speed_mps=0.0,
            stage_seconds={},
        )

        combined = combine_filter_results([None, cable_two])

        self.assertEqual(len(combined.particle_diagnostics), 1)
        self.assertEqual(combined.particle_diagnostics[0][0], 1)
        self.assertIs(combined.particle_diagnostics[0][1], estimate_diagnostics)
        self.assertAlmostEqual(combined.map_to_representative_node_error_m, 0.012)
        self.assertAlmostEqual(combined.max_node_spread_m, 0.020)
        np.testing.assert_allclose(combined.endpoint_direction_delta_deg, [4.0, 7.0])


if __name__ == "__main__":
    unittest.main()
