import unittest
from types import SimpleNamespace

import numpy as np

from zed_split_viewer import ZedDepthGLViewer


class ParticleDiagnosticsUiTests(unittest.TestCase):
    def test_particle_diagnostics_are_prepared_as_batched_line_arrays(self):
        average = np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        particles = np.stack((
            average + np.asarray([0.0, 0.01, 0.0], dtype=np.float32),
            average - np.asarray([0.0, 0.01, 0.0], dtype=np.float32),
        ))
        diagnostics = SimpleNamespace(
            average_points_xyz=average,
            map_points_xyz=particles[0],
            top_particle_points_xyz=particles,
            node_principal_std_xyz=np.tile([0.0, 0.01, 0.0], (3, 1)),
            map_to_average_node_error_m=0.01,
            mean_node_spread_m=0.01,
            max_node_spread_m=0.01,
            endpoint_direction_delta_deg=np.asarray([2.0, 3.0], dtype=np.float32),
        )

        prepared = ZedDepthGLViewer._prepare_particle_diagnostics(((1, diagnostics),))

        self.assertEqual(len(prepared), 1)
        group = prepared[0]
        self.assertEqual(group["cable_id"], 1)
        self.assertEqual(group["top_particle_count"], 2)
        self.assertEqual(group["top_line_vertices"].shape, (8, 3))
        self.assertEqual(group["map_line_vertices"].shape, (4, 3))
        self.assertEqual(group["spread_line_vertices"].shape, (6, 3))
        self.assertIn("PF2 TOP=2", group["label"])
        self.assertIn("MAP-AVG=10.0mm", group["label"])


if __name__ == "__main__":
    unittest.main()
