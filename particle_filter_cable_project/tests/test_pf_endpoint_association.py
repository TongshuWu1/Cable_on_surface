import unittest

import numpy as np

from cable_detection import EndpointMarker3D
from cable_particle_filter import (
    PerCableEndpointAssociationConfig,
    PerCableEndpointAssociator,
)


def endpoint_group(points):
    points = np.asarray(points, dtype=np.float32)
    centers_xyz = np.column_stack([points[:, :2], np.ones(len(points), dtype=np.float32)])
    return EndpointMarker3D(
        mask=np.zeros((8, 8), dtype=np.uint8),
        centers_xyz=np.ascontiguousarray(centers_xyz, dtype=np.float32),
        centers_xy=np.ascontiguousarray(100.0 * points[:, :2], dtype=np.float32),
        endpoint_nodes=None,
        component_count=len(points),
        points_xyz=np.empty((0, 3), dtype=np.float32),
    )


class PerCableEndpointAssociatorTests(unittest.TestCase):
    def make_associator(self):
        return PerCableEndpointAssociator(
            cable_count=2,
            config=PerCableEndpointAssociationConfig(
                ambiguity_margin_m=0.01,
                support_weight=0.0,
            ),
        )

    def test_endpoint_channels_are_physical_cable_identity(self):
        associator = self.make_associator()
        result = associator.associate(
            [endpoint_group([[0.0, 0.0], [1.0, 0.0]]), endpoint_group([[3.0, 0.0], [4.0, 0.0]])],
            [None, None],
            np.empty((0, 3), dtype=np.float32),
            [1.5, 1.5],
            [0.0, 0.0],
        )

        self.assertEqual(result.diagnostics["endpoint_association_status"], "initialized")
        np.testing.assert_allclose(result.markers_by_cable[0].endpoint_nodes[:, 0], [0.0, 1.0])
        np.testing.assert_allclose(result.markers_by_cable[1].endpoint_nodes[:, 0], [3.0, 4.0])
        self.assertIn("PF1<-endpoints_1", result.diagnostics["endpoint_assignment_text"])
        self.assertIn("PF2<-endpoints_2", result.diagnostics["endpoint_assignment_text"])

    def test_component_reordering_preserves_start_end_orientation(self):
        associator = self.make_associator()
        first = associator.associate(
            [endpoint_group([[0.0, 0.0], [1.0, 0.0]]), endpoint_group([[3.0, 0.0], [4.0, 0.0]])],
            [None, None],
            np.empty((0, 3), dtype=np.float32),
            [1.5, 1.5],
            [0.0, 0.0],
        )
        references = [marker.endpoint_nodes for marker in first.markers_by_cable]
        second = associator.associate(
            [endpoint_group([[1.02, 0.0], [0.02, 0.0]]), endpoint_group([[4.02, 0.0], [3.02, 0.0]])],
            references,
            np.empty((0, 3), dtype=np.float32),
            [1.5, 1.5],
            [0.0, 0.0],
        )

        self.assertEqual(second.diagnostics["endpoint_association_status"], "stable")
        np.testing.assert_allclose(second.markers_by_cable[0].endpoint_nodes[:, 0], [0.02, 1.02], atol=1e-6)
        np.testing.assert_allclose(second.markers_by_cable[1].endpoint_nodes[:, 0], [3.02, 4.02], atol=1e-6)

    def test_endpoint_channels_are_never_swapped_between_pfs(self):
        associator = self.make_associator()
        references = [
            np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([[10.0, 0.0, 1.0], [11.0, 0.0, 1.0]], dtype=np.float32),
        ]
        result = associator.associate(
            [endpoint_group([[10.0, 0.0], [11.0, 0.0]]), endpoint_group([[0.0, 0.0], [1.0, 0.0]])],
            references,
            np.empty((0, 3), dtype=np.float32),
            [1.5, 1.5],
            [0.0, 0.0],
        )

        np.testing.assert_allclose(result.markers_by_cable[0].endpoint_nodes[:, 0], [10.0, 11.0])
        np.testing.assert_allclose(result.markers_by_cable[1].endpoint_nodes[:, 0], [0.0, 1.0])

    def test_incomplete_cable_predicts_while_other_cable_updates(self):
        associator = self.make_associator()
        references = [
            np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([[3.0, 0.0, 1.0], [4.0, 0.0, 1.0]], dtype=np.float32),
        ]
        result = associator.associate(
            [endpoint_group([[0.02, 0.0], [1.02, 0.0]]), endpoint_group([[3.02, 0.0]])],
            references,
            np.empty((0, 3), dtype=np.float32),
            [1.5, 1.5],
            [0.0, 0.0],
        )

        self.assertEqual(result.diagnostics["endpoint_association_status"], "partial")
        self.assertIsNotNone(result.markers_by_cable[0])
        self.assertIsNone(result.markers_by_cable[1])
        np.testing.assert_allclose(result.markers_by_cable[0].endpoint_nodes[:, 0], [0.02, 1.02])

    def test_reference_selects_pair_when_a_channel_has_extra_components(self):
        associator = self.make_associator()
        references = [
            np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([[0.0, 2.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float32),
        ]
        candidates = endpoint_group([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        result = associator.associate(
            [candidates, candidates],
            references,
            np.empty((0, 3), dtype=np.float32),
            [2.2, 2.2],
            [0.0, 0.0],
        )

        self.assertEqual(result.diagnostics["endpoint_association_status"], "stable")
        np.testing.assert_allclose(result.markers_by_cable[0].endpoint_nodes[:, 1], [0.0, 0.0])
        np.testing.assert_allclose(result.markers_by_cable[1].endpoint_nodes[:, 1], [2.0, 2.0])

    def test_stale_endpoint_memory_expires(self):
        associator = PerCableEndpointAssociator(
            cable_count=2,
            config=PerCableEndpointAssociationConfig(max_stale_frames=1),
        )
        associator.last_endpoint_nodes_by_cable = [np.ones((2, 3)), np.ones((2, 3))]
        associator.associate([], [None, None], None, [1.0, 1.0], [0.0, 0.0])
        self.assertTrue(all(value is not None for value in associator.last_endpoint_nodes_by_cable))
        associator.associate([], [None, None], None, [1.0, 1.0], [0.0, 0.0])
        self.assertEqual(associator.last_endpoint_nodes_by_cable, [None, None])


if __name__ == "__main__":
    unittest.main()
