import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cable_detection import CableObservation3D
from main import (
    draw_crossing_observations,
    format_tracking_diagnostics,
    update_viewer_cable,
)
from zed_split_viewer import (
    CABLE_SAMPLE_COLOR,
    REJECTED_CABLE_SAMPLE_COLOR,
    ZedDepthGLViewer,
)
from zed_spatial import live_point_cloud_to_vertices


class ParticleDiagnosticsUiTests(unittest.TestCase):
    def test_scene_cloud_preserves_original_zed_rgb(self):
        packed_colors = np.asarray((0x001E140A, 0x003C3228), dtype=np.uint32).view(np.float32)
        point_data = np.empty((1, 2, 4), dtype=np.float32)
        point_data[0, :, :3] = ((0.0, 0.0, 0.5), (0.1, 0.0, 0.5))
        point_data[0, :, 3] = packed_colors
        cloud = SimpleNamespace(get_data=lambda: point_data)

        vertices = live_point_cloud_to_vertices(
            cloud,
            stride=1,
            depth_min=0.01,
            depth_max=1.0,
        )

        np.testing.assert_allclose(
            vertices[:, 3:],
            np.asarray(((10, 20, 30), (40, 50, 60)), dtype=np.float32) / 255.0,
        )

    def test_observation_classes_survive_thread_safe_viewer_handoff(self):
        viewer = ZedDepthGLViewer(640, 480)
        points = np.asarray(((0.0, 0.0, 1.0), (0.1, 0.0, 1.0)), dtype=np.float32)
        rejected = np.asarray(((0.2, 0.0, 1.0),), dtype=np.float32)

        viewer.update_cable(
            points,
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
            rejected_cable_points=rejected,
        )
        viewer._consume_pending_vertices()

        np.testing.assert_allclose(viewer.cable_points, points)
        np.testing.assert_allclose(viewer.rejected_cable_points, rejected)
        np.testing.assert_allclose(CABLE_SAMPLE_COLOR, (1.0, 0.58, 0.08))
        np.testing.assert_allclose(REJECTED_CABLE_SAMPLE_COLOR, (1.0, 0.08, 0.08))

    def test_crossing_points_survive_the_thread_safe_viewer_handoff(self):
        viewer = ZedDepthGLViewer(640, 480)
        crossing_points = np.asarray(
            [
                [0.10, 0.20, 0.80],
                [0.11, 0.21, 0.81],
                [np.nan, 0.22, 0.82],
            ],
            dtype=np.float32,
        )

        viewer.update_cable(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
            crossing_points=crossing_points,
            crossing_proposal_count=2,
        )
        viewer._consume_pending_vertices()

        np.testing.assert_allclose(viewer.crossing_points, crossing_points[:2])
        self.assertEqual(viewer.crossing_proposal_count, 2)

    def test_raw_observation_remains_visible_without_a_pf_estimate(self):
        viewer = ZedDepthGLViewer(640, 480)
        accepted = np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32)
        rejected = np.asarray(((0.2, 0.0, 1.0),), dtype=np.float32)
        measurement = SimpleNamespace(
            observation=CableObservation3D(
                accepted_points_xyz=accepted,
                rejected_points_xyz=rejected,
            )
        )

        update_viewer_cable(
            viewer,
            measurement,
            estimate=None,
            filter_result=None,
            max_points=100,
        )
        viewer._consume_pending_vertices()

        np.testing.assert_allclose(viewer.cable_points, accepted)
        np.testing.assert_allclose(viewer.rejected_cable_points, rejected)
        self.assertEqual(len(viewer.cable_nodes), 0)

    def test_moving_pf_nodes_cannot_reclassify_or_recolor_observations(self):
        self.assertNotIn(
            "cable_point_colors",
            inspect.signature(ZedDepthGLViewer.update_cable).parameters,
        )
        accepted = np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32)
        rejected = np.asarray(((0.2, 0.0, 1.0),), dtype=np.float32)
        measurement = SimpleNamespace(
            observation=CableObservation3D(
                accepted_points_xyz=accepted,
                rejected_points_xyz=rejected,
            )
        )
        viewer = ZedDepthGLViewer(640, 480)
        for offset in (0.0, 5.0):
            estimate = SimpleNamespace(
                points_xyz=np.asarray(
                    ((offset, 0.0, 1.0), (offset + 0.1, 0.0, 1.0)),
                    dtype=np.float32,
                ),
                pf_node_runs=(),
            )
            update_viewer_cable(viewer, measurement, estimate, None, 100)
            viewer._consume_pending_vertices()
            np.testing.assert_allclose(viewer.cable_points, accepted)
            np.testing.assert_allclose(viewer.rejected_cable_points, rejected)

    def test_crossing_diagnostics_report_only_image_likelihood_state(self):
        text = format_tracking_diagnostics({
            "crossing_proposal_count": 1,
            "crossing_axis_count": 1,
            "crossing_target_count": 2,
            "crossing_reward": 0.8,
        })
        self.assertIn("cross=1 axes=1 targets=2", text)
        self.assertIn("R=0.80", text)
        self.assertNotIn("contact", text)

    def test_rgb_panel_does_not_draw_rejected_candidate_segments(self):
        proposal = SimpleNamespace(
            proposal_id=7,
            bbox_xywh=(10, 12, 8, 9),
            centroid_xy=np.asarray([14.0, 16.0], dtype=np.float32),
            mean_probability=0.99,
        )
        panel = np.zeros((80, 100, 3), dtype=np.uint8)
        with patch("main.cv2.line") as draw_line, patch("main.cv2.circle") as draw_circle:
            draw_crossing_observations(panel, None, (proposal,), ())
        draw_line.assert_not_called()
        draw_circle.assert_not_called()

    def test_rgb_crossing_overlay_bounds_extreme_diagnostic_axes(self):
        proposal = SimpleNamespace(
            proposal_id=7,
            bbox_xywh=(10, 12, 8, 9),
            centroid_xy=np.asarray([14.0, 16.0], dtype=np.float32),
            axes_xy=np.asarray([[1e30, 0.0], [0.0, 1e30]], dtype=np.float64),
            mean_probability=0.99,
        )
        target = SimpleNamespace(
            proposal_id=7,
            axis_xy=np.asarray([1e30, 0.0], dtype=np.float64),
        )
        panel = np.zeros((80, 100, 3), dtype=np.uint8)

        with np.errstate(all="raise"):
            draw_crossing_observations(panel, None, (proposal,), ((target,),))

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
            representative_points_xyz=average,
            map_points_xyz=particles[0],
            top_particle_points_xyz=particles,
            node_principal_std_xyz=np.tile([0.0, 0.01, 0.0], (3, 1)),
            map_to_representative_node_error_m=0.01,
            mean_node_spread_m=0.01,
            max_node_spread_m=0.01,
            endpoint_direction_delta_deg=np.asarray([2.0, 3.0], dtype=np.float32),
            endpoint_tangents_xyz=np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
            endpoint_tangent_confidence=np.asarray([0.8, 0.9], dtype=np.float32),
            endpoint_tangent_support_count=np.asarray([12, 14], dtype=np.int32),
            mean_support_affinity=0.46,
            supported_sample_fraction=0.72,
            visible_segment_fraction=0.75,
            endpoint_conditioned_proposal_ratio=0.25,
        )

        prepared = ZedDepthGLViewer._prepare_particle_diagnostics(((1, diagnostics),))

        self.assertEqual(len(prepared), 1)
        group = prepared[0]
        self.assertEqual(group["cable_id"], 1)
        self.assertEqual(group["top_particle_count"], 2)
        self.assertEqual(group["top_line_vertices"].shape, (8, 3))
        self.assertEqual(group["map_line_vertices"].shape, (4, 3))
        self.assertEqual(group["spread_line_vertices"].shape, (6, 3))
        self.assertEqual(group["tangent_line_vertices"].shape, (4, 3))
        self.assertIn("PF2 TOP=2", group["label"])
        self.assertIn("MAP-MED=10.0mm", group["label"])
        self.assertIn("support=0.46", group["label"])


if __name__ == "__main__":
    unittest.main()
