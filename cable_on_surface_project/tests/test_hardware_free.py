from pathlib import Path
import unittest

import cv2
import numpy as np

from scripts.cable_detect.cable_vision import (
    clean_cable_mask,
    create_raw_mask,
    extract_centerline,
    load_hsv_ranges,
)
from scripts.cable_detect.trackdlo_spatial_pipeline import (
    DEFAULT_CONFIG_PATH,
    load_config,
    transform_points_matrix,
    transform_vertices_matrix,
)
from scripts.cable_detect.tools.cable_color_collect import (
    CAMERA_RESOLUTION_NAME,
    AppState,
    build_profile,
)
from scripts.cable_detect.trackdlo_tracker import (
    TrackDLOParams,
    TrackDLOTracker,
    cumulative_node_distances,
    geodesic_node_point_squared_distances,
    pairwise_squared_distances,
)
from scripts.cable_detect.zed_depth_gl_viewer import ZedDepthGLViewer


class HardwareFreeSmokeTest(unittest.TestCase):
    def test_config_profile_and_centerline_extract(self):
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertIn(config["resolution"], {"HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"})
        self.assertTrue(config["camera_origin_view"])
        self.assertGreater(config["trackdlo_max_iter"], 0)
        self.assertGreater(config["trackdlo_preprocess_max_iter"], 0)
        self.assertTrue(config["fast_motion_recovery"])
        self.assertGreater(config["max_prediction_step_m"], 0.035)

        hsv_ranges = load_hsv_ranges(Path(config["profile"]))
        self.assertGreaterEqual(len(hsv_ranges), 1)

        lower, upper = hsv_ranges[0]
        center_hsv = ((lower.astype(np.int16) + upper.astype(np.int16)) // 2).astype(
            np.uint8
        )

        hsv = np.zeros((64, 64, 3), dtype=np.uint8)
        hsv[14:52, 30:35] = center_hsv
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        raw_mask = create_raw_mask(bgr, hsv_ranges)
        mask, component_count, _components = clean_cable_mask(raw_mask, min_area=5)
        centerline = extract_centerline(mask, node_count=12)

        self.assertGreaterEqual(component_count, 1)
        self.assertGreaterEqual(len(centerline["nodes_xy"]), 2)

    def test_calibration_profile_metadata_uses_configured_resolution(self):
        profile = build_profile(AppState())

        self.assertEqual(profile["camera_resolution"], CAMERA_RESOLUTION_NAME)

    def test_trackdlo_initializes_from_full_observation(self):
        nodes = np.column_stack(
            [
                np.linspace(0.0, 0.59, 60),
                np.zeros(60),
                np.ones(60),
            ]
        )
        valid = np.ones(60, dtype=bool)

        tracker = TrackDLOTracker(
            60,
            TrackDLOParams(init_stable_frames=1, min_init_valid_ratio=1.0),
        )
        result = tracker.step(nodes, nodes, valid)

        self.assertTrue(result["tracker_initialized"])
        self.assertEqual(result["tracked_valid_segments"], 59)

    def test_trackdlo_fast_motion_recovery_uses_observed_centerline(self):
        nodes = np.column_stack(
            [
                np.linspace(0.0, 0.59, 60),
                np.zeros(60),
                np.ones(60),
            ]
        )
        valid = np.ones(60, dtype=bool)
        moved_nodes = nodes.copy()
        moved_nodes[:, 1] += 0.10

        tracker = TrackDLOTracker(
            60,
            TrackDLOParams(
                init_stable_frames=1,
                min_init_valid_ratio=1.0,
                visibility_threshold=0.004,
                fast_motion_recovery=True,
                fast_motion_min_observed_ratio=1.0,
                fast_motion_recovery_alpha=1.0,
                fast_motion_max_step_m=0.12,
                max_prediction_step_m=0.02,
                max_iter=2,
                preprocess_max_iter=1,
            ),
        )
        tracker.step(nodes, nodes, valid)
        result = tracker.step(moved_nodes, moved_nodes, valid)

        self.assertIn("fast-motion recovery", result["tracker_status"])
        self.assertGreater(float(np.mean(result["Y_t"][:, 1])), 0.07)

    def test_geodesic_node_point_distances_match_reference_loop(self):
        Y = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.1, 0.0],
                [0.3, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        X = np.array(
            [
                [0.08, 0.02, 0.0],
                [0.24, 0.09, 0.0],
                [0.4, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        base = np.ones((len(Y), len(X)), dtype=np.float64)

        actual = geodesic_node_point_squared_distances(Y, X, base)
        expected = self._reference_geodesic_node_point_squared_distances(Y, X, base)

        np.testing.assert_allclose(actual, expected)

    def test_camera_frame_transform_helpers(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = np.array([1.0, -2.0, 0.5], dtype=np.float32)

        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [np.nan, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        transformed = transform_points_matrix(matrix, points)
        transformed_preserved = transform_points_matrix(matrix, points, preserve_shape=True)

        self.assertEqual(transformed.shape, (1, 3))
        np.testing.assert_allclose(transformed[0], [1.0, -2.0, 1.5])
        np.testing.assert_allclose(transformed_preserved[0], [1.0, -2.0, 1.5])
        self.assertTrue(np.all(np.isnan(transformed_preserved[1])))

        vertices = np.array(
            [
                [0.0, 0.0, 1.0, 0.1, 0.2, 0.3],
                [np.nan, 0.0, 1.0, 0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        )
        transformed_vertices = transform_vertices_matrix(matrix, vertices)

        self.assertEqual(transformed_vertices.shape, (1, 6))
        np.testing.assert_allclose(transformed_vertices[0, :3], [1.0, -2.0, 1.5])
        np.testing.assert_allclose(transformed_vertices[0, 3:], [0.1, 0.2, 0.3])

    def test_viewer_cable_endpoint_state(self):
        viewer = ZedDepthGLViewer(width=900, height=600, left_panel_width=0)
        nodes = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.6],
                [0.2, 0.0, 0.7],
            ],
            dtype=np.float32,
        )
        valid = np.array([True, True, True])

        viewer.update_cable(np.empty((0, 3), dtype=np.float32), nodes, valid, coordinate_frame="camera")
        viewer._consume_pending_vertices()

        start_idx, end_idx = viewer._valid_endpoint_indices()
        start_label_pos = viewer._endpoint_label_position(nodes[start_idx], 1.0)

        self.assertEqual(viewer.coordinate_frame, "camera")
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 2)
        self.assertEqual(viewer._format_xyz(nodes[start_idx]), "(+0.000,+0.000,+0.500)")
        self.assertEqual(start_label_pos.shape, (3,))
        self.assertNotEqual(
            viewer._node_color(start_idx, start_idx, end_idx),
            viewer._node_color(1, start_idx, end_idx),
        )

    @staticmethod
    def _reference_geodesic_node_point_squared_distances(Y, X, base):
        node_count, point_count = base.shape
        geodesic = cumulative_node_distances(Y)
        geodesic_between_nodes = np.abs(geodesic[:, None] - geodesic[None, :])
        euclidean_distances = np.sqrt(pairwise_squared_distances(Y, X))
        output = euclidean_distances * euclidean_distances

        for point_idx in range(point_count):
            finite = np.isfinite(euclidean_distances[:, point_idx])
            if np.count_nonzero(finite) < 2:
                continue

            candidates = np.flatnonzero(finite)
            local_order = np.argsort(euclidean_distances[candidates, point_idx])
            c1 = int(candidates[local_order[0]])
            c2 = int(candidates[local_order[1]])
            d1 = float(euclidean_distances[c1, point_idx])
            d2 = float(euclidean_distances[c2, point_idx])

            to_c1 = geodesic_between_nodes[:, c1]
            to_c2 = geodesic_between_nodes[:, c2]
            distances = np.where(to_c1 <= to_c2, d1 + to_c1, d2 + to_c2)
            distances[c1] = d1
            distances[c2] = d2
            output[:, point_idx] = distances * distances

        return output


if __name__ == "__main__":
    unittest.main()
