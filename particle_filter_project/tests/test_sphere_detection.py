import unittest

import cv2
import numpy as np

from sphere_detection import (
    SphereDetection,
    SphereDetector,
    TablePlane,
    estimate_known_radius_from_depth_cap,
    estimate_metric_radius_from_stereo,
    masked_point_cloud_points,
    masked_surface_patches,
    remove_table_background_points,
    segment_scene_from_zed_point_cloud,
)


class SphereDetectionGeometryTest(unittest.TestCase):
    def test_partial_blob_detector_recovers_occluded_color_patch_near_prediction(self):
        detector = SphereDetector(
            {
                "hsv_ranges": [],
                "blob_filter": {
                    "min_area": 250,
                    "min_circularity": 0.85,
                    "max_aspect_ratio": 1.4,
                },
            }
        )
        mask = np.zeros((100, 120), dtype=np.uint8)
        cv2.rectangle(mask, (58, 42), (78, 58), 255, -1)

        detection = detector.detect_partial_blob(
            mask,
            predicted_center_xy=(60.0, 50.0),
            predicted_radius_px=22.0,
            min_area_ratio=0.04,
        )

        self.assertIsNotNone(detection)
        self.assertLess(np.linalg.norm(np.asarray(detection.center_xy) - np.array([60.0, 50.0])), 8.0)
        self.assertAlmostEqual(detection.radius_px, 22.0, places=6)

    def test_known_radius_depth_cap_recovers_lifted_center(self):
        rng = np.random.default_rng(9)
        radius = 0.04
        center = np.array([0.08, 0.13, 0.85], dtype=np.float64)

        directions = rng.normal(0.0, 1.0, (900, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        surface = center + radius * directions
        visible = surface[np.linalg.norm(surface, axis=1) < np.linalg.norm(center, axis=0)]
        # Keep a front-facing partial cap like the ZED usually returns.
        ray_dirs = visible / np.linalg.norm(visible, axis=1)[:, None]
        normals = (visible - center) / radius
        visible = visible[np.sum(normals * ray_dirs, axis=1) < -0.55]
        visible = visible[:250] + rng.normal(0.0, 0.0015, (min(250, len(visible)), 3))

        intrinsics = {"fx": 318.0, "fy": 318.0, "cx": 640.0, "cy": 360.0}
        detection = SphereDetection(
            center_xy=(
                intrinsics["cx"] + intrinsics["fx"] * center[0] / center[2],
                intrinsics["cy"] - intrinsics["fy"] * center[1] / center[2],
            ),
            radius_px=intrinsics["fx"] * radius / center[2],
            area=2400.0,
            circularity=0.9,
            fill_ratio=0.85,
            aspect=1.0,
            contour=np.empty((0, 1, 2), dtype=np.int32),
        )

        estimate = estimate_known_radius_from_depth_cap(
            visible,
            radius,
            detection=detection,
            camera_intrinsics=intrinsics,
        )

        self.assertIsNotNone(estimate)
        self.assertLess(np.linalg.norm(estimate.center_xyz - center), 0.025)
        self.assertEqual(estimate.method, "known-radius depth-cap sphere")

    def test_known_radius_top_cap_recovers_center_from_rgb_ray(self):
        rng = np.random.default_rng(17)
        radius = 0.04
        center = np.array([0.06, 0.16, 0.88], dtype=np.float64)

        directions = rng.normal(0.0, 1.0, (900, 3))
        directions[:, 1] = np.abs(directions[:, 1]) + 2.3
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        top_cap = center + radius * directions
        top_cap = top_cap[:280] + rng.normal(0.0, 0.001, (280, 3))

        intrinsics = {"fx": 320.0, "fy": 320.0, "cx": 640.0, "cy": 360.0}
        detection = SphereDetection(
            center_xy=(
                intrinsics["cx"] + intrinsics["fx"] * center[0] / center[2],
                intrinsics["cy"] - intrinsics["fy"] * center[1] / center[2],
            ),
            radius_px=intrinsics["fx"] * radius / center[2],
            area=2400.0,
            circularity=0.9,
            fill_ratio=0.85,
            aspect=1.0,
            contour=np.empty((0, 1, 2), dtype=np.int32),
        )

        estimate = estimate_known_radius_from_depth_cap(
            top_cap,
            radius,
            detection=detection,
            camera_intrinsics=intrinsics,
        )

        self.assertIsNotNone(estimate)
        self.assertLess(np.linalg.norm(estimate.center_xyz - center), 0.018)
        self.assertLess(estimate.residual_m, 0.004)

    def test_stereo_metric_radius_recovers_real_world_size(self):
        rng = np.random.default_rng(14)
        radius = 0.04
        center = np.array([0.06, 0.10, 0.90], dtype=np.float64)

        directions = rng.normal(0.0, 1.0, (1200, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        surface = center + radius * directions
        ray_dirs = surface / np.linalg.norm(surface, axis=1)[:, None]
        normals = (surface - center) / radius
        visible = surface[np.sum(normals * ray_dirs, axis=1) < -0.30]
        visible = visible[:320] + rng.normal(0.0, 0.001, (min(320, len(visible)), 3))

        intrinsics = {"fx": 320.0, "fy": 320.0, "cx": 640.0, "cy": 360.0}
        radius_px = intrinsics["fx"] * radius / np.sqrt(center[2] * center[2] - radius * radius)
        detection = SphereDetection(
            center_xy=(
                intrinsics["cx"] + intrinsics["fx"] * center[0] / center[2],
                intrinsics["cy"] - intrinsics["fy"] * center[1] / center[2],
            ),
            radius_px=radius_px,
            area=2400.0,
            circularity=0.9,
            fill_ratio=0.85,
            aspect=1.0,
            contour=np.empty((0, 1, 2), dtype=np.int32),
        )

        estimated_radius = estimate_metric_radius_from_stereo(visible, detection, intrinsics)

        self.assertGreater(estimated_radius, 0.0)
        self.assertLess(abs(estimated_radius - radius), 0.008)

    def test_remove_table_background_points_rejects_flat_plane_leakage(self):
        rng = np.random.default_rng(3)
        radius = 0.04
        table = np.column_stack(
            [
                rng.uniform(-0.08, 0.08, 160),
                rng.normal(0.0, 0.001, 160),
                rng.uniform(0.78, 0.92, 160),
            ]
        )
        ball = np.column_stack(
            [
                rng.uniform(-0.03, 0.03, 180),
                rng.uniform(0.04, 0.08, 180),
                rng.uniform(0.82, 0.88, 180),
            ]
        )
        points = np.vstack([table, ball]).astype(np.float32)
        plane = TablePlane(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=0.0,
            inlier_count=1000,
            residual_m=0.001,
        )

        cleaned = remove_table_background_points(points, plane, radius_m=radius)

        self.assertGreaterEqual(len(cleaned), 150)
        self.assertLess(len(cleaned), len(points))
        self.assertGreater(float(np.min(cleaned[:, 1])), 0.006)

    def test_masked_points_filter_poor_zed_confidence(self):
        point_data = np.zeros((2, 3, 4), dtype=np.float32)
        point_data[:, :, :3] = np.array(
            [
                [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.2, 0.0, 0.5]],
                [[0.0, 0.1, 0.5], [0.1, 0.1, 0.5], [0.2, 0.1, 0.5]],
            ],
            dtype=np.float32,
        )
        mask = np.full((2, 3), 255, dtype=np.uint8)
        confidence = np.array(
            [
                [10, 20, 95],
                [15, 99, 30],
            ],
            dtype=np.float32,
        )

        points = masked_point_cloud_points(
            ArrayMat(point_data),
            mask,
            depth_min=0.0,
            depth_max=2.0,
            confidence_map=ArrayMat(confidence),
            max_confidence=90,
        )

        self.assertEqual(len(points), 4)
        self.assertFalse(np.any(np.all(np.isclose(points, [0.2, 0.0, 0.5]), axis=1)))
        self.assertFalse(np.any(np.all(np.isclose(points, [0.1, 0.1, 0.5]), axis=1)))

    def test_masked_surface_patches_estimate_normals_and_weights(self):
        height, width = 5, 5
        point_data = np.zeros((height, width, 4), dtype=np.float32)
        for row in range(height):
            for col in range(width):
                point_data[row, col, :3] = [
                    0.01 * (col - 2),
                    -0.01 * (row - 2),
                    0.80,
                ]

        mask = np.full((height, width), 255, dtype=np.uint8)
        patches = masked_surface_patches(
            ArrayMat(point_data),
            mask,
            depth_min=0.0,
            depth_max=2.0,
            max_points=0,
        )

        normal_lengths = np.linalg.norm(patches.normals, axis=1)
        valid_normals = patches.normals[normal_lengths > 0.5]
        self.assertEqual(len(patches.points), height * width)
        self.assertGreaterEqual(len(valid_normals), 9)
        self.assertTrue(np.all(patches.weights > 0.0))
        self.assertLess(np.linalg.norm(np.median(valid_normals, axis=0) - np.array([0.0, 0.0, -1.0])), 0.05)

    def test_scene_segmentation_splits_ball_table_and_other(self):
        height, width = 80, 80
        point_data = np.zeros((height, width, 4), dtype=np.float32)
        for row in range(height):
            for col in range(width):
                point_data[row, col, :3] = [
                    0.004 * (col - width // 2),
                    0.0,
                    0.55 + 0.004 * row,
                ]

        point_data[10:18, 10:18, 1] = 0.08
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[34:46, 34:46] = 255
        point_data[34:46, 34:46, 1] = 0.055
        point_data[34:38, 34:46, 1] = 0.0

        segmentation = segment_scene_from_zed_point_cloud(
            ArrayMat(point_data),
            mask,
            radius_m=0.04,
            depth_min=0.0,
            depth_max=2.0,
            max_ball_points=0,
            max_table_points=2500,
        )

        self.assertIsNotNone(segmentation.table_plane)
        self.assertTrue(segmentation.table_updated)
        self.assertGreater(len(segmentation.table_points), 100)
        self.assertGreater(len(segmentation.other_points), 20)
        self.assertGreaterEqual(len(segmentation.ball_points), 80)
        self.assertLess(len(segmentation.ball_points), len(segmentation.raw_ball_points))
        self.assertGreater(float(np.min(segmentation.ball_points[:, 1])), 0.006)

class ArrayMat:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


if __name__ == "__main__":
    unittest.main()
