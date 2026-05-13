import unittest
from types import SimpleNamespace

import numpy as np

from main import (
    BallTrack,
    associate_measurements_to_tracks,
    project_sphere_to_image,
    projected_track_has_detection,
    recover_occluded_track_measurement,
)
from sphere_detection import SphereDetection, SphereEstimate3D


class MultiBallTrackingTest(unittest.TestCase):
    def test_association_matches_nearest_tracks_once(self):
        tracks = [
            BallTrack(1, None, estimate=estimate_at([0.0, 0.1, 0.8])),
            BallTrack(2, None, estimate=estimate_at([0.4, 0.1, 0.8])),
        ]
        measurements = [
            estimate_at([0.39, 0.1, 0.8]),
            estimate_at([0.02, 0.1, 0.8]),
        ]

        matches, unmatched_tracks, unmatched_measurements = associate_measurements_to_tracks(
            tracks,
            measurements,
            max_distance=0.08,
        )

        self.assertEqual(set(matches), {(0, 1), (1, 0)})
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_measurements, [])

    def test_association_leaves_far_measurement_for_new_track(self):
        tracks = [BallTrack(1, None, estimate=estimate_at([0.0, 0.1, 0.8]))]
        measurements = [estimate_at([0.5, 0.1, 0.8])]

        matches, unmatched_tracks, unmatched_measurements = associate_measurements_to_tracks(
            tracks,
            measurements,
            max_distance=0.08,
        )

        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_measurements, [0])

    def test_projection_gate_detects_when_track_already_has_rgb_blob(self):
        intrinsics = {"fx": 320.0, "fy": 320.0, "cx": 640.0, "cy": 360.0}
        projection = project_sphere_to_image([0.10, 0.08, 0.80], 0.04, intrinsics)
        self.assertIsNotNone(projection)
        center_xy, radius_px = projection
        detection = SphereDetection(
            center_xy=(center_xy[0] + 4.0, center_xy[1] - 3.0),
            radius_px=radius_px,
            area=300.0,
            circularity=0.8,
            fill_ratio=0.6,
            aspect=1.0,
            contour=np.empty((0, 1, 2), dtype=np.int32),
        )

        self.assertTrue(projected_track_has_detection(center_xy, radius_px, [detection], gate_px=20.0))

    def test_depth_occlusion_recovery_uses_projected_sphere_surface(self):
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
        center = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        radius = 0.10
        point_data = synthetic_organized_front_cap(center, radius, intrinsics, height=100, width=100)
        point_data[50, 50, :3] = [0.0, 0.0, 0.70]

        track = BallTrack(1, None, estimate=estimate_at(center, radius=radius))
        measurement = recover_occluded_track_measurement(
            track,
            ArrayMat(point_data),
            intrinsics,
            occlusion_args(),
            confidence_map=None,
            max_confidence=None,
        )

        self.assertIsNotNone(measurement)
        self.assertGreaterEqual(len(measurement.surface_points), 18)
        self.assertLess(np.linalg.norm(measurement.center_xyz - center), 0.035)
        self.assertFalse(np.any(np.all(np.isclose(measurement.surface_points, [0.0, 0.0, 0.70]), axis=1)))
        self.assertIn("occlusion-aware", measurement.method)

    def test_depth_occlusion_recovery_rejects_foreground_only_points(self):
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
        center = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        radius = 0.10
        point_data = np.full((100, 100, 4), np.nan, dtype=np.float32)
        for row in range(43, 58):
            for col in range(43, 58):
                point_data[row, col, :3] = [0.0, 0.0, 0.70]

        track = BallTrack(1, None, estimate=estimate_at(center, radius=radius))
        measurement = recover_occluded_track_measurement(
            track,
            ArrayMat(point_data),
            intrinsics,
            occlusion_args(),
            confidence_map=None,
            max_confidence=None,
        )

        self.assertIsNone(measurement)


def estimate_at(center, radius=0.04):
    return SphereEstimate3D(
        center_xyz=np.asarray(center, dtype=np.float32),
        radius_m=float(radius),
        surface_points=np.empty((0, 3), dtype=np.float32),
        residual_m=0.002,
        method="synthetic",
    )


def synthetic_organized_front_cap(center, radius, intrinsics, height, width):
    point_data = np.full((height, width, 4), np.nan, dtype=np.float32)
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    fx = float(intrinsics["fx"])
    center = np.asarray(center, dtype=np.float64)
    for row in range(height):
        for col in range(width):
            x = (col - cx) / fx
            y = -(row - cy) / fx
            lateral_sq = x * x + y * y
            if lateral_sq > float(radius * radius) * 0.80:
                continue
            z = float(center[2]) - np.sqrt(float(radius * radius) - lateral_sq)
            point_data[row, col, :3] = [
                float(center[0]) + x,
                float(center[1]) + y,
                z,
            ]
    return point_data


def occlusion_args():
    return SimpleNamespace(
        occlusion_recovery=True,
        occlusion_depth_recovery=True,
        occlusion_depth_search_scale=1.45,
        occlusion_surface_std=0.015,
        occlusion_foreground_margin=0.02,
        occlusion_min_visible_points=18,
        occlusion_max_surface_points=500,
        occlusion_depth_track_gate=0.20,
        occlusion_detection_gate_px=20.0,
        depth_min=0.0,
        depth_max=2.0,
    )


class ArrayMat:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


if __name__ == "__main__":
    unittest.main()
