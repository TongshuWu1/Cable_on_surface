import unittest

import numpy as np

from sphere_detection import SphereEstimate3D
from sphere_particle_filter import (
    ParticleFilterConfig,
    SphereParticleFilter,
    filtered_sphere_estimate,
)


class SphereParticleFilterTest(unittest.TestCase):
    def test_filter_smooths_ball_measurements(self):
        config = ParticleFilterConfig(
            particle_count=500,
            measurement_position_std_m=0.018,
            measurement_radius_std_m=0.004,
            process_position_std_m=0.002,
        )
        pf = SphereParticleFilter(config, seed=22)
        radius = 0.04
        true_center = np.array([0.15, radius, 0.85], dtype=np.float32)

        rng = np.random.default_rng(5)
        result = None
        for _ in range(12):
            measured = true_center + rng.normal(0.0, 0.006, 3).astype(np.float32)
            measurement = SphereEstimate3D(
                center_xyz=measured,
                radius_m=radius,
                surface_points=np.empty((0, 3), dtype=np.float32),
                residual_m=0.004,
                method="synthetic ball measurement",
                table_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                table_offset=0.0,
            )
            result = pf.step(measurement, 1.0 / 30.0)

        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result.center_xyz - true_center), 0.015)

        prediction = pf.step(None, 1.0 / 30.0)
        self.assertIsNotNone(prediction)
        self.assertTrue(prediction.prediction_only)

        estimate = filtered_sphere_estimate(None, prediction)
        self.assertIn("particle filter ball prediction", estimate.method)

    def test_filter_tracks_lifted_ball_without_table_projection(self):
        config = ParticleFilterConfig(
            particle_count=500,
            measurement_position_std_m=0.018,
            measurement_radius_std_m=0.004,
            process_position_std_m=0.002,
            reinitialize_distance_m=0.060,
        )
        pf = SphereParticleFilter(config, seed=31)
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        radius = 0.04

        first_measurement = SphereEstimate3D(
            center_xyz=np.array([0.10, radius, 0.80], dtype=np.float32),
            radius_m=radius,
            surface_points=np.empty((0, 3), dtype=np.float32),
            residual_m=0.004,
            method="table-reference ball measurement",
            table_normal=normal,
            table_offset=0.0,
        )
        pf.step(first_measurement, 1.0 / 30.0)

        lifted_center = np.array([0.10, 0.14, 0.80], dtype=np.float32)
        lifted_measurement = SphereEstimate3D(
            center_xyz=lifted_center,
            radius_m=radius,
            surface_points=np.empty((0, 3), dtype=np.float32),
            residual_m=0.004,
            method="ball measurement + table background",
            table_normal=normal,
            table_offset=0.0,
        )
        result = pf.step(lifted_measurement, 1.0 / 30.0)

        self.assertIsNotNone(result)
        self.assertGreater(float(result.center_xyz[1]), 0.10)
        self.assertLess(np.linalg.norm(result.center_xyz - lifted_center), 0.035)

    def test_filter_responds_to_fast_lift_without_reinitialize(self):
        config = ParticleFilterConfig(
            particle_count=600,
            reinitialize_distance_m=0.30,
        )
        pf = SphereParticleFilter(config, seed=44)
        radius = 0.04
        dt = 1.0 / 30.0
        result = None

        for index in range(8):
            center = np.array([0.10, radius + 0.012 * index, 0.80], dtype=np.float32)
            measurement = SphereEstimate3D(
                center_xyz=center,
                radius_m=radius,
                surface_points=np.empty((0, 3), dtype=np.float32),
                residual_m=0.004,
                method="known-radius depth-cap sphere + table reference",
                table_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                table_offset=0.0,
            )
            result = pf.step(measurement, dt)

        self.assertIsNotNone(result)
        self.assertLess(abs(float(result.center_xyz[1]) - (radius + 0.012 * 7)), 0.025)
        self.assertTrue(np.allclose(result.velocity_xyz, 0.0))
        self.assertEqual(pf.particles.shape[1], 3)

    def test_default_tune_tracks_fast_lift_with_surface_model(self):
        config = ParticleFilterConfig(
            fixed_radius_m=0.04,
            lock_radius=True,
        )
        pf = SphereParticleFilter(config, seed=45)
        radius = 0.04
        dt = 1.0 / 30.0
        result = None

        for index in range(9):
            center = np.array([0.10, radius + 0.014 * index, 0.84], dtype=np.float64)
            surface = synthetic_sphere_surface_points(center, radius, count=180)
            measurement = SphereEstimate3D(
                center_xyz=(center + np.array([0.004, -0.002, 0.003])).astype(np.float32),
                radius_m=radius,
                surface_points=surface.astype(np.float32),
                residual_m=0.004,
                method="stereo metric radius depth-cap sphere + table reference",
                table_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                table_offset=0.0,
            )
            result = pf.step(measurement, dt)

        expected_y = radius + 0.014 * 8
        self.assertIsNotNone(result)
        self.assertLess(abs(float(result.center_xyz[1]) - expected_y), 0.020)
        self.assertTrue(np.allclose(result.velocity_xyz, 0.0))
        self.assertTrue(result.model_likelihood_used)

    def test_measurement_proposal_catches_fast_lift_without_blend(self):
        config = ParticleFilterConfig(
            particle_count=360,
            fixed_radius_m=0.04,
            lock_radius=True,
            measurement_proposal=True,
            measurement_proposal_ratio=0.40,
            proposal_lateral_std_m=0.006,
            proposal_depth_std_m=0.010,
            measurement_position_std_m=0.025,
            surface_likelihood=False,
            projection_likelihood=False,
            process_position_std_m=0.0,
            position_measurement_blend=0.0,
            resample_effective_ratio=0.0,
            reinitialize_distance_m=0.50,
        )
        pf = SphereParticleFilter(config, seed=46)
        radius = 0.04
        old_center = np.array([0.10, radius, 0.84], dtype=np.float32)
        lifted_center = np.array([0.10, 0.19, 0.84], dtype=np.float32)
        dt = 1.0 / 60.0

        first = SphereEstimate3D(
            center_xyz=old_center,
            radius_m=radius,
            surface_points=np.empty((0, 3), dtype=np.float32),
            residual_m=0.003,
            method="initial ball",
            table_normal=None,
            table_offset=None,
        )
        pf.step(first, dt)

        lifted = SphereEstimate3D(
            center_xyz=lifted_center,
            radius_m=radius,
            surface_points=np.empty((0, 3), dtype=np.float32),
            residual_m=0.003,
            method="fast lifted ball",
            table_normal=None,
            table_offset=None,
        )
        result = pf.step(lifted, dt)

        self.assertIsNotNone(result)
        self.assertTrue(result.measurement_proposal_used)
        self.assertGreater(float(result.center_xyz[1]), 0.15)
        self.assertLess(np.linalg.norm(result.center_xyz - lifted_center), 0.040)

    def test_filter_locks_radius_to_ball_size(self):
        config = ParticleFilterConfig(
            particle_count=400,
            fixed_radius_m=0.64,
            lock_radius=True,
            measurement_radius_std_m=0.002,
        )
        pf = SphereParticleFilter(config, seed=71)
        dt = 1.0 / 30.0
        result = None

        for index, measured_radius in enumerate([0.58, 0.71, 0.55, 0.76, 0.61]):
            measurement = SphereEstimate3D(
                center_xyz=np.array([0.10, 0.64 + 0.01 * index, 1.50], dtype=np.float32),
                radius_m=measured_radius,
                surface_points=np.empty((0, 3), dtype=np.float32),
                residual_m=0.004,
                method="noisy radius measurement",
                table_normal=None,
                table_offset=None,
            )
            result = pf.step(measurement, dt)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result.radius_m), 0.64, places=6)
        self.assertEqual(pf.particles.shape[1], 3)
        self.assertTrue(result.radius_calibrated)

    def test_filter_locks_radius_to_first_measurement_when_no_fixed_radius(self):
        config = ParticleFilterConfig(
            particle_count=400,
            fixed_radius_m=0.0,
            lock_radius=True,
        )
        pf = SphereParticleFilter(config, seed=72)
        dt = 1.0 / 30.0

        for measured_radius in [0.043, 0.080, 0.025]:
            measurement = SphereEstimate3D(
                center_xyz=np.array([0.10, 0.08, 0.80], dtype=np.float32),
                radius_m=measured_radius,
                surface_points=np.empty((0, 3), dtype=np.float32),
                residual_m=0.004,
                method="noisy radius measurement",
                table_normal=None,
                table_offset=None,
            )
            result = pf.step(measurement, dt)

        self.assertAlmostEqual(float(result.radius_m), 0.043, places=6)
        self.assertTrue(result.radius_calibrated)

    def test_filter_locks_radius_after_multi_frame_calibration(self):
        config = ParticleFilterConfig(
            particle_count=300,
            fixed_radius_m=0.0,
            lock_radius=True,
            radius_calibration_min_frames=5,
            radius_calibration_frames=7,
            measurement_radius_std_m=0.002,
        )
        pf = SphereParticleFilter(config, seed=73)
        dt = 1.0 / 30.0
        result = None

        for measured_radius in [0.039, 0.041, 0.090, 0.040, 0.042, 0.038, 0.041]:
            measurement = SphereEstimate3D(
                center_xyz=np.array([0.10, 0.08, 0.80], dtype=np.float32),
                radius_m=measured_radius,
                surface_points=np.empty((0, 3), dtype=np.float32),
                residual_m=0.004,
                method="stereo metric radius measurement",
                table_normal=None,
                table_offset=None,
            )
            result = pf.step(measurement, dt)

        self.assertTrue(result.radius_calibrated)
        self.assertLess(abs(float(result.radius_m) - 0.041), 0.004)

    def test_surface_model_likelihood_prefers_sphere_consistent_center(self):
        radius = 0.04
        true_center = np.array([0.10, 0.08, 0.85], dtype=np.float64)
        biased_center = true_center + np.array([0.045, 0.0, 0.0], dtype=np.float64)
        surface_points = synthetic_sphere_surface_points(true_center, radius, count=180)
        config = ParticleFilterConfig(
            fixed_radius_m=radius,
            lock_radius=True,
            surface_likelihood=True,
            surface_distance_std_m=0.015,
            surface_pseudo_count=12.0,
            measurement_position_std_m=0.05,
            process_position_std_m=0.0,
            position_measurement_blend=0.0,
            resample_effective_ratio=0.0,
            reinitialize_distance_m=0.50,
        )
        pf = SphereParticleFilter(config, seed=82)
        pf.particles = np.array(
            [
                true_center,
                biased_center,
            ],
            dtype=np.float64,
        )
        pf.weights = np.array([0.5, 0.5], dtype=np.float64)
        pf.initialized = True
        pf.radius_m = radius
        pf.radius_calibrated = True

        measurement = SphereEstimate3D(
            center_xyz=biased_center.astype(np.float32),
            radius_m=radius,
            surface_points=surface_points.astype(np.float32),
            residual_m=0.003,
            method="biased center with correct sphere surface",
            table_normal=None,
            table_offset=None,
        )
        result = pf.step(measurement, 1.0 / 30.0)

        self.assertIsNotNone(result)
        self.assertTrue(result.model_likelihood_used)
        self.assertGreater(float(pf.weights[0]), float(pf.weights[1]))
        self.assertLess(
            np.linalg.norm(result.center_xyz - true_center),
            np.linalg.norm(result.center_xyz - biased_center),
        )

    def test_surface_patch_normals_prefer_sphere_consistent_center(self):
        radius = 0.04
        true_center = np.array([0.10, 0.08, 0.85], dtype=np.float64)
        biased_center = true_center + np.array([0.075, 0.0, 0.0], dtype=np.float64)
        surface_points, surface_normals = synthetic_sphere_surface_patches(true_center, radius, count=220)
        config = ParticleFilterConfig(
            fixed_radius_m=radius,
            lock_radius=True,
            surface_likelihood=True,
            surface_distance_std_m=10.0,
            surface_likelihood_weight=0.05,
            surface_normal_likelihood=True,
            surface_normal_std=0.12,
            surface_normal_weight=1.0,
            surface_pseudo_count=16.0,
            measurement_position_std_m=0.25,
            process_position_std_m=0.0,
            position_measurement_blend=0.0,
            resample_effective_ratio=0.0,
            reinitialize_distance_m=0.50,
        )
        pf = SphereParticleFilter(config, seed=84)
        pf.particles = np.array(
            [
                true_center,
                biased_center,
            ],
            dtype=np.float64,
        )
        pf.weights = np.array([0.5, 0.5], dtype=np.float64)
        pf.initialized = True
        pf.radius_m = radius
        pf.radius_calibrated = True

        measurement = SphereEstimate3D(
            center_xyz=biased_center.astype(np.float32),
            radius_m=radius,
            surface_points=surface_points.astype(np.float32),
            residual_m=0.003,
            method="biased center with correct surface patches",
            table_normal=None,
            table_offset=None,
            surface_normals=surface_normals.astype(np.float32),
            surface_weights=np.ones(len(surface_points), dtype=np.float32),
        )
        result = pf.step(measurement, 1.0 / 30.0)

        self.assertIsNotNone(result)
        self.assertTrue(result.model_likelihood_used)
        self.assertGreater(float(pf.weights[0]), float(pf.weights[1]))

    def test_projection_likelihood_prefers_rgb_consistent_center(self):
        radius = 0.04
        intrinsics = {"fx": 320.0, "fy": 320.0, "cx": 640.0, "cy": 360.0}
        true_center = np.array([0.10, 0.08, 0.90], dtype=np.float64)
        biased_center = np.array([0.18, 0.08, 0.90], dtype=np.float64)
        image_center = (
            intrinsics["cx"] + intrinsics["fx"] * true_center[0] / true_center[2],
            intrinsics["cy"] - intrinsics["fy"] * true_center[1] / true_center[2],
        )
        config = ParticleFilterConfig(
            fixed_radius_m=radius,
            lock_radius=True,
            surface_likelihood=False,
            projection_likelihood=True,
            projection_center_std_px=8.0,
            projection_radius_std_px=8.0,
            measurement_position_std_m=0.20,
            position_measurement_blend=0.0,
            resample_effective_ratio=0.0,
            reinitialize_distance_m=0.50,
        )
        pf = SphereParticleFilter(config, seed=83)
        pf.particles = np.array(
            [
                true_center,
                biased_center,
            ],
            dtype=np.float64,
        )
        pf.weights = np.array([0.5, 0.5], dtype=np.float64)
        pf.initialized = True
        pf.radius_m = radius
        pf.radius_calibrated = True

        measurement = SphereEstimate3D(
            center_xyz=biased_center.astype(np.float32),
            radius_m=radius,
            surface_points=np.empty((0, 3), dtype=np.float32),
            residual_m=0.003,
            method="biased center with correct RGB circle",
            table_normal=None,
            table_offset=None,
            image_center_xy=image_center,
            image_radius_px=intrinsics["fx"] * radius / true_center[2],
        )
        result = pf.step(measurement, 1.0 / 30.0, camera_intrinsics=intrinsics)

        self.assertIsNotNone(result)
        self.assertTrue(result.projection_likelihood_used)
        self.assertGreater(float(pf.weights[0]), float(pf.weights[1]))

def synthetic_sphere_surface_points(center, radius, count=180):
    rng = np.random.default_rng(10)
    directions = rng.normal(0.0, 1.0, (count * 4, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = directions[directions[:, 2] < -0.2][:count]
    return np.asarray(center, dtype=np.float64) + float(radius) * directions


def synthetic_sphere_surface_patches(center, radius, count=180):
    rng = np.random.default_rng(11)
    directions = rng.normal(0.0, 1.0, (count * 5, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = directions[directions[:, 2] < -0.2][:count]
    points = np.asarray(center, dtype=np.float64) + float(radius) * directions
    return points, directions


if __name__ == "__main__":
    unittest.main()
