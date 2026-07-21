import unittest

import cv2
import numpy as np

from cable_crossing import (
    CameraIntrinsics,
    CrossingProjectionTarget,
    assign_crossing_targets,
    estimate_crossing_axes,
    extract_crossing_proposals,
    particle_crossing_rewards,
    particle_crossing_rewards_torch,
    project_points,
)
from cable_particle_filter import (
    CableParticleFilter,
    CableParticleFilterConfig,
    cable_crossing_likelihood_update,
)

try:
    import torch
except Exception:
    torch = None


class CrossingObservationTests(unittest.TestCase):
    @staticmethod
    def _proposal(probability=0.99):
        mask = np.zeros((120, 160), dtype=np.uint8)
        probability_image = np.zeros_like(mask, dtype=np.float32)
        cv2.circle(mask, (80, 60), 6, 255, -1)
        cv2.circle(probability_image, (80, 60), 6, float(probability), -1)
        return extract_crossing_proposals(mask, probability_image, min_area_px=8)[0]

    @staticmethod
    def _x_mask():
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.line(mask, (30, 60), (130, 60), 255, 5)
        cv2.line(mask, (80, 15), (80, 105), 255, 5)
        return mask

    def test_connected_regions_become_probability_ranked_rgb_proposals(self):
        mask = np.zeros((120, 160), dtype=np.uint8)
        probability = np.zeros_like(mask, dtype=np.float32)
        cv2.circle(mask, (40, 60), 5, 255, -1)
        cv2.circle(probability, (40, 60), 5, 0.75, -1)
        cv2.circle(mask, (120, 60), 7, 255, -1)
        cv2.circle(probability, (120, 60), 7, 0.95, -1)

        proposals = extract_crossing_proposals(mask, probability, min_area_px=8)

        self.assertEqual(len(proposals), 2)
        np.testing.assert_allclose(proposals[0].centroid_xy, [120.0, 60.0], atol=0.25)
        self.assertGreater(proposals[0].mean_probability, proposals[1].mean_probability)

    def test_two_image_axes_are_estimated_from_the_cable_mask_annulus(self):
        proposal = self._proposal()

        result = estimate_crossing_axes(
            (proposal,),
            self._x_mask(),
            outer_radius_px=36,
            min_axis_separation_deg=25,
            min_axis_support_px=12,
        )[0]

        self.assertEqual(result.axes_xy.shape, (2, 2))
        self.assertGreater(result.axis_confidence, 0.5)
        alignments = np.abs(result.axes_xy @ np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertGreater(float(np.max(alignments)), 0.95)
        self.assertLess(float(np.min(alignments)), 0.15)

    def test_both_latent_axes_are_given_to_each_independent_pf(self):
        proposal = estimate_crossing_axes((self._proposal(),), self._x_mask())[0]
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        horizontal = np.asarray([[-0.4, 0.0, 1.0], [0.4, 0.0, 1.0]], dtype=np.float32)
        vertical = np.asarray([[0.0, -0.4, 1.0], [0.0, 0.4, 1.0]], dtype=np.float32)

        targets = assign_crossing_targets(
            (proposal,),
            (horizontal, vertical),
            intrinsics,
        )

        self.assertEqual([len(items) for items in targets], [2, 2])
        for cable_targets in targets:
            horizontal_alignment = [abs(float(target.axis_xy[0])) for target in cable_targets]
            vertical_alignment = [abs(float(target.axis_xy[1])) for target in cable_targets]
            self.assertGreater(max(horizontal_alignment), 0.95)
            self.assertGreater(max(vertical_alignment), 0.95)

    def test_zed_negative_z_camera_coordinates_project_in_front_of_camera(self):
        intrinsics = CameraIntrinsics(
            fx=100.0,
            fy=100.0,
            cx=80.0,
            cy=60.0,
            y_axis_up=True,
            z_axis_forward=False,
        )
        points = np.asarray([
            [0.0, 0.0, -1.0],
            [0.2, 0.1, -1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        projected = project_points(points, intrinsics)

        np.testing.assert_allclose(projected[:2], [[80.0, 60.0], [100.0, 50.0]], atol=1e-5)
        self.assertTrue(np.all(np.isnan(projected[2])))

        horizontal = np.asarray([[-0.4, 0.0, -1.0], [0.4, 0.0, -1.0]], dtype=np.float32)
        proposal = estimate_crossing_axes((self._proposal(),), self._x_mask())[0]
        targets = assign_crossing_targets(
            (proposal,),
            (horizontal,),
            intrinsics,
        )
        self.assertEqual(len(targets[0]), 2)

    def test_alternative_axes_of_one_crossing_are_not_double_counted(self):
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        targets = (
            CrossingProjectionTarget(1, np.asarray([80, 60], np.float32), np.asarray([1, 0], np.float32), 1.0),
            CrossingProjectionTarget(1, np.asarray([80, 60], np.float32), np.asarray([0, 1], np.float32), 1.0),
        )
        particles = np.asarray([
            [[-0.4, 0.0, 1.0], [0.0, 0.0, 1.0], [0.4, 0.0, 1.0]],
            [[0.0, -0.4, 1.0], [0.0, 0.0, 1.0], [0.0, 0.4, 1.0]],
        ], dtype=np.float32)

        reward, *_rest = particle_crossing_rewards(
            particles,
            targets,
            intrinsics,
            position_sigma_px=14.0,
            angle_sigma_deg=20.0,
            continuation_sigma_px=8.0,
        )

        np.testing.assert_allclose(reward, [1.0, 1.0], atol=1e-5)

    def test_reward_prefers_particles_that_pass_through_with_the_right_angle(self):
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        target = CrossingProjectionTarget(
            proposal_id=1,
            centroid_xy=np.asarray([80.0, 60.0], dtype=np.float32),
            axis_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            confidence=1.0,
        )
        particles = np.asarray([
            [[-0.4, 0.0, 1.0], [0.0, 0.0, 1.0], [0.4, 0.0, 1.0]],
            [[-0.4, 0.0, 1.0], [0.0, 0.30, 1.0], [0.4, 0.0, 1.0]],
            [[0.0, -0.4, 1.0], [0.0, 0.0, 1.0], [0.0, 0.4, 1.0]],
        ], dtype=np.float32)

        reward, distance, angle, continuation, closest, _target_index = particle_crossing_rewards(
            particles,
            (target,),
            intrinsics,
            position_sigma_px=14.0,
            angle_sigma_deg=20.0,
            continuation_sigma_px=8.0,
        )

        self.assertGreater(reward[0], 0.99)
        self.assertGreater(reward[0], reward[1])
        self.assertGreater(reward[0], reward[2])
        self.assertAlmostEqual(distance[0], 0.0, places=5)
        self.assertAlmostEqual(angle[0], 0.0, places=5)
        self.assertAlmostEqual(continuation[0], 0.0, places=5)
        np.testing.assert_allclose(closest[0], [80.0, 60.0], atol=1e-5)

    def test_two_sided_reward_rejects_a_particle_that_turns_onto_the_other_branch(self):
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        target = CrossingProjectionTarget(
            proposal_id=1,
            centroid_xy=np.asarray([80.0, 60.0], dtype=np.float32),
            axis_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            confidence=1.0,
            continuation_offset_px=18.0,
        )
        particles = np.asarray([
            [[-0.4, 0.0, 1.0], [0.0, 0.0, 1.0], [0.4, 0.0, 1.0]],
            [[-0.4, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.4, 1.0]],
        ], dtype=np.float32)

        reward, distance, angle, continuation, *_rest = particle_crossing_rewards(
            particles,
            (target,),
            intrinsics,
            position_sigma_px=14.0,
            angle_sigma_deg=20.0,
            continuation_sigma_px=8.0,
        )

        self.assertAlmostEqual(distance[1], 0.0, places=5)
        self.assertAlmostEqual(angle[1], 0.0, places=5)
        self.assertGreater(continuation[1], 10.0)
        self.assertGreater(reward[0], 4.0 * reward[1])

    def test_reward_uses_projection_only_and_does_not_compare_depths(self):
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        target = CrossingProjectionTarget(
            proposal_id=1,
            centroid_xy=np.asarray([80.0, 60.0], dtype=np.float32),
            axis_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            confidence=1.0,
        )
        same_projection_different_depth = np.asarray([
            [[-0.4, 0.0, 1.0], [0.0, 0.0, 1.0], [0.4, 0.0, 1.0]],
            [[-0.8, 0.0, 2.0], [0.0, 0.0, 2.0], [0.8, 0.0, 2.0]],
        ], dtype=np.float32)

        reward, *_rest = particle_crossing_rewards(
            same_projection_different_depth,
            (target,),
            intrinsics,
            position_sigma_px=14.0,
            angle_sigma_deg=20.0,
            continuation_sigma_px=8.0,
        )

        np.testing.assert_allclose(reward, [1.0, 1.0], atol=1e-6)

    def test_crossing_likelihood_reweights_one_pf_without_a_joint_state(self):
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        target = CrossingProjectionTarget(
            proposal_id=1,
            centroid_xy=np.asarray([80.0, 60.0], dtype=np.float32),
            axis_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            confidence=1.0,
        )
        config = CableParticleFilterConfig(
            particle_count=3,
            estimate_top_particle_count=1,
            segment_length_m=0.4,
            scoring_backend="cpu",
            crossing_log_reward=4.0,
        )
        particle_filter = CableParticleFilter(3, config=config)
        particle_filter.particles = np.asarray([
            [[-0.4, 0.0, 1.0], [0.0, 0.0, 1.0], [0.4, 0.0, 1.0]],
            [[-0.4, 0.0, 1.0], [0.0, 0.30, 1.0], [0.4, 0.0, 1.0]],
            [[0.0, -0.4, 1.0], [0.0, 0.0, 1.0], [0.0, 0.4, 1.0]],
        ], dtype=np.float64)
        particle_filter.weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
        particle_filter.initialized = True

        cable_crossing_likelihood_update(particle_filter, (target,), intrinsics)
        result = particle_filter._estimate(measurement_used=True, prediction_only=False)

        self.assertGreater(particle_filter.weights[0], particle_filter.weights[1])
        self.assertGreater(particle_filter.weights[0], particle_filter.weights[2])
        self.assertEqual(result.crossing_target_count, 1)
        self.assertGreater(result.crossing_reward, 0.99)
        self.assertAlmostEqual(result.crossing_distance_px, 0.0, places=5)
        self.assertAlmostEqual(result.crossing_angle_error_deg, 0.0, places=5)
        self.assertAlmostEqual(result.crossing_continuation_error_px, 0.0, places=5)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_crossing_geometry_matches_cpu_batch(self):
        rng = np.random.default_rng(8)
        particles = rng.normal(size=(800, 15, 3)).astype(np.float32)
        particles[:, :, :2] *= 0.15
        particles[:, :, 2] = np.abs(particles[:, :, 2]) * 0.2 + 0.45
        intrinsics = CameraIntrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0)
        targets = (
            CrossingProjectionTarget(1, np.asarray([640, 360], np.float32), np.asarray([1, 0], np.float32), 0.9),
            CrossingProjectionTarget(1, np.asarray([640, 360], np.float32), np.asarray([0, 1], np.float32), 0.9),
            CrossingProjectionTarget(2, np.asarray([600, 330], np.float32), np.asarray([0.6, 0.8], np.float32), 0.8),
        )
        cpu = particle_crossing_rewards(
            particles,
            targets,
            intrinsics,
            position_sigma_px=14,
            angle_sigma_deg=20,
            continuation_sigma_px=8,
        )
        gpu = particle_crossing_rewards_torch(
            torch.as_tensor(particles, device="cuda"),
            targets,
            intrinsics,
            position_sigma_px=14,
            angle_sigma_deg=20,
            continuation_sigma_px=8,
        )

        np.testing.assert_allclose(gpu[0].cpu().numpy(), cpu[0], atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(gpu[1].cpu().numpy(), cpu[1], atol=1e-3, rtol=1e-4)
        np.testing.assert_allclose(gpu[2].cpu().numpy(), cpu[2], atol=0.02, rtol=1e-4)
        np.testing.assert_allclose(gpu[3].cpu().numpy(), cpu[3], atol=1e-3, rtol=1e-4)
        np.testing.assert_allclose(gpu[4].cpu().numpy(), cpu[4], atol=1e-3, rtol=1e-4)
        np.testing.assert_array_equal(gpu[5].cpu().numpy(), cpu[5])


if __name__ == "__main__":
    unittest.main()
