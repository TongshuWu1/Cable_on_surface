import unittest

import cv2
import numpy as np

from cable_crossing import (
    CameraIntrinsics,
    extract_crossing_proposals,
    verify_crossing_proposals,
)


class CrossingObservationTests(unittest.TestCase):
    def test_rgb_proposal_and_3d_contact_are_separate(self):
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.circle(mask, (80, 60), 5, 255, -1)
        probability = np.zeros_like(mask, dtype=np.float32)
        probability[mask > 0] = 0.9
        proposals = extract_crossing_proposals(mask, probability, min_area_px=8)
        self.assertEqual(len(proposals), 1)
        np.testing.assert_allclose(proposals[0].centroid_xy, [80.0, 60.0], atol=0.25)

        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        cable1 = np.asarray([[-0.4, 0.0, 1.0], [0.4, 0.0, 1.0]], dtype=np.float32)
        cable2 = np.asarray([[0.0, -0.4, 1.0], [0.0, 0.4, 1.0]], dtype=np.float32)
        uncalibrated = verify_crossing_proposals(proposals, [cable1, cable2], intrinsics, [0.0, 0.0])
        self.assertEqual(len(uncalibrated), 1)
        self.assertFalse(uncalibrated[0].diameter_calibrated)
        self.assertFalse(uncalibrated[0].verified_contact)

        cable2_contact = cable2.copy()
        cable2_contact[:, 2] = 1.01
        calibrated = verify_crossing_proposals(proposals, [cable1, cable2_contact], intrinsics, [0.01, 0.01])
        self.assertTrue(calibrated[0].diameter_calibrated)
        self.assertTrue(calibrated[0].verified_contact)
        self.assertAlmostEqual(calibrated[0].s1_m, 0.4, places=4)
        self.assertAlmostEqual(calibrated[0].s2_m, 0.4, places=4)

    def test_image_crossing_at_different_depth_is_not_contact(self):
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.circle(mask, (80, 60), 4, 255, -1)
        proposals = extract_crossing_proposals(mask, mask.astype(np.float32) / 255.0)
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
        cable1 = np.asarray([[-0.4, 0.0, 1.0], [0.4, 0.0, 1.0]], dtype=np.float32)
        cable2 = np.asarray([[0.0, -0.8, 2.0], [0.0, 0.8, 2.0]], dtype=np.float32)
        observation = verify_crossing_proposals(
            proposals, [cable1, cable2], intrinsics, [0.01, 0.01], contact_tolerance_m=0.002
        )[0]
        self.assertFalse(observation.verified_contact)
        self.assertGreater(observation.gap_m, 0.9)
        self.assertEqual(observation.depth_order, "PF1 above PF2")
