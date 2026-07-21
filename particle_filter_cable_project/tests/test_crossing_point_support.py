import unittest

import numpy as np

from cable_detection import sampled_masked_point_cloud_point_sets


class CrossingPointSupportTests(unittest.TestCase):
    def test_cable_and_crossing_masks_remain_separate_after_batched_lift(self):
        point_cloud = np.asarray(
            [
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
                [[0.0, 0.1, 1.0], [0.1, 0.1, 1.0], [0.2, 0.1, 1.0]],
            ],
            dtype=np.float32,
        )
        cable_mask = np.asarray(
            [[255, 255, 0], [0, 0, 0]],
            dtype=np.uint8,
        )
        crossing_mask = np.asarray(
            [[0, 0, 0], [0, 0, 255]],
            dtype=np.uint8,
        )

        cable_points, crossing_points = sampled_masked_point_cloud_point_sets(
            point_cloud,
            (cable_mask, crossing_mask),
            depth_min=0.05,
            depth_max=2.0,
            max_points_by_mask=(16, 16),
        )

        np.testing.assert_allclose(cable_points, point_cloud[0, :2])
        np.testing.assert_allclose(crossing_points, point_cloud[1, 2:3])


if __name__ == "__main__":
    unittest.main()
