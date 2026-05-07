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
)
from scripts.cable_detect.tools.cable_color_collect import AppState, build_profile
from scripts.cable_detect.trackdlo_tracker import TrackDLOParams, TrackDLOTracker


class HardwareFreeSmokeTest(unittest.TestCase):
    def test_config_profile_and_centerline_extract(self):
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config["resolution"], "HD720")

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

    def test_calibration_profile_metadata_uses_hd720(self):
        profile = build_profile(AppState())

        self.assertEqual(profile["camera_resolution"], "HD720")

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


if __name__ == "__main__":
    unittest.main()
