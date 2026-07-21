import unittest

import numpy as np

from cable_detection import CableMaskDetector
from cable_pidnet import PidNetCableDetector
from pidnet_schema import (
    ANNOTATION_BODY_LAYER_COUNT,
    ANNOTATION_CHANNEL_COUNT,
    ENDPOINT_SEMANTICS,
    PIDNET_LABEL_MODE,
    PIDNET_SCHEMA_VERSION,
    crossing_label_value,
    endpoint_label_value,
    max_label_value,
    validate_checkpoint_schema,
)


class PidNetSchemaTests(unittest.TestCase):
    def test_annotation_layers_map_one_to_one_to_network_outputs(self):
        self.assertEqual(ANNOTATION_BODY_LAYER_COUNT, 1)
        self.assertEqual(ANNOTATION_CHANNEL_COUNT, 4)
        self.assertEqual(endpoint_label_value(1), 2)
        self.assertEqual(endpoint_label_value(2), 3)
        self.assertEqual(crossing_label_value(), 4)
        self.assertEqual(max_label_value(), 4)

    def test_explicit_per_cable_endpoint_schema_is_required(self):
        config = {
            "observation_schema_version": PIDNET_SCHEMA_VERSION,
            "endpoint_semantics": ENDPOINT_SEMANTICS,
            "endpoint_channel_count": 2,
            "label_mode": PIDNET_LABEL_MODE,
            "output_channels": 4,
            "crossing_channel": 3,
        }
        self.assertEqual(validate_checkpoint_schema(config), 2)
        legacy = dict(config)
        legacy.pop("endpoint_semantics")
        with self.assertRaises(ValueError):
            validate_checkpoint_schema(legacy)

    def test_runtime_uses_distinct_endpoint_thresholds_in_one_forward_pass(self):
        class FakeSegmenter:
            endpoint_channel_count = 2
            crossing_channels = True
            crossing_channel = 3
            output_channels = 4
            label_mode = PIDNET_LABEL_MODE

            def __init__(self):
                self.thresholds = None

            def observation_channels(self, bgr, thresholds):
                self.thresholds = tuple(thresholds)
                masks = tuple(np.zeros(bgr.shape[:2], dtype=np.uint8) for _ in range(4))
                probability = np.zeros(bgr.shape[:2], dtype=np.float16)
                return masks, probability

        detector = PidNetCableDetector.__new__(PidNetCableDetector)
        CableMaskDetector.__init__(detector, min_area=1, open_kernel=1, close_kernel=1)
        detector.segmenter = FakeSegmenter()
        detector.threshold = 0.95
        detector.detect_observation_masks(
            np.zeros((8, 12, 3), dtype=np.uint8),
            endpoint_thresholds=(0.50, 0.98),
            crossing_threshold=0.77,
        )
        self.assertEqual(detector.segmenter.thresholds, (0.95, 0.50, 0.98, 0.77))

        with self.assertRaises(ValueError):
            detector.detect_observation_masks(
                np.zeros((8, 12, 3), dtype=np.uint8),
                endpoint_thresholds=(0.50,),
            )
