import unittest

from pidnet_schema import (
    ENDPOINT_SEMANTICS,
    PIDNET_LABEL_MODE,
    PIDNET_SCHEMA_VERSION,
    validate_checkpoint_schema,
)


class PidNetSchemaTests(unittest.TestCase):
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
