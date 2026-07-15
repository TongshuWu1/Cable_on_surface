"""Migrate a compatible four-head checkpoint to the explicit current schema."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cable_pidnet import require_torch
from pidnet_schema import (
    ANNOTATION_BODY_LAYER_COUNT,
    CROSSING_CHANNEL,
    ENDPOINT_SEMANTICS,
    OUTPUT_CHANNEL_COUNT,
    PIDNET_LABEL_MODE,
    PIDNET_SCHEMA_VERSION,
)

require_torch()
import torch


def migrate_checkpoint(path):
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a config dictionary.")
    config = dict(payload["config"])
    endpoint_count = int(config.get("endpoint_channel_count", config.get("cable_count", 0)))
    if int(config.get("output_channels", 0)) != OUTPUT_CHANNEL_COUNT or endpoint_count != 2:
        raise ValueError(
            "Only structurally compatible cable/endpoints_cable1/endpoints_cable2/crossing checkpoints can be migrated."
        )
    if int(config.get("crossing_channel", -1)) != CROSSING_CHANNEL:
        raise ValueError(f"Expected crossing channel {CROSSING_CHANNEL}.")
    training = payload.get("training", {}) if isinstance(payload.get("training"), dict) else {}
    training_target = str(training.get("target", "")).strip().lower()
    label_mode = str(config.get("label_mode", "")).strip().lower()
    semantic_evidence = (
        "separate_endpoints" in training_target
        or "per_cable_endpoints" in label_mode
        or label_mode in {"cable_with_separate_endpoints", PIDNET_LABEL_MODE}
    )
    if not semantic_evidence:
        raise ValueError(
            "Checkpoint metadata does not prove that channels 1 and 2 are per-cable endpoint sets; "
            "refusing to relabel trained weights."
        )

    config.pop("cable_count", None)
    config.update(
        {
            "observation_schema_version": PIDNET_SCHEMA_VERSION,
            "endpoint_semantics": ENDPOINT_SEMANTICS,
            "annotation_body_layer_count": ANNOTATION_BODY_LAYER_COUNT,
            "label_mode": PIDNET_LABEL_MODE,
            "endpoint_channel_count": 2,
            "output_channels": OUTPUT_CHANNEL_COUNT,
            "crossing_channel": CROSSING_CHANNEL,
        }
    )
    payload["config"] = config
    temporary = path.with_name(path.name + f".schema-v{PIDNET_SCHEMA_VERSION}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    config = migrate_checkpoint(args.checkpoint)
    print(
        f"Migrated {args.checkpoint}: schema={config['observation_schema_version']} "
        "heads=cable,endpoints_cable1,endpoints_cable2,crossing"
    )


if __name__ == "__main__":
    main()
