import argparse
import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tomllib

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except Exception:
    sl = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from cable_detection import remove_small_components
from pidnet_dataset import (
    DATASET_SPLITS,
    load_dataset_manifest,
    metadata_for_stem,
    move_session_split,
    new_session_id,
    register_dataset_item,
    remove_dataset_item_metadata,
    sanitize_identifier,
    session_split_conflicts,
    unique_capture_stem,
    utc_now_text,
)
from pidnet_schema import (
    ANNOTATION_CHANNEL_COUNT,
    ANNOTATION_SCHEMA_VERSION,
    CROSSING_CHANNEL,
    ENDPOINT_CHANNELS,
    OUTPUT_CHANNEL_COUNT,
    PIDNET_LABEL_MODE,
    crossing_label_value,
    endpoint_label_value,
    label_bit,
    max_label_value,
)

DEFAULT_DATASET_DIR = PROJECT_DIR / "datasets/two_cable_pidnet"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models/pidnet_two_cable_best.pt"
DEFAULT_IMAGE_SIZE = "1280x720"
DEFAULT_PARAMS_PATH = PROJECT_DIR / "pidnet_two_cable_training_params.json"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
LABEL_COLORS_BGR = (
    (40, 255, 40),     # cable body
    (255, 0, 255),     # endpoints_cable1
    (255, 220, 0),     # endpoints_cable2
    (0, 255, 255),     # RGB crossing proposal
)
MASK_UNDO_LIMIT = 12


def label_color_bgr(label, cable_count=None):
    label = max(1, int(label))
    return LABEL_COLORS_BGR[(label - 1) % len(LABEL_COLORS_BGR)]


def parse_args():
    parser = argparse.ArgumentParser(description="Label cable masks/endpoints and train the PIDNet-S segmenter.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Live tracker config.toml to tune PIDNet cleanup values.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", action="append", default=[], help="Image path to label. Can be repeated.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--imgsz",
        default=DEFAULT_IMAGE_SIZE,
        help="Training image size. Use 1280x720 for full ZED HD720, or a single value like 512 for square training.",
    )
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--cable-count", type=int, choices=(2,), default=2, help="Fixed two-cable endpoint schema.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--endpoint-weight", type=float, default=2.0)
    parser.add_argument("--boundary-weight", type=float, default=0.20)
    parser.add_argument("--crossing-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stop", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resolution", choices=resolution_names(), default="HD720")
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def resolution_names():
    return ["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"]


def zed_resolution(name):
    if sl is None:
        raise RuntimeError("pyzed.sl is unavailable. Use Open Images for offline labeling.")
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA,
    }[name]


def make_frame_item(
    bgr,
    path=None,
    split="train",
    dataset_dir=DEFAULT_DATASET_DIR,
    cable_count=2,
    search_other_splits=True,
    session_id=None,
):
    bgr = np.asarray(bgr, dtype=np.uint8)
    if int(cable_count) != 2:
        raise ValueError(f"This annotation schema requires exactly two endpoint sets; got {cable_count}.")
    dataset_stem = dataset_stem_for_source(path, dataset_dir) if path is not None else None
    item = {
        "path": str(Path(path).expanduser().resolve()) if path is not None else None,
        "dataset_stem": dataset_stem,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bgr": bgr.copy(),
        "mask": np.zeros(bgr.shape[:2], dtype=np.uint16),
        "cable_count": 2,
        "split": split,
        "saved_image_path": None,
        "saved_mask_path": None,
        "saved_layer_path": None,
        "dirty": False,
        "session_id": str(session_id or new_session_id()),
        "verified": False,
        "negative": False,
        "notes": "",
        "undo_stack": [],
        "redo_stack": [],
        "near_duplicate_of": None,
        "near_duplicate_distance": None,
        "annotation": {"origin": "manual"},
    }
    if path is not None:
        metadata = metadata_for_stem(dataset_dir, dataset_stem)
        if metadata:
            item["session_id"] = str(metadata.get("session_id") or item["session_id"])
            item["verified"] = bool(metadata.get("verified", False))
            item["negative"] = bool(metadata.get("negative", False))
            item["notes"] = str(metadata.get("notes", ""))
            if isinstance(metadata.get("annotation"), dict):
                item["annotation"] = dict(metadata["annotation"])
            if metadata.get("split") in DATASET_SPLITS:
                split = str(metadata["split"])
                item["split"] = split
    existing_mask, existing_split, existing_mask_path = (
        find_existing_mask(
            path,
            dataset_dir,
            stem=dataset_stem,
            cable_count=cable_count,
            preferred_split=split,
            search_other_splits=search_other_splits,
        )
        if path is not None
        else (None, None, None)
    )
    if existing_mask is not None:
        item["mask"] = existing_mask
        item["split"] = existing_split
        item["saved_mask_path"] = str(existing_mask_path)
        layer_path = layered_mask_path_from_mask_path(existing_mask_path)
        item["saved_layer_path"] = str(layer_path) if layer_path.exists() else None

    if path is not None:
        source_path = Path(path).expanduser().resolve()
        images_root = (Path(dataset_dir).expanduser().resolve() / "images")
        if path_is_within(source_path, images_root):
            item["saved_image_path"] = str(source_path)
    return item


def find_existing_mask(
    image_path,
    dataset_dir,
    cable_count=2,
    preferred_split=None,
    search_other_splits=True,
    stem=None,
):
    stem = sanitize_stem(stem or Path(image_path).stem)
    preferred_split = str(preferred_split or "").strip().lower()
    splits = [preferred_split] if preferred_split in DATASET_SPLITS else []
    if search_other_splits:
        splits.extend(split for split in DATASET_SPLITS if split not in splits)
    for split in splits:
        mask_path = Path(dataset_dir) / "masks" / split / f"{stem}.png"
        layer_path = layered_mask_path_from_mask_path(mask_path)
        if layer_path.exists():
            return read_layered_mask(layer_path, cable_count), split, mask_path
        if mask_path.exists():
            raise ValueError(f"Flat preview mask exists without its required four-layer annotation: {layer_path}")
    return None, None, None


def path_is_within(path, root):
    return Path(path).expanduser().resolve().is_relative_to(Path(root).expanduser().resolve())


def dataset_stem_for_source(source_path, dataset_dir):
    """Choose a stable dataset stem without aliasing unrelated source files."""
    source = Path(source_path).expanduser().resolve()
    root = Path(dataset_dir).expanduser().resolve()
    base = sanitize_stem(source.stem)
    if path_is_within(source, root / "images"):
        return base
    items = load_dataset_manifest(root).get("items", {})
    existing = items.get(base)
    if existing is None:
        return base
    stored_source = str(existing.get("source_path", "")).strip()
    if stored_source:
        try:
            if Path(stored_source).expanduser().resolve() == source:
                return base
        except OSError:
            pass
    digest = hashlib.sha256(str(source).casefold().encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}_{digest}"
    suffix = 2
    while candidate in items:
        candidate = f"{base}_{digest}_{suffix}"
        suffix += 1
    return candidate


def item_mask_state(item):
    if item is None:
        return "no frame"
    if bool(item.get("dirty")):
        return "unsaved changes"
    mask_path = item.get("saved_mask_path")
    layer_path = item.get("saved_layer_path")
    if (mask_path and Path(mask_path).exists()) or (layer_path and Path(layer_path).exists()):
        return "saved mask"
    if np.any(np.asarray(item["mask"], dtype=np.uint16)):
        return "unsaved mask"
    return "MASK MISSING"


def prediction_channels_to_draft(predicted_channels, target_shape):
    """Convert the four independent network heads to the editable bitmask schema."""
    channels = tuple(predicted_channels)
    if len(channels) != ANNOTATION_CHANNEL_COUNT:
        raise ValueError(
            f"A prediction draft requires {ANNOTATION_CHANNEL_COUNT} channels; got {len(channels)}."
        )
    height, width = int(target_shape[0]), int(target_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid draft target shape: {target_shape!r}")
    draft = np.zeros((height, width), dtype=np.uint16)
    for channel, predicted in enumerate(channels, start=1):
        predicted = np.asarray(predicted, dtype=bool)
        if predicted.ndim != 2:
            raise ValueError(f"Prediction channel {channel} must be HxW; got {predicted.shape}.")
        if predicted.shape != draft.shape:
            predicted = cv2.resize(
                predicted.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        draft[predicted] |= label_bit(channel)

    # Endpoint and crossing pixels are still cable pixels in the annotation
    # schema. Preserve overlaps instead of collapsing the semantic heads.
    semantic_bits = np.uint16(~int(label_bit(1)) & 0xFFFF)
    draft[(draft & semantic_bits) != 0] |= label_bit(1)
    return np.ascontiguousarray(draft)


def compact_mask_state(mask):
    mask = np.asarray(mask, dtype=np.uint16)
    layers = np.stack(
        [(mask & label_bit(label)) != 0 for label in range(1, ANNOTATION_CHANNEL_COUNT + 1)]
    )
    return mask.shape, np.packbits(layers.reshape(-1))


def compact_item_edit_state(item):
    return {
        "mask": compact_mask_state(item["mask"]),
        "verified": bool(item.get("verified", False)),
        "negative": bool(item.get("negative", False)),
        "annotation": json.loads(json.dumps(item.get("annotation") or {"origin": "manual"})),
    }


def restore_item_edit_state(item, state):
    if isinstance(state, dict) and "mask" in state:
        item["mask"] = restore_mask_state(state["mask"])
        item["verified"] = bool(state.get("verified", False))
        item["negative"] = bool(state.get("negative", False))
        item["annotation"] = dict(state.get("annotation") or {"origin": "manual"})
        return
    # LEGACY COMPATIBILITY: undo entries created before metadata-aware edit
    # states contain only the packed mask tuple.  Keep this read path until
    # projects saved by that GUI generation no longer need to be reopened.
    item["mask"] = restore_mask_state(state)


def restore_mask_state(state):
    shape, packed = state
    height, width = int(shape[0]), int(shape[1])
    layer_size = height * width
    layers = np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        count=ANNOTATION_CHANNEL_COUNT * layer_size,
    ).reshape(ANNOTATION_CHANNEL_COUNT, height, width)
    mask = np.zeros((height, width), dtype=np.uint16)
    for label, layer in enumerate(layers, start=1):
        mask[layer != 0] |= label_bit(label)
    return mask


def normalize_label_mask(mask, cable_count):
    cable_count = max(1, int(cable_count))
    mask = np.asarray(mask, dtype=np.uint8)
    labels = np.zeros(mask.shape[:2], dtype=np.uint8)
    max_label = max_label_value(cable_count)
    for label in range(1, max_label + 1):
        labels[mask == label] = label
    return labels


def bitmask_to_label_mask(mask, cable_count):
    bitmask = np.asarray(mask, dtype=np.uint16)
    labels = np.zeros(bitmask.shape[:2], dtype=np.uint8)
    for label in range(1, max_label_value(cable_count) + 1):
        labels[(bitmask & label_bit(label)) != 0] = label
    return labels


def label_pixels(mask, label, cable_count, multilabel=False):
    if int(label) <= 0:
        return np.asarray(mask) == 0
    if bool(multilabel):
        return (np.asarray(mask, dtype=np.uint16) & label_bit(label)) != 0
    labels = normalize_label_mask(mask, cable_count)
    return labels == int(label)


def read_layered_mask(path, cable_count):
    with np.load(str(path), allow_pickle=False) as payload:
        required = {"mask", "cable_count", "annotation_schema_version"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Layered mask file missing {missing}: {path}")
        mask = np.asarray(payload["mask"], dtype=np.uint16)
        stored_cable_count = int(np.asarray(payload["cable_count"]).item())
        annotation_version = int(np.asarray(payload["annotation_schema_version"]).item())
    if mask.ndim != 2:
        raise ValueError(f"Layered mask must be HxW: {path}")
    if stored_cable_count != int(cable_count):
        raise ValueError(f"Layered mask cable_count={stored_cable_count}, expected {cable_count}: {path}")
    if annotation_version != ANNOTATION_SCHEMA_VERSION:
        raise ValueError(
            f"Layered mask annotation schema={annotation_version}, expected {ANNOTATION_SCHEMA_VERSION}: {path}"
        )
    valid_bits = np.uint16((1 << ANNOTATION_CHANNEL_COUNT) - 1)
    return np.ascontiguousarray(mask & valid_bits, dtype=np.uint16)


def label_display_name(label, cable_count):
    label = int(label)
    if label <= 0:
        return "background"
    if label == 1:
        return "generic cable"
    if label == endpoint_label_value(1):
        return "cable 1 endpoints (both ends)"
    if label == endpoint_label_value(2):
        return "cable 2 endpoints (both ends)"
    if label == crossing_label_value():
        return "RGB crossing proposal"
    return f"unknown label {label}"


def sanitize_stem(stem):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem)).strip("._")
    return stem or "frame"


def label_paths_for_item(item, dataset_dir, index=0):
    split = item.get("split") or "train"
    source_path = item.get("path")
    stem = sanitize_stem(
        item.get("dataset_stem")
        or (Path(source_path).stem if source_path else f"frame_{int(index):04d}")
    )
    item["dataset_stem"] = stem
    dataset_dir = Path(dataset_dir)
    return (
        dataset_dir / "images" / split / f"{stem}.png",
        dataset_dir / "masks" / split / f"{stem}.png",
    )


def layered_mask_path_from_mask_path(mask_path):
    mask_path = Path(mask_path)
    parts = list(mask_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "masks":
            parts[index] = "masks_layers"
            return Path(*parts).with_suffix(".npz")
    return mask_path.with_suffix(".npz")


def save_label_pair(item, dataset_dir, index=0):
    split = str(item.get("split") or "train").strip().lower()
    if split not in DATASET_SPLITS:
        raise ValueError(f"Unsupported dataset split: {split!r}")
    if int(item.get("cable_count", 2)) != 2:
        raise ValueError("Cannot save an annotation that does not use exactly two endpoint sets.")
    item["split"] = split
    dataset_dir = Path(dataset_dir)
    image_path, mask_path = label_paths_for_item(item, dataset_dir, index=index)
    layer_path = layered_mask_path_from_mask_path(mask_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(item["mask"], dtype=np.uint16)
    preview_mask = bitmask_to_label_mask(mask, 2)
    annotation = json.loads(json.dumps(item.get("annotation") or {"origin": "manual"}))
    session_id = sanitize_identifier(item.get("session_id") or "session")
    item["session_id"] = session_id
    manifest = load_dataset_manifest(dataset_dir)
    stored_session = manifest.get("sessions", {}).get(session_id, {})
    stored_split = stored_session.get("split")
    if stored_split is not None and stored_split != split:
        raise ValueError(
            f"Session {session_id!r} is assigned to {stored_split!r}, not {split!r}. "
            "Move the whole session before saving."
        )
    existing_item = manifest.get("items", {}).get(image_path.stem)
    if existing_item is not None and sanitize_identifier(existing_item.get("session_id")) != session_id:
        raise ValueError(
            f"Dataset stem {image_path.stem!r} already belongs to session "
            f"{existing_item.get('session_id')!r}; refusing to overwrite it from {session_id!r}."
        )

    temporary_paths = (
        image_path.with_suffix(image_path.suffix + ".tmp"),
        mask_path.with_suffix(mask_path.suffix + ".tmp"),
        layer_path.with_suffix(layer_path.suffix + ".tmp"),
    )
    try:
        for array, temporary, description in (
            (np.asarray(item["bgr"], dtype=np.uint8), temporary_paths[0], "image"),
            (preview_mask, temporary_paths[1], "mask"),
        ):
            success, encoded = cv2.imencode(".png", array)
            if not success:
                raise IOError(f"Could not encode {description} PNG for {image_path.stem}")
            temporary.write_bytes(encoded.tobytes())
        with temporary_paths[2].open("wb") as stream:
            np.savez_compressed(
                stream,
                mask=mask,
                cable_count=2,
                annotation_schema_version=ANNOTATION_SCHEMA_VERSION,
            )
        for temporary, target in zip(temporary_paths, (image_path, mask_path, layer_path)):
            temporary.replace(target)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    register_dataset_item(
        dataset_dir,
        image_path.stem,
        session_id=session_id,
        split=split,
        verified=bool(item.get("verified", False)),
        negative=bool(item.get("negative", False)),
        source_path=item.get("path"),
        notes=item.get("notes", ""),
        annotation=annotation,
    )
    item["saved_image_path"] = str(image_path)
    item["saved_mask_path"] = str(mask_path)
    item["saved_layer_path"] = str(layer_path)
    item.setdefault("undo_stack", []).clear()
    item.setdefault("redo_stack", []).clear()
    item["dirty"] = False
    return image_path, mask_path


def count_labeled_pairs(dataset_dir):
    dataset_dir = Path(dataset_dir)
    counts = {}
    for split in DATASET_SPLITS:
        counts[split] = len(dataset_image_mask_pairs(dataset_dir, split))
    return counts


def dataset_image_mask_pairs(dataset_dir, split, verified_only=False):
    image_dir = Path(dataset_dir) / "images" / split
    mask_dir = Path(dataset_dir) / "masks" / split
    if not image_dir.exists() or not mask_dir.exists():
        return []
    mask_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    items = load_dataset_manifest(dataset_dir).get("items", {}) if bool(verified_only) else {}
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None and (
            not bool(verified_only)
            or bool(items.get(image_path.stem, {}).get("verified", False))
        ):
            pairs.append((image_path, mask_path))
    return pairs


def read_dataset_mask(mask_path, cable_count):
    layer_path = layered_mask_path_from_mask_path(mask_path)
    if not layer_path.exists():
        raise FileNotFoundError(f"Required four-layer annotation is missing: {layer_path}")
    return read_layered_mask(layer_path, cable_count), True


def binary_mask_metrics(predicted_mask, target_mask):
    predicted = np.asarray(predicted_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if predicted.shape != target.shape:
        target = cv2.resize(target.astype(np.uint8), (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    predicted_count = int(np.count_nonzero(predicted))
    target_count = int(np.count_nonzero(target))
    dice_den = predicted_count + target_count
    return {
        "iou": intersection / max(union, 1),
        "dice": (2.0 * intersection) / max(dice_den, 1),
        "predicted": predicted_count,
        "target": target_count,
        "intersection": intersection,
        "union": union,
    }


def frame_signature(bgr, width=32, height=18):
    gray = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def frame_signature_distance(left, right):
    return float(np.mean(np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))))


def dataset_metadata_counts(dataset_dir):
    manifest = load_dataset_manifest(dataset_dir)
    result = {
        split: {"items": 0, "verified": 0, "negative": 0, "sessions": set()}
        for split in DATASET_SPLITS
    }
    for item in manifest.get("items", {}).values():
        split = str(item.get("split", ""))
        if split not in result:
            continue
        result[split]["items"] += 1
        result[split]["verified"] += int(bool(item.get("verified", False)))
        result[split]["negative"] += int(bool(item.get("negative", False)))
        result[split]["sessions"].add(str(item.get("session_id", "")))
    for split in result:
        result[split]["sessions"] = len(result[split]["sessions"] - {""})
    return result, session_split_conflicts(manifest)


def body_label_mask(mask, cable_count, multilabel=False):
    labels = np.asarray(mask)
    return label_pixels(labels, 1, cable_count, multilabel=multilabel)


def crossing_label_mask(mask, cable_count, multilabel=False):
    return label_pixels(
        np.asarray(mask),
        crossing_label_value(),
        cable_count,
        multilabel=multilabel,
    )


def apply_binary_cleanup(mask, params):
    """Return the cleaned uint8 mask and the number of retained components."""
    mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if params["open_kernel"] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["open_kernel"], params["open_kernel"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if params["close_kernel"] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["close_kernel"], params["close_kernel"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned, component_count = remove_small_components(mask, min_area=params["min_area_px"])
    return cleaned, int(component_count)


def safe_int(var, default, min_value=None, max_value=None):
    try:
        value = int(float(var.get()))
    except Exception:
        value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def safe_float(var, default, min_value=None, max_value=None):
    try:
        value = float(var.get())
    except Exception:
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def load_toml_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def odd_kernel_value(var, default):
    value = safe_int(var, default, min_value=1, max_value=99)
    if value % 2 == 0:
        value += 1
    return value


def replace_toml_values(path, updates):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    section = None
    seen = set()
    output = []

    def append_missing_for_section(section_name):
        if section_name is None:
            return
        for update_key, value in updates.items():
            update_section, key = update_key
            if update_section == section_name and update_key not in seen:
                output.append(f"{key} = {toml_scalar(value)}")
                seen.add(update_key)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            append_missing_for_section(section)
            section = stripped.strip("[]").strip()
            output.append(line)
            continue
        key = None
        if section and "=" in line and not stripped.startswith("#"):
            key = line.split("=", 1)[0].strip()
        update_key = (section, key)
        if key is not None and update_key in updates:
            output.append(f"{key} = {toml_scalar(updates[update_key])}")
            seen.add(update_key)
        else:
            output.append(line)

    append_missing_for_section(section)
    missing = [key for key in updates if key not in seen]
    if missing:
        output.append("")
    missing_sections = []
    for section_name, _key in missing:
        if section_name not in missing_sections:
            missing_sections.append(section_name)
    for section_name in missing_sections:
        output.append(f"[{section_name}]")
        for update_key, value in updates.items():
            update_section, key = update_key
            if update_section == section_name and update_key not in seen:
                output.append(f"{key} = {toml_scalar(value)}")
                seen.add(update_key)

    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def toml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return f"{float(value):.8g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_scalar(item) for item in value) + "]"
    return json.dumps(str(value))


def crossing_prediction_label_mode(label_mode):
    return str(label_mode).strip().lower() == PIDNET_LABEL_MODE


class PidNetTrainingApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("Cable Label Studio - PIDNet")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(800, min(1840, screen_width - 40))
        window_height = max(640, min(1080, screen_height - 80))
        self.root.geometry(f"{window_width}x{window_height}")
        self.live_config = load_toml_config(args.config)
        pidnet_config = self.live_config.get("pidnet", {})
        detector_config = self.live_config.get("detector", {})
        crossing_config = self.live_config.get("crossing", {})
        body_threshold = float(pidnet_config.get("threshold", 0.50))
        endpoint_thresholds = tuple(
            float(value)
            for value in pidnet_config.get("endpoint_thresholds", (body_threshold, body_threshold))
        )
        if len(endpoint_thresholds) != len(ENDPOINT_CHANNELS):
            raise ValueError(
                f"pidnet.endpoint_thresholds must contain {len(ENDPOINT_CHANNELS)} values."
            )

        self.frames = []
        self.selected_frame_idx = -1
        self.latest_bgr = None
        if int(args.cable_count) != 2:
            raise ValueError(f"This project labels exactly two cables; got cable_count={args.cable_count}.")
        self.cable_count_var = tk.IntVar(value=2)
        self.mode_var = tk.StringVar(value="paint_1")
        self.split_var = tk.StringVar(value="train")
        self.session_var = tk.StringVar(value=new_session_id())
        self.verified_var = tk.BooleanVar(value=False)
        self.verification_action_var = tk.StringVar(value="Human Verify")
        self.notes_var = tk.StringVar(value="")
        self.pending_capture_session = True
        self.brush_radius_var = tk.IntVar(value=4)
        self.draw_when_zoomed_var = tk.BooleanVar(value=False)
        self.burst_count_var = tk.IntVar(value=5)
        self.burst_interval_ms_var = tk.IntVar(value=250)
        self.burst_remaining = 0
        self.epochs_var = tk.IntVar(value=int(args.epochs))
        self.batch_var = tk.IntVar(value=int(args.batch_size))
        self.imgsz_var = tk.StringVar(value=str(args.imgsz))
        self.base_channels_var = tk.IntVar(value=int(args.base_channels))
        self.lr_var = tk.DoubleVar(value=float(args.lr))
        self.weight_decay_var = tk.DoubleVar(value=float(args.weight_decay))
        self.num_workers_var = tk.IntVar(value=int(args.num_workers))
        self.prefetch_factor_var = tk.IntVar(value=int(args.prefetch_factor))
        self.endpoint_weight_var = tk.DoubleVar(value=float(args.endpoint_weight))
        self.boundary_weight_var = tk.DoubleVar(value=float(args.boundary_weight))
        self.crossing_weight_var = tk.DoubleVar(value=float(args.crossing_weight))
        self.focal_gamma_var = tk.DoubleVar(value=float(args.focal_gamma))
        self.dice_weight_var = tk.DoubleVar(value=float(args.dice_weight))
        self.ema_decay_var = tk.DoubleVar(value=float(args.ema_decay))
        self.grad_clip_var = tk.DoubleVar(value=float(args.grad_clip))
        self.early_stop_var = tk.IntVar(value=int(args.early_stop))
        self.min_delta_var = tk.DoubleVar(value=float(args.min_delta))
        self.seed_var = tk.IntVar(value=int(args.seed))
        self.amp_var = tk.BooleanVar(value=bool(args.amp))
        self.channels_last_var = tk.BooleanVar(value=bool(args.channels_last))
        self.tf32_var = tk.BooleanVar(value=bool(args.tf32))
        self.compile_var = tk.BooleanVar(value=bool(args.compile))
        self.gpu_augment_var = tk.BooleanVar(value=bool(args.gpu_augment))
        self.deterministic_var = tk.BooleanVar(value=bool(args.deterministic))
        self.resume_var = tk.BooleanVar(value=False)
        self.verified_only_var = tk.BooleanVar(value=True)
        self.evaluate_test_var = tk.BooleanVar(value=False)
        self.device_var = tk.StringVar(value=str(args.device or "cuda"))
        self.test_threshold_var = tk.DoubleVar(value=body_threshold)
        self.endpoint_threshold_vars = tuple(
            tk.DoubleVar(value=value) for value in endpoint_thresholds
        )
        self.crossing_threshold_var = tk.DoubleVar(
            value=float(crossing_config.get("threshold", body_threshold))
        )
        self.threshold_text_var = tk.StringVar(value=f"{body_threshold:.2f}")
        self.live_test_var = tk.BooleanVar(value=False)
        self.prediction_channel_var = tk.StringVar(value="combined")
        self.label_visibility_vars = {
            "cable": tk.BooleanVar(value=True),
            "endpoint1": tk.BooleanVar(value=True),
            "endpoint2": tk.BooleanVar(value=True),
            "crossing": tk.BooleanVar(value=True),
        }
        self.morph_kernel_var = tk.IntVar(value=5)
        self.morph_iterations_var = tk.IntVar(value=1)
        self.config_var = tk.StringVar(value=str(Path(args.config)))
        self.detector_min_area_var = tk.IntVar(value=int(detector_config.get("min_area_px", 80)))
        self.detector_open_kernel_var = tk.IntVar(value=int(detector_config.get("open_kernel", 3)))
        self.detector_close_kernel_var = tk.IntVar(value=int(detector_config.get("close_kernel", 5)))
        self.dataset_var = tk.StringVar(value=str(Path(args.dataset)))
        self.output_var = tk.StringVar(value=str(Path(args.output)))
        self.init_checkpoint_var = tk.StringVar(value=str(getattr(args, "init_checkpoint", None) or ""))
        self.status_var = tk.StringVar(value="Open or capture frames, paint cable masks, save labels, then train and test PIDNet.")
        self.draft_status_var = tk.StringVar(value="No editable model draft on the selected frame.")
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.train_summary_var = tk.StringVar(value="No training run active.")

        self.view_zoom = 1.0
        self.view_center_xy = None
        self.photo_refs = {}
        self.drawing = False
        self.panning = False
        self.last_image_xy = None
        self.pan_start_xy = None
        self.pan_start_center_xy = None
        self.pan_canvas_size = None

        self.zed = None
        self.runtime = None
        self.left_image = None
        self.train_process = None
        self.training_termination = None
        self.output_queue = queue.Queue()
        self.segmenter = None
        self.segmenter_key = None
        self.model_status_var = tk.StringVar(value="")
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_label_mode = ""
        self.prediction_summary = ""
        self.live_test_last_time = 0.0
        self.live_test_interval_s = 0.10
        self.checkpoint_hash_cache = {}

        self._build_ui()
        self._bind_keys()
        self.add_image_frames(args.image)
        if not self.frames:
            self.open_zed()
        self.refresh()
        self.update_model_status()
        self.poll_camera()
        self.poll_training_output()

    def _build_ui(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.root.configure(bg="#e9edf2")
        style.configure("App.TFrame", background="#e9edf2")
        style.configure("Header.TFrame", background="#172230")
        style.configure("HeaderTitle.TLabel", background="#172230", foreground="#ffffff", font=("Segoe UI", 15, "bold"))
        style.configure("HeaderSub.TLabel", background="#172230", foreground="#aebdca", font=("Segoe UI", 9))
        style.configure("Panel.TFrame", background="#f7f9fb")
        style.configure("PanelTitle.TLabel", background="#f7f9fb", foreground="#263747", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background="#dfe6ed", foreground="#243342", padding=(8, 5))
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"), foreground="#ffffff", background="#1677c8")
        style.map("Accent.TButton", background=[("active", "#0d65ad"), ("pressed", "#09558f")])
        style.configure("Danger.TButton", foreground="#a12020")
        style.configure("TNotebook", background="#e9edf2", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 7), font=("Segoe UI", 9))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#165f9b")])
        style.configure("TLabelframe", background="#ffffff")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#30465a", font=("Segoe UI", 9, "bold"))

        def section(parent, title, row, pady=(0, 8)):
            frame = ttk.LabelFrame(parent, text=title, padding=9)
            frame.grid(row=row, column=0, sticky="ew", pady=pady)
            frame.columnconfigure(1, weight=1)
            return frame

        def field(parent, row, label, variable, width=14, column=0):
            base = column * 2
            ttk.Label(parent, text=label).grid(row=row, column=base, sticky="w", padx=(0, 6), pady=3)
            entry = ttk.Entry(parent, textvariable=variable, width=width)
            entry.grid(row=row, column=base + 1, sticky="ew", pady=3, padx=(0, 8))
            entry.bind("<KeyRelease>", lambda _event: self.refresh_command_text())
            parent.columnconfigure(base + 1, weight=1)
            return entry

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(14, 9))
        header.pack(side=tk.TOP, fill=tk.X)
        title_block = ttk.Frame(header, style="Header.TFrame")
        title_block.pack(side=tk.LEFT)
        ttk.Label(title_block, text="Cable Label Studio", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text="Four-layer PIDNet annotation and training",
            style="HeaderSub.TLabel",
        ).pack(anchor="w")
        header_actions = ttk.Frame(header, style="Header.TFrame")
        header_actions.pack(side=tk.RIGHT)
        ttk.Button(header_actions, text="Open Images", command=self.open_images).pack(side=tk.LEFT, padx=3)
        ttk.Button(header_actions, text="Capture", command=self.capture_current_frame, style="Accent.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Separator(header_actions, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=7)
        ttk.Button(header_actions, text="Previous", command=self.previous_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_actions, text="Next", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Separator(header_actions, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=7)
        ttk.Button(header_actions, text="Undo", command=self.undo_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_actions, text="Redo", command=self.redo_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_actions, text="Save", command=self.save_current_label, style="Accent.TButton").pack(side=tk.LEFT, padx=(7, 2))

        footer = ttk.Frame(self.root, style="App.TFrame")
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel", anchor="w").pack(fill=tk.X)
        ttk.Label(
            footer,
            text="Shortcuts: 1/2/3/X layer | E erase | D prediction draft | Ctrl+Z/Y undo/redo | S save | mouse wheel zoom | right-drag pan",
            style="Status.TLabel",
            anchor="w",
            foreground="#5c6c79",
        ).pack(fill=tk.X)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        workspace = ttk.Frame(main, style="Panel.TFrame", padding=8)
        sidebar = ttk.Frame(main, style="App.TFrame", width=500)
        main.add(workspace, weight=5)
        main.add(sidebar, weight=2)

        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=1)
        workspace.rowconfigure(1, weight=1)
        ttk.Label(workspace, text="EDITABLE SOURCE IMAGE", style="PanelTitle.TLabel", anchor="w").grid(row=0, column=0, sticky="ew", padx=(2, 5), pady=(0, 6))
        ttk.Label(workspace, text="EXACT LAYERS / MODEL DIAGNOSTICS", style="PanelTitle.TLabel", anchor="w").grid(row=0, column=1, sticky="ew", padx=(5, 2), pady=(0, 6))
        self.canvases = []
        for column in range(2):
            canvas = tk.Canvas(
                workspace,
                bg="#11171d",
                highlightthickness=1,
                highlightbackground="#3f4b56",
                cursor="crosshair" if column == 0 else "arrow",
            )
            canvas.grid(row=1, column=column, sticky="nsew", padx=(2, 5) if column == 0 else (5, 2))
            canvas.bind("<Configure>", self.on_canvas_configure)
            canvas.bind("<MouseWheel>", self.on_wheel)
            canvas.bind("<Button-4>", self.on_wheel)
            canvas.bind("<Button-5>", self.on_wheel)
            canvas.bind("<ButtonPress-1>", self.on_left_down)
            canvas.bind("<B1-Motion>", self.on_left_drag)
            canvas.bind("<ButtonRelease-1>", self.on_left_up)
            canvas.bind("<ButtonPress-2>", self.on_pan_down)
            canvas.bind("<B2-Motion>", self.on_pan_drag)
            canvas.bind("<ButtonRelease-2>", self.on_pan_up)
            canvas.bind("<ButtonPress-3>", self.on_pan_down)
            canvas.bind("<B3-Motion>", self.on_pan_drag)
            canvas.bind("<ButtonRelease-3>", self.on_pan_up)
            self.canvases.append(canvas)
        workspace_actions = ttk.Frame(workspace, style="Panel.TFrame")
        workspace_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(workspace_actions, text="Reset View", command=self.reset_view).pack(side=tk.LEFT)
        ttk.Checkbutton(workspace_actions, text="Paint while zoomed", variable=self.draw_when_zoomed_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(
            workspace_actions,
            text="Paint on the source image; the right panel shows the exact saved layers.",
            style="PanelTitle.TLabel",
        ).pack(side=tk.RIGHT)

        self.sidebar_notebook = ttk.Notebook(sidebar)
        self.sidebar_notebook.pack(fill=tk.BOTH, expand=True)
        annotate_tab = ttk.Frame(self.sidebar_notebook, padding=10)
        data_tab = ttk.Frame(self.sidebar_notebook, padding=10)
        model_tab = ttk.Frame(self.sidebar_notebook, padding=10)
        train_tab = ttk.Frame(self.sidebar_notebook, padding=10)
        log_tab = ttk.Frame(self.sidebar_notebook, padding=8)
        for tab in (annotate_tab, data_tab, model_tab, train_tab, log_tab):
            tab.columnconfigure(0, weight=1)
        self.sidebar_notebook.add(annotate_tab, text="Annotate")
        self.sidebar_notebook.add(data_tab, text="Dataset")
        self.sidebar_notebook.add(model_tab, text="Model")
        self.sidebar_notebook.add(train_tab, text="Train")
        self.sidebar_notebook.add(log_tab, text="Log")
        self.annotate_tab = annotate_tab
        self.data_tab = data_tab
        self.model_tab = model_tab

        draft_section = section(annotate_tab, "Model-assisted editable draft", 0)
        ttk.Button(
            draft_section,
            text="Predict Current Capture as Draft",
            command=self.predict_current_capture_as_draft,
            style="Accent.TButton",
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            draft_section,
            textvariable=self.draft_status_var,
            wraplength=420,
            foreground="#53687a",
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 0))

        paint_section = section(annotate_tab, "Paint layer", 1)
        self.paint_mode_frame = tk.Frame(paint_section, bg="#ffffff")
        self.paint_mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.rebuild_paint_mode_buttons()
        brush_section = section(annotate_tab, "Brush and mask", 2)
        ttk.Label(brush_section, text="Brush radius").grid(row=0, column=0, sticky="w")
        ttk.Scale(brush_section, from_=1, to=40, variable=self.brush_radius_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=7)
        ttk.Label(brush_section, textvariable=self.brush_radius_var, width=4).grid(row=0, column=2)
        cleanup_controls = ttk.Frame(brush_section)
        cleanup_controls.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(cleanup_controls, text="Kernel").pack(side=tk.LEFT)
        ttk.Spinbox(cleanup_controls, from_=1, to=41, increment=2, width=4, textvariable=self.morph_kernel_var).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(cleanup_controls, text="Iterations").pack(side=tk.LEFT)
        ttk.Spinbox(cleanup_controls, from_=1, to=8, width=4, textvariable=self.morph_iterations_var).pack(side=tk.LEFT, padx=4)
        cleanup_buttons = ttk.Frame(brush_section)
        cleanup_buttons.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(cleanup_buttons, text="Open", command=lambda: self.apply_mask_morph("open")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(cleanup_buttons, text="Close", command=lambda: self.apply_mask_morph("close")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(cleanup_buttons, text="Clear Mask", command=self.clear_current_mask, style="Danger.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        view_section = section(annotate_tab, "Visible overlay", 3)
        for index, (channel_name, variable) in enumerate(self.label_visibility_vars.items()):
            ttk.Checkbutton(
                view_section,
                text=channel_name.replace("endpoint", "Endpoint ").title(),
                variable=variable,
                command=self.refresh,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 14), pady=3)
        review_section = section(annotate_tab, "Human review", 4)
        ttk.Label(
            review_section,
            text="After reviewing and correcting all four layers:",
            foreground="#53687a",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(
            review_section,
            textvariable=self.verification_action_var,
            command=self.toggle_human_verification,
            style="Accent.TButton",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(
            review_section,
            text="Save Human-Verified Label",
            command=self.save_reviewed_label,
            style="Accent.TButton",
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(
            annotate_tab,
            text=(
                "The prediction becomes the same editable paint layers as a manual label. "
                "Endpoint and crossing pixels retain the generic cable layer; verification is never automatic."
            ),
            wraplength=420,
            foreground="#586b7b",
            justify=tk.LEFT,
        ).grid(row=5, column=0, sticky="ew", pady=4)

        capture_section = section(data_tab, "Capture", 0)
        ttk.Button(capture_section, text="Capture Current Frame", command=self.capture_current_frame, style="Accent.TButton").grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(capture_section, text="Capture Burst", command=self.capture_burst).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        burst_options = ttk.Frame(capture_section)
        burst_options.grid(row=1, column=1, sticky="e")
        ttk.Label(burst_options, text="Frames").pack(side=tk.LEFT)
        ttk.Spinbox(burst_options, from_=2, to=100, width=4, textvariable=self.burst_count_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(burst_options, text="Every ms").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Spinbox(burst_options, from_=50, to=5000, increment=50, width=6, textvariable=self.burst_interval_ms_var).pack(side=tk.LEFT, padx=3)
        session_section = section(data_tab, "Session and split", 1)
        ttk.Entry(session_section, textvariable=self.session_var, state="readonly").grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(session_section, text="New Session", command=self.new_capture_session).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        split_frame = ttk.Frame(session_section)
        split_frame.grid(row=2, column=0, columnspan=2, sticky="w")
        for text, value in (("Train", "train"), ("Validation", "val"), ("Locked test", "test")):
            ttk.Radiobutton(split_frame, text=text, variable=self.split_var, value=value, command=self.set_active_split).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            session_section,
            text="Human verified",
            variable=self.verified_var,
            command=self.update_active_metadata,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 3))
        ttk.Label(session_section, text="Notes").grid(row=4, column=0, sticky="nw", pady=3)
        notes_entry = ttk.Entry(session_section, textvariable=self.notes_var)
        notes_entry.grid(row=4, column=1, sticky="ew", pady=3)
        notes_entry.bind("<FocusOut>", lambda _event: self.update_active_metadata())
        save_section = section(data_tab, "Save and review", 2)
        ttk.Button(save_section, text="Save Current Label", command=self.save_current_label, style="Accent.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        ttk.Button(save_section, text="Save All Painted", command=self.save_all_labels).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        ttk.Button(save_section, text="Save Verified Negative", command=self.save_verified_negative).grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        folder_buttons = ttk.Frame(save_section)
        folder_buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        for text, command in (
            ("Train", lambda: self.open_dataset_split("train")),
            ("Val", lambda: self.open_dataset_split("val")),
            ("Test", lambda: self.open_dataset_split("test")),
            ("Captures", self.open_capture_folder),
        ):
            ttk.Button(folder_buttons, text=text, command=command).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(save_section, text="Delete Current Item", command=self.delete_current_item, style="Danger.TButton").grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        path_section = section(data_tab, "Dataset location", 3, pady=(0, 0))
        ttk.Entry(path_section, textvariable=self.dataset_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_section, text="Browse", command=self.choose_dataset_dir).grid(row=0, column=1, padx=(6, 0))

        checkpoint_section = section(model_tab, "Checkpoint", 0)
        ttk.Entry(checkpoint_section, textvariable=self.output_var).grid(row=0, column=0, sticky="ew")
        checkpoint_buttons = ttk.Frame(checkpoint_section)
        checkpoint_buttons.grid(row=0, column=1, padx=(6, 0))
        ttk.Button(checkpoint_buttons, text="Browse", command=self.choose_output_path).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(checkpoint_buttons, text="Load", command=self.load_model_from_button, style="Accent.TButton").pack(side=tk.LEFT)
        ttk.Label(checkpoint_section, textvariable=self.model_status_var, wraplength=410, foreground="#53687a").grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 0))
        assist_section = section(model_tab, "Model-assisted annotation", 1)
        ttk.Button(assist_section, text="Preview Current Frame", command=self.test_current_frame).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(assist_section, text="Apply Preview as Draft", command=self.use_prediction_as_draft, style="Accent.TButton").grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Checkbutton(
            assist_section,
            text="Live segmentation preview",
            variable=self.live_test_var,
            command=self.toggle_live_segmentation,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(assist_section, text="Heatmap").grid(row=2, column=0, sticky="w", pady=(8, 0))
        channel_box = ttk.Combobox(
            assist_section,
            textvariable=self.prediction_channel_var,
            values=("combined", "cable", "endpoint1", "endpoint2", "crossing"),
            state="readonly",
            width=16,
        )
        channel_box.grid(row=2, column=1, sticky="ew", pady=(8, 0))
        channel_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        threshold_section = section(model_tab, "Channel thresholds", 2)
        for index, (label, variable) in enumerate((
            ("Cable body", self.test_threshold_var),
            ("Endpoint 1", self.endpoint_threshold_vars[0]),
            ("Endpoint 2", self.endpoint_threshold_vars[1]),
            ("Crossing", self.crossing_threshold_var),
        )):
            ttk.Label(threshold_section, text=label).grid(row=index, column=0, sticky="w", pady=3)
            threshold = ttk.Spinbox(threshold_section, from_=0.05, to=0.99, increment=0.01, textvariable=variable, width=8, command=self.on_threshold_change)
            threshold.grid(row=index, column=1, sticky="e", pady=3)
            threshold.bind("<KeyRelease>", self.on_threshold_change)
        evaluation_section = section(model_tab, "Evaluation", 3)
        for index, (text, command) in enumerate((
            ("Train", lambda: self.test_dataset_split("train")),
            ("Validation", lambda: self.test_dataset_split("val")),
            ("Locked Test", lambda: self.test_dataset_split("test")),
        )):
            ttk.Button(evaluation_section, text=text, command=command).grid(row=0, column=index, sticky="ew", padx=2)
            evaluation_section.columnconfigure(index, weight=1)
        ttk.Button(evaluation_section, text="Calibrate on Validation", command=self.calibrate_validation_thresholds).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=(6, 0))
        ttk.Button(evaluation_section, text="Clear Preview", command=self.clear_prediction).grid(row=1, column=2, sticky="ew", padx=2, pady=(6, 0))
        live_cleanup_section = section(model_tab, "Runtime mask cleanup", 4, pady=(0, 0))
        for column, (label, variable, upper) in enumerate((
            ("Min area", self.detector_min_area_var, 100000),
            ("Open", self.detector_open_kernel_var, 99),
            ("Close", self.detector_close_kernel_var, 99),
        )):
            ttk.Label(live_cleanup_section, text=label).grid(row=0, column=column, padx=3)
            ttk.Spinbox(live_cleanup_section, from_=0 if column == 0 else 1, to=upper, increment=1 if column == 0 else 2, width=7, textvariable=variable, command=self.on_live_cleanup_change).grid(row=1, column=column, padx=3)
            live_cleanup_section.columnconfigure(column, weight=1)
        ttk.Button(live_cleanup_section, text="Preview", command=self.refresh).grid(row=2, column=0, sticky="ew", padx=3, pady=(7, 0))
        ttk.Button(live_cleanup_section, text="Save to config.toml", command=self.save_live_cleanup_to_config).grid(row=2, column=1, columnspan=2, sticky="ew", padx=3, pady=(7, 0))
        ttk.Entry(live_cleanup_section, textvariable=self.config_var).grid(row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=(7, 0))
        ttk.Button(live_cleanup_section, text="Browse", command=self.choose_config_path).grid(row=3, column=2, padx=3, pady=(7, 0))

        train_actions = ttk.Frame(train_tab)
        train_actions.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(train_actions, text="Start Training", command=self.start_training, style="Accent.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(train_actions, text="Stop", command=self.stop_training).pack(side=tk.LEFT, padx=3)
        ttk.Button(train_actions, text="Check Dataset", command=self.check_dataset_health).pack(side=tk.LEFT, padx=(3, 0))
        ttk.Progressbar(train_tab, variable=self.train_progress_var, maximum=100.0).grid(row=1, column=0, sticky="ew")
        ttk.Label(train_tab, textvariable=self.train_summary_var, wraplength=430).grid(row=2, column=0, sticky="w", pady=(4, 8))
        train_settings = ttk.Notebook(train_tab)
        train_settings.grid(row=3, column=0, sticky="nsew")
        train_tab.rowconfigure(3, weight=1)
        basic_tab = ttk.Frame(train_settings, padding=9)
        loss_tab = ttk.Frame(train_settings, padding=9)
        runtime_tab = ttk.Frame(train_settings, padding=9)
        train_settings.add(basic_tab, text="Basic")
        train_settings.add(loss_tab, text="Loss")
        train_settings.add(runtime_tab, text="Runtime")
        for panel in (basic_tab, loss_tab, runtime_tab):
            for column in (1, 3):
                panel.columnconfigure(column, weight=1)
        basic_fields = (
            ("Epochs", self.epochs_var), ("Batch size", self.batch_var),
            ("Image WxH", self.imgsz_var), ("Base channels", self.base_channels_var),
            ("Device", self.device_var), ("Learning rate", self.lr_var),
            ("Weight decay", self.weight_decay_var), ("Workers", self.num_workers_var),
            ("Prefetch", self.prefetch_factor_var), ("Early-stop patience", self.early_stop_var),
            ("Seed", self.seed_var),
        )
        for index, (label, variable) in enumerate(basic_fields):
            field(basic_tab, index // 2, label, variable, column=index % 2)
        loss_fields = (
            ("Endpoint weight", self.endpoint_weight_var), ("Crossing weight", self.crossing_weight_var),
            ("Boundary weight", self.boundary_weight_var), ("Focal gamma", self.focal_gamma_var),
            ("Dice weight", self.dice_weight_var), ("EMA decay", self.ema_decay_var),
            ("Gradient clip", self.grad_clip_var), ("Min delta", self.min_delta_var),
        )
        for index, (label, variable) in enumerate(loss_fields):
            field(loss_tab, index // 2, label, variable, column=index % 2)
        for index, (text_value, variable) in enumerate((
            ("AMP mixed precision", self.amp_var),
            ("Channels-last", self.channels_last_var),
            ("TF32", self.tf32_var),
            ("GPU augmentation", self.gpu_augment_var),
            ("Deterministic", self.deterministic_var),
            ("torch.compile", self.compile_var),
            ("Verified only", self.verified_only_var),
            ("Evaluate locked test", self.evaluate_test_var),
            ("Resume last", self.resume_var),
        )):
            ttk.Checkbutton(runtime_tab, text=text_value, variable=variable, command=self.refresh_command_text).grid(row=index // 2, column=(index % 2) * 2, columnspan=2, sticky="w", pady=3)
        ttk.Label(runtime_tab, text="Initialization checkpoint").grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 3))
        ttk.Entry(runtime_tab, textvariable=self.init_checkpoint_var).grid(row=6, column=0, columnspan=3, sticky="ew")
        init_buttons = ttk.Frame(runtime_tab)
        init_buttons.grid(row=6, column=3, sticky="e", padx=(5, 0))
        ttk.Button(init_buttons, text="Browse", command=self.choose_init_checkpoint).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(init_buttons, text="Clear", command=lambda: self.init_checkpoint_var.set("")).pack(side=tk.LEFT)
        param_actions = ttk.Frame(train_tab)
        param_actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(param_actions, text="Save Parameters", command=self.save_pidnet_params).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(param_actions, text="Load Parameters", command=self.load_pidnet_params).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        log_tab.rowconfigure(3, weight=1)
        ttk.Label(log_tab, text="Run summary", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.command_text = tk.Text(log_tab, height=8, wrap=tk.WORD, relief=tk.FLAT, bg="#f1f4f7", fg="#34495b", padx=8, pady=8)
        self.command_text.grid(row=1, column=0, sticky="ew", pady=(5, 9))
        ttk.Label(log_tab, text="Training output", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
        self.output_text = tk.Text(log_tab, wrap=tk.WORD, bg="#111820", fg="#d9e5ef", insertbackground="#ffffff", relief=tk.FLAT, padx=8, pady=8)
        self.output_text.grid(row=3, column=0, sticky="nsew", pady=(5, 0))

    def rebuild_paint_mode_buttons(self):
        if not hasattr(self, "paint_mode_frame"):
            return
        for child in self.paint_mode_frame.winfo_children():
            child.destroy()
        for column in range(2):
            self.paint_mode_frame.columnconfigure(column, weight=1)
        tools = (
            ("Cable body  [1]", "paint_1", "#5de35d", "#102410"),
            ("Cable 1 endpoints  [2]", "endpoint_1", "#e058d1", "#ffffff"),
            ("Cable 2 endpoints  [3]", "endpoint_2", "#43cbe8", "#10242b"),
            ("Crossing proposal  [X]", "crossing", "#f0d84b", "#282205"),
            ("Erase crossing  [C]", "erase_crossing", "#d9e0e6", "#263746"),
            ("Erase all  [E]", "erase", "#f0c3c3", "#672020"),
        )
        for index, (text, value, color, foreground) in enumerate(tools):
            button = tk.Radiobutton(
                self.paint_mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self.refresh,
                indicatoron=False,
                relief=tk.FLAT,
                borderwidth=1,
                padx=8,
                pady=8,
                bg="#f4f6f8",
                fg="#263746",
                activebackground=color,
                activeforeground=foreground,
                selectcolor=color,
                anchor="w",
                font=("Segoe UI", 9, "bold"),
            )
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3)

    def _bind_keys(self):
        self.root.bind("1", lambda _event: self.set_mode("paint_1"))
        self.root.bind("2", lambda _event: self.set_mode("endpoint_1"))
        self.root.bind("3", lambda _event: self.set_mode("endpoint_2"))
        self.root.bind("x", lambda _event: self.set_mode("crossing"))
        self.root.bind("c", lambda _event: self.set_mode("erase_crossing"))
        self.root.bind("e", lambda _event: self.set_mode("erase"))
        self.root.bind("s", lambda _event: self.save_current_label())
        self.root.bind("a", lambda _event: self.save_all_labels())
        self.root.bind("t", lambda _event: self.start_training())
        self.root.bind("r", lambda _event: self.test_current_frame())
        self.root.bind("d", lambda _event: self.predict_current_capture_as_draft())
        self.root.bind("p", lambda _event: self.capture_current_frame())
        self.root.bind("n", lambda _event: self.next_frame())
        self.root.bind("b", lambda _event: self.previous_frame())
        self.root.bind("z", lambda _event: self.reset_view())
        self.root.bind("<Control-z>", lambda _event: self.undo_mask())
        self.root.bind("<Control-y>", lambda _event: self.redo_mask())
        self.root.bind("[", lambda _event: self.adjust_brush(-1))
        self.root.bind("]", lambda _event: self.adjust_brush(1))

    def active_item(self):
        if self.showing_live_camera():
            return None
        if not self.frames:
            return None
        self.selected_frame_idx %= len(self.frames)
        return self.frames[self.selected_frame_idx]

    def active_bgr(self):
        if self.showing_live_camera():
            return self.latest_bgr
        item = self.active_item()
        if item is not None:
            return item["bgr"]
        return self.latest_bgr

    def showing_live_camera(self):
        return bool(self.live_test_var.get()) and self.latest_bgr is not None

    def set_mode(self, mode):
        mode = str(mode)
        if mode.startswith("paint_"):
            try:
                label = int(mode.split("_", 1)[1])
            except Exception:
                label = 1
            label = int(np.clip(label, 1, max(1, int(self.cable_count_var.get()))))
            mode = f"paint_{label}"
        elif mode.startswith("endpoint_"):
            try:
                label = int(mode.split("_", 1)[1])
            except Exception:
                label = 1
            label = int(np.clip(label, 1, max(1, int(self.cable_count_var.get()))))
            mode = f"endpoint_{label}"
        self.mode_var.set(mode)
        mode_name = self.active_mode_name()
        self.status_var.set(f"Paint mode: {mode_name}.")
        self.refresh()

    def active_label_value(self):
        mode = str(self.mode_var.get())
        if mode == "erase":
            return 0
        cable_count = max(1, int(self.cable_count_var.get()))
        if mode in ("crossing", "erase_crossing"):
            return crossing_label_value(cable_count)
        if mode.startswith("paint_"):
            try:
                return int(np.clip(int(mode.split("_", 1)[1]), 1, cable_count))
            except Exception:
                return 1
        if mode.startswith("endpoint_"):
            try:
                cable_index = int(np.clip(int(mode.split("_", 1)[1]), 1, cable_count))
            except Exception:
                cable_index = 1
            return endpoint_label_value(cable_index, cable_count)
        return 1

    def active_mode_name(self):
        mode = str(self.mode_var.get())
        if mode == "erase":
            return "erase all"
        if mode == "erase_crossing":
            return "erase crossing"
        return label_display_name(self.active_label_value(), self.cable_count_var.get())

    def set_active_split(self):
        if self.training_is_running():
            item = self.active_item()
            if item is not None:
                self.split_var.set(item.get("split") or "train")
            self.status_var.set("Stop training before moving a session between dataset splits.")
            return
        if self.pending_capture_session:
            self.status_var.set(
                f"Pending capture session {self.session_var.get()} assigned to {self.split_var.get()}."
            )
            self.refresh()
            return
        item = self.active_item()
        if item is not None:
            new_split = self.split_var.get()
            if item.get("split") != new_split:
                session_id = str(item.get("session_id") or self.session_var.get())
                manifest = load_dataset_manifest(Path(self.dataset_var.get()))
                has_saved_session = session_id in manifest.get("sessions", {}) or any(
                    candidate.get("session_id") == session_id and candidate.get("saved_image_path")
                    for candidate in self.frames
                )
                if has_saved_session and not messagebox.askyesno(
                    "Move capture session",
                    f"Move the entire session {session_id!r} to {new_split!r}?\n\n"
                    "All saved images and masks in the session move together.",
                ):
                    self.split_var.set(item.get("split") or "train")
                    return
                if has_saved_session:
                    try:
                        move_session_split(Path(self.dataset_var.get()), session_id, new_split)
                    except Exception as exc:
                        self.split_var.set(item.get("split") or "train")
                        messagebox.showerror("Session move failed", str(exc))
                        return
                for candidate in self.frames:
                    if candidate.get("session_id") == session_id:
                        candidate["split"] = new_split
                        candidate["dirty"] = True
                        self.refresh_saved_paths(candidate)
        self.refresh()

    def refresh_saved_paths(self, item):
        if not item.get("saved_mask_path") and not item.get("saved_image_path"):
            return
        image_path, mask_path = label_paths_for_item(item, Path(self.dataset_var.get()))
        item["saved_image_path"] = str(image_path) if image_path.exists() else None
        item["saved_mask_path"] = str(mask_path) if mask_path.exists() else None
        layer_path = layered_mask_path_from_mask_path(mask_path)
        item["saved_layer_path"] = str(layer_path) if layer_path.exists() else None
        if item.get("path") and path_is_within(Path(item["path"]), Path(self.dataset_var.get()) / "images"):
            item["path"] = str(image_path)

    def update_active_metadata(self):
        if self.pending_capture_session:
            self.status_var.set(f"Pending capture session {self.session_var.get()} assigned to {self.split_var.get()}.")
            return
        item = self.active_item()
        if item is None:
            return
        item["session_id"] = sanitize_stem(self.session_var.get() or new_session_id())
        self.set_item_verified(item, bool(self.verified_var.get()))
        item["notes"] = str(self.notes_var.get())
        item["dirty"] = True
        self.refresh()

    def set_item_verified(self, item, verified):
        verified = bool(verified)
        annotation = dict(item.get("annotation") or {"origin": "manual"})
        annotation.setdefault("origin", "manual")
        if verified:
            annotation.setdefault("human_verified_at", utc_now_text())
        else:
            annotation.pop("human_verified_at", None)
        item["annotation"] = annotation
        item["verified"] = verified

    def update_verification_action_text(self):
        if not hasattr(self, "verification_action_var"):
            return
        item = self.active_item()
        verified = bool(item.get("verified")) if item is not None else bool(self.verified_var.get())
        self.verification_action_var.set(
            "Human Verified (click to undo)" if verified else "Human Verify"
        )

    def toggle_human_verification(self):
        item = self.active_item()
        if item is None:
            self.status_var.set("Capture or select a frame before changing human verification.")
            return False
        verified = not bool(item.get("verified", False))
        self.verified_var.set(verified)
        item["session_id"] = sanitize_stem(self.session_var.get() or item.get("session_id") or new_session_id())
        item["notes"] = str(self.notes_var.get())
        self.set_item_verified(item, verified)
        item["dirty"] = True
        self.update_verification_action_text()
        self.refresh()
        if verified:
            self.status_var.set(
                "Marked the current label as Human Verified. The Dataset-tab toggle is synchronized."
            )
        else:
            self.status_var.set(
                "Removed Human Verified from the current label. The Dataset-tab toggle is synchronized."
            )
        return True

    def mark_item_human_edited(self, item):
        """Invalidate verification whenever pixels change after human review."""
        annotation = dict(item.get("annotation") or {"origin": "manual"})
        annotation.setdefault("origin", "manual")
        annotation["human_edited"] = True
        annotation["last_human_edit_at"] = utc_now_text()
        item["annotation"] = annotation
        self.set_item_verified(item, False)
        if hasattr(self, "verified_var"):
            self.verified_var.set(False)
        item["negative"] = False
        item["dirty"] = True

    def update_draft_status(self):
        self.update_verification_action_text()
        if not hasattr(self, "draft_status_var"):
            return
        item = self.active_item()
        if item is None:
            self.draft_status_var.set("Capture or select a frame to create an editable model draft.")
            return
        annotation = item.get("annotation") if isinstance(item.get("annotation"), dict) else {}
        if annotation.get("origin") == "model_assisted":
            if bool(item.get("verified")):
                message = "Model-assisted label is human verified and ready to save."
            elif bool(annotation.get("human_edited")):
                message = "Model draft has manual corrections; review all layers, then mark Human verified."
            else:
                message = "Editable model draft is active; inspect and correct all four paint layers."
        elif bool(item.get("verified")):
            message = "Manual label is human verified and ready to save."
        else:
            message = "No editable model draft on the selected frame."
        self.draft_status_var.set(message)

    def sync_metadata_from_active(self):
        item = self.active_item()
        if item is None:
            return
        self.pending_capture_session = False
        self.session_var.set(str(item.get("session_id") or self.session_var.get()))
        self.verified_var.set(bool(item.get("verified", False)))
        self.notes_var.set(str(item.get("notes", "")))
        self.update_draft_status()

    def new_capture_session(self):
        self.pending_capture_session = True
        self.session_var.set(new_session_id())
        self.verified_var.set(False)
        self.update_verification_action_text()
        self.notes_var.set("")
        self.status_var.set(
            f"New capture session {self.session_var.get()} assigned to {self.split_var.get()}."
        )

    def on_threshold_change(self, _value=None):
        self.threshold_text_var.set(
            f"{safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.99):.2f}"
        )
        self.refresh()

    def on_live_cleanup_change(self):
        self.detector_open_kernel_var.set(odd_kernel_value(self.detector_open_kernel_var, 3))
        self.detector_close_kernel_var.set(odd_kernel_value(self.detector_close_kernel_var, 5))
        self.refresh()

    def adjust_brush(self, delta):
        self.brush_radius_var.set(int(np.clip(self.brush_radius_var.get() + delta, 1, 40)))
        self.refresh()

    def choose_config_path(self):
        path = filedialog.askopenfilename(
            title="Choose live tracker config.toml",
            initialdir=str(Path(self.config_var.get()).parent),
            filetypes=[("TOML config", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        self.config_var.set(path)
        self.load_live_cleanup_from_config(path)
        self.status_var.set(f"Loaded live cleanup values from {Path(path).name}")
        self.refresh()

    def choose_dataset_dir(self):
        path = filedialog.askdirectory(title="Choose PIDNet dataset folder", initialdir=str(Path(self.dataset_var.get()).parent))
        if path:
            self.dataset_var.set(path)
            self.refresh()

    def choose_output_path(self):
        path = filedialog.asksaveasfilename(
            title="Choose PIDNet checkpoint path",
            initialfile=Path(self.output_var.get()).name,
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)
            self.unload_segmenter()
            self.update_model_status()
            self.refresh_command_text()

    def choose_init_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Choose compatible initialization checkpoint",
            initialdir=str(Path(self.output_var.get()).parent),
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if path:
            self.init_checkpoint_var.set(path)
            self.refresh_command_text()

    def load_live_cleanup_from_config(self, path):
        config = load_toml_config(path)
        pidnet_config = config.get("pidnet", {})
        detector_config = config.get("detector", {})
        crossing_config = config.get("crossing", {})
        if "threshold" in pidnet_config:
            self.test_threshold_var.set(float(pidnet_config["threshold"]))
            self.threshold_text_var.set(f"{float(pidnet_config['threshold']):.2f}")
        if "endpoint_thresholds" in pidnet_config:
            values = tuple(float(value) for value in pidnet_config["endpoint_thresholds"])
            if len(values) != len(self.endpoint_threshold_vars):
                raise ValueError(
                    f"pidnet.endpoint_thresholds must contain {len(self.endpoint_threshold_vars)} values."
                )
            for variable, value in zip(self.endpoint_threshold_vars, values):
                variable.set(value)
        if "threshold" in crossing_config:
            self.crossing_threshold_var.set(float(crossing_config["threshold"]))
        if "min_area_px" in detector_config:
            self.detector_min_area_var.set(int(detector_config["min_area_px"]))
        if "open_kernel" in detector_config:
            self.detector_open_kernel_var.set(int(detector_config["open_kernel"]))
        if "close_kernel" in detector_config:
            self.detector_close_kernel_var.set(int(detector_config["close_kernel"]))

    def live_cleanup_params(self):
        open_kernel = odd_kernel_value(self.detector_open_kernel_var, 3)
        close_kernel = odd_kernel_value(self.detector_close_kernel_var, 5)
        self.detector_open_kernel_var.set(open_kernel)
        self.detector_close_kernel_var.set(close_kernel)
        return {
            "threshold": safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.99),
            "min_area_px": safe_int(self.detector_min_area_var, 80, min_value=0, max_value=1000000),
            "open_kernel": open_kernel,
            "close_kernel": close_kernel,
        }

    def preview_thresholds(self):
        return {
            "cable": safe_float(
                self.test_threshold_var,
                0.50,
                min_value=0.05,
                max_value=0.99,
            ),
            "endpoints": tuple(
                safe_float(variable, 0.50, min_value=0.05, max_value=0.99)
                for variable in self.endpoint_threshold_vars
            ),
            "crossing": safe_float(
                self.crossing_threshold_var,
                0.50,
                min_value=0.05,
                max_value=0.99,
            ),
        }

    def save_live_cleanup_to_config(self):
        params = self.live_cleanup_params()
        thresholds = self.preview_thresholds()
        config_path = Path(self.config_var.get() or DEFAULT_CONFIG_PATH)
        try:
            replace_toml_values(
                config_path,
                {
                    ("pidnet", "threshold"): thresholds["cable"],
                    ("pidnet", "endpoint_thresholds"): thresholds["endpoints"],
                    ("crossing", "threshold"): thresholds["crossing"],
                    ("detector", "min_area_px"): params["min_area_px"],
                    ("detector", "open_kernel"): params["open_kernel"],
                    ("detector", "close_kernel"): params["close_kernel"],
                },
            )
        except Exception as exc:
            self.status_var.set(f"Could not save live cleanup config: {exc}")
            return
        self.status_var.set(
            f"Saved live PIDNet cleanup to {config_path.name}: "
            f"thresholds={thresholds['cable']:.2f}/"
            f"{thresholds['endpoints'][0]:.2f}/{thresholds['endpoints'][1]:.2f}/"
            f"{thresholds['crossing']:.2f}, min_area={params['min_area_px']}, "
            f"open={params['open_kernel']}, close={params['close_kernel']}."
        )
        self.refresh_command_text()

    def unload_segmenter(self):
        self.segmenter = None
        self.segmenter_key = None
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_label_mode = ""
        self.prediction_summary = ""

    def checkpoint_signature(self, checkpoint_path):
        path = Path(checkpoint_path)
        stat = path.stat()
        return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)

    def checkpoint_sha256(self, checkpoint_path):
        signature = self.checkpoint_signature(checkpoint_path)
        cached = self.checkpoint_hash_cache.get(signature)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with Path(checkpoint_path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self.checkpoint_hash_cache.clear()
        self.checkpoint_hash_cache[signature] = value
        return value

    def model_draft_metadata(self):
        checkpoint_path = Path(self.output_var.get()).expanduser().resolve()
        signature = self.checkpoint_signature(checkpoint_path)
        thresholds = self.preview_thresholds()
        cleanup = self.live_cleanup_params()
        return {
            "origin": "model_assisted",
            "draft_created_at": utc_now_text(),
            "human_edited": False,
            "checkpoint": {
                "path": signature[0],
                "sha256": self.checkpoint_sha256(checkpoint_path),
                "modified_ns": signature[1],
                "bytes": signature[2],
            },
            "thresholds": {
                "cable": float(thresholds["cable"]),
                "endpoint1": float(thresholds["endpoints"][0]),
                "endpoint2": float(thresholds["endpoints"][1]),
                "crossing": float(thresholds["crossing"]),
            },
            "body_cleanup": {
                "min_area_px": int(cleanup["min_area_px"]),
                "open_kernel": int(cleanup["open_kernel"]),
                "close_kernel": int(cleanup["close_kernel"]),
            },
        }

    def update_model_status(self):
        if not hasattr(self, "model_status_var"):
            return
        checkpoint_path = Path(self.output_var.get())
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024.0 * 1024.0)
            loaded = "loaded" if self.segmenter is not None else "found"
            detail = ""
            if self.segmenter is not None:
                detail = (
                    f" {getattr(self.segmenter, 'label_mode', '?')}"
                    f"/{getattr(self.segmenter, 'input_mode', '?')}"
                    f" in={getattr(self.segmenter, 'input_channels', '?')}"
                    f" out={getattr(self.segmenter, 'output_channels', '?')}"
                    f" crossing_ch={getattr(self.segmenter, 'crossing_channel', None)}"
                )
            self.model_status_var.set(f"model {loaded}: {checkpoint_path.name} ({size_mb:.1f} MB){detail}")
        else:
            self.model_status_var.set("no checkpoint loaded")

    def load_model_from_button(self):
        try:
            segmenter = self.load_segmenter(force_reload=True)
        except Exception as exc:
            self.status_var.set(f"Could not load model: {exc}")
            self.update_model_status()
            return
        recommended = tuple(float(value) for value in segmenter.config.get("recommended_thresholds", ()))
        if len(recommended) == OUTPUT_CHANNEL_COUNT:
            self.test_threshold_var.set(recommended[0])
            self.endpoint_threshold_vars[0].set(recommended[1])
            self.endpoint_threshold_vars[1].set(recommended[2])
            self.crossing_threshold_var.set(recommended[3])
            self.on_threshold_change()
        self.status_var.set(f"Loaded model: {Path(self.output_var.get()).name}")
        self.update_model_status()

    def open_images(self):
        paths = filedialog.askopenfilenames(
            title="Open RGB images to label",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        self.add_image_frames(paths)
        self.refresh()

    def confirm_discard_unsaved(self, action):
        dirty_count = sum(bool(item.get("dirty")) for item in self.frames)
        if dirty_count == 0:
            return True
        return messagebox.askyesno(
            "Unsaved mask edits",
            f"{dirty_count} frame(s) have unsaved mask edits. Discard them and {action}?",
        )

    def open_dataset_split(self, split):
        split = str(split).strip().lower()
        if split not in DATASET_SPLITS:
            raise ValueError(f"Unsupported dataset split: {split}")
        dataset_dir = Path(self.dataset_var.get()).expanduser().resolve()
        self.load_review_folder(
            dataset_dir / "images" / split,
            split=split,
            search_other_splits=False,
            description=f"{split} folder",
        )

    def open_capture_folder(self):
        dataset_dir = Path(self.dataset_var.get()).expanduser().resolve()
        self.load_review_folder(
            dataset_dir / "captures",
            split=self.split_var.get(),
            search_other_splits=True,
            description="capture folder",
        )

    def load_review_folder(self, image_dir, split, search_other_splits, description):
        image_dir = Path(image_dir).expanduser().resolve()
        if not image_dir.is_dir():
            messagebox.showerror("Dataset folder missing", f"Folder does not exist:\n{image_dir}")
            return False
        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            messagebox.showerror("Dataset folder empty", f"No supported images found in:\n{image_dir}")
            return False
        if not self.confirm_discard_unsaved(f"open the {description}"):
            return False

        dataset_dir = Path(self.dataset_var.get()).expanduser().resolve()
        loaded = []
        for image_path in image_paths:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                messagebox.showerror("Unreadable image", f"Could not read dataset image:\n{image_path}")
                return False
            loaded.append(
                make_frame_item(
                    bgr,
                    path=image_path,
                    split=split,
                    dataset_dir=dataset_dir,
                    cable_count=max(1, int(self.cable_count_var.get())),
                    search_other_splits=search_other_splits,
                    session_id=self.session_var.get(),
                )
            )

        self.frames = loaded
        self.selected_frame_idx = 0
        self.live_test_var.set(False)
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_summary = ""
        self.sync_split_from_active()
        self.sync_metadata_from_active()
        self.reset_view()
        missing_count = sum(item_mask_state(item) == "MASK MISSING" for item in loaded)
        self.status_var.set(
            f"Opened {len(loaded)} image(s) from {description}; {missing_count} have no saved mask."
        )
        self.refresh()
        return True

    def delete_current_item(self):
        if self.training_is_running():
            self.status_var.set("Dataset deletion is locked while training is running.")
            return False
        item = self.active_item()
        if item is None or not item.get("path"):
            messagebox.showerror("Cannot delete item", "Select a saved dataset or capture image first.")
            return False

        dataset_dir = Path(self.dataset_var.get()).expanduser().resolve()
        source_path = Path(item["path"]).expanduser().resolve()
        if not path_is_within(source_path, dataset_dir):
            messagebox.showerror(
                "Delete refused",
                f"This image is outside the active dataset and will not be deleted:\n{source_path}",
            )
            return False

        image_path, mask_path = label_paths_for_item(item, dataset_dir, index=self.selected_frame_idx)
        candidate_paths = [
            source_path,
            image_path,
            mask_path,
            layered_mask_path_from_mask_path(mask_path),
        ]
        for key in ("saved_image_path", "saved_mask_path", "saved_layer_path"):
            if item.get(key):
                candidate_paths.append(Path(item[key]))

        resolved_paths = []
        seen = set()
        for path in candidate_paths:
            resolved = Path(path).expanduser().resolve()
            if not path_is_within(resolved, dataset_dir):
                messagebox.showerror("Delete refused", f"A related file is outside the active dataset:\n{resolved}")
                return False
            key = str(resolved).lower()
            if key not in seen and resolved.exists():
                seen.add(key)
                resolved_paths.append(resolved)

        if not resolved_paths:
            messagebox.showerror("Nothing to delete", "No files for the selected item exist on disk.")
            return False
        if not messagebox.askyesno(
            "Delete dataset item",
            f"Delete {source_path.name} and {len(resolved_paths) - 1} related file(s)?\n"
            f"Mask state: {item_mask_state(item)}\n\nThis cannot be undone.",
        ):
            return False

        try:
            for path in resolved_paths:
                path.unlink()
            remove_dataset_item_metadata(dataset_dir, item.get("dataset_stem") or source_path.stem)
        except OSError as exc:
            messagebox.showerror("Delete failed", f"Dataset deletion failed:\n{exc}")
            raise

        removed_name = source_path.name
        self.frames.pop(self.selected_frame_idx)
        self.selected_frame_idx = min(self.selected_frame_idx, len(self.frames) - 1) if self.frames else -1
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_summary = ""
        self.sync_split_from_active()
        self.sync_metadata_from_active()
        self.reset_view()
        self.status_var.set(f"Deleted dataset item {removed_name} ({len(resolved_paths)} file(s)).")
        self.refresh()
        return True

    def add_image_frames(self, paths):
        added = 0
        for image_path in paths:
            path = Path(image_path)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            self.frames.append(
                make_frame_item(
                    bgr,
                    path=path,
                    split=self.split_var.get(),
                    dataset_dir=Path(self.dataset_var.get()),
                    cable_count=max(1, int(self.cable_count_var.get())),
                    session_id=self.session_var.get(),
                )
            )
            added += 1
        if added and self.selected_frame_idx < 0:
            self.selected_frame_idx = 0
        if added:
            self.sync_split_from_active()
            self.sync_metadata_from_active()
            self.reset_view()
            self.status_var.set(f"Loaded {added} image(s). Paint cable bodies and endpoints, then save labels.")

    def open_zed(self):
        if sl is None:
            self.status_var.set("pyzed.sl unavailable. Use Open Images to label existing frames.")
            return
        zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = zed_resolution(self.args.resolution)
        init.camera_fps = self.args.fps
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.1
        init.depth_maximum_distance = 3.0
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.status_var.set(f"Could not open ZED camera: {status}. Use Open Images instead.")
            return
        self.zed = zed
        self.runtime = sl.RuntimeParameters()
        self.runtime.confidence_threshold = 60
        self.runtime.texture_confidence_threshold = 70
        self.runtime.remove_saturated_areas = False
        self.left_image = sl.Mat()
        self.status_var.set("ZED preview active. Press Capture ZED to freeze a label frame.")

    def poll_camera(self):
        if self.zed is not None and self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
            self.latest_bgr = cv2.cvtColor(self.left_image.get_data(), cv2.COLOR_BGRA2BGR)
            if self.live_test_var.get():
                self.update_live_prediction_if_needed()
                self.refresh()
            elif not self.frames:
                self.refresh()
        self.root.after(33, self.poll_camera)

    def capture_current_frame(self):
        if self.latest_bgr is None:
            self.status_var.set("No live ZED frame available. Use Open Images or wait for camera preview.")
            return False
        # Capture freezes the current camera image for editing. A live preview
        # must not keep hiding the newly selected frame behind the camera feed.
        self.live_test_var.set(False)
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_label_mode = ""
        self.prediction_summary = ""
        capture_dir = Path(self.dataset_var.get()).expanduser().resolve() / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        filename = capture_dir / f"{unique_capture_stem(self.session_var.get())}.png"
        if filename.exists():
            raise FileExistsError(f"Unique capture path unexpectedly exists: {filename}")
        if not cv2.imwrite(str(filename), self.latest_bgr):
            raise IOError(f"Could not write captured image: {filename}")
        current_signature = frame_signature(self.latest_bgr)
        closest_name = None
        closest_distance = float("inf")
        for previous in self.frames[-100:]:
            distance = frame_signature_distance(current_signature, frame_signature(previous["bgr"]))
            if distance < closest_distance:
                closest_distance = distance
                closest_name = Path(previous.get("path") or "loaded frame").name
        item = make_frame_item(
            self.latest_bgr,
            path=filename,
            split=self.split_var.get(),
            dataset_dir=Path(self.dataset_var.get()),
            cable_count=max(1, int(self.cable_count_var.get())),
            session_id=self.session_var.get(),
        )
        self.pending_capture_session = False
        if closest_distance < 0.025:
            item["near_duplicate_of"] = closest_name
            item["near_duplicate_distance"] = closest_distance
        self.frames.append(item)
        self.selected_frame_idx = len(self.frames) - 1
        self.sync_metadata_from_active()
        self.reset_view()
        duplicate_text = (
            f" Possible near-duplicate of {closest_name} (distance {closest_distance:.4f}); review manually."
            if item.get("near_duplicate_of") else ""
        )
        self.refresh()
        self.status_var.set(
            f"Captured {filename.name} in session {item['session_id']}. Paint the four independent layers."
            f"{duplicate_text}"
        )
        return True

    def previous_frame(self):
        if not self.frames:
            return
        self.selected_frame_idx = (self.selected_frame_idx - 1) % len(self.frames)
        self.sync_split_from_active()
        self.sync_metadata_from_active()
        self.reset_view()

    def next_frame(self):
        if not self.frames:
            return
        self.selected_frame_idx = (self.selected_frame_idx + 1) % len(self.frames)
        self.sync_split_from_active()
        self.sync_metadata_from_active()
        self.reset_view()

    def sync_split_from_active(self):
        item = self.active_item()
        if item is not None:
            self.split_var.set(item.get("split") or "train")

    def clear_current_mask(self):
        item = self.active_item()
        if item is None:
            return
        if np.any(item["mask"]):
            self.push_undo_state(item)
            self.mark_item_human_edited(item)
        item["mask"].fill(0)
        self.status_var.set("Cleared current mask.")
        self.refresh()

    def capture_burst(self):
        if self.burst_remaining > 0:
            self.status_var.set(f"A capture burst is already active ({self.burst_remaining} frames remaining).")
            return
        if self.latest_bgr is None:
            self.status_var.set("No live ZED frame is available for burst capture.")
            return
        self.burst_remaining = safe_int(self.burst_count_var, 5, min_value=2, max_value=100)
        self._capture_burst_step()

    def _capture_burst_step(self):
        if self.burst_remaining <= 0:
            self.status_var.set(f"Capture burst complete for session {self.session_var.get()}.")
            return
        self.capture_current_frame()
        self.burst_remaining -= 1
        interval = safe_int(self.burst_interval_ms_var, 250, min_value=50, max_value=5000)
        self.root.after(interval, self._capture_burst_step)

    def push_undo_state(self, item):
        stack = item.setdefault("undo_stack", [])
        stack.append(compact_item_edit_state(item))
        del stack[:-MASK_UNDO_LIMIT]
        item.setdefault("redo_stack", []).clear()

    def undo_mask(self):
        item = self.active_item()
        if item is None or not item.get("undo_stack"):
            self.status_var.set("Nothing to undo for the current frame.")
            return
        item.setdefault("redo_stack", []).append(compact_item_edit_state(item))
        restore_item_edit_state(item, item["undo_stack"].pop())
        item["dirty"] = True
        self.verified_var.set(bool(item.get("verified", False)))
        self.status_var.set("Undid the previous mask edit.")
        self.refresh()

    def redo_mask(self):
        item = self.active_item()
        if item is None or not item.get("redo_stack"):
            self.status_var.set("Nothing to redo for the current frame.")
            return
        item.setdefault("undo_stack", []).append(compact_item_edit_state(item))
        restore_item_edit_state(item, item["redo_stack"].pop())
        item["dirty"] = True
        self.verified_var.set(bool(item.get("verified", False)))
        self.status_var.set("Redid the mask edit.")
        self.refresh()

    def apply_mask_morph(self, operation):
        item = self.active_item()
        if item is None:
            self.status_var.set("No saved frame is selected for mask cleanup.")
            return
        kernel_size = safe_int(self.morph_kernel_var, 5, min_value=1, max_value=99)
        if kernel_size % 2 == 0:
            kernel_size += 1
            self.morph_kernel_var.set(kernel_size)
        iterations = safe_int(self.morph_iterations_var, 1, min_value=1, max_value=16)
        op_map = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
        }
        if operation not in op_map:
            return
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        label = self.active_label_value()
        if label <= 0:
            self.status_var.set("Select a cable label before applying label cleanup.")
            return
        before = int(np.count_nonzero(label_pixels(item["mask"], label, self.cable_count_var.get(), multilabel=True)))
        binary = label_pixels(item["mask"], label, self.cable_count_var.get(), multilabel=True).astype(np.uint8) * 255
        original_binary = binary.copy()
        binary = cv2.morphologyEx(binary, op_map[operation], kernel, iterations=iterations)
        if not np.array_equal(original_binary, binary):
            self.push_undo_state(item)
        bit = label_bit(label)
        active_pixels = (item["mask"] & bit) != 0
        item["mask"][active_pixels] = item["mask"][active_pixels] & np.uint16(~int(bit) & 0xFFFF)
        item["mask"][binary > 127] |= bit
        if label != 1:
            item["mask"][binary > 127] |= label_bit(1)
        after = int(np.count_nonzero(label_pixels(item["mask"], label, self.cable_count_var.get(), multilabel=True)))
        if not np.array_equal(original_binary, binary):
            self.mark_item_human_edited(item)
        self.status_var.set(
            f"Applied mask {operation} to {label_display_name(label, self.cable_count_var.get())}: "
            f"px {before} -> {after}."
        )
        self.refresh()

    def save_current_label(self):
        if self.training_is_running():
            self.status_var.set("Dataset saving is locked while training is running.")
            return False
        item = self.active_item()
        if item is None:
            self.status_var.set("No frame selected. Open or capture a frame first.")
            return False
        if int(np.count_nonzero(item["mask"])) == 0:
            if not messagebox.askyesno(
                "Background-only frame",
                "No cable pixels are painted. This will save the whole frame as background. Save it?",
            ):
                return False
            item["negative"] = True
        else:
            item["negative"] = False
        if not self.pending_capture_session:
            item["session_id"] = sanitize_stem(self.session_var.get() or item.get("session_id"))
            self.set_item_verified(item, bool(self.verified_var.get()))
            item["notes"] = str(self.notes_var.get())
        if not self.pending_capture_session:
            item["split"] = self.split_var.get()
        item["cable_count"] = max(1, int(self.cable_count_var.get()))
        image_path, mask_path = save_label_pair(item, Path(self.dataset_var.get()), index=self.selected_frame_idx)
        self.status_var.set(f"Saved {item['split']} label: {image_path.name} and {mask_path.name}")
        self.refresh_command_text()
        return True

    def save_reviewed_label(self):
        item = self.active_item()
        if item is None:
            self.status_var.set("Capture or select a frame before saving a reviewed label.")
            return False
        self.update_active_metadata()
        if not bool(item.get("verified")):
            self.status_var.set(
                "Review and correct all four layers, then check Human verified before saving this label."
            )
            if hasattr(self, "sidebar_notebook") and hasattr(self, "annotate_tab"):
                self.sidebar_notebook.select(self.annotate_tab)
            return False
        return self.save_current_label()

    def save_verified_negative(self):
        if self.training_is_running():
            self.status_var.set("Dataset saving is locked while training is running.")
            return False
        item = self.active_item()
        if item is None:
            self.status_var.set("No frame selected. Open or capture a frame first.")
            return False
        if np.any(item["mask"]) and not messagebox.askyesno(
            "Save verified negative",
            "Clear every painted layer and save this frame as a human-verified negative example?",
        ):
            return False
        if np.any(item["mask"]):
            self.push_undo_state(item)
        item["mask"].fill(0)
        item["negative"] = True
        item["annotation"] = {"origin": "manual", "human_edited": True}
        self.set_item_verified(item, True)
        self.verified_var.set(True)
        if not self.pending_capture_session:
            item["session_id"] = sanitize_stem(self.session_var.get() or item.get("session_id"))
            item["notes"] = str(self.notes_var.get())
        if not self.pending_capture_session:
            item["split"] = self.split_var.get()
        item["cable_count"] = max(1, int(self.cable_count_var.get()))
        image_path, _mask_path = save_label_pair(item, Path(self.dataset_var.get()), index=self.selected_frame_idx)
        self.status_var.set(f"Saved human-verified negative frame: {image_path.name}")
        self.refresh_command_text()
        self.refresh()
        return True

    def save_all_labels(self):
        if self.training_is_running():
            self.status_var.set("Dataset saving is locked while training is running.")
            return False
        if not self.frames:
            self.status_var.set("No frames to save.")
            return False
        self.update_active_metadata()
        saved = 0
        for index, item in enumerate(self.frames):
            if int(np.count_nonzero(item["mask"])) == 0 and not bool(item.get("negative", False)):
                continue
            item["cable_count"] = max(1, int(self.cable_count_var.get()))
            save_label_pair(item, Path(self.dataset_var.get()), index=index)
            saved += 1
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        skipped = len(self.frames) - saved
        self.status_var.set(
            f"Saved {saved} painted mask(s), skipped {skipped} empty mask(s). "
            f"Dataset now has train={counts['train']} val={counts['val']} test={counts['test']}."
        )
        self.refresh_command_text()
        return saved > 0

    def current_pidnet_params(self):
        thresholds = self.preview_thresholds()
        return {
            "version": 3,
            "dataset": str(Path(self.dataset_var.get())),
            "output": str(Path(self.output_var.get())),
            "init_checkpoint": str(self.init_checkpoint_var.get()).strip(),
            "epochs": safe_int(self.epochs_var, 120, min_value=1),
            "batch_size": safe_int(self.batch_var, 8, min_value=1),
            "imgsz": str(self.imgsz_var.get()).strip() or DEFAULT_IMAGE_SIZE,
            "base_channels": safe_int(self.base_channels_var, 24, min_value=1),
            "cable_count": 2,
            "device": str(self.device_var.get() or "cuda"),
            "lr": safe_float(self.lr_var, 1e-3, min_value=1e-8),
            "weight_decay": safe_float(self.weight_decay_var, 1e-4, min_value=0.0),
            "num_workers": safe_int(self.num_workers_var, 2, min_value=0),
            "prefetch_factor": safe_int(self.prefetch_factor_var, 1, min_value=1),
            "endpoint_weight": safe_float(self.endpoint_weight_var, 2.0, min_value=0.0),
            "boundary_weight": safe_float(self.boundary_weight_var, 0.20, min_value=0.0),
            "crossing_weight": safe_float(self.crossing_weight_var, 1.0, min_value=0.0),
            "focal_gamma": safe_float(self.focal_gamma_var, 2.0, min_value=0.0),
            "dice_weight": safe_float(self.dice_weight_var, 1.0, min_value=0.0),
            "ema_decay": safe_float(self.ema_decay_var, 0.995, min_value=0.0, max_value=0.99999),
            "grad_clip": safe_float(self.grad_clip_var, 5.0, min_value=0.0),
            "early_stop": safe_int(self.early_stop_var, 24, min_value=0),
            "min_delta": safe_float(self.min_delta_var, 1e-4, min_value=0.0),
            "seed": safe_int(self.seed_var, 17, min_value=0),
            "amp": bool(self.amp_var.get()),
            "channels_last": bool(self.channels_last_var.get()),
            "tf32": bool(self.tf32_var.get()),
            "compile": bool(self.compile_var.get()),
            "gpu_augment": bool(self.gpu_augment_var.get()),
            "deterministic": bool(self.deterministic_var.get()),
            "verified_only": bool(self.verified_only_var.get()),
            "evaluate_test": bool(self.evaluate_test_var.get()),
            "resume_last": bool(self.resume_var.get()),
            "test_threshold": thresholds["cable"],
            "endpoint_thresholds": list(thresholds["endpoints"]),
            "crossing_threshold": thresholds["crossing"],
            "config": str(Path(self.config_var.get() or DEFAULT_CONFIG_PATH)),
            "live_mask_cleanup": self.live_cleanup_params(),
            "label_mask_cleanup": {
                "kernel": safe_int(self.morph_kernel_var, 5, min_value=1, max_value=99),
                "iterations": safe_int(self.morph_iterations_var, 1, min_value=1, max_value=16),
            },
        }

    def apply_pidnet_params(self, params):
        if not isinstance(params, dict):
            raise ValueError("PIDNet parameter file must contain a JSON object.")
        if int(params.get("cable_count", 2)) != 2:
            raise ValueError("This project uses exactly two endpoint sets; cable_count must be 2.")
        if "dataset" in params:
            self.dataset_var.set(str(params["dataset"]))
        if "output" in params:
            self.output_var.set(str(params["output"]))
            self.unload_segmenter()
            self.update_model_status()
        if "init_checkpoint" in params:
            self.init_checkpoint_var.set(str(params["init_checkpoint"]))
        for key, var in (
            ("epochs", self.epochs_var),
            ("batch_size", self.batch_var),
            ("base_channels", self.base_channels_var),
            ("num_workers", self.num_workers_var),
            ("prefetch_factor", self.prefetch_factor_var),
            ("seed", self.seed_var),
        ):
            if key in params:
                var.set(int(params[key]))
        for key, var in (
            ("lr", self.lr_var),
            ("weight_decay", self.weight_decay_var),
            ("boundary_weight", self.boundary_weight_var),
            ("endpoint_weight", self.endpoint_weight_var),
            ("crossing_weight", self.crossing_weight_var),
            ("focal_gamma", self.focal_gamma_var),
            ("dice_weight", self.dice_weight_var),
            ("ema_decay", self.ema_decay_var),
            ("grad_clip", self.grad_clip_var),
            ("min_delta", self.min_delta_var),
            ("test_threshold", self.test_threshold_var),
        ):
            if key in params:
                var.set(float(params[key]))
        if "endpoint_thresholds" in params:
            values = tuple(float(value) for value in params["endpoint_thresholds"])
            if len(values) != len(self.endpoint_threshold_vars):
                raise ValueError(
                    f"endpoint_thresholds must contain {len(self.endpoint_threshold_vars)} values."
                )
            for variable, value in zip(self.endpoint_threshold_vars, values):
                variable.set(value)
        if "crossing_threshold" in params:
            self.crossing_threshold_var.set(float(params["crossing_threshold"]))
        if "imgsz" in params:
            self.imgsz_var.set(str(params["imgsz"]))
        if "device" in params:
            self.device_var.set(str(params["device"]))
        if "amp" in params:
            self.amp_var.set(bool(params["amp"]))
        for key, variable in (
            ("channels_last", self.channels_last_var),
            ("tf32", self.tf32_var),
            ("compile", self.compile_var),
            ("gpu_augment", self.gpu_augment_var),
            ("deterministic", self.deterministic_var),
            ("verified_only", self.verified_only_var),
            ("evaluate_test", self.evaluate_test_var),
            ("resume_last", self.resume_var),
        ):
            if key in params:
                variable.set(bool(params[key]))
        if "early_stop" in params:
            self.early_stop_var.set(int(params["early_stop"]))
        if "config" in params:
            self.config_var.set(str(params["config"]))
        self.rebuild_paint_mode_buttons()
        cleanup_params = params.get("live_mask_cleanup", {})
        if isinstance(cleanup_params, dict):
            if "threshold" in cleanup_params:
                self.test_threshold_var.set(float(cleanup_params["threshold"]))
            if "min_area_px" in cleanup_params:
                self.detector_min_area_var.set(int(cleanup_params["min_area_px"]))
            if "open_kernel" in cleanup_params:
                self.detector_open_kernel_var.set(int(cleanup_params["open_kernel"]))
            if "close_kernel" in cleanup_params:
                self.detector_close_kernel_var.set(int(cleanup_params["close_kernel"]))
        mask_params = params.get("label_mask_cleanup", params.get("mask_open_close", {}))
        if isinstance(mask_params, dict):
            if "kernel" in mask_params:
                self.morph_kernel_var.set(int(mask_params["kernel"]))
            if "iterations" in mask_params:
                self.morph_iterations_var.set(int(mask_params["iterations"]))
        self.on_threshold_change()
        self.refresh_command_text()

    def save_pidnet_params(self):
        output_path = DEFAULT_PARAMS_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(self.current_pidnet_params(), indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
        self.status_var.set(f"Saved PIDNet parameters: {output_path}")

    def load_pidnet_params(self):
        path = filedialog.askopenfilename(
            title="Load PIDNet training parameters",
            initialdir=str(PROJECT_DIR),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            params = json.loads(Path(path).read_text(encoding="utf-8"))
            self.apply_pidnet_params(params)
        except Exception as exc:
            self.status_var.set(f"Could not load PIDNet parameters: {exc}")
            return
        self.status_var.set(f"Loaded PIDNet parameters: {Path(path).name}")
        self.refresh()

    def check_dataset_health(self):
        dataset_dir = Path(self.dataset_var.get())
        lines = [f"Dataset check: {dataset_dir}"]
        total_pairs = 0
        total_empty = 0
        total_heavy = 0
        total_shape_mismatch = 0
        total_unreadable_images = 0
        total_invalid_layers = 0
        split_stems = {}
        for split in DATASET_SPLITS:
            image_dir = dataset_dir / "images" / split
            mask_dir = dataset_dir / "masks" / split
            layer_dir = dataset_dir / "masks_layers" / split
            pairs = dataset_image_mask_pairs(dataset_dir, split)
            total_pairs += len(pairs)
            image_stems = set()
            mask_stems = set()
            if image_dir.exists():
                image_stems = {path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
            if mask_dir.exists():
                mask_stems = {path.stem for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
            layer_stems = set()
            if layer_dir.exists():
                layer_stems = {path.stem for path in layer_dir.iterdir() if path.suffix.lower() == ".npz"}
            split_stems[split] = image_stems
            missing_masks = len(image_stems - mask_stems)
            missing_images = len(mask_stems - image_stems)
            missing_layers = len(image_stems - layer_stems)
            orphan_layers = len(layer_stems - image_stems)
            empty = 0
            heavy = 0
            shape_mismatch = 0
            overlap_pixels = 0
            crossing_pixels = 0
            unreadable_images = 0
            invalid_layers = 0
            positive_frames = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.int64)
            both_endpoint_frames = 0
            for image_path, mask_path in pairs:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    unreadable_images += 1
                    continue
                try:
                    mask, mask_is_multilabel = read_dataset_mask(mask_path, max(1, int(self.cable_count_var.get())))
                except Exception:
                    invalid_layers += 1
                    continue
                if image.shape[:2] != mask.shape[:2]:
                    shape_mismatch += 1
                foreground = mask > 0
                foreground_fraction = float(np.count_nonzero(foreground)) / max(mask.size, 1)
                if foreground_fraction <= 0.0:
                    empty += 1
                if foreground_fraction > 0.25:
                    heavy += 1
                if mask_is_multilabel:
                    bit_counts = np.unpackbits(mask.astype(np.uint16).view(np.uint8), axis=None).reshape(mask.size, 2, 8).sum(axis=(1, 2))
                    overlap_pixels += int(np.count_nonzero(bit_counts > 1))
                    crossing_pixels += int(
                        np.count_nonzero(
                            label_pixels(
                                mask,
                                crossing_label_value(),
                                max(1, int(self.cable_count_var.get())),
                                multilabel=True,
                            )
                        )
                    )
                channel_presence = np.asarray(
                    [
                        np.any(
                            label_pixels(
                                mask,
                                label,
                                max(1, int(self.cable_count_var.get())),
                                multilabel=mask_is_multilabel,
                            )
                        )
                        for label in range(1, OUTPUT_CHANNEL_COUNT + 1)
                    ],
                    dtype=np.int64,
                )
                positive_frames += channel_presence
                both_endpoint_frames += int(np.all(channel_presence[list(ENDPOINT_CHANNELS)] > 0))
            total_empty += empty
            total_heavy += heavy
            total_shape_mismatch += shape_mismatch
            total_unreadable_images += unreadable_images
            total_invalid_layers += invalid_layers
            lines.append(
                f"{split}: pairs={len(pairs)} missing_masks={missing_masks} missing_images={missing_images} "
                f"missing_layers={missing_layers} orphan_layers={orphan_layers} "
                f"empty={empty} very_large_masks={heavy} shape_mismatch={shape_mismatch} "
                f"unreadable_images={unreadable_images} invalid_layers={invalid_layers} "
                f"overlap_px={overlap_pixels} crossing_px={crossing_pixels} "
                f"positive_frames=body:{positive_frames[0]}/{len(pairs)},"
                f"endpoint1:{positive_frames[1]}/{len(pairs)},endpoint2:{positive_frames[2]}/{len(pairs)},"
                f"crossing:{positive_frames[3]}/{len(pairs)} both_endpoints:{both_endpoint_frames}/{len(pairs)}"
            )
        for left_index, left in enumerate(DATASET_SPLITS):
            for right in DATASET_SPLITS[left_index + 1:]:
                split_overlap = sorted(split_stems.get(left, set()) & split_stems.get(right, set()))
                lines.append(f"{left}_{right}_overlap={len(split_overlap)} {split_overlap}")
        metadata_counts, conflicts = dataset_metadata_counts(dataset_dir)
        for split in DATASET_SPLITS:
            lines.append(f"{split}_metadata={metadata_counts[split]}")
        lines.append(f"session_split_conflicts={conflicts}")
        if total_pairs == 0:
            lines.append("Need saved image/mask pairs before training.")
        report = "\n".join(lines) + "\n"
        self.output_text.insert(tk.END, report)
        self.output_text.see(tk.END)
        self.status_var.set(
            f"Dataset check complete: pairs={total_pairs}, empty={total_empty}, "
            f"large={total_heavy}, shape_mismatch={total_shape_mismatch}, "
            f"unreadable={total_unreadable_images}, invalid_layers={total_invalid_layers}."
        )

    def training_command(self):
        command = [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "tools" / "train_pidnet_cable.py"),
            "--dataset",
            str(Path(self.dataset_var.get())),
            "--output",
            str(Path(self.output_var.get())),
            "--epochs",
            str(safe_int(self.epochs_var, 120, min_value=1)),
            "--batch-size",
            str(safe_int(self.batch_var, 8, min_value=1)),
            "--imgsz",
            str(self.imgsz_var.get()).strip() or DEFAULT_IMAGE_SIZE,
            "--base-channels",
            str(safe_int(self.base_channels_var, 24, min_value=1)),
            "--cable-count",
            "2",
            "--device",
            str(self.device_var.get() or "cuda"),
            "--lr",
            f"{safe_float(self.lr_var, 1e-3, min_value=1e-8):.8g}",
            "--weight-decay",
            f"{safe_float(self.weight_decay_var, 1e-4, min_value=0.0):.8g}",
            "--num-workers",
            str(safe_int(self.num_workers_var, 2, min_value=0)),
            "--prefetch-factor",
            str(safe_int(self.prefetch_factor_var, 1, min_value=1)),
            "--endpoint-weight",
            f"{safe_float(self.endpoint_weight_var, 2.0, min_value=0.0):.8g}",
            "--boundary-weight",
            f"{safe_float(self.boundary_weight_var, 0.20, min_value=0.0):.8g}",
            "--crossing-weight",
            f"{safe_float(self.crossing_weight_var, 1.0, min_value=0.0):.8g}",
            "--focal-gamma",
            f"{safe_float(self.focal_gamma_var, 2.0, min_value=0.0):.8g}",
            "--dice-weight",
            f"{safe_float(self.dice_weight_var, 1.0, min_value=0.0):.8g}",
            "--ema-decay",
            f"{safe_float(self.ema_decay_var, 0.995, min_value=0.0, max_value=0.99999):.8g}",
            "--grad-clip",
            f"{safe_float(self.grad_clip_var, 5.0, min_value=0.0):.8g}",
            "--early-stop",
            str(safe_int(self.early_stop_var, 24, min_value=0)),
            "--min-delta",
            f"{safe_float(self.min_delta_var, 1e-4, min_value=0.0):.8g}",
            "--seed",
            str(safe_int(self.seed_var, 17, min_value=0)),
        ]
        for name, enabled in (
            ("amp", self.amp_var.get()),
            ("channels-last", self.channels_last_var.get()),
            ("tf32", self.tf32_var.get()),
            ("compile", self.compile_var.get()),
            ("gpu-augment", self.gpu_augment_var.get()),
            ("deterministic", self.deterministic_var.get()),
            ("verified-only", self.verified_only_var.get()),
            ("evaluate-test", self.evaluate_test_var.get()),
        ):
            command.append(f"--{name}" if bool(enabled) else f"--no-{name}")
        if bool(self.resume_var.get()):
            output = Path(self.output_var.get())
            last_path = output.with_name(f"{output.stem}_last{output.suffix}")
            command.extend(("--resume", str(last_path)))
        elif str(self.init_checkpoint_var.get()).strip():
            command.extend(("--init-checkpoint", str(Path(self.init_checkpoint_var.get()))))
        return command

    def refresh_command_text(self):
        if not hasattr(self, "command_text"):
            return
        cleanup_params = self.live_cleanup_params()
        thresholds = self.preview_thresholds()
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        metadata_counts, conflicts = dataset_metadata_counts(Path(self.dataset_var.get()))
        convention = "Mask layers: 1=generic cable, 2=Cable 1 endpoints, 3=Cable 2 endpoints, 4=RGB crossing proposal."
        text = (
            f"Dataset: train={counts['train']} val={counts['val']} locked_test={counts['test']} | "
            f"verified={metadata_counts['train']['verified']}/{metadata_counts['val']['verified']}/{metadata_counts['test']['verified']} | "
            f"sessions={metadata_counts['train']['sessions']}/{metadata_counts['val']['sessions']}/{metadata_counts['test']['sessions']}\n"
            f"{convention} Layers are stored as independent bits in masks_layers/*.npz.\n"
            "Training uses independent focal+Dice losses for cable, endpoint 1, endpoint 2, and crossing; "
            "the boundary target is the generic cable only.\n"
            f"Session split conflicts: {len(conflicts)}\n"
            f"Checkpoint: {Path(self.output_var.get())}\n"
            f"Live thresholds: cable={thresholds['cable']:.2f}, "
            f"endpoints={thresholds['endpoints'][0]:.2f}/{thresholds['endpoints'][1]:.2f}, "
            f"crossing={thresholds['crossing']:.2f}\n"
            f"Live cleanup preview: min_area={cleanup_params['min_area_px']}, "
            f"open={cleanup_params['open_kernel']}, close={cleanup_params['close_kernel']}\n"
            "Train: use the Train PIDNet-S on CUDA button; the GUI passes these settings to the trainer.\n"
            "PyCharm live run: press Run on main.py with no script parameters. main.py reads config.toml "
            "for cable.count, cable.lengths_m, endpoint_markers.tape_lengths_m, and pidnet.checkpoint.\n"
        )
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert(tk.END, text)

    def start_training(self):
        if self.training_is_running():
            self.status_var.set("Training is already running.")
            return
        self.save_all_labels()
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        metadata_counts, conflicts = dataset_metadata_counts(Path(self.dataset_var.get()))
        required_key = "verified" if bool(self.verified_only_var.get()) else "items"
        if counts["train"] < 1 or metadata_counts["train"][required_key] < 1:
            self.status_var.set("Need at least one saved human-verified training mask before training.")
            return
        if counts["val"] < 1 or metadata_counts["val"][required_key] < 1:
            self.status_var.set("Need at least one saved human-verified validation mask before training.")
            return
        if conflicts:
            self.status_var.set(f"Capture sessions span multiple splits: {sorted(conflicts)}")
            return
        if bool(self.resume_var.get()):
            output = Path(self.output_var.get())
            last_path = output.with_name(f"{output.stem}_last{output.suffix}")
            if not last_path.exists():
                self.status_var.set(f"Resume requested, but the last checkpoint does not exist: {last_path}")
                return
        elif str(self.init_checkpoint_var.get()).strip() and not Path(self.init_checkpoint_var.get()).exists():
            self.status_var.set(f"Initialization checkpoint does not exist: {self.init_checkpoint_var.get()}")
            return
        command = self.training_command()
        self.train_progress_var.set(0.0)
        self.train_summary_var.set("Training starting.")
        self.training_termination = None
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(
            tk.END,
            f"Starting PIDNet-S training on {self.device_var.get() or 'cuda'} from the GUI settings.\n\n",
        )
        try:
            self.train_process = subprocess.Popen(
                command,
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.status_var.set(f"Could not start training: {exc}")
            return
        threading.Thread(target=self._read_training_output, daemon=True).start()
        self.status_var.set("Training started. Watch the output panel for per-head loss and validation score.")

    def _read_training_output(self):
        process = self.train_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.output_queue.put(line)
        code = process.wait()
        # Keep the completion marker at the beginning of the queued line so the
        # progress parser can reliably recognize it and finalize the UI state.
        self.output_queue.put(f"Training finished with exit code {code}.\n")

    def update_training_progress_from_line(self, line):
        marker_line = line.lstrip("\r\n")
        epoch_match = re.search(
            r"epoch\s+(\d+)/(\d+)\s+loss\s+([0-9.eE+-]+)\s+val_score\s+([0-9.eE+-]+)\s+"
            r"val_iou\s+([0-9.eE+-]+)\s+val_dice\s+([0-9.eE+-]+)",
            line,
        )
        if epoch_match:
            epoch = int(epoch_match.group(1))
            total = max(1, int(epoch_match.group(2)))
            loss = float(epoch_match.group(3))
            score = float(epoch_match.group(4))
            iou = float(epoch_match.group(5))
            dice = float(epoch_match.group(6))
            self.train_progress_var.set(100.0 * epoch / total)
            self.train_summary_var.set(
                f"Epoch {epoch}/{total} | loss {loss:.4f} | score {score:.4f} | IoU {iou:.4f} | Dice {dice:.4f}"
            )
            return
        saved_match = re.search(r"saved\s+(.+?)\s+val_score=([0-9.eE+-]+)", line)
        if saved_match:
            self.train_summary_var.set(f"Saved best checkpoint | score {float(saved_match.group(2)):.4f}")
            return
        if marker_line.startswith("Training on "):
            self.train_summary_var.set(marker_line.strip())
            return
        if marker_line.startswith("EARLY STOP:"):
            stopped_match = re.search(r"Stopped at epoch\s+(\d+)", marker_line)
            stopped_epoch = int(stopped_match.group(1)) if stopped_match else None
            self.training_termination = "early_stop"
            suffix = f" at epoch {stopped_epoch}" if stopped_epoch is not None else ""
            self.train_summary_var.set(
                f"Training ended by early stopping{suffix}; see the log for the exact plateau criterion."
            )
            return
        if marker_line.startswith("TRAINING COMPLETE:"):
            self.training_termination = "maximum_epochs"
            self.train_progress_var.set(100.0)
            self.train_summary_var.set(marker_line.strip())
            return
        if marker_line.startswith("Training finished"):
            code_match = re.search(r"exit code\s+(-?\d+)", marker_line)
            code = int(code_match.group(1)) if code_match else 0
            if code == 0:
                if self.training_termination != "early_stop":
                    self.train_progress_var.set(100.0)
                    if self.training_termination != "maximum_epochs":
                        self.train_summary_var.set("Training finished successfully.")
            else:
                self.training_termination = "process_error"
                self.train_summary_var.set(f"Training stopped with exit code {code}.")

    def poll_training_output(self):
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.output_text.insert(tk.END, line)
            self.output_text.see(tk.END)
            self.update_training_progress_from_line(line)
            if line.lstrip("\r\n").startswith("Training finished"):
                self.unload_segmenter()
                self.update_model_status()
                self.status_var.set(self.train_summary_var.get())
        self.root.after(100, self.poll_training_output)

    def stop_training(self):
        if not self.training_is_running():
            self.status_var.set("No training process is running.")
            return
        self.train_process.terminate()
        self.train_summary_var.set("Training stop requested.")
        self.status_var.set("Requested training stop.")

    def training_is_running(self):
        return self.train_process is not None and self.train_process.poll() is None

    def active_frame_key(self):
        item = self.active_item()
        if item is not None:
            return ("frame", int(self.selected_frame_idx), item.get("path"), tuple(item["bgr"].shape))
        bgr = self.active_bgr()
        if bgr is None:
            return ("blank",)
        return ("live", id(bgr), tuple(bgr.shape))

    def cleaned_prediction_mask(self, probability):
        params = self.live_cleanup_params()
        probability = self.cable_probability_union(probability)
        raw = (np.asarray(probability) >= params["threshold"]).astype(np.uint8) * 255
        mask, component_count = apply_binary_cleanup(raw, params)
        return mask > 0, raw > 0, component_count

    def cleaned_prediction_label_mask(self, probability, include_endpoints=True):
        params = self.live_cleanup_params()
        thresholds = self.preview_thresholds()
        probability = np.asarray(probability, dtype=np.float32)
        configured_cable_count = max(1, int(self.cable_count_var.get()))
        expected_channels = OUTPUT_CHANNEL_COUNT
        if probability.ndim != 3 or probability.shape[2] != expected_channels:
            raise ValueError(
                f"PIDNet prediction must have shape HxWx{expected_channels}; got {probability.shape}."
            )
        if not crossing_prediction_label_mode(self.prediction_label_mode):
            raise ValueError(f"Unsupported PIDNet label mode: {self.prediction_label_mode!r}")
        labels = np.zeros(probability.shape[:2], dtype=np.uint8)
        raw_labels = np.zeros(probability.shape[:2], dtype=np.uint8)
        raw = (probability[:, :, 0] >= thresholds["cable"]).astype(np.uint8) * 255
        cleaned, component_count = apply_binary_cleanup(raw, params)
        raw_labels[raw > 0] = 1
        labels[cleaned > 0] = 1
        if bool(include_endpoints):
            for endpoint_index in range(configured_cable_count):
                endpoint_raw = (
                    probability[:, :, 1 + endpoint_index]
                    >= thresholds["endpoints"][endpoint_index]
                )
                endpoint_label = endpoint_label_value(endpoint_index + 1, configured_cable_count)
                raw_labels[endpoint_raw] = endpoint_label
                labels[endpoint_raw] = endpoint_label
        crossing_channel = CROSSING_CHANNEL
        crossing_raw = probability[:, :, crossing_channel] >= thresholds["crossing"]
        crossing_label = crossing_label_value(configured_cable_count)
        raw_labels[crossing_raw] = crossing_label
        labels[crossing_raw] = crossing_label
        return labels, raw_labels, component_count

    def cable_probability_union(self, probability):
        probability = np.asarray(probability, dtype=np.float32)
        if probability.ndim != 3:
            raise ValueError(f"PIDNet prediction must be HxWxC; got {probability.shape}.")
        return np.ascontiguousarray(probability[:, :, 0], dtype=np.float32)

    def endpoint_probability_union(self, probability):
        probability = np.asarray(probability, dtype=np.float32)
        cable_count = max(1, int(self.cable_count_var.get()))
        if probability.ndim != 3 or probability.shape[2] != OUTPUT_CHANNEL_COUNT:
            raise ValueError(f"PIDNet prediction must have {OUTPUT_CHANNEL_COUNT} channels; got {probability.shape}.")
        return np.ascontiguousarray(
            np.max(probability[:, :, 1:1 + cable_count], axis=2),
            dtype=np.float32,
        )

    def endpoint_prediction_mask(self, probability):
        masks = self.endpoint_prediction_masks(probability)
        result = np.zeros(masks[0].shape, dtype=bool)
        for mask in masks:
            result |= mask
        return np.ascontiguousarray(result)

    def endpoint_prediction_masks(self, probability):
        probability = np.asarray(probability, dtype=np.float32)
        cable_count = max(1, int(self.cable_count_var.get()))
        if probability.ndim != 3 or probability.shape[2] != OUTPUT_CHANNEL_COUNT:
            raise ValueError(
                f"PIDNet prediction must have {OUTPUT_CHANNEL_COUNT} channels; got {probability.shape}."
            )
        thresholds = self.preview_thresholds()["endpoints"]
        if cable_count != len(thresholds):
            raise ValueError(
                f"Preview has {len(thresholds)} endpoint thresholds for {cable_count} cables."
            )
        return tuple(
            np.ascontiguousarray(probability[:, :, 1 + endpoint_index] >= threshold)
            for endpoint_index, threshold in enumerate(thresholds)
        )

    def crossing_probability(self, probability):
        probability = np.asarray(probability, dtype=np.float32)
        if not crossing_prediction_label_mode(self.prediction_label_mode):
            raise ValueError(f"Unsupported PIDNet label mode: {self.prediction_label_mode!r}")
        crossing_channel = CROSSING_CHANNEL
        if probability.ndim != 3 or probability.shape[2] != crossing_channel + 1:
            raise ValueError(f"PIDNet crossing prediction is missing from shape {probability.shape}.")
        return np.ascontiguousarray(probability[:, :, crossing_channel], dtype=np.float32)

    def labeled_probability_union(self, probability):
        cable = self.cable_probability_union(probability)
        endpoint = self.endpoint_probability_union(probability)
        if endpoint.shape == cable.shape and np.any(endpoint):
            cable = np.maximum(cable, endpoint)
        crossing = self.crossing_probability(probability)
        if crossing.shape == cable.shape and np.any(crossing):
            cable = np.maximum(cable, crossing)
        return cable

    def segmenter_probability(self, segmenter, bgr, mask=None, mask_is_multilabel=False):
        probability = segmenter.probability_maps(bgr)
        cable_count = max(1, int(self.cable_count_var.get()))
        expected_channels = OUTPUT_CHANNEL_COUNT
        expected_crossing_channel = CROSSING_CHANNEL
        if probability.ndim != 3 or probability.shape[2] != expected_channels:
            raise RuntimeError(
                f"The selected checkpoint must output {expected_channels} channels: "
                f"cable, endpoints_1..endpoints_{cable_count}, and crossing."
            )
        if not bool(getattr(segmenter, "crossing_channels", False)):
            raise RuntimeError("The selected checkpoint does not declare a crossing prediction channel. Retrain it.")
        if int(getattr(segmenter, "crossing_channel", -1)) != expected_crossing_channel:
            raise RuntimeError(
                f"Checkpoint crossing channel is {getattr(segmenter, 'crossing_channel', None)}; "
                f"expected {expected_crossing_channel}."
            )
        return probability

    def load_segmenter(self, force_reload=False):
        checkpoint_path = Path(self.output_var.get())
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist yet: {checkpoint_path}")
        key = (
            self.checkpoint_signature(checkpoint_path),
            str(self.device_var.get() or "cuda"),
        )
        if force_reload or self.segmenter is None or self.segmenter_key != key:
            from cable_pidnet import PidNetSegmenter

            self.segmenter = PidNetSegmenter(
                checkpoint_path,
                device=str(self.device_var.get() or "cuda"),
            )
            self.segmenter_key = key
            self.prediction_probability = None
            self.prediction_frame_key = None
            self.prediction_summary = ""
            self.update_model_status()
        return self.segmenter

    def test_current_frame(self):
        bgr = self.active_bgr()
        if bgr is None:
            self.status_var.set("No frame to test. Open an image or capture from ZED first.")
            return False
        try:
            segmenter = self.load_segmenter()
        except Exception as exc:
            self.status_var.set(f"Could not load PIDNet checkpoint: {exc}")
            return False
        if self.active_item() is None:
            self.live_test_var.set(True)
            updated = self.update_live_prediction_if_needed(force=True)
            self.refresh()
            return bool(updated)
        try:
            item = self.active_item()
            probability = self.segmenter_probability(
                segmenter,
                bgr,
                mask=None if item is None else item["mask"],
                mask_is_multilabel=item is not None,
            )
        except Exception as exc:
            self.status_var.set(f"Could not test PIDNet checkpoint: {exc}")
            return False

        self.prediction_probability = probability
        self.prediction_frame_key = self.active_frame_key()
        self.prediction_label_mode = str(getattr(segmenter, "label_mode", "")).strip().lower()
        predicted, _raw_predicted, component_count = self.cleaned_prediction_mask(probability)
        if item is not None:
            cable_count = max(1, int(self.cable_count_var.get()))
            targets = self.target_masks_by_channel(item["mask"], multilabel=True)
            predictions = self.prediction_masks_by_channel(probability)
            metrics = [binary_mask_metrics(predicted_mask, target) for predicted_mask, target in zip(predictions, targets)]
            self.prediction_summary = (
                f"IoU body/e1/e2/cross {metrics[0]['iou']:.3f}/{metrics[1]['iou']:.3f}/"
                f"{metrics[2]['iou']:.3f}/{metrics[3]['iou']:.3f} | pred px "
                f"{metrics[0]['predicted']}/{metrics[1]['predicted']}/{metrics[2]['predicted']}/"
                f"{metrics[3]['predicted']} | body comp {component_count}"
            )
        else:
            self.prediction_summary = f"cleaned cable pixels {int(np.count_nonzero(predicted))} comp {component_count}"
        self.status_var.set(f"Tested PIDNet checkpoint on current frame: {self.prediction_summary}")
        self.refresh()
        return True

    def predict_current_capture_as_draft(self):
        """Freeze/select a capture, infer once, and enter normal editable annotation mode."""
        if self.training_is_running():
            self.status_var.set("Wait for training to finish before loading its checkpoint for a prediction draft.")
            return False

        # Leaving live preview reveals an already selected capture. If there is
        # no captured frame yet, freeze the current ZED image first.
        if self.active_item() is None and self.frames:
            self.live_test_var.set(False)
            self.prediction_probability = None
            self.prediction_frame_key = None
            self.prediction_label_mode = ""
            self.prediction_summary = ""
        if self.active_item() is None:
            if self.latest_bgr is None:
                self.status_var.set("No current capture is available. Wait for ZED or open an image first.")
                return False
            if not self.capture_current_frame():
                return False

        if not self.test_current_frame():
            return False
        return self.use_prediction_as_draft(run_if_missing=False)

    def use_prediction_as_draft(self, run_if_missing=True):
        item = self.active_item()
        if item is None:
            self.status_var.set("Select a captured or saved frame before applying a prediction draft.")
            return False
        if bool(run_if_missing) and (
            self.prediction_probability is None or self.prediction_frame_key != self.active_frame_key()
        ):
            if not self.test_current_frame():
                return False
        if self.prediction_probability is None or self.prediction_frame_key != self.active_frame_key():
            return False
        predicted_channels = self.prediction_masks_by_channel(self.prediction_probability)
        draft = prediction_channels_to_draft(predicted_channels, item["mask"].shape)
        if np.any(item["mask"]) and not messagebox.askyesno(
            "Replace current annotation",
            "Replace the current paint layers with this model prediction?\n\n"
            "The existing annotation will remain available through Undo.",
        ):
            self.status_var.set("Kept the current annotation; the prediction remains available in Model preview.")
            return False
        draft_metadata = self.model_draft_metadata()
        self.push_undo_state(item)
        item["mask"] = draft
        item["annotation"] = draft_metadata
        self.set_item_verified(item, False)
        item["negative"] = False
        item["dirty"] = True
        self.verified_var.set(False)
        # The prediction has now become the annotation. Clear the heatmap so
        # both canvases show the same editable layers used by manual painting.
        self.live_test_var.set(False)
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_label_mode = ""
        self.prediction_summary = ""
        if hasattr(self, "sidebar_notebook") and hasattr(self, "annotate_tab"):
            self.sidebar_notebook.select(self.annotate_tab)
        self.update_draft_status()
        self.refresh()
        self.status_var.set(
            "Created an unverified editable draft from the current prediction. "
            "Correct it on the source image, review all four layers, then mark Human verified and save."
        )
        return True

    def toggle_live_segmentation(self):
        if self.live_test_var.get():
            try:
                segmenter = self.load_segmenter()
            except Exception as exc:
                self.live_test_var.set(False)
                self.status_var.set(f"Could not start live segmentation: {exc}")
                return
            self.live_test_last_time = 0.0
            if self.latest_bgr is None:
                self.status_var.set("Live segmentation enabled; waiting for a ZED frame.")
                self.refresh()
                return
            self.reset_view()
            self.update_live_prediction_if_needed(force=True)
            self.status_var.set("Live segmentation view enabled.")
            self.refresh()
        else:
            self.clear_prediction()

    def update_live_prediction_if_needed(self, force=False):
        if not self.live_test_var.get():
            return False
        bgr = self.latest_bgr
        if bgr is None:
            return False
        now = time.monotonic()
        if not force and now - self.live_test_last_time < self.live_test_interval_s:
            return False
        try:
            segmenter = self.load_segmenter()
            probability = self.segmenter_probability(segmenter, bgr, mask=None)
        except Exception as exc:
            self.live_test_var.set(False)
            self.status_var.set(f"Live segmentation stopped: {exc}")
            return False

        self.prediction_probability = probability
        self.prediction_frame_key = ("live", tuple(bgr.shape[:2]))
        self.prediction_label_mode = str(getattr(segmenter, "label_mode", "")).strip().lower()
        predicted, _raw_predicted, component_count = self.cleaned_prediction_mask(probability)
        endpoint_pixels = int(np.count_nonzero(self.endpoint_prediction_mask(probability)))
        crossing_probability = self.crossing_probability(probability)
        crossing_pixels = int(np.count_nonzero(
            crossing_probability >= self.preview_thresholds()["crossing"]
        ))
        self.prediction_summary = (
            f"live cable pixels {int(np.count_nonzero(predicted))} endpoints {endpoint_pixels} "
            f"crossing {crossing_pixels} comp {component_count}"
        )
        self.live_test_last_time = now
        return True

    def target_masks_by_channel(self, mask, multilabel):
        cable_count = max(1, int(self.cable_count_var.get()))
        return (
            body_label_mask(mask, cable_count, multilabel=multilabel),
            label_pixels(mask, endpoint_label_value(1), cable_count, multilabel=multilabel),
            label_pixels(mask, endpoint_label_value(2), cable_count, multilabel=multilabel),
            crossing_label_mask(mask, cable_count, multilabel=multilabel),
        )

    def prediction_masks_by_channel(self, probability):
        body, _raw_body, _component_count = self.cleaned_prediction_mask(probability)
        endpoint1, endpoint2 = self.endpoint_prediction_masks(probability)
        crossing = self.crossing_probability(probability) >= self.preview_thresholds()["crossing"]
        return body, endpoint1, endpoint2, crossing

    def test_dataset_split(self, split):
        split = str(split).strip().lower()
        if split == "test" and not messagebox.askyesno(
            "Locked test set",
            "Evaluate the locked test set now? Do not use this result to tune thresholds, training, or architecture.",
        ):
            return
        try:
            segmenter = self.load_segmenter()
        except Exception as exc:
            self.status_var.set(f"Could not load PIDNet checkpoint: {exc}")
            return
        pairs = dataset_image_mask_pairs(Path(self.dataset_var.get()), split, verified_only=True)
        if not pairs:
            self.status_var.set(f"No saved {split} image/mask pairs to test.")
            return

        params = self.live_cleanup_params()
        thresholds = self.preview_thresholds()
        intersection = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.int64)
        union = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.int64)
        predicted_count = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.int64)
        target_count = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.int64)
        tested = 0
        cable_count = max(1, int(self.cable_count_var.get()))
        for image_path, mask_path in pairs:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None:
                continue
            mask, mask_is_multilabel = read_dataset_mask(mask_path, cable_count)
            probability = self.segmenter_probability(segmenter, bgr, mask=mask, mask_is_multilabel=mask_is_multilabel)
            self.prediction_label_mode = str(getattr(segmenter, "label_mode", "")).strip().lower()
            predicted_channels = self.prediction_masks_by_channel(probability)
            target_channels = self.target_masks_by_channel(mask, mask_is_multilabel)
            for channel, (predicted, target) in enumerate(zip(predicted_channels, target_channels)):
                if predicted.shape != target.shape:
                    target = cv2.resize(
                        target.astype(np.uint8),
                        (predicted.shape[1], predicted.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                intersection[channel] += int(np.count_nonzero(predicted & target))
                union[channel] += int(np.count_nonzero(predicted | target))
                predicted_count[channel] += int(np.count_nonzero(predicted))
                target_count[channel] += int(np.count_nonzero(target))
            tested += 1

        if tested == 0:
            self.status_var.set(f"Could not read any saved {split} pairs.")
            return
        iou = intersection / np.maximum(union, 1)
        dice = 2.0 * intersection / np.maximum(predicted_count + target_count, 1)
        line = (
            f"{split} test: {tested} images | thresholds "
            f"{thresholds['cable']:.2f}/"
            f"{thresholds['endpoints'][0]:.2f}/{thresholds['endpoints'][1]:.2f}/"
            f"{thresholds['crossing']:.2f} "
            f"open {params['open_kernel']} close {params['close_kernel']} min_area {params['min_area_px']} "
            f"| body IoU/Dice {iou[0]:.4f}/{dice[0]:.4f} "
            f"| endpoint1 IoU/Dice {iou[1]:.4f}/{dice[1]:.4f} "
            f"| endpoint2 IoU/Dice {iou[2]:.4f}/{dice[2]:.4f} "
            f"| crossing IoU/Dice {iou[3]:.4f}/{dice[3]:.4f}\n"
        )
        self.output_text.insert(tk.END, line)
        self.output_text.see(tk.END)
        self.status_var.set(line.strip())

    def calibrate_validation_thresholds(self):
        try:
            segmenter = self.load_segmenter()
        except Exception as exc:
            self.status_var.set(f"Could not load PIDNet checkpoint: {exc}")
            return
        pairs = dataset_image_mask_pairs(Path(self.dataset_var.get()), "val", verified_only=True)
        if not pairs:
            self.status_var.set("No validation image/mask pairs to calibrate.")
            return
        grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
        intersection = np.zeros((OUTPUT_CHANNEL_COUNT, len(grid)), dtype=np.int64)
        union = np.zeros_like(intersection)
        cleanup = self.live_cleanup_params()
        cable_count = max(1, int(self.cable_count_var.get()))
        tested = 0
        for image_path, mask_path in pairs:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            mask, multilabel = read_dataset_mask(mask_path, cable_count)
            probability = self.segmenter_probability(segmenter, bgr, mask=mask, mask_is_multilabel=multilabel)
            target_channels = self.target_masks_by_channel(mask, multilabel)
            for channel in range(OUTPUT_CHANNEL_COUNT):
                target = target_channels[channel]
                for threshold_index, threshold in enumerate(grid):
                    raw = probability[:, :, channel] >= float(threshold)
                    if channel == 0:
                        cleaned, _component_count = apply_binary_cleanup(
                            raw.astype(np.uint8) * 255,
                            cleanup,
                        )
                        predicted = cleaned > 0
                    else:
                        predicted = raw
                    if target.shape != predicted.shape:
                        resized_target = cv2.resize(
                            target.astype(np.uint8),
                            (predicted.shape[1], predicted.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ) > 0
                    else:
                        resized_target = target
                    intersection[channel, threshold_index] += int(np.count_nonzero(predicted & resized_target))
                    union[channel, threshold_index] += int(np.count_nonzero(predicted | resized_target))
            tested += 1
        if tested == 0:
            self.status_var.set("Could not read validation frames for calibration.")
            return
        iou = intersection / np.maximum(union, 1)
        indices = [int(np.argmax(iou[channel] - 1e-9 * np.abs(grid - 0.5))) for channel in range(OUTPUT_CHANNEL_COUNT)]
        calibrated = [float(grid[index]) for index in indices]
        self.test_threshold_var.set(calibrated[0])
        self.endpoint_threshold_vars[0].set(calibrated[1])
        self.endpoint_threshold_vars[1].set(calibrated[2])
        self.crossing_threshold_var.set(calibrated[3])
        self.on_threshold_change()
        line = (
            f"Validation calibration ({tested} frames): thresholds "
            f"{'/'.join(f'{value:.2f}' for value in calibrated)} | IoU "
            f"{'/'.join(f'{iou[channel, indices[channel]]:.4f}' for channel in range(OUTPUT_CHANNEL_COUNT))}\n"
        )
        self.output_text.insert(tk.END, line)
        self.output_text.see(tk.END)
        self.status_var.set(line.strip())

    def clear_prediction(self):
        self.live_test_var.set(False)
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_label_mode = ""
        self.prediction_summary = ""
        self.refresh()

    def active_prediction_probability(self):
        if self.live_test_var.get() and self.active_item() is None:
            bgr = self.active_bgr()
            if bgr is not None and self.prediction_probability is not None and self.prediction_probability.shape[:2] == bgr.shape[:2]:
                return self.prediction_probability
        if self.prediction_frame_key != self.active_frame_key():
            return None
        return self.prediction_probability

    def refresh(self):
        bgr = self.active_bgr()
        if bgr is None:
            bgr = blank_frame()
        self.ensure_view_center(bgr.shape[:2])
        item = self.active_item()
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8) if item is None else item["mask"]
        probability = self.active_prediction_probability()
        panels = [
            (self.make_overlay_panel(bgr, mask), cv2.INTER_LINEAR),
            (self.make_prediction_panel(bgr, mask, probability), cv2.INTER_LINEAR if probability is not None else cv2.INTER_NEAREST),
        ]
        for canvas, (source, interpolation), name in zip(self.canvases, panels, ("overlay", "mask")):
            image = self.render_view(
                source,
                canvas_width=max(1, canvas.winfo_width()),
                canvas_height=max(1, canvas.winfo_height()),
                interpolation=interpolation,
            )
            photo = bgr_to_photo(image)
            self.photo_refs[name] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.update_draft_status()
        self.update_status_counts()
        self.refresh_command_text()

    def visible_label_values(self):
        mapping = {
            "cable": 1,
            "endpoint1": endpoint_label_value(1),
            "endpoint2": endpoint_label_value(2),
            "crossing": crossing_label_value(),
        }
        return {
            label
            for name, label in mapping.items()
            if bool(self.label_visibility_vars[name].get())
        }

    def filtered_label_mask(self, mask, multilabel=True):
        visible = self.visible_label_values()
        if bool(multilabel):
            result = np.zeros_like(np.asarray(mask, dtype=np.uint16))
            for label in visible:
                bit = np.uint16(label_bit(label))
                result |= np.asarray(mask, dtype=np.uint16) & bit
            return result
        result = np.asarray(mask).copy()
        result[~np.isin(result, list(visible))] = 0
        return result

    def make_overlay_panel(self, bgr, mask):
        panel = bgr.copy()
        draw_label_mask(
            panel,
            self.filtered_label_mask(mask, multilabel=True),
            alpha=0.55,
            cable_count=max(1, int(self.cable_count_var.get())),
            multilabel=True,
        )
        item = self.active_item()
        if item is not None:
            state = item_mask_state(item)
            state_color = (40, 80, 255) if state == "MASK MISSING" else (60, 220, 255) if state.startswith("unsaved") else (80, 230, 80)
            title = f"{Path(item['path']).name if item.get('path') else 'unsaved frame'} | {state}"
            overlay = panel.copy()
            cv2.rectangle(overlay, (0, 0), (panel.shape[1], 44), (0, 0, 0), -1)
            panel = cv2.addWeighted(overlay, 0.72, panel, 0.28, 0.0)
            cv2.putText(panel, title[:110], (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, state_color, 2, cv2.LINE_AA)
        return panel

    def make_mask_panel(self, bgr, mask):
        panel = np.full_like(bgr, 18)
        draw_label_mask(
            panel,
            self.filtered_label_mask(mask, multilabel=True),
            alpha=1.0,
            cable_count=max(1, int(self.cable_count_var.get())),
            multilabel=True,
        )
        if not np.any(mask):
            state = item_mask_state(self.active_item())
            if state == "MASK MISSING":
                cv2.putText(panel, "MASK MISSING", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 80, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    panel,
                    "Paint the mask, then use Save Current Mask - or delete this item.",
                    (28, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif state == "unsaved changes":
                cv2.putText(panel, "EMPTY MASK - UNSAVED", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 255), 2, cv2.LINE_AA)
            elif state == "saved mask":
                cv2.putText(panel, "SAVED BACKGROUND-ONLY MASK", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (80, 230, 80), 2, cv2.LINE_AA)
            else:
                cv2.putText(panel, "Paint cable, endpoints, and crossing labels", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return panel

    def make_prediction_panel(self, bgr, mask, probability):
        if probability is None:
            return self.make_mask_panel(bgr, mask)

        params = self.live_cleanup_params()
        thresholds = self.preview_thresholds()
        cable_count = max(1, int(self.cable_count_var.get()))
        predicted_labels, _raw_predicted_labels, component_count = self.cleaned_prediction_label_mask(probability)
        predicted_channels = self.prediction_masks_by_channel(probability)
        target_channels = self.target_masks_by_channel(mask, multilabel=True)
        selected = str(self.prediction_channel_var.get() or "combined")
        channel_map = {"cable": 0, "endpoint1": 1, "endpoint2": 2, "crossing": 3}
        if selected == "combined":
            heat_probability = self.labeled_probability_union(probability)
        else:
            heat_probability = np.asarray(probability, dtype=np.float32)[:, :, channel_map[selected]]
        heat = cv2.applyColorMap(np.clip(heat_probability * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel = cv2.addWeighted(bgr, 0.72, heat, 0.28, 0.0)
        visible_prediction = self.filtered_label_mask(predicted_labels, multilabel=False)
        draw_label_mask(panel, visible_prediction, alpha=0.94, cable_count=cable_count, outline=True)
        if selected in channel_map:
            channel = channel_map[selected]
            predicted = predicted_channels[channel]
            target = target_channels[channel]
            if target.shape != predicted.shape:
                target = cv2.resize(
                    target.astype(np.uint8),
                    (predicted.shape[1], predicted.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            true_positive = predicted & target
            false_positive = predicted & ~target
            false_negative = ~predicted & target
            draw_stroke_mask(panel, true_positive.astype(np.uint8) * 255, (40, 220, 40), alpha=0.32)
            draw_mask_contours(panel, false_positive.astype(np.uint8) * 255, (40, 40, 255), thickness=2)
            draw_mask_contours(panel, false_negative.astype(np.uint8) * 255, (255, 80, 40), thickness=2)
            metrics = binary_mask_metrics(predicted, target)
            legend = (
                f"{selected}: green TP | red FP | blue FN | "
                f"IoU {metrics['iou']:.3f} Dice {metrics['dice']:.3f}"
            )
        else:
            predicted_body, raw_body, _body_component_count = self.cleaned_prediction_mask(probability)
            removed = raw_body & ~predicted_body
            draw_stroke_mask(panel, removed.astype(np.uint8) * 255, (64, 64, 180), alpha=0.42)
            legend = "combined heatmap | select one channel for TP/FP/FN diagnostics"
        cv2.putText(
            panel,
            f"channel {selected} | thresholds {thresholds['cable']:.2f}/{thresholds['endpoints'][0]:.2f}/"
            f"{thresholds['endpoints'][1]:.2f}/{thresholds['crossing']:.2f} | open {params['open_kernel']} "
            f"close {params['close_kernel']} min {params['min_area_px']}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(panel, legend, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        summary = self.prediction_summary or (
            f"components {component_count} | cleaned body px {int(np.count_nonzero(predicted_channels[0]))}"
        )
        cv2.putText(panel, summary[:80], (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        return panel

    def update_status_counts(self):
        item = self.active_item()
        if item is None:
            bgr = self.active_bgr()
            if bgr is None:
                bgr = blank_frame()
            frame_text = "live preview"
            mask_count = 0
            background_count = int(np.prod(bgr.shape[:2]))
            split = self.split_var.get()
        else:
            frame_text = f"frame {self.selected_frame_idx + 1}/{len(self.frames)}"
            if item.get("path"):
                frame_text += f" | {Path(item['path']).name}"
            mask_count = int(np.count_nonzero(item["mask"]))
            background_count = int(item["mask"].size - mask_count)
            split = item.get("split") or "train"
        mask_state_text = item_mask_state(item)
        label_counts = ""
        if item is not None:
            cable_count = max(1, int(self.cable_count_var.get()))
            counts = [
                f"cable={int(np.count_nonzero(body_label_mask(item['mask'], cable_count, multilabel=True)))}"
            ]
            counts.extend(
                f"endpoints_{index}={int(np.count_nonzero(label_pixels(item['mask'], endpoint_label_value(index), cable_count, multilabel=True)))}"
                for index in range(1, cable_count + 1)
            )
            counts.append(
                f"crossing={int(np.count_nonzero(label_pixels(item['mask'], crossing_label_value(), cable_count, multilabel=True)))}"
            )
            label_counts = " | " + " ".join(counts)
            duplicate_text = (
                f" | near-duplicate={item['near_duplicate_of']} d={item['near_duplicate_distance']:.4f}"
                if item.get("near_duplicate_of") else ""
            )
            label_counts += (
                f" | session={item.get('session_id')} verified={bool(item.get('verified'))} "
                f"negative={bool(item.get('negative'))}"
                f"{duplicate_text}"
            )
        drag_mode = "draw" if self.draw_when_zoomed_var.get() else "pan"
        test_text = ""
        if self.live_test_var.get() and item is None:
            test_text = " | live segmentation"
            if self.prediction_summary:
                test_text += f" | {self.prediction_summary}"
        elif self.prediction_summary:
            test_text = f" | {self.prediction_summary}"
        mode_name = self.active_mode_name()
        self.status_var.set(
            f"{frame_text} | split {split} | mask {mask_state_text} | mode {mode_name} | brush {self.brush_radius_var.get()} px | "
            f"zoom {self.view_zoom:.1f}x ({drag_mode} while zoomed) | labeled px {mask_count} | background px {background_count}"
            f"{label_counts}{test_text}"
        )

    def ensure_view_center(self, image_shape):
        height, width = image_shape[:2]
        self.view_zoom = float(np.clip(self.view_zoom, 1.0, 16.0))
        if self.view_center_xy is None:
            self.view_center_xy = (0.5 * width, 0.5 * height)
        self.view_center_xy = self.clamp_view_center(self.view_center_xy, (height, width))

    def viewport_bounds(self):
        bgr = self.active_bgr()
        if bgr is None:
            bgr = blank_frame()
        height, width = bgr.shape[:2]
        zoom = float(np.clip(self.view_zoom, 1.0, 16.0))
        view_w = max(1.0, width / zoom)
        view_h = max(1.0, height / zoom)
        center_x, center_y = self.view_center_xy if self.view_center_xy is not None else (0.5 * width, 0.5 * height)
        x0 = float(np.clip(center_x - 0.5 * view_w, 0.0, max(0.0, width - view_w)))
        y0 = float(np.clip(center_y - 0.5 * view_h, 0.0, max(0.0, height - view_h)))
        return x0, y0, x0 + view_w, y0 + view_h

    def render_view(self, image, canvas_width, canvas_height, interpolation=cv2.INTER_LINEAR):
        canvas_width = max(1, int(canvas_width))
        canvas_height = max(1, int(canvas_height))
        output = np.full((canvas_height, canvas_width, 3), 24, dtype=np.uint8)
        x0, y0, x1, y1 = self.viewport_bounds()
        ix0 = int(np.clip(np.floor(x0), 0, image.shape[1] - 1))
        iy0 = int(np.clip(np.floor(y0), 0, image.shape[0] - 1))
        ix1 = int(np.clip(np.ceil(x1), ix0 + 1, image.shape[1]))
        iy1 = int(np.clip(np.ceil(y1), iy0 + 1, image.shape[0]))
        crop = image[iy0:iy1, ix0:ix1]
        draw_x, draw_y, draw_w, draw_h = self.display_rect(canvas_width, canvas_height)
        resized = cv2.resize(crop, (draw_w, draw_h), interpolation=interpolation)
        output[draw_y:draw_y + draw_h, draw_x:draw_x + draw_w] = resized
        return output

    def panel_to_image_xy(self, canvas, x, y):
        bgr = self.active_bgr()
        if bgr is None:
            bgr = blank_frame()
        height, width = bgr.shape[:2]
        x0, y0, x1, y1 = self.viewport_bounds()
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        draw_x, draw_y, draw_w, draw_h = self.display_rect(canvas_width, canvas_height)
        local_x = float(np.clip(x, draw_x, draw_x + draw_w - 1)) - float(draw_x)
        local_y = float(np.clip(y, draw_y, draw_y + draw_h - 1)) - float(draw_y)
        image_x = x0 + (local_x / max(draw_w - 1, 1)) * (x1 - x0)
        image_y = y0 + (local_y / max(draw_h - 1, 1)) * (y1 - y0)
        return int(np.clip(round(image_x), 0, width - 1)), int(np.clip(round(image_y), 0, height - 1))

    def point_is_inside_display(self, canvas, x, y):
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        draw_x, draw_y, draw_w, draw_h = self.display_rect(canvas_width, canvas_height)
        return draw_x <= x < draw_x + draw_w and draw_y <= y < draw_y + draw_h

    def zoom_at(self, canvas, x, y, factor):
        bgr = self.active_bgr()
        if bgr is None:
            bgr = blank_frame()
        height, width = bgr.shape[:2]
        self.ensure_view_center((height, width))
        old_x0, old_y0, old_x1, old_y1 = self.viewport_bounds()
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        draw_x, draw_y, draw_w, draw_h = self.display_rect(canvas_width, canvas_height)
        local_x = float(np.clip(x, draw_x, draw_x + draw_w - 1)) - float(draw_x)
        local_y = float(np.clip(y, draw_y, draw_y + draw_h - 1)) - float(draw_y)
        anchor_x = old_x0 + (local_x / max(draw_w - 1, 1)) * (old_x1 - old_x0)
        anchor_y = old_y0 + (local_y / max(draw_h - 1, 1)) * (old_y1 - old_y0)
        self.view_zoom = float(np.clip(self.view_zoom * factor, 1.0, 16.0))
        new_view_w = width / self.view_zoom
        new_view_h = height / self.view_zoom
        frac_x = local_x / max(draw_w - 1, 1)
        frac_y = local_y / max(draw_h - 1, 1)
        center_x = anchor_x + (0.5 - frac_x) * new_view_w
        center_y = anchor_y + (0.5 - frac_y) * new_view_h
        self.view_center_xy = self.clamp_view_center((center_x, center_y), (height, width))
        self.refresh()

    def display_rect(self, canvas_width, canvas_height):
        x0, y0, x1, y1 = self.viewport_bounds()
        view_w = max(1.0, x1 - x0)
        view_h = max(1.0, y1 - y0)
        view_aspect = view_w / view_h
        canvas_aspect = float(canvas_width) / max(float(canvas_height), 1.0)
        if canvas_aspect > view_aspect:
            draw_h = int(canvas_height)
            draw_w = max(1, int(round(draw_h * view_aspect)))
        else:
            draw_w = int(canvas_width)
            draw_h = max(1, int(round(draw_w / view_aspect)))
        draw_w = int(np.clip(draw_w, 1, canvas_width))
        draw_h = int(np.clip(draw_h, 1, canvas_height))
        draw_x = int((canvas_width - draw_w) // 2)
        draw_y = int((canvas_height - draw_h) // 2)
        return draw_x, draw_y, draw_w, draw_h

    def clamp_view_center(self, center_xy, image_shape):
        height, width = image_shape[:2]
        zoom = float(np.clip(self.view_zoom, 1.0, 16.0))
        half_w = 0.5 * width / zoom
        half_h = 0.5 * height / zoom
        min_x = half_w
        max_x = width - half_w
        min_y = half_h
        max_y = height - half_h
        if min_x > max_x:
            min_x = max_x = 0.5 * width
        if min_y > max_y:
            min_y = max_y = 0.5 * height
        return float(np.clip(center_xy[0], min_x, max_x)), float(np.clip(center_xy[1], min_y, max_y))

    def reset_view(self):
        bgr = self.active_bgr()
        if bgr is None:
            bgr = blank_frame()
        height, width = bgr.shape[:2]
        self.view_zoom = 1.0
        self.view_center_xy = (0.5 * width, 0.5 * height)
        self.refresh()

    def should_paint_with_left_drag(self):
        return self.view_zoom <= 1.001 or self.draw_when_zoomed_var.get()

    def on_canvas_configure(self, _event):
        self.root.after_idle(self.refresh)

    def on_wheel(self, event):
        factor = 1.20 if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1.0 / 1.20
        self.zoom_at(event.widget, event.x, event.y, factor)

    def on_left_down(self, event):
        canvas_index = self.canvases.index(event.widget)
        if canvas_index == 0 and self.should_paint_with_left_drag() and self.point_is_inside_display(event.widget, event.x, event.y):
            item = self.active_item()
            if item is not None:
                self.push_undo_state(item)
            self.drawing = True
            self.last_image_xy = self.panel_to_image_xy(event.widget, event.x, event.y)
            self.paint_at(*self.last_image_xy)
            return
        self.start_pan(event)

    def on_left_drag(self, event):
        if self.drawing:
            image_xy = self.panel_to_image_xy(event.widget, event.x, event.y)
            self.paint_line(self.last_image_xy, image_xy)
            self.last_image_xy = image_xy
            return
        if self.panning:
            self.pan_to(event.x, event.y)

    def on_left_up(self, _event):
        self.drawing = False
        self.panning = False
        self.last_image_xy = None
        self.pan_start_xy = None
        self.pan_start_center_xy = None
        self.pan_canvas_size = None

    def on_pan_down(self, event):
        self.start_pan(event)

    def on_pan_drag(self, event):
        self.pan_to(event.x, event.y)

    def on_pan_up(self, _event):
        self.panning = False
        self.pan_start_xy = None
        self.pan_start_center_xy = None
        self.pan_canvas_size = None

    def start_pan(self, event):
        self.panning = True
        self.drawing = False
        self.pan_start_xy = (event.x, event.y)
        self.pan_start_center_xy = self.view_center_xy
        self.pan_canvas_size = (max(1, event.widget.winfo_width()), max(1, event.widget.winfo_height()))

    def pan_to(self, x, y):
        if self.pan_start_xy is None or self.pan_start_center_xy is None:
            return
        bgr = self.active_bgr()
        if bgr is None:
            bgr = blank_frame()
        height, width = bgr.shape[:2]
        x0, y0, x1, y1 = self.viewport_bounds()
        view_w = x1 - x0
        view_h = y1 - y0
        canvas_width, canvas_height = self.pan_canvas_size or (max(1, self.canvases[0].winfo_width()), max(1, self.canvases[0].winfo_height()))
        _draw_x, _draw_y, draw_w, draw_h = self.display_rect(canvas_width, canvas_height)
        dx = float(x - self.pan_start_xy[0]) / max(draw_w, 1) * view_w
        dy = float(y - self.pan_start_xy[1]) / max(draw_h, 1) * view_h
        center_x = self.pan_start_center_xy[0] - dx
        center_y = self.pan_start_center_xy[1] - dy
        self.view_center_xy = self.clamp_view_center((center_x, center_y), (height, width))
        self.refresh()

    def paint_at(self, x, y):
        item = self.active_item()
        if item is None:
            self.status_var.set("Open or capture a frame before painting labels.")
            return
        radius = int(self.brush_radius_var.get())
        brush = np.zeros(item["mask"].shape[:2], dtype=np.uint8)
        cv2.circle(brush, (x, y), radius, 255, -1, cv2.LINE_8)
        self.apply_brush(item, brush)
        self.refresh()

    def paint_line(self, start_xy, end_xy):
        if start_xy is None:
            self.paint_at(*end_xy)
            return
        item = self.active_item()
        if item is None:
            self.status_var.set("Open or capture a frame before painting labels.")
            return
        thickness = max(1, 2 * int(self.brush_radius_var.get()) - 1)
        brush = np.zeros(item["mask"].shape[:2], dtype=np.uint8)
        cv2.line(brush, start_xy, end_xy, 255, thickness, cv2.LINE_8)
        self.apply_brush(item, brush)
        self.refresh()

    def apply_brush(self, item, brush):
        pixels = np.asarray(brush, dtype=np.uint8) > 0
        if not np.any(pixels):
            return
        previous = item["mask"][pixels].copy()
        mode = str(self.mode_var.get())
        if mode == "erase":
            item["mask"][pixels] = 0
        elif mode == "erase_crossing":
            bit = label_bit(crossing_label_value())
            item["mask"][pixels] &= np.uint16(~int(bit) & 0xFFFF)
        else:
            active_label = self.active_label_value()
            item["mask"][pixels] |= label_bit(active_label)
            if active_label != 1:
                item["mask"][pixels] |= label_bit(1)
        if not np.array_equal(previous, item["mask"][pixels]):
            self.mark_item_human_edited(item)

    def request_close(self):
        if not self.confirm_discard_unsaved("close the labeling GUI"):
            return
        self.close()
        self.root.destroy()

    def close(self):
        if self.train_process is not None and self.train_process.poll() is None:
            self.train_process.terminate()
        if self.zed is not None:
            self.zed.close()
            self.zed = None
        if self.left_image is not None:
            self.left_image.free()
            self.left_image = None


def blank_frame():
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        bgr,
        "No frame. Open images or connect ZED, then Capture.",
        (48, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return bgr


def draw_stroke_mask(panel, mask, color, alpha=0.60):
    if mask is None or not np.any(mask):
        return
    pixels = mask > 0
    tint = np.zeros_like(panel)
    tint[:, :] = np.array(color, dtype=np.uint8)
    panel[pixels] = cv2.addWeighted(panel[pixels], 1.0 - alpha, tint[pixels], alpha, 0.0)


def draw_mask_contours(panel, mask, color, thickness=1):
    if mask is None or not np.any(mask):
        return
    mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, color, int(thickness), cv2.LINE_AA)


def draw_label_mask(panel, mask, alpha=0.60, cable_count=None, multilabel=False, outline=False):
    if mask is None or not np.any(mask):
        return
    labels = np.asarray(mask)
    if bool(multilabel):
        label_values = range(1, max_label_value(cable_count or 1) + 1)
    else:
        label_values = sorted(int(value) for value in np.unique(labels) if int(value) > 0)
    for label in label_values:
        pixels = label_pixels(labels, label, cable_count or 1, multilabel=multilabel)
        if not np.any(pixels):
            continue
        color = label_color_bgr(label, cable_count=cable_count)
        draw_stroke_mask(panel, pixels, color, alpha=alpha)
        if outline:
            draw_mask_contours(panel, pixels.astype(np.uint8) * 255, (0, 0, 0), thickness=3)
            draw_mask_contours(panel, pixels.astype(np.uint8) * 255, color, thickness=1)


def bgr_to_photo(bgr):
    rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


def main():
    args = parse_args()
    root = tk.Tk()
    app = PidNetTrainingApp(root, args)
    root.protocol("WM_DELETE_WINDOW", app.request_close)
    try:
        root.mainloop()
    finally:
        app.close()


if __name__ == "__main__":
    main()
