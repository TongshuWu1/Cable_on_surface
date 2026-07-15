import argparse
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
from pidnet_schema import (
    CROSSING_CHANNEL,
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
    (40, 255, 40),     # reserved cable body layer, merged for training
    (40, 255, 40),     # reserved cable body layer, merged for training
    (40, 255, 40),     # reserved cable body layer, merged for training
    (255, 0, 255),     # endpoints_1
    (255, 220, 0),     # endpoints_2
    (80, 80, 255),     # endpoints_3
    (0, 220, 255),     # endpoints_4
)
CROSSING_COLOR_BGR = (0, 255, 255)


def label_color_bgr(label, cable_count=None):
    label = max(1, int(label))
    if cable_count is None:
        return LABEL_COLORS_BGR[(label - 1) % len(LABEL_COLORS_BGR)]
    cable_count = max(1, int(cable_count))
    if label == crossing_label_value(cable_count):
        return CROSSING_COLOR_BGR
    if label <= cable_count:
        return LABEL_COLORS_BGR[(label - 1) % 4]
    endpoint_index = label - cable_count - 1
    return LABEL_COLORS_BGR[4 + (endpoint_index % 4)]


def parse_args():
    parser = argparse.ArgumentParser(description="Label cable masks/endpoints and train the PIDNet-S segmenter.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Live tracker config.toml to tune PIDNet cleanup values.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", action="append", default=[], help="Image path to label. Can be repeated.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--imgsz",
        default=DEFAULT_IMAGE_SIZE,
        help="Training image size. Use 1280x720 for full ZED HD720, or a single value like 512 for square training.",
    )
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--cable-count", type=int, default=2, help="Number of cable endpoint groups to label and train.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--boundary-weight", type=float, default=0.20)
    parser.add_argument("--crossing-weight", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
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
    cable_count=1,
    search_other_splits=True,
):
    bgr = np.asarray(bgr, dtype=np.uint8)
    item = {
        "path": str(Path(path).expanduser().resolve()) if path is not None else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bgr": bgr.copy(),
        "mask": np.zeros(bgr.shape[:2], dtype=np.uint16),
        "cable_count": max(1, int(cable_count)),
        "split": split,
        "saved_image_path": None,
        "saved_mask_path": None,
        "saved_layer_path": None,
        "dirty": False,
    }
    existing_mask, existing_split, existing_mask_path = (
        find_existing_mask(
            path,
            dataset_dir,
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
    cable_count=1,
    preferred_split=None,
    search_other_splits=True,
):
    stem = sanitize_stem(Path(image_path).stem)
    preferred_split = str(preferred_split or "").strip().lower()
    splits = [preferred_split] if preferred_split in ("train", "val") else []
    if search_other_splits:
        splits.extend(split for split in ("train", "val") if split not in splits)
    for split in splits:
        mask_path = Path(dataset_dir) / "masks" / split / f"{stem}.png"
        layer_path = layered_mask_path_from_mask_path(mask_path)
        if layer_path.exists():
            return read_layered_mask(layer_path, cable_count), split, mask_path
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return label_mask_to_bitmask(normalize_label_mask(mask, cable_count), cable_count), split, mask_path
            raise ValueError(f"Could not read mask: {mask_path}")
    return None, None, None


def path_is_within(path, root):
    return Path(path).expanduser().resolve().is_relative_to(Path(root).expanduser().resolve())


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


def normalize_label_mask(mask, cable_count):
    cable_count = max(1, int(cable_count))
    mask = np.asarray(mask, dtype=np.uint8)
    labels = np.zeros(mask.shape[:2], dtype=np.uint8)
    max_label = max_label_value(cable_count)
    for label in range(1, max_label + 1):
        labels[mask == label] = label
    if not np.any(labels) and np.any(mask > 127):
        labels[mask > 127] = 1
    return labels


def label_mask_to_bitmask(mask, cable_count):
    labels = normalize_label_mask(mask, cable_count)
    bitmask = np.zeros(labels.shape[:2], dtype=np.uint16)
    for label in range(1, max_label_value(cable_count) + 1):
        bitmask[labels == label] |= label_bit(label)
    return bitmask


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
    payload = np.load(str(path))
    if "mask" not in payload:
        raise ValueError(f"Layered mask file missing 'mask' array: {path}")
    mask = np.asarray(payload["mask"], dtype=np.uint16)
    if mask.ndim != 2:
        raise ValueError(f"Layered mask must be HxW: {path}")
    valid_bits = np.uint16((1 << max_label_value(cable_count)) - 1)
    return np.ascontiguousarray(mask & valid_bits, dtype=np.uint16)


def label_display_name(label, cable_count):
    label = int(label)
    cable_count = max(1, int(cable_count))
    if label <= 0:
        return "background"
    if label == crossing_label_value(cable_count):
        return "crossing"
    if label <= cable_count:
        return "cable" if label == 1 else f"cable_body_layer{label}"
    endpoint_index = label - cable_count
    return f"endpoints_{endpoint_index}"


def sanitize_stem(stem):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem)).strip("._")
    return stem or "frame"


def label_paths_for_item(item, dataset_dir, index=0):
    split = item.get("split") or "train"
    source_path = item.get("path")
    stem = sanitize_stem(Path(source_path).stem) if source_path else f"frame_{int(index):04d}"
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
    if split not in ("train", "val"):
        raise ValueError(f"Unsupported dataset split: {split!r}")
    item["split"] = split
    image_path, mask_path = label_paths_for_item(item, dataset_dir, index=index)
    layer_path = layered_mask_path_from_mask_path(mask_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(item["mask"], dtype=np.uint16)
    preview_mask = bitmask_to_label_mask(mask, max(1, int(item.get("cable_count", 2))))
    if not cv2.imwrite(str(image_path), np.asarray(item["bgr"], dtype=np.uint8)):
        raise IOError(f"Could not write image: {image_path}")
    if not cv2.imwrite(str(mask_path), preview_mask):
        raise IOError(f"Could not write mask: {mask_path}")
    np.savez_compressed(str(layer_path), mask=mask, cable_count=max(1, int(item.get("cable_count", 2))))
    other_split = "val" if split == "train" else "train"
    other_image = dataset_dir / "images" / other_split / image_path.name
    other_mask = dataset_dir / "masks" / other_split / mask_path.name
    other_layer = dataset_dir / "masks_layers" / other_split / layer_path.name
    for duplicate_path in (other_image, other_mask, other_layer):
        duplicate_path.unlink(missing_ok=True)
    item["saved_image_path"] = str(image_path)
    item["saved_mask_path"] = str(mask_path)
    item["saved_layer_path"] = str(layer_path)
    item["dirty"] = False
    return image_path, mask_path


def count_labeled_pairs(dataset_dir):
    dataset_dir = Path(dataset_dir)
    counts = {}
    for split in ("train", "val"):
        counts[split] = len(dataset_image_mask_pairs(dataset_dir, split))
    return counts


def dataset_image_mask_pairs(dataset_dir, split):
    image_dir = Path(dataset_dir) / "images" / split
    mask_dir = Path(dataset_dir) / "masks" / split
    if not image_dir.exists() or not mask_dir.exists():
        return []
    mask_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def read_dataset_mask(mask_path, cable_count):
    layer_path = layered_mask_path_from_mask_path(mask_path)
    if layer_path.exists():
        return read_layered_mask(layer_path, cable_count), True
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    return normalize_label_mask(mask, cable_count), False


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


def body_label_mask(mask, cable_count, multilabel=False):
    cable_count = max(1, int(cable_count))
    labels = np.asarray(mask)
    body = np.zeros(labels.shape[:2], dtype=bool)
    for label in range(1, cable_count + 1):
        body |= label_pixels(labels, label, cable_count, multilabel=multilabel)
    if not bool(multilabel) and not np.any(body) and np.any(labels > 127):
        body = labels > 127
    return body


def endpoint_label_mask(mask, cable_count, multilabel=False):
    cable_count = max(1, int(cable_count))
    labels = np.asarray(mask)
    endpoint = np.zeros(labels.shape[:2], dtype=bool)
    for label in range(1, cable_count + 1):
        endpoint |= label_pixels(labels, endpoint_label_value(label, cable_count), cable_count, multilabel=multilabel)
    return endpoint


def crossing_label_mask(mask, cable_count, multilabel=False):
    return label_pixels(
        np.asarray(mask),
        crossing_label_value(cable_count),
        cable_count,
        multilabel=multilabel,
    )


def apply_binary_cleanup(mask, params):
    mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if params["open_kernel"] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["open_kernel"], params["open_kernel"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if params["close_kernel"] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["close_kernel"], params["close_kernel"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return remove_small_components(mask, min_area=params["min_area_px"])


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
    return json.dumps(str(value))


def crossing_prediction_label_mode(label_mode):
    return str(label_mode).strip().lower() == PIDNET_LABEL_MODE


class PidNetTrainingApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("PIDNet-S Cable Segmenter Training")
        self.root.geometry("1720x980")
        self.live_config = load_toml_config(args.config)
        pidnet_config = self.live_config.get("pidnet", {})
        detector_config = self.live_config.get("detector", {})

        self.frames = []
        self.selected_frame_idx = -1
        self.latest_bgr = None
        if int(args.cable_count) != 2:
            raise ValueError(f"This project labels exactly two cables; got cable_count={args.cable_count}.")
        self.cable_count_var = tk.IntVar(value=2)
        self.mode_var = tk.StringVar(value="paint_1")
        self.split_var = tk.StringVar(value="train")
        self.brush_radius_var = tk.IntVar(value=4)
        self.draw_when_zoomed_var = tk.BooleanVar(value=False)
        self.epochs_var = tk.IntVar(value=int(args.epochs))
        self.batch_var = tk.IntVar(value=int(args.batch_size))
        self.imgsz_var = tk.StringVar(value=str(args.imgsz))
        self.base_channels_var = tk.IntVar(value=int(args.base_channels))
        self.lr_var = tk.DoubleVar(value=float(args.lr))
        self.weight_decay_var = tk.DoubleVar(value=float(args.weight_decay))
        self.num_workers_var = tk.IntVar(value=int(args.num_workers))
        self.boundary_weight_var = tk.DoubleVar(value=float(args.boundary_weight))
        self.crossing_weight_var = tk.DoubleVar(value=float(args.crossing_weight))
        self.amp_var = tk.BooleanVar(value=bool(args.amp))
        self.device_var = tk.StringVar(value=str(args.device or "cuda"))
        self.test_threshold_var = tk.DoubleVar(value=float(pidnet_config.get("threshold", 0.50)))
        self.threshold_text_var = tk.StringVar(value=f"{float(pidnet_config.get('threshold', 0.50)):.2f}")
        self.live_test_var = tk.BooleanVar(value=False)
        self.morph_kernel_var = tk.IntVar(value=5)
        self.morph_iterations_var = tk.IntVar(value=1)
        self.config_var = tk.StringVar(value=str(Path(args.config)))
        self.detector_min_area_var = tk.IntVar(value=int(detector_config.get("min_area_px", 80)))
        self.detector_open_kernel_var = tk.IntVar(value=int(detector_config.get("open_kernel", 3)))
        self.detector_close_kernel_var = tk.IntVar(value=int(detector_config.get("close_kernel", 5)))
        self.dataset_var = tk.StringVar(value=str(Path(args.dataset)))
        self.output_var = tk.StringVar(value=str(Path(args.output)))
        self.status_var = tk.StringVar(value="Open or capture frames, paint cable masks, save labels, then train and test PIDNet.")
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
        instructions = tk.Frame(self.root, padx=10, pady=8, bg="#f4f4f4")
        instructions.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            instructions,
            text="PIDNet-S cable mask training",
            font=("TkDefaultFont", 13, "bold"),
            bg="#f4f4f4",
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            instructions,
            text="Paint cable, endpoint, and crossing masks, train at 1280x720, then test the checkpoint before using it live.",
            bg="#f4f4f4",
            anchor="w",
            justify=tk.LEFT,
        ).pack(side=tk.TOP, fill=tk.X)

        toolbar = tk.Frame(self.root, padx=8, pady=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(toolbar, text="Open Images", command=self.open_images).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Capture ZED", command=self.capture_current_frame).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Prev", command=self.previous_frame).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Next", command=self.next_frame).pack(side=tk.LEFT, padx=3)

        tk.Label(toolbar, text="  Paint").pack(side=tk.LEFT)
        self.paint_mode_frame = tk.Frame(toolbar)
        self.paint_mode_frame.pack(side=tk.LEFT)
        self.rebuild_paint_mode_buttons()

        label_bar = tk.Frame(self.root, padx=8, pady=4)
        label_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(label_bar, text="Endpoint groups").pack(side=tk.LEFT)
        tk.Spinbox(label_bar, from_=2, to=2, width=3, textvariable=self.cable_count_var, state="readonly").pack(side=tk.LEFT, padx=3)

        tk.Label(label_bar, text="  Split").pack(side=tk.LEFT)
        for text, value in (("Train", "train"), ("Val", "val")):
            tk.Radiobutton(label_bar, text=text, variable=self.split_var, value=value, command=self.set_active_split).pack(side=tk.LEFT)

        tk.Label(label_bar, text="  Brush").pack(side=tk.LEFT)
        tk.Scale(label_bar, from_=1, to=40, orient=tk.HORIZONTAL, variable=self.brush_radius_var, length=110).pack(side=tk.LEFT)
        tk.Checkbutton(label_bar, text="Draw while zoomed", variable=self.draw_when_zoomed_var).pack(side=tk.LEFT, padx=8)
        tk.Button(label_bar, text="Save Current Mask", command=self.save_current_label).pack(side=tk.LEFT, padx=3)
        tk.Button(label_bar, text="Save All Painted Masks", command=self.save_all_labels).pack(side=tk.LEFT, padx=3)
        tk.Button(label_bar, text="Clear Mask", command=self.clear_current_mask).pack(side=tk.LEFT, padx=3)

        mask_tools_bar = tk.Frame(self.root, padx=8, pady=4)
        mask_tools_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(mask_tools_bar, text="Mask cleanup").pack(side=tk.LEFT)
        tk.Label(mask_tools_bar, text="Kernel").pack(side=tk.LEFT, padx=(8, 0))
        tk.Spinbox(mask_tools_bar, from_=1, to=41, increment=2, width=4, textvariable=self.morph_kernel_var).pack(side=tk.LEFT, padx=3)
        tk.Label(mask_tools_bar, text="Iterations").pack(side=tk.LEFT)
        tk.Spinbox(mask_tools_bar, from_=1, to=8, width=3, textvariable=self.morph_iterations_var).pack(side=tk.LEFT, padx=3)
        tk.Button(mask_tools_bar, text="Label Open", command=lambda: self.apply_mask_morph("open")).pack(side=tk.LEFT, padx=3)
        tk.Button(mask_tools_bar, text="Label Close", command=lambda: self.apply_mask_morph("close")).pack(side=tk.LEFT, padx=3)
        tk.Button(mask_tools_bar, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=(12, 3))

        paths = tk.Frame(self.root, padx=8, pady=4)
        paths.pack(side=tk.TOP, fill=tk.X)
        tk.Label(paths, text="Dataset").pack(side=tk.LEFT)
        tk.Entry(paths, textvariable=self.dataset_var, width=46).pack(side=tk.LEFT, padx=3)
        tk.Button(paths, text="Browse", command=self.choose_dataset_dir).pack(side=tk.LEFT, padx=3)
        tk.Label(paths, text="Cable+Endpoint model").pack(side=tk.LEFT, padx=(12, 0))
        tk.Entry(paths, textvariable=self.output_var, width=40).pack(side=tk.LEFT, padx=3)
        tk.Button(paths, text="Browse", command=self.choose_output_path).pack(side=tk.LEFT, padx=3)
        tk.Button(paths, text="Load Model", command=self.load_model_from_button).pack(side=tk.LEFT, padx=3)
        tk.Label(paths, textvariable=self.model_status_var, fg="#444").pack(side=tk.LEFT, padx=8)

        review_bar = tk.Frame(self.root, padx=8, pady=4)
        review_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(review_bar, text="Review dataset").pack(side=tk.LEFT)
        tk.Button(review_bar, text="Open Train Folder", command=lambda: self.open_dataset_split("train")).pack(side=tk.LEFT, padx=3)
        tk.Button(review_bar, text="Open Val Folder", command=lambda: self.open_dataset_split("val")).pack(side=tk.LEFT, padx=3)
        tk.Button(review_bar, text="Open Captures", command=self.open_capture_folder).pack(side=tk.LEFT, padx=3)
        tk.Button(review_bar, text="Delete Current Item", command=self.delete_current_item, fg="#a00000").pack(side=tk.LEFT, padx=(14, 3))
        tk.Label(
            review_bar,
            text="Train/Val loads every image, including images with no mask.",
            fg="#555",
        ).pack(side=tk.LEFT, padx=10)

        train_bar = tk.Frame(self.root, padx=8, pady=4)
        train_bar.pack(side=tk.TOP, fill=tk.X)
        for label, var, width in (
            ("Epochs", self.epochs_var, 6),
            ("Batch", self.batch_var, 5),
            ("Channels", self.base_channels_var, 5),
        ):
            tk.Label(train_bar, text=label).pack(side=tk.LEFT)
            tk.Spinbox(train_bar, from_=1, to=4096, width=width, textvariable=var, command=self.refresh_command_text).pack(side=tk.LEFT, padx=3)
        tk.Label(train_bar, text="Target: cable + endpoints + crossing").pack(side=tk.LEFT, padx=(8, 8))
        tk.Label(train_bar, text="Image WxH").pack(side=tk.LEFT)
        image_entry = tk.Entry(train_bar, textvariable=self.imgsz_var, width=10)
        image_entry.pack(side=tk.LEFT, padx=3)
        image_entry.bind("<KeyRelease>", lambda _event: self.refresh_command_text())
        tk.Label(train_bar, text="Device").pack(side=tk.LEFT)
        tk.Entry(train_bar, textvariable=self.device_var, width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(train_bar, text="Train PIDNet-S on CUDA", command=self.start_training).pack(side=tk.LEFT, padx=8)
        tk.Button(train_bar, text="Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=3)
        tk.Button(train_bar, text="Check Dataset", command=self.check_dataset_health).pack(side=tk.LEFT, padx=8)

        tune_bar = tk.Frame(self.root, padx=8, pady=4)
        tune_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(tune_bar, text="Tune").pack(side=tk.LEFT)
        for label, var, width in (
            ("LR", self.lr_var, 9),
            ("Weight Decay", self.weight_decay_var, 9),
            ("Boundary", self.boundary_weight_var, 6),
            ("Crossing", self.crossing_weight_var, 6),
        ):
            tk.Label(tune_bar, text=label).pack(side=tk.LEFT, padx=(10, 0))
            entry = tk.Entry(tune_bar, textvariable=var, width=width)
            entry.pack(side=tk.LEFT, padx=3)
            entry.bind("<KeyRelease>", lambda _event: self.refresh_command_text())
        tk.Label(tune_bar, text="Workers").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(tune_bar, from_=0, to=32, width=4, textvariable=self.num_workers_var, command=self.refresh_command_text).pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(tune_bar, text="AMP", variable=self.amp_var, command=self.refresh_command_text).pack(side=tk.LEFT, padx=8)
        tk.Button(tune_bar, text="Save Params", command=self.save_pidnet_params).pack(side=tk.LEFT, padx=3)
        tk.Button(tune_bar, text="Load Params", command=self.load_pidnet_params).pack(side=tk.LEFT, padx=3)

        progress_bar = tk.Frame(self.root, padx=8, pady=2)
        progress_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Progressbar(progress_bar, variable=self.train_progress_var, maximum=100.0, length=260).pack(side=tk.LEFT, padx=3)
        tk.Label(progress_bar, textvariable=self.train_summary_var, anchor="w").pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        test_bar = tk.Frame(self.root, padx=8, pady=4)
        test_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(test_bar, text="Test threshold").pack(side=tk.LEFT)
        tk.Scale(
            test_bar,
            from_=0.05,
            to=0.95,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.test_threshold_var,
            length=150,
            command=self.on_threshold_change,
        ).pack(side=tk.LEFT, padx=3)
        tk.Label(test_bar, textvariable=self.threshold_text_var, width=5).pack(side=tk.LEFT)
        tk.Button(test_bar, text="Test Current / Live Preview", command=self.test_current_frame).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(
            test_bar,
            text="Live Segmentation",
            variable=self.live_test_var,
            command=self.toggle_live_segmentation,
        ).pack(side=tk.LEFT, padx=3)
        tk.Button(test_bar, text="Test Val Set", command=lambda: self.test_dataset_split("val")).pack(side=tk.LEFT, padx=3)
        tk.Button(test_bar, text="Test Train Set", command=lambda: self.test_dataset_split("train")).pack(side=tk.LEFT, padx=3)
        tk.Button(test_bar, text="Clear Test View", command=self.clear_prediction).pack(side=tk.LEFT, padx=3)

        cleanup_bar = tk.Frame(self.root, padx=8, pady=4)
        cleanup_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(cleanup_bar, text="Live PIDNet cleanup").pack(side=tk.LEFT)
        tk.Label(cleanup_bar, text="Min area").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(
            cleanup_bar,
            from_=0,
            to=100000,
            width=7,
            textvariable=self.detector_min_area_var,
            command=self.refresh,
        ).pack(side=tk.LEFT, padx=3)
        tk.Label(cleanup_bar, text="Open").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(
            cleanup_bar,
            from_=1,
            to=99,
            increment=2,
            width=4,
            textvariable=self.detector_open_kernel_var,
            command=self.on_live_cleanup_change,
        ).pack(side=tk.LEFT, padx=3)
        tk.Label(cleanup_bar, text="Close").pack(side=tk.LEFT, padx=(10, 0))
        tk.Spinbox(
            cleanup_bar,
            from_=1,
            to=99,
            increment=2,
            width=4,
            textvariable=self.detector_close_kernel_var,
            command=self.on_live_cleanup_change,
        ).pack(side=tk.LEFT, padx=3)
        tk.Button(cleanup_bar, text="Preview Cleanup", command=self.refresh).pack(side=tk.LEFT, padx=8)
        tk.Button(cleanup_bar, text="Save to config.toml", command=self.save_live_cleanup_to_config).pack(side=tk.LEFT, padx=3)
        tk.Label(cleanup_bar, text="Config").pack(side=tk.LEFT, padx=(12, 0))
        tk.Entry(cleanup_bar, textvariable=self.config_var, width=44).pack(side=tk.LEFT, padx=3)
        tk.Button(cleanup_bar, text="Browse", command=self.choose_config_path).pack(side=tk.LEFT, padx=3)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=2)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(1, weight=1)
        titles = ["Captured RGB image", "Painted mask / model test"]
        self.canvases = []
        for column, title in enumerate(titles):
            tk.Label(body, text=title, font=("TkDefaultFont", 11, "bold")).grid(row=0, column=column, sticky="ew")
            canvas = tk.Canvas(body, bg="#181818", highlightthickness=1, highlightbackground="#555")
            canvas.grid(row=1, column=column, sticky="nsew", padx=4)
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

        tk.Label(body, text="Training command and output", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=2, sticky="ew")
        output_frame = tk.Frame(body)
        output_frame.grid(row=1, column=2, sticky="nsew", padx=4)
        output_frame.rowconfigure(1, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.command_text = tk.Text(output_frame, height=5, wrap=tk.WORD)
        self.command_text.grid(row=0, column=0, sticky="ew")
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, bg="#101010", fg="#e8e8e8", insertbackground="#e8e8e8")
        self.output_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        footer = tk.Frame(self.root, padx=8, pady=6)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(footer, textvariable=self.status_var, anchor="w", justify=tk.LEFT).pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            footer,
            text=(
                "Keys: p capture, 1 cable, 2 endpoint, x crossing, c erase crossing, e erase all, "
                "s save current, a save all, t train, "
                "r test current/start live, [/] brush, z reset view. Unpainted pixels train as background. "
                "Mouse wheel zooms; left-drag pans when zoomed unless Draw while zoomed is enabled."
            ),
            anchor="w",
            justify=tk.LEFT,
            fg="#444",
        ).pack(side=tk.TOP, fill=tk.X)

    def rebuild_paint_mode_buttons(self):
        if not hasattr(self, "paint_mode_frame"):
            return
        for child in self.paint_mode_frame.winfo_children():
            child.destroy()
        tk.Radiobutton(
            self.paint_mode_frame,
            text="cable",
            variable=self.mode_var,
            value="paint_1",
            command=self.refresh,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            self.paint_mode_frame,
            text="endpoints_1",
            variable=self.mode_var,
            value="endpoint_1",
            command=self.refresh,
        ).pack(side=tk.LEFT)
        for endpoint_index in range(2, max(1, int(self.cable_count_var.get())) + 1):
            tk.Radiobutton(
                self.paint_mode_frame,
                text=f"endpoints_{endpoint_index}",
                variable=self.mode_var,
                value=f"endpoint_{endpoint_index}",
                command=self.refresh,
            ).pack(side=tk.LEFT)
        tk.Radiobutton(
            self.paint_mode_frame,
            text="crossing",
            variable=self.mode_var,
            value="crossing",
            command=self.refresh,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            self.paint_mode_frame,
            text="Erase crossing",
            variable=self.mode_var,
            value="erase_crossing",
            command=self.refresh,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            self.paint_mode_frame,
            text="Erase all",
            variable=self.mode_var,
            value="erase",
            command=self.refresh,
        ).pack(side=tk.LEFT)

    def _bind_keys(self):
        self.root.bind("1", lambda _event: self.set_mode("paint_1"))
        self.root.bind("2", lambda _event: self.set_mode("endpoint_1"))
        self.root.bind("3", lambda _event: self.set_mode("endpoint_2"))
        self.root.bind("4", lambda _event: self.set_mode("endpoint_3"))
        self.root.bind("5", lambda _event: self.set_mode("endpoint_4"))
        self.root.bind("x", lambda _event: self.set_mode("crossing"))
        self.root.bind("c", lambda _event: self.set_mode("erase_crossing"))
        self.root.bind("e", lambda _event: self.set_mode("erase"))
        self.root.bind("s", lambda _event: self.save_current_label())
        self.root.bind("a", lambda _event: self.save_all_labels())
        self.root.bind("t", lambda _event: self.start_training())
        self.root.bind("r", lambda _event: self.test_current_frame())
        self.root.bind("p", lambda _event: self.capture_current_frame())
        self.root.bind("n", lambda _event: self.next_frame())
        self.root.bind("b", lambda _event: self.previous_frame())
        self.root.bind("z", lambda _event: self.reset_view())
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
        item = self.active_item()
        if item is not None:
            new_split = self.split_var.get()
            if item.get("split") != new_split:
                item["split"] = new_split
                item["dirty"] = True
        self.refresh()

    def on_threshold_change(self, _value=None):
        self.threshold_text_var.set(f"{safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95):.2f}")
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

    def load_live_cleanup_from_config(self, path):
        config = load_toml_config(path)
        pidnet_config = config.get("pidnet", {})
        detector_config = config.get("detector", {})
        if "threshold" in pidnet_config:
            self.test_threshold_var.set(float(pidnet_config["threshold"]))
            self.threshold_text_var.set(f"{float(pidnet_config['threshold']):.2f}")
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
            "threshold": safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95),
            "min_area_px": safe_int(self.detector_min_area_var, 80, min_value=0, max_value=1000000),
            "open_kernel": open_kernel,
            "close_kernel": close_kernel,
        }

    def save_live_cleanup_to_config(self):
        params = self.live_cleanup_params()
        config_path = Path(self.config_var.get() or DEFAULT_CONFIG_PATH)
        try:
            replace_toml_values(
                config_path,
                {
                    ("pidnet", "threshold"): params["threshold"],
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
            f"threshold={params['threshold']:.2f}, min_area={params['min_area_px']}, "
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
            self.load_segmenter(force_reload=True)
        except Exception as exc:
            self.status_var.set(f"Could not load model: {exc}")
            self.update_model_status()
            return
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
        if split not in ("train", "val"):
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
                )
            )

        self.frames = loaded
        self.selected_frame_idx = 0
        self.live_test_var.set(False)
        self.prediction_probability = None
        self.prediction_frame_key = None
        self.prediction_summary = ""
        self.sync_split_from_active()
        self.reset_view()
        missing_count = sum(item_mask_state(item) == "MASK MISSING" for item in loaded)
        self.status_var.set(
            f"Opened {len(loaded)} image(s) from {description}; {missing_count} have no saved mask."
        )
        self.refresh()
        return True

    def delete_current_item(self):
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
                )
            )
            added += 1
        if added and self.selected_frame_idx < 0:
            self.selected_frame_idx = 0
        if added:
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
            return
        capture_dir = Path(self.dataset_var.get()).expanduser().resolve() / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        filename = capture_dir / f"cable_frame_{time.strftime('%Y%m%d_%H%M%S')}_{len(self.frames):04d}.png"
        if not cv2.imwrite(str(filename), self.latest_bgr):
            raise IOError(f"Could not write captured image: {filename}")
        self.frames.append(
            make_frame_item(
                self.latest_bgr,
                path=filename,
                split=self.split_var.get(),
                dataset_dir=Path(self.dataset_var.get()),
                cable_count=max(1, int(self.cable_count_var.get())),
            )
        )
        self.selected_frame_idx = len(self.frames) - 1
        self.reset_view()
        self.status_var.set(
            f"Captured {filename.name}. Paint cable, endpoints, and the independent crossing layer; "
            "the rest is background."
        )
        self.refresh()

    def previous_frame(self):
        if not self.frames:
            return
        self.selected_frame_idx = (self.selected_frame_idx - 1) % len(self.frames)
        self.sync_split_from_active()
        self.reset_view()

    def next_frame(self):
        if not self.frames:
            return
        self.selected_frame_idx = (self.selected_frame_idx + 1) % len(self.frames)
        self.sync_split_from_active()
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
            item["dirty"] = True
        item["mask"].fill(0)
        self.status_var.set("Cleared current mask.")
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
        bit = label_bit(label)
        active_pixels = (item["mask"] & bit) != 0
        item["mask"][active_pixels] = item["mask"][active_pixels] & np.uint16(~int(bit) & 0xFFFF)
        item["mask"][binary > 127] |= bit
        after = int(np.count_nonzero(label_pixels(item["mask"], label, self.cable_count_var.get(), multilabel=True)))
        if not np.array_equal(original_binary, binary):
            item["dirty"] = True
        self.status_var.set(
            f"Applied mask {operation} to {label_display_name(label, self.cable_count_var.get())}: "
            f"px {before} -> {after}."
        )
        self.refresh()

    def save_current_label(self):
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
        item["split"] = self.split_var.get()
        item["cable_count"] = max(1, int(self.cable_count_var.get()))
        image_path, mask_path = save_label_pair(item, Path(self.dataset_var.get()), index=self.selected_frame_idx)
        self.status_var.set(f"Saved {item['split']} label: {image_path.name} and {mask_path.name}")
        self.refresh_command_text()
        return True

    def save_all_labels(self):
        if not self.frames:
            self.status_var.set("No frames to save.")
            return False
        saved = 0
        for index, item in enumerate(self.frames):
            if int(np.count_nonzero(item["mask"])) == 0:
                continue
            item["cable_count"] = max(1, int(self.cable_count_var.get()))
            save_label_pair(item, Path(self.dataset_var.get()), index=index)
            saved += 1
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        skipped = len(self.frames) - saved
        self.status_var.set(
            f"Saved {saved} painted mask(s), skipped {skipped} empty mask(s). "
            f"Dataset now has train={counts['train']} val={counts['val']}."
        )
        self.refresh_command_text()
        return saved > 0

    def current_pidnet_params(self):
        return {
            "version": 1,
            "dataset": str(Path(self.dataset_var.get())),
            "output": str(Path(self.output_var.get())),
            "epochs": safe_int(self.epochs_var, 80, min_value=1),
            "batch_size": safe_int(self.batch_var, 8, min_value=1),
            "imgsz": str(self.imgsz_var.get()).strip() or DEFAULT_IMAGE_SIZE,
            "base_channels": safe_int(self.base_channels_var, 24, min_value=1),
            "cable_count": safe_int(self.cable_count_var, 2, min_value=1, max_value=4),
            "device": str(self.device_var.get() or "cuda"),
            "lr": safe_float(self.lr_var, 1e-3, min_value=1e-8),
            "weight_decay": safe_float(self.weight_decay_var, 1e-4, min_value=0.0),
            "num_workers": safe_int(self.num_workers_var, 4, min_value=0),
            "boundary_weight": safe_float(self.boundary_weight_var, 0.20, min_value=0.0),
            "crossing_weight": safe_float(self.crossing_weight_var, 1.0, min_value=0.0),
            "amp": bool(self.amp_var.get()),
            "test_threshold": safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95),
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
        if "dataset" in params:
            self.dataset_var.set(str(params["dataset"]))
        if "output" in params:
            self.output_var.set(str(params["output"]))
            self.unload_segmenter()
            self.update_model_status()
        for key, var in (
            ("epochs", self.epochs_var),
            ("batch_size", self.batch_var),
            ("base_channels", self.base_channels_var),
            ("cable_count", self.cable_count_var),
            ("num_workers", self.num_workers_var),
        ):
            if key in params:
                var.set(int(params[key]))
        for key, var in (
            ("lr", self.lr_var),
            ("weight_decay", self.weight_decay_var),
            ("boundary_weight", self.boundary_weight_var),
            ("crossing_weight", self.crossing_weight_var),
            ("test_threshold", self.test_threshold_var),
        ):
            if key in params:
                var.set(float(params[key]))
        if "imgsz" in params:
            self.imgsz_var.set(str(params["imgsz"]))
        if "device" in params:
            self.device_var.set(str(params["device"]))
        if "amp" in params:
            self.amp_var.set(bool(params["amp"]))
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
        self.status_var.set(f"Saved PIDNet parameters automatically: {output_path}")

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
        split_stems = {}
        for split in ("train", "val"):
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
            for image_path, mask_path in pairs:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                try:
                    mask, mask_is_multilabel = read_dataset_mask(mask_path, max(1, int(self.cable_count_var.get())))
                except Exception:
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
                                crossing_label_value(max(1, int(self.cable_count_var.get()))),
                                max(1, int(self.cable_count_var.get())),
                                multilabel=True,
                            )
                        )
                    )
            total_empty += empty
            total_heavy += heavy
            total_shape_mismatch += shape_mismatch
            lines.append(
                f"{split}: pairs={len(pairs)} missing_masks={missing_masks} missing_images={missing_images} "
                f"missing_layers={missing_layers} orphan_layers={orphan_layers} "
                f"empty={empty} very_large_masks={heavy} shape_mismatch={shape_mismatch} "
                f"overlap_px={overlap_pixels} crossing_px={crossing_pixels}"
            )
        split_overlap = sorted(split_stems.get("train", set()) & split_stems.get("val", set()))
        lines.append(f"train_val_overlap={len(split_overlap)} {split_overlap}")
        if total_pairs == 0:
            lines.append("Need saved image/mask pairs before training.")
        report = "\n".join(lines) + "\n"
        self.output_text.insert(tk.END, report)
        self.output_text.see(tk.END)
        self.status_var.set(
            f"Dataset check complete: pairs={total_pairs}, empty={total_empty}, "
            f"large={total_heavy}, shape_mismatch={total_shape_mismatch}."
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
            str(safe_int(self.epochs_var, 80, min_value=1)),
            "--batch-size",
            str(safe_int(self.batch_var, 8, min_value=1)),
            "--imgsz",
            str(self.imgsz_var.get()).strip() or DEFAULT_IMAGE_SIZE,
            "--base-channels",
            str(safe_int(self.base_channels_var, 24, min_value=1)),
            "--cable-count",
            str(safe_int(self.cable_count_var, 2, min_value=1, max_value=4)),
            "--device",
            str(self.device_var.get() or "cuda"),
            "--lr",
            f"{safe_float(self.lr_var, 1e-3, min_value=1e-8):.8g}",
            "--weight-decay",
            f"{safe_float(self.weight_decay_var, 1e-4, min_value=0.0):.8g}",
            "--num-workers",
            str(safe_int(self.num_workers_var, 4, min_value=0)),
            "--boundary-weight",
            f"{safe_float(self.boundary_weight_var, 0.20, min_value=0.0):.8g}",
            "--crossing-weight",
            f"{safe_float(self.crossing_weight_var, 1.0, min_value=0.0):.8g}",
        ]
        command.append("--amp" if bool(self.amp_var.get()) else "--no-amp")
        return command

    def refresh_command_text(self):
        if not hasattr(self, "command_text"):
            return
        cleanup_params = self.live_cleanup_params()
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        cable_count = safe_int(self.cable_count_var, 2, min_value=1, max_value=4)
        convention = (
            "Mask labels: 0=background, labels 1.."
            f"{cable_count}=cable body layers merged as cable, labels {cable_count + 1}.."
            f"{2 * cable_count}=separate endpoint layers endpoints_1..endpoints_{cable_count}, "
            f"label {crossing_label_value(cable_count)}=crossing."
        )
        text = (
            f"Dataset: train={counts['train']} val={counts['val']}\n"
            f"{convention} All layers are stored as independent bits in masks_layers/*.npz; masks/*.png is the flat preview mask. Old 255 masks load as cable.\n"
            "Training target: RGB -> cable mask + separated endpoint masks + crossing mask. Crossing uses its own class-balanced loss.\n"
            f"Checkpoint: {Path(self.output_var.get())}\n"
            f"Live cleanup preview: threshold={cleanup_params['threshold']:.2f}, min_area={cleanup_params['min_area_px']}, "
            f"open={cleanup_params['open_kernel']}, close={cleanup_params['close_kernel']}\n"
            "Train: use the Train PIDNet-S on CUDA button; the GUI passes these settings to the trainer.\n"
            "PyCharm live run: press Run on main.py with no script parameters. main.py reads config.toml "
            "for cable.count, cable.lengths_m, endpoint_markers.tape_lengths_m, and pidnet.checkpoint.\n"
        )
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert(tk.END, text)

    def start_training(self):
        if self.train_process is not None and self.train_process.poll() is None:
            self.status_var.set("Training is already running.")
            return
        self.save_all_labels()
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        if counts["train"] < 1:
            self.status_var.set("Need at least one saved training mask before training.")
            return
        command = self.training_command()
        self.train_progress_var.set(0.0)
        self.train_summary_var.set("Training starting.")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Starting PIDNet-S training on CUDA from the GUI settings.\n\n")
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
        self.status_var.set("Training started. Watch the output panel for loss and validation IoU.")

    def _read_training_output(self):
        process = self.train_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.output_queue.put(line)
        code = process.wait()
        self.output_queue.put(f"\nTraining finished with exit code {code}.\n")

    def update_training_progress_from_line(self, line):
        epoch_match = re.search(
            r"epoch\s+(\d+)/(\d+)\s+loss\s+([0-9.eE+-]+)\s+val_iou\s+([0-9.eE+-]+)\s+val_dice\s+([0-9.eE+-]+)",
            line,
        )
        if epoch_match:
            epoch = int(epoch_match.group(1))
            total = max(1, int(epoch_match.group(2)))
            loss = float(epoch_match.group(3))
            iou = float(epoch_match.group(4))
            dice = float(epoch_match.group(5))
            self.train_progress_var.set(100.0 * epoch / total)
            self.train_summary_var.set(f"Epoch {epoch}/{total} | loss {loss:.4f} | val IoU {iou:.4f} | Dice {dice:.4f}")
            return
        saved_match = re.search(r"saved\s+(.+?)\s+val_iou=([0-9.eE+-]+)", line)
        if saved_match:
            self.train_summary_var.set(f"Saved best checkpoint | val IoU {float(saved_match.group(2)):.4f}")
            return
        if line.startswith("Training on "):
            self.train_summary_var.set(line.strip())
            return
        if line.startswith("Training finished"):
            code_match = re.search(r"exit code\s+(-?\d+)", line)
            code = int(code_match.group(1)) if code_match else 0
            if code == 0:
                self.train_progress_var.set(100.0)
                self.train_summary_var.set("Training finished.")
            else:
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
            if line.startswith("Training finished"):
                self.unload_segmenter()
                self.update_model_status()
                self.status_var.set(line.strip())
        self.root.after(100, self.poll_training_output)

    def stop_training(self):
        if self.train_process is None or self.train_process.poll() is not None:
            self.status_var.set("No training process is running.")
            return
        self.train_process.terminate()
        self.train_summary_var.set("Training stop requested.")
        self.status_var.set("Requested training stop.")

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
        raw = (probability[:, :, 0] >= params["threshold"]).astype(np.uint8) * 255
        cleaned, component_count = apply_binary_cleanup(raw, params)
        raw_labels[raw > 0] = 1
        labels[cleaned > 0] = 1
        if bool(include_endpoints):
            for endpoint_index in range(configured_cable_count):
                endpoint_raw = probability[:, :, 1 + endpoint_index] >= params["threshold"]
                endpoint_label = endpoint_label_value(endpoint_index + 1, configured_cable_count)
                raw_labels[endpoint_raw] = endpoint_label
                labels[endpoint_raw] = endpoint_label
        crossing_channel = CROSSING_CHANNEL
        crossing_raw = probability[:, :, crossing_channel] >= params["threshold"]
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
            return
        try:
            segmenter = self.load_segmenter()
        except Exception as exc:
            self.status_var.set(f"Could not load PIDNet checkpoint: {exc}")
            return
        if self.active_item() is None:
            self.live_test_var.set(True)
            self.update_live_prediction_if_needed(force=True)
            self.refresh()
            return
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
            return

        self.prediction_probability = probability
        self.prediction_frame_key = self.active_frame_key()
        self.prediction_label_mode = str(getattr(segmenter, "label_mode", "")).strip().lower()
        predicted, _raw_predicted, component_count = self.cleaned_prediction_mask(probability)
        if item is not None and np.any(item["mask"]):
            cable_count = max(1, int(self.cable_count_var.get()))
            body_target = body_label_mask(item["mask"], cable_count, multilabel=True)
            endpoint_target = endpoint_label_mask(item["mask"], cable_count, multilabel=True)
            endpoint_probability = self.endpoint_probability_union(probability)
            endpoint_text = ""
            if np.any(endpoint_target) and endpoint_probability.shape == endpoint_target.shape:
                endpoint_metrics = binary_mask_metrics(
                    endpoint_probability >= safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95),
                    endpoint_target,
                )
                endpoint_text = f" | end IoU {endpoint_metrics['iou']:.3f}"
            crossing_target = crossing_label_mask(item["mask"], cable_count, multilabel=True)
            crossing_probability = self.crossing_probability(probability)
            crossing_pixels = int(np.count_nonzero(crossing_probability >= safe_float(
                self.test_threshold_var,
                0.50,
                min_value=0.05,
                max_value=0.95,
            )))
            crossing_text = f" | cross {crossing_pixels}px"
            if np.any(crossing_target):
                crossing_metrics = binary_mask_metrics(
                    crossing_probability >= safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95),
                    crossing_target,
                )
                crossing_text = f" | cross IoU {crossing_metrics['iou']:.3f} ({crossing_pixels}px)"
            if np.any(body_target):
                metrics = binary_mask_metrics(predicted, body_target)
                self.prediction_summary = (
                    f"IoU {metrics['iou']:.3f} Dice {metrics['dice']:.3f} | "
                    f"pred {metrics['predicted']} label {metrics['target']} comp {component_count}"
                    f"{endpoint_text}{crossing_text}"
                )
            elif np.any(endpoint_target):
                self.prediction_summary = (
                    f"cleaned cable pixels {int(np.count_nonzero(predicted))} comp {component_count} | "
                    "paint cable bodies for IoU/Dice"
                )
            else:
                self.prediction_summary = (
                    f"cleaned cable pixels {int(np.count_nonzero(predicted))} comp {component_count} | "
                    "paint endpoint labels for endpoint preview"
                )
        else:
            self.prediction_summary = f"cleaned cable pixels {int(np.count_nonzero(predicted))} comp {component_count}"
        self.status_var.set(f"Tested PIDNet checkpoint on current frame: {self.prediction_summary}")
        self.refresh()

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
        endpoint_probability = self.endpoint_probability_union(probability)
        endpoint_pixels = int(np.count_nonzero(endpoint_probability >= safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95)))
        crossing_probability = self.crossing_probability(probability)
        crossing_pixels = int(np.count_nonzero(crossing_probability >= safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95)))
        self.prediction_summary = (
            f"live cable pixels {int(np.count_nonzero(predicted))} endpoints {endpoint_pixels} "
            f"crossing {crossing_pixels} comp {component_count}"
        )
        self.live_test_last_time = now
        return True

    def test_dataset_split(self, split):
        try:
            segmenter = self.load_segmenter()
        except Exception as exc:
            self.status_var.set(f"Could not load PIDNet checkpoint: {exc}")
            return
        pairs = dataset_image_mask_pairs(Path(self.dataset_var.get()), split)
        if not pairs:
            self.status_var.set(f"No saved {split} image/mask pairs to test.")
            return

        params = self.live_cleanup_params()
        intersection = 0
        union = 0
        dice_num = 0
        dice_den = 0
        endpoint_intersection = 0
        endpoint_union = 0
        endpoint_tested = 0
        crossing_intersection = 0
        crossing_union = 0
        crossing_tested = 0
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
            predicted, _raw_predicted, _component_count = self.cleaned_prediction_mask(probability)
            target = body_label_mask(mask, cable_count, multilabel=mask_is_multilabel)
            if predicted.shape != target.shape:
                target = cv2.resize(target.astype(np.uint8), (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            inter = int(np.count_nonzero(predicted & target))
            pred_count = int(np.count_nonzero(predicted))
            target_count = int(np.count_nonzero(target))
            intersection += inter
            union += int(np.count_nonzero(predicted | target))
            dice_num += 2 * inter
            dice_den += pred_count + target_count
            endpoint_target = endpoint_label_mask(mask, cable_count, multilabel=mask_is_multilabel)
            endpoint_probability = self.endpoint_probability_union(probability)
            if np.any(endpoint_target) and endpoint_probability.shape == endpoint_target.shape:
                endpoint_predicted = endpoint_probability >= params["threshold"]
                endpoint_intersection += int(np.count_nonzero(endpoint_predicted & endpoint_target))
                endpoint_union += int(np.count_nonzero(endpoint_predicted | endpoint_target))
                endpoint_tested += 1
            crossing_target = crossing_label_mask(mask, cable_count, multilabel=mask_is_multilabel)
            crossing_probability = self.crossing_probability(probability)
            if np.any(crossing_target) and crossing_probability.shape == crossing_target.shape:
                crossing_predicted = crossing_probability >= params["threshold"]
                crossing_intersection += int(np.count_nonzero(crossing_predicted & crossing_target))
                crossing_union += int(np.count_nonzero(crossing_predicted | crossing_target))
                crossing_tested += 1
            tested += 1

        if tested == 0:
            self.status_var.set(f"Could not read any saved {split} pairs.")
            return
        iou = intersection / max(union, 1)
        dice = dice_num / max(dice_den, 1)
        endpoint_text = ""
        if endpoint_tested > 0:
            endpoint_text = f" | endpoint IoU {endpoint_intersection / max(endpoint_union, 1):.4f}"
        crossing_text = ""
        if crossing_tested > 0:
            crossing_text = f" | crossing IoU {crossing_intersection / max(crossing_union, 1):.4f}"
        line = (
            f"{split} test: {tested} images | threshold {params['threshold']:.2f} "
            f"open {params['open_kernel']} close {params['close_kernel']} min_area {params['min_area_px']} "
            f"| body IoU {iou:.4f} | Dice {dice:.4f}{endpoint_text}{crossing_text}\n"
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
        self.update_status_counts()
        self.refresh_command_text()

    def make_overlay_panel(self, bgr, mask):
        panel = bgr.copy()
        draw_label_mask(
            panel,
            mask,
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
        draw_label_mask(panel, mask, alpha=1.0, cable_count=max(1, int(self.cable_count_var.get())), multilabel=True)
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
        threshold = params["threshold"]
        cable_count = max(1, int(self.cable_count_var.get()))
        predicted_labels, raw_predicted_labels, component_count = self.cleaned_prediction_label_mask(probability)
        predicted, raw_predicted, _cable_component_count = self.cleaned_prediction_mask(probability)
        label = body_label_mask(mask, cable_count, multilabel=True)
        if label.shape != predicted.shape:
            label = cv2.resize(label.astype(np.uint8), (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        probability_union = self.labeled_probability_union(probability)
        heat = cv2.applyColorMap(np.clip(probability_union * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel = cv2.addWeighted(bgr, 0.72, heat, 0.28, 0.0)
        draw_label_mask(panel, predicted_labels, alpha=0.94, cable_count=cable_count, outline=True)
        crossing_predicted = self.crossing_probability(probability) >= threshold
        crossing_target = crossing_label_mask(mask, cable_count, multilabel=True)
        if np.any(label):
            false_positive = predicted & ~label
            false_negative = ~predicted & label
            draw_mask_contours(panel, false_positive.astype(np.uint8) * 255, (40, 40, 255), thickness=2)
            draw_mask_contours(panel, false_negative.astype(np.uint8) * 255, (255, 80, 40), thickness=2)
            if np.any(crossing_target):
                crossing_extra = crossing_predicted & ~crossing_target
                crossing_missed = ~crossing_predicted & crossing_target
                draw_mask_contours(panel, crossing_extra.astype(np.uint8) * 255, (40, 40, 255), thickness=3)
                draw_mask_contours(panel, crossing_missed.astype(np.uint8) * 255, (255, 80, 40), thickness=3)
            legend = "cable green | endpoints magenta/cyan | crossing yellow | red extra | blue missed"
        else:
            removed = raw_predicted & ~predicted
            draw_stroke_mask(panel, removed.astype(np.uint8) * 255, (64, 64, 180), alpha=0.42)
            legend = "cable green | endpoints magenta/cyan | crossing yellow | dim red removed"
        cv2.putText(
            panel,
            f"thr {threshold:.2f} open {params['open_kernel']} close {params['close_kernel']} min {params['min_area_px']}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(panel, legend, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        summary = self.prediction_summary or f"components {component_count} | cleaned px {int(np.count_nonzero(predicted))}"
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
                f"endpoints_{index}={int(np.count_nonzero(label_pixels(item['mask'], endpoint_label_value(index, cable_count), cable_count, multilabel=True)))}"
                for index in range(1, cable_count + 1)
            )
            counts.append(
                f"crossing={int(np.count_nonzero(label_pixels(item['mask'], crossing_label_value(cable_count), cable_count, multilabel=True)))}"
            )
            label_counts = " | " + " ".join(counts)
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
        mode = str(self.mode_var.get())
        if mode == "erase":
            item["mask"][pixels] = 0
        elif mode == "erase_crossing":
            bit = label_bit(crossing_label_value(self.cable_count_var.get()))
            item["mask"][pixels] &= np.uint16(~int(bit) & 0xFFFF)
        else:
            item["mask"][pixels] |= label_bit(self.active_label_value())
        item["dirty"] = True

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
