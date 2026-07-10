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

DEFAULT_DATASET_DIR = PROJECT_DIR / "datasets/two_cable_pidnet"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models/pidnet_two_cable_best.pt"
DEFAULT_IMAGE_SIZE = "1280x720"
DEFAULT_PARAMS_PATH = PROJECT_DIR / "pidnet_two_cable_training_params.json"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"
CAPTURE_DIR = DEFAULT_DATASET_DIR / "captures"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
LABEL_COLORS_BGR = (
    (40, 255, 80),     # cable 1
    (255, 170, 40),    # cable 2
    (80, 220, 255),    # cable 3
    (200, 255, 80),    # cable 4
    (255, 80, 220),    # endpoint 1
    (255, 80, 80),     # endpoint 2
    (180, 80, 255),    # endpoint 3
    (80, 255, 255),    # endpoint 4
)


def parse_args():
    parser = argparse.ArgumentParser(description="Label cable masks and train the PIDNet-S cable segmenter.")
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
    parser.add_argument("--cable-count", type=int, default=2, help="Number of separate cable instance labels to paint/train.")
    parser.add_argument("--endpoint-labels", action=argparse.BooleanOptionalAction, default=True, help="Enable one endpoint label/channel per cable.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--boundary-weight", type=float, default=0.20)
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


def make_frame_item(bgr, path=None, split="train", dataset_dir=DEFAULT_DATASET_DIR, cable_count=1):
    bgr = np.asarray(bgr, dtype=np.uint8)
    item = {
        "path": str(Path(path).expanduser().resolve()) if path is not None else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bgr": bgr.copy(),
        "mask": np.zeros(bgr.shape[:2], dtype=np.uint8),
        "split": split,
        "saved_image_path": None,
        "saved_mask_path": None,
    }
    existing_mask, existing_split = find_existing_mask(path, dataset_dir, cable_count=cable_count) if path is not None else (None, None)
    if existing_mask is not None:
        item["mask"] = existing_mask
        item["split"] = existing_split
    return item


def find_existing_mask(image_path, dataset_dir, cable_count=1):
    stem = sanitize_stem(Path(image_path).stem)
    for split in ("train", "val"):
        mask_path = Path(dataset_dir) / "masks" / split / f"{stem}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return normalize_label_mask(mask, cable_count), split
    return None, None


def normalize_label_mask(mask, cable_count):
    cable_count = max(1, int(cable_count))
    mask = np.asarray(mask, dtype=np.uint8)
    labels = np.zeros(mask.shape[:2], dtype=np.uint8)
    max_label = 2 * cable_count
    for label in range(1, max_label + 1):
        labels[mask == label] = label
    if not np.any(labels) and np.any(mask > 127):
        labels[mask > 127] = 1
    return labels


def endpoint_label_value(cable_index, cable_count):
    return max(1, int(cable_count)) + int(cable_index)


def label_display_name(label, cable_count):
    label = int(label)
    cable_count = max(1, int(cable_count))
    if label <= 0:
        return "background"
    if label <= cable_count:
        return f"cable{label}"
    return f"endpoints_cable{label - cable_count}"


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


def save_label_pair(item, dataset_dir, index=0):
    image_path, mask_path = label_paths_for_item(item, dataset_dir, index=index)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(item["mask"], dtype=np.uint8)
    if not cv2.imwrite(str(image_path), np.asarray(item["bgr"], dtype=np.uint8)):
        raise IOError(f"Could not write image: {image_path}")
    if not cv2.imwrite(str(mask_path), mask):
        raise IOError(f"Could not write mask: {mask_path}")
    item["saved_image_path"] = str(image_path)
    item["saved_mask_path"] = str(mask_path)
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


def body_label_mask(mask, cable_count):
    cable_count = max(1, int(cable_count))
    labels = np.asarray(mask, dtype=np.uint8)
    body = np.zeros(labels.shape[:2], dtype=bool)
    for label in range(1, cable_count + 1):
        body |= labels == label
    if not np.any(body) and np.any(labels > 127):
        body = labels > 127
    return body


def endpoint_label_mask(mask, cable_count):
    cable_count = max(1, int(cable_count))
    labels = np.asarray(mask, dtype=np.uint8)
    endpoint = np.zeros(labels.shape[:2], dtype=bool)
    for label in range(1, cable_count + 1):
        endpoint |= labels == endpoint_label_value(label, cable_count)
    return endpoint


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
        self.cable_count_var = tk.IntVar(value=max(1, int(args.cable_count)))
        self.endpoint_labels_var = tk.BooleanVar(value=bool(args.endpoint_labels))
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
        self.val_split_var = tk.DoubleVar(value=float(args.val_split))
        self.num_workers_var = tk.IntVar(value=int(args.num_workers))
        self.boundary_weight_var = tk.DoubleVar(value=float(args.boundary_weight))
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
            text="Paint separate cable body masks and endpoint masks, train at 1280x720, then test the checkpoint before using it live.",
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

        tk.Label(toolbar, text="  Cables").pack(side=tk.LEFT, padx=(8, 0))
        tk.Spinbox(toolbar, from_=1, to=4, width=3, textvariable=self.cable_count_var, command=self.on_cable_count_changed).pack(side=tk.LEFT, padx=3)

        tk.Label(toolbar, text="  Split").pack(side=tk.LEFT)
        for text, value in (("Train", "train"), ("Val", "val")):
            tk.Radiobutton(toolbar, text=text, variable=self.split_var, value=value, command=self.set_active_split).pack(side=tk.LEFT)

        tk.Label(toolbar, text="  Brush").pack(side=tk.LEFT)
        tk.Scale(toolbar, from_=1, to=40, orient=tk.HORIZONTAL, variable=self.brush_radius_var, length=110).pack(side=tk.LEFT)
        tk.Checkbutton(toolbar, text="Draw while zoomed", variable=self.draw_when_zoomed_var).pack(side=tk.LEFT, padx=8)
        tk.Button(toolbar, text="Save Current Mask", command=self.save_current_label).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Save All Painted Masks", command=self.save_all_labels).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Clear Mask", command=self.clear_current_mask).pack(side=tk.LEFT, padx=3)
        tk.Label(toolbar, text="  Label cleanup").pack(side=tk.LEFT)
        tk.Spinbox(toolbar, from_=1, to=41, increment=2, width=4, textvariable=self.morph_kernel_var).pack(side=tk.LEFT, padx=3)
        tk.Label(toolbar, text="Iter").pack(side=tk.LEFT)
        tk.Spinbox(toolbar, from_=1, to=8, width=3, textvariable=self.morph_iterations_var).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Label Open", command=lambda: self.apply_mask_morph("open")).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Label Close", command=lambda: self.apply_mask_morph("close")).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=3)

        paths = tk.Frame(self.root, padx=8, pady=4)
        paths.pack(side=tk.TOP, fill=tk.X)
        tk.Label(paths, text="Dataset").pack(side=tk.LEFT)
        tk.Entry(paths, textvariable=self.dataset_var, width=58).pack(side=tk.LEFT, padx=3)
        tk.Button(paths, text="Browse", command=self.choose_dataset_dir).pack(side=tk.LEFT, padx=3)
        tk.Label(paths, text="Checkpoint").pack(side=tk.LEFT, padx=(16, 0))
        tk.Entry(paths, textvariable=self.output_var, width=58).pack(side=tk.LEFT, padx=3)
        tk.Button(paths, text="Browse", command=self.choose_output_path).pack(side=tk.LEFT, padx=3)
        tk.Button(paths, text="Load Model", command=self.load_model_from_button).pack(side=tk.LEFT, padx=3)
        tk.Label(paths, textvariable=self.model_status_var, fg="#444").pack(side=tk.LEFT, padx=8)

        train_bar = tk.Frame(self.root, padx=8, pady=4)
        train_bar.pack(side=tk.TOP, fill=tk.X)
        for label, var, width in (
            ("Epochs", self.epochs_var, 6),
            ("Batch", self.batch_var, 5),
            ("Channels", self.base_channels_var, 5),
        ):
            tk.Label(train_bar, text=label).pack(side=tk.LEFT)
            tk.Spinbox(train_bar, from_=1, to=4096, width=width, textvariable=var, command=self.refresh_command_text).pack(side=tk.LEFT, padx=3)
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
            ("Val Split", self.val_split_var, 6),
            ("Boundary", self.boundary_weight_var, 6),
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
        tk.Button(test_bar, text="Test Current / Start Live", command=self.test_current_frame).pack(side=tk.LEFT, padx=8)
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
        titles = ["RGB + cable foreground", "Model test / training mask"]
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
                "Keys: p capture, 1 cable1, 2 endpoints_cable1, 3 cable2, 4 endpoints_cable2, e erase, s save current, a save all, t train, "
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
        cable_count = max(1, int(self.cable_count_var.get()))
        for index in range(cable_count):
            cable_index = index + 1
            value = f"paint_{cable_index}"
            tk.Radiobutton(
                self.paint_mode_frame,
                text=f"cable{cable_index}",
                variable=self.mode_var,
                value=value,
                command=self.refresh,
            ).pack(side=tk.LEFT)
            if bool(self.endpoint_labels_var.get()):
                value = f"endpoint_{cable_index}"
                tk.Radiobutton(
                    self.paint_mode_frame,
                    text=f"endpoints_cable{cable_index}",
                    variable=self.mode_var,
                    value=value,
                    command=self.refresh,
                ).pack(side=tk.LEFT)
        tk.Radiobutton(
            self.paint_mode_frame,
            text="Erase",
            variable=self.mode_var,
            value="erase",
            command=self.refresh,
        ).pack(side=tk.LEFT)

    def _bind_keys(self):
        self.root.bind("1", lambda _event: self.set_mode("paint_1"))
        self.root.bind("2", lambda _event: self.set_mode("endpoint_1"))
        self.root.bind("3", lambda _event: self.set_mode("paint_2"))
        self.root.bind("4", lambda _event: self.set_mode("endpoint_2"))
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
        label_value = self.active_label_value()
        mode_name = "erase" if label_value <= 0 else label_display_name(label_value, self.cable_count_var.get())
        self.status_var.set(f"Paint mode: {mode_name}.")
        self.refresh()

    def on_cable_count_changed(self):
        count = max(1, safe_int(self.cable_count_var, 2, min_value=1, max_value=4))
        self.cable_count_var.set(count)
        self.rebuild_paint_mode_buttons()
        self.set_mode(self.mode_var.get())

    def active_label_value(self):
        mode = str(self.mode_var.get())
        if mode == "erase":
            return 0
        cable_count = max(1, int(self.cable_count_var.get()))
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

    def set_active_split(self):
        item = self.active_item()
        if item is not None:
            item["split"] = self.split_var.get()
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
            self.model_status_var.set(f"model {loaded}: {checkpoint_path.name} ({size_mb:.1f} MB)")
        else:
            self.model_status_var.set("no checkpoint loaded")

    def load_model_from_button(self):
        try:
            self.load_segmenter(force_reload=True)
        except Exception as exc:
            self.status_var.set(f"Could not load PIDNet model: {exc}")
            self.update_model_status()
            return
        self.status_var.set(f"Loaded PIDNet model: {Path(self.output_var.get()).name}")
        self.update_model_status()

    def open_images(self):
        paths = filedialog.askopenfilenames(
            title="Open RGB images to label",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        self.add_image_frames(paths)
        self.refresh()

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
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        filename = CAPTURE_DIR / f"cable_frame_{time.strftime('%Y%m%d_%H%M%S')}_{len(self.frames):04d}.png"
        cv2.imwrite(str(filename), self.latest_bgr)
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
        self.status_var.set(f"Captured {filename.name}. Paint cable bodies and endpoints; the rest is background.")
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
        before = int(np.count_nonzero(item["mask"] == label))
        binary = (item["mask"] == label).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, op_map[operation], kernel, iterations=iterations)
        item["mask"][item["mask"] == label] = 0
        item["mask"][binary > 127] = label
        after = int(np.count_nonzero(item["mask"] == label))
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
            "endpoint_labels": bool(self.endpoint_labels_var.get()),
            "device": str(self.device_var.get() or "cuda"),
            "lr": safe_float(self.lr_var, 1e-3, min_value=1e-8),
            "weight_decay": safe_float(self.weight_decay_var, 1e-4, min_value=0.0),
            "val_split": safe_float(self.val_split_var, 0.15, min_value=0.0, max_value=0.8),
            "num_workers": safe_int(self.num_workers_var, 4, min_value=0),
            "boundary_weight": safe_float(self.boundary_weight_var, 0.20, min_value=0.0),
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
        if "endpoint_labels" in params:
            self.endpoint_labels_var.set(bool(params["endpoint_labels"]))
            self.rebuild_paint_mode_buttons()
        for key, var in (
            ("lr", self.lr_var),
            ("weight_decay", self.weight_decay_var),
            ("val_split", self.val_split_var),
            ("boundary_weight", self.boundary_weight_var),
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
        path = filedialog.asksaveasfilename(
            title="Save PIDNet training parameters",
            initialfile=DEFAULT_PARAMS_PATH.name,
            initialdir=str(PROJECT_DIR),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.current_pidnet_params(), indent=2) + "\n", encoding="utf-8")
        self.status_var.set(f"Saved PIDNet parameters: {output_path.name}")

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
        for split in ("train", "val"):
            image_dir = dataset_dir / "images" / split
            mask_dir = dataset_dir / "masks" / split
            pairs = dataset_image_mask_pairs(dataset_dir, split)
            total_pairs += len(pairs)
            image_stems = set()
            mask_stems = set()
            if image_dir.exists():
                image_stems = {path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
            if mask_dir.exists():
                mask_stems = {path.stem for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
            missing_masks = len(image_stems - mask_stems)
            missing_images = len(mask_stems - image_stems)
            empty = 0
            heavy = 0
            shape_mismatch = 0
            for image_path, mask_path in pairs:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None:
                    continue
                if image.shape[:2] != mask.shape[:2]:
                    shape_mismatch += 1
                foreground_fraction = float(np.count_nonzero(mask > 0)) / max(mask.size, 1)
                if foreground_fraction <= 0.0:
                    empty += 1
                if foreground_fraction > 0.25:
                    heavy += 1
            total_empty += empty
            total_heavy += heavy
            total_shape_mismatch += shape_mismatch
            lines.append(
                f"{split}: pairs={len(pairs)} missing_masks={missing_masks} missing_images={missing_images} "
                f"empty={empty} very_large_masks={heavy} shape_mismatch={shape_mismatch}"
            )
        if total_pairs == 0:
            lines.append("Need saved image/mask pairs before training.")
        if total_pairs > 0 and count_labeled_pairs(dataset_dir)["val"] == 0:
            lines.append("No saved val split found; trainer will split train pairs automatically.")
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
            "--val-split",
            f"{safe_float(self.val_split_var, 0.15, min_value=0.0, max_value=0.8):.8g}",
            "--num-workers",
            str(safe_int(self.num_workers_var, 4, min_value=0)),
            "--boundary-weight",
            f"{safe_float(self.boundary_weight_var, 0.20, min_value=0.0):.8g}",
        ]
        command.append("--amp" if bool(self.amp_var.get()) else "--no-amp")
        command.append("--endpoint-labels" if bool(self.endpoint_labels_var.get()) else "--no-endpoint-labels")
        return command

    def refresh_command_text(self):
        if not hasattr(self, "command_text"):
            return
        cleanup_params = self.live_cleanup_params()
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        cable_count = safe_int(self.cable_count_var, 2, min_value=1, max_value=4)
        if bool(self.endpoint_labels_var.get()):
            label_pairs = []
            for label in range(1, cable_count + 1):
                label_pairs.append(f"{label}=cable{label}")
            for label in range(1, cable_count + 1):
                label_pairs.append(f"{cable_count + label}=endpoints_cable{label}")
            convention = "Mask labels: 0=background, " + ", ".join(label_pairs) + "."
        else:
            convention = f"Mask labels: 0=background, 1..{cable_count}=cable bodies."
        text = (
            f"Dataset: train={counts['train']} val={counts['val']}\n"
            f"{convention} Old 255 masks load as cable 1.\n"
            f"Live cleanup preview: threshold={cleanup_params['threshold']:.2f}, min_area={cleanup_params['min_area_px']}, "
            f"open={cleanup_params['open_kernel']}, close={cleanup_params['close_kernel']}\n"
            "Train: use the Train PIDNet-S on CUDA button; the GUI passes these settings to the trainer.\n"
            "PyCharm live run: press Run on main.py with no script parameters. main.py reads config.toml "
            "for cable.count, endpoint.mode, pidnet.instance_channels, and pidnet.checkpoint.\n"
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
        mask = raw
        if params["open_kernel"] > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["open_kernel"], params["open_kernel"]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if params["close_kernel"] > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["close_kernel"], params["close_kernel"]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask, component_count = remove_small_components(mask, min_area=params["min_area_px"])
        return mask > 0, raw > 0, component_count

    def cable_probability_union(self, probability):
        probability = np.asarray(probability, dtype=np.float32)
        if probability.ndim == 3:
            cable_count = min(max(1, int(self.cable_count_var.get())), probability.shape[2])
            probability = np.max(probability[:, :, :cable_count], axis=2)
        return np.ascontiguousarray(probability, dtype=np.float32)

    def endpoint_probability_union(self, probability):
        probability = np.asarray(probability, dtype=np.float32)
        if probability.ndim != 3:
            return np.zeros(probability.shape[:2], dtype=np.float32)
        cable_count = max(1, int(self.cable_count_var.get()))
        if probability.shape[2] < 2 * cable_count:
            return np.zeros(probability.shape[:2], dtype=np.float32)
        return np.ascontiguousarray(
            np.max(probability[:, :, cable_count:2 * cable_count], axis=2),
            dtype=np.float32,
        )

    def labeled_probability_union(self, probability):
        cable = self.cable_probability_union(probability)
        endpoint = self.endpoint_probability_union(probability)
        if endpoint.shape == cable.shape and np.any(endpoint):
            return np.maximum(cable, endpoint)
        return cable

    def segmenter_probability(self, segmenter, bgr):
        cable_count = max(1, int(self.cable_count_var.get()))
        if cable_count > 1 or int(getattr(segmenter, "output_channels", 1)) > 1:
            return segmenter.probability_maps(bgr)
        return segmenter.probability_map(bgr)

    def load_segmenter(self, force_reload=False):
        checkpoint_path = Path(self.output_var.get())
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist yet: {checkpoint_path}")
        key = (
            self.checkpoint_signature(checkpoint_path),
            str(self.device_var.get() or "cuda"),
            safe_int(self.base_channels_var, 24, min_value=1),
        )
        if force_reload or self.segmenter is None or self.segmenter_key != key:
            from cable_pidnet import PidNetSegmenter

            self.segmenter = PidNetSegmenter(
                checkpoint_path,
                device=str(self.device_var.get() or "cuda"),
                base_channels=safe_int(self.base_channels_var, 24, min_value=1),
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
        if self.active_item() is None:
            self.live_test_var.set(True)
            self.update_live_prediction_if_needed(force=True)
            self.refresh()
            return
        try:
            segmenter = self.load_segmenter()
            probability = self.segmenter_probability(segmenter, bgr)
        except Exception as exc:
            self.status_var.set(f"Could not test PIDNet checkpoint: {exc}")
            return

        self.prediction_probability = probability
        self.prediction_frame_key = self.active_frame_key()
        predicted, _raw_predicted, component_count = self.cleaned_prediction_mask(probability)
        item = self.active_item()
        if item is not None and np.any(item["mask"]):
            cable_count = max(1, int(self.cable_count_var.get()))
            metrics = binary_mask_metrics(predicted, body_label_mask(item["mask"], cable_count))
            endpoint_target = endpoint_label_mask(item["mask"], cable_count)
            endpoint_probability = self.endpoint_probability_union(probability)
            endpoint_text = ""
            if np.any(endpoint_target) and endpoint_probability.shape == endpoint_target.shape:
                endpoint_metrics = binary_mask_metrics(
                    endpoint_probability >= safe_float(self.test_threshold_var, 0.50, min_value=0.05, max_value=0.95),
                    endpoint_target,
                )
                endpoint_text = f" | end IoU {endpoint_metrics['iou']:.3f}"
            self.prediction_summary = (
                f"IoU {metrics['iou']:.3f} Dice {metrics['dice']:.3f} | "
                f"pred {metrics['predicted']} label {metrics['target']} comp {component_count}{endpoint_text}"
            )
        else:
            self.prediction_summary = f"cleaned cable pixels {int(np.count_nonzero(predicted))} comp {component_count}"
        self.status_var.set(f"Tested PIDNet checkpoint on current frame: {self.prediction_summary}")
        self.refresh()

    def toggle_live_segmentation(self):
        if self.live_test_var.get():
            try:
                self.load_segmenter()
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
            probability = self.segmenter_probability(segmenter, bgr)
        except Exception as exc:
            self.live_test_var.set(False)
            self.status_var.set(f"Live segmentation stopped: {exc}")
            return False

        self.prediction_probability = probability
        self.prediction_frame_key = ("live", tuple(bgr.shape[:2]))
        predicted, _raw_predicted, component_count = self.cleaned_prediction_mask(probability)
        self.prediction_summary = f"live cleaned cable pixels {int(np.count_nonzero(predicted))} comp {component_count}"
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
        tested = 0
        cable_count = max(1, int(self.cable_count_var.get()))
        for image_path, mask_path in pairs:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None:
                continue
            probability = self.segmenter_probability(segmenter, bgr)
            predicted, _raw_predicted, _component_count = self.cleaned_prediction_mask(probability)
            target = body_label_mask(mask, cable_count)
            if predicted.shape != target.shape:
                target = cv2.resize(target.astype(np.uint8), (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            inter = int(np.count_nonzero(predicted & target))
            pred_count = int(np.count_nonzero(predicted))
            target_count = int(np.count_nonzero(target))
            intersection += inter
            union += int(np.count_nonzero(predicted | target))
            dice_num += 2 * inter
            dice_den += pred_count + target_count
            endpoint_target = endpoint_label_mask(mask, cable_count)
            endpoint_probability = self.endpoint_probability_union(probability)
            if np.any(endpoint_target) and endpoint_probability.shape == endpoint_target.shape:
                endpoint_predicted = endpoint_probability >= params["threshold"]
                endpoint_intersection += int(np.count_nonzero(endpoint_predicted & endpoint_target))
                endpoint_union += int(np.count_nonzero(endpoint_predicted | endpoint_target))
                endpoint_tested += 1
            tested += 1

        if tested == 0:
            self.status_var.set(f"Could not read any saved {split} pairs.")
            return
        iou = intersection / max(union, 1)
        dice = dice_num / max(dice_den, 1)
        endpoint_text = ""
        if endpoint_tested > 0:
            endpoint_text = f" | endpoint IoU {endpoint_intersection / max(endpoint_union, 1):.4f}"
        line = (
            f"{split} test: {tested} images | threshold {params['threshold']:.2f} "
            f"open {params['open_kernel']} close {params['close_kernel']} min_area {params['min_area_px']} "
            f"| body IoU {iou:.4f} | Dice {dice:.4f}{endpoint_text}\n"
        )
        self.output_text.insert(tk.END, line)
        self.output_text.see(tk.END)
        self.status_var.set(line.strip())

    def clear_prediction(self):
        self.live_test_var.set(False)
        self.prediction_probability = None
        self.prediction_frame_key = None
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
        draw_label_mask(panel, mask, alpha=0.55)
        return panel

    def make_mask_panel(self, bgr, mask):
        panel = np.full_like(bgr, 18)
        draw_label_mask(panel, mask, alpha=1.0)
        if not np.any(mask):
            cv2.putText(panel, "Paint cable1, endpoints_cable1, cable2, endpoints_cable2", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return panel

    def make_prediction_panel(self, bgr, mask, probability):
        if probability is None:
            return self.make_mask_panel(bgr, mask)

        params = self.live_cleanup_params()
        threshold = params["threshold"]
        predicted, raw_predicted, component_count = self.cleaned_prediction_mask(probability)
        cable_count = max(1, int(self.cable_count_var.get()))
        label = body_label_mask(mask, cable_count)
        if label.shape != predicted.shape:
            label = cv2.resize(label.astype(np.uint8), (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        probability_union = self.labeled_probability_union(probability)
        endpoint_probability = self.endpoint_probability_union(probability)
        endpoint_predicted = endpoint_probability >= threshold if endpoint_probability.shape == probability_union.shape else np.zeros_like(predicted)
        heat = cv2.applyColorMap(np.clip(probability_union * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel = cv2.addWeighted(bgr, 0.45, heat, 0.55, 0.0)
        if np.any(label):
            true_positive = predicted & label
            false_positive = predicted & ~label
            false_negative = ~predicted & label
            draw_stroke_mask(panel, true_positive.astype(np.uint8) * 255, (40, 255, 80), alpha=0.68)
            draw_stroke_mask(panel, false_positive.astype(np.uint8) * 255, (40, 40, 255), alpha=0.70)
            draw_stroke_mask(panel, false_negative.astype(np.uint8) * 255, (255, 80, 40), alpha=0.70)
            legend = "green ok | red extra | blue missed"
        else:
            removed = raw_predicted & ~predicted
            draw_stroke_mask(panel, removed.astype(np.uint8) * 255, (64, 64, 180), alpha=0.42)
            draw_stroke_mask(panel, predicted.astype(np.uint8) * 255, (0, 255, 255), alpha=0.48)
            legend = "yellow cleaned cable | dim red removed"
        if np.any(endpoint_predicted):
            draw_stroke_mask(panel, endpoint_predicted.astype(np.uint8) * 255, (255, 80, 220), alpha=0.72)
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
        label_counts = ""
        if item is not None:
            cable_count = max(1, int(self.cable_count_var.get()))
            counts = [
                f"cable{label}={int(np.count_nonzero(item['mask'] == label))}"
                for label in range(1, cable_count + 1)
            ]
            if bool(self.endpoint_labels_var.get()):
                counts.extend(
                    f"endpoints_cable{label}={int(np.count_nonzero(item['mask'] == endpoint_label_value(label, cable_count)))}"
                    for label in range(1, cable_count + 1)
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
        label_value = self.active_label_value()
        mode_name = "erase" if label_value <= 0 else label_display_name(label_value, self.cable_count_var.get())
        self.status_var.set(
            f"{frame_text} | split {split} | mode {mode_name} | brush {self.brush_radius_var.get()} px | "
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
        color = self.active_label_value()
        cv2.circle(item["mask"], (x, y), radius, color, -1, cv2.LINE_8)
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
        color = self.active_label_value()
        cv2.line(item["mask"], start_xy, end_xy, color, thickness, cv2.LINE_8)
        self.refresh()

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


def draw_label_mask(panel, mask, alpha=0.60):
    if mask is None or not np.any(mask):
        return
    labels = np.asarray(mask, dtype=np.uint8)
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        color = LABEL_COLORS_BGR[(label - 1) % len(LABEL_COLORS_BGR)]
        draw_stroke_mask(panel, labels == label, color, alpha=alpha)


def bgr_to_photo(bgr):
    rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


def shell_quote(value):
    value = str(value)
    if not value:
        return "''"
    if re.search(r"[^A-Za-z0-9_./:=+-]", value):
        return "'" + value.replace("'", "'\\''") + "'"
    return value


def main():
    args = parse_args()
    root = tk.Tk()
    app = PidNetTrainingApp(root, args)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    try:
        root.mainloop()
    finally:
        app.close()


if __name__ == "__main__":
    main()
