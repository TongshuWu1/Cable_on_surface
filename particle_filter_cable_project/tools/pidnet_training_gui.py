import argparse
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except Exception:
    sl = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
DEFAULT_DATASET_DIR = PROJECT_DIR / "datasets/cable_pidnet"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models/pidnet_cable_best.pt"
CAPTURE_DIR = DEFAULT_DATASET_DIR / "captures"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(description="Label cable masks and train the PIDNet-S cable segmenter.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", action="append", default=[], help="Image path to label. Can be repeated.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=24)
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


def make_frame_item(bgr, path=None, split="train", dataset_dir=DEFAULT_DATASET_DIR):
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
    existing_mask, existing_split = find_existing_mask(path, dataset_dir) if path is not None else (None, None)
    if existing_mask is not None:
        item["mask"] = existing_mask
        item["split"] = existing_split
    return item


def find_existing_mask(image_path, dataset_dir):
    stem = sanitize_stem(Path(image_path).stem)
    for split in ("train", "val"):
        mask_path = Path(dataset_dir) / "masks" / split / f"{stem}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return (mask > 127).astype(np.uint8) * 255, split
    return None, None


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
    mask = (np.asarray(item["mask"], dtype=np.uint8) > 0).astype(np.uint8) * 255
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


class PidNetTrainingApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("PIDNet-S Cable Segmenter Training")
        self.root.geometry("1720x980")

        self.frames = []
        self.selected_frame_idx = -1
        self.latest_bgr = None
        self.mode_var = tk.StringVar(value="paint")
        self.split_var = tk.StringVar(value="train")
        self.brush_radius_var = tk.IntVar(value=4)
        self.draw_when_zoomed_var = tk.BooleanVar(value=False)
        self.epochs_var = tk.IntVar(value=int(args.epochs))
        self.batch_var = tk.IntVar(value=int(args.batch_size))
        self.imgsz_var = tk.IntVar(value=int(args.imgsz))
        self.base_channels_var = tk.IntVar(value=int(args.base_channels))
        self.device_var = tk.StringVar(value=str(args.device or "cuda"))
        self.test_threshold_var = tk.DoubleVar(value=0.50)
        self.live_test_var = tk.BooleanVar(value=False)
        self.dataset_var = tk.StringVar(value=str(Path(args.dataset)))
        self.output_var = tk.StringVar(value=str(Path(args.output)))
        self.status_var = tk.StringVar(value="Step 1: open images or capture from ZED, then paint only the cable. Unpainted pixels are background.")

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
            text="PIDNet-S cable training workflow",
            font=("TkDefaultFont", 13, "bold"),
            bg="#f4f4f4",
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            instructions,
            text=(
                "1. Open/capture RGB frames.  2. Paint only the cable pixels; every unpainted pixel is background.  "
                "3. Save labels to dataset/images and dataset/masks.  "
                "4. Train PIDNet-S on CUDA.  5. Run main.py with the saved checkpoint."
            ),
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

        tk.Label(toolbar, text="  Paint mode").pack(side=tk.LEFT)
        for text, value in (("Cable", "paint"), ("Erase", "erase")):
            tk.Radiobutton(toolbar, text=text, variable=self.mode_var, value=value, command=self.refresh).pack(side=tk.LEFT)

        tk.Label(toolbar, text="  Split").pack(side=tk.LEFT)
        for text, value in (("Train", "train"), ("Val", "val")):
            tk.Radiobutton(toolbar, text=text, variable=self.split_var, value=value, command=self.set_active_split).pack(side=tk.LEFT)

        tk.Label(toolbar, text="  Brush").pack(side=tk.LEFT)
        tk.Scale(toolbar, from_=1, to=40, orient=tk.HORIZONTAL, variable=self.brush_radius_var, length=110).pack(side=tk.LEFT)
        tk.Checkbutton(toolbar, text="Draw while zoomed", variable=self.draw_when_zoomed_var).pack(side=tk.LEFT, padx=8)
        tk.Button(toolbar, text="Save Current Mask", command=self.save_current_label).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Save All Painted Masks", command=self.save_all_labels).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="Clear Mask", command=self.clear_current_mask).pack(side=tk.LEFT, padx=3)
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
            ("Image", self.imgsz_var, 6),
            ("Channels", self.base_channels_var, 5),
        ):
            tk.Label(train_bar, text=label).pack(side=tk.LEFT)
            tk.Spinbox(train_bar, from_=1, to=4096, width=width, textvariable=var, command=self.refresh_command_text).pack(side=tk.LEFT, padx=3)
        tk.Label(train_bar, text="Device").pack(side=tk.LEFT)
        tk.Entry(train_bar, textvariable=self.device_var, width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(train_bar, text="Train PIDNet-S on CUDA", command=self.start_training).pack(side=tk.LEFT, padx=8)
        tk.Button(train_bar, text="Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=3)

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
            command=lambda _value: self.refresh(),
        ).pack(side=tk.LEFT, padx=3)
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

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
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
                "Keys: p capture, 1 paint, e erase, s save current, a save all, t train, "
                "r test current/start live, [/] brush, z reset view. Only paint cable; unpainted pixels train as background. "
                "Mouse wheel zooms; left-drag pans when zoomed unless Draw while zoomed is enabled."
            ),
            anchor="w",
            justify=tk.LEFT,
            fg="#444",
        ).pack(side=tk.TOP, fill=tk.X)

    def _bind_keys(self):
        self.root.bind("1", lambda _event: self.set_mode("paint"))
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
        self.mode_var.set(mode)
        self.status_var.set(f"Paint mode: {mode}.")
        self.refresh()

    def set_active_split(self):
        item = self.active_item()
        if item is not None:
            item["split"] = self.split_var.get()
        self.refresh()

    def adjust_brush(self, delta):
        self.brush_radius_var.set(int(np.clip(self.brush_radius_var.get() + delta, 1, 40)))
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
            self.frames.append(make_frame_item(bgr, path=path, split=self.split_var.get(), dataset_dir=Path(self.dataset_var.get())))
            added += 1
        if added and self.selected_frame_idx < 0:
            self.selected_frame_idx = 0
        if added:
            self.reset_view()
            self.status_var.set(f"Loaded {added} image(s). Paint cable pixels, then save labels.")

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
        self.frames.append(make_frame_item(self.latest_bgr, path=filename, split=self.split_var.get(), dataset_dir=Path(self.dataset_var.get())))
        self.selected_frame_idx = len(self.frames) - 1
        self.reset_view()
        self.status_var.set(f"Captured {filename.name}. Paint cable pixels only; the rest is background.")
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

    def training_command(self):
        return [
            sys.executable,
            str(PROJECT_DIR / "tools" / "train_pidnet_cable.py"),
            "--dataset",
            str(Path(self.dataset_var.get())),
            "--output",
            str(Path(self.output_var.get())),
            "--epochs",
            str(int(self.epochs_var.get())),
            "--batch-size",
            str(int(self.batch_var.get())),
            "--imgsz",
            str(int(self.imgsz_var.get())),
            "--base-channels",
            str(int(self.base_channels_var.get())),
            "--device",
            str(self.device_var.get() or "cuda"),
        ]

    def refresh_command_text(self):
        if not hasattr(self, "command_text"):
            return
        command = " ".join(shell_quote(part) for part in self.training_command())
        live = (
            f"{shell_quote(sys.executable)} {shell_quote(str(PROJECT_DIR / 'main.py'))} "
            f"--neural-detector-checkpoint {shell_quote(str(Path(self.output_var.get())))} "
            f"--neural-detector-device {shell_quote(str(self.device_var.get() or 'cuda'))}"
        )
        counts = count_labeled_pairs(Path(self.dataset_var.get()))
        text = (
            f"Dataset: train={counts['train']} val={counts['val']}\n"
            "Mask convention: white pixels are cable; all black/unpainted pixels are background.\n"
            f"Train:\n{command}\n\n"
            f"Run live after training:\n{live}\n"
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
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Starting PIDNet-S training on CUDA...\n")
        self.output_text.insert(tk.END, " ".join(shell_quote(part) for part in command) + "\n\n")
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

    def poll_training_output(self):
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.output_text.insert(tk.END, line)
            self.output_text.see(tk.END)
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
        self.status_var.set("Requested training stop.")

    def active_frame_key(self):
        item = self.active_item()
        if item is not None:
            return ("frame", int(self.selected_frame_idx), item.get("path"), tuple(item["bgr"].shape))
        bgr = self.active_bgr()
        if bgr is None:
            return ("blank",)
        return ("live", id(bgr), tuple(bgr.shape))

    def load_segmenter(self, force_reload=False):
        checkpoint_path = Path(self.output_var.get())
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist yet: {checkpoint_path}")
        key = (
            self.checkpoint_signature(checkpoint_path),
            str(self.device_var.get() or "cuda"),
            int(self.base_channels_var.get()),
        )
        if force_reload or self.segmenter is None or self.segmenter_key != key:
            from cable_pidnet import PidNetSegmenter

            self.segmenter = PidNetSegmenter(
                checkpoint_path,
                device=str(self.device_var.get() or "cuda"),
                base_channels=int(self.base_channels_var.get()),
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
            probability = segmenter.probability_map(bgr)
        except Exception as exc:
            self.status_var.set(f"Could not test PIDNet checkpoint: {exc}")
            return

        self.prediction_probability = probability
        self.prediction_frame_key = self.active_frame_key()
        threshold = float(self.test_threshold_var.get())
        predicted = probability >= threshold
        item = self.active_item()
        if item is not None and np.any(item["mask"]):
            metrics = binary_mask_metrics(predicted, item["mask"] > 0)
            self.prediction_summary = (
                f"IoU {metrics['iou']:.3f} Dice {metrics['dice']:.3f} | "
                f"pred {metrics['predicted']} label {metrics['target']}"
            )
        else:
            self.prediction_summary = f"predicted cable pixels {int(np.count_nonzero(predicted))}"
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
            probability = segmenter.probability_map(bgr)
        except Exception as exc:
            self.live_test_var.set(False)
            self.status_var.set(f"Live segmentation stopped: {exc}")
            return False

        self.prediction_probability = probability
        self.prediction_frame_key = ("live", tuple(bgr.shape[:2]))
        threshold = float(self.test_threshold_var.get())
        predicted = probability >= threshold
        self.prediction_summary = f"live predicted cable pixels {int(np.count_nonzero(predicted))}"
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

        threshold = float(self.test_threshold_var.get())
        intersection = 0
        union = 0
        dice_num = 0
        dice_den = 0
        tested = 0
        for image_path, mask_path in pairs:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None:
                continue
            probability = segmenter.probability_map(bgr)
            predicted = probability >= threshold
            target = mask > 127
            if predicted.shape != target.shape:
                target = cv2.resize(target.astype(np.uint8), (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            inter = int(np.count_nonzero(predicted & target))
            pred_count = int(np.count_nonzero(predicted))
            target_count = int(np.count_nonzero(target))
            intersection += inter
            union += int(np.count_nonzero(predicted | target))
            dice_num += 2 * inter
            dice_den += pred_count + target_count
            tested += 1

        if tested == 0:
            self.status_var.set(f"Could not read any saved {split} pairs.")
            return
        iou = intersection / max(union, 1)
        dice = dice_num / max(dice_den, 1)
        line = f"{split} test: {tested} images | threshold {threshold:.2f} | IoU {iou:.4f} | Dice {dice:.4f}\n"
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
            if bgr is not None and self.prediction_probability is not None and self.prediction_probability.shape == bgr.shape[:2]:
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
        draw_stroke_mask(panel, mask, (40, 255, 80), alpha=0.55)
        return panel

    def make_mask_panel(self, bgr, mask):
        panel = np.full_like(bgr, 18)
        panel[mask > 0] = (255, 255, 255)
        if not np.any(mask):
            cv2.putText(panel, "Paint cable only; black is background", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return panel

    def make_prediction_panel(self, bgr, mask, probability):
        if probability is None:
            return self.make_mask_panel(bgr, mask)

        threshold = float(self.test_threshold_var.get())
        predicted = probability >= threshold
        label = np.asarray(mask, dtype=np.uint8) > 0
        if label.shape != predicted.shape:
            label = cv2.resize(label.astype(np.uint8), (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        heat = cv2.applyColorMap(np.clip(probability * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
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
            draw_stroke_mask(panel, predicted.astype(np.uint8) * 255, (0, 255, 255), alpha=0.48)
            legend = "yellow predicted cable"
        cv2.putText(panel, f"threshold {threshold:.2f}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, legend, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if self.prediction_summary:
            cv2.putText(panel, self.prediction_summary[:80], (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
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
        drag_mode = "draw" if self.draw_when_zoomed_var.get() else "pan"
        test_text = ""
        if self.live_test_var.get() and item is None:
            test_text = " | live segmentation"
            if self.prediction_summary:
                test_text += f" | {self.prediction_summary}"
        elif self.prediction_summary:
            test_text = f" | {self.prediction_summary}"
        self.status_var.set(
            f"{frame_text} | split {split} | mode {self.mode_var.get()} | brush {self.brush_radius_var.get()} px | "
            f"zoom {self.view_zoom:.1f}x ({drag_mode} while zoomed) | cable px {mask_count} | background px {background_count}"
            f"{test_text}"
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
        color = 0 if self.mode_var.get() == "erase" else 255
        cv2.circle(item["mask"], (x, y), radius, color, -1, cv2.LINE_AA)
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
        color = 0 if self.mode_var.get() == "erase" else 255
        cv2.line(item["mask"], start_xy, end_xy, color, thickness, cv2.LINE_AA)
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
