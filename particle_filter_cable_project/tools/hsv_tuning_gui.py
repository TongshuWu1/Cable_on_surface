import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox
import tomllib

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cable_detection import HsvCableDetector
from tools.tune_hsv_from_masks import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    collect_labeled_hsv_samples,
    dataset_pairs,
    detector_cleanup_from_config,
    fit_gaussian_hsv_model,
    grid_search_hsv_bounds,
    update_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune HSV cable segmentation from labeled masks.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


class HsvTuningApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("HSV Cable Segmentation Tuning")
        self.pairs = dataset_pairs(args.dataset, ("train", "val"))
        if not self.pairs:
            raise ValueError(f"No labeled image/mask pairs found in {args.dataset}")
        self.index = 0
        self.cleanup = detector_cleanup_from_config(args.config)
        self.config = load_config(args.config)
        hsv_config = self.config.get("hsv", {})

        self.mode_var = tk.StringVar(value=str(hsv_config.get("mode", "range")))
        self.backend_var = tk.BooleanVar(value=True)
        self.h_min = tk.IntVar(value=int(hsv_config.get("h_min", 42)))
        self.h_max = tk.IntVar(value=int(hsv_config.get("h_max", 56)))
        self.s_min = tk.IntVar(value=int(hsv_config.get("s_min", 58)))
        self.s_max = tk.IntVar(value=int(hsv_config.get("s_max", 137)))
        self.v_min = tk.IntVar(value=int(hsv_config.get("v_min", 93)))
        self.v_max = tk.IntVar(value=int(hsv_config.get("v_max", 170)))
        self.gaussian_threshold = tk.DoubleVar(value=float(hsv_config.get("gaussian_threshold", 0.0)))
        self.gaussian_positive_mean = list_or_default(hsv_config.get("gaussian_positive_mean"), [0.0, 1.0, 0.5, 0.5])
        self.gaussian_positive_std = list_or_default(hsv_config.get("gaussian_positive_std"), [1.0, 1.0, 0.25, 0.25])
        self.gaussian_negative_mean = list_or_default(hsv_config.get("gaussian_negative_mean"), [0.0, 1.0, 0.5, 0.5])
        self.gaussian_negative_std = list_or_default(hsv_config.get("gaussian_negative_std"), [1.0, 1.0, 0.25, 0.25])
        self.photos = []
        self.last_action_message = ""
        self.range_scales = []
        self.gaussian_scales = []

        self.build_ui()
        self.refresh()

    def build_ui(self):
        controls = tk.Frame(self.root)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(controls, text="Mode").pack(anchor="w")
        tk.OptionMenu(controls, self.mode_var, "range", "gaussian", command=self.on_mode_change).pack(fill=tk.X)
        tk.Checkbutton(controls, text="Save as HSV backend", variable=self.backend_var).pack(anchor="w", pady=(4, 10))

        self.control_hint = tk.StringVar(value="")
        tk.Label(controls, textvariable=self.control_hint, justify=tk.LEFT, anchor="w", wraplength=260).pack(fill=tk.X, pady=(0, 8))

        range_box = tk.LabelFrame(controls, text="Range HSV")
        range_box.pack(fill=tk.X, pady=(0, 8))
        self.range_scales.append(self.add_scale(range_box, "H min", self.h_min, 0, 179))
        self.range_scales.append(self.add_scale(range_box, "H max", self.h_max, 0, 179))
        self.range_scales.append(self.add_scale(range_box, "S min", self.s_min, 0, 255))
        self.range_scales.append(self.add_scale(range_box, "S max", self.s_max, 0, 255))
        self.range_scales.append(self.add_scale(range_box, "V min", self.v_min, 0, 255))
        self.range_scales.append(self.add_scale(range_box, "V max", self.v_max, 0, 255))

        gaussian_box = tk.LabelFrame(controls, text="Gaussian HSV")
        gaussian_box.pack(fill=tk.X, pady=(0, 8))
        self.gaussian_scales.append(self.add_scale(gaussian_box, "Score threshold", self.gaussian_threshold, -80, 80, resolution=0.1))

        tk.Button(controls, text="Fit Range From Masks", command=self.fit_range).pack(fill=tk.X, pady=(8, 2))
        tk.Button(controls, text="Fit Gaussian From Masks", command=self.fit_gaussian).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Save Config", command=self.save_config).pack(fill=tk.X, pady=(10, 2))

        nav = tk.Frame(controls)
        nav.pack(fill=tk.X, pady=(10, 2))
        tk.Button(nav, text="Prev", command=self.prev_image).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(nav, text="Next", command=self.next_image).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status = tk.StringVar(value="")
        tk.Label(controls, textvariable=self.status, justify=tk.LEFT, anchor="w", wraplength=260).pack(fill=tk.X, pady=(10, 0))

        body = tk.Frame(self.root)
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        header = tk.Frame(body)
        header.pack(fill=tk.X)
        for title in ("Label overlay", "HSV prediction", "Error map"):
            tk.Label(header, text=title).pack(side=tk.LEFT, expand=True)

        images = tk.Frame(body)
        images.pack(fill=tk.BOTH, expand=True)
        self.image_labels = []
        for _ in range(3):
            label = tk.Label(images, bg="#151515")
            label.pack(side=tk.LEFT, padx=4)
            self.image_labels.append(label)

    def add_scale(self, parent, label, variable, from_, to, resolution=1):
        tk.Label(parent, text=label).pack(anchor="w")
        scale = tk.Scale(
            parent,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            resolution=resolution,
            length=260,
            command=lambda _v: self.refresh(),
        )
        scale.pack(fill=tk.X)
        return scale

    def on_mode_change(self, _value=None):
        self.last_action_message = mode_hint(self.mode_var.get())
        self.refresh()

    def fit_range(self):
        try:
            samples = collect_labeled_hsv_samples(self.pairs)
            bounds, metrics = grid_search_hsv_bounds(samples)
        except Exception as exc:
            messagebox.showerror("HSV Fit Failed", str(exc))
            return
        self.mode_var.set("range")
        self.h_min.set(bounds["h_min"])
        self.h_max.set(bounds["h_max"])
        self.s_min.set(bounds["s_min"])
        self.s_max.set(bounds["s_max"])
        self.v_min.set(bounds["v_min"])
        self.v_max.set(bounds["v_max"])
        self.last_action_message = f"Range fit on labels: IoU {metrics['iou']:.4f} | Dice {metrics['dice']:.4f}"
        self.refresh()

    def fit_gaussian(self):
        try:
            samples = collect_labeled_hsv_samples(self.pairs)
            bounds, metrics = fit_gaussian_hsv_model(samples)
        except Exception as exc:
            messagebox.showerror("Gaussian Fit Failed", str(exc))
            return
        self.mode_var.set("gaussian")
        self.gaussian_threshold.set(float(bounds["gaussian_threshold"]))
        self.gaussian_positive_mean = bounds["gaussian_positive_mean"]
        self.gaussian_positive_std = bounds["gaussian_positive_std"]
        self.gaussian_negative_mean = bounds["gaussian_negative_mean"]
        self.gaussian_negative_std = bounds["gaussian_negative_std"]
        self.last_action_message = f"Gaussian fit on labels: IoU {metrics['iou']:.4f} | Dice {metrics['dice']:.4f}"
        self.refresh()

    def save_config(self):
        try:
            update_config(self.args.config, self.current_bounds(), set_backend_hsv=self.backend_var.get())
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))
            return
        self.last_action_message = f"Saved HSV settings to {self.args.config}"
        self.refresh()

    def prev_image(self):
        self.index = (self.index - 1) % len(self.pairs)
        self.refresh()

    def next_image(self):
        self.index = (self.index + 1) % len(self.pairs)
        self.refresh()

    def current_bounds(self):
        return {
            "mode": self.mode_var.get(),
            "h_min": int(self.h_min.get()),
            "h_max": int(self.h_max.get()),
            "s_min": int(self.s_min.get()),
            "s_max": int(self.s_max.get()),
            "v_min": int(self.v_min.get()),
            "v_max": int(self.v_max.get()),
            "gaussian_threshold": float(self.gaussian_threshold.get()),
            "gaussian_positive_mean": list(self.gaussian_positive_mean),
            "gaussian_positive_std": list(self.gaussian_positive_std),
            "gaussian_negative_mean": list(self.gaussian_negative_mean),
            "gaussian_negative_std": list(self.gaussian_negative_std),
        }

    def detector(self):
        bounds = self.current_bounds()
        return HsvCableDetector(
            **bounds,
            min_area=self.cleanup["min_area"],
            open_kernel=self.cleanup["open_kernel"],
            close_kernel=self.cleanup["close_kernel"],
        )

    def refresh(self):
        image_path, mask_path = self.pairs[self.index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            return
        self.update_control_state()
        if mask.shape[:2] != bgr.shape[:2]:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        target = mask > 127
        predicted = self.detector().detect(bgr, extract_geometry=False).mask > 0
        metrics = binary_metrics(predicted, target)

        panels = [
            overlay_mask(bgr, target, (0, 220, 80)),
            overlay_mask(bgr, predicted, (255, 180, 40)),
            error_map(target, predicted),
        ]
        self.photos = [bgr_to_photo(resize_for_display(panel)) for panel in panels]
        for label, photo in zip(self.image_labels, self.photos):
            label.configure(image=photo)

        base = (
            f"{self.index + 1}/{len(self.pairs)} {image_path.name}\n"
            f"{self.mode_summary()} | IoU {metrics['iou']:.4f} | Dice {metrics['dice']:.4f}\n"
            f"predicted {metrics['predicted']} px | target {metrics['target']} px\n"
            f"{mode_hint(self.mode_var.get())}"
        )
        if self.last_action_message:
            base = base + "\n" + self.last_action_message
        self.status.set(base)

    def update_control_state(self):
        mode = self.mode_var.get()
        range_state = tk.NORMAL if mode == "range" else tk.DISABLED
        gaussian_state = tk.NORMAL if mode == "gaussian" else tk.DISABLED
        for scale in self.range_scales:
            scale.configure(state=range_state)
        for scale in self.gaussian_scales:
            scale.configure(state=gaussian_state)
        self.control_hint.set(mode_hint(mode))

    def mode_summary(self):
        if self.mode_var.get() == "gaussian":
            return f"mode=gaussian threshold={float(self.gaussian_threshold.get()):.2f}"
        return (
            f"mode=range H {int(self.h_min.get())}-{int(self.h_max.get())} "
            f"S {int(self.s_min.get())}-{int(self.s_max.get())} "
            f"V {int(self.v_min.get())}-{int(self.v_max.get())}"
        )


def load_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def list_or_default(value, default):
    try:
        values = [float(item) for item in value]
    except Exception:
        return list(default)
    return values if len(values) == len(default) else list(default)


def mode_hint(mode):
    if str(mode) == "gaussian":
        return "Gaussian mode: use Fit Gaussian, then tune only Score threshold. H/S/V sliders are ignored."
    return "Range mode: tune H/S/V sliders directly. Gaussian threshold is ignored."


def binary_metrics(predicted, target):
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    predicted_count = int(np.count_nonzero(predicted))
    target_count = int(np.count_nonzero(target))
    return {
        "iou": intersection / max(union, 1),
        "dice": 2.0 * intersection / max(predicted_count + target_count, 1),
        "predicted": predicted_count,
        "target": target_count,
    }


def overlay_mask(bgr, mask, color):
    panel = np.asarray(bgr, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return panel
    tint = np.zeros_like(panel)
    tint[:, :] = np.array(color, dtype=np.uint8)
    blended = 0.45 * panel[mask].astype(np.float32) + 0.55 * tint[mask].astype(np.float32)
    panel[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    return panel


def error_map(target, predicted):
    target = np.asarray(target, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    panel = np.full((*target.shape, 3), 28, dtype=np.uint8)
    panel[target & predicted] = (50, 220, 70)
    panel[~target & predicted] = (230, 60, 210)
    panel[target & ~predicted] = (30, 40, 240)
    return panel


def resize_for_display(bgr, max_width=420, max_height=520):
    h, w = bgr.shape[:2]
    scale = min(float(max_width) / max(w, 1), float(max_height) / max(h, 1), 1.0)
    if scale >= 0.999:
        return bgr
    return cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def bgr_to_photo(bgr):
    rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


def main():
    args = parse_args()
    root = tk.Tk()
    app = HsvTuningApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
