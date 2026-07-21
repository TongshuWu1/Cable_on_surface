import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cable_pidnet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PIDNetSmallBinary,
    require_torch,
    resolve_device,
)
from pidnet_dataset import (
    DATASET_SPLITS,
    dataset_snapshot,
    load_dataset_manifest,
    session_split_conflicts,
)
from pidnet_schema import (
    ANNOTATION_BODY_LAYER_COUNT,
    ANNOTATION_CHANNEL_COUNT,
    ANNOTATION_SCHEMA_VERSION,
    CROSSING_CHANNEL,
    ENDPOINT_CHANNELS,
    ENDPOINT_SEMANTICS,
    OUTPUT_CHANNEL_COUNT,
    PIDNET_LABEL_MODE,
    PIDNET_SCHEMA_VERSION,
    crossing_label_value,
    endpoint_label_value,
    label_bit,
)

require_torch()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_IMAGE_SIZE = "1280x720"
DEFAULT_THRESHOLD_GRID = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the four-head PIDNet cable observation model.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "datasets/two_cable_pidnet")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "models/pidnet_two_cable_best.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=parse_image_size, default=parse_image_size(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--cable-count", type=int, choices=(2,), default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verified-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the locked test split once after checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--endpoint-weight", type=float, default=2.0)
    parser.add_argument("--crossing-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.20)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stop", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional compatible PIDNet checkpoint used only to initialize matching tensors.",
    )
    return parser.parse_args()


class GenericCableEndpointDataset(Dataset):
    def __init__(self, pairs, image_size=DEFAULT_IMAGE_SIZE, augment=False, gpu_augment=False, cable_count=2):
        self.pairs = list(pairs)
        self.image_size = parse_image_size(image_size)
        self.augment = bool(augment)
        self.gpu_augment = bool(gpu_augment)
        if int(cable_count) != 2:
            raise ValueError(f"This training schema requires exactly two endpoint sets; got {cable_count}.")
        self.cable_count = 2

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask, multilabel = read_training_mask(mask_path, self.cable_count)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image, mask = resize_pair(image, mask, self.image_size)
        if self.augment:
            image, mask = augment_pair(image, mask, photometric=not self.gpu_augment)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        body = generic_body_mask(mask, self.cable_count, multilabel=multilabel).astype(np.uint8)
        endpoint_channels = generic_endpoint_masks(mask, self.cable_count, multilabel=multilabel)
        crossing = generic_crossing_mask(mask, self.cable_count, multilabel=multilabel).astype(np.uint8)
        mask_channels = np.stack([body, *endpoint_channels, crossing], axis=0).astype(np.uint8, copy=False)
        boundary = mask_boundary(body).astype(np.uint8, copy=False)

        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask_channels = torch.from_numpy(np.ascontiguousarray(mask_channels))
        boundary = torch.from_numpy(boundary[None, :, :])
        return image, mask_channels, boundary


def layered_mask_path_from_mask_path(mask_path):
    mask_path = Path(mask_path)
    parts = list(mask_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "masks":
            parts[index] = "masks_layers"
            return Path(*parts).with_suffix(".npz")
    return mask_path.with_suffix(".npz")


def read_training_mask(mask_path, cable_count):
    layer_path = layered_mask_path_from_mask_path(mask_path)
    if not layer_path.exists():
        raise FileNotFoundError(f"Required layered mask not found: {layer_path}")
    with np.load(str(layer_path), allow_pickle=False) as payload:
        required = {"mask", "cable_count", "annotation_schema_version"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Layered mask is missing {missing}: {layer_path}")
        mask = np.asarray(payload["mask"], dtype=np.uint16)
        stored_cable_count = int(np.asarray(payload["cable_count"]).item())
        annotation_version = int(np.asarray(payload["annotation_schema_version"]).item())
    if mask.ndim != 2:
        raise ValueError(f"Layered mask must be HxW; got shape {mask.shape}: {layer_path}")
    if stored_cable_count != int(cable_count):
        raise ValueError(
            f"Layered mask cable_count={stored_cable_count} does not match cable_count={cable_count}: {layer_path}"
        )
    if annotation_version != ANNOTATION_SCHEMA_VERSION:
        raise ValueError(
            f"Layered mask annotation schema={annotation_version}; expected {ANNOTATION_SCHEMA_VERSION}: {layer_path}"
        )
    valid_bits = np.uint16((1 << ANNOTATION_CHANNEL_COUNT) - 1)
    return np.ascontiguousarray(mask & valid_bits, dtype=np.uint16), True


def label_pixels(mask, label, _cable_count=None, multilabel=False):
    if bool(multilabel):
        return (np.asarray(mask, dtype=np.uint16) & label_bit(label)) != 0
    return np.asarray(mask) == int(label)


def generic_body_mask(mask, cable_count, multilabel=False):
    return label_pixels(mask, 1, cable_count, multilabel=multilabel)


def generic_endpoint_masks(mask, cable_count, multilabel=False):
    return [
        label_pixels(mask, endpoint_label_value(cable_index), cable_count, multilabel=multilabel).astype(np.float32)
        for cable_index in range(1, int(cable_count) + 1)
    ]


def generic_crossing_mask(mask, cable_count, multilabel=False):
    return label_pixels(mask, crossing_label_value(), cable_count, multilabel=multilabel)


def find_dataset_splits(dataset_root, verified_only=True):
    dataset_root = Path(dataset_root)
    manifest = load_dataset_manifest(dataset_root)
    conflicts = session_split_conflicts(manifest)
    if conflicts:
        raise ValueError(f"Capture sessions span multiple splits: {conflicts}")
    result = {
        split: strict_split_pairs(
            dataset_root,
            split,
            manifest=manifest,
            verified_only=verified_only,
            require_nonempty=split in ("train", "val"),
        )
        for split in DATASET_SPLITS
    }
    stems = {split: {image.stem for image, _mask in pairs} for split, pairs in result.items()}
    for left_index, left in enumerate(DATASET_SPLITS):
        for right in DATASET_SPLITS[left_index + 1:]:
            overlap = sorted(stems[left] & stems[right])
            if overlap:
                raise ValueError(f"{left}/{right} filename overlap: {overlap}")
    return result


def strict_split_pairs(dataset_root, split, manifest=None, verified_only=True, require_nonempty=True):
    split = str(split).strip().lower()
    if split not in DATASET_SPLITS:
        raise ValueError(f"Unsupported dataset split: {split}")
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "images" / split
    mask_dir = dataset_root / "masks" / split
    layer_dir = dataset_root / "masks_layers" / split
    if not require_nonempty and not any(directory.exists() for directory in (image_dir, mask_dir, layer_dir)):
        return []
    for directory in (image_dir, mask_dir, layer_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required dataset directory not found: {directory}")

    images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    layers = {path.stem: path for path in layer_dir.iterdir() if path.suffix.lower() == ".npz"}
    if set(images) != set(masks) or set(images) != set(layers):
        raise ValueError(
            f"Incomplete {split} dataset: images_without_masks={sorted(set(images) - set(masks))}, "
            f"images_without_layers={sorted(set(images) - set(layers))}, "
            f"masks_without_images={sorted(set(masks) - set(images))}, "
            f"layers_without_images={sorted(set(layers) - set(images))}"
        )
    manifest = manifest or load_dataset_manifest(dataset_root)
    items = manifest.get("items", {})
    selected = []
    for stem in sorted(images):
        metadata = items.get(stem)
        if metadata is None:
            if verified_only:
                continue
        else:
            if str(metadata.get("split", "")) != split:
                raise ValueError(f"Manifest split mismatch for {stem}: {metadata.get('split')!r} versus {split!r}")
            if verified_only and not bool(metadata.get("verified", False)):
                continue
        selected.append((images[stem], masks[stem]))
    if require_nonempty and not selected:
        qualifier = " human-verified" if verified_only else ""
        raise ValueError(f"No{qualifier} image/mask pairs found in {image_dir}")
    return selected


def parse_image_size(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise argparse.ArgumentTypeError("Image size tuple must be (width, height).")
        width, height = value
    else:
        text = str(value).strip().lower()
        if text in {"720p", "hd720"}:
            return 1280, 720
        text = text.replace(",", "x").replace("*", "x")
        if "x" in text:
            parts = [part.strip() for part in text.split("x") if part.strip()]
            if len(parts) != 2:
                raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 1280x720.")
            width, height = parts
        else:
            width = height = text
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Image size must be an integer or WIDTHxHEIGHT.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Image width and height must be positive.")
    return width, height


def image_size_text(image_size):
    width, height = parse_image_size(image_size)
    return f"{width}x{height}" if width != height else str(width)


def resize_pair(image, mask, image_size):
    target_w, target_h = parse_image_size(image_size)
    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return image, mask


def _warp_pair(image, mask):
    height, width = image.shape[:2]
    angle = random.uniform(-14.0, 14.0)
    scale = random.uniform(0.90, 1.10)
    center = (0.5 * width, 0.5 * height)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += random.uniform(-0.05, 0.05) * width
    matrix[1, 2] += random.uniform(-0.05, 0.05) * height
    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def _photometric_augment(image, include_basic=True):
    image = image.astype(np.float32)
    if include_basic and random.random() < 0.75:
        image = image * random.uniform(0.72, 1.32) + random.uniform(-24.0, 24.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if include_basic and random.random() < 0.55:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + random.uniform(-4.0, 4.0), 180.0)
        hsv[:, :, 1] *= random.uniform(0.78, 1.24)
        hsv[:, :, 2] *= random.uniform(0.82, 1.18)
        image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if include_basic and random.random() < 0.20:
        gamma = random.uniform(0.75, 1.35)
        table = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
        image = cv2.LUT(image, table)
    if random.random() < 0.18:
        height, width = image.shape[:2]
        overlay = np.ones((height, width), dtype=np.float32)
        x0 = random.randint(-width // 4, width)
        x1 = x0 + random.randint(max(16, width // 8), max(32, width // 2))
        cv2.rectangle(overlay, (x0, 0), (x1, height), random.uniform(0.45, 0.80), -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(width, height) * 0.03)
        image = np.clip(image.astype(np.float32) * overlay[:, :, None], 0, 255).astype(np.uint8)
    if random.random() < 0.22:
        kernel_size = random.choice((3, 5, 7))
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            if random.random() < 0.5:
                kernel[kernel_size // 2, :] = 1.0 / kernel_size
            else:
                kernel[:, kernel_size // 2] = 1.0 / kernel_size
            image = cv2.filter2D(image, -1, kernel)
    if include_basic and random.random() < 0.20:
        noise = np.random.normal(0.0, random.uniform(2.0, 9.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.12:
        quality = random.randint(55, 92)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image


def _visible_occlusion(image, mask):
    height, width = image.shape[:2]
    box_w = random.randint(max(8, width // 40), max(16, width // 10))
    box_h = random.randint(max(8, height // 40), max(16, height // 8))
    x0 = random.randint(0, max(0, width - box_w))
    y0 = random.randint(0, max(0, height - box_h))
    color = tuple(int(value) for value in np.median(image.reshape(-1, 3), axis=0))
    cv2.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), color, -1)
    cv2.rectangle(mask, (x0, y0), (x0 + box_w, y0 + box_h), 0, -1)
    return image, mask


def augment_pair(image, mask, photometric=True):
    if random.random() < 0.5:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    if random.random() < 0.20:
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
    if random.random() < 0.65:
        image, mask = _warp_pair(image, mask)
    if random.random() < 0.16:
        image, mask = _visible_occlusion(image, mask)
    image = _photometric_augment(image, include_basic=photometric)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def gpu_photometric_augment(image):
    if image.device.type != "cuda" or image.ndim != 4:
        return image
    batch = image.shape[0]
    mean = torch.as_tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    rgb = image * std + mean
    apply = (torch.rand((batch, 1, 1, 1), device=image.device) < 0.80).to(image.dtype)
    contrast = 1.0 + apply * (torch.rand((batch, 1, 1, 1), device=image.device) * 0.50 - 0.25)
    brightness = apply * (torch.rand((batch, 1, 1, 1), device=image.device) * 0.16 - 0.08)
    rgb = rgb * contrast + brightness
    gray = torch.sum(
        rgb * torch.as_tensor((0.299, 0.587, 0.114), device=image.device, dtype=image.dtype).view(1, 3, 1, 1),
        dim=1,
        keepdim=True,
    )
    saturation = 0.82 + torch.rand((batch, 1, 1, 1), device=image.device) * 0.36
    rgb = gray + (rgb - gray) * saturation
    channel_gain = 0.92 + torch.rand((batch, 3, 1, 1), device=image.device) * 0.16
    rgb = rgb * channel_gain
    noise_apply = (torch.rand((batch, 1, 1, 1), device=image.device) < 0.25).to(image.dtype)
    noise_sigma = noise_apply * torch.rand((batch, 1, 1, 1), device=image.device) * 0.025
    rgb = torch.clamp(rgb + torch.randn_like(rgb) * noise_sigma, 0.0, 1.0)
    return (rgb - mean) / std


def prepare_image_batch(image, device, channels_last=False):
    image = image.to(device, non_blocking=True)
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32).mul_(1.0 / 255.0)
        mean = torch.as_tensor(IMAGENET_MEAN, device=device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD, device=device, dtype=image.dtype).view(1, 3, 1, 1)
        image = (image - mean) / std
    elif not image.is_floating_point():
        image = image.to(dtype=torch.float32)
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
    return image


def prepare_target_batch(target, device):
    return target.to(device=device, dtype=torch.float32, non_blocking=True)


def mask_boundary(mask):
    mask_u8 = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return (cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel) > 0).astype(np.float32)


def soft_dice_loss(logits, target, eps=1e-6):
    probability = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(probability * target, dim=(1, 2, 3)) + eps
    denominator = torch.sum(probability + target, dim=(1, 2, 3)) + eps
    return 1.0 - torch.mean(numerator / denominator)


def focal_bce_loss(logits, target, alpha, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target)
    weighted = alpha_t * torch.pow(torch.clamp(1.0 - pt, min=0.0), float(gamma)) * bce
    return torch.sum(weighted) / torch.clamp(torch.sum(alpha_t), min=1e-6)


def segmentation_loss(
    outputs,
    mask,
    boundary,
    channel_alpha,
    endpoint_weight,
    crossing_weight,
    boundary_weight,
    focal_gamma,
    dice_weight,
):
    seg_logits = outputs["seg"]
    boundary_logits = outputs["boundary"]
    if seg_logits.shape != mask.shape or seg_logits.shape[1] != OUTPUT_CHANNEL_COUNT:
        raise ValueError(f"Segmentation logits/target mismatch: {tuple(seg_logits.shape)} versus {tuple(mask.shape)}")
    head_losses = []
    for channel in range(OUTPUT_CHANNEL_COUNT):
        logits = seg_logits[:, channel:channel + 1]
        target = mask[:, channel:channel + 1]
        focal = focal_bce_loss(logits, target, channel_alpha[channel], focal_gamma)
        dice = soft_dice_loss(logits, target)
        head_losses.append(focal + float(dice_weight) * dice)
    endpoint_loss = torch.mean(torch.stack([head_losses[channel] for channel in ENDPOINT_CHANNELS]))
    boundary_loss = focal_bce_loss(boundary_logits, boundary, alpha=0.80, gamma=focal_gamma)
    total = (
        head_losses[0]
        + float(endpoint_weight) * endpoint_loss
        + float(crossing_weight) * head_losses[CROSSING_CHANNEL]
        + float(boundary_weight) * boundary_loss
    )
    detached = {
        "body": float(head_losses[0].detach().item()),
        "endpoint1": float(head_losses[ENDPOINT_CHANNELS[0]].detach().item()),
        "endpoint2": float(head_losses[ENDPOINT_CHANNELS[1]].detach().item()),
        "crossing": float(head_losses[CROSSING_CHANNEL].detach().item()),
        "boundary": float(boundary_loss.detach().item()),
    }
    return total, detached


def compute_channel_profile(pairs, cable_count):
    positive_pixels = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.int64)
    positive_frames = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.int64)
    total_pixels = 0
    both_endpoint_frames = 0
    for _image_path, mask_path in pairs:
        mask, multilabel = read_training_mask(mask_path, cable_count)
        channels = [
            generic_body_mask(mask, cable_count, multilabel),
            *generic_endpoint_masks(mask, cable_count, multilabel),
            generic_crossing_mask(mask, cable_count, multilabel),
        ]
        frame_pixels = np.asarray([np.count_nonzero(channel) for channel in channels], dtype=np.int64)
        positive_pixels += frame_pixels
        positive_frames += frame_pixels > 0
        both_endpoint_frames += int(np.all(frame_pixels[list(ENDPOINT_CHANNELS)] > 0))
        total_pixels += int(mask.size)
    prevalence = positive_pixels.astype(np.float64) / max(total_pixels, 1)
    return {
        "frame_count": int(len(pairs)),
        "total_pixels": int(total_pixels),
        "positive_pixels": positive_pixels.tolist(),
        "positive_frames": positive_frames.tolist(),
        "both_endpoint_frames": int(both_endpoint_frames),
        "prevalence": prevalence.tolist(),
    }


def selected_session_counts(dataset_root, pairs_by_split):
    manifest = load_dataset_manifest(dataset_root)
    items = manifest.get("items", {})
    counts = {}
    for split in DATASET_SPLITS:
        session_ids = {
            str(items.get(image_path.stem, {}).get("session_id", "")).strip()
            for image_path, _mask_path in pairs_by_split[split]
        }
        session_ids.discard("")
        counts[split] = len(session_ids)
    return counts


def dataset_profile_warnings(profiles, session_counts):
    warnings = []
    train = profiles["train"]
    validation = profiles["val"]
    if session_counts["train"] < 2:
        warnings.append(
            "training uses fewer than two independent capture sessions; burst frames do not provide arrangement diversity"
        )
    minimum_joint = max(5, math.ceil(0.25 * train["frame_count"]))
    if train["both_endpoint_frames"] < minimum_joint:
        warnings.append(
            f"only {train['both_endpoint_frames']}/{train['frame_count']} training frames contain both endpoint groups; "
            "the two-cable validation distribution is underrepresented"
        )
    if validation["frame_count"] < 8:
        warnings.append(
            f"validation contains only {validation['frame_count']} frames; threshold and component metrics will be noisy"
        )
    channel_names = ("body", "endpoint1", "endpoint2", "crossing")
    for split in ("train", "val"):
        for name, count in zip(channel_names, profiles[split]["positive_frames"]):
            if count == 0:
                warnings.append(f"{split} has no positive {name} frames")
    return warnings


def early_stop_summary(epoch, best_epoch, best_score, current_score, patience, min_delta, metrics):
    epoch = int(epoch)
    patience = int(patience)
    first_plateau_epoch = epoch - patience + 1
    required_score = float(best_score) + float(min_delta)
    return {
        "type": "validation_plateau",
        "stopped_epoch": epoch,
        "plateau_start_epoch": int(first_plateau_epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "current_score": float(current_score),
        "required_score": float(required_score),
        "patience": patience,
        "min_delta": float(min_delta),
        "body_iou": float(metrics["body_iou"]),
        "endpoint1_quality": float(metrics["endpoint1_quality"]),
        "endpoint2_quality": float(metrics["endpoint2_quality"]),
        "crossing_quality": float(metrics["crossing_quality"]),
    }


def print_early_stop_summary(summary):
    print(
        "EARLY STOP: validation score plateaued. "
        f"Stopped at epoch {summary['stopped_epoch']} because val_score did not exceed "
        f"best_score + min_delta = {summary['required_score']:.6f} for "
        f"{summary['patience']} consecutive epochs "
        f"({summary['plateau_start_epoch']}--{summary['stopped_epoch']})."
    )
    print(
        f"  best: epoch={summary['best_epoch']} score={summary['best_score']:.6f}; "
        f"current: score={summary['current_score']:.6f}; min_delta={summary['min_delta']:.6f}"
    )
    print(
        f"  current validation heads: body_iou={summary['body_iou']:.4f} "
        f"endpoint1_quality={summary['endpoint1_quality']:.4f} "
        f"endpoint2_quality={summary['endpoint2_quality']:.4f} "
        f"crossing_quality={summary['crossing_quality']:.4f}"
    )
    print("  To continue regardless of a plateau, set Early-stop patience to 0.")


def connected_component_summaries(mask, min_area=3, max_components=32, return_total=False):
    binary = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8) > 0)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            components.append((area, np.asarray(centroids[index], dtype=np.float64)))
    components.sort(key=lambda item: item[0], reverse=True)
    selected = components[:max_components]
    areas = np.asarray([area for area, _centroid in selected], dtype=np.float64)
    points = np.asarray([centroid for _area, centroid in selected], dtype=np.float64).reshape(-1, 2)
    if return_total:
        return points, areas, len(components)
    return points, areas


def linear_sum_assignment(cost):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("Assignment cost must be a matrix.")
    original_rows, original_cols = cost.shape
    transposed = original_rows > original_cols
    if transposed:
        cost = cost.T
    rows, cols = cost.shape
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(cols + 1, dtype=np.float64)
    p = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        min_value = np.full(cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, cols + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(cols + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignments = [(int(p[column] - 1), int(column - 1)) for column in range(1, cols + 1) if p[column] > 0]
    if transposed:
        assignments = [(column, row) for row, column in assignments]
    return [(row, column) for row, column in assignments if row < original_rows and column < original_cols]


def component_match_counts(predicted, target, max_distance_px, min_area=3, min_area_ratio=0.20):
    predicted_centroids, predicted_areas, predicted_total = connected_component_summaries(
        predicted,
        min_area=min_area,
        return_total=True,
    )
    target_centroids, target_areas, target_total = connected_component_summaries(
        target,
        min_area=min_area,
        return_total=True,
    )
    if len(predicted_centroids) == 0 or len(target_centroids) == 0:
        return 0, predicted_total, target_total, 0.0
    distance = np.linalg.norm(target_centroids[:, None, :] - predicted_centroids[None, :, :], axis=2)
    area_ratio = np.minimum(target_areas[:, None], predicted_areas[None, :]) / np.maximum(
        target_areas[:, None], predicted_areas[None, :]
    )
    valid = (distance <= float(max_distance_px)) & (area_ratio >= float(min_area_ratio))
    invalid_penalty = max(float(max_distance_px), 1.0) * (max(distance.shape) + 1) * 10.0
    assignment_cost = np.where(valid, distance, invalid_penalty)
    assignments = linear_sum_assignment(assignment_cost)
    accepted = [(row, column) for row, column in assignments if valid[row, column]]
    true_positive = len(accepted)
    # Matching is bounded to the largest components for predictable runtime,
    # but every discarded component still counts against precision/recall.
    false_positive = predicted_total - true_positive
    false_negative = target_total - true_positive
    error_sum = float(sum(distance[row, column] for row, column in accepted))
    return true_positive, false_positive, false_negative, error_sum


def component_metric_summary(counts):
    true_positive, false_positive, false_negative, error_sum = counts
    if true_positive == 0 and false_positive == 0 and false_negative == 0:
        precision = recall = f1 = 1.0
    else:
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "centroid_mae_px": float(error_sum / true_positive) if true_positive > 0 else None,
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
    }


def weighted_harmonic_mean(values, weights, floor=0.05):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    weights = np.asarray(weights, dtype=np.float64)
    floor = float(np.clip(floor, 1e-6, 0.25))
    softened = floor + (1.0 - floor) * values
    harmonic = float(np.sum(weights) / np.sum(weights / softened))
    return float(np.clip((harmonic - floor) / (1.0 - floor), 0.0, 1.0))


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    use_amp=False,
    channels_last=False,
    threshold_grid=DEFAULT_THRESHOLD_GRID,
    fixed_thresholds=None,
):
    model.eval()
    thresholds = torch.as_tensor(threshold_grid, device=device, dtype=torch.float32)
    intersection = np.zeros((OUTPUT_CHANNEL_COUNT, len(threshold_grid)), dtype=np.float64)
    union = np.zeros_like(intersection)
    pred_count = np.zeros_like(intersection)
    target_count = np.zeros(OUTPUT_CHANNEL_COUNT, dtype=np.float64)
    for image, mask, _boundary in loader:
        image = prepare_image_batch(image, device, channels_last=channels_last)
        mask = prepare_target_batch(mask, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            probability = torch.sigmoid(model(image)["seg"]).float()
        target = mask > 0.5
        for channel in range(OUTPUT_CHANNEL_COUNT):
            channel_probability = probability[:, channel]
            channel_target = target[:, channel]
            target_count[channel] += float(torch.sum(channel_target).item())
            for start in range(0, len(threshold_grid), 5):
                chunk = thresholds[start:start + 5].view(-1, 1, 1, 1)
                predicted = channel_probability.unsqueeze(0) >= chunk
                expanded_target = channel_target.unsqueeze(0)
                inter = torch.sum(predicted & expanded_target, dim=(1, 2, 3))
                combined = torch.sum(predicted | expanded_target, dim=(1, 2, 3))
                count = torch.sum(predicted, dim=(1, 2, 3))
                end = start + len(inter)
                intersection[channel, start:end] += inter.cpu().numpy()
                union[channel, start:end] += combined.cpu().numpy()
                pred_count[channel, start:end] += count.cpu().numpy()

    channel_iou_grid = intersection / np.maximum(union, 1.0)
    selected_indices = []
    if fixed_thresholds is not None:
        if len(fixed_thresholds) != OUTPUT_CHANNEL_COUNT:
            raise ValueError(f"Expected {OUTPUT_CHANNEL_COUNT} fixed thresholds; got {fixed_thresholds}.")
        selected_indices = [
            int(np.argmin(np.abs(np.asarray(threshold_grid) - float(value))))
            for value in fixed_thresholds
        ]
    else:
        for channel in range(OUTPUT_CHANNEL_COUNT):
            tie_break = -1e-9 * np.abs(np.asarray(threshold_grid) - 0.5)
            selected_indices.append(int(np.argmax(channel_iou_grid[channel] + tie_break)))
    selected_thresholds = tuple(float(threshold_grid[index]) for index in selected_indices)
    channel_iou = np.asarray([channel_iou_grid[c, selected_indices[c]] for c in range(OUTPUT_CHANNEL_COUNT)])
    channel_dice = np.asarray([
        2.0 * intersection[c, selected_indices[c]]
        / max(pred_count[c, selected_indices[c]] + target_count[c], 1.0)
        for c in range(OUTPUT_CHANNEL_COUNT)
    ])

    component_counts = {channel: np.zeros(4, dtype=np.float64) for channel in (*ENDPOINT_CHANNELS, CROSSING_CHANNEL)}
    for image, mask, _boundary in loader:
        image = prepare_image_batch(image, device, channels_last=channels_last)
        with torch.amp.autocast("cuda", enabled=use_amp):
            probability = torch.sigmoid(model(image)["seg"]).float().cpu().numpy()
        target = mask.numpy() > 0.5
        height, width = target.shape[-2:]
        diagonal = math.hypot(width, height)
        for batch_index in range(target.shape[0]):
            for channel in (*ENDPOINT_CHANNELS, CROSSING_CHANNEL):
                predicted = probability[batch_index, channel] >= selected_thresholds[channel]
                maximum_distance = diagonal * (0.020 if channel in ENDPOINT_CHANNELS else 0.025)
                counts = component_match_counts(predicted, target[batch_index, channel], maximum_distance)
                component_counts[channel] += np.asarray(counts, dtype=np.float64)

    component_metrics = {
        channel: component_metric_summary(component_counts[channel])
        for channel in component_counts
    }
    endpoint1 = component_metrics[ENDPOINT_CHANNELS[0]]
    endpoint2 = component_metrics[ENDPOINT_CHANNELS[1]]
    crossing = component_metrics[CROSSING_CHANNEL]
    endpoint1_quality = 0.5 * (channel_iou[ENDPOINT_CHANNELS[0]] + endpoint1["f1"])
    endpoint2_quality = 0.5 * (channel_iou[ENDPOINT_CHANNELS[1]] + endpoint2["f1"])
    crossing_quality = 0.5 * (channel_iou[CROSSING_CHANNEL] + crossing["f1"])
    score = weighted_harmonic_mean(
        (channel_iou[0], endpoint1_quality, endpoint2_quality, crossing_quality),
        (0.35, 0.25, 0.25, 0.15),
    )
    return {
        "score": score,
        "iou": float(np.mean(channel_iou)),
        "dice": float(np.mean(channel_dice)),
        "body_iou": float(channel_iou[0]),
        "body_dice": float(channel_dice[0]),
        "endpoint1_iou": float(channel_iou[ENDPOINT_CHANNELS[0]]),
        "endpoint2_iou": float(channel_iou[ENDPOINT_CHANNELS[1]]),
        "crossing_iou": float(channel_iou[CROSSING_CHANNEL]),
        "endpoint1_component": endpoint1,
        "endpoint2_component": endpoint2,
        "crossing_component": crossing,
        "endpoint1_quality": float(endpoint1_quality),
        "endpoint2_quality": float(endpoint2_quality),
        "crossing_quality": float(crossing_quality),
        "thresholds": list(selected_thresholds),
        "threshold_grid": list(threshold_grid),
        "channel_iou_grid": channel_iou_grid.tolist(),
    }


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        self.effective_decay = 0.0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # A constant 0.995 EMA needs hundreds of optimizer updates just to
        # forget random initialization.  Small datasets may have only a few
        # updates per epoch, so use the standard update-count warmup and treat
        # the configured decay as a ceiling.  The ceiling is approached only
        # after the estimator has accumulated enough observations.
        warmup_decay = (1.0 + self.updates) / (10.0 + self.updates)
        self.effective_decay = min(self.decay, warmup_decay)
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.effective_decay).add_(incoming, alpha=1.0 - self.effective_decay)
            else:
                value.copy_(incoming)

    def load_state_dict(self, state, updates=0):
        self.model.load_state_dict(state)
        self.updates = max(0, int(updates))
        self.effective_decay = (
            min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
            if self.updates > 0
            else 0.0
        )


def checkpoint_sibling(path, suffix):
    path = Path(path)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model_state, args, metrics, epoch, channel_alpha, extra=None):
    payload = {
        "model_state": model_state,
        "config": {
            "model": "pidnet_small_binary",
            "base_channels": int(args.base_channels),
            "input_channels": 3,
            "input_mode": "rgb",
            "output_channels": OUTPUT_CHANNEL_COUNT,
            "label_mode": PIDNET_LABEL_MODE,
            "observation_schema_version": PIDNET_SCHEMA_VERSION,
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "endpoint_semantics": ENDPOINT_SEMANTICS,
            "annotation_body_layer_count": ANNOTATION_BODY_LAYER_COUNT,
            "endpoint_channels": True,
            "endpoint_channel_count": len(ENDPOINT_CHANNELS),
            "crossing_channels": True,
            "crossing_channel": CROSSING_CHANNEL,
            "imgsz": image_size_text(args.imgsz),
            "recommended_thresholds": list(metrics.get("thresholds", (0.5,) * OUTPUT_CHANNEL_COUNT)),
        },
        "training": {
            "target": "cable+endpoints_cable1+endpoints_cable2+crossing",
            "epoch": int(epoch),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "endpoint_weight": float(args.endpoint_weight),
            "crossing_weight": float(args.crossing_weight),
            "boundary_weight": float(args.boundary_weight),
            "focal_gamma": float(args.focal_gamma),
            "dice_weight": float(args.dice_weight),
            "ema_decay": float(args.ema_decay),
            "ema_strategy": "update_count_warmup_ceiling",
            "seed": int(args.seed),
            "channel_alpha": [float(value) for value in channel_alpha],
        },
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    return payload


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def run_metadata(args, device, pairs_by_split, snapshot_path):
    cuda = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda = {
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory": int(properties.total_memory),
            "cuda_version": torch.version.cuda,
        }
    return {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": cuda,
        "git": git_state(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "split_counts": {split: len(pairs_by_split[split]) for split in DATASET_SPLITS},
        "dataset_snapshot": str(snapshot_path),
    }


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_loader(dataset, batch_size, shuffle, workers, device, prefetch_factor):
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(workers) > 0,
        "worker_init_fn": worker_seed,
    }
    if int(workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**kwargs)


def load_compatible_initialization(model, checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = payload.get("model_state", payload)
    destination = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in destination and tuple(value.shape) == tuple(destination[name].shape)
    }
    if not compatible:
        raise ValueError(f"No compatible PIDNet tensors found in initialization checkpoint: {checkpoint_path}")
    result = model.load_state_dict(compatible, strict=False)
    parameter_total = sum(tensor.numel() for tensor in destination.values())
    parameter_loaded = sum(destination[name].numel() for name in compatible)
    return {
        "tensor_count": len(compatible),
        "coverage": float(parameter_loaded / max(parameter_total, 1)),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


def validate_resume_checkpoint(payload, args, dataset_sha256):
    stored_hash = str(payload.get("dataset_sha256", "")).strip()
    if not stored_hash:
        raise ValueError(
            "Resume checkpoint predates immutable dataset tracking. Start a fresh run, or use it as an "
            "initialization checkpoint instead of an exact resume."
        )
    if stored_hash != str(dataset_sha256):
        raise ValueError(
            "Resume dataset does not match the checkpoint dataset snapshot. Exact resume is refused; "
            "start a fresh run after dataset edits."
        )
    stored = payload.get("training", {})
    expected = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "endpoint_weight": float(args.endpoint_weight),
        "crossing_weight": float(args.crossing_weight),
        "boundary_weight": float(args.boundary_weight),
        "focal_gamma": float(args.focal_gamma),
        "dice_weight": float(args.dice_weight),
        "ema_decay": float(args.ema_decay),
        "seed": int(args.seed),
    }
    mismatches = []
    for key, current in expected.items():
        if key not in stored:
            mismatches.append(f"{key}=missing")
            continue
        previous = stored[key]
        equal = (
            math.isclose(float(previous), float(current), rel_tol=1e-12, abs_tol=1e-12)
            if isinstance(current, float)
            else int(previous) == current
        )
        if not equal:
            mismatches.append(f"{key}={previous!r}->{current!r}")
    if mismatches:
        raise ValueError(
            "Exact resume settings changed: " + ", ".join(mismatches) + ". "
            "Use the previous settings, or start a new run with --init-checkpoint."
        )


def validate_training_arguments(args):
    positive_integers = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_channels": args.base_channels,
        "prefetch_factor": args.prefetch_factor,
    }
    for name, value in positive_integers.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be at least 1; got {value!r}.")
    if int(args.num_workers) < 0 or int(args.early_stop) < 0:
        raise ValueError("num_workers and early_stop must be non-negative.")
    if float(args.lr) <= 0.0:
        raise ValueError("Learning rate must be positive.")
    for name in (
        "weight_decay", "endpoint_weight", "crossing_weight", "boundary_weight",
        "focal_gamma", "dice_weight", "grad_clip", "min_delta",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if not 0.0 <= float(args.ema_decay) < 1.0:
        raise ValueError("ema_decay must be in [0, 1).")


def train(args):
    validate_training_arguments(args)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    device = resolve_device(args.device)
    if int(args.cable_count) != 2:
        raise ValueError(f"This project trains exactly two endpoint sets; got {args.cable_count}.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not bool(args.deterministic)
        torch.backends.cudnn.deterministic = bool(args.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    if bool(args.deterministic):
        torch.use_deterministic_algorithms(True)

    pairs_by_split = find_dataset_splits(args.dataset, verified_only=bool(args.verified_only))
    train_pairs = pairs_by_split["train"]
    val_pairs = pairs_by_split["val"]
    cable_count = int(args.cable_count)
    dataset_profiles = {
        split: compute_channel_profile(pairs_by_split[split], cable_count)
        for split in DATASET_SPLITS
    }
    prevalence = np.asarray(dataset_profiles["train"]["prevalence"], dtype=np.float64)
    # Alpha = 1 - prevalence balances total positive/negative focal weight in
    # expectation.  Loss normalization keeps rare heads on a comparable scale.
    channel_alpha = np.clip(1.0 - prevalence, 0.50, 0.9999)
    session_counts = selected_session_counts(args.dataset, pairs_by_split)
    snapshot = dataset_snapshot(args.dataset, pairs_by_split)

    use_gpu_augment = bool(args.gpu_augment) and device.type == "cuda"
    train_dataset = GenericCableEndpointDataset(
        train_pairs,
        args.imgsz,
        augment=True,
        gpu_augment=use_gpu_augment,
        cable_count=cable_count,
    )
    val_dataset = GenericCableEndpointDataset(val_pairs, args.imgsz, augment=False, cable_count=cable_count)
    test_dataset = GenericCableEndpointDataset(pairs_by_split["test"], args.imgsz, augment=False, cable_count=cable_count)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device, args.prefetch_factor)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device, args.prefetch_factor)
    test_loader = (
        make_loader(test_dataset, args.batch_size, False, args.num_workers, device, args.prefetch_factor)
        if bool(getattr(args, "evaluate_test", False)) and len(test_dataset) else None
    )

    model = PIDNetSmallBinary(base_channels=int(args.base_channels), output_channels=OUTPUT_CHANNEL_COUNT, input_channels=3).to(device)
    if bool(args.channels_last):
        model = model.to(memory_format=torch.channels_last)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if args.resume is not None and init_checkpoint is not None:
        raise ValueError("Use either --resume or --init-checkpoint, not both.")
    if init_checkpoint is not None:
        initialization = load_compatible_initialization(model, Path(init_checkpoint), device)
        print(
            f"initialized {init_checkpoint} tensors={initialization['tensor_count']} "
            f"coverage={initialization['coverage']:.3f}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.lr) * 0.02,
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, decay=float(args.ema_decay))
    forward_model = torch.compile(model, mode="max-autotune") if bool(args.compile) and hasattr(torch, "compile") else model

    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume_checkpoint(payload, args, snapshot["dataset_sha256"])
        model.load_state_dict(payload["model_state"])
        if "ema_state" in payload and "ema_updates" in payload:
            ema.load_state_dict(payload["ema_state"], updates=payload["ema_updates"])
        else:
            # LEGACY COMPATIBILITY: early resume checkpoints stored EMA
            # parameters without their update count.  Reusing that state would
            # apply the wrong decay history, so resume safely from the raw
            # model and make the compatibility path explicit in the log.
            ema = ModelEMA(model, decay=float(args.ema_decay))
            if "ema_state" in payload:
                print(
                    "EMA WARNING: legacy resume checkpoint has no update count; "
                    "resetting EMA from the resumed raw model to avoid random-initialization bias"
                )
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_score = float(payload.get("best_score", -1.0))
        best_epoch = int(payload.get("best_epoch", 0))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        print(f"resumed {resume_path} at epoch {start_epoch}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_sibling(args.output, "last")
    history_path = checkpoint_sibling(args.output, "history").with_suffix(".csv")
    snapshot_path = checkpoint_sibling(args.output, "dataset").with_suffix(".json")
    run_path = checkpoint_sibling(args.output, "run").with_suffix(".json")
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_info = run_metadata(args, device, pairs_by_split, snapshot_path)
    run_info["channel_prevalence"] = prevalence.tolist()
    run_info["focal_alpha"] = channel_alpha.tolist()
    run_info["dataset_profiles"] = dataset_profiles
    run_info["selected_session_counts"] = session_counts
    run_path.write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    history_fields = (
        "epoch", "loss", "body_loss", "endpoint1_loss", "endpoint2_loss", "crossing_loss", "boundary_loss",
        "val_score", "val_iou", "val_dice", "body_iou", "endpoint1_iou", "endpoint2_iou", "crossing_iou",
        "endpoint1_f1", "endpoint2_f1", "crossing_f1", "threshold_body", "threshold_endpoint1",
        "threshold_endpoint2", "threshold_crossing", "lr", "seconds",
    )
    write_header = not history_path.exists() or start_epoch == 1
    history_stream = history_path.open("a" if not write_header else "w", newline="", encoding="utf-8")
    history_writer = csv.DictWriter(history_stream, fieldnames=history_fields)
    if write_header:
        history_writer.writeheader()

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(
        f"Training on {len(train_pairs)} verified images, validating on {len(val_pairs)}, "
        f"locked_test_frames={len(pairs_by_split['test'])}, evaluate_test={bool(getattr(args, 'evaluate_test', False))}, "
        f"imgsz={image_size_text(args.imgsz)}, device={device_name}, "
        f"amp={use_amp}, channels_last={bool(args.channels_last)}, gpu_augment={use_gpu_augment}"
    )
    print(f"channel_prevalence={prevalence.tolist()} focal_alpha={channel_alpha.tolist()}")
    channel_names = ("body", "endpoint1", "endpoint2", "crossing")
    for split in DATASET_SPLITS:
        profile = dataset_profiles[split]
        coverage = ",".join(
            f"{name}:{count}/{profile['frame_count']}"
            for name, count in zip(channel_names, profile["positive_frames"])
        )
        print(
            f"dataset_profile {split} sessions={session_counts[split]} positive_frames={coverage} "
            f"both_endpoints={profile['both_endpoint_frames']}/{profile['frame_count']}"
        )
    print(
        f"optimizer_steps_per_epoch={len(train_loader)} "
        f"planned_max_updates={len(train_loader) * max(0, int(args.epochs) - start_epoch + 1)}"
    )
    for warning in dataset_profile_warnings(dataset_profiles, session_counts):
        print(f"DATASET WARNING: {warning}")

    termination = None
    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            epoch_start = time.time()
            model.train()
            totals = {name: 0.0 for name in ("loss", "body", "endpoint1", "endpoint2", "crossing", "boundary")}
            for image, mask, boundary in train_loader:
                image = prepare_image_batch(image, device, channels_last=bool(args.channels_last))
                mask = prepare_target_batch(mask, device)
                boundary = prepare_target_batch(boundary, device)
                if use_gpu_augment:
                    image = gpu_photometric_augment(image)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, head = segmentation_loss(
                        forward_model(image),
                        mask,
                        boundary,
                        channel_alpha,
                        args.endpoint_weight,
                        args.crossing_weight,
                        args.boundary_weight,
                        args.focal_gamma,
                        args.dice_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if float(args.grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                totals["loss"] += float(loss.item())
                for name in head:
                    totals[name] += head[name]
            scheduler.step()

            metrics = evaluate(
                ema.model,
                val_loader,
                device,
                use_amp=use_amp,
                channels_last=bool(args.channels_last),
            )
            batches = max(len(train_loader), 1)
            elapsed = time.time() - epoch_start
            thresholds = metrics["thresholds"]
            e1_f1 = metrics["endpoint1_component"]["f1"]
            e2_f1 = metrics["endpoint2_component"]["f1"]
            cross_f1 = metrics["crossing_component"]["f1"]
            train_loss = totals["loss"] / batches
            print(
                f"epoch {epoch:03d}/{args.epochs} loss {train_loss:.4f} "
                f"val_score {metrics['score']:.4f} val_iou {metrics['iou']:.4f} val_dice {metrics['dice']:.4f} "
                f"body_iou {metrics['body_iou']:.4f} e1_f1 {e1_f1:.4f} e2_f1 {e2_f1:.4f} "
                f"cross_f1 {cross_f1:.4f} thresholds {'/'.join(f'{value:.2f}' for value in thresholds)} "
                f"lr {optimizer.param_groups[0]['lr']:.3g} time {elapsed:.1f}s"
            )
            row = {
                "epoch": epoch,
                "loss": train_loss,
                "body_loss": totals["body"] / batches,
                "endpoint1_loss": totals["endpoint1"] / batches,
                "endpoint2_loss": totals["endpoint2"] / batches,
                "crossing_loss": totals["crossing"] / batches,
                "boundary_loss": totals["boundary"] / batches,
                "val_score": metrics["score"],
                "val_iou": metrics["iou"],
                "val_dice": metrics["dice"],
                "body_iou": metrics["body_iou"],
                "endpoint1_iou": metrics["endpoint1_iou"],
                "endpoint2_iou": metrics["endpoint2_iou"],
                "crossing_iou": metrics["crossing_iou"],
                "endpoint1_f1": e1_f1,
                "endpoint2_f1": e2_f1,
                "crossing_f1": cross_f1,
                "threshold_body": thresholds[0],
                "threshold_endpoint1": thresholds[1],
                "threshold_endpoint2": thresholds[2],
                "threshold_crossing": thresholds[3],
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
            }
            history_writer.writerow(row)
            history_stream.flush()

            improved = metrics["score"] > best_score + float(args.min_delta)
            if improved:
                best_score = float(metrics["score"])
                best_epoch = epoch
                epochs_without_improvement = 0
                best_payload = checkpoint_payload(
                    ema.model.state_dict(),
                    args,
                    metrics,
                    epoch,
                    channel_alpha,
                    extra={"dataset_sha256": snapshot["dataset_sha256"]},
                )
                atomic_torch_save(best_payload, args.output)
                print(
                    f"saved {args.output} val_score={best_score:.4f} val_iou={metrics['iou']:.4f} "
                    f"thresholds={'/'.join(f'{value:.2f}' for value in thresholds)}"
                )
            else:
                epochs_without_improvement += 1

            last_payload = checkpoint_payload(
                model.state_dict(),
                args,
                metrics,
                epoch,
                channel_alpha,
                extra={
                    "ema_state": ema.model.state_dict(),
                    "ema_updates": int(ema.updates),
                    "ema_effective_decay": float(ema.effective_decay),
                    "dataset_sha256": snapshot["dataset_sha256"],
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )
            atomic_torch_save(last_payload, last_path)
            if int(args.early_stop) > 0 and epochs_without_improvement >= int(args.early_stop):
                termination = early_stop_summary(
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_score=best_score,
                    current_score=metrics["score"],
                    patience=args.early_stop,
                    min_delta=args.min_delta,
                    metrics=metrics,
                )
                print_early_stop_summary(termination)
                break
    finally:
        history_stream.close()

    if termination is None:
        termination = {
            "type": "maximum_epochs",
            "stopped_epoch": int(args.epochs),
            "configured_epochs": int(args.epochs),
        }
        print(f"TRAINING COMPLETE: reached the configured maximum of {int(args.epochs)} epochs.")

    try:
        final_snapshot = dataset_snapshot(args.dataset, pairs_by_split)
        dataset_unchanged = final_snapshot["dataset_sha256"] == snapshot["dataset_sha256"]
    except Exception as exc:
        dataset_unchanged = False
        final_snapshot = {"error": str(exc)}
    if not dataset_unchanged:
        termination = {
            "type": "dataset_changed",
            "message": "Selected training or validation files changed while the run was active.",
        }
        run_info.update({
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "termination": termination,
            "final_dataset_snapshot": final_snapshot,
        })
        run_path.write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(
            "Dataset changed during training; the run is not reproducible and its checkpoint must not be used. "
            "Stop annotation edits and start a fresh run."
        )

    best_payload = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    test_metrics = None
    if test_loader is not None:
        validation_thresholds = best_payload.get("metrics", {}).get("thresholds", (0.5,) * OUTPUT_CHANNEL_COUNT)
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            use_amp=use_amp,
            channels_last=bool(args.channels_last),
            fixed_thresholds=validation_thresholds,
        )
        print(
            f"locked_test score={test_metrics['score']:.4f} body_iou={test_metrics['body_iou']:.4f} "
            f"e1_f1={test_metrics['endpoint1_component']['f1']:.4f} "
            f"e2_f1={test_metrics['endpoint2_component']['f1']:.4f} "
            f"cross_f1={test_metrics['crossing_component']['f1']:.4f}"
        )
    run_info.update({
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_payload.get("metrics"),
        "locked_test_metrics": test_metrics,
        "termination": termination,
    })
    run_path.write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return best_payload.get("metrics"), test_metrics


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
