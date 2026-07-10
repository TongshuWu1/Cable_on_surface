import argparse
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cable_pidnet import IMAGENET_MEAN, IMAGENET_STD, PIDNetSmallBinary, require_torch, resolve_device

require_torch()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_IMAGE_SIZE = "1280x720"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PIDNet-style binary cable segmentation model.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "datasets/two_cable_pidnet", help="Dataset root with images/ and masks/ folders.")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "models/pidnet_two_cable_best.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--imgsz",
        type=parse_image_size,
        default=parse_image_size(DEFAULT_IMAGE_SIZE),
        help="Training image size. Use 1280x720 for full ZED HD720, or a single value like 512 for square training.",
    )
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--cable-count", type=int, default=1, help="Number of separately labeled cable instances in the mask. Use 2 for two independent cables.")
    parser.add_argument(
        "--endpoint-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train one endpoint channel per cable. Mask labels are N+1..2N for endpoints.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA automatic mixed precision.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--boundary-weight", type=float, default=0.20)
    return parser.parse_args()


class CableMaskDataset(Dataset):
    def __init__(self, pairs, image_size=DEFAULT_IMAGE_SIZE, augment=False, cable_count=1, endpoint_labels=True):
        self.pairs = list(pairs)
        self.image_size = parse_image_size(image_size)
        self.augment = bool(augment)
        self.cable_count = max(1, int(cable_count))
        self.endpoint_labels = bool(endpoint_labels)
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        image, mask = resize_pair(image, mask, self.image_size)

        if self.augment:
            image, mask = augment_pair(image, mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        mask = instance_mask_channels(mask, self.cable_count, endpoint_labels=self.endpoint_labels)
        boundary = mask_boundary(np.any(mask > 0.5, axis=0).astype(np.float32))

        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        boundary = torch.from_numpy(boundary[None, :, :])
        return image, mask, boundary


def find_dataset_splits(dataset_root, val_split, seed):
    dataset_root = Path(dataset_root)
    train_image_dir = dataset_root / "images" / "train"
    train_mask_dir = dataset_root / "masks" / "train"
    val_image_dir = dataset_root / "images" / "val"
    val_mask_dir = dataset_root / "masks" / "val"
    if train_image_dir.exists() and train_mask_dir.exists():
        train_pairs = match_image_mask_pairs(train_image_dir, train_mask_dir)
        val_pairs = []
        if val_image_dir.exists() and val_mask_dir.exists():
            val_pairs = match_image_mask_pairs(val_image_dir, val_mask_dir)
        if not val_pairs:
            train_pairs, val_pairs = split_pairs(train_pairs, val_split, seed)
        return train_pairs, val_pairs

    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    if image_dir.exists() and mask_dir.exists():
        return split_pairs(match_image_mask_pairs(image_dir, mask_dir), val_split, seed)
    raise FileNotFoundError("Expected dataset/images and dataset/masks folders.")


def match_image_mask_pairs(image_dir, mask_dir):
    images = [path for path in Path(image_dir).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    mask_by_stem = {path.stem: path for path in Path(mask_dir).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    for image_path in sorted(images):
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def split_pairs(pairs, val_split, seed):
    pairs = list(pairs)
    random.Random(int(seed)).shuffle(pairs)
    val_count = int(round(len(pairs) * float(np.clip(val_split, 0.0, 0.8))))
    val_count = min(max(1 if len(pairs) > 1 else 0, val_count), max(0, len(pairs) - 1))
    return pairs[val_count:], pairs[:val_count]


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
                raise argparse.ArgumentTypeError("Use image size as WIDTHxHEIGHT, for example 1280x720.")
            width, height = parts
        else:
            width = height = text

    try:
        width = int(width)
        height = int(height)
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
    image_interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=image_interpolation)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_x = target_w - new_w
    pad_y = target_h - new_h
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return image, mask


def augment_pair(image, mask):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < 0.2:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    if random.random() < 0.7:
        alpha = random.uniform(0.75, 1.30)
        beta = random.uniform(-20.0, 20.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        noise = np.random.normal(0.0, random.uniform(2.0, 8.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, mask


def instance_mask_channels(mask, cable_count, endpoint_labels=True):
    cable_count = max(1, int(cable_count))
    output_channels = cable_count * (2 if bool(endpoint_labels) else 1)
    mask = np.asarray(mask, dtype=np.uint8)
    channels = np.zeros((output_channels, mask.shape[0], mask.shape[1]), dtype=np.float32)
    if cable_count == 1:
        body = mask == 1
        if not np.any(body) and np.any(mask > 127):
            body = mask > 127
        channels[0] = body.astype(np.float32)
        if bool(endpoint_labels) and output_channels > 1:
            channels[1] = (mask == 2).astype(np.float32)
        return channels

    for label in range(1, cable_count + 1):
        channels[label - 1] = (mask == label).astype(np.float32)
        if bool(endpoint_labels):
            channels[cable_count + label - 1] = (mask == cable_count + label).astype(np.float32)

    if not np.any(channels) and np.any(mask > 127):
        # Legacy binary masks are still useful as cable 1 examples.
        channels[0] = (mask > 127).astype(np.float32)
    return channels


def mask_boundary(mask):
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    return (grad > 0).astype(np.float32)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(prob * target, dim=(1, 2, 3)) + eps
    denominator = torch.sum(prob + target, dim=(1, 2, 3)) + eps
    return 1.0 - torch.mean(numerator / denominator)


def foreground_balanced_bce(logits, target, max_pos_weight=40.0):
    positive = torch.sum(target)
    negative = torch.sum(1.0 - target)
    if float(positive.detach().cpu()) <= 0.0:
        pos_weight = torch.ones((), dtype=logits.dtype, device=logits.device)
    else:
        pos_weight = torch.clamp(negative / torch.clamp(positive, min=1.0), min=1.0, max=float(max_pos_weight))
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def segmentation_loss(outputs, mask, boundary, boundary_weight):
    seg_logits = outputs["seg"]
    boundary_logits = outputs["boundary"]
    bce = foreground_balanced_bce(seg_logits, mask)
    dice = dice_loss(seg_logits, mask)
    boundary_bce = foreground_balanced_bce(boundary_logits, boundary)
    return bce + dice + float(boundary_weight) * boundary_bce


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    intersection = 0.0
    union = 0.0
    dice_num = 0.0
    dice_den = 0.0
    for image, mask, _boundary in loader:
        image = image.to(device)
        mask = mask.to(device)
        pred = torch.sigmoid(model(image)["seg"]) > 0.5
        target = mask > 0.5
        intersection += torch.sum(pred & target).item()
        union += torch.sum(pred | target).item()
        dice_num += 2.0 * torch.sum(pred & target).item()
        dice_den += torch.sum(pred).item() + torch.sum(target).item()
    return {
        "iou": intersection / max(union, 1.0),
        "dice": dice_num / max(dice_den, 1.0),
    }


def train(args):
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_pairs, val_pairs = find_dataset_splits(args.dataset, args.val_split, args.seed)
    if not train_pairs:
        raise ValueError("Need at least one training image/mask pair.")
    if not val_pairs:
        val_pairs = train_pairs[:]

    cable_count = max(1, int(args.cable_count))
    endpoint_labels = bool(args.endpoint_labels)
    output_channels = cable_count * (2 if endpoint_labels else 1)
    train_loader = DataLoader(
        CableMaskDataset(train_pairs, args.imgsz, augment=True, cable_count=cable_count, endpoint_labels=endpoint_labels),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CableMaskDataset(val_pairs, args.imgsz, augment=False, cable_count=cable_count, endpoint_labels=endpoint_labels),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = PIDNetSmallBinary(base_channels=int(args.base_channels), output_channels=output_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_iou = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(
        f"Training on {len(train_pairs)} images, validating on {len(val_pairs)} images, "
        f"imgsz={image_size_text(args.imgsz)}, cables={cable_count}, "
        f"outputs={output_channels}, endpoint_labels={endpoint_labels}, device={device_name}, amp={use_amp}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for image, mask, boundary in train_loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = segmentation_loss(model(image), mask, boundary, args.boundary_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

        metrics = evaluate(model, val_loader, device)
        train_loss = total_loss / max(len(train_loader), 1)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss {train_loss:.4f} "
            f"val_iou {metrics['iou']:.4f} val_dice {metrics['dice']:.4f} "
            f"time {time.time() - start:.1f}s"
        )
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "model": "pidnet_small_binary",
                        "base_channels": int(args.base_channels),
                        "output_channels": int(output_channels),
                        "label_mode": (
                            "instances_with_endpoints"
                            if endpoint_labels
                            else ("instances" if cable_count > 1 else "binary")
                        ),
                        "cable_count": int(cable_count),
                        "endpoint_channels": bool(endpoint_labels),
                        "imgsz": image_size_text(args.imgsz),
                    },
                    "training": {
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "val_split": float(args.val_split),
                        "num_workers": int(args.num_workers),
                        "boundary_weight": float(args.boundary_weight),
                        "amp": bool(use_amp),
                        "seed": int(args.seed),
                    },
                    "metrics": metrics,
                },
                args.output,
            )
            print(f"saved {args.output} val_iou={best_iou:.4f}")


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
