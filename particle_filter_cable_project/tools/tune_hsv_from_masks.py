import argparse
from pathlib import Path
import re
import sys
import tomllib

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cable_detection import HsvCableDetector, diagonal_gaussian_logpdf, hsv_gaussian_features


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_DATASET = PROJECT_DIR / "datasets/cable_pidnet"
DEFAULT_CONFIG = PROJECT_DIR / "config.toml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit HSV cable thresholds from labeled image/mask pairs.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fit-splits", default="train", help="Comma list of splits used to fit HSV, or 'all'.")
    parser.add_argument("--eval-splits", default="train,val", help="Comma list of splits used for reporting, or 'all'.")
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--method", choices=("gaussian", "grid"), default="gaussian")
    parser.add_argument("--max-pixels", type=int, default=250000, help="Maximum labeled cable pixels sampled for fitting.")
    parser.add_argument("--max-background-pixels", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--write-config", action="store_true", help="Write fitted [hsv] values into config.toml.")
    parser.add_argument("--set-backend-hsv", action="store_true", help="Also set detector.backend = \"hsv\" when writing config.")
    return parser.parse_args()


def main():
    args = parse_args()
    fit_pairs = dataset_pairs(args.dataset, split_names(args.fit_splits))
    eval_pairs = dataset_pairs(args.dataset, split_names(args.eval_splits))
    if not fit_pairs:
        raise ValueError(f"No image/mask pairs found for fit splits in {args.dataset}")
    if not eval_pairs:
        eval_pairs = fit_pairs

    cleanup = detector_cleanup_from_config(args.config)
    samples = collect_labeled_hsv_samples(
        fit_pairs,
        mask_threshold=args.mask_threshold,
        max_pixels=args.max_pixels,
        max_background_pixels=args.max_background_pixels,
        seed=args.seed,
    )
    if len(samples["negative"]) == 0:
        raise ValueError("No background pixels found. Masks should mark only the cable; every other pixel is background.")

    if args.method == "gaussian":
        bounds, fit_metrics = fit_gaussian_hsv_model(samples)
    else:
        bounds, fit_metrics = grid_search_hsv_bounds(samples)
        bounds["mode"] = "range"
    metrics = evaluate_hsv_bounds(eval_pairs, bounds, cleanup, mask_threshold=args.mask_threshold)

    print(
        f"Fit pairs: {len(fit_pairs)} | eval pairs: {len(eval_pairs)} | "
        f"cable pixels sampled: {len(samples['positive'])} | "
        f"background pixels sampled: {len(samples['negative'])}"
    )
    print(format_hsv_block(bounds))
    print(f"Fit sample score: IoU {fit_metrics['iou']:.4f} | Dice {fit_metrics['dice']:.4f}")
    print(
        f"Eval after detector cleanup: IoU {metrics['iou']:.4f} | "
        f"Dice {metrics['dice']:.4f} | predicted px {metrics['predicted']} | target px {metrics['target']}"
    )

    if args.write_config:
        update_config(args.config, bounds, set_backend_hsv=args.set_backend_hsv)
        print(f"Updated {args.config}")


def split_names(value):
    names = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    if not names or "all" in names:
        return ("train", "val")
    return tuple(dict.fromkeys(names))


def dataset_pairs(dataset_root, splits=("train",)):
    dataset_root = Path(dataset_root)
    pairs = []
    for split in splits:
        image_dir = dataset_root / "images" / split
        mask_dir = dataset_root / "masks" / split
        if image_dir.exists() and mask_dir.exists():
            pairs.extend(match_image_mask_pairs(image_dir, mask_dir))

    if not pairs and (dataset_root / "images").exists() and (dataset_root / "masks").exists():
        pairs = match_image_mask_pairs(dataset_root / "images", dataset_root / "masks")
    return pairs


def match_image_mask_pairs(image_dir, mask_dir):
    images = [path for path in Path(image_dir).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    mask_by_stem = {path.stem: path for path in Path(mask_dir).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    for image_path in sorted(images):
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def detector_cleanup_from_config(config_path):
    defaults = {"min_area": 80, "open_kernel": 3, "close_kernel": 5}
    config_path = Path(config_path)
    if not config_path.exists():
        return defaults
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    detector = config.get("detector", {})
    return {
        "min_area": int(detector.get("min_area_px", defaults["min_area"])),
        "open_kernel": int(detector.get("open_kernel", defaults["open_kernel"])),
        "close_kernel": int(detector.get("close_kernel", defaults["close_kernel"])),
    }


def collect_labeled_hsv_samples(
    pairs,
    mask_threshold=127,
    max_pixels=250000,
    max_background_pixels=100000,
    seed=17,
):
    positive_chunks = []
    negative_chunks = []
    positive_total = 0
    negative_total = 0
    rng = np.random.default_rng(int(seed))
    pairs = list(pairs)
    per_image_negative = max(1000, int(max_background_pixels) // max(len(pairs), 1))

    for image_path, mask_path in pairs:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cable = mask > int(mask_threshold)
        background = ~cable
        positive = hsv[cable].reshape(-1, 3)
        negative = hsv[background].reshape(-1, 3)
        positive_total += len(positive)
        negative_total += len(negative)
        if len(positive) > 0:
            positive_chunks.append(positive)
        if len(negative) > per_image_negative > 0:
            negative = negative[rng.choice(len(negative), size=per_image_negative, replace=False)]
        if len(negative) > 0:
            negative_chunks.append(negative)

    if not positive_chunks:
        raise ValueError("No positive cable pixels found in the selected masks.")
    positive = sample_rows(np.vstack(positive_chunks), max_pixels, rng)
    negative = sample_rows(np.vstack(negative_chunks), max_background_pixels, rng) if negative_chunks else np.empty((0, 3), dtype=np.uint8)
    return {
        "positive": np.ascontiguousarray(positive, dtype=np.uint8),
        "negative": np.ascontiguousarray(negative, dtype=np.uint8),
        "positive_total": int(positive_total),
        "negative_total": int(negative_total),
    }


def sample_rows(rows, max_count, rng):
    rows = np.asarray(rows, dtype=np.uint8)
    max_count = max(0, int(max_count))
    if max_count > 0 and len(rows) > max_count:
        indices = rng.choice(len(rows), size=max_count, replace=False)
        rows = rows[np.sort(indices)]
    return rows


def fit_gaussian_hsv_model(samples):
    positive_features = hsv_gaussian_features(samples["positive"])
    negative_features = hsv_gaussian_features(samples["negative"])
    positive_mean, positive_std = gaussian_diag_stats(positive_features)
    negative_mean, negative_std = gaussian_diag_stats(negative_features)
    positive_scores = gaussian_score_features(
        positive_features,
        positive_mean,
        positive_std,
        negative_mean,
        negative_std,
    )
    negative_scores = gaussian_score_features(
        negative_features,
        positive_mean,
        positive_std,
        negative_mean,
        negative_std,
    )
    threshold, metrics = best_score_threshold(positive_scores, negative_scores, samples)
    return (
        {
            "mode": "gaussian",
            "gaussian_threshold": round(float(threshold), 6),
            "gaussian_positive_mean": round_list(positive_mean),
            "gaussian_positive_std": round_list(positive_std),
            "gaussian_negative_mean": round_list(negative_mean),
            "gaussian_negative_std": round_list(negative_std),
        },
        metrics,
    )


def gaussian_diag_stats(features):
    features = np.asarray(features, dtype=np.float32).reshape(-1, 4)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return mean.astype(np.float32), np.maximum(std, 0.025).astype(np.float32)


def gaussian_score_features(features, positive_mean, positive_std, negative_mean, negative_std):
    positive = diagonal_gaussian_logpdf(features, positive_mean, positive_std)
    negative = diagonal_gaussian_logpdf(features, negative_mean, negative_std)
    return positive - negative


def best_score_threshold(positive_scores, negative_scores, samples):
    all_scores = np.concatenate([positive_scores, negative_scores]).astype(np.float64)
    percentiles = np.linspace(0.0, 100.0, 401)
    candidates = np.unique(np.percentile(all_scores, percentiles))
    best_threshold = float(candidates[0])
    best_metrics = None
    best_iou = -1.0
    for threshold in candidates:
        metrics = score_masks(positive_scores >= threshold, negative_scores >= threshold, samples)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def round_list(values, digits=6):
    return [round(float(value), int(digits)) for value in np.asarray(values).reshape(-1)]


def grid_search_hsv_bounds(samples):
    positive = np.asarray(samples["positive"], dtype=np.uint8)
    negative = np.asarray(samples["negative"], dtype=np.uint8)
    if len(positive) == 0:
        raise ValueError("Need positive HSV samples for grid search.")

    hue_candidates = candidate_hue_bounds(positive[:, 0])
    s_candidates = candidate_scalar_bounds(positive[:, 1])
    v_candidates = candidate_scalar_bounds(positive[:, 2])

    positive_masks = precompute_candidate_masks(positive, hue_candidates, s_candidates, v_candidates)
    negative_masks = precompute_candidate_masks(negative, hue_candidates, s_candidates, v_candidates)
    best_bounds = None
    best_metrics = None
    best_score = -1.0

    for h_index, h_bounds in enumerate(hue_candidates):
        positive_h = positive_masks["h"][h_index]
        negative_h = negative_masks["h"][h_index]
        for s_index, s_bounds in enumerate(s_candidates):
            positive_hs = positive_h & positive_masks["s"][s_index]
            negative_hs = negative_h & negative_masks["s"][s_index]
            for v_index, v_bounds in enumerate(v_candidates):
                bounds = {
                    "h_min": h_bounds[0],
                    "h_max": h_bounds[1],
                    "s_min": s_bounds[0],
                    "s_max": s_bounds[1],
                    "v_min": v_bounds[0],
                    "v_max": v_bounds[1],
                }
                metrics = score_masks(
                    positive_hs & positive_masks["v"][v_index],
                    negative_hs & negative_masks["v"][v_index],
                    samples,
                )
                score = metrics["iou"]
                if score > best_score:
                    best_score = score
                    best_bounds = bounds
                    best_metrics = metrics

    return best_bounds, best_metrics


def candidate_hue_bounds(hue):
    candidates = []
    seen = set()
    for trim in (0.0, 1.0, 2.0, 5.0, 8.0, 12.0):
        coverage = 1.0 - 2.0 * trim / 100.0
        for margin in (0.0, 2.0, 4.0, 7.0):
            bounds = shortest_hue_interval(hue, coverage=coverage, margin=margin)
            if bounds not in seen:
                seen.add(bounds)
                candidates.append(bounds)
    return candidates


def candidate_scalar_bounds(values):
    candidates = []
    seen = set()
    for trim in (0.0, 1.0, 2.0, 5.0, 8.0, 12.0):
        for margin in (0.0, 6.0, 12.0, 20.0):
            bounds = scalar_bounds(values, trim, margin=margin, limit=255)
            if bounds not in seen:
                seen.add(bounds)
                candidates.append(bounds)
    return candidates


def precompute_candidate_masks(samples, hue_candidates, s_candidates, v_candidates):
    samples = np.asarray(samples, dtype=np.uint8)
    return {
        "h": [hue_in_range(samples[:, 0], h_min, h_max) for h_min, h_max in hue_candidates],
        "s": [channel_in_range(samples[:, 1], s_min, s_max) for s_min, s_max in s_candidates],
        "v": [channel_in_range(samples[:, 2], v_min, v_max) for v_min, v_max in v_candidates],
    }


def hue_in_range(hue, h_min, h_max):
    hue = np.asarray(hue, dtype=np.uint8)
    if int(h_min) <= int(h_max):
        return (hue >= int(h_min)) & (hue <= int(h_max))
    return (hue >= int(h_min)) | (hue <= int(h_max))


def channel_in_range(values, lower, upper):
    values = np.asarray(values, dtype=np.uint8)
    return (values >= int(lower)) & (values <= int(upper))


def score_masks(positive_predicted, negative_predicted, samples):
    positive_predicted = np.asarray(positive_predicted, dtype=bool)
    negative_predicted = np.asarray(negative_predicted, dtype=bool)
    positive_total = max(1, int(samples.get("positive_total", len(positive_predicted))))
    negative_total = max(0, int(samples.get("negative_total", len(negative_predicted))))
    tp = int(np.count_nonzero(positive_predicted)) * positive_total / max(1, len(positive_predicted))
    fp = int(np.count_nonzero(negative_predicted)) * negative_total / max(1, len(negative_predicted))
    fn = positive_total - tp
    iou = tp / max(tp + fp + fn, 1.0)
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    return {"iou": float(iou), "dice": float(dice), "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def shortest_hue_interval(hue, coverage=0.96, margin=3.0):
    values = np.sort(np.mod(np.asarray(hue, dtype=np.float64), 180.0))
    if len(values) == 0:
        raise ValueError("Need at least one hue sample.")

    coverage = float(np.clip(coverage, 0.50, 1.0))
    window = int(np.ceil(len(values) * coverage))
    window = min(max(1, window), len(values))
    extended = np.concatenate([values, values + 180.0])
    starts = np.arange(len(values))
    ends = starts + window - 1
    widths = extended[ends] - extended[starts]
    best = int(np.argmin(widths))
    start = extended[best] - float(margin)
    end = extended[best + window - 1] + float(margin)
    if end - start >= 179.0:
        return 0, 179
    return int(np.floor(start) % 180), int(np.ceil(end) % 180)


def scalar_bounds(values, trim_percent, margin=0.0, limit=255):
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(values, trim_percent)) - float(margin)
    hi = float(np.percentile(values, 100.0 - trim_percent)) + float(margin)
    return int(np.clip(np.floor(lo), 0, limit)), int(np.clip(np.ceil(hi), 0, limit))


def evaluate_hsv_bounds(pairs, bounds, cleanup=None, mask_threshold=127):
    cleanup = dict(cleanup or {})
    detector = HsvCableDetector(**bounds, **cleanup)
    intersection = 0
    union = 0
    predicted_total = 0
    target_total = 0

    for image_path, mask_path in pairs:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        predicted = detector.detect(image).mask > 0
        target = mask > int(mask_threshold)
        intersection += int(np.count_nonzero(predicted & target))
        union += int(np.count_nonzero(predicted | target))
        predicted_total += int(np.count_nonzero(predicted))
        target_total += int(np.count_nonzero(target))

    dice_den = predicted_total + target_total
    return {
        "iou": intersection / max(union, 1),
        "dice": 2.0 * intersection / max(dice_den, 1),
        "intersection": intersection,
        "union": union,
        "predicted": predicted_total,
        "target": target_total,
    }


def format_hsv_block(bounds):
    lines = ["Recommended [hsv]:"]
    keys = (
        "mode",
        "h_min",
        "h_max",
        "s_min",
        "s_max",
        "v_min",
        "v_max",
        "gaussian_threshold",
        "gaussian_positive_mean",
        "gaussian_positive_std",
        "gaussian_negative_mean",
        "gaussian_negative_std",
    )
    for key in keys:
        if key in bounds:
            lines.append(f"{key} = {format_toml_value(bounds[key])}")
    return "\n".join(lines)


def update_config(config_path, hsv_bounds, set_backend_hsv=False):
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    text = update_toml_section(text, "hsv", hsv_bounds)
    if set_backend_hsv:
        text = update_toml_section(text, "detector", {"backend": "hsv"})
    config_path.write_text(text, encoding="utf-8")


def update_toml_section(text, section, values):
    lines = text.splitlines()
    header_re = re.compile(r"^\s*\[([A-Za-z0-9_.-]+)\]\s*$")
    key_re = re.compile(r"^(\s*)([A-Za-z0-9_.-]+)(\s*=\s*)(.*)$")
    start = None
    end = len(lines)

    for index, line in enumerate(lines):
        match = header_re.match(line)
        if match and match.group(1) == section:
            start = index
            continue
        if start is not None and index > start and match:
            end = index
            break

    formatted = {key: format_toml_value(value) for key, value in values.items()}
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {value}" for key, value in formatted.items())
        return "\n".join(lines) + "\n"

    found = set()
    for index in range(start + 1, end):
        match = key_re.match(lines[index])
        if not match:
            continue
        indent, key, equals, _old_value = match.groups()
        if key in formatted:
            lines[index] = f"{indent}{key}{equals}{formatted[key]}"
            found.add(key)

    insert_at = end
    for key, value in formatted.items():
        if key not in found:
            lines.insert(insert_at, f"{key} = {value}")
            insert_at += 1
    return "\n".join(lines) + "\n"


def format_toml_value(value):
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(format_toml_value(item) for item in list(value)) + "]"
    return str(int(value)) if isinstance(value, (np.integer, int)) else str(value)


if __name__ == "__main__":
    main()
