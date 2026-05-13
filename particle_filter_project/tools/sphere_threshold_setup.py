import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


PROFILE_PATH = Path(__file__).with_name("sphere_color_profile.json")
CALIB_IMAGE_DIR = Path(__file__).with_name("calibration_images")

WINDOW_LIVE = "Sphere Threshold Setup - Live"
WINDOW_CAPTURE = "Sphere Threshold Setup - Captured"
WINDOW_ROI = "Select Sphere ROI"

DISPLAY_SCALE = 0.55


RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1200": sl.RESOLUTION.HD1200,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "SVGA": sl.RESOLUTION.SVGA,
    "VGA": sl.RESOLUTION.VGA,
}


class AppState:
    def __init__(self):
        self.samples = []
        self.imported_ranges = []
        self.captured_frames = []
        self.selected_capture_idx = -1
        self.latest_bgr = None
        self.frame_count = 0
        self.extra_h = 4
        self.extra_s = 20
        self.extra_v = 20
        self.min_area = 250
        self.min_circularity_pct = 55
        self.max_aspect_pct = 160
        self.open_kernel = 3
        self.close_kernel = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate robust RGB/HSV thresholding for colored sphere detection.",
    )
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--image", action="append", default=[], help="Calibration image path. Can be repeated.")
    parser.add_argument("--resolution", choices=RESOLUTIONS, default="HD720")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--display-scale", type=float, default=DISPLAY_SCALE)
    return parser.parse_args()


def hue_circular_mean(h_values):
    if len(h_values) == 0:
        return 0.0

    angles = h_values.astype(np.float32) / 180.0 * 2.0 * np.pi
    angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle / (2.0 * np.pi) * 180.0)


def circular_hue_distance(h1, h2):
    distance = abs(float(h1) - float(h2))
    return min(distance, 180.0 - distance)


def make_hsv_ranges(h_center, s_center, v_center, h_margin, s_margin, v_margin):
    h_low = int(round(h_center - h_margin))
    h_high = int(round(h_center + h_margin))
    s_low = int(max(0, round(s_center - s_margin)))
    s_high = int(min(255, round(s_center + s_margin)))
    v_low = int(max(0, round(v_center - v_margin)))
    v_high = int(min(255, round(v_center + v_margin)))

    if h_low < 0:
        return [
            {"lower": [0, s_low, v_low], "upper": [h_high, s_high, v_high]},
            {"lower": [180 + h_low, s_low, v_low], "upper": [179, s_high, v_high]},
        ]
    if h_high > 179:
        return [
            {"lower": [h_low, s_low, v_low], "upper": [179, s_high, v_high]},
            {"lower": [0, s_low, v_low], "upper": [h_high - 180, s_high, v_high]},
        ]
    return [{"lower": [h_low, s_low, v_low], "upper": [h_high, s_high, v_high]}]


def expand_range(hsv_range, state):
    lower = np.array(hsv_range["lower"], dtype=np.int16)
    upper = np.array(hsv_range["upper"], dtype=np.int16)
    lower[1] = max(0, lower[1] - state.extra_s)
    lower[2] = max(0, lower[2] - state.extra_v)
    upper[1] = min(255, upper[1] + state.extra_s)
    upper[2] = min(255, upper[2] + state.extra_v)

    # Hue expansion is intentionally conservative. If it crosses the 0/179 edge,
    # make_hsv_ranges already produced split ranges during sampling.
    lower[0] = max(0, lower[0] - state.extra_h)
    upper[0] = min(179, upper[0] + state.extra_h)
    return {"lower": lower.astype(int).tolist(), "upper": upper.astype(int).tolist()}


def get_all_hsv_ranges(state, expanded=True):
    ranges = list(state.imported_ranges)
    for sample in state.samples:
        ranges.extend(sample.get("hsv_ranges", []))
    if not expanded:
        return ranges
    return [expand_range(item, state) for item in ranges]


def sample_roi_color(frame_bgr, source_name, source_path=None):
    clone = frame_bgr.copy()
    cv2.putText(
        clone,
        "Select the ball only. ENTER/SPACE confirm. C cancels.",
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.namedWindow(WINDOW_ROI, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(WINDOW_ROI, clone, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW_ROI)

    x, y, w, h = roi
    if w <= 2 or h <= 2:
        print("ROI cancelled or too small.")
        return None

    roi_bgr = frame_bgr[y:y + h, x:x + w]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    hsv_pixels = roi_hsv.reshape(-1, 3).astype(np.float32)
    rgb_pixels = roi_rgb.reshape(-1, 3).astype(np.float32)

    s_vals = hsv_pixels[:, 1]
    v_vals = hsv_pixels[:, 2]
    valid = (s_vals > 35) & (v_vals > 35)
    if np.count_nonzero(valid) < 30:
        valid = np.ones(len(hsv_pixels), dtype=bool)

    hsv_valid = hsv_pixels[valid]
    rgb_valid = rgb_pixels[valid]

    if len(hsv_valid) >= 80:
        sat_cutoff = np.percentile(hsv_valid[:, 1], 35)
        val_cutoff = np.percentile(hsv_valid[:, 2], 12)
        keep = (hsv_valid[:, 1] >= sat_cutoff) & (hsv_valid[:, 2] >= val_cutoff)
        hsv_valid = hsv_valid[keep]
        rgb_valid = rgb_valid[keep]

    if len(hsv_valid) < 20:
        print("Not enough usable color pixels in ROI.")
        return None

    h_center = hue_circular_mean(hsv_valid[:, 0])
    s_center = float(np.median(hsv_valid[:, 1]))
    v_center = float(np.median(hsv_valid[:, 2]))
    hue_dist = np.array(
        [circular_hue_distance(value, h_center) for value in hsv_valid[:, 0]],
        dtype=np.float32,
    )
    h_margin = int(np.clip(max(6, np.percentile(hue_dist, 92) + 3), 6, 35))
    s_margin = int(np.clip(max(35, np.std(hsv_valid[:, 1]) * 2.1), 35, 140))
    v_margin = int(np.clip(max(40, np.std(hsv_valid[:, 2]) * 2.1), 40, 155))

    sample = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": source_name,
        "source_path": str(source_path) if source_path else None,
        "roi_xywh": [int(x), int(y), int(w), int(h)],
        "num_pixels_used": int(len(hsv_valid)),
        "rgb_median": [int(round(v)) for v in np.median(rgb_valid, axis=0)],
        "rgb_std": [float(v) for v in np.std(rgb_valid, axis=0)],
        "hsv_median": [float(h_center), float(s_center), float(v_center)],
        "hsv_percentile_10": [float(v) for v in np.percentile(hsv_valid, 10, axis=0)],
        "hsv_percentile_90": [float(v) for v in np.percentile(hsv_valid, 90, axis=0)],
        "hsv_margin": [int(h_margin), int(s_margin), int(v_margin)],
        "hsv_ranges": make_hsv_ranges(h_center, s_center, v_center, h_margin, s_margin, v_margin),
    }

    print("\nAdded sphere color sample:")
    print(f"  Source: {source_name}")
    print(f"  ROI: {sample['roi_xywh']}")
    print(f"  RGB median: {sample['rgb_median']}")
    print(f"  HSV median: H={h_center:.1f}, S={s_center:.1f}, V={v_center:.1f}")
    print(f"  HSV margin: {sample['hsv_margin']}")
    print(f"  HSV ranges: {sample['hsv_ranges']}")
    return sample


def create_mask_from_ranges(bgr, hsv_ranges, state):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for item in hsv_ranges:
        lower = np.array(item["lower"], dtype=np.uint8)
        upper = np.array(item["upper"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    open_size = _odd_kernel_size(state.open_kernel)
    close_size = _odd_kernel_size(state.close_kernel)
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _odd_kernel_size(value):
    value = int(max(0, value))
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def detect_sphere_blob(mask, state):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(state.min_area):
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < float(state.min_circularity_pct) / 100.0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > float(state.max_aspect_pct) / 100.0:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        fill_ratio = area / max(np.pi * radius * radius, 1.0)
        score = area * circularity * np.clip(fill_ratio, 0.35, 1.2)
        candidates.append(
            {
                "contour": contour,
                "center_xy": (float(cx), float(cy)),
                "radius_px": float(radius),
                "area": area,
                "circularity": circularity,
                "fill_ratio": float(fill_ratio),
                "aspect": float(aspect),
                "score": float(score),
            }
        )

    if not candidates:
        return None, []
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0], candidates


def overlay_detection(bgr, mask, detection, candidates):
    overlay = bgr.copy()
    if mask is not None and np.any(mask):
        pixels = mask > 0
        tint = np.zeros_like(overlay)
        tint[:, :, 1] = 180
        tint[:, :, 2] = 255
        overlay[pixels] = cv2.addWeighted(overlay[pixels], 0.60, tint[pixels], 0.40, 0.0)

    for candidate in candidates[:8]:
        cv2.drawContours(overlay, [candidate["contour"]], -1, (80, 170, 255), 1, cv2.LINE_AA)

    if detection is not None:
        cx, cy = detection["center_xy"]
        radius = detection["radius_px"]
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(overlay, center, int(round(radius)), (0, 255, 80), 2, cv2.LINE_AA)
        cv2.circle(overlay, center, 4, (0, 255, 255), -1, cv2.LINE_AA)
        label = (
            f"ball r={radius:.0f}px area={detection['area']:.0f} "
            f"circ={detection['circularity']:.2f}"
        )
        cv2.putText(
            overlay,
            label,
            (max(12, center[0] - 90), max(28, center[1] - int(radius) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )

    return overlay


def make_display(bgr, state):
    hsv_ranges = get_all_hsv_ranges(state, expanded=True)
    if hsv_ranges:
        mask = create_mask_from_ranges(bgr, hsv_ranges, state)
        detection, candidates = detect_sphere_blob(mask, state)
        overlay = overlay_detection(bgr, mask, detection, candidates)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        detection = None
        candidates = []
        overlay = bgr.copy()
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    lines = [
        f"samples {len(state.samples)} | ranges {len(hsv_ranges)} | candidates {len(candidates)} | detect {'yes' if detection else 'no'}",
        "p capture | i sample capture | a sample live | n/b browse | u undo | x clear | s save | q quit",
        f"extra HSV {state.extra_h}/{state.extra_s}/{state.extra_v} | min area {state.min_area} | circularity {state.min_circularity_pct}%",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (24, 40 + idx * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    combined = np.hstack([overlay, mask_bgr])
    if DISPLAY_SCALE != 1.0:
        combined = cv2.resize(combined, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
    return combined


def capture_current_frame(state):
    if state.latest_bgr is None:
        print("No live frame available yet.")
        return

    CALIB_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = CALIB_IMAGE_DIR / f"sphere_calib_{time.strftime('%Y%m%d_%H%M%S')}_{len(state.captured_frames):03d}.png"
    frame = state.latest_bgr.copy()
    cv2.imwrite(str(filename), frame)
    state.captured_frames.append(
        {
            "path": str(filename),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "bgr": frame,
        }
    )
    state.selected_capture_idx = len(state.captured_frames) - 1
    print(f"Captured calibration image: {filename}")
    show_capture_preview(state)


def get_selected_capture(state):
    if not state.captured_frames:
        return None
    if state.selected_capture_idx < 0:
        state.selected_capture_idx = 0
    state.selected_capture_idx %= len(state.captured_frames)
    return state.captured_frames[state.selected_capture_idx]


def show_capture_preview(state):
    item = get_selected_capture(state)
    if item is None:
        print("No captured images yet.")
        return
    display = make_display(item["bgr"].copy(), state)
    cv2.namedWindow(WINDOW_CAPTURE, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_CAPTURE, display)


def sample_from_selected_capture(state):
    item = get_selected_capture(state)
    if item is None:
        print("No captured image selected. Press p first or run with --image.")
        return

    sample = sample_roi_color(
        item["bgr"].copy(),
        source_name=f"captured_image_{state.selected_capture_idx + 1}",
        source_path=item["path"],
    )
    if sample is not None:
        state.samples.append(sample)
        show_capture_preview(state)


def build_profile(state):
    return {
        "profile_type": "sphere_color_hsv",
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": state.samples,
        "imported_ranges": state.imported_ranges,
        "hsv_ranges": get_all_hsv_ranges(state, expanded=False),
        "range_expansion": {
            "h_extra": int(state.extra_h),
            "s_extra": int(state.extra_s),
            "v_extra": int(state.extra_v),
        },
        "blob_filter": {
            "min_area": int(state.min_area),
            "min_circularity": float(state.min_circularity_pct) / 100.0,
            "max_aspect_ratio": float(state.max_aspect_pct) / 100.0,
            "open_kernel": int(_odd_kernel_size(state.open_kernel)),
            "close_kernel": int(_odd_kernel_size(state.close_kernel)),
        },
        "notes": [
            "OpenCV HSV uses H in [0, 179] and S/V in [0, 255].",
            "Use multiple samples from different distances and lighting conditions.",
            "The tracker should OR all hsv_ranges, apply range_expansion, then filter blobs by blob_filter.",
        ],
    }


def save_profile(state, path):
    path = Path(path)
    if not get_all_hsv_ranges(state, expanded=False):
        print("No HSV samples/ranges to save yet.")
        return
    profile = build_profile(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved sphere threshold profile: {path}")
    print(f"  samples: {len(state.samples)}")
    print(f"  hsv ranges: {len(profile['hsv_ranges'])}")


def load_existing_profile(state, path):
    path = Path(path)
    if not path.exists():
        print("No existing sphere threshold profile found. Starting fresh.")
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        state.samples = profile.get("samples", [])
        state.imported_ranges = profile.get("imported_ranges", [])
        if not state.samples:
            state.imported_ranges = profile.get("hsv_ranges", [])
        expansion = profile.get("range_expansion", {})
        state.extra_h = int(expansion.get("h_extra", state.extra_h))
        state.extra_s = int(expansion.get("s_extra", state.extra_s))
        state.extra_v = int(expansion.get("v_extra", state.extra_v))
        blob = profile.get("blob_filter", {})
        state.min_area = int(blob.get("min_area", state.min_area))
        state.min_circularity_pct = int(round(float(blob.get("min_circularity", 0.55)) * 100))
        state.max_aspect_pct = int(round(float(blob.get("max_aspect_ratio", 1.60)) * 100))
        state.open_kernel = int(blob.get("open_kernel", state.open_kernel))
        state.close_kernel = int(blob.get("close_kernel", state.close_kernel))
        print(f"Loaded existing profile: {len(get_all_hsv_ranges(state, expanded=False))} HSV ranges.")
    except Exception as exc:
        print(f"Could not load existing profile: {exc}")


def add_image_captures(state, paths):
    for image_path in paths:
        path = Path(image_path)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Could not read image: {path}")
            continue
        state.captured_frames.append(
            {
                "path": str(path),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bgr": bgr,
            }
        )
    if state.captured_frames:
        state.selected_capture_idx = 0
        show_capture_preview(state)


def create_trackbars(state):
    cv2.namedWindow(WINDOW_LIVE, cv2.WINDOW_NORMAL)

    def noop(_value):
        return

    cv2.createTrackbar("H extra", WINDOW_LIVE, state.extra_h, 40, noop)
    cv2.createTrackbar("S extra", WINDOW_LIVE, state.extra_s, 120, noop)
    cv2.createTrackbar("V extra", WINDOW_LIVE, state.extra_v, 120, noop)
    cv2.createTrackbar("Min area x10", WINDOW_LIVE, max(1, state.min_area // 10), 1000, noop)
    cv2.createTrackbar("Min circularity %", WINDOW_LIVE, state.min_circularity_pct, 100, noop)
    cv2.createTrackbar("Max aspect %", WINDOW_LIVE, state.max_aspect_pct, 300, noop)
    cv2.createTrackbar("Open kernel", WINDOW_LIVE, state.open_kernel, 15, noop)
    cv2.createTrackbar("Close kernel", WINDOW_LIVE, state.close_kernel, 21, noop)


def read_trackbars(state):
    state.extra_h = cv2.getTrackbarPos("H extra", WINDOW_LIVE)
    state.extra_s = cv2.getTrackbarPos("S extra", WINDOW_LIVE)
    state.extra_v = cv2.getTrackbarPos("V extra", WINDOW_LIVE)
    state.min_area = max(10, cv2.getTrackbarPos("Min area x10", WINDOW_LIVE) * 10)
    state.min_circularity_pct = max(1, cv2.getTrackbarPos("Min circularity %", WINDOW_LIVE))
    state.max_aspect_pct = max(100, cv2.getTrackbarPos("Max aspect %", WINDOW_LIVE))
    state.open_kernel = cv2.getTrackbarPos("Open kernel", WINDOW_LIVE)
    state.close_kernel = cv2.getTrackbarPos("Close kernel", WINDOW_LIVE)


def open_zed(args):
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[args.resolution]
    init.camera_fps = args.fps
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.1
    init.depth_maximum_distance = 3.0
    status = zed.open(init)
    print("Open status:", status)
    if status != sl.ERROR_CODE.SUCCESS:
        return None
    return zed


def print_controls():
    print("\nControls:")
    print("  p       capture current live frame")
    print("  i       sample sphere ROI from selected captured image")
    print("  a       sample sphere ROI from current live frame")
    print("  n/b     next/previous captured image")
    print("  u       undo last color sample")
    print("  x       clear samples/imported ranges")
    print("  s       save sphere_color_profile.json")
    print("  q       quit")
    print("\nTrackbars adjust HSV expansion and blob filters live.")
    print("Recommended: capture/sample the ball in near, far, shadow, and bright positions.\n")


def main():
    global DISPLAY_SCALE
    args = parse_args()
    DISPLAY_SCALE = max(0.1, float(args.display_scale))
    state = AppState()
    load_existing_profile(state, args.profile)
    create_trackbars(state)
    add_image_captures(state, args.image)
    print_controls()

    zed = None if args.image else open_zed(args)
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 60
    runtime.texture_confidence_threshold = 70
    runtime.remove_saturated_areas = False
    left_image = sl.Mat()

    try:
        while True:
            read_trackbars(state)

            if zed is not None and zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                state.latest_bgr = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2BGR)
                state.frame_count += 1

            display_source = state.latest_bgr
            if display_source is None and state.captured_frames:
                display_source = get_selected_capture(state)["bgr"]

            if display_source is not None:
                cv2.imshow(WINDOW_LIVE, make_display(display_source, state))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                capture_current_frame(state)
            elif key == ord("i"):
                sample_from_selected_capture(state)
            elif key == ord("a"):
                if state.latest_bgr is None:
                    print("No live frame available.")
                    continue
                sample = sample_roi_color(state.latest_bgr.copy(), "live_frame")
                if sample is not None:
                    state.samples.append(sample)
            elif key == ord("n"):
                if state.captured_frames:
                    state.selected_capture_idx = (state.selected_capture_idx + 1) % len(state.captured_frames)
                    show_capture_preview(state)
            elif key == ord("b"):
                if state.captured_frames:
                    state.selected_capture_idx = (state.selected_capture_idx - 1) % len(state.captured_frames)
                    show_capture_preview(state)
            elif key == ord("u"):
                if state.samples:
                    removed = state.samples.pop()
                    print(f"Removed sample: {removed.get('source_name')} ROI {removed.get('roi_xywh')}")
            elif key == ord("x"):
                state.samples = []
                state.imported_ranges = []
                print("Cleared color samples and imported ranges.")
            elif key == ord("s"):
                save_profile(state, args.profile)

    finally:
        if zed is not None:
            zed.close()
        left_image.free()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
