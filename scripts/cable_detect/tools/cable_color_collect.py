import cv2
import json
import time
from pathlib import Path

import numpy as np
import pyzed.sl as sl


# ============================================================
# Output
# ============================================================

PROFILE_PATH = Path(__file__).with_name("cable_color_profile.json")
CALIB_IMAGE_DIR = Path(__file__).with_name("calibration_images")


# ============================================================
# ZED camera settings
# ============================================================

CAMERA_RESOLUTION_NAME = "HD720"
CAMERA_RESOLUTION = sl.RESOLUTION.HD720
CAMERA_FPS = 60

BRIGHTNESS = 4
CONTRAST = 4
HUE = 0
SATURATION = 4
SHARPNESS = 4
GAMMA = 5

AUTO_WHITE_BALANCE = True
AUTO_EXPOSURE = True


# ============================================================
# Display / mask settings
# ============================================================

WINDOW_LIVE = "Cable Color Collect - Live"
WINDOW_CAPTURE = "Captured Calibration Image"
WINDOW_ROI = "Select Cable ROI"

DISPLAY_SCALE = 0.45
MIN_COMPONENT_AREA = 20


# ============================================================
# App state
# ============================================================

class AppState:
    def __init__(self):
        self.samples = []
        self.imported_ranges = []

        self.latest_bgr = None
        self.frame_count = 0

        self.captured_frames = []
        self.selected_capture_idx = -1


# ============================================================
# Camera control
# ============================================================

def apply_camera_settings(zed):
    def safe_set(setting, value, name):
        try:
            err = zed.set_camera_settings(setting, value)
            print(f"{name}: {value} -> {err}")
        except Exception as e:
            print(f"Could not set {name}: {e}")

    print("\nApplying camera settings...")

    safe_set(sl.VIDEO_SETTINGS.BRIGHTNESS, BRIGHTNESS, "BRIGHTNESS")
    safe_set(sl.VIDEO_SETTINGS.CONTRAST, CONTRAST, "CONTRAST")
    safe_set(sl.VIDEO_SETTINGS.HUE, HUE, "HUE")
    safe_set(sl.VIDEO_SETTINGS.SATURATION, SATURATION, "SATURATION")
    safe_set(sl.VIDEO_SETTINGS.SHARPNESS, SHARPNESS, "SHARPNESS")
    safe_set(sl.VIDEO_SETTINGS.GAMMA, GAMMA, "GAMMA")

    if AUTO_WHITE_BALANCE:
        safe_set(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1, "WHITEBALANCE_AUTO")
    else:
        safe_set(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0, "WHITEBALANCE_AUTO")

    if AUTO_EXPOSURE:
        safe_set(sl.VIDEO_SETTINGS.EXPOSURE, -1, "AUTO_EXPOSURE")
    else:
        safe_set(sl.VIDEO_SETTINGS.EXPOSURE, 50, "MANUAL_EXPOSURE")

    print("Camera settings applied.\n")


# ============================================================
# HSV tools
# ============================================================

def make_hsv_ranges(h_median, s_median, v_median, h_margin, s_margin, v_margin):
    """
    OpenCV HSV:
        H: 0 to 179
        S: 0 to 255
        V: 0 to 255
    """

    h_low = int(round(h_median - h_margin))
    h_high = int(round(h_median + h_margin))

    s_low = int(max(0, round(s_median - s_margin)))
    s_high = int(min(255, round(s_median + s_margin)))

    v_low = int(max(0, round(v_median - v_margin)))
    v_high = int(min(255, round(v_median + v_margin)))

    ranges = []

    if h_low < 0:
        ranges.append({
            "lower": [0, s_low, v_low],
            "upper": [h_high, s_high, v_high],
        })
        ranges.append({
            "lower": [180 + h_low, s_low, v_low],
            "upper": [179, s_high, v_high],
        })

    elif h_high > 179:
        ranges.append({
            "lower": [h_low, s_low, v_low],
            "upper": [179, s_high, v_high],
        })
        ranges.append({
            "lower": [0, s_low, v_low],
            "upper": [h_high - 180, s_high, v_high],
        })

    else:
        ranges.append({
            "lower": [h_low, s_low, v_low],
            "upper": [h_high, s_high, v_high],
        })

    return ranges


def hue_circular_mean(h_values):
    if len(h_values) == 0:
        return 0.0

    angles = h_values.astype(np.float32) / 180.0 * 2.0 * np.pi

    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    angle = np.arctan2(mean_sin, mean_cos)

    if angle < 0:
        angle += 2.0 * np.pi

    return float(angle / (2.0 * np.pi) * 180.0)


def circular_hue_distance(h1, h2):
    d = abs(float(h1) - float(h2))
    return min(d, 180.0 - d)


def sample_roi_color(frame_bgr, source_name, source_path=None):
    """
    Select one ROI around the cable from a frozen image.
    """

    clone = frame_bgr.copy()

    cv2.putText(
        clone,
        f"Source: {source_name}",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        clone,
        "Select ONLY cable pixels if possible. ENTER/SPACE confirm. C cancel.",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.namedWindow(WINDOW_ROI, cv2.WINDOW_NORMAL)

    roi = cv2.selectROI(
        WINDOW_ROI,
        clone,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(WINDOW_ROI)

    x, y, w, h = roi

    if w <= 2 or h <= 2:
        print("ROI cancelled or too small.")
        return None

    roi_bgr = frame_bgr[y:y + h, x:x + w]

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    hsv_pixels = roi_hsv.reshape(-1, 3).astype(np.float32)
    rgb_pixels = roi_rgb.reshape(-1, 3).astype(np.float32)

    s_vals = hsv_pixels[:, 1]
    v_vals = hsv_pixels[:, 2]

    # Remove dark/gray pixels if possible.
    valid = (s_vals > 30) & (v_vals > 30)

    if np.count_nonzero(valid) < 20:
        valid = np.ones_like(s_vals, dtype=bool)

    hsv_valid = hsv_pixels[valid]
    rgb_valid = rgb_pixels[valid]

    # Keep more saturated pixels to reduce background contamination.
    if len(hsv_valid) >= 40:
        sat_cutoff = np.percentile(hsv_valid[:, 1], 45)
        high_sat = hsv_valid[:, 1] >= sat_cutoff

        hsv_valid = hsv_valid[high_sat]
        rgb_valid = rgb_valid[high_sat]

    if len(hsv_valid) < 10:
        print("Not enough valid pixels in ROI.")
        return None

    h_center = hue_circular_mean(hsv_valid[:, 0])
    s_median = float(np.median(hsv_valid[:, 1]))
    v_median = float(np.median(hsv_valid[:, 2]))

    hue_dists = np.array(
        [circular_hue_distance(h, h_center) for h in hsv_valid[:, 0]],
        dtype=np.float32,
    )

    h_std = float(np.std(hue_dists))
    s_std = float(np.std(hsv_valid[:, 1]))
    v_std = float(np.std(hsv_valid[:, 2]))

    hsv_median = [
        float(h_center),
        float(s_median),
        float(v_median),
    ]

    hsv_std = [
        float(h_std),
        float(s_std),
        float(v_std),
    ]

    hsv_p10 = [
        float(np.percentile(hsv_valid[:, 0], 10)),
        float(np.percentile(hsv_valid[:, 1], 10)),
        float(np.percentile(hsv_valid[:, 2], 10)),
    ]

    hsv_p90 = [
        float(np.percentile(hsv_valid[:, 0], 90)),
        float(np.percentile(hsv_valid[:, 1], 90)),
        float(np.percentile(hsv_valid[:, 2], 90)),
    ]

    rgb_median = np.median(rgb_valid, axis=0)
    rgb_std = np.std(rgb_valid, axis=0)

    # Cable color:
    # Hue should be relatively constrained.
    # S/V need to be wider because distance, lighting, and shadows change them.
    h_margin = int(np.clip(max(8, 2.5 * h_std), 8, 30))
    s_margin = int(np.clip(max(60, 2.3 * s_std), 60, 155))
    v_margin = int(np.clip(max(65, 2.3 * v_std), 65, 170))

    hsv_ranges = make_hsv_ranges(
        h_center,
        s_median,
        v_median,
        h_margin,
        s_margin,
        v_margin,
    )

    sample = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": source_name,
        "source_path": str(source_path) if source_path is not None else None,

        "roi_xywh": [int(x), int(y), int(w), int(h)],
        "num_pixels_used": int(len(hsv_valid)),

        "rgb_median": [int(round(v)) for v in rgb_median],
        "rgb_std": [float(v) for v in rgb_std],

        "hsv_median": [float(v) for v in hsv_median],
        "hsv_std": [float(v) for v in hsv_std],
        "hsv_percentile_10": [float(v) for v in hsv_p10],
        "hsv_percentile_90": [float(v) for v in hsv_p90],

        "hsv_margin": [int(h_margin), int(s_margin), int(v_margin)],
        "hsv_ranges": hsv_ranges,
    }

    print("\nAdded sample:")
    print(f"  Source: {source_name}")
    print(f"  ROI: {sample['roi_xywh']}")
    print(
        "  HSV median: "
        f"H={hsv_median[0]:.1f}, S={hsv_median[1]:.1f}, V={hsv_median[2]:.1f}"
    )
    print(f"  HSV margin: {sample['hsv_margin']}")
    print(f"  Ranges: {sample['hsv_ranges']}")

    return sample


# ============================================================
# Mask tools
# ============================================================

def get_all_hsv_ranges(state):
    ranges = []

    ranges.extend(state.imported_ranges)

    for sample in state.samples:
        ranges.extend(sample.get("hsv_ranges", []))

    return ranges


def remove_small_components(mask, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def create_mask_from_ranges(bgr, hsv_ranges):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for r in hsv_ranges:
        lower = np.array(r["lower"], dtype=np.uint8)
        upper = np.array(r["upper"], dtype=np.uint8)

        this_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, this_mask)

    # Gentle cleanup.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big, iterations=1)

    mask = remove_small_components(mask, min_area=MIN_COMPONENT_AREA)

    return mask


def overlay_mask(bgr, mask):
    out = bgr.copy()

    if mask is None:
        return out

    mask_pixels = mask > 0

    if np.any(mask_pixels):
        alpha = 0.35
        red = np.array([0, 0, 255], dtype=np.float32)

        original = out[mask_pixels].astype(np.float32)
        blended = (1.0 - alpha) * original + alpha * red

        out[mask_pixels] = np.clip(blended, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    return out


# ============================================================
# Captured image tools
# ============================================================

def capture_current_frame(state):
    if state.latest_bgr is None:
        print("No live frame available yet.")
        return

    CALIB_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = CALIB_IMAGE_DIR / f"cable_calib_{timestamp}_{len(state.captured_frames):03d}.png"

    frame = state.latest_bgr.copy()
    cv2.imwrite(str(filename), frame)

    item = {
        "path": str(filename),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bgr": frame,
    }

    state.captured_frames.append(item)
    state.selected_capture_idx = len(state.captured_frames) - 1

    print(f"Captured calibration image: {filename}")
    show_capture_preview(state)


def get_selected_capture(state):
    if len(state.captured_frames) == 0:
        return None

    if state.selected_capture_idx < 0:
        state.selected_capture_idx = 0

    state.selected_capture_idx %= len(state.captured_frames)

    return state.captured_frames[state.selected_capture_idx]


def show_capture_preview(state):
    item = get_selected_capture(state)

    if item is None:
        print("No captured images yet. Press p to capture one.")
        return

    bgr = item["bgr"].copy()
    hsv_ranges = get_all_hsv_ranges(state)

    if len(hsv_ranges) > 0:
        mask = create_mask_from_ranges(bgr, hsv_ranges)
        overlay = overlay_mask(bgr, mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        overlay = bgr
        mask_bgr = np.zeros_like(bgr)

    text1 = (
        f"Captured {state.selected_capture_idx + 1}/{len(state.captured_frames)} | "
        f"{Path(item['path']).name}"
    )
    text2 = "i: sample ROI from this image | n/b: next/previous capture"

    cv2.putText(
        overlay,
        text1,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        text2,
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    combined = np.hstack([overlay, mask_bgr])

    if DISPLAY_SCALE != 1.0:
        combined = cv2.resize(
            combined,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )

    cv2.namedWindow(WINDOW_CAPTURE, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_CAPTURE, combined)


def sample_from_selected_capture(state):
    item = get_selected_capture(state)

    if item is None:
        print("No captured image selected. Press p to capture one first.")
        return

    source_name = f"captured_image_{state.selected_capture_idx + 1}"
    source_path = item["path"]

    sample = sample_roi_color(
        item["bgr"].copy(),
        source_name=source_name,
        source_path=source_path,
    )

    if sample is not None:
        state.samples.append(sample)
        print(f"Total samples: {len(state.samples)}")
        show_capture_preview(state)


# ============================================================
# Profile save/load
# ============================================================

def build_profile(state):
    hsv_ranges = get_all_hsv_ranges(state)

    captured_images = []

    for item in state.captured_frames:
        captured_images.append({
            "path": item["path"],
            "created_at": item["created_at"],
        })

    profile = {
        "profile_type": "multi_image_multi_sample_hsv",
        "version": 3,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),

        "camera_resolution": CAMERA_RESOLUTION_NAME,
        "camera_fps": CAMERA_FPS,

        "camera_settings": {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "hue": HUE,
            "saturation": SATURATION,
            "sharpness": SHARPNESS,
            "gamma": GAMMA,
            "auto_white_balance": AUTO_WHITE_BALANCE,
            "auto_exposure": AUTO_EXPOSURE,
        },

        "captured_images": captured_images,

        "num_samples": len(state.samples),
        "samples": state.samples,

        # Compatibility field:
        # Your detector already reads this.
        "hsv_ranges": hsv_ranges,

        "imported_ranges": state.imported_ranges,

        "notes": [
            "This is a multi-image, multi-sample HSV cable profile.",
            "Each selected cable ROI contributes one or more HSV ranges.",
            "The detector can OR all hsv_ranges together.",
            "OpenCV HSV uses H in [0, 179], S/V in [0, 255].",
            "ZED image conversion should use BGRA2BGR.",
        ],
    }

    all_hsv_medians = []

    for sample in state.samples:
        all_hsv_medians.append(sample["hsv_median"])

    if len(all_hsv_medians) > 0:
        arr = np.array(all_hsv_medians, dtype=np.float32)

        profile["summary"] = {
            "hsv_median_mean": [float(v) for v in np.mean(arr, axis=0)],
            "hsv_median_min": [float(v) for v in np.min(arr, axis=0)],
            "hsv_median_max": [float(v) for v in np.max(arr, axis=0)],
        }

    return profile


def save_profile(state, path=PROFILE_PATH):
    hsv_ranges = get_all_hsv_ranges(state)

    if len(hsv_ranges) == 0:
        print("No samples/ranges to save yet.")
        print("Press p to capture images, then i to sample cable ROIs.")
        return

    profile = build_profile(state)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"\nSaved profile to: {path}")
    print(f"  Captured images: {len(state.captured_frames)}")
    print(f"  Samples: {len(state.samples)}")
    print(f"  Total HSV ranges: {len(profile['hsv_ranges'])}")


def load_existing_profile(state, path=PROFILE_PATH):
    path = Path(path)

    if not path.exists():
        print("No existing cable_color_profile.json found. Starting fresh.")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        samples = profile.get("samples", [])

        if isinstance(samples, list) and len(samples) > 0:
            state.samples = samples
            state.imported_ranges = profile.get("imported_ranges", [])
            print(f"Loaded existing profile: {len(state.samples)} samples.")
        else:
            old_ranges = profile.get("hsv_ranges", [])
            state.imported_ranges = old_ranges
            print(f"Loaded old profile as imported ranges: {len(old_ranges)} ranges.")

    except Exception as e:
        print(f"Could not load existing profile: {e}")


# ============================================================
# Display
# ============================================================

def make_live_display(bgr, state):
    hsv_ranges = get_all_hsv_ranges(state)

    if len(hsv_ranges) > 0:
        mask = create_mask_from_ranges(bgr, hsv_ranges)
        overlay = overlay_mask(bgr, mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        overlay = bgr.copy()
        mask_bgr = np.zeros_like(bgr)

    status_lines = [
        f"Samples: {len(state.samples)} | Captures: {len(state.captured_frames)} | Total ranges: {len(hsv_ranges)}",
        "p: capture photo | i: sample selected photo | a/c: sample live frame",
        "n/b: next/prev photo | u: undo sample | x: clear | s: save | q: quit",
    ]

    y0 = 45

    for i, line in enumerate(status_lines):
        cv2.putText(
            overlay,
            line,
            (30, y0 + i * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    combined = np.hstack([overlay, mask_bgr])

    if DISPLAY_SCALE != 1.0:
        combined = cv2.resize(
            combined,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )

    return combined


# ============================================================
# Main
# ============================================================

def main():
    state = AppState()
    load_existing_profile(state, PROFILE_PATH)

    zed = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = CAMERA_RESOLUTION
    init.camera_fps = CAMERA_FPS

    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.3
    init.depth_maximum_distance = 3.0
    init.depth_stabilization = 30

    status = zed.open(init)
    print("Open status:", status)

    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("Could not open ZED camera.")

    apply_camera_settings(zed)

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 60
    runtime.texture_confidence_threshold = 70
    runtime.remove_saturated_areas = False

    left_image = sl.Mat()

    cv2.namedWindow(WINDOW_LIVE, cv2.WINDOW_NORMAL)

    print("\nControls:")
    print("  p       -> capture current live frame as calibration image")
    print("  i       -> sample cable ROI from selected captured image")
    print("  a or c  -> sample cable ROI directly from current live frame")
    print("  n       -> next captured image")
    print("  b       -> previous captured image")
    print("  u       -> undo last sample")
    print("  x       -> clear all samples and imported ranges")
    print("  s       -> save cable_color_profile.json")
    print("  q       -> quit")

    print("\nRecommended workflow:")
    print("  1. Move cable near camera, press p")
    print("  2. Move cable farther, press p")
    print("  3. Put cable in shadow, press p")
    print("  4. Put cable in bright area, press p")
    print("  5. Use n/b to browse photos")
    print("  6. Press i on each photo to select cable ROI")
    print("  7. Press s to save\n")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            left_raw = left_image.get_data()

            # Important:
            # ZED image data should be treated as BGRA for OpenCV.
            bgr = cv2.cvtColor(left_raw, cv2.COLOR_BGRA2BGR)

            state.latest_bgr = bgr.copy()
            state.frame_count += 1

            display = make_live_display(bgr, state)
            cv2.imshow(WINDOW_LIVE, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("p"):
            capture_current_frame(state)

        elif key == ord("i"):
            sample_from_selected_capture(state)

        elif key == ord("a") or key == ord("c"):
            if state.latest_bgr is None:
                print("No live frame available yet.")
                continue

            sample = sample_roi_color(
                state.latest_bgr.copy(),
                source_name="live_frame",
                source_path=None,
            )

            if sample is not None:
                state.samples.append(sample)
                print(f"Total samples: {len(state.samples)}")

        elif key == ord("n"):
            if len(state.captured_frames) == 0:
                print("No captured images yet.")
            else:
                state.selected_capture_idx = (state.selected_capture_idx + 1) % len(state.captured_frames)
                show_capture_preview(state)

        elif key == ord("b"):
            if len(state.captured_frames) == 0:
                print("No captured images yet.")
            else:
                state.selected_capture_idx = (state.selected_capture_idx - 1) % len(state.captured_frames)
                show_capture_preview(state)

        elif key == ord("u"):
            if len(state.samples) > 0:
                removed = state.samples.pop()
                print(f"Removed last sample: {removed.get('source_name', 'unknown')} ROI {removed['roi_xywh']}")
                show_capture_preview(state)
            else:
                print("No sample to undo.")

        elif key == ord("x"):
            state.samples = []
            state.imported_ranges = []
            print("Cleared samples and imported ranges.")
            show_capture_preview(state)

        elif key == ord("s"):
            save_profile(state, PROFILE_PATH)

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
