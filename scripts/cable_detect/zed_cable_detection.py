import cv2
import json
import time
from pathlib import Path

import numpy as np
import pyzed.sl as sl
import torch


# ============================================================
# Files
# ============================================================

PROFILE_PATH = "cable_color_profile.json"


# ============================================================
# ZED camera parameters
# ============================================================

# Current code only uses RGB. Turn depth on later for 3D cable reconstruction.
USE_ZED_DEPTH = False

CAMERA_RESOLUTION = sl.RESOLUTION.HD1080
CAMERA_FPS = 30

BRIGHTNESS = 4
CONTRAST = 4
HUE = 0
SATURATION = 4
SHARPNESS = 4
GAMMA = 5

AUTO_WHITE_BALANCE = True
AUTO_EXPOSURE = True


# ============================================================
# Performance parameters
# ============================================================

# 0.5 is much faster. The displayed window is the processed-resolution view.
PROCESS_SCALE = 0.5

# Do not optimize the chain every frame.
CHAIN_UPDATE_INTERVAL = 2

# If the chain is already tracking, use fewer optimizer steps.
TORCH_STEPS_COLD = 90
TORCH_STEPS_WARM = 30

TORCH_LR_COLD = 0.055
TORCH_LR_WARM = 0.025


# ============================================================
# Detection parameters
# ============================================================

SHOW_FILTERED_MASK = True
SHOW_DETECTED_CENTERLINE = True
SHOW_CHAIN = True
SHOW_CHAIN_NODES = True

MIN_COMPONENT_AREA = 35
KEEP_LARGEST_N_COMPONENTS = 10

MORPH_OPEN_SIZE = 3
MORPH_CLOSE_SIZE = 5

X_STEP = 4
MIN_POINTS_PER_COLUMN_BIN = 2
MIN_CENTERLINE_POINTS = 6

# This helps reject vertical blobs / side branches touching the cable.
USE_MAIN_PATH_FILTER = True
MAX_COLUMN_RUN_HEIGHT = 45
MAIN_PATH_DY_COST = 1.0
MAIN_PATH_SUPPORT_REWARD = 0.08
MAIN_PATH_MAX_DY_JUMP = 45

# Merge tiny breaks, but keep real object occlusion gaps.
MERGE_SMALL_GAP_X_PX = 14
MERGE_SMALL_GAP_Y_PX = 28

# If neighboring visible segments are separated by more than this,
# the gap is treated as hidden / occluded cable.
OCCLUSION_GAP_X_PX = 22


# ============================================================
# Endpoint locking
# ============================================================

USE_LOCKED_ENDPOINTS = True
LOCKED_ENDPOINTS = None


# ============================================================
# Catenary-guided chain model parameters
# ============================================================

CHAIN_ENABLED = True

NUM_CHAIN_NODES = 75

MIN_CHAIN_OBS_POINTS = 20
MIN_CHAIN_SPAN_PX = 45
MAX_CHAIN_DATA_POINTS = 650

# More slack = more sag / longer cable.
CABLE_LENGTH_MULTIPLIER = 1.08

# Catenary-like hidden gap initialization.
CATENARY_SHAPE_K = 1.7
CATENARY_SAG_FRACTION = 0.16
MIN_HIDDEN_SAG_PX = 4.0
MAX_HIDDEN_SAG_PX = 120.0

# Data loss.
DATA_WEIGHT = 2.2
VISIBLE_CHAIN_COVERAGE_WEIGHT = 1.0

DATA_X_SCALE = 0.35
DATA_Y_SCALE = 1.00
DATA_ROBUST_SCALE_PX = 10.0

# Physics / chain constraints.
SEGMENT_LENGTH_WEIGHT = 35.0
TOTAL_LENGTH_WEIGHT = 6.0
SMOOTHNESS_WEIGHT = 0.075

# Gravity: image y increases downward.
GRAVITY_WEIGHT = 0.010
HIDDEN_GRAVITY_WEIGHT = 0.045

# Prevent chain folding backward for current left-to-right setup.
ORDER_WEIGHT = 120.0
MIN_NODE_X_SPACING = 0.25

# Temporal consistency.
TEMPORAL_WEIGHT = 0.12
USE_TEMPORAL_CHAIN_SMOOTHING = True
TEMPORAL_ALPHA = 0.60
PREVIOUS_CHAIN_NODES = None

# Sampling along chain for coverage loss.
CHAIN_COVERAGE_SAMPLES_PER_EDGE = 2


# ============================================================
# Drawing parameters
# ============================================================

MASK_OVERLAY_COLOR = (0, 0, 255)
CENTERLINE_COLOR = (0, 255, 0)

CHAIN_VISIBLE_COLOR = (255, 255, 255)
CHAIN_HIDDEN_COLOR = (255, 0, 0)
CHAIN_NODE_COLOR = (0, 255, 255)

LOCKED_ENDPOINT_COLOR = (0, 0, 255)

CHAIN_LINE_THICKNESS = 4
CENTERLINE_THICKNESS = 2
MASK_ALPHA = 0.22


# ============================================================
# Camera settings
# ============================================================

def apply_camera_settings(zed):
    def safe_set(setting, value, name):
        err = zed.set_camera_settings(setting, value)
        print(f"{name}: {value} -> {err}")

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
# Profile / HSV mask
# ============================================================

def load_profile(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put this script in the same folder as cable_color_profile.json."
        )

    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    hsv_ranges = profile.get("hsv_ranges", [])

    if len(hsv_ranges) == 0:
        raise ValueError(
            "Profile has no hsv_ranges. Re-run cable_color_collect.py and save a profile."
        )

    print(f"Loaded profile from: {path}")
    print(f"Profile type: {profile.get('profile_type', 'old_single_profile')}")
    print(f"Version: {profile.get('version', 'unknown')}")
    print(f"Samples: {profile.get('num_samples', 'unknown')}")
    print(f"Captured images: {len(profile.get('captured_images', []))}")
    print(f"HSV ranges: {len(hsv_ranges)}")

    for i, r in enumerate(hsv_ranges):
        print(f"  range {i}: lower={r['lower']} upper={r['upper']}")

    return profile


def create_raw_mask_from_profile(bgr, profile):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for r in profile.get("hsv_ranges", []):
        lower = np.array(r["lower"], dtype=np.uint8)
        upper = np.array(r["upper"], dtype=np.uint8)

        this_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, this_mask)

    return mask


def clean_raw_mask(raw_mask):
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE),
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE),
    )

    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return mask


def keep_candidate_components(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(mask), []

    components = []

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side

        components.append({
            "label": label,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "aspect": float(aspect),
        })

    components.sort(key=lambda item: item["area"], reverse=True)
    components = components[:KEEP_LARGEST_N_COMPONENTS]

    filtered = np.zeros_like(mask)

    for comp in components:
        filtered[labels == comp["label"]] = 255

    return filtered, components


def create_masks_from_profile(bgr, profile):
    raw_mask = create_raw_mask_from_profile(bgr, profile)
    cleaned_mask = clean_raw_mask(raw_mask)
    filtered_mask, components = keep_candidate_components(cleaned_mask)

    return raw_mask, cleaned_mask, filtered_mask, components


# ============================================================
# Centerline extraction
# ============================================================

def smooth_1d(values, window=7):
    if len(values) < window:
        return values

    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2

    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    return smoothed


def get_column_candidates(mask, x0, x1):
    column = mask[:, x0:x1]
    ys, _ = np.where(column > 0)

    if len(ys) < MIN_POINTS_PER_COLUMN_BIN:
        return []

    ys_sorted = np.sort(ys)

    runs = []
    start = ys_sorted[0]
    prev = ys_sorted[0]

    for y in ys_sorted[1:]:
        if y <= prev + 1:
            prev = y
        else:
            runs.append((start, prev))
            start = y
            prev = y

    runs.append((start, prev))

    candidates = []

    for y_start, y_end in runs:
        height = int(y_end - y_start + 1)

        if USE_MAIN_PATH_FILTER and height > MAX_COLUMN_RUN_HEIGHT:
            continue

        support = int(np.count_nonzero((ys >= y_start) & (ys <= y_end)))

        if support < MIN_POINTS_PER_COLUMN_BIN:
            continue

        x_center = int((x0 + x1) / 2)
        y_center = int((y_start + y_end) / 2)

        candidates.append({
            "x": x_center,
            "y": y_center,
            "support": support,
            "height": height,
        })

    # Fallback: if everything was rejected because the run was tall,
    # still use median so the detector does not completely die.
    if len(candidates) == 0 and len(ys) >= MIN_POINTS_PER_COLUMN_BIN:
        x_center = int((x0 + x1) / 2)
        y_center = int(np.median(ys))

        candidates.append({
            "x": x_center,
            "y": y_center,
            "support": int(len(ys)),
            "height": int(np.max(ys) - np.min(ys) + 1),
        })

    return candidates


def select_smooth_path_from_candidate_columns(candidate_columns):
    if len(candidate_columns) == 0:
        return []

    dp = []
    parent = []

    first_col = candidate_columns[0]
    dp_first = []

    for cand in first_col:
        support_cost = -MAIN_PATH_SUPPORT_REWARD * cand["support"]
        dp_first.append(support_cost)

    dp.append(dp_first)
    parent.append([-1] * len(first_col))

    for i in range(1, len(candidate_columns)):
        prev_col = candidate_columns[i - 1]
        curr_col = candidate_columns[i]

        dp_curr = []
        parent_curr = []

        for curr in curr_col:
            best_cost = None
            best_parent = -1

            for k, prev in enumerate(prev_col):
                dy = abs(curr["y"] - prev["y"])

                transition_cost = MAIN_PATH_DY_COST * dy
                support_cost = -MAIN_PATH_SUPPORT_REWARD * curr["support"]

                total_cost = dp[i - 1][k] + transition_cost + support_cost

                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_parent = k

            dp_curr.append(best_cost)
            parent_curr.append(best_parent)

        dp.append(dp_curr)
        parent.append(parent_curr)

    best_last = int(np.argmin(dp[-1]))
    selected_indices = [best_last]

    for i in range(len(candidate_columns) - 1, 0, -1):
        best_last = parent[i][best_last]
        selected_indices.append(best_last)

    selected_indices.reverse()

    selected = []

    for i, idx in enumerate(selected_indices):
        if idx >= 0:
            selected.append(candidate_columns[i][idx])

    return selected


def extract_centerline_by_columns(mask, x_step=4):
    h, w = mask.shape

    columns = []

    for x0 in range(0, w, x_step):
        x1 = min(x0 + x_step, w)
        candidates = get_column_candidates(mask, x0, x1)
        columns.append(candidates)

    groups = []
    current = []

    for i, candidates in enumerate(columns):
        if len(candidates) > 0:
            current.append(i)
        else:
            if len(current) >= MIN_CENTERLINE_POINTS:
                groups.append(current)
            current = []

    if len(current) >= MIN_CENTERLINE_POINTS:
        groups.append(current)

    if len(groups) == 0:
        return np.empty((0, 2), dtype=np.float64)

    all_points = []

    for group in groups:
        selected = select_smooth_path_from_candidate_columns(
            [columns[i] for i in group]
        )

        if len(selected) >= MIN_CENTERLINE_POINTS:
            for p in selected:
                all_points.append([p["x"], p["y"]])

    if len(all_points) < MIN_CENTERLINE_POINTS:
        return np.empty((0, 2), dtype=np.float64)

    points = np.array(all_points, dtype=np.float64)

    if len(points) >= 7:
        points[:, 1] = smooth_1d(points[:, 1].astype(np.float32), window=7)

    return points


def split_centerline_into_segments(points):
    if len(points) == 0:
        return []

    segments = []
    current = [points[0]]

    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]

        dx = abs(float(curr[0] - prev[0]))
        dy = abs(float(curr[1] - prev[1]))

        bad_jump = (
            dx > OCCLUSION_GAP_X_PX
            or dy > MAIN_PATH_MAX_DY_JUMP
        )

        if bad_jump:
            if len(current) >= MIN_CENTERLINE_POINTS:
                segments.append(np.array(current, dtype=np.float64))
            current = [curr]
        else:
            current.append(curr)

    if len(current) >= MIN_CENTERLINE_POINTS:
        segments.append(np.array(current, dtype=np.float64))

    return segments


def merge_close_segments(segments):
    if len(segments) <= 1:
        return segments

    segments = sorted(segments, key=lambda s: float(s[0, 0]))

    merged = [segments[0]]

    for seg in segments[1:]:
        last = merged[-1]

        last_end = last[-1]
        this_start = seg[0]

        dx = abs(float(this_start[0] - last_end[0]))
        dy = abs(float(this_start[1] - last_end[1]))

        if dx <= MERGE_SMALL_GAP_X_PX and dy <= MERGE_SMALL_GAP_Y_PX:
            merged[-1] = np.vstack([last, seg])
        else:
            merged.append(seg)

    return merged


def extract_visible_centerline_segments(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    all_segments = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        component_mask = np.zeros_like(mask)
        component_mask[labels == label] = 255

        centerline = extract_centerline_by_columns(
            component_mask,
            x_step=X_STEP,
        )

        pieces = split_centerline_into_segments(centerline)
        all_segments.extend(pieces)

    all_segments.sort(key=lambda seg: float(seg[0, 0]))
    all_segments = merge_close_segments(all_segments)

    return all_segments


# ============================================================
# Chain geometry helpers
# ============================================================

def orient_segment_left_to_right(seg):
    if len(seg) < 2:
        return seg

    if seg[0, 0] > seg[-1, 0]:
        return seg[::-1].copy()

    return seg


def polyline_length(points):
    if len(points) < 2:
        return 0.0

    diffs = np.diff(points.astype(np.float64), axis=0)
    lengths = np.linalg.norm(diffs, axis=1)

    return float(np.sum(lengths))


def point_to_segments_distance_numpy(points, nodes):
    if len(points) == 0 or len(nodes) < 2:
        return np.array([], dtype=np.float64)

    points = points.astype(np.float64)
    a = nodes[:-1].astype(np.float64)
    b = nodes[1:].astype(np.float64)

    p = points[:, None, :]
    aa = a[None, :, :]
    bb = b[None, :, :]

    ab = bb - aa
    ap = p - aa

    denom = np.sum(ab * ab, axis=2) + 1e-8
    t = np.sum(ap * ab, axis=2) / denom
    t = np.clip(t, 0.0, 1.0)

    closest = aa + t[:, :, None] * ab
    dist = np.linalg.norm(p - closest, axis=2)

    return np.min(dist, axis=1)


def estimate_endpoints_from_segments(segments):
    """
    Estimate cable endpoints from currently visible cable segments.

    Usage:
        hold cable steady -> press L -> endpoints are locked.
    """

    if len(segments) == 0:
        return None

    sorted_segments = [
        orient_segment_left_to_right(seg)
        for seg in sorted(segments, key=lambda s: float(np.min(s[:, 0])))
    ]

    if len(sorted_segments[0]) == 0 or len(sorted_segments[-1]) == 0:
        return None

    left_endpoint = sorted_segments[0][0].copy()
    right_endpoint = sorted_segments[-1][-1].copy()

    if left_endpoint[0] > right_endpoint[0]:
        left_endpoint, right_endpoint = right_endpoint, left_endpoint

    return [
        left_endpoint.astype(np.float64),
        right_endpoint.astype(np.float64),
    ]


# ============================================================
# Catenary initialization
# ============================================================

def catenary_sag_shape(t):
    """
    Returns a 0-at-ends, 1-at-center sag profile.
    This is catenary-like and numerically stable for initialization.
    """

    k = CATENARY_SHAPE_K
    denom = np.cosh(k) - 1.0

    if abs(denom) < 1e-6:
        return np.sin(np.pi * t)

    shape = 1.0 - ((np.cosh(k * (2.0 * t - 1.0)) - 1.0) / denom)
    return np.clip(shape, 0.0, 1.0)


def estimate_gap_sag(gap_width):
    slack_ratio = max(CABLE_LENGTH_MULTIPLIER - 1.0, 0.015)

    sag = (
        CATENARY_SAG_FRACTION
        * gap_width
        * slack_ratio
        / 0.08
    )

    sag = float(np.clip(sag, MIN_HIDDEN_SAG_PX, MAX_HIDDEN_SAG_PX))

    return sag


def get_occlusion_gaps_from_segments(segments):
    if len(segments) < 2:
        return []

    sorted_segments = [
        orient_segment_left_to_right(seg)
        for seg in sorted(segments, key=lambda s: float(np.min(s[:, 0])))
    ]

    gaps = []

    for i in range(len(sorted_segments) - 1):
        left = sorted_segments[i]
        right = sorted_segments[i + 1]

        left_end = left[-1]
        right_start = right[0]

        x0 = float(left_end[0])
        x1 = float(right_start[0])

        if x1 <= x0:
            continue

        if x1 - x0 >= OCCLUSION_GAP_X_PX:
            gaps.append({
                "x0": x0,
                "x1": x1,
                "p0": left_end.copy(),
                "p1": right_start.copy(),
                "width": float(x1 - x0),
            })

    return gaps


def is_x_inside_any_gap(x, gaps):
    for gap in gaps:
        if gap["x0"] <= x <= gap["x1"]:
            return True

    return False


def make_initial_chain_with_catenary_gaps(segments, locked_endpoints=None):
    """
    Create one continuous initial cable chain.

    Visible areas follow detected centerline.
    Hidden gaps are initialized using a catenary-like sag curve.
    Locked endpoints, if available, define fixed chain boundaries.
    """

    sorted_segments = [
        orient_segment_left_to_right(seg)
        for seg in sorted(segments, key=lambda s: float(np.min(s[:, 0])))
    ]

    if len(sorted_segments) == 0:
        return np.empty((0, 2), dtype=np.float64), []

    obs_points = np.vstack(sorted_segments).astype(np.float64)
    obs_order = np.argsort(obs_points[:, 0])
    obs_points = obs_points[obs_order]

    if locked_endpoints is not None and len(locked_endpoints) == 2:
        p_start = locked_endpoints[0].copy()
        p_end = locked_endpoints[1].copy()

        if p_start[0] > p_end[0]:
            p_start, p_end = p_end, p_start
    else:
        p_start = sorted_segments[0][0].copy()
        p_end = sorted_segments[-1][-1].copy()

    x_min = float(p_start[0])
    x_max = float(p_end[0])

    if x_max <= x_min:
        return np.empty((0, 2), dtype=np.float64), []

    # Include endpoints as weak interpolation anchors.
    interp_points = np.vstack([
        p_start.reshape(1, 2),
        obs_points,
        p_end.reshape(1, 2),
    ])

    interp_order = np.argsort(interp_points[:, 0])
    interp_points = interp_points[interp_order]

    x_nodes = np.linspace(x_min, x_max, NUM_CHAIN_NODES)

    x_obs = interp_points[:, 0]
    y_obs = interp_points[:, 1]

    x_unique, unique_idx = np.unique(x_obs, return_index=True)
    y_unique = y_obs[unique_idx]

    if len(x_unique) >= 2:
        y_nodes = np.interp(x_nodes, x_unique, y_unique)
    else:
        y_nodes = np.full_like(x_nodes, float(p_start[1]))

    init_nodes = np.column_stack([x_nodes, y_nodes]).astype(np.float64)

    gaps = get_occlusion_gaps_from_segments(sorted_segments)

    # Replace hidden-gap initialization with catenary-like curve.
    for gap in gaps:
        x0 = gap["x0"]
        x1 = gap["x1"]
        p0 = gap["p0"]
        p1 = gap["p1"]

        idx = np.where((x_nodes >= x0) & (x_nodes <= x1))[0]

        if len(idx) == 0:
            continue

        width = max(float(x1 - x0), 1.0)
        sag = estimate_gap_sag(width)

        for node_idx in idx:
            x = x_nodes[node_idx]
            t = (x - x0) / width
            t = float(np.clip(t, 0.0, 1.0))

            y_line = (1.0 - t) * p0[1] + t * p1[1]
            sag_profile = catenary_sag_shape(t)

            # Image y increases downward, so positive sag moves cable down.
            y = y_line + sag * sag_profile

            init_nodes[node_idx, 1] = y

    init_nodes[0] = p_start
    init_nodes[-1] = p_end

    return init_nodes, gaps


def estimate_total_cable_length(segments, init_nodes):
    visible_len = 0.0

    sorted_segments = [
        orient_segment_left_to_right(seg)
        for seg in sorted(segments, key=lambda s: float(np.min(s[:, 0])))
    ]

    for seg in sorted_segments:
        visible_len += polyline_length(seg)

    gap_len = 0.0

    for i in range(len(sorted_segments) - 1):
        left = sorted_segments[i]
        right = sorted_segments[i + 1]

        left_end = left[-1]
        right_start = right[0]

        dx = float(right_start[0] - left_end[0])

        if dx >= OCCLUSION_GAP_X_PX:
            gap_len += float(np.linalg.norm(right_start - left_end))

    endpoint_distance = float(np.linalg.norm(init_nodes[-1] - init_nodes[0]))

    base_len = max(visible_len + gap_len, endpoint_distance)
    total_len = base_len * CABLE_LENGTH_MULTIPLIER

    return float(total_len), float(endpoint_distance), float(visible_len), float(gap_len)


def build_chain_problem(segments, image_shape, previous_nodes=None, locked_endpoints=None):
    if len(segments) == 0:
        return None, {
            "success": False,
            "reason": "no_segments",
            "message": "No visible cable segments",
        }

    sorted_segments = [
        orient_segment_left_to_right(seg)
        for seg in sorted(segments, key=lambda s: float(np.min(s[:, 0])))
    ]

    obs_points = np.vstack(sorted_segments).astype(np.float64)

    if len(obs_points) < MIN_CHAIN_OBS_POINTS:
        return None, {
            "success": False,
            "reason": "not_enough_points",
            "message": f"Need at least {MIN_CHAIN_OBS_POINTS} centerline points",
        }

    if len(obs_points) > MAX_CHAIN_DATA_POINTS:
        idx = np.linspace(0, len(obs_points) - 1, MAX_CHAIN_DATA_POINTS).astype(int)
        data_points = obs_points[idx]
    else:
        data_points = obs_points

    init_nodes, gaps = make_initial_chain_with_catenary_gaps(
        sorted_segments,
        locked_endpoints=locked_endpoints,
    )

    if len(init_nodes) != NUM_CHAIN_NODES:
        return None, {
            "success": False,
            "reason": "bad_initialization",
            "message": "Could not initialize catenary chain",
        }

    x_all = [float(np.min(obs_points[:, 0])), float(np.max(obs_points[:, 0]))]

    if locked_endpoints is not None and len(locked_endpoints) == 2:
        x_all.extend([
            float(locked_endpoints[0][0]),
            float(locked_endpoints[1][0]),
        ])

    x_min = float(min(x_all))
    x_max = float(max(x_all))
    span = x_max - x_min

    if span < MIN_CHAIN_SPAN_PX:
        return None, {
            "success": False,
            "reason": "span_too_small",
            "message": f"Visible span too small: {span:.1f}px",
        }

    if (
        USE_TEMPORAL_CHAIN_SMOOTHING
        and previous_nodes is not None
        and previous_nodes.shape == init_nodes.shape
    ):
        prev_span = previous_nodes[-1, 0] - previous_nodes[0, 0]
        compatible = abs(prev_span - span) < 0.35 * max(span, 1.0)

        if compatible:
            p0 = init_nodes[0].copy()
            pN = init_nodes[-1].copy()

            init_nodes = (
                TEMPORAL_ALPHA * previous_nodes
                + (1.0 - TEMPORAL_ALPHA) * init_nodes
            )

            init_nodes[0] = p0
            init_nodes[-1] = pN

    total_len, endpoint_distance, visible_len, gap_len = estimate_total_cable_length(
        sorted_segments,
        init_nodes,
    )

    target_seg_len = total_len / max(1, NUM_CHAIN_NODES - 1)

    hidden_node_mask = np.zeros(NUM_CHAIN_NODES, dtype=bool)

    for i, p in enumerate(init_nodes):
        if is_x_inside_any_gap(float(p[0]), gaps):
            hidden_node_mask[i] = True

    hidden_node_mask[0] = False
    hidden_node_mask[-1] = False

    hidden_edge_mask = np.zeros(NUM_CHAIN_NODES - 1, dtype=bool)

    for i in range(NUM_CHAIN_NODES - 1):
        mid_x = 0.5 * (init_nodes[i, 0] + init_nodes[i + 1, 0])

        if is_x_inside_any_gap(float(mid_x), gaps):
            hidden_edge_mask[i] = True

    visible_edge_mask = np.logical_not(hidden_edge_mask)

    h, w = image_shape[:2]

    margin_x = 0.25 * max(span, 1.0)
    x_lower = max(0.0, x_min - margin_x)
    x_upper = min(float(w - 1), x_max + margin_x)

    problem = {
        "obs_points": obs_points,
        "data_points": data_points,
        "init_nodes": init_nodes,
        "p0": init_nodes[0].copy(),
        "pN": init_nodes[-1].copy(),
        "gaps": gaps,
        "hidden_node_mask": hidden_node_mask,
        "hidden_edge_mask": hidden_edge_mask,
        "visible_edge_mask": visible_edge_mask,
        "target_seg_len": float(target_seg_len),
        "total_len": float(total_len),
        "endpoint_distance": float(endpoint_distance),
        "visible_len": float(visible_len),
        "gap_len": float(gap_len),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "span": float(span),
        "bounds": {
            "x_lower": float(x_lower),
            "x_upper": float(x_upper),
            "y_lower": 0.0,
            "y_upper": float(h - 1),
        },
    }

    return problem, None


# ============================================================
# GPU optimization
# ============================================================

def robust_l1_like(distances, scale):
    return scale * scale * (torch.sqrt(1.0 + (distances / scale) ** 2) - 1.0)


def point_to_segments_distance_torch(points, segment_a, segment_b):
    """
    points:    [M, 2]
    segment_a: [K, 2]
    segment_b: [K, 2]
    returns: distances [M, K]
    """

    p = points[:, None, :]
    a = segment_a[None, :, :]
    b = segment_b[None, :, :]

    ab = b - a
    ap = p - a

    denom = torch.sum(ab * ab, dim=2) + 1e-8
    t = torch.sum(ap * ab, dim=2) / denom
    t = torch.clamp(t, 0.0, 1.0)

    closest = a + t[:, :, None] * ab
    dist = torch.linalg.norm(p - closest, dim=2)

    return dist


def sample_visible_chain_points_torch(nodes, visible_edge_mask):
    edge_a = nodes[:-1]
    edge_b = nodes[1:]

    edge_a = edge_a[visible_edge_mask]
    edge_b = edge_b[visible_edge_mask]

    if edge_a.shape[0] == 0:
        edge_a = nodes[:-1]
        edge_b = nodes[1:]

    t_values = torch.linspace(
        0.0,
        1.0,
        CHAIN_COVERAGE_SAMPLES_PER_EDGE,
        dtype=torch.float32,
        device=nodes.device,
    )

    samples = (
        edge_a[:, None, :] * (1.0 - t_values[None, :, None])
        + edge_b[:, None, :] * t_values[None, :, None]
    )

    return samples.reshape(-1, 2)


def optimize_chain_torch_cuda(problem, previous_nodes=None):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script requires CUDA-enabled PyTorch."
        )

    device = torch.device("cuda")

    init_nodes = torch.tensor(problem["init_nodes"], dtype=torch.float32, device=device)
    data_points = torch.tensor(problem["data_points"], dtype=torch.float32, device=device)

    p0 = torch.tensor(problem["p0"], dtype=torch.float32, device=device)
    pN = torch.tensor(problem["pN"], dtype=torch.float32, device=device)

    target_seg_len = torch.tensor(problem["target_seg_len"], dtype=torch.float32, device=device)
    target_total_len = torch.tensor(problem["total_len"], dtype=torch.float32, device=device)

    hidden_node_mask = torch.tensor(problem["hidden_node_mask"], dtype=torch.bool, device=device)
    visible_edge_mask = torch.tensor(problem["visible_edge_mask"], dtype=torch.bool, device=device)

    if previous_nodes is not None and previous_nodes.shape == problem["init_nodes"].shape:
        previous_nodes_torch = torch.tensor(
            previous_nodes,
            dtype=torch.float32,
            device=device,
        )
        warm_start = True
    else:
        previous_nodes_torch = None
        warm_start = False

    steps = TORCH_STEPS_WARM if warm_start else TORCH_STEPS_COLD
    lr = TORCH_LR_WARM if warm_start else TORCH_LR_COLD

    x_lower = problem["bounds"]["x_lower"]
    x_upper = problem["bounds"]["x_upper"]
    y_lower = problem["bounds"]["y_lower"]
    y_upper = problem["bounds"]["y_upper"]

    internal = torch.nn.Parameter(init_nodes[1:-1].clone().detach())
    optimizer = torch.optim.Adam([internal], lr=lr)

    data_scale = torch.tensor(
        [DATA_X_SCALE, DATA_Y_SCALE],
        dtype=torch.float32,
        device=device,
    )

    scaled_data = data_points * data_scale

    last_loss = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        nodes = torch.cat([
            p0.view(1, 2),
            internal,
            pN.view(1, 2),
        ], dim=0)

        segment_a = nodes[:-1]
        segment_b = nodes[1:]

        scaled_a = segment_a * data_scale
        scaled_b = segment_b * data_scale

        # ----------------------------------------------------
        # Visible observation loss:
        # Detected centerline points should be near the chain.
        # Hidden parts have no data, so they are governed by physics.
        # ----------------------------------------------------
        distances = point_to_segments_distance_torch(
            scaled_data,
            scaled_a,
            scaled_b,
        )

        min_dist_to_chain = torch.min(distances, dim=1).values

        data_loss = torch.mean(
            robust_l1_like(min_dist_to_chain, DATA_ROBUST_SCALE_PX)
        )

        # ----------------------------------------------------
        # Visible chain coverage:
        # Visible chain sections should stay near observations.
        # Hidden gap sections are excluded.
        # ----------------------------------------------------
        visible_samples = sample_visible_chain_points_torch(
            nodes,
            visible_edge_mask,
        )

        scaled_visible_samples = visible_samples * data_scale

        coverage_dist = torch.cdist(
            scaled_visible_samples,
            scaled_data,
        )

        min_visible_to_data = torch.min(coverage_dist, dim=1).values

        coverage_loss = torch.mean(
            robust_l1_like(min_visible_to_data, DATA_ROBUST_SCALE_PX)
        )

        # ----------------------------------------------------
        # Chain physics.
        # ----------------------------------------------------
        diffs = nodes[1:] - nodes[:-1]
        seg_lengths = torch.linalg.norm(diffs, dim=1)

        segment_length_loss = torch.mean((seg_lengths - target_seg_len) ** 2)

        total_len = torch.sum(seg_lengths)
        total_length_loss = ((total_len - target_total_len) / target_total_len) ** 2

        second = nodes[:-2] - 2.0 * nodes[1:-1] + nodes[2:]
        smoothness_loss = torch.mean(torch.sum(second * second, dim=1))

        internal_nodes = nodes[1:-1]

        # Image y increases downward, so negative y energy encourages sagging.
        gravity_energy = -torch.mean(internal_nodes[:, 1])

        if torch.any(hidden_node_mask):
            hidden_nodes = nodes[hidden_node_mask]
            hidden_gravity_energy = -torch.mean(hidden_nodes[:, 1])
        else:
            hidden_gravity_energy = torch.tensor(0.0, dtype=torch.float32, device=device)

        dx = nodes[1:, 0] - nodes[:-1, 0]
        order_loss = torch.mean(torch.relu(MIN_NODE_X_SPACING - dx) ** 2)

        if previous_nodes_torch is not None:
            temporal_loss = torch.mean((nodes - previous_nodes_torch) ** 2)
        else:
            temporal_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        total_loss = (
            DATA_WEIGHT * data_loss
            + VISIBLE_CHAIN_COVERAGE_WEIGHT * coverage_loss
            + SEGMENT_LENGTH_WEIGHT * segment_length_loss
            + TOTAL_LENGTH_WEIGHT * total_length_loss
            + SMOOTHNESS_WEIGHT * smoothness_loss
            + GRAVITY_WEIGHT * gravity_energy
            + HIDDEN_GRAVITY_WEIGHT * hidden_gravity_energy
            + ORDER_WEIGHT * order_loss
            + TEMPORAL_WEIGHT * temporal_loss
        )

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            internal[:, 0].clamp_(x_lower, x_upper)
            internal[:, 1].clamp_(y_lower, y_upper)

        last_loss = total_loss

    with torch.no_grad():
        final_nodes = torch.cat([
            p0.view(1, 2),
            internal,
            pN.view(1, 2),
        ], dim=0)

        nodes_np = final_nodes.detach().cpu().numpy()
        loss_value = float(last_loss.detach().cpu().item()) if last_loss is not None else 0.0

    return nodes_np.astype(np.float64), loss_value


def compute_chain_metrics(nodes, problem):
    obs_points = problem["obs_points"]

    dists = point_to_segments_distance_numpy(obs_points, nodes)

    if len(dists) > 0:
        rmse = float(np.sqrt(np.mean(dists ** 2)))
        median_abs = float(np.median(np.abs(dists)))
    else:
        rmse = 9999.0
        median_abs = 9999.0

    diffs = np.diff(nodes, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    length_rmse = float(
        np.sqrt(np.mean((seg_lengths - problem["target_seg_len"]) ** 2))
    )

    hidden_edges = []

    for i in range(len(nodes) - 1):
        mid_x = 0.5 * (nodes[i, 0] + nodes[i + 1, 0])
        hidden_edges.append(is_x_inside_any_gap(float(mid_x), problem["gaps"]))

    return {
        "rmse_px": rmse,
        "median_abs_error_px": median_abs,
        "length_rmse_px": length_rmse,
        "hidden_edges": hidden_edges,
    }


def fit_catenary_chain_2d(segments, image_shape, previous_nodes=None, locked_endpoints=None):
    problem, early_result = build_chain_problem(
        segments=segments,
        image_shape=image_shape,
        previous_nodes=previous_nodes,
        locked_endpoints=locked_endpoints,
    )

    if early_result is not None:
        return early_result

    nodes, best_loss = optimize_chain_torch_cuda(
        problem,
        previous_nodes=previous_nodes,
    )

    metrics = compute_chain_metrics(nodes, problem)

    return {
        "success": True,
        "nodes": nodes,
        "gaps": problem["gaps"],
        "hidden_edges": metrics["hidden_edges"],
        "rmse_px": metrics["rmse_px"],
        "median_abs_error_px": metrics["median_abs_error_px"],
        "length_rmse_px": metrics["length_rmse_px"],
        "target_segment_length_px": problem["target_seg_len"],
        "estimated_total_length_px": problem["total_len"],
        "endpoint_distance_px": problem["endpoint_distance"],
        "visible_len_px": problem["visible_len"],
        "gap_len_px": problem["gap_len"],
        "num_points": int(len(problem["obs_points"])),
        "num_nodes": int(len(nodes)),
        "num_gaps": int(len(problem["gaps"])),
        "x_min": float(problem["x_min"]),
        "x_max": float(problem["x_max"]),
        "span_px": float(problem["span"]),
        "optimizer": "torch_cuda_catenary_chain_locked_endpoints",
        "loss": float(best_loss),
        "cable_length_multiplier": float(CABLE_LENGTH_MULTIPLIER),
        "data_weight": float(DATA_WEIGHT),
        "gravity_weight": float(GRAVITY_WEIGHT),
        "hidden_gravity_weight": float(HIDDEN_GRAVITY_WEIGHT),
        "locked_endpoints_used": locked_endpoints is not None,
    }


# ============================================================
# Drawing
# ============================================================

def blend_mask_overlay(bgr, mask, color=(0, 0, 255), alpha=0.22):
    out = bgr.copy()

    if mask is None:
        return out

    mask_pixels = mask > 0

    if np.any(mask_pixels):
        color_arr = np.array(color, dtype=np.float32)
        original = out[mask_pixels].astype(np.float32)
        blended = (1.0 - alpha) * original + alpha * color_arr
        out[mask_pixels] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def draw_visible_segments(out, segments):
    for seg in segments:
        if len(seg) < 2:
            continue

        pts = seg.astype(np.int32)

        for i in range(1, len(pts)):
            p0 = (int(pts[i - 1, 0]), int(pts[i - 1, 1]))
            p1 = (int(pts[i, 0]), int(pts[i, 1]))

            cv2.line(
                out,
                p0,
                p1,
                CENTERLINE_COLOR,
                CENTERLINE_THICKNESS,
                cv2.LINE_AA,
            )

        cv2.circle(
            out,
            (int(pts[0, 0]), int(pts[0, 1])),
            5,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            out,
            (int(pts[-1, 0]), int(pts[-1, 1])),
            5,
            (0, 165, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_locked_endpoints(out):
    if LOCKED_ENDPOINTS is None:
        return out

    h, w = out.shape[:2]

    for i, p in enumerate(LOCKED_ENDPOINTS):
        x, y = int(round(p[0])), int(round(p[1]))

        if not (0 <= x < w and 0 <= y < h):
            continue

        cv2.circle(
            out,
            (x, y),
            8,
            LOCKED_ENDPOINT_COLOR,
            -1,
            cv2.LINE_AA,
        )

        label = "LOCK L" if i == 0 else "LOCK R"

        cv2.putText(
            out,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            LOCKED_ENDPOINT_COLOR,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_catenary_chain(out, chain_result):
    if chain_result is None or not chain_result.get("success", False):
        return out

    nodes = chain_result["nodes"]
    hidden_edges = chain_result["hidden_edges"]

    h, w = out.shape[:2]

    for i in range(len(nodes) - 1):
        p0 = nodes[i]
        p1 = nodes[i + 1]

        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))

        if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
            continue

        hidden = hidden_edges[i]
        color = CHAIN_HIDDEN_COLOR if hidden else CHAIN_VISIBLE_COLOR

        cv2.line(
            out,
            (x0, y0),
            (x1, y1),
            color,
            CHAIN_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    if SHOW_CHAIN_NODES:
        for i, p in enumerate(nodes):
            if i % 5 != 0 and i not in [0, len(nodes) - 1]:
                continue

            x, y = int(round(p[0])), int(round(p[1]))

            if 0 <= x < w and 0 <= y < h:
                cv2.circle(
                    out,
                    (x, y),
                    3,
                    CHAIN_NODE_COLOR,
                    -1,
                    cv2.LINE_AA,
                )

    for gap in chain_result.get("gaps", []):
        mid_x = int(round(0.5 * (gap["x0"] + gap["x1"])))

        node_x = nodes[:, 0]
        node_y = nodes[:, 1]

        order = np.argsort(node_x)
        node_x = node_x[order]
        node_y = node_y[order]

        x_unique, unique_idx = np.unique(node_x, return_index=True)
        y_unique = node_y[unique_idx]

        if len(x_unique) >= 2:
            y_mid = int(round(np.interp(mid_x, x_unique, y_unique)))
        else:
            y_mid = int(round(nodes[len(nodes) // 2, 1]))

        if 0 <= mid_x < w and 0 <= y_mid < h:
            cv2.putText(
                out,
                "hidden cable prediction",
                (mid_x - 105, max(24, y_mid - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                CHAIN_HIDDEN_COLOR,
                2,
                cv2.LINE_AA,
            )

    return out


def draw_detection_overlay(bgr, raw_mask, filtered_mask, segments, chain_result):
    out = bgr.copy()

    out = blend_mask_overlay(
        out,
        filtered_mask,
        color=MASK_OVERLAY_COLOR,
        alpha=MASK_ALPHA,
    )

    contours, _ = cv2.findContours(
        filtered_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)

    if SHOW_DETECTED_CENTERLINE:
        draw_visible_segments(out, segments)

    if CHAIN_ENABLED and SHOW_CHAIN:
        draw_catenary_chain(out, chain_result)

    draw_locked_endpoints(out)

    return out


def make_display(
    overlay,
    raw_mask,
    filtered_mask,
    segments,
    components,
    profile,
    chain_result,
    fps,
    show_filtered=True,
):
    if show_filtered:
        shown_mask = filtered_mask
        mask_title = "FILTERED MASK"
    else:
        shown_mask = raw_mask
        mask_title = "RAW HSV MASK"

    mask_bgr = cv2.cvtColor(shown_mask, cv2.COLOR_GRAY2BGR)

    profile_type = profile.get("profile_type", "old_profile")
    num_samples = profile.get("num_samples", "?")
    num_ranges = len(profile.get("hsv_ranges", []))

    locked_text = "LOCKED" if LOCKED_ENDPOINTS is not None else "AUTO"

    cv2.putText(
        overlay,
        f"FPS: {fps:.1f} | Visible segments: {len(segments)} | Components: {len(components)} | endpoints: {locked_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        f"Profile: {profile_type} | Samples: {num_samples} | HSV ranges: {num_ranges}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if CHAIN_ENABLED and chain_result is not None:
        if chain_result.get("success", False):
            cv2.putText(
                overlay,
                (
                    f"Catenary chain: RMSE={chain_result['rmse_px']:.1f}px | "
                    f"LenErr={chain_result['length_rmse_px']:.1f}px | "
                    f"nodes={chain_result['num_nodes']} | gaps={chain_result['num_gaps']}"
                ),
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                overlay,
                (
                    f"slack={chain_result['cable_length_multiplier']:.2f} | "
                    f"data={chain_result['data_weight']:.2f} | "
                    f"gravity={chain_result['gravity_weight']:.3f} | "
                    f"hiddenG={chain_result['hidden_gravity_weight']:.3f}"
                ),
                (20, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay,
                f"Chain not ready: {chain_result.get('reason', 'unknown')}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        "white=visible | blue=hidden | L lock endpoints | X clear | q quit | f mask | p centerline | c chain | +/- slack | [] data | h/j hiddenG",
        (20, overlay.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        mask_bgr,
        mask_title,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    combined = np.hstack([overlay, mask_bgr])

    return combined


# ============================================================
# Main
# ============================================================

def main():
    global SHOW_FILTERED_MASK
    global SHOW_DETECTED_CENTERLINE
    global SHOW_CHAIN
    global SHOW_CHAIN_NODES
    global PREVIOUS_CHAIN_NODES
    global CABLE_LENGTH_MULTIPLIER
    global DATA_WEIGHT
    global GRAVITY_WEIGHT
    global HIDDEN_GRAVITY_WEIGHT
    global LOCKED_ENDPOINTS

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script requires CUDA-enabled PyTorch."
        )

    print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

    profile = load_profile(PROFILE_PATH)

    zed = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = CAMERA_RESOLUTION
    init.camera_fps = CAMERA_FPS

    if USE_ZED_DEPTH:
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.depth_stabilization = 30
    else:
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.depth_stabilization = 0

    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.3
    init.depth_maximum_distance = 3.0

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

    window_name = "Cable Detection - Catenary Chain with Endpoint Lock"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nControls:")
    print("  q       -> quit")
    print("  r       -> reload cable_color_profile.json")
    print("  f       -> toggle raw / filtered mask")
    print("  p       -> show/hide detected visible centerline")
    print("  c       -> show/hide catenary chain")
    print("  n       -> show/hide chain nodes")
    print("  l       -> lock endpoints from current visible cable")
    print("  x       -> clear locked endpoints")
    print("  + or =  -> increase slack / more sag")
    print("  - or _  -> decrease slack / tighter cable")
    print("  ]       -> increase data weight / follow detection more")
    print("  [       -> decrease data weight / trust physics more")
    print("  .       -> increase gravity")
    print("  ,       -> decrease gravity")
    print("  h       -> increase hidden gravity")
    print("  j       -> decrease hidden gravity")
    print("  s       -> save screenshot\n")

    frame_idx = 0
    last_time = time.time()
    fps = 0.0

    last_chain_result = None
    last_segments = []
    last_filtered_mask = None

    last_overlay = None
    last_mask = None

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            frame_idx += 1

            now = time.time()
            dt = now - last_time
            last_time = now

            if dt > 1e-6:
                fps = 0.90 * fps + 0.10 * (1.0 / dt)

            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            left_raw = left_image.get_data()

            # ZED returns BGRA-like image for OpenCV.
            bgr = cv2.cvtColor(left_raw, cv2.COLOR_BGRA2BGR)

            if PROCESS_SCALE != 1.0:
                proc_bgr = cv2.resize(
                    bgr,
                    None,
                    fx=PROCESS_SCALE,
                    fy=PROCESS_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                proc_bgr = bgr

            raw_mask, cleaned_mask, filtered_mask, components = create_masks_from_profile(
                proc_bgr,
                profile,
            )

            segments = extract_visible_centerline_segments(filtered_mask)

            should_update_chain = (
                CHAIN_ENABLED
                and frame_idx % CHAIN_UPDATE_INTERVAL == 0
            )

            if should_update_chain:
                locked_endpoints = LOCKED_ENDPOINTS if USE_LOCKED_ENDPOINTS else None

                chain_result = fit_catenary_chain_2d(
                    segments=segments,
                    image_shape=proc_bgr.shape[:2],
                    previous_nodes=PREVIOUS_CHAIN_NODES,
                    locked_endpoints=locked_endpoints,
                )

                if chain_result.get("success", False):
                    PREVIOUS_CHAIN_NODES = chain_result["nodes"].copy()
                    last_chain_result = chain_result
                    last_segments = segments
                    last_filtered_mask = filtered_mask
            else:
                chain_result = last_chain_result

                if last_filtered_mask is not None:
                    filtered_mask = last_filtered_mask

                if len(last_segments) > 0:
                    segments = last_segments

            overlay = draw_detection_overlay(
                proc_bgr,
                raw_mask,
                filtered_mask,
                segments,
                chain_result,
            )

            display = make_display(
                overlay,
                raw_mask,
                filtered_mask,
                segments,
                components,
                profile,
                chain_result,
                fps=fps,
                show_filtered=SHOW_FILTERED_MASK,
            )

            last_overlay = overlay
            last_mask = filtered_mask

            cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            print("Reloading profile...")
            profile = load_profile(PROFILE_PATH)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None

        elif key == ord("f"):
            SHOW_FILTERED_MASK = not SHOW_FILTERED_MASK
            print("Showing filtered mask." if SHOW_FILTERED_MASK else "Showing raw HSV mask.")

        elif key == ord("p"):
            SHOW_DETECTED_CENTERLINE = not SHOW_DETECTED_CENTERLINE
            print("Showing detected centerline." if SHOW_DETECTED_CENTERLINE else "Hiding detected centerline.")

        elif key == ord("c"):
            SHOW_CHAIN = not SHOW_CHAIN
            print("Showing chain." if SHOW_CHAIN else "Hiding chain.")

        elif key == ord("n"):
            SHOW_CHAIN_NODES = not SHOW_CHAIN_NODES
            print("Showing chain nodes." if SHOW_CHAIN_NODES else "Hiding chain nodes.")

        elif key == ord("l"):
            estimated = estimate_endpoints_from_segments(segments)

            if estimated is not None:
                LOCKED_ENDPOINTS = estimated
                PREVIOUS_CHAIN_NODES = None
                last_chain_result = None

                print("Locked endpoints:")
                print(f"  left:  {LOCKED_ENDPOINTS[0]}")
                print(f"  right: {LOCKED_ENDPOINTS[1]}")
            else:
                print("Could not lock endpoints: no valid cable segments.")

        elif key == ord("x"):
            LOCKED_ENDPOINTS = None
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print("Locked endpoints cleared.")

        elif key == ord("+") or key == ord("="):
            CABLE_LENGTH_MULTIPLIER = min(1.40, CABLE_LENGTH_MULTIPLIER + 0.02)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Cable slack increased: CABLE_LENGTH_MULTIPLIER={CABLE_LENGTH_MULTIPLIER:.2f}")

        elif key == ord("-") or key == ord("_"):
            CABLE_LENGTH_MULTIPLIER = max(1.00, CABLE_LENGTH_MULTIPLIER - 0.02)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Cable slack decreased: CABLE_LENGTH_MULTIPLIER={CABLE_LENGTH_MULTIPLIER:.2f}")

        elif key == ord("]"):
            DATA_WEIGHT = min(8.0, DATA_WEIGHT + 0.25)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Data weight increased: DATA_WEIGHT={DATA_WEIGHT:.2f}")

        elif key == ord("["):
            DATA_WEIGHT = max(0.25, DATA_WEIGHT - 0.25)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Data weight decreased: DATA_WEIGHT={DATA_WEIGHT:.2f}")

        elif key == ord("."):
            GRAVITY_WEIGHT = min(0.20, GRAVITY_WEIGHT + 0.005)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Gravity increased: GRAVITY_WEIGHT={GRAVITY_WEIGHT:.3f}")

        elif key == ord(","):
            GRAVITY_WEIGHT = max(0.0, GRAVITY_WEIGHT - 0.005)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Gravity decreased: GRAVITY_WEIGHT={GRAVITY_WEIGHT:.3f}")

        elif key == ord("h"):
            HIDDEN_GRAVITY_WEIGHT = min(0.30, HIDDEN_GRAVITY_WEIGHT + 0.005)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Hidden gravity increased: HIDDEN_GRAVITY_WEIGHT={HIDDEN_GRAVITY_WEIGHT:.3f}")

        elif key == ord("j"):
            HIDDEN_GRAVITY_WEIGHT = max(0.0, HIDDEN_GRAVITY_WEIGHT - 0.005)
            PREVIOUS_CHAIN_NODES = None
            last_chain_result = None
            print(f"Hidden gravity decreased: HIDDEN_GRAVITY_WEIGHT={HIDDEN_GRAVITY_WEIGHT:.3f}")

        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if last_overlay is not None:
                overlay_name = f"catenary_chain_locked_overlay_{timestamp}.png"
                cv2.imwrite(overlay_name, last_overlay)
                print(f"Saved overlay: {overlay_name}")

            if last_mask is not None:
                mask_name = f"catenary_chain_locked_mask_{timestamp}.png"
                cv2.imwrite(mask_name, last_mask)
                print(f"Saved mask: {mask_name}")

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()