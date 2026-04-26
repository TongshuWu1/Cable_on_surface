import math
from typing import Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl


# ============================================================
# ZED parameters from your screenshots
# ============================================================

CAMERA_RESOLUTION = sl.RESOLUTION.HD1080
CAMERA_FPS = 30

DEPTH_MODE = sl.DEPTH_MODE.NEURAL
DEPTH_MIN_M = 0.01
DEPTH_MAX_M = 5.0
DEPTH_STABILIZATION = 40

CONFIDENCE_THRESHOLD = 100
TEXTURE_CONFIDENCE_THRESHOLD = 100
REMOVE_SATURATED_AREAS = False
ENABLE_FILL_MODE = False

# Camera tab parameters
BRIGHTNESS = 4
CONTRAST = 4
HUE = 0
SATURATION = 4
SHARPNESS = 4
GAMMA = 5

AUTO_WHITE_BALANCE = True
AUTO_EXPOSURE_GAIN = False
GAIN = 9
EXPOSURE = 70


# ============================================================
# Red balloon HSV thresholds
# Red wraps around in HSV, so use two ranges.
# You will likely tune these live.
# ============================================================

LOWER_RED_1 = np.array([0, 120, 70], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)

LOWER_RED_2 = np.array([170, 120, 70], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 800
POINT_CLOUD_RADIUS = 2


def try_set_camera_setting(zed: sl.Camera, setting, value) -> None:
    try:
        if zed.is_camera_setting_supported(setting):
            zed.set_camera_settings(setting, value)
        else:
            print(f"Skipping unsupported camera setting: {setting}")
    except Exception as e:
        print(f"Warning: failed to set {setting} -> {value}: {e}")


def apply_camera_settings(zed: sl.Camera) -> None:
    try_set_camera_setting(zed, sl.VIDEO_SETTINGS.BRIGHTNESS, BRIGHTNESS)
    try_set_camera_setting(zed, sl.VIDEO_SETTINGS.CONTRAST, CONTRAST)
    try_set_camera_setting(zed, sl.VIDEO_SETTINGS.HUE, HUE)
    try_set_camera_setting(zed, sl.VIDEO_SETTINGS.SATURATION, SATURATION)
    try_set_camera_setting(zed, sl.VIDEO_SETTINGS.SHARPNESS, SHARPNESS)
    try_set_camera_setting(zed, sl.VIDEO_SETTINGS.GAMMA, GAMMA)

    try_set_camera_setting(
        zed, sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1 if AUTO_WHITE_BALANCE else 0
    )

    try_set_camera_setting(
        zed, sl.VIDEO_SETTINGS.AEC_AGC, 1 if AUTO_EXPOSURE_GAIN else 0
    )

    if not AUTO_EXPOSURE_GAIN:
        try_set_camera_setting(zed, sl.VIDEO_SETTINGS.GAIN, GAIN)
        try_set_camera_setting(zed, sl.VIDEO_SETTINGS.EXPOSURE, EXPOSURE)


def open_zed() -> Tuple[sl.Camera, sl.RuntimeParameters]:
    zed = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = CAMERA_RESOLUTION
    init.camera_fps = CAMERA_FPS
    init.depth_mode = DEPTH_MODE
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = DEPTH_MIN_M
    init.depth_maximum_distance = DEPTH_MAX_M
    init.depth_stabilization = DEPTH_STABILIZATION

    status = zed.open(init)
    print("Open status:", status)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {status}")

    apply_camera_settings(zed)

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = CONFIDENCE_THRESHOLD
    runtime.texture_confidence_threshold = TEXTURE_CONFIDENCE_THRESHOLD
    runtime.remove_saturated_areas = REMOVE_SATURATED_AREAS
    runtime.enable_fill_mode = ENABLE_FILL_MODE

    return zed, runtime


def detect_red_balloon(bgr: np.ndarray):
    """
    Returns:
        center: (cx, cy) or None
        radius: float or None
        mask: binary mask
        contour: largest contour or None
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask, None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return None, None, mask, None

    (x, y), radius = cv2.minEnclosingCircle(contour)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None, mask, contour

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy), float(radius), mask, contour


def robust_xyz_from_point_cloud(
    point_cloud: sl.Mat, x: int, y: int, radius: int = 2
) -> Optional[np.ndarray]:
    samples = []
    width = point_cloud.get_width()
    height = point_cloud.get_height()

    x0 = max(0, x - radius)
    x1 = min(width - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(height - 1, y + radius)

    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            err, value = point_cloud.get_value(xx, yy)
            if err == sl.ERROR_CODE.SUCCESS:
                X, Y, Z = value[0], value[1], value[2]
                if math.isfinite(X) and math.isfinite(Y) and math.isfinite(Z) and Z > 0:
                    samples.append([X, Y, Z])

    if not samples:
        return None

    return np.median(np.array(samples, dtype=np.float32), axis=0)


def main():
    zed, runtime = open_zed()

    left_image = sl.Mat()
    point_cloud = sl.Mat()
    depth_view = sl.Mat()

    print("Controls:")
    print("  q: quit")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Left image + point cloud are aligned in the same frame
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(depth_view, sl.VIEW.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            left_rgba = left_image.get_data()
            left_bgr = cv2.cvtColor(left_rgba, cv2.COLOR_BGRA2BGR)

            depth_rgba = depth_view.get_data()
            depth_bgr = cv2.cvtColor(depth_rgba, cv2.COLOR_BGRA2BGR)

            center, radius, mask, contour = detect_red_balloon(left_bgr)

            if contour is not None:
                cv2.drawContours(left_bgr, [contour], -1, (0, 255, 255), 2)

            if center is not None and radius is not None:
                cx, cy = center

                cv2.circle(left_bgr, (cx, cy), int(radius), (0, 255, 0), 2)
                cv2.circle(left_bgr, (cx, cy), 5, (255, 0, 0), -1)

                cv2.circle(depth_bgr, (cx, cy), int(radius), (0, 255, 0), 2)
                cv2.circle(depth_bgr, (cx, cy), 5, (255, 0, 0), -1)

                xyz = robust_xyz_from_point_cloud(point_cloud, cx, cy, POINT_CLOUD_RADIUS)

                if xyz is not None:
                    X, Y, Z = xyz.tolist()
                    dist = math.sqrt(X * X + Y * Y + Z * Z)

                    text1 = f"Balloon center: ({cx}, {cy})"
                    text2 = f"XYZ: ({X:.3f}, {Y:.3f}, {Z:.3f}) m"
                    text3 = f"Distance: {dist:.3f} m"

                    print(text3)

                    for img in (left_bgr, depth_bgr):
                        cv2.putText(img, text1, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, text2, (20, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, text3, (20, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    for img in (left_bgr, depth_bgr):
                        cv2.putText(img, "Depth: invalid", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                for img in (left_bgr, depth_bgr):
                    cv2.putText(img, "No red balloon detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            top = np.hstack((left_bgr, depth_bgr))
            bottom = np.hstack((mask_bgr, mask_bgr))
            combined = np.vstack((top, bottom))

            cv2.imshow("Red Balloon Depth Tracker", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()