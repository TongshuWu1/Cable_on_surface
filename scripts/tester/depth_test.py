import cv2
import math
import numpy as np
import pyzed.sl as sl

def make_close_range_depth_vis(depth_np, dmin=0.30, dmax=1.20):
    depth = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth > 0

    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth, dmin, dmax)
        vis[valid] = ((clipped[valid] - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    vis_color[~valid] = 0
    return vis_color

zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1080
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.METER
init.depth_minimum_distance = 0.3
init.depth_maximum_distance = 5
init.depth_stabilization = 30

status = zed.open(init)
print("Open status:", status)
if status != sl.ERROR_CODE.SUCCESS:
    raise SystemExit

runtime = sl.RuntimeParameters()
runtime.remove_saturated_areas = False
runtime.confidence_threshold = 95
runtime.texture_confidence_threshold = 100

left_image = sl.Mat()
depth_map = sl.Mat()
point_cloud = sl.Mat()

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        left_np = left_image.get_data()
        left_bgr = cv2.cvtColor(left_np, cv2.COLOR_RGBA2BGR)

        depth_np = depth_map.get_data()
        depth_vis = make_close_range_depth_vis(depth_np, dmin=0.30, dmax=1.20)

        h, w = left_bgr.shape[:2]
        cx, cy = w // 2, h // 2

        cv2.circle(left_bgr, (cx, cy), 5, (0, 255, 0), -1)
        cv2.circle(depth_vis, (cx, cy), 5, (0, 255, 0), -1)

        err, pc = point_cloud.get_value(cx, cy)
        if err == sl.ERROR_CODE.SUCCESS:
            x, y, z = pc[0], pc[1], pc[2]
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                dist = math.sqrt(x*x + y*y + z*z)
                text = f"Center distance: {dist:.3f} m"
            else:
                text = "Center distance: invalid"
        else:
            text = "Center distance: error"

        cv2.putText(left_bgr, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(depth_vis, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        combined = np.hstack((left_bgr, depth_vis))
        cv2.imshow("ZED Live | RGB + Close-Range Depth", combined)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

zed.close()
cv2.destroyAllWindows()