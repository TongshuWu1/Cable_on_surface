import numpy as np
import pyzed.sl as sl


RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1200": sl.RESOLUTION.HD1200,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "SVGA": sl.RESOLUTION.SVGA,
    "VGA": sl.RESOLUTION.VGA,
}

DEPTH_MODES = {
    "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
}

def configure_input_source(init, args):
    if args.input_svo_file:
        init.set_from_svo_file(args.input_svo_file)
        print(f"Using SVO input: {args.input_svo_file}")
        return

    if not args.ip_address:
        return

    ip = args.ip_address
    if ip.replace(":", "").replace(".", "").isdigit() and len(ip.split(".")) == 4 and len(ip.split(":")) == 2:
        host, port = ip.split(":")
        init.set_from_stream(host, int(port))
        print(f"Using stream input: {ip}")
    elif ip.replace(":", "").replace(".", "").isdigit() and len(ip.split(".")) == 4:
        init.set_from_stream(ip)
        print(f"Using stream input: {ip}")
    else:
        raise ValueError(f"Invalid IP address format: {ip}")


def live_point_cloud_to_vertices(
    point_cloud,
    stride=4,
    max_points=0,
    depth_min=0.1,
    depth_max=None,
    return_stats=False,
):
    stride = max(1, int(stride))
    max_points = max(0, int(max_points))
    stats = {
        "shape": None,
        "sampled": 0,
        "finite": 0,
        "in_range": 0,
        "returned": 0,
        "capped": False,
    }

    try:
        point_data = np.asarray(point_cloud.get_data())
    except Exception:
        empty = np.empty((0, 6), dtype=np.float32)
        return (empty, stats) if return_stats else empty

    if point_data.ndim != 3 or point_data.shape[2] < 4:
        empty = np.empty((0, 6), dtype=np.float32)
        return (empty, stats) if return_stats else empty

    stats["shape"] = tuple(int(v) for v in point_data.shape)

    sampled = point_data[::stride, ::stride]
    xyz = sampled[:, :, :3].reshape(-1, 3).astype(np.float32)
    rgba = sampled[:, :, 3].reshape(-1)
    stats["sampled"] = int(len(xyz))

    finite = np.all(np.isfinite(xyz), axis=1)
    valid = finite.copy()
    distances = np.linalg.norm(xyz, axis=1)
    if depth_min is not None:
        valid &= distances >= float(depth_min)
    if depth_max is not None:
        valid &= distances <= float(depth_max)

    stats["finite"] = int(np.count_nonzero(finite))
    stats["in_range"] = int(np.count_nonzero(valid))
    if not np.any(valid):
        empty = np.empty((0, 6), dtype=np.float32)
        return (empty, stats) if return_stats else empty

    xyz = xyz[valid]
    rgb = decode_zed_rgba_to_rgb_float(rgba[valid])

    vertices = np.empty((len(xyz), 6), dtype=np.float32)
    vertices[:, :3] = xyz
    vertices[:, 3:] = rgb

    if max_points > 0 and len(vertices) > max_points:
        selected = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
        vertices = vertices[selected]
        stats["capped"] = True

    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    stats["returned"] = int(len(vertices))
    return (vertices, stats) if return_stats else vertices


def decode_zed_rgba_to_rgb_float(color_values):
    color_values = np.asarray(color_values)
    default_rgb = np.full((color_values.size, 3), 0.72, dtype=np.float32)

    try:
        packed = color_values.astype(np.float32, copy=False).reshape(-1).view(np.uint32)
    except Exception:
        try:
            packed = color_values.astype(np.uint32, copy=False).reshape(-1)
        except Exception:
            return default_rgb

    red = (packed & 0x000000FF).astype(np.float32)
    green = ((packed & 0x0000FF00) >> 8).astype(np.float32)
    blue = ((packed & 0x00FF0000) >> 16).astype(np.float32)
    rgb = np.column_stack([red, green, blue]) / 255.0

    weak = np.sum(rgb, axis=1) <= 0.01
    if np.any(weak):
        rgb[weak] = default_rgb[weak]

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def configure_viewer_from_zed(zed, viewer):
    try:
        info = zed.get_camera_information()
        left_cam = info.camera_configuration.calibration_parameters.left_cam
        viewer.set_camera_fov(getattr(left_cam, "v_fov", None))
        print(f"Using ZED vertical FOV: {viewer.fov_y_deg:.1f} deg")
    except Exception as exc:
        print(f"Using default OpenGL FOV ({exc})")
