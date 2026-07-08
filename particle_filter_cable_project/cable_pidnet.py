from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - handled by require_torch
    torch = None
    nn = None
    F = None

from cable_detection import CableDetection2D, CableMaskDetector, resize_detection


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for the PIDNet cable detector. Install the CUDA torch build for this machine.")


def torch_inference_mode():
    if torch is None:
        def decorator(fn):
            return fn

        return decorator
    return torch.inference_mode()


TorchModule = nn.Module if nn is not None else object


class ConvBNAct(TorchModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        require_torch()
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvAct(TorchModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        require_torch()
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(TorchModule):
    def __init__(self, channels):
        require_torch()
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class DownsampleBlock(TorchModule):
    def __init__(self, in_channels, out_channels):
        require_torch()
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels, stride=2),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class SimplePyramidPooling(TorchModule):
    def __init__(self, channels, out_channels):
        require_torch()
        super().__init__()
        hidden = max(8, channels // 4)
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvAct(channels, hidden, kernel_size=1))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), ConvAct(channels, hidden, kernel_size=1))
        self.fuse = ConvBNAct(channels + hidden * 2, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.shape[-2:]
        p1 = F.interpolate(self.pool1(x), size=size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.pool2(x), size=size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, p1, p2], dim=1))


class PIDNetSmallBinary(TorchModule):
    """Small PIDNet-inspired binary segmenter.

    The model keeps separate detail, context, and boundary branches like PIDNet,
    but is compact enough to train quickly for a single cable class.
    """

    def __init__(self, base_channels=24, output_channels=1):
        require_torch()
        super().__init__()
        c = int(base_channels)
        output_channels = max(1, int(output_channels))
        self.stem = nn.Sequential(
            ConvBNAct(3, c, stride=2),
            ConvBNAct(c, c * 2, stride=2),
            ResidualBlock(c * 2),
        )

        self.detail = nn.Sequential(
            ResidualBlock(c * 2),
            ResidualBlock(c * 2),
        )

        self.context8 = DownsampleBlock(c * 2, c * 4)
        self.context16 = DownsampleBlock(c * 4, c * 6)
        self.context_ppm = SimplePyramidPooling(c * 6, c * 4)
        self.context_fuse = ConvBNAct(c * 4, c * 2, kernel_size=1)

        self.boundary = nn.Sequential(
            ConvBNAct(c * 2, c, kernel_size=3),
            nn.Conv2d(c, 1, kernel_size=1),
        )
        self.boundary_feature = ConvBNAct(1, c, kernel_size=3)

        self.seg_head = nn.Sequential(
            ConvBNAct(c * 5, c * 2, kernel_size=3),
            ResidualBlock(c * 2),
            nn.Conv2d(c * 2, output_channels, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        stem = self.stem(x)
        detail = self.detail(stem)

        context = self.context8(stem)
        context = self.context16(context)
        context = self.context_ppm(context)
        context = self.context_fuse(F.interpolate(context, size=detail.shape[-2:], mode="bilinear", align_corners=False))

        boundary_logits_low = self.boundary(detail)
        boundary_features = self.boundary_feature(torch.sigmoid(boundary_logits_low))
        fused = torch.cat([detail, context, boundary_features], dim=1)
        seg_logits = self.seg_head(fused)

        seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(boundary_logits_low, size=input_size, mode="bilinear", align_corners=False)
        return {"seg": seg_logits, "boundary": boundary_logits}


class PidNetSegmenter:
    def __init__(self, checkpoint_path, device="auto", base_channels=24):
        require_torch()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PIDNet checkpoint not found: {self.checkpoint_path}")
        self.device = resolve_device(device)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        base_channels = int(config.get("base_channels", base_channels))
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        output_channels = int(config.get("output_channels", infer_output_channels_from_state_dict(state_dict)))
        self.model = PIDNetSmallBinary(base_channels=base_channels, output_channels=output_channels).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @torch_inference_mode()
    def probability_maps(self, bgr):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return np.zeros((*bgr.shape[:2], 1), dtype=np.float32)
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device, non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image = (image - self.mean) / self.std
        logits = self.model(image)["seg"]
        probability = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return np.ascontiguousarray(np.moveaxis(probability, 0, -1), dtype=np.float32)

    @torch_inference_mode()
    def probability_map(self, bgr):
        probability = self.probability_maps(bgr)
        return np.ascontiguousarray(probability[:, :, 0], dtype=np.float32)


class PidNetCableDetector(CableMaskDetector):
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        threshold=0.50,
        base_channels=24,
        min_area=80,
        keep_largest_component=False,
        max_components=0,
        allow_occluded_fragments=True,
        open_kernel=3,
        close_kernel=5,
        skeleton_prune_px=10,
        skeleton_prune_passes=2,
        centerline_smooth_window=9,
    ):
        super().__init__(
            min_area=min_area,
            keep_largest_component=keep_largest_component,
            max_components=max_components,
            allow_occluded_fragments=allow_occluded_fragments,
            open_kernel=open_kernel,
            close_kernel=close_kernel,
            skeleton_prune_px=skeleton_prune_px,
            skeleton_prune_passes=skeleton_prune_passes,
            centerline_smooth_window=centerline_smooth_window,
        )
        self.segmenter = PidNetSegmenter(checkpoint_path, device=device, base_channels=base_channels)
        self.threshold = float(threshold)

    def create_mask(self, bgr):
        probability = self.segmenter.probability_map(bgr)
        return (probability >= self.threshold).astype(np.uint8) * 255

    def create_endpoint_mask(self, bgr, threshold=None):
        probability = self.segmenter.probability_maps(bgr)
        threshold = self.threshold if threshold is None else float(threshold)
        return probability_channel_mask(probability, 1, threshold)

    def detect_with_channel_masks(self, bgr, scale=1.0, extract_geometry=True, endpoint_threshold=None):
        bgr = np.asarray(bgr, dtype=np.uint8)
        original_h, original_w = bgr.shape[:2]
        scale = float(np.clip(scale, 0.10, 1.0))
        if scale < 0.999:
            scaled_w = max(2, int(round(original_w * scale)))
            scaled_h = max(2, int(round(original_h * scale)))
            detector_input = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            detector_input = bgr

        probability = self.segmenter.probability_maps(detector_input)
        cable_raw = probability_channel_mask(probability, 0, self.threshold)
        endpoint_mask = probability_channel_mask(
            probability,
            1,
            self.threshold if endpoint_threshold is None else float(endpoint_threshold),
        )
        detection = self._detection_from_raw_mask(cable_raw, extract_geometry=extract_geometry)
        if scale < 0.999:
            detection = resize_detection(detection, (original_h, original_w))
            endpoint_mask = resize_mask(endpoint_mask, (original_h, original_w))
        return detection, endpoint_mask

    def _detection_from_raw_mask(self, raw_mask, extract_geometry=True):
        mask, component_count = self.clean_mask(raw_mask)
        if not extract_geometry:
            skeleton = np.zeros_like(mask, dtype=np.uint8)
            return CableDetection2D(
                mask=mask,
                skeleton=skeleton,
                centerline_xy=np.empty((0, 2), dtype=np.float32),
                component_count=component_count,
                branch_count=0,
                centerline_paths_xy=(),
            )
        return self._detect_crop_from_clean_mask(mask, component_count)

    def _detect_crop_from_clean_mask(self, mask, component_count):
        from cable_detection import prune_short_skeleton_branches, skeleton_centerline_paths, skeletonize_mask, smooth_polyline_xy, stitch_centerline_paths

        skeleton = skeletonize_mask(mask)
        skeleton = prune_short_skeleton_branches(
            skeleton,
            max_branch_length=self.skeleton_prune_px,
            max_passes=self.skeleton_prune_passes,
        )
        centerline_paths_xy, branch_count = skeleton_centerline_paths(skeleton)
        centerline_xy = stitch_centerline_paths(centerline_paths_xy)
        centerline_xy = smooth_polyline_xy(centerline_xy, self.centerline_smooth_window)
        return CableDetection2D(
            mask=mask,
            skeleton=skeleton,
            centerline_xy=centerline_xy,
            component_count=component_count,
            branch_count=branch_count,
            centerline_paths_xy=tuple(centerline_paths_xy),
        )


def resolve_device(device):
    require_torch()
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for PIDNet, but torch.cuda.is_available() is false.")
    return resolved


def infer_output_channels_from_state_dict(state_dict):
    if isinstance(state_dict, dict):
        weight = state_dict.get("seg_head.2.weight")
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
            return int(weight.shape[0])
    return 1


def probability_channel_mask(probability, channel, threshold):
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 3 or probability.shape[2] <= int(channel):
        return np.zeros(probability.shape[:2], dtype=np.uint8)
    mask = (probability[:, :, int(channel)] >= float(threshold)).astype(np.uint8) * 255
    return np.ascontiguousarray(mask, dtype=np.uint8)


def resize_mask(mask, output_shape):
    output_h, output_w = [int(v) for v in output_shape[:2]]
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[:2] == (output_h, output_w):
        return np.ascontiguousarray(mask, dtype=np.uint8)
    return np.ascontiguousarray(cv2.resize(mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST), dtype=np.uint8)
