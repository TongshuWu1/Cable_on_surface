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
    def __init__(self, checkpoint_path, device="auto", base_channels=24, amp=True, channels_last=True):
        require_torch()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PIDNet checkpoint not found: {self.checkpoint_path}")
        self.device = resolve_device(device)
        self.use_amp = bool(amp) and self.device.type == "cuda"
        self.channels_last = bool(channels_last) and self.device.type == "cuda"
        configure_torch_inference(self.device)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        self.config = dict(config) if isinstance(config, dict) else {}
        base_channels = int(config.get("base_channels", base_channels) or base_channels)
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        output_channels = int(config.get("output_channels", infer_output_channels_from_state_dict(state_dict)) or infer_output_channels_from_state_dict(state_dict))
        self.output_channels = int(output_channels)
        self.label_mode = str(config.get("label_mode") or "binary").strip().lower()
        self.cable_count = max(1, int(config.get("cable_count") or 1))
        self.endpoint_channels = bool(config.get("endpoint_channels", False)) or "endpoint" in self.label_mode
        self.model = PIDNetSmallBinary(base_channels=base_channels, output_channels=output_channels).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _input_tensor(self, bgr):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return None
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device, non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        return (image - self.mean) / self.std

    def _forward_logits(self, bgr):
        image = self._input_tensor(bgr)
        if image is None:
            return None
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            return self.model(image)["seg"]

    @torch_inference_mode()
    def probability_maps(self, bgr):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return np.zeros((*bgr.shape[:2], 1), dtype=np.float32)
        logits = self._forward_logits(bgr)
        if logits is None:
            return np.zeros((*bgr.shape[:2], 1), dtype=np.float32)
        probability = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return np.ascontiguousarray(np.moveaxis(probability, 0, -1), dtype=np.float32)

    @torch_inference_mode()
    def probability_map(self, bgr):
        probability = self.probability_maps(bgr)
        return np.ascontiguousarray(probability[:, :, 0], dtype=np.float32)

    @torch_inference_mode()
    def mask_channels(self, bgr, channels, thresholds):
        bgr = np.asarray(bgr, dtype=np.uint8)
        output_shape = bgr.shape[:2]
        logits = self._forward_logits(bgr)
        if logits is None:
            return [np.zeros(output_shape, dtype=np.uint8) for _channel in channels]

        logits = logits[0]
        masks = []
        for channel, threshold in zip(channels, thresholds):
            channel = int(channel)
            if channel < 0 or channel >= int(logits.shape[0]):
                masks.append(np.zeros(output_shape, dtype=np.uint8))
                continue
            threshold_logit = probability_to_logit_threshold(threshold)
            mask = (logits[channel] >= threshold_logit).to(torch.uint8).mul_(255)
            masks.append(np.ascontiguousarray(mask.detach().cpu().numpy(), dtype=np.uint8))
        return masks


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
        amp=True,
        channels_last=True,
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
        self.segmenter = PidNetSegmenter(
            checkpoint_path,
            device=device,
            base_channels=base_channels,
            amp=amp,
            channels_last=channels_last,
        )
        self.threshold = float(threshold)

    @property
    def output_channels(self):
        return int(getattr(self.segmenter, "output_channels", 1))

    @property
    def label_mode(self):
        return str(getattr(self.segmenter, "label_mode", "binary")).strip().lower()

    @property
    def trained_cable_count(self):
        return max(1, int(getattr(self.segmenter, "cable_count", 1)))

    @property
    def has_endpoint_channels(self):
        return bool(getattr(self.segmenter, "endpoint_channels", False))

    def has_instance_channels(self, cable_count):
        return self.can_use_instance_channels(cable_count, force=False)

    def can_use_instance_channels(self, cable_count, force=False):
        cable_count = max(1, int(cable_count))
        if bool(force):
            return self.output_channels >= cable_count
        return (
            self.label_mode in ("instances", "instances_with_endpoints")
            and self.trained_cable_count >= cable_count
            and self.output_channels >= cable_count
        )

    def endpoint_channel_for_cable(self, cable_index, cable_count, force_instances=False):
        cable_count = max(1, int(cable_count))
        cable_index = int(np.clip(int(cable_index), 0, cable_count - 1))
        if self.can_use_instance_channels(cable_count, force=force_instances):
            if (self.has_endpoint_channels or bool(force_instances)) and self.output_channels >= 2 * cable_count:
                return cable_count + cable_index
            channel = cable_count
            return channel if channel < self.output_channels else None
        return 1 if self.output_channels > 1 else None

    def endpoint_channel_for_cable_count(self, cable_count, force_instances=False):
        return self.endpoint_channel_for_cable(0, cable_count, force_instances=force_instances)

    def create_mask(self, bgr):
        return self.segmenter.mask_channels(bgr, (0,), (self.threshold,))[0]

    def create_endpoint_mask(self, bgr, threshold=None, channel=None):
        threshold = self.threshold if threshold is None else float(threshold)
        channel = self.endpoint_channel_for_cable_count(1) if channel is None else int(channel)
        if channel is None:
            return np.zeros(np.asarray(bgr).shape[:2], dtype=np.uint8)
        return self.segmenter.mask_channels(bgr, (channel,), (threshold,))[0]

    def detect_with_channel_masks(
        self,
        bgr,
        scale=1.0,
        extract_geometry=True,
        endpoint_threshold=None,
        include_endpoint_mask=True,
    ):
        bgr = np.asarray(bgr, dtype=np.uint8)
        original_h, original_w = bgr.shape[:2]
        scale = float(np.clip(scale, 0.10, 1.0))
        if scale < 0.999:
            scaled_w = max(2, int(round(original_w * scale)))
            scaled_h = max(2, int(round(original_h * scale)))
            detector_input = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            detector_input = bgr

        channels = [0]
        thresholds = [self.threshold]
        endpoint_mask_index = None
        if bool(include_endpoint_mask):
            endpoint_channel = self.endpoint_channel_for_cable_count(1)
            if endpoint_channel is not None:
                endpoint_mask_index = len(channels)
                channels.append(endpoint_channel)
                thresholds.append(self.threshold if endpoint_threshold is None else float(endpoint_threshold))
        masks = self.segmenter.mask_channels(detector_input, channels, thresholds)
        cable_raw = masks[0]
        endpoint_mask = masks[endpoint_mask_index] if endpoint_mask_index is not None else None
        detection = self._detection_from_raw_mask(cable_raw, extract_geometry=extract_geometry)
        if scale < 0.999:
            detection = resize_detection(detection, (original_h, original_w))
            if endpoint_mask is not None:
                endpoint_mask = resize_mask(endpoint_mask, (original_h, original_w))
        return detection, endpoint_mask

    def detect_instance_channel_masks(
        self,
        bgr,
        cable_count=2,
        scale=1.0,
        extract_geometry=True,
        endpoint_threshold=None,
        include_endpoint_mask=True,
        force=False,
    ):
        cable_count = max(1, int(cable_count))
        if not self.can_use_instance_channels(cable_count, force=force):
            return [], None, None, None

        bgr = np.asarray(bgr, dtype=np.uint8)
        original_h, original_w = bgr.shape[:2]
        scale = float(np.clip(scale, 0.10, 1.0))
        if scale < 0.999:
            scaled_w = max(2, int(round(original_w * scale)))
            scaled_h = max(2, int(round(original_h * scale)))
            detector_input = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            detector_input = bgr

        channels = list(range(cable_count))
        thresholds = [self.threshold for _index in channels]
        endpoint_channels_by_cable = [None for _index in range(cable_count)]
        endpoint_channel_to_mask_index = {}
        if bool(include_endpoint_mask):
            for cable_index in range(cable_count):
                endpoint_channel = self.endpoint_channel_for_cable(
                    cable_index,
                    cable_count,
                    force_instances=force,
                )
                endpoint_channels_by_cable[cable_index] = endpoint_channel
                if endpoint_channel is None or endpoint_channel in endpoint_channel_to_mask_index:
                    continue
                endpoint_channel_to_mask_index[endpoint_channel] = len(channels)
                channels.append(endpoint_channel)
                thresholds.append(self.threshold if endpoint_threshold is None else float(endpoint_threshold))

        masks = self.segmenter.mask_channels(detector_input, channels, thresholds)
        cable_masks = masks[:cable_count]
        endpoint_masks = []
        for endpoint_channel in endpoint_channels_by_cable:
            if endpoint_channel is None:
                endpoint_masks.append(None)
                continue
            endpoint_masks.append(masks[endpoint_channel_to_mask_index[endpoint_channel]])
        endpoint_mask = None
        for mask in endpoint_masks:
            if mask is None:
                continue
            endpoint_mask = mask.copy() if endpoint_mask is None else cv2.bitwise_or(endpoint_mask, mask)

        detections = [
            self._detection_from_raw_mask(mask, extract_geometry=extract_geometry)
            for mask in cable_masks
        ]
        combined_raw = np.zeros_like(cable_masks[0], dtype=np.uint8) if cable_masks else np.zeros(detector_input.shape[:2], dtype=np.uint8)
        for mask in cable_masks:
            combined_raw = cv2.bitwise_or(combined_raw, mask)
        combined_detection = self._detection_from_raw_mask(combined_raw, extract_geometry=extract_geometry)

        if scale < 0.999:
            detections = [resize_detection(detection, (original_h, original_w)) for detection in detections]
            combined_detection = resize_detection(combined_detection, (original_h, original_w))
            if endpoint_mask is not None:
                endpoint_mask = resize_mask(endpoint_mask, (original_h, original_w))
            endpoint_masks = [
                None if mask is None else resize_mask(mask, (original_h, original_w))
                for mask in endpoint_masks
            ]
        return detections, combined_detection, endpoint_mask, endpoint_masks

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


def configure_torch_inference(device):
    require_torch()
    if torch is None or device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def infer_output_channels_from_state_dict(state_dict):
    if isinstance(state_dict, dict):
        weight = state_dict.get("seg_head.2.weight")
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
            return int(weight.shape[0])
    return 1


def probability_to_logit_threshold(threshold):
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    return float(np.log(threshold / (1.0 - threshold)))


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
