from dataclasses import dataclass
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
from pidnet_schema import (
    CABLE_CHANNEL,
    CROSSING_CHANNEL,
    ENDPOINT_CHANNELS,
    OUTPUT_CHANNEL_COUNT,
    PIDNET_LABEL_MODE,
    validate_checkpoint_schema,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ACTIVE_LABEL_MODE = PIDNET_LABEL_MODE


@dataclass(frozen=True)
class PidNetObservationMasks:
    """One four-head observation with one endpoint mask per physical cable."""

    cable_detection: CableDetection2D
    endpoint_mask: np.ndarray
    endpoint_masks_by_cable: tuple
    crossing_mask: np.ndarray
    crossing_probability: np.ndarray


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

    def __init__(self, base_channels=24, output_channels=1, input_channels=3):
        require_torch()
        super().__init__()
        c = int(base_channels)
        input_channels = max(1, int(input_channels))
        output_channels = max(1, int(output_channels))
        self.stem = nn.Sequential(
            ConvBNAct(input_channels, c, stride=2),
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
    def __init__(self, checkpoint_path, device="auto", amp=True, channels_last=True):
        require_torch()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PIDNet checkpoint not found: {self.checkpoint_path}")
        self.device = resolve_device(device)
        self.use_amp = bool(amp) and self.device.type == "cuda"
        self.channels_last = bool(channels_last) and self.device.type == "cuda"
        configure_torch_inference(self.device)
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError("PIDNet checkpoint must contain a config dictionary.")
        if "model_state" not in payload:
            raise ValueError("PIDNet checkpoint is missing model_state.")
        config = payload["config"]
        self.config = dict(config)
        required_keys = {
            "base_channels",
            "input_channels",
            "output_channels",
            "input_mode",
            "label_mode",
            "endpoint_channels",
            "crossing_channels",
            "crossing_channel",
        }
        missing = sorted(required_keys.difference(config))
        if missing:
            raise ValueError(f"PIDNet checkpoint config is missing: {', '.join(missing)}")
        validate_checkpoint_schema(config)
        base_channels = int(config["base_channels"])
        state_dict = payload["model_state"]
        output_channels = int(config["output_channels"])
        input_channels = int(config["input_channels"])
        self.output_channels = int(output_channels)
        self.input_channels = int(input_channels)
        self.input_mode = str(config["input_mode"]).strip().lower()
        self.label_mode = str(config["label_mode"]).strip().lower()
        self.endpoint_channel_count = int(config["endpoint_channel_count"])
        self.endpoint_channels = bool(config["endpoint_channels"])
        self.crossing_channels = bool(config["crossing_channels"])
        self.crossing_channel = int(config["crossing_channel"])
        expected_output_channels = OUTPUT_CHANNEL_COUNT
        if self.input_mode != "rgb" or self.input_channels != 3:
            raise ValueError(
                f"PIDNet checkpoint must use RGB input with 3 channels; got mode={self.input_mode!r}, "
                f"channels={self.input_channels}."
            )
        if not self.endpoint_channels or not self.crossing_channels:
            raise ValueError("PIDNet checkpoint must provide endpoint and crossing channels.")
        if self.output_channels != expected_output_channels:
            raise ValueError(
                f"PIDNet checkpoint must output {expected_output_channels} channels; got {self.output_channels}."
            )
        if self.crossing_channel != CROSSING_CHANNEL:
            raise ValueError(
                f"PIDNet crossing channel must be {CROSSING_CHANNEL}; "
                f"got {self.crossing_channel}."
            )
        self.model = PIDNetSmallBinary(
            base_channels=base_channels,
            output_channels=output_channels,
            input_channels=input_channels,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _input_tensor(self, bgr):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            raise ValueError(f"PIDNet input must be an HxWx3 BGR image; got shape {bgr.shape}.")
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device, non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        return (image - self.mean) / self.std

    def _forward_logits(self, bgr):
        image = self._input_tensor(bgr)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            return self.model(image)["seg"]

    @torch_inference_mode()
    def probability_maps(self, bgr):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            raise ValueError(f"PIDNet input must be an HxWx3 BGR image; got shape {bgr.shape}.")
        logits = self._forward_logits(bgr)
        probability = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return np.ascontiguousarray(np.moveaxis(probability, 0, -1), dtype=np.float32)

    @torch_inference_mode()
    def mask_channels(self, bgr, channels, thresholds):
        bgr = np.asarray(bgr, dtype=np.uint8)
        channels = tuple(int(channel) for channel in channels)
        thresholds = tuple(float(threshold) for threshold in thresholds)
        if len(channels) != len(thresholds):
            raise ValueError("PIDNet channels and thresholds must have the same length.")
        invalid = [channel for channel in channels if channel < 0 or channel >= self.output_channels]
        if invalid:
            raise ValueError(f"PIDNet channel indices out of range: {invalid}")
        logits = self._forward_logits(bgr)
        logits = logits[0]
        selected = logits[list(channels)]
        threshold_logits = torch.tensor(
            [probability_to_logit_threshold(threshold) for threshold in thresholds],
            dtype=selected.dtype,
            device=selected.device,
        ).view(-1, 1, 1)
        masks = (selected >= threshold_logits).to(torch.uint8).mul_(255).cpu().numpy()
        return [np.ascontiguousarray(mask, dtype=np.uint8) for mask in masks]

    @torch_inference_mode()
    def observation_channels(self, bgr, thresholds):
        """Evaluate all observation heads once and transfer one compact result."""

        thresholds = tuple(float(value) for value in thresholds)
        if len(thresholds) != OUTPUT_CHANNEL_COUNT:
            raise ValueError(f"Expected {OUTPUT_CHANNEL_COUNT} thresholds; got {len(thresholds)}.")
        logits = self._forward_logits(bgr)[0]
        threshold_logits = torch.tensor(
            [probability_to_logit_threshold(value) for value in thresholds],
            dtype=logits.dtype,
            device=logits.device,
        ).view(-1, 1, 1)
        masks = (logits >= threshold_logits).to(torch.uint8).mul_(255)
        crossing_probability = torch.sigmoid(logits[CROSSING_CHANNEL]).to(torch.float16)
        masks_cpu, crossing_cpu = masks.cpu().numpy(), crossing_probability.cpu().numpy()
        return (
            tuple(np.ascontiguousarray(mask, dtype=np.uint8) for mask in masks_cpu),
            np.ascontiguousarray(crossing_cpu, dtype=np.float16),
        )


class PidNetCableDetector(CableMaskDetector):
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        threshold=0.50,
        min_area=80,
        open_kernel=3,
        close_kernel=5,
        amp=True,
        channels_last=True,
    ):
        super().__init__(
            min_area=min_area,
            open_kernel=open_kernel,
            close_kernel=close_kernel,
        )
        self.segmenter = PidNetSegmenter(
            checkpoint_path,
            device=device,
            amp=amp,
            channels_last=channels_last,
        )
        self.threshold = float(threshold)

    @property
    def output_channels(self):
        return int(getattr(self.segmenter, "output_channels", 1))

    @property
    def label_mode(self):
        return str(self.segmenter.label_mode)

    @property
    def trained_endpoint_channel_count(self):
        return int(self.segmenter.endpoint_channel_count)

    @property
    def has_crossing_channel(self):
        return bool(self.segmenter.crossing_channels)

    @property
    def crossing_channel(self):
        return int(self.segmenter.crossing_channel)

    def detect_observation_masks(
        self,
        bgr,
        endpoint_channel_count=2,
        scale=1.0,
        endpoint_threshold=None,
        crossing_threshold=None,
        include_endpoint_mask=True,
    ):
        endpoint_channel_count = int(endpoint_channel_count)
        if endpoint_channel_count != self.trained_endpoint_channel_count:
            raise ValueError(
                f"Runtime endpoint_channel_count={endpoint_channel_count} does not match checkpoint "
                f"endpoint channel count={self.trained_endpoint_channel_count}."
            )
        if not include_endpoint_mask:
            raise ValueError("The live two-cable tracker requires endpoint masks.")

        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            raise ValueError(f"PIDNet input must be an HxWx3 BGR image; got shape {bgr.shape}.")
        original_h, original_w = bgr.shape[:2]
        scale = float(np.clip(scale, 0.10, 1.0))
        if scale < 0.999:
            scaled_w = max(2, int(round(original_w * scale)))
            scaled_h = max(2, int(round(original_h * scale)))
            detector_input = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            detector_input = bgr

        endpoint_threshold = self.threshold if endpoint_threshold is None else float(endpoint_threshold)
        crossing_threshold = self.threshold if crossing_threshold is None else float(crossing_threshold)
        thresholds = (
            self.threshold,
            *(endpoint_threshold for _ in ENDPOINT_CHANNELS),
            crossing_threshold,
        )
        masks, crossing_probability = self.segmenter.observation_channels(detector_input, thresholds)
        cable_mask = masks[CABLE_CHANNEL]
        endpoint_masks = [masks[channel] for channel in ENDPOINT_CHANNELS]
        crossing_mask = masks[CROSSING_CHANNEL]
        combined_detection = self._detection_from_raw_mask(cable_mask)
        endpoint_mask = np.zeros_like(cable_mask, dtype=np.uint8)
        for mask in endpoint_masks:
            endpoint_mask = cv2.bitwise_or(endpoint_mask, mask)

        if scale < 0.999:
            combined_detection = resize_detection(combined_detection, (original_h, original_w))
            endpoint_mask = resize_mask(endpoint_mask, (original_h, original_w))
            endpoint_masks = [resize_mask(mask, (original_h, original_w)) for mask in endpoint_masks]
            crossing_mask = resize_mask(crossing_mask, (original_h, original_w))
            crossing_probability = cv2.resize(
                crossing_probability.astype(np.float32),
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float16, copy=False)
        return PidNetObservationMasks(
            cable_detection=combined_detection,
            endpoint_mask=np.ascontiguousarray(endpoint_mask, dtype=np.uint8),
            endpoint_masks_by_cable=tuple(endpoint_masks),
            crossing_mask=np.ascontiguousarray(crossing_mask, dtype=np.uint8),
            crossing_probability=np.ascontiguousarray(crossing_probability, dtype=np.float16),
        )

    def _detection_from_raw_mask(self, raw_mask):
        mask, component_count = self.clean_mask(raw_mask)
        return CableDetection2D(
            mask=mask,
            component_count=component_count,
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


def probability_to_logit_threshold(threshold):
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    return float(np.log(threshold / (1.0 - threshold)))


def resize_mask(mask, output_shape):
    output_h, output_w = [int(v) for v in output_shape[:2]]
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[:2] == (output_h, output_w):
        return np.ascontiguousarray(mask, dtype=np.uint8)
    return np.ascontiguousarray(cv2.resize(mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST), dtype=np.uint8)
