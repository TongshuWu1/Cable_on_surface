from pathlib import Path
from contextlib import nullcontext

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - handled by require_torch
    torch = None
    nn = None
    F = None

from cable_detection import CableMaskDetector


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

    def __init__(self, base_channels=24):
        require_torch()
        super().__init__()
        c = int(base_channels)
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
            nn.Conv2d(c * 2, 1, kernel_size=1),
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
    def __init__(self, checkpoint_path, device="auto", base_channels=24, amp=True):
        require_torch()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PIDNet checkpoint not found: {self.checkpoint_path}")
        self.device = resolve_device(device)
        self.use_amp = bool(amp) and self.device.type == "cuda"
        if self.device.type == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        base_channels = int(config.get("base_channels", base_channels))
        self.model = PIDNetSmallBinary(base_channels=base_channels)
        if self.device.type == "cuda":
            self.model = self.model.to(device=self.device, memory_format=torch.channels_last)
        else:
            self.model = self.model.to(self.device)
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _input_tensor(self, bgr):
        bgr = np.asarray(bgr, dtype=np.uint8)
        image = torch.from_numpy(np.ascontiguousarray(bgr[:, :, :3]))
        image = image.to(self.device, non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image[:, [2, 1, 0], :, :].to(dtype=torch.float32).mul_(1.0 / 255.0)
        if self.device.type == "cuda":
            image = image.contiguous(memory_format=torch.channels_last)
        else:
            image = image.contiguous()
        return (image - self.mean) / self.std

    def _seg_logits(self, bgr):
        image = self._input_tensor(bgr)
        with cuda_autocast(self.use_amp):
            return self.model(image)["seg"]

    @torch_inference_mode()
    def probability_map(self, bgr):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return np.zeros(bgr.shape[:2], dtype=np.float32)
        logits = self._seg_logits(bgr)
        probability = torch.sigmoid(logits.float())[0, 0].detach().cpu().numpy()
        return np.ascontiguousarray(probability, dtype=np.float32)

    @torch_inference_mode()
    def binary_mask(self, bgr, threshold):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return np.zeros(bgr.shape[:2], dtype=np.uint8)
        logits = self._seg_logits(bgr).float()[0, 0]
        logit_threshold = probability_threshold_to_logit(threshold)
        mask = (logits >= logit_threshold).to(torch.uint8) * 255
        return np.ascontiguousarray(mask.detach().cpu().numpy(), dtype=np.uint8)


class PidNetCableDetector(CableMaskDetector):
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        threshold=0.50,
        base_channels=24,
        amp=True,
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
        self.segmenter = PidNetSegmenter(checkpoint_path, device=device, base_channels=base_channels, amp=amp)
        self.threshold = float(threshold)

    def create_mask(self, bgr):
        return self.segmenter.binary_mask(bgr, self.threshold)


def resolve_device(device):
    require_torch()
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for PIDNet, but torch.cuda.is_available() is false.")
    return resolved


def cuda_autocast(enabled):
    if torch is None or not bool(enabled):
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    except TypeError:
        return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)


def probability_threshold_to_logit(threshold):
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    return float(np.log(threshold / (1.0 - threshold)))
