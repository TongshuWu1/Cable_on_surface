from pathlib import Path
import heapq

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
RGB_ENDPOINT_PROMPT_INPUT_MODE = "rgb_endpoint_prompt"
RGB_ENDPOINT_PAIR_PROMPT_INPUT_MODE = "rgb_endpoint_pair_prompt"
GENERIC_CABLE_ENDPOINT_LABEL_MODES = {
    "generic",
    "cable_with_endpoints",
    "cable_with_separate_endpoints",
    "cable_endpoint",
    "binary_with_endpoints",
}


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for the PIDNet cable detector. Install the CUDA torch build for this machine.")


def torch_inference_mode():
    if torch is None:
        def decorator(fn):
            return fn

        return decorator
    return torch.inference_mode()


def rgb_prompt_features(bgr, endpoint_prompt, negative_endpoint_prompt=None):
    bgr = np.asarray(bgr, dtype=np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError("RGB endpoint-prompt input requires a BGR image with three channels.")
    prompt = np.asarray(endpoint_prompt, dtype=np.float32)
    if prompt.ndim != 2:
        raise ValueError("Endpoint prompt must be a single-channel image.")
    if prompt.shape[:2] != bgr.shape[:2]:
        prompt = cv2.resize(prompt, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    channels = [
        rgb[:, :, 0],
        rgb[:, :, 1],
        rgb[:, :, 2],
        np.clip(prompt, 0.0, 1.0),
    ]
    if negative_endpoint_prompt is not None:
        negative_prompt = np.asarray(negative_endpoint_prompt, dtype=np.float32)
        if negative_prompt.ndim != 2:
            raise ValueError("Negative endpoint prompt must be a single-channel image.")
        if negative_prompt.shape[:2] != bgr.shape[:2]:
            negative_prompt = cv2.resize(negative_prompt, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        channels.append(np.clip(negative_prompt, 0.0, 1.0))
    features = np.stack(channels, axis=2)
    return np.ascontiguousarray(features, dtype=np.float32)


def endpoint_prompt_heatmap(endpoint_mask, radius_px=14, blur_px=11):
    mask = (np.asarray(endpoint_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if mask.ndim != 2:
        raise ValueError("Endpoint mask must be a single-channel image.")
    if not np.any(mask):
        return np.zeros(mask.shape[:2], dtype=np.float32)
    radius_px = max(0, int(radius_px))
    if radius_px > 0:
        diameter = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        mask = cv2.dilate(mask, kernel, iterations=1)
    heatmap = mask.astype(np.float32) / 255.0
    blur_px = max(0, int(blur_px))
    if blur_px > 1:
        if blur_px % 2 == 0:
            blur_px += 1
        heatmap = cv2.GaussianBlur(heatmap, (blur_px, blur_px), 0)
        max_value = float(np.max(heatmap))
        if max_value > 1e-6:
            heatmap = heatmap / max_value
    return np.ascontiguousarray(np.clip(heatmap, 0.0, 1.0), dtype=np.float32)


def generic_cable_endpoint_label_mode(label_mode):
    return str(label_mode).strip().lower() in GENERIC_CABLE_ENDPOINT_LABEL_MODES


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
        input_channels = int(config.get("input_channels", infer_input_channels_from_state_dict(state_dict)) or infer_input_channels_from_state_dict(state_dict))
        self.output_channels = int(output_channels)
        self.input_channels = int(input_channels)
        self.input_mode = str(config.get("input_mode") or ("rgb" if self.input_channels == 3 else "")).strip().lower()
        self.label_mode = str(config.get("label_mode") or "binary").strip().lower()
        self.cable_count = max(1, int(config.get("cable_count") or 1))
        self.endpoint_channels = bool(config.get("endpoint_channels", False)) or "endpoint" in self.label_mode
        self.prompt_radius_px = int(config.get("prompt_radius_px", 14) or 14)
        self.prompt_blur_px = int(config.get("prompt_blur_px", 11) or 11)
        default_path_association = (
            self.input_mode in {RGB_ENDPOINT_PROMPT_INPUT_MODE, RGB_ENDPOINT_PAIR_PROMPT_INPUT_MODE}
            or self.label_mode == "endpoint_conditioned"
        )
        self.path_association = bool(config.get("path_association", default_path_association))
        self.path_association_max_size = int(config.get("path_association_max_size", 320) or 320)
        self.path_association_radius_px = int(config.get("path_association_radius_px", 5) or 5)
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

    @property
    def requires_endpoint_prompt(self):
        return self.input_mode in {RGB_ENDPOINT_PROMPT_INPUT_MODE, RGB_ENDPOINT_PAIR_PROMPT_INPUT_MODE} or self.label_mode == "endpoint_conditioned"

    @property
    def requires_negative_endpoint_prompt(self):
        return self.input_mode == RGB_ENDPOINT_PAIR_PROMPT_INPUT_MODE or int(self.input_channels) >= 5

    def _input_tensor(self, bgr):
        if self.requires_endpoint_prompt:
            raise RuntimeError("This PIDNet checkpoint is endpoint-conditioned and requires endpoint prompt heatmaps.")
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

    def _prompt_input_tensor(self, bgr, prompt_maps, negative_prompt_maps=None):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return None
        prompts = np.asarray(prompt_maps, dtype=np.float32)
        if prompts.ndim == 2:
            prompts = prompts[None, :, :]
        if prompts.ndim != 3:
            raise ValueError("Endpoint prompts must have shape HxW or NxHxW.")
        negative_prompts = None
        if self.requires_negative_endpoint_prompt:
            negative_prompts = negative_prompt_maps
            if negative_prompts is None:
                negative_prompts = negative_prompt_maps_from_positive(prompts)
            negative_prompts = np.asarray(negative_prompts, dtype=np.float32)
            if negative_prompts.ndim == 2:
                negative_prompts = negative_prompts[None, :, :]
            if negative_prompts.shape != prompts.shape:
                raise ValueError("Negative endpoint prompts must have the same shape as positive endpoint prompts.")
        features = [
            rgb_prompt_features(
                bgr,
                prompt,
                None if negative_prompts is None else negative_prompts[index],
            )
            for index, prompt in enumerate(prompts)
        ]
        image = torch.from_numpy(np.ascontiguousarray(np.stack(features, axis=0))).to(self.device, non_blocking=True)
        image = image.permute(0, 3, 1, 2).float()
        if self.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        return image

    def _forward_prompt_logits(self, bgr, prompt_maps, negative_prompt_maps=None):
        image = self._prompt_input_tensor(bgr, prompt_maps, negative_prompt_maps=negative_prompt_maps)
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
    def prompt_probability_maps(self, bgr, prompt_maps, negative_prompt_maps=None, path_associate=None):
        bgr = np.asarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] < 3:
            return np.zeros((*bgr.shape[:2], 1), dtype=np.float32)
        logits = self._forward_prompt_logits(bgr, prompt_maps, negative_prompt_maps=negative_prompt_maps)
        if logits is None:
            return np.zeros((*bgr.shape[:2], 1), dtype=np.float32)
        probability = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
        probability = np.ascontiguousarray(np.moveaxis(probability, 0, -1), dtype=np.float32)
        use_path_association = self.path_association if path_associate is None else bool(path_associate)
        if use_path_association:
            probability = endpoint_path_probability_maps(
                probability,
                prompt_maps,
                max_size=self.path_association_max_size,
                path_radius_px=self.path_association_radius_px,
            )
        return probability

    @torch_inference_mode()
    def mask_channels(self, bgr, channels, thresholds):
        if self.requires_endpoint_prompt:
            raise RuntimeError("Use prompt_mask_channels for endpoint-conditioned PIDNet checkpoints.")
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

    @torch_inference_mode()
    def prompt_mask_channels(self, bgr, prompt_maps, threshold, negative_prompt_maps=None, path_associate=None):
        bgr = np.asarray(bgr, dtype=np.uint8)
        output_shape = bgr.shape[:2]
        probability = self.prompt_probability_maps(
            bgr,
            prompt_maps,
            negative_prompt_maps=negative_prompt_maps,
            path_associate=path_associate,
        )
        if probability is None or probability.ndim != 3:
            return []
        if probability.shape[:2] != output_shape:
            probability = cv2.resize(probability, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
        return [
            np.ascontiguousarray((probability[:, :, channel] >= float(threshold)).astype(np.uint8) * 255, dtype=np.uint8)
            for channel in range(probability.shape[2])
        ]


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

    @property
    def requires_endpoint_prompt(self):
        return bool(getattr(self.segmenter, "requires_endpoint_prompt", False))

    def has_instance_channels(self, cable_count):
        return self.can_use_instance_channels(cable_count, force=False)

    def can_use_instance_channels(self, cable_count, force=False):
        cable_count = max(1, int(cable_count))
        if self.requires_endpoint_prompt:
            return self.trained_cable_count >= cable_count and self.output_channels >= 1
        if generic_cable_endpoint_label_mode(self.label_mode):
            return self.output_channels >= 1 + cable_count
        if bool(force):
            return self.output_channels >= cable_count
        return (
            self.label_mode in ("instances", "instances_with_endpoints")
            and self.trained_cable_count >= cable_count
            and self.output_channels >= cable_count
        )

    def endpoint_channel_for_cable(self, cable_index, cable_count, force_instances=False):
        if self.requires_endpoint_prompt:
            return None
        cable_count = max(1, int(cable_count))
        cable_index = int(np.clip(int(cable_index), 0, cable_count - 1))
        if generic_cable_endpoint_label_mode(self.label_mode):
            channel = 1 + cable_index
            return channel if channel < self.output_channels else None
        if self.can_use_instance_channels(cable_count, force=force_instances):
            if (self.has_endpoint_channels or bool(force_instances)) and self.output_channels >= 2 * cable_count:
                return cable_count + cable_index
            channel = cable_count
            return channel if channel < self.output_channels else None
        return 1 if self.output_channels > 1 else None

    def endpoint_channel_for_cable_count(self, cable_count, force_instances=False):
        return self.endpoint_channel_for_cable(0, cable_count, force_instances=force_instances)

    def create_mask(self, bgr):
        if self.requires_endpoint_prompt:
            raise RuntimeError("Endpoint-conditioned PIDNet requires endpoint prompts; use detect_instance_channel_masks with endpoint_prompt_masks.")
        return self.segmenter.mask_channels(bgr, (0,), (self.threshold,))[0]

    def create_endpoint_mask(self, bgr, threshold=None, channel=None):
        if self.requires_endpoint_prompt:
            return np.zeros(np.asarray(bgr).shape[:2], dtype=np.uint8)
        threshold = self.threshold if threshold is None else float(threshold)
        if channel is not None:
            channel = int(channel)
            if channel < 0 or channel >= self.output_channels:
                return np.zeros(np.asarray(bgr).shape[:2], dtype=np.uint8)
            return self.segmenter.mask_channels(bgr, (channel,), (threshold,))[0]
        if generic_cable_endpoint_label_mode(self.label_mode):
            cable_count = max(1, int(self.trained_cable_count))
            channels = [
                endpoint_channel
                for endpoint_channel in (self.endpoint_channel_for_cable(index, cable_count) for index in range(cable_count))
                if endpoint_channel is not None
            ]
            if not channels:
                return np.zeros(np.asarray(bgr).shape[:2], dtype=np.uint8)
            masks = self.segmenter.mask_channels(bgr, channels, [threshold for _channel in channels])
            union = np.zeros(np.asarray(bgr).shape[:2], dtype=np.uint8)
            for mask in masks:
                union = cv2.bitwise_or(union, mask)
            return union
        channel = self.endpoint_channel_for_cable_count(1)
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
        if self.requires_endpoint_prompt:
            bgr = np.asarray(bgr, dtype=np.uint8)
            empty = np.zeros(bgr.shape[:2], dtype=np.uint8)
            return self._detection_from_raw_mask(empty, extract_geometry=False), None

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
        endpoint_mask_indices = []
        if bool(include_endpoint_mask):
            if generic_cable_endpoint_label_mode(self.label_mode):
                for cable_index in range(self.trained_cable_count):
                    endpoint_channel = self.endpoint_channel_for_cable(cable_index, self.trained_cable_count)
                    if endpoint_channel is None:
                        continue
                    endpoint_mask_indices.append(len(channels))
                    channels.append(endpoint_channel)
                    thresholds.append(self.threshold if endpoint_threshold is None else float(endpoint_threshold))
            else:
                endpoint_channel = self.endpoint_channel_for_cable_count(1)
                if endpoint_channel is not None:
                    endpoint_mask_index = len(channels)
                    channels.append(endpoint_channel)
                    thresholds.append(self.threshold if endpoint_threshold is None else float(endpoint_threshold))
        masks = self.segmenter.mask_channels(detector_input, channels, thresholds)
        cable_raw = masks[0]
        endpoint_mask = masks[endpoint_mask_index] if endpoint_mask_index is not None else None
        for mask_index in endpoint_mask_indices:
            endpoint_mask = masks[mask_index].copy() if endpoint_mask is None else cv2.bitwise_or(endpoint_mask, masks[mask_index])
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
        endpoint_prompt_masks=None,
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

        if self.requires_endpoint_prompt:
            if endpoint_prompt_masks is None:
                return [], None, None, None
            prompt_masks = list(endpoint_prompt_masks)
            if len(prompt_masks) < cable_count:
                return [], None, None, None
            prompt_maps = []
            for mask in prompt_masks[:cable_count]:
                if mask is None:
                    return [], None, None, None
                mask = np.asarray(mask, dtype=np.float32)
                if mask.shape[:2] != detector_input.shape[:2]:
                    mask = cv2.resize(mask, (detector_input.shape[1], detector_input.shape[0]), interpolation=cv2.INTER_LINEAR)
                prompt_maps.append(np.clip(mask, 0.0, 1.0))
            cable_masks = self.segmenter.prompt_mask_channels(
                detector_input,
                np.stack(prompt_maps, axis=0),
                self.threshold,
            )
            detections = [
                self._detection_from_raw_mask(mask, extract_geometry=extract_geometry)
                for mask in cable_masks
            ]
            combined_raw = np.zeros(detector_input.shape[:2], dtype=np.uint8)
            for mask in cable_masks:
                combined_raw = cv2.bitwise_or(combined_raw, mask)
            combined_detection = self._detection_from_raw_mask(combined_raw, extract_geometry=extract_geometry)
            endpoint_masks = [None for _index in range(cable_count)]
            endpoint_mask = None
            if scale < 0.999:
                detections = [resize_detection(detection, (original_h, original_w)) for detection in detections]
                combined_detection = resize_detection(combined_detection, (original_h, original_w))
            return detections, combined_detection, endpoint_mask, endpoint_masks

        if generic_cable_endpoint_label_mode(self.label_mode):
            channels = [0]
            thresholds = [self.threshold]
            endpoint_channels_by_cable = [None for _index in range(cable_count)]
            endpoint_channel_to_mask_index = {}
            if bool(include_endpoint_mask):
                for cable_index in range(cable_count):
                    endpoint_channel = self.endpoint_channel_for_cable(cable_index, cable_count)
                    endpoint_channels_by_cable[cable_index] = endpoint_channel
                    if endpoint_channel is None:
                        continue
                    endpoint_channel_to_mask_index[endpoint_channel] = len(channels)
                    channels.append(endpoint_channel)
                    thresholds.append(self.threshold if endpoint_threshold is None else float(endpoint_threshold))
            masks = self.segmenter.mask_channels(detector_input, channels, thresholds)
            cable_mask = masks[0]
            combined_detection = self._detection_from_raw_mask(cable_mask, extract_geometry=extract_geometry)
            detections = [combined_detection for _index in range(cable_count)]
            endpoint_masks = []
            endpoint_mask = None
            for endpoint_channel in endpoint_channels_by_cable:
                if endpoint_channel is None:
                    endpoint_masks.append(None)
                    continue
                mask = masks[endpoint_channel_to_mask_index[endpoint_channel]]
                endpoint_masks.append(mask)
                endpoint_mask = mask.copy() if endpoint_mask is None else cv2.bitwise_or(endpoint_mask, mask)
            if scale < 0.999:
                combined_detection = resize_detection(combined_detection, (original_h, original_w))
                detections = [combined_detection for _index in range(cable_count)]
                if endpoint_mask is not None:
                    endpoint_mask = resize_mask(endpoint_mask, (original_h, original_w))
                endpoint_masks = [
                    None if mask is None else resize_mask(mask, (original_h, original_w))
                    for mask in endpoint_masks
                ]
            return detections, combined_detection, endpoint_mask, endpoint_masks

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


def negative_prompt_maps_from_positive(prompt_maps):
    prompts = np.asarray(prompt_maps, dtype=np.float32)
    if prompts.ndim == 2:
        return np.zeros_like(prompts, dtype=np.float32)
    if prompts.ndim != 3:
        raise ValueError("Positive prompts must have shape HxW or NxHxW.")
    if prompts.shape[0] <= 1:
        return np.zeros_like(prompts, dtype=np.float32)
    negative = np.zeros_like(prompts, dtype=np.float32)
    for index in range(prompts.shape[0]):
        others = [prompts[other] for other in range(prompts.shape[0]) if other != index]
        negative[index] = np.max(np.stack(others, axis=0), axis=0)
    return np.ascontiguousarray(np.clip(negative, 0.0, 1.0), dtype=np.float32)


def endpoint_path_probability_maps(probability_maps, prompt_maps, max_size=320, path_radius_px=5):
    probability = np.asarray(probability_maps, dtype=np.float32)
    if probability.ndim == 2:
        probability = probability[:, :, None]
    if probability.ndim != 3:
        raise ValueError("Cable probability maps must have shape HxW or HxWxN.")
    prompts = np.asarray(prompt_maps, dtype=np.float32)
    if prompts.ndim == 2:
        prompts = prompts[None, :, :]
    if prompts.ndim != 3:
        raise ValueError("Endpoint prompts must have shape HxW or NxHxW.")

    height, width, channel_count = probability.shape
    prompt_count = prompts.shape[0]
    if prompt_count < channel_count:
        raise ValueError(f"Need at least {channel_count} endpoint prompts, got {prompt_count}.")

    max_size = max(64, int(max_size))
    scale = min(1.0, float(max_size) / float(max(height, width)))
    small_w = max(2, int(round(width * scale)))
    small_h = max(2, int(round(height * scale)))
    if scale < 0.999:
        support = cv2.resize(np.max(probability, axis=2), (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_prompts = np.stack(
            [
                cv2.resize(prompts[index], (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                for index in range(channel_count)
            ],
            axis=0,
        )
    else:
        support = np.max(probability, axis=2)
        small_prompts = prompts[:channel_count]

    support = np.ascontiguousarray(np.clip(support, 0.0, 1.0), dtype=np.float32)
    support_graph = cable_support_graph_mask(support)
    output = np.zeros((height, width, channel_count), dtype=np.float32)
    radius_small = max(1, int(round(float(path_radius_px) * scale)))
    radius_full = max(1, int(path_radius_px))
    for channel in range(channel_count):
        start_xy, end_xy = endpoint_pair_from_prompt(small_prompts[channel])
        path_xy = shortest_mask_probability_path(support, support_graph, start_xy, end_xy)
        small_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        if len(path_xy) >= 2:
            cv2.polylines(small_mask, [path_xy.astype(np.int32)], isClosed=False, color=255, thickness=2 * radius_small + 1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(small_mask, tuple(path_xy[0].astype(int)), radius_small, 255, -1, cv2.LINE_AA)
        if scale < 0.999:
            full_mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_LINEAR)
            full_mask = (full_mask > 0).astype(np.uint8) * 255
            if radius_full > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_full + 1, 2 * radius_full + 1))
                full_mask = cv2.dilate(full_mask, kernel, iterations=1)
        else:
            full_mask = small_mask
        output[:, :, channel] = (full_mask > 0).astype(np.float32)
    return np.ascontiguousarray(output, dtype=np.float32)


def cable_support_graph_mask(support):
    support = np.asarray(support, dtype=np.float32)
    if support.ndim != 2:
        raise ValueError("Cable support must be a single-channel probability image.")
    threshold = max(0.05, min(0.25, 0.35 * float(np.max(support))))
    mask = (support >= threshold).astype(np.uint8) * 255
    if not np.any(mask):
        raise ValueError("Cable support is empty; cannot build endpoint path graph.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if not np.any(mask):
        raise ValueError("Cable support graph mask is empty; cannot build endpoint path graph.")
    return mask


def endpoint_pair_from_prompt(prompt):
    prompt = np.asarray(prompt, dtype=np.float32)
    if prompt.ndim != 2:
        raise ValueError("Endpoint prompt must be HxW.")
    max_value = float(np.max(prompt)) if prompt.size else 0.0
    if max_value <= 1e-6:
        raise ValueError("Endpoint prompt is empty; cannot build endpoint path.")
    threshold = max(0.20, 0.35 * max_value)
    mask = (prompt >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        components.append((area, np.asarray(centroids[label], dtype=np.float32)))
    if len(components) >= 2:
        components.sort(key=lambda item: item[0], reverse=True)
        return components[0][1], components[1][1]

    coords_yx = np.argwhere(prompt > max(0.02, 0.05 * max_value))
    if len(coords_yx) < 2:
        raise ValueError("Endpoint prompt has fewer than two endpoint pixels.")
    return farthest_xy_pair(coords_yx)


def farthest_xy_pair(coords_yx):
    coords_yx = np.asarray(coords_yx, dtype=np.float32)
    if len(coords_yx) > 1200:
        step = int(np.ceil(len(coords_yx) / 1200.0))
        coords_yx = coords_yx[::step]
    coords_xy = coords_yx[:, ::-1]
    center = np.mean(coords_xy, axis=0)
    first = coords_xy[int(np.argmax(np.sum((coords_xy - center) ** 2, axis=1)))]
    second = coords_xy[int(np.argmax(np.sum((coords_xy - first) ** 2, axis=1)))]
    return first.astype(np.float32), second.astype(np.float32)


def shortest_mask_probability_path(support, graph_mask, start_xy, end_xy):
    support = np.asarray(support, dtype=np.float32)
    graph_mask = np.asarray(graph_mask, dtype=np.uint8)
    height, width = support.shape[:2]
    from cable_detection import build_skeleton_graph

    coords_yx, neighbors = build_skeleton_graph(graph_mask)
    if len(coords_yx) < 2:
        raise ValueError("Cable support graph has fewer than two nodes.")
    start_index = nearest_skeleton_node(coords_yx, start_xy)
    end_index = nearest_skeleton_node(coords_yx, end_xy)
    if start_index == end_index:
        y, x = coords_yx[start_index]
        return np.asarray([[x, y]], dtype=np.float32)

    support = np.clip(support, 0.0, 1.0)
    distances = np.full(len(coords_yx), np.inf, dtype=np.float64)
    previous = np.full(len(coords_yx), -1, dtype=np.int32)
    distances[start_index] = 0.0
    queue = [(0.0, int(start_index))]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > float(distances[node]):
            continue
        if node == end_index:
            break
        y, x = coords_yx[node]
        for neighbor in neighbors[node]:
            ny, nx = coords_yx[neighbor]
            step_cost = 1.41421356 if abs(int(nx) - int(x)) == 1 and abs(int(ny) - int(y)) == 1 else 1.0
            support_cost = 0.10 + 12.0 * (1.0 - float(support[ny, nx])) ** 2
            candidate = distance + step_cost * support_cost
            if candidate < float(distances[neighbor]):
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, int(neighbor)))

    if previous[end_index] < 0:
        raise ValueError("Could not connect endpoint pair through the cable probability field.")
    nodes = []
    node = int(end_index)
    while node >= 0:
        y, x = coords_yx[node]
        nodes.append((x, y))
        if node == start_index:
            break
        node = int(previous[node])
    nodes.reverse()
    return np.asarray(nodes, dtype=np.float32)


def nearest_skeleton_node(coords_yx, xy):
    coords_xy = np.asarray(coords_yx, dtype=np.float32)[:, ::-1]
    xy = np.asarray(xy, dtype=np.float32).reshape(2)
    return int(np.argmin(np.sum((coords_xy - xy) ** 2, axis=1)))


def clamp_xy_to_index(xy, width, height):
    xy = np.asarray(xy, dtype=np.float32).reshape(2)
    x = int(np.clip(round(float(xy[0])), 0, int(width) - 1))
    y = int(np.clip(round(float(xy[1])), 0, int(height) - 1))
    return x, y


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


def infer_input_channels_from_state_dict(state_dict):
    if isinstance(state_dict, dict):
        weight = state_dict.get("stem.0.block.0.weight")
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 2:
            return int(weight.shape[1])
    return 3


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
