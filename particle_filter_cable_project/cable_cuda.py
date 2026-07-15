import ctypes
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
import torch


@dataclass(frozen=True)
class CudaPointCloudView:
    pointer: int
    width: int
    height: int
    step_bytes: int
    confidence_pointer: int = 0
    confidence_step_bytes: int = 0
    owner: object = field(default=None, repr=False, compare=False)
    confidence_owner: object = field(default=None, repr=False, compare=False)


class CableCudaKernels:
    def __init__(self, source_path=None):
        if not torch.cuda.is_available():
            raise RuntimeError("Fused cable CUDA kernels require torch.cuda.is_available().")
        torch.cuda.init()
        self._context_anchor = torch.empty(1, dtype=torch.uint8, device="cuda")
        self.source_path = Path(source_path or Path(__file__).with_name("cable_cuda_kernels.cu")).resolve()
        if not self.source_path.exists():
            raise FileNotFoundError(f"CUDA kernel source not found: {self.source_path}")
        self.driver = ctypes.WinDLL("nvcuda.dll")
        self._configure_driver_api()
        self._check(self.driver.cuInit(0), "cuInit")
        self.module = ctypes.c_void_p()
        ptx = self._compile_ptx()
        self._ptx_buffer = ctypes.create_string_buffer(ptx)
        self._check(
            self.driver.cuModuleLoadData(ctypes.byref(self.module), ctypes.cast(self._ptx_buffer, ctypes.c_void_p)),
            "cuModuleLoadData",
        )
        self.functions = {}
        self.function_lock = threading.Lock()

    def _configure_driver_api(self):
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.driver.cuModuleLoadData.restype = ctypes.c_int
        self.driver.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.driver.cuModuleGetFunction.restype = ctypes.c_int
        self.driver.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.driver.cuLaunchKernel.restype = ctypes.c_int
        self.driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.driver.cuGetErrorString.restype = ctypes.c_int

    def _compile_ptx(self):
        source = self.source_path.read_bytes()
        major, minor = torch.cuda.get_device_capability()
        digest = hashlib.sha256(source + f"sm_{major}{minor}".encode("ascii")).hexdigest()[:16]
        cache_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "cable_pf_cuda"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ptx_path = cache_dir / f"cable_kernels_sm{major}{minor}_{digest}.ptx"
        if not ptx_path.exists():
            ptx_path.write_bytes(self._compile_with_nvrtc(source, major, minor))
        return ptx_path.read_bytes()

    def _compile_with_nvrtc(self, source, major, minor):
        cuda_path = Path(os.environ.get("CUDA_PATH", ""))
        bin_dir = cuda_path / "bin"
        candidates = sorted(bin_dir.glob("nvrtc64_*.dll")) if bin_dir.exists() else []
        candidates = [path for path in candidates if ".alt." not in path.name]
        if not candidates:
            raise RuntimeError("NVRTC was not found under CUDA_PATH; fused cable kernels cannot compile.")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(bin_dir))
        nvrtc = ctypes.WinDLL(str(candidates[-1]))
        program_type = ctypes.c_void_p
        nvrtc.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(program_type),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
        nvrtc.nvrtcCompileProgram.argtypes = [program_type, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
        nvrtc.nvrtcGetPTXSize.argtypes = [program_type, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
        nvrtc.nvrtcGetPTX.argtypes = [program_type, ctypes.c_char_p]
        nvrtc.nvrtcGetPTX.restype = ctypes.c_int
        nvrtc.nvrtcGetProgramLogSize.argtypes = [program_type, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
        nvrtc.nvrtcGetProgramLog.argtypes = [program_type, ctypes.c_char_p]
        nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
        nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(program_type)]
        nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int

        program = program_type()
        result = nvrtc.nvrtcCreateProgram(
            ctypes.byref(program),
            source,
            self.source_path.name.encode("utf-8"),
            0,
            None,
            None,
        )
        if result != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed with error {result}.")
        try:
            option_values = [
                f"--gpu-architecture=compute_{major}{minor}".encode("ascii"),
                b"--use_fast_math",
            ]
            options = (ctypes.c_char_p * len(option_values))(*option_values)
            result = nvrtc.nvrtcCompileProgram(program, len(option_values), options)
            if result != 0:
                log_size = ctypes.c_size_t()
                nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
                log = ctypes.create_string_buffer(max(1, log_size.value))
                nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError("NVRTC cable-kernel compilation failed:\n" + log.value.decode("utf-8", errors="replace"))
            ptx_size = ctypes.c_size_t()
            result = nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
            if result != 0:
                raise RuntimeError(f"nvrtcGetPTXSize failed with error {result}.")
            ptx = ctypes.create_string_buffer(ptx_size.value)
            result = nvrtc.nvrtcGetPTX(program, ptx)
            if result != 0:
                raise RuntimeError(f"nvrtcGetPTX failed with error {result}.")
            return ptx.raw
        finally:
            nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    def function(self, name):
        if name not in self.functions:
            with self.function_lock:
                if name not in self.functions:
                    function = ctypes.c_void_p()
                    self._check(
                        self.driver.cuModuleGetFunction(ctypes.byref(function), self.module, name.encode("ascii")),
                        f"cuModuleGetFunction({name})",
                    )
                    self.functions[name] = function
        return self.functions[name]

    def launch(self, name, count, arguments, stream, block_size=128):
        count = int(count)
        if count <= 0:
            return
        holders = []
        for kind, value in arguments:
            if kind == "tensor":
                if not value.is_cuda or not value.is_contiguous():
                    raise ValueError("CUDA kernel tensor arguments must be contiguous CUDA tensors.")
                holders.append(ctypes.c_uint64(value.data_ptr()))
            elif kind == "pointer":
                holders.append(ctypes.c_uint64(int(value)))
            elif kind == "int":
                holders.append(ctypes.c_int(int(value)))
            elif kind == "float":
                holders.append(ctypes.c_float(float(value)))
            else:
                raise ValueError(f"Unknown CUDA argument kind: {kind}")
        parameters = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
        )
        grid_size = (count + int(block_size) - 1) // int(block_size)
        self._check(
            self.driver.cuLaunchKernel(
                self.function(name),
                grid_size,
                1,
                1,
                int(block_size),
                1,
                1,
                0,
                ctypes.c_void_p(int(stream.cuda_stream)),
                parameters,
                None,
            ),
            f"cuLaunchKernel({name})",
        )

    def _check(self, result, operation):
        if int(result) == 0:
            return
        message = ctypes.c_char_p()
        self.driver.cuGetErrorString(int(result), ctypes.byref(message))
        detail = message.value.decode("utf-8", errors="replace") if message.value else f"CUDA error {result}"
        raise RuntimeError(f"{operation} failed: {detail}")


_KERNELS = None
_KERNEL_LOCK = threading.Lock()


def cable_cuda_kernels():
    global _KERNELS
    if _KERNELS is None:
        with _KERNEL_LOCK:
            if _KERNELS is None:
                _KERNELS = CableCudaKernels()
    return _KERNELS


def constrain_chains(chains, endpoints, segment_length_m, iterations, tolerance_m, stream):
    chains = chains.contiguous()
    endpoints = endpoints.contiguous()
    cable_cuda_kernels().launch(
        "constrain_chains_kernel",
        len(chains),
        [
            ("tensor", chains),
            ("tensor", endpoints),
            ("int", len(chains)),
            ("int", chains.shape[1]),
            ("float", segment_length_m),
            ("int", iterations),
            ("float", tolerance_m),
        ],
        stream,
    )
    return chains


def build_ransac_chains(anchors, endpoints, node_count, segment_length_m, iterations, tolerance_m, stream):
    anchors = anchors.contiguous()
    endpoints = endpoints.contiguous()
    chains = torch.empty(
        (anchors.shape[0], int(node_count), 3),
        dtype=torch.float32,
        device=anchors.device,
    )
    cable_cuda_kernels().launch(
        "build_ransac_chains_kernel",
        len(chains),
        [
            ("tensor", anchors),
            ("tensor", endpoints),
            ("tensor", chains),
            ("int", len(chains)),
            ("int", anchors.shape[1]),
            ("int", node_count),
            ("float", segment_length_m),
            ("int", iterations),
            ("float", tolerance_m),
        ],
        stream,
    )
    return chains


_SAMPLER_LOCAL = threading.local()


def sample_masked_points(
    point_cloud,
    mask,
    depth_min,
    depth_max,
    max_confidence,
    max_points,
    oversample=4,
):
    if not isinstance(point_cloud, CudaPointCloudView):
        raise TypeError("GPU point sampling requires CudaPointCloudView.")
    if point_cloud.pointer <= 0 or point_cloud.step_bytes <= 0:
        raise ValueError("CUDA point-cloud view has an invalid device pointer or row pitch.")
    mask_array = np.asarray(mask, dtype=np.uint8)
    if tuple(mask_array.shape) != (int(point_cloud.height), int(point_cloud.width)):
        raise ValueError(
            f"GPU mask shape {tuple(mask_array.shape)} does not match point cloud "
            f"{point_cloud.height}x{point_cloud.width}."
        )
    flat_indices = np.flatnonzero(mask_array.reshape(-1))
    if len(flat_indices) == 0:
        return torch.empty((0, 3), dtype=torch.float32, device="cuda")
    max_points = max(0, int(max_points))
    candidate_limit = len(flat_indices)
    if max_points > 0:
        candidate_limit = min(candidate_limit, max_points * max(1, int(oversample)))
    if candidate_limit < len(flat_indices):
        selection = np.linspace(0, len(flat_indices) - 1, candidate_limit, dtype=np.int64)
        flat_indices = flat_indices[selection]
    points = sample_indexed_points(
        point_cloud,
        flat_indices,
        depth_min=depth_min,
        depth_max=depth_max,
        max_confidence=max_confidence,
    )
    points = points[torch.isfinite(points).all(dim=1)]
    if max_points > 0 and len(points) > max_points:
        indices = torch.linspace(0, len(points) - 1, max_points, device=points.device).to(torch.int64)
        points = points.index_select(0, indices)
    return points.contiguous()


def sample_indexed_points(
    point_cloud,
    flat_indices,
    *,
    depth_min,
    depth_max,
    max_confidence,
):
    """Gather exact image pixels from a pitched ZED CUDA point cloud.

    Invalid/depth-rejected samples remain NaN so callers can preserve component
    boundaries while performing one CUDA gather for several regions.
    """

    if not isinstance(point_cloud, CudaPointCloudView):
        raise TypeError("GPU point sampling requires CudaPointCloudView.")
    if point_cloud.pointer <= 0 or point_cloud.step_bytes <= 0:
        raise ValueError("CUDA point-cloud view has an invalid device pointer or row pitch.")
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    if len(flat_indices) == 0:
        return torch.empty((0, 3), dtype=torch.float32, device="cuda")
    pixel_count = int(point_cloud.width) * int(point_cloud.height)
    if np.any(flat_indices < 0) or np.any(flat_indices >= pixel_count):
        raise ValueError("Pixel index lies outside the CUDA point-cloud view.")
    pixel_indices = torch.as_tensor(
        np.ascontiguousarray(flat_indices),
        dtype=torch.int64,
        device="cuda",
    )
    stream = getattr(_SAMPLER_LOCAL, "stream", None)
    if stream is None:
        stream = torch.cuda.Stream()
        _SAMPLER_LOCAL.stream = stream
    caller_stream = torch.cuda.current_stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        output = torch.empty((len(pixel_indices), 3), dtype=torch.float32, device=pixel_indices.device)
        use_confidence = max_confidence is not None
        if use_confidence and (
            point_cloud.confidence_pointer <= 0 or point_cloud.confidence_step_bytes <= 0
        ):
            raise ValueError("Confidence filtering requested without a CUDA confidence-map view.")
        cable_cuda_kernels().launch(
            "gather_indexed_points_kernel",
            len(pixel_indices),
            [
                ("pointer", point_cloud.pointer),
                ("int", point_cloud.step_bytes),
                ("tensor", pixel_indices),
                ("pointer", point_cloud.confidence_pointer),
                ("int", point_cloud.confidence_step_bytes),
                ("tensor", output),
                ("int", point_cloud.width),
                ("int", len(pixel_indices)),
                ("float", 0.0 if depth_min is None else depth_min),
                ("float", float("inf") if depth_max is None else depth_max),
                ("float", 0.0 if max_confidence is None else max_confidence),
                ("int", int(use_confidence)),
            ],
            stream,
            block_size=256,
        )
    caller_stream.wait_stream(stream)
    output.record_stream(caller_stream)
    return output
