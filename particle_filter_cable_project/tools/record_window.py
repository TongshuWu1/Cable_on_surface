import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np


SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0
PHONE_VIDEO_WIDTH = 1920
PHONE_VIDEO_HEIGHT = 1080


class BitmapInfoHeader(ctypes.Structure):
    _fields_ = (
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    )


class BitmapInfo(ctypes.Structure):
    _fields_ = (
        ("bmiHeader", BitmapInfoHeader),
        ("bmiColors", wintypes.DWORD * 3),
    )


user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
user32.FindWindowW.restype = wintypes.HWND
user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.GetWindowRect.restype = wintypes.BOOL
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL
user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
user32.GetSystemMetrics.restype = ctypes.c_int
user32.GetDC.argtypes = (wintypes.HWND,)
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
user32.ReleaseDC.restype = ctypes.c_int

gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = (wintypes.HDC, ctypes.c_int, ctypes.c_int)
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HGDIOBJ)
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.BitBlt.argtypes = (
    wintypes.HDC,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HDC,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.DWORD,
)
gdi32.BitBlt.restype = wintypes.BOOL
gdi32.GetDIBits.argtypes = (
    wintypes.HDC,
    wintypes.HBITMAP,
    wintypes.UINT,
    wintypes.UINT,
    wintypes.LPVOID,
    ctypes.POINTER(BitmapInfo),
    wintypes.UINT,
)
gdi32.GetDIBits.restype = ctypes.c_int
gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = (wintypes.HDC,)
gdi32.DeleteDC.restype = wintypes.BOOL


def parse_args():
    parser = argparse.ArgumentParser(description="Record the full Windows desktop or one visible window.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--desktop", action="store_true", help="Record the entire virtual desktop")
    source.add_argument("--title", help="Record the window with this exact title")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--wait", type=float, default=30.0, help="Seconds to wait for the window")
    parser.add_argument(
        "--codec",
        default="h264",
        help="Output codec: h264 (phone-compatible default) or an OpenCV FourCC such as mp4v",
    )
    return parser.parse_args()


def wait_for_window(title, timeout_s):
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        window = user32.FindWindowW(None, str(title))
        if window:
            return window
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Window did not appear within {timeout_s:.1f}s: {title!r}")
        time.sleep(0.1)


def window_rectangle(window):
    rectangle = wintypes.RECT()
    if not user32.GetWindowRect(window, ctypes.byref(rectangle)):
        raise ctypes.WinError(ctypes.get_last_error())
    width = int(rectangle.right - rectangle.left)
    height = int(rectangle.bottom - rectangle.top)
    if width < 2 or height < 2:
        raise RuntimeError(f"Window has invalid dimensions: {width}x{height}")
    return int(rectangle.left), int(rectangle.top), width, height


def virtual_desktop_rectangle():
    # SM_X/Y/CX/CYVIRTUALSCREEN include every attached display and preserve
    # negative monitor coordinates when the primary display is not top-left.
    rectangle = tuple(user32.GetSystemMetrics(index) for index in (76, 77, 78, 79))
    left, top, width, height = rectangle
    if width < 2 or height < 2:
        raise RuntimeError(f"Virtual desktop has invalid dimensions: {width}x{height}")
    return left, top, width, height


class DesktopRegionCapture:
    def __init__(self, left, top, width, height):
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)
        self.screen_dc = user32.GetDC(None)
        if not self.screen_dc:
            raise ctypes.WinError(ctypes.get_last_error())
        self.memory_dc = gdi32.CreateCompatibleDC(self.screen_dc)
        self.bitmap = gdi32.CreateCompatibleBitmap(self.screen_dc, self.width, self.height)
        if not self.memory_dc or not self.bitmap:
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        self.previous_object = gdi32.SelectObject(self.memory_dc, self.bitmap)
        self.buffer = ctypes.create_string_buffer(self.width * self.height * 4)
        self.bitmap_info = BitmapInfo()
        self.bitmap_info.bmiHeader = BitmapInfoHeader(
            biSize=ctypes.sizeof(BitmapInfoHeader),
            biWidth=self.width,
            biHeight=-self.height,
            biPlanes=1,
            biBitCount=32,
            biCompression=BI_RGB,
            biSizeImage=self.width * self.height * 4,
        )

    def frame(self):
        copied = gdi32.BitBlt(
            self.memory_dc,
            0,
            0,
            self.width,
            self.height,
            self.screen_dc,
            self.left,
            self.top,
            SRCCOPY | CAPTUREBLT,
        )
        if not copied:
            raise ctypes.WinError(ctypes.get_last_error())
        rows = gdi32.GetDIBits(
            self.memory_dc,
            self.bitmap,
            0,
            self.height,
            self.buffer,
            ctypes.byref(self.bitmap_info),
            DIB_RGB_COLORS,
        )
        if rows != self.height:
            raise RuntimeError(f"GDI returned {rows}/{self.height} scan lines.")
        bgra = np.frombuffer(self.buffer, dtype=np.uint8).reshape(self.height, self.width, 4)
        return np.ascontiguousarray(bgra[:, :, :3])

    def close(self):
        if getattr(self, "memory_dc", None) and getattr(self, "previous_object", None):
            gdi32.SelectObject(self.memory_dc, self.previous_object)
            self.previous_object = None
        if getattr(self, "bitmap", None):
            gdi32.DeleteObject(self.bitmap)
            self.bitmap = None
        if getattr(self, "memory_dc", None):
            gdi32.DeleteDC(self.memory_dc)
            self.memory_dc = None
        if getattr(self, "screen_dc", None):
            user32.ReleaseDC(None, self.screen_dc)
            self.screen_dc = None


def transcode_h264(source_path, output_path):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "H.264 recording requires imageio-ffmpeg. Install project requirements first."
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    video_filter = (
        f"scale={PHONE_VIDEO_WIDTH}:{PHONE_VIDEO_HEIGHT}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={PHONE_VIDEO_WIDTH}:{PHONE_VIDEO_HEIGHT}:"
        "(ow-iw)/2:(oh-ih)/2:black,format=yuv420p"
    )
    command = (
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg H.264 encoding failed:\n{completed.stderr.strip()}")


def record_rectangle(rectangle, output_path, duration_s, fps, codec, is_available=None):
    left, top, width, height = rectangle
    output_width = width - (width % 2)
    output_height = height - (height % 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_h264 = str(codec).lower() in {"h264", "avc", "avc1"}
    temporary_path = None
    writer_path = output_path
    writer_codec = str(codec)
    if use_h264:
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("Phone-compatible H.264 output must use an .mp4 filename.")
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}-",
            suffix=".mp4",
            dir=output_path.parent,
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
        writer_path = temporary_path
        writer_codec = "mp4v"
    capture = DesktopRegionCapture(left, top, width, height)
    writer = cv2.VideoWriter(
        str(writer_path),
        cv2.VideoWriter_fourcc(*writer_codec),
        float(fps),
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"OpenCV could not open video output {writer_path} with codec {writer_codec!r}."
        )
    interval = 1.0 / max(float(fps), 1e-3)
    start = time.monotonic()
    next_frame = start
    count = 0
    try:
        while time.monotonic() - start < max(float(duration_s), 0.0):
            if is_available is not None and not is_available():
                raise RuntimeError("Target window closed during recording.")
            now = time.monotonic()
            if now < next_frame:
                time.sleep(next_frame - now)
            frame = capture.frame()[:output_height, :output_width]
            writer.write(frame)
            count += 1
            next_frame = start + count * interval
    finally:
        writer.release()
        capture.close()
    elapsed = time.monotonic() - start
    if use_h264:
        try:
            transcode_h264(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        output_width = PHONE_VIDEO_WIDTH
        output_height = PHONE_VIDEO_HEIGHT
    return count, output_width, output_height, elapsed


def record_window(window, output_path, duration_s, fps, codec):
    return record_rectangle(
        window_rectangle(window),
        output_path,
        duration_s,
        fps,
        codec,
        is_available=lambda: bool(user32.IsWindow(window)),
    )


def record_desktop(output_path, duration_s, fps, codec):
    return record_rectangle(
        virtual_desktop_rectangle(),
        output_path,
        duration_s,
        fps,
        codec,
    )


def main():
    args = parse_args()
    if args.desktop:
        count, width, height, elapsed = record_desktop(
            args.output,
            args.duration,
            args.fps,
            args.codec,
        )
        source = "virtual desktop"
    else:
        window = wait_for_window(args.title, args.wait)
        count, width, height, elapsed = record_window(
            window,
            args.output,
            args.duration,
            args.fps,
            args.codec,
        )
        source = f"window {args.title!r}"
    print(
        f"Recorded {source}: {count} frames at {width}x{height} over {elapsed:.2f}s to "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
