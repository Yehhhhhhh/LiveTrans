"""Resolve a native window handle (from Chrome's window-capture track label) to its title and app name."""

from __future__ import annotations

import base64
import struct
import sys
import zlib
from pathlib import Path


def window_info(hwnd: int) -> dict | None:
    """Return {title, app, exe} for a top-level window, or None if it doesn't exist (or not on Windows)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    h = wintypes.HWND(hwnd)
    if not user32.IsWindow(h):
        return None
    n = user32.GetWindowTextLengthW(h)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(h, buf, n + 1)
    title = buf.value

    exe_path = ""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
    proc = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if proc:
        try:
            size = wintypes.DWORD(1024)
            path = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(proc, 0, path, ctypes.byref(size)):
                exe_path = path.value
        finally:
            kernel32.CloseHandle(proc)

    exe = Path(exe_path).name if exe_path else ""
    app = _file_description(exe_path) if exe_path else ""
    info = {"title": title, "app": app or Path(exe).stem, "exe": exe}
    try:
        png = _window_icon_png(hwnd, exe_path)
    except OSError:
        png = None
    if png:
        info["icon"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return info


def _window_icon_png(hwnd: int, exe_path: str) -> bytes | None:
    """The window's own icon (as on the taskbar), else the exe's first icon, as PNG."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    user32.SendMessageTimeoutW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t),
    ]
    user32.GetClassLongPtrW.restype = ctypes.c_size_t
    user32.GetClassLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.DestroyIcon.argtypes = [wintypes.HICON]
    shell32.ExtractIconExW.argtypes = [
        wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.HICON), ctypes.POINTER(wintypes.HICON), wintypes.UINT
    ]

    WM_GETICON, SMTO_ABORTIFHUNG = 0x7F, 0x2
    # Big first: it stays sharp at 16 CSS px on high-DPI screens.
    for kind in (1, 2, 0):  # ICON_BIG, ICON_SMALL2, ICON_SMALL
        res = ctypes.c_size_t()
        if user32.SendMessageTimeoutW(hwnd, WM_GETICON, kind, 0, SMTO_ABORTIFHUNG, 200, ctypes.byref(res)) and res.value:
            png = _icon_to_png(res.value)  # owned by the window: not destroyed
            if png:
                return png
    for index in (-14, -34):  # GCLP_HICON, GCLP_HICONSM
        icon = user32.GetClassLongPtrW(hwnd, index)
        if icon:
            png = _icon_to_png(icon)
            if png:
                return png
    if exe_path:
        large, small = wintypes.HICON(), wintypes.HICON()
        if shell32.ExtractIconExW(exe_path, 0, ctypes.byref(large), ctypes.byref(small), 1):
            try:
                for icon in (large.value, small.value):
                    if icon:
                        png = _icon_to_png(icon)
                        if png:
                            return png
            finally:
                for icon in (large.value, small.value):
                    if icon:
                        user32.DestroyIcon(icon)
    return None


def _icon_to_png(hicon: int) -> bytes | None:
    import ctypes
    from ctypes import wintypes

    class ICONINFO(ctypes.Structure):
        _fields_ = [("fIcon", wintypes.BOOL), ("xHotspot", wintypes.DWORD), ("yHotspot", wintypes.DWORD),
                    ("hbmMask", wintypes.HBITMAP), ("hbmColor", wintypes.HBITMAP)]

    class BITMAP(ctypes.Structure):
        _fields_ = [("bmType", wintypes.LONG), ("bmWidth", wintypes.LONG), ("bmHeight", wintypes.LONG),
                    ("bmWidthBytes", wintypes.LONG), ("bmPlanes", wintypes.WORD), ("bmBitsPixel", wintypes.WORD),
                    ("bmBits", ctypes.c_void_p)]

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

    user32 = ctypes.WinDLL("user32")
    gdi32 = ctypes.WinDLL("gdi32")
    user32.GetIconInfo.argtypes = [wintypes.HICON, ctypes.POINTER(ICONINFO)]
    user32.GetDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    gdi32.GetObjectW.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p]
    gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]

    info = ICONINFO()
    if not user32.GetIconInfo(hicon, ctypes.byref(info)):
        return None
    try:
        if not info.hbmColor:
            return None  # monochrome icon: not worth handling
        bm = BITMAP()
        if not gdi32.GetObjectW(info.hbmColor, ctypes.sizeof(bm), ctypes.byref(bm)):
            return None
        w, h = bm.bmWidth, bm.bmHeight
        if not (0 < w <= 256 and 0 < h <= 256):
            return None

        def bits(hbm) -> bytes | None:
            bi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0)  # top-down BGRA
            buf = ctypes.create_string_buffer(w * h * 4)
            hdc = user32.GetDC(None)
            try:
                ok = gdi32.GetDIBits(hdc, hbm, 0, h, buf, ctypes.byref(bi), 0)
            finally:
                user32.ReleaseDC(None, hdc)
            return buf.raw if ok else None

        bgra = bits(info.hbmColor)
        if bgra is None:
            return None
        px = bytearray(bgra)
        if not any(px[3::4]):
            # No alpha channel: transparency comes from the AND mask (white = transparent).
            mask = bits(info.hbmMask)
            for i in range(0, len(px), 4):
                px[i + 3] = 0 if mask and mask[i] else 255
        px[0::4], px[2::4] = px[2::4], px[0::4]  # BGRA -> RGBA
        return _png(w, h, bytes(px))
    finally:
        for hbm in (info.hbmColor, info.hbmMask):
            if hbm:
                gdi32.DeleteObject(hbm)


def _png(w: int, h: int, rgba: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    rows = b"".join(b"\x00" + rgba[y * w * 4:(y + 1) * w * 4] for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def _file_description(path: str) -> str:
    """The exe's FileDescription version string, e.g. 'Microsoft Teams'."""
    import ctypes
    from ctypes import wintypes

    try:
        version = ctypes.WinDLL("version")
        version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        version.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        version.VerQueryValueW.argtypes = [
            ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)
        ]
        size = version.GetFileVersionInfoSizeW(path, None)
        if not size:
            return ""
        data = ctypes.create_string_buffer(size)
        if not version.GetFileVersionInfoW(path, 0, size, data):
            return ""
        ptr = ctypes.c_void_p()
        length = wintypes.UINT()
        langs = []
        if version.VerQueryValueW(data, "\\VarFileInfo\\Translation", ctypes.byref(ptr), ctypes.byref(length)):
            pairs = (wintypes.WORD * (length.value // 2)).from_address(ptr.value)
            langs = [f"{pairs[i]:04x}{pairs[i + 1]:04x}" for i in range(0, len(pairs) - 1, 2)]
        for lang in langs + ["040904b0", "040904e4"]:
            key = f"\\StringFileInfo\\{lang}\\FileDescription"
            if version.VerQueryValueW(data, key, ctypes.byref(ptr), ctypes.byref(length)) and length.value:
                text = ctypes.wstring_at(ptr.value, length.value).rstrip("\0").strip()
                if text:
                    return text
    except OSError:
        pass
    return ""
