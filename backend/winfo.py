"""Resolve a native window handle (from Chrome's window-capture track label) to its title and app name."""

from __future__ import annotations

import sys
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
    return {"title": title, "app": app or Path(exe).stem, "exe": exe}


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
