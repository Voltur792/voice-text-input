"""Unicode input for X11, Wayland virtual-keyboard and macOS Accessibility.

Never use an X11 tool in a Wayland session: it cannot reach native clients.
Text is supplied through stdin, never interpreted as a command or an option.
"""
import ctypes
import os
import shutil
import subprocess
import sys


def wayland():
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland")


def _run(args, text=None):
    try:
        subprocess.run(args, input=text, text=True, encoding="utf-8", check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        # stderr can contain the dictated text. Only expose setup guidance.
        raise RuntimeError("Не удалось напечатать текст. Проверьте доступ к вводу; в Wayland нужен wtype и поддержка virtual-keyboard композитором.") from exc


def _tool():
    name = "wtype" if wayland() else "xdotool"
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"Для набора текста в {'Wayland' if wayland() else 'X11'} установите {name}. "
                           "GNOME Wayland может не поддерживать virtual-keyboard; используйте поддерживающий его сеанс или X11.")
    return executable


_quartz = None


def _mac():
    global _quartz
    if _quartz is None:
        q = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        q.AXIsProcessTrusted.restype = ctypes.c_bool
        q.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_ushort, ctypes.c_bool]
        q.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        q.CGEventKeyboardSetUnicodeString.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ushort)]
        q.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        q.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        q.CFRelease.argtypes = [ctypes.c_void_p]
        _quartz = q
    if not _quartz.AXIsProcessTrusted():
        raise RuntimeError("Разрешите управление для Astra/Python в macOS: Системные настройки → Конфиденциальность и безопасность → Универсальный доступ.")
    return _quartz


def _mac_key(code, flags=0, text=None):
    q = _mac()
    for down in (True, False):
        event = q.CGEventCreateKeyboardEvent(None, code, down)
        if not event:
            raise RuntimeError("macOS не создала событие клавиатуры")
        try:
            q.CGEventSetFlags(event, flags)
            if text:
                raw = text.encode("utf-16-le")
                units = (ctypes.c_ushort * (len(raw) // 2)).from_buffer_copy(raw)
                q.CGEventKeyboardSetUnicodeString(event, len(units), units)
            q.CGEventPost(0, event)
        finally:
            q.CFRelease(event)


def key(name, ctrl=False, shift=False, count=1):
    count = min(max(0, count), 10_000)
    if not count:
        return
    if sys.platform == "darwin":
        for _ in range(count):
            _mac_key({"Return": 36, "BackSpace": 51}[name], (1 << 18 if ctrl else 0) | (1 << 17 if shift else 0))
        return
    tool = _tool()
    if wayland():
        args = [tool]
        for flag, enabled in (("ctrl", ctrl), ("shift", shift)):
            if enabled:
                args += ["-M", flag]
        for _ in range(count):
            args += ["-P", name, "-p", name]
        for flag, enabled in (("shift", shift), ("ctrl", ctrl)):
            if enabled:
                args += ["-m", flag]
        _run(args)
    else:
        combination = ("ctrl+" if ctrl else "") + ("shift+" if shift else "") + name
        _run([tool, "key", "--clearmodifiers", "--repeat", str(count), combination])


def type_text(text):
    # Keep the Windows convention: newline inserts Shift+Enter, never sends.
    parts = text.replace("\r", "").split("\n")
    for index, part in enumerate(parts):
        if index:
            key("Return", shift=True)
        if not part:
            continue
        if sys.platform == "darwin":
            for char in part:
                _mac_key(0, text=char)
        elif wayland():
            _run([_tool(), "-"], part)
        else:
            _run([_tool(), "type", "--clearmodifiers", "--file", "-"], part)
