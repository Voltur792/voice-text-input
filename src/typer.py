"""Typing into whatever window has keyboard focus — Windows SendInput.

Uses KEYEVENTF_UNICODE, which injects characters by code point and therefore
works regardless of the active keyboard layout (Russian text lands correctly
even with an English layout selected). No third-party dependency: ctypes only.

Everything is a no-op with a logged warning on non-Windows platforms, so the
test suite and a Linux Astra can import this module freely.
"""

import logging
import sys

log = logging.getLogger("voice-text-input.typer")

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    VK_BACK = 0x08
    VK_RETURN = 0x0D
    VK_SHIFT = 0x10
    VK_CONTROL = 0x11

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = (
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        )

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        )

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = (
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        )

    class _INPUTUNION(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT))

    class INPUT(ctypes.Structure):
        _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))

    _user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    _user32.SendInput.restype = wintypes.UINT

    def _kb_event(code: int, flags: int) -> INPUT:
        item = INPUT()
        item.type = INPUT_KEYBOARD
        item.union.ki = KEYBDINPUT(0, code & 0xFFFF, flags, 0, 0)
        return item

    def _vk_event(vk: int, up: bool) -> INPUT:
        flags = KEYEVENTF_KEYUP if up else 0
        item = INPUT()
        item.type = INPUT_KEYBOARD
        item.union.ki = KEYBDINPUT(vk, 0, flags, 0, 0)
        return item

    def _send(events: list[INPUT]) -> None:
        if not events:
            return
        arr = (INPUT * len(events))(*events)
        sent = _user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
        if sent != len(events):
            # BlockedInput usually means an elevated window has focus (UIPI).
            log.warning("SendInput delivered %d/%d events (error=%s)",
                        sent, len(events), ctypes.get_last_error())

    def _type_unicode(text: str) -> None:
        events: list[INPUT] = []
        for ch in text:
            if ch == "\n":
                # Shift+Enter: a line break that does NOT send the message in
                # Telegram/VK messengers (a bare Enter would send it).
                events.append(_vk_event(VK_SHIFT, False))
                events.append(_vk_event(VK_RETURN, False))
                events.append(_vk_event(VK_RETURN, True))
                events.append(_vk_event(VK_SHIFT, True))
                continue
            if ch == "\r":
                continue
            code = ord(ch)
            if code > 0xFFFF:  # astral plane → UTF-16 surrogates
                code -= 0x10000
                units = (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF))
            else:
                units = (code,)
            for u in units:
                events.append(_kb_event(u, KEYEVENTF_UNICODE))
                events.append(_kb_event(u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        # Batches keep a single SendInput call small enough that apps with a
        # sluggish message loop do not drop input.
        for i in range(0, len(events), 32):
            _send(events[i:i + 32])


def type_text(text: str) -> None:
    """Type `text` into the focused window, character by character."""
    if not IS_WINDOWS:
        log.warning("typing requested on non-Windows platform: ignored")
        return
    if text:
        _type_unicode(text)


def press_enter(ctrl: bool = False) -> None:
    """Press Enter (or Ctrl+Enter) in the focused window."""
    if not IS_WINDOWS:
        return
    events: list[INPUT] = []
    if ctrl:
        events.append(_vk_event(VK_CONTROL, False))
    events.append(_vk_event(VK_RETURN, False))
    events.append(_vk_event(VK_RETURN, True))
    if ctrl:
        events.append(_vk_event(VK_CONTROL, True))
    _send(events)


def press_shift_enter() -> None:
    """Shift+Enter: newline in the focused window without sending (messengers)."""
    if not IS_WINDOWS:
        return
    _send([
        _vk_event(VK_SHIFT, False),
        _vk_event(VK_RETURN, False),
        _vk_event(VK_RETURN, True),
        _vk_event(VK_SHIFT, True),
    ])


def backspace(count: int) -> None:
    """Send `count` Backspace presses (erases `count` characters backwards)."""
    if not IS_WINDOWS or count <= 0:
        return
    events: list[INPUT] = []
    for _ in range(min(count, 10_000)):
        events.append(_vk_event(VK_BACK, False))
        events.append(_vk_event(VK_BACK, True))
    for i in range(0, len(events), 64):
        _send(events[i:i + 64])
