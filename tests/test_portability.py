import subprocess
from unittest.mock import patch
import pytest
from src import posix_typer as backend


def test_wayland_unicode_is_stdin_and_newline_never_sends(monkeypatch):
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(backend.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []
    monkeypatch.setattr(backend, "_run", lambda args, text=None: calls.append((args, text)))
    backend.type_text('Привет, мир!\n--опасный-параметр')
    assert calls[0] == (["/usr/bin/wtype", "-"], "Привет, мир!")
    assert calls[1][0] == ["/usr/bin/wtype", "-M", "shift", "-P", "Return", "-p", "Return", "-m", "shift"]
    assert calls[2][1] == "--опасный-параметр"
    assert all("xdotool" not in str(args) for args, _ in calls)


def test_x11_controls_and_unicode(monkeypatch):
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setattr(backend.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []
    monkeypatch.setattr(backend, "_run", lambda args, text=None: calls.append((args, text)))
    backend.type_text("Съешь ещё этих булок")
    backend.key("BackSpace", count=3)
    backend.key("Return", ctrl=True)
    assert calls[0] == (["/usr/bin/xdotool", "type", "--clearmodifiers", "--file", "-"], "Съешь ещё этих булок")
    assert calls[1][0][-3:] == ["--repeat", "3", "BackSpace"]
    assert calls[2][0][-1] == "ctrl+Return"


def test_missing_wayland_backend_never_falls_back_to_x11(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(backend.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="wtype"):
        backend._tool()


def test_macos_unicode_and_modifiers_use_quartz_not_shell(monkeypatch):
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    events = []
    monkeypatch.setattr(backend, "_mac_key", lambda code, flags=0, text=None: events.append((code, flags, text)))
    backend.type_text("Я😊\nДа")
    backend.key("Return", ctrl=True)
    assert [item[2] for item in events[:2]] == ["Я", "😊"]
    assert events[2] == (36, 1 << 17, None)
    assert events[-1] == (36, 1 << 18, None)


def test_failed_input_does_not_quote_dictated_text_in_error(monkeypatch):
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, a[0], stderr="private words")))
    with pytest.raises(RuntimeError) as error:
        backend._run(["wtype", "-"], "private words")
    assert "private words" not in str(error.value)
