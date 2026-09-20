"""Self-healing of the plugin's Python environment.

Astra installs the plugin's dependencies itself from requirements.lock, but
that install is one flat all-or-nothing set: when any pin cannot be installed
(no wheel for the user's Python, an interrupted download, a corporate proxy),
NOTHING ends up installed — not even astra_plugin_sdk. The plugin then dies at
import time ("ModuleNotFoundError: No module named 'astra_plugin_sdk'") before
any of its own code runs, and Astra restarts it into the same traceback.

Two layers fix that:

* ensure_runtime() is called from src/plugin.py BEFORE the SDK import. It
  installs the core set into the interpreter that runs us (lock → loose
  requirements.txt → explicit minimal pins, plain pip then --user) and only
  then lets the import happen. When nothing can be installed it returns False
  and the entry point prints one actionable line instead of a traceback.
* ensure_whisper() installs faster-whisper on demand. It is deliberately not
  in the core set — it is the heavy, wheel-fragile part of the dependency
  tree, and dictation works (Vosk only) without it.
"""

import importlib
import logging
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger("voice-text-input.deps")

#: Importable names the plugin needs before it can even talk to Astra.
RUNTIME_MODULES = ("numpy", "sounddevice")
#: The Astra SDK on top of them — only src/plugin.py needs it (the standalone
#: app shares this module and must not try to install the SDK).
SDK_MODULES = ("astra_plugin_sdk",)
#: Needed to dictate at all: vosk is the always-on listener, websockets
#: powers the Yandex engine.
CORE_MODULES = ("vosk", "websockets")
#: Optional quality engine, installed on demand (heavy ctranslate2 chain).
WHISPER_MODULE = "faster_whisper"

_ROOT = Path(__file__).resolve().parent.parent
_LOCK_FILE = _ROOT / "requirements.lock"
_REQUIREMENTS_FILE = _ROOT / "requirements.txt"

# Only used when neither requirements.lock nor requirements.txt is available.
_MINIMAL_PKGS = ("astra-plugin-sdk", "grpcio", "protobuf", "numpy",
                 "sounddevice", "vosk", "websockets", "requests")
_WHISPER_PKGS = ("faster-whisper==1.2.1",)

_mutex = threading.Lock()
#: Set of module-group keys whose install already failed in this process:
#: retrying every dictation segment would spam the log and stall the worker.
_failed: set[str] = set()
#: Set of module-group keys already verified present.
_ready: set[str] = set()


def missing(modules: tuple[str, ...]) -> list[str]:
    """Names of `modules` that fail to import right now."""
    gone = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            gone.append(name)
    return gone


def _pip(args: list[str]) -> bool:
    """Run one pip install; True when it exited 0."""
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
           "--quiet", *args]
    log.warning("installing plugin dependencies: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=1800,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:
        log.warning("pip install failed (%s): %s", " ".join(args), exc)
        return False


def _install_core() -> bool:
    """Try every source of the core dependency set, plain then --user.

    Every source goes through `-r`: a bare path argument to `pip install` is
    an artifact/directory, not a requirements file — passing the lock that way
    fails with "No matching distribution found: requirements.lock", which is
    exactly how a self-heal meant to save a broken install ended up doing
    nothing at all. The lock is tried first (exact versions, reproducible),
    then the loose requirements — a lock pin that has no wheel for this
    interpreter is worth failing on, because pip can then resolve working
    versions for it.
    """
    sources: list[list[str]] = []
    if _LOCK_FILE.exists():
        sources.append(["-r", str(_LOCK_FILE)])
    if _REQUIREMENTS_FILE.exists():
        sources.append(["-r", str(_REQUIREMENTS_FILE)])
    sources.append(list(_MINIMAL_PKGS))
    for source in sources:
        if _pip(source):
            return True
        if _pip(["--user", *source]):
            return True
    return False


def _frozen() -> bool:
    """Are we inside a PyInstaller bundle (the standalone VoiceTyper app)?

    Then sys.executable IS the .exe: `sys.executable -m pip install …` would
    relaunch the application with pip arguments instead of installing anything.
    A frozen build carries its dependencies in the bundle — nothing to heal.
    """
    return bool(getattr(sys, "frozen", False))


def _ensure(modules: tuple[str, ...], key: str, announce: bool = False) -> bool:
    """Import `modules`, installing what is missing once per process."""
    with _mutex:
        gone = missing(modules)
        if not gone:
            _ready.add(key)
            return True
        if key in _failed:
            return False
        if _frozen():
            _failed.add(key)
            log.error("missing modules %s inside a frozen build — they must "
                      "be collected by the .spec, installing at runtime is "
                      "not possible here", ", ".join(gone))
            return False
        if announce:
            # Astra waits 20 s for the plugin's first stdout line, and a first
            # install can take minutes — say something before going quiet.
            print(f"voice-text-input: installing missing dependencies "
                  f"({', '.join(gone)}) — one-time, may take a few minutes…",
                  flush=True)
        log.warning("missing modules %s — installing them into %s (one-time)",
                    ", ".join(gone), sys.executable)
        if key == "whisper":
            ok = _pip(list(_WHISPER_PKGS)) or _pip(["--user", *_WHISPER_PKGS])
        else:
            ok = _install_core()
        if not ok:
            _failed.add(key)
            log.error("could not install %s — run this to fix it manually: "
                      '"%s" -m pip install -r "%s"',
                      ", ".join(gone), sys.executable, _REQUIREMENTS_FILE)
            return False
        still = missing(modules)
        if still:
            _failed.add(key)
            log.error("pip reported success but %s still cannot be imported "
                      "(site-packages of %s may be unusable)", ", ".join(still), sys.executable)
            return False
        log.info("dependency install finished: %s OK", ", ".join(gone))
        _ready.add(key)
        return True


def ensure_runtime() -> bool:
    """Make the SDK importable (call from src/plugin.py before importing it)."""
    return _ensure(SDK_MODULES + RUNTIME_MODULES, "runtime", announce=True)


def ensure_core_deps() -> bool:
    """Make vosk/websockets importable — dictation cannot work without them.

    Deliberately does NOT require the SDK: the standalone app shares this
    module and runs without astra_plugin_sdk.
    """
    if not _ensure(RUNTIME_MODULES, "runtime"):
        return False
    return _ensure(CORE_MODULES, "core")


def ensure_whisper() -> bool:
    """Install faster-whisper on demand. False = unavailable (Vosk-only)."""
    return _ensure((WHISPER_MODULE,), "whisper")


def whisper_installed() -> bool:
    return WHISPER_MODULE not in missing((WHISPER_MODULE,))


def missing_core() -> list[str]:
    """Names of core modules that fail to import right now."""
    return missing(CORE_MODULES)
