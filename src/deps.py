"""One-time self-healing of the plugin's Python environment.

Astra runs the plugin with its `python` runtime, but the dependency install
can end up partial (interrupted download, heavy ML wheels): the plugin then
starts without vosk/faster-whisper and dictation cannot work at all. When a
core import fails, install the pinned set from requirements.lock (it sits
next to the sources) into the interpreter that runs us — once per process —
and log it clearly. Everything is already declared in the manifest; this
only finishes an install that broke halfway.
"""

import logging
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger("voice-text-input.deps")

# Modules dictation cannot work without (vosk is the always-on listener,
# websockets powers the yandex engine).
CORE_MODULES = ("vosk", "websockets")

_LOCK_FILE = Path(__file__).resolve().parent.parent / "requirements.lock"
# Only used when requirements.lock is not shipped next to the sources.
_FALLBACK_PKGS = ("vosk==0.3.45", "websockets==17.1", "faster-whisper==1.2.1")

_mutex = threading.Lock()
_fixed = False


def missing_core() -> list[str]:
    """Names of core modules that fail to import right now."""
    gone = []
    for name in CORE_MODULES:
        try:
            __import__(name)
        except ImportError:
            gone.append(name)
    return gone


def ensure_core_deps() -> bool:
    """Make sure core modules import, installing them once if needed.

    Fast path (everything present) is two imports. The install path blocks
    the caller for minutes — call it from a worker thread, which the
    dictation orchestrator and stt_load both are.
    """
    global _fixed
    with _mutex:
        if _fixed:
            return True
        gone = missing_core()
        if not gone:
            _fixed = True
            return True
        if _LOCK_FILE.exists():
            cmd = [sys.executable, "-m", "pip", "install",
                   "--disable-pip-version-check", "--quiet", str(_LOCK_FILE)]
        else:
            cmd = [sys.executable, "-m", "pip", "install",
                   "--disable-pip-version-check", "--quiet", *_FALLBACK_PKGS]
        log.warning("core modules missing (%s) — finishing the dependency "
                    "install, one-time, may take a few minutes: %s",
                    ", ".join(gone), " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, timeout=1800,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            log.error("dependency install failed: %s — it will be retried on "
                      "the next engine start; to fix manually run: %s",
                      exc, " ".join(cmd))
            return False
        _fixed = not missing_core()
        if _fixed:
            log.info("dependency install finished, core modules OK")
        else:
            log.error("dependency install ran but core modules still missing "
                      "— see the pip output above")
        return _fixed
