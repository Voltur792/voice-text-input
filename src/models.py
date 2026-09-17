"""Model discovery and one-time download.

The .astraplugin bundle ships NO models (hundreds of MB): a fresh install
from the plugin catalog starts with an empty plugin folder, so vosk is never
found there and dictation stays dead ("vosk model not found" every 30 s).
Two fixes live here:

* models are looked up in the plugin's own models/ (sideload dev folder)
  and then in a per-user shared dir %APPDATA%/voice-text-input/models, which
  survives plugin updates (the plugin folder is wiped on every update);
* when no vosk model exists anywhere, it is downloaded once (~45 MB from
  alphacephei.com — the same source the standalone app uses) into the shared
  dir. Runs from a worker thread (the dictation orchestrator is one).
"""

import logging
import os
import shutil
import threading
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("voice-text-input.models")

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
VOSK_MODEL_SIZE_MB = 45

_download_mutex = threading.Lock()


def shared_models_dir() -> Path | None:
    """Per-user model storage that survives plugin updates, or None."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "voice-text-input" / "models"


def model_dirs(plugin_root: Path) -> list[Path]:
    """Where to look for models, best first: plugin folder, then shared."""
    dirs = [Path(plugin_root) / "models"]
    shared = shared_models_dir()
    if shared is not None and shared not in dirs:
        dirs.append(shared)
    return dirs


def looks_like_vosk_model(path: Path) -> bool:
    """Structural completeness check (conf/model.conf + am/ + graph/).

    A download can be interrupted mid-extraction: picking up a half-unpacked
    folder would put vosk into a permanent load-fail retry loop, so both
    discovery and the download result filter through this.
    """
    path = Path(path)
    return ((path / "conf" / "model.conf").is_file()
            and (path / "am").is_dir()
            and (path / "graph").is_dir())


def find_vosk_model(plugin_root: Path) -> Path | None:
    """First complete vosk model folder across model_dirs(), or None.

    Incomplete folders (a crashed extraction) are skipped silently — they
    are replaced on the next download attempt.
    """
    for d in model_dirs(plugin_root):
        if not d.is_dir():
            continue
        for candidate in sorted(d.glob("vosk-model*ru*")):
            if looks_like_vosk_model(candidate):
                return candidate
    return None


def download_dir(plugin_root: Path) -> Path:
    """Where a missing model is downloaded to — the shared dir, so it
    survives plugin updates; the plugin folder only without %APPDATA%."""
    shared = shared_models_dir()
    if shared is not None:
        return shared
    return Path(plugin_root) / "models"


def download_vosk_model(dest_dir: Path) -> Path | None:
    """Download and unpack the vosk model into dest_dir. None on failure.

    Blocks for minutes on a slow link — call from a worker thread (the
    dictation orchestrator is one). Returns the unpacked model folder.

    Interruption-safe: the archive is unpacked into a temp dir and renamed
    into place only when complete, so a crash mid-extraction never leaves a
    half-unpacked model folder behind (it would be picked up forever by
    discovery). A broken partial folder from an older attempt is replaced.
    The mutex keeps a concurrent stt_load + orchestrator retry from running
    two downloads into the same dir at once.
    """
    dest_dir = Path(dest_dir)
    archive = dest_dir / "_vosk-model.zip"
    tmp_dir = dest_dir / "_vosk-unpack-tmp"
    with _download_mutex:
        try:
            # Another thread may have finished the download while we waited.
            existing = [p for p in dest_dir.glob("vosk-model*ru*")
                        if looks_like_vosk_model(p)]
            if existing:
                return existing[0]
            dest_dir.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(
                VOSK_MODEL_URL, headers={"User-Agent": "voice-text-input/1.0"})
            log.warning("downloading vosk model (~%d MB) into %s — one-time; "
                        "see earlier lines if this fails",
                        VOSK_MODEL_SIZE_MB, dest_dir)
            with urllib.request.urlopen(request, timeout=60) as response, \
                    open(archive, "wb") as out:
                # shutil.copyfileobj has no progress logging — read in chunks.
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                mb = 0
                while True:
                    chunk = response.read(262144)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total and done // (10 * 1024 * 1024) != mb:
                        mb = done // (10 * 1024 * 1024)
                        log.info("vosk model download: %d/%d MB",
                                 done // (1024 * 1024), total // (1024 * 1024))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp_dir)
            unpacked = [p for p in sorted(tmp_dir.glob("vosk-model*ru*"))
                        if looks_like_vosk_model(p)]
            if not unpacked:
                log.error("vosk model archive unpacked but no complete model "
                          "appeared in %s", tmp_dir)
                return None
            final = dest_dir / unpacked[0].name
            if final.is_dir():
                # a broken partial folder from an earlier attempt — replace
                shutil.rmtree(final)
            elif final.exists():
                final.unlink()
            unpacked[0].rename(final)
            log.info("vosk model ready at %s", final)
            return final
        except Exception as exc:
            log.error("vosk model download failed: %s — retried on the next "
                      "engine cycle", exc)
            return None
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)


def models_home(plugin_root: Path) -> Path:
    """Where faster-whisper keeps its HF cache (download_root).

    An already populated plugin models/ (sideload dev folder, or a manual
    install) keeps being used so an existing cache is not re-downloaded; a
    fresh catalog install has none, so the cache goes to the shared dir and
    survives plugin updates.
    """
    local = Path(plugin_root) / "models"
    if local.is_dir() and any(local.iterdir()):
        return local
    shared = shared_models_dir()
    if shared is not None:
        return shared
    return local
