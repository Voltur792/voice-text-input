"""Model discovery/download for fresh catalog installs (src/models.py).

Regression: the .astraplugin bundle ships no models, so a catalog install
found no vosk model and dictation stayed dead. The engine must find models
in the shared per-user dir, download once when nothing exists, and keep the
whisper HF cache in a place that survives plugin updates.
"""

import io
import zipfile

import src.dictation as dictation
from src import models
from src.dictation import DictationEngine
from src.settings import Settings


# ── discovery ──────────────────────────────────────────────────────────────

def _make_model(path) -> None:
    """A structurally complete model folder (passes looks_like_vosk_model)."""
    (path / "conf").mkdir(parents=True)
    (path / "conf" / "model.conf").write_text("sample-rate=16000\n")
    (path / "am").mkdir()
    (path / "graph").mkdir()


def test_find_vosk_model_prefers_plugin_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: tmp_path / "shared")
    local = tmp_path / "plugin" / "models" / "vosk-model-small-ru-0.22"
    _make_model(local)
    shared = tmp_path / "shared" / "vosk-model-small-ru-0.22"
    _make_model(shared)
    assert models.find_vosk_model(tmp_path / "plugin") == local


def test_find_vosk_model_falls_back_to_shared(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: tmp_path / "shared")
    shared = tmp_path / "shared" / "vosk-model-small-ru-0.22"
    _make_model(shared)
    assert models.find_vosk_model(tmp_path / "plugin") == shared


def test_find_vosk_model_missing_everywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: None)
    assert models.find_vosk_model(tmp_path / "plugin") is None


def test_download_dir_prefers_shared(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: tmp_path / "shared")
    assert models.download_dir(tmp_path / "plugin") == tmp_path / "shared"


def test_download_dir_without_appdata_uses_plugin_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: None)
    assert models.download_dir(tmp_path / "plugin") == tmp_path / "plugin" / "models"


def test_models_home_populated_plugin_folder_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: tmp_path / "shared")
    local = tmp_path / "plugin" / "models"
    local.mkdir(parents=True)
    (local / "vosk-model-small-ru-0.22").mkdir()
    assert models.models_home(tmp_path / "plugin") == local


def test_models_home_fresh_install_uses_shared(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: tmp_path / "shared")
    assert models.models_home(tmp_path / "plugin") == tmp_path / "shared"


def test_models_home_without_appdata_falls_back_to_plugin_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "shared_models_dir", lambda: None)
    assert models.models_home(tmp_path / "plugin") == tmp_path / "plugin" / "models"


# ── download ───────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _zip_bytes(name: str) -> bytes:
    """A zip with a structurally complete vosk model (conf + am + graph)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{name}/conf/model.conf", "sample-rate=16000\n")
        zf.writestr(f"{name}/am/final.mdl", "x")
        zf.writestr(f"{name}/graph/words.txt", "x")
    return buf.getvalue()


def test_download_vosk_model_unpacks_and_cleans_up(tmp_path, monkeypatch):
    payload = _zip_bytes("vosk-model-small-ru-0.22")
    monkeypatch.setattr(
        models.urllib.request, "urlopen",
        lambda request, timeout=None: _FakeResponse(payload))
    dest = tmp_path / "dl"
    model = models.download_vosk_model(dest)
    assert model is not None
    assert model.name == "vosk-model-small-ru-0.22"
    assert (model / "conf" / "model.conf").read_text() == "sample-rate=16000\n"
    assert not (dest / "_vosk-model.zip").exists()  # archive cleaned up


def test_download_vosk_model_network_failure_returns_none(tmp_path, monkeypatch):
    def boom(request, timeout=None):
        raise OSError("no network")

    monkeypatch.setattr(models.urllib.request, "urlopen", boom)
    assert models.download_vosk_model(tmp_path / "dl") is None
    assert not (tmp_path / "dl" / "_vosk-model.zip").exists()


def test_find_vosk_model_ignores_incomplete_folder(tmp_path, monkeypatch):
    """A half-extracted folder (crashed download) must not be picked up —
    otherwise vosk retries loading it forever and never recovers."""
    monkeypatch.setattr(models, "shared_models_dir", lambda: None)
    broken = tmp_path / "plugin" / "models" / "vosk-model-small-ru-0.22"
    (broken / "conf").mkdir(parents=True)  # no am/, no graph/
    assert models.find_vosk_model(tmp_path / "plugin") is None


def test_download_replaces_broken_partial_folder(tmp_path, monkeypatch):
    payload = _zip_bytes("vosk-model-small-ru-0.22")
    monkeypatch.setattr(models.urllib.request, "urlopen",
                        lambda request, timeout=None: _FakeResponse(payload))
    dest = tmp_path / "dl"
    dest.mkdir()
    broken = dest / "vosk-model-small-ru-0.22"
    (broken / "conf").mkdir(parents=True)  # half-extracted from an earlier attempt
    model = models.download_vosk_model(dest)
    assert model == broken
    assert models.looks_like_vosk_model(model)
    assert not (dest / "_vosk-unpack-tmp").exists()


def test_download_crashed_extraction_leaves_no_model_folder(tmp_path, monkeypatch):
    """Extraction dies halfway — no partial model folder, no tmp dir left."""
    payload = _zip_bytes("vosk-model-small-ru-0.22")
    monkeypatch.setattr(models.urllib.request, "urlopen",
                        lambda request, timeout=None: _FakeResponse(payload))

    class _BoomZip:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extractall(self, dest):
            raise OSError("disk full")

    monkeypatch.setattr(models.zipfile, "ZipFile", _BoomZip)
    dest = tmp_path / "dl"
    assert models.download_vosk_model(dest) is None
    assert list(dest.glob("vosk-model*ru*")) == []  # no half-extracted folder
    assert not (dest / "_vosk-unpack-tmp").exists()
    assert not (dest / "_vosk-model.zip").exists()


def test_download_second_call_after_success_reuses_model(tmp_path, monkeypatch):
    """A concurrent caller (stt_load + orchestrator retry) must not start a
    second download once the first one has finished."""
    payload = _zip_bytes("vosk-model-small-ru-0.22")
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        return _FakeResponse(payload)

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "dl"
    first = models.download_vosk_model(dest)
    second = models.download_vosk_model(dest)
    assert first is not None and second == first
    assert len(calls) == 1  # the second call reused the downloaded model


# ── dictation wiring: _ensure_vosk ─────────────────────────────────────────

class _FakeVosk:
    def __init__(self, path):
        self.path = path

    def new_recognizer(self):
        return object()


def bare_engine(plugin_root) -> DictationEngine:
    eng = object.__new__(DictationEngine)
    eng.plugin_root = plugin_root
    eng.settings = Settings.from_config({"engine": "vosk"})
    eng._vosk = None
    eng._rec = None
    eng._vosk_failed_at = 0.0
    eng._last_vosk_err = None
    eng._last_vosk_err_at = 0.0
    eng._status = "off"
    return eng


def test_ensure_vosk_downloads_model_on_fresh_catalog_install(tmp_path, monkeypatch):
    """No model anywhere -> download once into the shared dir, then load."""
    eng = bare_engine(tmp_path)
    downloaded = tmp_path / "shared" / "vosk-model-small-ru-0.22"
    monkeypatch.setattr(models, "shared_models_dir", lambda: tmp_path / "shared")
    monkeypatch.setattr(models, "find_vosk_model", lambda root: None)

    def fake_download(dest):
        assert dest == tmp_path / "shared"  # must survive plugin updates
        downloaded.mkdir(parents=True)
        return downloaded

    monkeypatch.setattr(models, "download_vosk_model", fake_download)
    monkeypatch.setattr(dictation, "VoskListener", _FakeVosk)
    assert eng._ensure_vosk() is True
    assert eng._vosk is not None and eng._vosk.path == downloaded


def test_ensure_vosk_uses_existing_model_without_downloading(tmp_path, monkeypatch):
    eng = bare_engine(tmp_path)
    existing = tmp_path / "shared" / "vosk-model-small-ru-0.22"
    monkeypatch.setattr(models, "find_vosk_model", lambda root: existing)

    def no_download(dest):
        raise AssertionError("must not download when a model exists")

    monkeypatch.setattr(models, "download_vosk_model", no_download)
    monkeypatch.setattr(dictation, "VoskListener", _FakeVosk)
    assert eng._ensure_vosk() is True
    assert eng._vosk.path == existing


def test_ensure_vosk_download_failure_is_retryable(tmp_path, monkeypatch):
    eng = bare_engine(tmp_path)
    monkeypatch.setattr(models, "shared_models_dir", lambda: tmp_path / "shared")
    monkeypatch.setattr(models, "find_vosk_model", lambda root: None)
    monkeypatch.setattr(models, "download_vosk_model", lambda dest: None)
    assert eng._ensure_vosk() is False
    assert "download failed" in eng._status
