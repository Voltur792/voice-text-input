"""Dependency self-healing: the bootstrap that runs BEFORE the SDK import.

Astra installs requirements.lock as one flat all-or-nothing set; when a single
pin fails, astra_plugin_sdk is missing too and the plugin used to die at import
time without ever reaching its own recovery code. These tests cover the
fallback chain (lock → loose requirements → explicit pins, plain → --user), the
give-up path (no pip spam per retry) and the on-demand whisper install.
"""

import sys

import pytest

from src import deps


@pytest.fixture()
def pip_calls(monkeypatch):
    """Reset the per-process install bookkeeping, capture (and fail) pip."""
    monkeypatch.setattr(deps, "_ready", set())
    monkeypatch.setattr(deps, "_failed", set())
    calls: list[list[str]] = []
    monkeypatch.setattr(deps, "_pip", lambda args: calls.append(list(args)) or False)
    return calls


def _nothing_missing(monkeypatch):
    monkeypatch.setattr(deps, "missing", lambda modules: [])


def _never_installable(monkeypatch, *names):
    monkeypatch.setattr(deps, "missing", lambda modules: [n for n in modules if n in names])


def test_missing_core_empty_in_dev_env():
    # The dev venv has everything; on a broken Astra runtime this returns
    # the missing names instead — the contract ensure_runtime() relies on.
    assert deps.missing_core() == []


def test_missing_reports_unimportable_names():
    assert deps.missing(("src.settings", "no_such_module_at_all")) == ["no_such_module_at_all"]


def test_ensure_runtime_fast_path_without_pip(pip_calls, monkeypatch):
    _nothing_missing(monkeypatch)
    assert deps.ensure_runtime() is True
    assert pip_calls == []


def test_core_deps_fast_path_without_pip(pip_calls, monkeypatch):
    _nothing_missing(monkeypatch)
    assert deps.ensure_core_deps() is True
    assert pip_calls == []


def test_install_chain_lock_then_requirements_then_pins(pip_calls, monkeypatch):
    """A lock pin with no wheel for this interpreter must not be the end.

    Every source is tried plain and then with --user (protected system
    interpreters), the loose requirements before the bare package names. All
    file sources go through -r: a bare path is an artifact to pip, not a
    requirements file.
    """
    _never_installable(monkeypatch, "astra_plugin_sdk")
    assert deps.ensure_runtime() is False
    lock, req = str(deps._LOCK_FILE), str(deps._REQUIREMENTS_FILE)
    assert pip_calls == [
        ["-r", lock],
        ["--user", "-r", lock],
        ["-r", req],
        ["--user", "-r", req],
        list(deps._MINIMAL_PKGS),
        ["--user", *deps._MINIMAL_PKGS],
    ]


def test_requirements_used_when_lock_is_absent(pip_calls, monkeypatch):
    _never_installable(monkeypatch, "astra_plugin_sdk")
    monkeypatch.setattr(deps, "_LOCK_FILE", deps.Path("does-not-exist.lock"))
    deps.ensure_runtime()
    assert pip_calls[0] == ["-r", str(deps._REQUIREMENTS_FILE)]


def test_successful_install_unblocks_the_bootstrap(pip_calls, monkeypatch):
    """After pip succeeds the modules import, so the bootstrap proceeds."""
    installed = {"done": False}

    def fake_missing(modules):
        if "astra_plugin_sdk" not in modules or installed["done"]:
            return []
        return ["astra_plugin_sdk"]

    def fake_pip(args):
        installed["done"] = True
        return True

    monkeypatch.setattr(deps, "missing", fake_missing)
    monkeypatch.setattr(deps, "_pip", fake_pip)
    assert deps.ensure_runtime() is True


def test_failed_install_is_not_retried_endlessly(pip_calls, monkeypatch):
    """The whisper worker asks per dictation segment — a dead install must
    not run pip again for every one of them."""
    _never_installable(monkeypatch, "astra_plugin_sdk")
    assert deps.ensure_runtime() is False
    runs = len(pip_calls)
    assert runs > 0
    assert deps.ensure_runtime() is False
    assert len(pip_calls) == runs


def test_bootstrap_announces_the_install_on_stdout(pip_calls, monkeypatch, capsys):
    """Astra waits 20 s for the plugin's first stdout line; installing the
    first time takes longer — say something before going quiet."""
    _never_installable(monkeypatch, "astra_plugin_sdk")
    deps.ensure_runtime()
    out = capsys.readouterr().out
    assert "astra_plugin_sdk" in out and "dependencies" in out


def test_whisper_is_installed_on_demand_with_its_own_pins(pip_calls, monkeypatch):
    _never_installable(monkeypatch, "faster_whisper")
    assert deps.ensure_whisper() is False
    assert pip_calls[0] == list(deps._WHISPER_PKGS)
    assert "vosk" not in " ".join(pip_calls[0])  # not the core set again


def test_whisper_failure_is_remembered(pip_calls, monkeypatch):
    _never_installable(monkeypatch, "faster_whisper")
    assert deps.ensure_whisper() is False
    runs = len(pip_calls)
    assert deps.ensure_whisper() is False
    assert len(pip_calls) == runs


def test_whisper_present_needs_no_install(pip_calls, monkeypatch):
    _nothing_missing(monkeypatch)
    assert deps.ensure_whisper() is True
    assert pip_calls == []


def test_core_deps_does_not_require_the_sdk(pip_calls, monkeypatch):
    """The standalone app shares this module and runs without the SDK:
    vosk missing must go straight to the core install, not the runtime one."""
    _never_installable(monkeypatch, "astra_plugin_sdk", "vosk")
    assert deps.ensure_core_deps() is False
    assert pip_calls[0] == ["-r", str(deps._LOCK_FILE)]
    # and the SDK alone missing does not stop dictation deps from installing
    pip_calls.clear()
    monkeypatch.setattr(deps, "_ready", set())
    monkeypatch.setattr(deps, "_failed", set())
    _never_installable(monkeypatch, "astra_plugin_sdk")
    assert deps.ensure_core_deps() is True
    assert pip_calls == []


def test_frozen_build_never_runs_pip(pip_calls, monkeypatch):
    """In a PyInstaller bundle sys.executable IS the app: `… -m pip install`
    would relaunch the exe with pip arguments. The frozen build carries its
    dependencies — a missing module there is logged, never installed."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _never_installable(monkeypatch, "vosk")
    assert deps.ensure_core_deps() is False
    assert pip_calls == []
    # the failure is remembered — no pip on the next ask either
    assert deps.ensure_core_deps() is False
    assert pip_calls == []
