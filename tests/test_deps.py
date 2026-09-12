"""Self-healing dependency install (deps.ensure_core_deps)."""

from src import deps


def test_missing_core_empty_in_dev_env():
    # The dev venv has everything; on a broken Astra runtime this returns
    # the missing names instead — the contract ensure_core_deps() relies on.
    assert deps.missing_core() == []


def test_ensure_core_deps_fast_path():
    # Everything present -> returns True without running pip.
    assert deps.ensure_core_deps() is True
