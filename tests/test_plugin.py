"""Tests for VoiceTextInput.

Run: `pytest`.

This is level 1: in process, no daemon, no socket, fast enough to run on every
save. It still goes through the real gRPC servicer, so a tool that is declared
but not routed fails here. When you want the other level — a real handshake, a
real session token, real protobuf encoding — reach for `WireHarness` from the
same module.
"""

import sys
from pathlib import Path

# The daemon puts the bundle root on `sys.path` before importing `src.plugin`;
# do the same so `pytest` from the project root finds it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra_plugin_sdk.testing import Harness, fuzz_configs  # noqa: E402

from src.plugin import VoiceTextInput  # noqa: E402

def test_the_plugin_starts_and_answers_a_health_check():
    with Harness(VoiceTextInput()) as h:
        healthy, _status = h.health()
        assert healthy


def test_no_config_the_daemon_can_deliver_crashes_this_plugin():
    # The daemon delivers config it did not author: the user's typing, and an
    # older version of this plugin's own schema. `{}` — a fresh install — is
    # the first payload every plugin ever sees. None of it may throw.
    with Harness(VoiceTextInput()) as h:
        for payload in fuzz_configs():
            h.set_config(payload)
