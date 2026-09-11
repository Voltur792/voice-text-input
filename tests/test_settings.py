"""Unit tests for Settings.from_config — it must never raise on bad input."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import ENGINE_WHISPER, ENGINE_VOSK, Settings  # noqa: E402


def test_empty_config_gives_defaults():
    s = Settings.from_config({})
    assert s.enabled is True
    assert s.start_word == "напиши"
    assert s.send_word == "отправить"
    assert s.engine == ENGINE_WHISPER
    assert s.whisper_model == "small"
    assert s.send_key == "enter"


def test_none_and_garbage_are_defaults():
    for bad in (None, 42, "строка", [], {"enabled": 7}):
        s = Settings.from_config(bad)  # must not raise
        assert isinstance(s, Settings)


def test_string_booleans_are_understood():
    assert Settings.from_config({"enabled": "false"}).enabled is False
    assert Settings.from_config({"enabled": "yes"}).enabled is True
    assert Settings.from_config({"enabled": "выкл"}).enabled is False


def test_choices_fall_back_on_unknown_values():
    assert Settings.from_config({"engine": "skynet"}).engine == ENGINE_WHISPER
    assert Settings.from_config({"engine": "VOSK"}).engine == ENGINE_VOSK
    assert Settings.from_config({"whisper_model": "gigantic"}).whisper_model == "small"
    assert Settings.from_config({"send_key": "space"}).send_key == "enter"


def test_send_wait_is_clamped():
    assert Settings.from_config({"send_wait_secs": "abc"}).send_wait_secs == 1.2
    assert Settings.from_config({"send_wait_secs": -5}).send_wait_secs == 0.0
    assert Settings.from_config({"send_wait_secs": 99}).send_wait_secs == 5.0


def test_words_are_trimmed_nonempty():
    assert Settings.from_config({"start_word": "  пиши  "}).start_word == "пиши"
    assert Settings.from_config({"start_word": "   "}).start_word == "напиши"
