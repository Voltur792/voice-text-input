"""Unit tests for the Yandex SpeechKit engine's offline logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import ENGINE_YANDEX, ENGINE_WHISPER, Settings  # noqa: E402
from src.yandex import YandexEngine, auth_header, best_from_messages  # noqa: E402


def test_auth_header_api_key_and_iam():
    assert auth_header("AQVN123") == "Api-Key AQVN123"
    assert auth_header("t1.9eulet4PRIVATE") == "Bearer t1.9eulet4PRIVATE"
    assert auth_header("  ") == "Api-Key "


def test_best_from_messages_prefers_last_final():
    msgs = [
        '{"type": "session_started"}',
        '{"type": "chunk", "result": {"final": false, '
        '"alternatives": [{"text": "приве"}]}}',
        '{"type": "chunk", "result": {"final": true, '
        '"alternatives": [{"text": "привет, мир"}]}}',
        '{"type": "chunk", "result": {"final": false, '
        '"alternatives": [{"text": "привет, мир как"}]}}',
    ]
    assert best_from_messages(msgs) == "привет, мир"


def test_best_from_messages_falls_back_to_partial():
    msgs = [
        '{"type": "chunk", "result": {"final": false, '
        '"alternatives": [{"text": "частичный текст"}]}}',
    ]
    assert best_from_messages(msgs) == "частичный текст"


def test_best_from_messages_skips_garbage_and_empty():
    msgs = ["not json", '{"type": "session_finished"}', 42, None]
    assert best_from_messages(msgs) == ""
    assert best_from_messages([]) == ""


def test_engine_not_configured_without_key():
    assert not YandexEngine("").configured
    assert YandexEngine("AQVN1").configured


def test_transcribe_without_key_or_audio_is_empty():
    import numpy as np

    assert YandexEngine("").transcribe(np.zeros(16000, dtype="int16")) == ""
    assert YandexEngine("AQVN1").transcribe(np.zeros(0, dtype="int16")) == ""


def test_settings_yandex_fields():
    s = Settings.from_config({"engine": "yandex", "yandex_api_key": " AQVN9 ",
                              "wake_words": "астра, окей астра"})
    assert s.engine == ENGINE_YANDEX
    assert s.yandex_api_key == "AQVN9"
    assert s.yandex_configured
    # wake words were removed in 1.1.0; a saved config that still carries the
    # key must not break parsing and must not resurrect the feature
    assert not hasattr(s, "wake_word_list")


def test_settings_yandex_without_key_falls_back_clean():
    s = Settings.from_config({"engine": "yandex"})
    assert s.engine == ENGINE_YANDEX
    assert not s.yandex_configured


def test_settings_default_engine_is_whisper():
    assert Settings.from_config({}).engine == ENGINE_WHISPER
