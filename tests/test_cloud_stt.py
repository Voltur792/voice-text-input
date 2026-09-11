"""Unit tests for the cloud one-shot engines' offline logic."""

import io
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cloud_stt import (  # noqa: E402
    GoogleEngine,
    OpenAICompatibleEngine,
    parse_google_response,
    parse_openai_response,
    pcm16_to_wav,
)
from src.settings import ENGINE_GOOGLE, ENGINE_OPENAI, Settings  # noqa: E402


def test_pcm16_to_wav_roundtrip():
    pcm = np.array([0, 1000, -1000, 32767, -32768], dtype="int16")
    wav = pcm16_to_wav(pcm, 16000)
    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 16000
        raw = reader.readframes(reader.getnframes())
    assert np.frombuffer(raw, dtype="int16").tolist() == pcm.tolist()


def test_parse_openai_response():
    assert parse_openai_response({"text": " привет мир "}) == "привет мир"
    assert parse_openai_response({"nope": 1}) == ""
    assert parse_openai_response("garbage") == ""
    assert parse_openai_response(None) == ""


def test_parse_google_response():
    payload = {
        "results": [
            {"alternatives": [{"transcript": "первый"}]},
            {"alternatives": [{"transcript": "второй"}]},
        ]
    }
    assert parse_google_response(payload) == "второй"
    assert parse_google_response({"results": []}) == ""
    assert parse_google_response(None) == ""


def test_engines_not_configured_without_key():
    assert not OpenAICompatibleEngine("").configured
    assert not GoogleEngine("").configured
    assert OpenAICompatibleEngine("sk-1").configured
    assert GoogleEngine("AIza1").configured


def test_engines_transcribe_empty_or_unconfigured():
    pcm = np.zeros(16000, dtype="int16")
    assert OpenAICompatibleEngine("").transcribe(pcm) == ""
    assert OpenAICompatibleEngine("sk-1").transcribe(np.zeros(0, dtype="int16")) == ""
    assert GoogleEngine("").transcribe(pcm) == ""


def test_settings_cloud_fields():
    s = Settings.from_config({
        "engine": "openai",
        "openai_api_key": " sk-1 ",
        "openai_base_url": "https://api.groq.com/openai/v1/",
        "openai_model": "whisper-large-v3",
    })
    assert s.engine == ENGINE_OPENAI
    assert s.openai_api_key == "sk-1"
    assert s.openai_base_url == "https://api.groq.com/openai/v1"
    assert s.openai_model == "whisper-large-v3"

    g = Settings.from_config({"engine": "google", "google_api_key": " AIza9 "})
    assert g.engine == ENGINE_GOOGLE
    assert g.google_api_key == "AIza9"


def test_settings_openai_defaults_survive_blank():
    s = Settings.from_config({"engine": "openai", "openai_api_key": "sk",
                              "openai_base_url": "", "openai_model": ""})
    assert s.openai_base_url == "https://api.openai.com/v1"
    assert s.openai_model == "whisper-1"
