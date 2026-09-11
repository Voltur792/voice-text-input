"""Unit tests for SttProvider with a fake engine (no models loaded)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra_plugin_sdk.types import AudioChunk, SttOptions  # noqa: E402

from src.settings import (
    ENGINE_GOOGLE,
    ENGINE_OPENAI,
    ENGINE_VOSK,
    ENGINE_WHISPER,
    ENGINE_YANDEX,
    Settings,
)  # noqa: E402
from src.stt_provider import SttProvider  # noqa: E402


def collect(provider, audio_iter, options=None):
    """Drain the streaming hook synchronously (the SDK harness style)."""

    async def run():
        return [e async for e in provider.stream_transcribe(audio_iter, options)]

    return asyncio.run(run())


class FakeEngine:
    def __init__(self, engine_name=ENGINE_WHISPER, vosk=None, whisper=None):
        self.settings = Settings.from_config({"engine": engine_name})
        self._vosk = vosk
        self._whisper = whisper

    def vosk_listener(self):
        return self._vosk

    def whisper_engine(self):
        if self._whisper is None:
            raise RuntimeError("no whisper in test")
        return self._whisper


class FakeVosk:
    def new_recognizer(self):
        return FakeRecognizer()


class FakeRecognizer:
    def AcceptWaveform(self, pcm: bytes) -> bool:
        return len(pcm) > 32000  # >2s of audio counts as a finished utterance

    def FinalResult(self):
        return '{"text": "финал"}'

    def PartialResult(self):
        return '{"partial": "частич"}'


class FakeWhisper:
    def transcribe(self, pcm, language):
        return f"whisper:{language}:{len(pcm)}"


def test_empty_audio_is_empty_text():
    provider = SttProvider(FakeEngine())
    assert provider.transcribe(b"", 16000) == ""


def test_vosk_one_shot_uses_final_result():
    provider = SttProvider(FakeEngine(ENGINE_VOSK, vosk=FakeVosk()))
    assert provider.transcribe(b"\x01\x00" * 40000, 16000) == "финал"


def test_vosk_one_shot_without_model_is_empty_not_crash():
    provider = SttProvider(FakeEngine(ENGINE_VOSK, vosk=None))
    assert provider.transcribe(b"\x01\x00" * 100, 16000) == ""


def test_whisper_one_shot_receives_resampled_pcm():
    provider = SttProvider(FakeEngine(ENGINE_WHISPER, whisper=FakeWhisper()))
    # 48000 Hz → 16000 Hz: one second of audio (48000 samples) becomes 16000.
    pcm = b"\x01\x00" * 48000
    out = provider.transcribe(pcm, 48000, "ru")
    assert out.startswith("whisper:ru:") and out.endswith(":16000")


def test_stream_vosk_yields_partial_then_final():
    provider = SttProvider(FakeEngine(ENGINE_VOSK, vosk=FakeVosk()))

    async def run():
        async def audio():
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000)
            yield AudioChunk(data=b"\x01\x00" * 40000, sample_rate=16000, is_last=True)
        return [e async for e in provider.stream_transcribe(audio(), SttOptions(language="ru"))]

    events = asyncio.run(run())
    assert events[0] == {"text": "частич", "is_final": False}
    assert events[-1]["is_final"] is True


def test_stream_whisper_buffers_until_end():
    provider = SttProvider(FakeEngine(ENGINE_WHISPER, whisper=FakeWhisper()))

    async def run():
        async def audio():
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000)
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000, is_last=True)
        return [e async for e in provider.stream_transcribe(audio(), None)]

    events = asyncio.run(run())
    assert len(events) == 1 and events[0]["is_final"] is True
    assert events[0]["text"].startswith("whisper:ru:")


class FakeCloud:
    def transcribe(self, pcm):
        return f"cloud:{len(pcm)}"


def test_openai_one_shot_is_used_when_configured():
    engine = FakeEngine(ENGINE_OPENAI)
    engine.cloud_engine = lambda: FakeCloud()
    provider = SttProvider(engine)
    assert provider.transcribe(b"\x01\x00" * 16000, 16000, "ru") == "cloud:16000"


def test_google_one_shot_without_key_falls_back_to_vosk():
    engine = FakeEngine(ENGINE_GOOGLE, vosk=FakeVosk())
    engine.cloud_engine = lambda: None
    provider = SttProvider(engine)
    assert provider.transcribe(b"\x01\x00" * 40000, 16000) == "финал"


def test_openai_stream_buffers_and_answers_once():
    engine = FakeEngine(ENGINE_OPENAI)
    engine.cloud_engine = lambda: FakeCloud()
    provider = SttProvider(engine)

    async def run():
        async def audio():
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000)
            yield AudioChunk(data=b"\x01\x00" * 800, sample_rate=16000, is_last=True)
        return [e async for e in provider.stream_transcribe(audio(), None)]

    events = asyncio.run(run())
    # 1600 + 800 samples of int16 → 2400 samples
    assert events == [{"text": "cloud:2400", "is_final": True}]


def test_google_stream_without_key_falls_back_to_vosk():
    engine = FakeEngine(ENGINE_GOOGLE, vosk=FakeVosk())
    engine.cloud_engine = lambda: None
    provider = SttProvider(engine)

    async def run():
        async def audio():
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000)
            yield AudioChunk(data=b"\x01\x00" * 40000, sample_rate=16000, is_last=True)
        return [e async for e in provider.stream_transcribe(audio(), None)]

    events = asyncio.run(run())
    assert events[0] == {"text": "частич", "is_final": False}
    assert events[-1]["is_final"] is True


def test_stream_without_vosk_model_is_silent_not_crash():
    provider = SttProvider(FakeEngine(ENGINE_VOSK, vosk=None))

    async def run():
        async def audio():
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000, is_last=True)
        return [e async for e in provider.stream_transcribe(audio(), None)]

    assert asyncio.run(run()) == []


class FakeYandex:
    def transcribe(self, pcm):
        return f"yandex:{len(pcm)}"

    async def stream_transcribe(self, audio_iter, resample):
        async for _ in audio_iter:
            pass
        return "yandex stream"


def test_yandex_one_shot_is_used_when_configured():
    engine = FakeEngine(ENGINE_YANDEX)
    engine.cloud_engine = lambda: FakeYandex()
    provider = SttProvider(engine)
    assert provider.transcribe(b"\x01\x00" * 16000, 16000, "ru") == "yandex:16000"


def test_yandex_one_shot_without_key_falls_back_to_vosk():
    engine = FakeEngine(ENGINE_YANDEX, vosk=FakeVosk())
    engine.cloud_engine = lambda: None
    provider = SttProvider(engine)
    assert provider.transcribe(b"\x01\x00" * 40000, 16000) == "финал"


def test_yandex_stream_is_used_when_configured():
    engine = FakeEngine(ENGINE_YANDEX)
    engine.yandex_engine = lambda: FakeYandex()
    provider = SttProvider(engine)

    async def run():
        async def audio():
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000, is_last=True)
        return [e async for e in provider.stream_transcribe(audio(), None)]

    events = asyncio.run(run())
    assert events == [{"text": "yandex stream", "is_final": True}]


def test_yandex_stream_without_key_falls_back_to_vosk():
    engine = FakeEngine(ENGINE_YANDEX, vosk=FakeVosk())
    engine.yandex_engine = lambda: None
    provider = SttProvider(engine)

    async def run():
        async def audio():
            yield AudioChunk(data=b"\x01\x00" * 1600, sample_rate=16000)
            yield AudioChunk(data=b"\x01\x00" * 40000, sample_rate=16000, is_last=True)
        return [e async for e in provider.stream_transcribe(audio(), None)]

    events = asyncio.run(run())
    assert events[0] == {"text": "частич", "is_final": False}
    assert events[-1]["is_final"] is True
