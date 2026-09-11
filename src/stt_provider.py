"""STT provider: lets Astra use this plugin's engines as its own recognizer.

When the user picks this plugin as the recognizer on Astra's Voice page, the
daemon stops running its built-in STT and streams utterance audio here
instead — so one model serves both Astra's conversation and the dictation
engine. The model objects are shared with `dictation.DictationEngine` (one
copy in RAM); only per-stream recognizers are separate, which is how Vosk is
meant to be used.

The daemon's `stt_load(model_path, …)` offers a path from its own model
catalog; this plugin manages its models in `models/` and ignores that path.
"""

import json
import logging

import numpy as np

from .audio import TARGET_SR, resample_to_16k
from .listener import VoskListener
from .settings import (
    CLOUD_ENGINES,
    ENGINE_GOOGLE,
    ENGINE_OPENAI,
    ENGINE_VOSK,
    ENGINE_YANDEX,
)

log = logging.getLogger("voice-text-input.stt")


class SttProvider:
    """Transcription on top of the dictation engine's shared models."""

    def __init__(self, engine):
        # engine: dictation.DictationEngine — settings + shared vosk/whisper/yandex.
        self.engine = engine

    def _pcm16(self, data: bytes, sample_rate: int) -> np.ndarray:
        pcm = np.frombuffer(data, dtype=np.int16)
        return resample_to_16k(pcm, sample_rate or TARGET_SR)

    # ── one-shot (SDK buffers the utterance and calls this once) ─────────

    def transcribe(self, audio: bytes, sample_rate: int,
                   language: str = "", initial_prompt: str = "") -> str:
        if not audio:
            return ""
        settings = self.engine.settings
        pcm = self._pcm16(audio, sample_rate)
        if pcm.size == 0:
            return ""
        if settings.engine in CLOUD_ENGINES:
            cloud = self.engine.cloud_engine()
            if cloud is not None:
                try:
                    return cloud.transcribe(pcm)
                except Exception as exc:
                    log.warning("cloud stt failed, falling back to vosk: %s", exc)
            else:
                log.warning("cloud stt without API key, falling back to vosk")
        if settings.engine == ENGINE_VOSK or settings.engine in CLOUD_ENGINES:
            listener = self.engine.vosk_listener()
            if listener is None:
                log.warning("stt: vosk model unavailable")
                return ""
            rec = listener.new_recognizer()
            rec.AcceptWaveform(pcm.tobytes())
            return json.loads(rec.FinalResult()).get("text", "")
        try:
            return self.engine.whisper_engine().transcribe(
                pcm, language or settings.language)
        except Exception as exc:
            log.warning("stt transcribe failed: %s", exc)
            return ""

    # ── streaming (partials while the user speaks) ───────────────────────

    async def stream_transcribe(self, audio_iter, options):
        """Async generator: yield {text, is_final} events as audio arrives."""
        settings = self.engine.settings
        language = (options.language if options is not None else "") or settings.language

        if settings.engine == ENGINE_YANDEX:
            yandex = self.engine.yandex_engine()
            if yandex is not None:
                text = await yandex.stream_transcribe(
                    audio_iter,
                    lambda data, sr: self._pcm16(data, sr).tobytes())
                if text:
                    yield {"text": text, "is_final": True}
                else:
                    yield {"text": "", "is_final": True}
                return
            log.warning("yandex stt without API key, falling back to vosk")

        if settings.engine in (ENGINE_OPENAI, ENGINE_GOOGLE):
            cloud = self.engine.cloud_engine()
            if cloud is None:
                log.warning("cloud stt without API key, falling back to vosk")
            else:
                # Not streaming: buffer the utterance, transcribe once.
                chunks: list[np.ndarray] = []
                sample_rate = TARGET_SR
                async for chunk in audio_iter:
                    if chunk.sample_rate:
                        sample_rate = chunk.sample_rate
                    if chunk.data:
                        chunks.append(self._pcm16(chunk.data, sample_rate))
                pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype="int16")
                try:
                    text = cloud.transcribe(pcm) if pcm.size else ""
                except Exception as exc:
                    log.warning("cloud stt stream failed: %s", exc)
                    text = ""
                yield {"text": text or "", "is_final": True}
                return

        if settings.engine == ENGINE_VOSK or settings.engine in CLOUD_ENGINES:
            listener = self.engine.vosk_listener()
            if listener is None:
                log.warning("stt: vosk model unavailable")
                return
            rec = listener.new_recognizer()
            sample_rate = TARGET_SR
            emitted = False
            async for chunk in audio_iter:
                if chunk.sample_rate:
                    sample_rate = chunk.sample_rate
                if not chunk.data:
                    continue
                pcm = self._pcm16(chunk.data, sample_rate)
                if pcm.size == 0:
                    continue
                partial, final = VoskListener.feed(rec, pcm.tobytes())
                if final:
                    emitted = True
                    yield {"text": final, "is_final": True}
                elif partial:
                    emitted = True
                    yield {"text": partial, "is_final": False}
            # Stream end: flush whatever was still mid-decode. An empty final
            # event is still an event — a caller that sent a whole utterance
            # and got silence back cannot tell "recognized nothing" from
            # "recognizer died", so answer explicitly either way.
            tail = json.loads(rec.FinalResult()).get("text", "")
            yield {"text": tail, "is_final": True}
            if not emitted:
                log.debug("stt stream produced no transcript")
            return

        # Whisper is not streaming: buffer the utterance, transcribe once.
        chunks: list[np.ndarray] = []
        sample_rate = TARGET_SR
        async for chunk in audio_iter:
            if chunk.sample_rate:
                sample_rate = chunk.sample_rate
            if chunk.data:
                chunks.append(self._pcm16(chunk.data, sample_rate))
        if not chunks:
            return
        pcm = np.concatenate(chunks)
        try:
            text = self.engine.whisper_engine().transcribe(pcm, language)
        except Exception as exc:
            log.warning("stt stream failed: %s", exc)
            return
        yield {"text": text or "", "is_final": True}
