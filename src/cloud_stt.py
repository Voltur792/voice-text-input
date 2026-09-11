"""Cloud one-shot STT engines: OpenAI-compatible and Google Cloud.

Both take a complete 16 kHz PCM16 segment and return text — they are not
streaming, so in dictation they refine finished segments (the live draft still
comes from Vosk) and in the STT provider they answer once per utterance.

OpenAICompatibleEngine covers any server speaking the OpenAI audio API:
OpenAI itself, Groq, a local whisper.cpp server, LM Studio, etc. — the base
URL and model are configurable.
GoogleEngine uses the v1 REST `speech:recognize` endpoint.

Every failure logs and returns "" — the caller keeps the Vosk draft.
"""

import base64
import io
import logging
import wave

import numpy as np

log = logging.getLogger("voice-text-input.cloud")

OPENAI_TIMEOUT = 60
GOOGLE_TIMEOUT = 30


def pcm16_to_wav(pcm16: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM16 mono in a minimal WAV container (for the OpenAI API)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


def parse_openai_response(payload) -> str:
    """`{"text": "…"}` from the OpenAI audio API; '' on anything else."""
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def parse_google_response(payload) -> str:
    """Best transcript out of a Google speech:recognize response."""
    if not isinstance(payload, dict):
        return ""
    best = ""
    for result in payload.get("results") or []:
        alts = result.get("alternatives") or []
        if alts:
            text = (alts[0].get("transcript") or "").strip()
            if text:
                best = text
    return best


class OpenAICompatibleEngine:
    """Any OpenAI-audio-API server: OpenAI, Groq, whisper.cpp server…"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "whisper-1", language: str = "ru"):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
        self.model = (model or "whisper-1").strip()
        self.language = (language or "ru").strip()
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, pcm16: np.ndarray) -> str:
        if not self.configured or pcm16.size == 0:
            return ""
        try:
            import requests

            wav = pcm16_to_wav(pcm16)
            response = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": self.model, "language": self.language},
                timeout=OPENAI_TIMEOUT,
            )
            if response.status_code != 200:
                self.last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                log.warning("openai stt failed: %s", self.last_error)
                return ""
            return parse_openai_response(response.json())
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("openai stt failed: %s", exc)
            return ""


class GoogleEngine:
    """Google Cloud Speech-to-Text v1, sync recognize with an API key."""

    def __init__(self, api_key: str, language: str = "ru-RU"):
        self.api_key = (api_key or "").strip()
        self.language = (language or "ru-RU").strip()
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, pcm16: np.ndarray) -> str:
        if not self.configured or pcm16.size == 0:
            return ""
        try:
            import requests

            body = {
                "config": {
                    "encoding": "LINEAR16",
                    "sampleRateHertz": 16000,
                    "languageCode": self.language,
                    "enableAutomaticPunctuation": True,
                },
                "audio": {"content": base64.b64encode(pcm16.tobytes()).decode("ascii")},
            }
            response = requests.post(
                "https://speech.googleapis.com/v1/speech:recognize",
                params={"key": self.api_key},
                json=body,
                timeout=GOOGLE_TIMEOUT,
            )
            if response.status_code != 200:
                self.last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                log.warning("google stt failed: %s", self.last_error)
                return ""
            return parse_google_response(response.json())
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("google stt failed: %s", exc)
            return ""
