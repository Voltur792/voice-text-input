"""Yandex SpeechKit v3 engine — cloud streaming recognition.

Protocol (SpeechKit v3, streaming): a WebSocket at
wss://transcribe.api.cloud.yandex.net/speech/stt/v3/ws with query params
format=lpcm, sampleRateHertz=16000, lang=<lang>; header
``Authorization: Api-Key <key>`` (or ``Bearer <token>`` for an IAM token).
Client sends raw PCM16 frames; the server answers JSON messages
(``{"type": "chunk", "result": {...}}``) with partial and final hypotheses.

End of utterance: the server runs its own VAD and finalizes on a pause, so we
pad the tail of every submitted segment with silence — no EOU control message
is needed, which keeps the client independent of that part of the protocol.

Both a synchronous transcribe (quality worker thread, one-shot STT) and an
async streaming generator (Astra's SttProcess) are provided. Every failure is
logged and yields an empty result — the dictation draft then simply stays as
Vosk typed it.
"""

import asyncio
import json
import logging
import threading

import numpy as np

log = logging.getLogger("voice-text-input.yandex")

WS_URL = "wss://transcribe.api.cloud.yandex.net/speech/stt/v3/ws"
SAMPLE_RATE = 16000
CHUNK_BYTES = 4000          # 2000 samples ≈ 125 ms of PCM16
TAIL_SILENCE_SECS = 0.6     # lets the server's VAD close the utterance
RECV_TIMEOUT = 5.0
TOTAL_TIMEOUT = 20.0


def auth_header(api_key: str) -> str:
    """IAM tokens travel as Bearer, everything else as Api-Key."""
    key = (api_key or "").strip()
    if key.startswith("t1."):
        return f"Bearer {key}"
    return f"Api-Key {key}"


def best_from_messages(raw_messages) -> str:
    """Best transcript out of received server messages.

    Prefers the last final chunk; falls back to the last non-empty partial.
    Malformed lines are skipped — the protocol also carries session events.
    """
    final = ""
    partial = ""
    for raw in raw_messages:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(msg, dict) or msg.get("type") != "chunk":
            continue
        result = msg.get("result") or {}
        alts = result.get("alternatives") or []
        text = (alts[0].get("text", "") if alts else "").strip()
        if not text:
            continue
        if result.get("final"):
            final = text
        else:
            partial = text
    return final or partial


class YandexEngine:
    """Cloud recognizer sharing the settings' key and language."""

    def __init__(self, api_key: str, lang: str = "ru-RU"):
        self.api_key = (api_key or "").strip()
        self.lang = lang or "ru-RU"
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _url(self) -> str:
        from urllib.parse import urlencode

        return f"{WS_URL}?{urlencode({'format': 'lpcm', 'sampleRateHertz': SAMPLE_RATE, 'lang': self.lang})}"

    # ── synchronous one-shot (quality worker, one-shot STT) ──────────────

    def transcribe(self, pcm16: np.ndarray) -> str:
        """Transcribe a complete 16 kHz PCM16 segment; '' on any failure."""
        if not self.configured or pcm16.size == 0:
            return ""
        audio = pcm16.tobytes() + b"\x00\x00" * int(SAMPLE_RATE * TAIL_SILENCE_SECS)
        messages: list[str] = []
        try:
            from websockets.sync.client import connect

            with connect(
                self._url(),
                additional_headers={"Authorization": auth_header(self.api_key)},
                close_timeout=2,
            ) as ws:
                for i in range(0, len(audio), CHUNK_BYTES):
                    ws.send(audio[i:i + CHUNK_BYTES])
                deadline = threading.Event()
                while True:
                    try:
                        msg = ws.recv(timeout=RECV_TIMEOUT)
                    except TimeoutError:
                        break
                    if msg is None:
                        break
                    if isinstance(msg, bytes):
                        continue
                    messages.append(msg)
                    if '"session_finished"' in msg:
                        break
                    if deadline.wait(0) or len(messages) > 2000:
                        break
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("yandex transcribe failed: %s", exc)
        return best_from_messages(messages)

    # ── async streaming (Astra SttProcess) ───────────────────────────────

    async def stream_transcribe(self, audio_iter, resample) -> str:
        """Consume an async iterator of AudioChunk, return the final text.

        `resample(chunk_bytes, sample_rate)` → 16 kHz PCM16 bytes. Audio is
        forwarded as it arrives; after the stream ends a silence tail closes
        the utterance and the final text is collected.
        """
        if not self.configured:
            return ""
        messages: list[str] = []
        try:
            from websockets.asyncio.client import connect

            async def sender(ws):
                sent_any = False
                async for chunk in audio_iter:
                    if not chunk.data:
                        continue
                    sent_any = True
                    await ws.send(resample(chunk.data, chunk.sample_rate))
                if sent_any:
                    await ws.send(b"\x00\x00" * int(SAMPLE_RATE * TAIL_SILENCE_SECS))

            async with connect(
                self._url(),
                additional_headers={"Authorization": auth_header(self.api_key)},
                close_timeout=2,
            ) as ws:
                sender_task = asyncio.create_task(sender(ws))
                try:
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                        except (asyncio.TimeoutError, TimeoutError):
                            break
                        if isinstance(msg, bytes):
                            continue
                        messages.append(msg)
                        if '"session_finished"' in msg:
                            break
                finally:
                    sender_task.cancel()
                    try:
                        await sender_task
                    except (asyncio.CancelledError, Exception):
                        pass
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("yandex stream failed: %s", exc)
        return best_from_messages(messages)
