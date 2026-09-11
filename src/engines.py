"""Quality engine: faster-whisper, loaded lazily, CPU int8 by default.

Vosk types live text instantly but its Russian quality is modest. Every
finished segment is re-transcribed here and the better text replaces what
Vosk typed. The model loads on first use (or on warm-up) and stays resident;
`download_root` keeps everything inside the plugin folder.
"""

import logging
import threading

import numpy as np

from .settings import WHISPER_MODELS

log = logging.getLogger("voice-text-input.whisper")


class WhisperEngine:
    def __init__(self, models_dir, model_size: str = "small",
                 device: str = "cpu", compute_type: str = "int8"):
        if model_size not in WHISPER_MODELS:
            model_size = "small"
        self.model_size = model_size
        self.models_dir = models_dir
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        self.load_error: str | None = None

    def warm_up(self) -> bool:
        """Load the model now (background thread) so the first correction is fast."""
        try:
            self._ensure()
            return True
        except Exception as exc:
            self.load_error = str(exc)
            log.warning("whisper model unavailable: %s", exc)
            return False

    def _ensure(self):
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel  # heavy import, lazy on purpose

            log.info("loading faster-whisper %s (%s/%s)…",
                     self.model_size, self.device, self.compute_type)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.models_dir),
            )
            log.info("faster-whisper %s ready", self.model_size)

    def reset(self) -> None:
        """Drop the resident model (settings changed) — reloaded on next use."""
        with self._lock:
            self._model = None
            self.load_error = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, pcm16: np.ndarray, language: str = "ru") -> str:
        """Transcribe 16 kHz mono int16 audio; returns plain text."""
        self._ensure()
        audio = pcm16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=language or None,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments if s.text).strip()
