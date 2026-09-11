"""Vosk offline recognizer wrapper (always-on listener).

The small Russian model decodes the whole audio stream continuously and is
cheap enough (~50 MB RAM, a few percent of one core) to run 24/7. It provides:
partial hypotheses while the user speaks (live typing) and finals on every
pause (segment boundaries for the quality engine).
"""

import json
import logging

log = logging.getLogger("voice-text-input.vosk")


class VoskListener:
    def __init__(self, model_path):
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self.model = Model(str(model_path))
        log.info("vosk model loaded from %s", model_path)

    def new_recognizer(self):
        from vosk import KaldiRecognizer

        return KaldiRecognizer(self.model, 16000)

    @staticmethod
    def feed(rec, pcm_bytes: bytes) -> tuple[str, str]:
        """Feed one block; return (partial, final) after it.

        `final` is non-empty exactly when the recognizer decided the utterance
        ended (a pause), which closes the current dictation segment.
        """
        final = ""
        if rec.AcceptWaveform(pcm_bytes):
            final = json.loads(rec.FinalResult()).get("text", "")
        partial = json.loads(rec.PartialResult()).get("partial", "")
        return partial, final
