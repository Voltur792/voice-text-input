"""Component smoke test — run once after setup, or when something breaks.

    .venv\\Scripts\\python.exe scripts\\smoke_components.py

Checks, in order: vosk model loads and decodes, faster-whisper model loads
(downloading it into models/ on first run) and transcribes, the microphone
opens and carries signal, the typing bridge is available. Prints one line per
component: OK / FAIL with details.
"""

import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))

SILENCE = b"\x00\x00" * 16000  # one second of digital silence at 16 kHz


def check_vosk() -> None:
    from src.listener import VoskListener

    candidates = sorted(MODELS.glob("vosk-model*ru*"))
    if not candidates:
        print("vosk    FAIL: model not found in models/ (download vosk-model-small-ru-0.22)")
        return
    t0 = time.perf_counter()
    listener = VoskListener(candidates[0])
    rec = listener.new_recognizer()
    listener.feed(rec, SILENCE)
    print(f"vosk    OK: {candidates[0].name} loaded in {time.perf_counter() - t0:.1f}s, decodes")


def check_whisper(model_size: str = "small") -> None:
    from src.engines import WhisperEngine

    t0 = time.perf_counter()
    engine = WhisperEngine(MODELS, model_size)
    try:
        text = engine.transcribe(__import__("numpy").frombuffer(SILENCE, dtype="int16"), "ru")
        print(f"whisper OK: {model_size} ready in {time.perf_counter() - t0:.1f}s, "
              f"transcribes (silence -> {text!r})")
    except Exception as exc:
        print(f"whisper FAIL: {exc}")


def check_mic(seconds: float = 1.5) -> None:
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        print(f"mic     FAIL: {exc}")
        return
    try:
        info = sd.query_devices(kind="input")
        sr = int(info.get("default_samplerate") or 48000)
        frames = []
        with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                            blocksize=int(sr * 0.1)):
            sd.sleep(int(seconds * 1000))
        # separate recording pass with callback
        with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                            blocksize=int(sr * 0.1)) as stream:
            for _ in range(int(seconds / 0.1)):
                data, _overflow = stream.read(int(sr * 0.1))
                frames.append(data.copy())
        pcm = np.concatenate([f[:, 0] for f in frames])
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        print(f"mic     OK: {info.get('name')} @ {sr} Hz, RMS over {seconds}s = {rms:.0f} "
              f"({'signal present' if rms > 30 else 'silence — say something or check the device'})")
    except Exception as exc:
        print(f"mic     FAIL: {exc}")


def check_typer() -> None:
    from src.typer import IS_WINDOWS

    print(f"typer   {'OK: Windows SendInput available' if IS_WINDOWS else 'FAIL: not Windows'}")


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    print(f"voice-text-input component check — models dir: {MODELS}")
    check_vosk()
    check_whisper(size)
    check_mic()
    check_typer()
