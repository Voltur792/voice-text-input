"""Microphone capture: sounddevice → 16 kHz mono int16 blocks.

Opens the input device in WASAPI shared mode at the device's own sample rate
(so Astra or anything else can use the microphone at the same time) and
resamples to 16 kHz for Vosk/Whisper with a linear interpolator — plenty for
speech. Runs in its own daemon thread and never raises into the caller:
failures (device busy, unplugged) are logged and retried every few seconds.
"""

import logging
import queue
import threading
import time

import numpy as np

log = logging.getLogger("voice-text-input.audio")

TARGET_SR = 16000
BLOCK_SECONDS = 0.1
RETRY_SECS = 3.0


def resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == TARGET_SR or len(x) == 0:
        return x
    n_out = int(round(len(x) * TARGET_SR / sr_in))
    if n_out <= 0:
        return x[:0]
    pos = np.linspace(0.0, len(x) - 1.0, num=n_out)
    return np.interp(pos, np.arange(len(x), dtype=np.float64), x.astype(np.float64)).astype(np.int16)


class MicCapture:
    """Continuous mic reader. `get_block(timeout)` yields 16 kHz int16 chunks."""

    def __init__(self, device=None):
        self.device = device
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=300)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vti-mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_block(self, timeout: float) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        import sounddevice as sd  # lazy: keeps plugin startup fast

        while not self._stop.is_set():
            try:
                info = sd.query_devices(self.device, kind="input")
                sr = int(info.get("default_samplerate") or 48000)
                if sr <= 0:
                    sr = 48000
                blocksize = max(1, int(sr * BLOCK_SECONDS))
                self.last_error = None

                def callback(indata, frames, time_info, status):  # noqa: ANN001
                    if status:
                        log.debug("mic status: %s", status)
                    mono = indata[:, 0] if indata.ndim > 1 else indata
                    pcm = resample_to_16k(np.asarray(mono), sr)
                    if pcm.size:
                        try:
                            self._q.put_nowait(pcm)
                        except queue.Full:
                            try:
                                self._q.get_nowait()
                                self._q.put_nowait(pcm)
                            except (queue.Empty, queue.Full):
                                pass

                with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                                    blocksize=blocksize, callback=callback,
                                    device=self.device):
                    log.info("microphone opened: %s @ %d Hz", info.get("name"), sr)
                    while not self._stop.wait(0.2):
                        pass
            except Exception as exc:  # device busy/gone — retry
                self.last_error = str(exc)
                log.warning("microphone unavailable (%s); retry in %ss", exc, RETRY_SECS)
                if self._stop.wait(RETRY_SECS):
                    return
