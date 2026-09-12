"""The dictation engine: one orchestrator thread owning the whole state.

Thread layout
-------------
* orchestrator (this module's `_run`) — owns ALL state: session, typed tail,
  segment buffers. The only place that types or erases.
* mic thread (`audio.MicCapture`) — posts ("audio", block) events.
* whisper thread (`_whisper_loop`) — serial corrections; posts ("correction", …).

Everything crosses threads through one event queue, so no locks are needed on
the dictation state itself.

Flow
----
IDLE: Vosk partials are scanned for the start word. On a match the engine
types the remainder and switches to DICTATING.

DICTATING: every partial is typed as a delta (with revisions: a changed last
word is erased and retyped; a growing last word is extended in place). On a
pause Vosk emits a final — the segment closes and its audio goes to the
whisper thread; when the better transcript arrives, the segment's text is
erased and retyped (skipped if the user has kept talking since). "отправить"
flushes the tail, optionally waits briefly for the pending correction, presses
Enter and returns to IDLE. "отмена" erases everything typed this session.
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .audio import TARGET_SR, MicCapture
from . import deps
from .engines import WhisperEngine
from .listener import VoskListener
from .settings import (
    ENGINE_GOOGLE,
    ENGINE_OPENAI,
    ENGINE_VOSK,
    ENGINE_WHISPER,
    ENGINE_YANDEX,
    Settings,
)
from .textproc import (
    apply_spoken_punctuation,
    capitalize_sentences,
    match_end_command,
    match_start,
    strip_leading_word,
    strip_session_head,
    strip_trailing_word,
    strip_wake_words,
)
from .typer import backspace, press_enter, press_shift_enter, type_text

log = logging.getLogger("voice-text-input")

IDLE = "idle"
DICTATING = "dictating"
SENDING = "sending"

MIN_SEGMENT_SECS = 0.4
VOSK_RETRY_SECS = 30.0


class TypedTail:
    """Words of the current segment already typed into the window.

    `sync` diffs the typed words against a new hypothesis and returns
    (erase_chars, text_to_type). Handles the two cases that matter:
    a revised last word (erase it, retype) and a growing last word
    (just type the extension — no flicker).
    """

    def __init__(self):
        self.words: list[str] = []
        self.chars = 0

    def clear(self) -> None:
        self.words = []
        self.chars = 0

    @staticmethod
    def _prefix_chars(words: list[str], k: int) -> int:
        total = 0
        for i, w in enumerate(words[:k]):
            total += len(w) + (1 if i > 0 else 0)
        return total

    def sync(self, new_words: list[str]) -> tuple[int, str]:
        old = self.words
        # growing last word: "текс" → "текст" — extend without erasing
        if (len(old) == len(new_words) and old
                and new_words[-1].startswith(old[-1])
                and len(new_words[-1]) > len(old[-1])):
            ext = new_words[-1][len(old[-1]):]
            self.words = list(new_words)
            self.chars += len(ext)
            return 0, ext

        k = 0
        for a, b in zip(old, new_words):
            if a != b:
                break
            k += 1
        keep = self._prefix_chars(old, k)
        erase = self.chars - keep
        tail = new_words[k:]
        text = ""
        if tail:
            text = (" " if keep > 0 else "") + " ".join(tail)
        self.words = list(new_words)
        self.chars = keep + len(text)
        return erase, text


class DictationEngine:
    def __init__(self, plugin_root: Path):
        self.plugin_root = Path(plugin_root)
        self.models_dir = self.plugin_root / "models"
        self.settings = Settings()
        self.settings_seen = False

        self._q: queue.Queue = queue.Queue()
        self._whisper_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._whisper_thread: threading.Thread | None = None

        self._mic: MicCapture | None = None
        self._vosk: VoskListener | None = None
        self._rec = None
        self._vosk_failed_at = 0.0
        self._last_vosk_err: str | None = None
        self._last_vosk_err_at = 0.0
        self._whisper: WhisperEngine | None = None
        self._yandex = None
        self._yandex_key_seen: str | None = None
        self._openai = None
        self._openai_key_seen = None
        self._google = None
        self._google_key_seen: str | None = None

        # dictation state (orchestrator thread only)
        self.state = IDLE
        self.session_id = 0
        self.seg_seq = 0
        self.gen = 0
        self.tail = TypedTail()
        self.session_chars = 0
        self.last_seg_chars = 0
        self.seg_audio: list[np.ndarray] = []
        self.first_segment = True
        self.seg_submitted = False
        self.pending_corr = None
        self._status = "off"

    # ── public API (any thread) ──────────────────────────────────────────

    def apply_settings(self, settings: Settings) -> None:
        self._ensure_thread()
        self._q.put(("settings", settings))

    def status(self) -> str:
        return self._status

    def stop(self) -> None:
        self._stop.set()
        self._q.put(("quit",))
        self._whisper_q.put(None)
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._whisper_thread:
            self._whisper_thread.join(timeout=3.0)

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="vti-engine", daemon=True)
            self._thread.start()
        if self._whisper_thread is None or not self._whisper_thread.is_alive():
            self._whisper_thread = threading.Thread(
                target=self._whisper_loop, name="vti-whisper", daemon=True)
            self._whisper_thread.start()

    # ── shared engines (used by the STT provider too) ────────────────────

    def vosk_ready(self) -> bool:
        return self._vosk is not None

    def vosk_listener(self) -> VoskListener | None:
        """The shared Vosk model, loading it on first use. None = unavailable."""
        if self._vosk is None and not self._ensure_vosk():
            return None
        return self._vosk

    def whisper_engine(self) -> WhisperEngine:
        """The shared faster-whisper engine (created lazily, loaded lazily)."""
        return self._get_whisper()

    def yandex_engine(self):
        """The shared Yandex SpeechKit engine, or None without a key."""
        key = self.settings.yandex_api_key
        if not key:
            return None
        if self._yandex is None or self._yandex_key_seen != key:
            from .yandex import YandexEngine

            self._yandex = YandexEngine(key, self.settings.yandex_lang)
            self._yandex_key_seen = key
        return self._yandex

    def cloud_engine(self):
        """The active cloud engine for the current settings, or None.

        None means "no cloud engine applies or is configured" — callers fall
        back to vosk/whisper.
        """
        s = self.settings
        if s.engine == ENGINE_YANDEX:
            return self.yandex_engine()
        if s.engine == ENGINE_OPENAI:
            if not s.openai_api_key:
                return None
            signature = (s.openai_api_key, s.openai_base_url, s.openai_model)
            if self._openai is None or self._openai_key_seen != signature:
                from .cloud_stt import OpenAICompatibleEngine

                self._openai = OpenAICompatibleEngine(
                    s.openai_api_key, s.openai_base_url, s.openai_model, s.language)
                self._openai_key_seen = signature
            return self._openai
        if s.engine == ENGINE_GOOGLE:
            if not s.google_api_key:
                return None
            if self._google is None or self._google_key_seen != s.google_api_key:
                from .cloud_stt import GoogleEngine

                self._google = GoogleEngine(s.google_api_key, s.yandex_lang)
                self._google_key_seen = s.google_api_key
            return self._google
        return None

    def quality_transcribe(self, pcm: np.ndarray, language: str) -> str:
        """Refine a finished segment with the active quality engine."""
        cloud = self.cloud_engine()
        if cloud is not None:
            return cloud.transcribe(pcm)
        return self._get_whisper().transcribe(pcm, language)

    # ── orchestrator thread ──────────────────────────────────────────────

    def _run(self) -> None:
        log.info("dictation engine started")
        while not self._stop.is_set():
            try:
                ev = self._q.get(timeout=0.5)
            except queue.Empty:
                self._poll_retry()
                continue
            kind = ev[0]
            if kind == "quit":
                break
            if kind == "settings":
                self._on_settings(ev[1])
            elif kind == "audio":
                self._on_audio(ev[1])
            elif kind == "correction":
                self._on_correction(ev[1], ev[2])
        self._teardown()
        log.info("dictation engine stopped")

    def _poll_retry(self) -> None:
        """Periodic recovery: vosk model appearing, mic dying, etc."""
        s = self.settings
        if not s.enabled:
            return
        if self._rec is None and time.monotonic() - self._vosk_failed_at > VOSK_RETRY_SECS:
            if self._ensure_vosk():
                self._start_mic()
                self._status = "listening"
        if self._mic is None and self._rec is not None:
            self._start_mic()

    def _on_settings(self, s: Settings) -> None:
        model_changed = s.whisper_model != self.settings.whisper_model
        self.settings = s
        self.settings_seen = True
        if model_changed and self._whisper is not None:
            self._whisper.reset()
        if not s.enabled:
            self._stop_mic()
            self._end_session()
            self._status = "disabled"
            return
        if not self._ensure_vosk():
            return
        self._start_mic()
        if s.engine == ENGINE_WHISPER and s.corrections:
            self._whisper_q.put(("warmup",))
        elif s.engine == ENGINE_YANDEX and not s.yandex_configured:
            log.warning("engine=yandex selected, but yandex_api_key is empty")
        elif s.engine == ENGINE_OPENAI and not s.openai_api_key:
            log.warning("engine=openai selected, but openai_api_key is empty")
        elif s.engine == ENGINE_GOOGLE and not s.google_api_key:
            log.warning("engine=google selected, but google_api_key is empty")
        self._status = "listening"

    def _ensure_vosk(self) -> bool:
        if self._vosk is not None:
            return True
        if deps.missing_core():
            self._status = "installing dependencies (one-time, see logs)…"
        if not deps.ensure_core_deps():
            self._vosk_failed_at = time.monotonic()
            self._log_vosk_failure_once("vosk module unavailable — dependency "
                                        "install failed (see earlier logs)")
            self._status = "vosk unavailable (dependency install failed)"
            return False
        candidates = sorted(self.models_dir.glob("vosk-model*ru*"))
        if not candidates:
            self._vosk_failed_at = time.monotonic()
            log.warning("vosk model not found in %s", self.models_dir)
            self._status = "vosk model missing (see models/)"
            return False
        try:
            self._vosk = VoskListener(candidates[0])
            self._rec = self._vosk.new_recognizer()
            return True
        except Exception as exc:
            self._vosk_failed_at = time.monotonic()
            self._log_vosk_failure_once(f"vosk load failed: {exc}")
            return False

    def _log_vosk_failure_once(self, message: str) -> None:
        """The retry loop calls this every VOSK_RETRY_SECS — log each distinct
        message immediately, repeats at most once a minute."""
        now = time.monotonic()
        if message != self._last_vosk_err or now - self._last_vosk_err_at > 60:
            log.error(message)
            self._last_vosk_err = message
            self._last_vosk_err_at = now

    def _start_mic(self) -> None:
        if self._mic is not None:
            return
        mic = MicCapture()
        mic.start()
        self._mic = mic
        threading.Thread(target=self._mic_relay, args=(mic,), name="vti-relay", daemon=True).start()

    def _mic_relay(self, mic: MicCapture) -> None:
        while not self._stop.is_set() and mic is self._mic:
            block = mic.get_block(timeout=0.5)
            if block is not None:
                self._q.put(("audio", block))

    def _stop_mic(self) -> None:
        mic, self._mic = self._mic, None
        if mic is not None:
            mic.stop()

    def _teardown(self) -> None:
        self._stop_mic()

    # ── audio → vosk → state machine ─────────────────────────────────────

    def _on_audio(self, pcm: np.ndarray) -> None:
        if self._rec is None:
            return
        try:
            partial, final = VoskListener.feed(self._rec, pcm.tobytes())
        except Exception as exc:
            log.warning("vosk feed failed: %s", exc)
            self._rec = self._vosk.new_recognizer()
            return
        if self.state == DICTATING:
            self.seg_audio.append(pcm)
        if final:
            if self.state == DICTATING:
                self._on_final(final)
            elif self.state == IDLE:
                # the start word may be the whole utterance
                self._try_start(final)
        if not partial:
            return
        if self.state == DICTATING:
            self._on_partial(partial)
        elif self.state == IDLE:
            self._try_start(partial)

    def _try_start(self, partial: str) -> None:
        # The user may address Astra first ("Астра напиши") — wake words go.
        candidate = strip_wake_words(partial, self.settings.wake_word_list)
        matched, skip = match_start(candidate, self.settings.start_word)
        if not matched:
            return
        remainder = candidate.split()[skip:]
        self._begin_session()
        log.info("dictation started")
        self._type_delta(remainder)

    def _begin_session(self) -> None:
        self.state = DICTATING
        self.session_id += 1
        self.seg_seq = 0
        self.gen += 1
        self.tail.clear()
        self.session_chars = 0
        self.last_seg_chars = 0
        self.seg_audio = []
        self.first_segment = True
        self.seg_submitted = False
        self.pending_corr = None
        self._status = "dictating"

    def _end_session(self) -> None:
        self.state = IDLE
        self.tail.clear()
        self.session_chars = 0
        self.last_seg_chars = 0
        self.seg_audio = []
        self.seg_seq = 0
        self.seg_submitted = False
        self.first_segment = True
        self.pending_corr = None
        self.gen += 1
        self._status = "listening" if self.settings.enabled else "disabled"

    def _leading_words(self, text: str) -> list[str]:
        """Words of a hypothesis with the session head stripped.

        The first segment's audio begins with the wake/command words that
        started the session, so EVERY hypothesis of that segment (partial and
        final alike) carries them at its head — strip on every call, not just
        once. (Stripping only once let the segment's final re-add "напиши",
        and the tail diff then typed the command word into the window.)
        """
        words = text.split()
        if self.first_segment:
            words = strip_wake_words(
                " ".join(words), self.settings.wake_word_list).split()
            matched, skip = match_start(" ".join(words), self.settings.start_word)
            if matched:
                words = words[skip:]
        return words

    def _on_partial(self, partial: str) -> None:
        canceled, _keep = match_end_command(partial, self.settings.cancel_word)
        if canceled:
            self._do_cancel()
            return
        # With a finish word configured, "отправить" keeps the session alive
        # (continuous dictation); without one it ends it, as before.
        send_finishes = not self.settings.finish_word
        sent, keep = match_end_command(partial, self.settings.send_word)
        if sent:
            self._do_send(partial.split()[:keep], finish=send_finishes)
            return
        if self.settings.finish_word:
            done, keep = match_end_command(partial, self.settings.finish_word)
            if done:
                self._do_send(partial.split()[:keep], finish=True)
                return
        self._type_delta(self._leading_words(partial))

    def _on_final(self, final_text: str) -> None:
        raw_words = final_text.split()
        canceled, _keep = match_end_command(final_text, self.settings.cancel_word)
        if canceled:
            self._do_cancel()
            self.seg_audio = []
            return
        send_finishes = not self.settings.finish_word
        sent, keep = match_end_command(final_text, self.settings.send_word)
        done, finish_keep = (match_end_command(final_text, self.settings.finish_word)
                             if self.settings.finish_word else (False, 0))
        newline, nl_keep = (match_end_command(final_text, self.settings.new_line_word)
                            if self.settings.new_line_word else (False, 0))
        if sent:
            self._do_send(raw_words[:keep], finish=send_finishes)
            return
        if done:
            self._do_send(raw_words[:finish_keep], finish=True)
            return
        if newline:
            self._type_delta(self._leading_words(" ".join(raw_words[:nl_keep])))
            # Shift+Enter: a real line break that does NOT send the message
            # in messengers (Telegram/VK send on bare Enter)
            press_shift_enter()
            self.session_chars += 1
            self._close_segment(final_text)
            return
        words = self._leading_words(final_text)
        self._type_delta(words)
        self._close_segment(final_text)

    def _type_delta(self, words: list[str]) -> None:
        if words and not self.tail.words and self.session_chars > 0:
            # a new segment after a pause: keep it apart from the previous text
            type_text(" ")
            self.session_chars += 1
        erase, text = self.tail.sync(words)
        if erase > 0:
            backspace(erase)
            self.session_chars = max(0, self.session_chars - erase)
        if text:
            type_text(text)
            self.session_chars += len(text)
        if erase or text:
            self.gen += 1

    # ── segment closing and corrections ──────────────────────────────────

    def _close_segment(self, final_text: str) -> None:
        """Pause detected: close the segment, then improve its text."""
        seg_pcm = np.concatenate(self.seg_audio) if self.seg_audio else None
        self.seg_audio = []
        typed = self.tail.chars
        typed_text = " ".join(self.tail.words)
        typed_first = typed_text.split()[0] if typed_text.split() else ""
        was_first_segment = self.first_segment
        self.tail.clear()
        self.seg_seq += 1
        self.first_segment = False
        self.seg_submitted = True

        if self.settings.engine == ENGINE_VOSK or not self.settings.corrections:
            # offline mode: polish punctuation/caps ourselves, synchronously.
            # The final text may still carry the wake + command words at its
            # head (this segment is where the session began) — strip them.
            headless = strip_session_head(
                final_text, self.settings.wake_word_list,
                self.settings.start_word, typed_first) if final_text else ""
            pretty = apply_spoken_punctuation(capitalize_sentences(headless)) if headless else ""
            if pretty and pretty != typed_text:
                if typed > 0:
                    backspace(typed)
                    self.session_chars = max(0, self.session_chars - typed)
                type_text(pretty)
                self.session_chars += len(pretty)
                self.gen += 1
                self.last_seg_chars = len(pretty)
            else:
                self.last_seg_chars = typed
            return

        self.last_seg_chars = typed
        if seg_pcm is not None and len(seg_pcm) >= TARGET_SR * MIN_SEGMENT_SECS:
            # `first` + `typed_first` let the correction drop the wake and
            # command words the quality engine re-hears at the segment's head
            # while keeping genuinely dictated words that merely look similar.
            meta = {"first": was_first_segment, "typed_first": typed_first}
            self._submit_correction(seg_pcm, meta)

    def _submit_correction(self, seg_pcm: np.ndarray, meta: dict) -> None:
        key = (self.session_id, self.seg_seq - 1, self.gen, dict(meta))
        self._whisper_q.put(("job", key, seg_pcm))
        self.pending_corr = key

    def _on_correction(self, key, text) -> None:
        if self.pending_corr == key:
            self.pending_corr = None
        session_id, seg_seq, gen, meta = key
        if session_id != self.session_id:
            return
        if seg_seq != self.seg_seq - 1 or gen != self.gen:
            return  # newer typing happened — the correction is stale
        if not text or not text.strip():
            return
        corrected = text.strip()
        if meta.get("first"):
            corrected = strip_session_head(
                corrected, self.settings.wake_word_list,
                self.settings.start_word, meta.get("typed_first") or "")
        if meta.get("strip_leading"):
            corrected = strip_leading_word(corrected, meta["strip_leading"])
        if meta.get("strip_trailing"):
            corrected = strip_trailing_word(corrected, meta["strip_trailing"])
        corrected = corrected.strip()
        if not corrected:
            return
        old = self.last_seg_chars
        if old > 0:
            backspace(old)
            self.session_chars = max(0, self.session_chars - old)
        type_text(corrected)
        self.session_chars += len(corrected)
        self.last_seg_chars = len(corrected)
        self.gen += 1
        log.debug("segment corrected: %r", corrected)

    # ── send / cancel ────────────────────────────────────────────────────

    def _do_send(self, keep_words: list[str], finish: bool = True) -> None:
        """Type the kept words, press Enter, then end or continue the session.

        With finish=False (continuous dictation) the session stays in
        DICTATING: the next utterance is typed without the start word.
        """
        # On the first segment the kept words still carry the wake and command
        # words ("астра напиши … отправить") — drop them before typing.
        if self.first_segment:
            candidate = strip_wake_words(
                " ".join(keep_words), self.settings.wake_word_list)
            candidate = strip_leading_word(candidate, self.settings.start_word)
            keep_words = candidate.split()
        self._type_delta(keep_words)
        if not self.seg_submitted and self.seg_audio:
            typed_first = keep_words[0] if keep_words else ""
            meta = {"strip_trailing": self.settings.send_word,
                    "first": True, "typed_first": typed_first}
            seg_pcm = np.concatenate(self.seg_audio)
            self.seg_audio = []
            self.seg_seq += 1
            self.seg_submitted = True
            self.first_segment = False
            if (self.settings.engine != ENGINE_VOSK
                    and self.settings.corrections
                    and len(seg_pcm) >= TARGET_SR * MIN_SEGMENT_SECS):
                self._submit_correction(seg_pcm, meta)

        self.state = SENDING
        self._status = "sending"
        deadline = time.monotonic() + self.settings.send_wait_secs
        while self.pending_corr is not None and time.monotonic() < deadline:
            try:
                ev = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if ev[0] == "correction":
                self._on_correction(ev[1], ev[2])
            elif ev[0] == "audio":
                self._feed_during_send(ev[1])
            else:
                self._q.put(ev)
                time.sleep(0.02)

        press_enter(ctrl=self.settings.send_key == "ctrl+enter")
        log.info("sent %d chars", self.session_chars)
        if finish:
            self._end_session()
            return
        # continuous dictation: keep listening, start a fresh tail
        self.tail.clear()
        self.last_seg_chars = 0
        self.seg_submitted = False
        self.pending_corr = None
        self.state = DICTATING
        self._status = "dictating"

    def _feed_during_send(self, pcm: np.ndarray) -> None:
        """Keep vosk decoding while waiting to send, without typing anything."""
        if self._rec is None:
            return
        try:
            VoskListener.feed(self._rec, pcm.tobytes())
        except Exception:
            pass

    def _do_cancel(self) -> None:
        if self.session_chars > 0:
            backspace(self.session_chars)
            log.info("cancelled %d chars", self.session_chars)
        self._end_session()

    # ── quality worker (whisper or yandex) ───────────────────────────────

    def _get_whisper(self) -> WhisperEngine:
        if self._whisper is None:
            self._whisper = WhisperEngine(self.models_dir, self.settings.whisper_model)
        return self._whisper

    def _whisper_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._whisper_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            if item[0] == "warmup":
                self._get_whisper().warm_up()
                continue
            _kind, key, pcm = item
            text = None
            try:
                text = self.quality_transcribe(pcm, self.settings.language)
            except Exception as exc:
                log.warning("quality correction failed: %s", exc)
                self._whisper = None  # reload on next job
            self._q.put(("correction", key, text))
