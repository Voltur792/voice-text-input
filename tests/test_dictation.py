"""Dictation engine state-machine tests with mocked typing.

The engine is built without __init__ (no mic/vosk/threads) — only the fields
the orchestrator methods touch are set. type_text/backspace/press_enter are
monkeypatched, so the tests assert on what would appear in the target window.

Regression tests here cover the "напиши gets typed" bug: the first segment's
hypotheses (partials AND the final) all carry the command word, so the head
must be stripped on every hypothesis, not just once.
"""

import queue
import threading

import pytest

import src.dictation as dictation
from src.dictation import DICTATING, IDLE, DictationEngine, TypedTail
from src.settings import Settings


def bare_engine(engine_name: str = "faster-whisper") -> DictationEngine:
    eng = object.__new__(DictationEngine)
    eng.state = IDLE
    eng.settings = Settings.from_config({"engine": engine_name, "corrections": True})
    eng.tail = TypedTail()
    eng.session_id = 1
    eng.seg_seq = 0
    eng.gen = 0
    eng.session_chars = 0
    eng.last_seg_chars = 0
    eng.seg_audio = []
    eng.first_segment = True
    eng.seg_submitted = False
    eng.pending_corr = None
    eng._whisper_unavailable = False
    eng._status = "listening"
    eng._last_vosk_err = ""
    eng._last_vosk_err_at = 0.0
    eng._yandex = None
    eng._yandex_lock = threading.Lock()
    eng._openai = None
    eng._openai_lock = threading.Lock()
    eng._google = None
    eng._google_lock = threading.Lock()
    eng._q = queue.Queue()
    eng._whisper_q = queue.Queue()
    return eng


@pytest.fixture()
def keys(monkeypatch):
    """Capture typing: returns (typed_list, enters_list)."""
    typed: list[str] = []
    enters: list[int] = []
    monkeypatch.setattr(dictation, "type_text", lambda s: typed.append(s))
    monkeypatch.setattr(dictation, "backspace", lambda n: typed.append("\b" * n))
    monkeypatch.setattr(dictation, "press_enter", lambda ctrl=False: enters.append(ctrl))
    monkeypatch.setattr(dictation, "press_shift_enter", lambda: typed.append("\n"))
    return typed, enters


def rendered(typed: list[str]) -> str:
    """Simulate the target window's buffer from the typing/backspace calls."""
    buf = ""
    for chunk in typed:
        if chunk.startswith("\b"):
            buf = buf[:max(0, len(buf) - len(chunk))]
        else:
            buf += chunk
    return buf


def test_first_segment_final_does_not_retype_start_word(keys):
    """partial «напиши» → partial «напиши привет» → final «напиши привет».

    The final carries «напиши» again; the head must be stripped again,
    otherwise the tail diff erases «привет» and types «напиши привет».
    """
    typed, _ = keys
    eng = bare_engine()
    eng._try_start("напиши")
    assert eng.state == DICTATING
    eng._on_partial("напиши привет")
    eng._on_final("напиши привет")
    assert rendered(typed) == "привет"


def test_addressed_to_astra_does_not_start_dictation(keys):
    """1.1.0: no wake words. "Астра напиши…" is the assistant's cue — the
    plugin stays silent (and Astra keeps answering its users). Used to start
    dictation after cutting the wake word off."""
    typed, _ = keys
    eng = bare_engine()
    eng._try_start("астра напиши")
    assert eng.state == IDLE
    assert rendered(typed) == ""


def test_send_from_first_segment_strips_head(keys):
    typed, enters = keys
    eng = bare_engine()
    eng._try_start("напиши")
    eng._on_partial("напиши привет")
    eng._on_final("напиши привет отправить")
    assert rendered(typed) == "привет"
    assert enters == [False]
    # continuous dictation: the session stays alive after «отправить»
    assert eng.state == DICTATING


def test_send_keeps_listening_and_next_text_goes_without_start_word(keys):
    typed, enters = keys
    eng = bare_engine()
    eng._try_start("напиши")
    say(eng, "напиши привет мир")
    say(eng, "отправить")
    assert enters == [False]
    assert eng.state == DICTATING
    say(eng, "как дела")
    assert rendered(typed) == "привет мир как дела"
    assert eng.state == DICTATING


def test_finish_word_ends_session(keys):
    typed, enters = keys
    eng = bare_engine()
    eng._try_start("напиши")
    say(eng, "напиши привет")
    say(eng, "закончить")
    assert rendered(typed) == "привет"
    assert enters == [False]
    assert eng.state == IDLE


def test_finish_word_empty_restores_old_send(keys):
    """With no finish word «отправить» ends the session, as it used to."""
    typed, enters = keys
    eng = bare_engine()
    eng.settings = Settings.from_config(
        {"engine": "faster-whisper", "corrections": True, "finish_word": ""})
    eng._try_start("напиши")
    say(eng, "напиши привет")
    say(eng, "отправить")
    assert rendered(typed) == "привет"
    assert enters == [False]
    assert eng.state == IDLE


def test_new_line_word_types_newline(keys):
    typed, enters = keys
    eng = bare_engine()
    eng._try_start("напиши")
    say(eng, "напиши первая строка")
    say(eng, "вторая строка с новой строки")
    # «с новой строки» is consumed; a newline character is typed instead
    assert rendered(typed) == "первая строка вторая строка\n"
    assert enters == []  # newline is typed, Enter is not pressed


def test_new_line_word_disabled_when_empty(keys):
    typed, _ = keys
    eng = bare_engine()
    eng.settings = Settings.from_config(
        {"engine": "faster-whisper", "corrections": True, "new_line_word": ""})
    eng._try_start("напиши")
    say(eng, "напиши привет с новой строки")
    assert rendered(typed) == "привет с новой строки"


def say(eng: DictationEngine, text: str, secs: float = 1.0) -> None:
    """Feed a final hypothesis as if Vosk had decoded `secs` of audio."""
    import numpy as np

    if eng.state == DICTATING:
        eng.seg_audio.append(np.zeros(int(16000 * secs), dtype="int16"))
    eng._on_final(text)


def test_second_segment_keeps_word_napishi(keys):
    """After the first segment «напиши» is ordinary dictation again."""
    typed, _ = keys
    eng = bare_engine()
    eng._try_start("напиши")
    say(eng, "напиши привет")
    say(eng, "напиши письмо дяде")
    assert rendered(typed) == "привет напиши письмо дяде"


def test_segments_are_separated_by_a_space(keys):
    typed, _ = keys
    eng = bare_engine()
    eng._try_start("напиши")
    say(eng, "напиши привет")
    say(eng, "как дела")
    assert rendered(typed) == "привет как дела"


def test_correction_strips_session_head(keys):
    """Whisper re-hears «напиши» at the segment head; correction drops it."""
    typed, _ = keys
    eng = bare_engine()
    eng._try_start("напиши")
    eng._on_partial("напиши привет")
    say(eng, "напиши привет")
    assert rendered(typed) == "привет"
    kind, key, pcm = eng._whisper_q.get_nowait()
    assert kind == "job"
    assert key[3].get("first") is True
    eng._on_correction(key, "Напиши: привет, как дела")
    assert rendered(typed) == "привет, как дела"


def test_correction_keeps_legitimate_head(keys):
    """Vosk typed «написал…»; the correction's matching head is real text."""
    typed, _ = keys
    eng = bare_engine()
    eng._try_start("напиши")
    eng._on_partial("напиши написал маме")
    say(eng, "напиши написал маме")
    assert rendered(typed) == "написал маме"
    _kind, key, _pcm = eng._whisper_q.get_nowait()
    eng._on_correction(key, "Написал маме утром")
    # the whole segment is replaced; capital at session start is fine
    assert rendered(typed) == "Написал маме утром"


def test_short_segment_no_correction(keys):
    """A segment shorter than MIN_SEGMENT_SECS is not sent to correction."""
    import numpy as np

    typed, _ = keys
    eng = bare_engine()
    eng._try_start("напиши")
    eng.seg_audio.append(np.zeros(1600, dtype="int16"))  # 0.1 s < 0.4 s
    eng._on_final("напиши привет")
    assert rendered(typed) == "привет"
    assert eng._whisper_q.empty()


# ── faster-whisper is optional: dictation must survive without it ─────────


def test_prepare_quality_marks_unavailable_after_a_failed_install(monkeypatch):
    """One failed pip attempt and the flag stays up — no retry per segment."""
    eng = bare_engine()
    calls = []

    def fake_ensure():
        calls.append(1)
        return False

    monkeypatch.setattr(dictation.deps, "ensure_whisper", fake_ensure)
    assert eng._prepare_quality() is False
    assert eng._prepare_quality() is False
    assert eng._whisper_unavailable is True
    assert len(calls) == 1


def test_prepare_quality_cloud_engine_needs_no_whisper(monkeypatch):
    """A configured cloud engine is the quality engine — no whisper install."""
    eng = bare_engine()
    eng.settings = Settings.from_config({"engine": "yandex", "yandex_api_key": "AQVN1"})
    monkeypatch.setattr(dictation.deps, "ensure_whisper",
                        lambda: (_ for _ in ()).throw(AssertionError("must not install")))
    assert eng._prepare_quality() is True
    assert eng._whisper_unavailable is False


def test_whisper_unavailable_polishes_offline_instead_of_queueing(keys):
    """No quality engine → the Vosk text gets the offline punctuation polish
    and nothing is queued (no ImportError per segment in the log)."""
    import numpy as np

    typed, _ = keys
    eng = bare_engine()
    eng._whisper_unavailable = True
    eng._try_start("напиши")
    eng.seg_audio.append(np.zeros(16000, dtype="int16"))  # 1 s, would qualify
    eng._on_final("напиши привет точка")
    assert eng._whisper_q.empty()
    assert rendered(typed) == "Привет."
