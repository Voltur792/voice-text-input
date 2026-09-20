"""Unit tests for text processing: command matching and spoken punctuation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.textproc import (  # noqa: E402
    apply_spoken_punctuation,
    capitalize_sentences,
    levenshtein,
    match_end_command,
    match_start,
    normalize,
    similar,
    strip_trailing_word,
    strip_session_head,
)


def test_normalize_drops_case_punct_and_yo():
    assert normalize("Напиши, Ёлка!") == "напиши елка"


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("кот", "кот") == 0
    assert levenshtein("кот", "код") == 1
    assert levenshtein("напиши", "напишы") == 1


def test_similar_tolerates_mishearings():
    assert similar("напишы", "напиши")
    assert similar("напишу", "напиши")
    assert similar("аправить", "отправить")
    assert not similar("привет", "напиши")


def test_match_start_found_with_remainder():
    matched, skip = match_start("напиши привет как дела", "напиши")
    assert matched and skip == 1


def test_match_start_requires_the_head():
    matched, skip = match_start("вчера написал письмо", "напиши")
    assert not matched and skip == 0


def test_match_start_two_word_command():
    matched, skip = match_start("запиши мне заметку", "запиши мне")
    assert matched and skip == 2


def test_match_end_command_at_tail():
    matched, keep = match_end_command("привет как дела отправить", "отправить")
    assert matched and keep == 3


def test_match_end_command_alone():
    matched, keep = match_end_command("отправить", "отправить")
    assert matched and keep == 0


def test_match_end_command_inside_text_is_ignored():
    matched, keep = match_end_command("он отправит письмо завтра", "отправить")
    assert not matched and keep == 0


def test_match_end_command_misheard_tail():
    matched, keep = match_end_command("текст отправит", "отправить")
    assert matched and keep == 1


def test_strip_trailing_word():
    assert strip_trailing_word("текст отправить", "отправить") == "текст"
    assert strip_trailing_word("текст", "отправить") == "текст"


def test_spoken_punctuation_basic():
    assert apply_spoken_punctuation("привет запятая мир точка") == "привет, мир."


def test_spoken_punctuation_multiword_phrases_win():
    assert apply_spoken_punctuation("раз точка с запятой два") == "раз; два"
    assert apply_spoken_punctuation("раз с новой строки два") == "раз\nдва"


def test_spoken_punctuation_no_space_before_marks():
    out = apply_spoken_punctuation("слово точка другое слово")
    assert out == "слово. другое слово"


def test_capitalize_sentences():
    assert capitalize_sentences("привет. мир!") == "Привет. Мир!"


def test_no_wake_words_anymore():
    """1.1.0 dropped wake words: "Астра" is Astra's own cue, not ours.

    A hypothesis that begins with an address must NOT match the start word —
    the plugin stays out of the assistant's way.
    """
    matched, _skip = match_start("астра напиши привет", "напиши")
    assert not matched


def test_strip_session_head_removes_the_command_word():
    out = strip_session_head("напиши привет", "напиши", "привет")
    assert out == "привет"


def test_strip_session_head_without_command_word():
    out = strip_session_head("привет как дела", "напиши", "")
    assert out == "привет как дела"


def test_strip_session_head_keeps_legit_dictation():
    # Vosk typed the same first word the correction starts with — the head is
    # real dictation ("написал маме…"), not the command word.
    out = strip_session_head("написал маме привет", "напиши", "написал")
    assert out == "написал маме привет"


def test_strip_session_head_empty():
    assert strip_session_head("", "напиши", "") == ""
