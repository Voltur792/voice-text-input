"""Unit tests for TypedTail — the typed-segment diff logic.

This is the piece that decides how many characters to erase and what to type
for every new recognizer hypothesis, so its edge cases are the ones that would
corrupt user text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dictation import TypedTail  # noqa: E402


def test_first_word_is_typed_plain():
    tail = TypedTail()
    assert tail.sync(["привет"]) == (0, "привет")
    assert tail.chars == 6


def test_appended_word_gets_a_leading_space():
    tail = TypedTail()
    tail.sync(["привет"])
    assert tail.sync(["привет", "мир"]) == (0, " мир")
    assert tail.chars == 10


def test_growing_last_word_is_extended_in_place():
    tail = TypedTail()
    tail.sync(["текс"])
    assert tail.sync(["текст"]) == (0, "т")
    assert tail.chars == 5
    assert tail.sync(["текст", "до"]) == (0, " до")


def test_revised_last_word_is_erased_and_retyped():
    tail = TypedTail()
    tail.sync(["привет", "мир"])
    assert tail.sync(["привет", "солнце"]) == (4, " солнце")
    assert tail.chars == len("привет солнце")


def test_shrinking_hypothesis_erases_the_tail():
    tail = TypedTail()
    tail.sync(["привет", "мир"])
    assert tail.sync(["привет"]) == (4, "")
    assert tail.chars == 6


def test_full_rewrite_erases_everything():
    tail = TypedTail()
    tail.sync(["абв"])
    assert tail.sync(["где"]) == (3, "где")
    assert tail.chars == 3


def test_identical_hypothesis_is_a_no_op():
    tail = TypedTail()
    tail.sync(["привет", "мир"])
    assert tail.sync(["привет", "мир"]) == (0, "")
    assert tail.chars == 10


def test_clear_resets_accounting():
    tail = TypedTail()
    tail.sync(["привет"])
    tail.clear()
    assert tail.chars == 0 and tail.words == []
    assert tail.sync(["новое"]) == (0, "новое")
