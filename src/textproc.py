"""Text processing for command words and spoken punctuation.

All matching is fuzzy on purpose: a small offline recognizer (Vosk) mishears
command words regularly ("напишы", "аправить"), and a dictation trigger that
only fires on exact matches feels broken. Levenshtein ratio over normalized
words keeps the false-positive rate low while tolerating mishearings.
"""

import re

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, ё→е, drop punctuation, collapse whitespace."""
    t = (text or "").lower().replace("ё", "е")
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def levenshtein(a: str, b: str) -> int:
    """Classic DP edit distance; inputs are short words, so O(len*len) is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def similar(spoken: str, target: str, threshold: float = 0.7) -> bool:
    """True when normalized `spoken` is close enough to normalized `target`."""
    a, b = normalize(spoken), normalize(target)
    if not a or not b:
        return False
    if a == b:
        return True
    dist = levenshtein(a, b)
    return (1.0 - dist / max(len(a), len(b))) >= threshold


def strip_wake_words(text: str, wake_words: list[str], max_strips: int = 2) -> str:
    """Drop leading wake phrases ("астра", "окей астра"…) from a hypothesis.

    Astra's wake word precedes the command word when the user addresses the
    assistant ("Астра напиши…"), so it must not break command matching.
    Fuzzy, longest phrases first, at most `max_strips` in a row.
    """
    words = (text or "").split()
    phrases = sorted((w for w in wake_words if w), key=len, reverse=True)
    for _ in range(max_strips):
        stripped = False
        for phrase in phrases:
            n = len(phrase.split())
            if len(words) < n:
                continue
            if similar(" ".join(words[:n]), phrase, 0.75):
                words = words[n:]
                stripped = True
                break
        if not stripped or not words:
            break
    return " ".join(words)


def match_start(partial: str, start_word: str, threshold: float = 0.7):
    """Does a partial hypothesis begin with (something like) the start word?

    Returns (matched, skip_words): `skip_words` is how many leading raw words
    the command consumed; the rest is the first dictated fragment.
    """
    raw_words = (partial or "").split()
    target_n = len((start_word or "").split())
    if not raw_words or target_n == 0 or len(raw_words) < target_n:
        return False, 0
    head = " ".join(raw_words[:target_n])
    if similar(head, start_word, threshold):
        return True, target_n
    return False, 0


def match_end_command(partial: str, word: str, threshold: float = 0.72):
    """Does a partial hypothesis END with (something like) the command word?

    Returns (matched, keep_words): `keep_words` is how many leading raw words
    survive before the command word. Matching only at the end (or as the whole
    hypothesis) means dictated text that merely *contains* the command word is
    safe.
    """
    raw_words = (partial or "").split()
    target_n = len((word or "").split())
    if not raw_words or target_n == 0 or len(raw_words) < target_n:
        return False, 0
    tail = " ".join(raw_words[-target_n:])
    if similar(tail, word, threshold):
        return True, len(raw_words) - target_n
    return False, 0


def strip_leading_word(text: str, word: str, threshold: float = 0.7) -> str:
    """Drop the first word of `text` when it is (probably) the command word."""
    words = (text or "").split()
    if words and similar(words[0], word, threshold):
        return " ".join(words[1:])
    return text


def strip_session_head(text: str, wake_words: list[str], start_word: str,
                       typed_first: str = "") -> str:
    """Remove wake + command words from the head of a corrected segment text.

    The quality engine re-transcribes audio that begins with the wake and
    command words ("астра напиши …"), so its text often starts with them even
    though they were never typed. `typed_first` is the first word Vosk typed
    from this segment: when the correction starts with the *same* word, the
    head is legitimate dictation (the user really said "написал…") and is kept.
    """
    words = (text or "").split()
    if not words:
        return text or ""
    typed_first = (typed_first or "").strip()
    if typed_first and similar(words[0], typed_first, 0.75):
        return text
    stripped = strip_wake_words(text, wake_words)
    stripped = strip_leading_word(stripped, start_word)
    return stripped


def strip_trailing_word(text: str, word: str, threshold: float = 0.72) -> str:
    """Drop the last word of `text` when it is (probably) the command word."""
    words = (text or "").split()
    if len(words) > 1 and similar(words[-1], word, threshold):
        return " ".join(words[:-1])
    return text


# Multi-word phrases first: the matcher walks the word list and prefers the
# longest phrase that fits, so "точка с запятой" never becomes ", ;".
_SPOKEN_PUNCT = (
    ("точка с запятой", ";"),
    ("вопросительный знак", "?"),
    ("восклицательный знак", "!"),
    ("с новой строки", "\n"),
    ("новая строка", "\n"),
    ("с красной строки", "\n"),
    ("с нового абзаца", "\n\n"),
    ("новый абзац", "\n\n"),
    ("открыть скобку", "("),
    ("закрыть скобку", ")"),
    ("открыть кавычки", " «"),
    ("закрыть кавычки", "»"),
    ("двоеточие", ":"),
    ("многоточие", "…"),
    ("запятая", ","),
    ("точка", "."),
    ("тире", " — "),
    ("дефис", "-"),
    ("плюс", "+"),
    ("равно", "="),
    ("процент", "%"),
)

_NO_SPACE_BEFORE = set(",.;:!?…»)")
_NO_SPACE_AFTER = set("(«")


def apply_spoken_punctuation(text: str) -> str:
    """Replace spoken punctuation ("запятая", "точка"…) with real marks.

    Applied to final segment texts (Vosk-only mode and Whisper corrections).
    Whisper already inserts punctuation, so spoken words usually will not
    survive to this point in hybrid mode — the mapping is for the offline path.
    """
    words = (text or "").split()
    if not words:
        return text or ""
    out: list[str] = []
    i = 0
    while i < len(words):
        piece = None
        for n in (3, 2, 1):
            if i + n > len(words):
                continue
            phrase = " ".join(words[i:i + n])
            for spoken, mark in _SPOKEN_PUNCT:
                if normalize(phrase) == normalize(spoken):
                    piece = (n, mark)
                    break
            if piece:
                break
        if piece:
            n, mark = piece
            out.append(mark)
            i += n
        else:
            out.append(words[i])
            i += 1

    buf = ""
    for token in out:
        if token in _NO_SPACE_BEFORE or (token.startswith(" ") and token.strip() in _NO_SPACE_BEFORE):
            buf += token.lstrip() if token.strip() in _NO_SPACE_BEFORE else token
            continue
        if token == "\n" or token == "\n\n":
            buf += token
            continue
        if token == " — ":
            buf = buf.rstrip() + " — "
            continue
        if token.startswith(" «"):
            buf = buf.rstrip() + " «"
            continue
        if buf and not buf.endswith((" ", "\n", "(", "«")) and token not in _NO_SPACE_BEFORE:
            buf += " "
        buf += token
    return buf.strip()


def capitalize_sentences(text: str) -> str:
    """Capitalize the first letter of every sentence (Vosk-only mode output)."""
    out = []
    cap = True
    for ch in text or "":
        if cap and ch.isalpha():
            out.append(ch.upper())
            cap = False
        else:
            out.append(ch)
        if ch in ".!?\n":
            cap = True
    return "".join(out)
