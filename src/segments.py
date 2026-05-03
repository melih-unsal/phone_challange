"""Detect email / phone segments in a word-timestamped transcript.

Used by Layer 4.5 (segment-targeted re-listen) to find where in the audio
the caller is spelling their email or dictating their phone, so we can crop
those windows and re-prompt the audio LLM with the field's rules in front
of it.

Both detectors return a list of `(start_s, end_s)` time spans, sorted and
non-overlapping. They are deliberately conservative:
- a span is only emitted when at least `min_indicators` evidence words are
  found, AND
- the span is widened with a small contextual padding on each side so the
  whole utterance is captured even when a few words at the boundary aren't
  themselves "indicators".

When the transcriber doesn't expose word timestamps (Voxtral), the caller
can pass `words=None` and these functions return `[]` - Layer 4.5 falls
back to whole-audio prompts.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from src.transcription import Word


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
GERMAN_DIGIT_WORDS = {
    "null", "eins", "ein", "eine", "zwei", "drei", "vier", "fünf", "fuenf",
    "sechs", "sieben", "acht", "neun", "zehn",
    "elf", "zwölf", "zwoelf",
    "zwanzig", "dreißig", "dreissig", "vierzig", "fünfzig", "fuenfzig",
    "sechzig", "siebzig", "achtzig", "neunzig",
    "hundert", "tausend",
    "und",  # "ein und vierzig" → 41
}

EMAIL_SEPARATOR_WORDS = {"punkt", "bindestrich", "unterstrich", "dot", "dash", "hyphen"}
EMAIL_AT_WORDS = {"ät", "at", "@", "atzeichen"}
EMAIL_DOMAIN_HINTS = {
    "gmail", "gmx", "web", "hotmail", "outlook", "yahoo", "t-online", "tonline",
    "freenet", "icloud", "mail",
    "de", "com", "net", "org", "eu", "fr", "es", "it", "uk", "ch", "at",
}

PHONE_PREFIX_WORDS = {"plus", "vorwahl", "ländervorwahl", "laendervorwahl"}


# ---------------------------------------------------------------------------
# Word classification helpers
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\wäöüßÄÖÜẞ]+")


def _norm(token: str) -> str:
    return _PUNCT_RE.sub("", token or "").strip().lower()


def _is_digit(token: str) -> bool:
    t = _norm(token)
    return t.isdigit() or t in GERMAN_DIGIT_WORDS


def _is_phone_indicator(token: str) -> bool:
    t = _norm(token)
    return t in PHONE_PREFIX_WORDS or _is_digit(t)


def _is_single_letter(token: str) -> bool:
    t = _norm(token)
    return len(t) == 1 and t.isalpha()


def _is_email_indicator(token: str) -> bool:
    t = _norm(token)
    if not t:
        return False
    if t in EMAIL_SEPARATOR_WORDS or t in EMAIL_AT_WORDS:
        return True
    if t in EMAIL_DOMAIN_HINTS:
        return True
    if "@" in (token or "") or t.startswith(".") or t.endswith(".de") or t.endswith(".com"):
        return True
    return False


# ---------------------------------------------------------------------------
# Segment detection
# ---------------------------------------------------------------------------
def _merge_runs(
    flags: list[bool],
    words: list[Word],
    *,
    max_gap: int,
    min_run_indicators: int,
    pad_words: int,
) -> list[tuple[float, float]]:
    """Walk a boolean flag array and emit time spans for runs of True values.

    `max_gap` lets a few non-indicator words slip into a run without breaking it
    (e.g. "j … punkt … meyer … ät … gmail" - the names between "punkt"/"ät"
    aren't indicators, but they're inside the segment).
    `min_run_indicators` filters out tiny coincidental matches.
    `pad_words` widens the span by N words on each side for context.
    """
    n = len(words)
    spans: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        # Start of a candidate run; greedily extend while gaps are small.
        run_start = i
        run_end = i
        indicator_count = 1 if flags[i] else 0
        j = i + 1
        gap = 0
        while j < n:
            if flags[j]:
                run_end = j
                indicator_count += 1
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        if indicator_count >= min_run_indicators:
            lo = max(0, run_start - pad_words)
            hi = min(n - 1, run_end + pad_words)
            spans.append((words[lo].start, words[hi].end))
        i = run_end + 1
    return _merge_overlapping(spans)


def _merge_overlapping(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def find_phone_segments(words: Optional[list[Word]]) -> list[tuple[float, float]]:
    """Time spans where the caller is dictating a phone number."""
    if not words:
        return []
    flags = [_is_phone_indicator(w.text) for w in words]
    return _merge_runs(
        flags, words,
        max_gap=2,            # tolerate "und" / fillers between digits
        min_run_indicators=6, # phone has many digits
        pad_words=2,
    )


def find_email_segments(words: Optional[list[Word]]) -> list[tuple[float, float]]:
    """Time spans where the caller is spelling out an email."""
    if not words:
        return []
    # Indicator: explicit email words, OR a single-letter word (likely letter-by-letter spelling).
    flags = [
        _is_email_indicator(w.text) or _is_single_letter(w.text)
        for w in words
    ]
    return _merge_runs(
        flags, words,
        max_gap=3,            # whole names ("meyer") between separator words
        min_run_indicators=2, # at least two indicators (e.g. "punkt" + "at")
        pad_words=3,
    )


def merge_or_fallback(
    spans: list[tuple[float, float]],
    audio_duration: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    """Collapse multiple segments into one (start of first → end of last) so the
    audio-LLM gets a single contiguous crop rather than a stitch of pieces.

    Returns None when no segments were found - caller falls back to full audio.
    """
    if not spans:
        return None
    return (spans[0][0], spans[-1][1])
