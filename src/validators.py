"""Field-level validators / normalizers.

Used both by `schema.CallerInfo` (production output) and by `evaluate`
(comparison against ground truth). Centralised here so prediction-side and
evaluation-side normalization stay byte-identical — this is what stops the
pipeline losing points to formatting mismatches like "+49 172 84492" vs
"+4917284492" or "Jurgen" vs "Jürgen".
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import phonenumbers


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def normalize_name(value: str) -> str:
    """NFC-normalise + strip; keep diacritics intact (Jürgen stays Jürgen)."""
    if not value:
        return ""
    return unicodedata.normalize("NFC", value).strip()


def name_match_key(value: str) -> str:
    """Canonical key for comparison: NFC, casefold, collapse internal whitespace."""
    if not value:
        return ""
    cleaned = unicodedata.normalize("NFC", value).strip().casefold()
    return re.sub(r"\s+", " ", cleaned)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _country_dialcode(region: str) -> str:
    """Return the dialing prefix for a phonenumbers region (e.g. 'DE' → '49')."""
    try:
        return str(phonenumbers.country_code_for_region(region))
    except Exception:
        return "49"


def normalize_email(value: str) -> str:
    """Lowercase + strip. We don't rewrite the local-part — formatting was
    already enforced upstream by the extraction prompt."""
    if not value:
        return ""
    return value.strip().lower()


def is_valid_email_shape(value: str) -> bool:
    return bool(_EMAIL_RE.match(value or ""))


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------
def normalize_phone(value: str, default_region: str = "DE") -> str:
    """Parse with the `phonenumbers` library and emit E.164 ("+CCNNNNNNNNN").

    Falls back to a hand-rolled prefix fixer if the parse fails (e.g. an LLM
    emitted "+04..." or "+0049..."). Empty input returns "".
    """
    if not value:
        return ""

    raw = re.sub(r"[\s\-()]", "", value)

    # Hand-rolled prefix repair to give phonenumbers a valid string to chew on
    if raw.startswith("+00"):
        raw = "+" + raw[3:]      # "+00CC..." → "+CC..."
    elif raw.startswith("+0"):
        raw = "+" + raw[2:]      # "+0CC..."  → "+CC..."  (drops a stray 0 between + and CC)
    elif raw.startswith("00"):
        raw = "+" + raw[2:]      # "00CC..."  → "+CC..."
    elif raw.startswith("0"):
        raw = f"+{int(_country_dialcode(default_region))}" + raw[1:]   # national format → international
    elif not raw.startswith("+"):
        raw = "+" + raw

    # Stutter (final repeated digit at unusually long lengths)
    digits = raw[1:]
    if len(digits) >= 14 and digits[-1] == digits[-2]:
        raw = "+" + digits[:-1]

    try:
        parsed = phonenumbers.parse(raw, default_region)
        if phonenumbers.is_valid_number(parsed) or phonenumbers.is_possible_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        pass

    # Final fallback: just return the prefix-repaired form
    return raw


def is_valid_phone(value: str, default_region: str = "DE") -> bool:
    if not value:
        return False
    try:
        parsed = phonenumbers.parse(value, default_region)
        return phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException:
        return False


# ---------------------------------------------------------------------------
# Comparison helpers used by the evaluator
# ---------------------------------------------------------------------------
def field_match(predicted: str, ground_truth, *, key: Optional[str] = None) -> bool:
    """Compare a predicted value against a ground-truth value (str or list).

    Each field gets a normalisation tailored to its tolerable variation:
      - email          → lowercase + strip
      - phone_number   → E.164
      - first_name / last_name → NFC + casefold + whitespace-collapse
    """
    if isinstance(ground_truth, list):
        return any(field_match(predicted, gt, key=key) for gt in ground_truth)

    p = predicted or ""
    g = ground_truth or ""

    if key == "email":
        return normalize_email(p) == normalize_email(g)
    if key == "phone_number":
        return normalize_phone(p) == normalize_phone(g)
    if key in ("first_name", "last_name"):
        return name_match_key(p) == name_match_key(g)
    return p.strip() == g.strip()
