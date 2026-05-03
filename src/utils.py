"""Shared helpers: phone normalisation, candidate aggregation, JSON parsing."""

from collections import Counter


def normalize_phone_number(phone_number: str) -> str:
    """Apply the prefix + stutter + spacing rules from config.PHONE_RULES.

    The same algorithm is used post-Voxtral-direct, post-LLM-extraction and
    post-reconcile so the final output is always in canonical form.
    """
    if not phone_number:
        return phone_number
    phone_number = phone_number.replace(" ", "")

    # Prefix normalisation
    if phone_number.startswith("+00"):
        phone_number = "+" + phone_number[3:]
    elif phone_number.startswith("+0"):
        phone_number = "+" + phone_number[2:]
    elif phone_number.startswith("00"):
        phone_number = "+" + phone_number[2:]
    elif phone_number.startswith("0"):
        phone_number = "+49" + phone_number[1:]
    elif not phone_number.startswith("+"):
        phone_number = "+" + phone_number

    # Stutter rule: only triggers when the number is clearly one digit too long
    digits_only = phone_number[1:]
    if len(digits_only) >= 14 and digits_only[-1] == digits_only[-2]:
        phone_number = "+" + digits_only[:-1]

    # "+CC AAA NNNNNNNN" spacing
    k = min(3, len(phone_number) - 11)
    if k > 0:
        phone_number = phone_number[:3] + " " + phone_number[3:3 + k] + " " + phone_number[3 + k:]
    return phone_number


def majority_vote(runs: list[dict], keys: list[str]) -> dict:
    """Per-key majority vote across a list of run dicts."""
    voted = {}
    for k in keys:
        values = [str(r.get(k, "")) for r in runs if isinstance(r, dict)]
        if not values:
            voted[k] = ""
            continue
        voted[k] = Counter(values).most_common(1)[0][0]
    return voted


def safe_json_loads(raw: str) -> dict:
    """Parse a JSON object from LLM/Voxtral output that may be wrapped in
    markdown fences or have leading/trailing prose. Falls back to {}."""
    import json
    import re

    if not raw:
        return {}
    raw = raw.strip()
    # Strip ``` fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Find the first { ... } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}
