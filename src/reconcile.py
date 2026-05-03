"""Layer 5 — final reconciler.

Sees: transcripts (one per STT model) + country info + a list of candidate_1
records (one per transcript) + candidate_2 (audio LLM whole-record) + targeted
re-listen.
Produces: a reasoning paragraph + the final caller-info JSON.

Four deterministic safety nets wrap the LLM call:

1. `_filter_invalid_targeted` (BEFORE the LLM) — drops a targeted re-listen
   value if it looks structurally broken (email without "@" + TLD, phone that
   doesn't parse). Stops a truncated crop from hijacking the choice.

2. `_unanimous_override` (AFTER the LLM) — if every whole-record candidate
   (each candidate_1 + candidate_2) agrees on a key, that value is locked
   regardless of what the LLM picked.

3. `_majority_override` (AFTER `_unanimous_override`) — when whole-record
   candidates have a strict majority (e.g. 2 of 3 agree on a value), that
   majority wins. Catches cases where the LLM was charmed by a confident-
   wrong targeted re-listen (call_09 phone, call_26 email): the targeted
   said one thing, the audio LLM agreed with it, but two transcript-LLMs
   agreed on the correct value.

4. `_email_name_alignment` (AFTER `_majority_override`) — NAME ↔ EMAIL
   CONSISTENCY: if candidates disagree on a name AND their ASCII forms
   actually differ (i.e. the disagreement is real letter-substitution, not
   just diacritics), and the FINAL email's local-part has a token within
   edit-distance 2 of every candidate spelling, replace the name with the
   email's token. The ASCII-distinct guard prevents this from stripping
   accents on cases like "García"/"Garcia" where 2/3 candidates correctly
   produced the accented form.
"""

import json
import re
import unicodedata

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import KEYS, LLM_BASE_URL, LLM_MODEL_NAME
from src.prompts import RECONCILE_HUMAN, RECONCILE_SYSTEM
from src.validators import is_valid_email_shape, is_valid_phone


def build_reconcile_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", RECONCILE_SYSTEM), ("human", RECONCILE_HUMAN)]
    )
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0,
        max_tokens=800,
        base_url=LLM_BASE_URL,
    )
    return prompt | llm | JsonOutputParser()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _candidate_to_str(cand: dict) -> str:
    payload = {k: cand.get(k, "") for k in KEYS}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _candidates_1_to_str(candidates_1: list[dict]) -> str:
    """Render a list of [{transcriber, value}] entries for the prompt."""
    if not candidates_1:
        return "(no transcript-LLM candidates)"
    lines = []
    for entry in candidates_1:
        src = entry.get("transcriber", "?")
        payload = {k: entry.get("value", {}).get(k, "") for k in KEYS}
        lines.append(f"  [{src}]\n  " + json.dumps(payload, ensure_ascii=False, indent=2).replace("\n", "\n  "))
    return "\n".join(lines)


def _transcripts_to_str(transcriptions: list) -> str:
    if not transcriptions:
        return "(no transcripts)"
    return "\n".join(
        f"  [{getattr(t, 'backend', '?')}]: {getattr(t, 'text', '')}"
        for t in transcriptions
    )


def _is_empty_candidate(cand: dict) -> bool:
    return not any((cand.get(k) or "").strip() for k in KEYS)


def _filter_invalid_targeted(targeted: dict | None) -> dict | None:
    """Strip targeted values that are structurally invalid so the reconciler
    doesn't even see them. A truncated crop yielding 'marie.lefevre@' is worse
    than no signal at all."""
    if not targeted:
        return targeted
    out = dict(targeted)
    email = (out.get("email") or "").strip()
    if email and not is_valid_email_shape(email):
        out["email"] = ""
    phone = (out.get("phone_number") or "").strip()
    if phone and not is_valid_phone(phone):
        out["phone_number"] = ""
    return out


def _targeted_to_str(targeted: dict | None) -> str:
    if not targeted:
        return "(targeted re-listen not run for this record)"
    email = (targeted.get("email") or "").strip()
    phone = (targeted.get("phone_number") or "").strip()
    if not email and not phone:
        return "(targeted re-listen returned no usable values)"
    payload = {"email": email, "phone_number": phone}
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Deterministic safety nets
# ---------------------------------------------------------------------------
def _unanimous_override(final: dict, all_candidates: list[dict]) -> dict:
    """If every non-empty candidate agrees on a key, that value wins. Runs
    AFTER the LLM so we can guarantee it instead of trusting the prompt.
    `all_candidates` is the flat list: each candidate_1 + candidate_2.
    """
    out = dict(final)
    for k in KEYS:
        values = [(c.get(k) or "").strip() for c in all_candidates if c]
        non_empty = [v for v in values if v]
        if non_empty and all(v == non_empty[0] for v in non_empty):
            if out.get(k, "") != non_empty[0]:
                out[k] = non_empty[0]
    return out


def _majority_override(final: dict, all_candidates: list[dict]) -> dict:
    """If a STRICT majority of the whole-record candidates agree on a key
    (more than half, treating empties as abstentions), that value wins.
    Runs AFTER `_unanimous_override`, so it only matters when at least one
    candidate dissented. Catches cases where the LLM was swayed by a
    confident-wrong targeted re-listen but the transcript-LLMs already
    converged on the right answer."""
    out = dict(final)
    for k in KEYS:
        values = [(c.get(k) or "").strip() for c in all_candidates if c]
        non_empty = [v for v in values if v]
        if len(non_empty) < 2:
            continue
        counts: dict[str, int] = {}
        for v in non_empty:
            counts[v] = counts.get(v, 0) + 1
        max_v, max_c = max(counts.items(), key=lambda x: x[1])
        # Strict majority: more than half.
        if max_c > len(non_empty) / 2 and out.get(k, "") != max_v:
            out[k] = max_v
    return out


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,                                  # delete
                curr[j - 1] + 1,                              # insert
                prev[j - 1] + (0 if ca == cb else 1),         # substitute
            ))
        prev = curr
    return prev[-1]


def _email_local_runs(email: str) -> list[str]:
    """Alphabetic runs (length ≥ 2) inside the email's local-part, lowercased ASCII."""
    if "@" not in email:
        return []
    local = email.split("@", 1)[0].lower()
    runs: list[str] = []
    for token in re.split(r"[._\-]", local):
        for m in re.finditer(r"[a-z]+", token):
            run = m.group(0)
            if len(run) >= 2:
                runs.append(run)
    return runs


def _ascii_fold(value: str) -> str:
    """Strip diacritics and lowercase. 'García' → 'garcia', 'Le Fèvre' → 'le fevre'."""
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFD", value)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


def _email_name_alignment(
    final: dict,
    all_candidates: list[dict],
    max_edit_distance: int = 2,
) -> dict:
    """NAME ↔ EMAIL CONSISTENCY, applied deterministically.

    Triggers when:
      - candidates disagree on a name field, AND
      - their ASCII-folded forms also differ (i.e. it's a real letter
        substitution, NOT a pure diacritic disagreement — without this guard
        we'd strip the accent on cases like 'García' / 'García' / 'Garcia'
        because 'garcia' is within edit-distance 2 of every candidate), AND
      - the FINAL email's local-part contains a token within edit-distance 2
        of every candidate's ASCII-folded spelling.

    When all three hold, the email's token (capitalised) replaces the name.

    Worked example for call_25:
        cand_1A.last_name = 'Lefebvre'
        cand_1B.last_name = 'Lefevre'
        cand_2.last_name  = 'Le Fèvre'
        final.email       = 'marie.lefevre@yahoo.fr'   (LLM picked correctly)
    ASCII-folded names: {'lefebvre', 'lefevre', 'le fevre'} → distinct → proceed.
    Edit distance from 'lefevre' (email run) to each: [1, 0, 1] — all ≤ 2.
    → final.last_name = 'Lefevre'  (was 'Lefebvre' from the LLM).
    """
    final_email = (final.get("email") or "").strip().lower()
    if "@" not in final_email:
        return final
    runs = _email_local_runs(final_email)
    if not runs:
        return final

    out = dict(final)
    for name_key in ("first_name", "last_name"):
        names = [(c.get(name_key) or "").strip() for c in all_candidates if c]
        names = [n for n in names if n]
        if not names:
            continue

        # No real disagreement → leave alone (unanimous/majority already handled).
        if len({n.casefold() for n in names}) <= 1:
            continue

        # Diacritic-only disagreement → don't strip accents based on the
        # ASCII email. The country-aware orthography rule wants the accent.
        if len({_ascii_fold(n) for n in names}) <= 1:
            continue

        # Find an email-token within edit-distance 2 of EVERY candidate
        # (using ASCII-folded comparison so accents don't inflate the distance).
        for run in runs:
            distances = [_edit_distance(_ascii_fold(n), run) for n in names]
            if all(d <= max_edit_distance for d in distances):
                out[name_key] = run.capitalize()
                break
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def reconcile(
    chain,
    transcriptions: list,
    country_info: dict,
    candidates_1: list[dict],
    candidate_2: dict,
    targeted: dict | None = None,
) -> dict:
    """`candidates_1` is a list of {"transcriber": str, "value": dict}; one
    entry per TRANSCRIPTION_MODELS spec."""
    targeted = _filter_invalid_targeted(targeted)

    if _is_empty_candidate(candidate_2):
        cand2_str = "(unavailable for this run — rely on candidate_1 list)"
    else:
        cand2_str = _candidate_to_str(candidate_2)

    # Flat list used by the deterministic safety nets.
    all_whole_record = [c.get("value", {}) for c in candidates_1 if c] + [candidate_2]

    try:
        result = chain.invoke({
            "country": country_info.get("country", ""),
            "language_code": country_info.get("language_code", ""),
            "transcripts": _transcripts_to_str(transcriptions),
            "candidates_1": _candidates_1_to_str(candidates_1),
            "candidate_2": cand2_str,
            "targeted": _targeted_to_str(targeted),
        })
    except Exception:
        # Fallback: take the first candidate_1 verbatim, no reasoning
        first = candidates_1[0]["value"] if candidates_1 else {}
        result = {k: first.get(k, "") for k in KEYS}
        result["reasoning"] = "reconciler failed; fell back to first candidate"
        result = _unanimous_override(result, all_whole_record)
        result = _majority_override(result, all_whole_record)
        result = _email_name_alignment(result, all_whole_record)
        return result

    if not isinstance(result, dict):
        first = candidates_1[0]["value"] if candidates_1 else {}
        result = {k: first.get(k, "") for k in KEYS}
        result["reasoning"] = "reconciler returned non-dict; fell back to first candidate"
    else:
        for k in KEYS:
            result.setdefault(k, "")
        result.setdefault("reasoning", "")

    # Deterministic safety nets, applied in order. Each net only modifies
    # the keys it has confidence about; later nets see the (possibly fixed)
    # output of earlier nets.
    result = _unanimous_override(result, all_whole_record)
    result = _majority_override(result, all_whole_record)
    result = _email_name_alignment(result, all_whole_record)
    return result
