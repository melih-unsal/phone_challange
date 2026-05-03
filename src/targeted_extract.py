"""Layer 4.5 - Segment-targeted re-listening for email + phone.

The audio LLM is re-prompted with two field-focused prompts. When the
transcriber exposed word timestamps (e.g. ElevenLabs Scribe), we use
`src/segments.py` to locate the relevant audio window and crop a small
±pad-second slice via `src/audio_crop.py`, so the model only listens to
the part of the call that matters for the field.

When timestamps aren't available (Voxtral transcription), we fall back to
prompting the whole audio with the same field-focused prompt - still better
than the whole-record prompt, but without the rewind benefit.

Self-consistency: each prompt is run NUM_TARGETED_VOTES times and majority-voted.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from src.audio_crop import cropped_audio
from src.config import NUM_TARGETED_VOTES, TARGETED_PAD_SECONDS
from src.prompts import EMAIL_TARGETED_PROMPT, PHONE_TARGETED_PROMPT
from src.segments import find_email_segments, find_phone_segments, merge_or_fallback
from src.transcription import Word
from src.utils import safe_json_loads


def _vote(values: list[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def _run_field(instructor, audio_path: str, prompt: str, key: str) -> list[str]:
    runs: list[str] = []
    for _ in range(NUM_TARGETED_VOTES):
        try:
            raw = instructor.instruct(audio_path, prompt)
            parsed = safe_json_loads(raw)
            if isinstance(parsed, dict):
                value = parsed.get(key, "") or ""
                if value:
                    runs.append(str(value))
        except Exception:
            pass
    return runs


def _re_listen_one_field(
    instructor,
    audio_path: str,
    spans: list[tuple[float, float]],
    prompt: str,
    key: str,
) -> tuple[str, dict]:
    """Crop the audio to the merged span (if any) and run the prompt.

    Returns (voted_value, metadata) where metadata records which window was
    actually used so we can put it in the report.
    """
    window = merge_or_fallback(spans)
    if window is not None:
        with cropped_audio(audio_path, window[0], window[1], pad_s=TARGETED_PAD_SECONDS) as clip:
            runs = _run_field(instructor, clip, prompt, key)
        meta = {"mode": "cropped", "window_s": [round(window[0], 2), round(window[1], 2)], "n_runs": len(runs)}
    else:
        runs = _run_field(instructor, audio_path, prompt, key)
        meta = {"mode": "full_audio", "window_s": None, "n_runs": len(runs)}
    return _vote(runs), {**meta, "values": runs}


def targeted_extract(
    instructor,
    audio_path: str,
    words: Optional[list[Word]] = None,
    run_email: bool = True,
    run_phone: bool = True,
) -> dict:
    """Returns a dict with email + phone re-listen values plus per-field meta.

    `instructor` may be None - in that case we return empty values so the
    reconciler interprets the targeted re-listen as unavailable.
    `words` is the timestamped word list from Layer 1 (None if Voxtral STT).
    `run_email` / `run_phone` let the orchestrator skip a field when the
    whole-record candidates already agree on it (saves Voxtral inferences
    AND avoids the failure mode where a tight crop overrides an agreeing
    cand_1 == cand_2 with a slightly-misheard value).
    """
    if instructor is None or (not run_email and not run_phone):
        return {
            "email": "", "phone_number": "",
            "_email_meta": {"mode": "skipped"},
            "_phone_meta": {"mode": "skipped"},
        }

    if run_email:
        email_spans = find_email_segments(words)
        email_value, email_meta = _re_listen_one_field(
            instructor, audio_path, email_spans, EMAIL_TARGETED_PROMPT, "email"
        )
    else:
        email_value, email_meta = "", {"mode": "skipped_unanimous"}

    if run_phone:
        phone_spans = find_phone_segments(words)
        phone_value, phone_meta = _re_listen_one_field(
            instructor, audio_path, phone_spans, PHONE_TARGETED_PROMPT, "phone_number"
        )
    else:
        phone_value, phone_meta = "", {"mode": "skipped_unanimous"}

    return {
        "email": email_value,
        "phone_number": phone_value,
        "_email_meta": email_meta,
        "_phone_meta": phone_meta,
    }
