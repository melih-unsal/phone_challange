"""Layer 4 - Candidate 2.

Direct audio instruction-following on the WAV (no transcript involved).
The backend is whichever AudioInstructor was configured (Voxtral by default,
or anything else that implements `.instruct(audio_path, prompt)`).

We run the same prompt NUM_INSTRUCT_VOTES times with light sampling so a
single off-by-one digit doesn't dominate, and majority-vote per key.
"""

from src.config import KEYS, NUM_INSTRUCT_VOTES
from src.prompts import VOXTRAL_DIRECT_PROMPT
from src.utils import majority_vote, safe_json_loads


def extract_candidate_2(instructor, audio_path: str) -> dict:
    """`instructor`: any object exposing `.instruct(audio_path, prompt)`."""
    runs: list[dict] = []
    for _ in range(NUM_INSTRUCT_VOTES):
        try:
            raw = instructor.instruct(audio_path, VOXTRAL_DIRECT_PROMPT)
            parsed = safe_json_loads(raw)
            if isinstance(parsed, dict):
                runs.append({k: str(parsed.get(k, "") or "") for k in KEYS})
        except Exception:
            pass

    voted = majority_vote(runs, KEYS) if runs else {k: "" for k in KEYS}
    voted["_runs"] = runs
    return voted
