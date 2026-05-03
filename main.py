"""End-to-end pipeline orchestrator.

Architecture:
  Layer 1   - Audio transcription                       (src/transcription.py)
  Layer 2   - Country / linguistic-origin detection     (src/country.py)
  Layer 3   - Candidate 1: self-consistent LLM          (src/extract_llm.py)
  Layer 4   - Candidate 2: audio LLM, whole-record      (src/extract_voxtral.py)
  Layer 4.5 - Targeted re-listen for email + phone      (src/targeted_extract.py)
  Layer 5   - Reconciler (transcript + cand1 + cand2 + targeted)
                                                        (src/reconcile.py)
  Layer 6   - Schema validation + normalisation         (src/schema.py)

Audio backends are model-agnostic. Pick them in src/config.py:
    TRANSCRIPTION_MODEL = "voxtral:mistralai/Voxtral-Mini-3B-2507"
    TRANSCRIPTION_MODEL = "elevenlabs:scribe_v2"
    INSTRUCT_MODEL      = "voxtral:mistralai/Voxtral-Mini-3B-2507"

Each run writes a timestamped structured report to RESULTS_DIR/.
"""

# faulthandler MUST go before anything that loads torch/CUDA - it installs
# a SIGSEGV / SIGABRT handler that prints a Python + C stack trace, which
# is the only way to diagnose native crashes in transformers / soundfile / cuda.
import faulthandler
faulthandler.enable()

# Prevent transformers' tokenizer thread pool from clashing with httpx /
# requests inside the elevenlabs SDK - a known cause of intermittent
# segfaults when an HTTP call follows a HF-tokenizer-touched code path.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Mute the transformers / pytorch advisory warnings that flood every
# Whisper invocation ("Using `chunk_length_s` is very experimental..." etc).
# Set HF_VERBOSITY=info if you want them back.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import json
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.config import (
    AUTO_RESUME,
    DEFAULT_PHONE_REGION,
    GROUND_TRUTH_PATH,
    INSTRUCT_MODEL,
    KEYS,
    LLM_MODEL_NAME,
    NUM_INSTRUCT_VOTES,
    NUM_LLM_VOTES,
    NUM_TARGETED_VOTES,
    PROMPT_VERSION,
    RECORDINGS_DIR,
    RESULTS_DIR,
    TIMESTAMP_MODEL,
    TRANSCRIPTION_LANGUAGE,
    TRANSCRIPTION_MODELS,
    USE_TARGETED_EXTRACTION,
)
from src.country import build_country_chain, detect_country
from src.evaluate import evaluate
from src.extract_llm import build_extract_chain, extract_candidate_1
from src.extract_voxtral import extract_candidate_2
from src.reconcile import build_reconcile_chain, reconcile
from src.schema import validate_caller_info
from src.targeted_extract import targeted_extract
from src.transcription import (
    Transcription,
    build_instructor,
    build_timestamper,
    build_transcriber,
)


def _timestamped_report_path() -> Path:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(RESULTS_DIR) / f"report_{stamp}.json"


def _build_run_metadata(num_recordings: int, instructor_available: bool, timestamper_available: bool) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "models": {
            "transcription": list(TRANSCRIPTION_MODELS),
            "instruct": INSTRUCT_MODEL if instructor_available else f"{INSTRUCT_MODEL} (UNAVAILABLE)",
            "timestamp": TIMESTAMP_MODEL if timestamper_available else (TIMESTAMP_MODEL or "(disabled)"),
            "llm": LLM_MODEL_NAME,
        },
        "settings": {
            "num_llm_votes": NUM_LLM_VOTES,
            "num_instruct_votes": NUM_INSTRUCT_VOTES,
            "num_targeted_votes": NUM_TARGETED_VOTES,
            "use_targeted_extraction": USE_TARGETED_EXTRACTION and instructor_available,
            "transcription_language": TRANSCRIPTION_LANGUAGE,
            "default_phone_region": DEFAULT_PHONE_REGION,
        },
        "num_recordings": num_recordings,
        "recordings_dir": RECORDINGS_DIR,
        "ground_truth": GROUND_TRUTH_PATH if Path(GROUND_TRUTH_PATH).exists() else None,
    }


def _find_resumable_report(num_recordings: int) -> tuple[Path, dict] | None:
    """Find the most recent report in RESULTS_DIR that hasn't completed yet.

    A report is "resumable" when it has fewer than `num_recordings` entries
    and no `evaluation` block. Returns (path, report_dict) or None.
    """
    rdir = Path(RESULTS_DIR)
    if not rdir.exists():
        return None
    for path in sorted(rdir.glob("report_*.json"), reverse=True):
        try:
            with open(path) as f:
                report = json.load(f)
        except Exception:
            continue
        if not isinstance(report, dict):
            continue
        recs = report.get("recordings", [])
        if report.get("evaluation") is None and 0 < len(recs) < num_recordings:
            return path, report
    return None


def _models_compatible(prev_run: dict) -> bool:
    """Refuse to resume if the model config has changed since the previous run."""
    prev = prev_run.get("models", {})
    prev_trans = prev.get("transcription")
    if isinstance(prev_trans, str):
        prev_trans = [prev_trans]
    return (
        list(prev_trans or []) == list(TRANSCRIPTION_MODELS)
        and prev.get("instruct", "").split(" ")[0] == INSTRUCT_MODEL
        and prev.get("llm") == LLM_MODEL_NAME
    )


def _per_key_agreement(candidates_1: list[dict], cand2: dict, targeted: dict | None) -> dict:
    """Cheap per-call observability: how many candidates agree on each key,
    treating empty / missing values as abstentions. `candidates_1` is the
    list of {transcriber, value} entries (one per STT model)."""
    agreement = {}
    for k in KEYS:
        values = [c.get("value", {}).get(k, "") for c in candidates_1]
        values.append(cand2.get(k, ""))
        if targeted and k in ("email", "phone_number"):
            values.append(targeted.get(k, ""))
        non_empty = [v for v in values if v]
        unique = set(non_empty)
        agreement[k] = {
            "n_candidates": len(non_empty),
            "n_unique": len(unique),
            "unanimous": len(unique) == 1 and len(non_empty) >= 2,
        }
    return agreement


def _write_report(path: Path, report: dict) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False, default=str)


def run() -> Path:
    # Build all configured transcribers. Each will produce its own
    # transcript per recording, and Layer 3 will derive an independent
    # candidate from each.
    transcribers = []
    for spec in TRANSCRIPTION_MODELS:
        try:
            transcribers.append(build_transcriber(spec))
        except Exception as e:
            print(f"WARNING: could not build transcriber {spec!r}: {e}")
    if not transcribers:
        raise RuntimeError("No transcribers could be built - check TRANSCRIPTION_MODELS.")

    # Layer 4 / 4.5 are best-effort: if the audio-instructor backend can't load
    # (broken cuda driver, missing API key, etc.) we keep going with only
    # candidate 1 so the rest of the pipeline still produces a report.
    try:
        instructor = build_instructor(INSTRUCT_MODEL)
    except Exception as e:
        print(f"WARNING: could not build INSTRUCT_MODEL={INSTRUCT_MODEL!r}: {e}")
        print("         Layer 4 (candidate 2) and Layer 4.5 (targeted re-listen) will be skipped.")
        instructor = None

    # Layer 1b - dedicated timestamper. Only built when no transcriber
    # provides word timestamps AND TIMESTAMP_MODEL is set.
    timestamper = None
    if TIMESTAMP_MODEL:
        try:
            timestamper = build_timestamper(TIMESTAMP_MODEL)
        except Exception as e:
            print(f"WARNING: could not build TIMESTAMP_MODEL={TIMESTAMP_MODEL!r}: {e}")
            print("         Layer 4.5 will fall back to whole-audio prompts.")
            timestamper = None

    country_chain = build_country_chain()
    extract_chain = build_extract_chain()
    reconcile_chain = build_reconcile_chain()

    recordings = sorted(
        f for f in os.listdir(RECORDINGS_DIR) if f.lower().endswith(".wav")
    )

    # ------------------------------------------------------------------
    # Auto-resume: if the previous run crashed mid-pipeline, pick up
    # where it left off instead of throwing away the work.
    # ------------------------------------------------------------------
    resume = _find_resumable_report(len(recordings)) if AUTO_RESUME else None
    if resume and not _models_compatible(resume[1].get("run", {})):
        print(f"Found {resume[0]} but model config changed; starting fresh.")
        resume = None

    if resume:
        output_path, report = resume
        done_ids = {r["id"] for r in report.get("recordings", [])}
        print(f"Resuming {output_path}: {len(done_ids)}/{len(recordings)} already done.")
    else:
        output_path = _timestamped_report_path()
        report = {
            "run": _build_run_metadata(len(recordings), instructor is not None, timestamper is not None),
            "recordings": [],
            "evaluation": None,
        }
        done_ids = set()

    do_targeted = USE_TARGETED_EXTRACTION and instructor is not None

    pending = [f for f in recordings if f.split(".")[0] not in done_ids]
    pbar = tqdm(
        pending,
        desc="Processing recordings",
        total=len(recordings),
        initial=len(done_ids),
    )
    for filename in pbar:
        path = os.path.join(RECORDINGS_DIR, filename)
        per_call_timings: dict[str, float] = {}

        # Layer 1 - transcribe with EVERY configured STT model.
        t0 = time.perf_counter()
        transcriptions: list[Transcription] = []
        for tr in transcribers:
            try:
                transcriptions.append(tr.transcribe(path, language=TRANSCRIPTION_LANGUAGE))
            except Exception as e:
                print(f"WARNING: transcriber {tr.name!r} failed on {filename}: {e}")
        if not transcriptions:
            print(f"ERROR: every transcriber failed on {filename}; skipping")
            continue
        per_call_timings["transcription_s"] = round(time.perf_counter() - t0, 2)

        # Layer 1b - fill in word timestamps if no transcriber provided them
        # (Voxtral doesn't, ElevenLabs Scribe does). Used by Layer 4.5 cropping.
        words = next((t.words for t in transcriptions if t.words), None)
        if words is None and timestamper is not None:
            t0 = time.perf_counter()
            try:
                words = timestamper.timestamp(path, language=TRANSCRIPTION_LANGUAGE)
            except Exception as e:
                print(f"WARNING: timestamper failed on {filename}: {e}")
                words = None
            per_call_timings["timestamp_s"] = round(time.perf_counter() - t0, 2)

        # Pick the first transcript as the "primary" for Layer 2 country detection.
        primary_text = transcriptions[0].text

        # Layer 2 - country / linguistic origin
        t0 = time.perf_counter()
        country_info = detect_country(country_chain, primary_text)
        per_call_timings["country_s"] = round(time.perf_counter() - t0, 2)

        # Layer 3 - candidate per transcript (independent extraction × N each)
        t0 = time.perf_counter()
        candidates_1: list[dict] = []
        for trans in transcriptions:
            cand = extract_candidate_1(
                extract_chain,
                trans.text,
                country_info["country"],
                country_info["language_code"],
            )
            candidates_1.append({
                "transcriber": trans.backend,
                "value": {k: cand.get(k, "") for k in KEYS},
            })
        per_call_timings["candidate_1_s"] = round(time.perf_counter() - t0, 2)

        # Layer 4 - candidate 2 (audio LLM whole-record). Skipped if the
        # instructor failed to build at startup.
        t0 = time.perf_counter()
        if instructor is not None:
            cand2 = extract_candidate_2(instructor, path)
        else:
            cand2 = {k: "" for k in KEYS}
        per_call_timings["candidate_2_s"] = round(time.perf_counter() - t0, 2)

        # Layer 4.5 - segment-targeted re-listen. Only run on fields where
        # any whole-record candidate disagrees with another. Most records
        # end up unanimous → skip the layer (and Whisper) entirely.
        t0 = time.perf_counter()
        all_emails = {(c["value"].get("email") or "").strip() for c in candidates_1}
        all_emails.add((cand2.get("email") or "").strip())
        all_phones = {(c["value"].get("phone_number") or "").strip() for c in candidates_1}
        all_phones.add((cand2.get("phone_number") or "").strip())
        run_email = len({e for e in all_emails if e}) > 1
        run_phone = len({p for p in all_phones if p}) > 1
        if do_targeted and (run_email or run_phone):
            targeted = targeted_extract(
                instructor, path, words=words,
                run_email=run_email, run_phone=run_phone,
            )
        else:
            targeted = None
        per_call_timings["targeted_s"] = round(time.perf_counter() - t0, 2)

        # Layer 5 - reconciler (deterministic safety nets applied inside)
        t0 = time.perf_counter()
        final = reconcile(
            reconcile_chain, transcriptions, country_info,
            candidates_1, cand2, targeted,
        )
        per_call_timings["reconcile_s"] = round(time.perf_counter() - t0, 2)

        # Layer 6 - schema validation + normalisation
        validated = validate_caller_info(final)
        info = validated.info

        expected = {
            "first_name": info.first_name,
            "last_name": info.last_name,
            "email": info.email,
            "phone_number": info.phone_number,
        }

        # Always emit a targeted block with explicit mode = cropped|full_audio|skipped|skipped_unanimous
        if targeted is None:
            targeted_block = {
                "email": "",
                "phone_number": "",
                "email_meta": {"mode": "skipped"},
                "phone_meta": {"mode": "skipped"},
            }
        else:
            targeted_block = {
                "email": targeted.get("email", ""),
                "phone_number": targeted.get("phone_number", ""),
                "email_meta": targeted.get("_email_meta", {}),
                "phone_meta": targeted.get("_phone_meta", {}),
            }

        report["recordings"].append({
            "id": filename.split(".")[0],
            "file": filename,
            "expected": expected,
            "transcripts": [
                {
                    "backend": t.backend,
                    "text": t.text,
                    "has_timestamps": t.has_timestamps(),
                }
                for t in transcriptions
            ],
            "country_info": country_info,
            "candidates_1": candidates_1,
            "candidate_2": {k: cand2.get(k, "") for k in KEYS},
            "targeted": targeted_block,
            "reasoning": final.get("reasoning", ""),
            "agreement": _per_key_agreement(candidates_1, cand2, targeted),
            "warnings": validated.warnings,
            "timings": per_call_timings,
        })

        # Periodic checkpoint so a crash doesn't lose everything
        _write_report(output_path, report)

    # Final evaluation against ground truth
    if Path(GROUND_TRUTH_PATH).exists():
        print()
        summary = evaluate(str(output_path), GROUND_TRUTH_PATH)
        report["evaluation"] = summary
        _write_report(output_path, report)
    else:
        print(f"\nresults written to {output_path} (no {GROUND_TRUTH_PATH} found, skipping evaluation)")

    print(f"\nReport written to: {output_path}")
    return output_path


if __name__ == "__main__":
    run()
