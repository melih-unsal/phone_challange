"""Top-level `phonebot` CLI.

Subcommands:

    phonebot run                       - batch over data/recordings/
    phonebot extract <file.wav>        - process a single recording
    phonebot evaluate <report.json>    - re-evaluate an existing report
    phonebot ablate ...                - run an ablation (wraps study.cli)
    phonebot doctor                    - print resolved config + GPU/API status

Every subcommand respects the same env vars as `python main.py`:
    OPENAI_API_KEY, ELEVENLABS_API_KEY,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.

The CLI degrades gracefully when no GPU is present: any GPU-only
backend (Voxtral, Whisper) fails to build, main.py logs a warning,
and the run continues with whatever backends did load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make `main.py` (which lives at the repo root, not inside `src/`)
# importable when phonebot is launched as an installed console script.
# Without this, `import main` in cmd_run/cmd_extract fails with
# ModuleNotFoundError because sys.path[0] is the script directory
# (/usr/local/bin in the container), not REPO_ROOT.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# `phonebot doctor`
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    from src import config

    print("== phonebot doctor ==")
    print(f"  repo root           : {REPO_ROOT}")
    print(f"  python              : {sys.version.split()[0]}")
    # GPU
    try:
        import torch
        cuda = torch.cuda.is_available()
        n = torch.cuda.device_count() if cuda else 0
        print(f"  torch               : {torch.__version__}  cuda={cuda} devices={n}")
        if cuda:
            for i in range(n):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("  torch               : not installed (CPU-only path)")
    # vLLM
    try:
        import vllm
        print(f"  vllm                : {vllm.__version__}")
    except ImportError:
        print("  vllm                : not installed")
    # API keys
    print(f"  OPENAI_API_KEY      : {'set' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")
    print(f"  ELEVENLABS_API_KEY  : {'set' if os.environ.get('ELEVENLABS_API_KEY') else 'MISSING'}")
    # Langfuse
    from src.observability import is_enabled as obs_enabled
    print(f"  Langfuse            : {'enabled' if obs_enabled() else 'disabled'}")
    if obs_enabled():
        print(f"    host              : {os.environ.get('LANGFUSE_HOST', 'cloud')}")
    # Resolved pipeline config
    print(f"  TRANSCRIPTION_MODELS: {config.TRANSCRIPTION_MODELS}")
    print(f"  INSTRUCT_MODEL      : {config.INSTRUCT_MODEL!r}")
    print(f"  TIMESTAMP_MODEL     : {config.TIMESTAMP_MODEL!r}")
    print(f"  USE_TARGETED_EXTRACTION: {config.USE_TARGETED_EXTRACTION}")
    print(f"  RECORDINGS_DIR      : {config.RECORDINGS_DIR}")
    print(f"  RESULTS_DIR         : {config.RESULTS_DIR}")
    return 0


# ---------------------------------------------------------------------------
# `phonebot run` (batch over data/recordings/)
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    if args.no_resume:
        from src import config
        config.AUTO_RESUME = False
    if args.results_dir:
        from src import config
        config.RESULTS_DIR = args.results_dir
    if args.recordings_dir:
        from src import config
        config.RECORDINGS_DIR = args.recordings_dir
    import main
    main.run()
    return 0


# ---------------------------------------------------------------------------
# `phonebot extract <wav>` (single recording, no ground truth needed)
# ---------------------------------------------------------------------------
def cmd_extract(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"file not found: {audio_path}", file=sys.stderr)
        return 2

    from src.config import (
        INSTRUCT_MODEL,
        KEYS,
        TIMESTAMP_MODEL,
        TRANSCRIPTION_LANGUAGE,
        TRANSCRIPTION_MODELS,
        USE_TARGETED_EXTRACTION,
    )
    from datetime import datetime

    from src.country import build_country_chain, detect_country
    from src.extract_llm import build_extract_chain, extract_candidate_1
    from src.extract_voxtral import extract_candidate_2
    from src.observability import is_enabled as obs_enabled
    from src.observability import set_session as obs_set_session
    from src.observability import shutdown as obs_shutdown
    from src.observability import span as obs_span
    from src.reconcile import build_reconcile_chain, reconcile
    from src.schema import validate_caller_info
    from src.targeted_extract import targeted_extract
    from src.transcription import (
        Transcription,
        build_instructor,
        build_timestamper,
        build_transcriber,
    )

    # Tag this single-file extraction so it shows up under a Langfuse
    # session named after the audio file + timestamp. Without this every
    # extract call produces an unlabelled trace and the team can't tell
    # which run it came from.
    if obs_enabled():
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_id = f"extract_{audio_path.stem}_{stamp}"
        obs_set_session(session_id, [
            "command=extract",
            f"file={audio_path.name}",
        ])
        print(f"Langfuse session: {session_id}", file=sys.stderr)

    transcribers = []
    for spec in TRANSCRIPTION_MODELS:
        try:
            transcribers.append(build_transcriber(spec))
        except Exception as e:
            print(f"WARNING: could not build transcriber {spec!r}: {e}",
                  file=sys.stderr)
    if not transcribers:
        print("ERROR: no transcribers available", file=sys.stderr)
        return 3

    try:
        instructor = build_instructor(INSTRUCT_MODEL)
    except Exception as e:
        print(f"WARNING: could not build INSTRUCT_MODEL={INSTRUCT_MODEL!r}: {e}",
              file=sys.stderr)
        instructor = None

    timestamper = None
    if TIMESTAMP_MODEL:
        try:
            timestamper = build_timestamper(TIMESTAMP_MODEL)
        except Exception as e:
            print(f"WARNING: could not build TIMESTAMP_MODEL={TIMESTAMP_MODEL!r}: {e}",
                  file=sys.stderr)

    # Layer 1 - transcribe with every configured STT model. Each call
    # gets its own Langfuse trace named layer1.transcribe with the
    # actual transcript text as the trace output (visible in the UI).
    transcriptions: list[Transcription] = []
    for tr in transcribers:
        try:
            with obs_span("layer1.transcribe", backend=tr.name,
                          file=audio_path.name,
                          language=TRANSCRIPTION_LANGUAGE) as s:
                transcription = tr.transcribe(str(audio_path),
                                               language=TRANSCRIPTION_LANGUAGE)
                transcriptions.append(transcription)
                s.set_output({
                    "text": transcription.text,
                    "has_word_timestamps": transcription.has_timestamps(),
                    "n_words": len(transcription.words or []),
                })
        except Exception as e:
            print(f"WARNING: transcriber {tr.name!r} failed: {e}",
                  file=sys.stderr)
    if not transcriptions:
        print("ERROR: every transcriber failed", file=sys.stderr)
        return 4

    # Layer 1b - lazy Whisper word timestamps when no STT provided them.
    words = next((t.words for t in transcriptions if t.words), None)
    if words is None and timestamper is not None:
        try:
            with obs_span("layer1b.timestamp", backend=timestamper.name,
                          file=audio_path.name) as s:
                words = timestamper.timestamp(str(audio_path),
                                               language=TRANSCRIPTION_LANGUAGE)
                s.set_output({"n_words": len(words or []),
                              "preview": [
                                  {"text": w.text, "start": w.start, "end": w.end}
                                  for w in (words or [])[:10]
                              ]})
        except Exception:
            words = None

    primary_text = transcriptions[0].text
    country_chain = build_country_chain()
    extract_chain = build_extract_chain()
    reconcile_chain = build_reconcile_chain()

    # Layer 2 - country detection (LLM, traced via langchain callback)
    country_info = detect_country(country_chain, primary_text)

    # Layer 3 - candidate-1 per transcript (LLM, traced via langchain callback)
    candidates_1 = []
    for trans in transcriptions:
        cand = extract_candidate_1(extract_chain, trans.text,
                                    country_info["country"],
                                    country_info["language_code"])
        candidates_1.append({
            "transcriber": trans.backend,
            "value": {k: cand.get(k, "") for k in KEYS},
        })

    # Layer 4 - candidate-2 (audio LLM whole-record). Wrapped in obs_span
    # so it shows up as its own trace next to the LangChain traces.
    if instructor is not None:
        with obs_span("layer4.candidate_2", backend=instructor.name,
                      file=audio_path.name) as s:
            cand2 = extract_candidate_2(instructor, str(audio_path))
            s.set_output({k: cand2.get(k, "") for k in KEYS})
            s.add_metadata(n_runs=len(cand2.get("_runs", [])))
    else:
        cand2 = {k: "" for k in KEYS}

    # Layer 4.5 - targeted re-listen on contested fields, also traced.
    targeted = None
    if USE_TARGETED_EXTRACTION and instructor is not None:
        emails = {(c["value"].get("email") or "").strip() for c in candidates_1}
        emails.add((cand2.get("email") or "").strip())
        phones = {(c["value"].get("phone_number") or "").strip() for c in candidates_1}
        phones.add((cand2.get("phone_number") or "").strip())
        run_email = len({e for e in emails if e}) > 1
        run_phone = len({p for p in phones if p}) > 1
        if run_email or run_phone:
            with obs_span("layer4_5.targeted_extract", file=audio_path.name,
                          run_email=run_email, run_phone=run_phone) as s:
                targeted = targeted_extract(instructor, str(audio_path),
                                             words=words,
                                             run_email=run_email,
                                             run_phone=run_phone)
                s.set_output({
                    "email": (targeted or {}).get("email", ""),
                    "phone_number": (targeted or {}).get("phone_number", ""),
                    "email_meta": (targeted or {}).get("_email_meta", {}),
                    "phone_meta": (targeted or {}).get("_phone_meta", {}),
                })

    # Layer 5 - reconciler (LLM, traced via langchain callback)
    final = reconcile(reconcile_chain, transcriptions, country_info,
                      candidates_1, cand2, targeted)
    validated = validate_caller_info(final)
    info = validated.info

    output = {
        "first_name": info.first_name,
        "last_name": info.last_name,
        "email": info.email,
        "phone_number": info.phone_number,
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for k, v in output.items():
            print(f"{k:14s} {v}")
    obs_shutdown()
    return 0


# ---------------------------------------------------------------------------
# `phonebot evaluate <report>`
# ---------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    from src.evaluate import evaluate
    summary = evaluate(args.report, args.ground_truth)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("full_match", 0) == summary.get("recordings_evaluated", 0) else 1


# ---------------------------------------------------------------------------
# `phonebot ablate ...` (wraps study.cli for backward compatibility)
# ---------------------------------------------------------------------------
def cmd_ablate(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    # Re-build sys.argv to forward to study.cli
    sys.argv = ["study.cli"] + args.passthrough
    from study import cli as study_cli
    study_cli.main()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phonebot",
        description=(
            "Phonebot caller-info extraction pipeline. Subcommands: "
            "run, extract, evaluate, ablate, doctor."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Batch-process every recording in data/recordings/")
    run.add_argument("--recordings-dir", default=None,
                     help="Override the recordings directory.")
    run.add_argument("--results-dir", default=None,
                     help="Override RESULTS_DIR.")
    run.add_argument("--no-resume", action="store_true",
                     help="Always start a fresh report (disable AUTO_RESUME).")
    run.set_defaults(func=cmd_run)

    extract = sub.add_parser("extract", help="Extract caller info from one WAV file.")
    extract.add_argument("audio", help="Path to a 16-bit WAV recording.")
    extract.add_argument("--json", action="store_true",
                         help="Emit the result as JSON instead of plain text.")
    extract.set_defaults(func=cmd_extract)

    evalp = sub.add_parser("evaluate", help="Re-evaluate an existing report against ground truth.")
    evalp.add_argument("report", help="Path to a results report JSON.")
    evalp.add_argument("--ground-truth", default="data/ground_truth.json",
                       help="Path to the ground-truth JSON.")
    evalp.add_argument("--json", action="store_true",
                       help="Emit the summary block as JSON.")
    evalp.set_defaults(func=cmd_evaluate)

    ablate = sub.add_parser("ablate",
                             help="Run an ablation experiment via study/cli.py.")
    ablate.add_argument("passthrough", nargs=argparse.REMAINDER,
                         help="Arguments forwarded to study.cli.")
    ablate.set_defaults(func=cmd_ablate)

    doctor = sub.add_parser("doctor",
                             help="Print resolved config and check GPU/API access.")
    doctor.set_defaults(func=cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
