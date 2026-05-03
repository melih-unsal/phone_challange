"""CLI wrapper that runs the full pipeline (main.run) with arbitrary
overrides applied to src.config BEFORE main is imported.

Usage:
    python -m study.cli --transcription voxtral:mistralai/Voxtral-Mini-3B-2507
    python -m study.cli --transcription elevenlabs:scribe_v2 --no-targeted
    python -m study.cli --transcription voxtral:mistralai/Voxtral-Mini-3B-2507,elevenlabs:scribe_v2
    python -m study.cli --instruct "" --no-targeted        # disables Voxtral as audio LLM
    python -m study.cli --timestamp ""                      # disables Whisper Layer 1b
    python -m study.cli --num-llm-votes 1 --num-instruct-votes 1 --num-targeted-votes 1
    python -m study.cli --tag baseline_dual                 # tag added to the report filename

The wrapper mutates the live `src.config` module (which is what every layer
imports) and patches `main._timestamped_report_path` so the report ends up
under study/results/ with a deterministic name when --tag is given.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="study.cli",
        description=(
            "Run the phonebot pipeline with config overrides for ablation "
            "experiments. Every flag corresponds to a constant in src/config.py."
        ),
    )

    # Transcription / audio backends
    p.add_argument(
        "--transcription",
        type=str,
        default=None,
        help=(
            "Comma-separated TRANSCRIPTION_MODELS list. "
            "e.g. 'voxtral:mistralai/Voxtral-Mini-3B-2507' OR "
            "'elevenlabs:scribe_v2' OR both joined by commas."
        ),
    )
    p.add_argument(
        "--instruct",
        type=str,
        default=None,
        help=(
            "INSTRUCT_MODEL spec. Pass empty string to disable the audio LLM "
            "entirely (skips Layer 4 candidate 2 and Layer 4.5 re-listen)."
        ),
    )
    p.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help=(
            "TIMESTAMP_MODEL spec for Layer 1b. Pass empty string to disable. "
            "When disabled and the primary STT lacks word timestamps, "
            "Layer 4.5 falls back to whole-audio prompts."
        ),
    )

    # Targeted re-listen (Layer 4.5)
    targeted = p.add_mutually_exclusive_group()
    targeted.add_argument(
        "--targeted",
        dest="targeted",
        action="store_true",
        default=None,
        help="Enable Layer 4.5 targeted re-listen (default).",
    )
    targeted.add_argument(
        "--no-targeted",
        dest="targeted",
        action="store_false",
        help="Disable Layer 4.5 targeted re-listen.",
    )
    p.add_argument(
        "--targeted-pad",
        type=float,
        default=None,
        help="TARGETED_PAD_SECONDS for the targeted-crop window.",
    )

    # Vote counts (self-consistency)
    p.add_argument("--num-llm-votes", type=int, default=None,
                   help="NUM_LLM_VOTES (Layer 3 self-consistency).")
    p.add_argument("--num-instruct-votes", type=int, default=None,
                   help="NUM_INSTRUCT_VOTES (Layer 4 audio LLM self-consistency).")
    p.add_argument("--num-targeted-votes", type=int, default=None,
                   help="NUM_TARGETED_VOTES (Layer 4.5 self-consistency).")

    # Misc
    p.add_argument("--language", type=str, default=None,
                   help="TRANSCRIPTION_LANGUAGE (e.g. 'de').")
    p.add_argument("--phone-region", type=str, default=None,
                   help="DEFAULT_PHONE_REGION (e.g. 'DE').")
    p.add_argument("--llm", type=str, default=None,
                   help="LLM_MODEL_NAME for the country/extract/reconcile chains.")
    p.add_argument("--prompt-version", type=str, default=None,
                   help="PROMPT_VERSION tag stored in the report.")

    # Resume / output
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--auto-resume", dest="auto_resume",
                        action="store_true", default=None,
                        help="AUTO_RESUME=True (default).")
    resume.add_argument("--no-auto-resume", dest="auto_resume",
                        action="store_false",
                        help="AUTO_RESUME=False - always start fresh.")
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="RESULTS_DIR override (defaults to 'results' in repo root).",
    )
    p.add_argument(
        "--tag", type=str, default=None,
        help=(
            "Optional tag baked into the report filename. "
            "Useful for grouping ablation runs."
        ),
    )

    return p


def _apply_overrides(args: argparse.Namespace) -> None:
    """Mutate the live src.config module with values from `args`."""
    cfg = importlib.import_module("src.config")

    if args.transcription is not None:
        cfg.TRANSCRIPTION_MODELS = _split_csv(args.transcription)
    if args.instruct is not None:
        cfg.INSTRUCT_MODEL = args.instruct
    if args.timestamp is not None:
        cfg.TIMESTAMP_MODEL = args.timestamp

    if args.targeted is not None:
        cfg.USE_TARGETED_EXTRACTION = bool(args.targeted)
    if args.targeted_pad is not None:
        cfg.TARGETED_PAD_SECONDS = float(args.targeted_pad)

    if args.num_llm_votes is not None:
        cfg.NUM_LLM_VOTES = int(args.num_llm_votes)
    if args.num_instruct_votes is not None:
        cfg.NUM_INSTRUCT_VOTES = int(args.num_instruct_votes)
    if args.num_targeted_votes is not None:
        cfg.NUM_TARGETED_VOTES = int(args.num_targeted_votes)

    if args.language is not None:
        cfg.TRANSCRIPTION_LANGUAGE = args.language
    if args.phone_region is not None:
        cfg.DEFAULT_PHONE_REGION = args.phone_region
    if args.llm is not None:
        cfg.LLM_MODEL_NAME = args.llm
    if args.prompt_version is not None:
        cfg.PROMPT_VERSION = args.prompt_version

    if args.auto_resume is not None:
        cfg.AUTO_RESUME = bool(args.auto_resume)
    if args.results_dir is not None:
        cfg.RESULTS_DIR = args.results_dir


def _patch_report_path(tag: str | None) -> None:
    """Make main use a tag-stamped report filename when --tag is supplied."""
    if not tag:
        return
    import main as _main

    def _tagged_path() -> Path:
        results_dir = importlib.import_module("src.config").RESULTS_DIR
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return Path(results_dir) / f"report_{tag}_{stamp}.json"

    _main._timestamped_report_path = _tagged_path


def _print_resolved_config() -> None:
    cfg = importlib.import_module("src.config")
    print("== resolved config ==")
    print(f"  TRANSCRIPTION_MODELS  : {cfg.TRANSCRIPTION_MODELS}")
    print(f"  INSTRUCT_MODEL        : {cfg.INSTRUCT_MODEL!r}")
    print(f"  TIMESTAMP_MODEL       : {cfg.TIMESTAMP_MODEL!r}")
    print(f"  USE_TARGETED_EXTRACTION: {cfg.USE_TARGETED_EXTRACTION}")
    print(f"  NUM_LLM_VOTES         : {cfg.NUM_LLM_VOTES}")
    print(f"  NUM_INSTRUCT_VOTES    : {cfg.NUM_INSTRUCT_VOTES}")
    print(f"  NUM_TARGETED_VOTES    : {cfg.NUM_TARGETED_VOTES}")
    print(f"  TARGETED_PAD_SECONDS  : {cfg.TARGETED_PAD_SECONDS}")
    print(f"  PROMPT_VERSION        : {cfg.PROMPT_VERSION}")
    print(f"  RESULTS_DIR           : {cfg.RESULTS_DIR}")
    print(f"  AUTO_RESUME           : {cfg.AUTO_RESUME}")
    print("=====================")


def main() -> None:
    args = _build_parser().parse_args()
    os.chdir(REPO_ROOT)
    _apply_overrides(args)
    _print_resolved_config()
    _patch_report_path(args.tag)
    import main as _main
    _main.run()


if __name__ == "__main__":
    main()
