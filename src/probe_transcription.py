"""Tiny end-to-end probe for the configured TRANSCRIPTION_MODEL.

Usage:
    python -m src.probe_transcription data/recordings/call_01.wav

Useful to verify ElevenLabs auth / permissions / network without touching
Voxtral or the rest of the pipeline.
"""

import sys
import time
from pathlib import Path

from src.config import TRANSCRIPTION_LANGUAGE, TRANSCRIPTION_MODEL
from src.transcription import build_transcriber


def main(audio_path: str) -> None:
    if not Path(audio_path).exists():
        print(f"audio file not found: {audio_path}")
        sys.exit(1)

    print(f"backend spec: {TRANSCRIPTION_MODEL}")
    print(f"audio file:   {audio_path}")
    print(f"language:     {TRANSCRIPTION_LANGUAGE}")
    print()

    print("building transcriber...")
    t0 = time.perf_counter()
    transcriber = build_transcriber(TRANSCRIPTION_MODEL)
    print(f"  built in {time.perf_counter() - t0:.2f}s - {transcriber.name}")

    print("transcribing...")
    t0 = time.perf_counter()
    trans = transcriber.transcribe(audio_path, language=TRANSCRIPTION_LANGUAGE)
    print(f"  done in {time.perf_counter() - t0:.2f}s")
    print()
    print("=" * 60)
    print(trans.text)
    print("=" * 60)
    if trans.has_timestamps():
        print(f"\nword timestamps: {len(trans.words)} words, {trans.words[0].start:.2f}s → {trans.words[-1].end:.2f}s")
        # Show first 8 words for sanity
        for w in trans.words[:8]:
            print(f"  {w.start:6.2f}s  {w.text}")
    else:
        print("\nword timestamps: NOT AVAILABLE (backend doesn't expose them)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.probe_transcription <audio_path>")
        sys.exit(2)
    main(sys.argv[1])
