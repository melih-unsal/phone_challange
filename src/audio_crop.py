"""Crop a window of audio to a temporary WAV file.

Uses only `soundfile` (libsndfile) — deliberately avoiding `librosa`
because librosa pulls in `numba` JIT and `audioread`, both of which have
been observed to interact badly with torch's CUDA init on this machine
(intermittent segfaults during the first iteration of the pipeline).

`soundfile.read(path, start=, stop=)` reads only the byte range we need
without loading the full file into memory.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional

import soundfile as sf


@contextmanager
def cropped_audio(
    audio_path: str,
    start_s: float,
    end_s: float,
    pad_s: float = 2.0,
) -> Iterator[str]:
    """Yield a path to a temp WAV containing audio[start_s-pad : end_s+pad].

    The temp file is deleted on context exit. If the requested window is
    empty/invalid, yields the ORIGINAL audio path so callers fall back
    transparently.
    """
    if end_s <= start_s:
        yield audio_path
        return

    try:
        info = sf.info(audio_path)
    except Exception:
        yield audio_path
        return

    sr = info.samplerate
    duration = float(info.frames) / sr if sr else 0.0
    lo = max(0.0, start_s - pad_s)
    hi = min(duration, end_s + pad_s)
    if hi <= lo:
        yield audio_path
        return

    f0 = int(lo * sr)
    f1 = int(hi * sr)
    if f1 <= f0:
        yield audio_path
        return

    try:
        chunk, _ = sf.read(audio_path, start=f0, stop=f1, always_2d=False)
    except Exception:
        yield audio_path
        return

    if len(chunk) == 0:
        yield audio_path
        return

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="crop_")
    os.close(fd)
    try:
        sf.write(tmp_path, chunk, sr, subtype="PCM_16")
        yield tmp_path
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def audio_duration(audio_path: str) -> Optional[float]:
    """Best-effort total duration in seconds (None on failure)."""
    try:
        info = sf.info(audio_path)
        return float(info.frames) / info.samplerate if info.samplerate else None
    except Exception:
        return None
