"""Audio backends used by Layer 1 (transcription / timestamping) and Layer 4 (instruction-following on audio).

Backends are model-agnostic. Pick them in `config.py`:
    TRANSCRIPTION_MODEL = "voxtral:mistralai/Voxtral-Mini-3B-2507"
    TRANSCRIPTION_MODEL = "elevenlabs:scribe_v2"
    INSTRUCT_MODEL      = "voxtral:mistralai/Voxtral-Mini-3B-2507"
    TIMESTAMP_MODEL     = "whisper:openai/whisper-large-v3-turbo"   # optional; "" disables

Three abstract roles:

- Transcriber:    `.transcribe(audio_path) -> Transcription`
- Timestamper:    `.timestamp(audio_path) -> list[Word]`
- AudioInstructor: `.instruct(audio_path, instruction) -> str`

Voxtral can transcribe + instruct (no timestamps). ElevenLabs Scribe can
transcribe + timestamp. Whisper is timestamp-only (Layer 1b, fills in word
timestamps when the main transcriber doesn't).

`Transcription` carries `.text` plus optionally `.words` (per-word timestamps).
When the main transcriber doesn't return words and a TIMESTAMP_MODEL is
configured, Layer 1b (Whisper or Scribe) fills them in.

`build_transcriber` / `build_instructor` / `build_timestamper` cache backend
instances so a single model is loaded at most once per process even when
multiple roles share the spec.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Optional, Protocol


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class Word:
    text: str
    start: float   # seconds from audio start
    end: float


@dataclass
class Transcription:
    text: str
    backend: str
    words: Optional[list[Word]] = None  # None when backend has no timestamps

    def has_timestamps(self) -> bool:
        return self.words is not None and len(self.words) > 0


# ---------------------------------------------------------------------------
# Abstract roles
# ---------------------------------------------------------------------------
class Transcriber(Protocol):
    name: str

    def transcribe(self, audio_path: str, language: str = "de") -> Transcription: ...


class AudioInstructor(Protocol):
    name: str

    def instruct(self, audio_path: str, instruction: str, **kwargs) -> str: ...


class Timestamper(Protocol):
    name: str

    def timestamp(self, audio_path: str, language: str = "de") -> list[Word]: ...


# ---------------------------------------------------------------------------
# Voxtral backend (transcription + instruction-following)
# ---------------------------------------------------------------------------
class VoxtralBackend:
    """Local HuggingFace Voxtral; supports transcribe + instruct."""

    def __init__(self, repo_id: str, device: str = "cuda", attn_implementation: str = "eager"):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self.name = f"voxtral:{repo_id}"
        self.repo_id = repo_id
        self.device = device
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(repo_id)
        # `attn_implementation="eager"` avoids torch's SDPA attention path,
        # which has been observed to segfault during generation on some
        # torch+CUDA combos. "sdpa" / "flash_attention_2" are also valid.
        kwargs = {"dtype": torch.bfloat16, "device_map": device}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = VoxtralForConditionalGeneration.from_pretrained(repo_id, **kwargs)
        self.model.eval()

    def _free(self):
        # Order matters: synchronize first so all in-flight kernels finish and
        # release their KV-cache tensors, THEN gc + empty_cache. Skipping the
        # synchronize has been observed to leak KV-cache state across calls
        # and eventually segfault `transformers/cache_utils.py:update`.
        if self.device == "cuda":
            try:
                self._torch.cuda.synchronize()
            except Exception:
                pass
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()

    def transcribe(self, audio_path: str, language: str = "de") -> Transcription:
        inputs = self.processor.apply_transcription_request(
            language=language, audio=audio_path, model_id=self.repo_id
        )
        inputs = inputs.to(self.device, dtype=self._torch.bfloat16)
        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=500)
        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        text = " ".join(d.strip() for d in decoded)
        del inputs, outputs
        self._free()
        # Voxtral does not expose per-word timestamps - leave words=None.
        return Transcription(text=text, backend=self.name, words=None)

    def instruct(
        self,
        audio_path: str,
        instruction: str,
        max_new_tokens: int = 500,
        temperature: float = 0.01,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(conversation)
        inputs = inputs.to(self.device, dtype=self._torch.bfloat16)
        with self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        raw = decoded[0] if decoded else ""
        del inputs, outputs
        self._free()
        return raw


# Back-compat alias for any code that still imports `VoxtralEngine`.
VoxtralEngine = VoxtralBackend


# ---------------------------------------------------------------------------
# ElevenLabs Scribe backend (transcription only)
# ---------------------------------------------------------------------------
class ElevenLabsTranscriber:
    """ElevenLabs Speech-to-Text (Scribe). Transcription only.

    Uses the official `elevenlabs` SDK. Supports the optional `keyterms`
    biasing field - useful for our domain (call recordings often use
    German separator words "Punkt"/"Bindestrich"/"Unterstrich" inside
    spelled-out emails) - wired through `config.ELEVENLABS_KEYTERMS`.
    """

    def __init__(self, model_id: str, api_key_env: str = "ELEVENLABS_API_KEY"):
        from elevenlabs.client import ElevenLabs  # lazy import

        self.name = f"elevenlabs:{model_id}"
        self.model_id = model_id
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"ElevenLabs requested but ${api_key_env} is not set."
            )
        self.client = ElevenLabs(api_key=api_key)

    def transcribe(self, audio_path: str, language: str = "de") -> Transcription:
        import time

        from src.config import ELEVENLABS_KEYTERMS

        kwargs: dict = {"model_id": self.model_id}
        # ElevenLabs accepts ISO-639-1 ("de") or ISO-639-3 ("deu"). Pass through.
        if language:
            kwargs["language_code"] = language
        if ELEVENLABS_KEYTERMS:
            kwargs["keyterms"] = list(ELEVENLABS_KEYTERMS)

        # Retry transient errors (rate-limit 429, network blips) with
        # exponential backoff. Without this, a single 429 on one record
        # silently dropped Scribe for that call - which on a 30-call
        # batch turned the dual-STT cell into single-Voxtral and cost a
        # full-record match. Five retries with 1, 2, 4, 8, 16s sleeps
        # cover the worst per-minute rate-limit windows.
        last_err = None
        for attempt in range(6):
            try:
                with open(audio_path, "rb") as f:
                    response = self.client.speech_to_text.convert(file=f, **kwargs)
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Don't retry hard auth errors; they will not heal.
                if "401" in msg or "403" in msg or "invalid api key" in msg:
                    raise
                if attempt == 5:
                    raise
                sleep_s = 2 ** attempt
                print(
                    f"  ElevenLabs transcribe failed (attempt {attempt + 1}/6) on "
                    f"{os.path.basename(audio_path)}: {e!r}; retrying in {sleep_s}s",
                    flush=True,
                )
                time.sleep(sleep_s)

        text = (getattr(response, "text", "") or "").strip()

        # Scribe returns word-level timestamps (text/start/end/type).
        # `type == "word"` filters out spacing/audio_event entries.
        words: list[Word] = []
        raw_words = getattr(response, "words", None) or []
        for w in raw_words:
            wtype = getattr(w, "type", "word")
            if wtype != "word":
                continue
            wtext = getattr(w, "text", "") or ""
            wstart = getattr(w, "start", None)
            wend = getattr(w, "end", None)
            if wstart is None or wend is None:
                continue
            words.append(Word(text=wtext, start=float(wstart), end=float(wend)))

        return Transcription(
            text=text,
            backend=self.name,
            words=words if words else None,
        )

    def timestamp(self, audio_path: str, language: str = "de") -> list[Word]:
        """Reuse `.transcribe()` but discard the text - for the Timestamper role."""
        return self.transcribe(audio_path, language=language).words or []


# ---------------------------------------------------------------------------
# Whisper backend (timestamp only; Layer 1b)
# ---------------------------------------------------------------------------
_WHISPER_LANG = {
    "de": "german", "en": "english", "fr": "french", "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "ja": "japanese",
    "pl": "polish", "sv": "swedish", "no": "norwegian", "da": "danish",
    "tr": "turkish",
}


class WhisperBackend:
    """Local HuggingFace Whisper, used purely for word-level timestamps when
    the main transcriber (e.g. Voxtral) doesn't expose them.

    Uses transformers' `pipeline("automatic-speech-recognition")` with fp32
    weights and word-level `return_timestamps`. The pipeline returns a clean
    `{"chunks": [{"text", "timestamp": (start, end)}]}` structure - far less
    fragile than parsing `tokenizer.batch_decode(..., output_offsets=True)`,
    which silently returns plain strings on transformers 5.7.0.

    Why fp32: at fp16 we saw NaN/inf inside `SuppressTokensLogitsProcessor`
    crashing the run with SIGILL. Whisper-large-v3-turbo at fp32 is ~3.2 GB,
    which fits comfortably alongside Voxtral on a 24 GB GPU.

    No `chunk_length_s` is passed: that triggers the "experimental long-form"
    warning AND has been observed to crash. Audio under 30 s runs in one
    forward; longer audio uses Whisper's built-in long-form generation.
    """

    def __init__(self, repo_id: str, device: str = "cuda"):
        import torch
        from transformers import pipeline

        self.name = f"whisper:{repo_id}"
        self.repo_id = repo_id
        self.device = device
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=repo_id,
            dtype=torch.float32,
            device=device,
        )

    def timestamp(self, audio_path: str, language: str = "de") -> list[Word]:
        gen_kwargs: dict = {"task": "transcribe"}
        lang_full = _WHISPER_LANG.get(language)
        if lang_full:
            gen_kwargs["language"] = lang_full

        try:
            result = self.pipe(
                audio_path,
                return_timestamps="word",
                generate_kwargs=gen_kwargs,
            )
        except Exception as e:
            print(f"WhisperBackend: timestamp call failed on {audio_path}: {e}")
            return []

        if not isinstance(result, dict):
            return []
        chunks = result.get("chunks") or []
        words: list[Word] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = (chunk.get("text") or "").strip()
            ts = chunk.get("timestamp")
            if not text or not ts or len(ts) != 2:
                continue
            start, end = ts
            if start is None or end is None:
                continue
            words.append(Word(text=text, start=float(start), end=float(end)))
        return words


# ---------------------------------------------------------------------------
# vLLM backend (transcription + instruction-following)
# ---------------------------------------------------------------------------
class VoxtralVLLMBackend:
    """Voxtral served by vLLM (offline inference mode).

    vLLM has been observed to be substantially more stable than the HF
    transformers eager-attention path for Voxtral, which intermittently
    SIGSEGVs in `sdpa_attention_forward` during generation. Using vLLM
    eliminates that class of failure.

    The backend uses vLLM's chat API with audio_url content blocks for
    both transcription and instruction-following, so a single LLM
    instance covers both roles.
    """

    def __init__(self, repo_id: str, max_model_len: int = 8192):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise RuntimeError(
                "vllm is not installed. Install it with `pip install vllm` "
                "or use the HF backend `voxtral:...` instead."
            ) from e

        self.name = f"voxtral-vllm:{repo_id}"
        self.repo_id = repo_id
        self._SamplingParams = SamplingParams
        # Mistral-format Voxtral models need these flags.
        self.llm = LLM(
            model=repo_id,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            max_model_len=max_model_len,
            enforce_eager=True,
        )

    def _audio_url(self, audio_path: str) -> str:
        # vLLM's chat API accepts file:// URLs for local audio.
        from pathlib import Path
        return Path(audio_path).resolve().as_uri()

    def transcribe(self, audio_path: str, language: str = "de") -> Transcription:
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio_url",
                 "audio_url": {"url": self._audio_url(audio_path)}},
                {"type": "text",
                 "text": f"Transcribe the audio in {language}. "
                         f"Return only the verbatim transcription, no commentary."},
            ],
        }]
        sp = self._SamplingParams(temperature=0.0, max_tokens=500)
        outputs = self.llm.chat(messages, sampling_params=sp)
        text = outputs[0].outputs[0].text.strip() if outputs else ""
        return Transcription(text=text, backend=self.name, words=None)

    def instruct(
        self,
        audio_path: str,
        instruction: str,
        max_new_tokens: int = 500,
        temperature: float = 0.01,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio_url",
                 "audio_url": {"url": self._audio_url(audio_path)}},
                {"type": "text", "text": instruction},
            ],
        }]
        sp = self._SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.chat(messages, sampling_params=sp)
        return outputs[0].outputs[0].text if outputs else ""


# ---------------------------------------------------------------------------
# GPU availability helper
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    """True iff a CUDA GPU is reachable. Treats torch import failures as 'no GPU'
    so the pipeline still runs in environments where torch isn't installed."""
    try:
        import torch
        return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
_BACKEND_CACHE: dict[str, object] = {}


def _parse_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            f"model spec {spec!r} must be '<backend>:<model_id>' (e.g. "
            "'voxtral-vllm:mistralai/Voxtral-Mini-3B-2507')"
        )
    backend, model = spec.split(":", 1)
    return backend.strip().lower(), model.strip()


def _get_or_build(spec: str):
    if spec in _BACKEND_CACHE:
        return _BACKEND_CACHE[spec]
    backend, model = _parse_spec(spec)
    if backend in ("voxtral-vllm", "vllm"):
        if not _cuda_available():
            raise RuntimeError(
                f"backend {spec!r} requires a CUDA GPU; none is available."
            )
        instance = VoxtralVLLMBackend(repo_id=model)
    elif backend == "voxtral":
        if not _cuda_available():
            raise RuntimeError(
                f"backend {spec!r} requires a CUDA GPU; none is available."
            )
        from src.config import VOXTRAL_ATTN_IMPL, VOXTRAL_DEVICE
        instance = VoxtralBackend(
            repo_id=model, device=VOXTRAL_DEVICE, attn_implementation=VOXTRAL_ATTN_IMPL
        )
    elif backend == "elevenlabs":
        from src.config import ELEVENLABS_API_KEY_ENV
        instance = ElevenLabsTranscriber(model_id=model, api_key_env=ELEVENLABS_API_KEY_ENV)
    elif backend == "whisper":
        if not _cuda_available():
            raise RuntimeError(
                f"backend {spec!r} requires a CUDA GPU; none is available."
            )
        from src.config import WHISPER_DEVICE
        instance = WhisperBackend(repo_id=model, device=WHISPER_DEVICE)
    else:
        raise ValueError(f"unknown audio backend {backend!r} in spec {spec!r}")
    _BACKEND_CACHE[spec] = instance
    return instance


def build_transcriber(spec: str) -> Transcriber:
    inst = _get_or_build(spec)
    if not hasattr(inst, "transcribe"):
        raise RuntimeError(f"backend {spec!r} does not implement transcribe()")
    return inst


def build_instructor(spec: str) -> AudioInstructor:
    inst = _get_or_build(spec)
    if not hasattr(inst, "instruct"):
        raise RuntimeError(
            f"backend {spec!r} cannot follow audio instructions - pick a "
            "model that supports it (e.g. voxtral:...)."
        )
    return inst


def build_timestamper(spec: str | None) -> Timestamper | None:
    """Return None when spec is empty/None - the pipeline can run without one."""
    if not spec or spec.lower() in ("none", "off"):
        return None
    inst = _get_or_build(spec)
    if not hasattr(inst, "timestamp"):
        raise RuntimeError(
            f"backend {spec!r} cannot produce word timestamps - pick "
            "whisper:* or elevenlabs:*."
        )
    return inst
