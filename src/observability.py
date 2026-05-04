"""Optional Langfuse observability.

Langfuse is the open-source equivalent of what Portkey was doing in the
production setup: it captures every LLM call (model, prompt, response,
tokens, latency, cost) and every audio backend invocation, stores them
in a self-hostable (or cloud) backend, and exposes a UI for inspection
and prompt management.

This module is a no-op when Langfuse credentials are not configured, so
the pipeline runs identically with or without observability enabled.

Activate by setting these environment variables (typically in `.env`):

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://localhost:3000     (or https://cloud.langfuse.com)

The bundled docker-compose runs Langfuse v2 (which only needs Postgres),
so this module targets the v2 Python SDK surface (`langfuse.callback`
for the LangChain handler and `Langfuse` for the standalone client).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Callable


def is_enabled() -> bool:
    """True iff Langfuse credentials are present in the environment."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


# ---------------------------------------------------------------------------
# Run-level session + tags. Set once per `phonebot run` from main.py so
# every trace from the same batch shows up under the same Langfuse
# Session, and ablation runs are filterable by tag in the UI.
# ---------------------------------------------------------------------------
_SESSION_ID: str | None = None
_TAGS: list[str] = []


def set_session(session_id: str, tags: list[str] | None = None) -> None:
    """Tag every subsequent trace with this session id and tag list.

    Langfuse's UI groups traces sharing a session_id under one collapsible
    Session row, which is exactly the view the team needs to compare runs.
    Tags become filterable chips, so an ablation run started with
    `phonebot ablate --tag scribe_only` shows up under the
    "scribe_only" filter alongside its siblings.
    """
    global _SESSION_ID, _TAGS
    _SESSION_ID = session_id
    _TAGS = list(tags or [])


def get_session_id() -> str | None:
    return _SESSION_ID


def get_tags() -> list[str]:
    return list(_TAGS)


# ---------------------------------------------------------------------------
# LangChain callback handler
# ---------------------------------------------------------------------------
_CALLBACK_HANDLER = None


def get_langchain_callback():
    """Return a Langfuse CallbackHandler if configured; None otherwise.

    Cached on first call; safe to call from every layer.
    """
    global _CALLBACK_HANDLER
    if _CALLBACK_HANDLER is not None:
        return _CALLBACK_HANDLER
    if not is_enabled():
        return None
    try:
        # v2 Python SDK exposes the LangChain handler at langfuse.callback
        from langfuse.callback import CallbackHandler
    except ImportError:
        # langfuse not installed; silently disable
        return None
    _CALLBACK_HANDLER = CallbackHandler()
    return _CALLBACK_HANDLER


def langchain_config() -> dict:
    """Return a `config` kwargs dict for langchain `chain.invoke(input, config=...)`.

    Empty dict when observability is disabled. When a session has been
    set via `set_session()`, attaches `session_id` and `tags` to the
    callback handler so the resulting trace is grouped + filterable in
    the Langfuse UI.
    """
    cb = get_langchain_callback()
    if cb is None:
        return {}
    # v2 langchain CallbackHandler attaches session/tags via attribute
    # mutation on each invoke. The handler picks them up off itself.
    try:
        if _SESSION_ID:
            cb.session_id = _SESSION_ID
        if _TAGS:
            cb.tags = list(_TAGS)
    except Exception:
        pass
    return {"callbacks": [cb]}


# ---------------------------------------------------------------------------
# Standalone client + span context manager (used by audio backends)
# ---------------------------------------------------------------------------
_LANGFUSE_CLIENT = None


def _get_client():
    """Cached `langfuse.Langfuse` client for non-LangChain calls."""
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        return None
    try:
        _LANGFUSE_CLIENT = Langfuse()
    except Exception:
        return None
    return _LANGFUSE_CLIENT


class _SpanProxy:
    """Tiny wrapper that lets callers attach an output / extra metadata
    to a Langfuse trace from inside a `with obs_span(...)` block.

    The proxy is a no-op when observability is disabled, so call-sites
    can use the same `s.set_output(...)` / `s.add_metadata(...)` pattern
    in every environment.
    """

    def __init__(self, trace):
        self._trace = trace
        self._extra_metadata: dict = {}

    def set_output(self, value) -> None:
        if self._trace is None:
            return
        try:
            self._trace.update(output=_to_serializable(value))
        except Exception:
            pass

    def add_metadata(self, **kwargs) -> None:
        if self._trace is None:
            return
        for k, v in kwargs.items():
            self._extra_metadata[k] = _to_serializable(v)


def _to_serializable(value):
    """Best-effort conversion of arbitrary values to something Langfuse
    can JSON-encode. Falls back to str() for unknown types."""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    # Common dataclass-like / pydantic-like objects.
    for attr in ("model_dump", "dict", "_asdict"):
        m = getattr(value, attr, None)
        if callable(m):
            try:
                return m()
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return {k: _to_serializable(v) for k, v in vars(value).items()
                    if not k.startswith("_")}
        except Exception:
            pass
    return str(value)


@contextmanager
def span(name: str, **metadata):
    """Context manager that emits a Langfuse trace if configured, otherwise
    a no-op timer. Use to trace audio-backend calls (Voxtral, Scribe,
    Whisper) so their latencies, inputs, and outputs land alongside the
    LLM traces from the same run.

    The yielded `_SpanProxy` exposes `set_output(value)` and
    `add_metadata(**kwargs)` so the call-site can attach the actual
    content for debugging:

        with obs_span("layer1.transcribe", backend=tr.name, file=path) as s:
            transcription = tr.transcribe(path)
            s.set_output(transcription.text)
            s.add_metadata(has_word_timestamps=transcription.has_timestamps())

    Each call emits one Langfuse trace, tagged with the current session
    and tags so the UI groups them together.
    """
    if not is_enabled():
        yield _SpanProxy(None)
        return

    client = _get_client()
    if client is None:
        yield _SpanProxy(None)
        return

    t0 = time.perf_counter()
    trace = None
    try:
        trace = client.trace(
            name=name,
            input={k: _to_serializable(v) for k, v in metadata.items()},
            session_id=_SESSION_ID,
            tags=list(_TAGS) if _TAGS else None,
        )
    except Exception:
        # Never let observability break the pipeline.
        yield _SpanProxy(None)
        return

    proxy = _SpanProxy(trace)
    try:
        yield proxy
    finally:
        try:
            md = {"duration_s": round(time.perf_counter() - t0, 3)}
            md.update(proxy._extra_metadata)
            trace.update(metadata=md)
        except Exception:
            pass


def trace(name: str | None = None) -> Callable:
    """Decorator equivalent of `span()` for plain functions."""

    def _decorator(fn: Callable) -> Callable:
        label = name or fn.__name__

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            with span(label):
                return fn(*args, **kwargs)

        _wrapped.__name__ = fn.__name__
        _wrapped.__doc__ = fn.__doc__
        return _wrapped

    return _decorator


def shutdown() -> None:
    """Flush any pending Langfuse events at program exit."""
    if not is_enabled():
        return
    # Flush both the langchain callback handler and the standalone client.
    for obj in (_CALLBACK_HANDLER, _LANGFUSE_CLIENT):
        if obj is None:
            continue
        try:
            obj.flush()
        except Exception:
            pass
