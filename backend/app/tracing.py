"""Langfuse tracing that degrades to a no-op when keys aren't configured.

Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST (or
LANGFUSE_BASE_URL) to turn it on. Every graph node, LLM call and memory
decision is wrapped with `traced(...)`, so a single chat turn shows up in
Langfuse as one trace with route / retrieve / respond / extract / reconcile
spans underneath it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

from . import config  # noqa: F401  - loads .env before we read LANGFUSE_* below

F = TypeVar("F", bound=Callable[..., Any])
log = logging.getLogger(__name__)

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if ENABLED:
    from langfuse import get_client, observe
else:  # pragma: no cover - trivial
    observe = None
    get_client = None


def traced(name: str, as_type: str | None = None) -> Callable[[F], F]:
    if not ENABLED:
        return lambda fn: fn
    return observe(name=name, as_type=as_type)  # type: ignore[return-value]


def annotate(**fields: Any) -> None:
    """Attach output/metadata to the current span (no-op without Langfuse)."""
    if not ENABLED:
        return
    try:
        get_client().update_current_span(**fields)
    except Exception as e:  # tracing must never break the app
        log.debug("langfuse annotate failed: %s", e)


def annotate_generation(model: str, usage: Any, **fields: Any) -> None:
    if not ENABLED:
        return
    try:
        details = {
            "input": getattr(usage, "input_tokens", 0) or 0,
            "output": getattr(usage, "output_tokens", 0) or 0,
        }
        get_client().update_current_generation(model=model, usage_details=details, **fields)
    except Exception as e:
        log.debug("langfuse generation annotate failed: %s", e)


def check_connection() -> bool:
    """Verify the keys against the configured host so a wrong region or key
    shows up at startup instead of traces silently never arriving."""
    if not ENABLED:
        return False
    try:
        return bool(get_client().auth_check())
    except Exception as e:
        log.warning(
            "Langfuse rejected the keys (%s). Traces will NOT be recorded. Check LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY, and that LANGFUSE_HOST matches your project's region "
            "(EU: https://cloud.langfuse.com, US: https://us.cloud.langfuse.com).", type(e).__name__)
        return False


def flush() -> None:
    if ENABLED:
        try:
            get_client().flush()
        except Exception:
            pass
