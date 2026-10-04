"""Thin helpers around the Anthropic SDK."""
from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from .config import settings
from .tracing import annotate_generation, traced

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=3)


class LLMRefusal(RuntimeError):
    pass


def structured(schema: type[T], system: str, user: str, *, model: str | None = None,
               max_tokens: int = 2048, name: str = "structured_call") -> T:
    """One cheap structured-output call (used by routing, extraction, conflict resolution)."""
    model = model or settings.memory_model

    @traced(name, as_type="generation", capture_output=False)
    def _call() -> T:
        resp = client().messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        annotate_generation(model, resp.usage, input=user, output=str(resp.parsed_output))
        if resp.stop_reason == "refusal":
            raise LLMRefusal(getattr(resp.stop_details, "explanation", None) or "refused")
        if resp.parsed_output is None:
            raise RuntimeError(f"{name}: no parsed output (stop_reason={resp.stop_reason})")
        return resp.parsed_output

    return _call()
