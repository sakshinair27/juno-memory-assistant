"""Step 1 of the memory loop: decide what in this exchange is worth remembering.

A cheap model reads the latest user message (with a little surrounding
context) and returns candidate facts, each scored for durability. Anything
below the threshold is treated as conversational noise and dropped.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..config import settings
from ..llm import structured
from ..tracing import annotate, traced

Category = Literal["preference", "personal", "location", "work", "project", "relationship", "goal", "health", "other"]


class CandidateFact(BaseModel):
    content: str = Field(description="One self-contained fact about the user, in third person, present tense, e.g. 'User lives in Austin, Texas'.")
    category: Category
    durability: float = Field(description="0.0-1.0: how likely this is still true and useful weeks from now.")
    pinned: bool = Field(description="True only for facts that should shape EVERY reply regardless of topic (response style/length/tone preferences, preferred name, preferred language).")
    reason: str = Field(description="Very short justification.")


class ExtractionResult(BaseModel):
    facts: list[CandidateFact]
    forget_requests: list[str] = Field(description="Things the user explicitly asked the assistant to forget, as short descriptions. Usually empty.")


SYSTEM = """You are the memory-extraction component of a personal assistant. You read the user's latest message and decide which durable facts about the USER are worth remembering across future conversations.

Extract ONLY facts that are:
- about the user themself (or their stable relationships, pets, projects, workplace, home, goals, health constraints, preferences, routines)
- stated or clearly implied by the USER (never facts the assistant said, never general world knowledge)
- likely to still be true and useful in a few weeks

Do NOT extract:
- transient states or one-off events ("I'm tired today", "I'm eating lunch", "I had a meeting this morning") unless they reveal something lasting
- questions, requests, hypotheticals, or jokes
- task requests like "remind me to..." (those are handled by a separate task tool)
- facts about other people that don't relate to the user

Writing rules:
- Each fact is one atomic, self-contained statement in third person starting with "User", present tense, describing the CURRENT state. For a correction like "actually I moved to Austin", write "User lives in Austin" (the current truth), not "User moved".
- When the user says something stopped being true without a replacement ("I don't have a car anymore"), write the current state: "User no longer has a car".
- Keep specifics (names, places, numbers) and drop filler.
- Use the conversation context only to resolve references (e.g. the assistant asked "where do you live?" and the user answered "Austin").
- pinned=true ONLY for preferences about how the assistant should respond to them in general (concise vs detailed, tone, format, language, what to call them). Everything else is pinned=false.
- durability: 0.9+ for stable identity/preferences, ~0.7 for ongoing projects or plans, below 0.5 for things that will likely be stale soon.

If the user asks you to forget something, put it in forget_requests (and do not extract it as a fact).
Return an empty list when nothing qualifies — most small-talk messages contain nothing durable."""


def _render(user_message: str, context: list[dict]) -> str:
    ctx = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in context[-4:]) or "(none)"
    return f"<recent_context>\n{ctx}\n</recent_context>\n\n<latest_user_message>\n{user_message}\n</latest_user_message>"


@traced("extract_facts", as_type="chain", capture_output=False)
def extract_facts(user_message: str, context: list[dict] | None = None,
                  min_durability: float | None = None) -> tuple[list[CandidateFact], list[str]]:
    threshold = settings.extraction_min_durability if min_durability is None else min_durability
    result = structured(ExtractionResult, SYSTEM, _render(user_message, context or []), name="llm_extract")
    kept = [f for f in result.facts if f.durability >= threshold and f.content.strip()]
    dropped = [f for f in result.facts if f not in kept]
    annotate(input=user_message, output={"kept": [f.model_dump() for f in kept], "dropped": [f.model_dump() for f in dropped],
                     "forget": result.forget_requests})
    return kept, result.forget_requests
