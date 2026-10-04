"""The LangGraph agent that runs one chat turn.

    route ──needs memory?──> retrieve ──> respond ──facts to save?──> remember ──> END
      └──────────── no ────────────────────┘    └────────── no ─────────────────> END

route    - cheap model decides: does this turn need contextual memory, and might
           the user's message contain durable facts (or a "forget" request)?
           Also loads pinned facts, which apply to every reply.
retrieve - LangChain retriever pulls only the facts relevant to this question.
respond  - main model answers with the memories in context; it can call the
           MCP tasks tools ("remind me to...").
remember - extraction + conflict resolution on the user's message.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..config import settings
from ..embeddings import Embedder
from ..llm import client, structured
from ..mcp_client import TaskTools
from ..memory.reconcile import Reconciler, remember
from ..memory.retriever import MemoryRetriever
from ..memory.store import MemoryStore
from ..tracing import annotate, annotate_generation, traced

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOOL_ROUNDS = 5
_TRIVIAL = re.compile(r"^\s*(hi|hey|hello|thanks|thank you|ok(ay)?|cool|nice|great|yes|no|yep|nope|bye|good (morning|night))[\s!.?]*$", re.I)


class TurnState(TypedDict, total=False):
    session_id: str
    history: list[dict]
    user_message: str
    route: dict
    pinned: list[dict]
    retrieved: list[dict]
    reply: str
    tool_calls: list[dict]
    candidates: list[dict]
    memory_ops: list[dict]


class RouteDecision(BaseModel):
    needs_memory: bool = Field(description="Would knowing stored facts about the user (preferences, life details, projects) help answer this message well?")
    search_queries: list[str] = Field(description="1-3 standalone search queries for the user's stored facts, covering the topics the REPLY will likely touch, not just the literal message (e.g. 'I just moved to Austin' while discussing dinner -> ['where the user lives', 'user dietary restrictions and food preferences']). Empty if needs_memory is false.")
    may_contain_facts: bool = Field(description="Does the user's message reveal anything durable about themself (preferences, life details, projects, corrections) or ask to forget something?")


ROUTER_SYSTEM = """You route messages for a personal assistant with long-term memory about its user. For the latest user message, decide (1) whether stored facts about the user would help answer it, and (2) whether the message itself reveals durable facts about the user worth saving, or asks to forget something. Be generous with needs_memory for anything personal, recommendations, planning, or questions about the user themself ("what do you know about me?"). Pure trivia, math or small talk usually needs no memory. Use the recent context: if the conversation is about a topic (food, travel, a project), the reply will likely continue it, so search for facts relevant to that topic too."""

RESPOND_SYSTEM = """You are Juno, a warm, capable personal assistant that remembers the user across conversations.

{memory_block}

How to use memory:
- Let what you know shape your answer naturally (recommendations, examples, tone). Don't recite memories back unless asked.
- Always follow the response-style preferences above.
- If the user says something that contradicts a memory, trust the user — your memory will be updated automatically.
- Never claim to remember something that isn't listed here. If asked what you know about them, answer from the list.

You have a task list tool (via MCP). When the user says "remind me to…" or asks to note/check/complete a to-do, use it, then confirm briefly.

Replies may be read aloud, so prefer plain conversational prose; use lists or markdown only when they genuinely help.
Today's date is {today}."""


def _readable(messages: list) -> list[dict]:
    """Messages as plain JSON for the trace: SDK content blocks -> dicts, thinking blocks dropped."""
    out = []
    for m in messages:
        c = m["content"]
        if not isinstance(c, str):
            c = [b if isinstance(b, dict) else b.model_dump() for b in c]
            c = [b for b in c if b.get("type") not in ("thinking", "redacted_thinking")]
        out.append({"role": m["role"], "content": c})
    return out


def _memory_block(pinned: list[dict], retrieved: list[dict]) -> str:
    if not pinned and not retrieved:
        return "<memory>\nNothing is stored about this user yet.\n</memory>"
    out = ["<memory>"]
    if pinned:
        out.append("Response-style preferences and core facts (apply to every reply):")
        out += [f"- {m['content']}" for m in pinned]
    if retrieved:
        out.append("Facts relevant to this message:")
        out += [f"- {m['content']}" for m in retrieved]
    out.append("</memory>")
    return "\n".join(out)


class MemoryAgent:
    def __init__(self, store: MemoryStore, embedder: Embedder, tools: TaskTools | None = None):
        self.store = store
        self.retriever = MemoryRetriever(store=store, embedder=embedder)
        self.reconciler = Reconciler(store=store, embedder=embedder)
        self.tools = tools
        self.graph = self._build()

    # ---------------- nodes ----------------
    @traced("route", as_type="chain", capture_output=False)
    def route(self, s: TurnState) -> dict:
        pinned = [{"id": d.id, "content": d.page_content, **d.metadata} for d in self.retriever.pinned_documents()]
        msg = s["user_message"]
        if _TRIVIAL.match(msg):
            r = RouteDecision(needs_memory=False, search_queries=[], may_contain_facts=False)
        else:
            ctx = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in s.get("history", [])[-4:])
            r = structured(RouteDecision, ROUTER_SYSTEM,
                           f"<recent_context>\n{ctx or '(none)'}\n</recent_context>\n<latest_user_message>\n{msg}\n</latest_user_message>",
                           max_tokens=512, name="llm_route")
        annotate(input=msg, output=r.model_dump())
        return {"route": r.model_dump(), "pinned": pinned}

    @traced("retrieve", as_type="retriever", capture_output=False)
    def retrieve(self, s: TurnState) -> dict:
        queries = [q for q in s["route"].get("search_queries", []) if q.strip()][:3] or [s["user_message"]]
        best: dict[str, dict] = {}
        for q in queries:  # each query is filtered for relevance on its own; merge keeping the best score
            for d in self.retriever.invoke(q):
                m = {"id": d.id, "content": d.page_content, **d.metadata}
                if d.id not in best or (m["similarity"] or 0) > (best[d.id]["similarity"] or 0):
                    best[d.id] = m
        retrieved = sorted(best.values(), key=lambda m: -(m["similarity"] or 0))[: self.retriever.k]
        self.store.touch([m["id"] for m in retrieved] + [m["id"] for m in s.get("pinned", [])])
        annotate(input=queries, output=retrieved)
        return {"retrieved": retrieved}

    @traced("respond", as_type="agent")
    def respond(self, s: TurnState) -> dict:
        annotate(input=s["user_message"])
        system = RESPOND_SYSTEM.format(memory_block=_memory_block(s.get("pinned", []), s.get("retrieved", [])),
                                       today=date.today().isoformat())
        messages: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in s.get("history", [])[-20:]
            if m.get("content") and m.get("role") in ("user", "assistant")
        ]
        messages.append({"role": "user", "content": s["user_message"]})
        tool_defs = self.tools.definitions() if self.tools else []
        tool_calls: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS + 1):
            resp = self._chat_call(system, messages, tool_defs)
            if resp.stop_reason == "refusal":
                return {"reply": "Sorry — I can't help with that one.", "tool_calls": tool_calls}
            if resp.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out, is_err = self.tools.call(block.name, dict(block.input or {}))
                    tool_calls.append({"name": block.name, "input": block.input, "output": out, "is_error": is_err})
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out, "is_error": is_err})
            messages.append({"role": "user", "content": results})

        reply = "".join(b.text for b in resp.content if b.type == "text").strip()
        return {"reply": reply or "(no reply)", "tool_calls": tool_calls}

    @traced("llm_chat", as_type="generation", capture_output=False)
    def _chat_call(self, system: str, messages: list, tools: list):
        kwargs: dict[str, Any] = dict(
            model=settings.chat_model,
            max_tokens=16000,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": settings.chat_effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",  # re-runs a safety-declined request on Anthropic's recommended fallback
        )
        if tools:
            kwargs["tools"] = tools
        resp = client().beta.messages.create(**kwargs)
        annotate_generation(settings.chat_model, resp.usage, input=_readable(messages),
                            output=[b.model_dump() for b in resp.content if b.type in ("text", "tool_use")],
                            metadata={"stop_reason": resp.stop_reason, "tools": [t["name"] for t in tools]})
        return resp

    @traced("remember", as_type="chain")
    def remember(self, s: TurnState) -> dict:
        annotate(input=s["user_message"])
        context = list(s.get("history", [])[-4:])
        candidates, ops = remember(s["user_message"], context, self.reconciler, s.get("session_id"))
        return {"candidates": [c.model_dump() for c in candidates], "memory_ops": [o.to_dict() for o in ops]}

    # ---------------- wiring ----------------
    def _build(self):
        g = StateGraph(TurnState)
        g.add_node("route", self.route)
        g.add_node("retrieve", self.retrieve)
        g.add_node("respond", self.respond)
        g.add_node("remember", self.remember)
        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: "retrieve" if s["route"]["needs_memory"] else "respond",
                                {"retrieve": "retrieve", "respond": "respond"})
        g.add_edge("retrieve", "respond")
        g.add_conditional_edges("respond", lambda s: "remember" if s["route"]["may_contain_facts"] else END,
                                {"remember": "remember", END: END})
        g.add_edge("remember", END)
        return g.compile()

    @traced("chat_turn", as_type="agent", capture_output=False)
    def run(self, session_id: str, user_message: str, history: list[dict] | None = None) -> TurnState:
        out = self.graph.invoke({"session_id": session_id, "user_message": user_message, "history": history or [],
                                 "retrieved": [], "pinned": [], "tool_calls": [], "candidates": [], "memory_ops": []})
        annotate(input=user_message, output=out.get("reply"),
                 metadata={"session_id": session_id, "route": out.get("route"), "memory_ops": out.get("memory_ops")})
        return out
