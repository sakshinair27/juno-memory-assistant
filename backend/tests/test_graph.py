"""LangGraph wiring test with every model call faked (no API key needed)."""
from __future__ import annotations

from types import SimpleNamespace

from app.agent import graph as G
from app.memory import extraction as X
from app.memory.extraction import CandidateFact, ExtractionResult
from app.memory.reconcile import Decision
from app.mcp_client import TaskTools

from .test_memory import embedder, pool, store  # noqa: F401  (fixtures)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool_use(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def test_full_turn_routes_retrieves_responds_and_remembers(store, embedder, monkeypatch):  # noqa: F811
    store.add("User prefers concise answers", "preference", embedder.embed_passage("User prefers concise answers"), pinned=True)
    store.add("User lives in Seattle", "location", embedder.embed_passage("User lives in Seattle"))
    store.add("User is vegetarian", "health", embedder.embed_passage("User is vegetarian"))

    def fake_structured(schema, system, user, **kw):
        if schema is G.RouteDecision:
            return G.RouteDecision(needs_memory=True, search_queries=["where does the user live", "user dietary preferences"], may_contain_facts=True)
        if schema is ExtractionResult:
            return ExtractionResult(facts=[CandidateFact(content="User lives in Austin", category="location",
                                                         durability=0.95, pinned=False, reason="moved")], forget_requests=[])
        if schema is Decision:
            return Decision(action="UPDATE", target_index=0, final_content="User lives in Austin",
                            also_delete=[], reason="moved")
        raise AssertionError(schema)

    monkeypatch.setattr(G, "structured", fake_structured)
    monkeypatch.setattr(X, "structured", fake_structured)
    monkeypatch.setattr("app.memory.reconcile.structured", fake_structured)

    seen_systems = []

    def fake_chat(self, system, messages, tools):
        seen_systems.append(system)
        if len(seen_systems) == 1:
            return SimpleNamespace(stop_reason="tool_use", content=[_tool_use("t1", "add_task", {"title": "pack boxes"})])
        return SimpleNamespace(stop_reason="end_turn", content=[_text("Congrats on Austin! Added 'pack boxes'.")])

    monkeypatch.setattr(G.MemoryAgent, "_chat_call", fake_chat)
    with store.pool.connection() as conn:
        conn.execute("TRUNCATE tasks")

    agent = G.MemoryAgent(store, embedder, TaskTools())
    out = agent.run("s1", "I moved to Austin! Remind me to pack boxes.", [])

    assert "Austin" in out["reply"]
    assert "User prefers concise answers" in seen_systems[0], "pinned fact must always be in context"
    assert "User lives in Seattle" in seen_systems[0], "relevant fact must be retrieved"
    assert "User is vegetarian" in seen_systems[0], "second router query must retrieve topic-relevant facts"
    assert out["tool_calls"][0]["name"] == "add_task" and not out["tool_calls"][0]["is_error"]
    assert [o["op"] for o in out["memory_ops"]] == ["UPDATE"]
    contents = {m.content for m in store.list_all()}
    assert contents == {"User prefers concise answers", "User lives in Austin", "User is vegetarian"}


def test_trivial_message_skips_router_retrieval_and_memory(store, embedder, monkeypatch):  # noqa: F811
    monkeypatch.setattr(G, "structured", lambda *a, **k: (_ for _ in ()).throw(AssertionError("router called")))
    monkeypatch.setattr(G.MemoryAgent, "_chat_call",
                        lambda self, s, m, t: SimpleNamespace(stop_reason="end_turn", content=[_text("You're welcome!")]))
    out = G.MemoryAgent(store, embedder, None).run("s1", "thanks!", [])
    assert out["reply"] == "You're welcome!"
    assert out["retrieved"] == [] and out["memory_ops"] == []
