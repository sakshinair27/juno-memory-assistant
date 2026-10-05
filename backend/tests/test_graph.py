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


def test_background_memory_returns_reply_first_then_writes(store, embedder, monkeypatch):  # noqa: F811
    import threading

    store.add("User lives in Seattle", "location", embedder.embed_passage("User lives in Seattle"))
    release = threading.Event()

    def fake_structured(schema, system, user, **kw):
        if schema is G.RouteDecision:
            return G.RouteDecision(needs_memory=True, search_queries=["where the user lives"], may_contain_facts=True)
        if schema is ExtractionResult:
            release.wait(5)  # hold the memory write until the reply has been checked
            return ExtractionResult(facts=[CandidateFact(content="User lives in Austin", category="location",
                                                         durability=0.95, pinned=False, reason="moved")], forget_requests=[])
        if schema is Decision:
            return Decision(action="UPDATE", target_index=0, final_content="User lives in Austin",
                            also_delete=[], reason="moved")
        raise AssertionError(schema)

    monkeypatch.setattr(G, "structured", fake_structured)
    monkeypatch.setattr(X, "structured", fake_structured)
    monkeypatch.setattr("app.memory.reconcile.structured", fake_structured)
    monkeypatch.setattr(G.MemoryAgent, "_chat_call",
                        lambda self, s, m, t: SimpleNamespace(stop_reason="end_turn", content=[_text("Welcome to Austin!")]))

    agent = G.MemoryAgent(store, embedder, None)
    out = agent.run("s1", "I moved to Austin.", [], background_memory=True)

    # The reply is back while extraction is still blocked: memory hasn't changed yet.
    assert out["reply"] == "Welcome to Austin!" and out["memory_pending"] is True
    assert agent.memory_result(out["turn_id"])["status"] == "pending"
    assert {m.content for m in store.list_all()} == {"User lives in Seattle"}

    release.set()
    agent.shutdown()  # waits for the queued write
    result = agent.memory_result(out["turn_id"])
    assert result["status"] == "done" and [o["op"] for o in result["memory_ops"]] == ["UPDATE"]
    assert {m.content for m in store.list_all()} == {"User lives in Austin"}


def test_history_carries_tool_log_so_model_sees_past_tool_calls(store, embedder, monkeypatch):  # noqa: F811
    seen = []
    monkeypatch.setattr(G, "structured", lambda *a, **k: G.RouteDecision(
        needs_memory=False, search_queries=[], may_contain_facts=False))
    monkeypatch.setattr(G.MemoryAgent, "_chat_call",
                        lambda self, s, m, t: seen.append(m) or SimpleNamespace(stop_reason="end_turn", content=[_text("ok")]))
    history = [
        {"role": "user", "content": "Remind me to book flights Friday."},
        {"role": "assistant", "content": "Done, I've added it.",
         "tool_calls": [{"name": "add_task", "input": {"title": "Book flights", "due": "Friday"}, "output": "Added task #4"}]},
    ]
    G.MemoryAgent(store, embedder, None).run("s1", "Is my flight reminder saved?", history)
    past_reply = seen[0][1]["content"]
    assert past_reply.startswith("Done, I've added it.")
    assert "<tool_log>" in past_reply and "add_task" in past_reply and "Added task #4" in past_reply
    assert "<tool_log>" not in seen[0][0]["content"], "user turns are passed through unchanged"


def test_user_messages_from_earlier_days_carry_their_date(store, embedder, monkeypatch):  # noqa: F811
    from datetime import date

    seen = []
    monkeypatch.setattr(G, "structured", lambda *a, **k: G.RouteDecision(
        needs_memory=False, search_queries=[], may_contain_facts=False))
    monkeypatch.setattr(G.MemoryAgent, "_chat_call",
                        lambda self, s, m, t: seen.append(m) or SimpleNamespace(stop_reason="end_turn", content=[_text("ok")]))
    history = [{"role": "user", "content": "Remind me Friday.", "at": "2026-10-01"},
               {"role": "assistant", "content": "Okay.", "at": "2026-10-01"},
               {"role": "user", "content": "Thanks!", "at": date.today().isoformat()},
               {"role": "assistant", "content": "Sure."}]
    G.MemoryAgent(store, embedder, None).run("s1", "What's due?", history)
    msgs = seen[0]
    assert msgs[0]["content"] == "[sent 2026-10-01] Remind me Friday."
    assert msgs[1]["content"] == "Okay.", "assistant text is never prefixed"
    assert msgs[2]["content"] == "Thanks!", "today's messages need no date"
