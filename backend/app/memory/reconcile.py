"""Step 2 of the memory loop: conflict resolution.

Before writing a candidate fact we retrieve the most similar existing facts
and decide between:

  ADD    - genuinely new information
  UPDATE - same attribute, new value (moved cities, changed preference,
           project status changed) -> rewrite the existing row in place
  DELETE - the candidate says an existing fact stopped being true and
           there's nothing to replace it with
  NOOP   - already known

Two fast paths skip the LLM entirely: no similar facts at all (ADD) and a
near-identical fact (NOOP). Everything in between goes to a small judge
model, which sees neighbours by index (not UUID) so it can't hallucinate ids.
The judge can also consolidate pre-existing duplicates via `also_delete`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, Field

from ..config import settings
from ..embeddings import Embedder
from ..llm import structured
from ..tracing import annotate, traced
from .extraction import CandidateFact
from .store import Memory, MemoryStore


class Decision(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    target_index: int | None = Field(description="Index of the existing fact to UPDATE or DELETE; null for ADD/NOOP.")
    final_content: str | None = Field(description="For UPDATE: the full rewritten fact (current truth, merging any details from the old fact that are still valid). For ADD: the fact to store. Otherwise null.")
    also_delete: list[int] = Field(description="Indices of OTHER existing facts that are now redundant or contradicted and should be removed. Usually empty.")
    reason: str


class ForgetDecision(BaseModel):
    delete_indices: list[int]
    reason: str


SYSTEM = """You maintain a personal assistant's long-term memory about one user. A new candidate fact has been extracted from the latest conversation. You are shown the most similar facts already stored. Decide how to reconcile them so the memory stays accurate, current and free of duplicates.

Actions:
- ADD: the candidate is new information that does not overlap any existing fact. Facts about different things (a second pet, another hobby, a different project) are ADD even if they look similar.
- UPDATE: the candidate is about the SAME attribute/entity as an existing fact and changes, corrects or refines it (new city, new job, changed preference, project moved to a new stage, a more specific version). Rewrite that fact in place with final_content = the current truth. Keep still-valid details from the old fact; drop the superseded value entirely (do not write "previously X").
- DELETE: the candidate only says an existing fact is no longer true and there is nothing useful to store instead (e.g. "User no longer has a car" vs stored "User drives a Honda Civic"). Choose UPDATE instead if the negation itself is worth remembering.
- NOOP: the candidate is already fully captured by an existing fact.

Rules:
- The candidate is newer than every stored fact; when they conflict, the candidate wins.
- You also get the user's original message the candidate was extracted from. Use it as evidence: words like "switched", "instead", "actually", "now", "anymore", "not ... after all", "was wrong" mean an existing fact is being REPLACED or retracted, even if the candidate on its own looks like an additional fact. If the message retracts an existing fact, UPDATE or DELETE it (use also_delete when the candidate itself is added separately).
- Prefer UPDATE over ADD whenever both would describe the same attribute — two facts claiming different values for one attribute is the worst outcome.
- Use also_delete for any other stored facts that become contradicted or duplicated by your result.
- Indices refer to the numbered list of existing facts. Never invent indices."""

FORGET_SYSTEM = """The user asked the assistant to forget something. Given the request and a numbered list of stored facts, return the indices of the facts that the request refers to. Be conservative: only facts clearly covered by the request. Return an empty list if none match."""


@dataclass
class MemoryOp:
    op: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    content: str
    memory_id: str | None = None
    old_content: str | None = None
    reason: str = ""
    path: str = "llm"  # llm | fast_add | fast_dup

    def to_dict(self) -> dict:
        return self.__dict__.copy()


JudgeFn = Callable[[CandidateFact, list[Memory], str], Decision]


def _render(candidate: CandidateFact, neighbours: list[Memory], source: str = "") -> str:
    lines = [f"[{i}] ({m.category}) {m.content}   (similarity {m.similarity:.2f}, last updated {m.updated_at:%Y-%m-%d})"
             for i, m in enumerate(neighbours)]
    return ("<existing_facts>\n" + "\n".join(lines) + "\n</existing_facts>\n\n"
            f"<user_message>\n{source or '(not available)'}\n</user_message>\n\n"
            f"<candidate_fact category=\"{candidate.category}\">\n{candidate.content}\n</candidate_fact>")


def llm_judge(candidate: CandidateFact, neighbours: list[Memory], source: str = "") -> Decision:
    return structured(Decision, SYSTEM, _render(candidate, neighbours, source), name="llm_reconcile")


@dataclass
class Reconciler:
    store: MemoryStore
    embedder: Embedder
    judge: JudgeFn = llm_judge
    min_sim: float = field(default_factory=lambda: settings.conflict_min_sim)
    dup_sim: float = field(default_factory=lambda: settings.duplicate_sim)
    k: int = field(default_factory=lambda: settings.conflict_k)

    @traced("reconcile_fact", as_type="guardrail", capture_output=False)
    def apply(self, candidate: CandidateFact, session_id: str | None = None, source: str = "") -> list[MemoryOp]:
        annotate(input={"candidate": candidate.content, "user_message": source})
        emb = self.embedder.embed_passage(candidate.content)
        neighbours = self.store.search(emb, k=self.k, min_sim=self.min_sim)

        if not neighbours:
            m = self.store.add(candidate.content, candidate.category, emb, pinned=candidate.pinned,
                               confidence=candidate.durability, session_id=session_id, reason="no similar facts")
            ops = [MemoryOp("ADD", m.content, m.id, reason="no similar facts", path="fast_add")]
            annotate(output=[o.to_dict() for o in ops])
            return ops

        top = neighbours[0]
        if top.similarity is not None and top.similarity >= self.dup_sim:
            ops = [MemoryOp("NOOP", top.content, top.id, reason=f"near-duplicate (sim {top.similarity:.3f})", path="fast_dup")]
            annotate(output=[o.to_dict() for o in ops])
            return ops

        d = self.judge(candidate, neighbours, source)
        ops = self._execute(d, candidate, emb, neighbours, session_id)
        annotate(input={"candidate": candidate.content, "user_message": source,
                        "similar_existing_facts": [f"{n.content} (sim {n.similarity:.2f})" for n in neighbours]},
                 output={"decision": d.model_dump(), "ops": [o.to_dict() for o in ops]})
        return ops

    def _valid(self, i: int | None, n: int) -> bool:
        return i is not None and 0 <= i < n

    def _execute(self, d: Decision, c: CandidateFact, emb, neighbours: list[Memory], session_id: str | None) -> list[MemoryOp]:
        ops: list[MemoryOp] = []
        n = len(neighbours)
        action = d.action
        # Defensive: an UPDATE/DELETE that points nowhere degrades to ADD/NOOP.
        if action in ("UPDATE", "DELETE") and not self._valid(d.target_index, n):
            action = "ADD" if action == "UPDATE" else "NOOP"

        if action == "ADD":
            content = (d.final_content or c.content).strip()
            e = emb if content == c.content else self.embedder.embed_passage(content)
            m = self.store.add(content, c.category, e, pinned=c.pinned, confidence=c.durability,
                               session_id=session_id, reason=d.reason)
            ops.append(MemoryOp("ADD", content, m.id, reason=d.reason))
        elif action == "UPDATE":
            target = neighbours[d.target_index]
            content = (d.final_content or c.content).strip()
            e = emb if content == c.content else self.embedder.embed_passage(content)
            m = self.store.update(target.id, content, e, category=c.category, pinned=c.pinned or target.pinned,
                                  session_id=session_id, reason=d.reason)
            if m is not None:
                ops.append(MemoryOp("UPDATE", content, target.id, old_content=target.content, reason=d.reason))
        elif action == "DELETE":
            target = neighbours[d.target_index]
            if self.store.delete(target.id, session_id=session_id, reason=d.reason):
                ops.append(MemoryOp("DELETE", target.content, target.id, old_content=target.content, reason=d.reason))
        else:
            ops.append(MemoryOp("NOOP", c.content, reason=d.reason))

        touched = {d.target_index} if action in ("UPDATE", "DELETE") else set()
        for i in d.also_delete:
            if self._valid(i, n) and i not in touched:
                t = neighbours[i]
                if self.store.delete(t.id, session_id=session_id, reason=f"consolidated: {d.reason}"):
                    ops.append(MemoryOp("DELETE", t.content, t.id, old_content=t.content, reason="consolidated duplicate/contradiction"))
                touched.add(i)
        return ops

    @traced("forget", as_type="guardrail", capture_output=False)
    def forget(self, request: str, session_id: str | None = None) -> list[MemoryOp]:
        annotate(input=request)
        emb = self.embedder.embed_query(request)
        neighbours = self.store.search(emb, k=8, min_sim=0.5)
        if not neighbours:
            return []
        listing = "\n".join(f"[{i}] {m.content}" for i, m in enumerate(neighbours))
        d = structured(ForgetDecision, FORGET_SYSTEM,
                       f"<request>{request}</request>\n<facts>\n{listing}\n</facts>", name="llm_forget")
        ops = []
        for i in sorted(set(d.delete_indices)):
            if 0 <= i < len(neighbours) and self.store.delete(neighbours[i].id, session_id=session_id, reason=f"user asked to forget: {request}"):
                ops.append(MemoryOp("DELETE", neighbours[i].content, neighbours[i].id, old_content=neighbours[i].content,
                                    reason="user asked to forget"))
        annotate(output=[o.to_dict() for o in ops])
        return ops


def remember(user_message: str, context: list[dict], reconciler: Reconciler, session_id: str | None = None) -> tuple[list[CandidateFact], list[MemoryOp]]:
    """Extraction + reconciliation for one user message. Candidates are applied
    sequentially so a later candidate sees the writes of an earlier one."""
    from .extraction import extract_facts

    candidates, forgets = extract_facts(user_message, context)
    ops: list[MemoryOp] = []
    for req in forgets:
        ops += reconciler.forget(req, session_id)
    for c in candidates:
        ops += reconciler.apply(c, session_id, source=user_message)
    return candidates, ops
