"""Memory evals.

Conflict suite (the headline metric): each case feeds earlier "sessions" of
user messages through the real extraction + conflict-resolution pipeline,
then a final message that contradicts / extends / restates them. A grader
model inspects the resulting memory store:

  pass = current truth stored (if any) AND no stale fact still asserted
         AND no duplicate of the current truth AND all must_keep facts kept

Conflict-resolution accuracy = pass rate on the `contradiction` cases.
`addition` cases catch over-merging; `restatement` cases catch duplication.

Extraction suite: single messages; checks expected durable facts were
extracted and that noise produced nothing.

Usage (from backend/):
    python -m evals.run_evals                       # both suites
    python -m evals.run_evals --suite conflict
    python -m evals.run_evals --suite conflict --baseline   # append-only memory, for comparison
    python -m evals.run_evals --only city-move,kids-count

Runs against a separate Postgres schema ("eval") so your real memory is untouched.
Cost: roughly 3 cheap model calls per message plus one grader call per case.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from app.db import make_pool  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.llm import client  # noqa: E402
from app.memory.extraction import extract_facts  # noqa: E402
from app.memory.reconcile import Decision, Reconciler, remember  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402

HERE = Path(__file__).resolve().parent
GRADER_MODEL = "claude-sonnet-5-5"


class ConflictGrade(BaseModel):
    current_present: bool
    stale_still_asserted: bool
    current_duplicated: bool
    kept_all: bool
    explanation: str


class ExtractionGrade(BaseModel):
    captured_all_expected: bool
    has_spurious_facts: bool
    explanation: str


CONFLICT_RUBRIC = """You grade a personal assistant's long-term memory after a conversation. You get the final list of stored facts and the expectations. Judge meaning, not wording.

- current_present: some stored fact expresses the expected current truth. If expected_current is "null", answer true when no fact asserts the stale claim.
- stale_still_asserted: some stored fact still claims the stale information is currently true. A fact that explicitly says it is NO LONGER true (e.g. "no longer has a car") is not stale. If stale is "null", answer false.
- current_duplicated: two or more separate stored facts express the same current truth (redundant entries). Different facts about different things are not duplicates.
- kept_all: every must_keep fact is still represented. If must_keep is empty, answer true."""

EXTRACTION_RUBRIC = """You grade a memory-extraction step. Given a user message, the facts that SHOULD have been extracted, and what WAS extracted, decide:
- captured_all_expected: every expected fact is represented (meaning, not wording). True if expected is empty.
- has_spurious_facts: anything was extracted that isn't a durable fact about the user (noise, transient states, task requests, other people's opinions, hypotheticals). Extra *valid* durable details about the user are NOT spurious."""


def grade(schema: type[BaseModel], rubric: str, body: str) -> BaseModel:
    resp = client().beta.messages.parse(
        model=GRADER_MODEL,
        max_tokens=4000,
        system=rubric,
        messages=[{"role": "user", "content": body}],
        output_format=schema,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"grader returned no verdict (stop_reason={resp.stop_reason})")
    return resp.parsed_output


def always_add(candidate, neighbours, source="") -> Decision:
    return Decision(action="ADD", target_index=None, final_content=None, also_delete=[], reason="baseline")


def run_conflict(cases: list[dict], baseline: bool) -> list[dict]:
    store = MemoryStore(make_pool(schema="eval"))
    rec = Reconciler(store=store, embedder=get_embedder())
    if baseline:
        rec.judge = always_add
        rec.dup_sim = 1.01  # never short-circuit as duplicate

    results = []
    for case in cases:
        store.clear()
        t0 = time.time()
        ops_log = []
        try:
            for si, session in enumerate(case["sessions"]):
                for msg in session:
                    _, ops = remember(msg, [], rec, session_id=f"s{si}")
                    ops_log += [{"msg": msg, **o.to_dict()} for o in ops]
            _, ops = remember(case["final"], [], rec, session_id="final")
            ops_log += [{"msg": case["final"], **o.to_dict()} for o in ops]
            final = [m.content for m in store.list_all()]
            body = (
                "<stored_facts>\n" + ("\n".join(f"- {f}" for f in final) or "(empty)") + "\n</stored_facts>\n"
                f"<expected_current>{case.get('expected_current') or 'null'}</expected_current>\n"
                f"<stale>{case.get('stale') or 'null'}</stale>\n"
                f"<must_keep>{json.dumps(case.get('must_keep', []))}</must_keep>"
            )
            g: ConflictGrade = grade(ConflictGrade, CONFLICT_RUBRIC, body)
            passed = g.current_present and not g.stale_still_asserted and not g.current_duplicated and g.kept_all
            res = {"id": case["id"], "type": case["type"], "passed": passed, "grade": g.model_dump(),
                   "final_memory": final, "ops": ops_log}
        except Exception as e:  # record and continue; an error counts as a failure
            res = {"id": case["id"], "type": case["type"], "passed": False, "error": repr(e), "ops": ops_log}
        res["seconds"] = round(time.time() - t0, 1)
        print(f"  [{'PASS' if res['passed'] else 'FAIL'}] {case['type']:<13} {case['id']}"
              + (f"  -> {res.get('final_memory')}" if not res["passed"] else ""), flush=True)
        results.append(res)
    store.clear()
    return results


def run_extraction(cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        try:
            facts, forgets = extract_facts(case["message"], case.get("context", []))
            extracted = [f.content for f in facts]
            body = (f"<user_message>{case['message']}</user_message>\n"
                    f"<expected>{json.dumps(case['expected'])}</expected>\n"
                    f"<extracted>{json.dumps(extracted)}</extracted>")
            g: ExtractionGrade = grade(ExtractionGrade, EXTRACTION_RUBRIC, body)
            passed = g.captured_all_expected and not g.has_spurious_facts
            if case.get("expect_forget"):
                passed = passed and bool(forgets)
            if case.get("expect_pinned"):
                passed = passed and any(f.pinned for f in facts)
            res = {"id": case["id"], "noise": not case["expected"], "passed": passed, "extracted": extracted,
                   "pinned": [f.pinned for f in facts], "forget_requests": forgets, "grade": g.model_dump()}
        except Exception as e:
            res = {"id": case["id"], "noise": not case["expected"], "passed": False, "error": repr(e)}
        print(f"  [{'PASS' if res['passed'] else 'FAIL'}] {case['id']}" +
              (f"  -> {res.get('extracted')}" if not res["passed"] else ""), flush=True)
        results.append(res)
    return results


def pct(rows: list[dict]) -> str:
    return f"{sum(r['passed'] for r in rows)}/{len(rows)} ({100 * sum(r['passed'] for r in rows) / max(len(rows), 1):.1f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["conflict", "extraction", "all"], default="all")
    ap.add_argument("--baseline", action="store_true", help="append-only memory (no conflict resolution)")
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--cases", default="conflict_cases.json", help="conflict case file in evals/ (e.g. conflict_heldout.json)")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))

    report: dict = {"timestamp": datetime.now().isoformat(timespec="seconds"), "baseline": args.baseline}
    lines = [f"# Memory eval — {report['timestamp']}" + ("  (BASELINE: append-only)" if args.baseline else ""), ""]

    if args.suite in ("conflict", "all"):
        cases = [c for c in json.loads((HERE / args.cases).read_text(encoding="utf-8")) if not only or c["id"] in only]
        print(f"Conflict suite ({args.cases}): {len(cases)} cases")
        report["cases_file"] = args.cases
        res = run_conflict(cases, args.baseline)
        by = {t: [r for r in res if r["type"] == t] for t in ("contradiction", "addition", "restatement")}
        report["conflict"] = {"results": res, "summary": {t: pct(v) for t, v in by.items() if v}}
        lines += ["## Conflict resolution", "",
                  f"**Conflict-resolution accuracy (contradiction cases): {pct(by['contradiction'])}**", "",
                  f"- No over-merging (addition cases): {pct(by['addition'])}",
                  f"- No duplication (restatement cases): {pct(by['restatement'])}",
                  f"- Overall: {pct(res)}", "", "| case | type | result | final memory |", "|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['type']} | {'✅' if r['passed'] else '❌'} | "
                  f"{'; '.join(r.get('final_memory', [])) or r.get('error', '')} |" for r in res]
        lines.append("")

    if args.suite in ("extraction", "all") and not args.baseline:
        cases = [c for c in json.loads((HERE / "extraction_cases.json").read_text(encoding="utf-8")) if not only or c["id"] in only]
        print(f"Extraction suite: {len(cases)} cases")
        res = run_extraction(cases)
        noise = [r for r in res if r["noise"]]
        durable = [r for r in res if not r["noise"]]
        report["extraction"] = {"results": res, "summary": {"noise_rejected": pct(noise), "durable_captured": pct(durable)}}
        lines += ["## Extraction", "", f"- Noise correctly ignored: {pct(noise)}",
                  f"- Durable facts correctly captured: {pct(durable)}", f"- Overall: {pct(res)}", ""]

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d-%H%M%S") + ("-baseline" if args.baseline else "")
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:12]))
    print(f"\nFull report: evals/results/{stem}.md")


if __name__ == "__main__":
    main()
