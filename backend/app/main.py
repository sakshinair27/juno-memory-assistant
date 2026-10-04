"""FastAPI app: chat, memory panel, tasks panel, voice."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from . import tracing, voice
from .agent.graph import MemoryAgent
from .config import settings
from .db import get_pool
from .embeddings import get_embedder
from .mcp_client import get_task_tools
from .memory.store import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("juno")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    pool = get_pool()
    store = MemoryStore(pool)
    embedder = get_embedder()  # loads the ONNX model once (downloads on first run)
    tools = get_task_tools()
    log.info("MCP tools: %s", [t["name"] for t in tools.definitions()])
    state.update(pool=pool, store=store, agent=MemoryAgent(store, embedder, tools))
    traced_ok = tracing.check_connection()
    log.info("ready (chat=%s memory=%s tracing=%s)", settings.chat_model, settings.memory_model,
             "on" if traced_ok else ("REJECTED - see warning above" if tracing.ENABLED else "off"))
    yield
    state["agent"].shutdown()  # finish queued memory writes before closing the pool
    tracing.flush()
    pool.close()


app = FastAPI(title="Juno", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = []


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/api/health")
def health():
    return {"ok": True, "chat_model": settings.chat_model, "memory_model": settings.memory_model,
            "tracing": tracing.ENABLED}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set — add it to .env and restart the backend.")
    try:
        # Reply first; extraction + conflict resolution run in the background (poll /api/turns/{turn_id}).
        out = state["agent"].run(req.session_id, req.message, [m.model_dump() for m in req.history],
                                 background_memory=True)
    except anthropic.APIStatusError as e:
        log.exception("model call failed")
        raise HTTPException(502, f"Model API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Couldn't reach the Anthropic API.")
    finally:
        tracing.flush()
    return {
        "reply": out.get("reply", ""),
        "route": out.get("route"),
        "used_memories": out.get("pinned", []) + out.get("retrieved", []),
        "memory_ops": out.get("memory_ops", []),
        "memory_pending": out.get("memory_pending", False),
        "turn_id": out.get("turn_id"),
        "tool_calls": out.get("tool_calls", []),
    }


@app.get("/api/turns/{turn_id}")
def turn_memory(turn_id: str):
    """Outcome of a turn's background memory write: pending | done | error."""
    result = state["agent"].memory_result(turn_id)
    if result is None:
        raise HTTPException(404, "unknown turn")
    return result


@app.get("/api/memories")
def memories():
    return [m.to_dict() for m in state["store"].list_all()]


@app.get("/api/memories/events")
def memory_events(limit: int = 50):
    return state["store"].events(limit)


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str):
    if not state["store"].delete(memory_id, reason="deleted from UI"):
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.delete("/api/memories")
def clear_memories():
    state["store"].clear()
    return {"ok": True}


@app.get("/api/tasks")
def tasks():
    with state["pool"].connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, title, due, done, created_at FROM tasks ORDER BY done, id DESC LIMIT 100")
        rows = cur.fetchall()
    for r in rows:
        r["created_at"] = r["created_at"].isoformat()
    return rows


@app.post("/api/tasks/{task_id}/done")
def task_done(task_id: int):
    out, is_err = get_task_tools().call("complete_task", {"task_id": task_id})
    return {"ok": not is_err, "result": out}


@app.get("/api/voice/status")
def voice_status():
    return {"whisper": voice.whisper_available(), "elevenlabs": voice.tts_available()}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not voice.whisper_available():
        raise HTTPException(501, "faster-whisper not installed")
    data = await audio.read()
    from starlette.concurrency import run_in_threadpool

    return {"text": await run_in_threadpool(voice.transcribe, data)}


@app.post("/api/tts")
def tts(req: TTSRequest):
    if not voice.tts_available():
        raise HTTPException(501, "ELEVENLABS_API_KEY not set")
    return Response(voice.synthesize(req.text), media_type="audio/mpeg")


# Serve the built frontend (npm run build) from the same origin, if present.
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = (_dist / path).resolve()
        ok = path and f.is_file() and f.is_relative_to(_dist.resolve())
        return FileResponse(f if ok else _dist / "index.html")
