"""Central configuration. Everything is overridable via environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (one level above backend/) and from backend/.
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "backend" / ".env")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5434/memory")
    db_schema: str = os.getenv("DB_SCHEMA", "public")

    # Models. The main chat model answers the user; the memory model does the
    # cheap structured calls (routing, extraction, conflict resolution).
    chat_model: str = os.getenv("CHAT_MODEL", "claude-opus-5-5")
    chat_effort: str = os.getenv("CHAT_EFFORT", "low")  # low keeps voice replies snappy
    memory_model: str = os.getenv("MEMORY_MODEL", "claude-haiku-4-5")

    # Local embeddings (no API key needed). bge-small = 384 dims.
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = _i("EMBED_DIM", 384)

    # Retrieval: at most k contextual facts, each above an absolute floor AND
    # within `retrieval_margin` of the best hit (bge similarities are compressed,
    # so a relative cutoff separates relevant from merely-about-the-user).
    retrieval_k: int = _i("RETRIEVAL_K", 6)
    retrieval_min_sim: float = _f("RETRIEVAL_MIN_SIM", 0.40)
    retrieval_margin: float = _f("RETRIEVAL_MARGIN", 0.08)
    max_pinned_facts: int = _i("MAX_PINNED_FACTS", 10)

    # Extraction: candidates below this durability score are dropped as noise.
    extraction_min_durability: float = _f("EXTRACTION_MIN_DURABILITY", 0.6)

    # Memory-poisoning screen: an LLM judge quarantines candidate facts that are
    # really instructions (overrides, promotions, exfiltration, third-party directives).
    memory_screen: bool = os.getenv("MEMORY_SCREEN", "true").lower() != "false"

    # Conflict resolution: existing facts above conflict_min_sim are sent to the
    # judge; above duplicate_sim we short-circuit to NOOP without an LLM call.
    conflict_min_sim: float = _f("CONFLICT_MIN_SIM", 0.55)
    conflict_k: int = _i("CONFLICT_K", 5)
    duplicate_sim: float = _f("DUPLICATE_SIM", 0.985)

    # MCP: empty = connect to the tasks server in-process (still over the MCP
    # protocol); set to e.g. http://127.0.0.1:8765/mcp to use a standalone server.
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "")

    # Voice
    whisper_model: str = os.getenv("WHISPER_MODEL", "base.en")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    elevenlabs_model: str = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    )


settings = Settings()
