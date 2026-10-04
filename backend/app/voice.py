"""Voice in/out.

Speech-to-text: Whisper, run locally with faster-whisper (optional install:
`pip install faster-whisper`). Text-to-speech: ElevenLabs when a key is set.
The frontend falls back to the browser's Web Speech API for whichever side
isn't available, so voice works out of the box either way.
"""
from __future__ import annotations

import io
import logging
import threading
from functools import lru_cache

import httpx

from .config import settings
from .tracing import annotate, traced

log = logging.getLogger(__name__)
_lock = threading.Lock()


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")


@traced("transcribe", as_type="span")
def transcribe(audio: bytes) -> str:
    annotate(input=f"{len(audio):,} bytes of audio")
    with _lock:
        segments, _info = _whisper().transcribe(io.BytesIO(audio), beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


def tts_available() -> bool:
    return bool(settings.elevenlabs_api_key)


@traced("tts", as_type="span", capture_output=False)
def synthesize(text: str) -> bytes:
    annotate(input=text)
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}",
        headers={"xi-api-key": settings.elevenlabs_api_key, "accept": "audio/mpeg"},
        json={"text": text[:2500], "model_id": settings.elevenlabs_model},
        timeout=60,
    )
    r.raise_for_status()
    return r.content
