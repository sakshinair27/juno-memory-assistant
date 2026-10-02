"""Local sentence embeddings via fastembed (ONNX, CPU, no API key).

bge models are asymmetric: queries get an instruction prefix, stored passages
don't. Facts are embedded as passages; user questions as queries. When we
compare a new fact against stored facts (conflict search) both sides are
passages, which is the symmetric case bge handles well.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from .config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name or settings.embed_model
        self._model = TextEmbedding(model_name=self.model_name)
        self._lock = threading.Lock()  # onnxruntime sessions are safest used one call at a time

    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        with self._lock:
            return [np.asarray(v, dtype=np.float32) for v in self._model.passage_embed(texts)]

    def embed_passage(self, text: str) -> np.ndarray:
        return self.embed_passages([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            return np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
