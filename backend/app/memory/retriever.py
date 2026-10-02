"""Step 3 of the memory loop: relevance at response time.

A LangChain retriever over the pgvector fact store. Only facts that clear a
similarity threshold for the *current* question are injected — not the whole
memory. Pinned facts (response-style preferences, preferred name) are the one
exception: they apply to every reply, so they're fetched separately and always
included.
"""
from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ..config import settings
from ..embeddings import Embedder
from .store import Memory, MemoryStore


def to_document(m: Memory) -> Document:
    return Document(page_content=m.content, id=m.id, metadata={
        "id": m.id, "category": m.category, "pinned": m.pinned, "similarity": m.similarity,
        "updated_at": m.updated_at.isoformat(),
    })


class MemoryRetriever(BaseRetriever):
    """Contextual (non-pinned) facts relevant to the query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: MemoryStore
    embedder: Embedder
    k: int = settings.retrieval_k
    min_similarity: float = settings.retrieval_min_sim
    margin: float = settings.retrieval_margin

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        emb = self.embedder.embed_query(query)
        hits = self.store.search(emb, k=self.k, min_sim=self.min_similarity, pinned=False)
        if hits:
            cutoff = hits[0].similarity - self.margin
            hits = [m for m in hits if m.similarity >= cutoff]
        return [to_document(m) for m in hits]

    def pinned_documents(self, limit: int | None = None) -> list[Document]:
        return [to_document(m) for m in self.store.pinned(limit or settings.max_pinned_facts)]
