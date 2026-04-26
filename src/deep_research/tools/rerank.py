"""Text chunking + BGE-small cross-encoder rerank."""

from __future__ import annotations

import re
from functools import cached_property

from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder


class Chunk(BaseModel):
    """Text chunk with provenance."""

    chunk_id: str
    text: str
    url: str
    title: str = ""
    score: float = Field(default=0.0)


def chunk_text(
    text: str,
    url: str,
    title: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Naive word-based chunking with overlap. Good enough for research."""
    words = text.split()
    chunks: list[Chunk] = []
    i = 0
    chunk_idx = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        # sanitize chunk_id
        url_safe = re.sub(r"[^a-zA-Z0-9]", "_", url)[:50]
        chunks.append(
            Chunk(
                chunk_id=f"{url_safe}_{chunk_idx}",
                text=chunk_text,
                url=url,
                title=title,
            )
        )
        i += chunk_size - overlap
        chunk_idx += 1
    return chunks


class Reranker:
    """BGE-small cross-encoder for chunk reranking."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name

    @cached_property
    def _model(self) -> CrossEncoder:
        logger.info(f"Loading reranker model: {self._model_name}")
        return CrossEncoder(self._model_name, device="cpu")

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 20) -> list[Chunk]:
        """Return top-k chunks by cross-encoder score."""
        if not chunks:
            return []
        pairs = [[query, c.text] for c in chunks]
        scores = self._model.predict(pairs)
        scored = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]
