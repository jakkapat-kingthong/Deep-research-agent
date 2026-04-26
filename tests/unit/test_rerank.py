"""Unit tests for chunker (rerank needs network, goes in integration)."""

from __future__ import annotations

from deep_research.tools.rerank import chunk_text


def test_chunk_text_basic() -> None:
    text = "word " * 1200
    chunks = chunk_text(text.strip(), url="https://example.com", title="t")
    assert len(chunks) > 1
    assert all(c.url == "https://example.com" for c in chunks)


def test_chunk_overlap() -> None:
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, url="u", title="t", chunk_size=40, overlap=10)
    # expect 4 chunks: [0-39], [30-69], [60-99], [90-99]
    assert len(chunks) >= 3
    # verify overlap exists between consecutive
    c0_words = chunks[0].text.split()
    c1_words = chunks[1].text.split()
    assert set(c0_words) & set(c1_words)
