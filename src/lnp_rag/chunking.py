"""Recursive character-based chunking with overlap.

Splits on the largest natural boundary that keeps chunks under `chunk_size`,
falling back through paragraph -> line -> sentence -> word -> hard cut. Overlap
preserves context across chunk boundaries so retrieval doesn't lose a sentence
that straddles two chunks.
"""
from __future__ import annotations

from typing import List

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # No separator left: hard-cut into chunk_size slices.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *rest = separators
    parts = text.split(sep)
    chunks: List[str] = []
    current = ""
    for part in parts:
        piece = part + sep
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            if len(piece) > chunk_size:
                chunks.extend(_split_recursive(piece, chunk_size, rest))
                current = ""
            else:
                current = piece
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append((tail + " " + cur).strip())
    return out


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Return a list of overlapping text chunks."""
    base = _split_recursive(text, chunk_size, _SEPARATORS)
    return _apply_overlap(base, overlap)
