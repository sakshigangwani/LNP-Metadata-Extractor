"""Search index: keyword (BM25) + semantic (embeddings) + hybrid (RRF).

The index is a flat corpus of chunks accumulated across every ingested PDF, so a
single query searches the whole library. It persists to disk as JSON (chunk
records) + .npy (embedding matrix) and rebuilds the BM25 index on load.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import embed_query, embed_texts

Method = Literal["keyword", "semantic", "hybrid"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    id: str           # f"{doc_id}:{chunk_index}"
    doc_id: str       # content hash of the source PDF
    doc_name: str     # original filename
    chunk_index: int
    text: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    method: Method


class SearchIndex:
    def __init__(self, provider: Optional[str] = None) -> None:
        # Provider fixes the embedding model/dim for this index. Indexes are
        # persisted per-provider so vectors of different dimensions never mix.
        self.provider, _, self._dim = settings.resolve_embedding(provider)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = np.zeros((0, self._dim), dtype=np.float32)
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: List[List[str]] = []

    # ----- mutation -----
    def has_doc(self, doc_id: str) -> bool:
        return any(c.doc_id == doc_id for c in self.chunks)

    def add_document(self, doc_id: str, doc_name: str, texts: List[str]) -> int:
        """Embed and add a document's chunks. Skips if doc_id already present."""
        if self.has_doc(doc_id) or not texts:
            return 0
        new_embeds = embed_texts(texts, provider=self.provider)
        start = len(self.chunks)
        for i, text in enumerate(texts):
            self.chunks.append(Chunk(f"{doc_id}:{i}", doc_id, doc_name, i, text))
        if self.embeddings.size:
            self.embeddings = np.vstack([self.embeddings, new_embeds])
        else:
            self.embeddings = new_embeds
        self._rebuild_bm25()
        return len(self.chunks) - start

    def remove_document(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.embeddings = self.embeddings[keep] if keep else np.zeros(
            (0, self._dim), dtype=np.float32
        )
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._tokenized = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    # ----- search -----
    def keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        ranked = [SearchResult(self.chunks[i], float(scores[i]), "keyword") for i in order]
        # Prefer positively-scored hits, but fall back to the ranked top-k when
        # BM25 yields no positive scores (e.g. terms common to every chunk give
        # negative IDF) — the relative ranking is still meaningful.
        positive = [r for r in ranked if r.score > 0]
        return positive if positive else ranked

    def semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        if not self.chunks:
            return []
        q = embed_query(query, provider=self.provider)
        sims = self.embeddings @ q  # cosine (vectors are normalized)
        order = np.argsort(sims)[::-1][:top_k]
        return [SearchResult(self.chunks[i], float(sims[i]), "semantic") for i in order]

    def hybrid_search(self, query: str, top_k: int, candidate_k: Optional[int] = None) -> List[SearchResult]:
        """Reciprocal-rank fusion of keyword and semantic results."""
        candidate_k = candidate_k or max(top_k * 4, 20)
        kw = self.keyword_search(query, candidate_k)
        sem = self.semantic_search(query, candidate_k)

        rrf: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for results in (kw, sem):
            for rank, r in enumerate(results):
                rrf[r.chunk.id] = rrf.get(r.chunk.id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
                by_id[r.chunk.id] = r.chunk

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [SearchResult(by_id[cid], score, "hybrid") for cid, score in fused]

    def search(self, query: str, method: Method = "hybrid", top_k: Optional[int] = None) -> List[SearchResult]:
        top_k = top_k or settings.default_top_k
        if method == "keyword":
            return self.keyword_search(query, top_k)
        if method == "semantic":
            return self.semantic_search(query, top_k)
        return self.hybrid_search(query, top_k)

    # ----- persistence -----
    @property
    def doc_names(self) -> dict[str, str]:
        return {c.doc_id: c.doc_name for c in self.chunks}

    def save(self, index_dir: Optional[Path] = None) -> None:
        index_dir = index_dir or settings.index_dir(self.provider)
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "chunks.json", "w") as f:
            json.dump([asdict(c) for c in self.chunks], f)
        np.save(index_dir / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, provider: Optional[str] = None, index_dir: Optional[Path] = None) -> "SearchIndex":
        idx = cls(provider)
        index_dir = index_dir or settings.index_dir(idx.provider)
        chunks_path = index_dir / "chunks.json"
        embeds_path = index_dir / "embeddings.npy"
        if chunks_path.exists() and embeds_path.exists():
            with open(chunks_path) as f:
                idx.chunks = [Chunk(**c) for c in json.load(f)]
            idx.embeddings = np.load(embeds_path)
            idx._rebuild_bm25()
        return idx
