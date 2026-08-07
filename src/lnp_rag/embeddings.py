"""Embeddings for semantic search — pluggable provider.

  provider="local"  -> sentence-transformers, on-device, no API key.
  provider="openai" -> OpenAI embeddings API (needs OPENAI_API_KEY).

Structured metadata extraction always uses OpenAI GPT (see extractor.py); only
the embedding vectors are affected by this choice.
"""
from __future__ import annotations

import os

# Force the torch-only backend in transformers BEFORE it is imported. Some envs
# (e.g. Anaconda with a broken TensorFlow install) segfault when transformers
# eagerly imports TF/Flax. We only ever use the torch path for local embeddings.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from functools import lru_cache
from typing import List, Optional

import numpy as np

from .config import settings


# ----- local (sentence-transformers) -----
@lru_cache(maxsize=4)
def _local_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_local(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    arr = _local_model(model_name).encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # dot product == cosine
        show_progress_bar=False,
    )
    return np.asarray(arr, dtype=np.float32)


# ----- openai -----
@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY


def _embed_openai(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    client = _openai_client()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=model_name, input=texts[start : start + batch_size])
        vectors.extend(item.embedding for item in resp.data)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms  # normalize so dot product == cosine


# ----- public API -----
def embed_texts(texts: List[str], provider: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
    """Embed texts with the chosen provider. Returns (N, dim) float32, L2-normalized."""
    prov, model, dim = settings.resolve_embedding(provider)
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    if prov == "openai":
        return _embed_openai(texts, model, batch_size or 100)
    return _embed_local(texts, model, batch_size or 64)


def embed_query(text: str, provider: Optional[str] = None) -> np.ndarray:
    """Embed a single query string. Returns a (dim,) float32 vector."""
    return embed_texts([text], provider=provider)[0]
