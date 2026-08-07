"""Central configuration. Reads from environment (.env supported).

Everything runs on OpenAI: GPT for structured metadata extraction, and OpenAI
embeddings for semantic search. A local sentence-transformers embedding provider
is available as a no-key fallback, but no Anthropic key is ever required.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Project root = two levels up from this file (src/lnp_rag/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (default model, embedding dimension) per embedding provider.
PROVIDER_DEFAULTS: dict[str, Tuple[str, int]] = {
    "openai": ("text-embedding-3-small", 1536),   # OpenAI embeddings API
    "local": ("all-MiniLM-L6-v2", 384),           # sentence-transformers, on-device
}


@dataclass(frozen=True)
class Settings:
    # --- Models ---
    # OpenAI chat model used for structured metadata extraction.
    openai_extraction_model: str = os.getenv("LNP_OPENAI_MODEL", "gpt-4o")

    # Embedding provider for semantic search: "openai" (default) or "local".
    embedding_provider: str = os.getenv("LNP_EMBEDDING_PROVIDER", "openai").lower()
    # Optional overrides; when set they apply to the active provider only.
    _model_override: Optional[str] = os.getenv("LNP_EMBEDDING_MODEL")
    _dim_override: Optional[str] = os.getenv("LNP_EMBEDDING_DIM")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("LNP_CHUNK_SIZE", "1200"))       # characters
    chunk_overlap: int = int(os.getenv("LNP_CHUNK_OVERLAP", "200"))  # characters

    # --- Search ---
    default_top_k: int = int(os.getenv("LNP_TOP_K", "8"))
    rrf_k: int = int(os.getenv("LNP_RRF_K", "60"))  # reciprocal-rank-fusion constant

    # --- Extraction ---
    extraction_max_tokens: int = int(os.getenv("LNP_EXTRACTION_MAX_TOKENS", "8000"))
    # Soft cap on characters of paper text sent to the model. Keeps input under
    # gpt-4o's 128k-token context with room for the schema + output (~350k chars
    # is ~87k tokens). Most papers are far smaller.
    max_extraction_chars: int = int(os.getenv("LNP_MAX_EXTRACTION_CHARS", "350000"))

    # --- Storage ---
    data_dir: Path = PROJECT_ROOT / "data"

    # ----- embedding resolution -----
    def resolve_embedding(self, provider: Optional[str] = None) -> Tuple[str, str, int]:
        """Return (provider, model, dim) for the given (or active) provider."""
        provider = (provider or self.embedding_provider).lower()
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown embedding provider {provider!r}. Use one of {list(PROVIDER_DEFAULTS)}."
            )
        model, dim = PROVIDER_DEFAULTS[provider]
        # Apply overrides only to the active provider.
        if provider == self.embedding_provider:
            if self._model_override:
                model = self._model_override
            if self._dim_override:
                dim = int(self._dim_override)
        return provider, model, dim

    @property
    def embedding_model(self) -> str:
        return self.resolve_embedding()[1]

    @property
    def embedding_dim(self) -> int:
        return self.resolve_embedding()[2]

    # ----- storage paths -----
    def index_dir(self, provider: Optional[str] = None) -> Path:
        """Per-provider index namespace, so different-dimension vectors never mix."""
        prov, model, dim = self.resolve_embedding(provider)
        sig = re.sub(r"[^a-zA-Z0-9]+", "-", f"{prov}-{model}-{dim}").strip("-")
        d = self.data_dir / "index" / sig
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def metadata_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "metadata.jsonl"


settings = Settings()
