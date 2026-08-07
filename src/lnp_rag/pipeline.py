"""End-to-end ingestion: PDF -> text -> chunks -> index, plus OpenAI extraction.

Persists three things under ./data:
  - index/chunks.json + index/embeddings.npy  (the search corpus)
  - metadata.jsonl                             (one extracted record per PDF)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .chunking import chunk_text
from .config import settings
from .extractor import extract_metadata
from .pdf_loader import load_pdf_text
from .schema import LNPMetadata
from .search import SearchIndex
from .smiles import resolve_smiles

# (name field, SMILES field) pairs the reference resolver can auto-fill.
_SMILES_PAIRS = [
    ("il_name", "il_smiles"),
    ("pl_name", "pl_smiles"),
    ("chol_type", "chol_smiles"),
    ("peg_lipid_name", "peg_lipid_smiles"),
]


def _slug(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def ensure_formulation_id(meta: LNPMetadata, doc_id: str, doc_name: str) -> bool:
    """If the paper gave no formulation label, derive a deterministic one.

    Returns True if an ID was generated. Format: '<first-author-or-filename>-<hash6>',
    stable per PDF so it works as a database key. Records the synthesis in `notes`.
    """
    if meta.formulation_id and meta.formulation_id.strip():
        return False
    base = _slug(meta.paper_first_author) or _slug(Path(doc_name).stem) or "lnp"
    meta.formulation_id = f"{base}-{doc_id[:6]}"
    note = "formulation_id auto-generated (no explicit label in the paper)."
    meta.notes = f"{meta.notes} {note}".strip() if meta.notes else note
    return True


def enrich_smiles(meta: LNPMetadata, *, overwrite: bool = False) -> Dict[str, str]:
    """Fill empty *_smiles fields from the component name via the reference resolver.

    Returns {smiles_field: provenance} for the fields that were filled.
    """
    filled: Dict[str, str] = {}
    for name_field, smiles_field in _SMILES_PAIRS:
        name = getattr(meta, name_field, None)
        current = getattr(meta, smiles_field, None)
        if not name or (current and not overwrite):
            continue
        hit = resolve_smiles(name)
        if hit:
            setattr(meta, smiles_field, hit.smiles)
            filled[smiles_field] = hit.source
    return filled


def _doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class IngestResult:
    doc_id: str
    doc_name: str
    num_chunks: int
    already_indexed: bool
    metadata: Optional[LNPMetadata]
    text_chars: int
    smiles_filled: Dict[str, str] = field(default_factory=dict)


def load_index(provider: Optional[str] = None) -> SearchIndex:
    """Load the persisted search index for an embedding provider (empty if none)."""
    return SearchIndex.load(provider)


def build_index(provider: Optional[str] = None) -> SearchIndex:
    """Alias for load_index for callers that prefer the name."""
    return load_index(provider)


def _append_metadata(doc_id: str, doc_name: str, meta: LNPMetadata) -> None:
    record = {"doc_id": doc_id, "doc_name": doc_name, **meta.model_dump()}
    with open(settings.metadata_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def delete_document(doc_id: str, *, index: Optional[SearchIndex] = None, provider: Optional[str] = None) -> SearchIndex:
    """Remove a document from the search index and its extracted metadata record."""
    index = index if index is not None else load_index(provider)
    index.remove_document(doc_id)
    index.save()

    path = settings.metadata_path
    if path.exists():
        with open(path) as f:
            kept = [json.loads(line) for line in f if line.strip()]
        kept = [r for r in kept if r.get("doc_id") != doc_id]
        with open(path, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
    return index


def load_all_metadata() -> List[dict]:
    """Return every extracted metadata record (deduplicated by doc_id, last wins)."""
    path = settings.metadata_path
    if not path.exists():
        return []
    by_doc: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_doc[rec["doc_id"]] = rec
    return list(by_doc.values())


def ingest_pdf(
    source: Union[str, Path, bytes],
    *,
    doc_name: Optional[str] = None,
    index: Optional[SearchIndex] = None,
    provider: Optional[str] = None,
    extract: bool = True,
    add_smiles: bool = True,
    ensure_id: bool = True,
    persist: bool = True,
) -> IngestResult:
    """Ingest one PDF: index it for search and (optionally) extract metadata.

    Args:
        source: path or raw bytes of the PDF.
        doc_name: display name; inferred from path if omitted.
        index: an existing SearchIndex to add to; loaded from disk if omitted.
        extract: run OpenAI metadata extraction.
        persist: write the index and metadata to ./data.
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        doc_name = doc_name or "uploaded.pdf"
    else:
        path = Path(source)
        data = path.read_bytes()
        doc_name = doc_name or path.name

    doc_id = _doc_id(data)
    index = index if index is not None else load_index(provider)

    text = load_pdf_text(data)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)

    already = index.has_doc(doc_id)
    num_chunks = 0 if already else index.add_document(doc_id, doc_name, chunks)

    metadata: Optional[LNPMetadata] = None
    smiles_filled: Dict[str, str] = {}
    if extract:
        metadata = extract_metadata(text)
        if ensure_id:
            ensure_formulation_id(metadata, doc_id, doc_name)
        if add_smiles:
            smiles_filled = enrich_smiles(metadata)

    if persist:
        if num_chunks:
            index.save()
        if metadata is not None:
            _append_metadata(doc_id, doc_name, metadata)

    return IngestResult(
        doc_id=doc_id,
        doc_name=doc_name,
        num_chunks=num_chunks,
        already_indexed=already,
        metadata=metadata,
        text_chars=len(text),
        smiles_filled=smiles_filled,
    )
