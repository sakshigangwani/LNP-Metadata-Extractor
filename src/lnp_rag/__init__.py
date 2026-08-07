"""LNP Metadata Extractor — RAG pipeline for lipid-nanoparticle papers.

Public surface:
    from lnp_rag import ingest_pdf, search, load_index, LNPMetadata
"""
from .config import settings
from .schema import LNPMetadata, FIELD_GROUPS
from .pipeline import ingest_pdf, load_index, build_index
from .search import SearchIndex, SearchResult

__all__ = [
    "settings",
    "LNPMetadata",
    "FIELD_GROUPS",
    "ingest_pdf",
    "load_index",
    "build_index",
    "SearchIndex",
    "SearchResult",
]
