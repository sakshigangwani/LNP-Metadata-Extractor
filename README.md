# LNP Metadata Extractor

A RAG pipeline for **lipid-nanoparticle (LNP) research papers**. Upload a PDF and it:

1. **Chunks + embeds** the paper for retrieval.
2. **Extracts ~65 structured metadata fields** with OpenAI (GPT, Structured Outputs) — ionizable lipid, phospholipid, cholesterol, PEG/coating, formulation, particle, cargo, biology, immunogenicity.
3. Lets you run **keyword + semantic + hybrid search** across the whole indexed library.

Everything runs on OpenAI (no Anthropic key needed).

## Architecture

```
PDF ─▶ pdf_loader ─▶ chunking ─▶ embeddings (sentence-transformers, local) ─▶ SearchIndex
                          │                                                      ├─ keyword  (BM25)
                          │                                                      ├─ semantic (cosine)
                          │                                                      └─ hybrid   (RRF fusion)
                          └─▶ extractor (OpenAI Structured Outputs) ─▶ LNPMetadata (Pydantic, 65 fields)

persisted under ./data:  index/chunks.json + index/embeddings.npy   and   metadata.jsonl
```

- **Extraction → OpenAI** (`gpt-4o` by default) via Structured Outputs: the `LNPMetadata` Pydantic model is the `response_format`, so the model returns JSON matching the schema, which is parsed + validated.
- **Embeddings → pluggable**: `openai` (`text-embedding-3-small`, 1536-dim, default) **or** `local` (`sentence-transformers`, `all-MiniLM-L6-v2`, 384-dim, no key). Pick in the sidebar or via `LNP_EMBEDDING_PROVIDER`. Each provider keeps its **own** index (vectors of different dimensions never mix).
- **Keyword → BM25** (`rank-bm25`); **Semantic → cosine** over normalized vectors; **Hybrid → reciprocal-rank fusion**.
- **SMILES enrichment** — after extraction, empty `*_smiles` fields are auto-filled from the component **name** via a PubChem-backed reference (`smiles.py`). Small molecules (ionizable lipid, phospholipid, sterol) get authoritative SMILES; PEG-lipids get the lipid **anchor** (PEG omitted); polymer coatings get the **repeat-unit** monomer. Anchor/repeat-unit entries are flagged as approximate.

## SMILES reference

A version-controlled baseline (`src/lnp_rag/reference/smiles_cache.json`) ships SMILES for the common LNP components, so name→SMILES resolution works **offline**. Names not in the baseline are looked up live on PubChem (set `LNP_PUBCHEM=0` to disable) and cached under `./data/reference/`.

- View / export the full table in the app: **Library → 🧬 SMILES reference** (download CSV to paste into a spreadsheet).
- Add components by editing `REGISTRY` in `src/lnp_rag/smiles.py`, then rebuild the baseline:
  ```bash
  python scripts/build_smiles_cache.py
  ```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY
```

Only `OPENAI_API_KEY` is required — it powers both extraction and (by default)
embeddings. To run embeddings on-device with no key, set `LNP_EMBEDDING_PROVIDER=local`
(or pick "local" in the sidebar); the model (~80 MB) downloads on first use.

## Run the app

```bash
streamlit run app.py
```

- **📥 Upload & Extract** — drop a PDF, see the extracted metadata grouped by category (uncertain fields flagged), download a per-paper CSV.
- **🔎 Search** — query the indexed library; pick keyword / semantic / hybrid in the sidebar.
- **📊 Metadata Library** — every extracted record as a table; download the whole library as CSV.

## Use as a library

```python
import sys; sys.path.insert(0, "src")
from lnp_rag import ingest_pdf, load_index

result = ingest_pdf("paper.pdf")          # indexes + extracts + persists
print(result.metadata.il_name, result.metadata.apparent_pka_formulation)

index = load_index()
for hit in index.search("ionizable lipid pKa", method="hybrid", top_k=5):
    print(hit.score, hit.chunk.doc_name, hit.chunk.text[:80])
```

## Configuration

Override via environment variables (or `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `LNP_OPENAI_MODEL` | `gpt-4o` | OpenAI extraction model (Structured Outputs) |
| `LNP_EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `LNP_EMBEDDING_MODEL` | provider default | Override the embedding model |
| `LNP_EMBEDDING_DIM` | provider default | Must match the embedding model |
| `LNP_CHUNK_SIZE` / `LNP_CHUNK_OVERLAP` | `1200` / `200` | Chunking (characters) |
| `LNP_TOP_K` | `8` | Default results per search |
| `LNP_PUBCHEM` | `1` | Live PubChem SMILES lookup for names not in the baseline (`0` = offline only) |

## Notes

- Scanned PDFs without a text layer are rejected with a clear message (OCR would be needed).
- Re-ingesting the same PDF is a no-op for the index (deduplicated by content hash).
- The extractor is instructed not to fabricate values — missing fields are `null` and flagged in `uncertain_fields`.
- On Anaconda envs with a broken TensorFlow install, `transformers` can segfault on import; the embeddings module forces `USE_TF=0` automatically to avoid this.
