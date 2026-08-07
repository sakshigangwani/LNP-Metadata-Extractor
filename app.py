"""Streamlit UI for the LNP Metadata Extractor RAG pipeline.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

# Make the src/ package importable when run via `streamlit run app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from lnp_rag.config import settings
from lnp_rag.pipeline import delete_document, ingest_pdf, load_all_metadata, load_index
from lnp_rag.schema import ALL_FIELDS, FIELD_GROUPS
from lnp_rag.smiles import reference_entries

st.set_page_config(page_title="LNP Metadata Extractor", page_icon="🧬", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1300px;}
      div[data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e6e9ef;
        padding: 10px 16px; border-radius: 12px;
      }
      .app-header {
        background: linear-gradient(100deg, #0f4c81 0%, #2a9d8f 100%);
        color: #fff; padding: 20px 26px; border-radius: 16px; margin-bottom: 14px;
      }
      .app-header h1 {margin: 0; font-size: 1.55rem; font-weight: 700;}
      .app-header p {margin: 4px 0 0; opacity: .9; font-size: .92rem;}
      .pill {display: inline-block; padding: 2px 11px; border-radius: 999px;
             font-size: .78rem; font-weight: 600; line-height: 1.5;}
      .pill-ok  {background: #e6f4ea; color: #137333;}
      .pill-bad {background: #fce8e6; color: #c5221f;}
      .pill-info{background: #e8f0fe; color: #1a56db;}
      .res-meta {color: #5f6b7a; font-size: .82rem; margin-bottom: 4px;}
      .res-text {font-size: .92rem; line-height: 1.5;}
      mark {background: #fff3bf; padding: 0 2px; border-radius: 3px;}
      .stTabs [data-baseweb="tab"] {font-size: 0.95rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers
@st.cache_resource
def get_index(provider: str):
    return load_index(provider)


def fmt(value) -> str:
    if value is None or value == [] or value == "":
        return "—"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def is_empty(value) -> bool:
    return value is None or value == [] or value == ""


def pill(label: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'


def completeness(meta: dict) -> tuple[int, int]:
    filled = sum(0 if is_empty(meta.get(f)) else 1 for f in ALL_FIELDS)
    return filled, len(ALL_FIELDS)


def highlight_html(text: str, query: str) -> str:
    safe = html.escape(text)
    tokens = sorted({t for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1}, key=len, reverse=True)
    for tok in tokens:
        safe = re.sub(f"(?i)({re.escape(tok)})", r"<mark>\1</mark>", safe)
    return safe


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🧬 LNP Extractor")
st.sidebar.caption("OpenAI extraction + embeddings")

openai_ok = bool(os.getenv("OPENAI_API_KEY"))

provider = st.sidebar.selectbox(
    "Embedding provider",
    ["openai", "local"],
    index=["openai", "local"].index(
        settings.embedding_provider if settings.embedding_provider in ("openai", "local") else "openai"
    ),
    help="openai = embeddings API (needs OPENAI_API_KEY) · local = on-device "
    "sentence-transformers (no key). Each provider keeps its own index.",
)
_prov, emb_model, emb_dim = settings.resolve_embedding(provider)
emb_needs_key = provider == "openai"

st.sidebar.markdown("###### Status")
st.sidebar.markdown(
    pill(f"GPT {settings.openai_extraction_model}", "ok" if openai_ok else "bad")
    + " "
    + pill("key set" if openai_ok else "no key", "ok" if openai_ok else "bad"),
    unsafe_allow_html=True,
)
emb_ok = openai_ok or not emb_needs_key
st.sidebar.markdown(
    pill(f"emb: {emb_model}", "info")
    + " "
    + pill("ready" if emb_ok else "no key", "ok" if emb_ok else "bad"),
    unsafe_allow_html=True,
)

index = get_index(provider)
st.sidebar.markdown("###### Library")
c1, c2 = st.sidebar.columns(2)
c1.metric("Docs", len(index.doc_names))
c2.metric("Chunks", len(index.chunks))
st.sidebar.caption(f"index: `{provider}` · dim {emb_dim}")

st.sidebar.markdown("###### Search defaults")
search_method = st.sidebar.radio(
    "Method", ["hybrid", "semantic", "keyword"], index=0, horizontal=True,
    help="keyword = BM25 · semantic = embeddings · hybrid = reciprocal-rank fusion",
)
top_k = st.sidebar.slider("Results (top-k)", 1, 25, settings.default_top_k)

if not openai_ok:
    st.sidebar.error("Set OPENAI_API_KEY (in .env) and restart to enable extraction"
                     + (" and embeddings." if emb_needs_key else "."))

# ----------------------------------------------------------------------------- header
st.markdown(
    '<div class="app-header"><h1>🧬 LNP Metadata Extractor</h1>'
    '<p>Upload a lipid-nanoparticle paper → extract 65 structured fields → '
    'search with keyword + semantic + hybrid retrieval</p></div>',
    unsafe_allow_html=True,
)

tab_ingest, tab_search, tab_library = st.tabs(
    ["📥  Upload & Extract", "🔎  Search", "📊  Library"]
)


# ====================================================================== Upload & Extract
def render_metadata(meta: dict, source_name: str, ctx: str = "", smiles_filled: dict | None = None) -> None:
    uncertain = set(meta.get("uncertain_fields") or [])
    smiles_filled = smiles_filled or {}
    filled, total = completeness(meta)

    # Highlight row
    m = st.columns(5)
    m[0].metric("Diameter", fmt(meta.get("hydrodynamic_diameter_nm")))
    m[1].metric("PDI", fmt(meta.get("pdi")))
    m[2].metric("Zeta", fmt(meta.get("zeta_potential_mv")))
    m[3].metric("Encaps. eff.", fmt(meta.get("encapsulation_efficiency_pct")))
    m[4].metric("Completeness", f"{filled}/{total}")
    st.progress(filled / total)

    if uncertain:
        st.warning("⚠ Fields the model flagged as uncertain: " + ", ".join(sorted(uncertain)))
    if smiles_filled:
        st.info(
            "🧬 SMILES auto-filled from the reference (not from the paper): "
            + "  ·  ".join(f"**{k.replace('_smiles', '')}** ({v})" for k, v in smiles_filled.items())
        )

    # Formulation composition mini-table
    comp = [
        ("Ionizable lipid", meta.get("il_name"), meta.get("il_mol_pct")),
        ("Phospholipid", meta.get("pl_name"), meta.get("pl_mol_pct")),
        ("Cholesterol", meta.get("chol_type"), meta.get("chol_mol_pct")),
        ("PEG-lipid", meta.get("peg_lipid_name"), meta.get("peg_lipid_mol_pct")),
    ]
    comp_rows = [
        {"Component": c, "Name": fmt(n), "mol %": fmt(p)}
        for c, n, p in comp
        if not (is_empty(n) and is_empty(p))
    ]
    if comp_rows:
        st.markdown("**Lipid composition**")
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")

    only_filled = st.toggle("Hide empty fields", value=False, key=f"hide_{ctx}_{source_name}")
    st.markdown("**All fields**")
    for group, fields in FIELD_GROUPS.items():
        g_filled = sum(0 if is_empty(meta.get(f)) else 1 for f in fields)
        rows = [
            {"Field": f, "Value": fmt(meta.get(f)), "⚠": "⚠" if f in uncertain else ""}
            for f in fields
            if not (only_filled and is_empty(meta.get(f)))
        ]
        if not rows:
            continue
        with st.expander(f"{group}  ·  {g_filled}/{len(fields)} filled",
                         expanded=group in ("ID", "Ionizable Lipid", "Particle")):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    flat = {f: meta.get(f) for f in ALL_FIELDS}
    st.download_button(
        "⬇  Download this record (CSV)",
        pd.DataFrame([flat]).to_csv(index=False).encode(),
        file_name=f"{Path(source_name).stem}_metadata.csv",
        mime="text/csv",
        key=f"dl_{ctx}_{source_name}",
    )


with tab_ingest:
    left, right = st.columns([3, 2])
    with left:
        uploaded = st.file_uploader("Drop a PDF here", type=["pdf"], label_visibility="collapsed")
    with right:
        do_extract = st.checkbox("Run metadata extraction (GPT)", value=True)
        add_smiles = st.checkbox(
            "Auto-fill SMILES from component names", value=True,
            help="Resolves ionizable lipid / phospholipid / cholesterol / PEG-lipid "
            "names to SMILES via a PubChem-backed reference. Polymers use anchor / "
            "repeat-unit representations (flagged).",
        )
        st.caption("Always chunks + embeds for search. Extraction is optional.")

    if uploaded is not None:
        size_kb = len(uploaded.getvalue()) / 1024
        st.caption(f"📄 **{uploaded.name}** · {size_kb:,.0f} KB")
        go = st.button("⚙  Ingest & extract", type="primary", width="stretch")

        if go:
            if (do_extract or emb_needs_key) and not openai_ok:
                st.error(
                    "OPENAI_API_KEY is not set — set it"
                    + ("" if do_extract else ", or switch the embedding provider to 'local'.")
                )
            else:
                steps = "Reading → chunking → embedding" + (" → extracting…" if do_extract else "…")
                with st.status(steps, expanded=False) as status:
                    try:
                        result = ingest_pdf(
                            uploaded.getvalue(), doc_name=uploaded.name, index=index,
                            extract=do_extract, add_smiles=add_smiles,
                        )
                        status.update(label="Done", state="complete")
                    except Exception as exc:
                        status.update(label="Failed", state="error")
                        st.error(f"Ingestion failed: {exc}")
                        result = None

                if result is not None:
                    get_index.clear()
                    if result.already_indexed:
                        st.info(f"`{result.doc_name}` was already in the index "
                                f"({result.text_chars:,} chars). Re-extracted below.")
                    else:
                        st.success(f"Indexed `{result.doc_name}` — {result.num_chunks} chunks "
                                   f"from {result.text_chars:,} characters.")
                    if result.metadata is not None:
                        st.divider()
                        render_metadata(result.metadata.model_dump(), result.doc_name,
                                        ctx="ingest", smiles_filled=result.smiles_filled)
    else:
        st.info("Upload a PDF to extract its LNP metadata and add it to the searchable library.")


# ====================================================================== Search
with tab_search:
    if "query" not in st.session_state:
        st.session_state.query = ""

    st.markdown("**Try:**")
    examples = [
        "ionizable lipid pKa",
        "encapsulation efficiency",
        "hydrodynamic diameter and PDI",
        "route of administration and dose",
        "anti-PEG immunogenicity",
    ]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, key=f"ex_{ex}", width="stretch"):
            st.session_state.query = ex

    query = st.text_input(
        "Search query", key="query",
        placeholder="e.g. ionizable lipid pKa and encapsulation efficiency",
    )

    doc_options = sorted(set(index.doc_names.values()))
    doc_filter = st.multiselect("Filter by document (optional)", doc_options, default=[])

    if query:
        if not index.chunks:
            st.info("No documents indexed yet — upload a PDF in the first tab.")
        else:
            results = index.search(query, method=search_method, top_k=top_k)
            if doc_filter:
                results = [r for r in results if r.chunk.doc_name in doc_filter]
            if not results:
                st.warning("No matches.")
            else:
                max_score = max((r.score for r in results), default=1.0) or 1.0
                st.caption(f"{len(results)} result(s) · method **{search_method}**")
                for i, r in enumerate(results, 1):
                    with st.container(border=True):
                        meta_line = (
                            f'<div class="res-meta">#{i} · 📄 <b>{html.escape(r.chunk.doc_name)}</b>'
                            f' · chunk {r.chunk.chunk_index} · '
                            f'{pill(r.method, "info")} · score {r.score:.3f}</div>'
                        )
                        st.markdown(meta_line, unsafe_allow_html=True)
                        st.progress(min(max(r.score / max_score, 0.0), 1.0))
                        st.markdown(
                            f'<div class="res-text">{highlight_html(r.chunk.text, query)}</div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.info("Enter a query above, or click an example.")


# ====================================================================== Library
with tab_library:
    records = load_all_metadata()
    if not records:
        st.info("No metadata extracted yet. Extract a paper in the first tab.")
    else:
        comps = [completeness(r)[0] for r in records]
        avg_pct = 100 * sum(comps) / (len(records) * len(ALL_FIELDS))
        k = st.columns(3)
        k[0].metric("Documents", len(records))
        k[1].metric("Fields tracked", len(ALL_FIELDS))
        k[2].metric("Avg completeness", f"{avg_pct:.0f}%")

        st.divider()
        flt = st.text_input("Filter by document name", placeholder="substring match…")
        group_choice = st.multiselect(
            "Field groups to show", list(FIELD_GROUPS.keys()),
            default=["ID", "Particle"],
        )
        shown_fields = [f for g in group_choice for f in FIELD_GROUPS[g]] or ALL_FIELDS

        df = pd.DataFrame(records)
        for c in ["doc_name"] + ALL_FIELDS:
            if c not in df.columns:
                df[c] = None
        if flt:
            df = df[df["doc_name"].str.contains(flt, case=False, na=False)]
        # Coerce every displayed cell to a string ("—" for empties). Avoids Arrow
        # serialization errors when a column mixes old numeric records with new
        # verbatim-string ones, and renders consistently.
        view = df[["doc_name"] + shown_fields].apply(lambda col: col.map(fmt))
        st.dataframe(
            view, width="stretch", hide_index=True,
            column_config={"doc_name": st.column_config.TextColumn("Document", pinned=True)},
        )
        st.download_button(
            "⬇  Download full library (CSV)",
            df[["doc_name"] + ALL_FIELDS].to_csv(index=False).encode(),
            file_name="lnp_metadata_library.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("##### Inspect / manage a document")
        names = {f'{r["doc_name"]}  ({r["doc_id"][:8]})': r for r in records}
        pick = st.selectbox("Select a document", list(names.keys()))
        if pick:
            rec = names[pick]
            render_metadata(rec, rec["doc_name"], ctx="lib")
            with st.popover("🗑  Delete this document"):
                st.write(f"Remove **{rec['doc_name']}** from the index and metadata?")
                if st.button("Confirm delete", type="primary"):
                    delete_document(rec["doc_id"], index=index)
                    get_index.clear()
                    st.success("Deleted. Rerun / switch tabs to refresh.")
                    st.rerun()

    st.divider()
    with st.expander("🧬 SMILES reference (name → SMILES)"):
        st.caption(
            "Small molecules: authoritative PubChem SMILES. PEG-lipids: lipid **anchor** "
            "(PEG block omitted). Polymers: **repeat-unit** monomer. Always verify before "
            "downstream chemistry. This is the table used to auto-fill SMILES during extraction."
        )
        ref_df = pd.DataFrame(reference_entries())
        st.dataframe(ref_df, width="stretch", hide_index=True)
        st.download_button(
            "⬇  Download SMILES reference (CSV)",
            ref_df.to_csv(index=False).encode(),
            file_name="lnp_smiles_reference.csv",
            mime="text/csv",
        )
