"""Name -> SMILES resolution for LNP components.

- Small molecules (ionizable lipids, phospholipids, sterols) -> authoritative
  SMILES from PubChem.
- PEG-lipids -> the lipid-anchor SMILES from PubChem, with the PEG block omitted
  (a polymer has no single canonical SMILES); flagged approximate.
- Polymer coatings -> the repeat-unit monomer SMILES; flagged approximate.

A version-controlled baseline cache ships in `reference/smiles_cache.json` so the
common components resolve offline. Names not in the baseline are looked up live on
PubChem (when enabled) and written to a runtime cache under ./data.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings

# Shipped, version-controlled baseline (correct SMILES for common components).
BASELINE_PATH = Path(__file__).resolve().parent / "reference" / "smiles_cache.json"
# Writable cache for names resolved live at runtime.
RUNTIME_PATH = settings.data_dir / "reference" / "smiles_cache.json"

_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/SMILES/TXT"
_PUBCHEM_CID = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/TXT"

# kind: "small" (full molecule), "anchor" (PEG-lipid anchor; PEG omitted),
#       "polymer" (repeat-unit monomer). query = the PubChem name to look up.
REGISTRY: List[dict] = [
    # Ionizable / cationic lipids
    {"key": "dlin-mc3-dma", "query": "DLin-MC3-DMA", "kind": "small", "aliases": ["mc3", "mc-3", "d-lin-mc3-dma"]},
    {"key": "sm-102", "query": "SM-102", "kind": "small", "aliases": ["sm102"]},
    {"key": "alc-0315", "query": "ALC-0315", "kind": "small", "aliases": ["alc0315", "alc 0315"]},
    {"key": "dlin-kc2-dma", "query": "DLin-KC2-DMA", "kind": "small", "aliases": ["kc2", "dlin-kc2"]},
    {"key": "c12-200", "query": "C12-200", "kind": "small", "aliases": ["c12200"]},
    {"key": "dodap", "query": "DODAP", "kind": "small", "aliases": []},
    {"key": "dotap", "query": "DOTAP", "kind": "small", "aliases": []},
    {"key": "dotma", "query": "DOTMA", "kind": "small", "aliases": []},
    {"key": "dodma", "query": "DODMA", "kind": "small", "aliases": []},
    # Phospholipids
    {"key": "dspc", "query": "DSPC", "kind": "small", "aliases": []},
    {"key": "dope", "query": "DOPE", "kind": "small", "aliases": []},
    {"key": "dppc", "query": "DPPC", "kind": "small", "aliases": []},
    {"key": "dopc", "query": "DOPC", "kind": "small", "aliases": []},
    {"key": "popc", "query": "POPC", "kind": "small", "aliases": []},
    # Sterols
    {"key": "cholesterol", "query": "cholesterol", "kind": "small", "aliases": ["chol"]},
    {"key": "beta-sitosterol", "query": "beta-sitosterol", "kind": "small",
     "aliases": ["sitosterol", "b-sitosterol", "β-sitosterol"]},
    # PEG-lipids -> anchor only (PEG block omitted)
    {"key": "dmg-peg2000", "query": "1,2-dimyristoylglycerol", "kind": "anchor", "anchor": "DMG",
     "aliases": ["peg2000-dmg", "dmg-peg 2000", "dmg-peg", "peg-dmg", "dmg-peg2000", "peg-dmg2000"]},
    {"key": "dspe-peg2000", "query": "1,2-distearoyl-sn-glycero-3-phosphoethanolamine", "kind": "anchor", "anchor": "DSPE",
     "aliases": ["peg2000-dspe", "dspe-peg", "peg-dspe", "dspe-peg2000"]},
    {"key": "dsg-peg2000", "query": "1,2-distearoylglycerol", "kind": "anchor", "anchor": "DSG",
     "aliases": ["peg2000-dsg", "dsg-peg", "dsg-peg2000"]},
    # Polymer coatings -> repeat-unit monomer
    {"key": "polysarcosine", "query": "sarcosine", "kind": "polymer", "monomer": "sarcosine",
     "aliases": ["psar", "poly-sarcosine", "poly(sarcosine)"]},
    {"key": "polyoxazoline", "query": "2-ethyl-2-oxazoline", "kind": "polymer", "monomer": "2-ethyl-2-oxazoline",
     "aliases": ["poly(2-ethyl-2-oxazoline)", "pox", "peox", "poly-2-ethyl-2-oxazoline"]},
    {"key": "polyglycerol", "query": "glycerol", "kind": "polymer", "monomer": "glycerol",
     "aliases": ["pg", "hyperbranched polyglycerol", "hpg"]},
]

_BY_KEY = {e["key"]: e for e in REGISTRY}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _squash(name: str) -> str:
    # collapse to alphanumerics for tolerant alias matching
    return re.sub(r"[^a-z0-9]+", "", name.lower())


# alias map keyed by both normalized and squashed forms -> canonical key
_ALIAS_TO_KEY: Dict[str, str] = {}
for _e in REGISTRY:
    for _name in [_e["key"], *_e["aliases"]]:
        _ALIAS_TO_KEY[_norm(_name)] = _e["key"]
        _ALIAS_TO_KEY[_squash(_name)] = _e["key"]


@dataclass
class SmilesHit:
    smiles: str
    source: str        # provenance, e.g. "PubChem CID 5997" or "anchor (DMG); PEG omitted"
    approximate: bool   # True for anchor-only / repeat-unit representations


# ----------------------------------------------------------------- cache I/O
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _cache() -> dict:
    merged = _read_json(BASELINE_PATH)
    merged.update(_read_json(RUNTIME_PATH))  # runtime overrides/extends baseline
    return merged


def _save_runtime(key: str, hit: SmilesHit) -> None:
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _read_json(RUNTIME_PATH)
    data[key] = {"smiles": hit.smiles, "source": hit.source, "approximate": hit.approximate}
    RUNTIME_PATH.write_text(json.dumps(data, indent=2))
    _cache.cache_clear()


# ----------------------------------------------------------------- PubChem
def _pubchem(query: str, timeout: float = 10.0) -> Optional[tuple[str, Optional[str]]]:
    """Return (smiles, cid) for a name, or None. Network call."""
    q = urllib.parse.quote(query, safe="")
    try:
        with urllib.request.urlopen(_PUBCHEM.format(q=q), timeout=timeout) as r:
            smiles = r.read().decode().strip().splitlines()[0].strip()
        if not smiles or smiles.lower().startswith(("status", "fault")):
            return None
    except Exception:
        return None
    cid = None
    try:
        with urllib.request.urlopen(_PUBCHEM_CID.format(q=q), timeout=timeout) as r:
            cid = r.read().decode().strip().splitlines()[0].strip()
    except Exception:
        pass
    return smiles, cid


def _source_for(entry: dict, cid: Optional[str]) -> str:
    cid_str = f"PubChem CID {cid}" if cid else "PubChem"
    if entry["kind"] == "anchor":
        return f"anchor ({entry['anchor']}); PEG omitted · {cid_str}"
    if entry["kind"] == "polymer":
        return f"repeat unit ({entry['monomer']}) · {cid_str}"
    return cid_str


def _resolve_registry_entry(entry: dict, allow_network: bool) -> Optional[SmilesHit]:
    key = entry["key"]
    cached = _cache().get(key)
    if cached:
        return SmilesHit(cached["smiles"], cached["source"], cached["approximate"])
    if not allow_network:
        return None
    res = _pubchem(entry["query"])
    if not res:
        return None
    smiles, cid = res
    hit = SmilesHit(smiles, _source_for(entry, cid), entry["kind"] != "small")
    _save_runtime(key, hit)
    return hit


def _network_default() -> bool:
    return os.getenv("LNP_PUBCHEM", "1") not in ("0", "false", "False", "")


def resolve_smiles(name: Optional[str], allow_network: Optional[bool] = None) -> Optional[SmilesHit]:
    """Resolve a component name to a SmilesHit, or None if unknown.

    Matching: registry alias (exact / squashed) first, then a direct PubChem
    name lookup as a fallback for small molecules not in the registry.
    """
    if not name or not name.strip():
        return None
    if allow_network is None:
        allow_network = _network_default()

    key = _ALIAS_TO_KEY.get(_norm(name)) or _ALIAS_TO_KEY.get(_squash(name))
    if key:
        return _resolve_registry_entry(_BY_KEY[key], allow_network)

    # Unknown name: cached direct hit, else live PubChem (treated as a full molecule).
    direct = _cache().get(_norm(name))
    if direct:
        return SmilesHit(direct["smiles"], direct["source"], direct["approximate"])
    if not allow_network:
        return None
    res = _pubchem(name)
    if not res:
        return None
    smiles, cid = res
    hit = SmilesHit(smiles, f"PubChem CID {cid}" if cid else "PubChem", False)
    _save_runtime(_norm(name), hit)
    return hit


def reference_entries() -> List[dict]:
    """All known registry components with their (cached) SMILES — for display/export."""
    rows = []
    for e in REGISTRY:
        hit = _resolve_registry_entry(e, allow_network=False)
        rows.append({
            "name": e["key"],
            "kind": e["kind"],
            "smiles": hit.smiles if hit else "",
            "source": hit.source if hit else "(not cached — run build_smiles_cache.py)",
            "approximate": hit.approximate if hit else None,
            "aliases": ", ".join(e["aliases"]),
        })
    return rows
