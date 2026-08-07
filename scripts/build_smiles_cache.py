"""Build the shipped SMILES baseline cache from PubChem.

Run when REGISTRY changes:  python scripts/build_smiles_cache.py
Writes src/lnp_rag/reference/smiles_cache.json (version-controlled).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnp_rag.smiles import BASELINE_PATH, REGISTRY, _pubchem, _source_for  # noqa: E402


def main() -> None:
    out = {}
    for entry in REGISTRY:
        res = _pubchem(entry["query"])
        if not res:
            print(f"MISS  {entry['key']:18s} (query={entry['query']!r})")
            continue
        smiles, cid = res
        out[entry["key"]] = {
            "smiles": smiles,
            "source": _source_for(entry, cid),
            "approximate": entry["kind"] != "small",
        }
        print(f"OK    {entry['key']:18s} {smiles[:50]}")
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nWrote {len(out)} entries -> {BASELINE_PATH}")


if __name__ == "__main__":
    main()
