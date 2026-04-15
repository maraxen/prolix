#!/usr/bin/env bash
# Example: download a PDB mmCIF from RCSF into data/pdb/ (requires curl, optional: gzip).
# Usage: PDB_ID=4m8j bash scripts/fetch_pdb_example.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ID="${PDB_ID:-4m8j}"
OUT="${ROOT}/data/pdb/${ID}.pdb"
mkdir -p "${ROOT}/data/pdb"
# RCSB HTTP API: assembly PDB (may be large). Prefer placing your lab-prepared PDB if available.
URL="https://files.rcsb.org/download/${ID}.pdb"
echo "Fetching ${URL} -> ${OUT}"
curl -fsSL -o "${OUT}.tmp" "${URL}"
mv "${OUT}.tmp" "${OUT}"
echo "Wrote ${OUT}"
