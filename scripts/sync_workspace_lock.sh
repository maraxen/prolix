#!/usr/bin/env bash
# Refresh workspace/uv.lock from the sibling-layout parent (``../prolix`` = this repo).
# **proxide** is resolved from PyPI; a local ``../proxide`` checkout is not required.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(cd "$ROOT/.." && pwd)"
if [[ ! -d "$PARENT/prolix" ]]; then
  echo "error: expected prolix at $PARENT/prolix (this repo should live at <parent>/prolix)." >&2
  exit 1
fi
cp "$ROOT/workspace/pyproject.toml" "$PARENT/pyproject.toml"
(
  cd "$PARENT"
  uv lock
)
cp "$PARENT/uv.lock" "$ROOT/workspace/uv.lock"
echo "Updated $ROOT/workspace/uv.lock"
