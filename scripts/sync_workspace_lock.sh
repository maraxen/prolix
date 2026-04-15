#!/usr/bin/env bash
# Refresh workspace/uv.lock from a sibling prolix + proxide layout.
# Expects: ../proxide next to this repo; runs `uv lock` from the common parent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(cd "$ROOT/.." && pwd)"
if [[ ! -d "$PARENT/proxide" ]]; then
  echo "error: expected proxide at $PARENT/proxide (clone proxide next to prolix)." >&2
  exit 1
fi
cp "$ROOT/workspace/pyproject.toml" "$PARENT/pyproject.toml"
(
  cd "$PARENT"
  uv lock
)
cp "$PARENT/uv.lock" "$ROOT/workspace/uv.lock"
echo "Updated $ROOT/workspace/uv.lock"
