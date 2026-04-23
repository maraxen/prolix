# shellcheck shell=bash
# Materialize ``~/projects/.venv``: **proxide** resolves from PyPI (prebuilt wheels).
# Expects ``WORKSPACE_ROOT`` and ``UV_PROJECT`` set.

_workspace_uv_sync_run() {
  cd "${WORKSPACE_ROOT}"
  uv python install
  uv venv --allow-existing
  uv sync --extra cuda --extra dev --extra openmm --package prolix
}
