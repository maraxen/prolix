# shellcheck shell=bash
# Materialize a venv: **proxide** resolves from PyPI (prebuilt wheels).
# Expects ``WORKSPACE_ROOT`` and ``PROLIX_ROOT`` set. If UV_PROJECT is set (shared
# workspace; see _uv_project_guard.sh), syncs the shared ``~/projects`` venv via its
# `prolix` workspace member. If UV_PROJECT is unset (isolated worktree), syncs a
# local venv from PROLIX_ROOT's own pyproject.toml/uv.lock instead -- forcing the
# shared workspace here would sync the wrong branch (debt 750/937).

_workspace_uv_sync_run() {
  uv python install
  if [[ -n "${UV_PROJECT:-}" ]]; then
    cd "${WORKSPACE_ROOT}"
    uv venv --allow-existing
    uv sync --extra cuda --extra dev --extra openmm --package prolix
  else
    cd "${PROLIX_ROOT}"
    uv venv --allow-existing
    uv sync --extra cuda --extra dev --extra openmm
  fi
}
