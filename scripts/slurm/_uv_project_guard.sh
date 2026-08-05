# shellcheck shell=bash
# Source after computing WORKSPACE_ROOT, with PROLIX_ROOT already exported
# (e.g. by _common_env.sh). Call _resolve_uv_project_for_workspace "$WORKSPACE_ROOT".
#
# Only forces UV_PROJECT to the shared workspace when PROLIX_ROOT actually IS the
# workspace's `prolix` uv-workspace member (i.e. this job is running from the
# shared ~/projects/prolix checkout, not an isolated worktree elsewhere). Debt
# 750 (verify_debt750_1ake_nan.slurm, commit b743dea) hit this the hard way:
# forcing UV_PROJECT unconditionally made an isolated-worktree job silently
# import prolix from whatever branch happened to be checked out in the shared
# workspace instead of this job's own source (job 18523653, AttributeError on
# MolecularShapeSpec.has_real_water because the shared checkout predated debt
# 841's fix). Debt 937 promoted this fix from one script to a shared guard used
# by every script that exercises prolix's own source under an isolated worktree.
#
# Sets:
#   UV_PROJECT     -- exported to WORKSPACE_ROOT if shared, unset if isolated
#   UV_SYNC_EXTRAS -- array; pass as "${UV_SYNC_EXTRAS[@]}" to every uv run/sync
#                     invocation so an isolated venv's implicit sync requests the
#                     same extras the shared workspace already has cached.
_resolve_uv_project_for_workspace() {
  local workspace_root="$1"
  if [[ "$(cd "${workspace_root}/prolix" 2>/dev/null && pwd -P)" == "${PROLIX_ROOT}" ]]; then
    export UV_PROJECT="${workspace_root}"
    UV_SYNC_EXTRAS=()
  else
    echo "Isolated worktree detected (PROLIX_ROOT=${PROLIX_ROOT} != shared ${workspace_root}/prolix) -- using worktree-local uv project, not the shared workspace."
    unset UV_PROJECT || true
    UV_SYNC_EXTRAS=(--extra cuda --extra dev)
  fi
}
