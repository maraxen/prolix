# shellcheck shell=bash
# Shared uv environment step for Engaging SLURM scripts. `source` it AFTER cd-ing to the repo root.
#
# Under myxcel (myxcel.toml [uv] venv = "lock-keyed") the job env carries MYXCEL_UV_READY and
# UV_PROJECT_ENVIRONMENT, naming ONE venv per (uv.lock, sync_args, python) key, shared by every
# worktree and array task. Because that venv outlives any single checkout, prolix itself is NOT
# installed into it (an editable install would point at whichever checkout built it, and every
# other worktree would silently import that source -- the debt-750 failure mode). Instead:
#   * the venv is built once with `uv sync --no-install-project` (dependencies only), under flock
#     so array tasks do not race, and the ready marker is touched LAST;
#   * the CURRENT checkout's src/ is put on PYTHONPATH, so `import prolix` always resolves to this
#     worktree;
#   * UV_NO_SYNC=1 makes every later `uv run` read-only (no implicit sync, no editable install).
# Without myxcel (raw sbatch, MYXCEL_UV_READY unset) prolix_uv_sync does nothing and the script
# keeps its old behaviour (`uv run` syncs the project's own .venv).
#
# Usage:
#   prolix_uv_sync     # converted scripts: build-once + read-only runs under myxcel
#   prolix_uv_opt_out  # scripts that manage their own env (UV_PROJECT workspace mode,
#                      #   `uv run --python`, ~/projects/.venv): drop the myxcel venv injection so
#                      #   they behave exactly as before. Set PROLIX_UV_OPT_OUT=1 before sourcing
#                      #   _common_env.sh to get this automatically.
#
# Keep PROLIX_UV_SYNC_ARGS identical to myxcel.toml [uv] sync_args: the shared venv is built from
# this array, while myxcel's lock key hashes the toml value.

PROLIX_UV_SYNC_ARGS=(--frozen --extra cuda --extra openmm --extra dev --no-install-project)

prolix_uv_opt_out() {
    if [ -n "${MYXCEL_UV_READY:-}" ]; then
        echo "uv: myxcel lock-keyed venv skipped for this script (own env management); using the project default"
        unset MYXCEL_UV_READY UV_PROJECT_ENVIRONMENT UV_NO_SYNC
    fi
}

prolix_uv_sync() {
    if [ -z "${MYXCEL_UV_READY:-}" ]; then
        return 0
    fi
    local root="${PROLIX_ROOT:-${PWD}}"
    if [ -n "${UV_PROJECT:-}" ]; then
        # Shared-workspace mode resolves a different lock; the lock-keyed venv does not apply.
        prolix_uv_opt_out
        return 0
    fi
    if [ ! -f "${root}/src/prolix/__init__.py" ] || [ ! -f "${root}/uv.lock" ]; then
        echo "uv: ${root} is not the prolix repo root (src/prolix or uv.lock missing)" >&2
        return 1
    fi
    if [ ! -e "${MYXCEL_UV_READY}" ]; then
        local lock_file="${UV_PROJECT_ENVIRONMENT}.build.lock"
        mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
        (
            flock 9
            if [ ! -e "${MYXCEL_UV_READY}" ]; then
                echo "uv: building lock-keyed venv ${UV_PROJECT_ENVIRONMENT} (first task for this uv.lock)"
                (cd "${root}" && uv sync "${PROLIX_UV_SYNC_ARGS[@]}") || exit 1
                touch "${MYXCEL_UV_READY}"
            fi
        ) 9>"${lock_file}" || { echo "uv: venv build failed" >&2; return 1; }
    fi
    test -e "${MYXCEL_UV_READY}" || { echo "uv: ready marker missing after build" >&2; return 1; }
    export UV_NO_SYNC=1
    case ":${PYTHONPATH:-}:" in
        *":${root}/src:"*) ;;
        *) export PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
    echo "uv: lock-keyed venv ready at ${UV_PROJECT_ENVIRONMENT} (uv run is --no-sync; prolix from ${root}/src)"
}
