#!/bin/bash
#SBATCH --job-name=prolix_p7_cmdbuf_off
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%j.prof_stage2_cmdbuf_off.out
#SBATCH --error=outputs/logs/slurm/%j.prof_stage2_cmdbuf_off.err

# Diagnostic: one P7 cell (1vii, flash, PME-on, 1.0 Å) with XLA command
# buffers empty so executed thunks are ordinary fusions/FFTs, not one
# command_buffer_*. Not the production program (P5). Compare host
# total_step_seconds and hlo_op dump to the cmdbuf-on capture.
#
#   myxcel submit engaging prolix scripts/slurm/prof_stage2_cmdbuf_off.slurm \
#     --partition mit_preemptable --time 00:15:00 \
#     --sbatch-arg chdir=/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628

set -euo pipefail

_locate_root() {
    local cand
    for cand in \
        "${PWD}" \
        "${SLURM_SUBMIT_DIR:-}" \
        "${HOME}/projects/.worktrees/prolix/wt-20260807-132628" \
        "${HOME}/projects/prolix"
    do
        if [[ -n "${cand}" && -f "${cand}/scripts/slurm/_bth_env.sh" ]]; then
            echo "${cand}"
            return 0
        fi
    done
    return 1
}
_PROJ="$(_locate_root)" || {
    echo "ERROR: cannot locate prolix root (PWD=${PWD} SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2
    exit 1
}
cd "${_PROJ}"
echo "P7 cwd=${_PROJ}"
if [[ -z "${PROLIX_GIT_SHA:-}" && -f "${_PROJ}/.git_sha" ]]; then
    PROLIX_GIT_SHA="$(head -1 "${_PROJ}/.git_sha" | tr -d '[:space:]')"
    export PROLIX_GIT_SHA
fi
if [[ -z "${PROLIX_GIT_SHA:-}" ]]; then
    PROLIX_GIT_SHA="$(git -C "${_PROJ}" rev-parse HEAD 2>/dev/null || true)"
    export PROLIX_GIT_SHA
fi
echo "PROLIX_GIT_SHA=${PROLIX_GIT_SHA:-unset}"

# Empty command-buffer list: individual thunks visible to jax.profiler.
# Set before any JAX import (uv run python below).
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
echo "XLA_FLAGS=${XLA_FLAGS}"

source "scripts/slurm/_bth_env.sh"

mkdir -p outputs/profiling/stage2/cmdbuf_off outputs/logs/slurm

echo "=== P7 cmdbuf-off 1vii flash pme-on 1.0 ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

uv run bth run python scripts/experiments/profile_b1_flash_vs_autodiff_forces.py \
    --no-sidecar \
    --tag p7-stage2-cmdbuf-off \
    --tag protein:1vii \
    --tag mode:flash \
    --tag pme:on \
    --tag grid:1.0 \
    --out outputs/profiling/stage2/cmdbuf_off/summary.json \
    -- \
    --protein 1vii \
    --mode flash \
    --pme on \
    --grid-spacing 1.0 \
    --n-warmup 3 --n-trials 20 --n-inner 5 \
    --stage 2 --scopes on \
    --emit-record outputs/profiling/stage2/cmdbuf_off/record.json \
    --trace-dir outputs/profiling/stage2/cmdbuf_off/trace \
    --out outputs/profiling/stage2/cmdbuf_off/summary.json

EXIT=$?
echo "=== probe exit=${EXIT} ==="
if [[ "${EXIT}" -eq 0 ]]; then
    GZ=$(find outputs/profiling/stage2/cmdbuf_off/trace -name '*.trace.json.gz' | head -1 || true)
    if [[ -n "${GZ}" ]]; then
        PYTHONPATH=. uv run python scripts/explore/prof_dump_hlo_ops.py "${GZ}" \
            --out outputs/profiling/stage2/cmdbuf_off/hlo_op_dump.json \
            --top 40
    else
        echo "WARNING: no trace gzip to dump" >&2
    fi
fi
exit ${EXIT}
