#!/bin/bash
#SBATCH --job-name=prolix_pme_stencil
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_pme_stencil.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_pme_stencil.err

# L40S 1vii: vectorized SPME stencil.
#   0 force-only flash (perf vs ~1.22 ms baseline)
#   1 md flash buckets on
#   2 Q/energy/FD dump vs 4³ oracle

set -euo pipefail

_locate_root() {
    local cand
    for cand in "${PWD}" "${SLURM_SUBMIT_DIR:-}" \
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
_PROJ="$(_locate_root)" || exit 1
cd "${_PROJ}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi
source "scripts/slurm/_bth_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?}"
mkdir -p outputs/logs/slurm outputs/profiling/pme_stencil
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
uv run --extra cuda python -c "import jax; print('jax.default_backend=', jax.default_backend()); print('jax.devices=', jax.devices())"

set -o pipefail
case "${TASK}" in
    0)
        OUTDIR="outputs/profiling/pme_stencil/l40s_1vii_force_flash"
        mkdir -p "${OUTDIR}"
        uv run --extra cuda python scripts/explore/prof_ensemble_apply_probe.py \
            --system 1vii --arm force --nonbonded flash --use-size-buckets on \
            --emit-record "${OUTDIR}/record.json" --trace-dir "${OUTDIR}/trace" \
            2>&1 | tee "${OUTDIR}/probe.log"
        ;;
    1)
        OUTDIR="outputs/profiling/pme_stencil/l40s_1vii_md_flash"
        mkdir -p "${OUTDIR}"
        uv run --extra cuda python scripts/explore/prof_ensemble_apply_probe.py \
            --system 1vii --arm md --nonbonded flash --use-size-buckets on \
            --emit-record "${OUTDIR}/record.json" --trace-dir "${OUTDIR}/trace" \
            2>&1 | tee "${OUTDIR}/probe.log"
        ;;
    2)
        OUTDIR="outputs/profiling/pme_stencil/l40s_1vii_parity"
        mkdir -p "${OUTDIR}"
        uv run --extra cuda python scripts/explore/check_pme_stencil_parity.py \
            --out "${OUTDIR}/parity.json" \
            2>&1 | tee "${OUTDIR}/parity.log"
        ;;
    *) exit 1 ;;
esac
EXIT=${PIPESTATUS[0]}
echo "=== exit=${EXIT} ==="
exit ${EXIT}
