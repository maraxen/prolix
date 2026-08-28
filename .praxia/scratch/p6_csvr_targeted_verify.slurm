#!/bin/bash
# Phase 6 VERIFY remediation for backlog #4167: targeted re-run of the 3 CSVR tests
# the auditor flagged as at concrete risk of flipping xfail(strict=True) -> XPASS
# now that _langevin_settle_vel actually removes COM momentum (commit 2950fb0).
# Same 3 tests as scripts/slurm/validate_csvr_temperature.slurm; that script's
# relative source of _common_env.sh fails under myxcel's worktree-routed submission
# (SLURM_SUBMIT_DIR comes through empty), so this copy uses the known-working
# hardcoded-cwd workaround from p6_csvr_full_verify.slurm instead of touching the
# shared reference script.
#SBATCH --job-name=p6-csvr-targeted-verify
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

_PROJ="/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628"
cd "${_PROJ}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# node4007 has 32 total CPUs shared across other jobs (CPUAlloc=17 observed while
# this job only holds --cpus-per-task=4 via cgroup); JAX/XLA's CPU backend does not
# respect the cgroup limit on its own and was oversubscribing threads against the
# node's other tenants, which is the likely cause of prior runs hanging past the
# pytest-timeout 900s limit mid-primitive-dispatch. Cap explicitly to the allocation.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4 inter_op_parallelism_threads=4"

echo "p6-csvr-targeted-verify job=${SLURM_JOB_ID:-local} host=$(hostname) cwd=$(pwd) JAX_PLATFORMS=${JAX_PLATFORMS} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

uv run pytest \
  tests/physics/test_settle_temperature_control.py::test_temperature_csvr_dt1fs_near_target \
  tests/physics/test_settle_temperature_control.py::test_temperature_csvr_dt2fs_near_target \
  tests/physics/test_settle_temperature_control.py::test_equipartition_csvr_dt2fs_chi2 \
  -v --tb=short --timeout=3600 --json-report --json-report-file=tmp/pytest_csvr_targeted_verify.json
