#!/bin/bash
# Phase 6 VERIFY remediation for backlog #4167: full test_settle_temperature_control.py
# run post-CSVR-COM-momentum-fix (commit 2950fb0), checking whether either
# xfail(strict=True) CSVR test flips to XPASS now that remove_com=True has a real effect.
#SBATCH --job-name=p6-csvr-full-verify
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --time=11:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

# SLURM_SUBMIT_DIR came through empty for this myxcel-worktree-routed submission
# (known myxcel chdir mismatch, cf. commit 9deee3e) and the BASH_SOURCE fallback
# lands in the spool dir's grandparent, not the repo -- hardcode the known
# myxcel worktree scratch path instead of trusting either.
_PROJ="/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628"
cd "${_PROJ}"

# These are CPU-validated tests (see scripts/slurm/validate_csvr_temperature.slurm);
# pi_so3 nodes (node4007/4008) carry an H200 the job doesn't request via --gres,
# so force CPU explicitly rather than let JAX probe for a GPU it can't exclusively use.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

echo "p6-csvr-full-verify job=${SLURM_JOB_ID:-local} host=$(hostname) cwd=$(pwd) JAX_PLATFORMS=${JAX_PLATFORMS}"

uv run pytest tests/physics/test_settle_temperature_control.py -v \
  --json-report --json-report-file=tmp/pytest_full_verify.json
