#!/bin/bash
# Phase 2 scale: DHFR (catalog key DHFR → ``data/pdb/4m8j.pdb`` when present, else ``1UBQ`` proxy).
# Long wall time on preemptable queue; increase ``--mem`` if your site requires it.
#
# Recommended: place ``4m8j.pdb`` under ``data/pdb/`` (see ``scripts/fetch_pdb_example.sh``).
#
#SBATCH --job-name=plx-dhfr
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

uv run python scripts/benchmark_nlvsdense.py --systems DHFR --replicas 1 --cutoffs 8.0 12.0 16.0 \
  2>&1 | tee "${LOG_ROOT}/app/dhfr_nlvsdense_${SLURM_JOB_ID:-local}.log"

# Optional OpenMM spot-check (same FF/box assumptions as your validation script):
# uv run python benchmarks/compare_jax_openmm_validity.py ... 2>&1 | tee "${LOG_ROOT}/app/dhfr_openmm_${SLURM_JOB_ID:-local}.log"
