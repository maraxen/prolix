#!/bin/bash
#SBATCH -J prolix-explicit-openmm
#SBATCH -o explicit_openmm_%j.out
#SBATCH -e explicit_openmm_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#
# Explicit-solvent / OpenMM optional checks (not the implicit chignolin NL benchmark).
# Adjust partition/account/GPU flags for your site.
#
# Usage:
#   sbatch scripts/slurm/bench_explicit_openmm_pi_so3.slurm
#
# Expects a venv or module with: pip install -e ".[dev,openmm]"
# and a working OpenMM wheel for the compute node's glibc.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"

echo "=== pytest openmm marker (subset) ==="
pytest -q -m openmm tests/physics/test_openmm_explicit_anchor.py tests/physics/test_solvated_openmm_explicit_parity.py || true

echo "=== T0 throughput JSON ==="
python scripts/benchmarks/prolix_vs_openmm_speed.py --skip-reference --json "t0_bench_${SLURM_JOB_ID:-local}.json"

echo "=== T1-scaled throughput JSON (minimal box, tier T1) ==="
python scripts/benchmarks/prolix_vs_openmm_t1_solvated.py --skip-reference --json "t1_bench_${SLURM_JOB_ID:-local}.json"

echo "Done."
