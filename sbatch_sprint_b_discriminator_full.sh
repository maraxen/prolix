#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sprint_b_discriminator
#SBATCH --output=outputs/logs/sprint_b_discriminator_%j.out
#SBATCH --error=outputs/logs/sprint_b_discriminator_%j.err

cd ~/projects/prolix

echo "=== Sprint B Step 1: Discriminator Test (FULL) ==="
echo "Configuration: 64 waters, dt=0.25 fs, 100 ps NVT (400k steps)"
echo "Purpose: Determine H1 (integration error) vs H3 (force-side mechanism)"
echo "Started: $(date)"
echo

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu

# Run a minimal Python snippet that calls _mean_rigid_t_after_burn via pytest internals
uv run python3 -m pytest tests/physics/test_settle_temperature_control.py -k "dt0_5fs" -xvs --tb=short 2>&1 | tail -200

echo
echo "Completed: $(date)"
