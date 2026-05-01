#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sprint_b_discriminator
#SBATCH --output=outputs/logs/sprint_b_discriminator_%j.out
#SBATCH --error=outputs/logs/sprint_b_discriminator_%j.err

cd ~/projects/prolix

echo "=== Sprint B Step 1: Discriminator Test ==="
echo "Configuration: 64 waters, dt=0.25 fs, 100 ps NVT"
echo "Running direct test measurement..."
echo

# Set JAX to CPU + enable float64
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

# Run pytest on the single test function
uv run pytest -xvs tests/physics/test_settle_temperature_control.py::test_temperature_langevin_dt0_5fs_green -k "dt0_5fs" 2>&1 | head -100

echo
echo "Test measurement completed."
