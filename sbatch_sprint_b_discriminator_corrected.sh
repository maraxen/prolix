#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sprint_b_disc
#SBATCH --output=outputs/logs/sprint_b_discriminator_%j.out
#SBATCH --error=outputs/logs/sprint_b_discriminator_%j.err

cd ~/projects/prolix

echo "=== Sprint B Step 1: Corrected Discriminator Test ==="
echo "Configuration: Baseline (8w, dt=0.5fs) + Discriminator (64w, dt=0.25fs)"
echo "Starting at: $(date)"
echo

# Set JAX to CPU + enable float64
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

# Run the custom discriminator measurement
uv run python3 sprint_b_discriminator_measurement.py

echo
echo "Test measurement completed at: $(date)"
