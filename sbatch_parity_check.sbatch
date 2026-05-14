#!/bin/bash
#SBATCH --job-name=prolix-parity
#SBATCH --output=logs/parity_%j.log
#SBATCH --error=logs/parity_%j.err
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Navigate to project directory
cd ~/projects/prolix

echo "Starting Parity Tests on $(hostname)"

# Run tests using the local environment
uv run pytest tests/physics/test_openmm_parity.py -m "not slow"
