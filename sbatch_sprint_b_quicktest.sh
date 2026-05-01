#!/bin/bash
#SBATCH --partition=mit_quicktest
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=sprint_b_quicktest
#SBATCH --output=outputs/logs/sprint_b_quicktest_%j.out
#SBATCH --error=outputs/logs/sprint_b_quicktest_%j.err

cd ~/projects/prolix

echo "=== Sprint B Quicktest: Validate code runs on cluster ==="
echo "Running 1 small test (8 waters, 10 ps) via pytest..."
echo

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu

# Run the simplest existing test to validate setup works
uv run pytest -xvs tests/physics/test_settle_temperature_control.py::test_temperature_dt1fs_near_target --tb=short 2>&1 | head -100

echo
echo "✅ If above test ran (even if xfail), code setup is working."
echo "Ready for full discriminator job submission."
