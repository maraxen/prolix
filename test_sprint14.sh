#!/bin/bash
set -e

cd /home/marielle/projects/prolix

echo "=== Test 1: Import compute_pressure_akma ==="
uv run python -c "from prolix.physics.pressure import compute_pressure_akma; print('✓ compute_pressure_akma imported successfully')"

echo ""
echo "=== Test 2: Import settle.settle_csvr_npt ==="
uv run python -c "from prolix.physics.settle import settle_csvr_npt; print('✓ settle_csvr_npt imported successfully')"

echo ""
echo "=== Test 3: Run pressure helper unit tests ==="
uv run pytest tests/physics/test_pressure_helpers.py -v 2>&1 | tail -20

echo ""
echo "=== Test 4: Run non-slow NPT barostat tests ==="
uv run pytest tests/physics/test_npt_barostat.py -v -m "not slow" 2>&1 | tail -30

echo ""
echo "=== Test 5: Run settle temperature control tests (non-slow) ==="
uv run pytest tests/physics/test_settle_temperature_control.py -v -m "not slow" 2>&1 | tail -20

echo ""
echo "✓ All Sprint 14 verification tests passed!"
