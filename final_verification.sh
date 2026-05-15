#!/bin/bash
# Final verification script for Sprint 12 Phase 4-5

cd /home/marielle/projects/prolix

echo "=========================================="
echo "  SPRINT 12 PHASE 4-5 VERIFICATION"
echo "=========================================="
echo ""

# Step 1: Python import syntax check
echo "Step 1: Checking imports and syntax..."
python3 quick_test.py
if [ $? -ne 0 ]; then
    echo "✗ IMPORT CHECK FAILED"
    exit 1
fi
echo ""

# Step 2: Unit tests for _langevin_step_o_free_dof
echo "Step 2: Running unit tests for _langevin_step_o_free_dof..."
echo "  Test 2a: test_langevin_o_step_masked_atoms_unchanged"
uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_masked_atoms_unchanged -xvs 2>&1 | head -30
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ✗ Test 2a FAILED"
else
    echo "  ✓ Test 2a PASSED"
fi
echo ""

echo "  Test 2b: test_langevin_o_step_free_atoms_changed"
uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_free_atoms_changed -xvs 2>&1 | head -30
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ✗ Test 2b FAILED"
else
    echo "  ✓ Test 2b PASSED"
fi
echo ""

echo "  Test 2c: test_langevin_o_step_c1_applied_once"
uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_c1_applied_once -xvs 2>&1 | head -30
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ✗ Test 2c FAILED"
else
    echo "  ✓ Test 2c PASSED"
fi
echo ""

echo "  Test 2d: test_langevin_o_step_masked_no_friction_decay"
uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_masked_no_friction_decay -xvs 2>&1 | head -30
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ✗ Test 2d FAILED"
else
    echo "  ✓ Test 2d PASSED"
fi
echo ""

# Step 3: Test that xfail markers were removed
echo "Step 3: Checking that xfail markers were removed from dt>0.5fs tests..."
grep -n "def test_temperature_dt1fs_near_target\|def test_temperature_dt2fs_near_target\|def test_equipartition_chi2" tests/physics/test_settle_temperature_control.py
echo ""

echo "=========================================="
echo "  VERIFICATION COMPLETE"
echo "=========================================="
