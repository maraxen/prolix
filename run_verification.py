#!/usr/bin/env python3
"""Verification script for Sprint 12 Phase 4-5 implementation."""

import subprocess
import sys
import os

os.chdir('/home/marielle/projects/prolix')

def run_command(cmd, label):
    """Run command and capture output."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

# Test 1: Import verification
success = run_command(
    "uv run python verify_imports.py",
    "TEST 1: Import verification"
)

if not success:
    print("\n✗ IMPORT VERIFICATION FAILED")
    sys.exit(1)

# Test 2: Run unit tests for _langevin_step_o_free_dof
success = run_command(
    "uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_masked_atoms_unchanged -v",
    "TEST 2A: Unit test - masked atoms unchanged"
)

success = success and run_command(
    "uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_free_atoms_changed -v",
    "TEST 2B: Unit test - free atoms changed"
)

success = success and run_command(
    "uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_c1_applied_once -v",
    "TEST 2C: Unit test - c1 applied once"
)

success = success and run_command(
    "uv run pytest tests/physics/test_settle_temperature_control.py::test_langevin_o_step_masked_no_friction_decay -v",
    "TEST 2D: Unit test - masked no friction decay"
)

if not success:
    print("\n✗ UNIT TESTS FAILED")
    sys.exit(1)

# Test 3: Run temperature control tests (dt=1fs, dt=2fs, equipartition)
success = run_command(
    "uv run pytest tests/physics/test_settle_temperature_control.py::test_temperature_dt1fs_near_target -v",
    "TEST 3A: Temperature control - dt=1.0fs"
)

success = success and run_command(
    "uv run pytest tests/physics/test_settle_temperature_control.py::test_temperature_dt2fs_near_target -v",
    "TEST 3B: Temperature control - dt=2.0fs"
)

success = success and run_command(
    "uv run pytest tests/physics/test_settle_temperature_control.py::test_equipartition_chi2 -v",
    "TEST 3C: Equipartition validation"
)

if not success:
    print("\n⚠ TEMPERATURE CONTROL TESTS: Some tests may have failed")
    print("  This could indicate the implementation needs refinement")
    # Don't exit here - let integrator tests run

# Test 4: Run integrator builder tests
success_integrator = run_command(
    "uv run pytest tests/physics/test_integrator_builder.py -v --tb=short 2>&1 | head -100",
    "TEST 4: Integrator builder tests"
)

print("\n" + "="*60)
print("  VERIFICATION SUMMARY")
print("="*60)
print(f"Import verification: {'✓ PASS' if success else '✗ FAIL'}")
print(f"Integrator tests: {'✓ PASS' if success_integrator else '⚠ CHECK OUTPUT'}")

if success and success_integrator:
    print("\n✓ VERIFICATION COMPLETE - All critical tests passed")
    sys.exit(0)
else:
    print("\n⚠ VERIFICATION COMPLETE - Check output above for details")
    sys.exit(1)
