#!/usr/bin/env python3
"""Quick verification that modified modules import without syntax errors."""

import sys
import traceback

print("Verifying imports of modified modules...")

try:
    print("  - Importing prolix.physics.settle...")
    from prolix.physics import settle
    print("    ✓ settle imported successfully")

    # Check that the new function exists
    if hasattr(settle, '_langevin_step_o_free_dof'):
        print("    ✓ _langevin_step_o_free_dof function found")
    else:
        print("    ✗ _langevin_step_o_free_dof function NOT found")
        sys.exit(1)

except Exception as e:
    print(f"    ✗ Error importing settle: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("  - Importing prolix.physics.step_system...")
    from prolix.physics import step_system
    print("    ✓ step_system imported successfully")

    # Check that O_Step exists and has apply method
    if hasattr(step_system, 'O_Step'):
        print("    ✓ O_Step class found")
        if hasattr(step_system.O_Step, 'apply'):
            print("    ✓ O_Step.apply method found")
        else:
            print("    ✗ O_Step.apply method NOT found")
            sys.exit(1)
    else:
        print("    ✗ O_Step class NOT found")
        sys.exit(1)

except Exception as e:
    print(f"    ✗ Error importing step_system: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("  - Importing tests.physics.test_settle_temperature_control...")
    import tests.physics.test_settle_temperature_control as test_module
    print("    ✓ test module imported successfully")

    # Check that new test functions exist
    test_funcs = [
        'test_langevin_o_step_masked_atoms_unchanged',
        'test_langevin_o_step_free_atoms_changed',
        'test_langevin_o_step_c1_applied_once',
        'test_langevin_o_step_masked_no_friction_decay',
    ]

    for func_name in test_funcs:
        if hasattr(test_module, func_name):
            print(f"    ✓ {func_name} found")
        else:
            print(f"    ✗ {func_name} NOT found")
            sys.exit(1)

except Exception as e:
    print(f"    ✗ Error importing test module: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All imports successful! Modules are syntactically correct.")
