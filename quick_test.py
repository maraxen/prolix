#!/usr/bin/env python3
"""Quick syntax check of modified files."""

import sys
import traceback

try:
    print("Importing settle module...")
    from prolix.physics import settle
    print("  ✓ settle module imported")

    # Check function exists
    if hasattr(settle, '_langevin_step_o_free_dof'):
        print("  ✓ _langevin_step_o_free_dof function found")

        # Get function signature
        import inspect
        sig = inspect.signature(settle._langevin_step_o_free_dof)
        print(f"    Signature: {sig}")
    else:
        print("  ✗ _langevin_step_o_free_dof NOT found")
        sys.exit(1)

except SyntaxError as e:
    print(f"  ✗ Syntax error: {e}")
    traceback.print_exc()
    sys.exit(1)
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\nImporting step_system module...")
    from prolix.physics import step_system
    print("  ✓ step_system module imported")

    if hasattr(step_system, 'O_Step'):
        print("  ✓ O_Step class found")

        # Check apply method
        if hasattr(step_system.O_Step, 'apply'):
            print("  ✓ O_Step.apply method found")
        else:
            print("  ✗ O_Step.apply NOT found")
            sys.exit(1)
    else:
        print("  ✗ O_Step class NOT found")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\nImporting test module...")
    from tests.physics import test_settle_temperature_control
    print("  ✓ test module imported")

    # List key test functions
    test_functions = [
        'test_temperature_dt1fs_near_target',
        'test_temperature_dt2fs_near_target',
        'test_equipartition_chi2',
        'test_langevin_o_step_masked_atoms_unchanged',
        'test_langevin_o_step_free_atoms_changed',
        'test_langevin_o_step_c1_applied_once',
        'test_langevin_o_step_masked_no_friction_decay',
    ]

    for func_name in test_functions:
        if hasattr(test_settle_temperature_control, func_name):
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} NOT found")
            sys.exit(1)

except Exception as e:
    print(f"  ✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All syntax checks passed!")
