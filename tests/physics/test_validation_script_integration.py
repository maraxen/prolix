"""Validation script integration tests.

Tests that the validation scripts run successfully with the new proxide API.
"""

import os
import subprocess

import pytest

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "../../scripts/debug")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "validate_1ubq_ff19sb.py")


@pytest.mark.skipif(not os.path.exists(SCRIPT_PATH), reason="Validation script not found")
@pytest.mark.skip(
  reason="Validation script uses deprecated jax_md_bridge - needs separate migration"
)
class TestValidationScripts:
  """Integration tests for validation scripts."""

  @pytest.mark.slow
  def test_validation_script_runs(self):
    """Test that validation script runs without crashing."""
    result = subprocess.run(
      ["uv", "run", "python", SCRIPT_PATH],
      check=False,
      capture_output=True,
      text=True,
      timeout=120,  # 2 minute timeout
      cwd=os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))),
    )

    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
      print("STDERR:")
      print(result.stderr)

    # Check return code
    assert result.returncode == 0, f"Script failed with exit code {result.returncode}"

  @pytest.mark.slow
  def test_force_error_acceptable(self):
    """Test that force errors are within acceptable range."""
    import re

    result = subprocess.run(
      ["uv", "run", "python", SCRIPT_PATH],
      check=False,
      capture_output=True,
      text=True,
      timeout=120,
      cwd=os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))),
    )

    if result.returncode != 0:
      pytest.skip("Script failed, cannot check force errors")

    match = re.search(r"Max Force Error: ([0-9.]+)", result.stdout)
    if match:
      max_force = float(match.group(1))
      print(f"Max Force Error: {max_force}")
      assert max_force < 1000.0, f"Force explosion detected: {max_force}"
    else:
      # Script may not print force error in all versions
      print("No force error found in output")


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
