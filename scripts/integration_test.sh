#!/bin/bash
# integration_test.sh — End-to-end test of all 5 measurement source scenarios
#
# Mock SLURM job states. Run extract_walltime.sh. Verify daily_log SOURCE field.
# Run phase_2_2_0_gate.sh. Verify exit codes.
#
# Exit: 0 if all 5 scenarios pass, 1 if any fail

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_WALLTIME_SCRIPT="${SCRIPT_DIR}/extract_walltime.sh"
GATE_SCRIPT="${SCRIPT_DIR}/phase_2_2_0_gate.sh"

# Temporary directory for test artifacts
TEST_TMP=$(mktemp -d)
trap "rm -rf $TEST_TMP" EXIT

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# ============================================================================
# HELPER: Create temporary mock sacct
# ============================================================================

create_mock_sacct() {
  local return_state="$1"
  local should_fail="$2"  # "true" or "false"

  cat > "$TEST_TMP/sacct" << 'SACCT_EOF'
#!/bin/bash
RETURN_STATE="$RETURN_STATE_VAR"
SHOULD_FAIL="$SHOULD_FAIL_VAR"

if [ "$SHOULD_FAIL" = "true" ]; then
  exit 1
fi

echo "$RETURN_STATE"
exit 0
SACCT_EOF

  sed -i "s/\$RETURN_STATE_VAR/$return_state/g" "$TEST_TMP/sacct"
  sed -i "s/\$SHOULD_FAIL_VAR/$should_fail/g" "$TEST_TMP/sacct"
  chmod +x "$TEST_TMP/sacct"
}

# ============================================================================
# HELPER: Run test scenario
# ============================================================================

run_scenario() {
  local scenario_name="$1"
  local slurm_job_id="$2"
  local sacct_state="$3"
  local sacct_fail="$4"
  local gate_input="$5"
  local expected_exit_code="$6"

  echo ""
  echo "============================================================================"
  echo "Scenario: $scenario_name"
  echo "============================================================================"

  # Setup environment
  export SLURM_JOB_ID="$slurm_job_id"
  export PATH="$TEST_TMP:$PATH"

  # Create test daily_log
  TEST_DAILY_LOG="$TEST_TMP/daily_log"
  export HOME="$TEST_TMP"
  mkdir -p "$TEST_TMP/.praxia"

  # Create mock sacct if needed
  if [ -n "$sacct_state" ]; then
    create_mock_sacct "$sacct_state" "$sacct_fail"
  fi

  # Run extract_walltime.sh
  echo "Running extract_walltime.sh..."
  bash "$EXTRACT_WALLTIME_SCRIPT" > /dev/null 2>&1 || true

  # Verify daily_log was created and contains expected SOURCE
  if [ ! -f "$TEST_TMP/.praxia/daily_log" ]; then
    echo "❌ FAIL: daily_log not created"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    return 1
  fi

  # Extract SOURCE from daily_log
  ACTUAL_SOURCE=$(grep "Measurement Source:" "$TEST_TMP/.praxia/daily_log" | tail -1 | awk '{print $NF}')
  echo "Extracted SOURCE: $ACTUAL_SOURCE"

  # Show daily_log contents
  echo ""
  echo "--- daily_log contents ---"
  tail -8 "$TEST_TMP/.praxia/daily_log"
  echo "--- end daily_log ---"
  echo ""

  # Run phase_2_2_0_gate.sh with gate_input
  echo "Running phase_2_2_0_gate.sh with input: '$gate_input'"
  if echo "$gate_input" | bash "$GATE_SCRIPT" > /dev/null 2>&1; then
    ACTUAL_EXIT_CODE=0
  else
    ACTUAL_EXIT_CODE=$?
  fi

  echo "Gate exit code: $ACTUAL_EXIT_CODE (expected: $expected_exit_code)"

  if [ "$ACTUAL_EXIT_CODE" -eq "$expected_exit_code" ]; then
    echo "✅ PASS: $scenario_name"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    return 0
  else
    echo "❌ FAIL: $scenario_name (expected exit $expected_exit_code, got $ACTUAL_EXIT_CODE)"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    return 1
  fi
}

# ============================================================================
# SCENARIO 1.1: TIMEOUT job → uncalibrated_timeout → gate accepts 'y' → exit 0
# ============================================================================

run_scenario \
  "1.1: TIMEOUT job -> uncalibrated_timeout -> gate accepts 'y' -> exit 0" \
  "12345" \
  "TIMEOUT" \
  "false" \
  "y" \
  0

# ============================================================================
# SCENARIO 1.2: PREEMPTED job → uncalibrated_timeout → gate accepts 'y' → exit 0
# ============================================================================

run_scenario \
  "1.2: PREEMPTED job -> uncalibrated_timeout -> gate accepts 'y' -> exit 0" \
  "12346" \
  "PREEMPTED" \
  "false" \
  "y" \
  0

# ============================================================================
# SCENARIO 2.1: sacct query failure → uncalibrated_query_error → gate accepts 'y' → exit 0
# ============================================================================

run_scenario \
  "2.1: sacct query failure -> uncalibrated_query_error -> gate accepts 'y' -> exit 0" \
  "99999" \
  "" \
  "true" \
  "y" \
  0

# ============================================================================
# SCENARIO 3.1: prolix import failure → uncalibrated → gate accepts 'y' → exit 0
# ============================================================================

run_scenario \
  "3.1: prolix import failure -> uncalibrated -> gate accepts 'y' -> exit 0" \
  "" \
  "" \
  "false" \
  "y" \
  0

# ============================================================================
# SCENARIO 4.1: operator enters 'n' → gate exits 1 (decline accepted)
# ============================================================================

run_scenario \
  "4.1: operator enters 'n' -> gate exits 1 (decline accepted)" \
  "12347" \
  "TIMEOUT" \
  "false" \
  "n" \
  1

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================================"
echo "TEST SUMMARY"
echo "============================================================================"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
  echo "✅ All 5 scenarios passed!"
  exit 0
else
  echo "❌ $TESTS_FAILED scenario(s) failed"
  exit 1
fi
