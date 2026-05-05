#!/bin/bash
# extract_walltime.sh — Measure 1000-step NVT simulation walltime
# and extrapolate to 50k-step walltime. Determine measurement source
# (prolix vs uncalibrated). Append decision to ~/.praxia/daily_log.
#
# Exit: 0 (success, regardless of measurement source)

set -euo pipefail

# ============================================================================
# 1. IMPORT CHECK: prolix.physics availability
# ============================================================================

SOURCE='uncalibrated'
ELAPSED=0

if python3 -c "from prolix.physics import settle" 2>/dev/null; then
  SOURCE='prolix'
else
  # Prolix import failed; skip to decision rules with SOURCE='uncalibrated'
  SOURCE='uncalibrated'
fi

# ============================================================================
# 2. MEASUREMENT: 1000-step NVT (only if SOURCE='prolix')
# ============================================================================

if [ "$SOURCE" = 'prolix' ]; then
  # Run 1000-step NVT simulation
  START=$(date +%s%N)
  python3 scripts/measure_1000step_nvt.py 2>/dev/null || {
    SOURCE='uncalibrated'
    ELAPSED=0
  }
  END=$(date +%s%N)

  # Calculate elapsed time in seconds
  ELAPSED=$(( ($END - $START) / 1000000000 ))

  # Avoid division by zero in extrapolation
  if [ "$ELAPSED" -lt 1 ]; then
    ELAPSED=1
  fi
fi

# ============================================================================
# 3. SLURM STATUS CHECK
# ============================================================================

JOB_STATUS='N/A'

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Running under SLURM; query job status
  if JOB_STATUS=$(sacct -j "$SLURM_JOB_ID" --format=State --noheader 2>/dev/null | head -1); then
    JOB_STATUS="${JOB_STATUS// /}"  # trim whitespace

    case "$JOB_STATUS" in
      TIMEOUT|FAILED|OUT_OF_MEMORY|CANCELLED|PREEMPTED|NODE_FAIL)
        SOURCE='uncalibrated_timeout'
        ELAPSED=0
        ;;
      COMPLETED)
        # Keep SOURCE from step 1
        ;;
      *)
        # Unknown status; conservative fallback
        SOURCE='uncalibrated_query_error'
        ELAPSED=0
        ;;
    esac
  else
    # sacct query failed (race condition or SLURM disconnect)
    SOURCE='uncalibrated_query_error'
    ELAPSED=0
  fi
fi

# ============================================================================
# 4. DECISION RULES: Walltime selection
# ============================================================================

DEFAULT_WALLTIME=''
PARTITION_COMMENT=''
DECISION_RULE=''

case $SOURCE in
  prolix)
    # Extrapolate 1000-step to 50k-step
    EXTRAPOLATED_SECS=$((ELAPSED * 50))
    EXTRAPOLATED_MINS=$((EXTRAPOLATED_SECS / 60))
    EXTRAPOLATED_HOURS=$((EXTRAPOLATED_MINS / 60))

    if [ "$EXTRAPOLATED_SECS" -lt 360 ]; then
      # < 6 min
      DEFAULT_WALLTIME='06:00:00'
      PARTITION_COMMENT='(partition: mit_preemptable, cap 48h)'
    elif [ "$EXTRAPOLATED_SECS" -lt 720 ]; then
      # < 12 min
      DEFAULT_WALLTIME='12:00:00'
      PARTITION_COMMENT='(partition: mit_preemptable, cap 48h)'
    elif [ "$EXTRAPOLATED_SECS" -lt 1440 ]; then
      # < 24 min
      DEFAULT_WALLTIME='24:00:00'
      PARTITION_COMMENT='(partition: mit_preemptable, cap 48h)'
    else
      # >= 24 min (>= 48h extrapolated)
      DEFAULT_WALLTIME='ESCALATE'
      PARTITION_COMMENT='(exceeds mit_preemptable 48h; manual review required)'
    fi
    DECISION_RULE="Measured ${ELAPSED}s -> extrapolated ${EXTRAPOLATED_SECS}s (${EXTRAPOLATED_MINS} min) -> ${DEFAULT_WALLTIME} default for mit_preemptable"
    ;;
  uncalibrated*)
    # Conservative default (prolix not importable or measurement failed)
    DEFAULT_WALLTIME='24:00:00'
    PARTITION_COMMENT='(partition: mit_preemptable, cap 48h)'
    DECISION_RULE="Uncalibrated mode (SOURCE=${SOURCE}); conservative 24h default applied"
    ;;
esac

# ============================================================================
# 5. DAILY LOG APPEND
# ============================================================================

DAILY_LOG="${HOME}/.praxia/daily_log"
mkdir -p "$(dirname "$DAILY_LOG")"

{
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Job ID: ${SLURM_JOB_ID:-interactive}"
  echo "Job Status: ${JOB_STATUS}"
  echo "Measured Wall Time: ${ELAPSED}s"
  echo "Measurement Source: ${SOURCE}"
  echo "Decision Rule Branch: ${DECISION_RULE}"
  echo "Default Walltime Selected: ${DEFAULT_WALLTIME} ${PARTITION_COMMENT}"
  echo "---"
} >> "$DAILY_LOG"

# ============================================================================
# 6. EXIT
# ============================================================================

exit 0
