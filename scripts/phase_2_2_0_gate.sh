#!/bin/bash
# phase_2_2_0_gate.sh — Operator acknowledgement gate for measurement source
#
# Read Measurement Source from ~/.praxia/daily_log.
# If SOURCE == 'prolix': proceed (exit 0).
# If SOURCE matches uncalibrated*: prompt operator for explicit 'y' ack.
#   Exit 0 if operator acks ('y')
#   Exit 1 if operator declines ('n') or SOURCE unknown
#
# Exit: 0 (prolix or operator acked) or 1 (operator declined or unknown SOURCE)

set -euo pipefail

# ============================================================================
# 1. READ DAILY_LOG AND EXTRACT SOURCE
# ============================================================================

DAILY_LOG="${HOME}/.praxia/daily_log"
SOURCE='uncalibrated'

if [ -f "$DAILY_LOG" ]; then
  # Extract Measurement Source from the last entry
  SOURCE=$(grep "Measurement Source:" "$DAILY_LOG" | tail -1 | awk '{print $NF}')
  if [ -z "$SOURCE" ]; then
    SOURCE='uncalibrated'
  fi
fi

# ============================================================================
# 2. PROLIX SUCCESS PATH
# ============================================================================

if [ "$SOURCE" = 'prolix' ]; then
  echo "✅ Prolix calibration confirmed. Proceeding with measured walltime."
  exit 0
fi

# ============================================================================
# 3. UNCALIBRATED SOURCES GATE
# ============================================================================

case $SOURCE in
  uncalibrated)
    echo "⚠️  WARNING: Uncalibrated measurement source"
    echo "Reason: Prolix physics library not importable on this system."
    echo "Impact: Conservative 24h walltime will be applied."
    echo ""
    echo "Do you acknowledge this and wish to proceed? (type 'y' to continue, 'n' to abort)"
    read -r ACK
    [ "$ACK" = "y" ] && exit 0 || exit 1
    ;;
  uncalibrated_timeout)
    echo "⚠️  WARNING: Uncalibrated measurement source"
    echo "Reason: Previous measurement job exceeded 12:00 walltime (SLURM killed it)."
    echo "Impact: Elapsed time was truncated; conservative 24h walltime will be applied."
    echo ""
    echo "Do you acknowledge this and wish to proceed? (type 'y' to continue, 'n' to abort)"
    read -r ACK
    [ "$ACK" = "y" ] && exit 0 || exit 1
    ;;
  uncalibrated_measurement_error)
    echo "⚠️  WARNING: Uncalibrated measurement source"
    echo "Reason: Measurement job ran but elapsed time was not captured (file I/O error)."
    echo "Impact: Conservative 24h walltime will be applied."
    echo ""
    echo "Do you acknowledge this and wish to proceed? (type 'y' to continue, 'n' to abort)"
    read -r ACK
    [ "$ACK" = "y" ] && exit 0 || exit 1
    ;;
  uncalibrated_query_error)
    echo "⚠️  WARNING: Uncalibrated measurement source"
    echo "Reason: SLURM job status query (sacct) failed (race condition or SLURM disconnect)."
    echo "Impact: Conservative 24h walltime will be applied."
    echo ""
    echo "Do you acknowledge this and wish to proceed? (type 'y' to continue, 'n' to abort)"
    read -r ACK
    [ "$ACK" = "y" ] && exit 0 || exit 1
    ;;
  *)
    echo "ERROR: Unknown SOURCE value: $SOURCE"
    exit 1
    ;;
esac
