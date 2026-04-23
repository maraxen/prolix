#!/bin/bash
#
# Phase 1 Langevin Thermostat Diagnostic Runner
# Usage: ./run_phase1_diagnostics.sh [--n-replicas N] [--quick] [--output DIR]
#
# Quick mode (--quick): Use n_replicas=3 for fast validation (~15 min)
# Default mode: Use n_replicas=10 for rigorous statistics (~1-2 hours)
#

set -e

# === Configuration ===
N_REPLICAS=10
OUTPUT_DIR="./phase1_results"
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-replicas)
            N_REPLICAS="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            N_REPLICAS=3
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--n-replicas N] [--quick] [--output DIR]"
            exit 1
            ;;
    esac
done

# === Helper Functions ===
print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

print_step() {
    echo "▶ $1"
}

print_success() {
    echo "✓ $1"
}

print_error() {
    echo "✗ $1"
    exit 1
}

# === Main Script ===
print_header "PHASE 1: LANGEVIN THERMOSTAT ROOT-CAUSE ISOLATION"

# Show configuration
echo "Configuration:"
echo "  n_replicas: $N_REPLICAS"
echo "  output_dir: $OUTPUT_DIR"
if [ "$QUICK_MODE" = true ]; then
    echo "  mode: QUICK (fast validation)"
else
    echo "  mode: DEFAULT (rigorous statistics)"
fi
echo ""

# Step 1: Verify Python environment
print_header "Step 1: Verify Environment"
print_step "Checking uv and Python..."
if ! command -v uv &> /dev/null; then
    print_error "uv not found in PATH. Install with: pip install uv"
fi
print_success "uv found"

print_step "Checking Python version..."
uv run python --version
print_success "Python environment ready"
echo ""

# Step 2: Syntax check
print_header "Step 2: Syntax Validation"
print_step "Checking phase1_diagnostic_runner.py..."
uv run python -m py_compile scripts/phase1_diagnostic_runner.py && print_success "Syntax OK" || print_error "Syntax error in runner"

print_step "Checking test_phase1_ablation.py..."
uv run python -m py_compile tests/physics/test_phase1_ablation.py && print_success "Syntax OK" || print_error "Syntax error in tests"

print_step "Checking phase1_plotting_and_verdict.py..."
uv run python -m py_compile scripts/phase1_plotting_and_verdict.py && print_success "Syntax OK" || print_error "Syntax error in plotting"
echo ""

# Step 3: Smoke tests
print_header "Step 3: Unit Tests (Smoke Check)"
print_step "Running vmap ensemble test..."
uv run python -m pytest tests/physics/test_phase1_ablation.py::TestPhase1VmapEnsemble -v --tb=short || print_error "vmap test failed"
print_success "vmap ensemble test passed"

print_step "Running single-config test..."
uv run python -m pytest tests/physics/test_phase1_ablation.py::TestPhase1AblationRATTLEIterations::test_single_config -v --tb=short || print_error "single-config test failed"
print_success "single-config test passed"
echo ""

# Step 4: Create output directory
print_header "Step 4: Prepare Output"
print_step "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
print_success "Directory ready"
echo ""

# Step 5: Run full diagnostic
print_header "Step 5: Phase 1 Diagnostic Run"
print_step "Starting ensemble trajectories (n_replicas=$N_REPLICAS)..."
echo "  (This may take 15 min – 3 hours depending on n_replicas and hardware)"
echo ""

uv run python scripts/phase1_diagnostic_runner.py \
    --n-replicas "$N_REPLICAS" \
    --output "$OUTPUT_DIR" || print_error "Diagnostic runner failed"

print_success "Diagnostic trajectories completed"
echo ""

# Step 6: Generate verdict
print_header "Step 6: Analyze Results & Generate Verdict"
print_step "Computing hypothesis scores..."
uv run python scripts/phase1_plotting_and_verdict.py \
    --input "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR" || print_error "Verdict generation failed"

print_success "Verdict generated"
echo ""

# Step 7: Display results
print_header "Phase 1 Results"
VERDICT_FILE="$OUTPUT_DIR/PHASE1_VERDICT.md"
if [ -f "$VERDICT_FILE" ]; then
    print_success "Verdict saved to: $VERDICT_FILE"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    cat "$VERDICT_FILE"
    echo "════════════════════════════════════════════════════════════════════════════════"
else
    print_error "Verdict file not found: $VERDICT_FILE"
fi
echo ""

# Step 8: Summary
print_header "Phase 1 Complete"
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.{png,csv,json,md} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Next steps:"
echo "  1. Review PHASE1_VERDICT.md above for hypothesis scores"
echo "  2. If H1 CONFIRMED (score > 0.7): Proceed to Phase 2 (adaptive RATTLE)"
echo "  3. If H2 CONFIRMED: Investigate float64-only fallback"
echo "  4. If H3 CONFIRMED: Audit projection timing"
echo "  5. If ESCALATE: Return to Oracle for alternative hypotheses"
echo ""
print_success "Run complete!"
