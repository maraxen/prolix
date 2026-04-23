# Phase 1 Diagnostics Index

**Created**: 2026-04-23  
**Status**: All deliverables complete, ready for execution

## Quick Links

### For Getting Started
1. **`PHASE1_EXECUTION_GUIDE.md`** - Start here! 3 ways to run Phase 1
2. **`PHASE1_DELIVERABLES_SUMMARY.md`** - Complete overview of Phase 1

### For Understanding Results
3. **`phase1_results/README.md`** - How to interpret outputs
4. **`phase1_results/PHASE1_VERDICT.template.md`** - Example output structure

### For Running Diagnostics
5. **`tests/physics/test_phase1_ablation.py`** - Pytest test suite (400 lines)
6. **`scripts/phase1_diagnostic_runner.py`** - Full runner with ensemble parallelization (500 lines)
7. **`scripts/phase1_plotting_and_verdict.py`** - Analysis + hypothesis scoring (300 lines)

## 30-Second Summary

**Problem**: Langevin thermostat showing ΔT = 28.8 K @ dt=1.0 fs (should be < 5 K)

**Solution**: Phase 1 ablates across 3 hypotheses:
1. H1 (HIGH): RATTLE saturation at n_iters=10
2. H2 (MEDIUM): Float32 precision loss in inertia tensor
3. H3 (MEDIUM): Projection timing bias

**Timeline**:
- 30-45 min: Run diagnostics
- 5 min: Generate plots + verdict
- Decision: Which hypothesis to fix in Phase 2

## Execution in 3 Steps

```bash
# Step 1: Run diagnostics (30 min - 3 hours)
python3 scripts/phase1_diagnostic_runner.py --n-replicas 5 --output ./phase1_results

# Step 2: Generate verdict (5 min)
python3 scripts/phase1_plotting_and_verdict.py --input ./phase1_results --output ./phase1_results

# Step 3: Read verdict
cat phase1_results/PHASE1_VERDICT.md
```

## Key Files

```
prolix/
├── PHASE1_INDEX.md (this file)
├── PHASE1_EXECUTION_GUIDE.md ← START HERE
├── PHASE1_DELIVERABLES_SUMMARY.md
├── phase1_results/
│   ├── README.md
│   └── PHASE1_VERDICT.template.md
├── tests/physics/
│   └── test_phase1_ablation.py (3 test classes)
└── scripts/
    ├── phase1_diagnostic_runner.py
    └── phase1_plotting_and_verdict.py
```

## Outputs After Execution

```
phase1_results/
├── summary.json (master data)
├── PHASE1_VERDICT.md (hypothesis scores - READ THIS)
├── residual_convergence.png
├── ke_precision_comparison.png
├── temperature_by_projection_site.png
├── axis1_iterations/*.csv
├── axis2_precision/*.csv
└── axis3_projection/*.csv
```

## What Each Hypothesis Explains

| Hypothesis | If True | Evidence | Phase 2 Fix |
|-----------|---------|----------|------------|
| H1: RATTLE saturation | Temperature improves 28→5 K with n_iters=50 | Residual norm decreases exponentially | Increase settle_velocity_iters |
| H2: Float32 precision | Float32 T >> Float64 T, cond_number > 1e8 | Divergence visible by step ~50 | Force float64 in projection |
| H3: Projection timing | 'both' better than 'post_o' by > 5 K | projection_site='both' reduces ΔT | Switch projection_site default |

## Success Criteria

✓ All hypothesis scores computed  
✓ Each score either > 0.7 (confirmed) or < 0.4 (rejected)  
✓ Go/No-Go decision is clear and actionable  
✓ Phase 2 fix is specific (which code line to change)

## Estimated Duration

- **Phase 1 Setup**: ~4 hours (done ✓)
- **Phase 1 Diagnostics**: 30 min - 3 hours (depends on n_replicas)
- **Phase 1 Analysis**: 5 minutes
- **Total Phase 1**: ~3-4 hours including setup
- **Phase 2 Implementation**: 1-2 hours (based on verdict)
- **Total Timeline**: 1 week to completion

---

Start with: **`PHASE1_EXECUTION_GUIDE.md`**
