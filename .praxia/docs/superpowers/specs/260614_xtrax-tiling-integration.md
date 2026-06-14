---
title: xtrax.tiling Integration for prolix
task_id: 260614_sprint37_paper_preground
date: 260614
status: draft
brainstorm_session: false
---

# xtrax.tiling Integration Strategy for prolix

## Executive Summary

prolix currently has its own `BatchPlanner` at `src/prolix/tiling/planner.py` (physics-specific, memory-budget-focused).
xtrax provides a generic `BatchPlanner` at `src/xtrax/tiling/plan.py` (strategy-polymorphic, extensible).
Other projects (aminx, plegadx, denxity) already standardize on xtrax.tiling.

**Recommendation: Wrapper strategy** — create a thin prolix adapter around xtrax.BatchPlanner to preserve physics-domain logic while adopting xtrax's strategy dispatch and AxisSpec metadata layer.

---

## 1. Gap Analysis: prolix vs xtrax BatchPlanner

### prolix.tiling.planner (current)

**Strengths:**
- Memory-budget-driven: `estimate_memory_theoretical()` computes peak memory per decision set
- Greedy budget loop: phases out vmap axes innermost-first when budget exceeded
- Granularity support: `tile_granularity` rounds tile sizes to multiples
- Integration: axes indexed and sorted; physics-aware naming

**Semantic model:**
- `AxisSpec`: name, axis_index, cardinality, default_batch_size (0=vmap, >0=safe_map), tile_granularity, heterogeneous, doc
- `BatchPlan`: decisions list + total_memory_estimate + axes_by_index + budget_exceeded flag
- `BatchPlanner`: takes axes + budget_bytes + memory_estimator function; calls `plan()` to return BatchPlan

**Physics coupling:**
- Default batch size named `default_batch_size` (prolix specific)
- No explicit strategy objects; just vmap (batch_size=0) vs safe_map (batch_size>0)
- No bucket_boundaries or dedup support
- Memory budget is the primary lever for vmap/safe_map decisions

### xtrax.tiling.plan (target)

**Strengths:**
- Strategy polymorphism: Vmap, SafeMap, Scan, DedupGather, Bucket explicitly modeled
- Flexible decision logic: phases for pre-demotion (Carry, Dedup), standard rules, memory override
- Bucket strategy: explicit length-padding for variable-length axes
- Carry/Dedup declarations: external contracts via CarrySpec/DedupSpec (enable scanning/deduplication)
- Extensible reasoning: human-readable decision strings

**Semantic model:**
- `AxisSpec`: name, cardinality, batch_size (default threshold), granularity, heterogeneous, dedup_eligible, bucket_boundaries
- `AxisDecision`: spec + batch_size + reasoning + strategy (Vmap | SafeMap | Scan | DedupGather | Bucket)
- `BatchPlan`: decisions tuple
- `BatchPlanner`: takes memory_estimator (optional), carry_specs, dedup_specs; calls `plan(specs)` to return BatchPlan

**Generic by design:**
- No budget loop; rules are static priority-ordered
- Strategy objects are responsible for their own dispatch (via `make_axis_dispatch()` elsewhere)
- Memory estimate overrides Vmap→SafeMap decision if provided
- Carry/Dedup phases allow upstream layer to declare stateful axes

### Key Differences

| Aspect | prolix | xtrax | Gap |
|--------|--------|-------|-----|
| **Decision output** | batch_size ∈ {0, >0} | strategy object (Vmap/SafeMap/Scan/...) | xtrax is more structured; prolix conflates size with strategy |
| **Budget handling** | Greedy loop in planner | Memory estimator overrides rules; no budget loop | xtrax delegates budget logic to memory_estimator callback |
| **Carry/Dedup** | Not supported | CarrySpec/DedupSpec pre-demotion phases | prolix has no stateful/dedup axis support |
| **Bucket strategy** | TODO comment only | Full implementation (length-padding) | prolix needs bucketing for variable-length data |
| **Granularity** | `tile_granularity` field in AxisSpec | `granularity` field in AxisSpec | Same concept, xtrax calls it `granularity` |
| **Memory estimate** | Callable takes decisions list; only output is estimate | Callable takes AxisSpec; more modular | prolix couples estimate to decision set; xtrax couples to axis |

---

## 2. Integration Strategy: Wrapper + Adapter

### Why Not Full Replacement?

Full replacement (`prolix.tiling.planner` → `from xtrax.tiling.plan import *`) would break prolix physics domain abstractions:

1. **Memory budget loop**: prolix's greedy demotion (innermost-first, until budget satisfied) is not in xtrax. xtrax assumes upstream sets a memory_estimator callback that makes per-axis decisions, not a global budget.
2. **estimate_memory_theoretical signature**: prolix expects `estimate_memory_theoretical(decisions, base_shape_bytes, activation_multiplier)`. xtrax has no public estimate function; it expects user to supply the callback.
3. **axes_by_index dictionary**: prolix physics code queries axes by index. xtrax decisions don't carry this.
4. **budget_exceeded flag**: prolix has explicit budget tracking. xtrax relies on the estimator to veto decisions.

### Solution: Wrapper Architecture

**Create `src/prolix/tiling/xtrax_adapter.py`:**
- Translate prolix.AxisSpec → xtrax.AxisSpec
- Wrap xtrax.BatchPlanner with prolix's greedy budget loop (if needed)
- Preserve `estimate_memory_theoretical` as helper
- Export prolix.AxisSpec, AxisDecision, BatchPlan, BatchPlanner as before

**Key invariant:** Public API of prolix.tiling.planner remains unchanged. Internal implementation delegates to xtrax.

**Backward compatibility:** Existing callers (axes.py, run/spec.py, tiling/__init__.py) import the same names from the same module; no refactoring needed.

### Adapter Design

```python
# src/prolix/tiling/xtrax_adapter.py

from xtrax.tiling.plan import (
    AxisSpec as XtraxAxisSpec,
    AxisDecision as XtraxAxisDecision,
    BatchPlan as XtraxBatchPlan,
    BatchPlanner as XtraxBatchPlanner,
)

# Re-export prolix types
from prolix.tiling.planner import (
    AxisSpec,
    AxisDecision,
    BatchPlan,
    BatchPlanner,
    estimate_memory_theoretical,
)
```

**Then refactor `src/prolix/tiling/planner.py`:**

1. Keep public API (AxisSpec, AxisDecision, BatchPlan, BatchPlanner, estimate_memory_theoretical)
2. Implement BatchPlanner.plan() as:
   - Convert prolix.AxisSpec list → xtrax.AxisSpec list
   - Call xtrax.BatchPlanner.plan()
   - Convert xtrax.BatchPlan decisions → prolix AxisDecision (strategy object → batch_size)
   - Run greedy budget loop if needed (optional, based on estimator presence)
   - Return prolix BatchPlan

**Type conversion map:**

```python
def prolix_to_xtrax_axis_spec(spec: AxisSpec) -> XtraxAxisSpec:
    """Convert prolix AxisSpec to xtrax AxisSpec."""
    return XtraxAxisSpec(
        name=spec.name,
        cardinality=spec.cardinality,
        batch_size=spec.default_batch_size,
        granularity=spec.tile_granularity,
        heterogeneous=spec.heterogeneous,
        dedup_eligible=False,  # prolix doesn't declare dedup
        bucket_boundaries=None,  # prolix doesn't use buckets yet
    )

def xtrax_decision_to_prolix_batch_size(xtrax_decision: XtraxAxisDecision) -> int:
    """Extract batch_size from xtrax strategy."""
    strategy = xtrax_decision.strategy
    if isinstance(strategy, Vmap):
        return 0  # prolix convention: 0 = vmap
    elif isinstance(strategy, SafeMap):
        return strategy.batch_size
    elif isinstance(strategy, (Scan, DedupGather, Bucket)):
        # If xtrax emits these, prolix ignores them for now
        # (will be used in future phases)
        return strategy.batch_size if hasattr(strategy, 'batch_size') else 1
```

---

## 3. Dependency Decision: Path Dependency

### Options

**A. Path dependency** (`uv add --path ../xtrax xtrax`):
- Pros: Immediate access to xtrax changes; projects are siblings
- Cons: Hard coupling to xtrax checkout location
- Workflow: `uv lock --upgrade` pulls xtrax changes into prolix.lock

**B. Git URL** (`uv add --git https://github.com/maraxen/xtrax`):
- Pros: Decoupled; works in CI
- Cons: Slower; requires network
- Best for: Released versions

**C. PyPI** (future):
- Pros: Stable releases
- Cons: Need to release xtrax first
- Timeline: Out of scope for this sprint

### Recommendation: **Path dependency**

**Rationale:**
- Both projects developed locally in `/home/marielle/projects/`
- Sprint 36–37 are active development; path dep simplifies iteration
- Once xtrax v0.1 is released on PyPI, switch to stable version

**Implementation:**
```bash
cd /home/marielle/projects/prolix
uv add --path ../xtrax xtrax
```

This adds to `pyproject.toml`:
```toml
dependencies = [
    ...
    "xtrax @ file://../xtrax",  # or similar syntax
]
```

---

## 4. Migration Plan: Files and Imports

### Step 1: Add dependency to pyproject.toml
- Run `uv add --path ../xtrax xtrax`
- Verify `uv.lock` updates with xtrax entries

### Step 2: Refactor src/prolix/tiling/planner.py
- Import xtrax types at top
- Keep prolix AxisSpec, AxisDecision, BatchPlan, BatchPlanner class names
- Implement BatchPlanner.plan() to delegate to xtrax.BatchPlanner
- Keep estimate_memory_theoretical() helper
- Export from __init__.py unchanged

### Step 3: Update src/prolix/tiling/__init__.py
- No code changes needed (re-exports unchanged)

### Step 4: Verify callers

**src/prolix/tiling/axes.py:**
- `from prolix.tiling.planner import AxisSpec` — unchanged

**src/prolix/run/spec.py:**
- `from prolix.tiling.planner import BatchPlan, BatchPlanner, estimate_memory_theoretical` — unchanged

### Step 5: Test and commit
- Run `uv run pytest -m "not slow" -q`
- Verify tiling tests pass
- Commit with message: `refactor(tiling): adopt xtrax.BatchPlanner via adapter`

---

## 5. Backward Compatibility

### Public API (prolix.tiling.planner)
- `AxisSpec(name, axis_index, cardinality, default_batch_size, tile_granularity, heterogeneous, doc)` → **unchanged signature**
- `AxisDecision(axis, batch_size, reasoning)` → **unchanged signature**
- `BatchPlan(decisions, total_memory_estimate, axes_by_index, budget_exceeded)` → **unchanged signature**
- `BatchPlanner(axes, budget_bytes, estimate_memory)` → **unchanged signature**
- `BatchPlanner.plan()` → **unchanged method signature**
- `estimate_memory_theoretical(decisions, base_shape_bytes, activation_multiplier)` → **unchanged signature**

### Tests
- Existing tests in `tests/tiling/` should continue to pass without modification
- No test rewrites required if public API is stable

### Re-export Chain
```
prolix/__init__.py
  ↓
prolix/tiling/__init__.py
  → from prolix.tiling.planner import AxisSpec, AxisDecision, BatchPlan, BatchPlanner
  ↓
prolix/tiling/planner.py (adapter)
  → from xtrax.tiling.plan import ...
  ← reimplements prolix classes as thin wrappers
```

---

## 6. Future Extensions (Out of Scope)

Once wrapper is working, prolix can optionally adopt xtrax features:

1. **Carry/Scan axes**: For stateful batch dimensions (trajectories, sequence dynamics)
   - Declare via CarrySpec; xtrax.BatchPlanner pre-demotes to Scan
   - prolix ignores Scan strategy for v1.0, becomes available in v1.1

2. **Bucket strategy**: For variable-length data (sequences, structures)
   - Set AxisSpec.bucket_boundaries; xtrax selects Bucket strategy
   - Requires prolix work: bucketing kernel, padding logic

3. **Dedup/DedupGather**: For repeated element patterns
   - Declare via DedupSpec; xtrax.BatchPlanner pre-demotes to DedupGather
   - Enables efficient batching of heterogeneous-shaped molecules

4. **Memory estimator callback**: Replace greedy loop with per-axis estimation
   - Callback signature: `(spec: AxisSpec) -> int` (bytes)
   - xtrax uses this to override Vmap→SafeMap if over budget

---

## Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Replacement vs Wrapper** | Wrapper | Preserves prolix budget loop and estimate_memory_theoretical semantics |
| **Dependency type** | Path (`--path ../xtrax`) | Local development; ease of iteration |
| **New module?** | No — refactor planner.py in-place | Minimal API churn; backward compatible |
| **Test rewrites** | None (backward compatible) | Public signatures unchanged |
| **xtrax strategies adopted** | None in v1.0 | Bucket/Carry/Dedup deferred to v1.1 phases |

---

## Implementation Checklist

- [ ] Add xtrax to pyproject.toml (path dependency)
- [ ] Update uv.lock
- [ ] Refactor src/prolix/tiling/planner.py with xtrax delegation
- [ ] Run tiling tests (all pass)
- [ ] Run full pytest suite (all pass)
- [ ] Commit with clear message
- [ ] Verify run/spec.py and axes.py work unchanged
- [ ] Update this spec with final implementation notes

