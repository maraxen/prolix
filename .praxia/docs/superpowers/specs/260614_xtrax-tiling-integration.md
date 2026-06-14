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

## 2. Integration Strategy: Type Imports Only (v1.0), Behavioral Delegation Deferred (v1.1)

### Decision: Imports + Own Budget Loop (v1.0)

For v1.0, prolix imports xtrax types into `pyproject.toml` and keeps its proven greedy budget loop. Behavioral delegation (calling xtrax.BatchPlanner to make strategy decisions) is deferred to v1.1 after xtrax matures and achieves stable API.

### Why Deferred?

1. **Proven local implementation**: prolix's greedy demotion (innermost-first, until budget satisfied) is battle-tested and working. Switching to xtrax's memory_estimator callback model mid-sprint introduces risk.
2. **Budget loop mismatch**: xtrax assumes upstream sets a memory_estimator callback that makes per-axis decisions. prolix's global budget loop is a separate concern. Marrying them is a v1.1 refactoring task.
3. **Zero behavioral change in v1.0**: Keeping prolix's logic means existing tests pass, callers are unaffected, and integration is transparent. This is the lowest-risk path.
4. **Foundation for v1.1+**: Importing xtrax establishes the dependency and allows v1.1 to adopt Bucket/Carry/Dedup strategies incrementally.

### Implementation (v1.0)

**`src/prolix/tiling/planner.py`:**
- Keep all prolix classes: AxisSpec, AxisDecision, BatchPlan, BatchPlanner
- Keep greedy budget loop in BatchPlanner.plan() (unchanged logic)
- Remove xtrax method calls and converter functions (dead code)
- Add import comment noting xtrax is available for v1.1 phases

**Key invariant:** Public API of prolix.tiling.planner is unchanged. No refactoring needed.

**Backward compatibility:** Existing callers (axes.py, run/spec.py, tiling/__init__.py) import the same names from the same module; all tests pass.

### v1.1 Roadmap: Behavioral Delegation

When ready, BatchPlanner.plan() will:
1. Convert prolix.AxisSpec list → xtrax.AxisSpec list
2. Call xtrax.BatchPlanner.plan() with optional memory_estimator callback
3. Convert xtrax.AxisDecision (strategy objects) → prolix batch_size integers
4. Run secondary greedy loop if budget still exceeded (fallback to prolix semantics)
5. Return prolix BatchPlan

This phased approach lets xtrax mature while prolix remains stable in v1.0.

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

### Step 2: Clean src/prolix/tiling/planner.py
- Remove xtrax imports and method calls (dead code in v1.0)
- Keep greedy budget loop as-is
- Add comment noting xtrax available for v1.1 delegation
- No changes to public API (AxisSpec, AxisDecision, BatchPlan, BatchPlanner, estimate_memory_theoretical)

### Step 3: Update src/prolix/tiling/__init__.py
- No code changes needed (re-exports unchanged)

### Step 4: Verify callers (no changes needed)

**src/prolix/tiling/axes.py:**
- `from prolix.tiling.planner import AxisSpec` — unchanged

**src/prolix/run/spec.py:**
- `from prolix.tiling.planner import BatchPlan, BatchPlanner, estimate_memory_theoretical` — unchanged

### Step 5: Test and commit
- Run `uv run pytest -m "not slow" -q`
- Verify tiling tests pass
- Commit with message: `spec(tiling): defer xtrax.BatchPlanner delegation to v1.1, keep v1.0 greedy loop`

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
| **v1.0 strategy** | Import only (no delegation) | Proven greedy loop; zero behavioral change; lowest risk |
| **v1.1 strategy** | Behavioral delegation to xtrax.BatchPlanner | After xtrax matures; enables Bucket/Carry/Dedup |
| **Dependency type** | Path (`--path ../xtrax`) in pyproject.toml | Establishes foundation for future phases |
| **Code changes** | Remove dead xtrax calls; keep greedy loop | Minimal diff; backward compatible API |
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

