---
title: 'named_scope attribution: known limitations at production scale'
description: Why some prolix named_scope labels never resolve under XLA fusion, preserved from scripts/profiling/trace.py before the xtrax.profiling migration
status: draft
task_id: 260901_kernel_bottleneck_sprint
date: '260901'
---
# named_scope attribution: known limitations at production scale

Preserved from `scripts/profiling/trace.py` when that package was retired in
favour of the published `xtrax.profiling` (xtrax 0.4.0a7, 2026-09-01). Upstream
carries the parsing code and the general two-input attribution design, but not
this prolix-specific evidence about where the technique stops working. It is
recorded here so the migration is not lossy.

## How attribution actually works (two-input, not trace-only)

Attribution joins **two** sources, which is why it is fragile:

* the **executed** Perfetto trace supplies exclusive durations and occurrence
  counts per thunk, keyed by each event's `args["hlo_op"]` — a post-fusion
  thunk name like `wrapped_multiply` or `reduce_add_fusion`, which by itself
  names no `named_scope` at all;
* the **compiled** HLO text supplies the thunk-name → scope-path mapping, because
  the `named_scope` path survives only as a `/`-delimited prefix on
  per-instruction `op_name` metadata
  (e.g. `metadata={op_name="jit(f)/outer_scope/inner_a/..."}`, obtainable via
  `jax.jit(fn).lower(*args).compile().as_text()`).

Each instruction's time is attributed to the **deepest** known label appearing in
its `op_name` path, following `calls=`/`to_apply=` references down through nested
computations when the top-level instruction carries no metadata of its own. A
resolved label is therefore always exactly one string per instruction — there is
no mixed-scope ambiguity to arbitrate.

Restricting the scan to the ENTRY computation resolves **zero** labels on a real
multi-step program: every `settle_*`-wrapped op lives inside the while-body
computation, not ENTRY.

## Known limitations (#4330 Fix 2b)

Some scopes never resolve, and adding more `named_scope` calls does not help.

1. **Scopes wrapping `scan(..., unroll=N)` at the outer level may be absorbed
   into fusion.** Observed on `flash_nonbonded_tiles`
   (`src/prolix/physics/flash_explicit.py`, the tiled nonbonded scan).
2. **Scopes inside conditional branches backed by control flow resolve
   inconsistently between runs.** Observed with `pme_bwd_gather`, present in some
   production traces and absent from others. Control flow with conditional
   execution interacts with scope-path rewriting unpredictably under the XLA
   backend's optimization passes.

**Evidence.** A compiled HLO dump from a production-scale MD force computation
shows these scopes simply absent from the `op_name` metadata, even though the
underlying computation is verified correct and executes. No simple substitution
("replace X wrapper with Y") recovers the labels — the cause is structural fusion
or control-flow interaction at the backend level, not a parser defect.

Measured consequence at DHFR scale (2026-09-01): an instrumented neighbor-list
trace recovered **1 of 26** labels, accounting for 0.737 µs of a 2323 µs step —
**0.03% of runtime attributed**. Three of four scopes added during that session
vanished the same way. xtrax's own GPU dogfood hit the identical wall (0 of 2)
and ships it as a documented open item.

## The working fallback

Do not try to fix this by adding scopes. Two mechanisms actually recover a
ranking:

* **`parse_hlo_op_times`** (`xtrax.profiling.trace`) ranks by post-fusion thunk
  name rather than by scope, so it reports *something* even when scope recovery is
  negligible. It is already wired into `scripts/explore/prof_ensemble_apply_probe.py`'s
  `_capture_trace`, which also warns when the top thunk is a `command_buffer_*`
  entry — that means the ranking is not the production program.
* **`XLA_FLAGS=--xla_gpu_enable_command_buffer=`** (empty) explodes the CUDA-graph
  command buffer into separately-timed thunks, which is what makes a per-component
  breakdown possible at all on GPU. This is a **diagnostic compile, not a
  production config**: never mix its timings with a normally-compiled run.
  See `scripts/slurm/prof_ensemble_apply_cmdbuf_off.slurm`.

## Related

* Backlog #4330 (attribution survival), #4326 (analysis/report step).
* `.praxia/docs/specs/260820_jax-profiling-skill-DRAFT.md` §command-buffer.
* `.praxia/docs/specs/260817_jax-profiling-optimization-workflow.md` §P4 (the
  fallback clause this prefix-depth rule implements).
