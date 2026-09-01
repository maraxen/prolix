---
title: use_size_buckets defaults impose a permanent NL runtime tax with no compile-sharing benefit for single-system production
description: Size-bucketing (meant to share compiled programs across differently-shaped systems) inflates neighbor-list capacity K by ~2.1x based on bucket padding, imposing a permanent ~2.7x per-step runtime tax on single-protein production trajectories that have no other shape to share a compile with.
status: draft
task_id: 260715_b1_physics_parity
date: '260901'
confidence: ''
sources: ''
---
# use_size_buckets defaults impose a permanent NL runtime tax with no compile-sharing benefit for single-system production

## Summary

`solvate_protein_to_bundle(..., use_size_buckets: bool = True)` defaults ON
regardless of whether the caller is actually going to batch multiple
differently-shaped systems together. For the neighbor-list (NL) nonbonded
path specifically, this default imposes a large, permanent per-step runtime
tax with **zero offsetting benefit** in the common case of a single protein
run for a single long production trajectory — because there is no other
shape to share a compiled program with.

## Mechanism (already correctly designed for its intended case)

Per debt 760/772 (`src/prolix/physics/neighbor_list.py`, `_pad_positions_with_ghost_lattice`
in `src/prolix/physics/system.py`), NL capacity `K` for a shape-class is
computed **statically from the bucket's padded atom count**, not the real
system's atom count:

```
K = ceil_to(round_to, safety_factor * rho_padded * (4/3) * pi * (cutoff + dr_threshold)^3)
rho_padded = atom_bucket_size / box_volume
```

This is deliberate (decision D1 in `.praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md`):
a capacity computed per-bundle from real occupancy would fragment
cross-protein compile-sharing, which is the core value proposition of this
project's heterogeneous-batching architecture. The earlier bug where ghost
(padding) atoms clustered at the coordinate origin — causing silent
neighbor-list corruption, not just a performance cost — was already fixed
(debt 772, uniform ghost lattice).

## Measured cost (full-md-path-matrix campaign, L40S, 1VII, n_real=1963)

| config | n_padded | nl_k_max | per_step_corrected_s | fit r² |
|---|---|---|---|---|
| NL, buckets ON | 5000 | 895 | 0.1081s | 0.924 |
| NL, buckets OFF | 1963 (exact) | 428 | 0.0398s | 0.578 (noisy) |
| dense, buckets ON | 5000 | n/a | 0.0507s | 0.304 (noisy) |
| flash, buckets ON | 5000 | n/a | 0.0725s | 0.991 |

Bucketed NL (K=895) is ~2.7x slower per step than unbucketed NL (K=428) at
the exact same real system, and — notably — bucketed NL is *slower than
bucketed dense* (108ms vs 51ms), inverting the entire point of using a
neighbor list. Unbucketed NL is the fastest of all four measured
configurations. (Caveat: dense's and unbucketed-NL's regression fits are
noisy — r² 0.30 and 0.58 respectively, only 3 points in the n_steps sweep —
so the exact per-step numbers for those two rows should not be treated as
final; the *qualitative* ordering and the bucketed-vs-unbucketed NL gap are
the load-bearing parts of this finding, not the last decimal place.)

## Why this matters

`use_size_buckets` is decided once, at bundle-construction time
(`solvate_protein_to_bundle`), with no visibility into whether the resulting
bundle will ever actually be co-batched with a differently-shaped bundle.
`EnsemblePlan.from_bundle`/`.run()` take no `use_size_buckets` parameter at
all — they simply inherit whatever the bundle was built with. The existing
harness script `scripts/experiments/prof_ensemble_step.py` even defaults its
own `--use-size-buckets` CLI flag to `"on"` for exactly this single-system
benchmark scenario, which is the case where bucketing offers no benefit.

For "produce frames for one protein, one long NVT/NPT trajectory" — the
common production case — bucketing should almost certainly be off: there is
no second shape to amortize a compile against, so the entire NL capacity
inflation (2.7x) is pure loss, paid on every step, for the life of the
trajectory. Bucketing is correctly the default only when the caller is
about to batch multiple *different* real proteins in one `EnsemblePlan`
(the heterogeneous-batching thesis case), where a shared compile is a real,
measured win (see `project_b1_heterogeneous_batching_confirmed_260717`
memory / debt 760/802).

## Recommended fix (not implemented — flagged for a deliberate decision)

`solvate_protein_to_bundle` (or a higher-level constructor) should not
default `use_size_buckets=True` unconditionally. Options, in rough order of
invasiveness:

1. Default `use_size_buckets=False` at `solvate_protein_to_bundle`, and have
   whatever code path actually constructs a multi-bundle ensemble
   (stacked-dispatch / `EnsemblePlan` multi-bundle construction) opt into
   bucketing explicitly, since *that* code path is the only place that
   actually knows multiple shapes are being combined.
2. Add a single-bundle vs. multi-bundle heuristic at the `EnsemblePlan`
   construction layer that overrides bucketing off when only one bundle is
   present, regardless of what the bundle was built with.
3. At minimum: document this trade-off explicitly wherever
   `solvate_protein_to_bundle`/`use_size_buckets` is discussed, so callers
   running a single production trajectory know to pass
   `use_size_buckets=False`.

This was explicitly *not* implemented this session (user directive,
2026-09-01): document only, no code change, since it touches the core
heterogeneous-batching default behavior and deserves a deliberate decision
rather than a quick default flip.

## Update 2026-09-01: a much larger confound was found and fixed first

The isolated-force-call numbers used to motivate this doc (dense/flash/NL on
H200 at 1VII) were originally measured with NL's `custom_vjp` backward pass
missing (see backlog #4623) — NL was paying for generic autodiff through a
gather-heavy forward pass on top of whatever bucketing tax it also paid.
After restoring the analytic backward pass, NL forces dropped from 19.317ms
to 2.321ms (8.3x) at the same bucketed config (n_padded=5000, K=895) — most
of the earlier gap was the missing custom_vjp, not bucketing. Bucketing's
own contribution is smaller in absolute terms than it first appeared, but
the underlying argument in this doc is unchanged: K=895 (bucketed) vs. K=428
(unbucketed) is still a real, deliberate inflation with no benefit for a
single-system production trajectory, and remains worth fixing on its own
terms. See [[project_debt4623_nl_custom_vjp_restored_260901]] and
[[project_debt761_4793_resolved_260901]] for the full picture.

## Related

- `.praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md` (decision D1, NL capacity formula)
- `src/prolix/physics/neighbor_list.py` (capacity formula, debt 760)
- `src/prolix/physics/system.py::_pad_positions_with_ghost_lattice` (debt 772)
- `scripts/experiments/prof_ensemble_step.py` (source of the measured numbers above)
- Backlog #4793 (debt-761 flash-vs-dense reconciliation) — this finding surfaced while investigating why isolated microbenchmark numbers didn't match production per-step costs.
- Separately found and fixed while investigating this: the shared uv workspace venv on Engaging (`/orcd/.../projects/.venv`, workspace root `projects-workspace`) had no CUDA-enabled jaxlib, silently falling back every `uv run`-based GPU job to CPU (fixed 2026-09-01 via `uv sync --extra cuda --extra dev --package prolix` from the workspace root). Any prior "GPU" cluster measurement that went through a bare `uv run python` without an explicit venv pin should be treated as unverified until re-checked against a run that confirms `jax.default_backend() == "gpu"` and a device name.

