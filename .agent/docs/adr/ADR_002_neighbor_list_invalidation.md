# ADR 002: Batched neighbor-list invalidation (incremental “patch” sketch)

## Status

Draft / deferred (2026-04-29). Informs a future Prolix analogue to kUPs `Potential` **patches** without adopting the full kUPs `Lens`/`Patch` stack.

## Problem

- Batched explicit-solvent dynamics use a padded neighbor index tensor `(B, N, K)` (see `build_batched_neighbor_list_jaxmd` in `src/prolix/batched_simulate.py`).
- `make_langevin_step_nl_dynamic` already updates a JAX-MD `NeighborList` after each step on GPU.
- A **subset** of replicas may invalidate their NL at different times when systems diffuse at different rates under heterogeneous masks.

## Direction (non-binding)

1. **Per-replica dirty flag** `(B,)` bool carried outside or inside a small side-car pytree (not necessarily `LangevinState`).
2. **Rebuild policy**: when `dirty[b]` or JAX-MD `nbrs.did_overflow` / safe displacement threshold — rebuild only row `b` on host or use batched `allocate` with masked gather (perf trade-off).
3. **Potential “patch” API (optional)**: pass a struct `{replica_id, moved_atom_slice}` into a future `PaddedPotential` wrapper; PME still recomputes global mesh energy so **true incremental PME** is out of scope until a separate electrostatics milestone.

## Out of scope (v1)

- Partial energy accumulation across replicas.
- kUPs-style `IndexLensPatch` composition inside Prolix core.
