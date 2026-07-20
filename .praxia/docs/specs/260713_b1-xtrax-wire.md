---
title: "B1-XTRAX-WIRE"
backlog_id: B1-XTRAX-WIRE
epic: 260528_b1-full
depends_on: [B1-INFER]
priority: P1
status: completed
challenge_verdict: accept
challenge_summary: "Same-bucket can_jit_vmap; host partition by shape_spec; DedupGather topology-keyed only; real masses on stacked path."
completed_2026_07_13: true
---

# B1-XTRAX-WIRE

## Goal
Make EnsemblePlan honor the xtrax decision tree and Claim-1 same-bucket substrate: no Python-over-systems inside a shape class; DedupGather only for topology-keyed bodies.

## Locked composition

1. Host `partition_bundles_by_shape` → K classes (Python over K only).
2. Per class: `can_jit_vmap_n_mols` = stack-compatible `shape_spec` (equal `n_atoms` **not** required); `integration_prefix` = atom bucket; real `masses_with_prefix`.
3. Vmap/SafeMap over replicas with distinct seeds.
4. DedupGather / `dispatch_n_mols_dedup` for topology-keyed bodies only — never seeded Langevin.

## Acceptance

- Same-bucket different `n_atoms` stacks and runs finite inference.
- Grouped run of 4 identical-shape replicas → one stacked dispatch, zero per-system singles.
- Distinct seeds ⇒ non-identical final frames.
- Masses on prefix path are not unit ones when bundle carries real masses.

## Specs / code

- [`src/prolix/api/bundle_stack.py`](../../src/prolix/api/bundle_stack.py)
- [`src/prolix/api/ensemble_plan.py`](../../src/prolix/api/ensemble_plan.py)
- [`src/prolix/api/ensemble_dedup.py`](../../src/prolix/api/ensemble_dedup.py)
- [`tests/api/test_b1_xtrax_wire.py`](../../tests/api/test_b1_xtrax_wire.py)
