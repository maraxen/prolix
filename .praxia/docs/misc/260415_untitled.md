# Daily log — 2026-04-13

## Done today

- (template) Espaloma validation, doc scope reconciliation, proxide golden/CI, fused GB alignment, gap backlog tracking.

## In progress

- See repository commits / PR for this session.

## Blockers

- None recorded.

## Next session

- Optional: run proxide `pytest -m espaloma` in conda env with `espaloma_charge` + DGL.

## Validation / Espaloma

- Tiers: API/import (T0), physics sanity (T1), golden regression (T2) under `proxide/tests/data/espaloma_golden/`.
- Prolix remains charge-array-only; no `espaloma_charge` in default CI.

## Gap backlog pointers

- `.agent/docs/plans/explicit_solvent_gaps_and_espaloma_charges.md`
- `docs/source/explicit_solvent/current_implementation.md`

---

### Session append-only

- Initialized daily log and execution checklist for Espaloma validation + remaining gaps plan.
- Reconciled implicit GB NL wording in `docs/source/explicit_solvent/current_implementation.md` (Scope vs limitations).
- Proxide: golden `tests/data/espaloma_golden/n2_charges.npy`, `pytest` marker `espaloma`, optional CI job `test-espaloma`, README section; `partial_charges` conformer protocol note.
- Prolix: restored `compute_pairwise_nl` + precomputed LJ in `fused_energy.py`; GB NL forwards `scaled_radii` and OpenMM-style masks when `dense_excl_scale_vdw` set; added `tests/test_fused_gb_alignment.py`.
- Added `.agent/docs/plans/remaining_gap_backlog.md` for P2–P6 + RESPA deferral.
