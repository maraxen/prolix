---
title: Prolix core MD-path fix — make MolecularBundle run real proteins (B1 unblock)
task_id: 260706_xtrax_pathc
date: 260706
status: ready-to-execute
blocks: B1-full (Claim 1) — .praxia/docs/specs/260528_b1-preregistration.md
---

# Sprint A — Fix the prolix bundle MD path for real proteins

## Why (verified diagnosis)

Building the B1-full Claim-1 benchmark surfaced that prolix's `MolecularBundle`
MD path (the P1a bucketed-JIT boundary) has **never run a real vacuum-protein
trajectory** — it has been exercised only on water and on unit-mass synthetic
toys. Four verified findings, plus one pre-existing gate breakage:

1. **Dropped nonbonded exclusions (false-pass keystone).** `scripts/benchmarks/_b1_paramize.py`
   (commit `1ca3978`) built bundles **without** an `exclusion_spec`, so every
   1-2/1-3 bonded neighbor (~1.5 Å) is double-counted as a clashing LJ pair.
   Measured on 2GB1: energy **+2.02e7** kcal/mol, **median |grad| 3.84e5**,
   max 6.18e5 kcal/mol/Å. The keystone's "finite 2-step run = PASS" was a
   **false pass** — 2 steps was too short for float32 to overflow, and force
   *scale* was never checked (BATHOS measurement-verification miss).
2. **Exclusions can't flow through the bundle energy path under JIT.**
   `_dense_excl_matrices_from_bundle` (`src/prolix/api/bundle_md.py:166`) uses a
   **numpy loop over what become tracers** — it early-returns on empty exclusions,
   so the JIT'd `energy_fn` had never seen populated exclusions. Populating them
   raises `TracerArrayConversionError` (host path) — confirmed by direct run.
3. **`MolecularBundle` has no mass field.** `masses_for_bundle`
   (`bundle_md.py:99-101`) returns `jnp.ones()` for anything that isn't pure
   water, so **proteins run at unit mass** (C/N/O = 1.0 instead of 12–16).
   `make_bundle_from_system` receives `system.masses` but has nowhere to store it
   (no field on the eqx.Module — verified `src/prolix/types/bundles.py:101-141`).
4. **`settle_langevin` needs dt ≤ 0.01 fs for vacuum proteins** — 50× below the
   pinned 0.5 fs — even after exclusions + masses + minimization. Consistent with
   the rigid-water/SETTLE projection being mis-applied to a non-water system.
5. **Pre-existing (NOT introduced by this work):** `tests/api/test_v4_hlo_hetero_compile_once.py`
   + `tests/api/test_v5_observable_parity.py` fail with `ConcretizationTypeError`
   at `bundle_md.py:211` (an `int(traced int32[])` in the stacked compile-once
   path). Bisect confirmed these are **red at the session-start base `6b9e588`**,
   before Path B or any B1 work. They live in the exact code A2 must touch.

**Minimization is NOT the fix** (it was the initial hypothesis; the evaluation
disproved it — minimization plateaus at 2.2e5 kcal/mol on the broken energy).

## Key leverage — the fix is mostly REUSE, not invention

Prolix already has a working, JIT-safe exclusion path that the bundle route
simply fails to call:
- `ExclusionSpec.from_protein(protein)` — used in `src/prolix/simulate.py:489`
  and `src/prolix/padding.py:169`.
- `map_exclusions_to_dense_padded` — `src/prolix/padding.py:166`
  (`prolix.physics.neighbor_list`).
- `make_bundle_from_system(..., exclusion_spec=...)` — the param already exists
  (`src/prolix/physics/system.py:499`) and populates `excl_*`/`exception_*`.

## Work items (sequenced)

### A1 — Mass field on MolecularBundle (tractable first; independently correct)
- Add `masses: Float[Array, N]` to `MolecularBundle` (eqx.Module,
  `types/bundles.py:101`). Per-atom, padded to the atom bucket like `charges`.
- `make_bundle_from_system` stores `system.masses` (pad + mask) into the field;
  default to real element masses (never `ones`) — see `proxide.assign_masses`.
- `masses_for_bundle` (`bundle_md.py:99`) returns `bundle.masses[:n_atoms]`;
  drop the water-only special case (water masses come from the field too).
- **Verify:** pytree round-trips (tree_flatten/unflatten); all bundle
  construction sites updated; full `tests/api tests/tiling tests/bench -m "not slow"`
  stays green (baseline: currently 102 pass / pre-existing V4-V5 red — must not
  add failures); a protein bundle reports real masses (O≈16, not 1).

### A2 — JIT-safe exclusions through the bundle path (+ fix pre-existing V4/V5)
- In `_b1_paramize.paramize_pdb_to_bundle`: build `ExclusionSpec.from_protein(protein)`
  and pass `exclusion_spec=` to `make_bundle_from_system` (proxide `Protein`
  carries `exclusion_mask`, `pairs_14`, `scale_matrix_vdw/elec`, `coulomb14scale`,
  `lj14scale`, `scaled_radii`).
- Replace the numpy loop in `_dense_excl_matrices_from_bundle` with the existing
  JIT-safe padded machinery (`map_exclusions_to_dense_padded`) so it works under
  trace on both the host and stacked dispatch paths. This should also resolve the
  `int(traced n_excl)` concretization at `bundle_md.py:211` (finding #5).
- Add a **pipeline invariant** to the keystone (BATHOS): after building a bundle,
  assert median |grad| < ~1e3 kcal/mol/Å and finite energy over REAL atoms; make
  the self-test FAIL loudly on violation, and add a 100-step finite check (not 2).
- **Verify:** 2GB1 median |grad| drops 3.84e5 → O(10²); 1UBQ/1AKE finite at 100
  steps; **V4-HLO + V5 gates go GREEN** (this is the acceptance bar — they must
  flip from red to green); no other api/tiling/bench regressions.

### A3 — settle_langevin vacuum-protein dt (physics investigation; may be its own sub-sprint)
- Root-cause why dt ≤ 0.01 fs is required for a well-formed vacuum protein
  (A1+A2 applied). Suspect: rigid-water SETTLE projection / thermostat coupling
  applied to non-water DOF. Compare `settle_langevin` vs a plain BAOAB Langevin
  (no SETTLE) on a protein-only bundle at dt=0.5 fs.
- **Verify:** a real protein (2GB1) runs a stable finite trajectory at dt=0.5 fs
  (the B1-pinned timestep) for ≥1000 steps without divergence.

## Acceptance for B1 unblock
B1-full is unblocked when a protein bundle from `_b1_paramize` runs a finite
trajectory at the pinned dt=0.5 fs through `EnsemblePlan.from_bundles(...).run()`,
with median |grad| O(10²) and real masses, AND the V4/V5 gates are green.

## Reusable artifacts from the investigation (fork adf2c4a6, uncommitted then reverted)
- `_b1_paramize` exclusion-extraction via `ExclusionSpec.from_protein` + the
  `bundle_force_stats` invariant — correct approach; recover from the fork
  transcript / reflog when executing A2.
- `_b1_equilibrate.py` (standalone minimization) + `b1_init_exec.py --equilibrate`
  — NOT the fix; dropped. Reintroduce only if A3 shows raw crystals still need a
  light minimization once the energy is correct.

## Out of scope
Rescoping B1 to init-bound-only (the "Option B" alternative) — deferred; revisit
only if A3 proves too deep for the paper timeline.
