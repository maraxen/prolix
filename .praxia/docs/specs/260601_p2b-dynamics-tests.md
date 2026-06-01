---
task_id: 260601_p2b_crossval_harness
sprint: P2b cross-validation harness — dynamics + shim + kUPS
date: 260601
status: draft
gates:
  t1_nve: "max |E_total - E_init| / |E_init| < 0.01 over 1000 steps at dt=0.5 fs"
  t2_nvt: "|T_mean - 300| < 5 K over 10000 production steps, 216-water box"
  t3_npt: "no NaN in 2000-step NPT trajectory, T_mean in [250, 350] K, 4 waters"
  t4_shim: "bond + angle analytical forces agree with AD reference, atol=1e-4 kcal/mol/Å"
  t5_kups: "kUPS cross-val passes at n=64, 100, 500, 2000 within existing tolerance gates"
fixer_tasks:
  - t4-shim-parity
  - t1-nve-conservation
  - t2-nvt-216water
  - t3-npt-crossval
  - t5-kups-scale
---

# Spec: P2b Dynamics + Shim + kUPS Cross-Validation

## Context

This spec covers the **five remaining P2b test categories** that complete the Phase 2b
cross-validation suite per the strategic roadmap
(`.praxia/docs/roadmaps/prolix/260515_prolix-strategic-roadmap.md` §Phase 2b).

The first slice (nonbonded energy/force parity, f1–f7) is already done and merged:
`tests/physics/test_openmm_parity_nonbonded.py` (10 tests pass, PME included).

This spec provides the remaining evidence needed to file the §7.1 external-comparator figure.

## Prerequisites

- P1a MolecularBundle: **CLOSED** (commit 64249b7, 22/22 tests)
- NPT KE init bug: **FIXED** (commit b6e5bb9, T=260.6 K cold start confirmed)
- Nonbonded parity (f1–f7): **MERGED** (test_openmm_parity_nonbonded.py)

## Target API

- t4 (shim parity): uses `MolecularBundle` fixtures — P1a is the substrate here
- t1/t2/t3 (dynamics): uses `settle_langevin` / `settle_csvr_npt` from `prolix.physics.settle`
- t5 (kUPS): modifies existing `test_kups_thermostat_crossval.py`

## Constraints

- dt ≤ 0.5 fs for all NVT/NPT dynamics (SETTLE + Langevin coupling; CLAUDE.md Phase 2)
- NPT: short trajectory only (≤ 5 ps); long-trajectory divergence is Phase 6
- ANALYTICAL shim covers bonded terms only (bonds, angles) — NOT LJ/PME; dihedral deferred
- Float64 required for energy parity tests; use existing x64 fixture pattern
- All tests must work with `uv run pytest -m "not slow"` for CI; mark slow tests

---

## t1: NVE Energy Conservation Test

**Context**

No NVE integrator test exists. `settle_langevin` with `gamma=0.0` produces deterministic
Hamiltonian dynamics (BAOAB collapses to BAB = velocity Verlet). No SETTLE constraints
needed — use a vacuum harmonic oscillator with no `water_indices`.

There is no `velocity_verlet` entry in `step_sequences` (verified `step_system.py:746`).
Do not register a new sequence — use `settle_langevin(gamma=0.0, water_indices=None)`.

The kUPS test `test_kups_thermostat_crossval.py` demonstrates the same physical setup;
reuse the harmonic k=0.01 eV/Å², 64-particle, m=1 amu pattern.

**Implementation Steps**

1. Create `tests/physics/test_p2b_nve_conservation.py`.
2. Imports: `jax`, `jax.numpy as jnp`, `numpy as np`, `pytest`, `space` from `jax_md`,
   `settle` from `prolix.physics`, `spring_constant_ev_per_angstrom_sq_to_kcal_mol`
   from `prolix.physics.kups_adapter`, `AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL` from
   `prolix.simulate`.
3. Define helpers:
   - `_harmonic_ke_kcal(momentum, mass)` = `jnp.sum(momentum**2 / (2 * mass[:,None]))`
   - `_harmonic_pe_kcal(positions, k_kcal)` = `0.5 * k_kcal * jnp.sum(positions**2)`
4. In `test_nve_energy_conservation()`:
   - `jax.config.update("jax_enable_x64", True)`
   - N=64, k_kcal = `spring_constant_ev_per_angstrom_sq_to_kcal_mol(0.01)`, m_amu=1.0, dt_fs=0.5
   - `positions = jax.random.normal(jax.random.key(42), (N, 3), dtype=jnp.float64) * 0.5`
   - `mass = jnp.full(N, m_amu, dtype=jnp.float64)`
   - `disp_fn, shift_fn = space.free()`
   - `energy_fn = lambda pos: 0.5 * k_kcal * jnp.sum(pos**2)`
   - `dt_akma = 0.5 / AKMA_TIME_UNIT_FS`
   - `init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=300.0 * BOLTZMANN_KCAL, gamma=0.0, mass=mass, water_indices=None)`
   - `state = init_s(jax.random.key(0), positions, mass=mass)`
   - Run 1000 steps with `jax.jit(apply_s)`, collect E_total = KE + PE at each step
   - `assert max(abs(E_total - E_total[0])) / abs(E_total[0]) < 0.01`

**Files**: `tests/physics/test_p2b_nve_conservation.py` (create)

**Gate**: `uv run pytest tests/physics/test_p2b_nve_conservation.py::test_nve_energy_conservation -v`
— max energy drift < 1%.

**Scope estimate**: ~60 LOC

---

## t2: NVT 216-Water Temperature Stability Test

**Context**

Existing NVT tests use n=2 waters. The paper requires 216-water validation for §7.1 claims.

`_equil_water_positions(216, seed=42)` in
`tests/physics/test_explicit_langevin_tip3p_parity.py:50` returns positions subsampled
from a pre-equilibrated 30 Å TIP3P box (895 waters available). Box edge remains 30.0 Å.
PME grid: use 32 (covers 30 Å box adequately).

The energy function and integrator construction follows exactly the pattern in
`test_settle_temperature_control.py:27–56`. `rigid_tip3p_box_ke_kcal` computes kinetic
energy from rigid-body water decomposition.

216 × 3 = 648 atoms. PME on 648 atoms at 15000 steps ≈ 15–30 minutes on CPU.
Mark `@pytest.mark.slow`. Burn = 5000 steps (2.5 ps). Production = 10000 steps (5 ps).

**Implementation Steps**

1. Create `tests/physics/test_p2b_nvt_216water.py`.
2. Imports: follow `test_settle_temperature_control.py` exactly — `pbc`, `settle`, `system`,
   `rigid_tip3p_box_ke_kcal`, `AKMA_TIME_UNIT_FS`, `BOLTZMANN_KCAL`. Also import
   `_equil_water_positions`, `_proxide_params_pure_water` from
   `tests.physics.test_explicit_langevin_tip3p_parity`.
3. Helper: `_dof_rigid_tip3p_waters(n) = 6*n - 3` (as in existing test).
4. In `test_nvt_216water_temperature_stability()` with `@pytest.mark.slow`:
   - `jax.config.update("jax_enable_x64", True)`
   - n_waters=216, dt_fs=0.5, steps=15000, burn=5000, seed=42
   - `positions_a, box_edge = _equil_water_positions(216, seed=42)`
   - `box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)`
   - dt_akma = `0.5 / AKMA_TIME_UNIT_FS`, kT = `300.0 * BOLTZMANN_KCAL`
   - gamma_reduced = `1.0 * AKMA_TIME_UNIT_FS * 1e-3`
   - `sys_dict = _proxide_params_pure_water(n_waters)`
   - `displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)`
   - Build energy_fn and integrator following `test_settle_temperature_control.py:27–56`
     exactly (PME grid=32, pme_alpha=0.34, cutoff=9.0)
   - `water_indices = settle.get_water_indices(0, n_waters)`
   - `init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec, project_ou_momentum_rigid=True, projection_site="post_o")`
   - Run steps loop; collect T = `2 * ke / (dof * BOLTZMANN_KCAL)` for step >= burn
   - `assert abs(mean(temps) - 300.0) < 5.0`

**Files**: `tests/physics/test_p2b_nvt_216water.py` (create)

**Gate**: `uv run pytest tests/physics/test_p2b_nvt_216water.py -v -m slow` — `|T_mean - 300| < 5 K`.

**Scope estimate**: ~75 LOC

---

## t3: NPT Short Cross-Validation Test

**Context**

NPT is unblocked by commit b6e5bb9 (2026-06-01). Cold-start T=260.6 K confirmed.
Long-trajectory NPT divergence (>= 20 ps) remains a known issue (Phase 6).
This test validates the short-trajectory regime (1 ps, 4 waters) which is fully stable.

The existing `test_npt_barostat.py:test_npt_compiles_and_runs` uses only 10 steps (no
thermal measurement). Do NOT copy the xfail 5 ps / 64-water test. This new test uses
4 waters, 2000 steps (1 ps), and checks thermal sanity only.

Constructor follows exactly `test_npt_barostat.py:51-103` with n_waters=4.

**Implementation Steps**

1. Create `tests/physics/test_p2b_npt_crossval.py`.
2. Imports: `pbc`, `settle`, `system`, `rigid_tip3p_box_ke_kcal`, `AKMA_TIME_UNIT_FS`,
   `BOLTZMANN_KCAL`; `_grid_water_positions`, `_proxide_params_pure_water` from
   `tests.physics.test_explicit_langevin_tip3p_parity`.
3. In `test_npt_1ps_temperature_finite()`:
   - `jax.config.update("jax_enable_x64", True)`
   - n_waters=4, dt_fs=0.5, steps=2000, burn=200
   - `positions_a, box_edge = _grid_water_positions(4, spacing_angstrom=10.0)`
   - `box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)`
   - dt_akma = `0.5 / AKMA_TIME_UNIT_FS`, kT = `300.0 * BOLTZMANN_KCAL`
   - tau_baro_akma = 2000.0, tau_thermo_akma = 2000.0
   - `sys_dict = _proxide_params_pure_water(4)`, pme_grid = 16
   - Build energy_fn via `system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, ...)`
   - `init_s, apply_s = settle.settle_csvr_npt(energy_fn, shift_fn, dt=dt_akma, kT=kT, target_pressure_bar=1.0, tau_barostat_akma=tau_baro_akma, tau_thermostat_akma=tau_thermo_akma, mass=mass, water_indices=water_indices, box_init=box_vec)`
   - `apply_j = jax.jit(apply_s)`
   - `state = init_s(jax.random.key(42), jnp.array(positions_a), mass=mass, box=box_vec)`
   - Run 2000 steps; after each step assert `jnp.all(jnp.isfinite(state.positions))`
   - After loop, compute T_mean from steps >= burn; `assert 250 < T_mean < 350`

**Files**: `tests/physics/test_p2b_npt_crossval.py` (create)

**Gate**: `uv run pytest tests/physics/test_p2b_npt_crossval.py::test_npt_1ps_temperature_finite -v`
— no NaN + `T_mean in [250, 350] K`.

**Scope estimate**: ~70 LOC

---

## t4: ANALYTICAL vs AUTOGRAD Per-Term Force Parity

**Context**

`test_shim_mode.py:test_bond_forces_match_autograd` already tests `bond_forces` on a
3-atom system. t4 extends this to a full MolecularBundle fixture and adds angle force parity.
Scope is bonds + angles only.

**Dihedral out of scope**: `dihedral_forces_analytical` at `analytical_forces.py:472` expects
params shaped `(D, N_terms, 3)` with format `[n, phase, k]` per term. The MolecularBundle
stores `dihedral_params: (D, 4)` as `[n, phase, k, spare]` (flat, one term). These formats
are incompatible without reshaping. Add a TODO comment; dihedral parity is a follow-up.

The MolecularBundle constructor requires ~32 fields. Reuse the `_minimal_bundle` pattern
from `tests/physics/test_molecular_bundle.py:28`. Copy verbatim (do not import from test
file — fragile import chains in pytest). Keep copy in sync manually.

**Implementation Steps**

1. Create `tests/physics/test_p2b_shim_parity.py`.
2. Imports (adapt based on actual exported names):
   ```python
   import jax, jax.numpy as jnp, numpy as np, pytest
   from jax_md import space
   from prolix.physics.analytical_forces import bond_forces, angle_forces_analytical
   from prolix.types.bundles import MolecularBundle, MolecularShapeSpec
   ```
3. Copy `_minimal_bundle` from `test_molecular_bundle.py` verbatim as a local helper.
4. Write `test_bond_forces_vs_ad_on_bundle()`:
   - `jax.config.update("jax_enable_x64", True)`
   - Build a 10-atom bundle with 5 active bonds via `_minimal_bundle`, override positions,
     `bond_idx`, `bond_params=[r0=1.5, k=100]`, `bond_mask`
   - `disp_fn, _ = space.free()`
   - `f_analytical = bond_forces(positions, bond_idx[:5], bond_params[:5], bond_mask[:5], disp_fn)`
   - AD reference: `f_ad = -jax.grad(e_bonds)(positions)` where `e_bonds(r)` sums
     `0.5 * k * (dist - r0)^2` over active bonds
   - `assert jnp.allclose(f_analytical, f_ad, atol=1e-4)`
5. Write `test_angle_forces_vs_ad_on_bundle()`:
   - `jax.config.update("jax_enable_x64", True)`
   - Build 12-atom bundle with 5 active angles: zigzag chain at ~109.5 degrees
   - `angle_params: theta0=1.911 rad, k=50 kcal/mol/rad^2`
   - `f_analytical = angle_forces_analytical(positions, angle_idx, angle_params, disp_fn, angle_mask)`
   - AD reference via `jax.grad` of harmonic angle energy
   - `assert jnp.allclose(f_analytical, f_ad, atol=1e-4)`
6. Add at file top:
   ```python
   # TODO: dihedral force parity is deferred — dihedral_forces_analytical expects
   # params (D, N_terms, 3) but MolecularBundle stores dihedral_params (D, 4).
   # Format incompatibility requires param reshape; scope for follow-up task.
   ```

**Files**: `tests/physics/test_p2b_shim_parity.py` (create)

**Gate**: `uv run pytest tests/physics/test_p2b_shim_parity.py -v` — both bond and angle parity pass atol=1e-4.

**Scope estimate**: ~130 LOC

---

## t5: kUPS Cross-Validation at 3 System Sizes

**Context**

`tests/physics/test_kups_thermostat_crossval.py` hardcodes `N_PARTICLES = 64`.
The roadmap requires 100, 500, 2000 particles.

**Bug to fix first**: `create_harmonic_system` at line ~139 has `rng=None` in the signature
but the body checks `if key is None:` — rename `rng` -> `key` throughout.

The existing 4 test cases (BAOAB 0.5fs, BAOAB 1.0fs, CSVR 0.5fs, CSVR 1.0fs) at N=64
must continue to pass. New sizes are additive: 4 cases x 3 new sizes = 12 new combos.

N=500 and N=2000 are `@pytest.mark.slow`.

Step counts (`N_EQUIL_STEPS = 40000`, `N_SAMPLE_STEPS = 60000`) are calibrated per-DOF
— for larger N, variance per DOF decreases, so same step count gives tighter statistics.
No change needed.

**Implementation Steps**

1. Open `tests/physics/test_kups_thermostat_crossval.py`.
2. Fix bug: rename `rng` -> `key` in `create_harmonic_system` signature and all call-sites.
3. Remove module-level `N_PARTICLES = 64` constant.
4. Add parametrize fixture:
   ```python
   @pytest.fixture(params=[64, 100, 500, 2000], ids=lambda n: f"n{n}")
   def n_particles(request):
       return request.param
   ```
5. Thread `n_particles` into `test_kups_proxide_temperature_crossval`; replace all
   `N_PARTICLES` / `DOF` references with `n_particles` / `dof = 3 * n_particles`.
6. Add slow skip at test body start:
   ```python
   if n_particles >= 500:
       pytest.skip("slow — run with -m slow")
   ```
   Use `pytest.skip` conditional (not `add_marker`) for reliability with parametrize.
7. Verify: `pytest --collect-only -m "kups and not slow"` shows 8 tests (4 integrators x n=64,100).

**Files**: `tests/physics/test_kups_thermostat_crossval.py` (modify)

**Gate**: `uv run pytest tests/physics/test_kups_thermostat_crossval.py -m "kups and not slow" -v`
— all 8 not-slow tests pass at existing tolerance gates.

**Scope estimate**: ~45 LOC changed

---

## Dependency Ordering

```
t4 (shim parity)    — new file, no deps → implement first
t1 (NVE)            — new file, no deps → can parallel with t4
t5 (kUPS)           — modifies existing file, no deps
t2 (NVT 216-water)  — new file; needs dynamics pattern proven (t1)
t3 (NPT)            — new file; NPT KE fix confirmed (commit b6e5bb9)
```

Recommended order: t4 → t1 → t5 → t2 → t3

---

## Risks

| Risk | Mitigation |
|---|---|
| `settle_langevin` with `gamma=0.0` may not be supported | If raises/NaN, use gamma=1e-10 as effective zero; document in test |
| 216-water NVT wall-clock (~20-40 min CPU) fails CI | Mark @pytest.mark.slow; cluster CI only |
| NPT 4-water T range [250, 350] K may be flaky for short trajectory | If flaky: relax to [200, 400] K; anti-NaN check is the hard gate |
| `angle_forces_analytical` has internal Python loop that may fail under jit | Test without jit; if fails under trace, compare outside jit |
| Dihedral param shape mismatch (bundle (D,4) vs analytical (D,N_terms,3)) | Out of scope; document with TODO |
| `_minimal_bundle` copy vs import from test_molecular_bundle | Copy verbatim; keep in sync manually |

---

## Files Summary

| File | Action | Tasks |
|---|---|---|
| `tests/physics/test_p2b_nve_conservation.py` | create (~60 LOC) | t1 |
| `tests/physics/test_p2b_nvt_216water.py` | create (~75 LOC) | t2 |
| `tests/physics/test_p2b_npt_crossval.py` | create (~70 LOC) | t3 |
| `tests/physics/test_p2b_shim_parity.py` | create (~130 LOC) | t4 |
| `tests/physics/test_kups_thermostat_crossval.py` | modify (~45 LOC) | t5 |

No `src/` files are changed unless a bug surfaces during implementation.
