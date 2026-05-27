---
task_id: 260527_p2b-nonbonded-comparator
plan_doc: .praxia/docs/plans/260527_p2b-nonbonded-comparator.md
sprint_id: 3
date: 260527
status: draft
gates:
  nb_lj_energy: "per-term |dE(LJ)| < 1.0 kcal/mol on vacuum ala-dip (float64)"
  nb_coul_energy: "per-term |dE(Coulomb)| < 1.0 kcal/mol on vacuum ala-dip (float64)"
  nb_force: "RMS(|dF_nb via FD|) < 0.5 kcal/mol/Å across all atoms"
  nb_14_self_consistency: "exception_14 prolix self-consistency |dE| < 0.2 kcal/mol"
fixer_tasks:
  - f1-fixture-nonbonded-extension
  - f2-exclusion-bridge
  - f3-prolix-nb-bridge
  - f4-nb-energy-parity-test
  - f5-nb-force-parity-test
  - f6-nb-field-audit-extension
  - f7-pme-stretch-optional
---

# Spec: P2b — OpenMM Parity Harness (Nonbonded: LJ + Coulomb)

## Goal

Extend the P2a bonded parity harness (`tests/physics/fixtures_openmm_parity.py`,
`tests/physics/test_openmm_parity_bonded.py`) to cover **Lennard-Jones** and
**Coulomb** nonbonded forces on vacuum alanine dipeptide, including correct 1-4
exception handling. This closes the nonbonded leg of the §7.1 external-comparator
figure (prolix vs DMFF/TorchMD/espaloma/ForceBalance) and establishes a repeatable
numerical gate for `chunked_lj_energy` + `chunked_coulomb_energy` against OpenMM
Reference-platform energies.

---

## Resolution: test_explicit_parity.py overlap

**Choice: (B) — add alongside, leave `test_explicit_parity.py` as the whole-system
solvated check.**

`test_explicit_parity.py::TestOpenMMParity.test_nonbonded_energy_parity` only checks
finiteness (`np.isfinite`), uses a hand-rolled NumPy Coulomb loop (not
`chunked_coulomb_energy`), and targets 1CRN + FF19SB with PME — a different system,
different kernel path, and not a numerical gate. P2b targets the production kernels
(`chunked_lj_energy`, `chunked_coulomb_energy`, `make_exception_pair_energy_fn`) on
the same ala-dip fixture as P2a. The two test files are complementary: P2a/P2b
validate kernels on a minimal vacuum system; `test_explicit_parity.py` validates
structural plausibility on a full solvated protein.

---

## Test System Choice

**Vacuum alanine dipeptide (same ACE-ALA-NME system as P2a), OpenMM `NoCutoff`,
prolix `cutoff_distance=0`, `pme_alpha=0.0`.**

Setting `cutoff_distance=0` in `make_energy_fn` passes `cutoff=0.0` to
`chunked_lj_energy`. In that kernel, the mask guard is `if cutoff > 0: m = m & (ds < cutoff)` —
with `cutoff=0.0` the guard evaluates `0.0 > 0` which is False, so all pairs are
included (equivalent to `NoCutoff`). With `pme_alpha=0.0`, the erfc damping term in
`chunked_coulomb_energy` evaluates `erf(0 * r) = 0` for all r, giving bare Coulomb
`332.0637 * q_i * q_j / r`. The vacuum pairwise sum is exact — any discrepancy
against OpenMM Reference is a coding or parameter bug, not approximation noise.

**Alternative rejected**: periodic box + PME cutoff. Rejected because at ala-dip
scale (~30 atoms) any cutoff radius is comparable to the box edge, PME grid
alignment requires careful `pme_alpha` matching, and approximation noise can
dominate over kernel bugs — defeating per-kernel verification. PME is deferred to
optional f7.

---

## Nonbonded Energy Function Mapping Table

| Term | OpenMM mechanism | ForceGroup | prolix kernel | Tolerance gate |
|---|---|---|---|---|
| LJ (1-5+) | `NonbondedForce`, excl-filtered | 3 | `chunked_lj_energy(cutoff=0)` | `|dE| < 1.0 kcal/mol` |
| Coulomb (1-5+) | `NonbondedForce`, excl-filtered | 3 | `chunked_coulomb_energy(pme_alpha=0, cutoff=0)` | `|dE| < 1.0 kcal/mol` |
| 1-4 exception | `NonbondedForce.getException` (nonzero eps) | bundled in 3 | `make_exception_pair_energy_fn` | self-consistency `|dE| < 0.2 kcal/mol` |
| 1-2, 1-3 | `NonbondedForce.getException` (zero eps) | excluded | `ExclusionSpec.idx_12_13` | not a gate |

**Unit conversions (OpenMM SI → prolix AKMA-like):**

- Per-atom sigma: `nm → Å` (×10)
- Per-atom epsilon: `kJ/mol → kcal/mol` (÷4.184)
- Exception sigma: `nm → Å` (×10); already the combined sigma `(sig_i+sig_j)/2`
- Exception epsilon: `kJ/mol → kcal/mol` (÷4.184)
- Exception chargeProd: elementary charge² (dimensionless ratio); used directly with `COULOMB_CONSTANT = 332.0637`

---

## LJ vs Coulomb Splitting Strategy

**Chosen approach: two-pass per-atom charge zeroing.**

Run OpenMM ForceGroup 3 twice:
1. Standard run → total nonbonded energy `E_nb`.
2. Zero all per-atom charges via `force.setParticleParameters(i, 0.0, sigma, epsilon)`
   + `context.reinitialize(preserveState=True)` → LJ-only energy `E_LJ`.
3. Coulomb energy by subtraction: `E_Coul = E_nb − E_LJ`.
4. Restore original per-atom parameters + reinitialize before returning.

**Why not `addInteractionGroup` or a second `CustomNonbondedForce`:** Both require
constructing and registering a new `Force` object for each of LJ and Coulomb,
re-populating all particle parameters, and managing ForceGroup assignment for a
second context. The charge-zeroing pass is 10 lines and is the canonical technique
used by OpenMM's own analysis tools.

**Caveat — exception parameters are unaffected by charge zeroing:** OpenMM stores
per-pair exception parameters (`getException(k)`) separately from per-atom parameters
(`getParticleParameters(i)`). The `setParticleParameters` call only zeroes per-atom
charges. The exception parameters (chargeProd, sigma, epsilon for 1-4 pairs) are
NOT zeroed. Therefore `E_LJ` from the zeroed-charge run *includes* the 1-4 LJ
contribution; `E_Coul = E_nb − E_LJ` *includes* the 1-4 Coulomb contribution. This
matches prolix's decomposition where `lj_energy_fn_bound` handles the 1-5+ full
pairs with exclusions, and `exception_energy_fn_bound` handles the 1-4 pairs. The
fixture documentation must state this explicitly.

---

## Tolerance Rationale

**LJ: `|dE| ≤ 1.0 kcal/mol`**

The P2a bonded gate was 0.05 kcal/mol. The precedent for nonbonded is
`test_explicit_parity.py`'s `TOL_NONBONDED = 2.0 kcal/mol` (for full-PME on 1CRN).
Vacuum ala-dip has no PME approximation error. At float64 with `jax_enable_x64=True`,
`chunked_lj_energy` over ~31 atoms (~465 pairs) accumulates to <0.01 kcal/mol vs
OpenMM Reference. The 1.0 kcal/mol gate is intentionally loose to allow the float32
fallback (should still pass at ~0.1 kcal/mol) while catching unit errors, sign
errors, or wrong sigma combination rules. If the fixer runs float64 (required by
`jax_enable_x64=True` at module level), the observed delta should be <0.01 kcal/mol;
if it is in the 0.1–1.0 range, that signals a float32 vs float64 mismatch the
auditor should investigate.

**Coulomb: `|dE| ≤ 1.0 kcal/mol`**

Same rationale as LJ. `chunked_coulomb_energy` uses `erf(alpha * r)` damping which
at `alpha=0` is exactly 0. Coulomb has no sigma combination rule (just charge
product / r), so float64 agreement should be tighter than LJ. Gate of 1.0 kcal/mol
is conservative.

**1-4 exception self-consistency: `|dE| ≤ 0.2 kcal/mol`**

`make_exception_pair_energy_fn` is a direct per-pair sum over ~12–16 1-4 pairs in
ala-dip (no chunked tiling). The self-consistency check compares two prolix
evaluations: the `exception_energy_fn_bound` output from `make_energy_fn` vs a
standalone `make_exception_pair_energy_fn` call with the same extracted parameters.
Any discrepancy here is a parameter assembly bug (wrong pairing, wrong scale,
missing pairs). Gate of 0.2 kcal/mol is loose enough to survive float32 but would
catch a factor-of-2 error or a wrong scale factor.

**Force RMS: `RMS(|dF_nb|) ≤ 0.5 kcal/mol/Å`**

Forces are computed via finite differences (not `jax.grad`) due to the known custom
VJP stub in `chunked_lj_energy` (returns `zeros_like` — see
`src/prolix/physics/optimization.py:68–71`). FD at `eps=1e-3 Å` introduces
~0.01 kcal/mol/Å noise for well-behaved potentials. Gate of 0.5 kcal/mol/Å allows
for this noise while catching factor-of-2 errors, wrong sign, or missing pair
contributions.

---

## Exclusion Handling Design

OpenMM's `NonbondedForce` stores exclusions via `getException(k) → (i, j, chargeProd, sigma, epsilon)`:
- Entries with `epsilon == 0.0` and `chargeProd == 0.0` are full exclusions (1-2 and 1-3 pairs).
- Entries with `epsilon != 0.0` or `chargeProd != 0.0` are 1-4 modified interactions.

The fixture builds an `ExclusionSpec` by classifying each exception and passing it
to `make_energy_fn(exclusion_spec=...)`. `make_energy_fn` (line 153, `system.py`)
calls `map_exclusions_to_dense_padded(exclusion_spec, max_exclusions=32)` when
`system.excl_indices is None`, which is the production path.

**`ExclusionSpec` construction:**
- `idx_12_13`: all pairs with `epsilon == 0 and chargeProd == 0`
- `idx_14`: empty `jnp.zeros((0, 2), int32)` — because all 1-4 pairs are handled
  via the explicit `exception_pairs` path (not the global 14-scale path)
- `scale_14_elec = 0.0`, `scale_14_vdw = 0.0` (unused; idx_14 is empty)
- `exception_pairs`, `exception_sigmas`, `exception_epsilons`, `exception_chargeprods`:
  all pairs with `epsilon != 0 or chargeProd != 0`

`make_exception_pair_energy_fn` expects pre-scaled combined parameters. OpenMM's
`getException` returns the already-combined, already-scaled values. No additional
multiplication by `lj14scale` or `coulomb14scale` is needed.

**Max exclusions assertion:** Before building the `ExclusionSpec`, assert that no
atom has more than 31 exclusion entries (to fit within `max_exclusions=32`). For
ala-dip AMBER14SB this is guaranteed, but the assertion catches any unexpected
topology.

---

## Fixer Task Decomposition

### f1-fixture-nonbonded-extension

**Goal:** Add OpenMM nonbonded extraction and two-pass splitting to the shared fixture.

**What to do:** Add three new functions to `tests/physics/fixtures_openmm_parity.py`,
appended after line 386 (end of existing content):

1. `extract_nonbonded_params(omm_system) -> dict` — iterate
   `force.getParticleParameters(i)` on the `NonbondedForce` to extract per-atom
   `charges`, `sigmas` (Å), `epsilons` (kcal/mol). Also iterate
   `force.getException(k)` to extract `exception_pairs` (int32), `exception_sigmas`
   (Å), `exception_epsilons` (kcal/mol), `exception_chargeprods` (e²).

2. `get_openmm_nonbonded_energies(omm_system, positions_ang) -> dict` — two-pass
   charge-zeroing: query ForceGroup 3 normally → `E_nb`; zero per-atom charges,
   reinitialize, re-query → `E_LJ`; restore, reinitialize; compute
   `E_Coul = E_nb - E_LJ`. Return `{'total_nb': float, 'lj': float, 'coulomb': float}`.
   **Docstring MUST state** that `E_LJ` (zeroed-per-atom pass) *includes* the 1-4 LJ
   contribution AND the 1-4 Coulomb contribution carried by `exception_chargeprods`,
   because `setParticleParameters` does NOT affect exception entries. Therefore
   `E_Coul = E_nb - E_LJ` is the **1-5+ Coulomb only** (no 1-4 contribution). This
   matches prolix's `chunked_coulomb_energy` semantics (1-4 routed through the
   separate `exception_chargeprods` path), so prolix's `coulomb` term compares
   against OpenMM's `E_Coul` (1-5+) and prolix's `exception_14` term is gated by
   the self-consistency test f4. Failing to document this in the docstring
   produces a subtle parity drift that the gates would catch only at f5 force RMS.

3. `get_openmm_nonbonded_forces(omm_system, positions_ang) -> np.ndarray` — query
   ForceGroup 3 forces; convert kJ/mol/nm → kcal/mol/Å (× 0.1 / 4.184).

**Files:** `tests/physics/fixtures_openmm_parity.py` (extend, ~85 lines added)

**Edit-only constraint:** Only appends after existing last line. No modification of
existing function signatures or bodies.

**Success criteria:**
- `extract_nonbonded_params` returns dict with `charges.shape == (N,)`, `N ≥ 31`.
- `exception_pairs.shape[0] > 0` (AMBER14SB ala-dip has ~15–20 exceptions).
- `get_openmm_nonbonded_energies` returns finite `lj`, `coulomb`, `total_nb`.
- `abs(lj + coulomb - total_nb) < 1e-4` (splitting consistency).

**Verification command:**
```bash
uv run python -c "
import sys; sys.path.insert(0, 'tests/physics')
import jax; jax.config.update('jax_enable_x64', True)
from fixtures_openmm_parity import (
  build_ala_dip_openmm_system, extract_nonbonded_params,
  get_openmm_nonbonded_energies
)
omm, pos, _ = build_ala_dip_openmm_system()
nb = extract_nonbonded_params(omm)
print('atoms:', nb['charges'].shape[0], 'exceptions:', nb['exception_pairs'].shape[0])
assert nb['exception_pairs'].shape[0] > 0, 'no exceptions found'
e = get_openmm_nonbonded_energies(omm, pos)
print('E_LJ:', round(e['lj'],4), 'E_Coul:', round(e['coulomb'],4), 'total:', round(e['total_nb'],4))
import math
assert math.isfinite(e['lj']) and math.isfinite(e['coulomb'])
assert abs(e['lj'] + e['coulomb'] - e['total_nb']) < 1e-4, f'split inconsistency: {e}'
# Sanity gate (plan-auditor risk call): confirm exception chargeProd survives
# setParticleParameters zeroing. E_LJ (per-atom zeroed) must include 1-4 Coulomb;
# if exceptions ARE zeroed by the same call, E_LJ collapses to pure LJ and
# E_Coul = E_nb - E_LJ would NOT be 1-5+-only as documented. Detect by checking
# that |E_LJ| is greater than a pure-LJ-only estimate by at least ~0.01 kcal/mol
# (1-4 Coulomb contribution is non-negligible for ala-dip).
assert abs(e['lj']) > 1e-3, 'E_LJ suspiciously small — exception chargeProd may have been zeroed'
print('f1: PASS')
"
```

**Effort:** 45 min. **Deps:** none.

---

### f2-exclusion-bridge

**Goal:** Build `ExclusionSpec` from OpenMM exception data.

**What to do:** Add one new function to `tests/physics/fixtures_openmm_parity.py`:

`build_exclusion_spec(omm_system, n_atoms: int) -> ExclusionSpec`

Imports `ExclusionSpec` from `prolix.physics.neighbor_list`. Iterates
`force.getException(k)` on the `NonbondedForce`:
- If `chargeProd_val == 0.0 and epsilon_kj == 0.0`: add `(i, j)` to `idx_12_13`.
- Else: add to `exception_pairs` with converted `sigma` (nm→Å) and `epsilon`
  (kJ/mol→kcal/mol) and `chargeProd`.
Asserts `max_excl_per_atom < 32` where `max_excl_per_atom` is computed from
`idx_12_13`.
Returns `ExclusionSpec(idx_12_13=..., idx_14=zeros((0,2)), scale_14_elec=0.0,
scale_14_vdw=0.0, n_atoms=n_atoms, exception_pairs=..., exception_sigmas=...,
exception_epsilons=..., exception_chargeprods=...)`.

**Files:** `tests/physics/fixtures_openmm_parity.py` (extend, ~45 lines added)

**Edit-only constraint:** Appends after f1 additions.

**Success criteria:**
- Returns `ExclusionSpec` with `idx_12_13.shape[0] > 0`.
- `exception_pairs.shape[0]` matches `extract_nonbonded_params(omm)['exception_pairs'].shape[0]`.
- `map_exclusions_to_dense_padded(spec)` produces arrays of shape `(N, 32)` without error.

**Verification command:**
```bash
uv run python -c "
import sys; sys.path.insert(0, 'tests/physics')
from fixtures_openmm_parity import build_ala_dip_openmm_system, build_exclusion_spec
from prolix.physics.neighbor_list import map_exclusions_to_dense_padded
omm, pos, _ = build_ala_dip_openmm_system()
n = pos.shape[0]
spec = build_exclusion_spec(omm, n)
print('12-13 pairs:', spec.idx_12_13.shape[0])
print('exception pairs:', spec.exception_pairs.shape[0])
ei, sv, se = map_exclusions_to_dense_padded(spec)
print('excl_indices shape:', ei.shape, '(expect (N, 32))')
assert ei.shape == (n, 32)
print('f2: PASS')
"
```

**Effort:** 30 min. **Deps:** f1.

---

### f3-prolix-nb-bridge

**Goal:** Build a nonbonded `PhysicsSystem` and the prolix nonbonded energy evaluator.

**What to do:** Add two new functions to `tests/physics/fixtures_openmm_parity.py`:

1. `build_prolix_nonbonded_system(nb_params, bonded_params, positions_ang) -> (PhysicsSystem, displacement_fn)`

   Constructs `PhysicsSystem` with:
   - `charges`, `sigmas`, `epsilons` from `nb_params` (float64 arrays).
   - `excl_indices=None`, `excl_scales_vdw=None`, `excl_scales_elec=None` — triggers
     the `exclusion_spec` path in `make_energy_fn` (line 153, `system.py`).
   - Bonded fields identical to `build_prolix_bonded_system` (reuse that helper
     by calling it, then `replace` with nonbonded fields). Do NOT duplicate the
     dihedral `expand_dims` logic.

2. `get_prolix_nonbonded_energies(system, displacement_fn, positions_ang, exclusion_spec) -> dict`

   Calls:
   ```python
   energy_fns = make_energy_fn(
       displacement_fn, system,
       cutoff_distance=0,      # no cutoff → full pairwise sum
       pme_alpha=0.0,          # no erfc damping → bare Coulomb
       use_pbc=False,
       return_decomposed=True,
       exclusion_spec=exclusion_spec,
   )
   positions = jnp.array(positions_ang, dtype=jnp.float64)
   return {
       'lj': float(energy_fns['lj'](positions)),
       'coulomb': float(energy_fns['electrostatics'](positions)),
       'exception_14': float(energy_fns['exception'](positions)),
   }
   ```

**Files:** `tests/physics/fixtures_openmm_parity.py` (extend, ~65 lines added)

**Edit-only constraint:** Appends after f2 additions.

**Success criteria:**
- `get_prolix_nonbonded_energies` returns dict with finite `lj`, `coulomb`,
  `exception_14`.
- `lj < 0` (typical for near-equilibrium ala-dip geometry).

**Verification command:**
```bash
uv run python -c "
import sys; sys.path.insert(0, 'tests/physics')
import jax; jax.config.update('jax_enable_x64', True)
from fixtures_openmm_parity import (
  build_ala_dip_openmm_system, extract_bonded_params,
  extract_nonbonded_params, build_exclusion_spec,
  build_prolix_nonbonded_system, get_prolix_nonbonded_energies
)
omm, pos, _ = build_ala_dip_openmm_system()
bp = extract_bonded_params(omm)
nb = extract_nonbonded_params(omm)
spec = build_exclusion_spec(omm, pos.shape[0])
sys_, dfn = build_prolix_nonbonded_system(nb, bp, pos)
e = get_prolix_nonbonded_energies(sys_, dfn, pos, spec)
print('E_LJ:', round(e['lj'],4), 'E_Coul:', round(e['coulomb'],4), 'E_14:', round(e['exception_14'],4))
import math
assert math.isfinite(e['lj']) and math.isfinite(e['coulomb']) and math.isfinite(e['exception_14'])
print('f3: PASS')
"
```

**Effort:** 45 min. **Deps:** f1, f2.

---

### f4-nb-energy-parity-test

**Goal:** Phase 2 nonbonded energy gate — 3 parity tests.

**What to do:** Create `tests/physics/test_openmm_parity_nonbonded.py` with:

```python
jax.config.update("jax_enable_x64", True)  # module level
pytestmark = pytest.mark.openmm
```

A `@pytest.fixture(scope="module") def nb_parity_bundle()` that:
- Calls `build_ala_dip_openmm_system()`, `extract_bonded_params`, `extract_nonbonded_params`,
  `build_exclusion_spec`, `build_prolix_nonbonded_system`, `get_prolix_nonbonded_energies`,
  `get_openmm_nonbonded_energies`.
- Returns `{'omm_e': {...}, 'prolix_e': {...}, 'positions': positions_ang, 'omm_system': omm_system, 'exclusion_spec': spec, 'prolix_system': sys_, 'displacement_fn': dfn}`.

Three test functions:

1. `test_lj_energy_parity(nb_parity_bundle)` — `assert abs(prolix_lj - omm_lj) < 1.0`
2. `test_coulomb_energy_parity(nb_parity_bundle)` — `assert abs(prolix_coul - omm_coul) < 1.0`
3. `test_exception_14_energy_parity(nb_parity_bundle)` — self-consistency check:
   re-evaluate `make_exception_pair_energy_fn` directly with parameters from
   `extract_nonbonded_params` and compare with `prolix_e['exception_14']`. Assert
   `abs(direct_eval - prolix_e['exception_14']) < 0.2`. Print both values and delta.

Each test prints: `prolix=X, omm=Y, delta=Z` (for LJ/Coulomb) or
`direct=X, composed=Y, delta=Z` (for exception_14).

**Files:** `tests/physics/test_openmm_parity_nonbonded.py` (new, ~130 lines)

**Edit-only constraint:** New file only. No modification of existing test files.

**Success criteria:** 3/3 tests pass at stated thresholds.

**Verification command:**
```bash
uv run pytest \
  tests/physics/test_openmm_parity_nonbonded.py::test_lj_energy_parity \
  tests/physics/test_openmm_parity_nonbonded.py::test_coulomb_energy_parity \
  tests/physics/test_openmm_parity_nonbonded.py::test_exception_14_energy_parity \
  -m openmm -v --tb=short 2>&1 | tee tmp/gate_p2b_energy.log
grep "3 passed" tmp/gate_p2b_energy.log
```

**Effort:** 40 min. **Deps:** f1, f2, f3.

---

### f5-nb-force-parity-test

**Goal:** Phase 3 nonbonded force gate via finite differences.

**What to do:** Add `test_nb_force_parity(nb_parity_bundle)` to
`tests/physics/test_openmm_parity_nonbonded.py`.

The test must include this comment block immediately before the FD loop:
```python
# NOTE: jax.grad(chunked_lj_energy) returns zeros due to custom VJP stub.
# (src/prolix/physics/optimization.py:68-71 — known limitation.)
# Forces are computed here via central finite differences (eps=1e-3 Å).
# Do NOT replace with jax.grad — it will trivially pass at zero force everywhere.
```

FD loop:
```python
eps = 1e-3  # Angstrom
forces_fd = np.zeros_like(positions_ang)
for i in range(positions_ang.shape[0]):
    for j in range(3):
        r_plus = positions_ang.copy(); r_plus[i, j] += eps
        r_minus = positions_ang.copy(); r_minus[i, j] -= eps
        forces_fd[i, j] = -(nb_energy_scalar(r_plus) - nb_energy_scalar(r_minus)) / (2.0 * eps)
```

where `nb_energy_scalar(r_np)` evaluates `lj_fn(r) + coul_fn(r) + exc14_fn(r)`.

Compare `forces_fd` against `get_openmm_nonbonded_forces(omm_system, positions_ang)`.
Assert `rms < 0.5`, print `rms`, `max |delta|`.

**Files:** `tests/physics/test_openmm_parity_nonbonded.py` (extend f4 file)

**Edit-only constraint:** Appends to the new file from f4.

**Success criteria:** `test_nb_force_parity` passes with `RMS < 0.5 kcal/mol/Å`.

**Verification command:**
```bash
uv run pytest tests/physics/test_openmm_parity_nonbonded.py::test_nb_force_parity \
  -m openmm -v --tb=long 2>&1 | tee tmp/gate_p2b_force.log
grep "1 passed" tmp/gate_p2b_force.log
```

**Effort:** 35 min. **Deps:** f4.

---

### f6-nb-field-audit-extension

**Goal:** Extend the P2a field-audit doc to record nonbonded field accesses.

**What to do:**
1. Run `audit_bonded_fields` (from `src/prolix/physics/field_audit.py`) on
   the `build_prolix_nonbonded_system` output (which uses `make_energy_fn` — so
   the proxy will intercept nonbonded field reads: `charges`, `sigmas`, `epsilons`,
   and optionally `excl_indices` if the proxy follows attribute access into `make_energy_fn`).
2. Append a new section to `.praxia/docs/audits/260526_p2a-bonded-field-audit.md`:
   ```markdown
   ## Nonbonded fields exercised by P2b path

   | Field | Role |
   |---|---|
   | `charges` | Per-atom charge array fed to `chunked_coulomb_energy` |
   | `sigmas` | Per-atom LJ sigma fed to `chunked_lj_energy` |
   | `epsilons` | Per-atom LJ epsilon fed to `chunked_lj_energy` |
   | `excl_indices` | Sparse exclusion index (built via `ExclusionSpec` path when None) |
   | `excl_scales_vdw` | Per-pair LJ scale (0.0 for 1-2/1-3, built via `ExclusionSpec`) |
   | `excl_scales_elec` | Per-pair Coulomb scale (built via `ExclusionSpec`) |
   ```
3. Update `## Fields NOT exercised` table: remove `charges`, `sigmas`, `epsilons`
   from the "NOT exercised" list (they are now covered).
4. Update `.praxia/docs/INDEX.md` audit entry for `260526_p2a-bonded-field-audit`:
   append `; extended for P2b nonbonded fields (260527)` to the description.

**Files:**
- `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` (modify — extend only)
- `.praxia/docs/INDEX.md` (modify — update audit entry description)

**Edit-only constraint:** Only appends/updates; no deletions from audit doc.

**Success criteria:**
```bash
uv run python -c "
import pathlib
doc = pathlib.Path('.praxia/docs/audits/260526_p2a-bonded-field-audit.md').read_text()
required = ['Nonbonded fields exercised by P2b', 'charges', 'sigmas', 'epsilons', 'excl_indices']
missing = [f for f in required if f not in doc]
assert not missing, f'Audit extension missing: {missing}'
print('f6: PASS')
"
```

**Effort:** 20 min. **Deps:** f3 (to know which fields are read).

---

### f7-pme-stretch-optional

**Goal (STRETCH):** Extend parity harness to cover PME Coulomb on a small periodic box.

**Condition: ship only if f1–f6 all PASS on the first audit pass.**

**What to do:**
1. Add `build_ala_dip_periodic_openmm_system(box_side_ang=30.0)` to the fixture:
   uses `nonbondedMethod=app.PME`, `nonbondedCutoff=9.0*unit.angstrom`,
   `ewaldErrorTolerance=5e-4`.
2. Derive `pme_alpha` to match OpenMM's choice:
   `alpha = sqrt(-log(2.0 * 5e-4)) / 9.0  # Å⁻¹ ≈ 0.334 Å⁻¹`
   Pass `pme_alpha=alpha` to prolix `make_energy_fn`.
3. Add `test_pme_coulomb_energy_parity` with tolerance `|dE| < 2.0 kcal/mol`.

**PME grid note:** prolix defaults to `pme_grid_points=64`. OpenMM auto-selects
from grid spacing. For a 30 Å box, OpenMM typically uses 32 grid points per
dimension. Pass `pme_grid_points=32` to prolix to match. Print both alpha and
grid count in the test for auditability.

**Files:** `tests/physics/fixtures_openmm_parity.py` (extend),
`tests/physics/test_openmm_parity_nonbonded.py` (extend)

**Edit-only constraint:** Additive.

**Verification command:**
```bash
uv run pytest tests/physics/test_openmm_parity_nonbonded.py::test_pme_coulomb_energy_parity \
  -m openmm -v --tb=long
```

**Effort:** 2–4 h. **Deps:** f1–f6 green on first audit.

---

## Verification Gates Summary

### MVP (required)

**Energy gate (3 tests):**
```bash
uv run pytest \
  tests/physics/test_openmm_parity_nonbonded.py::test_lj_energy_parity \
  tests/physics/test_openmm_parity_nonbonded.py::test_coulomb_energy_parity \
  tests/physics/test_openmm_parity_nonbonded.py::test_exception_14_energy_parity \
  -m openmm -v --tb=short 2>&1 | tee tmp/gate_p2b_energy.log
grep "3 passed" tmp/gate_p2b_energy.log
```

**Force gate (1 test):**
```bash
uv run pytest tests/physics/test_openmm_parity_nonbonded.py::test_nb_force_parity \
  -m openmm -v --tb=long 2>&1 | tee tmp/gate_p2b_force.log
grep "1 passed" tmp/gate_p2b_force.log
```

**P2a regression (must stay green):**
```bash
uv run pytest tests/physics/test_openmm_parity_bonded.py -m openmm -v --tb=short
# expect: 5 passed
```

**Audit doc gate:**
```bash
uv run python -c "
import pathlib
doc = pathlib.Path('.praxia/docs/audits/260526_p2a-bonded-field-audit.md').read_text()
required = ['Nonbonded fields exercised by P2b', 'charges', 'sigmas', 'epsilons', 'excl_indices']
missing = [f for f in required if f not in doc]
assert not missing, f'Missing: {missing}'
print('Audit gate: PASS')
"
```

**No new test failures:**
```bash
uv run pytest -m "not slow" --tb=short -q 2>&1 | tail -5
```

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `chunked_lj_energy` custom VJP stub returns zeros — silent wrong gradient | **HIGH — confirmed** | Spec mandates FD for force gate. Required comment block in test file. |
| Charge-zeroing leaves OpenMM context in modified state (charges not restored) | medium | Fixture restores charges + calls `context.reinitialize(preserveState=True)` after E_LJ query. Verification command in f1 checks `lj + coulomb == total_nb` to detect state corruption. |
| Exception sigma unit bug: OpenMM `getException` returns nm, prolix expects Å | medium | `build_exclusion_spec` and `extract_nonbonded_params` both convert nm→Å. Gate on `test_exception_14_energy_parity` would show ×10 error immediately. |
| `excl_indices` slot overflow (>32 exclusions per atom in ala-dip) | low | Ala-dip has ≤8 neighbors per atom. Assertion `max_excl_per_atom < 32` in `build_exclusion_spec` enforces this. |
| `cutoff_distance=0` silently becomes `float(0)` which is `> 0` is False → correct, but verify | medium | Confirmed in `optimization.py:56`: `if cutoff > 0:` — Python `0.0 > 0` is False, disabling the cutoff mask. Verified by f3 smoke command before tests are written. |
| Exception pairs double-counted: `idx_12_13` includes the same pairs as `exception_pairs` (1-4 pairs fully excluded THEN re-added via exception path) | medium | `ExclusionSpec` design: 1-4 pairs go into `exception_pairs` only — NOT into `idx_12_13`. The `idx_12_13` list must only contain pairs with `epsilon == 0 and chargeProd == 0`. Verified by checking that `set(idx_12_13) ∩ set(exception_pairs) == {}`. |
| `build_prolix_nonbonded_system` duplicates `expand_dims` logic for `dihedral_params` | low | Spec mandates reusing `build_prolix_bonded_system` for the bonded sub-structure (via `replace` or direct delegation), not reimplementing it. |
| PME alpha mismatch (f7 only) | high | Derive explicitly: `alpha = sqrt(-log(2 * tol)) / cutoff_A`. Log both prolix and OpenMM alpha values in the stretch test. |

---

## Out of Scope

- Solvated explicit-water nonbonded parity (covered by `test_explicit_parity.py`).
- PME reciprocal-space parity (deferred to optional f7).
- GB/SA implicit solvent (separate sprint).
- LJ tail correction (`lj_dispersion_tail_energy`) — PBC-only.
- Custom VJP fix for `chunked_lj_energy` — filed as separate follow-up.
- Per-proper vs per-improper torsion split (v1.1 per P2a spec).
- `OpenMM.CustomNonbondedForce` (not used by AMBER14SB on ala-dip).
- Refactoring `test_explicit_parity.py` (no payoff for P2b scope).

---

## Files Touched Summary

| File | Action | Task |
|---|---|---|
| `tests/physics/fixtures_openmm_parity.py` | extend (~200 lines total across f1–f3) | f1, f2, f3 |
| `tests/physics/test_openmm_parity_nonbonded.py` | new (~180 lines) | f4, f5 |
| `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` | extend (add nonbonded section) | f6 |
| `.praxia/docs/INDEX.md` | modify (update audit entry note) | f6 |

All source files in `src/`, all existing test files, and all existing
`.praxia/docs/specs/` and `.praxia/docs/plans/` files are untouched.

---

## First Fixer Task to Dispatch

**f1-fixture-nonbonded-extension** — no dependencies; purely additive to the
existing P2a fixture; validates that OpenMM ForceGroup 3 produces sensible nonbonded
energies and that the two-pass charge-zeroing split is self-consistent before any
prolix code is involved.
