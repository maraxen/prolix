# Nonbonded Exception Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 80 kcal/mol LJ+Coulomb energy gap between prolix and OpenMM by correctly resolving per-atom-pair 1-4 nonbonded exceptions from the AMBER force field XML.

**Architecture:** proxide resolves each topology-derived 1-4 pair to explicit (chargeProd, sigma, epsilon) at parameterize time, replacing the current misaligned type-based exception table. prolix reads these resolved per-pair params, routes zero-epsilon pairs to full exclusion and nonzero pairs to a separate explicit-exception energy term computed outside the main LJ/Coulomb tiled kernels.

**Tech Stack:** Rust (proxide Rust crates), Python/JAX (prolix), uv, cargo, pytest

---

## File Map

| File | Change |
|------|--------|
| `proxide/crates/proxide-physics/src/physics/md_params.rs` | Add `resolved_nonbonded_14` + `resolved_nonbonded_14_params` fields; populate by resolving type-keyed exceptions against actual atom pairs |
| `proxide/crates/proxide_py/src/py_parsers.rs` | Export new aligned fields; remove old misaligned `exception_14_params` export |
| `prolix/src/prolix/physics/neighbor_list.py` | Add `exception_pairs`, `exception_sigmas`, `exception_epsilons`, `exception_chargeprods` to `ExclusionSpec`; wire in `from_protein()` |
| `prolix/src/prolix/physics/bonded.py` | Add `make_exception_pair_energy_fn` |
| `prolix/src/prolix/physics/system.py` | Wire exception energy into `make_energy_fn` total |
| `prolix/tests/physics/test_openmm_parity.py` | Already has the regression tests; all 5 failures should pass after this plan |

---

## Task 1: proxide — Resolve per-atom-pair 1-4 exception table

**Files:**
- Modify: `proxide/crates/proxide-physics/src/physics/md_params.rs`

The `MDParameters` struct currently has `pairs_14: Vec<[usize; 2]>` (from topology dihedrals) and `exception_14_params: Vec<[f32; 3]>` (from FF XML type-based exceptions — NOT aligned with `pairs_14`). We replace `exception_14_params` with `resolved_nonbonded_14_params: Vec<[f32; 3]>` that IS row-aligned with `pairs_14`.

- [ ] **Step 1: Add the new aligned field to the struct**

In `MDParameters` (line ~91), replace:
```rust
    /// Numeric-only view of exceptions for Python export: [chargeProd, sigma, epsilon]
    pub exception_14_params: Vec<[f32; 3]>,
```
with:
```rust
    /// Per-atom-pair resolved 1-4 exception params, row-aligned with pairs_14: [chargeProd, sigma, epsilon]
    /// For pairs with an explicit FF <Exception> entry: uses that entry's values.
    /// For all other pairs: chargeProd = coulomb14scale * q_i * q_j,
    ///                      sigma = (sigma_i + sigma_j) / 2,
    ///                      epsilon = lj14scale * sqrt(eps_i * eps_j)
    pub resolved_nonbonded_14_params: Vec<[f32; 3]>,
```

- [ ] **Step 2: Add a helper to build the resolved table**

After the `lookup_improper` function (around line ~740), add:

```rust
fn resolve_14_params(
    pairs_14: &[[usize; 2]],
    charges: &[f32],
    sigmas: &[f32],
    epsilons: &[f32],
    atom_types: &[String],
    exceptions: &[proxide_core::forcefield::NonbondedException],
    lj14scale: f32,
    coulomb14scale: f32,
) -> Vec<[f32; 3]> {
    pairs_14
        .iter()
        .map(|&[i, j]| {
            let ti = &atom_types[i];
            let tj = &atom_types[j];
            // Look for explicit exception matching these atom types (either order)
            if let Some(exc) = exceptions.iter().find(|e| {
                (&e.type1 == ti && &e.type2 == tj) || (&e.type1 == tj && &e.type2 == ti)
            }) {
                [exc.charge_prod, exc.sigma, exc.epsilon]
            } else {
                // Lorentz-Berthelot combining rules + global 14 scaling
                let charge_prod = coulomb14scale * charges[i] * charges[j];
                let sigma = 0.5 * (sigmas[i] + sigmas[j]);
                let epsilon = lj14scale * (epsilons[i] * epsilons[j]).sqrt();
                [charge_prod, sigma, epsilon]
            }
        })
        .collect()
}
```

- [ ] **Step 3: Call it in the protein FF parameterizer**

In `parameterize_structure` (around line ~490, where the `Ok(MDParameters { ... })` block is), replace:
```rust
        exception_14_params: nonbonded_exceptions.iter().map(|(_, _, q, s, e)| [*q, *s, *e]).collect(),
```
with:
```rust
        resolved_nonbonded_14_params: resolve_14_params(
            &pairs_14,
            &charges,
            &sigmas,
            &epsilons,
            &atom_types,
            &ff.exceptions,
            ff.lj14scale,
            ff.coulomb14scale,
        ),
```

- [ ] **Step 4: Update the GAFF builder**

In the GAFF builder's `Ok(MDParameters { ... })` block (around line ~616), replace:
```rust
        exception_14_params: Vec::new(),
```
with:
```rust
        resolved_nonbonded_14_params: resolve_14_params(
            &pairs_14,
            &charges,
            &sigmas,
            &epsilons,
            &atom_types,
            &[],    // GAFF has no explicit exceptions
            0.5,
            0.833333,
        ),
```

- [ ] **Step 5: Verify it compiles**

```bash
cd /home/marielle/projects/proxide && cargo check -p proxide-physics 2>&1 | tail -5
```
Expected: `Finished` with only pre-existing warnings (unused import `NonbondedException`).

- [ ] **Step 6: Commit**

```bash
cd /home/marielle/projects/proxide
git add crates/proxide-physics/src/physics/md_params.rs
git commit -m "feat(physics): resolve per-atom-pair 1-4 nonbonded exceptions aligned with pairs_14"
```

---

## Task 2: proxide — Update Python export

**Files:**
- Modify: `proxide/crates/proxide_py/src/py_parsers.rs`

The Python export currently emits `exception_14_params` (misaligned). Replace it with `resolved_nonbonded_14_params` under the same Python key so prolix code that reads `protein.exception_14_params` gets the aligned data.

- [ ] **Step 1: Replace the export block**

In `py_parsers.rs` around line ~892, replace the entire `exception_14_params` export block:
```rust
        if !params.exception_14_params.is_empty() {
            let mut flat = Vec::with_capacity(params.exception_14_params.len() * 3);
            for x in &params.exception_14_params {
                flat.extend_from_slice(x);
            }
            scale_col(&mut flat, 3, COL_EXCEPTION_SIGMA, conv.length);
            scale_col(&mut flat, 3, COL_EXCEPTION_EPSILON, conv.energy);
            let arr = PyArray1::from_slice_bound(py, &flat);
            dict_bound.set_item(
                "exception_14_params",
                arr.reshape((params.exception_14_params.len(), 3)).unwrap(),
            )?;
        }
```
with:
```rust
        if !params.resolved_nonbonded_14_params.is_empty() {
            let mut flat = Vec::with_capacity(params.resolved_nonbonded_14_params.len() * 3);
            for x in &params.resolved_nonbonded_14_params {
                flat.extend_from_slice(x);
            }
            scale_col(&mut flat, 3, COL_EXCEPTION_SIGMA, conv.length);
            scale_col(&mut flat, 3, COL_EXCEPTION_EPSILON, conv.energy);
            let arr = PyArray1::from_slice_bound(py, &flat);
            dict_bound.set_item(
                "exception_14_params",
                arr.reshape((params.resolved_nonbonded_14_params.len(), 3)).unwrap(),
            )?;
        }
```

- [ ] **Step 2: Verify workspace compiles and tests pass**

```bash
cd /home/marielle/projects/proxide && cargo check --workspace 2>&1 | tail -5
cargo test --workspace 2>&1 | grep -E "test result|FAILED|error\["
```
Expected: all `test result: ok`, no `FAILED`.

- [ ] **Step 3: Bump version and commit**

In `/home/marielle/projects/proxide/Cargo.toml`, change:
```toml
version = "0.1.0-alpha.6"
```
to:
```toml
version = "0.1.0-alpha.7"
```

In `/home/marielle/projects/proxide/pyproject.toml`, change:
```toml
version = "0.1.0a6"
```
to:
```toml
version = "0.1.0a7"
```

```bash
cd /home/marielle/projects/proxide
git add crates/proxide_py/src/py_parsers.rs Cargo.toml Cargo.lock pyproject.toml
git commit -m "feat(py): export aligned exception_14_params and bump to v0.1.0-alpha.7"
git tag v0.1.0-alpha.7
git push origin main && git push origin v0.1.0-alpha.7
```

Wait for PyPI CI. Then in prolix:
```bash
cd /home/marielle/projects/prolix && uv add proxide==0.1.0a7 && uv sync
```

---

## Task 3: prolix — Extend ExclusionSpec with per-pair exception fields

**Files:**
- Modify: `prolix/src/prolix/physics/neighbor_list.py`

`ExclusionSpec` currently has only global `scale_14_vdw`/`scale_14_elec`. Add per-pair arrays for the explicit exception pairs that need separate energy computation.

- [ ] **Step 1: Write a failing test**

Add to `tests/physics/test_openmm_parity.py` in `TestEnergyDecomposition`:
```python
def test_exclusion_spec_exception_pairs_present(self, jax_openmm_system):
    """ExclusionSpec built from a Proline-containing protein must carry exception pairs."""
    data = jax_openmm_system
    spec = data["excl_spec"]
    assert hasattr(spec, "exception_pairs"), "ExclusionSpec missing exception_pairs"
    assert hasattr(spec, "exception_sigmas"), "ExclusionSpec missing exception_sigmas"
    assert hasattr(spec, "exception_epsilons"), "ExclusionSpec missing exception_epsilons"
    assert hasattr(spec, "exception_chargeprods"), "ExclusionSpec missing exception_chargeprods"
    # 1UAO has Proline — expect nonzero exception pairs
    assert len(spec.exception_pairs) > 0, "Expected nonzero exception pairs for 1UAO"
```

Run: `uv run pytest tests/physics/test_openmm_parity.py::TestEnergyDecomposition::test_exclusion_spec_exception_pairs_present -xvs 2>&1 | tail -5`
Expected: FAIL — `ExclusionSpec missing exception_pairs`

- [ ] **Step 2: Add fields to ExclusionSpec**

In `neighbor_list.py`, extend the dataclass (after `scale_14_vdw` around line ~38):
```python
  # Per-pair explicit exception data (resolved from FF XML, row-aligned)
  # Pairs with epsilon=0 are routed to idx_12_13 (full exclusion); these carry only nonzero-epsilon pairs.
  exception_pairs: Array        # (E, 2) int32 — atom index pairs
  exception_sigmas: Array       # (E,)   float32 — explicit sigma per pair (nm in AKMA)
  exception_epsilons: Array     # (E,)   float32 — explicit epsilon per pair
  exception_chargeprods: Array  # (E,)   float32 — explicit chargeProd per pair
```

- [ ] **Step 3: Update `from_protein()` to populate exception fields**

Replace the `return cls(...)` block at the end of `from_protein()` with:

```python
    # Resolve explicit per-pair exceptions from proxide (pairs_14 + exception_14_params are row-aligned)
    exc_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
    exc_sigmas = jnp.zeros((0,), dtype=jnp.float32)
    exc_epsilons = jnp.zeros((0,), dtype=jnp.float32)
    exc_chargeprods = jnp.zeros((0,), dtype=jnp.float32)

    if (hasattr(protein, 'pairs_14') and protein.pairs_14 is not None and len(protein.pairs_14) > 0
        and hasattr(protein, 'exception_14_params') and protein.exception_14_params is not None
        and len(protein.exception_14_params) > 0):

      pairs_np = np.asarray(protein.pairs_14, dtype=np.int32)        # (N, 2)
      params_np = np.asarray(protein.exception_14_params, dtype=np.float32)  # (N, 3) [chargeProd, sigma, epsilon]

      # epsilon is column 2; split into hard-excluded (eps≈0) and soft-exception (eps>0)
      eps_col = params_np[:, 2]
      zero_mask = np.abs(eps_col) < 1e-10

      # Hard-excluded: add to idx_12_13 (fully suppressed by main LJ/Coulomb kernel)
      hard_pairs = pairs_np[zero_mask]
      if len(hard_pairs) > 0:
        idx_12_13 = jnp.concatenate([idx_12_13, jnp.asarray(hard_pairs, dtype=jnp.int32)], axis=0)

      # ALL pairs_14 should be excluded from the main kernel (either hard or handled below separately)
      # Add remaining (nonzero-epsilon) pairs to idx_12_13 too; their energy is computed separately.
      soft_pairs = pairs_np[~zero_mask]
      soft_params = params_np[~zero_mask]
      if len(soft_pairs) > 0:
        idx_12_13 = jnp.concatenate([idx_12_13, jnp.asarray(soft_pairs, dtype=jnp.int32)], axis=0)
        exc_pairs = jnp.asarray(soft_pairs, dtype=jnp.int32)
        exc_sigmas = jnp.asarray(soft_params[:, 1], dtype=jnp.float32)
        exc_epsilons = jnp.asarray(soft_params[:, 2], dtype=jnp.float32)
        exc_chargeprods = jnp.asarray(soft_params[:, 0], dtype=jnp.float32)

    return cls(
      idx_12_13=idx_12_13,
      idx_14=excl.idx_14,
      scale_14_elec=c14,
      scale_14_vdw=l14,
      n_atoms=n_atoms,
      exception_pairs=exc_pairs,
      exception_sigmas=exc_sigmas,
      exception_epsilons=exc_epsilons,
      exception_chargeprods=exc_chargeprods,
    )
```

Also update the early-return path (no bonds) to include the new empty fields:
```python
      return cls(
        idx_12_13=jnp.zeros((0, 2), dtype=jnp.int32),
        idx_14=jnp.zeros((0, 2), dtype=jnp.int32),
        scale_14_elec=c14,
        scale_14_vdw=l14,
        n_atoms=n_atoms,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=jnp.float32),
        exception_epsilons=jnp.zeros((0,), dtype=jnp.float32),
        exception_chargeprods=jnp.zeros((0,), dtype=jnp.float32),
      )
```

- [ ] **Step 4: Run failing test — expect it to pass now**

```bash
uv run pytest tests/physics/test_openmm_parity.py::TestEnergyDecomposition::test_exclusion_spec_exception_pairs_present -xvs 2>&1 | tail -5
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/marielle/projects/prolix
git add src/prolix/physics/neighbor_list.py tests/physics/test_openmm_parity.py
git commit -m "feat(nl): add per-pair exception fields to ExclusionSpec"
```

---

## Task 4: prolix — Exception pair energy function

**Files:**
- Modify: `prolix/src/prolix/physics/bonded.py`

Add a function that computes 12-6 LJ + Coulomb for a small set of explicitly parameterized atom pairs without combining rules.

- [ ] **Step 1: Write a failing test**

Add to `tests/physics/test_openmm_parity.py`:
```python
def test_exception_pair_energy_fn_smoke(self, jax_openmm_system):
    """Exception pair energy fn must return a scalar for a known pair set."""
    from prolix.physics.bonded import make_exception_pair_energy_fn
    data = jax_openmm_system
    pos = data["jax_positions"]
    disp = data["displacement_fn"]
    pairs = jnp.array([[0, 5]], dtype=jnp.int32)
    sigmas = jnp.array([0.3], dtype=jnp.float32)
    epsilons = jnp.array([0.5], dtype=jnp.float32)
    chargeprods = jnp.array([0.1], dtype=jnp.float32)
    fn = make_exception_pair_energy_fn(disp, pairs, sigmas, epsilons, chargeprods, coulomb_constant=332.0637)
    e = fn(pos)
    assert jnp.isfinite(e), f"Exception pair energy must be finite, got {e}"
```

Run: `uv run pytest tests/physics/test_openmm_parity.py::TestEnergyDecomposition::test_exception_pair_energy_fn_smoke -xvs 2>&1 | tail -5`
Expected: FAIL — `cannot import name 'make_exception_pair_energy_fn'`

- [ ] **Step 2: Implement `make_exception_pair_energy_fn` in bonded.py**

Add at the end of `src/prolix/physics/bonded.py`:

```python
def make_exception_pair_energy_fn(
  displacement_fn,
  exception_pairs,      # (E, 2) int32
  exception_sigmas,     # (E,)   float32  — already in AKMA length units
  exception_epsilons,   # (E,)   float32  — already in AKMA energy units
  exception_chargeprods,# (E,)   float32  — q_i * q_j (with coulomb14scale already baked in)
  coulomb_constant: float = 332.0637,
):
  """Returns a function r -> scalar energy for explicit per-pair nonbonded exceptions.

  Uses Lennard-Jones 12-6 with the given sigma/epsilon (no combining rules),
  plus Coulomb with the given chargeProd. Intended for the small set of 1-4
  pairs that have explicit AMBER XML exception overrides.
  """
  if len(exception_pairs) == 0:
    return lambda r: jnp.zeros(())

  pairs = jnp.asarray(exception_pairs, dtype=jnp.int32)   # (E, 2)
  sigs  = jnp.asarray(exception_sigmas,    dtype=jnp.float32)  # (E,)
  eps   = jnp.asarray(exception_epsilons,  dtype=jnp.float32)  # (E,)
  qprods = jnp.asarray(exception_chargeprods, dtype=jnp.float32)  # (E,)

  def _energy(r):
    ri = r[pairs[:, 0]]   # (E, 3)
    rj = r[pairs[:, 1]]   # (E, 3)
    dij = jax.vmap(displacement_fn)(ri, rj)   # (E, 3)
    dist2 = jnp.sum(dij ** 2, axis=-1)        # (E,)
    dist  = jnp.sqrt(dist2)                   # (E,)

    # LJ 12-6
    sr6 = (sigs / jnp.maximum(dist, 1e-6)) ** 6
    e_lj = jnp.sum(eps * (sr6 ** 2 - 2.0 * sr6))

    # Coulomb
    e_coul = jnp.sum(coulomb_constant * qprods / jnp.maximum(dist, 1e-6))

    return e_lj + e_coul

  return _energy
```

- [ ] **Step 3: Run test — expect PASS**

```bash
uv run pytest tests/physics/test_openmm_parity.py::TestEnergyDecomposition::test_exception_pair_energy_fn_smoke -xvs 2>&1 | tail -5
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
cd /home/marielle/projects/prolix
git add src/prolix/physics/bonded.py tests/physics/test_openmm_parity.py
git commit -m "feat(bonded): add make_exception_pair_energy_fn for explicit AMBER 1-4 overrides"
```

---

## Task 5: prolix — Wire exception energy into system.py

**Files:**
- Modify: `prolix/src/prolix/physics/system.py`

Add the exception pair energy term to `make_energy_fn` so it contributes to `lj_energy_fn_bound` (or as a separate term added into `total_energy`).

- [ ] **Step 1: Build the exception energy fn inside `make_energy_fn`**

After the line `excl_idx, excl_sv, excl_se = neighbor_list.map_exclusions_to_dense_padded(exclusion_spec, max_exclusions=32)` (around line ~126), add:

```python
  # Exception pair energy (explicit AMBER 1-4 overrides, excluded from main kernel)
  _exception_energy_fn = None
  if exclusion_spec is not None and hasattr(exclusion_spec, 'exception_pairs'):
      ep = exclusion_spec.exception_pairs
      if ep is not None and ep.shape[0] > 0:
          from prolix.physics.bonded import make_exception_pair_energy_fn
          _exception_energy_fn = make_exception_pair_energy_fn(
              displacement_fn,
              ep,
              exclusion_spec.exception_sigmas,
              exclusion_spec.exception_epsilons,
              exclusion_spec.exception_chargeprods,
              coulomb_constant=COULOMB_CONSTANT,
          )
```

- [ ] **Step 2: Add to total_energy**

In `total_energy` (around line ~200), add exception energy after the LJ term:

```python
  def total_energy(r, neighbor=None, **kwargs_run):
    e_lj = lj_energy_fn_bound(r, neighbor)
    if _exception_energy_fn is not None:
        e_lj = e_lj + _exception_energy_fn(r)
    elec = electrostatics_energy_fn_bound(r, neighbor)
    ...
```

Also add it in the `return_decomposed` branch. Find where `"lj"` is returned in the decomposed dict (around line ~210) and update its lambda:

```python
      "lj": lambda r, n=None: lj_energy_fn_bound(r, n) + (
          _exception_energy_fn(r) if _exception_energy_fn is not None else 0.0
      ),
```

- [ ] **Step 3: Run the nonbonded parity test and record the new gap**

```bash
uv run pytest tests/physics/test_openmm_parity.py::TestEnergyDecomposition::test_nonbonded_combined_matches_openmm -xvs 2>&1 | grep -E "JAX=|OpenMM=|diff=|PASSED|FAILED"
```
Record the new JAX vs OpenMM values. The gap should be substantially reduced (target: < 1.0 kcal/mol).

- [ ] **Step 4: Run full parity suite**

```bash
uv run pytest tests/physics/test_openmm_parity.py -q 2>&1 | tail -15
```

- [ ] **Step 5: Commit**

```bash
cd /home/marielle/projects/prolix
git add src/prolix/physics/system.py
git commit -m "feat(system): wire explicit exception pair energy into total nonbonded"
```

---

## Task 6: Push proxide alpha.7 and update prolix

**Note:** This task should be done after Task 2's proxide changes are verified locally but before deploying to PyPI. Complete Tasks 3–5 using the local proxide build (via `uv add /home/marielle/projects/proxide` path dependency) so you don't block on PyPI propagation.

- [ ] **Step 1: Use local proxide during development**

```bash
cd /home/marielle/projects/prolix
uv add /home/marielle/projects/proxide
uv sync
```

- [ ] **Step 2: Once all prolix tests pass locally, push proxide tag**

```bash
cd /home/marielle/projects/proxide
git push origin main && git push origin v0.1.0-alpha.7
```

Wait for PyPI (check: `curl -s https://pypi.org/pypi/proxide/json | python3 -c "import sys,json; d=json.load(sys.stdin); print(sorted(d['releases'].keys())[-3:])"`)

- [ ] **Step 3: Switch prolix back to PyPI dependency**

```bash
cd /home/marielle/projects/prolix
uv add proxide==0.1.0a7
uv sync
```

- [ ] **Step 4: Sync to cluster and resubmit parity job**

```bash
rsync -azP --filter=':- .gitignore' \
  --exclude=.venv --exclude=.git --exclude=.agent --exclude=__pycache__ \
  /home/marielle/projects/prolix/ engaging:~/projects/prolix/

ssh engaging "cd ~/projects/prolix && uv sync --extra cuda 2>&1 | tail -5"

ssh engaging "mkdir -p ~/projects/prolix/logs && sbatch ~/projects/prolix/sbatch_parity_check.sbatch"
```

- [ ] **Step 5: Final commit on prolix**

```bash
cd /home/marielle/projects/prolix
git add pyproject.toml uv.lock
git commit -m "chore: bump proxide to 0.1.0a7 with aligned exception_14_params"
```

---

## Self-Review

**Spec coverage:**
- ✅ Proxide resolves per-atom-pair 1-4 exceptions at parameterize time (Task 1)
- ✅ Python export uses aligned arrays (Task 2)
- ✅ ExclusionSpec carries exception fields (Task 3)
- ✅ Exception pairs are fully excluded from the main tiled kernel (Task 3, added to idx_12_13)
- ✅ Separate exception energy term uses explicit sigma/epsilon/chargeProd (Task 4)
- ✅ Wired into make_energy_fn total and decomposed dict (Task 5)
- ✅ Cluster resubmit for parity regression (Task 6)

**Placeholder scan:** None found.

**Type consistency:** `exception_pairs (E,2) int32`, `exception_sigmas/epsilons/chargeprods (E,) float32` used consistently across Tasks 3, 4, 5.
