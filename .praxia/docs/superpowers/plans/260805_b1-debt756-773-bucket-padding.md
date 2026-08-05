# B1 Bucket-Padding Reliability (debt 756 + debt 773) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make heterogeneous cross-protein bucket-matching a deliberate mechanism instead of a lucky find, and shrink the ghost-padding overhead for mid-size solvated proteins.

**Architecture:** Two independent, additive changes to the existing bucket/padding system in `src/prolix/types/bundles.py` and `src/prolix/physics/system.py`. Neither changes default behavior for any existing caller — both are opt-in (a new tuple entry that only affects counts that fall in the new range; a new optional kwarg that defaults to `None`).

1. **Debt 773** — insert one new rung in `ATOM_BUCKETS` (`types/bundles.py:33`) between the 1024→5000 gap, cutting ghost-padding overhead for ~1963-atom solvated proteins (e.g. 1VII) from 61% to ~4%.
2. **Debt 756** — add an optional `target_bucket_counts: dict[str, int] | None` parameter to `make_bundle_from_system` (`physics/system.py:609`), threaded through `solvate_protein_to_bundle` (`physics/solvation.py:566`), that lets a caller force any of the 7 non-atom bucket axes (bond/angle/dihedral/water/excl/cmap/exception) to pad up to an explicit count instead of the smallest bucket that fits the real data. This is the same semantic pattern already proven in the (deprecated) `pad_protein`/`pad_array` in `src/prolix/padding.py:65` — ported as a pattern, not reused as code, because that module is deprecated and doesn't do solvent merging.

**Tech Stack:** JAX (`jnp.pad`), pytest, existing `prolix.types.bundles` / `prolix.physics.system` bucket machinery.

**Recon basis:** This plan is grounded in a live trace of `origin/main` (commit `3ae0491`, 2026-08-05) — every file:line reference below was read directly off that commit, not inferred from the debt-item text.

---

## ⚠️ Landmine: two unrelated `ATOM_BUCKETS` constants exist

- `src/prolix/types/bundles.py:33` — `(64, 128, 256, 1_024, 5_000, 25_000, 60_000)`. **This is the one debt 773 means** — used by `MolecularBundle` / `make_bundle_from_system` / `MolecularShapeSpec`, the live B1 dispatch path.
- `src/prolix/padding.py:21` — `(1024, 2048, 2816, ..., 65536)`. Older, deprecated (`pad_protein` warns `DeprecationWarning` as of v1.1), unrelated system, already has a ~2048 rung. **Do not touch this one.**

Every task below only touches `types/bundles.py`.

---

### Task 0: Debt 773 — add an intermediate atom bucket

**Files:**
- Modify: `src/prolix/types/bundles.py:33`
- Test: `tests/physics/test_molecular_bundle.py`

**Step 1: Write the failing test**

Add to `tests/physics/test_molecular_bundle.py` (it already imports `ATOM_BUCKETS`, `BOND_BUCKETS`, etc. — check the top of the file and reuse that import):

```python
def test_atom_bucket_granularity_1963_atoms():
    """debt 773: a 1963-atom solvated protein (1VII, see
    scripts/experiments/profile_b1_heterogeneous_solvated_compile.py) should
    not fall into the 1024->5000 gap and pay 61% ghost padding."""
    next_bucket = min(b for b in ATOM_BUCKETS if b >= 1963)
    ghost_fraction = (next_bucket - 1963) / next_bucket
    assert next_bucket == 2_048, (
        f"expected an intermediate bucket at 2048 between 1024 and 5000, "
        f"got {next_bucket} (ghost fraction {ghost_fraction:.1%})"
    )
    assert ghost_fraction < 0.10, f"ghost fraction {ghost_fraction:.1%} still too high"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/physics/test_molecular_bundle.py::test_atom_bucket_granularity_1963_atoms -v`
Expected: FAIL — `next_bucket` currently resolves to `5000`, not `2048`.

**Step 3: Write minimal implementation**

In `src/prolix/types/bundles.py:33`, change:

```python
ATOM_BUCKETS = (64, 128, 256, 1_024, 5_000, 25_000, 60_000)
```

to:

```python
ATOM_BUCKETS = (64, 128, 256, 1_024, 2_048, 5_000, 25_000, 60_000)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/physics/test_molecular_bundle.py::test_atom_bucket_granularity_1963_atoms tests/physics/test_molecular_bundle.py::test_bucket_sizes_ordered -v`
Expected: both PASS. The `test_bucket_sizes_ordered` monotonicity test (already in the file) will pass automatically since the insertion preserves ordering.

**Step 5: Regression-check the PR #8 test that depends on current bucket boundaries**

`tests/api/test_v4_hlo_hetero_compile_once.py` (merged in PR #8, commit `3ae0491`) picks bundle sizes `(10, 100, 200, 300)` specifically to straddle buckets `64/128/256/1024` — all *before* the 1024→5000 gap this task touches. Confirm it's unaffected:

Run: `uv run pytest tests/api/test_v4_hlo_hetero_compile_once.py -v`
Expected: PASS, unchanged (2 passed).

**Step 6: Commit**

```bash
git add src/prolix/types/bundles.py tests/physics/test_molecular_bundle.py
git commit -m "fix(debt773): add intermediate 2048 atom bucket to cut ghost padding for ~2000-atom systems"
```

---

### Task 1: Debt 756a — `target_bucket_counts` support in `make_bundle_from_system`

**Files:**
- Modify: `src/prolix/physics/system.py`
  - Add helper after `_next_bucket` (currently `system.py:378-383`)
  - Modify `make_bundle_from_system` signature (currently `system.py:609-613`)
  - Modify the 7 count-computation sites (currently `system.py:640, 646, 651, 661, 674, 690, 750, 758`)
  - Modify the `*_bucket_idx` block (currently `system.py:771-778`) to reuse the same effective counts
- Test: `tests/physics/test_bundle_factory.py`

**Context — the exact current code you're changing** (`system.py:378-383`):

```python
def _next_bucket(n: int, buckets: tuple) -> int:
    """Return the smallest bucket >= n, or the largest if n exceeds all."""
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]
```

And the pattern repeated 7 times in `make_bundle_from_system` (example, bonds, `system.py:645-647`):

```python
nb = 0 if bonds is None or bonds.size == 0 else bonds.shape[0]
bb = _next_bucket(max(nb, 1), BOND_BUCKETS)
```

And the mirrored index computation (`system.py:771-778`):

```python
atom_bucket_idx = _bucket_idx(n, ATOM_BUCKETS)
bond_bucket_idx = _bucket_idx(max(nb, 1), BOND_BUCKETS)
angle_bucket_idx = _bucket_idx(max(na, 1), ANGLE_BUCKETS)
dihedral_bucket_idx = _bucket_idx(max(nd, 1), DIHEDRAL_BUCKETS)
water_bucket_idx = _bucket_idx(max(nw, 1), WATER_BUCKETS)
excl_bucket_idx = _bucket_idx(max(ne, 1), EXCL_BUCKETS)
cmap_bucket_idx = _bucket_idx(max(nc, 1), CMAP_BUCKETS)
exception_bucket_idx = _bucket_idx(max(nx, 1), EXCEPTION_BUCKETS)
```

**Both blocks must be changed together** — if you only patch the padding call (`bb = ...`) and not the `*_bucket_idx` call, the bundle's actual array size and its `MolecularShapeSpec` bucket index will disagree, breaking the JIT cache-key invariant the whole bucket system exists for.

**Step 1: Write the failing tests**

Add to `tests/physics/test_bundle_factory.py` (reuse the existing `_make_minimal_physics_system()` fixture — 10 atoms, 0 bonds/angles/dihedrals):

```python
def test_target_bucket_counts_pads_bond_bucket_higher():
    """debt 756: an explicit target should force a bigger bucket than the
    real (here: zero) bond count would naturally select."""
    sys = _make_minimal_physics_system()
    bundle = make_bundle_from_system(
        sys, boundary_condition="free", target_bucket_counts={"bond": 256}
    )
    assert bundle.bond_idx.shape[0] == 256
    from prolix.types.bundles import BOND_BUCKETS
    assert bundle.shape_spec.bond_bucket_idx == BOND_BUCKETS.index(256)


def test_target_bucket_counts_two_systems_can_be_forced_to_match():
    """The actual debt-756 use case: two systems with different real bond
    counts land in the SAME bucket (and therefore the same shape_spec) when
    both are given the same explicit target."""
    sys_a = _make_minimal_physics_system()  # 0 real bonds
    sys_b = _make_minimal_physics_system()  # also 0 real bonds in this fixture;
    # the assertion below is what matters -- both forced to the same target
    # produce identical bond_bucket_idx regardless of their real counts.
    bundle_a = make_bundle_from_system(
        sys_a, boundary_condition="free", target_bucket_counts={"bond": 1024}
    )
    bundle_b = make_bundle_from_system(
        sys_b, boundary_condition="free", target_bucket_counts={"bond": 1024}
    )
    assert bundle_a.shape_spec.bond_bucket_idx == bundle_b.shape_spec.bond_bucket_idx


def test_target_bucket_counts_rejects_target_below_real_count():
    """Overflow safety: an explicit target smaller than the real count must
    raise, not silently truncate real topology data (mirrors the safety
    check already proven in the deprecated padding.pad_array/pad_protein)."""
    sys = _make_minimal_physics_system()
    # This fixture has 10 real atoms; a target below that must raise.
    with pytest.raises(ValueError, match="atom"):
        make_bundle_from_system(
            sys, boundary_condition="free", target_bucket_counts={"atom": 5}
        )


def test_target_bucket_counts_none_is_backward_compatible():
    """Omitting target_bucket_counts (or passing None) must reproduce
    exactly the pre-existing behavior -- this is the regression guard."""
    sys = _make_minimal_physics_system()
    bundle_default = make_bundle_from_system(sys, boundary_condition="free")
    bundle_explicit_none = make_bundle_from_system(
        sys, boundary_condition="free", target_bucket_counts=None
    )
    assert bundle_default.shape_spec == bundle_explicit_none.shape_spec
    assert bundle_default.bond_idx.shape == bundle_explicit_none.bond_idx.shape
```

(`pytest` is already imported at the top of this test file.)

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/physics/test_bundle_factory.py -k target_bucket_counts -v`
Expected: FAIL — `make_bundle_from_system() got an unexpected keyword argument 'target_bucket_counts'` on all four.

**Step 3: Write the minimal implementation**

In `src/prolix/physics/system.py`, add this helper immediately after `_next_bucket` (after line 383):

```python
def _effective_bucket_count(real_n: int, target_n: int | None, axis_name: str) -> int:
    """Return the count to select a bucket against: the real count, or an
    explicit override that must be >= the real count.

    An explicit target can only pad UP, never truncate real topology data --
    mirrors the same safety check in the deprecated prolix.padding.pad_array.
    """
    if target_n is None:
        return real_n
    if target_n < real_n:
        raise ValueError(
            f"target_bucket_counts['{axis_name}']={target_n} is smaller than "
            f"the real count ({real_n}); an explicit target can only pad up, "
            f"never truncate real data."
        )
    return target_n
```

Change the signature at `system.py:609-613` from:

```python
def make_bundle_from_system(
    system,
    boundary_condition: str = "periodic",
    exclusion_spec: Any = None,
) -> MolecularBundle:
```

to:

```python
def make_bundle_from_system(
    system,
    boundary_condition: str = "periodic",
    exclusion_spec: Any = None,
    target_bucket_counts: dict[str, int] | None = None,
) -> MolecularBundle:
```

Also update the docstring's `Args:` block to document the new parameter:

```python
        target_bucket_counts: Optional per-axis overrides (keys: "atom",
            "bond", "angle", "dihedral", "water", "excl", "cmap",
            "exception") forcing that axis to pad up to an explicit count
            instead of the smallest bucket that fits the real data. Lets
            two systems with different real topology be forced into the
            same MolecularShapeSpec bucket (debt 756). Raises ValueError if
            an override is smaller than the real count for that axis.
```

Right after `pos = system.positions` / `n = pos.shape[0]` (around `system.py:638-640`), add:

```python
    _tbc = target_bucket_counts or {}
    n_eff = _effective_bucket_count(n, _tbc.get("atom"), "atom")
    a = _next_bucket(n_eff, ATOM_BUCKETS)
```

replacing the old `a = _next_bucket(n, ATOM_BUCKETS)` (line 640).

Then for each of the 7 remaining axes, insert an `*_eff` computation immediately after the existing real-count line and use it in place of the raw count in the `_next_bucket` call. Concretely (each is a 1-line insertion + 1-line edit at the sites already identified above):

- Bonds (`system.py:645-646`):
  ```python
  nb = 0 if bonds is None or bonds.size == 0 else bonds.shape[0]
  nb_eff = _effective_bucket_count(nb, _tbc.get("bond"), "bond")
  bb = _next_bucket(max(nb_eff, 1), BOND_BUCKETS)
  ```
- Angles (`system.py:650-651`):
  ```python
  na = 0 if angles is None or angles.size == 0 else angles.shape[0]
  na_eff = _effective_bucket_count(na, _tbc.get("angle"), "angle")
  ab = _next_bucket(max(na_eff, 1), ANGLE_BUCKETS)
  ```
- Dihedrals (`system.py:660-661`, after the `_flatten_multi_term_torsions` call that produces `nd`):
  ```python
  nd_eff = _effective_bucket_count(nd, _tbc.get("dihedral"), "dihedral")
  db = _next_bucket(max(nd_eff, 1), DIHEDRAL_BUCKETS)
  ```
- Water (`system.py:673-674`):
  ```python
  nw = 0 if wi is None or wi.size == 0 else wi.shape[0]
  nw_eff = _effective_bucket_count(nw, _tbc.get("water"), "water")
  wb = _next_bucket(max(nw_eff, 1), WATER_BUCKETS)
  ```
- Exclusions (`system.py:689-690`, after `ne` is computed by `_dense_excl_to_pair_list`):
  ```python
  ne_eff = _effective_bucket_count(ne, _tbc.get("excl"), "excl")
  eb = _next_bucket(max(ne_eff, 1), EXCL_BUCKETS)
  ```
- Exceptions (`system.py:749-750`):
  ```python
  nx = 0 if exc_pairs is None or exc_pairs.size == 0 else int(exc_pairs.shape[0])
  nx_eff = _effective_bucket_count(nx, _tbc.get("exception"), "exception")
  xb = _next_bucket(max(nx_eff, 1), EXCEPTION_BUCKETS)
  ```
- CMAP (`system.py:757-758`):
  ```python
  nc = 0 if cmap_t is None or cmap_t.size == 0 else int(cmap_t.shape[0])
  nc_eff = _effective_bucket_count(nc, _tbc.get("cmap"), "cmap")
  cb = _next_bucket(max(nc_eff, 1), CMAP_BUCKETS)
  ```

**Important:** the exclusion "per-atom-row form" block (`system.py:696-730`, the `excl_dense_indices` computation) uses `a` for its padding width (`pad_rows = a - dense_idx.shape[0]`) — that already picks up the atom-axis override automatically since `a` is now derived from `n_eff`. No separate change needed there.

Finally, update the `*_bucket_idx` block (`system.py:771-778`) to reuse the `*_eff` variables computed above instead of the raw counts:

```python
    atom_bucket_idx = _bucket_idx(n_eff, ATOM_BUCKETS)
    bond_bucket_idx = _bucket_idx(max(nb_eff, 1), BOND_BUCKETS)
    angle_bucket_idx = _bucket_idx(max(na_eff, 1), ANGLE_BUCKETS)
    dihedral_bucket_idx = _bucket_idx(max(nd_eff, 1), DIHEDRAL_BUCKETS)
    water_bucket_idx = _bucket_idx(max(nw_eff, 1), WATER_BUCKETS)
    excl_bucket_idx = _bucket_idx(max(ne_eff, 1), EXCL_BUCKETS)
    cmap_bucket_idx = _bucket_idx(max(nc_eff, 1), CMAP_BUCKETS)
    exception_bucket_idx = _bucket_idx(max(nx_eff, 1), EXCEPTION_BUCKETS)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/physics/test_bundle_factory.py -v`
Expected: all pass, including the 4 new ones and every pre-existing test in the file (regression check for the backward-compat requirement).

**Step 5: Run the full bundle-factory-adjacent suite to catch any missed call site**

Run: `uv run pytest tests/physics/test_bundle_factory.py tests/physics/test_hp2_bundle_factory_parity.py tests/api/test_hp2_energy_fn_from_bundle.py tests/api/test_v1_ensemble_plan_parity.py tests/physics/test_molecular_bundle.py -v`
Expected: all pass (these are every test file that calls `make_bundle_from_system` directly per the recon sweep — none of them pass `target_bucket_counts`, so this is purely a backward-compatibility check).

**Step 6: Commit**

```bash
git add src/prolix/physics/system.py tests/physics/test_bundle_factory.py
git commit -m "feat(debt756): add target_bucket_counts override to make_bundle_from_system"
```

---

### Task 2: Debt 756b — thread `target_bucket_counts` through `solvate_protein_to_bundle`

**Files:**
- Modify: `src/prolix/physics/solvation.py:566-627` (`solvate_protein_to_bundle`)
- Test: wherever `tests/physics/test_flash_dense_parity.py` or `tests/physics/test_nl_kernel_exclusion_parity.py` already exercises `solvate_protein_to_bundle` — check both files' imports first; add a new small test near one of them, or create `tests/physics/test_solvate_protein_bucket_match.py` if neither is a natural fit (check before deciding — do not duplicate an existing solvation test file's fixtures).

**Context — current signature** (`solvation.py:566-574`):

```python
def solvate_protein_to_bundle(
    protein: Protein,
    padding: float = 10.0,
    model_type: WaterModelType = WaterModelType.TIP3P,
    ionic_strength: float = 0.0,
    neutralize: bool = True,
    target_box_size: Array | None = None,
    pme_alpha: float = 0.34,
    nonbonded_cutoff: float = 9.0,
):
```

and the final call (`solvation.py:622-627`):

```python
    return make_bundle_from_system(
        _BundleSourceProxy(),
        boundary_condition="periodic",
        exclusion_spec=merged.exclusion_spec,
    )
```

**Note:** `solvate_protein` (called inside `solvate_protein_to_bundle`, `solvation.py:602-609`) does NOT need any change — bond/angle/dihedral counts are a pure function of the protein's own topology and are untouched by solvation; only `make_bundle_from_system`'s bucket selection needs the override.

**Step 1: Write the failing test**

Use `tests/physics/test_flash_dense_parity.py` — it's the one of the two candidates that actually calls `solvate_protein_to_bundle` (`test_nl_kernel_exclusion_parity.py` deliberately avoids it: "thousands of atoms, too slow"). It already has `DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pdb"` (line 42), `TARGET_BOX_SIZE = [32.0, 32.0, 32.0]` (line 43), a `_ff_path()` helper (line 70), and a single-protein loader `_solvated_1vii_bundle()` (line 76) you can generalize rather than duplicate.

Add a new, parameterized sibling helper (leave `_solvated_1vii_bundle` itself untouched — other tests in the file depend on its exact fixture) plus the test, both at module level near the existing helper:

```python
def _load_solvated_bundle(pdb_name: str, target_bucket_counts=None):
    """Like _solvated_1vii_bundle but for any PDB, with target_bucket_counts
    threaded through (debt 756 validation)."""
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.solvation import solvate_protein_to_bundle
    from prolix.physics.water_models import WaterModelType

    jax.config.update("jax_enable_x64", False)

    pdb_path = DATA_DIR / pdb_name
    if not pdb_path.is_file():
        pytest.skip(f"Missing test PDB: {pdb_path}")
    ff_path = _ff_path()
    if not ff_path.is_file():
        pytest.skip(f"Missing bundled force field: {ff_path}")

    spec = OutputSpec(
        parameterize_md=True, force_field=str(ff_path), coord_format=CoordFormat.Full
    )
    protein = parse_structure(str(pdb_path), spec)
    return solvate_protein_to_bundle(
        protein,
        padding=8.0,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array(TARGET_BOX_SIZE),
        target_bucket_counts=target_bucket_counts,
    )


def test_solvate_protein_to_bundle_target_bucket_counts_forces_match():
    """debt 756: 1VII (villin headpiece) and 1UAO (10-residue Chignolin) do
    NOT naturally bucket-match on bond/angle/dihedral axes -- both proteins
    are far smaller than 5000 real bonds/angles/dihedrals, so a shared
    target of 5000 is safely >= either protein's real count (satisfies the
    _effective_bucket_count overflow guard) while forcing both into the
    identical bucket for each axis."""
    target_bucket_counts = {"bond": 5000, "angle": 5000, "dihedral": 5000}
    bundle_1vii = _load_solvated_bundle("1VII.pdb", target_bucket_counts)
    bundle_1uao = _load_solvated_bundle("1UAO.pdb", target_bucket_counts)
    assert bundle_1vii.shape_spec.bond_bucket_idx == bundle_1uao.shape_spec.bond_bucket_idx
    assert bundle_1vii.shape_spec.angle_bucket_idx == bundle_1uao.shape_spec.angle_bucket_idx
    assert (
        bundle_1vii.shape_spec.dihedral_bucket_idx
        == bundle_1uao.shape_spec.dihedral_bucket_idx
    )
```

(`data/pdb/1UAO.pdb` is confirmed present in the repo.)

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/physics/test_flash_dense_parity.py::test_solvate_protein_to_bundle_target_bucket_counts_forces_match -v`
Expected: FAIL — `solvate_protein_to_bundle() got an unexpected keyword argument 'target_bucket_counts'`.

**Step 3: Write the minimal implementation**

In `src/prolix/physics/solvation.py`, change the signature (`solvation.py:566-574`) to add one parameter:

```python
def solvate_protein_to_bundle(
    protein: Protein,
    padding: float = 10.0,
    model_type: WaterModelType = WaterModelType.TIP3P,
    ionic_strength: float = 0.0,
    neutralize: bool = True,
    target_box_size: Array | None = None,
    pme_alpha: float = 0.34,
    nonbonded_cutoff: float = 9.0,
    target_bucket_counts: dict[str, int] | None = None,
):
```

And thread it into the final call (`solvation.py:622-627`):

```python
    return make_bundle_from_system(
        _BundleSourceProxy(),
        boundary_condition="periodic",
        exclusion_spec=merged.exclusion_spec,
        target_bucket_counts=target_bucket_counts,
    )
```

Add one line to the docstring's parameter documentation noting the pass-through (mirror the phrasing already used for the two "correctness traps" paragraphs above it, but keep it to one sentence — this one isn't a trap, it's a direct passthrough).

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/physics/test_flash_dense_parity.py::test_solvate_protein_to_bundle_target_bucket_counts_forces_match -v`
Expected: PASS.

**Step 5: Regression-check callers of `solvate_protein_to_bundle`**

Run: `uv run pytest tests/physics/test_flash_dense_parity.py tests/physics/test_nl_kernel_exclusion_parity.py -v`
Expected: all pass unchanged (neither existing test passes `target_bucket_counts`, so this is a pure backward-compatibility check for the new default `None`).

**Step 6: Commit**

```bash
git add src/prolix/physics/solvation.py tests/physics/test_flash_dense_parity.py
git commit -m "feat(debt756): thread target_bucket_counts through solvate_protein_to_bundle"
```

---

### Task 3: Debt 756c — end-to-end validation against the real heterogeneous-batching confirmation script

**Files:**
- Modify: `scripts/experiments/profile_b1_heterogeneous_solvated_compile.py`

This is the script that produced the original "core thesis confirmed" result (1VII+2GB1 co-bucket by luck). Extending it to add a THIRD, deliberately-mismatched protein (1UAO) forced into the same bucket via the new mechanism is the real proof that debt 756 works end-to-end, not just in a unit test.

**Step 1: Add 1UAO to the protein table**

In `scripts/experiments/profile_b1_heterogeneous_solvated_compile.py`, extend `PROTEIN_PDBS` (currently just `1vii`/`2gb1`):

```python
PROTEIN_PDBS = {
    "1vii": "1VII.pdb",
    "2gb1": "2GB1.pdb",
    "1uao": "1UAO.pdb",  # debt 756 validation: does NOT naturally bucket-match
                          # 1vii/2gb1 on bond/angle/dihedral axes; forced via
                          # target_bucket_counts below.
}
```

**Step 2: Thread `target_bucket_counts` through `_load_and_solvate`**

Change `_load_and_solvate` (currently takes only `protein_key`) to accept an optional `target_bucket_counts` and pass it through to `solvate_protein_to_bundle`:

```python
def _load_and_solvate(protein_key: str, target_bucket_counts: dict | None = None):
    ...
    return solvate_protein_to_bundle(
        protein,
        padding=PADDING,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array(TARGET_BOX_SIZE),
        target_bucket_counts=target_bucket_counts,
    )
```

**Step 3: Add a `three_way` mode that includes 1uao with a forced bucket match**

In `_build_bundles`, add a branch that loads all three proteins, uses `partition_bundles_by_shape` (already imported/used elsewhere per the module's own docstring claim about `partition_bundles_by_shape returning 1 group`) with 1vii/2gb1 loaded at their natural bucket, and 1uao loaded WITH an explicit `target_bucket_counts` matching 1vii/2gb1's actual bond/angle/dihedral bucket sizes (read those off `b_1vii.shape_spec` first, then construct the dict), asserting all three now share one `shape_spec`. Follow the existing `combined` branch's `RuntimeError` guard pattern for the assertion.

**Step 4: Dry-run locally (L1 gate per CLUSTER.md)**

Run: `uv run python scripts/experiments/profile_b1_heterogeneous_solvated_compile.py --dry-run --mode three_way`
Expected: exits 0, prints shape_spec match confirmation for all three proteins, no GPU required.

**Step 5: Smoke-test locally (L2 gate, CPU, per CLUSTER.md's local-gate sequence — do NOT run a full-scale physics suite locally, see `local-compute-limits.md`)**

Run: `JAX_PLATFORMS=cpu uv run python scripts/experiments/profile_b1_heterogeneous_solvated_compile.py --mode three_way --replicas 3 --n-steps-list 2,5 --out /tmp/smoke_three_way.json`
Expected: exits 0, small JSON output confirming shared compile across all three real, differently-topologized proteins.

**Step 6: Extend the existing bathos sidecar before any cluster (L3) run**

`scripts/experiments/profile_b1_heterogeneous_solvated_compile.bth.toml` already exists (canonical `[experiment]` schema) — but its `hypothesis` and `[outcomes.pass]`/`[outcomes.marginal]` conditions are written specifically for the 1vii+2gb1 two-protein `--mode combined` case (`condition = "n_groups_from_partition = 1 AND r_squared IS NOT NULL AND r_squared > 0.7 AND ..."`). Per this project's bathos hard rules (CLAUDE.md §Experiment Tracking), do not run `--mode three_way` via `bth run` on the cluster until the sidecar's `hypothesis` paragraph is extended to describe the three-protein/forced-bucket-match case and you've confirmed the existing `n_groups_from_partition = 1` condition still means the right thing for three bundles instead of two (it should — `partition_bundles_by_shape` returning 1 group is agnostic to how many distinct bundles went in — but verify rather than assume). Fix the sidecar, don't bypass it with `--no-sidecar`.

**Step 7: Commit**

```bash
git add scripts/experiments/profile_b1_heterogeneous_solvated_compile.py scripts/experiments/profile_b1_heterogeneous_solvated_compile.bth.toml
git commit -m "feat(debt756): validate target_bucket_counts end-to-end with a genuinely mismatched third protein (1UAO)"
```

---

## Summary of what "done" looks like

- `ATOM_BUCKETS` has a new 2048 rung; 1963-atom systems pay ~4% ghost padding instead of 61%.
- `make_bundle_from_system` and `solvate_protein_to_bundle` both accept an optional `target_bucket_counts` dict; omitting it is provably identical to current behavior (Task 1's backward-compat test).
- A genuinely mismatched protein pair (1VII/2GB1 vs. 1UAO) can be deliberately forced into one shared `MolecularShapeSpec`, proven both by a fast unit test (Task 2) and the same experiment script that produced the original "core thesis confirmed" result (Task 3) — turning debt 756 from "found by trial" into "reproducible on demand."
- Debt 773 and 756 are independent commits; either can ship alone if review wants to split them.

## Explicitly out of scope (do not do these in this plan)

- Changing `src/prolix/padding.py` — separate, deprecated system, do not touch.
- CMAP axis validation beyond passthrough — no real protein in this codebase's test fixtures currently populates CMAP terms, so Task 1's CMAP branch is implemented (for API completeness/symmetry with the other 6 axes) but not exercised by Task 2/3's real-protein tests. If a future CMAP-bearing system is added, it inherits the mechanism for free.
- Any change to `EnsemblePlan`/`ensemble_dispatch.py` — this plan only makes bucket-matching *possible* on demand; wiring a policy for *when* to auto-select `target_bucket_counts` across a batch of proteins is separate follow-on work, not scoped here.
