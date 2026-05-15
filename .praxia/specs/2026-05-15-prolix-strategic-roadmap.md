# Prolix Strategic Roadmap — v1.2 → Paper

**Date:** 2026-05-15  
**Status:** Architecture-reviewed — ready for backlog commit  
**Sequence:** Internal validation → open-source release → preprint  
**Review sources:** advisor tool + code-architecture-advisor agent

---

## Project Positioning

| Dimension | vs OpenMM | vs kUPS |
|---|---|---|
| Differentiability | E2E JAX AD through energy and trajectory; OpenMM cannot backprop through MD | Both support AD; prolix adds tiling + heterogeneous batching |
| Accelerator | XLA-native, no custom CUDA kernels required (Pallas optional; see note below) | kUPS is also JAX-native but lacks tiling planner |
| Tiling | `AxisSpec`/`BatchPlan` planner controls vmap-vs-safe_map per axis; memory-bounded | Not present in kUPS |
| Heterogeneous batch | `N_SYSTEMS` axis uses safe_map; varying system sizes in one batch | kUPS requires same-size batching (vmap only) |
| Export / WASM | StableHLO → IREE-WASM; browser-deployable without server compute | Not available |
| Speed mode | `custom_jvp` shim provides analytical forces without AD overhead (bonded terms) | Not available |

**Pallas caveat:** "No custom CUDA kernels" is an accurate contrast with OpenMM's custom CUDA kernel approach. Pallas is an XLA-native kernel abstraction — if Pallas tiling is used in benchmarks, the paper notes this distinction explicitly rather than claiming "zero custom kernels."

---

## Phase 0: v1.1 Stability Work (parallel to Phase 1)

Physics-body changes inside existing function boundaries — no API contract changes. Feed the Phase 2 cross-validation test systems and inform the Phase 1 field audit.

| Task | Goal | Risk | Phase 2 connection |
|---|---|---|---|
| LFMiddle hypothesis | Test O-step splitting; may lift dt ≤ 0.5 fs cap | Exploratory; outcome still informative either way | Informs NVE dt range in P2 benchmark |
| NPT KE init bug diagnosis | Diagnose T ≈ 5000 K at first NPT record; likely bad momentum at NVT→NPT handoff | May escalate to full CSVR redesign | **Hard gate on P2 NPT cross-validation tests** |
| Large-scale SETTLE batching | Expand 4-water smoke test → 64-water 10 ps trajectory | Low — architecture validated at small scale | Validates NVT temperature stability test system |

**NPT KE bug → P2 NPT tests**: The NPT temperature spike at step 0 means NPT parity tests cannot be trusted until the bug is fixed. This is a hard prerequisite edge in the DAG, not parallel work.

**P0 vs P1 interaction**: Any NVT→NPT handoff code touched by P0 NPT-KE bug fix should be re-validated after the P1b `IntegratorState`/`IntegratorConfig` split. The split may expose the same bug via different code paths.

---

## Phase 1: Protocol Contract Refactor

**Goal:** Replace `PhysicsSystem | dict` ambiguity and `IntegratorState` Optional fields with a typed, flat contract at the JIT boundary. This unblocks Phase 2 (clean test harness), Phase 4 (flat WASM signatures), and the `xtrax` extraction later.

**Dependency:** Phase 2a (OpenMM parity harness against current `PhysicsSystem`, run in parallel) empirically identifies which fields `MolecularBundle` must carry. Do not freeze the bundle field list before Phase 2a closes the field audit.

### 1a. `MolecularBundle` (eqx.Module)

Strict host-to-device boundary. All Optional fields resolved by the host before entering JIT. Topology arrays are **dynamic** — padded to bucket boundaries, not `static=True`.

**Why dynamic topology:** `eqx.field(static=True)` makes array content part of the JIT cache key — each distinct topology recompiles (minutes for PME graphs). The paper benchmarks sweep {1k, 5k, 20k, 50k} atoms across multiple proteins; static topology multiplies compile cost by the number of distinct topologies. NumPy arrays as static fields are also fragile (hashed by `bytes`). Use bucketed dynamic shapes instead, matching the `LENGTH_BUCKETS` philosophy in `make_energy_fn_pure._sba`.

**Bucket sizes (to be validated against real system inventory):**

```python
ATOM_BUCKETS    = (256, 1_024, 5_000, 25_000, 60_000)
BOND_BUCKETS    = (256, 1_024, 5_000, 25_000)
ANGLE_BUCKETS   = (256, 1_024, 5_000, 25_000)
DIHEDRAL_BUCKETS = (512, 2_048, 10_000, 50_000)
WATER_BUCKETS   = (16, 128, 1_024, 8_000)
EXCL_BUCKETS    = (512, 2_048, 10_000, 50_000)
```

A static `ShapeSpec` descriptor holds only the bucketed shapes (hashable scalars) — one compile per bucket combination, not per protein.

```python
@dataclass(frozen=True)
class MolecularShapeSpec:
    n_atoms: int
    n_bonds: int
    n_angles: int
    n_dihedrals: int
    n_impropers: int
    n_urey_bradley: int
    n_waters: int
    n_excl: int
    n_cmap: int
    n_exception_pairs: int
    has_pbc: bool
    has_implicit_solvent: bool
```

```python
class MolecularBundle(eqx.Module):
    # Dynamic arrays — positions and per-atom fields
    positions:       Float[Array, "N 3"]
    charges:         Float[Array, "N"]
    sigmas:          Float[Array, "N"]
    epsilons:        Float[Array, "N"]
    radii:           Float[Array, "N"]           # mbondi2 for GB/implicit solvent
    scaled_radii:    Float[Array, "N"]
    atom_mask:       Bool[Array, "N"]            # False for padding atoms

    # Box: zero array when has_pbc=False (source of truth from ShapeSpec)
    box:             Float[Array, "3 3"]

    # Bonded topology — dynamic, padded to bucket sizes
    bond_idx:        Int[Array, "B 2"]
    bond_params:     Float[Array, "B 2"]         # k, r0
    bond_mask:       Bool[Array, "B"]

    angle_idx:       Int[Array, "A 3"]
    angle_params:    Float[Array, "A 2"]         # k, theta0
    angle_mask:      Bool[Array, "A"]

    dihedral_idx:    Int[Array, "D 4"]
    dihedral_params: Float[Array, "D 4"]         # k, n, delta, (periodicity flag)
    dihedral_mask:   Bool[Array, "D"]

    improper_idx:    Int[Array, "I 4"]
    improper_params: Float[Array, "I 3"]         # k, n, delta
    improper_mask:   Bool[Array, "I"]
    improper_is_periodic: Bool[Array, ""]        # selects harmonic vs periodic branch

    urey_bradley_idx:    Int[Array, "U 3"]       # (i, j=unused, k) — 1-3 pair
    urey_bradley_params: Float[Array, "U 2"]     # k_ub, r_ub
    urey_bradley_mask:   Bool[Array, "U"]

    cmap_torsion_idx:    Int[Array, "CM 8"]      # two consecutive phi/psi dihedral quads
    cmap_energy_grids:   Float[Array, "CM G G"]  # G×G bicubic grid per CMAP term
    cmap_mask:           Bool[Array, "CM"]

    water_indices:   Int[Array, "W 3"]           # (O, H1, H2) per water molecule
    water_mask:      Bool[Array, "W"]

    # Nonbonded exclusions and 1-4 exceptions
    excl_indices:    Int[Array, "E 2"]
    excl_scales_vdw:  Float[Array, "E"]
    excl_scales_elec: Float[Array, "E"]
    excl_mask:       Bool[Array, "E"]

    exception_pairs:       Int[Array, "X 2"]
    exception_sigmas:      Float[Array, "X"]
    exception_epsilons:    Float[Array, "X"]
    exception_chargeprods: Float[Array, "X"]
    exception_mask:        Bool[Array, "X"]

    # PME / nonbonded config
    pme_alpha:         Float[Array, ""]
    cutoff_distance:   Float[Array, ""]

    # Static shape descriptor (hashable scalars only — JIT cache key)
    shape_spec: MolecularShapeSpec = eqx.field(static=True)
```

**Field audit against `system.py`:** The field list above is derived from `make_energy_fn` at `system.py:49–151`. Any field still missing after Phase 2a closes should be added before Phase 1 is declared complete.

**`displacement_fn` placement:** `EnergyFn(bundle) -> scalar` cannot capture `displacement_fn` in a factory closure and remain `jax.export`-compatible (captured callables do not serialize). Options:
1. `displacement_fn` as a `MolecularShapeSpec.boundary_condition: Literal["free", "periodic"]` flag — factory reconstructs it at lowering time from the static spec.
2. Hard-code two bundle variants: `PeriodicBundle` and `FreeBundle` — separate export artifacts.

**Decision: Option 1** — `MolecularShapeSpec` carries `boundary_condition`; factory reconstructs `displacement_fn` from it before entering JIT. This keeps `EnergyFn(bundle) -> scalar` clean.

### 1b. `IntegratorState` + `IntegratorConfig` (eqx.Module)

The existing `step_system.py` correctly separates `IntegratorState` (dynamic) from step config (static). Phase 1b codifies this boundary instead of collapsing it.

**`IntegratorState`** — dynamic, carried through JIT steps:

```python
class IntegratorState(eqx.Module):
    positions: Float[Array, "N 3"]
    momenta:   Float[Array, "N 3"]
    forces:    Float[Array, "N 3"]
    key:       PRNGKeyArray
    box:       Float[Array, "3 3"]          # zero array when no PBC
    # Thermostat state — only live fields (use thermostat subclass; see below)
```

**Thermostat subclasses** — mirror existing `O_Step`/`CSVR_Step`/`NHC_Step` polymorphism:

```python
class LangevinState(IntegratorState): ...   # no extra fields
class CSVRState(IntegratorState):
    csvr_ke_half: Float[Array, ""]          # running KE half-step accumulator
class NHCState(IntegratorState):
    nhc_xi:  Float[Array, "M"]
    nhc_vxi: Float[Array, "M"]
```

Each subclass carries only its live fields. `IntegratorFn` is satisfied structurally regardless of thermostat.

**`IntegratorConfig`** — static (compile-time):

```python
@dataclass(frozen=True)
class IntegratorConfig:
    thermostat: Literal["langevin", "csvr", "nhc"]
    has_pbc: bool
    dt: float
    kT: float
    gamma: float
    n_nhc_chains: int = 0
```

**`export_langevin_step` unblocked:** `IntegratorState` has no Optional fields; flat signature is `jax.export`-compatible. `LangevinState` carries only positions/momenta/forces/key/box — serialization is clean.

### 1c. Protocols

```python
@runtime_checkable
class EnergyFn(Protocol):
    """Callable: (bundle: MolecularBundle) -> Float scalar. displacement_fn reconstructed from bundle.shape_spec."""
    def __call__(self, bundle: MolecularBundle) -> Float[Array, ""]: ...

@runtime_checkable
class IntegratorFn(Protocol):
    """Callable: (state: IntegratorState) -> IntegratorState. Flat — jax.export compatible."""
    def __call__(self, state: IntegratorState) -> IntegratorState: ...
```

**`SystemProtocol` is dropped.** It was an extra indirection over `MolecularBundle` with no concrete use case in the current scope. If coarse-grained or implicit-only variants are added in a future phase, a protocol seam can be introduced then. YAGNI.

### 1d. `ShimMode` — custom_jvp Analytical Force Shim

When end-to-end differentiability is not needed (production MD, equilibration), the energy function registers a `@jax.custom_jvp` rule that computes forces from analytical expressions rather than AD. This eliminates the AD trace overhead for terms where analytical gradient is simpler than the traced one.

```python
class ShimMode(enum.Enum):
    AUTOGRAD   = "autograd"    # Full JAX AD — required for differentiable MD, force matching
    ANALYTICAL = "analytical"  # custom_jvp with hand-coded force expressions — faster production MD
```

**Trade-offs and scope restrictions:**

- `ANALYTICAL` supports **bonded terms only** in Phase 1: bonds, angles, dihedrals, impropers, Urey-Bradley. LJ and PME are excluded from the shim because `lj_forces_dense` uses a dense N² broadcast (`r[:,None,:] - r[None,:,:]`) that allocates O(N² · 3 · 4 bytes) ≈ 30 GB at N=50k — this OOMs on the same systems Phase 3 benchmarks. LJ/PME remain AD-traced in both modes. This restriction is documented as a known Phase 1 limitation; tiled analytical LJ/PME paths are a Phase 3 extension candidate.
- `ANALYTICAL` mode supports 1st-order derivatives only (forces). 2nd-order derivatives (Hessians) and differentiating through a trajectory require `AUTOGRAD`.
- `ShimMode` is an **in-process optimization only**. It has no interaction with `jax.export`/StableHLO: `custom_jvp` rules are consumed during tracing; only primal HLO lands in StableHLO. The shim does not optimize the WASM artifact — the WASM path uses whichever HLO the primal trace produces.
- `AUTOGRAD` and `ANALYTICAL` modes produce **different StableHLO artifacts** when the gradient is exported (the JVP rule is inlined during tracing). The IREE-WASM export in Phase 4 picks one mode; the spec defaults to `AUTOGRAD` for WASM to preserve correctness.

```python
def make_energy_fn(
    bundle: MolecularBundle,
    shim_mode: ShimMode = ShimMode.AUTOGRAD,
) -> EnergyFn: ...
```

In `ANALYTICAL` mode:

```python
@jax.custom_jvp
def energy(bundle):
    return _energy_impl(bundle)   # same energy value

@energy.defjvp
def energy_jvp(primals, tangents):
    (bundle,) = primals
    (bundle_dot,) = tangents
    e = energy(bundle)
    # analytical_forces covers bonded terms only; nonbonded use AD via _energy_impl
    f_bonded = -analytical_forces_bonded(bundle)
    positions_dot = bundle_dot.positions
    return e, jnp.sum(f_bonded * positions_dot)
```

**Phase 2 validation requirement:** Shim agreement test must be validated **per-term** (bonds, angles, dihedrals separately) not only on total force — per-term analytical bugs can cancel in total force comparisons.

**`analytical_forces.py` expansion:** Phase 1 extends the module to cover all bonded terms required for the shim: bonds, angles, proper dihedrals, impropers, Urey-Bradley.

### 1e. `AtomAxisSpec` + `BatchPlan` Tiling Layer

Vendored from prxteinmpnn's `tiling/planner.py` into `prolix/tiling/`. The vendored copy is pinned at the SHA at time of vendoring and kept minimal — extraction to `xtrax` is deferred but the boundary is explicit.

```python
N_ATOMS = AxisSpec(
    name="n_atoms",
    axis_index=0,
    cardinality=50000,
    default_batch_size=0,       # vmap when homogeneous
    tile_granularity=128,       # Pallas kernel alignment
    heterogeneous=False,
)

N_SYSTEMS = AxisSpec(
    name="n_systems",
    axis_index=1,
    cardinality=64,
    default_batch_size=1,
    tile_granularity=1,
    heterogeneous=True,         # lengths vary; safe_map required
)
```

`BatchPlanner` decides vmap-vs-safe_map per axis given a memory budget. `N_SYSTEMS` heterogeneous batching (safe_map over systems with different bucket sizes) is the key differentiator vs kUPS.

**Exit criterion — Phase 1 complete:** `make_energy_fn(MolecularBundle(...)) -> EnergyFn` passes a type-checker; `export_langevin_step(LangevinState)` runs without `NotImplementedError`; Phase 2a parity harness closes (all required fields identified and present in bundle).

---

## Phase 2: Cross-Validation Suite

**Goal:** Establish correctness as a precondition for any performance or positioning claim.

**Structure:**
- **Phase 2a** (parallel to Phase 1): Stand up OpenMM parity harness against the *current* `PhysicsSystem`. This empirically identifies which fields `MolecularBundle` must carry. Results feed back into Phase 1a field list.
- **Phase 2b** (after Phase 1): Migrate harness to `MolecularBundle`; complete full test matrix.

### Phase 2a — OpenMM Parity Harness (current PhysicsSystem)

Load same `.pdb` + `.xml` force field into both OpenMM and prolix via `PhysicsSystem`. Compare energy per term for solvated alanine dipeptide. Record which terms differ and which fields were missing or incorrectly initialized. Close as a field-audit deliverable for Phase 1a.

| Test | System | Tolerance |
|---|---|---|
| Single-point energy per term (bonds, angles, dihedrals, LJ, PME) | Solvated alanine dipeptide | < 0.05 kcal/mol per term |
| Force parity `dE/dr` | Solvated alanine dipeptide | < 0.01 kcal/mol/Å RMS |
| Field-audit report | — | All fields in `system.py:49–151` mapped to bundle entries |

### Phase 2b — Full Cross-Validation (MolecularBundle)

| Test | System | Tolerance |
|---|---|---|
| Single-point energy per term | Solvated alanine dipeptide | < 0.05 kcal/mol per term |
| Force parity `dE/dr` | Solvated alanine dipeptide | < 0.01 kcal/mol/Å RMS |
| NVE energy conservation | Same | < 1% drift over 1000 steps at dt=0.5 fs |
| NVT temperature stability | 216-water box | ±5 K over 5000 steps |
| NPT cross-validation | 216-water box | pressure ±20 bar, T ±5 K; **requires P0-npt-ke-bug fix** |
| `ANALYTICAL` vs `AUTOGRAD` force agreement (bonded terms) | Same | < 1e-6 kcal/mol/Å per term |

### kUPS Cross-Validation (expand existing)

Expand `kups_adapter.py` tests to three system sizes: 100, 500, 2000 atoms across AMBER14 force field. Validates that correctness is not system-size-specific.

**Exit criterion — Phase 2 complete:** All Phase 2b tests pass; ANALYTICAL/AUTOGRAD per-term agreement confirmed; kUPS parity confirmed at all three system sizes.

---

## Phase 3: Benchmark Suite

**Goal:** Quantify the three differentiators. Builds on Phase 2 test harness (same systems, same force fields — measure time not values).

**Depends on:** Phase 2b.

### vs OpenMM

| Metric | Systems | Notes |
|---|---|---|
| Throughput (ns/day) | DHFR 23,558 atoms on H100 | OpenMM uses custom CUDA; prolix uses XLA |
| Gradient throughput | Same | `jax.grad(energy)` — prolix unique; OpenMM requires finite diff |
| Memory scaling | N = {1k, 5k, 20k, 50k} atoms | Tiling should show sub-quadratic HBM growth |
| `AUTOGRAD` vs `ANALYTICAL` throughput | Solvated dipeptide | Quantifies shim speedup for bonded-dominated runs |

### vs kUPS

| Metric | Setup | Notes |
|---|---|---|
| Heterogeneous batch throughput | M = {4, 16, 64} systems, varying size | kUPS requires same-size; prolix uses N_SYSTEMS axis safe_map |
| E2E gradient batch | Same | `jax.jacrev` across batch |

**Exit criterion — Phase 3 complete:** Benchmark numbers reproducible on cluster; memory scaling plot shows expected bucket-step behavior; hetero-batch advantage vs kUPS quantified.

---

## Phase 4: WASM / Browser Deployment

**Goal:** Browser-deployable MD that runs without server-side compute. Proof-of-concept for "no custom kernels" claim.

**Depends on:** Phase 1b (flat `IntegratorState`/`LangevinState` → `export_langevin_step` implementable).

**Gate before committing P4 resources:** Validate that threefry PRNG and the PME/LJ HLO primitives used in `LangevinState` can be compiled to WASM by IREE. Run a 100-step smoke export of `export_langevin_step` and confirm the `.wasm` module runs in a browser before Phase 4 begins. Fallback: `jax.random.key_impl="rbg"` if threefry has coverage gaps.

### Sub-track A: IREE-WASM (primary)

1. Implement `export_langevin_step` — unblocked by `LangevinState` flat signature.
2. Lower energy fn + langevin step via `jax.export` → StableHLO artifacts. Use `ShimMode.AUTOGRAD` for correctness; `ANALYTICAL` does not produce a different WASM artifact.
3. Compile with `iree-compile --target=wasm-bare-minimum` → `.wasm` module.
4. Static demo page: load WASM, run 100 steps of solvated alanine dipeptide, render energy/temperature trace in `<canvas>`.
5. Embed compiled `.wasm` + serialized `MolecularBundle` parameters as static assets.

### Sub-track B: Experimental JAX WebGPU (stretch)

Use JAX's experimental WASM/WebGPU build for GPU-in-browser execution. Higher ceiling, less stable. Runs in parallel with Sub-track A; not a gate on Phase 5.

**Exit criterion — Phase 4 complete:** A static HTML page (no server) runs 100 MD steps in a browser; energy/temperature trace renders correctly; `.wasm` artifact < some target size (TBD after smoke export).

---

## Phase 5: Paper / Preprint

**Depends on:** Phases 2b, 3, 4.

### Narrative structure

1. **Motivation:** MD is differentiable in principle but not in practice — OpenMM requires custom CUDA for accelerators and cannot backprop through trajectories.
2. **Prolix design:** E2E differentiable MD in pure JAX; XLA-native tiling for memory efficiency; `custom_jvp` shim for production mode (bonded terms); browser-deployable via IREE-WASM.
3. **Correctness:** OpenMM parity on energy/forces (Phase 2); kUPS cross-validation.
4. **Performance:** Gradient throughput benchmark; heterogeneous batch benchmark (Phase 3); `AUTOGRAD` vs `ANALYTICAL` mode comparison for bonded-dominated systems.
5. **Application experiment:** One real use-case exploiting differentiability — e.g., differentiable force matching (optimize force field parameters to match ab initio forces) or free energy gradient estimation via backpropagation through a short NVT trajectory.
6. **Demo:** Browser figure from Phase 4.

**Exit criterion — Phase 5 complete:** Preprint submitted; WASM demo URL live; code open-sourced with Phase 1–4 tests passing in CI.

---

## Full DAG

```
[P0-lfmiddle]        ─────────────────────────────────────────────────────┐
[P0-npt-ke-bug]      ────► [P2b-NPT-tests] (hard prerequisite)           │
[P0-settle-batch]    ─────────────────────────────────────────────────────┤
                                                                           │
[P2a: OpenMM harness on current PhysicsSystem]  ──────────────────────────┤
  │ (field audit feeds back into P1a MolecularBundle field list)           │
  ▼                                                                        │
P1a: MolecularBundle (bucketed dynamic topology)                          │
P1b: IntegratorState / IntegratorConfig / thermostat subclasses           │
P1c: EnergyFn / IntegratorFn Protocols                                    │
P1d: ShimMode + custom_jvp shim (bonded terms) + analytical_forces expand │
P1e: AtomAxisSpec + BatchPlan tiling layer                                 │
  │                                                                        │
  ├──────────────────────────────────────────────────► P4a: IREE-WASM      │
  │                                                    P4b: JAX WebGPU     │
  ▼                                                                        │
P2b: Cross-validation (OpenMM parity + kUPS expand + shim per-term) ◄─────┘
  │
  ▼
P3: Benchmarks (throughput, gradient, memory, AUTOGRAD vs ANALYTICAL)
  │
  ├──────────────────────────────────────────────────► P5: Paper / preprint
  ▲                                                         ▲
P4a: WASM demo (browser figure) ─────────────────────────────┘
```

---

## xtrax Extraction Note

`prolix/tiling/` (Phase 1e) is the seed for the future `xtrax` library. The vendored copy is pinned at the SHA of prxteinmpnn's `tiling/planner.py` at time of vendoring. Keep it minimal and bounded so extraction is mechanical. Do not add prolix-specific logic into the vendored files — use a thin `prolix/tiling/axes.py` wrapper for `N_ATOMS`/`N_SYSTEMS` instead.

---

## Open Questions (Resolved)

1. **Topology fields static vs dynamic:** Use bucketed dynamic arrays + static `MolecularShapeSpec`. Do NOT use `eqx.field(static=True)` for array fields.

2. **Thermostat dispatch mechanism:** Use eqx.Module subclasses per thermostat (`LangevinState`, `CSVRState`, `NHCState`) mirroring existing `Step` subclass polymorphism in `step_system.py`. Not a `thermostat_mode: str` string — that pays both costs (wasted memory + recompile per mode).

3. **`custom_jvp` × `jax.export`:** Clean — rules are consumed at tracing time, only primal HLO lands in StableHLO. `ShimMode.ANALYTICAL` vs `AUTOGRAD` produce different StableHLO when the gradient is exported. WASM path defaults to `AUTOGRAD`.

4. **`PRNGKeyArray` in WASM:** Keys are `uint32[2]` (threefry) — trivial to serialize, but threefry coverage in IREE-WASM must be confirmed by a smoke export gate before Phase 4 commits.

5. **`SystemProtocol`:** Dropped (YAGNI — one indirection layer too many given `MolecularBundle` is already an `eqx.Module`).

6. **`displacement_fn` placement:** Reconstructed from `MolecularShapeSpec.boundary_condition` flag inside the factory, not captured in a closure. Keeps `EnergyFn(bundle) -> scalar` serializable.

7. **Phase ordering:** Phase 2a (parity harness on current `PhysicsSystem`) runs in parallel with Phase 1 and feeds back into the bundle field list. Phase 1 is not declared complete until Phase 2a closes.

8. **`ShimMode.ANALYTICAL` scope:** Bonded terms only in Phase 1. Dense-N² `lj_forces_dense` OOMs at benchmark sizes; LJ/PME remain AD-traced. Tiled analytical nonbonded is a Phase 3 extension.

9. **`make_energy_fn_pure` feature gap:** The pure path (`system.py:253–349`, used for export) is missing CMAP, exceptions, GB-aware electrostatics, and dispersion tail. Phase 1 must not silently consolidate on the pure path's reduced feature set — the new `MolecularBundle` factory must reach feature parity with `make_energy_fn`.
