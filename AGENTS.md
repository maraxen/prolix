# Prolix — Project Instructions

Prolix is a JAX and Pallas GPU molecular dynamics engine with a drop-in OpenMM backend (`prolix.openmm`), multi-replica heterogeneous/ragged batch execution (`EnsemblePlan`, `ProlixBatchSimulation`), analytical `@jax.custom_vjp` coordinate/parameter differentiation, and portable StableHLO/WASM export (`prolix.export`).

---

## Import Architecture & Leaf Invariants

Enforced by `tests/test_import_architecture.py` (`scripts/ci/import_graph.py`, pure AST, zero GPU needed):

1. **Zero module-level `prolix.*` cycles** and **zero unallowlisted function-level `prolix.*` lazy imports**. `ALLOWED_CYCLE_DODGES` and `ALLOWED_PHYSICS_LAZY` only ever shrink. When two modules share a type or helper, extract a leaf module instead of deferring an import inside a function.
2. **Strict zero-`prolix`-dependency leaves**: `prolix.typing`, `prolix.constants`, `prolix.openmm.config`, `prolix.openmm.exceptions`, `prolix.openmm.units`, `prolix.physics.water_models`, `prolix.physics.electrostatic_methods`, `prolix.physics.constraint_projection`, and `prolix.tiling.plan_types` (`prolix.knob_surface` and `prolix.physics.step_base` may import only `prolix.typing`).
3. **`prolix.openmm` PEP 562 boundary**: `src/prolix/openmm/__init__.py` resolves `context`, `simulation`, `ProlixContext`, `ProlixSimulation`, and `ProlixBatchSimulation` lazily via `__getattr__` so `prolix.openmm.converter` and `prolix.openmm.config` import cleanly at module top level in `prolix.api.ensemble_plan` without runtime circular imports.

---

## Unit Systems & Potential Conventions

| Boundary | Length | Time | Energy | Force | Pressure | Friction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prolix internal (AKMA)** | `Å` | `τ_AKMA = 48.88821291839 fs` | `kcal/mol` | `kcal/(mol·Å)` | `kcal/(mol·Å³)` (`69,476.95 bar` / `68,568.42 atm`) | `τ_AKMA⁻¹` |
| **`EnsemblePlan.run` kwargs** | `Å` | `fs` (`dt_unit="fs"` default) | `kcal/mol` (`kT`) | `kcal/(mol·Å)` | `bar` / `atm` | `ps⁻¹` (`gamma`) |
| **`prolix.openmm` (`openmm.unit`)** | `nm` (`×10 → Å`) | `ps` (`×1000/48.88821291839 → τ_AKMA`) | `kJ/mol` (`/4.184 → kcal/mol`) | `kJ/(mol·nm)` (`/41.84`) | `bar` (`/69477.0`) | `ps⁻¹` (`×0.0488882129`) |
| **`kups_adapter` (kUPS)** | `Å` | `τ_kUPS ≈ 10.18050571 fs` | `eV` (`×23.060549 → kcal/mol`) | `eV/Å` (`×23.060549`) | — | `τ_kUPS⁻¹` |

- **Electrostatic & GBSA constants**: Direct-space (`pallas_kernels.py`, `explicit_corrections.py`) and reciprocal-space (`pme.py`, `pme_pallas.py`) must share exact `COULOMB_CONSTANT = 332.06371329919216` (`kcal·Å/(mol·e²)`) so 1-2/1-3/1-4 $\operatorname{erf}(\alpha r)/r$ terms cancel to `0.00 ppm`; ACE nonpolar SASA uses $\sigma_{\text{np}} = 0.0054\text{ kcal/(mol}\cdot\text{Å}^2)$ (`2.25936 kJ/(mol·nm²)`).
- **Ewald / SPME `alpha`**: Prolix stores `pme_alpha` in `Å⁻¹`; OpenMM `NonbondedForce.setPMEParameters` expects `nm⁻¹` (`alpha_nm = 10.0 * alpha_A`).
- **Harmonic bonds & angles**: Both OpenMM and Prolix use $E = \tfrac{1}{2} k (x - x_0)^2$ (`bond_k`: `/418.4`; `angle_k`: `/4.184`).
- **Harmonic impropers**: OpenMM uses $E = \tfrac{1}{2} k_{\text{omm}} (\psi - \psi_0)^2$ (`kJ/mol`), whereas Prolix uses $E = k_{\text{prolix}} (\psi - \psi_0)^2$ (`kcal/mol`), so `k_prolix = 0.5 * k_omm / 4.184` (`IMPROPER_K_OMM_TO_AKMA` in `prolix.openmm.units`).
- **Degrees of freedom (`N_dof`)**: Always compute $N_{\text{dof}} = 3 N_{\text{real}} - N_{\text{constraints}} - (3 \text{ if remove\_com else } 0)$ from masked real atoms (`atom_mask`), never from padded bucket capacity; padding atoms carry unit mass (`masses_with_prefix`) and otherwise corrupt kinetic temperature (`Temperature(dof=..., atom_mask=...)`).

---

## Differentiability & StableHLO Export Guardrails

Enforced by `tests/test_stablehlo_export_guardrails.py`:

1. **Analytical `jax.grad` / `@jax.custom_vjp`**: Every energy and Pallas force kernel (bonded, Morton $32 \times 32$ tile-pair direct nonbonded, SPME spread/gather, 2-pass tile OBC2 GBSA, virtual sites) must support `jax.grad` with respect to coordinates and force-field parameters (`charges`, `sigmas`, `epsilons`, GBSA radii, bonded parameters) in $O(N \cdot K)$ memory without materializing dense `(N, N)` matrices.
2. **Portable StableHLO lowering (`prolix.export`)**: Every Pallas/Triton entry point must inspect `prolix.export.in_export_context()` (via `resolve_pallas_mode`) and fall back to pure JAX when lowering inside `with prolix.export.export_context():`. Lowered StableHLO (`export_energy_fn`, `jax.export.export`) must contain **zero** `triton`, `mosaic`, `pallas`, `tpu_custom_call`, `xla.gpu`, `lapack`, `cusolver`, `cublas`, or `cudnn` `custom_call` targets (`WASM_ARTIFACT_MAX_BYTES = 50 MiB`).
3. **Explicit-parameter export**: `make_energy_fn` closes over force-field arrays; use `make_energy_fn_pure` returning `(DifferentiableParams, fn)` whenever lowering via `export_energy_fn`.

---

## Knob Governance (`KnobSurface` & `redsox` U1/U5)

Enforced by `src/prolix/governance/redsox_adapter.py` and `tests/test_redsox_openmm_knob_governance.py`:

- **`KnobSurface` (`src/prolix/knob_surface.py`)** is the single zero-cycle composite configuration schema across all 9 control sub-surfaces (`thermostat`, `barostat`, `nonbonded`, `implicit_solvent`, `constraints`, `cosolvent`, `softcore_restraints`, `cv_bias`, `pallas`).
- **U5 Superset (`check_superset`)**: Every field on `ProlixOpenMMConfig`, `NonbondedConfig`, `NeighborListConfig`, and supported OpenMM `Platform`/`System`/`Integrator` properties must resolve to a `KnobSurface` leaf (directly or via `SUPERSET_ALIAS_MAP`) with **0 missing knobs**.
- **U1 Consumer Reachability (`check_prolix_reachability`)**: Every non-inert `KnobSurface` leaf must be AST-read inside runtime consumer modules (`src/prolix/physics`, `src/prolix/api`, `src/prolix/openmm` excluding declaration-only `config.py`, `src/prolix/pallas_kernels.py`, `src/prolix/batched_energy.py`, `src/prolix/batched_simulate.py`), with `PENDING_EPIC_KNOBS == frozenset()`. Never introduce dead or unrouted configuration fields.

---

## Pallas v3.5 Kernel & Integrator Rules

- **Explicit-solvent default (`nonbonded_engine="auto"` / `"morton_tile_pairs"`)**: Routes orthorhombic periodic systems through Morton-sorted $32 \times 32$ Newton-pair tiles (`pallas_tile_pairs.py`), $O(N_{\text{tiles}})$ GPU Morton AABB skin-rebuild, smooth $2^a 3^b 5^c$ SPME (`snap_fft_dim`, fused gather `force_scale`), and inter-step fused persistent BAOAB + analytical SETTLE + RATTLE + COM-removal across `nl_update_every` chunks. Triclinic boxes (`reduce_triclinic_box_vectors`) or CPU/export contexts fall back cleanly to `"neighbor_list"` or pure JAX.
- **Implicit-solvent default (`pallas_gbsa_tile.py`)**: Two-pass $32 \times 32$ symmetric tile-pair OBC2 Generalized Born + ACE nonpolar SASA (`dielectric_offset = 0.09 Å`) with analytical Born-radius chain-rule forces and `@jax.custom_vjp`.
- **Multi-replica & ragged batching**: `EnsemblePlan` and `ProlixBatchSimulation` deduplicate replica-invariant topology index/mask tables at `stride = 0` for congeneric ligand/mutant series (`~45 MB/replica` marginal VRAM vs `~450–580 MB/replica` under `CUDA MPS`) and flatten cross-size buckets onto a 1D prefix-sum ragged CTA work queue (`pallas_ragged_queue.py`) with cost-modeled multi-tier PME (`pme_tier_mode="auto" | "grouped" | "uniform"`).
- **Precision modes**: `"single"` (`float32` state and forces), `"mixed"` (`float64` coordinate/momentum/SETTLE/RATTLE accumulators + pre-cast `float32` Pallas force/PME kernels and `float32` Philox-2x32 thermal deviates), `"double"` (`jax_enable_x64=True` `float64` reference path).
- **Triton lowering traps**:
  1. `resolve_pallas_compiler_params("megakernel")` must keep `num_warps=4, num_stages=1` across all GPUs (and `num_stages=1` on Blackwell `sm_120` `cc >= 10.0`) to prevent per-thread `.local` stack overflow (`CUDA_ERROR_ILLEGAL_ADDRESS`).
  2. Never apply unary `-` directly to a `jnp.asarray(c, dtype=...)` constant inside a Pallas Triton kernel (`TypedNdArray.type` crash in `lax.neg_p`); negate the Python float first (`jnp.asarray(-c, dtype=...)`).
  3. Never wrap an `input_output_aliases` buffer in `jnp.pad` or slice it on GPU (`not interpret`); use masked loads/stores (`offs < N`) so `76.8 MB/rep` neighbor tables alias in place with zero HBM copies.
- **Timestep & initialization bounds**: With `HBonds` + `SETTLE` (`ProlixSimulation` / `EnsemblePlan`), `dt = 2.0 fs` is standard. For unconstrained-solute + rigid-TIP3P pure-JAX `settle_langevin`, keep `dt <= 1.0 fs` at `n_waters >= 16, gamma >= 10 ps^-1` (`dt <= 0.5 fs` at weak friction `gamma ~ 1 ps^-1` or `n_waters < 16`). Legacy `batched_equilibrate` is deprecated; use `EnsemblePlan`, `ProlixBatchSimulation`, or cold-start `LangevinState` with evaluated initial forces.

---

## OpenMM CUDA Anti-Fallback Guard

Every OpenMM benchmark or parity reference in `scripts/benchmarks/` and `scripts/validation/` must create its context via `require_openmm_cuda_context` (or `run_openmm_cuda_mps_replicas`) from `scripts/benchmarks/openmm_cuda_guard.py`. It verifies `ctx.getPlatform().getName() == "CUDA"`, mapped `libOpenMMCUDA.so`, and exact `Precision`, stamping all 7 provenance fields into output JSONs and raising `RuntimeError` on any silent `OpenCL`/`CPU`/`Reference` fallback.

---

## Testing, Static Gates & Experiment Tracking (`bathos`)

```bash
python3 scripts/ci/import_graph.py                     # AST cycle + lazy-import gate
python3 scripts/export_regression_pme.py --check       # Verify REGRESSION_EXPLICIT_PME doc sync
uv run ruff check src/                                 # Lint
uv run ty check                                        # Static type check
uv run pytest -m "not slow"                            # Fast CI suite (writes tmp/pytest.json)
sphinx-build -W --keep-going -b html docs/source docs/_build/html  # Docs gate
```

**`bathos` (`bth`) hard rules** for `scripts/experiments/` and `scripts/benchmarks/`:
1. Invoke via `bth run --tags <tag> --campaign-id <id> --output-paths <out.json> -- uv run python <script.py> --out <out.json>` (never raw `uv run python`, never `uv run bth`, never `--no-sidecar` outside `scripts/explore/`).
2. Pair every script with `<stem>.bth.toml` (`[experiment]` schema, DuckDB SQL `condition`s, exactly one `is_residual = true`).
3. Create the campaign (`bth campaign create "<slug>" --mode exploration --question "..."`) before the first run; Slurm wrappers `source scripts/slurm/_bth_env.sh` without pinning `BTH_CATALOG_DIR`; sync cluster runs via `bth sync engaging --pull`.
4. Query tags in `bth sql` with `WHERE list_contains(from_json(tags, '["VARCHAR"]'), 'my-tag')`.

---

## Pointers

- **Adding an intra-package import trips `test_import_architecture.py`** → [import hardening](.praxia/docs/reference/261025_import_architecture_hardening.md)
- **Modifying Pallas kernels, tile pairs, SPME, or `EnsemblePlan` loops** → [megakernel ledger](docs/plans/2026-10-28-xtrax-megakernel-composition-ledger.md)
- **Comparing throughput or VRAM against OpenMM CUDA / `CUDA MPS`** → [v3.4 ladder report](docs/benchmarks/261110_pallas_v34_cuda_mps_ladder_report.md)
- **Editing `REGRESSION_EXPLICIT_PME` or explicit-solvent parity tests** → [OpenMM protocol](docs/source/explicit_solvent/openmm_comparison_protocol.md)
- **Profiling JAX/Pallas sub-steps with `named_scope` or `xtrax.profiling`** → [attribution limits](.praxia/docs/reference/260901_named-scope-attribution-limits.md)
- **Changing unconstrained-solute rigid-TIP3P `dt` or `gamma` defaults** → [dt=1fs crossover](.praxia/docs/research/260612_p5-dt1fs-size-crossover.md)
- **Building or extending ReadTheDocs Sphinx documentation** → [Sphinx config](docs/source/conf.py)
