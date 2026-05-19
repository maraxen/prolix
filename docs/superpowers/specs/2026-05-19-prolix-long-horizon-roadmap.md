# Prolix Long-Horizon Roadmap — Engine Paper + Science Substrate

**Date:** 2026-05-19
**Status:** Approved (post brainstorm + oracle critique)
**Supersedes (does not delete):** `.praxia/specs/2026-05-15-prolix-strategic-roadmap.md` — that document remains the *execution-slice detail* for the items it covers; this document re-cuts those items along claim-axes and extends the horizon past v1.2/paper.
**Brainstorm transcript:** session 2026-05-19 (advisor/oracle-critiqued)

---

## §0 Identity & Thesis

### Project identity (hybrid two-lane)

| Lane | Purpose | Primary deliverable |
|---|---|---|
| **A — Engine** | Differentiable MD engine as a publishable artifact | Paper + WASM demo + open-source library |
| **B — Substrate** | Science substrate for downstream research (MTT log-det electrostatics, allostery-as-circuits, EFA, PT) | Future papers, one per topic |

Lanes share the engine foundation but have distinct success criteria. Lane B is "future work" in the lane-A paper *except* for the single Lane-B figure that gates the paper (see §7.1).

### Engine paper thesis (final)

> Prolix runs ensembles of *short, structurally-varied* trajectories as single SIMD batches under the hood, dominating wall-clock for init-bound workloads (umbrella sampling, FEP intermediates, NEB images, MTT samples), and ships portably via IREE-WASM to environments where heavy MD typically can't run.

**What is and isn't a headline:**
- **Headlines:** heterogeneous batching of varied systems (the differentiator); portability via WASM (the deploy claim).
- **Supporting infrastructure, not headlines:** E2E differentiability (also in torch-md, kUPS, jax-md); XLA-native (table stakes).
- **Floor claim (defensive, not headline):** steady-state throughput matches OpenMM at GPU saturation within 0.7×–1.3×.
- **Pinned regime:** init-bound short-trajectory workloads. The thesis does *not* claim wins on single-system long-trajectory throughput.

---

## §1 Three-Claim Architecture

```
CLAIM 1 — Heterogeneous-batch substrate (★ execution mechanism)
CLAIM 2 — Portability / lightweight deploy (co-headline)
CLAIM 3 — Declarative ensemble API (user-facing surface, ergonomics)
─────────────────────────────────────────────────────────────────
SUPPORTING S1 — Differentiability infra (validation only, not paper claim)
SUPPORTING S2 — Throughput-parity defensive benchmark (not paper claim)
─────────────────────────────────────────────────────────────────
LANE B §7 — Science substrate (one figure HARD-gates paper submission)
```

### Split between Claim 1 and Claim 3 (same code, different claims)

- **Claim 1** owns the *mechanism*: `BatchPlanner` integration, `safe_map`-across-varying-`shape_spec`, bucket policy, bucket sensitivity, hetero-batch SETTLE. Validation answers: *"does the SIMD batch actually work?"*
- **Claim 3** owns the *expressibility*: `MolecularBundle` factories, `EnsemblePlan` API, `Observable` protocol, `Trajectory` shape, tutorial, public API freeze, migration. Validation answers: *"is this easy to use, well-typed, well-documented?"*

The Bundle + EnsemblePlan code is referenced by both, but the claims are independent.

---

## §1.5 Scoping Discipline

### Three concentric rings

| Ring | Gate | Stop rule | Disposition |
|---|---|---|---|
| **Core** | "Without this, the paper either fails review or can't be reproduced." | Claims 1–3 validation matrices closed + §7.1 figure reproducible in <30 min from clean clone + reviewer-rebuttal table green | Stop adding once stop rule fires; write the paper |
| **Capability** | "A lane-B project needs this AND it's <1 week of work, OR a community user has asked for it." | v2.0 planning begins | Snapshot as v1.3; new ambitions become v2.0 candidates |
| **Speculative** | "Strategic bet, investigated when it bubbles up." | None | Backlog graveyard / idea queue, kept alive forever |

Every roadmap item and praxia backlog entry carries a `ring` field. Promotions are tracked decisions:
- **speculative → capability:** lane-B project explicitly needs it AND 1-week spike confirms feasibility
- **capability → core:** reviewer would likely demand it OR Claims 1–3 validation depends on it
- **core → capability (demotion):** validation passes without it AND adding it delays the paper

### Reviewer Rebuttal Table (adversarial complement to rings)

| # | Likely Reviewer-2 objection | Where evidence lives | Ring | Status |
|---|---|---|---|---|
| RR1 | "How is this different from kUPS / OpenMM swarm / GROMACS multi-sim?" | DR-claim1-1; §Claim 1 §Claim 3 comparison narrative | core | TODO |
| RR2 | "Show me a real scientific result, not just an engine" | §7.1 paper-gating figure | core | TODO |
| RR3 | "AOT compile time invalidates the init-bound win" | §Claim 1 B1 protocol: AOT segment line-itemized; cold-start enforced | core | TODO |
| RR4 | "Why these bucket boundaries and not log-uniform / quantile-derived?" | §Claim 1 R3 mitigation: DR-claim1-3 + B4 ±25% sensitivity sweep + configurable per-deployment | core | TODO |
| RR5 | "Throughput at scale loses to OpenMM custom CUDA" | §S2 defensive benchmark; show ≥0.7× at GPU saturation, regime irrelevant to thesis | core | TODO |
| RR6 | "Browser MD is a toy; nothing real runs in WASM" | §Claim 2 W4 + WB1 informational benchmark | core | TODO |
| RR7 | "Differentiability is already done elsewhere (jax-md, torch-md)" | §S1 framed as supporting capability; Claim 1 × S1 multiplicativity in §7.1 figure | core | TODO |

Update the **Status** column as evidence lands. Adding an objection mid-development requires either (a) producing the evidence or (b) explicit acknowledgement of vulnerability in the paper's Limitations section.

---

## §2 Claim 1 — Heterogeneous-batch substrate

**Sub-claim:** *Prolix executes ensembles of varied-size systems as a single SIMD batch via safe_map/vmap dispatch over bucketed dynamic topology, with init+exec wall-clock dominating per-system engines on short-trajectory ensembles.*

### 2.1 Mechanism owned by this claim

- `BatchPlanner` integration at `EnsemblePlan.from_bundles` construction site → `BatchPlan` pre-computed before JIT (current location: `src/prolix/tiling/planner.py`)
- `safe_map` heterogeneous dispatch across varying `MolecularShapeSpec` (current location: needs work — V8 day-1 smoke test gates this)
- Bucket ladder (default; derived from DR-claim1-3 in §8):
  ```
  ATOM_BUCKETS    = (256, 1_024, 5_000, 25_000, 60_000)
  BOND_BUCKETS    = (256, 1_024, 5_000, 25_000)
  ANGLE_BUCKETS   = (256, 1_024, 5_000, 25_000)
  DIHEDRAL_BUCKETS = (512, 2_048, 10_000, 50_000)
  WATER_BUCKETS   = (16, 128, 1_024, 8_000)
  EXCL_BUCKETS    = (512, 2_048, 10_000, 50_000)
  ```
- Bucket boundaries are configurable per-deployment via `MolecularShapeSpec.bucket_overrides`
- Bucket sensitivity guarantee: ±25% boundary shifts produce <10% throughput change on B4 workloads

### 2.2 Validation matrix

| # | Test | System | Tolerance | CI gate? |
|---|---|---|---|---|
| V1 | `EnsemblePlan(B=1).run` parity vs `settle_langevin` | Solvated AKE, 1k steps | RMSD < 1e-12 Å | yes |
| V3 | Homogeneous (same-size) parity vs B independent runs | DHFR ×4 | RMSD < 1e-10 Å | yes |
| **V4** | **Heterogeneous parity** vs N independent runs | {1ake, 1ubq, 2gb1, 4-water} | Per-system RMSD < 1e-10 Å | **yes (headline)** |
| **V4-HLO** | **HLO-fingerprint assertion: hetero-batch run produced a single safe_map (not N traces)** | Same as V4 | `jax.make_jaxpr` count of `safe_map`/`scan` primitives = 1 | **yes** |
| V5 | Observable parity (single vs batched) | Energy, KE, T, RMSD, Pressure | < 1e-6 relative | yes |
| V6 | `jax.grad` finite-diff parity (composes with S1) | Force-matching loss | < 1e-4 RMS | nightly |
| V7 | BatchPlanner decision correctness | Synthetic axis workloads | Plan output = expected vmap/safe_map split | yes |
| **V8** | **Day-1 smoke: safe_map across varying `shape_spec`** | 2 bundles, different bucket | No retrace; `make_jaxpr` trace-count = 1 | **yes (day-1)** |

**V4-HLO note (oracle-added):** per-system RMSD parity alone is consistent with a naive implementation that runs N independent traces under the hood. The HLO-fingerprint assertion is what proves the batched mechanism is actually batching. Implementation: use `jax.make_jaxpr` on the compiled `EnsemblePlan.run`, parse the resulting jaxpr, and assert the count of `safe_map`/`scan` primitives is exactly 1 (or whatever the canonical hetero-dispatch primitive is).

### 2.3 Benchmark matrix

**B1 — Pre-registered protocol (paper headline plot).** Pre-registration is binding; deviations require an explicit spec amendment.

| Field | Pinned value |
|---|---|
| Ensemble | 64 systems × 4 system types × 16 each: `{1ake (~3k atoms), 1ubq (~1.4k), 2gb1 (~560), 4-water cluster (~12)}` |
| Trajectory per system | 100 ps (short-trajectory regime where init dominates) |
| Cold-start | Fresh Python process per run; XLA cache cleared via `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_$(uuidgen)`; OS page cache flushed via `sync && echo 3 > /proc/sys/vm/drop_caches` (or `vm_stat purge` on macOS) |
| Wall-clock segments (reported separately) | `t_ff_load`, `t_bundle_construct`, `t_aot_compile`, `t_first_step`, `t_steady_state`, `t_total` |
| OpenMM baseline | N independent `Context` objects (forced by varying topology), cold-start; FF load + Context creation per system; reuse forbidden |
| XLA flags pinned | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"`, all others default |
| Seeds | 3 minimum |
| Reporting | Median + range across seeds; full CSV at `outputs/bench/b1_init_exec.csv` |
| Hardware | H100 (primary) + RTX 4090 (secondary); cluster runs via bathos |
| Re-run cadence | On any change to `EnsemblePlan`, `BatchPlanner`, `Bundle`, or `safe_map` integration |

**Other benchmarks:**

| # | Metric | Setup |
|---|---|---|
| B3 | Memory scaling | N ∈ {1k, 5k, 20k, 50k} atoms single-system; bucket steps must be visible |
| B4 | Hetero-batch vs naive-pad-to-max | {1k, 5k, 25k} atoms × 4 each; bucket savings over pad-to-max; ±25% sensitivity sweep on bucket boundaries |
| B5 | BatchPlanner decision quality | Sweep memory budgets and ensemble sizes; assert planner picks expected vmap/safe_map split |
| ~~B2~~ | **Moved to §S2 — not a Claim 1 plot** | — |

### 2.4 Exit criteria

- ✅ V1, V3, V4, V4-HLO, V5, V7, V8 green in CI on every PR
- ✅ V6 green in nightly CI
- ✅ B1 reproduced 3-seed on cluster; full CSV in `outputs/bench/b1_init_exec.csv`
- ✅ B3, B4, B5 reproducible; ±25% bucket sensitivity sweep satisfied
- ✅ R1 closed: large-scale (64-water mixed-system) SETTLE batching validated end-to-end (escalated from v1.1 capability → core)

### 2.5 Capability axes touched + ring tagging

| Axis | Item | Ring |
|---|---|---|
| Batching | `BatchPlanner` wiring into `EnsemblePlan.from_bundles` | core |
| Batching | `safe_map`-across-varying-`shape_spec` (V8) | core |
| Batching | Bucket policy tuning from PDB-derived ladder | core |
| Batching | Hetero-batch SETTLE (extend smoke → 64-water mixed) | core |
| Batching | Bucket override per-deployment API | capability |
| Batching | HLO empirical memory estimator (replace theoretical) | speculative |
| Sampling | PT-as-EnsemblePlan composition | capability |
| Integrators | Mixed-integrator ensembles | capability (deferred per user) |

### 2.6 Risks (top 3 inline; full register in §12)

- **R1 — Hetero SETTLE only smoke-tested.** 4-water × 100 steps today; V4 demands 64-water mixed-system parity. **Mitigation:** existing v1.1 backlog item "Large-Scale SETTLE Batching" escalated capability → core; hard prerequisite for V4.
- **R2 — Trajectory in-memory blowup.** T×N×3 floats for long trajectories on large systems exceeds HBM. **Mitigation:** `save_every` + streaming-to-host with optional checkpoint protocol; designed-in to `Trajectory` shape, tested in capability ring (HDF5/Zarr decision sub-spec).
- **R3 — Bucket policy looks ad-hoc to reviewer.** If the reviewer picks system sizes that fall in awkward buckets, B3/B4 plots look worse than they are. **Mitigation:** (a) DR-claim1-3 derives buckets from PDB scan with stated criterion ("95% of PDB chains fit ≤ 5k atoms"); (b) B4 ±25% sensitivity sweep; (c) `MolecularShapeSpec.bucket_overrides` makes buckets configurable per-deployment.

### 2.7 Hard prerequisites (oracle-escalated, locked)

- **HP1 — Migration policy decided** for legacy entry-points (`batched_produce`, `LangevinState`, `pad_protein`, `PaddedSystem`, `collate_batch`) before Claim 1 implementation starts. Sub-spec required. *Not* an open question.
- **HP2 — `make_bundle_from_system` feature parity audit** vs `make_energy_fn` (CMAP, exceptions, GB-aware electrostatics, dispersion tail per existing roadmap Open Question #9). V1 risks silently passing on a degraded force field if this is skipped.
- **HP3 — V8 (safe_map across varying `shape_spec`) green** before any other Claim 1 implementation work. If `shape_spec` is `eqx.field(static=True)` and varies per element, the thesis collapses to per-element retrace.

### 2.8 Key open questions (full list in §13)

1. Mixed-integrator ensembles: deferred to capability ring (locked).
2. Observable scalar vs tensor: protocol returns `Array` (policy = scalar in core, tensor in capability — locked).
3. Trajectory checkpoint format: HDF5 / Zarr / JAX-native pytree — defer to capability sub-spec.
4. Bucket override API ergonomics: positional list vs named dict — defer to Claim 3 sub-spec.

### 2.9 Deep-research pointers

- **DR-claim1-1** — Hetero-batched MD precedent scan (Related Work section)
- **DR-claim1-2** — API-design pattern survey (Dask, sklearn, Equinox)
- **DR-claim1-3** — Bucket-policy literature (PDB/CASP/AlphaFold-DB size distributions)

Full query list in §8.

### 2.10 Cross-claim edges

- **Depends on:** HP1, HP2, HP3 (above); P0 NPT-KE bug fix (if NPT plans included in V1).
- **Unblocks:** Claim 2 (WASM needs flat Plan state schema), Claim 3 (API surface stabilization), S1 (gradient composability), §7.1 Lane-B figure.

---

## §3 Claim 2 — Portability / lightweight deploy

**Sub-claim:** *Prolix exports the full ensemble execution as portable IREE-WASM artifacts that run in a browser with no server compute, demonstrating that intensive MD computations can ship to lightweight environments.*

### 3.1 Scope

- **Primary target:** IREE-WASM via `jax.export` → StableHLO → `iree-compile --target=wasm-bare-minimum`
- **Secondary targets (capability ring, noted in §Speculative):** Apple Silicon native (Metal via IREE), ARM cloud (Lambda), Android/iOS

### 3.2 Validation

| # | Test | CI gate? |
|---|---|---|
| W1 | `jax.export` of single-system `EnsemblePlan(B=1).run` (1k step trajectory) produces valid StableHLO | yes |
| W2 | `jax.export` of `EnsemblePlan.run` (B=4 mixed bundles) produces valid StableHLO | yes |
| W3 | IREE-WASM compile succeeds; `.wasm` artifact < 50 MB target | nightly |
| W4 | Browser smoke run: 100 steps of solvated AKE in static HTML page; energy/temperature trace renders | weekly |

### 3.3 Benchmark / Demo

- **WB1** — Wall-clock for 100-step browser run vs same on H100 (informational only — show feasibility, not racing)
- **WB2** — Cold-start size: `.wasm` load + warm-up < 5 s target

### 3.4 Risks

- **WR1** — threefry PRNG coverage in IREE-WASM may be incomplete. **Mitigation:** smoke export gate before committing Phase 4 implementation; fallback to `jax.random.key_impl="rbg"` if needed.
- **WR2** — PME/LJ HLO primitives may not lower cleanly to WASM. **Mitigation:** smoke gate; fallback to short-range-only for the browser demo if PME fails (documented as known limitation in paper).
- **WR3** — `.wasm` artifact size could exceed practical browser limits. **Mitigation:** WB2 monitors; code-splitting sub-spec if hit.

### 3.5 Exit criteria

- ✅ W1–W4 pass; static demo page reproducible from clean clone
- ✅ WB2 under target (< 5 s cold-start)
- ✅ Demo URL live (capability — post-paper)

### 3.6 Cross-claim edges

- **Depends on:** Claim 1's flat Plan-state schema (Claim 1's state shape *is* the WASM serialization schema).
- **Unblocks:** RR6 evidence; the "browser figure" in the paper's demo section.

---

## §4 Claim 3 — Declarative ensemble API

**Sub-claim:** *Researchers declare a heterogeneous ensemble of systems in a handful of lines using the `MolecularBundle` + `EnsemblePlan` API with composable typed `Observable`s, without manual padding or vmap orchestration.*

### 4.1 API surface (locked, 2-layer)

```python
# Layer 1 — data
bundle = MolecularBundle.from_pdb("1ake.pdb", forcefield="amber14")
# Migration path
bundle = MolecularBundle.from_system_dict(legacy_dict)

# Layer 2 — ensemble execution (B=1 is degenerate ensemble)
ensemble = EnsemblePlan.from_bundles(
    bundles=[bundle_1, bundle_2, ...],
    integrator=settle_langevin(dt=0.5, kT=300*kB, gamma=1.0),
    observables=[Energy(), KineticEnergy(), Temperature(),
                  RMSD(ref=ref_positions), Pressure()],
    batch_budget_bytes=None,  # default to device limit
)
trajectories = ensemble.run(n_steps=10_000, save_every=100, keys=keys)
# trajectories[i] is a Trajectory(positions, momenta?, observables, final_state)

# Gradient composes with S1
loss_grad = ensemble.gradient(
    lambda traj: traj.observables["Energy"][-1],
    wrt="bundles.positions",
)
```

### 4.2 Observable protocol (widened per oracle)

```python
@runtime_checkable
class Observable(Protocol):
    def __call__(self, state: IntegratorState, bundle: MolecularBundle) -> Array: ...
    # Returns Array of any shape. Policy enforced by ring tagging:
    #   Core ring observables ship scalars (Energy, KE, Temperature, RMSD, Pressure)
    #   Capability ring observables return tensors (RDF, RMSF, SASA, ContactMap)
```

The Array return signature is locked now to avoid a v1.3 protocol break. Restricting to scalars happens via ring tagging (a policy, not a signature constraint).

### 4.3 Trajectory shape

```python
class Trajectory(eqx.Module):
    positions: Float[Array, "T N 3"]              # always present
    momenta: Float[Array, "T N 3"] | None         # opt-in for memory
    observables: dict[str, Array]                  # per-step observable evaluations
    final_state: IntegratorState                   # for resume
```

### 4.4 Validation (ergonomics / contract, not numerics — numerics in Claim 1)

| # | Test | CI gate? |
|---|---|---|
| A1 | API contract test: `EnsemblePlan.from_bundles` accepts all registered observable combinations | yes |
| A2 | Tutorial notebook reproduces cold from clean clone in <30 min wall-clock | weekly |
| A3 | All public names typed; `mypy --strict` passes on `prolix.api` module | yes |
| A4 | Migration smoke: each deprecated entry-point has a one-line replacement; CHANGELOG migration table complete | yes |

### 4.5 Exit criteria

- ✅ A1–A4 green
- ✅ `prolix.__init__` re-exports: `MolecularBundle`, `EnsemblePlan`, `Trajectory`, `Energy`, `KineticEnergy`, `Temperature`, `RMSD`, `Pressure`
- ✅ Deprecation warnings on legacy entry-points; removal in v2.0
- ✅ Tutorial notebook at `notebooks/02_ensemble_plan_tutorial.ipynb`
- ✅ One additional notebook showing differentiable workflow at `notebooks/03_differentiable_workflow.ipynb`

### 4.6 Risks

- **AR1** — Migration breaks downstream user code. **Mitigation:** HP1 (locked in §2.7).
- **AR2** — Observable framework feature creep. **Mitigation:** ring tagging enforced; new observables require RR-table entry showing what reviewer concern they address.
- **AR3** — Three-layer instinct returns. The 2-layer choice can feel awkward when documenting "single-system case." **Mitigation:** doc-pattern of `EnsemblePlan.from_bundles([bundle])` for single-system; named in tutorial.

### 4.7 Cross-claim edges

- **Depends on:** Claim 1 mechanism; HP1 migration policy
- **Unblocks:** §7.1 Lane B figure (uses this API end-to-end); user documentation

---

## §5 Supporting S1 — Differentiability infra (validation only)

**Not a paper claim.** Supporting capability that makes Claim 1 × §7.1 figure possible.

### 5.1 Scope

- `ShimMode.AUTOGRAD` vs `ShimMode.ANALYTICAL` (already shipped — Phase 1d, commit `f154266`)
- `jax.grad` / `jax.jacrev` composability through `EnsemblePlan.run`
- Per-term shim validation (bonds, angles, dihedrals, impropers, Urey-Bradley)

### 5.2 Validation

| # | Test |
|---|---|
| D1 | Per-term `ANALYTICAL` vs `AUTOGRAD` force agreement, each bonded term separately (< 1e-6 kcal/mol/Å per term) |
| D2 | `jax.grad` through trajectory finite-diff parity — cross-references V6 |
| D3 | Performance gap: `ANALYTICAL` faster than `AUTOGRAD` for bonded-dominated systems (relative; specific number reported, not gated) |

### 5.3 Exit

- ✅ D1–D3 pass
- ✅ One paper subsection (1–2 paragraphs) documenting `ShimMode` as supporting capability

---

## §6 Supporting S2 — Throughput-parity defensive benchmark

**Not a paper claim — defensive measurement.** Exists to answer RR5.

### 6.1 Goal

Show that steady-state ns/day at GPU saturation falls within 0.7×–1.3× of OpenMM custom-CUDA, so the "we don't beat them but we don't lose" floor holds.

### 6.2 Setup

- **System:** DHFR 23,558 atoms
- **Hardware:** H100 (primary) + RTX 4090 (secondary)
- **Trajectory:** 1 ns after 10 ps warmup; steady-state window only
- **Mode:** `ShimMode.AUTOGRAD` (paper-representative)
- **Reporting:** ns/day median + envelope across 3 seeds

### 6.3 Failure mode (named explicitly)

If we hit <0.5× OpenMM at GPU saturation, the floor claim collapses. **Falsification trigger** escalates to an S2 follow-up sub-spec investigating PME/LJ HLO codegen gap (a multi-week investigation). Paper either delays or adds a Limitations subsection explicitly disclaiming throughput parity at scale.

### 6.4 Exit

- ✅ Reproducible ns/day in 0.7×–1.3× envelope
- ✅ CSV at `outputs/bench/s2_dhfr_h100.csv`, `outputs/bench/s2_dhfr_4090.csv`

---

## §7 Lane B — Science substrate

Lane B has TWO scopes in this roadmap:
1. **§7.1 — Paper-gating figure** (one figure; hard gate on paper submission)
2. **§7.2 — Future-work items** (post-engine-paper; capability/speculative ring)

### §7.1 Paper-gating figure (HARD Phase 5 gate)

**Differentiable bonded-parameter fitting on a heterogeneous ensemble.** Lowest-risk option; combines Claim 1 × S1 multiplicatively.

**Setup:**
- 16 small varied systems (mix of dipeptides, water clusters, sugar units)
- Ground-truth ab initio forces precomputed (DFT or MP2 reference); stored as static dataset
- Objective: optimize bonded force-field parameters (`bond_k`, `bond_r0`, `angle_k`, `angle_θ0`) to minimize `MSE(predicted_forces, reference_forces)` via `jax.grad` through `EnsemblePlan.run(n_steps=0).gradient(...)`
- Baseline: naive loop-over-systems gradient
- Expected story: 50–100× speedup via hetero-batched gradient

**Deliverable:**
- Convergence curves (loss vs wall-clock) for batched-prolix vs baseline
- Final fitted parameters table
- Reproducible from clean clone in <30 min via bathos run

**Why this figure:**
- Most directly demonstrates the thesis (Claim 1 × S1 multiplicativity)
- Lowest risk: bonded-parameter fitting is well-established; we add the batching speedup
- Self-contained dataset (precomputed ab initio forces)
- Easy reviewer story: "scientist needs to fit FF parameters to ab initio reference; this is the workflow"

**Alternative figures (documented; pick if §7.1 underperforms or if we finish faster):**

| ID | Alternative | When to use |
|---|---|---|
| §7.1-alt-A | Batched free-energy gradient via reweighting on a small mutant series | Higher reviewer impact; medium risk; needs reliable PT |
| §7.1-alt-B | Parallel-tempering of varied mutants of the same protein with batched replicas | Showcases lane B PT productionization; medium-high risk; needs §7.2 LB4 mature |
| §7.1-alt-C | MTT log-det as teaser figure (Coulomb screening for varied system sizes) | Stretch — moves LB1 forward; depends on NPT fix + MTT N=64 investigation |
| §7.1-alt-D | Learned-potential coupling (small ML potential trained against ab initio, batched across mutants) | High reviewer impact; high risk; needs additional ML scaffolding |

**Extension hooks** (if §7.1 finishes ahead of schedule):
- Extend to a 64-system ensemble (more compelling speedup numbers)
- Add an angle/dihedral round and report final RMSD on held-out systems
- Run convergence as a function of ensemble heterogeneity (homogeneous → fully varied)

### §7.2 Future-work items (paper "Outlook" subsection only)

Each future-work item gets a 1-paragraph sketch in the paper's "Applications enabled" subsection. No figures.

| ID | Item | Ring | Status | Depends on |
|---|---|---|---|---|
| LB1 | MTT log-det electrostatics — Coulomb-Laplacian free-energy correction; existing backlog entry `mtt_phase2` | capability | unblocked (EFA validated 2026-04-26) | NPT stability fix + MTT N=64 accuracy investigation |
| LB2 | Allostery-as-circuits — effective-resistance / Chebyshev modes on protein contact graphs | speculative | brainstorm | LB1 machinery + literature validation |
| LB3 | EFA Coulomb extensions — SORF variance reduction, indefinite-Laplacian theory | capability | speculative | DR-LB1-1 |
| LB4 | Parallel-tempering productionization — existing thin `pt/` module → first-class composable with `EnsemblePlan` | capability | brainstorm | Claim 3 API freeze |

Each item links to its deep-research entries in §8.

### §7.3 Paper handling

- **In the paper:**
  - §7.1 figure as its own section ("Application: differentiable force-field fitting")
  - §7.2 items in a single "Outlook / Applications enabled" subsection (1 paragraph each, no figures)
  - Lane B labelled clearly so reviewers understand which results are infrastructure validation vs future scope

---

## §8 Research Track — deep-research queue (shared section)

### 8.1 Convention

- **One shared MD notebook** in NotebookLM (existing) keeps its role as the protocol/algorithms reference
- **New claim-tagged sources** added to the existing MD notebook as needed (PDB statistics scans, related-work papers, etc.)
- **One additional notebook** created for paper-prep work: venue scans, related work, framing
- Each deep-research query writes to `.praxia/research/synthesis.jsonl` with a `claim_id` (or `paper`/`LB<n>`) tag
- Per-claim sections in this spec reference specific `DR-<context>-<n>` IDs

### 8.2 Standing query list (today's snapshot; queue continues to grow)

| ID | Query | Target | Owner | Status |
|---|---|---|---|---|
| DR-claim1-1 | Hetero-batched MD precedent scan (kUPS, jax-md, OpenMM swarm, GROMACS multi-sim, F@H) — what they do/don't do | "Related Work" section; RR1 evidence | NotebookLM + manual | TODO |
| DR-claim1-2 | API-design patterns for "Plan-then-execute" sci-comp (Dask delayed, sklearn Estimators, Equinox modules) | Validates 2-layer API choice; Claim 3 doc | NotebookLM | TODO |
| DR-claim1-3 | Bucket-policy literature — PDB/CASP/AlphaFold-DB protein-size distributions | R3 / RR4 mitigation; bucket ladder defensibility | NotebookLM | TODO |
| DR-claim2-1 | IREE-WASM ecosystem status (May 2026); threefry coverage; large-artifact splitting | Claim 2 smoke-export gate | Web + NotebookLM | TODO |
| DR-claim2-2 | Browser MD precedent (NGL-viewer, MMTF, Mol*) — what's been done, what's the gap | Claim 2 framing; RR6 evidence | NotebookLM | TODO |
| DR-claim3-1 | Declarative scientific API surveys — "ensemble" abstractions in MD/HEP/AI | Claim 3 framing | NotebookLM | TODO |
| DR-LBfig-1 | Differentiable force-field fitting prior art — Espaloma, OpenFF, ML-FFs that backprop into FF params | §7.1 framing + comparison narrative | NotebookLM + web | TODO |
| DR-LB1-1 | Indefinite-Laplacian theory — extending Avron-Toledo to signed Laplacians (Coulomb) | LB1 theoretical foundation | NotebookLM | TODO |
| DR-LB2-1 | Effective-resistance protein networks — literature gap on Coulomb-weighted R_eff | LB2 framing | NotebookLM | TODO |
| DR-paper-1 | Venue fit scan (JCP, JCTC, JCIM, JOSS, ICML/ICLR systems, MLSys) — submission criteria, expected scope | Paper venue decision | NotebookLM | TODO |
| DR-paper-2 | Recent "MD engine" papers (last 24 months) — narrative tropes, expected sections, anticipated reviewer criticisms | Paper structure & rebuttal-table inputs | NotebookLM | TODO |

### 8.3 Output convention

Each completed DR appends a `synthesis.jsonl` entry with the schema already in use (see existing entries):
```json
{
  "timestamp": "ISO-8601",
  "research_id": "DR-claim1-1",
  "query": "...",
  "confidence": "high|medium|low",
  "key_findings": ["..."],
  "relevance_to_prolix": "...",
  "source_file": "references/notes/...",
  "manifest": ["..."],
  "claim_id": "claim1"
}
```

---

## §9 Engineering Substrate (cross-cutting invariants)

These apply to ALL claims uniformly.

### 9.1 Code health invariants

- `mypy --strict` clean on `prolix.api` and all public-API modules
- `ruff` clean across the codebase
- No `Any` in public Protocols (`EnergyFn`, `IntegratorFn`, `Observable`, etc.)
- 90% line coverage on `prolix.api/`; tracked in CI

### 9.2 CI structure

- Every PR: V1, V3, V4, V4-HLO, V5, V7, V8 + A1, A3, A4 + W1, W2
- Weekly: V6, V4 (full system inventory), W4, A2 tutorial
- Nightly: W3 (artifact size), B1 smoke (B=4 sub-protocol), full mypy/ruff sweep
- On-demand (manual): full B1 (B=64) cluster run; S2 throughput

### 9.3 Documentation standards

- Every public API symbol has a docstring with at least one usage example
- One tutorial notebook per claim:
  - `notebooks/02_ensemble_plan_tutorial.ipynb` (Claim 3)
  - `notebooks/03_differentiable_workflow.ipynb` (Claim 3 × S1)
  - `notebooks/04_browser_demo.ipynb` (Claim 2; references the static demo page)
- README's quickstart updated to use `EnsemblePlan` (currently shows legacy `make_batched_energy_fn`)

### 9.4 Deprecation policy

- Legacy entry-points (`batched_produce`, `LangevinState` public re-export, `pad_protein`, `PaddedSystem`, `collate_batch`) emit `DeprecationWarning` for one full version cycle (deprecated in v1.2, removed in v2.0)
- Each deprecation has a documented one-line replacement in CHANGELOG migration table

### 9.5 Bathos (experiment infrastructure)

- **Use bathos for:** genuine experiments (B1 sweeps, §7.1 figure runs, S2 throughput, B4 sensitivity, Lane B exploration)
- **Do NOT use bathos for:** CI tests, core code development, validation matrix tests V1–V8 (those are pytest)
- **Required bathos remotes:** `engaging` (primary cluster), local development workstation, optional H100 cloud burst
- **Bathos run convention:** all experiment runs tagged with `claim_id` and `ring`; results stored in `outputs/bath/<run_id>/`
- Bathos init lives in `.bth.toml` (already exists per commit `4f811f0`)
- Re-run policy: B1 re-runs on any change to `EnsemblePlan`/`BatchPlanner`/`Bundle`/`safe_map` integration

### 9.6 Community signals (post-paper concern; planning hooks now)

- **Issue triage standard:** every issue gets a `claim_id` and `ring` tag at triage; capability-ring issues hold in queue; speculative-ring items get a thanks-and-acknowledge response with "in idea queue"
- **Contributor guidelines:** `CONTRIBUTING.md` (TODO) documents ring system + PR template asking which claim/ring an item belongs to
- **External-user signal:** track which observables/integrators get requested → informs capability ring → core promotion decisions
- All community-signal work is **capability ring** until post-paper; do not divert engine-paper resources

### 9.7 Code organization (post-paper hardening)

- `physics/` (48 modules) subpackaging into `physics/electrostatics/`, `physics/integrators/`, `physics/thermostats/` — **capability ring; post-paper**
- `batched_simulate.py` (1900 LOC monolith) decomposition into focused modules — **capability ring; post-paper**
- Both are tech debt; neither blocks v1.2/paper

---

## §10 Praxia Backlog Schema

### 10.1 Required fields

```json
{
  "id": "string (kebab-case)",
  "title": "string (≤80 chars)",
  "claim_id": "claim1 | claim2 | claim3 | S1 | S2 | LBfig | LB1 | LB2 | LB3 | LB4 | infra | paper",
  "capability_axis": "batching | api | export | integrators | thermostats | electrostatics | sampling | observables | constraints | docs | ci | bench",
  "ring": "core | capability | speculative",
  "status": "ideabin | ready | in_progress | blocked | done | abandoned"
}
```

### 10.2 Optional fields (populated as item moves through lifecycle)

```json
{
  "depends_on": ["other-item-id", ...],
  "blocks": ["other-item-id", ...],
  "deep_research_ids": ["DR-claim1-1", ...],
  "validation_tests": ["tests/path/test_x.py::test_y", ...],
  "success_criterion": "concrete measurable condition",
  "exit_criterion": "how we know it's done",
  "paper_section": "string or null",
  "rebuttal_table_id": "RR1 | RR2 | ... | null",
  "estimated_days": int,
  "owner": "marielle",
  "created": "YYYY-MM-DD",
  "promoted_from": "speculative | capability | null",
  "promoted_on": "YYYY-MM-DD",
  "notes": "..."
}
```

### 10.3 Lifecycle

```
ideabin (in ideas.jsonl) ──promotion──► ready (in backlog.jsonl)
ready ──assign──► in_progress
in_progress ──block──► blocked ──unblock──► in_progress
in_progress ──complete──► done
ready | in_progress ──defer──► abandoned (kept in record, deletes nothing)
```

---

## §11 Dependency DAG

```
HP1 migration policy decided         ─────────────────────────┐
HP2 bundle field audit (Open Q #9)    ─────────────────────┐  │
HP3 V8 safe_map varying shape_spec    ─────────────────┐    │  │
                                                        │    │  │
                                                        ▼    ▼  ▼
P0 NPT-KE bug fix          ──────►   S2 throughput      Claim 1 (V1, V3-V8, B1, B3, B4, B5)
P0 large-scale SETTLE       ──────►   Claim 1 V4 (R1)       │
                                                            │
                                                            ├─► Claim 3 (A1-A4, public API freeze)
                                                            │       │
                                                            │       └─► §7.1 Lane-B figure
                                                            │                │
                                                            ├─► Claim 2 (W1-W4, demo)            │
                                                            │       (needs flat Plan state)     │
                                                            │                                    │
                                                            └─► S1 (D1-D3 differentiability)    │
                                                                       │                         │
                                                                       └─────────────────────────┤
                                                                                                  │
                                                                                                  ▼
                                                                                Paper-submit gate
                                                                                  + Reviewer rebuttal table green
                                                                                  + §7.1 figure reproducible <30 min
                                                                                  + v1.2 tag
```

---

## §12 Milestones (capability gates, NO quarters)

| Gate | Definition of done | Output |
|---|---|---|
| **v1.2 ship** | Claims 1–3 exit criteria all green; S1 & S2 validation passing; §7.1 figure reproducible from clean clone <30 min | tagged release; public API stable; migration table in CHANGELOG |
| **Paper-submit** | v1.2 shipped + RR1–RR7 all "green" status + venue selected (DR-paper-1) + co-authors aligned | preprint to arXiv + venue submission |
| **Post-paper hardening** | Community feedback reviewed; bug-fix sprint; tutorial expansion based on reader confusion points | v1.2.x patch releases |
| **First lane-B publishable result** | LB1 (MTT) or LB2 (allostery) figure-complete + draft writeup | lane-B paper-1 draft |
| **v2.0** | Constraint-aware thermostat (dt ≥ 1.0 fs) + NPT long-traj fix + `physics/` subpackaging + legacy API removal + first lane-B paper submitted | tagged v2.0 |

No wall-clock budgets are pinned. Capability-gated only. Capability-creep risk acknowledged (oracle); mitigation = ring tagging strict + reviewer-rebuttal table acts as scope brake.

---

## §13 Open Questions Carried Forward

| # | Question | Affects | Decision deadline | Resolution path |
|---|---|---|---|---|
| OQ1 | Mixed-integrator ensembles support timeline | Claim 1 + Claim 3 API | v2.0 planning | Capability ring; defer |
| OQ2 | Trajectory checkpoint format (HDF5 / Zarr / pytree) | Claim 1 R2; Claim 3 Trajectory shape | Before §7.1 figure if long trajectories needed | Sub-spec; capability ring |
| OQ3 | Bucket override API ergonomics (positional vs dict) | Claim 3 A1 | Claim 3 implementation start | Sub-spec |
| OQ4 | Whether §7.1 figure extension to 64-system ensemble is in scope of v1.2 | §7.1 deliverable | After §7.1 base figure lands | Decide based on remaining budget |
| OQ5 | Specific reference quantum-chemistry dataset for §7.1 ab initio forces | §7.1 dataset | Before §7.1 starts | Sub-spec; likely small Tinker/Psi4 or curated ANI-1x subset |
| OQ6 | Paper venue selection | Paper-submit gate | After DR-paper-1 + DR-paper-2 | Single-author or with co-authors? |
| OQ7 | Whether `EnsemblePlan` is the published name vs `run_ensemble` function form | Claim 3 public API | Before Claim 3 implementation | Sub-spec; default = `EnsemblePlan` class |
| OQ8 | Threefry vs RBG default for export path | Claim 2 W1-W3 | Before Claim 2 implementation | Smoke gate decides |
| OQ9 | Bathos remote topology — engaging-only or also cloud burst | §9.5 | Pre-§7.1 figure | Configure both; default to engaging |
| OQ10 | When/whether to introduce `Observable` tensor return (RDF, etc.) — capability ring trigger | Claim 3 capability promotions | When first lane-B project asks | Promotion criterion |

---

## §14 Appendix — Re-mapping from existing 2026-05-15 strategic roadmap

The earlier roadmap's phases re-cut as follows. Items keep their numerical identity for git/commit traceability.

| Old phase | Re-cut destination |
|---|---|
| P0 — Stability (LFMiddle, NPT-KE bug, SETTLE batching) | NPT-KE bug → S2 prereq; SETTLE batching → Claim 1 R1; LFMiddle → capability ring (no longer P0 blocker, since dt ≤ 0.5 fs doesn't gate the engine paper's regime) |
| P1a — `MolecularBundle` | Claim 1 mechanism + HP2 audit |
| P1b — `IntegratorState`/`IntegratorConfig` | shipped (commit `d4e8ce8`); referenced by Claim 1 |
| P1c — `EnergyFn`/`IntegratorFn` Protocols | shipped (commit `fc7e7df`); referenced by Claim 1 |
| P1d — `ShimMode` + `custom_jvp` shim | shipped (commit `f154266`); referenced by S1 |
| P1e — `AtomAxisSpec` + `BatchPlan` tiling | shipped (commit `64f7a5d`); core mechanism in Claim 1 |
| P2a — OpenMM parity harness (current `PhysicsSystem`) | folds into HP2 audit + V1 |
| P2b — Full cross-validation matrix | distributed across V1–V8 + S1 D1 |
| P3 — Benchmark suite | distributed across B1, B3, B4, B5, S2 (with B1 as headline pre-registered) |
| P4 — WASM | Claim 2 |
| P5 — Paper | Paper-submit gate (§12), with §7.1 Lane B figure now a hard requirement |

---

**End of spec.** Next step: see §10 schema for seeding `.praxia/backlog.jsonl` from this spec.
