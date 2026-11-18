# Numerical Parity & GPU OpenMM Fast-Path (Pallas v3.6–v3.7)

Building on the {doc}`Pallas v3.5 megakernel stack <pallas_megakernel>` and the four-tier L40S multi-engine physics audit ({doc}`physics_validation`), **Pallas v3.6 and v3.7** close the remaining numerical discretization, mixed-precision throughput, tile-pruning, and Python–GPU host-synchronization gaps across `prolix.physics`, `prolix.api`, and `prolix.openmm`.

---

## Summary of Pallas v3.6–v3.7 Advancements

| Subsystem | Pre-v3.6 Behavior | Pallas v3.6–v3.7 Design | Physical / Throughput Impact |
| :--- | :--- | :--- | :--- |
| **SPME Box Rescaling & Deconvolution** | Cached $G(\mathbf{k}; \mathbf{L}_{\text{ref}})$ reused across NPT trial volume moves without wavevector Gaussian update | Exact isotropic/orthorhombic $G(\mathbf{k}; \mathbf{L})$ rescaling + optional 16-offset RMS discrete $i\mathbf{k}$ deconvolution (`ik_rms16`) | Eliminates **`+352.8 bar`** reciprocal-pressure bias and **`-1.6%`** fixed-mesh NPT liquid density offset |
| **Switched & `P=6` Dispersion (`LJ-PME`)** | Unswitched Lorentz–Berthelot continuum tail only | Closed-form + cancellation-free power-series quintic-switched tail (`lj_switched_tail_integrals`), geometric combining, and full `P=6` dispersion SPME (`NonbondedForce.LJPME`, `use_lj_pme`) | Exact `float64` parity with OpenMM `Reference` & analytical `jax.grad` wrt $(\mathbf{r}, \sigma, \varepsilon, \mathbf{B})$ |
| **Holonomic `RATTLE` & Disjoint Constraint Kernels** | Monolithic 3-way `lax.cond` CTA + redundant post-`R2` velocity projection before `B2` | Symplectic single post-`B2` velocity projection + disjoint unconstrained / `SETTLE` / `SHAKE`–`RATTLE` kernel launches (`input_output_aliases`) | Eliminates spurious rotational KE damping and **`-4.9` to `-5.2 K`** solute–water NVT temperature split while cutting register pressure |
| **`fltflt` `(hi, lo)` `rn.f32` Mixed Precision** | Hardware `float64` constraint/state accumulators (`1:64` FP64 rate on Ada/Blackwell) | Compensated float-float `(hi, lo)` pair arithmetic using explicit round-to-nearest `rn.f32` inline PTX primitives | Mixed-precision conservation at near-FP32 throughput on `L40S` (`sm_89`) and Blackwell (`sm_120`) |
| **Morton Tile Pruning & Deterministic Forces** | $O(N_{\text{tiles}}^2)$ coarse tile AABB scan; bonded/CMAP registers shared with tile-pair CTAs | $O(N_{\text{tiles}})$ 3D cell-grid (`nb >= 64`) + 8-octant sub-AABB pruning, separate bonded CTAs, and `(hi_i32, lo_i32)` $2^{20}$ fixed-point force planes | Prunes empty solvent tile pairs on `DHFR_24k` / `ApoA1_92k` and guarantees bitwise-invariant forces under `jax_enable_x64=False` |
| **OpenMM GPU Fast-Path & Staged Reporters** | Per-report host neighbor-list / state round-trips in `ProlixContext` & `ProlixSimulation` | Persistent GPU `_morton_runner` in {class}`~prolix.openmm.ProlixContext`, device ring-buffer reporter batching in {class}`~prolix.openmm.ProlixSimulation`, and vectorized alchemical offsets in {class}`~prolix.openmm.ProlixBatchSimulation` | Single-launch multi-interval OpenMM rollouts + vectorized 8+ window $\partial U / \partial \lambda$ free-energy evaluation |

---

## 1. Exact SPME Box Scaling, Switched Tails, & `LJ-PME`

### Isotropic & Orthorhombic Green's Function Scaling in NPT

On a fixed FFT mesh $(K_x, K_y, K_z)$, the SPME reciprocal influence function depends on the cell edge lengths $\mathbf{L} = (L_x, L_y, L_z)$ through **both** the reciprocal volume prefactor $1/V$ and the wavevector Gaussian $\exp(-\pi^2 |\mathbf{m}^*(\mathbf{L})|^2 / \alpha^2)$, where $m_d^*(\mathbf{L}) = n_d / L_d$:

$$G(\mathbf{n}; \mathbf{L}) = \frac{\exp\!\left(-\frac{\pi^2}{\alpha^2}\sum_{d} (n_d / L_d)^2\right)}{\pi V(\mathbf{L}) \left(\sum_d (n_d / L_d)^2\right) |b_x(n_x)|^2 |b_y(n_y)|^2 |b_z(n_z)|^2}$$

Under an isotropic Monte Carlo barostat volume trial $\mathbf{L}' = \lambda \mathbf{L}_{\text{ref}}$ ($V' = \lambda^3 V_{\text{ref}}$, $|\mathbf{m}^*(\mathbf{L}')|^2 = \lambda^{-2} |\mathbf{m}^*(\mathbf{L}_{\text{ref}})|^2$), holding $G$ fixed or scaling only by $1/\lambda$ omits the Gaussian derivative with respect to volume—introducing a systematic **`+352.8 bar`** reciprocal-pressure offset on rigid `TIP3P` water. `scale_influence_function_isotropic` updates $G(\mathbf{n}; \mathbf{L}')$ analytically on device with underflow-safe Nyquist masking, matching fresh `influence_function` evaluation to `< 10⁻¹³` relative error in `float64` (`< 10⁻⁶` in `float32`) and eliminating the reciprocal virial bias.

### Quintic-Switched Dispersion Tails & `P=6` `LJ-PME`

- **Switched Continuum Tail (`lj_switched_tail_integrals`)**: When OpenMM's quintic $C^2$ switching function $S(r)$ is active on $[r_{\text{switch}}, r_{\text{cut}}]$, Prolix integrates $\int_{r_{\text{switch}}}^{r_{\text{cut}}} [1 - S(r)] r^{2-2p}\,dr + \int_{r_{\text{cut}}}^{\infty} r^{2-2p}\,dr$ ($p \in \{3, 6\}$) in closed form with cancellation-free Taylor series when $(r_{\text{cut}} - r_{\text{switch}})/r_{\text{cut}}$ is small, supporting both Lorentz–Berthelot and geometric combining rules.
- **Multiplicative `P=6` Dispersion Ewald (`LJ-PME`)**: For `openmm.NonbondedForce.LJPME` (or `use_lj_pme=True` in {class}`~prolix.openmm.ProlixOpenMMConfig` / {class}`~prolix.knob_surface.KnobSurface`), Prolix decomposes attractive dispersion via $b_i = 2\sqrt{\varepsilon_i}\sigma_i^3$ and $g_6(y) = e^{-y}(1 + y + \tfrac{1}{2}y^2)$ ($y = \alpha_{\text{lj}}^2 r^2$), retains the finite $\mathbf{k} = \mathbf{0}$ reciprocal mode $R_6(0) = \tfrac{2}{3}\alpha_{\text{lj}}^3$, and evaluates excluded/1-4 reciprocal subtractions $+b_i b_j [1 - g_6(\alpha_{\text{lj}} r)] / r^6$ via a degree-24 alternating Taylor expansion for $y < 2$ so both primal energies and `jax.grad` remain smooth as $r \to 0$.

---

## 2. Symplectic `RATTLE`, Disjoint Constraint Kernels, & `fltflt` Arithmetic

### Eliminating Spurious Post-`R2` Velocity Damping

In velocity-Verlet / `BAOAB` `RATTLE` integration (`B1 -> A1 -> O -> A2 -> R_pos -> B2 -> R_vel`), the position constraint impulse during `R_pos` acts along the **pre-step** bond vector $\mathbf{r}_{ij}(t)$, leaving an $O(\Delta t\,\omega^2)$ velocity component along the **updated** bond vector $\mathbf{r}_{ij}(t + \Delta t)$ prior to the second half-kick `B2`. Applying an orthogonal velocity projection onto $\mathbf{r}_{ij}(t + \Delta t)$ *before* `B2` (in addition to the true `R_vel` projection *after* `B2`) drains rotational kinetic energy from `HBonds`-constrained solute clusters and produced the `-4.9` to `-5.2 K` solute–water temperature split observed in v3.5. Pallas v3.6 restores exact symplectic `RATTLE` momentum updates, conserving rigid-cluster rotational kinetic energy and solute–solvent equipartition.

### Disjoint Constraint `pallas_call`s & `rn.f32` Float-Float Pairs

1. **Disjoint CTA Partitioning**: Instead of branching across unconstrained solute atoms, 3-atom rigid `SETTLE` waters, and iterative `SHAKE`/`RATTLE` solute clusters inside a single high-register kernel, `settle.py` dispatches up to three specialized `pl.pallas_call` kernels chained in place via `input_output_aliases` (zero extra HBM copies).
2. **`fltflt` `(hi, lo)` `rn.f32` Accumulators**: Workstation and datacenter inference GPUs (`NVIDIA L40S` `sm_89`, `RTX PRO 6000 Blackwell` `sm_120`) execute IEEE `float64` at `1:64` of FP32 throughput. In `precision="mixed"`, Prolix represents sensitive coordinate and constraint accumulators as unevaluated FP32 pairs $x = x_{\text{hi}} + x_{\text{lo}}$ (`|x_lo| <= 0.5 ulp(x_hi)`) backed by explicit round-to-nearest PTX (`add.rn.f32`, `mul.rn.f32`, `fma.rn.f32`) so compiler contraction never invalidates Knuth `TwoSum` or Dekker `TwoProdFMA` error-free transformations (~48-bit effective mantissa in pure FP32 registers).

---

## 3. Cell-Grid + Octant Sub-AABB Tile Pruning & `(hi_i32, lo_i32)` Deterministic Forces

- **3D Cell-Grid & 8-Octant Sub-AABB Pruning**: When the number of 32-atom Morton tiles satisfies `nb >= 64` (all solvated protein benchmarks from `2.2k` atoms upward, including `DHFR_24k`, `1AKE_45k`, `ApoA1_92k`, and `Cellulose_408k`), `pallas_build_morton_tile_pairs_gpu` bins tile AABBs into a 3D spatial cell grid (`O(N_tiles)` discovery) and subdivides each 32-atom Morton tile into **8 octant sub-bounding boxes** (4 atoms per sub-box). Tile pairs whose coarse 32-atom AABBs overlap at a corner—but whose occupied 4-atom sub-boxes all lie beyond $(r_{\text{cut}} + r_{\text{skin}})$—are pruned before force evaluation.
- **Separate Bonded CTAs & `int32` `(hi, lo)` Fixed-Point Plane**: Nonbonded tile-pair CTAs and bonded/CMAP CTAs launch as separate kernels sharing aliased force buffers, keeping CMAP bicubic spline state out of nonbonded tile registers. When `deterministic_forces=True`, each masked force contribution $\Delta f$ is split into `hi = floor(df)` (`int32`) and `lo = rint((df - floor(df)) * 2^20)` (`int32`), accumulated via PTX `atom.global.add.s32`, and reconstructed as `hi + lo * 2^-20`—providing bitwise-reproducible forces independent of CTA scheduling without relying on `int64` (`jax_enable_x64=False` safe).

---

## 4. OpenMM GPU Fast-Path, Staged Reporters, & Alchemical Parameter Offsets

```python
import io
import openmm
from openmm import app, unit
from prolix.openmm import ProlixBatchSimulation, ProlixOpenMMConfig, ProlixSimulation

# 1. Single-system GPU fast-path + device ring-buffer reporter staging
log_buf = io.StringIO()
sim = ProlixSimulation(
    pdb.topology,
    system,
    openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds),
    config=ProlixOpenMMConfig(precision="mixed", nonbonded_engine="morton_tile_pairs", use_pallas=True),
)
sim.context.setPositions(pdb.positions)
sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 42)
sim.reporters.append(
    app.StateDataReporter(log_buf, reportInterval=100, step=True, potentialEnergy=True, temperature=True)
)
# Executes multiple report intervals on device before bulk jax.device_get replay
sim.step(10_000)

# 2. Multi-window alchemical lambda sweep in ProlixBatchSimulation
lambdas = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0]
batch = ProlixBatchSimulation(
    systems=[alchemical_system] * len(lambdas),
    integrators=openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds),
    positions_list=[pdb.positions] * len(lambdas),
    config=ProlixOpenMMConfig(precision="mixed", use_pallas=True),
)
batch.setParameter("lambda_electrostatics", lambdas)
du_dlambda_per_window = batch.compute_parameter_derivatives()
```

---

## 5. Reproducing the A/B Suite & Cross-Architecture Aggregator

Run the turnkey A/B suite (`S1`–`S5`) on a GPU node via `bathos` (strictly enforcing `require_openmm_cuda_context` on every OpenMM baseline) or in deterministic CI `--cpu-smoke` mode:

```bash
# Production GPU run tracked in bathos (`bth`)
bth run --tags v36-ab --campaign-id <campaign-id> \
  --output-paths docs/benchmarks/261117_l40s_v36_ab.json -- \
  uv run python scripts/benchmarks/bench_v36_cupotential_ab.py \
  --gpu-tag l40s --out docs/benchmarks/261117_l40s_v36_ab.json

# Fast CI smoke check (CPU interpret, zero GPU required)
uv run python scripts/benchmarks/bench_v36_cupotential_ab.py \
  --cpu-smoke --out tmp/bench_v36_cpu_smoke.json

# Aggregate 4-GPU Pallas v3.5 release ladders (validating all 7 CUDA provenance fields)
uv run python scripts/benchmarks/aggregate_v35_cross_arch_report.py \
  --bench-dir docs/benchmarks \
  --out-json tmp/v35_cross_arch.json \
  --out-md tmp/v35_cross_arch.md
```
