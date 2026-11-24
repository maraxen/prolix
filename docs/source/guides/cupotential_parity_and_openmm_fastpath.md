# Numerical Parity, Prolate Splittings, & GPU OpenMM Fast-Path (Pallas v3.6–v3.9)

Building on the {doc}`Pallas v3.5 megakernel stack <pallas_megakernel>` and the four-tier L40S multi-engine physics audit ({doc}`physics_validation`), **Pallas v3.6 through v3.9** extend Prolix across reduced-form triclinic explicit solvent, clean-room $C^2$-continuous Prolate Spheroidal Ewald splittings, compensated float-float (`fltflt`) mixed precision, sparse alchemical `InteractionGroup`s, anisotropic/membrane GPU barostats, 4.0 fs Hydrogen Mass Repartitioning (`HMR`), and zero-copy JAX/DLPack OpenMM state buffers.

## Summary of Pallas v3.6–v3.9 Advancements

| Subsystem | Legacy Behavior | Pallas v3.6–v3.9 Design | Physical / Throughput Impact |
| :--- | :--- | :--- | :--- |
| **Reduced-Form Triclinic GPU Fast-Path** | Orthorhombic-only Morton tiles, SPME, `SETTLE`, and ragged queues (`_is_triclinic_box_array` fallback) | Lower-triangular $\mathbf{H} = [\mathbf{a}; \mathbf{b}; \mathbf{c}]$ fractional Morton sorting, sequential `z -> y -> x` MIC + $\delta\mathbf{n} \in \{-1,0,1\}^3$ tile discovery, triclinic `8x8x8` SPME, `SETTLE`+`RATTLE`, and 1D ragged queues | Full GPU fast-path on rhombic dodecahedron (~29% fewer waters than cube) and truncated octahedron (~23% fewer waters) |
| **$C^2$ Prolate Spheroidal Ewald (`P=1` & `P=6`)** | Gaussian $\operatorname{erfc}(\alpha r)/r$ and incomplete-gamma $g_6(\alpha_{\text{lj}} r)/r^6$ splittings with jump discontinuities at $r_c$ | Clean-room compact polynomial splittings (`arXiv:2505.09727`) minimizing $\int_{c_p}^\infty |\hat{L}_p(\kappa)|^2 \kappa^2\,d\kappa$ with $S_p(r_c^2) = S_p'(r_c^2) = S_p''(r_c^2) = 0$ (`c_1 = 10.75`, `c_6 = 12.764031957789484`) | Exact $C^2$ force/hessian continuity at $r_c$, improved NVE conservation, and sharper reciprocal decay at fixed FFT mesh |
| **SPME Box Rescaling, Switched Tails, & `LJ-PME`** | Cached $G(\mathbf{k}; \mathbf{L}_{\text{ref}})$ in NPT + unswitched Lorentz–Berthelot continuum tail | Exact isotropic/anisotropic $G(\mathbf{k}; \mathbf{L})$ rescaling, 16-offset `ik_rms16` deconvolution, `lj_switched_tail_integrals`, geometric combining, and `NonbondedForce.LJPME` | Eliminates **`+352.8 bar`** NPT reciprocal-pressure bias and **`-1.6%`** liquid density offset; exact `LJPME` parity |
| **Symplectic `RATTLE`, `fltflt` `rn.f32`, & `HMR`** | Hardware `float64` constraints (`1:64` FP64 rate on Ada/Blackwell) + hardcoded water masses + pre-`B2` damping | Single post-`B2` `RATTLE` projection, disjoint constraint CTAs, `(hi, lo)` `rn.f32` PTX + bit-exact `ff_double_rt`, and dynamic `(r_OH, r_HH, m_O, m_H)` extraction | Eliminates **`-5 K`** solute–water NVT split, preserves ~48-bit constraint precision at FP32 speed, and unlocks **`dt = 4.0 fs`** HMR (`2x` `ns/day`) |
| **Morton Tile Pruning & Deterministic Forces** | $O(N_{\text{tiles}}^2)$ coarse AABB scan + non-associative FP32 atomic adds | $O(N_{\text{tiles}})$ 3D cell-grid (`nb >= 64`) + octant sub-AABB pruning, separate bonded CTAs, and `(hi_i32, lo_i32)` $2^{20}$ fixed-point force planes | Prunes corner-touching solvent tiles and guarantees bitwise-invariant forces across homogeneous and ragged runs |
| **Sparse `InteractionGroup`s, Barostats, & Zero-Copy DLPack** | Dense $O(N^2)$ `CustomNonbondedForce` masks, host barostat/custom-force sync, and NumPy host copies | $O(|U|\cdot N)$ sparse `InteractionGroup` compilation (`is_gpu_runner_fast`), dynamic `global_params` JIT scalars, GPU anisotropic/membrane barostats, batch `minimizeEnergy`/checkpoints, and `*Jax` DLPack views | Zero `(N, N)` HBM blowup at `N >= 10k`, zero-recompilation multi-window $\lambda$ sweeps, and zero-copy device handoff to ML potentials / CV biases |

## 1. SPME Box Scaling, `LJ-PME`, & $C^2$ Prolate Spheroidal Splittings

### Isotropic & Anisotropic Green's Function Scaling in NPT

On a fixed FFT mesh $(K_x, K_y, K_z)$, the SPME reciprocal influence function depends on the lattice dimensions through both the volume prefactor $1/V$ and the wavevector Gaussian $\exp(-\pi^2 |\mathbf{m}^*(\mathbf{H})|^2 / \alpha^2)$. Under isotropic (`MonteCarloBarostat`), axis-independent (`MonteCarloAnisotropicBarostat`), or XY-coupled (`MonteCarloMembraneBarostat`) Monte Carlo box moves inside the persistent GPU JIT loop, `scale_influence_function_isotropic` updates $G(\mathbf{n}; \mathbf{L}')$ analytically on device with Nyquist masking—matching fresh `influence_function` evaluation to `< 10⁻¹³` relative error in `float64` (`< 10⁻⁶` in `float32`) and eliminating the `+352.8 bar` reciprocal-pressure offset on rigid `TIP3P` water.

### Quintic-Switched Dispersion Tails & Multiplicative `P=6` `LJ-PME`

- **Switched Continuum Tail (`lj_switched_tail_integrals`)**: Evaluates $\int_{r_{\text{switch}}}^{r_{\text{cut}}} [1 - S(r)] r^{2-2p}\,dr + \int_{r_{\text{cut}}}^{\infty} r^{2-2p}\,dr$ ($p \in \{3, 6\}$) in closed form with cancellation-free Taylor series when $(r_{\text{cut}} - r_{\text{switch}})/r_{\text{cut}}$ is small across Lorentz–Berthelot and geometric combining rules.
- **Standard `P=6` Dispersion Ewald (`prolix.physics.lj_pme`)**: Implements `openmm.NonbondedForce.LJPME` (`use_lj_pme=True`) via multiplicative dispersion parameters $b_i = 2\sqrt{\varepsilon_i}\sigma_i^3$, $g_6(y) = e^{-y}(1 + y + \tfrac{1}{2}y^2)$ ($y = \alpha_{\text{lj}}^2 r^2$), finite $\mathbf{k} = \mathbf{0}$ reciprocal mode $R_6(0) = \tfrac{2}{3}\alpha_{\text{lj}}^3$, and degree-24 Taylor stabilization for excluded/1-4 pairs at small $y$.

### Clean-Room $C^2$ Prolate Spheroidal Splittings (`prolix.physics.prolate_splitting`)

Standard Gaussian Ewald splittings leave a nonzero short-range residual at $r = r_c$, introducing small potential and force impulses whenever pairs cross the cutoff. Following the variational prolate spheroidal framework (`arXiv:2505.09727`), {mod}`prolix.physics.prolate_splitting` constructs compactly supported $C^2$-continuous splittings

$$r^{-p} = S_p(r^2; r_c) + L_p(r^2; r_c), \qquad p \in \{1, 6\},$$

where $L_p(r; r_c) = r_c^{-p} P_p(r^2 / r_c^2)$ for $r \le r_c$ and $L_p(r; r_c) = r^{-p}$ for $r \ge r_c$:

1. **Exact $C^2$ Cutoff Vanishing**: $P_p(u)$ ($u = r^2 / r_c^2 \in [0, 1]$) satisfies $P_p(1) = 1$, $P_p'(1) = -p/2$, and $P_p''(1) = p(p+2)/4$, guaranteeing $S_p(r_c^2) = S_p'(r_c^2) = S_p''(r_c^2) = 0$ with zero switching window required.
2. **Band-Limited Reciprocal Tail**: Subject to the three boundary constraints at $u = 1$, $P_p$ minimizes the out-of-band Fourier energy $\int_{c_p}^{\infty} |\hat{L}_p(\kappa)|^2 \kappa^2\,d\kappa$ ($c_1 = 10.75$ for `P=1` electrostatics; $c_6 = 12.764031957789484$ for `P=6` dispersion) via exact spherical-Bessel Gram matrices.
3. **Differentiable Ortho + Triclinic SPME & Portable Export**: {func}`~prolix.physics.prolate_splitting.prolate_spme_electrostatics` and {func}`~prolix.physics.prolate_splitting.prolate_dispersion_pme` support orthorhombic `(3,)` and lower-triangular triclinic `(3, 3)` boxes, finite `P=6` $k = 0$ limits with zero gradient at $k^2 = 0$, analytical `@jax.custom_vjp` coordinate/parameter derivatives, and pure-JAX StableHLO export inside `prolix.export.export_context()`.

## 2. Symplectic `RATTLE`, `fltflt` (`ff_double_rt`), & 4.0 fs HMR

- **Symplectic Single Post-`B2` `RATTLE` & Disjoint CTAs**: `settle.py` applies a single symplectic velocity projection after `B2` (preserving solute rotational kinetic energy and eliminating the `-5 K` solute–water temperature split) and partitions unconstrained atoms, rigid `SETTLE` waters, and `SHAKE`/`RATTLE` clusters into specialized `pl.pallas_call` kernels chained via `input_output_aliases`.
- **`fltflt` `(hi, lo)` `rn.f32` + Bit-Exact `ff_double_rt`**: Compensated float-float pairs (`prolix.physics.fltflt_pallas.FltFlt`) use explicit `add.rn.f32` / `mul.rn.f32` / `fma.rn.f32` PTX on GPU and integer-bit-exact power-of-two doubling (`ff_double_rt`, falling back cleanly to pure JAX inside `in_export_context()`), delivering ~48-bit constraint accuracy without `1:64` FP64 throughput penalties on `L40S` and Blackwell GPUs.
- **Hydrogen Mass Repartitioning (`HMR`, `dt = 4.0 fs`)**: `EnsemblePlan`, `ProlixContext`, `ProlixSimulation`, and `ProlixBatchSimulation` extract the exact per-system `(r_OH, r_HH, m_O, m_H)` geometry and masses directly from `MolecularBundle` / `KnobSurface.constraints` rather than assuming `m_H = 1.008 amu`. Systems built with `hydrogenMass=3.0 * unit.amu` and `constraints=app.HBonds` run on the persistent GPU fast-path at **`dt = 4.0 fs`** (`2x` simulated `ns/day` per force evaluation).

## 3. Reduced-Form Triclinic GPU Fast-Path & Cross-Shape Ragged Batching

Many globular proteins pack in **rhombic dodecahedron** ($V \approx 0.707\,d^3$, ~29% fewer solvent molecules than an equivalent cube) or **truncated octahedron** ($V \approx 0.770\,d^3$, ~23% fewer solvent molecules) cells. Pallas v3.9 brings OpenMM reduced-form lower-triangular triclinic boxes

$$\mathbf{H} = \begin{bmatrix} a_x & 0 & 0 \\ b_x & b_y & 0 \\ c_x & c_y & c_z \end{bmatrix}, \qquad a_x, b_y, c_z > 0, \quad |b_x|, |c_x| \le \tfrac{1}{2}a_x, \quad |c_y| \le \tfrac{1}{2}b_y,$$

onto the full GPU Morton fast-path (`is_triclinic=True`, safely disambiguating `B == 3` orthorhombic `(3, 3)` batches from single-system `(3, 3)` triclinic matrices):

1. **Triclinic Morton Tiles & Sound 27-Shift Discovery**: Atoms are sorted by fractional coordinates $\mathbf{s} = \mathbf{r}\mathbf{H}^{-1} \bmod 1$. Tile AABBs unwrap sequentially (`z -> y -> x`), and GPU tile-pair discovery inspects $\delta\mathbf{n} \in \{-1, 0, 1\}^3$ lattice shifts around the sequential-MIC image so zero cutoff pairs are missed near skewed dodecahedron/octahedron corners. Direct $32 \times 32$ nonbonded CTAs and bonded CTAs evaluate minimum-image vectors in 6 FMAs from flattened `B * 9` box parameters.
2. **Triclinic `8x8x8` SPME & `SETTLE`+`RATTLE`**: Pallas charge spreading maps $\mathbf{s} = \mathbf{r}\mathbf{H}^{-1}$ while preserving `8x8x8` shared-memory brick coalescing, and reciprocal force gather applies the exact transpose $\mathbf{F} = -\mathbf{H}^{-\top}\nabla_{\mathbf{s}} E$. Both `SETTLE` and `RATTLE` wrap and project along reduced triclinic vectors in `fltflt` precision.
3. **1D Ragged Cross-Shape Batching**: `pallas_ragged_queue.py` and `ProlixBatchSimulation` mix `cubic`, `rhombic_dodecahedron`, and `truncated_octahedron` replicas of differing atom counts inside a single 1D prefix-sum CTA work queue with optional `(hi_i32, lo_i32)` $2^{20}$ deterministic force accumulation.

## 4. Sparse Alchemical `InteractionGroup`s, GPU Barostats, & Zero-Copy DLPack

- **Sparse $O(|U| \cdot N)$ `CustomNonbondedForce` (`is_gpu_runner_fast`)**: Alchemical sterics and host–guest potentials typically couple a small ligand group $U$ ($|U| \sim 10\text{--}100$) to an environment $V$ ($N \sim 10^4\text{--}10^5$). When `num_groups > 0` and $|U| \le \max(512, N // 4)$, `lepton_compiler.py` compiles sparse index/multiplicity tables (`sparse_u_indices`, `sparse_multiplicity`) instead of allocating dense `(N, N)` exclusion/interaction matrices, marks `is_gpu_runner_fast = True`, and threads OpenMM global context parameters (`lambda_sterics`, `lambda_electrostatics`) as dynamic JAX scalars into `jit_step`—eliminating XLA recompilation across `setParameter` sweeps.
- **GPU Anisotropic & Membrane Barostats + Batch Minimization/Checkpoints**: `MonteCarloAnisotropicBarostat` and `MonteCarloMembraneBarostat` run inside the persistent GPU Morton loop in `ProlixContext` and `ProlixSimulation`, while `ProlixBatchSimulation` supports vectorized `minimizeEnergy`, `createCheckpoint`/`loadCheckpoint`, and `saveStates`/`loadStates`.
- **Zero-Copy JAX / DLPack Buffers**: `ProlixContext`, `ProlixSimulation`, and `ProlixBatchSimulation` expose `getPositionsJax`, `setPositionsJax` (accepting `jax.Array` or any `__dlpack__` tensor via `jax.dlpack.from_dlpack`), `getVelocitiesJax`, `setVelocitiesJax`, and `getForcesJax` so hybrid ML/MM loops, collective-variable biases, and custom neural reporters read and write device memory with zero host round-trips:

```python
import jax.numpy as jnp
import openmm
from openmm import app, unit
from prolix.openmm import ProlixBatchSimulation, ProlixOpenMMConfig, ProlixSimulation

# 1. Rhombic-dodecahedron HMR (4.0 fs) + zero-copy JAX/DLPack access
sim = ProlixSimulation(
    pdb.topology,
    dodecahedron_hmr_system,  # Created with hydrogenMass=3.0*unit.amu, constraints=app.HBonds
    openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds),
    config=ProlixOpenMMConfig(precision="mixed", nonbonded_engine="morton_tile_pairs", use_pallas=True),
)
sim.context.setPositions(pdb.positions)
sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 42)
sim.step(5_000)

# Zero-copy device jax.Array (nm, nm/ps, kJ/(mol*nm)) -- no D2H numpy copy
pos_nm_jax = sim.getPositionsJax(in_nm=True, enforcePeriodicBox=True)
forces_jax = sim.getForcesJax(in_kj_per_mol_nm=True)
sim.setPositionsJax(pos_nm_jax + jnp.zeros_like(pos_nm_jax), in_nm=True)

# 2. Sparse InteractionGroup alchemical lambda batch with zero recompilation
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
batch = ProlixBatchSimulation(
    systems=[sparse_alchemical_system] * len(lambdas),
    integrators=openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds),
    positions_list=[pdb.positions] * len(lambdas),
    config=ProlixOpenMMConfig(precision="mixed", deterministic_forces=True, use_pallas=True),
)
batch.setParameter("lambda_sterics", lambdas)
batch.minimizeEnergy(maxIterations=100)
batch.step(2_000)
du_dlambda = batch.compute_parameter_derivatives()
batch_pos_jax = batch.getPositionsJax(in_nm=True)  # shape (B, N_padded, 3) on device
```

## 5. Reproducing the Benchmark Suites (`v3.5`–`v3.9`)

Run the turnkey `v3.6/v3.7` and `v3.9` benchmark suites (`S1`–`S5`) on a GPU node via `bathos` (strictly enforcing `require_openmm_cuda_context` on every OpenMM baseline) or in deterministic CI `--cpu-smoke` mode:

```bash
# 1. Production GPU v3.9 suite (triclinic, ragged cross-shape, C^2 prolate, sparse alchem, HMR + DLPack)
bth run --tags v39-complete --campaign-id <campaign-id> \
  --output-paths docs/benchmarks/261118_l40s_v39_complete.json -- \
  uv run python scripts/benchmarks/bench_v39_triclinic_alchem_prolate.py \
  --gpu-tag l40s --out docs/benchmarks/261118_l40s_v39_complete.json

# Fast CI smoke check for v3.9 (CPU interpret, zero GPU required)
uv run python scripts/benchmarks/bench_v39_triclinic_alchem_prolate.py \
  --cpu-smoke --suites S1,S2,S3,S4,S5 --out tmp/bench_v39_cpu_smoke.json

# 2. Production GPU v3.6/v3.7 A/B suite (NPT SPME scaling, RATTLE equipartition, octant pruning, fltflt)
bth run --tags v36-ab --campaign-id <campaign-id> \
  --output-paths docs/benchmarks/261117_l40s_v36_ab.json -- \
  uv run python scripts/benchmarks/bench_v36_cupotential_ab.py \
  --gpu-tag l40s --out docs/benchmarks/261117_l40s_v36_ab.json

# 3. Aggregate 4-GPU Pallas v3.5 release ladders (validating all 7 CUDA provenance fields)
uv run python scripts/benchmarks/aggregate_v35_cross_arch_report.py \
  --bench-dir docs/benchmarks \
  --out-json tmp/v35_cross_arch.json \
  --out-md tmp/v35_cross_arch.md
```
