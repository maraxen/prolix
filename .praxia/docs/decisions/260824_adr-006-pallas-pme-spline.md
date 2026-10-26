# ADR 006: Atom-parallel Pallas kernels for SPME B-spline spread/gather

## Status

**Deferred (2026-08-24).** Approach A remains the accepted kernel shape, but Pallas/Triton/Mosaic work is paused until implicit and explicit solvent paths have been squeezed in **high-level JAX** (XLA stencil, tiling, remat, padding, exclusion scatter, GB chunking). Opt-in code may stay in tree; do not compile-gate, default-flip, or continue the L40S Triton spike as the next bottleneck hunt. Revisit only after those JAX leaves are ranked and mitigated.

## Context

After the SETTLE O-step water-`scan` was replaced by batched dispatch, Stage-2 1vii `apply_fn` on L40S became **force-bound** (jitted flash MD ≈ force-only ~1.2 ms). Remaining electrostatic cost is reciprocal-space **spread/gather** plus FFT, not rigid-water linear algebra.

`src/prolix/physics/pme.py` historically unrolled a Python 4³ loop of `Q.at[ix,iy,iz].add` (and the matching gather in `_spme_bwd`). That lowers to **64 sequential XLA scatters**. A vectorized rewrite builds `(N, 64)` indices/weights and one scatter-add / one gather. That is the correct JAX stencil, and it can be bitwise to the loop on `Q`, but it is still **not** OpenMM’s CUDA spline:

- OpenMM: one thread (or warp) per atom, B-spline weights in **registers**, 64-iteration loop **inside one kernel**, `atomicAdd` into the mesh, then cuFFT.
- XLA scatter: generic “add these HBM values at these indices.” The 64-point stencil is a batch axis in DRAM.

Prolix already uses Pallas for a different hot loop ([`pallas_kernels.py`](../../src/prolix/pallas_kernels.py) GB/Coulomb): tile the work, **`custom_vjp` for AD**, XLA fallback. SPME should copy that contract. `spme_energy_with_forces` already has a custom VJP whose backward **is** gather; autodiff through `atomic_add` is rejected.

FFT (`rfftn` / `irfftn`) and the Green’s function stay XLA. This ADR is only the **particle–mesh spline**.

## Decision

**Approach A — atom-parallel Pallas spline (OpenMM-shaped), opt-in until gated.**

Two Mosaic/Pallas kernels, **f32**, **order 4 only**:

1. **`pme_spread`** — launch `grid=(n_pad,)`. Each program loads `r, q, mask`; computes fractional coordinate `u` and four B-spline weights per axis **in registers**; 64-iteration loop; `atomic_add` into `Q[ix % Kx, iy % Ky, iz % Kz]`. Masked atoms contribute zero.
2. **`pme_gather`** — same launch. Each program loads `theta` at the 64 stencil points, accumulates potential and `dE/dr` with `dw` in registers, writes `(N, 3)` forces and `(N,)` potential. **No atomics** (one writer per atom).

Host glue:

- `grid_dims` remain Python ints (existing PME JIT contract).
- `spread_charges` / `_spme_bwd` branch to Pallas when `use_pallas_pme` (name TBD) is on **and** the backend can compile the kernels; otherwise keep the vectorized XLA stencil.
- Do not `jax.grad` through spread atomics. Keep `spme_energy_with_forces.custom_vjp`.

Numeric gate for the kernel: `Q` and forces vs the **vectorized stencil / 4³ oracle**, f32, padded 1vii. OpenMM energy parity is a **separate** electrostatics check, not the kernel acceptance test.

Perf gate: L40S 1vii **force-only** vs the XLA-stencil baseline (~1.2 ms class). Success is a real drop in `pme_charge_spread` / `pme_bwd_gather` (or host step if scopes stay fused). **Do not default-flip** if 1vii spread loses to XLA scatter (small ~32³ grids are the atomic-contention worst case). DHFR is a second cell after 1vii is green.

Ensemble `vmap` over molecules stays **outside** the kernel (one system per launch), matching GB Pallas.

## Alternatives considered

| ID | Alternative | Why not (or when) |
|----|-------------|-------------------|
| **XLA 4³ Python loop** | Trace-time unroll of 64 `Scatter` HLOs | Status quo before vectorization. Compile-fat and sequential atomics. Rejected as the production stencil. |
| **XLA vectorized `(N, 64)` scatter/gather** | One scatter-add, one gather; shared `_bspline4_stencil` | **Accepted as the CPU / fallback / interpret path** and as the parity oracle. Rejected as the GPU performance ceiling: still generic scatter, stencil in HBM. |
| **xtrax `WhileCarry` / `Scan` over 64 offsets** | Carry `Q` for 64 serial scatters | Wrong axis. `WhileCarry` is the N_STEPS inference loop (B1-INFER), carry-only, not reverse-AD safe. Scan AD is unnecessary because the VJP is gather. Serializing 64 reductions does not become a register spline. |
| **`vmap` over 64 offsets** | Easy to sum 64 full grids | Memory blowup at DHFR-scale `(K,K,K)`. Rejected. |
| **C. Pallas on pre-materialized `(N, 64)` weights** | Kernel only the atomic loop | Low complexity, little win vs XLA scatter. Rejected as the target; the weights must live in registers. |
| **B. Grid-tile owner (no / fewer atomics)** | Each block owns a mesh tile; loop overlapping atoms | Higher ceiling, different algorithm (atom-to-tile lists). Deferred as a **second** project if A loses on contention. |
| **Custom CUDA / OpenMM plugin** | Hand-written spline + cuFFT | Leaves the JAX program. Not the Prolix kernel strategy; Pallas stays in-trace. |
| **Replace FFT in the same change** | Pallas or cuFFT custom-call | Out of scope. Reciprocal space has two bottlenecks; mix them and the A/B is unreadable. |
| **f16 / bf16 / fp8 spline** | Mixed-precision compute | JAX can set dtypes; OpenMM mixed precision is **f32 forces + mixed integration**, not fp16 PME. Positions/SETTLE need more than f16 ULP. Revisit only after f32 A is gated. f64 on GPU is a known throughput trap; Stage-2 1vii records were already `x64_enabled=false`. |

## Consequences

- New module (proposed): `src/prolix/physics/pme_pallas.py`. Branch in [`pme.py`](../../src/prolix/physics/pme.py). Tests: interpret + GPU parity vs stencil oracle.
- Production default remains the **vectorized XLA stencil** until the 1vii force-only gate passes. A loss on 1vii is a recorded result, not a silent fallback bug.

## Implementation note (2026-08-24)

Opt-in path is in tree: `src/prolix/physics/pme_pallas.py`, `use_pallas_pme` / `PROLIX_USE_PALLAS_PME`, interpret tests green on CPU. **Default is still XLA.** L40S A/B is ungated explore array `scripts/slurm/prof_pme_pallas.slurm` (force XLA vs Pallas vs MD Pallas vs Q dump). Do not record a default flip or ms/day claim until those records are pulled.
- GB Pallas is **not** a scatter kernel; Mosaic `atomic_add` + 3D wrap is a new failure mode and needs both `interpret` and one L40S compile.
- `n_pad` and `(Kx, Ky, Kz)` are compile keys (already true of atom buckets / `compute_pme_grid_dims`).

## Out of scope

- Replacing FFT with a Pallas or cuFFT custom call.
- Grid-tile / no-atomic spread (Approach B).
- Changing Ewald α, grid snapping, or Flash/NL direct space.
- Making Flash the cutoff default because NL lost a race (lesson 379).
- Claiming µs/day or OpenMM CUDA ns/day from kernel-only cells.

## Risks (first spike)

- **Atomic contention** on small grids (1vii ~32³ worst; DHFR mesh is easier).
- Mosaic GPU `atomic_add` into a 3D wrapped grid.
- Static-shape compile explosion if `grid_dims` leak as traced values (existing PME debt: dims must stay Python ints).

## Related

- Vectorized stencil: `_bspline4_stencil` in `pme.py`; `tests/physics/test_pme_stencil_vectorized.py`.
- GB Pallas precedent: `src/prolix/pallas_kernels.py`.
- Stage-2 apply-fn probe: `scripts/explore/prof_ensemble_apply_probe.py`.
