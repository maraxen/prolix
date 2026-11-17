# Pallas v3.5 Megakernel Architecture

Classical molecular dynamics is bottlenecked not just by raw floating-point arithmetic, but by **HBM bandwidth**, **kernel launch latency** (~3–8 µs per launch × thousands of timesteps), and **per-process GPU context overhead** when scaling across ensembles of molecules.

Prolix's **Pallas v3.5** execution stack (`MK-43` through `MK-74`) eliminates host-device round-trips and dense $O(N^2)$ pair allocations by fusing spatial tiling, direct-space electrostatics + Lennard-Jones, smooth particle-mesh Ewald (SPME), Generalized Born implicit solvation, and constrained Langevin integration directly into Triton-backed JAX Pallas GPU kernels.

---

## 1. End-to-End Explicit-Solvent Timestep Pipeline

```{mermaid}
flowchart TD
    SKIN["GPU Morton 30-Bit Z-Order Sort & O(N_tiles) AABB Skin Check<br/>(Rebuilds 32x32 Tile-Pair List Only When max|Δr| > skin/2)"]
    BOND["Fused Bonded Megakernel (pallas_bonded_all_fused)<br/>Harmonic Bonds • Angles • Periodic/RB Torsions • Bicubic CMAP • 1-4 Exceptions"]
    DIRECT["Morton 32x32 Newton-Pair Direct Nonbonded (MK-51 / MK-65)<br/>SRAM Shared-Memory Tile Broadcast • Bitmask 1-2/1-3/1-4 Exclusions<br/>Switched LJ 12-6 + erfc(αr)/r Real-Space Ewald"]
    SPME["Smooth 2^a 3^b 5^c SRAM SPME (MK-54 / MK-70)<br/>Tile-Local Shared-Memory B-Spline Spread -> Batched 3D rfft/irfft -> Fused Scaled Gather"]
    INTEG["Persistent Inter-Step Fused Integrator (MK-46 / MK-58 / MK-66)<br/>BAOAB Splitting • Analytical Water SETTLE • Solute SHAKE/RATTLE • Philox-2x32 Noise"]

    SKIN --> BOND
    SKIN --> DIRECT
    SKIN --> SPME
    BOND --> INTEG
    DIRECT --> INTEG
    SPME --> INTEG
    INTEG -->|"Loop nl_update_every steps on GPU"| BOND
```

---

## 2. Core Pallas v3.5 Innovations (`MK-51` .. `MK-74`)

### 2.1 Morton $32 \times 32$ Tile-Pairs & $O(N_{\text{tiles}})$ Spatial Rebuild (`MK-51`, `MK-65`, `MK-71`)

Instead of maintaining a per-atom `(N, K_max)` Verlet neighbor table—which consumes `76.8 MB/replica` at 100k atoms and incurs uncoalesced random gathers—Prolix sorts real atoms along a **30-bit 3D Morton Z-order space-filling curve** and groups consecutive 32-atom blocks into compact **axis-aligned bounding boxes (AABBs)**:

- **Tile-level pruning (`MK-71`)**: A GPU spatial hash / cell-binning kernel tests AABB-to-AABB minimum-image distances against $r_{\text{cut}} + r_{\text{skin}}$ in **$O(N_{\text{tiles}})$** time (`N_tiles = ceil(N / 32)`), scaling cleanly past **408k atoms** (`Cellulose_408k_pme`) without $O(N_{\text{tiles}}^2)$ memory spikes.
- **Register + SRAM tile evaluation (`MK-51`, `MK-65`)**: Each CTA loads 32 $i$-atoms into registers and streams interacting 32-atom $j$-tiles through shared memory, evaluating $32 \times 32 = 1{,}024$ distances in warp-synchronous SIMD while decoding 32-bit topological exclusion bitmasks (`1-2`, `1-3`, and `1-4` pairs) in registers.
- **Multi-tile CTA packing (`tp_group`, `MK-63`)**: On compact biomolecular buckets ($N \le 4{,}096$), packing `2`, `4`, or `8` target tiles per CTA amortizes launch overhead and maximizes warp occupancy.

### 2.2 Tile-Local SRAM SPME Spread & Gather (`MK-54`, `MK-70`)

Standard particle-mesh Ewald charge spreading hammers global HBM with $4^3 = 64$ conflicting `atomicAdd` operations per atom. Because Morton-sorted 32-atom tiles occupy spatially localized sub-volumes (~$6 \times 6 \times 6$ mesh nodes), `pme_pallas.py` (`MK-70`) accumulates B-spline weights inside an **on-chip SRAM bounding sub-brick** first and flushes consolidated non-zero mesh nodes to HBM once per tile. Reciprocal convolution uses smooth **$2^a 3^b 5^c$** FFT dimensions (`snap_fft_dim`), absorbing the $-\frac{1}{2}$ reciprocal force factor directly into the reciprocal influence tensor to save a full grid pass.

### 2.3 Persistent Inter-Step Fused BAOAB + SETTLE + RATTLE (`MK-46`, `MK-58`, `MK-66`)

Between neighbor-skin checks (`nl_update_every = 10..25` steps), `EnsemblePlan` and `ProlixSimulation` fuse the entire **BAOAB** Langevin integrator step:

1. **B**: Half-step momentum kick $p \leftarrow p + \frac{\Delta t}{2} F(r)$ + **RATTLE** velocity projection.
2. **A**: Half-step drift $r \leftarrow r + \frac{\Delta t}{2} m^{-1} p$.
3. **O**: Exact Ornstein-Uhlenbeck thermostat $p \leftarrow c_1 p + c_2 \sqrt{m} \, \xi$ using counter-based **Philox-2x32** Gaussian deviates inside the kernel (`c_1 = e^{-\gamma \Delta t}`, `c_2 = \sqrt{k_B T (1 - c_1^2)}`).
4. **A**: Second half-step drift + analytical 3-atom triangle **SETTLE** for rigid water and iterative **SHAKE** for constrained solute X–H bonds.
5. **B** *(next step)*: Reuses the freshly evaluated forces without re-reading positions/momenta from host or launching separate constraint kernels.

In **`precision="mixed"` (`MK-66`)**, positions, momenta, and SETTLE/RATTLE triangle geometry remain in **IEEE `float64`** registers to eliminate coordinate rounding drift, while force/SPME inputs are pre-cast to `float32` so direct-space and reciprocal kernels run at full Tensor/FP32 throughput.

### 2.4 Two-Pass $32 \times 32$ Tile OBC2 GBSA (`MK-67`)

For implicit-solvent folding and peptide screening (`src/prolix/physics/pallas_gbsa_tile.py`), Prolix replaces $O(N^2)$ dense JAX arrays with a **two-pass symmetric $32 \times 32$ tile engine**:

- **Pass 1 (Born Descreening)**: Accumulates pairwise HCT descreening integrals $I_{ij}$ across upper-triangular tiles, applies the OBC-II $\tanh(\alpha \Psi - \beta \Psi^2 + \gamma \Psi^3)$ rescaling to obtain effective Born radii $R_i$, and computes exact ACE nonpolar SASA energies $E_{\text{ACE},i} = 4\pi \sigma_{\text{np}} (r_i + r_p)^2 (r_i / R_i)^6$ plus Born chain-rule multipliers $\partial E / \partial R_i$.
- **Pass 2 (Polar + Born Chain-Rule Forces)**: Evaluates Generalized Born pair electrostatics (with optional Debye-Hückel salt screening $\kappa_{\text{eff}} = 0.73 \kappa$) alongside the exact $\frac{\partial E}{\partial R_i} \frac{\partial R_i}{\partial r_{ij}}$ Born-radius derivative forces in a single tile sweep, backed by an analytical `@jax.custom_vjp` rule.

### 2.5 `stride = 0` Congeneric Deduplication (`MK-64`) & 1D Ragged CTA Queue (`MK-68`)

- **Congeneric ligand/mutant screens (`MK-64`)**: When simulating $R$ ligands bound to a shared receptor, bond/angle/torsion indices, water indices, and exclusion bitmasks are identical across the receptor core. Prolix detects replica-invariant arrays and binds them with **`stride = 0`** across the batch axis—allocating topology tables **once** in VRAM while varying ligand charges, LJ parameters, and coordinates per replica.
- **Cross-size ragged screens (`MK-68`, `pallas_ragged_queue.py`)**: When a batch mixes systems of wildly different atom counts (e.g., `2,100` to `45,000` atoms), naive padding to $\max_r N_r$ wastes $>80\%$ of SMs on ghost atoms. Prolix instead computes a **1D prefix-sum tile offset table** across all systems and launches a single flat CTA grid where every block binary-searches its owning replica $(r, \text{tile}_i)$ in $O(\log R)$ registers, paired with cost-modeled multi-tier SPME (`pme_tier_mode="auto"`).

---

## 3. Memory Moat vs NVIDIA `CUDA MPS`

Because native OpenMM `Context` instances cannot batch distinct topologies inside a single kernel launch, multi-ligand screening in OpenMM typically launches $R$ worker processes sharing one GPU via **NVIDIA Multi-Process Service (`CUDA MPS`)**:

| Metric (`DHFR` ~23.5k atoms, 48 GB L40S GPU) | OpenMM CUDA + `CUDA MPS` | Prolix v3.5 (`ProlixBatchSimulation` / `EnsemblePlan`) |
| :--- | :--- | :--- |
| **Per-Replica Context + Plan Overhead** | ~450 – 580 MB / process | **~45 – 55 MB / replica** (shared XLA executable + `stride=0` topology) |
| **Max Concurrent Replicas on 48 GB VRAM** | OOM cliff at $R \approx 64$ | **$R = 128+$** comfortably resident |
| **Ragged Heterogeneous Topologies** | $R$ separate process launches & scheduler contention | **1 fused 1D ragged CTA work queue** |
| **Analytical $\nabla_{r, q, \sigma, \varepsilon} E$** | Finite differences / host transfers | **In-graph `@jax.custom_vjp` ($O(N \cdot K)$ VRAM)** |

---

## 4. GPU Architecture Compiler Guardrails (`sm_80` – `sm_120`)

{func}`prolix.export.resolve_pallas_compiler_params` automatically tunes Triton compilation parameters across NVIDIA generations:

| GPU Family | Compute Capability | Bonded Fused | Megakernel / SETTLE | Nonbonded Tiles / SPME |
| :--- | :--- | :--- | :--- | :--- |
| **Ampere** (`A100`, `RTX 3090`) | `sm_80` (`8.0`) | `warps=2, stages=1` | `warps=4, stages=1` | `warps=4, stages=2` (`1` if `<256` blocks) |
| **Ada Lovelace** (`L40S`, `RTX 4090`) | `sm_89` (`8.9`) | `warps=2, stages=1` | `warps=4, stages=1` | `warps=4, stages=2` (`1` if `<256` blocks) |
| **Hopper** (`H100`, `H200`, `GH200`) | `sm_90` (`9.0`) | `warps=2, stages=1` | `warps=4, stages=1` | `warps=4, stages=2` (`1` if `<256` blocks) |
| **Blackwell** (`RTX PRO 6000`, `B200`) | `sm_120` (`10.0+`) | `warps=2, stages=1` | `warps=4, stages=1` | `warps=4, stages=1` *(prevents `.local` stack spill)* |
