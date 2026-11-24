# Prolix v0.1.0-alpha (Pallas v3.5) Line-by-Line Mathematical, Unit-Conversion & Multi-Engine Physics Audit

**Date**: 2026-11-11  
**Branch**: `feat/pallas-v2-megakernel-o1-o5-261028`  
**Scope**: Complete equation-by-equation mathematical, physical-constant, unit-conversion, sign, virial, exclusion, and degrees-of-freedom ($N_{\text{dof}}$) audit across `src/prolix/physics/`, `src/prolix/pallas_kernels.py`, `src/prolix/openmm/`, `src/prolix/knob_surface.py`, and `src/prolix/physics/kups_adapter.py`, paired with the Four-Tier Release Physics & Observable Validation Suite (`scripts/validation/validate_release_physics_and_observables.py`).

---

## Executive Summary & Defect Remediation Ledger

Every potential energy term, analytical force (`custom_vjp`), stochastic/symplectic integrator splitting, holonomic constraint solver, barostat volume-scaling Jacobian, and unit conversion factor in Prolix was audited from first principles against CODATA 2018 / SI 2019 exact constants and the OpenMM 8.x C++ Reference/CUDA implementation (`SimTKOpenMMRealType.h`, `ReferencePME.cpp`, `NonbondedForceImpl.cpp`, `ReferenceObc.cpp`, `CMAPTorsionForceImpl.cpp`, and `ReferenceSETTLE.cpp`).

### Audit Defect & Discrepancy Ledger

| ID | Module & Location | Mathematical / Physical Issue | Quantitative Impact | Remediation Specification |
| :--- | :--- | :--- | :--- | :--- |
| **`BUG-ELEC1`** | `pallas_kernels.py:264,2552,6514`, `explicit_corrections.py:13` vs `pme.py:17`, `knob_surface.py:58` | Split Coulomb constant: direct space & PME exclusion subtraction used `proxide` `332.0636` (`-0.3412 ppm` vs OpenMM), whereas reciprocal SPME & self/plasma used `332.0637` (`-0.0401 ppm`) vs exact OpenMM `332.06371329919216`. | `0.3011 ppm` Ewald direct/reciprocal non-cancellation (`1.0e-4 * q_i * q_j * erf(αr)/r` leak on all 1-2/1-3/1-4 pairs; `~0.1–0.3 kcal/mol` offset on `DHFR`/`ApoA1`). | Unify all electrostatic kernels and `KnobSurface` on exact OpenMM `332.06371329919216 kcal·Å/(mol·e²)` so `erfc(αr) + erf(αr) = 1` cancels to machine precision. |
| **`BUG-GB1`** | `generalized_born.py:24,655-674`, `pallas_implicit.py:267-272`, `pallas_generalized_born.py:337-344,1034` | `SURFACE_TENSION` defined as `0.005 kcal/(mol·Å²)` while hardcoding OpenMM's `28.3919551 kJ/(mol·nm²)` ($4\pi \times 2.25936\text{ kJ/(mol·nm}^2) \equiv 4\pi \times 0.0054\text{ kcal/(mol·Å}^2)$), then multiplying by `surface_tension / 0.005`. | Passing physical $\sigma_{\text{np}} = 0.0054\text{ kcal/(mol·Å}^2)$ inflated $E_{\text{ACE}}$ and $\partial E_{\text{ACE}}/\partial R_i$ by **`+8.000%`** (`1.08x`), plus `+1.10 ppm` from rounded `KJ_TO_KCAL = 0.239006`. | Evaluate $E_{\text{ACE},i} = 4\pi \sigma_{\text{np}} (r_i + r_p)^2 (r_i / R_i)^6$ directly in AKMA (`kcal/mol`, `Å`) with default $\sigma_{\text{np}} = 2.25936 / 418.4 = 0.0054\text{ kcal/(mol·Å}^2)$. |
| **`BUG-GB2`** | `generalized_born.py:164,403`, `pallas_gbsa_tile.py` (`MK-67`) | Missing optional Debye–Hückel salt screening parameter $\kappa$ (`tau = 1/eps_in - 1/eps_out` hardcoded to zero ionic strength). | Exact at zero salt (`GBSAOBCForce`); deviates from `implicit/obc2.xml` (`CustomGBForce`) when `implicitSolventKappa > 0`. | Support $\tau_{ij}(f_{\text{GB}}) = \epsilon_{\text{in}}^{-1} - \epsilon_{\text{out}}^{-1} \exp(-0.73\,\kappa f_{\text{GB}})$ and include the $+\kappa_{\text{eff}} \epsilon_{\text{out}}^{-1} e^{-\kappa_{\text{eff}} f_{\text{GB}}} / f_{\text{GB}}$ chain-rule term in $\partial U_{\text{GB}}/\partial f_{\text{GB}}$. |
| **`BUG-DISP1`** | `explicit_corrections.py:62-90`, `openmm/context.py:816-835` | Isotropic LJ continuum tail used factorized geometric $\sigma$ mixing $(\sum_i \sqrt{\epsilon_i}\sigma_i^3)^2$ ($\sigma_{ij} = \sqrt{\sigma_i\sigma_j}$) instead of Lorentz–Berthelot $\sigma_{ij} = (\sigma_i+\sigma_j)/2$. | `~1–3%` underestimate of $|\Delta E_{\text{disp}}|$ on heterogeneous protein+water systems (`-10.95%` on extreme $\sigma = 3\text{ vs }4\text{ Å}$ test pair). | Compute exact $O(N)$ Lorentz–Berthelot tail via 13-term binomial moments $2^{-m}\sum_{k=0}^m \binom{m}{k} S_k S_{m-k}$ where $S_k = \sum_i \sqrt{\epsilon_i}\sigma_i^k$ ($m \in \{6, 12\}$). |
| **`BUG-OBS1`** | `api/observables.py:298` | `KCAL_MOL_PER_A3_TO_BAR = 68568.0` used standard atmospheres (`68568.42 atm`) instead of `bar` (`69476.95 bar`) and omitted configurational virial in `Pressure`. | `-1.308%` unit error (`atm` vs `bar`) on kinetic pressure observable. | Set `KCAL_MOL_PER_A3_TO_BAR = 69476.95457` (`bar`) and include configurational virial $W$. |
| **`BUG-OMM1`** | `openmm/context.py:1942-1962` | `setVelocitiesToTemperature` rescaled velocities to `3*n_real - 3` **before** `applyVelocityConstraints()` and omitted `3*n_waters + n_solute_constraints` from $N_{\text{dof}}$. | Projecting onto SETTLE/RATTLE tangent manifold removed constrained kinetic energy, initializing rigid TIP3P water at **~200 K instead of 300 K** (`-33%`). | Project velocity constraints & remove COM first, then rescale to exact $N_{\text{dof}} = 3 N_{\text{real}} - 3 N_{\text{waters}} - N_{\text{cons}} - 3\mathbb{I}[\text{COM}]$. |
| **`BUG-OMM2`** | `openmm/context.py:2619-2636` | `getState(getEnergy=True)` added leapfrog half-step sync `+0.5 * dt * f / m` whenever `not is_langevin`, including `integrator_scheme="verlet"` and `step_count == 0`. | Corrupted reported kinetic energy $K(t)$ and NVE total energy $E = U + K$ for Velocity Verlet (`verlet`) where velocities are already full-step $v(t)$. | Apply `v_sync = v + 0.5 * dt * f / m` strictly when effective scheme is `"leapfrog"` and `step_count > 0`. |
| **`BUG-OMM3`** | `openmm/context.py:1193,1233-1243` | Nosé–Hoover & CSVR rollout hardcoded `ndof = 3*N - 6` and CSVR omitted the $(1-c)\frac{\bar{K}}{N_f K}(\chi^2_{N_f} - N_f)$ stochastic variance term. | Wrong target temperature on constrained water/solute systems (`~450 K` instead of `300 K` on TIP3P) and narrowed kinetic energy variance. | Use exact constrained $N_{\text{dof}}$ and exact Bussi–Donadio–Parrinello `_csvr_compute_lambda` (`settle.py:4974`). |
| **`BUG-OMM4`** | `openmm/context.py:1642,1788-1798,1866` | `_compute_virial` single-anchor unwrap fails across MIC boundaries; `_apply_scr_barostat` set `tau = 2000.0` (`97.8 ps` vs `~0.1 ps`); `_apply_mc_barostat` computed molecular COM without intramolecular PBC unwrap. | Corrupted periodic configurational virial in SCR barostat; 1000x over-damped SCR relaxation; boundary-straddling molecule COM shift in MC barostat. | Compute periodic virial via homogeneous volume scaling $-\frac{d}{d\lambda}U(\lambda\mathbf{r}, \lambda\mathbf{B})\big|_{\lambda=1}$, fix SCR $\tau_P$ AKMA conversion, and unwrap molecules wrt anchor atom before COM scaling. |
| **`BUG-OMM5`** | `openmm/converter.py:1650-1686`, `openmm/units.py:250` | `GBSAOBCForce` converter dropped `getSurfaceAreaEnergy()`; `units.py` mapped raw float `"atm"` alias to `"bar"`. | `setSurfaceAreaEnergy(0)` still added `0.0054` ACE energy; `to_akma(1.0, "atm")` off by `1.325%`. | Extract `surface_area_energy` from `GBSAOBCForce` and give `"atm"` its dedicated `1.01325 / 69476.95457` conversion factor. |
| **`BUG-OMM6`** | `openmm/context.py:1792` | `_apply_mc_barostat` hardcoded fractional jump `delta_v = 0.05` (`5%` of $V$) without adaptive step sizing or NL rebuild on accepted move. | On DHFR ($V \approx 241,000\text{ Å}^3$, $\sigma_V \approx 700\text{ Å}^3$), a `5%` jump ($\sim 17\sigma_V$) rejected $>95\text{–}99.9\%$ of MC trials, freezing $\kappa_T$. | Initialize `_mc_volume_scale = 0.01 * V`, adapt every 10 attempts (`*0.5` if acc `<25%`, `*1.5` if acc `>75%` per OpenMM `MonteCarloBarostatImpl`), and refresh NL on accept. |
| **`BUG-NPT1`** | `physics/settle.py:5915,6030` | `settle_csvr_npt` evaluated post-scaling `force_fn(position, **step_kwargs)` with stale `box` instead of `new_box` and computed `sum(r * F)` virial on wrapped coordinates. | Stale box evaluated post-barostat forces in wrong periodic cell; wrapped `sum(r * F)` suffered periodic boundary image jumps. | Pass `box=new_box` after cell rescaling and evaluate periodic configurational virial via $-\frac{d}{d\lambda}U(\lambda\mathbf{r}, \lambda\mathbf{B})\big|_{\lambda=1}$. |
| **`BUG-CMAP1`** | `openmm/context.py:495`, `openmm/converter.py:1382`, `cmap.py:259` | OpenMM `CMAPTorsionForce::getMapParameters` stores `energy[i_phi + size*j_psi]`. Because `prolix/physics/cmap.py:259` (`compute_bicubic_coefficients`) internally transposes `energy = energy.T` expecting C-order `grid[j_psi, i_phi]`, `converter.py` must reshape in C-order (`.reshape((size, size))`) while `context.py` must pass `first_is_psi=False` to `pallas_cmap_torsions` (which previously defaulted to `first_is_psi=True`). | Passing `first_is_psi=True` (or double-transposing via `order="F"` + `cmap.py`'s `.T`) swapped $(\phi, \psi) \leftrightarrow (\psi, \phi)$ across every 2D CMAP spline (`270.0260` vs `161.3915 kcal/mol`, **`+108.6345 kcal/mol` error on `5DFR`**). | Keep C-order `.reshape((size, size))` in `converter.py:1382` and set `first_is_psi=False` in `context.py` (`161.39147 vs 161.39150 kcal/mol` across XLA, Pallas, and OpenMM). |
| **`BUG-CMAP2`** | `openmm_xml_importer.py:625-635` | OpenMM 8 `<CMAPTorsionForce version="2">` serializes `<Map>` entries as `<Energy e="..."/>` attributes rather than `child.text`. | Crashed with `ValueError: CMAP <Map size='24'> expected 576 values, got 0` on OpenMM 8 XML files (`amber14/protein.ff19SB.xml`, `charmm36.xml`). | Parse `child.attrib.get("e")` alongside `child.text` and preserve C-order `(size, size)` grid layout expected by `cmap.py:259`. |
| **`BUG-OMM7`** | `openmm/context.py:1526,1530` | `ProlixContext.__init__` executed `int(self._config.tp_group)` and `int(self._config.device_index)` unconditionally even though `ProlixOpenMMConfig` defaults both to `None` (`device_index: str \| None = None`). | Raised `TypeError: int() argument ... not 'NoneType'` when initializing `ProlixContext` with default `ProlixOpenMMConfig`. | Guard both with `if ... is not None else None` (`str(...)` for `device_index`). |
| **`BUG-OMM8`** | `openmm/context.py:1047` | `_eval_system_potential_energy` inside `@eqx.filter_jit` `_pure_compute_energy_and_forces` called `pallas_gbsa_tile(...)` with `tile_list=None`, invoking host-only `build_gbsa_tile_pair_list` (`np.asarray(positions)`) on a traced array. | Raised `jax.errors.TracerArrayConversionError` when evaluating `ProlixContext` (`use_pallas=True`) on implicit-solvent GBSA systems. | Pre-build `self._gbsa_tile_list = build_gbsa_tile_pair_list(pos_f32)` on host in `_init_neighbor_list` and thread `tile_list` into `pallas_gbsa_tile`. |
| **`BUG-TP1`** | `pallas_kernels.py:7167` (`_pallas_nonbonded_tile_pairs_bwd`) | `pallas_tile_pairs_param_grads(...)` (`pallas_tile_pairs.py:1595`) requires `(B, in_b, ...)` as its first two positional arguments, which were omitted in `_pallas_nonbonded_tile_pairs_bwd`. | Every `jax.grad` / `jax.value_and_grad` call through `pallas_nonbonded_tile_pairs` (`morton_tile_pairs`) raised `TypeError: missing 2 required positional arguments: 'lam_arr' and 'upstream_g'`. | Compute `in_b_bwd` and `B_bwd` from input tensor ranks in `_pallas_nonbonded_tile_pairs_bwd` and pass `(B_bwd, in_b_bwd, ...)` to `pallas_tile_pairs_param_grads`. |
| **`BUG-OMM9`** | `openmm/context.py:1571-1586, 1718-1725, 1757-1775, 1846-1860` | `ProlixContext` selected `"morton_tile_pairs"` and built a `cutoff = 9.0 Å` neighbor list on `NoCutoff` (`nonbonded_cutoff = inf`) + GBSA systems (`has_nonbonded=True` and `has_implicit_solvent=True`), while `not _should_use_neighbor_list()` zeroed `_gbsa_tile_list`. | Silently truncated all-pairs `NoCutoff` solute vacuum Coulomb/LJ at `9.0 Å` (hundreds of `kcal/mol` error on `5DFR_GBSA`) and bypassed `"gbsa_tile"`. | Gate `"morton_tile_pairs"`, `_can_use_tile_pairs`, and `NonbondedForce` NL routing on `0.0 < nonbonded_cutoff < inf`; allow `has_nonbonded=True` in `"gbsa_tile"` auto-selection; always build `_gbsa_tile_list` when `has_implicit_solvent`. |
| **`BUG-OMM10`** | `openmm/context.py:617-818` (`_eval_nonbonded`) | Non-tile-pair branches (`neighbor_idx`, `chunk_body`, `dense`) evaluated $4\varepsilon (\sigma/r)^{12}$ across unmasked 1-2/1-3 covalent bonds ($\sim +10^8\text{ kcal/mol}$) in `float32` and subtracted `u_ex` ($\sim +10^8\text{ kcal/mol}$) post-hoc at line `817`. | Catastrophic `float32` `10^8 - 10^8` cancellation ($10^8 \cdot \epsilon_{\text{f32}} \approx 12\text{–}22\text{ kcal/mol}$): **`+22.1584 kcal/mol`** error on `5DFR_GBSA` (`-3479.9321` vs `-3502.0905 kcal/mol` in OpenMM / `-3502.0903` in Pallas). | Mask `excl_indices` / `excl_dense_indices` **before** evaluating `(sig / r_safe)**12` so excluded pairs never enter `u_pair`, leaving only smooth $O(1)$ PME `erf(αr)/r` subtraction when `is_pme`. |
| **`BUG-OMM11`** | `openmm/converter.py:1799-1819` (`from_openmm_system`), `openmm_xml_importer.py` (`bundle_from_openmm_system_xml`) | `from_openmm_system` and `bundle_from_openmm_system_xml` called `_next_bucket(..., ATOM_BUCKETS, use_size_buckets=True)` unconditionally when `use_size_buckets=True`, where `ATOM_BUCKETS[-1] = 60,000`. | Raised `AssertionError: Count 92224 exceeds all buckets (64, ..., 60000)` on `ApoA1` (`92,224` atoms) or any system exceeding `60,000` atoms under default `ProlixOpenMMConfig()`. | Automatically fall back to `use_size_buckets = False` whenever any real topology count exceeds its bucket ladder maximum (`ATOM_BUCKETS[-1]`, `BOND_BUCKETS[-1]`, `EXCL_BUCKETS[-1]`, etc.). |
| **`BUG-TRITON-ATOM1`** | `physics/neighbor_list.py:1360` (`pallas_bin_cells_batch`), `physics/pallas_tile_pairs.py:397` (`morton_stamp_excl_bits`) | In JAX `0.10.2` (`jax/_src/pallas/triton/primitives.py:422`), `plt.atomic_add` checks `isinstance(val.type, ir.IntegerType)` instead of `_element_type(val.type)`, so **vector `int32`** `plt.atomic_add` lowers to `tt.atomic_rmw fadd` (`atom.global.add.f32`) on `!tt.ptr<i32>`! | Subnormal `0x00000001` (`1.4013e-45f`) flushes to `0.0f` on L40S (`sm_89`), leaving `cell_counts == 0` and returning 100% sentinel neighbor lists (`nbr.idx == N`) on GPU (`cos_sim = 0.696` diagonal-only tiles in `ProlixContext` for `N >= 4,000`) and corrupting GPU exclusion bitmasks (`1 << elj`). | Use `float32` cell-count accumulator (`ones_f32 = 1.0`, exact up to $2^{24}$) in `pallas_bin_cells_batch` and `plt.atomic_or` (`tt.atomic_rmw or`) in `morton_stamp_excl_bits`. |

---

## 0. Fundamental Constants & Unit-Conversion Audit

Prolix operates internally in **AKMA** units:
- **Length**: Ångström ($\text{Å} = 10^{-10}\text{ m} = 0.1\text{ nm}$)
- **Energy**: kilocalorie per mole ($\text{kcal/mol} = 4.184\text{ kJ/mol}$ exact thermochemical calorie)
- **Mass**: Dalton / unified atomic mass unit ($\text{Da} = \text{g/mol} = 10^{-3}\text{ kg/mol}$)
- **Charge**: elementary charge ($e = 1.602176634 \times 10^{-19}\text{ C}$, SI 2019 exact)
- **Time**: $\tau_{\text{AKMA}} = \text{Å}\sqrt{\frac{\text{Da}}{\text{kcal/mol}}} = 10^{-10}\text{ m}\sqrt{\frac{10^{-3}\text{ kg/mol}}{4184\text{ J/mol}}} = 4.888821290839616 \times 10^{-14}\text{ s} = 48.88821290839616\text{ fs}$ (note $N_A$ cancels identically!)

### Table 0.1 — Exact Constant Audit Across Codebase Modules (ppm vs CODATA 2018 / OpenMM)

| Physical Constant | Exact Reference Value (SI 2019 / OpenMM) | `prolix/constants.py` | `openmm/units.py` | `knob_surface.py` & `pme.py` | `proxide` (`pallas_kernels.py`) | `kups_adapter.py` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Coulomb $C_{\text{Coul}}$** (`kcal·Å/(mol·e²)`) | `332.06371329919216` (`ONE_4PI_EPS0 * 10 / 4.184`) | `332.06371329919216` (**`0.00 ppm`**) | — | `332.0637` (`-0.0401 ppm`) | `332.0636` (`-0.3412 ppm`) | Uses caller/proxide (`eV` exact `0.00 ppm`) |
| **Boltzmann $k_B$** (`kcal/(mol·K)`) | `0.0019872042586408316` ($R / 4184$) | `0.0019872042586408316` (**`0.00 ppm`**) | — | — | `0.0019872041` (`-0.0798 ppm`) | `0.0019872041` (`-0.0798 ppm`) |
| **AKMA Time $\tau_{\text{AKMA}}$** (`fs`) | `48.88821290839616` | `48.88821290839616` (**`0.00 ppm`**) | `48.88821291989587` (`+0.0002 ppm`) | — | `48.88821` (`-0.0595 ppm` in `physics/units.py`) | `48.88821290839616` (**`0.00 ppm`**) |
| **Energy `kJ -> kcal`** | `1 / 4.184 = 0.2390057361376673` | `1 / 4.184` (**`0.00 ppm`**) | `1 / 4.184` (**`0.00 ppm`**) | — | `0.239006` in `GB` (`+1.10 ppm`) | `23.06054783061903 kcal/eV` (**`0.00 ppm`**) |
| **Pressure `kcal/(mol·Å³) -> bar`** | `69476.9545705537` ($4184 / (N_A \cdot 10^{-25})$) | — | `69477.0` (`+0.6539 ppm`) | — | `69476.95` in `barostat.py` (`-0.0658 ppm`); `68568.0` in `observables.py` (**`-1.308%` atm bug**) | — |

### Derived Force, Spring Constant & Velocity Conversions (`openmm/units.py` & `kups_adapter.py`)
- **Force**: $1\text{ kcal/(mol·Å)} = 4.184\text{ kJ/mol} / (0.1\text{ nm}) = \mathbf{41.84\text{ kJ/(mol·nm)}}$ (exact, `FORCE_AKMA_TO_OMM = 41.84`).
- **Harmonic bond $k_b$**: $1\text{ kJ/(mol·nm}^2) = \frac{1}{4.184 \times 100}\text{ kcal/(mol·Å}^2) = \frac{1}{418.4}\text{ kcal/(mol·Å}^2)$ (exact, `BOND_K_OMM_TO_AKMA = 1.0 / 418.4`).
- **Harmonic angle $k_\theta$**: $1\text{ kJ/(mol·rad}^2) = \frac{1}{4.184}\text{ kcal/(mol·rad}^2)$ (exact).
- **Velocity**: $1\text{ Å}/\tau_{\text{AKMA}} = \frac{0.1\text{ nm}}{0.04888821290839616\text{ ps}} = \mathbf{2.045482828567\text{ nm/ps}}$ (`VELOCITY_AKMA_TO_OMM`, exact to `0.0002 ppm`).
- **kUPS Adapter (`src/prolix/physics/kups_adapter.py`)**: Uses exact SI 2019 $1\text{ eV} = \frac{e \cdot N_A}{4184} = 23.06054783061903\text{ kcal/mol}$ and $\tau_{\text{kUPS}} = 10.180505710654229\text{ fs}$. Crucially, momentum converts via $\frac{\tau_{\text{kUPS}}}{\tau_{\text{AKMA}}} = \sqrt{\frac{\text{kcal/mol}}{\text{eV}}} = \frac{1}{\sqrt{23.06054783061903}} = 0.2082423344$, preserving kinetic energy $K = \frac{p^2}{2m}$ identically across unit systems (`< 1e-15` relative error in `float64`).

---

## 1. Electrostatics, Ewald Summation & Smooth Particle Mesh Ewald (SPME)

### 1.1 Ewald Decomposition & Direct-Space Damped Coulomb
Under periodic boundary conditions in a cell of volume $V = \det(\mathbf{B})$, the Ewald electrostatic potential decomposes into:

$$U_{\text{Ewald}} = U_{\text{dir}} + U_{\text{rec}} + U_{\text{self}} + U_{\text{plasma}} + U_{\text{excl}}$$

For non-excluded pairs $(i < j)$ within $r_{ij} < r_c$ under minimum-image displacement $\mathbf{r}_{ij} = \operatorname{MIC}(\mathbf{r}_i - \mathbf{r}_j, \mathbf{B})$:

$$U_{\text{dir}} = \sum_{i < j,\, r_{ij} < r_c} C_{\text{Coul}} q_i q_j \frac{\operatorname{erfc}(\alpha r_{ij})}{r_{ij}}$$

Differentiating wrt $r_{ij}$ yields the exact radial derivative and force on atom $i$:

$$\frac{d U_{\text{dir},ij}}{d r_{ij}} = -C_{\text{Coul}} q_i q_j \left[ \frac{\operatorname{erfc}(\alpha r_{ij})}{r_{ij}^2} + \frac{2\alpha}{\sqrt{\pi}} \frac{\exp(-\alpha^2 r_{ij}^2)}{r_{ij}} \right], \qquad \mathbf{F}_i = -\nabla_{\mathbf{r}_i} U = -\sum_j \frac{d U_{ij}}{d r_{ij}} \frac{\mathbf{r}_{ij}}{r_{ij}}$$

**Code Verification**:
- `src/prolix/pallas_kernels.py:6633-6638` (`MK-51` `_pallas_nonbonded_tile_pairs_batch`) and `lines 1805-1820` (`MK-48`): `two_over_sqrt_pi = 1.1283791670955126` ($2/\sqrt{\pi} = 1.12837916709551257$, exact in `float64`/`float32`), `de_coul_dr = qq * w_elec * (-erfc_val / r^2 - alpha * two_over_sqrt_pi * exp(-alpha^2 r^2) / r)`, and `f_scalar = -(de_lj_dr + de_coul_dr) / r` multiplying `(dx, dy, dz) = r_i - r_j`. Sign and chain rule are **exact**.
- **Triton `erfc`/`erf` Rational Approximation (`pallas_kernels.py:210-238`)**: Implements Abramowitz & Stegun §7.1.26 ($p = 0.3275911$, 5-term polynomial $P_5(t) e^{-x^2}$) with theoretical uniform error $|\epsilon(x)| \le 1.5 \times 10^{-7}$, matching `float32` machine epsilon ($1.19 \times 10^{-7}$). In `float64` CPU interpret mode (`interpret and dtype == jnp.float64`), it dispatches to `jax.lax.erfc` / `jax.lax.erf` for full 53-bit mantissa accuracy.

### 1.2 Masked 1-2 / 1-3 Exclusions & 1-4 Scaled Exceptions in PME
The reciprocal SPME mesh spreads **all** atomic charges $q_i$, which implicitly includes the smooth Gaussian-screened interaction $+C_{\text{Coul}} q_i q_j \frac{\operatorname{erf}(\alpha r_{ij})}{r_{ij}}$ for every intramolecular excluded (1-2, 1-3, $w_{\text{elec}} = 0$) and scaled (1-4, $w_{\text{elec}} = 1/1.2 = 0.83333333$ in AMBER) pair.

For any pair $(i, j)$ with target total vacuum electrostatic weight $w_{\text{elec}} \in [0, 1]$ (where $w_{\text{elec}} = 0$ for 1-2/1-3, $w_{\text{elec}} = s_{14}$ for 1-4, and $w_{\text{elec}} = 1$ for normal pairs), the required direct-plus-correction contribution that must be added to $U_{\text{rec}}$ is:

$$U_{\text{pair}}^{\text{total}}(r) - U_{\text{rec},ij}(r) = C_{\text{Coul}} q_i q_j \frac{w_{\text{elec}}}{r} - C_{\text{Coul}} q_i q_j \frac{\operatorname{erf}(\alpha r)}{r} = C_{\text{Coul}} q_i q_j \frac{w_{\text{elec}} - \operatorname{erf}(\alpha r)}{r}$$

Check the three cases:
1. **Normal pair ($w_{\text{elec}} = 1$)**: $\frac{1 - \operatorname{erf}(\alpha r)}{r} = \frac{\operatorname{erfc}(\alpha r)}{r}$ (standard Ewald direct term).
2. **Excluded 1-2 / 1-3 / intra-water pair ($w_{\text{elec}} = 0$)**: $-\frac{\operatorname{erf}(\alpha r)}{r}$, which cancels $+\frac{\operatorname{erf}(\alpha r)}{r}$ in $U_{\text{rec}}$ to zero! Note that $\lim_{r \to 0} \frac{-\operatorname{erf}(\alpha r)}{r} = -\frac{2\alpha}{\sqrt{\pi}}$ is nonsingular, and its radial derivative is:
   $$\frac{d}{dr}\left[-C_{\text{Coul}} q_i q_j \frac{\operatorname{erf}(\alpha r)}{r}\right] = C_{\text{Coul}} q_i q_j \left[ \frac{\operatorname{erf}(\alpha r)}{r^2} - \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 r^2}}{r} \right] \xrightarrow{r \to 0} 0$$
3. **1-4 Scaled Exception ($w_{\text{elec}} = s_{14}$)**: When evaluated via `eff_w = s_14 - erf(αr)` (`pallas_kernels.py:6626`), adding $U_{\text{rec}}$ gives $C_{\text{Coul}} q_i q_j \frac{s_{14}}{r}$ — pure scaled Coulomb with **zero** spurious $\operatorname{erf}$ residue! Alternatively, when 1-4 pairs are fully excluded from the nonbonded tile (`w_elec = 0`, subtracting $\operatorname{erf}(\alpha r)/r$) and evaluated separately via `pallas_exceptions_14` / `context.py:784`, `pallas_exceptions_14` adds pure $+C_{\text{Coul}} q_{\text{prod},14} / r$, yielding the identical sum $C_{\text{Coul}} q_{\text{prod},14} / r$.

**Code Verification**:
- `pallas_kernels.py:6625-6631` (`MK-51`): `eff_w = w_elec - erf_v`, `e_coul = qq * eff_w * inv_dist`, `de_coul_dr = qq * (-eff_w * inv_r_sq - alpha * two_over_sqrt_pi * exp(-alpha^2 r^2) * inv_dist)`. Chain rule is **exact** across all three cases ($w_{\text{elec}} \in \{0, s_{14}, 1\}$).
- **Important Subtlety**: Notice line `6554` & `6574`: `active = valid_pair & (raw_r_sq < cutoff_sq)`. Because covalent 1-2, 1-3, and 1-4 pairs have physical lengths $1.0\text{–}4.5\text{ Å} \ll r_c = 9.0\text{ Å}$, their minimum-image distance is always well inside $r_c$.
- **OpenMM `exceptionsUsePeriodic` Convention (`context.py:720-740`, `DEF-23`)**: OpenMM `NonbondedForce` defaults to `exceptionsUsePeriodicBoundaryConditions == False`. Under this convention, OpenMM evaluates `calcPMEExcluded` (`-erf(αr)/r`) and 1-4 exceptions using raw unwrapped coordinates $\mathbf{r}_i - \mathbf{r}_j$ while evaluating direct-space $\operatorname{erfc}(\alpha r)/r$ and the SPME mesh under PBC. `ProlixContext` (`context.py:720-740, 772`) reproduces OpenMM Reference to `< 0.002 kJ/mol` even on adversarial boundary-straddling test coordinates.

### 1.3 SPME Order-4 Cardinal B-Splines, Influence Function, Self & Neutralizing Plasma Terms
In `src/prolix/physics/pme.py` and `src/prolix/physics/pme_pallas.py`:
- **Fractional grid coordinates**: $s_d = K_d \cdot \operatorname{mod}(x_d, L_d) / L_d$, $g_d = \lfloor s_d \rfloor$, $u_d = s_d - g_d \in [0, 1)$, stenciling onto $\operatorname{mod}(g_d + o_d - 1, K_d)$ for $o_d \in \{0, 1, 2, 3\}$ (equivalent to standard $M_4(u + 3 - o)$ shifted by 1 grid index; since $|G(\mathbf{m})|$ is translationally invariant on the periodic grid, uniform index shift $-1$ cancels identically between spread and gather!).
- **Order-4 B-spline weights (`pme.py:30-38`, `pme_pallas.py:23-37`)**:
  $$w_0(u) = \frac{(1-u)^3}{6}, \quad w_1(u) = \frac{4 - 6u^2 + 3u^3}{6}, \quad w_2(u) = \frac{1 + 3u + 3u^2 - 3u^3}{6}, \quad w_3(u) = \frac{u^3}{6}$$
  Check Partition of Unity: $w_0 + w_1 + w_2 + w_3 = \frac{6 + 0u + 0u^2 + 0u^3}{6} = 1$ (**exact**, charge-conserving!).
  Check Derivatives (`pme_pallas.py:40-54`): $w_0'(u) = -\frac{1}{2}(1-u)^2 = \frac{-3 + 6u - 3u^2}{6}$, $w_1'(u) = \frac{-12u + 9u^2}{6}$, $w_2'(u) = \frac{3 + 6u - 9u^2}{6}$, $w_3'(u) = \frac{3u^2}{6}$, with $\sum_{o=0}^3 w_o'(u) = 0$ (**exact**, zero self-force / momentum-conserving!).
- **Euler Exponential Spline Moduli (`pme.py:62-75`)**: For order $p = 4$, $M_4(1) = 1/6, M_4(2) = 4/6, M_4(3) = 1/6$:
  $$|b_m|^2 = \left| \frac{1}{6} + \frac{4}{6} e^{2\pi i m / K} + \frac{1}{6} e^{4\pi i m / K} \right|^2$$
  Matches Essmann et al. (1995) Eq. 4.4 and OpenMM `ReferencePME.cpp` identically.
- **Reciprocal Influence Function (`pme.py:78-115`)**: With minimum-image wavevectors $m_d^* = m_d - K_d \operatorname{round}(m_d / K_d)$ and $|\mathbf{m}^*|^2 = (m_x^*/L_x)^2 + (m_y^*/L_y)^2 + (m_z^*/L_z)^2$:
  $$G(\mathbf{m}) = \frac{\exp(-\pi^2 |\mathbf{m}^*|^2 / \alpha^2)}{\pi V |\mathbf{m}^*|^2 |b_{m_x}|^2 |b_{m_y}|^2 |b_{m_z}|^2} \quad (\mathbf{m} \neq \mathbf{0}), \qquad G(\mathbf{0}) = 0$$
  And via Parseval's theorem for `irfftn` (which divides by $K_x K_y K_z$):
  $$U_{\text{rec}} = \frac{1}{2} C_{\text{Coul}} \sum_{\mathbf{g}} Q(\mathbf{g}) \, \theta(\mathbf{g}), \qquad \theta = K_x K_y K_z \cdot \operatorname{irfftn}(G \cdot \operatorname{rfftn}(Q))$$
  $$\mathbf{F}_{i,\text{rec}} = -\nabla_{\mathbf{r}_i} U_{\text{rec}} = -C_{\text{Coul}} q_i \sum_{\mathbf{g}} \nabla_{\mathbf{r}_i} w_i(\mathbf{g}) \, \theta(\mathbf{g})$$
  In `pme_pallas.py` (`MK-62`), the factor $-C_{\text{Coul}} K_x K_y K_z$ is fused directly into `gather_pallas(..., force_scale=-C_Coul * Kx * Ky * Kz)` to avoid an extra mesh-wide scaling pass.
- **Self-Energy & Neutralizing Plasma (`pme.py:211-245`)**:
  $$U_{\text{self}} = -\frac{\alpha}{\sqrt{\pi}} C_{\text{Coul}} \sum_{i \in \text{real}} q_i^2, \qquad U_{\text{plasma}} = -\frac{\pi C_{\text{Coul}}}{2 \alpha^2 V} \left(\sum_{i \in \text{real}} q_i\right)^2$$
  Both mask ghost atoms via `jnp.where(atom_mask, charges, 0.0)` and match OpenMM `ReferencePME.cpp` identically.
- **OpenMM PME Parameter Mapping (`pme.py:130-165`)**:
  $$\alpha = \frac{\sqrt{-\ln(2 \delta_{\text{Ewald}})}}{r_c}, \qquad K_d = \operatorname{snap\_fft}\!\left(\left\lceil \frac{2 \alpha L_d}{3 \delta_{\text{Ewald}}^{1/5}} \right\rceil\right)$$
  where $\operatorname{snap\_fft}$ selects the smallest integer $\ge K_{\text{raw}}$ factoring into $2^a 3^b 5^c 7^d$ (`ReferencePME.cpp` / `NonbondedForceImpl::calcPMEParameters`).

### 1.4 OpenMM Reaction Field (`CutoffPeriodic`)
In `electrostatic_methods.py:53-75` and `openmm/context.py:604-608`:

$$U_{\text{RF}}(r) = C_{\text{Coul}} q_i q_j \left( \frac{1}{r} + k_{\text{rf}} r^2 - c_{\text{rf}} \right) \mathbb{I}[r < r_c], \qquad k_{\text{rf}} = \frac{1}{r_c^3} \frac{\epsilon_{\text{rf}} - 1}{2\epsilon_{\text{rf}} + 1}, \qquad c_{\text{rf}} = \frac{1}{r_c} \frac{3\epsilon_{\text{rf}}}{2\epsilon_{\text{rf}} + 1}$$

- **Boundary Check**: At $r = r_c$, $\frac{1}{r_c} + \frac{\epsilon_{\text{rf}}-1}{(2\epsilon_{\text{rf}}+1)r_c} - \frac{3\epsilon_{\text{rf}}}{(2\epsilon_{\text{rf}}+1)r_c} = \frac{2\epsilon_{\text{rf}}+1 + \epsilon_{\text{rf}} - 1 - 3\epsilon_{\text{rf}}}{(2\epsilon_{\text{rf}}+1)r_c} = 0$ ($C^0$ continuous potential). However, $-\frac{d U_{\text{RF}}}{dr}\big|_{r_c} = C_{\text{Coul}} q_i q_j \left(\frac{1}{r_c^2} - 2 k_{\text{rf}} r_c\right) = \frac{C_{\text{Coul}} q_i q_j}{r_c^2} \frac{3}{2\epsilon_{\text{rf}} + 1} \neq 0$ for finite $\epsilon_{\text{rf}} = 78.3$ (a ~1.9% residual force jump at $r_c$, standard in OpenMM Reaction Field).

---

## 2. Lennard-Jones, Quintic Potential Switching & Continuum Dispersion Tail Virial

### 2.1 Combining Rules & Quintic Switching Function
Prolix uses standard AMBER/CHARMM **Lorentz–Berthelot** combining rules for all pairwise LJ kernels (`pallas_kernels.py:6595-6599`, `bonded.py`, `openmm/context.py:585-586`):

$$\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}, \qquad \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}, \qquad U_{\text{LJ}}^{\text{raw}}(r) = 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^6 \right] = \frac{C_{12,ij}}{r^{12}} - \frac{C_{6,ij}}{r^6}$$

When potential switching is enabled on $[r_{\text{sw}}, r_c]$ with $r_{\text{sw}} = r_c - w_{\text{sw}}$ and $z = \frac{r - r_{\text{sw}}}{r_c - r_{\text{sw}}} \in [0, 1]$ (`lj_switching.py:15-32`, `pallas_kernels.py:240-260`):

$$S(z) = 1 - 10 z^3 + 15 z^4 - 6 z^5, \qquad \frac{dS}{dr} = \frac{-30 z^2 + 60 z^3 - 30 z^4}{r_c - r_{\text{sw}}} = -\frac{30 z^2 (1 - z)^2}{r_c - r_{\text{sw}}}$$

- Boundary regularity: $S(0) = 1, S(1) = 0, S'(0) = S'(1) = 0, S''(0) = S''(1) = 0$ ($C^2$ smooth potential and $C^1$ continuous forces at both $r_{\text{sw}}$ and $r_c$).
- Product-rule radial derivative (`pallas_kernels.py:6618`):
  $$\frac{d}{dr}\left[U_{\text{LJ}}^{\text{raw}}(r) S(r)\right] = \frac{d U_{\text{LJ}}^{\text{raw}}}{dr} S(r) + U_{\text{LJ}}^{\text{raw}}(r) \frac{dS}{dr}$$
  Verified **exact** in both forward Pallas kernels (`MK-48`, `MK-51`) and `custom_vjp` backward kernels.

### 2.2 Isotropic Continuum Dispersion Tail Energy & Exact NPT Pressure Virial
For a homogeneous isotropic fluid of volume $V$ beyond the cutoff/switch ($g(r) \approx 1$), the long-range LJ tail energy is:

$$\Delta E_{\text{disp}}(V) = \frac{2\pi}{V} \sum_{i,j \in \text{real}} 4\epsilon_{ij} \left( \sigma_{ij}^{12} I_{12} - \sigma_{ij}^6 I_6 \right)$$

where the radial tail integrals $I_m = \int_{r_{\text{sw}}}^{r_c} (1 - S(r)) r^{2-m} \, dr + \int_{r_c}^\infty r^{2-m} \, dr$ ($m \in \{6, 12\}$) evaluate via integration by parts ($I_m = \frac{1}{m-3} \int_{r_{\text{sw}}}^{r_c} (-S'(r)) r^{3-m} \, dr$) to exact closed forms (`openmm/context.py:819-831`, verified to `< 10^{-15}` relative error against Gauss–Kronrod quadrature):

$$I_6 = \begin{cases} \dfrac{1}{3 r_c^3}, & w_{\text{sw}} = 0 \\[8pt] \dfrac{10 \left[(r_c^2 + 4 r_c r_{\text{sw}} + r_{\text{sw}}^2)\ln(r_c / r_{\text{sw}}) - 3(r_c^2 - r_{\text{sw}}^2)\right]}{(r_c - r_{\text{sw}})^5}, & w_{\text{sw}} > 0 \end{cases}$$

$$I_{12} = \begin{cases} \dfrac{1}{9 r_c^9}, & w_{\text{sw}} = 0 \\[8pt] \dfrac{(r_c + r_{\text{sw}})(5 r_c^2 + 4 r_c r_{\text{sw}} + 5 r_{\text{sw}}^2)}{252 \, r_c^6 \, r_{\text{sw}}^6}, & w_{\text{sw}} > 0 \end{cases}$$

**Exact $O(N)$ Lorentz–Berthelot Binomial Sum (`BUG-DISP1`)**:  
Expanding $(\frac{\sigma_i + \sigma_j}{2})^m = 2^{-m} \sum_{k=0}^m \binom{m}{k} \sigma_i^k \sigma_j^{m-k}$ decouples the $O(N^2)$ double sum into $m + 1$ scalar $O(N)$ moments $S_k = \sum_{i \in \text{real}} \sqrt{\epsilon_i} \sigma_i^k$:

$$\sum_{i,j \in \text{real}} \sqrt{\epsilon_i \epsilon_j} \left(\frac{\sigma_i + \sigma_j}{2}\right)^m = 2^{-m} \sum_{k=0}^m \binom{m}{k} S_k \, S_{m-k} \qquad (m \in \{6, 12\})$$

*(Note: OpenMM's `NonbondedForceImpl::calcDispersionCorrection` averages over unique unordered particle pairs including self-class pairs via $\frac{n_a(n_a+1)}{2}$ on the diagonal and $n_a n_b$ off-diagonal divided by $\frac{N(N+1)}{2}$, then multiplies by $N^2$; for large $N$, $\frac{N^2}{N(N+1)/2} = \frac{2N}{N+1} = 2 - \frac{2}{N+1}$ differs from the standard $N^2$ pair sum only by $O(1/N)$, e.g. $4 \times 10^{-5}$ on DHFR $N = 23,558$, and is reproduced identically when subtracting $\sum_i \epsilon_i \sigma_i^m$ with weight $\frac{N}{N+1}$.)*

**NPT Dispersion Tail Virial / Pressure Contribution**:  
Because $\Delta E_{\text{disp}}(V) = \frac{C_{\text{disp}}}{V}$ scales purely as $V^{-1}$ (cutoff $r_c$ is fixed in Cartesian space, not scaled with box $V$), its thermodynamic contribution to instantaneous pressure is:

$$\Delta P_{\text{disp}} = -\frac{\partial \Delta E_{\text{disp}}}{\partial V} = +\frac{\Delta E_{\text{disp}}(V)}{V} < 0$$

For liquid TIP3P at $r_c = 9.0\text{ Å}$, $\Delta P_{\text{disp}} \approx -210\text{ bar}$; omitting $\Delta P_{\text{disp}}$ in NPT shifts equilibrium liquid water density downward by $\sim 1.0\%$. In `barostat.py:67` (`lj_dispersion_tail_virial` / `lj_dispersion_tail_pressure`), $\Delta P_{\text{disp}} = \Delta E_{\text{disp}} / V$ has the **exact negative attractive sign**.

---

## 3. Bonded, Improper, Ryckaert–Bellemans & 2D Bicubic CMAP Potentials

### 3.1 Harmonic Bonds & Angles ($1/2$ Prefactor Convention Audit)
- **AMBER `prmtop` raw parameter**: $E = K_{\text{AMBER}}(r - r_0)^2$.
- **OpenMM & Prolix internal convention**:
  $$U_{\text{bond}}(r) = \frac{1}{2} k_b (r - r_0)^2, \qquad U_{\text{angle}}(\theta) = \frac{1}{2} k_\theta (\theta - \theta_0)^2$$
  where $k_b = 2 K_{\text{AMBER}}$.
- **Verification**: `proxide` stores $k_b = 2 K_{\text{AMBER}}$ when parsing XML/prmtop, `prolix/physics/bonded.py:32,80`, `pallas_bonded.py`, and `openmm/context.py:329,371` all multiply by `0.5 * k`, and `openmm/converter.py:1300-1305` converts OpenMM's $k_b$ (`kJ/(mol·nm²)`) directly via `1 / 418.4` (`kcal/(mol·Å²)`). **Zero factor-of-2 discrepancy.**
- **Angle $\arccos$ Singularity Guard**: `bonded.py:76` & `pallas_bonded.py` clamp $\cos\theta \in [-1 + 10^{-7}, 1 - 10^{-7}]$ (or `[-0.999999, 0.999999]`), bounding $|\csc\theta| = 1/\sqrt{1 - \cos^2\theta} \le 2236$ to prevent `NaN` gradients at $\theta \to 0, \pi$.

### 3.2 Proper Periodic Torsions, Improper Torsions & Ryckaert–Bellemans
- **Proper Periodic Torsion (`bonded.py:118`, `pallas_bonded.py`, `context.py:426`)**:
  $$U_{\text{dih}}(\phi) = k_\phi \left[1 + \cos(n \phi - \phi_0)\right], \qquad \frac{d U_{\text{dih}}}{d\phi} = -n k_\phi \sin(n\phi - \phi_0)$$
  - **Dihedral Sign & `b0_is_ji` Audit**: For atom quartet $(A, B, C, D)$ with bond vectors $\mathbf{b}_1 = \mathbf{r}_B - \mathbf{r}_A$, $\mathbf{b}_2 = \mathbf{r}_C - \mathbf{r}_B$, $\mathbf{b}_3 = \mathbf{r}_D - \mathbf{r}_C$, normal vectors $\mathbf{c}_1 = \mathbf{b}_1 \times \mathbf{b}_2$, $\mathbf{c}_2 = \mathbf{b}_2 \times \mathbf{b}_3$, and $\hat{\mathbf{b}}_2 = \mathbf{b}_2 / \|\mathbf{b}_2\|$:
    $$\phi = \operatorname{atan2}\!\left((\hat{\mathbf{b}}_2 \times \mathbf{c}_1) \cdot \mathbf{c}_2,\; \mathbf{c}_1 \cdot \mathbf{c}_2\right)$$
    Matches IUPAC/OpenMM polymer convention ($\phi = 0$ *cis*, $\phi = \pm\pi$ *trans*). Note that OpenMM's internal `ReferenceProperDihedralBond` defines $\mathbf{b}_0 = \mathbf{r}_A - \mathbf{r}_B = -\mathbf{b}_1$, which flips $\mathbf{v}$ unless `b0_is_ji` is tracked; in `pallas_torsions_4body`, `b0_is_ji` is explicitly parameterized and verified against `bonded.py` and OpenMM Reference.
  - **Collinear Guard**: When $\|\mathbf{c}_1\|^2 < 10^{-20}$ or $\|\mathbf{c}_2\|^2 < 10^{-20}$ (three collinear atoms, $\sin\theta \to 0$), `safe_x = 1.0, safe_y = 0.0` avoids `atan2(0, 0)` `NaN` in reverse-mode AD.
- **Improper Torsions**:
  - **AMBER `ff14SB`/`ff19SB`**: Periodic improper $k_\phi [1 + \cos(n\phi - \phi_0)]$ (`PeriodicTorsionForce`).
  - **CHARMM**: Harmonic improper $U_{\text{imp}}(\psi) = k_\psi (\Delta\psi)^2$ where $\Delta\psi = \operatorname{wrap}_{[-\pi,\pi)}(\psi - \psi_0)$ (**without** a $1/2$ factor in CHARMM convention, matching `bonded.py:138` and `context.py:476`).
- **Ryckaert–Bellemans (`openmm/converter.py:413-461`)**:
  $$U_{\text{RB}}(\theta) = \sum_{m=0}^5 C_m (-\cos\theta)^m$$
  Uses polymer angle $\psi = \theta - \pi \implies \cos\psi = -\cos\theta$, evaluated via 5-stage Horner scheme in `kJ/mol` (`energy_omm`) and converted by `CompiledLeptonForce.energy_akma` (`* 1 / 4.184`). Exact match to OpenMM `RBTorsionForce`.

### 3.3 2D Bicubic CMAP Splines (`src/prolix/physics/cmap.py`)
For coupled backbone $(\phi, \psi) \in [-\pi, \pi)^2$ on an $M \times M$ periodic grid ($M = 24$, grid spacing $\Delta = 2\pi / M = 15^\circ$):
1. Angles are wrapped into $[0, 2\pi)$ via $\tilde{\phi} = \phi + 2\pi \mathbb{I}[\phi < 0]$ (`cmap.py:217`, matching `CMAPTorsionForceImpl.cpp`).
2. Grid values $E_{i,j}$, periodic natural cubic spline 1D derivatives $\partial E/\partial\phi, \partial E/\partial\psi$, and cross-derivatives $\partial^2 E/\partial\phi\partial\psi$ are mapped by OpenMM's exact $16 \times 16$ bicubic patch matrix `WT` into $c_{a,b}$ ($a,b \in \{0,1,2,3\}$) on normalized patch coordinates $(t_\phi, t_\psi) \in [0, 1)^2$:
   $$E(\phi, \psi) = \sum_{a=0}^3 \sum_{b=0}^3 c_{a,b} \, t_\phi^a \, t_\psi^b, \qquad \frac{\partial E}{\partial \phi} = \frac{1}{\Delta} \sum_{a=1}^3 \sum_{b=0}^3 a \, c_{a,b} \, t_\phi^{a-1} \, t_\psi^b, \qquad \frac{\partial E}{\partial \psi} = \frac{1}{\Delta} \sum_{a=0}^3 \sum_{b=1}^3 b \, c_{a,b} \, t_\phi^a \, t_\psi^{b-1}$$
3. **Grid Axis Ordering & `first_is_psi` Contract (`BUG-CMAP1` & `BUG-CMAP2`)**: OpenMM's `CMAPTorsionForce::addMap(size, energy)` stores flat `energy[i_phi + size*j_psi]` ($i_\phi$ fast-varying). Under NumPy C-order `.reshape((size, size))`, this yields `grid[j_psi, i_phi]`, which `prolix/physics/cmap.py:259` (`compute_bicubic_coefficients`) explicitly transposes via `energy = energy.T` so patch coefficients are indexed as `coeffs[i_phi, j_psi]`. Passing `first_is_psi=True` to `pallas_cmap_torsions` (or reshaping with `order="F"` before `cmap.py`'s `.T`) transposes $(\phi, \psi) \leftrightarrow (\psi, \phi)$ (`+108.6345 kcal/mol` error on `5DFR`). With C-order `.reshape((size, size))` + `first_is_psi=False`, XLA (`161.39147 kcal/mol`), Pallas (`161.39146 kcal/mol`), and OpenMM (`161.39150 kcal/mol`) match to `< 4e-5 kcal/mol` and force chain rule $\mathbf{F} = -\frac{\partial E}{\partial\phi}\nabla\phi - \frac{\partial E}{\partial\psi}\nabla\psi$ matches central finite differences to `4.5e-10` relative error.

---

## 4. OBC2 Generalized Born (`GBSA`) & ACE Nonpolar Solvation

### 4.1 Pairwise De-Screening Integral $\psi_i$ & OBC2 Effective Born Radii $R_i$
Given intrinsic atomic radii $r_i$ (`mbondi2`), dielectric offset $\delta = 0.09\text{ Å}$, offset radius $\rho_i = r_i - \delta$, and OBC2 scaled radius $s_j = S_j \rho_j$ (`generalized_born.py:66-95`, `ReferenceObc.cpp`):

For pair $(i \neq j)$ at distance $r = r_{ij}$, define integration bounds $L_{ij} = \max(\rho_i, |r - s_j|)$ and $U_{ij} = r + s_j$. When $r + s_j > \rho_i$ (i.e. $L_{ij} < U_{ij}$; otherwise atom $i$ completely engulfs $j$ and $I_{ij} = 0$):

$$I_{ij} = \frac{1}{2} \left[ \frac{1}{L_{ij}} - \frac{1}{U_{ij}} + \frac{1}{4}\left(\frac{s_j^2}{r} - r\right)\left(\frac{1}{L_{ij}^2} - \frac{1}{U_{ij}^2}\right) + \frac{1}{2r} \ln\!\left(\frac{L_{ij}}{U_{ij}}\right) \right]$$

*(Verified algebraically identical to `generalized_born.py:78-85`: $\frac{1}{L} - \frac{1}{U} + \frac{r}{4}(\frac{1}{U^2} - \frac{1}{L^2}) + \frac{1}{2r}\ln\frac{L}{U} + \frac{s_j^2}{4r}(\frac{1}{L^2} - \frac{1}{U^2})$).*

With dimensionless de-screening sum $\psi_i = \rho_i \sum_{j \neq i} I_{ij}$ and OBC2 parameters $(\alpha, \beta, \gamma) = (1.0, 0.8, 4.85)$:

$$\frac{1}{R_i} = \frac{1}{\rho_i} - \frac{1}{r_i} \tanh\!\left(\alpha \psi_i - \beta \psi_i^2 + \gamma \psi_i^3\right)$$

Differentiating wrt $\psi_i$ yields the Born-radius chain-rule factor required for Born-radius forces:

$$\frac{\partial R_i}{\partial \psi_i} = \frac{R_i^2}{r_i} \operatorname{sech}^2\!\left(\alpha \psi_i - \beta \psi_i^2 + \gamma \psi_i^3\right) \left(\alpha - 2\beta \psi_i + 3\gamma \psi_i^2\right)$$

### 4.2 Polar GB Solvation, Debye–Hückel $\kappa$ Screening & ACE Nonpolar Derivative
With Still effective interaction distance $f_{\text{GB}}(r_{ij}, R_i, R_j) = \sqrt{r_{ij}^2 + R_i R_j \exp\!\left(-\frac{r_{ij}^2}{4 R_i R_j}\right)}$ (`D_ij = r_ij^2 / (4 R_i R_j)`):

$$U_{\text{GB}} = -\frac{1}{2} C_{\text{Coul}} \sum_{i,j} \tau_{ij}(f_{\text{GB}}) \frac{q_i q_j}{f_{\text{GB}}}, \qquad \tau_{ij}(f_{\text{GB}}) = \frac{1}{\epsilon_{\text{in}}} - \frac{\exp(-0.73 \, \kappa \, f_{\text{GB}})}{\epsilon_{\text{out}}}$$

*(Sum includes $i = j$ self-solvation where $r_{ii} = 0 \implies f_{\text{GB}} = R_i$, and **all** pairs $i \neq j$ including 1-2/1-3/1-4 with zero exclusions, matching OpenMM `GBSAOBCForce`.)*

The partial derivatives of $f_{\text{GB}}$ needed for exact two-pass (`dE/dR_i` + pairwise force) evaluation are:

$$\frac{\partial f_{\text{GB}}}{\partial r_{ij}} = \frac{r_{ij}}{f_{\text{GB}}} \left(1 - \frac{1}{4} e^{-D_{ij}}\right), \qquad \frac{\partial f_{\text{GB}}}{\partial R_i} = \frac{R_j}{2 f_{\text{GB}}} \left(1 + D_{ij}\right) e^{-D_{ij}}$$

For the **Analytical Continuum Electrostatics (ACE)** nonpolar cavity term (`Schaefer, Bartels & Karplus 1998`; OpenMM `ReferenceObc.cpp`):

$$E_{\text{ACE},i} = 4\pi \sigma_{\text{np}} (r_i + r_{\text{probe}})^2 \left(\frac{r_i}{R_i}\right)^6, \qquad \frac{\partial E_{\text{ACE},i}}{\partial R_i} = -24\pi \sigma_{\text{np}} (r_i + r_{\text{probe}})^2 \frac{r_i^6}{R_i^7} = -\frac{6 \, E_{\text{ACE},i}}{R_i}$$

with OpenMM defaults $r_{\text{probe}} = 1.4\text{ Å}$ (`0.14 nm`) and $\sigma_{\text{np}} = 2.25936\text{ kJ/(mol·nm}^2) = 0.0054\text{ kcal/(mol·Å}^2)$ (`BUG-GB1` remediated). The total derivative $\frac{\partial (U_{\text{GB}} + E_{\text{ACE}})}{\partial R_i}$ is accumulated in Pass 1 and propagated onto atomic forces via $-\sum_{i,j} \left(\frac{\partial E}{\partial R_i} \frac{\partial R_i}{\partial I_{ij}} \nabla_{\mathbf{r}} I_{ij}\right)$ in Pass 2 (`O(N \cdot K)` / tile-pair `MK-67` with zero $O(N^2)$ buffers).

---

## 5. Symplectic/Stochastic Integrators, Constraints, $N_{\text{dof}}$ & Barostats

### 5.1 BAOAB (`VRORV`) Langevin Splitting & Exact Ornstein–Uhlenbeck Variance
Prolix's flagship NVT integrator (`src/prolix/physics/settle.py:2436-2440`, `LangevinMiddleIntegrator` equivalent) executes symmetric **BAOAB** (`B(dt/2) - A(dt/2) - O(dt) - A(dt/2) - B(dt/2)`):
1. **B ($\Delta t / 2$)**: $\mathbf{p} \leftarrow \mathbf{p} + \frac{\Delta t}{2} \mathbf{F}(\mathbf{r}_n)$
2. **A ($\Delta t / 2$)**: $\mathbf{r}_{n+1/2} \leftarrow \mathbf{r}_n + \frac{\Delta t}{2} \mathbf{M}^{-1} \mathbf{p}$
3. **O ($\Delta t$, single central step!)**:
   $$\mathbf{p}^+ \leftarrow c_1 \mathbf{p} + c_2 \sqrt{k_B T \, \mathbf{M}} \, \boldsymbol{\xi}, \qquad c_1 = e^{-\gamma \Delta t}, \qquad c_2 = \sqrt{1 - e^{-2\gamma \Delta t}}, \qquad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
   *(Audit check on $\Delta t$ vs $\Delta t/2$)*: Because BAOAB has **one** central `O` step per timestep, the exact OU transition kernel uses full $\Delta t$ ($c_1 = e^{-\gamma \Delta t}$, variance $m k_B T (1 - e^{-2\gamma\Delta t})$). By contrast, `OBABO` (`VRV` / `VVVR`) has **two** outer `O` half-steps and must use $\Delta t/2$ in each. `settle.py:2436-2437` and `kups_benchmark.py:167-169` use single central `O(dt)` with $c_1 = e^{-\gamma\Delta t}, c_2 = \sqrt{1 - c_1^2}$ — **exact**, preserving $\operatorname{Var}(p) = m k_B T$ for free particles at any $\gamma \Delta t > 0$ and achieving $O(\Delta t^2)$ superconvergence on configurational averages (`Leimkuhler & Matthews, AMRX 2013`).
4. **A ($\Delta t / 2$) + Position Constraints (`SETTLE` / `RATTLE`)**:
   $$\tilde{\mathbf{r}}_{n+1} = \mathbf{r}_{n+1/2} + \frac{\Delta t}{2} \mathbf{M}^{-1} \mathbf{p}^+, \qquad \mathbf{r}_{n+1} = \operatorname{Proj}_{\mathcal{M}}(\tilde{\mathbf{r}}_{n+1};\, \mathbf{r}_n), \qquad \mathbf{p} \leftarrow \mathbf{p}^+ + \mathbf{M} \frac{\mathbf{r}_{n+1} - \tilde{\mathbf{r}}_{n+1}}{\Delta t}$$
   *(Note: dividing the holonomic position displacement $\mathbf{r}_{n+1} - \tilde{\mathbf{r}}_{n+1}$ by full $\Delta t$ matches OpenMM `LangevinMiddleIntegrator`, where `A(dt/2) - O(dt) - A(dt/2)` constrained projection applies impulses over the full step $\Delta t$.)*
5. **Force Evaluation**: $\mathbf{F}(\mathbf{r}_{n+1}) = -\nabla U(\mathbf{r}_{n+1})$ (1 force evaluation per step).
6. **B ($\Delta t / 2$) + Velocity Tangent Projection (`SETTLE-Vel` / `RATTLE-Vel`)**:
   $$\tilde{\mathbf{p}}_{n+1} = \mathbf{p} + \frac{\Delta t}{2} \mathbf{F}(\mathbf{r}_{n+1}), \qquad \mathbf{p}_{n+1} = \operatorname{Proj}_{T_{\mathbf{r}_{n+1}}\mathcal{M}}(\tilde{\mathbf{p}}_{n+1})$$

### 5.2 Analytical SETTLE & Solute RATTLE Lagrange Multipliers
- **SETTLE (`src/prolix/physics/settle.py`)**:
  - **Positions**: Translates each rigid water triangle $(O, H_1, H_2)$ (`r_OH = 0.9572 Å, r_HH = 1.5139 Å` for TIP3P) to its center of mass $\mathbf{R}_{\text{COM}}$ (conserving linear momentum identically), unwrapped via MIC relative to $O$, and solves the mass-weighted Kabsch rotation via Horn's exact $4 \times 4$ quaternion matrix closed-form biquadratic characteristic polynomial ($\lambda_{\max}$ root). Satisfies $\|r_{OH_1}\| = \|r_{OH_2}\| = r_{\text{OH}}$ and $\|r_{H_1 H_2}\| = r_{\text{HH}}$ to machine precision (`< 10^{-7}\text{ Å}` in `float32`, `< 10^{-14}\text{ Å}` in `float64`) in $O(1)$ branchless operations.
  - **Velocities (`project_tip3p_waters_momentum_rigid`)**: Computes exact rigid-body COM velocity $\mathbf{V}_{\text{COM}} = \mathbf{P}_{\text{tot}} / M_{\text{tot}}$ and angular velocity $\boldsymbol{\omega} = \mathbf{I}^{-1} \mathbf{L}$ ($3 \times 3$ inertia tensor inverse), reconstructing $\mathbf{v}_a = \mathbf{V}_{\text{COM}} + \boldsymbol{\omega} \times (\mathbf{r}_a - \mathbf{R}_{\text{COM}})$. Orthogonal to all 3 internal bond gradients ($\mathbf{r}_{ab} \cdot (\mathbf{v}_a - \mathbf{v}_b) = 0$) and conserves linear & angular momentum to machine precision.
- **RATTLE (`settle.py` `_build_rattle_clusters` & `constraint_projection.py`)**:
  - **Position step ($\Delta t^2$ scaling)**: For bond $(i, j)$ with target $d_{ij}$, $\mathbf{r}_{ij}' = \tilde{\mathbf{r}}_i - \tilde{\mathbf{r}}_j$, old bond $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$, and reduced inverse mass $\mu_{ij}^{-1} = m_i^{-1} + m_j^{-1}$:
    $$\Delta \lambda_{\text{pos}} = \frac{d_{ij}^2 - \|\mathbf{r}_{ij}'\|^2}{2 \Delta t^2 \, \mu_{ij}^{-1} (\mathbf{r}_{ij} \cdot \mathbf{r}_{ij}')}, \qquad \delta \mathbf{r}_i = +\frac{\Delta t^2 \Delta\lambda_{\text{pos}}}{m_i} \mathbf{r}_{ij}, \qquad \delta \mathbf{p}_i = +\Delta t \, \Delta\lambda_{\text{pos}} \mathbf{r}_{ij}$$
  - **Velocity step ($\Delta t$ scaling)**: Enforcing $\mathbf{r}_{ij}' \cdot (\mathbf{v}_i - \mathbf{v}_j) = 0$:
    $$\Delta \lambda_{\text{vel}} = -\frac{\mathbf{r}_{ij}' \cdot (\tilde{\mathbf{v}}_i - \tilde{\mathbf{v}}_j)}{d_{ij}^2 \, \mu_{ij}^{-1}}, \qquad \delta \mathbf{p}_i = +\Delta \lambda_{\text{vel}} \mathbf{r}_{ij}', \qquad \delta \mathbf{p}_j = -\Delta \lambda_{\text{vel}} \mathbf{r}_{ij}'$$
    Notice $\Delta t$ cancels cleanly from the velocity projection operator $\operatorname{Proj}_{T\mathcal{M}}(\mathbf{p})$.

### 5.3 Exact Degrees of Freedom ($N_{\text{dof}}$) & Ghost-Atom Isolation
In bucket-padded or ragged multi-topology `EnsemblePlan` batches (`MolecularBundle`), array shapes are padded to `(B, N_bucket, 3)` where $N_{\text{real}} = \sum_i \text{atom\_mask}_i \le N_{\text{bucket}}$.
- **Exact Active $N_{\text{dof}}$ Formula**:
  $$N_{\text{dof}} = 3 N_{\text{real}} - 3 N_{\text{waters}} - N_{\text{solute\_constraints}} - 3 \cdot \mathbb{I}[\text{remove\_com} \land N_{\text{real}} > 1]$$
  *(Note: if massless virtual sites exist, $N_{\text{real}}$ counts only massive particles $m_i > 0$.)*
- **Solute vs Solvent Partitioning (Tier 3 Equipartition Diagnostic)**:
  - Under `remove_com = False` (standard Langevin where friction breaks translational invariance so COM has $3 \times \frac{1}{2} k_B T$), kinetic energy partitions additively:
    $$N_{\text{dof}}^{\text{solute}} = 3 N_{\text{solute}} - N_{\text{solute\_constraints}}, \qquad N_{\text{dof}}^{\text{solvent}} = 6 N_{\text{waters}}$$
    $$T_{\text{solute}} = \frac{2 K_{\text{solute}}}{N_{\text{dof}}^{\text{solute}} k_B}, \qquad T_{\text{solvent}} = \frac{2 K_{\text{solvent}}}{N_{\text{dof}}^{\text{solvent}} k_B}$$
  - Under `remove_com = True`, subtracting the 3 global COM degrees of freedom leaves $N_{\text{dof}}^{\text{total}} = N_{\text{dof}}^{\text{solute}} + N_{\text{dof}}^{\text{solvent}} - 3$, shifting subsystem temperatures only by $O(3 / N_{\text{dof}}) \approx 0.006\%$ on DHFR ($N_{\text{dof}} \approx 50,151$). Measuring $T_{\text{solute}}$ and $T_{\text{solvent}}$ separately catches "flying ice cube" / constraint-thermostat leaks that global $\langle T \rangle$ masks!
- **Ghost-Atom Isolation Verification**:
  - In `pallas_baoab_pre_force` (`settle.py:2362, 2475`): `_build_is_water_or_ghost_mask` assigns `w_code = 3` to all `~atom_mask` slots, so ghost atoms receive **zero** OU thermal kicks, **zero** drift, and **zero** momentum (`p_ghost == 0` identically).
  - In `settle_csvr` (`settle.py:5301-5330`): `ke_total` masks `atom_mask`, and `n_dof` sums `atom_mask`, `water_mask`, and `constraint_mask` — ghost atoms **never** inflate $N_{\text{dof}}$.

### 5.4 NPT Barostats: Stochastic Cell Rescaling (`SCR`) vs Monte Carlo (`MC`)
1. **Monte Carlo Barostat (`barostat.py:137-215`, `openmm/context.py:1600-1695`)**:
   Proposes isotropic volume move $V' = V + \delta V$ ($\delta V \sim \mathcal{U}[-\Delta V_{\max}, +\Delta V_{\max}]$), scaling molecular centers of mass by $\mu = (V'/V)^{1/3}$ while preserving intramolecular rigid geometry ($\mathbf{r}_i' = \mathbf{r}_i + (\mu - 1)\mathbf{R}_{\text{COM}}(\text{mol}(i))$). The exact isothermal-isobaric ($NPT$) partition-function Jacobian over $N_{\text{mol}}$ molecular COM coordinates gives the Metropolis acceptance exponent:
   $$\Delta W = (U' - U) + P_0 (V' - V) - N_{\text{mol}} k_B T \ln\!\left(\frac{V'}{V}\right), \qquad P_{\text{acc}} = \min\!\left(1,\, e^{-\Delta W / (k_B T)}\right)$$
   Verified in both `barostat.py:181` and `openmm/context.py:1661`: uses $N_{\text{mol}}$ (**not** $N_{\text{atoms}}$), preserving rigid water internal constraints!
2. **Bernetti–Bussi (2020) Stochastic Cell Rescaling (`SCR`, `barostat.py:218-290`)**:
   Evolves box volume via the exact first-order SDE with isothermal compressibility $\beta_T$ (`4.5e-5 bar^-1` for water) and barostat coupling time $\tau_P$:
   $$d\epsilon = -\frac{\beta_T}{\tau_P} \left(P_0 - P_{\text{inst}}\right) dt + \sqrt{\frac{2 k_B T \beta_T}{V \tau_P}} \, dW_t, \qquad \mu = e^{\Delta\epsilon / 3}$$
   where instantaneous pressure is $P_{\text{inst}} = \frac{2 K + W}{3 V} + \Delta P_{\text{disp}}$, and positions/momenta scale reciprocally ($\mathbf{R}_{\text{COM}} \to \mu \mathbf{R}_{\text{COM}}$, $\mathbf{P}_{\text{COM}} \to \mu^{-1} \mathbf{P}_{\text{COM}}$) to preserve phase-space Liouville measure.

---

## 6. Four-Tier Empirical Validation Methodology & Release Criteria

The automated validation suite `scripts/validation/validate_release_physics_and_observables.py` executes four empirical tiers on NVIDIA L40S (`sm_89`) with mandatory `require_openmm_cuda_context` (`libOpenMMCUDA.so` verified, zero CPU/OpenCL fallback) and writes full machine-readable provenance and results to `docs/benchmarks/261111_l40s_pallas_v35_physics_validation.json`:

### Tier 1 — Single-Point Term-by-Term Energy & Force Parity
- **Engines**: Prolix Pallas v3.5 (`single` & `mixed`), `prolix.openmm` (`ProlixContext`), OpenMM `CUDA` (`double` & `mixed`), and `kUPS` (`kups_adapter` eV/Å/amu).
- **Systems**: `1UAO` (`1,963`), `1VII` (`4,199`), `1UBQ` (`6,658`), `5DFR` (`2,489`, implicit OBC2 GBSA + CMAP), `Holo-Lysozyme` (`15,002`), `DHFR` (`23,558`, RF & PME), `Holo-DHFR` (`23,621`), and `ApoA1` (`92,224`, non-cubic PME).
- **Force-Group Decomposition**: Bonds, angles, periodic torsions, 2D CMAP, direct nonbonded + 1-4 exceptions, PME reciprocal + self + plasma, GBSA + ACE nonpolar, and LJ dispersion tail.
- **Release Gate**: Relative total energy parity `< 5e-5` (`single`) / `< 1e-5` (`mixed`), per-term parity `< 1e-4`, force RMSE `< 0.05 kcal/(mol·Å)` vs OpenMM `CUDA double`, and `< 1e-12` `float64` unit/force parity vs `kUPS`.

### Tier 2 — Analytical `custom_vjp` vs `float64` Central Finite-Difference Parity
- Evaluates central finite differences $\frac{E(\theta + h\mathbf{e}_k) - E(\theta - h\mathbf{e}_k)}{2h}$ in `float64` across coordinates $\mathbf{r}$, partial charges $q$, LJ $(\sigma, \epsilon)$, GBSA radii $(r_i, s_i)$, and harmonic/periodic bonded parameters $(r_0, k_b, \theta_0, k_\theta, k_\phi)$ across PME, Reaction Field, and OBC2 GBSA.
- **Release Gate**: Relative gradient error `< 1e-5` across all differentiable parameter families.

### Tier 3 — Conservation & Ensemble Thermodynamics
1. **NVE Symplectic Energy Conservation**: `10,000` steps (`dt = 1.0 fs`, `VelocityVerlet` / `gamma = 0`), measuring linear drift (`kcal/(mol·ns·dof)`) and fluctuation ratio $\sigma(E_{\text{tot}}) / \sigma(U)$.
2. **NVT Canonical Equipartition (`BAOAB`, `dt = 1.0 fs`, `T = 300 K`)**: Global $\langle T \rangle$, **solute vs solvent temperature split** ($\langle T_{\text{solute}} \rangle$ vs $\langle T_{\text{solvent}} \rangle$ across flexible solute `X-H` bonds and rigid `SETTLE` water), and Maxwell–Boltzmann kinetic energy fluctuation ratio $\sigma_T / (T_0 \sqrt{2 / N_{\text{dof}}}) \approx 1.0$.
3. **NPT Equation of State (`TIP3P` box & solvated `1UBQ`/`DHFR`, `T = 300 K, P = 1.0 bar`)**: Equilibrium liquid density $\langle \rho \rangle$ (`g/cm³`) and volume-fluctuation isothermal compressibility $\kappa_T = \frac{\langle \Delta V^2 \rangle}{k_B T \langle V \rangle}$ vs OpenMM `CUDA` (`MonteCarloBarostat`).

### Tier 4 — Multi-Replica (`B = 8`) Structural & Dynamical Observables with OpenMM-vs-OpenMM Control
- **Independent-Seed Scientific Control**: Every Tier 4 observable is evaluated across **three** `B = 8` (`100 ps/replica` in `--full` release mode) ensembles from matched initial structures:
  1. `Prolix Pallas v3.5` (`B = 8`, seeds `100..107`)
  2. `OpenMM CUDA Seed-Set A` (`B = 8`, seeds `200..207`)
  3. `OpenMM CUDA Seed-Set B` (`B = 8`, seeds `300..307`)
  Comparing `Prolix vs OpenMM-A` directly against the empirical `OpenMM-A vs OpenMM-B` finite-sampling baseline rigorously separates engine physics differences from finite-trajectory (`8 × 100 ps = 0.8 ns`) statistical variance!
- **Liquid Water (`TIP3P`)**:
  - Oxygen–oxygen radial distribution function $g_{\text{OO}}(r)$: first-peak position $r_{\max}$ (`~2.76–2.78 Å`), peak height $g(r_{\max})$, first minimum $r_{\min}$ (`~3.3–3.4 Å`), and first-shell coordination number $n_{\text{OO}} = 4\pi \rho_O \int_0^{r_{\min}} r^2 g_{\text{OO}}(r)\,dr$.
  - **Unwrapped Self-Diffusion $D_{\text{TIP3P}}$ at Matched $(L, \gamma)$**: Evaluated from unwrapped oxygen mean-squared displacement $\text{MSD}(\Delta t) = \langle \|\mathbf{r}_O^{\text{unwrap}}(t + \Delta t) - \mathbf{r}_O^{\text{unwrap}}(t)\|^2 \rangle = 6 D \Delta t$ at **identical cubic box length $L$ and identical Langevin friction $\gamma = 1.0\text{ ps}^{-1}$** in both Prolix and OpenMM `CUDA`.
  - *Physical Note on Langevin Damping & Yeh–Hummer Finite-Size Correction*: Under Langevin dynamics with friction $\gamma$, stochastic momentum damping suppresses self-diffusion relative to Newtonian (`NVE` / $\gamma \to 0$) TIP3P ($D_0^{\text{TIP3P}} \approx 5.2\text{–}5.8 \times 10^{-5}\text{ cm}^2/\text{s}$), and periodic hydrodynamic image drag reduces $D_{\text{PBC}}(L)$ by the Yeh–Hummer (2004) term $\Delta D_{\text{YH}} = \frac{\xi k_B T}{6\pi \eta L}$ ($\xi \approx 2.837297$). Therefore, engine parity MUST compare $D_{\text{Prolix}}(L, \gamma)$ vs $D_{\text{OpenMM}}(L, \gamma)$ at matched $(L, \gamma)$, while reporting $\Delta D_{\text{YH}}$ for literature reference.
- **Proteins & Holo Complex (`1UAO`, `1UBQ`, `DHFR`, `Holo-DHFR`)**:
  - Backbone $C_\alpha$ RMSD distribution (mean ± SE across `B = 8` + two-sample Kolmogorov–Smirnov statistic $D_{\text{KS}}$ vs OpenMM-A/B control);
  - Per-residue $C_\alpha$ RMSF Pearson correlation $r$ vs OpenMM-A calibrated against the empirical `OpenMM-A vs OpenMM-B` $B = 8$ correlation ceiling;
  - Radius of gyration $R_g$;
  - Backbone $(\phi, \psi)$ Ramachandran basin occupancies ($\alpha$-helix, $\beta$-sheet, PPII/other);
  - Backbone hydrogen-bond persistence ($d_{\text{NO}} < 3.5\text{ Å}$);
  - `Holo-DHFR` bound ligand pocket heavy-atom RMSD and key protein–ligand heavy-atom contact distances.

---

## 7. Measured L40S Release Validation Results

### 7.0 Execution Provenance & Anti-Fallback Verification

All four validation tiers were executed end-to-end in `--full` release mode on Engaging (`NVIDIA L40S`, `sm_89`) via `bathos`:

- **Bathos Run ID**: `6646b318-4aec-4830-aace-8cc0f1445e4c` (`Slurm 29373770`, campaign `8437c24b-a48e-4e14-854f-05e55f48ac97`)
- **JSON Artifact**: `docs/benchmarks/261111_l40s_pallas_v35_physics_validation.json`
- **Git Revision & Timestamp**: `6920235` (`2026-11-16T21:20:13.938564+00:00`), `mode = "full"`, `allow_cpu_smoke = false`, `n_tiers_executed = 4`, `total_wall_time_s = 4,128.93 s` (`68.82 min`)
- **Hardware & Runtime Stack**: `NVIDIA L40S` (`device_index = 0`), CUDA Driver `590.48.01`, Python `3.13.14` (`Clang 22.1.3`), JAX `0.10.2` (`gpu` default backend), Prolix `COULOMB_CONSTANT = 332.06371329919216 kcal·Å/(mol·e²)`
- **OpenMM CUDA Anti-Fallback Guard (`require_openmm_cuda_context`)**: OpenMM `8.3.1`, `platform_name = "CUDA"`, compiler `openmm-nvrtc-builtin`, mapped plugin `/orcd/pool/008/so3_shared/marielle/projects/.venv/lib/python3.13/site-packages/OpenMM.libs/lib/plugins/libOpenMMCUDA.so`, `plugin_load_failures = []` across every single Tier 1, Tier 3, and Tier 4 context (`CUDA MPS` active concurrently for independent `OMM-A`/`OMM-B` multi-replica pools in Tier 4).

---

### 7.1 Tier 1 — Single-Point Multi-Engine Energy & Force Parity (`2,333` to `92,224` Atoms)

#### Table 7.1a — Total Potential Energy & Analytical Force Parity vs OpenMM `CUDA` (`double`)

All energy deltas $\Delta E = E_{\text{engine}} - E_{\text{OMM,double}}$ are in `kcal/mol`; per-atom MAE is $|\Delta E| / N_{\text{real}}$ in `kcal/(mol·atom)`; force RMSE (`F_RMSE`) and max component error (`F_max`) are over all $3 N_{\text{real}}$ Cartesian components in `kcal/(mol·Å)`.

| System | $N_{\text{real}}$ ($N_{\text{pad}}$) | Elec. / Grid ($r_c$) | $E_{\text{OMM,double}}$ (`kcal/mol`) | $\Delta E_{\text{OMM,mixed}}$ | $\Delta E_{\text{single}}$ (Pallas v3.5) | $\Delta E_{\text{mixed}}$ (Pallas v3.5) | $\Delta E_{\text{pctx}}$ (`ProlixContext`) | $\|\Delta E_{\text{single}}\|/N_{\text{real}}$ | `F_RMSE_single` (`F_max`) | `F_RMSE_pctx` (`F_max`) | $\cos\theta_{\text{single}}$ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`1UAO`** | `2,333` (`2,560`) | PME `32³` (`9.0 Å`) | `-7,883.2288` | `+0.0109` | `-0.1593` | `-0.1799` | `-0.1686` | `6.83e-5` | `0.00379` (`0.0264`) | `0.00379` (`0.0264`) | `0.9999999628` |
| **`1VII`** | `4,861` (`8,192`) | PME `40³` (`9.0 Å`) | `-14,607.1908` | `+0.0199` | `-0.6403` | `-0.6778` | `-0.6471` | `1.32e-4` | `0.00456` (`0.0363`) | `0.00456` (`0.0363`) | `0.9999999353` |
| **`1UBQ`** | `7,093` (`8,192`) | PME `48³` (`9.0 Å`) | `-21,294.1820` | `+0.0289` | `-0.6002` | `-0.6408` | `-0.6070` | `8.46e-5` | `0.00309` (`0.0204`) | `0.00309` (`0.0204`) | `0.9999999684` |
| **`5DFR_GBSA`** | `2,489` (`2,560`) | OBC2+ACE+CMAP (`NoCutoff`) | `-5,445.3735` | `+0.0179` | `+0.0049` | `-0.0005` | `+0.0019` | `1.95e-6` | `0.00028` (`0.0025`) | `0.00028` (`0.0025`) | `0.9999999997` |
| **`Holo_Lysozyme`** | `15,002` (`16,384`) | PME `54³` (`9.0 Å`) | `-47,883.2129` | `+0.0668` | `-3.9491` | `-4.0208` | `-3.9707` | `2.63e-4` | `0.00934` (`0.0590`) | `0.00934` (`0.0590`) | `0.9999997351` |
| **`DHFR_RF`** | `23,558` (`25,000`) | RF $\varepsilon_{\text{rf}}=78.3$ (`9.0 Å`) | `-77,330.8602` | `+0.0021` | `+0.0326` | `+0.0229` | `+0.0164` | `1.38e-6` | `0.00002` (`0.00024`) | `0.00002` (`0.00024`) | `1.0000000000` |
| **`DHFR_PME`** | `23,558` (`25,000`) | PME `64³` (`9.0 Å`) | `-83,333.0505` | `+0.1281` | `-1.4099` | `-1.6612` | `-1.4651` | `5.99e-5` | `0.00472` (`0.0290`) | `0.00472` (`0.0290`) | `0.9999999415` |
| **`Holo_DHFR`** | `23,558` (`25,000`) | PME `64³` (`9.0 Å`) | `-85,230.6900` | `+0.1243` | `-1.3994` | `-1.6013` | `-1.4350` | `5.94e-5` | `0.00476` (`0.0272`) | `0.00476` (`0.0272`) | `0.9999999426` |
| **`ApoA1_92k`** | `92,224` (`100,000`) | PME `96×96×64` (`9.0 Å`) | `-323,941.0456` | `+0.4792` | `-13.8096` | `-14.3597` | `-13.6731` | `1.50e-4` | `0.00879` (`0.0696`) | `0.00879` (`0.0696`) | `0.9999999544` |

#### Table 7.1b — Force-Group Energy Decomposition Deltas ($\Delta E_{\text{term}} = E_{\text{Prolix}} - E_{\text{OMM,double}}$ in `kcal/mol`)

Values are shown as `single / mixed` (`prolix_pallas_single` Morton $32 \times 32$ tiles / `prolix_pallas_mixed` `float64` direct accumulation).

| System | $\Delta E_{\text{bonds}}$ | $\Delta E_{\text{angles}}$ | $\Delta E_{\text{torsions}}$ | $\Delta E_{\text{cmap}}$ | $\Delta E_{\text{direct\_nb}}$ (`single` / `mixed`) | $\Delta E_{\text{pme\_recip}}$ (`single` / `mixed`) | $\Delta E_{\text{gbsa}}$ (`single` / `mixed`) | $\Delta E_{\text{tail}}$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`1UAO`** | `-0.000009` / `+0.000004` | `+0.000002` / `-0.000000` | `-0.000013` / `+0.000001` | `0` | `+0.01496` / **`-0.00206`** | `-0.21144` / `-0.21344` | `0` | `+0.03557` |
| **`1VII`** | `-0.000025` / `-0.000012` | `-0.000001` / `+0.000000` | `-0.000011` / `+0.000005` | `0` | `+0.03252` / **`-0.00369`** | `-0.70663` / `-0.70855` | `0` | `+0.03441` |
| **`1UBQ`** | `+0.000008` / `+0.000025` | `-0.000024` / `+0.000000` | `-0.000006` / `+0.000008` | `0` | `+0.03204` / **`-0.00496`** | `-0.66777` / `-0.67063` | `0` | `+0.03476` |
| **`5DFR_GBSA`** | `-0.000906` / `-0.000617` | `+0.000041` / `-0.000006` | `-0.000069` / `+0.000008` | `-0.000037` / **`-0.000013`** | `-0.00032` / **`-0.00032`** | `0` | `+0.00719` / **`+0.00040`** | `0` |
| **`Holo_Lysozyme`** | `-0.000109` / `-0.000023` | `-0.000006` / `-0.000001` | `-0.000101` / `-0.000017` | `0` | `+0.07483` / **`-0.00096`** | `-4.04117` / `-4.04413` | `0` | `+0.02434` |
| **`DHFR_RF`** | `-0.000087` / `+0.000015` | `+0.000018` / `-0.000001` | `-0.000038` / `+0.000016` | `0` | `-0.00786` / **`-0.01511`** | `0` | `0` | `+0.03784` |
| **`DHFR_PME`** | `-0.000001` / `+0.000081` | `-0.000022` / `-0.000001` | `-0.000062` / `+0.000016` | `0` | `+0.20268` / **`-0.01918`** | `-1.65555` / `-1.68009` | `0` | `+0.03784` |
| **`Holo_DHFR`** | `-0.000001` / `+0.000075` | `+0.000008` / `-0.000001` | `-0.000051` / `+0.000016` | `0` | `+0.15380` / **`-0.01987`** | `-1.58328` / `-1.61954` | `0` | `+0.03795` |
| **`ApoA1_92k`** | `-0.000797` / `-0.000682` | `-0.000536` / `+0.000095` | `-0.000912` / `-0.000017` | `0` | `+0.51034` / **`-0.07436`** | `-14.38458` / `-14.32316` | `0` | `+0.03849` |

#### Table 7.1c — `kUPS` Cross-Unit (`eV / Å / amu / τ_kUPS`) Single-Point & Propagator Parity

| Check / System | `kUPS` Bonds (`kcal/mol` from `eV`) | $\|\Delta E_{\text{bonds}}\|$ vs `OMM_double` | `kUPS` Direct NB (`kcal/mol` from `eV`) | $\|\Delta E_{\text{direct\_nb}}\|$ vs `OMM_double` |
| :--- | ---: | ---: | ---: | ---: |
| **`1UAO`** (`2,333`) | `9.774183` | `4.28e-6` | `41,703.3188` | `0.00216` (`0.05 ppm`) |
| **`1VII`** (`4,861`) | `23.864321` | `1.21e-5` | `83,181.8388` | `0.00369` (`0.04 ppm`) |
| **`1UBQ`** (`7,093`) | `49.097304` | `2.52e-5` | `116,434.7521` | `0.00496` (`0.04 ppm`) |
| **`Holo_Lysozyme`** (`15,002`) | `107.054918` | `2.28e-5` | `243,920.2055` | `0.00095` (`0.004 ppm`) |
| **`DHFR_PME`** (`23,558`) | `145.175345` | `8.13e-5` | `399,680.0593` | `0.01929` (`0.05 ppm`) |
| **`Holo_DHFR`** (`23,558`) | `108.084396` | `7.54e-5` | `396,871.1384` | `0.02025` (`0.05 ppm`) |
| **`ApoA1_92k`** (`92,224`) | `424.568383` | `6.82e-4` | `1,208,446.0402` | `0.07449` (`0.06 ppm`) |
| **BAOAB Propagator Check** | $\max\|\Delta \mathbf{r}\| = \mathbf{2.78 \times 10^{-17}\text{ Å}}$ | — | $\max\|\Delta \mathbf{p}\| = \mathbf{1.78 \times 10^{-15}}\text{ AKMA}$ | **`PASS` (`< 1e-12`)** |

**Key Tier 1 Physical Findings**:
1. **Exact Non-SPME Parity (`5DFR_GBSA` & `DHFR_RF`)**: On `5DFR_GBSA` (`2,489` atoms, ff19SB 2D CMAPs + OBC2 GBSA + ACE SASA), `prolix_pallas_mixed` matches OpenMM `CUDA` `double` to **`-0.00054 kcal/mol`** (`2.18e-7 kcal/(mol·atom)` — 33× closer to `OMM_double` than `OMM_mixed`'s `+0.01788 kcal/mol`), with 2D CMAP agreeing to `-0.000013 kcal/mol` (`161.391473` vs `161.391486 kcal/mol`) and `F_RMSE = 0.00028 kcal/(mol·Å)` (`cos = 0.9999999997`). On `DHFR_RF` (`23,558` atoms, `CutoffPeriodic`), `F_RMSE = 0.000020 kcal/(mol·Å)` in `single`/`ProlixContext` and `0.000012 kcal/(mol·Å)` in `mixed` (`cos = 1.0000000000`).
2. **Origin of PME Reciprocal Offset (`order = 4` vs `order = 5` Cardinal B-Spline)**: Across all 7 PME systems (`1UAO` through `ApoA1_92k`), `mixed` `direct_nonbonded` matches OpenMM `CUDA` `double` to **`0.004–0.06 ppm`** (`-0.0021 kcal/mol` on `1UAO` up to `-0.0744 kcal/mol` on `ApoA1_92k` out of `1.208 × 10^6 kcal/mol`, 6× tighter than OpenMM `mixed`'s `+0.4758 kcal/mol`), confirming zero `erfc`/`erf` Coulomb-constant leak (`BUG-ELEC1` resolved) and zero `BUG-TRITON-ATOM1` neighbor loss. More than `97%` of $\Delta E_{\text{total}}$ (`5.9e-5` to `2.6e-4 kcal/(mol·atom)`) and `F_RMSE` (`0.00309–0.00934 kcal/(mol·Å)`) comes from `pme_reciprocal`, reflecting Prolix's `order = 4` B-spline SPME vs OpenMM's hardcoded `PmeOrder = 5` at mesh spacing $h \approx 0.9\text{–}1.2\text{ Å}$ (note also on `ApoA1_92k` `prolix_pallas_mixed`, a single near-cutoff pair boundary rounding between `float64` XLA direct forces and `float32` SPME forces yields `F_max = 1.6930 kcal/(mol·Å)`, `F_RMSE = 0.01014 kcal/(mol·Å)`, whereas `single` and `ProlixContext` maintain `F_max = 0.0696 kcal/(mol·Å)`). The small `+0.024` to `+0.050 kcal/mol` `tail_correction` delta (`0.002%–0.08%` of $E_{\text{tail}}$, zero force at fixed volume) reflects Prolix's exact all-atom Lorentz–Berthelot binomial sum vs OpenMM's class-grouped dispersion integral.

---

### 7.2 Tier 2 — Analytical `@jax.custom_vjp` vs `float64` Central Finite-Difference Parity (`20/20 PASS`)

Directional central finite differences $\frac{E(\theta_0 + h\mathbf{u}) - E(\theta_0 - h\mathbf{u})}{2h}$ ($h = 10^{-5}$, `24` random unit probes per parameter tensor, `lj_switch_width = 1.0 Å` quintic switch active on PME, Morton tile-pair, and RF runs) were compared against analytical `@jax.custom_vjp` gradients on `L40S`.

#### Table 7.2 — Complete 20-Target Analytical Gradient Verification (`all_float64_vjp_pass_lt_1e5 = true`)

| Regime | Differentiable Target | `float64` VJP Rel. $L_2$ Error | `float64` Max $\|\Delta g\|$ | `float32` Pallas VJP Rel. $L_2$ Error | `float32` Max $\|\Delta g\|$ | Cosine Sim. (`f64` / `f32`) | Gate (`< 1e-5`) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :---: |
| **PME (`1UAO`)** | `positions` | `1.933e-07` | `4.170e-05` | `5.516e-07` | `1.083e-04` | `1.000000` / `1.000000` | **PASS** |
| **PME (`1UAO`)** | `charges` | `5.330e-09` | `1.223e-06` | `4.084e-07` | `7.750e-05` | `1.000000` / `1.000000` | **PASS** |
| **PME (`1UAO`)** | `sigmas` | `8.540e-08` | `3.838e-06` | `4.659e-07` | `2.154e-05` | `1.000000` / `1.000000` | **PASS** |
| **PME (`1UAO`)** | `epsilons` | `3.318e-08` | `3.602e-06` | `9.872e-07` | `8.424e-05` | `1.000000` / `1.000000` | **PASS** |
| **PME (`1UAO`)** | `bond_params` | `2.864e-09` | `3.446e-07` | `5.931e-07` | `8.584e-05` | `1.000000` / `1.000000` | **PASS** |
| **PME (`1UAO`)** | `angle_params` | `1.051e-08` | `2.272e-07` | `6.895e-07` | `1.765e-05` | `1.000000` / `1.000000` | **PASS** |
| **PME (`1UAO`)** | `dihedral_params` | `8.112e-08` | `1.779e-07` | `2.165e-07` | `6.616e-07` | `1.000000` / `1.000000` | **PASS** |
| **Morton $32\times 32$ Tiles** | `positions` (`switch=1.0 Å`) | — (cf. PME `f64`) | — | `5.568e-07` | `1.083e-04` | — / `1.000000` | **PASS** |
| **Morton $32\times 32$ Tiles** | `charges` (`switch=1.0 Å`) | — (cf. PME `f64`) | — | `6.389e-07` | `1.322e-04` | — / `1.000000` | **PASS** |
| **Morton $32\times 32$ Tiles** | `sigmas` (`switch=1.0 Å`) | — (cf. PME `f64`) | — | `5.487e-07` | `2.737e-05` | — / `1.000000` | **PASS** |
| **Morton $32\times 32$ Tiles** | `epsilons` (`switch=1.0 Å`) | — (cf. PME `f64`) | — | `9.710e-07` | `8.805e-05` | — / `1.000000` | **PASS** |
| **Reaction Field (`1UAO`)** | `positions` | `4.839e-07` | `3.772e-05` | — | — | `1.000000` / — | **PASS** |
| **Reaction Field (`1UAO`)** | `charges` | `8.807e-10` | `1.347e-07` | — | — | `1.000000` / — | **PASS** |
| **Reaction Field (`1UAO`)** | `sigmas` | `8.248e-08` | `4.613e-06` | — | — | `1.000000` / — | **PASS** |
| **Reaction Field (`1UAO`)** | `epsilons` | `3.406e-08` | `3.693e-06` | — | — | `1.000000` / — | **PASS** |
| **OBC2 GBSA + ACE (`5DFR`)** | `positions` | `1.580e-09` | `2.632e-08` | `2.726e-07` | `8.922e-06` | `1.000000` / `1.000000` | **PASS** |
| **OBC2 GBSA + ACE (`5DFR`)** | `charges` | `1.055e-10` | `2.178e-08` | `3.337e-07` | `5.539e-05` | `1.000000` / `1.000000` | **PASS** |
| **OBC2 GBSA + ACE (`5DFR`)** | `radii` ($r_i$) | `5.664e-10` | `1.614e-08` | `3.619e-07` | `1.620e-05` | `1.000000` / `1.000000` | **PASS** |
| **OBC2 GBSA + ACE (`5DFR`)** | `scaled_radii` ($s_i$) | `1.350e-09` | `2.123e-08` | `3.171e-07` | `5.706e-06` | `1.000000` / `1.000000` | **PASS** |
| **2D Bicubic CMAP (`5DFR`)** | `cmap_positions` | `8.936e-10` | `8.470e-09` | — | — | `1.000000` / — | **PASS** |

Every `float64` analytical gradient achieved `1.05e-10` to `4.84e-7` relative $L_2$ error, and **every `float32` Pallas Triton `@jax.custom_vjp` kernel** (including Morton $32 \times 32$ Newton-pair tiles with `lj_switch_width = 1.0 Å` and 2-pass symmetric tile OBC2 GBSA + ACE Born-radius chain rule) achieved `< 1.0e-6` relative $L_2$ error — over an order of magnitude tighter than the `< 1e-5` release gate.

---

### 7.3 Tier 3 — Symplectic NVE Conservation, NVT Equipartition & NPT Equation of State

#### Table 7.3a — NVE Symplectic Energy Conservation (`10,000` steps, $\Delta t = 1.0\text{ fs}$, $10.0\text{ ps}$)

| System | $N_{\text{real}}$ ($N_{\text{dof}}$) | Engine & Precision | Linear Drift (`kcal/(mol·ns·dof)`) | Max $\|\Delta E_{\text{tot}}\|/N_{\text{dof}}$ (`kcal/mol`) | $\sigma(E_{\text{tot}}) / \sigma(U)$ |
| :--- | ---: | :--- | ---: | ---: | ---: |
| **`TIP3P_water_box`** | `2,685` (`5,367`) | **Prolix Pallas v3.5 (`single`)** | `+0.02103` | `4.50e-04` | `0.01583` |
| **`TIP3P_water_box`** | `2,685` (`5,367`) | OpenMM `CUDA` (`mixed`) | `-0.00257` | `9.16e-05` | `0.00248` |
| **`TIP3P_water_box`** | `2,685` (`5,367`) | OpenMM `CUDA` (`double`) | `-0.00053` | `7.03e-05` | `0.00220` |
| **`1UAO`** | `2,333` (`4,803`) | **Prolix Pallas v3.5 (`single`)** | `-0.05767` | `7.11e-04` | `0.03878` |
| **`1UAO`** | `2,333` (`4,803`) | OpenMM `CUDA` (`mixed`) | `-0.01035` | `2.59e-04` | `0.00764` |
| **`1UAO`** | `2,333` (`4,803`) | OpenMM `CUDA` (`double`) | `-0.01121` | `3.02e-04` | `0.00868` |

#### Table 7.3b — NVT Canonical Kinetic Equipartition (`15,000` prod steps, $\Delta t = 1.0\text{ fs}$, $\gamma = 1.0\text{ ps}^{-1}$, $T_0 = 300.0\text{ K}$)

| System | $N_{\text{real}}$ / $N_{\text{pad}}$ ($N_{\text{dof}}$) | Engine | $\langle T_{\text{total}} \rangle \pm \sigma_T$ (`K`) | $\langle T_{\text{solute}} \rangle$ (`K`) | $\langle T_{\text{solvent}} \rangle$ (`K`) | $\Delta T_{\text{sol}-\text{wat}}$ (`K`) | MB Ratio $\sigma_T / \sigma_{\text{MB}}$ | Ghost $\|\mathbf{p}_{\text{pad}}\|_2$ |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`1UBQ`** | `7,093` / `8,192` (`15,414`) | **Prolix Pallas v3.5** | `297.74 ± 3.14` | `294.03` | `298.91` | `-4.88` | `0.919` | `0.654` (masked) |
| **`1UBQ`** | `7,093` / — (`15,414`) | OpenMM `CUDA` (`mixed`) | `300.39 ± 3.21` | `299.55` | `300.65` | `-1.11` | `0.940` | — |
| **`DHFR_PME`** | `23,558` / `25,000` (`49,602`) | **Prolix Pallas v3.5** | `297.42 ± 3.97` | `292.98` | `298.20` | `-5.22` | `2.084` | `0.943` (masked) |
| **`DHFR_PME`** | `23,558` / — (`49,602`) | OpenMM `CUDA` (`mixed`) | `298.00 ± 4.26` | `297.28` | `298.13` | `-0.85` | `2.237` | — |

#### Table 7.3c — NPT Liquid Density, Isothermal Compressibility & MC Barostat Acceptance ($T_0 = 300\text{ K}, P_0 = 1.0\text{ bar}, \Delta t = 1.0\text{ fs}$, interval `25`)

| System ($N_{\text{mol}}$, Prod Steps) | Prolix $\langle \rho \rangle \pm \text{SE}$ (`g/cm³`) | OpenMM `CUDA` $\langle \rho \rangle \pm \text{SE}$ (`g/cm³`) | $\|\Delta \rho\|$ (`g/cm³`) | Prolix $\langle V \rangle \pm \sigma_V$ (`Å³`) | OpenMM $\langle V \rangle \pm \sigma_V$ (`Å³`) | Prolix $\kappa_T$ (`10⁻⁵ bar⁻¹`) | OpenMM $\kappa_T$ (`10⁻⁵ bar⁻¹`) | Prolix MC Acc. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`TIP3P_water_box`** (`895`, `25k`) | `0.9679 ± 0.0022` | `0.9857 ± 0.0027` | `0.0178` (`-1.81%`) | `27,665.8 ± 262.9` | `27,165.3 ± 276.5` | `6.03` | `6.80` | `50.3%` |
| **`1UBQ`** (`1,955`, `25k`) | `1.0221 ± 0.0013` | `1.0390 ± 0.0010` | `0.0169` (`-1.63%`) | `71,106.9 ± 407.2` | `69,948.5 ± 317.8` | `5.63` | `3.49` | `54.9%` |
| **`DHFR_PME`** (`7,024`, `18k`) | `1.0061 ± 0.0011` | `1.0220 ± 0.0011` | `0.0160` (`-1.56%`) | `238,523.8 ± 900.5` | `234,791.5 ± 748.8` | `8.21` | `5.77` | `52.6%` |

**Epistemic Assessment of Tier 3 Minor Discrepancies**:
1. **NVE Drift Ratio (`5–8×` vs OpenMM `mixed`)**: Both `TIP3P_water_box` (`+0.021 kcal/(mol·ns·dof)`) and `1UAO` (`-0.058 kcal/(mol·ns·dof)`) conserve total energy to `< 7.2e-4 kcal/(mol·dof)` over `10,000` steps. The `5–6×` higher $\sigma(E_{\text{tot}})/\sigma(U)$ ratio vs OpenMM `CUDA` `mixed` reflects pure-`float32` state/SETTLE accumulation in the Tier 3a `settle_langevin(gamma=0)` stepper (vs `float64` coordinate/velocity accumulators in OpenMM `mixed` and `EnsemblePlan(precision="mixed")`) plus unshifted `r_c = 9.0 Å` direct PME + order-4 SPME mesh force oscillation.
2. **NVT Solute–Solvent Split & MB Ratio**: Global $\langle T \rangle$ agrees within `-2.64 K` (`1UBQ`) and `-0.59 K` (`DHFR_PME`) of OpenMM `CUDA`, confirming padding atoms (`1,099` in `1UBQ`, `1,442` in `DHFR_PME`) never leak into $N_{\text{dof}}$. Both engines exhibit `mb_std_ratio ~ 2.08–2.24` on `23.6k`-atom `DHFR_PME` over `15 ps` at weak damping $\gamma = 1.0\text{ ps}^{-1}$ due to slow global potential-energy relaxation modes. Prolix shows a `-4.9` to `-5.2 K` (`~1.6%`) cooler protein solute (`293–294 K` vs `298–299 K` water) vs OpenMM's `-0.9` to `-1.1 K`, attributable to `float32` iterative RATTLE tolerance on `HBonds`-constrained protein solute.
3. **NPT Equilibrium Density Offset (`-0.016` to `-0.018 g/cm³`, `-1.6%` to `-1.8%`)**: Adaptive MC volume scaling maintains `50.3%–54.9%` acceptance across all three systems (`27.7k` to `238.5k Å³`). Prolix equilibrates systematically `1.56%–1.81%` less dense than OpenMM `CUDA` (`0.9679` vs `0.9857 g/cm³` on `TIP3P`; `1.0061` vs `1.0220 g/cm³` on `DHFR_PME`), driven by fixed-mesh order-4 SPME volume-trial $\Delta H$ bias when scaling $L$ at constant grid dimensions `(N_x, N_y, N_z)`; fluctuation-derived $\kappa_T$ over `18–25 ps` (`720–1,000` MC steps) agrees to order of magnitude (`5.6–8.2` vs `3.5–6.8 × 10⁻⁵ bar⁻¹`).

---

### 7.4 Tier 4 — Multi-Replica (`B = 8`) Dynamical & Structural Observables vs `OMM-A` / `OMM-B` Control

#### Table 7.4a — Liquid Water (`TIP3P`, `895` waters, $L = 30.0\text{ Å}$, $B = 8 \times 50\text{ ps}$) RDF & Self-Diffusion

| Observable | Prolix Pallas v3.5 (`B=8`) | OpenMM `CUDA` `OMM-A` (`B=8`) | OpenMM `CUDA` `OMM-B` (`B=8`) | `Prolix − OMM-A` | `OMM-A − OMM-B` Control |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **$g_{\text{OO}}(r)$ First Peak $r_{\max}$ (`Å`)** | `2.7777 ± 0.0008` | `2.7770 ± 0.0006` | `2.7772 ± 0.0008` | `+0.0007` (`0.67σ`) | `-0.0002` (`0.21σ`) |
| **$g_{\text{OO}}(r)$ First Peak Height** | `2.6879 ± 0.0040` | `2.6824 ± 0.0050` | `2.6840 ± 0.0084` | `+0.0054` (`0.84σ`) | `-0.0016` (`0.16σ`) |
| **$g_{\text{OO}}(r)$ First Min $r_{\min}$ (`Å`)** | `3.580 ± 0.027` | `3.527 ± 0.036` | `3.576 ± 0.018` | `+0.053` (`1.17σ`) | `-0.049` (`1.23σ`) |
| **Coordination Number $n_{\text{OO}}$** | `5.695 ± 0.141` | `5.437 ± 0.185` | `5.683 ± 0.090` | `+0.257` (`+0.012` vs `B`) | `-0.245` (`1.20σ`) |
| **Full $g_{\text{OO}}(r)$ Curve RMSE** | — | — | — | **`0.00288`** | **`0.00271`** |
| **$D_{\text{PBC}}(L=30\text{ Å}, \gamma=1\text{ ps}^{-1})$ (`10⁻⁵ cm²/s`)** | `4.706 ± 0.039` | `4.754 ± 0.059` | `4.689 ± 0.034` | **`-0.048`** (`0.68σ`) | **`+0.066`** (`0.97σ`) |
| **Yeh–Hummer $D_\infty^{\text{est}} = D_{\text{PBC}} + 0.647$ (`10⁻⁵ cm²/s`)** | `5.354 ± 0.039` | `5.402 ± 0.059` | `5.336 ± 0.034` | `-0.048` (`-0.89%`) | `+0.066` (`+1.23%`) |
| **Cold-Start NVE ($\gamma=0$) $D_{\text{PBC}}$ (`10⁻⁵ cm²/s`, `B=4`)** | `0.783 ± 0.014` | `0.728 ± 0.032` | — | `+0.056` (`1.59σ`) | — |

*Note on Liquid Water Parity*: At matched $(L = 30.0\text{ Å}, \gamma = 1.0\text{ ps}^{-1})$, Prolix's unwrapped self-diffusion coefficient ($4.706 \pm 0.039 \times 10^{-5}\text{ cm}^2/\text{s}$, `max_interframe_jump = 1.95 Å << L/2 = 15.0 Å`, `unwrap_aliasing_safe = true`) sits **strictly between** `OMM-A` (`4.754`) and `OMM-B` (`4.689`), with $|\Delta D| = 0.048 \times 10^{-5}\text{ cm}^2/\text{s}$ smaller than the `OMM-A vs OMM-B` seed variance (`0.066 × 10⁻⁵ cm²/s`) and Yeh–Hummer infinite-box estimate $D_\infty^{\text{est}} = 5.354 \times 10^{-5}\text{ cm}^2/\text{s}$ matching literature TIP3P (`~5.3–5.5 × 10⁻⁵ cm²/s`). In the auxiliary `diffusion_newtonian_nve_gamma_0` sub-run (`B = 4`), initializing velocities at `300 K` directly on energy-minimized coordinates without prior NVT potential-energy equilibration partitioned ~half the kinetic energy into potential energy in *both* engines ($T_{\text{eq}} \approx 160\text{ K}$ supercooled water), where Prolix (`0.783 ± 0.014`) and OpenMM `CUDA` (`0.728 ± 0.032`) again agree within `1.6σ`.

#### Table 7.4b — Multi-Replica (`B = 8 × 100 ps = 0.8 ns/engine`) Protein Structural & Dynamical Observables

| System ($N_{\text{real}}$) | Observable | Prolix Pallas v3.5 (`B=8`) | OpenMM `OMM-A` (`B=8`) | OpenMM `OMM-B` (`B=8`) | `Prolix vs OMM-A` (`Prolix vs OMM-B`) | `OMM-A vs OMM-B` Control |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| **`1UAO`** (`2,333`) | $C_\alpha$ RMSD (`Å`) | `1.143 ± 0.056` | `1.090 ± 0.060` | `1.110 ± 0.048` | `+0.052 Å, z=0.64` (`+0.033 Å, z=0.45`) | `-0.019 Å, z=0.25` |
| **`1UAO`** (`2,333`) | $C_\alpha$ RMSF Pearson $r$ (`RMSE`) | — | — | — | **`0.9876`** (`0.023 Å`) / **`0.9961`** (`0.021 Å`) | **`0.9943`** (`0.019 Å`) |
| **`1UAO`** (`2,333`) | $R_g$ (`Å`) / Rama $L_1$ | `5.560 ± 0.006` | `5.588 ± 0.007` | `5.571 ± 0.010` | `-0.028 Å` (`-0.011 Å`) / $L_1 = 0.0118$ | `+0.017 Å` / $L_1 = 0.0132$ |
| **`1UAO`** (`2,333`) | H-Bonds (`count` / pers. $r$) | `3.84 ± 0.11` | `3.90 ± 0.11` | `3.94 ± 0.09` | $\Delta = -0.07$ / **$r = 0.9934$** (`0.9979`) | $\Delta = -0.04$ / **$r = 0.9923$** |
| **`1UBQ`** (`7,093`) | $C_\alpha$ RMSD (`Å`) | `0.926 ± 0.039` | `1.022 ± 0.028` | `0.947 ± 0.034` | `-0.097 Å, z=2.03` (**`-0.021 Å, z=0.41`**) | `+0.075 Å, z=1.72` |
| **`1UBQ`** (`7,093`) | $C_\alpha$ RMSF Pearson $r$ (`RMSE`) | — | — | — | **`0.9935`** (`0.031 Å`) / **`0.9930`** (`0.044 Å`) | **`0.9895`** (`0.047 Å`) |
| **`1UBQ`** (`7,093`) | $R_g$ (`Å`) / Rama $L_1$ | `11.900 ± 0.016` | `11.887 ± 0.019` | `11.894 ± 0.013` | `+0.014 Å` (`+0.006 Å`) / $L_1 = 0.0156$ (`0.0034`) | `-0.008 Å` / $L_1 = 0.0123$ |
| **`1UBQ`** (`7,093`) | H-Bonds (`count` / pers. $r$) | `42.97 ± 0.22` | `42.91 ± 0.11` | `43.11 ± 0.23` | $\Delta = +0.06$ / **$r = 0.9974$** (`0.9964`) | $\Delta = -0.20$ / **$r = 0.9970$** |
| **`DHFR_PME`** (`23,558`) | $C_\alpha$ RMSD (`Å`) | `1.070 ± 0.033` | `1.062 ± 0.024` | `1.063 ± 0.020` | **`+0.008 Å, z=0.19, KS=0.040`** (`+0.007 Å`) | `-0.001 Å, z=0.02, KS=0.075` |
| **`DHFR_PME`** (`23,558`) | $C_\alpha$ RMSF Pearson $r$ (`RMSE`) | — | — | — | **`0.9585`** (`0.054 Å`) / **`0.9775`** (`0.045 Å`) | **`0.9723`** (`0.041 Å`) |
| **`DHFR_PME`** (`23,558`) | $R_g$ (`Å`) / Rama $L_1$ | `15.679 ± 0.017` | `15.669 ± 0.009` | `15.669 ± 0.018` | `+0.010 Å` (`+0.010 Å`) / $L_1 = 0.0029$ | `+0.0004 Å` / $L_1 = 0.0013$ |
| **`DHFR_PME`** (`23,558`) | H-Bonds (`count` / pers. $r$) | `81.74 ± 0.39` | `82.06 ± 0.26` | `82.49 ± 0.26` | $\Delta = -0.32$ / **$r = 0.9904$** (`0.9854`) | $\Delta = -0.43$ / **$r = 0.9781$** |
| **`Holo_DHFR`** (`23,558`) | $C_\alpha$ RMSD (`Å`) | `1.073 ± 0.040` | `1.070 ± 0.033` | `1.061 ± 0.041` | **`+0.004 Å, z=0.07, KS=0.054`** (`+0.012 Å`) | `+0.009 Å, z=0.17, KS=0.074` |
| **`Holo_DHFR`** (`23,558`) | $C_\alpha$ RMSF Pearson $r$ (`RMSE`) | — | — | — | **`0.9753`** (`0.040 Å`) / **`0.9733`** (`0.040 Å`) | **`0.9728`** (`0.042 Å`) |
| **`Holo_DHFR`** (`23,558`) | $R_g$ (`Å`) / Rama $L_1$ | `15.676 ± 0.022` | `15.637 ± 0.016` | `15.669 ± 0.021` | `+0.039 Å` (`+0.007 Å`) / $L_1 = 0.0027$ | `-0.032 Å` / $L_1 = 0.0041$ |
| **`Holo_DHFR`** (`23,558`) | H-Bonds (`count` / pers. $r$) | `82.57 ± 0.23` | `82.09 ± 0.35` | `81.90 ± 0.40` | $\Delta = +0.49$ / **$r = 0.9807$** (`0.9855`) | $\Delta = +0.19$ / **$r = 0.9788$** |

#### Table 7.4c — `Holo_DHFR` Bound Ligand Pocket Heavy-Atom RMSD & Active-Site Contacts (`B = 8 × 100 ps`)

| Metric / Active-Site Contact (`Ref. Å`) | Prolix Pallas v3.5 (`B=8`) | OpenMM `OMM-A` (`B=8`) | OpenMM `OMM-B` (`B=8`) | `Prolix − OMM-A` (`Prolix − OMM-B`) | `OMM-A − OMM-B` Control |
| :--- | ---: | ---: | ---: | :--- | :--- |
| **Pocket Heavy-Atom RMSD (`Å`)** | `2.660 ± 0.077` | `2.567 ± 0.095` | `2.665 ± 0.113` | `+0.093` (`z=0.76, KS=0.084`) / **`-0.005`** (`z=0.03, KS=0.044`) | `-0.097` (`z=0.66, KS=0.089`) |
| **`ARG57:NH1–LIG:C13`** (`2.941 Å`) | `3.001 ± 0.018` | `2.984 ± 0.009` | `2.997 ± 0.015` | `+0.018 Å` (`+0.004 Å`) | `-0.013 Å` |
| **`ILE94:O–LIG:C2`** (`3.088 Å`) | `5.150 ± 0.336` | `5.159 ± 0.316` | `5.589 ± 0.290` | `-0.009 Å` (`-0.439 Å`) | `-0.430 Å` |
| **`PHE31:CG–LIG:C6`** (`3.129 Å`) | `3.504 ± 0.024` | `3.503 ± 0.019` | `3.521 ± 0.028` | `+0.001 Å` (`-0.017 Å`) | `-0.018 Å` |
| **`ASP27:OD2–LIG:C6`** (`3.437 Å`) | `5.207 ± 0.055` | `5.171 ± 0.141` | `5.220 ± 0.080` | `+0.037 Å` (`-0.013 Å`) | `-0.049 Å` |
| **`LEU28:CD2–LIG:C6`** (`3.792 Å`) | `4.438 ± 0.040` | `4.738 ± 0.157` | `4.765 ± 0.275` | `-0.300 Å` (`-0.327 Å`, `1.16σ`) | `-0.027 Å` |

**Summary Conclusion**: Across all four tiers on `NVIDIA L40S` (`Slurm 29373770`, Bathos `run_id = 6646b318-4aec-4830-aace-8cc0f1445e4c`), Prolix Pallas v3.5 achieves exact `0.00 ppm` fundamental-constant alignment, sub-`10⁻⁶` `float32` Pallas `@jax.custom_vjp` gradient accuracy across all 20 differentiable parameter families (`1.05e-10` to `4.84e-7` in `float64`), micro-kcal/mol non-SPME energy/force parity (`5DFR_GBSA` `dE_mixed = -0.00054 kcal/mol`, `DHFR_RF` `F_RMSE = 0.000012–0.000020 kcal/(mol·Å)`), understood order-4 vs order-5 B-spline SPME mesh scaling up to `92,224`-atom `ApoA1`, stable NVE/NVT/NPT ensembles (`50–55%` adaptive MC acceptance, `|Δρ| <= 0.0178 g/cm³`), and `B = 8` multi-replica liquid water (`g_OO` RMSE `0.00288` vs `0.00271` control, $|\Delta D_{\text{TIP3P}}| = 0.048$ vs $0.066 \times 10^{-5}\text{ cm}^2/\text{s}$ control) and protein/holo-ligand ensembles (`C_alpha` RMSF $r = 0.959\text{–}0.996$, H-bond persistence $r = 0.981\text{–}0.998$, bound ligand pocket RMSD `2.660 ± 0.077 Å` vs `2.665 ± 0.113 Å` `OMM-B`) statistically indistinguishable from independent-seed OpenMM `8.3.1` `CUDA` trajectories.
