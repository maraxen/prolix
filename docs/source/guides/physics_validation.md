# Physics, Units, & Multi-Engine Validation

Community trust in a molecular dynamics engine rests on exact unit conversions, transparent potential-energy functional forms, zero silent CPU fallbacks during GPU benchmarking, and reproducible multi-engine parity across precision modes.

---

## 1. Canonical Unit Systems & Conversion Factors

Prolix uses **AKMA** (**Å**ngströms, **k**ilocalories/mol, atomic **m**ass units `u`, elementary charge `e`) internally across `prolix.physics` and `EnsemblePlan` kernels, while `prolix.openmm` exposes native **OpenMM** (`nm`, `ps`, `kJ/mol`, `bar`/`atm`) quantities at every public API boundary (`src/prolix/openmm/units.py`), and `prolix.physics.kups_adapter` converts to/from **kUPS** (`Å`, `eV`, `u`).

| Physical Quantity | Prolix Internal (AKMA) | `prolix.openmm` (`openmm.unit`) | `EnsemblePlan.run` Kwargs | `kups_adapter` (kUPS) |
| :--- | :--- | :--- | :--- | :--- |
| **Length** | `Å` | `nm` (`× 10.0 → Å`) | `Å` | `Å` |
| **Mass** | `amu` (`g/mol`) | `Da` (`1:1`) | `amu` | `amu` |
| **Time** | `τ_AKMA = 48.88821291839 fs` | `ps` (`× 1000 / 48.88821291839`) | `fs` (`dt_unit="fs"` default) | `τ_kUPS ≈ 10.18050571 fs` |
| **Energy** | `kcal/mol` | `kJ/mol` (`/ 4.184 → kcal/mol`) | `kcal/mol` (`kT`) | `eV` (`× 23.060549 → kcal/mol`) |
| **Force** | `kcal/(mol·Å)` | `kJ/(mol·nm)` (`/ 41.84`) | `kcal/(mol·Å)` | `eV/Å` (`× 23.060549`) |
| **Velocity** | `Å / τ_AKMA` | `nm/ps` (`× 0.4888821291839 → Å / τ_AKMA`) | `Å / τ_AKMA` | `Å / τ_kUPS` |
| **Friction ($\gamma$)** | `τ_AKMA⁻¹` | `ps⁻¹` (`× 0.04888821291839`) | `ps⁻¹` | `τ_kUPS⁻¹` |
| **Pressure** | `kcal/(mol·Å³)` | `bar` (`/ 69,476.95`) or `atm` (`/ 68,568.42`) | `bar` / `atm` | — |
| **Charge** | `e` | `e` (`1:1`) | `e` | `e` |

### Fundamental Physical Constants

- **Boltzmann constant ($k_B$)**: `0.0019872041 kcal/(mol·K)` (`8.314462618e-3 kJ/(mol·K)`).
- **Coulomb constant ($C_{\text{Coul}} = \frac{1}{4\pi\varepsilon_0}$)**: **`332.06371329919216 kcal·Å/(mol·e²)`** (`138.935457644382 kJ·nm/(mol·e²)`), unified across direct-space tiles, 1-4 exceptions, SPME reciprocal/self energies, and Generalized Born solvation so Ewald $\operatorname{erf}(\alpha r)/r$ exclusions cancel to `0.00 ppm`.
- **Ewald splitting parameter ($\alpha$)**: Stored in **`Å⁻¹`** in Prolix; multiply by `10.0` (`nm⁻¹`) when comparing against `openmm.NonbondedForce.setPMEParameters(alpha_nm, nx, ny, nz)`.

---

## 2. Potential Energy Functional Forms

Every term in $U_{\text{total}}(\mathbf{r})$ matches OpenMM's analytical definitions up to explicit convention conversions handled automatically in `from_openmm_system`:

1. **Harmonic Bonds & Angles**:
   $$U_{\text{bond}} = \sum_{(i,j)} \frac{1}{2} k_b (r_{ij} - r_0)^2, \qquad U_{\text{angle}} = \sum_{(i,j,k)} \frac{1}{2} k_\theta (\theta_{ijk} - \theta_0)^2$$
   *(Conversion: $k_b^{\text{AKMA}} = k_b^{\text{OMM}} / 418.4$; $k_\theta^{\text{AKMA}} = k_\theta^{\text{OMM}} / 4.184$.)*

2. **Periodic, Ryckaert-Bellemans, Harmonic Improper, & CMAP Torsions**:
   - Periodic proper/improper: $U_{\text{tors}} = \sum k_\phi [1 + \cos(n \phi - \phi_0)]$.
   - Harmonic improper (`CustomTorsionForce`): OpenMM writes $\frac{1}{2} k_{\text{OMM}} (\psi - \psi_0)^2$ (`kJ/mol`), whereas Prolix's internal harmonic improper kernel evaluates $k_{\text{Prolix}} (\psi - \psi_0)^2$ (`kcal/mol`), so `from_openmm_system` scales $k_{\text{Prolix}} = \frac{0.5}{4.184} k_{\text{OMM}}$ (`IMPROPER_K_OMM_TO_AKMA`).
   - CMAP (`CMAPTorsionForce`): 2D periodic bicubic spline over backbone $(\phi, \psi)$ maps.

3. **Explicit-Solvent Nonbonded (SPME + Switched Lennard-Jones + Continuum Tail)**:
   - **Real-space Coulomb + 12-6 Lorentz-Berthelot LJ** ($\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}$, $\varepsilon_{ij} = \sqrt{\varepsilon_i \varepsilon_j}$):
     $$U_{\text{direct}} = \sum_{i < j,\, r_{ij} < r_{\text{cut}}} \left[ C_{\text{Coul}} q_i q_j \frac{\operatorname{erfc}(\alpha r_{ij})}{r_{ij}} + 4 \varepsilon_{ij} S(r_{ij}) \left(\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6\right) \right]$$
     where $S(r)$ is the $C^2$ quintic switching polynomial over $[r_{\text{switch}}, r_{\text{cut}}]$.
   - **Reciprocal + Self + Charged-Background SPME**: Cardinal B-spline charge assignment + 3D FFT Poisson solve minus self-interaction $-\frac{\alpha}{\sqrt{\pi}} C_{\text{Coul}} \sum_i q_i^2$ and neutralizing plasma background $-\frac{\pi C_{\text{Coul}}}{2 \alpha^2 V} (\sum_i q_i)^2$.
   - **Exact $O(N)$ Binomial Lorentz-Berthelot Dispersion Tail**: Evaluates the exact continuum integral $\Delta U_{\text{disp}}$ (and pressure virial $-\partial \Delta U_{\text{disp}} / \partial V$) in $O(N)$ work via binomial moments $S_k = \sum_i \sqrt{\varepsilon_i} \sigma_i^k$ ($k = 0 \dots 6$), avoiding dense $O(N_{\text{types}}^2)$ loops or geometric-mean approximations.

4. **Implicit-Solvent OBC2 GBSA + ACE Nonpolar SASA**:
   $$U_{\text{GB}} = -\frac{1}{2} C_{\text{Coul}} \sum_{i,j} q_i q_j \frac{\tau_{ij}(f_{\text{GB}})}{f_{\text{GB}}}, \qquad f_{\text{GB}} = \sqrt{r_{ij}^2 + R_i R_j \exp\!\left(-\frac{r_{ij}^2}{4 R_i R_j}\right)}$$
   with Debye-Hückel screening $\tau_{ij}(f_{\text{GB}}) = \varepsilon_{\text{in}}^{-1} - e^{-0.73 \kappa f_{\text{GB}}} \varepsilon_{\text{out}}^{-1}$ and ACE nonpolar cavitation energy:
   $$U_{\text{ACE}} = \sum_{i=1}^N 4\pi \sigma_{\text{np}} (r_i + r_p)^2 \left(\frac{r_i}{R_i}\right)^6, \qquad \sigma_{\text{np}} = 0.0054\text{ kcal/(mol}\cdot\text{Å}^2)$$

---

## 3. Anti-Fallback Benchmark Provenance (`openmm_cuda_guard`)

OpenMM's default `Platform.getPlatformByName("CUDA")` or automatic platform selection can fall back to `OpenCL` or `CPU` if `libOpenMMCUDA.so` fails to initialize on a cluster node—silently invalidating throughput comparisons.

Every benchmark and validation harness in Prolix calls `require_openmm_cuda_context` (or `run_openmm_cuda_mps_replicas`) from `scripts/benchmarks/openmm_cuda_guard.py`, which:
1. Explicitly requests `Platform.getPlatformByName("CUDA")` with explicit `{"Precision": precision}`.
2. Verifies `context.getPlatform().getName() == "CUDA"` and inspects `/proc/self/maps` on Linux to confirm `libOpenMMCUDA.so` is loaded in memory.
3. Stamps 7 mandatory provenance fields (`openmm_platform_name`, `openmm_precision`, `openmm_device_name`, `openmm_device_index`, `openmm_cuda_lib_loaded`, `openmm_plugin_errors`, `openmm_version`) into every output JSON artifact, raising `RuntimeError` immediately on any mismatch.

---

## 4. Four-Tier Multi-Engine Validation Summary

Prolix is continuously audited across a **four-tier validation hierarchy** against **OpenMM Reference (`float64`)**, **OpenMM CUDA (`single` / `mixed` / `double`)**, and **kUPS (`float64` / `float32`)** (full mathematical derivations and audit evidence in `docs/validation/261111_math_and_physics_audit.md` and {doc}`../explicit_solvent/openmm_comparison_protocol`):

| Tier | Scope | Oracle Engines | Acceptance Gate |
| :--- | :--- | :--- | :--- |
| **Tier 1: Term-by-Term Static Energy & Force Parity (`float64`)** | Isolated bonds, angles, proper/RB/improper/CMAP torsions, 1-4 exceptions, direct + SPME electrostatics, switched LJ + dispersion tail, OBC2 GB + ACE SASA, virtual sites | OpenMM `Reference` & `kUPS` (`float64`) | $\|\Delta E\| < 10^{-5}\text{ kcal/mol}$, max per-atom $\|\Delta \mathbf{F}_i\| < 10^{-4}\text{ kcal/(mol·Å)}$ |
| **Tier 2: Production GPU Kernel Parity (`single` & `mixed`)** | Morton $32 \times 32$ tile-pairs (`MK-51`/`MK-65`), SRAM SPME (`MK-70`), 2-pass tile OBC2 GBSA (`MK-67`), 1D ragged queue (`MK-68`) across `2k–408k` atoms | OpenMM `CUDA` (`single` & `mixed`) + Prolix `float64` reference | Relative energy $< 5 \times 10^{-5}$; RMS force deviation within SPME B-spline / FP32 rounding floor (`< 10^{-2} kcal/(mol·Å)`) |
| **Tier 3: Analytical `@jax.custom_vjp` Gradient Parity** | `jax.grad` wrt coordinates $\mathbf{r}$, charges $q$, LJ $(\sigma, \varepsilon)$, GBSA radii, bonded parameters, and OpenMM global context derivatives | Central finite differences (`float64`) & pure-JAX autodiff | Max relative gradient error $< 10^{-6}$ (`float64`) / $< 5 \times 10^{-4}$ (`float32`), $O(N \cdot K)$ peak VRAM |
| **Tier 4: Thermodynamic Ensemble & Integrator Audit** | NVE energy conservation, NVT kinetic temperature mean & variance ($\sigma_T = T \sqrt{2 / N_{\text{dof}}}$), cold-start SETTLE/RATTLE velocity projection, NPT equilibrium density & volume fluctuations (`mc` & `scr`) | Analytical Boltzmann / $\chi^2_{N_{\text{dof}}}$ theory, OpenMM `CUDA` `LangevinMiddleIntegrator`, & `kUPS` BAOAB/CSVR | $|\langle T \rangle - T_{\text{target}}| < 1.5\text{ K}$, exact $N_{\text{dof}}$ masked kinetic variance, stable 2.0 fs `HBonds`+`SETTLE` trajectories |
