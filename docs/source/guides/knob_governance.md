# `KnobSurface` & `ProlixOpenMMConfig` Reference

Prolix centralizes every simulation, thermostat, barostat, nonbonded, implicit-solvent, holonomic constraint, cosolvent, alchemical softcore/restraint, collective-variable bias, and Pallas GPU kernel dispatch parameter onto a single zero-dependency leaf schema: {class}`prolix.knob_surface.KnobSurface`.

Static governance checks in {mod}`prolix.governance.redsox_adapter` (`tests/test_redsox_openmm_knob_governance.py`) enforce two compile-time guarantees across every PR:

- **U5 Superset Guarantee (`check_superset`)**: Every field declared on {class}`~prolix.openmm.ProlixOpenMMConfig`, `NonbondedConfig`, `NeighborListConfig`, and supported OpenMM `Platform`/`System`/`Integrator` properties resolves to a canonical `KnobSurface` leaf (directly or via `SUPERSET_ALIAS_MAP`) with **zero unmapped knobs**.
- **U1 Consumer Reachability (`check_prolix_reachability`)**: Every non-inert leaf declared on `KnobSurface` is AST-verified to be read inside runtime execution modules (`src/prolix/physics/`, `src/prolix/api/`, `src/prolix/openmm/` excluding declaration-only `config.py`, `src/prolix/pallas_kernels.py`, `src/prolix/batched_energy.py`, and `src/prolix/batched_simulate.py`), preventing dead or silently ignored configuration flags (`PENDING_EPIC_KNOBS == frozenset()`).

---

## 1. `KnobSurface` Composite Schema (9 Sub-Surfaces)

`KnobSurface` composes nine frozen `@dataclass` sub-surfaces. Leaf fields can be accessed hierarchically (`surface.nonbonded.cutoff_angstrom`), via flat delegation (`surface.cutoff_angstrom`), or serialized round-trip with `surface.to_flat_dict()` and `KnobSurface.from_flat_dict(...)`.

### 1.1 `ThermostatControls` (`surface.thermostat`)

| Field | Type | Default | Units / Allowed Values |
| :--- | :--- | :--- | :--- |
| `integrator_method` | `str` | `"settle_langevin"` | `"settle_langevin"`, `"baoab"`, `"langevin"`, `"verlet"`, `"brownian"`, `"nose_hoover"` |
| `temperature_k` | `float` | `300.0` | Kelvin (`K`) |
| `gamma_inv_ps` | `float` | `1.0` | Collision frequency (`ps⁻¹`) |
| `timestep_fs` | `float` | `2.0` | Integration step (`fs`) |
| `n_steps` | `int` | `1000` | Total MD steps |
| `save_interval` | `int` | `500` | Trajectory / observable cadence (steps) |
| `seed` | `int` | `0` | Deterministic PRNG seed |

### 1.2 `BarostatControls` (`surface.barostat`)

| Field | Type | Default | Units / Allowed Values |
| :--- | :--- | :--- | :--- |
| `pressure_atm` | `float \| None` | `None` | Target pressure (`atm`; `None` = NVT/NVE) |
| `barostat_method` | `str` | `"mc"` | `"mc"` (Monte Carlo) or `"scr"` (Stochastic Cell Rescaling) |
| `barostat_interval` | `int` | `25` | Volume move cadence (steps) |
| `anisotropic` | `bool` | `False` | Independent $(L_x, L_y, L_z)$ axis scaling |
| `compressibility_inv_bar` | `float` | `4.5e-5` | Isothermal compressibility (`bar⁻¹`, water at 300 K) |
| `tau_p_ps` | `float` | `1.0` | Barostat relaxation time (`ps`, used by `"scr"`) |

### 1.3 `NonbondedControls` (`surface.nonbonded`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `nonbonded_method` | `str` | `"pme"` | `"pme"`, `"cutoff_periodic"`, `"cutoff_nonperiodic"`, `"no_cutoff"`, `"dsf"` |
| `cutoff_angstrom` | `float` | `9.0` | Direct-space nonbonded cutoff (`Å`) |
| `switch_width_angstrom` | `float` | `1.0` | Smooth LJ switching window width (`Å`; `0.0` disables switch) |
| `ewald_error_tolerance` | `float` | `5e-4` | Target relative force error determining Ewald $\alpha$ and SPME mesh |
| `pme_alpha` | `float` | `0.34` | Ewald splitting parameter $\alpha$ (`Å⁻¹`) |
| `pme_grid_dims` | `tuple[int, int, int] \| None` | `None` | Explicit SPME mesh `(nx, ny, nz)` (`None` = auto smooth $2^a 3^b 5^c$) |
| `dispersion_correction` | `bool` | `True` | Analytical isotropic long-range Lennard-Jones tail energy & virial |
| `reaction_field_dielectric` | `float` | `78.3` | Solvent dielectric $\varepsilon_{\text{RF}}$ for `CutoffPeriodic` |
| `dsf_alpha` | `float \| None` | `None` | Damped shifted force damping (`Å⁻¹`) |
| `coulomb_constant` | `float` | `332.06371329919216` | Electrostatic prefactor $1/(4\pi\varepsilon_0)$ in `kcal·Å/(mol·e²)` |
| `scale_14_lj` | `float` | `0.5` | 1-4 intramolecular Lennard-Jones scaling (AMBER default) |
| `scale_14_elec` | `float` | `0.8333333333333334` | 1-4 intramolecular Coulomb scaling (`1 / 1.2`, AMBER default) |

### 1.4 `ImplicitSolventControls` (`surface.implicit_solvent`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `solvent_model` | `str \| None` | `None` | `None` (explicit/vacuum) or `"obc2"` |
| `solvent_dielectric` | `float` | `78.5` | Exterior solvent dielectric $\varepsilon_{\text{out}}$ |
| `solute_dielectric` | `float` | `1.0` | Interior solute dielectric $\varepsilon_{\text{in}}$ |
| `obc_alpha`, `obc_beta`, `obc_gamma` | `float` | `1.0, 0.8, 4.85` | Onufriev-Bashford-Case II (`OBC2`) $\tanh$ rescaling coefficients |
| `surface_tension_kcal_mol_a2` | `float` | `0.0054` | ACE nonpolar SASA tension (`kcal/(mol·Å²)`; `2.25936 kJ/(mol·nm²)`) |
| `probe_radius_angstrom` | `float` | `1.4` | Water probe radius $r_p$ (`Å`) |
| `dielectric_offset_angstrom` | `float` | `0.09` | Born intrinsic radius offset (`Å`) |
| `salt_concentration_m` | `float` | `0.0` | Monovalent salt molarity (`M`) |
| `kappa_inv_angstrom` | `float` | `0.0` | Debye-Hückel inverse screening length $\kappa$ (`Å⁻¹`) |

### 1.5 `ConstraintControls` (`surface.constraints`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `constraint_mode` | `str` | `"hbonds"` | `"none"`, `"hbonds"`, `"allbonds"`, `"hangles"` |
| `rigid_water` | `bool` | `True` | Enforce analytical **SETTLE** on water triangles |
| `r_oh_angstrom`, `r_hh_angstrom` | `float` | `0.9572, 1.5139` | Reference TIP3P water geometry (`Å`) |
| `shake_tolerance`, `rattle_tolerance` | `float` | `1e-5` | Relative holonomic position & velocity projection tolerances |
| `max_shake_iterations` | `int` | `50` | Maximum iterative SHAKE/RATTLE sweeps per step |

### 1.6 `CosolventControls`, `SoftcoreRestraintControls`, & `CollectiveVariableBiasControls`

- **`surface.cosolvent`**: Multi-component cosolvent placement and anti-aggregation repulsion (`cosolvent_entries`, `protein_radius_angstrom`, `cosolvent_placement_radius_angstrom`, `repulsive_eps_kcal`, `repulsive_r0_angstrom`, `repulsive_cutoff_angstrom`, `deduplicate_interaction_groups`).
- **`surface.softcore_restraints`**: Beutler alchemical $\lambda$ schedules and flat-bottom positional restraints (`lambda_schedule`, `softcore_alpha`, `softcore_exponent`, `soft_coulomb`, `restraint_selection`, `restraint_k_kcal_mol_a2`, `flat_bottom_radius_angstrom`, `per_stage_max_iterations`, `tolerance_kcal_mol_a`).
- **`surface.cv_bias`**: Differentiable collective variables, umbrella sampling, steered MD, and well-tempered metadynamics (`cv_specs`, `bias_mode`, `umbrella_k`, `umbrella_center`, `smd_velocity`, `metad_height`, `metad_sigma`, `metad_bias_factor`, `metad_pace`, `plumed_min_beta_nm`).

### 1.7 `PallasControls` (`surface.pallas`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `use_pallas` | `bool \| None` | `None` | Enable Pallas Triton GPU kernels (`None` = auto-enable on CUDA GPUs) |
| `use_pallas_pme` | `bool \| None` | `None` | Enable fused SRAM B-spline SPME spread/gather kernels |
| `gb_use_neighbor_list` | `bool` | `False` | Cutoff neighbor-list mode for implicit solvent (`False` = exact symmetric $32 \times 32$ tiles) |
| `precision` | `str` | `"single"` | `"single"`, `"mixed"`, or `"double"` |
| `nonbonded_engine` | `str` | `"auto"` | `"auto"`, `"morton_tile_pairs"`, `"neighbor_list"`, `"gbsa_tile"`, `"flash"`, `"dense"` |
| `pallas_interpret` | `bool` | `False` | Run Pallas kernels in CPU interpret mode for debugging |
| `nb_k_splits` | `int \| None` | `None` | Cooperative split-K override (`None` = adaptive per-GPU SM count) |
| `tp_group` | `int \| None` | `None` | Multi-tile CTA packing factor `1 \| 2 \| 4 \| 8` (`None` = adaptive) |
| `deterministic_forces` | `bool` | `False` | Deterministic force reduction order (disables atomic scatter reorder) |
| `disable_pme_stream` | `bool` | `False` | Serialize SPME reciprocal stream onto primary compute stream |
| `custom_force_mode` | `str` | `"auto"` | `"auto"`, `"lepton_only"`, or `"pattern_only"` |
| `custom_nonbonded_chunk_size` | `int` | `128` | Tile chunk size for Lepton `CustomNonbondedForce` |
| `use_size_buckets` | `bool` | `True` | Bucket-pad bundles for zero-retrace multi-system XLA caching |
| `max_steps_per_scan` | `int` | `1000` | Maximum device `lax.scan` chunk length |
| `execution_mode` | `str` | `"inference"` | `"inference"` (`while_loop` carry) or `"trajectory"` (`scan` stack) |
| `minimizer_backend` | `str` | `"lbfgs"` | `"lbfgs"`, `"fire"`, or `"openmm_cpu"` |

---

## 2. `ProlixOpenMMConfig` & OpenMM Platform Properties

When creating a `ProlixPlatform`, `ProlixContext`, `ProlixSimulation`, or `ProlixBatchSimulation`, configuration can be passed either as a strongly typed {class}`~prolix.openmm.ProlixOpenMMConfig` or as an OpenMM `platformProperties: dict[str, str]` dictionary (`CamelCase` or `snake_case`, parsed by {func}`~prolix.openmm.parse_platform_properties`):

| `ProlixOpenMMConfig` Field | OpenMM `Platform` Property Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `precision` | `"Precision"` | `"single"` | `"single"`, `"mixed"`, or `"double"` |
| `nonbonded_engine` | `"NonbondedEngine"` | `"auto"` | `"auto"`, `"morton_tile_pairs"`, `"neighbor_list"`, `"gbsa_tile"`, `"flash"`, `"dense"` |
| `use_pallas` | `"UsePallas"` | `False` / `"true"` | Dispatch Pallas Triton GPU kernels |
| `use_pallas_pme` | `"UsePallasPme"` | `False` / `"true"` | Dispatch Pallas SRAM SPME spread/gather |
| `pallas_interpret` | `"PallasInterpret"` | `False` | Run Pallas kernels in JAX CPU interpret mode (debugging) |
| `pme_grid_spacing` | `"PmeGridSpacing"` | `1.0` | Target SPME mesh spacing (`Å`) when `System` leaves grid unset |
| `ewald_error_tolerance` | `"EwaldErrorTolerance"` | `5e-4` | Fallback Ewald force tolerance |
| `lj_switch_width` | `"LjSwitchWidth"` | `1.5` | Fallback LJ switch width (`Å`) |
| `use_dispersion_correction` | `"UseDispersionCorrection"` | `True` | Apply analytical continuum LJ tail energy & pressure correction |
| `nl_skin` | `"NlSkin"` | `2.0` | Verlet / Morton AABB buffer skin (`Å`) |
| `nl_capacity_multiplier` | `"NlCapacityMultiplier"` | `1.5` | Neighbor-list headroom factor (`> 1.0`) |
| `nl_update_every` | `"NlUpdateEvery"` | `10` | Fused persistent device steps per displacement skin check |
| `custom_force_mode` | `"CustomForceMode"` | `"auto"` | `"auto"`, `"lepton_only"`, `"pattern_only"` |
| `integrator_scheme` | `"IntegratorScheme"` | `"auto"` | `"auto"`, `"baoab"`, `"leapfrog"`, `"langevin"`, `"verlet"`, `"brownian"`, `"nose_hoover"`, `"lfmiddle"` |
| `barostat_mode` | `"BarostatMode"` | `"auto"` | `"auto"`, `"mc"` (Monte Carlo), `"scr"` (Stochastic Cell Rescaling) |
| `water_constraint_mode` | `"WaterConstraintMode"` | `"settle"` | `"settle"` or `"none"` |
| `solute_constraint_mode` | `"SoluteConstraintMode"` | `"shake"` | `"shake"` or `"none"` |
| `minimizer_backend` | `"MinimizerBackend"` | `"lbfgs"` | `"lbfgs"`, `"fire"`, `"openmm_cpu"` |
| `deterministic_seed` | `"DeterministicSeed"` | `42` | Default PRNG seed when OpenMM integrator seed is `0` |

---

## 3. OpenMM Integrator Parameter Mapping

`prolix.openmm` inspects the concrete `openmm.Integrator` class attached to a `Context` or `Simulation` and maps its parameters into internal AKMA quantities (`τ_AKMA = 48.88821291839 fs`, `k_B = 0.0019872041 kcal/(mol·K)`):

| OpenMM Integrator Class | Prolix `integrator_scheme` (`"auto"`) | Mapped OpenMM Getters | Notes |
| :--- | :--- | :--- | :--- |
| `openmm.LangevinMiddleIntegrator` | `"baoab"` / `"lfmiddle"` | `getTemperature()`, `getFriction()`, `getStepSize()`, `getRandomNumberSeed()`, `getConstraintTolerance()` | **Recommended production default** (`dt = 2.0 fs` with `HBonds` + `SETTLE`; exact configurational temperature distribution). |
| `openmm.LangevinIntegrator` | `"langevin"` | `getTemperature()`, `getFriction()`, `getStepSize()`, `getRandomNumberSeed()`, `getConstraintTolerance()` | Velocity-Verlet / OBABO Langevin splitting. |
| `openmm.VerletIntegrator` | `"verlet"` | `getStepSize()`, `getConstraintTolerance()` | Symplectic NVE Velocity-Verlet (`v(t)` synchronized at full steps). |
| `openmm.BrownianIntegrator` | `"brownian"` | `getTemperature()`, `getFriction()`, `getStepSize()`, `getRandomNumberSeed()` | Overdamped diffusive dynamics ($\Delta r = \frac{\Delta t}{m \gamma} F + \sqrt{\frac{2 k_B T \Delta t}{m \gamma}} \xi$). |
| `openmm.NoseHooverIntegrator` | `"nose_hoover"` | `getTemperature()`, `getCollisionFrequency()`, `getStepSize()`, chain length | Deterministic Nosé-Hoover chain NVT thermostat. |
| `openmm.CustomIntegrator` (CSVR / BAOAB) | Pattern-matched or Lepton | Global/per-DOF variables | Stochastic velocity rescaling (Bussi-Donadio-Parrinello exact $\chi^2_{N_{\text{dof}}}$ kinetic update). |

---

## 4. Running `redsox` Governance Checks Programmatically

Contributors adding new physics or OpenMM platform options can run U5 and U1 checks directly in Python or via `pytest`:

```python
from prolix.governance.redsox_adapter import (
    check_prolix_reachability,
    check_superset,
)
from prolix.knob_surface import KnobSurface
from prolix.openmm import ProlixOpenMMConfig

# U5: Verify KnobSurface covers 100% of ProlixOpenMMConfig
u5_report = check_superset(KnobSurface, ProlixOpenMMConfig)
assert u5_report.passed, f"Unmapped knobs: {u5_report.missing_fields}"

# U1: Verify every KnobSurface leaf is consumed in runtime modules
u1_report = check_prolix_reachability()
assert u1_report.passed, f"Unread/unwired knobs: {u1_report.unread_fields}"
```
