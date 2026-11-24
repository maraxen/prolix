# Tutorial 1: Drop-In OpenMM Acceleration (`prolix.openmm`)

`prolix.openmm` provides a drop-in JAX and Pallas GPU execution backend for standard OpenMM `System`, `Topology`, and `Integrator` objects. Existing OpenMM scripts switch to Prolix by importing `Simulation`, `Context`, `Platform`, and `State` (or their explicit `Prolix*` names) from `prolix.openmm`.

---

## 1. Drop-In `ProlixSimulation` Workflow

Given any solvated PDB and OpenMM `ForceField`, construct your `openmm.System` and `openmm.Integrator` exactly as usual, then instantiate {class}`~prolix.openmm.ProlixSimulation`:

```python
import jax
jax.config.update("jax_enable_x64", True)  # Required for precision="mixed" or "double"

import openmm
from openmm import app, unit
from prolix.openmm import ProlixOpenMMConfig, ProlixPlatform, ProlixSimulation

# 1. Standard OpenMM topology & force-field parameterization
pdb = app.PDBFile("dhfr_solvated.pdb")
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.9 * unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
    ewaldErrorTolerance=5e-4,
)

# 2. Standard OpenMM BAOAB / LangevinMiddleIntegrator (dt = 2.0 fs)
integrator = openmm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds,
)

# 3. Configure Prolix via ProlixPlatform (OpenMM string dict) or ProlixOpenMMConfig
config = ProlixOpenMMConfig(
    precision="mixed",          # fp64 state/SETTLE + fp32 Pallas force/SPME kernels
    nonbonded_engine="auto",    # Morton 32x32 tile-pairs + SRAM SPME on orthorhombic PBC
    use_pallas=True,
    use_pallas_pme=True,
    nl_update_every=20,         # Persistent fused 20-step device chunks between skin checks
    minimizer_backend="lbfgs",  # GPU jaxopt.LBFGS minimizer
)

sim = ProlixSimulation(pdb.topology, system, integrator, config=config)
sim.context.setPositions(pdb.positions)

# 4. GPU Energy Minimization + Thermalization
sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=500)
sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)

# 5. Attach standard OpenMM reporters (StateDataReporter, DCDReporter, XTCReporter)
sim.reporters.append(
    app.StateDataReporter(
        "production.log",
        reportInterval=1000,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
    )
)

# 6. Run fused device integration (zero host synchronization between reporter intervals!)
sim.step(10_000)
```

### Fused Multi-Step Execution Bypass

When `sim.reporters` is empty—or between `reportInterval` boundaries—`ProlixSimulation.step(N)` dispatches all steps directly inside a compiled JAX/Pallas `lax.scan` + persistent `nl_update_every` megakernel without returning to Python on intermediate steps. Standard OpenMM reporters (`DCDReporter`, `StateDataReporter`, `CheckpointReporter`) are queried only at exact multiples of their `reportInterval`, requesting the minimal union of state flags (`positions`, `velocities`, `forces`, `energy`) needed at that step.

---

## 2. Direct `ProlixContext` & `ProlixPlatform` Usage

For low-level control over forces, force groups, periodic boxes, and analytical parameter derivatives without `openmm.app.Topology`, instantiate {class}`~prolix.openmm.ProlixContext` directly or pass OpenMM string properties via {class}`~prolix.openmm.ProlixPlatform`:

```python
import numpy as np
import openmm
from openmm import unit
from prolix.openmm import ProlixContext, ProlixPlatform

platform = ProlixPlatform()
props = {
    "Precision": "mixed",
    "NonbondedEngine": "auto",
    "UsePallas": "true",
    "UsePallasPme": "true",
    "NlSkin": "2.0",              # Ångströms
    "NlUpdateEvery": "20",
}

ctx = ProlixContext(system, integrator, platform, props)
ctx.setPositions(pdb.positions)
ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 1234)

# Advance 500 steps directly on GPU
ctx.step(500)

# Query state in native openmm.unit.Quantity (nm, ps, kJ/mol, kJ/(mol·nm))
state = ctx.getState(
    getPositions=True,
    getVelocities=True,
    getForces=True,
    getEnergy=True,
    enforcePeriodicBox=True,
)
pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
forces_omm = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
pe_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
print(f"PE = {pe_kj:.2f} kJ/mol | max|F| = {np.linalg.norm(forces_omm, axis=-1).max():.1f} kJ/(mol·nm)")
```

### Selective Force-Group Evaluation

Like native OpenMM, `getState(..., groups=...)` accepts either a bitmask integer or a Python `set[int]` of force-group indices (`0..31`), enabling decomposed energy inspection (e.g., separating bonded, direct-space nonbonded, and custom restraint groups):

```python
# Evaluate only ForceGroup 0 and ForceGroup 2
sub_state = ctx.getState(getEnergy=True, getForces=True, groups={0, 2})
```

---

## 3. Supported OpenMM Forces, Constraints, & Ensembles

`prolix.openmm.converter.from_openmm_system` translates OpenMM `System` definitions into static, bucket-padded `MolecularBundle` PyTrees while converting all physical constants from OpenMM (`nm`, `ps`, `kJ/mol`) to internal AKMA (`Å`, `τ_AKMA`, `kcal/mol`) at conversion time:

| Category | Supported OpenMM Classes | Prolix Execution Path |
| :--- | :--- | :--- |
| **Bonded** | `HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`, `RBTorsionForce`, `CMAPTorsionForce` | Fused Pallas bonded kernel (`pallas_bonded_all_fused`) + bicubic spline CMAP |
| **Explicit Nonbonded** | `NonbondedForce` (`PME`, `CutoffPeriodic`, `CutoffNonPeriodic`, `NoCutoff`) | Morton $32 \times 32$ Newton-pair tiles + smooth $2^a 3^b 5^c$ SRAM SPME + analytical LJ tail |
| **Implicit Solvent** | `GBSAOBCForce`, `CustomGBForce` (OBC-II pattern) | Two-pass $32 \times 32$ symmetric tile OBC2 Generalized Born + ACE nonpolar SASA |
| **Custom Forces (Lepton)** | `CustomBondForce`, `CustomAngleForce`, `CustomTorsionForce`, `CustomExternalForce`, `CustomCompoundBondForce`, `CustomNonbondedForce` | Symbolic Lepton AST → vectorized JAX/XLA compiler (`prolix.openmm.lepton_compiler`) |
| **Constraints & Virtual Sites** | `HBonds`, `AllBonds`, `HAngles`, rigid 3-site/4-site waters (`TIP3P`, `SPC/E`, `TIP4P-Ew`, `OPC`), `TwoParticleAverageSite`, `ThreeParticleAverageSite`, `OutOfPlaneSite`, `LocalCoordinatesSite` | Analytical **SETTLE** for rigid water + iterative **SHAKE/RATTLE** for solute bonds + differentiable virtual-site frame reconstruction & force redistribution |
| **Integrators & Barostats** | `LangevinMiddleIntegrator` (`BAOAB`), `LangevinIntegrator`, `VerletIntegrator`, `BrownianIntegrator`, `NoseHooverIntegrator`, `MonteCarloBarostat` | Inter-step fused BAOAB + SETTLE + RATTLE megakernel; Monte Carlo (`mc`) or Stochastic Cell Rescaling (`scr`) NPT barostats |

:::{tip}
If an `openmm.System` contains an unrecognized `Force` subclass, `from_openmm_system` raises {class}`~prolix.openmm.UnsupportedOpenMMForceError` immediately at context construction rather than silently dropping energy terms.
:::

---

## 4. Choosing a Minimizer Backend

`ProlixOpenMMConfig(minimizer_backend=...)` supports three local energy minimizers:

1. **`"lbfgs"` *(default)***: Quasi-Newton `jaxopt.LBFGS` with backtracking line search entirely on GPU, followed by virtual-site reconstruction and holonomic constraint projection (`applyConstraints`).
2. **`"fire"`**: Fast Inertial Relaxation Engine (`jax_md.minimize.fire_descent`), resilient on severely clashed initial structures before switching to `"lbfgs"` or `BAOAB`.
3. **`"openmm_cpu"`**: Delegates minimization to `openmm.LocalEnergyMinimizer` on the OpenMM `Reference` platform and transfers relaxed coordinates back to the GPU context.
