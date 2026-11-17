# Tutorial 2: Multi-Replica & Ragged Ligand Screening

Single-trajectory GPU MD engines leave substantial streaming-multiprocessor capacity idle on small-to-medium biomolecular systems (`2k–25k` atoms), while launching dozens of independent processes under **NVIDIA `CUDA MPS`** duplicates CUDA contexts, FFT plans, and JIT caches (`~450–580 MB/replica`, hitting VRAM walls on 48 GB GPUs around `R = 64`).

Prolix solves multi-system throughput natively inside a single GPU context via two complementary interfaces:

- **{class}`~prolix.openmm.ProlixBatchSimulation`**: OpenMM-native multi-system runner accepting `openmm.System` and `openmm.Integrator` objects directly (`nm`, `ps`, `kJ/mol`).
- **{class}`~prolix.api.EnsemblePlan`** (`EnsemblePlan.from_openmm_systems` / `EnsemblePlan.from_bundles`): JAX-native ensemble planner returning device-resident `Trajectory` PyTrees and streaming `Observable` reductions (`Å`, `fs`, `kcal/mol`).

---

## Three Batching Regimes Supported Automatically

| Regime | System Topology Pattern | Prolix Dispatch Strategy | Marginal VRAM / Replica |
| :--- | :--- | :--- | :--- |
| **1. Homogeneous Ensemble** | $R$ replicas of the same protein/water system (e.g., replica-exchange, MSM seeding) | Batched 2D Pallas grid `(ceil(N/32), R)` + batched `cuFFT` SPME | **~45 MB/rep** |
| **2. Congeneric Series (`MK-64`)** | Protein + $R$ congeneric ligands or point mutants padded to a shared atom bucket | Replica-invariant bond/angle/torsion/exclusion index tables deduplicated at **`stride = 0`**; per-replica charges/LJ/coordinates | **~45–55 MB/rep** |
| **3. Cross-Size Ragged (`MK-68`)** | Arbitrary panel of systems spanning `2k` to `50k+` atoms (`Trp-cage`, `Villin`, `DHFR`, `Kinase`) | Flattened **1D prefix-sum ragged CTA work queue** (`pallas_ragged_queue`) + cost-modeled multi-tier SPME (`pme_tier_mode="auto"`) | Proportional to $\sum_r N_r$ |

---

## 1. OpenMM-Native Screening with `ProlixBatchSimulation`

`ProlixBatchSimulation` accepts a sequence of `openmm.System` instances (homogeneous, congeneric, or ragged), either a single broadcast `openmm.Integrator` or per-replica integrators (e.g., a temperature ladder), and a list of initial coordinate arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)

import openmm
from openmm import unit
from prolix.openmm import ProlixBatchSimulation, ProlixOpenMMConfig

# Suppose we have 32 congeneric protein-ligand OpenMM Systems & initial positions
# (each System may differ slightly in ligand atom count, charges, and torsions!)
systems: list[openmm.System] = [...]
positions_list: list[unit.Quantity] = [...]

integrator = openmm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds,
)

config = ProlixOpenMMConfig(
    precision="mixed",
    nonbonded_engine="auto",   # Morton 32x32 tiles + ragged queue when sizes differ
    use_pallas=True,
    use_pallas_pme=True,
    nl_update_every=20,
)

batch = ProlixBatchSimulation(
    systems=systems,
    integrators=integrator,
    positions_list=positions_list,
    config=config,
)

# Initialize Maxwell-Boltzmann velocities across all 32 complexes
batch.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=2026)

# Integrate 25,000 steps (50 ps per replica = 1.6 ns aggregate) in one fused call
batch.step(25_000)

# Retrieve native ProlixState objects (trimmed to each ligand system's exact atom count)
states = batch.getStates(getPositions=True, getForces=True, getEnergy=True)
for idx, state in enumerate(states[:4]):
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    n_atoms = state.getPositions(asNumpy=True).shape[0]
    print(f"Complex {idx:02d} ({n_atoms} atoms): PE = {pe:.2f} kJ/mol")
```

### Multi-Temperature Ensembles

Pass a list of `openmm.Integrator` objects to run a parallel temperature or friction sweep in a single fused kernel launch:

```python
temps_k = [280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0]
integrators = [
    openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    for T in temps_k
]

replica_batch = ProlixBatchSimulation(
    systems=[base_system] * len(temps_k),
    integrators=integrators,
    positions_list=[base_positions] * len(temps_k),
    config=config,
)
replica_batch.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=7)
replica_batch.step(50_000)
```

---

## 2. JAX-Native Screening & Observables with `EnsemblePlan.from_openmm_systems`

When feeding trajectories directly into JAX downstream analysis, Markov state models, or ML pipelines without round-tripping through `openmm.unit.Quantity`, construct an {class}`~prolix.api.EnsemblePlan` via `EnsemblePlan.from_openmm_systems`:

```python
import jax.numpy as jnp
from prolix.api import EnsemblePlan, KineticEnergy, RMSD, Temperature
from prolix.openmm import ProlixOpenMMConfig
from prolix.simulate import BOLTZMANN_KCAL

# Convert heterogeneous or congeneric OpenMM systems into a unified EnsemblePlan
plan = EnsemblePlan.from_openmm_systems(
    systems,
    positions_list,
    integrator=integrator,
    align_buckets=True,
    config=ProlixOpenMMConfig(
        precision="mixed",
        nonbonded_engine="auto",
        use_pallas=True,
        use_pallas_pme=True,
    ),
)

# Attach streaming JAX Observables (evaluated on-device during integration)
ref_bundle = plan.bundles[0]
n_real = int(jnp.sum(ref_bundle.atom_mask))
n_constraints = int(jnp.sum(ref_bundle.constraint_mask)) if ref_bundle.constraint_mask is not None else 0
dof = 3 * n_real - n_constraints - 3

observables = {
    "kinetic_energy": KineticEnergy(atom_mask=ref_bundle.atom_mask),
    "temperature": Temperature(dof=dof, atom_mask=ref_bundle.atom_mask),
    "backbone_rmsd": RMSD(reference=ref_bundle.positions),
}

# Run 10,000 steps (dt = 2.0 fs, T = 300 K -> kT in kcal/mol, gamma = 1.0 ps^-1)
trajectories = plan.run(
    n_steps=10_000,
    dt=2.0,
    kT=300.0 * BOLTZMANN_KCAL,
    seed=42,
    gamma=1.0,
    run_mode="inference",
    observables=observables,
)

for i, traj in enumerate(trajectories[:3]):
    mean_T = float(jnp.mean(traj.observable_values["temperature"]))
    final_rmsd = float(jnp.asarray(traj.observable_values["backbone_rmsd"]).reshape(-1)[-1])
    print(f"Replica {i}: <T> = {mean_T:.1f} K | final RMSD = {final_rmsd:.2f} Å")
```

:::{important}
**Unit & `N_dof` Invariant on `EnsemblePlan.run`:**
- `EnsemblePlan.run` accepts `dt` in **femtoseconds** (`dt_unit="fs"` default), `kT` in **`kcal/mol`** (`T_kelvin * 0.0019872041`), and `gamma` in **`ps⁻¹`**. Returned `Trajectory.positions` are in **Ångströms** (`(n_saved, n_real_atoms, 3)`).
- Always pass `atom_mask` and real-atom `dof` to {class}`~prolix.api.Temperature`: padded bucket slots carry unit mass (`masses_with_prefix`) to avoid division-by-zero in velocity updates and must be masked out of kinetic thermometers.
:::

---

## 3. Tuning Multi-Replica Pallas Knobs

For extreme batch counts (`R = 64..256`) or tiny implicit-solvent peptides where CTA occupancy matters:

- **`tp_group`** (`int | None`, default `None` = adaptive `1 | 2 | 4 | 8`): Packs multiple $32$-atom target tiles per Triton CTA on small atom buckets (`N <= 4096`) to reduce launch and Morton-table overhead.
- **`nb_k_splits`** (`int | None`, default `None` = adaptive split-K): Splits long neighbor-tile interaction lists across cooperative warps when total replica count $R$ is small on high-SM GPUs (e.g., `H200` / `RTX PRO 6000 Blackwell`).
- **`deterministic_forces`** (`bool`, default `False`): Replaces atomic scatter-adds with fixed-order reductions when bitwise run-to-run force reproducibility is required for regression debugging.
