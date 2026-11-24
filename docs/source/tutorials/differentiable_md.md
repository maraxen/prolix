# Tutorial 3: Differentiable MD & Force-Field Optimization

Every potential energy term and Pallas GPU kernel in Prolix—bonded harmonics, periodic/RB/CMAP torsions, Morton $32 \times 32$ direct-space nonbonded tiles, smooth particle-mesh Ewald (SPME) B-spline spread/gather, two-pass OBC2 Generalized Born + ACE nonpolar SASA, and virtual-site reconstructions—implements analytical `@jax.custom_vjp` backward rules in **$O(N \cdot K)$ memory** without ever materializing dense $(N, N)$ pair matrices.

This enables three complementary differentiation workflows:

1. **Pure-JAX Coordinate & Force-Field Parameter Gradients** via {func}`~prolix.physics.make_energy_fn_pure` and {class}`~prolix.typing.DifferentiableParams`.
2. **Batched OpenMM Coordinate Gradients** via `ProlixBatchSimulation.value_and_grad_positions`.
3. **OpenMM Global Context Parameter Derivatives** ($\partial E / \partial \lambda$) via `ProlixState.getEnergyParameterDerivatives()` and `ProlixBatchSimulation.compute_parameter_derivatives()`.

---

## 1. Differentiating Coordinates & Force-Field Parameters (`make_energy_fn_pure`)

Unlike closure-based energy functions, {func}`~prolix.physics.make_energy_fn_pure` separates topological indices (bonds, angles, exclusions) from continuous physical parameters stored in an `equinox.Module` PyTree ({class}`~prolix.typing.DifferentiableParams`):

- `params.charges`: Partial atomic charges `(N,)` in elementary charge units ($e$)
- `params.sigmas`, `params.epsilons`: Lennard-Jones 12-6 parameters `(N,)` in `Å` and `kcal/mol`
- `params.bond_params`: Harmonic bond `[r0, k]` `(N_bonds, 2)`
- `params.angle_params`: Harmonic angle `[theta0, k]` `(N_angles, 2)`
- `params.dihedral_params`: Periodic torsion `[periodicity, phase, k]` `(N_dihedrals, 3)`
- `params.improper_params`, `params.urey_bradley_params`
- `params.exception_sigmas`, `params.exception_epsilons`, `params.exception_chargeprods`: 1-4 scaled pair overrides
- `params.pme_alpha`, `params.box_size`

### Simultaneous Coordinate Forces and Parameter Sensitivities

```python
import jax
import jax.numpy as jnp
from jax_md import space
from prolix.physics import DifferentiableParams, make_energy_fn_pure

# Suppose `physics_system` is a Prolix PhysicsSystem (e.g., from proxide or converter)
box = jnp.asarray(physics_system.box_size)
displacement_fn, _ = space.periodic(box)

# Build pure, stateless energy callable: (DifferentiableParams, positions) -> scalar kcal/mol
params, energy_fn = make_energy_fn_pure(
    displacement_fn,
    physics_system,
    cutoff_distance=9.0,
    pme_grid_points=64,
    use_pallas=True,
)

# Differentiate simultaneously wrt force-field parameters (arg 0) and coordinates (arg 1)
val_and_grad = jax.jit(jax.value_and_grad(energy_fn, argnums=(0, 1)))
energy_kcal, (dE_dparams, dE_dpos) = val_and_grad(params, positions_angstrom)

# Atomic forces in kcal/(mol·Å) are simply the negative coordinate gradient
forces_akma = -dE_dpos

print(f"E = {float(energy_kcal):.4f} kcal/mol")
print(f"||dE/dq||_2     = {float(jnp.linalg.norm(dE_dparams.charges)):.4f} kcal/(mol·e)")
print(f"||dE/dsigma||_2 = {float(jnp.linalg.norm(dE_dparams.sigmas)):.4f} kcal/(mol·Å)")
print(f"||dE/deps||_2   = {float(jnp.linalg.norm(dE_dparams.epsilons)):.4f}")
```

---

## 2. Fitting Force-Field Parameters to QM Energies & Forces with `optax`

Because `energy_fn` supports nested differentiation (computing forces $-\nabla_{\mathbf{r}} E(\theta, \mathbf{r})$ inside a parameter loss $\mathcal{L}(\theta)$), you can optimize partial charges, Lennard-Jones radii, and torsion barriers directly against reference DFT/QM datasets:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

# Freeze topological/box fields if desired; optimize charges, LJ epsilons, and torsions
def make_masked_params(base: DifferentiableParams, trainable: DifferentiableParams) -> DifferentiableParams:
    # Enforce exact total-charge conservation on trainable charges
    net_q_delta = jnp.mean(trainable.charges - base.charges)
    conserved_q = trainable.charges - net_q_delta
    return eqx.tree_at(
        lambda p: (p.charges, p.epsilons, p.dihedral_params),
        base,
        (conserved_q, jnp.nn.softplus(trainable.epsilons), trainable.dihedral_params),
    )

def force_matching_loss(
    trainable: DifferentiableParams,
    base: DifferentiableParams,
    batch_coords: jnp.ndarray,   # (B, N, 3) Å
    qm_energies: jnp.ndarray,    # (B,) kcal/mol
    qm_forces: jnp.ndarray,      # (B, N, 3) kcal/(mol·Å)
    w_force: float = 10.0,
) -> jnp.ndarray:
    eff_params = make_masked_params(base, trainable)

    def single_frame(r: jnp.ndarray):
        e, grad_r = jax.value_and_grad(lambda x: energy_fn(eff_params, x))(r)
        return e, -grad_r

    pred_e, pred_f = jax.vmap(single_frame)(batch_coords)
    # Center relative conformational energies before MSE
    de_pred = pred_e - jnp.mean(pred_e)
    de_qm = qm_energies - jnp.mean(qm_energies)
    loss_e = jnp.mean((de_pred - de_qm) ** 2)
    loss_f = jnp.mean((pred_f - qm_forces) ** 2)
    return loss_e + w_force * loss_f

optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(params)

@jax.jit
def train_step(trainable, opt_state, coords, qm_e, qm_f):
    loss, grads = jax.value_and_grad(force_matching_loss)(
        trainable, params, coords, qm_e, qm_f
    )
    updates, opt_state = optimizer.update(grads, opt_state, trainable)
    trainable = optax.apply_updates(trainable, updates)
    return trainable, opt_state, loss
```

---

## 3. Batched Coordinate & Alchemical Parameter Gradients in `prolix.openmm`

When working with OpenMM `System` objects—including custom alchemical Lepton potentials (`CustomNonbondedForce`, `CustomBondForce`) parameterized by global context variables such as `lambda_sterics` or `lambda_electrostatics`—`prolix.openmm` exposes analytical derivatives in OpenMM units (`kJ/mol` and `kJ/(mol·nm)`):

### Single-Context Global Parameter Derivatives ($\partial E / \partial \lambda$)

```python
import openmm
from openmm import unit
from prolix.openmm import ProlixContext, ProlixPlatform

# Example: CustomExternalForce with a global stiffness parameter `k_restraint`
restraint = openmm.CustomExternalForce("0.5 * k_restraint * (x^2 + y^2 + z^2)")
restraint.addGlobalParameter("k_restraint", 100.0)
restraint.addEnergyParameterDerivative("k_restraint")
for atom_idx in range(10):
    restraint.addParticle(atom_idx, [])
system.addForce(restraint)

ctx = ProlixContext(
    system,
    openmm.VerletIntegrator(0.001 * unit.picoseconds),
    ProlixPlatform(),
    {"UsePallas": "true"},
)
ctx.setPositions(pdb.positions)

# Sweep global parameter without recompiling and read exact dE/dk_restraint
for k_val in (50.0, 100.0, 200.0):
    ctx.setParameter("k_restraint", k_val)
    state = ctx.getState(getEnergy=True, getParameterDerivatives=True)
    derivs = state.getEnergyParameterDerivatives()
    print(f"k = {k_val:6.1f} -> dE/dk = {derivs['k_restraint']:.6f} kJ/mol")
```

### Multi-System Batched Gradients (`ProlixBatchSimulation`)

`ProlixBatchSimulation` provides vectorized helper methods across an entire batch of OpenMM systems:

```python
# 1. Batched (energies_kj_mol, grads_positions_kj_mol_nm) across all replicas
energies_kj, grads_nm = batch.value_and_grad_positions()
for rep_idx in range(min(3, len(energies_kj))):
    # Note: force = -grads_nm[rep_idx]
    print(f"Replica {rep_idx}: E = {float(energies_kj[rep_idx]):.2f} kJ/mol, grad shape = {grads_nm[rep_idx].shape}")

# 2. Batched global parameter derivatives [{param_name: dE_dp_kj_mol}, ...]
param_derivs = batch.compute_parameter_derivatives()
```
