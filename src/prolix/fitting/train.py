"""Training loops for bonded-parameter fitting.

Implements single-molecule training step, scan-based training loop,
and both looped and batched orchestration strategies.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from prolix.fitting.loss import bonded_loss, default_sigma
from prolix.fitting.params import BondedParams
from prolix.fitting.state import TrainState
from prolix.fitting.topology import BondedTopology

if TYPE_CHECKING:
    from prolix.fitting.batched import BondedParamsBundle, BondedTopologyBundle


class TrainMetrics(NamedTuple):
    """Metrics collected during a training step."""

    step: int
    loss: float
    loss_force: float
    loss_energy: float
    loss_reg: float


def train_step_one_mol(
    state: TrainState,
    conformer_idx: int,
    *,
    positions_all: Float[Array, "N_conf N_atoms 3"],
    forces_all: Float[Array, "N_conf N_atoms 3"],
    energies_all: Float[Array, "N_conf"],
    params_init: BondedParams,
    topology: BondedTopology,
    optimizer: optax.GradientTransformation,
    alpha: float = 0.25,
    w_reg: float = 0.01,
) -> tuple[TrainState, TrainMetrics]:
    """Single training step for one molecule: loss + grad + optimizer.update.

    Pure function; suitable as a scan body.

    Args:
        state: Current TrainState.
        conformer_idx: Index of conformer to train on (0-based within the molecule).
        positions_all: All conformers for this molecule.
        forces_all: Reference forces for all conformers.
        energies_all: Reference energies for all conformers.
        params_init: Initial (frozen) parameters for regularization.
        topology: Static bonded connectivity.
        optimizer: optax optimizer.
        alpha: Weight for energy loss (default 0.25).
        w_reg: Weight for regularization (default 0.01).

    Returns:
        (new_state, metrics)
    """
    # Extract single conformer
    positions = positions_all[conformer_idx]
    forces_ref = forces_all[conformer_idx]
    energy_ref = energies_all[conformer_idx]

    # Stack into batch dimension for loss function (expects [N_conf, ...])
    positions_batch = jnp.expand_dims(positions, axis=0)  # [1, N_atoms, 3]
    forces_batch = jnp.expand_dims(forces_ref, axis=0)    # [1, N_atoms, 3]
    energies_batch = jnp.expand_dims(energy_ref, axis=0)  # [1]

    def loss_fn(params):
        return bonded_loss(
            positions_batch,
            forces_batch,
            energies_batch,
            params,
            params_init,
            topology,
            alpha=alpha,
            w_reg=w_reg,
        )

    # Compute loss and gradients
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(state.params)

    # For metrics, we need to decompose the loss into components.
    # Since bonded_loss doesn't expose components, we recompute them here
    # (small overhead, acceptable for monitoring).
    sigma = default_sigma(params_init)

    def energy_fn(pos):
        from prolix.fitting.energy import bonded_energy
        return bonded_energy(pos, state.params, topology)

    forces_fn = jax.grad(energy_fn)
    forces_pred = forces_fn(positions)
    energies_pred = energy_fn(positions)

    # Force loss: per-atom MSE
    force_diff = forces_pred - forces_ref
    loss_force = jnp.mean(force_diff**2) / positions.shape[0]

    # Energy loss: per-conformer MSE (shifted to zero mean)
    shift = jnp.mean(energies_pred) - jnp.mean(energy_ref) * 627.5094740631
    energy_diff = energies_pred - (energy_ref * 627.5094740631 - shift)
    loss_energy = jnp.mean(energy_diff**2)

    # Regularization
    reg_bond = jnp.sum(((state.params.k_bond - params_init.k_bond) / sigma.k_bond) ** 2)
    reg_r0 = jnp.sum(((state.params.r0 - params_init.r0) / sigma.r0) ** 2)
    reg_theta = jnp.sum(((state.params.k_theta - params_init.k_theta) / sigma.k_theta) ** 2)
    reg_theta0 = jnp.sum(
        ((state.params.theta0_rad - params_init.theta0_rad) / sigma.theta0_rad) ** 2
    )
    reg_phi = jnp.sum(((state.params.k_phi - params_init.k_phi) / sigma.k_phi) ** 2)
    loss_reg = w_reg * (reg_bond + reg_r0 + reg_theta + reg_theta0 + reg_phi)

    # Update parameters via optimizer
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = eqx.apply_updates(state.params, updates)

    # Build new state
    new_state = eqx.tree_at(
        lambda s: (s.params, s.opt_state, s.step),
        state,
        (new_params, new_opt_state, state.step + 1),
    )

    # Note: state.step is a JAX array in scan context, so we keep it as-is for metrics
    # The metrics will be materialized outside the trace when io_callback is called
    metrics = TrainMetrics(
        step=state.step,
        loss=loss_val,
        loss_force=loss_force,
        loss_energy=loss_energy,
        loss_reg=loss_reg,
    )

    return new_state, metrics


def train_loop_one_mol(
    init_state: TrainState,
    n_steps: int,
    *,
    positions_all: Float[Array, "N_conf N_atoms 3"],
    forces_all: Float[Array, "N_conf N_atoms 3"],
    energies_all: Float[Array, "N_conf"],
    params_init: BondedParams,
    topology: BondedTopology,
    optimizer: optax.GradientTransformation,
    alpha: float = 0.25,
    w_reg: float = 0.01,
    io_callback_fn: Callable[[TrainMetrics], None] | None = None,
) -> tuple[TrainState, list]:
    """Training loop for one molecule via jax.lax.scan.

    Args:
        init_state: Initial TrainState.
        n_steps: Number of training steps.
        positions_all, forces_all, energies_all: Conformer data.
        params_init, topology: Bonded system.
        optimizer: optax optimizer.
        alpha, w_reg: Loss weights.
        io_callback_fn: Optional function(metrics) called via io_callback each step.

    Returns:
        (final_state, stacked_metrics)
        Note: stacked_metrics is a PyTree with fields stacked by jax.lax.scan.
              Use TrainMetrics unpacking to convert back to list of metrics.
    """
    n_conf = positions_all.shape[0]

    def step_fn(state, step_idx):
        # Sample a random conformer index for this step
        state, rng_key = state.split_rng()
        conf_idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=n_conf)

        # Train step
        new_state, metrics = train_step_one_mol(
            state,
            conf_idx,
            positions_all=positions_all,
            forces_all=forces_all,
            energies_all=energies_all,
            params_init=params_init,
            topology=topology,
            optimizer=optimizer,
            alpha=alpha,
            w_reg=w_reg,
        )

        # Log metrics via io_callback (non-blocking)
        if io_callback_fn is not None:
            jax.experimental.io_callback(
                io_callback_fn,
                None,
                metrics,
                ordered=False,
            )

        return new_state, metrics

    # Use jax.lax.scan with thin loop body (io_callback pattern from jaxbeans)
    final_state, stacked_metrics = jax.lax.scan(
        step_fn,
        init_state,
        jnp.arange(n_steps),
    )

    # stacked_metrics is a PyTree with all step values stacked.
    # Convert to list of TrainMetrics for convenience.
    metrics_list = [
        TrainMetrics(
            step=int(stacked_metrics.step[i]),
            loss=float(stacked_metrics.loss[i]),
            loss_force=float(stacked_metrics.loss_force[i]),
            loss_energy=float(stacked_metrics.loss_energy[i]),
            loss_reg=float(stacked_metrics.loss_reg[i]),
        )
        for i in range(n_steps)
    ]

    return final_state, metrics_list


def train_loop_looped_baseline(
    per_mol_states: list[TrainState],
    n_steps: int,
    *,
    per_mol_data: list[dict],
    params_init_list: list[BondedParams],
    topology_list: list[BondedTopology],
    optimizer: optax.GradientTransformation,
    alpha: float = 0.25,
    w_reg: float = 0.01,
    io_callback_fn: Callable[[TrainMetrics], None] | None = None,
) -> dict:
    """Sequential train_loop_one_mol per molecule (looped baseline).

    Wall-clock = sum of per-molecule times. Used for falsification (R6).

    Args:
        per_mol_states: List of TrainState (one per molecule).
        n_steps: Training steps per molecule.
        per_mol_data: List of dicts with 'positions_all', 'forces_all', 'energies_all'.
        params_init_list: List of BondedParams.
        topology_list: List of BondedTopology.
        optimizer: optax optimizer.
        alpha, w_reg: Loss weights.
        io_callback_fn: Optional logging callback.

    Returns:
        dict with keys:
            'final_states': list of final TrainState
            'final_losses': list of final per-mol loss values
            'all_metrics': list of list of metrics (per mol)
    """
    n_mol = len(per_mol_states)
    final_states = []
    final_losses = []
    all_metrics = []

    for mol_idx in range(n_mol):
        state = per_mol_states[mol_idx]
        data = per_mol_data[mol_idx]
        params_init = params_init_list[mol_idx]
        topology = topology_list[mol_idx]

        final_state, metrics_list = train_loop_one_mol(
            state,
            n_steps,
            positions_all=data["positions_all"],
            forces_all=data["forces_all"],
            energies_all=data["energies_all"],
            params_init=params_init,
            topology=topology,
            optimizer=optimizer,
            alpha=alpha,
            w_reg=w_reg,
            io_callback_fn=io_callback_fn,
        )

        final_states.append(final_state)
        final_losses.append(float(metrics_list[-1].loss))
        all_metrics.append(metrics_list)

    return {
        "final_states": final_states,
        "final_losses": final_losses,
        "all_metrics": all_metrics,
    }


def train_loop_batched(
    batched_state: TrainState,
    n_steps: int,
    *,
    batched_data: dict,
    batched_params_init: "BondedParamsBundle",
    batched_topology: "BondedTopologyBundle",
    optimizer: optax.GradientTransformation,
    alpha: float = 0.25,
    w_reg: float = 0.01,
    io_callback_fn: Callable[[int, list], None] | None = None,
) -> dict:
    """Batched training via jit(vmap(train_step)) over molecule axis.

    Implements single batched scan where each step vmap's over molecules.
    Positions/forces/energies are pre-padded and stacked by caller.
    Per-molecule masking gates which bonds/angles/torsions contribute to loss.

    Args:
        batched_state: TrainState with params/opt_state shape (B, ...) over molecules.
        n_steps: Training steps.
        batched_data: Dict with arrays stacked over B molecules:
                      - 'positions_all': (B, N_conf_max, N_atoms_padded, 3)
                      - 'forces_all': (B, N_conf_max, N_atoms_padded, 3)
                      - 'energies_all': (B, N_conf_max)
                      - 'n_real_conf': (B,) number of real conformers per mol
        batched_params_init: BondedParamsBundle (B, max_bonds, ...).
        batched_topology: BondedTopologyBundle with masks.
        optimizer: optax optimizer.
        alpha, w_reg: Loss weights.
        io_callback_fn: Optional function(step, per_mol_losses_list) for logging.

    Returns:
        dict with keys:
            'final_state': final batched TrainState (shape over B)
            'final_losses': [B] array of per-molecule final loss values
            'wallclock_s': wall-clock time (JAX block_until_ready)
    """

    B = batched_state.params.k_bond.shape[0]
    positions_all = batched_data["positions_all"]  # (B, N_conf_max, N_atoms_padded, 3)
    forces_all = batched_data["forces_all"]
    energies_all = batched_data["energies_all"]
    n_real_conf = batched_data["n_real_conf"]  # (B,) per-mol real conformer count

    def step_fn_one_mol(
        mol_state: TrainState,
        mol_rng_key,
        mol_positions_all,
        mol_forces_all,
        mol_energies_all,
        mol_n_real_conf,
        mol_params_init,
        mol_topology_bond_idx,
        mol_topology_angle_idx,
        mol_topology_torsion_idx,
        mol_topology_torsion_periodicity,
        mol_topology_torsion_phase_rad,
        mol_bond_mask,
        mol_angle_mask,
        mol_torsion_mask,
    ):
        """One training step for one molecule inside vmap."""
        # Sample a conformer index
        conf_idx = jax.random.randint(
            mol_rng_key, shape=(), minval=0, maxval=mol_n_real_conf
        )

        # Extract single conformer
        positions = mol_positions_all[conf_idx]  # (N_atoms_padded, 3)
        forces_ref = mol_forces_all[conf_idx]  # (N_atoms_padded, 3)
        energy_ref = mol_energies_all[conf_idx]  # scalar

        # Stack for loss function: bonded_loss expects (N_conf, N_atoms, 3).
        # Phase C uses 1-conformer-per-step SGD; same convention as the looped path.
        positions_batch = jnp.expand_dims(positions, axis=0)
        forces_batch = jnp.expand_dims(forces_ref, axis=0)
        energies_batch = jnp.expand_dims(energy_ref, axis=0)

        # Build topology for this molecule (unbatched) from per-mol slices.
        mol_topology = BondedTopology(
            bond_idx=mol_topology_bond_idx,
            angle_idx=mol_topology_angle_idx,
            torsion_idx=mol_topology_torsion_idx,
            torsion_periodicity=mol_topology_torsion_periodicity,
            torsion_phase_rad=mol_topology_torsion_phase_rad,
        )

        def loss_fn(params):
            # Single source of truth: bonded_loss with masks. The previous
            # inline implementation diverged from the looped path's loss
            # formula (wrong sign on energy shift) — verified bug
            # 2026-05-20, fixed by routing through bonded_loss.
            return bonded_loss(
                positions_batch,
                forces_batch,
                energies_batch,
                params,
                mol_params_init,
                mol_topology,
                alpha=alpha,
                w_reg=w_reg,
                bond_mask=mol_bond_mask,
                angle_mask=mol_angle_mask,
                torsion_mask=mol_torsion_mask,
            )

        # Compute loss and gradients
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(mol_state.params)

        # Update parameters via optimizer
        updates, new_opt_state = optimizer.update(grads, mol_state.opt_state, mol_state.params)
        new_params = eqx.apply_updates(mol_state.params, updates)

        # Build new state
        new_state = eqx.tree_at(
            lambda s: (s.params, s.opt_state, s.step),
            mol_state,
            (new_params, new_opt_state, mol_state.step + 1),
        )

        return new_state, loss_val

    # vmap step_fn_one_mol over molecules (axis 0)
    step_fn_batched = jax.vmap(
        step_fn_one_mol,
        in_axes=(
            0,  # mol_state (batch over params/opt_state/step/rng)
            0,  # mol_rng_key
            0,  # mol_positions_all
            0,  # mol_forces_all
            0,  # mol_energies_all
            0,  # mol_n_real_conf
            0,  # mol_params_init (B, max_bonds, ...)
            0,  # mol_topology_bond_idx
            0,  # mol_topology_angle_idx
            0,  # mol_topology_torsion_idx
            0,  # mol_topology_torsion_periodicity
            0,  # mol_topology_torsion_phase_rad
            0,  # mol_bond_mask (B, max_bonds)
            0,  # mol_angle_mask (B, max_angles)
            0,  # mol_torsion_mask (B, max_torsions)
        ),
    )

    def scan_body(state, step_idx):
        """Scan body: split RNG per molecule, call vmapped step, callback."""
        # state.rng has shape (B, 2) — one key per molecule
        # Split each molecule's RNG into two parts: new state key + step key
        def split_rng_fn(rng_key):
            key1, key2 = jax.random.split(rng_key)
            return key1, key2

        # vmap split_rng_fn over the B molecules
        new_rngs, mol_keys = jax.vmap(split_rng_fn)(state.rng)  # (B, 2), (B, 2)

        # Update state with new RNGs
        state = eqx.tree_at(lambda s: s.rng, state, new_rngs)

        # Call vmapped step (step_fn_batched already vmaps over B molecules)
        new_state, losses = step_fn_batched(
            state,
            mol_keys,
            positions_all,
            forces_all,
            energies_all,
            n_real_conf,
            batched_params_init,
            batched_topology.bond_idx,
            batched_topology.angle_idx,
            batched_topology.torsion_idx,
            batched_topology.torsion_periodicity,
            batched_topology.torsion_phase_rad,
            batched_topology.bond_mask,
            batched_topology.angle_mask,
            batched_topology.torsion_mask,
        )

        # Callback (non-blocking)
        if io_callback_fn is not None:
            losses_list = [float(losses[b]) for b in range(B)]
            jax.experimental.io_callback(
                io_callback_fn,
                None,
                int(step_idx),
                losses_list,
                ordered=False,
            )

        return new_state, losses

    # JIT compile the scan
    @eqx.filter_jit
    def run_scan():
        final_state, all_losses = jax.lax.scan(
            scan_body,
            batched_state,
            jnp.arange(n_steps),
            unroll=8,
        )
        return final_state, all_losses

    # Time the execution
    import time

    start_wall = time.time()
    final_state, all_losses = run_scan()
    # Force JAX to complete
    jax.effects_barrier()
    wallclock_s = time.time() - start_wall

    # Extract final losses per molecule
    final_losses = all_losses[-1]  # Last step's losses, shape (B,)

    return {
        "final_state": final_state,
        "final_losses": final_losses,
        "wallclock_s": wallclock_s,
    }
