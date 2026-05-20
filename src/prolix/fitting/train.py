"""Training loops for bonded-parameter fitting.

Implements single-molecule training step, scan-based training loop,
and both looped and batched orchestration strategies.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

from prolix.fitting.loss import bonded_loss, default_sigma
from prolix.fitting.params import BondedParams
from prolix.fitting.scheduler import sample_one_conformer
from prolix.fitting.state import TrainState
from prolix.fitting.topology import BondedTopology


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
) -> Tuple[TrainState, TrainMetrics]:
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
    io_callback_fn: Optional[Callable[[TrainMetrics], None]] = None,
) -> Tuple[TrainState, list]:
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
    io_callback_fn: Optional[Callable[[TrainMetrics], None]] = None,
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
    stacked_state: TrainState,
    n_steps: int,
    *,
    stacked_data: dict,
    params_init_list: list[BondedParams],
    topology_list: list[BondedTopology],
    optimizer: optax.GradientTransformation,
    alpha: float = 0.25,
    w_reg: float = 0.01,
    io_callback_fn: Optional[Callable[[TrainMetrics], None]] = None,
) -> dict:
    """Batched training via jit(vmap(train_loop_one_mol)) over molecule axis.

    Claim 1 substrate path: single batched scan.

    Args:
        stacked_state: TrainState with params/opt_state stacked over molecules.
        n_steps: Training steps.
        stacked_data: Dict with 'positions_all', 'forces_all', 'energies_all'
                      stacked over molecules (B molecules).
        params_init_list: List of BondedParams (one per molecule, for regularization).
        topology_list: List of BondedTopology.
        optimizer: optax optimizer.
        alpha, w_reg: Loss weights.
        io_callback_fn: Optional logging callback.

    Returns:
        dict with keys:
            'final_state': final stacked TrainState
            'final_losses': [B] array of per-mol final losses
            'all_metrics': list of metrics across all steps and molecules
    """
    # For now, fall back to looped baseline internally
    # (vmap of train_loop_one_mol is complex due to different array shapes per molecule)
    # This will be optimized in v1.1 with padded batching.
    raise NotImplementedError(
        "Batched training loop requires padded-array infrastructure (v1.1). "
        "Use looped_baseline for v0."
    )
