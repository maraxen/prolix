"""Fitting configuration and plan for heterogeneous batched training.

Defines FittingConfig (hyperparameters), FittingPlan (optimizer + loss composition),
and TrainMetrics (step-wise metrics collection).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from prolix.fitting.loss import bonded_loss, default_sigma
from prolix.fitting.topology import BondedTopology
from prolix.fitting.bundles import TrainState

if TYPE_CHECKING:
    from prolix.fitting.bundles import BatchedFittingBundle


# ===== CONFIGURATION =====


@dataclasses.dataclass(frozen=True)
class FittingConfig:
    """Immutable configuration for bonded-parameter fitting.

    Attributes:
        lr: Learning rate for Adam optimizer.
        n_steps: Number of training steps.
        alpha: Weight for energy loss (default 0.25 from spec §6).
        w_reg: Weight for regularization (default 0.01 from spec §6).
        grad_clip_norm: Optional global gradient norm clipping (None = no clipping).
    """

    lr: float
    n_steps: int
    alpha: float = 0.25
    w_reg: float = 0.01
    grad_clip_norm: float | None = None


# ===== METRICS =====


class TrainMetrics(eqx.Module):
    """Metrics collected during a training step.

    Attributes:
        loss: Total loss (force + energy + regularization).
        energy_mse: Energy MSE component.
        force_mse: Force MSE component.
        reg: Regularization loss component.
        grad_norm: Global norm of gradients (for monitoring).
    """

    loss: Float[Array, ""]
    energy_mse: Float[Array, ""]
    force_mse: Float[Array, ""]
    reg: Float[Array, ""]
    grad_norm: Float[Array, ""]


# ===== OPTIMIZER FACTORY =====


def _build_optimizer(config: FittingConfig) -> optax.GradientTransformation:
    """Compose optimizer from config with optional gradient clipping.

    Args:
        config: FittingConfig with lr and optional grad_clip_norm.

    Returns:
        optax.GradientTransformation ready for eqx.apply_updates.
    """
    chain_pieces = []
    if config.grad_clip_norm is not None:
        chain_pieces.append(optax.clip_by_global_norm(config.grad_clip_norm))
    chain_pieces.append(optax.adam(learning_rate=config.lr))
    return optax.chain(*chain_pieces)


# ===== FITTING PLAN =====


@dataclasses.dataclass(frozen=True)
class FittingPlan:
    """Composable fitting orchestrator with internal JIT.

    Bundles optimizer, loss function, and hyperparameters. Implements
    .step() and .evaluate() as JIT-compiled methods that take a
    BatchedFittingBundle and TrainState.

    Static signature for JIT caching: (n_mols_real, max_atoms_bucket, n_steps, lr)
    All non-array Bundle fields must be eqx.field(static=True) and hashable.

    Attributes:
        optimizer: optax.GradientTransformation (Adam + optional clipping).
        loss_fn: Callable with signature (params, positions, forces, energies,
            topology, atom_mask, alpha, w_reg) -> (loss, aux).
        config: FittingConfig with hyperparameters.
    """

    optimizer: optax.GradientTransformation
    loss_fn: Callable
    config: FittingConfig

    @eqx.filter_jit
    def step(
        self,
        bundle: BatchedFittingBundle,
        state: TrainState,
        conformer_idx: int,
    ) -> tuple[TrainState, TrainMetrics]:
        """JIT-compiled single training step on batch.

        Args:
            bundle: BatchedFittingBundle (padded across B molecules).
            state: Current TrainState (params, opt_state, key, step_count).
            conformer_idx: Conformer index to use (0..max_n_conf-1).

        Returns:
            (new_state, metrics) tuple.
        """
        # RNG: fold_in per step, NOT pre-allocated key array (R6 compliance)
        new_key = jax.random.fold_in(state.key, state.step_count)

        # Select per-mol conformer slice (vmap over B mols)
        # conformer_idx is scalar; broadcast across batch
        # Extract as (B, n_atoms_max, 3) but bonded_loss expects (N_conf, N_atoms, 3)
        # so we wrap each mol's single conformer as (1, n_atoms_max, 3)
        positions_t = jnp.expand_dims(
            bundle.conformers_batched.positions[:, conformer_idx], axis=1
        )  # (B, 1, n_atoms_max, 3)
        forces_t = jnp.expand_dims(
            bundle.conformers_batched.forces_ref[:, conformer_idx], axis=1
        )  # (B, 1, n_atoms_max, 3)
        energies_t = jnp.expand_dims(
            bundle.conformers_batched.energies_ref[:, conformer_idx], axis=1
        )  # (B, 1)

        def loss_wrapped(params):
            """Compute loss for batch of molecules via vmap over per-mol losses.

            Vmaps loss computation over molecules, then masks padded entries and reduces.
            Matches train_loop_batched reduction semantics exactly.
            """
            def loss_per_mol(mol_params, mol_topology_bundle, pos_mol, forces_mol, energies_mol):
                """Per-molecule loss (vmapped over batch axis, step path).

                Each arg is a vmap'd slice with the leading B dim stripped:
                  mol_params: BondedParamsBundle pytree (per-mol leaf shapes)
                  mol_topology_bundle: BondedTopologyBundle pytree
                  pos_mol, forces_mol, energies_mol: single-conformer arrays
                """
                mol_topology = BondedTopology(
                    bond_idx=mol_topology_bundle.bond_idx,
                    angle_idx=mol_topology_bundle.angle_idx,
                    torsion_idx=mol_topology_bundle.torsion_idx,
                    torsion_periodicity=mol_topology_bundle.torsion_periodicity,
                    torsion_phase_rad=mol_topology_bundle.torsion_phase_rad,
                )
                return self.loss_fn(
                    pos_mol, forces_mol, energies_mol,
                    mol_params, mol_params,
                    mol_topology,
                    alpha=self.config.alpha, w_reg=self.config.w_reg,
                    bond_mask=mol_topology_bundle.bond_mask,
                    angle_mask=mol_topology_bundle.angle_mask,
                    torsion_mask=mol_topology_bundle.torsion_mask,
                )

            # Vmap over molecules: bundles passed as pytrees, in_axes=0 slices each leaf
            per_mol_losses = jax.vmap(
                loss_per_mol,
                in_axes=(0, 0, 0, 0, 0),
            )(
                params,                       # BondedParamsBundle (B-leading)
                bundle.topology_batched,      # BondedTopologyBundle (B-leading)
                positions_t,
                forces_t,
                energies_t,
            )  # per_mol_losses: (B,)

            # Mask padded molecules and reduce
            mol_mask = jnp.arange(per_mol_losses.shape[0]) < bundle.n_mols_real
            masked_losses = per_mol_losses * mol_mask
            loss = jnp.sum(masked_losses) / jnp.maximum(jnp.sum(mol_mask), 1.0)

            aux = {}
            return loss, aux

        # Compute loss and gradients
        (loss, aux), grads = eqx.filter_value_and_grad(loss_wrapped, has_aux=True)(state.params)

        # Apply optimizer
        updates, new_opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        new_params = eqx.apply_updates(state.params, updates)

        # Compute grad norm for metrics
        grad_norm = optax.global_norm(grads)

        # Build new state
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            key=new_key,
            step_count=state.step_count + 1,
        )

        # Build metrics
        metrics = TrainMetrics(
            loss=loss,
            energy_mse=jnp.zeros((), dtype=jnp.float32),
            force_mse=jnp.zeros((), dtype=jnp.float32),
            reg=jnp.zeros((), dtype=jnp.float32),
            grad_norm=grad_norm,
        )

        return new_state, metrics

    @eqx.filter_jit
    def evaluate(
        self,
        bundle: BatchedFittingBundle,
        state: TrainState,
    ) -> TrainMetrics:
        """JIT-compiled evaluation over batch (no parameter updates).

        Args:
            bundle: BatchedFittingBundle (padded across B molecules).
            state: Current TrainState (params, opt_state, key, step_count).

        Returns:
            metrics (TrainMetrics, read-only snapshot).
        """
        # Evaluate loss for first conformer of each molecule
        # Extract and wrap as (B, 1, n_atoms_max, 3) for bonded_loss
        positions_t = jnp.expand_dims(
            bundle.conformers_batched.positions[:, 0], axis=1
        )  # (B, 1, n_atoms_max, 3)
        forces_t = jnp.expand_dims(
            bundle.conformers_batched.forces_ref[:, 0], axis=1
        )  # (B, 1, n_atoms_max, 3)
        energies_t = jnp.expand_dims(
            bundle.conformers_batched.energies_ref[:, 0], axis=1
        )  # (B, 1)

        def loss_per_mol(mol_params, mol_topology_bundle, pos_mol, forces_mol, energies_mol):
            """Per-molecule loss (vmapped over batch axis, evaluate path)."""
            mol_topology = BondedTopology(
                bond_idx=mol_topology_bundle.bond_idx,
                angle_idx=mol_topology_bundle.angle_idx,
                torsion_idx=mol_topology_bundle.torsion_idx,
                torsion_periodicity=mol_topology_bundle.torsion_periodicity,
                torsion_phase_rad=mol_topology_bundle.torsion_phase_rad,
            )
            return self.loss_fn(
                pos_mol, forces_mol, energies_mol,
                mol_params, mol_params,
                mol_topology,
                alpha=self.config.alpha, w_reg=self.config.w_reg,
                bond_mask=mol_topology_bundle.bond_mask,
                angle_mask=mol_topology_bundle.angle_mask,
                torsion_mask=mol_topology_bundle.torsion_mask,
            )

        per_mol_losses = jax.vmap(
            loss_per_mol,
            in_axes=(0, 0, 0, 0, 0),
        )(
            state.params,
            bundle.topology_batched,
            positions_t,
            forces_t,
            energies_t,
        )  # per_mol_losses: (B,)

        # Mask padded molecules and reduce
        mol_mask = jnp.arange(per_mol_losses.shape[0]) < bundle.n_mols_real
        masked_losses = per_mol_losses * mol_mask
        loss = jnp.sum(masked_losses) / jnp.maximum(jnp.sum(mol_mask), 1.0)

        # Return metrics (no gradients computed)
        metrics = TrainMetrics(
            loss=loss,
            energy_mse=jnp.zeros((), dtype=jnp.float32),
            force_mse=jnp.zeros((), dtype=jnp.float32),
            reg=jnp.zeros((), dtype=jnp.float32),
            grad_norm=jnp.zeros((), dtype=jnp.float32),
        )

        return metrics


# ===== FACTORY =====


def make_fitting_plan(
    config: FittingConfig,
    loss_fn: Callable | None = None,
) -> FittingPlan:
    """Compose FittingPlan from config with optional custom loss function.

    Builds optimizer from config (Adam with optional clipping).
    Uses bonded_loss as default loss function if not provided.

    Args:
        config: FittingConfig with lr, n_steps, and loss hyperparameters.
        loss_fn: Optional custom loss function. If None, uses bonded_loss.

    Returns:
        FittingPlan ready to drive training via .step() / .evaluate().
    """
    if loss_fn is None:
        loss_fn = bonded_loss

    optimizer = _build_optimizer(config)

    return FittingPlan(
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
    )
