"""Bundle types for JAX-based fitting and heterogeneous training.

This module defines the host-prepped, JIT-ready bundle types for bonded-parameter
fitting. Logic (builder, training, evaluation) lives in bundle_builder.py (Phase 4)
and FittingPlan (Phase 5).

Type annotations use jaxtyping (Float, Int, Bool, Array, PRNGKeyArray) without
runtime @jaxtyped decorators (type-annotation only, matching tev_design pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from prolix.fitting.params import BondedParams
from prolix.fitting.topology import BondedTopology

if TYPE_CHECKING:
    from prolix.fitting.batched import BondedParamsBundle, BondedTopologyBundle


class ConformerBundle(eqx.Module):
    """Per-molecule conformer data (no batch axis).

    Wraps multiple 3D structures (conformations) of a single molecule, along with
    reference forces and energies for fitting. Used as building block for
    FittingBundle.

    Attributes:
        positions: Float[Array, "N_conf n_atoms 3"]
            Atomic coordinates for N_conf conformers of the same molecule.
        forces_ref: Float[Array, "N_conf n_atoms 3"]
            Reference forces (e.g., from QM) to fit against.
        energies_ref: Float[Array, "N_conf"]
            Reference energies for each conformer.
        atom_mask: Bool[Array, "n_atoms"]
            Binary mask indicating which atoms are real (True) vs padding (False).
        n_conf: int
            Number of conformers (static field, does not participate in pytree leaves).
        n_atoms: int
            Number of atoms per conformer (static field).
    """

    positions: Float[Array, "N_conf n_atoms 3"]
    forces_ref: Float[Array, "N_conf n_atoms 3"]
    energies_ref: Float[Array, "N_conf"]
    atom_mask: Bool[Array, "n_atoms"]
    n_conf: int = eqx.field(static=True)
    n_atoms: int = eqx.field(static=True)


class FittingBundle(eqx.Module):
    """Per-molecule top-level bundle (no batch axis).

    Wraps all inputs needed to fit bonded parameters for a single molecule:
    conformer data, trainable parameters, static topology, and periodic box.

    Attributes:
        conformers: ConformerBundle
            Per-mol conformer data.
        params: BondedParams
            Trainable bonded parameters (bonds, angles, torsions).
        topology: BondedTopology
            Static connectivity (atom indices, periodicity). Marked static=True
            since BondedTopology contains numpy arrays (not JAX arrays).
        box: Float[Array, "3 3"]
            Simulation box matrix. For vacuum systems, use jnp.zeros((3, 3)) as sentinel.
    """

    conformers: ConformerBundle
    params: BondedParams
    topology: BondedTopology = eqx.field(static=True)
    box: Float[Array, "3 3"]


class BatchedConformerBundle(eqx.Module):
    """Batched conformer data (padded across molecules).

    Wraps conformers for B molecules in a single batch, with padding to max sizes.
    Parallel structure to BondedParamsBundle from batched.py.

    Attributes:
        positions: Float[Array, "B N_conf n_atoms 3"]
            Padded conformer positions.
        forces_ref: Float[Array, "B N_conf n_atoms 3"]
            Padded reference forces.
        energies_ref: Float[Array, "B N_conf"]
            Reference energies (no padding in N_conf axis).
        atom_mask: Bool[Array, "B n_atoms"]
            Per-mol atom mask (padded across B).
        n_conf_real: Int[Array, "B"]
            Real number of conformers per molecule (≤ max_n_conf).
        n_atoms_real: Int[Array, "B"]
            Real number of atoms per molecule (≤ max_n_atoms).
        n_mols: int
            Batch size (static field).
        max_n_atoms: int
            Maximum atoms across batch (static field).
        max_n_conf: int
            Maximum conformers across batch (static field).
    """

    positions: Float[Array, "B max_n_conf max_n_atoms 3"]
    forces_ref: Float[Array, "B max_n_conf max_n_atoms 3"]
    energies_ref: Float[Array, "B max_n_conf"]
    atom_mask: Bool[Array, "B max_n_atoms"]
    n_conf_real: Int[Array, "B"]
    n_atoms_real: Int[Array, "B"]
    n_mols: int = eqx.field(static=True)
    max_n_atoms: int = eqx.field(static=True)
    max_n_conf: int = eqx.field(static=True)


class BatchedFittingBundle(eqx.Module):
    """Top-level batched bundle for heterogeneous fitting.

    Produced by stacking N FittingBundle instances via .stack() constructor.
    Wraps batched conformers, parameters, topology, and box matrices.
    This is the type FittingPlan.step() and .evaluate() consume.

    Attributes:
        conformers_batched: BatchedConformerBundle
            Padded conformers across B molecules.
        params_batched: BondedParamsBundle
            Padded parameters from prolix.fitting.batched.
        topology_batched: BondedTopologyBundle
            Padded topology indices from prolix.fitting.batched.
        box_batched: Float[Array, "B 3 3"]
            Box matrices for each molecule in batch.
        n_mols_real: int
            Real batch size (≤ padded B; static field).
    """

    conformers_batched: BatchedConformerBundle
    params_batched: "BondedParamsBundle"
    topology_batched: "BondedTopologyBundle"
    box_batched: Float[Array, "B 3 3"]
    n_mols_real: int = eqx.field(static=True)

    @staticmethod
    def stack(bundles: list[FittingBundle]) -> "BatchedFittingBundle":
        """Typed constructor: stack N FittingBundle instances into batched form.

        Calls existing stack_molecules() internally (Phase 5). This method
        declaration is type-only in Phase 3; implementation lands in Phase 5.

        Args:
            bundles: List of N FittingBundle (one per molecule).

        Returns:
            BatchedFittingBundle with B=N and all shapes padded appropriately.

        Raises:
            NotImplementedError: Full implementation in Phase 5.
        """
        raise NotImplementedError("BatchedFittingBundle.stack lands in Phase 5")

    def step(self, state: "TrainState", conformer_idx: int):
        """JIT-compiled training step for this batch.

        This method is type-only in Phase 3. Full implementation (delegating to
        train_loop_batched internals) lands in Phase 5.

        Args:
            state: TrainState bundling params, opt_state, key, step_count.
            conformer_idx: Index into the conformer axis (0..max_n_conf-1).

        Returns:
            (updated_state, metrics)

        Raises:
            NotImplementedError: Full implementation in Phase 5.
        """
        raise NotImplementedError("BatchedFittingBundle.step lands in Phase 5")

    def evaluate(self, state: "TrainState"):
        """JIT-compiled evaluation over the batch.

        This method is type-only in Phase 3. Full implementation (computing loss,
        metrics without parameter updates) lands in Phase 5.

        Args:
            state: TrainState bundling params, opt_state, key, step_count.

        Returns:
            metrics (TrainMetrics or equivalent)

        Raises:
            NotImplementedError: Full implementation in Phase 5.
        """
        raise NotImplementedError("BatchedFittingBundle.evaluate lands in Phase 5")


class TrainState(eqx.Module):
    """Training state for heterogeneous batched fitting.

    Bundles all state carried across .step() calls: trainable parameters,
    optimizer state, RNG key, and step counter. Follows eqx.Module pattern
    for seamless pytree round-tripping and JIT compilation.

    Attributes:
        params: BondedParams
            Current trainable bonded parameters.
        opt_state: optax.OptState
            Optimizer state (e.g., Adam momentum buffers).
        key: PRNGKeyArray
            JAX PRNG key for stochastic sampling (e.g., dropout, augmentation).
        step_count: Int[Array, ""]
            Global step counter (scalar integer array for JIT safety).
    """

    params: BondedParams
    opt_state: optax.OptState
    key: PRNGKeyArray
    step_count: Int[Array, ""]
