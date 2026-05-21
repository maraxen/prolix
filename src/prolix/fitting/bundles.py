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

        Calls existing stack_molecules() internally to batch parameters and topology.
        Pads conformer data across molecules to maximum sizes.

        Args:
            bundles: List of N FittingBundle (one per molecule).

        Returns:
            BatchedFittingBundle with B=N and all shapes padded appropriately.

        Raises:
            ValueError: If bundles list is empty.
        """
        from prolix.fitting.batched import stack_molecules

        if not bundles:
            raise ValueError("Cannot stack zero bundles")

        B = len(bundles)

        # Extract per-mol params and topology
        params_list = [b.params for b in bundles]
        topology_list = [b.topology for b in bundles]
        params_batched, topology_batched = stack_molecules(params_list, topology_list)

        # Pad conformer data across mols (max N_conf, max n_atoms)
        max_n_conf = max(b.conformers.n_conf for b in bundles)
        max_n_atoms = max(b.conformers.n_atoms for b in bundles)

        # Initialize padded arrays
        positions_padded = jnp.zeros((B, max_n_conf, max_n_atoms, 3), dtype=jnp.float32)
        forces_padded = jnp.zeros((B, max_n_conf, max_n_atoms, 3), dtype=jnp.float32)
        energies_padded = jnp.zeros((B, max_n_conf), dtype=jnp.float32)
        atom_mask_padded = jnp.zeros((B, max_n_atoms), dtype=jnp.bool_)
        box_padded = jnp.zeros((B, 3, 3), dtype=jnp.float32)
        n_conf_real = jnp.zeros((B,), dtype=jnp.int32)
        n_atoms_real = jnp.zeros((B,), dtype=jnp.int32)

        # Fill in per-bundle data
        for b_idx, bundle in enumerate(bundles):
            c = bundle.conformers
            n_c = c.n_conf
            n_a = c.n_atoms

            positions_padded = positions_padded.at[b_idx, :n_c, :n_a].set(c.positions)
            forces_padded = forces_padded.at[b_idx, :n_c, :n_a].set(c.forces_ref)
            energies_padded = energies_padded.at[b_idx, :n_c].set(c.energies_ref)
            atom_mask_padded = atom_mask_padded.at[b_idx, :n_a].set(c.atom_mask)
            box_padded = box_padded.at[b_idx].set(bundle.box)
            n_conf_real = n_conf_real.at[b_idx].set(n_c)
            n_atoms_real = n_atoms_real.at[b_idx].set(n_a)

        # Build BatchedConformerBundle
        conformers_batched = BatchedConformerBundle(
            positions=positions_padded,
            forces_ref=forces_padded,
            energies_ref=energies_padded,
            atom_mask=atom_mask_padded,
            n_conf_real=n_conf_real,
            n_atoms_real=n_atoms_real,
            n_mols=B,
            max_n_atoms=max_n_atoms,
            max_n_conf=max_n_conf,
        )

        return BatchedFittingBundle(
            conformers_batched=conformers_batched,
            params_batched=params_batched,
            topology_batched=topology_batched,
            box_batched=box_padded,
            n_mols_real=B,
        )

    def step(self, *args, **kwargs):
        """BatchedFittingBundle.step is not directly callable.

        Use FittingPlan.step(bundle, state, conformer_idx) instead.
        """
        raise RuntimeError(
            "BatchedFittingBundle.step is not directly callable. "
            "Use FittingPlan.step(bundle, state, conformer_idx) instead."
        )

    def evaluate(self, *args, **kwargs):
        """BatchedFittingBundle.evaluate is not directly callable.

        Use FittingPlan.evaluate(bundle, state) instead.
        """
        raise RuntimeError(
            "BatchedFittingBundle.evaluate is not directly callable. "
            "Use FittingPlan.evaluate(bundle, state) instead."
        )


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
