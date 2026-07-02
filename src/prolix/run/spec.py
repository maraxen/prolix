"""Host-side batch planning and fitting configuration.

This module bridges user-level BatchingConfig (input knobs) with device-side
BatchPlan (vmap vs safe_map dispatch decisions). All logic is host-side pure Python
over AxisSpec dataclasses, with no JAX runtime operations.
"""

from __future__ import annotations

import dataclasses

import jax

from prolix.tiling.axes import N_CONFORMERS, N_MOLS
from prolix.tiling.planner import BatchPlan, BatchPlanner, estimate_memory_theoretical


@dataclasses.dataclass(frozen=True)
class BatchingConfig:
    """Host-side knobs for fitting batch layout.

    Each field is consumed by make_fitting_planner to construct AxisSpec
    cardinalities. The planner then decides vmap vs safe_map per axis
    based on memory budget.
    """

    n_mols: int = 1  # cardinality of N_MOLS axis (total mols in dataset)
    n_conformers: int = 1  # cardinality of N_CONFORMERS axis (max conf per mol)
    mols_batch_size: int | None = None  # explicit override; None → planner decides
    conformers_batch_size: int | None = None  # explicit override; None → planner decides


class FittingAxisNames:
    """Named string constants for axis lookups in BatchPlan.decision_for(...)."""

    N_ATOMS = "n_atoms"
    N_BONDS = "n_bonds"
    N_ANGLES = "n_angles"
    N_TORSIONS = "n_torsions"
    N_CONFORMERS = "n_conformers"
    N_MOLS = "n_mols"


def make_fitting_planner(
    spec: BatchingConfig,
    param_bytes: float = 0.0,
    headroom: float = 0.80,
    activation_multiplier: float = 2.5,
) -> BatchPlan:
    """Build a BatchPlan for fitting dispatch.

    Args:
        spec: BatchingConfig with cardinalities and optional overrides.
        param_bytes: Estimated parameter footprint in bytes.
        headroom: Fraction of device memory to budget (0.80 = 80%).
        activation_multiplier: Multiplier for activation memory estimation.

    Returns:
        BatchPlan describing batch size decisions for N_MOLS and N_CONFORMERS.
    """
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is None:
            raise AttributeError("memory_stats returned None")
        device_limit = stats["bytes_limit"]
    except (KeyError, IndexError, AttributeError, TypeError):
        # Fallback for CPU/test environments where memory_stats unavailable
        device_limit = 4 * 1024**3  # 4 GB

    budget = device_limit * headroom - param_bytes

    # Build axes with cardinalities overridden from spec
    axes = [
        dataclasses.replace(
            N_MOLS,
            cardinality=max(1, spec.n_mols),
            # If explicit batch size given, use it; else planner decides
            default_batch_size=(
                spec.mols_batch_size
                if spec.mols_batch_size is not None
                else N_MOLS.default_batch_size
            ),
        ),
        dataclasses.replace(
            N_CONFORMERS,
            cardinality=max(1, spec.n_conformers),
            default_batch_size=(
                spec.conformers_batch_size
                if spec.conformers_batch_size is not None
                else N_CONFORMERS.default_batch_size
            ),
        ),
    ]

    return BatchPlanner(
        axes=axes,
        budget_bytes=budget,
        estimate_memory=lambda ds: estimate_memory_theoretical(
            ds, 1.0, activation_multiplier
        ),
    ).plan_with_xtrax()  # xtrax.tiling.BatchPlanner is the production strategy engine (#1842)


def extract_batch_sizes(plan: BatchPlan) -> tuple[int, int]:
    """Extract (mols_bs, conformers_bs) from a BatchPlan.

    Args:
        plan: BatchPlan from make_fitting_planner.

    Returns:
        Tuple of (mols_batch_size, conformers_batch_size).
    """
    mols_bs = plan.decision_for(FittingAxisNames.N_MOLS).batch_size
    conformers_bs = plan.decision_for(FittingAxisNames.N_CONFORMERS).batch_size
    return mols_bs, conformers_bs
