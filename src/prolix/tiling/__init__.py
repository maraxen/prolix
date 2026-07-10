"""Tiling layer: AxisSpec, BatchPlan, bucketing utilities.

Thin facade over xtrax joint MemoryBudget planning (XR-KILL-FORK / XR-CARRY).
Prolix axes (N_ATOMS … N_MOLS, N_STEPS, N_SYSTEMS) are in axes.py.
"""
from prolix.tiling.axes import (
    ALL_AXES,
    N_ANGLES,
    N_ATOMS,
    N_BONDS,
    N_CONFORMERS,
    N_MOLS,
    N_STEPS,
    N_SYSTEMS,
    N_TORSIONS,
)
from prolix.tiling.buckets import get_length_bucket, pad_to_bucket
from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan, BatchPlanner

__all__ = [
    "AxisSpec",
    "AxisDecision",
    "BatchPlan",
    "BatchPlanner",
    "get_length_bucket",
    "pad_to_bucket",
    "N_ATOMS",
    "N_BONDS",
    "N_ANGLES",
    "N_TORSIONS",
    "N_CONFORMERS",
    "N_MOLS",
    "N_STEPS",
    "N_SYSTEMS",
    "ALL_AXES",
]
