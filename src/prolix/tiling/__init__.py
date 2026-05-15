"""Tiling layer: AxisSpec, BatchPlan, bucketing utilities.

Vendored from prxteinmpnn at SHA 5a28f3e9.
No prolix-specific logic in planner.py or buckets.py.
Prolix axes (N_ATOMS, N_SYSTEMS) are in axes.py.
"""
from prolix.tiling.planner import AxisSpec, AxisDecision, BatchPlan, BatchPlanner
from prolix.tiling.buckets import get_length_bucket, pad_to_bucket

__all__ = [
    "AxisSpec",
    "AxisDecision",
    "BatchPlan",
    "BatchPlanner",
    "get_length_bucket",
    "pad_to_bucket",
]
