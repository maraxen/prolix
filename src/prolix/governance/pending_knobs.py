"""Governance allowlists for documented inert knobs and pending Epic work items.

``DOCUMENTED_INERT_KNOBS`` contains fields that are legitimately informational,
descriptive, host-I/O metadata, or retained strictly for backward-compatible API
stability (such as ``NeighborListConfig.safety_factor``, ``max_exclusions``, and
``on_overflow``).

``PENDING_EPIC_KNOBS`` is the monotonically shrinking allowlist of control knobs
being wired into downstream consumers across backlog items ``#6186``–``#6193``.
Each work item removes its owned knobs as it lands; Tier 3 item ``#6194`` asserts
``PENDING_EPIC_KNOBS == frozenset()``.
"""

from __future__ import annotations

DOCUMENTED_INERT_KNOBS: frozenset[str] = frozenset(
    {
        # NeighborListConfig documented API-stability inert knobs
        "safety_factor",
        "max_exclusions",
        "on_overflow",
        # Descriptive / informational / host-side I/O metadata fields
        "name",
        "description",
        "notes",
        "schema_version",
        "input_file",
        "output_dir",
        "save_path",
        "accumulate_steps",
        "platform",
        "precision",
        "device_index",
        "trajectory_format",
        "log_level",
        "verbose",
        "skip_minimization",
        "skip_equilibration_phases",
        "input_stage",
        "target_stage",
        "step_configs",
        "checkpoint",
        "CheckpointConfig.enabled",
        "checkpoint_interval_ps",
        "checkpoint_dir",
        "keep_last_n",
        "simulation_format",
        "small_molecule_ff",
        "protein_ff",
        "water_ff",
        "fix_missing_atoms",
        "cosolv_metadata_path",
    }
)

PENDING_EPIC_KNOBS_BY_ITEM: dict[str, frozenset[str]] = {
    "#6186": frozenset(),
    "#6187": frozenset(),
    "#6188": frozenset(),
    "#6189": frozenset(),
    "#6190": frozenset(),
    "#6191": frozenset(),
    "#6192": frozenset(),
    "#6193": frozenset(),
}

PENDING_EPIC_KNOBS: frozenset[str] = frozenset().union(
    *PENDING_EPIC_KNOBS_BY_ITEM.values()
)

__all__ = [
    "DOCUMENTED_INERT_KNOBS",
    "PENDING_EPIC_KNOBS",
    "PENDING_EPIC_KNOBS_BY_ITEM",
]
