import pytest
from prolix.tiling.planner import AxisSpec, BatchPlanner
from prolix.tiling.axes import N_ATOMS, N_SYSTEMS


def test_n_atoms_axis_spec():
    assert N_ATOMS.name == "n_atoms"
    assert N_ATOMS.cardinality == 60_000
    assert N_ATOMS.heterogeneous is False
    assert N_ATOMS.tile_granularity == 128
    assert N_ATOMS.default_batch_size == 0  # 0 means vmap


def test_n_systems_axis_spec():
    assert N_SYSTEMS.name == "n_systems"
    assert N_SYSTEMS.cardinality == 64
    assert N_SYSTEMS.heterogeneous is True
    assert N_SYSTEMS.default_batch_size == 1  # safe_map
    assert N_SYSTEMS.tile_granularity == 1


def test_axes_are_hashable():
    d = {N_ATOMS: "vmap", N_SYSTEMS: "safe_map"}
    assert d[N_ATOMS] == "vmap"
    assert d[N_SYSTEMS] == "safe_map"


def test_batch_planner_demotes_heterogeneous_first():
    """Heterogeneous axes should always use safe_map."""
    planner = BatchPlanner(
        axes=[N_ATOMS, N_SYSTEMS],
        budget_bytes=2 * 1024**3,
        estimate_memory=lambda decisions: 1e6  # Mock: always within budget
    )
    plan = planner.plan()
    # N_SYSTEMS is heterogeneous so always safe_map
    sys_decision = next(d for d in plan.decisions if d.axis.name == "n_systems")
    assert sys_decision.batch_size > 0  # safe_map, not vmap (batch_size=0)
