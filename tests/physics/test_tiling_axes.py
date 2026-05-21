import pytest
from prolix.tiling.planner import AxisSpec, BatchPlanner
from prolix.tiling.axes import (
    N_ATOMS, N_BONDS, N_ANGLES, N_TORSIONS, N_CONFORMERS, N_MOLS, N_SYSTEMS, ALL_AXES
)


def test_n_atoms_axis_spec():
    assert N_ATOMS.name == "n_atoms"
    assert N_ATOMS.axis_index == 0
    assert N_ATOMS.cardinality == 60_000
    assert N_ATOMS.heterogeneous is False
    assert N_ATOMS.tile_granularity == 128
    assert N_ATOMS.default_batch_size == 0  # 0 means vmap


def test_n_bonds_axis_spec():
    assert N_BONDS.name == "n_bonds"
    assert N_BONDS.axis_index == 1
    assert N_BONDS.cardinality == 512
    assert N_BONDS.heterogeneous is False
    assert N_BONDS.tile_granularity == 64
    assert N_BONDS.default_batch_size == 0


def test_n_angles_axis_spec():
    assert N_ANGLES.name == "n_angles"
    assert N_ANGLES.axis_index == 2
    assert N_ANGLES.cardinality == 512
    assert N_ANGLES.heterogeneous is False
    assert N_ANGLES.tile_granularity == 64
    assert N_ANGLES.default_batch_size == 0


def test_n_torsions_axis_spec():
    assert N_TORSIONS.name == "n_torsions"
    assert N_TORSIONS.axis_index == 3
    assert N_TORSIONS.cardinality == 512
    assert N_TORSIONS.heterogeneous is False
    assert N_TORSIONS.tile_granularity == 64
    assert N_TORSIONS.default_batch_size == 0


def test_n_conformers_axis_spec():
    assert N_CONFORMERS.name == "n_conformers"
    assert N_CONFORMERS.axis_index == 4
    assert N_CONFORMERS.cardinality == 2048
    assert N_CONFORMERS.heterogeneous is True
    assert N_CONFORMERS.tile_granularity == 1
    assert N_CONFORMERS.default_batch_size == 1


def test_n_mols_axis_spec():
    assert N_MOLS.name == "n_mols"
    assert N_MOLS.axis_index == 5
    assert N_MOLS.cardinality == 64
    assert N_MOLS.heterogeneous is True
    assert N_MOLS.default_batch_size == 1  # safe_map
    assert N_MOLS.tile_granularity == 1


def test_n_systems_alias_preserved():
    """N_SYSTEMS alias should resolve to N_MOLS for backward compatibility."""
    assert N_SYSTEMS is N_MOLS


def test_axes_are_hashable():
    d = {N_ATOMS: "innermost", N_MOLS: "outermost"}
    assert d[N_ATOMS] == "innermost"
    assert d[N_MOLS] == "outermost"


def test_all_axes_registered():
    """All axes should be in ALL_AXES registry."""
    assert len(ALL_AXES) == 6
    assert N_ATOMS in ALL_AXES
    assert N_BONDS in ALL_AXES
    assert N_ANGLES in ALL_AXES
    assert N_TORSIONS in ALL_AXES
    assert N_CONFORMERS in ALL_AXES
    assert N_MOLS in ALL_AXES


def test_all_axes_contiguous_indices():
    """Axis indices should be contiguous [0..5]."""
    indices = [ax.axis_index for ax in ALL_AXES]
    assert sorted(indices) == list(range(6))


def test_batch_planner_demotes_heterogeneous_first():
    """Heterogeneous axes should always use safe_map."""
    planner = BatchPlanner(
        axes=[N_ATOMS, N_MOLS],
        budget_bytes=2 * 1024**3,
        estimate_memory=lambda decisions: 1e6  # Mock: always within budget
    )
    plan = planner.plan()
    # N_MOLS is heterogeneous so always safe_map
    mol_decision = next(d for d in plan.decisions if d.axis.name == "n_mols")
    assert mol_decision.batch_size > 0  # safe_map, not vmap (batch_size=0)
