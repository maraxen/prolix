"""Prolix-specific AxisSpec instances for molecular simulation axes."""
from prolix.tiling.planner import AxisSpec

# Innermost axes (homogeneous, low tile granularity)
N_ATOMS = AxisSpec(
    name="n_atoms",
    axis_index=0,
    cardinality=60_000,
    default_batch_size=0,     # 0 = vmap when homogeneous
    tile_granularity=128,     # Pallas kernel alignment
    heterogeneous=False,
    doc="Atom count axis (homogeneous trajectory batching)",
)

N_BONDS = AxisSpec(
    name="n_bonds",
    axis_index=1,
    cardinality=512,
    default_batch_size=0,     # 0 = vmap when homogeneous
    tile_granularity=64,      # Bonded term granularity
    heterogeneous=False,
    doc="Bond term dimension within a single molecule (fixed after topology bucketing).",
)

N_ANGLES = AxisSpec(
    name="n_angles",
    axis_index=2,
    cardinality=512,
    default_batch_size=0,     # 0 = vmap when homogeneous
    tile_granularity=64,      # Angle term granularity
    heterogeneous=False,
    doc="Angle term dimension within a single molecule.",
)

N_TORSIONS = AxisSpec(
    name="n_torsions",
    axis_index=3,
    cardinality=512,
    default_batch_size=0,     # 0 = vmap when homogeneous
    tile_granularity=64,      # Torsion term granularity
    heterogeneous=False,
    doc="Torsion term dimension within a single molecule.",
)

# Middle axis (heterogeneous conformer scheduling)
N_CONFORMERS = AxisSpec(
    name="n_conformers",
    axis_index=4,
    cardinality=2048,
    default_batch_size=1,     # safe_map: varies per molecule
    tile_granularity=1,       # No tiling; per-molecule unique
    heterogeneous=True,
    doc="Per-molecule conformer sweep (BatchingConfig.conformers_batch_size). Variable across molecules.",
)

# Outermost axis (heterogeneous molecule batching)
N_MOLS = AxisSpec(
    name="n_mols",
    axis_index=5,
    cardinality=64,
    default_batch_size=1,     # safe_map: systems have varying sizes
    tile_granularity=1,
    heterogeneous=True,
    doc="Molecule/system count axis (heterogeneous batch of molecules).",
)

# Backward-compatibility alias (deprecated; planned removal 2026-08-21)
N_SYSTEMS = N_MOLS

# Registry of all axes, sorted by axis_index
ALL_AXES = [N_ATOMS, N_BONDS, N_ANGLES, N_TORSIONS, N_CONFORMERS, N_MOLS]
