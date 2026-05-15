"""Prolix-specific AxisSpec instances for molecular simulation axes."""
from prolix.tiling.planner import AxisSpec

N_ATOMS = AxisSpec(
    name="n_atoms",
    axis_index=1,
    cardinality=60_000,
    default_batch_size=0,     # 0 = vmap when homogeneous
    tile_granularity=128,     # Pallas kernel alignment
    heterogeneous=False,
    doc="Atom count axis (homogeneous trajectory batching)",
)

N_SYSTEMS = AxisSpec(
    name="n_systems",
    axis_index=0,
    cardinality=64,
    default_batch_size=1,     # safe_map: systems have varying sizes
    tile_granularity=1,
    heterogeneous=True,
    doc="System count axis (heterogeneous batch of systems)",
)
