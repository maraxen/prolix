"""Prolix: Protein Physics and Molecular Dynamics in JAX."""

__version__ = "0.1.0"

from .batched_energy import make_batched_energy_fn, single_padded_energy
from .padding import bucket_proteins, collate_batch, pad_protein
from .padding import PaddedSystem as _PaddedSystem
from .export import export_energy_fn, export_langevin_step, save_artifact, load_artifact

__all__ = [
    "pad_protein",
    "bucket_proteins",
    "collate_batch",
    "PaddedSystem",
    "make_batched_energy_fn",
    "single_padded_energy",
    # StableHLO export (v1.1 Item 4)
    "export_energy_fn",
    "export_langevin_step",
    "save_artifact",
    "load_artifact",
]


def __getattr__(name: str):
    """Emit deprecation warning for legacy PaddedSystem import path."""
    if name == "PaddedSystem":
        import warnings
        warnings.warn(
            "PaddedSystem is a deprecated alias for PhysicsSystem. "
            "Migrate to MolecularBundle (prolix.types.bundles) for new code. "
            "Removal planned for v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _PaddedSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
