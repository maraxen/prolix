"""Static bonded connectivity for differentiable force-field fitting.

BondedTopology stores atom indices and torsion periodicity (static).
These are frozen at construction; they do not participate in gradient computation.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class BondedTopology:
    """Static bonded connectivity. NOT trainable — atom indices and Fourier periodicity.

    Stored as numpy arrays at construction time; the BondedParams module will
    reference these with eqx.field(static=True).

    Attributes:
        bond_idx: (N_bonds, 2) int32 array of [i, j] atom indices.
        angle_idx: (N_angles, 3) int32 array of [i, j, k] atom indices (j is vertex).
        torsion_idx: (N_torsions, 4) int32 array of [i, j, k, l] atom indices.
        torsion_periodicity: (N_torsions, n_terms) int32 array of Fourier periodicities.
            n_terms=1 in v0 (one term per torsion); may extend to multi-term cosine series.
        torsion_phase_rad: (N_torsions, n_terms) float32 array of phase offsets (radians).
            Static; not updated during fitting.
    """

    bond_idx: np.ndarray
    angle_idx: np.ndarray
    torsion_idx: np.ndarray
    torsion_periodicity: np.ndarray
    torsion_phase_rad: np.ndarray

    def as_jnp_arrays(self):
        """Convert numpy arrays to jax.numpy arrays.

        Used internally when topology is passed to JAX-traced functions.
        """
        return BondedTopology(
            bond_idx=jnp.asarray(self.bond_idx),
            angle_idx=jnp.asarray(self.angle_idx),
            torsion_idx=jnp.asarray(self.torsion_idx),
            torsion_periodicity=jnp.asarray(self.torsion_periodicity),
            torsion_phase_rad=jnp.asarray(self.torsion_phase_rad),
        )

    @property
    def n_bonds(self) -> int:
        return len(self.bond_idx)

    @property
    def n_angles(self) -> int:
        return len(self.angle_idx)

    @property
    def n_torsions(self) -> int:
        return len(self.torsion_idx)

    @property
    def n_torsion_terms(self) -> int:
        """Number of Fourier terms per torsion (e.g., 1 for v0)."""
        if len(self.torsion_periodicity) == 0:
            return 0
        return self.torsion_periodicity.shape[1]

    @property
    def n_atoms(self) -> int:
        """Infer number of atoms from max atom index across all bonds, angles, torsions.

        Note: This is an upper bound; if there are isolated atoms with no bonded terms,
        they won't be counted. For safety, call with knowledge of the system.
        """
        max_idx = 0
        if len(self.bond_idx) > 0:
            max_idx = max(max_idx, int(jnp.max(self.bond_idx)))
        if len(self.angle_idx) > 0:
            max_idx = max(max_idx, int(jnp.max(self.angle_idx)))
        if len(self.torsion_idx) > 0:
            max_idx = max(max_idx, int(jnp.max(self.torsion_idx)))
        return max_idx + 1
