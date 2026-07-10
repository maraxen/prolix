"""Type definitions for proxide."""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

ArrayLike = Array | np.ndarray

# Scalar Types
Scalar = Int[ArrayLike, ""]
ScalarFloat = Float[ArrayLike, ""]

# Common Shapes
Coordinates = Float[ArrayLike, "num_atoms 3"]
AtomsMask = Bool[ArrayLike, "num_atoms"]
Radii = Float[ArrayLike, "num_atoms"]
Energy = Float[ArrayLike, ""]
CmapGrid = Float[ArrayLike, "grid_size grid_size"]
CmapCoeffs = Float[ArrayLike, "num_maps grid_size grid_size 16"]
CmapEnergyGrids = Float[ArrayLike, "num_maps grid_size grid_size"]
CmapPoints = Float[ArrayLike, "num_points"]
TorsionAngles = Float[ArrayLike, "num_torsions"]
TorsionIndices = Int[ArrayLike, "num_torsions"]

# Hybrid & Packed Types
VirtualSiteDef = Int[ArrayLike, "num_sites 4"]
VirtualSiteParamsPacked = Float[ArrayLike, "num_sites 12"]
WaterIndicesArray = Int[ArrayLike, "num_waters 3"]
BondIndices = Int[ArrayLike, "num_bonds 2"]
BondParamsPacked = Float[ArrayLike, "num_bonds 2"]
AngleIndices = Int[ArrayLike, "num_angles 3"]
AngleParamsPacked = Float[ArrayLike, "num_angles 2"]

# aliases
PRNGKey = PRNGKeyArray

# PhysicsSystem and IntegratorState definitions
class PhysicsSystem:
    pass

class IntegratorState:
    pass

# -----------------------------------------------------------------------------
# Data Layout Definitions
# -----------------------------------------------------------------------------
# These constants define the layout of packed arrays in SystemParams.
# They provide a single source of truth for "magic indices".

# Virtual Site Parameter Row Layout
# Source: SystemParams["virtual_site_params"] (N_sites, 12)
# [vx, vy, vz, w_o1, w_o2, w_o3, w_x1, w_x2, w_x3, w_y1, w_y2, w_y3]
VS_IDX_POS = slice(0, 3)
VS_IDX_W_ORIGIN = slice(3, 6)
VS_IDX_W_X = slice(6, 9)
VS_IDX_W_Y = slice(9, 12)

# CMAP Torsion Index Row Layout
# Source: SystemParams["cmap_torsions"] (N_torsions, 5)
# [type_idx, a1, a2, a3, a4]
# Phi torsion atoms: a1-a2-a3-a4 -> indices [0, 1, 2, 3] relative to slice?
# OpenMM/Gromacs CMAP torsion format: [map_index, atom1, atom2, atom3, atom4, atom5?] -> usually 5 indices
# Let's verify existing usage in system.py:
# phi_indices = cmap_torsions[:, 0:4] -> indices 0,1,2,3
# psi_indices = cmap_torsions[:, 1:5] -> indices 1,2,3,4
# So the row is [i, j, k, l, m] where Phi=i,j,k,l and Psi=j,k,l,m
CMAP_IDX_PHI = slice(0, 4)
CMAP_IDX_PSI = slice(1, 5)

# Bonded Parameter Row Layouts
BOND_IDX_LENGTH = 0
BOND_IDX_K = 1

ANGLE_IDX_THETA0 = 0
ANGLE_IDX_K = 1

DIHEDRAL_IDX_PERIODICITY = 0
DIHEDRAL_IDX_PHASE = 1
DIHEDRAL_IDX_K = 2

# Water Indices Layout (Oxygen, Hydrogen 1, Hydrogen 2)
WATER_IDX_O = 0
WATER_IDX_H1 = 1
WATER_IDX_H2 = 2


class VirtualSiteParams(NamedTuple):
    """Parameters for virtual site reconstruction.

    Represents a single row from `SystemParams["virtual_site_params"]`.

    Fields:
        p_local: Local coordinates (x, y, z) in the local frame.
        origin_weights: Weights (w1, w2, w3) for parent atoms to define the origin.
        x_weights: Weights (w1, w2, w3) for parent atoms to define the X-axis.
        y_weights: Weights (w1, w2, w3) for parent atoms to define the Y-axis.

    """

    p_local: Array
    origin_weights: Array
    x_weights: Array
    y_weights: Array

    @classmethod
    def from_row(cls, row: Array) -> VirtualSiteParams:
        """Construct params from a packed parameter row.

        Args:
            row: A 1D array of shape (12,) from `SystemParams["virtual_site_params"]`.
                 See Data Layout Definitions above for details.

        """
        return cls(
            p_local=row[VS_IDX_POS],
            origin_weights=row[VS_IDX_W_ORIGIN],
            x_weights=row[VS_IDX_W_X],
            y_weights=row[VS_IDX_W_Y],
        )


class CmapTorsionIndices(NamedTuple):
    """Indices for atoms in a CMAP torsion pair.

    Fields:
        phi_indices: Indices for phi torsion (i, j, k, l)
        psi_indices: Indices for psi torsion (j, k, l, m)
    """

    phi_indices: Int[ArrayLike, 4]
    psi_indices: Int[ArrayLike, 4]

    @classmethod
    def from_row(cls, row: Array) -> CmapTorsionIndices:
        """Construct indices from a packed CMAP torsion row.

        Args:
            row: A 1D array of shape (5,) from `SystemParams["cmap_torsions"]`.
                 Format: [atom_i, atom_j, atom_k, atom_l, atom_m]
                 Defines two sharing torsions:
                 - Phi: i-j-k-l
                 - Psi: j-k-l-m

        """
        return cls(
            phi_indices=row[CMAP_IDX_PHI],
            psi_indices=row[CMAP_IDX_PSI],
        )


class BondParams(NamedTuple):
    """Parameters for a harmonic bond.

    Fields:
        length: Equilibrium bond length.
        k: Bond spring constant (force constant).
    """

    length: ScalarFloat
    k: ScalarFloat

    @classmethod
    def from_row(cls, row: Array) -> BondParams:
        """Construct params from a packed parameter row."""
        return cls(
            length=row[BOND_IDX_LENGTH],
            k=row[BOND_IDX_K],
        )


class AngleParams(NamedTuple):
    """Parameters for a harmonic angle.

    Fields:
        theta0: Equilibrium angle in radians.
        k: Angle spring constant.
    """

    theta0: ScalarFloat
    k: ScalarFloat

    @classmethod
    def from_row(cls, row: Array) -> AngleParams:
        """Construct params from a packed parameter row."""
        return cls(
            theta0=row[ANGLE_IDX_THETA0],
            k=row[ANGLE_IDX_K],
        )


class DihedralParams(NamedTuple):
    """Parameters for a periodic dihedral.

    Fields:
        periodicity: Periodicity of the torsion.
        phase: Phase shift in radians.
        k: Force constant.
    """

    periodicity: ScalarFloat
    phase: ScalarFloat
    k: ScalarFloat

    @classmethod
    def from_row(cls, row: Array) -> DihedralParams:
        """Construct params from a packed parameter row."""
        return cls(
            periodicity=row[DIHEDRAL_IDX_PERIODICITY],
            phase=row[DIHEDRAL_IDX_PHASE],
            k=row[DIHEDRAL_IDX_K],
        )


class WaterIndices(NamedTuple):
    """Indices for atoms in a water molecule.

    Fields:
        oxygen: Index of the oxygen atom.
        hydrogen1: Index of the first hydrogen atom.
        hydrogen2: Index of the second hydrogen atom.
    """

    oxygen: Int[ArrayLike, ""]
    hydrogen1: Int[ArrayLike, ""]
    hydrogen2: Int[ArrayLike, ""]

    @classmethod
    def from_row(cls, row: Array) -> WaterIndices:
        """Construct indices from a row/array of length 3."""
        return cls(
            oxygen=row[WATER_IDX_O],
            hydrogen1=row[WATER_IDX_H1],
            hydrogen2=row[WATER_IDX_H2],
        )


# -----------------------------------------------------------------------------
# Core Equinox Modules (Standardized Types)
# -----------------------------------------------------------------------------


class PhysicsSystem(eqx.Module):
    """A protein system padded to a fixed atom count for vmap compatibility.

    This unifies the previous PhysicsSystem and PaddedSystem into a single
    Equinox-based representation.
    """

    # Per-atom arrays (all shape: (N_padded, ...))
    positions: Array  # (N_padded, 3)
    charges: Array  # (N_padded,)
    sigmas: Array  # (N_padded,)
    epsilons: Array  # (N_padded,)
    radii: Array  # (N_padded,)   — GB radii
    scaled_radii: Array  # (N_padded,)   — OBC scaling factors
    masses: Array  # (N_padded,)
    element_ids: Array  # (N_padded,) int — atomic number (1=H, 6=C, 7=N, 8=O, 16=S)
    atom_mask: Array  # (N_padded,) bool — True for real atoms
    is_hydrogen: Array  # (N_padded,) bool — True for hydrogen atoms
    is_backbone: Array  # (N_padded,) bool — True for backbone atoms (N, CA, C, O)
    is_heavy: Array  # (N_padded,) bool — True for real non-hydrogen atoms
    protein_atom_mask: Array  # (N_padded,) bool — True for protein atoms
    water_atom_mask: Array  # (N_padded,) bool — True for water atoms

    # Bonded term arrays (padded to max per bucket)
    bonds: Array  # (N_bonds_padded, 2) int
    bond_params: Array  # (N_bonds_padded, 2) float
    bond_mask: Array  # (N_bonds_padded,) bool

    angles: Array  # (N_angles_padded, 3) int
    angle_params: Array  # (N_angles_padded, 2) float
    angle_mask: Array  # (N_angles_padded,) bool

    dihedrals: Array  # (N_dih_padded, 4) int
    dihedral_params: Array  # (N_dih_padded, N_terms, 3) float
    dihedral_mask: Array  # (N_padded,) bool

    impropers: Array  # (N_imp_padded, 4) int
    improper_params: Array  # (N_imp_padded, N_terms, 3) float
    improper_mask: Array  # (N_imp_padded,) bool

    # Urey-Bradley
    urey_bradley_bonds: Array | None = None  # (N_ub_padded, 2) int
    urey_bradley_params: Array | None = None  # (N_ub_padded, 2) float
    urey_bradley_mask: Array | None = None  # (N_ub_padded,) bool

    # CMAP (optional)
    cmap_torsions: Array | None = None  # (N_cmap_padded, 5) int
    cmap_indices: Array | None = None  # (N_cmap_padded,) int
    cmap_mask: Array | None = None  # (N_cmap_padded,) bool
    cmap_coeffs: Array | None = None  # (N_maps, G, G, 16) — shared across batch

    # Non-bonded exclusions — sparse per-atom arrays
    excl_indices: Array | None = None  # (N_padded, max_excl) int32
    excl_scales_vdw: Array | None = None  # (N_padded, max_excl) float32
    excl_scales_elec: Array | None = None  # (N_padded, max_excl) float32

    # RATTLE/SHAKE constraints — X-H bond pairs with target lengths
    constraint_pairs: Array | None = None  # (N_constr_padded, 2) int
    constraint_lengths: Array | None = None  # (N_constr_padded,) float
    constraint_mask: Array | None = None  # (N_constr_padded,) bool

    # Metadata
    n_real_atoms: Array | None = None
    n_padded_atoms: int | Array = eqx.field(static=True, default=0)
    bucket_size: int | Array = eqx.field(static=True, default=0)

    # Water molecule indices for SETTLE
    water_indices: Array | None = None  # (N_waters_padded, 3) int
    water_mask: Array | None = None  # (N_waters_padded,) bool
    box_size: Array | None = eqx.field(static=True, default=None)  # (3,)
    pme_alpha: float = eqx.field(static=True, default=0.0)
    pme_grid_points: int = eqx.field(static=True, default=64)
    nonbonded_cutoff: float = eqx.field(static=True, default=9.0)

    # Precomputed dense exclusion matrices
    dense_excl_scale_vdw: Array | None = None  # (N_padded, N_padded) float32
    dense_excl_scale_elec: Array | None = None  # (N_padded, N_padded) float32

    def replace(self: T, **kwargs) -> T:
        """Return a copy with specified fields replaced."""
        return eqx.tree_at(
            lambda s: tuple(getattr(s, k) for k in kwargs.keys()),
            self,
            tuple(kwargs.values()),
            is_leaf=lambda x: x is None,
        )

    def __replace__(self: T, **kwargs) -> T:
        """Python 3.13+ compatibility."""
        return self.replace(**kwargs)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        positions: Array,
        box_size: Array | None = None,
        cutoff_distance: float = 9.0,
    ) -> PhysicsSystem:
        """Creates a PhysicsSystem from a dictionary (legacy test format)."""
        n_atoms = positions.shape[0]

        dihedrals = d.get("dihedrals")
        if dihedrals is None:
            dihedrals = d.get("proper_dihedrals")

        return cls(
            positions=positions,
            charges=jnp.asarray(d["charges"]),
            sigmas=jnp.asarray(d["sigmas"]),
            epsilons=jnp.asarray(d["epsilons"]),
            masses=jnp.asarray(d.get("masses", jnp.ones(n_atoms))),
            box_size=box_size,
            bonds=d.get("bonds"),
            bond_params=d.get("bond_params"),
            angles=d.get("angles"),
            angle_params=d.get("angle_params"),
            dihedrals=dihedrals,
            dihedral_params=d.get("dihedral_params"),
            impropers=d.get("impropers"),
            improper_params=d.get("improper_params"),
            urey_bradley_bonds=d.get("urey_bradley_bonds"),
            urey_bradley_params=d.get("urey_bradley_params"),
            # Mandatory fields with defaults
            radii=d.get("radii", jnp.ones(n_atoms)),
            scaled_radii=d.get("scaled_radii", jnp.ones(n_atoms)),
            element_ids=d.get("element_ids", jnp.zeros(n_atoms, dtype=jnp.int32)),
            atom_mask=d.get("atom_mask", jnp.ones(n_atoms, dtype=bool)),
            is_hydrogen=d.get("is_hydrogen", jnp.zeros(n_atoms, dtype=bool)),
            is_backbone=d.get("is_backbone", jnp.zeros(n_atoms, dtype=bool)),
            is_heavy=d.get("is_heavy", jnp.ones(n_atoms, dtype=bool)),
            protein_atom_mask=d.get("protein_atom_mask", jnp.ones(n_atoms, dtype=bool)),
            water_atom_mask=d.get("water_atom_mask", jnp.zeros(n_atoms, dtype=bool)),
            bond_mask=d.get(
                "bond_mask", jnp.ones(d.get("bonds", jnp.zeros((0, 2))).shape[0], dtype=bool)
            ),
            angle_mask=d.get(
                "angle_mask",
                jnp.ones(d.get("angles", jnp.zeros((0, 3))).shape[0], dtype=bool),
            ),
            dihedral_mask=d.get(
                "dihedral_mask",
                jnp.ones(dihedrals.shape[0] if dihedrals is not None else 0, dtype=bool),
            ),
            improper_mask=d.get(
                "improper_mask",
                jnp.ones(d.get("impropers", jnp.zeros((0, 4))).shape[0], dtype=bool),
            ),
            urey_bradley_mask=d.get(
                "urey_bradley_mask",
                jnp.ones(
                    d.get("urey_bradley_bonds", jnp.zeros((0, 2))).shape[0], dtype=bool
                ),
            ),
            excl_indices=d.get("excl_indices"),
            excl_scales_vdw=d.get("excl_scales_vdw"),
            excl_scales_elec=d.get("excl_scales_elec"),
            dense_excl_scale_vdw=d.get("exclusion_mask"),
            dense_excl_scale_elec=d.get("exclusion_mask"),
            nonbonded_cutoff=cutoff_distance,
        )


class DifferentiableParams(eqx.Module):
    """Parameters for energy functions, structured for easy autodiff."""

    charges: Array
    sigmas: Array
    epsilons: Array
    bond_params: Array
    angle_params: Array
    dihedral_params: Array
    improper_params: Array
    urey_bradley_params: Array
    pme_alpha: Array
    box_size: Array

    @classmethod
    def from_system(cls, system: PhysicsSystem) -> DifferentiableParams:
        """Initialize from a PhysicsSystem."""
        return cls(
            charges=system.charges,
            sigmas=system.sigmas,
            epsilons=system.epsilons,
            bond_params=system.bond_params,
            angle_params=system.angle_params,
            dihedral_params=system.dihedral_params,
            improper_params=system.improper_params,
            urey_bradley_params=system.urey_bradley_params
            if system.urey_bradley_params is not None
            else jnp.zeros((0, 2)),
            pme_alpha=jnp.array(system.pme_alpha),
            box_size=system.box_size if system.box_size is not None else jnp.zeros((3,)),
        )


class EnergyParams(eqx.Module):
    """Container for energy function parameters."""

    params: dict[str, Any] | DifferentiableParams

    @property
    def charges(self) -> Array:
        if isinstance(self.params, DifferentiableParams):
            return self.params.charges
        return self.params["charges"]

    @property
    def sigmas(self) -> Array:
        if isinstance(self.params, DifferentiableParams):
            return self.params.sigmas
        return self.params["sigmas"]

    @property
    def epsilons(self) -> Array:
        if isinstance(self.params, DifferentiableParams):
            return self.params.epsilons
        return self.params["epsilons"]


class IntegratorParams(eqx.Module):
    """Parameters for MD integration steps."""

    dt: float | Array
    kT: float | Array
    gamma: float | Array
    energy_params: EnergyParams
    water_indices: Array = eqx.field(
        default_factory=lambda: jnp.zeros((0, 3), dtype=jnp.int32)
    )
    constraint_dofs: Array | None = eqx.field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.int32)
    )
    molecule_indices: Array = eqx.field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.int32)
    )
    box: Array = eqx.field(default_factory=lambda: jnp.zeros((3,)))
    positions_old: Array | None = eqx.field(default_factory=lambda: jnp.zeros((0, 3)))
    n_dof: float | Array | None = 0.0
    target_pressure_bar: float = 0.0
    barostat_interval: int = 1
    compressibility: float = 4.5e-5  # bar^-1 (isothermal compressibility of water)
    tau_barostat: float = 2000.0  # AKMA (~0.1 ps)
    virtual_site_params: Any = None


class IntegratorState(eqx.Module):
    """Minimal integrator state for step composition.

    Supports both unbatched and batched (vmap) modes seamlessly.

    Attributes:
        positions: Atomic positions. Shape (N, 3) unbatched or (B, N, 3) batched.
        momentum: Atomic momenta. Shape (N, 3) unbatched or (B, N, 3) batched.
        force: Atomic forces. Shape (N, 3) unbatched or (B, N, 3) batched.
        mass: Atomic masses. Shape (N,) or (N, 1), shared across batch.
        rng: JAX PRNGKey. Shape (2,) unbatched or (B, 2) batched.
        cap_count: Optional force capping counter.
        warn_counts: Optional quality gate counters.
        potential_energy: Optional cached potential energy.
        did_overflow: Optional neighbor list overflow flag.
        last_update_positions: Optional positions at last neighbor list update.
        box: Optional box dimensions. Shape (3,) if provided, shared across batch.
        step_count: Current step number.
    """

    positions: Array
    momentum: Array
    force: Array
    mass: Array
    rng: PRNGKey
    cap_count: Array = eqx.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    warn_counts: Array = eqx.field(default_factory=lambda: jnp.zeros(4, dtype=jnp.int32))
    potential_energy: Array | None = eqx.field(default=None)
    did_overflow: Array = eqx.field(default_factory=lambda: jnp.array(False, dtype=jnp.bool_))
    last_update_positions: Array | None = eqx.field(default=None)
    box: Array | None = eqx.field(default=None)
    step_count: Array = eqx.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

    def __post_init__(self):
        """Auto-init warn_counts when explicitly passed as None (batch-aware)."""
        if self.warn_counts is None:
            n_types = 4  # NUM_WARN_TYPES — class attrs not yet bound in __post_init__
            if self.positions.ndim >= 3:
                b = self.positions.shape[0]
                object.__setattr__(
                    self, "warn_counts", jnp.zeros((b, n_types), dtype=jnp.int32)
                )
            else:
                object.__setattr__(
                    self, "warn_counts", jnp.zeros(n_types, dtype=jnp.int32)
                )

    # Named indices for warn_counts (Langevin compatibility)
    WARN_VLIMIT = 0
    WARN_FORCE_CAP = 1
    WARN_CONSTR_VIOL = 2
    WARN_DX_CAP = 3
    NUM_WARN_TYPES = 4

    @property
    def key(self) -> PRNGKey:
        """Alias for rng."""
        return self.rng

    @property
    def position(self) -> Array:
        """Alias for positions."""
        return self.positions

    @property
    def momenta(self) -> Array:
        """Alias for momentum (legacy batched API)."""
        return self.momentum

    @property
    def forces(self) -> Array:
        """Alias for force (legacy batched API)."""
        return self.force

    def replace(self: S, **kwargs) -> S:
        """Return a copy with specified fields replaced."""
        # Handle 'key' -> 'rng' alias in replace
        if "key" in kwargs:
            kwargs["rng"] = kwargs.pop("key")
        if "position" in kwargs:
            kwargs["positions"] = kwargs.pop("position")
        if "momenta" in kwargs:
            kwargs["momentum"] = kwargs.pop("momenta")
        if "forces" in kwargs:
            kwargs["force"] = kwargs.pop("forces")

        return eqx.tree_at(
            lambda s: tuple(getattr(s, k) for k in kwargs.keys()),
            self,
            tuple(kwargs.values()),
            is_leaf=lambda x: x is None,
        )

    def __replace__(self: S, **kwargs) -> S:
        """Python 3.13+ compatibility."""
        return self.replace(**kwargs)



@eqx.filter_jit
class NVTLangevinState(eqx.Module):
    """Minimal state for NVT Langevin integrators."""

    positions: Array
    momentum: Array
    force: Array
    mass: Array
    rng: PRNGKey

    @property
    def position(self) -> Array:
        """Alias for positions (JAX-MD compatibility)."""
        return self.positions

    @property
    def key(self) -> PRNGKey:
        """Alias for rng (JAX-MD compatibility)."""
        return self.rng


@eqx.filter_jit
class NPTState(eqx.Module):
    """State for NPT (isothermal-isobaric) ensemble integrators."""

    positions: Array
    momentum: Array
    force: Array
    mass: Array
    rng: PRNGKey
    box: Array

    @property
    def position(self) -> Array:
        """Alias for positions (JAX-MD compatibility)."""
        return self.positions

    @property
    def key(self) -> PRNGKey:
        """Alias for rng (JAX-MD compatibility)."""
        return self.rng


class SimulationState(eqx.Module):
    """State of the simulation at a specific timepoint, optimized for storage."""

    positions: Array
    velocities: Array

    # Scalar state (required)
    step: int | Array
    time_ns: float | Array

    # Optional fields (must come after required)
    forces: Array | None = None
    mass: Array | None = None

    # Energies (optional)
    potential_energy: float | Array | None = None
    kinetic_energy: float | Array | None = None

    @property
    def position(self) -> Array:
        """Alias for positions."""
        return self.positions

    def numpy(self) -> dict[str, Any]:
        """Convert state to a dictionary of numpy arrays (on CPU)."""

        def to_cpu(x):
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return x
            # Use jax.device_put to ensure it's on CPU if needed, then asarray
            return np.asarray(x)

        cpu_state = jax.tree_util.tree_map(to_cpu, self)

        return {
            "positions": cpu_state.positions,
            "velocities": cpu_state.velocities,
            "forces": cpu_state.forces,
            "mass": cpu_state.mass,
            "step": cpu_state.step,
            "time_ns": cpu_state.time_ns,
            "potential_energy": cpu_state.potential_energy,
            "kinetic_energy": cpu_state.kinetic_energy,
        }

    def to_array_record(self) -> bytes:
        """Serialize the state to msgpack bytes for ArrayRecord."""
        import msgpack_numpy as m

        data = self.numpy()
        # Filter None values to save space
        data = {k: v for k, v in data.items() if v is not None}
        return m.packb(data)

    @classmethod
    def from_array_record(cls, packed: bytes) -> SimulationState:
        """Deserialize a SimulationState from msgpack bytes."""
        import msgpack_numpy as m

        data = m.unpackb(packed)

        def to_jax(x):
            return jnp.array(x) if x is not None else None

        return cls(
            positions=to_jax(data["positions"]),
            velocities=to_jax(data["velocities"]),
            forces=to_jax(data.get("forces")),
            mass=to_jax(data.get("mass")),
            step=to_jax(data["step"]),
            time_ns=to_jax(data["time_ns"]),
            potential_energy=to_jax(data.get("potential_energy")),
            kinetic_energy=to_jax(data.get("kinetic_energy")),
        )


PaddedSystem = PhysicsSystem
