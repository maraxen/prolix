"""OpenMM System Converter for the Prolix Backend.

Converts openmm.System, openmm.Integrator, and coordinates into Prolix
MolecularBundle, CompiledLeptonForces, and ConvertedOpenMMSystem.
"""

from __future__ import annotations

import collections
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import openmm
from jaxtyping import Array, Float, Int
from openmm import unit

from prolix.batched_energy import solve_ewald_alpha
from prolix.openmm.config import ProlixOpenMMConfig
from prolix.openmm.exceptions import (
    UnsupportedOpenMMForceError,
    check_unsupported_force,
)
from prolix.openmm.lepton_compiler import (
    CompiledLeptonForce,
    compile_custom_force,
    minimum_image_displacement,
)
from prolix.openmm.pattern_matcher import (
    fold_matched_pattern_into_bundle,
    match_custom_force_pattern,
)
from prolix.openmm.units import (
    AKMA_PRESSURE_PER_BAR,
    ANGLE_K_OMM_TO_AKMA,
    BOND_K_OMM_TO_AKMA,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
    TORSION_K_OMM_TO_AKMA,
)
from prolix.physics.cmap import precompute_cmap_coefficients
from prolix.physics.neighbor_list import pair_list_to_dense_padded
from prolix.physics.pbc import reduce_triclinic_box_vectors
from prolix.physics.pme import compute_pme_grid_dims
from prolix.types.bundles import (
    ANGLE_BUCKETS,
    ATOM_BUCKETS,
    BOND_BUCKETS,
    CMAP_BUCKETS,
    CMAP_MAP_BUCKETS,
    CONSTRAINT_BUCKETS,
    DIHEDRAL_BUCKETS,
    EXCEPTION_BUCKETS,
    EXCL_BUCKETS,
    WATER_BUCKETS,
    MolecularBundle,
    MolecularShapeSpec,
    _bucket_idx,
)

# Maximum supported exclusions per atom in dense row representations
_EXCL_DENSE_MAX = 32

# Standard TIP3P default geometry (in Ångströms)
_DEFAULT_TIP3P_ROH = 0.9572
_DEFAULT_TIP3P_RHH = 1.5139


class CompoundMapping(dict[Any, Any]):
    """Dictionary supporting compound tuple keys with single-force flat integer lookup."""

    def __init__(
        self,
        data: dict[tuple[int, int], Any] | None = None,
        default_force_idx: int = 0,
        is_single: bool = True,
    ) -> None:
        super().__init__(data or {})
        self.default_force_idx = default_force_idx
        self.is_single = is_single

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int) and self.is_single:
            return super().__getitem__((self.default_force_idx, key))
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, int) and self.is_single:
            return super().__contains__((self.default_force_idx, key))
        return super().__contains__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, int) and self.is_single:
            return super().get((self.default_force_idx, key), default)
        return super().get(key, default)


# =============================================================================
# 1. Virtual Sites Datastructure & Reconstruction
# =============================================================================


@jax.custom_vjp
def _eval_local_coordinates_site(
    r1: Float[Array, 3],
    r2: Float[Array, 3],
    r3: Float[Array, 3],
    p_local: Float[Array, 3],
    wo: Float[Array, 3],
    wx: Float[Array, 3],
    wy: Float[Array, 3],
) -> Float[Array, 3]:
    """Compute LocalCoordinatesSite position with safe coordinate frame."""
    origin = wo[0] * r1 + wo[1] * r2 + wo[2] * r3
    xdir = wx[0] * r1 + wx[1] * r2 + wx[2] * r3
    ydir = wy[0] * r1 + wy[1] * r2 + wy[2] * r3
    zdir = jnp.cross(xdir, ydir)

    norm_x = jnp.linalg.norm(xdir)
    norm_z = jnp.linalg.norm(zdir)

    safe_norm_x = jnp.where(norm_x > 0.0, norm_x, 1.0)
    safe_norm_z = jnp.where(norm_z > 0.0, norm_z, 1.0)

    inv_norm_x = jnp.where(norm_x > 0.0, 1.0 / safe_norm_x, 0.0)
    inv_norm_z = jnp.where(norm_z > 0.0, 1.0 / safe_norm_z, 0.0)

    dx = xdir * inv_norm_x
    dz = zdir * inv_norm_z
    dy = jnp.cross(dz, dx)

    return origin + dx * p_local[0] + dy * p_local[1] + dz * p_local[2]


def _eval_local_coordinates_site_fwd(
    r1: Float[Array, 3],
    r2: Float[Array, 3],
    r3: Float[Array, 3],
    p_local: Float[Array, 3],
    wo: Float[Array, 3],
    wx: Float[Array, 3],
    wy: Float[Array, 3],
):
    pos = _eval_local_coordinates_site(r1, r2, r3, p_local, wo, wx, wy)
    return pos, (r1, r2, r3, p_local, wo, wx, wy)


def _eval_local_coordinates_site_bwd(res, f):
    """Analytical OpenMM ReferenceVirtualSites Jacobian VJP."""
    r1, r2, r3, p_local, wo, wx, wy = res
    xdir = wx[0] * r1 + wx[1] * r2 + wx[2] * r3
    ydir = wy[0] * r1 + wy[1] * r2 + wy[2] * r3
    zdir = jnp.cross(xdir, ydir)

    norm_x = jnp.linalg.norm(xdir)
    norm_z = jnp.linalg.norm(zdir)

    safe_norm_x = jnp.where(norm_x > 0.0, norm_x, 1.0)
    safe_norm_z = jnp.where(norm_z > 0.0, norm_z, 1.0)

    inv_norm_x = jnp.where(norm_x > 0.0, 1.0 / safe_norm_x, 0.0)
    inv_norm_z = jnp.where(norm_z > 0.0, 1.0 / safe_norm_z, 0.0)

    dx = xdir * inv_norm_x
    dz = zdir * inv_norm_z
    dy = jnp.cross(dz, dx)

    wx_scaled = wx * inv_norm_x
    fp1 = p_local * f[0]
    fp2 = p_local * f[1]
    fp3 = p_local * f[2]

    def _force_on_particle(j: int):
        wx_j = wx[j]
        wy_j = wy[j]
        wo_j = wo[j]
        wx_s = wx_scaled[j]

        t1 = (wx_j * ydir[0] - wy_j * xdir[0]) * inv_norm_z
        t2 = (wx_j * ydir[1] - wy_j * xdir[1]) * inv_norm_z
        t3 = (wx_j * ydir[2] - wy_j * xdir[2]) * inv_norm_z

        sx = t3 * dz[1] - t2 * dz[2]
        sy = t1 * dz[2] - t3 * dz[0]
        sz = t2 * dz[0] - t1 * dz[1]

        fx = (
            fp1[0] * wx_s * (1.0 - dx[0] * dx[0])
            + fp1[2] * (dz[0] * sx)
            + fp1[1] * ((-dx[0] * dy[0]) * wx_s + dy[0] * sx - dx[1] * t2 - dx[2] * t3)
            + f[0] * wo_j
            + fp2[0] * wx_s * (-dx[1] * dx[0])
            + fp2[2] * (dz[1] * sx - t3)
            - fp2[1] * ((dx[0] * dy[1] - dz[2]) * wx_s - dy[1] * sx - dx[0] * t2)
            + fp3[0] * wx_s * (-dx[2] * dx[0])
            + fp3[2] * (dz[2] * sx + t2)
            + fp3[1] * ((-dx[0] * dy[2] - dz[1]) * wx_s + dy[2] * sx + dx[0] * t3)
        )
        fy = (
            fp1[0] * wx_s * (-dx[0] * dx[1])
            + fp1[2] * (dz[0] * sy + t3)
            + fp1[1] * ((-dx[1] * dy[0] - dz[2]) * wx_s + dy[0] * sy + dx[1] * t1)
            + fp2[0] * wx_s * (1.0 - dx[1] * dx[1])
            + fp2[2] * (dz[1] * sy)
            - fp2[1] * ((dx[1] * dy[1]) * wx_s - dy[1] * sy + dx[0] * t1 + dx[2] * t3)
            + f[1] * wo_j
            + fp3[0] * wx_s * (-dx[2] * dx[1])
            + fp3[2] * (dz[2] * sy - t1)
            + fp3[1] * ((-dx[1] * dy[2] + dz[0]) * wx_s + dy[2] * sy + dx[1] * t3)
        )
        fz = (
            fp1[0] * wx_s * (-dx[0] * dx[2])
            + fp1[2] * (dz[0] * sz - t2)
            + fp1[1] * ((-dx[2] * dy[0] + dz[1]) * wx_s + dy[0] * sz + dx[2] * t1)
            + fp2[0] * wx_s * (-dx[1] * dx[2])
            + fp2[2] * (dz[1] * sz + t1)
            - fp2[1] * ((dx[2] * dy[1] + dz[0]) * wx_s - dy[1] * sz - dx[2] * t2)
            + fp3[0] * wx_s * (1.0 - dx[2] * dx[2])
            + fp3[2] * (dz[2] * sz)
            + fp3[1] * ((-dx[2] * dy[2]) * wx_s + dy[2] * sz - dx[0] * t1 - dx[1] * t2)
            + f[2] * wo_j
        )
        return jnp.array([fx, fy, fz])

    f1 = _force_on_particle(0)
    f2 = _force_on_particle(1)
    f3 = _force_on_particle(2)
    return (f1, f2, f3, None, None, None, None)


_eval_local_coordinates_site.defvjp(
    _eval_local_coordinates_site_fwd,
    _eval_local_coordinates_site_bwd,
)


class VirtualSiteGeneration(eqx.Module):
    """A single topological generation of virtual sites evaluated in parallel.

    All virtual sites in this generation depend strictly on particles from
    earlier generations (or real particles), guaranteeing race-free parallel evaluation.
    """

    linear_indices: Int[Array, "N_lin 4"]
    linear_params: Float[Array, "N_lin 4"]
    local_indices: Int[Array, "N_loc 4"]
    local_params: Float[Array, "N_loc 12"]

    n_linear: int = eqx.field(static=True)
    n_local: int = eqx.field(static=True)

    def reconstruct(
        self,
        pos: Float[Array, "... 3"],
    ) -> Float[Array, "... 3"]:
        """Reconstruct positions for all virtual sites in this generation."""
        # 1. Linear & Out-of-plane sites: branchless, bilinear, zero NaN risk
        if self.n_linear > 0:
            lin_params = self.linear_params.astype(pos.dtype)

            def _eval_linear(idx: Array, prm: Array) -> Array:
                r1 = pos[idx[1]]
                r2 = pos[idx[2]]
                r3 = pos[idx[3]]
                w1, w2, w3, w_cross = prm[0], prm[1], prm[2], prm[3]
                v12 = r2 - r1
                v13 = r3 - r1
                cross = jnp.cross(v12, v13)
                return w1 * r1 + w2 * r2 + w3 * r3 + w_cross * cross

            vs_linear_pos = jax.vmap(_eval_linear)(self.linear_indices, lin_params)
            pos = pos.at[self.linear_indices[:, 0]].set(vs_linear_pos.astype(pos.dtype))

        # 2. Local coordinates sites: exact analytical Jacobian VJP
        if self.n_local > 0:
            loc_params = self.local_params.astype(pos.dtype)

            def _eval_local(idx: Array, prm: Array) -> Array:
                r1 = pos[idx[1]]
                r2 = pos[idx[2]]
                r3 = pos[idx[3]]
                p_local = prm[0:3]
                wo = prm[3:6]
                wx = prm[6:9]
                wy = prm[9:12]
                return _eval_local_coordinates_site(r1, r2, r3, p_local, wo, wx, wy)

            vs_local_pos = jax.vmap(_eval_local)(self.local_indices, loc_params)
            pos = pos.at[self.local_indices[:, 0]].set(vs_local_pos.astype(pos.dtype))

        return pos


class VirtualSites(eqx.Module):
    """Immutable Equinox module orchestrating topological virtual site generations."""

    generations: tuple[VirtualSiteGeneration, ...]
    order: Int[Array, N_total]
    real_particle_indices: Int[Array, N_real]

    n_generations: int = eqx.field(static=True)
    n_linear: int = eqx.field(static=True)
    n_local: int = eqx.field(static=True)
    n_total: int = eqx.field(static=True)
    n_particles: int = eqx.field(static=True)

    def reconstruct_positions(
        self,
        positions: Float[Array, "... 3"],
        eps: float = 1e-12,
    ) -> Float[Array, "... 3"]:
        """Reconstruct all virtual site positions in forward generational order.

        Args:
            positions: Coordinates with real atoms populated (shape (..., 3)).
            eps: Epsilon parameter preserved for API backward compatibility.

        Returns:
            Coordinates with all virtual site slots updated.
        """
        if positions.ndim > 2:
            leading_shape = positions.shape[:-2]
            flat_pos = positions.reshape(-1, positions.shape[-2], positions.shape[-1])
            flat_res = jax.vmap(lambda p: self.reconstruct_positions(p, eps=eps))(flat_pos)
            return flat_res.reshape(*leading_shape, positions.shape[-2], positions.shape[-1])

        pos = positions
        for gen in self.generations:
            pos = gen.reconstruct(pos)
        return pos

    def redistribute_forces(
        self,
        raw_forces: Float[Array, "... 3"],
        positions: Float[Array, "... 3"],
    ) -> Float[Array, "... 3"]:
        """Redistribute virtual site forces to parent atoms using reverse-mode autodiff.

        Matches OpenMM Reference Platform:
        - Parent atoms receive direct force + accumulated virtual site forces.
        - Virtual site slots retain their raw direct forces.

        Args:
            raw_forces: Raw potential forces on all particles (shape (..., 3)).
            positions: Current atomic positions (shape (..., 3)).

        Returns:
            Forces with parent forces updated via VJP.
        """
        if self.n_total == 0:
            return raw_forces

        real_idx = self.real_particle_indices
        real_pos = positions[real_idx]

        def _full_positions(r_real: Array) -> Array:
            p = positions.at[real_idx].set(r_real)
            return self.reconstruct_positions(p)

        _, vjp_fn = jax.vjp(_full_positions, real_pos)
        redistributed_real_forces = vjp_fn(raw_forces)[0]

        return raw_forces.at[real_idx].set(redistributed_real_forces.astype(raw_forces.dtype))

    # Backward compatibility properties for introspection
    @property
    def linear_indices(self) -> Int[Array, "N_lin 4"]:
        arrays = [g.linear_indices for g in self.generations if g.n_linear > 0]
        return jnp.concatenate(arrays, axis=0) if arrays else jnp.zeros((0, 4), dtype=jnp.int32)

    @property
    def linear_params(self) -> Float[Array, "N_lin 4"]:
        arrays = [g.linear_params for g in self.generations if g.n_linear > 0]
        return jnp.concatenate(arrays, axis=0) if arrays else jnp.zeros((0, 4), dtype=jnp.float32)

    @property
    def local_indices(self) -> Int[Array, "N_loc 4"]:
        arrays = [g.local_indices for g in self.generations if g.n_local > 0]
        return jnp.concatenate(arrays, axis=0) if arrays else jnp.zeros((0, 4), dtype=jnp.int32)

    @property
    def local_params(self) -> Float[Array, "N_loc 12"]:
        arrays = [g.local_params for g in self.generations if g.n_local > 0]
        return jnp.concatenate(arrays, axis=0) if arrays else jnp.zeros((0, 12), dtype=jnp.float32)


# =============================================================================
# 2. Harmonic Position Restraint Container
# =============================================================================


class PositionRestraints(eqx.Module):
    """Harmonic position restraints extracted from CustomExternalForce."""

    indices: Int[Array, R]
    k_x: Float[Array, R]
    k_y: Float[Array, R]
    k_z: Float[Array, R]
    x0: Float[Array, R]
    y0: Float[Array, R]
    z0: Float[Array, R]
    force_group: int = eqx.field(static=True, default=0)


# =============================================================================
# 3. Ryckaert-Bellemans Torsion Force
# =============================================================================


class CompiledRBTorsionForce(CompiledLeptonForce):
    """Evaluates Ryckaert-Bellemans torsions in Prolix via Horner polynomial."""

    torsion_indices: Int[Array, "M 4"]
    c_params: Float[Array, "M 6"]  # c0..c5 in kJ/mol

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        del global_params
        if self.torsion_indices.shape[0] == 0:
            return jnp.asarray(0.0, dtype=positions_nm.dtype)

        rA = positions_nm[self.torsion_indices[:, 0]]
        rB = positions_nm[self.torsion_indices[:, 1]]
        rC = positions_nm[self.torsion_indices[:, 2]]
        rD = positions_nm[self.torsion_indices[:, 3]]

        box = box_vectors_nm if self.uses_periodic else None
        b0 = minimum_image_displacement(rA, rB, box)
        b1 = minimum_image_displacement(rC, rB, box)
        b2 = minimum_image_displacement(rD, rC, box)

        b1_norm = jnp.sqrt(jnp.sum(b1 * b1, axis=-1, keepdims=True) + 1e-12)
        b1_unit = b1 / b1_norm

        v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
        w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

        x_val = jnp.sum(v * w, axis=-1)
        y_val = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

        is_deg = (x_val * x_val + y_val * y_val) < 1e-20
        safe_x = jnp.where(is_deg, jnp.ones_like(x_val), x_val)
        safe_y = jnp.where(is_deg, jnp.zeros_like(y_val), y_val)
        theta = jnp.where(is_deg, jnp.zeros_like(x_val), jnp.arctan2(safe_y, safe_x))

        # Ryckaert-Bellemans: psi = theta - pi -> cos(psi) = -cos(theta)
        x = -jnp.cos(theta)

        # Horner polynomial: c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*c5))))
        c = self.c_params
        energy = c[:, 0] + x * (
            c[:, 1] + x * (c[:, 2] + x * (c[:, 3] + x * (c[:, 4] + x * c[:, 5])))
        )
        return jnp.sum(energy)


# =============================================================================
# 4. Converted OpenMM System Datastructure
# =============================================================================


class ConvertedOpenMMSystem(eqx.Module):
    """Complete converted OpenMM System bundle ready for ProlixContext execution."""

    bundle: MolecularBundle
    custom_forces: list[Any]
    force_groups: dict[int, list[str]] = eqx.field(static=True)
    global_parameters: dict[str, float] = eqx.field(static=True)
    system_metadata: dict[str, Any] = eqx.field(static=True)
    virtual_sites: VirtualSites | None = None
    position_restraints: list[PositionRestraints] = eqx.field(
        default_factory=list
    )
    force_group_by_component: dict[str, int] = eqx.field(
        static=True, default_factory=dict
    )
    default_parameters: dict[str, float] = eqx.field(static=True, default_factory=dict)
    nonbonded_method: str = eqx.field(static=True, default="pme")
    nonbonded_cutoff: float = eqx.field(static=True, default=9.0)
    lj_switch_width: float = eqx.field(static=True, default=0.0)
    use_dispersion_correction: bool = eqx.field(static=True, default=True)
    reaction_field_dielectric: float = eqx.field(static=True, default=78.3)
    pme_grid_dims: tuple[int, int, int] | None = eqx.field(static=True, default=None)
    pme_order: int = eqx.field(static=True, default=4)
    barostat: Any = eqx.field(static=True, default=None)
    remove_com_motion: bool = eqx.field(static=True, default=False)
    com_motion_frequency: int = eqx.field(static=True, default=10)
    n_real_particles: int = eqx.field(static=True, default=0)
    has_pbc: bool = eqx.field(static=True, default=False)
    is_triclinic: bool = eqx.field(static=True, default=False)
    bond_force_groups: Array | None = None
    angle_force_groups: Array | None = None
    dihedral_force_groups: Array | None = None
    cmap_force_groups: Array | None = None
    has_nonbonded: bool = eqx.field(static=True, default=False)
    has_implicit_solvent: bool = eqx.field(static=True, default=False)
    solvent_dielectric: float = eqx.field(static=True, default=78.5)
    solute_dielectric: float = eqx.field(static=True, default=1.0)
    custom_nonbonded_chunk_size: int = eqx.field(static=True, default=128)

    def compute_virtual_sites(
        self, positions: Float[Array, "... 3"]
    ) -> Float[Array, "... 3"]:
        """Reconstruct virtual sites on the given atomic coordinates."""
        if self.virtual_sites is not None:
            return self.virtual_sites.reconstruct_positions(positions)
        return positions

    def redistribute_virtual_site_forces(
        self,
        forces: Float[Array, "... 3"],
        positions: Float[Array, "... 3"],
    ) -> Float[Array, "... 3"]:
        """Redistribute forces from virtual sites to real parent atoms."""
        if self.virtual_sites is not None:
            return self.virtual_sites.redistribute_forces(forces, positions)
        return forces

    @property
    def inv_mass(self) -> Float[Array, ...]:
        """Safe inverse mass array (zero for virtual sites and fixed particles)."""
        safe_m = jnp.where(self.bundle.masses > 1e-12, self.bundle.masses, 1.0)
        return jnp.where(self.bundle.masses > 1e-12, 1.0 / safe_m, 0.0)

    def get_bundle_bond_index(self, force_idx: int, bond_idx: int) -> int | None:
        """Get the MolecularBundle bond index for an OpenMM bond."""
        mapping = self.system_metadata.get("omm_forces_bond_to_bundle_idx", {})
        return mapping.get(force_idx, {}).get(bond_idx)

    def get_bundle_angle_index(self, force_idx: int, angle_idx: int) -> int | None:
        """Get the MolecularBundle angle index for an OpenMM angle."""
        mapping = self.system_metadata.get("omm_forces_angle_to_bundle_idx", {})
        return mapping.get(force_idx, {}).get(angle_idx)

    def get_bundle_torsion_index(self, force_idx: int, torsion_idx: int) -> int | None:
        """Get the MolecularBundle dihedral index for an OpenMM torsion."""
        mapping = self.system_metadata.get("omm_forces_torsion_to_bundle_idx", {})
        return mapping.get(force_idx, {}).get(torsion_idx)

    def get_bundle_exception_index(self, exception_idx: int) -> int | None:
        """Get the MolecularBundle 1-4 exception index for an OpenMM exception."""
        mapping = self.system_metadata.get("omm_exception_to_bundle_idx", {})
        return mapping.get(exception_idx)


# =============================================================================
# 5. Extraction Helpers
# =============================================================================


def _pad_1d(
    arr: np.ndarray | Array,
    target_len: int,
    val: float = 0.0,
    dtype: Any = jnp.float32,
) -> Array:
    """Pad a 1D array to target_len with constant value."""
    n = int(arr.shape[0])
    if n >= target_len:
        return jnp.asarray(arr[:target_len], dtype=dtype)
    pad_len = target_len - n
    return jnp.pad(
        jnp.asarray(arr, dtype=dtype), (0, pad_len), constant_values=val
    )


def _pad_2d(
    arr: np.ndarray | Array,
    target_len: int,
    cols: int,
    val: float = 0,
    dtype: Any = jnp.int32,
) -> Array:
    """Pad a 2D array of shape (N, cols) to (target_len, cols)."""
    n = int(arr.shape[0]) if arr.ndim == 2 else 0
    if n >= target_len:
        return jnp.asarray(arr[:target_len, :cols], dtype=dtype)
    pad_len = target_len - n
    if n == 0:
        return jnp.full((target_len, cols), val, dtype=dtype)
    return jnp.pad(
        jnp.asarray(arr, dtype=dtype),
        ((0, pad_len), (0, 0)),
        constant_values=val,
    )


def _mask(real_count: int, target_len: int) -> Array:
    """Create a boolean validity mask of shape (target_len,)."""
    return jnp.arange(target_len) < real_count


def _next_bucket(count: int, ladder: tuple[int, ...], use_buckets: bool = True) -> int:
    """Determine padded capacity based on ladder thresholds or exact count."""
    if not use_buckets:
        return max(count, 1)
    return ladder[_bucket_idx(count, ladder)]


def _extract_masses_and_particles(
    omm_system: openmm.System,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Extract particle count, masses (Daltons), virtual site mask, and frozen atom mask."""
    n = omm_system.getNumParticles()
    if n <= 0:
        raise ValueError("Cannot convert an openmm.System with 0 particles.")

    masses_list = [
        omm_system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n)
    ]
    is_vs_list = [omm_system.isVirtualSite(i) for i in range(n)]

    masses = np.asarray(masses_list, dtype=np.float32)
    is_vs = np.asarray(is_vs_list, dtype=bool)
    is_frozen = (masses == 0.0) & (~is_vs)

    return n, masses, is_vs, is_frozen


def _extract_box_vectors(
    omm_system: openmm.System,
) -> tuple[np.ndarray, bool, bool, str]:
    """Extract and validate periodic box vectors into Prolix AKMA representation."""
    uses_pbc = omm_system.usesPeriodicBoundaryConditions()
    raw_bv = omm_system.getDefaultPeriodicBoxVectors()

    a_vec = (
        np.array([
            raw_bv[0][0].value_in_unit(unit.nanometer),
            raw_bv[0][1].value_in_unit(unit.nanometer),
            raw_bv[0][2].value_in_unit(unit.nanometer),
        ])
        * NM_TO_ANGSTROM
    )
    b_vec = (
        np.array([
            raw_bv[1][0].value_in_unit(unit.nanometer),
            raw_bv[1][1].value_in_unit(unit.nanometer),
            raw_bv[1][2].value_in_unit(unit.nanometer),
        ])
        * NM_TO_ANGSTROM
    )
    c_vec = (
        np.array([
            raw_bv[2][0].value_in_unit(unit.nanometer),
            raw_bv[2][1].value_in_unit(unit.nanometer),
            raw_bv[2][2].value_in_unit(unit.nanometer),
        ])
        * NM_TO_ANGSTROM
    )

    box_np = np.stack([a_vec, b_vec, c_vec])
    volume = np.linalg.det(box_np)

    if not uses_pbc or abs(volume) < 1e-6 or np.all(box_np == 0.0):
        return np.zeros((3, 3), dtype=np.float32), False, False, "free"

    if (
        abs(box_np[0, 1]) > 1e-6
        or abs(box_np[0, 2]) > 1e-6
        or abs(box_np[1, 2]) > 1e-6
    ):
        raise ValueError(
            "Periodic box vectors do not satisfy OpenMM lower-triangular canonical orientation "
            "(a_y = a_z = b_z = 0)."
        )

    box_np[0, 1] = 0.0
    box_np[0, 2] = 0.0
    box_np[1, 2] = 0.0

    box_reduced = np.asarray(
        reduce_triclinic_box_vectors(jnp.asarray(box_np, dtype=jnp.float32))
    )
    off_diag = np.max(np.abs(box_reduced - np.diag(np.diag(box_reduced))))
    is_triclinic = bool(off_diag > 1e-6)

    return box_reduced.astype(np.float32), True, is_triclinic, "periodic"


def _extract_virtual_sites(
    omm_system: openmm.System,
    n_particles: int,
) -> VirtualSites | None:
    """Extract all virtual sites and partition into topological generations."""
    vs_indices_set = {i for i in range(n_particles) if omm_system.isVirtualSite(i)}
    if not vs_indices_set:
        return None

    # Step 1: Assign depth to particles and partition into generations
    depths: dict[int, int] = {}
    for i in range(n_particles):
        if i not in vs_indices_set:
            depths[i] = 0  # Real particles have depth 0

    remaining = set(vs_indices_set)
    generations_indices: list[list[int]] = []

    while remaining:
        can_compute = []
        for idx in sorted(remaining):
            vs = omm_system.getVirtualSite(idx)
            parents = [vs.getParticle(p) for p in range(vs.getNumParticles())]
            if all(p in depths for p in parents):
                can_compute.append(idx)
        if not can_compute:
            raise ValueError("Virtual site definitions are circular.")
        generations_indices.append(can_compute)
        current_depth = len(generations_indices)
        for idx in can_compute:
            depths[idx] = current_depth
            remaining.remove(idx)

    # Step 2: Build VirtualSiteGeneration objects
    generations_list: list[VirtualSiteGeneration] = []
    total_order: list[int] = []
    total_linear = 0
    total_local = 0

    for gen_indices in generations_indices:
        lin_idx_list: list[list[int]] = []
        lin_prm_list: list[list[float]] = []
        loc_idx_list: list[list[int]] = []
        loc_prm_list: list[list[float]] = []

        for idx in gen_indices:
            total_order.append(idx)
            vs = omm_system.getVirtualSite(idx)
            if isinstance(vs, openmm.TwoParticleAverageSite):
                p1 = vs.getParticle(0)
                p2 = vs.getParticle(1)
                w1 = vs.getWeight(0)
                w2 = vs.getWeight(1)
                lin_idx_list.append([idx, p1, p2, p2])
                lin_prm_list.append([w1, w2, 0.0, 0.0])

            elif isinstance(vs, openmm.ThreeParticleAverageSite):
                p1 = vs.getParticle(0)
                p2 = vs.getParticle(1)
                p3 = vs.getParticle(2)
                w1 = vs.getWeight(0)
                w2 = vs.getWeight(1)
                w3 = vs.getWeight(2)
                lin_idx_list.append([idx, p1, p2, p3])
                lin_prm_list.append([w1, w2, w3, 0.0])

            elif isinstance(vs, openmm.OutOfPlaneSite):
                p1 = vs.getParticle(0)
                p2 = vs.getParticle(1)
                p3 = vs.getParticle(2)
                w12 = vs.getWeight12()
                w13 = vs.getWeight13()
                w_cross_omm = vs.getWeightCross()
                w_cross_akma = 0.1 * w_cross_omm  # nm^-1 to Å^-1
                w1 = 1.0 - w12 - w13
                w2 = w12
                w3 = w13
                lin_idx_list.append([idx, p1, p2, p3])
                lin_prm_list.append([w1, w2, w3, w_cross_akma])

            elif isinstance(vs, openmm.LocalCoordinatesSite):
                num_parents = vs.getNumParticles()
                if num_parents != 3:
                    raise NotImplementedError(
                        f"LocalCoordinatesSite supports 3 parent particles, got {num_parents}."
                    )
                p1 = vs.getParticle(0)
                p2 = vs.getParticle(1)
                p3 = vs.getParticle(2)
                wo = vs.getOriginWeights()
                wx = vs.getXWeights()
                wy = vs.getYWeights()
                lp = vs.getLocalPosition().value_in_unit(unit.nanometer)
                lp_akma = [lp[0] * NM_TO_ANGSTROM, lp[1] * NM_TO_ANGSTROM, lp[2] * NM_TO_ANGSTROM]

                loc_idx_list.append([idx, p1, p2, p3])
                loc_prm_list.append([
                    lp_akma[0],
                    lp_akma[1],
                    lp_akma[2],
                    wo[0],
                    wo[1],
                    wo[2],
                    wx[0],
                    wx[1],
                    wx[2],
                    wy[0],
                    wy[1],
                    wy[2],
                ])
            else:
                raise UnsupportedOpenMMForceError(
                    type(vs).__name__,
                    f"Virtual site type {type(vs).__name__} is not supported.",
                )

        lin_idx = (
            jnp.asarray(lin_idx_list, dtype=jnp.int32)
            if lin_idx_list
            else jnp.zeros((0, 4), dtype=jnp.int32)
        )
        lin_prm = (
            jnp.asarray(lin_prm_list, dtype=jnp.float32)
            if lin_prm_list
            else jnp.zeros((0, 4), dtype=jnp.float32)
        )
        loc_idx = (
            jnp.asarray(loc_idx_list, dtype=jnp.int32)
            if loc_idx_list
            else jnp.zeros((0, 4), dtype=jnp.int32)
        )
        loc_prm = (
            jnp.asarray(loc_prm_list, dtype=jnp.float32)
            if loc_prm_list
            else jnp.zeros((0, 12), dtype=jnp.float32)
        )

        gen_linear = int(lin_idx.shape[0])
        gen_local = int(loc_idx.shape[0])
        total_linear += gen_linear
        total_local += gen_local

        generations_list.append(
            VirtualSiteGeneration(
                linear_indices=lin_idx,
                linear_params=lin_prm,
                local_indices=loc_idx,
                local_params=loc_prm,
                n_linear=gen_linear,
                n_local=gen_local,
            )
        )

    real_indices = jnp.asarray(
        [i for i in range(n_particles) if i not in vs_indices_set],
        dtype=jnp.int32,
    )

    return VirtualSites(
        generations=tuple(generations_list),
        order=jnp.asarray(total_order, dtype=jnp.int32),
        real_particle_indices=real_indices,
        n_generations=len(generations_list),
        n_linear=total_linear,
        n_local=total_local,
        n_total=len(total_order),
        n_particles=n_particles,
    )


def _partition_constraints(
    omm_system: openmm.System,
    masses: np.ndarray,
    config: ProlixOpenMMConfig,
) -> tuple[
    np.ndarray,
    float | None,
    float | None,
    set[tuple[int, int]],
    set[tuple[int, int, int]],
    np.ndarray,
    np.ndarray,
]:
    """Partition OpenMM constraints into rigid water SETTLE triangles and solute SHAKE."""
    n_particles = omm_system.getNumParticles()
    n_constraints = omm_system.getNumConstraints()

    if n_constraints == 0 or config.water_constraint_mode == "none":
        solute_pairs = []
        solute_lengths = []
        for i in range(n_constraints):
            p1, p2, d = omm_system.getConstraintParameters(i)
            solute_pairs.append((min(p1, p2), max(p1, p2)))
            solute_lengths.append(d.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM)

        return (
            np.zeros((0, 3), dtype=np.int32),
            None,
            None,
            set(),
            set(),
            np.asarray(solute_pairs, dtype=np.int32).reshape(-1, 2)
            if solute_pairs
            else np.zeros((0, 2), dtype=np.int32),
            np.asarray(solute_lengths, dtype=np.float32)
            if solute_lengths
            else np.zeros((0,), dtype=np.float32),
        )

    # Precompute bonded adjacency across all bond forces and angle map across angle forces
    bonded_adj: dict[int, set[int]] = collections.defaultdict(set)
    angle_triplets_map: dict[tuple[int, int, int], float] = {}
    for f_idx in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_idx)
        if isinstance(f, openmm.HarmonicBondForce):
            for b_i in range(f.getNumBonds()):
                p1, p2, _, _ = f.getBondParameters(b_i)
                bonded_adj[p1].add(p2)
                bonded_adj[p2].add(p1)
        elif isinstance(f, openmm.CustomBondForce):
            for b_i in range(f.getNumBonds()):
                p1, p2, _ = f.getBondParameters(b_i)
                bonded_adj[p1].add(p2)
                bonded_adj[p2].add(p1)
        elif isinstance(f, openmm.HarmonicAngleForce):
            for a_i in range(f.getNumAngles()):
                p1, p2, p3, th0, _ = f.getAngleParameters(a_i)
                th0_rad = th0.value_in_unit(unit.radian)
                angle_triplets_map[(p1, p2, p3)] = th0_rad
                angle_triplets_map[(p3, p2, p1)] = th0_rad

    # 1. Build constraint adjacency graph
    adj: dict[int, dict[int, float]] = collections.defaultdict(dict)
    raw_constraints: list[tuple[int, int, float]] = []

    for i in range(n_constraints):
        p1, p2, d = omm_system.getConstraintParameters(i)
        d_A = d.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
        u, v = min(p1, p2), max(p1, p2)
        adj[u][v] = d_A
        adj[v][u] = d_A
        raw_constraints.append((u, v, d_A))

    water_indices_list: list[list[int]] = []
    water_r_OH_list: list[float] = []
    water_r_HH_list: list[float] = []
    consumed_edges: set[tuple[int, int]] = set()
    water_bond_pairs: set[tuple[int, int]] = set()
    water_angle_triplets: set[tuple[int, int, int]] = set()

    # 2. Detect 3-cycles (triangles) with 1 heavy atom and 2 light atoms
    visited_nodes: set[int] = set()
    for u in range(n_particles):
        if u in visited_nodes:
            continue
        neighbors = [v for v in adj[u] if v not in visited_nodes]
        found_triangle = False
        for i_idx, v in enumerate(neighbors):
            if found_triangle:
                break
            for w in neighbors[i_idx + 1:]:
                if w in adj[v]:
                    tri = [u, v, w]
                    m_tri = [masses[idx] for idx in tri]

                    # Oxygen ~16 Da (11.5..22 to cover HMR & 18O); Hydrogen ~1 Da (0.5..3.8)
                    oxy_cand = [
                        idx
                        for idx, m in zip(tri, m_tri, strict=True)
                        if 11.5 <= m <= 22.0
                    ]
                    hyd_cand = [
                        idx
                        for idx, m in zip(tri, m_tri, strict=True)
                        if 0.5 <= m <= 3.8
                    ]

                    if len(oxy_cand) == 1 and len(hyd_cand) == 2:
                        oxy = oxy_cand[0]
                        h1, h2 = hyd_cand[0], hyd_cand[1]

                        r_oh1 = adj[oxy][h1]
                        r_oh2 = adj[oxy][h2]
                        r_hh = adj[h1][h2]

                        # Geometric filter: water bond lengths and H-H distance
                        # Covers OPC (r_OH~0.87), TIP3P/TIP4P (0.957), SPC/E (1.00)
                        if not (
                            0.85 <= r_oh1 <= 1.05
                            and 0.85 <= r_oh2 <= 1.05
                            and 1.30 <= r_hh <= 1.70
                        ):
                            continue

                        # Chemical isolation filter: candidate water atoms must not have
                        # external covalent bonds or external constraints to other atoms
                        cand_set = {oxy, h1, h2}
                        if (
                            (bonded_adj[oxy] - cand_set)
                            or (bonded_adj[h1] - cand_set)
                            or (bonded_adj[h2] - cand_set)
                        ):
                            continue
                        if (
                            (set(adj[oxy].keys()) - cand_set)
                            or (set(adj[h1].keys()) - cand_set)
                            or (set(adj[h2].keys()) - cand_set)
                        ):
                            continue

                        # Strictly place Oxygen at column 0
                        water_indices_list.append([oxy, h1, h2])
                        water_r_OH_list.append((r_oh1 + r_oh2) / 2.0)
                        water_r_HH_list.append(r_hh)

                        e1 = (min(oxy, h1), max(oxy, h1))
                        e2 = (min(oxy, h2), max(oxy, h2))
                        e3 = (min(h1, h2), max(h1, h2))
                        consumed_edges.update([e1, e2, e3])

                        water_bond_pairs.update([e1, e2])
                        water_angle_triplets.add((h1, oxy, h2))
                        water_angle_triplets.add((h2, oxy, h1))

                        visited_nodes.update([oxy, h1, h2])
                        found_triangle = True
                        break

    # 3. Fallback: 2-constraint water molecules (when H-H constraint is omitted)
    for u in range(n_particles):
        if u in visited_nodes:
            continue
        if len(adj[u]) == 2:
            h_nodes = list(adj[u].keys())
            if all(0.5 <= masses[h] <= 3.8 and len(adj[h]) == 1 for h in h_nodes):
                h1, h2 = h_nodes[0], h_nodes[1]
                cand_set = {u, h1, h2}

                # Chemical isolation check: must not be bonded or constrained to other atoms
                if (
                    (bonded_adj[u] - cand_set)
                    or (bonded_adj[h1] - cand_set)
                    or (bonded_adj[h2] - cand_set)
                ):
                    continue
                if set(adj[u].keys()) - cand_set:
                    continue

                r_oh1 = adj[u][h1]
                r_oh2 = adj[u][h2]
                r_oh = (r_oh1 + r_oh2) / 2.0
                m_u = masses[u]

                # Mass and bond length filter:
                # - Standard water: m in [15.5, 19.0] and r_oh in [0.85, 1.05] Å (excludes C=12.01, N=14.01)
                # - HMR water: m in [11.5, 15.5] and r_oh in [0.85, 1.03] Å (excludes aliphatic C-H=1.09 Å)
                is_valid_water = False
                if 15.5 <= m_u <= 19.0 and 0.85 <= r_oh <= 1.05:
                    is_valid_water = True
                elif 11.5 <= m_u < 15.5 and 0.85 <= r_oh <= 1.03:
                    is_valid_water = True

                if is_valid_water:
                    # Check angle in HarmonicAngleForce if defined
                    th_rad = angle_triplets_map.get((h1, u, h2), math.radians(104.52))
                    th_deg = math.degrees(th_rad)
                    if not (95.0 <= th_deg <= 115.0):
                        continue

                    r_hh = 2.0 * r_oh * math.sin(th_rad / 2.0)
                    if 1.30 <= r_hh <= 1.70:
                        water_indices_list.append([u, h1, h2])
                        water_r_OH_list.append(r_oh)
                        water_r_HH_list.append(r_hh)

                        e1 = (min(u, h1), max(u, h1))
                        e2 = (min(u, h2), max(u, h2))
                        consumed_edges.update([e1, e2])
                        water_bond_pairs.update([e1, e2])
                        water_angle_triplets.add((h1, u, h2))
                        water_angle_triplets.add((h2, u, h1))
                        visited_nodes.update([u, h1, h2])

    # 4. Solute constraints are remaining unconsumed constraints
    solute_pairs = []
    solute_lengths = []
    if getattr(config, "solute_constraint_mode", "shake") != "none":
        for u, v, d_A in raw_constraints:
            if (u, v) not in consumed_edges:
                solute_pairs.append((u, v))
                solute_lengths.append(d_A)

    w_arr = (
        np.asarray(water_indices_list, dtype=np.int32).reshape(-1, 3)
        if water_indices_list
        else np.zeros((0, 3), dtype=np.int32)
    )
    avg_r_OH = float(np.mean(water_r_OH_list)) if water_r_OH_list else None
    avg_r_HH = float(np.mean(water_r_HH_list)) if water_r_HH_list else None

    c_pairs = (
        np.asarray(solute_pairs, dtype=np.int32).reshape(-1, 2)
        if solute_pairs
        else np.zeros((0, 2), dtype=np.int32)
    )
    c_lengths = (
        np.asarray(solute_lengths, dtype=np.float32)
        if solute_lengths
        else np.zeros((0,), dtype=np.float32)
    )

    return (
        w_arr,
        avg_r_OH,
        avg_r_HH,
        water_bond_pairs,
        water_angle_triplets,
        c_pairs,
        c_lengths,
    )


# =============================================================================
# 6. Main Converter Function
# =============================================================================


def from_openmm_system(
    omm_system: openmm.System,
    integrator: openmm.Integrator | None = None,
    config: ProlixOpenMMConfig | None = None,
    positions: Any | None = None,
) -> ConvertedOpenMMSystem:
    """Convert an OpenMM System into a Prolix ConvertedOpenMMSystem.

    Args:
        omm_system: An openmm.System instance.
        integrator: Optional openmm.Integrator to inspect for simulation defaults.
        config: Prolix configuration instance; defaults to ProlixOpenMMConfig().
        positions: Optional coordinates (OpenMM Quantity or NumPy array) to populate.

    Returns:
        ConvertedOpenMMSystem containing MolecularBundle, custom forces, and metadata.
    """
    if config is None:
        config = ProlixOpenMMConfig()

    use_size_buckets = config.use_size_buckets

    # --- Step 1: Particles & Masses ---
    n_particles, masses_np, is_vs_np, _is_frozen_np = _extract_masses_and_particles(
        omm_system
    )

    # --- Step 2: Periodic Box Vectors ---
    box_mat_np, has_pbc, is_triclinic, boundary_cond = _extract_box_vectors(omm_system)

    # --- Step 3: Virtual Sites ---
    virtual_sites = _extract_virtual_sites(omm_system, n_particles)

    # --- Step 4: Constraints Partitioning (SETTLE vs SHAKE) ---
    (
        water_indices_np,
        water_r_OH,
        water_r_HH,
        water_bond_pairs,
        water_angle_triplets,
        solute_pairs_np,
        solute_lengths_np,
    ) = _partition_constraints(omm_system, masses_np, config)

    n_waters = int(water_indices_np.shape[0])
    n_solute_constraints = int(solute_pairs_np.shape[0])

    # --- Step 5: Fail-Fast & Force Cataloging ---
    supported_force_types = {
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "CMAPTorsionForce",
        "RBTorsionForce",
        "NonbondedForce",
        "GBSAOBCForce",
        "CustomGBForce",
        "MonteCarloBarostat",
        "CMMotionRemover",
        "AndersenThermostat",
        "CustomBondForce",
        "CustomAngleForce",
        "CustomTorsionForce",
        "CustomCompoundBondForce",
        "CustomExternalForce",
        "CustomNonbondedForce",
    }

    force_groups: dict[int, list[str]] = collections.defaultdict(list)
    force_group_by_component: dict[str, int] = {}
    global_params: dict[str, float] = {}

    for f_idx in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_idx)
        fname = type(f).__name__

        # Fail-fast check on known unsupported catalogue
        check_unsupported_force(f)
        if fname not in supported_force_types:
            raise UnsupportedOpenMMForceError(
                fname, f"Force '{fname}' is not supported by the Prolix OpenMM converter."
            )

        fg = f.getForceGroup()
        force_groups[fg].append(fname)

        # Collect global parameters defined on forces
        if hasattr(f, "getNumGlobalParameters"):
            for p_idx in range(f.getNumGlobalParameters()):
                p_name = f.getGlobalParameterName(p_idx)
                p_val = f.getGlobalParameterDefaultValue(p_idx)
                global_params[p_name] = float(p_val)

    # --- Step 6: Bonded Forces Extraction with Internal Water Zeroing ---
    bond_indices_list: list[list[int]] = []
    bond_params_list: list[list[float]] = []
    bond_fg_list: list[int] = []
    bundle_bond_counter = 0

    angle_indices_list: list[list[int]] = []
    angle_params_list: list[list[float]] = []
    angle_fg_list: list[int] = []
    bundle_angle_counter = 0

    dih_indices_list: list[list[int]] = []
    dih_params_list: list[list[float]] = []
    dih_fg_list: list[int] = []
    bundle_dihedral_counter = 0

    cmap_torsions_list: list[list[int]] = []
    cmap_map_indices_list: list[int] = []
    cmap_fg_list: list[int] = []
    cmap_grids_list: list[np.ndarray] = []
    cmap_grid_size = 24

    rb_forces: list[CompiledRBTorsionForce] = []
    omm_custom_force_to_compiled_idx: dict[int, int] = {}

    # Count force instances to determine single-force backwards compatibility
    bond_force_indices = [
        idx for idx in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(idx), openmm.HarmonicBondForce)
    ]
    angle_force_indices = [
        idx for idx in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(idx), openmm.HarmonicAngleForce)
    ]
    torsion_force_indices = [
        idx for idx in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(idx), openmm.PeriodicTorsionForce)
    ]

    single_bond_force = len(bond_force_indices) == 1
    single_angle_force = len(angle_force_indices) == 1
    single_torsion_force = len(torsion_force_indices) == 1

    omm_bond_to_bundle_idx = CompoundMapping(
        default_force_idx=bond_force_indices[0] if bond_force_indices else 0,
        is_single=single_bond_force,
    )
    omm_forces_bond_to_bundle_idx: dict[int, dict[int, int | None]] = collections.defaultdict(dict)

    omm_angle_to_bundle_idx = CompoundMapping(
        default_force_idx=angle_force_indices[0] if angle_force_indices else 0,
        is_single=single_angle_force,
    )
    omm_forces_angle_to_bundle_idx: dict[int, dict[int, int | None]] = collections.defaultdict(dict)

    omm_torsion_to_bundle_idx = CompoundMapping(
        default_force_idx=torsion_force_indices[0] if torsion_force_indices else 0,
        is_single=single_torsion_force,
    )
    omm_forces_torsion_to_bundle_idx: dict[int, dict[int, int | None]] = collections.defaultdict(dict)
    omm_exception_to_bundle_idx: dict[int, int | None] = {}

    for f_idx in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_idx)

        if isinstance(f, openmm.HarmonicBondForce):
            force_group_by_component["bond"] = f.getForceGroup()
            for b_i in range(f.getNumBonds()):
                p1, p2, r0, k = f.getBondParameters(b_i)
                pair = (min(p1, p2), max(p1, p2))
                # Internal water bonds handled by SETTLE: zeroed and filtered
                if pair in water_bond_pairs:
                    omm_bond_to_bundle_idx[(f_idx, b_i)] = None
                    omm_forces_bond_to_bundle_idx[f_idx][b_i] = None
                else:
                    r0_A = r0.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                    k_akma = (
                        k.value_in_unit(
                            unit.kilojoules_per_mole / (unit.nanometer**2)
                        )
                        * BOND_K_OMM_TO_AKMA
                    )
                    bond_indices_list.append([p1, p2])
                    bond_params_list.append([r0_A, k_akma])
                    bond_fg_list.append(f.getForceGroup())
                    omm_bond_to_bundle_idx[(f_idx, b_i)] = bundle_bond_counter
                    omm_forces_bond_to_bundle_idx[f_idx][b_i] = bundle_bond_counter
                    bundle_bond_counter += 1

        elif isinstance(f, openmm.HarmonicAngleForce):
            force_group_by_component["angle"] = f.getForceGroup()
            for a_i in range(f.getNumAngles()):
                p1, p2, p3, th0, k = f.getAngleParameters(a_i)
                triplet = (p1, p2, p3)
                # Internal water angles handled by SETTLE: zeroed and filtered
                if triplet in water_angle_triplets:
                    omm_angle_to_bundle_idx[(f_idx, a_i)] = None
                    omm_forces_angle_to_bundle_idx[f_idx][a_i] = None
                else:
                    th0_rad = th0.value_in_unit(unit.radian)
                    k_akma = (
                        k.value_in_unit(unit.kilojoules_per_mole / (unit.radian**2))
                        * ANGLE_K_OMM_TO_AKMA
                    )
                    angle_indices_list.append([p1, p2, p3])
                    angle_params_list.append([th0_rad, k_akma])
                    angle_fg_list.append(f.getForceGroup())
                    omm_angle_to_bundle_idx[(f_idx, a_i)] = bundle_angle_counter
                    omm_forces_angle_to_bundle_idx[f_idx][a_i] = bundle_angle_counter
                    bundle_angle_counter += 1

        elif isinstance(f, openmm.PeriodicTorsionForce):
            force_group_by_component["dihedral"] = f.getForceGroup()
            for t_i in range(f.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = f.getTorsionParameters(t_i)
                phase_rad = phase.value_in_unit(unit.radian)
                k_akma = (
                    k.value_in_unit(unit.kilojoules_per_mole)
                    * TORSION_K_OMM_TO_AKMA
                )
                dih_indices_list.append([p1, p2, p3, p4])
                dih_params_list.append([float(periodicity), phase_rad, k_akma, 0.0])
                dih_fg_list.append(f.getForceGroup())
                omm_torsion_to_bundle_idx[(f_idx, t_i)] = bundle_dihedral_counter
                omm_forces_torsion_to_bundle_idx[f_idx][t_i] = bundle_dihedral_counter
                bundle_dihedral_counter += 1

        elif isinstance(f, openmm.CMAPTorsionForce):
            force_group_by_component["cmap"] = f.getForceGroup()
            for m_i in range(f.getNumMaps()):
                size, energy_flat = f.getMapParameters(m_i)
                cmap_grid_size = size
                energy_arr = (
                    np.asarray(
                        [
                            e.value_in_unit(unit.kilojoules_per_mole)
                            if hasattr(e, "value_in_unit")
                            else float(e)
                            for e in energy_flat
                        ],
                        dtype=np.float32,
                    ).reshape((size, size))
                    * KJ_PER_MOL_TO_KCAL_PER_MOL
                )
                cmap_grids_list.append(energy_arr)

            for t_i in range(f.getNumTorsions()):
                map_idx, a1, a2, a3, a4, b1, b2, b3, b4 = f.getTorsionParameters(t_i)
                cmap_torsions_list.append([a1, a2, a3, a4, b1, b2, b3, b4])
                cmap_map_indices_list.append(map_idx)
                cmap_fg_list.append(f.getForceGroup())

        elif isinstance(f, openmm.RBTorsionForce):
            force_group_by_component["rb_torsion"] = f.getForceGroup()
            rb_t_indices = []
            rb_c_params = []
            for t_i in range(f.getNumTorsions()):
                p1, p2, p3, p4, c0, c1, c2, c3, c4, c5 = f.getTorsionParameters(t_i)
                coeffs = [
                    c.value_in_unit(unit.kilojoules_per_mole)
                    if hasattr(c, "value_in_unit")
                    else float(c)
                    for c in [c0, c1, c2, c3, c4, c5]
                ]
                rb_t_indices.append([p1, p2, p3, p4])
                rb_c_params.append(coeffs)

            rb_force = CompiledRBTorsionForce(
                name="RBTorsionForce",
                force_group=f.getForceGroup(),
                uses_periodic=f.usesPeriodicBoundaryConditions(),
                energy_param_derivatives=(),
                global_param_names=(),
                torsion_indices=jnp.asarray(rb_t_indices, dtype=jnp.int32).reshape(
                    -1, 4
                ),
                c_params=jnp.asarray(rb_c_params, dtype=jnp.float32).reshape(-1, 6),
            )
            omm_custom_force_to_compiled_idx[f_idx] = len(rb_forces)
            rb_forces.append(rb_force)

    bonds_np = (
        np.asarray(bond_indices_list, dtype=np.int32).reshape(-1, 2)
        if bond_indices_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    bond_params_np = (
        np.asarray(bond_params_list, dtype=np.float32).reshape(-1, 2)
        if bond_params_list
        else np.zeros((0, 2), dtype=np.float32)
    )

    angles_np = (
        np.asarray(angle_indices_list, dtype=np.int32).reshape(-1, 3)
        if angle_indices_list
        else np.zeros((0, 3), dtype=np.int32)
    )
    angle_params_np = (
        np.asarray(angle_params_list, dtype=np.float32).reshape(-1, 2)
        if angle_params_list
        else np.zeros((0, 2), dtype=np.float32)
    )

    dih_np = (
        np.asarray(dih_indices_list, dtype=np.int32).reshape(-1, 4)
        if dih_indices_list
        else np.zeros((0, 4), dtype=np.int32)
    )
    dih_params_np = (
        np.asarray(dih_params_list, dtype=np.float32).reshape(-1, 4)
        if dih_params_list
        else np.zeros((0, 4), dtype=np.float32)
    )

    cmap_torsions_np = (
        np.asarray(cmap_torsions_list, dtype=np.int32).reshape(-1, 8)
        if cmap_torsions_list
        else np.zeros((0, 8), dtype=np.int32)
    )
    cmap_map_indices_np = (
        np.asarray(cmap_map_indices_list, dtype=np.int32)
        if cmap_map_indices_list
        else np.zeros((0,), dtype=np.int32)
    )

    n_maps = len(cmap_grids_list)
    if n_maps > 0:
        cmap_grids_np = np.stack(cmap_grids_list, axis=0)
        cmap_coeffs_np = np.asarray(
            precompute_cmap_coefficients(jnp.asarray(cmap_grids_np))
        )
    else:
        cmap_grids_np = np.zeros((1, cmap_grid_size, cmap_grid_size), dtype=np.float32)
        cmap_coeffs_np = np.zeros(
            (1, cmap_grid_size, cmap_grid_size, 16), dtype=np.float32
        )

    # --- Step 7: NonbondedForce Extraction ---
    charges_np = np.zeros(n_particles, dtype=np.float32)
    sigmas_np = np.ones(n_particles, dtype=np.float32)
    epsilons_np = np.zeros(n_particles, dtype=np.float32)

    excl_pairs_list: list[list[int]] = []
    excl_vdw_list: list[float] = []
    excl_elec_list: list[float] = []

    exc14_pairs_list: list[list[int]] = []
    exc14_sigmas_list: list[float] = []
    exc14_eps_list: list[float] = []
    exc14_qprod_list: list[float] = []

    nonbonded_method = "pme" if has_pbc else "coulomb"
    has_nonbonded = False
    nonbonded_cutoff = 9.0  # Å
    lj_switch_width = 0.0  # Å
    use_dispersion_correction = True
    reaction_field_dielectric = 78.3
    pme_alpha = 0.0
    pme_grid_dims: tuple[int, int, int] | None = None

    for f_idx in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_idx)
        if isinstance(f, openmm.NonbondedForce):
            has_nonbonded = True
            force_group_by_component["nonbonded"] = f.getForceGroup()

            # Method validation & mapping
            omm_method = f.getNonbondedMethod()
            if omm_method == 5:  # LJPME
                raise UnsupportedOpenMMForceError(
                    "NonbondedForce (LJPME)",
                    "Lennard-Jones PME dispersion mesh is not implemented in Prolix.",
                )
            if omm_method == 0:  # NoCutoff
                nonbonded_method = "coulomb"
                nonbonded_cutoff = float("inf")
            elif omm_method == 1:  # CutoffNonPeriodic
                nonbonded_method = "cutoff"
                nonbonded_cutoff = (
                    f.getCutoffDistance().value_in_unit(unit.nanometer)
                    * NM_TO_ANGSTROM
                )
            elif omm_method == 2:  # CutoffPeriodic
                nonbonded_method = "reaction_field"
                nonbonded_cutoff = (
                    f.getCutoffDistance().value_in_unit(unit.nanometer)
                    * NM_TO_ANGSTROM
                )
                reaction_field_dielectric = float(f.getReactionFieldDielectric())
            elif omm_method in (3, 4):  # Ewald, PME
                nonbonded_method = "pme"
                nonbonded_cutoff = (
                    f.getCutoffDistance().value_in_unit(unit.nanometer)
                    * NM_TO_ANGSTROM
                )

            # Switching function
            if f.getUseSwitchingFunction():
                sw_dist_A = (
                    f.getSwitchingDistance().value_in_unit(unit.nanometer)
                    * NM_TO_ANGSTROM
                )
                lj_switch_width = max(nonbonded_cutoff - sw_dist_A, 0.0)
            else:
                lj_switch_width = 0.0

            use_dispersion_correction = bool(f.getUseDispersionCorrection())

            # Particle nonbonded parameters
            for p_i in range(f.getNumParticles()):
                q, sig, eps = f.getParticleParameters(p_i)
                charges_np[p_i] = q.value_in_unit(unit.elementary_charge)
                sigmas_np[p_i] = (
                    sig.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                )
                epsilons_np[p_i] = (
                    eps.value_in_unit(unit.kilojoules_per_mole)
                    * KJ_PER_MOL_TO_KCAL_PER_MOL
                )

            # Exceptions & Exclusions
            bundle_exc_counter = 0
            for e_i in range(f.getNumExceptions()):
                p1, p2, qprod, sig, eps = f.getExceptionParameters(e_i)
                p1_idx, p2_idx = min(p1, p2), max(p1, p2)
                qprod_val = qprod.value_in_unit(unit.elementary_charge**2)
                sig_val = sig.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                eps_val = (
                    eps.value_in_unit(unit.kilojoules_per_mole)
                    * KJ_PER_MOL_TO_KCAL_PER_MOL
                )

                # Unified rule: all exclusions/exceptions zeroed out in bulk pair list
                excl_pairs_list.append([p1_idx, p2_idx])
                excl_vdw_list.append(0.0)
                excl_elec_list.append(0.0)

                # Non-zero 1-4 exceptions populated in dedicated exception arrays
                if abs(qprod_val) > 0.0 or abs(eps_val) > 0.0:
                    exc14_pairs_list.append([p1_idx, p2_idx])
                    exc14_sigmas_list.append(sig_val)
                    exc14_eps_list.append(eps_val)
                    exc14_qprod_list.append(qprod_val)
                    omm_exception_to_bundle_idx[e_i] = bundle_exc_counter
                    bundle_exc_counter += 1
                else:
                    omm_exception_to_bundle_idx[e_i] = None

            # PME grid parameters
            alpha_omm, nx, ny, nz = f.getPMEParameters()
            alpha_omm_val = (
                alpha_omm.value_in_unit(unit.nanometer**-1)
                if hasattr(alpha_omm, "value_in_unit")
                else float(alpha_omm)
            )
            if alpha_omm_val > 0.0:
                pme_alpha = alpha_omm_val / 10.0  # nm^-1 to Å^-1
            elif nonbonded_method == "pme":
                pme_alpha = float(
                    solve_ewald_alpha(
                        nonbonded_cutoff, f.getEwaldErrorTolerance()
                    )
                )

            if nx > 0 and ny > 0 and nz > 0:
                pme_grid_dims = (int(nx), int(ny), int(nz))
            elif has_pbc:
                grid_sp = config.pme_grid_spacing
                if (
                    config.ewald_error_tolerance <= 0.0005
                    and abs(config.pme_grid_spacing - 1.0) < 1e-6
                ):
                    grid_sp = 0.375
                pme_grid_dims = compute_pme_grid_dims(
                    jnp.diag(jnp.asarray(box_mat_np)),
                    grid_spacing=grid_sp,
                )

    excl_indices_np = (
        np.asarray(excl_pairs_list, dtype=np.int32).reshape(-1, 2)
        if excl_pairs_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    excl_vdw_np = (
        np.asarray(excl_vdw_list, dtype=np.float32)
        if excl_vdw_list
        else np.zeros((0,), dtype=np.float32)
    )
    excl_elec_np = (
        np.asarray(excl_elec_list, dtype=np.float32)
        if excl_elec_list
        else np.zeros((0,), dtype=np.float32)
    )

    exc14_pairs_np = (
        np.asarray(exc14_pairs_list, dtype=np.int32).reshape(-1, 2)
        if exc14_pairs_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    exc14_sig_np = (
        np.asarray(exc14_sigmas_list, dtype=np.float32)
        if exc14_sigmas_list
        else np.zeros((0,), dtype=np.float32)
    )
    exc14_eps_np = (
        np.asarray(exc14_eps_list, dtype=np.float32)
        if exc14_eps_list
        else np.zeros((0,), dtype=np.float32)
    )
    exc14_qp_np = (
        np.asarray(exc14_qprod_list, dtype=np.float32)
        if exc14_qprod_list
        else np.zeros((0,), dtype=np.float32)
    )

    # --- Step 8: Implicit Solvent (GBSAOBCForce) ---
    radii_np = np.ones(n_particles, dtype=np.float32)
    scaled_radii_np = np.ones(n_particles, dtype=np.float32)
    has_implicit_solvent = False
    solvent_dielectric = 78.5
    solute_dielectric = 1.0

    for f_idx in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_idx)
        if isinstance(f, openmm.GBSAOBCForce):
            has_implicit_solvent = True
            force_group_by_component["gbsa"] = f.getForceGroup()
            solvent_dielectric = float(f.getSolventDielectric())
            solute_dielectric = float(f.getSoluteDielectric())
            for p_i in range(f.getNumParticles()):
                q, r, s = f.getParticleParameters(p_i)
                r_A = r.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                radii_np[p_i] = r_A
                scaled_radii_np[p_i] = max(r_A - 0.09, 0.0) * float(s)
                # If charge not yet set by NonbondedForce, take from GBSA
                if charges_np[p_i] == 0.0:
                    charges_np[p_i] = q.value_in_unit(unit.elementary_charge)
        elif isinstance(f, openmm.CustomGBForce):
            has_implicit_solvent = True
            force_group_by_component["gbsa"] = f.getForceGroup()
            param_names = [
                f.getPerParticleParameterName(p)
                for p in range(f.getNumPerParticleParameters())
            ]
            for p_i in range(f.getNumParticles()):
                p_vals = f.getParticleParameters(p_i)
                p_dict = dict(zip(param_names, p_vals, strict=True))
                if "charge" in p_dict and charges_np[p_i] == 0.0:
                    charges_np[p_i] = float(p_dict["charge"])
                if "or" in p_dict:
                    r_A = float(p_dict["or"]) * NM_TO_ANGSTROM
                    radii_np[p_i] = r_A
                elif "radius" in p_dict:
                    r_A = float(p_dict["radius"]) * NM_TO_ANGSTROM
                    radii_np[p_i] = r_A
                if "sr" in p_dict:
                    sr_A = float(p_dict["sr"]) * NM_TO_ANGSTROM
                    scaled_radii_np[p_i] = sr_A
                elif "scale" in p_dict:
                    scaled_radii_np[p_i] = radii_np[p_i] * float(p_dict["scale"])

    # --- Step 9: Barostat and Center-of-Mass Motion Remover ---
    barostat_info: dict[str, Any] | None = None
    remove_com_motion = False
    com_frequency = 10

    for f_idx in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_idx)
        if isinstance(f, openmm.MonteCarloBarostat):
            p_bar = f.getDefaultPressure().value_in_unit(unit.bar)
            t_k = f.getDefaultTemperature().value_in_unit(unit.kelvin)
            freq = f.getFrequency()
            barostat_info = {
                "type": "MonteCarloBarostat",
                "pressure_bar": float(p_bar),
                "pressure_akma": float(p_bar * AKMA_PRESSURE_PER_BAR),
                "temperature_k": float(t_k),
                "frequency": int(freq),
            }
        elif isinstance(f, openmm.CMMotionRemover):
            remove_com_motion = True
            com_frequency = int(f.getFrequency())

    # --- Step 10: Array Bucketing and Allocation ---
    n_atoms_real = n_particles
    n_bonds_real = int(bonds_np.shape[0])
    n_angles_real = int(angles_np.shape[0])
    n_dihedrals_real = int(dih_np.shape[0])
    n_excl_real = int(excl_indices_np.shape[0])
    n_exc14_real = int(exc14_pairs_np.shape[0])
    n_cmap_real = int(cmap_torsions_np.shape[0])

    a_cap = _next_bucket(n_atoms_real, ATOM_BUCKETS, use_size_buckets)
    b_cap = _next_bucket(max(n_bonds_real, 1), BOND_BUCKETS, use_size_buckets)
    ang_cap = _next_bucket(max(n_angles_real, 1), ANGLE_BUCKETS, use_size_buckets)
    dih_cap = _next_bucket(max(n_dihedrals_real, 1), DIHEDRAL_BUCKETS, use_size_buckets)
    wat_cap = _next_bucket(max(n_waters, 1), WATER_BUCKETS, use_size_buckets)
    excl_cap = _next_bucket(max(n_excl_real, 1), EXCL_BUCKETS, use_size_buckets)
    exc14_cap = _next_bucket(max(n_exc14_real, 1), EXCEPTION_BUCKETS, use_size_buckets)
    cmap_cap = _next_bucket(max(n_cmap_real, 1), CMAP_BUCKETS, use_size_buckets)

    cmap_map_bidx = (
        _bucket_idx(max(n_maps, 1), CMAP_MAP_BUCKETS) if use_size_buckets else 0
    )
    constr_bidx = (
        _bucket_idx(n_solute_constraints, CONSTRAINT_BUCKETS)
        if use_size_buckets
        else 0
    )
    constr_cap = (
        CONSTRAINT_BUCKETS[constr_bidx] if use_size_buckets else n_solute_constraints
    )

    # Dense exclusion map and 32-cap validation
    if n_excl_real > 0:
        (
            excl_dense_idx,
            excl_dense_vdw,
            excl_dense_elec,
        ) = pair_list_to_dense_padded(
            excl_indices_np,
            excl_vdw_np,
            excl_elec_np,
            jnp.ones(n_excl_real, dtype=bool),
            a_cap,
            max_exclusions=_EXCL_DENSE_MAX,
        )
    else:
        excl_dense_idx = jnp.full((a_cap, _EXCL_DENSE_MAX), -1, dtype=jnp.int32)
        excl_dense_vdw = jnp.ones((a_cap, _EXCL_DENSE_MAX), dtype=jnp.float32)
        excl_dense_elec = jnp.ones((a_cap, _EXCL_DENSE_MAX), dtype=jnp.float32)

    # Padded CMAP per-torsion energy grids
    if n_cmap_real > 0 and n_maps > 0:
        safe_m = np.clip(cmap_map_indices_np, 0, max(n_maps - 1, 0))
        per_torsion_grids = cmap_grids_np[safe_m]
        pad_cm = cmap_cap - n_cmap_real
        if pad_cm > 0:
            cmap_grids_padded = np.pad(
                per_torsion_grids,
                ((0, pad_cm), (0, 0), (0, 0)),
                constant_values=0.0,
            )
        else:
            cmap_grids_padded = per_torsion_grids[:cmap_cap]
    else:
        cmap_grids_padded = np.zeros(
            (cmap_cap, cmap_grid_size, cmap_grid_size), dtype=np.float32
        )

    # Initial positions handling
    if positions is not None:
        if hasattr(positions, "value_in_unit"):
            pos_raw = (
                np.asarray(positions.value_in_unit(unit.nanometer))
                * NM_TO_ANGSTROM
            )
        else:
            pos_raw = np.asarray(positions, dtype=np.float32)
    else:
        pos_raw = np.zeros((n_atoms_real, 3), dtype=np.float32)

    pos_padded = _pad_2d(pos_raw, a_cap, 3, val=0.0, dtype=jnp.float32)

    # Masses: real masses in 0..N-1, padding receives 1.0 (prevents div-by-zero)
    masses_padded = jnp.pad(
        jnp.asarray(masses_np, dtype=jnp.float32),
        (0, a_cap - n_atoms_real),
        constant_values=1.0,
    )

    radii_padded = jnp.pad(
        jnp.asarray(radii_np, dtype=jnp.float32),
        (0, a_cap - n_atoms_real),
        constant_values=1.0,
    )
    scaled_radii_padded = jnp.pad(
        jnp.asarray(scaled_radii_np, dtype=jnp.float32),
        (0, a_cap - n_atoms_real),
        constant_values=1.0,
    )

    # Construct ShapeSpec
    shape_spec = MolecularShapeSpec(
        atom_bucket_idx=_bucket_idx(n_atoms_real, ATOM_BUCKETS)
        if use_size_buckets
        else 0,
        bond_bucket_idx=_bucket_idx(max(n_bonds_real, 1), BOND_BUCKETS)
        if use_size_buckets
        else 0,
        angle_bucket_idx=_bucket_idx(max(n_angles_real, 1), ANGLE_BUCKETS)
        if use_size_buckets
        else 0,
        dihedral_bucket_idx=_bucket_idx(max(n_dihedrals_real, 1), DIHEDRAL_BUCKETS)
        if use_size_buckets
        else 0,
        water_bucket_idx=_bucket_idx(max(n_waters, 1), WATER_BUCKETS)
        if use_size_buckets
        else 0,
        excl_bucket_idx=_bucket_idx(max(n_excl_real, 1), EXCL_BUCKETS)
        if use_size_buckets
        else 0,
        cmap_bucket_idx=_bucket_idx(max(n_cmap_real, 1), CMAP_BUCKETS)
        if use_size_buckets
        else 0,
        exception_bucket_idx=_bucket_idx(max(n_exc14_real, 1), EXCEPTION_BUCKETS)
        if use_size_buckets
        else 0,
        has_pbc=has_pbc and boundary_cond == "periodic",
        has_implicit_solvent=has_implicit_solvent,
        boundary_condition=boundary_cond,
        has_real_water=n_waters > 0,
        use_size_buckets=use_size_buckets,
        exact_n_atoms=0 if use_size_buckets else a_cap,
        exact_n_waters=0 if use_size_buckets else wat_cap,
        cmap_map_bucket_idx=cmap_map_bidx,
        constraint_bucket_idx=constr_bidx,
        has_real_constraints=n_solute_constraints > 0,
        is_triclinic=is_triclinic,
    )

    # Assemble MolecularBundle
    bundle = MolecularBundle(
        positions=pos_padded,
        masses=masses_padded,
        charges=_pad_1d(charges_np, a_cap, dtype=jnp.float32),
        sigmas=_pad_1d(sigmas_np, a_cap, dtype=jnp.float32),
        epsilons=_pad_1d(epsilons_np, a_cap, dtype=jnp.float32),
        radii=radii_padded,
        scaled_radii=scaled_radii_padded,
        atom_mask=_mask(n_atoms_real, a_cap),
        n_atoms=jnp.array(n_atoms_real, dtype=jnp.int32),
        box=jnp.asarray(box_mat_np, dtype=jnp.float32),
        bond_idx=_pad_2d(bonds_np, b_cap, 2, dtype=jnp.int32),
        bond_params=_pad_2d(bond_params_np, b_cap, 2, dtype=jnp.float32),
        bond_mask=_mask(n_bonds_real, b_cap),
        n_bonds=jnp.array(n_bonds_real, dtype=jnp.int32),
        angle_idx=_pad_2d(angles_np, ang_cap, 3, dtype=jnp.int32),
        angle_params=_pad_2d(angle_params_np, ang_cap, 2, dtype=jnp.float32),
        angle_mask=_mask(n_angles_real, ang_cap),
        n_angles=jnp.array(n_angles_real, dtype=jnp.int32),
        dihedral_idx=_pad_2d(dih_np, dih_cap, 4, dtype=jnp.int32),
        dihedral_params=_pad_2d(dih_params_np, dih_cap, 4, dtype=jnp.float32),
        dihedral_mask=_mask(n_dihedrals_real, dih_cap),
        n_dihedrals=jnp.array(n_dihedrals_real, dtype=jnp.int32),
        improper_idx=_pad_2d(np.zeros((0, 4)), dih_cap, 4, dtype=jnp.int32),
        improper_params=_pad_2d(np.zeros((0, 3)), dih_cap, 3, dtype=jnp.float32),
        improper_mask=_mask(0, dih_cap),
        improper_is_periodic=jnp.array(False, dtype=bool),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=_pad_2d(np.zeros((0, 3)), ang_cap, 3, dtype=jnp.int32),
        urey_bradley_params=_pad_2d(np.zeros((0, 2)), ang_cap, 2, dtype=jnp.float32),
        urey_bradley_mask=_mask(0, ang_cap),
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=_pad_2d(cmap_torsions_np, cmap_cap, 8, dtype=jnp.int32),
        cmap_energy_grids=jnp.asarray(cmap_grids_padded, dtype=jnp.float32),
        cmap_mask=_mask(n_cmap_real, cmap_cap),
        n_cmap=jnp.array(n_cmap_real, dtype=jnp.int32),
        water_indices=_pad_2d(water_indices_np, wat_cap, 3, dtype=jnp.int32),
        water_mask=_mask(n_waters, wat_cap),
        n_waters=jnp.array(n_waters, dtype=jnp.int32),
        excl_indices=_pad_2d(excl_indices_np, excl_cap, 2, dtype=jnp.int32),
        excl_scales_vdw=_pad_1d(excl_vdw_np, excl_cap, dtype=jnp.float32),
        excl_scales_elec=_pad_1d(excl_elec_np, excl_cap, dtype=jnp.float32),
        excl_mask=_mask(n_excl_real, excl_cap),
        n_excl=jnp.array(n_excl_real, dtype=jnp.int32),
        exception_pairs=_pad_2d(exc14_pairs_np, exc14_cap, 2, dtype=jnp.int32),
        exception_sigmas=_pad_1d(exc14_sig_np, exc14_cap, dtype=jnp.float32),
        exception_epsilons=_pad_1d(exc14_eps_np, exc14_cap, dtype=jnp.float32),
        exception_chargeprods=_pad_1d(exc14_qp_np, exc14_cap, dtype=jnp.float32),
        exception_mask=_mask(n_exc14_real, exc14_cap),
        n_exception_pairs=jnp.array(n_exc14_real, dtype=jnp.int32),
        pme_alpha=jnp.array(float(pme_alpha), dtype=jnp.float32),
        cutoff_distance=jnp.array(float(nonbonded_cutoff), dtype=jnp.float32),
        shape_spec=shape_spec,
        excl_dense_indices=excl_dense_idx,
        excl_dense_scales_vdw=excl_dense_vdw,
        excl_dense_scales_elec=excl_dense_elec,
        cmap_indices=_pad_1d(cmap_map_indices_np, cmap_cap, dtype=jnp.int32),
        cmap_coeffs=jnp.asarray(cmap_coeffs_np, dtype=jnp.float32),
        constraint_pairs=_pad_2d(
            solute_pairs_np, constr_cap, 2, dtype=jnp.int32
        ),
        constraint_lengths=_pad_1d(
            solute_lengths_np, constr_cap, dtype=jnp.float32
        ),
        constraint_mask=_mask(n_solute_constraints, constr_cap),
        n_constraints=jnp.array(n_solute_constraints, dtype=jnp.int32),
        water_r_OH=jnp.array(
            float(water_r_OH if water_r_OH is not None else _DEFAULT_TIP3P_ROH),
            dtype=jnp.float32,
        ),
        water_r_HH=jnp.array(
            float(water_r_HH if water_r_HH is not None else _DEFAULT_TIP3P_RHH),
            dtype=jnp.float32,
        ),
    )

    # --- Step 11: Custom Forces & Pattern Matching ---
    custom_forces: list[Any] = list(rb_forces)
    position_restraints: list[PositionRestraints] = []

    for f_idx in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_idx)
        fname = type(f).__name__

        if fname.startswith("Custom") and fname != "CustomGBForce":
            # Lepton-only mode: compile directly without pattern matching
            if config.custom_force_mode == "lepton_only":
                compiled_cf = compile_custom_force(f, config)
                omm_custom_force_to_compiled_idx[f_idx] = len(custom_forces)
                custom_forces.append(compiled_cf)
                continue

            # Auto or pattern-only: attempt pattern matching
            matched = match_custom_force_pattern(f)

            if matched is not None:
                if matched.target_term == "position_restraint":
                    # In pattern_matcher.py: params has columns [x0, y0, z0, k]
                    x0 = matched.params[:, 0]
                    y0 = matched.params[:, 1]
                    z0 = matched.params[:, 2]
                    k = matched.params[:, 3]
                    idx_arr = matched.indices.flatten()
                    pr = PositionRestraints(
                        indices=jnp.asarray(idx_arr, dtype=jnp.int32),
                        k_x=jnp.asarray(k, dtype=jnp.float32),
                        k_y=jnp.asarray(k, dtype=jnp.float32),
                        k_z=jnp.asarray(k, dtype=jnp.float32),
                        x0=jnp.asarray(x0, dtype=jnp.float32),
                        y0=jnp.asarray(y0, dtype=jnp.float32),
                        z0=jnp.asarray(z0, dtype=jnp.float32),
                        force_group=f.getForceGroup(),
                    )
                    position_restraints.append(pr)
                else:
                    # If force group is 0 (or matches component group), fold into bundle
                    target_group = force_group_by_component.get(
                        matched.target_term, 0
                    )
                    if f.getForceGroup() == target_group:
                        bundle = fold_matched_pattern_into_bundle(bundle, matched)
                    else:
                        # Non-default force group must remain isolated for decomposed evaluation
                        compiled_cf = compile_custom_force(f, config)
                        omm_custom_force_to_compiled_idx[f_idx] = len(custom_forces)
                        custom_forces.append(compiled_cf)
            elif config.custom_force_mode == "pattern_only":
                raise UnsupportedOpenMMForceError(
                    fname,
                    f"Custom force '{fname}' could not be matched to a native pattern "
                    f"in custom_force_mode='pattern_only'.",
                )
            else:
                # Fallback to Lepton compilation
                compiled_cf = compile_custom_force(f, config)
                omm_custom_force_to_compiled_idx[f_idx] = len(custom_forces)
                custom_forces.append(compiled_cf)

    # --- Step 12: Package Metadata ---
    system_metadata: dict[str, Any] = {
        "n_particles": n_atoms_real,
        "n_virtual_sites": virtual_sites.n_total if virtual_sites else 0,
        "has_pbc": has_pbc,
        "is_triclinic": is_triclinic,
        "boundary_condition": boundary_cond,
        "n_waters": n_waters,
        "water_r_OH_akma": water_r_OH,
        "water_r_HH_akma": water_r_HH,
        "n_solute_constraints": n_solute_constraints,
        "barostat": barostat_info,
        "remove_com_motion": remove_com_motion,
        "com_motion_frequency": com_frequency,
        "omm_bond_to_bundle_idx": omm_bond_to_bundle_idx,
        "omm_angle_to_bundle_idx": omm_angle_to_bundle_idx,
        "omm_torsion_to_bundle_idx": omm_torsion_to_bundle_idx,
        "omm_forces_bond_to_bundle_idx": dict(omm_forces_bond_to_bundle_idx),
        "omm_forces_angle_to_bundle_idx": dict(omm_forces_angle_to_bundle_idx),
        "omm_forces_torsion_to_bundle_idx": dict(omm_forces_torsion_to_bundle_idx),
        "omm_exception_to_bundle_idx": omm_exception_to_bundle_idx,
        "omm_custom_force_to_compiled_idx": dict(omm_custom_force_to_compiled_idx),
    }

    if integrator is not None:
        system_metadata["integrator_type"] = type(integrator).__name__
        if hasattr(integrator, "getStepSize"):
            system_metadata["step_size_ps"] = integrator.getStepSize().value_in_unit(
                unit.picosecond
            )
        if hasattr(integrator, "getTemperature"):
            system_metadata["temperature_k"] = integrator.getTemperature().value_in_unit(
                unit.kelvin
            )
        if hasattr(integrator, "getFriction"):
            system_metadata["friction_ps_inv"] = integrator.getFriction().value_in_unit(
                unit.picosecond**-1
            )

    bond_fg_padded = _pad_1d(
        np.asarray(bond_fg_list, dtype=np.int32)
        if bond_fg_list
        else np.zeros((0,), dtype=np.int32),
        b_cap,
        dtype=jnp.int32,
    )
    angle_fg_padded = _pad_1d(
        np.asarray(angle_fg_list, dtype=np.int32)
        if angle_fg_list
        else np.zeros((0,), dtype=np.int32),
        ang_cap,
        dtype=jnp.int32,
    )
    dih_fg_padded = _pad_1d(
        np.asarray(dih_fg_list, dtype=np.int32)
        if dih_fg_list
        else np.zeros((0,), dtype=np.int32),
        dih_cap,
        dtype=jnp.int32,
    )
    cmap_fg_padded = _pad_1d(
        np.asarray(cmap_fg_list, dtype=np.int32)
        if cmap_fg_list
        else np.zeros((0,), dtype=np.int32),
        cmap_cap,
        dtype=jnp.int32,
    )

    return ConvertedOpenMMSystem(
        bundle=bundle,
        custom_forces=custom_forces,
        force_groups=dict(force_groups),
        global_parameters=global_params,
        system_metadata=system_metadata,
        virtual_sites=virtual_sites,
        position_restraints=position_restraints,
        force_group_by_component=force_group_by_component,
        default_parameters=dict(global_params),
        nonbonded_method=nonbonded_method,
        nonbonded_cutoff=float(nonbonded_cutoff),
        lj_switch_width=float(lj_switch_width),
        use_dispersion_correction=use_dispersion_correction,
        reaction_field_dielectric=float(reaction_field_dielectric),
        pme_grid_dims=pme_grid_dims,
        pme_order=4,
        barostat=barostat_info,
        remove_com_motion=remove_com_motion,
        com_motion_frequency=com_frequency,
        n_real_particles=n_atoms_real,
        has_pbc=has_pbc,
        is_triclinic=is_triclinic,
        bond_force_groups=bond_fg_padded,
        angle_force_groups=angle_fg_padded,
        dihedral_force_groups=dih_fg_padded,
        cmap_force_groups=cmap_fg_padded,
        has_nonbonded=has_nonbonded,
        has_implicit_solvent=has_implicit_solvent,
        solvent_dielectric=float(solvent_dielectric),
        solute_dielectric=float(solute_dielectric),
        custom_nonbonded_chunk_size=int(config.custom_nonbonded_chunk_size),
    )
