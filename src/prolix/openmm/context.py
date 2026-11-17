"""Drop-in openmm.Context and openmm.State replacements powered by Prolix JAX backend."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import math
import pickle
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

import prolix.api.ensemble_plan as _ensemble_plan_mod
from prolix.constants import COULOMB_CONSTANT
from prolix.openmm.config import (
    ProlixOpenMMConfig,
    ProlixPlatform,
    _DefaultFloat,
    parse_platform_properties,
)
from prolix.openmm.converter import (
    CompiledRBTorsionForce,
    ConvertedOpenMMSystem,
    PositionRestraints,
    VirtualSites,
    from_openmm_system,
)
from prolix.openmm.lepton_compiler import (
    CompiledCustomAngleForce,
    CompiledCustomBondForce,
    CompiledCustomCompoundBondForce,
    CompiledCustomExternalForce,
    CompiledCustomNonbondedForce,
    CompiledCustomTorsionForce,
    CompiledLeptonForce,
)
from prolix.openmm.units import (
    ANGLE_K_OMM_TO_AKMA,
    ANGSTROM_TO_NM,
    BAR_PER_AKMA_PRESSURE,
    BOND_K_OMM_TO_AKMA,
    FORCE_AKMA_TO_OMM,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
    PS_TO_TAU_AKMA,
    TAU_AKMA_TO_PS,
    TORSION_K_OMM_TO_AKMA,
    VELOCITY_AKMA_TO_OMM,
    to_akma,
)
from prolix.pallas_kernels import (
    TilePairNeighborList,
    _triton_gpu_available,
    build_tile_pair_neighbor_list,
    pallas_angles_harmonic,
    pallas_bonds_2body,
    pallas_cmap_torsions,
    pallas_nonbonded_tile_pairs,
    pallas_torsions_4body,
)
from prolix.physics.bonded_masked import _cmap_energy_masked
from prolix.physics.constraints import ConstraintAlgorithm, make_constraint
from prolix.physics.explicit_corrections import lj_dispersion_tail_energy
from prolix.physics.generalized_born import (
    compute_gb_energy,
)
from prolix.physics.neighbor_list import (
    get_neighbor_exclusion_scales,
    make_neighbor_list_fn,
)
from prolix.physics.pallas_gbsa_tile import (
    GBSATilePairList,
    build_gbsa_tile_pair_list,
    pallas_gbsa_tile,
)
from prolix.physics.pallas_implicit import pallas_gb_ace_nl
from prolix.physics.pbc import (
    box_volume,
    minimum_image_displacement,
    reduce_triclinic_box_vectors,
)
from prolix.physics.pme import (
    compute_openmm_pme_params,
    compute_pme_grid_dims,
    spme_background_energy,
    spme_reciprocal_energy,
    spme_self_energy,
)
from prolix.types.bundles import MolecularBundle  # noqa: TC001

if TYPE_CHECKING:
    import openmm
    import openmm.unit as _omm_unit
    from jaxtyping import Array

    from prolix.api.ensemble_plan import EnsemblePlan
else:
    try:
        import openmm
        import openmm.unit as _omm_unit
    except ImportError:  # pragma: no cover
        openmm = None
        _omm_unit = None

unit = cast("Any", _omm_unit)

BOLTZMANN_KCAL: float = 0.0019872041  # kB in kcal/(mol·K)


# =============================================================================
# 1. ProlixState
# =============================================================================


if TYPE_CHECKING:
    _StateBase = openmm.State
else:
    _StateBase = getattr(openmm, "State", object) if openmm is not None else object


class ProlixState(_StateBase):
    """Drop-in replacement for openmm.State returning native OpenMM Quantities."""

    def __init__(
        self,
        positions_nm: np.ndarray | None = None,
        velocities_omm: np.ndarray | None = None,
        forces_omm: np.ndarray | None = None,
        box_vectors_nm: np.ndarray | None = None,
        time_ps: float = 0.0,
        step_count: int = 0,
        pe_kj: float | None = None,
        ke_kj: float | None = None,
        parameters: dict[str, float] | None = None,
        param_derivatives: dict[str, float] | None = None,
        data_types: int = 0,
    ) -> None:
        self._positions_nm = positions_nm
        self._velocities_omm = velocities_omm
        self._forces_omm = forces_omm
        self._box_vectors_nm = (
            np.asarray(box_vectors_nm, dtype=np.float64)
            if box_vectors_nm is not None
            else np.eye(3, dtype=np.float64)
        )
        self._time_ps = float(time_ps)
        self._step_count = int(step_count)
        self._pe_kj = float(pe_kj) if pe_kj is not None else None
        self._ke_kj = float(ke_kj) if ke_kj is not None else None
        self._parameters = dict(parameters) if parameters is not None else None
        self._param_derivatives = (
            dict(param_derivatives) if param_derivatives is not None else None
        )
        self._data_types = int(data_types)
        self._c_state: openmm.State | None = None

    def getTime(self) -> unit.Quantity:  # noqa: N802
        return self._time_ps * unit.picosecond

    def getStepCount(self) -> int:  # noqa: N802
        return self._step_count

    def getPeriodicBoxVectors(self, asNumpy: bool = False) -> unit.Quantity:  # noqa: N802
        if asNumpy:
            return self._box_vectors_nm * unit.nanometers
        return [
            openmm.Vec3(float(row[0]), float(row[1]), float(row[2]))
            for row in self._box_vectors_nm
        ] * unit.nanometers

    def getPeriodicBoxVolume(self) -> unit.Quantity:  # noqa: N802
        vol = abs(float(np.linalg.det(self._box_vectors_nm)))
        return vol * (unit.nanometers**3)

    def getPositions(self, asNumpy: bool = False) -> unit.Quantity:  # noqa: N802
        if self._positions_nm is None:
            raise openmm.OpenMMException(
                "Invoked getPositions() on a State which does not contain positions."
            )
        if asNumpy:
            return self._positions_nm * unit.nanometers
        return [
            openmm.Vec3(float(row[0]), float(row[1]), float(row[2]))
            for row in self._positions_nm
        ] * unit.nanometers

    def getVelocities(self, asNumpy: bool = False) -> unit.Quantity:  # noqa: N802
        if self._velocities_omm is None:
            raise openmm.OpenMMException(
                "Invoked getVelocities() on a State which does not contain velocities."
            )
        if asNumpy:
            return self._velocities_omm * (unit.nanometer / unit.picosecond)
        return [
            openmm.Vec3(float(row[0]), float(row[1]), float(row[2]))
            for row in self._velocities_omm
        ] * (unit.nanometer / unit.picosecond)

    def getForces(self, asNumpy: bool = False) -> unit.Quantity:  # noqa: N802
        if self._forces_omm is None:
            raise openmm.OpenMMException(
                "Invoked getForces() on a State which does not contain forces."
            )
        if asNumpy:
            return self._forces_omm * (unit.kilojoules_per_mole / unit.nanometer)
        return [
            openmm.Vec3(float(row[0]), float(row[1]), float(row[2]))
            for row in self._forces_omm
        ] * (unit.kilojoules_per_mole / unit.nanometer)

    def getPotentialEnergy(self) -> unit.Quantity:  # noqa: N802
        if self._pe_kj is None:
            raise openmm.OpenMMException(
                "Invoked getPotentialEnergy() on a State which does not contain energies."
            )
        return self._pe_kj * unit.kilojoules_per_mole

    def getKineticEnergy(self) -> unit.Quantity:  # noqa: N802
        if self._ke_kj is None:
            raise openmm.OpenMMException(
                "Invoked getKineticEnergy() on a State which does not contain energies."
            )
        return self._ke_kj * unit.kilojoules_per_mole

    def getParameters(self) -> dict[str, float]:  # noqa: N802
        if self._parameters is None:
            raise openmm.OpenMMException(
                "Invoked getParameters() on a State which does not contain parameters."
            )
        return dict(self._parameters)

    def getEnergyParameterDerivatives(self) -> dict[str, float]:  # noqa: N802
        if self._param_derivatives is None:
            raise openmm.OpenMMException(
                "Invoked getEnergyParameterDerivatives() on a State which does not contain parameter derivatives."
            )
        return dict(self._param_derivatives)

    def getDataTypes(self) -> int:  # noqa: N802
        return self._data_types

    def _build_c_state(self) -> openmm.State:
        omm_mod = cast("Any", openmm)
        omm_ver = getattr(omm_mod, "__version__", "8.0")
        bv = self._box_vectors_nm
        lines = [
            '<?xml version="1.0" ?>',
            f'<State openmmVersion="{omm_ver}" stepCount="{self._step_count}" time="{self._time_ps}" type="State" version="1">',
            "  <PeriodicBoxVectors>",
            f'    <A x="{bv[0, 0]}" y="{bv[0, 1]}" z="{bv[0, 2]}"/>',
            f'    <B x="{bv[1, 0]}" y="{bv[1, 1]}" z="{bv[1, 2]}"/>',
            f'    <C x="{bv[2, 0]}" y="{bv[2, 1]}" z="{bv[2, 2]}"/>',
            "  </PeriodicBoxVectors>",
        ]
        if self._parameters is not None:
            p_str = " ".join(f'{k}="{v}"' for k, v in self._parameters.items())
            lines.append(f"  <Parameters {p_str}/>")

        if self._pe_kj is not None or self._ke_kj is not None:
            ke = self._ke_kj if self._ke_kj is not None else 0.0
            pe = self._pe_kj if self._pe_kj is not None else 0.0
            lines.append(f'  <Energies KineticEnergy="{ke}" PotentialEnergy="{pe}"/>')

        if self._positions_nm is not None:
            lines.append("  <Positions>")
            for p in self._positions_nm:
                lines.append(f'    <Position x="{p[0]}" y="{p[1]}" z="{p[2]}"/>')
            lines.append("  </Positions>")

        if self._velocities_omm is not None:
            lines.append("  <Velocities>")
            for v in self._velocities_omm:
                lines.append(f'    <Velocity x="{v[0]}" y="{v[1]}" z="{v[2]}"/>')
            lines.append("  </Velocities>")

        if self._forces_omm is not None:
            lines.append("  <Forces>")
            for f in self._forces_omm:
                lines.append(f'    <Force x="{f[0]}" y="{f[1]}" z="{f[2]}"/>')
            lines.append("  </Forces>")

        lines.append("</State>")
        xml_str = "\n".join(lines)
        return omm_mod.XmlSerializer.deserialize(xml_str)

    @property
    def this(self) -> Any:
        if self._c_state is None:
            self._c_state = self._build_c_state()
        c_state_any = cast("Any", self._c_state)
        return c_state_any.this

    @property
    def thisown(self) -> int:
        return 0

    @thisown.setter
    def thisown(self, val: Any) -> None:
        pass


# =============================================================================
# 2. Pure JIT-Compatible Energy & Rollout Primitives
# =============================================================================


def _minimum_image_dr(dr: jnp.ndarray, box: jnp.ndarray | None, has_pbc: bool) -> jnp.ndarray:
    """Compute minimum-image displacement vector dr = r_i - r_j."""
    if not has_pbc or box is None:
        return dr
    return minimum_image_displacement(dr, jnp.zeros_like(dr), box)


def _resolve_term_force_group_mask(
    fg_arr: jnp.ndarray | None,
    target_len: int,
    groups_mask: int,
    default_group: int,
) -> jnp.ndarray | bool:
    """Return boolean force-group activity mask aligned to `target_len`."""
    if fg_arr is None:
        return bool((groups_mask & (1 << int(default_group))) != 0)
    cur_len = int(fg_arr.shape[0])
    if cur_len > target_len:
        fg_arr = fg_arr[:target_len]
    elif cur_len < target_len:
        fg_arr = jnp.pad(fg_arr, (0, target_len - cur_len), constant_values=0)
    return (jnp.int32(groups_mask) & (jnp.int32(1) << fg_arr.astype(jnp.int32))) != 0


def _eval_bonded_terms(
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
    bond_force_groups: jnp.ndarray | None = None,
    angle_force_groups: jnp.ndarray | None = None,
    dihedral_force_groups: jnp.ndarray | None = None,
    cmap_force_groups: jnp.ndarray | None = None,
    use_pallas: bool = False,
    pallas_interpret: bool = False,
) -> jnp.ndarray:
    """Evaluate harmonic bonds, angles, dihedrals, and restraints for active groups."""
    e_total = jnp.array(0.0, dtype=pos.dtype)
    n = pos.shape[0]
    effective_interpret = bool(pallas_interpret) or not _triton_gpu_available()

    # Harmonic Bonds
    if bundle.bond_idx is not None and bundle.bond_params is not None:
        b_idx = bundle.bond_idx
        b_params = bundle.bond_params
        b_mask = bundle.bond_mask
        fg_active = _resolve_term_force_group_mask(
            bond_force_groups,
            int(b_mask.shape[0]),
            groups_mask,
            force_group_by_component.get("bond", 0),
        )
        p1 = b_idx[:, 0]
        p2 = b_idx[:, 1]
        valid = b_mask & (p1 < n) & (p2 < n) & fg_active
        p1_safe = jnp.where(valid, p1, 0)
        p2_safe = jnp.where(valid, p2, 0)
        if use_pallas and pallas_bonds_2body is not None:
            box_vec = box if (has_pbc and box is not None) else None
            e_b = cast(
                "jnp.ndarray",
                pallas_bonds_2body(
                    pos,
                    b_idx,
                    b_params,
                    mask=valid.astype(pos.dtype),
                    box=box_vec,
                    dist_eps=1e-10,
                    interpret=effective_interpret,
                ),
            )
            e_total = e_total + e_b
        else:
            dr = _minimum_image_dr(pos[p1_safe] - pos[p2_safe], box, has_pbc)
            r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
            r0 = b_params[:, 0]
            k = b_params[:, 1]
            e_b = 0.5 * k * (r - r0) ** 2
            e_total = e_total + jnp.sum(jnp.where(valid, e_b, 0.0))

    # Harmonic Angles
    if bundle.angle_idx is not None and bundle.angle_params is not None:
        a_idx = bundle.angle_idx
        a_params = bundle.angle_params
        a_mask = bundle.angle_mask
        fg_active = _resolve_term_force_group_mask(
            angle_force_groups,
            int(a_mask.shape[0]),
            groups_mask,
            force_group_by_component.get("angle", 0),
        )
        p1 = a_idx[:, 0]
        p2 = a_idx[:, 1]
        p3 = a_idx[:, 2]
        valid = a_mask & (p1 < n) & (p2 < n) & (p3 < n) & fg_active
        if use_pallas and pallas_angles_harmonic is not None:
            box_vec = box if (has_pbc and box is not None) else None
            e_a = pallas_angles_harmonic(
                pos,
                a_idx,
                a_params,
                mask=valid.astype(pos.dtype),
                box=box_vec,
                dist_eps=1e-10,
                interpret=effective_interpret,
            )
            e_total = e_total + e_a
        else:
            p1_s = jnp.where(valid, p1, 0)
            p2_s = jnp.where(valid, p2, 0)
            p3_s = jnp.where(valid, p3, 0)
            v1 = _minimum_image_dr(pos[p1_s] - pos[p2_s], box, has_pbc)
            v2 = _minimum_image_dr(pos[p3_s] - pos[p2_s], box, has_pbc)
            v1_norm = jnp.sqrt(jnp.sum(v1**2, axis=-1) + 1e-12)
            v2_norm = jnp.sqrt(jnp.sum(v2**2, axis=-1) + 1e-12)
            dot = jnp.sum(v1 * v2, axis=-1) / (v1_norm * v2_norm)
            dot_safe = jnp.clip(dot, -0.999999, 0.999999)
            th = jnp.arccos(dot_safe)
            th0 = a_params[:, 0]
            k_a = a_params[:, 1]
            e_a = 0.5 * k_a * (th - th0) ** 2
            e_total = e_total + jnp.sum(jnp.where(valid, e_a, 0.0))

    # Periodic Torsions / Dihedrals
    if bundle.dihedral_idx is not None and bundle.dihedral_params is not None:
        d_idx = bundle.dihedral_idx
        d_params = bundle.dihedral_params
        d_mask = bundle.dihedral_mask
        fg_active = _resolve_term_force_group_mask(
            dihedral_force_groups,
            int(d_mask.shape[0]),
            groups_mask,
            force_group_by_component.get("dihedral", 0),
        )
        p1 = d_idx[:, 0]
        p2 = d_idx[:, 1]
        p3 = d_idx[:, 2]
        p4 = d_idx[:, 3]
        valid = d_mask & (p1 < n) & (p2 < n) & (p3 < n) & (p4 < n) & fg_active
        if use_pallas and pallas_torsions_4body is not None:
            box_vec = box if (has_pbc and box is not None) else None
            e_d = pallas_torsions_4body(
                pos,
                d_idx,
                d_params,
                mask=valid.astype(pos.dtype),
                box=box_vec,
                b0_is_ji=True,
                interpret=effective_interpret,
            )
            e_total = e_total + e_d
        else:
            p1_s = jnp.where(valid, p1, 0)
            p2_s = jnp.where(valid, p2, 0)
            p3_s = jnp.where(valid, p3, 0)
            p4_s = jnp.where(valid, p4, 0)

            b1 = _minimum_image_dr(pos[p2_s] - pos[p1_s], box, has_pbc)
            b2 = _minimum_image_dr(pos[p3_s] - pos[p2_s], box, has_pbc)
            b3 = _minimum_image_dr(pos[p4_s] - pos[p3_s], box, has_pbc)

            c1 = jnp.cross(b1, b2)
            c2 = jnp.cross(b2, b3)
            b2_norm = jnp.sqrt(jnp.sum(b2**2, axis=-1, keepdims=True) + 1e-12)
            b2_u = b2 / b2_norm

            m = jnp.cross(b2_u, c1)
            x = jnp.sum(c1 * c2, axis=-1)
            y = jnp.sum(m * c2, axis=-1)
            x_safe = jnp.where(valid, x, 1.0)
            y_safe = jnp.where(valid, y, 0.0)
            phi = jnp.arctan2(y_safe, x_safe)

            per = d_params[:, 0]
            phase = d_params[:, 1]
            k_d = d_params[:, 2]
            e_d = k_d * (1.0 + jnp.cos(per * phi - phase))
            e_total = e_total + jnp.sum(jnp.where(valid, e_d, 0.0))

    # Harmonic Impropers
    g_imp = force_group_by_component.get("improper", 0)
    if ((groups_mask & (1 << int(g_imp))) != 0) and bundle.improper_idx is not None and bundle.improper_params is not None:
        i_idx = bundle.improper_idx
        i_params = bundle.improper_params
        i_mask = bundle.improper_mask
        p1 = i_idx[:, 0]
        p2 = i_idx[:, 1]
        p3 = i_idx[:, 2]
        p4 = i_idx[:, 3]
        valid = i_mask & (p1 < n) & (p2 < n) & (p3 < n) & (p4 < n)
        p1_s = jnp.where(valid, p1, 0)
        p2_s = jnp.where(valid, p2, 0)
        p3_s = jnp.where(valid, p3, 0)
        p4_s = jnp.where(valid, p4, 0)

        b1 = _minimum_image_dr(pos[p2_s] - pos[p1_s], box, has_pbc)
        b2 = _minimum_image_dr(pos[p3_s] - pos[p2_s], box, has_pbc)
        b3 = _minimum_image_dr(pos[p4_s] - pos[p3_s], box, has_pbc)

        c1 = jnp.cross(b1, b2)
        c2 = jnp.cross(b2, b3)
        b2_norm = jnp.sqrt(jnp.sum(b2**2, axis=-1, keepdims=True) + 1e-12)
        b2_u = b2 / b2_norm

        m = jnp.cross(b2_u, c1)
        x = jnp.sum(c1 * c2, axis=-1)
        y = jnp.sum(m * c2, axis=-1)
        x_safe = jnp.where(valid, x, 1.0)
        y_safe = jnp.where(valid, y, 0.0)
        phi = jnp.arctan2(y_safe, x_safe)

        is_periodic = (
            jnp.asarray(bundle.improper_is_periodic, dtype=bool)
            if hasattr(bundle, "improper_is_periodic") and bundle.improper_is_periodic is not None
            else jnp.array(False, dtype=bool)
        )

        per = i_params[:, 0]
        phase = i_params[:, 1]
        k_per = i_params[:, 2]
        e_per = k_per * (1.0 + jnp.cos(per * phi - phase))

        k_harm = i_params[:, 0]
        phi0 = i_params[:, 1]
        diff_raw = phi - phi0 + jnp.pi
        dphi = diff_raw - 2.0 * jnp.pi * jnp.floor(diff_raw / (2.0 * jnp.pi)) - jnp.pi
        e_harm = k_harm * (dphi ** 2)

        e_i = jnp.where(is_periodic, e_per, e_harm)
        e_total = e_total + jnp.sum(jnp.where(valid, e_i, 0.0))

    # CMAP Torsions
    if (
        bundle.cmap_torsion_idx is not None
        and bundle.cmap_indices is not None
        and bundle.cmap_mask is not None
        and bundle.cmap_coeffs is not None
        and bundle.cmap_coeffs.shape[0] > 0
    ):
        fg_active = _resolve_term_force_group_mask(
            cmap_force_groups,
            int(bundle.cmap_mask.shape[0]),
            groups_mask,
            force_group_by_component.get("cmap", 0),
        )
        valid_cmap = bundle.cmap_mask & fg_active
        if use_pallas and pallas_cmap_torsions is not None:
            box_vec = box if (has_pbc and box is not None) else None
            e_cmap = pallas_cmap_torsions(
                pos,
                bundle.cmap_torsion_idx,
                bundle.cmap_indices,
                bundle.cmap_coeffs,
                box=box_vec,
                mask=valid_cmap.astype(pos.dtype),
                first_is_psi=False,
                interpret=effective_interpret,
            )
            e_total = e_total + e_cmap
        else:
            disp_fn = (
                (lambda ra, rb: _minimum_image_dr(ra - rb, box, has_pbc))
                if has_pbc and box is not None
                else (lambda ra, rb: ra - rb)
            )
            e_cmap = _cmap_energy_masked(
                pos,
                bundle.cmap_torsion_idx,
                bundle.cmap_indices,
                valid_cmap,
                bundle.cmap_coeffs,
                disp_fn,
                first_is_psi=False,
            )
            e_total = e_total + e_cmap

    return e_total


def _eval_nonbonded_terms(  # noqa: PLR0913
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
    cutoff: float,
    nonbonded_method: str = "nocutoff",
    has_nonbonded: bool = True,
    use_dispersion_correction: bool = False,
    ewald_error_tolerance: float = 0.0005,
    pme_alpha: float | None = None,
    rf_dielectric: float = 78.3,
    nonbonded_engine: str = "dense",
    pme_grid_dims: tuple[int, int, int] | None = None,
    use_pallas: bool = False,
    pallas_interpret: bool = False,
    lj_switch_width: float = 0.0,
    neighbor_idx: jnp.ndarray | None = None,
    use_pallas_pme: bool = False,
    custom_nonbonded_chunk_size: int = 128,
    tile_pair_nl: TilePairNeighborList | None = None,
    is_triclinic: bool = False,
) -> jnp.ndarray:
    """Evaluate pairwise Lennard-Jones, Coulomb, exclusions, 1-4 exceptions, and PME/RF/dispersion."""
    g_nb = force_group_by_component.get("nonbonded", 0)
    if ((groups_mask & (1 << int(g_nb))) == 0) or bundle.charges is None or not has_nonbonded:
        return jnp.array(0.0, dtype=pos.dtype)

    n = pos.shape[0]
    q = bundle.charges[:n].astype(pos.dtype)
    sig = bundle.sigmas[:n].astype(pos.dtype)
    eps = bundle.epsilons[:n].astype(pos.dtype)
    box_eval = box.astype(pos.dtype) if box is not None else None
    box_pbc = (
        jnp.diag(box_eval)
        if (box_eval is not None and box_eval.ndim == 2 and not is_triclinic)
        else box_eval
    )

    nb_method = str(nonbonded_method).lower()
    is_pme = nb_method in ("pme", "spme")
    is_rf = (
        nb_method in ("cutoffperiodic", "cutoff_periodic", "reactionfield", "reaction_field", "rf")
        and has_pbc
    )
    bundle_alpha_concrete = 0.0
    if is_pme and pme_alpha is None and bundle.pme_alpha is not None:
        with contextlib.suppress(Exception):
            bundle_alpha_concrete = float(bundle.pme_alpha)

    if is_pme and pme_alpha is not None and float(pme_alpha) > 0.0:
        alpha_float = float(pme_alpha)
    elif is_pme and bundle_alpha_concrete > 0.0:
        alpha_float = bundle_alpha_concrete
    elif is_pme and cutoff < float("inf"):
        alpha_float = float(
            (1.0 / cutoff) * math.sqrt(-math.log(2.0 * float(ewald_error_tolerance)))
        )
    else:
        alpha_float = 0.34
    alpha = jnp.asarray(alpha_float, dtype=pos.dtype)

    use_tile_pairs = (
        tile_pair_nl is not None
        and (use_pallas or nonbonded_engine == "morton_tile_pairs")
        and not is_triclinic
        and not is_rf
        and pos.dtype == jnp.float32
    )

    # Evaluate pair terms
    direct_exclusions_premasked = False
    if use_tile_pairs and tile_pair_nl is not None:
        direct_exclusions_premasked = True
        box_diag = (
            (jnp.diag(box) if box.ndim == 2 else box)
            if (has_pbc and box is not None)
            else None
        )
        eff_cutoff = cutoff if cutoff < float("inf") else 1.0e6
        tot = cast(
            "jnp.ndarray",
            pallas_nonbonded_tile_pairs(
                pos,
                q,
                sig,
                eps,
                tile_pair_nl,
                box=box_diag,
                pme_alpha=alpha_float if is_pme else 0.0,
                cutoff=float(eff_cutoff),
                coulomb_constant=COULOMB_CONSTANT,
                lj_switch_width=float(lj_switch_width) if cutoff < float("inf") else 0.0,
                include_pme_excl_erf=False,
                return_forces=False,
                interpret=pallas_interpret,
            ),
        )
    elif neighbor_idx is not None:
        # Genuine NeighborList O(N * K) direct path (O(N * K * log2 M) exclusion binary search)
        i_idx = jnp.arange(n, dtype=jnp.int32)[:, None]
        nb_clean = jnp.where(neighbor_idx < 0, n, neighbor_idx & ((1 << 29) - 1))
        mask_pair = (nb_clean >= 0) & (nb_clean < n) & (i_idx < nb_clean)
        sv_nl: jnp.ndarray | None = None
        se_nl: jnp.ndarray | None = None
        if bundle.excl_dense_indices is not None and n > 0:
            excl_dense_n = bundle.excl_dense_indices[:n]
            vdw_sc = (
                bundle.excl_dense_scales_vdw[:n]
                if bundle.excl_dense_scales_vdw is not None
                else jnp.zeros_like(excl_dense_n, dtype=pos.dtype)
            )
            elec_sc = (
                bundle.excl_dense_scales_elec[:n]
                if bundle.excl_dense_scales_elec is not None
                else jnp.zeros_like(excl_dense_n, dtype=pos.dtype)
            )
            sv_raw, se_raw = get_neighbor_exclusion_scales(
                excl_dense_n, vdw_sc, elec_sc, nb_clean
            )
            sv_nl = sv_raw.astype(pos.dtype)
            se_nl = se_raw.astype(pos.dtype)
            direct_exclusions_premasked = True
        j_safe = jnp.where(mask_pair, nb_clean, 0)

        pos_i = pos[:, None, :]
        pos_j = pos[j_safe]
        dr = pos_i - pos_j
        if has_pbc and box_pbc is not None:
            dr = _minimum_image_dr(dr, box_pbc, has_pbc)
        r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
        r_safe = jnp.where(mask_pair, r, 1.0)

        sig_mat = 0.5 * (sig[:, None] + sig[j_safe])
        eps_mat = jnp.sqrt(jnp.maximum(eps[:, None] * eps[j_safe], 0.0))
        q_mat = q[:, None] * q[j_safe]

        sr = sig_mat / r_safe
        sr6 = sr**6
        sr12 = sr6**2
        u_lj = 4.0 * eps_mat * (sr12 - sr6)

        if lj_switch_width > 0.0 and cutoff < float("inf"):
            r_sw = cutoff - lj_switch_width
            z = jnp.clip((r_safe - r_sw) / lj_switch_width, 0.0, 1.0)
            s_poly = 1.0 - 10.0 * (z**3) + 15.0 * (z**4) - 6.0 * (z**5)
            s_lj = jnp.where(r_safe <= r_sw, 1.0, jnp.where(r_safe < cutoff, s_poly, 0.0))
            u_lj = u_lj * s_lj

        if is_pme:
            u_elec = COULOMB_CONSTANT * q_mat * (jax.lax.erfc(alpha * r_safe) / r_safe)
        elif is_rf:
            eps_rf = rf_dielectric
            k_rf = (1.0 / (cutoff**3)) * ((eps_rf - 1.0) / (2.0 * eps_rf + 1.0))
            c_rf = (1.0 / cutoff) * ((3.0 * eps_rf) / (2.0 * eps_rf + 1.0))
            u_elec = COULOMB_CONSTANT * q_mat * (1.0 / r_safe + k_rf * (r_safe**2) - c_rf)
        else:
            u_elec = COULOMB_CONSTANT * (q_mat / r_safe)

        if sv_nl is not None and se_nl is not None:
            u_lj = u_lj * sv_nl
            u_elec = u_elec * se_nl

        u_pair = u_lj + u_elec
        if cutoff < float("inf"):
            u_pair = jnp.where(r_safe < cutoff, u_pair, 0.0)

        tot = jnp.sum(jnp.where(mask_pair, u_pair, 0.0))
    elif n >= 4000 or nonbonded_engine in ("neighbor_list", "flash", "morton_tile_pairs"):
        chunk_size = custom_nonbonded_chunk_size if custom_nonbonded_chunk_size > 0 else 256
        n_chunks = (n + chunk_size - 1) // chunk_size
        direct_exclusions_premasked = bool(
            n > 0 and bundle.excl_dense_indices is not None
        )

        def chunk_body(c_idx: int, accum: jnp.ndarray) -> jnp.ndarray:
            i_start = c_idx * chunk_size
            i_indices = i_start + jnp.arange(chunk_size)
            valid_i = i_indices < n
            safe_i = jnp.where(valid_i, i_indices, 0)
            pos_chunk = pos[safe_i]
            dr = pos_chunk[:, None, :] - pos[None, :, :]
            if has_pbc and box_pbc is not None:
                dr = _minimum_image_dr(dr, box_pbc, has_pbc)
            r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)

            all_j = jnp.arange(n)[None, :]
            mask_pair = valid_i[:, None] & (safe_i[:, None] < all_j)
            if bundle.excl_dense_indices is not None and n > 0:
                ex_row = bundle.excl_dense_indices[safe_i]
                valid_ex_slot = (ex_row >= 0) & (ex_row < n)
                row_idx = jnp.arange(chunk_size)[:, None]
                col_idx = jnp.where(valid_ex_slot, ex_row, 0)
                mask_pair = mask_pair.at[row_idx, col_idx].set(False)
            r_safe = jnp.where(mask_pair, r, 1.0)

            sig_mat = 0.5 * (sig[safe_i][:, None] + sig[None, :])
            eps_mat = jnp.sqrt(jnp.maximum(eps[safe_i][:, None] * eps[None, :], 0.0))
            q_mat = q[safe_i][:, None] * q[None, :]

            sr = sig_mat / r_safe
            sr6 = sr**6
            sr12 = sr6**2
            u_lj = 4.0 * eps_mat * (sr12 - sr6)

            if lj_switch_width > 0.0 and cutoff < float("inf"):
                r_sw = cutoff - lj_switch_width
                z = jnp.clip((r_safe - r_sw) / lj_switch_width, 0.0, 1.0)
                s_poly = 1.0 - 10.0 * (z**3) + 15.0 * (z**4) - 6.0 * (z**5)
                s_lj = jnp.where(r_safe <= r_sw, 1.0, jnp.where(r_safe < cutoff, s_poly, 0.0))
                u_lj = u_lj * s_lj

            if is_pme:
                u_elec = COULOMB_CONSTANT * q_mat * (jax.lax.erfc(alpha * r_safe) / r_safe)
            elif is_rf:
                eps_rf = rf_dielectric
                k_rf = (1.0 / (cutoff**3)) * ((eps_rf - 1.0) / (2.0 * eps_rf + 1.0))
                c_rf = (1.0 / cutoff) * ((3.0 * eps_rf) / (2.0 * eps_rf + 1.0))
                u_elec = COULOMB_CONSTANT * q_mat * (1.0 / r_safe + k_rf * (r_safe**2) - c_rf)
            else:
                u_elec = COULOMB_CONSTANT * (q_mat / r_safe)

            u_pair = u_lj + u_elec
            if cutoff < float("inf"):
                u_pair = jnp.where(r_safe < cutoff, u_pair, 0.0)

            return accum + jnp.sum(jnp.where(mask_pair, u_pair, 0.0))

        tot = jax.lax.fori_loop(0, n_chunks, chunk_body, jnp.array(0.0, dtype=pos.dtype))
    else:
        # Standard upper triangular all-pairs
        dr = pos[:, None, :] - pos[None, :, :]
        if has_pbc and box_pbc is not None:
            dr = _minimum_image_dr(dr, box_pbc, has_pbc)
        r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
        mask_tri = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
        if n > 0 and bundle.excl_indices is not None and bundle.excl_mask is not None:
            ex_idx = bundle.excl_indices
            valid_ex = bundle.excl_mask & (ex_idx[:, 0] < n) & (ex_idx[:, 1] < n)
            p_lo = jnp.where(valid_ex, jnp.minimum(ex_idx[:, 0], ex_idx[:, 1]), 0)
            p_hi = jnp.where(valid_ex, jnp.maximum(ex_idx[:, 0], ex_idx[:, 1]), 0)
            mask_tri = mask_tri.at[p_lo, p_hi].set(False)
            direct_exclusions_premasked = True
        r_safe = jnp.where(mask_tri, r, 1.0)

        sig_mat = 0.5 * (sig[:, None] + sig[None, :])
        eps_mat = jnp.sqrt(jnp.maximum(eps[:, None] * eps[None, :], 0.0))
        q_mat = q[:, None] * q[None, :]

        sr = sig_mat / r_safe
        sr6 = sr**6
        sr12 = sr6**2
        u_lj = 4.0 * eps_mat * (sr12 - sr6)

        if lj_switch_width > 0.0 and cutoff < float("inf"):
            r_sw = cutoff - lj_switch_width
            z = jnp.clip((r_safe - r_sw) / lj_switch_width, 0.0, 1.0)
            s_poly = 1.0 - 10.0 * (z**3) + 15.0 * (z**4) - 6.0 * (z**5)
            s_lj = jnp.where(r_safe <= r_sw, 1.0, jnp.where(r_safe < cutoff, s_poly, 0.0))
            u_lj = u_lj * s_lj

        if is_pme:
            u_elec = COULOMB_CONSTANT * q_mat * (jax.lax.erfc(alpha * r_safe) / r_safe)
        elif is_rf:
            eps_rf = rf_dielectric
            k_rf = (1.0 / (cutoff**3)) * ((eps_rf - 1.0) / (2.0 * eps_rf + 1.0))
            c_rf = (1.0 / cutoff) * ((3.0 * eps_rf) / (2.0 * eps_rf + 1.0))
            u_elec = COULOMB_CONSTANT * q_mat * (1.0 / r_safe + k_rf * (r_safe**2) - c_rf)
        else:
            u_elec = COULOMB_CONSTANT * (q_mat / r_safe)

        u_pair = u_lj + u_elec
        if cutoff < float("inf"):
            cutoff_mask = r_safe < cutoff
            u_pair = jnp.where(cutoff_mask, u_pair, 0.0)

        tot = jnp.sum(jnp.where(mask_tri, u_pair, 0.0))

    # Exclusions
    if bundle.excl_indices is not None and bundle.excl_mask is not None:
        ex_idx = bundle.excl_indices
        ex_mask = bundle.excl_mask
        p1 = ex_idx[:, 0]
        p2 = ex_idx[:, 1]
        valid_ex = ex_mask & (p1 < n) & (p2 < n)
        p1_s = jnp.where(valid_ex, p1, 0)
        p2_s = jnp.where(valid_ex, p2, 0)

        dr_ex_raw = pos[p1_s] - pos[p2_s]
        r_ex_raw = jnp.sqrt(jnp.sum(dr_ex_raw**2, axis=-1) + 1e-12)
        r_ex_raw_s = jnp.where(valid_ex, r_ex_raw, 1.0)

        if direct_exclusions_premasked:
            # Direct-space exclusions were masked before r_safe / (sig/r)^12 to prevent
            # float32 catastrophic cancellation (~10^8 kcal/mol on 1-2/1-3 covalent bonds; BUG-OMM10);
            # only subtract PME reciprocal-space erf self-pair contribution.
            if is_pme:
                u_ex_recip = COULOMB_CONSTANT * q[p1_s] * q[p2_s] * (
                    jax.lax.erf(alpha * r_ex_raw_s) / r_ex_raw_s
                )
                tot = tot - jnp.sum(jnp.where(valid_ex, u_ex_recip, 0.0))
        else:
            dr_ex = dr_ex_raw
            if has_pbc and box_pbc is not None:
                dr_ex = _minimum_image_dr(dr_ex, box_pbc, has_pbc)
            r_ex = jnp.sqrt(jnp.sum(dr_ex**2, axis=-1) + 1e-12)
            r_ex_s = jnp.where(valid_ex, r_ex, 1.0)
            sig_ex = 0.5 * (sig[p1_s] + sig[p2_s])
            eps_ex = jnp.sqrt(jnp.maximum(eps[p1_s] * eps[p2_s], 0.0))
            q_ex = q[p1_s] * q[p2_s]

            sr_e = sig_ex / r_ex_s
            sr6_e = sr_e**6
            sr12_e = sr6_e**2
            u_ex_lj = 4.0 * eps_ex * (sr12_e - sr6_e)
            if lj_switch_width > 0.0 and cutoff < float("inf"):
                r_sw = cutoff - lj_switch_width
                z_e = jnp.clip((r_ex_s - r_sw) / lj_switch_width, 0.0, 1.0)
                s_poly_e = 1.0 - 10.0 * (z_e**3) + 15.0 * (z_e**4) - 6.0 * (z_e**5)
                s_lj_e = jnp.where(
                    r_ex_s <= r_sw, 1.0, jnp.where(r_ex_s < cutoff, s_poly_e, 0.0)
                )
                u_ex_lj = u_ex_lj * s_lj_e

            if is_pme:
                u_ex_elec = COULOMB_CONSTANT * q_ex * (jax.lax.erfc(alpha * r_ex_s) / r_ex_s)
                u_ex_recip = COULOMB_CONSTANT * q_ex * (jax.lax.erf(alpha * r_ex_raw_s) / r_ex_raw_s)
            elif is_rf:
                eps_rf = rf_dielectric
                k_rf = (1.0 / (cutoff**3)) * ((eps_rf - 1.0) / (2.0 * eps_rf + 1.0))
                c_rf = (1.0 / cutoff) * ((3.0 * eps_rf) / (2.0 * eps_rf + 1.0))
                u_ex_elec = COULOMB_CONSTANT * q_ex * (1.0 / r_ex_s + k_rf * (r_ex_s**2) - c_rf)
                u_ex_recip = 0.0
            else:
                u_ex_elec = COULOMB_CONSTANT * (q_ex / r_ex_s)
                u_ex_recip = 0.0

            u_ex = u_ex_lj + u_ex_elec
            if cutoff < float("inf"):
                u_ex = jnp.where(r_ex_s < cutoff, u_ex, 0.0)
            tot = tot - jnp.sum(jnp.where(valid_ex, u_ex + u_ex_recip, 0.0))

    # 1-4 Exceptions (DEF-20: Pure direct Coulomb and LJ; no erf double subtraction, no continuum contamination)
    if (
        bundle.exception_pairs is not None
        and bundle.exception_sigmas is not None
        and bundle.exception_epsilons is not None
        and bundle.exception_chargeprods is not None
        and bundle.exception_mask is not None
    ):
        exc_pairs = bundle.exception_pairs
        exc_mask = bundle.exception_mask
        ep1 = exc_pairs[:, 0]
        ep2 = exc_pairs[:, 1]
        valid_exc = exc_mask & (ep1 < n) & (ep2 < n)
        ep1_s = jnp.where(valid_exc, ep1, 0)
        ep2_s = jnp.where(valid_exc, ep2, 0)

        dr_exc = pos[ep1_s] - pos[ep2_s]
        r_exc = jnp.sqrt(jnp.sum(dr_exc**2, axis=-1) + 1e-12)
        r_exc_s = jnp.where(valid_exc, r_exc, 1.0)
        sig_14 = bundle.exception_sigmas.astype(pos.dtype)
        eps_14 = bundle.exception_epsilons.astype(pos.dtype)
        qp_14 = bundle.exception_chargeprods.astype(pos.dtype)

        sr_14 = sig_14 / r_exc_s
        sr6_14 = sr_14**6
        sr12_14 = sr6_14**2
        u_14_lj = 4.0 * eps_14 * (sr12_14 - sr6_14)

        u_14_elec = COULOMB_CONSTANT * (qp_14 / r_exc_s)

        u_14 = u_14_lj + u_14_elec
        tot = tot + jnp.sum(jnp.where(valid_exc, u_14, 0.0))

    # PME reciprocal, self, and plasma background terms
    if is_pme and has_pbc and box_pbc is not None and pme_grid_dims is not None:
        atom_mask = jnp.ones(n, dtype=bool)
        with jax.default_matmul_precision("float32"):
            e_recip = spme_reciprocal_energy(
                pos,
                q,
                atom_mask,
                box_pbc,
                pme_grid_dims,
                alpha=alpha_float,
                order=4,
                use_pallas_pme=use_pallas_pme,
            )
        e_self = spme_self_energy(q, atom_mask, alpha=alpha_float)
        e_plasma = spme_background_energy(q, atom_mask, alpha=alpha_float, box_size=box_pbc)
        tot = tot + e_recip + e_self + e_plasma

    # LJ dispersion tail correction
    if use_dispersion_correction and has_pbc and box_pbc is not None and cutoff < float("inf"):
        atom_mask = jnp.ones(n, dtype=bool)
        if lj_switch_width > 0.0 and cutoff > lj_switch_width:
            rc = cutoff
            d = lj_switch_width
            rsw = rc - d
            volume = box_volume(box_pbc)
            inv_vol = 1.0 / volume
            atom_mask_f = atom_mask.astype(eps.dtype)
            sum_c6 = jnp.sum(jnp.sqrt(eps) * (sig**3) * atom_mask_f)
            sum_c12 = jnp.sum(jnp.sqrt(eps) * (sig**6) * atom_mask_f)

            I6_val = (
                10.0
                * (
                    (rc**2 + 4.0 * rc * rsw + rsw**2) * jnp.log(rc / rsw)
                    - 3.0 * (rc**2 - rsw**2)
                )
                / (d**5)
            )
            I12_val = (
                (rc + rsw)
                * (5.0 * rc**2 + 4.0 * rc * rsw + 5.0 * rsw**2)
                / (252.0 * (rc**6) * (rsw**6))
            )

            e_c12 = (8.0 * jnp.pi * inv_vol) * (sum_c12**2) * I12_val
            e_c6 = (8.0 * jnp.pi * inv_vol) * (sum_c6**2) * I6_val
            e_tail = e_c12 - e_c6
        else:
            e_tail = lj_dispersion_tail_energy(box_pbc, sig, eps, cutoff, atom_mask)
        tot = tot + e_tail

    return tot


def _eval_system_potential_energy(  # noqa: PLR0913
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    custom_forces: tuple[CompiledLeptonForce, ...],
    position_restraints: tuple[PositionRestraints, ...],
    global_params: dict[str, jnp.ndarray],
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
    cutoff: float,
    bond_force_groups: jnp.ndarray | None = None,
    angle_force_groups: jnp.ndarray | None = None,
    dihedral_force_groups: jnp.ndarray | None = None,
    cmap_force_groups: jnp.ndarray | None = None,
    nonbonded_method: str = "nocutoff",
    has_nonbonded: bool = True,
    use_dispersion_correction: bool = False,
    ewald_error_tolerance: float = 0.0005,
    rf_dielectric: float = 78.3,
    has_implicit_solvent: bool = False,
    solvent_dielectric: float = 78.5,
    solute_dielectric: float = 1.0,
    nonbonded_engine: str = "dense",
    pme_grid_dims: tuple[int, int, int] | None = None,
    use_pallas: bool = False,
    pallas_interpret: bool = False,
    lj_switch_width: float = 0.0,
    neighbor_idx: jnp.ndarray | None = None,
    use_pallas_pme: bool = False,
    custom_nonbonded_chunk_size: int = 128,
    tile_pair_nl: TilePairNeighborList | None = None,
    gbsa_tile_list: GBSATilePairList | None = None,
    is_triclinic: bool = False,
    gb_use_neighbor_list: bool = False,
    gbsa_surface_tension: float = 0.0054,
    pme_alpha: float | None = None,
) -> jnp.ndarray:
    """Compute total potential energy (in kcal/mol) for active force groups."""
    e = _eval_bonded_terms(
        pos,
        box,
        bundle,
        force_group_by_component,
        groups_mask,
        has_pbc,
        bond_force_groups=bond_force_groups,
        angle_force_groups=angle_force_groups,
        dihedral_force_groups=dihedral_force_groups,
        cmap_force_groups=cmap_force_groups,
        use_pallas=use_pallas,
        pallas_interpret=pallas_interpret,
    )
    e = e + _eval_nonbonded_terms(
        pos,
        box,
        bundle,
        force_group_by_component,
        groups_mask,
        has_pbc,
        cutoff,
        nonbonded_method=nonbonded_method,
        has_nonbonded=has_nonbonded,
        use_dispersion_correction=use_dispersion_correction,
        ewald_error_tolerance=ewald_error_tolerance,
        pme_alpha=pme_alpha,
        rf_dielectric=rf_dielectric,
        nonbonded_engine=nonbonded_engine,
        pme_grid_dims=pme_grid_dims,
        use_pallas=use_pallas,
        pallas_interpret=pallas_interpret,
        lj_switch_width=lj_switch_width,
        neighbor_idx=neighbor_idx,
        use_pallas_pme=use_pallas_pme,
        custom_nonbonded_chunk_size=custom_nonbonded_chunk_size,
        tile_pair_nl=tile_pair_nl,
        is_triclinic=is_triclinic,
    )

    # Position Restraints
    n = pos.shape[0]
    for pr in position_restraints:
        if ((groups_mask & (1 << int(pr.force_group))) != 0):
            idx = pr.indices
            valid = idx < n
            idx_s = jnp.where(valid, idx, 0)
            dx = pos[idx_s, 0] - pr.x0
            dy = pos[idx_s, 1] - pr.y0
            dz = pos[idx_s, 2] - pr.z0
            e_pr = 0.5 * (pr.k_x * dx**2 + pr.k_y * dy**2 + pr.k_z * dz**2)
            e = e + jnp.sum(jnp.where(valid, e_pr, 0.0))

    # Custom Forces
    for cf in custom_forces:
        if ((groups_mask & (1 << int(cf.force_group))) != 0):
            e = e + cf.energy_akma(pos, box, global_params)

    # GBSA-OBC2 Implicit Solvent
    g_gb = force_group_by_component.get("gbsa", 0)
    if ((groups_mask & (1 << int(g_gb))) != 0) and has_implicit_solvent:
        radii = (
            bundle.radii[:n]
            if hasattr(bundle, "radii") and bundle.radii is not None
            else bundle.sigmas[:n]
        )
        scaled_radii = (
            bundle.scaled_radii[:n]
            if hasattr(bundle, "scaled_radii") and bundle.scaled_radii is not None
            else None
        )
        charges = bundle.charges[:n]
        box_diag = (
            (jnp.diag(box) if box.ndim == 2 else box)
            if (has_pbc and box is not None and not is_triclinic)
            else None
        )
        use_pallas_gb = (
            (use_pallas or nonbonded_engine == "gbsa_tile")
            and not is_triclinic
            and pos.dtype == jnp.float32
        )
        if (
            use_pallas_gb
            and nonbonded_engine != "gbsa_tile"
            and gb_use_neighbor_list
            and neighbor_idx is not None
        ):
            e_gb_tot = cast(
                "jnp.ndarray",
                pallas_gb_ace_nl(
                    pos,
                    charges,
                    radii,
                    neighbor_idx,
                    scaled_radii=scaled_radii,
                    solvent_dielectric=solvent_dielectric,
                    solute_dielectric=solute_dielectric,
                    surface_tension=gbsa_surface_tension,
                    include_ace=bool(gbsa_surface_tension > 0.0),
                    box=box_diag,
                    interpret=pallas_interpret,
                    return_forces=False,
                ),
            )
            e = e + e_gb_tot
        elif use_pallas_gb and gbsa_tile_list is not None:
            sr_in = (
                scaled_radii
                if scaled_radii is not None
                else jnp.maximum(radii - 0.09, 0.0) * 0.85
            )
            e_gb_tot = cast(
                "jnp.ndarray",
                pallas_gbsa_tile(
                    pos,
                    charges,
                    radii,
                    sr_in,
                    tile_list=gbsa_tile_list,
                    box=box_diag,
                    solvent_dielectric=solvent_dielectric,
                    solute_dielectric=solute_dielectric,
                    surface_tension=gbsa_surface_tension,
                    include_gb=True,
                    include_ace=bool(gbsa_surface_tension > 0.0),
                    include_coulomb=False,
                    include_lj=False,
                    return_forces=False,
                    interpret=pallas_interpret,
                ),
            )
            e = e + e_gb_tot
        else:
            e_gb, born_radii = compute_gb_energy(
                pos,
                charges,
                radii,
                solvent_dielectric=solvent_dielectric,
                solute_dielectric=solute_dielectric,
                scaled_radii=scaled_radii,
            )
            if gbsa_surface_tension > 0.0:
                safe_br = jnp.maximum(born_radii, 1e-6)
                e_ace = (4.0 * jnp.pi * float(gbsa_surface_tension)) * jnp.sum(
                    ((radii + 1.4) ** 2) * ((radii / safe_br) ** 6)
                )
            else:
                e_ace = jnp.array(0.0, dtype=pos.dtype)
            e = e + e_gb + e_ace

    return e


@eqx.filter_jit
def _pure_compute_energy_and_forces(  # noqa: PLR0913
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    custom_forces: tuple[CompiledLeptonForce, ...],
    position_restraints: tuple[PositionRestraints, ...],
    global_params: dict[str, jnp.ndarray],
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
    cutoff: float,
    bond_force_groups: jnp.ndarray | None = None,
    angle_force_groups: jnp.ndarray | None = None,
    dihedral_force_groups: jnp.ndarray | None = None,
    cmap_force_groups: jnp.ndarray | None = None,
    nonbonded_method: str = "nocutoff",
    has_nonbonded: bool = True,
    use_dispersion_correction: bool = False,
    ewald_error_tolerance: float = 0.0005,
    rf_dielectric: float = 78.3,
    has_implicit_solvent: bool = False,
    solvent_dielectric: float = 78.5,
    solute_dielectric: float = 1.0,
    nonbonded_engine: str = "dense",
    pme_grid_dims: tuple[int, int, int] | None = None,
    use_pallas: bool = False,
    pallas_interpret: bool = False,
    lj_switch_width: float = 0.0,
    neighbor_idx: jnp.ndarray | None = None,
    use_pallas_pme: bool = False,
    custom_nonbonded_chunk_size: int = 128,
    tile_pair_nl: TilePairNeighborList | None = None,
    gbsa_tile_list: GBSATilePairList | None = None,
    is_triclinic: bool = False,
    mixed_precision: bool = False,
    gb_use_neighbor_list: bool = False,
    gbsa_surface_tension: float = 0.0054,
    pme_alpha: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure JIT evaluation of potential energy and negative gradient forces."""
    eval_pos = pos.astype(jnp.float32) if mixed_precision else pos
    eval_box = (
        box.astype(jnp.float32)
        if (mixed_precision and box is not None)
        else box
    )

    def e_fn(p: jnp.ndarray) -> jnp.ndarray:
        return _eval_system_potential_energy(
            p,
            eval_box,
            bundle,
            custom_forces,
            position_restraints,
            global_params,
            force_group_by_component,
            groups_mask,
            has_pbc,
            cutoff,
            bond_force_groups=bond_force_groups,
            angle_force_groups=angle_force_groups,
            dihedral_force_groups=dihedral_force_groups,
            cmap_force_groups=cmap_force_groups,
            nonbonded_method=nonbonded_method,
            has_nonbonded=has_nonbonded,
            use_dispersion_correction=use_dispersion_correction,
            ewald_error_tolerance=ewald_error_tolerance,
            rf_dielectric=rf_dielectric,
            has_implicit_solvent=has_implicit_solvent,
            solvent_dielectric=solvent_dielectric,
            solute_dielectric=solute_dielectric,
            nonbonded_engine=nonbonded_engine,
            pme_grid_dims=pme_grid_dims,
            use_pallas=use_pallas,
            pallas_interpret=pallas_interpret,
            lj_switch_width=lj_switch_width,
            neighbor_idx=neighbor_idx,
            use_pallas_pme=use_pallas_pme,
            custom_nonbonded_chunk_size=custom_nonbonded_chunk_size,
            tile_pair_nl=tile_pair_nl,
            gbsa_tile_list=gbsa_tile_list,
            is_triclinic=is_triclinic,
            gb_use_neighbor_list=gb_use_neighbor_list,
            gbsa_surface_tension=gbsa_surface_tension,
            pme_alpha=pme_alpha,
        )

    pe, grad = jax.value_and_grad(e_fn)(eval_pos)
    forces = -grad
    if mixed_precision:
        pe = pe.astype(pos.dtype)
        forces = forces.astype(pos.dtype)
    return pe, forces


@eqx.filter_jit
def _pure_step_rollout(  # noqa: PLR0913
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    custom_forces: tuple[CompiledLeptonForce, ...],
    position_restraints: tuple[PositionRestraints, ...],
    global_params: dict[str, jnp.ndarray],
    force_group_by_component: dict[str, int],
    key: jax.Array,
    dt: float,
    kT: float,
    gamma: float,
    inv_mass: jnp.ndarray,
    masses: jnp.ndarray,
    is_langevin: bool,
    num_steps: jax.Array,
    has_pbc: bool,
    cutoff: float,
    remove_com_motion: bool,
    constraints: ConstraintAlgorithm | None = None,
    virtual_sites: VirtualSites | None = None,
    bond_force_groups: jnp.ndarray | None = None,
    angle_force_groups: jnp.ndarray | None = None,
    dihedral_force_groups: jnp.ndarray | None = None,
    cmap_force_groups: jnp.ndarray | None = None,
    nonbonded_method: str = "nocutoff",
    has_nonbonded: bool = True,
    use_dispersion_correction: bool = False,
    ewald_error_tolerance: float = 0.0005,
    rf_dielectric: float = 78.3,
    has_implicit_solvent: bool = False,
    solvent_dielectric: float = 78.5,
    solute_dielectric: float = 1.0,
    nonbonded_engine: str = "dense",
    pme_grid_dims: tuple[int, int, int] | None = None,
    integrator_scheme: str = "auto",
    thermostat_override: str | None = None,
    use_pallas: bool = False,
    pallas_interpret: bool = False,
    lj_switch_width: float = 0.0,
    neighbor_idx: jnp.ndarray | None = None,
    use_pallas_pme: bool = False,
    custom_nonbonded_chunk_size: int = 128,
    tile_pair_nl: TilePairNeighborList | None = None,
    gbsa_tile_list: GBSATilePairList | None = None,
    is_triclinic: bool = False,
    mixed_precision: bool = False,
    gb_use_neighbor_list: bool = False,
    gbsa_surface_tension: float = 0.0054,
    cmm_frequency: int = 1,
    step_offset: jax.Array | None = None,
    n_dof: int | None = None,
    pme_alpha: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """Pure compiled multi-step rollout kernel via jax.lax.fori_loop."""
    c1 = jnp.exp(-gamma * dt) if is_langevin else 1.0
    c2 = jnp.sqrt(jnp.maximum(1.0 - c1 * c1, 0.0)) if is_langevin else 0.0
    eff_ndof = int(n_dof) if n_dof is not None else max(1, 3 * int(positions.shape[0]) - 3)
    base_step = jnp.asarray(0, dtype=jnp.int32) if step_offset is None else step_offset

    def compute_f(p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        p_recon = virtual_sites.reconstruct_positions(p) if virtual_sites is not None else p
        pe, f_raw = _pure_compute_energy_and_forces(
            p_recon,
            box,
            bundle,
            custom_forces,
            position_restraints,
            global_params,
            force_group_by_component,
            -1,
            has_pbc,
            cutoff,
            bond_force_groups=bond_force_groups,
            angle_force_groups=angle_force_groups,
            dihedral_force_groups=dihedral_force_groups,
            cmap_force_groups=cmap_force_groups,
            nonbonded_method=nonbonded_method,
            has_nonbonded=has_nonbonded,
            use_dispersion_correction=use_dispersion_correction,
            ewald_error_tolerance=ewald_error_tolerance,
            rf_dielectric=rf_dielectric,
            has_implicit_solvent=has_implicit_solvent,
            solvent_dielectric=solvent_dielectric,
            solute_dielectric=solute_dielectric,
            nonbonded_engine=nonbonded_engine,
            pme_grid_dims=pme_grid_dims,
            use_pallas=use_pallas,
            pallas_interpret=pallas_interpret,
            lj_switch_width=lj_switch_width,
            neighbor_idx=neighbor_idx,
            use_pallas_pme=use_pallas_pme,
            custom_nonbonded_chunk_size=custom_nonbonded_chunk_size,
            tile_pair_nl=tile_pair_nl,
            gbsa_tile_list=gbsa_tile_list,
            is_triclinic=is_triclinic,
            mixed_precision=mixed_precision,
            gb_use_neighbor_list=gb_use_neighbor_list,
            gbsa_surface_tension=gbsa_surface_tension,
            pme_alpha=pme_alpha,
        )
        f = (
            virtual_sites.redistribute_forces(f_raw, p_recon)
            if virtual_sites is not None
            else f_raw
        )
        return pe, f

    def step_body(i: int, val: tuple[jnp.ndarray, jnp.ndarray, jax.Array]):
        pos, vel, k = val

        # 1. Force evaluation at current positions
        _, f = compute_f(pos)

        scheme = str(integrator_scheme).lower()
        if scheme == "auto":
            scheme = "baoab" if is_langevin else "leapfrog"

        if scheme in ("baoab", "langevin", "lfmiddle"):
            # BAOAB scheme
            v = vel + 0.5 * dt * (f * inv_mass[:, None])
            r = pos + 0.5 * dt * v
            k, subk = jax.random.split(k)
            thermal_sigma = jnp.sqrt(kT * inv_mass)[:, None]
            noise = jax.random.normal(subk, shape=r.shape, dtype=r.dtype)
            v_noisy = c1 * v + c2 * thermal_sigma * noise
            v = jnp.where(inv_mass[:, None] > 1e-12, v_noisy, 0.0)
            r = r + 0.5 * dt * v
            if constraints is not None:
                r_c = constraints.apply_positions(pos, r, masses[:, None], box).astype(r.dtype)
                v = v + (r_c - r) / dt
                r = r_c
            if virtual_sites is not None:
                r = virtual_sites.reconstruct_positions(r)
            _, f_new = compute_f(r)
            v = v + 0.5 * dt * (f_new * inv_mass[:, None])
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)
        elif scheme == "verlet":
            # Velocity Verlet
            v_half = vel + 0.5 * dt * (f * inv_mass[:, None])
            r = pos + dt * v_half
            if constraints is not None:
                r_c = constraints.apply_positions(pos, r, masses[:, None], box).astype(r.dtype)
                v_half = v_half + (r_c - r) / dt
                r = r_c
            if virtual_sites is not None:
                r = virtual_sites.reconstruct_positions(r)
            _, f_new = compute_f(r)
            v = v_half + 0.5 * dt * (f_new * inv_mass[:, None])
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)
        elif scheme == "brownian":
            # Overdamped Langevin / Brownian dynamics
            k, subk = jax.random.split(k)
            noise = jax.random.normal(subk, shape=pos.shape, dtype=pos.dtype)
            drift = (f * inv_mass[:, None] / jnp.maximum(gamma, 1e-6)) * dt
            diff = jnp.sqrt(2.0 * kT * inv_mass[:, None] / jnp.maximum(gamma, 1e-6) * dt) * noise
            r = pos + drift + diff
            if constraints is not None:
                r_c = constraints.apply_positions(pos, r, masses[:, None], box).astype(r.dtype)
                r = r_c
            if virtual_sites is not None:
                r = virtual_sites.reconstruct_positions(r)
            v = (r - pos) / dt
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)
        elif scheme == "nose_hoover":
            # Nose-Hoover thermostat dynamics
            v_half = vel + 0.5 * dt * (f * inv_mass[:, None])
            ke2 = jnp.sum(masses[:, None] * (v_half ** 2))
            ndof = jnp.asarray(float(eff_ndof), dtype=pos.dtype)
            target_ke2 = ndof * kT
            tau_nh = 0.1 * dt * 100.0
            q_mass = target_ke2 * (tau_nh ** 2)
            dxi = (ke2 - target_ke2) * (dt / jnp.maximum(q_mass, 1e-6))
            scale_v = 1.0 - 0.5 * jnp.clip(dxi, -0.5, 0.5)
            v_half_scaled = v_half * scale_v
            r = pos + dt * v_half_scaled
            if constraints is not None:
                r_c = constraints.apply_positions(pos, r, masses[:, None], box).astype(r.dtype)
                v_half_scaled = v_half_scaled + (r_c - r) / dt
                r = r_c
            if virtual_sites is not None:
                r = virtual_sites.reconstruct_positions(r)
            _, f_new = compute_f(r)
            v = v_half_scaled + 0.5 * dt * (f_new * inv_mass[:, None])
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)
        else:
            # Leapfrog (standard VerletIntegrator)
            v = vel + dt * (f * inv_mass[:, None])
            r = pos + dt * v
            if constraints is not None:
                r_c = constraints.apply_positions(pos, r, masses[:, None], box).astype(r.dtype)
                v = v + (r_c - r) / dt
                r = r_c
            if virtual_sites is not None:
                r = virtual_sites.reconstruct_positions(r)
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)

        # Apply exact Bussi-Donadio-Parrinello CSVR thermostat override if requested (BUG-OMM3)
        if thermostat_override is not None:
            t_str = str(thermostat_override).lower()
            if "csvr" in t_str or "bussi" in t_str:
                ke2 = jnp.sum(masses[:, None] * (v ** 2))
                ndof = jnp.asarray(float(eff_ndof), dtype=pos.dtype)
                k, subk1, subk2 = jax.random.split(k, 3)
                r_gauss = jax.random.normal(subk1, dtype=pos.dtype)
                s_chi2 = (
                    jnp.sum(jax.random.normal(subk2, (eff_ndof - 1,), dtype=pos.dtype) ** 2)
                    if eff_ndof > 1
                    else jnp.asarray(0.0, dtype=pos.dtype)
                )
                tau_csvr = jnp.asarray(10.0 * dt, dtype=pos.dtype)
                c_csvr = jnp.exp(-dt / tau_csvr).astype(pos.dtype)
                ke2_safe = jnp.maximum(ke2, 1e-12)
                kt_d = jnp.asarray(kT, dtype=pos.dtype)
                alpha_sq = (
                    c_csvr
                    + (1.0 - c_csvr) * (kt_d / ke2_safe) * (r_gauss**2 + s_chi2)
                    + 2.0
                    * r_gauss
                    * jnp.sqrt(jnp.maximum(c_csvr * (1.0 - c_csvr) * (kt_d / ke2_safe), 0.0))
                )
                sign_term = r_gauss + jnp.sqrt(
                    jnp.maximum(
                        c_csvr * ke2_safe / jnp.maximum((1.0 - c_csvr) * kt_d, 1e-12),
                        0.0,
                    )
                )
                alpha_sign = jnp.where(sign_term >= 0.0, 1.0, -1.0)
                alpha = (alpha_sign * jnp.sqrt(jnp.maximum(alpha_sq, 0.0))).astype(vel.dtype)
                v = jnp.where(ke2 > 1e-12, (v * alpha).astype(vel.dtype), v)

        # Remove center of mass motion periodically when requested (only when >1 mobile atom)
        if remove_com_motion and cmm_frequency > 0:
            active_m_mask = inv_mass > 1e-12
            n_active_m = jnp.sum(active_m_mask)
            total_m = jnp.sum(jnp.where(active_m_mask, masses, 0.0))
            v_com = (
                jnp.sum(jnp.where(active_m_mask[:, None], v * masses[:, None], 0.0), axis=0)
                / jnp.maximum(total_m, 1e-12)
            )
            v_no_com = jnp.where(
                active_m_mask[:, None],
                jnp.where(n_active_m > 1, v - v_com, v),
                0.0,
            ).astype(vel.dtype)
            if cmm_frequency == 1:
                v = v_no_com
            else:
                do_cmm = ((base_step + i + 1) % int(cmm_frequency)) == 0
                v = jnp.where(do_cmm, v_no_com, v)

        r = r.astype(pos.dtype)
        v = v.astype(vel.dtype)
        return r, v, k

    init_val = (positions, velocities, key)
    final_pos, final_vel, final_key = jax.lax.fori_loop(0, num_steps, step_body, init_val)
    if virtual_sites is not None:
        final_pos = virtual_sites.reconstruct_positions(final_pos)
    return final_pos, final_vel, final_key


# =============================================================================
# 3. ProlixContext
# =============================================================================


def _is_same_openmm_force(f1: Any, f2: Any) -> bool:
    """Check if two OpenMM Force objects refer to the same underlying C++ force."""
    if f1 is f2 or f1 == f2:
        return True
    this1 = getattr(f1, "this", None)
    this2 = getattr(f2, "this", None)
    return this1 is not None and this1 == this2


class ProlixContext:
    """Drop-in OpenMM Context replacement executing on JAX and Pallas kernels."""

    def __init__(
        self,
        system: openmm.System,
        integrator: openmm.Integrator,
        platform: openmm.Platform | None = None,
        platformProperties: dict[str, str] | None = None,
        config: ProlixOpenMMConfig | None = None,
    ) -> None:
        self._system = system
        self._integrator = integrator

        # Resolve config and platform shim
        if isinstance(platform, ProlixPlatform):
            self._platform = platform
            if config is not None:
                self._config = config
            else:
                self._config = platform.create_config(platformProperties)
        elif isinstance(platform, openmm.Platform):
            self._platform = ProlixPlatform()
            self._config = config or self._platform.create_config(platformProperties)
        else:
            self._platform = ProlixPlatform()
            if config is not None:
                self._config = config
            else:
                self._config = parse_platform_properties(platformProperties or {})

        # Convert OpenMM System to Prolix representation
        self._converted_system: ConvertedOpenMMSystem = from_openmm_system(
            system, integrator, self._config
        )
        self._bundle: MolecularBundle = self._converted_system.bundle
        self._custom_forces: tuple[CompiledLeptonForce, ...] = tuple(
            self._converted_system.custom_forces
        )
        self._position_restraints: tuple[PositionRestraints, ...] = tuple(
            self._converted_system.position_restraints
        )

        # Particle counts & allocation
        self._n_particles = self._converted_system.n_real_particles
        self._precision = str(self._config.precision)
        if self._precision in ("double", "mixed"):
            jax.config.update("jax_enable_x64", True)
        self._dtype = (
            jnp.float64 if self._precision in ("double", "mixed") else jnp.float32
        )
        self._mixed_precision = self._precision == "mixed"
        self._positions = jnp.zeros((self._n_particles, 3), dtype=self._dtype)
        self._velocities = jnp.zeros((self._n_particles, 3), dtype=self._dtype)
        self._velocities_initialized = False
        if self._dtype == jnp.float64:
            self._bundle = eqx.tree_at(
                lambda b: b.masses, self._bundle, self._bundle.masses.astype(jnp.float64)
            )

        # Wire configuration knobs and engine attributes
        self._use_pallas = bool(self._config.use_pallas)
        self._use_pallas_pme = bool(self._config.use_pallas_pme)
        self._pallas_interpret = bool(self._config.pallas_interpret)
        self._nb_k_splits = (
            int(self._config.nb_k_splits)
            if self._config.nb_k_splits is not None
            else None
        )
        self._tp_group = (
            int(self._config.tp_group)
            if self._config.tp_group is not None
            else None
        )
        self._gb_use_neighbor_list = bool(self._config.gb_use_neighbor_list)
        self._deterministic_forces = bool(self._config.deterministic_forces)
        self._disable_pme_stream = bool(self._config.disable_pme_stream)
        self._device_index = (
            str(self._config.device_index)
            if self._config.device_index is not None
            else None
        )
        self._custom_force_mode = str(self._config.custom_force_mode)
        self._virtual_site_mode = str(self._config.virtual_site_mode)
        self._cmm_frequency = self._config.cmm_frequency
        self._anisotropic_barostat = bool(self._config.anisotropic_barostat)
        self._execution_mode = str(self._config.execution_mode)
        self._integrator_scheme = str(self._config.integrator_scheme)
        self._thermostat_override = (
            str(self._config.thermostat_override)
            if self._config.thermostat_override is not None
            else None
        )
        self._is_triclinic = bool(self._converted_system.is_triclinic)
        self._has_finite_cutoff = (
            0.0 < float(self._converted_system.nonbonded_cutoff) < float("inf")
        )

        cfg_nb_engine = str(self._config.nonbonded_engine)
        if cfg_nb_engine in (
            "morton_tile_pairs",
            "gbsa_tile",
            "neighbor_list",
            "flash",
            "dense",
        ):
            self._nonbonded_engine = cfg_nb_engine
        elif (
            self._use_pallas
            and self._converted_system.has_nonbonded
            and self._has_finite_cutoff
            and not self._is_triclinic
            and self._n_particles >= 256
            and self._precision in ("single", "mixed")
        ):
            self._nonbonded_engine = "morton_tile_pairs"
        elif (
            self._use_pallas
            and self._converted_system.has_implicit_solvent
            and not self._has_finite_cutoff
            and not self._is_triclinic
            and self._n_particles >= 64
            and self._precision in ("single", "mixed")
        ):
            self._nonbonded_engine = "gbsa_tile"
        elif self._n_particles >= 4000 and self._has_finite_cutoff:
            self._nonbonded_engine = "neighbor_list"
        elif self._use_pallas:
            self._nonbonded_engine = "pallas"
        else:
            self._nonbonded_engine = "dense"

        self._neighbor_fn = None
        self._neighbor = None
        self._neighbor_idx = None
        self._tile_pair_nl: TilePairNeighborList | None = None
        self._max_tile_pairs: int | None = None
        self._gbsa_tile_list: GBSATilePairList | None = None

        # Adaptive Monte Carlo barostat step-size tracking (OpenMM MonteCarloBarostatImpl)
        self._mc_volume_scale: float = 0.01
        self._mc_num_attempted: int = 0
        self._mc_num_accepted: int = 0

        if self._converted_system.has_pbc and self._converted_system.bundle.box is not None:
            self._box_vectors = self._converted_system.bundle.box.astype(self._dtype)
        else:
            try:
                raw_bv = system.getDefaultPeriodicBoxVectors()
                box_akma = np.asarray([to_akma(v, "length") for v in raw_bv], dtype=np.float64)
                if abs(float(np.linalg.det(box_akma))) > 1e-6:
                    self._box_vectors = reduce_triclinic_box_vectors(
                        jnp.asarray(box_akma, dtype=self._dtype)
                    )
                    off_diag = np.abs(np.asarray(self._box_vectors) - np.diag(np.diag(np.asarray(self._box_vectors))))
                    self._is_triclinic = bool(np.max(off_diag) > 1e-6)
                else:
                    self._box_vectors = None
            except Exception:
                self._box_vectors = None
        self._time = 0.0  # τ_AKMA
        self._step_count = 0
        self._parameters = dict(self._converted_system.default_parameters)
        self._prng_key = jax.random.PRNGKey(self._config.deterministic_seed)

        # Constraints infrastructure
        b = self._converted_system.bundle
        water_indices = (
            b.water_indices[: int(b.n_waters)]
            if (
                b.water_indices is not None
                and b.n_waters is not None
                and int(b.n_waters) > 0
            )
            else None
        )
        constraint_pairs = (
            b.constraint_pairs[: int(b.n_constraints)]
            if (
                b.constraint_pairs is not None
                and b.n_constraints is not None
                and int(b.n_constraints) > 0
            )
            else None
        )
        constraint_lengths = (
            b.constraint_lengths[: int(b.n_constraints)]
            if (
                b.constraint_lengths is not None
                and b.n_constraints is not None
                and int(b.n_constraints) > 0
            )
            else None
        )
        self._constraints: ConstraintAlgorithm = make_constraint(
            water_indices=water_indices,
            constraint_pairs=constraint_pairs,
            constraint_lengths=constraint_lengths,
            r_OH=self._converted_system.system_metadata.get("water_r_OH_akma"),
            r_HH=self._converted_system.system_metadata.get("water_r_HH_akma"),
            settle_velocity_tol=self._config.settle_velocity_tol,
            shake_iterations=self._config.shake_iterations,
            constraint_tolerance=self._config.constraint_tolerance,
        )

        # Molecular topology graph for getMolecules()
        self._molecules = self._build_molecules()

        # Wire integrator link
        ig_init = cast("Any", integrator)
        if hasattr(ig_init, "_prolix_context"):
            ig_init._prolix_context = self
        ig_init.step = self.step

    def _effective_cmm_frequency(self) -> int:
        """Resolve effective CMMotionRemover frequency (0 disables COM removal)."""
        if not self._config.remove_com_motion or self._n_particles <= 1:
            return 0
        if self._cmm_frequency is not None:
            return max(0, int(self._cmm_frequency))
        if self._converted_system.remove_com_motion:
            return max(1, int(self._converted_system.com_motion_frequency))
        return 1

    def _effective_remove_com(self) -> bool:
        """Return True if center-of-mass motion removal is active."""
        return self._effective_cmm_frequency() > 0

    def _effective_pme_alpha(self) -> float | None:
        """Resolve effective Ewald alpha (Å⁻¹), honoring explicit NonbondedForce.setPMEParameters."""
        if str(self._converted_system.nonbonded_method).lower() not in ("pme", "spme"):
            return None
        if (
            self._converted_system.explicit_pme_alpha is not None
            and float(self._converted_system.explicit_pme_alpha) > 0.0
        ):
            return float(self._converted_system.explicit_pme_alpha)
        ewald_tol = (
            self._converted_system.ewald_error_tolerance
            if self._converted_system.ewald_error_tolerance is not None
            else self._config.ewald_error_tolerance
        )
        cutoff = float(self._converted_system.nonbonded_cutoff)
        if 0.0 < cutoff < float("inf"):
            return float((1.0 / cutoff) * math.sqrt(-math.log(2.0 * float(ewald_tol))))
        if self._bundle.pme_alpha is not None and float(self._bundle.pme_alpha) > 0.0:
            return float(self._bundle.pme_alpha)
        return 0.34

    def _effective_integrator_scheme(self) -> str:
        """Resolve canonical integrator scheme string for rollout & KE evaluation."""
        scheme = str(self._integrator_scheme).lower()
        if scheme != "auto":
            return scheme
        if isinstance(
            self._integrator, (openmm.LangevinMiddleIntegrator, openmm.LangevinIntegrator)
        ):
            return "baoab"
        if hasattr(openmm, "BrownianIntegrator") and isinstance(
            self._integrator, openmm.BrownianIntegrator
        ):
            return "brownian"
        if hasattr(openmm, "NoseHooverIntegrator") and isinstance(
            self._integrator, openmm.NoseHooverIntegrator
        ):
            return "nose_hoover"
        return "leapfrog"

    def _active_ndof(self) -> int:
        """Compute physical translational degrees of freedom accounting for virtual sites & constraints."""
        safe_inv_m = np.asarray(self._converted_system.inv_mass[: self._n_particles])
        active_mask = safe_inv_m > 1e-12
        if self._converted_system.virtual_sites is not None:
            vs_idx = np.asarray(self._converted_system.virtual_sites.order, dtype=np.int32)
            valid_vs = vs_idx[(vs_idx >= 0) & (vs_idx < self._n_particles)]
            active_mask[valid_vs] = False
        n_active = int(np.sum(active_mask))
        n_waters = (
            int(self._bundle.n_waters)
            if (
                self._bundle.water_indices is not None
                and self._bundle.n_waters is not None
            )
            else 0
        )
        n_solute_constraints = (
            int(self._bundle.n_constraints)
            if (
                self._bundle.constraint_pairs is not None
                and self._bundle.n_constraints is not None
            )
            else 0
        )
        com_dof = 3 if (self._effective_remove_com() and n_active > 1) else 0
        return max(1, 3 * n_active - 3 * n_waters - n_solute_constraints - com_dof)

    def _should_use_neighbor_list(self) -> bool:
        if str(self._config.nonbonded_engine) in ("neighbor_list", "flash"):
            return True
        if not self._has_finite_cutoff:
            return False
        return (
            self._nonbonded_engine in ("neighbor_list", "flash", "morton_tile_pairs")
            or self._n_particles >= 4000
            or (
                self._converted_system.has_implicit_solvent
                and self._gb_use_neighbor_list
            )
        )

    def _can_use_gbsa_tiles(self) -> bool:
        return (
            self._converted_system.has_implicit_solvent
            and not self._is_triclinic
            and self._precision in ("single", "mixed")
            and (self._use_pallas or self._nonbonded_engine == "gbsa_tile")
        )

    def _rebuild_gbsa_tile_list(self, pos_f32: jnp.ndarray) -> None:
        """Build or refresh 32x32 symmetric `GBSATilePairList` on host for `pallas_gbsa_tile`."""
        if not self._can_use_gbsa_tiles():
            self._gbsa_tile_list = None
            return
        box_diag = (
            (
                jnp.diag(self._box_vectors)
                if self._box_vectors.ndim == 2
                else self._box_vectors
            ).astype(jnp.float32)
            if (self._converted_system.has_pbc and self._box_vectors is not None)
            else None
        )
        use_gb_cutoff = bool(self._gb_use_neighbor_list and self._has_finite_cutoff)
        self._gbsa_tile_list = build_gbsa_tile_pair_list(
            pos_f32,
            box=box_diag,
            cutoff=(
                float(self._converted_system.nonbonded_cutoff)
                if use_gb_cutoff
                else None
            ),
            neighbor_idx=self._neighbor_idx if use_gb_cutoff else None,
        )

    def _can_use_tile_pairs(self) -> bool:
        nb_method = str(self._converted_system.nonbonded_method).lower()
        is_rf = (
            nb_method
            in ("cutoffperiodic", "cutoff_periodic", "reactionfield", "reaction_field", "rf")
            and self._converted_system.has_pbc
        )
        return (
            self._converted_system.has_nonbonded
            and self._has_finite_cutoff
            and not self._is_triclinic
            and not is_rf
            and self._precision in ("single", "mixed")
            and (
                self._nonbonded_engine == "morton_tile_pairs"
                or (
                    self._use_pallas
                    and self._neighbor_idx is not None
                    and self._n_particles >= 256
                )
            )
        )

    def _rebuild_tile_pair_nl(self, pos_f32: jnp.ndarray, cutoff: float) -> None:
        """Build or refresh Morton-sorted 32x32 TilePairNeighborList with static capacity."""
        if self._neighbor_idx is None or not self._can_use_tile_pairs():
            self._tile_pair_nl = None
            return

        n = self._n_particles
        excl_idx = (
            self._bundle.excl_dense_indices[:n]
            if self._bundle.excl_dense_indices is not None
            else None
        )
        excl_vdw = (
            self._bundle.excl_dense_scales_vdw[:n]
            if self._bundle.excl_dense_scales_vdw is not None
            else None
        )
        excl_elec = (
            self._bundle.excl_dense_scales_elec[:n]
            if self._bundle.excl_dense_scales_elec is not None
            else None
        )
        box_diag = (
            (
                jnp.diag(self._box_vectors)
                if self._box_vectors.ndim == 2
                else self._box_vectors
            ).astype(jnp.float32)
            if (self._converted_system.has_pbc and self._box_vectors is not None)
            else None
        )
        eff_cutoff = cutoff if cutoff < float("inf") else None
        pos_for_tp = (
            jnp.mod(pos_f32, box_diag[None, :]) if box_diag is not None else pos_f32
        )

        if self._max_tile_pairs is None:
            tp_probe = build_tile_pair_neighbor_list(
                self._neighbor_idx,
                excl_indices=excl_idx,
                excl_scales_vdw=excl_vdw,
                excl_scales_elec=excl_elec,
                positions=pos_for_tp,
                box=box_diag,
                cutoff=eff_cutoff,
                morton_sort=True,
            )
            raw_cap = int(tp_probe.tile_pairs.shape[0])
            n_blocks = max(1, (n + 31) // 32)
            max_ut = (n_blocks * (n_blocks + 1)) // 2
            padded_cap = max(64, ((math.ceil(raw_cap * 1.25) + 63) // 64) * 64)
            self._max_tile_pairs = min(max_ut, max(raw_cap, padded_cap))
            if self._max_tile_pairs == raw_cap:
                self._tile_pair_nl = tp_probe
                return

        self._tile_pair_nl = build_tile_pair_neighbor_list(
            self._neighbor_idx,
            excl_indices=excl_idx,
            excl_scales_vdw=excl_vdw,
            excl_scales_elec=excl_elec,
            max_tile_pairs=self._max_tile_pairs,
            positions=pos_for_tp,
            box=box_diag,
            cutoff=eff_cutoff,
            morton_sort=True,
        )

    def _init_neighbor_list(self, pos: jnp.ndarray | None = None) -> None:
        """Construct genuine NeighborList (and Pallas v3.5 TilePairNeighborList / GBSATilePairList) for sparse evaluations."""
        if pos is None:
            pos = self._positions[: self._n_particles]
        pos_f32 = pos.astype(jnp.float32)

        if not self._should_use_neighbor_list():
            self._neighbor_fn = None
            self._neighbor = None
            self._neighbor_idx = None
            self._tile_pair_nl = None
            self._rebuild_gbsa_tile_list(pos_f32)
            return

        cutoff = float(self._converted_system.nonbonded_cutoff)
        if cutoff <= 0.0 or cutoff >= float("inf"):
            if pos_f32.shape[0] > 0:
                span = float(
                    jnp.max(jnp.linalg.norm(pos_f32 - jnp.min(pos_f32, axis=0), axis=-1))
                )
                cutoff = max(9.0, span + float(self._config.nl_skin) + 2.0)
            else:
                cutoff = 9.0

        if (
            self._converted_system.has_pbc
            and self._box_vectors is not None
            and not self._is_triclinic
        ):
            box_size = (
                jnp.diag(self._box_vectors)
                if self._box_vectors.ndim == 2
                else self._box_vectors
            ).astype(jnp.float32)
            disp_fn, _ = space.periodic(box_size)
        else:
            box_size = (
                jnp.max(pos_f32, axis=0) - jnp.min(pos_f32, axis=0) + 2.0 * cutoff + 10.0
            ).astype(jnp.float32)
            disp_fn, _ = space.free()

        cap_mult = float(self._config.nl_capacity_multiplier)
        nl_fn = cast(
            "Any",
            make_neighbor_list_fn(
                disp_fn,
                box_size=box_size,
                cutoff=cutoff,
                capacity_multiplier=cap_mult,
                disable_cell_list=bool(self._config.disable_cell_list),
                dr_threshold=float(self._config.nl_skin),
                use_pallas_nl=bool(self._config.use_pallas),
            ),
        )
        self._neighbor_fn = nl_fn
        self._neighbor = nl_fn.allocate(pos_f32)
        if getattr(self._neighbor, "did_buffer_overflow", False):
            nl_fn = cast(
                "Any",
                make_neighbor_list_fn(
                    disp_fn,
                    box_size=box_size,
                    cutoff=cutoff,
                    capacity_multiplier=cap_mult * 2.0,
                    disable_cell_list=bool(self._config.disable_cell_list),
                    dr_threshold=float(self._config.nl_skin),
                    use_pallas_nl=bool(self._config.use_pallas),
                ),
            )
            self._neighbor_fn = nl_fn
            self._neighbor = nl_fn.allocate(pos_f32)
        self._neighbor_idx = self._neighbor.idx
        self._rebuild_tile_pair_nl(pos_f32, cutoff)
        self._rebuild_gbsa_tile_list(pos_f32)

    def create_ensemble_plan(self) -> EnsemblePlan:
        """Create an optimized Pallas v3.5 EnsemblePlan for this system."""
        plan_cls = _ensemble_plan_mod.EnsemblePlan
        candidate_kwargs: dict[str, Any] = {
            "use_pallas": self._use_pallas,
            "use_pallas_pme": self._use_pallas_pme,
            "precision": self._precision,
            "nonbonded_engine": self._nonbonded_engine,
            "nb_k_splits": self._nb_k_splits,
            "tp_group": self._tp_group,
            "cmm_frequency": self._effective_cmm_frequency(),
            "deterministic_forces": self._deterministic_forces,
            "gb_use_neighbor_list": self._gb_use_neighbor_list,
        }
        try:
            sig = inspect.signature(plan_cls.from_bundle)
            accepts_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not accepts_varkw:
                candidate_kwargs = {
                    k: v for k, v in candidate_kwargs.items() if k in sig.parameters
                }
        except (TypeError, ValueError):
            candidate_kwargs = {
                "use_pallas": self._use_pallas,
                "use_pallas_pme": self._use_pallas_pme,
            }
        return plan_cls.from_bundle(self._bundle, **candidate_kwargs)

    # --- Core Getters ---
    def getSystem(self) -> openmm.System:  # noqa: N802
        return self._system

    def getIntegrator(self) -> openmm.Integrator:  # noqa: N802
        return self._integrator

    def getPlatform(self) -> openmm.Platform:  # noqa: N802
        return cast("openmm.Platform", self._platform)

    def getMolecules(self) -> tuple[tuple[int, ...], ...]:  # noqa: N802
        return self._molecules

    # --- Step Rollout ---
    def step(self, steps: int) -> None:
        """Advance simulation rollout by `steps` steps."""
        if steps <= 0:
            return

        # Extract integrator properties in AKMA
        ig_any = cast("Any", self._integrator)
        dt_akma = 2.0 * 1e-3 * PS_TO_TAU_AKMA  # default 2 fs
        if hasattr(ig_any, "getStepSize"):
            dt_ps = ig_any.getStepSize().value_in_unit(unit.picosecond)
            dt_akma = dt_ps * PS_TO_TAU_AKMA

        eff_scheme = self._effective_integrator_scheme()
        is_langevin = eff_scheme in ("baoab", "langevin", "lfmiddle")
        temp_k = 300.0
        if hasattr(ig_any, "getTemperature"):
            temp_k = ig_any.getTemperature().value_in_unit(unit.kelvin)
        kT_akma = BOLTZMANN_KCAL * temp_k

        gamma_akma = 1.0 * TAU_AKMA_TO_PS  # default 1 ps^-1
        if hasattr(ig_any, "getFriction"):
            gamma_ps = ig_any.getFriction().value_in_unit(unit.picosecond**-1)
            gamma_akma = gamma_ps * TAU_AKMA_TO_PS

        masses = self._bundle.masses[: self._n_particles]
        inv_m = jnp.where(masses > 1e-12, 1.0 / masses, 0.0)

        # Barostat resolution
        baro_info = self._converted_system.barostat
        baro_mode = self._config.barostat_mode
        if baro_mode == "auto":
            baro_mode = "mc" if baro_info is not None else None

        has_barostat = (
            baro_info is not None
            and baro_mode in ("mc", "scr")
            and (self._converted_system.has_pbc or self._box_vectors is not None)
        )
        baro_freq = (
            self._config.barostat_frequency_override
            if self._config.barostat_frequency_override is not None
            else (baro_info.get("frequency", 25) if baro_info else 25)
        )
        nl_freq = int(self._config.nl_update_every)
        eff_cmm = self._effective_cmm_frequency()
        eff_remove_com = eff_cmm > 0
        eff_ndof = self._active_ndof()

        remaining = steps
        chunk_size = self._config.max_steps_per_scan

        while remaining > 0:
            chunk = min(remaining, chunk_size)
            if has_barostat:
                steps_to_baro = baro_freq - (self._step_count % baro_freq)
                if steps_to_baro == 0:
                    steps_to_baro = baro_freq
                chunk = min(chunk, steps_to_baro)
            if self._neighbor is not None and nl_freq > 0:
                steps_to_nl = nl_freq - (self._step_count % nl_freq)
                if steps_to_nl == 0:
                    steps_to_nl = nl_freq
                chunk = min(chunk, steps_to_nl)

            gp = self._get_dynamic_global_params()
            ewald_tol = (
                self._converted_system.ewald_error_tolerance
                if self._converted_system.ewald_error_tolerance is not None
                else self._config.ewald_error_tolerance
            )
            self._positions, self._velocities, self._prng_key = _pure_step_rollout(
                self._positions,
                self._velocities,
                self._box_vectors,
                self._bundle,
                self._custom_forces,
                self._position_restraints,
                gp,
                self._converted_system.force_group_by_component,
                self._prng_key,
                dt_akma,
                kT_akma,
                gamma_akma,
                inv_m,
                masses,
                is_langevin,
                jnp.asarray(chunk, dtype=jnp.int32),
                self._converted_system.has_pbc,
                self._converted_system.nonbonded_cutoff,
                eff_remove_com,
                self._constraints,
                self._converted_system.virtual_sites,
                self._converted_system.bond_force_groups,
                self._converted_system.angle_force_groups,
                self._converted_system.dihedral_force_groups,
                self._converted_system.cmap_force_groups,
                self._converted_system.nonbonded_method,
                self._converted_system.has_nonbonded,
                self._config.use_dispersion_correction and self._converted_system.use_dispersion_correction,
                ewald_tol,
                self._converted_system.reaction_field_dielectric,
                self._converted_system.has_implicit_solvent,
                self._converted_system.solvent_dielectric,
                self._converted_system.solute_dielectric,
                self._nonbonded_engine,
                self._converted_system.pme_grid_dims,
                integrator_scheme=eff_scheme,
                thermostat_override=self._config.thermostat_override,
                use_pallas=self._config.use_pallas,
                pallas_interpret=self._config.pallas_interpret,
                lj_switch_width=self._converted_system.lj_switch_width,
                neighbor_idx=self._neighbor_idx,
                use_pallas_pme=self._config.use_pallas_pme,
                custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
                tile_pair_nl=self._tile_pair_nl,
                gbsa_tile_list=self._gbsa_tile_list,
                is_triclinic=self._is_triclinic,
                mixed_precision=self._mixed_precision,
                gb_use_neighbor_list=self._gb_use_neighbor_list,
                gbsa_surface_tension=self._converted_system.gbsa_surface_tension,
                cmm_frequency=eff_cmm,
                step_offset=jnp.asarray(self._step_count, dtype=jnp.int32),
                n_dof=eff_ndof,
                pme_alpha=self._effective_pme_alpha(),
            )
            remaining -= chunk
            self._step_count += chunk
            self._time += chunk * dt_akma

            # Update neighbor list (and Morton / GBSA tile-pair lists) periodically if allocated
            if (
                self._neighbor is not None
                and nl_freq > 0
                and (self._step_count % nl_freq == 0)
            ):
                pos_f32 = self._positions[: self._n_particles].astype(jnp.float32)
                self._neighbor = self._neighbor.update(pos_f32)
                if getattr(self._neighbor, "did_buffer_overflow", False):
                    self._init_neighbor_list(self._positions[: self._n_particles])
                else:
                    self._neighbor_idx = self._neighbor.idx
                    cutoff = float(self._converted_system.nonbonded_cutoff)
                    if cutoff <= 0.0 or cutoff >= float("inf"):
                        cutoff = 9.0
                    self._rebuild_tile_pair_nl(pos_f32, cutoff)
                    self._rebuild_gbsa_tile_list(pos_f32)

            # Reconstruct virtual sites & apply constraints
            self.applyConstraints()

            # Attempt barostat volume move at frequency boundary
            if (
                has_barostat
                and baro_mode is not None
                and (self._step_count % baro_freq == 0)
            ):
                self._apply_barostat_step(
                    str(baro_mode), baro_info, chunk * dt_akma, kT_akma
                )

    def _get_dynamic_global_params(self) -> dict[str, jnp.ndarray]:
        """Convert current dynamic context parameters to JAX arrays."""
        return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in self._parameters.items()}

    def _apply_barostat_step(
        self,
        baro_mode: str,
        baro_info: dict[str, Any],
        dt_chunk_akma: float,
        kT_akma: float,
    ) -> None:
        """Dispatch barostat update to MC or SCR engine."""
        if baro_mode == "mc":
            self._apply_mc_barostat(baro_info, kT_akma)
        elif baro_mode == "scr":
            self._apply_scr_barostat(baro_info, dt_chunk_akma, kT_akma)

    def _unwrap_molecules_and_com(
        self, pos_akma: np.ndarray, box_akma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unwrap intramolecular coordinates wrt molecule anchor across PBC and return per-atom molecule COMs."""
        pos_unwrapped = pos_akma.copy()
        com_per_atom = pos_akma.copy()
        box_jnp = jnp.asarray(box_akma, dtype=jnp.float64)
        for mol in self._molecules:
            if not mol:
                continue
            mol_idx = list(mol)
            if len(mol_idx) > 1 and self._converted_system.has_pbc:
                anchor = pos_akma[mol_idx[0]]
                dr_intra = np.asarray(
                    _minimum_image_dr(
                        jnp.asarray(pos_akma[mol_idx] - anchor, dtype=jnp.float64),
                        box_jnp,
                        True,
                    ),
                    dtype=np.float64,
                )
                mol_coords = anchor + dr_intra
                pos_unwrapped[mol_idx] = mol_coords
            else:
                mol_coords = pos_akma[mol_idx]
            com = np.mean(mol_coords, axis=0)
            com_per_atom[mol_idx] = com
        return pos_unwrapped, com_per_atom

    def _apply_mc_barostat(
        self,
        baro_info: dict[str, Any],
        kT_akma: float,
    ) -> None:
        """Apply Monte Carlo volume move with PBC-unwrapped molecular COM scaling and OpenMM adaptive step size."""
        if self._box_vectors is None:
            return

        current_box = np.asarray(self._box_vectors, dtype=np.float64)
        vol_current = abs(float(np.linalg.det(current_box)))
        if vol_current <= 0:
            return

        baro_type = str(baro_info.get("type", "MonteCarloBarostat"))
        target_p_akma = float(baro_info["pressure_akma"])
        n_mol = len(self._molecules) or self._n_particles
        delta_v = float(self._mc_volume_scale)

        self._prng_key, subkey_scale, subkey_axis, subkey_accept = jax.random.split(
            self._prng_key, 4
        )
        u_rand = float(jax.random.uniform(subkey_scale, minval=-1.0, maxval=1.0))

        surface_tension_work = 0.0
        if baro_type == "MonteCarloAnisotropicBarostat":
            scale_axes = baro_info.get("scale_axes", (True, True, True))
            active_axes = [ax for ax in range(3) if scale_axes[ax]]
            if not active_axes:
                return
            ax_choice = active_axes[
                int(jax.random.randint(subkey_axis, (), 0, len(active_axes)))
            ]
            scale_ax = 1.0 + delta_v * u_rand
            if scale_ax <= 0:
                return
            axis_scales = np.ones(3, dtype=np.float64)
            axis_scales[ax_choice] = scale_ax
            scale = scale_ax
            p_xyz = baro_info.get(
                "pressure_xyz_akma", (target_p_akma, target_p_akma, target_p_akma)
            )
            target_p_akma = float(p_xyz[ax_choice])
            new_box = current_box * axis_scales[:, None]
        elif baro_type == "MonteCarloMembraneBarostat":
            scale_xy = 1.0 + delta_v * u_rand
            if scale_xy <= 0:
                return
            s_xy = math.sqrt(scale_xy)
            z_const_vol = (
                int(baro_info.get("z_mode", 0))
                == int(openmm.MonteCarloMembraneBarostat.ConstantVolume)
            )
            s_z = (1.0 / scale_xy) if z_const_vol else 1.0
            axis_scales = np.array([s_xy, s_xy, s_z], dtype=np.float64)
            scale = float(axis_scales[0] * axis_scales[1] * axis_scales[2])
            new_box = current_box * axis_scales[:, None]
            area_curr = abs(float(current_box[0, 0] * current_box[1, 1]))
            area_new = abs(float(new_box[0, 0] * new_box[1, 1]))
            st_akma = float(baro_info.get("surface_tension_akma", 0.0))
            surface_tension_work = -st_akma * (area_new - area_curr)
        else:
            scale = 1.0 + delta_v * u_rand
            if scale <= 0:
                return
            cbrt_scale = scale ** (1.0 / 3.0)
            axis_scales = np.array([cbrt_scale, cbrt_scale, cbrt_scale], dtype=np.float64)
            new_box = current_box * cbrt_scale

        vol_new = vol_current * scale
        delta_v_val = vol_new - vol_current

        current_pos = np.asarray(self._positions, dtype=np.float64)
        _, com_per_atom = self._unwrap_molecules_and_com(current_pos, current_box)
        new_pos = current_pos + (axis_scales[None, :] - 1.0) * com_per_atom

        new_pos_jnp = jnp.asarray(new_pos, dtype=self._positions.dtype)
        if self._converted_system.virtual_sites is not None:
            new_pos_jnp = self._converted_system.compute_virtual_sites(new_pos_jnp)

        gp = self._get_dynamic_global_params()
        ewald_tol = (
            self._converted_system.ewald_error_tolerance
            if self._converted_system.ewald_error_tolerance is not None
            else self._config.ewald_error_tolerance
        )
        e_curr, _ = _pure_compute_energy_and_forces(
            self._positions,
            self._box_vectors,
            self._bundle,
            self._custom_forces,
            self._position_restraints,
            gp,
            self._converted_system.force_group_by_component,
            -1,
            self._converted_system.has_pbc,
            self._converted_system.nonbonded_cutoff,
            self._converted_system.bond_force_groups,
            self._converted_system.angle_force_groups,
            self._converted_system.dihedral_force_groups,
            self._converted_system.cmap_force_groups,
            self._converted_system.nonbonded_method,
            self._converted_system.has_nonbonded,
            self._config.use_dispersion_correction and self._converted_system.use_dispersion_correction,
            ewald_tol,
            self._converted_system.reaction_field_dielectric,
            self._converted_system.has_implicit_solvent,
            self._converted_system.solvent_dielectric,
            self._converted_system.solute_dielectric,
            self._nonbonded_engine,
            self._converted_system.pme_grid_dims,
            use_pallas=self._config.use_pallas,
            pallas_interpret=self._config.pallas_interpret,
            lj_switch_width=self._converted_system.lj_switch_width,
            neighbor_idx=getattr(self, "_neighbor_idx", None),
            use_pallas_pme=self._config.use_pallas_pme,
            custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
            tile_pair_nl=None,
            gbsa_tile_list=self._gbsa_tile_list,
            is_triclinic=self._is_triclinic,
            mixed_precision=self._mixed_precision,
            gb_use_neighbor_list=self._gb_use_neighbor_list,
            gbsa_surface_tension=self._converted_system.gbsa_surface_tension,
            pme_alpha=self._effective_pme_alpha(),
        )
        new_box_jnp = jnp.asarray(new_box, dtype=self._box_vectors.dtype)
        new_bundle = eqx.tree_at(lambda b: b.box, self._bundle, new_box_jnp)
        e_new, _ = _pure_compute_energy_and_forces(
            new_pos_jnp,
            new_box_jnp,
            new_bundle,
            self._custom_forces,
            self._position_restraints,
            gp,
            self._converted_system.force_group_by_component,
            -1,
            self._converted_system.has_pbc,
            self._converted_system.nonbonded_cutoff,
            self._converted_system.bond_force_groups,
            self._converted_system.angle_force_groups,
            self._converted_system.dihedral_force_groups,
            self._converted_system.cmap_force_groups,
            self._converted_system.nonbonded_method,
            self._converted_system.has_nonbonded,
            self._config.use_dispersion_correction and self._converted_system.use_dispersion_correction,
            ewald_tol,
            self._converted_system.reaction_field_dielectric,
            self._converted_system.has_implicit_solvent,
            self._converted_system.solvent_dielectric,
            self._converted_system.solute_dielectric,
            self._nonbonded_engine,
            self._converted_system.pme_grid_dims,
            use_pallas=self._config.use_pallas,
            pallas_interpret=self._config.pallas_interpret,
            lj_switch_width=self._converted_system.lj_switch_width,
            neighbor_idx=getattr(self, "_neighbor_idx", None),
            use_pallas_pme=self._config.use_pallas_pme,
            custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
            tile_pair_nl=None,
            gbsa_tile_list=self._gbsa_tile_list,
            is_triclinic=self._is_triclinic,
            mixed_precision=self._mixed_precision,
            gb_use_neighbor_list=self._gb_use_neighbor_list,
            gbsa_surface_tension=self._converted_system.gbsa_surface_tension,
            pme_alpha=self._effective_pme_alpha(),
        )

        delta_u = float(e_new - e_curr)
        delta_w = (
            delta_u
            + target_p_akma * delta_v_val
            + surface_tension_work
            - float(n_mol) * kT_akma * np.log(max(scale, 1e-12))
        )
        exponent = -delta_w / max(kT_akma, 1e-12)
        accept_prob = min(1.0, float(np.exp(min(exponent, 50.0))))
        rand_val = float(jax.random.uniform(subkey_accept))

        self._mc_num_attempted += 1
        if rand_val < accept_prob:
            self._mc_num_accepted += 1
            self._positions = new_pos_jnp
            self._box_vectors = new_box_jnp
            self._bundle = new_bundle
            self._converted_system = eqx.tree_at(
                lambda s: s.bundle.box, self._converted_system, new_box_jnp
            )
            self.applyConstraints()
            if (
                self._neighbor is not None
                or self._tile_pair_nl is not None
                or self._gbsa_tile_list is not None
            ):
                self._init_neighbor_list(self._positions[: self._n_particles])

        if self._mc_num_attempted >= 10:
            if self._mc_num_accepted < 0.25 * self._mc_num_attempted:
                self._mc_volume_scale = max(self._mc_volume_scale * 0.5, 1e-6)
            elif self._mc_num_accepted > 0.75 * self._mc_num_attempted:
                self._mc_volume_scale = min(self._mc_volume_scale * 1.1, 0.3)
            self._mc_num_attempted = 0
            self._mc_num_accepted = 0

    def _compute_virial(self, forces: jnp.ndarray | np.ndarray | None = None) -> float:
        """Compute scalar thermodynamic virial W = -dU/d(lam)|_{lam=1} under periodic molecular COM scaling."""
        if forces is None and self._converted_system.has_pbc and self._box_vectors is not None:
            gp = self._get_dynamic_global_params()
            ewald_tol = (
                self._converted_system.ewald_error_tolerance
                if self._converted_system.ewald_error_tolerance is not None
                else self._config.ewald_error_tolerance
            )
            pos_np = np.asarray(self._positions, dtype=np.float64)
            box_np = np.asarray(self._box_vectors, dtype=np.float64)
            pos_unwrapped_np, com_np = self._unwrap_molecules_and_com(pos_np, box_np)
            pos_u = jnp.asarray(pos_unwrapped_np, dtype=self._positions.dtype)
            com_u = jnp.asarray(com_np, dtype=self._positions.dtype)
            box_u = self._box_vectors.astype(self._positions.dtype)

            def _u_of_lam(lam: jax.Array) -> jax.Array:
                p_scaled = pos_u + (lam - 1.0) * com_u
                if self._converted_system.virtual_sites is not None:
                    p_scaled = self._converted_system.compute_virtual_sites(p_scaled)
                b_scaled = box_u * lam
                bnd_scaled = eqx.tree_at(lambda b: b.box, self._bundle, b_scaled)
                return _eval_system_potential_energy(
                    p_scaled,
                    b_scaled,
                    bnd_scaled,
                    self._custom_forces,
                    self._position_restraints,
                    gp,
                    self._converted_system.force_group_by_component,
                    -1,
                    self._converted_system.has_pbc,
                    self._converted_system.nonbonded_cutoff,
                    bond_force_groups=self._converted_system.bond_force_groups,
                    angle_force_groups=self._converted_system.angle_force_groups,
                    dihedral_force_groups=self._converted_system.dihedral_force_groups,
                    cmap_force_groups=self._converted_system.cmap_force_groups,
                    nonbonded_method=self._converted_system.nonbonded_method,
                    has_nonbonded=self._converted_system.has_nonbonded,
                    use_dispersion_correction=(
                        self._config.use_dispersion_correction
                        and self._converted_system.use_dispersion_correction
                    ),
                    ewald_error_tolerance=ewald_tol,
                    rf_dielectric=self._converted_system.reaction_field_dielectric,
                    has_implicit_solvent=self._converted_system.has_implicit_solvent,
                    solvent_dielectric=self._converted_system.solvent_dielectric,
                    solute_dielectric=self._converted_system.solute_dielectric,
                    nonbonded_engine=self._nonbonded_engine,
                    pme_grid_dims=self._converted_system.pme_grid_dims,
                    use_pallas=False,
                    pallas_interpret=False,
                    lj_switch_width=self._converted_system.lj_switch_width,
                    neighbor_idx=getattr(self, "_neighbor_idx", None),
                    use_pallas_pme=False,
                    custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
                    tile_pair_nl=None,
                    is_triclinic=self._is_triclinic,
                    gb_use_neighbor_list=self._gb_use_neighbor_list,
                    gbsa_surface_tension=self._converted_system.gbsa_surface_tension,
                    pme_alpha=self._effective_pme_alpha(),
                )

            du_dlam = jax.grad(_u_of_lam)(jnp.asarray(1.0, dtype=pos_u.dtype))
            return float(-du_dlam)

        if forces is None:
            gp = self._get_dynamic_global_params()
            ewald_tol = (
                self._converted_system.ewald_error_tolerance
                if self._converted_system.ewald_error_tolerance is not None
                else self._config.ewald_error_tolerance
            )
            _, forces_akma = _pure_compute_energy_and_forces(
                self._positions,
                self._box_vectors,
                self._bundle,
                self._custom_forces,
                self._position_restraints,
                gp,
                self._converted_system.force_group_by_component,
                -1,
                self._converted_system.has_pbc,
                self._converted_system.nonbonded_cutoff,
                self._converted_system.bond_force_groups,
                self._converted_system.angle_force_groups,
                self._converted_system.dihedral_force_groups,
                self._converted_system.cmap_force_groups,
                self._converted_system.nonbonded_method,
                self._converted_system.has_nonbonded,
                self._config.use_dispersion_correction and self._converted_system.use_dispersion_correction,
                ewald_tol,
                self._converted_system.reaction_field_dielectric,
                self._converted_system.has_implicit_solvent,
                self._converted_system.solvent_dielectric,
                self._converted_system.solute_dielectric,
                self._nonbonded_engine,
                self._converted_system.pme_grid_dims,
                use_pallas=self._config.use_pallas,
                pallas_interpret=self._config.pallas_interpret,
                lj_switch_width=self._converted_system.lj_switch_width,
                neighbor_idx=getattr(self, "_neighbor_idx", None),
                use_pallas_pme=self._config.use_pallas_pme,
                custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
                tile_pair_nl=self._tile_pair_nl,
                gbsa_tile_list=self._gbsa_tile_list,
                is_triclinic=self._is_triclinic,
                mixed_precision=self._mixed_precision,
                gb_use_neighbor_list=self._gb_use_neighbor_list,
                gbsa_surface_tension=self._converted_system.gbsa_surface_tension,
                pme_alpha=self._effective_pme_alpha(),
            )
            if self._converted_system.virtual_sites is not None:
                forces_akma = self._converted_system.redistribute_virtual_site_forces(
                    forces_akma, self._positions
                )
            forces_arr = forces_akma
        else:
            forces_arr = jnp.asarray(forces)

        if self._converted_system.has_pbc and self._box_vectors is not None:
            rel_anchor = _minimum_image_dr(
                self._positions - self._positions[0],
                self._box_vectors,
                self._converted_system.has_pbc,
            )
            com = jnp.mean(rel_anchor, axis=0)
            rel_pos = rel_anchor - com
            return float(jnp.sum(rel_pos * forces_arr))
        return float(jnp.sum(self._positions * forces_arr))

    def _apply_scr_barostat(
        self,
        baro_info: dict[str, Any],
        dt_chunk_akma: float,
        kT_akma: float,
    ) -> None:
        """Apply Stochastic Cell Rescaling barostat move (Bernetti & Bussi 2020)."""
        if self._box_vectors is None:
            return

        current_box = np.asarray(self._box_vectors, dtype=np.float64)
        volume = abs(float(np.linalg.det(current_box)))
        if volume <= 0:
            return

        target_p_akma = baro_info["pressure_akma"]
        masses = self._bundle.masses[: self._n_particles, None]
        vel = self._velocities[: self._n_particles]
        ke_total = float(0.5 * jnp.sum(masses * (vel**2)))

        virial = self._compute_virial()
        inst_pressure = (2.0 * ke_total + virial) / (3.0 * volume)

        tau = 20.0  # AKMA time (~0.98 ps; cf. barostat.py tau_p=20.0)
        compressibility_akma = 4.5e-5 * BAR_PER_AKMA_PRESSURE
        pressure_deviation = inst_pressure - target_p_akma

        self._prng_key, subk = jax.random.split(self._prng_key)
        noise = float(jax.random.normal(subk))
        depsilon_det = (dt_chunk_akma / tau) * compressibility_akma * pressure_deviation
        depsilon_stoch = np.sqrt(
            2.0 * kT_akma * compressibility_akma * dt_chunk_akma / (tau * volume)
        ) * noise
        depsilon = depsilon_det + depsilon_stoch
        mu = float(np.clip(np.exp(depsilon / 3.0), 0.99, 1.01))

        new_box = current_box * mu
        new_box_jnp = jnp.asarray(new_box, dtype=self._box_vectors.dtype)
        self._box_vectors = new_box_jnp
        self._bundle = eqx.tree_at(lambda b: b.box, self._bundle, new_box_jnp)
        self._converted_system = eqx.tree_at(
            lambda s: s.bundle.box, self._converted_system, new_box_jnp
        )
        pos_np = np.asarray(self._positions, dtype=np.float64)
        _, com_np = self._unwrap_molecules_and_com(pos_np, current_box)
        scaled_pos = pos_np + (mu - 1.0) * com_np
        self._positions = jnp.asarray(scaled_pos, dtype=self._positions.dtype)
        self._velocities = self._velocities / mu
        self.applyConstraints()
        self.applyVelocityConstraints(self._config.settle_velocity_tol)
        if (
            self._neighbor is not None
            or self._tile_pair_nl is not None
            or self._gbsa_tile_list is not None
        ):
            self._init_neighbor_list(self._positions[: self._n_particles])

    # --- Positions & Velocities ---
    def setPositions(self, positions: Any) -> None:  # noqa: N802
        """Set particle positions, converting from OpenMM nm to Prolix Å and reconstructing virtual sites."""
        pos_arr = np.asarray(to_akma(positions, "length"))
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            raise ValueError(
                f"Positions array must have shape (N, 3), got {pos_arr.shape}"
            )
        if pos_arr.shape[0] != self._n_particles:
            raise openmm.OpenMMException(
                "Called setPositions() on a Context with the wrong number of positions"
            )
        pos_akma = jnp.asarray(pos_arr, dtype=self._dtype)
        if self._converted_system.virtual_sites is not None:
            pos_akma = self._converted_system.compute_virtual_sites(pos_akma)
        self._positions = pos_akma
        if self._should_use_neighbor_list() or self._can_use_gbsa_tiles():
            self._init_neighbor_list(self._positions[: self._n_particles])

    def setVelocities(self, velocities: Any) -> None:  # noqa: N802
        """Set particle velocities, converting from OpenMM nm/ps to Prolix Å/τ_AKMA."""
        vel_arr = np.asarray(to_akma(velocities, "velocity"))
        if vel_arr.ndim != 2 or vel_arr.shape[1] != 3:
            raise ValueError(
                f"Velocities array must have shape (N, 3), got {vel_arr.shape}"
            )
        if vel_arr.shape[0] != self._n_particles:
            raise openmm.OpenMMException(
                "Called setVelocities() on a Context with the wrong number of velocities"
            )
        vel_akma = jnp.asarray(vel_arr, dtype=self._dtype)
        if self._converted_system.virtual_sites is not None:
            vs_idx = self._converted_system.virtual_sites.order
            vel_akma = vel_akma.at[vs_idx].set(0.0)
        self._velocities = vel_akma
        self._velocities_initialized = True

    def setVelocitiesToTemperature(  # noqa: N802
        self, temperature: Any, randomSeed: int | None = None
    ) -> None:
        """Assign Maxwell-Boltzmann thermal velocities accounting for constraints & virtual sites (BUG-OMM1)."""
        temp_k = (
            float(temperature.value_in_unit(unit.kelvin))
            if isinstance(temperature, unit.Quantity)
            else float(temperature)
        )
        seed = int(randomSeed) if randomSeed is not None else int(self._config.deterministic_seed)
        key = jax.random.PRNGKey(seed)

        masses = self._bundle.masses[: self._n_particles].astype(self._dtype)
        safe_inv_m = self._converted_system.inv_mass[: self._n_particles].astype(self._dtype)
        sigmas = jnp.sqrt(
            jnp.asarray(BOLTZMANN_KCAL * max(temp_k, 0.0), dtype=self._dtype) * safe_inv_m
        )[:, None]

        noise = jax.random.normal(key, shape=self._positions.shape, dtype=self._dtype)
        vel = jnp.where(safe_inv_m[:, None] > 1e-12, noise * sigmas, 0.0)
        if self._converted_system.virtual_sites is not None:
            vs_idx = self._converted_system.virtual_sites.order
            vel = vel.at[vs_idx].set(0.0)

        # Project onto constraint tangent space FIRST, then remove COM velocity, then rescale to exact N_dof
        self._velocities = vel
        self.applyVelocityConstraints(self._config.settle_velocity_tol)
        vel = self._velocities.astype(self._dtype)

        active_m_mask = safe_inv_m > 1e-12
        if self._effective_remove_com() and int(jnp.sum(active_m_mask)) > 1:
            total_m = jnp.sum(jnp.where(active_m_mask, masses, 0.0))
            v_com = (
                jnp.sum(jnp.where(active_m_mask[:, None], vel * masses[:, None], 0.0), axis=0)
                / jnp.maximum(total_m, 1e-12)
            )
            vel = jnp.where(active_m_mask[:, None], vel - v_com, 0.0)
            if self._converted_system.virtual_sites is not None:
                vs_idx = self._converted_system.virtual_sites.order
                vel = vel.at[vs_idx].set(0.0)
            self._velocities = vel
            self.applyVelocityConstraints(self._config.settle_velocity_tol)
            vel = self._velocities.astype(self._dtype)

        n_dof = self._active_ndof()
        curr_ke = 0.5 * jnp.sum(masses[:, None] * (vel**2))
        target_ke = jnp.asarray(
            0.5 * float(n_dof) * BOLTZMANN_KCAL * max(temp_k, 0.0), dtype=self._dtype
        )
        scale = jnp.where(curr_ke > 1e-15, jnp.sqrt(target_ke / curr_ke), 0.0)
        self._velocities = (vel * scale).astype(self._dtype)
        self._velocities_initialized = True

    def setPeriodicBoxVectors(self, a: Any, b: Any, c: Any) -> None:  # noqa: N802
        """Set periodic box vectors in canonical lower-triangular form."""
        a_akma = np.asarray(to_akma(a, "length"), dtype=np.float64)
        b_akma = np.asarray(to_akma(b, "length"), dtype=np.float64)
        c_akma = np.asarray(to_akma(c, "length"), dtype=np.float64)
        box_mat = np.stack([a_akma, b_akma, c_akma])

        box_canonical = reduce_triclinic_box_vectors(jnp.asarray(box_mat, dtype=self._dtype))
        box_np = np.asarray(box_canonical, dtype=np.float64)
        off_diag = np.abs(box_np - np.diag(np.diag(box_np)))
        self._is_triclinic = bool(np.max(off_diag) > 1e-6)
        self._box_vectors = box_canonical
        box_bundle = box_canonical.astype(jnp.float32)
        self._bundle = eqx.tree_at(lambda bnd: bnd.box, self._bundle, box_bundle)
        new_pme_dims = self._converted_system.pme_grid_dims
        if (
            self._converted_system.has_pbc
            and self._converted_system.nonbonded_method == "pme"
        ):
            explicit_omm_grid = False
            for f_idx in range(self._system.getNumForces()):
                f_obj = self._system.getForce(f_idx)
                if isinstance(f_obj, openmm.NonbondedForce):
                    _, nx, ny, nz = f_obj.getPMEParameters()
                    if nx > 0 and ny > 0 and nz > 0:
                        explicit_omm_grid = True
                    break
            if not explicit_omm_grid:
                box_diag_np = np.diag(box_np)
                cutoff_a = float(self._converted_system.nonbonded_cutoff)
                if (
                    isinstance(self._config.pme_grid_spacing, _DefaultFloat)
                    and 0.0 < cutoff_a < float("inf")
                ):
                    eff_ewald_tol = (
                        float(self._converted_system.ewald_error_tolerance)
                        if self._converted_system.ewald_error_tolerance is not None
                        else float(self._config.ewald_error_tolerance)
                    )
                    _, omm_dims = compute_openmm_pme_params(
                        box_diag_np,
                        cutoff=cutoff_a,
                        ewald_error_tolerance=eff_ewald_tol,
                    )
                    if float(np.max(box_diag_np)) <= 36.0:
                        min_floor = 96 if eff_ewald_tol <= 5e-4 else 8
                        new_pme_dims = (
                            max(omm_dims[0], min_floor),
                            max(omm_dims[1], min_floor),
                            max(omm_dims[2], min_floor),
                        )
                    else:
                        new_pme_dims = omm_dims
                else:
                    new_pme_dims = compute_pme_grid_dims(
                        jnp.asarray(box_diag_np),
                        grid_spacing=float(self._config.pme_grid_spacing),
                    )
        self._converted_system = dataclasses.replace(
            self._converted_system,
            bundle=self._bundle,
            is_triclinic=self._is_triclinic,
            pme_grid_dims=new_pme_dims,
        )
        if self._should_use_neighbor_list() or self._can_use_gbsa_tiles():
            self._init_neighbor_list(self._positions[: self._n_particles])

    # --- Time & Step Count ---
    def setTime(self, time: Any) -> None:  # noqa: N802
        self._time = float(to_akma(time, "time"))

    def getTime(self) -> unit.Quantity:  # noqa: N802
        return (self._time * TAU_AKMA_TO_PS) * unit.picosecond

    def setStepCount(self, step: int) -> None:  # noqa: N802
        self._step_count = int(step)

    def getStepCount(self) -> int:  # noqa: N802
        return self._step_count

    # --- Parameters ---
    def getParameter(self, name: str) -> float:  # noqa: N802
        if name not in self._parameters:
            raise openmm.OpenMMException(f"Parameter '{name}' not found in Context.")
        return float(self._parameters[name])

    def setParameter(self, name: str, value: float) -> None:  # noqa: N802
        self._parameters[name] = float(value)

    def getParameters(self) -> dict[str, float]:  # noqa: N802
        return dict(self._parameters)

    # --- Constraints & Virtual Sites ---
    def computeVirtualSites(self) -> None:  # noqa: N802
        if self._converted_system.virtual_sites is not None:
            self._positions = self._converted_system.compute_virtual_sites(self._positions)

    def applyConstraints(self, tolerance: float = 1e-5) -> None:  # noqa: N802, ARG002
        """Apply distance constraints on current positions (SETTLE and SHAKE)."""
        masses = self._bundle.masses[: self._n_particles]
        self._positions = self._constraints.apply_positions(
            self._positions, self._positions, masses[:, None], self._box_vectors
        ).astype(self._positions.dtype)
        self.computeVirtualSites()

    def applyVelocityConstraints(self, tolerance: float = 1e-6) -> None:  # noqa: N802, ARG002
        """Apply velocity constraints on current velocities (SETTLE and RATTLE)."""
        masses = self._bundle.masses[: self._n_particles]
        momenta = self._velocities * masses[:, None]
        dt = 2.0 * 1e-3 * PS_TO_TAU_AKMA
        mom_constrained = self._constraints.apply_velocities(
            self._positions, self._positions, momenta, masses[:, None], dt
        )
        safe_inv_m = self._converted_system.inv_mass[: self._n_particles]
        self._velocities = mom_constrained * safe_inv_m[:, None]

    # --- Checkpoints & State Serialization ---
    def createCheckpoint(self) -> bytes:  # noqa: N802
        state_dict = {
            "positions": np.asarray(self._positions),
            "velocities": np.asarray(self._velocities),
            "box_vectors": np.asarray(self._box_vectors) if self._box_vectors is not None else None,
            "time": float(self._time),
            "step_count": int(self._step_count),
            "parameters": dict(self._parameters),
            "prng_key": np.asarray(self._prng_key),
        }
        return pickle.dumps(state_dict, protocol=pickle.HIGHEST_PROTOCOL)

    def loadCheckpoint(self, checkpoint: bytes | dict[str, Any]) -> None:  # noqa: N802
        if isinstance(checkpoint, bytes):
            try:
                state_dict = pickle.loads(checkpoint)  # noqa: S301
            except Exception as e:
                raise openmm.OpenMMException(
                    f"loadCheckpoint: Checkpoint header was not correct ({e})"
                ) from e
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(f"Expected bytes or dict, got {type(checkpoint).__name__}")

        required_keys = ("positions", "velocities", "time", "step_count")
        for k in required_keys:
            if k not in state_dict:
                raise openmm.OpenMMException(
                    f"loadCheckpoint: Checkpoint header was not correct (missing key '{k}')"
                )

        pos_arr = np.asarray(state_dict["positions"])
        vel_arr = np.asarray(state_dict["velocities"])
        if pos_arr.ndim != 2 or pos_arr.shape != (self._n_particles, 3):
            raise openmm.OpenMMException(
                f"loadCheckpoint: Checkpoint header was not correct (invalid positions shape {pos_arr.shape}, expected ({self._n_particles}, 3))"
            )
        if vel_arr.ndim != 2 or vel_arr.shape != (self._n_particles, 3):
            raise openmm.OpenMMException(
                f"loadCheckpoint: Checkpoint header was not correct (invalid velocities shape {vel_arr.shape}, expected ({self._n_particles}, 3))"
            )

        self._positions = jnp.asarray(pos_arr, dtype=self._dtype)
        self._velocities = jnp.asarray(vel_arr, dtype=self._dtype)
        self._velocities_initialized = True
        bv = state_dict.get("box_vectors")
        self._box_vectors = jnp.asarray(bv, dtype=self._dtype) if bv is not None else None
        if self._box_vectors is not None:
            bv_np = np.asarray(self._box_vectors, dtype=np.float64)
            off_diag = np.abs(bv_np - np.diag(np.diag(bv_np)))
            self._is_triclinic = bool(np.max(off_diag) > 1e-6)
        self._time = float(state_dict["time"])
        self._step_count = int(state_dict["step_count"])
        self._parameters = dict(state_dict.get("parameters", {}))
        if "prng_key" in state_dict:
            self._prng_key = jnp.asarray(state_dict["prng_key"])
        if self._should_use_neighbor_list() or self._can_use_gbsa_tiles():
            self._init_neighbor_list(self._positions[: self._n_particles])

    def reinitialize(self, preserveState: bool = False) -> None:
        chk = self.createCheckpoint() if preserveState else None
        self._converted_system = from_openmm_system(
            self._system, self._integrator, self._config
        )
        self._has_finite_cutoff = (
            0.0 < float(self._converted_system.nonbonded_cutoff) < float("inf")
        )
        self._bundle = self._converted_system.bundle
        if self._dtype == jnp.float64:
            self._bundle = eqx.tree_at(
                lambda b: b.masses, self._bundle, self._bundle.masses.astype(jnp.float64)
            )
        self._custom_forces = tuple(self._converted_system.custom_forces)
        self._position_restraints = tuple(self._converted_system.position_restraints)
        if preserveState and chk is not None:
            self.loadCheckpoint(chk)

    def setState(self, state: openmm.State) -> None:  # noqa: N802
        types = state.getDataTypes() if hasattr(state, "getDataTypes") else 0xFFFFFFFF
        with contextlib.suppress(Exception):
            self.setTime(state.getTime())
            self.setStepCount(state.getStepCount())
            bv = state.getPeriodicBoxVectors(asNumpy=True)
            self.setPeriodicBoxVectors(bv[0], bv[1], bv[2])
        if types & openmm.State.Positions:
            self.setPositions(state.getPositions(asNumpy=True))
        if types & openmm.State.Velocities:
            self.setVelocities(state.getVelocities(asNumpy=True))
        if types & openmm.State.Parameters:
            for k, v in state.getParameters().items():
                self.setParameter(k, v)

    # --- Dynamic Zero-Recompile Parameter Update ---
    def updateParametersInContext(self, force: openmm.Force | None = None) -> None:  # noqa: N802
        """Update MolecularBundle array leaves from OpenMM forces with zero XLA recompilations."""
        bundle = self._bundle
        conv = self._converted_system
        meta = conv.system_metadata
        custom_force_map = meta.get("omm_custom_force_to_compiled_idx", {})
        updated_cf = list(self._custom_forces)

        new_gbsa_st = conv.gbsa_surface_tension
        new_explicit_alpha = conv.explicit_pme_alpha
        new_pme_grid_dims = conv.pme_grid_dims

        for f_idx in range(self._system.getNumForces()):
            f = self._system.getForce(f_idx)
            if force is not None and not _is_same_openmm_force(f, force):
                continue

            if isinstance(f, openmm.NonbondedForce):
                if f.getNonbondedMethod() == openmm.NonbondedForce.PME:
                    alpha_omm, nx, ny, nz = f.getPMEParameters()
                    alpha_nm = (
                        alpha_omm.value_in_unit(unit.nanometer**-1)
                        if hasattr(alpha_omm, "value_in_unit")
                        else float(alpha_omm)
                    )
                    new_explicit_alpha = (
                        float(alpha_nm / NM_TO_ANGSTROM) if alpha_nm > 0.0 else None
                    )
                    if int(nx) > 0 and int(ny) > 0 and int(nz) > 0:
                        new_pme_grid_dims = (int(nx), int(ny), int(nz))

                n_particles = f.getNumParticles()
                charges_np = np.asarray(bundle.charges).copy()
                sigmas_np = np.asarray(bundle.sigmas).copy()
                epsilons_np = np.asarray(bundle.epsilons).copy()

                for p_i in range(n_particles):
                    q, sig, eps = f.getParticleParameters(p_i)
                    charges_np[p_i] = q.value_in_unit(unit.elementary_charge)
                    sigmas_np[p_i] = sig.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                    epsilons_np[p_i] = (
                        eps.value_in_unit(unit.kilojoules_per_mole)
                        * KJ_PER_MOL_TO_KCAL_PER_MOL
                    )

                p_dtype = bundle.charges.dtype
                bundle = eqx.tree_at(
                    lambda bnd: (bnd.charges, bnd.sigmas, bnd.epsilons),
                    bundle,
                    (
                        jnp.asarray(charges_np, dtype=p_dtype),
                        jnp.asarray(sigmas_np, dtype=p_dtype),
                        jnp.asarray(epsilons_np, dtype=p_dtype),
                    ),
                )

                exc_map = meta.get("omm_exception_to_bundle_idx")
                if exc_map and bundle.exception_sigmas is not None:
                    exc_sig_np = np.asarray(bundle.exception_sigmas).copy()
                    exc_eps_np = np.asarray(bundle.exception_epsilons).copy()
                    exc_qp_np = np.asarray(bundle.exception_chargeprods).copy()

                    for e_i in range(f.getNumExceptions()):
                        b_exc_idx = exc_map.get(e_i)
                        if b_exc_idx is not None:
                            _, _, qprod, sig, eps = f.getExceptionParameters(e_i)
                            exc_qp_np[b_exc_idx] = qprod.value_in_unit(
                                unit.elementary_charge**2
                            )
                            exc_sig_np[b_exc_idx] = (
                                sig.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                            )
                            exc_eps_np[b_exc_idx] = (
                                eps.value_in_unit(unit.kilojoules_per_mole)
                                * KJ_PER_MOL_TO_KCAL_PER_MOL
                            )

                    bundle = eqx.tree_at(
                        lambda bnd: (
                            bnd.exception_sigmas,
                            bnd.exception_epsilons,
                            bnd.exception_chargeprods,
                        ),
                        bundle,
                        (
                            jnp.asarray(exc_sig_np, dtype=p_dtype),
                            jnp.asarray(exc_eps_np, dtype=p_dtype),
                            jnp.asarray(exc_qp_np, dtype=p_dtype),
                        ),
                    )

            elif isinstance(f, openmm.HarmonicBondForce):
                bond_map = meta.get("omm_forces_bond_to_bundle_idx", {}).get(f_idx, {})
                if bond_map and bundle.bond_params is not None:
                    bp_dtype = bundle.bond_params.dtype
                    bp_np = np.asarray(bundle.bond_params).copy()
                    for b_i in range(f.getNumBonds()):
                        b_idx = bond_map.get(b_i)
                        if b_idx is not None:
                            _, _, r0, k = f.getBondParameters(b_i)
                            bp_np[b_idx, 0] = (
                                r0.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                            )
                            bp_np[b_idx, 1] = (
                                k.value_in_unit(
                                    unit.kilojoules_per_mole / (unit.nanometer**2)
                                )
                                * BOND_K_OMM_TO_AKMA
                            )
                    bundle = eqx.tree_at(
                        lambda bnd: bnd.bond_params,
                        bundle,
                        jnp.asarray(bp_np, dtype=bp_dtype),
                    )

            elif isinstance(f, openmm.HarmonicAngleForce):
                angle_map = meta.get("omm_forces_angle_to_bundle_idx", {}).get(f_idx, {})
                if angle_map and bundle.angle_params is not None:
                    ap_dtype = bundle.angle_params.dtype
                    ap_np = np.asarray(bundle.angle_params).copy()
                    for a_i in range(f.getNumAngles()):
                        a_idx = angle_map.get(a_i)
                        if a_idx is not None:
                            _, _, _, th0, k = f.getAngleParameters(a_i)
                            ap_np[a_idx, 0] = th0.value_in_unit(unit.radian)
                            ap_np[a_idx, 1] = (
                                k.value_in_unit(
                                    unit.kilojoules_per_mole / (unit.radian**2)
                                )
                                * ANGLE_K_OMM_TO_AKMA
                            )
                    bundle = eqx.tree_at(
                        lambda bnd: bnd.angle_params,
                        bundle,
                        jnp.asarray(ap_np, dtype=ap_dtype),
                    )

            elif isinstance(f, openmm.PeriodicTorsionForce):
                torsion_map = meta.get("omm_forces_torsion_to_bundle_idx", {}).get(f_idx, {})
                if torsion_map and bundle.dihedral_params is not None:
                    dp_dtype = bundle.dihedral_params.dtype
                    dp_np = np.asarray(bundle.dihedral_params).copy()
                    for t_i in range(f.getNumTorsions()):
                        t_idx = torsion_map.get(t_i)
                        if t_idx is not None:
                            _, _, _, _, per, phase, k = f.getTorsionParameters(t_i)
                            dp_np[t_idx, 0] = float(per)
                            dp_np[t_idx, 1] = phase.value_in_unit(unit.radian)
                            dp_np[t_idx, 2] = (
                                k.value_in_unit(unit.kilojoules_per_mole)
                                * TORSION_K_OMM_TO_AKMA
                            )
                    bundle = eqx.tree_at(
                        lambda bnd: bnd.dihedral_params,
                        bundle,
                        jnp.asarray(dp_np, dtype=dp_dtype),
                    )

            elif isinstance(f, openmm.GBSAOBCForce):
                sa_kj_nm2 = f.getSurfaceAreaEnergy().value_in_unit(
                    unit.kilojoule_per_mole / (unit.nanometer**2)
                )
                new_gbsa_st = float(
                    sa_kj_nm2 * KJ_PER_MOL_TO_KCAL_PER_MOL / (NM_TO_ANGSTROM**2)
                )
                if bundle.radii is not None and bundle.scaled_radii is not None:
                    r_dtype = bundle.radii.dtype
                    radii_np = np.asarray(bundle.radii).copy()
                    scaled_radii_np = np.asarray(bundle.scaled_radii).copy()
                    for p_i in range(f.getNumParticles()):
                        _, r, s = f.getParticleParameters(p_i)
                        r_a = r.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                        radii_np[p_i] = r_a
                        scaled_radii_np[p_i] = max(r_a - 0.09, 0.0) * float(s)
                    bundle = eqx.tree_at(
                        lambda bnd: (bnd.radii, bnd.scaled_radii),
                        bundle,
                        (
                            jnp.asarray(radii_np, dtype=r_dtype),
                            jnp.asarray(scaled_radii_np, dtype=r_dtype),
                        ),
                    )

            elif f_idx in custom_force_map:
                f_cf = cast("Any", f)
                cf_idx = custom_force_map[f_idx]
                cf = updated_cf[cf_idx]
                if isinstance(cf, CompiledCustomBondForce):
                    num_params = f_cf.getNumPerBondParameters()
                    param_names = [
                        f_cf.getPerBondParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_bond_params[p]).copy() for p in param_names
                    }
                    for b_i in range(f_cf.getNumBonds()):
                        p1, p2, params = f_cf.getBondParameters(b_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][b_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_bond_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomAngleForce):
                    num_params = f_cf.getNumPerAngleParameters()
                    param_names = [
                        f_cf.getPerAngleParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_angle_params[p]).copy() for p in param_names
                    }
                    for a_i in range(f_cf.getNumAngles()):
                        p1, p2, p3, params = f_cf.getAngleParameters(a_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][a_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_angle_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomTorsionForce):
                    num_params = f_cf.getNumPerTorsionParameters()
                    param_names = [
                        f_cf.getPerTorsionParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_torsion_params[p]).copy() for p in param_names
                    }
                    for t_i in range(f_cf.getNumTorsions()):
                        p1, p2, p3, p4, params = f_cf.getTorsionParameters(t_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][t_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_torsion_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomExternalForce):
                    num_params = f_cf.getNumPerParticleParameters()
                    param_names = [
                        f_cf.getPerParticleParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_particle_params[p]).copy() for p in param_names
                    }
                    for p_i in range(f_cf.getNumParticles()):
                        _, params = f_cf.getParticleParameters(p_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][p_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_particle_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomCompoundBondForce):
                    num_params = f_cf.getNumPerBondParameters()
                    param_names = [
                        f_cf.getPerBondParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_bond_params[p]).copy() for p in param_names
                    }
                    for b_i in range(f_cf.getNumBonds()):
                        _, params = f_cf.getBondParameters(b_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][b_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_bond_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomNonbondedForce):
                    num_params = f_cf.getNumPerParticleParameters()
                    param_names = [
                        f_cf.getPerParticleParameterName(i)
                        for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_particle_params[p]).copy()
                        for p in param_names
                    }
                    for p_i in range(f_cf.getNumParticles()):
                        params = f_cf.getParticleParameters(p_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][p_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_particle_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledRBTorsionForce):
                    c_params_np = np.asarray(cf.c_params).copy()
                    for t_i in range(f_cf.getNumTorsions()):
                        _, _, _, _, c0, c1, c2, c3, c4, c5 = f_cf.getTorsionParameters(t_i)
                        coeffs = [
                            c.value_in_unit(unit.kilojoules_per_mole)
                            if hasattr(c, "value_in_unit")
                            else float(c)
                            for c in [c0, c1, c2, c3, c4, c5]
                        ]
                        c_params_np[t_i] = coeffs
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.c_params,
                        cf,
                        jnp.asarray(c_params_np, dtype=jnp.float32),
                    )

        self._bundle = bundle
        self._custom_forces = tuple(updated_cf)
        conv_updated = eqx.tree_at(
            lambda s: (s.bundle, s.custom_forces),
            self._converted_system,
            (bundle, tuple(updated_cf)),
        )
        if (
            new_gbsa_st != conv_updated.gbsa_surface_tension
            or new_explicit_alpha != conv_updated.explicit_pme_alpha
            or new_pme_grid_dims != conv_updated.pme_grid_dims
        ):
            conv_updated = dataclasses.replace(
                conv_updated,
                gbsa_surface_tension=float(new_gbsa_st),
                explicit_pme_alpha=new_explicit_alpha,
                pme_grid_dims=new_pme_grid_dims,
            )
        self._converted_system = conv_updated

    # --- Parameter Derivatives ---
    def _compute_parameter_derivatives(self, groups_mask: int = -1) -> dict[str, float]:
        """Compute analytical energy parameter derivatives dE/dlambda via jax.grad."""
        pos_akma = self._positions
        pos_nm = pos_akma * ANGSTROM_TO_NM
        box_nm = (
            self._box_vectors * ANGSTROM_TO_NM
            if self._converted_system.has_pbc and self._box_vectors is not None
            else None
        )

        gp_dict = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in self._parameters.items()
        }
        param_derivatives: dict[str, float] = {}

        for cf in self._custom_forces:
            if not ((1 << cf.force_group) & groups_mask):
                continue
            if len(cf.energy_param_derivatives) == 0:
                continue

            def _force_e(gp: dict[str, Array], cf=cf) -> Array:
                return cf.energy_omm(pos_nm, box_nm, gp)

            _, grad_gp = jax.value_and_grad(_force_e)(gp_dict)

            for p_name in cf.energy_param_derivatives:
                if p_name in grad_gp:
                    deriv_val = float(grad_gp[p_name])
                    param_derivatives[p_name] = (
                        param_derivatives.get(p_name, 0.0) + deriv_val
                    )

        return param_derivatives

    # --- Molecule Wrapping ---
    def _wrap_molecules(self, pos_akma: np.ndarray) -> np.ndarray:
        """Wrap molecules (connected components) into the primary periodic box."""
        if self._box_vectors is None:
            return pos_akma
        box = np.asarray(self._box_vectors, dtype=np.float64)
        wrapped = pos_akma.copy()
        a_vec, b_vec, c_vec = box[0], box[1], box[2]

        for mol in self._molecules:
            if not mol:
                continue
            mol_idx = list(mol)
            com = np.mean(wrapped[mol_idx], axis=0)
            # OpenMM Reference 3-step sequential back-substitution (z -> y -> x)
            diff = np.zeros(3, dtype=np.float64)
            if abs(c_vec[2]) > 1e-12:
                diff += c_vec * np.floor(com[2] / c_vec[2])
            if abs(b_vec[1]) > 1e-12:
                diff += b_vec * np.floor((com[1] - diff[1]) / b_vec[1])
            if abs(a_vec[0]) > 1e-12:
                diff += a_vec * np.floor((com[0] - diff[0]) / a_vec[0])
            wrapped[mol_idx] -= diff

        return wrapped

    def _build_molecules(self) -> tuple[tuple[int, ...], ...]:
        """Compute connected components of particles linked by bonds and constraints."""
        n_p = self._n_particles
        adj: dict[int, list[int]] = {i: [] for i in range(n_p)}

        bundle = self._bundle
        if bundle.bond_idx is not None and bundle.bond_mask is not None:
            pairs = np.asarray(bundle.bond_idx)
            mask = np.asarray(bundle.bond_mask)
            for (p1, p2), active in zip(pairs, mask, strict=False):
                if active and p1 < n_p and p2 < n_p:
                    adj[int(p1)].append(int(p2))
                    adj[int(p2)].append(int(p1))

        if bundle.constraint_pairs is not None:
            cpairs = np.asarray(bundle.constraint_pairs)
            cmask = np.asarray(bundle.constraint_mask)
            for (p1, p2), active in zip(cpairs, cmask, strict=False):
                if active and p1 < n_p and p2 < n_p:
                    adj[int(p1)].append(int(p2))
                    adj[int(p2)].append(int(p1))

        visited = set()
        molecules = []
        for i in range(n_p):
            if i not in visited:
                comp = []
                stack = [i]
                visited.add(i)
                while stack:
                    curr = stack.pop()
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            stack.append(nbr)
                molecules.append(tuple(sorted(comp)))
        return tuple(molecules)

    # --- getState ---
    def getState(  # noqa: PLR0913, N802
        self,
        positions: bool = False,
        velocities: bool = False,
        forces: bool = False,
        energy: bool = False,
        parameters: bool = False,
        parameterDerivatives: bool = False,
        integratorParameters: bool = False,  # noqa: ARG002
        enforcePeriodicBox: bool = False,
        groups: int | set[int] = -1,
        getPositions: bool = False,
        getVelocities: bool = False,
        getForces: bool = False,
        getEnergy: bool = False,
        getParameters: bool = False,
        getParameterDerivatives: bool = False,
        getIntegratorParameters: bool = False,  # noqa: ARG002
    ) -> ProlixState:
        """Query simulation state observables matching OpenMM getState API."""
        req_pos = positions or getPositions
        req_vel = velocities or getVelocities
        req_forces = forces or getForces
        req_energy = energy or getEnergy
        req_params = parameters or getParameters
        req_derivs = parameterDerivatives or getParameterDerivatives

        data_types = 0
        if req_pos:
            data_types |= openmm.State.Positions
        if req_vel:
            data_types |= openmm.State.Velocities
        if req_forces:
            data_types |= openmm.State.Forces
        if req_energy:
            data_types |= openmm.State.Energy
        if req_params:
            data_types |= openmm.State.Parameters
        if req_derivs:
            data_types |= openmm.State.ParameterDerivatives

        # Groups normalization
        if isinstance(groups, set):
            raw_groups = sum(1 << g for g in groups if 0 <= g <= 31)
        else:
            raw_groups = int(groups)
        groups_mask = int(np.int32(raw_groups))

        # Box vectors
        box_nm = (
            np.asarray(self._box_vectors, dtype=np.float64) * ANGSTROM_TO_NM
            if self._box_vectors is not None
            else np.eye(3, dtype=np.float64)
        )

        # Positions
        pos_nm = None
        if req_pos:
            pos_akma = np.asarray(self._positions, dtype=np.float64)
            if enforcePeriodicBox and (self._converted_system.has_pbc or self._box_vectors is not None):
                pos_akma = self._wrap_molecules(pos_akma)
            pos_nm = pos_akma * ANGSTROM_TO_NM

        # Velocities
        vel_omm = None
        if req_vel:
            vel_omm = (
                np.asarray(self._velocities, dtype=np.float64) * VELOCITY_AKMA_TO_OMM
            )

        # Energies & Forces
        pe_kj = None
        ke_kj = None
        forces_omm = None

        if req_energy or req_forces:
            gp = {
                k: jnp.asarray(val, dtype=jnp.float32)
                for k, val in self._parameters.items()
            }
            ewald_tol = (
                self._converted_system.ewald_error_tolerance
                if self._converted_system.ewald_error_tolerance is not None
                else self._config.ewald_error_tolerance
            )
            pe_akma, forces_akma = _pure_compute_energy_and_forces(
                self._positions,
                self._box_vectors,
                self._bundle,
                self._custom_forces,
                self._position_restraints,
                gp,
                self._converted_system.force_group_by_component,
                groups_mask,
                self._converted_system.has_pbc,
                self._converted_system.nonbonded_cutoff,
                self._converted_system.bond_force_groups,
                self._converted_system.angle_force_groups,
                self._converted_system.dihedral_force_groups,
                self._converted_system.cmap_force_groups,
                self._converted_system.nonbonded_method,
                self._converted_system.has_nonbonded,
                self._config.use_dispersion_correction and self._converted_system.use_dispersion_correction,
                ewald_tol,
                self._converted_system.reaction_field_dielectric,
                self._converted_system.has_implicit_solvent,
                self._converted_system.solvent_dielectric,
                self._converted_system.solute_dielectric,
                self._nonbonded_engine,
                self._converted_system.pme_grid_dims,
                use_pallas=self._config.use_pallas,
                pallas_interpret=self._config.pallas_interpret,
                lj_switch_width=self._converted_system.lj_switch_width,
                neighbor_idx=getattr(self, "_neighbor_idx", None),
                use_pallas_pme=self._config.use_pallas_pme,
                custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
                tile_pair_nl=self._tile_pair_nl,
                gbsa_tile_list=self._gbsa_tile_list,
                is_triclinic=self._is_triclinic,
                mixed_precision=self._mixed_precision,
                gb_use_neighbor_list=self._gb_use_neighbor_list,
                gbsa_surface_tension=self._converted_system.gbsa_surface_tension,
                pme_alpha=self._effective_pme_alpha(),
            )
            if req_energy:
                pe_kj = float(pe_akma * KCAL_PER_MOL_TO_KJ_PER_MOL)
                m = np.asarray(self._bundle.masses[: self._n_particles], dtype=np.float64)
                v = np.asarray(self._velocities[: self._n_particles], dtype=np.float64)
                eff_scheme = self._effective_integrator_scheme()
                ig_state = cast("Any", self._integrator)
                dt_omm = (
                    ig_state.getStepSize()
                    if hasattr(ig_state, "getStepSize")
                    else 0.001 * unit.picosecond
                )
                dt_akma = float(to_akma(dt_omm, "time"))
                # BUG-OMM2: Only leapfrog stores half-step staggered v(t - dt/2); Velocity Verlet,
                # BAOAB/LFMiddle, Brownian, Nose-Hoover, and cold uninitialized contexts store on-step v(t).
                if (
                    eff_scheme == "leapfrog"
                    and (self._step_count > 0 or self._velocities_initialized)
                    and dt_akma > 0
                    and forces_akma is not None
                ):
                    inv_m = np.asarray(self._converted_system.inv_mass[: self._n_particles], dtype=np.float64)
                    f_akma_np = np.asarray(forces_akma[: self._n_particles], dtype=np.float64)
                    v_sync = v + 0.5 * dt_akma * (f_akma_np * inv_m[:, None])
                    ke_akma = 0.5 * np.sum(m[:, None] * (v_sync**2))
                else:
                    ke_akma = 0.5 * np.sum(m[:, None] * (v**2))
                ke_kj = float(ke_akma * KCAL_PER_MOL_TO_KJ_PER_MOL)

            if req_forces:
                if self._converted_system.virtual_sites is not None:
                    forces_akma = self._converted_system.redistribute_virtual_site_forces(
                        forces_akma, self._positions
                    )
                forces_omm = np.asarray(forces_akma, dtype=np.float64) * FORCE_AKMA_TO_OMM

        params_dict = dict(self._parameters) if req_params else None
        derivs_dict = (
            self._compute_parameter_derivatives(groups_mask) if req_derivs else None
        )

        return ProlixState(
            positions_nm=pos_nm,
            velocities_omm=vel_omm,
            forces_omm=forces_omm,
            box_vectors_nm=box_nm,
            time_ps=self._time * TAU_AKMA_TO_PS,
            step_count=self._step_count,
            pe_kj=pe_kj,
            ke_kj=ke_kj,
            parameters=params_dict,
            param_derivatives=derivs_dict,
            data_types=data_types,
        )
