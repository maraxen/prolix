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
import jax.core
import jax.dlpack
import jax.numpy as jnp
import numpy as np
from jax_md import space

import prolix.api.ensemble_plan as _ensemble_plan_mod
from prolix.constants import COULOMB_CONSTANT
from prolix.export import in_export_context
from prolix.openmm.config import (
    ProlixOpenMMConfig,
    ProlixPlatform,
    _DefaultFloat,
    parse_platform_properties,
)
from prolix.openmm.converter import (
    CompiledRBTorsionForce,
    ConvertedOpenMMSystem,
    NonbondedParameterOffsets,
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
from prolix.openmm.pattern_matcher import match_custom_force_pattern
from prolix.openmm.units import (
    ANGLE_K_OMM_TO_AKMA,
    ANGSTROM_TO_NM,
    BAR_PER_AKMA_PRESSURE,
    BOND_K_OMM_TO_AKMA,
    FORCE_AKMA_TO_OMM,
    FORCE_OMM_TO_AKMA,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
    PS_TO_TAU_AKMA,
    TAU_AKMA_TO_PS,
    TORSION_K_OMM_TO_AKMA,
    VELOCITY_AKMA_TO_OMM,
    VELOCITY_OMM_TO_AKMA,
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
from prolix.physics.lj_pme import (
    dispersion_charges,
    ewald_p6_long_range_value,
    lj_pme_direct_pair_energy,
    lj_pme_reciprocal_energy,
    lj_pme_self_energy,
    resolve_ljpme_parameters,
)
from prolix.physics.neighbor_list import (
    get_neighbor_exclusion_scales,
    make_neighbor_list_fn,
)
from prolix.physics.pallas_bonded import pallas_bonded_all_fused
from prolix.physics.pallas_gbsa_tile import (
    GBSATilePairList,
    build_gbsa_tile_pair_list,
    pallas_gbsa_tile,
    pallas_gbsa_tile_energy_and_forces,
)
from prolix.physics.pallas_implicit import pallas_gb_ace_nl
from prolix.physics.pallas_tile_pairs import (
    FusedBondedBuffers,
    MortonTilePairState,
    _pallas_bonded_and_tile_pairs_soa,
    estimate_max_tile_pairs,
    pallas_build_morton_tile_pairs_gpu,
    precompute_dedup_excl_edges,
    prepare_fused_bonded_buffers,
)
from prolix.physics.pbc import (
    box_volume,
    minimum_image_displacement,
    reduce_triclinic_box_vectors,
)
from prolix.physics.pme import (
    _cached_host_influence_function_np,
    compute_openmm_pme_params,
    compute_pme_grid_dims,
    scale_influence_function_isotropic,
    spme_background_energy,
    spme_reciprocal_energy,
    spme_self_energy,
)
from prolix.physics.pme_pallas import (
    _gather_pallas_soa,
    _spread_charges_pallas_soa,
)
from prolix.physics.settle import (
    _pallas_baoab_post_soa,
    _pallas_baoab_pre_force_batch,
    build_baoab_soa_static_plan,
)
from prolix.types.bundles import MolecularBundle  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

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


_LJ_TAIL_SIG_PARAMS = frozenset(inspect.signature(lj_dispersion_tail_energy).parameters.keys())
_LJ_TAIL_ACCEPTS_SWITCH_AND_RULE = (
    "lj_switch_width" in _LJ_TAIL_SIG_PARAMS and "combining_rule" in _LJ_TAIL_SIG_PARAMS
)
_PALLAS_NB_TP_SIG_PARAMS = frozenset(
    inspect.signature(pallas_nonbonded_tile_pairs).parameters.keys()
)
_BINOM_6_CTX: tuple[float, ...] = tuple(float(math.comb(6, k)) for k in range(7))
_BINOM_12_CTX: tuple[float, ...] = tuple(float(math.comb(12, k)) for k in range(13))
_SWITCH_I6_SERIES_CTX: tuple[float, ...] = tuple(
    (n + 1.0) / ((2.0 * n + 3.0) * (2.0 * n + 5.0)) for n in range(85)
)


def _get_pallas_soa_sig_params() -> frozenset[str]:
    """Inspect `_pallas_bonded_and_tile_pairs_soa` parameters dynamically at call time."""
    return frozenset(
        inspect.signature(_pallas_bonded_and_tile_pairs_soa).parameters.keys()
    )


def _from_jax_or_dlpack(tensor: Any) -> jax.Array:
    """Ingest a `jax.Array`, DLPack producer (`__dlpack__`), or array-like onto device without host copy."""
    if isinstance(tensor, jax.Array):
        return tensor
    if unit is not None and unit.is_quantity(tensor):
        raw_val = getattr(tensor, "_value", tensor)
        if isinstance(raw_val, jax.Array):
            return raw_val
        if hasattr(raw_val, "__dlpack__"):
            return jax.dlpack.from_dlpack(raw_val)
        return jnp.asarray(raw_val)
    if hasattr(tensor, "__dlpack__"):
        return jax.dlpack.from_dlpack(tensor)
    return jnp.asarray(tensor)


def _custom_forces_are_gpu_fast(
    custom_forces: Sequence[CompiledLeptonForce],
) -> bool:
    """Return True if all custom forces run in O(N) / O(|U|*N) on persistent GPU runners."""
    return all(
        bool(
            getattr(
                cf,
                "is_gpu_runner_fast",
                not isinstance(cf, CompiledCustomNonbondedForce),
            )
        )
        for cf in custom_forces
    )


def _is_reduced_triclinic_box_np(box_np: np.ndarray) -> bool:
    """Return True if `(3, 3)` matrix satisfies OpenMM lower-triangular reduced-basis invariants."""
    if box_np.ndim != 2 or box_np.shape != (3, 3):
        return False
    ax = float(box_np[0, 0])
    by = float(box_np[1, 1])
    cz = float(box_np[2, 2])
    if not (ax > 0.0 and by > 0.0 and cz > 0.0):
        return False
    if (
        abs(float(box_np[0, 1])) > 1e-5
        or abs(float(box_np[0, 2])) > 1e-5
        or abs(float(box_np[1, 2])) > 1e-5
    ):
        return False
    bx = abs(float(box_np[1, 0]))
    cx = abs(float(box_np[2, 0]))
    cy = abs(float(box_np[2, 1]))
    return (
        bx <= 0.5 * ax + 1e-4
        and cx <= 0.5 * ax + 1e-4
        and cy <= 0.5 * by + 1e-4
    )


def _triclinic_morton_kernels_supported(do_pme: bool = True) -> bool:
    """Inspect whether triclinic keywords are active across Morton tile-pair, SPME, and SETTLE kernels."""
    tp_ok = (
        "is_triclinic" in inspect.signature(pallas_build_morton_tile_pairs_gpu).parameters
        and "is_triclinic" in _get_pallas_soa_sig_params()
    )
    settle_ok = (
        "is_triclinic" in inspect.signature(_pallas_baoab_pre_force_batch).parameters
    )
    if not (tp_ok and settle_ok):
        return False
    if do_pme:
        return (
            "is_triclinic" in inspect.signature(_spread_charges_pallas_soa).parameters
            and "is_triclinic" in inspect.signature(_gather_pallas_soa).parameters
        )
    return True


def _system_has_anisotropic_or_membrane_barostat(system: Any) -> bool:
    """Return True if `system` contains `MonteCarloAnisotropicBarostat` or `MonteCarloMembraneBarostat`."""
    if openmm is None:
        return False
    aniso_cls = getattr(openmm, "MonteCarloAnisotropicBarostat", None)
    mem_cls = getattr(openmm, "MonteCarloMembraneBarostat", None)
    baro_types = tuple(c for c in (aniso_cls, mem_cls) if c is not None)
    if not baro_types:
        return False
    for f_idx in range(system.getNumForces()):
        if isinstance(system.getForce(f_idx), baro_types):
            return True
    return False


_MORTON_RUNNER_CACHE: dict[tuple[Any, ...], tuple[Any, dict[str, Any]]] = {}


def _safe_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    """AD-safe square root avoiding 1 / (2 * sqrt(0)) NaNs on zero inputs."""
    pos_mask = x > 0.0
    return jnp.where(
        pos_mask,
        jnp.sqrt(jnp.where(pos_mask, x, jnp.ones_like(x))),
        jnp.zeros_like(x),
    )


def _rf_k_c(cutoff: float, rf_dielectric: float) -> tuple[float, float]:
    """Reaction-field (k_rf, c_rf) safe in the metallic conductor limit rf_dielectric=inf."""
    inv_eps = 1.0 / float(rf_dielectric)
    k_rf = ((1.0 - inv_eps) / (2.0 + inv_eps)) / (cutoff**3)
    c_rf = (3.0 / (2.0 + inv_eps)) / cutoff
    return k_rf, c_rf


def _combine_sig_eps(
    sig_a: jnp.ndarray,
    sig_b: jnp.ndarray,
    eps_a: jnp.ndarray,
    eps_b: jnp.ndarray,
    lj_combining_rule: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Combine pairwise LJ (sigma_ij, epsilon_ij) under Lorentz-Berthelot or Geometric mixing."""
    if lj_combining_rule == "geometric":
        sig_ij = _safe_sqrt(sig_a * sig_b)
    else:
        sig_ij = 0.5 * (sig_a + sig_b)
    eps_ij = _safe_sqrt(eps_a * eps_b)
    return sig_ij, eps_ij


def _apply_particle_offsets(
    q: jnp.ndarray,
    sig: jnp.ndarray,
    eps: jnp.ndarray,
    offsets: NonbondedParameterOffsets | None,
    gp_map: dict[str, Any] | None,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply NonbondedForce ParticleParameterOffsets on-GPU with exact autodiff wrt globals."""
    if offsets is None or not gp_map or len(offsets.param_names) == 0:
        return q, sig, eps
    if int(offsets.particle_indices.shape[0]) == 0:
        return q, sig, eps
    dtype = q.dtype
    gp_vec = jnp.stack(
        [jnp.asarray(gp_map.get(nm, 0.0), dtype=dtype) for nm in offsets.param_names]
    )
    valid_p = (offsets.particle_indices >= 0) & (offsets.particle_indices < n)
    safe_p = jnp.where(valid_p, offsets.particle_indices.astype(jnp.int32), jnp.int32(0))
    lam_p = jnp.where(valid_p, gp_vec[offsets.particle_param_idx.astype(jnp.int32)], 0.0)
    deltas = lam_p[:, None] * offsets.particle_scales.astype(dtype)
    q = q.at[safe_p].add(deltas[:, 0])
    sig = sig.at[safe_p].add(deltas[:, 1])
    eps = eps.at[safe_p].add(deltas[:, 2])
    return q, sig, eps


def _apply_exception_offsets(
    qp_14: jnp.ndarray,
    sig_14: jnp.ndarray,
    eps_14: jnp.ndarray,
    offsets: NonbondedParameterOffsets | None,
    gp_map: dict[str, Any] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply NonbondedForce ExceptionParameterOffsets on-GPU with exact autodiff wrt globals."""
    if offsets is None or not gp_map or len(offsets.param_names) == 0:
        return qp_14, sig_14, eps_14
    if int(offsets.exception_bundle_indices.shape[0]) == 0:
        return qp_14, sig_14, eps_14
    n_exc = int(qp_14.shape[0])
    if n_exc == 0:
        return qp_14, sig_14, eps_14
    dtype = qp_14.dtype
    gp_vec = jnp.stack(
        [jnp.asarray(gp_map.get(nm, 0.0), dtype=dtype) for nm in offsets.param_names]
    )
    valid_e = (offsets.exception_bundle_indices >= 0) & (
        offsets.exception_bundle_indices < n_exc
    )
    safe_e = jnp.where(valid_e, offsets.exception_bundle_indices.astype(jnp.int32), jnp.int32(0))
    lam_e = jnp.where(valid_e, gp_vec[offsets.exception_param_idx.astype(jnp.int32)], 0.0)
    deltas = lam_e[:, None] * offsets.exception_scales.astype(dtype)
    qp_14 = qp_14.at[safe_e].add(deltas[:, 0])
    sig_14 = sig_14.at[safe_e].add(deltas[:, 1])
    eps_14 = eps_14.at[safe_e].add(deltas[:, 2])
    return qp_14, sig_14, eps_14


def _compute_lj_tail_energy_compat(
    box_pbc: jnp.ndarray,
    sig: jnp.ndarray,
    eps: jnp.ndarray,
    cutoff: float,
    atom_mask: jnp.ndarray,
    *,
    lj_switch_width: float = 0.0,
    combining_rule: str = "lorentz_berthelot",
) -> jnp.ndarray:
    """Evaluate switched/unswitched isotropic LJ tail correction compatibly across Track 1."""
    if _LJ_TAIL_ACCEPTS_SWITCH_AND_RULE:
        return lj_dispersion_tail_energy(
            box_pbc,
            sig,
            eps,
            cutoff,
            atom_mask,
            lj_switch_width=float(lj_switch_width),
            combining_rule=str(combining_rule),
        )
    dtype = eps.dtype
    mask_f = atom_mask.astype(dtype)
    sqrt_eps = _safe_sqrt(eps) * mask_f
    if combining_rule == "geometric":
        m3 = jnp.sum(sqrt_eps * (sig**3))
        m6 = jnp.sum(sqrt_eps * (sig**6))
        c6_sum, c12_sum = m3 * m3, m6 * m6
    else:
        moments = [jnp.sum(sqrt_eps * (sig**k)) for k in range(13)]
        c6_sum = jnp.zeros((), dtype=dtype)
        for k_b, b_k in enumerate(_BINOM_6_CTX):
            c6_sum = c6_sum + b_k * moments[k_b] * moments[6 - k_b]
        c6_sum = c6_sum * (1.0 / 64.0)
        c12_sum = jnp.zeros((), dtype=dtype)
        for k_b, b_k in enumerate(_BINOM_12_CTX):
            c12_sum = c12_sum + b_k * moments[k_b] * moments[12 - k_b]
        c12_sum = c12_sum * (1.0 / 4096.0)

    rc = float(cutoff)
    sw = float(lj_switch_width)
    if sw > 0.0:
        rsw = rc - sw
        if rsw <= 0.0:
            raise ValueError("switched LJ tail requires positive switching distance (0 < start <= cutoff)")
        t = rsw / rc
        i12 = (((5.0 * t + 9.0) * t + 9.0) * t + 5.0) / (252.0 * (rsw**6) * (rc**3))
        z = (rc - rsw) / (rc + rsw)
        z2 = z * z
        if z2 <= 0.61:
            s_acc = 0.0
            z2_pow = 1.0
            for coeff in _SWITCH_I6_SERIES_CTX:
                s_acc += coeff * z2_pow
                z2_pow *= z2
            i6 = 8.0 * s_acc / ((rc + rsw) ** 3)
        else:
            i6 = (
                10.0
                * ((t * t + 4.0 * t + 1.0) * math.log(rc / rsw) - 3.0 * (1.0 - t * t))
                / ((rc**3) * ((1.0 - t) ** 5))
            )
    else:
        i6 = 1.0 / (3.0 * (rc**3))
        i12 = 1.0 / (9.0 * (rc**9))
    inv_vol = 1.0 / box_volume(box_pbc)
    return (8.0 * jnp.pi * inv_vol) * (c12_sum * i12 - c6_sum * i6)


def _eval_tile_pairs_jax(
    pos: jnp.ndarray,
    q: jnp.ndarray,
    sig: jnp.ndarray,
    eps: jnp.ndarray,
    tile_pair_nl: TilePairNeighborList,
    box_diag: jnp.ndarray | None,
    *,
    cutoff: float,
    lj_switch_width: float,
    is_pme: bool,
    alpha_float: float,
    is_rf: bool,
    rf_dielectric: float,
    lj_combining_rule: str,
    is_lj_pme: bool = False,
    lj_alpha_float: float = 0.34,
) -> jnp.ndarray:
    """Pure-JAX O(P) tile-pair evaluation supporting RF, LJPME, geometric mixing, and box autodiff."""
    dtype = pos.dtype
    n_real = int(pos.shape[0])
    tp_arr = jnp.asarray(tile_pair_nl.tile_pairs, dtype=jnp.int32)
    ef_arr = jnp.asarray(tile_pair_nl.excl_full_mask, dtype=jnp.int32)
    perm = tile_pair_nl.atom_perm

    if perm is not None:
        perm_arr = jnp.asarray(perm, dtype=jnp.int32)
        n_perm = int(perm_arr.shape[-1])
    else:
        perm_arr = None
        n_perm = n_real

    n_blocks = max(1, (max(n_real, n_perm) + 31) // 32)
    n_pad = n_blocks * 32
    pad_len = n_pad - n_real
    if pad_len > 0:
        pos_p = jnp.pad(pos, ((0, pad_len), (0, 0)))
        q_p = jnp.pad(q, (0, pad_len))
        sig_p = jnp.pad(sig, (0, pad_len), constant_values=1.0)
        eps_p = jnp.pad(eps, (0, pad_len))
        msk_p = jnp.arange(n_pad) < n_real
    else:
        pos_p, q_p, sig_p, eps_p = pos, q, sig, eps
        msk_p = jnp.ones((n_pad,), dtype=jnp.bool_)

    if perm_arr is not None:
        if int(perm_arr.shape[-1]) < n_pad:
            extra = jnp.arange(int(perm_arr.shape[-1]), n_pad, dtype=jnp.int32)
            perm_arr = jnp.concatenate([perm_arr, extra], axis=0)
        pos_p = pos_p[perm_arr]
        q_p = q_p[perm_arr]
        sig_p = sig_p[perm_arr]
        eps_p = eps_p[perm_arr]
        msk_p = msk_p[perm_arr]

    b_p = dispersion_charges(sig_p, eps_p, msk_p) if is_lj_pme else sig_p
    lane_32 = jnp.arange(32, dtype=jnp.int32)
    upper_tri = lane_32[:, None] < lane_32[None, :]
    bit_shifts = lane_32[None, :]
    has_box = box_diag is not None
    bx_v = (
        box_diag.astype(dtype)
        if box_diag is not None
        else jnp.ones((3,), dtype=dtype)
    )
    has_fin_cut = cutoff < float("inf")
    cutoff_sq = jnp.asarray(cutoff * cutoff if has_fin_cut else 1.0e30, dtype=dtype)
    alpha_c = jnp.asarray(alpha_float, dtype=dtype)
    k_rf, c_rf = _rf_k_c(cutoff, rf_dielectric) if (is_rf and has_fin_cut) else (0.0, 0.0)

    def _pair_tile_energy(tp_row: jnp.ndarray, ef_row: jnp.ndarray) -> jnp.ndarray:
        ti, tj = tp_row[0], tp_row[1]
        v_tp = (ti >= 0) & (tj >= 0) & (ti < n_blocks) & (tj < n_blocks)
        s_ti = jnp.clip(ti, 0, n_blocks - 1)
        s_tj = jnp.clip(tj, 0, n_blocks - 1)
        oi = s_ti * 32 + lane_32
        oj = s_tj * 32 + lane_32

        ri, rj = pos_p[oi], pos_p[oj]
        qi, qj = q_p[oi], q_p[oj]
        si, sj = sig_p[oi], sig_p[oj]
        ei, ej = eps_p[oi], eps_p[oj]
        mi, mj = msk_p[oi], msk_p[oj]

        is_full = ((ef_row[:, None] >> bit_shifts) & jnp.int32(1)) != 0
        geom_ok = jnp.where(s_ti == s_tj, upper_tri, jnp.bool_(True))
        valid_p = v_tp & mi[:, None] & mj[None, :] & geom_ok & (~is_full)

        dr = ri[:, None, :] - rj[None, :, :]
        if has_box:
            dr = dr - bx_v[None, None, :] * jnp.floor(dr / bx_v[None, None, :] + 0.5)
        r_sq = jnp.sum(dr * dr, axis=-1)
        active = valid_p & (r_sq < cutoff_sq) if has_fin_cut else valid_p
        r_safe = jnp.sqrt(jnp.where(active, r_sq, 1.0) + jnp.asarray(1e-12, dtype=dtype))

        sig_ij, eps_ij = _combine_sig_eps(
            si[:, None], sj[None, :], ei[:, None], ej[None, :], lj_combining_rule
        )
        if is_lj_pme:
            bi, bj = b_p[oi], b_p[oj]
            u_lj = lj_pme_direct_pair_energy(
                r_safe, sig_ij, eps_ij, bi[:, None] * bj[None, :], lj_alpha_float, cutoff
            )
        else:
            sr = sig_ij / r_safe
            sr6 = sr**6
            u_lj = jnp.where(eps_ij > 0.0, 4.0 * eps_ij * (sr6 * sr6 - sr6), 0.0)
            if lj_switch_width > 0.0 and has_fin_cut:
                r_sw = cutoff - lj_switch_width
                z = jnp.clip((r_safe - r_sw) / lj_switch_width, 0.0, 1.0)
                s_poly = 1.0 - 10.0 * (z**3) + 15.0 * (z**4) - 6.0 * (z**5)
                s_lj = jnp.where(
                    r_safe <= r_sw, 1.0, jnp.where(r_safe < cutoff, s_poly, 0.0)
                )
                u_lj = u_lj * s_lj

        q_mat = qi[:, None] * qj[None, :]
        if is_pme:
            u_elec = COULOMB_CONSTANT * q_mat * (jax.lax.erfc(alpha_c * r_safe) / r_safe)
        elif is_rf:
            u_elec = COULOMB_CONSTANT * q_mat * (1.0 / r_safe + k_rf * (r_safe**2) - c_rf)
        else:
            u_elec = COULOMB_CONSTANT * (q_mat / r_safe)

        return jnp.sum(jnp.where(active, u_lj + u_elec, 0.0))

    m_pairs = int(tp_arr.shape[0])
    if m_pairs == 0:
        return jnp.asarray(0.0, dtype=dtype)
    if m_pairs <= 512:
        return jnp.sum(jax.vmap(_pair_tile_energy)(tp_arr, ef_arr))
    chunk_sz = 256
    n_chunks = (m_pairs + chunk_sz - 1) // chunk_sz
    pad_m = n_chunks * chunk_sz - m_pairs
    if pad_m > 0:
        tp_c = jnp.pad(tp_arr, ((0, pad_m), (0, 0)), constant_values=-1)
        ef_c = jnp.pad(ef_arr, ((0, pad_m), (0, 0)), constant_values=0)
    else:
        tp_c, ef_c = tp_arr, ef_arr
    tp_c = tp_c.reshape(n_chunks, chunk_sz, 2)
    ef_c = ef_c.reshape(n_chunks, chunk_sz, 32)

    def _scan_step(acc: jnp.ndarray, elem: tuple[jnp.ndarray, jnp.ndarray]):
        t_ch, e_ch = elem
        return acc + jnp.sum(jax.vmap(_pair_tile_energy)(t_ch, e_ch)), None

    e_tot, _ = jax.lax.scan(_scan_step, jnp.asarray(0.0, dtype=dtype), (tp_c, ef_c))
    return e_tot


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
    global_params: dict[str, jnp.ndarray] | None = None,
    nonbonded_offsets: NonbondedParameterOffsets | None = None,
    default_params: dict[str, float] | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    exceptions_use_pbc: bool = False,
    dispersion_follows_parameters: bool = True,
    force_pure_jax_tile_pairs: bool = False,
    lj_pme_alpha: float | None = None,
    lj_pme_grid_dims: tuple[int, int, int] | None = None,
) -> jnp.ndarray:
    """Evaluate pairwise Lennard-Jones, Coulomb, exclusions, 1-4 exceptions, and PME/LJPME/RF/dispersion."""
    g_nb = force_group_by_component.get("nonbonded", 0)
    if ((groups_mask & (1 << int(g_nb))) == 0) or bundle.charges is None or not has_nonbonded:
        return jnp.array(0.0, dtype=pos.dtype)

    n = pos.shape[0]
    q_base = bundle.charges[:n].astype(pos.dtype)
    sig_base = bundle.sigmas[:n].astype(pos.dtype)
    eps_base = bundle.epsilons[:n].astype(pos.dtype)
    q, sig, eps = _apply_particle_offsets(
        q_base, sig_base, eps_base, nonbonded_offsets, global_params, n
    )
    box_eval = box.astype(pos.dtype) if box is not None else None
    box_pbc = (
        jnp.diag(box_eval)
        if (box_eval is not None and box_eval.ndim == 2 and not is_triclinic)
        else box_eval
    )

    nb_method = str(nonbonded_method).lower()
    is_lj_pme = nb_method in ("lj_pme", "ljpme") and has_pbc
    is_pme = nb_method in ("pme", "spme", "lj_pme", "ljpme")
    is_rf = (
        nb_method in ("cutoffperiodic", "cutoff_periodic", "reactionfield", "reaction_field", "rf")
        and has_pbc
    )
    # OpenMM NonbondedForce.LJPME ignores both switching function and isotropic dispersion tail.
    eff_lj_switch_width = 0.0 if is_lj_pme else lj_switch_width
    eff_use_disp = False if is_lj_pme else use_dispersion_correction

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

    if is_lj_pme and lj_pme_alpha is not None and float(lj_pme_alpha) > 0.0:
        lj_alpha_float = float(lj_pme_alpha)
    elif is_lj_pme and cutoff < float("inf"):
        lj_alpha_float = float(
            (1.0 / cutoff) * math.sqrt(-math.log(2.0 * float(ewald_error_tolerance)))
        )
    else:
        lj_alpha_float = alpha_float
    b_disp = dispersion_charges(sig, eps) if is_lj_pme else sig

    use_tile_pairs = (
        tile_pair_nl is not None
        and (use_pallas or nonbonded_engine == "morton_tile_pairs")
        and not is_triclinic
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
        can_call_pallas_tp = (
            not force_pure_jax_tile_pairs
            and not is_lj_pme
            and lj_combining_rule == "lorentz_berthelot"
            and (not is_rf or "rf_dielectric" in _PALLAS_NB_TP_SIG_PARAMS)
        )
        if can_call_pallas_tp:
            tp_extra_kw: dict[str, Any] = {}
            if is_rf and "rf_dielectric" in _PALLAS_NB_TP_SIG_PARAMS:
                tp_extra_kw["rf_dielectric"] = float(rf_dielectric)
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
                    lj_switch_width=float(eff_lj_switch_width) if cutoff < float("inf") else 0.0,
                    include_pme_excl_erf=False,
                    return_forces=False,
                    interpret=pallas_interpret or not _triton_gpu_available(),
                    **tp_extra_kw,
                ),
            )
        else:
            tot = _eval_tile_pairs_jax(
                pos,
                q,
                sig,
                eps,
                tile_pair_nl,
                box_diag,
                cutoff=cutoff,
                lj_switch_width=eff_lj_switch_width,
                is_pme=is_pme,
                alpha_float=alpha_float,
                is_rf=is_rf,
                rf_dielectric=rf_dielectric,
                lj_combining_rule=lj_combining_rule,
                is_lj_pme=is_lj_pme,
                lj_alpha_float=lj_alpha_float,
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

        sig_mat, eps_mat = _combine_sig_eps(
            sig[:, None], sig[j_safe], eps[:, None], eps[j_safe], lj_combining_rule
        )
        q_mat = q[:, None] * q[j_safe]

        if is_lj_pme:
            u_lj = lj_pme_direct_pair_energy(
                r_safe,
                sig_mat,
                eps_mat,
                b_disp[:, None] * b_disp[j_safe],
                lj_alpha_float,
                cutoff,
            )
        else:
            sr = sig_mat / r_safe
            sr6 = sr**6
            sr12 = sr6**2
            u_lj = jnp.where(eps_mat > 0.0, 4.0 * eps_mat * (sr12 - sr6), 0.0)
            if eff_lj_switch_width > 0.0 and cutoff < float("inf"):
                r_sw = cutoff - eff_lj_switch_width
                z = jnp.clip((r_safe - r_sw) / eff_lj_switch_width, 0.0, 1.0)
                s_poly = 1.0 - 10.0 * (z**3) + 15.0 * (z**4) - 6.0 * (z**5)
                s_lj = jnp.where(r_safe <= r_sw, 1.0, jnp.where(r_safe < cutoff, s_poly, 0.0))
                u_lj = u_lj * s_lj

        if is_pme:
            u_elec = COULOMB_CONSTANT * q_mat * (jax.lax.erfc(alpha * r_safe) / r_safe)
        elif is_rf:
            k_rf, c_rf = _rf_k_c(cutoff, rf_dielectric)
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

            sig_mat, eps_mat = _combine_sig_eps(
                sig[safe_i][:, None],
                sig[None, :],
                eps[safe_i][:, None],
                eps[None, :],
                lj_combining_rule,
            )
            q_mat = q[safe_i][:, None] * q[None, :]

            if is_lj_pme:
                u_lj = lj_pme_direct_pair_energy(
                    r_safe,
                    sig_mat,
                    eps_mat,
                    b_disp[safe_i][:, None] * b_disp[None, :],
                    lj_alpha_float,
                    cutoff,
                )
            else:
                sr = sig_mat / r_safe
                sr6 = sr**6
                sr12 = sr6**2
                u_lj = jnp.where(eps_mat > 0.0, 4.0 * eps_mat * (sr12 - sr6), 0.0)
                if eff_lj_switch_width > 0.0 and cutoff < float("inf"):
                    r_sw = cutoff - eff_lj_switch_width
                    z = jnp.clip((r_safe - r_sw) / eff_lj_switch_width, 0.0, 1.0)
                    s_poly = 1.0 - 10.0 * (z**3) + 15.0 * (z**4) - 6.0 * (z**5)
                    s_lj = jnp.where(
                        r_safe <= r_sw, 1.0, jnp.where(r_safe < cutoff, s_poly, 0.0)
                    )
                    u_lj = u_lj * s_lj

            if is_pme:
                u_elec = COULOMB_CONSTANT * q_mat * (jax.lax.erfc(alpha * r_safe) / r_safe)
            elif is_rf:
                k_rf, c_rf = _rf_k_c(cutoff, rf_dielectric)
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

        sig_mat, eps_mat = _combine_sig_eps(
            sig[:, None], sig[None, :], eps[:, None], eps[None, :], lj_combining_rule
        )
        q_mat = q[:, None] * q[None, :]

        if is_lj_pme:
            u_lj = lj_pme_direct_pair_energy(
                r_safe,
                sig_mat,
                eps_mat,
                b_disp[:, None] * b_disp[None, :],
                lj_alpha_float,
                cutoff,
            )
        else:
            sr = sig_mat / r_safe
            sr6 = sr**6
            sr12 = sr6**2
            u_lj = jnp.where(eps_mat > 0.0, 4.0 * eps_mat * (sr12 - sr6), 0.0)
            if eff_lj_switch_width > 0.0 and cutoff < float("inf"):
                r_sw = cutoff - eff_lj_switch_width
                z = jnp.clip((r_safe - r_sw) / eff_lj_switch_width, 0.0, 1.0)
                s_poly = 1.0 - 10.0 * (z**3) + 15.0 * (z**4) - 6.0 * (z**5)
                s_lj = jnp.where(r_safe <= r_sw, 1.0, jnp.where(r_safe < cutoff, s_poly, 0.0))
                u_lj = u_lj * s_lj

        if is_pme:
            u_elec = COULOMB_CONSTANT * q_mat * (jax.lax.erfc(alpha * r_safe) / r_safe)
        elif is_rf:
            k_rf, c_rf = _rf_k_c(cutoff, rf_dielectric)
            u_elec = COULOMB_CONSTANT * q_mat * (1.0 / r_safe + k_rf * (r_safe**2) - c_rf)
        else:
            u_elec = COULOMB_CONSTANT * (q_mat / r_safe)

        u_pair = u_lj + u_elec
        if cutoff < float("inf"):
            cutoff_mask = r_safe < cutoff
            u_pair = jnp.where(cutoff_mask, u_pair, 0.0)

        tot = jnp.sum(jnp.where(mask_tri, u_pair, 0.0))

    # Exclusions: obey exceptions_use_pbc (OpenMM default False -> r_ex_raw; True -> r_ex_mic)
    # on unwrapped host coordinates so ProlixContext matches stock OpenMM Reference/CUDA
    # under both conventions (including test_far_exceptions.py when True).
    if bundle.excl_indices is not None and bundle.excl_mask is not None:
        ex_idx = bundle.excl_indices
        ex_mask = bundle.excl_mask
        p1 = ex_idx[:, 0]
        p2 = ex_idx[:, 1]
        valid_ex = ex_mask & (p1 < n) & (p2 < n)
        p1_s = jnp.where(valid_ex, p1, 0)
        p2_s = jnp.where(valid_ex, p2, 0)

        dr_ex_raw = pos[p1_s] - pos[p2_s]
        dr_ex_mic = (
            _minimum_image_dr(dr_ex_raw, box_pbc, True)
            if (has_pbc and box_pbc is not None)
            else dr_ex_raw
        )
        dr_ex_recip = (
            dr_ex_mic
            if (exceptions_use_pbc and has_pbc and box_pbc is not None)
            else dr_ex_raw
        )
        r2_ex_recip = jnp.sum(dr_ex_recip**2, axis=-1)
        r2_ex_recip_s = jnp.where(valid_ex, r2_ex_recip, jnp.ones_like(r2_ex_recip))
        r_ex_recip_s = jnp.sqrt(r2_ex_recip_s + 1e-12)
        r_ex_mic = jnp.sqrt(jnp.sum(dr_ex_mic**2, axis=-1) + 1e-12)
        r_ex_mic_s = jnp.where(valid_ex, r_ex_mic, 1.0)

        if direct_exclusions_premasked:
            if is_pme:
                u_ex_recip = COULOMB_CONSTANT * q[p1_s] * q[p2_s] * (
                    jax.lax.erf(alpha * r_ex_recip_s) / r_ex_recip_s
                )
                tot = tot - jnp.sum(jnp.where(valid_ex, u_ex_recip, 0.0))
            if is_lj_pme:
                u_ex_lj_recip = (
                    b_disp[p1_s]
                    * b_disp[p2_s]
                    * ewald_p6_long_range_value(lj_alpha_float, r2_ex_recip_s)
                )
                tot = tot + jnp.sum(jnp.where(valid_ex, u_ex_lj_recip, 0.0))
        else:
            r_ex_s = r_ex_mic_s
            sig_ex, eps_ex = _combine_sig_eps(
                sig[p1_s], sig[p2_s], eps[p1_s], eps[p2_s], lj_combining_rule
            )
            q_ex = q[p1_s] * q[p2_s]

            if is_lj_pme:
                u_ex_lj = lj_pme_direct_pair_energy(
                    r_ex_s,
                    sig_ex,
                    eps_ex,
                    b_disp[p1_s] * b_disp[p2_s],
                    lj_alpha_float,
                    cutoff,
                )
            else:
                sr_e = sig_ex / r_ex_s
                sr6_e = sr_e**6
                sr12_e = sr6_e**2
                u_ex_lj = jnp.where(eps_ex > 0.0, 4.0 * eps_ex * (sr12_e - sr6_e), 0.0)
                if eff_lj_switch_width > 0.0 and cutoff < float("inf"):
                    r_sw = cutoff - eff_lj_switch_width
                    z_e = jnp.clip((r_ex_s - r_sw) / eff_lj_switch_width, 0.0, 1.0)
                    s_poly_e = 1.0 - 10.0 * (z_e**3) + 15.0 * (z_e**4) - 6.0 * (z_e**5)
                    s_lj_e = jnp.where(
                        r_ex_s <= r_sw, 1.0, jnp.where(r_ex_s < cutoff, s_poly_e, 0.0)
                    )
                    u_ex_lj = u_ex_lj * s_lj_e

            if is_pme:
                u_ex_elec = COULOMB_CONSTANT * q_ex * (jax.lax.erfc(alpha * r_ex_s) / r_ex_s)
                u_ex_recip = COULOMB_CONSTANT * q_ex * (
                    jax.lax.erf(alpha * r_ex_recip_s) / r_ex_recip_s
                )
            elif is_rf:
                k_rf, c_rf = _rf_k_c(cutoff, rf_dielectric)
                u_ex_elec = COULOMB_CONSTANT * q_ex * (1.0 / r_ex_s + k_rf * (r_ex_s**2) - c_rf)
                u_ex_recip = 0.0
            else:
                u_ex_elec = COULOMB_CONSTANT * (q_ex / r_ex_s)
                u_ex_recip = 0.0

            u_ex = u_ex_lj + u_ex_elec
            if cutoff < float("inf"):
                u_ex = jnp.where(r_ex_s < cutoff, u_ex, 0.0)
            tot = tot - jnp.sum(jnp.where(valid_ex, u_ex + u_ex_recip, 0.0))
            if is_lj_pme:
                u_ex_lj_recip = (
                    b_disp[p1_s]
                    * b_disp[p2_s]
                    * ewald_p6_long_range_value(lj_alpha_float, r2_ex_recip_s)
                )
                tot = tot + jnp.sum(jnp.where(valid_ex, u_ex_lj_recip, 0.0))

    # 1-4 / Explicit Exceptions: obey exceptions_use_pbc (default False in OpenMM)
    # and apply ExceptionParameterOffsets on-GPU.
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
        if exceptions_use_pbc and has_pbc and box_pbc is not None:
            dr_exc = _minimum_image_dr(dr_exc, box_pbc, True)
        r_exc = jnp.sqrt(jnp.sum(dr_exc**2, axis=-1) + 1e-12)
        r_exc_s = jnp.where(valid_exc, r_exc, 1.0)
        qp_14, sig_14, eps_14 = _apply_exception_offsets(
            bundle.exception_chargeprods.astype(pos.dtype),
            bundle.exception_sigmas.astype(pos.dtype),
            bundle.exception_epsilons.astype(pos.dtype),
            nonbonded_offsets,
            global_params,
        )

        sr_14 = sig_14 / r_exc_s
        sr6_14 = sr_14**6
        sr12_14 = sr6_14**2
        u_14_lj = jnp.where(eps_14 != 0.0, 4.0 * eps_14 * (sr12_14 - sr6_14), 0.0)
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

    # Dispersion LJ-PME reciprocal and self terms (order=5 matches OpenMM Reference)
    if is_lj_pme and has_pbc and box_pbc is not None:
        eff_lj_dims = lj_pme_grid_dims
        if eff_lj_dims is None and not isinstance(box_pbc, jax.core.Tracer):
            _, eff_lj_dims = resolve_ljpme_parameters(
                np.asarray(box_pbc, dtype=np.float64),
                cutoff=float(cutoff) if cutoff < float("inf") else 9.0,
                ewald_error_tolerance=float(ewald_error_tolerance),
                explicit_alpha=lj_alpha_float,
            )
        if eff_lj_dims is None:
            eff_lj_dims = pme_grid_dims if pme_grid_dims is not None else (12, 12, 12)
        atom_mask_lj = jnp.ones(n, dtype=bool)
        e_lj_recip = lj_pme_reciprocal_energy(
            pos,
            b_disp,
            atom_mask_lj,
            box_pbc,
            eff_lj_dims,
            alpha_lj=lj_alpha_float,
            order=5,
        )
        e_lj_self = lj_pme_self_energy(b_disp, atom_mask=atom_mask_lj, alpha_lj=lj_alpha_float)
        tot = tot + e_lj_recip + e_lj_self

    # LJ dispersion tail correction (obeying dispersion_follows_parameters & lj_combining_rule)
    if eff_use_disp and has_pbc and box_pbc is not None and cutoff < float("inf"):
        atom_mask = jnp.ones(n, dtype=bool)
        if dispersion_follows_parameters or nonbonded_offsets is None:
            sig_tail, eps_tail = sig, eps
        else:
            _, sig_tail, eps_tail = _apply_particle_offsets(
                q_base, sig_base, eps_base, nonbonded_offsets, default_params, n
            )
        e_tail = _compute_lj_tail_energy_compat(
            box_pbc,
            sig_tail,
            eps_tail,
            cutoff,
            atom_mask,
            lj_switch_width=eff_lj_switch_width,
            combining_rule=lj_combining_rule,
        )
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
    nonbonded_offsets: NonbondedParameterOffsets | None = None,
    default_params: dict[str, float] | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    exceptions_use_pbc: bool = False,
    dispersion_follows_parameters: bool = True,
    force_pure_jax_tile_pairs: bool = False,
    lj_pme_alpha: float | None = None,
    lj_pme_grid_dims: tuple[int, int, int] | None = None,
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
        global_params=global_params,
        nonbonded_offsets=nonbonded_offsets,
        default_params=default_params,
        lj_combining_rule=lj_combining_rule,
        exceptions_use_pbc=exceptions_use_pbc,
        dispersion_follows_parameters=dispersion_follows_parameters,
        force_pure_jax_tile_pairs=force_pure_jax_tile_pairs,
        lj_pme_alpha=lj_pme_alpha,
        lj_pme_grid_dims=lj_pme_grid_dims,
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
            e_pr = pr.k_x * dx**2 + pr.k_y * dy**2 + pr.k_z * dz**2
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
    nonbonded_offsets: NonbondedParameterOffsets | None = None,
    default_params: dict[str, float] | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    exceptions_use_pbc: bool = False,
    dispersion_follows_parameters: bool = True,
    lj_pme_alpha: float | None = None,
    lj_pme_grid_dims: tuple[int, int, int] | None = None,
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
            nonbonded_offsets=nonbonded_offsets,
            default_params=default_params,
            lj_combining_rule=lj_combining_rule,
            exceptions_use_pbc=exceptions_use_pbc,
            dispersion_follows_parameters=dispersion_follows_parameters,
            lj_pme_alpha=lj_pme_alpha,
            lj_pme_grid_dims=lj_pme_grid_dims,
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
    nonbonded_offsets: NonbondedParameterOffsets | None = None,
    default_params: dict[str, float] | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    exceptions_use_pbc: bool = False,
    dispersion_follows_parameters: bool = True,
    lj_pme_alpha: float | None = None,
    lj_pme_grid_dims: tuple[int, int, int] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """Pure compiled multi-step rollout kernel via jax.lax.fori_loop with inter-step force reuse."""
    c1 = jnp.exp(-gamma * dt) if is_langevin else 1.0
    c2 = jnp.sqrt(jnp.maximum(1.0 - c1 * c1, 0.0)) if is_langevin else 0.0
    eff_ndof = int(n_dof) if n_dof is not None else max(1, 3 * int(positions.shape[0]) - 3)
    base_step = jnp.asarray(0, dtype=jnp.int32) if step_offset is None else step_offset

    scheme = str(integrator_scheme).lower()
    if scheme == "auto":
        scheme = "baoab" if is_langevin else "leapfrog"
    carries_end_force = scheme in ("baoab", "langevin", "lfmiddle", "verlet", "nose_hoover")

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
            nonbonded_offsets=nonbonded_offsets,
            default_params=default_params,
            lj_combining_rule=lj_combining_rule,
            exceptions_use_pbc=exceptions_use_pbc,
            dispersion_follows_parameters=dispersion_follows_parameters,
            lj_pme_alpha=lj_pme_alpha,
            lj_pme_grid_dims=lj_pme_grid_dims,
        )
        f = (
            virtual_sites.redistribute_forces(f_raw, p_recon)
            if virtual_sites is not None
            else f_raw
        )
        return pe, f

    f_init = (
        jax.lax.cond(
            num_steps > 0,
            lambda p0: compute_f(p0)[1],
            jnp.zeros_like,
            positions,
        )
        if carries_end_force
        else jnp.zeros_like(positions)
    )

    def step_body(i: int, val: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]):
        pos, vel, f_carried, k = val
        f = f_carried if carries_end_force else compute_f(pos)[1]
        f_next = f_carried

        if scheme in ("baoab", "langevin", "lfmiddle"):
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
            _, f_next = compute_f(r)
            v = v + 0.5 * dt * (f_next * inv_mass[:, None])
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)
        elif scheme == "verlet":
            v_half = vel + 0.5 * dt * (f * inv_mass[:, None])
            r = pos + dt * v_half
            if constraints is not None:
                r_c = constraints.apply_positions(pos, r, masses[:, None], box).astype(r.dtype)
                v_half = v_half + (r_c - r) / dt
                r = r_c
            if virtual_sites is not None:
                r = virtual_sites.reconstruct_positions(r)
            _, f_next = compute_f(r)
            v = v_half + 0.5 * dt * (f_next * inv_mass[:, None])
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)
        elif scheme == "brownian":
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
            _, f_next = compute_f(r)
            v = v_half_scaled + 0.5 * dt * (f_next * inv_mass[:, None])
            if constraints is not None:
                p = v * masses[:, None]
                p_proj = constraints.apply_velocities(pos, r, p, masses[:, None], dt, box=box)
                v = jnp.where(inv_mass[:, None] > 1e-12, p_proj * inv_mass[:, None], 0.0)
        else:
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
        return r, v, f_next.astype(pos.dtype), k

    init_val = (positions, velocities, f_init, key)
    final_pos, final_vel, _, final_key = jax.lax.fori_loop(0, num_steps, step_body, init_val)
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


@eqx.filter_jit
def _unwrap_molecules_and_com_jax(
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    anchor_of_atom: jnp.ndarray,
    mol_of_atom: jnp.ndarray,
    inv_mol_size: jnp.ndarray,
    has_pbc: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """O(N) GPU-resident molecular PBC unwrap and per-atom molecular COM computation."""
    anchors = pos[anchor_of_atom]
    if has_pbc and box is not None:
        dr_intra = _minimum_image_dr(pos - anchors, box, True)
        pos_unwrapped = anchors + dr_intra
    else:
        pos_unwrapped = pos
    n_mols = int(inv_mol_size.shape[0])
    mol_sums = jnp.zeros((n_mols, 3), dtype=pos.dtype).at[mol_of_atom].add(pos_unwrapped)
    mol_coms = mol_sums * inv_mol_size[:, None].astype(pos.dtype)
    com_per_atom = mol_coms[mol_of_atom]
    return pos_unwrapped, com_per_atom


@jax.jit
def _wrap_molecules_jax(
    pos: jnp.ndarray,
    box: jnp.ndarray,
    mol_of_atom: jnp.ndarray,
    inv_mol_size: jnp.ndarray,
) -> jnp.ndarray:
    """O(N) GPU-resident whole-molecule PBC wrapping in AKMA Å (0 host loops)."""
    n_mols = int(inv_mol_size.shape[0])
    if n_mols == 0 or int(pos.shape[0]) == 0:
        return pos
    a_vec, b_vec, c_vec = box[0], box[1], box[2]
    mol_sums = jnp.zeros((n_mols, 3), dtype=pos.dtype).at[mol_of_atom].add(pos)
    coms = mol_sums * inv_mol_size[:, None].astype(pos.dtype)
    diffs = jnp.zeros((n_mols, 3), dtype=pos.dtype)
    cz_ok = jnp.abs(c_vec[2]) > 1e-12
    diffs = jnp.where(
        cz_ok,
        diffs + jnp.floor(coms[:, 2:3] / jnp.where(cz_ok, c_vec[2], 1.0)) * c_vec[None, :],
        diffs,
    )
    by_ok = jnp.abs(b_vec[1]) > 1e-12
    diffs = jnp.where(
        by_ok,
        diffs
        + jnp.floor((coms[:, 1:2] - diffs[:, 1:2]) / jnp.where(by_ok, b_vec[1], 1.0))
        * b_vec[None, :],
        diffs,
    )
    ax_ok = jnp.abs(a_vec[0]) > 1e-12
    diffs = jnp.where(
        ax_ok,
        diffs
        + jnp.floor((coms[:, 0:1] - diffs[:, 0:1]) / jnp.where(ax_ok, a_vec[0], 1.0))
        * a_vec[None, :],
        diffs,
    )
    return pos - diffs[mol_of_atom]


def _pack_position_restraints(
    position_restraints: tuple[PositionRestraints, ...],
    n_act: int,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Flatten all PositionRestraints into 1D dynamic GPU buffers (AKMA Å and kcal/(mol·Å²))."""
    if not position_restraints:
        z_i = jnp.zeros((0,), dtype=jnp.int32)
        z_f = jnp.zeros((0,), dtype=jnp.float32)
        z_b = jnp.zeros((0,), dtype=jnp.bool_)
        return z_i, z_f, z_f, z_f, z_f, z_f, z_f, z_b
    idx_parts: list[jnp.ndarray] = []
    kx_parts: list[jnp.ndarray] = []
    ky_parts: list[jnp.ndarray] = []
    kz_parts: list[jnp.ndarray] = []
    x0_parts: list[jnp.ndarray] = []
    y0_parts: list[jnp.ndarray] = []
    z0_parts: list[jnp.ndarray] = []
    for pr in position_restraints:
        if int(pr.indices.shape[0]) == 0:
            continue
        idx_parts.append(jnp.asarray(pr.indices, dtype=jnp.int32).reshape(-1))
        kx_parts.append(jnp.asarray(pr.k_x, dtype=jnp.float32).reshape(-1))
        ky_parts.append(jnp.asarray(pr.k_y, dtype=jnp.float32).reshape(-1))
        kz_parts.append(jnp.asarray(pr.k_z, dtype=jnp.float32).reshape(-1))
        x0_parts.append(jnp.asarray(pr.x0, dtype=jnp.float32).reshape(-1))
        y0_parts.append(jnp.asarray(pr.y0, dtype=jnp.float32).reshape(-1))
        z0_parts.append(jnp.asarray(pr.z0, dtype=jnp.float32).reshape(-1))
    if not idx_parts:
        z_i = jnp.zeros((0,), dtype=jnp.int32)
        z_f = jnp.zeros((0,), dtype=jnp.float32)
        z_b = jnp.zeros((0,), dtype=jnp.bool_)
        return z_i, z_f, z_f, z_f, z_f, z_f, z_f, z_b
    idx_all = jnp.concatenate(idx_parts, axis=0)
    valid_all = (idx_all >= 0) & (idx_all < int(n_act))
    safe_idx = jnp.where(valid_all, idx_all, jnp.int32(0))
    return (
        safe_idx,
        jnp.concatenate(kx_parts, axis=0),
        jnp.concatenate(ky_parts, axis=0),
        jnp.concatenate(kz_parts, axis=0),
        jnp.concatenate(x0_parts, axis=0),
        jnp.concatenate(y0_parts, axis=0),
        jnp.concatenate(z0_parts, axis=0),
        valid_all,
    )


def _pack_bundle_fused_bonded(bundle: MolecularBundle) -> FusedBondedBuffers:
    """Pack active MolecularBundle bonded terms into 1-replica FusedBondedBuffers."""
    has_b = bundle.n_bonds is not None and int(bundle.n_bonds) > 0
    has_ub = bundle.n_urey_bradley is not None and int(bundle.n_urey_bradley) > 0
    has_a = bundle.n_angles is not None and int(bundle.n_angles) > 0
    has_d = bundle.n_dihedrals is not None and int(bundle.n_dihedrals) > 0
    has_i = bundle.n_impropers is not None and int(bundle.n_impropers) > 0
    has_e = bundle.n_exception_pairs is not None and int(bundle.n_exception_pairs) > 0
    has_c = bundle.n_cmap is not None and int(bundle.n_cmap) > 0
    imp_harm = bool(
        has_i
        and bundle.improper_params is not None
        and (
            bundle.improper_params.shape[-1] == 2
            or (
                hasattr(bundle, "improper_is_periodic")
                and bundle.improper_is_periodic is not None
                and not bool(np.asarray(bundle.improper_is_periodic))
            )
        )
    )
    dih_p3 = (
        bundle.dihedral_params[..., :3]
        if (has_d and bundle.dihedral_params is not None)
        else None
    )
    imp_p = (
        (bundle.improper_params[..., :2] if imp_harm else bundle.improper_params[..., :3])
        if (has_i and bundle.improper_params is not None)
        else None
    )
    return prepare_fused_bonded_buffers(
        1,
        bonds=bundle.bond_idx if has_b else None,
        bond_params=bundle.bond_params if has_b else None,
        bond_mask=bundle.bond_mask if has_b else None,
        urey_bradley_bonds=bundle.urey_bradley_idx if has_ub else None,
        urey_bradley_params=bundle.urey_bradley_params if has_ub else None,
        urey_bradley_mask=bundle.urey_bradley_mask if has_ub else None,
        angles=bundle.angle_idx if has_a else None,
        angle_params=bundle.angle_params if has_a else None,
        angle_mask=bundle.angle_mask if has_a else None,
        dihedrals=bundle.dihedral_idx if has_d else None,
        dihedral_params=dih_p3,
        dihedral_mask=bundle.dihedral_mask if has_d else None,
        impropers=bundle.improper_idx if has_i else None,
        improper_params=imp_p,
        improper_mask=bundle.improper_mask if has_i else None,
        improper_is_harmonic=imp_harm,
        exception_pairs=bundle.exception_pairs if has_e else None,
        exception_sigmas=bundle.exception_sigmas if has_e else None,
        exception_epsilons=bundle.exception_epsilons if has_e else None,
        exception_chargeprods=bundle.exception_chargeprods if has_e else None,
        exception_mask=bundle.exception_mask if has_e else None,
        cmap_torsions=bundle.cmap_torsion_idx if has_c else None,
        cmap_indices=bundle.cmap_indices if has_c else None,
        cmap_coeffs=bundle.cmap_coeffs if has_c else None,
        cmap_mask=bundle.cmap_mask if has_c else None,
    )


def _build_context_morton_runner(  # noqa: PLR0913
    bundle: MolecularBundle,
    *,
    n_act: int,
    precision: str,
    pallas_interpret: bool,
    cutoff: float,
    nl_skin: float,
    lj_switch_width: float,
    tp_group: int | None,
    do_pme: bool,
    pme_alpha: float,
    grid_dims: tuple[int, int, int],
    is_rf: bool,
    rf_dielectric: float,
    gamma_akma: float,
    nl_freq: int,
    cmm_freq: int,
    has_barostat: bool,
    baro_mode: str,
    baro_freq: int,
    target_p_akma: float,
    r_oh_akma: float,
    r_hh_akma: float,
    m_o_akma: float,
    m_h_akma: float,
    anchor_jnp: jnp.ndarray,
    mol_jnp: jnp.ndarray,
    inv_mol_size_jnp: jnp.ndarray,
    host_box0_np: np.ndarray,
    init_positions: jnp.ndarray,
    lj_combining_rule: str = "lorentz_berthelot",
    is_lj_pme: bool = False,
    lj_pme_alpha: float = 0.0,
    lj_pme_grid_dims: tuple[int, int, int] = (12, 12, 12),
    deterministic_forces: bool = False,
    has_restraints: bool = False,
    virtual_sites: VirtualSites | None = None,
    pr_params_getter: Any = None,
    is_triclinic: bool = False,
    baro_type: str = "MonteCarloBarostat",
    baro_scale_axes: tuple[bool, bool, bool] = (True, True, True),
    baro_pressure_xyz_akma: tuple[float, float, float] | None = None,
    baro_surface_tension_akma: float = 0.0,
    baro_xy_isotropic: bool = True,
    baro_z_mode: int = 0,
    has_custom_forces: bool = False,
    param_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a persistent B=1 GPU-resident BAOAB + Morton 32x32 tile-pair + SPME/LJPME/RF + VS/PR/CF + NPT runner."""
    B = 1
    N = int(bundle.positions.shape[0])
    n_act_blocks = max(1, (n_act + 31) // 32)
    n_act_pad = n_act_blocks * 32
    pad_tail = n_act_pad - n_act
    pad_N = N - n_act

    if precision == "mixed":
        state_dtype: Any = jnp.float64
        f64_acc = True
    else:
        state_dtype = jnp.float32
        f64_acc = False

    interpret = (
        bool(pallas_interpret)
        or not _triton_gpu_available()
        or jax.default_backend() == "cpu"
    )
    n_waters_real = (
        int(np.sum(np.asarray(bundle.water_mask)))
        if bundle.shape_spec.has_real_water
        else 0
    )
    mass_b = bundle.masses[:N].astype(state_dtype)[None, :]
    mass_act = bundle.masses[:n_act].astype(state_dtype)
    inv_mass_act = jnp.where(
        mass_act > 1e-12, 1.0 / jnp.where(mass_act > 1e-12, mass_act, 1.0), 0.0
    )
    # Full active-particle mask (includes virtual sites for charges/LJ/SPME/LJPME)
    am_b = bundle.atom_mask[:N].astype(jnp.bool_)[None, :]
    # Real-mass mask (excludes mass == 0 virtual sites from BAOAB kicks, OU noise, and COM removal)
    am_real_b = am_b & (mass_b > 1e-12)
    am_real_act = am_real_b[0, :n_act]

    has_vs = virtual_sites is not None and virtual_sites.n_total > 0
    vs_order = (
        virtual_sites.order
        if (has_vs and virtual_sites is not None)
        else jnp.zeros((0,), dtype=jnp.int32)
    )

    if bundle.shape_spec.has_real_water and bundle.water_indices is not None:
        wi_b = bundle.water_indices.astype(jnp.int32)[None, :, :]
        wm_b = bundle.water_mask.astype(jnp.int32)[None, :]
    else:
        wi_b = jnp.zeros((1, 0, 3), dtype=jnp.int32)
        wm_b = jnp.zeros((1, 0), dtype=jnp.int32)

    has_cons = bool(
        getattr(bundle.shape_spec, "has_real_constraints", False)
        and bundle.constraint_pairs is not None
        and int(bundle.constraint_pairs.shape[0]) > 0
    )
    cp_b = (
        bundle.constraint_pairs.astype(jnp.int32)[None, :, :]
        if (has_cons and bundle.constraint_pairs is not None)
        else None
    )
    cl_b = (
        bundle.constraint_lengths.astype(state_dtype)[None, :]
        if (has_cons and bundle.constraint_lengths is not None)
        else None
    )
    cm_b = (
        bundle.constraint_mask.astype(jnp.int32)[None, :]
        if (has_cons and bundle.constraint_mask is not None)
        else None
    )

    soa_plan = build_baoab_soa_static_plan(
        B,
        N,
        state_dtype,
        mass_b,
        wi_b,
        wm_b,
        atom_mask_b=am_real_b,
        constraint_pairs_b=cp_b,
        constraint_lengths_b=cl_b,
        constraint_mask_b=cm_b,
        n_active_atoms=n_act,
        n_active_waters=n_waters_real,
    )

    eff_skin = max(float(nl_skin), 2.0)
    thresh_sq = float((0.5 * eff_skin) ** 2)
    npt_safety = 1.15 if has_barostat else 1.0
    max_pairs = estimate_max_tile_pairs(
        n_act_blocks, cutoff=cutoff, skin=eff_skin, box_safety=npt_safety
    )
    host_box0_f32_np = np.asarray(host_box0_np, dtype=np.float32)
    host_box0_f32_dev = jnp.asarray(host_box0_f32_np, dtype=jnp.float32)
    host_min_box = (
        float(np.min(np.diag(host_box0_np)))
        if host_box0_np.ndim == 2
        else float(np.min(host_box0_np))
    )
    dedup_excl = precompute_dedup_excl_edges(
        bundle.excl_dense_indices,
        bundle.excl_dense_scales_vdw,
        bundle.excl_dense_scales_elec,
        n_act,
    )

    bonded_static: FusedBondedBuffers = _pack_bundle_fused_bonded(bundle)

    if do_pme:
        G_init = jnp.asarray(
            _cached_host_influence_function_np(
                grid_dims,
                tuple(float(x) for x in host_box0_f32_np.reshape(-1)),
                host_box0_f32_np.shape,
                "float32",
                pme_alpha,
                4,
            ),
            dtype=jnp.float32,
        )[None, ...]
        n_grid_f = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
        pme_force_scale = -float(COULOMB_CONSTANT) * n_grid_f
    else:
        G_init = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
        n_grid_f = 1.0
        pme_force_scale = 0.0

    chunk_k = max(1, int(nl_freq) if int(nl_freq) > 0 else 20)
    if has_barostat:
        chunk_k = min(chunk_k, max(1, int(baro_freq)))

    lam_flat = jnp.ones((1,), dtype=jnp.float32)
    in_b_pre = (True,) * (12 + (3 if has_cons else 0))
    dummy_pos_b = jnp.zeros((1, N, 3), dtype=state_dtype)
    n_mols = int(inv_mol_size_jnp.shape[0])
    eff_lj_sw = 0.0 if is_lj_pme else float(lj_switch_width)

    soa_params = _get_pallas_soa_sig_params()
    soa_has_rf_kw = "rf_dielectric" in soa_params
    soa_has_comb_kw = "lj_combining_rule" in soa_params
    soa_has_ljpme_kw = (
        "lj_pme_alpha" in soa_params and "include_ljpme_excl" in soa_params
    )
    soa_has_det_kw = "deterministic_forces" in soa_params
    soa_has_tri_kw = "is_triclinic" in soa_params
    tp_build_has_tri_kw = (
        "is_triclinic" in inspect.signature(pallas_build_morton_tile_pairs_gpu).parameters
    )
    spread_has_tri_kw = (
        "is_triclinic" in inspect.signature(_spread_charges_pallas_soa).parameters
    )
    gather_has_tri_kw = (
        "is_triclinic" in inspect.signature(_gather_pallas_soa).parameters
    )
    pre_has_tri_kw = (
        "is_triclinic" in inspect.signature(_pallas_baoab_pre_force_batch).parameters
    )
    post_soa_params = inspect.signature(_pallas_baoab_post_soa).parameters
    post_has_box_kw = "box" in post_soa_params and "is_triclinic" in post_soa_params

    active_aniso_axes = tuple(ax for ax in range(3) if baro_scale_axes[ax])
    eff_p_xyz = (
        baro_pressure_xyz_akma
        if baro_pressure_xyz_akma is not None
        else (target_p_akma, target_p_akma, target_p_akma)
    )

    def _inv_for_n(inv_pad: jax.Array) -> jax.Array:
        pw = int(inv_pad.shape[1])
        return inv_pad[:, :N] if pw >= N else jnp.pad(inv_pad, ((0, 0), (0, N - pw)))

    def _pad_atom_1d(arr_n: jax.Array, pad_val: float = 0.0) -> jax.Array:
        arr_act = arr_n[:, :n_act]
        if pad_tail > 0:
            return jnp.pad(arr_act, ((0, 0), (0, pad_tail)), constant_values=pad_val)
        return arr_act

    def _mic_dr(dr: jax.Array, bx_0: jax.Array) -> jax.Array:
        if is_triclinic:
            a_v, b_v, c_v = bx_0[0], bx_0[1], bx_0[2]
            d1 = dr - jnp.round(dr[:, 2:3] / c_v[2]) * c_v[None, :]
            d2 = d1 - jnp.round(d1[:, 1:2] / b_v[1]) * b_v[None, :]
            return d2 - jnp.round(d2[:, 0:1] / a_v[0]) * a_v[None, :]
        return dr - bx_0[None, :] * jnp.round(dr / bx_0[None, :])

    def _wrap_pos(pos_any: jax.Array, bx_0: jax.Array) -> jax.Array:
        if is_triclinic:
            a_v, b_v, c_v = bx_0[0], bx_0[1], bx_0[2]
            r1 = pos_any - jnp.floor(pos_any[:, 2:3] / c_v[2]) * c_v[None, :]
            r2 = r1 - jnp.floor(r1[:, 1:2] / b_v[1]) * b_v[None, :]
            return r2 - jnp.floor(r2[:, 0:1] / a_v[0]) * a_v[None, :]
        return jnp.mod(pos_any, bx_0[None, :])

    def _vol_f32(bx_0: jax.Array) -> jax.Array:
        bx_f = bx_0.astype(jnp.float32)
        if is_triclinic:
            return bx_f[0, 0] * bx_f[1, 1] * bx_f[2, 2]
        return jnp.prod(bx_f)

    def _area_xy_f32(bx_0: jax.Array) -> jax.Array:
        bx_f = bx_0.astype(jnp.float32)
        if is_triclinic:
            return jnp.abs(bx_f[0, 0] * bx_f[1, 1])
        return jnp.abs(bx_f[0] * bx_f[1])

    def _scale_box_axes(bx_b_in: jax.Array, axis_scales: jax.Array) -> jax.Array:
        if is_triclinic:
            return reduce_triclinic_box_vectors(
                bx_b_in[0] * axis_scales[:, None]
            )[None, :, :]
        return bx_b_in * axis_scales[None, :]

    def _greens_for_box(bx_0: jax.Array) -> jax.Array:
        if not do_pme:
            return G_init
        return scale_influence_function_isotropic(
            G_init[0],
            host_box0_f32_dev,
            bx_0.astype(jnp.float32),
            alpha=pme_alpha,
            grid_dims=grid_dims,
            order=4,
        ).astype(jnp.float32)[None, ...]

    def _build_tp_gpu(
        pos_pad_arg: jax.Array,
        q_arg: jax.Array,
        sig_arg: jax.Array,
        eps_arg: jax.Array,
        bx_b_arg: jax.Array,
    ) -> MortonTilePairState:
        tp_kw: dict[str, Any] = {}
        if is_triclinic and tp_build_has_tri_kw:
            tp_kw["is_triclinic"] = True
        return pallas_build_morton_tile_pairs_gpu(
            pos_pad_arg,
            q_arg,
            sig_arg,
            eps_arg,
            am_b,
            bx_b_arg,
            dedup_excl,
            n_act=n_act,
            max_pairs=max_pairs,
            cutoff=cutoff,
            skin=eff_skin,
            host_min_box=host_min_box,
            interpret=interpret,
            **tp_kw,
        )

    def _unwrap_whole_and_reconstruct_vs(
        pos_any: jax.Array, box_d0: jax.Array
    ) -> jax.Array:
        anc = pos_any[anchor_jnp]
        intra = _mic_dr(pos_any - anc, box_d0)
        whole = anc + intra
        if has_vs and virtual_sites is not None:
            whole = virtual_sites.reconstruct_positions(whole)
        return whole

    def _sync_vs_from_wrapped_and_unw(
        px_v: jax.Array,
        py_v: jax.Array,
        pz_v: jax.Array,
        pos_unw_v: jax.Array,
        box_d0: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        if not has_vs or virtual_sites is None:
            return px_v, py_v, pz_v, pos_unw_v, pos_unw_v
        pos_wr_act = jnp.stack([px_v[:n_act], py_v[:n_act], pz_v[:n_act]], axis=-1)
        pos_whole = _unwrap_whole_and_reconstruct_vs(pos_wr_act, box_d0)
        vs_wr = _wrap_pos(pos_whole[vs_order], box_d0)
        px_v = px_v.at[vs_order].set(vs_wr[:, 0])
        py_v = py_v.at[vs_order].set(vs_wr[:, 1])
        pz_v = pz_v.at[vs_order].set(vs_wr[:, 2])
        pos_unw_whole = _unwrap_whole_and_reconstruct_vs(pos_unw_v, box_d0)
        pos_unw_v = pos_unw_v.at[vs_order].set(pos_unw_whole[vs_order])
        return px_v, py_v, pz_v, pos_unw_v, pos_whole

    def _permute_wrapped_to_morton(
        px_v: jax.Array,
        py_v: jax.Array,
        pz_v: jax.Array,
        perm_b: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        p_wr_f32 = jnp.stack(
            [px_v[:n_act], py_v[:n_act], pz_v[:n_act]], axis=-1
        ).astype(jnp.float32)[None, :, :]
        if pad_tail > 0:
            p_wr_f32 = jnp.pad(p_wr_f32, ((0, 0), (0, pad_tail), (0, 0)))
        pm_cur = jnp.take_along_axis(p_wr_f32, perm_b[:, :, None], axis=1)
        return (
            pm_cur[:, :, 0].reshape(n_act_pad),
            pm_cur[:, :, 1].reshape(n_act_pad),
            pm_cur[:, :, 2].reshape(n_act_pad),
        )

    def _make_eval_morton_forces(
        bonded_dyn: FusedBondedBuffers,
        e_self_b: jax.Array,
        e_lj_self_b: jax.Array,
        pr_tuple: tuple[jax.Array, ...],
        cf_tuple: tuple[CompiledLeptonForce, ...],
        gp_vec: jax.Array,
    ):
        (
            pr_idx_in,
            pr_kx_in,
            pr_ky_in,
            pr_kz_in,
            pr_x0_in,
            pr_y0_in,
            pr_z0_in,
            pr_valid_in,
        ) = pr_tuple

        def _eval_morton_forces(
            px_m: jax.Array,
            py_m: jax.Array,
            pz_m: jax.Array,
            tp_st: MortonTilePairState,
            box_b_cur: jax.Array,
            G_b_cur: jax.Array,
            pos_unw_cur: jax.Array,
            pos_whole_cur: jax.Array,
            compute_energy: bool = False,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            box_f32_flat = box_b_cur.astype(jnp.float32).reshape(
                9 if is_triclinic else 3
            )
            soa_extra: dict[str, Any] = {}
            if is_rf and soa_has_rf_kw:
                soa_extra["rf_dielectric"] = float(rf_dielectric)
            if soa_has_comb_kw:
                soa_extra["lj_combining_rule"] = str(lj_combining_rule)
            if is_lj_pme and soa_has_ljpme_kw:
                soa_extra["lj_pme_alpha"] = float(lj_pme_alpha)
                soa_extra["include_ljpme_excl"] = True
            if soa_has_det_kw:
                soa_extra["deterministic_forces"] = bool(deterministic_forces)
            if is_triclinic and soa_has_tri_kw:
                soa_extra["is_triclinic"] = True
            e_dir, fx_d, fy_d, fz_d = _pallas_bonded_and_tile_pairs_soa(
                1,
                px_m,
                py_m,
                pz_m,
                tp_st.q_m.reshape(n_act_pad),
                tp_st.sig_m.reshape(n_act_pad),
                tp_st.eps_m.reshape(n_act_pad),
                tp_st.mask_m.reshape(n_act_pad),
                tp_st.inv_perm.reshape(n_act_pad),
                tp_st.pair_ti.reshape(max_pairs),
                tp_st.pair_tj.reshape(max_pairs),
                tp_st.excl_full_bits.reshape(max_pairs * 32),
                tp_st.excl_14_bits.reshape(max_pairs * 32),
                tp_st.n_valid_pairs.reshape(1),
                box_f32_flat,
                lam_flat,
                bonded_dyn,
                n_act=n_act,
                n_act_blocks=n_act_blocks,
                max_pairs=max_pairs,
                pme_alpha=(pme_alpha if do_pme else 0.0),
                cutoff=cutoff,
                lj_switch_width=eff_lj_sw,
                include_pme_excl_erf=do_pme,
                scale_14_vdw=dedup_excl.scale_14_vdw,
                scale_14_elec=dedup_excl.scale_14_elec,
                cmap_first_is_psi=False,
                tp_group=tp_group,
                compute_energy=compute_energy,
                interpret=interpret,
                **soa_extra,
            )
            if do_pme:
                q_mf = tp_st.q_m.reshape(n_act_pad)
                m_mf = tp_st.mask_m.reshape(n_act_pad)
                sp_kw = {"is_triclinic": True} if (is_triclinic and spread_has_tri_kw) else {}
                ga_kw = {"is_triclinic": True} if (is_triclinic and gather_has_tri_kw) else {}
                mesh = _spread_charges_pallas_soa(
                    1,
                    (True, True, True, True),
                    px_m,
                    py_m,
                    pz_m,
                    q_mf,
                    m_mf,
                    box_f32_flat,
                    grid_dims,
                    n_act_pad,
                    n_act_pad,
                    interpret,
                    **sp_kw,
                )
                q_hat = jnp.fft.rfftn(mesh, s=grid_dims, axes=(-3, -2, -1))
                theta = jnp.fft.irfftn(
                    q_hat * G_b_cur, s=grid_dims, axes=(-3, -2, -1)
                ).astype(jnp.float32)
                fx_tot, fy_tot, fz_tot, _ = _gather_pallas_soa(
                    1,
                    (True, True, True, True, True),
                    px_m,
                    py_m,
                    pz_m,
                    q_mf,
                    m_mf,
                    box_f32_flat,
                    theta.reshape(-1),
                    grid_dims,
                    n_act_pad,
                    n_act_pad,
                    n_act_blocks,
                    interpret,
                    force_scale=pme_force_scale,
                    force_accum=(fx_d, fy_d, fz_d),
                    **ga_kw,
                )
                if compute_energy:
                    e_rec = (
                        0.5
                        * float(COULOMB_CONSTANT)
                        * jnp.sum(mesh * theta, axis=(-3, -2, -1))
                        * n_grid_f
                        + e_self_b
                    )
                    e_tot = e_dir + e_rec
                else:
                    e_tot = e_dir
            else:
                e_tot, fx_tot, fy_tot, fz_tot = e_dir, fx_d, fy_d, fz_d

            if is_lj_pme:
                sig_mf = tp_st.sig_m.reshape(n_act_pad)
                eps_mf = tp_st.eps_m.reshape(n_act_pad)
                m_mf_bool = tp_st.mask_m.reshape(n_act_pad) > 0
                b_mf = dispersion_charges(sig_mf, eps_mf, m_mf_bool)
                pos_m_3 = jnp.stack([px_m, py_m, pz_m], axis=-1)
                box_lj_f32 = box_b_cur[0].astype(jnp.float32)

                def _lj_rec_fn(p_m_in: jax.Array) -> jax.Array:
                    return lj_pme_reciprocal_energy(
                        p_m_in,
                        b_mf,
                        m_mf_bool,
                        box_lj_f32,
                        lj_pme_grid_dims,
                        alpha_lj=lj_pme_alpha,
                        order=5,
                    )

                if compute_energy:
                    e_lj_rec, g_lj_m = jax.value_and_grad(_lj_rec_fn)(pos_m_3)
                    e_tot = e_tot + (e_lj_rec.astype(jnp.float32) + e_lj_self_b[0])[None]
                else:
                    g_lj_m = jax.grad(_lj_rec_fn)(pos_m_3)
                fx_tot = fx_tot - g_lj_m[:, 0].astype(jnp.float32)
                fy_tot = fy_tot - g_lj_m[:, 1].astype(jnp.float32)
                fz_tot = fz_tot - g_lj_m[:, 2].astype(jnp.float32)

            if has_restraints:
                dx_pr = (
                    pos_unw_cur[pr_idx_in, 0] - pr_x0_in.astype(state_dtype)
                ).astype(jnp.float32)
                dy_pr = (
                    pos_unw_cur[pr_idx_in, 1] - pr_y0_in.astype(state_dtype)
                ).astype(jnp.float32)
                dz_pr = (
                    pos_unw_cur[pr_idx_in, 2] - pr_z0_in.astype(state_dtype)
                ).astype(jnp.float32)
                fx_pr = jnp.where(pr_valid_in, -2.0 * pr_kx_in * dx_pr, 0.0)
                fy_pr = jnp.where(pr_valid_in, -2.0 * pr_ky_in * dy_pr, 0.0)
                fz_pr = jnp.where(pr_valid_in, -2.0 * pr_kz_in * dz_pr, 0.0)
                m_pr = jnp.clip(tp_st.inv_perm[0, pr_idx_in], 0, n_act_pad - 1)
                fx_tot = fx_tot.at[m_pr].add(fx_pr)
                fy_tot = fy_tot.at[m_pr].add(fy_pr)
                fz_tot = fz_tot.at[m_pr].add(fz_pr)
                if compute_energy:
                    e_pr = jnp.sum(
                        jnp.where(
                            pr_valid_in,
                            pr_kx_in * dx_pr * dx_pr
                            + pr_ky_in * dy_pr * dy_pr
                            + pr_kz_in * dz_pr * dz_pr,
                            0.0,
                        )
                    ).astype(jnp.float32)
                    e_tot = e_tot + e_pr[None]

            if has_custom_forces and len(cf_tuple) > 0:
                gp_dict = {
                    nm: gp_vec[i_gp] for i_gp, nm in enumerate(param_names)
                }
                box_cf_mat = (
                    box_b_cur[0].astype(jnp.float32)
                    if is_triclinic
                    else jnp.diag(box_b_cur[0].astype(jnp.float32))
                )
                pos_cf_f32 = pos_unw_cur.astype(jnp.float32)

                def _cf_energy_sum(p_cf: jax.Array) -> jax.Array:
                    e_cf_acc = jnp.asarray(0.0, dtype=jnp.float32)
                    for cf_obj in cf_tuple:
                        e_cf_acc = e_cf_acc + cf_obj.energy_akma(
                            p_cf, box_cf_mat, gp_dict
                        ).astype(jnp.float32)
                    return e_cf_acc

                if compute_energy:
                    e_cf_val, g_cf = jax.value_and_grad(_cf_energy_sum)(pos_cf_f32)
                    e_tot = e_tot + e_cf_val[None]
                else:
                    g_cf = jax.grad(_cf_energy_sum)(pos_cf_f32)
                f_cf_pad = (
                    jnp.pad(-g_cf, ((0, pad_tail), (0, 0)))[None, :, :]
                    if pad_tail > 0
                    else (-g_cf)[None, :, :]
                )
                fm_cf = jnp.take_along_axis(
                    f_cf_pad, tp_st.perm[:, :, None], axis=1
                )
                fx_tot = fx_tot + fm_cf[:, :, 0].reshape(n_act_pad)
                fy_tot = fy_tot + fm_cf[:, :, 1].reshape(n_act_pad)
                fz_tot = fz_tot + fm_cf[:, :, 2].reshape(n_act_pad)

            if has_vs and virtual_sites is not None:
                inv_act_vs = tp_st.inv_perm[0, :n_act]
                f_raw_act = jnp.stack(
                    [fx_tot[inv_act_vs], fy_tot[inv_act_vs], fz_tot[inv_act_vs]],
                    axis=-1,
                )
                f_red_act = virtual_sites.redistribute_forces(
                    f_raw_act, pos_whole_cur.astype(jnp.float32)
                )
                f_red_pad = (
                    jnp.pad(f_red_act, ((0, pad_tail), (0, 0)))[None, :, :]
                    if pad_tail > 0
                    else f_red_act[None, :, :]
                )
                fm_red = jnp.take_along_axis(f_red_pad, tp_st.perm[:, :, None], axis=1)
                fx_tot = fm_red[:, :, 0].reshape(n_act_pad)
                fy_tot = fm_red[:, :, 1].reshape(n_act_pad)
                fz_tot = fm_red[:, :, 2].reshape(n_act_pad)

            return e_tot, fx_tot, fy_tot, fz_tot

        return _eval_morton_forces

    host_box0_dev = jnp.asarray(host_box0_np, dtype=state_dtype)
    host_box0_b_dev = (
        host_box0_dev[None, :, :] if is_triclinic else host_box0_dev[None, :]
    )
    init_pos_act = init_positions[:n_act].astype(state_dtype)
    if has_vs and virtual_sites is not None:
        init_pos_act = _unwrap_whole_and_reconstruct_vs(init_pos_act, host_box0_dev)
    init_pos_wr = _wrap_pos(init_pos_act, host_box0_dev)
    init_pos_pad = (
        jnp.pad(init_pos_wr, ((0, pad_N), (0, 0)))[None, :, :]
        if pad_N > 0
        else init_pos_wr[None, :, :]
    )
    tp_init = _build_tp_gpu(
        init_pos_pad,
        bundle.charges[:N].astype(jnp.float32)[None, :],
        bundle.sigmas[:N].astype(jnp.float32)[None, :],
        bundle.epsilons[:N].astype(jnp.float32)[None, :],
        host_box0_b_dev,
    )
    z_f = jnp.zeros((n_act_pad,), dtype=jnp.float32)
    empty_gp_vec = jnp.zeros((len(param_names),), dtype=jnp.float32)

    @jax.jit
    def _jit_step(  # noqa: PLR0913
        pos_in: jax.Array,
        vel_in: jax.Array,
        key_in: jax.Array,
        step_offset_in: int | jax.Array,
        n_steps_in: int | jax.Array,
        dt_s_in: float | jax.Array,
        kT_s_in: float | jax.Array,
        box_diag_in: jax.Array,
        q_b_in: jax.Array,
        sig_b_in: jax.Array,
        eps_b_in: jax.Array,
        b_r0_in: jax.Array,
        b_k_in: jax.Array,
        a_th0_in: jax.Array,
        a_kval_in: jax.Array,
        d_p0_in: jax.Array,
        d_p1_in: jax.Array,
        d_p2_in: jax.Array,
        e_sig_in: jax.Array,
        e_eps_in: jax.Array,
        e_qp_in: jax.Array,
        tp_in: MortonTilePairState,
        fx_in: jax.Array,
        fy_in: jax.Array,
        fz_in: jax.Array,
        G_in: jax.Array,
        nl_dirty_in: bool | jax.Array,
        forces_dirty_in: bool | jax.Array,
        mc_scale_in: jax.Array,
        mc_att_in: jax.Array,
        mc_acc_in: jax.Array,
        c_tail_in: jax.Array,
        pr_idx_in: jax.Array,
        pr_kx_in: jax.Array,
        pr_ky_in: jax.Array,
        pr_kz_in: jax.Array,
        pr_x0_in: jax.Array,
        pr_y0_in: jax.Array,
        pr_z0_in: jax.Array,
        pr_valid_in: jax.Array,
        custom_forces_in: tuple[CompiledLeptonForce, ...] = (),
        gp_vec_in: jax.Array | None = None,
    ):
        step_offset_i32 = jnp.asarray(step_offset_in, dtype=jnp.int32)
        n_steps_i32 = jnp.asarray(n_steps_in, dtype=jnp.int32)
        nl_dirty_b = jnp.asarray(nl_dirty_in, dtype=jnp.bool_)
        forces_dirty_b = jnp.asarray(forces_dirty_in, dtype=jnp.bool_)
        dt_b = jnp.broadcast_to(jnp.asarray(dt_s_in, dtype=state_dtype), (1,))
        kT_b = jnp.broadcast_to(jnp.asarray(kT_s_in, dtype=state_dtype), (1,))
        box_b0 = (
            box_diag_in.astype(state_dtype)[None, :, :]
            if is_triclinic
            else box_diag_in.astype(state_dtype)[None, :]
        )
        mc_sc0 = jnp.broadcast_to(jnp.asarray(mc_scale_in, dtype=jnp.float32), (3,))
        mc_at0 = jnp.broadcast_to(jnp.asarray(mc_att_in, dtype=jnp.int32), (3,))
        mc_ac0 = jnp.broadcast_to(jnp.asarray(mc_acc_in, dtype=jnp.int32), (3,))
        eff_gp_vec = empty_gp_vec if gp_vec_in is None else gp_vec_in
        bonded_dyn = bonded_static._replace(
            b_r0=b_r0_in,
            b_k=b_k_in,
            a_th0=a_th0_in,
            a_kval=a_kval_in,
            d_p0=d_p0_in,
            d_p1=d_p1_in,
            d_p2=d_p2_in,
            e_sig=e_sig_in,
            e_eps=e_eps_in,
            e_qp=e_qp_in,
        )
        e_self_b = (
            spme_self_energy(q_b_in, am_b, alpha=pme_alpha).astype(jnp.float32)
            if do_pme
            else jnp.zeros((1,), dtype=jnp.float32)
        )
        e_lj_self_b = (
            lj_pme_self_energy(
                dispersion_charges(sig_b_in, eps_b_in, am_b),
                atom_mask=am_b,
                alpha_lj=lj_pme_alpha,
            ).astype(jnp.float32)
            if is_lj_pme
            else jnp.zeros((1,), dtype=jnp.float32)
        )
        q_net = jnp.sum(jnp.where(am_b, q_b_in, 0.0), axis=-1)
        c_plasma_b = (
            -(jnp.pi * float(COULOMB_CONSTANT) / (2.0 * (pme_alpha**2))) * (q_net * q_net)
            if do_pme
            else jnp.zeros((1,), dtype=jnp.float32)
        )
        _eval_morton_forces = _make_eval_morton_forces(
            bonded_dyn,
            e_self_b,
            e_lj_self_b,
            (
                pr_idx_in,
                pr_kx_in,
                pr_ky_in,
                pr_kz_in,
                pr_x0_in,
                pr_y0_in,
                pr_z0_in,
                pr_valid_in,
            ),
            custom_forces_in,
            eff_gp_vec,
        )

        pos_unw0 = pos_in.astype(state_dtype)
        if has_vs and virtual_sites is not None:
            pos_whole0 = _unwrap_whole_and_reconstruct_vs(pos_unw0, box_b0[0])
            pos_unw0 = pos_unw0.at[vs_order].set(pos_whole0[vs_order])
            pos_wr0 = _wrap_pos(pos_whole0, box_b0[0])
            vel_act0 = vel_in.astype(state_dtype).at[vs_order].set(0.0)
        else:
            pos_whole0 = pos_unw0
            pos_wr0 = _wrap_pos(pos_unw0, box_b0[0])
            vel_act0 = vel_in.astype(state_dtype)

        mom_act0 = vel_act0 * mass_act[:, None]
        if pad_N > 0:
            pos_pad0 = jnp.pad(pos_wr0, ((0, pad_N), (0, 0)))[None, :, :]
            mom_pad0 = jnp.pad(mom_act0, ((0, pad_N), (0, 0)))[None, :, :]
        else:
            pos_pad0 = pos_wr0[None, :, :]
            mom_pad0 = mom_act0[None, :, :]

        def _entry_rebuild_nl(_: Any):
            tp_r = _build_tp_gpu(pos_pad0, q_b_in, sig_b_in, eps_b_in, box_b0)
            G_r = _greens_for_box(box_b0[0]) if do_pme else G_in
            return tp_r, G_r

        def _entry_keep_nl(_: Any):
            def _refresh_params(_u: Any) -> MortonTilePairState:
                qp = _pad_atom_1d(q_b_in, 0.0)
                sp = _pad_atom_1d(sig_b_in, 1.0)
                ep = _pad_atom_1d(eps_b_in, 0.0)
                return MortonTilePairState(
                    tp_in.perm,
                    tp_in.inv_perm,
                    jnp.take_along_axis(qp, tp_in.perm, axis=1),
                    jnp.take_along_axis(sp, tp_in.perm, axis=1),
                    jnp.take_along_axis(ep, tp_in.perm, axis=1),
                    tp_in.mask_m,
                    tp_in.pair_ti,
                    tp_in.pair_tj,
                    tp_in.excl_full_bits,
                    tp_in.excl_14_bits,
                    tp_in.n_valid_pairs,
                    tp_in.ref_positions,
                    n_act=n_act,
                    n_act_blocks=n_act_blocks,
                    max_pairs=max_pairs,
                )

            tp_k = jax.lax.cond(forces_dirty_b, _refresh_params, lambda _: tp_in, None)
            return tp_k, G_in

        tp0, G0 = jax.lax.cond(nl_dirty_b, _entry_rebuild_nl, _entry_keep_nl, None)

        def _entry_recompute_f(_: Any):
            p_a0 = pos_wr0.astype(jnp.float32)[None, :, :]
            if pad_tail > 0:
                p_a0 = jnp.pad(p_a0, ((0, 0), (0, pad_tail), (0, 0)))
            pm0 = jnp.take_along_axis(p_a0, tp0.perm[:, :, None], axis=1)
            _, fx_n, fy_n, fz_n = _eval_morton_forces(
                pm0[:, :, 0].reshape(n_act_pad),
                pm0[:, :, 1].reshape(n_act_pad),
                pm0[:, :, 2].reshape(n_act_pad),
                tp0,
                box_b0,
                G0,
                pos_unw0,
                pos_whole0,
                compute_energy=False,
            )
            return fx_n, fy_n, fz_n

        fx0, fy0, fz0 = jax.lax.cond(
            nl_dirty_b | forces_dirty_b,
            _entry_recompute_f,
            lambda _: (fx_in, fy_in, fz_in),
            None,
        )

        px0 = pos_pad0[:, :, 0].reshape(N)
        py0 = pos_pad0[:, :, 1].reshape(N)
        pz0 = pos_pad0[:, :, 2].reshape(N)
        mx0 = mom_pad0[:, :, 0].reshape(N)
        my0 = mom_pad0[:, :, 1].reshape(N)
        mz0 = mom_pad0[:, :, 2].reshape(N)

        def _chunk_cond(oc: Any) -> Any:
            return oc[0] < n_steps_i32

        def _chunk_body(oc: Any) -> Any:
            (
                step_i,
                px,
                py,
                pz,
                pos_unw,
                mx,
                my,
                mz,
                fx_m,
                fy_m,
                fz_m,
                tp_c,
                box_c,
                G_c,
                mc_sc,
                mc_at,
                mc_ac,
            ) = oc
            pos_cur_act = jnp.stack(
                [px[:n_act], py[:n_act], pz[:n_act]], axis=-1
            ).astype(jnp.float32)
            dr = _mic_dr(
                pos_cur_act - tp_c.ref_positions[0, :n_act, :].astype(jnp.float32),
                box_c[0].astype(jnp.float32),
            )[None, :, :]
            dr_sq = jnp.where(am_b[:, :n_act], jnp.sum(dr * dr, axis=-1), 0.0)
            need_reb = (step_i > 0) & jnp.any(jnp.max(dr_sq, axis=-1) > thresh_sq)

            def _do_rebuild(ops: Any) -> Any:
                px_r, py_r, pz_r, fx_r, fy_r, fz_r, tp_old, bx_r = ops
                pos_full = jnp.stack(
                    [px_r.reshape(1, N), py_r.reshape(1, N), pz_r.reshape(1, N)],
                    axis=-1,
                )
                tp_new = _build_tp_gpu(pos_full, q_b_in, sig_b_in, eps_b_in, bx_r)
                remap = jnp.take_along_axis(tp_old.inv_perm, tp_new.perm, axis=1)
                fx_n = jnp.take_along_axis(
                    fx_r.reshape(1, n_act_pad), remap, axis=1
                ).reshape(n_act_pad)
                fy_n = jnp.take_along_axis(
                    fy_r.reshape(1, n_act_pad), remap, axis=1
                ).reshape(n_act_pad)
                fz_n = jnp.take_along_axis(
                    fz_r.reshape(1, n_act_pad), remap, axis=1
                ).reshape(n_act_pad)
                return fx_n, fy_n, fz_n, tp_new

            fx_m, fy_m, fz_m, tp_c = jax.lax.cond(
                need_reb,
                _do_rebuild,
                lambda ops: (ops[3], ops[4], ops[5], ops[6]),
                (px, py, pz, fx_m, fy_m, fz_m, tp_c, box_c),
            )

            g_step = step_offset_i32 + step_i
            k_cur = jnp.minimum(jnp.int32(chunk_k), n_steps_i32 - step_i)
            if has_barostat:
                rem_b = jnp.int32(baro_freq) - (g_step % jnp.int32(baro_freq))
                k_cur = jnp.minimum(k_cur, rem_b)
            if cmm_freq > 1:
                rem_c = jnp.int32(cmm_freq) - (g_step % jnp.int32(cmm_freq))
                k_cur = jnp.minimum(k_cur, rem_c)

            bx_d0 = box_c[0]
            inv_n = _inv_for_n(tp_c.inv_perm)
            pos_wr_prev0 = jnp.stack([px[:n_act], py[:n_act], pz[:n_act]], axis=-1)
            seeds_0 = jax.random.fold_in(key_in, g_step)[None, :]
            px, py, pz, mxh, myh, mzh, px_m, py_m, pz_m = _pallas_baoab_pre_force_batch(
                1,
                in_b_pre,
                dummy_pos_b,
                dummy_pos_b,
                dummy_pos_b,
                mass_b,
                seeds_0,
                wi_b,
                wm_b,
                box_c,
                dt_b,
                kT_b,
                gamma_akma,
                r_oh_akma,
                r_hh_akma,
                m_o_akma,
                m_h_akma,
                True,
                True,
                interpret,
                atom_mask=am_real_b,
                constraint_pairs=cp_b,
                constraint_lengths=cl_b,
                constraint_mask=cm_b,
                n_active_atoms=n_act,
                n_active_waters=n_waters_real,
                f64_accumulators=f64_acc,
                fuse_prev_post=False,
                morton_inv_perm=inv_n,
                morton_pad=n_act_pad,
                prebuilt_soa=(px, py, pz, mx, my, mz, fx_m, fy_m, fz_m, soa_plan),
                is_triclinic=bool(is_triclinic and pre_has_tri_kw),
            )
            pos_wr_next0 = jnp.stack([px[:n_act], py[:n_act], pz[:n_act]], axis=-1)
            dr_step0 = _mic_dr(pos_wr_next0 - pos_wr_prev0, bx_d0)
            if has_vs:
                dr_step0 = jnp.where(am_real_act[:, None], dr_step0, 0.0)
            pos_unw = pos_unw + dr_step0
            if has_vs:
                px, py, pz, pos_unw, pos_whole0_step = _sync_vs_from_wrapped_and_unw(
                    px, py, pz, pos_unw, bx_d0
                )
                px_m, py_m, pz_m = _permute_wrapped_to_morton(px, py, pz, tp_c.perm)
            else:
                pos_whole0_step = pos_unw
            _, fx_m, fy_m, fz_m = _eval_morton_forces(
                px_m,
                py_m,
                pz_m,
                tp_c,
                box_c,
                G_c,
                pos_unw,
                pos_whole0_step,
                False,
            )

            def _inner_step(s_rel: Any, ic: Any) -> Any:
                (
                    px_s,
                    py_s,
                    pz_s,
                    pos_unw_s,
                    mxh_s,
                    myh_s,
                    mzh_s,
                    fx_s,
                    fy_s,
                    fz_s,
                ) = ic
                pos_wr_prev_s = jnp.stack(
                    [px_s[:n_act], py_s[:n_act], pz_s[:n_act]], axis=-1
                )
                seeds_s = jax.random.fold_in(key_in, g_step + s_rel)[None, :]
                (
                    px_s,
                    py_s,
                    pz_s,
                    mxh_s,
                    myh_s,
                    mzh_s,
                    px_ms,
                    py_ms,
                    pz_ms,
                ) = _pallas_baoab_pre_force_batch(
                    1,
                    in_b_pre,
                    dummy_pos_b,
                    dummy_pos_b,
                    dummy_pos_b,
                    mass_b,
                    seeds_s,
                    wi_b,
                    wm_b,
                    box_c,
                    dt_b,
                    kT_b,
                    gamma_akma,
                    r_oh_akma,
                    r_hh_akma,
                    m_o_akma,
                    m_h_akma,
                    True,
                    True,
                    interpret,
                    atom_mask=am_real_b,
                    constraint_pairs=cp_b,
                    constraint_lengths=cl_b,
                    constraint_mask=cm_b,
                    n_active_atoms=n_act,
                    n_active_waters=n_waters_real,
                    f64_accumulators=f64_acc,
                    fuse_prev_post=True,
                    morton_inv_perm=inv_n,
                    morton_pad=n_act_pad,
                    prebuilt_soa=(
                        px_s,
                        py_s,
                        pz_s,
                        mxh_s,
                        myh_s,
                        mzh_s,
                        fx_s,
                        fy_s,
                        fz_s,
                        soa_plan,
                    ),
                    is_triclinic=bool(is_triclinic and pre_has_tri_kw),
                )
                pos_wr_next_s = jnp.stack(
                    [px_s[:n_act], py_s[:n_act], pz_s[:n_act]], axis=-1
                )
                dr_s = _mic_dr(pos_wr_next_s - pos_wr_prev_s, bx_d0)
                if has_vs:
                    dr_s = jnp.where(am_real_act[:, None], dr_s, 0.0)
                pos_unw_s = pos_unw_s + dr_s
                if has_vs:
                    px_s, py_s, pz_s, pos_unw_s, pos_whole_s = (
                        _sync_vs_from_wrapped_and_unw(
                            px_s, py_s, pz_s, pos_unw_s, bx_d0
                        )
                    )
                    px_ms, py_ms, pz_ms = _permute_wrapped_to_morton(
                        px_s, py_s, pz_s, tp_c.perm
                    )
                else:
                    pos_whole_s = pos_unw_s
                _, fx_s, fy_s, fz_s = _eval_morton_forces(
                    px_ms,
                    py_ms,
                    pz_ms,
                    tp_c,
                    box_c,
                    G_c,
                    pos_unw_s,
                    pos_whole_s,
                    False,
                )
                return (
                    px_s,
                    py_s,
                    pz_s,
                    pos_unw_s,
                    mxh_s,
                    myh_s,
                    mzh_s,
                    fx_s,
                    fy_s,
                    fz_s,
                )

            px, py, pz, pos_unw, mxh, myh, mzh, fx_m, fy_m, fz_m = jax.lax.fori_loop(
                jnp.int32(1),
                k_cur,
                _inner_step,
                (px, py, pz, pos_unw, mxh, myh, mzh, fx_m, fy_m, fz_m),
            )
            mx, my, mz = _pallas_baoab_post_soa(
                px,
                py,
                pz,
                mxh,
                myh,
                mzh,
                fx_m,
                fy_m,
                fz_m,
                inv_n,
                n_act_pad,
                dt_b,
                m_o_akma,
                m_h_akma,
                interpret,
                soa_plan,
                f64_accumulators=f64_acc,
                box=box_c if post_has_box_kw else None,
                is_triclinic=bool(is_triclinic and post_has_box_kw),
            )

            step_next = step_i + k_cur
            g_step_next = step_offset_i32 + step_next

            if cmm_freq > 0 and n_act > 1:
                do_cmm = (g_step_next // jnp.int32(max(1, cmm_freq))) > (
                    g_step // jnp.int32(max(1, cmm_freq))
                )

                def _rm_cmm(m_xyz: Any) -> Any:
                    mx_i = m_xyz[0].reshape(1, N)
                    my_i = m_xyz[1].reshape(1, N)
                    mz_i = m_xyz[2].reshape(1, N)
                    m_real = jnp.where(
                        am_real_b, mass_b, jnp.asarray(0.0, dtype=state_dtype)
                    )
                    m_tot = jnp.maximum(
                        jnp.sum(m_real, axis=1, keepdims=True),
                        jnp.asarray(1e-12, dtype=state_dtype),
                    )
                    vx_cm = (
                        jnp.sum(jnp.where(am_real_b, mx_i, 0.0), axis=1, keepdims=True)
                        / m_tot
                    )
                    vy_cm = (
                        jnp.sum(jnp.where(am_real_b, my_i, 0.0), axis=1, keepdims=True)
                        / m_tot
                    )
                    vz_cm = (
                        jnp.sum(jnp.where(am_real_b, mz_i, 0.0), axis=1, keepdims=True)
                        / m_tot
                    )
                    return (
                        jnp.where(am_real_b, mx_i - m_real * vx_cm, 0.0).reshape(N),
                        jnp.where(am_real_b, my_i - m_real * vy_cm, 0.0).reshape(N),
                        jnp.where(am_real_b, mz_i - m_real * vz_cm, 0.0).reshape(N),
                    )

                mx, my, mz = jax.lax.cond(
                    do_cmm, _rm_cmm, lambda m_xyz: m_xyz, (mx, my, mz)
                )

            if has_barostat:
                do_baro = (g_step_next % jnp.int32(baro_freq)) == 0

                def _baro_step(b_ops: Any) -> Any:
                    (
                        px_b,
                        py_b,
                        pz_b,
                        punw_b,
                        mx_b,
                        my_b,
                        mz_b,
                        fx_b,
                        fy_b,
                        fz_b,
                        tp_b,
                        bx_b,
                        G_b,
                        sc_b,
                        at_b,
                        ac_b,
                    ) = b_ops
                    if (
                        baro_type == "MonteCarloAnisotropicBarostat"
                        and len(active_aniso_axes) == 0
                    ):
                        return b_ops
                    bx_vec = bx_b[0]
                    pos_whole = _unwrap_whole_and_reconstruct_vs(punw_b, bx_vec)
                    mol_sums = (
                        jnp.zeros((n_mols, 3), dtype=state_dtype)
                        .at[mol_jnp]
                        .add(pos_whole)
                    )
                    com_u = (mol_sums * inv_mol_size_jnp[:, None].astype(state_dtype))[
                        mol_jnp
                    ]
                    b_key = jax.random.fold_in(
                        key_in, g_step_next + jnp.int32(999983)
                    )

                    if baro_mode == "scr":
                        eps_fd = jnp.asarray(1e-3, dtype=state_dtype)
                        v_cur = _vol_f32(bx_vec)

                        def _energy_at_scale(mu_fd: jax.Array) -> jax.Array:
                            bx_fd = bx_b * mu_fd
                            p_fd_unw = pos_whole + (mu_fd - 1.0) * com_u
                            if has_vs and virtual_sites is not None:
                                p_fd_unw = virtual_sites.reconstruct_positions(p_fd_unw)
                            p_fd_wr = _wrap_pos(p_fd_unw, bx_fd[0]).astype(
                                jnp.float32
                            )[None, :, :]
                            if pad_tail > 0:
                                p_fd_wr = jnp.pad(
                                    p_fd_wr, ((0, 0), (0, pad_tail), (0, 0))
                                )
                            pm_fd = jnp.take_along_axis(
                                p_fd_wr, tp_b.perm[:, :, None], axis=1
                            )
                            G_fd = _greens_for_box(bx_fd[0]) if do_pme else G_b
                            e_fd, _, _, _ = _eval_morton_forces(
                                pm_fd[:, :, 0].reshape(n_act_pad),
                                pm_fd[:, :, 1].reshape(n_act_pad),
                                pm_fd[:, :, 2].reshape(n_act_pad),
                                tp_b,
                                bx_fd,
                                G_fd,
                                p_fd_unw,
                                p_fd_unw,
                                True,
                            )
                            return e_fd[0]

                        e_hi = _energy_at_scale(
                            jnp.asarray(1.0, dtype=state_dtype) + eps_fd
                        )
                        e_lo = _energy_at_scale(
                            jnp.asarray(1.0, dtype=state_dtype) - eps_fd
                        )
                        vir_core = -(e_hi - e_lo) / (2.0 * eps_fd.astype(jnp.float32))
                        vir_lr = (
                            3.0
                            * (c_tail_in + c_plasma_b[0])
                            / jnp.maximum(v_cur, 1e-12)
                        )
                        vir_tot = vir_core + vir_lr

                        ke_tot = 0.5 * jnp.sum(
                            (
                                mx_b[:n_act] ** 2
                                + my_b[:n_act] ** 2
                                + mz_b[:n_act] ** 2
                            )
                            * inv_mass_act
                        ).astype(jnp.float32)
                        inst_p = (2.0 * ke_tot + vir_tot) / (
                            3.0 * jnp.maximum(v_cur, 1e-12)
                        )
                        dt_baro = (dt_b[0] * float(baro_freq)).astype(jnp.float32)
                        tau_scr = jnp.float32(20.0)
                        comp_akma = jnp.float32(4.5e-5 * BAR_PER_AKMA_PRESSURE)
                        noise_scr = jax.random.normal(b_key, (), dtype=jnp.float32)
                        deps_det = (
                            (dt_baro / tau_scr)
                            * comp_akma
                            * (inst_p - jnp.float32(target_p_akma))
                        )
                        deps_sto = (
                            jnp.sqrt(
                                jnp.maximum(
                                    2.0
                                    * kT_b[0].astype(jnp.float32)
                                    * comp_akma
                                    * dt_baro
                                    / (tau_scr * jnp.maximum(v_cur, 1e-12)),
                                    0.0,
                                )
                            )
                            * noise_scr
                        )
                        mu = jnp.clip(
                            jnp.exp((deps_det + deps_sto) / 3.0), 0.99, 1.01
                        ).astype(state_dtype)
                        bx_new = bx_b * mu
                        punw_new = pos_whole + (mu - 1.0) * com_u
                        if has_vs and virtual_sites is not None:
                            punw_new = virtual_sites.reconstruct_positions(punw_new)
                        pwr_new = _wrap_pos(punw_new, bx_new[0])
                        pos_pad_new = (
                            jnp.pad(pwr_new, ((0, pad_N), (0, 0)))[None, :, :]
                            if pad_N > 0
                            else pwr_new[None, :, :]
                        )
                        G_new = _greens_for_box(bx_new[0]) if do_pme else G_b
                        tp_new = _build_tp_gpu(
                            pos_pad_new, q_b_in, sig_b_in, eps_b_in, bx_new
                        )
                        p_ap = pwr_new.astype(jnp.float32)[None, :, :]
                        if pad_tail > 0:
                            p_ap = jnp.pad(p_ap, ((0, 0), (0, pad_tail), (0, 0)))
                        pm_new = jnp.take_along_axis(
                            p_ap, tp_new.perm[:, :, None], axis=1
                        )
                        _, fx_n, fy_n, fz_n = _eval_morton_forces(
                            pm_new[:, :, 0].reshape(n_act_pad),
                            pm_new[:, :, 1].reshape(n_act_pad),
                            pm_new[:, :, 2].reshape(n_act_pad),
                            tp_new,
                            bx_new,
                            G_new,
                            punw_new,
                            punw_new,
                            False,
                        )
                        inv_mu = 1.0 / mu
                        return (
                            pos_pad_new[:, :, 0].reshape(N),
                            pos_pad_new[:, :, 1].reshape(N),
                            pos_pad_new[:, :, 2].reshape(N),
                            punw_new,
                            mx_b * inv_mu,
                            my_b * inv_mu,
                            mz_b * inv_mu,
                            fx_n,
                            fy_n,
                            fz_n,
                            tp_new,
                            bx_new,
                            G_new,
                            sc_b,
                            at_b,
                            ac_b,
                        )

                    # GPU Monte Carlo Barostats: Isotropic, Anisotropic, and Membrane
                    p_cur_wr = _wrap_pos(pos_whole, bx_vec).astype(jnp.float32)[
                        None, :, :
                    ]
                    if pad_tail > 0:
                        p_cur_wr = jnp.pad(p_cur_wr, ((0, 0), (0, pad_tail), (0, 0)))
                    pm_cur = jnp.take_along_axis(
                        p_cur_wr, tp_b.perm[:, :, None], axis=1
                    )
                    e_old, _, _, _ = _eval_morton_forces(
                        pm_cur[:, :, 0].reshape(n_act_pad),
                        pm_cur[:, :, 1].reshape(n_act_pad),
                        pm_cur[:, :, 2].reshape(n_act_pad),
                        tp_b,
                        bx_b,
                        G_b,
                        pos_whole,
                        pos_whole,
                        True,
                    )
                    u_tri = jax.random.uniform(b_key, (3,), dtype=jnp.float32)
                    u_scale = 2.0 * u_tri[0] - 1.0

                    if baro_type == "MonteCarloAnisotropicBarostat":
                        act_ax_jnp = jnp.asarray(active_aniso_axes, dtype=jnp.int32)
                        n_act_ax = len(active_aniso_axes)
                        pick_i = jnp.minimum(
                            (u_tri[2] * float(n_act_ax)).astype(jnp.int32),
                            jnp.int32(n_act_ax - 1),
                        )
                        mode_slot = act_ax_jnp[pick_i]
                        sc_cur = sc_b[mode_slot]
                        vol_scale = jnp.maximum(1.0 + sc_cur * u_scale, 1e-6)
                        axis_scales = (
                            jnp.ones((3,), dtype=state_dtype)
                            .at[mode_slot]
                            .set(vol_scale.astype(state_dtype))
                        )
                        p_xyz_jnp = jnp.asarray(eff_p_xyz, dtype=jnp.float32)
                        p_eff = p_xyz_jnp[mode_slot]
                        use_st = False
                    elif baro_type == "MonteCarloMembraneBarostat":
                        p_eff = jnp.float32(target_p_akma)
                        use_st = True
                        z_free = baro_z_mode == 0
                        z_const_vol = baro_z_mode == 2
                        if baro_xy_isotropic:
                            if z_free:
                                is_z_move = u_tri[2] >= 0.5
                                mode_slot = jnp.where(
                                    is_z_move, jnp.int32(2), jnp.int32(0)
                                )
                                sc_cur = sc_b[mode_slot]
                                raw_sc = jnp.maximum(1.0 + sc_cur * u_scale, 1e-6)
                                s_xy = jnp.where(
                                    is_z_move,
                                    jnp.float32(1.0),
                                    jnp.sqrt(raw_sc),
                                )
                                s_z = jnp.where(is_z_move, raw_sc, jnp.float32(1.0))
                            else:
                                mode_slot = jnp.int32(0)
                                sc_cur = sc_b[0]
                                raw_sc = jnp.maximum(1.0 + sc_cur * u_scale, 1e-6)
                                s_xy = jnp.sqrt(raw_sc)
                                s_z = (1.0 / raw_sc) if z_const_vol else jnp.float32(1.0)
                            axis_scales = jnp.stack(
                                [s_xy, s_xy, s_z]
                            ).astype(state_dtype)
                        else:
                            n_m_axes = 3 if z_free else 2
                            mode_slot = jnp.minimum(
                                (u_tri[2] * float(n_m_axes)).astype(jnp.int32),
                                jnp.int32(n_m_axes - 1),
                            )
                            sc_cur = sc_b[mode_slot]
                            raw_sc = jnp.maximum(1.0 + sc_cur * u_scale, 1e-6)
                            s_vec = jnp.ones((3,), dtype=jnp.float32).at[mode_slot].set(
                                raw_sc
                            )
                            if z_const_vol:
                                s_vec = s_vec.at[2].set(1.0 / (s_vec[0] * s_vec[1]))
                            axis_scales = s_vec.astype(state_dtype)
                    else:
                        mode_slot = jnp.int32(0)
                        sc_cur = sc_b[0]
                        vol_scale = jnp.maximum(1.0 + sc_cur * u_scale, 1e-6)
                        mu = (vol_scale ** (1.0 / 3.0)).astype(state_dtype)
                        axis_scales = jnp.stack([mu, mu, mu])
                        p_eff = jnp.float32(target_p_akma)
                        use_st = False

                    bx_prop = _scale_box_axes(bx_b, axis_scales)
                    punw_prop = pos_whole + (axis_scales[None, :] - 1.0) * com_u
                    if has_vs and virtual_sites is not None:
                        punw_prop = virtual_sites.reconstruct_positions(punw_prop)
                    pwr_prop = _wrap_pos(punw_prop, bx_prop[0])
                    pos_pad_prop = (
                        jnp.pad(pwr_prop, ((0, pad_N), (0, 0)))[None, :, :]
                        if pad_N > 0
                        else pwr_prop[None, :, :]
                    )
                    G_prop = _greens_for_box(bx_prop[0]) if do_pme else G_b
                    tp_prop = _build_tp_gpu(
                        pos_pad_prop, q_b_in, sig_b_in, eps_b_in, bx_prop
                    )
                    p_ap = pwr_prop.astype(jnp.float32)[None, :, :]
                    if pad_tail > 0:
                        p_ap = jnp.pad(p_ap, ((0, 0), (0, pad_tail), (0, 0)))
                    pm_prop = jnp.take_along_axis(
                        p_ap, tp_prop.perm[:, :, None], axis=1
                    )
                    e_new, fx_p, fy_p, fz_p = _eval_morton_forces(
                        pm_prop[:, :, 0].reshape(n_act_pad),
                        pm_prop[:, :, 1].reshape(n_act_pad),
                        pm_prop[:, :, 2].reshape(n_act_pad),
                        tp_prop,
                        bx_prop,
                        G_prop,
                        punw_prop,
                        punw_prop,
                        True,
                    )
                    v_old = _vol_f32(bx_b[0])
                    v_new = _vol_f32(bx_prop[0])
                    vol_ratio = jnp.maximum(
                        v_new / jnp.maximum(v_old, 1e-12), 1e-12
                    )
                    st_work = (
                        -jnp.float32(baro_surface_tension_akma)
                        * (_area_xy_f32(bx_prop[0]) - _area_xy_f32(bx_b[0]))
                        if use_st
                        else jnp.float32(0.0)
                    )
                    kT_f = kT_b[0].astype(jnp.float32)
                    du_lr = (c_tail_in + c_plasma_b[0]) * (
                        1.0 / jnp.maximum(v_new, 1e-12)
                        - 1.0 / jnp.maximum(v_old, 1e-12)
                    )
                    dh = (
                        (e_new[0] - e_old[0])
                        + du_lr
                        + p_eff * (v_new - v_old)
                        + st_work
                        - jnp.float32(n_mols) * kT_f * jnp.log(vol_ratio)
                    )
                    acc = u_tri[1] < jnp.exp(
                        jnp.clip(-dh / jnp.maximum(kT_f, 1e-12), -50.0, 50.0)
                    )
                    px_sel = jnp.where(acc, pos_pad_prop[:, :, 0].reshape(N), px_b)
                    py_sel = jnp.where(acc, pos_pad_prop[:, :, 1].reshape(N), py_b)
                    pz_sel = jnp.where(acc, pos_pad_prop[:, :, 2].reshape(N), pz_b)
                    punw_sel = jnp.where(acc, punw_prop, punw_b)
                    bx_sel = jnp.where(acc, bx_prop, bx_b)
                    G_sel = jnp.where(acc, G_prop, G_b)
                    tp_sel = jax.tree.map(
                        lambda nv, ov: jnp.where(acc, nv, ov), tp_prop, tp_b
                    )
                    fx_sel = jnp.where(acc, fx_p, fx_b)
                    fy_sel = jnp.where(acc, fy_p, fy_b)
                    fz_sel = jnp.where(acc, fz_p, fz_b)

                    at_s = at_b[mode_slot] + jnp.int32(1)
                    ac_s = ac_b[mode_slot] + acc.astype(jnp.int32)
                    do_adj = at_s >= jnp.int32(10)
                    sc_s = sc_b[mode_slot]
                    sc_shrunk = jnp.maximum(sc_s * jnp.float32(0.5), jnp.float32(1e-6))
                    sc_grown = jnp.minimum(sc_s * jnp.float32(1.1), jnp.float32(0.3))
                    sc_cand = jnp.where(
                        ac_s.astype(jnp.float32)
                        < jnp.float32(0.25) * at_s.astype(jnp.float32),
                        sc_shrunk,
                        jnp.where(
                            ac_s.astype(jnp.float32)
                            > jnp.float32(0.75) * at_s.astype(jnp.float32),
                            sc_grown,
                            sc_s,
                        ),
                    )
                    sc_out = sc_b.at[mode_slot].set(jnp.where(do_adj, sc_cand, sc_s))
                    at_out = at_b.at[mode_slot].set(
                        jnp.where(do_adj, jnp.int32(0), at_s)
                    )
                    ac_out = ac_b.at[mode_slot].set(
                        jnp.where(do_adj, jnp.int32(0), ac_s)
                    )
                    return (
                        px_sel,
                        py_sel,
                        pz_sel,
                        punw_sel,
                        mx_b,
                        my_b,
                        mz_b,
                        fx_sel,
                        fy_sel,
                        fz_sel,
                        tp_sel,
                        bx_sel,
                        G_sel,
                        sc_out,
                        at_out,
                        ac_out,
                    )

                (
                    px,
                    py,
                    pz,
                    pos_unw,
                    mx,
                    my,
                    mz,
                    fx_m,
                    fy_m,
                    fz_m,
                    tp_c,
                    box_c,
                    G_c,
                    mc_sc,
                    mc_at,
                    mc_ac,
                ) = jax.lax.cond(
                    do_baro,
                    _baro_step,
                    lambda b_ops: b_ops,
                    (
                        px,
                        py,
                        pz,
                        pos_unw,
                        mx,
                        my,
                        mz,
                        fx_m,
                        fy_m,
                        fz_m,
                        tp_c,
                        box_c,
                        G_c,
                        mc_sc,
                        mc_at,
                        mc_ac,
                    ),
                )

            return (
                step_next,
                px,
                py,
                pz,
                pos_unw,
                mx,
                my,
                mz,
                fx_m,
                fy_m,
                fz_m,
                tp_c,
                box_c,
                G_c,
                mc_sc,
                mc_at,
                mc_ac,
            )

        (
            _,
            _,
            _,
            _,
            pos_unw_f,
            mxf,
            myf,
            mzf,
            fxf,
            fyf,
            fzf,
            tpf,
            boxf,
            Gf,
            mc_sc_f,
            mc_at_f,
            mc_ac_f,
        ) = jax.lax.while_loop(
            _chunk_cond,
            _chunk_body,
            (
                jnp.int32(0),
                px0,
                py0,
                pz0,
                pos_unw0,
                mx0,
                my0,
                mz0,
                fx0,
                fy0,
                fz0,
                tp0,
                box_b0,
                G0,
                mc_sc0,
                mc_at0,
                mc_ac0,
            ),
        )
        mom_act_f = jnp.stack([mxf[:n_act], myf[:n_act], mzf[:n_act]], axis=-1)
        vel_out = mom_act_f * inv_mass_act[:, None]
        key_out = jax.random.fold_in(key_in, step_offset_i32 + n_steps_i32)
        return (
            pos_unw_f,
            vel_out,
            boxf[0],
            key_out,
            tpf,
            fxf,
            fyf,
            fzf,
            Gf,
            mc_sc_f,
            mc_at_f,
            mc_ac_f,
        )

    @jax.jit
    def _jit_unpermute_forces(
        fx_m: jax.Array,
        fy_m: jax.Array,
        fz_m: jax.Array,
        inv_perm: jax.Array,
    ) -> jax.Array:
        inv_act = inv_perm[0, :n_act]
        f_akma = jnp.stack([fx_m[inv_act], fy_m[inv_act], fz_m[inv_act]], axis=-1)
        return f_akma * float(FORCE_AKMA_TO_OMM)

    @jax.jit
    def _jit_energy_and_forces_impl(  # noqa: PLR0913
        pos_in: jax.Array,
        vel_in: jax.Array,
        box_diag_in: jax.Array,
        q_b_in: jax.Array,
        sig_b_in: jax.Array,
        eps_b_in: jax.Array,
        b_r0_in: jax.Array,
        b_k_in: jax.Array,
        a_th0_in: jax.Array,
        a_kval_in: jax.Array,
        d_p0_in: jax.Array,
        d_p1_in: jax.Array,
        d_p2_in: jax.Array,
        e_sig_in: jax.Array,
        e_eps_in: jax.Array,
        e_qp_in: jax.Array,
        tp_in: MortonTilePairState,
        G_in: jax.Array,
        nl_dirty_in: bool | jax.Array,
        forces_dirty_in: bool | jax.Array,
        c_tail_in: jax.Array,
        pr_idx_in: jax.Array,
        pr_kx_in: jax.Array,
        pr_ky_in: jax.Array,
        pr_kz_in: jax.Array,
        pr_x0_in: jax.Array,
        pr_y0_in: jax.Array,
        pr_z0_in: jax.Array,
        pr_valid_in: jax.Array,
        custom_forces_in: tuple[CompiledLeptonForce, ...] = (),
        gp_vec_in: jax.Array | None = None,
    ):
        box_b0 = (
            box_diag_in.astype(state_dtype)[None, :, :]
            if is_triclinic
            else box_diag_in.astype(state_dtype)[None, :]
        )
        eff_gp_vec = empty_gp_vec if gp_vec_in is None else gp_vec_in
        bonded_dyn = bonded_static._replace(
            b_r0=b_r0_in,
            b_k=b_k_in,
            a_th0=a_th0_in,
            a_kval=a_kval_in,
            d_p0=d_p0_in,
            d_p1=d_p1_in,
            d_p2=d_p2_in,
            e_sig=e_sig_in,
            e_eps=e_eps_in,
            e_qp=e_qp_in,
        )
        e_self_b = (
            spme_self_energy(q_b_in, am_b, alpha=pme_alpha).astype(jnp.float32)
            if do_pme
            else jnp.zeros((1,), dtype=jnp.float32)
        )
        e_lj_self_b = (
            lj_pme_self_energy(
                dispersion_charges(sig_b_in, eps_b_in, am_b),
                atom_mask=am_b,
                alpha_lj=lj_pme_alpha,
            ).astype(jnp.float32)
            if is_lj_pme
            else jnp.zeros((1,), dtype=jnp.float32)
        )
        q_net = jnp.sum(jnp.where(am_b, q_b_in, 0.0), axis=-1)
        c_plasma_b = (
            -(jnp.pi * float(COULOMB_CONSTANT) / (2.0 * (pme_alpha**2))) * (q_net * q_net)
            if do_pme
            else jnp.zeros((1,), dtype=jnp.float32)
        )
        _eval_morton_forces = _make_eval_morton_forces(
            bonded_dyn,
            e_self_b,
            e_lj_self_b,
            (
                pr_idx_in,
                pr_kx_in,
                pr_ky_in,
                pr_kz_in,
                pr_x0_in,
                pr_y0_in,
                pr_z0_in,
                pr_valid_in,
            ),
            custom_forces_in,
            eff_gp_vec,
        )

        pos_unw0 = pos_in.astype(state_dtype)
        if has_vs and virtual_sites is not None:
            pos_whole0 = _unwrap_whole_and_reconstruct_vs(pos_unw0, box_b0[0])
            pos_unw0 = pos_unw0.at[vs_order].set(pos_whole0[vs_order])
            pos_wr0 = _wrap_pos(pos_whole0, box_b0[0])
        else:
            pos_whole0 = pos_unw0
            pos_wr0 = _wrap_pos(pos_unw0, box_b0[0])

        pos_pad0 = (
            jnp.pad(pos_wr0, ((0, pad_N), (0, 0)))[None, :, :]
            if pad_N > 0
            else pos_wr0[None, :, :]
        )
        nl_dirty_b = jnp.asarray(nl_dirty_in, dtype=jnp.bool_)
        forces_dirty_b = jnp.asarray(forces_dirty_in, dtype=jnp.bool_)

        def _obs_rebuild_nl(_: Any):
            tp_r = _build_tp_gpu(pos_pad0, q_b_in, sig_b_in, eps_b_in, box_b0)
            G_r = _greens_for_box(box_b0[0]) if do_pme else G_in
            return tp_r, G_r

        def _obs_keep_nl(_: Any):
            def _refresh_params(_u: Any) -> MortonTilePairState:
                qp = _pad_atom_1d(q_b_in, 0.0)
                sp = _pad_atom_1d(sig_b_in, 1.0)
                ep = _pad_atom_1d(eps_b_in, 0.0)
                return MortonTilePairState(
                    tp_in.perm,
                    tp_in.inv_perm,
                    jnp.take_along_axis(qp, tp_in.perm, axis=1),
                    jnp.take_along_axis(sp, tp_in.perm, axis=1),
                    jnp.take_along_axis(ep, tp_in.perm, axis=1),
                    tp_in.mask_m,
                    tp_in.pair_ti,
                    tp_in.pair_tj,
                    tp_in.excl_full_bits,
                    tp_in.excl_14_bits,
                    tp_in.n_valid_pairs,
                    tp_in.ref_positions,
                    n_act=n_act,
                    n_act_blocks=n_act_blocks,
                    max_pairs=max_pairs,
                )

            tp_k = jax.lax.cond(forces_dirty_b, _refresh_params, lambda _: tp_in, None)
            return tp_k, G_in

        tp0, G0 = jax.lax.cond(nl_dirty_b, _obs_rebuild_nl, _obs_keep_nl, None)
        p_a0 = pos_wr0.astype(jnp.float32)[None, :, :]
        if pad_tail > 0:
            p_a0 = jnp.pad(p_a0, ((0, 0), (0, pad_tail), (0, 0)))
        pm0 = jnp.take_along_axis(p_a0, tp0.perm[:, :, None], axis=1)
        e_core, fx_tot, fy_tot, fz_tot = _eval_morton_forces(
            pm0[:, :, 0].reshape(n_act_pad),
            pm0[:, :, 1].reshape(n_act_pad),
            pm0[:, :, 2].reshape(n_act_pad),
            tp0,
            box_b0,
            G0,
            pos_unw0,
            pos_whole0,
            compute_energy=True,
        )
        v_cur = _vol_f32(box_b0[0])
        pe_akma = e_core[0] + (c_tail_in + c_plasma_b[0]) / jnp.maximum(v_cur, 1e-12)
        ke_akma = 0.5 * jnp.sum(
            mass_act.astype(jnp.float64)[:, None] * (vel_in.astype(jnp.float64) ** 2)
        )
        scalars_kj = jnp.stack(
            [
                pe_akma.astype(jnp.float64) * float(KCAL_PER_MOL_TO_KJ_PER_MOL),
                ke_akma * float(KCAL_PER_MOL_TO_KJ_PER_MOL),
            ]
        )
        inv_act = tp0.inv_perm[0, :n_act]
        f_omm = (
            jnp.stack([fx_tot[inv_act], fy_tot[inv_act], fz_tot[inv_act]], axis=-1)
            * float(FORCE_AKMA_TO_OMM)
        )
        return scalars_kj, f_omm, tp0, fx_tot, fy_tot, fz_tot, G0

    empty_pr_bufs = _pack_position_restraints((), n_act)

    def _jit_energy_and_forces(  # noqa: PLR0913
        pos_in: jax.Array,
        vel_in: jax.Array,
        box_diag_in: jax.Array,
        q_b_in: jax.Array,
        sig_b_in: jax.Array,
        eps_b_in: jax.Array,
        b_r0_in: jax.Array,
        b_k_in: jax.Array,
        a_th0_in: jax.Array,
        a_kval_in: jax.Array,
        d_p0_in: jax.Array,
        d_p1_in: jax.Array,
        d_p2_in: jax.Array,
        e_sig_in: jax.Array,
        e_eps_in: jax.Array,
        e_qp_in: jax.Array,
        tp_in: MortonTilePairState,
        G_in: jax.Array,
        nl_dirty_in: bool | jax.Array,
        forces_dirty_in: bool | jax.Array,
        c_tail_in: jax.Array,
        pr_bufs: tuple[jax.Array, ...] | None = None,
        custom_forces_in: tuple[CompiledLeptonForce, ...] = (),
        gp_vec_in: jax.Array | None = None,
    ):
        eff_pr = (
            pr_bufs
            if pr_bufs is not None
            else (
                pr_params_getter()
                if (has_restraints and callable(pr_params_getter))
                else empty_pr_bufs
            )
        )
        return _jit_energy_and_forces_impl(
            pos_in,
            vel_in,
            box_diag_in,
            q_b_in,
            sig_b_in,
            eps_b_in,
            b_r0_in,
            b_k_in,
            a_th0_in,
            a_kval_in,
            d_p0_in,
            d_p1_in,
            d_p2_in,
            e_sig_in,
            e_eps_in,
            e_qp_in,
            tp_in,
            G_in,
            nl_dirty_in,
            forces_dirty_in,
            c_tail_in,
            *eff_pr,
            custom_forces_in,
            gp_vec_in,
        )

    return {
        "jit_step": _jit_step,
        "jit_unpermute_forces": _jit_unpermute_forces,
        "jit_energy_and_forces": _jit_energy_and_forces,
        "bonded_static": bonded_static,
        "tp_init": tp_init,
        "fx_init": z_f,
        "fy_init": z_f,
        "fz_init": z_f,
        "G_init": G_init,
        "N": N,
        "n_act": n_act,
        "state_dtype": state_dtype,
        "is_triclinic": is_triclinic,
        "param_names": param_names,
    }


def _build_context_gbsa_runner(  # noqa: PLR0913
    bundle: MolecularBundle,
    *,
    n_act: int,
    precision: str,
    pallas_interpret: bool,
    has_nonbonded: bool,
    has_finite_cutoff: bool,
    cutoff: float,
    lj_switch_width: float,
    gb_use_neighbor_list: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    gbsa_surface_tension: float,
    gamma_akma: float,
    cmm_freq: int,
    r_oh_akma: float,
    r_hh_akma: float,
    m_o_akma: float,
    m_h_akma: float,
    has_restraints: bool = False,
    virtual_sites: VirtualSites | None = None,
    has_custom_forces: bool = False,
    param_names: tuple[str, ...] = (),
    init_positions: jnp.ndarray,
) -> dict[str, Any]:
    """Build a persistent B=1 GPU-resident BAOAB + 2-pass OBC2 GBSA + bonded + VS/PR runner."""
    B = 1
    N = int(bundle.positions.shape[0])
    pad_N = N - n_act

    if precision == "mixed":
        state_dtype: Any = jnp.float64
        f64_acc = True
    else:
        state_dtype = jnp.float32
        f64_acc = False

    interpret = (
        bool(pallas_interpret)
        or not _triton_gpu_available()
        or jax.default_backend() == "cpu"
    )
    n_waters_real = (
        int(np.sum(np.asarray(bundle.water_mask)))
        if bundle.shape_spec.has_real_water
        else 0
    )
    mass_b = bundle.masses[:N].astype(state_dtype)[None, :]
    mass_act = bundle.masses[:n_act].astype(state_dtype)
    inv_mass_act = jnp.where(
        mass_act > 1e-12, 1.0 / jnp.where(mass_act > 1e-12, mass_act, 1.0), 0.0
    )
    am_b = bundle.atom_mask[:N].astype(jnp.bool_)[None, :]
    am_real_b = am_b & (mass_b > 1e-12)

    has_vs = virtual_sites is not None and virtual_sites.n_total > 0
    vs_order = (
        virtual_sites.order
        if (has_vs and virtual_sites is not None)
        else jnp.zeros((0,), dtype=jnp.int32)
    )

    if bundle.shape_spec.has_real_water and bundle.water_indices is not None:
        wi_b = bundle.water_indices.astype(jnp.int32)[None, :, :]
        wm_b = bundle.water_mask.astype(jnp.int32)[None, :]
    else:
        wi_b = jnp.zeros((1, 0, 3), dtype=jnp.int32)
        wm_b = jnp.zeros((1, 0), dtype=jnp.int32)

    has_cons = bool(
        getattr(bundle.shape_spec, "has_real_constraints", False)
        and bundle.constraint_pairs is not None
        and int(bundle.constraint_pairs.shape[0]) > 0
    )
    cp_b = (
        bundle.constraint_pairs.astype(jnp.int32)[None, :, :]
        if (has_cons and bundle.constraint_pairs is not None)
        else None
    )
    cl_b = (
        bundle.constraint_lengths.astype(state_dtype)[None, :]
        if (has_cons and bundle.constraint_lengths is not None)
        else None
    )
    cm_b = (
        bundle.constraint_mask.astype(jnp.int32)[None, :]
        if (has_cons and bundle.constraint_mask is not None)
        else None
    )

    soa_plan = build_baoab_soa_static_plan(
        B,
        N,
        state_dtype,
        mass_b,
        wi_b,
        wm_b,
        atom_mask_b=am_real_b,
        constraint_pairs_b=cp_b,
        constraint_lengths_b=cl_b,
        constraint_mask_b=cm_b,
        n_active_atoms=n_act,
        n_active_waters=n_waters_real,
    )

    excl_idx_gb = (
        np.asarray(bundle.excl_dense_indices[:n_act], dtype=np.int32)
        if (has_nonbonded and bundle.excl_dense_indices is not None)
        else None
    )
    excl_vdw_gb = (
        np.asarray(bundle.excl_dense_scales_vdw[:n_act], dtype=np.float32)
        if (has_nonbonded and bundle.excl_dense_scales_vdw is not None)
        else None
    )
    excl_elec_gb = (
        np.asarray(bundle.excl_dense_scales_elec[:n_act], dtype=np.float32)
        if (has_nonbonded and bundle.excl_dense_scales_elec is not None)
        else None
    )
    gbsa_tl = build_gbsa_tile_pair_list(
        np.asarray(init_positions[:n_act], dtype=np.float32),
        atom_mask=np.ones((n_act,), dtype=np.int32),
        excl_indices=excl_idx_gb,
        excl_scales_vdw=excl_vdw_gb,
        excl_scales_elec=excl_elec_gb,
        cutoff=None,
        born_cutoff=None,
        use_morton=False,
    )

    has_b = bundle.n_bonds is not None and int(bundle.n_bonds) > 0
    has_ub = bundle.n_urey_bradley is not None and int(bundle.n_urey_bradley) > 0
    has_a = bundle.n_angles is not None and int(bundle.n_angles) > 0
    has_d = bundle.n_dihedrals is not None and int(bundle.n_dihedrals) > 0
    has_i = bundle.n_impropers is not None and int(bundle.n_impropers) > 0
    has_e = (
        has_nonbonded
        and bundle.n_exception_pairs is not None
        and int(bundle.n_exception_pairs) > 0
    )
    has_c = bundle.n_cmap is not None and int(bundle.n_cmap) > 0
    has_any_bonded = has_b or has_ub or has_a or has_d or has_i or has_e or has_c
    imp_harm = bool(
        has_i
        and bundle.improper_params is not None
        and (
            bundle.improper_params.shape[-1] == 2
            or (
                hasattr(bundle, "improper_is_periodic")
                and bundle.improper_is_periodic is not None
                and not bool(np.asarray(bundle.improper_is_periodic))
            )
        )
    )
    ub_idx = (
        (
            bundle.urey_bradley_idx[:, ::2]
            if bundle.urey_bradley_idx.shape[-1] == 3
            else bundle.urey_bradley_idx[:, :2]
        )
        if (has_ub and bundle.urey_bradley_idx is not None)
        else None
    )

    nb_cut = float(cutoff) if (has_nonbonded and has_finite_cutoff) else None
    gb_cut = float(cutoff) if (gb_use_neighbor_list and has_finite_cutoff) else None
    sw_dist = (
        float(cutoff) - float(lj_switch_width)
        if (
            nb_cut is not None
            and float(lj_switch_width) > 0.0
            and float(cutoff) > float(lj_switch_width)
        )
        else None
    )
    in_b_pre = (True,) * (12 + (3 if has_cons else 0))
    dummy_pos_b = jnp.zeros((1, N, 3), dtype=state_dtype)
    dummy_box_b = jnp.ones((1, 3), dtype=state_dtype)
    identity_inv_perm = jnp.arange(N, dtype=jnp.int32)[None, :]
    z_f = jnp.zeros((N,), dtype=jnp.float32)

    @jax.jit
    def _jit_step(  # noqa: PLR0913
        pos_in: jax.Array,
        vel_in: jax.Array,
        key_in: jax.Array,
        step_offset_in: int | jax.Array,
        n_steps_in: int | jax.Array,
        dt_s_in: float | jax.Array,
        kT_s_in: float | jax.Array,
        q_act_in: jax.Array,
        sig_act_in: jax.Array,
        eps_act_in: jax.Array,
        rad_act_in: jax.Array,
        sr_act_in: jax.Array,
        bond_params_in: jax.Array | None,
        ub_params_in: jax.Array | None,
        angle_params_in: jax.Array | None,
        dihedral_params_in: jax.Array | None,
        improper_params_in: jax.Array | None,
        e_sig_in: jax.Array | None,
        e_eps_in: jax.Array | None,
        e_qp_in: jax.Array | None,
        cmap_coeffs_in: jax.Array | None,
        fx_in: jax.Array,
        fy_in: jax.Array,
        fz_in: jax.Array,
        forces_dirty_in: bool | jax.Array,
        pr_idx_in: jax.Array,
        pr_kx_in: jax.Array,
        pr_ky_in: jax.Array,
        pr_kz_in: jax.Array,
        pr_x0_in: jax.Array,
        pr_y0_in: jax.Array,
        pr_z0_in: jax.Array,
        pr_valid_in: jax.Array,
        custom_forces_in: tuple[CompiledLeptonForce, ...] = (),
        gp_vec_in: jax.Array | None = None,
    ):
        step_offset_i32 = jnp.asarray(step_offset_in, dtype=jnp.int32)
        n_steps_i32 = jnp.asarray(n_steps_in, dtype=jnp.int32)
        forces_dirty_b = jnp.asarray(forces_dirty_in, dtype=jnp.bool_)
        dt_b = jnp.broadcast_to(jnp.asarray(dt_s_in, dtype=state_dtype), (1,))
        kT_b = jnp.broadcast_to(jnp.asarray(kT_s_in, dtype=state_dtype), (1,))

        def _eval_gbsa_forces(
            px_cur: jax.Array, py_cur: jax.Array, pz_cur: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            pos_act = jnp.stack(
                [px_cur[:n_act], py_cur[:n_act], pz_cur[:n_act]], axis=-1
            )
            if has_vs and virtual_sites is not None:
                pos_act = virtual_sites.reconstruct_positions(pos_act)
                px_cur = px_cur.at[vs_order].set(pos_act[vs_order, 0])
                py_cur = py_cur.at[vs_order].set(pos_act[vs_order, 1])
                pz_cur = pz_cur.at[vs_order].set(pos_act[vs_order, 2])
            p_f32 = pos_act.astype(jnp.float32)

            if has_any_bonded:
                _, f_b = cast(
                    "tuple[jax.Array, jax.Array]",
                    pallas_bonded_all_fused(
                        p_f32,
                        bonds=bundle.bond_idx if has_b else None,
                        bond_params=bond_params_in if has_b else None,
                        bond_mask=bundle.bond_mask if has_b else None,
                        urey_bradley_bonds=ub_idx,
                        urey_bradley_params=ub_params_in if has_ub else None,
                        urey_bradley_mask=bundle.urey_bradley_mask if has_ub else None,
                        angles=bundle.angle_idx if has_a else None,
                        angle_params=angle_params_in if has_a else None,
                        angle_mask=bundle.angle_mask if has_a else None,
                        dihedrals=bundle.dihedral_idx if has_d else None,
                        dihedral_params=dihedral_params_in if has_d else None,
                        dihedral_mask=bundle.dihedral_mask if has_d else None,
                        impropers=bundle.improper_idx if has_i else None,
                        improper_params=improper_params_in if has_i else None,
                        improper_mask=bundle.improper_mask if has_i else None,
                        improper_is_harmonic=imp_harm,
                        exception_pairs=bundle.exception_pairs if has_e else None,
                        exception_sigmas=e_sig_in if has_e else None,
                        exception_epsilons=e_eps_in if has_e else None,
                        exception_chargeprods=e_qp_in if has_e else None,
                        exception_mask=bundle.exception_mask if has_e else None,
                        cmap_torsions=bundle.cmap_torsion_idx if has_c else None,
                        cmap_indices=bundle.cmap_indices if has_c else None,
                        cmap_mask=bundle.cmap_mask if has_c else None,
                        cmap_coeffs=cmap_coeffs_in if has_c else None,
                        box=None,
                        dist_eps=1e-10,
                        b0_is_ji=True,
                        cmap_first_is_psi=False,
                        coulomb_constant=COULOMB_CONSTANT,
                        return_forces=True,
                        interpret=interpret,
                    ),
                )
            else:
                f_b = jnp.zeros_like(p_f32)

            _, f_gb = pallas_gbsa_tile_energy_and_forces(
                p_f32,
                q_act_in,
                rad_act_in,
                sr_act_in,
                sig_act_in if has_nonbonded else None,
                eps_act_in if has_nonbonded else None,
                gbsa_tl,
                solvent_dielectric=solvent_dielectric,
                solute_dielectric=solute_dielectric,
                coulomb_dielectric=1.0,
                surface_tension=gbsa_surface_tension,
                include_gb=True,
                include_ace=bool(gbsa_surface_tension > 0.0),
                include_coulomb=has_nonbonded,
                include_lj=has_nonbonded,
                cutoff=nb_cut,
                born_cutoff=gb_cut,
                switch_dist=sw_dist,
                coulomb_constant=COULOMB_CONSTANT,
                interpret=interpret,
            )
            f_tot = f_b + f_gb

            if has_restraints:
                dx_pr = (pos_act[pr_idx_in, 0] - pr_x0_in.astype(state_dtype)).astype(
                    jnp.float32
                )
                dy_pr = (pos_act[pr_idx_in, 1] - pr_y0_in.astype(state_dtype)).astype(
                    jnp.float32
                )
                dz_pr = (pos_act[pr_idx_in, 2] - pr_z0_in.astype(state_dtype)).astype(
                    jnp.float32
                )
                f_tot = (
                    f_tot.at[pr_idx_in, 0]
                    .add(jnp.where(pr_valid_in, -2.0 * pr_kx_in * dx_pr, 0.0))
                    .at[pr_idx_in, 1]
                    .add(jnp.where(pr_valid_in, -2.0 * pr_ky_in * dy_pr, 0.0))
                    .at[pr_idx_in, 2]
                    .add(jnp.where(pr_valid_in, -2.0 * pr_kz_in * dz_pr, 0.0))
                )

            if has_custom_forces and len(custom_forces_in) > 0:
                gp_dict_cf: dict[str, jax.Array] = (
                    {
                        k_nm: gp_vec_in[idx_nm]
                        for idx_nm, k_nm in enumerate(param_names)
                    }
                    if (gp_vec_in is not None and len(param_names) > 0)
                    else {}
                )

                def _cf_gb_total_e(p_cf: jax.Array) -> jax.Array:
                    e_acc_cf = jnp.asarray(0.0, dtype=jnp.float32)
                    for cf_item in custom_forces_in:
                        e_acc_cf = e_acc_cf + cf_item.energy_akma(
                            p_cf, None, gp_dict_cf
                        )
                    return e_acc_cf

                f_tot = f_tot + (-jax.grad(_cf_gb_total_e)(p_f32))

            if has_vs and virtual_sites is not None:
                f_tot = virtual_sites.redistribute_forces(f_tot, p_f32)

            if pad_N > 0:
                f_pad = jnp.pad(f_tot, ((0, pad_N), (0, 0)))
            else:
                f_pad = f_tot
            return px_cur, py_cur, pz_cur, f_pad[:, 0], f_pad[:, 1], f_pad[:, 2]

        pos_act0 = pos_in.astype(state_dtype)
        if has_vs and virtual_sites is not None:
            pos_act0 = virtual_sites.reconstruct_positions(pos_act0)
            vel_act0 = vel_in.astype(state_dtype).at[vs_order].set(0.0)
        else:
            vel_act0 = vel_in.astype(state_dtype)
        mom_act0 = vel_act0 * mass_act[:, None]
        if pad_N > 0:
            pos_pad0 = jnp.pad(pos_act0, ((0, pad_N), (0, 0)))
            mom_pad0 = jnp.pad(mom_act0, ((0, pad_N), (0, 0)))
        else:
            pos_pad0 = pos_act0
            mom_pad0 = mom_act0

        px0, py0, pz0 = pos_pad0[:, 0], pos_pad0[:, 1], pos_pad0[:, 2]
        mx0, my0, mz0 = mom_pad0[:, 0], mom_pad0[:, 1], mom_pad0[:, 2]

        px0, py0, pz0, fx0, fy0, fz0 = jax.lax.cond(
            forces_dirty_b,
            lambda p_xyz: _eval_gbsa_forces(p_xyz[0], p_xyz[1], p_xyz[2]),
            lambda p_xyz: (p_xyz[0], p_xyz[1], p_xyz[2], fx_in, fy_in, fz_in),
            (px0, py0, pz0),
        )

        def _chunk_cond(oc: Any) -> Any:
            return oc[0] < n_steps_i32

        def _chunk_body(oc: Any) -> Any:
            step_i, px, py, pz, mx, my, mz, fx_c, fy_c, fz_c = oc
            g_step = step_offset_i32 + step_i
            k_cur = n_steps_i32 - step_i
            if cmm_freq == 1:
                k_cur = jnp.minimum(k_cur, jnp.int32(1))
            elif cmm_freq > 1:
                rem_c = jnp.int32(cmm_freq) - (g_step % jnp.int32(cmm_freq))
                k_cur = jnp.minimum(k_cur, rem_c)

            seeds_0 = jax.random.fold_in(key_in, g_step)[None, :]
            px, py, pz, mxh, myh, mzh = _pallas_baoab_pre_force_batch(
                1,
                in_b_pre,
                dummy_pos_b,
                dummy_pos_b,
                dummy_pos_b,
                mass_b,
                seeds_0,
                wi_b,
                wm_b,
                dummy_box_b,
                dt_b,
                kT_b,
                gamma_akma,
                r_oh_akma,
                r_hh_akma,
                m_o_akma,
                m_h_akma,
                False,
                True,
                interpret,
                atom_mask=am_real_b,
                constraint_pairs=cp_b,
                constraint_lengths=cl_b,
                constraint_mask=cm_b,
                n_active_atoms=n_act,
                n_active_waters=n_waters_real,
                f64_accumulators=f64_acc,
                fuse_prev_post=False,
                morton_inv_perm=None,
                morton_pad=0,
                prebuilt_soa=(px, py, pz, mx, my, mz, fx_c, fy_c, fz_c, soa_plan),
            )
            px, py, pz, fx_c, fy_c, fz_c = _eval_gbsa_forces(px, py, pz)

            def _inner_step(s_rel: Any, ic: Any) -> Any:
                px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, fx_s, fy_s, fz_s = ic
                seeds_s = jax.random.fold_in(key_in, g_step + s_rel)[None, :]
                px_s, py_s, pz_s, mxh_s, myh_s, mzh_s = _pallas_baoab_pre_force_batch(
                    1,
                    in_b_pre,
                    dummy_pos_b,
                    dummy_pos_b,
                    dummy_pos_b,
                    mass_b,
                    seeds_s,
                    wi_b,
                    wm_b,
                    dummy_box_b,
                    dt_b,
                    kT_b,
                    gamma_akma,
                    r_oh_akma,
                    r_hh_akma,
                    m_o_akma,
                    m_h_akma,
                    False,
                    True,
                    interpret,
                    atom_mask=am_real_b,
                    constraint_pairs=cp_b,
                    constraint_lengths=cl_b,
                    constraint_mask=cm_b,
                    n_active_atoms=n_act,
                    n_active_waters=n_waters_real,
                    f64_accumulators=f64_acc,
                    fuse_prev_post=True,
                    morton_inv_perm=None,
                    morton_pad=0,
                    prebuilt_soa=(
                        px_s,
                        py_s,
                        pz_s,
                        mxh_s,
                        myh_s,
                        mzh_s,
                        fx_s,
                        fy_s,
                        fz_s,
                        soa_plan,
                    ),
                )
                px_s, py_s, pz_s, fx_s, fy_s, fz_s = _eval_gbsa_forces(px_s, py_s, pz_s)
                return (px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, fx_s, fy_s, fz_s)

            px, py, pz, mxh, myh, mzh, fx_c, fy_c, fz_c = jax.lax.fori_loop(
                jnp.int32(1),
                k_cur,
                _inner_step,
                (px, py, pz, mxh, myh, mzh, fx_c, fy_c, fz_c),
            )
            mx, my, mz = _pallas_baoab_post_soa(
                px,
                py,
                pz,
                mxh,
                myh,
                mzh,
                fx_c,
                fy_c,
                fz_c,
                identity_inv_perm,
                N,
                dt_b,
                m_o_akma,
                m_h_akma,
                interpret,
                soa_plan,
                f64_accumulators=f64_acc,
            )

            step_next = step_i + k_cur
            g_step_next = step_offset_i32 + step_next
            if cmm_freq > 0 and n_act > 1:
                do_cmm = (g_step_next // jnp.int32(max(1, cmm_freq))) > (
                    g_step // jnp.int32(max(1, cmm_freq))
                )

                def _rm_cmm(m_xyz: Any) -> Any:
                    mx_i = m_xyz[0].reshape(1, N)
                    my_i = m_xyz[1].reshape(1, N)
                    mz_i = m_xyz[2].reshape(1, N)
                    m_real = jnp.where(
                        am_real_b, mass_b, jnp.asarray(0.0, dtype=state_dtype)
                    )
                    m_tot = jnp.maximum(
                        jnp.sum(m_real, axis=1, keepdims=True),
                        jnp.asarray(1e-12, dtype=state_dtype),
                    )
                    vx_cm = (
                        jnp.sum(jnp.where(am_real_b, mx_i, 0.0), axis=1, keepdims=True)
                        / m_tot
                    )
                    vy_cm = (
                        jnp.sum(jnp.where(am_real_b, my_i, 0.0), axis=1, keepdims=True)
                        / m_tot
                    )
                    vz_cm = (
                        jnp.sum(jnp.where(am_real_b, mz_i, 0.0), axis=1, keepdims=True)
                        / m_tot
                    )
                    return (
                        jnp.where(am_real_b, mx_i - m_real * vx_cm, 0.0).reshape(N),
                        jnp.where(am_real_b, my_i - m_real * vy_cm, 0.0).reshape(N),
                        jnp.where(am_real_b, mz_i - m_real * vz_cm, 0.0).reshape(N),
                    )

                mx, my, mz = jax.lax.cond(
                    do_cmm, _rm_cmm, lambda m_xyz: m_xyz, (mx, my, mz)
                )

            return (step_next, px, py, pz, mx, my, mz, fx_c, fy_c, fz_c)

        _, pxf, pyf, pzf, mxf, myf, mzf, fxf, fyf, fzf = jax.lax.while_loop(
            _chunk_cond,
            _chunk_body,
            (jnp.int32(0), px0, py0, pz0, mx0, my0, mz0, fx0, fy0, fz0),
        )
        pos_out = jnp.stack([pxf[:n_act], pyf[:n_act], pzf[:n_act]], axis=-1)
        if has_vs and virtual_sites is not None:
            pos_out = virtual_sites.reconstruct_positions(pos_out)
        mom_act_f = jnp.stack([mxf[:n_act], myf[:n_act], mzf[:n_act]], axis=-1)
        vel_out = mom_act_f * inv_mass_act[:, None]
        key_out = jax.random.fold_in(key_in, step_offset_i32 + n_steps_i32)
        return pos_out, vel_out, key_out, fxf, fyf, fzf

    return {
        "jit_step": _jit_step,
        "fx_init": z_f,
        "fy_init": z_f,
        "fz_init": z_f,
        "gbsa_tl": gbsa_tl,
        "imp_harm": imp_harm,
        "N": N,
        "n_act": n_act,
        "state_dtype": state_dtype,
        "param_names": param_names,
    }


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
        eff_conv_cfg = (
            dataclasses.replace(self._config, anisotropic_barostat=True)
            if (
                not self._config.anisotropic_barostat
                and _system_has_anisotropic_or_membrane_barostat(system)
            )
            else self._config
        )
        self._converted_system: ConvertedOpenMMSystem = from_openmm_system(
            system, integrator, eff_conv_cfg
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
        self._anisotropic_barostat = bool(
            self._config.anisotropic_barostat
            or _system_has_anisotropic_or_membrane_barostat(system)
        )
        self._execution_mode = str(self._config.execution_mode)
        self._integrator_scheme = str(self._config.integrator_scheme)
        self._thermostat_override = (
            str(self._config.thermostat_override)
            if self._config.thermostat_override is not None
            else None
        )
        self._lj_combining_rule = (
            str(self._config.lj_combining_rule)
            if str(self._config.lj_combining_rule) != "lorentz_berthelot"
            else str(self._converted_system.lj_combining_rule)
        )
        self._exceptions_use_pbc = bool(
            self._converted_system.exceptions_use_pbc or self._config.exceptions_use_pbc
        )
        self._dispersion_follows_parameters = bool(
            self._converted_system.dispersion_follows_parameters
            and self._config.dispersion_follows_parameters
        )
        self._use_lj_pme = bool(
            self._config.use_lj_pme
            or str(self._converted_system.nonbonded_method).lower() in ("lj_pme", "ljpme")
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
            and (not self._is_triclinic or _triclinic_morton_kernels_supported())
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
        self._host_nl_dirty: bool = False

        # Adaptive Monte Carlo barostat step-size tracking (OpenMM MonteCarloBarostatImpl)
        self._mc_volume_scale: float = 0.01
        self._mc_num_attempted: int = 0
        self._mc_num_accepted: int = 0

        # Persistent GPU-resident Morton BAOAB + SPME/RF + NPT fast-path state
        self._morton_runner: dict[str, Any] | None = None
        self._morton_runner_key: tuple[Any, ...] | None = None
        self._morton_tp_state: MortonTilePairState | None = None
        self._morton_fx: jax.Array | None = None
        self._morton_fy: jax.Array | None = None
        self._morton_fz: jax.Array | None = None
        self._morton_G: jax.Array | None = None
        self._morton_bonded_dyn: FusedBondedBuffers | None = None
        self._cached_morton_params: tuple[jax.Array, ...] | None = None
        self._cached_morton_restraints: tuple[jax.Array, ...] | None = None
        self._morton_params_dirty: bool = True
        self._morton_restraints_dirty: bool = True
        self._morton_nl_dirty: bool = True
        self._morton_forces_dirty: bool = True
        self._morton_mc_scale: jax.Array = jnp.full((3,), 0.01, dtype=jnp.float32)
        self._morton_mc_att: jax.Array = jnp.zeros((3,), dtype=jnp.int32)
        self._morton_mc_acc: jax.Array = jnp.zeros((3,), dtype=jnp.int32)
        self._morton_box_diag: jax.Array | None = None

        # Persistent GPU-resident Implicit-Solvent OBC2 GBSA BAOAB fast-path state
        self._gbsa_runner: dict[str, Any] | None = None
        self._gbsa_runner_key: tuple[Any, ...] | None = None
        self._gbsa_fx: jax.Array | None = None
        self._gbsa_fy: jax.Array | None = None
        self._gbsa_fz: jax.Array | None = None
        self._cached_gbsa_params: tuple[Any, ...] | None = None
        self._gbsa_params_dirty: bool = True
        self._gbsa_forces_dirty: bool = True
        self._gbsa_last_pos_id: int | None = None

        self._cached_box_nm: np.ndarray = np.eye(3, dtype=np.float64)
        self._box_nm_dirty: bool = False
        self._bundle_box_dirty: bool = False

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
        self._refresh_box_caches_from_vectors()
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

        # Molecular topology graph & flat-CSR arrays for O(N) wrap/unwrap
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

    def _effective_nonbonded_method(self) -> str:
        """Resolve effective nonbonded method string, honoring ``config.use_lj_pme``."""
        raw = str(self._converted_system.nonbonded_method)
        if self._use_lj_pme and raw.lower() in ("pme", "spme", "lj_pme", "ljpme"):
            return "lj_pme"
        return raw

    def _effective_pme_alpha(self) -> float | None:
        """Resolve effective Ewald alpha (Å⁻¹), honoring explicit NonbondedForce.setPMEParameters."""
        if self._effective_nonbonded_method().lower() not in ("pme", "spme", "lj_pme", "ljpme"):
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

    def _effective_lj_pme_alpha(self) -> float | None:
        """Resolve effective dispersion Ewald alpha (Å⁻¹) for ``NonbondedForce.LJPME``."""
        if self._effective_nonbonded_method().lower() not in ("lj_pme", "ljpme"):
            return None
        if (
            self._converted_system.explicit_lj_pme_alpha is not None
            and float(self._converted_system.explicit_lj_pme_alpha) > 0.0
        ):
            return float(self._converted_system.explicit_lj_pme_alpha)
        ewald_tol = (
            self._converted_system.ewald_error_tolerance
            if self._converted_system.ewald_error_tolerance is not None
            else self._config.ewald_error_tolerance
        )
        cutoff = float(self._converted_system.nonbonded_cutoff)
        if 0.0 < cutoff < float("inf"):
            return float((1.0 / cutoff) * math.sqrt(-math.log(2.0 * float(ewald_tol))))
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

    def _is_reduced_triclinic_supported(self) -> bool:
        """Return True when periodic box is orthorhombic or reduced-form triclinic with kernel support."""
        if not self._is_triclinic:
            return True
        nb_m = self._effective_nonbonded_method().lower()
        do_pme = nb_m in ("pme", "spme", "lj_pme", "ljpme")
        if not _triclinic_morton_kernels_supported(do_pme):
            return False
        if self._morton_box_diag is not None and self._morton_box_diag.ndim == 2:
            bv_np = np.asarray(self._morton_box_diag, dtype=np.float64)
        elif self._box_vectors is not None:
            bv_np = np.asarray(self._box_vectors, dtype=np.float64)
        else:
            return False
        return _is_reduced_triclinic_box_np(bv_np)

    def _can_use_tile_pairs(self) -> bool:
        return (
            self._converted_system.has_nonbonded
            and self._has_finite_cutoff
            and (not self._is_triclinic or self._is_reduced_triclinic_supported())
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
        if self._is_triclinic or self._neighbor_idx is None or not self._can_use_tile_pairs():
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
            npt_mult = 1.25 * (1.15**3) if self._converted_system.barostat is not None else 1.25
            padded_cap = max(64, ((math.ceil(raw_cap * npt_mult) + 63) // 64) * 64)
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
        self._host_nl_dirty = False

        if not self._should_use_neighbor_list() or (
            self._is_triclinic and self._converted_system.has_pbc
        ):
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

    def _refresh_box_caches_from_vectors(self) -> None:
        """Cache host `box_nm` and GPU `box_diag` (`(3,)` ortho or `(3, 3)` triclinic) from `self._box_vectors`."""
        if self._box_vectors is not None:
            bv_np = np.asarray(self._box_vectors, dtype=np.float64)
            self._cached_box_nm = bv_np * ANGSTROM_TO_NM
            if self._is_triclinic:
                self._morton_box_diag = (
                    self._box_vectors
                    if self._box_vectors.ndim == 2
                    else jnp.diag(self._box_vectors)
                )
            else:
                self._morton_box_diag = (
                    jnp.diag(self._box_vectors)
                    if self._box_vectors.ndim == 2
                    else self._box_vectors
                )
        else:
            self._cached_box_nm = np.eye(3, dtype=np.float64)
            self._morton_box_diag = None
        self._box_nm_dirty = False
        self._bundle_box_dirty = False

    def _sync_bundle_box(self) -> None:
        """Lazily materialize NPT GPU `box_diag` into `self._box_vectors` and `self._bundle.box`."""
        if self._bundle_box_dirty and self._morton_box_diag is not None:
            new_bv = (
                self._morton_box_diag.astype(self._dtype)
                if self._morton_box_diag.ndim == 2
                else jnp.diag(self._morton_box_diag.astype(self._dtype))
            )
            self._box_vectors = new_bv
            self._bundle = eqx.tree_at(
                lambda b: b.box, self._bundle, new_bv.astype(jnp.float32)
            )
            self._bundle_box_dirty = False

    def _get_box_nm_host(self) -> np.ndarray:
        """Return cached `(3, 3)` `box_vectors_nm` on host, syncing GPU box only if NPT modified it."""
        if self._box_nm_dirty and self._morton_box_diag is not None:
            bv_arr = (
                np.asarray(self._morton_box_diag, dtype=np.float64) * ANGSTROM_TO_NM
            )
            self._cached_box_nm = bv_arr if bv_arr.ndim == 2 else np.diag(bv_arr)
            self._box_nm_dirty = False
        return self._cached_box_nm

    def _ensure_host_nl_fresh(self) -> None:
        """Lazily refresh host NeighborList / TilePairNeighborList only when host fallback/virial needs it."""
        self._sync_bundle_box()
        if self._host_nl_dirty and (self._should_use_neighbor_list() or self._can_use_gbsa_tiles()):
            self._init_neighbor_list(self._positions[: self._n_particles])

    def create_ensemble_plan(self) -> EnsemblePlan:
        """Create an optimized Pallas v3.5 EnsemblePlan for this system."""
        self._sync_bundle_box()
        plan_cls = _ensemble_plan_mod.EnsemblePlan
        bnd = self._bundle
        if self._converted_system.nonbonded_offsets is not None:
            gp = self._get_dynamic_global_params()
            n_full = int(bnd.charges.shape[0])
            q_eff, sig_eff, eps_eff = _apply_particle_offsets(
                bnd.charges.astype(jnp.float32),
                bnd.sigmas.astype(jnp.float32),
                bnd.epsilons.astype(jnp.float32),
                self._converted_system.nonbonded_offsets,
                gp,
                self._n_particles,
            )
            if n_full > self._n_particles:
                q_eff = jnp.concatenate([q_eff, bnd.charges[self._n_particles :].astype(jnp.float32)])
                sig_eff = jnp.concatenate([sig_eff, bnd.sigmas[self._n_particles :].astype(jnp.float32)])
                eps_eff = jnp.concatenate([eps_eff, bnd.epsilons[self._n_particles :].astype(jnp.float32)])
            bnd = eqx.tree_at(
                lambda b: (b.charges, b.sigmas, b.epsilons),
                bnd,
                (q_eff, sig_eff, eps_eff),
            )
            if (
                bnd.exception_chargeprods is not None
                and bnd.exception_sigmas is not None
                and bnd.exception_epsilons is not None
            ):
                eqp_eff, esig_eff, eeps_eff = _apply_exception_offsets(
                    bnd.exception_chargeprods.astype(jnp.float32),
                    bnd.exception_sigmas.astype(jnp.float32),
                    bnd.exception_epsilons.astype(jnp.float32),
                    self._converted_system.nonbonded_offsets,
                    gp,
                )
                bnd = eqx.tree_at(
                    lambda b: (b.exception_chargeprods, b.exception_sigmas, b.exception_epsilons),
                    bnd,
                    (eqp_eff, esig_eff, eeps_eff),
                )
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
        return plan_cls.from_bundle(bnd, **candidate_kwargs)

    # --- Core Getters ---
    def getSystem(self) -> openmm.System:  # noqa: N802
        return self._system

    def getIntegrator(self) -> openmm.Integrator:  # noqa: N802
        return self._integrator

    def getPlatform(self) -> openmm.Platform:  # noqa: N802
        return cast("openmm.Platform", self._platform)

    def getMolecules(self) -> tuple[tuple[int, ...], ...]:  # noqa: N802
        return self._molecules

    def _refresh_morton_bonded_dyn(self) -> FusedBondedBuffers:
        """Pack current bundle bonded arrays into flat SoA buffers matching `_morton_runner` shapes."""
        bd = _pack_bundle_fused_bonded(self._bundle)
        self._morton_bonded_dyn = bd
        return bd

    def _ensure_morton_params(
        self,
        runner: dict[str, Any],
        bd: FusedBondedBuffers,
        use_disp: bool,
    ) -> tuple[jax.Array, ...]:
        """Return cached GPU parameter buffers for Morton runner, refreshing only on param mutation."""
        if not self._morton_params_dirty and self._cached_morton_params is not None:
            return self._cached_morton_params

        N_full = int(runner["N"])
        n_act = int(runner["n_act"])
        gp = self._get_dynamic_global_params()
        nb_off = self._converted_system.nonbonded_offsets

        q_base = self._bundle.charges[:n_act].astype(jnp.float32)
        sig_base = self._bundle.sigmas[:n_act].astype(jnp.float32)
        eps_base = self._bundle.epsilons[:n_act].astype(jnp.float32)
        q_act, sig_act, eps_act = _apply_particle_offsets(
            q_base, sig_base, eps_base, nb_off, gp, n_act
        )
        if N_full > n_act:
            pad_n = N_full - n_act
            q_b = jnp.pad(q_act, ((0, pad_n),), constant_values=0.0)[None, :]
            sig_b = jnp.pad(sig_act, ((0, pad_n),), constant_values=1.0)[None, :]
            eps_b = jnp.pad(eps_act, ((0, pad_n),), constant_values=0.0)[None, :]
        else:
            q_b = q_act[None, :]
            sig_b = sig_act[None, :]
            eps_b = eps_act[None, :]

        if (
            bd.E_real > 0
            and nb_off is not None
            and int(nb_off.exception_bundle_indices.shape[0]) > 0
            and self._bundle.exception_chargeprods is not None
            and self._bundle.exception_sigmas is not None
            and self._bundle.exception_epsilons is not None
        ):
            eqp_a, esig_a, eeps_a = _apply_exception_offsets(
                self._bundle.exception_chargeprods.astype(jnp.float32),
                self._bundle.exception_sigmas.astype(jnp.float32),
                self._bundle.exception_epsilons.astype(jnp.float32),
                nb_off,
                gp,
            )
            e_qp_in = eqp_a.reshape(-1)
            e_sig_in = esig_a.reshape(-1)
            e_eps_in = eeps_a.reshape(-1)
        else:
            e_qp_in = bd.e_qp
            e_sig_in = bd.e_sig
            e_eps_in = bd.e_eps

        eff_nb_l = self._effective_nonbonded_method().lower()
        eff_use_disp = bool(use_disp and eff_nb_l not in ("lj_pme", "ljpme"))
        if eff_use_disp:
            if self._dispersion_follows_parameters or nb_off is None:
                sig_t, eps_t = sig_act, eps_act
            else:
                _, sig_t, eps_t = _apply_particle_offsets(
                    q_base,
                    sig_base,
                    eps_base,
                    nb_off,
                    self._converted_system.default_parameters,
                    n_act,
                )
            c_tail_s = _compute_lj_tail_energy_compat(
                jnp.ones((3,), dtype=jnp.float32),
                sig_t,
                eps_t,
                float(self._converted_system.nonbonded_cutoff),
                jnp.ones((n_act,), dtype=bool),
                lj_switch_width=float(self._converted_system.lj_switch_width),
                combining_rule=self._lj_combining_rule,
            ).astype(jnp.float32)
        else:
            c_tail_s = jnp.asarray(0.0, dtype=jnp.float32)

        cached = (q_b, sig_b, eps_b, e_sig_in, e_eps_in, e_qp_in, c_tail_s)
        self._cached_morton_params = cached
        self._morton_params_dirty = False
        return cached

    def _ensure_restraint_params(self) -> tuple[jax.Array, ...]:
        """Return cached flattened GPU PositionRestraints arrays for Morton/GBSA runners."""
        if (
            not self._morton_restraints_dirty
            and self._cached_morton_restraints is not None
        ):
            return self._cached_morton_restraints
        packed = _pack_position_restraints(self._position_restraints, self._n_particles)
        self._cached_morton_restraints = packed
        self._morton_restraints_dirty = False
        return packed

    def _ensure_gbsa_params(self, runner: dict[str, Any]) -> tuple[Any, ...]:
        """Return cached GPU parameter buffers for GBSA runner, refreshing only on mutation."""
        if not self._gbsa_params_dirty and self._cached_gbsa_params is not None:
            return self._cached_gbsa_params

        n_act = int(runner["n_act"])
        imp_harm = bool(runner["imp_harm"])
        gp = self._get_dynamic_global_params()
        nb_off = self._converted_system.nonbonded_offsets
        bnd = self._bundle

        q_base = bnd.charges[:n_act].astype(jnp.float32)
        sig_base = bnd.sigmas[:n_act].astype(jnp.float32)
        eps_base = bnd.epsilons[:n_act].astype(jnp.float32)
        q_act, sig_act, eps_act = _apply_particle_offsets(
            q_base, sig_base, eps_base, nb_off, gp, n_act
        )
        rad_act = (
            bnd.radii[:n_act].astype(jnp.float32)
            if (hasattr(bnd, "radii") and bnd.radii is not None)
            else sig_act
        )
        sr_act = (
            bnd.scaled_radii[:n_act].astype(jnp.float32)
            if (hasattr(bnd, "scaled_radii") and bnd.scaled_radii is not None)
            else jnp.maximum(rad_act - 0.09, 0.0) * 0.85
        )
        bp = bnd.bond_params.astype(jnp.float32) if bnd.bond_params is not None else None
        ubp = (
            bnd.urey_bradley_params.astype(jnp.float32)
            if bnd.urey_bradley_params is not None
            else None
        )
        ap = bnd.angle_params.astype(jnp.float32) if bnd.angle_params is not None else None
        dp = (
            bnd.dihedral_params[..., :3].astype(jnp.float32)
            if bnd.dihedral_params is not None
            else None
        )
        ip = (
            (
                bnd.improper_params[..., :2]
                if imp_harm
                else bnd.improper_params[..., :3]
            ).astype(jnp.float32)
            if bnd.improper_params is not None
            else None
        )
        if (
            bnd.exception_chargeprods is not None
            and bnd.exception_sigmas is not None
            and bnd.exception_epsilons is not None
        ):
            e_qp, e_sig, e_eps = _apply_exception_offsets(
                bnd.exception_chargeprods.astype(jnp.float32),
                bnd.exception_sigmas.astype(jnp.float32),
                bnd.exception_epsilons.astype(jnp.float32),
                nb_off,
                gp,
            )
        else:
            e_sig = e_eps = e_qp = None
        cc = bnd.cmap_coeffs.astype(jnp.float32) if bnd.cmap_coeffs is not None else None

        cached = (
            q_act,
            sig_act,
            eps_act,
            rad_act,
            sr_act,
            bp,
            ubp,
            ap,
            dp,
            ip,
            e_sig,
            e_eps,
            e_qp,
            cc,
        )
        self._cached_gbsa_params = cached
        self._gbsa_params_dirty = False
        return cached

    def _ensure_gp_vec(self, param_names: tuple[str, ...]) -> jax.Array | None:
        """Pack dynamic context parameters into a 1D `jnp.float32` vector matching `param_names`."""
        if len(param_names) == 0:
            return None
        return jnp.asarray(
            [float(self._parameters.get(k, 0.0)) for k in param_names],
            dtype=jnp.float32,
        )

    def _morton_method_and_rule_supported(self) -> bool:
        """Check whether current nonbonded method and LJ combining rule are supported on Morton SoA."""
        soa_params = _get_pallas_soa_sig_params()
        comb_ok = self._lj_combining_rule == "lorentz_berthelot" or (
            self._lj_combining_rule == "geometric"
            and "lj_combining_rule" in soa_params
        )
        if not comb_ok:
            return False
        nb_method = self._effective_nonbonded_method().lower()
        is_rf = nb_method in (
            "cutoffperiodic",
            "cutoff_periodic",
            "reactionfield",
            "reaction_field",
            "rf",
        )
        is_lj_pme = nb_method in ("lj_pme", "ljpme")
        do_pme = nb_method in ("pme", "spme", "lj_pme", "ljpme")
        return bool(
            nb_method
            in (
                "pme",
                "spme",
                "lj_pme",
                "ljpme",
                "cutoffperiodic",
                "cutoff_periodic",
                "reactionfield",
                "reaction_field",
                "rf",
                "cutoff",
                "cutoffnonperiodic",
            )
            and (not is_rf or "rf_dielectric" in soa_params)
            and (
                not is_lj_pme
                or (
                    "lj_pme_alpha" in soa_params
                    and "include_ljpme_excl" in soa_params
                )
            )
            and (not do_pme or self._converted_system.pme_grid_dims is not None)
        )

    def _can_use_morton_obs(self) -> bool:
        """Return True when GPU Morton runner can serve `getState(groups=-1)` / `getForcesJax` on GPU."""
        return bool(
            self._nonbonded_engine == "morton_tile_pairs"
            and not in_export_context()
            and self._precision in ("single", "mixed")
            and self._converted_system.has_pbc
            and (self._morton_box_diag is not None or self._box_vectors is not None)
            and (not self._is_triclinic or self._is_reduced_triclinic_supported())
            and self._converted_system.has_nonbonded
            and self._has_finite_cutoff
            and not self._converted_system.has_implicit_solvent
            and _custom_forces_are_gpu_fast(self._custom_forces)
            and (self._exceptions_use_pbc or not self._has_intermolecular_exceptions)
            and self._morton_method_and_rule_supported()
            and not (
                self._morton_runner is None
                and self._effective_integrator_scheme() == "leapfrog"
                and (self._step_count > 0 or self._velocities_initialized)
            )
        )

    def _can_use_morton_step_runner(
        self,
        has_barostat: bool,
        baro_mode: str | None,
        baro_info: dict[str, Any] | None,
        eff_scheme: str,
    ) -> bool:
        """Return True when `step(steps)` can run end-to-end on the persistent GPU Morton BAOAB runner."""
        base_ok = (
            self._nonbonded_engine == "morton_tile_pairs"
            and not in_export_context()
            and self._precision in ("single", "mixed")
            and self._converted_system.has_pbc
            and (self._morton_box_diag is not None or self._box_vectors is not None)
            and (not self._is_triclinic or self._is_reduced_triclinic_supported())
            and self._converted_system.has_nonbonded
            and self._has_finite_cutoff
            and not self._converted_system.has_implicit_solvent
            and _custom_forces_are_gpu_fast(self._custom_forces)
            and self._thermostat_override is None
            and eff_scheme in ("baoab", "langevin", "lfmiddle")
            and not self._has_frozen_atoms
            and self._constraints_are_disjoint_stars
            and self._waters_uniform
            and (self._exceptions_use_pbc or not self._has_intermolecular_exceptions)
            and self._morton_method_and_rule_supported()
        )
        if not base_ok:
            return False
        if has_barostat:
            if baro_info is None:
                return False
            b_type = str(baro_info.get("type", "MonteCarloBarostat"))
            if b_type == "MonteCarloBarostat":
                return baro_mode in ("mc", "scr")
            if b_type in (
                "MonteCarloAnisotropicBarostat",
                "MonteCarloMembraneBarostat",
            ):
                return baro_mode == "mc"
            return False
        return True

    def _can_use_gbsa_step_runner(
        self,
        has_barostat: bool,
        eff_scheme: str,
    ) -> bool:
        """Return True when non-periodic OBC2 GBSA `step(steps)` can run on the persistent GPU runner."""
        return bool(
            self._converted_system.has_implicit_solvent
            and not self._converted_system.has_pbc
            and not has_barostat
            and not in_export_context()
            and (
                self._use_pallas
                or self._nonbonded_engine == "gbsa_tile"
                or str(self._config.nonbonded_engine) == "auto"
            )
            and self._n_particles <= 8192
            and _custom_forces_are_gpu_fast(self._custom_forces)
            and self._thermostat_override is None
            and eff_scheme in ("baoab", "langevin", "lfmiddle")
            and self._precision in ("single", "mixed")
            and not self._has_frozen_atoms
            and self._constraints_are_disjoint_stars
            and self._waters_uniform
            and (
                not self._converted_system.has_nonbonded
                or self._lj_combining_rule == "lorentz_berthelot"
            )
        )

    def _ensure_morton_runner(
        self,
        *,
        gamma_akma: float | None = None,
        has_barostat: bool | None = None,
        baro_mode: str | None = None,
        baro_info: dict[str, Any] | None = None,
        baro_freq: int | None = None,
    ) -> dict[str, Any]:
        """Build or reuse the persistent GPU Morton runner for `step`, `getState`, and `getForcesJax`."""
        ig_any = cast("Any", self._integrator)
        if gamma_akma is None:
            gamma_akma = 1.0 * TAU_AKMA_TO_PS
            if hasattr(ig_any, "getFriction"):
                gamma_akma = (
                    ig_any.getFriction().value_in_unit(unit.picosecond**-1)
                    * TAU_AKMA_TO_PS
                )
        if baro_info is None:
            baro_info = self._converted_system.barostat
        if baro_mode is None:
            raw_bm = self._config.barostat_mode
            baro_mode = "mc" if (raw_bm == "auto" and baro_info is not None) else raw_bm
        if has_barostat is None:
            eff_scheme = self._effective_integrator_scheme()
            cand_baro = bool(
                baro_info is not None
                and baro_mode in ("mc", "scr")
                and (self._converted_system.has_pbc or self._box_vectors is not None)
            )
            has_barostat = cand_baro and self._can_use_morton_step_runner(
                cand_baro, baro_mode, baro_info, eff_scheme
            )
        if baro_freq is None:
            baro_freq = (
                self._config.barostat_frequency_override
                if self._config.barostat_frequency_override is not None
                else (baro_info.get("frequency", 25) if baro_info else 25)
            )

        target_p_akma = (
            float(baro_info["pressure_akma"])
            if (has_barostat and baro_info is not None)
            else 0.0
        )
        baro_type = (
            str(baro_info.get("type", "MonteCarloBarostat"))
            if (has_barostat and baro_info is not None)
            else "MonteCarloBarostat"
        )
        raw_s_axes = (
            baro_info.get("scale_axes", (True, True, True))
            if (has_barostat and baro_info is not None)
            else (True, True, True)
        )
        baro_scale_axes = (
            bool(raw_s_axes[0]),
            bool(raw_s_axes[1]),
            bool(raw_s_axes[2]),
        )
        raw_p_xyz = (
            baro_info.get("pressure_xyz_akma")
            if (has_barostat and baro_info is not None)
            else None
        )
        baro_p_xyz = (
            (float(raw_p_xyz[0]), float(raw_p_xyz[1]), float(raw_p_xyz[2]))
            if raw_p_xyz is not None
            else None
        )
        baro_st_akma = (
            float(baro_info.get("surface_tension_akma", 0.0))
            if (has_barostat and baro_info is not None)
            else 0.0
        )
        baro_xymode = (
            int(baro_info.get("xymode", 0))
            if (has_barostat and baro_info is not None)
            else 0
        )
        baro_z_mode = (
            int(baro_info.get("z_mode", 0))
            if (has_barostat and baro_info is not None)
            else 0
        )

        nl_freq = int(self._config.nl_update_every)
        eff_cmm = self._effective_cmm_frequency()
        eff_alpha = self._effective_pme_alpha()
        nb_method_l = self._effective_nonbonded_method().lower()
        is_lj_pme = nb_method_l in ("lj_pme", "ljpme")
        do_pme = nb_method_l in ("pme", "spme", "lj_pme", "ljpme")
        is_rf = nb_method_l in (
            "cutoffperiodic",
            "cutoff_periodic",
            "reactionfield",
            "reaction_field",
            "rf",
        )
        pme_dims = self._converted_system.pme_grid_dims or (16, 16, 16)
        eff_lj_alpha = float(self._effective_lj_pme_alpha() or 0.34)
        lj_pme_dims = self._converted_system.lj_pme_grid_dims
        if is_lj_pme and lj_pme_dims is None:
            self._sync_bundle_box()
            bv_probe = np.diag(self._get_box_nm_host()) * NM_TO_ANGSTROM
            ewald_tol_probe = float(
                self._converted_system.ewald_error_tolerance
                if self._converted_system.ewald_error_tolerance is not None
                else self._config.ewald_error_tolerance
            )
            _, lj_pme_dims = resolve_ljpme_parameters(
                bv_probe,
                cutoff=float(self._converted_system.nonbonded_cutoff),
                ewald_error_tolerance=ewald_tol_probe,
                explicit_alpha=eff_lj_alpha,
            )
        eff_lj_dims = lj_pme_dims or (12, 12, 12)
        n_pr_terms = sum(
            int(pr.indices.shape[0]) for pr in self._position_restraints
        )
        has_vs_flag = self._converted_system.virtual_sites is not None
        param_names = tuple(sorted(self._parameters.keys()))
        cf_sig = tuple(
            (
                type(cf).__name__,
                int(getattr(cf, "force_group", 0)),
                bool(getattr(cf, "use_pbc", False)),
            )
            for cf in self._custom_forces
        )
        r_key = (
            bool(self._is_triclinic),
            round(float(gamma_akma), 12),
            bool(has_barostat),
            str(baro_mode or ""),
            str(baro_type),
            baro_scale_axes,
            tuple(round(x, 12) for x in baro_p_xyz) if baro_p_xyz is not None else (),
            round(baro_st_akma, 12),
            int(baro_xymode),
            int(baro_z_mode),
            int(baro_freq),
            round(target_p_akma, 12),
            int(eff_cmm),
            int(nl_freq),
            pme_dims,
            round(float(eff_alpha or 0.0), 12),
            str(self._lj_combining_rule),
            bool(is_lj_pme),
            round(eff_lj_alpha if is_lj_pme else 0.0, 12),
            eff_lj_dims if is_lj_pme else (0, 0, 0),
            len(self._position_restraints),
            int(n_pr_terms),
            bool(has_vs_flag),
            bool(self._deterministic_forces),
            cf_sig,
            param_names,
        )
        if self._morton_runner is None or self._morton_runner_key != r_key:
            self._sync_bundle_box()
            box_host_nm = self._get_box_nm_host()
            bv0_np = (
                box_host_nm * NM_TO_ANGSTROM
                if self._is_triclinic
                else np.diag(box_host_nm) * NM_TO_ANGSTROM
            )
            cache_sig = (
                id(self._system),
                int(self._n_particles),
                int(self._system.getNumForces()),
                int(self._system.getNumConstraints()),
                str(self._precision),
                bool(self._pallas_interpret),
                round(
                    float(np.sum(np.asarray(self._bundle.masses[: self._n_particles]))),
                    6,
                ),
                tuple(round(float(x), 7) for x in bv0_np.reshape(-1)),
                r_key,
            )
            cached_pair = _MORTON_RUNNER_CACHE.get(cache_sig)
            if cached_pair is not None and cached_pair[0] is self._system:
                runner = cached_pair[1]
            else:
                r_oh = float(
                    self._converted_system.system_metadata.get("water_r_OH_akma")
                    or 0.9572
                )
                r_hh = float(
                    self._converted_system.system_metadata.get("water_r_HH_akma")
                    or 1.5139
                )
                runner = _build_context_morton_runner(
                    self._bundle,
                    n_act=self._n_particles,
                    precision=self._precision,
                    pallas_interpret=self._pallas_interpret,
                    cutoff=float(self._converted_system.nonbonded_cutoff),
                    nl_skin=float(self._config.nl_skin),
                    lj_switch_width=float(self._converted_system.lj_switch_width),
                    tp_group=self._tp_group,
                    do_pme=do_pme,
                    pme_alpha=float(eff_alpha or 0.34),
                    grid_dims=pme_dims,
                    is_rf=is_rf,
                    rf_dielectric=float(
                        self._converted_system.reaction_field_dielectric
                    ),
                    gamma_akma=float(gamma_akma),
                    nl_freq=nl_freq,
                    cmm_freq=eff_cmm,
                    has_barostat=bool(has_barostat),
                    baro_mode=str(baro_mode or "mc"),
                    baro_freq=int(baro_freq),
                    target_p_akma=target_p_akma,
                    baro_type=baro_type,
                    baro_scale_axes=baro_scale_axes,
                    baro_pressure_xyz_akma=baro_p_xyz,
                    baro_surface_tension_akma=baro_st_akma,
                    baro_xy_isotropic=(
                        baro_xymode
                        == int(openmm.MonteCarloMembraneBarostat.XYIsotropic)
                    ),
                    baro_z_mode=baro_z_mode,
                    r_oh_akma=r_oh,
                    r_hh_akma=r_hh,
                    m_o_akma=self._water_m_O,
                    m_h_akma=self._water_m_H,
                    anchor_jnp=self._anchor_of_atom_jnp,
                    mol_jnp=self._mol_of_atom_jnp,
                    inv_mol_size_jnp=self._inv_mol_size_jnp,
                    host_box0_np=bv0_np,
                    init_positions=self._positions[: self._n_particles],
                    lj_combining_rule=self._lj_combining_rule,
                    is_lj_pme=is_lj_pme,
                    lj_pme_alpha=eff_lj_alpha,
                    lj_pme_grid_dims=eff_lj_dims,
                    deterministic_forces=self._deterministic_forces,
                    has_restraints=(n_pr_terms > 0),
                    virtual_sites=self._converted_system.virtual_sites,
                    pr_params_getter=None,
                    is_triclinic=self._is_triclinic,
                    has_custom_forces=(len(self._custom_forces) > 0),
                    param_names=param_names,
                )
                if len(_MORTON_RUNNER_CACHE) >= 16:
                    _MORTON_RUNNER_CACHE.pop(next(iter(_MORTON_RUNNER_CACHE)))
                _MORTON_RUNNER_CACHE[cache_sig] = (self._system, runner)

            self._morton_runner = runner
            self._morton_runner_key = r_key
            self._morton_tp_state = runner["tp_init"]
            self._morton_fx = runner["fx_init"]
            self._morton_fy = runner["fy_init"]
            self._morton_fz = runner["fz_init"]
            self._morton_G = runner["G_init"]
            self._morton_bonded_dyn = _pack_bundle_fused_bonded(self._bundle)
            self._morton_params_dirty = True
            self._morton_restraints_dirty = True
            self._morton_nl_dirty = True
            self._morton_forces_dirty = True

        return self._morton_runner

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
        eff_alpha = self._effective_pme_alpha()
        use_disp = bool(
            self._config.use_dispersion_correction
            and self._converted_system.use_dispersion_correction
        )

        if self._can_use_morton_step_runner(
            has_barostat, baro_mode, baro_info, eff_scheme
        ):
            runner = self._ensure_morton_runner(
                gamma_akma=float(gamma_akma),
                has_barostat=bool(has_barostat),
                baro_mode=baro_mode,
                baro_info=baro_info,
                baro_freq=int(baro_freq),
            )
            bd = self._morton_bonded_dyn or runner["bonded_static"]
            n_act = int(runner["n_act"])
            (
                q_b,
                sig_b,
                eps_b,
                e_sig_in,
                e_eps_in,
                e_qp_in,
                c_tail_s,
            ) = self._ensure_morton_params(runner, bd, use_disp)
            pr_bufs = self._ensure_restraint_params()
            gp_vec = self._ensure_gp_vec(runner["param_names"])
            box_diag_in = self._morton_box_diag
            if box_diag_in is None:
                if self._box_vectors is None:
                    raise RuntimeError(
                        "Periodic Morton step runner requires non-null box vectors"
                    )
                box_diag_in = (
                    (
                        self._box_vectors
                        if self._box_vectors.ndim == 2
                        else jnp.diag(self._box_vectors)
                    )
                    if self._is_triclinic
                    else (
                        jnp.diag(self._box_vectors)
                        if self._box_vectors.ndim == 2
                        else self._box_vectors
                    )
                )
                self._morton_box_diag = box_diag_in

            (
                pos_out,
                vel_out,
                box_diag_out,
                key_out,
                tp_out,
                fx_out,
                fy_out,
                fz_out,
                G_out,
                mc_sc_out,
                mc_at_out,
                mc_ac_out,
            ) = runner["jit_step"](
                self._positions[:n_act],
                self._velocities[:n_act],
                self._prng_key,
                int(self._step_count),
                int(steps),
                float(dt_akma),
                float(kT_akma),
                box_diag_in,
                q_b,
                sig_b,
                eps_b,
                bd.b_r0,
                bd.b_k,
                bd.a_th0,
                bd.a_kval,
                bd.d_p0,
                bd.d_p1,
                bd.d_p2,
                e_sig_in,
                e_eps_in,
                e_qp_in,
                self._morton_tp_state,
                self._morton_fx,
                self._morton_fy,
                self._morton_fz,
                self._morton_G,
                bool(self._morton_nl_dirty),
                bool(self._morton_forces_dirty),
                self._morton_mc_scale,
                self._morton_mc_att,
                self._morton_mc_acc,
                c_tail_s,
                *pr_bufs,
                self._custom_forces,
                gp_vec,
            )

            self._positions = pos_out
            self._velocities = vel_out
            self._velocities_initialized = True
            self._prng_key = key_out
            self._morton_tp_state = tp_out
            self._morton_fx = fx_out
            self._morton_fy = fy_out
            self._morton_fz = fz_out
            self._morton_G = G_out
            self._morton_mc_scale = mc_sc_out
            self._morton_mc_att = mc_at_out
            self._morton_mc_acc = mc_ac_out
            self._morton_nl_dirty = False
            self._morton_forces_dirty = False
            self._host_nl_dirty = True
            if has_barostat:
                self._morton_box_diag = box_diag_out
                self._box_nm_dirty = True
                self._bundle_box_dirty = True
            self._step_count += steps
            self._time += steps * dt_akma
            return

        if self._can_use_gbsa_step_runner(has_barostat, eff_scheme):
            n_pr_gb = sum(
                int(pr.indices.shape[0]) for pr in self._position_restraints
            )
            param_names_gb = tuple(sorted(self._parameters.keys()))
            cf_sig_gb = tuple(
                (type(cf).__name__, int(getattr(cf, "force_group", 0)))
                for cf in self._custom_forces
            )
            gb_key = (
                round(float(gamma_akma), 12),
                int(eff_cmm),
                bool(self._converted_system.has_nonbonded),
                bool(self._has_finite_cutoff),
                round(float(self._converted_system.nonbonded_cutoff), 12)
                if self._has_finite_cutoff
                else 0.0,
                round(float(self._converted_system.lj_switch_width), 12),
                bool(self._gb_use_neighbor_list),
                round(float(self._converted_system.solvent_dielectric), 12),
                round(float(self._converted_system.solute_dielectric), 12),
                round(float(self._converted_system.gbsa_surface_tension), 12),
                len(self._position_restraints),
                int(n_pr_gb),
                self._converted_system.virtual_sites is not None,
                cf_sig_gb,
                param_names_gb,
            )
            if self._gbsa_runner is None or self._gbsa_runner_key != gb_key:
                r_oh = float(
                    self._converted_system.system_metadata.get("water_r_OH_akma")
                    or 0.9572
                )
                r_hh = float(
                    self._converted_system.system_metadata.get("water_r_HH_akma")
                    or 1.5139
                )
                gb_runner = _build_context_gbsa_runner(
                    self._bundle,
                    n_act=self._n_particles,
                    precision=self._precision,
                    pallas_interpret=self._pallas_interpret,
                    has_nonbonded=bool(self._converted_system.has_nonbonded),
                    has_finite_cutoff=bool(self._has_finite_cutoff),
                    cutoff=float(self._converted_system.nonbonded_cutoff),
                    lj_switch_width=float(self._converted_system.lj_switch_width),
                    gb_use_neighbor_list=bool(self._gb_use_neighbor_list),
                    solvent_dielectric=float(
                        self._converted_system.solvent_dielectric
                    ),
                    solute_dielectric=float(self._converted_system.solute_dielectric),
                    gbsa_surface_tension=float(
                        self._converted_system.gbsa_surface_tension
                    ),
                    gamma_akma=float(gamma_akma),
                    cmm_freq=eff_cmm,
                    r_oh_akma=r_oh,
                    r_hh_akma=r_hh,
                    m_o_akma=self._water_m_O,
                    m_h_akma=self._water_m_H,
                    has_restraints=(n_pr_gb > 0),
                    virtual_sites=self._converted_system.virtual_sites,
                    has_custom_forces=(len(self._custom_forces) > 0),
                    param_names=param_names_gb,
                    init_positions=self._positions[: self._n_particles],
                )
                self._gbsa_runner = gb_runner
                self._gbsa_runner_key = gb_key
                self._gbsa_fx = gb_runner["fx_init"]
                self._gbsa_fy = gb_runner["fy_init"]
                self._gbsa_fz = gb_runner["fz_init"]
                self._gbsa_params_dirty = True
                self._gbsa_forces_dirty = True
                self._morton_restraints_dirty = True

            gb_runner = self._gbsa_runner
            n_act = int(gb_runner["n_act"])
            gb_params = self._ensure_gbsa_params(gb_runner)
            pr_bufs = self._ensure_restraint_params()
            gp_vec_gb = self._ensure_gp_vec(gb_runner["param_names"])
            gb_forces_dirty = bool(
                self._gbsa_forces_dirty
                or self._morton_forces_dirty
                or self._gbsa_last_pos_id != id(self._positions)
            )
            pos_out, vel_out, key_out, fx_out, fy_out, fz_out = gb_runner["jit_step"](
                self._positions[:n_act],
                self._velocities[:n_act],
                self._prng_key,
                int(self._step_count),
                int(steps),
                float(dt_akma),
                float(kT_akma),
                *gb_params,
                self._gbsa_fx,
                self._gbsa_fy,
                self._gbsa_fz,
                gb_forces_dirty,
                *pr_bufs,
                self._custom_forces,
                gp_vec_gb,
            )
            self._positions = pos_out
            self._velocities = vel_out
            self._velocities_initialized = True
            self._prng_key = key_out
            self._gbsa_fx = fx_out
            self._gbsa_fy = fy_out
            self._gbsa_fz = fz_out
            self._gbsa_forces_dirty = False
            self._morton_forces_dirty = False
            self._gbsa_last_pos_id = id(self._positions)
            self._host_nl_dirty = True
            self._step_count += steps
            self._time += steps * dt_akma
            return

        self._sync_bundle_box()
        self._ensure_host_nl_fresh()
        masses = self._bundle.masses[: self._n_particles]
        inv_m = jnp.where(masses > 1e-12, 1.0 / masses, 0.0)
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
                self._effective_nonbonded_method(),
                self._converted_system.has_nonbonded,
                use_disp,
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
                pme_alpha=eff_alpha,
                nonbonded_offsets=self._converted_system.nonbonded_offsets,
                default_params=self._converted_system.default_parameters,
                lj_combining_rule=self._lj_combining_rule,
                exceptions_use_pbc=self._exceptions_use_pbc,
                dispersion_follows_parameters=self._dispersion_follows_parameters,
                lj_pme_alpha=self._effective_lj_pme_alpha(),
                lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
            )
            remaining -= chunk
            self._step_count += chunk
            self._time += chunk * dt_akma
            self._morton_nl_dirty = True
            self._morton_forces_dirty = True

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
        return {k: jnp.asarray(v, dtype=self._dtype) for k, v in self._parameters.items()}

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
        """Vectorized O(N) intramolecular PBC unwrap and per-atom molecular COMs via flat-CSR arrays."""
        n = int(pos_akma.shape[0])
        if n == 0:
            return pos_akma.copy(), pos_akma.copy()
        pos_jnp = jnp.asarray(pos_akma[: self._n_particles], dtype=jnp.float64)
        box_jnp = jnp.asarray(box_akma, dtype=jnp.float64)
        pos_u_jnp, com_jnp = _unwrap_molecules_and_com_jax(
            pos_jnp,
            box_jnp,
            self._anchor_of_atom_jnp,
            self._mol_of_atom_jnp,
            self._inv_mol_size_jnp,
            bool(self._converted_system.has_pbc),
        )
        return np.asarray(pos_u_jnp, dtype=np.float64), np.asarray(com_jnp, dtype=np.float64)

    def _apply_mc_barostat(  # noqa: PLR0911
        self,
        baro_info: dict[str, Any],
        kT_akma: float,
    ) -> None:
        """Apply Monte Carlo volume move with PBC-unwrapped molecular COM scaling and OpenMM adaptive step size."""
        if self._box_vectors is None:
            return
        self._ensure_host_nl_fresh()

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
            xymode_val = int(baro_info.get("xymode", 0))
            zmode_val = int(baro_info.get("z_mode", 0))
            z_free = zmode_val == int(openmm.MonteCarloMembraneBarostat.ZFree)
            z_const_vol = zmode_val == int(
                openmm.MonteCarloMembraneBarostat.ConstantVolume
            )
            xy_aniso = xymode_val == int(
                openmm.MonteCarloMembraneBarostat.XYAnisotropic
            )
            do_z_move = False
            ax_choice = 0
            if z_free:
                if xy_aniso:
                    ax_choice = int(jax.random.randint(subkey_axis, (), 0, 3))
                    do_z_move = ax_choice == 2
                else:
                    do_z_move = int(jax.random.randint(subkey_axis, (), 0, 3)) == 2
            elif xy_aniso:
                ax_choice = int(jax.random.randint(subkey_axis, (), 0, 2))
            if do_z_move:
                s_z = 1.0 + delta_v * u_rand
                if s_z <= 0:
                    return
                axis_scales = np.array([1.0, 1.0, s_z], dtype=np.float64)
            elif xy_aniso:
                s_ax = 1.0 + delta_v * u_rand
                if s_ax <= 0:
                    return
                s_x = s_ax if ax_choice == 0 else 1.0
                s_y = s_ax if ax_choice == 1 else 1.0
                s_z = (1.0 / s_ax) if z_const_vol else 1.0
                axis_scales = np.array([s_x, s_y, s_z], dtype=np.float64)
            else:
                scale_xy = 1.0 + delta_v * u_rand
                if scale_xy <= 0:
                    return
                s_xy = math.sqrt(scale_xy)
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

        _, com_per_atom_jnp = _unwrap_molecules_and_com_jax(
            self._positions[: self._n_particles],
            self._box_vectors,
            self._anchor_of_atom_jnp,
            self._mol_of_atom_jnp,
            self._inv_mol_size_jnp,
            bool(self._converted_system.has_pbc),
        )
        axis_scales_jnp = jnp.asarray(axis_scales, dtype=self._positions.dtype)
        new_pos_jnp = (
            self._positions[: self._n_particles]
            + (axis_scales_jnp[None, :] - 1.0) * com_per_atom_jnp
        ).astype(self._positions.dtype)
        if self._converted_system.virtual_sites is not None:
            new_pos_jnp = self._converted_system.compute_virtual_sites(new_pos_jnp)

        gp = self._get_dynamic_global_params()
        ewald_tol = (
            self._converted_system.ewald_error_tolerance
            if self._converted_system.ewald_error_tolerance is not None
            else self._config.ewald_error_tolerance
        )
        use_disp = bool(
            self._config.use_dispersion_correction
            and self._converted_system.use_dispersion_correction
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
            self._effective_nonbonded_method(),
            self._converted_system.has_nonbonded,
            use_disp,
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
            nonbonded_offsets=self._converted_system.nonbonded_offsets,
            default_params=self._converted_system.default_parameters,
            lj_combining_rule=self._lj_combining_rule,
            exceptions_use_pbc=self._exceptions_use_pbc,
            dispersion_follows_parameters=self._dispersion_follows_parameters,
            lj_pme_alpha=self._effective_lj_pme_alpha(),
            lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
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
            self._effective_nonbonded_method(),
            self._converted_system.has_nonbonded,
            use_disp,
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
            nonbonded_offsets=self._converted_system.nonbonded_offsets,
            default_params=self._converted_system.default_parameters,
            lj_combining_rule=self._lj_combining_rule,
            exceptions_use_pbc=self._exceptions_use_pbc,
            dispersion_follows_parameters=self._dispersion_follows_parameters,
            lj_pme_alpha=self._effective_lj_pme_alpha(),
            lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
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
            self._refresh_box_caches_from_vectors()
            self._morton_nl_dirty = True
            self._morton_forces_dirty = True
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
        self._morton_mc_scale = jnp.full((3,), self._mc_volume_scale, dtype=jnp.float32)
        self._morton_mc_att = jnp.full((3,), self._mc_num_attempted, dtype=jnp.int32)
        self._morton_mc_acc = jnp.full((3,), self._mc_num_accepted, dtype=jnp.int32)

    def _compute_virial(self, forces: jnp.ndarray | np.ndarray | None = None) -> float:
        """Compute scalar thermodynamic virial W = -dU/d(lam)|_{lam=1} under periodic molecular COM scaling."""
        self._ensure_host_nl_fresh()
        if forces is None and self._converted_system.has_pbc and self._box_vectors is not None:
            gp = self._get_dynamic_global_params()
            ewald_tol = (
                self._converted_system.ewald_error_tolerance
                if self._converted_system.ewald_error_tolerance is not None
                else self._config.ewald_error_tolerance
            )
            pos_u, com_u = _unwrap_molecules_and_com_jax(
                self._positions[: self._n_particles],
                self._box_vectors,
                self._anchor_of_atom_jnp,
                self._mol_of_atom_jnp,
                self._inv_mol_size_jnp,
                bool(self._converted_system.has_pbc),
            )
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
                    nonbonded_method=self._effective_nonbonded_method(),
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
                    use_pallas=self._config.use_pallas,
                    pallas_interpret=self._config.pallas_interpret,
                    lj_switch_width=self._converted_system.lj_switch_width,
                    neighbor_idx=getattr(self, "_neighbor_idx", None),
                    use_pallas_pme=False,
                    custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
                    tile_pair_nl=self._tile_pair_nl,
                    is_triclinic=self._is_triclinic,
                    gb_use_neighbor_list=self._gb_use_neighbor_list,
                    gbsa_surface_tension=self._converted_system.gbsa_surface_tension,
                    pme_alpha=self._effective_pme_alpha(),
                    nonbonded_offsets=self._converted_system.nonbonded_offsets,
                    default_params=self._converted_system.default_parameters,
                    lj_combining_rule=self._lj_combining_rule,
                    exceptions_use_pbc=self._exceptions_use_pbc,
                    dispersion_follows_parameters=self._dispersion_follows_parameters,
                    force_pure_jax_tile_pairs=True,
                    lj_pme_alpha=self._effective_lj_pme_alpha(),
                    lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
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
                self._effective_nonbonded_method(),
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
                nonbonded_offsets=self._converted_system.nonbonded_offsets,
                default_params=self._converted_system.default_parameters,
                lj_combining_rule=self._lj_combining_rule,
                exceptions_use_pbc=self._exceptions_use_pbc,
                dispersion_follows_parameters=self._dispersion_follows_parameters,
                lj_pme_alpha=self._effective_lj_pme_alpha(),
                lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
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

        _, com_jnp = _unwrap_molecules_and_com_jax(
            self._positions[: self._n_particles],
            self._box_vectors,
            self._anchor_of_atom_jnp,
            self._mol_of_atom_jnp,
            self._inv_mol_size_jnp,
            bool(self._converted_system.has_pbc),
        )
        new_box = current_box * mu
        new_box_jnp = jnp.asarray(new_box, dtype=self._box_vectors.dtype)
        self._box_vectors = new_box_jnp
        self._bundle = eqx.tree_at(lambda b: b.box, self._bundle, new_box_jnp)
        self._converted_system = eqx.tree_at(
            lambda s: s.bundle.box, self._converted_system, new_box_jnp
        )
        self._refresh_box_caches_from_vectors()
        self._positions = (
            self._positions[: self._n_particles] + (mu - 1.0) * com_jnp
        ).astype(self._positions.dtype)
        self._velocities = self._velocities / mu
        self._morton_nl_dirty = True
        self._morton_forces_dirty = True
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
        self._morton_nl_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_forces_dirty = True
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

    def getPositionsJax(  # noqa: N802
        self,
        *,
        in_nm: bool = True,
        enforcePeriodicBox: bool = False,
    ) -> jax.Array:
        """Return zero-copy device `jax.Array` of particle positions `(N, 3)` in `nm` (default) or `Å`."""
        pos = self._positions[: self._n_particles]
        has_box = (
            self._converted_system.has_pbc
            or self._morton_box_diag is not None
            or self._box_vectors is not None
        )
        if enforcePeriodicBox and has_box:
            if self._morton_box_diag is not None:
                box_gpu = (
                    self._morton_box_diag.astype(pos.dtype)
                    if self._morton_box_diag.ndim == 2
                    else jnp.diag(self._morton_box_diag.astype(pos.dtype))
                )
            elif self._box_vectors is not None:
                box_gpu = (
                    self._box_vectors.astype(pos.dtype)
                    if self._box_vectors.ndim == 2
                    else jnp.diag(self._box_vectors.astype(pos.dtype))
                )
            else:
                box_gpu = None
            if box_gpu is not None:
                pos = _wrap_molecules_jax(
                    pos,
                    box_gpu,
                    self._mol_of_atom_jnp,
                    self._inv_mol_size_jnp,
                )
        if in_nm:
            return pos * jnp.asarray(ANGSTROM_TO_NM, dtype=pos.dtype)
        return pos

    def setPositionsJax(  # noqa: N802
        self,
        positions: Any,
        *,
        in_nm: bool = True,
    ) -> None:
        """Set particle positions from a `jax.Array` or DLPack (`__dlpack__`) tensor without host copy."""
        pos_dev = _from_jax_or_dlpack(positions)
        if pos_dev.ndim != 2 or pos_dev.shape[1] != 3:
            raise ValueError(
                f"Positions tensor must have shape (N, 3), got {pos_dev.shape}"
            )
        if pos_dev.shape[0] != self._n_particles:
            raise openmm.OpenMMException(
                "Called setPositionsJax() on a Context with the wrong number of positions"
            )
        pos_akma = pos_dev.astype(self._dtype)
        if in_nm:
            pos_akma = pos_akma * jnp.asarray(NM_TO_ANGSTROM, dtype=self._dtype)
        if self._converted_system.virtual_sites is not None:
            pos_akma = self._converted_system.compute_virtual_sites(pos_akma)
        self._positions = pos_akma
        self._morton_nl_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_forces_dirty = True
        self._host_nl_dirty = True

    def getVelocitiesJax(  # noqa: N802
        self,
        *,
        in_nm_per_ps: bool = True,
    ) -> jax.Array:
        """Return zero-copy device `jax.Array` of particle velocities `(N, 3)` in `nm/ps` (default) or `Å/τ_AKMA`."""
        vel = self._velocities[: self._n_particles]
        if in_nm_per_ps:
            return vel * jnp.asarray(VELOCITY_AKMA_TO_OMM, dtype=vel.dtype)
        return vel

    def setVelocitiesJax(  # noqa: N802
        self,
        velocities: Any,
        *,
        in_nm_per_ps: bool = True,
    ) -> None:
        """Set particle velocities from a `jax.Array` or DLPack (`__dlpack__`) tensor without host copy."""
        vel_dev = _from_jax_or_dlpack(velocities)
        if vel_dev.ndim != 2 or vel_dev.shape[1] != 3:
            raise ValueError(
                f"Velocities tensor must have shape (N, 3), got {vel_dev.shape}"
            )
        if vel_dev.shape[0] != self._n_particles:
            raise openmm.OpenMMException(
                "Called setVelocitiesJax() on a Context with the wrong number of velocities"
            )
        vel_akma = vel_dev.astype(self._dtype)
        if in_nm_per_ps:
            vel_akma = vel_akma * jnp.asarray(VELOCITY_OMM_TO_AKMA, dtype=self._dtype)
        if self._converted_system.virtual_sites is not None:
            vs_idx = self._converted_system.virtual_sites.order
            vel_akma = vel_akma.at[vs_idx].set(0.0)
        self._velocities = vel_akma
        self._velocities_initialized = True

    def getForcesJax(  # noqa: N802
        self,
        *,
        in_kj_per_mol_nm: bool = True,
        groups: int | set[int] = -1,
    ) -> jax.Array:
        """Return device `jax.Array` of particle forces `(N, 3)` in `kJ/(mol·nm)` (default) or `kcal/(mol·Å)`."""
        if isinstance(groups, set):
            raw_groups = sum(1 << g for g in groups if 0 <= g <= 31)
        else:
            raw_groups = int(groups)
        groups_mask = int(np.int32(raw_groups))
        use_disp = bool(
            self._config.use_dispersion_correction
            and self._converted_system.use_dispersion_correction
        )

        if groups_mask == -1 and self._can_use_morton_obs():
            runner = self._ensure_morton_runner()
            bd = self._morton_bonded_dyn or runner["bonded_static"]
            n_act = int(runner["n_act"])
            if (
                not self._morton_nl_dirty
                and not self._morton_forces_dirty
                and self._morton_tp_state is not None
            ):
                f_omm_dev = runner["jit_unpermute_forces"](
                    self._morton_fx,
                    self._morton_fy,
                    self._morton_fz,
                    self._morton_tp_state.inv_perm,
                )
            else:
                (
                    q_b,
                    sig_b,
                    eps_b,
                    e_sig_in,
                    e_eps_in,
                    e_qp_in,
                    c_tail_s,
                ) = self._ensure_morton_params(runner, bd, use_disp)
                pr_bufs = self._ensure_restraint_params()
                gp_vec = self._ensure_gp_vec(runner["param_names"])
                box_diag_in = self._morton_box_diag
                if box_diag_in is None:
                    if self._box_vectors is None:
                        raise RuntimeError(
                            "Periodic Morton getForcesJax requires non-null box vectors"
                        )
                    box_diag_in = (
                        (
                            self._box_vectors
                            if self._box_vectors.ndim == 2
                            else jnp.diag(self._box_vectors)
                        )
                        if self._is_triclinic
                        else (
                            jnp.diag(self._box_vectors)
                            if self._box_vectors.ndim == 2
                            else self._box_vectors
                        )
                    )
                    self._morton_box_diag = box_diag_in
                (
                    _scalars_kj_dev,
                    f_omm_dev,
                    tp_out,
                    fx_out,
                    fy_out,
                    fz_out,
                    G_out,
                ) = runner["jit_energy_and_forces"](
                    self._positions[:n_act],
                    self._velocities[:n_act],
                    box_diag_in,
                    q_b,
                    sig_b,
                    eps_b,
                    bd.b_r0,
                    bd.b_k,
                    bd.a_th0,
                    bd.a_kval,
                    bd.d_p0,
                    bd.d_p1,
                    bd.d_p2,
                    e_sig_in,
                    e_eps_in,
                    e_qp_in,
                    self._morton_tp_state,
                    self._morton_G,
                    bool(self._morton_nl_dirty),
                    bool(self._morton_forces_dirty),
                    c_tail_s,
                    pr_bufs,
                    self._custom_forces,
                    gp_vec,
                )
                self._morton_tp_state = tp_out
                self._morton_fx = fx_out
                self._morton_fy = fy_out
                self._morton_fz = fz_out
                self._morton_G = G_out
                self._morton_nl_dirty = False
                self._morton_forces_dirty = False
            if in_kj_per_mol_nm:
                return f_omm_dev
            return f_omm_dev * jnp.asarray(FORCE_OMM_TO_AKMA, dtype=f_omm_dev.dtype)

        self._ensure_host_nl_fresh()
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
            groups_mask,
            self._converted_system.has_pbc,
            self._converted_system.nonbonded_cutoff,
            self._converted_system.bond_force_groups,
            self._converted_system.angle_force_groups,
            self._converted_system.dihedral_force_groups,
            self._converted_system.cmap_force_groups,
            self._effective_nonbonded_method(),
            self._converted_system.has_nonbonded,
            use_disp,
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
            nonbonded_offsets=self._converted_system.nonbonded_offsets,
            default_params=self._converted_system.default_parameters,
            lj_combining_rule=self._lj_combining_rule,
            exceptions_use_pbc=self._exceptions_use_pbc,
            dispersion_follows_parameters=self._dispersion_follows_parameters,
            lj_pme_alpha=self._effective_lj_pme_alpha(),
            lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
        )
        if self._converted_system.virtual_sites is not None:
            forces_akma = self._converted_system.redistribute_virtual_site_forces(
                forces_akma, self._positions
            )
        if in_kj_per_mol_nm:
            return forces_akma * jnp.asarray(FORCE_AKMA_TO_OMM, dtype=forces_akma.dtype)
        return forces_akma

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
        prev_is_triclinic = self._is_triclinic
        self._is_triclinic = bool(np.max(off_diag) > 1e-6)
        self._box_vectors = box_canonical
        box_bundle = box_canonical.astype(jnp.float32)
        self._bundle = eqx.tree_at(lambda bnd: bnd.box, self._bundle, box_bundle)
        new_pme_dims = self._converted_system.pme_grid_dims
        new_lj_pme_dims = self._converted_system.lj_pme_grid_dims
        eff_nb_m = self._effective_nonbonded_method().lower()
        if (
            self._converted_system.has_pbc
            and eff_nb_m in ("pme", "lj_pme")
        ):
            explicit_omm_grid = False
            explicit_lj_omm_grid = False
            for f_idx in range(self._system.getNumForces()):
                f_obj = self._system.getForce(f_idx)
                if isinstance(f_obj, openmm.NonbondedForce):
                    _, nx, ny, nz = f_obj.getPMEParameters()
                    if nx > 0 and ny > 0 and nz > 0:
                        explicit_omm_grid = True
                    get_ljpme = getattr(f_obj, "getLJPMEParameters", None)
                    if callable(get_ljpme):
                        _, nxl, nyl, nzl = cast("tuple[Any, int, int, int]", get_ljpme())
                        if nxl > 0 and nyl > 0 and nzl > 0:
                            explicit_lj_omm_grid = True
                    break
            box_diag_np = np.diag(box_np)
            cutoff_a = float(self._converted_system.nonbonded_cutoff)
            eff_ewald_tol = (
                float(self._converted_system.ewald_error_tolerance)
                if self._converted_system.ewald_error_tolerance is not None
                else float(self._config.ewald_error_tolerance)
            )
            if not explicit_omm_grid:
                if (
                    isinstance(self._config.pme_grid_spacing, _DefaultFloat)
                    and 0.0 < cutoff_a < float("inf")
                ):
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
            if eff_nb_m == "lj_pme" and not explicit_lj_omm_grid and 0.0 < cutoff_a < float("inf"):
                _, new_lj_pme_dims = resolve_ljpme_parameters(
                    box_diag_np,
                    cutoff=cutoff_a,
                    ewald_error_tolerance=eff_ewald_tol,
                    explicit_alpha=self._converted_system.explicit_lj_pme_alpha or 0.0,
                )
        prev_pme_dims = self._converted_system.pme_grid_dims
        prev_lj_pme_dims = self._converted_system.lj_pme_grid_dims
        self._converted_system = dataclasses.replace(
            self._converted_system,
            bundle=self._bundle,
            is_triclinic=self._is_triclinic,
            pme_grid_dims=new_pme_dims,
            lj_pme_grid_dims=new_lj_pme_dims,
        )
        if (
            new_pme_dims != prev_pme_dims
            or new_lj_pme_dims != prev_lj_pme_dims
            or self._is_triclinic != prev_is_triclinic
        ):
            self._morton_runner = None
            self._morton_runner_key = None
        self._refresh_box_caches_from_vectors()
        self._morton_nl_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_forces_dirty = True
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
        self._morton_params_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_params_dirty = True
        self._gbsa_forces_dirty = True

    def getParameters(self) -> dict[str, float]:  # noqa: N802
        return dict(self._parameters)

    # --- Constraints & Virtual Sites ---
    def computeVirtualSites(self) -> None:  # noqa: N802
        if self._converted_system.virtual_sites is not None:
            self._positions = self._converted_system.compute_virtual_sites(self._positions)

    def applyConstraints(self, tolerance: float = 1e-5) -> None:  # noqa: N802, ARG002
        """Apply distance constraints on current positions (SETTLE and SHAKE)."""
        self._sync_bundle_box()
        masses = self._bundle.masses[: self._n_particles]
        self._positions = self._constraints.apply_positions(
            self._positions, self._positions, masses[:, None], self._box_vectors
        ).astype(self._positions.dtype)
        self.computeVirtualSites()
        self._morton_nl_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_forces_dirty = True

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
        self._sync_bundle_box()
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
            self._bundle = eqx.tree_at(
                lambda bnd: bnd.box, self._bundle, self._box_vectors.astype(jnp.float32)
            )
        self._refresh_box_caches_from_vectors()
        self._time = float(state_dict["time"])
        self._step_count = int(state_dict["step_count"])
        self._parameters = dict(state_dict.get("parameters", {}))
        if "prng_key" in state_dict:
            self._prng_key = jnp.asarray(state_dict["prng_key"])
        self._morton_params_dirty = True
        self._morton_nl_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_params_dirty = True
        self._gbsa_forces_dirty = True
        if self._should_use_neighbor_list() or self._can_use_gbsa_tiles():
            self._init_neighbor_list(self._positions[: self._n_particles])

    def reinitialize(self, preserveState: bool = False) -> None:
        chk = self.createCheckpoint() if preserveState else None
        for k_sig in [
            k_s
            for k_s, pair in _MORTON_RUNNER_CACHE.items()
            if pair[0] is self._system
        ]:
            _MORTON_RUNNER_CACHE.pop(k_sig, None)
        eff_conv_cfg = (
            dataclasses.replace(self._config, anisotropic_barostat=True)
            if (
                not self._config.anisotropic_barostat
                and _system_has_anisotropic_or_membrane_barostat(self._system)
            )
            else self._config
        )
        self._converted_system = from_openmm_system(
            self._system, self._integrator, eff_conv_cfg
        )
        self._anisotropic_barostat = bool(
            self._config.anisotropic_barostat
            or _system_has_anisotropic_or_membrane_barostat(self._system)
        )
        self._is_triclinic = bool(self._converted_system.is_triclinic)
        self._n_particles = self._converted_system.n_real_particles
        self._has_finite_cutoff = (
            0.0 < float(self._converted_system.nonbonded_cutoff) < float("inf")
        )
        self._lj_combining_rule = (
            str(self._config.lj_combining_rule)
            if str(self._config.lj_combining_rule) != "lorentz_berthelot"
            else str(self._converted_system.lj_combining_rule)
        )
        self._exceptions_use_pbc = bool(
            self._converted_system.exceptions_use_pbc or self._config.exceptions_use_pbc
        )
        self._dispersion_follows_parameters = bool(
            self._converted_system.dispersion_follows_parameters
            and self._config.dispersion_follows_parameters
        )
        self._use_lj_pme = bool(
            self._config.use_lj_pme
            or str(self._converted_system.nonbonded_method).lower() in ("lj_pme", "ljpme")
        )
        self._bundle = self._converted_system.bundle
        if self._dtype == jnp.float64:
            self._bundle = eqx.tree_at(
                lambda b: b.masses, self._bundle, self._bundle.masses.astype(jnp.float64)
            )
        if self._converted_system.has_pbc and self._converted_system.bundle.box is not None:
            self._box_vectors = self._converted_system.bundle.box.astype(self._dtype)
        self._refresh_box_caches_from_vectors()
        self._custom_forces = tuple(self._converted_system.custom_forces)
        self._position_restraints = tuple(self._converted_system.position_restraints)
        self._parameters = dict(self._converted_system.default_parameters)
        self._molecules = self._build_molecules()
        self._morton_runner = None
        self._morton_runner_key = None
        self._morton_bonded_dyn = None
        self._cached_morton_params = None
        self._cached_morton_restraints = None
        self._morton_params_dirty = True
        self._morton_restraints_dirty = True
        self._morton_nl_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_runner = None
        self._gbsa_runner_key = None
        self._cached_gbsa_params = None
        self._gbsa_params_dirty = True
        self._gbsa_forces_dirty = True
        self._gbsa_last_pos_id = None
        if not preserveState:
            self._positions = jnp.zeros((self._n_particles, 3), dtype=self._dtype)
            self._velocities = jnp.zeros((self._n_particles, 3), dtype=self._dtype)
            self._velocities_initialized = False
            self._morton_mc_scale = jnp.full((3,), 0.01, dtype=jnp.float32)
            self._morton_mc_att = jnp.zeros((3,), dtype=jnp.int32)
            self._morton_mc_acc = jnp.zeros((3,), dtype=jnp.int32)
            self._time = 0.0
            self._step_count = 0
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
        self._sync_bundle_box()
        bundle = self._bundle
        conv = self._converted_system
        meta = conv.system_metadata
        custom_force_map = meta.get("omm_custom_force_to_compiled_idx", {})
        updated_cf = list(self._custom_forces)

        new_gbsa_st = conv.gbsa_surface_tension
        new_explicit_alpha = conv.explicit_pme_alpha
        new_pme_grid_dims = conv.pme_grid_dims
        new_explicit_lj_alpha = conv.explicit_lj_pme_alpha
        new_lj_pme_grid_dims = conv.lj_pme_grid_dims
        new_nb_offsets = conv.nonbonded_offsets
        new_exc_use_pbc = conv.exceptions_use_pbc

        for f_idx in range(self._system.getNumForces()):
            f = self._system.getForce(f_idx)
            if force is not None and not _is_same_openmm_force(f, force):
                continue

            if isinstance(f, openmm.NonbondedForce):
                get_exc_pbc = getattr(f, "getExceptionsUsePeriodicBoundaryConditions", None)
                if callable(get_exc_pbc):
                    new_exc_use_pbc = bool(get_exc_pbc()) or bool(self._config.exceptions_use_pbc)

                if f.getNonbondedMethod() in (
                    openmm.NonbondedForce.PME,
                    openmm.NonbondedForce.LJPME,
                ):
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

                if f.getNonbondedMethod() == openmm.NonbondedForce.LJPME or self._use_lj_pme:
                    get_ljpme = getattr(f, "getLJPMEParameters", None)
                    if callable(get_ljpme):
                        alpha_lj_omm, nxl, nyl, nzl = cast(
                            "tuple[Any, int, int, int]", get_ljpme()
                        )
                        alpha_lj_nm = (
                            float(alpha_lj_omm.value_in_unit(unit.nanometer**-1))
                            if hasattr(alpha_lj_omm, "value_in_unit")
                            else float(alpha_lj_omm)
                        )
                        new_explicit_lj_alpha = (
                            float(alpha_lj_nm / NM_TO_ANGSTROM) if alpha_lj_nm > 0.0 else None
                        )
                        if int(nxl) > 0 and int(nyl) > 0 and int(nzl) > 0:
                            new_lj_pme_grid_dims = (int(nxl), int(nyl), int(nzl))

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

                # Validate and update NonbondedForce particle/exception parameter offset scales
                get_num_p_off = getattr(f, "getNumParticleParameterOffsets", None)
                get_p_off = getattr(f, "getParticleParameterOffset", None)
                get_num_e_off = getattr(f, "getNumExceptionParameterOffsets", None)
                get_e_off = getattr(f, "getExceptionParameterOffset", None)
                n_p_off = int(cast("int | float", get_num_p_off())) if callable(get_num_p_off) else 0
                n_e_off = int(cast("int | float", get_num_e_off())) if callable(get_num_e_off) else 0
                prev_p_keys = new_nb_offsets.particle_omm_keys if new_nb_offsets is not None else ()
                prev_e_keys = new_nb_offsets.exception_omm_keys if new_nb_offsets is not None else ()
                if n_p_off != len(prev_p_keys) or n_e_off != len(prev_e_keys):
                    raise openmm.OpenMMException(
                        "updateParametersInContext: The number of offsets cannot be changed"
                    )
                if new_nb_offsets is not None and (n_p_off > 0 or n_e_off > 0):
                    off_dt = new_nb_offsets.particle_scales.dtype
                    np_off_dt = np.float64 if off_dt == jnp.float64 else np.float32
                    p_scales_upd = np.zeros((n_p_off, 3), dtype=np_off_dt)
                    if callable(get_p_off):
                        for k_off in range(n_p_off):
                            pn_raw, pidx_raw, cs_raw, ss_raw, es_raw = get_p_off(k_off)
                            if (str(pn_raw), int(pidx_raw)) != prev_p_keys[k_off]:
                                raise openmm.OpenMMException(
                                    f"updateParametersInContext: Particle parameter offset {k_off} "
                                    "has changed its particle or its parameter"
                                )
                            p_scales_upd[k_off, 0] = (
                                float(cs_raw.value_in_unit(unit.elementary_charge))
                                if hasattr(cs_raw, "value_in_unit")
                                else float(cs_raw)
                            )
                            p_scales_upd[k_off, 1] = (
                                float(ss_raw.value_in_unit(unit.nanometer)) * NM_TO_ANGSTROM
                                if hasattr(ss_raw, "value_in_unit")
                                else float(ss_raw) * NM_TO_ANGSTROM
                            )
                            p_scales_upd[k_off, 2] = (
                                float(es_raw.value_in_unit(unit.kilojoules_per_mole))
                                * KJ_PER_MOL_TO_KCAL_PER_MOL
                                if hasattr(es_raw, "value_in_unit")
                                else float(es_raw) * KJ_PER_MOL_TO_KCAL_PER_MOL
                            )
                    e_scales_upd = np.zeros((n_e_off, 3), dtype=np_off_dt)
                    if callable(get_e_off):
                        for k_off in range(n_e_off):
                            pn_raw, eidx_raw, qps_raw, ss_raw, es_raw = get_e_off(k_off)
                            if (str(pn_raw), int(eidx_raw)) != prev_e_keys[k_off]:
                                raise openmm.OpenMMException(
                                    f"updateParametersInContext: Exception parameter offset {k_off} "
                                    "has changed its exception or its parameter"
                                )
                            e_scales_upd[k_off, 0] = (
                                float(qps_raw.value_in_unit(unit.elementary_charge**2))
                                if hasattr(qps_raw, "value_in_unit")
                                else float(qps_raw)
                            )
                            e_scales_upd[k_off, 1] = (
                                float(ss_raw.value_in_unit(unit.nanometer)) * NM_TO_ANGSTROM
                                if hasattr(ss_raw, "value_in_unit")
                                else float(ss_raw) * NM_TO_ANGSTROM
                            )
                            e_scales_upd[k_off, 2] = (
                                float(es_raw.value_in_unit(unit.kilojoules_per_mole))
                                * KJ_PER_MOL_TO_KCAL_PER_MOL
                                if hasattr(es_raw, "value_in_unit")
                                else float(es_raw) * KJ_PER_MOL_TO_KCAL_PER_MOL
                            )
                    new_nb_offsets = eqx.tree_at(
                        lambda o: (o.particle_scales, o.exception_scales),
                        new_nb_offsets,
                        (
                            jnp.asarray(p_scales_upd, dtype=off_dt),
                            jnp.asarray(e_scales_upd, dtype=off_dt),
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

        refreshed_pr: list[PositionRestraints] | None = None
        if len(self._position_restraints) > 0 and (
            force is None or isinstance(force, openmm.CustomExternalForce)
        ):
            cand_pr: list[PositionRestraints] = []
            for f_scan_idx in range(self._system.getNumForces()):
                if f_scan_idx in custom_force_map:
                    continue
                f_scan = self._system.getForce(f_scan_idx)
                if isinstance(f_scan, openmm.CustomExternalForce):
                    m_pr = match_custom_force_pattern(f_scan)
                    if m_pr is not None and m_pr.target_term == "position_restraint":
                        k_col = m_pr.params[:, 3]
                        cand_pr.append(
                            cast(
                                "PositionRestraints",
                                PositionRestraints(
                                    indices=jnp.asarray(
                                        m_pr.indices.flatten(), dtype=jnp.int32
                                    ),
                                    k_x=jnp.asarray(k_col, dtype=jnp.float32),
                                    k_y=jnp.asarray(k_col, dtype=jnp.float32),
                                    k_z=jnp.asarray(k_col, dtype=jnp.float32),
                                    x0=jnp.asarray(
                                        m_pr.params[:, 0], dtype=jnp.float32
                                    ),
                                    y0=jnp.asarray(
                                        m_pr.params[:, 1], dtype=jnp.float32
                                    ),
                                    z0=jnp.asarray(
                                        m_pr.params[:, 2], dtype=jnp.float32
                                    ),
                                    force_group=f_scan.getForceGroup(),
                                ),
                            )
                        )
            if len(cand_pr) == len(self._position_restraints):
                refreshed_pr = cand_pr
                self._position_restraints = tuple(cand_pr)
                self._morton_restraints_dirty = True

        self._bundle = bundle
        self._custom_forces = tuple(updated_cf)
        self._exceptions_use_pbc = bool(new_exc_use_pbc)
        conv_updated = eqx.tree_at(
            lambda s: (s.bundle, s.custom_forces),
            self._converted_system,
            (bundle, tuple(updated_cf)),
        )
        if refreshed_pr is not None:
            conv_updated = dataclasses.replace(
                conv_updated,
                position_restraints=refreshed_pr,
            )
        if new_nb_offsets is not conv_updated.nonbonded_offsets:
            conv_updated = dataclasses.replace(
                conv_updated,
                nonbonded_offsets=new_nb_offsets,
            )
        static_runner_changed = (
            new_explicit_alpha != conv_updated.explicit_pme_alpha
            or new_pme_grid_dims != conv_updated.pme_grid_dims
            or new_explicit_lj_alpha != conv_updated.explicit_lj_pme_alpha
            or new_lj_pme_grid_dims != conv_updated.lj_pme_grid_dims
            or new_exc_use_pbc != conv_updated.exceptions_use_pbc
        )
        if (
            new_gbsa_st != conv_updated.gbsa_surface_tension
            or static_runner_changed
        ):
            if new_gbsa_st != conv_updated.gbsa_surface_tension:
                self._gbsa_runner = None
                self._gbsa_runner_key = None
            conv_updated = dataclasses.replace(
                conv_updated,
                gbsa_surface_tension=float(new_gbsa_st),
                explicit_pme_alpha=new_explicit_alpha,
                pme_grid_dims=new_pme_grid_dims,
                explicit_lj_pme_alpha=new_explicit_lj_alpha,
                lj_pme_grid_dims=new_lj_pme_grid_dims,
                exceptions_use_pbc=bool(new_exc_use_pbc),
            )
        if static_runner_changed:
            self._morton_runner = None
            self._morton_runner_key = None
        self._converted_system = conv_updated
        self._morton_params_dirty = True
        self._morton_forces_dirty = True
        self._gbsa_params_dirty = True
        self._gbsa_forces_dirty = True
        if self._morton_runner is not None:
            self._refresh_morton_bonded_dyn()

    # --- Parameter Derivatives ---
    def _compute_parameter_derivatives(self, groups_mask: int = -1) -> dict[str, float]:
        """Compute analytical energy parameter derivatives dE/dlambda via single-pass jax.grad."""
        self._sync_bundle_box()
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

        nb_off = self._converted_system.nonbonded_offsets
        g_nb = int(self._converted_system.force_group_by_component.get("nonbonded", 0))
        if (
            nb_off is not None
            and len(nb_off.param_names) > 0
            and self._converted_system.has_nonbonded
            and (((1 << g_nb) & groups_mask) != 0)
        ):
            self._ensure_host_nl_fresh()
            gp_nb = {
                k: jnp.asarray(v, dtype=self._dtype) for k, v in self._parameters.items()
            }
            ewald_tol = (
                self._converted_system.ewald_error_tolerance
                if self._converted_system.ewald_error_tolerance is not None
                else self._config.ewald_error_tolerance
            )
            use_disp = bool(
                self._config.use_dispersion_correction
                and self._converted_system.use_dispersion_correction
            )
            eff_alpha = self._effective_pme_alpha()
            eff_lj_alpha = self._effective_lj_pme_alpha()

            def _nb_energy_of_gp(gp_in: dict[str, Array]) -> Array:
                return _eval_nonbonded_terms(
                    pos_akma,
                    self._box_vectors,
                    self._bundle,
                    self._converted_system.force_group_by_component,
                    groups_mask,
                    self._converted_system.has_pbc,
                    self._converted_system.nonbonded_cutoff,
                    nonbonded_method=self._effective_nonbonded_method(),
                    has_nonbonded=self._converted_system.has_nonbonded,
                    use_dispersion_correction=use_disp,
                    ewald_error_tolerance=ewald_tol,
                    rf_dielectric=self._converted_system.reaction_field_dielectric,
                    nonbonded_engine=self._nonbonded_engine,
                    pme_grid_dims=self._converted_system.pme_grid_dims,
                    use_pallas=False,
                    pallas_interpret=False,
                    lj_switch_width=self._converted_system.lj_switch_width,
                    neighbor_idx=getattr(self, "_neighbor_idx", None),
                    use_pallas_pme=False,
                    custom_nonbonded_chunk_size=self._config.custom_nonbonded_chunk_size,
                    tile_pair_nl=self._tile_pair_nl,
                    is_triclinic=self._is_triclinic,
                    pme_alpha=eff_alpha,
                    global_params=gp_in,
                    nonbonded_offsets=nb_off,
                    default_params=self._converted_system.default_parameters,
                    lj_combining_rule=self._lj_combining_rule,
                    exceptions_use_pbc=self._exceptions_use_pbc,
                    dispersion_follows_parameters=self._dispersion_follows_parameters,
                    force_pure_jax_tile_pairs=True,
                    lj_pme_alpha=eff_lj_alpha,
                    lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
                )

            _, grad_nb = jax.value_and_grad(_nb_energy_of_gp)(gp_nb)
            for p_name in nb_off.param_names:
                if p_name in grad_nb:
                    deriv_kj = float(grad_nb[p_name]) * KCAL_PER_MOL_TO_KJ_PER_MOL
                    param_derivatives[p_name] = (
                        param_derivatives.get(p_name, 0.0) + deriv_kj
                    )

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
        """Vectorized O(N) whole-molecule PBC wrapping via flat-CSR molecular COMs."""
        self._sync_bundle_box()
        if self._box_vectors is None or pos_akma.shape[0] == 0:
            return pos_akma
        n_mols = int(self._inv_mol_size_np.shape[0])
        if n_mols == 0:
            return pos_akma
        box = np.asarray(self._box_vectors, dtype=np.float64)
        a_vec, b_vec, c_vec = box[0], box[1], box[2]
        mol_ids = self._mol_of_atom_np
        pos_f64 = np.asarray(pos_akma[: self._n_particles], dtype=np.float64)
        com_x = np.bincount(mol_ids, weights=pos_f64[:, 0], minlength=n_mols) * self._inv_mol_size_np
        com_y = np.bincount(mol_ids, weights=pos_f64[:, 1], minlength=n_mols) * self._inv_mol_size_np
        com_z = np.bincount(mol_ids, weights=pos_f64[:, 2], minlength=n_mols) * self._inv_mol_size_np
        # OpenMM Reference 3-step sequential back-substitution (z -> y -> x)
        diffs = np.zeros((n_mols, 3), dtype=np.float64)
        if abs(c_vec[2]) > 1e-12:
            diffs += np.floor(com_z / c_vec[2])[:, None] * c_vec[None, :]
        if abs(b_vec[1]) > 1e-12:
            diffs += np.floor((com_y - diffs[:, 1]) / b_vec[1])[:, None] * b_vec[None, :]
        if abs(a_vec[0]) > 1e-12:
            diffs += np.floor((com_x - diffs[:, 0]) / a_vec[0])[:, None] * a_vec[None, :]
        return pos_f64 - diffs[mol_ids]

    def _build_molecules(self) -> tuple[tuple[int, ...], ...]:
        """Compute connected components of particles and precompute flat-CSR topology arrays."""
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

        active_c_edges: list[tuple[int, int]] = []
        if bundle.constraint_pairs is not None:
            cpairs = np.asarray(bundle.constraint_pairs)
            cmask = np.asarray(bundle.constraint_mask)
            for (p1, p2), active in zip(cpairs, cmask, strict=False):
                if active and 0 <= p1 < n_p and 0 <= p2 < n_p:
                    u_i, v_i = int(p1), int(p2)
                    adj[u_i].append(v_i)
                    adj[v_i].append(u_i)
                    if u_i != v_i:
                        active_c_edges.append((u_i, v_i))

        if (
            bundle.water_indices is not None
            and bundle.water_mask is not None
            and int(bundle.water_indices.shape[0]) > 0
        ):
            w_idx_np = np.asarray(bundle.water_indices, dtype=np.int32)
            w_msk_np = np.asarray(bundle.water_mask, dtype=bool)
            for (o_i, h1_i, h2_i), active in zip(w_idx_np, w_msk_np, strict=False):
                if active and 0 <= o_i < n_p and 0 <= h1_i < n_p and 0 <= h2_i < n_p:
                    adj[int(o_i)].append(int(h1_i))
                    adj[int(h1_i)].append(int(o_i))
                    adj[int(o_i)].append(int(h2_i))
                    adj[int(h2_i)].append(int(o_i))

        vs_set: set[int] = set()
        if self._converted_system.virtual_sites is not None:
            vs_obj = self._converted_system.virtual_sites
            for v_i in np.asarray(vs_obj.order, dtype=np.int32).tolist():
                if 0 <= int(v_i) < n_p:
                    vs_set.add(int(v_i))
            for gen in vs_obj.generations:
                if gen.n_linear > 0:
                    for row in np.asarray(gen.linear_indices, dtype=np.int32):
                        v_i = int(row[0])
                        if 0 <= v_i < n_p:
                            for p_k in (int(row[1]), int(row[2]), int(row[3])):
                                if 0 <= p_k < n_p and p_k != v_i:
                                    adj[v_i].append(p_k)
                                    adj[p_k].append(v_i)
                if gen.n_local > 0:
                    for row in np.asarray(gen.local_indices, dtype=np.int32):
                        v_i = int(row[0])
                        if 0 <= v_i < n_p:
                            for p_k in (int(row[1]), int(row[2]), int(row[3])):
                                if 0 <= p_k < n_p and p_k != v_i:
                                    adj[v_i].append(p_k)
                                    adj[p_k].append(v_i)

        visited: set[int] = set()
        molecules: list[tuple[int, ...]] = []
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

        n_mols = len(molecules)
        mol_of_atom = np.zeros((n_p,), dtype=np.int32)
        anchor_of_atom = np.zeros((n_p,), dtype=np.int32)
        inv_mol_size = np.ones((max(n_mols, 1),), dtype=np.float64)
        for m_idx, mol in enumerate(molecules):
            if not mol:
                continue
            anchor = next((int(a) for a in mol if int(a) not in vs_set), int(mol[0]))
            idx_arr = np.asarray(mol, dtype=np.int32)
            mol_of_atom[idx_arr] = m_idx
            anchor_of_atom[idx_arr] = anchor
            inv_mol_size[m_idx] = 1.0 / float(len(mol))
        if n_mols == 0:
            inv_mol_size = np.zeros((0,), dtype=np.float64)

        self._mol_of_atom_np = mol_of_atom
        self._anchor_of_atom_np = anchor_of_atom
        self._inv_mol_size_np = inv_mol_size
        self._mol_of_atom_jnp = jnp.asarray(mol_of_atom, dtype=jnp.int32)
        self._anchor_of_atom_jnp = jnp.asarray(anchor_of_atom, dtype=jnp.int32)
        self._inv_mol_size_jnp = jnp.asarray(inv_mol_size, dtype=jnp.float32)

        masses_np = (
            np.asarray(bundle.masses[:n_p], dtype=np.float64)
            if n_p > 0
            else np.zeros((0,))
        )
        if vs_set and n_p > 0:
            non_vs = np.ones((n_p,), dtype=bool)
            non_vs[list(vs_set)] = False
            self._has_frozen_atoms = bool(np.any((masses_np <= 1e-12) & non_vs))
        else:
            self._has_frozen_atoms = bool(n_p > 0 and np.any(masses_np <= 1e-12))

        intermol_exc = False
        if (
            n_p > 0
            and bundle.exception_pairs is not None
            and bundle.exception_mask is not None
        ):
            ep_np = np.asarray(bundle.exception_pairs, dtype=np.int32)
            em_np = np.asarray(bundle.exception_mask, dtype=bool)
            if ep_np.shape[0] > 0:
                v_e = em_np & (ep_np[:, 0] >= 0) & (ep_np[:, 0] < n_p) & (ep_np[:, 1] >= 0) & (ep_np[:, 1] < n_p)
                if np.any(v_e):
                    e0 = ep_np[v_e, 0]
                    e1 = ep_np[v_e, 1]
                    intermol_exc = bool(np.any(mol_of_atom[e0] != mol_of_atom[e1]))
        self._has_intermolecular_exceptions = intermol_exc

        disjoint_stars = True
        if active_c_edges and n_p > 0:
            cdeg = np.zeros((n_p,), dtype=np.int32)
            for u_i, v_i in active_c_edges:
                cdeg[u_i] += 1
                cdeg[v_i] += 1
            if int(np.max(cdeg)) > 4:
                disjoint_stars = False
            else:
                for u_i, v_i in active_c_edges:
                    if min(int(cdeg[u_i]), int(cdeg[v_i])) != 1:
                        disjoint_stars = False
                        break
        self._constraints_are_disjoint_stars = disjoint_stars

        if (
            n_p > 0
            and bundle.water_indices is not None
            and bundle.n_waters is not None
            and int(bundle.n_waters) > 0
        ):
            wi_np = np.asarray(bundle.water_indices[: int(bundle.n_waters)], dtype=np.int32)
            m_o_arr = masses_np[wi_np[:, 0]]
            m_h1_arr = masses_np[wi_np[:, 1]]
            m_h2_arr = masses_np[wi_np[:, 2]]
            self._water_m_O = float(m_o_arr[0])
            self._water_m_H = float(m_h1_arr[0])
            self._waters_uniform = bool(
                np.max(np.abs(m_o_arr - self._water_m_O)) < 1e-4
                and np.max(np.abs(m_h1_arr - self._water_m_H)) < 1e-4
                and np.max(np.abs(m_h2_arr - self._water_m_H)) < 1e-4
            )
        else:
            self._water_m_O = 15.999
            self._water_m_H = 1.008
            self._waters_uniform = True

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

        # Box vectors (host-cached in NVT; 3-float GPU sync only when NPT updated box)
        box_nm = self._get_box_nm_host()

        # Positions (GPU-wrapped before single D2H transfer when enforcePeriodicBox=True)
        pos_nm = None
        if req_pos:
            has_box = self._converted_system.has_pbc or self._morton_box_diag is not None or self._box_vectors is not None
            if enforcePeriodicBox and has_box:
                if self._morton_box_diag is not None:
                    box_gpu = (
                        self._morton_box_diag.astype(self._positions.dtype)
                        if self._morton_box_diag.ndim == 2
                        else jnp.diag(self._morton_box_diag.astype(self._positions.dtype))
                    )
                elif self._box_vectors is not None:
                    box_gpu = (
                        self._box_vectors.astype(self._positions.dtype)
                        if self._box_vectors.ndim == 2
                        else jnp.diag(self._box_vectors.astype(self._positions.dtype))
                    )
                else:
                    box_gpu = None
                if box_gpu is not None:
                    pos_wrapped_dev = _wrap_molecules_jax(
                        self._positions[: self._n_particles],
                        box_gpu,
                        self._mol_of_atom_jnp,
                        self._inv_mol_size_jnp,
                    )
                    pos_nm = np.asarray(pos_wrapped_dev, dtype=np.float64) * ANGSTROM_TO_NM
                else:
                    pos_nm = (
                        np.asarray(self._positions[: self._n_particles], dtype=np.float64)
                        * ANGSTROM_TO_NM
                    )
            else:
                pos_nm = (
                    np.asarray(self._positions[: self._n_particles], dtype=np.float64)
                    * ANGSTROM_TO_NM
                )

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
            use_disp = bool(
                self._config.use_dispersion_correction
                and self._converted_system.use_dispersion_correction
            )
            if groups_mask == -1 and self._can_use_morton_obs():
                runner = self._ensure_morton_runner()
                bd = self._morton_bonded_dyn or runner["bonded_static"]
                n_act = int(runner["n_act"])
                if (
                    req_forces
                    and not req_energy
                    and not self._morton_nl_dirty
                    and not self._morton_forces_dirty
                    and self._morton_tp_state is not None
                ):
                    f_omm_dev = runner["jit_unpermute_forces"](
                        self._morton_fx,
                        self._morton_fy,
                        self._morton_fz,
                        self._morton_tp_state.inv_perm,
                    )
                    forces_omm = np.asarray(f_omm_dev, dtype=np.float64)
                else:
                    (
                        q_b,
                        sig_b,
                        eps_b,
                        e_sig_in,
                        e_eps_in,
                        e_qp_in,
                        c_tail_s,
                    ) = self._ensure_morton_params(runner, bd, use_disp)
                    pr_bufs = self._ensure_restraint_params()
                    gp_vec = self._ensure_gp_vec(runner["param_names"])
                    box_diag_in = self._morton_box_diag
                    if box_diag_in is None:
                        if self._box_vectors is None:
                            raise RuntimeError(
                                "Periodic Morton getState requires non-null box vectors"
                            )
                        box_diag_in = (
                            (
                                self._box_vectors
                                if self._box_vectors.ndim == 2
                                else jnp.diag(self._box_vectors)
                            )
                            if self._is_triclinic
                            else (
                                jnp.diag(self._box_vectors)
                                if self._box_vectors.ndim == 2
                                else self._box_vectors
                            )
                        )
                        self._morton_box_diag = box_diag_in
                    (
                        scalars_kj_dev,
                        f_omm_dev,
                        tp_out,
                        fx_out,
                        fy_out,
                        fz_out,
                        G_out,
                    ) = runner["jit_energy_and_forces"](
                        self._positions[:n_act],
                        self._velocities[:n_act],
                        box_diag_in,
                        q_b,
                        sig_b,
                        eps_b,
                        bd.b_r0,
                        bd.b_k,
                        bd.a_th0,
                        bd.a_kval,
                        bd.d_p0,
                        bd.d_p1,
                        bd.d_p2,
                        e_sig_in,
                        e_eps_in,
                        e_qp_in,
                        self._morton_tp_state,
                        self._morton_G,
                        bool(self._morton_nl_dirty),
                        bool(self._morton_forces_dirty),
                        c_tail_s,
                        pr_bufs,
                        self._custom_forces,
                        gp_vec,
                    )
                    self._morton_tp_state = tp_out
                    self._morton_fx = fx_out
                    self._morton_fy = fy_out
                    self._morton_fz = fz_out
                    self._morton_G = G_out
                    self._morton_nl_dirty = False
                    self._morton_forces_dirty = False
                    if req_energy:
                        pe_ke_np = np.asarray(scalars_kj_dev, dtype=np.float64)
                        pe_kj = float(pe_ke_np[0])
                        ke_kj = float(pe_ke_np[1])
                    if req_forces:
                        forces_omm = np.asarray(f_omm_dev, dtype=np.float64)
            else:
                self._ensure_host_nl_fresh()
                gp = self._get_dynamic_global_params()
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
                    self._effective_nonbonded_method(),
                    self._converted_system.has_nonbonded,
                    use_disp,
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
                    nonbonded_offsets=self._converted_system.nonbonded_offsets,
                    default_params=self._converted_system.default_parameters,
                    lj_combining_rule=self._lj_combining_rule,
                    exceptions_use_pbc=self._exceptions_use_pbc,
                    dispersion_follows_parameters=self._dispersion_follows_parameters,
                    lj_pme_alpha=self._effective_lj_pme_alpha(),
                    lj_pme_grid_dims=self._converted_system.lj_pme_grid_dims,
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
