"""Drop-in openmm.Context and openmm.State replacements powered by Prolix JAX backend."""

from __future__ import annotations

import contextlib
import pickle
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import openmm
from openmm import unit

from prolix.openmm.config import ProlixOpenMMConfig, ProlixPlatform, parse_platform_properties
from prolix.openmm.converter import (
    CompiledRBTorsionForce,
    ConvertedOpenMMSystem,
    PositionRestraints,
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
from prolix.physics.constraints import ConstraintAlgorithm, make_constraint
from prolix.physics.pbc import minimum_image_displacement, reduce_triclinic_box_vectors
from prolix.types.bundles import MolecularBundle  # noqa: TC001

if TYPE_CHECKING:
    from jaxtyping import Array

BOLTZMANN_KCAL: float = 0.0019872041  # kB in kcal/(mol·K)


# =============================================================================
# 1. ProlixState
# =============================================================================


class ProlixState(openmm.State):
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
        bv = self._box_vectors_nm
        lines = [
            '<?xml version="1.0" ?>',
            f'<State openmmVersion="{openmm.__version__}" stepCount="{self._step_count}" time="{self._time_ps}" type="State" version="1">',
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
        return openmm.XmlSerializer.deserialize(xml_str)

    @property
    def this(self) -> Any:
        if self._c_state is None:
            self._c_state = self._build_c_state()
        return self._c_state.this

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


def _eval_bonded_terms(
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
) -> jnp.ndarray:
    """Evaluate harmonic bonds, angles, dihedrals, and restraints for active groups."""
    e_total = jnp.array(0.0, dtype=jnp.float32)
    n = pos.shape[0]

    # Harmonic Bonds
    g_bond = force_group_by_component.get("bond", 0)
    if (groups_mask & (1 << g_bond)) != 0 and bundle.bond_idx is not None and bundle.bond_params is not None:
        b_idx = bundle.bond_idx
        b_params = bundle.bond_params
        b_mask = bundle.bond_mask
        p1 = b_idx[:, 0]
        p2 = b_idx[:, 1]
        valid = b_mask & (p1 < n) & (p2 < n)
        p1_safe = jnp.where(valid, p1, 0)
        p2_safe = jnp.where(valid, p2, 0)
        dr = _minimum_image_dr(pos[p1_safe] - pos[p2_safe], box, has_pbc)
        r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
        r0 = b_params[:, 0]
        k = b_params[:, 1]
        e_b = 0.5 * k * (r - r0) ** 2
        e_total = e_total + jnp.sum(jnp.where(valid, e_b, 0.0))

    # Harmonic Angles
    g_angle = force_group_by_component.get("angle", 0)
    if (groups_mask & (1 << g_angle)) != 0 and bundle.angle_idx is not None and bundle.angle_params is not None:
        a_idx = bundle.angle_idx
        a_params = bundle.angle_params
        a_mask = bundle.angle_mask
        p1 = a_idx[:, 0]
        p2 = a_idx[:, 1]
        p3 = a_idx[:, 2]
        valid = a_mask & (p1 < n) & (p2 < n) & (p3 < n)
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
    g_dih = force_group_by_component.get("dihedral", 0)
    if (groups_mask & (1 << g_dih)) != 0 and bundle.dihedral_idx is not None and bundle.dihedral_params is not None:
        d_idx = bundle.dihedral_idx
        d_params = bundle.dihedral_params
        d_mask = bundle.dihedral_mask
        p1 = d_idx[:, 0]
        p2 = d_idx[:, 1]
        p3 = d_idx[:, 2]
        p4 = d_idx[:, 3]
        valid = d_mask & (p1 < n) & (p2 < n) & (p3 < n) & (p4 < n)
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
    if (groups_mask & (1 << g_imp)) != 0 and bundle.improper_idx is not None and bundle.improper_params is not None:
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

        # Periodic dihedrals: k * (1 + cos(per * phi - phase))
        per = i_params[:, 0]
        phase = i_params[:, 1]
        k_per = i_params[:, 2]
        e_per = k_per * (1.0 + jnp.cos(per * phi - phase))

        # Harmonic impropers: k * (dphi ** 2)
        # In pattern_matcher.py, params are stored as [k_akma, theta0, 0.0]
        k_harm = i_params[:, 0]
        phi0 = i_params[:, 1]
        diff_raw = phi - phi0 + jnp.pi
        dphi = diff_raw - 2.0 * jnp.pi * jnp.floor(diff_raw / (2.0 * jnp.pi)) - jnp.pi
        e_harm = k_harm * (dphi ** 2)

        e_i = jnp.where(is_periodic, e_per, e_harm)
        e_total = e_total + jnp.sum(jnp.where(valid, e_i, 0.0))

    return e_total


def _eval_nonbonded_terms(
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
    cutoff: float,
) -> jnp.ndarray:
    """Evaluate pairwise Lennard-Jones, Coulomb, exclusions, and 1-4 exceptions."""
    g_nb = force_group_by_component.get("nonbonded", 0)
    if (groups_mask & (1 << g_nb)) == 0 or bundle.charges is None:
        return jnp.array(0.0, dtype=jnp.float32)

    n = pos.shape[0]
    q = bundle.charges[:n]
    sig = bundle.sigmas[:n]
    eps = bundle.epsilons[:n]

    # Upper triangular all-pairs
    dr = pos[:, None, :] - pos[None, :, :]
    if has_pbc and box is not None:
        dr = _minimum_image_dr(dr, box, has_pbc)
    r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
    mask_tri = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    r_safe = jnp.where(mask_tri, r, 1.0)

    sig_mat = 0.5 * (sig[:, None] + sig[None, :])
    eps_mat = jnp.sqrt(jnp.maximum(eps[:, None] * eps[None, :], 0.0))
    q_mat = q[:, None] * q[None, :]

    sr = sig_mat / r_safe
    sr6 = sr**6
    sr12 = sr6**2
    u_lj = 4.0 * eps_mat * (sr12 - sr6)
    u_elec = 332.0637 * (q_mat / r_safe)
    u_pair = u_lj + u_elec

    # Cutoff truncation if non-periodic cutoff
    if cutoff < float("inf") and not has_pbc:
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

        dr_ex = _minimum_image_dr(pos[p1_s] - pos[p2_s], box, has_pbc)
        r_ex = jnp.sqrt(jnp.sum(dr_ex**2, axis=-1) + 1e-12)
        r_ex_s = jnp.where(valid_ex, r_ex, 1.0)
        sig_ex = 0.5 * (sig[p1_s] + sig[p2_s])
        eps_ex = jnp.sqrt(jnp.maximum(eps[p1_s] * eps[p2_s], 0.0))
        q_ex = q[p1_s] * q[p2_s]

        sr_e = sig_ex / r_ex_s
        sr6_e = sr_e**6
        sr12_e = sr6_e**2
        u_ex = 4.0 * eps_ex * (sr12_e - sr6_e) + 332.0637 * (q_ex / r_ex_s)
        if cutoff < float("inf") and not has_pbc:
            u_ex = jnp.where(r_ex_s < cutoff, u_ex, 0.0)
        tot = tot - jnp.sum(jnp.where(valid_ex, u_ex, 0.0))

    # 1-4 Exceptions
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

        dr_exc = _minimum_image_dr(pos[ep1_s] - pos[ep2_s], box, has_pbc)
        r_exc = jnp.sqrt(jnp.sum(dr_exc**2, axis=-1) + 1e-12)
        r_exc_s = jnp.where(valid_exc, r_exc, 1.0)
        sig_14 = bundle.exception_sigmas
        eps_14 = bundle.exception_epsilons
        qp_14 = bundle.exception_chargeprods

        sr_14 = sig_14 / r_exc_s
        sr6_14 = sr_14**6
        sr12_14 = sr6_14**2
        u_14 = 4.0 * eps_14 * (sr12_14 - sr6_14) + 332.0637 * (qp_14 / r_exc_s)
        tot = tot + jnp.sum(jnp.where(valid_exc, u_14, 0.0))

    return tot


def _eval_system_potential_energy(
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
) -> jnp.ndarray:
    """Compute total potential energy (in kcal/mol) for active force groups."""
    e = _eval_bonded_terms(pos, box, bundle, force_group_by_component, groups_mask, has_pbc)
    e = e + _eval_nonbonded_terms(
        pos, box, bundle, force_group_by_component, groups_mask, has_pbc, cutoff
    )

    # Position Restraints
    n = pos.shape[0]
    for pr in position_restraints:
        if (groups_mask & (1 << pr.force_group)) != 0:
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
        if (groups_mask & (1 << cf.force_group)) != 0:
            e = e + cf.energy_akma(pos, box, global_params)

    return e


@eqx.filter_jit
def _pure_compute_energy_and_forces(
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure JIT evaluation of potential energy and negative gradient forces."""

    def e_fn(p: jnp.ndarray) -> jnp.ndarray:
        return _eval_system_potential_energy(
            p,
            box,
            bundle,
            custom_forces,
            position_restraints,
            global_params,
            force_group_by_component,
            groups_mask,
            has_pbc,
            cutoff,
        )

    pe, grad = jax.value_and_grad(e_fn)(pos)
    forces = -grad
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
    num_steps: int,
    has_pbc: bool,
    cutoff: float,
    remove_com_motion: bool,
    constraints: ConstraintAlgorithm | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """Pure compiled multi-step rollout kernel via jax.lax.fori_loop."""
    c1 = jnp.exp(-gamma * dt) if is_langevin else 1.0
    c2 = jnp.sqrt(jnp.maximum(1.0 - c1 * c1, 0.0)) if is_langevin else 0.0

    def step_body(i: int, val: tuple[jnp.ndarray, jnp.ndarray, jax.Array]):
        pos, vel, k = val

        # 1. Force evaluation at current positions
        _, f = _pure_compute_energy_and_forces(
            pos,
            box,
            bundle,
            custom_forces,
            position_restraints,
            global_params,
            force_group_by_component,
            0xFFFFFFFF,
            has_pbc,
            cutoff,
        )

        # 2. V(0.5): half kick
        v = vel + 0.5 * dt * (f * inv_mass[:, None])

        # 3. A(0.5): half drift
        r = pos + 0.5 * dt * v

        # 4. O(1.0): Ornstein-Uhlenbeck stochastic update (if Langevin)
        if is_langevin:
            k, subk = jax.random.split(k)
            thermal_sigma = jnp.sqrt(kT * inv_mass)[:, None]
            noise = jax.random.normal(subk, shape=r.shape, dtype=r.dtype)
            v_noisy = c1 * v + c2 * thermal_sigma * noise
            v = jnp.where(inv_mass[:, None] > 1e-12, v_noisy, 0.0)

        # 5. A(0.5): second half drift
        r = r + 0.5 * dt * v

        # Constraint projection: project unconstrained positions and correct velocity
        if constraints is not None:
            r_c = constraints.apply_positions(pos, r, masses[:, None], box).astype(r.dtype)
            v = v + (r_c - r) / dt
            r = r_c

        # 6. Force evaluation at new positions
        _, f_new = _pure_compute_energy_and_forces(
            r,
            box,
            bundle,
            custom_forces,
            position_restraints,
            global_params,
            force_group_by_component,
            0xFFFFFFFF,
            has_pbc,
            cutoff,
        )

        # 7. V(0.5): second half kick
        v = v + 0.5 * dt * (f_new * inv_mass[:, None])

        # 8. Remove center of mass motion if requested
        if remove_com_motion:
            total_m = jnp.sum(masses)
            v_com = jnp.sum(v * masses[:, None], axis=0) / jnp.maximum(total_m, 1e-12)
            v = jnp.where(inv_mass[:, None] > 1e-12, v - v_com, 0.0)

        return r, v, k

    init_val = (positions, velocities, key)
    final_pos, final_vel, final_key = jax.lax.fori_loop(0, num_steps, step_body, init_val)
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
        self._positions = jnp.zeros((self._n_particles, 3), dtype=jnp.float32)
        self._velocities = jnp.zeros((self._n_particles, 3), dtype=jnp.float32)
        if self._converted_system.has_pbc and self._converted_system.bundle.box is not None:
            self._box_vectors = self._converted_system.bundle.box
        else:
            try:
                raw_bv = system.getDefaultPeriodicBoxVectors()
                box_akma = np.asarray([to_akma(v, "length") for v in raw_bv], dtype=np.float64)
                if abs(float(np.linalg.det(box_akma))) > 1e-6:
                    self._box_vectors = reduce_triclinic_box_vectors(
                        jnp.asarray(box_akma, dtype=jnp.float32)
                    )
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
            b.water_indices[: b.n_waters]
            if (b.water_indices is not None and b.n_waters > 0)
            else None
        )
        constraint_pairs = (
            b.constraint_pairs[: b.n_constraints]
            if (b.constraint_pairs is not None and b.n_constraints > 0)
            else None
        )
        constraint_lengths = (
            b.constraint_lengths[: b.n_constraints]
            if (b.constraint_lengths is not None and b.n_constraints > 0)
            else None
        )
        self._constraints: ConstraintAlgorithm = make_constraint(
            water_indices=water_indices,
            constraint_pairs=constraint_pairs,
            constraint_lengths=constraint_lengths,
            r_OH=self._converted_system.system_metadata.get("water_r_OH_akma"),
            r_HH=self._converted_system.system_metadata.get("water_r_HH_akma"),
            settle_velocity_tol=self._config.settle_velocity_tol,
        )

        # Molecular topology graph for getMolecules()
        self._molecules = self._build_molecules()

        # Wire integrator link
        if hasattr(integrator, "_prolix_context"):
            integrator._prolix_context = self
        integrator.step = self.step

    # --- Core Getters ---
    def getSystem(self) -> openmm.System:  # noqa: N802
        return self._system

    def getIntegrator(self) -> openmm.Integrator:  # noqa: N802
        return self._integrator

    def getPlatform(self) -> openmm.Platform:  # noqa: N802
        return self._platform

    def getMolecules(self) -> tuple[tuple[int, ...], ...]:  # noqa: N802
        return self._molecules

    # --- Step Rollout ---
    def step(self, steps: int) -> None:
        """Advance simulation rollout by `steps` steps."""
        if steps <= 0:
            return

        # Extract integrator properties in AKMA
        dt_akma = 2.0 * 1e-3 * PS_TO_TAU_AKMA  # default 2 fs
        if hasattr(self._integrator, "getStepSize"):
            dt_ps = self._integrator.getStepSize().value_in_unit(unit.picosecond)
            dt_akma = dt_ps * PS_TO_TAU_AKMA

        is_langevin = isinstance(
            self._integrator, (openmm.LangevinMiddleIntegrator, openmm.LangevinIntegrator)
        )
        temp_k = 300.0
        if hasattr(self._integrator, "getTemperature"):
            temp_k = self._integrator.getTemperature().value_in_unit(unit.kelvin)
        kT_akma = BOLTZMANN_KCAL * temp_k

        gamma_akma = 1.0 * TAU_AKMA_TO_PS  # default 1 ps^-1
        if hasattr(self._integrator, "getFriction"):
            gamma_ps = self._integrator.getFriction().value_in_unit(unit.picosecond**-1)
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

        remaining = steps
        chunk_size = self._config.max_steps_per_scan

        while remaining > 0:
            if has_barostat:
                steps_to_baro = baro_freq - (self._step_count % baro_freq)
                if steps_to_baro == 0:
                    steps_to_baro = baro_freq
                chunk = min(remaining, steps_to_baro, chunk_size)
            else:
                chunk = min(remaining, chunk_size)

            gp = self._get_dynamic_global_params()
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
                chunk,
                self._converted_system.has_pbc,
                self._converted_system.nonbonded_cutoff,
                self._config.remove_com_motion,
                self._constraints,
            )
            remaining -= chunk
            self._step_count += chunk
            self._time += chunk * dt_akma

            # Reconstruct virtual sites & apply constraints
            self.applyConstraints()

            # Attempt barostat volume move at frequency boundary
            if has_barostat and (self._step_count % baro_freq == 0):
                self._apply_barostat_step(baro_mode, baro_info, chunk * dt_akma, kT_akma)

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

    def _apply_mc_barostat(
        self,
        baro_info: dict[str, Any],
        kT_akma: float,
    ) -> None:
        """Apply isotropic Monte Carlo volume move with molecular COM scaling."""
        if self._box_vectors is None:
            return

        current_box = np.asarray(self._box_vectors, dtype=np.float64)
        vol_current = abs(float(np.linalg.det(current_box)))
        if vol_current <= 0:
            return

        target_p_akma = baro_info["pressure_akma"]
        n_mol = len(self._molecules) or self._n_particles

        delta_v = 0.05
        self._prng_key, subkey_scale, subkey_accept = jax.random.split(self._prng_key, 3)
        u_rand = float(jax.random.uniform(subkey_scale, minval=-1.0, maxval=1.0))
        scale = 1.0 + delta_v * u_rand
        if scale <= 0:
            return
        cbrt_scale = scale ** (1.0 / 3.0)

        new_box = current_box * cbrt_scale
        vol_new = vol_current * scale
        delta_v_val = vol_new - vol_current

        current_pos = np.asarray(self._positions, dtype=np.float64)
        new_pos = current_pos.copy()

        for mol in self._molecules:
            if not mol:
                continue
            mol_idx = list(mol)
            com = np.mean(current_pos[mol_idx], axis=0)
            offset = (cbrt_scale - 1.0) * com
            new_pos[mol_idx] += offset

        new_pos_jnp = jnp.asarray(new_pos, dtype=self._positions.dtype)
        if self._converted_system.virtual_sites is not None:
            new_pos_jnp = self._converted_system.compute_virtual_sites(new_pos_jnp)

        gp = self._get_dynamic_global_params()
        e_curr, _ = _pure_compute_energy_and_forces(
            self._positions,
            self._box_vectors,
            self._bundle,
            self._custom_forces,
            self._position_restraints,
            gp,
            self._converted_system.force_group_by_component,
            0xFFFFFFFF,
            self._converted_system.has_pbc,
            self._converted_system.nonbonded_cutoff,
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
            0xFFFFFFFF,
            self._converted_system.has_pbc,
            self._converted_system.nonbonded_cutoff,
        )

        delta_u = float(e_new - e_curr)
        delta_w = delta_u + target_p_akma * delta_v_val - float(n_mol) * kT_akma * np.log(scale)
        exponent = -delta_w / max(kT_akma, 1e-12)
        accept_prob = min(1.0, float(np.exp(min(exponent, 50.0))))
        rand_val = float(jax.random.uniform(subkey_accept))

        if rand_val < accept_prob:
            self._positions = new_pos_jnp
            self._box_vectors = new_box_jnp
            self._bundle = new_bundle
            self._converted_system = eqx.tree_at(
                lambda s: s.bundle.box, self._converted_system, new_box_jnp
            )
            self.applyConstraints()

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

        gp = self._get_dynamic_global_params()
        _, forces = _pure_compute_energy_and_forces(
            self._positions,
            self._box_vectors,
            self._bundle,
            self._custom_forces,
            self._position_restraints,
            gp,
            self._converted_system.force_group_by_component,
            0xFFFFFFFF,
            self._converted_system.has_pbc,
            self._converted_system.nonbonded_cutoff,
        )
        if self._converted_system.virtual_sites is not None:
            forces = self._converted_system.redistribute_virtual_site_forces(
                forces, self._positions
            )

        virial = float(jnp.sum(self._positions * forces))
        inst_pressure = (2.0 * ke_total + virial) / (3.0 * volume)

        tau = 2000.0  # AKMA time (~0.1 ps)
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
        self._positions = self._positions * mu
        self._velocities = self._velocities / mu
        self.applyConstraints()

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
        pos_akma = jnp.asarray(pos_arr, dtype=jnp.float32)
        if self._converted_system.virtual_sites is not None:
            pos_akma = self._converted_system.compute_virtual_sites(pos_akma)
        self._positions = pos_akma

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
        vel_akma = jnp.asarray(vel_arr, dtype=jnp.float32)
        if self._converted_system.virtual_sites is not None:
            vs_idx = self._converted_system.virtual_sites.order
            vel_akma = vel_akma.at[vs_idx].set(0.0)
        self._velocities = vel_akma

    def setVelocitiesToTemperature(  # noqa: N802
        self, temperature: Any, randomSeed: int | None = None
    ) -> None:
        """Assign Maxwell-Boltzmann thermal velocities accounting for constraints."""
        temp_k = (
            float(temperature.value_in_unit(unit.kelvin))
            if isinstance(temperature, unit.Quantity)
            else float(temperature)
        )
        seed = int(randomSeed) if randomSeed is not None else int(self._config.deterministic_seed)
        key = jax.random.PRNGKey(seed)

        masses = self._bundle.masses[: self._n_particles]
        safe_inv_m = self._converted_system.inv_mass[: self._n_particles]
        sigmas = jnp.sqrt(BOLTZMANN_KCAL * temp_k * safe_inv_m)[:, None]

        noise = jax.random.normal(key, shape=self._positions.shape, dtype=jnp.float32)
        vel = noise * sigmas

        vel = jnp.where(safe_inv_m[:, None] > 1e-12, vel, 0.0)

        if self._config.remove_com_motion:
            total_m = jnp.sum(masses)
            v_com = jnp.sum(vel * masses[:, None], axis=0) / jnp.maximum(total_m, 1e-12)
            vel = jnp.where(safe_inv_m[:, None] > 1e-12, vel - v_com, 0.0)

        self._velocities = vel
        self.applyVelocityConstraints(self._config.settle_velocity_tol)

    def setPeriodicBoxVectors(self, a: Any, b: Any, c: Any) -> None:  # noqa: N802
        """Set periodic box vectors in canonical lower-triangular form."""
        a_akma = np.asarray(to_akma(a, "length"), dtype=np.float64)
        b_akma = np.asarray(to_akma(b, "length"), dtype=np.float64)
        c_akma = np.asarray(to_akma(c, "length"), dtype=np.float64)
        box_mat = np.stack([a_akma, b_akma, c_akma])

        box_canonical = reduce_triclinic_box_vectors(jnp.asarray(box_mat, dtype=jnp.float32))
        self._box_vectors = box_canonical
        self._bundle = eqx.tree_at(lambda bnd: bnd.box, self._bundle, box_canonical)
        self._converted_system = eqx.tree_at(
            lambda s: s.bundle.box, self._converted_system, box_canonical
        )

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

        self._positions = jnp.asarray(pos_arr, dtype=jnp.float32)
        self._velocities = jnp.asarray(vel_arr, dtype=jnp.float32)
        bv = state_dict.get("box_vectors")
        self._box_vectors = jnp.asarray(bv, dtype=jnp.float32) if bv is not None else None
        self._time = float(state_dict["time"])
        self._step_count = int(state_dict["step_count"])
        self._parameters = dict(state_dict.get("parameters", {}))
        if "prng_key" in state_dict:
            self._prng_key = jnp.asarray(state_dict["prng_key"])

    def reinitialize(self, preserveState: bool = False) -> None:
        chk = self.createCheckpoint() if preserveState else None
        self._converted_system = from_openmm_system(
            self._system, self._integrator, self._config
        )
        self._bundle = self._converted_system.bundle
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

        for f_idx in range(self._system.getNumForces()):
            f = self._system.getForce(f_idx)
            if force is not None and not _is_same_openmm_force(f, force):
                continue

            if isinstance(f, openmm.NonbondedForce):
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

                bundle = eqx.tree_at(
                    lambda bnd: (bnd.charges, bnd.sigmas, bnd.epsilons),
                    bundle,
                    (
                        jnp.asarray(charges_np, dtype=jnp.float32),
                        jnp.asarray(sigmas_np, dtype=jnp.float32),
                        jnp.asarray(epsilons_np, dtype=jnp.float32),
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
                            jnp.asarray(exc_sig_np, dtype=jnp.float32),
                            jnp.asarray(exc_eps_np, dtype=jnp.float32),
                            jnp.asarray(exc_qp_np, dtype=jnp.float32),
                        ),
                    )

            elif isinstance(f, openmm.HarmonicBondForce):
                bond_map = meta.get("omm_forces_bond_to_bundle_idx", {}).get(f_idx, {})
                if bond_map and bundle.bond_params is not None:
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
                        jnp.asarray(bp_np, dtype=jnp.float32),
                    )

            elif isinstance(f, openmm.HarmonicAngleForce):
                angle_map = meta.get("omm_forces_angle_to_bundle_idx", {}).get(f_idx, {})
                if angle_map and bundle.angle_params is not None:
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
                        jnp.asarray(ap_np, dtype=jnp.float32),
                    )

            elif isinstance(f, openmm.PeriodicTorsionForce):
                torsion_map = meta.get("omm_forces_torsion_to_bundle_idx", {}).get(f_idx, {})
                if torsion_map and bundle.dihedral_params is not None:
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
                        jnp.asarray(dp_np, dtype=jnp.float32),
                    )

            elif isinstance(f, openmm.GBSAOBCForce):
                if bundle.radii is not None and bundle.scaled_radii is not None:
                    radii_np = np.asarray(bundle.radii).copy()
                    scaled_radii_np = np.asarray(bundle.scaled_radii).copy()
                    for p_i in range(f.getNumParticles()):
                        _, r, s = f.getParticleParameters(p_i)
                        r_a = r.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM
                        radii_np[p_i] = r_a
                        scaled_radii_np[p_i] = r_a * float(s)
                    bundle = eqx.tree_at(
                        lambda bnd: (bnd.radii, bnd.scaled_radii),
                        bundle,
                        (
                            jnp.asarray(radii_np, dtype=jnp.float32),
                            jnp.asarray(scaled_radii_np, dtype=jnp.float32),
                        ),
                    )

            elif f_idx in custom_force_map:
                cf_idx = custom_force_map[f_idx]
                cf = updated_cf[cf_idx]
                if isinstance(cf, CompiledCustomBondForce):
                    num_params = f.getNumPerBondParameters()
                    param_names = [
                        f.getPerBondParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_bond_params[p]).copy() for p in param_names
                    }
                    for b_i in range(f.getNumBonds()):
                        p1, p2, params = f.getBondParameters(b_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][b_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_bond_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomAngleForce):
                    num_params = f.getNumPerAngleParameters()
                    param_names = [
                        f.getPerAngleParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_angle_params[p]).copy() for p in param_names
                    }
                    for a_i in range(f.getNumAngles()):
                        p1, p2, p3, params = f.getAngleParameters(a_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][a_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_angle_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomTorsionForce):
                    num_params = f.getNumPerTorsionParameters()
                    param_names = [
                        f.getPerTorsionParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_torsion_params[p]).copy() for p in param_names
                    }
                    for t_i in range(f.getNumTorsions()):
                        p1, p2, p3, p4, params = f.getTorsionParameters(t_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][t_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_torsion_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomExternalForce):
                    num_params = f.getNumPerParticleParameters()
                    param_names = [
                        f.getPerParticleParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_particle_params[p]).copy() for p in param_names
                    }
                    for p_i in range(f.getNumParticles()):
                        _, params = f.getParticleParameters(p_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][p_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_particle_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomCompoundBondForce):
                    num_params = f.getNumPerBondParameters()
                    param_names = [
                        f.getPerBondParameterName(i) for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_bond_params[p]).copy() for p in param_names
                    }
                    for b_i in range(f.getNumBonds()):
                        _, params = f.getBondParameters(b_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][b_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_bond_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledCustomNonbondedForce):
                    num_params = f.getNumPerParticleParameters()
                    param_names = [
                        f.getPerParticleParameterName(i)
                        for i in range(num_params)
                    ]
                    new_params = {
                        p: np.asarray(cf.per_particle_params[p]).copy()
                        for p in param_names
                    }
                    for p_i in range(f.getNumParticles()):
                        params = f.getParticleParameters(p_i)
                        for p_idx, p_name in enumerate(param_names):
                            new_params[p_name][p_i] = params[p_idx]
                    updated_cf[cf_idx] = eqx.tree_at(
                        lambda c: c.per_particle_params,
                        cf,
                        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in new_params.items()},
                    )
                elif isinstance(cf, CompiledRBTorsionForce):
                    c_params_np = np.asarray(cf.c_params).copy()
                    for t_i in range(f.getNumTorsions()):
                        _, _, _, _, c0, c1, c2, c3, c4, c5 = f.getTorsionParameters(t_i)
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
        self._converted_system = eqx.tree_at(
            lambda s: (s.bundle, s.custom_forces),
            self._converted_system,
            (bundle, updated_cf),
        )

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
            groups_mask = sum(1 << g for g in groups if 0 <= g <= 31)
        else:
            groups_mask = int(groups) & 0xFFFFFFFF

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

        if req_energy:
            m = np.asarray(self._bundle.masses[: self._n_particles], dtype=np.float64)
            v = np.asarray(self._velocities[: self._n_particles], dtype=np.float64)
            ke_akma = 0.5 * np.sum(m[:, None] * (v**2))
            ke_kj = float(ke_akma * KCAL_PER_MOL_TO_KJ_PER_MOL)

        if req_energy or req_forces:
            gp = {
                k: jnp.asarray(val, dtype=jnp.float32)
                for k, val in self._parameters.items()
            }
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
            )
            if req_energy:
                pe_kj = float(pe_akma * KCAL_PER_MOL_TO_KJ_PER_MOL)
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
