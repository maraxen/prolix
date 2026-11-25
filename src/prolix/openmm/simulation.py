"""Drop-in openmm.app.Simulation and multi-replica ProlixBatchSimulation replacements."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import inspect
import json
import math
import os
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as _omm_unit
from jax_md import minimize as jax_md_minimize
from openmm import app

import prolix.api.ensemble_plan as _ensemble_plan_mod
from prolix.knob_surface import (
    BarostatControls,
    KnobSurface,
    NonbondedControls,
    PallasControls,
)
from prolix.openmm.config import ProlixOpenMMConfig, parse_platform_properties
from prolix.openmm.context import (
    BOLTZMANN_KCAL,
    ProlixContext,
    ProlixState,
    _apply_exception_offsets,
    _apply_particle_offsets,
    _eval_nonbonded_terms,
    _eval_system_potential_energy,
    _from_jax_or_dlpack,
    _is_reduced_triclinic_box_np,
    _minimum_image_dr,
    _pure_compute_energy_and_forces,
    _system_has_anisotropic_or_membrane_barostat,
    _triclinic_morton_kernels_supported,
    _wrap_molecules_jax,
)
from prolix.openmm.converter import (
    ConvertedOpenMMSystem,
    NonbondedParameterOffsets,
    PositionRestraints,
    from_openmm_system,
)
from prolix.openmm.units import (
    ANGSTROM_TO_NM,
    ATM_TO_BAR,
    FORCE_AKMA_TO_OMM,
    FORCE_OMM_TO_AKMA,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    NM_TO_ANGSTROM,
    PS_TO_TAU_AKMA,
    TAU_AKMA_TO_PS,
    VELOCITY_AKMA_TO_OMM,
    VELOCITY_OMM_TO_AKMA,
    to_akma,
)
from prolix.physics.constraints import ConstraintAlgorithm, make_constraint
from prolix.physics.lj_pme import resolve_ljpme_parameters
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
)

_jaxopt: Any = None
try:
    import jaxopt as _jaxopt_mod

    _jaxopt = _jaxopt_mod
except ImportError:
    pass

if TYPE_CHECKING:
    from openmm.app import Topology

unit = cast("Any", _omm_unit)

_ENSEMBLE_PLAN_RUN_PARAMS: frozenset[str] = frozenset(
    inspect.signature(_ensemble_plan_mod.EnsemblePlan.run).parameters
)
_SUPPORTED_STAGED_INCLUDES: frozenset[str] = frozenset(
    {"positions", "velocities", "forces", "energy"}
)


class _BatchVelocityObservable(eqx.Module):
    """Lightweight device-side Observable capturing post-step AKMA velocities."""

    def compute(self, state: Any) -> jnp.ndarray:
        mom = state.momentum
        mass = state.mass
        if mass.ndim == 1:
            mass = mass[:, None]
        safe_m = jnp.maximum(mass, 1e-12)
        return jnp.where(mass > 1e-12, mom / safe_m, jnp.zeros_like(mom))


class _BatchParameterDerivatives(list[dict[str, float]]):
    """Per-replica `{param: dU/dparam}` list also supporting `derivs['param'] -> (B,) np.ndarray`."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return np.asarray([float(d.get(key, 0.0)) for d in self], dtype=np.float64)
        return super().__getitem__(key)


@eqx.filter_jit
def _compute_ke_kj_device(
    vel_akma: jnp.ndarray,
    masses: jnp.ndarray,
    inv_mass: jnp.ndarray,
    forces_akma: jnp.ndarray | None,
    dt_akma: float,
    sync_leapfrog: bool,
) -> jnp.ndarray:
    """Device-side kinetic energy in kJ/mol without host synchronization."""
    v = vel_akma
    if sync_leapfrog and forces_akma is not None and dt_akma > 0.0:
        v = v + 0.5 * float(dt_akma) * (forces_akma * inv_mass[:, None])
    ke_akma = 0.5 * jnp.sum(masses[:, None] * (v * v))
    return ke_akma * float(KCAL_PER_MOL_TO_KJ_PER_MOL)


class _SimStepProxy:
    """Lightweight Simulation proxy presenting a historical `currentStep` during replay."""

    __slots__ = ("_sim", "_step")

    def __init__(self, sim: ProlixSimulation, step: int) -> None:
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_step", int(step))

    @property
    def currentStep(self) -> int:  # noqa: N802
        return int(object.__getattribute__(self, "_step"))

    @currentStep.setter
    def currentStep(self, step: int) -> None:  # noqa: N802
        sim = object.__getattribute__(self, "_sim")
        sim.currentStep = int(step)
        object.__setattr__(self, "_step", int(step))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_sim"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("currentStep", "_sim", "_step"):
            object.__setattr__(self, name, value) if name != "currentStep" else setattr(
                object.__getattribute__(self, "_sim"), "currentStep", value
            )
            return
        setattr(object.__getattribute__(self, "_sim"), name, value)


def _normalize_report_desc(raw_report: Any) -> dict[str, Any]:
    """Normalize OpenMM tuple or dict `describeNextReport` output into canonical dict."""
    if isinstance(raw_report, tuple):
        report_dict: dict[str, Any] = {"steps": int(raw_report[0])}
        if len(raw_report) > 5:
            report_dict["periodic"] = raw_report[5]
        includes = ["positions", "velocities", "forces", "energy"]
        report_dict["include"] = [
            includes[j] for j in range(4) if bool(raw_report[j + 1])
        ]
        return report_dict
    out = dict(raw_report)
    out["steps"] = int(out["steps"])
    out["include"] = list(out.get("include", ()))
    return out


def _context_mutation_fingerprint(
    ctx: ProlixContext,
) -> tuple[int, dict[str, float], int, int, int]:
    """Capture context mutation fingerprint without editing `context.py`."""
    return (
        int(ctx._step_count),
        dict(ctx._parameters),
        id(ctx._positions),
        id(ctx._velocities),
        id(ctx._box_vectors),
    )


class ProlixSimulation(app.Simulation):
    """Drop-in replacement for openmm.app.Simulation powered by Prolix JAX backend.

    Features:

    - Fused Rollout Bypass & Multi-Interval Device Ring-Buffer Staging: Dispatches full
      reporter intervals directly to `context.step(nextSteps)` and batches up to `K`
      constant-cadence report snapshots on device before a single bulk `jax.device_get`.
    - Energy Minimization: Routes to GPU L-BFGS (jaxopt), native FIRE, or OpenMM CPU fallback.
    """

    def __init__(
        self,
        topology: Topology,
        system: openmm.System | str,
        integrator: openmm.Integrator | str,
        platform: openmm.Platform | None = None,
        platformProperties: dict[str, str] | None = None,
        state: Any = None,
        config: ProlixOpenMMConfig | None = None,
    ) -> None:
        omm_mod = cast("Any", openmm)
        top_any = cast("Any", topology)
        self.topology = topology

        # Resolve system if string
        if isinstance(system, str):
            self.system = omm_mod.XmlSerializer.deserialize(system)
        else:
            self.system = system

        # Resolve integrator if string
        if isinstance(integrator, str):
            self.integrator = omm_mod.XmlSerializer.deserialize(integrator)
        else:
            self.integrator = integrator

        self.reporters: list[Any] = []

        # Parse configuration
        if config is not None:
            self.config = config
        elif platformProperties is not None:
            self.config = parse_platform_properties(platformProperties)
        else:
            self.config = ProlixOpenMMConfig()

        # Instantiate ProlixContext
        self.context: ProlixContext = ProlixContext(
            system=self.system,
            integrator=self.integrator,
            platform=platform,
            platformProperties=platformProperties,
            config=self.config,
        )

        # Synchronize topology box vectors from system if topology does not have them
        if (
            hasattr(top_any, "getPeriodicBoxVectors")
            and top_any.getPeriodicBoxVectors() is None
        ):
            with contextlib.suppress(Exception):
                raw_bv = self.system.getDefaultPeriodicBoxVectors()
                if raw_bv is not None:
                    top_any.setPeriodicBoxVectors(raw_bv)

        if state is not None:
            if isinstance(state, str):
                self.context.setState(omm_mod.XmlSerializer.deserialize(state))
            else:
                self.context.setState(state)

        try:
            self._usesPBC = self.system.usesPeriodicBoundaryConditions()
        except Exception:
            self._usesPBC = top_any.getUnitCellDimensions() is not None

    @property
    def currentStep(self) -> int:  # noqa: N802
        return self.context.getStepCount()

    @currentStep.setter
    def currentStep(self, step: int) -> None:  # noqa: N802
        self.context.setStepCount(step)

    def step(self, steps: int) -> None:
        """Advance simulation by integrating a specified number of steps."""
        self._simulate(endStep=self.currentStep + steps)

    def _plan_staged_report_batch(  # noqa: PLR0911
        self,
        start_step: int,
        end_step: int,
    ) -> list[tuple[int, int, list[tuple[Any, dict[str, Any]]], list[tuple[Any, dict[str, Any]]]]] | None:
        """Plan up to `K` constant-cadence report intervals for device ring-buffer staging.

        Returns `None` if any active reporter uses non-constant/non-positive step cadences,
        unsupported state requests, or mutates context state during `describeNextReport`.
        """
        if not self.reporters or end_step >= sys.maxsize:
            return None

        ctx = self.context
        n_particles = int(ctx._n_particles)
        k_cap = max(1, min(32, 64_000_000 // max(1, n_particles * 12)))
        fp0 = _context_mutation_fingerprint(ctx)

        # Verify constant positive cadence per reporter before staging multiple intervals
        cadences: list[int | None] = []
        for rep in self.reporters:
            if isinstance(rep, (app.StateDataReporter, app.DCDReporter, app.PDBReporter)):
                iv = int(getattr(rep, "_reportInterval", 0))
                if iv <= 0:
                    return None
                cadences.append(iv)
                continue
            probe0 = _normalize_report_desc(
                rep.describeNextReport(_SimStepProxy(self, start_step))
            )
            if _context_mutation_fingerprint(ctx) != fp0:
                return None
            s0 = int(probe0["steps"])
            if s0 <= 0:
                cadences.append(None)
                continue
            if not set(probe0["include"]).issubset(_SUPPORTED_STAGED_INCLUDES):
                return None
            # Probe at first fire step `t1 = start_step + s0` and `t1 + iv_cand` to confirm
            # constant positive cadence `iv_cand` without assuming attribute names.
            t1 = start_step + s0
            probe1 = _normalize_report_desc(
                rep.describeNextReport(_SimStepProxy(self, t1))
            )
            if _context_mutation_fingerprint(ctx) != fp0:
                return None
            iv_cand = int(probe1["steps"])
            if iv_cand <= 0 or not set(probe1["include"]).issubset(_SUPPORTED_STAGED_INCLUDES):
                return None
            probe2 = _normalize_report_desc(
                rep.describeNextReport(_SimStepProxy(self, t1 + iv_cand))
            )
            if _context_mutation_fingerprint(ctx) != fp0:
                return None
            if int(probe2["steps"]) != iv_cand or probe2["include"] != probe1["include"]:
                return None
            cadences.append(iv_cand)

        schedule: list[
            tuple[int, int, list[tuple[Any, dict[str, Any]]], list[tuple[Any, dict[str, Any]]]]
        ] = []
        cur = start_step
        while cur < end_step and len(schedule) < k_cap:
            next_steps = end_step - cur
            any_due = False
            descs: list[dict[str, Any] | None] = [None] * len(self.reporters)
            proxy_cur = _SimStepProxy(self, cur)
            for i, rep in enumerate(self.reporters):
                if cadences[i] is None:
                    continue
                d = _normalize_report_desc(rep.describeNextReport(proxy_cur))
                if _context_mutation_fingerprint(ctx) != fp0:
                    return None
                if not set(d["include"]).issubset(_SUPPORTED_STAGED_INCLUDES):
                    return None
                descs[i] = d
                st_rem = int(d["steps"])
                if 0 < st_rem <= next_steps:
                    next_steps = st_rem
                    any_due = True

            if not any_due or cur + next_steps > end_step:
                break

            target_step = cur + next_steps
            wrapped: list[tuple[Any, dict[str, Any]]] = []
            unwrapped: list[tuple[Any, dict[str, Any]]] = []
            either: list[tuple[Any, dict[str, Any]]] = []
            for rep, d in zip(self.reporters, descs, strict=False):
                if d is not None and int(d["steps"]) == next_steps:
                    want_wrap = (
                        self._usesPBC
                        if d.get("periodic") is None
                        else bool(d["periodic"])
                    )
                    if "positions" not in d["include"]:
                        either.append((rep, d))
                    elif want_wrap:
                        wrapped.append((rep, d))
                    else:
                        unwrapped.append((rep, d))
            if len(wrapped) > len(unwrapped):
                wrapped += either
            else:
                unwrapped += either

            schedule.append((next_steps, target_step, wrapped, unwrapped))
            cur = target_step

        return schedule if len(schedule) >= 2 else None

    def _capture_device_report_snapshot(
        self,
        *,
        need_wrapped: bool,
        need_unwrapped: bool,
        req_pos_w: bool,
        req_vel: bool,
        req_forces: bool,
        req_energy: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Queue asynchronous device ops for a single staged report snapshot (zero D2H sync)."""
        ctx = self.context
        n_p = int(ctx._n_particles)
        raw_pos_d = ctx._positions[:n_p]
        raw_vel_d = ctx._velocities[:n_p]

        has_box = bool(
            ctx._converted_system.has_pbc
            or ctx._morton_box_diag is not None
            or ctx._box_vectors is not None
        )
        if ctx._morton_box_diag is not None:
            box_akma_d: jnp.ndarray | None = (
                ctx._morton_box_diag.astype(raw_pos_d.dtype)
                if ctx._morton_box_diag.ndim == 2
                else jnp.diag(ctx._morton_box_diag.astype(raw_pos_d.dtype))
            )
        elif ctx._box_vectors is not None:
            box_akma_d = ctx._box_vectors.astype(raw_pos_d.dtype)
        else:
            box_akma_d = None

        pos_wrapped_d: jnp.ndarray | None = None
        if need_wrapped and req_pos_w and has_box and box_akma_d is not None:
            pos_wrapped_d = _wrap_molecules_jax(
                raw_pos_d,
                box_akma_d,
                ctx._mol_of_atom_jnp,
                ctx._inv_mol_size_jnp,
            )

        scalars_kj_d: jnp.ndarray | None = None
        pe_akma_d: jnp.ndarray | None = None
        ke_kj_d: jnp.ndarray | None = None
        forces_omm_d: jnp.ndarray | None = None
        forces_akma_d: jnp.ndarray | None = None

        ig_groups = ctx.getIntegrator().getIntegrationForceGroups()
        groups_mask = int(np.int32(int(ig_groups)))

        if req_energy or req_forces:
            use_disp = bool(
                ctx._config.use_dispersion_correction
                and ctx._converted_system.use_dispersion_correction
            )
            if groups_mask == -1 and ctx._can_use_morton_obs():
                runner = ctx._ensure_morton_runner()
                bd = ctx._morton_bonded_dyn or runner["bonded_static"]
                n_act = int(runner["n_act"])
                if (
                    req_forces
                    and not req_energy
                    and not ctx._morton_nl_dirty
                    and not ctx._morton_forces_dirty
                    and ctx._morton_tp_state is not None
                ):
                    forces_omm_d = runner["jit_unpermute_forces"](
                        ctx._morton_fx,
                        ctx._morton_fy,
                        ctx._morton_fz,
                        ctx._morton_tp_state.inv_perm,
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
                    ) = ctx._ensure_morton_params(runner, bd, use_disp)
                    pr_bufs = ctx._ensure_restraint_params()
                    gp_vec = ctx._ensure_gp_vec(runner["param_names"])
                    box_diag_in = ctx._morton_box_diag
                    if box_diag_in is None and ctx._box_vectors is not None:
                        box_diag_in = (
                            (
                                ctx._box_vectors
                                if ctx._box_vectors.ndim == 2
                                else jnp.diag(ctx._box_vectors)
                            )
                            if ctx._is_triclinic
                            else (
                                jnp.diag(ctx._box_vectors)
                                if ctx._box_vectors.ndim == 2
                                else ctx._box_vectors
                            )
                        )
                        ctx._morton_box_diag = box_diag_in
                    (
                        scalars_kj_d,
                        f_omm_dev,
                        tp_out,
                        fx_out,
                        fy_out,
                        fz_out,
                        G_out,
                    ) = runner["jit_energy_and_forces"](
                        ctx._positions[:n_act],
                        ctx._velocities[:n_act],
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
                        ctx._morton_tp_state,
                        ctx._morton_G,
                        bool(ctx._morton_nl_dirty),
                        bool(ctx._morton_forces_dirty),
                        c_tail_s,
                        pr_bufs,
                        ctx._custom_forces,
                        gp_vec,
                    )
                    ctx._morton_tp_state = tp_out
                    ctx._morton_fx = fx_out
                    ctx._morton_fy = fy_out
                    ctx._morton_fz = fz_out
                    ctx._morton_G = G_out
                    ctx._morton_nl_dirty = False
                    ctx._morton_forces_dirty = False
                    if req_forces:
                        forces_omm_d = f_omm_dev
            else:
                ctx._ensure_host_nl_fresh()
                gp = ctx._get_dynamic_global_params()
                ewald_tol = (
                    ctx._converted_system.ewald_error_tolerance
                    if ctx._converted_system.ewald_error_tolerance is not None
                    else ctx._config.ewald_error_tolerance
                )
                pe_akma_d, f_akma_raw = _pure_compute_energy_and_forces(
                    ctx._positions,
                    ctx._box_vectors,
                    ctx._bundle,
                    ctx._custom_forces,
                    ctx._position_restraints,
                    gp,
                    ctx._converted_system.force_group_by_component,
                    groups_mask,
                    ctx._converted_system.has_pbc,
                    ctx._converted_system.nonbonded_cutoff,
                    ctx._converted_system.bond_force_groups,
                    ctx._converted_system.angle_force_groups,
                    ctx._converted_system.dihedral_force_groups,
                    ctx._converted_system.cmap_force_groups,
                    ctx._effective_nonbonded_method(),
                    ctx._converted_system.has_nonbonded,
                    use_disp,
                    ewald_tol,
                    ctx._converted_system.reaction_field_dielectric,
                    ctx._converted_system.has_implicit_solvent,
                    ctx._converted_system.solvent_dielectric,
                    ctx._converted_system.solute_dielectric,
                    ctx._nonbonded_engine,
                    ctx._converted_system.pme_grid_dims,
                    use_pallas=ctx._config.use_pallas,
                    pallas_interpret=ctx._config.pallas_interpret,
                    lj_switch_width=ctx._converted_system.lj_switch_width,
                    neighbor_idx=getattr(ctx, "_neighbor_idx", None),
                    use_pallas_pme=ctx._config.use_pallas_pme,
                    custom_nonbonded_chunk_size=ctx._config.custom_nonbonded_chunk_size,
                    tile_pair_nl=ctx._tile_pair_nl,
                    gbsa_tile_list=ctx._gbsa_tile_list,
                    is_triclinic=ctx._is_triclinic,
                    mixed_precision=ctx._mixed_precision,
                    gb_use_neighbor_list=ctx._gb_use_neighbor_list,
                    gbsa_surface_tension=ctx._converted_system.gbsa_surface_tension,
                    pme_alpha=ctx._effective_pme_alpha(),
                    nonbonded_offsets=ctx._converted_system.nonbonded_offsets,
                    default_params=ctx._converted_system.default_parameters,
                    lj_combining_rule=ctx._lj_combining_rule,
                    exceptions_use_pbc=ctx._exceptions_use_pbc,
                    dispersion_follows_parameters=ctx._dispersion_follows_parameters,
                    lj_pme_alpha=ctx._effective_lj_pme_alpha(),
                    lj_pme_grid_dims=_resolve_cs_lj_pme_grid_dims(
                        ctx._converted_system, ctx._config, ctx._box_vectors
                    ),
                )
                if req_energy:
                    eff_scheme = ctx._effective_integrator_scheme()
                    ig_state = cast("Any", ctx._integrator)
                    dt_omm = (
                        ig_state.getStepSize()
                        if hasattr(ig_state, "getStepSize")
                        else 0.001 * unit.picosecond
                    )
                    dt_akma = float(to_akma(dt_omm, "time"))
                    sync_lf = bool(
                        eff_scheme == "leapfrog"
                        and (ctx._step_count > 0 or ctx._velocities_initialized)
                        and dt_akma > 0.0
                    )
                    ke_kj_d = _compute_ke_kj_device(
                        raw_vel_d,
                        ctx._bundle.masses[:n_p].astype(raw_vel_d.dtype),
                        ctx._converted_system.inv_mass[:n_p].astype(raw_vel_d.dtype),
                        f_akma_raw[:n_p] if sync_lf else None,
                        dt_akma,
                        sync_lf,
                    )
                if req_forces:
                    if ctx._converted_system.virtual_sites is not None:
                        f_akma_raw = ctx._converted_system.redistribute_virtual_site_forces(
                            f_akma_raw, ctx._positions
                        )
                    forces_akma_d = f_akma_raw

        dev_tensors: dict[str, Any] = {
            "raw_pos": raw_pos_d,
            "raw_vel": raw_vel_d,
            "box_akma": box_akma_d,
            "pos_wrapped": pos_wrapped_d,
            "scalars_kj": scalars_kj_d,
            "pe_akma": pe_akma_d,
            "ke_kj": ke_kj_d,
            "forces_omm": forces_omm_d,
            "forces_akma": forces_akma_d,
        }
        host_meta: dict[str, Any] = {
            "time_ps": float(ctx._time * TAU_AKMA_TO_PS),
            "time_akma": float(ctx._time),
            "step_count": int(ctx._step_count),
            "parameters": dict(ctx._parameters),
            "has_box": has_box,
            "is_triclinic": bool(ctx._is_triclinic),
            "need_wrapped": need_wrapped,
            "need_unwrapped": need_unwrapped,
            "req_vel": req_vel,
            "device_state": {
                "positions": ctx._positions,
                "velocities": ctx._velocities,
                "prng_key": ctx._prng_key,
                "box_vectors": ctx._box_vectors,
                "morton_box_diag": ctx._morton_box_diag,
                "morton_mc_scale": ctx._morton_mc_scale,
                "morton_mc_att": ctx._morton_mc_att,
                "morton_mc_acc": ctx._morton_mc_acc,
            },
        }
        return dev_tensors, host_meta

    @staticmethod
    def _build_prolix_state_from_host_snap(
        ctx: ProlixContext,
        host_dev: dict[str, Any],
        meta: dict[str, Any],
        includes: set[str],
        *,
        enforce_pbc: bool,
    ) -> ProlixState:
        """Materialize `ProlixState` on CPU from a bulk `jax.device_get` snapshot."""
        req_pos = "positions" in includes
        req_vel = "velocities" in includes
        req_forces = "forces" in includes
        req_energy = "energy" in includes

        data_types = int(openmm.State.Parameters)
        if req_pos:
            data_types |= int(openmm.State.Positions)
        if req_vel:
            data_types |= int(openmm.State.Velocities)
        if req_forces:
            data_types |= int(openmm.State.Forces)
        if req_energy:
            data_types |= int(openmm.State.Energy)

        if host_dev["box_akma"] is not None:
            box_nm = np.asarray(host_dev["box_akma"], dtype=np.float64) * ANGSTROM_TO_NM
        else:
            box_nm = np.eye(3, dtype=np.float64) * 10.0

        pos_nm: np.ndarray | None = None
        if req_pos:
            if enforce_pbc and meta["has_box"]:
                if host_dev["pos_wrapped"] is not None:
                    pos_nm = np.asarray(host_dev["pos_wrapped"], dtype=np.float64) * ANGSTROM_TO_NM
                elif meta["is_triclinic"] and host_dev["box_akma"] is not None:
                    raw_pos_f64 = np.asarray(host_dev["raw_pos"], dtype=np.float64)
                    # Vectorized CSR float64 wrap using frame k's box
                    n_mols = int(ctx._inv_mol_size_np.shape[0])
                    if n_mols > 0 and raw_pos_f64.shape[0] > 0:
                        bx = np.asarray(host_dev["box_akma"], dtype=np.float64)
                        a_v, b_v, c_v = bx[0], bx[1], bx[2]
                        m_ids = ctx._mol_of_atom_np
                        cx = np.bincount(m_ids, weights=raw_pos_f64[:, 0], minlength=n_mols) * ctx._inv_mol_size_np
                        cy = np.bincount(m_ids, weights=raw_pos_f64[:, 1], minlength=n_mols) * ctx._inv_mol_size_np
                        cz = np.bincount(m_ids, weights=raw_pos_f64[:, 2], minlength=n_mols) * ctx._inv_mol_size_np
                        diffs = np.zeros((n_mols, 3), dtype=np.float64)
                        if abs(c_v[2]) > 1e-12:
                            diffs += np.floor(cz / c_v[2])[:, None] * c_v[None, :]
                        if abs(b_v[1]) > 1e-12:
                            diffs += np.floor((cy - diffs[:, 1]) / b_v[1])[:, None] * b_v[None, :]
                        if abs(a_v[0]) > 1e-12:
                            diffs += np.floor((cx - diffs[:, 0]) / a_v[0])[:, None] * a_v[None, :]
                        pos_nm = (raw_pos_f64 - diffs[m_ids]) * ANGSTROM_TO_NM
                    else:
                        pos_nm = raw_pos_f64 * ANGSTROM_TO_NM
                else:
                    pos_nm = np.asarray(host_dev["raw_pos"], dtype=np.float64) * ANGSTROM_TO_NM
            else:
                pos_nm = np.asarray(host_dev["raw_pos"], dtype=np.float64) * ANGSTROM_TO_NM

        vel_omm: np.ndarray | None = None
        if req_vel:
            vel_omm = np.asarray(host_dev["raw_vel"], dtype=np.float64) * VELOCITY_AKMA_TO_OMM

        pe_kj: float | None = None
        ke_kj: float | None = None
        if req_energy:
            if host_dev["scalars_kj"] is not None:
                sc_np = np.asarray(host_dev["scalars_kj"], dtype=np.float64)
                pe_kj = float(sc_np[0])
                ke_kj = float(sc_np[1])
            else:
                pe_kj = float(np.asarray(host_dev["pe_akma"], dtype=np.float64)) * KCAL_PER_MOL_TO_KJ_PER_MOL
                ke_kj = float(np.asarray(host_dev["ke_kj"], dtype=np.float64))

        forces_omm: np.ndarray | None = None
        if req_forces:
            if host_dev["forces_omm"] is not None:
                forces_omm = np.asarray(host_dev["forces_omm"], dtype=np.float64)
            else:
                forces_omm = np.asarray(host_dev["forces_akma"], dtype=np.float64) * FORCE_AKMA_TO_OMM

        return ProlixState(
            positions_nm=pos_nm,
            velocities_omm=vel_omm,
            forces_omm=forces_omm,
            box_vectors_nm=box_nm,
            time_ps=meta["time_ps"],
            step_count=meta["step_count"],
            pe_kj=pe_kj,
            ke_kj=ke_kj,
            parameters=meta["parameters"],
            param_derivatives=None,
            data_types=data_types,
        )

    def _restore_unmutated_context_at_frame(
        self,
        fp_before: tuple[int, dict[str, float], int, int, int],
        meta_k: dict[str, Any],
    ) -> None:
        """Rewind unmutated context components (including PRNGKey) to `step_k` when mutated mid-batch."""
        ctx = self.context
        fp_after = _context_mutation_fingerprint(ctx)
        dev_st = meta_k["device_state"]
        ctx._prng_key = dev_st["prng_key"]
        ctx._morton_mc_scale = dev_st["morton_mc_scale"]
        ctx._morton_mc_att = dev_st["morton_mc_att"]
        ctx._morton_mc_acc = dev_st["morton_mc_acc"]
        if fp_after[0] == fp_before[0]:
            ctx._step_count = int(meta_k["step_count"])
            ctx._time = float(meta_k["time_akma"])
        if fp_after[4] == fp_before[4]:
            ctx._box_vectors = dev_st["box_vectors"]
            ctx._morton_box_diag = dev_st["morton_box_diag"]
            ctx._box_nm_dirty = True
            ctx._bundle_box_dirty = True
        if fp_after[2] == fp_before[2]:
            ctx._positions = dev_st["positions"]
        if fp_after[3] == fp_before[3]:
            ctx._velocities = dev_st["velocities"]
        ctx._morton_nl_dirty = True
        ctx._morton_forces_dirty = True
        ctx._host_nl_dirty = True

    def _simulate(
        self,
        endStep: int | None = None,
        endTime: datetime.datetime | None = None,
    ) -> None:
        """Fused rollout bypass with multi-interval device ring-buffer reporter staging."""
        if endStep is None:
            endStep = sys.maxsize

        while self.currentStep < endStep and (
            endTime is None or datetime.datetime.now() < endTime  # noqa: DTZ005
        ):
            if endTime is None and self.reporters:
                staged_plan = self._plan_staged_report_batch(self.currentStep, endStep)
                if staged_plan is not None:
                    dev_snaps: list[dict[str, Any]] = []
                    metas: list[dict[str, Any]] = []
                    for steps_k, _target_k, wrapped_k, unwrapped_k in staged_plan:
                        self.context.step(steps_k)
                        inc_w = (
                            set().union(*(r[1]["include"] for r in wrapped_k))
                            if wrapped_k
                            else set()
                        )
                        inc_u = (
                            set().union(*(r[1]["include"] for r in unwrapped_k))
                            if unwrapped_k
                            else set()
                        )
                        inc_all = inc_w | inc_u
                        dev_t, meta_t = self._capture_device_report_snapshot(
                            need_wrapped=bool(wrapped_k),
                            need_unwrapped=bool(unwrapped_k),
                            req_pos_w=("positions" in inc_w),
                            req_vel=("velocities" in inc_all),
                            req_forces=("forces" in inc_all),
                            req_energy=("energy" in inc_all),
                        )
                        meta_t["inc_w"] = inc_w
                        meta_t["inc_u"] = inc_u
                        dev_snaps.append(dev_t)
                        metas.append(meta_t)

                    # Single bulk device-to-host transfer across all K staged intervals
                    host_snaps: list[dict[str, Any]] = jax.device_get(dev_snaps)
                    fp_before_replay = _context_mutation_fingerprint(self.context)
                    n_staged = len(staged_plan)

                    for k_idx, (
                        (_steps_k, target_k, wrapped_k, unwrapped_k),
                        host_dev_k,
                        meta_k,
                    ) in enumerate(zip(staged_plan, host_snaps, metas, strict=True)):
                        proxy_k = _SimStepProxy(self, target_k)
                        if wrapped_k:
                            state_w = self._build_prolix_state_from_host_snap(
                                self.context,
                                host_dev_k,
                                meta_k,
                                meta_k["inc_w"],
                                enforce_pbc=True,
                            )
                            for rep, _desc in wrapped_k:
                                rep.report(proxy_k, state_w)
                        if unwrapped_k:
                            state_u = self._build_prolix_state_from_host_snap(
                                self.context,
                                host_dev_k,
                                meta_k,
                                meta_k["inc_u"],
                                enforce_pbc=False,
                            )
                            for rep, _desc in unwrapped_k:
                                rep.report(proxy_k, state_u)

                        if _context_mutation_fingerprint(self.context) != fp_before_replay:
                            if k_idx < n_staged - 1:
                                self._restore_unmutated_context_at_frame(
                                    fp_before_replay, meta_k
                                )
                            break
                    continue

            next_report: list[dict[str, Any] | None] = [None] * len(self.reporters)
            next_steps = endStep - self.currentStep
            any_report = False

            # Query all reporters to determine earliest required report step
            for i, reporter in enumerate(self.reporters):
                report_dict = _normalize_report_desc(reporter.describeNextReport(self))
                next_report[i] = report_dict

                steps = report_dict["steps"]
                if 0 < steps <= next_steps:
                    next_steps = steps
                    any_report = True

            # Fused GPU Rollout: dispatch full interval directly without 10-step slicing
            steps_to_run = min(next_steps, endStep - self.currentStep)
            if steps_to_run > 0:
                self.context.step(steps_to_run)

            # Generate reports if due
            if any_report:
                wrapped = []
                unwrapped = []
                either = []
                for reporter, report in zip(self.reporters, next_report, strict=False):
                    if report is not None and report["steps"] == next_steps:
                        want_wrap = (
                            self._usesPBC
                            if report.get("periodic") is None
                            else report["periodic"]
                        )
                        if "positions" not in report["include"]:
                            either.append((reporter, report))
                        elif want_wrap:
                            wrapped.append((reporter, report))
                        else:
                            unwrapped.append((reporter, report))

                if len(wrapped) > len(unwrapped):
                    wrapped += either
                else:
                    unwrapped += either

                sim_any = cast("Any", self)
                if len(wrapped) > 0:
                    sim_any._generate_reports(wrapped, True)
                if len(unwrapped) > 0:
                    sim_any._generate_reports(unwrapped, False)

    def minimizeEnergy(  # noqa: N802
        self,
        tolerance: Any = 10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations: int = 0,
        reporter: Any = None,
    ) -> None:
        """Perform local energy minimization via configured minimizer backend."""
        # Convert tolerance to kJ/(mol*nm) and kcal/(mol*Å)
        if hasattr(tolerance, "value_in_unit"):
            tol_omm = tolerance.value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        else:
            tol_omm = float(tolerance)
        tol_akma = tol_omm * FORCE_OMM_TO_AKMA

        backend = self.config.minimizer_backend

        if backend == "lbfgs":
            self._minimize_lbfgs(tol_akma, maxIterations, reporter)
        elif backend == "fire":
            self._minimize_fire(tol_akma, maxIterations, reporter)
        elif backend == "openmm_cpu":
            self._minimize_openmm_cpu(tol_omm, maxIterations, reporter)
        else:
            raise ValueError(f"Unknown minimizer backend: {backend!r}")

    def getPositionsJax(  # noqa: N802
        self,
        *,
        in_nm: bool = True,
        enforcePeriodicBox: bool = False,
    ) -> jax.Array:
        """Return zero-copy device `jax.Array` of particle positions `(N, 3)` in `nm` (default) or `Å`."""
        return self.context.getPositionsJax(
            in_nm=in_nm, enforcePeriodicBox=enforcePeriodicBox
        )

    def setPositionsJax(  # noqa: N802
        self,
        positions: Any,
        *,
        in_nm: bool = True,
    ) -> None:
        """Set particle positions from a `jax.Array` or DLPack (`__dlpack__`) tensor without host copy."""
        self.context.setPositionsJax(positions, in_nm=in_nm)

    def getVelocitiesJax(  # noqa: N802
        self,
        *,
        in_nm_per_ps: bool = True,
    ) -> jax.Array:
        """Return zero-copy device `jax.Array` of particle velocities `(N, 3)` in `nm/ps` (default) or `Å/τ_AKMA`."""
        return self.context.getVelocitiesJax(in_nm_per_ps=in_nm_per_ps)

    def setVelocitiesJax(  # noqa: N802
        self,
        velocities: Any,
        *,
        in_nm_per_ps: bool = True,
    ) -> None:
        """Set particle velocities from a `jax.Array` or DLPack (`__dlpack__`) tensor without host copy."""
        self.context.setVelocitiesJax(velocities, in_nm_per_ps=in_nm_per_ps)

    def getForcesJax(  # noqa: N802
        self,
        *,
        in_kj_per_mol_nm: bool = True,
        groups: int | set[int] = -1,
    ) -> jax.Array:
        """Return device `jax.Array` of particle forces `(N, 3)` in `kJ/(mol·nm)` (default) or `kcal/(mol·Å)`."""
        return self.context.getForcesJax(
            in_kj_per_mol_nm=in_kj_per_mol_nm, groups=groups
        )

    def _minimize_lbfgs(
        self, tol_akma: float, max_iter: int, reporter: Any = None  # noqa: ARG002
    ) -> None:
        """Energy minimization using GPU jaxopt.LBFGS."""
        if _jaxopt is None:
            raise ImportError("jaxopt is required for minimizer_backend='lbfgs'.")

        ctx = self.context
        ctx.computeVirtualSites()
        ctx.applyConstraints()
        pos_init = ctx._positions
        max_iterations = max_iter if max_iter > 0 else 500

        gp = ctx._get_dynamic_global_params()
        ewald_tol = (
            ctx._converted_system.ewald_error_tolerance
            if ctx._converted_system.ewald_error_tolerance is not None
            else ctx._config.ewald_error_tolerance
        )
        eff_nb_method = ctx._effective_nonbonded_method()
        eff_pme_alpha = ctx._effective_pme_alpha()
        eff_lj_pme_alpha = ctx._effective_lj_pme_alpha()
        eff_lj_pme_dims = _resolve_cs_lj_pme_grid_dims(
            ctx._converted_system, ctx._config, ctx._box_vectors
        )

        def energy_fn(p: jnp.ndarray) -> jnp.ndarray:
            p_eval = ctx._converted_system.compute_virtual_sites(p)
            e_sys = _eval_system_potential_energy(
                p_eval,
                ctx._box_vectors,
                ctx._bundle,
                ctx._custom_forces,
                ctx._position_restraints,
                gp,
                ctx._converted_system.force_group_by_component,
                -1,
                ctx._converted_system.has_pbc,
                ctx._converted_system.nonbonded_cutoff,
                bond_force_groups=ctx._converted_system.bond_force_groups,
                angle_force_groups=ctx._converted_system.angle_force_groups,
                dihedral_force_groups=ctx._converted_system.dihedral_force_groups,
                cmap_force_groups=ctx._converted_system.cmap_force_groups,
                nonbonded_method=eff_nb_method,
                has_nonbonded=ctx._converted_system.has_nonbonded,
                use_dispersion_correction=ctx._config.use_dispersion_correction
                and ctx._converted_system.use_dispersion_correction,
                ewald_error_tolerance=ewald_tol,
                rf_dielectric=ctx._converted_system.reaction_field_dielectric,
                has_implicit_solvent=ctx._converted_system.has_implicit_solvent,
                solvent_dielectric=ctx._converted_system.solvent_dielectric,
                solute_dielectric=ctx._converted_system.solute_dielectric,
                nonbonded_engine=ctx._nonbonded_engine,
                pme_grid_dims=ctx._converted_system.pme_grid_dims,
                use_pallas=ctx._config.use_pallas,
                pallas_interpret=ctx._config.pallas_interpret,
                lj_switch_width=ctx._converted_system.lj_switch_width,
                neighbor_idx=getattr(ctx, "_neighbor_idx", None),
                tile_pair_nl=getattr(ctx, "_tile_pair_nl", None),
                gbsa_tile_list=getattr(ctx, "_gbsa_tile_list", None),
                use_pallas_pme=ctx._config.use_pallas_pme,
                custom_nonbonded_chunk_size=ctx._config.custom_nonbonded_chunk_size,
                is_triclinic=ctx._is_triclinic,
                gb_use_neighbor_list=ctx._gb_use_neighbor_list,
                gbsa_surface_tension=ctx._converted_system.gbsa_surface_tension,
                pme_alpha=eff_pme_alpha,
                nonbonded_offsets=ctx._converted_system.nonbonded_offsets,
                default_params=ctx._converted_system.default_parameters,
                lj_combining_rule=ctx._lj_combining_rule,
                exceptions_use_pbc=ctx._exceptions_use_pbc,
                dispersion_follows_parameters=ctx._dispersion_follows_parameters,
                lj_pme_alpha=eff_lj_pme_alpha,
                lj_pme_grid_dims=eff_lj_pme_dims,
            )
            return e_sys + _constraint_penalty_energy_akma(
                p_eval,
                ctx._box_vectors,
                ctx._bundle,
                ctx._converted_system.has_pbc,
                exceptions_use_pbc=ctx._exceptions_use_pbc,
            )

        solver = _jaxopt.LBFGS(
            fun=energy_fn,
            maxiter=max_iterations,
            tol=tol_akma,
            linesearch="zoom",
            maxls=30,
            stop_if_linesearch_fails=True,
        )
        res = solver.run(pos_init)
        min_pos = res.params

        # Project constraints & virtual sites, and refresh neighbor list
        ctx._positions = min_pos
        ctx.computeVirtualSites()
        ctx.applyConstraints()
        ctx._init_neighbor_list(ctx._positions)
        ctx._morton_nl_dirty = True
        ctx._morton_forces_dirty = True
        ctx._velocities = jnp.zeros_like(ctx._velocities)

    def _minimize_fire(
        self, tol_akma: float, max_iter: int, reporter: Any = None  # noqa: ARG002
    ) -> None:
        """Energy minimization using FIRE descent."""
        ctx = self.context
        ctx.computeVirtualSites()
        ctx.applyConstraints()
        pos_init = ctx._positions
        max_iterations = max_iter if max_iter > 0 else 500

        gp = ctx._get_dynamic_global_params()
        ewald_tol = (
            ctx._converted_system.ewald_error_tolerance
            if ctx._converted_system.ewald_error_tolerance is not None
            else ctx._config.ewald_error_tolerance
        )
        eff_nb_method = ctx._effective_nonbonded_method()
        eff_pme_alpha = ctx._effective_pme_alpha()
        eff_lj_pme_alpha = ctx._effective_lj_pme_alpha()
        eff_lj_pme_dims = _resolve_cs_lj_pme_grid_dims(
            ctx._converted_system, ctx._config, ctx._box_vectors
        )

        def energy_fn(p: jnp.ndarray) -> jnp.ndarray:
            p_eval = ctx._converted_system.compute_virtual_sites(p)
            e_sys = _eval_system_potential_energy(
                p_eval,
                ctx._box_vectors,
                ctx._bundle,
                ctx._custom_forces,
                ctx._position_restraints,
                gp,
                ctx._converted_system.force_group_by_component,
                -1,
                ctx._converted_system.has_pbc,
                ctx._converted_system.nonbonded_cutoff,
                bond_force_groups=ctx._converted_system.bond_force_groups,
                angle_force_groups=ctx._converted_system.angle_force_groups,
                dihedral_force_groups=ctx._converted_system.dihedral_force_groups,
                cmap_force_groups=ctx._converted_system.cmap_force_groups,
                nonbonded_method=eff_nb_method,
                has_nonbonded=ctx._converted_system.has_nonbonded,
                use_dispersion_correction=ctx._config.use_dispersion_correction
                and ctx._converted_system.use_dispersion_correction,
                ewald_error_tolerance=ewald_tol,
                rf_dielectric=ctx._converted_system.reaction_field_dielectric,
                has_implicit_solvent=ctx._converted_system.has_implicit_solvent,
                solvent_dielectric=ctx._converted_system.solvent_dielectric,
                solute_dielectric=ctx._converted_system.solute_dielectric,
                nonbonded_engine=ctx._nonbonded_engine,
                pme_grid_dims=ctx._converted_system.pme_grid_dims,
                use_pallas=ctx._config.use_pallas,
                pallas_interpret=ctx._config.pallas_interpret,
                lj_switch_width=ctx._converted_system.lj_switch_width,
                neighbor_idx=getattr(ctx, "_neighbor_idx", None),
                tile_pair_nl=getattr(ctx, "_tile_pair_nl", None),
                gbsa_tile_list=getattr(ctx, "_gbsa_tile_list", None),
                use_pallas_pme=ctx._config.use_pallas_pme,
                custom_nonbonded_chunk_size=ctx._config.custom_nonbonded_chunk_size,
                is_triclinic=ctx._is_triclinic,
                gb_use_neighbor_list=ctx._gb_use_neighbor_list,
                gbsa_surface_tension=ctx._converted_system.gbsa_surface_tension,
                pme_alpha=eff_pme_alpha,
                nonbonded_offsets=ctx._converted_system.nonbonded_offsets,
                default_params=ctx._converted_system.default_parameters,
                lj_combining_rule=ctx._lj_combining_rule,
                exceptions_use_pbc=ctx._exceptions_use_pbc,
                dispersion_follows_parameters=ctx._dispersion_follows_parameters,
                lj_pme_alpha=eff_lj_pme_alpha,
                lj_pme_grid_dims=eff_lj_pme_dims,
            )
            return e_sys + _constraint_penalty_energy_akma(
                p_eval,
                ctx._box_vectors,
                ctx._bundle,
                ctx._converted_system.has_pbc,
                exceptions_use_pbc=ctx._exceptions_use_pbc,
            )

        init_fn, step_fn = jax_md_minimize.fire_descent(
            energy_fn,
            shift_fn=lambda r, dr: r + dr,
            dt_start=0.002,
            dt_max=0.02,
        )
        state = init_fn(pos_init)

        for _ in range(max_iterations):
            state = step_fn(state)
            f_norm = float(jnp.sqrt(jnp.mean(jnp.sum(state.force**2, axis=-1))))
            if f_norm <= tol_akma:
                break

        ctx._positions = state.position
        ctx.computeVirtualSites()
        ctx.applyConstraints()
        ctx._init_neighbor_list(ctx._positions)
        ctx._morton_nl_dirty = True
        ctx._morton_forces_dirty = True
        ctx._velocities = jnp.zeros_like(ctx._velocities)

    def _minimize_openmm_cpu(
        self, tol_omm: float, max_iter: int, reporter: Any = None
    ) -> None:
        """Fallback minimization using OpenMM Reference / CPU platform."""
        omm_mod = cast("Any", openmm)
        pos_nm = self.context.getState(getPositions=True).getPositions()
        platform = omm_mod.Platform.getPlatformByName("Reference")
        cpu_ctx = omm_mod.Context(
            self.system, omm_mod.VerletIntegrator(0.001), platform
        )
        cpu_ctx.setPositions(pos_nm)
        if self.context._box_vectors is not None:
            bv: Any = self.context.getState().getPeriodicBoxVectors()
            cpu_ctx.setPeriodicBoxVectors(bv[0], bv[1], bv[2])

        if reporter is not None:
            omm_mod.LocalEnergyMinimizer.minimize(cpu_ctx, tol_omm, max_iter, reporter)
        else:
            omm_mod.LocalEnergyMinimizer.minimize(cpu_ctx, tol_omm, max_iter)
        min_state = cpu_ctx.getState(getPositions=True)
        self.context.setPositions(min_state.getPositions(asNumpy=True))
        self.context._velocities = jnp.zeros_like(self.context._velocities)


def _pad_positions_to_bucket(
    real_pos: jnp.ndarray,
    bucket: int,
    box_mat: jnp.ndarray | None,
) -> jnp.ndarray:
    """Pad `(N_real, 3)` positions to `(bucket, 3)` placing ghost atoms on a lattice."""
    n_real = int(real_pos.shape[0])
    n_ghost = bucket - n_real
    if n_ghost <= 0:
        return jnp.array(real_pos[:bucket], copy=True)

    box_np = None if box_mat is None else np.asarray(box_mat, dtype=np.float64)
    if box_np is not None and box_np.shape == (3, 3):
        box_np = np.diag(box_np)
    if box_np is None or box_np.shape != (3,) or np.any(box_np <= 0.0):
        ghost = jnp.zeros((n_ghost, 3), dtype=real_pos.dtype)
    else:
        m = max(int(np.ceil(n_ghost ** (1.0 / 3.0))), 1)
        axes = [np.linspace(0.0, box_np[d], num=m, endpoint=False) for d in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        ghost = jnp.asarray(grid[:n_ghost], dtype=real_pos.dtype)
    return jnp.concatenate([real_pos, ghost], axis=0)


def _pad_axis0(arr: jnp.ndarray | None, target: int, fill_value: float | bool = 0) -> jnp.ndarray | None:
    """Pad array along axis 0 up to `target` rows if `arr.shape[0] < target`."""
    if arr is None:
        return None
    cur = int(arr.shape[0])
    if cur == target:
        return arr
    if cur > target:
        return arr[:target]
    pad_spec = [(0, target - cur)] + [(0, 0)] * (arr.ndim - 1)
    return jnp.pad(arr, pad_spec, constant_values=fill_value)


def _re_pad_bundle_to_spec(bundle: MolecularBundle, target_spec: MolecularShapeSpec) -> MolecularBundle:
    """Promote a `MolecularBundle` to a larger `MolecularShapeSpec` bucket without recompilation."""
    if bundle.shape_spec == target_spec:
        return bundle

    a = ATOM_BUCKETS[target_spec.atom_bucket_idx]
    bb = BOND_BUCKETS[target_spec.bond_bucket_idx]
    ab = ANGLE_BUCKETS[target_spec.angle_bucket_idx]
    db = DIHEDRAL_BUCKETS[target_spec.dihedral_bucket_idx]
    wb = WATER_BUCKETS[target_spec.water_bucket_idx]
    eb = EXCL_BUCKETS[target_spec.excl_bucket_idx]
    cb = CMAP_BUCKETS[target_spec.cmap_bucket_idx]
    mb = CMAP_MAP_BUCKETS[target_spec.cmap_map_bucket_idx]
    xb = EXCEPTION_BUCKETS[target_spec.exception_bucket_idx]
    cons_b = CONSTRAINT_BUCKETS[target_spec.constraint_bucket_idx]

    n_real = int(jnp.asarray(bundle.n_atoms))
    real_pos = bundle.positions[:n_real]
    pos_padded = _pad_positions_to_bucket(
        real_pos, a, bundle.box if target_spec.has_pbc else None
    )

    if cons_b == 0:
        c_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
        c_lens = jnp.zeros((0,), dtype=jnp.float32)
        c_mask = jnp.zeros((0,), dtype=bool)
    else:
        cp_in = bundle.constraint_pairs
        cl_in = bundle.constraint_lengths
        cm_in = bundle.constraint_mask
        c_pairs = (
            _pad_axis0(cp_in, cons_b, 0)
            if cp_in is not None and cp_in.size > 0
            else jnp.zeros((cons_b, 2), dtype=jnp.int32)
        )
        c_lens = (
            _pad_axis0(cl_in, cons_b, 0.0)
            if cl_in is not None and cl_in.size > 0
            else jnp.zeros((cons_b,), dtype=jnp.float32)
        )
        c_mask = (
            _pad_axis0(cm_in, cons_b, False)
            if cm_in is not None and cm_in.size > 0
            else jnp.zeros((cons_b,), dtype=bool)
        )

    mol_id = bundle.molecule_id
    if mol_id is not None:
        cur_a = int(mol_id.shape[0])
        if cur_a < a:
            extra_ids = jnp.arange(cur_a, a, dtype=jnp.int32) + jnp.max(mol_id) + 1
            mol_id = jnp.concatenate([mol_id, extra_ids], axis=0)
        else:
            mol_id = mol_id[:a]

    excl_dense_i = _pad_axis0(bundle.excl_dense_indices, a, -1)
    excl_dense_sv = _pad_axis0(bundle.excl_dense_scales_vdw, a, 1.0)
    excl_dense_se = _pad_axis0(bundle.excl_dense_scales_elec, a, 1.0)
    if excl_dense_i is None:
        excl_dense_i = jnp.full((a, 32), -1, dtype=jnp.int32)
        excl_dense_sv = jnp.ones((a, 32), dtype=jnp.float32)
        excl_dense_se = jnp.ones((a, 32), dtype=jnp.float32)

    return dataclasses.replace(
        bundle,
        positions=pos_padded,
        masses=_pad_axis0(bundle.masses, a, 1.0),
        charges=_pad_axis0(bundle.charges, a, 0.0),
        sigmas=_pad_axis0(bundle.sigmas, a, 0.0),
        epsilons=_pad_axis0(bundle.epsilons, a, 0.0),
        radii=_pad_axis0(bundle.radii, a, 0.0),
        scaled_radii=_pad_axis0(bundle.scaled_radii, a, 0.0),
        atom_mask=_pad_axis0(bundle.atom_mask, a, False),
        bond_idx=_pad_axis0(bundle.bond_idx, bb, 0),
        bond_params=_pad_axis0(bundle.bond_params, bb, 0.0),
        bond_mask=_pad_axis0(bundle.bond_mask, bb, False),
        angle_idx=_pad_axis0(bundle.angle_idx, ab, 0),
        angle_params=_pad_axis0(bundle.angle_params, ab, 0.0),
        angle_mask=_pad_axis0(bundle.angle_mask, ab, False),
        dihedral_idx=_pad_axis0(bundle.dihedral_idx, db, 0),
        dihedral_params=_pad_axis0(bundle.dihedral_params, db, 0.0),
        dihedral_mask=_pad_axis0(bundle.dihedral_mask, db, False),
        improper_idx=_pad_axis0(bundle.improper_idx, db, 0),
        improper_params=_pad_axis0(bundle.improper_params, db, 0.0),
        improper_mask=_pad_axis0(bundle.improper_mask, db, False),
        urey_bradley_idx=_pad_axis0(bundle.urey_bradley_idx, ab, 0),
        urey_bradley_params=_pad_axis0(bundle.urey_bradley_params, ab, 0.0),
        urey_bradley_mask=_pad_axis0(bundle.urey_bradley_mask, ab, False),
        cmap_torsion_idx=_pad_axis0(bundle.cmap_torsion_idx, cb, 0),
        cmap_energy_grids=_pad_axis0(bundle.cmap_energy_grids, cb, 0.0),
        cmap_mask=_pad_axis0(bundle.cmap_mask, cb, False),
        cmap_indices=_pad_axis0(bundle.cmap_indices, cb, 0),
        cmap_coeffs=_pad_axis0(bundle.cmap_coeffs, mb, 0.0),
        water_indices=_pad_axis0(bundle.water_indices, wb, 0),
        water_mask=_pad_axis0(bundle.water_mask, wb, False),
        constraint_pairs=c_pairs,
        constraint_lengths=c_lens,
        constraint_mask=c_mask,
        excl_indices=_pad_axis0(bundle.excl_indices, eb, 0),
        excl_scales_vdw=_pad_axis0(bundle.excl_scales_vdw, eb, 0.0),
        excl_scales_elec=_pad_axis0(bundle.excl_scales_elec, eb, 0.0),
        excl_mask=_pad_axis0(bundle.excl_mask, eb, False),
        excl_dense_indices=excl_dense_i,
        excl_dense_scales_vdw=excl_dense_sv,
        excl_dense_scales_elec=excl_dense_se,
        exception_pairs=_pad_axis0(bundle.exception_pairs, xb, 0),
        exception_sigmas=_pad_axis0(bundle.exception_sigmas, xb, 0.0),
        exception_epsilons=_pad_axis0(bundle.exception_epsilons, xb, 0.0),
        exception_chargeprods=_pad_axis0(bundle.exception_chargeprods, xb, 0.0),
        exception_mask=_pad_axis0(bundle.exception_mask, xb, False),
        molecule_id=mol_id,
        cosolvent_group_id=_pad_axis0(bundle.cosolvent_group_id, a, 0),
        shape_spec=target_spec,
    )


def _align_bundles_to_common_bucket(bundles: list[MolecularBundle]) -> list[MolecularBundle]:
    """Align ragged `MolecularBundle`s with compatible physics regimes to a single bucket."""
    if len(bundles) <= 1:
        return list(bundles)

    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, b in enumerate(bundles):
        sp = b.shape_spec
        if not sp.use_size_buckets:
            continue
        grid_shape = tuple(b.cmap_energy_grids.shape[1:])
        tors_cols = int(b.cmap_torsion_idx.shape[-1])
        key = (
            sp.has_pbc,
            sp.has_implicit_solvent,
            sp.boundary_condition,
            sp.is_triclinic,
            grid_shape,
            tors_cols,
        )
        groups.setdefault(key, []).append(idx)

    aligned = list(bundles)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        specs = [aligned[i].shape_spec for i in indices]
        common_spec = MolecularShapeSpec(
            atom_bucket_idx=max(s.atom_bucket_idx for s in specs),
            bond_bucket_idx=max(s.bond_bucket_idx for s in specs),
            angle_bucket_idx=max(s.angle_bucket_idx for s in specs),
            dihedral_bucket_idx=max(s.dihedral_bucket_idx for s in specs),
            water_bucket_idx=max(s.water_bucket_idx for s in specs),
            excl_bucket_idx=max(s.excl_bucket_idx for s in specs),
            cmap_bucket_idx=max(s.cmap_bucket_idx for s in specs),
            exception_bucket_idx=max(s.exception_bucket_idx for s in specs),
            has_pbc=specs[0].has_pbc,
            has_implicit_solvent=specs[0].has_implicit_solvent,
            boundary_condition=specs[0].boundary_condition,
            has_real_water=any(s.has_real_water for s in specs),
            use_size_buckets=True,
            exact_n_atoms=0,
            exact_n_waters=0,
            cmap_map_bucket_idx=max(s.cmap_map_bucket_idx for s in specs),
            constraint_bucket_idx=max(s.constraint_bucket_idx for s in specs),
            has_real_constraints=any(s.has_real_constraints for s in specs),
            is_triclinic=specs[0].is_triclinic,
            has_real_cosolvents=any(s.has_real_cosolvents for s in specs),
        )
        for i in indices:
            aligned[i] = _re_pad_bundle_to_spec(aligned[i], common_spec)
    return aligned


def _constraint_penalty_energy_akma(
    pos: jnp.ndarray,
    box: jnp.ndarray | None,
    bundle: MolecularBundle,
    has_pbc: bool,
    exceptions_use_pbc: bool = False,
    k_pen: float = 1200.0,
) -> jnp.ndarray:
    """Harmonic penalty on SETTLE waters, SHAKE constraints, and non-PBC bond jumps during minimization."""
    n = pos.shape[0]
    e_pen = jnp.array(0.0, dtype=pos.dtype)
    half_k = jnp.asarray(0.5 * k_pen, dtype=pos.dtype)
    box_pbc = box.astype(pos.dtype) if (has_pbc and box is not None) else None
    use_pbc_intra = bool(has_pbc and exceptions_use_pbc and box_pbc is not None)

    if (
        bundle.water_indices is not None
        and bundle.water_mask is not None
        and int(bundle.water_indices.shape[0]) > 0
    ):
        w_idx = bundle.water_indices
        w_valid = (
            bundle.water_mask
              & (w_idx[:, 0] < n)
              & (w_idx[:, 1] < n)
              & (w_idx[:, 2] < n)
        )
        io = jnp.where(w_valid, w_idx[:, 0], 0)
        ih1 = jnp.where(w_valid, w_idx[:, 1], 0)
        ih2 = jnp.where(w_valid, w_idx[:, 2], 0)
        r_oh = (
            jnp.asarray(bundle.water_r_OH, dtype=pos.dtype)
            if bundle.water_r_OH is not None
            else jnp.asarray(0.9572, dtype=pos.dtype)
        )
        r_hh = (
            jnp.asarray(bundle.water_r_HH, dtype=pos.dtype)
            if bundle.water_r_HH is not None
            else jnp.asarray(1.5139, dtype=pos.dtype)
        )
        dr1 = _minimum_image_dr(pos[ih1] - pos[io], box_pbc, use_pbc_intra)
        dr2 = _minimum_image_dr(pos[ih2] - pos[io], box_pbc, use_pbc_intra)
        dr3 = _minimum_image_dr(pos[ih2] - pos[ih1], box_pbc, use_pbc_intra)
        d1 = jnp.sqrt(jnp.sum(dr1**2, axis=-1) + 1e-12)
        d2 = jnp.sqrt(jnp.sum(dr2**2, axis=-1) + 1e-12)
        d3 = jnp.sqrt(jnp.sum(dr3**2, axis=-1) + 1e-12)
        ew = half_k * ((d1 - r_oh) ** 2 + (d2 - r_oh) ** 2 + (d3 - r_hh) ** 2)
        e_pen = e_pen + jnp.sum(jnp.where(w_valid, ew, 0.0))

    if (
        bundle.constraint_pairs is not None
        and bundle.constraint_lengths is not None
        and bundle.constraint_mask is not None
        and int(bundle.constraint_pairs.shape[0]) > 0
    ):
        c_idx = bundle.constraint_pairs
        c_valid = bundle.constraint_mask & (c_idx[:, 0] < n) & (c_idx[:, 1] < n)
        p1 = jnp.where(c_valid, c_idx[:, 0], 0)
        p2 = jnp.where(c_valid, c_idx[:, 1], 0)
        d0 = bundle.constraint_lengths.astype(pos.dtype)
        drc = _minimum_image_dr(pos[p2] - pos[p1], box_pbc, use_pbc_intra)
        dc = jnp.sqrt(jnp.sum(drc**2, axis=-1) + 1e-12)
        ec = half_k * ((dc - d0) ** 2)
        e_pen = e_pen + jnp.sum(jnp.where(c_valid, ec, 0.0))

    # Prevent L-BFGS / FIRE line-search steps from teleporting bonded atoms into
    # distant periodic images when nonbonded exclusions use raw unwrapped distances.
    if (
        has_pbc
        and not exceptions_use_pbc
        and box_pbc is not None
        and bundle.bond_idx is not None
        and bundle.bond_params is not None
        and bundle.bond_mask is not None
        and int(bundle.bond_idx.shape[0]) > 0
    ):
        b_idx = bundle.bond_idx
        b_valid = bundle.bond_mask & (b_idx[:, 0] < n) & (b_idx[:, 1] < n)
        bp1 = jnp.where(b_valid, b_idx[:, 0], 0)
        bp2 = jnp.where(b_valid, b_idx[:, 1], 0)
        r0_b = bundle.bond_params[:, 0].astype(pos.dtype)
        k_b = bundle.bond_params[:, 1].astype(pos.dtype)
        dr_raw = pos[bp2] - pos[bp1]
        dr_mic = _minimum_image_dr(dr_raw, box_pbc, True)
        r_raw = jnp.sqrt(jnp.sum(dr_raw**2, axis=-1) + 1e-12)
        r_mic = jnp.sqrt(jnp.sum(dr_mic**2, axis=-1) + 1e-12)
        eb_unw = 0.5 * k_b * ((r_raw - r0_b) ** 2 - (r_mic - r0_b) ** 2)
        e_pen = e_pen + jnp.sum(jnp.where(b_valid, eb_unw, 0.0))

    return e_pen


def _clone_integrator(integrator: openmm.Integrator | str) -> openmm.Integrator:
    """Deserialize or clone an OpenMM `Integrator` so each replica has independent state."""
    omm_mod = cast("Any", openmm)
    if isinstance(integrator, str):
        return omm_mod.XmlSerializer.deserialize(integrator)
    return omm_mod.XmlSerializer.deserialize(omm_mod.XmlSerializer.serialize(integrator))


def _effective_cs_nonbonded_method(
    cs: ConvertedOpenMMSystem, config: ProlixOpenMMConfig
) -> str:
    """Resolve effective nonbonded method string, honoring `config.use_lj_pme`."""
    raw = str(cs.nonbonded_method)
    if bool(getattr(config, "use_lj_pme", False)) and raw.lower() in (
        "pme",
        "spme",
        "lj_pme",
        "ljpme",
    ):
        return "lj_pme"
    return raw


def _resolve_cs_pme_alpha(cs: ConvertedOpenMMSystem, config: ProlixOpenMMConfig) -> float | None:
    """Resolve effective Ewald alpha (Å⁻¹) for a `ConvertedOpenMMSystem`."""
    if _effective_cs_nonbonded_method(cs, config).lower() not in (
        "pme",
        "spme",
        "lj_pme",
        "ljpme",
    ):
        return None
    if cs.explicit_pme_alpha is not None and float(cs.explicit_pme_alpha) > 0.0:
        return float(cs.explicit_pme_alpha)
    ewald_tol = (
        cs.ewald_error_tolerance
        if cs.ewald_error_tolerance is not None
        else config.ewald_error_tolerance
    )
    cutoff = float(cs.nonbonded_cutoff)
    if 0.0 < cutoff < float("inf"):
        return float((1.0 / cutoff) * math.sqrt(-math.log(2.0 * float(ewald_tol))))
    if cs.bundle.pme_alpha is not None and float(cs.bundle.pme_alpha) > 0.0:
        return float(cs.bundle.pme_alpha)
    return 0.34


def _resolve_cs_lj_pme_alpha(
    cs: ConvertedOpenMMSystem, config: ProlixOpenMMConfig
) -> float | None:
    """Resolve effective dispersion Ewald alpha (Å⁻¹) for `NonbondedForce.LJPME`."""
    if _effective_cs_nonbonded_method(cs, config).lower() not in ("lj_pme", "ljpme"):
        return None
    if cs.explicit_lj_pme_alpha is not None and float(cs.explicit_lj_pme_alpha) > 0.0:
        return float(cs.explicit_lj_pme_alpha)
    ewald_tol = (
        cs.ewald_error_tolerance
        if cs.ewald_error_tolerance is not None
        else config.ewald_error_tolerance
    )
    cutoff = float(cs.nonbonded_cutoff)
    if 0.0 < cutoff < float("inf"):
        return float((1.0 / cutoff) * math.sqrt(-math.log(2.0 * float(ewald_tol))))
    return 0.34


def _resolve_cs_lj_pme_grid_dims(
    cs: ConvertedOpenMMSystem,
    config: ProlixOpenMMConfig,
    box_akma: jnp.ndarray | None,
) -> tuple[int, int, int] | None:
    """Resolve concrete host `(nx, ny, nz)` grid dimensions for `NonbondedForce.LJPME`."""
    if _effective_cs_nonbonded_method(cs, config).lower() not in ("lj_pme", "ljpme"):
        return None
    if cs.lj_pme_grid_dims is not None:
        return cs.lj_pme_grid_dims
    if box_akma is not None and cs.has_pbc:
        box_np = np.asarray(box_akma, dtype=np.float64)
        diag_np = np.diag(box_np) if box_np.ndim == 2 else box_np
        ewald_tol = (
            cs.ewald_error_tolerance
            if cs.ewald_error_tolerance is not None
            else config.ewald_error_tolerance
        )
        cutoff = float(cs.nonbonded_cutoff) if cs.nonbonded_cutoff < float("inf") else 9.0
        _, dims = resolve_ljpme_parameters(
            diag_np,
            cutoff=cutoff,
            ewald_error_tolerance=float(ewald_tol),
            explicit_alpha=float(cs.explicit_lj_pme_alpha or 0.0),
        )
        return dims
    return cs.pme_grid_dims


def _pr_signature(
    prs: Sequence[PositionRestraints],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return static shape & force-group signature for uniform `PositionRestraints` stacking."""
    return tuple((int(pr.force_group), tuple(pr.indices.shape)) for pr in prs)


@eqx.filter_jit
def _vmapped_batch_energy_and_forces_akma(  # noqa: PLR0913
    pos_b: jnp.ndarray,
    box_b: jnp.ndarray | None,
    base_bundles_b: MolecularBundle,
    bond_fg_b: jnp.ndarray | None,
    angle_fg_b: jnp.ndarray | None,
    dihedral_fg_b: jnp.ndarray | None,
    cmap_fg_b: jnp.ndarray | None,
    pr_b: tuple[PositionRestraints, ...],
    offsets_b: NonbondedParameterOffsets | None,
    gp_vec_b: jnp.ndarray,
    param_names: tuple[str, ...],
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
    cutoff: float,
    nonbonded_method: str,
    has_nonbonded: bool,
    use_dispersion_correction: bool,
    ewald_error_tolerance: float,
    pme_alpha: float | None,
    rf_dielectric: float,
    has_implicit_solvent: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    nonbonded_engine: str,
    pme_grid_dims: tuple[int, int, int] | None,
    lj_switch_width: float,
    custom_nonbonded_chunk_size: int,
    is_triclinic: bool,
    mixed_precision: bool,
    gbsa_surface_tension: float,
    default_params: dict[str, float],
    lj_combining_rule: str,
    exceptions_use_pbc: bool,
    dispersion_follows_parameters: bool,
    lj_pme_alpha: float | None = None,
    lj_pme_grid_dims: tuple[int, int, int] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-compile `jax.vmap` energy (kcal/mol) and forces (kcal/(mol*Å)) across `B` replicas."""

    def _one(
        p_i: jnp.ndarray,
        bx_i: jnp.ndarray | None,
        b_i: MolecularBundle,
        bfg_i: jnp.ndarray | None,
        afg_i: jnp.ndarray | None,
        dfg_i: jnp.ndarray | None,
        cfg_i: jnp.ndarray | None,
        pr_i: tuple[PositionRestraints, ...],
        off_i: NonbondedParameterOffsets | None,
        gp_v_i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        eval_p = p_i.astype(jnp.float32) if mixed_precision else p_i
        eval_bx = bx_i.astype(jnp.float32) if (mixed_precision and bx_i is not None) else bx_i
        gp_dict = {nm: gp_v_i[idx] for idx, nm in enumerate(param_names)}

        def _e_fn(p_arg: jnp.ndarray) -> jnp.ndarray:
            return _eval_system_potential_energy(
                p_arg,
                eval_bx,
                b_i,
                (),
                pr_i,
                gp_dict,
                force_group_by_component,
                groups_mask,
                has_pbc,
                cutoff,
                bond_force_groups=bfg_i,
                angle_force_groups=afg_i,
                dihedral_force_groups=dfg_i,
                cmap_force_groups=cfg_i,
                nonbonded_method=nonbonded_method,
                has_nonbonded=has_nonbonded,
                use_dispersion_correction=use_dispersion_correction,
                ewald_error_tolerance=ewald_error_tolerance,
                pme_alpha=pme_alpha,
                rf_dielectric=rf_dielectric,
                has_implicit_solvent=has_implicit_solvent,
                solvent_dielectric=solvent_dielectric,
                solute_dielectric=solute_dielectric,
                nonbonded_engine=nonbonded_engine,
                pme_grid_dims=pme_grid_dims,
                use_pallas=False,
                pallas_interpret=False,
                lj_switch_width=lj_switch_width,
                neighbor_idx=None,
                tile_pair_nl=None,
                use_pallas_pme=False,
                custom_nonbonded_chunk_size=custom_nonbonded_chunk_size,
                is_triclinic=is_triclinic,
                gb_use_neighbor_list=False,
                gbsa_surface_tension=gbsa_surface_tension,
                nonbonded_offsets=off_i,
                default_params=default_params,
                lj_combining_rule=lj_combining_rule,
                exceptions_use_pbc=exceptions_use_pbc,
                dispersion_follows_parameters=dispersion_follows_parameters,
                lj_pme_alpha=lj_pme_alpha,
                lj_pme_grid_dims=lj_pme_grid_dims,
            )

        pe, grad = jax.value_and_grad(_e_fn)(eval_p)
        forces = -grad
        if mixed_precision:
            pe = pe.astype(p_i.dtype)
            forces = forces.astype(p_i.dtype)
        return pe, forces

    in_axes = (
        0,
        0 if box_b is not None else None,
        0,
        0 if bond_fg_b is not None else None,
        0 if angle_fg_b is not None else None,
        0 if dihedral_fg_b is not None else None,
        0 if cmap_fg_b is not None else None,
        0 if pr_b else None,
        0 if offsets_b is not None else None,
        0,
    )
    return jax.vmap(_one, in_axes=in_axes)(
        pos_b,
        box_b,
        base_bundles_b,
        bond_fg_b,
        angle_fg_b,
        dihedral_fg_b,
        cmap_fg_b,
        pr_b,
        offsets_b,
        gp_vec_b,
    )


@eqx.filter_jit
def _vmapped_batch_value_and_grad_positions_kj_nm(  # noqa: PLR0913
    pos_nm_b: jnp.ndarray,
    box_b: jnp.ndarray | None,
    base_bundles_b: MolecularBundle,
    bond_fg_b: jnp.ndarray | None,
    angle_fg_b: jnp.ndarray | None,
    dihedral_fg_b: jnp.ndarray | None,
    cmap_fg_b: jnp.ndarray | None,
    pr_b: tuple[PositionRestraints, ...],
    offsets_b: NonbondedParameterOffsets | None,
    gp_vec_b: jnp.ndarray,
    param_names: tuple[str, ...],
    force_group_by_component: dict[str, int],
    has_pbc: bool,
    cutoff: float,
    nonbonded_method: str,
    has_nonbonded: bool,
    use_dispersion_correction: bool,
    ewald_error_tolerance: float,
    pme_alpha: float | None,
    rf_dielectric: float,
    has_implicit_solvent: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    nonbonded_engine: str,
    pme_grid_dims: tuple[int, int, int] | None,
    lj_switch_width: float,
    custom_nonbonded_chunk_size: int,
    is_triclinic: bool,
    gbsa_surface_tension: float,
    default_params: dict[str, float],
    lj_combining_rule: str,
    exceptions_use_pbc: bool,
    dispersion_follows_parameters: bool,
    lj_pme_alpha: float | None = None,
    lj_pme_grid_dims: tuple[int, int, int] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable `jax.vmap` energy (kJ/mol) and `dU/dx` (kJ/(mol*nm)) across `B` replicas."""

    def _one(
        p_nm_i: jnp.ndarray,
        bx_i: jnp.ndarray | None,
        b_i: MolecularBundle,
        bfg_i: jnp.ndarray | None,
        afg_i: jnp.ndarray | None,
        dfg_i: jnp.ndarray | None,
        cfg_i: jnp.ndarray | None,
        pr_i: tuple[PositionRestraints, ...],
        off_i: NonbondedParameterOffsets | None,
        gp_v_i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gp_dict = {nm: gp_v_i[idx] for idx, nm in enumerate(param_names)}

        def _u_nm(p_nm_arg: jnp.ndarray) -> jnp.ndarray:
            p_a = p_nm_arg * float(NM_TO_ANGSTROM)
            u_kcal = _eval_system_potential_energy(
                p_a,
                bx_i,
                b_i,
                (),
                pr_i,
                gp_dict,
                force_group_by_component,
                -1,
                has_pbc,
                cutoff,
                bond_force_groups=bfg_i,
                angle_force_groups=afg_i,
                dihedral_force_groups=dfg_i,
                cmap_force_groups=cfg_i,
                nonbonded_method=nonbonded_method,
                has_nonbonded=has_nonbonded,
                use_dispersion_correction=use_dispersion_correction,
                ewald_error_tolerance=ewald_error_tolerance,
                pme_alpha=pme_alpha,
                rf_dielectric=rf_dielectric,
                has_implicit_solvent=has_implicit_solvent,
                solvent_dielectric=solvent_dielectric,
                solute_dielectric=solute_dielectric,
                nonbonded_engine=nonbonded_engine,
                pme_grid_dims=pme_grid_dims,
                use_pallas=False,
                pallas_interpret=False,
                lj_switch_width=lj_switch_width,
                neighbor_idx=None,
                tile_pair_nl=None,
                use_pallas_pme=False,
                custom_nonbonded_chunk_size=custom_nonbonded_chunk_size,
                is_triclinic=is_triclinic,
                gb_use_neighbor_list=False,
                gbsa_surface_tension=gbsa_surface_tension,
                nonbonded_offsets=off_i,
                default_params=default_params,
                lj_combining_rule=lj_combining_rule,
                exceptions_use_pbc=exceptions_use_pbc,
                dispersion_follows_parameters=dispersion_follows_parameters,
                lj_pme_alpha=lj_pme_alpha,
                lj_pme_grid_dims=lj_pme_grid_dims,
            )
            return u_kcal * float(KCAL_PER_MOL_TO_KJ_PER_MOL)

        return jax.value_and_grad(_u_nm)(p_nm_i)

    in_axes = (
        0,
        0 if box_b is not None else None,
        0,
        0 if bond_fg_b is not None else None,
        0 if angle_fg_b is not None else None,
        0 if dihedral_fg_b is not None else None,
        0 if cmap_fg_b is not None else None,
        0 if pr_b else None,
        0 if offsets_b is not None else None,
        0,
    )
    return jax.vmap(_one, in_axes=in_axes)(
        pos_nm_b,
        box_b,
        base_bundles_b,
        bond_fg_b,
        angle_fg_b,
        dihedral_fg_b,
        cmap_fg_b,
        pr_b,
        offsets_b,
        gp_vec_b,
    )


@eqx.filter_jit
def _vmapped_batch_nonbonded_param_derivatives_kj(  # noqa: PLR0913
    pos_b: jnp.ndarray,
    box_b: jnp.ndarray | None,
    base_bundles_b: MolecularBundle,
    offsets_b: NonbondedParameterOffsets,
    gp_vec_b: jnp.ndarray,
    param_names: tuple[str, ...],
    force_group_by_component: dict[str, int],
    groups_mask: int,
    has_pbc: bool,
    cutoff: float,
    nonbonded_method: str,
    has_nonbonded: bool,
    use_dispersion_correction: bool,
    ewald_error_tolerance: float,
    pme_alpha: float | None,
    rf_dielectric: float,
    nonbonded_engine: str,
    pme_grid_dims: tuple[int, int, int] | None,
    lj_switch_width: float,
    custom_nonbonded_chunk_size: int,
    is_triclinic: bool,
    default_params: dict[str, float],
    lj_combining_rule: str,
    exceptions_use_pbc: bool,
    dispersion_follows_parameters: bool,
    lj_pme_alpha: float | None = None,
    lj_pme_grid_dims: tuple[int, int, int] | None = None,
) -> jnp.ndarray:
    """Single-pass `jax.vmap` alchemical `dU/dλ` (kJ/mol) across `B` λ windows."""

    def _one_grad(
        p_i: jnp.ndarray,
        bx_i: jnp.ndarray | None,
        b_i: MolecularBundle,
        off_i: NonbondedParameterOffsets,
        gp_v_i: jnp.ndarray,
    ) -> jnp.ndarray:
        def _e_kj(gp_arg: jnp.ndarray) -> jnp.ndarray:
            gp_dict = {nm: gp_arg[idx] for idx, nm in enumerate(param_names)}
            u_kcal = _eval_nonbonded_terms(
                p_i,
                bx_i,
                b_i,
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
                use_pallas=False,
                pallas_interpret=False,
                lj_switch_width=lj_switch_width,
                neighbor_idx=None,
                use_pallas_pme=False,
                custom_nonbonded_chunk_size=custom_nonbonded_chunk_size,
                tile_pair_nl=None,
                is_triclinic=is_triclinic,
                global_params=gp_dict,
                nonbonded_offsets=off_i,
                default_params=default_params,
                lj_combining_rule=lj_combining_rule,
                exceptions_use_pbc=exceptions_use_pbc,
                dispersion_follows_parameters=dispersion_follows_parameters,
                force_pure_jax_tile_pairs=True,
                lj_pme_alpha=lj_pme_alpha,
                lj_pme_grid_dims=lj_pme_grid_dims,
            )
            return u_kcal * float(KCAL_PER_MOL_TO_KJ_PER_MOL)

        return jax.grad(_e_kj)(gp_v_i)

    in_axes = (0, 0 if box_b is not None else None, 0, 0, 0)
    return jax.vmap(_one_grad, in_axes=in_axes)(
        pos_b, box_b, base_bundles_b, offsets_b, gp_vec_b
    )


@eqx.filter_jit
def _vmapped_batch_minimize_akma(  # noqa: PLR0913
    pos_b: jnp.ndarray,
    box_b: jnp.ndarray | None,
    base_bundles_b: MolecularBundle,
    bond_fg_b: jnp.ndarray | None,
    angle_fg_b: jnp.ndarray | None,
    dihedral_fg_b: jnp.ndarray | None,
    cmap_fg_b: jnp.ndarray | None,
    pr_b: tuple[PositionRestraints, ...],
    offsets_b: NonbondedParameterOffsets | None,
    gp_vec_b: jnp.ndarray,
    param_names: tuple[str, ...],
    force_group_by_component: dict[str, int],
    has_pbc: bool,
    cutoff: float,
    nonbonded_method: str,
    has_nonbonded: bool,
    use_dispersion_correction: bool,
    ewald_error_tolerance: float,
    pme_alpha: float | None,
    rf_dielectric: float,
    has_implicit_solvent: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    nonbonded_engine: str,
    pme_grid_dims: tuple[int, int, int] | None,
    lj_switch_width: float,
    custom_nonbonded_chunk_size: int,
    is_triclinic: bool,
    mixed_precision: bool,
    gbsa_surface_tension: float,
    default_params: dict[str, float],
    lj_combining_rule: str,
    exceptions_use_pbc: bool,
    dispersion_follows_parameters: bool,
    lj_pme_alpha: float | None,
    lj_pme_grid_dims: tuple[int, int, int] | None,
    backend: str,
    tol_akma: float,
    max_iterations: int,
) -> jnp.ndarray:
    """Simultaneous `jax.vmap` GPU `fire` or `lbfgs` energy minimization across `B` replicas."""

    def _one_min(
        p_i: jnp.ndarray,
        bx_i: jnp.ndarray | None,
        b_i: MolecularBundle,
        bfg_i: jnp.ndarray | None,
        afg_i: jnp.ndarray | None,
        dfg_i: jnp.ndarray | None,
        cfg_i: jnp.ndarray | None,
        pr_i: tuple[PositionRestraints, ...],
        off_i: NonbondedParameterOffsets | None,
        gp_v_i: jnp.ndarray,
    ) -> jnp.ndarray:
        eval_p = p_i.astype(jnp.float32) if mixed_precision else p_i
        eval_bx = bx_i.astype(jnp.float32) if (mixed_precision and bx_i is not None) else bx_i
        gp_dict = {nm: gp_v_i[idx] for idx, nm in enumerate(param_names)}

        def _e_fn(p_arg: jnp.ndarray) -> jnp.ndarray:
            e_sys = _eval_system_potential_energy(
                p_arg,
                eval_bx,
                b_i,
                (),
                pr_i,
                gp_dict,
                force_group_by_component,
                -1,
                has_pbc,
                cutoff,
                bond_force_groups=bfg_i,
                angle_force_groups=afg_i,
                dihedral_force_groups=dfg_i,
                cmap_force_groups=cfg_i,
                nonbonded_method=nonbonded_method,
                has_nonbonded=has_nonbonded,
                use_dispersion_correction=use_dispersion_correction,
                ewald_error_tolerance=ewald_error_tolerance,
                pme_alpha=pme_alpha,
                rf_dielectric=rf_dielectric,
                has_implicit_solvent=has_implicit_solvent,
                solvent_dielectric=solvent_dielectric,
                solute_dielectric=solute_dielectric,
                nonbonded_engine=nonbonded_engine,
                pme_grid_dims=pme_grid_dims,
                use_pallas=False,
                pallas_interpret=False,
                lj_switch_width=lj_switch_width,
                neighbor_idx=None,
                tile_pair_nl=None,
                use_pallas_pme=False,
                custom_nonbonded_chunk_size=custom_nonbonded_chunk_size,
                is_triclinic=is_triclinic,
                gb_use_neighbor_list=False,
                gbsa_surface_tension=gbsa_surface_tension,
                nonbonded_offsets=off_i,
                default_params=default_params,
                lj_combining_rule=lj_combining_rule,
                exceptions_use_pbc=exceptions_use_pbc,
                dispersion_follows_parameters=dispersion_follows_parameters,
                lj_pme_alpha=lj_pme_alpha,
                lj_pme_grid_dims=lj_pme_grid_dims,
            )
            return e_sys + _constraint_penalty_energy_akma(
                p_arg,
                eval_bx,
                b_i,
                has_pbc,
                exceptions_use_pbc=exceptions_use_pbc,
            )

        if backend == "lbfgs":
            solver = _jaxopt.LBFGS(
                fun=_e_fn,
                maxiter=max_iterations,
                tol=tol_akma,
                linesearch="zoom",
                maxls=30,
                stop_if_linesearch_fails=True,
                jit=True,
                unroll=False,
            )
            min_p = solver.run(eval_p).params
        else:
            init_fn, step_fn = jax_md_minimize.fire_descent(
                _e_fn,
                shift_fn=lambda r, dr: r + dr,
                dt_start=0.002,
                dt_max=0.02,
            )
            st0 = init_fn(eval_p)
            tol_t = jnp.asarray(tol_akma, dtype=eval_p.dtype)
            max_it_i32 = jnp.int32(max_iterations)

            def _cond(carry: tuple[jnp.ndarray, Any]) -> jnp.ndarray:
                it, st = carry
                f_rms = jnp.sqrt(jnp.mean(jnp.sum(st.force**2, axis=-1)))
                return (it < max_it_i32) & (f_rms > tol_t)

            def _body(carry: tuple[jnp.ndarray, Any]) -> tuple[jnp.ndarray, Any]:
                it, st = carry
                return (it + jnp.int32(1), step_fn(st))

            _, st_fin = jax.lax.while_loop(_cond, _body, (jnp.int32(0), st0))
            min_p = st_fin.position

        return min_p.astype(p_i.dtype) if mixed_precision else min_p

    in_axes = (
        0,
        0 if box_b is not None else None,
        0,
        0 if bond_fg_b is not None else None,
        0 if angle_fg_b is not None else None,
        0 if dihedral_fg_b is not None else None,
        0 if cmap_fg_b is not None else None,
        0 if pr_b else None,
        0 if offsets_b is not None else None,
        0,
    )
    return jax.vmap(_one_min, in_axes=in_axes)(
        pos_b,
        box_b,
        base_bundles_b,
        bond_fg_b,
        angle_fg_b,
        dihedral_fg_b,
        cmap_fg_b,
        pr_b,
        offsets_b,
        gp_vec_b,
    )


def _converted_system_has_intermolecular_exceptions(cs: Any) -> bool:
    """Return True iff `cs` has any active exception joining distinct covalent/constraint molecules."""
    bundle = cs.bundle
    n_p = int(cs.n_real_particles)
    if n_p <= 0 or bundle.exception_pairs is None or bundle.exception_mask is None:
        return False
    ep_np = np.asarray(bundle.exception_pairs, dtype=np.int32)
    em_np = np.asarray(bundle.exception_mask, dtype=bool)
    if ep_np.shape[0] == 0:
        return False
    v_e = (
        em_np
        & (ep_np[:, 0] >= 0)
        & (ep_np[:, 0] < n_p)
        & (ep_np[:, 1] >= 0)
        & (ep_np[:, 1] < n_p)
    )
    if not np.any(v_e):
        return False

    parent = list(range(n_p))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(u: int, v: int) -> None:
        if 0 <= u < n_p and 0 <= v < n_p and u != v:
            ru, rv = _find(u), _find(v)
            if ru != rv:
                parent[rv] = ru

    if bundle.bond_idx is not None and bundle.bond_mask is not None:
        bi = np.asarray(bundle.bond_idx, dtype=np.int32)
        bm = np.asarray(bundle.bond_mask, dtype=bool)
        for row in bi[bm]:
            _union(int(row[0]), int(row[1]))
    if bundle.constraint_pairs is not None and bundle.constraint_mask is not None:
        cp = np.asarray(bundle.constraint_pairs, dtype=np.int32)
        cm = np.asarray(bundle.constraint_mask, dtype=bool)
        for row in cp[cm]:
            _union(int(row[0]), int(row[1]))
    if bundle.water_indices is not None and bundle.water_mask is not None:
        wi = np.asarray(bundle.water_indices, dtype=np.int32)
        wm = np.asarray(bundle.water_mask, dtype=bool)
        for row in wi[wm]:
            _union(int(row[0]), int(row[1]))
            _union(int(row[0]), int(row[2]))

    for e0, e1 in ep_np[v_e]:
        if _find(int(e0)) != _find(int(e1)):
            return True
    return False


class ProlixBatchSimulation:
    """Multi-replica and heterogeneous OpenMM batch simulation frontend.

    Dispatches `B` OpenMM `System`s (homogeneous replicas, parameter-differentiated
    congeneric ligands / alchemical λ windows, or cross-size ragged topologies) onto
    a single compiled Pallas `EnsemblePlan` (`jax.vmap` + `MK-64` zero-stride shared
    topology broadcasting), including uniform isotropic `MonteCarloBarostat` (NPT) and
    `ReactionField` (`CutoffPeriodic`), with automatic stateful `ProlixContext` fallback
    only when custom Lepton forces, virtual sites, restraints, `CutoffNonPeriodic`,
    `LJPME`, or `precision == "double"` are active.
    """

    def __init__(
        self,
        systems: Sequence[openmm.System | str],
        integrators: openmm.Integrator | str | Sequence[openmm.Integrator | str],
        topologies: Topology | Sequence[Topology | None] | None = None,
        platform: openmm.Platform | None = None,  # noqa: ARG002
        platformProperties: dict[str, str] | None = None,
        config: ProlixOpenMMConfig | None = None,
        positions: Sequence[Any] | None = None,
        positions_list: Sequence[Any] | None = None,
    ) -> None:
        if not isinstance(systems, Sequence) or isinstance(systems, (str, bytes)) or len(systems) == 0:
            raise ValueError("ProlixBatchSimulation requires a non-empty sequence of OpenMM Systems.")

        if positions is not None and positions_list is not None:
            raise ValueError("Pass at most one of `positions` or `positions_list`, not both.")
        init_positions = positions_list if positions_list is not None else positions

        omm_mod = cast("Any", openmm)
        self.systems: list[openmm.System] = []
        raw_cache: dict[int | str, openmm.System] = {}
        for s in systems:
            if isinstance(s, str):
                if s not in raw_cache:
                    raw_cache[s] = omm_mod.XmlSerializer.deserialize(s)
                self.systems.append(raw_cache[s])
            else:
                self.systems.append(s)

        b_count = len(self.systems)

        if isinstance(integrators, (openmm.Integrator, str)):
            self.integrators: list[openmm.Integrator] = [
                _clone_integrator(integrators) for _ in range(b_count)
            ]
        else:
            if len(integrators) != b_count:
                raise ValueError(
                    f"Expected {b_count} integrators to match systems, got {len(integrators)}."
                )
            self.integrators = [
                omm_mod.XmlSerializer.deserialize(ig) if isinstance(ig, str) else ig
                for ig in integrators
            ]

        if topologies is None:
            self.topologies: list[Any] = [None] * b_count
        elif isinstance(topologies, Sequence) and not hasattr(topologies, "atoms"):
            if len(topologies) != b_count:
                raise ValueError(
                    f"Expected {b_count} topologies to match systems, got {len(topologies)}."
                )
            self.topologies = list(topologies)
        else:
            self.topologies = [topologies] * b_count

        if config is not None:
            self.config = config
        elif platformProperties is not None:
            self.config = parse_platform_properties(platformProperties)
        else:
            self.config = ProlixOpenMMConfig()

        self._dtype = jnp.float64 if self.config.precision == "double" else jnp.float32
        self.currentStep: int = 0
        self._seed: int = int(self.config.deterministic_seed)

        # Deduplicate conversion across identical OpenMM System instances so
        # homogeneous B-replica batches convert once and share topology arrays (MK-64).
        conv_cache: dict[int, ConvertedOpenMMSystem] = {}
        self._converted_systems: list[ConvertedOpenMMSystem] = []
        raw_bundles: list[MolecularBundle] = []
        for sys_obj in self.systems:
            sid = id(sys_obj)
            if sid not in conv_cache:
                eff_conv_cfg = (
                    dataclasses.replace(self.config, anisotropic_barostat=True)
                    if (
                        not self.config.anisotropic_barostat
                        and _system_has_anisotropic_or_membrane_barostat(sys_obj)
                    )
                    else self.config
                )
                conv_cache[sid] = from_openmm_system(sys_obj, config=eff_conv_cfg)
            conv_i = conv_cache[sid]
            self._converted_systems.append(conv_i)
            b_i = dataclasses.replace(
                conv_i.bundle,
                positions=jnp.array(conv_i.bundle.positions, copy=True),
            )
            raw_bundles.append(b_i)

        if self.config.use_size_buckets and b_count > 1:
            self._bundles: list[MolecularBundle] = _align_bundles_to_common_bucket(raw_bundles)
            aligned_cs: list[ConvertedOpenMMSystem] = []
            for cs_i, b_aligned in zip(self._converted_systems, self._bundles, strict=True):
                if b_aligned.shape_spec == cs_i.bundle.shape_spec:
                    aligned_cs.append(cs_i)
                    continue
                aligned_cs.append(
                    dataclasses.replace(
                        cs_i,
                        bundle=b_aligned,
                        bond_force_groups=_pad_axis0(
                            cs_i.bond_force_groups, int(b_aligned.bond_mask.shape[0]), 0
                        ),
                        angle_force_groups=_pad_axis0(
                            cs_i.angle_force_groups, int(b_aligned.angle_mask.shape[0]), 0
                        ),
                        dihedral_force_groups=_pad_axis0(
                            cs_i.dihedral_force_groups, int(b_aligned.dihedral_mask.shape[0]), 0
                        ),
                        cmap_force_groups=_pad_axis0(
                            cs_i.cmap_force_groups, int(b_aligned.cmap_mask.shape[0]), 0
                        ),
                    )
                )
            self._converted_systems = aligned_cs
        else:
            self._bundles = raw_bundles

        # Retain unshifted base bundles so setParameter() & jax.vmap autodiff apply
        # NonbondedParameterOffsets cleanly without double-shifting or duplicating stride-0 topology.
        self._base_bundles: list[MolecularBundle] = list(self._bundles)
        self._parameters_per_replica: list[dict[str, float]] = [
            dict(cs.default_parameters) for cs in self._converted_systems
        ]

        self._n_atoms_per_replica: list[int] = [
            sys_obj.getNumParticles() for sys_obj in self.systems
        ]
        self._positions_akma: list[jnp.ndarray] = [
            jnp.zeros((n_i, 3), dtype=self._dtype) for n_i in self._n_atoms_per_replica
        ]
        self._velocities_akma: list[jnp.ndarray] = [
            jnp.zeros((n_i, 3), dtype=self._dtype) for n_i in self._n_atoms_per_replica
        ]
        self._boxes_akma: list[jnp.ndarray | None] = [
            jnp.asarray(b.box, dtype=self._dtype) if b.shape_spec.has_pbc else None
            for b in self._bundles
        ]
        self._velocities_initialized: bool = False
        self._vel_obs = _BatchVelocityObservable()
        self._contexts: list[ProlixContext] | None = None
        self.plan: Any = None
        self._per_replica_plans: list[Any] | None = None
        self._dih3_cache: dict[int, jnp.ndarray] = {}
        self._constraint_algos: list[ConstraintAlgorithm | None] | None = None

        # Apply any nonzero default global parameter offsets onto `self._bundles`
        if any(cs.nonbonded_offsets is not None for cs in self._converted_systems):
            self._sync_offset_parameters_to_bundles()

        # Extract integrator parameters (dt_fs, T_kelvin, gamma_ps)
        dt_ps_list: list[float] = []
        temp_k_list: list[float] = []
        gamma_ps_list: list[float] = []
        all_langevin_compatible = True
        for ig in self.integrators:
            ig_any = cast("Any", ig)
            dt_ps = float(
                ig_any.getStepSize().value_in_unit(unit.picoseconds)
                if unit.is_quantity(ig_any.getStepSize())
                else ig_any.getStepSize()
            )
            dt_ps_list.append(dt_ps)
            if hasattr(ig_any, "getTemperature"):
                tk = ig_any.getTemperature()
                temp_k_list.append(
                    float(tk.value_in_unit(unit.kelvin) if unit.is_quantity(tk) else tk)
                )
            else:
                temp_k_list.append(300.0)
                all_langevin_compatible = False
            if hasattr(ig_any, "getFriction"):
                gm = ig_any.getFriction()
                gamma_ps_list.append(
                    float(
                        gm.value_in_unit(unit.picoseconds**-1)
                        if unit.is_quantity(gm)
                        else gm
                    )
                )
            else:
                gamma_ps_list.append(1.0)
                all_langevin_compatible = False

        uniform_integrators = (
            all_langevin_compatible
            and self.config.integrator_scheme in ("auto", "baoab", "langevin", "lfmiddle")
            and np.allclose(dt_ps_list, dt_ps_list[0], rtol=1e-6, atol=1e-9)
            and np.allclose(temp_k_list, temp_k_list[0], rtol=1e-6, atol=1e-6)
            and np.allclose(gamma_ps_list, gamma_ps_list[0], rtol=1e-6, atol=1e-6)
        )
        self._dt_fs: float = dt_ps_list[0] * 1000.0
        self._dt_akma: float = dt_ps_list[0] * PS_TO_TAU_AKMA
        self._temperature_k: float = temp_k_list[0]
        self._gamma_ps: float = gamma_ps_list[0]

        # Inspect barostat uniformity across replicas (isotropic MonteCarloBarostat in auto/mc)
        self._pressure_atm: float | None = None
        self._barostat_interval: int | None = None
        baro_fallback = False
        has_any_barostat = any(cs.barostat is not None for cs in self._converted_systems)
        if has_any_barostat:
            if (
                not all(cs.barostat is not None for cs in self._converted_systems)
                or self.config.barostat_mode not in ("auto", "mc")
            ):
                baro_fallback = True
            else:
                baro_dicts = [cast("dict[str, Any]", cs.barostat) for cs in self._converted_systems]
                if any(bd.get("type") != "MonteCarloBarostat" for bd in baro_dicts):
                    baro_fallback = True
                else:
                    p_bars = [float(bd["pressure_bar"]) for bd in baro_dicts]
                    t_baros = [float(bd["temperature_k"]) for bd in baro_dicts]
                    freqs = [
                        int(self.config.barostat_frequency_override)
                        if self.config.barostat_frequency_override is not None
                        else int(bd["frequency"])
                        for bd in baro_dicts
                    ]
                    if (
                        not np.allclose(p_bars, p_bars[0], rtol=1e-6, atol=1e-8)
                        or not np.allclose(t_baros, self._temperature_k, rtol=1e-4, atol=1e-3)
                        or len(set(freqs)) != 1
                    ):
                        baro_fallback = True
                    elif freqs[0] > 0:
                        self._pressure_atm = float(p_bars[0] / ATM_TO_BAR)
                        self._barostat_interval = int(freqs[0])

        nb_methods = {str(cs.nonbonded_method).lower() for cs in self._converted_systems}
        self._is_reaction_field: bool = nb_methods == {"reaction_field"}
        rf_mismatch = False
        if "reaction_field" in nb_methods:
            if len(nb_methods) > 1:
                rf_mismatch = True
            else:
                rf_eps = [float(cs.reaction_field_dielectric) for cs in self._converted_systems]
                if not all(
                    (math.isinf(x) and math.isinf(rf_eps[0]))
                    or (
                        math.isfinite(x)
                        and math.isfinite(rf_eps[0])
                        and math.isclose(x, rf_eps[0], rel_tol=1e-6, abs_tol=1e-8)
                    )
                    for x in rf_eps
                ):
                    rf_mismatch = True

        # Morton SoA runner handles NPT/RF only under Lorentz-Berthelot and non-dense/non-flash engines
        npt_or_rf_unsupported_on_plan = bool(
            (self._pressure_atm is not None or self._is_reaction_field)
            and (
                self.config.lj_combining_rule != "lorentz_berthelot"
                or self.config.nonbonded_engine in ("dense", "flash", "neighbor_list")
            )
        )

        tri_kernels_ok = _triclinic_morton_kernels_supported()
        needs_context_fallback = (
            not uniform_integrators
            or self.config.precision == "double"
            or bool(getattr(self.config, "use_lj_pme", False))
            or baro_fallback
            or rf_mismatch
            or npt_or_rf_unsupported_on_plan
            or any(
                bool(cs.custom_forces)
                or bool(cs.position_restraints)
                or bool(cs.virtual_sites is not None)
                or (
                    cs.is_triclinic
                    and (
                        not tri_kernels_ok
                        or self.config.nonbonded_engine in ("dense", "flash", "neighbor_list")
                        or cs.bundle.box is None
                        or not _is_reduced_triclinic_box_np(
                            np.asarray(cs.bundle.box, dtype=np.float64)
                        )
                    )
                )
                or str(cs.nonbonded_method).lower() in ("cutoff", "lj_pme", "cutoffnonperiodic")
                or (
                    cs.has_pbc
                    and not bool(cs.exceptions_use_pbc)
                    and _converted_system_has_intermolecular_exceptions(cs)
                )
                for cs in self._converted_systems
            )
        )

        self._boxes_match: bool = self._compute_boxes_match()

        self._contexts: list[ProlixContext] | None = None
        self.plan: Any = None
        if needs_context_fallback:
            self._contexts = [
                ProlixContext(self.systems[i], self.integrators[i], config=self.config)
                for i in range(b_count)
            ]
        else:
            self._rebuild_plan()

        if init_positions is not None:
            self.setPositions(positions_list=init_positions)

    def _compute_boxes_match(self) -> bool:
        """Return True when all periodic replicas share matching box dimensions & cutoffs."""
        if len(self._bundles) <= 1 or not all(b.shape_spec.has_pbc for b in self._bundles):
            return True
        ref_box = np.asarray(self._bundles[0].box, dtype=np.float64)
        ref_cut = float(np.asarray(self._bundles[0].cutoff_distance))
        for b_other in self._bundles[1:]:
            if not np.allclose(
                np.asarray(b_other.box, dtype=np.float64), ref_box, rtol=0.0, atol=1e-5
            ) or not np.isclose(
                float(np.asarray(b_other.cutoff_distance)), ref_cut, rtol=0.0, atol=1e-5
            ):
                return False
        return True

    @property
    def batch_size(self) -> int:
        """Number of replicas / systems in the batch."""
        return len(self.systems)

    @property
    def n_replicas(self) -> int:
        """Alias for `batch_size`."""
        return len(self.systems)

    @property
    def bundles(self) -> list[MolecularBundle]:
        """Bucket-aligned `MolecularBundle` sequence backing this batch."""
        return list(self._bundles)

    def _effective_plan_engine_and_pallas(self) -> tuple[str, bool, bool]:
        """Resolve `(nonbonded_engine, use_pallas, use_pallas_pme)` for `EnsemblePlan`."""
        periodic = all(b.shape_spec.has_pbc for b in self._bundles)
        implicit = any(b.shape_spec.has_implicit_solvent for b in self._bundles)
        any_triclinic = any(cs.is_triclinic for cs in self._converted_systems)
        need_morton = bool(
            periodic
            and not implicit
            and (
                self._pressure_atm is not None
                or self._is_reaction_field
                or self.config.precision == "mixed"
                or any_triclinic
            )
            and self.config.nonbonded_engine in ("auto", "morton_tile_pairs")
        )
        eff_engine = "morton_tile_pairs" if need_morton else self.config.nonbonded_engine
        eff_pallas = bool(
            self.config.use_pallas
            or eff_engine == "morton_tile_pairs"
            or (
                periodic
                and not implicit
                and eff_engine == "auto"
                and jax.default_backend() != "cpu"
            )
        )
        eff_pallas_pme = bool(self.config.use_pallas_pme or eff_pallas)
        return eff_engine, eff_pallas, eff_pallas_pme

    def _build_knob_surface(
        self,
        eff_engine: str,
        eff_pallas: bool,
        eff_pallas_pme: bool,
        replica_idx: int = 0,
    ) -> KnobSurface:
        """Construct composite `KnobSurface` carrying RF, PME, dispersion, and barostat knobs."""
        cs0 = self._converted_systems[replica_idx]
        eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
        ewald_tol = (
            cs0.ewald_error_tolerance
            if cs0.ewald_error_tolerance is not None
            else self.config.ewald_error_tolerance
        )
        use_disp = bool(
            self.config.use_dispersion_correction and cs0.use_dispersion_correction
        )
        nb_ctrl = NonbondedControls(
            nonbonded_method=str(cs0.nonbonded_method),
            cutoff_angstrom=(
                float(cs0.nonbonded_cutoff)
                if math.isfinite(float(cs0.nonbonded_cutoff))
                else 9.0
            ),
            switch_width_angstrom=float(cs0.lj_switch_width),
            ewald_error_tolerance=float(ewald_tol),
            pme_alpha=float(eff_alpha) if eff_alpha is not None else 0.0,
            pme_grid_dims=cs0.pme_grid_dims,
            dispersion_correction=use_disp,
            reaction_field_dielectric=float(cs0.reaction_field_dielectric),
            lj_combining_rule=str(cs0.lj_combining_rule),
            exceptions_use_pbc=bool(cs0.exceptions_use_pbc),
            dispersion_follows_parameters=bool(cs0.dispersion_follows_parameters),
        )
        baro_ctrl = BarostatControls(
            pressure_atm=self._pressure_atm,
            barostat_method="mc",
            barostat_interval=int(self._barostat_interval or 25),
        )
        pallas_ctrl = PallasControls(
            use_pallas=eff_pallas,
            use_pallas_pme=eff_pallas_pme,
            gb_use_neighbor_list=bool(self.config.gb_use_neighbor_list),
            precision=str(self.config.precision),
            nonbonded_engine=eff_engine,
            pallas_interpret=bool(self.config.pallas_interpret),
            nb_k_splits=self.config.nb_k_splits,
            tp_group=self.config.tp_group,
            deterministic_forces=bool(self.config.deterministic_forces),
            execution_mode=str(self.config.execution_mode),
            use_size_buckets=bool(self.config.use_size_buckets),
        )
        return KnobSurface(nonbonded=nb_ctrl, barostat=baro_ctrl, pallas=pallas_ctrl)

    def _bundles_for_plan(self) -> list[MolecularBundle]:
        """Prepare `MolecularBundle` list for `EnsemblePlan` (`dihedral_params` `(..., 3)` with `stride=0` sharing)."""
        out: list[MolecularBundle] = []
        for b in self._bundles:
            dp = b.dihedral_params
            if dp is not None and dp.shape[-1] == 4:
                key = id(dp)
                if key not in self._dih3_cache:
                    self._dih3_cache[key] = dp[..., :3]
                out.append(dataclasses.replace(b, dihedral_params=self._dih3_cache[key]))
            else:
                out.append(b)
        return out

    def _rebuild_plan(self) -> None:
        """Construct `EnsemblePlan` from current `self._bundles`."""
        plan_cls = _ensemble_plan_mod.EnsemblePlan
        eff_engine, eff_pallas, eff_pallas_pme = self._effective_plan_engine_and_pallas()
        ks = self._build_knob_surface(eff_engine, eff_pallas, eff_pallas_pme)
        cmm_freq = (
            int(self.config.cmm_frequency)
            if self.config.cmm_frequency is not None
            else (
                int(self._converted_systems[0].com_motion_frequency)
                if (
                    self.config.remove_com_motion
                    and self._converted_systems[0].remove_com_motion
                )
                else 0
            )
        )
        plan_bundles = self._bundles_for_plan()
        self.plan = plan_cls.from_bundles(
            plan_bundles,
            use_pallas=eff_pallas,
            use_pallas_pme=eff_pallas_pme,
            nonbonded_engine=eff_engine,
            precision=self.config.precision,
            nb_k_splits=self.config.nb_k_splits,
            tp_group=self.config.tp_group,
            cmm_frequency=cmm_freq,
            deterministic_forces=self.config.deterministic_forces,
            gb_use_neighbor_list=self.config.gb_use_neighbor_list,
            pressure_atm=self._pressure_atm,
            barostat_interval=self._barostat_interval,
            knob_surface=ks,
        )
        self._per_replica_plans = None

    def _sync_offset_parameters_to_bundles(self) -> None:
        """Evaluate `q(λ), σ(λ), ε(λ)` & exception offsets onto `self._bundles` preserving `stride=0` topology."""
        for i, cs in enumerate(self._converted_systems):
            offsets = cs.nonbonded_offsets
            if offsets is None or len(offsets.param_names) == 0:
                continue
            b_base = self._base_bundles[i]
            b_cur = self._bundles[i]
            gp_map = {
                k: jnp.asarray(v, dtype=b_base.charges.dtype)
                for k, v in self._parameters_per_replica[i].items()
            }
            n_pad = int(b_base.charges.shape[0])
            q_eff, sig_eff, eps_eff = _apply_particle_offsets(
                b_base.charges,
                b_base.sigmas,
                b_base.epsilons,
                offsets,
                gp_map,
                n_pad,
            )
            qp_eff, esig_eff, eeps_eff = _apply_exception_offsets(
                b_base.exception_chargeprods,
                b_base.exception_sigmas,
                b_base.exception_epsilons,
                offsets,
                gp_map,
            )
            self._bundles[i] = dataclasses.replace(
                b_cur,
                charges=q_eff,
                sigmas=sig_eff,
                epsilons=eps_eff,
                exception_chargeprods=qp_eff,
                exception_sigmas=esig_eff,
                exception_epsilons=eeps_eff,
            )
        if self.plan is not None or self._per_replica_plans is not None:
            plan_bundles = self._bundles_for_plan()
            if self.plan is not None:
                self.plan.bundles = plan_bundles
            if self._per_replica_plans is not None:
                for i, p_i in enumerate(self._per_replica_plans):
                    p_i.bundles = [plan_bundles[i]]

    def getParameters(self) -> list[dict[str, float]]:  # noqa: N802
        """Return per-replica OpenMM global parameter dictionaries."""
        if self._contexts is not None:
            return [dict(ctx._parameters) for ctx in self._contexts]
        return [dict(p) for p in self._parameters_per_replica]

    def setParameter(  # noqa: N802
        self,
        name: str,
        values: float | Sequence[float] | np.ndarray | jnp.ndarray,
    ) -> None:
        """Set global parameter `name` to a scalar or length-`B` vector across replicas."""
        for i, p_dict in enumerate(self._parameters_per_replica):
            if name not in p_dict:
                raise openmm.OpenMMException(
                    f"Called setParameter() with invalid parameter name: {name} (replica {i})"
                )

        raw_v: Any = getattr(values, "_value", values) if unit.is_quantity(values) else values

        if isinstance(raw_v, (int, float, np.floating, np.integer)):
            per_rep = [float(raw_v)] * self.batch_size
        elif isinstance(raw_v, (np.ndarray, jnp.ndarray)):
            arr = np.asarray(raw_v, dtype=np.float64)
            if arr.ndim == 0:
                per_rep = [float(arr)] * self.batch_size
            elif arr.shape == (self.batch_size,):
                per_rep = [float(x) for x in arr]
            else:
                raise ValueError(
                    f"Expected scalar or shape ({self.batch_size},) for parameter {name!r}, got {arr.shape}."
                )
        elif isinstance(raw_v, Sequence):
            if len(raw_v) != self.batch_size:
                raise ValueError(
                    f"Expected {self.batch_size} values for parameter {name!r}, got {len(raw_v)}."
                )
            per_rep = [
                float(getattr(v, "_value", v) if unit.is_quantity(v) else v)
                for v in raw_v
            ]
        else:
            per_rep = [float(cast("Any", raw_v))] * self.batch_size

        for i, val_f in enumerate(per_rep):
            self._parameters_per_replica[i][name] = val_f
            if self._contexts is not None:
                self._contexts[i].setParameter(name, val_f)

        if any(cs.nonbonded_offsets is not None for cs in self._converted_systems):
            self._sync_offset_parameters_to_bundles()

    def _ensure_contexts(self) -> list[ProlixContext]:
        """Lazily construct `ProlixContext`s synced with current batch state when needed."""
        if self._contexts is None:
            self._contexts = [
                ProlixContext(
                    self.systems[i],
                    _clone_integrator(self.integrators[i]),
                    config=self.config,
                )
                for i in range(self.batch_size)
            ]
        for i, ctx in enumerate(self._contexts):
            box_i = self._boxes_akma[i]
            if box_i is not None:
                bv_nm = np.asarray(box_i, dtype=np.float64) * ANGSTROM_TO_NM
                ctx.setPeriodicBoxVectors(
                    openmm.Vec3(float(bv_nm[0, 0]), float(bv_nm[0, 1]), float(bv_nm[0, 2])) * unit.nanometer,
                    openmm.Vec3(float(bv_nm[1, 0]), float(bv_nm[1, 1]), float(bv_nm[1, 2])) * unit.nanometer,
                    openmm.Vec3(float(bv_nm[2, 0]), float(bv_nm[2, 1]), float(bv_nm[2, 2])) * unit.nanometer,
                )
            for k, v in self._parameters_per_replica[i].items():
                if ctx._parameters.get(k) != v:
                    ctx.setParameter(k, v)
            ctx._positions = jnp.asarray(self._positions_akma[i], dtype=ctx._dtype)
            ctx._velocities = jnp.asarray(self._velocities_akma[i], dtype=ctx._dtype)
            ctx._velocities_initialized = self._velocities_initialized
            ctx._step_count = self.currentStep
            ctx._time = self.currentStep * self._dt_akma
            ctx._init_neighbor_list(ctx._positions)
            ctx._morton_nl_dirty = True
            ctx._morton_forces_dirty = True
        return self._contexts

    def setPositions(  # noqa: N802
        self,
        positions: Sequence[Any] | None = None,
        positions_list: Sequence[Any] | None = None,
    ) -> None:
        """Set positions (OpenMM nm or `unit.Quantity`) in place without rebuilding unchanged plans."""
        if positions is not None and positions_list is not None:
            raise ValueError("Pass at most one of `positions` or `positions_list`, not both.")
        pos_seq = positions_list if positions_list is not None else positions
        if pos_seq is None:
            raise ValueError("Expected `positions` or `positions_list` sequence.")
        if len(pos_seq) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} position arrays, got {len(pos_seq)}."
            )

        old_specs = [b.shape_spec for b in self._bundles]
        for i, pos_i in enumerate(pos_seq):
            if unit.is_quantity(pos_i):
                arr_ang = np.asarray(pos_i.value_in_unit(unit.angstrom), dtype=np.float64)
            else:
                arr_ang = np.asarray(pos_i, dtype=np.float64) * NM_TO_ANGSTROM
            n_i = self._n_atoms_per_replica[i]
            if arr_ang.shape != (n_i, 3):
                raise ValueError(
                    f"Replica {i} expected positions shape ({n_i}, 3), got {arr_ang.shape}."
                )
            pos_akma = jnp.asarray(arr_ang, dtype=self._dtype)
            self._positions_akma[i] = pos_akma

            b_old = self._bundles[i]
            pad_len = int(b_old.positions.shape[0])
            pos_f32 = _pad_positions_to_bucket(
                jnp.asarray(arr_ang, dtype=jnp.float32),
                pad_len,
                b_old.box if b_old.shape_spec.has_pbc else None,
            )
            self._bundles[i] = dataclasses.replace(b_old, positions=pos_f32)
            self._base_bundles[i] = dataclasses.replace(self._base_bundles[i], positions=pos_f32)
            if self._contexts is not None:
                self._contexts[i].setPositions(pos_i)

        if self.plan is not None:
            new_specs = [b.shape_spec for b in self._bundles]
            if new_specs == old_specs:
                plan_bundles = self._bundles_for_plan()
                self.plan.bundles = plan_bundles
                if self._per_replica_plans is not None:
                    for i, p_i in enumerate(self._per_replica_plans):
                        p_i.bundles = [plan_bundles[i]]
            else:
                self._rebuild_plan()

    def setVelocities(  # noqa: N802
        self,
        velocities: Sequence[Any] | None = None,
        velocities_list: Sequence[Any] | None = None,
    ) -> None:
        """Set velocities (OpenMM nm/ps or `unit.Quantity`) for all `B` replicas."""
        if velocities is not None and velocities_list is not None:
            raise ValueError("Pass at most one of `velocities` or `velocities_list`, not both.")
        vel_seq = velocities_list if velocities_list is not None else velocities
        if vel_seq is None:
            raise ValueError("Expected `velocities` or `velocities_list` sequence.")
        if len(vel_seq) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} velocity arrays, got {len(vel_seq)}."
            )
        for i, vel_i in enumerate(vel_seq):
            if unit.is_quantity(vel_i):
                arr_omm = np.asarray(
                    vel_i.value_in_unit(unit.nanometer / unit.picosecond), dtype=np.float64
                )
            else:
                arr_omm = np.asarray(vel_i, dtype=np.float64)
            n_i = self._n_atoms_per_replica[i]
            if arr_omm.shape != (n_i, 3):
                raise ValueError(
                    f"Replica {i} expected velocities shape ({n_i}, 3), got {arr_omm.shape}."
                )
            self._velocities_akma[i] = jnp.asarray(
                arr_omm * VELOCITY_OMM_TO_AKMA, dtype=self._dtype
            )
            if self._contexts is not None:
                self._contexts[i].setVelocities(vel_i)
        self._velocities_initialized = True

    def setVelocitiesToTemperature(  # noqa: N802
        self,
        temperature: Any,
        randomSeed: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize Maxwell-Boltzmann velocities across all `B` replicas."""
        temp_k = float(
            temperature.value_in_unit(unit.kelvin)
            if unit.is_quantity(temperature)
            else temperature
        )
        if temp_k < 0.0:
            raise ValueError(f"Temperature must be non-negative, got {temp_k}.")

        if randomSeed is None:
            seeds = [self._seed + i for i in range(self.batch_size)]
        elif isinstance(randomSeed, int):
            self._seed = int(randomSeed)
            seeds = [int(randomSeed) + i for i in range(self.batch_size)]
        else:
            if len(randomSeed) != self.batch_size:
                raise ValueError(
                    f"Expected {self.batch_size} seeds, got {len(randomSeed)}."
                )
            seeds = [int(s) for s in randomSeed]
            self._seed = seeds[0]

        if self._contexts is not None:
            for i, ctx in enumerate(self._contexts):
                ctx.setVelocitiesToTemperature(temp_k, seeds[i])
                self._velocities_akma[i] = ctx._velocities
            self._velocities_initialized = True
            return

        kt = BOLTZMANN_KCAL * temp_k
        for i in range(self.batch_size):
            n_i = self._n_atoms_per_replica[i]
            masses = jnp.asarray(self._bundles[i].masses[:n_i], dtype=self._dtype)
            active = masses > 0.0
            key = jax.random.PRNGKey(seeds[i])
            sigma_v = jnp.where(active, jnp.sqrt(kt / jnp.maximum(masses, 1e-12)), 0.0)
            v = jax.random.normal(key, shape=(n_i, 3), dtype=self._dtype) * sigma_v[:, None]
            if self.config.remove_com_motion:
                m_col = jnp.where(active, masses, 0.0)[:, None]
                m_tot = jnp.sum(m_col)
                v_com = jnp.sum(m_col * v, axis=0, keepdims=True) / jnp.maximum(m_tot, 1e-12)
                v = jnp.where(active[:, None], v - v_com, 0.0)
            self._velocities_akma[i] = v
        self._velocities_initialized = True

    def step(self, steps: int) -> None:
        """Advance all `B` replicas by `steps` MD steps on `EnsemblePlan` (`run_mode='inference'`)."""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError(f"steps must be a non-negative integer, got {steps!r}.")
        if steps == 0:
            return

        if self._contexts is not None:
            for ctx in self._contexts:
                ctx.step(steps)
            for i, ctx in enumerate(self._contexts):
                self._positions_akma[i] = ctx._positions
                self._velocities_akma[i] = ctx._velocities
                if ctx._morton_box_diag is not None:
                    self._boxes_akma[i] = (
                        ctx._morton_box_diag.astype(self._dtype)
                        if ctx._morton_box_diag.ndim == 2
                        else jnp.diag(ctx._morton_box_diag.astype(self._dtype))
                    )
                elif ctx._box_vectors is not None:
                    self._boxes_akma[i] = ctx._box_vectors.astype(self._dtype)
            self.currentStep += steps
            self._velocities_initialized = True
            return

        eff_engine, eff_pallas, eff_pallas_pme = self._effective_plan_engine_and_pallas()
        periodic = all(b.shape_spec.has_pbc for b in self._bundles)
        implicit = any(b.shape_spec.has_implicit_solvent for b in self._bundles)
        need_morton = bool(
            self._pressure_atm is not None
            or self._is_reaction_field
            or eff_engine == "morton_tile_pairs"
        )

        if eff_engine == "dense":
            use_nl: bool | None = False
            use_flash: bool | None = False
        elif eff_engine == "flash" or (
            periodic and not implicit and not self._boxes_match and not need_morton
        ):
            use_nl = False
            use_flash = bool(periodic and not implicit)
        elif eff_engine in ("neighbor_list", "morton_tile_pairs"):
            use_nl = bool(periodic and not implicit)
            use_flash = False
        else:
            use_nl = None
            use_flash = None

        kt_akma = BOLTZMANN_KCAL * self._temperature_k
        step_seed = self._seed + self.currentStep
        run_mode = "inference" if need_morton else self.config.execution_mode
        obs_arg = {"_final_vel": self._vel_obs}

        extra_run_kw: dict[str, Any] = {}
        if self._pressure_atm is not None:
            extra_run_kw["pressure_atm"] = self._pressure_atm
            extra_run_kw["barostat_interval"] = self._barostat_interval
        if "initial_velocities" in _ENSEMBLE_PLAN_RUN_PARAMS and self._velocities_initialized:
            extra_run_kw["initial_velocities"] = self._velocities_akma

        if need_morton and not self._boxes_match and self.batch_size > 1:
            # Replicas have distinct boxes (e.g. NPT divergence or cross-shape cubic/dodecahedron/octahedron):
            # run 1-bundle Morton SoA plans per replica with zero ProlixContext fallback.
            plan_bundles_pre = self._bundles_for_plan()
            if self._per_replica_plans is None:
                plan_cls = _ensemble_plan_mod.EnsemblePlan
                self._per_replica_plans = [
                    plan_cls.from_bundle(
                        plan_bundles_pre[i],
                        use_pallas=eff_pallas,
                        use_pallas_pme=eff_pallas_pme,
                        nonbonded_engine=eff_engine,
                        precision=self.config.precision,
                        nb_k_splits=self.config.nb_k_splits,
                        tp_group=self.config.tp_group,
                        deterministic_forces=self.config.deterministic_forces,
                        pressure_atm=self._pressure_atm,
                        barostat_interval=self._barostat_interval,
                        knob_surface=self._build_knob_surface(
                            eff_engine, eff_pallas, eff_pallas_pme, replica_idx=i
                        ),
                    )
                    for i in range(self.batch_size)
                ]
            traj_list = []
            for i, p_i in enumerate(self._per_replica_plans):
                p_i.bundles = [plan_bundles_pre[i]]
                extra_kw_i = dict(extra_run_kw)
                if "initial_velocities" in extra_kw_i:
                    extra_kw_i["initial_velocities"] = [self._velocities_akma[i]]
                tr_i = p_i.run(
                    steps,
                    dt=self._dt_fs,
                    kT=kt_akma,
                    seed=step_seed + i,
                    observables=obs_arg,
                    dt_unit="fs",
                    gamma=self._gamma_ps,
                    run_mode="inference",
                    use_neighbor_list=True,
                    nl_update_every=self.config.nl_update_every,
                    use_flash_forces=False,
                    nonbonded_engine=eff_engine,
                    use_pallas=eff_pallas,
                    use_pallas_pme=eff_pallas_pme,
                    **extra_kw_i,
                )
                traj_list.append(tr_i[0] if isinstance(tr_i, list) else tr_i)
        else:
            try:
                trajs = self.plan.run(
                    steps,
                    dt=self._dt_fs,
                    kT=kt_akma,
                    seed=step_seed,
                    observables=obs_arg,
                    dt_unit="fs",
                    gamma=self._gamma_ps,
                    run_mode=run_mode,
                    use_neighbor_list=use_nl,
                    nl_update_every=self.config.nl_update_every,
                    use_flash_forces=use_flash,
                    nonbonded_engine=eff_engine,
                    use_pallas=eff_pallas,
                    use_pallas_pme=eff_pallas_pme,
                    **extra_run_kw,
                )
            except ValueError:
                trajs = self.plan.run(
                    steps,
                    dt=self._dt_fs,
                    kT=kt_akma,
                    seed=step_seed,
                    observables=obs_arg,
                    dt_unit="fs",
                    gamma=self._gamma_ps,
                    run_mode=run_mode,
                    use_neighbor_list=False,
                    nl_update_every=self.config.nl_update_every,
                    use_flash_forces=bool(periodic and not implicit),
                    use_pallas=eff_pallas,
                    use_pallas_pme=eff_pallas_pme,
                    **extra_run_kw,
                )
            traj_list = [trajs] if not isinstance(trajs, list) else trajs

        for i, tr in enumerate(traj_list):
            n_i = self._n_atoms_per_replica[i]
            final_pos = tr.positions[-1, :n_i, :]
            self._positions_akma[i] = jnp.asarray(final_pos, dtype=self._dtype)

            obs_vals = getattr(tr, "observable_values", None) or {}
            if "_final_vel" in obs_vals:
                self._velocities_akma[i] = jnp.asarray(
                    obs_vals["_final_vel"][:n_i, :], dtype=self._dtype
                )
            elif getattr(tr, "velocities", None) is not None:
                self._velocities_akma[i] = jnp.asarray(
                    tr.velocities[-1, :n_i, :], dtype=self._dtype
                )

            b_old = self._bundles[i]
            b_base_old = self._base_bundles[i]
            new_box_f32 = b_old.box
            if "box" in obs_vals and obs_vals["box"] is not None:
                raw_bx = jnp.asarray(obs_vals["box"], dtype=jnp.float32)
                new_box_f32 = jnp.diag(raw_bx) if raw_bx.ndim == 1 else raw_bx
                self._boxes_akma[i] = new_box_f32.astype(self._dtype)

            pad_len = int(b_old.positions.shape[0])
            pos_f32 = _pad_positions_to_bucket(
                jnp.asarray(final_pos, dtype=jnp.float32),
                pad_len,
                new_box_f32 if b_old.shape_spec.has_pbc else None,
            )
            self._bundles[i] = dataclasses.replace(
                b_old, positions=pos_f32, box=new_box_f32
            )
            self._base_bundles[i] = dataclasses.replace(
                b_base_old, positions=pos_f32, box=new_box_f32
            )

        self._boxes_match = self._compute_boxes_match()
        self.plan.bundles = self._bundles_for_plan()
        self.currentStep += steps
        self._velocities_initialized = True

    def _sync_from_contexts_if_present(self) -> None:
        """Synchronize host-tracked arrays/parameters from `self._contexts` without D2H copies."""
        if self._contexts is None:
            return
        for i, ctx in enumerate(self._contexts):
            n_i = self._n_atoms_per_replica[i]
            self._positions_akma[i] = jnp.asarray(ctx._positions[:n_i], dtype=self._dtype)
            self._velocities_akma[i] = jnp.asarray(ctx._velocities[:n_i], dtype=self._dtype)
            if ctx._morton_box_diag is not None:
                bx = (
                    ctx._morton_box_diag.astype(self._dtype)
                    if ctx._morton_box_diag.ndim == 2
                    else jnp.diag(ctx._morton_box_diag.astype(self._dtype))
                )
            elif ctx._box_vectors is not None:
                bx = ctx._box_vectors.astype(self._dtype)
            else:
                bx = None
            self._boxes_akma[i] = bx
            self._parameters_per_replica[i] = dict(ctx._parameters)
            if bx is not None and self._bundles[i].shape_spec.has_pbc:
                bx_f32 = bx.astype(jnp.float32)
                self._bundles[i] = dataclasses.replace(self._bundles[i], box=bx_f32)
                self._base_bundles[i] = dataclasses.replace(self._base_bundles[i], box=bx_f32)
        self.currentStep = int(self._contexts[0]._step_count)
        if any(bool(c._velocities_initialized) for c in self._contexts):
            self._velocities_initialized = True

    def _get_replica_constraint_algo(self, idx: int) -> ConstraintAlgorithm | None:
        """Build and cache `ConstraintAlgorithm` (SETTLE + SHAKE) for replica `idx`."""
        if self._constraint_algos is None:
            algos: list[ConstraintAlgorithm | None] = []
            for cs in self._converted_systems:
                b = cs.bundle
                n_w = int(b.n_waters) if b.n_waters is not None else 0
                n_c = int(b.n_constraints) if b.n_constraints is not None else 0
                if n_w <= 0 and n_c <= 0:
                    algos.append(None)
                    continue
                w_idx = (
                    b.water_indices[:n_w]
                    if b.water_indices is not None and n_w > 0
                    else None
                )
                c_pairs = (
                    b.constraint_pairs[:n_c]
                    if b.constraint_pairs is not None and n_c > 0
                    else None
                )
                c_lens = (
                    b.constraint_lengths[:n_c]
                    if b.constraint_lengths is not None and n_c > 0
                    else None
                )
                algos.append(
                    make_constraint(
                        water_indices=w_idx,
                        constraint_pairs=c_pairs,
                        constraint_lengths=c_lens,
                        r_OH=cs.system_metadata.get("water_r_OH_akma"),
                        r_HH=cs.system_metadata.get("water_r_HH_akma"),
                        settle_velocity_tol=self.config.settle_velocity_tol,
                        shake_iterations=self.config.shake_iterations,
                        constraint_tolerance=self.config.constraint_tolerance,
                    )
                )
            self._constraint_algos = algos
        return self._constraint_algos[idx]

    def _apply_minimized_positions_to_replica(
        self, idx: int, pos_akma: jnp.ndarray
    ) -> None:
        """Reconstruct virtual sites, project constraints, zero velocities, and sync replica `idx`."""
        cs = self._converted_systems[idx]
        b_old = self._bundles[idx]
        n_i = self._n_atoms_per_replica[idx]
        p_cur = jnp.asarray(pos_akma[:n_i], dtype=self._dtype)
        p_cur = cs.compute_virtual_sites(p_cur).astype(self._dtype)

        algo = self._get_replica_constraint_algo(idx)
        if algo is not None:
            masses = b_old.masses[:n_i].astype(p_cur.dtype)
            box_i = (
                self._boxes_akma[idx]
                if (b_old.shape_spec.has_pbc and cs.exceptions_use_pbc)
                else None
            )
            p_cur = algo.apply_positions(
                p_cur, p_cur, masses[:, None], box_i
            ).astype(self._dtype)
            p_cur = cs.compute_virtual_sites(p_cur).astype(self._dtype)

        self._positions_akma[idx] = p_cur
        self._velocities_akma[idx] = jnp.zeros((n_i, 3), dtype=self._dtype)

        pad_len = int(b_old.positions.shape[0])
        pos_f32 = _pad_positions_to_bucket(
            p_cur.astype(jnp.float32),
            pad_len,
            b_old.box if b_old.shape_spec.has_pbc else None,
        )
        self._bundles[idx] = dataclasses.replace(b_old, positions=pos_f32)
        self._base_bundles[idx] = dataclasses.replace(self._base_bundles[idx], positions=pos_f32)

        if self._contexts is not None:
            ctx = self._contexts[idx]
            ctx._positions = jnp.asarray(p_cur, dtype=ctx._dtype)
            ctx._velocities = jnp.zeros_like(ctx._velocities)
            ctx._init_neighbor_list(ctx._positions)
            ctx._morton_nl_dirty = True
            ctx._morton_forces_dirty = True
            ctx._host_nl_dirty = True

    def _refresh_plan_bundles_after_state_update(self) -> None:
        """Sync `self.plan.bundles` and `self._per_replica_plans` after in-place state updates."""
        if self.plan is not None or self._per_replica_plans is not None:
            plan_bundles = self._bundles_for_plan()
            if self.plan is not None:
                self.plan.bundles = plan_bundles
            if self._per_replica_plans is not None:
                for i, p_i in enumerate(self._per_replica_plans):
                    p_i.bundles = [plan_bundles[i]]

    def minimizeEnergy(  # noqa: N802
        self,
        tolerance: Any = 10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations: int = 0,
        backend: str | None = None,
    ) -> None:
        """Minimize potential energy across all `B` replicas via `fire`, `lbfgs`, or `openmm_cpu`."""
        if hasattr(tolerance, "value_in_unit"):
            tol_omm = float(
                tolerance.value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
            )
        else:
            tol_omm = float(tolerance)
        tol_akma = tol_omm * FORCE_OMM_TO_AKMA

        eff_backend = (
            str(backend).lower()
            if backend is not None
            else str(self.config.minimizer_backend).lower()
        )
        if eff_backend not in ("lbfgs", "fire", "openmm_cpu"):
            raise ValueError(f"Unknown minimizer backend: {eff_backend!r}")
        if eff_backend == "lbfgs" and _jaxopt is None:
            raise ImportError("jaxopt is required for minimizer_backend='lbfgs'.")

        self._sync_from_contexts_if_present()
        max_iters_gpu = int(maxIterations) if int(maxIterations) > 0 else 500

        if eff_backend in ("fire", "lbfgs") and self._can_vmap_batch_observables(-1):
            cs0 = self._converted_systems[0]
            pos_b = jnp.stack(self._positions_akma, axis=0)
            (
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
            ) = self._stack_batch_inputs_for_vmap()
            ewald_tol = (
                cs0.ewald_error_tolerance
                if cs0.ewald_error_tolerance is not None
                else self.config.ewald_error_tolerance
            )
            eff_nb_method = _effective_cs_nonbonded_method(cs0, self.config)
            eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
            eff_lj_alpha = _resolve_cs_lj_pme_alpha(cs0, self.config)
            eff_lj_dims = _resolve_cs_lj_pme_grid_dims(
                cs0, self.config, self._boxes_akma[0]
            )
            use_disp = bool(
                self.config.use_dispersion_correction and cs0.use_dispersion_correction
            )
            nb_eng = (
                "flash"
                if (
                    cs0.has_pbc
                    and not cs0.is_triclinic
                    and not cs0.has_implicit_solvent
                    and cs0.nonbonded_offsets is None
                    and cs0.lj_combining_rule == "lorentz_berthelot"
                    and not cs0.exceptions_use_pbc
                )
                else (
                    "neighbor_list"
                    if (
                        not cs0.is_triclinic
                        and int(pos_b.shape[1]) >= 4000
                        and cs0.nonbonded_cutoff < float("inf")
                    )
                    else "dense"
                )
            )
            min_pos_b = _vmapped_batch_minimize_akma(
                pos_b,
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
                cs0.force_group_by_component,
                cs0.has_pbc,
                cs0.nonbonded_cutoff,
                eff_nb_method,
                cs0.has_nonbonded,
                use_disp,
                ewald_tol,
                eff_alpha,
                cs0.reaction_field_dielectric,
                cs0.has_implicit_solvent,
                cs0.solvent_dielectric,
                cs0.solute_dielectric,
                nb_eng,
                cs0.pme_grid_dims,
                cs0.lj_switch_width,
                self.config.custom_nonbonded_chunk_size,
                cs0.is_triclinic,
                self.config.precision == "mixed",
                cs0.gbsa_surface_tension,
                cs0.default_parameters,
                cs0.lj_combining_rule,
                cs0.exceptions_use_pbc,
                cs0.dispersion_follows_parameters,
                eff_lj_alpha,
                eff_lj_dims,
                eff_backend,
                tol_akma,
                max_iters_gpu,
            )
            for i in range(self.batch_size):
                self._apply_minimized_positions_to_replica(i, min_pos_b[i])
            self._refresh_plan_bundles_after_state_update()
            return

        if eff_backend == "openmm_cpu":
            omm_mod = cast("Any", openmm)
            platform = omm_mod.Platform.getPlatformByName("Reference")
            for i in range(self.batch_size):
                cpu_ctx = omm_mod.Context(
                    self.systems[i], omm_mod.VerletIntegrator(0.001), platform
                )
                box_i = self._boxes_akma[i]
                if box_i is not None:
                    bv_nm = np.asarray(box_i, dtype=np.float64) * ANGSTROM_TO_NM
                    cpu_ctx.setPeriodicBoxVectors(
                        openmm.Vec3(float(bv_nm[0, 0]), float(bv_nm[0, 1]), float(bv_nm[0, 2])) * unit.nanometer,
                        openmm.Vec3(float(bv_nm[1, 0]), float(bv_nm[1, 1]), float(bv_nm[1, 2])) * unit.nanometer,
                        openmm.Vec3(float(bv_nm[2, 0]), float(bv_nm[2, 1]), float(bv_nm[2, 2])) * unit.nanometer,
                    )
                for k_nm, v_val in self._parameters_per_replica[i].items():
                    with contextlib.suppress(Exception):
                        cpu_ctx.setParameter(k_nm, float(v_val))
                pos_nm_i = np.asarray(self._positions_akma[i], dtype=np.float64) * ANGSTROM_TO_NM
                cpu_ctx.setPositions(pos_nm_i * unit.nanometer)
                omm_mod.LocalEnergyMinimizer.minimize(cpu_ctx, tol_omm, int(maxIterations))
                min_pos_nm = (
                    cpu_ctx.getState(getPositions=True)
                    .getPositions(asNumpy=True)
                    .value_in_unit(unit.nanometer)
                )
                min_pos_akma = jnp.asarray(
                    np.asarray(min_pos_nm, dtype=np.float64) * NM_TO_ANGSTROM,
                    dtype=self._dtype,
                )
                self._apply_minimized_positions_to_replica(i, min_pos_akma)
            self._refresh_plan_bundles_after_state_update()
            return

        # Per-replica GPU/JAX fallback for custom Lepton forces, virtual sites, or ragged batches
        ctxs = self._ensure_contexts()
        for i, ctx in enumerate(ctxs):
            ctx.computeVirtualSites()
            ctx.applyConstraints()
            gp = ctx._get_dynamic_global_params()
            ewald_tol = (
                ctx._converted_system.ewald_error_tolerance
                if ctx._converted_system.ewald_error_tolerance is not None
                else ctx._config.ewald_error_tolerance
            )
            eff_nb_m = ctx._effective_nonbonded_method()
            eff_pa = ctx._effective_pme_alpha()
            eff_lpa = ctx._effective_lj_pme_alpha()
            eff_lpd = _resolve_cs_lj_pme_grid_dims(
                ctx._converted_system, ctx._config, ctx._box_vectors
            )

            def _rep_energy_fn(
                p: jnp.ndarray,
                _ctx: ProlixContext = ctx,
                _gp: dict[str, jnp.ndarray] = gp,
                _et: float = ewald_tol,
                _nbm: str = eff_nb_m,
                _pa: float | None = eff_pa,
                _lpa: float | None = eff_lpa,
                _lpd: tuple[int, int, int] | None = eff_lpd,
            ) -> jnp.ndarray:
                p_eval = _ctx._converted_system.compute_virtual_sites(p)
                e_sys = _eval_system_potential_energy(
                    p_eval,
                    _ctx._box_vectors,
                    _ctx._bundle,
                    _ctx._custom_forces,
                    _ctx._position_restraints,
                    _gp,
                    _ctx._converted_system.force_group_by_component,
                    -1,
                    _ctx._converted_system.has_pbc,
                    _ctx._converted_system.nonbonded_cutoff,
                    bond_force_groups=_ctx._converted_system.bond_force_groups,
                    angle_force_groups=_ctx._converted_system.angle_force_groups,
                    dihedral_force_groups=_ctx._converted_system.dihedral_force_groups,
                    cmap_force_groups=_ctx._converted_system.cmap_force_groups,
                    nonbonded_method=_nbm,
                    has_nonbonded=_ctx._converted_system.has_nonbonded,
                    use_dispersion_correction=(
                        _ctx._config.use_dispersion_correction
                        and _ctx._converted_system.use_dispersion_correction
                    ),
                    ewald_error_tolerance=_et,
                    rf_dielectric=_ctx._converted_system.reaction_field_dielectric,
                    has_implicit_solvent=_ctx._converted_system.has_implicit_solvent,
                    solvent_dielectric=_ctx._converted_system.solvent_dielectric,
                    solute_dielectric=_ctx._converted_system.solute_dielectric,
                    nonbonded_engine=_ctx._nonbonded_engine,
                    pme_grid_dims=_ctx._converted_system.pme_grid_dims,
                    use_pallas=_ctx._config.use_pallas,
                    pallas_interpret=_ctx._config.pallas_interpret,
                    lj_switch_width=_ctx._converted_system.lj_switch_width,
                    neighbor_idx=getattr(_ctx, "_neighbor_idx", None),
                    tile_pair_nl=getattr(_ctx, "_tile_pair_nl", None),
                    gbsa_tile_list=getattr(_ctx, "_gbsa_tile_list", None),
                    use_pallas_pme=_ctx._config.use_pallas_pme,
                    custom_nonbonded_chunk_size=_ctx._config.custom_nonbonded_chunk_size,
                    is_triclinic=_ctx._is_triclinic,
                    gb_use_neighbor_list=_ctx._gb_use_neighbor_list,
                    gbsa_surface_tension=_ctx._converted_system.gbsa_surface_tension,
                    pme_alpha=_pa,
                    nonbonded_offsets=_ctx._converted_system.nonbonded_offsets,
                    default_params=_ctx._converted_system.default_parameters,
                    lj_combining_rule=_ctx._lj_combining_rule,
                    exceptions_use_pbc=_ctx._exceptions_use_pbc,
                    dispersion_follows_parameters=_ctx._dispersion_follows_parameters,
                    lj_pme_alpha=_lpa,
                    lj_pme_grid_dims=_lpd,
                )
                return e_sys + _constraint_penalty_energy_akma(
                    p_eval,
                    _ctx._box_vectors,
                    _ctx._bundle,
                    _ctx._converted_system.has_pbc,
                    exceptions_use_pbc=_ctx._exceptions_use_pbc,
                )

            pos_init = ctx._positions
            if eff_backend == "lbfgs":
                solver = _jaxopt.LBFGS(
                    fun=_rep_energy_fn,
                    maxiter=max_iters_gpu,
                    tol=tol_akma,
                    linesearch="zoom",
                    maxls=30,
                    stop_if_linesearch_fails=True,
                )
                min_pos = solver.run(pos_init).params
            else:
                init_fn, step_fn = jax_md_minimize.fire_descent(
                    _rep_energy_fn,
                    shift_fn=lambda r, dr: r + dr,
                    dt_start=0.002,
                    dt_max=0.02,
                )
                st = init_fn(pos_init)
                for _ in range(max_iters_gpu):
                    st = step_fn(st)
                    f_norm = float(jnp.sqrt(jnp.mean(jnp.sum(st.force**2, axis=-1))))
                    if f_norm <= tol_akma:
                        break
                min_pos = st.position
            self._apply_minimized_positions_to_replica(i, min_pos)
        self._refresh_plan_bundles_after_state_update()

    # --- Batch State & Checkpoint Serialization ---
    def saveStates(  # noqa: N802
        self,
        files: str | os.PathLike[str] | Any | Sequence[Any],
    ) -> None:
        """Save portable XML OpenMM `State`s across all `B` replicas."""
        omm_mod = cast("Any", openmm)
        states = self.getStates(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            getParameters=True,
        )
        xml_list = [str(omm_mod.XmlSerializer.serialize(st)) for st in states]

        if isinstance(files, (str, os.PathLike)):
            payload = json.dumps(
                {
                    "format": "prolix_batch_states_v1",
                    "batch_size": int(self.batch_size),
                    "n_atoms_per_replica": [int(n) for n in self._n_atoms_per_replica],
                    "states_xml": xml_list,
                }
            )
            Path(cast("str | os.PathLike[str]", files)).write_text(
                payload, encoding="utf-8"
            )
            return
        if hasattr(files, "write") and not isinstance(files, Sequence):
            payload = json.dumps(
                {
                    "format": "prolix_batch_states_v1",
                    "batch_size": int(self.batch_size),
                    "n_atoms_per_replica": [int(n) for n in self._n_atoms_per_replica],
                    "states_xml": xml_list,
                }
            )
            files.write(payload.encode("utf-8") if "b" in str(getattr(files, "mode", "")) else payload)
            return
        if not isinstance(files, Sequence) or isinstance(files, bytes):
            raise openmm.OpenMMException(
                f"saveStates: expected file path, stream, or sequence of {self.batch_size} files."
            )
        if len(files) != self.batch_size:
            raise openmm.OpenMMException(
                f"saveStates: batch_size mismatch (expected {self.batch_size} files, got {len(files)})."
            )
        for f_target, xml_s in zip(files, xml_list, strict=True):
            if isinstance(f_target, (str, os.PathLike)):
                Path(f_target).write_text(xml_s, encoding="utf-8")
            elif hasattr(f_target, "write"):
                f_target.write(
                    xml_s.encode("utf-8")
                    if "b" in str(getattr(f_target, "mode", ""))
                    else xml_s
                )
            else:
                raise openmm.OpenMMException(
                    f"saveStates: unsupported target of type {type(f_target).__name__}."
                )

    def loadStates(  # noqa: N802
        self,
        files: str | os.PathLike[str] | Any | Sequence[Any],
    ) -> None:
        """Load portable OpenMM `State`s across all `B` replicas, validating `batch_size` and atom counts."""
        omm_mod = cast("Any", openmm)

        def _deserialize_xml(xml_text: str) -> Any:
            try:
                return omm_mod.XmlSerializer.deserialize(xml_text)
            except Exception as exc:
                raise openmm.OpenMMException(
                    f"loadStates: failed to deserialize OpenMM State XML ({exc})"
                ) from exc

        parsed_states: list[Any] = []
        if isinstance(files, (str, os.PathLike)) or (
            hasattr(files, "read") and not isinstance(files, Sequence)
        ):
            try:
                raw = (
                    Path(cast("str | os.PathLike[str]", files)).read_text(
                        encoding="utf-8"
                    )
                    if isinstance(files, (str, os.PathLike))
                    else files.read()
                )
            except Exception as exc:
                raise openmm.OpenMMException(
                    f"loadStates: unable to read state source ({exc})"
                ) from exc
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            stripped = text.lstrip()
            if stripped.startswith("{"):
                try:
                    data = json.loads(text)
                except Exception as exc:
                    raise openmm.OpenMMException(
                        f"loadStates: invalid batch state JSON ({exc})"
                    ) from exc
                if (
                    not isinstance(data, dict)
                    or data.get("format") != "prolix_batch_states_v1"
                    or not isinstance(data.get("states_xml"), list)
                ):
                    raise openmm.OpenMMException(
                        "loadStates: invalid batch state envelope header."
                    )
                data_d = cast("dict[str, Any]", data)
                saved_b = int(data_d.get("batch_size", len(data_d["states_xml"])))
                if saved_b != self.batch_size or len(data_d["states_xml"]) != self.batch_size:
                    raise openmm.OpenMMException(
                        f"loadStates: batch_size mismatch (saved {saved_b}, expected {self.batch_size})."
                    )
                saved_counts = data_d.get("n_atoms_per_replica")
                if isinstance(saved_counts, list):
                    if len(saved_counts) != self.batch_size:
                        raise openmm.OpenMMException(
                            f"loadStates: atom count header length {len(saved_counts)} != {self.batch_size}."
                        )
                    for i, n_saved in enumerate(saved_counts):
                        if int(cast("Any", n_saved)) != self._n_atoms_per_replica[i]:
                            raise openmm.OpenMMException(
                                f"loadStates: replica {i} atom count mismatch (saved {n_saved}, expected {self._n_atoms_per_replica[i]})."
                            )
                parsed_states = [_deserialize_xml(str(x)) for x in data_d["states_xml"]]
            elif stripped.startswith("<"):
                if self.batch_size != 1:
                    raise openmm.OpenMMException(
                        f"loadStates: batch_size mismatch (single State XML provided for batch_size={self.batch_size})."
                    )
                parsed_states = [_deserialize_xml(text)]
            else:
                raise openmm.OpenMMException("loadStates: unrecognized state format.")
        elif isinstance(files, Sequence) and not isinstance(files, bytes):
            if len(files) != self.batch_size:
                raise openmm.OpenMMException(
                    f"loadStates: batch_size mismatch (expected {self.batch_size} states, got {len(files)})."
                )
            for item in files:
                if hasattr(item, "getPositions") and hasattr(item, "getStepCount"):
                    parsed_states.append(item)
                elif isinstance(item, os.PathLike):
                    parsed_states.append(
                        _deserialize_xml(Path(item).read_text(encoding="utf-8"))
                    )
                elif isinstance(item, str):
                    if item.lstrip().startswith("<"):
                        parsed_states.append(_deserialize_xml(item))
                    else:
                        try:
                            txt = Path(item).read_text(encoding="utf-8")
                        except Exception as exc:
                            raise openmm.OpenMMException(
                                f"loadStates: unable to read state file {item!r} ({exc})"
                            ) from exc
                        parsed_states.append(_deserialize_xml(txt))
                elif hasattr(item, "read"):
                    raw_i = item.read()
                    txt_i = (
                        raw_i.decode("utf-8")
                        if isinstance(raw_i, (bytes, bytearray))
                        else str(raw_i)
                    )
                    parsed_states.append(_deserialize_xml(txt_i))
                else:
                    raise openmm.OpenMMException(
                        f"loadStates: unsupported state element of type {type(item).__name__}."
                    )
        else:
            raise openmm.OpenMMException(
                f"loadStates: expected path, stream, or sequence of {self.batch_size} states."
            )

        # Validate per-replica atom counts & parameter names BEFORE mutating batch state
        validated_pos_ang: list[np.ndarray | None] = []
        validated_vel_omm: list[np.ndarray | None] = []
        validated_box_ang: list[np.ndarray | None] = []
        validated_params: list[dict[str, float] | None] = []

        for i, st in enumerate(parsed_states):
            exp_n = self._n_atoms_per_replica[i]
            types = int(st.getDataTypes()) if hasattr(st, "getDataTypes") else 0xFFFFFFFF

            pos_ang_i: np.ndarray | None = None
            if types & int(openmm.State.Positions):
                pos_nm_i = np.asarray(
                    st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                    dtype=np.float64,
                )
                if pos_nm_i.shape != (exp_n, 3):
                    raise openmm.OpenMMException(
                        f"loadStates: replica {i} atom count mismatch (got positions shape {pos_nm_i.shape}, expected ({exp_n}, 3))."
                    )
                pos_ang_i = pos_nm_i * NM_TO_ANGSTROM
            validated_pos_ang.append(pos_ang_i)

            vel_omm_i: np.ndarray | None = None
            if types & int(openmm.State.Velocities):
                vel_arr = np.asarray(
                    st.getVelocities(asNumpy=True).value_in_unit(
                        unit.nanometer / unit.picosecond
                    ),
                    dtype=np.float64,
                )
                if vel_arr.shape != (exp_n, 3):
                    raise openmm.OpenMMException(
                        f"loadStates: replica {i} velocity atom count mismatch (got shape {vel_arr.shape}, expected ({exp_n}, 3))."
                    )
                vel_omm_i = vel_arr
            validated_vel_omm.append(vel_omm_i)

            box_ang_i: np.ndarray | None = None
            if self._bundles[i].shape_spec.has_pbc:
                with contextlib.suppress(Exception):
                    bv_nm = np.asarray(
                        st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                        dtype=np.float64,
                    )
                    if bv_nm.shape == (3, 3):
                        box_ang_i = bv_nm * NM_TO_ANGSTROM
            validated_box_ang.append(box_ang_i)

            params_i: dict[str, float] | None = None
            if types & int(openmm.State.Parameters):
                raw_p = dict(st.getParameters())
                for k_nm in raw_p:
                    if k_nm not in self._parameters_per_replica[i]:
                        raise openmm.OpenMMException(
                            f"loadStates: unknown parameter {k_nm!r} in replica {i}."
                        )
                params_i = {k_nm: float(v_v) for k_nm, v_v in raw_p.items()}
            validated_params.append(params_i)

        with contextlib.suppress(Exception):
            self.currentStep = int(parsed_states[0].getStepCount())

        any_vel = False
        for i in range(self.batch_size):
            b_old = self._bundles[i]
            b_base_old = self._base_bundles[i]
            new_box_f32 = b_old.box
            if validated_box_ang[i] is not None:
                bx_jnp = jnp.asarray(validated_box_ang[i], dtype=self._dtype)
                self._boxes_akma[i] = bx_jnp
                new_box_f32 = bx_jnp.astype(jnp.float32)

            if validated_pos_ang[i] is not None:
                p_jnp = jnp.asarray(validated_pos_ang[i], dtype=self._dtype)
                self._positions_akma[i] = p_jnp
                pad_len = int(b_old.positions.shape[0])
                pos_f32 = _pad_positions_to_bucket(
                    p_jnp.astype(jnp.float32),
                    pad_len,
                    new_box_f32 if b_old.shape_spec.has_pbc else None,
                )
            else:
                pos_f32 = b_old.positions

            self._bundles[i] = dataclasses.replace(
                b_old, positions=pos_f32, box=new_box_f32
            )
            self._base_bundles[i] = dataclasses.replace(
                b_base_old, positions=pos_f32, box=new_box_f32
            )

            if validated_vel_omm[i] is not None:
                self._velocities_akma[i] = jnp.asarray(
                    cast("np.ndarray", validated_vel_omm[i]) * VELOCITY_OMM_TO_AKMA,
                    dtype=self._dtype,
                )
                any_vel = True

            if validated_params[i] is not None:
                self._parameters_per_replica[i].update(cast("dict[str, float]", validated_params[i]))

            if self._contexts is not None:
                self._contexts[i].setState(parsed_states[i])

        if any_vel:
            self._velocities_initialized = True
        if any(cs.nonbonded_offsets is not None for cs in self._converted_systems):
            self._sync_offset_parameters_to_bundles()
        self._boxes_match = self._compute_boxes_match()
        self._refresh_plan_bundles_after_state_update()

    def createCheckpoint(self) -> bytes:  # noqa: N802
        """Create a binary checkpoint containing exact positions, velocities, boxes, parameters, and RNG seeds."""
        self._sync_from_contexts_if_present()
        ckpt_dict: dict[str, Any] = {
            "magic": "prolix_batch_checkpoint_v1",
            "batch_size": int(self.batch_size),
            "n_atoms_per_replica": [int(n) for n in self._n_atoms_per_replica],
            "current_step": int(self.currentStep),
            "seed": int(self._seed),
            "velocities_initialized": bool(self._velocities_initialized),
            "positions_akma": [np.asarray(p) for p in self._positions_akma],
            "velocities_akma": [np.asarray(v) for v in self._velocities_akma],
            "boxes_akma": [
                np.asarray(bx) if bx is not None else None for bx in self._boxes_akma
            ],
            "parameters_per_replica": [dict(p) for p in self._parameters_per_replica],
            "context_checkpoints": (
                [ctx.createCheckpoint() for ctx in self._contexts]
                if self._contexts is not None
                else None
            ),
        }
        return pickle.dumps(ckpt_dict, protocol=pickle.HIGHEST_PROTOCOL)

    def saveCheckpoint(self, file: str | os.PathLike[str] | Any) -> None:  # noqa: N802
        """Write binary batch checkpoint to `file` (path or binary file-like object)."""
        data = self.createCheckpoint()
        if isinstance(file, (str, os.PathLike)):
            Path(file).write_bytes(data)
        elif hasattr(file, "write"):
            file.write(data)
        else:
            raise openmm.OpenMMException(
                f"saveCheckpoint: expected file path or writable binary stream, got {type(file).__name__}."
            )

    def loadCheckpoint(  # noqa: N802
        self,
        checkpoint: bytes | bytearray | str | os.PathLike[str] | dict[str, Any] | Any,
    ) -> None:
        """Restore binary batch checkpoint, validating `batch_size` and per-replica atom counts."""
        if isinstance(checkpoint, dict):
            raw_ckpt: Any = checkpoint
        else:
            if isinstance(checkpoint, (bytes, bytearray, memoryview)):
                raw_bytes = bytes(checkpoint)
            elif isinstance(checkpoint, (str, os.PathLike)):
                try:
                    raw_bytes = Path(
                        cast("str | os.PathLike[str]", checkpoint)
                    ).read_bytes()
                except Exception as exc:
                    raise openmm.OpenMMException(
                        f"loadCheckpoint: unable to read checkpoint file ({exc})"
                    ) from exc
            elif hasattr(checkpoint, "read"):
                raw_in = checkpoint.read()
                if not isinstance(raw_in, (bytes, bytearray, memoryview)):
                    raise openmm.OpenMMException(
                        "loadCheckpoint: stream did not return binary bytes."
                    )
                raw_bytes = bytes(raw_in)
            else:
                raise openmm.OpenMMException(
                    f"loadCheckpoint: unsupported checkpoint argument type {type(checkpoint).__name__}."
                )
            try:
                raw_ckpt = pickle.loads(raw_bytes)  # noqa: S301
            except Exception as exc:
                raise openmm.OpenMMException(
                    f"loadCheckpoint: Checkpoint header was not correct ({exc})"
                ) from exc

        if not isinstance(raw_ckpt, dict):
            raise openmm.OpenMMException(
                "loadCheckpoint: Checkpoint header was not correct (expected dict)."
            )
        ckpt_dict = cast("dict[str, Any]", raw_ckpt)
        if ckpt_dict.get("magic") != "prolix_batch_checkpoint_v1":
            raise openmm.OpenMMException(
                "loadCheckpoint: Checkpoint header was not correct (invalid magic)."
            )

        req_keys = (
            "batch_size",
            "n_atoms_per_replica",
            "current_step",
            "positions_akma",
            "velocities_akma",
            "boxes_akma",
            "parameters_per_replica",
        )
        for k_req in req_keys:
            if k_req not in ckpt_dict:
                raise openmm.OpenMMException(
                    f"loadCheckpoint: Checkpoint header was not correct (missing key {k_req!r})."
                )

        saved_b = int(ckpt_dict["batch_size"])
        pos_list = ckpt_dict["positions_akma"]
        vel_list = ckpt_dict["velocities_akma"]
        box_list = ckpt_dict["boxes_akma"]
        param_list = ckpt_dict["parameters_per_replica"]
        saved_counts = ckpt_dict["n_atoms_per_replica"]

        if (
            saved_b != self.batch_size
            or len(saved_counts) != self.batch_size
            or len(pos_list) != self.batch_size
            or len(vel_list) != self.batch_size
            or len(box_list) != self.batch_size
            or len(param_list) != self.batch_size
        ):
            raise openmm.OpenMMException(
                f"loadCheckpoint: batch_size mismatch (checkpoint has {saved_b}, expected {self.batch_size})."
            )

        for i in range(self.batch_size):
            exp_n = self._n_atoms_per_replica[i]
            p_arr = np.asarray(pos_list[i])
            v_arr = np.asarray(vel_list[i])
            if (
                int(saved_counts[i]) != exp_n
                or p_arr.shape != (exp_n, 3)
                or v_arr.shape != (exp_n, 3)
            ):
                raise openmm.OpenMMException(
                    f"loadCheckpoint: replica {i} atom count mismatch "
                    f"(checkpoint has n_atoms={saved_counts[i]}, pos={p_arr.shape}, vel={v_arr.shape}; expected ({exp_n}, 3))."
                )
            if box_list[i] is not None and np.asarray(box_list[i]).shape != (3, 3):
                raise openmm.OpenMMException(
                    f"loadCheckpoint: replica {i} invalid box shape {np.asarray(box_list[i]).shape}."
                )
            for p_nm in param_list[i]:
                if p_nm not in self._parameters_per_replica[i]:
                    raise openmm.OpenMMException(
                        f"loadCheckpoint: unknown parameter {p_nm!r} in replica {i}."
                    )

        self.currentStep = int(ckpt_dict["current_step"])
        self._seed = int(ckpt_dict.get("seed", self._seed))
        self._velocities_initialized = bool(ckpt_dict.get("velocities_initialized", True))

        ctx_ckpts = ckpt_dict.get("context_checkpoints")
        for i in range(self.batch_size):
            b_old = self._bundles[i]
            b_base_old = self._base_bundles[i]
            p_jnp = jnp.asarray(pos_list[i], dtype=self._dtype)
            v_jnp = jnp.asarray(vel_list[i], dtype=self._dtype)
            self._positions_akma[i] = p_jnp
            self._velocities_akma[i] = v_jnp

            new_box_f32 = b_old.box
            if box_list[i] is not None:
                bx_jnp = jnp.asarray(box_list[i], dtype=self._dtype)
                self._boxes_akma[i] = bx_jnp
                if b_old.shape_spec.has_pbc:
                    new_box_f32 = bx_jnp.astype(jnp.float32)
            else:
                self._boxes_akma[i] = None

            pad_len = int(b_old.positions.shape[0])
            pos_f32 = _pad_positions_to_bucket(
                p_jnp.astype(jnp.float32),
                pad_len,
                new_box_f32 if b_old.shape_spec.has_pbc else None,
            )
            self._bundles[i] = dataclasses.replace(
                b_old, positions=pos_f32, box=new_box_f32
            )
            self._base_bundles[i] = dataclasses.replace(
                b_base_old, positions=pos_f32, box=new_box_f32
            )
            self._parameters_per_replica[i] = {
                str(k): float(v) for k, v in dict(param_list[i]).items()
            }

            if self._contexts is not None:
                if isinstance(ctx_ckpts, list) and len(ctx_ckpts) == self.batch_size:
                    self._contexts[i].loadCheckpoint(ctx_ckpts[i])

        if self._contexts is not None and not (
            isinstance(ctx_ckpts, list) and len(ctx_ckpts) == self.batch_size
        ):
            self._ensure_contexts()

        if any(cs.nonbonded_offsets is not None for cs in self._converted_systems):
            self._sync_offset_parameters_to_bundles()
        self._boxes_match = self._compute_boxes_match()
        self._refresh_plan_bundles_after_state_update()

    def _can_vmap_batch_observables(self, groups_mask: int) -> bool:  # noqa: PLR0911
        """Return True when all `B` replicas share uniform shapes & non-Lepton physics for `jax.vmap`."""
        if len(set(self._n_atoms_per_replica)) != 1:
            return False
        if len({b.shape_spec for b in self._base_bundles}) != 1:
            return False

        cs0 = self._converted_systems[0]
        if bool(cs0.custom_forces) or cs0.virtual_sites is not None:
            return False

        pr0_sig = _pr_signature(cs0.position_restraints)
        eff_nb0 = _effective_cs_nonbonded_method(cs0, self.config)
        lj_dims0 = _resolve_cs_lj_pme_grid_dims(cs0, self.config, self._boxes_akma[0])
        off0 = cs0.nonbonded_offsets
        off0_sig = (
            None
            if off0 is None
            else (
                off0.param_names,
                tuple(off0.particle_indices.shape),
                tuple(off0.exception_bundle_indices.shape),
            )
        )
        fg0 = tuple(sorted(cs0.force_group_by_component.items()))
        pnames0 = tuple(self._parameters_per_replica[0].keys())

        for i in range(1, self.batch_size):
            csi = self._converted_systems[i]
            if (
                bool(csi.custom_forces)
                or csi.virtual_sites is not None
                or csi.is_triclinic != cs0.is_triclinic
                or _pr_signature(csi.position_restraints) != pr0_sig
                or csi.has_pbc != cs0.has_pbc
                or _effective_cs_nonbonded_method(csi, self.config) != eff_nb0
                or csi.has_nonbonded != cs0.has_nonbonded
                or csi.nonbonded_cutoff != cs0.nonbonded_cutoff
                or csi.lj_switch_width != cs0.lj_switch_width
                or csi.use_dispersion_correction != cs0.use_dispersion_correction
                or csi.ewald_error_tolerance != cs0.ewald_error_tolerance
                or csi.explicit_pme_alpha != cs0.explicit_pme_alpha
                or csi.pme_grid_dims != cs0.pme_grid_dims
                or csi.explicit_lj_pme_alpha != cs0.explicit_lj_pme_alpha
                or _resolve_cs_lj_pme_grid_dims(csi, self.config, self._boxes_akma[i]) != lj_dims0
                or csi.reaction_field_dielectric != cs0.reaction_field_dielectric
                or csi.has_implicit_solvent != cs0.has_implicit_solvent
                or csi.solvent_dielectric != cs0.solvent_dielectric
                or csi.solute_dielectric != cs0.solute_dielectric
                or csi.gbsa_surface_tension != cs0.gbsa_surface_tension
                or csi.lj_combining_rule != cs0.lj_combining_rule
                or csi.exceptions_use_pbc != cs0.exceptions_use_pbc
                or csi.dispersion_follows_parameters != cs0.dispersion_follows_parameters
                or tuple(sorted(csi.force_group_by_component.items())) != fg0
                or tuple(self._parameters_per_replica[i].keys()) != pnames0
            ):
                return False
            offi = csi.nonbonded_offsets
            offi_sig = (
                None
                if offi is None
                else (
                    offi.param_names,
                    tuple(offi.particle_indices.shape),
                    tuple(offi.exception_bundle_indices.shape),
                )
            )
            if offi_sig != off0_sig:
                return False
            if groups_mask != -1:
                for attr in (
                    "bond_force_groups",
                    "angle_force_groups",
                    "dihedral_force_groups",
                    "cmap_force_groups",
                ):
                    if (getattr(csi, attr) is None) != (getattr(cs0, attr) is None):
                        return False
        return True

    def _stack_batch_inputs_for_vmap(
        self,
    ) -> tuple[
        jnp.ndarray | None,
        MolecularBundle,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
        tuple[PositionRestraints, ...],
        NonbondedParameterOffsets | None,
        jnp.ndarray,
        tuple[str, ...],
    ]:
        """Stack replica bundles, force-group masks, restraints, offsets, and global parameter vectors."""
        cs0 = self._converted_systems[0]
        box_b = (
            jnp.stack(
                [
                    cast("jnp.ndarray", self._boxes_akma[i]).astype(self._dtype)
                    for i in range(self.batch_size)
                ],
                axis=0,
            )
            if cs0.has_pbc and all(bx is not None for bx in self._boxes_akma)
            else None
        )
        base_bundles_b: MolecularBundle = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), *self._base_bundles
        )
        bond_fg_b = (
            jnp.stack([cast("jnp.ndarray", cs.bond_force_groups) for cs in self._converted_systems], axis=0)
            if cs0.bond_force_groups is not None
            else None
        )
        angle_fg_b = (
            jnp.stack([cast("jnp.ndarray", cs.angle_force_groups) for cs in self._converted_systems], axis=0)
            if cs0.angle_force_groups is not None
            else None
        )
        dihedral_fg_b = (
            jnp.stack([cast("jnp.ndarray", cs.dihedral_force_groups) for cs in self._converted_systems], axis=0)
            if cs0.dihedral_force_groups is not None
            else None
        )
        cmap_fg_b = (
            jnp.stack([cast("jnp.ndarray", cs.cmap_force_groups) for cs in self._converted_systems], axis=0)
            if cs0.cmap_force_groups is not None
            else None
        )
        pr_b: tuple[PositionRestraints, ...] = tuple(
            jax.tree.map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[cs.position_restraints[k] for cs in self._converted_systems],
            )
            for k in range(len(cs0.position_restraints))
        )
        offsets_b: NonbondedParameterOffsets | None = (
            jax.tree.map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[cast("NonbondedParameterOffsets", cs.nonbonded_offsets) for cs in self._converted_systems],
            )
            if cs0.nonbonded_offsets is not None
            else None
        )
        param_names = tuple(self._parameters_per_replica[0].keys())
        if param_names:
            gp_vec_b = jnp.asarray(
                [
                    [self._parameters_per_replica[i][nm] for nm in param_names]
                    for i in range(self.batch_size)
                ],
                dtype=self._dtype,
            )
        else:
            gp_vec_b = jnp.zeros((self.batch_size, 0), dtype=self._dtype)
        return (
            box_b,
            base_bundles_b,
            bond_fg_b,
            angle_fg_b,
            dihedral_fg_b,
            cmap_fg_b,
            pr_b,
            offsets_b,
            gp_vec_b,
            param_names,
        )

    def _compute_vmapped_param_derivatives(
        self, groups_mask: int
    ) -> list[dict[str, float]]:
        """Evaluate exact `dU/dλ` across all `B` replicas via a single `jax.vmap` pass."""
        cs0 = self._converted_systems[0]
        off0 = cs0.nonbonded_offsets
        g_nb = int(cs0.force_group_by_component.get("nonbonded", 0))
        if (
            off0 is None
            or len(off0.param_names) == 0
            or not cs0.has_nonbonded
            or ((1 << g_nb) & groups_mask) == 0
        ):
            return [{} for _ in range(self.batch_size)]

        pos_b = jnp.stack(self._positions_akma, axis=0)
        (
            box_b,
            base_bundles_b,
            _bfg,
            _afg,
            _dfg,
            _cfg,
            _pr_b,
            offsets_b,
            gp_vec_b,
            param_names,
        ) = self._stack_batch_inputs_for_vmap()
        ewald_tol = (
            cs0.ewald_error_tolerance
            if cs0.ewald_error_tolerance is not None
            else self.config.ewald_error_tolerance
        )
        eff_nb_method = _effective_cs_nonbonded_method(cs0, self.config)
        eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
        eff_lj_alpha = _resolve_cs_lj_pme_alpha(cs0, self.config)
        eff_lj_dims = _resolve_cs_lj_pme_grid_dims(
            cs0, self.config, self._boxes_akma[0]
        )
        use_disp = bool(
            self.config.use_dispersion_correction and cs0.use_dispersion_correction
        )
        nb_eng = (
            "neighbor_list"
            if (
                not cs0.is_triclinic
                and int(pos_b.shape[1]) >= 4000
                and cs0.nonbonded_cutoff < float("inf")
            )
            else "dense"
        )
        grads_kj = np.asarray(
            _vmapped_batch_nonbonded_param_derivatives_kj(
                pos_b,
                box_b,
                base_bundles_b,
                cast("NonbondedParameterOffsets", offsets_b),
                gp_vec_b,
                param_names,
                cs0.force_group_by_component,
                groups_mask,
                cs0.has_pbc,
                cs0.nonbonded_cutoff,
                eff_nb_method,
                cs0.has_nonbonded,
                use_disp,
                ewald_tol,
                eff_alpha,
                cs0.reaction_field_dielectric,
                nb_eng,
                cs0.pme_grid_dims,
                cs0.lj_switch_width,
                self.config.custom_nonbonded_chunk_size,
                cs0.is_triclinic,
                cs0.default_parameters,
                cs0.lj_combining_rule,
                cs0.exceptions_use_pbc,
                cs0.dispersion_follows_parameters,
                eff_lj_alpha,
                eff_lj_dims,
            ),
            dtype=np.float64,
        )
        name_to_col = {nm: idx for idx, nm in enumerate(param_names)}
        out: list[dict[str, float]] = []
        for i in range(self.batch_size):
            d_i: dict[str, float] = {}
            for p_nm in off0.param_names:
                if p_nm in name_to_col:
                    d_i[p_nm] = float(grads_kj[i, name_to_col[p_nm]])
            out.append(d_i)
        return out

    def getStates(  # noqa: N802
        self,
        getPositions: bool = False,
        getVelocities: bool = False,
        getForces: bool = False,
        getEnergy: bool = False,
        getParameters: bool = False,
        getParameterDerivatives: bool = False,
        enforcePeriodicBox: bool = False,
        groups: int | set[int] = -1,
    ) -> list[ProlixState]:
        """Extract `ProlixState` snapshots for all `B` replicas in OpenMM units."""
        if isinstance(groups, set):
            raw_groups = sum(1 << g for g in groups if 0 <= g <= 31)
        else:
            raw_groups = int(groups)
        groups_mask = int(np.int32(raw_groups))

        self._sync_from_contexts_if_present()
        can_vmap = self._can_vmap_batch_observables(groups_mask)
        if not can_vmap and (
            self._contexts is not None or getParameterDerivatives or groups_mask != -1
        ):
            ctxs = self._ensure_contexts()
            states: list[ProlixState] = []
            for ctx in ctxs:
                states.append(
                    ctx.getState(
                        getPositions=getPositions,
                        getVelocities=getVelocities,
                        getForces=getForces,
                        getEnergy=getEnergy,
                        getParameters=getParameters,
                        getParameterDerivatives=getParameterDerivatives,
                        enforcePeriodicBox=enforcePeriodicBox,
                        groups=groups,
                    )
                )
            return states

        data_types = 0
        if getPositions:
            data_types |= int(openmm.State.Positions)
        if getVelocities:
            data_types |= int(openmm.State.Velocities)
        if getForces:
            data_types |= int(openmm.State.Forces)
        if getEnergy:
            data_types |= int(openmm.State.Energy)
        if getParameters:
            data_types |= int(openmm.State.Parameters)
        if getParameterDerivatives:
            data_types |= int(openmm.State.ParameterDerivatives)

        pe_np_b: np.ndarray | None = None
        forces_akma_np_b: np.ndarray | None = None
        forces_np_b: np.ndarray | None = None
        if can_vmap and (getEnergy or getForces):
            cs0 = self._converted_systems[0]
            pos_b = jnp.stack(self._positions_akma, axis=0)
            (
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
            ) = self._stack_batch_inputs_for_vmap()
            ewald_tol = (
                cs0.ewald_error_tolerance
                if cs0.ewald_error_tolerance is not None
                else self.config.ewald_error_tolerance
            )
            eff_nb_method = _effective_cs_nonbonded_method(cs0, self.config)
            eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
            eff_lj_alpha = _resolve_cs_lj_pme_alpha(cs0, self.config)
            eff_lj_dims = _resolve_cs_lj_pme_grid_dims(
                cs0, self.config, self._boxes_akma[0]
            )
            use_disp = bool(
                self.config.use_dispersion_correction and cs0.use_dispersion_correction
            )
            nb_eng = (
                "flash"
                if (
                    cs0.has_pbc
                    and not cs0.is_triclinic
                    and not cs0.has_implicit_solvent
                    and cs0.nonbonded_offsets is None
                    and cs0.lj_combining_rule == "lorentz_berthelot"
                    and not cs0.exceptions_use_pbc
                )
                else (
                    "neighbor_list"
                    if (
                        not cs0.is_triclinic
                        and int(pos_b.shape[1]) >= 4000
                        and cs0.nonbonded_cutoff < float("inf")
                    )
                    else "dense"
                )
            )
            pe_akma_b, f_akma_b = _vmapped_batch_energy_and_forces_akma(
                pos_b,
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
                cs0.force_group_by_component,
                groups_mask,
                cs0.has_pbc,
                cs0.nonbonded_cutoff,
                eff_nb_method,
                cs0.has_nonbonded,
                use_disp,
                ewald_tol,
                eff_alpha,
                cs0.reaction_field_dielectric,
                cs0.has_implicit_solvent,
                cs0.solvent_dielectric,
                cs0.solute_dielectric,
                nb_eng,
                cs0.pme_grid_dims,
                cs0.lj_switch_width,
                self.config.custom_nonbonded_chunk_size,
                cs0.is_triclinic,
                self.config.precision == "mixed",
                cs0.gbsa_surface_tension,
                cs0.default_parameters,
                cs0.lj_combining_rule,
                cs0.exceptions_use_pbc,
                cs0.dispersion_follows_parameters,
                eff_lj_alpha,
                eff_lj_dims,
            )
            pe_host, f_host = jax.device_get((pe_akma_b, f_akma_b))
            pe_np_b = np.asarray(pe_host, dtype=np.float64) * KCAL_PER_MOL_TO_KJ_PER_MOL
            forces_akma_np_b = np.asarray(f_host, dtype=np.float64)
            forces_np_b = forces_akma_np_b * FORCE_AKMA_TO_OMM

        derivs_per_rep: list[dict[str, float]] | None = None
        if getParameterDerivatives:
            derivs_per_rep = self._compute_vmapped_param_derivatives(groups_mask)

        sync_lf_batch = bool(
            self._contexts is not None
            and len(self._contexts) > 0
            and self._contexts[0]._effective_integrator_scheme() == "leapfrog"
            and (self.currentStep > 0 or self._velocities_initialized)
            and self._dt_akma > 0.0
        )

        time_ps = self.currentStep * self._dt_akma * TAU_AKMA_TO_PS
        out_states: list[ProlixState] = []
        for i in range(self.batch_size):
            cs = self._converted_systems[i]
            b_i = self._bundles[i]
            b_base_i = self._base_bundles[i]
            n_i = self._n_atoms_per_replica[i]
            pos_i = self._positions_akma[i]
            vel_i = self._velocities_akma[i]
            box_i = self._boxes_akma[i] if b_i.shape_spec.has_pbc else None

            pos_nm = None
            if getPositions:
                p_eval = (
                    self._wrap_replica_molecules_dev(i, pos_i)
                    if (enforcePeriodicBox and box_i is not None)
                    else pos_i
                )
                pos_nm = np.asarray(p_eval, dtype=np.float64) * ANGSTROM_TO_NM

            vel_omm = None
            if getVelocities:
                vel_omm = np.asarray(vel_i, dtype=np.float64) * VELOCITY_AKMA_TO_OMM

            forces_omm = None
            pe_kj: float | None = None
            ke_kj: float | None = None
            if getForces or getEnergy:
                if (
                    pe_np_b is not None
                    and forces_np_b is not None
                    and forces_akma_np_b is not None
                ):
                    if getForces:
                        forces_omm = forces_np_b[i]
                    if getEnergy:
                        pe_kj = float(pe_np_b[i])
                        masses_np = np.asarray(b_i.masses[:n_i], dtype=np.float64)
                        vel_np = np.asarray(vel_i, dtype=np.float64)
                        if sync_lf_batch:
                            inv_m_np = np.where(masses_np > 1e-12, 1.0 / np.maximum(masses_np, 1e-12), 0.0)
                            vel_np = vel_np + 0.5 * self._dt_akma * (
                                forces_akma_np_b[i, :n_i] * inv_m_np[:, None]
                            )
                        ke_akma = 0.5 * float(np.sum(masses_np[:, None] * (vel_np**2)))
                        ke_kj = ke_akma * KCAL_PER_MOL_TO_KJ_PER_MOL
                else:
                    ewald_tol = (
                        cs.ewald_error_tolerance
                        if cs.ewald_error_tolerance is not None
                        else self.config.ewald_error_tolerance
                    )
                    gp_i = {
                        k: jnp.asarray(v, dtype=pos_i.dtype)
                        for k, v in self._parameters_per_replica[i].items()
                    }
                    eff_nb_i = _effective_cs_nonbonded_method(cs, self.config)
                    pe_akma, f_akma = _pure_compute_energy_and_forces(
                        pos_i,
                        box_i,
                        b_base_i,
                        (),
                        tuple(cs.position_restraints),
                        gp_i,
                        cs.force_group_by_component,
                        groups_mask,
                        cs.has_pbc,
                        cs.nonbonded_cutoff,
                        bond_force_groups=cs.bond_force_groups,
                        angle_force_groups=cs.angle_force_groups,
                        dihedral_force_groups=cs.dihedral_force_groups,
                        cmap_force_groups=cs.cmap_force_groups,
                        nonbonded_method=eff_nb_i,
                        has_nonbonded=cs.has_nonbonded,
                        use_dispersion_correction=(
                            self.config.use_dispersion_correction
                            and cs.use_dispersion_correction
                        ),
                        ewald_error_tolerance=ewald_tol,
                        rf_dielectric=cs.reaction_field_dielectric,
                        has_implicit_solvent=cs.has_implicit_solvent,
                        solvent_dielectric=cs.solvent_dielectric,
                        solute_dielectric=cs.solute_dielectric,
                        nonbonded_engine=(
                            "flash"
                            if (
                                cs.has_pbc
                                and not cs.is_triclinic
                                and not cs.has_implicit_solvent
                                and cs.nonbonded_offsets is None
                                and cs.lj_combining_rule == "lorentz_berthelot"
                                and not cs.exceptions_use_pbc
                            )
                            else "dense"
                        ),
                        pme_grid_dims=cs.pme_grid_dims,
                        use_pallas=self.config.use_pallas,
                        pallas_interpret=self.config.pallas_interpret,
                        lj_switch_width=cs.lj_switch_width,
                        use_pallas_pme=self.config.use_pallas_pme,
                        custom_nonbonded_chunk_size=self.config.custom_nonbonded_chunk_size,
                        is_triclinic=cs.is_triclinic,
                        mixed_precision=self.config.precision == "mixed",
                        gb_use_neighbor_list=self.config.gb_use_neighbor_list,
                        gbsa_surface_tension=cs.gbsa_surface_tension,
                        pme_alpha=_resolve_cs_pme_alpha(cs, self.config),
                        nonbonded_offsets=cs.nonbonded_offsets,
                        default_params=cs.default_parameters,
                        lj_combining_rule=cs.lj_combining_rule,
                        exceptions_use_pbc=cs.exceptions_use_pbc,
                        dispersion_follows_parameters=cs.dispersion_follows_parameters,
                        lj_pme_alpha=_resolve_cs_lj_pme_alpha(cs, self.config),
                        lj_pme_grid_dims=_resolve_cs_lj_pme_grid_dims(
                            cs, self.config, box_i
                        ),
                    )
                    if getForces:
                        forces_omm = np.asarray(f_akma, dtype=np.float64) * FORCE_AKMA_TO_OMM
                    if getEnergy:
                        pe_kj = float(pe_akma) * KCAL_PER_MOL_TO_KJ_PER_MOL
                        masses_np = np.asarray(b_i.masses[:n_i], dtype=np.float64)
                        vel_np = np.asarray(vel_i, dtype=np.float64)
                        ke_akma = 0.5 * float(np.sum(masses_np[:, None] * (vel_np**2)))
                        ke_kj = ke_akma * KCAL_PER_MOL_TO_KJ_PER_MOL

            box_nm = (
                np.asarray(box_i, dtype=np.float64) * ANGSTROM_TO_NM
                if box_i is not None
                else np.eye(3, dtype=np.float64) * 10.0
            )
            out_states.append(
                ProlixState(
                    positions_nm=pos_nm,
                    velocities_omm=vel_omm,
                    forces_omm=forces_omm,
                    box_vectors_nm=box_nm,
                    time_ps=time_ps,
                    step_count=self.currentStep,
                    pe_kj=pe_kj,
                    ke_kj=ke_kj,
                    parameters=dict(self._parameters_per_replica[i]) if getParameters else None,
                    param_derivatives=(
                        derivs_per_rep[i] if derivs_per_rep is not None else None
                    ),
                    data_types=data_types,
                )
            )
        return out_states

    def value_and_grad_positions(
        self,
        batch_positions: Sequence[Any] | jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | list[jnp.ndarray]]:
        """Differentiable batch energy (kJ/mol) and position gradients dU/dx (kJ/(mol*nm))."""
        self._sync_from_contexts_if_present()
        if self._can_vmap_batch_observables(-1):
            cs0 = self._converted_systems[0]
            if batch_positions is not None:
                if isinstance(batch_positions, jnp.ndarray) and batch_positions.ndim == 3:
                    pos_nm_b = batch_positions.astype(self._dtype)
                else:
                    p_rows: list[jnp.ndarray] = []
                    for i in range(self.batch_size):
                        p_in = batch_positions[i]
                        val_fn = getattr(p_in, "value_in_unit", None)
                        if unit.is_quantity(p_in) and callable(val_fn):
                            p_rows.append(jnp.asarray(val_fn(unit.nanometer), dtype=self._dtype))
                        else:
                            p_rows.append(jnp.asarray(p_in, dtype=self._dtype))
                    pos_nm_b = jnp.stack(p_rows, axis=0)
            else:
                pos_nm_b = jnp.stack(self._positions_akma, axis=0) * float(ANGSTROM_TO_NM)

            (
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
            ) = self._stack_batch_inputs_for_vmap()
            ewald_tol = (
                cs0.ewald_error_tolerance
                if cs0.ewald_error_tolerance is not None
                else self.config.ewald_error_tolerance
            )
            eff_nb_method = _effective_cs_nonbonded_method(cs0, self.config)
            eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
            eff_lj_alpha = _resolve_cs_lj_pme_alpha(cs0, self.config)
            eff_lj_dims = _resolve_cs_lj_pme_grid_dims(
                cs0, self.config, self._boxes_akma[0]
            )
            use_disp = bool(
                self.config.use_dispersion_correction and cs0.use_dispersion_correction
            )
            nb_eng = (
                "flash"
                if (
                    cs0.has_pbc
                    and not cs0.is_triclinic
                    and not cs0.has_implicit_solvent
                    and cs0.nonbonded_offsets is None
                    and cs0.lj_combining_rule == "lorentz_berthelot"
                    and not cs0.exceptions_use_pbc
                )
                else (
                    "neighbor_list"
                    if (
                        not cs0.is_triclinic
                        and int(pos_nm_b.shape[1]) >= 4000
                        and cs0.nonbonded_cutoff < float("inf")
                    )
                    else "dense"
                )
            )
            return _vmapped_batch_value_and_grad_positions_kj_nm(
                pos_nm_b,
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
                cs0.force_group_by_component,
                cs0.has_pbc,
                cs0.nonbonded_cutoff,
                eff_nb_method,
                cs0.has_nonbonded,
                use_disp,
                ewald_tol,
                eff_alpha,
                cs0.reaction_field_dielectric,
                cs0.has_implicit_solvent,
                cs0.solvent_dielectric,
                cs0.solute_dielectric,
                nb_eng,
                cs0.pme_grid_dims,
                cs0.lj_switch_width,
                self.config.custom_nonbonded_chunk_size,
                cs0.is_triclinic,
                cs0.gbsa_surface_tension,
                cs0.default_parameters,
                cs0.lj_combining_rule,
                cs0.exceptions_use_pbc,
                cs0.dispersion_follows_parameters,
                eff_lj_alpha,
                eff_lj_dims,
            )

        ctxs = self._ensure_contexts()
        energies: list[jnp.ndarray] = []
        grads: list[jnp.ndarray] = []

        for i, ctx in enumerate(ctxs):
            if batch_positions is not None:
                p_in = batch_positions[i]
                val_fn = getattr(p_in, "value_in_unit", None)
                if unit.is_quantity(p_in) and callable(val_fn):
                    p_ang = jnp.asarray(val_fn(unit.angstrom), dtype=ctx._dtype)
                else:
                    p_ang = jnp.asarray(p_in, dtype=ctx._dtype) * NM_TO_ANGSTROM
            else:
                p_ang = jnp.asarray(self._positions_akma[i], dtype=ctx._dtype)

            ewald_tol = (
                ctx._converted_system.ewald_error_tolerance
                if ctx._converted_system.ewald_error_tolerance is not None
                else ctx._config.ewald_error_tolerance
            )
            eff_nb_m = ctx._effective_nonbonded_method()
            eff_alpha = ctx._effective_pme_alpha()
            eff_lpa = ctx._effective_lj_pme_alpha()
            gp = {k: jnp.asarray(v, dtype=ctx._dtype) for k, v in ctx._parameters.items()}

            def _u_nm(
                p_nm: jnp.ndarray,
                _ctx: ProlixContext = ctx,
                _gp: dict[str, jnp.ndarray] = gp,
                _et: float = ewald_tol,
                _nbm: str = eff_nb_m,
                _pa: float | None = eff_alpha,
                _lpa: float | None = eff_lpa,
            ) -> jnp.ndarray:
                p_a = _ctx._converted_system.compute_virtual_sites(p_nm * NM_TO_ANGSTROM)
                u_kcal = _eval_system_potential_energy(
                    p_a,
                    _ctx._box_vectors,
                    _ctx._bundle,
                    _ctx._custom_forces,
                    _ctx._position_restraints,
                    _gp,
                    _ctx._converted_system.force_group_by_component,
                    -1,
                    _ctx._converted_system.has_pbc,
                    _ctx._converted_system.nonbonded_cutoff,
                    bond_force_groups=_ctx._converted_system.bond_force_groups,
                    angle_force_groups=_ctx._converted_system.angle_force_groups,
                    dihedral_force_groups=_ctx._converted_system.dihedral_force_groups,
                    cmap_force_groups=_ctx._converted_system.cmap_force_groups,
                    nonbonded_method=_nbm,
                    has_nonbonded=_ctx._converted_system.has_nonbonded,
                    use_dispersion_correction=(
                        _ctx._config.use_dispersion_correction
                        and _ctx._converted_system.use_dispersion_correction
                    ),
                    ewald_error_tolerance=_et,
                    pme_alpha=_pa,
                    rf_dielectric=_ctx._converted_system.reaction_field_dielectric,
                    has_implicit_solvent=_ctx._converted_system.has_implicit_solvent,
                    solvent_dielectric=_ctx._converted_system.solvent_dielectric,
                    solute_dielectric=_ctx._converted_system.solute_dielectric,
                    nonbonded_engine=(
                        "flash"
                        if (
                            _ctx._converted_system.has_pbc
                            and not _ctx._is_triclinic
                            and not _ctx._converted_system.has_implicit_solvent
                            and _ctx._converted_system.nonbonded_offsets is None
                            and _ctx._lj_combining_rule == "lorentz_berthelot"
                            and not _ctx._exceptions_use_pbc
                        )
                        else "dense"
                    ),
                    pme_grid_dims=_ctx._converted_system.pme_grid_dims,
                    use_pallas=False,
                    pallas_interpret=_ctx._config.pallas_interpret,
                    lj_switch_width=_ctx._converted_system.lj_switch_width,
                    neighbor_idx=None,
                    tile_pair_nl=None,
                    use_pallas_pme=False,
                    custom_nonbonded_chunk_size=_ctx._config.custom_nonbonded_chunk_size,
                    is_triclinic=_ctx._is_triclinic,
                    gb_use_neighbor_list=False,
                    gbsa_surface_tension=_ctx._converted_system.gbsa_surface_tension,
                    nonbonded_offsets=_ctx._converted_system.nonbonded_offsets,
                    default_params=_ctx._converted_system.default_parameters,
                    lj_combining_rule=_ctx._lj_combining_rule,
                    exceptions_use_pbc=_ctx._exceptions_use_pbc,
                    dispersion_follows_parameters=_ctx._dispersion_follows_parameters,
                    lj_pme_alpha=_lpa,
                    lj_pme_grid_dims=_ctx._converted_system.lj_pme_grid_dims,
                )
                return u_kcal * KCAL_PER_MOL_TO_KJ_PER_MOL

            p_nm_in = p_ang * ANGSTROM_TO_NM
            e_kj, g_kj_nm = jax.value_and_grad(_u_nm)(p_nm_in)
            energies.append(e_kj)
            grads.append(g_kj_nm)

        e_stack = jnp.stack(energies, axis=0)
        if len(set(self._n_atoms_per_replica)) == 1:
            return e_stack, jnp.stack(grads, axis=0)
        return e_stack, grads

    def _wrap_replica_molecules_dev(self, idx: int, pos_i: jnp.ndarray) -> jnp.ndarray:
        """Whole-molecule PBC wrapping on device (ortho & lower-triangular triclinic `z -> y -> x`)."""
        b_i = self._bundles[idx]
        n_i = self._n_atoms_per_replica[idx]
        box_i = self._boxes_akma[idx] if b_i.shape_spec.has_pbc else None
        if box_i is None or n_i == 0:
            return pos_i
        bx = (
            box_i.astype(pos_i.dtype)
            if box_i.ndim == 2
            else jnp.diag(box_i.astype(pos_i.dtype))
        )
        mol_id = (
            b_i.molecule_id[:n_i]
            if b_i.molecule_id is not None
            else jnp.arange(n_i, dtype=jnp.int32)
        )
        masses = jnp.maximum(b_i.masses[:n_i].astype(pos_i.dtype), 1e-12)
        n_mols = int(np.max(np.asarray(mol_id))) + 1
        if n_mols <= 0:
            return pos_i
        m_sum = jax.ops.segment_sum(masses, mol_id, num_segments=n_mols)
        mw_pos = jax.ops.segment_sum(
            pos_i * masses[:, None], mol_id, num_segments=n_mols
        )
        com = mw_pos / jnp.maximum(m_sum[:, None], 1e-12)
        a_v, b_v, c_v = bx[0], bx[1], bx[2]
        diffs = jnp.zeros((n_mols, 3), dtype=pos_i.dtype)
        cz_ok = jnp.abs(c_v[2]) > 1e-12
        diffs = jnp.where(
            cz_ok,
            diffs
            + jnp.floor(com[:, 2:3] / jnp.where(cz_ok, c_v[2], 1.0)) * c_v[None, :],
            diffs,
        )
        by_ok = jnp.abs(b_v[1]) > 1e-12
        diffs = jnp.where(
            by_ok,
            diffs
            + jnp.floor((com[:, 1:2] - diffs[:, 1:2]) / jnp.where(by_ok, b_v[1], 1.0))
            * b_v[None, :],
            diffs,
        )
        ax_ok = jnp.abs(a_v[0]) > 1e-12
        diffs = jnp.where(
            ax_ok,
            diffs
            + jnp.floor((com[:, 0:1] - diffs[:, 0:1]) / jnp.where(ax_ok, a_v[0], 1.0))
            * a_v[None, :],
            diffs,
        )
        return pos_i - diffs[mol_id]

    def getPositionsJax(  # noqa: N802
        self,
        *,
        in_nm: bool = True,
        enforcePeriodicBox: bool = False,
    ) -> jax.Array:
        """Return device `jax.Array` of positions `(B, N_max, 3)` in `nm` (default) or `Å`."""
        self._sync_from_contexts_if_present()
        n_max = max(self._n_atoms_per_replica)
        rows: list[jax.Array] = []
        for i in range(self.batch_size):
            n_i = self._n_atoms_per_replica[i]
            if self._contexts is not None:
                p_i = self._contexts[i].getPositionsJax(
                    in_nm=in_nm, enforcePeriodicBox=enforcePeriodicBox
                )
            else:
                p_i = self._positions_akma[i][:n_i]
                if enforcePeriodicBox:
                    p_i = self._wrap_replica_molecules_dev(i, p_i)
                if in_nm:
                    p_i = p_i * jnp.asarray(ANGSTROM_TO_NM, dtype=p_i.dtype)
            if n_i < n_max:
                p_i = jnp.pad(p_i, ((0, n_max - n_i), (0, 0)))
            rows.append(p_i)
        return jnp.stack(rows, axis=0)

    def setPositionsJax(  # noqa: N802
        self,
        positions: Any,
        *,
        in_nm: bool = True,
    ) -> None:
        """Set positions across all `B` replicas from a `(B, N_max, 3)` or per-replica `jax.Array`/DLPack tensor."""
        n_max = max(self._n_atoms_per_replica)
        per_rep_dev: list[jax.Array] = []
        if isinstance(positions, Sequence) and not hasattr(positions, "__dlpack__"):
            if len(positions) != self.batch_size:
                raise ValueError(
                    f"Expected {self.batch_size} position tensors, got {len(positions)}."
                )
            for i, p_item in enumerate(positions):
                p_d = _from_jax_or_dlpack(p_item)
                n_i = self._n_atoms_per_replica[i]
                if p_d.shape != (n_i, 3):
                    raise openmm.OpenMMException(
                        f"Replica {i} expected positions shape ({n_i}, 3), got {p_d.shape}."
                    )
                per_rep_dev.append(p_d)
        else:
            p_b = _from_jax_or_dlpack(positions)
            if p_b.ndim != 3 or p_b.shape[0] != self.batch_size or p_b.shape[2] != 3:
                raise ValueError(
                    f"Expected batch positions shape ({self.batch_size}, {n_max}, 3), got {p_b.shape}."
                )
            if p_b.shape[1] != n_max:
                raise openmm.OpenMMException(
                    f"Called setPositionsJax() with atom dimension {p_b.shape[1]}, expected {n_max}."
                )
            for i in range(self.batch_size):
                per_rep_dev.append(p_b[i, : self._n_atoms_per_replica[i], :])

        old_specs = [b.shape_spec for b in self._bundles]
        for i, p_dev_i in enumerate(per_rep_dev):
            cs = self._converted_systems[i]
            p_akma = p_dev_i.astype(self._dtype)
            if in_nm:
                p_akma = p_akma * jnp.asarray(NM_TO_ANGSTROM, dtype=self._dtype)
            if cs.virtual_sites is not None:
                p_akma = cs.compute_virtual_sites(p_akma).astype(self._dtype)
            self._positions_akma[i] = p_akma

            b_old = self._bundles[i]
            pad_len = int(b_old.positions.shape[0])
            pos_f32 = _pad_positions_to_bucket(
                p_akma.astype(jnp.float32),
                pad_len,
                b_old.box if b_old.shape_spec.has_pbc else None,
            )
            self._bundles[i] = dataclasses.replace(b_old, positions=pos_f32)
            self._base_bundles[i] = dataclasses.replace(
                self._base_bundles[i], positions=pos_f32
            )
            if self._contexts is not None:
                self._contexts[i].setPositionsJax(p_dev_i, in_nm=in_nm)

        if self.plan is not None:
            new_specs = [b.shape_spec for b in self._bundles]
            if new_specs == old_specs:
                self._refresh_plan_bundles_after_state_update()
            else:
                self._rebuild_plan()

    def getVelocitiesJax(  # noqa: N802
        self,
        *,
        in_nm_per_ps: bool = True,
    ) -> jax.Array:
        """Return device `jax.Array` of velocities `(B, N_max, 3)` in `nm/ps` (default) or `Å/τ_AKMA`."""
        self._sync_from_contexts_if_present()
        n_max = max(self._n_atoms_per_replica)
        rows: list[jax.Array] = []
        for i in range(self.batch_size):
            n_i = self._n_atoms_per_replica[i]
            if self._contexts is not None:
                v_i = self._contexts[i].getVelocitiesJax(in_nm_per_ps=in_nm_per_ps)
            else:
                v_i = self._velocities_akma[i][:n_i]
                if in_nm_per_ps:
                    v_i = v_i * jnp.asarray(VELOCITY_AKMA_TO_OMM, dtype=v_i.dtype)
            if n_i < n_max:
                v_i = jnp.pad(v_i, ((0, n_max - n_i), (0, 0)))
            rows.append(v_i)
        return jnp.stack(rows, axis=0)

    def setVelocitiesJax(  # noqa: N802
        self,
        velocities: Any,
        *,
        in_nm_per_ps: bool = True,
    ) -> None:
        """Set velocities across all `B` replicas from a `(B, N_max, 3)` or per-replica `jax.Array`/DLPack tensor."""
        n_max = max(self._n_atoms_per_replica)
        per_rep_dev: list[jax.Array] = []
        if isinstance(velocities, Sequence) and not hasattr(velocities, "__dlpack__"):
            if len(velocities) != self.batch_size:
                raise ValueError(
                    f"Expected {self.batch_size} velocity tensors, got {len(velocities)}."
                )
            for i, v_item in enumerate(velocities):
                v_d = _from_jax_or_dlpack(v_item)
                n_i = self._n_atoms_per_replica[i]
                if v_d.shape != (n_i, 3):
                    raise openmm.OpenMMException(
                        f"Replica {i} expected velocities shape ({n_i}, 3), got {v_d.shape}."
                    )
                per_rep_dev.append(v_d)
        else:
            v_b = _from_jax_or_dlpack(velocities)
            if v_b.ndim != 3 or v_b.shape[0] != self.batch_size or v_b.shape[2] != 3:
                raise ValueError(
                    f"Expected batch velocities shape ({self.batch_size}, {n_max}, 3), got {v_b.shape}."
                )
            if v_b.shape[1] != n_max:
                raise openmm.OpenMMException(
                    f"Called setVelocitiesJax() with atom dimension {v_b.shape[1]}, expected {n_max}."
                )
            for i in range(self.batch_size):
                per_rep_dev.append(v_b[i, : self._n_atoms_per_replica[i], :])

        for i, v_dev_i in enumerate(per_rep_dev):
            cs = self._converted_systems[i]
            v_akma = v_dev_i.astype(self._dtype)
            if in_nm_per_ps:
                v_akma = v_akma * jnp.asarray(VELOCITY_OMM_TO_AKMA, dtype=self._dtype)
            if cs.virtual_sites is not None:
                v_akma = v_akma.at[cs.virtual_sites.order].set(0.0)
            self._velocities_akma[i] = v_akma
            if self._contexts is not None:
                self._contexts[i].setVelocitiesJax(v_dev_i, in_nm_per_ps=in_nm_per_ps)
        self._velocities_initialized = True

    def getForcesJax(  # noqa: N802
        self,
        *,
        in_kj_per_mol_nm: bool = True,
        groups: int | set[int] = -1,
    ) -> jax.Array:
        """Return device `jax.Array` of forces `(B, N_max, 3)` without materializing `self._contexts`."""
        if isinstance(groups, set):
            raw_groups = sum(1 << g for g in groups if 0 <= g <= 31)
        else:
            raw_groups = int(groups)
        groups_mask = int(np.int32(raw_groups))

        self._sync_from_contexts_if_present()
        n_max = max(self._n_atoms_per_replica)

        if self._contexts is not None:
            rows_ctx: list[jax.Array] = []
            for i, ctx in enumerate(self._contexts):
                f_i = ctx.getForcesJax(
                    in_kj_per_mol_nm=in_kj_per_mol_nm, groups=groups
                )
                n_i = self._n_atoms_per_replica[i]
                if n_i < n_max:
                    f_i = jnp.pad(f_i, ((0, n_max - n_i), (0, 0)))
                rows_ctx.append(f_i)
            return jnp.stack(rows_ctx, axis=0)

        if self._can_vmap_batch_observables(groups_mask):
            cs0 = self._converted_systems[0]
            pos_b = jnp.stack(self._positions_akma, axis=0)
            (
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
            ) = self._stack_batch_inputs_for_vmap()
            ewald_tol = (
                cs0.ewald_error_tolerance
                if cs0.ewald_error_tolerance is not None
                else self.config.ewald_error_tolerance
            )
            eff_nb_method = _effective_cs_nonbonded_method(cs0, self.config)
            eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
            eff_lj_alpha = _resolve_cs_lj_pme_alpha(cs0, self.config)
            eff_lj_dims = _resolve_cs_lj_pme_grid_dims(
                cs0, self.config, self._boxes_akma[0]
            )
            use_disp = bool(
                self.config.use_dispersion_correction and cs0.use_dispersion_correction
            )
            nb_eng = (
                "flash"
                if (
                    cs0.has_pbc
                    and not cs0.is_triclinic
                    and not cs0.has_implicit_solvent
                    and cs0.nonbonded_offsets is None
                    and cs0.lj_combining_rule == "lorentz_berthelot"
                    and not cs0.exceptions_use_pbc
                )
                else (
                    "neighbor_list"
                    if (
                        not cs0.is_triclinic
                        and int(pos_b.shape[1]) >= 4000
                        and cs0.nonbonded_cutoff < float("inf")
                    )
                    else "dense"
                )
            )
            _, f_akma_b = _vmapped_batch_energy_and_forces_akma(
                pos_b,
                box_b,
                base_bundles_b,
                bond_fg_b,
                angle_fg_b,
                dihedral_fg_b,
                cmap_fg_b,
                pr_b,
                offsets_b,
                gp_vec_b,
                param_names,
                cs0.force_group_by_component,
                groups_mask,
                cs0.has_pbc,
                cs0.nonbonded_cutoff,
                eff_nb_method,
                cs0.has_nonbonded,
                use_disp,
                ewald_tol,
                eff_alpha,
                cs0.reaction_field_dielectric,
                cs0.has_implicit_solvent,
                cs0.solvent_dielectric,
                cs0.solute_dielectric,
                nb_eng,
                cs0.pme_grid_dims,
                cs0.lj_switch_width,
                self.config.custom_nonbonded_chunk_size,
                cs0.is_triclinic,
                self.config.precision == "mixed",
                cs0.gbsa_surface_tension,
                cs0.default_parameters,
                cs0.lj_combining_rule,
                cs0.exceptions_use_pbc,
                cs0.dispersion_follows_parameters,
                eff_lj_alpha,
                eff_lj_dims,
            )
            if in_kj_per_mol_nm:
                return f_akma_b * jnp.asarray(FORCE_AKMA_TO_OMM, dtype=f_akma_b.dtype)
            return f_akma_b

        # Cross-shape / ragged batch on `self.plan` (`self._contexts is None`): evaluate on device
        rows_rag: list[jax.Array] = []
        for i in range(self.batch_size):
            cs = self._converted_systems[i]
            b_base_i = self._base_bundles[i]
            n_i = self._n_atoms_per_replica[i]
            pos_i = self._positions_akma[i][:n_i]
            box_i = self._boxes_akma[i] if b_base_i.shape_spec.has_pbc else None
            ewald_tol = (
                cs.ewald_error_tolerance
                if cs.ewald_error_tolerance is not None
                else self.config.ewald_error_tolerance
            )
            gp_i = {
                k: jnp.asarray(v, dtype=pos_i.dtype)
                for k, v in self._parameters_per_replica[i].items()
            }
            eff_nb_i = _effective_cs_nonbonded_method(cs, self.config)
            _, f_akma_i = _pure_compute_energy_and_forces(
                pos_i,
                box_i,
                b_base_i,
                tuple(cs.custom_forces),
                tuple(cs.position_restraints),
                gp_i,
                cs.force_group_by_component,
                groups_mask,
                cs.has_pbc,
                cs.nonbonded_cutoff,
                bond_force_groups=cs.bond_force_groups,
                angle_force_groups=cs.angle_force_groups,
                dihedral_force_groups=cs.dihedral_force_groups,
                cmap_force_groups=cs.cmap_force_groups,
                nonbonded_method=eff_nb_i,
                has_nonbonded=cs.has_nonbonded,
                use_dispersion_correction=(
                    self.config.use_dispersion_correction
                    and cs.use_dispersion_correction
                ),
                ewald_error_tolerance=ewald_tol,
                rf_dielectric=cs.reaction_field_dielectric,
                has_implicit_solvent=cs.has_implicit_solvent,
                solvent_dielectric=cs.solvent_dielectric,
                solute_dielectric=cs.solute_dielectric,
                nonbonded_engine=(
                    "flash"
                    if (
                        cs.has_pbc
                        and not cs.is_triclinic
                        and not cs.has_implicit_solvent
                        and cs.nonbonded_offsets is None
                        and cs.lj_combining_rule == "lorentz_berthelot"
                        and not cs.exceptions_use_pbc
                    )
                    else "dense"
                ),
                pme_grid_dims=cs.pme_grid_dims,
                use_pallas=self.config.use_pallas,
                pallas_interpret=self.config.pallas_interpret,
                lj_switch_width=cs.lj_switch_width,
                use_pallas_pme=self.config.use_pallas_pme,
                custom_nonbonded_chunk_size=self.config.custom_nonbonded_chunk_size,
                is_triclinic=cs.is_triclinic,
                mixed_precision=self.config.precision == "mixed",
                gb_use_neighbor_list=self.config.gb_use_neighbor_list,
                gbsa_surface_tension=cs.gbsa_surface_tension,
                pme_alpha=_resolve_cs_pme_alpha(cs, self.config),
                nonbonded_offsets=cs.nonbonded_offsets,
                default_params=cs.default_parameters,
                lj_combining_rule=cs.lj_combining_rule,
                exceptions_use_pbc=cs.exceptions_use_pbc,
                dispersion_follows_parameters=cs.dispersion_follows_parameters,
                lj_pme_alpha=_resolve_cs_lj_pme_alpha(cs, self.config),
                lj_pme_grid_dims=_resolve_cs_lj_pme_grid_dims(
                    cs, self.config, box_i
                ),
            )
            if cs.virtual_sites is not None:
                f_akma_i = cs.redistribute_virtual_site_forces(f_akma_i, pos_i)
            f_out_i = (
                f_akma_i * jnp.asarray(FORCE_AKMA_TO_OMM, dtype=f_akma_i.dtype)
                if in_kj_per_mol_nm
                else f_akma_i
            )
            if n_i < n_max:
                f_out_i = jnp.pad(f_out_i, ((0, n_max - n_i), (0, 0)))
            rows_rag.append(f_out_i)
        return jnp.stack(rows_rag, axis=0)

    def compute_parameter_derivatives(
        self,
        groups: int | set[int] = -1,
    ) -> _BatchParameterDerivatives:
        """Compute OpenMM global-parameter energy derivatives dU/d(param) across all replicas."""
        if isinstance(groups, set):
            raw_groups = sum(1 << g for g in groups if 0 <= g <= 31)
        else:
            raw_groups = int(groups)
        groups_mask = int(np.int32(raw_groups))

        self._sync_from_contexts_if_present()
        if self._can_vmap_batch_observables(groups_mask):
            return _BatchParameterDerivatives(
                self._compute_vmapped_param_derivatives(groups_mask)
            )
        states = self.getStates(getParameterDerivatives=True, groups=groups)
        return _BatchParameterDerivatives(
            [st.getEnergyParameterDerivatives() for st in states]
        )


BatchSimulation = ProlixBatchSimulation
