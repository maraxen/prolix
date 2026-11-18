"""Drop-in openmm.app.Simulation and multi-replica ProlixBatchSimulation replacements."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import inspect
import math
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as _omm_unit
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
    _pure_compute_energy_and_forces,
    _wrap_molecules_jax,
)
from prolix.openmm.converter import (
    ConvertedOpenMMSystem,
    NonbondedParameterOffsets,
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
        if ctx._morton_box_diag is not None and not ctx._is_triclinic:
            box_akma_d: jnp.ndarray | None = jnp.diag(
                ctx._morton_box_diag.astype(raw_pos_d.dtype)
            )
        elif ctx._box_vectors is not None:
            box_akma_d = ctx._box_vectors.astype(raw_pos_d.dtype)
        else:
            box_akma_d = None

        pos_wrapped_d: jnp.ndarray | None = None
        if need_wrapped and req_pos_w and has_box and not ctx._is_triclinic and box_akma_d is not None:
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
                runner = cast("dict[str, Any]", ctx._morton_runner)
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
                    box_diag_in = ctx._morton_box_diag
                    if box_diag_in is None and ctx._box_vectors is not None:
                        box_diag_in = (
                            jnp.diag(ctx._box_vectors)
                            if ctx._box_vectors.ndim == 2
                            else ctx._box_vectors
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
                    ctx._converted_system.nonbonded_method,
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
                if meta["is_triclinic"] and host_dev["box_akma"] is not None:
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
                elif host_dev["pos_wrapped"] is not None:
                    pos_nm = np.asarray(host_dev["pos_wrapped"], dtype=np.float64) * ANGSTROM_TO_NM
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

    def _minimize_lbfgs(
        self, tol_akma: float, max_iter: int, reporter: Any = None  # noqa: ARG002
    ) -> None:
        """Energy minimization using GPU jaxopt.LBFGS."""
        import jaxopt

        ctx = self.context
        pos_init = ctx._positions
        max_iterations = max_iter if max_iter > 0 else 500

        gp = ctx._get_dynamic_global_params()
        ewald_tol = (
            ctx._converted_system.ewald_error_tolerance
            if ctx._converted_system.ewald_error_tolerance is not None
            else ctx._config.ewald_error_tolerance
        )

        def energy_fn(p: jnp.ndarray) -> jnp.ndarray:
            return _eval_system_potential_energy(
                p,
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
                nonbonded_method=ctx._converted_system.nonbonded_method,
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
                pme_alpha=ctx._effective_pme_alpha(),
                nonbonded_offsets=ctx._converted_system.nonbonded_offsets,
                default_params=ctx._converted_system.default_parameters,
                lj_combining_rule=ctx._lj_combining_rule,
                exceptions_use_pbc=ctx._exceptions_use_pbc,
                dispersion_follows_parameters=ctx._dispersion_follows_parameters,
            )

        solver = jaxopt.LBFGS(
            fun=energy_fn,
            maxiter=max_iterations,
            tol=tol_akma,
            linesearch="backtracking",
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
        from jax_md import minimize as jax_md_minimize

        ctx = self.context
        pos_init = ctx._positions
        max_iterations = max_iter if max_iter > 0 else 500

        gp = ctx._get_dynamic_global_params()
        ewald_tol = (
            ctx._converted_system.ewald_error_tolerance
            if ctx._converted_system.ewald_error_tolerance is not None
            else ctx._config.ewald_error_tolerance
        )

        def energy_fn(p: jnp.ndarray) -> jnp.ndarray:
            return _eval_system_potential_energy(
                p,
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
                nonbonded_method=ctx._converted_system.nonbonded_method,
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
                pme_alpha=ctx._effective_pme_alpha(),
                nonbonded_offsets=ctx._converted_system.nonbonded_offsets,
                default_params=ctx._converted_system.default_parameters,
                lj_combining_rule=ctx._lj_combining_rule,
                exceptions_use_pbc=ctx._exceptions_use_pbc,
                dispersion_follows_parameters=ctx._dispersion_follows_parameters,
            )

        init_fn, step_fn = jax_md_minimize.fire_descent(
            energy_fn, shift_fn=lambda r, dr: r + dr, dt_start=0.01
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


def _clone_integrator(integrator: openmm.Integrator | str) -> openmm.Integrator:
    """Deserialize or clone an OpenMM `Integrator` so each replica has independent state."""
    omm_mod = cast("Any", openmm)
    if isinstance(integrator, str):
        return omm_mod.XmlSerializer.deserialize(integrator)
    return omm_mod.XmlSerializer.deserialize(omm_mod.XmlSerializer.serialize(integrator))


def _resolve_cs_pme_alpha(cs: ConvertedOpenMMSystem, config: ProlixOpenMMConfig) -> float | None:
    """Resolve effective Ewald alpha (Å⁻¹) for a `ConvertedOpenMMSystem`."""
    if str(cs.nonbonded_method).lower() not in ("pme", "spme"):
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


@eqx.filter_jit
def _vmapped_batch_energy_and_forces_akma(  # noqa: PLR0913
    pos_b: jnp.ndarray,
    box_b: jnp.ndarray | None,
    base_bundles_b: MolecularBundle,
    bond_fg_b: jnp.ndarray | None,
    angle_fg_b: jnp.ndarray | None,
    dihedral_fg_b: jnp.ndarray | None,
    cmap_fg_b: jnp.ndarray | None,
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
                (),
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
                (),
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
            )
            return u_kcal * float(KCAL_PER_MOL_TO_KJ_PER_MOL)

        return jax.grad(_e_kj)(gp_v_i)

    in_axes = (0, 0 if box_b is not None else None, 0, 0, 0)
    return jax.vmap(_one_grad, in_axes=in_axes)(
        pos_b, box_b, base_bundles_b, offsets_b, gp_vec_b
    )


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
                conv_cache[sid] = from_openmm_system(sys_obj, config=self.config)
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
                or cs.is_triclinic
                or str(cs.nonbonded_method).lower() in ("cutoff", "lj_pme", "cutoffnonperiodic")
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
        need_morton = bool(
            periodic
            and not implicit
            and (self._pressure_atm is not None or self._is_reaction_field)
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

    def _build_knob_surface(self, eff_engine: str, eff_pallas: bool, eff_pallas_pme: bool) -> KnobSurface:
        """Construct composite `KnobSurface` carrying RF, PME, dispersion, and barostat knobs."""
        cs0 = self._converted_systems[0]
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
                if ctx._morton_box_diag is not None and not ctx._is_triclinic:
                    self._boxes_akma[i] = jnp.diag(ctx._morton_box_diag.astype(self._dtype))
                elif ctx._box_vectors is not None:
                    self._boxes_akma[i] = ctx._box_vectors.astype(self._dtype)
            self.currentStep += steps
            self._velocities_initialized = True
            return

        eff_engine, eff_pallas, eff_pallas_pme = self._effective_plan_engine_and_pallas()
        periodic = all(b.shape_spec.has_pbc for b in self._bundles)
        implicit = any(b.shape_spec.has_implicit_solvent for b in self._bundles)
        need_morton = bool(self._pressure_atm is not None or self._is_reaction_field)

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
            # Replicas diverged in box size on an earlier NPT step: run 1-bundle plans
            # per replica so Morton SoA NPT/RF stays active with zero ProlixContext fallback.
            plan_bundles_pre = self._bundles_for_plan()
            if self._per_replica_plans is None:
                plan_cls = _ensemble_plan_mod.EnsemblePlan
                ks = self._build_knob_surface(eff_engine, eff_pallas, eff_pallas_pme)
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
                        knob_surface=ks,
                    )
                    for i in range(self.batch_size)
                ]
            traj_list = []
            for i, p_i in enumerate(self._per_replica_plans):
                p_i.bundles = [plan_bundles_pre[i]]
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
                    **extra_run_kw,
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

    def _can_vmap_batch_observables(self, groups_mask: int) -> bool:  # noqa: PLR0911
        """Return True when all `B` replicas share uniform shapes & non-Lepton physics for `jax.vmap`."""
        if self._contexts is not None:
            return False
        if len(set(self._n_atoms_per_replica)) != 1:
            return False
        if len({b.shape_spec for b in self._base_bundles}) != 1:
            return False

        cs0 = self._converted_systems[0]
        if bool(cs0.custom_forces) or bool(cs0.position_restraints) or cs0.virtual_sites is not None:
            return False
        if str(cs0.nonbonded_method).lower() == "lj_pme":
            return False

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
                or bool(csi.position_restraints)
                or csi.virtual_sites is not None
                or csi.has_pbc != cs0.has_pbc
                or csi.is_triclinic != cs0.is_triclinic
                or csi.nonbonded_method != cs0.nonbonded_method
                or csi.has_nonbonded != cs0.has_nonbonded
                or csi.nonbonded_cutoff != cs0.nonbonded_cutoff
                or csi.lj_switch_width != cs0.lj_switch_width
                or csi.use_dispersion_correction != cs0.use_dispersion_correction
                or csi.ewald_error_tolerance != cs0.ewald_error_tolerance
                or csi.explicit_pme_alpha != cs0.explicit_pme_alpha
                or csi.pme_grid_dims != cs0.pme_grid_dims
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
        NonbondedParameterOffsets | None,
        jnp.ndarray,
        tuple[str, ...],
    ]:
        """Stack replica bundles, force-group masks, offsets, and global parameter vectors."""
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
            offsets_b,
            gp_vec_b,
            param_names,
        ) = self._stack_batch_inputs_for_vmap()
        ewald_tol = (
            cs0.ewald_error_tolerance
            if cs0.ewald_error_tolerance is not None
            else self.config.ewald_error_tolerance
        )
        eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
        use_disp = bool(
            self.config.use_dispersion_correction and cs0.use_dispersion_correction
        )
        nb_eng = (
            "neighbor_list"
            if (int(pos_b.shape[1]) >= 4000 and cs0.nonbonded_cutoff < float("inf"))
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
                cs0.nonbonded_method,
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

        can_vmap = self._can_vmap_batch_observables(groups_mask)
        if self._contexts is not None or (
            not can_vmap and (getParameterDerivatives or groups_mask != -1)
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
                offsets_b,
                gp_vec_b,
                param_names,
            ) = self._stack_batch_inputs_for_vmap()
            ewald_tol = (
                cs0.ewald_error_tolerance
                if cs0.ewald_error_tolerance is not None
                else self.config.ewald_error_tolerance
            )
            eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
            use_disp = bool(
                self.config.use_dispersion_correction and cs0.use_dispersion_correction
            )
            nb_eng = (
                "flash"
                if (
                    cs0.has_pbc
                    and not cs0.has_implicit_solvent
                    and cs0.nonbonded_offsets is None
                    and cs0.lj_combining_rule == "lorentz_berthelot"
                    and not cs0.exceptions_use_pbc
                )
                else (
                    "neighbor_list"
                    if (int(pos_b.shape[1]) >= 4000 and cs0.nonbonded_cutoff < float("inf"))
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
                offsets_b,
                gp_vec_b,
                param_names,
                cs0.force_group_by_component,
                groups_mask,
                cs0.has_pbc,
                cs0.nonbonded_cutoff,
                cs0.nonbonded_method,
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
            )
            pe_host, f_host = jax.device_get((pe_akma_b, f_akma_b))
            pe_np_b = np.asarray(pe_host, dtype=np.float64) * KCAL_PER_MOL_TO_KJ_PER_MOL
            forces_np_b = np.asarray(f_host, dtype=np.float64) * FORCE_AKMA_TO_OMM

        derivs_per_rep: list[dict[str, float]] | None = None
        if getParameterDerivatives:
            derivs_per_rep = self._compute_vmapped_param_derivatives(groups_mask)

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
                p_eval = pos_i
                if enforcePeriodicBox and box_i is not None:
                    diag_b = jnp.diag(box_i).astype(pos_i.dtype)
                    mol_id = (
                        b_i.molecule_id[:n_i]
                        if b_i.molecule_id is not None
                        else jnp.arange(n_i, dtype=jnp.int32)
                    )
                    masses = jnp.maximum(b_i.masses[:n_i].astype(pos_i.dtype), 1e-12)
                    n_mols = int(jnp.max(mol_id)) + 1 if n_i > 0 else 0
                    if n_mols > 0:
                        m_sum = jax.ops.segment_sum(masses, mol_id, num_segments=n_mols)
                        mw_pos = jax.ops.segment_sum(
                            pos_i * masses[:, None], mol_id, num_segments=n_mols
                        )
                        com = mw_pos / jnp.maximum(m_sum[:, None], 1e-12)
                        shift = jnp.floor(com / diag_b[None, :]) * diag_b[None, :]
                        p_eval = pos_i - shift[mol_id]
                pos_nm = np.asarray(p_eval, dtype=np.float64) * ANGSTROM_TO_NM

            vel_omm = None
            if getVelocities:
                vel_omm = np.asarray(vel_i, dtype=np.float64) * VELOCITY_AKMA_TO_OMM

            forces_omm = None
            pe_kj: float | None = None
            ke_kj: float | None = None
            if getForces or getEnergy:
                if pe_np_b is not None and forces_np_b is not None:
                    if getForces:
                        forces_omm = forces_np_b[i]
                    if getEnergy:
                        pe_kj = float(pe_np_b[i])
                        masses_np = np.asarray(b_i.masses[:n_i], dtype=np.float64)
                        vel_np = np.asarray(vel_i, dtype=np.float64)
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
                        nonbonded_method=cs.nonbonded_method,
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
                offsets_b,
                gp_vec_b,
                param_names,
            ) = self._stack_batch_inputs_for_vmap()
            ewald_tol = (
                cs0.ewald_error_tolerance
                if cs0.ewald_error_tolerance is not None
                else self.config.ewald_error_tolerance
            )
            eff_alpha = _resolve_cs_pme_alpha(cs0, self.config)
            use_disp = bool(
                self.config.use_dispersion_correction and cs0.use_dispersion_correction
            )
            nb_eng = (
                "flash"
                if (
                    cs0.has_pbc
                    and not cs0.has_implicit_solvent
                    and cs0.nonbonded_offsets is None
                    and cs0.lj_combining_rule == "lorentz_berthelot"
                    and not cs0.exceptions_use_pbc
                )
                else (
                    "neighbor_list"
                    if (int(pos_nm_b.shape[1]) >= 4000 and cs0.nonbonded_cutoff < float("inf"))
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
                offsets_b,
                gp_vec_b,
                param_names,
                cs0.force_group_by_component,
                cs0.has_pbc,
                cs0.nonbonded_cutoff,
                cs0.nonbonded_method,
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
            eff_alpha = ctx._effective_pme_alpha()
            gp = {k: jnp.asarray(v, dtype=ctx._dtype) for k, v in ctx._parameters.items()}

            def _u_nm(
                p_nm: jnp.ndarray,
                _ctx: ProlixContext = ctx,
                _gp: dict[str, jnp.ndarray] = gp,
                _et: float = ewald_tol,
                _pa: float | None = eff_alpha,
            ) -> jnp.ndarray:
                p_a = p_nm * NM_TO_ANGSTROM
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
                    nonbonded_method=_ctx._converted_system.nonbonded_method,
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

    def compute_parameter_derivatives(
        self,
        groups: int | set[int] = -1,
    ) -> list[dict[str, float]]:
        """Compute OpenMM global-parameter energy derivatives dU/d(param) across all replicas."""
        if isinstance(groups, set):
            raw_groups = sum(1 << g for g in groups if 0 <= g <= 31)
        else:
            raw_groups = int(groups)
        groups_mask = int(np.int32(raw_groups))

        if self._can_vmap_batch_observables(groups_mask):
            return self._compute_vmapped_param_derivatives(groups_mask)
        states = self.getStates(getParameterDerivatives=True, groups=groups)
        return [st.getEnergyParameterDerivatives() for st in states]


BatchSimulation = ProlixBatchSimulation
