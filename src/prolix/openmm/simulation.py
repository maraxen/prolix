"""Drop-in openmm.app.Simulation and multi-replica ProlixBatchSimulation replacements."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as _omm_unit
from openmm import app

import prolix.api.ensemble_plan as _ensemble_plan_mod
from prolix.openmm.config import ProlixOpenMMConfig, parse_platform_properties
from prolix.openmm.context import (
    BOLTZMANN_KCAL,
    ProlixContext,
    ProlixState,
    _eval_system_potential_energy,
    _pure_compute_energy_and_forces,
)
from prolix.openmm.converter import ConvertedOpenMMSystem, from_openmm_system
from prolix.openmm.units import (
    ANGSTROM_TO_NM,
    FORCE_AKMA_TO_OMM,
    FORCE_OMM_TO_AKMA,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    NM_TO_ANGSTROM,
    PS_TO_TAU_AKMA,
    TAU_AKMA_TO_PS,
    VELOCITY_AKMA_TO_OMM,
    VELOCITY_OMM_TO_AKMA,
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


class ProlixSimulation(app.Simulation):
    """Drop-in replacement for openmm.app.Simulation powered by Prolix JAX backend.

    Features:

    - Fused Rollout Bypass: Overrides `_simulate()` to dispatch full reporter intervals
      directly to `context.step(nextSteps)`, eliminating Python 10-step slicing bottlenecks.
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

    def _simulate(
        self,
        endStep: int | None = None,
        endTime: datetime.datetime | None = None,
    ) -> None:
        """Fused rollout bypass eliminating OpenMM's 10-step slicing bottleneck."""
        if endStep is None:
            endStep = sys.maxsize

        next_report: list[dict[str, Any] | None] = [None] * len(self.reporters)

        while self.currentStep < endStep and (
            endTime is None or datetime.datetime.now() < endTime  # noqa: DTZ005
        ):
            next_steps = endStep - self.currentStep
            any_report = False

            # Query all reporters to determine earliest required report step
            for i, reporter in enumerate(self.reporters):
                raw_report = reporter.describeNextReport(self)
                if isinstance(raw_report, tuple):
                    report_dict: dict[str, Any] = {"steps": raw_report[0]}
                    if len(raw_report) > 5:
                        report_dict["periodic"] = raw_report[5]
                    includes = ["positions", "velocities", "forces", "energy"]
                    report_dict["include"] = [
                        includes[j] for j in range(4) if raw_report[j + 1]
                    ]
                else:
                    report_dict = raw_report
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

        gp = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in ctx._parameters.items()}
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
        # Zero velocities after minimization
        ctx._velocities = jnp.zeros_like(ctx._velocities)

    def _minimize_fire(
        self, tol_akma: float, max_iter: int, reporter: Any = None  # noqa: ARG002
    ) -> None:
        """Energy minimization using FIRE descent."""
        from jax_md import minimize as jax_md_minimize

        ctx = self.context
        pos_init = ctx._positions
        max_iterations = max_iter if max_iter > 0 else 500

        gp = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in ctx._parameters.items()}
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
            )

        init_fn, step_fn = jax_md_minimize.fire_descent(
            energy_fn, shift_fn=lambda r, dr: r + dr, dt_start=0.01
        )
        state = init_fn(pos_init)

        for _ in range(max_iterations):
            state = step_fn(state)
            # Check RMS force convergence
            f_norm = float(jnp.sqrt(jnp.mean(jnp.sum(state.force**2, axis=-1))))
            if f_norm <= tol_akma:
                break

        ctx._positions = state.position
        ctx.computeVirtualSites()
        ctx.applyConstraints()
        ctx._init_neighbor_list(ctx._positions)
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


class ProlixBatchSimulation:
    """Multi-replica and heterogeneous OpenMM batch simulation frontend.

    Dispatches `B` OpenMM `System`s (homogeneous replicas, parameter-differentiated
    congeneric ligands, or cross-size ragged topologies) onto a single compiled
    Pallas v3.5 `EnsemblePlan` (`jax.vmap` + `MK-64` zero-stride shared topology
    broadcasting), with automatic stateful `ProlixContext` fallback when custom
    Lepton forces, virtual sites, or barostats are active.
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
            # Ensure each replica owns a distinct position buffer while sharing topology arrays.
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

        self._n_atoms_per_replica: list[int] = [
            sys_obj.getNumParticles() for sys_obj in self.systems
        ]
        self._positions_akma: list[jnp.ndarray] = [
            jnp.zeros((n_i, 3), dtype=self._dtype) for n_i in self._n_atoms_per_replica
        ]
        self._velocities_akma: list[jnp.ndarray] = [
            jnp.zeros((n_i, 3), dtype=self._dtype) for n_i in self._n_atoms_per_replica
        ]
        self._velocities_initialized: bool = False

        # Extract integrator parameters (dt_fs, T_kelvin, gamma_ps) from replica 0
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

        needs_context_fallback = (
            not uniform_integrators
            or self.config.precision == "double"
            or any(
                bool(cs.custom_forces)
                or bool(cs.position_restraints)
                or bool(cs.virtual_sites)
                or cs.barostat is not None
                or cs.is_triclinic
                or cs.nonbonded_method in ("CutoffNonPeriodic", "CutoffPeriodic")
                for cs in self._converted_systems
            )
        )

        # Check whether boxes match when periodic neighbor list is requested
        self._boxes_match: bool = True
        if b_count > 1 and all(b.shape_spec.has_pbc for b in self._bundles):
            ref_box = np.asarray(self._bundles[0].box)
            ref_cut = float(np.asarray(self._bundles[0].cutoff_distance))
            for b_other in self._bundles[1:]:
                if not np.allclose(np.asarray(b_other.box), ref_box, rtol=0.0, atol=1e-5) or not np.isclose(
                    float(np.asarray(b_other.cutoff_distance)), ref_cut, rtol=0.0, atol=1e-5
                ):
                    self._boxes_match = False
                    break

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

    def _rebuild_plan(self) -> None:
        """Construct or refresh `EnsemblePlan` from current `self._bundles`."""
        plan_cls = _ensemble_plan_mod.EnsemblePlan
        eff_pallas = bool(
            self.config.use_pallas or self.config.nonbonded_engine == "morton_tile_pairs"
        )
        eff_pallas_pme = bool(self.config.use_pallas_pme or eff_pallas)
        self.plan = plan_cls.from_bundles(
            list(self._bundles),
            use_pallas=eff_pallas,
            use_pallas_pme=eff_pallas_pme,
        )

    def _ensure_contexts(self) -> list[ProlixContext]:
        """Lazily construct `ProlixContext`s synced with current batch state when needed."""
        if self._contexts is None:
            self._contexts = [
                ProlixContext(self.systems[i], _clone_integrator(self.integrators[i]), config=self.config)
                for i in range(self.batch_size)
            ]
            for i, ctx in enumerate(self._contexts):
                ctx._positions = jnp.asarray(self._positions_akma[i], dtype=ctx._dtype)
                ctx._velocities = jnp.asarray(self._velocities_akma[i], dtype=ctx._dtype)
                ctx._velocities_initialized = self._velocities_initialized
                ctx._step_count = self.currentStep
                ctx._time = self.currentStep * self._dt_akma
                ctx._init_neighbor_list(ctx._positions)
        return self._contexts

    def setPositions(  # noqa: N802
        self,
        positions: Sequence[Any] | None = None,
        positions_list: Sequence[Any] | None = None,
    ) -> None:
        """Set positions (OpenMM nm or `unit.Quantity`) for all `B` replicas."""
        if positions is not None and positions_list is not None:
            raise ValueError("Pass at most one of `positions` or `positions_list`, not both.")
        pos_seq = positions_list if positions_list is not None else positions
        if pos_seq is None:
            raise ValueError("Expected `positions` or `positions_list` sequence.")
        if len(pos_seq) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} position arrays, got {len(pos_seq)}."
            )

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
            if self._contexts is not None:
                self._contexts[i].setPositions(pos_i)

        if self.plan is not None:
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
        """Advance all `B` replicas by `steps` MD steps."""
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
            self.currentStep += steps
            self._velocities_initialized = True
            return

        eff_pallas = bool(
            self.config.use_pallas or self.config.nonbonded_engine == "morton_tile_pairs"
        )
        eff_pallas_pme = bool(self.config.use_pallas_pme or eff_pallas)
        periodic = all(b.shape_spec.has_pbc for b in self._bundles)
        implicit = any(b.shape_spec.has_implicit_solvent for b in self._bundles)

        if self.config.nonbonded_engine == "dense":
            use_nl: bool | None = False
            use_flash: bool | None = False
        elif self.config.nonbonded_engine == "flash" or (
            periodic and not implicit and not self._boxes_match
        ):
            use_nl = False
            use_flash = bool(periodic and not implicit)
        elif self.config.nonbonded_engine in ("neighbor_list", "morton_tile_pairs"):
            use_nl = bool(periodic and not implicit and self._boxes_match)
            use_flash = False if use_nl else bool(periodic and not implicit)
        else:
            use_nl = None
            use_flash = None

        kt_akma = BOLTZMANN_KCAL * self._temperature_k
        step_seed = self._seed + self.currentStep
        run_mode = self.config.execution_mode

        try:
            trajs = self.plan.run(
                steps,
                dt=self._dt_fs,
                kT=kt_akma,
                seed=step_seed,
                dt_unit="fs",
                gamma=self._gamma_ps,
                run_mode=run_mode,
                use_neighbor_list=use_nl,
                nl_update_every=self.config.nl_update_every,
                use_flash_forces=use_flash,
                use_pallas=eff_pallas,
                use_pallas_pme=eff_pallas_pme,
            )
        except ValueError:
            # Fallback to FlashMD when ragged periodic boxes have differing dimensions
            trajs = self.plan.run(
                steps,
                dt=self._dt_fs,
                kT=kt_akma,
                seed=step_seed,
                dt_unit="fs",
                gamma=self._gamma_ps,
                run_mode=run_mode,
                use_neighbor_list=False,
                nl_update_every=self.config.nl_update_every,
                use_flash_forces=bool(periodic and not implicit),
                use_pallas=eff_pallas,
                use_pallas_pme=eff_pallas_pme,
            )

        traj_list = [trajs] if not isinstance(trajs, list) else trajs
        for i, tr in enumerate(traj_list):
            n_i = self._n_atoms_per_replica[i]
            final_pos = tr.positions[-1, :n_i, :]
            self._positions_akma[i] = jnp.asarray(final_pos, dtype=self._dtype)
            if getattr(tr, "velocities", None) is not None:
                self._velocities_akma[i] = jnp.asarray(
                    tr.velocities[-1, :n_i, :], dtype=self._dtype
                )
            b_old = self._bundles[i]
            pad_len = int(b_old.positions.shape[0])
            pos_f32 = _pad_positions_to_bucket(
                jnp.asarray(final_pos, dtype=jnp.float32),
                pad_len,
                b_old.box if b_old.shape_spec.has_pbc else None,
            )
            self._bundles[i] = dataclasses.replace(b_old, positions=pos_f32)

        self.plan.bundles = list(self._bundles)
        self.currentStep += steps
        self._velocities_initialized = True

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
        if self._contexts is not None or getParameters or getParameterDerivatives or groups != -1:
            ctxs = self._ensure_contexts()
            states: list[ProlixState] = []
            for i, ctx in enumerate(ctxs):
                ctx._positions = jnp.asarray(self._positions_akma[i], dtype=ctx._dtype)
                ctx._velocities = jnp.asarray(self._velocities_akma[i], dtype=ctx._dtype)
                ctx._velocities_initialized = self._velocities_initialized
                ctx._step_count = self.currentStep
                ctx._time = self.currentStep * self._dt_akma
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

        time_ps = self.currentStep * self._dt_akma * TAU_AKMA_TO_PS
        out_states: list[ProlixState] = []
        for i in range(self.batch_size):
            cs = self._converted_systems[i]
            b_i = self._bundles[i]
            n_i = self._n_atoms_per_replica[i]
            pos_i = self._positions_akma[i]
            vel_i = self._velocities_akma[i]
            box_i = b_i.box if b_i.shape_spec.has_pbc else None

            pos_nm = None
            if getPositions:
                p_eval = pos_i
                if enforcePeriodicBox and box_i is not None:
                    diag_b = jnp.diag(box_i).astype(pos_i.dtype)
                    mol_id = b_i.molecule_id[:n_i] if b_i.molecule_id is not None else jnp.arange(n_i)
                    # Wrap per-molecule COM into primary box
                    masses = jnp.maximum(b_i.masses[:n_i].astype(pos_i.dtype), 1e-12)
                    n_mols = int(jnp.max(mol_id)) + 1 if n_i > 0 else 0
                    if n_mols > 0:
                        m_sum = jax.ops.segment_sum(masses, mol_id, num_segments=n_mols)
                        mw_pos = jax.ops.segment_sum(pos_i * masses[:, None], mol_id, num_segments=n_mols)
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
                ewald_tol = (
                    cs.ewald_error_tolerance
                    if cs.ewald_error_tolerance is not None
                    else self.config.ewald_error_tolerance
                )
                pe_akma, f_akma = _pure_compute_energy_and_forces(
                    pos_i,
                    box_i,
                    b_i,
                    (),
                    cs.position_restraints,
                    {},
                    cs.force_group_by_component,
                    -1,
                    cs.has_pbc,
                    cs.nonbonded_cutoff,
                    bond_force_groups=cs.bond_force_groups,
                    angle_force_groups=cs.angle_force_groups,
                    dihedral_force_groups=cs.dihedral_force_groups,
                    cmap_force_groups=cs.cmap_force_groups,
                    nonbonded_method=cs.nonbonded_method,
                    has_nonbonded=cs.has_nonbonded,
                    use_dispersion_correction=(
                        self.config.use_dispersion_correction and cs.use_dispersion_correction
                    ),
                    ewald_error_tolerance=ewald_tol,
                    rf_dielectric=cs.reaction_field_dielectric,
                    has_implicit_solvent=cs.has_implicit_solvent,
                    solvent_dielectric=cs.solvent_dielectric,
                    solute_dielectric=cs.solute_dielectric,
                    nonbonded_engine=(
                        "flash"
                        if cs.has_pbc and not cs.has_implicit_solvent
                        else "dense"
                    ),
                    pme_grid_dims=cs.pme_grid_dims,
                    use_pallas=self.config.use_pallas,
                    pallas_interpret=self.config.pallas_interpret,
                    lj_switch_width=cs.lj_switch_width,
                    use_pallas_pme=self.config.use_pallas_pme,
                    custom_nonbonded_chunk_size=self.config.custom_nonbonded_chunk_size,
                    is_triclinic=cs.is_triclinic,
                    gb_use_neighbor_list=self.config.gb_use_neighbor_list,
                    gbsa_surface_tension=cs.gbsa_surface_tension,
                )
                if getForces:
                    forces_omm = np.asarray(f_akma, dtype=np.float64) * FORCE_AKMA_TO_OMM
                if getEnergy:
                    pe_kj = float(pe_akma) * KCAL_PER_MOL_TO_KJ_PER_MOL
                    masses = jnp.asarray(b_i.masses[:n_i], dtype=vel_i.dtype)
                    ke_akma = 0.5 * float(jnp.sum(masses[:, None] * (vel_i**2)))
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
                    parameters={} if getParameters else None,
                    param_derivatives={} if getParameterDerivatives else None,
                )
            )
        return out_states

    def value_and_grad_positions(
        self,
        batch_positions: Sequence[Any] | jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | list[jnp.ndarray]]:
        """Differentiable batch energy (kJ/mol) and position gradients dU/dx (kJ/(mol*nm))."""
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
                        if _ctx._converted_system.has_pbc
                        and not _ctx._converted_system.has_implicit_solvent
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
        states = self.getStates(getParameterDerivatives=True, groups=groups)
        return [st.getEnergyParameterDerivatives() for st in states]


BatchSimulation = ProlixBatchSimulation

