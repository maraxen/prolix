"""Drop-in openmm.app.Simulation replacement with fused rollout bypass and energy minimization."""

from __future__ import annotations

import contextlib
import datetime
import sys
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import openmm
from openmm import app, unit

from prolix.openmm.config import ProlixOpenMMConfig, parse_platform_properties
from prolix.openmm.context import ProlixContext, _eval_system_potential_energy
from prolix.openmm.units import (
    FORCE_OMM_TO_AKMA,
)

if TYPE_CHECKING:
    from openmm.app import Topology


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
        self.topology = topology

        # Resolve system if string
        if isinstance(system, str):
            self.system = openmm.XmlSerializer.deserialize(system)
        else:
            self.system = system

        # Resolve integrator if string
        if isinstance(integrator, str):
            self.integrator = openmm.XmlSerializer.deserialize(integrator)
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
            hasattr(self.topology, "getPeriodicBoxVectors")
            and self.topology.getPeriodicBoxVectors() is None
        ):
            with contextlib.suppress(Exception):
                raw_bv = self.system.getDefaultPeriodicBoxVectors()
                if raw_bv is not None:
                    self.topology.setPeriodicBoxVectors(raw_bv)

        if state is not None:
            if isinstance(state, str):
                self.context.setState(openmm.XmlSerializer.deserialize(state))
            else:
                self.context.setState(state)

        try:
            self._usesPBC = self.system.usesPeriodicBoundaryConditions()
        except Exception:
            self._usesPBC = topology.getUnitCellDimensions() is not None

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

        next_report = [None] * len(self.reporters)

        while self.currentStep < endStep and (
            endTime is None or datetime.datetime.now() < endTime  # noqa: DTZ005
        ):
            next_steps = endStep - self.currentStep
            any_report = False

            # Query all reporters to determine earliest required report step
            for i, reporter in enumerate(self.reporters):
                report = reporter.describeNextReport(self)
                if isinstance(report, tuple):
                    report_dict = {"steps": report[0]}
                    if len(report) > 5:
                        report_dict["periodic"] = report[5]
                    includes = ["positions", "velocities", "forces", "energy"]
                    report_dict["include"] = [
                        includes[j] for j in range(4) if report[j + 1]
                    ]
                    report = report_dict
                next_report[i] = report

                steps = next_report[i]["steps"]
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
                    if report["steps"] == next_steps:
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

                if len(wrapped) > 0:
                    self._generate_reports(wrapped, True)
                if len(unwrapped) > 0:
                    self._generate_reports(unwrapped, False)

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

        def energy_fn(p: jnp.ndarray) -> jnp.ndarray:
            return _eval_system_potential_energy(
                p,
                ctx._box_vectors,
                ctx._bundle,
                ctx._custom_forces,
                ctx._position_restraints,
                gp,
                ctx._converted_system.force_group_by_component,
                0xFFFFFFFF,
                ctx._converted_system.has_pbc,
                ctx._converted_system.nonbonded_cutoff,
            )

        solver = jaxopt.LBFGS(
            fun=energy_fn,
            maxiter=max_iterations,
            tol=tol_akma,
            linesearch="backtracking",
        )
        res = solver.run(pos_init)
        min_pos = res.params

        # Project constraints & virtual sites
        ctx._positions = min_pos
        ctx.computeVirtualSites()
        ctx.applyConstraints()
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

        def energy_fn(p: jnp.ndarray) -> jnp.ndarray:
            return _eval_system_potential_energy(
                p,
                ctx._box_vectors,
                ctx._bundle,
                ctx._custom_forces,
                ctx._position_restraints,
                gp,
                ctx._converted_system.force_group_by_component,
                0xFFFFFFFF,
                ctx._converted_system.has_pbc,
                ctx._converted_system.nonbonded_cutoff,
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
        ctx._velocities = jnp.zeros_like(ctx._velocities)

    def _minimize_openmm_cpu(
        self, tol_omm: float, max_iter: int, reporter: Any = None
    ) -> None:
        """Fallback minimization using OpenMM Reference / CPU platform."""
        pos_nm = self.context.getState(getPositions=True).getPositions()
        platform = openmm.Platform.getPlatformByName("Reference")
        cpu_ctx = openmm.Context(
            self.system, openmm.VerletIntegrator(0.001), platform
        )
        cpu_ctx.setPositions(pos_nm)
        if self.context._box_vectors is not None:
            bv = self.context.getState().getPeriodicBoxVectors()
            cpu_ctx.setPeriodicBoxVectors(bv[0], bv[1], bv[2])

        if reporter is not None:
            openmm.LocalEnergyMinimizer.minimize(cpu_ctx, tol_omm, max_iter, reporter)
        else:
            openmm.LocalEnergyMinimizer.minimize(cpu_ctx, tol_omm, max_iter)
        min_state = cpu_ctx.getState(getPositions=True)
        self.context.setPositions(min_state.getPositions(asNumpy=True))
        self.context._velocities = jnp.zeros_like(self.context._velocities)
