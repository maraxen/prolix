"""EnsemblePlan: high-level API for batch molecular dynamics simulations.

Status: v1.1 stub. run() implementation deferred to Sprint 38 pending xtrax.tiling
integration (#1842). For now, use settle_langevin directly.
"""

from __future__ import annotations

from typing import Any


class EnsemblePlan:
    """Orchestrates batch MD simulations over multiple MolecularBundle instances.

    This is a stub for the v1.1 API layer. The full run() implementation will
    integrate with a mature planner backend (either prolix.tiling.BatchPlanner or
    xtrax.BatchPlanner) once #1842 (xtrax.tiling integration) is resolved.

    Args:
        bundles: List of MolecularBundle instances to simulate in parallel.
        planner: Optional planner backend (BatchPlanner or compatible). If provided,
                 planner.plan(bundles) is called immediately. If None, batch_plan is
                 set to None (deferred to Sprint 38).

    Attributes:
        bundles: The input bundle list.
        batch_plan: Result of planner.plan() if planner was provided, else None.
    """

    def __init__(self, bundles: list, planner: Any = None) -> None:
        """Initialize EnsemblePlan with bundles and optional planner.

        Args:
            bundles: List of MolecularBundle instances.
            planner: Optional planner with a plan() method. If provided,
                     self.batch_plan = planner.plan(bundles).
                     If None, self.batch_plan = None.
        """
        self.bundles = bundles
        if planner is not None:
            self.batch_plan = planner.plan(bundles)
        else:
            self.batch_plan = None

    def run(
        self,
        n_steps: int,
        dt: float,
        kT: float,
        seed: int = 0,
    ) -> dict:
        """Run batch MD simulation over all bundles.

        Status: NOT IMPLEMENTED. Full integration with planner backend pending
        xtrax.tiling resolution (#1842) in Sprint 38.

        Expected behavior (v1.1):
          - Use self.batch_plan to decide vmap vs safe_map per axis
          - Compile batched integrator (settle_langevin or settle_csvr_npt)
          - Execute n_steps on GPU/TPU
          - Return trajectory dict with energies, temperatures, final state

        For now: Use settle_langevin directly on unbatched systems.

        Args:
            n_steps: Number of MD steps.
            dt: Timestep in AKMA units (1.0 fs).
            kT: Thermal energy (default 300 K ≈ 2.479e-3 AKMA).
            seed: PRNG seed for thermostat noise.

        Raises:
            NotImplementedError: Always. Full implementation pending #1842.
        """
        raise NotImplementedError(
            "EnsemblePlan.run() implementation pending xtrax.tiling integration (#1842). "
            "Expected in Sprint 38. Use settle_langevin directly for now."
        )
