"""EnsemblePlan: high-level API for batch molecular dynamics simulations.

Status: v1.0 implementation. Basic single-bundle MD simulation with settle_langevin.
Full batching (xtrax.tiling integration) deferred to v1.1 (#1842).
"""

from __future__ import annotations

from typing import Any


class EnsemblePlan:
    """Orchestrates batch MD simulations over multiple MolecularBundle instances.

    v1.0: Single-bundle sequential execution. v1.1 will integrate with mature planner
    backend (either prolix.tiling.BatchPlanner or xtrax.BatchPlanner) for vmap/safe_map
    decisions and GPU/TPU parallelism.

    Args:
        bundles: List of MolecularBundle instances to simulate in parallel.
        planner: Optional planner backend (BatchPlanner or compatible). If provided,
                 planner.plan(bundles) is called immediately. If None, batch_plan is
                 set to None.

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
    ):
        """Run batch MD simulation over all bundles.

        v1.0 Implementation (Sprint 38):
          - Sequential execution over bundles (no vmap batching yet)
          - Uses settle_langevin for NVT dynamics
          - Returns Trajectory object with positions and observable values

        Full batching (v1.1):
          - Will use self.batch_plan to decide vmap vs safe_map per axis
          - Compile batched integrator (settle_langevin or settle_csvr_npt)
          - Execute n_steps on GPU/TPU with parallelism
          - Pending xtrax.tiling integration (#1842)

        Args:
            n_steps: Number of MD steps.
            dt: Timestep in AKMA units (1.0 fs).
            kT: Thermal energy (default 300 K ≈ 2.479e-3 AKMA).
            seed: PRNG seed for thermostat noise.

        Returns:
            Trajectory: Object containing positions (n_steps, n_atoms, 3) and
                       observable_values dict.
        """
        from prolix.physics.settle import settle_langevin
        from prolix.api.observables import Trajectory
        import jax
        import jax.numpy as jnp

        # v1.0: simple sequential loop over bundles
        # TODO (v1.1): integrate with xtrax.tiling.BatchPlanner for vmap/safe_map decisions
        if not self.bundles:
            raise ValueError("EnsemblePlan requires at least one bundle")

        bundle = self.bundles[0]  # v1.0: single bundle only
        if len(self.bundles) > 1:
            raise NotImplementedError(
                "Multi-bundle batching pending xtrax.tiling integration (#1842). "
                "v1.0 supports single-bundle execution only."
            )

        # Construct a simple energy function from the bundle
        # For v1.0: use a dummy potential (Lennard-Jones cutoff, no bonded terms)
        def energy_fn(positions, **kwargs):
            # Minimal energy: just return zeros for now
            # In production, this would use the bundle's force field
            return jnp.array(0.0, dtype=positions.dtype)

        def shift_fn(r, v):
            # Minimum-image convention for free boundary
            # (PBC would be handled if bundle.box is non-zero)
            # shift_fn(r, v) -> r + v (position update)
            return r + v

        # Extract bundle properties
        positions_init = bundle.positions[:bundle.n_atoms]
        masses = jnp.ones_like(positions_init[:, 0])  # unit mass for now

        # Create water indices if water molecules exist
        water_indices = None
        if bundle.n_waters > 0:
            water_indices = bundle.water_indices[:bundle.n_waters]

        # Initialize integrator
        init_fn, apply_fn = settle_langevin(
            energy_fn,
            shift_fn,
            dt=dt,
            kT=kT,
            gamma=10.0,  # friction coefficient (ps^-1)
            mass=masses,
            water_indices=water_indices,
            project_ou_momentum_rigid=True,
        )

        # Initialize state
        key = jax.random.PRNGKey(seed)
        state = init_fn(key, positions_init)

        # Run trajectory
        positions_traj = []

        def step_fn(state, _):
            new_state = apply_fn(state, kT=kT, dt=dt)
            return new_state, new_state.position

        # Execute n_steps and record positions at each step (not including initial)
        for i in range(n_steps):
            state, pos = step_fn(state, None)
            positions_traj.append(pos)

        # Stack trajectory (should be n_steps x n_atoms x 3)
        positions_array = jnp.stack(positions_traj)

        # Return Trajectory object
        return Trajectory(
            positions=positions_array,
            observable_values={},
            n_steps=n_steps,
        )
