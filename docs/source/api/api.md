# `prolix.api` — Ensemble Planning & Observables

Top-level JAX orchestration interfaces for single-system, homogeneous multi-replica, `stride=0` congeneric, and 1D ragged batch molecular dynamics.

```{eval-rst}
.. currentmodule:: prolix.api

.. autosummary::
   :nosignatures:

   EnsemblePlan
   Trajectory
   Observable
   Temperature
   Energy
   KineticEnergy
   RMSD
   Pressure
```

---

## Ensemble Execution Planner

```{eval-rst}
.. autoclass:: prolix.api.EnsemblePlan
   :members:
   :special-members: __init__
```

---

## Trajectory & Streaming Observables

```{eval-rst}
.. autoclass:: prolix.api.Trajectory
   :members:

.. autoclass:: prolix.api.Observable
   :members:

.. autoclass:: prolix.api.Temperature
   :members:

.. autoclass:: prolix.api.Energy
   :members:

.. autoclass:: prolix.api.KineticEnergy
   :members:

.. autoclass:: prolix.api.RMSD
   :members:

.. autoclass:: prolix.api.Pressure
   :members:
```

---

## Export-Shaped Trajectory Runners (`prolix.api.export_run`)

Stateless `(seed, dt_fs, kT_kcal)` callables designed for `jax.jit.lower` and StableHLO / WebAssembly compilation without retracing across thermodynamic sweeps.

```{eval-rst}
.. automodule:: prolix.api.export_run
   :members: make_single_trajectory_fn, make_hetero_trajectory_fn, make_smoke_diagnostics_fn
```
