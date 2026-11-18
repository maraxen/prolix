# `prolix.openmm` — Drop-In OpenMM Backend

Drop-in JAX and Pallas GPU replacements for `openmm.Platform`, `openmm.Context`, `openmm.State`, and `openmm.app.Simulation`, plus vectorized multi-replica and congeneric batch execution via `ProlixBatchSimulation`.

```{eval-rst}
.. currentmodule:: prolix.openmm

.. autosummary::
   :nosignatures:

   ProlixPlatform
   ProlixOpenMMConfig
   parse_platform_properties
   ProlixContext
   ProlixState
   ProlixSimulation
   ProlixBatchSimulation
   ConvertedOpenMMSystem
   from_openmm_system
   UnsupportedOpenMMForceError
   to_akma
   from_akma
```

## Platform & Typed Configuration

```{eval-rst}
.. autoclass:: prolix.openmm.ProlixPlatform
   :members:
   :special-members: __init__

.. autoclass:: prolix.openmm.ProlixOpenMMConfig
   :members:

.. autofunction:: prolix.openmm.parse_platform_properties
```

## Context, State, & Single-System Simulation

```{eval-rst}
.. autoclass:: prolix.openmm.ProlixContext
   :members: getPlatform, getSystem, getIntegrator, setPositions, setVelocities, setVelocitiesToTemperature, setPeriodicBoxVectors, setParameter, getParameter, getParameters, getState, step, applyConstraints, applyVelocityConstraints, computeVirtualSites, reinitialize, createCheckpoint, loadCheckpoint, updateParametersInContext, create_ensemble_plan
   :undoc-members:
   :special-members: __init__

.. autoclass:: prolix.openmm.ProlixState
   :members: getTime, getStepCount, getPeriodicBoxVectors, getPeriodicBoxVolume, getPositions, getVelocities, getForces, getPotentialEnergy, getKineticEnergy, getParameters, getEnergyParameterDerivatives
   :undoc-members:

.. autoclass:: prolix.openmm.ProlixSimulation
   :members: currentStep, step, minimizeEnergy
   :undoc-members:
   :special-members: __init__
```

## Multi-System & Ragged Batch Simulation

`ProlixBatchSimulation` provides persistent GPU ensemble stepping (`step`), per-replica state & coordinate management (`batch_size`, `n_replicas`, `bundles`, `setPositions`, `setVelocities`, `setVelocitiesToTemperature`, `getStates`), batch energy minimization & checkpoint/state serialization (`minimizeEnergy`, `saveStates`, `loadStates`, `createCheckpoint`, `loadCheckpoint`, `saveCheckpoint`), and vectorized alchemical / force-field differentiation (`setParameter`, `getParameters`, `value_and_grad_positions`, `compute_parameter_derivatives`).

```{eval-rst}
.. autoclass:: prolix.openmm.ProlixBatchSimulation
   :members:
   :undoc-members:
   :special-members: __init__
```

## System Converter, Exceptions, & Unit Helpers

```{eval-rst}
.. autoclass:: prolix.openmm.ConvertedOpenMMSystem
   :members:

.. autofunction:: prolix.openmm.from_openmm_system

.. autoclass:: prolix.openmm.UnsupportedOpenMMForceError
   :show-inheritance:

.. autofunction:: prolix.openmm.to_akma

.. autofunction:: prolix.openmm.from_akma
```
