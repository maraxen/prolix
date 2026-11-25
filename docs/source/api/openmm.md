# `prolix.openmm` — Drop-In OpenMM Backend

Drop-in JAX and Pallas GPU replacements for `openmm.Platform`, `openmm.Context`, `openmm.State`, and `openmm.app.Simulation`, plus vectorized multi-replica, cross-shape triclinic, and sparse-alchemical batch execution via `ProlixBatchSimulation`. Supports orthorhombic and reduced-form triclinic periodic boxes, isotropic/anisotropic/membrane Monte Carlo barostats, `dt = 4.0 fs` Hydrogen Mass Repartitioning (`HMR`), and zero-copy JAX/DLPack device state access (`getPositionsJax`, `setPositionsJax`, `getVelocitiesJax`, `setVelocitiesJax`, `getForcesJax`).

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

`ProlixContext` and `ProlixSimulation` run orthorhombic and reduced-form triclinic explicit-solvent systems, isotropic/anisotropic/membrane Monte Carlo barostats, and sparse `InteractionGroup` custom forces (`is_gpu_runner_fast`) on persistent GPU runners, and expose zero-copy JAX/DLPack buffers (`getPositionsJax`, `setPositionsJax`, `getVelocitiesJax`, `setVelocitiesJax`, `getForcesJax`).

```{eval-rst}
.. autoclass:: prolix.openmm.ProlixContext
   :members: getPlatform, getSystem, getIntegrator, setPositions, setVelocities, setVelocitiesToTemperature, setPeriodicBoxVectors, getPositionsJax, setPositionsJax, getVelocitiesJax, setVelocitiesJax, getForcesJax, setParameter, getParameter, getParameters, getState, step, applyConstraints, applyVelocityConstraints, computeVirtualSites, reinitialize, createCheckpoint, loadCheckpoint, updateParametersInContext, create_ensemble_plan
   :undoc-members:
   :special-members: __init__

.. autoclass:: prolix.openmm.ProlixState
   :members: getTime, getStepCount, getPeriodicBoxVectors, getPeriodicBoxVolume, getPositions, getVelocities, getForces, getPotentialEnergy, getKineticEnergy, getParameters, getEnergyParameterDerivatives
   :undoc-members:

.. autoclass:: prolix.openmm.ProlixSimulation
   :members: currentStep, step, minimizeEnergy, getPositionsJax, setPositionsJax, getVelocitiesJax, setVelocitiesJax, getForcesJax
   :undoc-members:
   :special-members: __init__
```

## Multi-System & Ragged Batch Simulation

`ProlixBatchSimulation` provides persistent GPU ensemble stepping (`step`) across homogeneous, congeneric, and cross-shape ragged (`cubic` / `rhombic_dodecahedron` / `truncated_octahedron`) batches, per-replica state & zero-copy JAX/DLPack management (`batch_size`, `n_replicas`, `bundles`, `setPositions`, `setVelocities`, `setVelocitiesToTemperature`, `getStates`, `getPositionsJax`, `setPositionsJax`, `getVelocitiesJax`, `setVelocitiesJax`, `getForcesJax`), batch energy minimization & checkpoint/state serialization (`minimizeEnergy`, `saveStates`, `loadStates`, `createCheckpoint`, `loadCheckpoint`, `saveCheckpoint`), and zero-recompilation alchemical / force-field differentiation (`setParameter`, `getParameters`, `value_and_grad_positions`, `compute_parameter_derivatives`).

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
