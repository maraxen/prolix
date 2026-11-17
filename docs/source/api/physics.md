# `prolix.physics` — Potentials, Integrators & Bundles

Differentiable potential-energy factories, PyTree parameter containers, bucketed host-to-device molecular bundles, isothermal-isobaric (NPT) barostats, and cross-engine unit adapters.

```{eval-rst}
.. currentmodule:: prolix.physics

.. autosummary::
   :nosignatures:

   make_energy_fn_pure
   ~prolix.typing.DifferentiableParams
   make_integrator
   make_integrator_batched
   settle_csvr_npt
   NPTState
   instantaneous_pressure_akma
   virial_trace
   box_volume
   isotropic_box_scale
```

---

## Pure & Differentiable Energy Factories

```{eval-rst}
.. autofunction:: prolix.physics.make_energy_fn_pure

.. autofunction:: prolix.physics.system.make_energy_fn

.. autoclass:: prolix.typing.DifferentiableParams
   :members:
   :undoc-members:

.. autoclass:: prolix.typing.PhysicsSystem
   :members:
   :undoc-members:
```

---

## Bucketed Device Bundles (`prolix.types.bundles`)

```{eval-rst}
.. autoclass:: prolix.types.bundles.MolecularShapeSpec
   :members:

.. autoclass:: prolix.types.bundles.MolecularBundle
   :members:
   :undoc-members:
```

---

## Modular Integrators & NPT Barostats

```{eval-rst}
.. autofunction:: prolix.physics.make_integrator

.. autofunction:: prolix.physics.make_integrator_batched

.. autofunction:: prolix.physics.settle_csvr_npt

.. autoclass:: prolix.physics.NPTState
   :members:
   :undoc-members:

.. autofunction:: prolix.physics.instantaneous_pressure_akma

.. autofunction:: prolix.physics.virial_trace

.. autofunction:: prolix.physics.box_volume

.. autofunction:: prolix.physics.isotropic_box_scale
```

---

## kUPS Cross-Validation Unit Adapter (`prolix.physics.kups_adapter`)

```{eval-rst}
.. automodule:: prolix.physics.kups_adapter
   :members:
```
