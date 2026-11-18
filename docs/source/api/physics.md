# `prolix.physics` — Potentials, Integrators & Bundles

Differentiable potential-energy factories, PyTree parameter containers, `P=6` dispersion Ewald (`LJ-PME`), clean-room $C^2$-continuous Prolate Spheroidal Ewald splittings (`P=1` electrostatics and `P=6` dispersion), compensated float-float (`fltflt`) GPU accumulators, bucketed host-to-device molecular bundles, isothermal-isobaric (NPT) barostats, and cross-engine unit adapters.

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
   ~lj_pme.lj_pme_total_dispersion_energy
   ~prolate_splitting.prolate_spme_electrostatics
   ~prolate_splitting.prolate_dispersion_pme
   ~fltflt_pallas.FltFlt
```

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

## `P=6` Dispersion Particle-Mesh Ewald (`prolix.physics.lj_pme`)

Pure-JAX differentiable `1/r^6` dispersion Ewald / SPME matching OpenMM `NonbondedForce.LJPME` across Lorentz–Berthelot and geometric combining rules.

```{eval-rst}
.. py:module:: prolix.physics.lj_pme

.. autofunction:: prolix.physics.lj_pme.dispersion_charges

.. autofunction:: prolix.physics.lj_pme.resolve_ljpme_parameters

.. autofunction:: prolix.physics.lj_pme.spread_dispersion_charges

.. autofunction:: prolix.physics.lj_pme.lj_pme_influence_function

.. autofunction:: prolix.physics.lj_pme.lj_pme_reciprocal_energy

.. autofunction:: prolix.physics.lj_pme.lj_pme_self_energy

.. autofunction:: prolix.physics.lj_pme.lj_pme_exclusion_energy

.. autofunction:: prolix.physics.lj_pme.lj_pme_direct_pair_energy

.. autofunction:: prolix.physics.lj_pme.lj_pme_direct_energy_neighbor_list

.. autofunction:: prolix.physics.lj_pme.lj_pme_total_dispersion_energy

.. autofunction:: prolix.physics.lj_pme.ewald_p6_fourier_kernel_value

.. autofunction:: prolix.physics.lj_pme.ewald_p6_fourier_kernel_grad

.. autofunction:: prolix.physics.lj_pme.ewald_p6_short_range_value

.. autofunction:: prolix.physics.lj_pme.ewald_p6_long_range_value
```

## $C^2$ Prolate Spheroidal Ewald Splittings (`prolix.physics.prolate_splitting`)

Clean-room $C^2$-continuous compact polynomial splittings $r^{-p} = S_p(r^2; r_c) + L_p(r^2; r_c)$ (`arXiv:2505.09727`) for `P=1` electrostatics ($c_1 = 10.75$) and `P=6` dispersion ($c_6 = 12.764031957789484$) with $S_p(r_c^2) = S_p'(r_c^2) = S_p''(r_c^2) = 0$, orthorhombic and reduced-triclinic SPME solvers, analytical `@jax.custom_vjp` derivatives, and portable StableHLO export.

```{eval-rst}
.. py:module:: prolix.physics.prolate_splitting

.. autofunction:: prolix.physics.prolate_splitting.prolate_p1_short_range_value_and_d_dr2

.. autofunction:: prolix.physics.prolate_splitting.prolate_p1_long_range_value_and_d_dr2

.. autofunction:: prolix.physics.prolate_splitting.prolate_p6_short_range_value_and_d_dr2

.. autofunction:: prolix.physics.prolate_splitting.prolate_p6_long_range_value_and_d_dr2

.. autofunction:: prolix.physics.prolate_splitting.prolate_p1_fourier_kernel_and_d_dk2

.. autofunction:: prolix.physics.prolate_splitting.prolate_p6_fourier_kernel_and_d_dk2

.. autofunction:: prolix.physics.prolate_splitting.prolate_p1_self_energy_coeff

.. autofunction:: prolix.physics.prolate_splitting.prolate_p6_self_energy_coeff

.. autofunction:: prolix.physics.prolate_splitting.prolate_p1_dc_coeff

.. autofunction:: prolix.physics.prolate_splitting.prolate_spme_electrostatics

.. autofunction:: prolix.physics.prolate_splitting.prolate_dispersion_pme
```

## Compensated Float-Float Arithmetic (`prolix.physics.fltflt_pallas`)

```{eval-rst}
.. automodule:: prolix.physics.fltflt_pallas
   :members:
```

## Bucketed Device Bundles (`prolix.types.bundles`)

```{eval-rst}
.. autoclass:: prolix.types.bundles.MolecularShapeSpec
   :members:

.. autoclass:: prolix.types.bundles.MolecularBundle
   :members:
   :undoc-members:
```

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

## kUPS Cross-Validation Unit Adapter (`prolix.physics.kups_adapter`)

```{eval-rst}
.. automodule:: prolix.physics.kups_adapter
   :members:
```
