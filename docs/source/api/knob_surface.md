# `prolix.knob_surface` & `redsox` Governance

Zero-dependency composite configuration schema (`KnobSurface`) and static AST governance verifiers (`prolix.governance.redsox_adapter`).

```{eval-rst}
.. currentmodule:: prolix.knob_surface

.. autosummary::
   :nosignatures:

   KnobSurface
   ThermostatControls
   BarostatControls
   NonbondedControls
   ImplicitSolventControls
   ConstraintControls
   CosolventControls
   SoftcoreRestraintControls
   CollectiveVariableBiasControls
   PallasControls
```

---

## Composite Knob Surface

```{eval-rst}
.. autoclass:: prolix.knob_surface.KnobSurface
   :members:
   :undoc-members:
```

---

## Sub-Surface Control Dataclasses

```{eval-rst}
.. autoclass:: prolix.knob_surface.ThermostatControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.BarostatControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.NonbondedControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.ImplicitSolventControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.ConstraintControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.CosolventControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.SoftcoreRestraintControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.CollectiveVariableBiasControls
   :members:
   :undoc-members:

.. autoclass:: prolix.knob_surface.PallasControls
   :members:
   :undoc-members:
```

---

## Static Governance (`prolix.governance.redsox_adapter`)

```{eval-rst}
.. py:module:: prolix.governance.redsox_adapter

.. autoclass:: prolix.governance.redsox_adapter.SupersetResult
   :members:
   :undoc-members:

.. autoclass:: prolix.governance.redsox_adapter.ReachabilityResult
   :members:
   :undoc-members:

.. autofunction:: prolix.governance.redsox_adapter.check_superset

.. autofunction:: prolix.governance.redsox_adapter.check_reachability

.. autofunction:: prolix.governance.redsox_adapter.check_prolix_reachability
```
