"""Structural protocols for energy functions and integrator steps.

EnergyFn and IntegratorFn are @runtime_checkable so that isinstance() works
for structural type checking at system boundaries. These protocols carry no
implementation — they only declare expected call signatures.

Key design principles:
- Both protocols are callable-only; isinstance() checks work at runtime
- EnergyFn: (MolecularBundle) -> Float scalar, no closure capture, jax.export-safe
- IntegratorFn: (IntegratorState) -> IntegratorState, flat signature
- Enables runtime dispatch at system boundaries without runtime type assertions
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float

from prolix.types.bundles import MolecularBundle
from prolix.types.integrators import IntegratorState


@runtime_checkable
class EnergyFn(Protocol):
    """(bundle: MolecularBundle) -> Float scalar.

    An energy function computes total potential energy from a molecular bundle.
    The function must:
    - Accept a MolecularBundle as sole argument
    - Return a scalar Float array (shape ())
    - Contain no closure capture of trainable parameters
    - Be jax.export-safe (pure functional, no side effects)
    """

    def __call__(self, bundle: MolecularBundle) -> Float[Array, ""]: ...


@runtime_checkable
class IntegratorFn(Protocol):
    """(state: IntegratorState) -> IntegratorState.

    An integrator step function advances the system by one timestep.
    The function must:
    - Accept an IntegratorState (or subclass) as sole argument
    - Return an IntegratorState with all fields updated
    - Be pure functional (no side effects)
    - Preserve state type (e.g., LangevinState -> LangevinState)
    """

    def __call__(self, state: IntegratorState) -> IntegratorState: ...
