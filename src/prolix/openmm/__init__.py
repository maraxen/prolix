"""Prolix OpenMM Backend.

Drop-in replacement for OpenMM execution utilizing Prolix's JAX backend and Pallas kernels.
Uses PEP 562 lazy submodule loading for JAX-dependent runtime modules so leaf configuration
(`prolix.openmm.config`) and converter imports (`prolix.openmm.converter`) remain cycle-free
in any import order.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from prolix.openmm import config, exceptions
from prolix.openmm.config import (
    ProlixOpenMMConfig,
    ProlixPlatform,
    parse_platform_properties,
)
from prolix.openmm.exceptions import (
    UnsupportedOpenMMForceError,
)

if TYPE_CHECKING:
    from prolix.openmm import context, converter, simulation, units
    from prolix.openmm.context import ProlixContext, ProlixState
    from prolix.openmm.converter import ConvertedOpenMMSystem, from_openmm_system
    from prolix.openmm.simulation import ProlixBatchSimulation, ProlixSimulation
    from prolix.openmm.units import from_akma, to_akma

# OpenMM Drop-in Aliases (eagerly bound for leaf Platform; lazily resolved for Context/Simulation/State)
Platform = ProlixPlatform

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {"context", "converter", "simulation", "units"}
)

_LAZY_ATTR_TO_SUBMODULE: dict[str, tuple[str, str]] = {
    "ProlixContext": ("context", "ProlixContext"),
    "Context": ("context", "ProlixContext"),
    "ProlixState": ("context", "ProlixState"),
    "State": ("context", "ProlixState"),
    "ConvertedOpenMMSystem": ("converter", "ConvertedOpenMMSystem"),
    "from_openmm_system": ("converter", "from_openmm_system"),
    "ProlixSimulation": ("simulation", "ProlixSimulation"),
    "Simulation": ("simulation", "ProlixSimulation"),
    "ProlixBatchSimulation": ("simulation", "ProlixBatchSimulation"),
    "BatchSimulation": ("simulation", "ProlixBatchSimulation"),
    "from_akma": ("units", "from_akma"),
    "to_akma": ("units", "to_akma"),
}

__all__ = [
    "BatchSimulation",
    "Context",
    "ConvertedOpenMMSystem",
    "Platform",
    "ProlixBatchSimulation",
    "ProlixContext",
    "ProlixOpenMMConfig",
    "ProlixPlatform",
    "ProlixSimulation",
    "ProlixState",
    "Simulation",
    "State",
    "UnsupportedOpenMMForceError",
    "config",
    "context",
    "converter",
    "exceptions",
    "from_akma",
    "from_openmm_system",
    "parse_platform_properties",
    "simulation",
    "to_akma",
    "units",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loader for JAX/OpenMM runtime submodules and classes."""
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"prolix.openmm.{name}")
        globals()[name] = mod
        return mod
    target = _LAZY_ATTR_TO_SUBMODULE.get(name)
    if target is not None:
        submod_name, attr_name = target
        mod = importlib.import_module(f"prolix.openmm.{submod_name}")
        globals()[submod_name] = mod
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
