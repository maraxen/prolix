"""Prolix OpenMM Backend.

Drop-in replacement for OpenMM execution utilizing Prolix's JAX backend and Pallas kernels.
"""

from __future__ import annotations

from prolix.openmm import config, context, converter, exceptions, simulation, units
from prolix.openmm.config import (
    ProlixOpenMMConfig,
    ProlixPlatform,
    parse_platform_properties,
)
from prolix.openmm.context import (
    ProlixContext,
    ProlixState,
)
from prolix.openmm.converter import (
    ConvertedOpenMMSystem,
    from_openmm_system,
)
from prolix.openmm.exceptions import (
    UnsupportedOpenMMForceError,
)
from prolix.openmm.simulation import (
    ProlixSimulation,
)
from prolix.openmm.units import (
    from_akma,
    to_akma,
)

# OpenMM Drop-in Aliases
Context = ProlixContext
Simulation = ProlixSimulation
Platform = ProlixPlatform
State = ProlixState

__all__ = [
    "Context",
    "ConvertedOpenMMSystem",
    "Platform",
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
