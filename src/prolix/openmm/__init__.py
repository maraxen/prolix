"""Prolix OpenMM Backend.

Drop-in replacement for OpenMM execution utilizing Prolix's JAX backend and Pallas kernels.
"""

from __future__ import annotations

from prolix.openmm import config, units
from prolix.openmm.config import (
    ProlixOpenMMConfig,
    ProlixPlatform,
    parse_platform_properties,
)
from prolix.openmm.units import (
    from_akma,
    to_akma,
)

# OpenMM Platform alias
Platform = ProlixPlatform

__all__ = [
    "Platform",
    "ProlixOpenMMConfig",
    "ProlixPlatform",
    "config",
    "from_akma",
    "parse_platform_properties",
    "to_akma",
    "units",
]
