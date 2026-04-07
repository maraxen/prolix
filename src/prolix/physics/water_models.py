"""Water model registry and parameter sets.

All constants are in AMBER reduced units:
- Energy: kcal/mol
- Distance: Å
- Charge: e
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import jax.numpy as jnp


class WaterModelType(Enum):
    """Supported 3-site water models."""
    TIP3P = "tip3p"
    OPC3 = "opc3"


class WaterModelParams(NamedTuple):
    """Constant parameters for a 3-site water model."""
    name: str

    # Partial charges (e)
    charge_O: float
    charge_H: float

    # Lennard-Jones parameters (Å, kcal/mol)
    # AMBER water models only have LJ on the oxygen site.
    sigma_O: float
    epsilon_O: float

    # Equilibrium geometry (Å, radians)
    r_OH: float
    r_HH: float
    theta_HOH: float

    # Harmonic force constants (kcal/mol/Å^2, kcal/mol/rad^2)
    # Used for initial minimization where SETTLE is not yet active.
    k_bond: float
    k_angle: float

    @property
    def water_radius(self) -> float:
        """Clash radius for solvation pruning (Å).
        Computed as Rmin/2 = sigma * 2^(1/6) / 2 = sigma * 0.56123.
        """
        return self.sigma_O * 0.56123102415
    

# Registry of parameter sets (extracted from OpenMM amber14 XMLs)
# Note: k_bond is stored as "k" where E = 0.5 * k * (r-r0)^2
WATER_MODEL_REGISTRY: dict[WaterModelType, WaterModelParams] = {
    WaterModelType.TIP3P: WaterModelParams(
        name="TIP3P",
        charge_O=-0.834,
        charge_H=0.417,
        sigma_O=3.15075,
        epsilon_O=0.1521,
        r_OH=0.9572,
        r_HH=1.5139,
        theta_HOH=104.52 * jnp.pi / 180.0,
        k_bond=1106.1,  # 462750.4 kJ/mol/nm^2
        k_angle=200.0,   # OpenMM k=836.8 kJ/mol/rad^2 -> ~200 kcal/mol/rad^2
    ),
    WaterModelType.OPC3: WaterModelParams(
        name="OPC3",
        charge_O=-0.89517,
        charge_H=0.447585,
        sigma_O=3.17427,
        epsilon_O=0.163406,
        r_OH=0.97888,
        r_HH=1.5533,
        theta_HOH=109.47 * jnp.pi / 180.0,
        k_bond=1200.8,  # 502416.0 kJ/mol/nm^2
        k_angle=150.1,   # 628.02 kJ/mol/rad^2
    ),
}


def get_water_params(model_type: WaterModelType | str) -> WaterModelParams:
    """Retrieve water model parameters."""
    if isinstance(model_type, str):
        model_type = WaterModelType(model_type.lower())
    return WATER_MODEL_REGISTRY[model_type]
