"""Ion parameter sets for common water models.

All constants are in AMBER reduced units:
- Energy: kcal/mol
- Distance: Å
- Charge: e
"""

from __future__ import annotations

from typing import NamedTuple
from prolix.physics.water_models import WaterModelType


class IonParams(NamedTuple):
    """Lennard-Jones parameters for monovalent ion (Ag, Li, Na, K, Rb, Cs, F, Cl, Br, I)."""
    name: str
    charge: float      # e
    sigma: float       # Å
    epsilon: float     # kcal/mol
    mass: float        # amu


# Registry keyed by (WaterModelType, ion_name)
ION_REGISTRY: dict[tuple[WaterModelType, str], IonParams] = {
    # Joung-Cheatham ions for TIP3P
    (WaterModelType.TIP3P, "NA"): IonParams("Na+", 1.0, 2.43928, 0.087439, 22.9897),
    (WaterModelType.TIP3P, "CL"): IonParams("Cl-", -1.0, 4.47766, 0.035591, 35.45),
    (WaterModelType.TIP3P, "LI"): IonParams("Li+", 1.0, 1.82634, 0.027989, 6.941),
    (WaterModelType.TIP3P, "K"): IonParams("K+", 1.0, 3.03796, 0.19368, 39.0983),
    
    # Li/Merz ions for OPC3 (12-6 set optimized for OPC3)
    (WaterModelType.OPC3, "NA"): IonParams("Na+", 1.0, 2.61746, 0.030122, 22.9897),
    (WaterModelType.OPC3, "CL"): IonParams("Cl-", -1.0, 4.10882, 0.642267, 35.45),
    (WaterModelType.OPC3, "LI"): IonParams("Li+", 1.0, 2.26288, 0.003256, 6.941),
    (WaterModelType.OPC3, "K"): IonParams("K+", 1.0, 3.03440, 0.140218, 39.0983),
}


def get_ion_params(model_type: WaterModelType | str, ion_name: str) -> IonParams:
    """Retrieve ion parameters for a specific water model."""
    if isinstance(model_type, str):
        model_type = WaterModelType(model_type.lower())
    return ION_REGISTRY[(model_type, ion_name.upper())]
