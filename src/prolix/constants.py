"""Physical constants for molecular dynamics simulation."""

# Element masses in Daltons (atomic mass units).
# Used as fallback when force-field masses are unavailable.
ELEMENT_MASS: dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06,
    "P": 30.974, "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.904,
    "Fe": 55.845, "Zn": 65.38, "Ca": 40.078, "Mg": 24.305, "Na": 22.990,
    "K": 39.098, "Se": 78.971, "Mn": 54.938, "Cu": 63.546, "Co": 58.933,
}

# Default mass for unknown elements (carbon).
DEFAULT_MASS: float = 12.011


# Element → atomic number mapping for element-based identification.
# Used for constraint detection (H identification) instead of mass-based
# checks, which break under HMR (Hydrogen Mass Repartitioning).
ELEMENT_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16,
    "P": 15, "F": 9, "Cl": 17, "Br": 35, "I": 53,
    "Fe": 26, "Zn": 30, "Ca": 20, "Mg": 12, "Na": 11,
    "K": 19, "Se": 34, "Mn": 25, "Cu": 29, "Co": 27,
}


def atomic_numbers_from_elements(elements: list[str]) -> list[int]:
    """Derive per-atom atomic numbers from element symbols.

    Args:
        elements: List of element symbols (e.g. ["C", "N", "O", "H"]).

    Returns:
        List of atomic numbers, one per element. Unknown elements → 0.
    """
    return [
        ELEMENT_ATOMIC_NUMBER.get(
            e, ELEMENT_ATOMIC_NUMBER.get(e.capitalize(), 0)
        )
        for e in elements
    ]


def masses_from_elements(elements: list[str]) -> list[float]:
    """Derive per-atom masses from element symbols.

    Args:
        elements: List of element symbols (e.g. ["C", "N", "O", "H"]).

    Returns:
        List of masses in Daltons, one per element.
    """
    return [
        ELEMENT_MASS.get(e, ELEMENT_MASS.get(e.capitalize(), DEFAULT_MASS))
        for e in elements
    ]
