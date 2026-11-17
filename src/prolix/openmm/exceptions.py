"""Exceptions for the Prolix OpenMM backend."""

from __future__ import annotations

from typing import Any


class UnsupportedOpenMMForceError(Exception):
    """Raised when an OpenMM force or feature is encountered that Prolix does not support."""

    def __init__(self, force_name: str, reason: str = "") -> None:
        self.force_name = force_name
        self.reason = reason
        msg = f"Unsupported OpenMM force: '{force_name}'."
        if reason:
            msg += f" {reason}"
        super().__init__(msg)


# Exhaustive catalogue of unsupported OpenMM force classes and explanatory reasons.
UNSUPPORTED_OPENMM_FORCES: dict[str, str] = {
    "AmoebaGeneralizedKirkwoodForce": (
        "AMOEBA polarizable multipole implicit solvent is not implemented in Prolix."
    ),
    "AmoebaMultipoleForce": (
        "AMOEBA permanent and induced multipole electrostatics is not implemented in Prolix."
    ),
    "AmoebaTorsionTorsionForce": (
        "AMOEBA coupled torsion-torsion energy grids are not implemented in Prolix."
    ),
    "AmoebaVdwForce": (
        "AMOEBA buffered 14-7 van der Waals potential is not implemented in Prolix."
    ),
    "AmoebaWcaDispersionForce": (
        "AMOEBA Weeks-Chandler-Andersen dispersion is not implemented in Prolix."
    ),
    "DrudeForce": (
        "Drude polarizable oscillator model is not implemented in Prolix."
    ),
    "HippoNonbondedForce": (
        "HIPPO polarizable force field is not implemented in Prolix."
    ),
    "RPMDMonteCarloBarostat": (
        "Ring polymer molecular dynamics (RPMD) barostat is not implemented in Prolix."
    ),
    "CustomCVForce": (
        "Custom collective variables force is not implemented in Prolix OpenMM converter."
    ),
    "CustomCentroidBondForce": (
        "Centroid-based bonded forces are not implemented in Prolix."
    ),
    "CustomHbondForce": (
        "Custom hydrogen bonding potentials are not implemented in Prolix."
    ),
    "CustomManyParticleForce": (
        "N-body (>4) interactions are not implemented in Prolix."
    ),
    "CustomVolumeForce": (
        "Custom volume-dependent forces are not implemented in Prolix."
    ),
    "MonteCarloAnisotropicBarostat": (
        "Anisotropic box volume scaling barostat is not implemented in Prolix "
        "(use MonteCarloBarostat for isotropic scaling)."
    ),
    "MonteCarloFlexibleBarostat": (
        "Fully flexible triclinic cell barostat is not implemented in Prolix."
    ),
    "MonteCarloMembraneBarostat": (
        "Membrane area/volume scaling barostat is not implemented in Prolix."
    ),
    "GayBerneForce": (
        "Anisotropic Gay-Berne ellipsoid potential is not implemented in Prolix."
    ),
    "ATMForce": (
        "Asymmetric thermodynamic integration force (ATM) is not implemented in Prolix."
    ),
    "ConstantPotentialForce": (
        "Constant potential electrode method is not implemented in Prolix."
    ),
    "LCPOForce": (
        "Linear combinations of pairwise overlaps (LCPO) implicit solvent is not implemented in Prolix."
    ),
    "OrientationRestraintForce": (
        "Dipolar coupling orientation restraints are not implemented in Prolix."
    ),
    "PythonForce": (
        "Arbitrary Python callback forces are not supported in Prolix JAX execution graph."
    ),
    "RGForce": (
        "Radius of gyration restraints are not implemented in Prolix."
    ),
    "RMSDForce": (
        "Coordinate RMSD restraints are not implemented in Prolix."
    ),
}


def check_unsupported_force(force: Any) -> None:
    """Check if an OpenMM force is known to be unsupported, raising UnsupportedOpenMMForceError if so.

    Args:
        force: An OpenMM Force instance.

    Raises:
        UnsupportedOpenMMForceError: If the force type is in the unsupported catalogue.
    """
    name = type(force).__name__
    if name in UNSUPPORTED_OPENMM_FORCES:
        raise UnsupportedOpenMMForceError(name, UNSUPPORTED_OPENMM_FORCES[name])
