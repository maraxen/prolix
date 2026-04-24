"""Explicit-solvent electrostatic models (OpenMM-aligned reference: user guide §19.6).

Default production path remains PME via ``physics.system.make_energy_fn``. Optional
methods are **opt-in** and intended for comparison or legacy workflows.
"""

from __future__ import annotations

from enum import StrEnum


class ElectrostaticMethod(StrEnum):
  """Nonbonded electrostatic treatment for explicit periodic systems."""

  PME = "pme"
  """Particle mesh Ewald (reciprocal + erfc-damped direct space). Default."""

  REACTION_FIELD = "reaction_field"
  """OpenMM *CutoffPeriodic* reaction-field approximation (no reciprocal sum).

  Uses the pairwise form from the OpenMM user guide (Coulomb with cutoff), with
  coefficients ``k_rf`` and ``c_rf`` derived from the cutoff and solvent
  dielectric. See :func:`openmm_reaction_field_coefficients`.
  """

  DAMPED_SHIFTED_FORCE = "dsf"
  """Shifted erfc-damped Coulomb: direct space only, energy vanishes at cutoff.

  ``V_ij(r) = C * q_i q_j * (erfc(alpha*r)/r - erfc(alpha*r_c)/r_c)`` for ``r <= r_c``,
  zero beyond. Continuity of the *energy* at ``r_c`` is enforced; forces are not
  guaranteed to vanish at ``r_c`` (use RF for OpenMM CutoffPeriodic parity).
  """

  EFA = "efa"
  """Euclidean Fast Attention-style Coulomb via Random Fourier Features (RFF).

  Replaces full PME (direct + reciprocal) with an O(N*D) global kernel
  approximation of erfc(a*r)/r. Experimental; valid for large-box or non-periodic
  systems. Requires soft_core_lambda=1.0 (no alchemical perturbation). D=512 default.
  See references/notes/rff_erfc_derivation.md for derivation.
  """


def openmm_reaction_field_coefficients(
  cutoff: float,
  solvent_dielectric: float = 78.3,
) -> tuple[float, float]:
  """Coefficients ``k_rf``, ``c_rf`` for OpenMM reaction-field Coulomb (user guide §19.6.3).

  Args:
    cutoff: Nonbonded cutoff distance ``r_cutoff`` (Å).
    solvent_dielectric: ``ε_solvent`` in the OpenMM formulas (default ~water).

  Returns:
    ``(k_rf, c_rf)`` such that the pair potential per (q1,q2) includes
    ``(1/r + k_rf r² - c_rf)`` scaled by ``k_elec * q1 * q2``.
  """
  rc = float(cutoff)
  if rc <= 0.0:
    raise ValueError("cutoff must be positive")
  eps = float(solvent_dielectric)
  inv_rc = 1.0 / rc
  inv_rc3 = inv_rc**3
  k_rf = inv_rc3 * ((eps - 1.0) / (2.0 * eps + 1.0))
  c_rf = inv_rc * (3.0 * eps / (2.0 * eps + 1.0))
  return k_rf, c_rf
