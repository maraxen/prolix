"""Explicit-solvent PME regression knobs shared by parity tests and benchmarks.

Single source of truth for ``REGRESSION_EXPLICIT_PME``. Tests import this dict;
``scripts/export_regression_pme.py`` embeds it in ``openmm_comparison_protocol.md``.
"""

from __future__ import annotations

REGRESSION_EXPLICIT_PME: dict[str, object] = {
  "pme_alpha_per_angstrom": 0.34,
  "pme_grid_points": 32,  # cubic grid nx=ny=nz
  "cutoff_angstrom": 9.0,  # two-particle tests; protein tests may use 10-12
  "use_dispersion_correction": False,
  "openmm_platform": "Reference",  # deterministic CPU parity
}
