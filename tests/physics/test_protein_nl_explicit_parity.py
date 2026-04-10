"""Explicit PBC + PME: neighbor-list vs dense ``make_energy_fn`` on real protein topologies.

Dense direct-space PME sums all minimum-image pairs; the neighbor list sums only pairs
within ``cutoff_distance``. Agreement therefore requires a cutoff large enough that
``erfc(alpha * r)`` makes omitted pairs negligible (see module docstrings in each test).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from proxide import CoordFormat, OutputSpec, parse_structure

from prolix.physics import neighbor_list as nl
from prolix.physics import pbc, system

jax.config.update("jax_enable_x64", True)

# Repo layout: ``prolix/data/pdb`` and sibling ``proxide/`` (see ``tests/physics/conftest.py``).
_DATA_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _DATA_ROOT / "data" / "pdb"
FF_PATH = (
    Path(__file__).resolve().parents[3]
    / "proxide"
    / "src"
    / "proxide"
    / "assets"
    / "protein.ff19SB.xml"
)


def _load_protein(pdb_name: str):
    pdb_path = DATA_DIR / pdb_name
    if not pdb_path.is_file():
        pytest.skip(f"Missing test PDB: {pdb_path}")
    spec = OutputSpec()
    spec.parameterize_md = True
    spec.force_field = str(FF_PATH)
    spec.coord_format = CoordFormat.Full
    return parse_structure(str(pdb_path), spec)


def _center_coords_in_orthorhombic_box(
    coords: jnp.ndarray, pad: float = 12.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return positions shifted into ``[0, box]^3`` and box edges (Å)."""
    lo = jnp.min(coords, axis=0)
    hi = jnp.max(coords, axis=0)
    span = hi - lo
    box = span + 2.0 * pad
    box = jnp.maximum(box, jnp.array([24.0, 24.0, 24.0], dtype=jnp.float64))
    shift = box * 0.5 - (lo + hi) * 0.5
    r = coords + shift
    return r, box


def _nl_dense_explicit_parity(
    protein,
    *,
    cutoff: float,
    pme_grid_points: int = 64,
    pme_alpha: float = 0.34,
    energy_rtol: float = 5e-3,
    energy_atol: float = 3.0,
    grad_rmse_max: float | None = 0.15,
):
    excl = nl.ExclusionSpec.from_protein(protein)
    need = nl.max_exclusion_slots_needed(excl)
    assert need <= 32, (
        f"Topology needs {need} exclusion slots per atom; map_exclusions cap is 32 — "
        "increase max_exclusions or use a smaller system."
    )

    coords = jnp.asarray(protein.coordinates, dtype=jnp.float64)
    r, box = _center_coords_in_orthorhombic_box(coords, pad=12.0)

    displacement_fn, _ = pbc.create_periodic_space(box)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        protein,
        exclusion_spec=excl,
        box=box,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=pme_grid_points,
        pme_alpha=pme_alpha,
        cutoff_distance=cutoff,
        strict_parameterization=False,
    )

    neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, box, cutoff)
    nbr0 = neighbor_fn.allocate(r)
    nbr = neighbor_fn.update(r, nbr0)

    e_dense = float(energy_fn(r))
    e_nl = float(energy_fn(r, neighbor=nbr))
    assert np.isclose(e_dense, e_nl, rtol=energy_rtol, atol=energy_atol), (
        f"NL vs dense mismatch: dense={e_dense:.6f} nl={e_nl:.6f} Δ={e_dense - e_nl:.6f} "
        f"(cutoff={cutoff} Å, N={r.shape[0]})"
    )

    if grad_rmse_max is not None:

        def e_nl_fresh(pos):
            n_init = neighbor_fn.allocate(pos)
            n_up = neighbor_fn.update(pos, n_init)
            return energy_fn(pos, neighbor=n_up)

        g_dense = jax.grad(energy_fn)(r)
        g_nl = jax.grad(e_nl_fresh)(r)
        rmse = float(jnp.sqrt(jnp.mean((g_dense - g_nl) ** 2)))
        assert rmse < grad_rmse_max, (
            f"Gradient RMSE {rmse:.6f} kcal/mol/Å exceeds {grad_rmse_max} "
            f"(cutoff={cutoff} Å, N={r.shape[0]})"
        )


def test_max_exclusion_slots_helpers_consistent():
    """Sanity: ``max_exclusion_slots_needed`` fits default padding cap."""
    protein = _load_protein("1VII.pdb")
    excl = nl.ExclusionSpec.from_protein(protein)
    need = nl.max_exclusion_slots_needed(excl)
    assert need >= 1
    assert need <= 32


@pytest.mark.slow
class TestExplicitProteinNLDenseSlow:
    """Protein NL vs dense (O(N²) reference): use ``pytest -m \"not slow\"`` to skip."""

    def test_1vii_nl_matches_dense_explicit_pme(self):
        protein = _load_protein("1VII.pdb")
        _nl_dense_explicit_parity(
            protein,
            cutoff=22.0,
            energy_rtol=5e-3,
            energy_atol=2.5,
            grad_rmse_max=0.12,
        )

    def test_1crn_nl_matches_dense_explicit_pme(self):
        protein = _load_protein("1CRN.pdb")
        _nl_dense_explicit_parity(
            protein,
            cutoff=22.0,
            energy_rtol=8e-3,
            energy_atol=5.0,
            grad_rmse_max=0.2,
        )


@pytest.mark.slow
class TestSolvatedExplicitPeriodicSlow:
    """Pre-solvated PDB: many atoms; dense reference is expensive."""

    def test_solvated_pdb_nl_matches_dense_explicit_pme(self):
        pdb_path = DATA_DIR / "1UAO_solvated_tip3p.pdb"
        if not pdb_path.is_file():
            pytest.skip("1UAO_solvated_tip3p.pdb not found")
        spec = OutputSpec()
        spec.parameterize_md = True
        spec.force_field = str(FF_PATH)
        spec.coord_format = CoordFormat.Full
        protein = parse_structure(str(pdb_path), spec)
        excl = nl.ExclusionSpec.from_protein(protein)
        need = nl.max_exclusion_slots_needed(excl)
        assert need <= 32, f"exclusion slots needed {need} > 32"

        coords = jnp.asarray(protein.coordinates, dtype=jnp.float64)
        r, box = _center_coords_in_orthorhombic_box(coords, pad=10.0)
        # CRYST1 in file ~29 Å — keep explicit box from pdb if parse_structure sets it?
        displacement_fn, _ = pbc.create_periodic_space(box)
        energy_fn = system.make_energy_fn(
            displacement_fn,
            protein,
            exclusion_spec=excl,
            box=box,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=32,
            pme_alpha=0.34,
            cutoff_distance=12.0,
            strict_parameterization=False,
        )
        cutoff = 12.0
        neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, box, cutoff)
        nbr0 = neighbor_fn.allocate(r)
        nbr = neighbor_fn.update(r, nbr0)
        e_dense = float(energy_fn(r))
        e_nl = float(energy_fn(r, neighbor=nbr))
        assert np.isclose(e_dense, e_nl, rtol=0.02, atol=10.0), (
            f"solvated NL vs dense: {e_dense} vs {e_nl}"
        )
