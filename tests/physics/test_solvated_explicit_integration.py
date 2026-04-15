"""Phase 4: solvated protein → merged topology → padded system → explicit energy (finite).

End-to-end check distinct from NL vs dense parity on a dry protein (see
``test_protein_nl_explicit_parity``): here we exercise ``solvate_protein``,
``pad_solvated_system``, and ``single_padded_energy_nl_cvjp`` on a **merged**
solvated topology.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from proxide import CoordFormat, OutputSpec, parse_structure

from prolix.batched_energy import single_padded_energy, single_padded_energy_nl_cvjp
from prolix.padding import pad_solvated_system, precompute_dense_exclusions
from prolix.physics import neighbor_list as nl
from prolix.physics import pbc
from prolix.physics.solvation import solvate_protein
from prolix.physics.water_models import WaterModelType

jax.config.update("jax_enable_x64", True)

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


def _load_1vii():
    pdb_path = DATA_DIR / "1VII.pdb"
    if not pdb_path.is_file():
        pytest.skip(f"Missing test PDB: {pdb_path}")
    spec = OutputSpec()
    spec.parameterize_md = True
    spec.force_field = str(FF_PATH)
    spec.coord_format = CoordFormat.Full
    return parse_structure(str(pdb_path), spec)


@pytest.mark.slow
def test_solvated_merged_padded_explicit_energy_finite():
    """Small orthorhombic box to bound water count; NL energy must be finite."""
    protein = _load_1vii()
    merged = solvate_protein(
        protein,
        padding=8.0,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array([32.0, 32.0, 32.0]),
    )
    need = nl.max_exclusion_slots_needed(merged.exclusion_spec)
    assert need <= 32, (
        f"Exclusion slots needed {need} > 32 — increase map cap or shrink system for this test."
    )

    padded = pad_solvated_system(merged)
    padded = precompute_dense_exclusions(padded)

    box = jnp.asarray(padded.box_size, dtype=jnp.float64)
    displacement_fn, _ = pbc.create_periodic_space(box)
    r = jnp.asarray(padded.positions, dtype=jnp.float64)
    n_real = int(padded.n_real_atoms)

    neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, box, 10.0)
    nbr0 = neighbor_fn.allocate(r)
    nbr = neighbor_fn.update(r, nbr0)

    e = single_padded_energy_nl_cvjp(
        padded,
        nbr.idx,
        displacement_fn,
        implicit_solvent=False,
    )
    assert jnp.isfinite(e), float(e)
    assert float(jnp.abs(e)) < 1.0e7


@pytest.mark.slow
def test_solvated_explicit_dense_matches_nl_energy():
    """Explicit PME: dense vs neighbor-list total energy at the same geometry (solvated 1VII)."""
    protein = _load_1vii()
    merged = solvate_protein(
        protein,
        padding=8.0,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array([32.0, 32.0, 32.0]),
    )
    need = nl.max_exclusion_slots_needed(merged.exclusion_spec)
    assert need <= 32, (
        f"Exclusion slots needed {need} > 32 — increase map cap or shrink system for this test."
    )

    padded = pad_solvated_system(merged)
    padded = precompute_dense_exclusions(padded)

    box = jnp.asarray(padded.box_size, dtype=jnp.float64)
    displacement_fn, _ = pbc.create_periodic_space(box)
    r = jnp.asarray(padded.positions, dtype=jnp.float64)
    n_real = int(padded.n_real_atoms)

    neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, box, 10.0)
    nbr0 = neighbor_fn.allocate(r)
    nbr = neighbor_fn.update(r, nbr0)

    e_dense = float(
        single_padded_energy(padded, displacement_fn, implicit_solvent=False)
    )
    e_nl = float(
        single_padded_energy_nl_cvjp(
            padded,
            nbr.idx,
            displacement_fn,
            implicit_solvent=False,
        )
    )
    assert np.isfinite(e_dense) and np.isfinite(e_nl), (e_dense, e_nl)
    # Align with protein NL parity tolerances (mesh + NL sparsity).
    assert np.isclose(e_dense, e_nl, rtol=8e-3, atol=5.0), (
        f"dense={e_dense} nl={e_nl} kcal/mol"
    )


def test_water_model_registry_has_tip3p_opc3():
    """OPC3/TIP3P parameters are registered; pre-equilibrated OPC3 box may be absent."""
    from prolix.physics.water_models import get_water_params

    tip = get_water_params(WaterModelType.TIP3P)
    opc = get_water_params(WaterModelType.OPC3)
    assert tip.charge_O < 0 and opc.charge_O < 0
    assert tip.sigma_O > 0 and opc.sigma_O > 0
