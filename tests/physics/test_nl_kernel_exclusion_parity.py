"""Regression test for debt 790: dense vs NL Coulomb parity under real PME exclusions.

Root cause (confirmed 2026-07-18, see .praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md):
NOT a bug in ``chunked_coulomb_energy_nl``/``chunked_lj_energy_nl`` (optimization.py) --
those kernels are self-contained and correct, matching ``system.py``'s own already-validated
NL convention (``pair_scale - erf(alpha * r)`` computed internally per pair).

The actual bug is in ``single_padded_energy``'s wiring (batched_energy.py):
``_coulomb_energy_masked`` (the dense fallback) uses a DIFFERENT, incomplete convention
(``erfc(alpha * r) * scale`` -- zeroes excluded pairs entirely instead of adding the
compensating ``-erf(alpha * r)`` term), which is completed by a *separate*
``_pme_exclusion_correction_from_pairs`` call inside ``_pme_reciprocal_and_corrections``.
That correction is applied unconditionally regardless of which direct-space path was
used -- so when ``neighbor=`` (NL) is passed, the already-self-contained
``chunked_coulomb_energy_nl`` result gets the exclusion correction added a *second* time,
double-counting it. The dense branch is unaffected (it needs the correction exactly once).

This test exercises the real, public ``single_padded_energy`` entry point end-to-end
(not the isolated kernels) so it fails specifically on the double-counting bug, not on
anything already covered by ``test_protein_nl_explicit_parity.py``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.slow

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pdb"

PME_ALPHA = 0.34
CUTOFF = 9.0


def _ff_path() -> Path:
    import proxide

    return Path(proxide.__file__).parent / "assets" / "protein.ff19SB.xml"


def _load_vacuum_boxed_1vii():
    """Build a small (~596-atom), periodic-boxed, vacuum (no solvent) 1VII system.

    Deliberately avoids ``solvate_protein_to_bundle`` (thousands of atoms, too slow
    for a fast/CI-safe regression test in eager float64 -- see local-compute-limits
    rule) while still exercising the full production wiring: real AMBER ff19SB
    exclusions/charges, a real periodic box, and a real jax_md neighbor list.
    """
    from proxide import CoordFormat, OutputSpec, parse_structure

    pdb_path = DATA_DIR / "1VII.pdb"
    if not pdb_path.is_file():
        pytest.skip(f"Missing test PDB: {pdb_path}")

    spec = OutputSpec(
        parameterize_md=True, force_field=str(_ff_path()), coord_format=CoordFormat.Full
    )
    protein = parse_structure(str(pdb_path), spec)

    coords = jnp.asarray(protein.coordinates, dtype=jnp.float64)
    lo = jnp.min(coords, axis=0)
    hi = jnp.max(coords, axis=0)
    box = jnp.maximum(hi - lo + 24.0, jnp.array([32.0, 32.0, 32.0], dtype=jnp.float64))
    shift = box * 0.5 - (lo + hi) * 0.5
    r = coords + shift

    class _Proxy:
        def __getattr__(self, name):
            if name == "positions":
                return r
            if name == "pme_alpha":
                return PME_ALPHA
            if name == "nonbonded_cutoff":
                return CUTOFF
            if name == "box_size":
                return box
            return getattr(protein, name)

    from prolix.physics import neighbor_list as nl
    from prolix.physics.system import make_bundle_from_system

    excl_spec = nl.ExclusionSpec.from_protein(protein)
    bundle = make_bundle_from_system(
        _Proxy(), boundary_condition="periodic", exclusion_spec=excl_spec
    )
    return bundle, box


@pytest.mark.slow
def test_single_padded_energy_dense_matches_nl_under_real_pme_exclusions():
    """single_padded_energy(neighbor=None) must match single_padded_energy(neighbor=nbrs).

    Both must represent the SAME physical direct-space + PME-reciprocal Coulomb energy
    for a real protein with real (non-trivial) 1-2/1-3/1-4 exclusions and pme_alpha > 0
    -- the exact configuration debt 790 was filed against. Before the fix, the NL branch
    double-counts the PME exclusion correction and diverges by O(1000) kcal/mol; after
    the fix, both branches agree to float64 precision (only cutoff-induced long-range
    truncation differences remain, negligible at this alpha/cutoff combination).
    """
    from prolix.api.bundle_md import physics_system_from_bundle
    from prolix.batched_energy import single_padded_energy
    from prolix.physics import neighbor_list as nl
    from prolix.physics import pbc

    bundle, box = _load_vacuum_boxed_1vii()
    sys_obj = physics_system_from_bundle(bundle, bundle.positions)

    displacement_fn, _ = pbc.create_periodic_space(sys_obj.box_size)
    neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, sys_obj.box_size, CUTOFF)
    nbr0 = neighbor_fn.allocate(sys_obj.positions)
    nbr = neighbor_fn.update(sys_obj.positions, nbr0)
    assert not bool(nbr.did_buffer_overflow), "neighbor list capacity too small for this test"

    e_dense = float(single_padded_energy(sys_obj, displacement_fn, implicit_solvent=False))
    e_nl = float(
        single_padded_energy(sys_obj, displacement_fn, implicit_solvent=False, neighbor=nbr)
    )

    rel = abs(e_dense - e_nl) / abs(e_dense)
    assert rel < 0.02, (
        f"dense vs NL Coulomb mismatch under real PME exclusions: "
        f"e_dense={e_dense:.4f} e_nl={e_nl:.4f} diff={e_dense - e_nl:.4f} rel={rel:.6f} "
        "(debt 790 -- see module docstring for root cause)"
    )
