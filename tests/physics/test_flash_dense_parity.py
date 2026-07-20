"""Regression test for debt 761: force_fn_from_bundle vs the dense-autodiff reference.

``EnsemblePlan.run(..., use_flash_forces=True)`` swaps ``energy_fn_from_bundle``
(dense O(N^2) energy + ``jax.grad``) for ``force_fn_from_bundle``
(``single_padded_force(..., use_flash=True)`` -- FlashMD's tiled, checkpointed
direct-space kernel) in ``EnsemblePlan._setup_integrator``. This is only a
faithful substitution once debts 763 (tile-size-256 padding), 770 (pme.py
ConcretizationTypeError), 804 (dispersion-tail atom_mask-vs-count), and 805
(missing periodic minimum-image) are all fixed -- before those fixes,
``flash_explicit_forces`` silently produced wildly wrong values (a 52000x
energy blowup at one point) while still returning a finite, crash-free
result. This test promotes the real-bundle dense-vs-flash comparison
performed ad hoc in this session (see
``.praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md``,
"Debt 761 verdict") to a permanent, committed regression guard, on the exact
``force_fn_from_bundle``/``energy_fn_from_bundle`` pair ``EnsemblePlan`` uses
-- not just the lower-level ``single_padded_force``/``single_padded_energy``
functions the original ad hoc benchmark compared -- so it also covers the
AMBER 1-4 exception-energy term ``force_fn_from_bundle`` adds back via a
separate ``jax.grad`` pass.

Deliberately real, bundle-scale (not a synthetic lattice): the synthetic
parity checks used earlier in this session's investigation only exercised
the trivial all-sentinel (no exclusions) case, which is exactly what missed
debts 804/805 (both invisible without a real, densely-populated exclusion
table and real periodic geometry). ``solvate_protein_to_bundle`` on 1VII
(~1963 real / 5000 padded atoms) runs in well under a second locally in
eager float32 (per local-compute-limits.md's narrow-run guidance) -- not the
"whole JAX/pytest suite" scale that rule warns about.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.slow

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pdb"
TARGET_BOX_SIZE = [32.0, 32.0, 32.0]


@pytest.fixture(autouse=True)
def _force_float32():
    """Explicit float32 (not just "default float32").

    ``jax.config`` is process-global and sticky across the whole pytest
    session -- a module-level ``jax.config.update`` call here would only
    win if this module happens to be *imported* after every other test
    module that turns x64 on (e.g. ``test_nl_kernel_exclusion_parity.py``),
    which is collection-order-dependent and not something to rely on. An
    autouse fixture re-asserts the desired state at *test execution* time
    instead, which always runs after all collection-time imports are done.

    This test guards the actual float32 production path debt 761 wires
    into ``EnsemblePlan`` (matching this session's own benchmark
    measurements), not a from-scratch x64 parity claim. (Under x64 this
    test also surfaces an unrelated, separate float32/float64 mixing bug
    in ``pme.py``'s ``spme_energy_with_forces`` -- filed as its own debt
    item rather than fixed here, out of debt 761's scope.)
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", previous)


def _ff_path() -> Path:
    import proxide

    return Path(proxide.__file__).parent / "assets" / "protein.ff19SB.xml"


def _solvated_1vii_bundle():
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.solvation import solvate_protein_to_bundle
    from prolix.physics.water_models import WaterModelType

    # Force float32 here too, not just in the autouse fixture -- this
    # class-scoped-fixture-backed helper can run during test *setup*,
    # before the function-scoped autouse fixture below has applied for the
    # first test in the class (pytest sets up broader-scope fixtures before
    # narrower-scope ones). See _force_float32's docstring for why this
    # can't just be a module-level jax.config.update call.
    jax.config.update("jax_enable_x64", False)

    pdb_path = DATA_DIR / "1VII.pdb"
    if not pdb_path.is_file():
        pytest.skip(f"Missing test PDB: {pdb_path}")

    ff_path = _ff_path()
    if not ff_path.is_file():
        pytest.skip(f"Missing bundled force field: {ff_path}")

    spec = OutputSpec(
        parameterize_md=True, force_field=str(ff_path), coord_format=CoordFormat.Full
    )
    protein = parse_structure(str(pdb_path), spec)
    return solvate_protein_to_bundle(
        protein,
        padding=8.0,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array(TARGET_BOX_SIZE),
    )


class TestFlashForcesDenseParity:
    """force_fn_from_bundle must match -grad(energy_fn_from_bundle) on a real bundle."""

    @pytest.fixture(scope="class")
    def bundle(self):
        try:
            import proxide  # noqa: F401
        except ImportError:
            pytest.skip("proxide not installed")
        return _solvated_1vii_bundle()

    def test_flash_matches_autodiff_reference(self, bundle):
        from prolix.api.bundle_md import energy_fn_from_bundle, force_fn_from_bundle

        positions = bundle.positions
        atom_mask = bundle.atom_mask[:, None]

        reference_force_fn = jax.grad(energy_fn_from_bundle(bundle))
        f_reference = -reference_force_fn(positions) * atom_mask

        f_flash = force_fn_from_bundle(bundle)(positions) * atom_mask

        assert jnp.all(jnp.isfinite(f_flash)), "flash forces contain NaN/Inf"

        max_abs_ref = jnp.max(jnp.abs(f_reference))
        max_abs_diff = jnp.max(jnp.abs(f_flash - f_reference))
        rel_diff = max_abs_diff / max_abs_ref

        # Measured 2026-07-19 on the real 1VII bundle: rel diff ~1.3e-5
        # (float32 total-energy agreement -2520.10 vs -2520.06 kcal/mol).
        # Generous 100x margin over that measurement absorbs platform/BLAS
        # summation-order differences without masking a real regression --
        # any of debts 763/770/804/805 recurring produces errors many
        # orders of magnitude larger than this (52000x, or 5.46% relative).
        assert rel_diff < 1e-3, (
            f"flash forces diverge from the dense-autodiff reference: "
            f"max_abs_diff={float(max_abs_diff):.6g}, "
            f"max_abs_ref={float(max_abs_ref):.6g}, rel_diff={float(rel_diff):.6g}"
        )

    def test_flash_rejects_implicit_solvent_bundle(self, bundle):
        """force_fn_from_bundle must raise, not silently drop the GB term."""
        import dataclasses

        from prolix.api.bundle_md import force_fn_from_bundle

        implicit_shape_spec = dataclasses.replace(
            bundle.shape_spec, has_implicit_solvent=True
        )
        implicit_bundle = dataclasses.replace(bundle, shape_spec=implicit_shape_spec)

        with pytest.raises(ValueError, match="implicit-solvent"):
            force_fn_from_bundle(implicit_bundle)

    def test_flash_rejects_non_periodic_bundle(self, bundle):
        """force_fn_from_bundle must raise, not silently use the unverified vacuum path."""
        import dataclasses

        from prolix.api.bundle_md import force_fn_from_bundle

        vacuum_shape_spec = dataclasses.replace(bundle.shape_spec, has_pbc=False)
        vacuum_bundle = dataclasses.replace(bundle, shape_spec=vacuum_shape_spec)

        with pytest.raises(ValueError, match="periodic"):
            force_fn_from_bundle(vacuum_bundle)
