"""Compiled-HLO scope inventory for the P2 ``named_scope`` labels.

Asserts against the **compiled** module (``jax.jit(fn).lower(*args).compile().as_text()``),
not the lowering, at N=64 -- ``scripts/experiments/profile_b1_water_trace.py`` (lines
31-44) documents that at toy scale every ``pme_*`` named_scope label can vanish from
compiled HLO even though the underlying computation genuinely executes and is
independently verified correct, while other labels (e.g. ``settle_*``) survive at the
same scale. A pre-optimization lowering is therefore not evidence of survival, and an
absent PME-class label at this scale is an expected, non-blocking deferral -- see
``.praxia/docs/specs/260817_jax-profiling-optimization-workflow.md`` section P2 for the
per-label-class terminal-state rules this test implements.

Marked ``@pytest.mark.slow`` (excluded from CI's default marker expression) because it
pays a real XLA compile; run deliberately with::

    uv run pytest tests/physics/test_scope_inventory.py -q -m slow
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.slow

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "b1_init_exec_scope_inventory",
    _ROOT / "scripts" / "benchmarks" / "b1_init_exec.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_b1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_b1)

# src/prolix/batched_energy.py::single_padded_energy (dense path). Excludes
# dense_pme_reciprocal, which the P2 gate treats as a conditionally-passed,
# recorded deferral rather than a hard requirement at this scale -- see
# test_dense_pme_reciprocal_label_recorded_but_not_gated below.
#
# _four_water_bundle's rigid TIP3P topology has real (non-empty-before-
# padding) bonds, urey_bradley_bonds, angles, LJ, and Coulomb terms, but
# ZERO real dihedrals/impropers/cmap torsions (rigid water has none) --
# their bucket-padded arrays are all-mask-zero. XLA constant-folds an
# all-mask-zero contribution away at compile time, and the named_scope
# label goes with it: this is the P2 gate's "fused away and the label with
# it" case (b), NOT "the wrapping never landed" (verified separately below
# via a source-level check, since there is no real op left in HLO to find
# either way at this fixture).
_DENSE_LABELS_WITH_REAL_TERMS = (
    "dense_bonded_bond",
    "dense_bonded_urey_bradley",
    "dense_bonded_angle",
    "dense_lj_direct",
    "dense_coulomb_direct",
)
_DENSE_LABELS_ZERO_REAL_TERMS_AT_THIS_FIXTURE = (
    "dense_bonded_dihedral",
    "dense_bonded_improper",
    "dense_bonded_cmap",
)
_DENSE_PME_LABEL = "dense_pme_reciprocal"

# src/prolix/physics/flash_explicit.py. flash_bonded lives in
# flash_explicit_total_energy (the bonded+nonbonded entry point);
# flash_nonbonded_tiles lives in chunked_explicit_nonbonded_energy, reached by
# the default PME electrostatic_method both flash_explicit_total_energy and
# flash_explicit_forces use.
_FLASH_ENERGY_LABELS = ("flash_bonded", "flash_nonbonded_tiles")
# flash_lj_only lives in _chunked_lj_only_energy, which only the EFA/
# EFA_LEBEDEV electrostatic_method branches call -- unreachable via the
# default PME path exercised above.
_FLASH_LJ_ONLY_LABEL = "flash_lj_only"
# flash_grad wraps the eqx.filter_grad call inside flash_explicit_forces --
# not exercised by flash_explicit_total_energy, which never differentiates.
_FLASH_GRAD_LABEL = "flash_grad"


def _four_water_bundle():
    return _b1._four_water_bundle()


def _missing_labels(hlo_text: str, labels: tuple[str, ...]) -> list[str]:
    return [label for label in labels if label not in hlo_text]


def _assert_no_missing(missing: list[str], where: str) -> None:
    assert not missing, (
        f"label(s) {missing} not found in compiled HLO ({where}) at N=64 -- per "
        "the P2 gate, an absent non-PME label requires confirming the wrapped "
        "call site survived at all (distinguish 'fused away and the label with "
        "it' from 'the wrapping never landed', the latter being a hard block); "
        "see .praxia/docs/specs/260817_jax-profiling-optimization-workflow.md "
        "section P2"
    )


def test_dense_labels_appear_in_compiled_hlo():
    """Dense labels backed by real (non-zero-mask) terms survive compilation at N=64."""
    from prolix.api.bundle_md import energy_fn_from_bundle

    bundle = _four_water_bundle()
    # bundle.positions is the full PADDED array (ATOM_BUCKETS-sized) -- the
    # bucket floor a 12-real-atom bundle lands in is 64, which is the shape
    # that actually gets traced/compiled. active_positions() would slice down
    # to the 12 real atoms, defeating the point of checking N=64.
    positions = bundle.positions
    assert positions.shape[0] == 64, (
        f"expected N=64 (ATOM_BUCKETS floor for a 12-real-atom bundle), got "
        f"{positions.shape[0]} -- this test's whole premise is checking the "
        "largest size P4 executes"
    )

    energy_fn = energy_fn_from_bundle(bundle)
    hlo_text = jax.jit(energy_fn).lower(positions).compile().as_text()

    _assert_no_missing(
        _missing_labels(hlo_text, _DENSE_LABELS_WITH_REAL_TERMS), "dense energy path"
    )


def test_dense_labels_with_zero_real_terms_have_source_wrap_but_may_fold():
    """dihedral/improper/cmap: this fixture's rigid-water topology has none.

    A bucket-padded, all-mask-zero term is legitimately constant-folded away
    by XLA at compile time -- there is no real op left to attribute a scope
    to, in HLO, with or without the wrap. Per the P2 gate's case (b), this
    requires confirming the wrapped call site *survived in source* (the
    alternative being "the wrapping never landed", a hard P2 implementation
    bug) rather than asserting HLO presence, which this fixture cannot
    demonstrate either way.
    """
    src = (
        _ROOT / "src" / "prolix" / "batched_energy.py"
    ).read_text()
    missing_wrap = [
        label
        for label in _DENSE_LABELS_ZERO_REAL_TERMS_AT_THIS_FIXTURE
        if f'jax.named_scope("{label}")' not in src
    ]
    assert not missing_wrap, (
        f"label(s) {missing_wrap} have no `jax.named_scope(...)` wrap in "
        "batched_energy.py source at all -- this IS the P2 hard-block case "
        "('the wrapping never landed'), not the expected compiler fold"
    )

    from prolix.api.bundle_md import energy_fn_from_bundle

    bundle = _four_water_bundle()
    energy_fn = energy_fn_from_bundle(bundle)
    hlo_text = jax.jit(energy_fn).lower(bundle.positions).compile().as_text()
    for label in _DENSE_LABELS_ZERO_REAL_TERMS_AT_THIS_FIXTURE:
        print(f"{label} present in compiled HLO at N=64 (expected folded away): {label in hlo_text}")


def test_dense_pme_reciprocal_label_recorded_but_not_gated():
    """dense_pme_reciprocal absence at N=64 is an expected, recorded deferral to P7.

    See this module's docstring and profile_b1_water_trace.py's documented
    finding. Not asserted either way -- only recorded for the step notes.
    """
    from prolix.api.bundle_md import energy_fn_from_bundle

    bundle = _four_water_bundle()
    positions = bundle.positions
    energy_fn = energy_fn_from_bundle(bundle)
    hlo_text = jax.jit(energy_fn).lower(positions).compile().as_text()

    found = _DENSE_PME_LABEL in hlo_text
    print(f"{_DENSE_PME_LABEL} present in compiled HLO at N=64: {found}")


def _flash_energy_fn(bundle, **kwargs):
    """Close over `bundle` (concrete topology) and trace only `positions`.

    Matches the production pattern in bundle_md.py's force_fn_from_bundle /
    energy_fn_from_bundle: physics_system_from_bundle(bundle, positions)
    reconstructs `sys` fresh from the closed-over (non-traced) bundle plus
    the traced `positions`, so host-side topology processing inside
    _total_energy_fn (e.g. topology.find_bonded_exclusions's numpy graph
    traversal on sys.bonds) sees concrete arrays rather than tracers.
    Jitting flash_explicit_total_energy/flash_explicit_forces directly over
    a freshly-reconstructed `sys` (with bonds as a dynamic pytree leaf)
    raises TracerArrayConversionError -- this is not how the codebase
    actually calls these functions under jit anywhere.
    """
    from prolix.api.bundle_md import physics_system_from_bundle
    from prolix.physics.flash_explicit import flash_explicit_total_energy

    def _fn(positions):
        sys = physics_system_from_bundle(bundle, positions)
        return flash_explicit_total_energy(sys, **kwargs)

    return _fn


def _flash_forces_fn(bundle, **kwargs):
    from prolix.api.bundle_md import physics_system_from_bundle
    from prolix.physics.flash_explicit import flash_explicit_forces

    def _fn(positions):
        sys = physics_system_from_bundle(bundle, positions)
        return flash_explicit_forces(sys, **kwargs)

    return _fn


def test_flash_energy_labels_appear_in_compiled_hlo():
    """flash_bonded + flash_nonbonded_tiles survive compilation at N=64 (PME method)."""
    bundle = _four_water_bundle()
    fn = _flash_energy_fn(bundle)
    hlo_text = jax.jit(fn).lower(bundle.positions).compile().as_text()

    _assert_no_missing(
        _missing_labels(hlo_text, _FLASH_ENERGY_LABELS), "flash_explicit_total_energy"
    )


def test_flash_lj_only_label_appears_under_efa_method():
    """flash_lj_only is only reachable via the EFA/EFA_LEBEDEV electrostatic path."""
    bundle = _four_water_bundle()
    fn = _flash_energy_fn(bundle, electrostatic_method="efa")
    hlo_text = jax.jit(fn).lower(bundle.positions).compile().as_text()

    _assert_no_missing(_missing_labels(hlo_text, (_FLASH_LJ_ONLY_LABEL,)), "EFA path")


def test_flash_grad_label_appears_in_compiled_hlo():
    """flash_grad wraps the eqx.filter_grad call inside flash_explicit_forces."""
    bundle = _four_water_bundle()
    fn = _flash_forces_fn(bundle)
    hlo_text = jax.jit(fn).lower(bundle.positions).compile().as_text()

    _assert_no_missing(_missing_labels(hlo_text, (_FLASH_GRAD_LABEL,)), "flash_explicit_forces")


def test_settle_scopes_at_production_scale():
    """Test settle_* named_scope labels at production scale (N≈895 waters).

    This test addresses #4330 Fix 2c: investigate whether settle_* scope labels
    survive compilation at production scale, or whether they are absorbed into
    fusion at larger system sizes (vs. the N=64 tests above).

    Hypothesis: at production scale (n_waters ~ 895), XLA's fusion pass absorbs
    the small per-water settle_* operations into a larger surrounding fusion,
    losing the scope boundary.

    Outcome: compile a settle_langevin integrator step at ~895 waters (the same
    scale as this project's validated dt=1.0fs production gate, see project
    CLAUDE.md) and report which settle_* labels (settle_pos_eigh,
    settle_cramers_rule, settle_rattle_loop) are present or absent in the
    resulting HLO text.

    Uses the same real-system construction pattern as
    test_settle_temperature_control.py's _mean_rigid_t_after_burn (pre-
    equilibrated TIP3P water box + a real PME energy function + the real
    settle.get_water_indices helper), not a hand-rolled fixture -- a hand-rolled
    n_waters-many-"atoms" grid would silently violate settle_langevin's actual
    (N_waters, 3) water_indices / (n_atoms,) mass contract and either crash or
    (worse) run a physically meaningless system.

    NOTE: This test compiles a real ~895-water JAX system and performs a real XLA
    compilation (module-level ``pytestmark = pytest.mark.slow`` above excludes it
    from CI's default ``-m "not slow"`` run). Per project rules, this test MUST
    NOT be executed locally -- only on cluster/titanix. It writes its HLO findings
    to the test output; the finding (present/absent) confirms or denies the
    fusion hypothesis and should be manually inspected from the cluster run logs.
    """
    from prolix.physics import pbc, settle, system
    from .test_explicit_langevin_tip3p_parity import (
        _equil_water_positions,
        _proxide_params_pure_water,
    )

    jax.config.update("jax_enable_x64", True)
    n_waters = 895  # matches this project's validated dt=1.0fs production gate scale
    n_atoms = n_waters * 3

    positions_a, box_edge = _equil_water_positions(n_waters, seed=0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    positions = jnp.asarray(positions_a, dtype=jnp.float64)

    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    pme_grid = max(16, round(box_edge / 1.0))
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=pme_grid, pme_alpha=0.34,
        cutoff_distance=9.0, strict_parameterization=False,
    )

    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

    init_fn, step_fn = settle.settle_langevin(
        energy_fn, shift_fn, dt=1.0, kT=1.0, gamma=10.0, mass=mass,
        water_indices=water_indices, box=box_vec,
        project_ou_momentum_rigid=True, projection_site="post_o",
    )

    state = init_fn(jax.random.key(0), positions, mass=mass)

    # Compile the step and dump HLO (apply_fn takes only state -- the RNG key is
    # threaded internally via state.key, per settle_langevin's LangevinState contract;
    # see test_settle_temperature_control.py's identical apply_j(state) call pattern).
    hlo_text = jax.jit(step_fn).lower(state).compile().as_text()

    # Report findings for settle_* scopes (do not assert, only record)
    settle_labels = ("settle_pos_eigh", "settle_cramers_rule", "settle_rattle_loop")
    print(f"\n=== Production-scale settle_* scope inventory (N={n_waters}) ===")
    for label in settle_labels:
        present = label in hlo_text
        print(f"{label}: {'PRESENT' if present else 'ABSENT'}")
    print("=== End inventory ===\n")

    # Optional: if a specific label is known to be required, add assertions here
    # For now, just report the findings and let the user inspect them
