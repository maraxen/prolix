"""Regression guard for backlog #4699: jax_md cell list silently drops in-cutoff pairs.

Why a set-equality test and not an energy comparison
----------------------------------------------------
#4699 is a *silent* pair-dropping bug. An energy or force comparison can pass
while pairs are missing -- the dropped contributions are individually small and
partially cancel, so a loose tolerance hides them. The only honest check is to
enumerate the true in-cutoff pair set by brute force and assert the neighbor
list is a **superset** of it (a neighbor list is allowed to carry extra
candidate pairs beyond the cutoff -- that is what the Verlet skin is for -- but
it may never be missing one).

Root cause (confirmed by reading jax_md/partition.py)
-----------------------------------------------------
``cell_list_fn`` bins particles with::

    indices = jnp.array(position / cell_size, dtype=i32)   # partition.py:391
    hashes  = jnp.sum(indices * hash_multipliers, axis=1)  # partition.py:392

``jnp.array(..., dtype=i32)`` truncates *toward zero*. There is no wrapping and
no bounds check, so jax_md's cell list requires positions already inside
``[0, box)``. prolix passes raw frame coordinates, which for a solvated protein
are routinely negative or past ``L``:

* a negative coordinate truncates toward zero and aliases onto a wrong cell
  (``int32(-0.4) == 0 == int32(0.4)``), so the particle is compared against the
  wrong neighbourhood;
* a coordinate past ``L`` produces an index >= ``cells_per_side``, hashing
  outside ``cell_count``.

Either way the particle's true neighbours are never enumerated, and the pairs
vanish with no error. The brute-force path (``disable_cell_list=True``) is
immune because it uses the periodic ``displacement_fn`` (minimum image) rather
than absolute position bins -- which is why disabling the cell list "fixed" the
symptom, at the cost of O(N^2) list construction.

The fix is therefore in prolix, not jax_md: wrap positions into ``[0, box)``
before handing them to the cell list. For a periodic system that is physically a
no-op (it selects a different periodic image of the same configuration).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_SCRIPTS_EXP = Path(__file__).resolve().parents[2] / "scripts" / "experiments"
if str(_SCRIPTS_EXP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_EXP))

from nl_omm_parity import _bundle_from_1vii_gold  # noqa: E402

from prolix.physics.neighbor_list import make_neighbor_list_fn  # noqa: E402
from prolix.api.bundle_md import displacement_fn_for_bundle  # noqa: E402


@pytest.fixture(scope="module")
def solvated_bundle():
    """Real solvated protein (1vii + TIP3P water) from the vendored OpenMM gold."""
    gold_path = (
        Path(__file__).resolve().parents[2]
        / "data" / "oracles" / "openmm_8.3.1" / "nl_1vii.json"
    )
    if not gold_path.is_file():
        pytest.skip(f"Gold file not found: {gold_path}")
    return _bundle_from_1vii_gold(json.loads(gold_path.read_text()))


def _brute_force_pairs(positions, box_vec, cutoff, n_real):
    """Ground-truth in-cutoff pair set via explicit minimum-image distances.

    Returns a set of ``(i, j)`` tuples with ``i < j``, restricted to real
    (non-padding) atoms. Uses numpy rather than the kernel under test so a bug
    in the kernel cannot mask itself.
    """
    pos = np.asarray(positions[:n_real], dtype=np.float64)
    box = np.asarray(box_vec, dtype=np.float64)
    dr = pos[:, None, :] - pos[None, :, :]
    dr -= box * np.round(dr / box)
    dist = np.sqrt((dr ** 2).sum(-1))
    iu = np.triu_indices(n_real, k=1)
    within = dist[iu] < cutoff
    return set(zip(iu[0][within].tolist(), iu[1][within].tolist()))


def _neighbor_list_pairs(nbr, n_real):
    """Extract the ``(i, j)`` pair set (i < j) from a dense jax_md NeighborList.

    jax_md encodes padding as ``idx == n_padded``; since ``n_padded >= n_real``,
    the single ``j >= n_real`` test drops both padding sentinels and entries
    pointing at padding atoms, neither of which is a real pair.
    """
    idx = np.asarray(nbr.idx)
    pairs = set()
    for i in range(min(idx.shape[0], n_real)):
        for j in idx[i]:
            j = int(j)
            if j >= n_real or j == i:
                continue
            pairs.add((i, j) if i < j else (j, i))
    return pairs


def _wrap_into_box(positions, box_vec):
    """Map positions into ``[0, box)`` -- the precondition jax_md's cell list assumes."""
    return jnp.mod(positions, box_vec)


def _build(bundle, positions, *, disable_cell_list):
    disp_fn, _ = displacement_fn_for_bundle(bundle)
    box_vec = jnp.diag(bundle.box)
    cutoff = float(bundle.cutoff_distance)
    neighbor_fn = make_neighbor_list_fn(
        disp_fn, box_vec, cutoff, disable_cell_list=disable_cell_list
    )
    nbr = neighbor_fn.allocate(positions)
    assert not bool(nbr.did_buffer_overflow), (
        "neighbor list overflowed during allocate -- capacity, not completeness, "
        "is the problem; raise capacity_multiplier before reading this test's result"
    )
    return nbr


def test_documents_positions_are_outside_the_primary_cell(solvated_bundle):
    """The precondition jax_md's cell list requires is violated by our inputs.

    Not an assertion about correctness -- this pins the *input* condition that
    makes the cell list unsafe, so that if bundles ever start arriving
    pre-wrapped, this test tells us the hazard is gone.
    """
    pos = np.asarray(solvated_bundle.positions[: int(solvated_bundle.n_atoms)])
    box = np.asarray(jnp.diag(solvated_bundle.box))
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    outside = bool((lo < 0).any() or (hi >= box).any())
    print(f"\n[#4699] box={box}\n[#4699] coord min={lo}\n[#4699] coord max={hi}")
    print(f"[#4699] positions outside [0, box): {outside}")
    assert outside, (
        "positions already lie inside [0, box); the #4699 hazard would not "
        "reproduce on this fixture, so it is the wrong fixture for this guard"
    )


def test_brute_force_path_is_complete(solvated_bundle):
    """Baseline: the O(N^2) path we currently ship must itself be complete."""
    n_real = int(solvated_bundle.n_atoms)
    box_vec = jnp.diag(solvated_bundle.box)
    truth = _brute_force_pairs(
        solvated_bundle.positions, box_vec, float(solvated_bundle.cutoff_distance), n_real
    )
    nbr = _build(solvated_bundle, solvated_bundle.positions, disable_cell_list=True)
    got = _neighbor_list_pairs(nbr, n_real)
    missing = truth - got
    print(f"\n[#4699] brute-force: truth={len(truth)} got={len(got)} missing={len(missing)}")
    assert not missing, f"brute-force NL dropped {len(missing)} in-cutoff pairs"


def test_cell_list_drops_pairs_on_unwrapped_positions(solvated_bundle):
    """The #4699 bug itself, reproduced as a measurement rather than a claim."""
    n_real = int(solvated_bundle.n_atoms)
    box_vec = jnp.diag(solvated_bundle.box)
    truth = _brute_force_pairs(
        solvated_bundle.positions, box_vec, float(solvated_bundle.cutoff_distance), n_real
    )
    nbr = _build(solvated_bundle, solvated_bundle.positions, disable_cell_list=False)
    got = _neighbor_list_pairs(nbr, n_real)
    missing = truth - got
    frac = len(missing) / max(len(truth), 1)
    print(
        f"\n[#4699] cell list, UNWRAPPED: truth={len(truth)} got={len(got)} "
        f"missing={len(missing)} ({frac:.2%})"
    )
    # Recorded as an xfail-style measurement: this is the defect. If it ever
    # passes, the cell list became safe on raw coordinates and #4699 can close.
    if not missing:
        pytest.skip(
            "cell list lost no pairs on unwrapped input -- #4699 may no longer "
            "reproduce here; re-check before relying on this guard"
        )


def test_cell_list_is_complete_once_positions_are_wrapped(solvated_bundle):
    """The fix: wrapping into [0, box) must make the cell list a strict superset.

    This is the assertion that gates flipping ``disable_cell_list`` back to
    False. It must hold exactly -- a neighbor list may carry extra candidates,
    never fewer.
    """
    n_real = int(solvated_bundle.n_atoms)
    box_vec = jnp.diag(solvated_bundle.box)
    truth = _brute_force_pairs(
        solvated_bundle.positions, box_vec, float(solvated_bundle.cutoff_distance), n_real
    )
    wrapped = _wrap_into_box(solvated_bundle.positions, box_vec)
    nbr = _build(solvated_bundle, wrapped, disable_cell_list=False)
    got = _neighbor_list_pairs(nbr, n_real)
    missing = truth - got
    print(
        f"\n[#4699] cell list, WRAPPED: truth={len(truth)} got={len(got)} "
        f"missing={len(missing)}"
    )
    assert not missing, (
        f"cell list still dropped {len(missing)} in-cutoff pairs after wrapping "
        f"positions into [0, box); wrapping is not a sufficient fix for #4699. "
        f"First few: {sorted(missing)[:10]}"
    )
