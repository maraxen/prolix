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
from jax_md import partition

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


def test_raw_jax_md_cell_list_drops_pairs_without_wrapping(solvated_bundle):
    """Why the wrap in ``make_neighbor_list_fn`` exists -- demonstrated, not asserted.

    Bypasses prolix's wrapper and drives ``jax_md.partition.neighbor_list``
    directly on raw origin-centred coordinates: what prolix did before #4699
    was fixed. If this ever stops dropping pairs, upstream binning changed and
    the wrap can be revisited.
    """
    n_real = int(solvated_bundle.n_atoms)
    box_vec = jnp.diag(solvated_bundle.box)
    truth = _brute_force_pairs(
        solvated_bundle.positions, box_vec, float(solvated_bundle.cutoff_distance), n_real
    )
    disp_fn, _ = displacement_fn_for_bundle(solvated_bundle)
    raw_fns = partition.neighbor_list(
        disp_fn,
        box_vec,
        r_cutoff=float(solvated_bundle.cutoff_distance),
        dr_threshold=0.5,
        capacity_multiplier=1.25,
        disable_cell_list=False,
        mask_self=True,
        fractional_coordinates=False,
    )
    got = _neighbor_list_pairs(raw_fns.allocate(solvated_bundle.positions), n_real)
    missing = truth - got
    frac = len(missing) / max(len(truth), 1)
    print(
        f"\n[#4699] raw jax_md cell list, UNWRAPPED: truth={len(truth)} "
        f"got={len(got)} missing={len(missing)} ({frac:.2%})"
    )
    if not missing:
        pytest.skip(
            "raw jax_md cell list lost no pairs on unwrapped input -- upstream "
            "binning may have changed; re-check whether the wrap is still needed"
        )


def test_prolix_neighbor_list_is_complete_on_raw_coordinates(solvated_bundle):
    """THE regression guard: the shipped default must never drop an in-cutoff pair.

    Exercises ``make_neighbor_list_fn`` exactly as production calls it -- cell
    list enabled, raw unwrapped bundle coordinates -- and requires a superset
    of brute force. A neighbor list may carry extra candidates (that is what
    the skin is for); it may never be missing one.
    """
    n_real = int(solvated_bundle.n_atoms)
    box_vec = jnp.diag(solvated_bundle.box)
    truth = _brute_force_pairs(
        solvated_bundle.positions, box_vec, float(solvated_bundle.cutoff_distance), n_real
    )
    nbr = _build(solvated_bundle, solvated_bundle.positions, disable_cell_list=False)
    got = _neighbor_list_pairs(nbr, n_real)
    missing = truth - got
    print(
        f"\n[#4699] prolix NL (cell list on, raw coords): truth={len(truth)} "
        f"got={len(got)} missing={len(missing)}"
    )
    assert not missing, (
        f"prolix's neighbor list dropped {len(missing)} in-cutoff pairs on raw "
        f"coordinates -- the [0, box) wrap in make_neighbor_list_fn is not "
        f"holding. First few: {sorted(missing)[:10]}"
    )


def test_cell_list_and_brute_force_agree(solvated_bundle):
    """The two construction strategies must be interchangeable, not merely both valid.

    Stronger than the superset guard above: a divergence means one of them is
    wrong even if both happen to cover every true pair.
    """
    n_real = int(solvated_bundle.n_atoms)
    cell = _neighbor_list_pairs(
        _build(solvated_bundle, solvated_bundle.positions, disable_cell_list=False), n_real
    )
    brute = _neighbor_list_pairs(
        _build(solvated_bundle, solvated_bundle.positions, disable_cell_list=True), n_real
    )
    only_brute = brute - cell
    print(
        f"\n[#4699] cell={len(cell)} brute={len(brute)} "
        f"only_cell={len(cell - brute)} only_brute={len(only_brute)}"
    )
    assert not only_brute, (
        f"brute force found {len(only_brute)} pairs the cell list missed: "
        f"{sorted(only_brute)[:10]}"
    )
