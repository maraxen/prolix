"""Exclusion overflow must raise (no silent truncate)."""

from __future__ import annotations

import numpy as np
import pytest

from prolix.physics.neighbor_list import ExclusionSpec, map_exclusions_to_dense_padded


def test_map_exclusions_raises_when_over_cap():
    empty_e = np.zeros((0, 2), dtype=np.int32)
    spec = ExclusionSpec(
        n_atoms=2,
        idx_12_13=np.array([[0, 1]] * 8, dtype=np.int32),
        idx_14=empty_e,
        scale_14_elec=1.0,
        scale_14_vdw=1.0,
        exception_pairs=empty_e,
        exception_sigmas=np.zeros((0,), dtype=np.float32),
        exception_epsilons=np.zeros((0,), dtype=np.float32),
        exception_chargeprods=np.zeros((0,), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="needs"):
        map_exclusions_to_dense_padded(spec, max_exclusions=2)
