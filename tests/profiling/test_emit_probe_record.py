"""P7: ProbeRecord emit must keep attribution_method when every scope is None.

Cluster array 20762713: timing succeeded, then ClaimValidityError because
`{... if value is not None} or None` dropped an empty attribution dict.
"""

from __future__ import annotations

from scripts.experiments.profile_b1_flash_vs_autodiff_forces import _emit_probe_record
from scripts.profiling.record import ProbeRecord


def test_emit_keeps_all_none_scopes(tmp_path):
    path = tmp_path / "record.json"
    _emit_probe_record(
        protein="1vii",
        stage=1,
        n_real=1963,
        n_padded=5000,
        n_warmup=3,
        n_trials=20,
        n_inner=5,
        results={"autodiff_forces": {"mean_ms": 0.8}},
        path=path,
        force_mode="dense",
        pme="off",
        grid_spacing="na",
        scopes_arm="on",
        platform="cpu",
        scopes={"dense_bonded_bond": None, "flash_bonded": None},
    )
    rec = ProbeRecord.read(path)
    assert rec.scopes == {"dense_bonded_bond": None, "flash_bonded": None}
    assert rec.attribution_method == {}
