"""Tests for xtrax.profiling: ProbeRecord + the claim-validity contract.

prolix consumes this package as a dependency (xtrax >= 0.4.0a7); these are
integration checks of the contract prolix's probes rely on, not a second copy
of xtrax's own unit tests.

Assertion list mirrors .praxia/docs/specs/260817_jax-profiling-optimization-workflow.md
section P1's Gate paragraph verbatim -- each test below is named after the
specific claim that paragraph makes, so a spec change that drops a guarantee
shows up as a named test disappearing, not a silent gap.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from xtrax.profiling.claims import (
    CONTRACT_VERSION,
    ClaimClass,
    ClaimValidityError,
    assert_claim_supported,
    paired_configs,
    select_sources,
)
from xtrax.profiling.record import ProbeRecord, _normalize_device_kind

def _record(
    *,
    stage: int,
    platform: str = "gpu",
    device_kind: str | None = "h200",
    n_atoms: int = 5000,
    metrics: dict[str, float] | None = None,
    scopes: dict[str, tuple[float, int] | None] | None = None,
    attribution_method: dict[str, str] | None = None,
    config: dict[str, str] | None = None,
    xla_flags: str = "",
    probe_id: str = "test-probe",
    # Fixed, not auto-captured from the real repo -- otherwise every
    # TERM_RANKING/END_TO_END "must not raise" test would become dependent
    # on the working tree being clean at test-run time.
    git_sha: str = "cafefeed",
) -> ProbeRecord:
    return ProbeRecord(
        probe_id=probe_id,
        stage=stage,
        n_atoms=n_atoms,
        platform=platform,
        device_kind=device_kind,
        metrics=dict(metrics or {}),
        scopes=scopes,
        attribution_method=attribution_method,
        config=dict(config or {}),
        xla_flags=xla_flags,
        git_sha=git_sha,
    )


def _term_ranking_record(stage: int, **overrides) -> ProbeRecord:
    defaults = dict(
        metrics={"total_step_seconds": 1.0},
        scopes={"bonded": (0.1, 10), "pme_recip": (0.2, 5)},
        attribution_method={"bonded": "named_scope", "pme_recip": "named_scope"},
    )
    defaults.update(overrides)
    return _record(stage=stage, **defaults)


def _end_to_end_record(n_atoms: int, **overrides) -> ProbeRecord:
    defaults = dict(metrics={"total_step_seconds": 1.0})
    defaults.update(overrides)
    return _record(stage=2, n_atoms=n_atoms, **defaults)


# --- ProbeRecord construction / validation ----------------------------------


def test_stage2_gpu_platform_mismatch_raises_at_construction_not_claim_time():
    with pytest.raises(ClaimValidityError):
        _record(stage=2, platform="cpu")


def test_stage2_requires_device_kind():
    with pytest.raises(ClaimValidityError):
        _record(stage=2, device_kind=None)


def test_invalid_stage_raises():
    with pytest.raises(ClaimValidityError):
        _record(stage=4)


def test_non_positive_n_atoms_raises():
    with pytest.raises(ClaimValidityError):
        _record(stage=0, n_atoms=0)


def test_json_round_trip_preserves_scopes_tuple_vs_none():
    rec = _term_ranking_record(
        stage=2, scopes={"bonded": (0.1, 10), "pme_recip": None}
    )
    restored = ProbeRecord.from_json(rec.to_json())
    assert restored.scopes["bonded"] == (0.1, 10)
    assert isinstance(restored.scopes["bonded"], tuple)
    assert restored.scopes["pme_recip"] is None


# --- TERM_RANKING: stage floor ----------------------------------------------


def test_stage1_record_raises_on_term_ranking():
    records = [_term_ranking_record(stage=1)]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.TERM_RANKING)


def test_stage2_record_permits_term_ranking():
    records = [_term_ranking_record(stage=2)]
    assert_claim_supported(records, ClaimClass.TERM_RANKING)  # must not raise


def test_mixed_stage_set_selects_stage2_and_permits_term_ranking():
    records = [
        _term_ranking_record(stage=0, platform="cpu", device_kind=None),
        _term_ranking_record(stage=1, platform="cpu", device_kind=None),
        _term_ranking_record(stage=2),
    ]
    assert_claim_supported(records, ClaimClass.TERM_RANKING)  # must not raise

    without_stage2 = records[:2]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(without_stage2, ClaimClass.TERM_RANKING)


def test_scopes_none_raises_on_term_ranking_even_at_stage2():
    records = [_term_ranking_record(stage=2, scopes=None)]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.TERM_RANKING)


# --- TERM_RANKING / END_TO_END: unanimity -----------------------------------


def test_differing_xla_flags_raises_on_term_ranking():
    records = [
        _term_ranking_record(stage=2, xla_flags="", probe_id="a"),
        _term_ranking_record(
            stage=2, xla_flags="--xla_gpu_shard_autotuning=false", probe_id="b"
        ),
    ]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.TERM_RANKING)


def test_differing_device_kind_raises_on_term_ranking():
    records = [
        _term_ranking_record(stage=2, device_kind="h200", probe_id="a"),
        _term_ranking_record(stage=2, device_kind="l40s", probe_id="b"),
    ]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.TERM_RANKING)


# --- END_TO_END: target declaration, scale guard, min-reducer --------------


def test_end_to_end_without_target_and_without_allow_no_target_raises():
    records = [_end_to_end_record(n_atoms=5000)]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.END_TO_END)


def test_end_to_end_allow_no_target_does_not_raise():
    records = [_end_to_end_record(n_atoms=5000)]
    assert_claim_supported(
        records, ClaimClass.END_TO_END, allow_no_target=True
    )  # must not raise


def test_stage2_n_atoms_2000_raises_end_to_end_against_dhfr_target():
    records = [_end_to_end_record(n_atoms=2000)]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.END_TO_END, target_n_atoms=23558)


def test_end_to_end_missing_total_step_seconds_metric_raises():
    records = [_record(stage=2, n_atoms=5000, metrics={})]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.END_TO_END, target_n_atoms=20000)


def test_end_to_end_min_reducer_flips_passing_claim_to_raising():
    """Adding a further valid same-stage record with a *smaller* n_atoms can
    tighten the scale guard and flip a passing END_TO_END claim to raising --
    the min-reducer's deliberate, fail-closed consequence (not a regression).
    """
    target = 20000
    passing = [_end_to_end_record(n_atoms=5000)]  # 20000/5000 = 4x, passes
    assert_claim_supported(
        passing, ClaimClass.END_TO_END, target_n_atoms=target
    )  # must not raise

    tightened = passing + [_end_to_end_record(n_atoms=1000, probe_id="smaller")]
    with pytest.raises(ClaimValidityError):
        # 20000/1000 = 20x > SCALE_EXTRAPOLATION_LIMIT (10x)
        assert_claim_supported(tightened, ClaimClass.END_TO_END, target_n_atoms=target)


# --- select_sources: fail-closed on empty result ----------------------------


def test_select_sources_raises_when_no_record_carries_required_metrics():
    records = [_record(stage=2, metrics={})]
    with pytest.raises(ClaimValidityError):
        select_sources(records, ClaimClass.DISPATCH_COUNT)


def test_select_sources_metric_vs_scope_empty_result_messages_differ():
    with pytest.raises(ClaimValidityError, match="required metrics"):
        select_sources([_record(stage=2, metrics={})], ClaimClass.TERM_RANKING)

    has_metric_no_scopes = _record(
        stage=2, metrics={"total_step_seconds": 1.0}, scopes=None
    )
    with pytest.raises(ClaimValidityError, match="attributed scopes"):
        select_sources([has_metric_no_scopes], ClaimClass.TERM_RANKING)


# --- device_kind: auto-capture -----------------------------------------


def test_device_kind_auto_captured_none_on_cpu_only_stage():
    # No explicit device_kind override, stage<2 so no device_kind is
    # required -- on this (CPU-only test) box, auto-capture yields None.
    rec = ProbeRecord(probe_id="p", stage=0, n_atoms=100, platform="cpu")
    assert rec.device_kind is None


def test_device_kind_explicit_override_still_works():
    rec = _record(stage=2, device_kind="l40s")
    assert rec.device_kind == "l40s"


# --- metrics: float coercion --------------------------------------------


def test_metrics_non_numeric_value_raises():
    with pytest.raises(ClaimValidityError):
        _record(stage=0, metrics={"x": "not-a-number"})


def test_metrics_numeric_string_is_coerced_to_float():
    rec = _record(stage=0, metrics={"x": "1.5"})
    assert rec.metrics["x"] == 1.5
    assert isinstance(rec.metrics["x"], float)


# --- git_sha: unanimity + unverifiable rejection ------------------------


def test_differing_git_sha_raises_on_term_ranking():
    records = [
        _term_ranking_record(stage=2, git_sha="aaaa111", probe_id="a"),
        _term_ranking_record(stage=2, git_sha="bbbb222", probe_id="b"),
    ]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.TERM_RANKING)


def test_unknown_git_sha_raises_on_term_ranking_even_if_unanimous():
    records = [_term_ranking_record(stage=2, git_sha="unknown")]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.TERM_RANKING)


def test_dirty_git_sha_raises_on_end_to_end():
    records = [_end_to_end_record(n_atoms=5000, git_sha="aaaa111-dirty")]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.END_TO_END, allow_no_target=True)


def test_clean_matching_git_sha_permits_term_ranking():
    records = [
        _term_ranking_record(stage=2, git_sha="aaaa111", probe_id="a"),
        _term_ranking_record(stage=2, git_sha="aaaa111", probe_id="b"),
    ]
    assert_claim_supported(records, ClaimClass.TERM_RANKING)  # must not raise


# --- from_json: read-side rules ------------------------------------------


def test_from_json_unknown_field_raises():
    rec = _record(stage=0)
    raw = json.loads(rec.to_json())
    raw["some_future_field"] = "value"
    with pytest.raises(ClaimValidityError):
        ProbeRecord.from_json(json.dumps(raw))


def test_from_json_missing_field_raises():
    rec = _record(stage=0)
    raw = json.loads(rec.to_json())
    del raw["jaxlib_version"]
    with pytest.raises(ClaimValidityError):
        ProbeRecord.from_json(json.dumps(raw))


def test_from_json_major_version_mismatch_raises():
    rec = _record(stage=0)
    raw = json.loads(rec.to_json())
    assert CONTRACT_VERSION.split(".")[0] != "999"
    raw["contract_version"] = "999.0"
    with pytest.raises(ClaimValidityError):
        ProbeRecord.from_json(json.dumps(raw))


def test_from_json_matching_major_version_round_trips():
    rec = _record(stage=0)
    restored = ProbeRecord.from_json(rec.to_json())
    assert restored.probe_id == rec.probe_id


# --- attribution_method: present, optional -------------------------------


def test_attribution_method_defaults_to_none():
    rec = _record(stage=0)
    assert rec.attribution_method is None


def test_attribution_method_round_trips():
    rec = ProbeRecord(
        probe_id="p",
        stage=0,
        n_atoms=100,
        platform="cpu",
        attribution_method={"bonded": "named_scope"},
    )
    restored = ProbeRecord.from_json(rec.to_json())
    assert restored.attribution_method == {"bonded": "named_scope"}


def test_attribution_method_field_exists_on_dataclass():
    field_names = {f.name for f in dataclasses.fields(ProbeRecord)}
    assert "attribution_method" in field_names


def test_attribution_method_invalid_vocabulary_raises():
    with pytest.raises(ClaimValidityError):
        ProbeRecord(
            probe_id="p",
            stage=0,
            n_atoms=100,
            platform="cpu",
            attribution_method={"bonded": "trace"},
        )


def test_attribution_method_required_when_scopes_not_none():
    with pytest.raises(ClaimValidityError):
        ProbeRecord(
            probe_id="p",
            stage=0,
            n_atoms=100,
            platform="cpu",
            scopes={"bonded": (0.1, 10)},
            attribution_method=None,
        )


# --- ProbeRecord: frozen -------------------------------------------------


def test_probe_record_is_frozen():
    rec = _record(stage=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.stage = 2


# --- metrics: non-finite values rejected ---------------------------------


def test_metrics_nan_raises():
    with pytest.raises(ClaimValidityError):
        _record(stage=0, metrics={"x": float("nan")})


def test_metrics_inf_raises():
    with pytest.raises(ClaimValidityError):
        _record(stage=0, metrics={"x": float("inf")})


# --- device_kind: normalization (not an allow-list) ----------------------


def test_normalize_device_kind_strips_vendor_and_lowercases():
    assert _normalize_device_kind("NVIDIA H200") == "h200"


def test_normalize_device_kind_collapses_whitespace():
    assert (
        _normalize_device_kind("NVIDIA RTX PRO 6000 Blackwell")
        == "rtx_pro_6000_blackwell"
    )


def test_normalize_device_kind_does_not_collide_gh200_with_h200():
    # A substring allow-list would collapse both to "h200"; the spec's
    # actual normalization (vendor-strip + lowercase) keeps them distinct.
    assert _normalize_device_kind("NVIDIA H200") != _normalize_device_kind(
        "NVIDIA GH200 480GB"
    )


# --- git_sha: unverified dirty-check sentinel ----------------------------


def test_unverified_git_sha_raises_on_term_ranking():
    records = [_term_ranking_record(stage=2, git_sha="aaaa111-unverified")]
    with pytest.raises(ClaimValidityError):
        assert_claim_supported(records, ClaimClass.TERM_RANKING)


# --- from_json: malformed input is always ClaimValidityError ------------


def test_from_json_empty_string_raises_claim_validity_error():
    with pytest.raises(ClaimValidityError):
        ProbeRecord.from_json("")


def test_from_json_non_dict_json_raises_claim_validity_error():
    for bad in ("[]", "null", "3", '"x"'):
        with pytest.raises(ClaimValidityError):
            ProbeRecord.from_json(bad)


def test_from_json_wrong_typed_field_raises_claim_validity_error():
    rec = _record(stage=0)
    raw = json.loads(rec.to_json())
    raw["n_atoms"] = "500"
    with pytest.raises(ClaimValidityError):
        ProbeRecord.from_json(json.dumps(raw))


def test_from_json_non_dict_scopes_raises_claim_validity_error():
    rec = _record(stage=0)
    raw = json.loads(rec.to_json())
    raw["scopes"] = "not-a-dict"
    with pytest.raises(ClaimValidityError):
        ProbeRecord.from_json(json.dumps(raw))


# --- from_json: rule ordering + minor-bump-field readability -------------


def test_from_json_missing_plain_default_field_does_not_raise():
    # attribution_method has a plain default (None), not a default_factory
    # -- a record predating it is a MINOR-bump case and must stay readable.
    rec = _record(stage=0)
    raw = json.loads(rec.to_json())
    del raw["attribution_method"]
    restored = ProbeRecord.from_json(json.dumps(raw))
    assert restored.attribution_method is None


def test_from_json_version_mismatch_diagnosed_even_with_missing_field():
    # A record skewed on BOTH axes (old major version AND missing a
    # default_factory field) must be diagnosed as version skew, not
    # misdiagnosed as field-level corruption -- the version check runs
    # first.
    rec = _record(stage=0)
    raw = json.loads(rec.to_json())
    del raw["jaxlib_version"]
    raw["contract_version"] = "999.0"
    with pytest.raises(ClaimValidityError, match="major version"):
        ProbeRecord.from_json(json.dumps(raw))


# --- select_sources: contract_version MAJOR unanimity (read-side rule iii) --


def test_select_sources_raises_on_mixed_contract_version_major():
    a = _term_ranking_record(stage=2, probe_id="a")
    b = _term_ranking_record(stage=2, probe_id="b")
    b = dataclasses.replace(b, contract_version="1.0")
    with pytest.raises(ClaimValidityError):
        select_sources([a, b], ClaimClass.TERM_RANKING)


# --- paired_configs -------------------------------------------------------


def test_paired_configs_returns_expected_pairs_on_well_formed_set():
    records = [
        _end_to_end_record(
            n_atoms=5000, config={"grid_spacing": "0.1", "mode": "dense"}, probe_id="a"
        ),
        _end_to_end_record(
            n_atoms=5000, config={"grid_spacing": "0.1", "mode": "flash"}, probe_id="b"
        ),
    ]
    pairs = paired_configs(
        records, ClaimClass.END_TO_END, axis="mode", hold_fixed=("grid_spacing",)
    )
    assert len(pairs) == 1
    assert {pairs[0][0].probe_id, pairs[0][1].probe_id} == {"a", "b"}


def test_paired_configs_raises_on_source_missing_hold_fixed_key():
    records = [
        _end_to_end_record(
            n_atoms=5000, config={"grid_spacing": "0.1", "mode": "dense"}, probe_id="a"
        ),
        _end_to_end_record(n_atoms=5000, config={"mode": "flash"}, probe_id="b"),
    ]
    with pytest.raises(ClaimValidityError):
        paired_configs(
            records, ClaimClass.END_TO_END, axis="mode", hold_fixed=("grid_spacing",)
        )


def test_paired_configs_returns_no_pair_for_single_axis_value():
    records = [
        _end_to_end_record(
            n_atoms=5000, config={"grid_spacing": "0.1", "mode": "dense"}, probe_id="a"
        ),
        _end_to_end_record(
            n_atoms=5000, config={"grid_spacing": "0.1", "mode": "dense"}, probe_id="b"
        ),
    ]
    pairs = paired_configs(
        records, ClaimClass.END_TO_END, axis="mode", hold_fixed=("grid_spacing",)
    )
    assert pairs == []
