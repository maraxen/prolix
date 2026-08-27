"""Parser ground-truth check for scripts/profiling/trace.py.

See .praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section
P4. Two legs, per the spec: (1) synthetic -- a hand-constructed input with
known nesting and known durations, asserting exact recovery; (2) real -- a
small (<1 MB) committed *.trace.json.gz fixture, produced locally on CPU,
asserting the parser recovers real labels with real non-zero exclusive time.

This gates trace.py itself, not any probe's output -- per the project rule
on verifying a measurement pipeline before trusting research conclusions.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

from scripts.profiling.trace import (
    parse_dispatch_counts,
    parse_scopes,
    scope_map_from_hlo_text,
)

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _ROOT / "tests" / "profiling" / "fixtures" / "b1_water_stage1.trace.json.gz"


# --- synthetic leg: parse_scopes -------------------------------------------


def test_parse_scopes_synthetic_known_nesting_and_durations():
    """One parent scope, two child scopes, one child entered 3x.

    Durations are in Perfetto's native microseconds; parse_scopes converts
    to seconds. "end: <name>" bookkeeping events, non-"X"-phase events, and
    events whose hlo_op has no scope mapping must all be excluded.
    """
    scope_map = {
        "outer_thunk": "outer",
        "inner_a_thunk": "inner_a",
        "inner_b_thunk": "inner_b",
    }
    trace_events = [
        {"ph": "X", "name": "outer_thunk", "dur": 100.0, "args": {"hlo_op": "outer_thunk"}},
        {"ph": "X", "name": "inner_a_thunk", "dur": 50.0, "args": {"hlo_op": "inner_a_thunk"}},
        {"ph": "X", "name": "inner_a_thunk", "dur": 30.0, "args": {"hlo_op": "inner_a_thunk"}},
        {"ph": "X", "name": "inner_a_thunk", "dur": 20.0, "args": {"hlo_op": "inner_a_thunk"}},
        {"ph": "X", "name": "inner_b_thunk", "dur": 40.0, "args": {"hlo_op": "inner_b_thunk"}},
        # bookkeeping completion marker -- must not count as real work
        {"ph": "X", "name": "end: inner_a_thunk", "dur": 1.0, "args": {"hlo_op": "inner_a_thunk"}},
        # unmapped hlo_op -- must be ignored, not raise
        {"ph": "X", "name": "unrelated_thunk", "dur": 999.0, "args": {"hlo_op": "unrelated_thunk"}},
        # non-"X" phase -- must be ignored
        {"ph": "M", "name": "outer_thunk", "args": {"hlo_op": "outer_thunk"}},
    ]

    result = parse_scopes(trace_events, scope_map)

    assert result["outer"] == (100.0 / 1e6, 1)
    assert result["inner_a"] == (100.0 / 1e6, 3)
    assert result["inner_b"] == (40.0 / 1e6, 1)
    assert "unrelated_thunk" not in result


def test_parse_scopes_empty_when_no_events_match():
    result = parse_scopes(
        [{"ph": "X", "name": "x", "dur": 5.0, "args": {"hlo_op": "x"}}],
        scope_map={"x": None},
    )
    assert result == {}


def test_parse_hlo_op_times_includes_unmapped_thunks():
    """Op-name dump keeps thunks that parse_scopes drops (no known_label map)."""
    from scripts.profiling.trace import parse_hlo_op_times

    events = [
        {"ph": "X", "name": "cb", "dur": 900.0, "args": {"hlo_op": "command_buffer_14"}},
        {"ph": "X", "name": "fft", "dur": 7.0, "args": {"hlo_op": "fft.0.0"}},
        {"ph": "X", "name": "end: cb", "dur": 1.0, "args": {"hlo_op": "command_buffer_14"}},
        {"ph": "X", "name": "no_hlo", "dur": 50.0},
    ]
    result = parse_hlo_op_times(events)
    assert result["command_buffer_14"] == (900.0 / 1e6, 1)
    assert result["fft.0.0"] == (7.0 / 1e6, 1)
    assert "no_hlo" not in result


# --- synthetic leg: parse_dispatch_counts ----------------------------------


def test_parse_dispatch_counts_synthetic():
    trace_events = [
        {"ph": "X", "name": "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"},
        {"ph": "X", "name": "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"},
        {"ph": "X", "name": "backend_compile_and_load"},
        {"ph": "X", "name": "PjitFunction(f)"},
        {"ph": "X", "name": "PjitFunction(f)"},
        {"ph": "X", "name": "PjitFunction(other_fn)"},
        # non-"X" phase -- must be ignored
        {"ph": "M", "name": "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"},
    ]

    counts_f_only = parse_dispatch_counts(trace_events, fn_name="f")
    assert counts_f_only == {"n_executions": 2, "n_compilations": 1, "n_jit_traces": 2}

    counts_all = parse_dispatch_counts(trace_events)
    assert counts_all["n_jit_traces"] == 3


# --- synthetic leg: scope_map_from_hlo_text --------------------------------


def test_scope_map_from_hlo_text_direct_and_indirect_resolution():
    """Direct op_name metadata, and resolution through calls=/ROOT chains."""
    hlo_text = """
%callee_computation (p: f32[]) -> f32[] {
  ROOT %inner_op = f32[] add(%p, %p), metadata={op_name="jit(f)/outer/inner_b/reduce_sum"}
}

ENTRY %main.1 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %direct_op = f32[] multiply(%x, %x), metadata={op_name="jit(f)/outer/inner_a/integer_pow"}
  ROOT %indirect_op = f32[] fusion(%x), kind=kLoop, calls=%callee_computation
}
"""
    known = frozenset({"outer", "inner_a", "inner_b"})
    scope_map = scope_map_from_hlo_text(hlo_text, known)

    assert scope_map["direct_op"] == "inner_a"
    assert scope_map["indirect_op"] == "inner_b"
    assert scope_map["x"] is None


def test_scope_map_from_hlo_text_unwraps_vmap_jvp_transform_wrapper():
    """Confirmed on a real vmapped+autodiffed program: op_name segments are
    decorated as "vmap(jvp(<label>))", not the bare label."""
    hlo_text = """
ENTRY %main.1 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %op = f32[] multiply(%x, %x), metadata={op_name="jit(_batched)/vmap(jvp(dense_bonded_improper))/reduce_sum"}
}
"""
    known = frozenset({"dense_bonded_improper"})
    scope_map = scope_map_from_hlo_text(hlo_text, known)
    assert scope_map["op"] == "dense_bonded_improper"


def test_scope_map_from_hlo_text_deepest_match_wins():
    hlo_text = """
ENTRY %main.1 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %op = f32[] multiply(%x, %x), metadata={op_name="jit(f)/outer/inner/leaf"}
}
"""
    known = frozenset({"outer", "inner"})
    scope_map = scope_map_from_hlo_text(hlo_text, known)
    assert scope_map["op"] == "inner"


# --- real leg: committed CPU fixture ----------------------------------------


def _load_fixture_events() -> list[dict]:
    with gzip.open(_FIXTURE, "rt") as fh:
        data = json.load(fh)
    return data["traceEvents"]


def _build_matching_hlo_text():
    """Regenerate the HLO text matching the committed trace fixture.

    Not committed as a second artifact (see fixtures/README.md) -- reused
    directly from scripts/experiments/profile_b1_water_trace.py's own
    helpers with the SAME --replicas/--n-steps params that produced the
    committed trace, per this project's "reuse the reference script, do not
    re-derive it" rule.
    """
    spec = importlib.util.spec_from_file_location(
        "b1_water_trace_fixture_hlo",
        _ROOT / "scripts" / "experiments" / "profile_b1_water_trace.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    plan, bundles = mod._build_water_plan(2)
    batched_fn, args = mod._build_batched_callable(plan, bundles, 5)
    return batched_fn.lower(*args).compile().as_text()


def test_real_fixture_recovers_labels_with_nonzero_exclusive_time():
    """P4's parser ground-truth check, real leg.

    Stated finding (accepted per user decision 2026-08-17, tracked as
    backlog follow-up #4330): at this fixture's toy scale
    (--replicas 2 --n-steps 5), settle.py's/pme.py's PRE-EXISTING labels
    (settle_pos_eigh, settle_cramers_rule, settle_rattle_loop, pme_*) do
    NOT survive to execution-thunk-level attribution -- they are present in
    the compiled HLO text but get fused into blocks whose top-level
    (execution-visible) thunk carries a different instruction's metadata.
    This is a DISTINCT phenomenon from the already-documented "vanishes
    from HLO text entirely at toy scale" pme_* limitation. P2's own new
    dense_bonded_*/dense_lj_direct/dense_coulomb_direct labels DO survive
    at this scale and satisfy this leg's actual purpose: proving the
    two-input (trace + HLO text) attribution pipeline recovers real,
    non-zero wall-clock time on a real program, not a parser defect masked
    by only ever testing synthetic input (leg 1 alone cannot detect that).
    """
    hlo_text = _build_matching_hlo_text()
    known_labels = frozenset(
        {
            "dense_bonded_angle", "dense_bonded_bond", "dense_bonded_cmap",
            "dense_bonded_dihedral", "dense_bonded_improper",
            "dense_bonded_urey_bradley", "dense_coulomb_direct",
            "dense_lj_direct", "dense_pme_reciprocal",
            "flash_bonded", "flash_grad", "flash_lj_only",
            "flash_nonbonded_tiles", "pme_bwd_gather", "pme_charge_spread",
            "pme_fft_forward", "pme_fft_inverse", "pme_greens_setup",
            "pme_greens_setup_standalone", "settle_cramers_rule",
            "settle_pos_eigh", "settle_rattle_loop",
        }
    )
    scope_map = scope_map_from_hlo_text(hlo_text, known_labels)
    events = _load_fixture_events()
    scopes = parse_scopes(events, scope_map)

    non_zero = {label: v for label, v in scopes.items() if v[0] > 0}
    assert len(non_zero) >= 2, (
        f"expected >=2 labels with non-zero exclusive time, got {non_zero}"
    )
