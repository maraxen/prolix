"""XR-SHADOW (#3276): fitting↔MD plan equality via shared xtrax path.

Both sides plan through the xtrax adapter. Production ``make_fitting_planner``
also uses ``plan_with_xtrax`` (XR-FIT-FLIP).
"""

from __future__ import annotations

import dataclasses

from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.iterator import SafeMapIterator, VmapIterator
from xtrax.tiling.strategy import SafeMap, Vmap

from prolix.api.ensemble_dispatch import n_mols_strategy
from prolix.tiling.axes import N_ATOMS, N_MOLS
from prolix.tiling.planner import (
    AxisDecision,
    BatchPlanner,
    estimate_memory_theoretical,
)
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax

# Shared fixture axes (homogeneous) — intersection of fitting/MD planning concerns.
SHADOW_AXES = [
    dataclasses.replace(
        N_ATOMS,
        cardinality=64,
        default_batch_size=8,
        tile_granularity=8,
        heterogeneous=False,
    ),
    dataclasses.replace(
        N_MOLS,
        cardinality=16,
        default_batch_size=4,
        tile_granularity=4,
        heterogeneous=False,
    ),
]

_BASE_SHAPE_BYTES = 10.0
_ACTIVATION = 1.0
# Both Vmap: 64*16*10 = 10240; demote atoms: 8*16*10 = 1280; demote both: 8*4*10 = 320
_LOOSE_BUDGET = 1 << 40
_TIGHT_BUDGET = 500  # forces demotion of both candidate axes


def _estimate(decisions: list[AxisDecision]) -> float:
    return estimate_memory_theoretical(decisions, _BASE_SHAPE_BYTES, _ACTIVATION)


def _plan_fitting(budget_bytes: int):
    """Fitting-side plan via shared xtrax path (not production greedy)."""
    return BatchPlanner(
        axes=list(SHADOW_AXES),
        budget_bytes=budget_bytes,
        estimate_memory=_estimate,
    ).plan_with_xtrax()


def _plan_md(budget_bytes: int):
    """MD-side plan via the same adapter EnsembleMDPlanner uses."""
    return plan_axes_with_xtrax(
        axes=list(SHADOW_AXES),
        budget_bytes=budget_bytes,
        estimate_memory=_estimate,
    )


def _strategy_from_decision(decision: AxisDecision) -> Vmap | SafeMap:
    """Recover xtrax strategy object from prolix int collapse (AC4)."""
    batch_size = decision.batch_size
    if batch_size == 0 or batch_size >= decision.axis.cardinality:
        return Vmap()
    return SafeMap(batch_size=batch_size)


def _assert_plans_equal(fit_plan, md_plan) -> None:
    assert {d.axis.name for d in fit_plan.decisions} == {
        d.axis.name for d in md_plan.decisions
    }
    for name in ("n_atoms", "n_mols"):
        fit_d = fit_plan.decision_for(name)
        md_d = md_plan.decision_for(name)
        assert fit_d.batch_size == md_d.batch_size, name

        fit_s = _strategy_from_decision(fit_d)
        md_s = _strategy_from_decision(md_d)
        # AC4: must compare strategy types, not ints alone.
        assert isinstance(fit_s, (Vmap, SafeMap))
        assert isinstance(md_s, (Vmap, SafeMap))
        assert type(fit_s) is type(md_s), name
        if isinstance(fit_s, SafeMap):
            assert fit_s.batch_size == md_s.batch_size, name


def _assert_n_mols_dispatch(plan) -> None:
    n_systems = plan.decision_for("n_mols").axis.cardinality
    strategy = n_mols_strategy(plan, n_systems)
    assert isinstance(strategy, (Vmap, SafeMap))
    planned = _strategy_from_decision(plan.decision_for("n_mols"))
    assert type(strategy) is type(planned)
    if isinstance(strategy, SafeMap):
        assert strategy.batch_size == planned.batch_size

    iterator = make_axis_dispatch(strategy, axis=N_MOLS.name)
    if isinstance(strategy, Vmap):
        assert isinstance(iterator, VmapIterator)
    else:
        assert isinstance(iterator, SafeMapIterator)
        assert iterator.tile == strategy.batch_size


def test_shadow_loose_budget_vmap_retain_and_dispatch():
    fit = _plan_fitting(_LOOSE_BUDGET)
    md = _plan_md(_LOOSE_BUDGET)
    _assert_plans_equal(fit, md)
    for name in ("n_atoms", "n_mols"):
        assert isinstance(_strategy_from_decision(fit.decision_for(name)), Vmap)
    _assert_n_mols_dispatch(md)


def test_shadow_tight_budget_safemap_demotion_and_dispatch():
    fit = _plan_fitting(_TIGHT_BUDGET)
    md = _plan_md(_TIGHT_BUDGET)
    _assert_plans_equal(fit, md)
    # Both axes are demotion candidates under joint budget.
    assert isinstance(_strategy_from_decision(fit.decision_for("n_atoms")), SafeMap)
    assert isinstance(_strategy_from_decision(fit.decision_for("n_mols")), SafeMap)
    _assert_n_mols_dispatch(md)


def test_shadow_ac4_rejects_int_only_comparison_helper():
    """Kill guard: reconstructing strategy must yield typed objects, not ints."""
    md = _plan_md(_LOOSE_BUDGET)
    for d in md.decisions:
        strategy = _strategy_from_decision(d)
        assert not isinstance(strategy, int)
        assert isinstance(strategy, (Vmap, SafeMap))
