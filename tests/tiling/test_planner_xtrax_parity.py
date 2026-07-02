"""Strict parity gate: prolix greedy ``plan()`` vs xtrax-delegated ``plan_with_xtrax()``.

Production strategy selection (``prolix.run.spec.make_fitting_planner``) delegates to
``BatchPlanner.plan_with_xtrax()`` as of the xtrax 0.3.0 cutover (Path B). This gate
guards that flip: for every AxisSpec configuration prolix actually plans over, the
xtrax-backed planner must produce a BatchPlan *identical* to the battle-tested greedy
loop — same per-axis batch_size, same budget-exceeded verdict.

If xtrax's rule ordering ever diverges from prolix's greedy innermost-first demotion,
this test fails loudly rather than silently changing which axes vmap vs safe_map.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

from prolix.tiling.axes import (
    N_ANGLES,
    N_ATOMS,
    N_BONDS,
    N_CONFORMERS,
    N_MOLS,
    N_TORSIONS,
)
from prolix.tiling.planner import (
    AxisDecision,
    AxisSpec,
    BatchPlanner,
    estimate_memory_theoretical,
)


def _estimate(decisions: list[AxisDecision]) -> float:
    # Same shape/activation multiplier convention as run.spec.make_fitting_planner.
    return estimate_memory_theoretical(decisions, 1.0, 1.0)


# (label, axes, budget_bytes) — spans homogeneous-within-budget, tight-demotion,
# mixed heterogeneous+homogeneous, the all-heterogeneous production fixture, and a
# 4-homogeneous-axis budget sweep. All exercise xtrax's BatchPlanner decision path.
_SCENARIOS: list[tuple[str, list[AxisSpec], float]] = [
    (
        "homogeneous_generous_all_vmap",
        [dc.replace(N_BONDS, cardinality=8), dc.replace(N_ANGLES, cardinality=8)],
        1e12,
    ),
    (
        "homogeneous_tight_demote_innermost",
        [dc.replace(N_ATOMS, cardinality=100), dc.replace(N_BONDS, cardinality=50)],
        500.0,
    ),
    (
        "homogeneous_very_tight",
        [dc.replace(N_ATOMS, cardinality=100), dc.replace(N_BONDS, cardinality=50)],
        10.0,
    ),
    (
        "mixed_het_hom_generous",
        [dc.replace(N_MOLS, cardinality=16), dc.replace(N_ATOMS, cardinality=32)],
        1e12,
    ),
    (
        "mixed_het_hom_tight",
        [dc.replace(N_MOLS, cardinality=16), dc.replace(N_ATOMS, cardinality=1000)],
        100.0,
    ),
    (
        "production_fixture_both_heterogeneous",
        [dc.replace(N_MOLS, cardinality=64), dc.replace(N_CONFORMERS, cardinality=2048)],
        1e9,
    ),
    (
        "four_homogeneous_mid_budget",
        [
            dc.replace(N_ATOMS, cardinality=50),
            dc.replace(N_BONDS, cardinality=40),
            dc.replace(N_ANGLES, cardinality=30),
            dc.replace(N_TORSIONS, cardinality=20),
        ],
        20_000.0,
    ),
]


@pytest.mark.parametrize("label,axes,budget", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_greedy_and_xtrax_plans_are_identical(
    label: str, axes: list[AxisSpec], budget: float
) -> None:
    greedy = BatchPlanner(axes=axes, budget_bytes=budget, estimate_memory=_estimate).plan()
    delegated = BatchPlanner(
        axes=axes, budget_bytes=budget, estimate_memory=_estimate
    ).plan_with_xtrax()

    greedy_sizes = {d.axis.name: d.batch_size for d in greedy.decisions}
    delegated_sizes = {d.axis.name: d.batch_size for d in delegated.decisions}

    assert delegated_sizes == greedy_sizes, (
        f"[{label}] xtrax-delegated batch sizes diverge from greedy: "
        f"greedy={greedy_sizes} xtrax={delegated_sizes}"
    )
    assert delegated.budget_exceeded == greedy.budget_exceeded, (
        f"[{label}] budget_exceeded verdict diverges: "
        f"greedy={greedy.budget_exceeded} xtrax={delegated.budget_exceeded}"
    )
