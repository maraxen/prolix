"""Architecture guardrails for the prolix import graph (Phase 1 hardening).

Backed by ``scripts/ci/import_graph.py`` (pure AST; needs no JAX). Fails when:

1. a module-level import cycle appears among ``prolix.*`` modules;
2. a *new* function-level import appears whose only purpose is to dodge a cycle (the
   target reaches the importer through module-level imports);
3. a new function-level ``prolix.*`` import appears anywhere outside the documented
   allowlist (optional, heavy Pallas backends and the types->physics late-binding pair).

To add an exception, extend the allowlist below *with a reason*; the intent is that the
lists only ever shrink.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "ci" / "import_graph.py"

# (importer, target) lazy edges that dodge a cycle and are accepted, with reasons.
ALLOWED_CYCLE_DODGES = {
    # MolecularBundle (types layer) convenience constructors forward to the higher-level
    # physics factory; late binding is the correct layering, not a smell.
    ("prolix.types.bundles", "prolix.physics.system"),
}

# (importer, target) lazy edges allowed inside prolix.physics: optional Pallas backends
# whose import pulls ``jax.experimental.pallas`` and must stay off the default import path.
ALLOWED_PHYSICS_LAZY = {
    ("prolix.physics.flash_explicit", "prolix.pallas_kernels"),
    ("prolix.physics.pme", "prolix.pallas_kernels"),
    ("prolix.physics.pme", "prolix.physics.pme_pallas"),
}


def _load_graph():
    spec = importlib.util.spec_from_file_location("prolix_import_graph", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.build(_ROOT / "src" / "prolix")


@pytest.fixture(scope="module")
def graph():
    return _load_graph()


def test_no_module_level_import_cycles(graph):
    assert graph.cycles == [], f"module-level import cycle(s): {graph.cycles}"


def test_no_new_cycle_dodging_lazy_imports(graph):
    found = {(m, t) for m, t, _ln in graph.dodges}
    new = sorted(found - ALLOWED_CYCLE_DODGES)
    assert not new, (
        "new function-level imports that only exist to dodge a cycle; extract the shared "
        f"piece into a leaf module instead: {new}"
    )


def test_no_unreviewed_lazy_imports_repo_wide(graph):
    """After Phase 1 every function-level ``prolix.*`` import is reviewed and allowlisted."""
    found = {(m, t) for m, lst in graph.lazy.items() for t, _ln in lst}
    allowed = ALLOWED_PHYSICS_LAZY | ALLOWED_CYCLE_DODGES
    new = sorted(found - allowed)
    assert not new, (
        "unreviewed lazy intra-package imports (hoist them to module level, or extract a "
        f"leaf module if they dodge a cycle): {new}"
    )


def test_leaf_modules_stay_leaves(graph):
    """Leaf modules extracted in Phase 1.2 must not grow upward dependencies."""
    leaves = {
        # Flat core-types module: every other layer may depend on it, it depends on none.
        "prolix.typing": set(),
        "prolix.physics.water_models": set(),
        "prolix.physics.electrostatic_methods": set(),
        "prolix.physics.constraint_projection": set(),
        "prolix.physics.step_base": {"prolix.typing"},
        "prolix.tiling.plan_types": set(),
    }
    for mod, allowed in leaves.items():
        deps = {d for d in graph.top.get(mod, set()) if d != "prolix"}
        assert deps <= allowed, f"{mod} imports {sorted(deps - allowed)} (allowed: {sorted(allowed)})"
