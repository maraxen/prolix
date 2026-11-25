"""v3.8 Track 4 tests: zero lazy physics imports (`ALLOWED_PHYSICS_LAZY == set()`),
transition-free Sphinx API Markdown pages, and `lj_pme` / `fltflt_pallas` API docstrings.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
SRC_PROLIX = WORKTREE_ROOT / "src" / "prolix"
IMPORT_GRAPH_SCRIPT = WORKTREE_ROOT / "scripts" / "ci" / "import_graph.py"
ARCH_TEST_FILE = WORKTREE_ROOT / "tests" / "test_import_architecture.py"
API_DOCS_DIR = WORKTREE_ROOT / "docs" / "source" / "api"


def _load_module_from_path(mod_name: str, file_path: Path):
  spec = importlib.util.spec_from_file_location(mod_name, file_path)
  assert spec is not None
  assert spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_allowed_physics_lazy_is_empty_and_ast_graph_has_at_most_two_lazy_imports() -> None:
  """Verify `ALLOWED_PHYSICS_LAZY == set()`, 0 AST cycles, and <= 2 total lazy imports."""
  arch_mod = _load_module_from_path("_v38_test_import_arch", ARCH_TEST_FILE)
  assert arch_mod.ALLOWED_PHYSICS_LAZY == set()

  ig_mod = _load_module_from_path("_v38_import_graph", IMPORT_GRAPH_SCRIPT)
  graph = ig_mod.build(SRC_PROLIX)
  assert graph.cycles == [], f"Unexpected top-level import cycles: {graph.cycles}"

  all_lazy = [(mod, target, lineno) for mod, items in graph.lazy.items() for target, lineno in items]
  assert len(all_lazy) <= 2, f"Expected <= 2 lazy intra-package imports, got {len(all_lazy)}: {all_lazy}"
  assert "prolix.physics.flash_explicit" not in graph.lazy
  assert "prolix.physics.pme" not in graph.lazy


def test_runtime_imports_have_zero_circular_dependency() -> None:
  """Verify top-level runtime imports of core subpackages succeed without circular import errors."""
  for mod_name in (
    "prolix",
    "prolix.physics",
    "prolix.physics.flash_explicit",
    "prolix.physics.pme",
    "prolix.physics.lj_pme",
    "prolix.physics.fltflt_pallas",
    "prolix.openmm",
    "prolix.api",
  ):
    mod = importlib.import_module(mod_name)
    assert mod is not None


def test_api_markdown_pages_have_zero_invalid_transitions_and_document_v38_surfaces() -> None:
  """Verify `docs/source/api/*.md` contain zero `---` transitions and cover `lj_pme` / `fltflt_pallas` / `ProlixBatchSimulation`."""
  expected_pages = ("api.md", "export.md", "knob_surface.md", "openmm.md", "physics.md")
  transition_re = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)

  for page_name in expected_pages:
    page_path = API_DOCS_DIR / page_name
    assert page_path.is_file(), f"Missing API doc page: {page_path}"
    text = page_path.read_text(encoding="utf-8")
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
      if line.strip().startswith("```"):
        in_fence = not in_fence
        continue
      if not in_fence:
        assert not transition_re.match(line), (
          f"{page_name}:{lineno} contains docutils horizontal transition {line!r} "
          "which triggers `Document or section may not begin with a transition`"
        )

  physics_md = (API_DOCS_DIR / "physics.md").read_text(encoding="utf-8")
  assert "prolix.physics.lj_pme" in physics_md
  assert "lj_pme_total_dispersion_energy" in physics_md
  assert ".. automodule:: prolix.physics.fltflt_pallas" in physics_md

  openmm_md = (API_DOCS_DIR / "openmm.md").read_text(encoding="utf-8")
  assert ".. autoclass:: prolix.openmm.ProlixBatchSimulation" in openmm_md
  for method_ref in (
    "setParameter",
    "getParameters",
    "value_and_grad_positions",
    "compute_parameter_derivatives",
    "minimizeEnergy",
    "createCheckpoint",
  ):
    assert method_ref in openmm_md


def test_lj_pme_and_fltflt_pallas_importable_with_public_docstrings() -> None:
  """Verify `prolix.physics.lj_pme` and `prolix.physics.fltflt_pallas` expose documented public APIs."""
  lj_pme = importlib.import_module("prolix.physics.lj_pme")
  fltflt = importlib.import_module("prolix.physics.fltflt_pallas")

  assert inspect.getdoc(lj_pme), "prolix.physics.lj_pme missing module docstring"
  assert inspect.getdoc(fltflt), "prolix.physics.fltflt_pallas missing module docstring"

  for fn_name in (
    "dispersion_charges",
    "resolve_ljpme_parameters",
    "spread_dispersion_charges",
    "lj_pme_influence_function",
    "lj_pme_reciprocal_energy",
    "lj_pme_self_energy",
    "lj_pme_exclusion_energy",
    "lj_pme_direct_pair_energy",
    "lj_pme_direct_energy_neighbor_list",
    "lj_pme_total_dispersion_energy",
  ):
    fn = getattr(lj_pme, fn_name)
    assert callable(fn)
    assert inspect.getdoc(fn), f"prolix.physics.lj_pme.{fn_name} missing docstring"

  assert hasattr(fltflt, "FltFlt")
  assert inspect.getdoc(fltflt.FltFlt), "FltFlt missing class docstring"
  for fn_name in (
    "fltflt_triton_asm_context",
    "should_use_triton_asm",
    "two_sum",
    "fast_two_sum",
    "two_prod_fma",
    "ff_add",
    "ff_sub",
    "ff_mul",
    "ff_div",
    "ff_sqrt",
    "ff_norm3d",
    "ff_min_image_f32",
  ):
    fn = getattr(fltflt, fn_name)
    assert callable(fn)
    assert inspect.getdoc(fn), f"prolix.physics.fltflt_pallas.{fn_name} missing docstring"
