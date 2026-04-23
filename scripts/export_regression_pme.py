#!/usr/bin/env python3
"""One-way export: ``REGRESSION_EXPLICIT_PME`` in ``prolix/physics/regression_explicit_pme.py`` → docs.

Updates the fenced Python block between sentinel comments in
``docs/source/explicit_solvent/openmm_comparison_protocol.md``.

Usage:
  python scripts/export_regression_pme.py          # rewrite protocol doc
  python scripts/export_regression_pme.py --check  # exit 1 if committed doc drifts
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_PY = REPO_ROOT / "src" / "prolix" / "physics" / "regression_explicit_pme.py"
PROTOCOL_MD = REPO_ROOT / "docs" / "source" / "explicit_solvent" / "openmm_comparison_protocol.md"

BEGIN = "<!-- REGRESSION_PME:BEGIN (auto-generated from src/prolix/physics/regression_explicit_pme.py; do not edit) -->"
END = "<!-- REGRESSION_PME:END -->"


def _literal_dict_from_dict_value(d_node: ast.Dict) -> dict[str, object]:
  out: dict[str, object] = {}
  for key_node, val_node in zip(d_node.keys, d_node.values, strict=True):
    if key_node is None:
      msg = "Dict unpacking is not supported in REGRESSION_EXPLICIT_PME"
      raise ValueError(msg)
    if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
      msg = "REGRESSION_EXPLICIT_PME keys must be string literals"
      raise TypeError(msg)
    if not isinstance(val_node, ast.Constant):
      msg = f"REGRESSION_EXPLICIT_PME[{key_node.value!r}] must be a literal constant"
      raise TypeError(msg)
    out[str(key_node.value)] = val_node.value
  return out


def _load_regression_dict() -> dict[str, object]:
  src = REGRESSION_PY.read_text(encoding="utf-8")
  tree = ast.parse(src)
  for node in tree.body:
    if not isinstance(node, ast.AnnAssign):
      continue
    if not isinstance(node.target, ast.Name) or node.target.id != "REGRESSION_EXPLICIT_PME":
      continue
    if not isinstance(node.value, ast.Dict):
      msg = "REGRESSION_EXPLICIT_PME must be assigned a dict literal"
      raise TypeError(msg)
    return _literal_dict_from_dict_value(node.value)
  msg = f"Could not find REGRESSION_EXPLICIT_PME dict literal in {REGRESSION_PY}"
  raise ValueError(msg)


def _format_python_block(d: dict[str, object]) -> str:
  items = list(d.items())
  lines = ["REGRESSION_EXPLICIT_PME = {"]
  for i, (key, val) in enumerate(items):
    comma = "," if i < len(items) - 1 else ""
    if isinstance(val, str):
      lines.append(f'  "{key}": "{val}"{comma}')
    elif isinstance(val, bool):
      lines.append(f'  "{key}": {val}{comma}')
    elif isinstance(val, (int, float)):
      lines.append(f'  "{key}": {val!r}{comma}')
    else:
      msg = f"Unsupported value type for {key!r}: {type(val)}"
      raise TypeError(msg)
  lines.append("}")
  return "\n".join(lines)


def _replace_sentinel_block(text: str, python_src: str) -> str:
  if BEGIN not in text or END not in text:
    msg = f"Missing {BEGIN!r} or {END!r} in {PROTOCOL_MD}"
    raise ValueError(msg)
  i0 = text.index(BEGIN)
  i1 = text.index(END) + len(END)
  new_seg = f"{BEGIN}\n\n```python\n{python_src}\n```\n\n{END}"
  return text[:i0] + new_seg + text[i1:]


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--check",
    action="store_true",
    help="Verify the protocol doc matches regression_explicit_pme.py without writing.",
  )
  args = parser.parse_args(argv)

  try:
    block = _format_python_block(_load_regression_dict())
  except Exception as e:  # noqa: BLE001 — CLI entrypoint
    print(f"export_regression_pme: failed to load regression dict: {e}", file=sys.stderr)
    return 2

  text = PROTOCOL_MD.read_text(encoding="utf-8")
  try:
    updated = _replace_sentinel_block(text, block)
  except ValueError as e:
    print(f"export_regression_pme: {e}", file=sys.stderr)
    return 2

  if args.check:
    if updated != text:
      print(
        "export_regression_pme: docs are stale — run `python scripts/export_regression_pme.py` "
        f"and commit changes to {PROTOCOL_MD.relative_to(REPO_ROOT)}",
        file=sys.stderr,
      )
      return 1
    return 0

  PROTOCOL_MD.write_text(updated, encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
