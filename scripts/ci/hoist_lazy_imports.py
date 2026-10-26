"""Hoist function-level ``prolix.*`` imports to module level (Phase 1.3).

Usage: python scripts/ci/hoist_lazy_imports.py FILE [FILE ...] [--keep MOD ...] [--dry-run]

Moves every function-local ``import prolix...`` / ``from prolix... import ...`` statement to
the module import block, except modules in ``--keep`` (optional/heavy deps that must stay
lazy). Run ``scripts/ci/import_graph.py`` afterwards: it must still report zero top-level
cycles. Import sorting is left to ``ruff check --select I --fix``.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def function_local_imports(tree: ast.Module):
    out = []

    class V(ast.NodeVisitor):
        depth = 0
        tc = 0

        def visit_If(self, node):
            t = node.test
            if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
                isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
            ):
                self.tc += 1
                for n in node.body:
                    self.visit(n)
                self.tc -= 1
                for n in node.orelse:
                    self.visit(n)
            else:
                self.generic_visit(node)

        def _fn(self, node):
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_FunctionDef = visit_AsyncFunctionDef = _fn

        def visit_ImportFrom(self, node):
            if self.depth and not self.tc and node.level == 0 and (node.module or "").startswith("prolix"):
                out.append(node)

        def visit_Import(self, node):
            if self.depth and not self.tc and any(a.name.startswith("prolix") for a in node.names):
                out.append(node)

    V().visit(tree)
    return out


def last_toplevel_import_end(tree: ast.Module) -> int:
    end = 0
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            end = n.end_lineno
        elif isinstance(n, ast.If):  # TYPE_CHECKING block etc.
            continue
        elif end and not (
            isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)
        ):
            # first non-import statement after the import block
            break
    return end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for f in args.files:
        p = Path(f)
        src = p.read_text()
        tree = ast.parse(src)
        nodes = [
            n
            for n in function_local_imports(tree)
            if not any(
                (isinstance(n, ast.ImportFrom) and (n.module or "").startswith(k))
                or (isinstance(n, ast.Import) and any(a.name.startswith(k) for a in n.names))
                for k in args.keep
            )
        ]
        if not nodes:
            print(f"{f}: nothing to hoist")
            continue
        lines = src.split("\n")
        stmts: list[str] = []
        for n in nodes:
            seg = "\n".join(l.lstrip() if i else l.strip() for i, l in enumerate(lines[n.lineno - 1 : n.end_lineno]))
            # normalise multi-line parenthesised imports back to ast.unparse form
            stmts.append(ast.unparse(n))
            _ = seg
        # remove from bottom to top so line numbers stay valid
        for n in sorted(nodes, key=lambda x: x.lineno, reverse=True):
            start, end = n.lineno - 1, n.end_lineno
            del lines[start:end]
            # drop one now-redundant blank line directly after the removed statement
            if start < len(lines) and lines[start].strip() == "" and start > 0 and lines[start - 1].strip() == "":
                del lines[start]
        tree2 = ast.parse("\n".join(lines))
        at = last_toplevel_import_end(tree2)
        have = {ast.unparse(n) for n in tree2.body if isinstance(n, (ast.Import, ast.ImportFrom))}
        new = []
        for s in stmts:
            if s not in have and s not in new:
                new.append(s)
        lines[at:at] = new
        print(f"{f}: hoisting {len(stmts)} import(s) ({len(new)} new lines)")
        if not args.dry_run:
            p.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
