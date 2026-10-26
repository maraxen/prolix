"""Static import-graph audit for prolix: module-level cycles and lazy (function-level) imports.

Usage: python scripts/ci/import_graph.py [--package src/prolix] [--json out.json]

* "top"  edges: imports executed at module import time (module body, incl. class bodies,
  excluding ``if TYPE_CHECKING`` blocks).
* "lazy" edges: imports inside functions/methods (executed at call time) -- these are the
  smells when they exist only to dodge a cycle.
Exit status is 1 if a *top-level* cycle exists (lazy edges never fail the check).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path


def module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve(node: ast.ImportFrom, mod: str, is_pkg: bool) -> str | None:
    if node.level == 0:
        return node.module
    base = mod.split(".")
    if not is_pkg:
        base = base[:-1]
    if node.level > 1:
        base = base[: -(node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


class Visitor(ast.NodeVisitor):
    def __init__(self, mod: str, is_pkg: bool, known: set[str]):
        self.mod, self.is_pkg, self.known = mod, is_pkg, known
        self.depth = 0  # function nesting
        self.type_checking = 0
        self.edges: list[tuple[str, str, int]] = []  # (target, kind, lineno)

    def _add(self, target: str | None, lineno: int, names: list[str] | None = None):
        if not target or not target.startswith("prolix"):
            return
        cands = [target]
        # ``from pkg import submodule`` imports the submodule too
        for n in names or []:
            if f"{target}.{n}" in self.known:
                cands.append(f"{target}.{n}")
        tgt = cands[-1] if len(cands) > 1 else target
        for t in cands[1:] or [target]:
            kind = "lazy" if self.depth else "top"
            self.edges.append((t, kind, lineno))
        del tgt

    def visit_Import(self, node):
        if self.type_checking:
            return
        for a in node.names:
            self._add(a.name, node.lineno)

    def visit_ImportFrom(self, node):
        if self.type_checking:
            return
        self._add(resolve(node, self.mod, self.is_pkg), node.lineno, [a.name for a in node.names])

    def visit_If(self, node):
        t = node.test
        is_tc = (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
            isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
        )
        if is_tc:
            self.type_checking += 1
            for n in node.body:
                self.visit(n)
            self.type_checking -= 1
            for n in node.orelse:
                self.visit(n)
        else:
            self.generic_visit(node)

    def _fn(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = _fn


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    sys.setrecursionlimit(10000)
    index, low, on, stack, out = {}, {}, set(), [], []
    counter = [0]

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in graph.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(sorted(comp))

    for v in list(graph):
        if v not in index:
            strong(v)
    return out


class Graph:
    """Result of :func:`build`: module count, top-level edges, lazy edges, cycles, dodges."""

    def __init__(self, mods, top, lazy, cycles, dodges):
        self.mods, self.top, self.lazy, self.cycles, self.dodges = mods, top, lazy, cycles, dodges


def build(root: Path) -> Graph:
    files = sorted(root.rglob("*.py"))
    mods = {module_name(f, root): f for f in files}
    known = set(mods)
    top: dict[str, set[str]] = defaultdict(set)
    lazy: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for m, f in mods.items():
        tree = ast.parse(f.read_text())
        v = Visitor(m, f.name == "__init__.py", known)
        v.visit(tree)
        for tgt, kind, ln in v.edges:
            # collapse to a known module (longest known prefix)
            t = tgt
            while t and t not in known:
                t = t.rpartition(".")[0]
            if not t or t == m:
                continue
            if kind == "top":
                top[m].add(t)
            else:
                lazy[m].append((t, ln))
    cycles = find_cycles(top)

    def reaches(a: str, b: str) -> bool:
        seen, todo = set(), [a]
        while todo:
            x = todo.pop()
            if x == b:
                return True
            if x in seen:
                continue
            seen.add(x)
            todo.extend(top.get(x, ()))
        return False

    dodges = [(m, t, ln) for m, lst in lazy.items() for (t, ln) in lst if reaches(t, m)]
    return Graph(mods, top, lazy, cycles, dodges)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="src/prolix")
    ap.add_argument("--json")
    args = ap.parse_args()
    g = build(Path(args.package))
    print(f"modules: {len(g.mods)}  top-level cycles (SCCs): {len(g.cycles)}")
    for c in g.cycles:
        print("  CYCLE:", " <-> ".join(c))
    nl = sum(len(v) for v in g.lazy.values())
    print(f"lazy intra-package imports: {nl}  (cycle-dodging: {len(g.dodges)})")
    for m, t, ln in sorted(g.dodges):
        print(f"  DODGE {m}:{ln} -> {t}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"cycles": g.cycles, "lazy": dict(g.lazy), "dodges": g.dodges}, indent=2
            )
        )
    return 1 if g.cycles else 0


if __name__ == "__main__":
    raise SystemExit(main())
