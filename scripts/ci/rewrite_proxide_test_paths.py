"""One-shot rewrite: sibling-proxide asset paths in tests -> tests._proxide_assets helpers."""

import re
import sys
from pathlib import Path

ASSET = re.compile(
    r"\(\s*Path\(__file__\)(?:\s*\.parent)+\s*/\s*\"proxide\"\s*/\s*\"src\"\s*/\s*\"proxide\""
    r"\s*/\s*\"assets\"((?:\s*/\s*\"[^\"]+\")+)\s*\)",
    re.S,
)
ASSET_BARE = re.compile(
    r"Path\(__file__\)(?:\s*\.parent)+\s*/\s*\"proxide\"\s*/\s*\"src\"\s*/\s*\"proxide\""
    r"\s*/\s*\"assets\"((?:\s*/\s*\"[^\"]+\")+)",
    re.S,
)
DATA_BARE = re.compile(
    r"Path\(__file__\)(?:\s*\.parent)+\s*/\s*\"proxide\"\s*/\s*\"tests\"\s*/\s*\"data\"((?:\s*/\s*\"[^\"]+\")+)",
    re.S,
)


def parts(group: str) -> str:
    return ", ".join('"' + s + '"' for s in re.findall(r'"([^"]+)"', group))


def rewrite(src: str):
    n = 0
    needs_data = needs_asset = False

    def a(m):
        nonlocal n, needs_asset
        n += 1
        needs_asset = True
        return f"proxide_asset({parts(m.group(1))})"

    def d(m):
        nonlocal n, needs_data
        n += 1
        needs_data = True
        return f"proxide_test_data({parts(m.group(1))})"

    src = ASSET.sub(a, src)
    src = ASSET_BARE.sub(a, src)
    src = DATA_BARE.sub(d, src)
    return src, n, needs_asset, needs_data


def add_import(src: str, names: list[str]) -> str:
    line = f"from tests._proxide_assets import {', '.join(sorted(names))}\n"
    if line in src:
        return src
    # after the last top-level 'from prolix' / 'import' line in the first import block
    lines = src.split("\n")
    last = 0
    for i, l in enumerate(lines[:80]):
        if l.startswith(("import ", "from ")) and not l.startswith("from __future__"):
            last = i
    lines.insert(last + 1, line.rstrip("\n"))
    return "\n".join(lines)


def main():
    apply = "--apply" in sys.argv
    total = 0
    for f in sorted(Path("tests").rglob("*.py")):
        if f.name == "_proxide_assets.py":
            continue
        s = f.read_text()
        new, n, na, nd = rewrite(s)
        if not n:
            continue
        names = (["proxide_asset"] if na else []) + (["proxide_test_data"] if nd else [])
        new = add_import(new, names)
        total += n
        print(f"{f}: {n} rewrite(s) -> {names}")
        if apply:
            f.write_text(new)
    print("total rewrites:", total)


main()
