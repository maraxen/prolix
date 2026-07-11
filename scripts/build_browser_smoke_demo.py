#!/usr/bin/env python3
"""Build Claim 2 W4 browser smoke demo artifacts (#278).

Usage::

    uv run python scripts/build_browser_smoke_demo.py [--out demo/browser_smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from prolix.api.browser_demo import build_browser_smoke_demo

_ROOT = Path(__file__).resolve().parents[1]
_B1 = _ROOT / "tests" / "bench" / "test_b1_smoke.py"


def _load_make_bundle():
    spec = importlib.util.spec_from_file_location("test_b1_smoke", _B1)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._make_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "demo" / "browser_smoke",
        help="Output directory for index.html, trace.json, trajectory.wasm",
    )
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--no-wasm", action="store_true")
    args = parser.parse_args()

    make_bundle = _load_make_bundle()
    bundle = make_bundle(10, seed=278)
    meta = build_browser_smoke_demo(
        args.out,
        bundle,
        n_steps=args.n_steps,
        compile_wasm=not args.no_wasm,
    )
    print(f"Wrote {meta['html_path']}")
    print(f"Wrote {meta['trace_path']}")
    if meta["wasm_path"] is not None:
        print(f"Wrote {meta['wasm_path']} ({meta['wasm_bytes']} bytes)")


if __name__ == "__main__":
    main()
