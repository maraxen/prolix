#!/usr/bin/env python3
"""P7 task 12: Stage-2 scopes on/off A/B on (1vii, flash, pme on, 1.0 Å).

Import-time jax.named_scope patch (P5 _NullScope ContextDecorator) lives here
so prolix gains no hot-path config flag. Driver spawns children A/B/A/B.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260818
TOLERANCE_MARGIN = 0.05
PILOT_N_PAIRS = 10
N_PAIRS_MIN = 21
N_PAIRS_MAX = 200
N_DISCARD = 2
PROBE = ROOT / "scripts" / "experiments" / "profile_b1_flash_vs_autodiff_forces.py"


class _NullScope(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False


def _run_child(arm: str, out_dir: Path, pair_idx: int, n_warmup: int, n_trials: int, n_inner: int) -> dict:
    if arm == "off":
        import jax

        jax.named_scope = lambda *_a, **_k: _NullScope()  # ty: ignore[invalid-assignment]

    import importlib.util

    spec = importlib.util.spec_from_file_location("p7_probe", PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    probe_main = mod.main

    rec = out_dir / f"pair{pair_idx}_{arm}.json"
    summary = out_dir / f"pair{pair_idx}_{arm}_summary.json"
    argv = [
        "--protein", "1vii",
        "--mode", "flash",
        "--pme", "on",
        "--grid-spacing", "1.0",
        "--scopes", arm,
        "--stage", "2",
        "--n-warmup", str(n_warmup),
        "--n-trials", str(n_trials),
        "--n-inner", str(n_inner),
        "--emit-record", str(rec),
        "--out", str(summary),
        "--trace-dir", str(out_dir / f"pair{pair_idx}_{arm}_trace"),
    ]
    rc = probe_main(argv)
    if rc != 0:
        raise SystemExit(rc)
    payload = json.loads(summary.read_text())
    ms = payload.get("flash_forces_ms")
    if ms is None:
        raise RuntimeError(f"child {arm} pair {pair_idx} missing flash_forces_ms")
    return {"median_seconds": float(ms) / 1000.0, "record": str(rec)}


def _spawn_child(arm: str, out_dir: Path, pair_idx: int, n_warmup: int, n_trials: int, n_inner: int) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arm",
        arm,
        "--out-dir",
        str(out_dir),
        "--pair-idx",
        str(pair_idx),
        "--n-warmup",
        str(n_warmup),
        "--n-trials",
        str(n_trials),
        "--n-inner",
        str(n_inner),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    last = result.stdout.strip().splitlines()[-1]
    return json.loads(last)


def _bootstrap_median_ci(deltas, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED, ci: float = 0.90):
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas)
    boot = np.median(rng.choice(arr, size=(B, len(arr)), replace=True), axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def _driver(out_dir: Path, n_pairs_override: int | None, n_warmup: int, n_trials: int, n_inner: int) -> dict:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== pilot ({PILOT_N_PAIRS} A/A pairs) ===")
    pilot = []
    for i in range(PILOT_N_PAIRS):
        a = _spawn_child("off", out_dir, 1000 + i, n_warmup, n_trials, n_inner)
        b = _spawn_child("off", out_dir, 2000 + i, n_warmup, n_trials, n_inner)
        pilot.append((a["median_seconds"] - b["median_seconds"]) / b["median_seconds"])
    pilot_arr = np.asarray(pilot)
    q75, q25 = np.percentile(pilot_arr, [75, 25])
    s = float(q75 - q25)
    raw = math.ceil((3.3 * s / TOLERANCE_MARGIN) ** 2) if s > 0 else N_PAIRS_MIN
    n_pairs = int(min(max(raw, N_PAIRS_MIN), N_PAIRS_MAX))
    if n_pairs_override is not None:
        n_pairs = int(max(n_pairs_override, N_PAIRS_MIN))
    unresolved = raw > N_PAIRS_MAX
    print(f"pilot s={s:.6g} n_pairs={n_pairs}")

    deltas = []
    for i in range(n_pairs):
        on = _spawn_child("on", out_dir, i, n_warmup, n_trials, n_inner)
        off = _spawn_child("off", out_dir, i, n_warmup, n_trials, n_inner)
        deltas.append((on["median_seconds"] - off["median_seconds"]) / off["median_seconds"])
    kept = np.asarray(deltas[N_DISCARD:] if len(deltas) > N_DISCARD else deltas)
    median_delta = float(np.median(kept))
    ci_lo, ci_hi = _bootstrap_median_ci(kept)
    within_median = abs(median_delta) <= TOLERANCE_MARGIN
    within_ci = (-TOLERANCE_MARGIN <= ci_lo) and (ci_hi <= TOLERANCE_MARGIN)
    if unresolved:
        verdict = "UNRESOLVED_ON_CPU"
    elif within_median and within_ci:
        verdict = "PASS"
    elif within_median and not within_ci:
        verdict = "INSUFFICIENT_POWER"
    else:
        verdict = "DETECTED_PERTURBATION"
    return {
        "verdict": verdict,
        "median_delta": median_delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_pairs": n_pairs,
        "pilot_s": s,
        "protein": "1vii",
        "mode": "flash",
        "pme": "on",
        "grid_spacing": "1.0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--arm", choices=("on", "off"))
    parser.add_argument("--pair-idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "profiling" / "stage2" / "scope_ab")
    parser.add_argument("--n-pairs", type=int, default=None)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--n-inner", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.child:
        assert args.arm is not None
        payload = _run_child(args.arm, args.out_dir, args.pair_idx, args.n_warmup, args.n_trials, args.n_inner)
        print(json.dumps(payload))
        return 0

    result = _driver(args.out_dir, args.n_pairs, args.n_warmup, args.n_trials, args.n_inner)
    text = json.dumps(result, indent=2)
    print(text)
    out = args.out or (args.out_dir / "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
