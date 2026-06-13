"""Aggregate the P5 dt=1.0 fs NVT system-size sweep into a crossover table.

Reads the per-size result JSONs written by p5_nvt_size_sweep_dt1fs.py (campaign
ba334c1f, jobs 15911790 [n=2] + 15929525 [n=16..895]) and prints T_rot / T_trans
/ T_total vs n_waters, with the |dev|<=15 K (unit-test tolerance) and <=5 K (gate
tolerance) crossover flags.

Finding (gamma=10 ps^-1, dt=1.0 fs):
  - T_rot is stable at ALL sizes (299-312 K), including n=2.
  - The warm bias is purely TRANSLATIONAL finite-size: T_trans ~ 1/N decay
    (n=2: 600 K -> n=16: 320 -> n=64: 306 -> n>=216: ~300).
  - T_total (unit-test metric) crosses the +-15 K bound between n=2 and n=16;
    n>=16 passes. The n=2 failure is the 3-DOF (3N-3) translational mode.

Usage:
    uv run python scripts/analysis/p5_size_sweep_aggregate.py \\
        --results-dir outputs/results [--md outputs/p5_size_crossover.md]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

# Gate reference point (job 15870804, n=895, 5 seeds, 50 ps, dt=1fs, gamma=10).
GATE_REF = {"n_waters": 895, "t_rot": 299.63, "t_trans": 300.32, "t_total": 299.97,
            "source": "job 15870804 (gate, 5 seeds/50ps)"}


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(results_dir / "p5_size_sweep_dt1fs_n*_*.json"))):
        d = json.load(open(f))
        rows.append(d)
    rows.sort(key=lambda d: d["n_waters"])
    return rows


def fmt_table(rows: list[dict]) -> str:
    lines = []
    hdr = (f"{'n_waters':>8} {'T_rot':>8} {'T_trans':>9} {'T_total':>8} "
           f"{'|dev_rot|':>9} {'|dev_tot|':>9} {'tot<=15':>8} {'tot<=5':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for d in rows:
        n, tr, tt, tot = d["n_waters"], d["mean_t_rot"], d["mean_t_trans"], d["mean_t_total"]
        drot, dtot = abs(tr - 300), abs(tot - 300)
        lines.append(f"{n:>8} {tr:>8.1f} {tt:>9.1f} {tot:>8.1f} {drot:>9.1f} "
                     f"{dtot:>9.1f} {str(dtot <= 15):>8} {str(dtot <= 5):>7}")
    return "\n".join(lines)


def crossover(rows: list[dict]) -> tuple[int | None, int | None]:
    """Smallest n with |T_total-300|<=15 and <=5 (None if never)."""
    n15 = next((d["n_waters"] for d in rows if abs(d["mean_t_total"] - 300) <= 15), None)
    n5 = next((d["n_waters"] for d in rows if abs(d["mean_t_total"] - 300) <= 5), None)
    return n15, n5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="outputs/results")
    p.add_argument("--md", default=None, help="Optional markdown output path.")
    args = p.parse_args()

    rows = load_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"no sweep result JSONs found in {args.results_dir}")

    table = fmt_table(rows)
    n15, n5 = crossover(rows)
    print("=== P5 dt=1.0 fs NVT size-crossover (gamma=10 ps^-1) ===")
    print(table)
    print()
    print(f"Crossover N* (|T_total-300| <= 15 K): {n15}")
    print(f"Crossover N* (|T_total-300| <=  5 K): {n5}")
    print(f"T_rot within 15 K at all sizes: {all(abs(d['mean_t_rot']-300)<=15 for d in rows)}")
    print(f"Gate ref: n=895 T_rot={GATE_REF['t_rot']} T_trans={GATE_REF['t_trans']} "
          f"({GATE_REF['source']})")

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# P5 dt=1.0 fs NVT size-crossover (gamma=10 ps^-1)\n\n"
            "```\n" + table + "\n```\n\n"
            f"- Crossover N* (|T_total-300|<=15 K): **{n15}**\n"
            f"- Crossover N* (|T_total-300|<= 5 K): **{n5}**\n"
            f"- T_rot within 15 K at all sizes: "
            f"**{all(abs(d['mean_t_rot']-300)<=15 for d in rows)}**\n"
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
