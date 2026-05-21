#!/usr/bin/env python
"""§7.1 ensemble scaling curve analysis.

Joins bath catalog tags with slurm-stdout-parsed metrics to produce:
  - Long-format CSV: (N, mode, precision, execution_median_s, compile_s, loss_mean, loss_max)
  - Speedup plot: looped_exec / batched_exec vs N, one line per precision
  - Precision-cost plot: f64_exec / f32_exec vs N, one line per mode
  - Compile-time plot: compile_s vs N, faceted by (mode × precision)

Data sources:
  - ~/.bth/catalog/bathos.db (run tags + duration_s)
  - outputs/logs/slurm/*.fit_bonded_hp4.out (training stdout with
    execution_median_s, compile_time, loss values)

Re-runnable as more cluster data lands. Outputs to outputs/analysis/.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# Regexes match the stdout lines emitted by scripts/experiments/fit_bonded_hp4.py.
# Sample lines:
#   "Mode: looped  Precision: float32  N_mols: 32  N_conf_cap: 100  N_steps: 500"
#   "Training complete. compile≈9.16s  execution_median=87.009s  (min=86.352, max=87.088)"
#   "Wall-clock: 87.01 seconds"
#   "Final loss (Lane A): max=385481.5625, mean=59998.3181"
#   "GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB"
#   "Array job: 14191171  task: 4"
RE_HEADER = re.compile(
    r"Mode:\s*(?P<mode>\w+)\s+Precision:\s*(?P<precision>\w+)\s+"
    r"N_mols:\s*(?P<n>\S+)\s+N_conf_cap:\s*(?P<n_conf_cap>\S+)"
)
RE_COMPILE = re.compile(
    r"compile≈(?P<compile_s>-?[\d.]+)s\s+"
    r"execution_median=(?P<execution_median_s>[\d.]+)s\s+"
    r"\(min=(?P<execution_min_s>[\d.]+),\s*max=(?P<execution_max_s>[\d.]+)\)"
)
RE_WALLCLOCK = re.compile(r"Wall-clock:\s*(?P<wallclock_s>[\d.]+)\s*seconds")
RE_LANE_A = re.compile(
    r"Final loss \(Lane A\):\s*max=(?P<lane_a_loss_max>[\d.eE+-]+),\s*"
    r"mean=(?P<lane_a_loss_mean>[\d.eE+-]+)"
)
RE_GPU = re.compile(r"GPU:\s*(?P<gpu>[^,]+),")
RE_JOB = re.compile(r"Array job:\s*(?P<array_job>\S+)\s+task:\s*(?P<array_task_id>\S+)")

FLOAT_FIELDS = (
    "compile_s", "execution_median_s", "execution_min_s",
    "execution_max_s", "wallclock_s", "lane_a_loss_max", "lane_a_loss_mean",
)


def parse_slurm_log(path: Path) -> Optional[dict]:
    """Extract fields from one fit_bonded_hp4.out log. Returns None if incomplete."""
    text = path.read_text(errors="replace")
    out: dict = {"log_path": str(path)}

    for rx in (RE_HEADER, RE_COMPILE, RE_WALLCLOCK, RE_LANE_A, RE_GPU, RE_JOB):
        m = rx.search(text)
        if m is None:
            continue
        out.update(m.groupdict())

    # Type coercion
    for fld in FLOAT_FIELDS:
        if fld in out and out[fld] is not None:
            try:
                out[fld] = float(out[fld])
            except (ValueError, TypeError):
                out[fld] = None
    if "n" in out and out["n"] not in (None, "<none>"):
        try:
            out["n_mols_target"] = int(out["n"])
            del out["n"]
        except (ValueError, TypeError):
            out["n_mols_target"] = None

    if out.get("mode") is None or out.get("execution_median_s") is None:
        return None
    return out


def load_log_table(logs_dir: Path) -> pd.DataFrame:
    """Walk all *.fit_bonded_hp4.out logs and parse."""
    rows = []
    for p in sorted(logs_dir.glob("*.fit_bonded_hp4.out")):
        parsed = parse_slurm_log(p)
        if parsed:
            rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def join_with_catalog(log_df: pd.DataFrame, catalog_db: Path) -> pd.DataFrame:
    """Optional enrichment via bath catalog (process duration, git_hash, run id)."""
    if not catalog_db.exists():
        return log_df
    try:
        con = duckdb.connect(str(catalog_db), read_only=True)
        runs = con.execute(
            "SELECT id, duration_s, slurm_job_id, command, timestamp, git_hash "
            "FROM runs WHERE list_contains(tags, 's71')"
        ).fetchdf()
        con.close()
    except Exception as e:
        print(f"warn: could not read bath catalog: {e}")
        return log_df

    # Join by array_job_id reconstruction (slurm_job_id in bath is the array_task id)
    log_df["slurm_job_id"] = log_df.apply(
        lambda r: f"{r.get('array_job')}_{r.get('array_task_id')}" if r.get("array_job") else None,
        axis=1,
    )
    runs["slurm_job_id"] = runs["slurm_job_id"].astype(str)
    joined = log_df.merge(
        runs[["id", "duration_s", "slurm_job_id", "timestamp", "git_hash"]],
        on="slurm_job_id",
        how="left",
    )
    return joined


def compute_speedup_and_precision_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to per-(N, precision) speedup and per-(N, mode) precision cost."""
    if df.empty:
        return df
    keep = ["n_mols_target", "mode", "precision", "execution_median_s",
            "execution_min_s", "execution_max_s", "compile_s",
            "lane_a_loss_max", "lane_a_loss_mean"]
    df = df[[c for c in keep if c in df.columns]].dropna(
        subset=["n_mols_target", "mode", "precision", "execution_median_s"],
    )
    # If duplicate runs for same cell, keep latest (or median across replicas)
    df = df.groupby(["n_mols_target", "mode", "precision"], as_index=False).agg({
        "execution_median_s": "median",
        "execution_min_s": "min",
        "execution_max_s": "max",
        "compile_s": "median",
        "lane_a_loss_max": "median",
        "lane_a_loss_mean": "median",
    })
    return df


def plot_speedup(df: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for precision, group in df.groupby("precision"):
        pivot = group.pivot_table(
            index="n_mols_target", columns="mode", values="execution_median_s",
        )
        if "looped" in pivot.columns and "batched" in pivot.columns:
            speedup = pivot["looped"] / pivot["batched"]
            ax.plot(speedup.index, speedup.values, "o-", label=f"{precision}")
    ax.axhline(10.0, ls="--", color="r", alpha=0.5, label="§7.1 floor (10×)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size N (Lane A replicated)")
    ax.set_ylabel("Speedup ratio (looped exec / batched exec)")
    ax.set_title("§7.1 substrate scaling: looped vs batched")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_precision_cost(df: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, group in df.groupby("mode"):
        pivot = group.pivot_table(
            index="n_mols_target", columns="precision", values="execution_median_s",
        )
        if "float32" in pivot.columns and "float64" in pivot.columns:
            cost = pivot["float64"] / pivot["float32"]
            ax.plot(cost.index, cost.values, "o-", label=f"{mode}")
    ax.axhline(1.0, ls=":", color="grey", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Ensemble size N")
    ax.set_ylabel("Precision cost (f64 exec / f32 exec)")
    ax.set_title("§7.1 precision cost across ensemble size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_raw_execution(df: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, precision), group in df.groupby(["mode", "precision"]):
        ax.plot(group["n_mols_target"], group["execution_median_s"],
                "o-", label=f"{mode} {precision}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size N (Lane A replicated)")
    ax.set_ylabel("Execution median (s)")
    ax.set_title("§7.1 raw execution time per cell")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path,
                        default=Path("outputs/logs/slurm"))
    parser.add_argument("--catalog-db", type=Path,
                        default=Path.home() / ".bth" / "catalog" / "bathos.db")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/analysis"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_df = load_log_table(args.logs_dir)
    if log_df.empty:
        print(f"No logs parsed under {args.logs_dir}")
        return 1
    print(f"Parsed {len(log_df)} slurm logs")

    enriched = join_with_catalog(log_df, args.catalog_db)
    enriched.to_csv(args.out_dir / "s71_runs_raw.csv", index=False)
    print(f"wrote {args.out_dir / 's71_runs_raw.csv'} ({len(enriched)} rows)")

    summary = compute_speedup_and_precision_cost(enriched)
    summary.to_csv(args.out_dir / "s71_runs_summary.csv", index=False)
    print(f"wrote {args.out_dir / 's71_runs_summary.csv'} ({len(summary)} unique cells)")

    if summary.empty:
        print("warn: summary is empty; nothing to plot")
        return 1

    print()
    print("=== Per-cell median executions (s) ===")
    pivot = summary.pivot_table(
        index="n_mols_target",
        columns=["mode", "precision"],
        values="execution_median_s",
    )
    print(pivot.round(2))

    print()
    print("=== Speedup (looped exec / batched exec) ===")
    for precision in summary["precision"].dropna().unique():
        sub = summary[summary["precision"] == precision]
        p = sub.pivot_table(index="n_mols_target", columns="mode", values="execution_median_s")
        if "looped" in p.columns and "batched" in p.columns:
            print(f"  {precision}:")
            print((p["looped"] / p["batched"]).round(2).to_string())

    print()
    print("=== Precision cost (f64 / f32) ===")
    for mode in summary["mode"].dropna().unique():
        sub = summary[summary["mode"] == mode]
        p = sub.pivot_table(index="n_mols_target", columns="precision", values="execution_median_s")
        if "float32" in p.columns and "float64" in p.columns:
            print(f"  {mode}:")
            print((p["float64"] / p["float32"]).round(3).to_string())

    plot_speedup(summary, args.out_dir / "s71_speedup.png")
    plot_precision_cost(summary, args.out_dir / "s71_precision_cost.png")
    plot_raw_execution(summary, args.out_dir / "s71_raw_execution.png")
    print()
    print(f"plots → {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
