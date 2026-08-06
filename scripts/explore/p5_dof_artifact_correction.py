"""Apply the system-COM DOF-counting correction to the recorded Phase 5 size sweep.

Context (task 260806_p5_measurement_pipeline_audit): every copy of the Phase 5
trans/rot decomposition computes

    v_com     = p3.sum(axis=1) / M_water     # per-water COM velocity
    ke_trans  = 0.5 * M_water * sum(v_com**2)
    dof_trans = 3 * n_waters - 3             # assumes the SYSTEM COM was removed

but the runs that produced the documented table were executed with
``remove_linear_com_momentum=False`` (settle.py:1022 default; explicit at
p5_nvt_gate_dt1fs.py:50,182). The system COM is therefore a live, thermalized
3-DOF mode: settle.py:2120 draws each water's COM velocity independently from
N(0, kT/M), so Var(P_sys) = M_tot*kT and E[KE_com] = 3*kT/2 exactly.

Because the COM mode is force-free, its BAOAB O-step is the exact OU solution and
its kinetic energy is pinned at the THERMOSTAT TARGET rather than at the system's
actual temperature. The bias is therefore additive, not multiplicative:

    T_trans_measured = T_trans_true + 3 * T_target / (3N - 3)
    T_total_measured = T_total_true + 3 * T_target / (6N - 3)

This script applies that zero-free-parameter correction to the recorded per-seed
JSONs and tests whether any size-dependent residual survives it.

Usage:
    uv run python scripts/analysis/p5_dof_artifact_correction.py \\
        --results-dir outputs/results --out outputs/p5_dof_artifact_correction.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("p5_dof_artifact_correction")

T_TARGET_K = 300.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correct the Phase 5 size sweep for the system-COM DOF miscount"
    )
    p.add_argument("--results-dir", default="outputs/results")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _trans_bias_k(n_waters: int, t_target: float = T_TARGET_K) -> float:
    """Additive bias in the reported T_trans, in K: 3*T_target / (3N - 3)."""
    return 3.0 * t_target / (3 * n_waters - 3)


def _total_bias_k(n_waters: int, t_target: float = T_TARGET_K) -> float:
    """Additive bias in the reported T_total, in K: 3*T_target / (6N - 3)."""
    return 3.0 * t_target / (6 * n_waters - 3)


def _sem(std: float, n: int) -> float:
    return std / math.sqrt(n) if n > 0 else float("nan")


def _load(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("p5_size_sweep_dt1fs_n*.json")):
        d = json.loads(path.read_text())
        d["_file"] = path.name
        d["_kind"] = "sweep"
        records.append(d)
    gate = results_dir / "p5_nvt_gate_dt1fs_15870804.json"
    if gate.exists():
        d = json.loads(gate.read_text())
        d["_file"] = gate.name
        d["_kind"] = "gate"
        records.append(d)
    if not records:
        raise RuntimeError(f"no Phase 5 result JSONs found under {results_dir}")
    return sorted(records, key=lambda r: (r["n_waters"], r["_kind"]))


def _analyse(records: list[dict]) -> dict:
    rows = []
    for r in records:
        n = int(r["n_waters"])
        n_seeds = len(r["seeds"])
        bias_trans = _trans_bias_k(n)
        bias_total = _total_bias_k(n)

        t_trans = float(r["mean_t_trans"])
        t_rot = float(r["mean_t_rot"])
        t_total = float(r["mean_t_total"])

        sem_trans = _sem(float(r["std_t_trans"]), n_seeds)
        sem_rot = _sem(float(r["std_t_rot"]), n_seeds)

        corrected_trans = t_trans - bias_trans
        corrected_total = t_total - bias_total

        # Residual: does the corrected translational temperature still differ
        # from the (artifact-free, Galilean-invariant) rotational temperature?
        residual = corrected_trans - t_rot
        sem_residual = math.hypot(sem_trans, sem_rot)
        sigma = residual / sem_residual if sem_residual > 0 else float("nan")

        rows.append({
            "n_waters": n,
            "kind": r["_kind"],
            "file": r["_file"],
            "n_seeds": n_seeds,
            "steps": int(r.get("steps", -1)),
            "t_rot": t_rot,
            "t_trans_reported": t_trans,
            "t_total_reported": t_total,
            "predicted_trans_bias_k": bias_trans,
            "predicted_total_bias_k": bias_total,
            "t_trans_corrected": corrected_trans,
            "t_total_corrected": corrected_total,
            "sem_t_trans": sem_trans,
            "sem_t_rot": sem_rot,
            "residual_corrected_trans_minus_rot": residual,
            "sem_residual": sem_residual,
            "residual_sigma": sigma,
        })

    # Inverse-variance combine the residuals for n >= 64. The n=895 sweep and the
    # n=895 gate share seeds 42-44, so the sweep row is dropped from the combine
    # to avoid double-counting correlated data (the gate has more seeds).
    combine = [
        row for row in rows
        if row["n_waters"] >= 64 and not (row["n_waters"] == 895 and row["kind"] == "sweep")
    ]
    w_sum = sum(1.0 / row["sem_residual"] ** 2 for row in combine)
    wx_sum = sum(row["residual_corrected_trans_minus_rot"] / row["sem_residual"] ** 2 for row in combine)
    combined = wx_sum / w_sum
    combined_sem = 1.0 / math.sqrt(w_sum)

    # Independent check of the n=2 rotational deviation, which the artifact
    # predicts to be exactly zero (T_rot is Galilean-invariant).
    n2 = next((row for row in rows if row["n_waters"] == 2), None)
    n2_rot_dev = (n2["t_rot"] - T_TARGET_K) if n2 else float("nan")
    n2_rot_sigma = (n2_rot_dev / n2["sem_t_rot"]) if n2 else float("nan")

    return {
        "t_target_k": T_TARGET_K,
        "rows": rows,
        "combined_residual_n_ge_64_k": combined,
        "combined_residual_sem_k": combined_sem,
        "combined_residual_sigma": combined / combined_sem,
        "n2_rot_deviation_k": n2_rot_dev,
        "n2_rot_deviation_sigma": n2_rot_sigma,
        "max_abs_corrected_trans_deviation_k": max(
            abs(row["t_trans_corrected"] - T_TARGET_K) for row in rows
        ),
    }


def main() -> None:
    args = _parse_args()
    records = _load(Path(args.results_dir))
    log.info("Loaded %d recorded Phase 5 result files", len(records))

    result = _analyse(records)

    log.info(
        "%-6s %-6s %8s %10s %10s %10s %9s %8s",
        "n", "kind", "T_rot", "T_tr(rep)", "bias", "T_tr(corr)", "resid", "sigma",
    )
    for row in result["rows"]:
        log.info(
            "%-6d %-6s %8.2f %10.2f %10.3f %10.2f %9.2f %8.2f",
            row["n_waters"], row["kind"], row["t_rot"], row["t_trans_reported"],
            row["predicted_trans_bias_k"], row["t_trans_corrected"],
            row["residual_corrected_trans_minus_rot"], row["residual_sigma"],
        )

    log.info(
        "Combined residual (n>=64, seed-correlated 895 sweep dropped): %+.3f +- %.3f K (%.2f sigma)",
        result["combined_residual_n_ge_64_k"],
        result["combined_residual_sem_k"],
        result["combined_residual_sigma"],
    )
    log.info(
        "n=2 T_rot deviation (artifact predicts exactly 0): %+.2f K (%.2f sigma)",
        result["n2_rot_deviation_k"], result["n2_rot_deviation_sigma"],
    )
    log.info(
        "Largest |T_trans_corrected - 300| across all sizes: %.2f K",
        result["max_abs_corrected_trans_deviation_k"],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info("Result written to %s", out)


if __name__ == "__main__":
    main()
