#!/usr/bin/env python3
"""§7.1 external comparator figure generator.

Reads the bath catalog for the s71-external-baseline campaign and produces:
  - outputs/analysis/s71_external_comparator.png — performance plot
  - outputs/analysis/s71_compatibility_matrix.md — compatibility matrix + threshold eval

Idempotent. Works incrementally on partial data (C3 smoke → C6 full sweep).

Locked decisions (260527):
  - Comparator: prolix vs DMFF / TorchMD / espaloma on bonded Scope A (one Adam step)
  - System: ANI-1x 16-base set tiled to N
  - Hardware policy: "what doesn't run IS data" — failures plot as ✗ markers
  - Pass threshold: prolix per_mol_step_seconds < 0.5 × min(dmff, torchmd) per hardware
                    (espaloma excluded from gate; reported separately as batched comparator)
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Colorblind-safe palette (from spec)
TOOL_COLORS = {
    "prolix": "#0173B2",      # blue
    "dmff": "#DE8F05",         # orange
    "torchmd": "#029E73",      # green
    "espaloma": "#CC78BC",     # purple
}

TOOL_MARKERS = {
    "prolix": "o",
    "dmff": "s",
    "torchmd": "^",
    "espaloma": "D",
}


def load_runs(
    catalog_glob: str,
    campaign_id: str,
    tag_required: list[str],
    tag_exclude: list[str],
) -> pd.DataFrame:
    """Load runs from bath catalog matching campaign and tag filters.

    Returns a DataFrame with columns: id, status, output_paths, tags, campaign_id,
    plus extracted fields from the JSON output files: tool, hardware_tag, n_mols,
    precision, per_mol_step_seconds, final_loss, etc.
    """
    con = duckdb.connect()

    # Build filter clauses
    tag_filters = []
    for tag in tag_required:
        tag_filters.append(f"list_contains(tags, '{tag}')")
    tag_filter_sql = " AND ".join(tag_filters) if tag_filters else "1=1"

    tag_exclude_sql = ""
    for tag in tag_exclude:
        tag_exclude_sql += f" AND NOT list_contains(tags, '{tag}')"

    # Query the catalog
    try:
        query = f"""
            SELECT id, status, output_paths, tags, campaign_id, timestamp, git_hash
            FROM read_parquet('{catalog_glob}')
            WHERE ({tag_filter_sql})
              AND (campaign_id LIKE '{campaign_id[:8]}%' OR campaign_id = '{campaign_id}')
              {tag_exclude_sql}
            ORDER BY timestamp DESC
        """
        logger.info(f"Querying catalog with: {query[:100]}...")
        df_catalog = con.execute(query).df()
    except Exception as e:
        logger.error(f"Failed to query catalog: {e}")
        raise

    if len(df_catalog) == 0:
        raise RuntimeError(
            f"No runs found for campaign '{campaign_id}' with tags {tag_required}. "
            "Did you `bth sync engaging --pull`?"
        )

    logger.info(f"Found {len(df_catalog)} catalog rows")

    # Load JSON results from output_paths
    results = []
    for idx, row in df_catalog.iterrows():
        if row["status"] == "failed":
            # Record as a failure cell
            results.append({
                "id": row["id"],
                "status": "failed",
                "campaign_id": row["campaign_id"],
                "timestamp": row["timestamp"],
                "git_hash": row["git_hash"],
                "tool": None,
                "hardware_tag": None,
                "n_mols": None,
                "precision": None,
                "per_mol_step_seconds": None,
                "final_loss": None,
                "failure_reason": "catalog status=failed",
            })
            continue

        # Try to load from output_paths
        output_paths = row["output_paths"]
        if not isinstance(output_paths, (list, np.ndarray)) or len(output_paths) == 0:
            logger.debug(f"Row {row['id']}: no output_paths")
            continue

        # Handle numpy array
        if isinstance(output_paths, np.ndarray):
            output_paths = output_paths.tolist()

        out_path = Path(output_paths[0])
        if not out_path.exists():
            logger.debug(f"Row {row['id']}: output file not found: {out_path}")
            continue

        try:
            with open(out_path) as f:
                payload = json.load(f)

            # Extract required fields
            result = {
                "id": row["id"],
                "status": row["status"],
                "campaign_id": row["campaign_id"],
                "timestamp": row["timestamp"],
                "git_hash": row["git_hash"],
                "tool": payload.get("tool"),
                "hardware_tag": payload.get("hardware_tag"),
                "n_mols": payload.get("n_mols"),
                "precision": payload.get("precision"),
                "per_mol_step_seconds": payload.get("per_mol_step_seconds"),
                "final_loss": payload.get("final_loss"),
                "failure_reason": None,
            }
            results.append(result)
        except Exception as e:
            logger.debug(f"Row {row['id']}: failed to load JSON: {e}")
            continue

    df_results = pd.DataFrame(results)
    logger.info(f"Loaded {len(df_results)} results with valid output")

    return df_results


def derive_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a compatibility matrix: rows=tool, cols=hardware.

    Each cell contains either:
      - A run outcome: per_mol_step_seconds value
      - A failure marker: tool × hardware combination with no successful run
    """
    # Filter to successful runs only
    df_success = df[df["status"] == "completed"].copy()

    # For each (tool, hardware) pair, take the most recent successful run
    df_success = df_success.sort_values("timestamp", ascending=False).drop_duplicates(
        subset=["tool", "hardware_tag", "n_mols", "precision"], keep="first"
    )

    return df_success


def evaluate_pass_threshold(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate pass threshold per hardware.

    Rule: prolix per_mol_step_seconds < 0.5 × min(dmff, torchmd) per hardware.
    Espaloma excluded from gate but reported separately.
    """
    evaluation = {}

    # Group by hardware
    for hardware in df["hardware_tag"].unique():
        if pd.isna(hardware):
            continue

        df_hw = df[df["hardware_tag"] == hardware]

        # Find prolix, dmff, torchmd values (espaloma separate)
        prolix_rows = df_hw[df_hw["tool"] == "prolix"]
        dmff_rows = df_hw[df_hw["tool"] == "dmff"]
        torchmd_rows = df_hw[df_hw["tool"] == "torchmd"]
        espaloma_rows = df_hw[df_hw["tool"] == "espaloma"]

        prolix_val = None
        dmff_val = None
        torchmd_val = None
        espaloma_val = None

        if len(prolix_rows) > 0:
            prolix_val = prolix_rows["per_mol_step_seconds"].iloc[0]
        if len(dmff_rows) > 0:
            dmff_val = dmff_rows["per_mol_step_seconds"].iloc[0]
        if len(torchmd_rows) > 0:
            torchmd_val = torchmd_rows["per_mol_step_seconds"].iloc[0]
        if len(espaloma_rows) > 0:
            espaloma_val = espaloma_rows["per_mol_step_seconds"].iloc[0]

        # Compute threshold
        gate_min = None
        if dmff_val is not None and torchmd_val is not None:
            gate_min = min(dmff_val, torchmd_val)
        elif dmff_val is not None:
            gate_min = dmff_val
        elif torchmd_val is not None:
            gate_min = torchmd_val

        gate_threshold = gate_min * 0.5 if gate_min is not None else None

        outcome = "UNKNOWN"
        if prolix_val is not None and gate_threshold is not None:
            if prolix_val < gate_threshold:
                outcome = "PASS"
            elif prolix_val < gate_threshold * 1.1:  # 10% margin = MARGINAL
                outcome = "MARGINAL"
            else:
                outcome = "FAIL"

        evaluation[hardware] = {
            "prolix": prolix_val,
            "dmff": dmff_val,
            "torchmd": torchmd_val,
            "espaloma": espaloma_val,
            "gate_min": gate_min,
            "gate_threshold": gate_threshold,
            "outcome": outcome,
        }

    return evaluation


def identify_failure_cells(df: pd.DataFrame) -> dict[tuple, str]:
    """Identify (tool, hardware) cells with no successful run.

    Returns dict: (tool, hardware) -> reason string
    """
    # Get all tool×hardware combinations that appear
    all_combos = set()
    for _, row in df.iterrows():
        if pd.notna(row["tool"]) and pd.notna(row["hardware_tag"]):
            all_combos.add((row["tool"], row["hardware_tag"]))

    # Get successful tool×hardware combinations
    df_success = df[df["status"] == "completed"]
    success_combos = set()
    for _, row in df_success.iterrows():
        if pd.notna(row["tool"]) and pd.notna(row["hardware_tag"]):
            success_combos.add((row["tool"], row["hardware_tag"]))

    # Failures = attempted but not successful
    failure_cells = {}
    for combo in all_combos:
        if combo not in success_combos:
            tool, hardware = combo
            # Find the failed row to extract reason if available
            failed_rows = df[
                (df["tool"] == tool)
                & (df["hardware_tag"] == hardware)
                & (df["status"] == "failed")
            ]
            reason = "did not run"
            if len(failed_rows) > 0:
                failure_reason = failed_rows.iloc[0].get("failure_reason")
                if failure_reason:
                    reason = failure_reason
            failure_cells[combo] = reason

    return failure_cells


def plot_performance(
    df: pd.DataFrame,
    failure_cells: dict[tuple, str],
    out_path: Path,
) -> None:
    """Generate performance plot with failure indicators."""

    # Get unique precisions and hardwares
    precisions = sorted(df["precision"].dropna().unique())
    hardwares = sorted(df["hardware_tag"].dropna().unique())

    logger.info(f"Precisions: {precisions}")
    logger.info(f"Hardwares: {hardwares}")

    # Handle empty case
    if len(precisions) == 0:
        logger.warning("No precision values found in data. Using default precision order.")
        precisions = ["float32", "float64"]
    if len(hardwares) == 0:
        logger.warning("No hardware values found in data. Using default hardware order.")
        hardwares = ["a100-sm80", "rtx-pro-6000-blackwell"]

    n_prec = len(precisions)
    n_hw = len(hardwares)

    fig, axes = plt.subplots(
        nrows=n_prec,
        ncols=n_hw,
        figsize=(5 * n_hw, 4 * n_prec),
        sharey=True,
    )

    # Ensure axes is 2D even if only 1 row/col
    if n_prec == 1 and n_hw == 1:
        axes = [[axes]]
    elif n_prec == 1:
        axes = [axes]
    elif n_hw == 1:
        axes = [[ax] for ax in axes]

    # Determine Y-axis range
    y_values = df[df["status"] == "completed"]["per_mol_step_seconds"].dropna()
    if len(y_values) > 0:
        y_min = y_values.min()
        y_max = y_values.max()
        # Add 50% above max for failure markers
        y_max_plot = y_max * 2.5
    else:
        y_min = 1e-5
        y_max_plot = 1e-2

    for i, prec in enumerate(precisions):
        for j, hw in enumerate(hardwares):
            ax = axes[i][j]

            df_cell = df[
                (df["precision"] == prec) & (df["hardware_tag"] == hw) & (df["status"] == "completed")
            ]

            # Plot successful runs
            tools = sorted(df_cell["tool"].unique())
            for tool in tools:
                df_tool = df_cell[df_cell["tool"] == tool].sort_values("n_mols")

                if len(df_tool) > 0:
                    ax.plot(
                        df_tool["n_mols"],
                        df_tool["per_mol_step_seconds"],
                        marker=TOOL_MARKERS.get(tool, "o"),
                        color=TOOL_COLORS.get(tool, "gray"),
                        label=tool,
                        linewidth=1.5,
                        markersize=6,
                    )

            # Plot failure markers
            for (tool, hw_fail), reason in failure_cells.items():
                if hw_fail == hw:
                    # Place at representative Y position
                    y_pos = y_max_plot
                    ax.scatter(
                        256,  # Arbitrary X position in middle of range
                        y_pos,
                        marker="x",
                        s=200,
                        color="red",
                        linewidths=2,
                        zorder=10,
                    )
                    ax.text(
                        256,
                        y_pos * 1.2,
                        f"✗",
                        ha="center",
                        fontsize=12,
                        color="red",
                    )

            ax.set_yscale("log")
            ax.set_ylabel("Time per mol-step (seconds)")
            ax.set_xlabel("N molecules")

            # Facet title
            hw_label = hw.replace("-", " ").replace("sm80", "A100").replace("blackwell", "RTX Pro 6000 Blackwell")
            prec_label = "float32" if prec == "float32" else "float64"
            ax.set_title(f"{hw_label.title()} ({prec_label})", fontsize=10)

            ax.grid(True, alpha=0.3)

    # Add legend outside the plot area
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(1.15, 0.5),
        )

    fig.suptitle("§7.1 Bonded-energy Scope A: prolix vs external comparators", fontsize=14, y=0.995)

    # Footer
    if len(df) > 0:
        git_hash = df.iloc[0].get("git_hash", "unknown")
        if isinstance(git_hash, float) or git_hash == "unknown":
            git_hash = "unknown"[:8]
        else:
            git_hash = str(git_hash)[:8]
        footer_text = (
            f"Anchor: campaign edbd0b84, HEAD {git_hash}. "
            "Pre-registered pass threshold: prolix < 0.5 × min(dmff, torchmd) per hardware "
            "(espaloma excluded from gate)."
        )
        fig.text(0.05, -0.02, footer_text, fontsize=8, style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {out_path}")
    plt.close()


def write_compatibility_md(
    df: pd.DataFrame,
    failure_cells: dict[tuple, str],
    threshold_eval: dict[str, Any],
    out_path: Path,
) -> None:
    """Write compatibility matrix markdown."""

    # Prepare data
    df_success = df[df["status"] == "completed"]
    tools = sorted(df_success["tool"].unique())
    hardwares = sorted(df_success["hardware_tag"].unique())

    # Build compatibility table
    compat_rows = []
    for tool in tools:
        row = {"tool": f"**{tool}**"}
        for hw in hardwares:
            df_cell = df_success[
                (df_success["tool"] == tool) & (df_success["hardware_tag"] == hw)
            ]
            if len(df_cell) > 0:
                row[hw] = "✓ ran"
            else:
                # Check if it was attempted
                df_attempted = df[
                    (df["tool"] == tool) & (df["hardware_tag"] == hw)
                ]
                if len(df_attempted) > 0:
                    reason = failure_cells.get((tool, hw), "did not run")
                    row[hw] = f"✗ {reason}"
                else:
                    row[hw] = "—"
        compat_rows.append(row)

    # Build markdown
    md_lines = [
        "# §7.1 Compatibility Matrix — Bonded Scope A",
        "",
        f"Source: campaign edbd0b84, HEAD {df.iloc[0]['git_hash'][:8]}, generated {datetime.now().isoformat()}.",
        "",
        "| Tool | " + " | ".join([hw.replace("-", " ") for hw in hardwares]) + " |",
        "|---|" + "|".join(["---"] * len(hardwares)) + "|",
    ]

    for row in compat_rows:
        cells = [row["tool"]]
        for hw in hardwares:
            cells.append(row.get(hw, "—"))
        md_lines.append("| " + " | ".join(cells) + " |")

    md_lines.extend([
        "",
        "Legend: ✓ ran = bonded-energy Scope A primitive completed; ✗ <reason> = build_failed / runtime_failed / oom / timeout / cpu_fallback.",
        "",
        "## Pass-threshold evaluation",
        "",
        "Locked rule (260527): prolix per_mol_step_seconds < 0.5 × min(dmff, torchmd) on the same hardware. Espaloma excluded from gate.",
        "",
        "| Hardware | min(dmff, torchmd) | 0.5× gate | prolix observed | Outcome |",
        "|---|---|---|---|---|",
    ])

    for hw in sorted(hardwares):
        eval_data = threshold_eval.get(hw, {})
        gate_min = eval_data.get("gate_min")
        gate_threshold = eval_data.get("gate_threshold")
        prolix_val = eval_data.get("prolix")
        outcome = eval_data.get("outcome", "UNKNOWN")

        if gate_min is not None and gate_threshold is not None and prolix_val is not None:
            md_lines.append(
                f"| {hw} | {gate_min:.6e} | {gate_threshold:.6e} | {prolix_val:.6e} | {outcome} |"
            )
        else:
            md_lines.append(f"| {hw} | — | — | — | {outcome} |")

    md_lines.extend([
        "",
        "## Espaloma reference (batched comparator, not in gate)",
        "",
        "| Hardware | espaloma | prolix | Ratio (prolix/espaloma) |",
        "|---|---|---|---|",
    ])

    for hw in sorted(hardwares):
        espaloma_val = threshold_eval.get(hw, {}).get("espaloma")
        prolix_val = threshold_eval.get(hw, {}).get("prolix")

        if espaloma_val is not None and prolix_val is not None:
            ratio = prolix_val / espaloma_val
            md_lines.append(f"| {hw} | {espaloma_val:.6e} | {prolix_val:.6e} | {ratio:.2f}× |")
        else:
            md_lines.append(f"| {hw} | — | — | — |")

    md_content = "\n".join(md_lines)
    out_path.write_text(md_content)
    logger.info(f"Saved compatibility matrix to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-id",
        default="edbd0b84-c0e2-4d2d-a89f-5f9fe8ae3aff",
        help="Campaign UUID filter (or 8-char prefix).",
    )
    parser.add_argument(
        "--catalog-glob",
        default=str(Path.home() / ".bth/catalog/runs/prolix/run_*.parquet"),
        help="Glob pattern for bath catalog parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/analysis",
        help="Output directory for PNG and MD.",
    )
    parser.add_argument(
        "--tag-required",
        action="append",
        default=["external-baseline"],
        help="Require these tags on every row (repeatable).",
    )
    parser.add_argument(
        "--tag-exclude",
        action="append",
        default=[],
        help="Exclude rows with these tags.",
    )
    parser.add_argument(
        "--png-name",
        default="s71_external_comparator.png",
        help="Output PNG filename.",
    )
    parser.add_argument(
        "--md-name",
        default="s71_compatibility_matrix.md",
        help="Output Markdown filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PNG output.",
    )

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading runs from catalog...")
    df = load_runs(
        args.catalog_glob,
        args.campaign_id,
        args.tag_required,
        args.tag_exclude,
    )

    logger.info("Deriving compatibility matrix...")
    df_compat = derive_compatibility(df)

    logger.info("Identifying failure cells...")
    failure_cells = identify_failure_cells(df)

    logger.info("Evaluating pass thresholds...")
    threshold_eval = evaluate_pass_threshold(df_compat)

    # Generate plot
    logger.info("Generating performance plot...")
    plot_out = out_dir / args.png_name
    plot_performance(df_compat, failure_cells, plot_out)

    # Generate markdown
    logger.info("Generating compatibility matrix...")
    md_out = out_dir / args.md_name
    write_compatibility_md(df, failure_cells, threshold_eval, md_out)

    logger.info("Done!")


if __name__ == "__main__":
    main()
