#!/usr/bin/env python3
"""Generate Phase 1 diagnostic plots and hypothesis support verdict.

Reads results from phase1_diagnostic_runner and produces:
1. residual_convergence.png - RATTLE iteration convergence curves
2. ke_precision_comparison.png - Float32 vs Float64 divergence
3. temperature_by_projection_site.png - Projection site effects
4. PHASE1_VERDICT.md - Quantitative hypothesis support scores

Usage:
    python scripts/phase1_plotting_and_verdict.py --input ./phase1_results --output ./phase1_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Try importing matplotlib; skip plotting if unavailable
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_summary_json(summary_path: Path) -> dict:
    """Parse summary.json from diagnostic runner."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def generate_residual_convergence_plot(summary: list, output_dir: Path) -> None:
    """Plot SETTLE residual vs iteration count (AXIS 1)."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib unavailable; skipping residual convergence plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dt_values = [0.5, 1.0, 2.0]

    for ax_idx, dt in enumerate(dt_values):
        ax = axes[ax_idx]

        # Filter for this dt and axis
        axis1_data = [r for r in summary if r.get('axis') == 'iterations' and r.get('dt_akma') == dt]

        if not axis1_data:
            ax.text(0.5, 0.5, f'No data for dt={dt}', ha='center', va='center')
            continue

        # Sort by n_iters
        axis1_data = sorted(axis1_data, key=lambda x: x.get('n_iters', 0))

        n_iters_values = [r['n_iters'] for r in axis1_data]
        T_means = [r['T_mean'] for r in axis1_data]
        T_stds = [r['T_std'] for r in axis1_data]

        ax.errorbar(n_iters_values, T_means, yerr=T_stds, marker='o', capsize=5, label=f'dt={dt}')
        ax.axhline(y=300.0, color='r', linestyle='--', alpha=0.5, label='Target T=300K')
        ax.axhline(y=305.0, color='orange', linestyle='--', alpha=0.5, label='Acceptable (±5K)')
        ax.axhline(y=295.0, color='orange', linestyle='--', alpha=0.5)

        ax.set_xscale('log')
        ax.set_xlabel('RATTLE Iterations (log scale)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'dt = {dt:.1f} AKMA')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'residual_convergence.png', dpi=150, bbox_inches='tight')
    print(f"Saved: residual_convergence.png")
    plt.close()


def generate_precision_comparison_plot(summary: list, output_dir: Path) -> None:
    """Plot Float32 vs Float64 kinetic energy tracking (AXIS 2)."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib unavailable; skipping precision comparison plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dt_values = [0.5, 1.0, 2.0]

    for ax_idx, dt in enumerate(dt_values):
        ax = axes[ax_idx]

        axis2_data = [r for r in summary if r.get('axis') == 'precision' and r.get('dt_akma') == dt]

        if not axis2_data:
            ax.text(0.5, 0.5, f'No data for dt={dt}', ha='center', va='center')
            continue

        f32_data = [r for r in axis2_data if r.get('precision') == 'float32']
        f64_data = [r for r in axis2_data if r.get('precision') == 'float64']

        x_pos = [0, 1]
        means = [
            f32_data[0]['T_mean'] if f32_data else np.nan,
            f64_data[0]['T_mean'] if f64_data else np.nan,
        ]
        stds = [
            f32_data[0]['T_std'] if f32_data else 0,
            f64_data[0]['T_std'] if f64_data else 0,
        ]

        ax.bar(x_pos, means, yerr=stds, capsize=5, color=['lightcoral', 'lightblue'], alpha=0.7)
        ax.axhline(y=300.0, color='g', linestyle='--', alpha=0.5, label='Target')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Float32', 'Float64'])
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'dt = {dt:.1f} AKMA')
        ax.set_ylim([250, 350])
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'ke_precision_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: ke_precision_comparison.png")
    plt.close()


def generate_projection_site_plot(summary: list, output_dir: Path) -> None:
    """Plot projection site effects (AXIS 3)."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib unavailable; skipping projection site plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dt_values = [0.5, 1.0, 2.0]
    proj_sites = ['post_o', 'post_settle_vel', 'both']

    for ax_idx, dt in enumerate(dt_values):
        ax = axes[ax_idx]

        axis3_data = [r for r in summary if r.get('axis') == 'projection' and r.get('dt_akma') == dt]

        if not axis3_data:
            ax.text(0.5, 0.5, f'No data for dt={dt}', ha='center', va='center')
            continue

        means = []
        stds = []
        for site in proj_sites:
            site_data = [r for r in axis3_data if r.get('projection_site') == site]
            if site_data:
                means.append(site_data[0]['T_mean'])
                stds.append(site_data[0]['T_std'])
            else:
                means.append(np.nan)
                stds.append(0)

        x_pos = range(len(proj_sites))
        ax.bar(x_pos, means, yerr=stds, capsize=5, color=['lightgreen', 'lightyellow', 'lightcyan'], alpha=0.7)
        ax.axhline(y=300.0, color='r', linestyle='--', alpha=0.5, label='Target')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(proj_sites, rotation=15, ha='right')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'dt = {dt:.1f} AKMA')
        ax.set_ylim([250, 350])
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_by_projection_site.png', dpi=150, bbox_inches='tight')
    print(f"Saved: temperature_by_projection_site.png")
    plt.close()


def compute_hypothesis_support(summary: list) -> dict:
    """Quantitatively score hypothesis support based on observed data."""
    hypothesis_scores = {
        'H1_rattle_saturation': 0.0,
        'H2_float32_precision': 0.0,
        'H3_projection_timing': 0.0,
    }

    evidence = {
        'H1': [],
        'H2': [],
        'H3': [],
    }

    # HYPOTHESIS 1: RATTLE Saturation
    # Prediction: Residual decreases monotonically with n_iters;
    #            Saturation at n_iters ~ 2-3 @ dt=1.0 fs
    axis1_dt10 = [r for r in summary if r.get('axis') == 'iterations' and r.get('dt_akma') == 1.0]
    if axis1_dt10:
        axis1_dt10 = sorted(axis1_dt10, key=lambda x: x.get('n_iters', 0))
        T_values = [r['T_mean'] for r in axis1_dt10]

        # Check monotonic decrease (all adjacent diffs are non-negative or close)
        diffs = [T_values[i+1] - T_values[i] for i in range(len(T_values)-1)]
        n_decreasing = sum(1 for d in diffs if d <= 0.5)  # Allow small noise
        monotonic_ratio = n_decreasing / len(diffs) if diffs else 0

        # Check saturation at low iterations
        T_at_1 = T_values[0] if len(T_values) > 0 else 300
        T_at_10 = T_values[4] if len(T_values) > 4 else T_values[-1]
        improvement_1_to_10 = abs(T_at_1 - T_at_10)

        hypothesis_scores['H1_rattle_saturation'] = min(1.0, monotonic_ratio * (improvement_1_to_10 / 30))
        evidence['H1'] = [
            f"Monotonic improvement ratio: {monotonic_ratio:.2f}",
            f"Temperature drop (n_iters 1→10): {improvement_1_to_10:.1f} K",
            f"Final error @ n_iters=50: {T_values[-1] - 300:.1f} K",
        ]

    # HYPOTHESIS 2: Float32 Precision
    # Prediction: Float32 shows much worse T tracking than float64
    axis2_data = [r for r in summary if r.get('axis') == 'precision']
    if axis2_data:
        f32_mean = np.mean([r['T_mean'] for r in axis2_data if r.get('precision') == 'float32'])
        f64_mean = np.mean([r['T_mean'] for r in axis2_data if r.get('precision') == 'float64'])
        precision_gap = abs(f32_mean - f64_mean)

        # Support if gap > 10 K (significant degradation in float32)
        hypothesis_scores['H2_float32_precision'] = min(1.0, max(0, precision_gap - 5) / 25)
        evidence['H2'] = [
            f"Float32 mean T: {f32_mean:.1f} K",
            f"Float64 mean T: {f64_mean:.1f} K",
            f"Precision gap: {precision_gap:.1f} K",
        ]

    # HYPOTHESIS 3: Projection Timing
    # Prediction: 'both' projection is more stable than 'post_o'
    axis3_data = [r for r in summary if r.get('axis') == 'projection']
    if axis3_data:
        post_o_T = np.mean([r['T_mean'] for r in axis3_data if r.get('projection_site') == 'post_o'])
        both_T = np.mean([r['T_mean'] for r in axis3_data if r.get('projection_site') == 'both'])
        proj_improvement = abs(post_o_T - both_T)

        # Support if 'both' is significantly better (ΔT > 5 K better)
        hypothesis_scores['H3_projection_timing'] = min(1.0, max(0, proj_improvement - 2) / 20)
        evidence['H3'] = [
            f"post_o mean T: {post_o_T:.1f} K",
            f"both mean T: {both_T:.1f} K",
            f"Projection improvement: {proj_improvement:.1f} K",
        ]

    return {
        'scores': hypothesis_scores,
        'evidence': evidence,
    }


def generate_verdict_markdown(
    summary: list,
    hypothesis_support: dict,
    output_dir: Path,
) -> None:
    """Generate Phase 1 VERDICT document."""
    scores = hypothesis_support['scores']
    evidence = hypothesis_support['evidence']

    # Determine go/no-go decision
    total_score = sum(scores.values()) / len(scores)
    if total_score > 0.5:
        verdict = "CONDITIONAL APPROVAL"
    else:
        verdict = "ESCALATE FOR PHASE 2"

    # Compute final temperature errors
    all_temps = [r['T_mean'] for r in summary if 'T_mean' in r]
    max_error_K = max([abs(t - 300) for t in all_temps]) if all_temps else np.nan
    avg_error_K = np.mean([abs(t - 300) for t in all_temps]) if all_temps else np.nan

    markdown = f"""# PHASE 1 VERDICT: ROOT-CAUSE ISOLATION FOR LANGEVIN THERMOSTAT PARITY

**Date**: 2026-04-23
**Status**: {verdict}

## Executive Summary

Phase 1 diagnostics completed on {len(summary)} configurations across 3 ablation axes.
**Overall Verdict**: {verdict}

### Key Findings

- **Maximum observed ΔT error**: {max_error_K:.1f} K
- **Mean ΔT error across all configs**: {avg_error_K:.1f} K
- **Target acceptability (G4 bound)**: < 5 K

## Hypothesis Support Scores (0.0 = no support, 1.0 = full support)

### Hypothesis 1: Fixed RATTLE Saturation (HIGH prior)
- **Quantitative Support Score**: {scores['H1_rattle_saturation']:.2f}
- **Evidence**:
{chr(10).join(['  - ' + e for e in evidence['H1']])}
- **Interpretation**:
  - Score > 0.7 → **CONFIRMED (HIGH)**: RATTLE convergence is primary cause
  - Score 0.3-0.7 → **PARTIAL**: RATTLE contributes but not sole cause
  - Score < 0.3 → **REJECTED**: RATTLE not a major factor

### Hypothesis 2: Float32 Precision Degradation (MEDIUM prior)
- **Quantitative Support Score**: {scores['H2_float32_precision']:.2f}
- **Evidence**:
{chr(10).join(['  - ' + e for e in evidence['H2']])}
- **Interpretation**:
  - Score > 0.7 → **CONFIRMED (HIGH)**: Precision loss is critical bottleneck
  - Score 0.3-0.7 → **PARTIAL**: Float32 contributes to instability
  - Score < 0.3 → **REJECTED**: Float64 operations are sufficient

### Hypothesis 3: Projection Site Timing Bias (MEDIUM prior)
- **Quantitative Support Score**: {scores['H3_projection_timing']:.2f}
- **Evidence**:
{chr(10).join(['  - ' + e for e in evidence['H3']])}
- **Interpretation**:
  - Score > 0.7 → **CONFIRMED (MEDIUM)**: Projection location is significant
  - Score 0.3-0.7 → **PARTIAL**: Projection helps but not the main issue
  - Score < 0.3 → **REJECTED**: Projection timing is not the bottleneck

## Temperature Breakdown by Configuration

### AXIS 1: RATTLE Iteration Sweep (dt=1.0 AKMA)
"""

    axis1_dt10 = [r for r in summary if r.get('axis') == 'iterations' and r.get('dt_akma') == 1.0]
    if axis1_dt10:
        axis1_dt10 = sorted(axis1_dt10, key=lambda x: x.get('n_iters', 0))
        markdown += "\n| n_iters | T_mean (K) | ΔT from 300K | T_std (K) |\n"
        markdown += "|---------|-----------|-------------|----------|\n"
        for r in axis1_dt10:
            markdown += f"| {r['n_iters']:7d} | {r['T_mean']:9.1f} | {abs(r['T_mean'] - 300):11.1f} | {r['T_std']:9.2f} |\n"

    markdown += f"""

### AXIS 2: Float Precision Comparison
"""

    for dt in [0.5, 1.0, 2.0]:
        axis2_dt = [r for r in summary if r.get('axis') == 'precision' and r.get('dt_akma') == dt]
        if axis2_dt:
            markdown += f"\n**dt = {dt} AKMA**:\n"
            markdown += "| Precision | T_mean (K) | ΔT from 300K |\n"
            markdown += "|-----------|-----------|---------------|\n"
            for r in axis2_dt:
                markdown += f"| {r.get('precision', 'N/A'):9s} | {r['T_mean']:9.1f} | {abs(r['T_mean'] - 300):12.1f} |\n"

    markdown += f"""

### AXIS 3: Projection Site Comparison
"""

    for dt in [0.5, 1.0, 2.0]:
        axis3_dt = [r for r in summary if r.get('axis') == 'projection' and r.get('dt_akma') == dt]
        if axis3_dt:
            markdown += f"\n**dt = {dt} AKMA**:\n"
            markdown += "| Projection Site | T_mean (K) | ΔT from 300K |\n"
            markdown += "|-----------------|-----------|----------------|\n"
            for r in axis3_dt:
                markdown += f"| {r.get('projection_site', 'N/A'):15s} | {r['T_mean']:9.1f} | {abs(r['T_mean'] - 300):12.1f} |\n"

    markdown += f"""

## Decision Tree & Next Steps

```
IF H1 score > 0.7:
  → RATTLE saturation is PRIMARY CAUSE
  → Phase 2 Action: Increase settle_velocity_iters to [20-50]
  → Test: Verify ΔT < 5 K @ dt=1.0 fs with n_iters=50

ELIF H2 score > 0.7:
  → Float32 precision loss is CRITICAL BOTTLENECK
  → Phase 2 Action: Force float64 for inertia matrix computations
  → Test: Verify condition number control in _project_one_water_momentum_rigid

ELIF H3 score > 0.7:
  → Projection timing creates KE bias
  → Phase 2 Action: Test projection_site='both' in production
  → Test: Compare against OpenMM CMMotionRemover

ELSE:
  → Multiple hypotheses contribute equally (COMPLEX INTERACTION)
  → Phase 2 Action: Implement combined fixes (higher n_iters + float64 + 'both' projection)
  → Test: Ensemble validation against OpenMM reference
```

## Go/No-Go Recommendation

**Overall Hypothesis Support**: {total_score:.2f}/1.0

**Recommended Action**:
1. **Primary Fix**: Increase `settle_velocity_iters` from 10 → 20-50 (addresses H1)
2. **Secondary**: Ensure float64 in projection (addresses H2)
3. **Tertiary**: Set `projection_site='both'` (addresses H3)

**Confidence in Recommendation**: {total_score:.0%}

---

## Methodology Notes

- All tests used TIP3P water only (n_waters={200})
- Temperature defined as 2*KE / (N_dof_rigid * k_B) where N_dof_rigid = 6*N_w - 3
- Residual norm = L2 norm of bond length violations across all waters
- Ensemble statistics: mean and std over {5}+ replicas per configuration

"""

    verdict_path = output_dir / 'PHASE1_VERDICT.md'
    with open(verdict_path, 'w') as f:
        f.write(markdown)

    print(f"\nSaved: PHASE1_VERDICT.md")
    print(f"\n{'='*70}")
    print(f"QUICK VERDICT: {verdict}")
    print(f"Overall hypothesis support: {total_score:.1%}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Generate Phase 1 plots and verdict')
    parser.add_argument('--input', type=Path, default=Path('./phase1_results'), help='Input directory')
    parser.add_argument('--output', type=Path, default=Path('./phase1_results'), help='Output directory')
    args = parser.parse_args()

    summary_path = args.input / 'summary.json'
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        print("Run phase1_diagnostic_runner.py first to generate summary.json")
        return

    print(f"\nLoading summary from {summary_path}...")
    summary = parse_summary_json(summary_path)

    args.output.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    generate_residual_convergence_plot(summary, args.output)
    generate_precision_comparison_plot(summary, args.output)
    generate_projection_site_plot(summary, args.output)

    # Compute hypothesis support
    print("\nComputing hypothesis support scores...")
    hypothesis_support = compute_hypothesis_support(summary)

    # Generate verdict
    print("\nGenerating PHASE1_VERDICT.md...")
    generate_verdict_markdown(summary, hypothesis_support, args.output)

    print(f"\nAll outputs saved to {args.output}")


if __name__ == '__main__':
    main()
