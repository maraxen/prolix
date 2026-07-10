---
title: "XR-PARITY-TORCH — external_baseline rewire regression"
backlog_id: XR-PARITY-TORCH
epic: 260709_xtrax_rewire
depends_on: [XR-PARITY-OMM-WATER]
priority: P2
difficulty: standard
status: completed
challenge_verdict: pass
challenge_summary: "Re-homed prolix Scope A as confirmation sidecar; bathos pass (29c5f27f); planner.plan→xtrax; TorchMD out of CI."
---

# XR-PARITY-TORCH

## Goal
Re-home the §7.1 `external_baseline` prolix Scope A primitive as a **rewire regression** bathos confirmation — secondary, not Claim-1 throughput and not on the kill path.

## Locked decisions

| Topic | Lock |
|-------|------|
| Always-on probe | Prolix Scope A: bonded forward+backward+Adam step on `data/ani1x_subset/lane_a`, small N (default 4) |
| Pass gates | `per_mol_step_seconds > 0` AND `0 < final_loss < 1e10` (same process gates as `bench_prolix.bth.toml`) |
| Structural gate | `BatchPlanner.plan` is an alias of `plan_with_xtrax` (FIT-FLIP / KILL-FORK) |
| Driver | `scripts/experiments/xr_parity_torch.py` + `.bth.toml` (wraps `bench_prolix.bench_one_step`) |
| Campaign slug | `xr-parity-torch` |
| TorchMD / DMFF / espaloma | **Out of CI** — optional cluster comparators remain under `scripts/benchmarks/external_baseline/`; this leaf does not require `torch` |
| Non-goals | Claim-1 ns/day; §7.1 2× speedup thresholds; installing TorchMD in default env |

## Acceptance Criteria
1. Given lane_a fixtures, when adapter/prolix smoke probe runs via `bth run`, then `gate_pass=1` with process gates above.
2. Given `BatchPlanner.plan`, when inspected/tested, then it delegates to `plan_with_xtrax`.
3. Given Claim-1 / comparator throughput scope, when reviewed, then rejected for this leaf.

## References
- `scripts/benchmarks/external_baseline/bench_prolix.py`
- `scripts/benchmarks/external_baseline/bench_prolix.bth.toml`
- Water/KUPS templates under `scripts/experiments/xr_parity_*.bth.toml`
