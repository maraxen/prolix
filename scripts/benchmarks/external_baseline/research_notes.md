# Pre-reg research notes — what we need to measure to finalize thresholds

Per `using-bathos` "Research Before Finalizing Pre-Registration": the sidecar's
thresholds must be **anchored in evidence**, not guessed. This doc tracks what
research is still needed before the campaign can run for real.

## Anchoring questions (all need empirical answers)

1. **What is the prolix v1.2 Bundle-code per-mol-step wall-clock at N=64 on RTX Pro 6000?**
   - Current evidence: pre-Bundle code measured ~14.5s / 500 steps for N=64 batched on the same hardware (sweep cell, 2026-05-20).
   - Bundle-code may differ. Need fresh single-cell smoke.
   - Action: submit `bench_prolix.py --n-mols 64 --n-steps 500` to cluster (single cell, ~1 min).

2. **What is DMFF's per-mol-step on the same primitive?**
   - DMFF recon (2026-05-21): single-mol fitting only; no batched loss.
   - For N mols, expected scaling: N separate forward+backward passes.
   - Build minimal harness using `dmff.api.Hamiltonian` + `optax` per-mol loop. Time at N=64.
   - Action: write `bench_dmff.py`, smoke at N=16 first (sanity), then N=64.

3. **What is TorchMD's per-mol-step?**
   - TorchMD recon (2026-05-21): Python for-loop in `torchmd/forces.py:102-116`.
   - Build harness: `Forces` object + `torch.autograd.grad` per system, looped.
   - Action: `bench_torchmd.py`, smoke at N=16, then N=64.

4. **What is espaloma's per-mol-step on bonded-energy forward+backward?**
   - Espaloma recon: DGL batched heterograph; `energy_in_graph` at `mm/energy.py:248-393`.
   - We want JUST the energy forward+backward, NOT the full GNN training step.
   - Action: `bench_espaloma.py`, drive `energy_in_graph` directly with pre-built batched heterograph.

5. **Does ForceBalance even fit a 500-step Adam-equivalent comparison?**
   - ForceBalance uses Newton-Raphson + FD Hessian; converges in ~10 iterations.
   - Each iteration cost is much higher than an Adam step.
   - Action: skip Scope A (no autograd). For Scope C: measure wall-clock to converge to a fixed residual on a 16-mol target. Different methodology, report as ecological floor.

## Smoke plan (~1 day of work)

| Step | Action | ETA |
|---|---|---|
| 1 | Set up per-tool venvs in `scripts/benchmarks/external_baseline/.venv-*` | 1h |
| 2 | Write `bench_prolix.py` (copy/adapt fit_bonded_hp4.py logic) | 30min |
| 3 | Write `bench_dmff.py` (single-mol Adam loop using DMFF Hamiltonian) | 2h |
| 4 | Write `bench_torchmd.py` (per-system Adam loop) | 2h |
| 5 | Write `bench_espaloma.py` (drive `energy_in_graph` + per-mol Adam) | 2h |
| 6 | Smoke each at N=16 locally (CPU, just confirms it runs) | 1h |
| 7 | Cluster smoke at N=64 (one cell per tool, pinned node4007/4008) | 30min wait |
| 8 | Anchor thresholds in sidecar based on N=64 numbers | 30min |

After step 8: full campaign sweep at N ∈ {16, 32, 64, 128, 256, 512} × {f32, f64}
for the autograd-capable tools (4 tools × 12 cells = 48 cells per Scope A).

## Threshold-anchoring math (placeholder)

Once smoke numbers land, plug into:

```
prolix_n64 = T_p  # measured: 14.9 µs/mol-step (run 6ce95e65, job 14263023)
dmff_n64 = T_d    # TODO: smoke bench_dmff.py
torchmd_n64 = T_t # TODO: smoke bench_torchmd.py
espaloma_n64 = T_e # TODO: smoke bench_espaloma.py

# Conservative pass threshold (50% better than fastest non-batched competitor)
# PyTorch-from-scratch removed (2026-05-22): not a citable comparator; TorchMD covers PyTorch slot
pass_threshold = 0.5 * min(T_d, T_t)  # exclude espaloma (it has DGL batching)

# Marginal: between 0.5× and 1.0×
marginal_threshold = 1.0 * min(T_d, T_t)
```

If actual prolix < pass_threshold → claim succeeds.
If pass_threshold ≤ prolix < marginal_threshold → claim narrows to "competitive".
If prolix ≥ marginal_threshold → claim collapses (Scope B fallback).

## Hardware pin (mandatory)

All cluster submissions:
```
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4007,node4008
#SBATCH --gres=gpu:1
```

Pin to RTX Pro 6000 Blackwell only. Exclude node4009 (H200) — heterogeneous GPU
would confound the comparison.

## Pre-reg validity contract

This campaign **CANNOT BEGIN** (no `bth run` against the sidecar) until:
- [ ] All comparator harnesses smoke successfully at N=16 locally
- [ ] All comparator harnesses smoke successfully at N=64 on cluster
- [ ] Thresholds in sidecar are replaced with anchored values
- [ ] The sidecar's `outcomes.pass.reasoning` cites the smoke run's git_hash
- [ ] Sidecar is re-reviewed (by oracle if novel territory, otherwise self-check)

Until those gates pass, the sidecar's `outcomes.pass.condition` uses the
conservative bias-toward-not-falsely-claiming-success: if any comparator's smoke
fails to produce a valid number, the pass condition cannot be evaluated and the
residual `fail` branch fires.
