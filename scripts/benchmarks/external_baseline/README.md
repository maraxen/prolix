# §7.1 External Baseline Harness

Each comparator gets its own bench script + bath sidecar (`.bth.toml`).
All scripts share the campaign tag `hp4-s71-external-baseline`.

## Slate (per recon, 2026-05-21)

| Tool | Framework | Harness file | Status |
|---|---|---|---|
| prolix (v1.2 Bundle code) | JAX/Equinox | `bench_prolix.py` | ✅ smoke done (14.9 µs/mol-step, N=64 f32) |
| DMFF | JAX | `bench_dmff.py` | TODO |
| TorchMD | PyTorch (eager) | `bench_torchmd.py` | TODO |
| espaloma | PyTorch + DGL | `bench_espaloma.py` | TODO |
| ForceBalance | Newton-Raphson + FD | `bench_forcebalance.py` | TODO (Scope C only) |

Note: PyTorch-from-scratch was removed (2026-05-22). It is a strawman — TorchMD is the
real, citable PyTorch comparator. A from-scratch impl would only demonstrate we wrote
it carelessly; it does not support the heterogeneous-batching substrate claim.

Captioned-only (not benchmarked, just referenced):
- TorchANI (NN potential, different objective)
- kUPS (JAX batched MD, no FF fitting)
- JAX-MD (general MD, no FF fitting)

## Scope

**Scope A — primitive** (forward + backward of bonded energy across N mols, mixed sizes)
- Same params, same positions, same topology fed to each tool
- Measure: per-mol-step wall-clock, peak memory
- All tools except ForceBalance can participate (no autograd in ForceBalance)
- N ladder: {16, 32, 64, 128, 256, 512} matching internal sweep

**Scope C — native** (each tool's idiomatic training step)
- Each tool runs its own training loop; report seconds/mol/step
- Objective-difference caveat in figure caption (espaloma trains a GNN; prolix direct-fits)
- ForceBalance enters here as ecological floor (FD convergence vs autograd)

## Common fixture

Same molecules across all tools: ANI-1x subset, 16 base mols (the existing
`data/ani1x_subset/lane_a/mol_*.params_init.json` set). Lane B held-out (4 mols).
Replicate base set as needed for N > 16 (matching the internal sweep methodology).

## Environment management

Each comparator may have conflicting deps. Use separate uv project envs per
harness directory, or use uv's `--with` overlays. DO NOT pollute prolix's main
env with TorchMD or DGL.

```bash
# Example pattern (one env per tool):
cd scripts/benchmarks/external_baseline
uv venv .venv-dmff && uv pip install --python .venv-dmff/bin/python jax dmff
uv venv .venv-torchmd && uv pip install --python .venv-torchmd/bin/python torch torchmd
# etc.
```

## Pre-reg discipline

Per `using-bathos` "Research Before Finalizing":
1. Smoke each comparator at N=64 (one-shot, just confirm it runs)
2. Anchor thresholds in `hp4_s71_external_baseline.bth.toml` based on smoke numbers
3. Then submit the full campaign sweep

DO NOT submit the full sweep until thresholds are anchored. The sidecar's
current threshold values are TODO placeholders.
