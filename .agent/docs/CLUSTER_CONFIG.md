# Cluster Configuration

This project uses the Engaging SLURM cluster for large-scale simulations and data processing.

## Defaults

- **Partition**: mit_preemptable (48h walltime, no GPU required)
- **Workspace**: ~/projects/prolix
- **GPU Count**: 0 (CPU-only, adjust if needed)
- **Output Directory**: outputs/logs/engaging/

## Prolix repo (local `Justfile`)

After `uv sync` locally, push and submit a short CPU smoke (batched tests + bundle/step_result):

```bash
just login-engaging    # optional
just push-engaging
just submit-batched-smoke
```

Slurm scripts: `scripts/slurm/smoke_batched_simulate_cpu.slurm` (`mit_quicktest`, ≤ 15 min), or
`smoke_batched_simulate_cpu_preemptable.slurm` via `just submit-batched-smoke-preemptable` (1 h, `mit_preemptable`).

## Quick Commands

```bash
# Login and establish SSH control master
just -g cluster-login engaging

# Sync workspace (pyproject.toml, uv.lock, source code)
just -g cluster-push-workspace prolix engaging

# Submit a job
just -g cluster-submit prolix sbatch_script.sh "--partition=mit_preemptable --time=12:00:00"

# Check queue
just -g cluster-queue engaging

# View job logs
just -g cluster-logs prolix 5 engaging
```

## Custom Configuration

If this project requires non-standard settings:

- **GPU Jobs**: Use `mit_normal_gpu` partition (6h max walltime, 64 nodes available)
- **Long CPU Jobs**: Use `pit_so3` partition (48h max, selective use) or `mit_normal` (12h max, unlimited)
- **Fast Testing**: Use `mit_quicktest` partition (15 min max, 26 nodes)

Document any custom rsync filters or job array specifications below:

[Add project-specific notes here]

## Reference

- Global cluster reference: ~/.claude/CLUSTER_INFRASTRUCTURE.md
- Global recipes: `just -g cluster-*`
- Cluster rules: ~/.claude/rules/CLUSTER.md
