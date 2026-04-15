# Engaging cluster: chignolin → DHFR benchmarks

Operational checklist for **MIT Engaging** (or any SLURM site). This repo cannot submit jobs from CI; run `rsync` + `sbatch` from a login node or your workstation with SSH access.

## Prerequisites

- Layout on the cluster: sibling checkouts `~/projects/prolix` and `~/projects/proxide`, with the **UV workspace root** at `~/projects/` (`pyproject.toml` + `uv.lock` list members `prolix` and `proxide`). `just push-engaging` syncs `workspace/pyproject.toml` and `workspace/uv.lock` from this repo to `~/projects/`, rsyncs `../proxide` when present locally, and pushes the prolix tree to `~/projects/prolix`. Set `ENGAGING_REMOTE_DIR` / `ENGAGING_WORKSPACE_REMOTE_DIR` if your paths differ.
- **Environment (workspace install):** from the workspace root, not from inside `prolix` alone:

  ```bash
  cd ~/projects
  uv sync --extra cuda --extra dev --package prolix
  ```

  Or after a push: `just sync-uv-engaging`. The virtualenv is created at `~/projects/.venv`; `uv run` from `~/projects/prolix` still resolves it. OpenMM is **not** in the `dev` extra — PyPI wheels target `manylinux_2_34`; older nodes like `manylinux_2_28` cannot install it. Parity tests that need OpenMM skip if missing. For OpenMM on the cluster, use conda or a host with a compatible glibc, then `uv sync --extra openmm` if wheels match.
- **Proxide:** declared as a workspace member in `prolix`’s `[tool.uv.sources]` (`proxide = { workspace = true }`). MDCompress (`rust_mdc`) is **not** part of default `oxidize` builds anymore — only the sibling `proxide` checkout is required.
- If a previous sync failed mid-way: `rm -rf ~/projects/.venv && uv sync ...` from `~/projects` (or from `~/projects/prolix` — `uv` discovers the workspace).
- Record versions in `run_manifest.json` (written automatically by SLURM scripts).
- Force field: `protein.ff19SB.xml` from a sibling `proxide` checkout, an installed `proxide` wheel, or `openmmforcefields/.../protein.ff19SB.xml` (see `scripts/run_batched_pipeline.py`).

## Local verification (before first cluster submit)

From the repo root on your laptop or a login node:

```bash
# Quick: manifest + pytest only (seconds to minutes)
ENGAGING_LOCAL_QUICK=1 bash scripts/verify_engaging_local.sh

# Full: same steps as combined SLURM template (includes GPU benchmarks; longer)
bash scripts/verify_engaging_local.sh
```

This writes `outputs/logs/engaging/<date>/run_manifest.json` and exercises the same commands as `scripts/slurm/bench_chignolin_pi_so3.slurm` (without `sbatch`). **JAX/CUDA/OpenMM versions** appear in `run_manifest.json` (and `jax`/`openmm` imports must succeed).

## Cluster smoke checklist (manual)

CI cannot SSH to Engaging. After `just push-engaging`:

1. `just submit-bench-chignolin` **or** `just submit-bench-chignolin-split` (accuracy then speed).
2. Confirm `outputs/logs/engaging/<date>/run_manifest.json` and `slurm/*.out` on the cluster.
3. `just pull-logs-engaging` and archive or attach log excerpts for your lab notes.

## Log layout

On the cluster (or after `just pull-logs-engaging`):

```text
outputs/logs/engaging/
  YYYYMMDD/
    run_manifest.json
    slurm/
      *.out / *.err
    app/
      chignolin_accuracy_<job>.log
      chignolin_speed_openmm_<job>.log
      chignolin_nlvsdense_<job>.log
```

`ENGAGING_LOG_DATE` defaults to `YYYYMMDD` (UTC) when running `scripts/write_engaging_manifest.py` via `scripts/slurm/_common_env.sh`.

### `run_manifest.json` schema (informal)

| Field | Meaning |
|-------|---------|
| `timestamp_utc` | When the manifest was written |
| `git_sha`, `git_branch` | Repository state |
| `slurm_job_id`, `slurm_array_*` | SLURM IDs when applicable |
| `slurm_job_partition` | Queue name |
| `jax_version`, `jaxlib_version` | JAX stack |
| `openmm_version` | OpenMM if importable (else `null`) |
| `engaging_log_date` | Same as directory name |
| `prolix_root` | Absolute path used for the run |

## Partitions (site-specific)

| Partition | Typical use |
|-----------|-------------|
| `pi_so3` | Short GPU smoke, debugging (`srun`, interactive) |
| `mit_preemptable` | Longer sweeps; assume preemption — use checkpointed workflows |

Override the default in `just submit-bench-chignolin` via `ENGAGING_PARTITION` (passed to `sbatch --partition`).

Some sites require **additional** `#SBATCH` lines (account, QoS, constraints). Add them to the templates or pass `sbatch --wrap` — the shipped files are examples only.

## Phase 1 — Chignolin (prove the pipeline)

1. **Sync:** `just push-engaging` (or `rsync` with the same filters as in the `Justfile`).
2. **Submit:** `just submit-bench-chignolin` (combined `scripts/slurm/bench_chignolin_pi_so3.slurm`) **or** `just submit-bench-chignolin-split` (accuracy job `bench_chignolin_accuracy_pi_so3.slurm`, then speed `bench_chignolin_speed_pi_so3.slurm` via `afterok`). Or run `bash scripts/slurm/submit_chignolin_split_pi_so3.sh` on the login node after `cd` to the repo.
3. **Follow logs:** `just logs-engaging` or `tail` under `outputs/logs/engaging/<date>/slurm/`.
4. **Pull:** `just pull-logs-engaging` for JSON + text logs (filters large binaries).

Scripts involved:

- Accuracy (subset): `tests/physics/test_pbc_end_to_end.py -m "not slow"` (edit the SLURM file for your time budget).
- Throughput (not parity): `scripts/benchmarks/prolix_vs_openmm_speed.py` — minimal PME box; see `docs/source/explicit_solvent/explicit_solvent_benchmarks.md`.
- NL vs dense: `scripts/benchmark_nlvsdense.py` with `SYSTEM_CATALOG` keys `CHIGNOLIN`, `1X2G` (see `scripts/run_batched_pipeline.py`).
- Optional: `scripts/profile_step.py` or JAX profiler → save under `app/`.

## Phase 2 — DHFR (scale)

- Add a canonical DHFR structure as `data/pdb/4m8j.pdb` (or your lab’s PDB ID). Example download (verify RCSB terms and cite the PDB ID in publications):

  ```bash
  PDB_ID=4m8j bash scripts/fetch_pdb_example.sh
  ```

  The catalog key `DHFR` in `scripts/run_batched_pipeline.py` prefers `4m8j.pdb` when present; otherwise it falls back to `1UBQ.pdb` as a **large-protein proxy** — label results accordingly.
- Reuse the same manifest + log layout. Use `scripts/slurm/bench_dhfr_preemptable.slurm` (or adapt) for `benchmark_nlvsdense.py --systems DHFR`.
- **Optional OpenMM spot-check:** run `benchmarks/compare_jax_openmm_validity.py` or `benchmarks/verify_pbc_end_to_end.py` with the **same** topology, box, and force field as your OpenMM reference; document tolerances in your lab notes. Full-protein parity tests may not cover production DHFR geometry — treat explicit solvent benchmarks as release gates, not as a substitute for system-specific validation.

## Code entrypoints

| File | Role |
|------|------|
| `scripts/run_batched_pipeline.py` | `SYSTEM_CATALOG`, `load_and_parameterize`, `prepare_batches` |
| `scripts/write_engaging_manifest.py` | Manifest JSON |
| `scripts/slurm/*.slurm` | SLURM templates (combined, split accuracy/speed, array, DHFR) |
| `scripts/slurm/submit_chignolin_split_pi_so3.sh` | Chained `sbatch` with `afterok` |
| `scripts/verify_engaging_local.sh` | Local dry-run of chignolin benchmark steps |
| `scripts/fetch_pdb_example.sh` | Example RCSB download for `data/pdb/` |
| `Justfile` | `login-engaging`, `push-engaging-workspace`, `push-engaging`, `sync-uv-engaging`, `submit-bench-*`, `submit-bench-chignolin-split`, `logs-engaging`, `pull-logs-engaging` |

## Related docs

- `docs/source/explicit_solvent/explicit_solvent_benchmarks.md` — local vs cluster matrix
- `docs/source/explicit_solvent/current_implementation.md` — parity tests table
- `docs/source/explicit_solvent/explicit_solvent_implementation_plan.md` — checkpointing / preemptable notes
