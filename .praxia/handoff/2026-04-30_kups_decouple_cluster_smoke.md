## Session Summary
**Date**: 2026-04-30 08:13 (UTC-7)
**Primary Goal**: Complete the Prolix-side convergence work without introducing a runtime dependency on kUPs, validate batched paths, and submit cluster smoke jobs.

### Key Decisions
- **Keep Prolix runtime independent of kUPs package**: kUPs remains optional reference context/tests only; production paths and docs are framed as Prolix/JAX-MD-native.
- **Standardize streaming callback contract**: all streaming producers should pass only trajectory payload + indices to host callbacks (`positions`, `batch_idx`, `save_idx/start_save_idx`), not full state tensors.
- **Prefer deterministic dtype carries in JAX loops**: explicit casting in `lax.scan`/`while_loop` carry paths prevents float32/float64 carry mismatches under cluster/local differences.
- **Use dual cluster smokes**: quick gate on `mit_quicktest` and headroom gate on `mit_preemptable` for cold-JIT compile spikes.

### Changes Made
- `/home/marielle/projects/prolix/src/prolix/batched_simulate.py:187` - `apply_com_correction` now skips step 0 (`do_recenter = jnp.logical_and(step_idx > 0, ...)`) to keep streaming/non-streaming parity on cold start.
- `/home/marielle/projects/prolix/src/prolix/batched_simulate.py:1014` - `batched_equilibrate` schedule tensors pinned to `positions.dtype` (`pos_dtype`) for `while_loop` carry stability.
- `/home/marielle/projects/prolix/src/prolix/batched_simulate.py:1054` - equilibration velocity init now uses `eq_dtype` and typed `jax.random.normal(..., dtype=eq_dtype)`.
- `/home/marielle/projects/prolix/src/prolix/batched_simulate.py:1384` - batched `io_callback` path (`write_batch_size > 1`) now calls `io_callback(write_fn, None, frames, batch_idx, start_save_idx)`.
- `/home/marielle/projects/prolix/src/prolix/batched_simulate.py:1811` - NL streaming callback reduced to `(s_next.positions, batch_idx, save_idx)`.
- `/home/marielle/projects/prolix/src/prolix/batched_simulate.py:1982` - NL-dynamic streaming callback reduced to `(s_next.positions, batch_idx, save_idx)`.
- `/home/marielle/projects/prolix/src/prolix/physics/flash_nonbonded.py:174` - cast `tile_I` to carry dtype in Born-radius scan.
- `/home/marielle/projects/prolix/src/prolix/physics/flash_nonbonded.py:299` - cast tile energy to carry dtype in fused-energy scan.
- `/home/marielle/projects/prolix/tests/physics/test_batched_simulate.py:23` - force-field path now resolves via installed proxide package (`Path(proxide.__file__).resolve().parent / "assets" / "protein.ff19SB.xml"`).
- `/home/marielle/projects/prolix/tests/physics/test_batched_simulate.py:181` - `test_batched_minimize` uses short smoke schedule (`fire_stage_steps=(2,2,2,2)`, `lbfgs_steps=0`) to keep CI/cluster compile tractable.
- `/home/marielle/projects/prolix/tests/physics/test_batched_simulate.py:343` - added `test_batched_produce_streaming_write_batch_size` validating batched callback payload shape and parity vs `batched_produce`.
- `/home/marielle/projects/prolix/tests/physics/test_kups_thermostat_crossval.py:37` - marked with `pytestmark = pytest.mark.kups` (optional dependency bucket).
- `/home/marielle/projects/prolix/pyproject.toml:82` - added pytest marker description for optional `kups` tests.
- `/home/marielle/projects/prolix/Justfile:164` - added `submit-batched-smoke`.
- `/home/marielle/projects/prolix/Justfile:168` - added `submit-batched-smoke-preemptable`.
- `/home/marielle/projects/prolix/scripts/slurm/smoke_batched_simulate_cpu.slurm:6` - quicktest smoke script (`mit_quicktest`, time `00:14:00` to satisfy 15-min cap).
- `/home/marielle/projects/prolix/scripts/slurm/smoke_batched_simulate_cpu_preemptable.slurm:6` - preemptable smoke script (`mit_preemptable`, `01:00:00`).
- `/home/marielle/projects/prolix/.agent/docs/CLUSTER_CONFIG.md` - documented new local `just` smoke flow.

### Current State
- **Working**:
  - Local: `uv run pytest tests/physics/test_batched_simulate.py -q` passes (`8 passed`).
  - Local: `uv run pytest tests/physics/test_step_result.py tests/physics/test_md_potential_bundle_parity.py -q` passes (`5 passed`).
  - Cluster quick smoke (job `12860383`) passed (`12 passed`).
  - Cluster preemptable smoke (job `12866902`) passed (`13 passed`, ~58.60s), including new batched streaming test count.
- **Broken**:
  - No active red test in this session after fixes.
  - Non-fatal warnings still present (haiku deprecation transitively from `jax-md`, cmap float64 truncation warning, SWIG/importlib deprecations).
- **In Progress**:
  - Repo is intentionally dirty with many pre-existing modified/untracked files; no commit created in this session.

### Blockers
- **No functional blocker for current batched smoke scope.**
- **Potential future blocker if warnings policy tightens**: current smoke logs include deprecation warnings from transitive deps (`dm-haiku` via `jax-md`) and dtype warnings from `cmap.py`.

### Next Steps
1. Keep both smoke gates in routine:
   - `just submit-batched-smoke` for fast checks.
   - `just submit-batched-smoke-preemptable` when compile/runtime headroom is needed.
2. If warning budget matters, decide policy:
   - suppress known third-party warnings in test config, or
   - upgrade/depin transitive stack (`jax-md` ecosystem) in a controlled branch.
3. Pick one physics milestone (not framework work) as next sprint focus:
   - NPT long-trajectory stability, or
   - batched equilibration robustness, or
   - neighbor-list invalidation policy from ADR 002.
4. If desired, rename historically kUPs-named docs/tests to Prolix naming while preserving intent.

### Critical Context
- Cluster submissions executed:
  - `just push-engaging` (successful sync).
  - `just submit-batched-smoke-preemptable` -> job `12866902` (passed).
- Useful status/log commands:
  - `ssh engaging 'squeue -j 12866902'`
  - `ssh engaging 'tail -80 ~/projects/prolix/outputs/logs/engaging/20260429/slurm/plx-batched-smoke_12866902.out'`
  - `ssh engaging 'tail -60 ~/projects/prolix/outputs/logs/engaging/20260429/app/batched_simulate_smoke_preempt_12866902.log'`
- Dependency context for haiku warning:
  - `/home/marielle/projects/prolix/uv.lock:1055` shows `jax-md` dependency set includes `dm-haiku` and `e3nn-jax` transitively.
