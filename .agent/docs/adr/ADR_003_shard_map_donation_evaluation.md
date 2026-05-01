# ADR 003: `shard_map` and buffer donation for batched Prolix MD

## Status

Evaluation notes (2026-04-29). No default behavior change.

## kUPs reference

- `kups/application/utils/propagate.py` uses `jit(..., donate_argnums=(1,))` on the **state** argument of a **Python** `for` loop over steps.
- `data_parallelism_vmap` / `shard_map` helpers exist for multi-device parallelism (see installed `kups` package).

## Prolix observations

- `lax.scan` over `LangevinState` with `donate_argnums=(0,)` on an outer `jit` produced **“Some donated buffers were not usable”** on the SETTLE Step-2 harness (heterogeneous carry + scan).
- **Recommendation**: treat **donation** as an **opt-in** perf knob after **correctness** and **compile memory** are stable on Engaging; prefer **Python chunk loops + `jit(step_fn)`** when debugging LLVM OOM during compile.

## Next experiments (when scheduled)

1. Profile **compile time vs peak host RAM** for `batched_produce` with `chunk_size` 1 vs full `vmap`.
2. If multi-device: prototype **`shard_map`** over batch dimension **only** on the energy/force kernel, not the full integrator scan, to limit communication surface.
3. Re-test `donate_argnums` only on **single-replica** `jit(scan)` with static metadata.
