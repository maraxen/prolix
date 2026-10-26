# tests/profiling/fixtures/

`b1_water_stage1.trace.json.gz` -- committed real CPU trace fixture for
`tests/profiling/test_trace_parse.py`'s "real" leg (P4's parser
ground-truth check, per
`.praxia/docs/specs/260817_jax-profiling-optimization-workflow.md` section
P4).

Produced locally on CPU with:

```bash
JAX_PLATFORMS=cpu uv run python scripts/experiments/profile_b1_water_trace.py \
    --trace --replicas 2 --n-steps 5 --n-trials 2
```

then copied from
`outputs/profiling/b1_water_trace_local/plugins/profile/<timestamp>/*.trace.json.gz`
(646 KB, under the 1 MB budget the reduced `--replicas`/`--n-steps` params
exist to keep it under -- the script's own defaults, `--replicas 16
`--n-steps 50 --n-trials 5`, do not fit).

`records/*.json` -- committed ProbeRecord fixtures for P8 (`test_report.py`).
Generated only via `ProbeRecord.write` (never hand-authored JSON). Round-trip
is asserted in `test_every_committed_fixture_roundtrips_probe_record_read`.

The matching compiled-HLO text (needed to build the thunk-name -> scope-path
map `xtrax.profiling.trace`'s parser requires) is NOT committed here --
it is regenerated at test time by re-calling
`profile_b1_water_trace.py`'s own `_build_water_plan`/`_build_batched_callable`
helpers with the SAME `--replicas 2 --n-steps 5` parameters and
`.lower().compile().as_text()`, matching this fixture's "no re-derivation of
jax.profiler.trace mechanics" reuse rule. Regenerating the (cheap,
no-execution) HLO text is preferable to committing a second, much larger
(~2.5 MB) artifact whose only content is derivable from the same script that
already produced the first.

**Stated limitation** (per spec P4): this is a CPU trace, so it validates
the parser against the CPU artifact shape only. P7 carries its own
additional gate leg running the parser against a real GPU trace.
