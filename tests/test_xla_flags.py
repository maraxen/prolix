"""Tests for XLA flag recipe merge (no JAX)."""

from prolix.xla_flags import INTENSIVE_GPU_FLAGS, merge_xla_flags


def test_merge_appends_missing_and_keeps_existing():
    merged = merge_xla_flags("--xla_gpu_shard_autotuning=false", INTENSIVE_GPU_FLAGS)
    assert "--xla_gpu_shard_autotuning=false" in merged
    for flag in INTENSIVE_GPU_FLAGS:
        assert flag in merged
    again = merge_xla_flags(merged, INTENSIVE_GPU_FLAGS)
    assert again.split() == merged.split()
