"""XLA runtime flag recipes — compiler lowering, not problem tiling.

High-level tiling (ATOM_BUCKETS, flash ``T``, ``lax.scan`` over tiles) chooses
the *shape of the program*: how many ghost atoms, how the pair loop is chunked.
XLA flags only pick implementations for ops already in that graph (GEMM
algorithms, Triton tile sizes, while-loop double buffering, latency hiding).

``--xla_gpu_autotune_level`` already defaults to 4 (full GEMM/conv autotune).
Raising it further is not the intensive knob. ``exhaustive_tiling_search``
expands the *Triton GEMM* search; it will not rewrite FlashMD's (N, T)
elementwise fusion into a neighbor list.
"""

from __future__ import annotations

# Set before importing JAX. Command buffers stay on (production graphs).
INTENSIVE_GPU_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_while_loop_double_buffering=true",
    "--xla_gpu_exhaustive_tiling_search=true",
    "--xla_gpu_triton_gemm_any=true",
)


def merge_xla_flags(existing: str, extra: tuple[str, ...] | list[str]) -> str:
    """Append flags that are not already present (substring match on the token)."""
    parts = [p for p in existing.split() if p]
    have = set(parts)
    for flag in extra:
        if flag not in have:
            parts.append(flag)
    return " ".join(parts)
