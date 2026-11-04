"""Prolix StableHLO export utilities for WASM/WebGPU deployment.

Wraps jax.jit lowering to produce portable StableHLO artifacts from proxide
energy functions built with the explicit-params API (make_energy_fn_pure).

v1.1 Item 4: unblocked by EnergyParams / make_energy_fn_pure (Item 1).
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import pathlib
import shutil
import subprocess
from typing import Any

import jax
from jax.experimental.pallas import triton as plt

from prolix.types.integrators import LangevinState

_IN_EXPORT_CONTEXT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_IN_EXPORT_CONTEXT", default=False
)


def in_export_context() -> bool:
  """Return True when currently lowering inside export_context()."""
  return _IN_EXPORT_CONTEXT.get()


@contextlib.contextmanager
def export_context():
  """Context manager forcing pure-JAX HLO lowering (disabling Pallas/Triton calls)."""
  token = _IN_EXPORT_CONTEXT.set(True)
  try:
    yield
  finally:
    _IN_EXPORT_CONTEXT.reset(token)


def resolve_pallas_compiler_params(
    kernel_kind: str = "default",
    grid_blocks: int = 0,
    *,
    device_kind: str | None = None,
    compute_capability: float | None = None,
) -> Any:
  """Resolve Triton ``CompilerParams`` across GPU architectures (Ampere, Ada, Hopper, Blackwell).

  Adapts ``num_warps`` and ``num_stages`` based on ``kernel_kind``, total launched CTAs
  (``grid_blocks``), and GPU architecture so kernels neither over-allocate pipeline stages
  on small-grid buckets nor under-utilize warp schedulers on large multi-replica batches.
  Supports explicit environment overrides via ``PROLIX_PALLAS_NUM_WARPS`` and
  ``PROLIX_PALLAS_NUM_STAGES``.
  """
  env_warps = os.environ.get("PROLIX_PALLAS_NUM_WARPS", "").strip()
  env_stages = os.environ.get("PROLIX_PALLAS_NUM_STAGES", "").strip()

  cc = compute_capability
  dk = (device_kind or "").upper()
  if cc is None or not dk:
    try:
      if jax.default_backend() == "gpu":
        dev = jax.devices("gpu")[0]
        if cc is None:
          raw_cc = getattr(dev, "compute_capability", None)
          if raw_cc is not None:
            cc = float(raw_cc)
        if not dk:
          dk = str(getattr(dev, "device_kind", "")).upper()
    except Exception:
      pass

  if cc is None:
    if any(k in dk for k in ("B100", "B200", "B300", "GB200", "BLACKWELL", "RTX 50", "PRO 6000")):
      cc = 10.0
    elif any(k in dk for k in ("H100", "H200", "GH200", "HOPPER")):
      cc = 9.0
    elif any(k in dk for k in ("L40", "L4", "RTX 40", "ADA")):
      cc = 8.9
    elif any(k in dk for k in ("A100", "A10", "A30", "A40", "RTX 30", "AMPERE")):
      cc = 8.0
    else:
      cc = 8.0

  if kernel_kind == "bonded_fused":
    num_warps = 2
    num_stages = 1
  elif kernel_kind in ("nonbonded_nl", "cell_prune", "pme"):
    num_warps = 4
    if 0 < grid_blocks < 256:
      num_stages = 1
    elif cc >= 9.0:
      num_stages = 2
    else:
      num_stages = 2
  else:
    num_warps = 4
    num_stages = 2

  if env_warps:
    num_warps = max(1, int(env_warps))
  if env_stages:
    num_stages = max(1, int(env_stages))

  return plt.CompilerParams(num_warps=num_warps, num_stages=num_stages)


# Claim 2 W3 gate (#277): roadmap target for browser-deployable artifacts.
WASM_ARTIFACT_MAX_BYTES = 50 * 1024 * 1024

# Default iree-compile flags for StableHLO → wasm32 (Claim 2 / IREE-WASM path).
_IREE_WASM_COMPILE_ARGS = (
    "--iree-input-type=stablehlo",
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
    "--iree-llvmcpu-target-triple=wasm32-unknown-unknown",
    "--iree-llvmcpu-target-cpu=generic",
)


def export_energy_fn(
    fn: Any,
    params: Any,
    example_positions: Any,
) -> Any:
  """Export an explicit-params energy function to a StableHLO artifact.

  Lowers fn(params, positions) -> float via jax.jit.lower(), producing a
  portable StableHLO artifact that can be compiled and called without
  retracing, or serialized to MLIR text for WASM/WebGPU deployment.

  Args:
      fn: Callable (params, positions) -> float from make_energy_fn_pure.
      params: EnergyParams (charges, sigmas, epsilons) — shapes/dtypes used
          as abstract traceable inputs.
      example_positions: (N, 3) array with correct dtype.

  Returns:
      jax._src.interpreters.mlir.LoweringResult with methods:
        .compile()      → CompiledFun callable with real data
        .as_text()      → str StableHLO / MLIR text
        .cost_analysis() → dict estimated FLOPs / memory

  Example::

      params, fn = make_energy_fn_pure(displacement_fn, sys_dict, box_vec)
      lowered = export_energy_fn(fn, params, positions)
      compiled = lowered.compile()
      energy = compiled(params, positions)
  """
  with export_context():
    return jax.jit(fn).lower(params, example_positions)


def export_langevin_step(
    step_fn: Any,
    config: Any,
) -> Any:
  """Wrap a Langevin step function for jax.export compatibility.

  The flat LangevinState signature (no Optional fields) makes this
  jax.export-compatible. Full StableHLO/WASM lowering happens in Phase 4.

  Args:
      step_fn: Function (LangevinState) -> LangevinState.
      config: Static IntegratorConfig (baked in at trace time).

  Returns:
      Callable with identical signature; JIT and vmap compatible.

  Example::

      from prolix.types.integrators import LangevinState, IntegratorConfig
      from prolix.export import export_langevin_step

      def step(state: LangevinState) -> LangevinState:
          # ... integration logic ...
          return new_state

      config = IntegratorConfig(
          thermostat="langevin",
          has_pbc=False,
          dt=0.5,
          kT=1.0,
          gamma=1.0,
      )

      exported = export_langevin_step(step, config)
      result = jax.jit(exported)(state)
  """

  def exported(state: LangevinState) -> LangevinState:
    with export_context():
      return step_fn(state)

  return exported


def save_artifact(artifact: Any, path: str | pathlib.Path) -> None:
  """Serialize a StableHLO artifact to an MLIR text file on disk.

  Args:
      artifact: LoweringResult from export_energy_fn.
      path: Destination file path (str or pathlib.Path).
  """
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(artifact.as_text())


def load_artifact(path: str | pathlib.Path) -> str:
  """Read a StableHLO MLIR text file from disk.

  Returns the raw MLIR text. To re-compile, call export_energy_fn again
  and use lowered.compile() — JAX has no standalone MLIR-text deserializer.

  Args:
      path: Source file path (str or pathlib.Path).

  Returns:
      str: StableHLO MLIR text.
  """
  return pathlib.Path(path).read_text()


def find_iree_compile() -> str | None:
  """Return path to ``iree-compile`` on PATH, or None if not installed."""
  return shutil.which("iree-compile")


def compile_stablehlo_mlir_to_wasm(
    mlir: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    iree_compile: str | None = None,
) -> pathlib.Path:
  """Compile StableHLO MLIR text to a wasm32 IREE artifact via ``iree-compile``.

  Requires the optional ``wasm`` extra (``iree-base-compiler``) or a system
  ``iree-compile`` on PATH.

  Args:
      mlir: StableHLO MLIR module text or path to a ``.mlir`` file.
      output: Destination ``.wasm`` (or ``.vmfb``) path.
      iree_compile: Optional explicit ``iree-compile`` binary path.

  Returns:
      Resolved output path.

  Raises:
      FileNotFoundError: ``iree-compile`` not found.
      subprocess.CalledProcessError: IREE compilation failed.
  """
  exe = iree_compile or find_iree_compile()
  if exe is None:
    msg = (
        "iree-compile not found on PATH; install with: "
        "uv sync --extra wasm  (provides iree-base-compiler)"
    )
    raise FileNotFoundError(msg)

  out_path = pathlib.Path(output)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  if isinstance(mlir, pathlib.Path):
    mlir_path = mlir
  elif isinstance(mlir, str):
    candidate = pathlib.Path(mlir)
    try:
      is_existing_file = candidate.is_file()
    except OSError:
      is_existing_file = False
    if is_existing_file:
      mlir_path = candidate
    else:
      mlir_path = out_path.with_suffix(".mlir")
      mlir_path.write_text(mlir)
  else:
    raise TypeError(f"mlir must be pathlib.Path or str, got {type(mlir)!r}")

  cmd = [exe, *_IREE_WASM_COMPILE_ARGS, str(mlir_path), "-o", str(out_path)]
  subprocess.run(cmd, check=True, capture_output=True, text=True)
  return out_path


def compile_lowered_to_wasm(
    lowered: Any,
    output: str | pathlib.Path,
    *,
    iree_compile: str | None = None,
) -> pathlib.Path:
  """Compile a JAX ``Lowered`` artifact (``.as_text()`` StableHLO) to wasm32."""
  return compile_stablehlo_mlir_to_wasm(
      lowered.as_text(), output, iree_compile=iree_compile
  )


def assert_wasm_artifact_under_limit(
    path: str | pathlib.Path,
    *,
    max_bytes: int = WASM_ARTIFACT_MAX_BYTES,
) -> int:
  """Return artifact size in bytes; raise ``AssertionError`` if over limit."""
  size = pathlib.Path(path).stat().st_size
  if size > max_bytes:
    mb = size / (1024 * 1024)
    cap = max_bytes / (1024 * 1024)
    raise AssertionError(
        f"WASM artifact {path} is {mb:.2f} MB; limit is {cap:.0f} MB (#277)"
    )
  return size


__all__ = [
    "_IN_EXPORT_CONTEXT",
    "WASM_ARTIFACT_MAX_BYTES",
    "assert_wasm_artifact_under_limit",
    "compile_lowered_to_wasm",
    "compile_stablehlo_mlir_to_wasm",
    "export_context",
    "export_energy_fn",
    "export_langevin_step",
    "find_iree_compile",
    "in_export_context",
    "load_artifact",
    "resolve_pallas_compiler_params",
    "save_artifact",
]
