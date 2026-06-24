"""Prolix StableHLO export utilities for WASM/WebGPU deployment.

Wraps jax.jit lowering to produce portable StableHLO artifacts from proxide
energy functions built with the explicit-params API (make_energy_fn_pure).

v1.1 Item 4: unblocked by EnergyParams / make_energy_fn_pure (Item 1).
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
from typing import Any

import jax

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
  from prolix.types.integrators import LangevinState

  def exported(state: LangevinState) -> LangevinState:
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
    "WASM_ARTIFACT_MAX_BYTES",
    "assert_wasm_artifact_under_limit",
    "compile_lowered_to_wasm",
    "compile_stablehlo_mlir_to_wasm",
    "export_energy_fn",
    "export_langevin_step",
    "find_iree_compile",
    "load_artifact",
    "save_artifact",
]
