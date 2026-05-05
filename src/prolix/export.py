"""Prolix StableHLO export utilities for WASM/WebGPU deployment.

Wraps jax.jit lowering to produce portable StableHLO artifacts from proxide
energy functions built with the explicit-params API (make_energy_fn_pure).

v1.1 Item 4: unblocked by EnergyParams / make_energy_fn_pure (Item 1).
"""

from __future__ import annotations

import pathlib
from typing import Any

import jax


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
    step: Any,
    example_state: Any,
    **params: Any,
) -> Any:
  """Export a Langevin integrator step to StableHLO.

  NOTE: Deferred to v1.2. IntegratorState.apply() has optional
  constraint_dofs and mixed return types not yet compatible with
  jax.export flat-signature requirements. See docs/EXPORT_GUIDE.md.

  Raises:
      NotImplementedError: Always, in v1.1.
  """
  raise NotImplementedError(
      "export_langevin_step is deferred to v1.2. "
      "IntegratorState.apply() needs refactoring (flat return type, "
      "no optional constraint_dofs). See docs/EXPORT_GUIDE.md."
  )


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


__all__ = [
    "export_energy_fn",
    "export_langevin_step",
    "save_artifact",
    "load_artifact",
]
