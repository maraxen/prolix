"""Tool registry for the prolix CLI and MCP server (spec section 3, D1, D7).

Seven tools, one body each. Sync bodies are registered in registry ``prolix`` (the CLI, which
cisternal runs synchronously); async twins with the same names are registered in ``prolix_mcp``
(the MCP server). An async twin runs its op in a worker thread under ``RUN_LOCK``, and a second
concurrent call raises ``RefusalError("busy")`` at once (D7, A10, A16).

This module imports cisternal at import time, so it is loaded only by the CLI and MCP entry points.
Bodies return dicts and raise ``AgentError``; they never print.
"""

from __future__ import annotations

import asyncio
from typing import Any

import cisternal

from prolix.agent import _runtime as rt
from prolix.agent import ops, render
from prolix.agent.errors import RefusalError
from prolix.agent.system import DEFAULT_FF

CLI_REGISTRY = "prolix"
MCP_REGISTRY = "prolix_mcp"
TOOL_NAMES: tuple[str, ...] = (
  "load_system",
  "inspect_params",
  "energy_breakdown",
  "minimize",
  "md_smoke",
  "render_job",
  "submit_job",
)
BUSY_MESSAGE = "busy: another prolix operation is running; retry when it finishes"


async def _locked(fn: Any, **kwargs: Any) -> dict[str, Any]:
  """Run ``fn(**kwargs)`` in a worker thread, one at a time; refuse at once when busy (D7)."""
  if rt.RUN_LOCK.locked():
    raise RefusalError(BUSY_MESSAGE)
  async with rt.RUN_LOCK:
    return await asyncio.to_thread(fn, **kwargs)


# ---------------------------------------------------------------- CLI (sync) registry


@cisternal.tool(registry=CLI_REGISTRY)
def load_system(
  pdb: str, out_dir: str, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """Parameterise a protein PDB with ff19SB and validate it; writes load_system.json to out_dir."""
  return ops.load_system(pdb=pdb, out_dir=out_dir, ff=ff, overwrite=overwrite)


@cisternal.tool(registry=CLI_REGISTRY)
def inspect_params(
  pdb: str, term: str, out_dir: str, limit: int = 50, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """List parameter rows for one term (bond, angle, dihedral, improper, charge, lj); limit is at most 200."""
  return ops.inspect_params(
    pdb=pdb, term=term, out_dir=out_dir, limit=limit, ff=ff, overwrite=overwrite
  )


@cisternal.tool(registry=CLI_REGISTRY)
def energy_breakdown(
  pdb: str, out_dir: str, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """Total and bonded energy and force statistics for a PDB on CPU (dense path, no neighbor list)."""
  return ops.energy_breakdown(pdb=pdb, out_dir=out_dir, ff=ff, overwrite=overwrite)


@cisternal.tool(registry=CLI_REGISTRY)
def minimize(
  pdb: str, out_dir: str, sd_steps: int = 50, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """Bounded steepest descent on a PDB; refuses unless the energy does not rise; writes minimized.npz."""
  return ops.minimize(pdb=pdb, out_dir=out_dir, sd_steps=sd_steps, ff=ff, overwrite=overwrite)


@cisternal.tool(registry=CLI_REGISTRY)
def md_smoke(
  pdb: str,
  out_dir: str,
  steps: int = 100,
  dt_fs: float = 1.0,
  ensemble: str = "nve",
  temperature_k: float | None = None,
  ff: str = DEFAULT_FF,
  overwrite: bool = False,
) -> dict[str, Any]:
  """Short NVE or NVT run on a PDB with a fixed seed, to check finiteness only; writes final.npz."""
  return ops.md_smoke(
    pdb=pdb,
    out_dir=out_dir,
    steps=steps,
    dt_fs=dt_fs,
    ensemble=ensemble,
    temperature_k=temperature_k,
    ff=ff,
    overwrite=overwrite,
  )


@cisternal.tool(registry=CLI_REGISTRY)
def render_job(
  script: str,
  name: str,
  preset: str,
  args: list[str] | None = None,
  time: str | None = None,
  repo_root: str | None = None,
) -> dict[str, Any]:
  """Render an sbatch script for an allowlisted scripts/ file; returns the myxcel push and submit argvs."""
  return render.render_job(
    script=script, name=name, preset=preset, args=args, time=time, repo_root=repo_root
  )


@cisternal.tool(registry=CLI_REGISTRY)
def submit_job(
  name: str,
  remote: str = render.DEFAULT_REMOTE,
  confirm: bool = False,
  repo_root: str | None = None,
) -> dict[str, Any]:
  """Push and submit a rendered job with myxcel; dry-run unless confirm and PROLIX_AGENT_ALLOW_SUBMIT=1."""
  return render.submit_job(name=name, remote=remote, confirm=confirm, repo_root=repo_root)


# ---------------------------------------------------------------- MCP (async) registry


@cisternal.tool(registry=MCP_REGISTRY, name="load_system")
async def load_system_mcp(
  pdb: str, out_dir: str, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """Parameterise a protein PDB with ff19SB and validate it; writes load_system.json to out_dir."""
  return await _locked(ops.load_system, pdb=pdb, out_dir=out_dir, ff=ff, overwrite=overwrite)


@cisternal.tool(registry=MCP_REGISTRY, name="inspect_params")
async def inspect_params_mcp(
  pdb: str, term: str, out_dir: str, limit: int = 50, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """List parameter rows for one term (bond, angle, dihedral, improper, charge, lj); limit is at most 200."""
  return await _locked(
    ops.inspect_params, pdb=pdb, term=term, out_dir=out_dir, limit=limit, ff=ff, overwrite=overwrite
  )


@cisternal.tool(registry=MCP_REGISTRY, name="energy_breakdown")
async def energy_breakdown_mcp(
  pdb: str, out_dir: str, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """Total and bonded energy and force statistics for a PDB on CPU (dense path, no neighbor list)."""
  return await _locked(ops.energy_breakdown, pdb=pdb, out_dir=out_dir, ff=ff, overwrite=overwrite)


@cisternal.tool(registry=MCP_REGISTRY, name="minimize")
async def minimize_mcp(
  pdb: str, out_dir: str, sd_steps: int = 50, ff: str = DEFAULT_FF, overwrite: bool = False
) -> dict[str, Any]:
  """Bounded steepest descent on a PDB; refuses unless the energy does not rise; writes minimized.npz."""
  return await _locked(
    ops.minimize, pdb=pdb, out_dir=out_dir, sd_steps=sd_steps, ff=ff, overwrite=overwrite
  )


@cisternal.tool(registry=MCP_REGISTRY, name="md_smoke")
async def md_smoke_mcp(
  pdb: str,
  out_dir: str,
  steps: int = 100,
  dt_fs: float = 1.0,
  ensemble: str = "nve",
  temperature_k: float | None = None,
  ff: str = DEFAULT_FF,
  overwrite: bool = False,
) -> dict[str, Any]:
  """Short NVE or NVT run on a PDB with a fixed seed, to check finiteness only; writes final.npz."""
  return await _locked(
    ops.md_smoke,
    pdb=pdb,
    out_dir=out_dir,
    steps=steps,
    dt_fs=dt_fs,
    ensemble=ensemble,
    temperature_k=temperature_k,
    ff=ff,
    overwrite=overwrite,
  )


@cisternal.tool(registry=MCP_REGISTRY, name="render_job")
async def render_job_mcp(
  script: str,
  name: str,
  preset: str,
  args: list[str] | None = None,
  time: str | None = None,
  repo_root: str | None = None,
) -> dict[str, Any]:
  """Render an sbatch script for an allowlisted scripts/ file; returns the myxcel push and submit argvs."""
  return await _locked(
    render.render_job,
    script=script,
    name=name,
    preset=preset,
    args=args,
    time=time,
    repo_root=repo_root,
  )


@cisternal.tool(registry=MCP_REGISTRY, name="submit_job")
async def submit_job_mcp(
  name: str,
  remote: str = render.DEFAULT_REMOTE,
  confirm: bool = False,
  repo_root: str | None = None,
) -> dict[str, Any]:
  """Push and submit a rendered job with myxcel; dry-run unless confirm and PROLIX_AGENT_ALLOW_SUBMIT=1."""
  return await _locked(
    render.submit_job, name=name, remote=remote, confirm=confirm, repo_root=repo_root
  )


__all__ = ["CLI_REGISTRY", "MCP_REGISTRY", "TOOL_NAMES"]
