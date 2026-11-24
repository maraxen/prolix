"""Console-script entry points for the prolix agent CLI and MCP server.

This package sits outside ``prolix`` on purpose. ``import prolix`` initialises the JAX
backend (``prolix/physics/cmap.py`` builds a ``jnp.array`` at import), so ``JAX_PLATFORMS``
must be set before the first ``import prolix``. Nothing here imports ``prolix`` at module
level; ``prepare()`` runs first and the ``prolix`` modules are imported afterwards.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Modules the agent extra provides. xtrax.agent is checked too (xtrax[plugin] ships it).
_AGENT_MODULES: tuple[str, ...] = ("cisternal", "cyclopts", "fastmcp", "xtrax.agent")
_HINT = "prolix agent features need the agent extra: uv sync --extra agent (or pip install 'prolix[agent]')"


def prepare() -> None:
  """Force the CPU backend before ``prolix`` is imported.

  Unconditional: an inherited ``JAX_PLATFORMS`` (for example ``cuda``) is replaced.
  """
  os.environ["JAX_PLATFORMS"] = "cpu"


def _require_agent_extra() -> None:
  """Exit 1 with the install hint if an agent-extra module is missing.

  Uses ``importlib.util.find_spec`` only, so no heavy module is imported.
  """
  for name in _AGENT_MODULES:
    try:
      found = importlib.util.find_spec(name)
    except ModuleNotFoundError:
      found = None
    if found is None:
      sys.stderr.write(f"{_HINT} (missing: {name})\n")
      raise SystemExit(1)


def cli_main() -> None:
  """Console script ``prolix``."""
  prepare()
  _require_agent_extra()
  from prolix.agent import cli

  cli.main()


def mcp_main() -> None:
  """Console script ``prolix-mcp``."""
  prepare()
  _require_agent_extra()
  from prolix.agent import mcp

  mcp.main()


__all__ = ["cli_main", "mcp_main", "prepare"]
