"""Agent-facing layer: CLI and MCP server for prolix, built on xtrax.agent and cisternal.

Importing this package imports neither ``cisternal``, ``fastmcp`` nor ``cyclopts``; those
belong to the ``agent`` extra and are loaded only by ``prolix.agent.cli`` / ``mcp``.
"""

from __future__ import annotations

import importlib.util

from prolix.agent.errors import AgentExtraMissing

AGENT_EXTRA_MODULES: tuple[str, ...] = ("cisternal", "fastmcp", "cyclopts")

_GUIDANCE = "prolix agent features need the agent extra: uv sync --extra agent (or pip install 'prolix[agent]')"


def require_agent_extra() -> None:
  """Raise ``AgentExtraMissing`` (an ``ImportError``) if the agent extra is not installed."""
  for name in AGENT_EXTRA_MODULES:
    if importlib.util.find_spec(name) is None:
      raise AgentExtraMissing(_GUIDANCE)


__all__ = ["AGENT_EXTRA_MODULES", "AgentExtraMissing", "require_agent_extra"]
