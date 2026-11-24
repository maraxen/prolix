"""Error types for the agent layer. Tool bodies raise these; they never print."""

from __future__ import annotations


class AgentExtraMissing(ImportError):  # noqa: N818 - name fixed by spec section 3
  """The ``prolix[agent]`` extra (cisternal, fastmcp, cyclopts) is not installed."""


class AgentError(Exception):
  """Base class for agent operation errors."""


class InputError(AgentError):
  """The request is malformed or refers to something that does not exist (exit 4)."""


class RefusalError(AgentError):
  """The request is valid but refused by a cap, the busy lock or a policy gate (exit 3)."""


__all__ = ["AgentError", "AgentExtraMissing", "InputError", "RefusalError"]
