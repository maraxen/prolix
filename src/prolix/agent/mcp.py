"""The ``prolix-mcp`` server: FastMCP over the MCP registry (spec section 3).

``require_agent_extra()`` runs before the cisternal and fastmcp imports below.
"""

from prolix.agent import require_agent_extra

require_agent_extra()

from typing import TYPE_CHECKING  # noqa: E402

from prolix.agent import tools  # noqa: E402

if TYPE_CHECKING:
  from fastmcp import FastMCP

_INSTRUCTIONS = (
  "prolix runs seconds-scale operations on protein PDB files on CPU: load_system parameterises and "
  "validates a PDB; inspect_params lists parameter rows; energy_breakdown and minimize report energies; "
  "md_smoke runs a short NVE or NVT check. Each writes a run record into out_dir. Long runs are not "
  "executed here: render_job writes a Slurm script for an allowlisted scripts/ file, and submit_job "
  "pushes and submits it with myxcel (dry-run unless the human enabled submission)."
)


def build_server() -> "FastMCP":
  """Build the FastMCP server with the MCP registry wired in and cisternal middleware installed."""
  from cisternal.adapters.base import PassthroughAdapter
  from cisternal.adapters.v3_middleware import CisternalMiddleware
  from cisternal.registration.wired import wire
  from fastmcp import FastMCP as FastMCPServer

  server = FastMCPServer("prolix", instructions=_INSTRUCTIONS)
  wire(server, registry=tools.MCP_REGISTRY, expected=list(tools.TOOL_NAMES))
  server.add_middleware(CisternalMiddleware(adapter=PassthroughAdapter(), reraise=True))
  return server


def main() -> None:
  """Start xtrax.agent (jax cache, then telemetry) and serve over stdio."""
  from xtrax import agent

  agent.serve(build_server())


__all__ = ["build_server", "main"]
