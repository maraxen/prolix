"""The ``prolix`` command: a cyclopts app wired to the CLI registry (spec section 3).

``require_agent_extra()`` runs before the cisternal, cyclopts and fastmcp imports below, so a
missing extra fails with the ``prolix[agent]`` hint. ``cli_main`` in ``prolix_agent_entry`` has
already fixed the JAX platform before this module imports ``prolix``.
"""

from prolix.agent import require_agent_extra

require_agent_extra()

import json  # noqa: E402
import sys  # noqa: E402
from typing import Any  # noqa: E402

import cyclopts  # noqa: E402
from cisternal.registration.cli_contract import CliContext, CliContract, json_option  # noqa: E402
from cisternal.registration.wired import wire  # noqa: E402
from cyclopts.core import App  # noqa: E402

from prolix.agent import tools  # noqa: E402
from prolix.agent.errors import InputError, RefusalError  # noqa: E402

EXIT_CODES_EPILOG = """\
Exit codes:
  0  success
  1  the prolix[agent] extra is missing (the install hint is printed); also an unhandled exception
  2  usage error (bad or missing argument, unknown command), or a jax cache configuration error
     (stderr starts with "prolix: jax cache configuration error:")
  3  refused: a cap, a busy server, or submission disabled
  4  input error: missing file, bad value, or a record that exists without --overwrite
"""


def _format_success(result: Any, ctx: CliContext) -> None:
  """Write the result to stdout: one JSON object with --json, else ``key: value`` lines."""
  if ctx.options.get("json_out"):
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
  elif isinstance(result, dict):
    for key in sorted(result):
      sys.stdout.write(f"{key}: {result[key]}\n")
  else:
    sys.stdout.write(f"{result}\n")


# ty 0.0.29 cannot resolve the signatures of cyclopts.App and cisternal's CliContract (lazy
# re-exports in the installed packages). _construct passes them through Any; runtime is unchanged.


def _construct(cls: Any, **kwargs: Any) -> Any:
  return cls(**kwargs)


CONTRACT = _construct(
  CliContract,
  format_success=_format_success,
  exit_codes={InputError: 4, RefusalError: 3},
  options=[json_option()],
)

app = _construct(
  App,
  name="prolix",
  help="CPU in-process operations on protein PDBs. Long runs are rendered as Slurm jobs.",
  help_epilogue=EXIT_CODES_EPILOG,
)
wire(
  None,
  app,
  registry=tools.CLI_REGISTRY,
  expected=list(tools.TOOL_NAMES),
  cli_contract=CONTRACT,
)


def main(argv: list[str] | None = None) -> None:
  """Start xtrax.agent (jax cache, then telemetry), then dispatch. Usage errors exit 2."""
  from xtrax import agent

  try:
    agent.run_cli(app, argv, app_kwargs={"exit_on_error": False, "print_error": True})
  except cyclopts.CycloptsError as exc:
    sys.stderr.write(f"prolix: {exc}\n")
    raise SystemExit(2) from exc


__all__ = ["CONTRACT", "app", "main"]
