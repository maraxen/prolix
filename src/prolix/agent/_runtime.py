"""Shared runtime helpers for the agent layer.

No jax, cisternal, cyclopts or fastmcp at module level. ``assert_cpu`` imports jax inside its body.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from prolix.agent.errors import RefusalError

# Hard caps. A call may lower them, never raise them; there is no CLI override.
# Placeholders: Task 2 lowers them to the calibration table in the PR.
MAX_ATOMS = 1000
MAX_MIN_STEPS = 200
MAX_MD_STEPS = 500
MAX_DT_FS = 2.0

# One in-process op at a time (D7). Acquired by the MCP wrappers in tools.py.
RUN_LOCK = asyncio.Lock()

RECORD_SCHEMA = "prolix.agent.run/1"


def assert_cpu() -> None:
  """Refuse unless the JAX default backend is CPU (fails closed for callers that imported prolix first)."""
  import jax

  backend = jax.default_backend()
  if backend != "cpu":
    raise RefusalError(
      f"in-process prolix operations run on CPU only (backend={backend!r}); "
      "use render_job for GPU work"
    )


def sha256_file(path: str | os.PathLike[str]) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
  """Write to a tempfile in the same directory, then ``os.replace``."""
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
  try:
    with os.fdopen(fd, "wb") as fh:
      fh.write(data)
    os.replace(tmp, target)
  except BaseException:
    if os.path.exists(tmp):
      os.unlink(tmp)
    raise


def versions() -> dict[str, Any]:
  """Package versions from importlib.metadata (prolix.__version__ is stale, A11)."""
  out: dict[str, Any] = {"python": platform.python_version()}
  for dist in ("prolix", "xtrax", "jax", "jaxlib", "proxide", "parmed", "cisternal"):
    try:
      out[dist] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
      out[dist] = None
  return out


def dumps_record(record: dict[str, Any]) -> bytes:
  return (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()


__all__ = [
  "MAX_ATOMS",
  "MAX_DT_FS",
  "MAX_MD_STEPS",
  "MAX_MIN_STEPS",
  "RECORD_SCHEMA",
  "RUN_LOCK",
  "assert_cpu",
  "atomic_write_bytes",
  "dumps_record",
  "sha256_file",
  "versions",
]
