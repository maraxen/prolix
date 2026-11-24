"""Wrapper entrypoint for ``scripts/experiments/knob_parity_matrix.py``."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.knob_parity_matrix import (  # noqa: E402
    evaluate_knob_parity_matrix,
    main,
)

__all__ = ["evaluate_knob_parity_matrix", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
