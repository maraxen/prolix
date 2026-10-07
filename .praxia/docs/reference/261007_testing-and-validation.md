---
title: "Testing & Validation"
description: "Pytest command catalog and CI JSON report queries moved verbatim from CLAUDE.md."
status: reference
date: '261007'
---

## Testing & Validation

For detailed testing instructions and CI-safe JSON report querying, see `.copilot/instructions.md`.

Quick reference (use `uv run` to ensure correct environment):
```bash
uv run pytest -m "not slow"              # Fast CI mode (smoke tests only)
uv run pytest                            # Full suite
uv run pytest tests/physics/test_settle.py  # Specific test file
jq '.summary' tmp/pytest.json            # Get pass/fail counts
jq '.tests[] | select(.outcome=="failed") | .nodeid' tmp/pytest.json  # List failures
```

Results are written to `tmp/pytest.json` (added to .gitignore) for grepping and CI integration.
