# Prolix Testing & Reporting

This file documents the standardized testing workflow for Prolix using pytest-json-report for CI-safe, grepping-friendly test output.

## Running Tests

This project uses `uv` for Python environment management. Prefix all pytest commands with `uv run`:

### Quick Test (Fast CI Mode)

Skip slow tests and focus on smoke tests:

```bash
uv run pytest -m "not slow"
```

### Full Test Suite

```bash
uv run pytest
```

### Specific Test File or Pattern

```bash
uv run pytest tests/physics/test_settle.py
uv run pytest -k "test_temperature"
```

### With Markers

```bash
uv run pytest -m "smoke"                    # Only smoke tests
uv run pytest -m "not slow and not openmm"  # Skip slow + OpenMM
uv run pytest -m "integration"              # Integration tests
```

### Without `uv run` (Manual Venv Activation)

If you've activated the venv manually (`. .venv/bin/activate`), you can run pytest directly:

```bash
pytest -m "not slow"
pytest
```

But **preferred**: always use `uv run` to ensure the correct environment is used.

## Reading Test Results

Test results are written to `tmp/pytest.json` (added to .gitignore). Use `jq` to query results:

### Get Pass/Fail Summary

```bash
jq '.summary' tmp/pytest.json
```

Example output:
```json
{
  "total": 147,
  "passed": 145,
  "failed": 2,
  "skipped": 0,
  "duration": 127.34
}
```

### List All Failed Tests

```bash
jq '.tests[] | select(.outcome=="failed") | .nodeid' tmp/pytest.json
```

Example:
```
tests/physics/test_settle.py::test_temperature_control
tests/physics/test_npt.py::test_long_trajectory
```

### Get Failed Test Details (Including Error)

```bash
jq '.tests[] | select(.outcome=="failed") | {nodeid, outcome, duration, traceback}' tmp/pytest.json
```

### Filter by Test Duration (e.g., Tests > 5 seconds)

```bash
jq '.tests[] | select(.duration > 5) | {nodeid, duration}' tmp/pytest.json | head -20
```

### Count Tests by Outcome

```bash
jq '.tests | group_by(.outcome) | map({outcome: .[0].outcome, count: length})' tmp/pytest.json
```

## CI Integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:
1. Runs `pytest -m "not slow"` (fast CI mode)
2. Generates `tmp/pytest.json` via `--json-report` option
3. Archives the report as a workflow artifact
4. Parses summary via `jq` to report pass/fail counts

To locally verify CI behavior:

```bash
pytest -m "not slow" && jq '.summary' tmp/pytest.json
```

## Known Test Markers

- `@pytest.mark.smoke` — Fast, critical tests (< 5s each, always in CI)
- `@pytest.mark.slow` — Skipped in fast CI, run on main branch only
- `@pytest.mark.openmm` — Requires optional OpenMM dependency
- `@pytest.mark.kups` — Requires third-party `kups` package (cross-validation)
- `@pytest.mark.integration` — Heavy parity or integration tests
- `@pytest.mark.dynamics` — Long MD trajectories or statistical checks

## Troubleshooting

### If tmp/pytest.json is Missing

Check that pytest ran successfully:

```bash
pytest --co -q tests/physics/test_settle.py  # List tests without running
```

If tests collected but json not generated, ensure pytest-json-report is installed:

```bash
pip list | grep pytest-json-report
# or
uv pip list | grep pytest-json-report
```

### Large JSON File

If `tmp/pytest.json` is very large, you can stream specific fields:

```bash
jq -r '.tests[] | "\(.nodeid): \(.outcome)"' tmp/pytest.json | grep -i failed
```

## Interpreting Test Output

Each test entry in `.tests[]` includes:
- `nodeid` — Full test path (e.g., `tests/physics/test_settle.py::test_temperature`)
- `outcome` — One of: `passed`, `failed`, `skipped`, `xfailed`, `xpassed`
- `duration` — Elapsed time in seconds
- `traceback` — (If failed) Full error traceback
- `call` — Test phase details (setup, call, teardown)

## Configuration

Test configuration lives in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
timeout = 900                    # 15 min timeout per test
timeout_method = "thread"
testpaths = ["tests"]
markers = [...]                  # See list above
addopts = "--json-report --json-report-file=tmp/pytest.json"
```

Modify `addopts` here to customize report output (e.g., add `--tb=short` for shorter tracebacks in ANSI output).

## Performance Profiling Tests

To find the slowest tests:

```bash
jq '.tests | sort_by(-.duration) | .[0:10] | .[] | {nodeid, duration}' tmp/pytest.json
```

This helps identify bottlenecks for optimization.
