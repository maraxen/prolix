# Justfile for prolix (MD Engine)

default:
    @just --list

# Run all tests
test:
    uv run pytest

# Run tests matching a pattern
test-grep pattern:
    uv run pytest -k {{pattern}}

# --- Benchmarks ---

# GB solvation forces benchmark
benchmark-gb:
    uv run scripts/benchmark_gb_forces.py

# Neighbor List vs Dense parity/performance
benchmark-nl:
    uv run scripts/benchmark_nlvsdense.py

# Profile a single simulation step
profile-step:
    uv run scripts/profile_step.py

# --- Quality ---

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

check:
    uv run pyright
