# Justfile for prolix (MD Engine)

# --- Engaging cluster (override via environment) ---
engaging_login := env_var_or_default("ENGAGING_LOGIN", "engaging")
# UV workspace root on the remote (sibling prolix + proxide; pyproject.toml + uv.lock live here)
engaging_workspace_remote_dir := env_var_or_default("ENGAGING_WORKSPACE_REMOTE_DIR", "~/projects")
engaging_remote_dir := env_var_or_default("ENGAGING_REMOTE_DIR", "~/projects/prolix")
engaging_partition := env_var_or_default("ENGAGING_PARTITION", "pi_so3")
ssh_opts := env_var_or_default("ENGAGING_SSH_OPTS", "")

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

# --- Engaging: SSH, sync, SLURM, logs ---

# Persistent SSH control master (optional; speeds repeated rsync/ssh)
login-engaging:
    @ssh -O check {{engaging_login}} 2>/dev/null || \
        (echo "Establishing SSH control master to {{engaging_login}}..." && ssh -fNM {{engaging_login}})

# Workspace manifest + lock at ~/projects/; optional sibling ../proxide → ~/projects/proxide/
push-engaging-workspace: login-engaging
    @echo "Syncing UV workspace root to {{engaging_login}}:{{engaging_workspace_remote_dir}}/"
    rsync -azP {{justfile_directory()}}/workspace/pyproject.toml {{justfile_directory()}}/workspace/uv.lock \
        {{engaging_login}}:{{engaging_workspace_remote_dir}}/
    @if [ -d "{{justfile_directory()}}/../proxide" ]; then \
        echo "Syncing sibling proxide to {{engaging_login}}:{{engaging_workspace_remote_dir}}/proxide/"; \
        rsync -azP {{justfile_directory()}}/../proxide/ {{engaging_login}}:{{engaging_workspace_remote_dir}}/proxide/ \
            --filter=':- .gitignore' \
            --exclude='.venv' \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='target' \
        ; \
    else \
        echo "Note: no local ../proxide — ensure ~/projects/proxide on the cluster (clone or rsync) before uv sync."; \
    fi

# Filtered rsync push (mirrors common lab pattern: respect .gitignore, skip venv and large outputs)
push-engaging: push-engaging-workspace
    @echo "Syncing repo to {{engaging_login}}:{{engaging_remote_dir}}"
    rsync -azP {{justfile_directory()}}/ {{engaging_login}}:{{engaging_remote_dir}}/ \
        --filter=':- .gitignore' \
        --exclude='.venv' \
        --exclude='.git' \
        --exclude='.agent' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='/outputs/*' \
        --include='/outputs/inputs/***'

# Chignolin benchmark on {{engaging_partition}} (override ENGAGING_REMOTE_DIR / ENGAGING_LOGIN as needed)
submit-bench-chignolin: push-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && sbatch --partition={{engaging_partition}} -o outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/%x_%j.out -e outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/%x_%j.err scripts/slurm/bench_chignolin_pi_so3.slurm'

submit-bench-chignolin-preemptable: push-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && sbatch -o outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/%x_%j.out -e outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/%x_%j.err scripts/slurm/bench_chignolin_preemptable.slurm'

submit-bench-array: push-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && sbatch -o outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/%x_%A_%a.out -e outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/%x_%A_%a.err scripts/slurm/bench_array.slurm'

# Chained accuracy → speed jobs (same partition as ENGAGING_PARTITION)
submit-bench-chignolin-split: push-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && export ENGAGING_PARTITION={{engaging_partition}} && bash scripts/slurm/submit_chignolin_split_pi_so3.sh'

# Follow the newest SLURM log under outputs/logs/engaging/
logs-engaging: login-engaging
    ssh -t {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && tail -f $(ls -t outputs/logs/engaging/*/slurm/*.out 2>/dev/null | head -1)'

# Pull dated engaging logs (JSON + text; exclude huge artifacts)
pull-logs-engaging: login-engaging
    @mkdir -p {{justfile_directory()}}/outputs/logs/engaging
    rsync -azP {{ssh_opts}} \
        --exclude='*.pkl' --exclude='*.npz' --exclude='*.eqx' --exclude='*.tmp' \
        {{engaging_login}}:{{engaging_remote_dir}}/outputs/logs/engaging/ \
        {{justfile_directory()}}/outputs/logs/engaging/

# After push: install from workspace root (venv at ~/projects/.venv)
sync-uv-engaging: login-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_workspace_remote_dir}} && uv sync --extra cuda --extra dev --package prolix'
