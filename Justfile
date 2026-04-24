# Justfile for prolix (MD Engine)

# --- Engaging cluster (override via environment) ---
engaging_login := env_var_or_default("ENGAGING_LOGIN", "engaging")
# UV workspace root on the remote (sibling prolix + proxide; pyproject.toml + uv.lock live here)
engaging_workspace_remote_dir := env_var_or_default("ENGAGING_WORKSPACE_REMOTE_DIR", "~/projects")
engaging_remote_dir := env_var_or_default("ENGAGING_REMOTE_DIR", "~/projects/prolix")
engaging_partition := env_var_or_default("ENGAGING_PARTITION", "pi_so3")
engaging_partition_preemptable := env_var_or_default("ENGAGING_PARTITION_PREEMPTABLE", "mit_preemptable")
# Extra sbatch args for TIP3P tightening (e.g. ``--array=0-3%2`` or ``--gres=gpu:1``).
tip3p_sbatch_opts := env_var_or_default("TIP3P_SBATCH_OPTS", "--array=0-7%4")
# Subdir under outputs/tip3p_tightening/<ENGAGING_LOG_DATE>/ (avoids resuming old checkpoints). Empty = legacy path.
tip3p_run_tag := env_var_or_default("TIP3P_RUN_TAG", "")
# Forwarded to Engaging before ``submit_tip3p_chain.sh`` (must match ``bench_tip3p_langevin_preemptable.slurm`` defaults).
tip3p_projection_site := env_var_or_default("TIP3P_PROJECTION_SITE", "post_o")
tip3p_settle_velocity_iters := env_var_or_default("TIP3P_SETTLE_VELOCITY_ITERS", "10")
tip3p_project_ou_momentum_rigid := env_var_or_default("TIP3P_PROJECT_OU_MOMENTUM_RIGID", "true")
tip3p_gamma_ps := env_var_or_default("TIP3P_GAMMA_PS", "1.0")
tip3p_diagnostics_level := env_var_or_default("TIP3P_DIAGNOSTICS_LEVEL", "off")
tip3p_diagnostics_decimation := env_var_or_default("TIP3P_DIAGNOSTICS_DECIMATION", "10")
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
        {{justfile_directory()}}/workspace/.python-version {{justfile_directory()}}/workspace/.gitignore \
        {{engaging_login}}:{{engaging_workspace_remote_dir}}/
    @if [ -d "{{justfile_directory()}}/../proxide" ]; then \
        echo "Optional: syncing sibling proxide for co-development (PyPI wheels are the default dep)."; \
        rsync -azP {{justfile_directory()}}/../proxide/ {{engaging_login}}:{{engaging_workspace_remote_dir}}/proxide/ \
            --filter=':- .gitignore' \
            --exclude='.venv' \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='target' \
        ; \
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

# Like ``push-engaging`` but ``rsync --delete`` so the remote tree matches (excluded paths untouched).
push-engaging-clean: login-engaging
    @echo "Syncing UV workspace root (clean) to {{engaging_login}}:{{engaging_workspace_remote_dir}}/"
    rsync -azP --delete {{justfile_directory()}}/workspace/pyproject.toml {{justfile_directory()}}/workspace/uv.lock \
        {{justfile_directory()}}/workspace/.python-version {{justfile_directory()}}/workspace/.gitignore \
        {{engaging_login}}:{{engaging_workspace_remote_dir}}/
    @if [ -d "{{justfile_directory()}}/../proxide" ]; then \
        echo "Optional: syncing sibling proxide (clean) for co-development."; \
        rsync -azP --delete {{justfile_directory()}}/../proxide/ {{engaging_login}}:{{engaging_workspace_remote_dir}}/proxide/ \
            --filter=':- .gitignore' \
            --exclude='.venv' \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='target' \
        ; \
    fi
    @echo "Syncing repo (clean) to {{engaging_login}}:{{engaging_remote_dir}}"
    rsync -azP --delete {{justfile_directory()}}/ {{engaging_login}}:{{engaging_remote_dir}}/ \
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

# Workspace venv (``proxide`` from PyPI). Python 3.12 via ``workspace/.python-version``.
submit-workspace-uv-sync-pi-so3: push-engaging-clean
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && sbatch --parsable --partition={{engaging_partition}} -o outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/uvsync_%j.out -e outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/uvsync_%j.err scripts/slurm/bench_workspace_uv_sync.slurm'

submit-workspace-uv-sync-preemptable: push-engaging-clean
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && sbatch --parsable --partition={{engaging_partition_preemptable}} -o outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/uvsync_%j.out -e outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm/uvsync_%j.err scripts/slurm/bench_workspace_uv_sync.slurm'

# P2a-B2: venv sync → TIP3P array (``submit_tip3p_chain.sh``).
submit-tip3p-tightening-preemptable: push-engaging-clean
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && export TIP3P_SBATCH_OPTS="{{tip3p_sbatch_opts}}" && export TIP3P_RUN_TAG="{{tip3p_run_tag}}" && export TIP3P_PROJECTION_SITE="{{tip3p_projection_site}}" && export TIP3P_SETTLE_VELOCITY_ITERS="{{tip3p_settle_velocity_iters}}" && export TIP3P_PROJECT_OU_MOMENTUM_RIGID="{{tip3p_project_ou_momentum_rigid}}" && export TIP3P_GAMMA_PS="{{tip3p_gamma_ps}}" && export TIP3P_DIAGNOSTICS_LEVEL="{{tip3p_diagnostics_level}}" && export TIP3P_DIAGNOSTICS_DECIMATION="{{tip3p_diagnostics_decimation}}" && bash scripts/slurm/submit_tip3p_chain.sh {{engaging_partition_preemptable}}'

# Same chain on ``pi_so3`` (non-preemptable preference).
submit-tip3p-tightening-pi-so3: push-engaging-clean
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && export TIP3P_SBATCH_OPTS="{{tip3p_sbatch_opts}}" && export TIP3P_RUN_TAG="{{tip3p_run_tag}}" && export TIP3P_PROJECTION_SITE="{{tip3p_projection_site}}" && export TIP3P_SETTLE_VELOCITY_ITERS="{{tip3p_settle_velocity_iters}}" && export TIP3P_PROJECT_OU_MOMENTUM_RIGID="{{tip3p_project_ou_momentum_rigid}}" && export TIP3P_GAMMA_PS="{{tip3p_gamma_ps}}" && export TIP3P_DIAGNOSTICS_LEVEL="{{tip3p_diagnostics_level}}" && export TIP3P_DIAGNOSTICS_DECIMATION="{{tip3p_diagnostics_decimation}}" && bash scripts/slurm/submit_tip3p_chain.sh {{engaging_partition}}'

# Falsification: 1 fs, shorter production, 4 replicas (faster than full Tier-1). Fresh tag ``dt1fs_smoke``.
submit-tip3p-tightening-1fs-smoke-pi-so3: push-engaging-clean
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && export TIP3P_RUN_TAG=dt1fs_smoke && export TIP3P_DT_FS=1.0 && export TIP3P_TOTAL_STEPS=20000 && export TIP3P_BURN_IN=5000 && export TIP3P_SBATCH_OPTS="--array=0-3%2" && bash scripts/slurm/submit_tip3p_chain.sh {{engaging_partition}}'

# TIP3P KE diagnostic: two jobs (OpenMM CPU, Prolix GPU); ``--exclude=node4009`` on both.
submit-tip3p-ke-compare: push-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm outputs/logs/engaging/$ENGAGING_LOG_DATE/app && o=outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && j1=$(sbatch --parsable --partition={{engaging_partition}} --exclude=node4009 -o $o/%x_%j.out -e $o/%x_%j.err scripts/slurm/bench_tip3p_ke_compare_openmm.slurm) && j2=$(sbatch --parsable --partition={{engaging_partition}} --exclude=node4009 -o $o/%x_%j.out -e $o/%x_%j.err scripts/slurm/bench_tip3p_ke_compare_prolix.slurm) && echo "openmm_job=${j1}" && echo "prolix_job=${j2}"'

# Local dt×γ matrix (one job, JSON in app log). Optional env on remote: ``TIP3P_MATRIX_*`` (see slurm file).
submit-tip3p-dt-gamma-matrix: push-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm outputs/logs/engaging/$ENGAGING_LOG_DATE/app && o=outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && sbatch --parsable --partition={{engaging_partition_preemptable}} -o $o/tip3p_dt_gamma_%j.out -e $o/tip3p_dt_gamma_%j.err scripts/slurm/bench_tip3p_dt_gamma_matrix.slurm'

# Sprint 3: CSVR thermostat temperature + equipartition validation (CPU, f64, 4h).
submit-csvr-validate: push-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_remote_dir}} && export ENGAGING_LOG_DATE=$(date +%Y%m%d) && mkdir -p outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm outputs/logs/engaging/$ENGAGING_LOG_DATE/app && o=outputs/logs/engaging/$ENGAGING_LOG_DATE/slurm && sbatch --parsable --partition={{engaging_partition_preemptable}} -o $o/%x_%j.out -e $o/%x_%j.err scripts/slurm/validate_csvr_temperature.slurm'

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

# After push: install from workspace root (venv at ~/projects/.venv). ``proxide`` resolves from PyPI.
sync-uv-engaging: login-engaging
    ssh {{ssh_opts}} {{engaging_login}} 'cd {{engaging_workspace_remote_dir}} && uv python install && uv sync --extra cuda --extra dev --extra openmm --package prolix'
