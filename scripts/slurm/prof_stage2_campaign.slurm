#!/bin/bash
#SBATCH --job-name=prolix_p7_campaign
#SBATCH --partition=mit_normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=outputs/logs/slurm/%j.prof_stage2_campaign.out
#SBATCH --error=outputs/logs/slurm/%j.prof_stage2_campaign.err

# Create the P7 campaign in the *cluster* bathos catalog. Login-node uv is
# forbidden; bth sync push of the local catalog failed (rsync 11).

set -euo pipefail

_locate_root() {
    local cand
    for cand in \
        "${PWD}" \
        "${SLURM_SUBMIT_DIR:-}" \
        "${HOME}/projects/.worktrees/prolix/wt-20260807-132628" \
        "${HOME}/projects/prolix"
    do
        if [[ -n "${cand}" && -f "${cand}/scripts/slurm/_bth_env.sh" ]]; then
            echo "${cand}"
            return 0
        fi
    done
    return 1
}
_PROJ="$(_locate_root)" || exit 1
cd "${_PROJ}"
source "scripts/slurm/_bth_env.sh"
mkdir -p outputs/logs/slurm

uv run bth campaign create "p7-stage2-h200" --mode exploration \
    --question "P7 Stage-2 H200 discovery sweep. Coverage pre-reg: prof_stage2_coverage.bth.toml. Cluster catalog (local 450f3583 was missing here)."
uv run bth campaign ls
