#!/bin/bash
# Pull NVIDIA NGC OpenMM image to a local .sif on a compute node.
#
#SBATCH --job-name=plx-pull-omm
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_so3
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Anchor paths to this script — safe from any submission cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common_env.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_openmm_ngc_env.sh"

openmm_load_container_module

RUNTIME="$(openmm_container_runtime)"
if [[ -z "$RUNTIME" ]]; then
    echo "Neither apptainer nor singularity found." >&2
    exit 1
fi

DEST="$(openmm_ngc_sif_path)"
mkdir -p "$(dirname "$DEST")"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/.apptainer/cache}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$HOME/.singularity/cache}"

echo "========================================================================"
echo "Pulling NGC OpenMM -> $DEST"
echo "URI: $OPENMM_NGC_DOCKER_URI"
echo "Runtime: $RUNTIME  HOST=$(hostname -s)  SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "========================================================================"

if [[ -f "$DEST" ]]; then
    echo "SIF already exists: $DEST (remove it to re-pull)"
    ls -la "$DEST"
    exit 0
fi

"$RUNTIME" pull "$DEST" "$OPENMM_NGC_DOCKER_URI"

echo "Done."
ls -la "$DEST"
