# Shared helpers for OpenMM via Apptainer/Singularity (NGC image) for Prolix.
# shellcheck shell=bash

# Default values if not set by cluster-config.toml parsing or env
export OPENMM_NGC_DOCKER_URI="${OPENMM_NGC_DOCKER_URI:-docker://nvcr.io/nvidia/openmm:8.1.1}"
export OPENMM_NGC_VERSION_TAG="${OPENMM_NGC_VERSION_TAG:-8.1.1}"
export PROLIX_OPENMM_SIF="${PROLIX_OPENMM_SIF:-$HOME/containers/openmm-${OPENMM_NGC_VERSION_TAG}.sif}"

openmm_container_runtime() {
    if command -v apptainer >/dev/null 2>&1; then
        echo apptainer
    elif command -v singularity >/dev/null 2>&1; then
        echo singularity
    else
        echo ""
    fi
}

openmm_ngc_sif_path() {
    echo "${PROLIX_OPENMM_SIF}"
}

openmm_load_container_module() {
    # Engaging: apptainer module
    if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
        if module load apptainer/1.4.2 2>/dev/null; then
            return 0
        fi
        if module load apptainer 2>/dev/null; then
            return 0
        fi
        if module load singularity 2>/dev/null; then
            return 0
        fi
    fi
    return 0
}
