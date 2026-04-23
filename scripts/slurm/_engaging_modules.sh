# Source after ``set -euo pipefail`` from SLURM driver scripts.
# ORCD Engaging (Lmod): full version, e.g. ``cmake/3.27.9`` — bare ``cmake`` is not loadable.
# Override: ``export ENGAGING_CMAKE_MODULE=cmake/3.24.3-x86_64``
#
# Batch jobs are often non-login shells: ``module`` is undefined unless we source an init script.

engaging_load_modules() {
  local _f
  local _cmake="${ENGAGING_CMAKE_MODULE:-cmake/3.27.9}"
  if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash \
      /usr/share/Modules/init/bash; do
      if [[ -f "${_f}" ]]; then
        # shellcheck source=/dev/null
        source "${_f}"
        break
      fi
    done
  fi
  if ! command -v module >/dev/null 2>&1; then
    echo "engaging_load_modules: no 'module' command (Lmod/Environment Modules init missing)." >&2
    return 1
  fi

  module load "${_cmake}"
  if ! command -v cmake >/dev/null 2>&1; then
    echo "engaging_load_modules: 'cmake' not on PATH after 'module load ${_cmake}'." >&2
    return 1
  fi
  # ``cmake`` crate / maturin subprocesses may not inherit Lmod's PATH; ``CMAKE`` is honored.
  CMAKE="$(command -v cmake)"
  export CMAKE
  export PATH="$(dirname "${CMAKE}"):${PATH}"
  if [[ -n "${ENGAGING_EXTRA_MODULES:-}" ]]; then
    local _m
    IFS=';' read -ra _arr <<< "${ENGAGING_EXTRA_MODULES}"
    for _m in "${_arr[@]}"; do
      _m="${_m#"${_m%%[![:space:]]*}"}"
      _m="${_m%"${_m##*[![:space:]]}"}"
      [[ -n "${_m}" ]] || continue
      module load "${_m}"
    done
  fi
}
