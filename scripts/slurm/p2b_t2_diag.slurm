#!/bin/bash
# Diagnostic: NVT t2 temperature trajectory — prints T at every step for 300 steps.
# Usage: cd ~/projects/prolix && sbatch scripts/slurm/p2b_t2_diag.slurm

#SBATCH --job-name=plx-p2b-t2diag
#SBATCH --partition=mit_preemptable
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/engaging/%j.out
#SBATCH --error=outputs/logs/engaging/%j.err

set -euo pipefail
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"
source "scripts/slurm/_common_env.sh"

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"
echo "job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-local}"

if [[ "${P2B_SKIP_UV_SYNC:-0}" != "1" ]]; then
  source "scripts/slurm/_workspace_uv_sync_inner.sh"
  _workspace_uv_sync_run
fi
cd "${PROLIX_ROOT}"

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
mkdir -p tmp

echo "=== T2 diagnostic: 300 steps, T at every step ==="
uv run python - << 'PY'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from prolix.physics import pbc, settle, system
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _equil_water_positions, _proxide_params_pure_water,
)

n_waters = 895
n_atoms = n_waters * 3
dof = float(6 * n_waters - 3)
kT = 300.0 * BOLTZMANN_KCAL
dt_akma = 0.5 / AKMA_TIME_UNIT_FS

excl = np.zeros((n_atoms, 2), dtype=np.int32)
for w in range(n_waters):
    o, h1, h2 = w*3, w*3+1, w*3+2
    excl[o]=[h1,h2]; excl[h1]=[o,h2]; excl[h2]=[o,h1]

positions_a, box_edge = _equil_water_positions(n_waters, seed=42)
box_vec = jnp.array([box_edge]*3, dtype=jnp.float64)
sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items() if k != "exclusion_mask"}
sys_dict["excl_indices"] = jnp.array(excl, dtype=jnp.int32)
displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True,
    implicit_solvent=False, pme_grid_points=30, pme_alpha=0.34, cutoff_distance=9.0,
    strict_parameterization=False)
mass = jnp.array([[15.999],[1.008],[1.008]]*n_waters, dtype=jnp.float64).reshape(n_atoms)
water_indices = settle.get_water_indices(0, n_waters)

for gamma_ps, label in [(1.0, "gamma=1"), (10.0, "gamma=10")]:
    gamma_reduced = gamma_ps * AKMA_TIME_UNIT_FS * 1e-3
    init_s, apply_s = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT,
        gamma=gamma_reduced, mass=mass, water_indices=water_indices, box=box_vec,
        remove_linear_com_momentum=False, project_ou_momentum_rigid=True,
        projection_site="post_o", settle_velocity_iters=10)

    state = init_s(jax.random.key(42), jnp.array(positions_a, dtype=jnp.float64), mass=mass)

    def step_fn(state, _):
        state = apply_s(state)
        m_flat = state.mass.reshape(-1)
        ke = jnp.sum(jnp.sum(state.momentum**2, axis=-1) / (2.0 * m_flat))
        return state, ke

    print(f"\n--- {label} ---")
    _, kes = jax.lax.scan(step_fn, state, None, length=300)
    temps = np.array(2.0 * kes / (dof * BOLTZMANN_KCAL))
    for i in [0, 9, 49, 99, 149, 199, 249, 299]:
        print(f"  step {i+1:3d}: T={temps[i]:.1f} K")
    print(f"  mean steps 1-100: {temps[:100].mean():.1f} K")
    print(f"  mean steps 101-300: {temps[100:].mean():.1f} K")
PY
echo "=== Done ==="
