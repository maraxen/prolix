"""Phase 5 R-step liquid-density divergence diagnostic (exploratory).

Reproduces the 895-water NVT thermal runaway locally over a short window and
separates two competing hypotheses for the divergence:

  H1 (integrator):   the dp-only R-step injects energy each step; a pure-NVE
                     run (gamma=0, no noise) will show total-energy blowup.
  H2 (force field):  forces are already wrong at liquid density (e.g. PME
                     intramolecular exclusion incomplete); PE / max|force| is
                     absurd at step 0 and NVE energy is conserved only because
                     it is conserved-ly-wrong.

For each mode we log per step: T (from KE), PE, total E (KE+PE), max|force|.
Divergence is fast (~286 K -> 4098 K by step 100 per the test docstring), so a
~120-step window is enough.

Exploratory: lives in scripts/explore/ (no bathos sidecar; ad-hoc debugging).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p5diag")


def _find_root(marker: str = "pyproject.toml") -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / marker).exists():
            return p
    raise RuntimeError("project root not found")


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prolix.physics import pbc, settle, system  # noqa: E402
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL  # noqa: E402
from tests.physics.test_explicit_langevin_tip3p_parity import (  # noqa: E402
    _equil_water_positions,
    _proxide_params_pure_water,
)
from tests.physics.test_p2b_nvt_216water import (  # noqa: E402
    _make_tip3p_excl_indices,
    _dof_rigid_tip3p_waters,
)


def _build(n_waters, box_edge, dt_fs, temperature_k, gamma_ps, excl_indices):
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items()
                if k != "exclusion_mask"}
    sys_dict["excl_indices"] = excl_indices

    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    grid = max(16, round(box_edge / 1.0))
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=grid, pme_alpha=0.34,
        cutoff_distance=9.0, strict_parameterization=False)

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters,
                     dtype=jnp.float64).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)
    return energy_fn, shift_fn, mass, water_indices, box_vec, dt_akma, kT, gamma_reduced


def run_mode(name, n_waters, positions_a, box_edge, steps, dt_fs,
             temperature_k, gamma_ps, excl_indices, seed):
    (energy_fn, shift_fn, mass, water_indices, box_vec,
     dt_akma, kT, gamma_reduced) = _build(
        n_waters, box_edge, dt_fs, temperature_k, gamma_ps, excl_indices)

    init_s, apply_s = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced, mass=mass,
        water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True, projection_site="post_o",
        settle_velocity_iters=10)

    state = init_s(jax.random.key(seed),
                   jnp.array(positions_a, dtype=jnp.float64), mass=mass)
    dof = _dof_rigid_tip3p_waters(n_waters)
    m_flat = state.mass.reshape(-1)

    def ke_of(mom):
        return float(jnp.sum(jnp.sum(mom ** 2, axis=-1) / (2.0 * m_flat)))

    pe0 = float(energy_fn(state.positions))
    ke0 = ke_of(state.momentum)
    fmax0 = float(jnp.max(jnp.abs(state.force)))
    log.info(f"\n=== mode={name} (gamma={gamma_ps} ps^-1) n_waters={n_waters} ===")
    log.info(f"step    T(K)        PE(kcal)        E_tot(kcal)     max|F|")
    log.info(f"   0  {2*ke0/(dof*BOLTZMANN_KCAL):9.1f}  {pe0:14.2f}  "
             f"{ke0+pe0:14.2f}  {fmax0:.3e}")

    trace = []
    apply_jit = jax.jit(apply_s)
    for s in range(1, steps + 1):
        state = apply_jit(state)
        ke = ke_of(state.momentum)
        t = 2.0 * ke / (dof * BOLTZMANN_KCAL)
        rec = {"step": s, "T": t}
        if (s <= 10) or (s % 10 == 0) or (not np.isfinite(t)):
            pe = float(energy_fn(state.positions))
            fmax = float(jnp.max(jnp.abs(state.force)))
            rec.update({"PE": pe, "E_tot": ke + pe, "maxF": fmax})
            log.info(f"{s:4d}  {t:9.1f}  {pe:14.2f}  {ke+pe:14.2f}  {fmax:.3e}")
        trace.append(rec)
        if not np.isfinite(t) or t > 1e6:
            log.info(f"  -> diverged/non-finite at step {s}; stopping mode")
            break
    return trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-waters", type=int, default=895)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--modes", default="thermostat,nve",
                   help="comma list: thermostat,nve")
    args = p.parse_args()

    jax.config.update("jax_enable_x64", True)

    positions_a, box_edge = _equil_water_positions(args.n_waters, seed=args.seed)
    excl = _make_tip3p_excl_indices(args.n_waters)
    log.info(f"loaded {args.n_waters} waters, box_edge={box_edge} A, "
             f"density={args.n_waters/box_edge**3:.4f} waters/A^3")

    results = {}
    if "thermostat" in args.modes:
        results["thermostat"] = run_mode(
            "thermostat", args.n_waters, positions_a, box_edge, args.steps,
            args.dt_fs, 300.0, 10.0, excl, args.seed)
    if "nve" in args.modes:
        results["nve"] = run_mode(
            "nve", args.n_waters, positions_a, box_edge, args.steps,
            args.dt_fs, 300.0, 0.0, excl, args.seed)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        log.info(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
