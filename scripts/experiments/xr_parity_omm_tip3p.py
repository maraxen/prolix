#!/usr/bin/env python3
"""XR-PARITY-OMM-WATER (#3278): OpenMM TIP3P ΔE/ΔF/T confirmatory probes.

Not a throughput benchmark. Emits durable JSON for bathos confirmation.

Usage (via bathos on Titanix)::

    CUDA_VISIBLE_DEVICES=0 uv run bth run python scripts/experiments/xr_parity_omm_tip3p.py \\
      --tag xr-parity-omm-water --campaign <id> \\
      --out outputs/xr_parity_omm_tip3p.json \\
      -- --probe all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import openmm
    from openmm import unit as omm_unit
    from openmm.app import ForceField, HBonds, PDBFile, PME

    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--probe",
        choices=("static", "temperature", "all"),
        default="all",
    )
    ap.add_argument("--out", type=Path, required=True, help="Durable JSON output path")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--n-waters-static", type=int, default=4)
    ap.add_argument("--n-waters-t", type=int, default=64)
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--gamma-ps", type=float, default=10.0)
    ap.add_argument("--temperature-k", type=float, default=300.0)
    # Liquid-density 64-water NVT: ~10 ps total @ 0.5 fs (equil subsample needs
    # ~10 ps; see test_explicit_langevin_tip3p_parity._equil_water_positions).
    # Default: 20000 steps (10 ps), burn 6667 (~3.3 ps).
    ap.add_argument("--n-steps", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=6667)
    return ap.parse_args()


def _tip3p_local_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from prolix.physics.water_models import WaterModelType, get_water_params

    tip = get_water_params(WaterModelType.TIP3P)
    r = float(tip.r_OH)
    theta = 104.52 * math.pi / 180.0
    o = np.zeros(3, dtype=np.float64)
    h1 = np.array([r, 0.0, 0.0])
    h2 = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
    return o, h1, h2


def _grid_water_positions(n_waters: int, spacing_angstrom: float) -> tuple[np.ndarray, float]:
    o0, h1l, h2l = _tip3p_local_frame()
    sites: list[tuple[int, int, int]] = []
    n = int(math.ceil(n_waters ** (1.0 / 3.0))) + 3
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                sites.append((ix, iy, iz))
                if len(sites) >= n_waters:
                    break
            if len(sites) >= n_waters:
                break
        if len(sites) >= n_waters:
            break
    sites = sites[:n_waters]
    base = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    pos: list[np.ndarray] = []
    for ix, iy, iz in sites:
        o = base + np.array(
            [ix * spacing_angstrom, iy * spacing_angstrom, iz * spacing_angstrom],
            dtype=np.float64,
        )
        pos.append(o + o0)
        pos.append(o + h1l)
        pos.append(o + h2l)
    arr = np.vstack(pos)
    span = np.max(arr, axis=0) - np.min(arr, axis=0)
    box_edge = float(np.max(span) + 16.0)
    return arr, box_edge


def _write_wat_pdb(path: Path, positions_angstrom: np.ndarray, box_angstrom: float) -> None:
    n_atoms = positions_angstrom.shape[0]
    assert n_atoms % 3 == 0
    n_res = n_atoms // 3
    lines: list[str] = [
        f"CRYST1{box_angstrom:9.3f}{box_angstrom:9.3f}{box_angstrom:9.3f}"
        f"  90.00  90.00  90.00 P 1\n",
    ]
    for i in range(n_atoms):
        serial = i + 1
        res_seq = i // 3 + 1
        x, y, z = positions_angstrom[i]
        if i % 3 == 0:
            name, elem = "O", "O"
        elif i % 3 == 1:
            name, elem = "H1", "H"
        else:
            name, elem = "H2", "H"
        lines.append(
            f"HETATM{serial:5d}  {name:<3s} HOH A{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}\n"
        )
    for r in range(n_res):
        o = 3 * r + 1
        h1 = 3 * r + 2
        h2 = 3 * r + 3
        lines.append(f"CONECT{o:5d}{h1:5d}{h2:5d}\n")
        lines.append(f"CONECT{h1:5d}{o:5d}\n")
        lines.append(f"CONECT{h2:5d}{o:5d}\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def _proxide_params_pure_water(n_waters: int):
    """TIP3P params with sparse 1-2/1-3 exclusions (required by make_energy_fn).

    Dense ``exclusion_mask`` alone is ignored by the chunked Coulomb path; without
    ``excl_indices`` / ``excl_scales_*``, intramolecular O–H / H–H Coulomb is
    included and OpenMM force RMSE blows up (~70 kcal/mol/Å on the 4-water grid).
    """
    import jax.numpy as jnp
    import numpy as np
    from prolix.physics.water_models import WaterModelType, get_water_params

    tip = get_water_params(WaterModelType.TIP3P)
    qo, qh = float(tip.charge_O), float(tip.charge_H)
    sig_o = float(tip.sigma_O)
    eps_o = float(tip.epsilon_O)
    n = n_waters * 3
    charges: list[float] = []
    sigmas: list[float] = []
    epsilons: list[float] = []
    for _ in range(n_waters):
        charges.extend([qo, qh, qh])
        sigmas.extend([sig_o, 1.0, 1.0])
        epsilons.extend([eps_o, 0.0, 0.0])
    # Per-atom padded exclusions: each water atom excludes its two partners.
    excl_indices = np.full((n, 2), -1, dtype=np.int32)
    excl_sv = np.ones((n, 2), dtype=np.float32)
    excl_se = np.ones((n, 2), dtype=np.float32)
    for w in range(n_waters):
        o, h1, h2 = 3 * w, 3 * w + 1, 3 * w + 2
        for atom, partners in ((o, (h1, h2)), (h1, (o, h2)), (h2, (o, h1))):
            for k, partner in enumerate(partners):
                excl_indices[atom, k] = partner
                excl_sv[atom, k] = 0.0
                excl_se[atom, k] = 0.0
    return {
        "charges": jnp.array(charges, dtype=jnp.float64),
        "sigmas": jnp.array(sigmas, dtype=jnp.float64),
        "epsilons": jnp.array(epsilons, dtype=jnp.float64),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "excl_indices": jnp.asarray(excl_indices),
        "excl_scales_vdw": jnp.asarray(excl_sv),
        "excl_scales_elec": jnp.asarray(excl_se),
    }


def run_static(n_waters: int) -> dict:
    """OpenMM Reference vs prolix make_energy_fn PME on a TIP3P grid."""
    if not HAS_OPENMM:
        raise RuntimeError(
            "OpenMM required for static probe; install with: uv sync --extra openmm"
        )

    import jax
    import jax.numpy as jnp
    from prolix.physics import pbc, system
    from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME

    jax.config.update("jax_enable_x64", True)

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    n_atoms = n_waters * 3
    alpha = float(REGRESSION_EXPLICIT_PME["pme_alpha_per_angstrom"])
    grid = int(REGRESSION_EXPLICIT_PME["pme_grid_points"])
    cutoff = float(REGRESSION_EXPLICIT_PME["cutoff_angstrom"])
    platform_name = str(REGRESSION_EXPLICIT_PME["openmm_platform"])
    use_disp = bool(REGRESSION_EXPLICIT_PME["use_dispersion_correction"])

    with tempfile.TemporaryDirectory() as td:
        pdb_path = Path(td) / "watbox_forces.pdb"
        _write_wat_pdb(pdb_path, positions_a, box_edge)
        pdb = PDBFile(str(pdb_path))
        ff = ForceField("amber14/tip3p.xml")
        omm_system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=cutoff / 10.0 * omm_unit.nanometer,
            constraints=HBonds,
            rigidWater=True,
        )
        for fi in range(omm_system.getNumForces()):
            f = omm_system.getForce(fi)
            if isinstance(f, openmm.NonbondedForce):
                f.setPMEParameters(alpha * 10.0, grid, grid, grid)
                f.setUseDispersionCorrection(use_disp)

        integrator = openmm.VerletIntegrator(0.001 * omm_unit.picoseconds)
        platform = openmm.Platform.getPlatformByName(platform_name)
        ctx = openmm.Context(omm_system, integrator, platform)
        ctx.setPositions(
            [
                openmm.Vec3(
                    positions_a[i, 0] / 10.0,
                    positions_a[i, 1] / 10.0,
                    positions_a[i, 2] / 10.0,
                )
                for i in range(n_atoms)
            ]
        )
        st = ctx.getState(getEnergy=True, getForces=True)
        e_omm = float(
            st.getPotentialEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
        )
        f_omm = st.getForces(asNumpy=True).value_in_unit(
            omm_unit.kilocalories_per_mole / omm_unit.angstrom
        )

    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, _shift = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=grid,
        pme_alpha=alpha,
        cutoff_distance=cutoff,
        strict_parameterization=False,
    )
    pos_jax = jnp.array(positions_a, dtype=jnp.float64)
    e_plx = float(energy_fn(pos_jax))
    f_plx = -jax.grad(energy_fn)(pos_jax)
    diff = np.asarray(f_omm, dtype=np.float64) - np.asarray(f_plx, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(diff**2)))
    delta_e = float(e_plx - e_omm)

    return {
        "n_waters_static": n_waters,
        "box_edge_angstrom": box_edge,
        "pme_alpha_per_angstrom": alpha,
        "pme_grid_points": grid,
        "cutoff_angstrom": cutoff,
        "openmm_platform": platform_name,
        "e_openmm_kcal": e_omm,
        "e_prolix_kcal": e_plx,
        "delta_e_kcal": delta_e,
        "force_rmse_kcal_mol_A": rmse,
        "static_finite": int(math.isfinite(e_omm) and math.isfinite(e_plx) and math.isfinite(rmse)),
    }


def run_temperature(
    n_waters: int,
    *,
    dt_fs: float,
    gamma_ps: float,
    temperature_k: float,
    n_steps: int,
    burn: int,
    seed: int,
) -> dict:
    """Prolix SETTLE+Langevin NVT thermometer on liquid-density TIP3P.

    Uses a ~3.1 Å spacing grid (same density as ``test_npt_barostat`` 64-water
    protocol) rather than a dilute subsample of the 30 Å asset. Rigid OU at
    ``post_o``; gamma in AKMA. Short dilute runs read ~330 K before ~10 ps.
    """
    import jax
    import jax.numpy as jnp
    from prolix.physics import pbc, settle, system
    from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME
    from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    jax.config.update("jax_enable_x64", True)

    # Liquid-density packing (≈3.1 Å O–O); box = span + padding like NPT fixture.
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=3.1)

    alpha = float(REGRESSION_EXPLICIT_PME["pme_alpha_per_angstrom"])
    grid = max(16, int(round(box_edge / 1.0)))
    cutoff = float(REGRESSION_EXPLICIT_PME["cutoff_angstrom"])
    half = 0.5 * box_edge
    if cutoff >= half:
        cutoff = float(max(half * 0.95, 1.0))

    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=grid,
        pme_alpha=alpha,
        cutoff_distance=cutoff,
        strict_parameterization=False,
    )

    dt_akma = dt_fs / float(AKMA_TIME_UNIT_FS)
    gamma_reduced = gamma_ps * float(AKMA_TIME_UNIT_FS) * 1e-3
    kT = temperature_k * float(BOLTZMANN_KCAL)
    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)
    dof = float(6 * n_waters - 3)

    init_s, apply_s = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma_reduced,
        mass=mass,
        water_indices=water_indices,
        box=box_vec,
        remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
        settle_velocity_iters=10,
    )
    apply_jit = jax.jit(apply_s)
    print("temperature: compiling apply_fn (first call)...", flush=True)
    warm = init_s(jax.random.key(seed ^ 0xA5A5), jnp.array(positions_a), mass=mass)
    warm = apply_jit(warm)
    warm.positions.block_until_ready()
    print("temperature: compile done; integrating", flush=True)

    state = init_s(jax.random.key(seed), jnp.array(positions_a), mass=mass)
    ts: list[float] = []
    for step in range(n_steps):
        state = apply_jit(state)
        if step % 500 == 0 or step == n_steps - 1:
            state.positions.block_until_ready()
            print(f"temperature: step {step}/{n_steps}", flush=True)
        if not bool(jnp.all(jnp.isfinite(state.positions))):
            return {
                "n_waters_t": n_waters,
                "dt_fs": dt_fs,
                "gamma_ps": gamma_ps,
                "n_steps": n_steps,
                "burn": burn,
                "mean_t_k": float("nan"),
                "std_t_k": float("nan"),
                "t_finite": 0,
                "gate_pass": 0,
            }
        ke = float(
            rigid_tip3p_box_ke_kcal(
                state.positions, state.momentum, state.mass, n_waters
            )
        )
        ts.append(2.0 * ke / (dof * float(BOLTZMANN_KCAL)))

    arr = np.array(ts[burn:], dtype=np.float64)
    mean_t = float(np.mean(arr))
    std_t = float(np.std(arr))
    t_ok = int(math.isfinite(mean_t) and math.isfinite(std_t))
    return {
        "n_waters_t": n_waters,
        "dt_fs": dt_fs,
        "gamma_ps": gamma_ps,
        "temperature_target_k": temperature_k,
        "n_steps": n_steps,
        "burn": burn,
        "mean_t_k": mean_t,
        "std_t_k": std_t,
        "t_finite": t_ok,
        "box_edge_angstrom_t": box_edge,
    }


def main() -> int:
    args = _parse_args()
    result: dict = {
        "campaign_slug": "xr-parity-omm-water",
        "probe": args.probe,
        "seed": args.seed,
    }

    if args.probe in ("static", "all"):
        print("probe=static starting", flush=True)
        result.update(run_static(args.n_waters_static))
        print(
            f"probe=static done delta_e={result.get('delta_e_kcal')} "
            f"rmse={result.get('force_rmse_kcal_mol_A')}",
            flush=True,
        )

    if args.probe in ("temperature", "all"):
        print(
            f"probe=temperature starting n_waters={args.n_waters_t} "
            f"n_steps={args.n_steps} burn={args.burn}",
            flush=True,
        )
        result.update(
            run_temperature(
                args.n_waters_t,
                dt_fs=args.dt_fs,
                gamma_ps=args.gamma_ps,
                temperature_k=args.temperature_k,
                n_steps=args.n_steps,
                burn=args.burn,
                seed=args.seed,
            )
        )
        print(
            f"probe=temperature done mean_t={result.get('mean_t_k')} "
            f"std_t={result.get('std_t_k')}",
            flush=True,
        )

    # Aggregate gate for sidecar DuckDB
    delta_e = float(result.get("delta_e_kcal", float("nan")))
    force_rmse = float(result.get("force_rmse_kcal_mol_A", float("nan")))
    mean_t = float(result.get("mean_t_k", float("nan")))
    static_ok = (
        math.isfinite(delta_e)
        and math.isfinite(force_rmse)
        and abs(delta_e) <= 0.1
        and force_rmse < 3.0
    )
    t_ok = math.isfinite(mean_t) and abs(mean_t - 300.0) <= 15.0
    if args.probe == "static":
        gate_pass = int(static_ok)
    elif args.probe == "temperature":
        gate_pass = int(t_ok)
    else:
        gate_pass = int(static_ok and t_ok)
    result["gate_pass"] = gate_pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "gate_pass": gate_pass}, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
