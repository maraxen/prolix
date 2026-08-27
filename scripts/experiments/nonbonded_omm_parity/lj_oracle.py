"""Two-particle LJ gold emit/compare for campaign nonbonded-omm-parity.

NL gold: CutoffPeriodic + OpenMM switch, pair in the switch window.
Flash gold: CutoffPeriodic at MIC diameter (half cubic box), no switch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

BOX_A = 20.0
# Mid-switch for NL (cutoff 9, switch 8). Still inside flash MIC (10 Å).
PAIR_SEP_A = 8.5
SIGMA_A = 3.0
EPS_KCAL = 0.2
MASS_DALTON = 12.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")


def emit_two_particle_lj(
    *,
    probe: str,
    cutoff_angstrom: float,
    switch_distance_angstrom: float | None,
    nonbonded_method: str,
) -> dict[str, Any]:
    """Reference-platform OpenMM gold. ``nonbonded_method``: CutoffPeriodic."""
    try:
        import openmm
        from openmm import unit
    except ImportError as e:
        raise SystemExit(f"OpenMM required for --emit-oracle: {e}") from e

    box_nm = BOX_A / 10.0
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors((box_nm, 0, 0), (0, box_nm, 0), (0, 0, box_nm))
    mass = MASS_DALTON * unit.dalton
    system.addParticle(mass)
    system.addParticle(mass)
    nb = openmm.NonbondedForce()
    if nonbonded_method != "CutoffPeriodic":
        raise ValueError(f"unsupported method {nonbonded_method!r}")
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance((cutoff_angstrom / 10.0) * unit.nanometer)
    if switch_distance_angstrom is None:
        nb.setUseSwitchingFunction(False)
    else:
        nb.setUseSwitchingFunction(True)
        nb.setSwitchingDistance((switch_distance_angstrom / 10.0) * unit.nanometer)
    q = 0.0 * unit.elementary_charge
    sig = (SIGMA_A / 10.0) * unit.nanometer
    eps = EPS_KCAL * unit.kilocalories_per_mole
    nb.addParticle(q, sig, eps)
    nb.addParticle(q, sig, eps)
    system.addForce(nb)
    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    pos_nm = np.array([[0.0, 0.0, 0.0], [PAIR_SEP_A / 10.0, 0.0, 0.0]])
    ctx.setPositions(pos_nm * unit.nanometer)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalories_per_mole
    )
    f = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilocalories_per_mole / unit.angstrom
    )
    return {
        "probe": probe,
        "openmm_version": getattr(openmm, "__version__", "unknown"),
        "openmm_platform": "Reference",
        "nonbonded_method": nonbonded_method,
        "cutoff_angstrom": float(cutoff_angstrom),
        "switch_distance_angstrom": (
            None if switch_distance_angstrom is None else float(switch_distance_angstrom)
        ),
        "energy_kcal": float(e),
        "forces_kcal_mol_A": np.asarray(f, dtype=float).tolist(),
        "positions_A": (pos_nm * 10.0).tolist(),
        "box_A": [BOX_A, BOX_A, BOX_A],
        "sigma_A": SIGMA_A,
        "epsilon_kcal": EPS_KCAL,
    }


def _campaign_row(
    *,
    probe: str,
    engine: str,
    gold: Path,
    rec: dict,
    repo: Path,
    gate_pass: int,
    delta_e: float,
    force_rmse: float,
    extra: dict | None = None,
) -> dict:
    row = {
        "campaign_slug": "nonbonded-omm-parity",
        "probe": probe,
        "engine": engine,
        "oracle_path": str(gold.relative_to(repo)) if gold.is_file() else str(gold),
        "oracle_sha256": sha256_file(gold) if gold.is_file() else "",
        "openmm_version": rec.get("openmm_version", ""),
        "gate_pass": int(gate_pass),
        "delta_e_kcal": float(delta_e),
        "force_rmse_kcal_mol_A": float(force_rmse),
        "cutoff_angstrom": float(rec.get("cutoff_angstrom", 9.0)),
        "switch_distance_angstrom": float(rec.get("switch_distance_angstrom") or 0.0),
        "skin_angstrom": 1.0,
    }
    if extra:
        row.update(extra)
    return row


def missing_gold_row(probe: str, engine: str, gold: Path) -> dict:
    return {
        "campaign_slug": "nonbonded-omm-parity",
        "probe": probe,
        "engine": engine,
        "oracle_path": str(gold),
        "oracle_sha256": "",
        "openmm_version": "",
        "gate_pass": 0,
        "delta_e_kcal": 1e9,
        "force_rmse_kcal_mol_A": 1e9,
        "cutoff_angstrom": 9.0,
        "switch_distance_angstrom": 8.0,
        "skin_angstrom": 1.0,
    }


def compare_nl_two_particle(rec: dict) -> tuple[float, float]:
    import jax
    import jax.numpy as jnp
    from jax_md import space

    jax.config.update("jax_enable_x64", True)

    from prolix.physics.optimization import chunked_lj_energy_nl

    pos = jnp.asarray(rec["positions_A"], dtype=jnp.float64)
    box = jnp.asarray(rec["box_A"], dtype=jnp.float64)
    disp, _ = space.periodic(box[0])
    n = int(pos.shape[0])
    sig = jnp.full((n,), float(rec.get("sigma_A", SIGMA_A)), dtype=jnp.float64)
    eps = jnp.full((n,), float(rec.get("epsilon_kcal", EPS_KCAL)), dtype=jnp.float64)
    excl_i = jnp.full((n, 2), -1, dtype=jnp.int32)
    excl_s = jnp.ones((n, 2), dtype=jnp.float64)
    nb = jnp.array([[1], [0]], dtype=jnp.int32)
    cutoff = float(rec["cutoff_angstrom"])
    sw_dist = rec.get("switch_distance_angstrom")
    sw = 0.0 if sw_dist in (None, 0, 0.0) else cutoff - float(sw_dist)

    def energy(r):
        return chunked_lj_energy_nl(
            r, sig, eps, excl_i, excl_s, nb, disp, cutoff, 32, sw
        )

    e_p = float(energy(pos))
    forces = -jax.grad(energy)(pos)
    gold_f = np.asarray(rec["forces_kcal_mol_A"], dtype=float)
    rmse = float(np.sqrt(np.mean((np.asarray(forces) - gold_f) ** 2)))
    return e_p - float(rec["energy_kcal"]), rmse


def compare_flash_two_particle(rec: dict) -> tuple[float, float]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from prolix.padding import PaddedSystem
    from prolix.physics.flash_explicit import _chunked_lj_only_energy

    pos = jnp.asarray(rec["positions_A"], dtype=jnp.float64)
    box = jnp.asarray(rec["box_A"], dtype=jnp.float64)
    n = int(pos.shape[0])
    sig = jnp.full((n,), float(rec.get("sigma_A", SIGMA_A)), dtype=jnp.float64)
    eps = jnp.full((n,), float(rec.get("epsilon_kcal", EPS_KCAL)), dtype=jnp.float64)
    mask = jnp.ones(n, dtype=bool)
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    sys = PaddedSystem(
        positions=pos,
        charges=jnp.zeros(n, dtype=jnp.float64),
        sigmas=sig,
        epsilons=eps,
        radii=jnp.ones(n, dtype=jnp.float64),
        scaled_radii=jnp.ones(n, dtype=jnp.float64),
        masses=jnp.ones(n, dtype=jnp.float64) * MASS_DALTON,
        element_ids=jnp.zeros(n, dtype=jnp.int32),
        atom_mask=mask,
        is_hydrogen=jnp.zeros(n, dtype=bool),
        is_backbone=jnp.zeros(n, dtype=bool),
        is_heavy=mask,
        protein_atom_mask=mask,
        water_atom_mask=jnp.zeros(n, dtype=bool),
        bonds=empty2,
        bond_params=jnp.zeros((0, 2), dtype=jnp.float64),
        bond_mask=jnp.zeros((0,), dtype=bool),
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((0, 2), dtype=jnp.float64),
        angle_mask=jnp.zeros((0,), dtype=bool),
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3), dtype=jnp.float64),
        dihedral_mask=jnp.zeros((0,), dtype=bool),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3), dtype=jnp.float64),
        improper_mask=jnp.zeros((0,), dtype=bool),
        urey_bradley_bonds=None,
        urey_bradley_params=None,
        urey_bradley_mask=None,
        excl_indices=jnp.full((n, 4), -1, dtype=jnp.int32),
        excl_scales_vdw=jnp.ones((n, 4), dtype=jnp.float64),
        excl_scales_elec=jnp.ones((n, 4), dtype=jnp.float64),
        n_padded_atoms=n,
        pme_alpha=0.0,
        box_size=box,
    )

    def energy(r):
        s = sys.replace(positions=r)
        return _chunked_lj_only_energy(r, sig, eps, mask, s, T=2, remat=False)

    e_p = float(energy(pos))
    forces = -jax.grad(energy)(pos)
    gold_f = np.asarray(rec["forces_kcal_mol_A"], dtype=float)
    rmse = float(np.sqrt(np.mean((np.asarray(forces) - gold_f) ** 2)))
    return e_p - float(rec["energy_kcal"]), rmse


def gate(delta_e: float, force_rmse: float) -> int:
    return int(abs(delta_e) <= 0.1 and force_rmse < 3.0)


def result_row(
    *,
    probe: str,
    engine: str,
    gold: Path,
    rec: dict,
    repo: Path,
    delta_e: float,
    force_rmse: float,
    e_prolix: float,
) -> dict:
    return _campaign_row(
        probe=probe,
        engine=engine,
        gold=gold,
        rec=rec,
        repo=repo,
        gate_pass=gate(delta_e, force_rmse),
        delta_e=delta_e,
        force_rmse=force_rmse,
        extra={"e_prolix_kcal": e_prolix, "e_gold_kcal": float(rec["energy_kcal"])},
    )
