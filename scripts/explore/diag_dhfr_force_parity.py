"""Single-point energy + force parity, prolix vs OpenMM, on the identical DHFR config.

Why this exists (task 260903_dhfr_divergence_parity, #4953)
-----------------------------------------------------------
The DHFR NVT trajectory does not integrate: ``min_dist`` collapses from 0.956 A at
step 0 to 0.078 A by step 20 on BOTH the neighbor-list and flash force paths, while
SETTLE holds every water rigid. Six structural mechanisms have been proposed and
falsified. So this script does not test a seventh -- it *measures*, at step 0, on one
deterministic evaluation, and lets the discrepancy name the culprit.

The measurement is ordered coarsest-first on purpose:

  1. TOTAL force + energy       -- is there a disagreement at all?
  2. NONBONDED force + energy   -- the sharpest discriminator, because it covers LJ,
                                   real-space Coulomb, PME reciprocal and exclusions
                                   (the whole suspect surface) while being immune to
                                   the HBonds constraint mismatch below.
  3. BONDED force (by residual) -- total minus nonbonded.

Why nonbonded is the clean comparison
-------------------------------------
``dhfr_jac_benchmark_system.xml`` is built by ``openmmtools.testsystems.DHFRExplicit``
with ``constraints=HBonds`` and rigid water: it carries 22290 constraints on 23558
particles (21069 rigid-water + ~1221 protein X-H). OpenMM *removes* constrained bonds
from its HarmonicBondForce, whereas prolix keeps all protein bonds as full harmonic
terms. So the bonded term is expected to disagree for reasons unrelated to any bug,
and a bonded mismatch proves nothing on its own. The nonbonded term has no such
confound -- constraints do not enter a single-point force evaluation.

Cosine similarity is reported alongside RMSE because a global sign or scale error is
invisible to an RMSE quoted without a reference magnitude, and glaring in a cosine.

OpenMM runs on the CPU platform: a single-point evaluation needs no CUDA, so unlike
``bench_dhfr_parity.py`` (which times throughput and therefore does) this needs no
two-process split and no conda env -- both engines live in the workspace venv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_EXP = _ROOT / "scripts" / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

logger = logging.getLogger("diag_dhfr_force_parity")

# OpenMM works in kJ/mol and nm; prolix in kcal/mol and Angstrom.
KJ_PER_KCAL = 4.184
NM_PER_ANGSTROM = 0.1


def _openmm_per_group(system, positions_ang, box_edge_ang):
    """Per-force-group energy (kcal/mol) and forces (kcal/mol/A) at one configuration.

    Each force is put in its own group so ``getState(groups={i})`` isolates it. The
    groups are assigned here rather than read, because the serialized system ships
    every force in group 0 (verified: all 5 forces, group 0).
    """
    import numpy as np
    import openmm
    from openmm import unit as u

    group_of: dict[str, int] = {}
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i + 1)
        group_of[type(force).__name__] = i + 1

    integrator = openmm.VerletIntegrator(1.0 * u.femtosecond)
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)

    box = np.eye(3) * (box_edge_ang * NM_PER_ANGSTROM)
    context.setPeriodicBoxVectors(*box)
    context.setPositions((np.asarray(positions_ang) * NM_PER_ANGSTROM) * u.nanometer)

    out: dict[str, dict] = {}
    for name, gid in group_of.items():
        st = context.getState(getEnergy=True, getForces=True, groups={gid})
        e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / KJ_PER_KCAL
        f = np.asarray(
            st.getForces(asNumpy=True).value_in_unit(
                u.kilojoule_per_mole / u.nanometer
            )
        ) / (KJ_PER_KCAL / NM_PER_ANGSTROM)
        out[name] = {"energy": float(e), "forces": f}

    st = context.getState(getEnergy=True, getForces=True)
    out["TOTAL"] = {
        "energy": float(
            st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / KJ_PER_KCAL
        ),
        "forces": np.asarray(
            st.getForces(asNumpy=True).value_in_unit(
                u.kilojoule_per_mole / u.nanometer
            )
        )
        / (KJ_PER_KCAL / NM_PER_ANGSTROM),
    }
    del context, integrator
    return out


def _compare(label: str, f_prolix, f_omm, e_prolix, e_omm) -> dict:
    """Force/energy agreement between two engines on identical coordinates."""
    import numpy as np

    fp = np.asarray(f_prolix, dtype=np.float64)
    fo = np.asarray(f_omm, dtype=np.float64)

    diff = fp - fo
    rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    max_err = float(np.max(np.linalg.norm(diff, axis=1)))
    rms_p = float(np.sqrt(np.mean(np.sum(fp**2, axis=1))))
    rms_o = float(np.sqrt(np.mean(np.sum(fo**2, axis=1))))

    denom = float(np.linalg.norm(fp) * np.linalg.norm(fo))
    cosine = float(np.sum(fp * fo) / denom) if denom > 0 else float("nan")

    worst = int(np.argmax(np.linalg.norm(diff, axis=1)))
    return {
        "term": label,
        "energy_prolix": None if e_prolix is None else float(e_prolix),
        "energy_openmm": None if e_omm is None else float(e_omm),
        "energy_delta": None
        if (e_prolix is None or e_omm is None)
        else float(e_prolix - e_omm),
        "force_rmse": rmse,
        "force_max_err": max_err,
        "force_rms_prolix": rms_p,
        "force_rms_openmm": rms_o,
        # A ratio far from 1.0 with cosine ~1.0 means right direction, wrong scale --
        # exactly the shape a units or prefactor bug takes.
        "force_magnitude_ratio": (rms_p / rms_o) if rms_o > 0 else float("nan"),
        "force_cosine_similarity": cosine,
        "worst_atom_index": worst,
    }


def _prolix_energy_and_force(bundle, use_neighbor_list: bool):
    """Total potential energy (kcal/mol) and force (kcal/mol/A) for a bundle."""
    import jax
    import jax.numpy as jnp

    from prolix.api.bundle_md import energy_fn_from_bundle

    energy_fn = energy_fn_from_bundle(bundle)

    neighbor = None
    if use_neighbor_list:
        from prolix.physics import neighbor_list as nl
        from prolix.physics.pbc import create_periodic_space

        displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
        neighbor_fn = nl.make_neighbor_list_fn(
            displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
        )
        nbr0 = neighbor_fn.allocate(bundle.positions)
        neighbor = neighbor_fn.update(bundle.positions, nbr0)
        if bool(neighbor.did_buffer_overflow):
            raise RuntimeError(
                "neighbor list overflowed while building the parity comparison -- "
                "capacity is too small, so the NL energy below would be wrong"
            )

    def _e(r):
        return energy_fn(r, neighbor=neighbor) if neighbor is not None else energy_fn(r)

    e, grad = jax.value_and_grad(_e)(bundle.positions)
    return float(e), -grad  # F = -dE/dr


def _strip_bonded(sys_data):
    """Same fixture with every bonded term removed, so energy_fn yields nonbonded only.

    Follows the precedent in ``_bundle_from_1vii_gold`` (tests/physics/
    test_1vii_nl_nonbonded_parity.py), which deliberately omits bonded topology to
    isolate the nonbonded terms. Charges/sigmas/epsilons/exclusions are untouched.
    """
    import jax.numpy as jnp

    sys_dict, positions, box_vec, water_indices, masses, exclusion_spec, n_prot = sys_data
    stripped = dict(sys_dict)
    stripped["bonds"] = jnp.zeros((0, 2), dtype=jnp.int32)
    stripped["bond_params"] = jnp.zeros((0, 2))
    stripped["angles"] = jnp.zeros((0, 3), dtype=jnp.int32)
    stripped["angle_params"] = jnp.zeros((0, 2))
    stripped["dihedrals"] = jnp.zeros((0, 4), dtype=jnp.int32)
    stripped["dihedral_params"] = jnp.zeros((0, 4, 3))
    stripped["impropers"] = jnp.zeros((0, 4), dtype=jnp.int32)
    stripped["improper_params"] = jnp.zeros((0, 4, 3))
    return (stripped, positions, box_vec, water_indices, masses, exclusion_spec, n_prot)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Path to write the JSON report")
    ap.add_argument(
        "--force-path",
        choices=["dense", "nl"],
        default="dense",
        help="prolix direct-space kernel. dense is the default because it is the "
        "simplest path; nl cross-checks that the kernel choice does not move forces.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    import bench_dhfr_parity as B  # noqa: N812  (matches tests/bench/test_bench_dhfr_backend_split.py)
    import numpy as np
    from openmm import XmlSerializer

    B._init_jax()

    logger.info("Building prolix bundle from the DHFR fixture ...")
    ff_path = B._resolve_ff_path()
    sys_data = B._build_dhfr_system_dict(ff_path)
    bundle = B._build_bundle_from_sys_data(sys_data)
    bundle_nb = B._build_bundle_from_sys_data(_strip_bonded(sys_data))

    positions_ang = np.asarray(bundle.positions, dtype=np.float64)
    box_edge = float(np.asarray(bundle.box).reshape(-1)[0])
    n_atoms = int(positions_ang.shape[0])
    logger.info("n_atoms=%d box_edge=%.4f A", n_atoms, box_edge)

    use_nl = args.force_path == "nl"
    logger.info("prolix total energy/force (%s path) ...", args.force_path)
    e_tot_p, f_tot_p = _prolix_energy_and_force(bundle, use_nl)
    logger.info("prolix nonbonded-only energy/force ...")
    e_nb_p, f_nb_p = _prolix_energy_and_force(bundle_nb, use_nl)

    f_tot_p = np.asarray(f_tot_p, dtype=np.float64)
    f_nb_p = np.asarray(f_nb_p, dtype=np.float64)
    f_bonded_p = f_tot_p - f_nb_p

    logger.info("OpenMM per-force-group energy/force (CPU platform) ...")
    system = XmlSerializer.deserialize(Path(B.DHFR_SYSTEM_XML).read_text())
    omm = _openmm_per_group(system, positions_ang, box_edge)

    f_nb_o = omm["NonbondedForce"]["forces"]
    e_nb_o = omm["NonbondedForce"]["energy"]
    f_bonded_o = sum(
        omm[k]["forces"]
        for k in ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce")
        if k in omm
    )
    e_bonded_o = sum(
        omm[k]["energy"]
        for k in ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce")
        if k in omm
    )

    comparisons = [
        _compare("TOTAL", f_tot_p, omm["TOTAL"]["forces"], e_tot_p, omm["TOTAL"]["energy"]),
        _compare("NONBONDED", f_nb_p, f_nb_o, e_nb_p, e_nb_o),
        _compare("BONDED", f_bonded_p, f_bonded_o, e_tot_p - e_nb_p, e_bonded_o),
    ]

    report = {
        "task": "260903_dhfr_divergence_parity",
        "issue": 4953,
        "n_atoms": n_atoms,
        "box_edge_angstrom": box_edge,
        "prolix_force_path": args.force_path,
        "openmm_platform": "CPU",
        "openmm_n_constraints": int(system.getNumConstraints()),
        # The HBonds confound, quantified: prolix keeps these as harmonic terms.
        "openmm_n_harmonic_bonds": next(
            (
                int(f.getNumBonds())
                for f in system.getForces()
                if type(f).__name__ == "HarmonicBondForce"
            ),
            None,
        ),
        "prolix_n_bonds": int(np.asarray(sys_data[0]["bonds"]).shape[0]),
        "comparisons": comparisons,
    }

    for c in comparisons:
        logger.info(
            "%-10s dE=%s  cos=%.6f  ratio=%.4f  rmse=%.4g  max=%.4g",
            c["term"],
            "n/a" if c["energy_delta"] is None else f"{c['energy_delta']:+.4g}",
            c["force_cosine_similarity"],
            c["force_magnitude_ratio"],
            c["force_rmse"],
            c["force_max_err"],
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
