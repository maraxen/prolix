"""RR5: Prolix vs OpenMM throughput parity benchmark on the DHFR JAC system (#295/#294).

Loads the 23,558-atom DHFR/TIP3P/PME benchmark fixture (materialized by
``scripts/data_prep/fetch_dhfr_benchmark.py``, backlog #295) via proxide's
CURRENT ``parse_structure`` API, runs ``n_warmup_ps`` + ``n_production_ns`` of
NVT dynamics through prolix's validated ``settle_langevin`` dt=1.0 fs path,
runs the matching step count through OpenMM (deserializing the pre-built
``dhfr_jac_benchmark_system.xml``), and reports the ns/day throughput ratio.

Two DEAD-CODE / API corrections baked in (see task 260813_triage_next_sprint,
recon 260813_dhfr_fixture_phase1):
  1. Uses ``from proxide import CoordFormat, OutputSpec, parse_structure``
     (the ``proxide.md.jax_md_bridge`` pattern in benchmark_ensembles.py is
     dead — ModuleNotFoundError against proxide==0.1.0a9).
  2. proxide's parsed ``Protein`` has no box field and, for this fixture with
     a bare protein force field, returns ALL-ZERO charges/sigma/epsilon for
     the water (HOH) region (confirmed empirically — NOT merely "shape
     populated", the values are actually zero). Box is read from the PDB's
     CRYST1 record; water nonbonded params are overridden with prolix's own
     TIP3P registry (``prolix.physics.water_models.get_water_params``),
     exactly the pattern already used by
     ``tests/physics/test_explicit_langevin_tip3p_parity.py`` and
     ``prolix.physics.topology_merger.merge_solvated_topology`` for
     solvent parameterization.

A THIRD correction found independently during this build (not anticipated by
the dispatching plan): proxide's joint protein+water topology parse includes
water's O-H bonds and H-O-H angles as ordinary harmonic bonded terms (16569
total bonds for DHFR, of which exactly 14046 = 2*7023 are pure water-water
pairs; 7023 water angles out of 11584 total). The project's VALIDATED
dt=1.0 fs SETTLE production configuration (CLAUDE.md, gate job 19774893) uses
*zero* harmonic bond/angle terms for water — SETTLE alone enforces rigidity
(see ``tests/physics/test_explicit_langevin_tip3p_parity.py::_proxide_params_pure_water``,
which explicitly zeros these). Leaving proxide's water bonds/angles in would
silently double-constrain water (SETTLE + harmonic restraint) and would not
be the same physical system CLAUDE.md's dt cap was validated on, corrupting
both the physics and the throughput comparison (extra bonded-force
evaluations per step). This script filters bonds/angles/dihedrals/impropers
to protein-only atom index ranges before constructing the energy function,
using proxide's ``molecule_type`` array to determine the protein/water atom
split point programmatically (not a hardcoded atom count).

KNOWN BLOCKER (found this build, filed as a finding — not worked around):
``--shim-mode analytical`` cannot be exercised at DHFR scale, or even at
smoke scale, because ``prolix.physics.analytical_forces.angle_forces_analytical``
and ``dihedral_forces_analytical`` are implemented as native Python ``for``
loops over every angle/dihedral term, wrapped in ``jax.grad`` — i.e. NOT
vectorized, despite the "analytical" naming. Empirically measured: JIT-
compiling ``angle_forces_analytical`` ALONE for 660 atoms / 818 angles
(1UBQ-scale) takes 61.8 seconds — already over this harness's entire 60s L2
smoke budget, before dihedrals/impropers/urey-bradley or the full BAOAB+
SETTLE integrator compile are even added, and light-years short of DHFR's
~4561 protein-only angles + thousands of dihedral terms. This script
implements ANALYTICAL mode faithfully against the current (buggy)
primitives, but guards it with ``_ANALYTICAL_MAX_BONDED_TERMS`` and raises a
clear, actionable error rather than hanging — see module-level docstring and
the final sprint report for the full finding. AUTOGRAD mode is unaffected
(``prolix.physics.bonded``'s energy functions are properly vectorized) and
is what CLAUDE.md's validated dt=1.0 fs production configuration actually
uses.

Usage:
    uv run python scripts/experiments/bench_dhfr_parity.py --dry-run --out /dev/null
    uv run python scripts/experiments/bench_dhfr_parity.py --smoke --shim-mode autograd --out /tmp/smoke.json
    uv run bth run python scripts/experiments/bench_dhfr_parity.py \\
        --shim-mode autograd --gpu-tag h200 --out outputs/bench_dhfr_parity.json --campaign <id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The validated dt=1.0 fs production configuration (gate jobs 15870804 /
# 19774893) and the explicit-solvent test suite both run in float64 --
# JAX defaults to float32, which would silently truncate positions/box/PME
# grid math and is NOT the configuration CLAUDE.md's dt cap was validated
# against. Must be set before any JAX array is created.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

DATA_DIR = ROOT / "data" / "pdb"
DHFR_PDB = DATA_DIR / "dhfr_jac_benchmark.pdb"
DHFR_SYSTEM_XML = DATA_DIR / "dhfr_jac_benchmark_system.xml"
SMOKE_PDB = DATA_DIR / "1VII.pdb"  # see module docstring: 1UBQ.pdb has no
# explicit hydrogens (0 H atoms) and proxide's AMBER parameterize_md path
# needs them; 1VII.pdb is the fixture test_solvated_explicit_integration.py
# already uses successfully for this exact solvated-explicit code path
# ("else whatever small PDB fixture the test suite already uses" per the
# dispatch instructions).

DT_FS = 1.0  # AKMA production cap, validated at production scale (CLAUDE.md)
GAMMA_PS = 10.0
TARGET_KELVIN = 300.0

# See module docstring "KNOWN BLOCKER". Empirically: 818 angles alone took
# 61.8s to JIT-compile in angle_forces_analytical (unvectorized Python loop).
# This threshold is deliberately conservative -- it exists purely to fail
# fast and legibly rather than hang for minutes on any real protein.
_ANALYTICAL_MAX_BONDED_TERMS = 100

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_ff_path(ff_name: str = "protein.ff14SB.xml") -> str:
    """Resolve a bundled proxide force-field XML by name (see scripts/benchmarks/_b1_paramize.py)."""
    import proxide

    assets = os.path.join(os.path.dirname(proxide.__file__), "assets")
    for candidate in (os.path.join(assets, ff_name), os.path.join(assets, "amber", ff_name)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Force field '{ff_name}' not found under proxide assets ({assets}).")


def _read_cryst1_box_edge(pdb_path: Path) -> float:
    """Read a cubic box edge length (Å) from a PDB's CRYST1 record."""
    with pdb_path.open() as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                a, b, c = float(line[6:15]), float(line[15:24]), float(line[24:33])
                if not (abs(a - b) < 1e-3 and abs(b - c) < 1e-3):
                    raise ValueError(f"Expected cubic box in {pdb_path}, got a={a} b={b} c={c}")
                return a
    raise ValueError(f"No CRYST1 record found in {pdb_path}")


def _filter_bonded_to_range(idx, params, n_keep_below: int):
    """Drop bonded terms (bonds/angles/dihedrals/impropers) with any atom index >= n_keep_below.

    Used to strip water's O-H bonds / H-O-H angles (proxide includes them as
    ordinary harmonic terms) so water rigidity comes ONLY from SETTLE,
    matching the project's validated dt=1.0 fs configuration. See module
    docstring correction #3.
    """
    import numpy as np

    if idx is None:
        return idx, params
    idx_np = np.asarray(idx)
    if idx_np.shape[0] == 0:
        return idx_np.astype(np.int32), params
    keep = np.all(idx_np < n_keep_below, axis=1)
    # merge_solvated_topology (prolix.physics.topology_merger) has a latent
    # dtype-promotion quirk: MergedTopology.bonds/.angles come back as
    # float64, not integer, after jnp.concatenate with the water block --
    # confirmed via direct inspection (merged.bonds.dtype == float64). Cast
    # defensively; these are always atom indices, never used as float.
    return idx_np[keep].astype(np.int32), (None if params is None else np.asarray(params)[keep])


def _build_dhfr_system_dict(ff_path: str):
    """Parse the DHFR fixture and build a filtered system dict + metadata.

    Returns (sys_dict, positions, box_vec, water_indices, masses, exclusion_spec).
    """
    import numpy as np
    import jax.numpy as jnp
    from proxide import CoordFormat, OutputSpec, parse_structure
    import proxide as proxide_mod

    from prolix.physics.neighbor_list import ExclusionSpec
    from prolix.physics.water_models import WaterModelType, get_water_params

    spec = OutputSpec(
        parameterize_md=True, remove_solvent=False, force_field=ff_path,
        coord_format=CoordFormat.Full,
    )
    protein = parse_structure(str(DHFR_PDB), spec)

    molecule_type = np.asarray(protein.molecule_type)
    protein_mask = molecule_type == 0
    n_protein_atoms = int(np.sum(protein_mask))
    if not np.array_equal(np.where(protein_mask)[0], np.arange(n_protein_atoms)):
        raise RuntimeError(
            "DHFR fixture: protein atoms (molecule_type==0) are not a contiguous "
            "leading block -- the SETTLE water_indices contiguous-OHH assumption "
            "(prolix.physics.settle.get_water_indices) would silently misassign "
            "atoms. Refusing to proceed rather than guess."
        )
    n_total = molecule_type.shape[0]
    n_water_atoms = n_total - n_protein_atoms
    if n_water_atoms % 3 != 0:
        raise RuntimeError(
            f"DHFR fixture: {n_water_atoms} solvent atoms is not a multiple of 3 "
            "(3-site water assumed) -- refusing to guess water_indices."
        )
    n_waters = n_water_atoms // 3

    elements = list(protein.elements)
    for w in range(min(n_waters, 50)):  # sanity-check O,H1,H2 ordering on a sample
        base = n_protein_atoms + 3 * w
        if elements[base] != "O" or elements[base + 1] != "H" or elements[base + 2] != "H":
            raise RuntimeError(
                f"DHFR fixture: water molecule {w} at atom {base} is not [O, H, H] "
                f"(got {elements[base:base + 3]}) -- contiguous-OHH ordering assumption "
                "used by settle.get_water_indices does not hold."
            )

    masses = np.asarray(proxide_mod.assign_masses(list(protein.atom_names)), dtype=np.float64)

    charges = np.asarray(protein.charges, dtype=np.float64).copy()
    sigmas = np.asarray(protein.sigmas, dtype=np.float64).copy()
    epsilons = np.asarray(protein.epsilons, dtype=np.float64).copy()

    # Correction #2: proxide's bare-protein-FF parse leaves water charges/LJ
    # at zero. Override with prolix's own TIP3P registry (O has LJ, H does
    # not -- standard TIP3P convention, matches
    # test_explicit_langevin_tip3p_parity.py::_proxide_params_pure_water).
    if n_waters > 0 and np.allclose(charges[n_protein_atoms:], 0.0):
        tip = get_water_params(WaterModelType.TIP3P)
        o_idx = n_protein_atoms + 3 * np.arange(n_waters)
        h_idx = np.concatenate([o_idx + 1, o_idx + 2])
        charges[o_idx] = tip.charge_O
        charges[h_idx] = tip.charge_H
        sigmas[o_idx] = tip.sigma_O
        sigmas[h_idx] = 1.0  # H has no LJ; sigma unused since epsilon=0 (matches _proxide_params_pure_water)
        epsilons[o_idx] = tip.epsilon_O
        epsilons[h_idx] = 0.0
        logger.info(
            "Overrode %d water atoms' charges/sigma/epsilon with TIP3P registry "
            "(proxide's protein-only FF assigned zero for HOH residues).",
            n_water_atoms,
        )

    exclusion_spec = ExclusionSpec.from_protein(protein)

    bonds, bond_params = _filter_bonded_to_range(protein.bonds, protein.bond_params, n_protein_atoms)
    angles, angle_params = _filter_bonded_to_range(protein.angles, protein.angle_params, n_protein_atoms)
    dihedrals, dihedral_params = _filter_bonded_to_range(
        protein.proper_dihedrals, protein.dihedral_params, n_protein_atoms
    )
    impropers, improper_params = _filter_bonded_to_range(
        protein.impropers, protein.improper_params, n_protein_atoms
    )

    sys_dict = {
        "charges": jnp.asarray(charges),
        "sigmas": jnp.asarray(sigmas),
        "epsilons": jnp.asarray(epsilons),
        "bonds": jnp.asarray(bonds) if bonds is not None else jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.asarray(bond_params) if bond_params is not None else jnp.zeros((0, 2)),
        "angles": jnp.asarray(angles) if angles is not None else jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.asarray(angle_params) if angle_params is not None else jnp.zeros((0, 2)),
        "dihedrals": jnp.asarray(dihedrals) if dihedrals is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.asarray(dihedral_params) if dihedral_params is not None else jnp.zeros((0, 4, 3)),
        "impropers": jnp.asarray(impropers) if impropers is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.asarray(improper_params) if improper_params is not None else jnp.zeros((0, 4, 3)),
    }

    box_edge = _read_cryst1_box_edge(DHFR_PDB)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    positions = jnp.asarray(np.asarray(protein.coordinates), dtype=jnp.float64)
    from prolix.physics.settle import get_water_indices
    water_indices = get_water_indices(n_protein_atoms, n_waters)

    return sys_dict, positions, box_vec, water_indices, jnp.asarray(masses), exclusion_spec, n_protein_atoms


def _build_smoke_system_dict(ff_path: str):
    """Solvate a small protein (1VII) for the L2 smoke fixture (same filtering as DHFR path)."""
    import numpy as np
    import jax.numpy as jnp
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.solvation import solvate_protein
    from prolix.physics.water_models import WaterModelType

    spec = OutputSpec(parameterize_md=True, force_field=ff_path, coord_format=CoordFormat.Full)
    protein = parse_structure(str(SMOKE_PDB), spec)

    merged = solvate_protein(
        protein, padding=8.0, model_type=WaterModelType.TIP3P,
        ionic_strength=0.0, neutralize=True,
        target_box_size=jnp.array([32.0, 32.0, 32.0]),
    )
    n_protein_atoms = int(merged.n_protein_atoms)

    bonds, bond_params = _filter_bonded_to_range(merged.bonds, merged.bond_params, n_protein_atoms)
    angles, angle_params = _filter_bonded_to_range(merged.angles, merged.angle_params, n_protein_atoms)
    dihedrals, dihedral_params = _filter_bonded_to_range(
        merged.proper_dihedrals, merged.dihedral_params, n_protein_atoms
    )
    impropers, improper_params = _filter_bonded_to_range(
        merged.impropers, merged.improper_params, n_protein_atoms
    )

    sys_dict = {
        "charges": jnp.asarray(merged.charges),
        "sigmas": jnp.asarray(merged.sigmas),
        "epsilons": jnp.asarray(merged.epsilons),
        "bonds": jnp.asarray(bonds) if bonds is not None else jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.asarray(bond_params) if bond_params is not None else jnp.zeros((0, 2)),
        "angles": jnp.asarray(angles) if angles is not None else jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.asarray(angle_params) if angle_params is not None else jnp.zeros((0, 2)),
        "dihedrals": jnp.asarray(dihedrals) if dihedrals is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.asarray(dihedral_params) if dihedral_params is not None else jnp.zeros((0, 4, 3)),
        "impropers": jnp.asarray(impropers) if impropers is not None else jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.asarray(improper_params) if improper_params is not None else jnp.zeros((0, 4, 3)),
    }

    return (
        sys_dict, jnp.asarray(merged.positions), jnp.asarray(merged.box_size),
        jnp.asarray(merged.water_indices), jnp.asarray(merged.masses),
        merged.exclusion_spec, n_protein_atoms,
    )


def _make_prolix_force_or_energy(sys_dict, positions, box_vec, exclusion_spec, shim_mode: str):
    """Build the callable passed to settle_langevin, per --shim-mode."""
    import jax
    from prolix.physics import pbc, system
    from prolix.physics.analytical_forces import (
        bond_forces_analytical, angle_forces_analytical, dihedral_forces_analytical,
        improper_forces_analytical,
    )

    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    n_angles = int(sys_dict["angles"].shape[0])
    n_dih_terms = int(sys_dict["dihedral_params"].shape[0] * max(sys_dict["dihedral_params"].shape[1], 1)) \
        if sys_dict["dihedrals"].shape[0] > 0 else 0
    n_imp = int(sys_dict["impropers"].shape[0])

    if shim_mode == "autograd":
        energy_fn = system.make_energy_fn(
            displacement_fn, sys_dict, box=box_vec, use_pbc=True,
            implicit_solvent=False, exclusion_spec=exclusion_spec,
            pme_grid_points=64, pme_alpha=0.34, cutoff_distance=9.0,
            positions=positions,
        )
        return energy_fn, displacement_fn, shift_fn

    if shim_mode == "analytical":
        if n_angles + n_dih_terms + n_imp > _ANALYTICAL_MAX_BONDED_TERMS:
            raise RuntimeError(
                f"--shim-mode analytical refused: {n_angles} angles + {n_dih_terms} dihedral "
                f"terms + {n_imp} impropers = {n_angles + n_dih_terms + n_imp} exceeds the safety "
                f"threshold ({_ANALYTICAL_MAX_BONDED_TERMS}). angle_forces_analytical and "
                "dihedral_forces_analytical (src/prolix/physics/analytical_forces.py) are "
                "unvectorized native Python for-loops wrapped in jax.grad -- empirically "
                "measured at 61.8s JIT-compile time for JUST 818 angles (660-atom system), "
                "in isolation, before dihedrals/full-integrator-compile stack on top. This is "
                "a pre-existing bug (angle/dihedral forces are not actually vectorized despite "
                "the 'analytical' name), NOT something introduced by this benchmark. Do not "
                "raise this threshold to force a run through -- fix the underlying "
                "vectorization (jax.vmap + segment_sum, matching bond_forces_analytical's own "
                "pattern) first. See scripts/experiments/bench_dhfr_parity.py module docstring "
                "'KNOWN BLOCKER' and the sprint report for task 260813_triage_next_sprint."
            )

        components = system.make_energy_fn(
            displacement_fn, sys_dict, box=box_vec, use_pbc=True,
            implicit_solvent=False, exclusion_spec=exclusion_spec,
            pme_grid_points=64, pme_alpha=0.34, cutoff_distance=9.0,
            positions=positions, return_decomposed=True,
        )
        nonbonded_energy_fn = lambda r: (
            components["lj"](r) + components["exception"](r) + components["electrostatics"](r)
            + components["cmap"](r)
        )
        bond_idx, bond_prm = sys_dict["bonds"], sys_dict["bond_params"]
        angle_idx, angle_prm = sys_dict["angles"], sys_dict["angle_params"]
        dih_idx, dih_prm = sys_dict["dihedrals"], sys_dict["dihedral_params"]
        imp_idx, imp_prm = sys_dict["impropers"], sys_dict["improper_params"]

        def force_fn(r, **kw):
            f_nonbonded = -jax.grad(nonbonded_energy_fn)(r)
            f_bond = bond_forces_analytical(r, bond_idx, bond_prm, displacement_fn) if bond_idx.shape[0] else 0.0
            f_angle = angle_forces_analytical(r, angle_idx, angle_prm, displacement_fn) if angle_idx.shape[0] else 0.0
            f_dih = dihedral_forces_analytical(r, dih_idx, dih_prm, displacement_fn) if dih_idx.shape[0] else 0.0
            f_imp = improper_forces_analytical(r, imp_idx, imp_prm, displacement_fn) if imp_idx.shape[0] else 0.0
            return f_nonbonded + f_bond + f_angle + f_dih + f_imp

        return force_fn, displacement_fn, shift_fn

    raise ValueError(f"Unknown --shim-mode {shim_mode!r}")


def _run_prolix_nvt(sys_data, shim_mode: str, n_warmup_steps: int, n_production_steps: int, seed: int):
    import jax
    import jax.numpy as jnp
    from prolix.physics import settle
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    sys_dict, positions, box_vec, water_indices, masses, exclusion_spec, n_protein_atoms = sys_data
    n_atoms = positions.shape[0]

    energy_or_force_fn, displacement_fn, shift_fn = _make_prolix_force_or_energy(
        sys_dict, positions, box_vec, exclusion_spec, shim_mode
    )

    dt_akma = DT_FS / AKMA_TIME_UNIT_FS
    kT = TARGET_KELVIN * BOLTZMANN_KCAL
    gamma = GAMMA_PS * AKMA_TIME_UNIT_FS * 1e-3

    init_fn, apply_fn = settle.settle_langevin(
        energy_or_force_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma,
        mass=masses, water_indices=water_indices, box=box_vec,
        project_ou_momentum_rigid=True, projection_site="post_o",
        settle_velocity_iters=10,
    )
    state = init_fn(jax.random.key(seed), positions, mass=masses)
    apply_j = jax.jit(apply_fn)

    def scan_body(state, _):
        return apply_j(state), None

    # Warmup (untimed, absorbs JIT compile).
    state, _ = jax.lax.scan(scan_body, state, None, length=max(n_warmup_steps, 1))
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    state, _ = jax.lax.scan(scan_body, state, None, length=n_production_steps)
    jax.block_until_ready(state)
    wallclock_s = time.perf_counter() - t0

    ns_simulated = n_production_steps * DT_FS * 1e-6
    ns_per_day = ns_simulated / wallclock_s * 86400.0 if wallclock_s > 0 else float("nan")
    return {"wallclock_s": wallclock_s, "ns_per_day": ns_per_day, "n_atoms": int(n_atoms)}


def _run_openmm_comparator(pdb_path: Path, system_xml_path: Path, n_warmup_steps: int,
                            n_production_steps: int, prefer_cuda: bool):
    import openmm
    from openmm import app, unit as omm_unit

    with system_xml_path.open() as fh:
        omm_system = openmm.XmlSerializer.deserialize(fh.read())
    pdb = app.PDBFile(str(pdb_path))

    integrator = openmm.LangevinMiddleIntegrator(
        TARGET_KELVIN * omm_unit.kelvin, GAMMA_PS / omm_unit.picoseconds,
        DT_FS * omm_unit.femtoseconds,
    )

    platform = None
    if prefer_cuda:
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
        except Exception as e:
            logger.warning("CUDA platform unavailable (%s); falling back to CPU.", e)
    if platform is None:
        platform = openmm.Platform.getPlatformByName("CPU")

    context = openmm.Context(omm_system, integrator, platform)
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(TARGET_KELVIN * omm_unit.kelvin)

    integrator.step(max(n_warmup_steps, 1))  # warmup, untimed

    t0 = time.perf_counter()
    integrator.step(n_production_steps)
    wallclock_s = time.perf_counter() - t0

    ns_simulated = n_production_steps * DT_FS * 1e-6
    ns_per_day = ns_simulated / wallclock_s * 86400.0 if wallclock_s > 0 else float("nan")
    return {
        "wallclock_s": wallclock_s, "ns_per_day": ns_per_day,
        "platform": platform.getName(), "n_atoms": omm_system.getNumParticles(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shim-mode", choices=["autograd", "analytical"], default="autograd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-warmup-ps", type=float, default=10.0)
    parser.add_argument("--n-production-ns", type=float, default=1.0)
    parser.add_argument("--gpu-tag", default="unspecified")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) if args.out != "/dev/null" else None

    if args.dry_run:
        import proxide  # noqa: F401
        import openmm  # noqa: F401
        from prolix.physics import settle, system, pbc  # noqa: F401
        from prolix.physics.analytical_forces import bond_forces_analytical  # noqa: F401

        for p in (DHFR_PDB, DHFR_SYSTEM_XML, SMOKE_PDB):
            if not p.exists():
                raise FileNotFoundError(f"dry-run: expected fixture missing: {p}")
        _resolve_ff_path()
        result = {"dry_run": True, "shim_mode": args.shim_mode}
        if out is not None:
            out.write_text(json.dumps(result))
        print("dry-run ok")
        return 0

    ff_path = _resolve_ff_path()

    if args.smoke:
        n_warmup_steps, n_production_steps = 5, 10
        pdb_path, system_xml_path = SMOKE_PDB, None
        sys_data = _build_smoke_system_dict(ff_path)
    else:
        n_warmup_steps = round(args.n_warmup_ps * 1000.0 / DT_FS)
        n_production_steps = round(args.n_production_ns * 1.0e6 / DT_FS)
        pdb_path, system_xml_path = DHFR_PDB, DHFR_SYSTEM_XML
        sys_data = _build_dhfr_system_dict(ff_path)

    logger.info(
        "shim_mode=%s smoke=%s n_warmup_steps=%d n_production_steps=%d",
        args.shim_mode, args.smoke, n_warmup_steps, n_production_steps,
    )

    prolix_result = _run_prolix_nvt(
        sys_data, args.shim_mode, n_warmup_steps, n_production_steps, args.seed
    )
    logger.info("prolix: %s", prolix_result)

    if args.smoke:
        # Build a matching tiny OpenMM system on the fly (no pre-built XML for
        # the smoke fixture) using the SAME solvated positions/box, so the
        # comparator exercises the identical code path (deserialize->Context)
        # minus the deserialize step itself (nothing to deserialize from).
        import numpy as np
        import openmm
        from openmm import unit as omm_unit
        from openmm.vec3 import Vec3

        sys_dict, positions, box_vec, water_indices, masses, exclusion_spec, n_protein_atoms = sys_data
        n_atoms = int(positions.shape[0])

        # Context only needs a System + Integrator (no Topology) -- build the
        # System's forces/constraints directly from sys_dict/water_indices
        # rather than round-tripping through a Topology + ForceField match.
        box_edge = float(np.asarray(box_vec)[0])
        box_vectors = (
            Vec3(box_edge, 0.0, 0.0) * omm_unit.angstroms,
            Vec3(0.0, box_edge, 0.0) * omm_unit.angstroms,
            Vec3(0.0, 0.0, box_edge) * omm_unit.angstroms,
        )

        omm_system = openmm.System()
        for m in np.asarray(masses):
            omm_system.addParticle(float(m))
        omm_system.setDefaultPeriodicBoxVectors(*box_vectors)
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.PME)
        nb.setCutoffDistance(9.0 * omm_unit.angstroms)
        for q, sig, eps in zip(
            np.asarray(sys_dict["charges"]), np.asarray(sys_dict["sigmas"]), np.asarray(sys_dict["epsilons"])
        ):
            nb.addParticle(float(q), float(sig) * 0.1, float(eps) * 4.184)  # Å->nm, kcal->kJ

        # 1-2/1-3/1-4 exceptions are mandatory: without them, directly-bonded
        # atoms (~1 Å apart) see full unshielded LJ/Coulomb repulsion from
        # NonbondedForce and blow up to NaN on the very first step. Build the
        # full bond graph (protein bonds, already in sys_dict, + water O-H
        # bonds reconstructed from water_indices) and let OpenMM derive
        # 1-2/1-3 exclusions + 1-4 scaling from it directly.
        bond_list = [(int(i), int(j)) for i, j in np.asarray(sys_dict["bonds"])]
        for o_i, h1_i, h2_i in np.asarray(water_indices):
            bond_list.append((int(o_i), int(h1_i)))
            bond_list.append((int(o_i), int(h2_i)))
        nb.createExceptionsFromBonds(bond_list, 0.8333333, 0.5)
        omm_system.addForce(nb)
        # Use water_indices directly (NOT a contiguous range(n_protein_atoms,
        # n_atoms, 3) assumption) -- solvate_protein's neutralize=True can
        # insert monatomic ions after the protein block, replacing some water
        # molecules, which breaks contiguous O,H,H tiling across the full
        # remaining atom range (confirmed empirically: 1VII smoke fixture had
        # 1963 total atoms = 596 protein + 3*455 water + 2 ions, not evenly
        # divisible by 3 past the protein block).
        for o_i, h1_i, h2_i in np.asarray(water_indices):
            omm_system.addConstraint(int(o_i), int(h1_i), 0.09572)
            omm_system.addConstraint(int(o_i), int(h2_i), 0.09572)
            omm_system.addConstraint(int(h1_i), int(h2_i), 0.15139)

        integrator = openmm.LangevinMiddleIntegrator(
            TARGET_KELVIN * omm_unit.kelvin, GAMMA_PS / omm_unit.picoseconds, DT_FS * omm_unit.femtoseconds
        )
        platform = openmm.Platform.getPlatformByName("CPU")
        context = openmm.Context(omm_system, integrator, platform)
        context.setPositions(np.asarray(positions) * omm_unit.angstrom)
        context.setVelocitiesToTemperature(TARGET_KELVIN * omm_unit.kelvin)
        integrator.step(max(n_warmup_steps, 1))
        t0 = time.perf_counter()
        integrator.step(n_production_steps)
        wallclock_s = time.perf_counter() - t0
        ns_simulated = n_production_steps * DT_FS * 1e-6
        openmm_result = {
            "wallclock_s": wallclock_s,
            "ns_per_day": ns_simulated / wallclock_s * 86400.0 if wallclock_s > 0 else float("nan"),
            "platform": platform.getName(), "n_atoms": n_atoms,
        }
    else:
        openmm_result = _run_openmm_comparator(
            pdb_path, system_xml_path, n_warmup_steps, n_production_steps,
            prefer_cuda=not args.smoke,
        )
    logger.info("openmm: %s", openmm_result)

    ratio = (
        prolix_result["ns_per_day"] / openmm_result["ns_per_day"]
        if openmm_result["ns_per_day"] and openmm_result["ns_per_day"] == openmm_result["ns_per_day"]
        else float("nan")
    )

    result = {
        "prolix_ns_per_day": prolix_result["ns_per_day"],
        "openmm_ns_per_day": openmm_result["ns_per_day"],
        "ratio": ratio,
        "shim_mode": args.shim_mode,
        "seed": args.seed,
        "gpu_tag": args.gpu_tag,
        "n_atoms": prolix_result["n_atoms"],
        "wallclock_s": prolix_result["wallclock_s"],
        "openmm_wallclock_s": openmm_result["wallclock_s"],
        "openmm_platform": openmm_result["platform"],
        "n_warmup_steps": n_warmup_steps,
        "n_production_steps": n_production_steps,
        "dt_fs": DT_FS,
        "smoke": args.smoke,
    }
    if out is not None:
        out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
