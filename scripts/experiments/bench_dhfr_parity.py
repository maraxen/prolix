"""RR5: Prolix vs OpenMM throughput parity benchmark on the DHFR JAC system (#295/#294).

Loads the 23,558-atom DHFR/TIP3P/PME benchmark fixture (materialized by
``scripts/data_prep/fetch_dhfr_benchmark.py``, backlog #295) via proxide's
CURRENT ``parse_structure`` API, wraps the corrected topology into a real
``MolecularBundle`` (``prolix.physics.system.make_bundle_from_system``), and
dispatches production NVT dynamics through the ACTUAL production entry point
-- ``prolix.api.EnsemblePlan`` (composed xtrax: ``EnsemblePlan`` routes
batching through ``xtrax.tiling`` primitives via
``prolix.tiling.xtrax_adapter`` / ``prolix.api.ensemble_dispatch`` /
``prolix.api.ensemble_dedup``) -- rather than hand-rolling
``settle_langevin`` + a raw ``jax.lax.scan`` loop in this script. Runs the
matching step count through OpenMM (deserializing the pre-built
``dhfr_jac_benchmark_system.xml``) and reports the ns/day throughput ratio.

Two DEAD-CODE / API corrections baked in (see task 260813_triage_next_sprint,
recon 260813_dhfr_fixture_phase1):
  1. Uses ``from proxide import CoordFormat, OutputSpec, parse_structure``
     (the ``proxide.md.jax_md_bridge`` pattern in benchmark_ensembles.py is
     dead -- ModuleNotFoundError against proxide==0.1.0a9).
  2. proxide's parsed ``Protein`` has no box field and, for this fixture with
     a bare protein force field, returns ALL-ZERO charges/sigma/epsilon for
     the water (HOH) region (confirmed empirically -- NOT merely "shape
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
*zero* harmonic bond/angle terms for water -- SETTLE alone enforces rigidity
(see ``tests/physics/test_explicit_langevin_tip3p_parity.py::_proxide_params_pure_water``,
which explicitly zeros these). Leaving proxide's water bonds/angles in would
silently double-constrain water (SETTLE + harmonic restraint) and would not
be the same physical system CLAUDE.md's dt cap was validated on, corrupting
both the physics and the throughput comparison (extra bonded-force
evaluations per step). This script filters bonds/angles/dihedrals/impropers
to protein-only atom index ranges before constructing the bundle, using
proxide's ``molecule_type`` array to determine the protein/water atom split
point programmatically (not a hardcoded atom count).

A FOURTH correction (this build, task 260814_dhfr-bench-redesign): the
original version of this script hand-rolled the physics stack directly
(``settle.settle_langevin`` + a raw ``jax.lax.scan``) and timed a naive
warmup/production ``lax.scan`` pair with DIFFERENT static ``length`` values.
Since ``lax.scan``'s ``length`` is a trace-time-static constant, the two
calls compile as two distinct XLA programs -- the "timed" call silently ate
a full fresh JIT compile, meaning the reported ns/day was compile-time-
contaminated, not real per-step throughput. Fixed by (a) calling the actual
production entry point (``EnsemblePlan.from_bundle(bundle).run(...,
run_mode="inference")``) instead of reimplementing the integrator here, and
(b) decomposing compile vs. compute via a linear-regression sweep across
``--n-steps-list`` (same methodology as
``scripts/experiments/profile_b1_heterogeneous_solvated_compile.py``, already
validated on real cluster runs): per-call fixed/compile overhead is roughly
independent of ``n_steps`` while real compute scales linearly with it, so
``np.polyfit(n_steps - 1, t_steady_state)`` robustly recovers true per-step
compute time (the slope, ``per_step_corrected_s``) and fixed overhead (the
intercept, ``compile_fixed_s``), with ``r_squared`` as a built-in fit-quality
sanity check -- regardless of whether JAX caches or re-pays compilation
across separate calls (verified empirically NOT cached across ``EnsemblePlan
.run()`` calls; the regression is robust either way, since it only assumes
overhead is roughly constant in ``n_steps``, not that it's paid once).

KNOWN BLOCKER, now OUT OF SCOPE rather than worked around (found in the
original build, filed as a finding): the old ``--shim-mode analytical`` path
hand-rolled a force function using
``prolix.physics.analytical_forces.angle_forces_analytical`` /
``dihedral_forces_analytical``, which are unvectorized native Python ``for``
loops wrapped in ``jax.grad`` (empirically 61.8s to JIT-compile for just 818
angles at 1UBQ scale). The ACTUAL production path this script now calls
(``EnsemblePlan`` -> ``prolix.api.bundle_md.energy_fn_from_bundle``) has no
such switch -- it always uses the properly vectorized autograd bonded energy
functions in ``prolix.physics.bonded``. "Analytical mode" was never a real
production configuration; it was a benchmark-only reimplementation. Passing
``--shim-mode analytical`` now raises immediately rather than attempting a
reimplementation this script no longer maintains -- see ``main()``. The
underlying vectorization bug (backlog #4241) is unaffected by this change and
remains open upstream.

Usage:
    uv run python scripts/experiments/bench_dhfr_parity.py --dry-run --out /dev/null
    uv run python scripts/experiments/bench_dhfr_parity.py --smoke --out /tmp/smoke.json
    uv run bth run python scripts/experiments/bench_dhfr_parity.py \\
        --gpu-tag h200 --out outputs/bench_dhfr_parity.json --campaign <id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Lazy JAX initialization moved to _init_jax() helper to allow running
# OpenMM without JAX import (the CUDA-capable OpenMM env has no JAX/prolix).
jax = None

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
PME_ALPHA = 0.34
NONBONDED_CUTOFF = 9.0

# Fixed smoke-mode compile/compute regression sweep -- user-selected
# (AskUserQuestion, task 260814_dhfr-bench-redesign): small enough to run in
# the L2 smoke budget while still giving >=2 points for a real regression fit.
_SMOKE_N_STEPS_LIST = [5, 20, 50, 100]

# Full-mode calibration points below the actual production target -- the
# target step count (derived from --n-production-ns) is appended as the
# LARGEST point per the user's explicit design choice (AskUserQuestion:
# "Include as largest sweep point"), so the regression's top point is a real
# production-scale run, not just an extrapolation.
_FULL_CALIBRATION_STEPS = [50, 200, 1000, 5000]

# OpenMM's kernel compiles once at Context creation (not per step-count, so
# no regression is needed there); this fixed warmup absorbs that one-time
# cost before the timed OpenMM window.
_OPENMM_WARMUP_STEPS = 500

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _init_jax():
    """Import JAX and enable float64. MUST run before any JAX array is created.

    The validated dt=1.0 fs production configuration (gate jobs 15870804 /
    19774893) and the explicit-solvent test suite both run in float64 --
    JAX defaults to float32, which would silently truncate positions/box/PME
    grid math and is NOT the configuration CLAUDE.md's dt cap was validated
    against. Must be set before any JAX array is created.
    """
    global jax
    if jax is not None:
        return jax  # Already initialized
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
    jax = _jax
    return _jax


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


def _build_bundle_from_sys_data(sys_data):
    """Wrap corrected topology components into a real MolecularBundle.

    Uses ``make_bundle_from_system`` (src/prolix/physics/system.py) directly
    on the ALREADY-assembled (and already-corrected: TIP3P override, water
    bonded-term filtering) topology components -- NOT
    ``solvate_protein_to_bundle``, which would re-solvate from scratch and
    silently produce a different water count/box than the reference
    ``dhfr_jac_benchmark_system.xml`` OpenMM system, breaking parity.
    """
    from types import SimpleNamespace

    import jax.numpy as jnp

    from prolix.physics.system import make_bundle_from_system

    sys_dict, positions, box_vec, water_indices, masses, exclusion_spec, _n_protein_atoms = sys_data

    # Force per-atom float arrays to a single consistent dtype (positions'
    # own dtype). Without this, sys_dict["charges"] (proxide-sourced) came
    # out float32 while positions/box_size came out float64 under this
    # script's jax_enable_x64=True -- an input-level precision mismatch, not
    # a pme.py/flash_explicit.py code bug. That mismatch is what actually
    # produced the persistent "lax.add requires ... float32, float64" crash
    # under use_flash_forces (debt 832/835's own extensive dtype-hardcode
    # fixes were necessary but not sufficient -- confirmed via direct
    # instrumentation of every intermediate in pme.py's _spme_fwd).
    work_dtype = jnp.asarray(positions).dtype
    ns = SimpleNamespace(
        positions=jnp.asarray(positions, dtype=work_dtype),
        box_size=jnp.asarray(box_vec, dtype=work_dtype),
        masses=jnp.asarray(masses, dtype=work_dtype),
        charges=jnp.asarray(sys_dict["charges"], dtype=work_dtype),
        sigmas=jnp.asarray(sys_dict["sigmas"], dtype=work_dtype),
        epsilons=jnp.asarray(sys_dict["epsilons"], dtype=work_dtype),
        bonds=sys_dict["bonds"],
        bond_params=sys_dict["bond_params"],
        angles=sys_dict["angles"],
        angle_params=sys_dict["angle_params"],
        dihedrals=sys_dict["dihedrals"],
        dihedral_params=sys_dict["dihedral_params"],
        impropers=sys_dict["impropers"],
        improper_params=sys_dict["improper_params"],
        water_indices=water_indices,
        pme_alpha=PME_ALPHA,
        nonbonded_cutoff=NONBONDED_CUTOFF,
    )
    return make_bundle_from_system(ns, boundary_condition="periodic", exclusion_spec=exclusion_spec)


def _run_prolix_regression(sys_data, n_steps_list, seed: int):
    """Time EnsemblePlan.run() across a step-count sweep and regress compile vs. compute.

    Mirrors scripts/experiments/profile_b1_heterogeneous_solvated_compile.py's
    validated methodology exactly: for each n_steps, time a fresh 1-step call
    (t_first_step, dominated by whatever per-call overhead exists) and a
    separate (n_steps - 1)-step call on a NEW EnsemblePlan (t_steady_state).
    Fit t_steady_state = per_step_corrected_s * (n_steps - 1) + compile_fixed_s
    via least squares across the sweep -- this recovers true per-step compute
    (the slope) independent of whether JIT compilation is cached or re-paid
    per call, since only the intercept (not the trend) absorbs fixed overhead.
    """
    from prolix.api import EnsemblePlan
    from prolix.simulate import BOLTZMANN_KCAL

    bundle = _build_bundle_from_sys_data(sys_data)
    kT = TARGET_KELVIN * BOLTZMANN_KCAL
    # use_neighbor_list=True is the production default for this workload
    # (_resolve_nonbonded_defaults: "NL inference+PBC, Flash trajectory+PBC"), and it
    # is now the only path with correct asymptotics at DHFR scale. Flash evaluates all
    # N^2 pairs with no distance cutoff at all (flash_explicit.py's tile body masks
    # only self-pairs and padding), i.e. ~566M pair evaluations per step here, doubled
    # again by remat's backward recompute. NL does O(N*K) with a real cutoff.
    #
    # This override previously read use_flash_forces=True, because the NL path OOMed a
    # single H200 at init (job 20528981). That has two causes, both now fixed:
    #   * the cell list silently dropped 19.6% of in-cutoff pairs (#4699), so NL had to
    #     run with disable_cell_list=True -- brute-force O(N^2) list construction, which
    #     threw away NL's whole asymptotic advantage;
    #   * _setup_integrator built two dense (N, N) exclusion matrices (~4.4 GB here) that
    #     the NL branch never reads. energy_fn_from_bundle now takes
    #     build_dense_exclusions, and ensemble_plan passes False whenever a neighbor is
    #     bound.
    # NL and flash remain mutually exclusive (ensemble_plan raises if both are set), and
    # NL requires run_mode="inference", which both legs below already use.

    def _block(traj) -> None:
        trajs = traj if isinstance(traj, list) else [traj]
        for t in trajs:
            jax.block_until_ready(t.positions)

    results = []
    for n_steps in n_steps_list:
        plan = EnsemblePlan.from_bundle(bundle)

        t_first0 = time.perf_counter()
        first = plan.run(
            n_steps=1, dt=DT_FS, kT=kT, seed=seed, gamma=GAMMA_PS, run_mode="inference", use_neighbor_list=True
        )
        _block(first)
        t_first = time.perf_counter() - t_first0
        del first

        t_ss = 0.0
        if n_steps > 1:
            t_ss0 = time.perf_counter()
            last = plan.run(
                n_steps=n_steps - 1,
                dt=DT_FS,
                kT=kT,
                seed=seed + 1,
                gamma=GAMMA_PS,
                run_mode="inference",
                use_neighbor_list=True,
            )
            _block(last)
            t_ss = time.perf_counter() - t_ss0
            del last

        results.append({"n_steps": n_steps, "t_first_step": t_first, "t_steady_state": t_ss})
        logger.info("n_steps=%d: t_first=%.4fs t_steady_state=%.4fs", n_steps, t_first, t_ss)

    x_vals = [r["n_steps"] - 1 for r in results if r["n_steps"] > 1]
    y_vals = [r["t_steady_state"] for r in results if r["n_steps"] > 1]
    slope = intercept = r_squared = float("nan")
    if len(x_vals) >= 2:
        x_arr, y_arr = np.array(x_vals, dtype=np.float64), np.array(y_vals, dtype=np.float64)
        coeffs = np.polyfit(x_arr, y_arr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        y_pred = slope * x_arr + intercept
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        logger.info("Regression: per_step=%.6e s, compile_fixed=%.6e s, R^2=%.6f", slope, intercept, r_squared)
    else:
        logger.warning("n_steps_list had < 2 points with n_steps > 1 -- cannot fit a regression.")

    ns_per_day = (
        DT_FS * 1e-6 / slope * 86400.0 if slope == slope and slope > 0 else float("nan")  # noqa: PLR0133 (NaN check)
    )

    return {
        "per_step_corrected_s": slope,
        "compile_fixed_s": intercept,
        "r_squared": r_squared,
        "ns_per_day": ns_per_day,
        "n_atoms": int(bundle.n_atoms),
        "n_steps_list": list(n_steps_list),
        "raw_results": results,
    }


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
            raise RuntimeError(
                f"CUDA platform unavailable for OpenMM (required for GPU-to-GPU speed comparison). "
                f"Error: {e}. "
                f"Fix: ensure CUDA-capable hardware is present, CUDA toolkit is installed, "
                f"and OpenMM was built with CUDA support. "
                f"Pass prefer_cuda=False to use CPU (not recommended for production benchmarks)."
            )
    else:
        platform = openmm.Platform.getPlatformByName("CPU")

    context = openmm.Context(omm_system, integrator, platform)
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(TARGET_KELVIN * omm_unit.kelvin)

    integrator.step(max(n_warmup_steps, 1))  # warmup, untimed -- OpenMM compiles its kernel once here

    t0 = time.perf_counter()
    integrator.step(n_production_steps)
    wallclock_s = time.perf_counter() - t0

    ns_simulated = n_production_steps * DT_FS * 1e-6
    ns_per_day = ns_simulated / wallclock_s * 86400.0 if wallclock_s > 0 else float("nan")
    return {
        "wallclock_s": wallclock_s, "ns_per_day": ns_per_day,
        "platform": platform.getName(), "n_atoms": omm_system.getNumParticles(),
        "n_production_steps": n_production_steps,
    }


def _combine_results(prolix_json_path: Path, openmm_json_path: Path) -> dict:
    """Merge prolix and openmm benchmark results, validating consistency.

    Raises RuntimeError if the runs are incompatible (different smoke modes, step
    counts, or openmm was not on CUDA).
    """
    with prolix_json_path.open() as fh:
        prolix = json.load(fh)
    with openmm_json_path.open() as fh:
        openmm = json.load(fh)

    # Validate backend fields
    if prolix.get("backend") != "prolix":
        raise RuntimeError(
            f"prolix result has unexpected backend field: {prolix.get('backend')} "
            "(expected 'prolix'); was it created with --backend prolix?"
        )
    if openmm.get("backend") != "openmm":
        raise RuntimeError(
            f"openmm result has unexpected backend field: {openmm.get('backend')} "
            "(expected 'openmm'); was it created with --backend openmm?"
        )

    # Validate smoke modes match
    prolix_smoke = prolix.get("smoke", False)
    openmm_smoke = openmm.get("smoke", False)
    if prolix_smoke != openmm_smoke:
        raise RuntimeError(
            f"smoke modes do not match: prolix={prolix_smoke}, openmm={openmm_smoke} "
            "-- both must be run with the same --smoke flag"
        )

    # Validate step counts match
    prolix_max_steps = max(prolix.get("n_steps_list", []))
    openmm_prod_steps = openmm.get("openmm_n_production_steps")
    if prolix_max_steps != openmm_prod_steps:
        raise RuntimeError(
            f"step counts do not match: prolix max n_steps={prolix_max_steps}, "
            f"openmm n_production_steps={openmm_prod_steps} -- both must run the same "
            "number of steps so the ns/day comparison is apples-to-apples"
        )

    # Validate OpenMM was on CUDA
    openmm_platform = openmm.get("openmm_platform")
    if openmm_platform != "CUDA":
        raise RuntimeError(
            f"openmm platform is '{openmm_platform}', not 'CUDA' -- a CPU baseline "
            "must never silently become the denominator. Fix: re-run with "
            "--backend openmm in the CUDA-capable OpenMM environment"
        )

    # Compute ratio
    prolix_ns = prolix.get("prolix_ns_per_day")
    openmm_ns = openmm.get("openmm_ns_per_day")
    ratio = (
        prolix_ns / openmm_ns
        if prolix_ns and openmm_ns and prolix_ns == prolix_ns and openmm_ns == openmm_ns
        else float("nan")
    )

    # Merge results
    merged = {
        **prolix,
        **openmm,
        "backend": "combined",
        "prolix_gpu_tag": prolix.get("gpu_tag"),
        "openmm_gpu_tag": openmm.get("gpu_tag"),
        "ratio": ratio,
    }
    # Remove the separate gpu_tag (ambiguous now that we have both)
    merged.pop("gpu_tag", None)
    return merged


def _run_openmm_smoke_comparator(sys_data, n_warmup_steps: int, n_production_steps: int):
    """Build a matching tiny OpenMM system on the fly (no pre-built XML for the smoke
    fixture) using the SAME solvated positions/box, so the comparator exercises the
    identical code path (deserialize->Context) minus the deserialize step itself."""
    import numpy as np
    import openmm
    from openmm import unit as omm_unit
    from openmm.vec3 import Vec3

    sys_dict, positions, box_vec, water_indices, masses, _exclusion_spec, _n_protein_atoms = sys_data
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
    return {
        "wallclock_s": wallclock_s,
        "ns_per_day": ns_simulated / wallclock_s * 86400.0 if wallclock_s > 0 else float("nan"),
        "platform": platform.getName(), "n_atoms": n_atoms,
        "n_production_steps": n_production_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shim-mode", choices=["autograd", "analytical"], default="autograd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-production-ns", type=float, default=1.0)
    parser.add_argument(
        "--n-steps-list", default=None,
        help=(
            "Comma-separated compile/compute regression sweep for full (non-smoke) mode. "
            "Default: calibration points 50,200,1000,5000 plus the --n-production-ns-derived "
            "target step count appended as the largest point. Ignored in --smoke mode, which "
            "always uses a fixed 5,20,50,100 sweep."
        ),
    )
    parser.add_argument("--gpu-tag", default="unspecified")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--backend", choices=["both", "prolix", "openmm"], default="both",
        help=(
            "Which backend to run: 'both' (default, byte-for-byte unchanged from original), "
            "'prolix' (prolix only, openmm fields set to None), "
            "'openmm' (openmm only, prolix fields set to None). "
            "Use 'prolix' and 'openmm' separately with the appropriate environment "
            "(CUDA-capable OpenMM env for 'openmm', prolix-installed env for 'prolix'), "
            "then combine with --combine-prolix and --combine-openmm."
        ),
    )
    parser.add_argument(
        "--combine-prolix", default=None, type=Path,
        help="Path to JSON from --backend prolix run. Must also provide --combine-openmm.",
    )
    parser.add_argument(
        "--combine-openmm", default=None, type=Path,
        help="Path to JSON from --backend openmm run. Must also provide --combine-prolix.",
    )
    args = parser.parse_args()

    out = Path(args.out) if args.out != "/dev/null" else None

    # Handle combine mode first (no run, just merge)
    if args.combine_prolix or args.combine_openmm:
        if not (args.combine_prolix and args.combine_openmm):
            raise RuntimeError(
                "--combine-prolix and --combine-openmm must both be provided together"
            )
        if not args.combine_prolix.exists():
            raise FileNotFoundError(f"prolix result JSON not found: {args.combine_prolix}")
        if not args.combine_openmm.exists():
            raise FileNotFoundError(f"openmm result JSON not found: {args.combine_openmm}")
        combined = _combine_results(args.combine_prolix, args.combine_openmm)
        if out is not None:
            out.write_text(json.dumps(combined, indent=2))
        print(json.dumps(combined, indent=2))
        return 0

    # Validate that --backend is compatible with other flags
    if args.backend == "openmm" and args.shim_mode == "analytical":
        raise RuntimeError(
            "--backend openmm is incompatible with --shim-mode analytical "
            "(analytical mode does not exist in production EnsemblePlan)"
        )

    if args.shim_mode == "analytical":
        # See module docstring "KNOWN BLOCKER": the production EnsemblePlan
        # path this script now calls has no analytical-forces switch -- it
        # always dispatches through the vectorized autograd bonded energy
        # functions (prolix.physics.bonded). Analytical mode was only ever a
        # benchmark-only hand-rolled alternative; this script no longer
        # reimplements the physics stack, so there is nothing left to guard
        # with a bonded-term-count threshold -- refuse immediately.
        raise RuntimeError(
            "--shim-mode analytical is not runnable: this benchmark now dispatches through "
            "the actual production entry point (prolix.api.EnsemblePlan), which has no "
            "analytical-bonded-forces mode -- it always uses the vectorized autograd bonded "
            "energy functions in prolix.physics.bonded. The old analytical hand-rolled path "
            "(prolix.physics.analytical_forces.angle_forces_analytical / "
            "dihedral_forces_analytical) was never part of the validated production "
            "configuration and remains unvectorized upstream (backlog #4241, still open). "
            "Use --shim-mode autograd (the default)."
        )

    # For openmm-only non-smoke mode, we don't need to resolve the force field
    # (which would import proxide and transitively JAX). Smoke mode and prolix
    # mode both need it.
    need_ff_path = args.backend != "openmm" or args.smoke

    if args.dry_run:
        if args.backend in ("prolix", "both"):
            _init_jax()
            import proxide  # noqa: F401
            from prolix.api import EnsemblePlan  # noqa: F401
            from prolix.physics.system import make_bundle_from_system  # noqa: F401

        if args.backend in ("openmm", "both"):
            import openmm  # noqa: F401

        for p in (DHFR_PDB, DHFR_SYSTEM_XML, SMOKE_PDB):
            if not p.exists():
                raise FileNotFoundError(f"dry-run: expected fixture missing: {p}")
        if need_ff_path:
            _resolve_ff_path()
        result = {"dry_run": True, "shim_mode": args.shim_mode, "backend": args.backend}
        if out is not None:
            out.write_text(json.dumps(result))
        print("dry-run ok")
        return 0

    # Resolve force field path only if needed (prolix mode or smoke mode)
    ff_path = _resolve_ff_path() if need_ff_path else None

    # Determine which test fixtures and step lists to use
    if args.smoke:
        n_steps_list = list(_SMOKE_N_STEPS_LIST)
        pdb_path, system_xml_path = SMOKE_PDB, None
    else:
        n_production_steps = round(args.n_production_ns * 1.0e6 / DT_FS)
        if args.n_steps_list:
            n_steps_list = sorted({int(x) for x in args.n_steps_list.split(",")})
        else:
            n_steps_list = sorted({*_FULL_CALIBRATION_STEPS, n_production_steps})
        pdb_path, system_xml_path = DHFR_PDB, DHFR_SYSTEM_XML

    logger.info("shim_mode=%s smoke=%s backend=%s n_steps_list=%s",
                args.shim_mode, args.smoke, args.backend, n_steps_list)

    # Run prolix if requested
    prolix_result = None
    if args.backend in ("prolix", "both"):
        _init_jax()
        if args.smoke:
            sys_data = _build_smoke_system_dict(ff_path)
        else:
            sys_data = _build_dhfr_system_dict(ff_path)
        prolix_result = _run_prolix_regression(sys_data, n_steps_list, args.seed)
        logger.info(
            "prolix: per_step=%.6e s compile_fixed=%.6e s R^2=%.4f ns_per_day=%.4f",
            prolix_result["per_step_corrected_s"], prolix_result["compile_fixed_s"],
            prolix_result["r_squared"], prolix_result["ns_per_day"],
        )

    # Run OpenMM if requested
    openmm_result = None
    if args.backend in ("openmm", "both"):
        # For non-smoke mode, OpenMM only needs the file paths, not the full sys_data
        # which requires JAX. Smoke mode needs sys_data for the OpenMM system builder.
        if args.smoke:
            if prolix_result is None:
                _init_jax()
                sys_data = _build_smoke_system_dict(ff_path)
            # else sys_data was already built above
            n_production_steps_for_comparator = max(n_steps_list)
            openmm_result = _run_openmm_smoke_comparator(
                sys_data, n_warmup_steps=min(n_steps_list), n_production_steps=n_production_steps_for_comparator
            )
        else:
            # Non-smoke: use the pre-built XML and PDB, no sys_data needed
            n_production_steps_for_comparator = max(n_steps_list) if prolix_result else max([
                round(args.n_production_ns * 1.0e6 / DT_FS)
            ])
            openmm_result = _run_openmm_comparator(
                pdb_path, system_xml_path, _OPENMM_WARMUP_STEPS, n_production_steps_for_comparator,
                prefer_cuda=True,
            )
        logger.info("openmm: %s", openmm_result)

    # Build result dict with appropriate None values based on backend
    if prolix_result is None:
        result_prolix = {
            "prolix_ns_per_day": None,
            "per_step_corrected_s": None,
            "compile_fixed_s": None,
            "r_squared": None,
            "n_atoms": None,
            "n_steps_list": None,
            "raw_results": None,
        }
    else:
        result_prolix = {
            "prolix_ns_per_day": prolix_result["ns_per_day"],
            "per_step_corrected_s": prolix_result["per_step_corrected_s"],
            "compile_fixed_s": prolix_result["compile_fixed_s"],
            "r_squared": prolix_result["r_squared"],
            "n_atoms": prolix_result["n_atoms"],
            "n_steps_list": prolix_result["n_steps_list"],
            "raw_results": prolix_result["raw_results"],
        }

    if openmm_result is None:
        result_openmm = {
            "openmm_ns_per_day": None,
            "openmm_wallclock_s": None,
            "openmm_platform": None,
            "openmm_n_production_steps": None,
        }
    else:
        result_openmm = {
            "openmm_ns_per_day": openmm_result["ns_per_day"],
            "openmm_wallclock_s": openmm_result["wallclock_s"],
            "openmm_platform": openmm_result["platform"],
            "openmm_n_production_steps": openmm_result["n_production_steps"],
        }

    ratio = float("nan")
    if (prolix_result and openmm_result and
        prolix_result["ns_per_day"] and openmm_result["ns_per_day"] and
        prolix_result["ns_per_day"] == prolix_result["ns_per_day"] and
        openmm_result["ns_per_day"] == openmm_result["ns_per_day"]):
        ratio = prolix_result["ns_per_day"] / openmm_result["ns_per_day"]

    result = {
        **result_prolix,
        **result_openmm,
        "ratio": ratio,
        "shim_mode": args.shim_mode,
        "seed": args.seed,
        "gpu_tag": args.gpu_tag,
        "dt_fs": DT_FS,
        "smoke": args.smoke,
        "backend": args.backend,
    }
    if out is not None:
        out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
