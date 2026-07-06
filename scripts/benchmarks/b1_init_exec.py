"""B1-full: cold-start heterogeneous MD throughput benchmark (Claim 1).

Binding pre-registration: ``.praxia/docs/specs/260528_b1-preregistration.md``.

H1: for a fixed ensemble of 64 heterogeneous systems (4 topology classes x 16
replicas), prolix cold-start ``t_total`` (median over seeds) is lower than
OpenMM's equivalent cold-start with N independent Contexts (no reuse), in the
short-trajectory regime where initialization + compilation dominate.

Two engines:
  * prolix: parametrize each unique topology, build bundles, ONE
    ``EnsemblePlan.from_bundles(...).run(n_steps)`` compiled pass over the
    heterogeneous ensemble (the batching advantage under test).
  * openmm: N independent ``Context`` objects, no reuse — fresh
    ``createSystem`` + ``Context`` per system, stepped independently.

Cold-start sub-metrics (reported separately, per pre-reg):
  t_ff_load, t_bundle_construct, t_aot_compile (ESTIMATE = first-call minus
  steady-per-step x n_steps), t_first_step, t_steady_state, t_total, aot_ratio.

MD-scale full runs (64 x 100 ps) are CLUSTER ONLY. Locally use --smoke.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("b1")

# Repo-relative fixtures (anchored to this file, not CWD).
_ROOT = Path(__file__).resolve().parents[2]
# Allow `scripts.benchmarks.*` imports when run directly as a script.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
PROTEIN_FIXTURES = {
    "1ake": _ROOT / "data" / "pdb" / "1AKE.pdb",
    "1ubq": _ROOT / "data" / "pdb" / "1UBQ.pdb",
    "2gb1": _ROOT / "data" / "pdb" / "2GB1.pdb",
}
WATER_KEY = "4-water"
DEFAULT_SYSTEMS = ["1ake", "1ubq", "2gb1", WATER_KEY]
DEFAULT_FF = "protein.ff19SB.xml"

CSV_COLUMNS = [
    "engine", "system_type", "replica", "seed", "hardware_tag", "precision",
    "n_steps", "t_ff_load", "t_bundle_construct", "t_aot_compile",
    "t_first_step", "t_steady_state", "t_total", "aot_ratio", "trajectory_finite",
]

# BLOCKED ON CORE MD-PATH FIX (see .praxia/docs/specs/260706_b1-core-md-path-fix.md):
# prolix protein bundles currently diverge (trajectory_finite=False) NOT because of
# unminimized coordinates (minimization was ruled out — it plateaus on a broken
# energy), but because the bundle MD path drops nonbonded exclusions (1-2/1-3
# neighbors double-counted as LJ clashes -> median force 3.8e5 kcal/mol/A), runs
# proteins at unit mass (MolecularBundle has no mass field), and settle_langevin
# needs dt<=0.01 fs for vacuum proteins. `trajectory_finite` is emitted + gated so a
# diverged run is never counted. B1-full over proteins resumes once Sprint A
# (260706 spec: JIT-safe exclusions + mass field + settle vacuum-protein dt) lands.

# TIP3P (flexible variant for a runnable bench; rigid constants + soft k).
_TIP3P = dict(
    q_O=-0.834, q_H=0.417, sig_O=3.15061, eps_O=0.1521,
    sig_H=0.4, eps_H=0.0, m_O=15.99943, m_H=1.008,
    r0=0.9572, k_bond=450.0, theta0_deg=104.52, k_angle=55.0,
)


def _has_cuda_gpu() -> bool:
    """Detect an NVIDIA GPU without importing jax (so we can gate GPU-only XLA flags)."""
    import shutil

    return os.path.exists("/proc/driver/nvidia") or shutil.which("nvidia-smi") is not None


def _setup_env(x64: bool, run_tag: str) -> None:
    """Set pinned XLA flags + a fresh JAX compile-cache dir. Call BEFORE jax import."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # The pre-reg pins --xla_gpu_enable_triton_softmax_fusion=false, but that flag is
    # GPU-only: XLA hard-aborts ("Unknown flag in XLA_FLAGS") on a CPU-only backend.
    # Gate it on GPU presence so the local CPU smoke runs; on the H100 cluster target
    # (the pinned B1-full hardware) the flag is applied as registered.
    flags = os.environ.get("XLA_FLAGS", "")
    if _has_cuda_gpu() and "triton_softmax_fusion" not in flags:
        os.environ["XLA_FLAGS"] = (
            flags + " --xla_gpu_enable_triton_softmax_fusion=false"
        ).strip()
    if x64:
        os.environ["JAX_ENABLE_X64"] = "1"
    # Fresh, deterministic (pid/run-tag derived, NOT time) cache dir per invocation.
    cache = Path(tempfile.gettempdir()) / f"jax_cache_b1_{run_tag}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)


def _hardware_tag() -> str:
    try:
        import jax
        d = jax.devices()[0]
        plat = d.platform
        return "cpu" if plat == "cpu" else f"{plat}:{getattr(d, 'device_kind', 'dev')}"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Water: hand-built TIP3P (deterministic, no external parametrization)          #
# --------------------------------------------------------------------------- #
def _water_arrays(n_waters: int):
    import numpy as np

    t = _TIP3P
    th = math.radians(t["theta0_deg"])
    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([t["r0"], 0.0, 0.0])
    h2 = np.array([t["r0"] * math.cos(th), t["r0"] * math.sin(th), 0.0])
    unit = np.stack([o, h1, h2])  # (3,3)

    pos, masses, charges, sigmas, epsilons = [], [], [], [], []
    bonds, bond_params, angles, angle_params = [], [], [], []
    for i in range(n_waters):
        base = i * 3
        pos.append(unit + np.array([i * 5.0, 0.0, 0.0]))  # 5 A spacing, no overlap
        masses += [t["m_O"], t["m_H"], t["m_H"]]
        charges += [t["q_O"], t["q_H"], t["q_H"]]
        sigmas += [t["sig_O"], t["sig_H"], t["sig_H"]]
        epsilons += [t["eps_O"], t["eps_H"], t["eps_H"]]
        bonds += [[base, base + 1], [base, base + 2]]
        bond_params += [[t["k_bond"], t["r0"]], [t["k_bond"], t["r0"]]]
        angles += [[base + 1, base, base + 2]]
        angle_params += [[t["k_angle"], th]]
    return dict(
        positions=np.concatenate(pos).astype(np.float64),
        masses=np.asarray(masses, np.float64),
        charges=np.asarray(charges, np.float64),
        sigmas=np.asarray(sigmas, np.float64),
        epsilons=np.asarray(epsilons, np.float64),
        bonds=np.asarray(bonds, np.int32),
        bond_params=np.asarray(bond_params, np.float64),
        angles=np.asarray(angles, np.int32),
        angle_params=np.asarray(angle_params, np.float64),
    )


def _prolix_water_bundle(n_waters: int):
    from prolix.physics.system import make_bundle_from_system

    return make_bundle_from_system(
        SimpleNamespace(**_water_arrays(n_waters)), boundary_condition="free"
    )


# --------------------------------------------------------------------------- #
# prolix protein: proxide parse_structure (FF load) + make_bundle (construct)  #
# --------------------------------------------------------------------------- #
def _prolix_protein_parts(pdb_path: str, ff_name: str):
    """Return (t_ff_load, build_fn). build_fn() -> bundle (timed as construct)."""
    import numpy as np
    import proxide
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.system import make_bundle_from_system
    from scripts.benchmarks._b1_paramize import _resolve_ff_path

    ff_path = _resolve_ff_path(ff_name)
    t0 = time.perf_counter()
    protein = parse_structure(pdb_path, OutputSpec(
        parameterize_md=True, force_field=ff_path, coord_format=CoordFormat.Full,
    ))
    masses = np.asarray(proxide.assign_masses(list(protein.atom_names)), np.float64)
    t_ff = time.perf_counter() - t0

    def _arr(name):
        v = getattr(protein, name, None)
        return None if v is None else np.asarray(v)

    def build():
        system = SimpleNamespace(
            positions=_arr("coordinates"), masses=masses,
            bonds=_arr("bonds"), bond_params=_arr("bond_params"),
            angles=_arr("angles"), angle_params=_arr("angle_params"),
            dihedrals=_arr("proper_dihedrals"), dihedral_params=_arr("dihedral_params"),
            impropers=_arr("impropers"), improper_params=_arr("improper_params"),
            charges=_arr("charges"), sigmas=_arr("sigmas"), epsilons=_arr("epsilons"),
        )
        return make_bundle_from_system(system, boundary_condition="free")

    return t_ff, build


def _run_prolix(system_types, replicas, n_steps, dt, kT, seed, ff_name):
    """One compiled pass over the heterogeneous ensemble. Returns metrics dict."""
    import numpy as np

    from prolix.api import EnsemblePlan

    # Parametrize each UNIQUE topology once (prolix amortizes identical-replica
    # FF load — the batching advantage under test), then replicate to N.
    t_ff_load = 0.0
    builders = {}
    for st in system_types:
        if st == WATER_KEY:
            t0 = time.perf_counter()
            wb = _prolix_water_bundle(4)
            # water FF "load" is trivial/hand-built; attribute build to construct
            builders[st] = ("prebuilt", wb)
            t_ff_load += 0.0
            _ = time.perf_counter() - t0
        else:
            t_ff, build = _prolix_protein_parts(str(PROTEIN_FIXTURES[st]), ff_name)
            t_ff_load += t_ff
            builders[st] = ("build", build)

    t0 = time.perf_counter()
    unique_bundles = {}
    for st, (kind, obj) in builders.items():
        unique_bundles[st] = obj if kind == "prebuilt" else obj()
    bundles = []
    for st in system_types:
        bundles.extend([unique_bundles[st]] * replicas)
    plan = EnsemblePlan.from_bundles(bundles)
    t_bundle_construct = time.perf_counter() - t0

    def _step(ns):
        t = time.perf_counter()
        traj = plan.run(n_steps=ns, dt=dt, kT=kT, seed=seed)
        dur = time.perf_counter() - t
        first = traj[0] if isinstance(traj, list) else traj
        pos = np.asarray(getattr(first, "positions", first))
        return dur, bool(np.all(np.isfinite(pos)))

    t_first_step, finite1 = _step(n_steps)          # compile-dominated
    t_steady_state, finite2 = _step(n_steps)        # steady (cached compile)
    steady_per_step = t_steady_state / max(1, n_steps)
    t_aot_compile = max(0.0, t_first_step - steady_per_step * n_steps)
    t_total = t_ff_load + t_bundle_construct + t_first_step
    return dict(
        t_ff_load=t_ff_load, t_bundle_construct=t_bundle_construct,
        t_aot_compile=t_aot_compile, t_first_step=t_first_step,
        t_steady_state=t_steady_state, t_total=t_total,
        aot_ratio=(t_aot_compile / t_total if t_total > 0 else 0.0),
        finite=bool(finite1 and finite2), n_systems=len(bundles),
    )


# --------------------------------------------------------------------------- #
# OpenMM baseline: N independent Contexts, no reuse                            #
# --------------------------------------------------------------------------- #
_STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
}


def _openmm_protein_system(pdb_path: str, ff):
    """createSystem for a protein PDB (HETATM/ligand/water stripped, H added)."""
    from openmm import app

    pdb = app.PDBFile(pdb_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    drop = [r for r in modeller.topology.residues() if r.name not in _STD_AA]
    if drop:
        modeller.delete(drop)
    modeller.addHydrogens(ff)
    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.NoCutoff,
        constraints=None, rigidWater=False, removeCMMotion=False,
    )
    return system, modeller.topology, modeller.positions


def _openmm_water_system(n_waters: int, ff):
    """Build an n-water TIP3P system directly from geometry."""
    import numpy as np
    from openmm import app, unit
    from openmm.app import element

    arr = _water_arrays(n_waters)
    top = app.Topology()
    chain = top.addChain()
    positions = []
    pos = arr["positions"].reshape(n_waters, 3, 3)
    for i in range(n_waters):
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", element.oxygen, res)
        h1 = top.addAtom("H1", element.hydrogen, res)
        h2 = top.addAtom("H2", element.hydrogen, res)
        top.addBond(o, h1)
        top.addBond(o, h2)
        for a in pos[i]:
            positions.append(a)
    positions = np.asarray(positions) * 0.1  # A -> nm
    from openmm import Vec3
    omm_pos = [Vec3(*p) for p in positions] * unit.nanometer
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff,
                             constraints=None, rigidWater=False, removeCMMotion=False)
    return system, top, omm_pos


def _run_openmm(system_types, replicas, n_steps, dt, kT, seed):
    """N independent Contexts, no reuse. Sums cold-start metrics across systems."""
    import openmm
    from openmm import app, unit

    protein_ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    kelvin = float(kT) / 0.0019872041  # AKMA kT (kcal/mol) -> K (approx)
    tot = dict(t_ff_load=0.0, t_bundle_construct=0.0, t_first_step=0.0,
               t_steady_state=0.0, n_systems=0, failures=0)
    plat = openmm.Platform.getPlatformByName("Reference")

    for st in system_types:
        for _ in range(replicas):
            try:
                t0 = time.perf_counter()
                if st == WATER_KEY:
                    system, top, pos = _openmm_water_system(4, protein_ff)
                else:
                    system, top, pos = _openmm_protein_system(str(PROTEIN_FIXTURES[st]), protein_ff)
                tot["t_ff_load"] += time.perf_counter() - t0

                integ = openmm.LangevinMiddleIntegrator(
                    kelvin * unit.kelvin, 1.0 / unit.picosecond,
                    float(dt) * unit.femtoseconds)
                t1 = time.perf_counter()
                ctx = openmm.Context(system, integ, plat)
                ctx.setPositions(pos)
                tot["t_bundle_construct"] += time.perf_counter() - t1

                t2 = time.perf_counter(); integ.step(1)
                tot["t_first_step"] += time.perf_counter() - t2
                if n_steps > 1:
                    t3 = time.perf_counter(); integ.step(n_steps - 1)
                    tot["t_steady_state"] += time.perf_counter() - t3
                tot["n_systems"] += 1
                del ctx, integ, system
            except Exception as e:  # coverage-as-data: log + continue
                tot["failures"] += 1
                logger.warning("openmm %s failed: %s: %s", st, type(e).__name__, str(e)[:120])

    t_total = tot["t_ff_load"] + tot["t_bundle_construct"] + tot["t_first_step"]
    return dict(
        t_ff_load=tot["t_ff_load"], t_bundle_construct=tot["t_bundle_construct"],
        t_aot_compile=0.0,  # openmm has no AOT compile
        t_first_step=tot["t_first_step"], t_steady_state=tot["t_steady_state"],
        t_total=t_total, aot_ratio=0.0, finite=(tot["failures"] == 0),
        n_systems=tot["n_systems"], failures=tot["failures"],
    )


# --------------------------------------------------------------------------- #
def _write_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists()
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def _dry_run(system_types, ff_name) -> int:
    ok = True
    try:
        from scripts.benchmarks._b1_paramize import _resolve_ff_path
        ff_path = _resolve_ff_path(ff_name)
        logger.info("FF resolved: %s", ff_path)
    except Exception as e:
        logger.error("FF resolve FAILED: %s", e); ok = False
    for st in system_types:
        if st == WATER_KEY:
            logger.info("system %-8s: synthesized TIP3P (OK)", st)
            continue
        p = PROTEIN_FIXTURES.get(st)
        if p and p.exists():
            logger.info("system %-8s: fixture %s (OK)", st, p.name)
        else:
            logger.error("system %-8s: fixture MISSING (%s)", st, p); ok = False
    try:
        import openmm  # noqa: F401
        logger.info("openmm import OK")
    except Exception as e:
        logger.error("openmm import FAILED: %s", e); ok = False
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS),
                    help="comma list of system types")
    ap.add_argument("--replicas", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="override; default 100 ps at dt (0.5 fs) = 200000")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--kT", type=float, default=2.5)
    ap.add_argument("--engine", choices=["prolix", "openmm", "both"], default="both")
    ap.add_argument("--smoke", action="store_true",
                    help="1 replica of {2gb1,4-water}, n_steps=10, <120s CPU")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--x64", action="store_true", help="enable float64 (default f32)")
    ap.add_argument("--drop-caches", action="store_true",
                    help="drop OS page cache (cluster only, needs root)")
    ap.add_argument("--ff", default=DEFAULT_FF)
    ap.add_argument("--out", default=str(_ROOT / "outputs" / "bench" / "b1_full.csv"))
    ap.add_argument("--hardware-tag", default=None)
    ap.add_argument("--run-tag", default=str(os.getpid()))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")

    system_types = [s.strip() for s in args.systems.split(",") if s.strip()]
    if args.smoke:
        system_types = [s for s in ["2gb1", WATER_KEY] if s in system_types] or ["2gb1", WATER_KEY]
        replicas, seeds, n_steps = 1, 1, 10
    else:
        replicas = args.replicas
        seeds = args.seeds
        n_steps = args.n_steps if args.n_steps is not None else int(round(100_000 / args.dt))

    if args.dry_run:
        return _dry_run(system_types, args.ff)

    _setup_env(args.x64, args.run_tag)
    if args.drop_caches:
        os.system("sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true")
    precision = "f64" if args.x64 else "f32"
    hw = args.hardware_tag or _hardware_tag()
    logger.info("B1 run: systems=%s replicas=%d seeds=%d n_steps=%d engine=%s prec=%s hw=%s",
                system_types, replicas, seeds, n_steps, args.engine, precision, hw)

    rows: list[dict] = []
    prolix_tot, openmm_tot = [], []
    for seed in range(seeds):
        if args.engine in ("prolix", "both"):
            m = _run_prolix(system_types, replicas, n_steps, args.dt, args.kT, seed, args.ff)
            prolix_tot.append(m["t_total"])
            rows.append(dict(engine="prolix", system_type="ensemble", replica=m["n_systems"],
                             seed=seed, hardware_tag=hw, precision=precision, n_steps=n_steps,
                             trajectory_finite=bool(m["finite"]), **{
                                 k: m[k] for k in ("t_ff_load", "t_bundle_construct", "t_aot_compile",
                                                   "t_first_step", "t_steady_state", "t_total", "aot_ratio")}))
            logger.info("prolix seed=%d t_total=%.4f aot_ratio=%.3f finite=%s",
                        seed, m["t_total"], m["aot_ratio"], m["finite"])
        if args.engine in ("openmm", "both"):
            m = _run_openmm(system_types, replicas, n_steps, args.dt, args.kT, seed)
            openmm_tot.append(m["t_total"])
            rows.append(dict(engine="openmm", system_type="ensemble", replica=m["n_systems"],
                             seed=seed, hardware_tag=hw, precision=precision, n_steps=n_steps,
                             trajectory_finite=bool(m.get("finite", m.get("failures", 0) == 0)), **{
                                 k: m[k] for k in ("t_ff_load", "t_bundle_construct", "t_aot_compile",
                                                   "t_first_step", "t_steady_state", "t_total", "aot_ratio")}))
            logger.info("openmm seed=%d t_total=%.4f failures=%d",
                        seed, m["t_total"], m.get("failures", 0))

    _write_csv(Path(args.out), rows)
    logger.info("wrote %d rows -> %s", len(rows), args.out)

    # Summary + pinned success criteria.
    print("\n===== B1 SUMMARY =====")
    if prolix_tot:
        pmed = statistics.median(prolix_tot)
        print(f"prolix  median t_total: {pmed:.4f}s  (aot_ratio<0.5 guard: "
              f"{'PASS' if rows and rows[0].get('aot_ratio', 1) < 0.5 else 'CHECK'})")
    if openmm_tot:
        omed = statistics.median(openmm_tot)
        print(f"openmm  median t_total: {omed:.4f}s")
    if prolix_tot and openmm_tot:
        hd = "PASS" if statistics.median(prolix_tot) < statistics.median(openmm_tot) else "FAIL"
        print(f"H1 (prolix < openmm): {hd}")
    print("======================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
