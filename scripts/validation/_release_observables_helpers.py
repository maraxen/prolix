"""System builders and trajectory observable kernels for Pallas v3.5 four-tier validation.

Provides:
1. Real PDB + OpenMM ForceField system builders (`TIP3P`, `1UAO`, `1VII`, `1UBQ`,
   `5DFR_GBSA`, `Holo_Lysozyme`, `DHFR_RF`, `DHFR_PME`, `Holo_DHFR`, `ApoA1`) with
   exact 1-to-1 `openmm.System` <-> `MolecularBundle` conversion (including Fortran-ordered
   2D bicubic CMAP tables and active-site-bound Holo-DHFR pocket topology).
2. Pure NumPy/SciPy structural & dynamical observable evaluators:
   - Liquid water `g_OO(r)` (peak position, height, first minimum, coordination number)
     and unwrapped multi-origin Einstein self-diffusion `D_TIP3P` + Yeh-Hummer correction.
   - Solvated protein & holo-complex observables across `B` replicas: PBC-unwrapped Kabsch
     `C_alpha` RMSD, per-residue `C_alpha` RMSF, mass-weighted `R_g`, Ramachandran basin
     occupancy, backbone `N-H...O` H-bond persistence, and pocket-aligned ligand heavy-atom
     RMSD + key active-site contact distances.
"""
# ruff: noqa: N815

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT.parent / "xtrax" / "src")):
    if _p not in sys.path and Path(_p).exists():
        sys.path.insert(0, _p)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from prolix.openmm.converter import from_openmm_system  # noqa: E402
from prolix.physics.cmap import precompute_cmap_coefficients  # noqa: E402
from scripts.benchmarks._b1_paramize import _resolve_ff_path  # noqa: E402
from scripts.benchmarks.bench_e2e_ensemble_buckets import _cast_bundle_precision  # noqa: E402
from scripts.benchmarks.bench_v33_diverse_panel import (  # noqa: E402
    _re_pad_bundle_to_custom_bucket,
)
from scripts.benchmarks.bench_v34_system_ladder import (  # noqa: E402
    _build_openmm_explicit_system,
    build_holo_protein_ligand_bundle,
)
from scripts.benchmarks.openmm_cuda_guard import require_openmm_cuda_context  # noqa: E402

if TYPE_CHECKING:
    from prolix.types.bundles import MolecularBundle

ROOT = Path(__file__).resolve().parents[2]

STD_PROTEIN_RESIDUES: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "CYX",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "HID",
        "HIE",
        "HIP",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ACE",
        "NME",
    }
)
WATER_RESIDUES: frozenset[str] = frozenset({"HOH", "WAT", "TIP3", "SOL"})


@dataclass
class ValidatedSystemSpec:
    """Matched OpenMM System + Prolix MolecularBundle + structural observable indices."""

    name: str
    electrostatics: str  # "pme", "rf", or "gbsa"
    omm_system: Any
    omm_topology: Any | None
    positions_A: np.ndarray  # (N_real, 3) float64
    bundle_f32: MolecularBundle
    bundle_f64: MolecularBundle
    pme_grid: tuple[int, int, int] | None
    pme_alpha: float
    cutoff_A: float
    n_real_atoms: int
    n_waters: int
    n_constraints_total: int
    n_constraints_solute: int
    solute_indices: np.ndarray
    solvent_indices: np.ndarray
    molecule_ids: np.ndarray  # (N_real,) int32
    n_molecules: int
    water_oxygen_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    ca_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    protein_heavy_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    phi_indices: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    psi_indices: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    hbond_donors: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    hbond_donor_res: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    hbond_acceptors: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    hbond_acceptor_res: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    pocket_heavy_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    ligand_heavy_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    key_contacts: list[dict[str, Any]] = field(default_factory=list)
    minimization_provenance: dict[str, Any] | None = None


def _extract_molecule_ids_from_system(omm_system: Any) -> tuple[np.ndarray, int]:
    """Compute connected-component molecule IDs across bonds and constraints (OpenMM MCBarostat convention)."""
    n = omm_system.getNumParticles()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for c_i in range(omm_system.getNumConstraints()):
        p1, p2, _ = omm_system.getConstraintParameters(c_i)
        if 0 <= p1 < n and 0 <= p2 < n:
            union(int(p1), int(p2))

    import openmm as mm

    for f_i in range(omm_system.getNumForces()):
        f = omm_system.getForce(f_i)
        if isinstance(f, mm.HarmonicBondForce):
            for b_i in range(f.getNumBonds()):
                p1, p2, _, _ = f.getBondParameters(b_i)
                if 0 <= p1 < n and 0 <= p2 < n:
                    union(int(p1), int(p2))

    comp_map: dict[int, int] = {}
    mol_ids = np.zeros((n,), dtype=np.int32)
    for i in range(n):
        r = find(i)
        if r not in comp_map:
            comp_map[r] = len(comp_map)
        mol_ids[i] = comp_map[r]
    return mol_ids, len(comp_map)


def _extract_topology_observables(topology: Any) -> dict[str, np.ndarray]:
    """Extract C_alpha, protein heavy, Ramachandran (phi, psi), backbone H-bond, and water O indices."""
    ca_list: list[int] = []
    heavy_list: list[int] = []
    solute_list: list[int] = []
    solvent_list: list[int] = []
    water_o_list: list[int] = []

    residues = list(topology.residues())
    res_backbone: list[dict[str, int] | None] = []
    donors: list[list[int]] = []
    donor_res: list[int] = []
    acceptors: list[int] = []
    acceptor_res: list[int] = []

    for r_idx, res in enumerate(residues):
        rname = res.name.strip().upper()
        atoms = list(res.atoms())
        if rname in WATER_RESIDUES:
            for a in atoms:
                solvent_list.append(a.index)
                if a.name.strip().upper() in ("O", "OW", "OH2"):
                    water_o_list.append(a.index)
            res_backbone.append(None)
            continue

        for a in atoms:
            solute_list.append(a.index)

        if rname not in STD_PROTEIN_RESIDUES:
            res_backbone.append(None)
            continue

        bb_map: dict[str, int] = {}
        for a in atoms:
            aname = a.name.strip().upper()
            elem_sym = a.element.symbol.upper() if a.element is not None else aname[:1]
            if elem_sym != "H":
                heavy_list.append(a.index)
            if aname == "CA":
                ca_list.append(a.index)
                bb_map["CA"] = a.index
            elif aname == "N":
                bb_map["N"] = a.index
            elif aname == "C":
                bb_map["C"] = a.index
            elif aname == "O":
                bb_map["O"] = a.index
            elif aname in ("H", "HN", "H1"):
                bb_map.setdefault("H", a.index)

        res_backbone.append(bb_map)
        if "N" in bb_map and "H" in bb_map and rname != "PRO":
            donors.append([bb_map["N"], bb_map["H"]])
            donor_res.append(r_idx)
        if "O" in bb_map:
            acceptors.append(bb_map["O"])
            acceptor_res.append(r_idx)

    phi_list: list[list[int]] = []
    psi_list: list[list[int]] = []
    for k in range(len(res_backbone)):
        cur = res_backbone[k]
        if cur is None or not all(x in cur for x in ("N", "CA", "C")):
            continue
        prev = res_backbone[k - 1] if k > 0 else None
        nxt = res_backbone[k + 1] if k + 1 < len(res_backbone) else None
        if prev is not None and "C" in prev:
            phi_list.append([prev["C"], cur["N"], cur["CA"], cur["C"]])
        if nxt is not None and "N" in nxt:
            psi_list.append([cur["N"], cur["CA"], cur["C"], nxt["N"]])

    return {
        "ca_indices": np.asarray(ca_list, dtype=np.int32),
        "protein_heavy_indices": np.asarray(heavy_list, dtype=np.int32),
        "solute_indices": np.asarray(solute_list, dtype=np.int32),
        "solvent_indices": np.asarray(solvent_list, dtype=np.int32),
        "water_oxygen_indices": np.asarray(water_o_list, dtype=np.int32),
        "phi_indices": np.asarray(phi_list, dtype=np.int32).reshape(-1, 4)
        if phi_list
        else np.zeros((0, 4), dtype=np.int32),
        "psi_indices": np.asarray(psi_list, dtype=np.int32).reshape(-1, 4)
        if psi_list
        else np.zeros((0, 4), dtype=np.int32),
        "hbond_donors": np.asarray(donors, dtype=np.int32).reshape(-1, 2)
        if donors
        else np.zeros((0, 2), dtype=np.int32),
        "hbond_donor_res": np.asarray(donor_res, dtype=np.int32),
        "hbond_acceptors": np.asarray(acceptors, dtype=np.int32),
        "hbond_acceptor_res": np.asarray(acceptor_res, dtype=np.int32),
    }


def minimize_with_verified_openmm(
    omm_system: Any,
    positions_A: np.ndarray,
    *,
    max_iterations: int = 100,
    allow_cpu_smoke: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize initial coordinates on an anti-fallback-verified OpenMM CUDA context."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx, prov = require_openmm_cuda_context(
        omm_system,
        integrator,
        expected_precision="mixed",
        allow_cpu_fallback=allow_cpu_smoke,
    )
    pos_nm = np.asarray(positions_A, dtype=np.float64) * 0.1
    ctx.setPositions(pos_nm * unit.nanometer)
    if max_iterations > 0:
        mm.LocalEnergyMinimizer.minimize(
            ctx,
            10.0 * unit.kilojoules_per_mole / unit.nanometer,
            int(max_iterations),
        )
    state = ctx.getState(getPositions=True)
    min_pos_A = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
        dtype=np.float64,
    )
    del ctx, integrator
    return min_pos_A, prov


def _convert_openmm_system_to_bundles(
    omm_system: Any,
    positions_A: np.ndarray,
    *,
    bucket_atoms: int,
    bucket_waters: int,
    electrostatics: str,
    pme_alpha: float = 0.34,
    cutoff_A: float = 9.0,
    include_constraints: bool = False,
) -> tuple[MolecularBundle, MolecularBundle, Any]:
    """Convert a live OpenMM System + coordinates into exact float32 and float64 MolecularBundles."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    from prolix.openmm.config import ProlixOpenMMConfig

    conv = from_openmm_system(
        omm_system, config=ProlixOpenMMConfig(use_size_buckets=False)
    )
    raw = conv.bundle
    n_real = int(raw.n_atoms)
    n_pad0 = int(raw.positions.shape[0])

    pos_pad0 = np.zeros((n_pad0, 3), dtype=np.float32)
    pos_pad0[:n_real] = np.asarray(positions_A[:n_real], dtype=np.float32)

    # Ensure C-ordered CMAP grids matching `prolix/physics/cmap.py:259` (`compute_bicubic_coefficients`
    # internally transposes `energy = energy.T` from C-order `[j_psi, i_phi]` to `[i_phi, j_psi]`)
    cmap_coeffs = raw.cmap_coeffs
    cmap_grids = raw.cmap_energy_grids
    cmap_forces = [
        f for f in omm_system.getForces() if isinstance(f, mm.CMAPTorsionForce)
    ]
    if cmap_forces and int(raw.n_cmap) > 0:
        cf = cmap_forces[0]
        maps_c: list[np.ndarray] = []
        for m_i in range(cf.getNumMaps()):
            sz, ef = cf.getMapParameters(m_i)
            arr = (
                np.asarray(
                    [
                        e.value_in_unit(unit.kilojoules_per_mole)
                        if hasattr(e, "value_in_unit")
                        else float(e)
                        for e in ef
                    ],
                    dtype=np.float32,
                ).reshape((sz, sz), order="C")
                / 4.184
            )
            maps_c.append(arr)
        grids_stack = jnp.asarray(np.stack(maps_c, axis=0), dtype=jnp.float32)
        raw_coeffs = precompute_cmap_coefficients(grids_stack).astype(jnp.float32)
        mb = int(cmap_coeffs.shape[0]) if cmap_coeffs is not None else int(raw_coeffs.shape[0])
        if mb > int(raw_coeffs.shape[0]):
            cmap_coeffs = jnp.pad(
                raw_coeffs,
                ((0, mb - int(raw_coeffs.shape[0])), (0, 0), (0, 0), (0, 0)),
            )
        else:
            cmap_coeffs = raw_coeffs
        n_c = int(raw.n_cmap)
        raw_c_idx = raw.cmap_indices if raw.cmap_indices is not None else jnp.zeros((n_c,), dtype=jnp.int32)
        c_idx = np.clip(np.asarray(raw_c_idx[:n_c], dtype=np.int32), 0, len(maps_c) - 1)
        per_torsion = np.zeros_like(np.asarray(cmap_grids, dtype=np.float32))
        per_torsion[:n_c] = np.stack(maps_c, axis=0)[c_idx]
        cmap_grids = jnp.asarray(per_torsion, dtype=jnp.float32)

    alpha_val = 0.0 if electrostatics.lower() in ("rf", "reaction_field", "gbsa") else float(pme_alpha)
    raw_fixed = dataclasses.replace(
        raw,
        positions=jnp.asarray(pos_pad0, dtype=jnp.float32),
        dihedral_params=raw.dihedral_params[..., :3],
        improper_params=raw.improper_params[..., :3],
        cmap_coeffs=cmap_coeffs,
        cmap_energy_grids=cmap_grids,
        pme_alpha=jnp.array(alpha_val, dtype=jnp.float32),
        cutoff_distance=jnp.array(float(cutoff_A), dtype=jnp.float32),
    )

    bun_f32 = _cast_bundle_precision(
        _re_pad_bundle_to_custom_bucket(
            raw_fixed,
            bucket_atoms=bucket_atoms,
            bucket_waters=max(bucket_waters, 1),
            electrostatic_method=electrostatics,
            include_constraints=include_constraints,
        ),
        jnp.float32,
    )
    # `_re_pad_bundle_to_custom_bucket` wraps `positions % box` atom-by-atom, splitting protein bonds
    # across periodic faces for OpenMM's non-periodic `HarmonicBondForce`. Reconstruct whole covalent
    # molecules with molecular COM inside `[0, box)` so OpenMM and Prolix share bit-exact whole geometry.
    if bool(bun_f32.shape_spec.has_pbc):
        box_f64 = np.diag(np.asarray(bun_f32.box, dtype=np.float64))
        if np.all(box_f64 > 0.0):
            adj: list[list[int]] = [[] for _ in range(n_real)]
            for c_i in range(omm_system.getNumConstraints()):
                p1, p2, _ = omm_system.getConstraintParameters(c_i)
                if 0 <= p1 < n_real and 0 <= p2 < n_real:
                    adj[int(p1)].append(int(p2))
                    adj[int(p2)].append(int(p1))
            for f_i in range(omm_system.getNumForces()):
                f_obj = omm_system.getForce(f_i)
                if isinstance(f_obj, mm.HarmonicBondForce):
                    for b_i in range(f_obj.getNumBonds()):
                        p1, p2, _, _ = f_obj.getBondParameters(b_i)
                        if 0 <= p1 < n_real and 0 <= p2 < n_real:
                            adj[int(p1)].append(int(p2))
                            adj[int(p2)].append(int(p1))
            pos_u = np.asarray(positions_A[:n_real], dtype=np.float64).copy()
            masses_r = np.asarray(bun_f32.masses[:n_real], dtype=np.float64)
            visited = np.zeros((n_real,), dtype=bool)
            for root in range(n_real):
                if visited[root]:
                    continue
                visited[root] = True
                comp = [root]
                q_head = 0
                while q_head < len(comp):
                    u_a = comp[q_head]
                    q_head += 1
                    for v_a in adj[u_a]:
                        if not visited[v_a]:
                            visited[v_a] = True
                            dr = pos_u[v_a] - pos_u[u_a]
                            pos_u[v_a] = pos_u[u_a] + (dr - box_f64 * np.round(dr / box_f64))
                            comp.append(v_a)
                idx_c = np.asarray(comp, dtype=np.int32)
                w_c = masses_r[idx_c]
                com_c = np.sum(pos_u[idx_c] * w_c[:, None], axis=0) / max(float(np.sum(w_c)), 1e-12)
                com_wrap = np.mod(com_c, box_f64)
                pos_u[idx_c] += com_wrap - com_c
            pos_pad_f32 = np.asarray(bun_f32.positions, dtype=np.float32).copy()
            pos_pad_f32[:n_real] = pos_u.astype(np.float32)
            bun_f32 = dataclasses.replace(bun_f32, positions=jnp.asarray(pos_pad_f32, dtype=jnp.float32))

    bun_f64 = _cast_bundle_precision(bun_f32, jnp.float64)
    return bun_f32, bun_f64, conv


def build_tip3p_system(*, allow_cpu_smoke: bool = False, minimize_iters: int = 80) -> ValidatedSystemSpec:
    """Build 895-water (2,685-atom) liquid TIP3P box in 30.0 A cube."""
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    pdb = app.PDBFile(str(ROOT / "data" / "water_boxes" / "tip3p.pdb"))
    ff = app.ForceField("amber14/tip3p.xml")
    omm_sys = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    omm_sys.setDefaultPeriodicBoxVectors(
        mm.Vec3(3.0, 0.0, 0.0),
        mm.Vec3(0.0, 3.0, 0.0),
        mm.Vec3(0.0, 0.0, 3.0),
    )
    for f in omm_sys.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setPMEParameters(3.4, 32, 32, 32)
            f.setUseDispersionCorrection(True)

    pos_init_A = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    min_pos_A, prov = minimize_with_verified_openmm(
        omm_sys, pos_init_A, max_iterations=minimize_iters, allow_cpu_smoke=allow_cpu_smoke
    )
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys,
        min_pos_A,
        bucket_atoms=3072,
        bucket_waters=1024,
        electrostatics="pme",
        pme_alpha=0.34,
        cutoff_A=9.0,
    )
    obs_meta = _extract_topology_observables(pdb.topology)
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys)
    n_w = int(bun_f32.n_waters)
    return ValidatedSystemSpec(
        name="TIP3P_water_box",
        electrostatics="pme",
        omm_system=omm_sys,
        omm_topology=pdb.topology,
        positions_A=np.asarray(bun_f64.positions[: int(bun_f32.n_atoms)], dtype=np.float64),
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=(32, 32, 32),
        pme_alpha=0.34,
        cutoff_A=9.0,
        n_real_atoms=int(bun_f32.n_atoms),
        n_waters=n_w,
        n_constraints_total=omm_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=obs_meta["solute_indices"],
        solvent_indices=obs_meta["solvent_indices"],
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        water_oxygen_indices=obs_meta["water_oxygen_indices"],
        minimization_provenance=prov,
    )


def build_1uao_system(
    *,
    electrostatics: str = "pme",
    allow_cpu_smoke: bool = False,
    minimize_iters: int = 100,
) -> ValidatedSystemSpec:
    """Build solvated Chignolin (1UAO, 2,333 atoms: 10 residues + 2 Na+ + 731 TIP3P in 29.334 A cube)."""
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    pdb = app.PDBFile(str(ROOT / "data" / "pdb" / "1UAO_solvated_tip3p.pdb"))
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    is_rf = electrostatics.lower() in ("rf", "reaction_field")
    omm_sys = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffPeriodic if is_rf else app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=True,
    )
    for f in omm_sys.getForces():
        if isinstance(f, mm.NonbondedForce):
            if not is_rf:
                f.setPMEParameters(3.4, 32, 32, 32)
            f.setUseDispersionCorrection(True)

    pos_init_A = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    min_pos_A, prov = minimize_with_verified_openmm(
        omm_sys, pos_init_A, max_iterations=minimize_iters, allow_cpu_smoke=allow_cpu_smoke
    )
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys,
        min_pos_A,
        bucket_atoms=2560,
        bucket_waters=1024,
        electrostatics="rf" if is_rf else "pme",
        pme_alpha=0.0 if is_rf else 0.34,
        cutoff_A=9.0,
    )
    obs_meta = _extract_topology_observables(pdb.topology)
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys)
    return ValidatedSystemSpec(
        name="1UAO" if not is_rf else "1UAO_RF",
        electrostatics="rf" if is_rf else "pme",
        omm_system=omm_sys,
        omm_topology=pdb.topology,
        positions_A=np.asarray(bun_f64.positions[: int(bun_f32.n_atoms)], dtype=np.float64),
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=None if is_rf else (32, 32, 32),
        pme_alpha=0.0 if is_rf else 0.34,
        cutoff_A=9.0,
        n_real_atoms=int(bun_f32.n_atoms),
        n_waters=int(bun_f32.n_waters),
        n_constraints_total=omm_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=obs_meta["solute_indices"],
        solvent_indices=obs_meta["solvent_indices"],
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        water_oxygen_indices=obs_meta["water_oxygen_indices"],
        ca_indices=obs_meta["ca_indices"],
        protein_heavy_indices=obs_meta["protein_heavy_indices"],
        phi_indices=obs_meta["phi_indices"],
        psi_indices=obs_meta["psi_indices"],
        hbond_donors=obs_meta["hbond_donors"],
        hbond_donor_res=obs_meta["hbond_donor_res"],
        hbond_acceptors=obs_meta["hbond_acceptors"],
        hbond_acceptor_res=obs_meta["hbond_acceptor_res"],
        minimization_provenance=prov,
    )


def build_1vii_system(*, allow_cpu_smoke: bool = False, minimize_iters: int = 120) -> ValidatedSystemSpec:
    """Build solvated Villin headpiece (1VII, 36 residues in 38.0 A TIP3P cube)."""
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    pdb = app.PDBFile(str(ROOT / "data" / "pdb" / "1VII.pdb"))
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.deleteWater()
    mod.addSolvent(
        ff,
        boxSize=mm.Vec3(3.8, 3.8, 3.8) * unit.nanometer,
        model="tip3p",
        neutralize=True,
    )
    omm_sys = ff.createSystem(
        mod.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=True,
    )
    for f in omm_sys.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setPMEParameters(3.4, 40, 40, 40)
            f.setUseDispersionCorrection(True)

    pos_init_A = np.asarray(mod.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    min_pos_A, prov = minimize_with_verified_openmm(
        omm_sys, pos_init_A, max_iterations=minimize_iters, allow_cpu_smoke=allow_cpu_smoke
    )
    n_real = omm_sys.getNumParticles()
    b_atoms = max(8192, ((n_real + 1023) // 1024) * 1024)
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys,
        min_pos_A,
        bucket_atoms=b_atoms,
        bucket_waters=2560,
        electrostatics="pme",
        pme_alpha=0.34,
        cutoff_A=9.0,
    )
    obs_meta = _extract_topology_observables(mod.topology)
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys)
    return ValidatedSystemSpec(
        name="1VII",
        electrostatics="pme",
        omm_system=omm_sys,
        omm_topology=mod.topology,
        positions_A=np.asarray(bun_f64.positions[:n_real], dtype=np.float64),
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=(40, 40, 40),
        pme_alpha=0.34,
        cutoff_A=9.0,
        n_real_atoms=n_real,
        n_waters=int(bun_f32.n_waters),
        n_constraints_total=omm_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=obs_meta["solute_indices"],
        solvent_indices=obs_meta["solvent_indices"],
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        water_oxygen_indices=obs_meta["water_oxygen_indices"],
        ca_indices=obs_meta["ca_indices"],
        protein_heavy_indices=obs_meta["protein_heavy_indices"],
        phi_indices=obs_meta["phi_indices"],
        psi_indices=obs_meta["psi_indices"],
        hbond_donors=obs_meta["hbond_donors"],
        hbond_donor_res=obs_meta["hbond_donor_res"],
        hbond_acceptors=obs_meta["hbond_acceptors"],
        hbond_acceptor_res=obs_meta["hbond_acceptor_res"],
        minimization_provenance=prov,
    )


def build_1ubq_system(*, allow_cpu_smoke: bool = False, minimize_iters: int = 150) -> ValidatedSystemSpec:
    """Build solvated Ubiquitin (1UBQ, 76 residues, ~7,099 atoms in 43.0 A TIP3P cube)."""
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    pdb = app.PDBFile(str(ROOT / "data" / "pdb" / "1UBQ.pdb"))
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.deleteWater()
    mod.addHydrogens(ff)
    mod.addSolvent(
        ff,
        boxSize=mm.Vec3(4.3, 4.3, 4.3) * unit.nanometer,
        model="tip3p",
        neutralize=True,
    )
    omm_sys = ff.createSystem(
        mod.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=True,
    )
    for f in omm_sys.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setPMEParameters(3.4, 48, 48, 48)
            f.setUseDispersionCorrection(True)

    pos_init_A = np.asarray(mod.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    min_pos_A, prov = minimize_with_verified_openmm(
        omm_sys, pos_init_A, max_iterations=minimize_iters, allow_cpu_smoke=allow_cpu_smoke
    )
    n_real = omm_sys.getNumParticles()
    b_atoms = max(8192, ((n_real + 1023) // 1024) * 1024)
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys,
        min_pos_A,
        bucket_atoms=b_atoms,
        bucket_waters=2560,
        electrostatics="pme",
        pme_alpha=0.34,
        cutoff_A=9.0,
    )
    obs_meta = _extract_topology_observables(mod.topology)
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys)
    return ValidatedSystemSpec(
        name="1UBQ",
        electrostatics="pme",
        omm_system=omm_sys,
        omm_topology=mod.topology,
        positions_A=np.asarray(bun_f64.positions[:n_real], dtype=np.float64),
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=(48, 48, 48),
        pme_alpha=0.34,
        cutoff_A=9.0,
        n_real_atoms=n_real,
        n_waters=int(bun_f32.n_waters),
        n_constraints_total=omm_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=obs_meta["solute_indices"],
        solvent_indices=obs_meta["solvent_indices"],
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        water_oxygen_indices=obs_meta["water_oxygen_indices"],
        ca_indices=obs_meta["ca_indices"],
        protein_heavy_indices=obs_meta["protein_heavy_indices"],
        phi_indices=obs_meta["phi_indices"],
        psi_indices=obs_meta["psi_indices"],
        hbond_donors=obs_meta["hbond_donors"],
        hbond_donor_res=obs_meta["hbond_donor_res"],
        hbond_acceptors=obs_meta["hbond_acceptors"],
        hbond_acceptor_res=obs_meta["hbond_acceptor_res"],
        minimization_provenance=prov,
    )


def build_5dfr_gbsa_system(*, allow_cpu_smoke: bool = False) -> ValidatedSystemSpec:
    """Build implicit-solvent DHFR (5DFR, 2,489 atoms, ff19SB 2D CMAPs + OBC2 GBSA + ACE SASA)."""
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    ff19 = app.ForceField(_resolve_ff_path("protein.ff19SB.xml"), "implicit/obc2.xml")
    pdb = app.PDBFile(str(ROOT / "submodules" / "openmm" / "examples" / "benchmarks" / "5dfr_minimized.pdb"))
    omm_sys = ff19.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=True,
    )
    pos_A = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    pos_A, prov = minimize_with_verified_openmm(
        omm_sys, pos_A, max_iterations=0, allow_cpu_smoke=allow_cpu_smoke
    )
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys,
        pos_A,
        bucket_atoms=2560,
        bucket_waters=8,
        electrostatics="gbsa",
        pme_alpha=0.0,
        cutoff_A=1000.0,
    )
    obs_meta = _extract_topology_observables(pdb.topology)
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys)
    return ValidatedSystemSpec(
        name="5DFR_GBSA",
        electrostatics="gbsa",
        omm_system=omm_sys,
        omm_topology=pdb.topology,
        positions_A=pos_A,
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=None,
        pme_alpha=0.0,
        cutoff_A=1000.0,
        n_real_atoms=int(bun_f32.n_atoms),
        n_waters=0,
        n_constraints_total=omm_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=obs_meta["solute_indices"],
        solvent_indices=obs_meta["solvent_indices"],
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        ca_indices=obs_meta["ca_indices"],
        protein_heavy_indices=obs_meta["protein_heavy_indices"],
        phi_indices=obs_meta["phi_indices"],
        psi_indices=obs_meta["psi_indices"],
        hbond_donors=obs_meta["hbond_donors"],
        hbond_donor_res=obs_meta["hbond_donor_res"],
        hbond_acceptors=obs_meta["hbond_acceptors"],
        hbond_acceptor_res=obs_meta["hbond_acceptor_res"],
        minimization_provenance=prov,
    )


def build_dhfr_system(
    *,
    electrostatics: str = "pme",
    allow_cpu_smoke: bool = False,
    minimize_iters: int = 80,
) -> ValidatedSystemSpec:
    """Build solvated DHFR (23,558 atoms in 62.23 A cube, PME 64^3 or CutoffPeriodic RF)."""
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    pdb = app.PDBFile(
        str(ROOT / "submodules" / "openmm" / "examples" / "benchmarks" / "5dfr_solv-cube_equil.pdb")
    )
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    is_rf = electrostatics.lower() in ("rf", "reaction_field")
    omm_sys = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffPeriodic if is_rf else app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=True,
    )
    omm_sys.setDefaultPeriodicBoxVectors(
        mm.Vec3(6.223, 0.0, 0.0),
        mm.Vec3(0.0, 6.223, 0.0),
        mm.Vec3(0.0, 0.0, 6.223),
    )
    for f in omm_sys.getForces():
        if isinstance(f, mm.NonbondedForce):
            if not is_rf:
                f.setPMEParameters(3.4, 64, 64, 64)
            f.setUseDispersionCorrection(True)

    pos_init_A = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    min_pos_A, prov = minimize_with_verified_openmm(
        omm_sys, pos_init_A, max_iterations=minimize_iters, allow_cpu_smoke=allow_cpu_smoke
    )
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys,
        min_pos_A,
        bucket_atoms=25000,
        bucket_waters=8192,
        electrostatics="rf" if is_rf else "pme",
        pme_alpha=0.0 if is_rf else 0.34,
        cutoff_A=9.0,
    )
    obs_meta = _extract_topology_observables(pdb.topology)
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys)
    n_real = int(bun_f32.n_atoms)
    return ValidatedSystemSpec(
        name="DHFR_RF" if is_rf else "DHFR_PME",
        electrostatics="rf" if is_rf else "pme",
        omm_system=omm_sys,
        omm_topology=pdb.topology,
        positions_A=np.asarray(bun_f64.positions[:n_real], dtype=np.float64),
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=None if is_rf else (64, 64, 64),
        pme_alpha=0.0 if is_rf else 0.34,
        cutoff_A=9.0,
        n_real_atoms=n_real,
        n_waters=int(bun_f32.n_waters),
        n_constraints_total=omm_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=obs_meta["solute_indices"],
        solvent_indices=obs_meta["solvent_indices"],
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        water_oxygen_indices=obs_meta["water_oxygen_indices"],
        ca_indices=obs_meta["ca_indices"],
        protein_heavy_indices=obs_meta["protein_heavy_indices"],
        phi_indices=obs_meta["phi_indices"],
        psi_indices=obs_meta["psi_indices"],
        hbond_donors=obs_meta["hbond_donors"],
        hbond_donor_res=obs_meta["hbond_donor_res"],
        hbond_acceptors=obs_meta["hbond_acceptors"],
        hbond_acceptor_res=obs_meta["hbond_acceptor_res"],
        minimization_provenance=prov,
    )


def build_holo_lysozyme_system(
    dhfr_spec: ValidatedSystemSpec,
    *,
    allow_cpu_smoke: bool = False,
) -> ValidatedSystemSpec:
    """Build 15,002-atom Holo-Lysozyme benchmark system (protein + bound ligand + TIP3P, 54^3 PME)."""
    import openmm as mm

    bun = build_holo_protein_ligand_bundle(
        dhfr_spec.bundle_f32,
        n_prot_atoms=1960,
        n_ligand_atoms=46,
        n_real_target=15002,
        bucket_atoms=16384,
        bucket_waters=4608,
        ligand_seed=1,
    )
    omm_sys, _, _ = _build_openmm_explicit_system(
        bun, electrostatic_method="pme", pme_grid_dims=(54, 54, 54)
    )
    for f in omm_sys.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(True)

    n_real = int(bun.n_atoms)
    pos_init_A = np.asarray(bun.positions[:n_real], dtype=np.float64)
    min_pos_A, prov = minimize_with_verified_openmm(
        omm_sys, pos_init_A, max_iterations=60, allow_cpu_smoke=allow_cpu_smoke
    )
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys,
        min_pos_A,
        bucket_atoms=16384,
        bucket_waters=4608,
        electrostatics="pme",
        pme_alpha=0.34,
        cutoff_A=9.0,
    )
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys)
    n_sol = 1960 + 48
    return ValidatedSystemSpec(
        name="Holo_Lysozyme",
        electrostatics="pme",
        omm_system=omm_sys,
        omm_topology=None,
        positions_A=np.asarray(bun_f64.positions[:n_real], dtype=np.float64),
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=(54, 54, 54),
        pme_alpha=0.34,
        cutoff_A=9.0,
        n_real_atoms=n_real,
        n_waters=int(bun_f32.n_waters),
        n_constraints_total=omm_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=np.arange(n_sol, dtype=np.int32),
        solvent_indices=np.arange(n_sol, n_real, dtype=np.int32),
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        minimization_provenance=prov,
    )


def build_holo_dhfr_system(
    *,
    allow_cpu_smoke: bool = False,
    minimize_iters: int = 150,
) -> ValidatedSystemSpec:
    """Build active-site-bound Holo-DHFR complex (DHFR + 45-atom cleft-bound folate mimic + TIP3P).

    Selects 15 solvent triplets nestled inside the E. coli DHFR active-site folate/MTX
    binding cleft (Asp27, Leu28, Phe31, Ile50, Leu54, Arg57, Ile94), reorders them
    immediately after the 2,489 protein atoms (`[2489:2534]`) so the solute prefix is
    contiguous for Pallas pure-water tile inference, links their heavy centers into a
    single covalently connected GAFF-like drug scaffold with bidentate electrostatic
    complementarity to Asp27 (-) and Arg57 (+), and minimizes the bound complex.
    """
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    pdb = app.PDBFile(
        str(ROOT / "submodules" / "openmm" / "examples" / "benchmarks" / "5dfr_solv-cube_equil.pdb")
    )
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    base_sys = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=True,
    )
    box_L = 62.23
    pos_raw_A = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)

    residues = list(pdb.topology.residues())
    pocket_res_1based = {27, 28, 31, 50, 54, 57, 94}
    pocket_heavy_list: list[int] = []
    asp27_oxygens: list[int] = []
    arg57_nitrogens: list[int] = []
    prot_atom_labels: dict[int, str] = {}
    water_triplets: list[list[int]] = []

    for r_idx1, res in enumerate(residues, start=1):
        atoms = list(res.atoms())
        if res.name.strip().upper() in WATER_RESIDUES:
            water_triplets.append([a.index for a in atoms])
        else:
            for a in atoms:
                aname = a.name.strip().upper()
                prot_atom_labels[a.index] = f"{res.name}{r_idx1}:{aname}"
                if r_idx1 in pocket_res_1based and not aname.startswith("H"):
                    pocket_heavy_list.append(a.index)
                if r_idx1 == 27 and aname in ("OD1", "OD2"):
                    asp27_oxygens.append(a.index)
                if r_idx1 == 57 and aname in ("NH1", "NH2", "CZ"):
                    arg57_nitrogens.append(a.index)

    n_prot = 2489
    pocket_heavy_arr = np.asarray(pocket_heavy_list, dtype=np.int32)
    pocket_center = pos_raw_A[pocket_heavy_arr].mean(axis=0)

    w_O = np.asarray([pos_raw_A[t[0]] for t in water_triplets], dtype=np.float64)
    dr_p = w_O - pocket_center[None, :]
    dr_p -= box_L * np.round(dr_p / box_L)
    d_p = np.linalg.norm(dr_p, axis=-1)

    n_lig_triplets = 15  # 45 ligand atoms (15 heavy C/N/O + 30 substituents/H)
    chosen_w_slots = sorted(np.argsort(d_p)[:n_lig_triplets].tolist())
    chosen_set = set(chosen_w_slots)

    # Permutation: [0..2488 protein] + [15 pocket ligand triplets = 45 atoms] + [remaining 7008 waters]
    old_to_new: dict[int, int] = {i: i for i in range(n_prot)}
    new_to_old: list[int] = list(range(n_prot))
    for w_slot in chosen_w_slots:
        for old_a in water_triplets[w_slot]:
            old_to_new[old_a] = len(new_to_old)
            new_to_old.append(old_a)
    kept_water_triplets_new: list[tuple[int, int, int]] = []
    for w_slot, trip in enumerate(water_triplets):
        if w_slot in chosen_set:
            continue
        new_trip = []
        for old_a in trip:
            old_to_new[old_a] = len(new_to_old)
            new_trip.append(len(new_to_old))
            new_to_old.append(old_a)
        kept_water_triplets_new.append((new_trip[0], new_trip[1], new_trip[2]))

    n_total = len(new_to_old)
    pos_perm_A = pos_raw_A[np.asarray(new_to_old, dtype=np.int32)].copy()
    # Unwrap ligand atoms into the same periodic image as pocket_center
    lig_start = n_prot
    lig_end = n_prot + 3 * n_lig_triplets
    for a_i in range(lig_start, lig_end):
        dr = pos_perm_A[a_i] - pocket_center
        pos_perm_A[a_i] = pocket_center + (dr - box_L * np.round(dr / box_L))

    # Build fresh OpenMM System with exact protein force-field + covalently connected pocket ligand + rigid waters
    holo_sys = mm.System()
    holo_sys.setDefaultPeriodicBoxVectors(
        mm.Vec3(6.223, 0.0, 0.0),
        mm.Vec3(0.0, 6.223, 0.0),
        mm.Vec3(0.0, 0.0, 6.223),
    )

    rng = np.random.default_rng(20261111)
    lig_heavy_indices: list[int] = []
    for new_i, old_i in enumerate(new_to_old):
        if new_i < n_prot or new_i >= lig_end:
            m = base_sys.getParticleMass(old_i).value_in_unit(unit.dalton)
            holo_sys.addParticle(m)
        else:
            offset = (new_i - lig_start) % 3
            if offset == 0:
                lig_heavy_indices.append(new_i)
                holo_sys.addParticle(float(rng.choice([12.011, 14.007, 15.999])))
            else:
                holo_sys.addParticle(1.008)

    # Rigid water constraints on remaining solvent waters ONLY
    r_oh_nm = 0.09572
    r_hh_nm = 0.15139
    for ow, h1, h2 in kept_water_triplets_new:
        holo_sys.addConstraint(ow, h1, r_oh_nm)
        holo_sys.addConstraint(ow, h2, r_oh_nm)
        holo_sys.addConstraint(h1, h2, r_hh_nm)

    # Copy protein HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce + add ligand scaffold
    bond_f = mm.HarmonicBondForce()
    angle_f = mm.HarmonicAngleForce()
    tors_f = mm.PeriodicTorsionForce()
    bonded_pairs_for_exc: list[tuple[int, int]] = []

    for f in base_sys.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                p1, p2, d0, kb = f.getBondParameters(k)
                if p1 < n_prot and p2 < n_prot:
                    bond_f.addBond(p1, p2, d0, kb)
                    bonded_pairs_for_exc.append((p1, p2))
        elif isinstance(f, mm.HarmonicAngleForce):
            for k in range(f.getNumAngles()):
                p1, p2, p3, th0, ka = f.getAngleParameters(k)
                if p1 < n_prot and p2 < n_prot and p3 < n_prot:
                    angle_f.addAngle(p1, p2, p3, th0, ka)
        elif isinstance(f, mm.PeriodicTorsionForce):
            for k in range(f.getNumTorsions()):
                p1, p2, p3, p4, per, ph, kt = f.getTorsionParameters(k)
                if p1 < n_prot and p2 < n_prot and p3 < n_prot and p4 < n_prot:
                    tors_f.addTorsion(p1, p2, p3, p4, per, ph, kt)

    # Intra-triplet C-H / N-H bonds & H-X-H angles on ligand
    for h_center in lig_heavy_indices:
        h1, h2 = h_center + 1, h_center + 2
        d1 = float(np.linalg.norm(pos_perm_A[h_center] - pos_perm_A[h1])) * 0.1
        d2 = float(np.linalg.norm(pos_perm_A[h_center] - pos_perm_A[h2])) * 0.1
        bond_f.addBond(h_center, h1, d1, 310.0 * 418.4)
        bond_f.addBond(h_center, h2, d2, 310.0 * 418.4)
        bonded_pairs_for_exc.extend([(h_center, h1), (h_center, h2)])
        angle_f.addAngle(h1, h_center, h2, 1.911, 50.0 * 4.184)

    # Connect all 15 ligand heavy atoms into a single covalent scaffold via Minimum Spanning Tree + 3 ring/frame struts
    n_lh = len(lig_heavy_indices)
    lh_pos = pos_perm_A[np.asarray(lig_heavy_indices, dtype=np.int32)]
    visited = {0}
    lig_adj: dict[int, list[int]] = {u: [] for u in range(n_lh)}
    lig_edges: set[tuple[int, int]] = set()
    while len(visited) < n_lh:
        best_pair = None
        best_dist = 1e9
        for u in visited:
            if len(lig_adj[u]) >= 3:
                continue
            for v in range(n_lh):
                if v in visited:
                    continue
                dist_uv = float(np.linalg.norm(lh_pos[u] - lh_pos[v]))
                if dist_uv < best_dist:
                    best_dist = dist_uv
                    best_pair = (u, v)
        if best_pair is None:
            raise RuntimeError("Failed to connect ligand heavy-atom spanning tree")
        u, v = best_pair
        visited.add(v)
        edge = (min(u, v), max(u, v))
        lig_edges.add(edge)
        lig_adj[u].append(v)
        lig_adj[v].append(u)
        au, av = lig_heavy_indices[u], lig_heavy_indices[v]
        bond_f.addBond(au, av, best_dist * 0.1, 260.0 * 418.4)
        bonded_pairs_for_exc.append((au, av))

    # Add up to 2 short non-tree ring-closure bonds (< 4.2 A) with max heavy-atom degree <= 3 (chemical sp2/sp3 valence)
    cand_closures: list[tuple[float, int, int]] = []
    for u in range(n_lh):
        for v in range(u + 1, n_lh):
            if (u, v) not in lig_edges:
                d_uv = float(np.linalg.norm(lh_pos[u] - lh_pos[v]))
                if d_uv < 4.2:
                    cand_closures.append((d_uv, u, v))
    cand_closures.sort()
    n_closed = 0
    for d_uv, u, v in cand_closures:
        if n_closed >= 2:
            break
        if len(lig_adj[u]) >= 3 or len(lig_adj[v]) >= 3:
            continue
        lig_edges.add((u, v))
        lig_adj[u].append(v)
        lig_adj[v].append(u)
        au, av = lig_heavy_indices[u], lig_heavy_indices[v]
        bond_f.addBond(au, av, d_uv * 0.1, 200.0 * 418.4)
        bonded_pairs_for_exc.append((au, av))
        n_closed += 1

    # Add heavy-atom scaffold angles & proper periodic torsions at pocket equilibrium angles
    for center_u in range(n_lh):
        nbrs = sorted(lig_adj[center_u])
        ac = lig_heavy_indices[center_u]
        for idx_a in range(len(nbrs)):
            for idx_b in range(idx_a + 1, len(nbrs)):
                aa = lig_heavy_indices[nbrs[idx_a]]
                ab = lig_heavy_indices[nbrs[idx_b]]
                v1 = pos_perm_A[aa] - pos_perm_A[ac]
                v2 = pos_perm_A[ab] - pos_perm_A[ac]
                cos_th = np.clip(
                    np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8),
                    -0.999,
                    0.999,
                )
                angle_f.addAngle(aa, ac, ab, float(np.arccos(cos_th)), 55.0 * 4.184)
        # Substituent H-C-C angle to keep sp3/sp2 orientation well-defined
        if nbrs:
            an0 = lig_heavy_indices[nbrs[0]]
            angle_f.addAngle(ac + 1, ac, an0, 1.911, 35.0 * 4.184)

    for u, v in sorted(lig_edges):
        au, av = lig_heavy_indices[u], lig_heavy_indices[v]
        u_nbrs = [x for x in lig_adj[u] if x != v]
        v_nbrs = [x for x in lig_adj[v] if x != u]
        a1 = lig_heavy_indices[u_nbrs[0]] if u_nbrs else (au + 1)
        a4 = lig_heavy_indices[v_nbrs[0]] if v_nbrs else (av + 1)
        if len({a1, au, av, a4}) == 4:
            tors_f.addTorsion(a1, au, av, a4, 2, np.pi, 1.2 * 4.184)

    holo_sys.addForce(bond_f)
    holo_sys.addForce(angle_f)
    holo_sys.addForce(tors_f)

    # NonbondedForce: copy protein & solvent, parameterize ligand with Asp27/Arg57 salt-bridge complementarity
    base_nb = next(f for f in base_sys.getForces() if isinstance(f, mm.NonbondedForce))
    nb_f = mm.NonbondedForce()
    nb_f.setNonbondedMethod(mm.NonbondedForce.PME)
    nb_f.setCutoffDistance(0.9 * unit.nanometer)
    nb_f.setPMEParameters(3.4, 64, 64, 64)
    nb_f.setUseDispersionCorrection(True)

    # Identify ligand heavy atom closest to Asp27 (-) and closest to Arg57 (+)
    asp_pos = pos_perm_A[np.asarray(asp27_oxygens, dtype=np.int32)].mean(axis=0)
    arg_pos = pos_perm_A[np.asarray(arg57_nitrogens, dtype=np.int32)].mean(axis=0)
    idx_near_asp = int(np.argmin(np.linalg.norm(lh_pos - asp_pos[None, :], axis=-1)))
    d_arg = np.linalg.norm(lh_pos - arg_pos[None, :], axis=-1)
    d_arg[idx_near_asp] = 1e9
    idx_near_arg = int(np.argmin(d_arg))

    lig_charges = np.zeros((3 * n_lig_triplets,), dtype=np.float64)
    for k_lh in range(n_lh):
        base_slot = 3 * k_lh
        if k_lh == idx_near_asp:
            # Protonated pteridine-like amidinium/N1+ group near Asp27 carboxylate
            lig_charges[base_slot] = 0.25
            lig_charges[base_slot + 1] = 0.25
            lig_charges[base_slot + 2] = 0.20
        elif k_lh == idx_near_arg:
            # Glutamate-like carboxylate group near Arg57 guanidinium
            lig_charges[base_slot] = -0.30
            lig_charges[base_slot + 1] = -0.20
            lig_charges[base_slot + 2] = -0.20
        else:
            q_c = float(rng.uniform(-0.18, 0.18))
            lig_charges[base_slot] = q_c
            lig_charges[base_slot + 1] = -0.5 * q_c
            lig_charges[base_slot + 2] = -0.5 * q_c
    # Enforce exact ligand charge neutrality (sum(q_lig) == 0.0)
    lig_charges -= np.mean(lig_charges)

    for new_i, old_i in enumerate(new_to_old):
        if new_i < n_prot or new_i >= lig_end:
            q, sig, eps = base_nb.getParticleParameters(old_i)
            nb_f.addParticle(q, sig, eps)
        else:
            rel = new_i - lig_start
            q_val = float(lig_charges[rel])
            if rel % 3 == 0:
                nb_f.addParticle(q_val, 0.330, 0.1094 * 4.184)
            else:
                nb_f.addParticle(q_val, 0.195, 0.0157 * 4.184)

    for ow, h1, h2 in kept_water_triplets_new:
        bonded_pairs_for_exc.append((ow, h1))
        bonded_pairs_for_exc.append((ow, h2))
    nb_f.createExceptionsFromBonds(bonded_pairs_for_exc, 1.0 / 1.2, 0.5)
    # Preserve exact protein 1-4 / 1-2 / 1-3 exceptions from base_nb
    for ex_i in range(base_nb.getNumExceptions()):
        p1, p2, qp, sig_e, eps_e = base_nb.getExceptionParameters(ex_i)
        if p1 < n_prot and p2 < n_prot:
            nb_f.addException(p1, p2, qp, sig_e, eps_e, True)
    holo_sys.addForce(nb_f)
    holo_sys.addForce(mm.CMMotionRemover(1))

    min_pos_A, prov = minimize_with_verified_openmm(
        holo_sys, pos_perm_A, max_iterations=minimize_iters, allow_cpu_smoke=allow_cpu_smoke
    )
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        holo_sys,
        min_pos_A,
        bucket_atoms=25000,
        bucket_waters=8192,
        electrostatics="pme",
        pme_alpha=0.34,
        cutoff_A=9.0,
    )

    obs_meta = _extract_topology_observables(pdb.topology)
    lig_heavy_arr = np.asarray(lig_heavy_indices, dtype=np.int32)
    min_real_pos = np.asarray(bun_f64.positions[:n_total], dtype=np.float64)

    # Identify top 5 key active-site protein-ligand heavy-atom contacts in minimized holo complex
    contact_candidates: list[tuple[float, int, int, str]] = []
    for p_idx in pocket_heavy_arr.tolist():
        p_coord = min_real_pos[p_idx]
        for k_l, l_idx in enumerate(lig_heavy_arr.tolist()):
            dr = min_real_pos[l_idx] - p_coord
            dr -= box_L * np.round(dr / box_L)
            dist = float(np.linalg.norm(dr))
            contact_candidates.append(
                (dist, p_idx, l_idx, f"{prot_atom_labels.get(p_idx, f'P{p_idx}')}-LIG:C{k_l + 1}")
            )
    contact_candidates.sort(key=lambda x: x[0])
    key_contacts: list[dict[str, Any]] = []
    used_res_prefixes: set[str] = set()
    for dist, p_idx, l_idx, label in contact_candidates:
        res_prefix = label.split(":")[0]
        if res_prefix in used_res_prefixes:
            continue
        used_res_prefixes.add(res_prefix)
        key_contacts.append(
            {
                "label": label,
                "prot_idx": int(p_idx),
                "lig_idx": int(l_idx),
                "ref_dist_A": float(dist),
            }
        )
        if len(key_contacts) >= 5:
            break

    mol_ids, n_mols = _extract_molecule_ids_from_system(holo_sys)
    solute_idx = np.arange(lig_end, dtype=np.int32)
    solvent_idx = np.arange(lig_end, n_total, dtype=np.int32)
    water_o_new = np.asarray([ow for ow, _, _ in kept_water_triplets_new], dtype=np.int32)

    return ValidatedSystemSpec(
        name="Holo_DHFR",
        electrostatics="pme",
        omm_system=holo_sys,
        omm_topology=None,
        positions_A=min_real_pos,
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=(64, 64, 64),
        pme_alpha=0.34,
        cutoff_A=9.0,
        n_real_atoms=n_total,
        n_waters=int(bun_f32.n_waters),
        n_constraints_total=holo_sys.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=solute_idx,
        solvent_indices=solvent_idx,
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        water_oxygen_indices=water_o_new,
        ca_indices=obs_meta["ca_indices"],
        protein_heavy_indices=obs_meta["protein_heavy_indices"],
        phi_indices=obs_meta["phi_indices"],
        psi_indices=obs_meta["psi_indices"],
        hbond_donors=obs_meta["hbond_donors"],
        hbond_donor_res=obs_meta["hbond_donor_res"],
        hbond_acceptors=obs_meta["hbond_acceptors"],
        hbond_acceptor_res=obs_meta["hbond_acceptor_res"],
        pocket_heavy_indices=pocket_heavy_arr,
        ligand_heavy_indices=lig_heavy_arr,
        key_contacts=key_contacts,
        minimization_provenance=prov,
    )


def build_apoa1_system(
    *,
    allow_cpu_smoke: bool = False,
    minimize_iters: int = 0,
) -> ValidatedSystemSpec:
    """Build 92,224-atom ApoA1 lipoprotein complex (108.86 x 108.86 x 77.76 A, 96x96x64 PME)."""
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    pdb_path = ROOT / "submodules" / "openmm" / "examples" / "benchmarks" / "apoa1.pdb"
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/lipid17.xml", "amber14/tip3p.xml")
    omm_sys_clean = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=True,
    )
    omm_sys_clean.setDefaultPeriodicBoxVectors(
        mm.Vec3(10.886, 0.0, 0.0),
        mm.Vec3(0.0, 10.886, 0.0),
        mm.Vec3(0.0, 0.0, 7.776),
    )
    for f in omm_sys_clean.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setPMEParameters(3.4, 96, 96, 64)
            f.setUseDispersionCorrection(True)

    n_real = omm_sys_clean.getNumParticles()
    pos_A = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    pos_A, prov = minimize_with_verified_openmm(
        omm_sys_clean, pos_A, max_iterations=minimize_iters, allow_cpu_smoke=allow_cpu_smoke
    )
    bun_f32, bun_f64, conv = _convert_openmm_system_to_bundles(
        omm_sys_clean,
        pos_A,
        bucket_atoms=100000,
        bucket_waters=24576,
        electrostatics="pme",
        pme_alpha=0.34,
        cutoff_A=9.0,
    )
    obs_meta = _extract_topology_observables(pdb.topology)
    mol_ids, n_mols = _extract_molecule_ids_from_system(omm_sys_clean)
    return ValidatedSystemSpec(
        name="ApoA1_92k",
        electrostatics="pme",
        omm_system=omm_sys_clean,
        omm_topology=pdb.topology,
        positions_A=np.asarray(bun_f64.positions[:n_real], dtype=np.float64),
        bundle_f32=bun_f32,
        bundle_f64=bun_f64,
        pme_grid=(96, 96, 64),
        pme_alpha=0.34,
        cutoff_A=9.0,
        n_real_atoms=n_real,
        n_waters=int(bun_f32.n_waters),
        n_constraints_total=omm_sys_clean.getNumConstraints(),
        n_constraints_solute=int(conv.system_metadata["n_solute_constraints"]),
        solute_indices=obs_meta["solute_indices"],
        solvent_indices=obs_meta["solvent_indices"],
        molecule_ids=mol_ids,
        n_molecules=n_mols,
        water_oxygen_indices=obs_meta["water_oxygen_indices"],
        ca_indices=obs_meta["ca_indices"],
        minimization_provenance=prov,
    )


# =============================================================================
# Structural & Dynamical Observable Kernels (Tier 4)
# =============================================================================


def compute_water_rdf_goo(
    replicas_O_A: np.ndarray,
    box_L: np.ndarray,
    *,
    r_max: float = 14.5,
    dr: float = 0.05,
    max_frames_total: int = 400,
) -> dict[str, Any]:
    """Compute liquid water O-O radial distribution function `g_OO(r)` across `(B, T, N_O, 3)`.

    Uses parabolic 3-point sub-bin interpolation around the first solvation peak (`~2.77 A`),
    first solvation minimum (`~3.35-3.65 A`), and spherical shell integration for first-shell
    coordination number `n_OO(r_min)`.
    """
    arr = np.asarray(replicas_O_A, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[None, ...]
    B, T, n_o, _ = arr.shape
    box_vec = np.asarray(box_L, dtype=np.float64).reshape(3)
    vol = float(np.prod(box_vec))
    rho_o = float(n_o) / vol

    r_eff_max = min(float(r_max), 0.49 * float(np.min(box_vec)))
    edges = np.arange(0.0, r_eff_max + dr, dr, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vols = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    stride = max(1, (B * T) // max(max_frames_total, B))
    per_rep_g: list[np.ndarray] = []
    per_rep_peak_r: list[float] = []
    per_rep_peak_h: list[float] = []
    per_rep_min_r: list[float] = []
    per_rep_coord: list[float] = []

    tri_i, tri_j = np.triu_indices(n_o, k=1)

    for b in range(B):
        frames_b = arr[b, ::stride]
        n_fb = int(frames_b.shape[0])
        counts = np.zeros((len(centers),), dtype=np.float64)
        for f_idx in range(n_fb):
            pos = frames_b[f_idx]
            dvec = pos[tri_i] - pos[tri_j]
            dvec -= box_vec[None, :] * np.round(dvec / box_vec[None, :])
            dists = np.linalg.norm(dvec, axis=-1)
            h, _ = np.histogram(dists, bins=edges)
            counts += 2.0 * h  # symmetric (i, j) and (j, i)
        ideal = float(n_fb) * float(n_o) * rho_o * shell_vols
        g_b = counts / np.maximum(ideal, 1e-30)
        per_rep_g.append(g_b)

        pk_r, pk_h, mn_r, coord_n = _extract_rdf_landmarks(centers, edges, g_b, rho_o)
        per_rep_peak_r.append(pk_r)
        per_rep_peak_h.append(pk_h)
        per_rep_min_r.append(mn_r)
        per_rep_coord.append(coord_n)

    g_stack = np.stack(per_rep_g, axis=0)
    g_mean = np.mean(g_stack, axis=0)
    g_se = np.std(g_stack, axis=0, ddof=max(0, min(1, B - 1))) / np.sqrt(max(B, 1))
    pk_r_m, pk_h_m, mn_r_m, coord_m = _extract_rdf_landmarks(centers, edges, g_mean, rho_o)

    def _se(vals: list[float]) -> float:
        if len(vals) <= 1:
            return 0.0
        return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    return {
        "r_A": [round(float(x), 4) for x in centers],
        "g_mean": [round(float(x), 5) for x in g_mean],
        "g_se": [round(float(x), 5) for x in g_se],
        "first_peak_r_A": float(pk_r_m),
        "first_peak_r_se_A": _se(per_rep_peak_r),
        "first_peak_height": float(pk_h_m),
        "first_peak_height_se": _se(per_rep_peak_h),
        "first_min_r_A": float(mn_r_m),
        "first_min_r_se_A": _se(per_rep_min_r),
        "first_min_height": float(np.interp(mn_r_m, centers, g_mean)),
        "coordination_number": float(coord_m),
        "coordination_number_se": _se(per_rep_coord),
    }


def _extract_rdf_landmarks(
    centers: np.ndarray,
    edges: np.ndarray,
    g_r: np.ndarray,
    rho_o: float,
) -> tuple[float, float, float, float]:
    """Parabolic sub-bin first peak, first minimum, and running coordination number."""
    pk_mask = (centers >= 2.40) & (centers <= 3.15)
    pk_idxs = np.flatnonzero(pk_mask)
    k_pk = int(pk_idxs[int(np.argmax(g_r[pk_idxs]))])
    r_pk, h_pk = float(centers[k_pk]), float(g_r[k_pk])
    if 0 < k_pk < len(centers) - 1:
        y0, y1, y2 = float(g_r[k_pk - 1]), float(g_r[k_pk]), float(g_r[k_pk + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12 and denom < 0.0:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            dr_bin = float(centers[1] - centers[0])
            r_pk = float(centers[k_pk]) + delta * dr_bin
            h_pk = y1 - 0.25 * (y0 - y2) * delta

    mn_mask = (centers >= max(r_pk + 0.25, 3.10)) & (centers <= 4.00)
    mn_idxs = np.flatnonzero(mn_mask)
    k_mn = int(mn_idxs[int(np.argmin(g_r[mn_idxs]))])
    r_mn = float(centers[k_mn])
    if 0 < k_mn < len(centers) - 1:
        y0, y1, y2 = float(g_r[k_mn - 1]), float(g_r[k_mn]), float(g_r[k_mn + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12 and denom > 0.0:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            dr_bin = float(centers[1] - centers[0])
            r_mn = float(centers[k_mn]) + delta * dr_bin

    # Integrate 4*pi*rho_O * r^2 * g(r) dr up to r_mn
    shell_vols = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    full_bins = edges[1:] <= r_mn
    coord = float(np.sum(rho_o * shell_vols[full_bins] * g_r[full_bins]))
    partial_idx = np.flatnonzero((edges[:-1] < r_mn) & (edges[1:] > r_mn))
    if partial_idx.size > 0:
        idx_p = int(partial_idx[0])
        v_part = (4.0 / 3.0) * np.pi * (r_mn**3 - float(edges[idx_p]) ** 3)
        coord += float(rho_o * v_part * g_r[idx_p])
    return r_pk, h_pk, r_mn, coord


def compute_unwrapped_water_diffusion(
    replicas_O_A: np.ndarray,
    box_L: np.ndarray,
    *,
    dt_save_ps: float,
    fit_start_ps: float,
    fit_end_ps: float,
    temperature_k: float = 300.0,
    eta_tip3p_mpa_s: float = 0.321,
) -> dict[str, Any]:
    """Compute unwrapped COM-drift-corrected Einstein self-diffusion `D_TIP3P` (`10^-5 cm^2/s`).

    Includes the exact Yeh-Hummer (J. Phys. Chem. B 2004, 108, 15873) periodic finite-size
    hydrodynamic correction `Delta D_YH = xi * k_B * T / (6 * pi * eta * L)` with `xi = 2.837297`.
    """
    arr = np.asarray(replicas_O_A, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[None, ...]
    B, T, _, _ = arr.shape
    box_vec = np.asarray(box_L, dtype=np.float64).reshape(3)
    L_mean_m = float(np.mean(box_vec)) * 1e-10

    # Unwrap inter-frame minimum-image increments
    d_step = arr[:, 1:, :, :] - arr[:, :-1, :, :]
    d_step -= box_vec[None, None, None, :] * np.round(d_step / box_vec[None, None, None, :])
    max_jump_A = float(np.max(np.abs(d_step))) if d_step.size > 0 else 0.0
    half_box_A = 0.5 * float(np.min(box_vec))

    unwrapped = np.zeros_like(arr)
    unwrapped[:, 0, :, :] = arr[:, 0, :, :]
    unwrapped[:, 1:, :, :] = arr[:, 0:1, :, :] + np.cumsum(d_step, axis=1)
    # Subtract instantaneous oxygen-COM displacement per frame to eliminate global thermostat COM walk
    com_t = np.mean(unwrapped, axis=2, keepdims=True)
    unwrapped = unwrapped - (com_t - com_t[:, 0:1, :, :])

    max_lag = min(T - 1, max(2, round(fit_end_ps / dt_save_ps)))
    min_lag = max(1, min(max_lag - 1, round(fit_start_ps / dt_save_ps)))
    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    tau_ps = lags.astype(np.float64) * float(dt_save_ps)

    msd_by_rep = np.zeros((B, len(lags)), dtype=np.float64)
    d_rep_1e5_cm2_s: list[float] = []

    fit_mask = (lags >= min_lag) & (lags <= max_lag)
    for b in range(B):
        u_b = unwrapped[b]
        for k_idx, lag in enumerate(lags):
            disp = u_b[lag:] - u_b[:-lag]
            msd_by_rep[b, k_idx] = float(np.mean(np.sum(disp * disp, axis=-1)))
        # 1 A^2 / ps = 10^-20 m^2 / 10^-12 s = 10^-8 m^2/s = 10^-4 cm^2/s = 10.0 * (10^-5 cm^2/s)
        slope_A2_ps, _ = np.polyfit(tau_ps[fit_mask], msd_by_rep[b, fit_mask], 1)
        d_rep_1e5_cm2_s.append(float((slope_A2_ps / 6.0) * 10.0))

    d_mean = float(np.mean(d_rep_1e5_cm2_s))
    d_se = (
        float(np.std(d_rep_1e5_cm2_s, ddof=1) / np.sqrt(B))
        if B > 1
        else 0.0
    )

    # Yeh-Hummer finite-size hydrodynamic correction: xi * k_B * T / (6 * pi * eta * L)
    xi_ewald = 2.837297
    kb_si = 1.380649e-23
    eta_pa_s = float(eta_tip3p_mpa_s) * 1e-3
    d_yh_m2_s = (xi_ewald * kb_si * float(temperature_k)) / (6.0 * np.pi * eta_pa_s * L_mean_m)
    d_yh_1e5_cm2_s = float(d_yh_m2_s * 1e9)  # 1 m^2/s = 10^4 cm^2/s = 10^9 * (10^-5 cm^2/s)

    return {
        "D_pbc_mean_1e5_cm2_s": d_mean,
        "D_pbc_se_1e5_cm2_s": d_se,
        "D_pbc_per_replica_1e5_cm2_s": [round(x, 4) for x in d_rep_1e5_cm2_s],
        "yeh_hummer_correction_1e5_cm2_s": d_yh_1e5_cm2_s,
        "D_infinite_box_est_1e5_cm2_s": float(d_mean + d_yh_1e5_cm2_s),
        "fit_window_ps": [float(tau_ps[fit_mask][0]), float(tau_ps[fit_mask][-1])],
        "dt_save_ps": float(dt_save_ps),
        "max_interframe_jump_A": max_jump_A,
        "half_box_A": half_box_A,
        "unwrap_aliasing_safe": bool(max_jump_A < 0.9 * half_box_A),
    }


def _unwrap_solute_frames_pbc(
    frames_A: np.ndarray,
    solute_idx: np.ndarray,
    ref_pos_A: np.ndarray,
    box_L: np.ndarray | None,
) -> np.ndarray:
    """Unwrap solute coordinates in `(T, N, 3)` cleanly relative to sequential covalent order & ref."""
    sub = np.asarray(frames_A[:, solute_idx, :], dtype=np.float64).copy()
    if box_L is None or not np.all(np.asarray(box_L) > 0):
        return sub
    box_vec = np.asarray(box_L, dtype=np.float64).reshape(3)
    ref_sub = np.asarray(ref_pos_A[solute_idx], dtype=np.float64)

    # First make solute chain contiguous within each frame by unwrapping atom k relative to atom k-1
    # whenever their reference distance is < 5.0 A (same chain/pocket)
    ref_step = ref_sub[1:] - ref_sub[:-1]
    ref_step_d = np.linalg.norm(ref_step, axis=-1)
    for k in range(1, sub.shape[1]):
        if ref_step_d[k - 1] < 8.0:
            dr = sub[:, k, :] - sub[:, k - 1, :]
            dr -= box_vec[None, :] * np.round(dr / box_vec[None, :])
            sub[:, k, :] = sub[:, k - 1, :] + dr
        else:
            # Separate solute component (e.g., bound pocket ligand): unwrap relative to solute anchor 0
            dr0 = sub[:, k, :] - sub[:, 0, :] - (ref_sub[k] - ref_sub[0])[None, :]
            dr0 -= box_vec[None, :] * np.round(dr0 / box_vec[None, :])
            sub[:, k, :] = sub[:, 0, :] + (ref_sub[k] - ref_sub[0])[None, :] + dr0
    return sub


def kabsch_align_to_reference(
    mobile: np.ndarray,
    ref: np.ndarray,
    align_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Superpose `mobile` `(T, M, 3)` onto `ref` `(M, 3)` using `align_indices` (Kabsch SVD)."""
    T = mobile.shape[0]
    ref_sel = ref[align_indices]
    ref_center = np.mean(ref_sel, axis=0)
    Q = ref_sel - ref_center[None, :]

    aligned = np.empty_like(mobile)
    rmsd_align = np.empty((T,), dtype=np.float64)

    for t in range(T):
        P_sel = mobile[t, align_indices, :]
        p_center = np.mean(P_sel, axis=0)
        P = P_sel - p_center[None, :]
        H = P.T @ Q
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        S = np.diag([1.0, 1.0, float(d) if d != 0 else 1.0])
        R_mat = U @ S @ Vt
        shifted = mobile[t] - p_center[None, :]
        rot = shifted @ R_mat
        aligned[t] = rot + ref_center[None, :]
        diff = aligned[t, align_indices, :] - ref_sel
        rmsd_align[t] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))

    return aligned, rmsd_align


def _dihedrals_deg(coords: np.ndarray, quad_idx: np.ndarray) -> np.ndarray:
    """Compute torsion angles in degrees `[-180, 180]` for `coords` `(T, M, 3)` and `quad_idx` `(K, 4)`."""
    if quad_idx.shape[0] == 0:
        return np.zeros((coords.shape[0], 0), dtype=np.float64)
    p0 = coords[:, quad_idx[:, 0], :]
    p1 = coords[:, quad_idx[:, 1], :]
    p2 = coords[:, quad_idx[:, 2], :]
    p3 = coords[:, quad_idx[:, 3], :]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_u = b1 / np.maximum(b1_norm, 1e-12)

    v = b0 - np.sum(b0 * b1_u, axis=-1, keepdims=True) * b1_u
    w = b2 - np.sum(b2 * b1_u, axis=-1, keepdims=True) * b1_u
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_u, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def compute_protein_replica_observables(
    replicas_pos_A: np.ndarray,
    spec: ValidatedSystemSpec,
) -> dict[str, Any]:
    """Evaluate full structural & pocket observable suite across `(B, T, N_real, 3)`."""
    arr = np.asarray(replicas_pos_A, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[None, ...]
    B, T, _, _ = arr.shape
    box_vec = np.diag(np.asarray(spec.bundle_f64.box, dtype=np.float64)) if spec.bundle_f64.shape_spec.has_pbc else None
    masses_real = np.asarray(spec.bundle_f64.masses[: spec.n_real_atoms], dtype=np.float64)

    # Map global atom indices onto compact solute array [0..N_solute-1]
    sol_idx = spec.solute_indices
    g2s = np.full((spec.n_real_atoms,), -1, dtype=np.int32)
    g2s[sol_idx] = np.arange(len(sol_idx), dtype=np.int32)

    ref_sol = _unwrap_solute_frames_pbc(
        spec.positions_A[None, ...], sol_idx, spec.positions_A, box_vec
    )[0]
    ca_s = g2s[spec.ca_indices]
    heavy_s = g2s[spec.protein_heavy_indices]
    heavy_masses = masses_real[spec.protein_heavy_indices]

    phi_s = g2s[spec.phi_indices] if spec.phi_indices.size > 0 else np.zeros((0, 4), dtype=np.int32)
    psi_s = g2s[spec.psi_indices] if spec.psi_indices.size > 0 else np.zeros((0, 4), dtype=np.int32)
    n_rama = min(phi_s.shape[0], psi_s.shape[0])

    don_s = g2s[spec.hbond_donors] if spec.hbond_donors.size > 0 else np.zeros((0, 2), dtype=np.int32)
    acc_s = g2s[spec.hbond_acceptors] if spec.hbond_acceptors.size > 0 else np.zeros((0,), dtype=np.int32)

    has_holo_lig = spec.ligand_heavy_indices.size > 0 and spec.pocket_heavy_indices.size > 0
    pocket_s = g2s[spec.pocket_heavy_indices] if has_holo_lig else np.zeros((0,), dtype=np.int32)
    lig_s = g2s[spec.ligand_heavy_indices] if has_holo_lig else np.zeros((0,), dtype=np.int32)

    ca_rmsd_rep_means: list[float] = []
    ca_rmsd_pooled: list[np.ndarray] = []
    ca_rmsf_reps: list[np.ndarray] = []
    rg_rep_means: list[float] = []
    rg_pooled: list[np.ndarray] = []
    rama_reps: list[np.ndarray] = []  # [alpha, beta, left_alpha, other]
    hbond_count_reps: list[float] = []
    hbond_pair_pers_reps: list[np.ndarray] = []

    lig_rmsd_rep_means: list[float] = []
    lig_rmsd_pooled: list[np.ndarray] = []
    contact_rep_means: dict[str, list[float]] = {c["label"]: [] for c in spec.key_contacts}
    contact_pooled: dict[str, list[np.ndarray]] = {c["label"]: [] for c in spec.key_contacts}

    # Precompute valid non-local backbone donor-acceptor pairs (|res_d - res_a| >= 3)
    valid_hb_pairs: tuple[np.ndarray, np.ndarray] | None = None
    if don_s.shape[0] > 0 and acc_s.shape[0] > 0:
        d_res = spec.hbond_donor_res[:, None]
        a_res = spec.hbond_acceptor_res[None, :]
        di_idx, aj_idx = np.nonzero(np.abs(d_res - a_res) >= 3)
        valid_hb_pairs = (di_idx, aj_idx)

    for b in range(B):
        sol_traj = _unwrap_solute_frames_pbc(arr[b], sol_idx, spec.positions_A, box_vec)
        aligned_ca, rmsd_ca = kabsch_align_to_reference(sol_traj, ref_sol, ca_s)
        ca_rmsd_rep_means.append(float(np.mean(rmsd_ca)))
        ca_rmsd_pooled.append(rmsd_ca)

        ca_coords = aligned_ca[:, ca_s, :]
        ca_mean = np.mean(ca_coords, axis=0, keepdims=True)
        rmsf_b = np.sqrt(np.mean(np.sum((ca_coords - ca_mean) ** 2, axis=-1), axis=0))
        ca_rmsf_reps.append(rmsf_b)

        # Mass-weighted radius of gyration on protein heavy atoms
        h_coords = sol_traj[:, heavy_s, :]
        m_tot = float(np.sum(heavy_masses))
        com_h = np.sum(h_coords * heavy_masses[None, :, None], axis=1, keepdims=True) / max(m_tot, 1e-12)
        rg_b = np.sqrt(
            np.sum(heavy_masses[None, :] * np.sum((h_coords - com_h) ** 2, axis=-1), axis=1)
            / max(m_tot, 1e-12)
        )
        rg_rep_means.append(float(np.mean(rg_b)))
        rg_pooled.append(rg_b)

        # Ramachandran basin occupancy on interior residues
        if n_rama > 0:
            phi_deg = _dihedrals_deg(sol_traj, phi_s[:n_rama])
            psi_deg = _dihedrals_deg(sol_traj, psi_s[:n_rama])
            is_alpha = (phi_deg >= -160.0) & (phi_deg <= -20.0) & (psi_deg >= -120.0) & (psi_deg <= 50.0)
            is_beta = ((phi_deg <= -40.0) | (phi_deg >= 160.0)) & (
                (psi_deg >= 50.0) | (psi_deg <= -120.0)
            ) & (~is_alpha)
            is_left = (phi_deg > 0.0) & (phi_deg <= 160.0) & (psi_deg >= -60.0) & (psi_deg <= 90.0)
            f_a = float(np.mean(is_alpha))
            f_b = float(np.mean(is_beta))
            f_l = float(np.mean(is_left))
            f_o = max(0.0, 1.0 - (f_a + f_b + f_l))
            rama_reps.append(np.array([f_a, f_b, f_l, f_o], dtype=np.float64))

        # Backbone H-bond persistence: d(N..O) < 3.5 A and angle(N-H..O) > 120 deg
        if valid_hb_pairs is not None and valid_hb_pairs[0].size > 0:
            di_idx, aj_idx = valid_hb_pairs
            r_N = sol_traj[:, don_s[di_idx, 0], :]
            r_H = sol_traj[:, don_s[di_idx, 1], :]
            r_O = sol_traj[:, acc_s[aj_idx], :]
            d_NO = np.linalg.norm(r_O - r_N, axis=-1)
            v_HN = r_N - r_H
            v_HO = r_O - r_H
            cos_nho = np.sum(v_HN * v_HO, axis=-1) / np.maximum(
                np.linalg.norm(v_HN, axis=-1) * np.linalg.norm(v_HO, axis=-1), 1e-12
            )
            intact = (d_NO < 3.5) & (cos_nho < -0.5)  # cos(120 deg) = -0.5 for angle(N-H-O) > 120
            hbond_count_reps.append(float(np.mean(np.sum(intact, axis=1))))
            hbond_pair_pers_reps.append(np.mean(intact, axis=0))

        # Holo-DHFR pocket-aligned ligand heavy-atom RMSD & key active-site contact distances
        if has_holo_lig:
            aligned_pkt, _ = kabsch_align_to_reference(sol_traj, ref_sol, pocket_s)
            lig_diff = aligned_pkt[:, lig_s, :] - ref_sol[None, lig_s, :]
            lig_rmsd_t = np.sqrt(np.mean(np.sum(lig_diff * lig_diff, axis=-1), axis=-1))
            lig_rmsd_rep_means.append(float(np.mean(lig_rmsd_t)))
            lig_rmsd_pooled.append(lig_rmsd_t)

            for c in spec.key_contacts:
                ps_i = int(g2s[c["prot_idx"]])
                ls_i = int(g2s[c["lig_idx"]])
                d_pl = np.linalg.norm(sol_traj[:, ls_i, :] - sol_traj[:, ps_i, :], axis=-1)
                contact_rep_means[c["label"]].append(float(np.mean(d_pl)))
                contact_pooled[c["label"]].append(d_pl)

    def _mean_se(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return m, s

    rmsd_m, rmsd_se = _mean_se(ca_rmsd_rep_means)
    rg_m, rg_se = _mean_se(rg_rep_means)
    rmsf_stack = np.stack(ca_rmsf_reps, axis=0)
    rmsf_mean = np.mean(rmsf_stack, axis=0)
    rmsf_se = np.std(rmsf_stack, axis=0, ddof=max(0, min(1, B - 1))) / np.sqrt(max(B, 1))

    rama_dict: dict[str, float] = {}
    if rama_reps:
        r_stk = np.stack(rama_reps, axis=0)
        r_m = np.mean(r_stk, axis=0)
        r_se = np.std(r_stk, axis=0, ddof=max(0, min(1, B - 1))) / np.sqrt(max(B, 1))
        for idx_k, key_n in enumerate(("alpha", "beta", "left_alpha", "other")):
            rama_dict[f"{key_n}_mean"] = float(r_m[idx_k])
            rama_dict[f"{key_n}_se"] = float(r_se[idx_k])

    hb_m, hb_se = _mean_se(hbond_count_reps)
    hb_pers_mean = (
        np.mean(np.stack(hbond_pair_pers_reps, axis=0), axis=0)
        if hbond_pair_pers_reps
        else np.zeros((0,), dtype=np.float64)
    )

    out: dict[str, Any] = {
        "n_replicas": B,
        "n_frames_per_replica": T,
        "ca_rmsd_mean_A": rmsd_m,
        "ca_rmsd_se_A": rmsd_se,
        "ca_rmsd_per_replica_A": [round(x, 4) for x in ca_rmsd_rep_means],
        "_ca_rmsd_pooled": np.concatenate(ca_rmsd_pooled, axis=0),
        "ca_rmsf_mean_profile_A": [round(float(x), 4) for x in rmsf_mean],
        "ca_rmsf_se_profile_A": [round(float(x), 4) for x in rmsf_se],
        "rg_mean_A": rg_m,
        "rg_se_A": rg_se,
        "_rg_pooled": np.concatenate(rg_pooled, axis=0),
        "ramachandran": rama_dict,
        "hbond_intact_count_mean": hb_m,
        "hbond_intact_count_se": hb_se,
        "_hbond_persistence_vec": hb_pers_mean,
    }

    if has_holo_lig:
        lm, lse = _mean_se(lig_rmsd_rep_means)
        contacts_summary: list[dict[str, Any]] = []
        for c in spec.key_contacts:
            lbl = c["label"]
            cm, cse = _mean_se(contact_rep_means[lbl])
            pooled_c = np.concatenate(contact_pooled[lbl], axis=0)
            contacts_summary.append(
                {
                    "label": lbl,
                    "ref_dist_A": float(c["ref_dist_A"]),
                    "mean_dist_A": cm,
                    "se_dist_A": cse,
                    "sd_dist_A": float(np.std(pooled_c)),
                }
            )
        out["holo_ligand"] = {
            "pocket_heavy_rmsd_mean_A": lm,
            "pocket_heavy_rmsd_se_A": lse,
            "pocket_heavy_rmsd_per_replica_A": [round(x, 4) for x in lig_rmsd_rep_means],
            "key_contacts": contacts_summary,
        }
        out["_lig_rmsd_pooled"] = np.concatenate(lig_rmsd_pooled, axis=0)

    return out


def strip_private_arrays(obs: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of `obs` with private `_` pooled arrays removed."""
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def compare_protein_ensembles(obs_a: dict[str, Any], obs_b: dict[str, Any]) -> dict[str, Any]:
    """Quantify statistical agreement between two `B`-replica ensembles (e.g. Prolix vs OpenMM-A or OpenMM-A vs B)."""
    ks_rmsd = stats.ks_2samp(obs_a["_ca_rmsd_pooled"], obs_b["_ca_rmsd_pooled"])
    ks_rg = stats.ks_2samp(obs_a["_rg_pooled"], obs_b["_rg_pooled"])

    prof_a = np.asarray(obs_a["ca_rmsf_mean_profile_A"], dtype=np.float64)
    prof_b = np.asarray(obs_b["ca_rmsf_mean_profile_A"], dtype=np.float64)
    if prof_a.size >= 2 and np.std(prof_a) > 1e-12 and np.std(prof_b) > 1e-12:
        rmsf_r = float(np.corrcoef(prof_a, prof_b)[0, 1])
    else:
        rmsf_r = 1.0
    rmsf_rmse = float(np.sqrt(np.mean((prof_a - prof_b) ** 2)))

    hb_a = obs_a["_hbond_persistence_vec"]
    hb_b = obs_b["_hbond_persistence_vec"]
    active_hb = (hb_a > 0.05) | (hb_b > 0.05)
    if np.sum(active_hb) >= 2 and np.std(hb_a[active_hb]) > 1e-12 and np.std(hb_b[active_hb]) > 1e-12:
        hb_r = float(np.corrcoef(hb_a[active_hb], hb_b[active_hb])[0, 1])
    else:
        hb_r = 1.0
    hb_mae = float(np.mean(np.abs(hb_a[active_hb] - hb_b[active_hb]))) if np.any(active_hb) else 0.0

    rama_a = obs_a.get("ramachandran", {})
    rama_b = obs_b.get("ramachandran", {})
    rama_l1 = sum(
        abs(float(rama_a.get(f"{k}_mean", 0.0)) - float(rama_b.get(f"{k}_mean", 0.0)))
        for k in ("alpha", "beta", "left_alpha", "other")
    )

    comb_rmsd_se = max(float(np.hypot(obs_a["ca_rmsd_se_A"], obs_b["ca_rmsd_se_A"])), 1e-6)
    comb_rg_se = max(float(np.hypot(obs_a["rg_se_A"], obs_b["rg_se_A"])), 1e-6)

    res: dict[str, Any] = {
        "ca_rmsd_diff_A": float(obs_a["ca_rmsd_mean_A"] - obs_b["ca_rmsd_mean_A"]),
        "ca_rmsd_z_score": float(abs(obs_a["ca_rmsd_mean_A"] - obs_b["ca_rmsd_mean_A"]) / comb_rmsd_se),
        "ca_rmsd_ks_stat": float(ks_rmsd.statistic),
        "ca_rmsd_ks_pvalue": float(ks_rmsd.pvalue),
        "ca_rmsf_pearson_r": rmsf_r,
        "ca_rmsf_rmse_A": rmsf_rmse,
        "rg_diff_A": float(obs_a["rg_mean_A"] - obs_b["rg_mean_A"]),
        "rg_z_score": float(abs(obs_a["rg_mean_A"] - obs_b["rg_mean_A"]) / comb_rg_se),
        "rg_ks_stat": float(ks_rg.statistic),
        "ramachandran_l1_diff": float(rama_l1),
        "hbond_count_diff": float(obs_a["hbond_intact_count_mean"] - obs_b["hbond_intact_count_mean"]),
        "hbond_persistence_pearson_r": hb_r,
        "hbond_persistence_mae": hb_mae,
    }

    if "_lig_rmsd_pooled" in obs_a and "_lig_rmsd_pooled" in obs_b:
        ks_lig = stats.ks_2samp(obs_a["_lig_rmsd_pooled"], obs_b["_lig_rmsd_pooled"])
        la = obs_a["holo_ligand"]
        lb = obs_b["holo_ligand"]
        comb_lig_se = max(float(np.hypot(la["pocket_heavy_rmsd_se_A"], lb["pocket_heavy_rmsd_se_A"])), 1e-6)
        contact_diffs: list[dict[str, Any]] = []
        for ca_c, cb_c in zip(la["key_contacts"], lb["key_contacts"], strict=False):
            contact_diffs.append(
                {
                    "label": ca_c["label"],
                    "diff_A": float(ca_c["mean_dist_A"] - cb_c["mean_dist_A"]),
                }
            )
        res["holo_ligand_comparison"] = {
            "pocket_heavy_rmsd_diff_A": float(la["pocket_heavy_rmsd_mean_A"] - lb["pocket_heavy_rmsd_mean_A"]),
            "pocket_heavy_rmsd_z_score": float(
                abs(la["pocket_heavy_rmsd_mean_A"] - lb["pocket_heavy_rmsd_mean_A"]) / comb_lig_se
            ),
            "pocket_heavy_rmsd_ks_stat": float(ks_lig.statistic),
            "key_contact_diffs": contact_diffs,
        }

    return res


def clone_openmm_system(omm_system: Any) -> Any:
    """Deep-copy an OpenMM System via XML roundtrip."""
    import openmm as mm

    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(omm_system))
