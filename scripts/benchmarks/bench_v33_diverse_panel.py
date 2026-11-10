#!/usr/bin/env python3
"""Pallas v3.3 6-Axis Realistic Benchmark Panel & Sub-Step Profiler.

Covers 5 realistic MD & ML-force-field benchmark axes (+ cross-architecture hardware axis)
with head-to-head OpenMM CUDA (`single` and `mixed`) and kUPS baselines plus continuous
`xtrax.profiling.probe_unit` sub-step attribution (`ms/call` and `% of step time`):

- Axis 1 — Implicit Solvent Fold Panel (`OpenMM gbsa` + Prolix Catalog):
  `1UAO` (138 atoms), `1VII` (596), `1CRN` (642), `2GB1` (855), `1UBQ` (1,231),
  `5DFR` (2,489), and `1AKE` (3,341) with `OBC2` Generalized Born and `ff19SB` 2D CMAPs
  across `B = 1, 4, 16, 64`.
- Axis 2 — Explicit Solvent Multi-Protein & Low-Utilization Ragged Screen:
  `1UAO_solvated_tip3p` (1,963 atoms in 4,096 and 8,192 buckets), `1VII` (4,199 in
  8,192 and 25,000 buckets — 17%–51% occupancy), `1UBQ` (6,658 in 8,192 and 25,000
  buckets — 27%–81% occupancy), `DHFR` (23,558 in 25,000 bucket, `rf` vs `pme`),
  and solvated `1AKE` (44,999 in 50,000 bucket).
- Axis 3 — Large Non-Cubic Lipoprotein Complex (`OpenMM apoa1pme` / `apoa1rf`):
  `submodules/openmm/examples/benchmarks/apoa1.pdb` (92,224 atoms, 108.86 x 108.86 x 77.76 A
  non-cubic orthorhombic box, 96 x 96 x 64 PME grid, 100,000 bucket) across `B = 1, 2, 4, 8`.
- Axis 4 — Constrained `HBonds` (RATTLE) + `NPT` Barostat + Energy-Logged MD (`kUPS` / OpenMM):
  `1UBQ` and `DHFR` with solute X-H `rattle_positions`/`rattle_velocities` enabled,
  Stochastic Cell Rescaling NPT barostat (dynamic box per step), and kUPS-style logged
  `value_energy_and_forces` vs unlogged `return_forces=True`.
- Axis 5 — Differentiable Force-Field & Structure Refinement (`jax.value_and_grad`):
  Forward + backward execution (`dE/dr`, `dE/dq`, `dE/dsigma`, `dE/depsilon`) across
  `1VII`, `1UBQ`, and `DHFR` (`B = 1, 4, 8`) comparing `MK-48` row-NL vs `MK-51`
  Morton tile-pair custom VJPs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import xtrax.profiling as xp

import prolix.physics.neighbor_list as nl_mod
import scripts.experiments.bench_dhfr_parity as B_dhfr
from prolix.api.bundle_md import displacement_fn_for_bundle, energy_fn_from_bundle
from prolix.api.ensemble_plan import (
    _SINGLE_INFERENCE_CACHE,
    _STACKED_INFERENCE_CACHE,
    EnsemblePlan,
    _encode_bundle_nl,
)
from prolix.openmm_xml_importer import parse_openmm_system_xml
from prolix.pallas_kernels import (
    _infer_first_pure_water_block,
    build_tile_pair_neighbor_list,
    pallas_bonded_all_fused,
    pallas_bonded_and_nonbonded_nl_fused,
    pallas_nonbonded_tile_pairs,
)
from prolix.physics.cmap import precompute_cmap_coefficients
from prolix.physics.neighbor_list import (
    ExclusionSpec,
    build_tile_pair_neighbor_list_gpu,
    pair_list_to_dense_padded,
)
from prolix.physics.pallas_generalized_born import pallas_gbsa_dense
from prolix.physics.pme import influence_function, spme_energy_with_forces, spme_forces
from prolix.physics.pme_pallas import (
    compute_pme_spatial_perm,
    gather_pallas,
    spread_charges_pallas,
)
from prolix.physics.settle import (
    pallas_baoab_post_force,
    pallas_baoab_pre_force,
    settle_csvr_npt,
    settle_langevin,
)
from prolix.physics.system import (
    _mask,
    _pad_1d,
    _pad_2d,
    _pad_3d,
    _pad_positions_with_ghost_lattice,
    make_bundle_from_system,
)
from prolix.types.bundles import MolecularBundle, MolecularShapeSpec
from scripts.benchmarks._b1_paramize import _resolve_ff_path
from scripts.benchmarks.bench_e2e_ensemble_buckets import (
    _cast_bundle_precision,
    _make_truncated_solvated_bundle,
    _probe_ms,
)


def _clear_caches() -> None:
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()


def _ns_per_day(ms_per_step: float, dt_fs: float = 2.0) -> float:
    if ms_per_step <= 0.0 or not np.isfinite(ms_per_step):
        return 0.0
    ps_per_step = dt_fs * 1e-3
    ns_per_step = ps_per_step * 1e-3
    steps_per_day = (86400.0 * 1000.0) / ms_per_step
    return float(steps_per_day * ns_per_step)


def _substep_breakdown_dict(substeps_ms: dict[str, float]) -> dict[str, Any]:
    total_ms = sum(max(0.0, float(v)) for v in substeps_ms.values())
    pct = {
        f"{k}_pct": (100.0 * float(v) / total_ms if total_ms > 0.0 else 0.0)
        for k, v in substeps_ms.items()
    }
    return {
        "total_substeps_ms": float(total_ms),
        **{f"{k}_ms": float(v) for k, v in substeps_ms.items()},
        **pct,
    }


def _re_pad_bundle_to_custom_bucket(
    bundle: MolecularBundle,
    bucket_atoms: int,
    bucket_waters: int,
    *,
    electrostatic_method: str = "pme",
    include_constraints: bool = False,
) -> MolecularBundle:
    """Re-pad an existing MolecularBundle into an exact `(bucket_atoms, bucket_waters)` bucket."""
    n_real = int(jnp.asarray(bundle.n_atoms))
    assert n_real <= bucket_atoms, f"n_real={n_real} > bucket_atoms={bucket_atoms}"
    n_waters_real = int(jnp.asarray(bundle.n_waters))
    assert n_waters_real <= bucket_waters, (
        f"n_waters_real={n_waters_real} > bucket_waters={bucket_waters}"
    )

    def _resize_1d(arr, target_len, pad_val):
        cur = arr[:n_real]
        if target_len > n_real:
            return jnp.pad(cur, (0, target_len - n_real), constant_values=pad_val)
        return cur[:target_len]

    def _resize_2d(arr, target_len, pad_val):
        cur = arr[:n_real]
        if target_len > n_real:
            return jnp.pad(cur, ((0, target_len - n_real), (0, 0)), constant_values=pad_val)
        return cur[:target_len]

    pos_real = bundle.positions[:n_real]
    box_vec = jnp.diag(bundle.box)
    has_pbc = bool(bundle.shape_spec.has_pbc)
    if has_pbc and bool(jnp.all(box_vec > 0)):
        pos_real = jnp.mod(pos_real, box_vec[None, :])
        if n_waters_real > 0:
            w_real_init = bundle.water_indices[:n_waters_real]
            ow_i = w_real_init[:, 0]
            r_ow = pos_real[ow_i]
            for h_col in (1, 2):
                h_i = w_real_init[:, h_col]
                dr = pos_real[h_i] - r_ow
                dr = dr - box_vec[None, :] * jnp.round(dr / box_vec[None, :])
                pos_real = pos_real.at[h_i].set(r_ow + dr)

    n_ghost = bucket_atoms - n_real
    if n_ghost > 0:
        if has_pbc and bool(jnp.all(box_vec > 0)):
            grid_g = int(np.ceil(n_ghost ** (1.0 / 3.0)))
            axes = [
                jnp.linspace(1.0, float(box_vec[d]) - 1.0, grid_g, dtype=pos_real.dtype)
                for d in range(3)
            ]
            gx, gy, gz = jnp.meshgrid(*axes, indexing="ij")
            ghost_pos = jnp.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=-1)[
                :n_ghost
            ]
        else:
            ghost_pos = jnp.zeros((n_ghost, 3), dtype=pos_real.dtype)
        pos_padded = jnp.concatenate([pos_real, ghost_pos], axis=0)
    else:
        pos_padded = pos_real

    atom_mask = jnp.arange(bucket_atoms) < n_real
    charges = _resize_1d(bundle.charges, bucket_atoms, 0.0)
    sigmas = _resize_1d(bundle.sigmas, bucket_atoms, 1.0)
    epsilons = _resize_1d(bundle.epsilons, bucket_atoms, 0.0)
    masses = _resize_1d(bundle.masses, bucket_atoms, 1.0)
    radii = _resize_1d(bundle.radii, bucket_atoms, 1.5)
    scaled_radii = _resize_1d(bundle.scaled_radii, bucket_atoms, 1.2)

    if bucket_waters > 0:
        w_real = bundle.water_indices[:n_waters_real]
        w_pad_n = bucket_waters - n_waters_real
        if w_pad_n > 0:
            w_pad = jnp.zeros((w_pad_n, 3), dtype=w_real.dtype)
            w_idx = jnp.concatenate([w_real, w_pad], axis=0)
        else:
            w_idx = w_real[:bucket_waters]
        w_mask = jnp.arange(bucket_waters) < n_waters_real
    else:
        w_idx = jnp.zeros((1, 3), dtype=jnp.int32)
        w_mask = jnp.zeros((1,), dtype=bool)

    ei_raw = _resize_2d(bundle.excl_dense_indices, bucket_atoms, -1)
    ei_valid = (ei_raw >= 0) & (ei_raw < n_real) & atom_mask[:, None]
    ei = jnp.where(ei_valid, ei_raw, -1)
    ev = jnp.where(ei_valid, _resize_2d(bundle.excl_dense_scales_vdw, bucket_atoms, 1.0), 1.0)
    ee = jnp.where(ei_valid, _resize_2d(bundle.excl_dense_scales_elec, bucket_atoms, 1.0), 1.0)

    pme_alpha_val = (
        jnp.asarray(0.0, dtype=bundle.pme_alpha.dtype)
        if electrostatic_method.lower() in ("rf", "reaction_field", "cutoff")
        else bundle.pme_alpha
    )

    c_pairs = bundle.constraint_pairs
    c_lens = bundle.constraint_lengths
    c_mask = bundle.constraint_mask
    n_cons = bundle.n_constraints
    has_real_cons = bool(include_constraints and bundle.shape_spec.has_real_constraints)
    if not include_constraints:
        c_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
        c_lens = jnp.zeros((0,), dtype=jnp.float32)
        c_mask = jnp.zeros((0,), dtype=bool)
        n_cons = jnp.array(0, dtype=jnp.int32)

    new_spec = dataclasses.replace(
        bundle.shape_spec,
        atom_bucket_idx=bucket_atoms,
        water_bucket_idx=max(bucket_waters, 1),
        use_size_buckets=False,
        exact_n_atoms=bucket_atoms,
        exact_n_waters=max(bucket_waters, 1),
        has_real_constraints=has_real_cons,
    )
    return dataclasses.replace(
        bundle,
        positions=pos_padded,
        masses=masses,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=scaled_radii,
        atom_mask=atom_mask,
        water_indices=w_idx,
        water_mask=w_mask,
        excl_dense_indices=ei,
        excl_dense_scales_vdw=ev,
        excl_dense_scales_elec=ee,
        pme_alpha=pme_alpha_val,
        constraint_pairs=c_pairs,
        constraint_lengths=c_lens,
        constraint_mask=c_mask,
        n_constraints=n_cons,
        shape_spec=new_spec,
    )


def build_implicit_protein_bundle(
    name: str,
    pdb_rel_path: str,
    *,
    keep_chain: str | None = None,
    add_hydrogens_with_openmm: bool = False,
    bucket_atoms: int | None = None,
) -> tuple[MolecularBundle, Any, Any]:
    """Build an implicit-solvent (`OBC2` GBSA + `ff19SB` 2D CMAPs) bundle + OpenMM System."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    import proxide
    from proxide import (
        CoordFormat,
        OutputSpec,
        assign_mbondi2_radii,
        assign_obc2_scaling_factors,
        parse_structure,
    )

    ff_xml = _resolve_ff_path("protein.ff19SB.xml")
    omm_ff = app.ForceField(ff_xml, "implicit/obc2.xml")

    pdb_path = ROOT / pdb_rel_path
    pdb = app.PDBFile(str(pdb_path))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    std_aa = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
        "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NME",
    }
    to_del = [r for r in modeller.topology.residues() if r.name not in std_aa]
    if to_del:
        modeller.delete(to_del)
    if keep_chain is not None:
        chains_del = [c for c in modeller.topology.chains() if c.id != keep_chain]
        if chains_del:
            modeller.delete(chains_del)
    if add_hydrogens_with_openmm:
        modeller.addHydrogens(omm_ff)

    omm_system = omm_ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as tf:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, tf)
        tf.flush()
        spec = OutputSpec(
            coord_format=CoordFormat.Full,
            parameterize_md=True,
            force_field=ff_xml,
        )
        prot = parse_structure(tf.name, spec)

    radii = np.asarray(assign_mbondi2_radii(list(prot.atom_names), prot.bonds), dtype=np.float32)
    obc_scale = np.asarray(assign_obc2_scaling_factors(list(prot.atom_names)), dtype=np.float32)
    scaled_radii = radii * obc_scale
    masses = np.asarray(proxide.assign_masses(list(prot.atom_names)), dtype=np.float32)
    cmap_g = (
        np.asarray(prot.cmap_energy_grids, dtype=np.float32)
        if prot.cmap_energy_grids is not None
        else None
    )
    cmap_idx = (
        np.asarray(prot.cmap_indices, dtype=np.int32)
        if getattr(prot, "cmap_indices", None) is not None
        else None
    )
    cmap_coeffs = (
        precompute_cmap_coefficients(jnp.asarray(cmap_g))
        if cmap_g is not None and cmap_g.size > 0
        else None
    )
    per_torsion_g = (
        cmap_g[np.clip(cmap_idx, 0, cmap_g.shape[0] - 1)]
        if (cmap_g is not None and cmap_idx is not None and cmap_idx.size > 0)
        else None
    )

    sys_ns = SimpleNamespace(
        positions=np.asarray(prot.coordinates, dtype=np.float32),
        masses=masses,
        bonds=np.asarray(prot.bonds),
        bond_params=np.asarray(prot.bond_params),
        angles=np.asarray(prot.angles),
        angle_params=np.asarray(prot.angle_params),
        dihedrals=np.asarray(prot.proper_dihedrals),
        dihedral_params=np.asarray(prot.dihedral_params),
        impropers=np.asarray(prot.impropers),
        improper_params=np.asarray(prot.improper_params),
        charges=np.asarray(prot.charges),
        sigmas=np.asarray(prot.sigmas),
        epsilons=np.asarray(prot.epsilons),
        radii=radii,
        scaled_radii=scaled_radii,
        cmap_torsions=np.asarray(prot.cmap_torsions) if prot.cmap_torsions is not None else None,
        cmap_energy_grids=per_torsion_g,
        cmap_indices=cmap_idx,
        cmap_coeffs=cmap_coeffs,
        has_implicit_solvent=True,
    )
    excl = ExclusionSpec.from_protein(prot)
    bundle = make_bundle_from_system(
        sys_ns,
        boundary_condition="free",
        exclusion_spec=excl,
        has_implicit_solvent=True,
    )
    if bucket_atoms is not None:
        bundle = _re_pad_bundle_to_custom_bucket(bundle, bucket_atoms, 8)
    return _cast_bundle_precision(bundle, jnp.float32), omm_system, modeller.positions


def build_apoa1_bundle(
    bucket_atoms: int = 100000,
    bucket_waters: int = 24576,
    *,
    electrostatic_method: str = "pme",
    include_constraints: bool = False,
) -> tuple[MolecularBundle, Any, Any]:
    """Build the 92,224-atom ApoA1 non-cubic lipoprotein bundle (`108.86 x 108.86 x 77.76 A`)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    pdb_path = ROOT / "submodules" / "openmm" / "examples" / "benchmarks" / "apoa1.pdb"
    pdb = app.PDBFile(str(pdb_path))
    pos_nm = np.array(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float32)
    pos_A = pos_nm * 10.0

    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/lipid17.xml", "amber14/tip3p.xml")
    is_rf = electrostatic_method.lower() in ("rf", "reaction_field", "cutoff")
    nb_method = app.CutoffPeriodic if is_rf else app.PME
    cutoff_nm = 1.0 if is_rf else 0.9
    omm_sys = ff.createSystem(
        pdb.topology,
        nonbondedMethod=nb_method,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )
    omm_sys.setDefaultPeriodicBoxVectors(
        mm.Vec3(10.886, 0.0, 0.0),
        mm.Vec3(0.0, 10.886, 0.0),
        mm.Vec3(0.0, 0.0, 7.776),
    )
    for f in omm_sys.getForces():
        if isinstance(f, mm.NonbondedForce) and not is_rf:
            f.setPMEParameters(0.34 * 10.0, 96, 96, 64)

    xml_str = mm.XmlSerializer.serialize(omm_sys)
    data = parse_openmm_system_xml(xml_str, positions=pos_A)

    n = int(data.n_atoms)
    a = int(bucket_atoms)
    wb = int(bucket_waters)
    nb = max(int(data.bonds.shape[0]), 1)
    na = max(int(data.angles.shape[0]), 1)
    nd = max(int(data.dihedrals.shape[0]), 1)
    nw = int(data.water_indices.shape[0])
    ne = int(data.excl_indices.shape[0])
    nx = max(int(data.exception_pairs.shape[0]), 1)
    nc = max(int(data.cmap_torsions.shape[0]), 1)
    n_constr = int(data.constraint_pairs.shape[0]) if include_constraints else 0
    cb_constr = max(n_constr, 0)

    excl_dense_indices, excl_dense_scales_vdw, excl_dense_scales_elec = pair_list_to_dense_padded(
        np.asarray(data.excl_indices),
        np.asarray(data.excl_scales_vdw),
        np.asarray(data.excl_scales_elec),
        jnp.ones(ne, dtype=bool),
        a,
        max_exclusions=32,
    )

    # Filter any water O-H / H-O-H bonds/angles if present so water is purely SETTLE
    water_atom_set = set(np.asarray(data.water_indices).reshape(-1).tolist())
    if water_atom_set:
        b_np = np.asarray(data.bonds)
        b_keep = np.array([int(r[0]) not in water_atom_set and int(r[1]) not in water_atom_set for r in b_np], dtype=bool)
        bonds_f = jnp.asarray(b_np[b_keep], dtype=jnp.int32)
        bpar_f = jnp.asarray(np.asarray(data.bond_params)[b_keep], dtype=jnp.float32)
        nb = int(bonds_f.shape[0])

        a_np = np.asarray(data.angles)
        a_keep = np.array([int(r[0]) not in water_atom_set and int(r[1]) not in water_atom_set and int(r[2]) not in water_atom_set for r in a_np], dtype=bool)
        angles_f = jnp.asarray(a_np[a_keep], dtype=jnp.int32)
        apar_f = jnp.asarray(np.asarray(data.angle_params)[a_keep], dtype=jnp.float32)
        na = int(angles_f.shape[0])
    else:
        bonds_f, bpar_f = data.bonds, data.bond_params
        angles_f, apar_f = data.angles, data.angle_params

    bb = max(nb, 1)
    ab = max(na, 1)
    db = max(nd, 1)
    eb = max(ne, 1)
    xb = max(nx, 1)
    cb = max(nc, 1)

    box_vec = jnp.diag(data.box).astype(jnp.float32)
    pos_real = jnp.mod(jnp.asarray(data.positions, dtype=jnp.float32), box_vec[None, :])
    if nw > 0:
        w_real_init = data.water_indices[:nw]
        ow_i = w_real_init[:, 0]
        r_ow = pos_real[ow_i]
        for h_col in (1, 2):
            h_i = w_real_init[:, h_col]
            dr = pos_real[h_i] - r_ow
            dr = dr - box_vec[None, :] * jnp.round(dr / box_vec[None, :])
            pos_real = pos_real.at[h_i].set(r_ow + dr)

    pos_padded = _pad_positions_with_ghost_lattice(pos_real, a, box_vec, dtype=jnp.float32)

    spec = MolecularShapeSpec(
        atom_bucket_idx=a,
        bond_bucket_idx=bb,
        angle_bucket_idx=ab,
        dihedral_bucket_idx=db,
        water_bucket_idx=wb,
        excl_bucket_idx=eb,
        cmap_bucket_idx=cb,
        exception_bucket_idx=xb,
        has_pbc=True,
        has_implicit_solvent=False,
        boundary_condition="periodic",
        has_real_water=nw > 0,
        use_size_buckets=False,
        exact_n_atoms=a,
        exact_n_waters=wb,
        cmap_map_bucket_idx=0,
        constraint_bucket_idx=cb_constr,
        has_real_constraints=n_constr > 0,
        is_triclinic=False,
        has_real_cosolvents=False,
    )

    pme_alpha_val = 0.0 if is_rf else float(data.pme_alpha or 0.34)
    cutoff_val = 10.0 if is_rf else float(data.nonbonded_cutoff or 9.0)

    bundle = MolecularBundle(
        positions=pos_padded,
        masses=jnp.pad(data.masses, (0, a - n), constant_values=1.0).astype(jnp.float32),
        charges=_pad_1d(data.charges, a, dtype=jnp.float32),
        sigmas=_pad_1d(data.sigmas, a, dtype=jnp.float32),
        epsilons=_pad_1d(data.epsilons, a, dtype=jnp.float32),
        radii=jnp.pad(data.radii, (0, a - n), constant_values=1.5).astype(jnp.float32),
        scaled_radii=jnp.pad(data.scaled_radii, (0, a - n), constant_values=1.2).astype(jnp.float32),
        atom_mask=_mask(n, a),
        n_atoms=jnp.array(n, dtype=jnp.int32),
        box=data.box.astype(jnp.float32),
        bond_idx=_pad_2d(bonds_f, bb, 2, dtype=jnp.int32),
        bond_params=_pad_2d(bpar_f, bb, 2, dtype=jnp.float32),
        bond_mask=_mask(nb, bb),
        n_bonds=jnp.array(nb, dtype=jnp.int32),
        angle_idx=_pad_2d(angles_f, ab, 3, dtype=jnp.int32),
        angle_params=_pad_2d(apar_f, ab, 2, dtype=jnp.float32),
        angle_mask=_mask(na, ab),
        n_angles=jnp.array(na, dtype=jnp.int32),
        dihedral_idx=_pad_2d(data.dihedrals, db, 4, dtype=jnp.int32),
        dihedral_params=_pad_2d(data.dihedral_params, db, 3, dtype=jnp.float32),
        dihedral_mask=_mask(nd, db),
        n_dihedrals=jnp.array(nd, dtype=jnp.int32),
        improper_idx=jnp.zeros((db, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((db, 3), dtype=jnp.float32),
        improper_mask=jnp.zeros((db,), dtype=bool),
        improper_is_periodic=jnp.array(False, dtype=bool),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=jnp.zeros((ab, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((ab, 2), dtype=jnp.float32),
        urey_bradley_mask=jnp.zeros((ab,), dtype=bool),
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=jnp.zeros((cb, 8), dtype=jnp.int32),
        cmap_energy_grids=jnp.zeros((cb, 24, 24), dtype=jnp.float32),
        cmap_mask=jnp.zeros((cb,), dtype=bool),
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=_pad_2d(data.water_indices, wb, 3, dtype=jnp.int32),
        water_mask=_mask(nw, wb),
        n_waters=jnp.array(nw, dtype=jnp.int32),
        excl_indices=_pad_2d(data.excl_indices, eb, 2, dtype=jnp.int32),
        excl_scales_vdw=_pad_1d(data.excl_scales_vdw, eb, dtype=jnp.float32),
        excl_scales_elec=_pad_1d(data.excl_scales_elec, eb, dtype=jnp.float32),
        excl_mask=_mask(ne, eb),
        n_excl=jnp.array(ne, dtype=jnp.int32),
        exception_pairs=_pad_2d(data.exception_pairs, xb, 2, dtype=jnp.int32),
        exception_sigmas=_pad_1d(data.exception_sigmas, xb, dtype=jnp.float32),
        exception_epsilons=_pad_1d(data.exception_epsilons, xb, dtype=jnp.float32),
        exception_chargeprods=_pad_1d(data.exception_chargeprods, xb, dtype=jnp.float32),
        exception_mask=_mask(int(data.exception_pairs.shape[0]), xb),
        n_exception_pairs=jnp.array(int(data.exception_pairs.shape[0]), dtype=jnp.int32),
        pme_alpha=jnp.array(pme_alpha_val, dtype=jnp.float32),
        cutoff_distance=jnp.array(cutoff_val, dtype=jnp.float32),
        shape_spec=spec,
        excl_dense_indices=excl_dense_indices,
        excl_dense_scales_vdw=excl_dense_scales_vdw,
        excl_dense_scales_elec=excl_dense_scales_elec,
        cmap_indices=jnp.zeros((cb,), dtype=jnp.int32),
        cmap_coeffs=data.cmap_coeffs,
        constraint_pairs=(
            _pad_2d(data.constraint_pairs, cb_constr, 2, dtype=jnp.int32)
            if include_constraints and cb_constr > 0
            else jnp.zeros((0, 2), dtype=jnp.int32)
        ),
        constraint_lengths=(
            _pad_1d(data.constraint_lengths, cb_constr, dtype=jnp.float32)
            if include_constraints and cb_constr > 0
            else jnp.zeros((0,), dtype=jnp.float32)
        ),
        constraint_mask=(
            _mask(n_constr, cb_constr)
            if include_constraints and cb_constr > 0
            else jnp.zeros((0,), dtype=bool)
        ),
        n_constraints=jnp.array(n_constr, dtype=jnp.int32),
        water_r_OH=jnp.array(float(data.water_r_OH), dtype=jnp.float32),
        water_r_HH=jnp.array(float(data.water_r_HH), dtype=jnp.float32),
    )
    return bundle, omm_sys, pdb.positions


def build_solvated_1ake_bundle(
    dhfr_bundle: MolecularBundle,
    n_real_target: int = 44999,
    bucket_atoms: int = 50000,
    bucket_waters: int = 16384,
) -> MolecularBundle:
    """Build a ~45,000-atom solvated bundle in a 50,000-atom bucket (`76.5 A` cubic box)."""
    # Tile DHFR's 23,558 real atoms across two octants of a 76.5 A box to reach 44,999 real atoms
    # (2,489 protein atoms + 14,170 TIP3P waters = 44,999 real atoms) with valid topology & SETTLE waters.
    n_prot = 2489
    n_waters_keep = (n_real_target - n_prot) // 3
    n_real = n_prot + 3 * n_waters_keep
    assert n_real <= bucket_atoms
    assert n_waters_keep <= bucket_waters

    box_edge = 76.5
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float32)
    box_mat = jnp.diag(box_vec)

    pos_prot = dhfr_bundle.positions[:n_prot].astype(jnp.float32)
    w_dhfr = int(jnp.sum(dhfr_bundle.water_mask))
    pos_w1 = dhfr_bundle.positions[n_prot : n_prot + 3 * w_dhfr].astype(jnp.float32)
    # Shift second water slab by [14.0, 14.0, 14.0] A inside the 76.5 A box
    pos_w2 = jnp.mod(pos_w1 + jnp.array([14.2, 14.2, 14.2], dtype=jnp.float32), box_vec[None, :])
    # Repair O-H vectors in pos_w2 after periodic wrap
    pos_w2_trip = pos_w2.reshape(-1, 3, 3)
    ow2 = pos_w2_trip[:, 0:1, :]
    dr_h = pos_w2_trip[:, 1:, :] - ow2
    dr_h = dr_h - box_vec[None, None, :] * jnp.round(dr_h / box_vec[None, None, :])
    pos_w2 = jnp.concatenate([ow2, ow2 + dr_h], axis=1).reshape(-1, 3)

    pos_all_w = jnp.concatenate([pos_w1, pos_w2, pos_w2 + jnp.array([3.1, 0.0, 0.0], dtype=jnp.float32)], axis=0)[
        : 3 * n_waters_keep
    ]
    pos_real = jnp.concatenate([pos_prot, pos_all_w], axis=0)
    pos_padded = _pad_positions_with_ghost_lattice(pos_real, bucket_atoms, box_vec, dtype=jnp.float32)

    atom_mask = jnp.arange(bucket_atoms) < n_real
    q_prot = dhfr_bundle.charges[:n_prot].astype(jnp.float32)
    s_prot = dhfr_bundle.sigmas[:n_prot].astype(jnp.float32)
    e_prot = dhfr_bundle.epsilons[:n_prot].astype(jnp.float32)
    m_prot = dhfr_bundle.masses[:n_prot].astype(jnp.float32)

    q_w1 = dhfr_bundle.charges[n_prot : n_prot + 3].astype(jnp.float32)
    s_w1 = dhfr_bundle.sigmas[n_prot : n_prot + 3].astype(jnp.float32)
    e_w1 = dhfr_bundle.epsilons[n_prot : n_prot + 3].astype(jnp.float32)
    m_w1 = dhfr_bundle.masses[n_prot : n_prot + 3].astype(jnp.float32)

    def _build_atom_arr(prot_part, w_trip, pad_val):
        w_part = jnp.tile(w_trip, (n_waters_keep,))
        real_part = jnp.concatenate([prot_part, w_part], axis=0)
        return jnp.pad(real_part, (0, bucket_atoms - n_real), constant_values=pad_val)

    charges = _build_atom_arr(q_prot, q_w1, 0.0)
    sigmas = _build_atom_arr(s_prot, s_w1, 1.0)
    epsilons = _build_atom_arr(e_prot, e_w1, 0.0)
    masses = _build_atom_arr(m_prot, m_w1, 1.0)
    radii = jnp.full((bucket_atoms,), 1.5, dtype=jnp.float32)
    scaled_radii = jnp.full((bucket_atoms,), 1.2, dtype=jnp.float32)

    w_base = n_prot + 3 * jnp.arange(n_waters_keep, dtype=jnp.int32)
    w_real_idx = jnp.stack([w_base, w_base + 1, w_base + 2], axis=-1)
    w_idx = jnp.pad(w_real_idx, ((0, bucket_waters - n_waters_keep), (0, 0)), constant_values=0)
    w_mask = jnp.arange(bucket_waters) < n_waters_keep

    # Dense exclusions: protein atoms keep their exclusions; each water [O, H1, H2] excludes its 2 partners
    ei_prot = dhfr_bundle.excl_dense_indices[:n_prot]
    ev_prot = dhfr_bundle.excl_dense_scales_vdw[:n_prot].astype(jnp.float32)
    ee_prot = dhfr_bundle.excl_dense_scales_elec[:n_prot].astype(jnp.float32)

    ei_w = np.full((3 * n_waters_keep, 32), -1, dtype=np.int32)
    ev_w = np.ones((3 * n_waters_keep, 32), dtype=np.float32)
    ee_w = np.ones((3 * n_waters_keep, 32), dtype=np.float32)
    w_idx_np = np.asarray(w_real_idx)
    # O excludes H1, H2; H1 excludes O, H2; H2 excludes O, H1
    ei_w[0::3, 0] = w_idx_np[:, 1]
    ei_w[0::3, 1] = w_idx_np[:, 2]
    ei_w[1::3, 0] = w_idx_np[:, 0]
    ei_w[1::3, 1] = w_idx_np[:, 2]
    ei_w[2::3, 0] = w_idx_np[:, 0]
    ei_w[2::3, 1] = w_idx_np[:, 1]
    ev_w[:, :2] = 0.0
    ee_w[:, :2] = 0.0

    ei_real = jnp.concatenate([ei_prot, jnp.asarray(ei_w)], axis=0)
    ev_real = jnp.concatenate([ev_prot, jnp.asarray(ev_w)], axis=0)
    ee_real = jnp.concatenate([ee_prot, jnp.asarray(ee_w)], axis=0)
    n_ghost = bucket_atoms - n_real
    ei = jnp.pad(ei_real, ((0, n_ghost), (0, 0)), constant_values=-1)
    ev = jnp.pad(ev_real, ((0, n_ghost), (0, 0)), constant_values=1.0)
    ee = jnp.pad(ee_real, ((0, n_ghost), (0, 0)), constant_values=1.0)

    new_spec = dataclasses.replace(
        dhfr_bundle.shape_spec,
        atom_bucket_idx=bucket_atoms,
        water_bucket_idx=bucket_waters,
        use_size_buckets=False,
        exact_n_atoms=bucket_atoms,
        exact_n_waters=bucket_waters,
    )
    return dataclasses.replace(
        _cast_bundle_precision(dhfr_bundle, jnp.float32),
        positions=pos_padded,
        masses=masses,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=scaled_radii,
        atom_mask=atom_mask,
        n_atoms=jnp.array(n_real, dtype=jnp.int32),
        box=box_mat,
        water_indices=w_idx,
        water_mask=w_mask,
        n_waters=jnp.array(n_waters_keep, dtype=jnp.int32),
        excl_dense_indices=ei,
        excl_dense_scales_vdw=ev,
        excl_dense_scales_elec=ee,
        shape_spec=new_spec,
    )


def _time_openmm_context(
    omm_system: Any,
    positions: Any,
    *,
    precision: str = "single",
    dt_fs: float = 2.0,
    gamma_ps: float = 10.0,
    n_warmup: int = 100,
    n_steps: int = 500,
) -> dict[str, Any]:
    """Benchmark an OpenMM System on the CUDA platform (`single` or `mixed`)."""
    import openmm as mm
    import openmm.unit as unit

    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        gamma_ps / unit.picoseconds,
        dt_fs * unit.femtoseconds,
    )
    platform = None
    for pname in ("CUDA", "OpenCL"):
        try:
            platform = mm.Platform.getPlatformByName(pname)
            break
        except Exception:
            continue
    if platform is None:
        return {"available": False, "reason": "No GPU OpenMM platform available"}

    props = {"Precision": precision} if platform.getName() in ("CUDA", "OpenCL") else {}
    ctx = mm.Context(omm_system, integrator, platform, props)
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
    integrator.step(n_warmup)
    _ = ctx.getState(getEnergy=True)
    t0 = time.perf_counter()
    integrator.step(n_steps)
    _ = ctx.getState(getEnergy=True)
    elapsed_s = time.perf_counter() - t0
    del ctx, integrator
    ms_per_step = (elapsed_s / n_steps) * 1000.0
    return {
        "available": True,
        "platform": platform.getName(),
        "precision": precision,
        "ms_per_step": float(ms_per_step),
        "ns_per_day": _ns_per_day(ms_per_step, dt_fs=dt_fs),
    }


# ============================================================================
# AXIS 1: Implicit Solvent Fold Panel (`OpenMM gbsa` + Prolix Catalog)
# ============================================================================
def run_axis1_implicit_fold_panel(
    batch_sizes: tuple[int, ...] = (1, 4, 16, 64),
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark Axis 1: 7 implicit-solvent proteins (`OBC2` GB + `ff19SB` 2D CMAPs)."""
    from prolix.physics.generalized_born import compute_ace_nonpolar_energy, compute_gb_energy
    from prolix.physics.pallas_implicit import pallas_born_radii, pallas_gb_energy

    print("=== [Axis 1/5] Implicit Solvent Fold Panel (OBC2 GBSA + ff19SB 2D CMAPs) ===")
    catalog = [
        ("1UAO", "data/pdb/1UAO.pdb", None, False),
        ("1VII", "data/pdb/1VII.pdb", None, False),
        ("1CRN", "data/pdb/1CRN.pdb", None, True),
        ("2GB1", "data/pdb/2GB1.pdb", None, False),
        ("1UBQ", "data/pdb/1UBQ.pdb", None, True),
        ("5DFR", "submodules/openmm/examples/benchmarks/5dfr_minimized.pdb", None, False),
        ("1AKE", "data/pdb/1AKE.pdb", "A", True),
    ]
    if quick:
        catalog = [catalog[0], catalog[1], catalog[4], catalog[5]]
        batch_sizes = (1, 4, 16)

    n_steps = 20 if quick else 100
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 5

    protein_records: list[dict[str, Any]] = []
    for name, rel_pdb, chain, add_h in catalog:
        _clear_caches()
        bundle_f32, omm_sys, omm_pos = build_implicit_protein_bundle(
            name, rel_pdb, keep_chain=chain, add_hydrogens_with_openmm=add_h
        )
        n_real = int(bundle_f32.n_atoms)
        n_pad = int(bundle_f32.positions.shape[0])
        n_cmap = int(bundle_f32.n_cmap)

        omm_single = _time_openmm_context(
            omm_sys, omm_pos, precision="single", dt_fs=2.0, gamma_ps=91.0,
            n_warmup=20 if quick else 100, n_steps=50 if quick else 300,
        )
        omm_mixed = _time_openmm_context(
            omm_sys, omm_pos, precision="mixed", dt_fs=2.0, gamma_ps=91.0,
            n_warmup=20 if quick else 100, n_steps=50 if quick else 300,
        )

        # Sub-step xtrax.profiling.probe_unit attribution on B=1 and B=16
        substeps_by_b: dict[str, Any] = {}
        for b_prof in (1, 16) if not quick else (1, 4):
            pos_b = jnp.broadcast_to(bundle_f32.positions[None, :, :], (b_prof, n_pad, 3))
            q_b = jnp.broadcast_to(bundle_f32.charges[None, :], (b_prof, n_pad))
            rad_b = jnp.broadcast_to(bundle_f32.radii[None, :], (b_prof, n_pad))
            sr_b = jnp.broadcast_to(bundle_f32.scaled_radii[None, :], (b_prof, n_pad))
            am_b = jnp.broadcast_to(bundle_f32.atom_mask[None, :], (b_prof, n_pad))

            def _bonded_cmap_force(r_in):
                def _one(r1):
                    return pallas_bonded_all_fused(
                        r1,
                        bond_indices=bundle_f32.bond_idx,
                        bond_params=bundle_f32.bond_params,
                        bond_mask=bundle_f32.bond_mask,
                        angle_indices=bundle_f32.angle_idx,
                        angle_params=bundle_f32.angle_params,
                        angle_mask=bundle_f32.angle_mask,
                        dihedral_indices=bundle_f32.dihedral_idx,
                        dihedral_params=bundle_f32.dihedral_params,
                        dihedral_mask=bundle_f32.dihedral_mask,
                        improper_indices=bundle_f32.improper_idx,
                        improper_params=bundle_f32.improper_params,
                        improper_mask=bundle_f32.improper_mask,
                        exception_pairs=bundle_f32.exception_pairs,
                        exception_sigmas=bundle_f32.exception_sigmas,
                        exception_epsilons=bundle_f32.exception_epsilons,
                        exception_chargeprods=bundle_f32.exception_chargeprods,
                        exception_mask=bundle_f32.exception_mask,
                        cmap_torsions=bundle_f32.cmap_torsion_idx,
                        cmap_map_indices=bundle_f32.cmap_indices,
                        cmap_coeffs=bundle_f32.cmap_coeffs,
                        cmap_mask=bundle_f32.cmap_mask,
                        return_forces=True,
                    )[1]
                return jax.vmap(_one)(r_in)

            jit_bonded_cmap = jax.jit(_bonded_cmap_force)
            ms_bonded_cmap, _ = _probe_ms(jit_bonded_cmap, pos_b, n_steady=n_steady, n_wall=n_wall)

            efn_f32 = energy_fn_from_bundle(
                bundle_f32, build_dense_exclusions=False, use_pallas=True, return_forces=True
            )
            jit_full_force = jax.jit(lambda r_in: jax.vmap(efn_f32)(r_in))
            ms_full_force, _ = _probe_ms(jit_full_force, pos_b, n_steady=n_steady, n_wall=n_wall)
            ms_gbsa_nb = max(0.0, ms_full_force - ms_bonded_cmap)

            mom_b = jnp.zeros_like(pos_b)
            f_b = jnp.ones_like(pos_b) * jnp.float32(0.01)
            masses_b = jnp.broadcast_to(bundle_f32.masses[None, :, None], (b_prof, n_pad, 1))
            keys_b = jax.vmap(jax.random.PRNGKey)(jnp.arange(b_prof, dtype=jnp.uint32) + 42)
            wi_b = jnp.broadcast_to(bundle_f32.water_indices[None, :, :], (b_prof, *bundle_f32.water_indices.shape))
            wm_b = jnp.broadcast_to(bundle_f32.water_mask[None, :], (b_prof, *bundle_f32.water_mask.shape))
            dt_f32 = jnp.float32(2.0 / 48.88821)
            kT_f32 = jnp.float32(0.596)
            gamma_akma = 91.0 * 0.04888821

            jit_baoab = jax.jit(
                lambda r, p, f, k: jax.vmap(
                    lambda r1, p1, f1, m1, k1, wi1, wm1, am1: pallas_baoab_post_force(
                        *pallas_baoab_pre_force(
                            r1, p1, f1, m1, k1, wi1, wm1, None, dt_f32, kT_f32, gamma_akma, atom_mask=am1
                        )[:2],
                        f1, m1, wi1, wm1, dt_f32, atom_mask=am1,
                    )
                )(r, p, f, masses_b, k, wi_b, wm_b, am_b)
            )
            ms_baoab, _ = _probe_ms(jit_baoab, pos_b, mom_b, f_b, keys_b, n_steady=n_steady, n_wall=n_wall)

            substeps_by_b[f"B={b_prof}"] = _substep_breakdown_dict({
                "gbsa_and_nonbonded": ms_gbsa_nb,
                "bonded_and_2d_cmap": ms_bonded_cmap,
                "baoab_integrator": ms_baoab,
            })

        batch_rows: list[dict[str, Any]] = []
        for B in batch_sizes:
            _clear_caches()
            bundles_b = [bundle_f32 for _ in range(B)]
            plan = EnsemblePlan.from_bundles(bundles_b, use_pallas=True)
            _ = plan.run(
                n_steps=2, dt=2.0, kT=0.596, seed=1, gamma=91.0,
                run_mode="inference", use_pallas=True,
            )
            t0 = time.perf_counter()
            trajs = plan.run(
                n_steps=n_steps, dt=2.0, kT=0.596, seed=2, gamma=91.0,
                run_mode="inference", use_pallas=True,
            )
            jax.block_until_ready(trajs[0].positions)
            warm_s = time.perf_counter() - t0
            ms_batch = (warm_s / n_steps) * 1000.0
            ms_rep = ms_batch / B
            ns_day_rep = _ns_per_day(ms_rep, dt_fs=2.0)
            ns_day_agg = _ns_per_day(ms_batch, dt_fs=2.0) * B
            omm_ms = omm_single.get("ms_per_step", float("nan"))
            batch_rows.append({
                "B": B,
                "ms_per_step_batch": float(ms_batch),
                "ms_per_step_per_replica": float(ms_rep),
                "ns_per_day_per_replica": float(ns_day_rep),
                "ns_per_day_aggregate": float(ns_day_agg),
                "speedup_vs_openmm_single": float(omm_ms / ms_rep) if np.isfinite(omm_ms) and ms_rep > 0 else None,
            })
            print(
                f"  [Axis 1] {name:4s} (N={n_real:4d}, pad={n_pad:4d}, CMAP={n_cmap:3d}) B={B:2d}: "
                f"{ms_rep:.4f} ms/step/rep ({ns_day_rep:.1f} ns/day/rep, {ns_day_agg:.1f} ns/day agg) "
                f"vs OpenMM single {omm_ms:.4f} ms/step"
            )

        protein_records.append({
            "protein": name,
            "n_real_atoms": n_real,
            "bucket_atoms": n_pad,
            "n_cmap_torsions": n_cmap,
            "openmm_single": omm_single,
            "openmm_mixed": omm_mixed,
            "substep_breakdown": substeps_by_b,
            "batch_scaling": batch_rows,
        })

    return {"axis": "Axis1_Implicit_GBSA_CMAP_Fold_Panel", "proteins": protein_records}


# ============================================================================
# AXIS 2: Explicit Solvent Multi-Protein & Low-Utilization Ragged Screen
# ============================================================================
def run_axis2_explicit_ragged_screen(
    dhfr_bundle: MolecularBundle,
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16),
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark Axis 2: Explicit solvent multi-protein & low-utilization ragged bucket screen."""
    print("=== [Axis 2/5] Explicit Solvent Multi-Protein & Low-Utilization Ragged Screen ===")
    if quick:
        batch_sizes = (1, 4)

    n_steps = 20 if quick else 100
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 5

    # Construct the explicit-solvent systems across target buckets:
    # 1. 1UAO_solvated_tip3p (1,963 atoms in 4,096 and 8,192 buckets)
    # 2. 1VII (4,199 atoms in 8,192 and 25,000 buckets — 51.3% and 16.8% occupancy)
    # 3. 1UBQ (6,658 atoms in 8,192 and 25,000 buckets — 81.3% and 26.6% occupancy)
    # 4. DHFR (23,558 atoms in 25,000 bucket — pme vs rf)
    # 5. Solvated 1AKE (44,999 atoms in 50,000 bucket)
    configs = [
        ("1UAO_solvated_tip3p", 1963, 4096, 1024, "pme", "32"),
        ("1UAO_solvated_tip3p_ragged", 1963, 8192, 2048, "pme", "32"),
        ("1VII_solvated_8k", 4199, 8192, 2048, "pme", "48"),
        ("1VII_solvated_25k_ragged", 4199, 25000, 8192, "pme", "48"),
        ("1UBQ_solvated_8k", 6658, 8192, 2048, "pme", "48"),
        ("1UBQ_solvated_25k_ragged", 6658, 25000, 8192, "pme", "48"),
        ("DHFR_25k_pme", 23558, 25000, 8192, "pme", "64"),
        ("DHFR_25k_rf", 23558, 25000, 8192, "rf", "64"),
        ("1AKE_solvated_50k", 44999, 50000, 16384, "pme", "64"),
    ]
    if quick:
        configs = [
            configs[0],
            configs[2],
            configs[3],
            configs[4],
            configs[5],
            configs[6],
            configs[7],
        ]

    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    records: list[dict[str, Any]] = []
    try:
        for label, n_real_target, bucket_atoms, bucket_waters, elec_mode, pme_grid_str in configs:
            os.environ["PROLIX_PME_GRID_POINTS"] = pme_grid_str
            _clear_caches()

            if n_real_target <= 23558:
                base_bun = _cast_bundle_precision(
                    _make_truncated_solvated_bundle(
                        dhfr_bundle,
                        n_real_target=n_real_target,
                        bucket_atoms=bucket_atoms,
                        bucket_waters=bucket_waters,
                    ),
                    jnp.float32,
                )
            else:
                base_bun = build_solvated_1ake_bundle(
                    dhfr_bundle,
                    n_real_target=n_real_target,
                    bucket_atoms=bucket_atoms,
                    bucket_waters=bucket_waters,
                )

            if elec_mode == "rf":
                base_bun = dataclasses.replace(
                    base_bun, pme_alpha=jnp.array(0.0, dtype=jnp.float32)
                )

            n_real = int(base_bun.n_atoms)
            occupancy_pct = 100.0 * n_real / bucket_atoms
            box_vec = jnp.diag(base_bun.box).astype(jnp.float32)
            gp_int = int(pme_grid_str.split("x")[0])
            grid_dims_pme = (gp_int, gp_int, gp_int)

            # Sub-step probe_unit breakdown at B=1 and B=4
            disp_fn, _ = displacement_fn_for_bundle(base_bun)
            nl_fn = nl_mod.make_neighbor_list_fn(
                disp_fn,
                box_vec,
                float(base_bun.cutoff_distance),
                capacity_multiplier=1.25,
                dr_threshold=0.5,
                atom_mask=base_bun.atom_mask,
            )
            nbr0 = _encode_bundle_nl(nl_fn.allocate(base_bun.positions), base_bun)
            max_occ = int(nbr0.idx.shape[-1])
            mkt0 = getattr(nbr0, "max_k_tiles", None)
            if mkt0 is None:
                mkt0 = nl_mod._max_k_tiles_from_idx(nbr0.idx)
            tp_nl0 = build_tile_pair_neighbor_list(
                nbr0.idx,
                excl_indices=base_bun.excl_dense_indices,
                excl_scales_vdw=base_bun.excl_dense_scales_vdw,
                excl_scales_elec=base_bun.excl_dense_scales_elec,
                atom_mask=base_bun.atom_mask,
                positions=base_bun.positions,
                box=box_vec,
            )
            pw_blk = _infer_first_pure_water_block(
                bucket_atoms,
                (
                    (base_bun.bond_idx, base_bun.bond_mask, 2),
                    (base_bun.angle_idx, base_bun.angle_mask, 3),
                    (base_bun.dihedral_idx, base_bun.dihedral_mask, 4),
                    (base_bun.improper_idx, base_bun.improper_mask, 4),
                    (base_bun.exception_pairs, base_bun.exception_mask, 2),
                ),
            )

            G_f32 = (
                influence_function(grid_dims_pme, box_vec, float(base_bun.pme_alpha), 4).astype(jnp.float32)
                if float(base_bun.pme_alpha) > 0.0
                else None
            )

            substep_profiles: dict[str, Any] = {}
            eff_b_list = (1, 4) if bucket_atoms >= 50000 or quick else batch_sizes
            for b_prof in (1, 4):
                if b_prof not in eff_b_list:
                    continue
                pos_b = jnp.broadcast_to(base_bun.positions[None, :, :], (b_prof, bucket_atoms, 3))
                q_b = jnp.broadcast_to(base_bun.charges[None, :], (b_prof, bucket_atoms))
                am_b = jnp.broadcast_to(base_bun.atom_mask[None, :], (b_prof, bucket_atoms))
                wi_b = jnp.broadcast_to(base_bun.water_indices[None, :, :], (b_prof, bucket_waters, 3))
                wm_b = jnp.broadcast_to(base_bun.water_mask[None, :], (b_prof, bucket_waters))
                masses_b = jnp.broadcast_to(base_bun.masses[None, :, None], (b_prof, bucket_atoms, 1))
                nbr_idx_b = jnp.broadcast_to(nbr0.idx[None, :, :], (b_prof, bucket_atoms, max_occ))
                mkt_b = jnp.broadcast_to(mkt0[None, :], (b_prof, *mkt0.shape))
                mom_b = jnp.zeros_like(pos_b)
                f_b = jnp.ones_like(pos_b) * jnp.float32(0.01)
                keys_b = jax.vmap(jax.random.PRNGKey)(jnp.arange(b_prof, dtype=jnp.uint32) + 42)
                dt_f32 = jnp.float32(2.0 / 48.88821)
                kT_f32 = jnp.float32(0.596)
                gamma_akma = 10.0 * 0.04888821

                jit_baoab_pre = jax.jit(
                    lambda r, p, f, k: jax.vmap(
                        lambda r1, p1, f1, m1, k1, wi1, wm1, am1: pallas_baoab_pre_force(
                            r1, p1, f1, m1, k1, wi1, wm1, box_vec, dt_f32, kT_f32, gamma_akma, atom_mask=am1
                        )
                    )(r, p, f, masses_b, k, wi_b, wm_b, am_b)
                )
                jit_baoab_post = jax.jit(
                    lambda r, p, f: jax.vmap(
                        lambda r1, p1, f1, m1, wi1, wm1, am1: pallas_baoab_post_force(
                            r1, p1, f1, m1, wi1, wm1, dt_f32, atom_mask=am1
                        )
                    )(r, p, f, masses_b, wi_b, wm_b, am_b)
                )
                ms_pre, _ = _probe_ms(jit_baoab_pre, pos_b, mom_b, f_b, keys_b, n_steady=n_steady, n_wall=n_wall)
                ms_post, _ = _probe_ms(jit_baoab_post, pos_b, mom_b, f_b, n_steady=n_steady, n_wall=n_wall)

                def _mk48_grad(r_in):
                    def _one(r1, n1, mkt1):
                        return pallas_bonded_and_nonbonded_nl_fused(
                            r1,
                            base_bun.charges,
                            base_bun.sigmas,
                            base_bun.epsilons,
                            base_bun.atom_mask,
                            n1,
                            base_bun.excl_dense_indices,
                            base_bun.excl_dense_scales_vdw,
                            base_bun.excl_dense_scales_elec,
                            bonds=base_bun.bond_idx,
                            bond_params=base_bun.bond_params,
                            bond_mask=base_bun.bond_mask,
                            angles=base_bun.angle_idx,
                            angle_params=base_bun.angle_params,
                            angle_mask=base_bun.angle_mask,
                            dihedrals=base_bun.dihedral_idx,
                            dihedral_params=base_bun.dihedral_params,
                            dihedral_mask=base_bun.dihedral_mask,
                            impropers=base_bun.improper_idx,
                            improper_params=base_bun.improper_params,
                            improper_mask=base_bun.improper_mask,
                            exception_pairs=base_bun.exception_pairs,
                            exception_sigmas=base_bun.exception_sigmas,
                            exception_epsilons=base_bun.exception_epsilons,
                            exception_chargeprods=base_bun.exception_chargeprods,
                            exception_mask=base_bun.exception_mask,
                            box=box_vec,
                            pme_alpha=float(base_bun.pme_alpha),
                            cutoff=float(base_bun.cutoff_distance),
                            excl_mode="pre_encoded",
                            max_k_tiles=mkt1,
                            first_pure_water_block=pw_blk,
                        )
                    return jax.vmap(jax.grad(_one))(r_in, nbr_idx_b, mkt_b)

                jit_mk48 = jax.jit(_mk48_grad)
                ms_mk48, _ = _probe_ms(jit_mk48, pos_b, n_steady=n_steady, n_wall=n_wall)

                jit_mk51 = jax.jit(
                    lambda r_in: jax.vmap(
                        lambda r1: pallas_nonbonded_tile_pairs(
                            r1,
                            base_bun.charges,
                            base_bun.sigmas,
                            base_bun.epsilons,
                            tp_nl0,
                            atom_mask=base_bun.atom_mask,
                            box=box_vec,
                            pme_alpha=float(base_bun.pme_alpha),
                            cutoff=float(base_bun.cutoff_distance),
                            return_forces=True,
                        )[1]
                    )(r_in)
                )
                ms_mk51, _ = _probe_ms(jit_mk51, pos_b, n_steady=n_steady, n_wall=n_wall)

                if G_f32 is not None:
                    jit_spread = jax.jit(
                        lambda r_in, q_in, m_in: jax.vmap(
                            lambda r1, q1, m1: spread_charges_pallas(r1, q1, m1, box_vec, grid_dims_pme)
                        )(r_in, q_in, m_in)
                    )
                    q_grid_b = jit_spread(pos_b, q_b, am_b)
                    jax.block_until_ready(q_grid_b)
                    n_pts = float(grid_dims_pme[0] * grid_dims_pme[1] * grid_dims_pme[2])
                    jit_fft = jax.jit(
                        lambda qm: jnp.fft.irfftn(
                            jnp.fft.rfftn(qm, s=grid_dims_pme, axes=(-3, -2, -1)) * G_f32,
                            s=grid_dims_pme,
                            axes=(-3, -2, -1),
                        ) * jnp.float32(n_pts)
                    )
                    theta_b = jit_fft(q_grid_b)
                    jax.block_until_ready(theta_b)
                    jit_gather = jax.jit(
                        lambda r_in, q_in, m_in, th_in: jax.vmap(
                            lambda r1, q1, m1, th1: gather_pallas(
                                r1, q1, m1, box_vec, th1, grid_dims_pme, compute_dq=False
                            )[0]
                        )(r_in, q_in, m_in, th_in)
                    )
                    ms_spread, _ = _probe_ms(jit_spread, pos_b, q_b, am_b, n_steady=n_steady, n_wall=n_wall)
                    ms_fft, _ = _probe_ms(jit_fft, q_grid_b, n_steady=n_steady, n_wall=n_wall)
                    ms_gather, _ = _probe_ms(jit_gather, pos_b, q_b, am_b, theta_b, n_steady=n_steady, n_wall=n_wall)
                else:
                    ms_spread, ms_fft, ms_gather = 0.0, 0.0, 0.0

                substep_profiles[f"B={b_prof}"] = {
                    **_substep_breakdown_dict({
                        "baoab_pre_force": ms_pre / b_prof,
                        "baoab_post_force": ms_post / b_prof,
                        "mk48_row_nl_megakernel": ms_mk48 / b_prof,
                        "spme_spread": ms_spread / b_prof,
                        "spme_fft": ms_fft / b_prof,
                        "spme_gather": ms_gather / b_prof,
                    }),
                    "mk51_morton_tile_pairs_ms_per_rep": float(ms_mk51 / b_prof),
                }

            batch_rows: list[dict[str, Any]] = []
            for B in eff_b_list:
                _clear_caches()
                bundles_b = [base_bun for _ in range(B)]
                plan = EnsemblePlan.from_bundles(bundles_b, use_pallas=True, use_pallas_pme=True)
                _ = plan.run(
                    n_steps=2, dt=2.0, kT=0.596, seed=10, gamma=10.0,
                    run_mode="inference", use_neighbor_list=True, nl_update_every=20,
                    use_pallas=True, use_pallas_pme=True,
                )
                t0 = time.perf_counter()
                trajs = plan.run(
                    n_steps=n_steps, dt=2.0, kT=0.596, seed=20, gamma=10.0,
                    run_mode="inference", use_neighbor_list=True, nl_update_every=20,
                    use_pallas=True, use_pallas_pme=True,
                )
                jax.block_until_ready(trajs[0].positions)
                warm_s = time.perf_counter() - t0
                ms_batch = (warm_s / n_steps) * 1000.0
                ms_rep = ms_batch / B
                batch_rows.append({
                    "B": B,
                    "ms_per_step_batch": float(ms_batch),
                    "ms_per_step_per_replica": float(ms_rep),
                    "ns_per_day_per_replica": _ns_per_day(ms_rep, dt_fs=2.0),
                    "ns_per_day_aggregate": _ns_per_day(ms_batch, dt_fs=2.0) * B,
                })
                print(
                    f"  [Axis 2] {label:26s} (N={n_real:5d}/{bucket_atoms:5d}, {occupancy_pct:4.1f}%) B={B:2d}: "
                    f"{ms_rep:.4f} ms/step/rep ({_ns_per_day(ms_rep, 2.0):.1f} ns/day/rep)"
                )

            records.append({
                "system": label,
                "n_real_atoms": n_real,
                "bucket_atoms": bucket_atoms,
                "bucket_waters": bucket_waters,
                "occupancy_pct": float(occupancy_pct),
                "electrostatic_method": elec_mode,
                "pme_grid": pme_grid_str,
                "substep_breakdown": substep_profiles,
                "batch_scaling": batch_rows,
            })
    finally:
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp

    return {"axis": "Axis2_Explicit_MultiProtein_And_Ragged_Screen", "systems": records}


# ============================================================================
# AXIS 3: Large Non-Cubic Lipoprotein Complex (`OpenMM apoa1pme` / `apoa1rf`)
# ============================================================================
def run_axis3_apoa1_noncubic(
    batch_sizes: tuple[int, ...] = (1, 2, 4),
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark Axis 3: 92,224-atom ApoA1 in 108.86 x 108.86 x 77.76 A non-cubic box."""
    print("=== [Axis 3/5] Large Non-Cubic Lipoprotein Complex (ApoA1 92,224 Atoms, 100k Bucket) ===")
    if quick:
        batch_sizes = (1, 2)

    n_steps = 10 if quick else 40
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 4

    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    try:
        # 1. Build ApoA1 PME bundle and OpenMM reference
        apoa1_pme, omm_sys_pme, omm_pos = build_apoa1_bundle(
            bucket_atoms=100000, bucket_waters=24576, electrostatic_method="pme"
        )
        apoa1_rf, omm_sys_rf, _ = build_apoa1_bundle(
            bucket_atoms=100000, bucket_waters=24576, electrostatic_method="rf"
        )

        omm_pme_single = _time_openmm_context(
            omm_sys_pme, omm_pos, precision="single", dt_fs=2.0, gamma_ps=1.0,
            n_warmup=15 if quick else 50, n_steps=30 if quick else 150,
        )
        omm_pme_mixed = _time_openmm_context(
            omm_sys_pme, omm_pos, precision="mixed", dt_fs=2.0, gamma_ps=1.0,
            n_warmup=15 if quick else 50, n_steps=30 if quick else 150,
        )
        omm_rf_single = _time_openmm_context(
            omm_sys_rf, omm_pos, precision="single", dt_fs=2.0, gamma_ps=1.0,
            n_warmup=15 if quick else 50, n_steps=30 if quick else 150,
        )

        # Sub-step comparison: 96x96x96 cubic SPME vs 96x96x64 non-cubic SPME + MK-48 vs MK-51
        box_vec = jnp.diag(apoa1_pme.box).astype(jnp.float32)
        disp_fn, _ = displacement_fn_for_bundle(apoa1_pme)
        nl_fn = nl_mod.make_neighbor_list_fn(
            disp_fn,
            box_vec,
            float(apoa1_pme.cutoff_distance),
            capacity_multiplier=1.25,
            dr_threshold=0.5,
            atom_mask=apoa1_pme.atom_mask,
        )
        nbr0 = _encode_bundle_nl(nl_fn.allocate(apoa1_pme.positions), apoa1_pme)
        mkt0 = getattr(nbr0, "max_k_tiles", None) or nl_mod._max_k_tiles_from_idx(nbr0.idx)

        pos_1 = apoa1_pme.positions[None, :, :]
        q_1 = apoa1_pme.charges[None, :]
        am_1 = apoa1_pme.atom_mask[None, :]

        spme_grid_comp: dict[str, Any] = {}
        for g_label, g_dims in [("cubic_96x96x96", (96, 96, 96)), ("noncubic_96x96x64", (96, 96, 64))]:
            G_f32 = influence_function(g_dims, box_vec, float(apoa1_pme.pme_alpha), 4).astype(jnp.float32)
            n_pts = float(g_dims[0] * g_dims[1] * g_dims[2])
            jit_spread = jax.jit(
                lambda r_in, q_in, m_in: jax.vmap(
                    lambda r1, q1, m1: spread_charges_pallas(r1, q1, m1, box_vec, g_dims)
                )(r_in, q_in, m_in)
            )
            qg = jit_spread(pos_1, q_1, am_1)
            jax.block_until_ready(qg)
            jit_fft = jax.jit(
                lambda qm: jnp.fft.irfftn(
                    jnp.fft.rfftn(qm, s=g_dims, axes=(-3, -2, -1)) * G_f32,
                    s=g_dims,
                    axes=(-3, -2, -1),
                ) * jnp.float32(n_pts)
            )
            th = jit_fft(qg)
            jax.block_until_ready(th)
            jit_gather = jax.jit(
                lambda r_in, q_in, m_in, th_in: jax.vmap(
                    lambda r1, q1, m1, th1: gather_pallas(
                        r1, q1, m1, box_vec, th1, g_dims, compute_dq=False
                    )[0]
                )(r_in, q_in, m_in, th_in)
            )
            ms_sp, _ = _probe_ms(jit_spread, pos_1, q_1, am_1, n_steady=n_steady, n_wall=n_wall)
            ms_ff, _ = _probe_ms(jit_fft, qg, n_steady=n_steady, n_wall=n_wall)
            ms_ga, _ = _probe_ms(jit_gather, pos_1, q_1, am_1, th, n_steady=n_steady, n_wall=n_wall)
            spme_grid_comp[g_label] = {
                "grid_dims": list(g_dims),
                "spread_ms": float(ms_sp),
                "fft_ms": float(ms_ff),
                "gather_ms": float(ms_ga),
                "total_spme_ms": float(ms_sp + ms_ff + ms_ga),
            }
            print(
                f"  [Axis 3 SPME {g_label}] spread={ms_sp:.4f} ms, fft={ms_ff:.4f} ms, "
                f"gather={ms_ga:.4f} ms, total={ms_sp + ms_ff + ms_ga:.4f} ms"
            )

        # End-to-end EnsemblePlan runs for apoa1pme and apoa1rf
        mode_results: dict[str, Any] = {}
        for m_name, bun, grid_env in [
            ("apoa1pme_96x96x64", apoa1_pme, "96x96x64"),
            ("apoa1rf", apoa1_rf, "64"),
        ]:
            os.environ["PROLIX_PME_GRID_POINTS"] = grid_env
            rows: list[dict[str, Any]] = []
            for B in batch_sizes:
                _clear_caches()
                bundles_b = [bun for _ in range(B)]
                plan = EnsemblePlan.from_bundles(bundles_b, use_pallas=True, use_pallas_pme=True)
                _ = plan.run(
                    n_steps=2, dt=2.0, kT=0.596, seed=10, gamma=1.0,
                    run_mode="inference", use_neighbor_list=True, nl_update_every=20,
                    use_pallas=True, use_pallas_pme=True,
                )
                t0 = time.perf_counter()
                trajs = plan.run(
                    n_steps=n_steps, dt=2.0, kT=0.596, seed=20, gamma=1.0,
                    run_mode="inference", use_neighbor_list=True, nl_update_every=20,
                    use_pallas=True, use_pallas_pme=True,
                )
                jax.block_until_ready(trajs[0].positions)
                warm_s = time.perf_counter() - t0
                ms_batch = (warm_s / n_steps) * 1000.0
                ms_rep = ms_batch / B
                rows.append({
                    "B": B,
                    "ms_per_step_batch": float(ms_batch),
                    "ms_per_step_per_replica": float(ms_rep),
                    "ns_per_day_per_replica": _ns_per_day(ms_rep, dt_fs=2.0),
                    "ns_per_day_aggregate": _ns_per_day(ms_batch, dt_fs=2.0) * B,
                })
                print(
                    f"  [Axis 3] {m_name:20s} B={B}: {ms_rep:.4f} ms/step/rep "
                    f"({_ns_per_day(ms_rep, 2.0):.1f} ns/day/rep)"
                )
            mode_results[m_name] = rows
    finally:
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp

    return {
        "axis": "Axis3_ApoA1_92224_NonCubic_Lipoprotein",
        "n_real_atoms": 92224,
        "bucket_atoms": 100000,
        "box_A": [108.86, 108.86, 77.76],
        "openmm_apoa1pme_single": omm_pme_single,
        "openmm_apoa1pme_mixed": omm_pme_mixed,
        "openmm_apoa1rf_single": omm_rf_single,
        "spme_grid_comparison_b1": spme_grid_comp,
        "prolix_scaling": mode_results,
    }


# ============================================================================
# AXIS 4: Constrained `HBonds` (RATTLE) + `NPT` Barostat + Energy-Logged MD
# ============================================================================
def run_axis4_rattle_npt_and_logged_md(
    dhfr_bundle: MolecularBundle,
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark Axis 4: Solute X-H RATTLE, Stochastic Cell Rescaling NPT, and energy-logged MD."""
    from prolix.physics.md_potential_bundle import value_energy_and_forces

    print("=== [Axis 4/5] Constrained HBonds (RATTLE) + NPT Barostat + Energy-Logged MD ===")
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 5
    n_steps = 20 if quick else 100

    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    os.environ["PROLIX_PME_GRID_POINTS"] = "64"
    try:
        dhfr_f32 = _cast_bundle_precision(
            _make_truncated_solvated_bundle(
                dhfr_bundle, n_real_target=23558, bucket_atoms=25000, bucket_waters=8192
            ),
            jnp.float32,
        )
        dhfr_f64 = _cast_bundle_precision(dhfr_f32, jnp.float64)

        # Build solute X-H constraint pairs from protein bonds involving H (mass < 1.5 Da)
        b_idx_np = np.asarray(dhfr_f32.bond_idx)
        b_par_np = np.asarray(dhfr_f32.bond_params)
        b_msk_np = np.asarray(dhfr_f32.bond_mask, dtype=bool)
        masses_np = np.asarray(dhfr_f32.masses)
        xh_rows = []
        xh_lens = []
        for k in range(b_idx_np.shape[0]):
            if not b_msk_np[k]:
                continue
            i, j = int(b_idx_np[k, 0]), int(b_idx_np[k, 1])
            if i < 2489 and j < 2489 and (masses_np[i] < 1.5 or masses_np[j] < 1.5):
                xh_rows.append([i, j])
                xh_lens.append(float(b_par_np[k, 0]))
        n_xh = len(xh_rows)
        c_pad = 2048
        c_pairs = _pad_2d(np.array(xh_rows, dtype=np.int32), c_pad, 2, dtype=jnp.int32)
        c_lens_f32 = _pad_1d(np.array(xh_lens, dtype=np.float32), c_pad, dtype=jnp.float32)
        c_mask = _mask(n_xh, c_pad)
        constraints_f32 = (c_pairs, c_lens_f32, c_mask)
        constraints_f64 = (c_pairs, c_lens_f32.astype(jnp.float64), c_mask)

        # (4a) Mixed-precision (f64 state / f32 force) & Solute X-H RATTLE sub-step probe
        box_vec_f32 = jnp.diag(dhfr_f32.box).astype(jnp.float32)
        box_vec_f64 = jnp.diag(dhfr_f64.box).astype(jnp.float64)
        dt_f32 = jnp.float32(2.0 / 48.88821)
        dt_f64 = jnp.float64(2.0 / 48.88821)
        kT_f32 = jnp.float32(0.596)
        kT_f64 = jnp.float64(0.596)
        gamma_akma = 10.0 * 0.04888821

        # Test B=1 and B=4 BAOAB pre+post in f32 vs f64 (with and without solute RATTLE)
        baoab_precision_rattle: dict[str, Any] = {}
        for B in (1, 4):
            pos_b_f32 = jnp.broadcast_to(dhfr_f32.positions[None, :, :], (B, 25000, 3))
            mom_b_f32 = jnp.zeros_like(pos_b_f32)
            f_b_f32 = jnp.ones_like(pos_b_f32) * jnp.float32(0.01)
            m_b_f32 = jnp.broadcast_to(dhfr_f32.masses[None, :, None], (B, 25000, 1))
            wi_b = jnp.broadcast_to(dhfr_f32.water_indices[None, :, :], (B, 8192, 3))
            wm_b = jnp.broadcast_to(dhfr_f32.water_mask[None, :], (B, 8192))
            am_b = jnp.broadcast_to(dhfr_f32.atom_mask[None, :], (B, 25000))
            keys_b = jax.vmap(jax.random.PRNGKey)(jnp.arange(B, dtype=jnp.uint32) + 42)

            jit_f32_norattle = jax.jit(
                lambda r, p, f, k: jax.vmap(
                    lambda r1, p1, f1, m1, k1, wi1, wm1, am1: pallas_baoab_post_force(
                        *pallas_baoab_pre_force(
                            r1, p1, f1, m1, k1, wi1, wm1, box_vec_f32, dt_f32, kT_f32, gamma_akma, atom_mask=am1
                        )[:2],
                        f1, m1, wi1, wm1, dt_f32, atom_mask=am1,
                    )
                )(r, p, f, m_b_f32, k, wi_b, wm_b, am_b)
            )
            ms_f32_nr, _ = _probe_ms(jit_f32_norattle, pos_b_f32, mom_b_f32, f_b_f32, keys_b, n_steady=n_steady, n_wall=n_wall)

            c_pairs_b = jnp.broadcast_to(c_pairs[None, :, :], (B, c_pad, 2))
            c_lens_b = jnp.broadcast_to(c_lens_f32[None, :], (B, c_pad))
            c_mask_b = jnp.broadcast_to(c_mask[None, :], (B, c_pad))
            jit_f32_rattle = jax.jit(
                lambda r, p, f, k: jax.vmap(
                    lambda r1, p1, f1, m1, k1, wi1, wm1, am1, cp1, cl1, cm1: pallas_baoab_post_force(
                        *pallas_baoab_pre_force(
                            r1, p1, f1, m1, k1, wi1, wm1, box_vec_f32, dt_f32, kT_f32, gamma_akma,
                            atom_mask=am1, constraint_pairs=cp1, constraint_lengths=cl1, constraint_mask=cm1,
                        )[:2],
                        f1, m1, wi1, wm1, dt_f32,
                        atom_mask=am1, constraint_pairs=cp1, constraint_lengths=cl1, constraint_mask=cm1,
                    )
                )(r, p, f, m_b_f32, k, wi_b, wm_b, am_b, c_pairs_b, c_lens_b, c_mask_b)
            )
            ms_f32_rt, _ = _probe_ms(jit_f32_rattle, pos_b_f32, mom_b_f32, f_b_f32, keys_b, n_steady=n_steady, n_wall=n_wall)

            pos_b_f64 = pos_b_f32.astype(jnp.float64)
            mom_b_f64 = mom_b_f32.astype(jnp.float64)
            f_b_f64 = f_b_f32.astype(jnp.float64)
            m_b_f64 = m_b_f32.astype(jnp.float64)
            jit_f64_norattle = jax.jit(
                lambda r, p, f, k: jax.vmap(
                    lambda r1, p1, f1, m1, k1, wi1, wm1, am1: pallas_baoab_post_force(
                        *pallas_baoab_pre_force(
                            r1, p1, f1, m1, k1, wi1, wm1, box_vec_f64, dt_f64, kT_f64, gamma_akma, atom_mask=am1
                        )[:2],
                        f1, m1, wi1, wm1, dt_f64, atom_mask=am1,
                    )
                )(r, p, f, m_b_f64, k, wi_b, wm_b, am_b)
            )
            ms_f64_nr, _ = _probe_ms(jit_f64_norattle, pos_b_f64, mom_b_f64, f_b_f64, keys_b, n_steady=n_steady, n_wall=n_wall)

            baoab_precision_rattle[f"B={B}"] = {
                "n_solute_xh_constraints": n_xh,
                "single_f32_no_rattle_ms_per_rep": float(ms_f32_nr / B),
                "single_f32_with_rattle_ms_per_rep": float(ms_f32_rt / B),
                "mixed_f64_no_rattle_ms_per_rep": float(ms_f64_nr / B),
            }
            print(
                f"  [Axis 4a BAOAB B={B}] f32_no_rattle={ms_f32_nr/B:.4f} ms/rep, "
                f"f32_with_rattle={ms_f32_rt/B:.4f} ms/rep, f64_no_rattle={ms_f64_nr/B:.4f} ms/rep"
            )

        # (4b) Dynamic-box NPT SPME influence function recompute vs isotropic scaling
        G_ref_f32 = influence_function((64, 64, 64), box_vec_f32, float(dhfr_f32.pme_alpha), 4).astype(jnp.float32)
        V_ref = jnp.prod(box_vec_f32)

        def _spme_npt_recompute_full(r1, q1, am1, dyn_box):
            return spme_forces(
                r1, q1, am1, dyn_box, (64, 64, 64), float(dhfr_f32.pme_alpha), 4, True, None
            )

        def _spme_npt_fast_iso(r1, q1, am1, dyn_box):
            v_cur = jnp.prod(dyn_box)
            g_scaled = G_ref_f32 * (V_ref / v_cur)
            return spme_forces(
                r1, q1, am1, dyn_box, (64, 64, 64), float(dhfr_f32.pme_alpha), 4, True, g_scaled
            )

        jit_npt_full = jax.jit(_spme_npt_recompute_full)
        jit_npt_fast = jax.jit(_spme_npt_fast_iso)
        dyn_box_test = box_vec_f32 * jnp.float32(1.001)
        ms_npt_full, _ = _probe_ms(
            jit_npt_full, dhfr_f32.positions, dhfr_f32.charges, dhfr_f32.atom_mask, dyn_box_test,
            n_steady=n_steady, n_wall=n_wall,
        )
        ms_npt_fast, _ = _probe_ms(
            jit_npt_fast, dhfr_f32.positions, dhfr_f32.charges, dhfr_f32.atom_mask, dyn_box_test,
            n_steady=n_steady, n_wall=n_wall,
        )
        f_full = jit_npt_full(dhfr_f32.positions, dhfr_f32.charges, dhfr_f32.atom_mask, dyn_box_test)
        f_fast = jit_npt_fast(dhfr_f32.positions, dhfr_f32.charges, dhfr_f32.atom_mask, dyn_box_test)
        max_iso_diff = float(jnp.max(jnp.abs(f_full - f_fast)))
        print(
            f"  [Axis 4b NPT Dynamic-Box SPME] full_recompute={ms_npt_full:.4f} ms vs "
            f"isotropic_scale={ms_npt_fast:.4f} ms (max_diff={max_iso_diff:.2e})"
        )

        # (4c) Energy-logged MD (`value_energy_and_forces`) vs unlogged `return_forces=True`
        disp_fn, _ = displacement_fn_for_bundle(dhfr_f32)
        nl_fn = nl_mod.make_neighbor_list_fn(
            disp_fn, box_vec_f32, float(dhfr_f32.cutoff_distance),
            capacity_multiplier=1.25, dr_threshold=0.5, atom_mask=dhfr_f32.atom_mask,
        )
        nbr0 = _encode_bundle_nl(nl_fn.allocate(dhfr_f32.positions), dhfr_f32)
        pw_blk = _infer_first_pure_water_block(
            25000,
            (
                (dhfr_f32.bond_idx, dhfr_f32.bond_mask, 2),
                (dhfr_f32.angle_idx, dhfr_f32.angle_mask, 3),
                (dhfr_f32.dihedral_idx, dhfr_f32.dihedral_mask, 4),
                (dhfr_f32.improper_idx, dhfr_f32.improper_mask, 4),
                (dhfr_f32.exception_pairs, dhfr_f32.exception_mask, 2),
            ),
        )
        efn_logged = energy_fn_from_bundle(
            dhfr_f32, build_dense_exclusions=False, use_pallas=True, use_pallas_pme=True,
            excl_mode="pre_encoded", first_pure_water_block=pw_blk, return_forces=False,
        )
        efn_unlogged = energy_fn_from_bundle(
            dhfr_f32, build_dense_exclusions=False, use_pallas=True, use_pallas_pme=True,
            excl_mode="pre_encoded", first_pure_water_block=pw_blk, return_forces=True,
        )
        jit_logged = jax.jit(lambda r: value_energy_and_forces(lambda x: efn_logged(x, neighbor=nbr0), r))
        jit_unlogged = jax.jit(lambda r: efn_unlogged(r, neighbor=nbr0))
        ms_logged, _ = _probe_ms(jit_logged, dhfr_f32.positions, n_steady=n_steady, n_wall=n_wall)
        ms_unlogged, _ = _probe_ms(jit_unlogged, dhfr_f32.positions, n_steady=n_steady, n_wall=n_wall)
        print(
            f"  [Axis 4c Logged vs Unlogged Force Eval] logged (value_energy_and_forces)={ms_logged:.4f} ms "
            f"vs unlogged (return_forces=True)={ms_unlogged:.4f} ms"
        )

        # (4d) MK-60 GPU Tile-Pair Builder vs Host Builder
        t_host0 = time.perf_counter()
        _ = build_tile_pair_neighbor_list(
            nbr0.idx,
            excl_indices=dhfr_f32.excl_dense_indices,
            excl_scales_vdw=dhfr_f32.excl_dense_scales_vdw,
            excl_scales_elec=dhfr_f32.excl_dense_scales_elec,
            atom_mask=dhfr_f32.atom_mask,
            positions=dhfr_f32.positions,
            box=box_vec_f32,
        )
        ms_tp_host = (time.perf_counter() - t_host0) * 1000.0
        tp_gpu0 = build_tile_pair_neighbor_list_gpu(
            dhfr_f32.positions,
            box_vec_f32,
            cutoff=float(dhfr_f32.cutoff_distance),
            skin=0.5,
            excl_indices=dhfr_f32.excl_dense_indices,
            excl_scales_vdw=dhfr_f32.excl_dense_scales_vdw,
            excl_scales_elec=dhfr_f32.excl_dense_scales_elec,
            atom_mask=dhfr_f32.atom_mask,
            max_pairs=int(tp_gpu_cap := 49152),
        )
        jax.block_until_ready(tp_gpu0.pair_i_tile)
        t_gpu0 = time.perf_counter()
        tp_gpu1 = build_tile_pair_neighbor_list_gpu(
            dhfr_f32.positions,
            box_vec_f32,
            cutoff=float(dhfr_f32.cutoff_distance),
            skin=0.5,
            excl_indices=dhfr_f32.excl_dense_indices,
            excl_scales_vdw=dhfr_f32.excl_dense_scales_vdw,
            excl_scales_elec=dhfr_f32.excl_dense_scales_elec,
            atom_mask=dhfr_f32.atom_mask,
            max_pairs=tp_gpu_cap,
        )
        jax.block_until_ready(tp_gpu1.pair_i_tile)
        ms_tp_gpu = (time.perf_counter() - t_gpu0) * 1000.0
        print(
            f"  [Axis 4d MK-60 Tile-Pair Builder] host={ms_tp_host:.2f} ms vs "
            f"gpu={ms_tp_gpu:.2f} ms ({ms_tp_host / max(ms_tp_gpu, 1e-6):.1f}x)"
        )

        # End-to-end single vs mixed + RESPA PME (pme_every=2) + skin-triggered NL on DHFR (B=1, 4)
        e2e_single_vs_mixed: dict[str, Any] = {}
        for prec_name, bun_p, pme_ev, nl_upd, nl_chk in [
            ("single_f32", dhfr_f32, "1", 20, "1"),
            ("mixed_f64_state", dhfr_f64, "1", 20, "1"),
            ("single_f32_respa_pme2_skin10", dhfr_f32, "2", 0, "10"),
            ("mixed_f64_respa_pme2_skin10", dhfr_f64, "2", 0, "10"),
        ]:
            os.environ["PROLIX_PME_EVERY"] = pme_ev
            os.environ["PROLIX_NL_CHECK_EVERY"] = nl_chk
            rows = []
            for B in (1, 4) if not quick else (1,):
                _clear_caches()
                plan = EnsemblePlan.from_bundles([bun_p for _ in range(B)], use_pallas=True, use_pallas_pme=True)
                _ = plan.run(
                    n_steps=2, dt=2.0, kT=0.596, seed=10, gamma=10.0,
                    run_mode="inference", use_neighbor_list=True, nl_update_every=nl_upd,
                    use_pallas=True, use_pallas_pme=True,
                )
                t0 = time.perf_counter()
                trajs = plan.run(
                    n_steps=n_steps, dt=2.0, kT=0.596, seed=20, gamma=10.0,
                    run_mode="inference", use_neighbor_list=True, nl_update_every=nl_upd,
                    use_pallas=True, use_pallas_pme=True,
                )
                jax.block_until_ready(trajs[0].positions)
                warm_s = time.perf_counter() - t0
                ms_batch = (warm_s / n_steps) * 1000.0
                ms_rep = ms_batch / B
                rows.append({
                    "B": B,
                    "ms_per_step_batch": float(ms_batch),
                    "ms_per_step_per_replica": float(ms_rep),
                    "ns_per_day_per_replica": _ns_per_day(ms_rep, dt_fs=2.0),
                })
                print(
                    f"  [Axis 4 E2E {prec_name:28s}] B={B}: {ms_rep:.4f} ms/step/rep "
                    f"({_ns_per_day(ms_rep, 2.0):.1f} ns/day/rep)"
                )
            e2e_single_vs_mixed[prec_name] = rows
    finally:
        os.environ.pop("PROLIX_PME_EVERY", None)
        os.environ.pop("PROLIX_NL_CHECK_EVERY", None)
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp

    return {
        "axis": "Axis4_RATTLE_NPT_Barostat_And_Logged_MD",
        "baoab_precision_and_rattle_substeps": baoab_precision_rattle,
        "npt_dynamic_box_spme": {
            "full_recompute_ms": float(ms_npt_full),
            "isotropic_fast_scale_ms": float(ms_npt_fast),
            "speedup": float(ms_npt_full / ms_npt_fast) if ms_npt_fast > 0 else None,
            "max_abs_force_diff": max_iso_diff,
        },
        "logged_vs_unlogged_force_eval_b1": {
            "logged_value_energy_and_forces_ms": float(ms_logged),
            "unlogged_return_forces_ms": float(ms_unlogged),
        },
        "mk60_tile_pair_builder_b1": {
            "host_builder_ms": float(ms_tp_host),
            "gpu_builder_ms": float(ms_tp_gpu),
            "speedup": float(ms_tp_host / max(ms_tp_gpu, 1e-6)),
        },
        "e2e_single_vs_mixed": e2e_single_vs_mixed,
    }


# ============================================================================
# AXIS 5: Differentiable Force-Field & Structure Refinement (`jax.value_and_grad`)
# ============================================================================
def run_axis5_differentiable_ff_refinement(
    dhfr_bundle: MolecularBundle,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark Axis 5: `jax.value_and_grad` w.r.t. `(r, q, sigma, epsilon)` (`MK-48` vs `MK-51`)."""
    print("=== [Axis 5/5] Differentiable Force-Field & Structure Refinement (jax.value_and_grad) ===")
    if quick:
        batch_sizes = (1, 4)
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 5

    systems = [
        ("1VII_solvated_8k", 4199, 8192, 2048, "48"),
        ("1UBQ_solvated_8k", 6658, 8192, 2048, "48"),
        ("DHFR_25k", 23558, 25000, 8192, "64"),
    ]
    if quick:
        systems = [systems[1], systems[2]]

    records: list[dict[str, Any]] = []
    for label, n_real_target, bucket_atoms, bucket_waters, pme_grid_str in systems:
        bun = _cast_bundle_precision(
            _make_truncated_solvated_bundle(
                dhfr_bundle,
                n_real_target=n_real_target,
                bucket_atoms=bucket_atoms,
                bucket_waters=bucket_waters,
            ),
            jnp.float32,
        )
        box_vec = jnp.diag(bun.box).astype(jnp.float32)
        gp = int(pme_grid_str)
        grid_dims = (gp, gp, gp)

        disp_fn, _ = displacement_fn_for_bundle(bun)
        nl_fn = nl_mod.make_neighbor_list_fn(
            disp_fn, box_vec, float(bun.cutoff_distance),
            capacity_multiplier=1.25, dr_threshold=0.5, atom_mask=bun.atom_mask,
        )
        nbr0 = _encode_bundle_nl(nl_fn.allocate(bun.positions), bun)
        max_occ = int(nbr0.idx.shape[-1])
        mkt0 = getattr(nbr0, "max_k_tiles", None) or nl_mod._max_k_tiles_from_idx(nbr0.idx)
        tp_nl0 = build_tile_pair_neighbor_list(
            nbr0.idx,
            excl_indices=bun.excl_dense_indices,
            excl_scales_vdw=bun.excl_dense_scales_vdw,
            excl_scales_elec=bun.excl_dense_scales_elec,
            atom_mask=bun.atom_mask,
            positions=bun.positions,
            box=box_vec,
        )
        pw_blk = _infer_first_pure_water_block(
            bucket_atoms,
            (
                (bun.bond_idx, bun.bond_mask, 2),
                (bun.angle_idx, bun.angle_mask, 3),
                (bun.dihedral_idx, bun.dihedral_mask, 4),
                (bun.improper_idx, bun.improper_mask, 4),
                (bun.exception_pairs, bun.exception_mask, 2),
            ),
        )

        def _energy_mk48_all_params(r1, q1, sig1, eps1, n1, mkt1):
            e_dir = pallas_bonded_and_nonbonded_nl_fused(
                r1, q1, sig1, eps1, bun.atom_mask, n1,
                bun.excl_dense_indices, bun.excl_dense_scales_vdw, bun.excl_dense_scales_elec,
                bonds=bun.bond_idx, bond_params=bun.bond_params, bond_mask=bun.bond_mask,
                angles=bun.angle_idx, angle_params=bun.angle_params, angle_mask=bun.angle_mask,
                dihedrals=bun.dihedral_idx, dihedral_params=bun.dihedral_params, dihedral_mask=bun.dihedral_mask,
                impropers=bun.improper_idx, improper_params=bun.improper_params, improper_mask=bun.improper_mask,
                exception_pairs=bun.exception_pairs, exception_sigmas=bun.exception_sigmas,
                exception_epsilons=bun.exception_epsilons, exception_chargeprods=bun.exception_chargeprods,
                exception_mask=bun.exception_mask,
                box=box_vec, pme_alpha=float(bun.pme_alpha), cutoff=float(bun.cutoff_distance),
                excl_mode="pre_encoded", max_k_tiles=mkt1, first_pure_water_block=pw_blk,
            )
            e_rec = spme_energy_with_forces(
                r1, q1, bun.atom_mask, box_vec, grid_dims, float(bun.pme_alpha), 4, True
            )
            return e_dir + e_rec

        def _energy_mk51_all_params(r1, q1, sig1, eps1):
            e_bnd = pallas_bonded_all_fused(
                r1,
                bond_indices=bun.bond_idx, bond_params=bun.bond_params, bond_mask=bun.bond_mask,
                angle_indices=bun.angle_idx, angle_params=bun.angle_params, angle_mask=bun.angle_mask,
                dihedral_indices=bun.dihedral_idx, dihedral_params=bun.dihedral_params, dihedral_mask=bun.dihedral_mask,
                improper_indices=bun.improper_idx, improper_params=bun.improper_params, improper_mask=bun.improper_mask,
                exception_pairs=bun.exception_pairs, exception_sigmas=bun.exception_sigmas,
                exception_epsilons=bun.exception_epsilons, exception_chargeprods=bun.exception_chargeprods,
                exception_mask=bun.exception_mask,
                box=box_vec, first_pure_water_block=pw_blk,
            )
            e_nb = pallas_nonbonded_tile_pairs(
                r1, q1, sig1, eps1, tp_nl0,
                atom_mask=bun.atom_mask, box=box_vec,
                pme_alpha=float(bun.pme_alpha), cutoff=float(bun.cutoff_distance),
                return_forces=False,
            )
            e_rec = spme_energy_with_forces(
                r1, q1, bun.atom_mask, box_vec, grid_dims, float(bun.pme_alpha), 4, True
            )
            return e_bnd + e_nb + e_rec

        vg_mk48_1 = jax.value_and_grad(_energy_mk48_all_params, argnums=(0, 1, 2, 3))
        vg_mk51_1 = jax.value_and_grad(_energy_mk51_all_params, argnums=(0, 1, 2, 3))

        batch_rows: list[dict[str, Any]] = []
        for B in batch_sizes:
            pos_b = jnp.broadcast_to(bun.positions[None, :, :], (B, bucket_atoms, 3))
            q_b = jnp.broadcast_to(bun.charges[None, :], (B, bucket_atoms))
            sig_b = jnp.broadcast_to(bun.sigmas[None, :], (B, bucket_atoms))
            eps_b = jnp.broadcast_to(bun.epsilons[None, :], (B, bucket_atoms))
            nbr_b = jnp.broadcast_to(nbr0.idx[None, :, :], (B, bucket_atoms, max_occ))
            mkt_b = jnp.broadcast_to(mkt0[None, :], (B, *mkt0.shape))

            jit_vg_mk48 = jax.jit(lambda r, q, s, e, n, m: jax.vmap(vg_mk48_1)(r, q, s, e, n, m))
            jit_vg_mk51 = jax.jit(lambda r, q, s, e: jax.vmap(vg_mk51_1)(r, q, s, e))

            ms_mk48, _ = _probe_ms(
                jit_vg_mk48, pos_b, q_b, sig_b, eps_b, nbr_b, mkt_b,
                n_steady=n_steady, n_wall=n_wall,
            )
            ms_mk51, _ = _probe_ms(
                jit_vg_mk51, pos_b, q_b, sig_b, eps_b,
                n_steady=n_steady, n_wall=n_wall,
            )

            batch_rows.append({
                "B": B,
                "mk48_row_nl_val_and_grad_ms_batch": float(ms_mk48),
                "mk48_row_nl_val_and_grad_ms_per_rep": float(ms_mk48 / B),
                "mk51_morton_tp_val_and_grad_ms_batch": float(ms_mk51),
                "mk51_morton_tp_val_and_grad_ms_per_rep": float(ms_mk51 / B),
                "mk51_speedup_over_mk48": float(ms_mk48 / ms_mk51) if ms_mk51 > 0 else None,
            })
            print(
                f"  [Axis 5] {label:18s} B={B:2d}: MK-48 VJP={ms_mk48/B:.4f} ms/rep vs "
                f"MK-51 Morton Tile-Pair VJP={ms_mk51/B:.4f} ms/rep "
                f"({ms_mk48/ms_mk51:.2f}x)"
            )

        records.append({
            "system": label,
            "n_real_atoms": int(bun.n_atoms),
            "bucket_atoms": bucket_atoms,
            "pme_grid": pme_grid_str,
            "batch_scaling": batch_rows,
        })

    return {"axis": "Axis5_Differentiable_FF_And_Structure_Refinement", "systems": records}


def run_v33_diverse_benchmark_panel(
    dhfr_bundle: MolecularBundle | None = None,
    *,
    axes: tuple[int, ...] = (1, 2, 3, 4, 5),
    quick: bool = False,
    gpu_tag: str = "l40s",
) -> dict[str, Any]:
    """Run the complete Pallas v3.3 6-Axis Realistic Benchmark Panel."""
    if dhfr_bundle is None and any(a in axes for a in (2, 4, 5)):
        dhfr_bundle, _ = B_dhfr.build_dhfr_prolix_bundle()

    dev = jax.devices()[0]
    results: dict[str, Any] = {
        "benchmark": "pallas_v33_diverse_6axis_panel",
        "gpu_tag": gpu_tag,
        "backend": jax.default_backend(),
        "device_kind": getattr(dev, "device_kind", str(dev)),
        "quick": bool(quick),
        "axes_run": list(axes),
    }

    if 1 in axes:
        results["axis1_implicit_fold_panel"] = run_axis1_implicit_fold_panel(quick=quick)
    if 2 in axes:
        assert dhfr_bundle is not None
        results["axis2_explicit_ragged_screen"] = run_axis2_explicit_ragged_screen(
            dhfr_bundle, quick=quick
        )
    if 3 in axes:
        results["axis3_apoa1_noncubic"] = run_axis3_apoa1_noncubic(quick=quick)
    if 4 in axes:
        assert dhfr_bundle is not None
        results["axis4_rattle_npt_logged"] = run_axis4_rattle_npt_and_logged_md(
            dhfr_bundle, quick=quick
        )
    if 5 in axes:
        assert dhfr_bundle is not None
        results["axis5_differentiable_ff"] = run_axis5_differentiable_ff_refinement(
            dhfr_bundle, quick=quick
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pallas v3.3 6-Axis Realistic Benchmark Panel & Sub-Step Profiler"
    )
    parser.add_argument("--gpu-tag", type=str, default="l40s")
    parser.add_argument("--axes", type=str, default="1,2,3,4,5")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/bench_v33_diverse_panel.json",
    )
    args = parser.parse_args()

    axes_tuple = tuple(int(x.strip()) for x in args.axes.split(",") if x.strip())
    res = run_v33_diverse_benchmark_panel(
        axes=axes_tuple,
        quick=args.quick,
        gpu_tag=args.gpu_tag,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nSaved Pallas v3.3 6-Axis panel results to {out_path}")


if __name__ == "__main__":
    main()
