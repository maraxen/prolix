#!/usr/bin/env python3
"""Diagnose inf energy from proxide parse_structure / OutputSpec path.

This script compares multiple loading strategies for a protein structure
and identifies the root cause of infinite initial energy:

1. New API (add_hydrogens=True, relax_hydrogens=False)  — suspected inf
2. New API (add_hydrogens=True, relax_hydrogens=True)   — potential fix
3. New API (add_hydrogens=False)                        — baseline (no H)
4. Old API (biotite/HYDRIDE)                            — known working

For each path, it:
  - Reports atom counts and coordinate statistics
  - Finds minimum interatomic distances (close contacts)
  - Decomposes energy into: bond, angle, dihedral, improper, LJ, elec, GB
  - If LJ is inf, identifies the offending atom pairs

Usage:
    uv run python prolix/scripts/debug/diagnose_inf_energy.py \
        --pdb prolix/data/pdb/1UAO.pdb
    uv run python prolix/scripts/debug/diagnose_inf_energy.py \
        --pdb projects/noised_cb/references/pdb/8T71.pdb
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose")

# Force field path (ff19SB is used by noised_cb trajectory generator)
FF_PATH = "proxide/src/proxide/assets/protein.ff19SB.xml"
FF14_PATH = "proxide/src/proxide/assets/protein.ff14SB.xml"


def find_close_contacts(coords: np.ndarray, threshold: float = 1.0):
    """Find atom pairs closer than threshold (Angstroms).

    Returns list of (i, j, distance) tuples, sorted by distance.
    Uses chunked computation to handle large systems.
    """
    n = coords.shape[0]
    contacts = []

    # Chunk to avoid memory explosion on large systems
    chunk_size = 500
    for start_i in range(0, n, chunk_size):
        end_i = min(start_i + chunk_size, n)
        for start_j in range(start_i, n, chunk_size):
            end_j = min(start_j + chunk_size, n)

            ci = coords[start_i:end_i]
            cj = coords[start_j:end_j]

            # Pairwise distances
            diff = ci[:, None, :] - cj[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))

            # Find close contacts (skip self)
            for ii in range(end_i - start_i):
                for jj in range(end_j - start_j):
                    abs_i = start_i + ii
                    abs_j = start_j + jj
                    if abs_i >= abs_j:
                        continue
                    if dist[ii, jj] < threshold:
                        contacts.append((abs_i, abs_j, float(dist[ii, jj])))

    contacts.sort(key=lambda x: x[2])
    return contacts


def compute_min_distances(coords: np.ndarray, top_k: int = 10):
    """Compute the top-K shortest atom-atom distances (excluding self)."""
    n = coords.shape[0]
    min_dists = []

    chunk_size = 1000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = coords[start:end]

        diff = chunk[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))

        # Mask self interactions
        for ii in range(end - start):
            dist[ii, start + ii] = 1e10  # mask self

        for ii in range(end - start):
            sorted_idx = np.argsort(dist[ii])
            for k in range(min(top_k, n)):
                j = sorted_idx[k]
                d = dist[ii, j]
                if d < 1e10:
                    min_dists.append((start + ii, int(j), float(d)))

    min_dists.sort(key=lambda x: x[2])
    return min_dists[:top_k]


def decompose_energy(protein, coords_flat, displacement_fn):
    """Decompose energy into individual terms for a Protein object."""
    from prolix.physics import bonded, generalized_born
    from prolix.physics import neighbor_list as nl
    from prolix.physics import system as physics_system

    r = jnp.array(coords_flat)

    # Build ExclusionSpec
    exclusion_spec = nl.ExclusionSpec.from_protein(protein)

    results = {}

    # Total energy
    energy_fn = physics_system.make_energy_fn(
        displacement_fn,
        protein,
        exclusion_spec=exclusion_spec,
        implicit_solvent=True,
        use_pbc=False,
    )
    try:
        e_total = float(energy_fn(r))
        results["total"] = e_total
    except Exception as e:
        results["total"] = f"ERROR: {e}"

    # Bond energy
    try:
        bonds = jnp.asarray(protein.bonds if protein.bonds is not None else jnp.zeros((0, 2), dtype=jnp.int32))
        bond_params = jnp.asarray(protein.bond_params if protein.bond_params is not None else jnp.zeros((0, 2)))
        bond_fn = bonded.make_bond_energy_fn(displacement_fn, bonds, bond_params)
        results["bond"] = float(bond_fn(r))
    except Exception as e:
        results["bond"] = f"ERROR: {e}"

    # Angle energy
    try:
        angles = jnp.asarray(protein.angles if protein.angles is not None else jnp.zeros((0, 3), dtype=jnp.int32))
        angle_params = jnp.asarray(protein.angle_params if protein.angle_params is not None else jnp.zeros((0, 2)))
        angle_fn = bonded.make_angle_energy_fn(displacement_fn, angles, angle_params)
        results["angle"] = float(angle_fn(r))
    except Exception as e:
        results["angle"] = f"ERROR: {e}"

    # Dihedral energy
    try:
        dihedrals = jnp.asarray(protein.proper_dihedrals if protein.proper_dihedrals is not None else jnp.zeros((0, 4), dtype=jnp.int32))
        dihedral_params = jnp.asarray(protein.dihedral_params if protein.dihedral_params is not None else jnp.zeros((0, 3)))
        dihe_fn = bonded.make_dihedral_energy_fn(displacement_fn, dihedrals, dihedral_params)
        results["dihedral"] = float(dihe_fn(r))
    except Exception as e:
        results["dihedral"] = f"ERROR: {e}"

    # Improper energy
    try:
        impropers = jnp.asarray(protein.impropers if protein.impropers is not None else jnp.zeros((0, 4), dtype=jnp.int32))
        improper_params = jnp.asarray(protein.improper_params if protein.improper_params is not None else jnp.zeros((0, 3)))
        imp_fn = bonded.make_dihedral_energy_fn(displacement_fn, impropers, improper_params)
        results["improper"] = float(imp_fn(r))
    except Exception as e:
        results["improper"] = f"ERROR: {e}"

    # LJ energy (N^2 dense, no exclusions for isolation)
    try:
        from jax_md import energy, space

        sigmas = jnp.asarray(protein.sigmas)
        epsilons = jnp.asarray(protein.epsilons)

        dr = space.map_product(displacement_fn)(r, r)
        dist = space.distance(jnp.asarray(dr))

        sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
        eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])

        e_lj_raw = energy.lennard_jones(jnp.asarray(dist), sig_ij, eps_ij)

        # Build proper scaling matrix from ExclusionSpec
        N = r.shape[0]
        mat_vdw = jnp.ones((N, N), dtype=jnp.float32)
        idx_self = jnp.arange(N)
        mat_vdw = mat_vdw.at[idx_self, idx_self].set(0.0)

        idx1213 = exclusion_spec.idx_12_13
        if idx1213.shape[0] > 0:
            mat_vdw = mat_vdw.at[idx1213[:, 0], idx1213[:, 1]].set(0.0)
            mat_vdw = mat_vdw.at[idx1213[:, 1], idx1213[:, 0]].set(0.0)

        idx14 = exclusion_spec.idx_14
        if idx14.shape[0] > 0:
            mat_vdw = mat_vdw.at[idx14[:, 0], idx14[:, 1]].set(exclusion_spec.scale_14_vdw)
            mat_vdw = mat_vdw.at[idx14[:, 1], idx14[:, 0]].set(exclusion_spec.scale_14_vdw)

        e_lj_scaled = e_lj_raw * mat_vdw
        results["lj"] = float(0.5 * jnp.sum(e_lj_scaled))

        # Also find per-pair LJ max contributions
        e_lj_pairs = e_lj_scaled

        # Find top offending pairs
        upper_tri = jnp.triu(e_lj_pairs, k=1)
        flat = upper_tri.flatten()
        top_k = min(10, flat.shape[0])
        top_indices = jnp.argsort(flat)[-top_k:][::-1]
        top_values = flat[top_indices]

        offenders = []
        for idx_flat, val in zip(np.array(top_indices), np.array(top_values)):
            i = idx_flat // N
            j = idx_flat % N
            d = float(dist[i, j])
            offenders.append((int(i), int(j), float(val), d))

        results["lj_top_offenders"] = offenders

    except Exception as e:
        results["lj"] = f"ERROR: {e}"

    # Electrostatics (direct Coulomb only, no GB)
    try:
        charges = jnp.asarray(protein.charges)
        q_ij = charges[:, None] * charges[None, :]
        dist_safe = dist + 1e-6
        COULOMB_CONST = 332.0637
        e_coul = COULOMB_CONST * (q_ij / dist_safe)

        # Apply scaling
        mat_elec = jnp.ones((N, N), dtype=jnp.float32)
        mat_elec = mat_elec.at[idx_self, idx_self].set(0.0)
        if idx1213.shape[0] > 0:
            mat_elec = mat_elec.at[idx1213[:, 0], idx1213[:, 1]].set(0.0)
            mat_elec = mat_elec.at[idx1213[:, 1], idx1213[:, 0]].set(0.0)
        if idx14.shape[0] > 0:
            c14 = protein.coulomb14scale if protein.coulomb14scale is not None else 0.83333333
            mat_elec = mat_elec.at[idx14[:, 0], idx14[:, 1]].set(c14)
            mat_elec = mat_elec.at[idx14[:, 1], idx14[:, 0]].set(c14)

        e_coul_scaled = e_coul * mat_elec
        results["elec_direct"] = float(0.5 * jnp.sum(e_coul_scaled))
    except Exception as e:
        results["elec_direct"] = f"ERROR: {e}"

    # GB energy
    try:
        radii_gb = protein.radii if protein.radii is not None else sigmas * 0.5
        e_gb, _ = generalized_born.compute_gb_energy(
            r,
            jnp.asarray(protein.charges),
            jnp.asarray(radii_gb),
            solvent_dielectric=78.5,
            solute_dielectric=1.0,
            dielectric_offset=0.09,
            mask=jnp.ones((N, N), dtype=jnp.float32),
            scaled_radii=jnp.asarray(protein.scaled_radii) if protein.scaled_radii is not None else None,
        )
        results["gb"] = float(e_gb)
    except Exception as e:
        results["gb"] = f"ERROR: {e}"

    return results


def load_new_api(pdb_path: str, add_hydrogens: bool, relax_hydrogens: bool = False, ff_path: str = FF_PATH):
    """Load with new proxide parse_structure API."""
    from proxide import CoordFormat, OutputSpec
    from proxide.io.parsing.backend import parse_structure

    kwargs = {
        "coord_format": CoordFormat.Full,
        "parameterize_md": True,
        "force_field": ff_path,
        "add_hydrogens": add_hydrogens,
        "remove_solvent": True,
    }
    if relax_hydrogens and add_hydrogens:
        kwargs["relax_hydrogens"] = True
        kwargs["relax_max_iterations"] = 100

    spec = OutputSpec(**kwargs)
    protein = parse_structure(pdb_path, spec)
    return protein


def load_old_api(pdb_path: str, ff_path: str = FF_PATH):
    """Load with old biotite/HYDRIDE API (the known-working path)."""
    try:
        from proxide.io.parsing import biotite as parsing_biotite
    except ImportError:
        logger.warning("Old biotite parsing not available, skipping old API path")
        return None

    try:
        atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
    except Exception as e:
        logger.warning("Old API load_structure_with_hydride failed: %s", e)
        return None

    # Need to convert to system_params and then to Protein via compat
    try:
        from prolix.compat import system_params_to_protein

        # Extract residue info and parameterize
        from proxide.io.parsing import biotite as pb

        residues = pb.extract_residues(atom_array)
        atom_names_per_res = pb.extract_atom_names(atom_array, residues)
        atom_counts = [len(names) for names in atom_names_per_res]
        res_names = [str(r.res_name) for r in residues]

        from proxide.io.parsing.backend import load_forcefield_rust

        ff = load_forcefield_rust(ff_path)

        # This is complex and may not work cleanly — log and skip if it fails
        logger.warning("Old API parameterization is complex — using reduced comparison")
        return None
    except Exception as e:
        logger.warning("Old API parameterization failed: %s", e)
        return None


def analyze_structure(label: str, protein, pdb_path: str):
    """Run full diagnostic analysis on a loaded structure."""
    from jax_md import space

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    if protein is None:
        print("  SKIPPED (load failed)")
        return

    coords = np.array(protein.coordinates)
    if coords.ndim == 3:
        # Atom37 format — flatten
        coords = coords.reshape(-1, 3)
    print(f"  Atoms:       {coords.shape[0]}")
    print(f"  Coord range: [{coords.min():.2f}, {coords.max():.2f}]")
    print(f"  Coord mean:  {coords.mean():.2f}")

    # Charges
    if protein.charges is not None:
        charges = np.array(protein.charges)
        print(f"  Charges:     min={charges.min():.4f}, max={charges.max():.4f}, sum={charges.sum():.4f}")
    else:
        print("  Charges:     None")

    # Sigmas / Epsilons
    if protein.sigmas is not None:
        sigmas = np.array(protein.sigmas)
        print(f"  Sigmas:      min={sigmas.min():.4f}, max={sigmas.max():.4f}")
    if protein.epsilons is not None:
        eps = np.array(protein.epsilons)
        print(f"  Epsilons:    min={eps.min():.6f}, max={eps.max():.6f}")

    # Bonds
    if protein.bonds is not None:
        n_bonds = len(protein.bonds)
        print(f"  Bonds:       {n_bonds}")
    else:
        print("  Bonds:       None")

    # Minimum distances
    print("\n  --- Minimum Interatomic Distances ---")
    top_dists = compute_min_distances(coords, top_k=10)
    for rank, (i, j, d) in enumerate(top_dists):
        flag = " *** CLASH ***" if d < 0.5 else (" (close)" if d < 1.0 else "")
        print(f"    [{rank+1}] atoms {i:5d} - {j:5d}: {d:.4f} Å{flag}")

    # Close contacts
    close = find_close_contacts(coords, threshold=0.5)
    if close:
        print(f"\n  !!! {len(close)} STERIC CLASHES (< 0.5 Å) !!!")
        for i, j, d in close[:20]:
            print(f"    atoms {i:5d} - {j:5d}: {d:.4f} Å")
    else:
        print("\n  No steric clashes (< 0.5 Å)")

    # Energy decomposition
    print("\n  --- Energy Decomposition (kcal/mol) ---")
    displacement_fn, _ = space.free()
    try:
        energies = decompose_energy(protein, coords, displacement_fn)
        for term, val in energies.items():
            if term == "lj_top_offenders":
                print(f"\n  --- Top LJ Offending Pairs ---")
                for i, j, e_val, d in val:
                    flag = " *** INF ***" if not np.isfinite(e_val) else ""
                    print(f"    atoms {i:5d} - {j:5d}: E_LJ={e_val:12.2f}, dist={d:.4f} Å{flag}")
            else:
                if isinstance(val, (int, float)):
                    finite_flag = "" if np.isfinite(val) else " *** NON-FINITE ***"
                    print(f"    {term:15s}: {val:15.2f}{finite_flag}")
                else:
                    print(f"    {term:15s}: {val}")
    except Exception as e:
        print(f"  Energy decomposition failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Diagnose inf energy from parse_structure path")
    parser.add_argument(
        "--pdb",
        type=str,
        default="prolix/data/pdb/1UAO.pdb",
        help="Path to PDB file",
    )
    parser.add_argument(
        "--ff",
        type=str,
        default=FF_PATH,
        help="Force field XML path",
    )
    parser.add_argument(
        "--skip-old-api",
        action="store_true",
        help="Skip old biotite/HYDRIDE API comparison",
    )
    args = parser.parse_args()

    pdb_path = args.pdb
    ff_path = args.ff

    if not Path(pdb_path).exists():
        print(f"ERROR: PDB file not found: {pdb_path}")
        sys.exit(1)
    if not Path(ff_path).exists():
        print(f"ERROR: Force field not found: {ff_path}")
        sys.exit(1)

    print(f"JAX devices: {jax.devices()}")
    print(f"PDB: {pdb_path}")
    print(f"Force Field: {ff_path}")

    # =========================================================================
    # Path 1: New API, add_hydrogens=True, relax_hydrogens=False (SUSPECTED INF)
    # =========================================================================
    t0 = time.time()
    try:
        protein_h_raw = load_new_api(pdb_path, add_hydrogens=True, relax_hydrogens=False, ff_path=ff_path)
    except Exception as e:
        logger.error("Path 1 (H raw) load failed: %s", e)
        protein_h_raw = None
    logger.info("Path 1 load: %.1fs", time.time() - t0)

    analyze_structure(
        "Path 1: New API (add_hydrogens=True, relax=False) — SUSPECTED INF",
        protein_h_raw,
        pdb_path,
    )

    # =========================================================================
    # Path 2: New API, add_hydrogens=True, relax_hydrogens=True (POTENTIAL FIX)
    # =========================================================================
    t0 = time.time()
    try:
        protein_h_relax = load_new_api(pdb_path, add_hydrogens=True, relax_hydrogens=True, ff_path=ff_path)
    except Exception as e:
        logger.error("Path 2 (H relaxed) load failed: %s", e)
        protein_h_relax = None
    logger.info("Path 2 load: %.1fs", time.time() - t0)

    analyze_structure(
        "Path 2: New API (add_hydrogens=True, relax=True) — POTENTIAL FIX",
        protein_h_relax,
        pdb_path,
    )

    # =========================================================================
    # Path 3: New API, add_hydrogens=False (BASELINE — no H)
    # =========================================================================
    t0 = time.time()
    try:
        protein_no_h = load_new_api(pdb_path, add_hydrogens=False, ff_path=ff_path)
    except Exception as e:
        logger.error("Path 3 (no H) load failed: %s", e)
        protein_no_h = None
    logger.info("Path 3 load: %.1fs", time.time() - t0)

    analyze_structure(
        "Path 3: New API (add_hydrogens=False) — BASELINE",
        protein_no_h,
        pdb_path,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")

    for label, p in [
        ("Path 1 (H raw)", protein_h_raw),
        ("Path 2 (H relaxed)", protein_h_relax),
        ("Path 3 (no H)", protein_no_h),
    ]:
        if p is None:
            print(f"  {label:25s}: SKIPPED")
            continue
        coords = np.array(p.coordinates)
        if coords.ndim == 3:
            coords = coords.reshape(-1, 3)
        n_atoms = coords.shape[0]

        close = find_close_contacts(coords, threshold=0.5)
        n_clashes = len(close)
        min_d = close[0][2] if close else compute_min_distances(coords, 1)[0][2]

        from jax_md import space
        displacement_fn, _ = space.free()
        try:
            from prolix.physics import neighbor_list as nl
            from prolix.physics import system as physics_system
            exclusion_spec = nl.ExclusionSpec.from_protein(p)
            energy_fn = physics_system.make_energy_fn(
                displacement_fn, p, exclusion_spec=exclusion_spec,
                implicit_solvent=True, use_pbc=False,
            )
            e = float(energy_fn(jnp.array(coords)))
            e_str = f"{e:.2f}" if np.isfinite(e) else "*** INF ***"
        except Exception:
            e_str = "ERROR"

        print(f"  {label:25s}: {n_atoms:5d} atoms, min_dist={min_d:.3f}Å, clashes={n_clashes:3d}, E={e_str}")

    print()


if __name__ == "__main__":
    main()
