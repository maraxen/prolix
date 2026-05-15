"""Diagnose the 7.88 kcal/mol torsion energy gap between JAX and OpenMM.

Strategy:
1. Load 1UAO through both proxide and OpenMM
2. Enumerate all PeriodicTorsionForce quads in OpenMM
3. Find quads NOT present in JAX (proper or improper), either forward or reversed
4. Compute individual energy contributions of missing quads
5. Report which quads account for the gap
"""
import os
import tempfile
import numpy as np
import jax.numpy as jnp
from openmm import app, unit
import openmm
from pdbfixer import PDBFixer
import proxide
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat, assign_mbondi2_radii, assign_obc2_scaling_factors

PDB_PATH = "data/pdb/1UAO.pdb"
FF_XML = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")


def fix_pdb(pdb_path):
    fixer = PDBFixer(filename=pdb_path)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return f.name


def dihedral_angle(r1, r2, r3, r4):
    """Compute dihedral angle (in radians) for atoms 1-2-3-4.
    The dihedral axis is the 2-3 bond (OpenMM convention: p1-p2-p3-p4, axis=p2-p3).
    """
    b0 = r2 - r1
    b1 = r3 - r2
    b2 = r4 - r3

    b1_norm = np.linalg.norm(b1) + 1e-12
    b1_unit = b1 / b1_norm

    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_unit, v), w)
    return np.arctan2(y, x)


def main():
    tmp_path = fix_pdb(PDB_PATH)

    # --- proxide parse ---
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        add_hydrogens=False,
        parameterize_md=True,
        force_field=FF_XML,
    )
    sys = parse_structure(tmp_path, spec=spec)
    pos_ang = np.array(sys.coordinates)  # already in Angstroms

    # Build sets of JAX quads (forward and reversed)
    def quad_set(idx_array):
        s = set()
        for row in idx_array:
            q = tuple(int(x) for x in row)
            s.add(q)
            s.add(q[::-1])
        return s

    jax_proper_quads = quad_set(sys.proper_dihedrals)
    jax_improper_quads = quad_set(sys.impropers)
    jax_all_quads = jax_proper_quads | jax_improper_quads

    print(f"JAX proper dihedrals:  {sys.proper_dihedrals.shape[0]}")
    print(f"JAX impropers:         {sys.impropers.shape[0]}")

    # --- OpenMM setup ---
    pdb_file = app.PDBFile(tmp_path)
    omm_ff = app.ForceField(FF_XML, "implicit/obc2.xml")
    omm_system = omm_ff.createSystem(
        pdb_file.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )

    torsion_force = next(
        f for f in omm_system.getForces()
        if f.__class__.__name__ == "PeriodicTorsionForce"
    )

    # OpenMM positions (convert to Angstroms)
    omm_pos_nm = np.array(
        [[v.x, v.y, v.z] for v in pdb_file.positions.value_in_unit(unit.nanometers)]
    )
    omm_pos_ang = omm_pos_nm * 10.0

    n_torsions = torsion_force.getNumTorsions()
    print(f"\nOMM PeriodicTorsionForce terms: {n_torsions}")

    # Collect all OMM quads (each quad can appear multiple times with different n)
    # Group by quad, accumulate all (n, phase, k) terms
    omm_quad_terms = {}
    for i in range(n_torsions):
        p1, p2, p3, p4, n, phase, k = torsion_force.getTorsionParameters(i)
        phase_rad = phase.value_in_unit(unit.radians)
        k_kcal = k.value_in_unit(unit.kilojoules_per_mole) / 4.184
        quad = (p1, p2, p3, p4)
        if quad not in omm_quad_terms:
            omm_quad_terms[quad] = []
        omm_quad_terms[quad].append((n, phase_rad, k_kcal))

    # Find quads missing from JAX
    missing_quads = {}
    for quad, terms in omm_quad_terms.items():
        fwd = quad
        rev = quad[::-1]
        if fwd not in jax_all_quads and rev not in jax_all_quads:
            missing_quads[quad] = terms

    print(f"Unique OMM quads:      {len(omm_quad_terms)}")
    print(f"Missing from JAX:      {len(missing_quads)} unique quads")

    # For each missing quad, compute energy contribution
    total_missing_E = 0.0
    print("\n--- Missing quad energies ---")
    print(f"{'Quad':40s} {'phi(deg)':>10s} {'E(kcal/mol)':>12s}")
    print("-" * 65)

    # Check atom count consistency
    n_omm = omm_pos_ang.shape[0]
    n_jax = pos_ang.shape[0]
    if n_omm != n_jax:
        print(f"\nWARNING: atom count mismatch OMM={n_omm} JAX={n_jax}")

    for quad, terms in sorted(missing_quads.items()):
        p1, p2, p3, p4 = quad
        # Use OpenMM positions (same structure)
        if max(p1, p2, p3, p4) >= n_omm:
            print(f"  {quad}: atom index out of range (n_omm={n_omm})")
            continue

        phi = dihedral_angle(
            omm_pos_ang[p1], omm_pos_ang[p2], omm_pos_ang[p3], omm_pos_ang[p4]
        )
        phi_deg = np.degrees(phi)

        e_quad = 0.0
        for n, phase, k in terms:
            e_quad += 0.5 * k * (1.0 + np.cos(n * phi - phase))

        total_missing_E += e_quad
        quad_str = f"({p1},{p2},{p3},{p4})"
        print(f"{quad_str:40s} {phi_deg:10.2f} {e_quad:12.4f}")

    print("-" * 65)
    print(f"Total missing energy:  {total_missing_E:.4f} kcal/mol")
    print(f"Reported gap:           7.88 kcal/mol")
    print(f"Unexplained:           {7.88 - total_missing_E:.4f} kcal/mol")

    # Also verify JAX total torsion vs OMM total
    print("\n--- JAX improper energies (sanity check) ---")
    imp_arr = np.array(sys.impropers)       # (N, 4)
    imp_params = np.array(sys.improper_params)  # (N, M, 3): [n, phase, k]

    jax_imp_total = 0.0
    for i in range(imp_arr.shape[0]):
        p1, p2, p3, p4 = imp_arr[i]
        phi = dihedral_angle(
            pos_ang[p1], pos_ang[p2], pos_ang[p3], pos_ang[p4]
        )
        e_imp = 0.0
        for term in imp_params[i]:
            n, phase, k = term
            e_imp += 0.5 * k * (1.0 + np.cos(n * phi - phase))
        jax_imp_total += e_imp

    print(f"JAX improper total (recomputed): {jax_imp_total:.4f} kcal/mol")

    # Per-quad analysis for matching (ordering mismatch check)
    print("\n--- Checking atom-ordering of missing quads ---")
    print("For each missing quad, check if center is at position j (JAX) vs k (OpenMM)")
    atoms = list(pdb_file.topology.atoms())
    for quad, terms in sorted(missing_quads.items()):
        p1, p2, p3, p4 = quad
        names = [atoms[i].name + f"({atoms[i].residue.name})" for i in [p1, p2, p3, p4]]
        k_sum = sum(k for _, _, k in terms)
        print(f"  {quad}: {' - '.join(names)}  k_sum={k_sum:.3f}")

    os.unlink(tmp_path)


if __name__ == "__main__":
    main()
