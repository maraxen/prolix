"""GB Solvation Relative Energy Diagnostic.

Validates that despite the ~80 kcal/mol absolute offset between JAX OBC2
(iterative Born radii) and OpenMM OBC2 (analytical), the relative energy
*differences* for small perturbations agree. This confirms the GB energy
surface shape matches, which is what matters for dynamics.

Protocol:
  1. Prepare 1UAO with PDBFixer
  2. Compute E(r) with both engines
  3. For 10 random perturbations (δr ~ 0.01 Å), compute ΔE = E(r+δr) - E(r)
  4. Compare ΔE_JAX vs ΔE_OpenMM
  Pass: Mean |ΔE_JAX - ΔE_OpenMM| < 0.5 kcal/mol
"""
import os
import sys
import numpy as np
import tempfile

try:
    import openmm
    from openmm import app, unit
except ImportError:
    print("Error: OpenMM not found.")
    sys.exit(1)

from pdbfixer import PDBFixer
import proxide
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat

PDB_PATH = "data/pdb/1UAO.pdb"
FF_XML = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")
N_PERTURBATIONS = 10
PERTURBATION_STD = 0.01  # Å
SEED = 42


def main():
    print("=" * 60)
    print("  GB Solvation Relative Energy Diagnostic")
    print("=" * 60)

    # 1. Prep structure
    fixer = PDBFixer(filename=PDB_PATH)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp)
        tmp_path = tmp.name

    # 2. Setup OpenMM
    pdb = app.PDBFile(tmp_path)
    omm_ff = app.ForceField(FF_XML, "implicit/obc2.xml")
    omm_sys = omm_ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    sim = app.Simulation(pdb.topology, omm_sys, integrator)

    positions_nm = np.array(
        pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )

    def omm_energy(pos_nm):
        sim.context.setPositions(pos_nm)
        state = sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    # 3. Setup JAX
    import jax
    import jax.numpy as jnp

    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        add_hydrogens=False,
        parameterize_md=True,
        force_field=FF_XML,
    )
    protein = parse_structure(tmp_path, spec=spec)
    os.unlink(tmp_path)

    from jax_md import space
    from prolix.physics import system as sys_module
    from prolix.physics import neighbor_list as nl
    from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors

    radii = assign_mbondi2_radii(list(protein.atom_names), protein.bonds)
    scaled_radii = assign_obc2_scaling_factors(list(protein.atom_names))
    object.__setattr__(protein, 'radii', np.array(radii, dtype=np.float32))
    object.__setattr__(protein, 'scaled_radii', np.array(scaled_radii, dtype=np.float32))

    disp_fn, shift_fn = space.free()
    exclusion_spec = nl.ExclusionSpec.from_protein(protein)
    energy_fn = sys_module.make_energy_fn(
        disp_fn, protein, implicit_solvent=True, exclusion_spec=exclusion_spec
    )
    coords_angstrom = np.array(positions_nm) * 10.0  # nm → Å
    jax_pos = jnp.asarray(coords_angstrom, dtype=jnp.float32)

    def jax_energy(pos):
        return float(energy_fn(pos))

    # 4. Baseline energies
    e_omm_base = omm_energy(positions_nm)
    e_jax_base = jax_energy(jax_pos)
    abs_gap = abs(e_jax_base - e_omm_base)
    print(f"\nBaseline: JAX={e_jax_base:.2f}, OpenMM={e_omm_base:.2f}, "
          f"absolute gap={abs_gap:.2f} kcal/mol")

    # 5. Perturbation sweep
    rng = np.random.default_rng(SEED)
    delta_e_diffs = []

    print(f"\nPerturbation sweep ({N_PERTURBATIONS} samples, σ={PERTURBATION_STD} Å):")
    print(f"{'#':>3}  {'ΔE_JAX':>10}  {'ΔE_OMM':>10}  {'|diff|':>10}")
    print("-" * 40)

    for i in range(N_PERTURBATIONS):
        # Random perturbation in Å
        delta = rng.normal(0, PERTURBATION_STD, size=coords_angstrom.shape).astype(
            np.float32
        )

        # JAX
        perturbed_jax = jax_pos + jnp.asarray(delta)
        de_jax = jax_energy(perturbed_jax) - e_jax_base

        # OpenMM (convert delta to nm)
        perturbed_omm = positions_nm + delta.astype(np.float64) / 10.0
        de_omm = omm_energy(perturbed_omm) - e_omm_base

        diff = abs(de_jax - de_omm)
        delta_e_diffs.append(diff)
        print(f"{i+1:3d}  {de_jax:10.4f}  {de_omm:10.4f}  {diff:10.4f}")

    mean_diff = np.mean(delta_e_diffs)
    max_diff = np.max(delta_e_diffs)

    print(f"\nMean |ΔE_JAX - ΔE_OMM|: {mean_diff:.4f} kcal/mol")
    print(f"Max  |ΔE_JAX - ΔE_OMM|: {max_diff:.4f} kcal/mol")

    if mean_diff < 0.5:
        print("\nSUCCESS: Relative energies match despite absolute offset.")
    else:
        print(f"\nFAILURE: Mean ΔE diff {mean_diff:.4f} exceeds 0.5 kcal/mol.")


if __name__ == "__main__":
    main()
