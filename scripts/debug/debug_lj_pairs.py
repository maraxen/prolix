import os
import openmm.app as app
import openmm
import jax.numpy as jnp
import jax
from proxide import parse_structure, OutputSpec

def debug_lj_pairs():
    pdb_path = "data/pdb/1UAO.pdb"
    pdb = app.PDBFile(pdb_path)
    
    spec = OutputSpec(parameterize_md=True, force_field="/home/marielle/projects/prolix/.venv/lib/python3.13/site-packages/openmm/app/data/amber14/protein.ff14SB.xml", add_hydrogens=False)
    protein = parse_structure(pdb_path, spec)
    
    n_atoms = len(protein.sigmas)
    pos = jnp.array(protein.coordinates)[:n_atoms]
    
    dr = pos[:, None, :] - pos[None, :, :]
    d = jnp.sqrt(jnp.sum(dr**2, axis=-1))
    
    mask = (jnp.arange(n_atoms)[:, None] != jnp.arange(n_atoms)[None, :])
    min_dist = jnp.min(jnp.where(mask, d, 1e9))
    print(f"Min distance (non-self): {min_dist:.4f}")
    
    # Find which pair is too close
    if min_dist < 0.1:
        idx = jnp.argmin(jnp.where(mask, d, 1e9))
        i = idx // n_atoms
        j = idx % n_atoms
        print(f"Clashing pair: {i}, {j} dist={d[i, j]:.6f}")

if __name__ == "__main__":
    debug_lj_pairs()
