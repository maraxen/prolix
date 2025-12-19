
import os
import sys

import jax.numpy as jnp
from jax_md import space

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from prolix.physics import system


def verify_pme():
    print("Verifying PME Parity...")

    # Setup System
    box_L = 30.0
    box = jnp.eye(3) * box_L
    displacement_fn, shift_fn = space.periodic(box_L)

    # 2 Particles
    # q1 = +1, q2 = -1
    # r = 5.0 A
    r = jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0]
    ])

    charges = jnp.array([1.0, -1.0])
    sigmas = jnp.array([1.0, 1.0])
    epsilons = jnp.array([0.0, 0.0]) # No LJ

    base_params = {
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2)),
        "angle_params": jnp.zeros((0, 2)),
        "dihedral_params": jnp.zeros((0, 3)),
        "improper_params": jnp.zeros((0, 3)),
        "exclusion_mask": jnp.zeros((2, 2)), # No exclusions
    }

    alpha = 0.34
    grid = 32
    cutoff = 9.0

    def get_energy(params, name):
        energy_fn = system.make_energy_fn(
            displacement_fn,
            params,
            use_pbc=True,
            box=box,
            pme_alpha=alpha,
            pme_grid_points=grid,
            cutoff_distance=cutoff,
            implicit_solvent=False
        )
        e = energy_fn(r)
        print(f"{name}: {e:.4f} kcal/mol")
        return e

    # Analytical Expectation (Vacuum Coulomb)
    # E = 332.0637 * q1*q2 / r
    expected_coulomb = 332.0637 * (1.0 * -1.0) / 5.0
    print(f"Analytical Coulomb (1/r): {expected_coulomb:.4f}")

    # Case 1: Normal Interaction
    # PME should approximate this (Direct + Recip - Self)
    e_normal = get_energy(base_params, "Normal (PME)")

    # Check error
    err = abs(e_normal - expected_coulomb)
    print(f"Error: {err:.4f} ({(err/abs(expected_coulomb))*100:.2f}%)")
    if err > 1.0: # PME matches exact coulomb within ~0.1% usually
        print("WARNING: PME Normal interaction mismatch!")

    # Case 2: Excluded (1-2)
    # Set exclusion mask
    params_excl = base_params.copy()
    mask = jnp.array([[0, 1], [1, 0]]) # Exclude 0-1
    # exclusion_mask: 1=Allow, 0=Exclude?
    # system.py: "interaction_allowed = exclusion_mask..."
    # Wait, system.py: "mask = 1.0 - eye... e_coul = where(exclusion_mask, e_coul, 0)"
    # Usually exclusion_mask is "Are they excluded?" NO.
    # OpenMM/Standard: exclusion_mask usually means "Is interaction ON?"
    # Let's check system.py line 308:
    # e_coul = jnp.where(exclusion_mask, e_coul, 0.0)
    # So 1 (True) = ON, 0 (False) = OFF.

    # So for Normal, we want 1s everywhere (except diag).
    # base_params["exclusion_mask"] was zeros... so everything OFF?
    # Wait.
    # In my base_params, I set exclusion_mask to zeros.
    # If system.py interprets 0 as OFF, then e_normal should have been 0?
    # Let's check output.

    # For Excluded, we want 0 for pair (0,1).
    mask_on = jnp.ones((2, 2)) - jnp.eye(2)
    base_params["exclusion_mask"] = mask_on # All ON

    # Re-run Normal
    print("--- Re-running Normal with Correct Mask ---")
    e_normal = get_energy(base_params, "Normal (Mask=1)")

    # Exclude 0-1
    mask_off = jnp.zeros((2, 2))
    params_excl["exclusion_mask"] = mask_off

    # HOWEVER, PME Exceptions (Recip subtraction) uses `pme_bonds` to find 1-2s!
    # `compute_pme_exceptions` rebuilds graph from `bonds`.
    # It does NOT use `exclusion_mask` for determining 1-2/1-3/1-4.
    # It uses `system_params["bonds"]`.
    # So to test Excluded 1-2, we must ADD A BOND between 0 and 1.

    params_bond = base_params.copy()
    params_bond["bonds"] = jnp.array([[0, 1]], dtype=jnp.int32)
    # Also set exclusion mask to 0 for this pair (Direct space exclusion)
    params_bond["exclusion_mask"] = mask_off

    e_excluded = get_energy(params_bond, "Excluded (Bond 0-1)")
    # Should be 0.0 (Direct=0, Recip - Recip_Correction = 0?)
    # E_bond term is 0 (params zero).

    if abs(e_excluded) < 0.1:
        print("Excluded Check: PASS")
    else:
        print(f"Excluded Check: FAIL (Expected 0.0, got {e_excluded:.4f})")

    # Case 3: Scaled 1-4
    # Need to create 1-2-3-4 chain to get 1-4.
    # 4 Particles.
    print("\n--- Testing 1-4 Scaling ---")
    r4 = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 0.0, 0.0]  # 0->3 is 5.0 A
    ])
    q4 = jnp.array([1.0, 0.0, 0.0, -1.0]) # Only 0 and 3 charged
    sig4 = jnp.ones(4)
    eps4 = jnp.zeros(4)
    bonds4 = jnp.array([[0,1], [1,2], [2,3]], dtype=jnp.int32)

    params_14 = {
        "charges": q4, "sigmas": sig4, "epsilons": eps4,
        "bonds": bonds4,
        "angles": jnp.zeros((0,3), dtype=jnp.int32),
        "dihedrals": jnp.zeros((0,4), dtype=jnp.int32),
        "impropers": jnp.zeros((0,4), dtype=jnp.int32),
        "bond_params": jnp.zeros((3,2)),
        "angle_params": jnp.zeros((0,2)),
        "dihedral_params": jnp.zeros((0,3)),
        "improper_params": jnp.zeros((0,3)),
        "coulomb14scale": 0.5, # Explicitly set scale
    }

    # Exclusion Mask: Exclude 1-2 (0-1, 1-2, 2-3) and 1-3 (0-2, 1-3).
    # 1-4 (0-3) should be enabled?
    # Usually 1-4 is handled by scaling matrix or mask?
    # If we rely on default mask generation (which `system.py` uses if scale matrix is None),
    # `system.py` logic:
    # "interaction_allowed = exclusion_mask..."
    # We need to construct the exclusion mask manually here?
    # `system.py` setup usually builds this from topology.
    # BRIDGE usually does this.
    # Here we perform manual setup.
    # Let's set 0-3 to allowed. All others allowed (zeros charged).
    # We exclude 0-1, 1-2, 2-3, 0-2, 1-3.
    # Actually since only 0 and 3 have charge, we only care about 0-3 pair.
    # Is 0-3 allowed in Direct space?
    # If we want scaling in Direct space, we must provide `scale_matrix_elec`.

    scale_mat = jnp.ones((4,4))
    # Direct space scaling for 0-3 should be 0.5?
    # Correct.
    scale_mat = scale_mat.at[0,3].set(0.5)
    scale_mat = scale_mat.at[3,0].set(0.5)

    params_14["scale_matrix_elec"] = scale_mat
    params_14["scale_matrix_vdw"] = jnp.ones((4,4)) # Dummy
    params_14["exclusion_mask"] = jnp.ones((4,4)) # Ignored if scale matrix present

    # We need an energy function for this 4-particle system
    energy_fn_14 = system.make_energy_fn(
        displacement_fn,
        params_14,
        use_pbc=True,
        box=box,
        pme_alpha=alpha,
        pme_grid_points=grid,
        cutoff_distance=cutoff,
        implicit_solvent=False
    )

    e_14 = energy_fn_14(r4)
    expected_14 = 0.5 * expected_coulomb

    print(f"1-4 Interaction: {e_14:.4f}")
    print(f"Expected (0.5 * 1/r): {expected_14:.4f}")

    err_14 = abs(e_14 - expected_14)
    if err_14 < 1.0:
        print("1-4 Check: PASS")
    else:
        print(f"1-4 Check: FAIL (Error {err_14:.4f})")

if __name__ == "__main__":
    verify_pme()
