"""Diagnostic script to compare Born Radius calculation step-by-step with OpenMM."""

import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, 'src')

# Load structure and get positions
from pathlib import Path
import openmm.app as app
import openmm.unit as unit
import openmm

# Load PDB
pdb_path = Path("data/pdb/1UAO.pdb")
pdb = app.PDBFile(str(pdb_path))
import os
ff19sb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
ff = app.ForceField(ff19sb_path, 'implicit/obc2.xml')
omm_system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

# Get positions in Angstroms
positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
print(f"Loaded {len(positions)} atoms")

# Extract OpenMM GBSA parameters
omm_offset_radii = []  # Offset radius (Å) = radius - 0.09
omm_scaled_radii = []  # Scaled radius (Å) = offset_radius * scale

for force in omm_system.getForces():
    if isinstance(force, openmm.CustomGBForce):
        print(f"CustomGBForce found with {force.getNumParticles()} particles")
        for i in range(force.getNumParticles()):
            params = force.getParticleParameters(i)
            # params = [charge, offset_radius (nm), scaled_radius (nm)]
            offset_radius_a = params[1] * 10.0  # nm -> Å
            scaled_radius_a = params[2] * 10.0  # nm -> Å
            
            omm_offset_radii.append(offset_radius_a)
            omm_scaled_radii.append(scaled_radius_a)

omm_offset_radii = np.array(omm_offset_radii)
omm_scaled_radii = np.array(omm_scaled_radii)
n_atoms = len(positions)

print(f"\nOpenMM Parameters (first 5 atoms):")
for i in range(5):
    intrinsic_r = omm_offset_radii[i] + 0.09
    print(f"  Atom {i}: intrinsic_r={intrinsic_r:.4f} Å, offset_r={omm_offset_radii[i]:.4f} Å, scaled_r={omm_scaled_radii[i]:.4f} Å")

# =============================================================================
# Compute Born Radii using OpenMM formula (reference implementation)
# =============================================================================
def compute_pair_integral_omm(r, or_i, sr_j):
    """OpenMM OBC2 pair integral formula."""
    D = abs(r - sr_j)
    L = max(or_i, D)
    U = r + sr_j
    
    if (r + sr_j) <= or_i:
        return 0.0
    
    inv_L = 1.0 / L
    inv_U = 1.0 / U
    
    # OpenMM: 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r)
    term1 = 0.5 * (inv_L - inv_U)
    term2 = 0.5 * np.log(L / U) / r  # OpenMM uses 0.5*log/r INSIDE the outer 0.5
    term3 = 0.25 * (r - sr_j**2 / r) * (inv_U**2 - inv_L**2)  # OpenMM uses 0.25 INSIDE
    
    return 0.5 * (inv_L - inv_U + term3 / 0.5 + term2 / 0.5)  # Wait, let me simplify

def compute_pair_integral_omm_v2(r, or_i, sr_j):
    """OpenMM OBC2 pair integral formula (cleaner)."""
    D = abs(r - sr_j)
    L = max(or_i, D)
    U = r + sr_j
    
    if (r + sr_j) <= or_i:
        return 0.0
    
    # OpenMM formula: 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r)
    # = 0.5/L - 0.5/U + 0.125*(r-sr2^2/r)*(1/U^2-1/L^2) + 0.25*log(L/U)/r
    inv_L = 1.0 / L
    inv_U = 1.0 / U
    
    term1 = 0.5 * (inv_L - inv_U)
    term2 = 0.25 * np.log(L / U) / r
    term3 = 0.125 * (r - sr_j**2 / r) * (inv_U**2 - inv_L**2)
    
    return term1 + term2 + term3

def compute_born_radii_omm(positions, offset_radii, scaled_radii):
    """Compute Born radii using OpenMM OBC2 formula."""
    n = len(positions)
    born_radii = np.zeros(n)
    
    for i in range(n):
        or_i = offset_radii[i]
        I_sum = 0.0
        
        for j in range(n):
            if i == j:
                continue
            r = np.sqrt(np.sum((positions[i] - positions[j])**2))
            sr_j = scaled_radii[j]
            I_sum += compute_pair_integral_omm_v2(r, or_i, sr_j)
        
        # psi = I * or
        psi = I_sum * or_i
        
        # tanh argument: psi - 0.8*psi^2 + 4.85*psi^3 (OBC2)
        tanh_arg = psi - 0.8 * psi**2 + 4.85 * psi**3
        tanh_val = np.tanh(tanh_arg)
        
        # B = 1 / (1/or - tanh(...)/radius)
        # radius = or + offset = or + 0.09
        full_radius = or_i + 0.09
        inv_B = 1.0 / or_i - tanh_val / full_radius
        born_radii[i] = 1.0 / inv_B
    
    return born_radii

print("\n" + "="*60)
print("Computing Born Radii using OpenMM formula...")
print("="*60)
omm_born_radii = compute_born_radii_omm(positions, omm_offset_radii, omm_scaled_radii)
print(f"OpenMM-formula Born Radii (first 5): {omm_born_radii[:5]}")
print(f"Mean: {np.mean(omm_born_radii):.4f} Å")

# =============================================================================
# Compute Born Radii using JAX MD
# =============================================================================
print("\n" + "="*60)
print("Computing Born Radii using JAX MD...")
print("="*60)

from prolix.physics import generalized_born

jax_positions = jnp.array(positions)
jax_radii = jnp.array(omm_offset_radii + 0.09)  # Intrinsic radii
jax_scaled_radii = jnp.array(omm_scaled_radii)

jax_born_radii = generalized_born.compute_born_radii(
    jax_positions, jax_radii,
    dielectric_offset=0.09,
    mask=None,  # Include all pairs
    scaled_radii=jax_scaled_radii
)
jax_born_radii = np.array(jax_born_radii)

print(f"JAX MD Born Radii (first 5): {jax_born_radii[:5]}")
print(f"Mean: {np.mean(jax_born_radii):.4f} Å")

# =============================================================================
# Compare
# =============================================================================
print("\n" + "="*60)
print("Comparison:")
print("="*60)
diff = np.abs(omm_born_radii - jax_born_radii)
print(f"Max Diff: {np.max(diff):.6f} Å")
print(f"Mean Diff: {np.mean(diff):.6f} Å")

if np.max(diff) < 0.001:
    print("✓ Born Radii MATCH!")
else:
    print("✗ Mismatch found. Investigating...")
    for i in range(min(5, n_atoms)):
        print(f"  Atom {i}: OMM={omm_born_radii[i]:.6f}, JAX={jax_born_radii[i]:.6f}, Diff={diff[i]:.6f}")


# Compare with OpenMM
# Get OpenMM Born radii from context
context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001*unit.picoseconds))
context.setPositions(pdb.getPositions())
state = context.getState(getEnergy=True)

# Get per-particle Born radii if available
# OpenMM stores I and B as computed values
for force in omm_system.getForces():
    if isinstance(force, openmm.CustomGBForce):
        # Get computed values
        n_computed = force.getNumComputedValues()
        print(f"\nOpenMM CustomGBForce has {n_computed} computed values")
        for cv_idx in range(n_computed):
            name, expr, cv_type = force.getComputedValueParameters(cv_idx)
            print(f"  {cv_idx}: {name} = {expr[:50]}...")

print("\n" + "="*60)
print("Computing GBSA Energy...")
print("="*60)

# Get OpenMM GBSA Energy
for force in omm_system.getForces():
    if isinstance(force, openmm.CustomGBForce):
        force.setForceGroup(5)
        
integrator = openmm.VerletIntegrator(0.001*unit.picoseconds)
platform = openmm.Platform.getPlatformByName('Reference')
simulation = app.Simulation(pdb.topology, omm_system, integrator, platform)
simulation.context.setPositions(pdb.getPositions())

omm_gbsa_energy = simulation.context.getState(getEnergy=True, groups=1<<5).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
print(f"OpenMM GBSA Energy: {omm_gbsa_energy:.4f} kcal/mol")

# Compute JAX GBSA Energy
from prolix.physics import generalized_born
from proxide.physics import constants

jax_positions = jnp.array(positions)
jax_radii = jnp.array(omm_offset_radii + 0.09)  # Intrinsic radii
jax_scaled_radii = jnp.array(omm_scaled_radii)

# Get charges from OpenMM
omm_charges = []
for force in omm_system.getForces():
    if isinstance(force, openmm.CustomGBForce):
        for i in range(force.getNumParticles()):
            params = force.getParticleParameters(i)
            omm_charges.append(params[0])  # charge is first param
jax_charges = jnp.array(omm_charges)

jax_gb_energy, jax_born_radii = generalized_born.compute_gb_energy(
    jax_positions, jax_charges, jax_radii,
    solvent_dielectric=78.5,
    solute_dielectric=1.0,
    dielectric_offset=0.09,
    mask=None,  # All pairs
    energy_mask=None,  # All pairs
    scaled_radii=jax_scaled_radii
)
jax_gb_energy = float(jax_gb_energy)

print(f"JAX MD GBSA Energy: {jax_gb_energy:.4f} kcal/mol")
print(f"Difference: {abs(omm_gbsa_energy - jax_gb_energy):.4f} kcal/mol")

# Check constants
print(f"\nConstants Check:")
print(f"  JAX COULOMB_CONSTANT: {constants.COULOMB_CONSTANT:.4f} kcal*A/(mol*e^2)")
print(f"  OpenMM uses 138.935485 kJ*nm/(mol*e^2) = {138.935485 * 0.239006:.4f} kcal*nm/(mol*e^2)")
print(f"  Converted to Angstroms: {138.935485 * 0.239006 * 10:.4f} kcal*A/(mol*e^2)")

print("\n" + "="*60)
print("Detailed GBSA Energy Breakdown...")
print("="*60)

# Compute energy manually to match OpenMM's formula
# OpenMM Self-Energy: -0.5 * k * tau * sum_i(q_i^2 / B_i)
# OpenMM Pairwise: -k * tau * sum_{i<j}(q_i * q_j / f_gb)

k = 332.0636  # Coulomb constant in kcal*A/(mol*e^2)
tau = (1.0 / 1.0) - (1.0 / 78.5)  # 1/eps_in - 1/eps_out

jax_born_radii = np.array(jax_born_radii)
jax_charges_np = np.array(jax_charges)
positions_np = np.array(positions)

# Self energy
self_energy = 0.0
for i in range(n_atoms):
    q_i = jax_charges_np[i]
    B_i = jax_born_radii[i]
    self_energy += q_i**2 / B_i

self_energy *= -0.5 * k * tau
print(f"Self-Energy: {self_energy:.4f} kcal/mol")

# Pairwise energy (using f_gb)
pairwise_energy = 0.0
for i in range(n_atoms):
    for j in range(i+1, n_atoms):  # Only count each pair once
        q_i, q_j = jax_charges_np[i], jax_charges_np[j]
        B_i, B_j = jax_born_radii[i], jax_born_radii[j]
        
        r = np.sqrt(np.sum((positions_np[i] - positions_np[j])**2))
        
        # f_gb = sqrt(r^2 + B_i*B_j*exp(-r^2/(4*B_i*B_j)))
        r_sq = r**2
        prod = B_i * B_j
        f_gb = np.sqrt(r_sq + prod * np.exp(-r_sq / (4 * prod)))
        
        pairwise_energy += q_i * q_j / f_gb

pairwise_energy *= -k * tau  # No 0.5 factor for pairwise
print(f"Pairwise Energy: {pairwise_energy:.4f} kcal/mol")

total_manual = self_energy + pairwise_energy
print(f"Total Manual: {total_manual:.4f} kcal/mol")
print(f"Total JAX MD: {jax_gb_energy:.4f} kcal/mol")
print(f"Total OpenMM: {omm_gbsa_energy:.4f} kcal/mol")
print(f"Manual - OpenMM: {total_manual - omm_gbsa_energy:.4f} kcal/mol")

# What if OpenMM uses different Born Radii?
# Let's reverse-engineer OpenMM's Born Radii from the energy.
# E_self = -0.5 * k * tau * sum(q^2 / B)
# If we know E_total and can compute E_pairwise with our B, 
# we can get implied E_self and compare.

# Or: Let's see what the energy SHOULD be with OpenMM's documented formula
# but using OUR computed Born Radii (which match the formula).

# Actually, the key insight: OpenMM may use a CUTOFF or other approximation
# internally that we're not accounting for.

# Let's check if the NonbondedMethod affects GBSA.
print("\nChecking OpenMM NonbondedMethod...")
for force in omm_system.getForces():
    if isinstance(force, openmm.CustomGBForce):
        method = force.getNonbondedMethod()
        print(f"  CustomGBForce NonbondedMethod: {method}")
        # 0 = NoCutoff, 1 = CutoffNonPeriodic, 2 = CutoffPeriodic

# Check if there's additional surface area term
print("\nNote: OpenMM obc2.xml may include ACE surface area term.")
print("This adds energy proportional to: 28.3919551*(radius+0.14)^2*(radius/B)^6")

# Compute ACE surface area energy
# OpenMM formula: 28.3919551*(radius+0.14)^2*(radius/B)^6
# where radius = or + offset (in nm)
# Note: 28.3919551 is in kJ/mol/nm^2

ace_energy = 0.0
for i in range(n_atoms):
    # OpenMM uses offset_radius + offset where offset = 0.009 nm
    # Our offset_radii are already in Angstroms, so we need offset_radii / 10 for nm
    or_nm = omm_offset_radii[i] / 10.0  # Convert Å to nm
    radius_nm = or_nm + 0.009  # Add offset (in nm)
    B_nm = jax_born_radii[i] / 10.0  # Convert Å to nm
    
    # ACE formula in kJ/mol
    ace_term_kj = 28.3919551 * (radius_nm + 0.14)**2 * (radius_nm / B_nm)**6
    ace_energy += ace_term_kj

# Convert to kcal/mol
ace_energy_kcal = ace_energy * 0.239006
print(f"\nACE Surface Area Energy: {ace_energy_kcal:.4f} kcal/mol")

# This should be POSITIVE and add to the energy
print(f"\nAdjusted Total (Manual - ACE): {total_manual + ace_energy_kcal:.4f} kcal/mol")
print(f"OpenMM Total: {omm_gbsa_energy:.4f} kcal/mol")
print(f"Difference after ACE: {(total_manual + ace_energy_kcal) - omm_gbsa_energy:.4f} kcal/mol")

print("\n" + "="*60)
print("Done.")
print("="*60)



