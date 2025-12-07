"""Explicit solvation tools."""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
from jax_md import util

Array = util.Array

# TIP3P Parameters
TIP3P_PARAMS = {
    # Charges
    "charge_O": -0.834,
    "charge_H": 0.417,
    # LJ (O only, H is 0)
    "sigma_O": 3.15061,  # Angstroms
    "epsilon_O": 0.1521,  # kcal/mol
    "sigma_H": 0.0,
    "epsilon_H": 0.0,
    # Geometry
    "OH_dist": 0.9572,  # Angstroms
    "HOH_angle": 104.52,  # Degrees
}

def create_water_box(
    box_size: Array,
    density: float = 1.0,  # g/cm^3
) -> Array:
    """Creates a box of equilibrated water (simple cubic lattice for start).
    
    Args:
        box_size: (3,) box dimensions in Angstroms
        density: Target density
        
    Returns:
        positions: (N_waters * 3, 3) flattened positions
    """
    # TODO: Implement proper filling or load pre-equilibrated box
    # For now, simplistic lattice filling
    
    # Water mass ~ 18 g/mol
    # Number density = (rho * N_A) / MW
    # 1 g/cm^3 = 1e-24 g/A^3
    # rho_num = 1 * 6.022e23 / 18 * 1e-24 ~= 0.033 molecules/A^3
    
    n_waters = int(jnp.prod(box_size) * 0.033)
    
    # Grid side
    k = int(n_waters**(1/3)) + 1
    spacing = box_size / k
    
    # Generate points
    x = jnp.linspace(0, box_size[0] - spacing[0], k)
    y = jnp.linspace(0, box_size[1] - spacing[1], k)
    z = jnp.linspace(0, box_size[2] - spacing[2], k)
    
    xv, yv, zv = jnp.meshgrid(x, y, z)
    oxygens = jnp.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
    
    # Truncate to box
    mask = (oxygens < box_size).all(axis=1)
    oxygens = oxygens[mask]
    
    # Add Hydrogens (simple orientation)
    # H1: +x
    # H2: in xy plane
    # Construct local frame
    
    # Need to expand to complete waters
    # Just returning placeholders for planning
    return oxygens

def add_solvent(
    solute_positions: Array,
    solute_radii: Array,
    box_padding: float = 10.0,
) -> Tuple[Array, Array]:
    """Adds solvent around solute.
    
    Args:
        solute_positions: (N_solute, 3)
        solute_radii: (N_solute,) VDW radii for exclusion
        box_padding: Padding in Angstroms
        
    Returns:
        (combined_positions, box_size)
    """
    # Simple rectangular box
    min_coords = jnp.min(solute_positions, axis=0)
    max_coords = jnp.max(solute_positions, axis=0)
    
    box_size = (max_coords - min_coords) + 2 * box_padding
    
    # Shift solute to center
    center = (max_coords + min_coords) / 2
    box_center = box_size / 2
    shift = box_center - center
    
    centered_solute = solute_positions + shift
    
    # Generate solvent (placeholder)
    # In real implementation, fill box and remove overlapping
    
    return centered_solute, box_size
