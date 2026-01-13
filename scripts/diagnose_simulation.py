#!/usr/bin/env python3
"""Diagnostic script to profile Prolix simulation failures."""

import argparse
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import proxide
from proxide import CoordFormat, OutputSpec
from proxide.io.parsing.backend import parse_structure

from prolix import simulate
from prolix.physics import neighbor_list as nl
from prolix.physics import system as physics_system

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DiagnosticLogger:
    def __init__(self, name: str):
        self.name = name
        self.report = {
            "name": name,
            "status": "NOT_STARTED",
            "phases": {},
            "failure_mode": None,
            "error_message": None,
        }

    def log_phase(self, phase_name: str, data: Dict[str, Any]):
        self.report["phases"][phase_name] = data
        logger.info(f"Phase {phase_name} completed for {self.name}")

    def set_failure(self, mode: str, message: str):
        self.report["status"] = "FAILED"
        self.report["failure_mode"] = mode
        self.report["error_message"] = message
        logger.error(f"FAILURE for {self.name}: {mode} - {message}")

    def set_success(self):
        self.report["status"] = "SUCCESS"
        logger.info(f"SUCCESS for {self.name}")

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"diagnostic_{self.name}.json")
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2)
        logger.info(f"Report saved to {path}")

def get_memory_stats() -> Dict[str, Any]:
    try:
        device = jax.devices()[0]
        if device.platform == "gpu":
            stats = device.memory_stats()
            return {
                "bytes_in_use": stats.get("bytes_in_use", 0),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                "bytes_limit": stats.get("bytes_limit", 0),
            }
    except Exception as e:
        logger.warning(f"Could not get memory stats: {e}")
    return {}

def run_diagnostic(pdb_path: str, output_dir: str):
    name = os.path.basename(pdb_path).split(".")[0]
    diag = DiagnosticLogger(name)
    
    try:
        # --- PHASE 1: Loading ---
        diag.log_phase("init", {"memory_before": get_memory_stats()})
        
        ff_path = "proxide/src/proxide/assets/protein.ff19SB.xml"
        if not os.path.exists(ff_path):
             # Fallback if running from different dir
             ff_path = os.path.join(os.getcwd(), "proxide/src/proxide/assets/protein.ff19SB.xml")
        
        spec = OutputSpec(
            coord_format=CoordFormat.Full,
            force_field=ff_path,
            parameterize_md=True,
            add_hydrogens=False,
        )
        
        protein = parse_structure(pdb_path, spec)
        
        # Assign radii if missing (common for GBSA)
        if protein.radii is None:
            radii_list = proxide.assign_mbondi2_radii(protein.atom_names, protein.bonds.tolist())
            protein = protein.replace(radii=jnp.array(radii_list))
            
        n_atoms = protein.coordinates.shape[0] if protein.coordinates.ndim == 2 else len(protein.coordinates) // 3
        diag.log_phase("loading", {
            "n_atoms": n_atoms,
            "memory_after": get_memory_stats()
        })

        # --- PHASE 2: Energy Function Setup ---
        # Convert protein to system_params dict for prolix
        def to_jnp(x, default=None):
            if x is None:
                return default
            return jnp.array(x)

        system_params = {
            "charges": to_jnp(protein.charges),
            "sigmas": to_jnp(protein.sigmas),
            "epsilons": to_jnp(protein.epsilons),
            "bonds": to_jnp(protein.bonds, jnp.zeros((0, 2), dtype=jnp.int32)),
            "bond_params": to_jnp(protein.bond_params, jnp.zeros((0, 2))),
            "angles": to_jnp(protein.angles, jnp.zeros((0, 3), dtype=jnp.int32)),
            "angle_params": to_jnp(protein.angle_params, jnp.zeros((0, 2))),
            "dihedrals": to_jnp(protein.proper_dihedrals, jnp.zeros((0, 4), dtype=jnp.int32)),
            "dihedral_params": to_jnp(protein.dihedral_params, jnp.zeros((0, 3))),
            "impropers": to_jnp(protein.impropers, jnp.zeros((0, 4), dtype=jnp.int32)),
            "improper_params": to_jnp(protein.improper_params, jnp.zeros((0, 3))),
            "gb_radii": to_jnp(protein.radii),
        }
        
        from jax_md import space
        displacement_fn, shift_fn = space.free()
        exclusion_spec = nl.ExclusionSpec.from_system_params(system_params)
        
        energy_fn = physics_system.make_energy_fn(
            displacement_fn,
            system_params,
            exclusion_spec=exclusion_spec,
            implicit_solvent=True
        )
        
        diag.log_phase("setup", {"memory_after": get_memory_stats()})

        # Helper to get term-by-term energies and max forces
        def get_term_breakdown(r):
            terms = {}
            
            # 1. Bonded
            from prolix.physics import bonded
            bond_fn = bonded.make_bond_energy_fn(displacement_fn, system_params["bonds"], system_params["bond_params"])
            terms["bond"] = (bond_fn, bond_fn(r))
            
            angle_fn = bonded.make_angle_energy_fn(displacement_fn, system_params["angles"], system_params["angle_params"])
            terms["angle"] = (angle_fn, angle_fn(r))
            
            dihedral_fn = bonded.make_dihedral_energy_fn(displacement_fn, system_params["dihedrals"], system_params["dihedral_params"])
            terms["dihedral"] = (dihedral_fn, dihedral_fn(r))
            
            # 2. Non-bonded (Implicit)
            # Recreate parts of make_energy_fn logic for breakdown
            from prolix.physics import generalized_born
            radii = system_params["gb_radii"]
            charges = system_params["charges"]
            sigmas = system_params["sigmas"]
            epsilons = system_params["epsilons"]
            
            def lj_only(r_pos):
                dr = space.map_product(displacement_fn)(r_pos, r_pos)
                dist = space.distance(dr)
                sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
                eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])
                e_lj = jax_md_energy.lennard_jones(dist, sig_ij, eps_ij)
                mask = 1.0 - jnp.eye(r_pos.shape[0])
                if exclusion_spec is not None:
                     # This is a bit complex to recreate perfectly here, but let's use a simple mask for now
                     pass
                return 0.5 * jnp.sum(e_lj * mask)

            from jax_md import energy as jax_md_energy
            terms["lj"] = (lj_only, lj_only(r))
            
            def elec_only(r_pos):
                dr = space.map_product(displacement_fn)(r_pos, r_pos)
                dist = space.distance(dr)
                dist_safe = dist + 1e-6
                q_ij = charges[:, None] * charges[None, :]
                e_coul = (332.0637) * (q_ij / dist_safe)
                mask = 1.0 - jnp.eye(r_pos.shape[0])
                return 0.5 * jnp.sum(e_coul * mask)
            
            terms["elec_coulomb"] = (elec_only, elec_only(r))

            results = {}
            for term_name, (fn, e_val) in terms.items():
                g = jax.grad(fn)(r)
                max_f = jnp.max(jnp.linalg.norm(g, axis=-1))
                results[term_name] = {
                    "energy": float(e_val),
                    "max_force": float(max_f)
                }
            return results

        # --- PHASE 3: Minimization ---
        logger.info("Starting detailed minimization profiling...")
        
        @jax.jit
        def get_energy_and_max_force(pos):
            e = energy_fn(pos)
            g = jax.grad(energy_fn)(pos)
            max_f = jnp.max(jnp.linalg.norm(g, axis=-1))
            return e, max_f, g

        initial_pos = jnp.array(protein.coordinates).reshape(-1, 3)
        energies = []
        max_forces = []
        breakdown_history = []
        
        curr_pos = initial_pos
        failed_step = -1
        
        # Log initial breakdown
        initial_breakdown = get_term_breakdown(initial_pos)
        diag.report["initial_breakdown"] = initial_breakdown
        logger.info(f"Initial breakdown: {initial_breakdown}")

        for step in range(30): # Sample 30 steps
            e, max_f, grad = get_energy_and_max_force(curr_pos)
            
            e_val = float(e)
            max_f_val = float(max_f)
            
            energies.append(e_val)
            max_forces.append(max_f_val)
            
            if not np.isfinite(e_val) or not np.isfinite(max_f_val):
                failed_step = step
                diag.set_failure("ENERGY_EXPLOSION", f"Inf/NaN detected at step {step}. Energy: {e_val}, Max Force: {max_f_val}")
                break
                
            if max_f_val > 1e6:
                 diag.set_failure("GRADIENT_INSTABILITY", f"Extremely large force at step {step}. Max Force: {max_f_val}")
                 # We don't break here, let it try to continue a bit
            
            # Simple gradient descent step for profiling
            curr_pos = curr_pos - 0.001 * grad
            
        diag.log_phase("minimization", {
            "energies": energies,
            "max_forces": max_forces,
            "failed_step": failed_step,
            "memory_after": get_memory_stats()
        })
        
        if diag.report["status"] == "NOT_STARTED":
            diag.set_success()

    except Exception as e:
        error_msg = str(e)
        if "Resource exhausted" in error_msg or "out of memory" in error_msg.lower():
            diag.set_failure("RESOURCE_EXHAUSTED", f"Memory error: {error_msg}")
        else:
            diag.set_failure("PARAMETER_ERROR", f"Unexpected error: {error_msg}\n{traceback.format_exc()}")
    
    diag.save(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Diagnose Prolix simulation failures.")
    parser.add_argument("--pdb", type=str, required=True, help="Path to PDB file")
    parser.add_argument("--output_dir", type=str, default="outputs/diagnostics", help="Output directory")
    args = parser.parse_args()

    run_diagnostic(args.pdb, args.output_dir)

if __name__ == "__main__":
    main()
