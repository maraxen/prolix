"""Load bonded parameters from Phase A params_init.json files.

The JSON schema comes from scripts/data/build_params_init.py:
  - bonds[*]: {i, j, k (force constant), r0, type_pair}
  - angles[*]: {i, j, k (vertex), k_theta, theta0_deg, type_triple}
  - proper_torsions[*]: {i, j, k, l, periodicity[], phase_deg[], k_phi[]}
      (In v0, each has len=1 per array)

Converts theta0_deg → theta0_rad and phase_deg → phase_rad.
"""

import json
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from prolix.fitting.params import BondedParams
from prolix.fitting.topology import BondedTopology


def load_params_init_json(path: Path) -> Tuple[BondedParams, BondedTopology]:
    """Parse a Phase A params_init.json file into (BondedParams, BondedTopology).

    Args:
        path: Path to params_init.json (e.g., data/ani1x_subset/lane_a/mol_000.params_init.json)

    Returns:
        (params, topology) tuple where:
        - params: BondedParams with k_bond, r0, k_theta, theta0_rad, k_phi (all jnp arrays)
        - topology: BondedTopology with bond_idx, angle_idx, torsion_idx, torsion_periodicity, torsion_phase_rad (all np arrays for static use)
    """
    with open(path, "r") as f:
        data = json.load(f)

    # ===== BONDS =====
    bonds = data["bonds"]
    if len(bonds) > 0:
        bond_idx_list = [[b["i"], b["j"]] for b in bonds]
        k_bond_list = [b["k"] for b in bonds]
        r0_list = [b["r0"] for b in bonds]

        bond_idx = np.array(bond_idx_list, dtype=np.int32)
        k_bond = jnp.array(k_bond_list, dtype=jnp.float32)
        r0 = jnp.array(r0_list, dtype=jnp.float32)
    else:
        bond_idx = np.zeros((0, 2), dtype=np.int32)
        k_bond = jnp.array([], dtype=jnp.float32)
        r0 = jnp.array([], dtype=jnp.float32)

    # ===== ANGLES =====
    angles = data["angles"]
    if len(angles) > 0:
        angle_idx_list = [[a["i"], a["j"], a["k"]] for a in angles]
        k_theta_list = [a["k_theta"] for a in angles]
        theta0_deg_list = [a["theta0_deg"] for a in angles]

        angle_idx = np.array(angle_idx_list, dtype=np.int32)
        k_theta = jnp.array(k_theta_list, dtype=jnp.float32)
        theta0_rad = jnp.array(theta0_deg_list, dtype=jnp.float32) * jnp.pi / 180.0
    else:
        angle_idx = np.zeros((0, 3), dtype=np.int32)
        k_theta = jnp.array([], dtype=jnp.float32)
        theta0_rad = jnp.array([], dtype=jnp.float32)

    # ===== PROPER TORSIONS =====
    torsions = data["proper_torsions"]
    if len(torsions) > 0:
        torsion_idx_list = [[t["i"], t["j"], t["k"], t["l"]] for t in torsions]
        # In v0, each torsion has a list of periodicities, phases, k_phi (all length 1)
        # We stack them to get (N_torsions, n_terms) arrays
        n_torsions = len(torsions)
        n_terms_per_torsion = [len(t["periodicity"]) for t in torsions]

        if len(set(n_terms_per_torsion)) == 1:
            # All torsions have same number of terms
            n_terms = n_terms_per_torsion[0]

            torsion_idx = np.array(torsion_idx_list, dtype=np.int32)
            periodicity_list = [t["periodicity"] for t in torsions]
            phase_deg_list = [t["phase_deg"] for t in torsions]
            k_phi_list = [t["k_phi"] for t in torsions]

            torsion_periodicity = np.array(periodicity_list, dtype=np.int32)  # (N_torsions, n_terms)
            torsion_phase_rad = np.array(phase_deg_list, dtype=np.float32) * np.pi / 180.0  # (N_torsions, n_terms)
            k_phi = jnp.array(k_phi_list, dtype=jnp.float32)  # (N_torsions, n_terms)
        else:
            # Torsions have varying number of terms (not expected in v0, but handle it)
            # Pad to max
            max_terms = max(n_terms_per_torsion)
            torsion_idx = np.array(torsion_idx_list, dtype=np.int32)

            periodicity_padded = []
            phase_deg_padded = []
            k_phi_padded = []

            for t in torsions:
                per = t["periodicity"] + [0] * (max_terms - len(t["periodicity"]))
                phase = t["phase_deg"] + [0.0] * (max_terms - len(t["phase_deg"]))
                kphi = t["k_phi"] + [0.0] * (max_terms - len(t["k_phi"]))
                periodicity_padded.append(per)
                phase_deg_padded.append(phase)
                k_phi_padded.append(kphi)

            torsion_periodicity = np.array(periodicity_padded, dtype=np.int32)
            torsion_phase_rad = np.array(phase_deg_padded, dtype=np.float32) * np.pi / 180.0
            k_phi = jnp.array(k_phi_padded, dtype=jnp.float32)
    else:
        torsion_idx = np.zeros((0, 4), dtype=np.int32)
        torsion_periodicity = np.zeros((0, 1), dtype=np.int32)
        torsion_phase_rad = np.zeros((0, 1), dtype=np.float32)
        k_phi = jnp.zeros((0, 1), dtype=jnp.float32)

    # ===== BUILD TOPOLOGY AND PARAMS =====
    topology = BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=torsion_periodicity,
        torsion_phase_rad=torsion_phase_rad,
    )

    params = BondedParams(
        k_bond=k_bond,
        r0=r0,
        k_theta=k_theta,
        theta0_rad=theta0_rad,
        k_phi=k_phi,
    )

    return params, topology
