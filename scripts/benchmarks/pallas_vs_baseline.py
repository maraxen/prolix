"""Apples-to-apples Pallas vs XLA-baseline scaling harness (Phase 3.1/3.2 + Backlog #6167).

Compares Prolix's Pallas kernels against their XLA references on identical
inputs at water-like density across system sizes:

* ``--kernel nonbonded``: fused Pallas neighbor-list nonbonded kernel
  (``pallas_nonbonded_nl``) vs chunked XLA NL reference
  (``chunked_lj_energy_nl`` + ``chunked_coulomb_energy_nl``).
* ``--kernel flash``: 2D-tiled Pallas FlashMD kernel
  (``pallas_flash_explicit_nonbonded``) vs tiled XLA ``jax.checkpoint`` scan
  (``chunked_explicit_nonbonded_energy(..., use_pallas=False)``).
* ``--kernel pme``: SPME B-spline spread/gather Pallas vs XLA.
* ``--kernel nl_build``: ``jax_md`` cell-list neighbor-list rebuild cost vs one
  Pallas nonbonded step.
* ``--kernel nl_skin``: neighbor-list skin (``dr_threshold``) and
  ``capacity_multiplier`` sweep measuring capacity ``K``, rebuild time, Pallas
  nonbonded step time, and net amortized step time.
* ``--kernel mem_scaling``: isolated **per-process** device memory scaling
  (spawning a fresh subprocess with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` for
  every ``(path, N)`` pair) comparing ``dense_xla`` ($O(N^2)$ un-tiled),
  ``flash_xla`` ($O(N\\cdot T)$ remat), ``flash_pallas`` ($O(N)$),
  ``nl_xla`` ($O(N\\cdot K)$ chunked), and ``nl_pallas`` ($O(N\\cdot K)$ fused
  forward-force Pallas).

Pass ``--isolate-memory`` with ``nonbonded``, ``flash``, or ``pme`` to measure
``peak_mem_after_xla_bytes`` and ``peak_mem_after_pallas_bytes`` (plus working
and activation memory deltas) in fresh per-path subprocesses instead of a
single cumulative process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from prolix.pallas_kernels import (
    _triton_gpu_available,
    pallas_flash_explicit_nonbonded,
    pallas_nonbonded_nl,
)
from prolix.physics.flash_explicit import chunked_explicit_nonbonded_energy
from prolix.physics.optimization import chunked_coulomb_energy_nl, chunked_lj_energy_nl

CUTOFF = 9.0
SWITCH_WIDTH = 1.5
PME_ALPHA = 0.34
DENSITY = 0.1  # atoms / A^3 (liquid water)


def build_system(n: int, seed: int = 0, include_nl: bool = True):
    """Random periodic system at water density with optional padded neighbor list."""
    rng = np.random.default_rng(seed)
    # Jittered cubic lattice: avoids the overlapping atoms of uniform-random
    # placement, whose ~1e13 energies make float32 parity meaningless.
    spacing = (1.0 / DENSITY) ** (1.0 / 3.0)
    m = int(np.ceil(n ** (1.0 / 3.0)))
    box_len = max(float(m * spacing), float(2.0 * CUTOFF + 2.0))
    grid = np.stack(np.meshgrid(*[np.arange(m)] * 3, indexing="ij"), -1).reshape(-1, 3)
    grid = grid[rng.permutation(len(grid))[:n]]
    pos = ((grid + 0.5) * spacing + rng.uniform(-0.1, 0.1, (n, 3)) * spacing).astype(np.float32)
    pos %= box_len

    nbr_jnp = None
    k = 0
    if include_nl:
        from scipy.spatial import cKDTree

        tree = cKDTree(pos, boxsize=box_len)
        pairs = tree.query_ball_point(pos, r=CUTOFF + 1.0)
        k_max = max(len(p) for p in pairs) - 1
        k = int(np.ceil(max(k_max, 1) / 8.0) * 8)
        nbr = np.full((n, k), n, dtype=np.int32)
        for i, p in enumerate(pairs):
            p = [j for j in p if j != i]
            nbr[i, : len(p)] = p
        nbr_jnp = jnp.asarray(nbr)

    sigma = rng.uniform(1.6, 2.0, size=n).astype(np.float32)
    eps = rng.uniform(0.05, 0.2, size=n).astype(np.float32)
    q = rng.uniform(-0.6, 0.6, size=n).astype(np.float32)
    q -= q.mean()
    return {
        "box": jnp.full((3,), box_len, dtype=jnp.float32),
        "pos": jnp.asarray(pos),
        "nbr": nbr_jnp,
        "sigma": jnp.asarray(sigma),
        "eps": jnp.asarray(eps),
        "q": jnp.asarray(q),
        "mask": jnp.ones((n,), dtype=bool),
        "excl_idx": jnp.full((n, 1), -1, dtype=jnp.int32),
        "excl_vdw": jnp.ones((n, 1), dtype=jnp.float32),
        "excl_elec": jnp.ones((n, 1), dtype=jnp.float32),
        "k": k,
    }


def make_fns(s, interpret: bool):
    disp, _ = space.periodic(s["box"])

    def xla_fn(p):
        e_lj = chunked_lj_energy_nl(
            p, s["sigma"], s["eps"], s["excl_idx"], s["excl_vdw"], s["nbr"], disp,
            cutoff=CUTOFF, tile_size=256, switch_width=SWITCH_WIDTH,
        )
        e_c = chunked_coulomb_energy_nl(
            p, s["q"], s["excl_idx"], s["excl_elec"], s["nbr"], disp,
            pme_alpha=PME_ALPHA, cutoff=CUTOFF, tile_size=256,
        )
        return e_lj + e_c

    def pallas_fn(p):
        return pallas_nonbonded_nl(
            p, s["q"], s["sigma"], s["eps"], s["mask"], s["nbr"], s["excl_idx"],
            s["excl_vdw"], s["excl_elec"], box=s["box"], pme_alpha=PME_ALPHA,
            cutoff=CUTOFF, lj_switch_width=SWITCH_WIDTH, soft_core_lambda=1.0,
            include_pme_excl_erf=True, interpret=interpret,
        )

    return jax.jit(jax.value_and_grad(xla_fn)), jax.jit(jax.value_and_grad(pallas_fn))


def make_flash_fns(s, interpret: bool):
    """FlashMD dense O(N^2) tiled nonbonded: XLA lax.scan+remat vs Pallas 2D-tiled."""
    sys_stub = SimpleNamespace(
        excl_indices=s["excl_idx"],
        excl_scales_vdw=s["excl_vdw"],
        excl_scales_elec=s["excl_elec"],
        box_size=s["box"],
    )

    def xla_fn(p):
        return chunked_explicit_nonbonded_energy(
            p, s["q"], s["sigma"], s["eps"], s["mask"], sys_stub,
            T=256, soft_core_lambda=1.0, pme_alpha=PME_ALPHA,
            remat=True, use_pallas=False,
        )

    def pallas_fn(p):
        return pallas_flash_explicit_nonbonded(
            p, s["q"], s["sigma"], s["eps"], s["mask"],
            s["excl_idx"], s["excl_vdw"], s["excl_elec"],
            box=s["box"], pme_alpha=PME_ALPHA, soft_core_lambda=1.0,
            interpret=interpret,
        )

    return jax.jit(jax.value_and_grad(xla_fn)), jax.jit(jax.value_and_grad(pallas_fn))


def make_dense_xla_fn(s):
    """Un-tiled O(N^2) dense pairwise LJ + damped Coulomb XLA reference (for memory scaling)."""
    from jax.scipy.special import erfc
    from proxide.physics.constants import COULOMB_CONSTANT

    n = s["pos"].shape[0]
    box = s["box"]

    def dense_fn(p):
        dr = p[:, None, :] - p[None, :, :]
        scaled = dr / box
        wrap_idx = jnp.sign(scaled) * jnp.floor(jnp.abs(scaled) + 0.5)
        dr = dr - box * wrap_idx
        dist_sq = jnp.sum(dr * dr, axis=-1) + 1e-10
        dist = jnp.sqrt(dist_sq)
        not_self = 1.0 - jnp.eye(n, dtype=p.dtype)
        sig_ij = 0.5 * (s["sigma"][:, None] + s["sigma"][None, :])
        eps_ij = jnp.sqrt(s["eps"][:, None] * s["eps"][None, :])
        u6 = (sig_ij / dist) ** 6
        e_lj = 4.0 * eps_ij * (u6 * u6 - u6) * not_self
        qq = s["q"][:, None] * s["q"][None, :] * COULOMB_CONSTANT
        e_coul = qq * erfc(PME_ALPHA * dist) / dist * not_self
        return 0.5 * jnp.sum(e_lj + e_coul)

    return jax.jit(jax.value_and_grad(dense_fn))


def make_pme_fns(s, interpret: bool):
    """SPME reciprocal+self energy ``value_and_grad``: XLA vs Pallas spread/gather."""
    from prolix.physics.pme import compute_pme_grid_dims, spme_energy_with_forces

    grid_dims = compute_pme_grid_dims(np.asarray(s["box"]), grid_spacing=1.0)

    def make(use_pallas: bool):
        def fn(p):
            return spme_energy_with_forces(
                p, s["q"], s["mask"], s["box"], grid_dims, PME_ALPHA, 4, use_pallas
            )

        return jax.jit(jax.value_and_grad(fn))

    return make(False), make(True)


def build_full_potential_topology(s: dict, seed: int = 0) -> dict:
    """Populate realistic protein+solvent bonded, CMAP, and 1-4 exception tables for size N."""
    from prolix.physics.cmap import precompute_cmap_coefficients

    rng = np.random.default_rng(seed + 17)
    n = int(s["pos"].shape[0])
    pos_np = np.asarray(s["pos"], dtype=np.float32)
    box_np = np.asarray(s["box"], dtype=np.float32)
    idx = np.arange(n, dtype=np.int32)

    def _pbc_dist(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
        dr = pos_np[idx_a] - pos_np[idx_b]
        dr = dr - box_np * np.round(dr / box_np)
        return np.sqrt(np.sum(dr * dr, axis=-1) + 1e-10).astype(np.float32)

    n_bonds = n
    b_i = idx[:n_bonds]
    b_j = (b_i + 1) % n
    bonds = jnp.asarray(np.stack([b_i, b_j], axis=-1), dtype=jnp.int32)
    b_r0 = _pbc_dist(b_i, b_j) + rng.normal(0.0, 0.04, size=n_bonds).astype(np.float32)
    bond_params = jnp.asarray(
        np.column_stack([b_r0, np.full(n_bonds, 300.0, np.float32)])
    )
    bond_mask = jnp.ones((n_bonds,), dtype=jnp.float32)

    n_ub = max(4, n // 4)
    u_i = idx[:n_ub]
    u_j = (u_i + 2) % n
    ub_bonds = jnp.asarray(np.stack([u_i, u_j], axis=-1), dtype=jnp.int32)
    u_r0 = _pbc_dist(u_i, u_j) + rng.normal(0.0, 0.04, size=n_ub).astype(np.float32)
    ub_params = jnp.asarray(
        np.column_stack([u_r0, np.full(n_ub, 50.0, np.float32)])
    )
    ub_mask = jnp.ones((n_ub,), dtype=jnp.float32)

    n_angles = max(8, int(0.7 * n))
    a_i = idx[:n_angles] % n
    angles = jnp.asarray(np.stack([a_i, (a_i + 1) % n, (a_i + 2) % n], axis=-1), dtype=jnp.int32)
    angle_params = jnp.asarray(
        np.column_stack([np.full(n_angles, 1.91, np.float32), np.full(n_angles, 60.0, np.float32)])
    )
    angle_mask = jnp.ones((n_angles,), dtype=jnp.float32)

    n_dih = max(8, int(1.2 * n))
    d_i = np.arange(n_dih, dtype=np.int32) % n
    dihedrals = jnp.asarray(
        np.stack([d_i, (d_i + 1) % n, (d_i + 2) % n, (d_i + 3) % n], axis=-1), dtype=jnp.int32
    )
    dihedral_params = jnp.asarray(
        np.column_stack(
            [
                np.full(n_dih, 2.0, np.float32),
                np.full(n_dih, 0.3, np.float32),
                np.full(n_dih, 1.2, np.float32),
            ]
        )
    )
    dihedral_mask = jnp.ones((n_dih,), dtype=jnp.float32)

    n_imp = max(4, int(0.1 * n))
    im_i = np.arange(n_imp, dtype=np.int32) % n
    impropers = jnp.asarray(
        np.stack([(im_i + 1) % n, im_i, (im_i + 2) % n, (im_i + 3) % n], axis=-1), dtype=jnp.int32
    )
    improper_params = jnp.asarray(
        np.column_stack(
            [
                np.full(n_imp, 2.0, np.float32),
                np.full(n_imp, 3.1415927, np.float32),
                np.full(n_imp, 2.0, np.float32),
            ]
        )
    )
    improper_mask = jnp.ones((n_imp,), dtype=jnp.float32)

    n_cmap = max(4, n // 20)
    c_i = np.arange(n_cmap, dtype=np.int32) % n
    cmap_torsions = jnp.asarray(
        np.stack([c_i, (c_i + 1) % n, (c_i + 2) % n, (c_i + 3) % n, (c_i + 4) % n], axis=-1),
        dtype=jnp.int32,
    )
    cmap_indices = jnp.zeros((n_cmap,), dtype=jnp.int32)
    cmap_mask = jnp.ones((n_cmap,), dtype=jnp.float32)
    raw_grids = jnp.asarray(rng.normal(0.0, 0.5, size=(2, 24, 24)).astype(np.float32))
    cmap_coeffs = precompute_cmap_coefficients(raw_grids)

    n_exc = max(8, int(0.8 * n))
    ex_i = np.arange(n_exc, dtype=np.int32) % n
    exception_pairs = jnp.asarray(np.stack([ex_i, (ex_i + 3) % n], axis=-1), dtype=jnp.int32)
    exception_sigmas = jnp.full((n_exc,), 1.8, dtype=jnp.float32)
    exception_epsilons = jnp.full((n_exc,), 0.08, dtype=jnp.float32)
    exception_chargeprods = jnp.full((n_exc,), -0.04, dtype=jnp.float32)
    exception_mask = jnp.ones((n_exc,), dtype=jnp.float32)

    # Realistic (N, 32) exclusion table matching 1-2, 1-3, and 1-4 chain neighbors
    max_excl = 32
    excl_idx_np = np.full((n, max_excl), -1, dtype=np.int32)
    excl_vdw_np = np.ones((n, max_excl), dtype=np.float32)
    excl_elec_np = np.ones((n, max_excl), dtype=np.float32)
    for delta, col_base, s_v, s_e in ((1, 0, 0.0, 0.0), (2, 2, 0.0, 0.0), (3, 4, 0.5, 0.8333333)):
        excl_idx_np[:, col_base] = (idx + delta) % n
        excl_idx_np[:, col_base + 1] = (idx - delta) % n
        excl_vdw_np[:, col_base : col_base + 2] = s_v
        excl_elec_np[:, col_base : col_base + 2] = s_e

    radii = jnp.asarray(rng.uniform(1.4, 1.85, size=n).astype(np.float32))
    scaled_radii = radii * 0.85

    return {
        **s,
        "excl_idx": jnp.asarray(excl_idx_np),
        "excl_vdw": jnp.asarray(excl_vdw_np),
        "excl_elec": jnp.asarray(excl_elec_np),
        "radii": radii,
        "scaled_radii": scaled_radii,
        "bonds": bonds,
        "bond_params": bond_params,
        "bond_mask": bond_mask,
        "ub_bonds": ub_bonds,
        "ub_params": ub_params,
        "ub_mask": ub_mask,
        "angles": angles,
        "angle_params": angle_params,
        "angle_mask": angle_mask,
        "dihedrals": dihedrals,
        "dihedral_params": dihedral_params,
        "dihedral_mask": dihedral_mask,
        "impropers": impropers,
        "improper_params": improper_params,
        "improper_mask": improper_mask,
        "cmap_torsions": cmap_torsions,
        "cmap_indices": cmap_indices,
        "cmap_mask": cmap_mask,
        "cmap_coeffs": cmap_coeffs,
        "exception_pairs": exception_pairs,
        "exception_sigmas": exception_sigmas,
        "exception_epsilons": exception_epsilons,
        "exception_chargeprods": exception_chargeprods,
        "exception_mask": exception_mask,
    }


def make_full_potential_fns(s: dict, interpret: bool):
    """Full molecular mechanics potential (bonded + CMAP + 1-4 + NL nonbonded + SPME): XLA vs Pallas."""
    from prolix.batched_energy import (
        _angle_energy_masked,
        _bond_energy_masked,
        _cmap_energy_masked,
        _dihedral_energy_masked,
        _exception_energy_masked,
    )
    from prolix.pallas_kernels import (
        encode_nl_exclusions,
        pallas_angles_harmonic,
        pallas_bonded_all_fused,
        pallas_bonds_2body,
        pallas_cmap_torsions,
        pallas_exceptions_14,
        pallas_torsions_4body,
        permute_and_sort_exclusions,
    )
    from prolix.physics import explicit_corrections
    from prolix.physics.pme import compute_pme_grid_dims, spme_background_energy, spme_energy_with_forces

    if "bonds" not in s:
        s = build_full_potential_topology(s)
    box = s["box"]
    disp, _ = space.periodic(box)
    grid_dims = compute_pme_grid_dims(np.asarray(box), grid_spacing=1.0)
    n_atoms = int(s["pos"].shape[0])
    perm_ident = jnp.arange(n_atoms, dtype=jnp.int32)
    excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted = permute_and_sort_exclusions(
        s["excl_idx"], s["excl_vdw"], s["excl_elec"], perm_ident, perm_ident, n_atoms=n_atoms
    )
    nbr_encoded = encode_nl_exclusions(
        s["nbr"], excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted, n_atoms=n_atoms
    )

    def xla_fn(p):
        e_bond = _bond_energy_masked(p, s["bonds"], s["bond_params"], s["bond_mask"], disp)
        e_ub = _bond_energy_masked(p, s["ub_bonds"], s["ub_params"], s["ub_mask"], disp)
        e_ang = _angle_energy_masked(p, s["angles"], s["angle_params"], s["angle_mask"], disp)
        e_dih = _dihedral_energy_masked(p, s["dihedrals"], s["dihedral_params"], s["dihedral_mask"], disp)
        e_imp = _dihedral_energy_masked(p, s["impropers"], s["improper_params"], s["improper_mask"], disp)
        e_cmap = _cmap_energy_masked(
            p, s["cmap_torsions"], s["cmap_indices"], s["cmap_mask"], s["cmap_coeffs"], disp
        )
        e_exc = _exception_energy_masked(
            p,
            s["exception_pairs"],
            s["exception_sigmas"],
            s["exception_epsilons"],
            s["exception_chargeprods"],
            s["exception_mask"],
            disp,
        )
        e_lj = chunked_lj_energy_nl(
            p, s["sigma"], s["eps"], s["excl_idx"], s["excl_vdw"], s["nbr"], disp,
            cutoff=CUTOFF, tile_size=256, switch_width=SWITCH_WIDTH,
        )
        e_c = chunked_coulomb_energy_nl(
            p, s["q"], s["excl_idx"], s["excl_elec"], s["nbr"], disp,
            pme_alpha=PME_ALPHA, cutoff=CUTOFF, tile_size=256,
        )
        e_pme = spme_energy_with_forces(p, s["q"], s["mask"], box, grid_dims, PME_ALPHA, 4, False)
        e_bg = spme_background_energy(s["q"], s["mask"], PME_ALPHA, box)
        e_tail = explicit_corrections.lj_dispersion_tail_energy(box, s["sigma"], s["eps"], CUTOFF, s["mask"])
        return e_bond + e_ub + e_ang + e_dih + e_imp + e_cmap + e_exc + e_lj + e_c + e_pme + e_bg + e_tail

    def pallas_fn(p):
        e_bonded = pallas_bonded_all_fused(
            p,
            bonds=s["bonds"],
            bond_params=s["bond_params"],
            bond_mask=s["bond_mask"],
            urey_bradley_bonds=s["ub_bonds"],
            urey_bradley_params=s["ub_params"],
            urey_bradley_mask=s["ub_mask"],
            angles=s["angles"],
            angle_params=s["angle_params"],
            angle_mask=s["angle_mask"],
            dihedrals=s["dihedrals"],
            dihedral_params=s["dihedral_params"],
            dihedral_mask=s["dihedral_mask"],
            impropers=s["impropers"],
            improper_params=s["improper_params"],
            improper_mask=s["improper_mask"],
            improper_is_harmonic=False,
            exception_pairs=s["exception_pairs"],
            exception_sigmas=s["exception_sigmas"],
            exception_epsilons=s["exception_epsilons"],
            exception_chargeprods=s["exception_chargeprods"],
            exception_mask=s["exception_mask"],
            cmap_torsions=s["cmap_torsions"],
            cmap_indices=s["cmap_indices"],
            cmap_mask=s["cmap_mask"],
            cmap_coeffs=s["cmap_coeffs"],
            box=box,
            dist_eps=1e-10,
            b0_is_ji=True,
            cmap_first_is_psi=True,
            soft_core_lambda=1.0,
            interpret=interpret,
        )
        e_nb = pallas_nonbonded_nl(
            p, s["q"], s["sigma"], s["eps"], s["mask"], nbr_encoded, excl_idx_sorted,
            excl_vdw_sorted, excl_elec_sorted, box=box, pme_alpha=PME_ALPHA,
            cutoff=CUTOFF, lj_switch_width=SWITCH_WIDTH, soft_core_lambda=1.0,
            include_pme_excl_erf=True, excl_mode="pre_encoded",
            scale_14_vdw=0.5, scale_14_elec=0.8333333, interpret=interpret,
        )
        e_pme = spme_energy_with_forces(p, s["q"], s["mask"], box, grid_dims, PME_ALPHA, 4, True)
        e_bg = spme_background_energy(s["q"], s["mask"], PME_ALPHA, box)
        e_tail = explicit_corrections.lj_dispersion_tail_energy(box, s["sigma"], s["eps"], CUTOFF, s["mask"])
        return e_bonded + e_nb + e_pme + e_bg + e_tail

    return jax.jit(jax.value_and_grad(xla_fn)), jax.jit(jax.value_and_grad(pallas_fn))


def make_full_implicit_fns(s: dict, interpret: bool):
    """Full implicit-solvent potential (bonded + CMAP + 1-4 + OBC-II GB/ACE + SASA + NL nonbonded): XLA vs Pallas."""
    from prolix.batched_energy import (
        _angle_energy_masked,
        _bond_energy_masked,
        _cmap_energy_masked,
        _dihedral_energy_masked,
        _exception_energy_masked,
    )
    from prolix.pallas_kernels import (
        encode_nl_exclusions,
        pallas_angles_harmonic,
        pallas_bonded_all_fused,
        pallas_bonds_2body,
        pallas_cmap_torsions,
        pallas_exceptions_14,
        pallas_gb_ace_nl,
        pallas_sasa_nl,
        pallas_torsions_4body,
        permute_and_sort_exclusions,
    )
    from prolix.physics.generalized_born import (
        compute_ace_nonpolar_energy,
        compute_gb_energy_neighbor_list,
        compute_nonpolar_energy,
    )

    if "bonds" not in s:
        s = build_full_potential_topology(s)
    box = s["box"]
    disp, _ = space.periodic(box)
    n_atoms = int(s["pos"].shape[0])
    perm_ident = jnp.arange(n_atoms, dtype=jnp.int32)
    excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted = permute_and_sort_exclusions(
        s["excl_idx"], s["excl_vdw"], s["excl_elec"], perm_ident, perm_ident, n_atoms=n_atoms
    )
    nbr_encoded = encode_nl_exclusions(
        s["nbr"], excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted, n_atoms=n_atoms
    )

    def xla_fn(p):
        e_bond = _bond_energy_masked(p, s["bonds"], s["bond_params"], s["bond_mask"], disp)
        e_ub = _bond_energy_masked(p, s["ub_bonds"], s["ub_params"], s["ub_mask"], disp)
        e_ang = _angle_energy_masked(p, s["angles"], s["angle_params"], s["angle_mask"], disp)
        e_dih = _dihedral_energy_masked(p, s["dihedrals"], s["dihedral_params"], s["dihedral_mask"], disp)
        e_imp = _dihedral_energy_masked(p, s["impropers"], s["improper_params"], s["improper_mask"], disp)
        e_cmap = _cmap_energy_masked(
            p, s["cmap_torsions"], s["cmap_indices"], s["cmap_mask"], s["cmap_coeffs"], disp
        )
        e_exc = _exception_energy_masked(
            p, s["exception_pairs"], s["exception_sigmas"], s["exception_epsilons"],
            s["exception_chargeprods"], s["exception_mask"], disp,
        )
        e_lj = chunked_lj_energy_nl(
            p, s["sigma"], s["eps"], s["excl_idx"], s["excl_vdw"], s["nbr"], disp,
            cutoff=CUTOFF, tile_size=256, switch_width=SWITCH_WIDTH,
        )
        e_c = chunked_coulomb_energy_nl(
            p, s["q"], s["excl_idx"], s["excl_elec"], s["nbr"], disp,
            pme_alpha=0.0, cutoff=CUTOFF, tile_size=256,
        )
        e_gb, br = compute_gb_energy_neighbor_list(
            p, s["q"], s["radii"], s["nbr"], scaled_radii=s["scaled_radii"]
        )
        e_ace = jnp.sum(compute_ace_nonpolar_energy(s["radii"], br) * s["mask"])
        e_sasa = compute_nonpolar_energy(p, s["radii"], neighbor_idx=s["nbr"], atom_mask=s["mask"])
        return e_bond + e_ub + e_ang + e_dih + e_imp + e_cmap + e_exc + e_lj + e_c + e_gb + e_ace + e_sasa

    def pallas_fn(p):
        e_bonded = pallas_bonded_all_fused(
            p,
            bonds=s["bonds"],
            bond_params=s["bond_params"],
            bond_mask=s["bond_mask"],
            urey_bradley_bonds=s["ub_bonds"],
            urey_bradley_params=s["ub_params"],
            urey_bradley_mask=s["ub_mask"],
            angles=s["angles"],
            angle_params=s["angle_params"],
            angle_mask=s["angle_mask"],
            dihedrals=s["dihedrals"],
            dihedral_params=s["dihedral_params"],
            dihedral_mask=s["dihedral_mask"],
            impropers=s["impropers"],
            improper_params=s["improper_params"],
            improper_mask=s["improper_mask"],
            improper_is_harmonic=False,
            exception_pairs=s["exception_pairs"],
            exception_sigmas=s["exception_sigmas"],
            exception_epsilons=s["exception_epsilons"],
            exception_chargeprods=s["exception_chargeprods"],
            exception_mask=s["exception_mask"],
            cmap_torsions=s["cmap_torsions"],
            cmap_indices=s["cmap_indices"],
            cmap_mask=s["cmap_mask"],
            cmap_coeffs=s["cmap_coeffs"],
            box=box,
            dist_eps=1e-10,
            b0_is_ji=True,
            cmap_first_is_psi=True,
            soft_core_lambda=1.0,
            interpret=interpret,
        )
        e_nb = pallas_nonbonded_nl(
            p, s["q"], s["sigma"], s["eps"], s["mask"], nbr_encoded, excl_idx_sorted,
            excl_vdw_sorted, excl_elec_sorted, box=box, pme_alpha=0.0,
            cutoff=CUTOFF, lj_switch_width=SWITCH_WIDTH, soft_core_lambda=1.0,
            include_pme_excl_erf=False, excl_mode="pre_encoded",
            scale_14_vdw=0.5, scale_14_elec=0.8333333, interpret=interpret,
        )
        e_gb_ace = pallas_gb_ace_nl(
            p, s["q"], s["radii"], s["nbr"], atom_mask=s["mask"],
            scaled_radii=s["scaled_radii"], box=box, interpret=interpret,
        )
        e_sasa = pallas_sasa_nl(
            p, s["radii"], s["nbr"], atom_mask=s["mask"], box=box, interpret=interpret
        )
        return e_bonded + e_nb + e_gb_ace + e_sasa

    return jax.jit(jax.value_and_grad(xla_fn)), jax.jit(jax.value_and_grad(pallas_fn))


def _mem_stats() -> dict[str, int] | None:
    try:
        st = jax.devices()[0].memory_stats()
        if not st:
            return None
        return {
            "bytes_in_use": int(st.get("bytes_in_use", 0)),
            "peak_bytes_in_use": int(st.get("peak_bytes_in_use", 0)),
        }
    except Exception:  # noqa: BLE001 - backend without memory stats
        return None


def _peak_mem() -> int | None:
    st = _mem_stats()
    return None if st is None else st["peak_bytes_in_use"]


def time_fn(fn, pos, repeats: int):
    t0 = time.perf_counter()
    out = fn(pos)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    # One untimed post-compile warmup iteration
    jax.block_until_ready(fn(pos))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(pos))
        times.append(time.perf_counter() - t0)
    return out, compile_s, statistics.median(times)


def fit_exponent(ns, ts):
    pairs = [(n, t) for n, t in zip(ns, ts) if n is not None and t is not None and t > 0]
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def _run_isolated_worker(spec: str, repeats: int, interpret: bool) -> dict:
    """Execute one (path, n_atoms) case in a fresh process for uncontaminated peak memory."""
    path_name, n_str = spec.split(":")
    n = int(n_str)
    # Touch device before building tensors to record baseline runtime memory.
    _ = jnp.zeros((1,), dtype=jnp.float32).block_until_ready()
    m0 = _mem_stats()
    base_bytes = m0["bytes_in_use"] if m0 else None

    needs_nl = path_name in (
        "nl_xla",
        "nl_pallas",
        "nonbonded_xla",
        "nonbonded_pallas",
        "full_potential_xla",
        "full_potential_pallas",
        "full_implicit_xla",
        "full_implicit_pallas",
    )
    s = build_system(n, include_nl=needs_nl)
    if path_name in ("full_potential_xla", "full_potential_pallas", "full_implicit_xla", "full_implicit_pallas"):
        s = build_full_potential_topology(s)
    jax.block_until_ready(s["pos"])
    if s["nbr"] is not None:
        jax.block_until_ready(s["nbr"])
    m1 = _mem_stats()
    input_bytes = m1["bytes_in_use"] if m1 else None

    if path_name in ("nl_xla", "nonbonded_xla"):
        fn, _ = make_fns(s, interpret)
    elif path_name in ("nl_pallas", "nonbonded_pallas"):
        _, fn = make_fns(s, interpret)
    elif path_name == "full_potential_xla":
        fn, _ = make_full_potential_fns(s, interpret)
    elif path_name == "full_potential_pallas":
        _, fn = make_full_potential_fns(s, interpret)
    elif path_name == "full_implicit_xla":
        fn, _ = make_full_implicit_fns(s, interpret)
    elif path_name == "full_implicit_pallas":
        _, fn = make_full_implicit_fns(s, interpret)
    elif path_name == "flash_xla":
        fn, _ = make_flash_fns(s, interpret)
    elif path_name == "flash_pallas":
        _, fn = make_flash_fns(s, interpret)
    elif path_name == "pme_xla":
        fn, _ = make_pme_fns(s, interpret)
    elif path_name == "pme_pallas":
        _, fn = make_pme_fns(s, interpret)
    elif path_name == "dense_xla":
        fn = make_dense_xla_fn(s)
    else:
        raise ValueError(f"Unknown worker path: {path_name}")

    out, compile_s, median_s = time_fn(fn, s["pos"], repeats)
    m2 = _mem_stats()
    peak_bytes = m2["peak_bytes_in_use"] if m2 else None
    working_peak = (peak_bytes - base_bytes) if (peak_bytes is not None and base_bytes is not None) else None
    activation_peak = (peak_bytes - input_bytes) if (peak_bytes is not None and input_bytes is not None) else None
    return {
        "path": path_name,
        "n_atoms": n,
        "k": int(s["k"]),
        "compile_s": compile_s,
        "median_s": median_s,
        "baseline_bytes": base_bytes,
        "input_bytes": input_bytes,
        "peak_bytes": peak_bytes,
        "working_peak_bytes": working_peak,
        "activation_peak_bytes": activation_peak,
        "energy": float(out[0]),
    }


def _spawn_worker(path_name: str, n: int, repeats: int, interpret: bool) -> dict:
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Bypass BFC 64 MiB arena rounding so peak_bytes_in_use reflects exact tensor/activation HBM.
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        f"{path_name}:{n}",
        "--repeats",
        str(repeats),
    ]
    if interpret:
        cmd.append("--interpret")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {
            "path": path_name,
            "n_atoms": n,
            "error": (proc.stderr or proc.stdout)[-400:],
            "peak_bytes": None,
            "working_peak_bytes": None,
            "activation_peak_bytes": None,
            "median_s": None,
        }
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    return json.loads(lines[-1])


def bench_mem_scaling(sizes: list[int], repeats: int, interpret: bool) -> tuple[list[dict], dict]:
    """Per-process memory scaling across dense O(N^2), Flash O(N*T)/O(N), and NL O(N*K)."""
    rows = []
    paths = ("dense_xla", "flash_xla", "flash_pallas", "nl_xla", "nl_pallas")
    for n in sizes:
        row: dict = {"n_atoms": n}
        for p in paths:
            # Skip un-tiled dense O(N^2) past 16k and O(N^2)-compute flash past 32k so runs stay fast.
            if p == "dense_xla" and n > 16384:
                continue
            if p in ("flash_xla", "flash_pallas") and n > 32768:
                continue
            res = _spawn_worker(p, n, repeats, interpret)
            if "k" in res and res["k"]:
                row["k"] = res["k"]
            row[p] = {
                "median_s": res.get("median_s"),
                "compile_s": res.get("compile_s"),
                "peak_bytes": res.get("peak_bytes"),
                "working_peak_bytes": res.get("working_peak_bytes"),
                "activation_peak_bytes": res.get("activation_peak_bytes"),
                "error": res.get("error"),
            }
        print(json.dumps(row), flush=True)
        rows.append(row)

    summary = {}
    for p in paths:
        ns_p = [r["n_atoms"] for r in rows if p in r and r[p].get("working_peak_bytes")]
        wp_p = [r[p]["working_peak_bytes"] for r in rows if p in r and r[p].get("working_peak_bytes")]
        ap_p = [r[p]["activation_peak_bytes"] for r in rows if p in r and r[p].get("activation_peak_bytes")]
        ts_p = [r[p]["median_s"] for r in rows if p in r and r[p].get("median_s")]
        summary[f"{p}_working_mem_exponent"] = fit_exponent(ns_p, wp_p)
        summary[f"{p}_activation_mem_exponent"] = fit_exponent(ns_p, ap_p)
        summary[f"{p}_time_exponent"] = fit_exponent(ns_p, ts_p)
    return rows, summary


def bench_nl_build(sizes: list[int], repeats: int, interpret: bool) -> list[dict]:
    """Neighbor-list (jax_md cell list) rebuild cost vs one Pallas nonbonded step."""
    from prolix.physics.neighbor_list import make_neighbor_list_fn

    rows = []
    for n in sizes:
        s = build_system(n)
        disp, _ = space.periodic(s["box"])
        nl_fns = make_neighbor_list_fn(disp, s["box"], CUTOFF)
        nbrs = nl_fns.allocate(s["pos"])
        upd = jax.jit(nl_fns.update)
        moved = s["pos"] + jnp.float32(0.4)
        _, _, t_upd = time_fn(lambda p: upd(p, nbrs).idx, moved, repeats)
        _, _, t_chk = time_fn(lambda p: upd(p, nbrs).idx, s["pos"], repeats)
        _, pal_vg = make_fns(s, interpret)
        _, _, t_nb = time_fn(pal_vg, s["pos"], repeats)
        row = {
            "n_atoms": n,
            "nl_capacity_k": int(nbrs.idx.shape[1]),
            "nl_update_median_s": t_upd,
            "nl_skin_check_only_median_s": t_chk,
            "pallas_nonbonded_median_s": t_nb,
            "rebuild_over_step": t_upd / t_nb,
        }
        for steps in (10, 20, 50):
            row[f"amortized_ratio_{steps}"] = (t_upd / steps) / (t_nb + t_upd / steps)
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def bench_nl_skin(sizes: list[int], repeats: int, interpret: bool) -> list[dict]:
    """Sweep NL skin (dr_threshold) and capacity_multiplier vs amortized step cost."""
    from prolix.physics.neighbor_list import make_neighbor_list_fn

    # Typical thermal RMS max-atom displacement per 2 fs step in 300 K water (~0.015 A/step).
    step_disp_a = 0.015
    rows = []
    for n in sizes:
        s_base = build_system(n, include_nl=False)
        disp, _ = space.periodic(s_base["box"])
        for dr_threshold in (0.5, 1.0, 1.5, 2.0):
            for cap_mult in (1.15, 1.25, 1.50):
                nl_fns = make_neighbor_list_fn(
                    disp, s_base["box"], CUTOFF,
                    dr_threshold=dr_threshold, capacity_multiplier=cap_mult,
                )
                nbrs = nl_fns.allocate(s_base["pos"])
                k_raw = int(nbrs.idx.shape[1])
                k_pad = int(np.ceil(max(k_raw, 1) / 8.0) * 8)
                nbr_pad = jnp.pad(
                    nbrs.idx, ((0, 0), (0, k_pad - k_raw)), constant_values=n
                )
                s = {**s_base, "nbr": nbr_pad, "k": k_pad}
                upd = jax.jit(nl_fns.update)
                shift = jnp.float32(dr_threshold * 0.6)
                moved = s["pos"] + shift
                _, _, t_upd = time_fn(lambda p: upd(p, nbrs).idx, moved, repeats)
                _, pal_vg = make_fns(s, interpret)
                _, _, t_nb = time_fn(pal_vg, s["pos"], repeats)
                est_steps_between_rebuilds = max(1.0, (dr_threshold / 2.0) / step_disp_a)
                amortized_step_s = t_nb + (t_upd / est_steps_between_rebuilds)
                row = {
                    "n_atoms": n,
                    "dr_threshold": dr_threshold,
                    "capacity_multiplier": cap_mult,
                    "nl_capacity_k": k_pad,
                    "est_steps_per_rebuild": round(est_steps_between_rebuilds, 1),
                    "nl_update_median_s": t_upd,
                    "pallas_nonbonded_median_s": t_nb,
                    "amortized_step_median_s": amortized_step_s,
                }
                print(json.dumps(row), flush=True)
                rows.append(row)
    return rows


def bench_bonded(sizes: list[int], repeats: int, interpret: bool) -> tuple[list[dict], dict]:
    """Compare XLA bonded vs Pallas v1 (7 separate launches) vs Pallas v2 fused megakernel (1 launch)."""
    from prolix.batched_energy import (
        _angle_energy_masked,
        _bond_energy_masked,
        _cmap_energy_masked,
        _dihedral_energy_masked,
        _exception_energy_masked,
    )
    from prolix.pallas_kernels import (
        pallas_angles_harmonic,
        pallas_bonded_all_fused,
        pallas_bonds_2body,
        pallas_cmap_torsions,
        pallas_exceptions_14,
        pallas_torsions_4body,
    )

    rows = []
    for n in sizes:
        s = build_full_potential_topology(build_system(n, include_nl=False))
        box = s["box"]
        disp, _ = space.periodic(box)

        def xla_bonded(p):
            e_bond = _bond_energy_masked(p, s["bonds"], s["bond_params"], s["bond_mask"], disp)
            e_ub = _bond_energy_masked(p, s["ub_bonds"], s["ub_params"], s["ub_mask"], disp)
            e_ang = _angle_energy_masked(p, s["angles"], s["angle_params"], s["angle_mask"], disp)
            e_dih = _dihedral_energy_masked(p, s["dihedrals"], s["dihedral_params"], s["dihedral_mask"], disp)
            e_imp = _dihedral_energy_masked(p, s["impropers"], s["improper_params"], s["improper_mask"], disp)
            e_cmap = _cmap_energy_masked(
                p, s["cmap_torsions"], s["cmap_indices"], s["cmap_mask"], s["cmap_coeffs"], disp
            )
            e_exc = _exception_energy_masked(
                p, s["exception_pairs"], s["exception_sigmas"], s["exception_epsilons"],
                s["exception_chargeprods"], s["exception_mask"], disp,
            )
            return e_bond + e_ub + e_ang + e_dih + e_imp + e_cmap + e_exc

        def pallas_v1_bonded(p):
            e_bond = pallas_bonds_2body(
                p, s["bonds"], s["bond_params"], s["bond_mask"], box=box, dist_eps=1e-10, interpret=interpret
            )
            e_ub = pallas_bonds_2body(
                p, s["ub_bonds"], s["ub_params"], s["ub_mask"], box=box, dist_eps=1e-10, interpret=interpret
            )
            e_ang = pallas_angles_harmonic(
                p, s["angles"], s["angle_params"], s["angle_mask"], box=box, dist_eps=1e-10, interpret=interpret
            )
            e_dih = pallas_torsions_4body(
                p, s["dihedrals"], s["dihedral_params"], s["dihedral_mask"],
                box=box, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
            e_imp = pallas_torsions_4body(
                p, s["impropers"], s["improper_params"], s["improper_mask"],
                box=box, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
            e_cmap = pallas_cmap_torsions(
                p, s["cmap_torsions"], s["cmap_indices"], s["cmap_mask"], s["cmap_coeffs"],
                box=box, first_is_psi=True, interpret=interpret,
            )
            e_exc = pallas_exceptions_14(
                p, s["exception_pairs"], s["exception_sigmas"], s["exception_epsilons"],
                s["exception_chargeprods"], s["exception_mask"], box=box, soft_core_lambda=1.0, interpret=interpret,
            )
            return e_bond + e_ub + e_ang + e_dih + e_imp + e_cmap + e_exc

        def pallas_v2_megakernel(p, return_forces: bool = False):
            return pallas_bonded_all_fused(
                p,
                bonds=s["bonds"],
                bond_params=s["bond_params"],
                bond_mask=s["bond_mask"],
                urey_bradley_bonds=s["ub_bonds"],
                urey_bradley_params=s["ub_params"],
                urey_bradley_mask=s["ub_mask"],
                angles=s["angles"],
                angle_params=s["angle_params"],
                angle_mask=s["angle_mask"],
                dihedrals=s["dihedrals"],
                dihedral_params=s["dihedral_params"],
                dihedral_mask=s["dihedral_mask"],
                impropers=s["impropers"],
                improper_params=s["improper_params"],
                improper_mask=s["improper_mask"],
                improper_is_harmonic=False,
                exception_pairs=s["exception_pairs"],
                exception_sigmas=s["exception_sigmas"],
                exception_epsilons=s["exception_epsilons"],
                exception_chargeprods=s["exception_chargeprods"],
                exception_mask=s["exception_mask"],
                cmap_torsions=s["cmap_torsions"],
                cmap_indices=s["cmap_indices"],
                cmap_mask=s["cmap_mask"],
                cmap_coeffs=s["cmap_coeffs"],
                box=box,
                dist_eps=1e-10,
                b0_is_ji=True,
                cmap_first_is_psi=True,
                soft_core_lambda=1.0,
                return_forces=return_forces,
                interpret=interpret,
            )

        xla_vg = jax.jit(jax.value_and_grad(xla_bonded))
        v1_vg = jax.jit(jax.value_and_grad(pallas_v1_bonded))
        v2_vg = jax.jit(jax.value_and_grad(lambda p: pallas_v2_megakernel(p, return_forces=False)))
        v2_rf = jax.jit(lambda p: pallas_v2_megakernel(p, return_forces=True))

        out_x, c_x, t_x = time_fn(xla_vg, s["pos"], repeats)
        out_v1, c_v1, t_v1 = time_fn(v1_vg, s["pos"], repeats)
        out_v2, c_v2, t_v2 = time_fn(v2_vg, s["pos"], repeats)
        out_rf, c_rf, t_rf = time_fn(v2_rf, s["pos"], repeats)

        row = {
            "n_atoms": n,
            "xla_compile_s": c_x,
            "xla_median_s": t_x,
            "pallas_v1_7launch_compile_s": c_v1,
            "pallas_v1_7launch_median_s": t_v1,
            "pallas_v2_megakernel_compile_s": c_v2,
            "pallas_v2_megakernel_median_s": t_v2,
            "pallas_v2_return_forces_median_s": t_rf,
            "speedup_v2_vs_xla": t_x / t_v2,
            "speedup_v2_vs_v1": t_v1 / t_v2,
            "speedup_v2_rf_vs_xla": t_x / t_rf,
            "max_abs_dE_v2_vs_xla": float(abs(out_v2[0] - out_x[0])),
            "max_abs_dF_v2_vs_xla": float(jnp.max(jnp.abs(out_v2[1] - out_x[1]))),
            "max_abs_dF_rf_vs_xla": float(jnp.max(jnp.abs(-out_rf[1] - out_x[1]))),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)

    ns = [r["n_atoms"] for r in rows]
    summary = {
        "xla_scaling_exponent": fit_exponent(ns, [r["xla_median_s"] for r in rows]),
        "pallas_v1_scaling_exponent": fit_exponent(ns, [r["pallas_v1_7launch_median_s"] for r in rows]),
        "pallas_v2_megakernel_scaling_exponent": fit_exponent(ns, [r["pallas_v2_megakernel_median_s"] for r in rows]),
    }
    return rows, summary


def bench_openmm_comparison(sizes: list[int], repeats: int, interpret: bool) -> tuple[list[dict], dict]:
    """Apples-to-apples comparison: Prolix XLA vs Prolix Pallas v1 vs Prolix Pallas v2 vs OpenMM."""
    import openmm
    import openmm.unit as omm_unit
    from prolix.batched_energy import (
        _angle_energy_masked,
        _bond_energy_masked,
        _dihedral_energy_masked,
        _exception_energy_masked,
    )
    from prolix.pallas_kernels import (
        encode_nl_exclusions,
        pallas_angles_harmonic,
        pallas_bonded_all_fused,
        pallas_bonds_2body,
        pallas_exceptions_14,
        pallas_torsions_4body,
        permute_and_sort_exclusions,
    )
    from prolix.physics.pme import compute_pme_grid_dims, spme_background_energy, spme_energy_with_forces

    # Pick highest-throughput OpenMM platform available (CUDA > OpenCL > CPU)
    avail = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    if "CUDA" in avail:
        omm_platform_name = "CUDA"
        omm_props = {"Precision": "single"}
    elif "OpenCL" in avail:
        omm_platform_name = "OpenCL"
        omm_props = {"Precision": "single"}
    elif "CPU" in avail:
        omm_platform_name = "CPU"
        omm_props = {}
    else:
        omm_platform_name = avail[0]
        omm_props = {}

    rows = []
    for n in sizes:
        s = build_full_potential_topology(build_system(n, include_nl=True))
        box = s["box"]
        box_len_a = float(box[0])
        disp, _ = space.periodic(box)
        grid_dims = compute_pme_grid_dims(np.asarray(box), grid_spacing=1.0)
        kx, ky, kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])

        perm_ident = jnp.arange(n, dtype=jnp.int32)
        excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted = permute_and_sort_exclusions(
            s["excl_idx"], s["excl_vdw"], s["excl_elec"], perm_ident, perm_ident, n_atoms=n
        )
        nbr_encoded = encode_nl_exclusions(
            s["nbr"], excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted, n_atoms=n
        )

        # Common physical subset shared by OpenMM standard forces & Prolix:
        # Harmonic bonds + Urey-Bradley + Harmonic angles + Periodic dihedrals + Periodic impropers +
        # 1-4 exceptions + Direct-space LJ (switched) + PME (order=4 in Prolix, locked grid & alpha)
        def prolix_xla_fn(p):
            e_bond = _bond_energy_masked(p, s["bonds"], s["bond_params"], s["bond_mask"], disp)
            e_ub = _bond_energy_masked(p, s["ub_bonds"], s["ub_params"], s["ub_mask"], disp)
            e_ang = _angle_energy_masked(p, s["angles"], s["angle_params"], s["angle_mask"], disp)
            e_dih = _dihedral_energy_masked(p, s["dihedrals"], s["dihedral_params"], s["dihedral_mask"], disp)
            e_imp = _dihedral_energy_masked(p, s["impropers"], s["improper_params"], s["improper_mask"], disp)
            e_exc = _exception_energy_masked(
                p, s["exception_pairs"], s["exception_sigmas"], s["exception_epsilons"],
                s["exception_chargeprods"], s["exception_mask"], disp,
            )
            e_lj = chunked_lj_energy_nl(
                p, s["sigma"], s["eps"], s["excl_idx"], s["excl_vdw"], s["nbr"], disp,
                cutoff=CUTOFF, tile_size=256, switch_width=SWITCH_WIDTH,
            )
            e_c = chunked_coulomb_energy_nl(
                p, s["q"], s["excl_idx"], s["excl_elec"], s["nbr"], disp,
                pme_alpha=PME_ALPHA, cutoff=CUTOFF, tile_size=256,
            )
            e_pme = spme_energy_with_forces(p, s["q"], s["mask"], box, grid_dims, PME_ALPHA, 4, False)
            e_bg = spme_background_energy(s["q"], s["mask"], PME_ALPHA, box)
            return e_bond + e_ub + e_ang + e_dih + e_imp + e_exc + e_lj + e_c + e_pme + e_bg

        def prolix_pallas_v1_fn(p):
            e_bond = pallas_bonds_2body(
                p, s["bonds"], s["bond_params"], s["bond_mask"], box=box, dist_eps=1e-10, interpret=interpret
            )
            e_ub = pallas_bonds_2body(
                p, s["ub_bonds"], s["ub_params"], s["ub_mask"], box=box, dist_eps=1e-10, interpret=interpret
            )
            e_ang = pallas_angles_harmonic(
                p, s["angles"], s["angle_params"], s["angle_mask"], box=box, dist_eps=1e-10, interpret=interpret
            )
            e_dih = pallas_torsions_4body(
                p, s["dihedrals"], s["dihedral_params"], s["dihedral_mask"],
                box=box, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
            e_imp = pallas_torsions_4body(
                p, s["impropers"], s["improper_params"], s["improper_mask"],
                box=box, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
            e_exc = pallas_exceptions_14(
                p, s["exception_pairs"], s["exception_sigmas"], s["exception_epsilons"],
                s["exception_chargeprods"], s["exception_mask"], box=box, soft_core_lambda=1.0, interpret=interpret,
            )
            e_nb = pallas_nonbonded_nl(
                p, s["q"], s["sigma"], s["eps"], s["mask"], s["nbr"], s["excl_idx"],
                s["excl_vdw"], s["excl_elec"], box=box, pme_alpha=PME_ALPHA,
                cutoff=CUTOFF, lj_switch_width=SWITCH_WIDTH, soft_core_lambda=1.0,
                include_pme_excl_erf=True, excl_mode="table", interpret=interpret,
            )
            e_pme = spme_energy_with_forces(p, s["q"], s["mask"], box, grid_dims, PME_ALPHA, 4, True)
            e_bg = spme_background_energy(s["q"], s["mask"], PME_ALPHA, box)
            return e_bond + e_ub + e_ang + e_dih + e_imp + e_exc + e_nb + e_pme + e_bg

        def prolix_pallas_v2_fn(p):
            e_bonded = pallas_bonded_all_fused(
                p,
                bonds=s["bonds"],
                bond_params=s["bond_params"],
                bond_mask=s["bond_mask"],
                urey_bradley_bonds=s["ub_bonds"],
                urey_bradley_params=s["ub_params"],
                urey_bradley_mask=s["ub_mask"],
                angles=s["angles"],
                angle_params=s["angle_params"],
                angle_mask=s["angle_mask"],
                dihedrals=s["dihedrals"],
                dihedral_params=s["dihedral_params"],
                dihedral_mask=s["dihedral_mask"],
                impropers=s["impropers"],
                improper_params=s["improper_params"],
                improper_mask=s["improper_mask"],
                improper_is_harmonic=False,
                exception_pairs=s["exception_pairs"],
                exception_sigmas=s["exception_sigmas"],
                exception_epsilons=s["exception_epsilons"],
                exception_chargeprods=s["exception_chargeprods"],
                exception_mask=s["exception_mask"],
                box=box,
                dist_eps=1e-10,
                b0_is_ji=True,
                soft_core_lambda=1.0,
                interpret=interpret,
            )
            e_nb = pallas_nonbonded_nl(
                p, s["q"], s["sigma"], s["eps"], s["mask"], nbr_encoded, excl_idx_sorted,
                excl_vdw_sorted, excl_elec_sorted, box=box, pme_alpha=PME_ALPHA,
                cutoff=CUTOFF, lj_switch_width=SWITCH_WIDTH, soft_core_lambda=1.0,
                include_pme_excl_erf=True, excl_mode="pre_encoded",
                scale_14_vdw=0.5, scale_14_elec=0.8333333, interpret=interpret,
            )
            e_pme = spme_energy_with_forces(p, s["q"], s["mask"], box, grid_dims, PME_ALPHA, 4, True)
            e_bg = spme_background_energy(s["q"], s["mask"], PME_ALPHA, box)
            return e_bonded + e_nb + e_pme + e_bg

        xla_vg = jax.jit(jax.value_and_grad(prolix_xla_fn))
        v1_vg = jax.jit(jax.value_and_grad(prolix_pallas_v1_fn))
        v2_vg = jax.jit(jax.value_and_grad(prolix_pallas_v2_fn))

        out_x, c_x, t_x = time_fn(xla_vg, s["pos"], repeats)
        out_v1, c_v1, t_v1 = time_fn(v1_vg, s["pos"], repeats)
        out_v2, c_v2, t_v2 = time_fn(v2_vg, s["pos"], repeats)

        # Build OpenMM System with identical settings (in nm, kJ/mol)
        kcal_to_kj = 4.184
        a_to_nm = 0.1
        omm_sys = openmm.System()
        box_nm = box_len_a * a_to_nm
        omm_sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3(box_nm, 0.0, 0.0),
            openmm.Vec3(0.0, box_nm, 0.0),
            openmm.Vec3(0.0, 0.0, box_nm),
        )
        for _ in range(n):
            omm_sys.addParticle(12.0)

        # HarmonicBondForce (bonds + Urey-Bradley): Prolix E = 0.5 * k * (r - r0)^2 kcal/mol
        hb_force = openmm.HarmonicBondForce()
        hb_force.setUsesPeriodicBoundaryConditions(True)
        for b_arr, bp_arr in ((np.asarray(s["bonds"]), np.asarray(s["bond_params"])), (np.asarray(s["ub_bonds"]), np.asarray(s["ub_params"]))):
            for (ai, aj), (r0_a, k_kcal_a2) in zip(b_arr, bp_arr):
                hb_force.addBond(int(ai), int(aj), float(r0_a) * a_to_nm, float(k_kcal_a2) * kcal_to_kj / (a_to_nm * a_to_nm))
        omm_sys.addForce(hb_force)

        # HarmonicAngleForce: Prolix E = 0.5 * k * (theta - theta0)^2 kcal/mol
        ha_force = openmm.HarmonicAngleForce()
        ha_force.setUsesPeriodicBoundaryConditions(True)
        for (ai, aj, ak), (th0, k_kcal) in zip(np.asarray(s["angles"]), np.asarray(s["angle_params"])):
            ha_force.addAngle(int(ai), int(aj), int(ak), float(th0), float(k_kcal) * kcal_to_kj)
        omm_sys.addForce(ha_force)

        # PeriodicTorsionForce: Prolix E = k * (1 + cos(periodicity * phi - phase))
        pt_force = openmm.PeriodicTorsionForce()
        pt_force.setUsesPeriodicBoundaryConditions(True)
        for t_arr, tp_arr in ((np.asarray(s["dihedrals"]), np.asarray(s["dihedral_params"])), (np.asarray(s["impropers"]), np.asarray(s["improper_params"]))):
            for (ai, aj, ak, al), (per, phase, k_kcal) in zip(t_arr, tp_arr):
                pt_force.addTorsion(int(ai), int(aj), int(ak), int(al), int(round(float(per))), float(phase), float(k_kcal) * kcal_to_kj)
        omm_sys.addForce(pt_force)

        # NonbondedForce (PME with locked alpha, (kx, ky, kz), cutoff, switch)
        nb_force = openmm.NonbondedForce()
        nb_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        nb_force.setCutoffDistance(CUTOFF * a_to_nm)
        nb_force.setUseSwitchingFunction(True)
        nb_force.setSwitchingDistance((CUTOFF - SWITCH_WIDTH) * a_to_nm)
        nb_force.setUseDispersionCorrection(False)
        nb_force.setPMEParameters(PME_ALPHA / a_to_nm, kx, ky, kz)

        q_np = np.asarray(s["q"])
        sig_np = np.asarray(s["sigma"])
        eps_np = np.asarray(s["eps"])
        for i in range(n):
            nb_force.addParticle(float(q_np[i]), float(sig_np[i]) * a_to_nm, float(eps_np[i]) * kcal_to_kj)

        # Add full exclusions (1-2 and 1-3) and 1-4 scaled exceptions + explicit exception_pairs
        # In build_full_potential_topology, 1-2 is i+1, 1-3 is i+2, and 1-4 is i+3 (with 0.5/0.8333333 in NL + explicit exception_pairs)
        seen_pairs: dict[tuple[int, int], tuple[float, float, float]] = {}
        for i in range(n):
            for delta in (1, 2):
                j = (i + delta) % n
                pair = (min(i, j), max(i, j))
                seen_pairs[pair] = (0.0, 0.1, 0.0)
            j14 = (i + 3) % n
            pair14 = (min(i, j14), max(i, j14))
            sig_ij = 0.5 * (float(sig_np[i]) + float(sig_np[j14])) * a_to_nm
            eps_ij = float(np.sqrt(eps_np[i] * eps_np[j14])) * 0.5 * kcal_to_kj
            qp_ij = float(q_np[i] * q_np[j14]) * 0.8333333
            seen_pairs[pair14] = (qp_ij, sig_ij, eps_ij)

        for (ai, aj), sig_e, eps_e, qp_e in zip(
            np.asarray(s["exception_pairs"]),
            np.asarray(s["exception_sigmas"]),
            np.asarray(s["exception_epsilons"]),
            np.asarray(s["exception_chargeprods"]),
        ):
            pair = (min(int(ai), int(aj)), max(int(ai), int(aj)))
            prev_qp, prev_sig, prev_eps = seen_pairs.get(pair, (0.0, float(sig_e) * a_to_nm, 0.0))
            seen_pairs[pair] = (prev_qp + float(qp_e), float(sig_e) * a_to_nm, prev_eps + float(eps_e) * kcal_to_kj)

        for (p_i, p_j), (qp_v, sig_v, eps_v) in seen_pairs.items():
            nb_force.addException(p_i, p_j, qp_v, sig_v, eps_v, True)

        omm_sys.addForce(nb_force)

        pos_nm = np.asarray(s["pos"], dtype=np.float64) * a_to_nm
        ext_omm_py = os.environ.get(
            "OPENMM_PYTHON",
            "/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm/bin/python",
        )
        n_rollout = 20
        t_omm_step_mixed = None
        t_omm_rollout_mixed = None

        if omm_platform_name != "CUDA" and os.path.isfile(ext_omm_py):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpd:
                xml_p = os.path.join(tmpd, "sys.xml")
                pos_p = os.path.join(tmpd, "pos.npy")
                frc_p = os.path.join(tmpd, "forces.npy")
                with open(xml_p, "w", encoding="utf-8") as fh:
                    fh.write(openmm.XmlSerializer.serialize(omm_sys))
                np.save(pos_p, pos_nm)
                worker_code = (
                    "import json, sys, time, statistics, numpy as np, openmm, openmm.unit as u\n"
                    "xml_p, pos_p, frc_p, repeats, n_roll = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])\n"
                    "with open(xml_p) as f: sys_obj = openmm.XmlSerializer.deserialize(f.read())\n"
                    "pos_nm = np.load(pos_p)\n"
                    "plat = openmm.Platform.getPlatformByName('CUDA')\n"
                    "out = {'platform': 'CUDA'}\n"
                    "for prec in ('single', 'mixed'):\n"
                    "    integ = openmm.LangevinMiddleIntegrator(300.0*u.kelvin, 1.0/u.picosecond, 0.001*u.picoseconds)\n"
                    "    ctx = openmm.Context(sys_obj, integ, plat, {'Precision': prec})\n"
                    "    ctx.setPositions(pos_nm)\n"
                    "    st0 = ctx.getState(getEnergy=True, getForces=True)\n"
                    "    if prec == 'single':\n"
                    "        out['e_kcal'] = float(st0.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole))\n"
                    "        np.save(frc_p, st0.getForces(asNumpy=True).value_in_unit(u.kilocalories_per_mole / u.angstrom))\n"
                    "    ts = []\n"
                    "    for _ in range(repeats):\n"
                    "        t0 = time.perf_counter()\n"
                    "        _ = ctx.getState(getEnergy=True, getForces=True)\n"
                    "        ts.append(time.perf_counter() - t0)\n"
                    "    out[f't_step_{prec}'] = statistics.median(ts)\n"
                    "    integ.step(2)\n"
                    "    t0 = time.perf_counter()\n"
                    "    integ.step(n_roll)\n"
                    "    _ = ctx.getState(getPositions=True)\n"
                    "    out[f't_roll_{prec}'] = (time.perf_counter() - t0) / n_roll\n"
                    "    del ctx, integ\n"
                    "print(json.dumps(out))\n"
                )
                env_sub = os.environ.copy()
                omm_root = os.path.dirname(os.path.dirname(ext_omm_py))
                env_sub["OPENMM_PLUGIN_DIR"] = os.path.join(omm_root, "lib", "plugins")
                proc = subprocess.run(
                    [ext_omm_py, "-c", worker_code, xml_p, pos_p, frc_p, str(repeats), str(n_rollout)],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env_sub,
                )
                omm_res = json.loads(proc.stdout.strip().splitlines()[-1])
                omm_platform_name = "CUDA"
                e_omm_kcal = float(omm_res["e_kcal"])
                f_omm_kcal_a = np.load(frc_p)
                t_omm_step = float(omm_res["t_step_single"])
                t_omm_rollout_per_step = float(omm_res["t_roll_single"])
                t_omm_step_mixed = float(omm_res["t_step_mixed"])
                t_omm_rollout_mixed = float(omm_res["t_roll_mixed"])
        else:
            integrator = openmm.LangevinMiddleIntegrator(
                300.0 * omm_unit.kelvin,
                1.0 / omm_unit.picosecond,
                0.001 * omm_unit.picoseconds,
            )
            platform_obj = openmm.Platform.getPlatformByName(omm_platform_name)
            ctx = openmm.Context(omm_sys, integrator, platform_obj, omm_props)
            ctx.setPositions(pos_nm)

            # Warmup + time OpenMM single-step energy+forces
            st0 = ctx.getState(getEnergy=True, getForces=True)
            e_omm_kcal = st0.getPotentialEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
            f_omm_kcal_a = st0.getForces(asNumpy=True).value_in_unit(
                omm_unit.kilocalories_per_mole / omm_unit.angstrom
            )
            omm_times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = ctx.getState(getEnergy=True, getForces=True)
                omm_times.append(time.perf_counter() - t0)
            t_omm_step = statistics.median(omm_times)

            # Also time a 20-step integration rollout in OpenMM vs Prolix lax.scan
            integrator.step(2)
            t0 = time.perf_counter()
            integrator.step(n_rollout)
            _ = ctx.getState(getPositions=True)
            t_omm_rollout_per_step = (time.perf_counter() - t0) / n_rollout

        @jax.jit
        def prolix_v2_rollout(p0):
            def body(_, p_cur):
                _, g = jax.value_and_grad(prolix_pallas_v2_fn)(p_cur)
                return p_cur - jnp.float32(1e-5) * g
            return jax.lax.fori_loop(0, n_rollout, body, p0)

        jax.block_until_ready(prolix_v2_rollout(s["pos"]))
        t0 = time.perf_counter()
        jax.block_until_ready(prolix_v2_rollout(s["pos"]))
        t_v2_rollout_per_step = (time.perf_counter() - t0) / n_rollout

        # Convert step time (dt = 2 fs = 2e-6 ns) to ns/day: (0.002 ps * 86400 s/day) / step_s = 172.8 / step_s
        ns_per_day_xla = 0.1728 / t_x
        ns_per_day_v1 = 0.1728 / t_v1
        ns_per_day_v2 = 0.1728 / t_v2
        ns_per_day_v2_rollout = 0.1728 / t_v2_rollout_per_step
        ns_per_day_omm_rollout = 0.1728 / t_omm_rollout_per_step
        ns_per_day_omm_rollout_mixed = (
            0.1728 / t_omm_rollout_mixed if t_omm_rollout_mixed is not None else None
        )

        row = {
            "n_atoms": n,
            "k": int(s["k"]),
            "pme_grid": [kx, ky, kz],
            "openmm_platform": omm_platform_name,
            "xla_median_s": t_x,
            "pallas_v1_median_s": t_v1,
            "pallas_v2_median_s": t_v2,
            "pallas_v2_rollout_per_step_s": t_v2_rollout_per_step,
            "openmm_getstate_median_s": t_omm_step,
            "openmm_rollout_per_step_s": t_omm_rollout_per_step,
            "openmm_getstate_mixed_median_s": t_omm_step_mixed,
            "openmm_rollout_mixed_per_step_s": t_omm_rollout_mixed,
            "ns_per_day_xla_2fs": ns_per_day_xla,
            "ns_per_day_pallas_v1_2fs": ns_per_day_v1,
            "ns_per_day_pallas_v2_2fs": ns_per_day_v2,
            "ns_per_day_pallas_v2_rollout_2fs": ns_per_day_v2_rollout,
            "ns_per_day_openmm_rollout_2fs": ns_per_day_omm_rollout,
            "ns_per_day_openmm_rollout_mixed_2fs": ns_per_day_omm_rollout_mixed,
            "speedup_v2_vs_xla": t_x / t_v2,
            "speedup_v2_vs_v1": t_v1 / t_v2,
            "speedup_v2_rollout_vs_openmm": t_omm_rollout_per_step / t_v2_rollout_per_step,
            "speedup_v2_rollout_vs_openmm_mixed": (
                t_omm_rollout_mixed / t_v2_rollout_per_step
                if t_omm_rollout_mixed is not None
                else None
            ),
            "prolix_v2_energy_kcal": float(out_v2[0]),
            "openmm_energy_kcal": float(e_omm_kcal),
            "rel_dE_v2_vs_xla": float(abs(out_v2[0] - out_x[0]) / max(abs(float(out_x[0])), 1e-12)),
            "max_abs_dF_v2_vs_xla": float(jnp.max(jnp.abs(out_v2[1] - out_x[1]))),
            "rms_dF_v2_vs_openmm": float(np.sqrt(np.mean((np.asarray(-out_v2[1]) - f_omm_kcal_a) ** 2))),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)

    summary = {
        "openmm_platform": omm_platform_name,
        "mean_speedup_v2_vs_xla": float(np.mean([r["speedup_v2_vs_xla"] for r in rows])),
        "mean_speedup_v2_vs_v1": float(np.mean([r["speedup_v2_vs_v1"] for r in rows])),
        "mean_speedup_v2_rollout_vs_openmm": float(
            np.mean([r["speedup_v2_rollout_vs_openmm"] for r in rows])
        ),
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,2048,8192,32768")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="outputs/bench/pallas_vs_baseline.json")
    ap.add_argument("--interpret", action="store_true", help="force Pallas interpret mode")
    ap.add_argument(
        "--isolate-memory",
        action="store_true",
        help="measure peak device memory for each path in a fresh subprocess",
    )
    ap.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    ap.add_argument(
        "--kernel",
        choices=(
            "nonbonded",
            "bonded",
            "flash",
            "pme",
            "full_potential",
            "full_implicit",
            "openmm_comparison",
            "nl_build",
            "nl_skin",
            "mem_scaling",
        ),
        default="nonbonded",
        help="kernel/experiment family to benchmark",
    )
    args = ap.parse_args()

    interpret = args.interpret or not _triton_gpu_available()
    if args._worker:
        res = _run_isolated_worker(args._worker, args.repeats, interpret)
        print(json.dumps(res), flush=True)
        return

    sizes = [int(x) for x in args.sizes.split(",")]
    meta = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
        "pallas_interpret": interpret,
        "kernel": args.kernel,
        "jax": jax.__version__,
        "host": platform.node(),
        "cutoff": CUTOFF,
        "density": DENSITY,
    }

    if args.kernel in ("nl_build", "nl_skin", "mem_scaling", "bonded", "openmm_comparison"):
        if args.kernel == "nl_build":
            rows = bench_nl_build(sizes, args.repeats, interpret)
            summary = {}
        elif args.kernel == "nl_skin":
            rows = bench_nl_skin(sizes, args.repeats, interpret)
            summary = {}
        elif args.kernel == "bonded":
            rows, summary = bench_bonded(sizes, args.repeats, interpret)
        elif args.kernel == "openmm_comparison":
            rows, summary = bench_openmm_comparison(sizes, args.repeats, interpret)
        else:
            rows, summary = bench_mem_scaling(sizes, args.repeats, interpret)
        payload = {"meta": meta, "rows": rows, "summary": summary}
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        if summary:
            print(json.dumps(summary), flush=True)
        return

    if args.kernel == "pme":
        maker = make_pme_fns
        worker_prefix = "pme"
        needs_nl = False
    elif args.kernel == "flash":
        maker = make_flash_fns
        worker_prefix = "flash"
        needs_nl = False
    elif args.kernel == "full_potential":
        maker = make_full_potential_fns
        worker_prefix = "full_potential"
        needs_nl = True
    elif args.kernel == "full_implicit":
        maker = make_full_implicit_fns
        worker_prefix = "full_implicit"
        needs_nl = True
    else:
        maker = make_fns
        worker_prefix = "nonbonded"
        needs_nl = True

    rows = []
    for n in sizes:
        s = build_system(n, include_nl=needs_nl)
        if args.kernel in ("full_potential", "full_implicit"):
            s = build_full_potential_topology(s)
        xla_vg, pal_vg = maker(s, interpret)
        row = {"n_atoms": n, "k": s["k"]}
        out_x, c_x, t_x = time_fn(xla_vg, s["pos"], args.repeats)
        row.update(xla_compile_s=c_x, xla_median_s=t_x)
        mem_x = _peak_mem()
        out_p, c_p, t_p = time_fn(pal_vg, s["pos"], args.repeats)
        row.update(pallas_compile_s=c_p, pallas_median_s=t_p, speedup=t_x / t_p)
        mem_p = _peak_mem()
        if args.isolate_memory:
            wx = _spawn_worker(f"{worker_prefix}_xla", n, args.repeats, interpret)
            wp = _spawn_worker(f"{worker_prefix}_pallas", n, args.repeats, interpret)
            row["peak_mem_after_xla_bytes"] = wx.get("peak_bytes")
            row["peak_mem_after_pallas_bytes"] = wp.get("peak_bytes")
            row["xla_working_peak_bytes"] = wx.get("working_peak_bytes")
            row["pallas_working_peak_bytes"] = wp.get("working_peak_bytes")
            row["xla_activation_peak_bytes"] = wx.get("activation_peak_bytes")
            row["pallas_activation_peak_bytes"] = wp.get("activation_peak_bytes")
        else:
            row["peak_mem_after_xla_bytes"] = mem_x
            row["peak_mem_after_pallas_bytes"] = mem_p
        row["max_abs_dE"] = float(abs(out_x[0] - out_p[0]))
        row["rel_dE"] = float(abs(out_x[0] - out_p[0]) / max(abs(float(out_x[0])), 1e-12))
        row["max_abs_dF"] = float(jnp.max(jnp.abs(out_x[1] - out_p[1])))
        print(json.dumps(row), flush=True)
        rows.append(row)

    ns = [r["n_atoms"] for r in rows]
    summary = {
        "xla_scaling_exponent": fit_exponent(ns, [r["xla_median_s"] for r in rows]),
        "pallas_scaling_exponent": fit_exponent(ns, [r["pallas_median_s"] for r in rows]),
    }
    if args.isolate_memory:
        summary["xla_working_mem_exponent"] = fit_exponent(
            ns, [r.get("xla_working_peak_bytes") for r in rows]
        )
        summary["pallas_working_mem_exponent"] = fit_exponent(
            ns, [r.get("pallas_working_peak_bytes") for r in rows]
        )
    payload = {
        "meta": meta,
        "rows": rows,
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
