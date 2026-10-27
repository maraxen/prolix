"""Tests for FR-4 nonbonded methods (RF, DSF, LJ switching, solve_ewald_alpha, 1-4 exceptions),
pad-safe & molecule-unwrapped barostats (MC_Barostat_Step, SCR_Barostat_Step), and CSVR max_chi.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.batched_energy import (
    _coulomb_energy_masked,
    _lj_energy_masked,
    single_padded_energy,
    single_padded_force,
    solve_ewald_alpha,
)
from prolix.physics import pbc
from prolix.physics.barostat import MC_Barostat_Step, SCR_Barostat_Step
from prolix.physics.lj_switching import lj_switch_weight
from prolix.physics.step_system import CSVR_Step, _csvr_compute_lambda_jit_safe
from prolix.typing import EnergyParams, IntegratorParams, IntegratorState, PhysicsSystem


def _make_minimal_padded_system(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray | None = None,
    **kwargs,
) -> PhysicsSystem:
    n = positions.shape[0]
    return PhysicsSystem(
        positions=positions,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=jnp.ones(n, dtype=positions.dtype) * 1.5,
        scaled_radii=jnp.ones(n, dtype=positions.dtype) * 1.2,
        masses=jnp.ones(n, dtype=positions.dtype) * 12.0,
        element_ids=jnp.ones(n, dtype=jnp.int32) * 6,
        atom_mask=atom_mask,
        is_hydrogen=jnp.zeros(n, dtype=bool),
        is_backbone=jnp.zeros(n, dtype=bool),
        is_heavy=atom_mask,
        protein_atom_mask=atom_mask,
        water_atom_mask=jnp.zeros(n, dtype=bool),
        bonds=jnp.zeros((1, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((1, 2), dtype=positions.dtype),
        bond_mask=jnp.zeros(1, dtype=bool),
        angles=jnp.zeros((1, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((1, 2), dtype=positions.dtype),
        angle_mask=jnp.zeros(1, dtype=bool),
        dihedrals=jnp.zeros((1, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((1, 3), dtype=positions.dtype),
        dihedral_mask=jnp.zeros(1, dtype=bool),
        impropers=jnp.zeros((1, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((1, 3), dtype=positions.dtype),
        improper_mask=jnp.zeros(1, dtype=bool),
        urey_bradley_bonds=jnp.zeros((1, 2), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((1, 2), dtype=positions.dtype),
        urey_bradley_mask=jnp.zeros(1, dtype=bool),
        excl_indices=kwargs.pop("excl_indices", jnp.full((n, 4), -1, dtype=jnp.int32)),
        excl_scales_vdw=kwargs.pop("excl_scales_vdw", jnp.ones((n, 4), dtype=positions.dtype)),
        excl_scales_elec=kwargs.pop("excl_scales_elec", jnp.ones((n, 4), dtype=positions.dtype)),
        box_size=box_size,
        **kwargs,
    )


def test_solve_ewald_alpha_satisfies_openmm_tolerance_equation():
    """Verify solve_ewald_alpha solves erfc(alpha * r_cut) / r_cut = delta."""
    for cutoff in (8.0, 9.0, 10.0, 12.0):
        for tol in (5e-4, 1e-4, 1e-5):
            alpha = solve_ewald_alpha(cutoff, tol)
            residual = math.erfc(alpha * cutoff) / cutoff
            np.testing.assert_allclose(residual, tol, rtol=1e-6, atol=1e-10)

    with pytest.raises(ValueError, match="cutoff_angstrom"):
        solve_ewald_alpha(0.0, 5e-4)
    with pytest.raises(ValueError, match="error_tolerance"):
        solve_ewald_alpha(9.0, 0.0)


def test_lj_switching_function_boundary_values_and_derivatives():
    """Verify S(r) = 1 - 10x^3 + 15x^4 - 6x^5 satisfies C2 boundary conditions at r_switch and r_cut."""
    r_cut = 9.0
    w = 1.5
    r_switch = r_cut - w

    # Values at boundaries and midpoint
    r_pts = jnp.array([r_switch - 0.5, r_switch, r_switch + 0.5 * w, r_cut, r_cut + 0.5], dtype=jnp.float32)
    s_vals = lj_switch_weight(r_pts, r_cut, w)
    np.testing.assert_allclose(s_vals[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(s_vals[1], 1.0, atol=1e-6)
    np.testing.assert_allclose(s_vals[2], 0.5, atol=1e-6)
    np.testing.assert_allclose(s_vals[3], 0.0, atol=1e-6)
    np.testing.assert_allclose(s_vals[4], 0.0, atol=1e-6)

    # First derivative S'(r) vanishes as r -> r_switch^+ and r -> r_cut^-
    grad_s = jax.grad(lambda r: jnp.sum(lj_switch_weight(r, r_cut, w)))
    eps = 1e-4
    deriv_near_bounds = grad_s(jnp.array([r_switch + eps, r_cut - eps], dtype=jnp.float32))
    np.testing.assert_allclose(deriv_near_bounds, jnp.zeros(2), atol=1e-5)

    # Verify _lj_energy_masked and single_padded_energy respect cutoff and lj_switch_width
    disp_fn, _ = space.free()
    pos = jnp.array([[0.0, 0.0, 0.0], [8.2, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=jnp.float32)
    sigmas = jnp.array([3.5, 3.5, 3.5], dtype=jnp.float32)
    epsilons = jnp.array([0.2, 0.2, 0.2], dtype=jnp.float32)
    mask = jnp.array([True, True, False])  # Atom 2 is padded

    e_unswitched = _lj_energy_masked(
        pos, sigmas, epsilons, mask, disp_fn, cutoff=9.0, lj_switch_width=0.0
    )
    e_switched = _lj_energy_masked(
        pos, sigmas, epsilons, mask, disp_fn, cutoff=9.0, lj_switch_width=1.5
    )
    expected_scale = float(lj_switch_weight(jnp.array(8.2, dtype=jnp.float32), 9.0, 1.5))
    np.testing.assert_allclose(e_switched, e_unswitched * expected_scale, rtol=1e-5)

    # Beyond cutoff (r = 9.5 > 9.0) energy is strictly zero
    pos_beyond = jnp.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=jnp.float32)
    e_beyond = _lj_energy_masked(
        pos_beyond,
        sigmas[:2],
        epsilons[:2],
        jnp.array([True, True]),
        disp_fn,
        cutoff=9.0,
        lj_switch_width=1.5,
    )
    np.testing.assert_allclose(e_beyond, 0.0, atol=1e-7)


def test_reaction_field_and_dsf_electrostatics_with_14_exceptions():
    """Verify Reaction-Field and DSF electrostatics and unscreened 1-4 exceptions in single_padded_energy & force."""
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
    disp_fn, _ = pbc.create_periodic_space(box)

    # Atom 0 and 1: bulk non-excluded pair at r = 4.0 A (< cutoff 9.0 A)
    # Atom 0 and 2: 1-4 exception pair at r = 5.0 A (excluded from bulk via dense_excl_scale = 0.0)
    # Atom 3: padded dummy atom
    pos = jnp.array(
        [
            [2.0, 2.0, 2.0],
            [6.0, 2.0, 2.0],
            [2.0, 7.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    charges = jnp.array([0.8, -0.5, 0.6, 0.0], dtype=jnp.float32)
    sigmas = jnp.array([3.0, 3.0, 3.0, 0.0], dtype=jnp.float32)
    epsilons = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    atom_mask = jnp.array([True, True, True, False])

    # Exclude (0, 2) and (1, 2) from bulk so only (0, 1) is in bulk direct space,
    # while (0, 2) is evaluated via explicit 1-4 exception_pairs.
    dense_excl = jnp.ones((4, 4), dtype=jnp.float32)
    dense_excl = dense_excl.at[0, 2].set(0.0).at[2, 0].set(0.0)
    dense_excl = dense_excl.at[1, 2].set(0.0).at[2, 1].set(0.0)

    exc_pairs = jnp.array([[0, 2], [0, 0]], dtype=jnp.int32)
    exc_sigmas = jnp.array([3.0, 1.0], dtype=jnp.float32)
    exc_epsilons = jnp.array([0.1, 0.0], dtype=jnp.float32)
    exc_qprod = jnp.array([0.8 * 0.6 * 0.8333333, 0.0], dtype=jnp.float32)
    exc_mask = jnp.array([True, False])

    cutoff = 9.0
    eps_rf = 78.3
    dsf_alpha = 0.22
    coulomb_k = 332.0637

    # 1. Reaction-Field test
    sys_rf = _make_minimal_padded_system(
        pos,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        box_size=box,
        nonbonded_cutoff=cutoff,
        electrostatic_method="reaction_field",
        reaction_field_dielectric=eps_rf,
        use_dispersion_correction=False,
        dense_excl_scale_vdw=dense_excl,
        dense_excl_scale_elec=dense_excl,
        exception_pairs=exc_pairs,
        exception_sigmas=exc_sigmas,
        exception_epsilons=exc_epsilons,
        exception_chargeprods=exc_qprod,
        exception_mask=exc_mask,
    )
    e_rf = single_padded_energy(sys_rf, disp_fn, implicit_solvent=False)

    # Analytic RF for pair (0, 1) at r=4.0 + bare 1-4 exception for pair (0, 2) at r=5.0
    r01 = 4.0
    r02 = 5.0
    k_rf = (1.0 / cutoff**3) * ((eps_rf - 1.0) / (2.0 * eps_rf + 1.0))
    c_rf = (1.0 / cutoff) * ((3.0 * eps_rf) / (2.0 * eps_rf + 1.0))
    e_bulk_rf = coulomb_k * (0.8 * -0.5) * (1.0 / r01 + k_rf * r01**2 - c_rf)
    inv_r02_6 = (3.0 / r02) ** 6
    e_exc_14 = 4.0 * 0.1 * (inv_r02_6**2 - inv_r02_6) + coulomb_k * float(exc_qprod[0]) / r02
    np.testing.assert_allclose(e_rf, e_bulk_rf + e_exc_14, rtol=1e-5, atol=1e-5)

    # Force parity against -grad(single_padded_energy)
    f_rf = single_padded_force(
        sys_rf, disp_fn, implicit_solvent=False, explicit_solvent=True, use_flash=False
    )
    g_rf = -jax.grad(
        lambda r: single_padded_energy(sys_rf.replace(positions=r), disp_fn, implicit_solvent=False)
    )(pos)
    np.testing.assert_allclose(f_rf, g_rf, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(f_rf[3], jnp.zeros(3), atol=1e-7)

    # 2. DSF test
    sys_dsf = sys_rf.replace(electrostatic_method="dsf", dsf_alpha=dsf_alpha)
    e_dsf = single_padded_energy(sys_dsf, disp_fn, implicit_solvent=False)
    f_shift_dsf = (
        math.erfc(dsf_alpha * cutoff) / (cutoff * cutoff)
        + (2.0 * dsf_alpha / math.sqrt(math.pi)) * math.exp(-(dsf_alpha * cutoff) ** 2) / cutoff
    )
    e_bulk_dsf = (
        coulomb_k
        * (0.8 * -0.5)
        * (
            math.erfc(dsf_alpha * r01) / r01
            - math.erfc(dsf_alpha * cutoff) / cutoff
            + f_shift_dsf * (r01 - cutoff)
        )
    )
    np.testing.assert_allclose(e_dsf, e_bulk_dsf + e_exc_14, rtol=1e-5, atol=1e-5)

    f_dsf = single_padded_force(
        sys_dsf, disp_fn, implicit_solvent=False, explicit_solvent=True, use_flash=False
    )
    g_dsf = -jax.grad(
        lambda r: single_padded_energy(sys_dsf.replace(positions=r), disp_fn, implicit_solvent=False)
    )(pos)
    np.testing.assert_allclose(f_dsf, g_dsf, rtol=1e-5, atol=1e-5)


def test_mc_barostat_step_pad_safe_and_straddling_molecule_unwrap():
    """Verify MC_Barostat_Step unwraps molecules straddling PBC boundaries, masks padded atoms,
    and preserves intramolecular bond vectors under both (3,) and (3, 3) boxes.
    """
    L = 20.0
    box_1d = jnp.array([L, L, L], dtype=jnp.float32)

    # Molecule 0 straddles the x=0 / x=20 boundary: atom 0 at x=19.5, atom 1 at x=0.5.
    # True minimum-image bond vector r_1 - r_0 is (+1.0, 0.0, 0.0).
    # Molecule 1 is interior: atom 2 at x=10.0, atom 3 at x=11.2.
    # Atom 4 and 5 are padded dummy atoms (atom_mask=False) assigned molecule_id=0!
    # Without atom_mask and MIC unwrap, molecule 0's COM would be corrupted by both the
    # PBC wrap (averaging 19.5 and 0.5 -> 10.0) and the padded atoms at the origin.
    pos = jnp.array(
        [
            [19.5, 5.0, 5.0],
            [0.5, 5.0, 5.0],
            [10.0, 10.0, 10.0],
            [11.2, 10.0, 10.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    atom_mask = jnp.array([True, True, True, True, False, False])
    molecule_indices = jnp.array([0, 0, 1, 1, 0, 0], dtype=jnp.int32)

    # Zero energy so Metropolis move is guaranteed to accept for small volume change at P=0
    barostat = MC_Barostat_Step(
        barostat_interval=1,
        pressure=0.0,
        energy_fn=lambda r, b: jnp.array(0.0, dtype=r.dtype),
        n_molecules=2,
        volume_scale_factor=0.15,
        atom_mask=atom_mask,
    )
    state = IntegratorState(
        positions=pos,
        momentum=jnp.zeros_like(pos),
        force=jnp.zeros_like(pos),
        mass=jnp.ones((6, 1), dtype=jnp.float32),
        rng=jax.random.PRNGKey(123),
        box=box_1d,
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    params = IntegratorParams(
        dt=0.5,
        kT=0.6,
        gamma=1.0,
        energy_params=EnergyParams(
            params={
                "charges": jnp.zeros(6),
                "sigmas": jnp.ones(6),
                "epsilons": jnp.zeros(6),
            }
        ),
        molecule_indices=molecule_indices,
        box=box_1d,
    )

    new_state = jax.jit(barostat.apply)(state, params)
    # Volume must have changed
    assert not jnp.allclose(new_state.box, box_1d)
    mu = new_state.box[0] / box_1d[0]

    # Minimum-image intramolecular displacements MUST be strictly preserved (rigid-body translation)
    disp_old, _ = pbc.create_periodic_space(box_1d)
    disp_new, _ = pbc.create_periodic_space(new_state.box)
    bond_mol0_old = disp_old(pos[1], pos[0])
    bond_mol0_new = disp_new(new_state.positions[1], new_state.positions[0])
    np.testing.assert_allclose(bond_mol0_new, bond_mol0_old, atol=1e-5)

    bond_mol1_old = disp_old(pos[3], pos[2])
    bond_mol1_new = disp_new(new_state.positions[3], new_state.positions[2])
    np.testing.assert_allclose(bond_mol1_new, bond_mol1_old, atol=1e-5)

    # Unwrapped COM of molecule 0 was (20.0, 5.0, 5.0); its shift must be (mu - 1) * (20.0, 5.0, 5.0)
    expected_shift_mol0 = (mu - 1.0) * jnp.array([20.0, 5.0, 5.0], dtype=jnp.float32)
    np.testing.assert_allclose(new_state.positions[0] - pos[0], expected_shift_mol0, atol=1e-5)

    # Padded atoms must remain untouched at 0.0
    np.testing.assert_allclose(new_state.positions[4:], jnp.zeros((2, 3)), atol=1e-7)

    # Also test lower-triangular (3, 3) triclinic box in MC_Barostat_Step and SCR_Barostat_Step
    box_3x3 = pbc.triclinic_box_from_lengths_angles(22.0, 22.0, 22.0, 60.0, 60.0, 60.0)
    state_3x3 = state.replace(box=box_3x3, step_count=jnp.array(0, dtype=jnp.int32))
    new_state_3x3 = jax.jit(barostat.apply)(state_3x3, params)
    assert new_state_3x3.box.shape == (3, 3)

    scr = SCR_Barostat_Step(atom_mask=atom_mask)
    scr_state = jax.jit(scr.apply)(state_3x3, params)
    assert scr_state.box.shape == (3, 3)
    np.testing.assert_allclose(scr_state.positions[4:], jnp.zeros((2, 3)), atol=1e-7)


def test_csvr_max_chi_supports_large_n_atoms():
    """Verify _csvr_compute_lambda_jit_safe and CSVR_Step sample all n_dof - 1 chi-squared DOFs for N > 200 atoms."""
    # For N = 400 atoms, n_dof = 1200 > 512 (old hardcoded max_chi=512).
    # If kinetic energy is tiny (ke_current = 1e-4) and dt >> tau (c1 -> 0, c2_sq -> 1),
    # then E[lambda^2 * ke_current] = 0.5 * kT * (1 + E[s_chi_squared]) / n_dof * n_dof = 0.5 * n_dof * kT
    # ONLY if s_chi_squared sums (n_dof - 1) standard normals instead of truncating at 512!
    n_atoms = 400
    n_dof = 3 * n_atoms  # 1200
    kT = 0.6
    ke_current = jnp.array(0.1, dtype=jnp.float32)
    dt = 100.0
    tau = 1.0  # exp(-100) ~ 0, so c1 ~ 0 and new KE is purely drawn from target canonical distribution

    keys = jax.random.split(jax.random.PRNGKey(99), 256)
    lambdas = jax.vmap(
        lambda k: _csvr_compute_lambda_jit_safe(
            k, ke_current, n_dof, kT, dt, tau=tau, max_chi=3 * n_atoms
        )[0]
    )(keys)
    mean_new_ke = float(jnp.mean((lambdas**2) * ke_current))
    target_ke = 0.5 * n_dof * kT  # 360.0 kcal/mol
    # With old max_chi=512, mean_new_ke would be ~360 * (513 / 1200) = 153.9 kcal/mol!
    np.testing.assert_allclose(mean_new_ke, target_ke, rtol=0.05)

    # Also test CSVR_Step.apply with N = 300 atoms (900 DOFs)
    pos = jnp.zeros((300, 3), dtype=jnp.float32)
    mom = jnp.ones((300, 3), dtype=jnp.float32) * 0.05
    mass = jnp.ones((300, 1), dtype=jnp.float32)
    state = IntegratorState(
        positions=pos,
        momentum=mom,
        force=jnp.zeros_like(pos),
        mass=mass,
        rng=jax.random.PRNGKey(7),
    )
    params = IntegratorParams(
        dt=100.0,
        kT=kT,
        gamma=1.0,
        n_dof=900,
        energy_params=EnergyParams(
            params={"charges": jnp.zeros(300), "sigmas": jnp.ones(300), "epsilons": jnp.zeros(300)}
        ),
    )
    csvr_step = CSVR_Step(tau=1.0)
    out_state = jax.jit(csvr_step.apply)(state, params)
    ke_after = float(0.5 * jnp.sum(out_state.momentum**2 / mass))
    # Should be close to 0.5 * 900 * 0.6 = 270.0 (and definitely well above the truncated 512/900 * 270 = 153.6)
    assert ke_after > 220.0
