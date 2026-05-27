"""Parity test for nonbonded energy decomposition in explicit solvent.

Validates that Prolix's nonbonded energy components (LJ, Coulomb, 1-4 exceptions)
match OpenMM reference calculations for an alanine dipeptide system.

Test objectives (per P2b spec):
- test_lj_energy_parity: Compare LJ energies (assert |dE| < 1.0 kcal/mol)
- test_coulomb_energy_parity: Compare Coulomb energies (assert |dE| < 1.0 kcal/mol)
- test_exception_14_energy_parity: Prolix self-consistency check (assert |dE| < 0.2 kcal/mol)
"""

import math
import sys
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])  # make fixtures importable

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.openmm


@pytest.fixture(scope="module")
def nb_parity_bundle():
    """Build OpenMM and prolix nonbonded energy evaluations for parity comparison.

    Constructs an alanine dipeptide system with OpenMM and extracts parameters
    to build an equivalent Prolix nonbonded system. Evaluates both energies
    and returns a bundle with all components needed for parity tests.

    Returns dict with keys:
        omm_e: dict with OpenMM energies ('lj', 'coulomb', 'total_nb')
        prolix_e: dict with Prolix energies ('lj', 'coulomb', 'exception_14')
        positions: (N, 3) positions in Angstroms
        omm_system: OpenMM System object
        exclusion_spec: ExclusionSpec for Prolix
        prolix_system: PhysicsSystem
        displacement_fn: JAX-MD displacement function
        nb_params: Nonbonded parameters from OpenMM
    """
    from fixtures_openmm_parity import (
        build_ala_dip_openmm_system,
        extract_bonded_params,
        extract_nonbonded_params,
        build_exclusion_spec,
        build_prolix_nonbonded_system,
        get_prolix_nonbonded_energies,
        get_openmm_nonbonded_energies,
    )

    omm_system, positions_ang, _ = build_ala_dip_openmm_system()
    bonded_params = extract_bonded_params(omm_system)
    nb_params = extract_nonbonded_params(omm_system)
    exclusion_spec = build_exclusion_spec(omm_system, positions_ang.shape[0])
    prolix_system, displacement_fn = build_prolix_nonbonded_system(nb_params, bonded_params, positions_ang)

    omm_e = get_openmm_nonbonded_energies(omm_system, positions_ang)
    prolix_e = get_prolix_nonbonded_energies(prolix_system, displacement_fn, positions_ang, exclusion_spec)

    return {
        'omm_e': omm_e,
        'prolix_e': prolix_e,
        'positions': positions_ang,
        'omm_system': omm_system,
        'exclusion_spec': exclusion_spec,
        'prolix_system': prolix_system,
        'displacement_fn': displacement_fn,
        'nb_params': nb_params,
    }


def test_lj_energy_parity(nb_parity_bundle):
    """Verify LJ energy parity between Prolix and OpenMM (gate: |dE| < 1.0 kcal/mol).

    Compares:
      - prolix_lj: LJ energy from chunked_lj_energy (1-5+ pairs via exclusion mask)
      - omm_lj: LJ energy from OpenMM ForceGroup 3 (charge-zeroed pass)

    Note: Per P2b spec, OpenMM's charge-zeroed pass leaves exception_chargeprods
    untouched, so omm_lj includes 1-4 Coulomb contributions. This is handled by
    the self-consistency test (test_exception_14_energy_parity).
    """
    bundle = nb_parity_bundle
    prolix_lj = bundle['prolix_e']['lj']
    omm_lj = bundle['omm_e']['lj']
    delta = abs(prolix_lj - omm_lj)

    print(f"\nLJ parity: prolix={prolix_lj:.4f}, omm={omm_lj:.4f}, delta={delta:.6f} kcal/mol")
    assert delta < 1.0, f"LJ parity exceeded: delta={delta:.6f} kcal/mol (gate: 1.0)"


def test_coulomb_energy_parity(nb_parity_bundle):
    """Verify Coulomb energy parity between Prolix and OpenMM (gate: |dE| < 1.0 kcal/mol).

    Compares:
      - prolix_coulomb: Coulomb energy from chunked_coulomb_energy (1-5+ pairs via exclusion mask)
      - omm_coulomb: Coulomb energy = omm_total_nb - omm_lj (residual after LJ subtraction)

    Per P2b spec, omm_coulomb = E_nb - E_lj(charge_zeroed). The LJ pass includes
    1-4 Coulomb from exception_chargeprods, so omm_coulomb is 1-5+ Coulomb only.
    """
    bundle = nb_parity_bundle
    prolix_coulomb = bundle['prolix_e']['coulomb']
    omm_coulomb = bundle['omm_e']['coulomb']
    delta = abs(prolix_coulomb - omm_coulomb)

    print(f"\nCoulomb parity: prolix={prolix_coulomb:.4f}, omm={omm_coulomb:.4f}, delta={delta:.6f} kcal/mol")
    assert delta < 1.0, f"Coulomb parity exceeded: delta={delta:.6f} kcal/mol (gate: 1.0)"


def test_exception_14_energy_parity(nb_parity_bundle):
    """Verify 1-4 exception energy self-consistency within Prolix (gate: |dE| < 0.2 kcal/mol).

    This is a self-consistency check comparing two Prolix evaluations:
    1. exception_14 from make_energy_fn(..., return_decomposed=True)['exception']
    2. Direct standalone evaluation via make_exception_pair_energy_fn

    Both should use the same parameters (exception_pairs, exception_sigmas,
    exception_epsilons, exception_chargeprods extracted from OpenMM). Any
    discrepancy indicates a parameter assembly or composition bug in Prolix.

    Note: This is NOT an OpenMM comparison (no ground truth for decomposed
    exception energy). It verifies internal Prolix consistency only.
    """
    from prolix.physics.bonded import make_exception_pair_energy_fn

    bundle = nb_parity_bundle

    # Composed evaluation (from make_energy_fn decomposed)
    composed_exc14 = bundle['prolix_e']['exception_14']

    # Direct standalone evaluation
    exc_spec = bundle['exclusion_spec']
    direct_fn = make_exception_pair_energy_fn(
        bundle['displacement_fn'],
        exc_spec.exception_pairs,
        exc_spec.exception_sigmas.astype(jnp.float32),
        exc_spec.exception_epsilons.astype(jnp.float32),
        exc_spec.exception_chargeprods.astype(jnp.float32),
    )

    positions = jnp.array(bundle['positions'], dtype=jnp.float64)
    direct_exc14 = float(direct_fn(positions))

    delta = abs(direct_exc14 - composed_exc14)

    print(f"\nException-14 self-consistency: direct={direct_exc14:.4f}, composed={composed_exc14:.4f}, delta={delta:.6f} kcal/mol")
    assert delta < 0.2, f"Exception-14 self-consistency exceeded: delta={delta:.6f} kcal/mol (gate: 0.2)"
