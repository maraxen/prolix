"""Sprint 2 Phase 3: EFA energy consistency tests.

Tests validate energy reproducibility and variance properties at fixed geometry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prolix.physics.eval_harness import (
    make_comparison_energies,
    make_tip3p_water_system,
)
from prolix.physics.electrostatic_methods import ElectrostaticMethod
from prolix.physics.flash_explicit import flash_explicit_energy


@pytest.mark.electrostatic_comparison
def test_pme_energy_reproducibility_frozen():
    """Test: PME energy is deterministic across repeated evaluations.

    Validates that the energy function is truly deterministic by computing
    PME energy 10 times at the same frozen geometry.
    """
    system = make_tip3p_water_system(n_waters=32, seed=0)

    energies = []
    for _ in range(10):
        e = flash_explicit_energy(
            system, electrostatic_method=ElectrostaticMethod.PME
        )
        energies.append(float(e))

    energies_arr = jnp.array(energies)
    std_energy = float(jnp.std(energies_arr))

    print(f"PME energy mean: {jnp.mean(energies_arr):.6f}")
    print(f"PME energy std: {std_energy:.2e}")

    assert (
        std_energy < 1e-6
    ), f"PME energy std {std_energy:.2e} >= 1e-6 — not deterministic"


@pytest.mark.electrostatic_comparison
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Sprint 2 blocker: EFA energy reproducibility test valid only if kernel is correct; "
        "fixed-ω determinism holds, but energies are biased vs PME reference. "
        "See rff_coulomb.py:16 and rff_erfc_derivation.md §2/§10."
    ),
)
def test_efa_fixed_omega_energy_reproducibility():
    """Test: EFA energy with SAME seed (fixed omega) is reproducible.

    Validates that EFA with a fixed omega vector produces bit-identical energies
    across multiple evaluations (modulo numerical precision).
    """
    system = make_tip3p_water_system(n_waters=32, seed=0)
    fixed_seed = 42

    energies = []
    for _ in range(10):
        e = flash_explicit_energy(
            system,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=512,
            rff_seed=fixed_seed,
        )
        energies.append(float(e))

    energies_arr = jnp.array(energies)
    std_energy = float(jnp.std(energies_arr))

    print(f"EFA (fixed omega) energy mean: {jnp.mean(energies_arr):.6f}")
    print(f"EFA (fixed omega) energy std: {std_energy:.2e}")

    assert (
        std_energy < 1e-6
    ), f"EFA fixed-omega energy std {std_energy:.2e} >= 1e-6 — reproducibility broken"


@pytest.mark.electrostatic_comparison
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Sprint 2 blocker: resampling variance test is spurious under kernel mismatch — "
        "fixed-ω variance is machine-zero (deterministic biased energies), making the "
        "ratio trivially large (observed 1.3M×). Not a meaningful EFA validation. "
        "See rff_coulomb.py:16 and rff_erfc_derivation.md §2/§10."
    ),
)
def test_efa_omega_resampling_increases_variance():
    """Test: EFA energy variance increases with omega resampling.

    Validates that resampling omega (different seed each evaluation) produces
    higher variance than using a fixed omega. The variance ratio should be > 5x
    per oracle recommendation.
    """
    system = make_tip3p_water_system(n_waters=32, seed=0)
    fixed_seed = 42

    # Fixed omega: 200 evaluations with same seed
    fixed_energies = []
    for _ in range(200):
        e = flash_explicit_energy(
            system,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=512,
            rff_seed=fixed_seed,
        )
        fixed_energies.append(float(e))

    fixed_var = float(jnp.var(jnp.array(fixed_energies)))

    # Resampled omega: 200 evaluations with different seeds
    resampled_energies = []
    for k in range(200):
        e = flash_explicit_energy(
            system,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=512,
            rff_seed=k,
        )
        resampled_energies.append(float(e))

    resampled_var = float(jnp.var(jnp.array(resampled_energies)))

    ratio = resampled_var / (fixed_var + 1e-12)
    print(
        f"Fixed-ω variance: {fixed_var:.6f}, Resampled-ω variance: {resampled_var:.6f}"
    )
    print(f"Variance ratio (resampled/fixed): {ratio:.1f}x")

    assert (
        resampled_var > 5 * fixed_var
    ), f"Resampled-ω variance ({resampled_var:.4f}) not > 5x fixed-ω variance ({fixed_var:.4f})"
