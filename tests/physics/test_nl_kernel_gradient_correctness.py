"""Regression test for NL kernel position-gradient correctness.

This test validates that the custom_vjp backward passes in chunked_lj_energy_nl
and chunked_coulomb_energy_nl correctly compute the position gradient.

Per #4623 spec AC1, AC4, AC7: jax.grad must match central finite-difference
within relative tolerance < 1e-3 on a real multi-atom system, with explicit
coverage of the LJ switch window and genuinely non-trivial (non-near-zero)
force magnitudes for both LJ and Coulomb kernels.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Always enable x64 precision for PME/Coulomb accuracy (per spec precision lock)
jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "oracles" / "openmm_8.3.1"


def _load_1vii_gold() -> dict:
    """Load the vendored 1vii NL gold record (no OpenMM needed)."""
    gold_path = DATA_DIR / "nl_1vii.json"
    if not gold_path.exists():
        pytest.skip(f"1vii gold not found at {gold_path}")
    return json.loads(gold_path.read_text())


def _bundle_from_1vii_gold(rec: dict):
    """Build MolecularBundle from gold record (imported from emit script for consistency)."""
    from scripts.experiments.nl_omm_parity import _bundle_from_1vii_gold as helper

    return helper(rec)


class TestNLKernelGradientCorrectness:
    """NL kernel position-gradient correctness via jax.grad vs finite-difference."""

    @pytest.fixture
    def gold(self):
        """Vendored 1vii gold record."""
        return _load_1vii_gold()

    @pytest.fixture
    def bundle(self, gold):
        """MolecularBundle built from gold."""
        return _bundle_from_1vii_gold(gold)

    @pytest.fixture
    def neighbor_list(self, bundle):
        """Allocated and updated neighbor list."""
        from prolix.physics import neighbor_list as nl
        from prolix.physics.pbc import create_periodic_space

        displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
        neighbor_fn = nl.make_neighbor_list_fn(
            displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
        )
        nbr0 = neighbor_fn.allocate(bundle.positions)
        nbr = neighbor_fn.update(bundle.positions, nbr0)

        assert not bool(nbr.did_buffer_overflow), "neighbor list capacity overflow"
        return nbr

    def test_gradient_vs_finite_difference_lj_coulomb(self, gold, bundle, neighbor_list):
        """jax.grad vs central finite-difference for both LJ and Coulomb NL kernels.

        AC1: relative error < 1e-3 on a real multi-atom system.
        AC4: Verify both kernels exercise genuinely non-trivial forces (not near-zero).
        AC7: Explicitly verify at least one sampled pair sits in LJ switch window [8.0, 9.0] A.
        """
        from prolix.api.bundle_md import energy_fn_from_bundle
        from prolix.physics import neighbor_list as nl
        from prolix.physics.pbc import create_periodic_space

        # Build energy function with LJ switching (lj_switch_width=1.0, same as gold)
        energy_fn = energy_fn_from_bundle(
            bundle,
            include_nonbonded=True,
            lj_switch_width=1.0,
            pme_grid_points=int(gold["pme_grid_points"]),
        )

        # Compute jax.grad forces for all atoms (cheap)
        grad_fn = jax.grad(energy_fn)
        f_grad = -np.asarray(grad_fn(bundle.positions, neighbor=neighbor_list))
        f_gold = np.asarray(gold["forces_kcal_mol_A"], dtype=np.float64)

        # Compute per-atom force magnitude
        f_magnitude_grad = np.linalg.norm(f_grad, axis=1)
        f_magnitude_gold = np.linalg.norm(f_gold, axis=1)

        # AC4: Verify genuinely non-trivial forces (not near-zero) for both LJ and Coulomb
        high_force_atoms = np.where(f_magnitude_gold > 1.0)[0]
        assert len(high_force_atoms) > 10, (
            f"Expected >10 atoms with |f_gold| > 1.0 kcal/mol/A for LJ+Coulomb kernel test; "
            f"found only {len(high_force_atoms)}"
        )
        print(f"\nGradient correctness vs finite-difference (eps=1e-4 A):")
        print(f"  High-force atoms (|f_gold| > 1.0): {len(high_force_atoms)} atoms")

        # AC7: Explicitly verify at least one pair sits in LJ switch window [8.0, 9.0] A
        # Sample high-force atoms to compute finite differences
        positions = np.asarray(bundle.positions)
        cutoff = float(bundle.cutoff_distance)
        switch_width = 1.0  # per spec
        r_switch = cutoff - switch_width  # 8.0 A
        r_cut = cutoff  # 9.0 A

        # Find the first pair (i, j) where distance falls in switch window
        switch_window_pair = None
        for i in high_force_atoms[:50]:  # Sample high-force atoms
            for j in range(i + 1, min(i + 100, bundle.n_atoms)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if r_switch < dist < r_cut:
                    switch_window_pair = (i, j, dist)
                    break
            if switch_window_pair is not None:
                break

        assert switch_window_pair is not None, (
            f"No pairs found in LJ switch window [{r_switch}, {r_cut}] A; "
            f"cannot verify switch-function gradient coverage"
        )

        # Sample atoms to verify gradient correctness: include switch-window atoms
        # plus a few other high-force atoms for diversity
        atoms_to_check = []
        atoms_to_check.append(switch_window_pair[0])
        atoms_to_check.append(switch_window_pair[1])
        # Add a few more high-force atoms
        for atom_idx in high_force_atoms[:8]:
            if atom_idx not in atoms_to_check:
                atoms_to_check.append(atom_idx)

        # Compute central finite-difference forces for sampled atoms
        eps = 1e-4  # Angstrom
        displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
        neighbor_fn = nl.make_neighbor_list_fn(
            displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance)
        )
        nbr0 = neighbor_fn.allocate(bundle.positions)

        f_fd_sampled = {}
        for atom_idx in atoms_to_check:
            f_fd_atom = np.zeros(3)
            for dim in range(3):
                pos_plus = np.array(bundle.positions)
                pos_plus[atom_idx, dim] += eps
                pos_minus = np.array(bundle.positions)
                pos_minus[atom_idx, dim] -= eps

                nbr_plus = neighbor_fn.update(jnp.asarray(pos_plus), nbr0)
                nbr_minus = neighbor_fn.update(jnp.asarray(pos_minus), nbr0)

                e_plus = float(energy_fn(jnp.asarray(pos_plus), neighbor=nbr_plus))
                e_minus = float(energy_fn(jnp.asarray(pos_minus), neighbor=nbr_minus))

                f_fd_atom[dim] = -(e_plus - e_minus) / (2 * eps)

            f_fd_sampled[atom_idx] = f_fd_atom

        # Verify gradient agreement for sampled atoms
        relative_errors = []
        for atom_idx in atoms_to_check:
            f_fd = f_fd_sampled[atom_idx]
            f_grad_atom = f_grad[atom_idx]
            deviation = np.linalg.norm(f_grad_atom - f_fd)
            f_magnitude_fd = np.linalg.norm(f_fd)
            f_magnitude_fd_safe = max(f_magnitude_fd, 1e-6)
            relative_error = deviation / f_magnitude_fd_safe
            relative_errors.append(relative_error)

            print(f"  Atom {atom_idx}: rel_err={relative_error:.6f}, "
                  f"|f_grad|={np.linalg.norm(f_grad_atom):.4f}, "
                  f"|f_fd|={f_magnitude_fd:.4f}")

        # AC1: tight relative tolerance for sampled atoms
        max_rel_error = max(relative_errors)
        assert max_rel_error < 1e-3, (
            f"Max relative error {max_rel_error:.6f} exceeds tolerance 1e-3"
        )

        # Print switch-window pair info
        i_switch, j_switch, dist_switch = switch_window_pair
        print(f"\n  Switch-window pair coverage (AC7 requirement):")
        print(f"    Pair ({i_switch}, {j_switch}) at distance {dist_switch:.4f} A "
              f"(in window [{r_switch}, {r_cut}] A)")
        print(f"    Atom {i_switch}: |f_grad|={f_magnitude_grad[i_switch]:.4f}, "
              f"|f_gold|={f_magnitude_gold[i_switch]:.4f}")
        print(f"    Atom {j_switch}: |f_grad|={f_magnitude_grad[j_switch]:.4f}, "
              f"|f_gold|={f_magnitude_gold[j_switch]:.4f}")
        print(f"  Max relative error (sampled atoms): {max_rel_error:.6f}")
