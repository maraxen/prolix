"""Tests for SETTLE constraint algorithm for rigid water molecules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from prolix.physics import settle

# Enable f64 for physics precision
jax.config.update("jax_enable_x64", True)


class TestSETTLEPositions:
    """Test position constraint application."""
    
    def test_single_water_constraints(self):
        """Test SETTLE maintains water geometry for single molecule."""
        # Create a single water molecule (O, H1, H2)
        R_old = jnp.array([
            [0.0, 0.0, 0.0],      # O at origin
            [0.757, 0.586, 0.0],  # H1
            [-0.757, 0.586, 0.0], # H2
        ])
        
        # Slightly perturb positions (as if from MD step)
        R_new = R_old + 0.05 * random.normal(random.PRNGKey(42), R_old.shape)
        
        water_indices = jnp.array([[0, 1, 2]])
        
        R_constrained = settle.settle_positions(
            R_new, R_old, water_indices,
            r_OH=settle.TIP3P_ROH,
            r_HH=settle.TIP3P_RHH
        )
        
        # Check that constrained distances match target
        r_OH1 = jnp.linalg.norm(R_constrained[1] - R_constrained[0])
        r_OH2 = jnp.linalg.norm(R_constrained[2] - R_constrained[0])
        r_HH = jnp.linalg.norm(R_constrained[2] - R_constrained[1])
        
        # Tolerance for analytical solution
        assert jnp.abs(r_OH1 - settle.TIP3P_ROH) < 0.01, f"O-H1 = {r_OH1:.4f}, expected {settle.TIP3P_ROH}"
        assert jnp.abs(r_OH2 - settle.TIP3P_ROH) < 0.01, f"O-H2 = {r_OH2:.4f}, expected {settle.TIP3P_ROH}"
        assert jnp.abs(r_HH - settle.TIP3P_RHH) < 0.02, f"H-H = {r_HH:.4f}, expected {settle.TIP3P_RHH}"
        
    def test_multiple_waters(self):
        """Test SETTLE works on multiple water molecules."""
        n_waters = 10
        key = random.PRNGKey(123)
        
        # Generate random water positions
        R_old = []
        for i in range(n_waters):
            # Base position for this water
            base = jnp.array([i * 5.0, 0.0, 0.0])
            O = base
            H1 = base + jnp.array([0.757, 0.586, 0.0])
            H2 = base + jnp.array([-0.757, 0.586, 0.0])
            R_old.extend([O, H1, H2])
        R_old = jnp.array(R_old)
        
        # Perturb
        key, subkey = random.split(key)
        R_new = R_old + 0.1 * random.normal(subkey, R_old.shape)
        
        # Create water indices
        water_indices = jnp.array([[i*3, i*3+1, i*3+2] for i in range(n_waters)])
        
        R_constrained = settle.settle_positions(R_new, R_old, water_indices)
        
        # Check all waters have correct geometry
        for i in range(n_waters):
            O_idx, H1_idx, H2_idx = i*3, i*3+1, i*3+2
            r_OH1 = jnp.linalg.norm(R_constrained[H1_idx] - R_constrained[O_idx])
            r_OH2 = jnp.linalg.norm(R_constrained[H2_idx] - R_constrained[O_idx])
            r_HH = jnp.linalg.norm(R_constrained[H2_idx] - R_constrained[H1_idx])
            
            assert jnp.abs(r_OH1 - settle.TIP3P_ROH) < 0.02, f"Water {i}: O-H1 = {r_OH1:.4f}"
            assert jnp.abs(r_OH2 - settle.TIP3P_ROH) < 0.02, f"Water {i}: O-H2 = {r_OH2:.4f}"
            assert jnp.abs(r_HH - settle.TIP3P_RHH) < 0.03, f"Water {i}: H-H = {r_HH:.4f}"

    def test_empty_water_indices(self):
        """Test SETTLE handles empty water indices gracefully."""
        R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        water_indices = jnp.zeros((0, 3), dtype=jnp.int32)
        
        R_out = settle.settle_positions(R, R, water_indices)
        
        # Should return unchanged positions
        assert jnp.allclose(R, R_out)

    def test_settle_with_pbc_crossing(self):
        """Test SETTLE handles water crossing periodic boundary correctly.
        
        When shift_fn wraps positions across the periodic boundary, the
        difference between R_old and R_new can be ~box_size. SETTLE must
        apply minimum image convention to handle this correctly.
        
        The output positions will be near R_old's coordinate frame (unwrapped),
        which is the correct behavior - we solve constraints in unwrapped space.
        """
        box = jnp.array([10.0, 10.0, 10.0])
        
        # Old positions: water near +x edge of box
        R_old = jnp.array([
            [9.5, 5.0, 5.0],    # O near +x edge
            [10.2, 5.5, 5.0],   # H1 would wrap
            [10.2, 4.5, 5.0],   # H2 would wrap
        ])
        
        # New positions: after wrapping, these jumped across the box
        # The H atoms wrapped from x~10.2 to x~0.2
        R_new = jnp.array([
            [0.1, 5.0, 5.0],    # O wrapped to -x side
            [0.8, 5.5, 5.0],    # H1 wrapped
            [0.8, 4.5, 5.0],    # H2 wrapped
        ])
        
        water_indices = jnp.array([[0, 1, 2]])
        
        # Without PBC handling, SETTLE would see ~9 Å differences
        # With PBC handling (MIC), it should see ~1 Å differences
        R_constrained = settle.settle_positions(
            R_new, R_old, water_indices, box=box
        )
        
        # Check that constrained geometry is correct
        r_OH1 = jnp.linalg.norm(R_constrained[1] - R_constrained[0])
        r_OH2 = jnp.linalg.norm(R_constrained[2] - R_constrained[0])
        r_HH = jnp.linalg.norm(R_constrained[2] - R_constrained[1])
        
        assert jnp.abs(r_OH1 - settle.TIP3P_ROH) < 0.05, f"O-H1 = {r_OH1:.4f}, expected {settle.TIP3P_ROH}"
        assert jnp.abs(r_OH2 - settle.TIP3P_ROH) < 0.05, f"O-H2 = {r_OH2:.4f}, expected {settle.TIP3P_ROH}"
        assert jnp.abs(r_HH - settle.TIP3P_RHH) < 0.1, f"H-H = {r_HH:.4f}, expected {settle.TIP3P_RHH}"
        
        # Constrained positions should be wrapped inside the box
        # Check that positions are within [0, box) range
        assert jnp.all(R_constrained >= 0), "Positions should be >= 0"
        assert jnp.all(R_constrained < 10.0), "Positions should be < box_size"


class TestSETTLEVelocities:
    """Test velocity constraint application."""
    
    def test_velocity_orthogonality(self):
        """Test that constrained velocities are orthogonal to bonds."""
        # Create single water
        R = jnp.array([
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ])
        
        # Random velocities
        V = random.normal(random.PRNGKey(77), R.shape)
        
        water_indices = jnp.array([[0, 1, 2]])
        
        V_constrained = settle.settle_velocities(V, R, R, water_indices, dt=0.002)
        
        # Check dot products with bonds are approximately zero
        r_OH1 = R[1] - R[0]
        r_OH2 = R[2] - R[0]
        r_HH = R[2] - R[1]
        
        v_OH1 = V_constrained[1] - V_constrained[0]
        v_OH2 = V_constrained[2] - V_constrained[0]
        v_HH = V_constrained[2] - V_constrained[1]
        
        dot_OH1 = jnp.dot(v_OH1, r_OH1 / jnp.linalg.norm(r_OH1))
        dot_OH2 = jnp.dot(v_OH2, r_OH2 / jnp.linalg.norm(r_OH2))
        dot_HH = jnp.dot(v_HH, r_HH / jnp.linalg.norm(r_HH))
        
        # Should be close to zero
        assert jnp.abs(dot_OH1) < 0.1, f"v.r_OH1 = {dot_OH1}"
        assert jnp.abs(dot_OH2) < 0.1, f"v.r_OH2 = {dot_OH2}"
        assert jnp.abs(dot_HH) < 0.1, f"v.r_HH = {dot_HH}"


class TestSETTLEIntegrator:
    """Test the full SETTLE Langevin integrator."""
    
    def test_integrator_energy_conservation(self):
        """Test that SETTLE integrator conserves energy approximately."""
        from jax_md import space
        
        # Simple harmonic potential for testing (not water-specific)
        def energy_fn(R):
            # Just a harmonic spring at origin
            return 0.5 * jnp.sum(R**2)
        
        # Single water
        R_init = jnp.array([
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ])
        
        water_indices = jnp.array([[0, 1, 2]])
        kT = 0.0  # NVE-like (no thermal noise)
        
        init_fn, apply_fn = settle.settle_langevin(
            energy_fn,
            shift_fn=space.free()[1],
            dt=0.001,  # 1 fs
            kT=kT,
            gamma=0.0,  # No friction (NVE)
            water_indices=water_indices,
        )
        
        key = random.PRNGKey(42)
        state = init_fn(key, R_init, kT=kT)
        
        # Run a few steps
        E_initial = energy_fn(state.position)
        
        for _ in range(100):
            state = apply_fn(state, kT=kT)
        
        E_final = energy_fn(state.position)
        
        # Energy should not explode (within 50% for simple test)
        assert E_final < E_initial * 2.0, f"Energy exploded: {E_initial} -> {E_final}"
        # Positions should be finite
        assert jnp.all(jnp.isfinite(state.position)), "Positions contain NaN/Inf"
        
    def test_water_geometry_preserved(self):
        """Test that integrator maintains water geometry over 1000 steps."""
        from jax_md import space
        
        def energy_fn(R):
            # Simple soft potential to test dynamics
            return 0.01 * jnp.sum(R**2)
        
        R_init = jnp.array([
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ])
        
        water_indices = jnp.array([[0, 1, 2]])
        kT = 0.001  # Very low temperature for stability
        
        init_fn, apply_fn = settle.settle_langevin(
            energy_fn,
            shift_fn=space.free()[1],
            dt=0.002,  # 2 fs
            kT=kT,
            gamma=1.0,
            water_indices=water_indices,
        )
        
        key = random.PRNGKey(0)
        state = init_fn(key, R_init)
        
        # Run 1000 steps
        apply_fn_jit = jax.jit(apply_fn)
        for _ in range(1000):
            state = apply_fn_jit(state)
        
        R_final = state.position
        
        # Check geometry is maintained
        r_OH1 = float(jnp.linalg.norm(R_final[1] - R_final[0]))
        r_OH2 = float(jnp.linalg.norm(R_final[2] - R_final[0]))
        r_HH = float(jnp.linalg.norm(R_final[2] - R_final[1]))
        
        # Allow some tolerance due to numerical integration
        assert abs(r_OH1 - settle.TIP3P_ROH) < 0.05, f"O-H1 drift: {r_OH1}"
        assert abs(r_OH2 - settle.TIP3P_ROH) < 0.05, f"O-H2 drift: {r_OH2}"
        assert abs(r_HH - settle.TIP3P_RHH) < 0.1, f"H-H drift: {r_HH}"
        
        print(f"After 1000 steps: O-H1={r_OH1:.4f}, O-H2={r_OH2:.4f}, H-H={r_HH:.4f}")


class TestGetWaterIndices:
    """Test water index generation helper."""
    
    def test_water_indices_generation(self):
        """Test that water indices are generated correctly."""
        n_protein = 100
        n_waters = 5
        
        indices = settle.get_water_indices(n_protein, n_waters)
        
        assert indices.shape == (5, 3)
        
        # Check indices
        assert int(indices[0, 0]) == 100  # First O
        assert int(indices[0, 1]) == 101  # First H1
        assert int(indices[0, 2]) == 102  # First H2
        
        assert int(indices[4, 0]) == 112  # Fifth O
        assert int(indices[4, 1]) == 113  # Fifth H1
        assert int(indices[4, 2]) == 114  # Fifth H2
        
    def test_empty_waters(self):
        """Test empty water list."""
        indices = settle.get_water_indices(100, 0)
        assert indices.shape == (0, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
