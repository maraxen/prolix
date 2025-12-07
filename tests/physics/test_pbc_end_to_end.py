"""
Integration tests for PBC/PME physics end-to-end.

These tests verify the complete simulation pipeline with periodic
boundary conditions and PME electrostatics against OpenMM.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space

# Enable x64 for precision
jax.config.update("jax_enable_x64", True)

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

from prolix.physics import system, pbc, pme


class TestPBCEndToEnd:
    """End-to-end tests for PBC/PME physics."""
    
    @pytest.fixture
    def simple_two_particle_system(self):
        """Two charged particles in a periodic box."""
        box_size = 30.0  # Angstroms
        charges = [1.0, -1.0]
        positions = [[5.0, 5.0, 5.0], [20.0, 5.0, 5.0]]  # 15A separation
        
        return {
            "box_size": box_size,
            "charges": charges,
            "positions": positions,
            "pme_alpha": 0.34,
            "pme_grid": 32,
            "cutoff": 9.0,
        }
    
    @pytest.fixture
    def four_particle_system(self):
        """Four charged particles (2 positive, 2 negative) for more complex test."""
        box_size = 40.0
        charges = [1.0, -1.0, 1.0, -1.0]
        positions = [
            [10.0, 10.0, 10.0],
            [30.0, 10.0, 10.0],
            [10.0, 30.0, 10.0],
            [30.0, 30.0, 10.0],
        ]
        
        return {
            "box_size": box_size,
            "charges": charges,
            "positions": positions,
            "pme_alpha": 0.34,
            "pme_grid": 32,
            "cutoff": 12.0,
        }
    
    def _create_mock_system_params(self, charges):
        """Create minimal system params for testing."""
        n = len(charges)
        return {
            "charges": jnp.array(charges),
            "sigmas": jnp.ones(n),
            "epsilons": jnp.zeros(n),  # No LJ for clean electrostatics test
            "bonds": jnp.zeros((0, 2), dtype=int),
            "bond_params": jnp.zeros((0, 2)),
            "angles": jnp.zeros((0, 3), dtype=int),
            "angle_params": jnp.zeros((0, 2)),
            "dihedrals": jnp.zeros((0, 4), dtype=int),
            "dihedral_params": jnp.zeros((0, 3)),
            "impropers": jnp.zeros((0, 4), dtype=int),
            "improper_params": jnp.zeros((0, 3)),
            "exclusion_mask": jnp.ones((n, n)) - jnp.eye(n),
        }
    
    def _setup_openmm_pme(self, config):
        """Setup OpenMM with PME for comparison."""
        box_size = config["box_size"]
        charges = config["charges"]
        positions = config["positions"]
        alpha = config["pme_alpha"]
        grid = config["pme_grid"]
        cutoff = config["cutoff"]
        
        omm_system = openmm.System()
        box_nm = box_size / 10.0
        omm_system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(box_nm, 0, 0),
            openmm.Vec3(0, box_nm, 0),
            openmm.Vec3(0, 0, box_nm)
        )
        
        for _ in charges:
            omm_system.addParticle(1.0)
            
        nonbonded = openmm.NonbondedForce()
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(cutoff / 10.0)  # A to nm
        nonbonded.setPMEParameters(alpha * 10.0, grid, grid, grid)  # alpha A^-1 to nm^-1
        nonbonded.setUseDispersionCorrection(False)
        
        for q in charges:
            nonbonded.addParticle(q, 0.1, 0.0)  # sigma in nm, epsilon 0
            
        omm_system.addForce(nonbonded)
        
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(omm_system, integrator, 
                                  openmm.Platform.getPlatformByName('Reference'))
        
        pos_nm = [openmm.Vec3(p[0]/10, p[1]/10, p[2]/10) for p in positions]
        context.setPositions(pos_nm)
        
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom)
        
        return energy, forces
    
    def _setup_jax_pme(self, config):
        """Setup JAX MD with PBC/PME."""
        box_size = config["box_size"]
        charges = config["charges"]
        positions = config["positions"]
        alpha = config["pme_alpha"]
        grid = config["pme_grid"]
        cutoff = config["cutoff"]
        
        box_vec = jnp.array([box_size, box_size, box_size])
        system_params = self._create_mock_system_params(charges)
        
        displacement_fn, _ = pbc.create_periodic_space(box_vec)
        
        energy_fn = system.make_energy_fn(
            displacement_fn,
            system_params,
            box=box_vec,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=grid,
            pme_alpha=alpha,
            cutoff_distance=cutoff,
        )
        
        pos_jax = jnp.array(positions)
        energy = float(energy_fn(pos_jax))
        forces = -jax.grad(energy_fn)(pos_jax)
        
        return energy, np.array(forces)
    
    @pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
    def test_two_particle_pme_energy(self, simple_two_particle_system):
        """Test PME energy matches OpenMM for simple 2-particle system."""
        omm_energy, _ = self._setup_openmm_pme(simple_two_particle_system)
        jax_energy, _ = self._setup_jax_pme(simple_two_particle_system)
        
        print(f"OpenMM Energy: {omm_energy:.4f} kcal/mol")
        print(f"JAX MD Energy: {jax_energy:.4f} kcal/mol")
        
        # PME approximation allows ~0.5 kcal/mol tolerance
        assert np.isclose(omm_energy, jax_energy, atol=0.5), \
            f"Energy mismatch: OMM={omm_energy:.4f}, JAX={jax_energy:.4f}"
    
    @pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
    def test_two_particle_pme_forces(self, simple_two_particle_system):
        """Test PME forces match OpenMM for simple 2-particle system."""
        _, omm_forces = self._setup_openmm_pme(simple_two_particle_system)
        _, jax_forces = self._setup_jax_pme(simple_two_particle_system)
        
        force_diff = omm_forces - jax_forces
        rmse = np.sqrt(np.mean(force_diff**2))
        
        print(f"Force RMSE: {rmse:.6f} kcal/mol/A")
        
        assert rmse < 0.1, f"Force RMSE too high: {rmse:.6f}"
    
    @pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
    def test_four_particle_pme_energy(self, four_particle_system):
        """Test PME energy matches OpenMM for 4-particle system."""
        omm_energy, _ = self._setup_openmm_pme(four_particle_system)
        jax_energy, _ = self._setup_jax_pme(four_particle_system)
        
        print(f"OpenMM Energy: {omm_energy:.4f} kcal/mol")
        print(f"JAX MD Energy: {jax_energy:.4f} kcal/mol")
        
        assert np.isclose(omm_energy, jax_energy, atol=1.0), \
            f"Energy mismatch: OMM={omm_energy:.4f}, JAX={jax_energy:.4f}"
    
    @pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
    def test_neutral_system_energy(self):
        """Test that a neutral system has near-zero interaction energy."""
        config = {
            "box_size": 50.0,
            "charges": [0.0, 0.0],  # Neutral
            "positions": [[10.0, 10.0, 10.0], [40.0, 40.0, 40.0]],
            "pme_alpha": 0.34,
            "pme_grid": 32,
            "cutoff": 12.0,
        }
        
        omm_energy, _ = self._setup_openmm_pme(config)
        jax_energy, _ = self._setup_jax_pme(config)
        
        print(f"OpenMM Energy (neutral): {omm_energy:.6f} kcal/mol")
        print(f"JAX MD Energy (neutral): {jax_energy:.6f} kcal/mol")
        
        # Both should be essentially zero
        assert abs(omm_energy) < 0.01
        assert abs(jax_energy) < 0.01
    
    def test_pbc_wrapping(self):
        """Test that positions outside box are handled correctly."""
        box_vec = jnp.array([10.0, 10.0, 10.0])
        displacement_fn, _ = pbc.create_periodic_space(box_vec)
        
        # Particle outside box
        r1 = jnp.array([1.0, 1.0, 1.0])
        r2 = jnp.array([12.0, 1.0, 1.0])  # 12 > 10, should wrap to 2
        
        dr = displacement_fn(r1, r2)
        dist = jnp.linalg.norm(dr)
        
        # Distance should be 1 (wrapped), not 11 (unwrapped)
        assert jnp.isclose(dist, 1.0, atol=0.01), f"Expected dist=1.0, got {dist}"
    
    def test_pme_gradients_finite(self):
        """Test that PME gradients are finite and reasonable."""
        box_vec = jnp.array([30.0, 30.0, 30.0])
        charges = [1.0, -1.0]
        positions = jnp.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
        
        system_params = self._create_mock_system_params(charges)
        displacement_fn, _ = pbc.create_periodic_space(box_vec)
        
        energy_fn = system.make_energy_fn(
            displacement_fn,
            system_params,
            box=box_vec,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=32,
            pme_alpha=0.34,
        )
        
        # Compute gradients
        grad_fn = jax.grad(energy_fn)
        forces = -grad_fn(positions)
        
        # Check finite
        assert jnp.all(jnp.isfinite(forces)), "Forces contain non-finite values"
        
        # Check reasonable magnitude (should be < 100 kcal/mol/A for this system)
        max_force = jnp.max(jnp.abs(forces))
        assert max_force < 100.0, f"Force too large: {max_force}"
    
    def test_pme_jit_compatible(self):
        """Test that PME energy function is JIT-compatible."""
        box_vec = jnp.array([30.0, 30.0, 30.0])
        charges = [1.0, -1.0, 0.5, -0.5]
        positions = jnp.array([
            [5.0, 5.0, 5.0],
            [25.0, 5.0, 5.0],
            [5.0, 25.0, 5.0],
            [25.0, 25.0, 5.0],
        ])
        
        system_params = self._create_mock_system_params(charges)
        displacement_fn, _ = pbc.create_periodic_space(box_vec)
        
        energy_fn = system.make_energy_fn(
            displacement_fn,
            system_params,
            box=box_vec,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=32,
            pme_alpha=0.34,
        )
        
        # JIT compile
        jit_energy_fn = jax.jit(energy_fn)
        
        # First call (compilation)
        e1 = jit_energy_fn(positions)
        
        # Second call (cached)
        e2 = jit_energy_fn(positions)
        
        assert jnp.isclose(e1, e2), "JIT results inconsistent"
        assert jnp.isfinite(e1), "JIT result not finite"


class TestPBCWithLJ:
    """Test PBC/PME with Lennard-Jones interactions."""
    
    @pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
    def test_lj_plus_pme(self):
        """Test that LJ + PME combined energy is correct."""
        box_size = 30.0
        charges = [0.5, -0.5]
        sigmas = [3.0, 3.0]  # Angstroms
        epsilons = [0.1, 0.1]  # kcal/mol
        positions = [[10.0, 10.0, 10.0], [17.0, 10.0, 10.0]]  # 7A apart
        
        # OpenMM
        omm_system = openmm.System()
        box_nm = box_size / 10.0
        omm_system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(box_nm, 0, 0),
            openmm.Vec3(0, box_nm, 0),
            openmm.Vec3(0, 0, box_nm)
        )
        
        for _ in charges:
            omm_system.addParticle(1.0)
            
        nonbonded = openmm.NonbondedForce()
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(0.9)
        nonbonded.setPMEParameters(3.4, 32, 32, 32)
        nonbonded.setUseDispersionCorrection(False)
        
        for i, q in enumerate(charges):
            nonbonded.addParticle(q, sigmas[i]/10.0, epsilons[i] * 4.184)  # Convert to kJ
            
        omm_system.addForce(nonbonded)
        
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(omm_system, integrator, 
                                  openmm.Platform.getPlatformByName('Reference'))
        context.setPositions([openmm.Vec3(p[0]/10, p[1]/10, p[2]/10) for p in positions])
        
        omm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole)
        
        # JAX MD
        box_vec = jnp.array([box_size, box_size, box_size])
        system_params = {
            "charges": jnp.array(charges),
            "sigmas": jnp.array(sigmas),
            "epsilons": jnp.array(epsilons),
            "bonds": jnp.zeros((0, 2), dtype=int),
            "bond_params": jnp.zeros((0, 2)),
            "angles": jnp.zeros((0, 3), dtype=int),
            "angle_params": jnp.zeros((0, 2)),
            "dihedrals": jnp.zeros((0, 4), dtype=int),
            "dihedral_params": jnp.zeros((0, 3)),
            "impropers": jnp.zeros((0, 4), dtype=int),
            "improper_params": jnp.zeros((0, 3)),
            "exclusion_mask": jnp.ones((2, 2)) - jnp.eye(2),
        }
        
        displacement_fn, _ = pbc.create_periodic_space(box_vec)
        
        energy_fn = system.make_energy_fn(
            displacement_fn,
            system_params,
            box=box_vec,
            use_pbc=True,
            implicit_solvent=False,
            pme_grid_points=32,
            pme_alpha=0.34,
        )
        
        jax_energy = float(energy_fn(jnp.array(positions)))
        
        print(f"OpenMM LJ+PME Energy: {omm_energy:.4f} kcal/mol")
        print(f"JAX MD LJ+PME Energy: {jax_energy:.4f} kcal/mol")
        
        # Relaxed tolerance for combined energy
        assert np.isclose(omm_energy, jax_energy, atol=1.0), \
            f"Energy mismatch: OMM={omm_energy:.4f}, JAX={jax_energy:.4f}"
