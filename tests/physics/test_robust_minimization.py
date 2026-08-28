
import logging
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# XA-CI: API/physics drift or heavy compile — deselect from GitHub-faithful suite; tracked under XA-DRIFT.
pytestmark = pytest.mark.slow
from jax_md import space

from prolix import simulate
from prolix.physics import neighbor_list as nl
from prolix.physics import system
from prolix.physics.pbc import create_periodic_space, minimum_image_distance
from prolix.physics.spec import PhysicsSpec

# Enable x64 for physics

@pytest.fixture
def clash_system():
    """Setup a simple harmonic systems with two particles that clash."""
    
    # Simple harmonic 2-particle system
    # V = k * (r - r0)^2
    # But we want to test LJ-like behavior where V -> inf as r -> 0
    
    # Let's use actual LJ potential for testing robustness
    sigma = 1.0
    epsilon = 1.0
    
    def energy_fn(R, neighbor=None):
        dr = R[0] - R[1]
        dist = jnp.linalg.norm(dr)
        
        # LJ potential
        sr6 = (sigma / dist) ** 6
        energy = 4 * epsilon * (sr6**2 - sr6)
        return energy

    # Initial position: Very close (clash)
    # sigma=1.0. Minimum at 2^(1/6) ~= 1.12
    # Place at 0.5 -> Huge Repulsion
    R_clash = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    
    return energy_fn, R_clash

def test_clash_minimization_survives(clash_system):
    """Test that minimization doesn't produce NaNs even with severe clashes."""
    energy_fn, R_init = clash_system
    
    # Standard minimization (might fail with simple GD, expecting success with robust)
    spec = simulate.SimulationSpec(total_time_ns=0.001) # Dummy spec
    
    # We invoke the robust minimization logic directly if possible, or via run_simulation
    # Since we are testing internal logic, we might need to expose it or run a minimal simulation loop
    
    # But proxide's minimize is inside run_simulation currently.
    # We will test using run_simulation with a mock system if possible, 
    # but run_simulation requires a SystemParams or AtomicSystem.
    
    # Let's verify the energy explosion with a simple manually constructed GD loop first
    # to confirm the baseline behavior, then verify the fix.
    
    # Baseline check: Does simple GD explode?
    grad_fn = jax.grad(energy_fn)
    
    R = R_init
    step_size = 0.001
    
    # Simple GD for 10 steps
    for i in range(10):
        g = grad_fn(R)
        R = R - step_size * g
        if not jnp.all(jnp.isfinite(R)):
            print(f"Exploded at step {i}")
            break
            
    # We expect this to likely be unstable or result in huge jumps.
    # The real test is running the NEW code in simulate.py.
    pass
@pytest.fixture
def minimal_lj_system():
    """Create a minimal 2-particle system params dict."""
    # 2 particles
    n_atoms = 2
    
    # LJ parameters
    sigmas = jnp.array([1.0, 1.0])
    epsilons = jnp.array([1.0, 1.0])
    charges = jnp.zeros(n_atoms)
    
    # Empty topology
    bonds = jnp.zeros((0, 2), dtype=jnp.int32)
    bond_params = jnp.zeros((0, 2))
    angles = jnp.zeros((0, 3), dtype=jnp.int32)
    angle_params = jnp.zeros((0, 2))
    dihedrals = jnp.zeros((0, 4), dtype=jnp.int32)
    dihedral_params = jnp.zeros((0, 3))
    impropers = jnp.zeros((0, 4), dtype=jnp.int32)
    improper_params = jnp.zeros((0, 3))
    
    system_params = {
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
        "bonds": bonds,
        "bond_params": bond_params,
        "angles": angles,
        "angle_params": angle_params,
        "dihedrals": dihedrals,
        "dihedral_params": dihedral_params,
        "impropers": impropers,
        "improper_params": improper_params,
        "gb_radii": jnp.ones(n_atoms) * 1.5, # Dummy radii
    }
    return system_params

def test_clash_minimization_survives(minimal_lj_system):
    """Test that minimization handles severe overlap without NaN."""
    system_params = minimal_lj_system
    
    # Initial position: Very close overlap (r=0.5 < sigma=1.0)
    # Potential is ~ (1/0.5)^12 = 4096 * 4 = ~16000 epsilon
    # Gradients will be huge.
    initial_positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ])
    
    spec = simulate.SimulationSpec(
        total_time_ns=0.0001, # Minimal run
        physics=PhysicsSpec(dt=1.0),
        save_interval_ns=0.0001,
        accumulate_steps=1,
        use_pbc=False,
        use_neighbor_list=False
    )
    
    # This should NOT rasie NaN/Inf error or RuntimeWarning
    # `run_simulation` runs minimization internally
    final_state = simulate.run_simulation(
        system=system_params,
        initial_positions=initial_positions,
        spec=spec
    )
    
    # Check that positions are finite and separated
    pos = final_state.positions
    dist = jnp.linalg.norm(pos[0] - pos[1])
    
    print(f"Final distance: {dist}")
    
    assert jnp.all(jnp.isfinite(pos))
    assert dist > 0.8 # Should have pushed apart significantly
    # Ideally < 0.0, but if it pushes to > cutoff (9.0) it might be 0. 
    # Or if slightly repulsive but safe, that's okay too compared to start.
    assert final_state.potential_energy < 100.0 

def test_standard_minimization_trajectory(minimal_lj_system):
    """Test that robust minimization still converges for normal cases."""
    system_params = minimal_lj_system
    
    # Start slightly outside minimum
    initial_positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0]
    ])
    
    spec = simulate.SimulationSpec(
        total_time_ns=0.0001,
        physics=PhysicsSpec(dt=1.0),
    )

    final_state = simulate.run_simulation(
        system=system_params,
        initial_positions=initial_positions,
        spec=spec
    )

    pos = final_state.positions
    dist = jnp.linalg.norm(pos[0] - pos[1])

    # Expected min is 2^(1/6) * sigma = 1.122
    assert jnp.abs(dist - 1.122) < 0.05


def test_neighbor_list_minimization_dt_start_not_pinned(minimal_lj_system, caplog):
    """#4623 Validation gate 4 / AC6: run_simulation's use_neighbor_list=True path.

    `run_simulation` sizes FIRE's dt_start off `jax.grad` through the NL kernels
    (simulate.py:647-653, the only jax.grad call site in the file) whenever
    `spec.use_neighbor_list and spec.box is not None` (simulate.py:517). No
    existing test in this file reached that path before this addition -- every
    prior fixture left `box=None`/`use_pbc=False`, so `use_neighbor_list=True`
    alone was never sufficient to build the NL branch.

    Before the #4623 fix, `_chunked_lj_nl_bwd` returned an all-zero position
    gradient, so `max_grad` here would be ~0 and dt_start would be clipped to
    its ceiling (0.001 ps, since 0.001/(0+1e-8) clips down to 0.001). Post-fix,
    a real initial force magnitude should size dt_start well below that ceiling.
    We assert dt_start is NOT pinned at the ceiling, and (independently) that
    FIRE minimization actually decreases the energy -- either alone could pass
    by accident (e.g. a system already at equilibrium won't decrease energy
    regardless of gradient correctness), but the two together can't.
    """
    system_params = minimal_lj_system

    # 2 atoms, 1.5 A apart -- well inside the 9.0 A default neighbor cutoff
    # (PhysicsSpec.neighbor_cutoff default), so the neighbor list is guaranteed
    # non-empty at this geometry (checked explicitly below, not assumed).
    initial_positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
    ])
    box = jnp.array([30.0, 30.0, 30.0])  # >> 2*cutoff, avoids periodic self-interaction

    spec = simulate.SimulationSpec(
        total_time_ns=0.0001,
        physics=PhysicsSpec(dt=1.0),
        save_interval_ns=0.0001,
        accumulate_steps=1,
        use_pbc=True,
        box=box,
        use_neighbor_list=True,
    )

    # Precondition: the neighbor list at this geometry is genuinely non-empty --
    # guards against a degenerate/empty-neighbor-list case coincidentally
    # "passing" gate 4 without exercising the fixed kernels at all (round-3
    # adversarial review finding on this gate).
    displacement_fn, _ = create_periodic_space(box)
    neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, box, spec.neighbor_cutoff)
    neighbor = neighbor_fn.allocate(initial_positions)
    n_atoms = initial_positions.shape[0]
    n_real_neighbors = int(jnp.sum(neighbor.idx < n_atoms))
    assert n_real_neighbors > 0, (
        "Neighbor list is empty at the tested geometry -- this test would "
        "coincidentally pass without exercising the NL+jax.grad path at all."
    )

    caplog.set_level(logging.INFO, logger="prolix.simulate")

    final_state = simulate.run_simulation(
        system=system_params,
        initial_positions=initial_positions,
        spec=spec,
    )

    # Extract dt_start from the "Dynamic FIRE dt: dt_start=X ps, ..." log line
    # (simulate.py:655-656) -- the only place dt_start is observable from here.
    dt_start_lines = [
        rec.message for rec in caplog.records if "Dynamic FIRE dt: dt_start=" in rec.message
    ]
    assert dt_start_lines, "Expected a 'Dynamic FIRE dt' log line from run_simulation's minimizer"
    match = re.search(r"dt_start=([\d.eE+-]+)", dt_start_lines[0])
    assert match is not None, f"Could not parse dt_start from log line: {dt_start_lines[0]!r}"
    dt_start = float(match.group(1))

    dt_start_ceiling = 0.001  # ps -- simulate.py:653's jnp.clip upper bound
    # Margin: pre-fix, max_grad ~= 0 hard-clips dt_start to *exactly* the
    # ceiling (0.001 / 1e-8 clips to 0.001). Post-fix, for this fixture (2
    # atoms, sigma=eps=1, r=1.5 A), the analytical LJ force is
    # |dE/dr| = (4*eps/r) * (-12*(sigma/r)^12 + 6*(sigma/r)^6) ~= 1.158
    # kcal/mol/A, giving dt_start = 0.001/1.158 ~= 8.64e-4 ps -- correct and
    # non-trivially below the ceiling, but not below half of it. The original
    # 0.5 margin assumed a larger max_grad than this fixture actually
    # produces; 0.95 still rejects the exact-ceiling pre-fix symptom while
    # accepting the fixture's real, verified-correct gradient.
    assert dt_start < dt_start_ceiling * 0.95, (
        f"dt_start={dt_start:.2e} ps is pinned near its ceiling ({dt_start_ceiling} ps), "
        f"the symptom of the pre-#4623-fix broken zero-gradient custom_vjp -- "
        f"max_grad from jax.grad through the NL kernels must have been ~0."
    )

    # Independently confirm genuine minimization progress: two clashing-ish
    # atoms starting at 1.5 A (outside the LJ minimum at 2^(1/6)*sigma ~ 1.122 A)
    # should relax toward the minimum, not stay put or diverge.
    # Positions are wrapped into [0, box) by the periodic shift_fn, so a raw
    # jnp.linalg.norm(pos[0] - pos[1]) can spuriously read as tens of A when
    # the pair wraps to opposite box faces while genuinely close under the
    # minimum-image convention (the only distance that is physically
    # meaningful under PBC). Max possible minimum-image distance in a 30 A
    # box is sqrt(3)*15 ~= 25.98 A -- a raw, non-periodic distance can and
    # did exceed that (36.85 A observed), which is what final_dist must
    # avoid measuring.
    pos = final_state.positions
    final_dist = float(minimum_image_distance(pos[0], pos[1], box))
    assert jnp.isfinite(final_dist)
    assert abs(final_dist - 1.122) < 0.1, (
        f"Final separation {final_dist:.4f} A did not converge toward the LJ "
        f"minimum (~1.122 A) -- minimization did not make real progress."
    )
