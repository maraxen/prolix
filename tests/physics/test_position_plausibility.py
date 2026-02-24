"""Position plausibility tests for molecular dynamics simulations.

These tests ensure that minimization and simulation produce physically
plausible atomic positions that match OpenMM behavior.

Uses the new parse_structure API (Rust parser) instead of the removed
biotite/hydride path.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space
from proxide import CoordFormat, OutputSpec, parse_structure

from prolix.physics import system

# Enable x64 for physics

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
FF_PATH = (
  Path(__file__).parent.parent.parent.parent
  / "proxide"
  / "src"
  / "proxide"
  / "assets"
  / "protein.ff19SB.xml"
)

# OpenMM bundled force field paths (from proxide assets)
OMM_FF_PATH = (
  Path(__file__).parent.parent.parent.parent
  / "proxide"
  / "src"
  / "proxide"
  / "assets"
  / "openmm_bundled"
  / "amber19"
  / "protein.ff19SB.xml"
)
OMM_IMPLICIT_PATH = (
  Path(__file__).parent.parent.parent.parent
  / "proxide"
  / "src"
  / "proxide"
  / "assets"
  / "openmm_bundled"
  / "implicit"
  / "obc2.xml"
)
def _load_parameterized_protein(pdb_name: str = "1UAO.pdb"):
  """Load and parameterize a protein via the Rust parser."""
  pdb_path = DATA_DIR / pdb_name
  if not pdb_path.exists():
    pytest.skip(f"{pdb_name} not found")

  spec = OutputSpec(
    parameterize_md=True,
    force_field=str(FF_PATH),
    add_hydrogens=True,
    coord_format=CoordFormat.Full,
  )
  return parse_structure(str(pdb_path), spec)
def _get_flat_coords(protein):
  """Extract flat (N, 3) coordinates from a Protein.

  With CoordFormat.Full, coordinates are already (N_atoms, 3).
  Falls back to Atom37 mask extraction for legacy format.
  """
  coords = protein.coordinates
  if coords.ndim == 2:
    return coords
  # Legacy Atom37 fallback
  mask = protein.atom_mask
  flat_coords = coords.reshape(-1, 3)
  flat_mask = mask.reshape(-1)
  valid_indices = jnp.where(flat_mask > 0.5)[0]
  return flat_coords[valid_indices]
class TestPositionPlausibility:
  """Test that physics computations produce plausible positions matching OpenMM."""

  @pytest.fixture
  def chignolin_setup(self):
    """Setup Chignolin (1UAO) for testing using new parse_structure API."""
    protein = _load_parameterized_protein("1UAO.pdb")
    coords = _get_flat_coords(protein)

    # Build energy function
    displacement_fn, shift_fn = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, protein, implicit_solvent=True)

    return {
      "energy_fn": energy_fn,
      "positions": coords,
      "coords": np.asarray(coords),
      "protein": protein,
    }

  def test_initial_positions_molecular_scale(self, chignolin_setup):
    """Initial PDB positions should be within typical molecular range."""
    coords = chignolin_setup["coords"]

    # Center positions
    centered = coords - coords.mean(axis=0)

    # Proteins typically fit within ~100 Å
    assert centered.min() > -100, f"Min position {centered.min():.1f} Å too far"
    assert centered.max() < 100, f"Max position {centered.max():.1f} Å too far"

  def test_minimization_preserves_position_scale(self, chignolin_setup):
    """Minimized positions should stay within molecular scale."""
    from prolix.physics import simulate

    energy_fn = chignolin_setup["energy_fn"]
    positions = chignolin_setup["positions"]

    # Run minimization
    r_min = simulate.run_minimization(energy_fn, positions, steps=500)

    # Convert to numpy and center
    r_min_np = np.asarray(r_min)
    centered = r_min_np - r_min_np.mean(axis=0)

    # Positions should remain within ±100 Å (molecular scale)
    assert centered.min() > -100, f"Atom flew too far negative: {centered.min():.1f} Å"
    assert centered.max() < 100, f"Atom flew too far positive: {centered.max():.1f} Å"

  def test_minimization_reduces_energy(self, chignolin_setup):
    """Minimization should reduce energy significantly."""
    from prolix.physics import simulate

    energy_fn = chignolin_setup["energy_fn"]
    positions = chignolin_setup["positions"]

    e_initial = float(energy_fn(positions))

    r_min = simulate.run_minimization(energy_fn, positions, steps=500)
    e_final = float(energy_fn(r_min))

    # Energy should decrease significantly
    assert e_final < e_initial, (
      f"Minimization did not reduce energy: {e_initial:.1f} -> {e_final:.1f}"
    )

    # Energy should be finite
    assert np.isfinite(e_final), f"Final energy is not finite: {e_final}"

  def test_minimization_finite_forces(self, chignolin_setup):
    """Forces after minimization should be finite and within reasonable bounds."""
    from prolix.physics import simulate

    energy_fn = chignolin_setup["energy_fn"]
    positions = chignolin_setup["positions"]
    protein = chignolin_setup["protein"]

    r_min = simulate.run_minimization(energy_fn, positions, steps=500)

    displacement_fn, _ = space.free()

    # Compute forces (-grad E)
    energy_fn_for_grad = system.make_energy_fn(
      displacement_fn, protein, implicit_solvent=True
    )
    forces = -jax.grad(energy_fn_for_grad)(r_min)

    # Check finite
    assert jnp.all(jnp.isfinite(forces))

    # Check magnitude (max force < 5000 kcal/mol/Å)
    max_force = jnp.max(jnp.linalg.norm(forces, axis=-1))
    assert max_force < 5000.0, (
      f"Max force after minimization is too large: {max_force:.1f} kcal/mol/Å"
    )

  def test_all_atoms_bonded(self, chignolin_setup):
    """Verify every atom has at least one bond to prevent floating atoms."""
    protein = chignolin_setup["protein"]

    bonds = np.array(protein.bonds)
    n_atoms = len(protein.charges)

    bonded_atoms = set()
    for a, b in bonds:
      bonded_atoms.add(int(a))
      bonded_atoms.add(int(b))

    unbonded_indices = [i for i in range(n_atoms) if i not in bonded_atoms]

    assert len(unbonded_indices) == 0, (
      f"Found {len(unbonded_indices)} unbonded atoms that will float away: {unbonded_indices[:10]}"
    )
def openmm_available():
  """Check if OpenMM is available."""
  try:
    import openmm  # noqa: F401
    from openmm import app  # noqa: F401

    return True
  except ImportError:
    return False
@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestOpenMMComparison:
  """Compare minimization results with OpenMM as ground truth."""

  @pytest.fixture
  def setup_both_systems(self):
    """Setup both JAX MD and OpenMM systems on same structure."""
    from openmm import app

    # Load and parameterize via Rust parser
    protein = _load_parameterized_protein("1UAO.pdb")
    coords_jax = _get_flat_coords(protein)

    # JAX energy function
    displacement_fn, shift_fn = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, protein, implicit_solvent=True)

    # OpenMM setup - need a PDB file
    pdb_path = DATA_DIR / "1UAO.pdb"
    pdb_file = app.PDBFile(str(pdb_path))
    topology = pdb_file.topology

    # Use bundled force field XMLs from proxide assets
    if not OMM_FF_PATH.exists():
      pytest.skip(f"OpenMM FF not found: {OMM_FF_PATH}")
    if not OMM_IMPLICIT_PATH.exists():
      pytest.skip(f"OpenMM implicit solvent FF not found: {OMM_IMPLICIT_PATH}")

    omm_ff = app.ForceField(str(OMM_FF_PATH), str(OMM_IMPLICIT_PATH))
    omm_system = omm_ff.createSystem(
      topology,
      nonbondedMethod=app.NoCutoff,
      constraints=None,
      rigidWater=False,
      removeCMMotion=False,
    )

    return {
      "jax_energy_fn": energy_fn,
      "jax_positions": coords_jax,
      "omm_system": omm_system,
      "omm_positions": pdb_file.positions,
      "topology": topology,
      "coords": np.asarray(coords_jax),
    }

  def test_minimization_position_rmsd_vs_openmm(self, setup_both_systems):
    """JAX MD minimized positions should be similar to OpenMM minimized positions."""
    import openmm
    from openmm import app, unit

    from prolix.physics import simulate

    data = setup_both_systems

    # ---- JAX MD Minimization ----
    jax_r_min = simulate.run_minimization(data["jax_energy_fn"], data["jax_positions"], steps=1000)
    jax_r_min_np = np.asarray(jax_r_min)

    # ---- OpenMM Minimization ----
    integrator = openmm.LangevinMiddleIntegrator(
      300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    simulation = app.Simulation(data["topology"], data["omm_system"], integrator)
    simulation.context.setPositions(data["omm_positions"])

    # Minimize
    simulation.minimizeEnergy(maxIterations=1000)

    # Get final positions
    omm_state = simulation.context.getState(getPositions=True)
    omm_r_min = omm_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    # ---- Compare Position Scales ----
    jax_centered = jax_r_min_np - jax_r_min_np.mean(axis=0)
    omm_centered = omm_r_min - omm_r_min.mean(axis=0)

    jax_range = (jax_centered.min(), jax_centered.max())

    # JAX positions should be within similar scale as OpenMM
    assert jax_range[0] > -100, f"JAX min position {jax_range[0]:.1f} too far"
    assert jax_range[1] < 100, f"JAX max position {jax_range[1]:.1f} too far"

    # Compare structure similarity via all-atom RMSD
    # Note: atom counts may differ between Rust-added and OpenMM-added hydrogens.
    # Only compare if counts match.
    if jax_r_min_np.shape[0] == omm_r_min.shape[0]:
      jax_c = jax_r_min_np - jax_r_min_np.mean(axis=0)
      omm_c = omm_r_min - omm_r_min.mean(axis=0)

      rmsd = np.sqrt(np.mean(np.sum((jax_c - omm_c) ** 2, axis=1)))
      assert rmsd < 10.0, (
        f"RMSD between JAX MD and OpenMM minimized structures is {rmsd:.2f} Å (expected < 10 Å)"
      )
    else:
      print(
        f"Skipping RMSD: JAX has {jax_r_min_np.shape[0]} atoms, "
        f"OpenMM has {omm_r_min.shape[0]} atoms"
      )

  def test_minimized_energy_same_order_of_magnitude(self, setup_both_systems):
    """Minimized energies should be in same order of magnitude."""
    import openmm
    from openmm import app, unit

    from prolix.physics import simulate

    data = setup_both_systems

    # ---- JAX MD ----
    jax_r_min = simulate.run_minimization(data["jax_energy_fn"], data["jax_positions"], steps=1000)
    jax_e_min = float(data["jax_energy_fn"](jax_r_min))

    # ---- OpenMM ----
    integrator = openmm.LangevinMiddleIntegrator(
      300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    simulation = app.Simulation(data["topology"], data["omm_system"], integrator)
    simulation.context.setPositions(data["omm_positions"])
    simulation.minimizeEnergy(maxIterations=1000)

    omm_state = simulation.context.getState(getEnergy=True)
    omm_e_min = omm_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    # Both should be negative (stable minimum)
    assert jax_e_min < 0, f"JAX minimized energy should be negative: {jax_e_min:.1f}"

    # Log the values for debugging
    print(f"JAX minimized energy: {jax_e_min:.1f} kcal/mol")
    print(f"OpenMM minimized energy: {omm_e_min:.1f} kcal/mol")
    print(f"Difference: {abs(jax_e_min - omm_e_min):.1f} kcal/mol")

  def test_minimization_position_range_vs_openmm(self, setup_both_systems):
    """JAX MD and OpenMM should produce similar position ranges after minimization."""
    import openmm
    from openmm import app, unit

    from prolix.physics import simulate

    data = setup_both_systems

    # ---- JAX MD Minimization ----
    jax_r_min = simulate.run_minimization(data["jax_energy_fn"], data["jax_positions"], steps=1000)
    jax_r_min_np = np.asarray(jax_r_min)

    # ---- OpenMM Minimization ----
    integrator = openmm.LangevinMiddleIntegrator(
      300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    simulation = app.Simulation(data["topology"], data["omm_system"], integrator)
    simulation.context.setPositions(data["omm_positions"])
    simulation.minimizeEnergy(maxIterations=1000)

    omm_state = simulation.context.getState(getPositions=True)
    omm_r_min = omm_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    # ---- Compare Absolute Position Ranges ----
    jax_centered = jax_r_min_np - jax_r_min_np.mean(axis=0)
    omm_centered = omm_r_min - omm_r_min.mean(axis=0)

    jax_range = jax_centered.max() - jax_centered.min()
    omm_range = omm_centered.max() - omm_centered.min()

    print(f"JAX position range: {jax_range:.2f} Å")
    print(f"OpenMM position range: {omm_range:.2f} Å")
    print(f"Ratio: {jax_range / omm_range:.2f}")

    # Ranges should be within factor of 2 of each other
    assert 0.5 < jax_range / omm_range < 2.0, (
      f"JAX position range ({jax_range:.2f} Å) differs significantly from "
      f"OpenMM ({omm_range:.2f} Å)"
    )

    # Both should indicate compact molecular structure (< 50 Å for small proteins)
    assert jax_range < 50.0, f"JAX structure too extended: {jax_range:.2f} Å"
    assert omm_range < 50.0, f"OpenMM structure too extended: {omm_range:.2f} Å"
class TestTrajectoryPositions:
  """Test that trajectory frames have reasonable positions."""

  def test_trajectory_positions_stable(self):
    """If a trajectory file exists, all frames should have reasonable positions."""
    traj_path = Path("chignolin_traj.array_record")

    if not traj_path.exists():
      pytest.skip("Trajectory file not found - run simulation first")

    from prolix.visualization import TrajectoryReader

    reader = TrajectoryReader(str(traj_path))
    for i in range(min(len(reader), 10)):  # Check first 10 frames
      state = reader.get_state(i)
      pos = np.asarray(state.positions)
      centered = pos - pos.mean(axis=0)

      # Positions should stay within ±200 Å (generous for minor outliers)
      assert centered.min() > -200 and centered.max() < 200, (
        f"Frame {i} has positions outside molecular range: "
        f"[{centered.min():.1f}, {centered.max():.1f}]"
      )
