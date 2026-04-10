#!/usr/bin/env python3
"""GB Force Kernel Performance Benchmark.

Compares three approaches for computing GB solvation forces on a single
PDZ domain system (~1400 atoms, bucket 2048):

1. JAX Analytical Forces (production path): gb_ace_forces_dense()
   Full N² with decomposed VJP — current bottleneck.

2. JAX Chunked (pallas_kernels.py): _gb_coulomb_forces_chunked()
   fori_loop over tiles, no N² materialization, custom_vjp.

3. OpenMM Reference: Single-system GBSA force evaluation.
   Gold standard for per-system implicit solvent speed.

Each test warms up with 3 runs (JIT compilation), then times 50 iterations.

Usage:
  uv run scripts/benchmark_gb_forces.py
"""

import os
import sys
import time
import logging

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "False")

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# Add prolix to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prolix", "src"))


def load_real_system(pdb_name: str = "1KEF_pdz"):
    """Load a real PDZ system from the references directory."""
    import dataclasses
    from proxide.io.parsing.backend import parse_structure
    from proxide import OutputSpec, CoordFormat
    from prolix.padding import pad_protein, select_bucket

    pdb_path = os.path.join(
        os.path.dirname(__file__), "..", "references", "pdb",
        f"{pdb_name}_fixed.pdb"
    )
    log.info(f"Loading {pdb_path}...")

    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        parameterize_md=True,
        force_field="proxide/src/proxide/assets/protein.ff19SB.xml",
        add_hydrogens=False,
        remove_solvent=True,
        remove_hetatm=True,
    )
    protein = parse_structure(pdb_path, spec)

    # Assign GB radii if missing
    if protein.radii is None:
        from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors
        _radii = assign_mbondi2_radii(list(protein.atom_names), protein.bonds)
        _scaled = assign_obc2_scaling_factors(list(protein.atom_names))
        protein = dataclasses.replace(
            protein,
            radii=jnp.asarray(_radii),
            scaled_radii=jnp.asarray(_scaled),
        )

    n_atoms = np.asarray(protein.coordinates).reshape(-1, 3).shape[0]
    bucket = select_bucket(n_atoms)
    log.info(f"  {n_atoms} atoms → bucket {bucket}")

    system = pad_protein(
        protein, bucket,
        target_bonds=int(1.2 * bucket),
        target_angles=int(2.2 * bucket),
        target_dihedrals=int(3.5 * bucket),
        target_impropers=int(0.5 * bucket),
        target_cmaps=int(0.3 * bucket),
        target_constraints=int(0.7 * bucket),
    )
    return system, protein, n_atoms


def benchmark_fn(name: str, fn, n_warmup: int = 3, n_iter: int = 50):
    """Benchmark a function, returning median time in ms."""
    log.info(f"  [{name}] Warming up ({n_warmup} iters)...")
    for _ in range(n_warmup):
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()

    log.info(f"  [{name}] Timing ({n_iter} iters)...")
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    log.info(
        f"  [{name}] median={np.median(times):.3f} ms, "
        f"mean={np.mean(times):.3f} ms, "
        f"std={np.std(times):.3f} ms, "
        f"min={np.min(times):.3f} ms, max={np.max(times):.3f} ms"
    )
    return np.median(times)


def test_jax_analytical(system):
    """Benchmark: JAX analytical forces (production path)."""
    from prolix.physics.analytical_forces import gb_ace_forces_dense

    log.info("=== Test 1: JAX Analytical Forces (gb_ace_forces_dense) ===")

    pos = system.positions
    charges = system.charges
    radii = system.radii
    scaled_radii = system.scaled_radii
    atom_mask = system.atom_mask

    @jax.jit
    def compute_forces():
        return gb_ace_forces_dense(
            pos, charges, radii, scaled_radii, atom_mask,
        )

    return benchmark_fn("analytical_forces", compute_forces)


def test_jax_chunked(system):
    """Benchmark: JAX chunked GB+Coulomb (pallas_kernels.py)."""
    from prolix.pallas_kernels import gb_coulomb_energy_dense
    from prolix.physics.generalized_born import compute_born_radii

    log.info("=== Test 2: JAX Chunked GB+Coulomb (custom_vjp, fori_loop) ===")

    pos = system.positions
    charges = system.charges
    radii = system.radii
    atom_mask = system.atom_mask
    N = pos.shape[0]
    mask_ij = atom_mask[:, None] & atom_mask[None, :]

    # Pre-compute born radii (this happens once at the start of each step)
    born_radii = compute_born_radii(
        pos, radii,
        mask=mask_ij.astype(jnp.float32),
        scaled_radii=system.scaled_radii,
    )

    # Test the energy + grad (forces) pipeline
    @jax.jit
    def compute_forces():
        grad_fn = jax.grad(gb_coulomb_energy_dense)
        return -grad_fn(pos, charges, born_radii, atom_mask)

    return benchmark_fn("chunked_fori_loop", compute_forces)


def test_jax_born_radii_only(system):
    """Benchmark: Just the Born radii computation (dense N²)."""
    from prolix.physics.generalized_born import compute_born_radii

    log.info("=== Test 3: Born Radii Only (dense N²) ===")

    pos = system.positions
    radii = system.radii
    atom_mask = system.atom_mask
    N = pos.shape[0]
    mask_ij = atom_mask[:, None] & atom_mask[None, :]

    @jax.jit
    def compute_br():
        return compute_born_radii(
            pos, radii,
            mask=mask_ij.astype(jnp.float32),
            scaled_radii=system.scaled_radii,
        )

    return benchmark_fn("born_radii_dense", compute_br)


def test_jax_full_energy(system):
    """Benchmark: Full GB energy (born radii + Coulomb), no grad."""
    from prolix.physics.generalized_born import compute_gb_energy

    log.info("=== Test 4: Full GB Energy (dense, no grad) ===")

    pos = system.positions
    charges = system.charges
    radii = system.radii
    atom_mask = system.atom_mask
    N = pos.shape[0]
    mask_ij = atom_mask[:, None] & atom_mask[None, :]

    @jax.jit
    def compute_energy():
        return compute_gb_energy(
            pos, charges, radii,
            mask=mask_ij.astype(jnp.float32),
            scaled_radii=system.scaled_radii,
        )[0]

    return benchmark_fn("full_gb_energy", compute_energy)


def test_openmm(protein, n_atoms):
    """Benchmark: OpenMM GBSA on the same system."""
    log.info("=== Test 5: OpenMM GBSA Reference ===")

    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError:
        log.warning("OpenMM not available, skipping.")
        return None

    # Build OpenMM system from the protein
    pdb_path = os.path.join(
        os.path.dirname(__file__), "..", "references", "pdb",
        "1KEF_pdz_fixed.pdb"
    )

    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
    )

    # Use CUDA platform
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {"CudaPrecision": "single"}
    except Exception:
        platform = mm.Platform.getPlatformByName("CPU")
        properties = {}
        log.warning("CUDA platform not available for OpenMM, using CPU.")

    integrator = mm.LangevinMiddleIntegrator(
        310.15 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    simulation = app.Simulation(
        pdb.topology, system, integrator, platform, properties
    )
    simulation.context.setPositions(pdb.positions)

    # Benchmark: getState with forces
    def compute_forces():
        state = simulation.context.getState(getForces=True, getEnergy=True)
        return state

    t_openmm = benchmark_fn("openmm_gbsa", compute_forces)

    # Also benchmark a single step
    def single_step():
        simulation.step(1)

    t_step = benchmark_fn("openmm_step", single_step)

    return t_openmm, t_step


def test_jax_vmap_batch(system, batch_sizes=[1, 4, 16, 64, 125]):
    """Benchmark: vmap over batch of systems to see scaling."""
    from prolix.physics.analytical_forces import gb_ace_forces_dense

    log.info("=== Test 6: vmap Scaling Over Batch Size ===")

    for B in batch_sizes:
        # Stack the same system B times
        pos_batch = jnp.stack([system.positions] * B)
        charges_batch = jnp.stack([system.charges] * B)
        radii_batch = jnp.stack([system.radii] * B)
        scaled_batch = jnp.stack([system.scaled_radii] * B)
        mask_batch = jnp.stack([system.atom_mask] * B)

        @jax.jit
        def compute_batch():
            return jax.vmap(gb_ace_forces_dense)(
                pos_batch, charges_batch, radii_batch, scaled_batch, mask_batch
            )

        t = benchmark_fn(f"vmap_B={B}", compute_batch, n_warmup=2, n_iter=20)
        per_system = t / B
        log.info(f"    → {per_system:.3f} ms/system at B={B}")


def main():
    devices = jax.devices()
    log.info(f"Devices: {devices}")
    log.info(f"Device kind: {devices[0].device_kind}")

    # Load real PDZ system
    system, protein, n_atoms = load_real_system()
    log.info(f"System: {n_atoms} real atoms, padded to {system.n_padded_atoms}")

    results = {}

    # Test 1: Production analytical forces
    results["analytical"] = test_jax_analytical(system)

    # Test 2: Chunked fori_loop forces
    results["chunked"] = test_jax_chunked(system)

    # Test 3: Born radii only
    results["born_radii"] = test_jax_born_radii_only(system)

    # Test 4: Full energy (no grad)
    results["energy_only"] = test_jax_full_energy(system)

    # Test 5: OpenMM reference
    openmm_result = test_openmm(protein, n_atoms)
    if openmm_result:
        results["openmm_forces"], results["openmm_step"] = openmm_result

    # Test 6: vmap scaling
    test_jax_vmap_batch(system)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY (median ms, single 2048-atom system)")
    log.info("=" * 60)
    for name, t in results.items():
        if t is not None:
            log.info(f"  {name:25s}: {t:8.3f} ms")

    if "openmm_step" in results and "analytical" in results:
        ratio = results["analytical"] / results["openmm_step"]
        log.info(f"\n  JAX/OpenMM ratio: {ratio:.1f}x slower per single system")
        log.info(f"  Break-even batch size: ~{int(ratio)} systems via vmap")


if __name__ == "__main__":
    main()
