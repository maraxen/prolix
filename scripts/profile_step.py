#!/usr/bin/env python3
"""Profile every component of single_padded_force.

Breaks down the full MD force evaluation into its individual components
and times each one separately to find the bottleneck.

Components profiled:
  1. Bonded terms (bonds, angles, dihedrals, impropers, CMAP) via jax.grad
  2. Exclusion matrix construction (_build_dense_exclusion_scales)
  3. LJ analytical forces (dense N²)
  4. Coulomb analytical forces (dense N²)
  5. GB+ACE solvation forces (decomposed VJP)
  6. Full single_padded_force (end-to-end)
  7. Full force + BAOAB step (including integrator overhead)
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
log = logging.getLogger("profile")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prolix", "src"))


def load_system():
    """Load 1KEF PDZ system."""
    import dataclasses
    from proxide.io.parsing.backend import parse_structure
    from proxide import OutputSpec, CoordFormat
    from prolix.padding import pad_protein, select_bucket

    pdb_path = os.path.join(
        os.path.dirname(__file__), "..", "references", "pdb",
        "1KEF_pdz_fixed.pdb"
    )
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        parameterize_md=True,
        force_field="proxide/src/proxide/assets/protein.ff19SB.xml",
        add_hydrogens=False,
        remove_solvent=True,
        remove_hetatm=True,
    )
    protein = parse_structure(pdb_path, spec)

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
    system = pad_protein(
        protein, bucket,
        target_bonds=int(1.2 * bucket),
        target_angles=int(2.2 * bucket),
        target_dihedrals=int(3.5 * bucket),
        target_impropers=int(0.5 * bucket),
        target_cmaps=int(0.3 * bucket),
        target_constraints=int(0.7 * bucket),
    )
    return system, n_atoms


def bench(name, fn, n_warmup=3, n_iter=50):
    """Benchmark a function, return median ms."""
    for _ in range(n_warmup):
        r = fn()
        if hasattr(r, 'block_until_ready'):
            r.block_until_ready()
        elif isinstance(r, tuple):
            for x in r:
                if hasattr(x, 'block_until_ready'):
                    x.block_until_ready()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        r = fn()
        if hasattr(r, 'block_until_ready'):
            r.block_until_ready()
        elif isinstance(r, tuple):
            for x in r:
                if hasattr(x, 'block_until_ready'):
                    x.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    med = np.median(times)
    log.info(f"  {name:40s}: {med:8.3f} ms  (mean={np.mean(times):.3f}, std={np.std(times):.3f})")
    return med


def main():
    from jax_md import space
    from prolix.batched_energy import (
        _bond_energy_masked, _angle_energy_masked,
        _dihedral_energy_masked, _cmap_energy_masked,
        _build_dense_exclusion_scales,
        _lj_energy_masked, _coulomb_energy_masked,
    )
    from prolix.physics.analytical_forces import (
        lj_forces_dense, coulomb_forces_dense, gb_ace_forces_dense,
    )
    from prolix.physics.generalized_born import (
        compute_gb_energy, compute_born_radii, compute_ace_nonpolar_energy,
    )
    from prolix.batched_energy import single_padded_force

    devices = jax.devices()
    log.info(f"Device: {devices[0].device_kind}")

    sys, n_atoms = load_system()
    N = sys.positions.shape[0]
    log.info(f"System: {n_atoms} real atoms, padded to {N}")

    displacement_fn, _ = space.free()
    r = sys.positions

    # ===================================================================
    log.info("\n=== Per-Component Profiling (single system, N=%d) ===" % N)
    # ===================================================================

    # 1. Bonded energy (forward only, no grad)
    @jax.jit
    def bonded_energy_fwd():
        e_bond = _bond_energy_masked(r, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
        e_angle = _angle_energy_masked(r, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
        e_dih = _dihedral_energy_masked(r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
        e_imp = _dihedral_energy_masked(r, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
        e_cmap = _cmap_energy_masked(r, sys.cmap_torsions, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)
        return e_bond + e_angle + e_dih + e_imp + e_cmap

    t_bonded_fwd = bench("bonded_energy (forward)", bonded_energy_fwd)

    # 2. Bonded forces (via jax.grad — the production path)
    @jax.jit
    def bonded_forces():
        def _e(positions):
            e_bond = _bond_energy_masked(positions, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
            e_angle = _angle_energy_masked(positions, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
            e_dih = _dihedral_energy_masked(positions, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
            e_imp = _dihedral_energy_masked(positions, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
            e_cmap = _cmap_energy_masked(positions, sys.cmap_torsions, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)
            return e_bond + e_angle + e_dih + e_imp + e_cmap
        return -jax.grad(_e)(r)

    t_bonded_grad = bench("bonded_forces (jax.grad)", bonded_forces)

    # 3. Exclusion matrix construction
    @jax.jit
    def build_excl():
        vdw = _build_dense_exclusion_scales(sys.excl_indices, sys.excl_scales_vdw, N)
        elec = _build_dense_exclusion_scales(sys.excl_indices, sys.excl_scales_elec, N)
        return vdw, elec

    t_excl = bench("exclusion_matrices (fori_loop)", build_excl)

    # 4. LJ analytical forces
    excl_vdw, excl_elec = build_excl()
    excl_vdw.block_until_ready()
    excl_elec.block_until_ready()

    @jax.jit
    def lj_f():
        return lj_forces_dense(
            r, sys.sigmas, sys.epsilons, sys.atom_mask,
            soft_core_lambda=jnp.float32(1.0),
            excl_scale_vdw=excl_vdw,
        )

    t_lj = bench("lj_forces_dense (N²)", lj_f)

    # 5. Coulomb analytical forces
    @jax.jit
    def coulomb_f():
        return coulomb_forces_dense(
            r, sys.charges, sys.atom_mask,
            excl_scale_elec=excl_elec,
        )

    t_coul = bench("coulomb_forces_dense (N²)", coulomb_f)

    # 6. GB+ACE forces (decomposed VJP)
    @jax.jit
    def gb_f():
        return gb_ace_forces_dense(
            r, sys.charges, sys.radii, sys.scaled_radii, sys.atom_mask,
        )

    t_gb = bench("gb_ace_forces_dense (decomposed VJP)", gb_f)

    # 7. Full single_padded_force (end-to-end)
    import dataclasses

    @jax.jit
    def full_force():
        return single_padded_force(sys, displacement_fn, implicit_solvent=True, soft_core_lambda=jnp.float32(1.0))

    t_full = bench("single_padded_force (TOTAL)", full_force)

    # 8. Full BAOAB step (force + integrator)
    @jax.jit
    def baoab_step():
        mass_3d = sys.masses[:, None]
        pad_mask_3d = sys.atom_mask[:, None]
        key = jax.random.PRNGKey(42)
        v = jax.random.normal(key, r.shape) * 0.01
        dt = 0.002
        gamma = 1.0
        kT = 310.0 * 0.001987204

        # B: half-step velocity
        import dataclasses as dc
        sys_curr = dc.replace(sys, positions=r)
        f_phys = single_padded_force(sys_curr, displacement_fn, soft_core_lambda=jnp.float32(1.0))
        forces = f_phys * pad_mask_3d
        v_new = v + 0.5 * dt * (forces / (mass_3d + 1e-12))

        # A: half-step position
        r_new = r + 0.5 * dt * v_new

        # O: stochastic
        c1 = jnp.exp(-gamma * dt)
        c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))
        _, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, v.shape)
        v_new = c1 * v_new + c2 * jnp.sqrt(kT / (mass_3d + 1e-12)) * noise

        # A: second half-step position
        r_new = r_new + 0.5 * dt * v_new

        return r_new, v_new

    t_baoab = bench("full_baoab_step (force+integrator)", baoab_step)

    # ===================================================================
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — Component Breakdown (single 2048-atom system)")
    log.info("=" * 70)

    components = [
        ("bonded_energy (fwd)", t_bonded_fwd),
        ("bonded_forces (grad)", t_bonded_grad),
        ("exclusion_matrices", t_excl),
        ("lj_forces (N²)", t_lj),
        ("coulomb_forces (N²)", t_coul),
        ("gb_ace_forces (VJP)", t_gb),
    ]

    sum_parts = sum(t for _, t in components)

    log.info(f"\n  {'Component':40s}  {'Time (ms)':>10s}  {'% of Total':>10s}")
    log.info(f"  {'-'*40}  {'-'*10}  {'-'*10}")
    for name, t in components:
        pct = (t / t_full) * 100 if t_full > 0 else 0
        bar = "█" * int(pct / 2)
        log.info(f"  {name:40s}  {t:10.3f}  {pct:9.1f}%  {bar}")

    log.info(f"  {'-'*40}  {'-'*10}  {'-'*10}")
    log.info(f"  {'SUM of parts':40s}  {sum_parts:10.3f}  {(sum_parts/t_full)*100:9.1f}%")
    log.info(f"  {'single_padded_force (measured)':40s}  {t_full:10.3f}  {'100.0':>9s}%")
    overhead = t_full - sum_parts
    log.info(f"  {'Overhead (JIT/dispatch/PyTree)':40s}  {overhead:10.3f}  {(overhead/t_full)*100:9.1f}%")
    log.info(f"  {'full_baoab_step':40s}  {t_baoab:10.3f}")

    # Extrapolate to batched
    log.info("\n--- Extrapolation to Batched Simulation ---")
    for B in [64, 125, 128]:
        t_batch = t_full * B  # Linear scaling (vmap)
        steps_per_sec = 1000.0 / (t_batch)  # ms → steps/sec
        ns_per_day = steps_per_sec * 0.002 * 86400  # 2fs timestep
        log.info(
            f"  B={B:3d}: {t_batch:8.1f} ms/step, "
            f"{steps_per_sec:.0f} steps/s, "
            f"{ns_per_day:.0f} ns/day aggregate"
        )

    # OpenMM reference numbers
    log.info("\n--- OpenMM GPU Reference (web benchmarks) ---")
    log.info(f"  A100 GBSA:     ~1837 ns/day")
    log.info(f"  RTX 5090 GBSA: ~3600 ns/day")
    log.info(f"  A4000 GBSA:    ~1977 ns/day (2489 atoms)")


if __name__ == "__main__":
    main()
