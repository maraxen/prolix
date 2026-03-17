"""Batched simulation core for Prolix."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Any, TypeVar

import jax
import jax.numpy as jnp
from jax import lax
from jax_md import space, partition

from prolix.padding import PaddedSystem

if TYPE_CHECKING:
    from jax_md.util import Array

T = TypeVar("T")

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class LangevinState:
    """State for Langevin dynamics."""
    positions: Array  # (B, N, 3) or (N, 3)
    momentum: Array   # (B, N, 3) or (N, 3)
    force: Array      # (B, N, 3) or (N, 3)
    mass: Array       # (B, N) or (N,)
    key: Array        # (B, 2) or (2,)
    cap_count: Array  # Scalar or (B,) - accumulated force cap events

    def tree_flatten(self):
        children = (
            self.positions, self.momentum, self.force, 
            self.mass, self.key, self.cap_count
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# ---------------------------------------------------------------------------
# COM re-centering and rotation removal (implicit solvent)
# ---------------------------------------------------------------------------
# Per AMBER manual and OpenMM defaults: in Langevin dynamics without periodic
# boundaries, positions drift via Brownian diffusion. We re-center positions
# only — never touch velocities/momenta (that would destroy the thermostat).
# ---------------------------------------------------------------------------

DEFAULT_RECENTER_EVERY: int = 100  # steps between COM re-centering


def recenter_com(
    positions: 'Array',
    masses: 'Array',
    mask: 'Array | None' = None,
) -> 'Array':
    """Subtract mass-weighted center of mass from positions.

    Positions-only operation — velocities/momenta are NOT touched.
    This is the AMBER nscm approach for Langevin dynamics.

    Args:
        positions: (N, 3) atom coordinates.
        masses: (N,) per-atom masses.
        mask: Optional (N,) boolean mask. If given, only masked atoms
            contribute to COM calculation but ALL atoms are shifted.
    """
    m = masses[:, None]  # (N, 1)
    if mask is not None:
        m_weighted = m * mask[:, None].astype(m.dtype)
    else:
        m_weighted = m
    total_mass = jnp.sum(m_weighted) + 1e-30  # avoid div-by-zero for empty
    com = jnp.sum(positions * m_weighted, axis=0) / total_mass
    return positions - com


def remove_rotation_kabsch(
    positions: 'Array',
    reference: 'Array',
    masses: 'Array',
    mask: 'Array | None' = None,
) -> 'Array':
    """Remove global rotation via mass-weighted Kabsch superposition.

    Aligns `positions` onto `reference` using SVD. Both must already
    be COM-centered. Pure linear algebra — stays inside JIT.

    Args:
        positions: (N, 3) already COM-centered.
        reference: (N, 3) COM-centered reference frame.
        masses: (N,) per-atom masses.
        mask: Optional (N,) boolean mask for weighted alignment.
    """
    w = jnp.sqrt(masses[:, None])  # mass-weighting
    if mask is not None:
        w = w * mask[:, None].astype(w.dtype)

    # Cross-covariance matrix: H = (W*P)^T (W*R)
    P = positions * w  # (N, 3)
    R = reference * w  # (N, 3)
    H = P.T @ R  # (3, 3)

    U, _, Vt = jnp.linalg.svd(H)

    # Correct for reflection
    d = jnp.linalg.det(U @ Vt)
    sign_matrix = jnp.diag(jnp.array([1.0, 1.0, jnp.sign(d)]))

    # Optimal rotation: R_opt = V * sign * U^T
    R_opt = (Vt.T @ sign_matrix) @ U.T

    return positions @ R_opt.T


def apply_com_correction(
    positions: 'Array',
    masses: 'Array',
    reference: 'Array',
    step_idx: 'Array',
    recenter_every: int = DEFAULT_RECENTER_EVERY,
    remove_rotation: bool = True,
    mask: 'Array | None' = None,
) -> 'Array':
    """Conditionally apply COM re-centering and rotation removal.

    Applied every `recenter_every` steps via lax.cond to avoid
    branching overhead. All operations are pure JAX — no JIT exit.
    """
    def _do_correction(r):
        r_centered = recenter_com(r, masses, mask)
        if remove_rotation:
            r_centered = remove_rotation_kabsch(
                r_centered, reference, masses, mask,
            )
        return r_centered

    return lax.cond(
        step_idx % recenter_every == 0,
        _do_correction,
        lambda r: r,
        positions,
    )


def check_position_sanity(
    positions: Array, 
    atom_mask: Array, 
    max_magnitude: float, 
    stage: str,
    system_names: list[str] | None = None,
) -> None:
    """Assert no real atom exceeds max_magnitude. Hard error if violated.
    
    This runs on the host (Python side) using concrete arrays.
    Reports per-system diagnostics before raising so the user knows
    which system(s) exploded.
    """
    import logging as _logging
    _log = _logging.getLogger("check_position_sanity")

    # positions: (B, N, 3) or (N, 3); atom_mask: (B, N) or (N,)
    if positions.ndim == 2:
        positions = positions[None, ...]
        atom_mask = atom_mask[None, ...]
    B = positions.shape[0]

    failed_systems = []
    for i in range(B):
        real_mask = atom_mask[i] > 0  # (N,)
        real_pos_i = positions[i][real_mask]  # (n_real, 3)
        if real_pos_i.size == 0:
            continue
        max_abs_i = float(jnp.max(jnp.abs(real_pos_i)))
        name = system_names[i] if system_names and i < len(system_names) else f"sys_{i}"
        if max_abs_i > max_magnitude:
            _log.error(
                "  [%s] %s: max|coord|=%.1f Å  EXCEEDED %.0f Å threshold",
                stage, name, max_abs_i, max_magnitude,
            )
            failed_systems.append((name, max_abs_i))
        else:
            _log.info(
                "  [%s] %s: max|coord|=%.1f Å  OK",
                stage, name, max_abs_i,
            )

    if failed_systems:
        details = ", ".join(f"{n}={v:.1f}Å" for n, v in failed_systems)
        raise RuntimeError(
            f"MD Explosion detected at {stage} stage! "
            f"{len(failed_systems)}/{B} systems exceeded {max_magnitude} Å: {details}"
        )


def make_langevin_step(dt: float, kT: float, gamma: float) -> Callable[[PaddedSystem, LangevinState], LangevinState]:
    """Create a BAOAB Langevin step function with RATTLE constraints.
    
    BAOAB scheme with SHAKE/RATTLE:
    B: p = p + 0.5 * dt * f
    A: r = r + 0.5 * dt * p / m
    O: p = exp(-gamma * dt) * p + sqrt(m * kT * (1 - exp(-2 * gamma * dt))) * R
    A: r = r + 0.5 * dt * p / m
    [SHAKE: project positions to satisfy bond constraints]
    [adjust momentum for position correction]
    B: p = p + 0.5 * dt * f(r_constrained)
    [RATTLE: project momenta orthogonal to constrained bonds]

    Uses analytical forces for LJ/Coulomb (no jax.grad on nonbonded terms)
    to avoid autodiff NaN from exclusion matrix scatter and padded-atom distances.
    """
    from prolix.batched_energy import single_padded_force
    from prolix.physics.simulate import project_positions, project_momenta
    
    # Precompute constants
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))
    
    displacement_fn, shift_fn = space.free()
    
    def step_fn(padded_sys: PaddedSystem, state: LangevinState) -> LangevinState:
        def force_fn(r):
            sys_with_r = dataclasses.replace(padded_sys, positions=r)
            return single_padded_force(
                sys_with_r, displacement_fn,
                soft_core_lambda=jnp.float32(0.9999),
            )
        
        r, p, f, m, key, cap_count = (
            state.positions, state.momentum, state.force, 
            state.mass, state.key, state.cap_count
        )
        
        # Constraint data from PaddedSystem
        constr_pairs = padded_sys.constraint_pairs
        constr_lengths = padded_sys.constraint_lengths
        constr_mask = padded_sys.constraint_mask
        has_constraints = jnp.any(constr_mask)
        # Mass reshaped to (N, 1) for RATTLE functions
        mass_2d = m[:, None]
        
        # B: first half-step momentum update
        p = p - 0.5 * dt * f
        
        # A: first half-step position update
        r = r + 0.5 * dt * p / m[:, None]
        
        # O: stochastic (Ornstein-Uhlenbeck) update
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, p.shape)
        p = c1 * p + jnp.sqrt(m[:, None] * kT) * c2 * noise
        
        # A: second half-step position update
        r = r + 0.5 * dt * p / m[:, None]
        
        # --- SHAKE: project positions to satisfy bond constraints ---
        # Store pre-constraint positions for momentum correction
        r_pre_shake = r
        r = jax.lax.cond(
            has_constraints,
            lambda r_: project_positions(
                r_, constr_pairs, constr_lengths, mass_2d, shift_fn,
                constraint_mask=constr_mask,
            ),
            lambda r_: r_,
            r,
        )
        # Correct momentum for the position displacement from SHAKE
        # p_corrected = m * (r_post - r_pre) / dt (implicit constraint force)
        p = p + m[:, None] * (r - r_pre_shake) / dt
        
        # B: second half-step momentum update with new forces
        f = force_fn(r)
        f = jnp.where(jnp.isfinite(f), f, 0.0)
        # Belt-and-suspenders: zero forces ON padding atoms
        mask = padded_sys.atom_mask[:, None]
        f = f * mask
        # Safety cap — prevents explosion from residual steric clashes
        f_mag = jnp.sqrt(jnp.sum(f ** 2, axis=-1, keepdims=True) + 1e-12)
        capped = jnp.any(f_mag > 10000.0)
        f = f * jnp.minimum(1.0, 10000.0 / f_mag)
        p = p - 0.5 * dt * f
        
        # --- RATTLE: project momenta orthogonal to constrained bonds ---
        p = jax.lax.cond(
            has_constraints,
            lambda p_: project_momenta(
                p_, r, constr_pairs, mass_2d, shift_fn,
                constraint_mask=constr_mask,
            ),
            lambda p_: p_,
            p,
        )
        
        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

        # Ghost atom pinning: restore padding atoms to their initial
        # far-field positions (9999 Å). This prevents transient overlap
        # with real atoms at the origin during BAOAB sub-steps.
        r = jnp.where(mask, r, padded_sys.positions)
        p = jnp.where(mask, p, jnp.float32(0.0))
        
        return LangevinState(r, p, f, m, key, cap_count + capped.astype(jnp.int32))
        
    return step_fn


def make_langevin_step_nl(
    dt: float, kT: float, gamma: float,
    energy_fn: Callable | None = None,
) -> Callable:
    """Create a BAOAB Langevin step using neighbor-list energy.

    Same BAOAB scheme as make_langevin_step, but uses O(N*K) energy
    via single_padded_energy_nl instead of O(N^2).

    The neighbor_idx is passed as a separate argument (not in LangevinState)
    since it's updated less frequently than the dynamics state.

    Args:
        energy_fn: Optional custom energy function. If provided, used instead
            of single_padded_energy_nl. Useful for jax.checkpoint wrapping.

    Returns:
        step_fn(padded_sys, state, neighbor_idx) -> LangevinState
    """
    if energy_fn is None:
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        energy_fn = single_padded_energy_nl_cvjp

    # Precompute constants
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))

    displacement_fn, _ = space.free()

    def step_fn(
        padded_sys: PaddedSystem,
        state: LangevinState,
        neighbor_idx: 'Array',
    ) -> LangevinState:
        def _energy_of_r(r):
            sys_with_r = dataclasses.replace(padded_sys, positions=r)
            return energy_fn(
                sys_with_r, neighbor_idx, displacement_fn,
            )

        force_fn = jax.grad(lambda r: _energy_of_r(r))

        r, p, f, m, key, cap_count = (
            state.positions, state.momentum, state.force,
            state.mass, state.key, state.cap_count
        )

        # B
        p = p - 0.5 * dt * f

        # A
        r = r + 0.5 * dt * p / m[:, None]

        # O
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, p.shape)
        p = c1 * p + jnp.sqrt(m[:, None] * kT) * c2 * noise

        # A
        r = r + 0.5 * dt * p / m[:, None]

        # B
        f = force_fn(r)
        f = jnp.where(jnp.isfinite(f), f, 0.0)
        # Belt-and-suspenders: zero forces ON padding atoms
        mask = padded_sys.atom_mask[:, None]
        f = f * mask
        # Safety cap
        f_mag = jnp.sqrt(jnp.sum(f ** 2, axis=-1, keepdims=True) + 1e-12)
        capped = jnp.any(f_mag > 10000.0)
        f = f * jnp.minimum(1.0, 10000.0 / f_mag)
        p = p - 0.5 * dt * f

        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

        # Ghost atom pinning: restore padding to far-field positions
        r = jnp.where(mask, r, padded_sys.positions)
        p = jnp.where(mask, p, jnp.float32(0.0))

        return LangevinState(r, p, f, m, key, cap_count + capped.astype(jnp.int32))

    return step_fn

def make_langevin_step_nl_fused(
    dt: float, kT: float, gamma: float,
) -> Callable:
    """Create a BAOAB Langevin step using the fused analytical energy+force kernel.

    Bypasses jax.grad entirely for the dominant non-bonded terms, 
    reducing VRAM bandwidth and yielding massive speedups.
    """
    from prolix.fused_energy import fused_energy_and_forces_nl

    # Precompute constants
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))

    displacement_fn, _ = space.free()

    def step_fn(
        padded_sys: PaddedSystem,
        state: LangevinState,
        neighbor_idx: 'Array',
    ) -> LangevinState:
        def force_fn(r):
            sys_with_r = dataclasses.replace(padded_sys, positions=r)
            _e, f = fused_energy_and_forces_nl(
                sys_with_r, neighbor_idx, displacement_fn,
            )
            return f

        r, p, f, m, key, cap_count = (
            state.positions, state.momentum, state.force,
            state.mass, state.key, state.cap_count
        )

        # B
        p = p - 0.5 * dt * f

        # A
        r = r + 0.5 * dt * p / m[:, None]

        # O
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, p.shape)
        p = c1 * p + jnp.sqrt(m[:, None] * kT) * c2 * noise

        # A
        r = r + 0.5 * dt * p / m[:, None]

        # B
        f = force_fn(r)
        f = jnp.where(jnp.isfinite(f), f, 0.0)
        # Belt-and-suspenders: zero forces ON padding atoms
        mask = padded_sys.atom_mask[:, None]
        f = f * mask
        # Safety cap
        f_mag = jnp.sqrt(jnp.sum(f ** 2, axis=-1, keepdims=True) + 1e-12)
        capped = jnp.any(f_mag > 10000.0)
        f = f * jnp.minimum(1.0, 10000.0 / f_mag)
        p = p - 0.5 * dt * f

        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

        # Ghost atom pinning: restore padding to far-field positions
        r = jnp.where(mask, r, padded_sys.positions)
        p = jnp.where(mask, p, jnp.float32(0.0))

        return LangevinState(r, p, f, m, key, cap_count + capped.astype(jnp.int32))

    return step_fn

def safe_map(fn: Callable[[T], Any], batch: T, chunk_size: int | None = 1) -> Any:
    """Safely map a function over a batch, falling back to sequential execution for memory.
    
    If chunk_size is None, uses jax.vmap directly.
    Otherwise, chunks the batch along the leading dimension and uses lax.scan.
    """
    if chunk_size is None:
        return jax.vmap(fn)(batch)
    
    leaves, treedef = jax.tree_util.tree_flatten(batch)
    if not leaves:
        return jax.vmap(fn)(batch)
        
    B = leaves[0].shape[0]
    
    if B % chunk_size != 0:
        # Find largest divisor of B that is <= chunk_size
        import logging
        _log = logging.getLogger("batched_simulate")
        original = chunk_size
        for cs in range(chunk_size, 0, -1):
            if B % cs == 0:
                chunk_size = cs
                break
        _log.warning(
            "safe_map: B=%d not divisible by chunk_size=%d, "
            "using largest divisor %d instead",
            B, original, chunk_size,
        )
        
    num_chunks = max(1, B // chunk_size)

    # Short-circuit: if everything fits in one chunk, skip scan overhead
    if num_chunks == 1:
        return jax.vmap(fn)(batch)
    
    def map_fn(chunk_leaves):
        chunk_batch = jax.tree_util.tree_unflatten(treedef, chunk_leaves)
        return jax.vmap(fn)(chunk_batch)
        
    chunked_leaves = [l.reshape((num_chunks, chunk_size) + l.shape[1:]) for l in leaves]
    
    # We use lax.scan over the leading dimension (num_chunks) to avoid unrolling memory
    def scan_fn(carry, chunk_leaf):
        return carry, map_fn(chunk_leaf)
        
    _, results = lax.scan(scan_fn, None, chunked_leaves)
    
    def unchunk(x):
        return x.reshape((B,) + x.shape[2:])
        
    return jax.tree_util.tree_map(unchunk, results)

def batched_minimize(
    batch: PaddedSystem,
    max_steps: int = 5000,
    lbfgs_steps: int = 200,
    chunk_size: int | None = 1,
) -> Array:
    """Minimize a batch of PaddedSystems using hybrid FIRE + L-BFGS.

    Architecture Decision — Hybrid Minimization Protocol
    =====================================================
    We use a two-phase approach modeled on standard MD practice (cf. AMBER's
    steepest-descent → conjugate-gradient; GROMACS's emtol-gated switching):

    **Phase 1: FIRE (clash resolution)** — JAX-MD's Fast Inertial Relaxation
    Engine with staged soft-core potentials and force capping.

    FIRE was chosen over steepest descent or L-BFGS for initial clash
    resolution because:
      1. Fixed iteration count via lax.fori_loop — composes cleanly with
         vmap and pmap for massive GPU parallelism across heterogeneous
         protein batches.
      2. Momentum-based descent navigates highly curved, rugged protein
         energy landscapes more efficiently than gradient-only methods.
      3. Force-only convergence criterion avoids numerical artifacts in
         flat energy regions (a known L-BFGS failure mode).

    Soft-core LJ (Beutler 1994) is used in stages λ=0.1→0.999 to prevent
    the LJ r⁻¹² singularity from producing Inf energies at overlapping
    atom positions common in crystal structures. We never reach λ=1.0 in
    FIRE because residual overlaps produce forces >10¹⁶ kcal/mol/Å that
    exceed float32 range and FIRE's momentum amplifies them to NaN.

    **Phase 2: L-BFGS (final polish)** — jaxopt.LBFGS at the standard
    (λ=1.0) potential with line search.

    L-BFGS is used for final polishing because:
      1. Quadratic convergence near minima — far faster than FIRE for
         the "last mile" of relaxation on a smooth energy surface.
      2. Built-in Wolfe-condition line search inherently limits step
         sizes, preventing the blow-up that raw FIRE produces at λ=1.0.
      3. jaxopt.LBFGS.update() is a single-step API driven via
         lax.fori_loop — same fixed-iteration-count guarantee as FIRE,
         preserving vmap/pmap composability.

    This is analogous to OpenMM's L-BFGS LocalEnergyMinimizer, but split
    into two phases because OpenMM's L-BFGS has internal maximum-displacement
    limits (hardcoded in C++) that we cannot replicate in JAX-MD's FIRE.

    Args:
        batch: Padded batch of protein systems (B, N, 3).
        max_steps: Total FIRE steps distributed across soft-core stages.
        lbfgs_steps: L-BFGS polish steps at full potential (λ=1.0).
        chunk_size: Systems per vmap chunk (1 = sequential, safe for memory).

    Returns:
        minimized_positions: (B, N, 3)
    """
    from jax_md import minimize as jax_md_minimize
    from prolix.batched_energy import single_padded_energy, _position_restraint_energy

    displacement_fn, shift_fn = space.free()

    # === Phase 1: FIRE soft-core stages ===
    # Progressive λ ramp resolves steric clashes without LJ singularities.
    # Force caps prevent momentum accumulation from extreme gradients.
    # Final FIRE stages: heavy budget at λ=0.9999 to resolve clashes.
    # FIRE is a momentum-accelerated steepest descent — it's excellent at
    # escaping high-energy local minima that L-BFGS alone cannot resolve.
    stage_lambdas = jnp.array([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.9999],
                              dtype=jnp.float32)
    # Force caps halved vs prior values to prevent inertial overshoot.
    # NLM diagnosis: aggressive caps + FIRE momentum → overprojection artifact.
    stage_caps = jnp.array([5.0, 50.0, 500.0, 2000.0, 5000.0, 2000.0, 500.0],
                           dtype=jnp.float32)
    # Position restraints (kcal/mol/Å²): strong early tethering → release.
    # Prevents bonded geometry distortion during clash resolution.
    stage_restraints = jnp.array([100.0, 50.0, 10.0, 1.0, 0.0, 0.0, 0.0],
                                 dtype=jnp.float32)
    # Give heavy budget to λ=0.9999 stages (stages 5 & 6)
    remaining = max(max_steps - 4000, 1000)
    stage_steps = jnp.array([500, 500, 500, 500, 1000, remaining, 2000])

    # Tightened FIRE dt: 0.15fs prevents inertial overshoot on stiff
    # soft-core walls (was 0.5fs, caused oscillatory blowup per sweep).
    dt_start = 0.00005
    dt_max = 0.00015

    def minimize_single(sys: PaddedSystem) -> Array:
        r_ref = sys.positions  # Reference geometry for restraints
        # Padding atom mask: (N, 1) for broadcasting against (N, 3)
        pad_mask_3d = sys.atom_mask[:, None]  # True for real atoms

        # Parameterized energy function — lambda and k_restraint are JAX arrays,
        # NOT Python floats, so this traces once for all stages.
        def energy_fn(
            r,
            soft_core_lambda=jnp.float32(1.0),
            k_restraint=jnp.float32(0.0),
        ):
            sys_with_r = dataclasses.replace(sys, positions=r)
            e = single_padded_energy(
                sys_with_r, displacement_fn,
                soft_core_lambda=soft_core_lambda,
            )
            # Harmonic position restraints prevent bonded geometry distortion
            e = e + _position_restraint_energy(
                r, r_ref, k_restraint, sys.atom_mask,
            )
            return e

        # --- Phase 1: FIRE with soft-core + position restraints ---
        fire_init_fn, fire_apply_fn = jax_md_minimize.fire_descent(
            energy_fn, shift_fn,
            dt_start=dt_start, dt_max=dt_max,
        )

        def capped_apply(state, force_cap, sc_lam, k_restr):
            """Apply FIRE step with runtime force capping + padding masking."""
            f = state.force
            # Zero forces on padding atoms — prevents jax.grad leakage
            # from dense N² interactions from moving padding atoms,
            # whose drift corrupts COM and destabilises real atoms.
            f = f * pad_mask_3d
            f_norm = jnp.linalg.norm(f, axis=-1, keepdims=True)
            use_cap = force_cap > 0.0
            cap_ratio = jnp.where(
                use_cap,
                jnp.minimum(1.0, force_cap / (f_norm + 1e-8)),
                1.0,
            )
            f_capped = f * cap_ratio
            f_safe = jnp.where(jnp.isfinite(f_capped), f_capped, 0.0)
            state = dataclasses.replace(state, force=f_safe)
            return fire_apply_fn(
                state, soft_core_lambda=sc_lam, k_restraint=k_restr,
            )

        def run_stage(carry, stage_params):
            """Run one FIRE minimization stage (called via lax.scan)."""
            positions = carry
            sc_lam, force_cap, k_restr, n_steps = stage_params

            fire_state = fire_init_fn(
                positions, mass=sys.masses,
                soft_core_lambda=sc_lam, k_restraint=k_restr,
            )

            def body_fn(i, s):
                s = capped_apply(s, force_cap, sc_lam, k_restr)
                # Reset padding atoms to initial positions with zero velocity.
                # FIRE's velocity Verlet moves ALL atoms; without this,
                # padding atoms accumulate momentum and drift, corrupting
                # the energy landscape for real atoms.
                new_pos = jnp.where(pad_mask_3d, s.position, r_ref)
                new_mom = jnp.where(pad_mask_3d, s.momentum, 0.0)
                new_force = jnp.where(pad_mask_3d, s.force, 0.0)
                return dataclasses.replace(
                    s, position=new_pos, momentum=new_mom, force=new_force,
                )

            final_state = jax.lax.fori_loop(0, n_steps, body_fn, fire_state)
            return final_state.position, None

        stage_params = (
            stage_lambdas, stage_caps, stage_restraints, stage_steps,
        )
        fire_positions, _ = lax.scan(
            run_stage, sys.positions, stage_params,
        )

        # --- Phase 2: L-BFGS polish at full potential (λ=1.0) ---
        # After FIRE has resolved clashes at λ=0.999, L-BFGS refines to
        # a true minimum on the standard (λ=1.0) energy surface. The
        # line search naturally limits step sizes, so residual close
        # contacts don't cause blow-up (unlike FIRE's momentum).
        if lbfgs_steps > 0:
            import jaxopt

            # Build a mask for real (non-padding) atoms.
            # Padding atoms have mass=0 and positions at 9999.0.
            # We must zero their gradients so L-BFGS doesn't move them.
            real_mask = sys.masses > 0.0  # (N,) bool
            real_mask_3d = real_mask[:, None]  # (N, 1) for broadcasting

            def lbfgs_value_and_grad(r):
                val, grad = jax.value_and_grad(
                    lambda pos: energy_fn(
                        pos, soft_core_lambda=jnp.float32(0.9999))
                )(r)
                # Zero gradient on padding atoms — prevents L-BFGS from
                # moving sentinel positions (9999.0) that are not physical
                grad = jnp.where(real_mask_3d, grad, 0.0)
                # Cap gradient magnitude instead of zeroing NaN — preserves
                # gradient direction so L-BFGS can resolve overlaps.
                # NaN/Inf gradients from residual sub-Å overlaps would
                # otherwise be zeroed, leaving atoms permanently stuck.
                grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
                g_mag = jnp.sqrt(
                    jnp.sum(grad ** 2, axis=-1, keepdims=True) + 1e-30)
                g_scale = jnp.minimum(1.0, 100000.0 / g_mag)
                grad = grad * g_scale
                # Sanitize any remaining pathological values
                val = jnp.where(jnp.isfinite(val), val, jnp.float32(1e10))
                return val, grad

            solver = jaxopt.LBFGS(
                fun=lbfgs_value_and_grad,
                value_and_grad=True,  # We provide (value, grad) directly
                maxiter=lbfgs_steps,
                tol=1e-6,           # Force convergence tolerance (strict)
                history_size=10,    # L-BFGS memory (standard for proteins)
                max_stepsize=0.1,   # Prevent overshooting with large clipped gradients
                min_stepsize=1e-10,
                linesearch="backtracking",
                stop_if_linesearch_fails=True,
                jit=True,           # Use lax.while_loop (not Python loop)
                unroll=False,       # Don't unroll — use while_loop for JIT
            )

            # solver.run() uses lax.while_loop internally with maxiter as
            # the iteration limit. This is compatible with JIT but NOT with
            # vmap (while_loop iteration count varies per system). That's
            # fine — we dispatch per-system via safe_map(chunk_size=1).
            lbfgs_positions, lbfgs_state = solver.run(fire_positions)

            # Restore padding positions exactly (belt-and-suspenders)
            final_positions = jnp.where(real_mask_3d, lbfgs_positions,
                                        fire_positions)
        else:
            final_positions = fire_positions

        # === Post-minimization convergence diagnostic ===
        # Compute forces at λ=0.9999 on final positions to check convergence.
        # Standard MD practice: RMS gradient should be < ~10 kcal/mol/Å
        # before starting dynamics (AMBER default drms = 1e-4, but that's
        # for equilibrium; for clash resolution, < 10 is pragmatic).
        final_f_val, final_g = jax.value_and_grad(
            lambda pos: energy_fn(pos, soft_core_lambda=jnp.float32(0.9999))
        )(final_positions)
        final_g = jnp.where(real_mask_3d, final_g, 0.0)
        final_g = jnp.where(jnp.isfinite(final_g), final_g, 0.0)
        g_per_atom = jnp.sqrt(jnp.sum(final_g ** 2, axis=-1))  # (N,)
        g_real = g_per_atom * (sys.masses > 0.0)  # mask padding
        n_real = jnp.sum(sys.masses > 0.0)
        rms_grad = jnp.sqrt(jnp.sum(g_real ** 2) / jnp.maximum(n_real, 1.0))
        max_grad = jnp.max(g_real)
        n_high = jnp.sum(g_real > 100.0)
        jax.debug.print(
            "  [min_conv] E={e:.1f} rms_grad={rms:.1f} max_grad={mx:.1f}"
            " n_atoms_grad>100={n}",
            e=final_f_val, rms=rms_grad, mx=max_grad, n=n_high,
        )

        # === Validation gate: post-minimization convergence ===
        # GROMACS default emtol ~2.4 kcal/mol/Å, AMBER drms ~1e-4.
        # For clash-resolution minimization, pragmatic thresholds are
        # rms_grad < 50 and max_grad < 1000.  Exceeding these means
        # equilibration will likely encounter extreme forces.
        min_converged = jnp.logical_and(rms_grad < 50.0, max_grad < 1000.0)
        jax.debug.print(
            "  [min_gate] converged={ok} (rms<50={r}, max<1000={m})",
            ok=min_converged,
            r=(rms_grad < 50.0),
            m=(max_grad < 1000.0),
        )

        return final_positions

    return safe_map(minimize_single, batch, chunk_size=chunk_size)

def batched_equilibrate(
    batch: PaddedSystem, 
    key: Array, 
    n_steps: int, 
    temperature_k: float = 310.15, 
    chunk_size: int | None = None
) -> LangevinState:
    """Equilibrate a batch of padded systems with 10-stage staged warmup.

    Uses a 10-stage protocol with ultra-gradual temperature ramp and
    position restraints that are NEVER fully released. The additional stages
    (vs prior 7-stage) prevent the balloon inflation artifact where
    MAX_FORCE capping + thermal noise → systematic outward drift.

      Stages 0-2: dt=0.01-0.05fs, T=0.1-2K, max_dx=0.002-0.005Å (quench)
      Stages 3-4: dt=0.1-0.2fs,   T=10-50K, max_dx=0.01-0.02Å  (warmup)
      Stages 5-7: dt=0.5-1.0fs,   T=100-200K, max_dx=0.03-0.05Å (ramp)
      Stages 8-9: dt=1.0-2.0fs,   T=target,   max_dx=0.08-0.1Å  (settle)

    dt and kT are JAX scalars inside the step function, so this compiles
    once and the stages vary only the scalar inputs (no shape changes).
    """
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
    from prolix.batched_energy import (
        single_padded_energy, single_padded_force, _position_restraint_energy,
    )
    
    displacement_fn, _ = space.free()
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3




    # 10-stage warmup: gentler ramp prevents balloon inflation.
    # Prior 7-stage protocol had too-aggressive jumps at high T, where
    # MAX_FORCE=10k + thermal noise → systematic outward drift.
    # Now: 3 ultra-low-T stages (0.1-2K), 3 warmup (10-150K), 4 approach.
    n_stages = 10
    s_each = max(n_steps // n_stages, 100)
    s_last = max(n_steps - 9 * s_each, 100)

    stage_dts = jnp.array(
        [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.5, 1.0, 1.0, 2.0],
        dtype=jnp.float32,
    ) / AKMA_TIME_UNIT_FS
    stage_kTs = jnp.array(
        [0.1, 0.5, 2.0, 10.0, 50.0, 100.0, 150.0, 200.0,
         temperature_k, temperature_k],
        dtype=jnp.float32,
    ) * BOLTZMANN_KCAL
    # Displacement caps: ultra-tight in early stages, relax gradually.
    # 0.002Å/step × 1000 steps = 2Å max cumulative drift (vs 70Å before).
    stage_force_caps = jnp.array(
        [0.002, 0.005, 0.005, 0.01, 0.02, 0.03, 0.05, 0.05, 0.08, 0.1],
        dtype=jnp.float32,
    )  # max displacement per step (Å)
    # Position restraint spring constants (kcal/mol/Å²): NEVER go to 0.
    # Strong early → gradual release. Final stages keep k=0.5 as safety net.
    stage_restraint_k = jnp.array(
        [10000.0, 5000.0, 1000.0, 500.0, 100.0, 10.0, 5.0, 1.0, 0.5, 0.5],
        dtype=jnp.float32,
    )
    stage_step_counts = jnp.array(
        [s_each, s_each, s_each, s_each, s_each, s_each, s_each, s_each,
         s_each, s_last],
    )

    leaves, _ = jax.tree_util.tree_flatten(batch)
    B = leaves[0].shape[0]
    keys = jax.random.split(key, B)
    
    def equilibrate_single(args: tuple[PaddedSystem, Array]) -> LangevinState:
        sys, k = args
        # Capture initial padding positions for ghost atom pinning.
        # Padding atoms sit at far-field (9999 Å) — we restore them here
        # after every Langevin step to prevent drift or overlap.
        ghost_positions = sys.positions
        
        # Use λ=0.9999 soft-core to match the minimization surface.
        # The full potential (λ=1.0) has r⁻¹² singularities that produce
        # 10¹⁰+ kcal/mol/Å forces at sub-Å overlaps remaining after
        # minimization. The α*(1-0.9999) = 5e-5 cushion is physically
        # negligible but numerically essential for float32 stability.
        _eq_lambda = jnp.float32(0.9999)

        # Reference geometry for position restraints — the post-minimization
        # structure. Restraints gradually release during equilibration.
        r_ref = sys.positions

        def energy_fn_with_restraint(r, k_restraint):
            """Physics energy + harmonic position restraint."""
            sys_with_r = dataclasses.replace(sys, positions=r)
            e_phys = single_padded_energy(
                sys_with_r, displacement_fn,
                soft_core_lambda=_eq_lambda,
            )
            e_restraint = _position_restraint_energy(
                r, r_ref, k_restraint, sys.atom_mask,
            )
            return e_phys + e_restraint

        def force_fn_with_restraint(r, k_restraint):
            """Negative gradient of energy (physics + restraint)."""
            # Restraint force is analytical: -k * (r - r_ref) * mask
            sys_with_r = dataclasses.replace(sys, positions=r)
            f_phys = single_padded_force(
                sys_with_r, displacement_fn,
                soft_core_lambda=_eq_lambda,
            )
            # Analytical restraint force: -dE/dr = -k*(r - r_ref)
            f_restraint = -k_restraint * (r - r_ref) * sys.atom_mask[:, None]
            return f_phys + f_restraint

        def force_fn(r):
            """Unrestrained force for diagnostics."""
            sys_with_r = dataclasses.replace(sys, positions=r)
            return single_padded_force(
                sys_with_r, displacement_fn,
                soft_core_lambda=_eq_lambda,
            )

        def langevin_step(state: LangevinState, dt, kT, force_cap, k_restraint):
            """BAOAB Langevin step with displacement capping + position restraints.
            
            Uses maximum-displacement-per-step instead of force capping.
            Force caps cause systematic balloon inflation when forces
            persistently exceed the cap — atoms receive constant-velocity
            drift. Displacement caps preserve the force landscape while
            preventing large excursions.
            
            Position restraints tether atoms to the minimized reference,
            preventing structural deformation during early warmup stages.
            Standard NVT equilibration protocol (NotebookLM grounded).
            
            max_dx: force_cap parameter reinterpreted as max displacement (Å).
            k_restraint: harmonic spring constant (kcal/mol/Å²), 0 = off.
            """
            c1 = jnp.exp(-gamma * dt)
            c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))

            r, p, f, m, key, cap_count = (
                state.positions, state.momentum, state.force,
                state.mass, state.key, state.cap_count
            )
            r_old = r

            # === Triple safety net ===
            # 1. FORCE CAP: prevent extreme forces from creating large
            #    momenta that displacement capping can't fully suppress.
            #    10000 kcal/mol/Å ≈ maximum covalent bond force.
            MAX_FORCE = jnp.float32(2000.0)
            f_mag = jnp.sqrt(
                jnp.sum(f ** 2, axis=-1, keepdims=True) + 1e-30)
            f_scale = jnp.minimum(1.0, MAX_FORCE / f_mag)
            f_capped = f * f_scale

            # B (first half-kick with capped forces)
            p = p - 0.5 * dt * f_capped
            # A
            r = r + 0.5 * dt * p / m[:, None]
            # O (thermostat)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, p.shape)
            p = c1 * p + jnp.sqrt(m[:, None] * kT) * c2 * noise
            # A
            r = r + 0.5 * dt * p / m[:, None]

            # 2. DISPLACEMENT CAP: limit max per-atom displacement
            max_dx = force_cap
            dr = r - r_old
            dr_mag = jnp.sqrt(jnp.sum(dr ** 2, axis=-1, keepdims=True) + 1e-30)
            cap_active = max_dx > 0
            scale = jnp.where(
                cap_active,
                jnp.minimum(1.0, max_dx / dr_mag),
                1.0,
            )
            capped = jnp.where(cap_active, jnp.any(dr_mag > max_dx), False)
            r = r_old + dr * scale
            # Adjust momentum to match the actual displacement
            p = jnp.where(
                cap_active & (dr_mag > max_dx),
                p * scale,
                p,
            )

            # --- SHAKE: project positions to satisfy bond constraints ---
            from prolix.physics.simulate import project_positions, project_momenta
            constr_pairs = sys.constraint_pairs
            constr_lengths = sys.constraint_lengths
            constr_mask = sys.constraint_mask
            has_constraints = jnp.any(constr_mask)
            mass_2d = m[:, None]

            r_pre_shake = r
            r = jax.lax.cond(
                has_constraints,
                lambda r_: project_positions(
                    r_, constr_pairs, constr_lengths, mass_2d,
                    shift_fn=lambda x, dx: x + dx,
                    constraint_mask=constr_mask,
                ),
                lambda r_: r_,
                r,
            )
            # Correct momentum for SHAKE displacement
            p = p + m[:, None] * (r - r_pre_shake) / dt

            # B (force evaluation at the capped position, with restraints)
            # 3. POSITION RESTRAINTS: tether to reference, prevent drift
            f = force_fn_with_restraint(r, k_restraint)
            # Apply force cap to new forces too
            f_mag2 = jnp.sqrt(
                jnp.sum(f ** 2, axis=-1, keepdims=True) + 1e-30)
            f = f * jnp.minimum(1.0, MAX_FORCE / f_mag2)
            # Belt-and-suspenders: zero forces ON padding atoms
            pad_mask = sys.atom_mask[:, None]
            f = f * pad_mask
            f = jnp.where(jnp.isfinite(f), f, 0.0)

            p = p - 0.5 * dt * f

            # --- RATTLE: project momenta orthogonal to constrained bonds ---
            p = jax.lax.cond(
                has_constraints,
                lambda p_: project_momenta(
                    p_, r, constr_pairs, mass_2d,
                    shift_fn=lambda x, dx: x + dx,
                    constraint_mask=constr_mask,
                ),
                lambda p_: p_,
                p,
            )

            # Ghost atom pinning: restore padding atoms to their initial
            # far-field positions. Prevents transient overlap with real
            # atoms and ensures zero contribution to dynamics.
            r = jnp.where(pad_mask, r, ghost_positions)
            p = jnp.where(pad_mask, p, jnp.float32(0.0))

            return LangevinState(
                r.astype(state.positions.dtype),
                p.astype(state.momentum.dtype),
                f.astype(state.force.dtype),
                m, key,
                cap_count + capped.astype(jnp.int32),
            )

        initial_f = force_fn(sys.positions)
        # Belt-and-suspenders: mask forces on padding atoms
        initial_f = initial_f * sys.atom_mask[:, None]
        # Sanitize NaN/Inf forces — these can occur from residual sub-Å
        # contacts at λ=1.0. Displacement capping in the Langevin step
        # handles force magnitude safety, so we only need NaN cleanup.
        initial_f = jnp.where(jnp.isfinite(initial_f), initial_f, 0.0)

        # Diagnostic: report initial force magnitude via host callback
        f_mag = jnp.sqrt(jnp.sum(initial_f ** 2, axis=-1) + 1e-12)
        max_f_mag = jnp.max(f_mag)
        max_r_mag = jnp.max(jnp.abs(sys.positions * sys.atom_mask[:, None]))
        n_high_f = jnp.sum(f_mag > 500.0)
        jax.debug.print(
            "  [eq_diag] initial: max_f={f:.1f} max_r={r:.1f} n_high_f={n}",
            f=max_f_mag, r=max_r_mag, n=n_high_f,
        )

        # COM-centered reference for rotation removal
        ref_positions = recenter_com(sys.positions, sys.masses, mask=sys.atom_mask)
        state = LangevinState(
            positions=ref_positions,  # start COM-centered
            momentum=jnp.zeros_like(sys.positions),
            force=initial_f,
            mass=sys.masses,
            key=k,
            cap_count=jnp.array(0, dtype=jnp.int32),
        )

        def run_stage(carry, stage_params):
            """Run one equilibration stage via lax.fori_loop."""
            s = carry
            dt, kT, fcap, k_rest, n = stage_params

            def body_fn(step_i, s_inner):
                s_next = langevin_step(s_inner, dt, kT, fcap, k_rest)
                # COM re-centering every DEFAULT_RECENTER_EVERY steps
                r_corrected = apply_com_correction(
                    s_next.positions, sys.masses, ref_positions,
                    step_i, DEFAULT_RECENTER_EVERY, True,
                    mask=sys.atom_mask,
                )
                return LangevinState(
                    r_corrected, s_next.momentum, s_next.force,
                    s_next.mass, s_next.key, s_next.cap_count
                )

            final_s = lax.fori_loop(0, n, body_fn, s)
            max_r = jnp.max(jnp.abs(final_s.positions * sys.atom_mask[:, None]))
            max_f_real = jnp.max(jnp.sqrt(
                jnp.sum(final_s.force ** 2, axis=-1) + 1e-12
            ) * sys.atom_mask)
            jax.debug.print(
                "  [eq_stage] dt={dt:.4f} kT={kT:.4f} cap={cap} k_rest={kr}"
                " steps={n} => max_r={r:.1f} max_f={f:.1f}",
                dt=dt, kT=kT, cap=fcap, kr=k_rest, n=n,
                r=max_r, f=max_f_real,
            )
            return final_s, None

        stage_params = (
            stage_dts, stage_kTs, stage_force_caps,
            stage_restraint_k, stage_step_counts,
        )
        final_state, _ = lax.scan(
            run_stage, state, stage_params,
        )
        # === Validation gate: post-equilibration structural quality ===
        # Check that the protein has not inflated during equilibration.
        # Compute Rg before and after, plus max per-atom displacement.
        def compute_rg(positions, masses, mask):
            """Radius of gyration for real atoms only."""
            m = masses * mask  # zero out padding
            total_m = jnp.sum(m) + 1e-12
            com = jnp.sum(positions * m[:, None], axis=0) / total_m
            dr = positions - com[None, :]
            rg2 = jnp.sum(m * jnp.sum(dr ** 2, axis=-1)) / total_m
            return jnp.sqrt(rg2 + 1e-12)

        rg_init = compute_rg(ref_positions, sys.masses, sys.atom_mask)
        rg_final = compute_rg(final_state.positions, sys.masses, sys.atom_mask)
        rg_ratio = rg_final / (rg_init + 1e-8)

        # Max per-atom displacement from minimized reference
        disp = jnp.sqrt(jnp.sum(
            (final_state.positions - ref_positions) ** 2, axis=-1) + 1e-12
        ) * sys.atom_mask
        max_disp = jnp.max(disp)

        eq_ok = jnp.logical_and(rg_ratio < 1.5, max_disp < 15.0)
        jax.debug.print(
            "  [eq_gate] Rg_init={ri:.1f} Rg_final={rf:.1f} ratio={rat:.2f}"
            " max_disp={md:.1f} PASS={ok}",
            ri=rg_init, rf=rg_final, rat=rg_ratio, md=max_disp, ok=eq_ok,
        )

        return final_state
        
    return safe_map(equilibrate_single, (batch, keys), chunk_size=chunk_size)

def batched_produce(
    batch: PaddedSystem, 
    state: LangevinState, 
    n_saves: int, 
    steps_per_save: int, 
    temperature_k: float = 310.15,
    chunk_size: int | None = None
) -> tuple[LangevinState, Array]:
    """Produces trajectory for a batch of padded systems."""
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
    
    dt = 2.0 / AKMA_TIME_UNIT_FS
    kT = temperature_k * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3
    
    step_fn = make_langevin_step(dt, kT, gamma)
    
    def produce_single(args: tuple[PaddedSystem, LangevinState]) -> tuple[LangevinState, Array]:
        sys, s = args
        
        def save_step(s, _):
            def single_step(s_inner, _):
                return step_fn(sys, s_inner), None
            s_next, _ = lax.scan(single_step, s, None, length=steps_per_save)
            return s_next, s_next.positions
            
        final_s, traj = lax.scan(save_step, s, None, length=n_saves)
        return final_s, traj
        
    return safe_map(produce_single, (batch, state), chunk_size=chunk_size)


def safe_map_no_output(
    fn: Callable[[T], Any], batch: T, chunk_size: int | None = 1
) -> Any:
    """Map a function over a batch, discarding scan outputs.

    Like safe_map, but for side-effect-only functions (e.g., streaming IO).
    Returns only the final carry (function results), not accumulated outputs.
    """
    if chunk_size is None:
        return jax.vmap(fn)(batch)

    leaves, treedef = jax.tree_util.tree_flatten(batch)
    if not leaves:
        return jax.vmap(fn)(batch)

    B = leaves[0].shape[0]

    if B % chunk_size != 0:
        import logging
        _log = logging.getLogger("batched_simulate")
        original = chunk_size
        for cs in range(chunk_size, 0, -1):
            if B % cs == 0:
                chunk_size = cs
                break
        _log.warning(
            "safe_map_no_output: B=%d not divisible by chunk_size=%d, "
            "using largest divisor %d instead",
            B, original, chunk_size,
        )

    num_chunks = max(1, B // chunk_size)

    # Short-circuit: if everything fits in one chunk, skip scan overhead
    if num_chunks == 1:
        return jax.vmap(fn)(batch)

    def map_fn(chunk_leaves):
        chunk_batch = jax.tree_util.tree_unflatten(treedef, chunk_leaves)
        return jax.vmap(fn)(chunk_batch)

    chunked_leaves = [
        l.reshape((num_chunks, chunk_size) + l.shape[1:]) for l in leaves
    ]

    def scan_fn(carry, chunk_leaf):
        result = map_fn(chunk_leaf)
        return carry, result

    _, results = lax.scan(scan_fn, None, chunked_leaves)

    def unchunk(x):
        return x.reshape((B,) + x.shape[2:])

    return jax.tree_util.tree_map(unchunk, results)


def batched_produce_streaming(
    batch: PaddedSystem,
    state: LangevinState,
    n_saves: int,
    steps_per_save: int,
    write_fn: Callable,
    temperature_k: float = 310.15,
    chunk_size: int | None = None,
    recenter_every: int = DEFAULT_RECENTER_EVERY,
    remove_rotation: bool = True,
    write_batch_size: int = 1,
) -> LangevinState:
    """Produce trajectory with streaming IO — no GPU trajectory accumulation.

    Instead of accumulating a (n_saves, N, 3) trajectory tensor on GPU,
    this function uses jax.experimental.io_callback to stream each frame
    to the host as it's produced. The host-side write_fn handles serialization
    and disk IO asynchronously while the GPU continues computing.

    COM re-centering (and optional rotation removal) is applied every
    `recenter_every` MD steps per AMBER/OpenMM standards for implicit solvent.
    Only positions are modified — velocities/momenta are never touched.

    When write_batch_size > 1, frames are accumulated on GPU and flushed
    in batches to reduce io_callback overhead.

    Args:
        batch: Batched padded systems (B, N, 3).
        state: Initial Langevin state from equilibration.
        n_saves: Number of trajectory frames to save.
        steps_per_save: MD steps between saves.
        write_fn: Host callback. If write_batch_size == 1:
            (positions, batch_idx, save_idx) -> None.
            If write_batch_size > 1:
            (positions_batch, batch_idx, start_save_idx) -> None
            where positions_batch is (write_batch_size, N, 3).
        temperature_k: Temperature in Kelvin.
        chunk_size: Chunk size for safe_map. None = full vmap.
        recenter_every: Apply COM re-centering every N MD steps.
        remove_rotation: If True, also remove global rotation via Kabsch.
        write_batch_size: Number of frames to accumulate before IO callback.
            Must divide n_saves evenly. 1 = write every frame (original).

    Returns:
        Final LangevinState (no trajectory — it was streamed to disk).
    """
    from jax.experimental import io_callback
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    dt = 2.0 / AKMA_TIME_UNIT_FS
    kT = temperature_k * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3

    step_fn = make_langevin_step(dt, kT, gamma)

    # Get batch size for index computation
    leaves, _ = jax.tree_util.tree_flatten(batch)
    B = leaves[0].shape[0]

    def produce_single_streaming(
        args: tuple[PaddedSystem, LangevinState, Any],
    ) -> LangevinState:
        sys, s, batch_idx = args

        # Reference frame for rotation removal: COM-centered initial positions
        ref_positions = recenter_com(s.positions, sys.masses, mask=sys.atom_mask)

        if write_batch_size > 1:
            # --- Batched IO path: accumulate frames, flush in batches ---
            assert n_saves % write_batch_size == 0, (
                f"write_batch_size={write_batch_size} must divide "
                f"n_saves={n_saves}"
            )
            n_batches = n_saves // write_batch_size

            def batch_step(carry, batch_save_idx):
                s_carry, global_step = carry

                def save_step(inner_carry, _local_idx):
                    s_inner, step_count = inner_carry

                    # Run steps_per_save MD steps with COM correction
                    def single_step(step_carry, _):
                        s_md, step_i = step_carry
                        s_next = step_fn(sys, s_md)
                        # COM re-centering (positions only, no velocity touch)
                        r_corrected = apply_com_correction(
                            s_next.positions, sys.masses, ref_positions,
                            step_i, recenter_every, remove_rotation,
                            mask=sys.atom_mask,
                        )
                        s_next = LangevinState(
                            r_corrected, s_next.momentum, s_next.force,
                            s_next.mass, s_next.key, s_next.cap_count
                        )
                        return (s_next, step_i + 1), None

                    (s_after, new_step_count), _ = lax.scan(
                        single_step, (s_inner, step_count),
                        None, length=steps_per_save,
                    )
                    return (s_after, new_step_count), s_after.positions

                # Accumulate write_batch_size frames
                (s_after_batch, new_global_step), frames = lax.scan(
                    save_step, (s_carry, global_step),
                    None, length=write_batch_size,
                )
                # frames: (write_batch_size, N, 3)

                start_save_idx = batch_save_idx * write_batch_size
                io_callback(
                    write_fn,
                    None,
                    frames,
                    batch_idx,
                    start_save_idx,
                )

                return (s_after_batch, new_global_step), None

            (final_s, _), _ = lax.scan(
                batch_step, (s, jnp.int32(0)),
                jnp.arange(n_batches),
            )

        else:
            # --- Original per-frame IO path with COM correction ---
            def save_step(carry, save_idx):
                s_carry, global_step = carry

                def single_step(step_carry, _):
                    s_inner, step_i = step_carry
                    s_next = step_fn(sys, s_inner)
                    r_corrected = apply_com_correction(
                        s_next.positions, sys.masses, ref_positions,
                        step_i, recenter_every, remove_rotation,
                        mask=sys.atom_mask,
                    )
                    s_next = LangevinState(
                        r_corrected, s_next.momentum, s_next.force,
                        s_next.mass, s_next.key, s_next.cap_count
                    )
                    return (s_next, step_i + 1), None

                (s_next, new_step), _ = lax.scan(
                    single_step, (s_carry, global_step),
                    None, length=steps_per_save,
                )

                io_callback(
                    write_fn,
                    None,
                    s_next.positions,
                    batch_idx,
                    save_idx,
                )

                return (s_next, new_step), None

            (final_s, _), _ = lax.scan(
                save_step, (s, jnp.int32(0)),
                jnp.arange(n_saves),
            )

        return final_s

    # Build batch indices for the callback
    batch_indices = jnp.arange(B)

    return safe_map_no_output(
        produce_single_streaming,
        (batch, state, batch_indices),
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Neighbor-list aware variants
# ---------------------------------------------------------------------------

def build_neighbor_list_jaxmd(
    positions: 'Array',
    cutoff: float = 20.0,
    capacity_multiplier: float = 1.25,
    dr_threshold: float = 1.0,
) -> tuple[Any, Any]:
    """Build a dynamic neighbor list using JAX-MD's native spatial partitioning.

    This produces a NeighborList object whose `update()` method is fully
    JIT-compatible and runs on GPU inside lax.scan / lax.fori_loop.

    Only the initial `allocate()` runs on CPU (uses Python int for sizing).
    Subsequent `update()` calls use fixed shapes and stay on-device.

    Args:
        positions: (N, 3) atom positions.
        cutoff: Neighbor cutoff in Angstroms.
        capacity_multiplier: Extra buffer for neighbor count fluctuations.
        dr_threshold: Rebuild neighbors only when atoms move > this distance.

    Returns:
        nbrs: JAX-MD NeighborList object.
        neighbor_fn: NeighborListFns for potential reallocation.
    """
    displacement_fn, _ = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=1000.0,  # Large virtual box for free-space (implicit solvent)
        r_cutoff=cutoff,
        disable_cell_list=True,  # Brute-force O(N^2) build — fine for proteins
        format=partition.Dense,
        capacity_multiplier=capacity_multiplier,
        dr_threshold=dr_threshold,
    )
    nbrs = neighbor_fn.allocate(positions)
    return nbrs, neighbor_fn


def build_batched_neighbor_list_jaxmd(
    batch_positions: 'Array',
    cutoff: float = 20.0,
    capacity_multiplier: float = 1.25,
    dr_threshold: float = 1.0,
) -> tuple[Any, Any]:
    """Build batched neighbor lists for heterogeneous protein systems.

    Allocates per-system on CPU, finds global max K, pads all to the same K,
    and stacks into a single batched NeighborList PyTree compatible with vmap.

    Args:
        batch_positions: (B, N, 3) batched positions.
        cutoff: Neighbor cutoff in Angstroms.
        capacity_multiplier: Extra buffer for neighbor count fluctuations.
        dr_threshold: Rebuild neighbors only when atoms move > this distance.

    Returns:
        batched_nbrs: Stacked NeighborList PyTree with shapes (B, N, K).
        neighbor_fn: NeighborListFns for potential reallocation.
    """
    import jax.tree_util as tu

    B = batch_positions.shape[0]
    displacement_fn, _ = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=1000.0,
        r_cutoff=cutoff,
        disable_cell_list=True,
        format=partition.Dense,
        capacity_multiplier=capacity_multiplier,
        dr_threshold=dr_threshold,
    )

    # Allocate per-system, find global max K
    nbrs_list = []
    max_k = 0
    for i in range(B):
        nbrs = neighbor_fn.allocate(batch_positions[i])
        nbrs_list.append(nbrs)
        max_k = max(max_k, nbrs.idx.shape[1])

    # Re-allocate with enough capacity for the densest system,
    # then trim all to exactly max_k for uniform shapes
    padded_nbrs_list = []
    for i in range(B):
        nbrs = neighbor_fn.allocate(batch_positions[i], extra_capacity=max_k)
        padded_idx = nbrs.idx[:, :max_k]
        padded_nbrs = dataclasses.replace(nbrs, idx=padded_idx, max_occupancy=max_k)
        padded_nbrs_list.append(padded_nbrs)

    # Stack array leaves, share static fields (update_fn, format, etc.)
    batched_nbrs = tu.tree_map(lambda *x: jnp.stack(x), *padded_nbrs_list)
    return batched_nbrs, neighbor_fn


def make_langevin_step_nl_dynamic(
    dt: float, kT: float, gamma: float,
    energy_fn: Callable | None = None,
) -> Callable:
    """Create a BAOAB Langevin step with dynamic JAX-MD neighbor list updates.

    Unlike make_langevin_step_nl which takes a static neighbor_idx array,
    this version takes a full JAX-MD NeighborList object, calls nbrs.update()
    after the position update, and returns the updated NeighborList.

    The neighbor list update runs fully on GPU inside JIT/lax.scan.

    Args:
        energy_fn: Energy function taking (sys, neighbor_idx, displacement_fn).
            Defaults to single_padded_energy_nl_cvjp.

    Returns:
        step_fn(padded_sys, state, nbrs) -> (LangevinState, NeighborList)
    """
    if energy_fn is None:
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        energy_fn = single_padded_energy_nl_cvjp

    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))

    displacement_fn, _ = space.free()

    def step_fn(
        padded_sys: PaddedSystem,
        state: LangevinState,
        nbrs: Any,
    ) -> tuple[LangevinState, Any]:
        # Use current neighbor list for this step's energy/forces
        neighbor_idx = nbrs.idx

        def _energy_of_r(r):
            sys_with_r = dataclasses.replace(padded_sys, positions=r)
            return energy_fn(sys_with_r, neighbor_idx, displacement_fn)

        force_fn = jax.grad(lambda r: _energy_of_r(r))

        r, p, f, m, key = (
            state.positions, state.momentum, state.force,
            state.mass, state.key,
        )

        # B
        p = p - 0.5 * dt * f
        # A
        r = r + 0.5 * dt * p / m[:, None]
        # O
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, p.shape)
        p = c1 * p + jnp.sqrt(m[:, None] * kT) * c2 * noise
        # A
        r = r + 0.5 * dt * p / m[:, None]

        # B — compute forces with CURRENT neighbor list
        f = force_fn(r)
        f = jnp.where(jnp.isfinite(f), f, 0.0)
        p = p - 0.5 * dt * f

        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

        # Zero out padding atoms
        amask = padded_sys.atom_mask[:, None]
        r = jnp.where(amask, r, jnp.float32(0.0))
        p = jnp.where(amask, p, jnp.float32(0.0))

        # Update neighbor list AFTER force computation.
        # The updated NL will be used by the NEXT step.
        # Cast to float32 to match JAX-MD's internal lax.cond branch types.
        new_nbrs = nbrs.update(r.astype(jnp.float32))

        new_state = LangevinState(r, p, f, m, key)
        return new_state, new_nbrs

    return step_fn


def build_neighbor_list(
    positions,
    n_real: int,
    n_padded: int,
    cutoff: float = 8.0,
    pad_k: int = 32,
) -> 'Array':
    """Build a static neighbor list for a single system on the host.

    Args:
        positions: (N, 3) positions array.
        n_real: Number of real (non-padding) atoms.
        n_padded: Total padded atom count (N).
        cutoff: Distance cutoff in Angstroms.
        pad_k: Extra neighbors to add as headroom.

    Returns:
        neighbor_idx: (N, K) int32 array. Padding sentinel = n_padded.
        max_k: Number of neighbor columns.
    """
    import numpy as np

    pos_np = np.asarray(positions)
    # Pairwise distances for real atoms only
    dists = np.linalg.norm(
        pos_np[:n_real, None, :] - pos_np[None, :n_real, :], axis=-1,
    )
    max_k = 0
    for i in range(n_real):
        k = int(np.sum((dists[i] < cutoff) & (dists[i] > 0)))
        max_k = max(max_k, k)
    max_k = min(max_k + pad_k, n_padded)

    neighbor_idx = np.full((n_padded, max_k), n_padded, dtype=np.int32)
    for i in range(n_real):
        nbrs = np.where((dists[i] < cutoff) & (dists[i] > 0))[0]
        neighbor_idx[i, : len(nbrs)] = nbrs

    return jnp.array(neighbor_idx), max_k


def batched_equilibrate_nl(
    batch: PaddedSystem,
    neighbor_idx: 'Array',
    key: 'Array',
    n_steps: int,
    temperature_k: float = 310.15,
    chunk_size: int | None = None,
    energy_fn: Callable | None = None,
) -> LangevinState:
    """Equilibrate using neighbor-list energy (O(N*K) per step).

    Same as batched_equilibrate but uses the NL energy path.
    Defaults to single_padded_energy_nl_cvjp for best gradient performance.

    Args:
        batch: Batched padded systems (B, N, 3).
        neighbor_idx: Batched neighbor lists (B, N, K) int32.
        key: PRNG key for Langevin noise.
        n_steps: Number of equilibration steps.
        temperature_k: Temperature in Kelvin.
        chunk_size: Chunk size for safe_map.
        energy_fn: Custom energy function (default: cvjp variant).
    """
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    if energy_fn is None:
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        energy_fn = single_padded_energy_nl_cvjp

    dt = 2.0 / AKMA_TIME_UNIT_FS
    kT = temperature_k * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3

    step_fn = make_langevin_step_nl(dt, kT, gamma, energy_fn=energy_fn)

    leaves, _ = jax.tree_util.tree_flatten(batch)
    B = leaves[0].shape[0]
    keys = jax.random.split(key, B)

    displacement_fn, _ = space.free()

    def equilibrate_single(
        args: tuple[PaddedSystem, 'Array', 'Array'],
    ) -> LangevinState:
        sys, nbr, k = args

        def _energy_of_r(r):
            sys_with_r = dataclasses.replace(sys, positions=r)
            return energy_fn(sys_with_r, nbr, displacement_fn)

        initial_f = jax.grad(_energy_of_r)(sys.positions)

        state = LangevinState(
            positions=sys.positions,
            momentum=jnp.zeros_like(sys.positions),
            force=initial_f,
            mass=sys.masses,
            key=k,
        )

        def scan_step(s, _):
            return step_fn(sys, s, nbr), None

        final_s, _ = lax.scan(scan_step, state, None, length=n_steps)
        return final_s

    return safe_map(
        equilibrate_single, (batch, neighbor_idx, keys),
        chunk_size=chunk_size,
    )


def batched_produce_streaming_nl(
    batch: PaddedSystem,
    neighbor_idx: 'Array',
    state: LangevinState,
    n_saves: int,
    steps_per_save: int,
    write_fn: Callable,
    temperature_k: float = 310.15,
    chunk_size: int | None = None,
    energy_fn: Callable | None = None,
) -> LangevinState:
    """Produce trajectory with streaming IO using neighbor-list energy.

    Same as batched_produce_streaming but uses O(N*K) energy via NL.
    Defaults to single_padded_energy_nl_cvjp for best gradient performance.

    Args:
        batch: Batched padded systems (B, N, 3).
        neighbor_idx: Batched neighbor lists (B, N, K) int32.
        state: Initial Langevin state from equilibration.
        n_saves: Number of trajectory frames to save.
        steps_per_save: MD steps between saves.
        write_fn: Host callback (positions, batch_idx, save_idx) -> None.
        temperature_k: Temperature in Kelvin.
        chunk_size: Chunk size for safe_map.
        energy_fn: Custom energy function (default: cvjp variant).

    Returns:
        Final LangevinState (trajectory streamed to disk).
    """
    from jax.experimental import io_callback
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    if energy_fn is None:
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        energy_fn = single_padded_energy_nl_cvjp

    dt = 2.0 / AKMA_TIME_UNIT_FS
    kT = temperature_k * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3

    step_fn = make_langevin_step_nl(dt, kT, gamma, energy_fn=energy_fn)

    leaves, _ = jax.tree_util.tree_flatten(batch)
    B = leaves[0].shape[0]

    def produce_single_streaming_nl(
        args: tuple[PaddedSystem, 'Array', LangevinState, Any],
    ) -> LangevinState:
        sys, nbr, s, batch_idx = args

        def save_step(s, save_idx):
            def single_step(s_inner, _):
                return step_fn(sys, s_inner, nbr), None

            s_next, _ = lax.scan(single_step, s, None, length=steps_per_save)

            io_callback(
                write_fn,
                None,
                s_next.positions,
                batch_idx,
                save_idx,
            )

            return s_next, None

        final_s, _ = lax.scan(save_step, s, jnp.arange(n_saves))
        return final_s

    batch_indices = jnp.arange(B)

    return safe_map_no_output(
        produce_single_streaming_nl,
        (batch, neighbor_idx, state, batch_indices),
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Dynamic (JAX-MD native) neighbor list variants
# ---------------------------------------------------------------------------

def batched_equilibrate_nl_dynamic(
    batch: PaddedSystem,
    batched_nbrs: Any,
    key: 'Array',
    n_steps: int,
    temperature_k: float = 310.15,
    chunk_size: int | None = 1,
    energy_fn: Callable | None = None,
) -> tuple[LangevinState, Any]:
    """Equilibrate using dynamic JAX-MD neighbor lists (GPU-native updates).

    Unlike batched_equilibrate_nl which uses a static neighbor_idx array,
    this version carries the full NeighborList PyTree through lax.scan
    and calls nbrs.update() every step on the GPU.

    Args:
        batch: Batched padded systems (B, N, 3).
        batched_nbrs: Batched JAX-MD NeighborList PyTree (B, N, K).
        key: PRNG key for Langevin noise.
        n_steps: Number of equilibration steps.
        temperature_k: Temperature in Kelvin.
        chunk_size: Chunk size for safe_map.
        energy_fn: Custom energy function (default: cvjp variant).

    Returns:
        Tuple of (final LangevinState, updated NeighborList).
    """
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    if energy_fn is None:
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        energy_fn = single_padded_energy_nl_cvjp

    dt = 2.0 / AKMA_TIME_UNIT_FS
    kT = temperature_k * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3

    step_fn = make_langevin_step_nl_dynamic(dt, kT, gamma, energy_fn=energy_fn)

    leaves, _ = jax.tree_util.tree_flatten(batch)
    B = leaves[0].shape[0]
    keys = jax.random.split(key, B)

    displacement_fn, _ = space.free()

    def equilibrate_single(
        args: tuple[PaddedSystem, Any, 'Array'],
    ) -> tuple[LangevinState, Any]:
        sys, nbrs, k = args

        neighbor_idx = nbrs.idx

        def _energy_of_r(r):
            sys_with_r = dataclasses.replace(sys, positions=r)
            return energy_fn(sys_with_r, neighbor_idx, displacement_fn)

        initial_f = jax.grad(_energy_of_r)(sys.positions)

        state = LangevinState(
            positions=sys.positions,
            momentum=jnp.zeros_like(sys.positions),
            force=initial_f,
            mass=sys.masses,
            key=k,
        )

        def scan_step(carry, _):
            s, nbrs = carry
            new_s, new_nbrs = step_fn(sys, s, nbrs)
            return (new_s, new_nbrs), None

        (final_s, final_nbrs), _ = lax.scan(
            scan_step, (state, nbrs), None, length=n_steps,
        )
        return final_s, final_nbrs

    results = safe_map(
        equilibrate_single, (batch, batched_nbrs, keys),
        chunk_size=chunk_size,
    )
    return results  # (batched LangevinState, batched NeighborList)


def batched_produce_streaming_nl_dynamic(
    batch: PaddedSystem,
    batched_nbrs: Any,
    state: LangevinState,
    n_saves: int,
    steps_per_save: int,
    write_fn: Callable,
    temperature_k: float = 310.15,
    chunk_size: int | None = None,
    energy_fn: Callable | None = None,
) -> tuple[LangevinState, Any]:
    """Produce trajectory with streaming IO using dynamic JAX-MD neighbor lists.

    Same as batched_produce_streaming_nl but uses native GPU-updated neighbor
    lists via nbrs.update() inside lax.scan.

    Args:
        batch: Batched padded systems (B, N, 3).
        batched_nbrs: Batched JAX-MD NeighborList PyTree (B, N, K).
        state: Initial Langevin state from equilibration.
        n_saves: Number of trajectory frames to save.
        steps_per_save: MD steps between saves.
        write_fn: Host callback (positions, batch_idx, save_idx) -> None.
        temperature_k: Temperature in Kelvin.
        chunk_size: Chunk size for safe_map.
        energy_fn: Custom energy function (default: cvjp variant).

    Returns:
        Tuple of (final LangevinState, updated NeighborList).
    """
    from jax.experimental import io_callback
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    if energy_fn is None:
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        energy_fn = single_padded_energy_nl_cvjp

    dt = 2.0 / AKMA_TIME_UNIT_FS
    kT = temperature_k * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3

    step_fn = make_langevin_step_nl_dynamic(dt, kT, gamma, energy_fn=energy_fn)

    leaves, _ = jax.tree_util.tree_flatten(batch)
    B = leaves[0].shape[0]

    def produce_single_streaming_nl_dynamic(
        args: tuple[PaddedSystem, Any, LangevinState, Any],
    ) -> tuple[LangevinState, Any]:
        sys, nbrs, s, batch_idx = args

        def save_step(carry, save_idx):
            s, nbrs = carry

            def single_step(carry_inner, _):
                s_inner, nbrs_inner = carry_inner
                new_s, new_nbrs = step_fn(sys, s_inner, nbrs_inner)
                return (new_s, new_nbrs), None

            (s_next, nbrs_next), _ = lax.scan(
                single_step, (s, nbrs), None, length=steps_per_save,
            )

            io_callback(
                write_fn,
                None,
                s_next.positions,
                batch_idx,
                save_idx,
            )

            return (s_next, nbrs_next), None

        (final_s, final_nbrs), _ = lax.scan(
            save_step, (s, nbrs), jnp.arange(n_saves),
        )
        return final_s, final_nbrs

    batch_indices = jnp.arange(B)

    return safe_map_no_output(
        produce_single_streaming_nl_dynamic,
        (batch, batched_nbrs, state, batch_indices),
        chunk_size=chunk_size,
    )

