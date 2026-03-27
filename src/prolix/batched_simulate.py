"""Batched simulation core for Prolix."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Any, TypeVar

import jax
import jax.numpy as jnp
from jax import lax
from jax_md import space, partition

from prolix.padding import PaddedSystem, precompute_dense_exclusions
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

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
    # Quality gate counters — accumulated per-step, zero-sync.
    # Layout: [0]=vlimit_exceeded  [1]=force_capped
    #         [2]=constr_violated  [3]=dx_capped
    warn_counts: Array = None  # (4,) int32, default None for backward compat

    # Named indices for warn_counts
    WARN_VLIMIT = 0
    WARN_FORCE_CAP = 1
    WARN_CONSTR_VIOL = 2
    WARN_DX_CAP = 3
    NUM_WARN_TYPES = 4

    def tree_flatten(self):
        wc = self.warn_counts if self.warn_counts is not None else jnp.zeros(self.NUM_WARN_TYPES, dtype=jnp.int32)
        children = (
            self.positions, self.momentum, self.force,
            self.mass, self.key, self.cap_count, wc,
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
                soft_core_lambda=jnp.float32(1.0),
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
        # f = single_padded_force = F = -∇U (true force, pointing downhill)
        # Correct BAOAB: p = p + (dt/2) · F
        p = p + 0.5 * dt * f

        # --- VELOCITY LIMIT: AMBER vlimit=20 Å/ps auto-correct ---
        VLIMIT = jnp.float32(20.0)
        v_per_atom = jnp.sqrt(jnp.sum((p / m[:, None]) ** 2, axis=-1) + 1e-30)
        v_exceeded = jnp.any((v_per_atom > VLIMIT) & padded_sys.atom_mask)
        v_scale = jnp.minimum(1.0, VLIMIT / (v_per_atom[:, None] + 1e-30))
        p = jnp.where((v_per_atom[:, None] > VLIMIT) & padded_sys.atom_mask[:, None], p * v_scale, p)
        
        # A: first half-step position update
        r_old = r
        r = r + 0.5 * dt * p / m[:, None]
        
        # O: stochastic (Ornstein-Uhlenbeck) update
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, p.shape)
        p = c1 * p + jnp.sqrt(m[:, None] * kT) * c2 * noise
        
        # A: second half-step position update
        r = r + 0.5 * dt * p / m[:, None]
        
        # --- DISPLACEMENT CAP: defense-in-depth against explosion ---
        # Generous 0.5 Å/step at 2 fs timestep. At 310K, typical
        # RMS displacement is ~0.05 Å/step — this cap only triggers
        # for pathological forces, preserving normal dynamics.
        MAX_DX_PROD = jnp.float32(0.5)
        dr = r - r_old
        dr_mag = jnp.sqrt(jnp.sum(dr ** 2, axis=-1, keepdims=True) + 1e-30)
        dx_capped = jnp.any(dr_mag > MAX_DX_PROD)
        dx_scale = jnp.minimum(1.0, MAX_DX_PROD / dr_mag)
        r = r_old + dr * dx_scale
        # Adjust momentum to match actual displacement
        p = jnp.where(
            dr_mag > MAX_DX_PROD,
            p * dx_scale,
            p,
        )
        
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
        # Check SHAKE deviation (AMBER tol=1e-5 Å, we use 1e-4 for warning)
        shake_dr = jnp.sqrt(
            jnp.sum((r - r_pre_shake) ** 2, axis=-1) + 1e-30,
        )  # (N,)
        constr_violated = jnp.any(shake_dr > 1e-4)

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
        force_capped = jnp.any(f_mag > 10000.0)
        f = f * jnp.minimum(1.0, 10000.0 / f_mag)
        p = p + 0.5 * dt * f
        
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

        # --- Quality gate accumulation (zero-sync) ---
        wc = state.warn_counts if state.warn_counts is not None else jnp.zeros(
            LangevinState.NUM_WARN_TYPES, dtype=jnp.int32,
        )
        wc = wc.at[LangevinState.WARN_VLIMIT].add(v_exceeded.astype(jnp.int32))
        wc = wc.at[LangevinState.WARN_FORCE_CAP].add(force_capped.astype(jnp.int32))
        wc = wc.at[LangevinState.WARN_CONSTR_VIOL].add(constr_violated.astype(jnp.int32))
        wc = wc.at[LangevinState.WARN_DX_CAP].add(dx_capped.astype(jnp.int32))

        return LangevinState(
            r, p, f, m, key,
            cap_count + force_capped.astype(jnp.int32),
            wc,
        )
        
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

        # --- VELOCITY LIMIT: AMBER vlimit=20 Å/ps ---
        VLIMIT = jnp.float32(20.0)
        v_per_atom = jnp.sqrt(jnp.sum((p / m[:, None]) ** 2, axis=-1) + 1e-30)
        v_exceeded = jnp.any((v_per_atom > VLIMIT) & padded_sys.atom_mask)
        v_scale = jnp.minimum(1.0, VLIMIT / (v_per_atom[:, None] + 1e-30))
        p = jnp.where(
            (v_per_atom[:, None] > VLIMIT) & padded_sys.atom_mask[:, None],
            p * v_scale, p,
        )

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
        force_capped = jnp.any(f_mag > 10000.0)
        f = f * jnp.minimum(1.0, 10000.0 / f_mag)
        p = p - 0.5 * dt * f

        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

        # Ghost atom pinning: restore padding to far-field positions
        r = jnp.where(mask, r, padded_sys.positions)
        p = jnp.where(mask, p, jnp.float32(0.0))

        # --- Quality gate accumulation (zero-sync) ---
        wc = state.warn_counts if state.warn_counts is not None else jnp.zeros(
            LangevinState.NUM_WARN_TYPES, dtype=jnp.int32,
        )
        wc = wc.at[LangevinState.WARN_VLIMIT].add(v_exceeded.astype(jnp.int32))
        wc = wc.at[LangevinState.WARN_FORCE_CAP].add(force_capped.astype(jnp.int32))

        return LangevinState(
            r, p, f, m, key,
            cap_count + force_capped.astype(jnp.int32),
            wc,
        )

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

        # B: f = fused forces = F = -∇U (true force)
        p = p + 0.5 * dt * f

        # --- VELOCITY LIMIT: AMBER vlimit=20 Å/ps ---
        VLIMIT = jnp.float32(20.0)
        v_per_atom = jnp.sqrt(jnp.sum((p / m[:, None]) ** 2, axis=-1) + 1e-30)
        v_exceeded = jnp.any((v_per_atom > VLIMIT) & padded_sys.atom_mask)
        v_scale = jnp.minimum(1.0, VLIMIT / (v_per_atom[:, None] + 1e-30))
        p = jnp.where(
            (v_per_atom[:, None] > VLIMIT) & padded_sys.atom_mask[:, None],
            p * v_scale, p,
        )

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
        force_capped = jnp.any(f_mag > 10000.0)
        f = f * jnp.minimum(1.0, 10000.0 / f_mag)
        p = p + 0.5 * dt * f

        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

        # Ghost atom pinning: restore padding to far-field positions
        r = jnp.where(mask, r, padded_sys.positions)
        p = jnp.where(mask, p, jnp.float32(0.0))

        # --- Quality gate accumulation (zero-sync) ---
        wc = state.warn_counts if state.warn_counts is not None else jnp.zeros(
            LangevinState.NUM_WARN_TYPES, dtype=jnp.int32,
        )
        wc = wc.at[LangevinState.WARN_VLIMIT].add(v_exceeded.astype(jnp.int32))
        wc = wc.at[LangevinState.WARN_FORCE_CAP].add(force_capped.astype(jnp.int32))

        return LangevinState(
            r, p, f, m, key,
            cap_count + force_capped.astype(jnp.int32),
            wc,
        )

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

def _selective_restraint_energy(r, r_ref, k, atom_mask, selection_mask):
    """Harmonic position restraint on a specific subset of atoms."""
    # dr: (N, 3)
    dr = (r - r_ref) * atom_mask[:, None] * selection_mask[:, None]
    return 0.5 * k * jnp.sum(dr ** 2)

def batched_minimize(
    batch: PaddedSystem,
    max_steps: int = 5000,
    lbfgs_steps: int = 2000,
    chunk_size: int | None = 1,
    min_convergence: float = 1e-4,
    fire_stage_steps: tuple[int, ...] = (500, 500, 500, 500),
) -> tuple[Array, Array]:
    """Minimize a batch of PaddedSystems using 6-stage SD → FIRE(×4) → L-BFGS.

    NLM-audited protocol (notebook 9230d5f7). All stages at λ=1.0.

    Stage 1 (SD):   500 kcal/mol/Å² heavy-atom restraints, dx0=0.01Å.
                    Single stage — SD only resolves clashes, does not release.
    Stage 2 (FIRE): 500 bb restraints, 0.1Å/step cap, no force cap.
    Stage 3 (FIRE): 125 bb restraints.
    Stage 4 (FIRE): 25 bb restraints.
    Stage 5 (FIRE): 0 (unrestrained).
    Stage 6 (L-BFGS): Unrestrained polish to drms < min_convergence.

    FIRE must NOT use force capping (corrupts P=F·v steering).
    FIRE velocity is reinitialized to zero between sub-stages.

    Args:
        batch: Padded batch of protein systems (B, N, 3).
        max_steps: Steps for the SD stage.
        lbfgs_steps: L-BFGS polish steps at full potential.
        chunk_size: Systems per vmap chunk (1 = sequential).
        min_convergence: Target RMS gradient (kcal/mol/Å).
        fire_stage_steps: Steps per FIRE sub-stage (4 values).

    Returns:
        minimized_positions: (B, N, 3)
        converged_mask: (B,) bool
    """
    from jax_md import minimize as jax_md_minimize
    from prolix.batched_energy import (
        single_padded_energy, single_padded_force
    )

    displacement_fn, shift_fn = space.free()

    # Validate fire_stage_steps
    if len(fire_stage_steps) != 4:
        raise ValueError(f"fire_stage_steps must have 4 values, got {len(fire_stage_steps)}")

    def minimize_single(sys: PaddedSystem) -> tuple[Array, Array]:
        sys = precompute_dense_exclusions(sys)
        r_ref = sys.positions  # Reference geometry for restraints
        atom_mask = sys.atom_mask
        pad_mask_3d = sys.atom_mask[:, None]  # (N, 1)

        def energy_fn(r, k_restraint=jnp.float32(0.0), selection_mask=None):
            if selection_mask is None: selection_mask = atom_mask
            sys_with_r = dataclasses.replace(sys, positions=r)
            e = single_padded_energy(sys_with_r, displacement_fn, soft_core_lambda=jnp.float32(1.0))
            e = e + _selective_restraint_energy(r, r_ref, k_restraint, atom_mask, selection_mask)
            return e

        # =================================================================
        # Phase 1: Steepest Descent — single stage, max restraints
        # =================================================================
        # SD only resolves clashes at 500 kcal/mol/Å² on heavy atoms.
        # NLM: "SD should NOT release restraints — FIRE handles the full
        # release schedule."
        def sd_step(carry, _):
            i, r, k_rest, sel_mask, max_dx = carry
            sys_with_r = dataclasses.replace(sys, positions=r)
            f_phys = single_padded_force(sys_with_r, displacement_fn, soft_core_lambda=jnp.float32(1.0))
            f_rest = -k_rest * (r - r_ref) * atom_mask[:, None] * sel_mask[:, None]
            forces = f_phys + f_rest

            # Mask padding atoms + NaN guard
            forces = forces * pad_mask_3d
            forces = jnp.where(jnp.isfinite(forces), forces, 0.0)

            # Calculate rms inside JIT
            f_mag = jnp.sqrt(jnp.sum(forces**2, axis=-1))
            rms = jnp.sqrt(jnp.sum(f_mag**2 * atom_mask) / jnp.maximum(jnp.sum(atom_mask), 1.0))

            # Also compute energy for diagnostics
            sys_diag = dataclasses.replace(sys, positions=r)
            e_diag = single_padded_energy(sys_diag, displacement_fn, soft_core_lambda=jnp.float32(1.0))

            def _log_sd(rms_val, step_val, e_val):
                import logging
                logging.getLogger("batched_simulate").info(
                    "      [SD step %d] rms_grad = %.4f  energy = %.2f",
                    int(step_val), float(rms_val), float(e_val),
                )

            jax.lax.cond(i % 1000 == 0, lambda: jax.debug.callback(_log_sd, rms, i, e_diag), lambda: None)

            # Per-atom displacement capping (AMBER dx0 = 0.01Å)
            dt = jnp.float32(0.002)
            dr = forces * dt
            dr_mag = jnp.sqrt(jnp.sum(dr**2, axis=-1, keepdims=True) + 1e-30)
            dr = dr * jnp.minimum(1.0, max_dx / dr_mag)

            return (i + 1, r + dr, k_rest, sel_mask, max_dx), None

        sd_carry = (jnp.int32(0), r_ref, jnp.float32(500.0), sys.is_heavy, jnp.float32(0.01))
        (sd_iters, r_after_sd, _, _, _), _ = lax.scan(sd_step, sd_carry, None, length=max_steps)

        # NaN rollback
        sd_failed = jnp.any(~jnp.isfinite(r_after_sd))
        r_after_sd = jnp.where(sd_failed, r_ref, r_after_sd)

        # =================================================================
        # Phase 2: FIRE — 4 sub-stages with restraint step-down
        # =================================================================
        # NLM: "FIRE must NOT use force capping — corrupts P=F·v."
        # NLM: "Use 0.1Å per-step displacement cap."
        # NLM: "Reinitialize velocity to zero between sub-stages."
        #
        # Sub-stage schedule:
        #   (500 bb, steps[0]), (125 bb, steps[1]),
        #   (25 bb, steps[2]),  (0 unrestrained, steps[3])
        fire_k_vals = [500.0, 125.0, 25.0, 0.0]
        fire_masks = [sys.is_backbone, sys.is_backbone, sys.is_backbone, atom_mask]

        r_current = r_after_sd
        for stage_idx in range(4):
            k_val = jnp.float32(fire_k_vals[stage_idx])
            sel_mask = fire_masks[stage_idx]
            n_fire_steps = fire_stage_steps[stage_idx]

            # Build FIRE energy with this stage's restraints
            def _make_fire_energy(k_r, s_m):
                def fire_energy(r):
                    return energy_fn(r, k_restraint=k_r, selection_mask=s_m)
                return fire_energy

            fire_e_fn = _make_fire_energy(k_val, sel_mask)
            fire_init_fn, fire_apply_fn = jax_md_minimize.fire_descent(
                fire_e_fn, shift_fn,
                dt_start=0.0001, dt_max=0.001
            )

            # Per-step displacement cap (0.1Å), NO force capping
            # Carry: (step, state, best_r, best_rms)
            def _make_fire_body(apply_fn, r_start):
                def fire_body_fn(carry):
                    i, state, best_r, best_rms = carry
                    r_before = state.position

                    # NaN guard on forces (but no magnitude cap)
                    f = state.force * pad_mask_3d
                    f = jnp.where(jnp.isfinite(f), f, 0.0)
                    state = dataclasses.replace(state, force=f)

                    # Calculate rms inside JIT
                    f_mag = jnp.sqrt(jnp.sum(f**2, axis=-1))
                    rms = jnp.sqrt(jnp.sum(f_mag**2 * atom_mask) / jnp.maximum(jnp.sum(atom_mask), 1.0))

                    # Update best snapshot if current is lower
                    is_better = rms < best_rms
                    new_best_r = jnp.where(is_better, state.position, best_r)
                    new_best_rms = jnp.where(is_better, rms, best_rms)

                    def _log_fire(rms_val, step_val, best_val):
                        import logging
                        logging.getLogger("batched_simulate").info("      [FIRE stage %d step %d] rms_grad = %.6f  (best = %.6f)", stage_idx, int(step_val), float(rms_val), float(best_val))

                    jax.lax.cond(i % 2000 == 0, lambda: jax.debug.callback(_log_fire, rms, i, new_best_rms), lambda: None)

                    # FIRE step (unmodified algorithm)
                    state = apply_fn(state)

                    # Per-step displacement cap: 0.1Å
                    # CRITICAL: scale velocity by the same factor to maintain
                    # FIRE's P=F·v consistency. Unscaled velocity causes
                    # runaway acceleration on subsequent steps.
                    dr = state.position - r_before
                    dr_mag = jnp.sqrt(jnp.sum(dr**2, axis=-1, keepdims=True) + 1e-30)
                    scale = jnp.minimum(1.0, jnp.float32(0.1) / dr_mag)
                    new_r = r_before + dr * scale
                    new_v = state.momentum * scale  # Match momentum to clamped step
                    # Pin padding atoms
                    new_r = jnp.where(pad_mask_3d, new_r, r_ref)
                    new_v = new_v * pad_mask_3d
                    return (i + 1, dataclasses.replace(state, position=new_r, momentum=new_v), new_best_r, new_best_rms)
                return fire_body_fn

            # Init FIRE with zero velocity (reinit between stages)
            fire_state = fire_init_fn(r_current, mass=sys.masses)
            body_fn = _make_fire_body(fire_apply_fn, r_current)
            # while_loop instead of fori_loop: no reverse-mode AD needed,
            # so avoid scan's O(N) intermediate storage.
            _, fire_final, best_r_stage, best_rms_stage = lax.while_loop(
                lambda carry: carry[0] < n_fire_steps,
                body_fn,
                (jnp.int32(0), fire_state, r_current, jnp.float32(1e10)),
            )
            # Use best-seen position instead of final (FIRE oscillates)
            r_current = best_r_stage

        r_after_fire = r_current

        # NaN guard after FIRE — don't feed junk to L-BFGS
        fire_failed = jnp.any(~jnp.isfinite(r_after_fire))
        r_after_fire = jnp.where(fire_failed, r_after_sd, r_after_fire)

        # =================================================================
        # Phase 3: L-BFGS — unrestrained final polish
        # =================================================================
        if lbfgs_steps > 0:
            import jaxopt

            # CRITICAL: L-BFGS requires EXACT energy-gradient consistency.
            # Previously used single_padded_force (analytical hand-coded LJ,
            # Coulomb, decomposed-VJP GB) which is a DIFFERENT code path from
            # single_padded_energy (autodiff-friendly LJ, Coulomb, autodiff GB).
            # Any discrepancy causes backtracking linesearch to fail Wolfe
            # conditions → coordinate explosion.
            #
            # Fix: use jax.value_and_grad on the energy function directly.
            # This mathematically guarantees grad = d(energy)/d(positions).
            def _lbfgs_objective(r):
                sys_with_r = dataclasses.replace(sys, positions=r)
                e = single_padded_energy(
                    sys_with_r, displacement_fn,
                    soft_core_lambda=jnp.float32(1.0),
                )
                e = jnp.where(jnp.isfinite(e), e, jnp.float32(1e10))
                return e

            _lbfgs_counter = [0]
            def l_val_and_grad(r):
                e, g = jax.value_and_grad(_lbfgs_objective)(r)
                # Zero out padded atom gradients and sanitize NaN
                g = g * pad_mask_3d
                g = jnp.where(jnp.isfinite(g), g, jnp.float32(0.0))
                
                # Calculate rms inside JIT
                g_mag = jnp.sqrt(jnp.sum(g**2, axis=-1))
                rms = jnp.sqrt(jnp.sum(g_mag**2 * atom_mask) / jnp.maximum(jnp.sum(atom_mask), 1.0))
                
                def _log_lbfgs(rms_val, e_val):
                    import logging
                    c = _lbfgs_counter[0]
                    _lbfgs_counter[0] += 1
                    # Log first 20 evals to catch onset, then every 200
                    if c < 20 or c % 200 == 0:
                        logging.getLogger("batched_simulate").info(
                            "      [L-BFGS eval %d] energy = %.2f  rms_grad = %.4f",
                            c, float(e_val), float(rms_val),
                        )
                
                jax.debug.callback(_log_lbfgs, rms, e)
                return e, g

            solver = jaxopt.LBFGS(
                fun=l_val_and_grad, value_and_grad=True, maxiter=lbfgs_steps,
                tol=1e-6, history_size=10, max_stepsize=0.1, linesearch="backtracking",
                jit=True, unroll=False
            )
            r_final, _ = solver.run(r_after_fire)
            r_final = jnp.where(pad_mask_3d, r_final, r_ref)

            # L-BFGS rollback: if any real atom exceeds 500Å after L-BFGS,
            # fall back to the stable FIRE output. Use pad-masked coords only.
            max_coord_real = jnp.max(jnp.abs(r_final) * atom_mask[:, None])
            lbfgs_diverged = max_coord_real > 500.0
            r_final = jnp.where(lbfgs_diverged, r_after_fire, r_final)
        else:
            r_final = r_after_fire

        # =================================================================
        # Convergence Gate
        # =================================================================
        sys_final = dataclasses.replace(sys, positions=r_final)
        final_f = single_padded_force(sys_final, displacement_fn, soft_core_lambda=jnp.float32(1.0))
        final_f = final_f * pad_mask_3d
        f_mag = jnp.sqrt(jnp.sum(final_f**2, axis=-1) + 1e-30)
        rms_grad = jnp.sqrt(jnp.sum(f_mag**2 * atom_mask) / jnp.maximum(jnp.sum(atom_mask), 1.0))
        converged = rms_grad < min_convergence

        return r_final, converged, rms_grad

    # L-BFGS is vmap-compatible (jaxopt uses lax.while_loop internally).
    # Previous chunk_size=1 restriction was removed after verifying vmap works.
    results = safe_map(minimize_single, batch, chunk_size=chunk_size)
    return results[0], results[1], results[2]

def batched_equilibrate(
    unique_batch: PaddedSystem,
    system_index: jax.Array,
    positions: jax.Array,
    key: jax.Array,
    duration_ps: float = 100.0,
    temp: float = 300.0,
    chunk_size: int | None = 1,
) -> LangevinState:
    """Run a 10-stage NVT equilibration warmup using indexed shared topology.

    Implements a 100ps production-grade protocol:
    1. 10 stages of 10ps each.
    2. Temperature ramp 10K -> 300K.
    3. Harmonic restraints 50.0 -> 0.0 (heavy then backbone).
    4. Displacement caps relax 0.1 -> 0.5 -> remove (vlimit active).
    5. All stages at λ=1.0.

    Args:
        unique_batch: Padded batch of unique protein systems.
        system_index: (B,) array mapping replica to unique system index.
        positions: (B, N, 3) initial positions (usually from batched_minimize).
        key: PRNG key for Langevin noise.
        duration_ps: Total equilibration time in ps (default 100).
        temp: Final target temperature in K (default 300).
        chunk_size: Systems per vmap chunk.

    Returns:
        equilibrated_state: Final LangevinState.
    """
    from prolix.batched_energy import single_padded_force
    
    n_stages = 10
    dt = 0.002 # 2 fs
    steps_per_stage = int((duration_ps / n_stages) / dt)
    
    # Temperature schedule: ramp 10K -> Target
    temps = jnp.linspace(10.0, temp, n_stages)
    # Restraint k-values: step down 50 -> 0 (AMBER standard)
    k_vals = jnp.array([50.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.0], dtype=jnp.float32)
    # Caps: 0.1 -> 0.5 -> None (vlimit=20.0 handles safety)
    caps = jnp.array([0.1, 0.1, 0.2, 0.2, 0.5, 0.5, 10.0, 10.0, 10.0, 10.0], dtype=jnp.float32)

    def equilibrate_single(args: tuple[int, jax.Array, jax.Array]) -> LangevinState:
        sys_idx, pos, key = args
        
        # 0. Reconstruct PaddedSystem from unique shared topology + dynamic pos
        sys = jax.tree_util.tree_map(lambda x: x[sys_idx], unique_batch)
        sys = dataclasses.replace(sys, positions=pos)
        # FlashMD uses sparse exclusions — no dense precomputation needed

        r_init = sys.positions
        atom_mask = sys.atom_mask
        pad_mask_3d = sys.atom_mask[:, None]
        ghost_positions = sys.positions
        mass = sys.masses
        mass_3d = mass[:, None] # (N, 1)
        r_ref = r_init # Reference for restraints

        # Stage masks selection: Heavy atoms then Backbone
        masks = jnp.stack([
            sys.is_heavy, sys.is_heavy, sys.is_backbone, sys.is_backbone,
            sys.is_backbone, sys.is_backbone, sys.is_backbone, sys.is_backbone,
            sys.is_backbone, sys.is_backbone
        ]) # (10, N)

        # Initialize velocities at 10K
        key, v_key = jax.random.split(key)
        v = jax.random.normal(v_key, sys.positions.shape) * jnp.sqrt(10.0 * BOLTZMANN_KCAL / (mass_3d + 1e-12))
        v = v * pad_mask_3d # Zero padding velocities
        
        displacement_fn, _ = space.free()
        gamma = 1.0 # 1/ps
        
        def stage_body_fn(carry, stage_params):
            # params: (r, v, key)
            t_target, k_rest, mask, d_cap = stage_params
            
            def step_fn(i, c):
                r_curr, v_curr, key_curr = c
                
                # 1. BAOAB - Half-step B: v = v + (dt/2) * (Force/m)
                sys_curr = dataclasses.replace(sys, positions=r_curr)
                f_phys = single_padded_force(sys_curr, displacement_fn, soft_core_lambda=jnp.float32(1.0))
                f_rest = -k_rest * (r_curr - r_ref) * mask[:, None]
                
                forces = (f_phys + f_rest) * pad_mask_3d
                forces = jnp.where(jnp.isfinite(forces), forces, 0.0)
                
                v_curr = v_curr + 0.5 * dt * (forces / (mass_3d + 1e-12))
                
                # 2. BAOAB - Half-step A: r = r + (dt/2) * v
                r_old = r_curr
                r_curr = r_curr + 0.5 * dt * v_curr
                
                # 3. BAOAB - Stochastic O: v = c1*v + c2*sqrt(kT/m)*R
                c1 = jnp.exp(-gamma * dt)
                c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))
                kT = t_target * BOLTZMANN_KCAL
                key_curr, subkey = jax.random.split(key_curr)
                noise = jax.random.normal(subkey, v_curr.shape)
                v_curr = c1 * v_curr + c2 * jnp.sqrt(kT / (mass_3d + 1e-12)) * noise
                
                # 4. BAOAB - Second half-step A: r = r + (dt/2) * v
                r_curr = r_curr + 0.5 * dt * v_curr
                
                # --- DISPLACEMENT CAP + vlimit (Safety) ---
                dr = r_curr - r_old
                dr_mag = jnp.sqrt(jnp.sum(dr**2, axis=-1, keepdims=True) + 1e-12)
                dx_scale = jnp.minimum(1.0, d_cap / dr_mag)
                r_curr = r_old + dr * dx_scale
                v_curr = v_curr * dx_scale # Scale velocity to match displacement
                
                # vlimit 20.0 A/ps
                VLIMIT = jnp.float32(20.0)
                v_mag = jnp.sqrt(jnp.sum(v_curr**2, axis=-1, keepdims=True) + 1e-12)
                v_scale = jnp.minimum(1.0, VLIMIT / v_mag)
                v_curr = v_curr * v_scale
                
                # 5. Second half-step B
                sys_new = dataclasses.replace(sys, positions=r_curr)
                f_phys_new = single_padded_force(sys_new, displacement_fn, soft_core_lambda=jnp.float32(1.0))
                f_rest_new = -k_rest * (r_curr - r_ref) * mask[:, None]
                forces_new = (f_phys_new + f_rest_new) * pad_mask_3d
                forces_new = jnp.where(jnp.isfinite(forces_new), forces_new, 0.0)
                
                v_curr = v_curr + 0.5 * dt * (forces_new / (mass_3d + 1e-12))
                
                # Final Safety: Ghost pinning
                r_curr = jnp.where(pad_mask_3d, r_curr, ghost_positions)
                v_curr = jnp.where(pad_mask_3d, v_curr, 0.0)
                
                return (r_curr, v_curr, key_curr)
            
            # while_loop instead of fori_loop: no reverse-mode AD needed,
            # so avoid scan's O(N) intermediate storage.
            def _eq_body(wl_carry):
                i, inner_carry = wl_carry
                next_carry = step_fn(i, inner_carry)
                return (i + 1, next_carry)

            _, carry_final = lax.while_loop(
                lambda wl_carry: wl_carry[0] < steps_per_stage,
                _eq_body,
                (jnp.int32(0), carry),
            )
            return carry_final, None

        # Pack stage params
        stage_params = (temps, k_vals, masks, caps)
        init_carry = (r_init, v, key)
        
        final_carry, _ = jax.lax.scan(stage_body_fn, init_carry, stage_params)
        final_r, _, _ = final_carry
        
        return LangevinState(
            positions=final_r,
            momentum=jnp.zeros_like(final_r), # Start prod with zero momenta or thermalized
            force=jnp.zeros_like(final_r),
            mass=sys.masses,
            key=key,
            cap_count=jnp.array(0, dtype=jnp.int32)
        )

    # Shard key for replicas
    B = positions.shape[0]
    keys = jax.random.split(key, B)
    
    results = safe_map(equilibrate_single, (system_index, positions, keys), chunk_size=chunk_size)
    return results

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
        # FlashMD uses sparse exclusions — no dense precomputation needed
        
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
    unique_batch: PaddedSystem,
    system_index: jax.Array,
    state: LangevinState,
    n_saves: int,
    steps_per_save: int,
    write_fn: Callable,
    temperature_k: float = 310.15,
    chunk_size: int | None = None,
    recenter_every: int = DEFAULT_RECENTER_EVERY,
    remove_rotation: bool = False,
    write_batch_size: int = 1,
    start_frame: int = 0,
    device_offset: int = 0,
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
        unique_batch: Batched padded systems (unique only) (B_unique, N, 3).
        system_index: (B,) array mapping replica to unique system index.
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
        device_offset: Global offset for batch_idx in distributed runs.

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
    B = state.positions.shape[0]

    def produce_single_streaming(
        args: tuple[int, LangevinState, Any],
    ) -> LangevinState:
        sys_idx, s, local_batch_idx = args
        batch_idx = local_batch_idx + device_offset
        
        # 0. Reconstruct PaddedSystem from unique shared topology + dynamic pos
        sys = jax.tree_util.tree_map(lambda x: x[sys_idx], unique_batch)
        sys = dataclasses.replace(sys, positions=s.positions)
        # FlashMD uses sparse exclusions — no dense precomputation needed

        # Reference frame for rotation removal: COM-centered initial positions
        ref_positions = recenter_com(s.positions, sys.masses, mask=sys.atom_mask)

        # Adjust for resumed production
        remaining_saves = n_saves - start_frame

        if write_batch_size > 1:
            # --- Batched IO path: accumulate frames, flush in batches ---
            assert remaining_saves % write_batch_size == 0, (
                f"write_batch_size={write_batch_size} must divide "
                f"remaining_saves={remaining_saves}"
            )
            n_batches = remaining_saves // write_batch_size

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

                # Write trajectory frames (offset by start_frame)
                start_save_idx = batch_save_idx * write_batch_size + start_frame
                io_callback(
                    write_fn,
                    None,
                    frames,
                    s_after_batch.positions,
                    s_after_batch.momentum,
                    s_after_batch.force,
                    s_after_batch.key,
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
                    s_next.positions,
                    s_next.momentum,
                    s_next.force,
                    s_next.key,
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
        (system_index, state, batch_indices),
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
                s_next.positions,
                s_next.momentum,
                s_next.force,
                s_next.key,
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
                s_next.positions,
                s_next.momentum,
                s_next.force,
                s_next.key,
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

