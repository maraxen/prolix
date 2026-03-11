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

    def tree_flatten(self):
        children = (self.positions, self.momentum, self.force, self.mass, self.key)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def make_langevin_step(dt: float, kT: float, gamma: float) -> Callable[[PaddedSystem, LangevinState], LangevinState]:
    """Create a BAOAB Langevin step function.
    
    BAOAB scheme:
    B: p = p + 0.5 * dt * f
    A: r = r + 0.5 * dt * p / m
    O: p = exp(-gamma * dt) * p + sqrt(m * kT * (1 - exp(-2 * gamma * dt))) * R
    A: r = r + 0.5 * dt * p / m
    B: p = p + 0.5 * dt * f
    """
    from prolix.batched_energy import single_padded_energy
    from jax.tree_util import tree_map
    
    # Precompute constants
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1.0 - jnp.exp(-2.0 * gamma * dt))
    
    displacement_fn, _ = space.free()
    
    def step_fn(padded_sys: PaddedSystem, state: LangevinState) -> LangevinState:
        # Reconstruct energy function from padded system data
        # This allows heterogeneous vmap because sys is passed in
        def energy_fn(r):
            # sys is from closure, but since we are inside step_fn which is vmapped,
            # jax will handle it.
            # Wait, if we are vmapped, padded_sys will be a single system's data.
            sys_with_r = dataclasses.replace(padded_sys, positions=r)
            return single_padded_energy(sys_with_r, displacement_fn)
        
        force_fn = jax.grad(lambda r: energy_fn(r))
        
        r, p, f, m, key = state.positions, state.momentum, state.force, state.mass, state.key
        
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
        p = p - 0.5 * dt * f
        
        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)
        
        return LangevinState(r, p, f, m, key)
        
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

        # B
        f = force_fn(r)
        p = p - 0.5 * dt * f

        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

        return LangevinState(r, p, f, m, key)

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
        chunk_size = 1  # Fallback
        
    num_chunks = max(1, B // chunk_size)
    
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
    from prolix.batched_energy import single_padded_energy

    displacement_fn, shift_fn = space.free()

    # === Phase 1: FIRE soft-core stages ===
    # Progressive λ ramp resolves steric clashes without LJ singularities.
    # Force caps prevent momentum accumulation from extreme gradients.
    # Final stage stays at λ=0.999 (not 1.0) — the soft-core contribution
    # α*(1-λ) = 0.0005 is physically negligible but numerically essential
    # to prevent Inf energies from residual sub-Å overlaps.
    stage_lambdas = jnp.array([0.1, 0.5, 0.9, 0.99, 0.999, 0.999],
                              dtype=jnp.float32)
    stage_caps = jnp.array([10.0, 100.0, 1000.0, 5000.0, 10000.0, 10000.0],
                           dtype=jnp.float32)
    remaining = max(max_steps - 3000, 1000)
    stage_steps = jnp.array([500, 500, 500, 500, 1000, remaining])

    # Fixed dt for all stages — avoids per-stage recompilation
    dt_start = 0.00005
    dt_max = 0.0005

    def minimize_single(sys: PaddedSystem) -> Array:
        # Parameterized energy function — lambda is a JAX array,
        # NOT a Python float, so this traces once for all stages.
        def energy_fn(r, soft_core_lambda=jnp.float32(1.0)):
            sys_with_r = dataclasses.replace(sys, positions=r)
            return single_padded_energy(
                sys_with_r, displacement_fn,
                soft_core_lambda=soft_core_lambda,
            )

        # --- Phase 1: FIRE with soft-core ---
        fire_init_fn, fire_apply_fn = jax_md_minimize.fire_descent(
            energy_fn, shift_fn,
            dt_start=dt_start, dt_max=dt_max,
        )

        def capped_apply(state, force_cap, sc_lam):
            """Apply FIRE step with runtime force capping."""
            f = state.force
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
            return fire_apply_fn(state, soft_core_lambda=sc_lam)

        def run_stage(carry, stage_params):
            """Run one FIRE minimization stage (called via lax.scan)."""
            positions = carry
            sc_lam, force_cap, n_steps = stage_params

            fire_state = fire_init_fn(
                positions, mass=sys.masses,
                soft_core_lambda=sc_lam,
            )

            def body_fn(i, s):
                return capped_apply(s, force_cap, sc_lam)

            final_state = jax.lax.fori_loop(0, n_steps, body_fn, fire_state)
            return final_state.position, None

        stage_params = (stage_lambdas, stage_caps, stage_steps)
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
                    lambda pos: energy_fn(pos, soft_core_lambda=jnp.float32(1.0))
                )(r)
                # Zero gradient on padding atoms — prevents L-BFGS from
                # moving sentinel positions (9999.0) that are not physical
                grad = jnp.where(real_mask_3d, grad, 0.0)
                # Sanitize any NaN/Inf gradients from residual overlaps
                grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
                return val, grad

            solver = jaxopt.LBFGS(
                fun=lbfgs_value_and_grad,
                value_and_grad=True,  # We provide (value, grad) directly
                maxiter=lbfgs_steps,
                tol=1e-3,           # Force convergence tolerance
                history_size=10,    # L-BFGS memory (standard for proteins)
                max_stepsize=1.0,   # Prevent wild jumps
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
            lbfgs_positions, _ = solver.run(fire_positions)

            # Restore padding positions exactly (belt-and-suspenders)
            final_positions = jnp.where(real_mask_3d, lbfgs_positions,
                                        fire_positions)
        else:
            final_positions = fire_positions

        return final_positions

    return safe_map(minimize_single, batch, chunk_size=chunk_size)

def batched_equilibrate(
    batch: PaddedSystem, 
    key: Array, 
    n_steps: int, 
    temperature_k: float = 310.15, 
    chunk_size: int | None = None
) -> LangevinState:
    """Equilibrate a batch of padded systems."""
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
    from prolix.batched_energy import single_padded_energy
    
    displacement_fn, _ = space.free()
    
    dt = 2.0 / AKMA_TIME_UNIT_FS
    kT = temperature_k * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3
    
    step_fn = make_langevin_step(dt, kT, gamma)
    
    leaves, _ = jax.tree_util.tree_flatten(batch)
    B = leaves[0].shape[0]
    keys = jax.random.split(key, B)
    
    def equilibrate_single(args: tuple[PaddedSystem, Array]) -> LangevinState:
        sys, k = args
        
        def energy_fn(r):
            sys_with_r = dataclasses.replace(sys, positions=r)
            return single_padded_energy(sys_with_r, displacement_fn)
            
        initial_f = jax.grad(energy_fn)(sys.positions)
        
        state = LangevinState(
            positions=sys.positions,
            momentum=jnp.zeros_like(sys.positions),
            force=initial_f,
            mass=sys.masses,
            key=k
        )
        def scan_step(s, _):
            return step_fn(sys, s), None
        final_s, _ = lax.scan(scan_step, state, None, length=n_steps)
        return final_s
        
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
        chunk_size = 1

    num_chunks = max(1, B // chunk_size)

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
) -> LangevinState:
    """Produce trajectory with streaming IO — no GPU trajectory accumulation.

    Instead of accumulating a (n_saves, N, 3) trajectory tensor on GPU,
    this function uses jax.experimental.io_callback to stream each frame
    to the host as it's produced. The host-side write_fn handles serialization
    and disk IO asynchronously while the GPU continues computing.

    Args:
        batch: Batched padded systems (B, N, 3).
        state: Initial Langevin state from equilibration.
        n_saves: Number of trajectory frames to save.
        steps_per_save: MD steps between saves.
        write_fn: Host callback with signature (positions, batch_idx, save_idx) -> None.
            positions: (N, 3) array for one system at one save point.
            batch_idx: int, index into the batch dimension.
            save_idx: int, which save frame this is.
        temperature_k: Temperature in Kelvin.
        chunk_size: Chunk size for safe_map. None = full vmap.

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

        def save_step(s, save_idx):
            # Run steps_per_save MD steps
            def single_step(s_inner, _):
                return step_fn(sys, s_inner), None

            s_next, _ = lax.scan(single_step, s, None, length=steps_per_save)

            # Stream frame to host — fires async when buffer is ready
            io_callback(
                write_fn,
                None,  # void return (no data back to GPU)
                s_next.positions,
                batch_idx,
                save_idx,
                # Note: ordered=True is incompatible with vmap.
                # Frame ordering within each system is guaranteed by lax.scan.
            )

            # Return state only — NO trajectory accumulation
            return s_next, None

        final_s, _ = lax.scan(save_step, s, jnp.arange(n_saves))
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
        p = p - 0.5 * dt * f

        r = r.astype(state.positions.dtype)
        p = p.astype(state.momentum.dtype)
        f = f.astype(state.force.dtype)

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

