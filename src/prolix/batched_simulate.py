"""Batched simulation core for Prolix."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Any, TypeVar

import jax
import jax.numpy as jnp
from jax import lax
from jax_md import space

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

def batched_minimize(batch: PaddedSystem, max_steps: int = 5000, chunk_size: int | None = 1) -> Array:
    """Minimizes a batch of PaddedSystems using staged FIRE.

    Uses 4-stage progressive minimization consistent with simulate.py:
      Stage 1: Soft-core λ=0.1, cap=10 kcal/mol/Å (500 steps)
      Stage 2: Soft-core λ=0.5, cap=100 (500 steps)
      Stage 3: Soft-core λ=0.9, cap=1000 (1000 steps)
      Stage 4: Standard energy, NaN-sanitize only (remaining steps)

    Dynamic dt_start is computed from initial gradient magnitude.

    Returns:
        minimized_positions: (B, N, 3)
    """
    from jax_md import minimize as jax_md_minimize
    from prolix.batched_energy import single_padded_energy

    displacement_fn, shift_fn = space.free()

    # Stage definitions (consistent with simulate.py)
    stages = [
        {"lambda": 0.1, "force_cap": 10.0,   "steps": 500},
        {"lambda": 0.5, "force_cap": 100.0,  "steps": 500},
        {"lambda": 0.9, "force_cap": 1000.0, "steps": 1000},
        {"lambda": None, "force_cap": None,   "steps": max(max_steps - 2000, 1000)},
    ]

    def _make_capped_apply(fire_apply_fn, force_cap):
        """Create force-capped FIRE apply function."""
        def capped_apply(state, **kwargs):
            f = state.force
            f_norm = jnp.linalg.norm(f, axis=-1, keepdims=True)
            cap = jnp.minimum(1.0, force_cap / (f_norm + 1e-8))
            f_capped = f * cap
            f_safe = jnp.where(jnp.isfinite(f_capped), f_capped, 0.0)
            state = dataclasses.replace(state, force=f_safe)
            return fire_apply_fn(state, **kwargs)
        return capped_apply

    def _make_nan_safe_apply(fire_apply_fn):
        """NaN-sanitize only (no force cap)."""
        def nan_safe_apply(state, **kwargs):
            f = state.force
            f_safe = jnp.where(jnp.isfinite(f), f, 0.0)
            state = dataclasses.replace(state, force=f_safe)
            return fire_apply_fn(state, **kwargs)
        return nan_safe_apply

    def minimize_single(sys: PaddedSystem) -> Array:
        # Dynamic dt_start from gradient magnitude
        def base_energy_fn(r):
            sys_with_r = dataclasses.replace(sys, positions=r)
            return single_padded_energy(sys_with_r, displacement_fn)

        initial_grads = jax.grad(base_energy_fn)(sys.positions)
        max_grad = jnp.max(jnp.linalg.norm(initial_grads, axis=-1))
        dt_start = jnp.clip(0.001 / (max_grad + 1e-8), 1e-7, 0.001)
        dt_max = jnp.minimum(dt_start * 10.0, 0.01)

        current_positions = sys.positions

        for stage in stages:
            sc_lam = stage["lambda"]
            force_cap = stage["force_cap"]
            n_steps = stage["steps"]

            # Create energy function for this stage
            def stage_energy_fn(r, _lam=sc_lam):
                sys_with_r = dataclasses.replace(sys, positions=r)
                return single_padded_energy(sys_with_r, displacement_fn, soft_core_lambda=_lam)

            # Create FIRE instance
            stage_init_fn, stage_apply_fn = jax_md_minimize.fire_descent(
                stage_energy_fn, shift_fn, dt_start=dt_start, dt_max=dt_max,
            )

            # Wrap with force capping or NaN sanitization
            if force_cap is not None:
                step_fn = _make_capped_apply(stage_apply_fn, force_cap)
            else:
                step_fn = _make_nan_safe_apply(stage_apply_fn)

            # Initialize and run
            fire_state = stage_init_fn(current_positions, mass=sys.masses)

            def _run_stage(state, _step=step_fn, _n=n_steps):
                def body_fn(i, s):
                    return _step(s)
                return jax.lax.fori_loop(0, _n, body_fn, state)

            final_state = _run_stage(fire_state)
            current_positions = final_state.position

            # Ramp dt for next stage
            dt_start = jnp.minimum(dt_start * 2.0, 0.001)
            dt_max = jnp.minimum(dt_start * 10.0, 0.01)

        return current_positions

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
