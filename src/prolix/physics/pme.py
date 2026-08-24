"""PME (Particle Mesh Ewald) implementation in JAX."""

from __future__ import annotations

import functools
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# kcal*A/(mol*e^2)
COULOMB_CONSTANT = 332.0637

# ===========================================================================
# Configuration
# ===========================================================================

class SPMEParams(NamedTuple):
    """Smooth PME configuration parameters."""
    alpha: float = 0.34        # Ewald splitting parameter (1/Å)
    grid_spacing: float = 1.0  # Target grid spacing (Å), actual snapped
    order: int = 4             # B-spline interpolation order


# ===========================================================================
# B-spline machinery
# ===========================================================================

def _bspline4(u: jnp.ndarray) -> jnp.ndarray:
    """Evaluate 4th order B-spline basis at fractional offsets u in [0, 1)."""
    u2 = u * u
    u3 = u2 * u
    
    w0 = (1.0 - 3.0*u + 3.0*u2 - u3) / 6.0
    w1 = (4.0 - 6.0*u2 + 3.0*u3) / 6.0
    w2 = (1.0 + 3.0*u + 3.0*u2 - 3.0*u3) / 6.0
    w3 = u3 / 6.0
    return jnp.stack([w0, w1, w2, w3])


def _bspline4_deriv(u: jnp.ndarray) -> jnp.ndarray:
    """Evaluate derivative of 4th order B-spline basis."""
    u2 = u * u
    
    dw0 = (-3.0 + 6.0*u - 3.0*u2) / 6.0
    dw1 = (-12.0*u + 9.0*u2) / 6.0
    dw2 = (3.0 + 6.0*u - 9.0*u2) / 6.0
    dw3 = (3.0*u2) / 6.0
    return jnp.stack([dw0, dw1, dw2, dw3])


def _factorizable(n: int) -> bool:
    """True if n is a product of small primes {2, 3, 5, 7} for efficient FFT."""
    if n <= 1: return True
    for p in [2, 3, 5, 7]:
        while n % p == 0:
            n //= p
    return n == 1


def compute_pme_grid_dims(
    box_size: jnp.ndarray,
    grid_spacing: float = 1.0,
    min_dim: int = 8
) -> tuple[int, int, int]:
    """Determine FFT grid dimensions based on box size and target spacing."""
    # grid_dims must be static Python ints (used as literal array shapes
    # downstream), so this must never touch jnp: float()/int() on a jax.Array
    # raises ConcretizationTypeError whenever any JAX trace is active,
    # regardless of whether the array is actually concrete (debt 770).
    box_size_np = np.asarray(box_size)
    dims = np.maximum(np.ceil(box_size_np / grid_spacing).astype(np.int32), min_dim)
    # Ensure even/optimal FFT dimensions? For now just cast.
    return (int(dims[0]), int(dims[1]), int(dims[2]))


# ===========================================================================
# SPME Kernels
# ===========================================================================

def _bspline4_stencil(positions, charges, atom_mask, box_size, grid_dims, order: int = 4):
    """Order-4 B-spline stencil: indices and weights with shape (N, order**3).

    Shared by charge spread (scatter-add) and the custom VJP gather. ``order``
    must be 4; modulation and the cubic basis are order-4 only.
    """
    if order != 4:
        raise NotImplementedError("vectorized SPME stencil is order-4 only")
    Kx, Ky, Kz = grid_dims
    work_dtype = positions.dtype
    K_arr = jnp.array([Kx, Ky, Kz], dtype=work_dtype)
    frac = (positions / box_size) * K_arr
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(work_dtype)
    wx = _bspline4(u[:, 0])
    wy = _bspline4(u[:, 1])
    wz = _bspline4(u[:, 2])
    q = charges * atom_mask.astype(charges.dtype)
    ox, oy, oz = jnp.meshgrid(
        jnp.arange(order), jnp.arange(order), jnp.arange(order), indexing="ij"
    )
    ox = ox.reshape(-1)
    oy = oy.reshape(-1)
    oz = oz.reshape(-1)
    ix = (grid_idx[:, None, 0] + ox[None, :] - 1) % Kx
    iy = (grid_idx[:, None, 1] + oy[None, :] - 1) % Ky
    iz = (grid_idx[:, None, 2] + oz[None, :] - 1) % Kz
    w = (wx[ox] * wy[oy] * wz[oz] * q[None, :]).astype(work_dtype)
    w = jnp.transpose(w, (1, 0))
    return {
        "ix": ix,
        "iy": iy,
        "iz": iz,
        "w": w,
        "wx": wx,
        "wy": wy,
        "wz": wz,
        "ox": ox,
        "oy": oy,
        "oz": oz,
        "q": q,
        "u": u,
        "K_arr": K_arr,
        "work_dtype": work_dtype,
    }


def resolve_use_pallas_pme(explicit: bool | None = None) -> bool:
    """Host-side opt-in (construction time). Never call inside a jitted body."""
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("PROLIX_USE_PALLAS_PME", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pallas_interpret() -> bool:
    return jax.default_backend() != "gpu"


def spread_charges(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    order: int = 4,
    use_pallas_pme: bool = False,
) -> jnp.ndarray:
    """Spread charges onto FFT grid using B-spline interpolation."""
    if use_pallas_pme:
        from prolix.physics.pme_pallas import spread_charges_pallas

        return spread_charges_pallas(
            positions,
            charges,
            atom_mask,
            box_size,
            grid_dims,
            order,
            interpret=_pallas_interpret(),
        )
    # Match positions' working precision rather than hardcoding float32 --
    # mixing a float32 grid with float64 positions/box_size works fine in the
    # forward pass (jnp operators auto-promote), but under jax.grad the
    # cotangent-accumulation machinery (ad_util.py's add_jaxvals) uses raw
    # lax.add, which does NOT auto-promote and raises a hard TypeError on any
    # dtype mismatch in the backward pass. Confirmed via a real DHFR-scale
    # gradient call (flash_explicit_forces -> spme_reciprocal_energy) that
    # never previously executed successfully end-to-end (blocked upstream by
    # the eval_shape/box_size bug this benchmark redesign also fixed).
    st = _bspline4_stencil(positions, charges, atom_mask, box_size, grid_dims, order)
    Q = jnp.zeros(grid_dims, dtype=st["work_dtype"])
    with jax.named_scope("pme_charge_spread"):
        Q = Q.at[st["ix"], st["iy"], st["iz"]].add(st["w"])
    return Q


def influence_function(grid_dims: tuple[int, int, int], box_size: jnp.ndarray, alpha: float, order: int = 4) -> jnp.ndarray:
    with jax.named_scope("pme_greens_setup_standalone"):
        Kx, Ky, Kz = grid_dims
        V = jnp.prod(box_size)

        mx = jnp.fft.fftfreq(Kx) * Kx / box_size[0]
        my = jnp.fft.fftfreq(Ky) * Ky / box_size[1]
        mz = jnp.fft.rfftfreq(Kz) * Kz / box_size[2]
        m_sq = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2

        gauss = jnp.exp(-jnp.pi**2 * m_sq / alpha**2)
        m_sq_safe = jnp.where(m_sq > 0, m_sq, jnp.asarray(1.0, dtype=m_sq.dtype))
        denom = jnp.pi * m_sq_safe * V

        bx = _bspline_modulation(jnp.fft.fftfreq(Kx, dtype=box_size.dtype), Kx, order)
        by = _bspline_modulation(jnp.fft.fftfreq(Ky, dtype=box_size.dtype), Ky, order)
        bz = _bspline_modulation(jnp.fft.rfftfreq(Kz, dtype=box_size.dtype), Kz, order)
        b_sq = (bx[:, None, None] * by[None, :, None] * bz[None, None, :]) ** 2
        b_sq_safe = jnp.maximum(b_sq, jnp.asarray(1e-10, dtype=b_sq.dtype))

        G = gauss / (denom * b_sq_safe)
        G = G.at[0, 0, 0].set(0.0)
        return G


def _bspline_modulation(freq_frac: jnp.ndarray, K: int, order: int) -> jnp.ndarray:
    if order == 4: m_vals = jnp.array([1.0/6.0, 4.0/6.0, 1.0/6.0, 0.0], dtype=freq_frac.dtype)
    else: raise NotImplementedError
    k_indices = jnp.arange(order, dtype=freq_frac.dtype)
    phases = jnp.exp(2j * jnp.pi * freq_frac[:, None] * k_indices[None, :])
    return jnp.maximum(
        jnp.abs(jnp.sum(m_vals[None, :] * phases, axis=-1)),
        jnp.asarray(1e-6, dtype=freq_frac.dtype),
    )


def spme_reciprocal_energy(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    alpha: float = 0.34,
    order: int = 4,
    use_pallas_pme: bool = False,
) -> jnp.ndarray:
    Q = spread_charges(
        positions, charges, atom_mask, box_size, grid_dims, order, use_pallas_pme
    )
    Q_hat = jnp.fft.rfftn(Q)
    G = influence_function(grid_dims, box_size, alpha, order)
    theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims)
    N_grid = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
    return 0.5 * COULOMB_CONSTANT * jnp.sum(Q * theta) * N_grid


def spme_self_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float = 0.34) -> jnp.ndarray:
    q_masked = charges * atom_mask.astype(charges.dtype)
    return -alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * jnp.sum(q_masked ** 2)


def spme_background_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float, box_size: jnp.ndarray) -> jnp.ndarray:
    Q = jnp.sum(charges * atom_mask.astype(charges.dtype))
    V = jnp.prod(box_size)
    return -jnp.pi * Q**2 / (2.0 * alpha**2 * V) * COULOMB_CONSTANT


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def spme_energy_with_forces(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    alpha: float,
    order: int,
    use_pallas_pme: bool = False,
) -> jnp.ndarray:
    return (
        spme_reciprocal_energy(
            positions,
            charges,
            atom_mask,
            box_size,
            grid_dims,
            alpha,
            order,
            use_pallas_pme,
        )
        + spme_self_energy(charges, atom_mask, alpha)
    )


def _spme_fwd(positions, charges, atom_mask, box_size, grid_dims, alpha, order, use_pallas_pme):
    Kx, Ky, Kz = grid_dims
    Q = spread_charges(
        positions, charges, atom_mask, box_size, grid_dims, order, use_pallas_pme
    )
    with jax.named_scope("pme_fft_forward"):
        Q_hat = jnp.fft.rfftn(Q)

    # Precompute G and m_sq for both energy and alpha-gradient
    with jax.named_scope("pme_greens_setup"):
        mx = jnp.fft.fftfreq(Kx) * Kx / box_size[0]
        my = jnp.fft.fftfreq(Ky) * Ky / box_size[1]
        mz = jnp.fft.rfftfreq(Kz) * Kz / box_size[2]
        m_sq = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2

        V = jnp.prod(box_size)
        gauss = jnp.exp(-jnp.pi**2 * m_sq / alpha**2)
        m_sq_safe = jnp.where(m_sq > 0, m_sq, jnp.asarray(1.0, dtype=m_sq.dtype))
        denom = jnp.pi * m_sq_safe * V

        bx = _bspline_modulation(jnp.fft.fftfreq(Kx, dtype=box_size.dtype), Kx, order)
        by = _bspline_modulation(jnp.fft.fftfreq(Ky, dtype=box_size.dtype), Ky, order)
        bz = _bspline_modulation(jnp.fft.rfftfreq(Kz, dtype=box_size.dtype), Kz, order)
        b_sq = (bx[:, None, None] * by[None, :, None] * bz[None, None, :]) ** 2
        b_sq_safe = jnp.maximum(b_sq, jnp.asarray(1e-10, dtype=b_sq.dtype))

        G = gauss / (denom * b_sq_safe)
        G = G.at[0, 0, 0].set(0.0)

    # irfftn normalization
    with jax.named_scope("pme_fft_inverse"):
        theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims)
    N_grid = float(Kx * Ky * Kz)
    theta_norm = theta * N_grid

    e_recip = 0.5 * COULOMB_CONSTANT * jnp.sum(Q * theta_norm)
    e_self = spme_self_energy(charges, atom_mask, alpha)

    return e_recip + e_self, (positions, charges, atom_mask, theta_norm)


def _spme_bwd(box_size, grid_dims, alpha, order, use_pallas_pme, residuals, g):
    positions, charges, atom_mask, theta_norm = residuals
    # debt 832: dtype follows positions (the ambient float precision -- float32
    # normally, float64 under jax_enable_x64) rather than hardcoding float32.
    # Hardcoded float32 intermediates here previously crashed under x64 with
    # "lax.add requires arguments to have the same dtypes, got float32,
    # float64" -- JAX's grad accumulation requires each returned cotangent to
    # exactly match its primal's dtype (unlike jnp-level ops, which
    # auto-promote), and this backward pass's internals stayed float32
    # regardless of the actual (float64-under-x64) primal dtype.
    dtype = positions.dtype
    if use_pallas_pme:
        from prolix.physics.pme_pallas import gather_pallas

        with jax.named_scope("pme_bwd_gather"):
            forces, potentials = gather_pallas(
                positions,
                charges,
                atom_mask,
                box_size,
                theta_norm,
                grid_dims,
                order,
                interpret=_pallas_interpret(),
            )
    else:
        st = _bspline4_stencil(positions, charges, atom_mask, box_size, grid_dims, order)
        K_arr = st["K_arr"]
        wx, wy, wz = st["wx"], st["wy"], st["wz"]
        ox, oy, oz = st["ox"], st["oy"], st["oz"]
        u = st["u"]
        dwx = _bspline4_deriv(u[:, 0])
        dwy = _bspline4_deriv(u[:, 1])
        dwz = _bspline4_deriv(u[:, 2])
        q_masked = st["q"].astype(dtype)

        with jax.named_scope("pme_bwd_gather"):
            t_val = theta_norm[st["ix"], st["iy"], st["iz"]]
            w_unit = jnp.transpose(wx[ox] * wy[oy] * wz[oz], (1, 0))
            potentials = jnp.sum(w_unit * t_val, axis=1)
            scale = K_arr / box_size
            fx = q_masked[:, None] * scale[0] * jnp.transpose(dwx[ox] * wy[oy] * wz[oz], (1, 0)) * t_val
            fy = q_masked[:, None] * scale[1] * jnp.transpose(wx[ox] * dwy[oy] * wz[oz], (1, 0)) * t_val
            fz = q_masked[:, None] * scale[2] * jnp.transpose(wx[ox] * wy[oy] * dwz[oz], (1, 0)) * t_val
            forces = jnp.stack(
                [jnp.sum(fx, axis=1), jnp.sum(fy, axis=1), jnp.sum(fz, axis=1)],
                axis=-1,
            )

    # 1. Position gradient
    dE_dpos = COULOMB_CONSTANT * forces * g

    # 2. Charge gradient
    dE_dq = COULOMB_CONSTANT * potentials
    dE_dq += -2.0 * alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * charges
    dE_dq = dE_dq * atom_mask.astype(dtype) * g

    # box_size and alpha are nondiff_argnums (debt 835 follow-up): no NPT/
    # barostat caller currently consumes a box_size or alpha gradient from
    # this function (pressure.py computes pressure via the standard virial
    # formula on forces/positions instead), while box_size/pme_alpha are
    # eqx.field(static=True) on PhysicsSystem -- excluded from eqx.filter_grad's
    # traced pytree entirely. This removes dead, never-consumed gradient code
    # (Q_hat_sq/mx/my/mz were kept alive in residuals only to compute these)
    # and the differentiability asymmetry it created between this custom_vjp
    # and the outer eqx partition. NOTE: this alone did NOT fix the "lax.add
    # requires ... float32, float64" crash under jax_enable_x64 (verified: the
    # crash persisted identically with only this change applied) -- the actual
    # fix was making bundle_md.py's _host_box_size return plain numpy instead
    # of a jnp array, which is what was really producing per-step tracer
    # contamination. Kept as a genuine simplification regardless. If a future
    # NPT/PME virial pressure path needs dE/dbox_size or dE/dalpha,
    # reintroduce them as differentiable args on a separate, dedicated entry
    # point rather than reverting this one.
    return dE_dpos, dE_dq, jnp.zeros_like(atom_mask, dtype=dtype)

spme_energy_with_forces.defvjp(_spme_fwd, _spme_bwd)

def make_spme_energy_fn(
    box_size: jnp.ndarray,
    alpha: float = 0.34,
    grid_spacing: float = 1.0,
    order: int = 4,
    use_pallas_pme: bool | None = None,
):
    grid_dims = compute_pme_grid_dims(box_size, grid_spacing)
    pallas = resolve_use_pallas_pme(use_pallas_pme)
    return lambda positions, charges, atom_mask: spme_energy_with_forces(
        positions, charges, atom_mask, box_size, grid_dims, alpha, order, pallas
    )

def make_pme_energy_fn(charges: jnp.ndarray, box_size: jnp.ndarray, *, grid_points: int = 64, alpha: float = 0.34, order: int = 4):
    charges, box_size = jnp.asarray(charges), jnp.asarray(box_size)
    mask = jnp.ones(charges.shape[0], dtype=bool)
    # numpy, not jnp: see compute_pme_grid_dims above (debt 770).
    grid_spacing = float(np.mean(np.asarray(box_size))) / float(max(grid_points, 1))
    inner = make_spme_energy_fn(box_size, alpha=alpha, grid_spacing=grid_spacing, order=order)
    return lambda positions: inner(positions, charges, mask) + spme_background_energy(charges, mask, alpha, box_size)
