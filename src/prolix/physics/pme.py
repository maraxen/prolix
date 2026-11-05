"""PME (Particle Mesh Ewald) implementation in JAX."""

from __future__ import annotations

import functools
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from prolix.export import in_export_context
from prolix.physics.pbc import box_volume, invert_box_lower_triangular

# kcal*A/(mol*e^2)
COULOMB_CONSTANT = 332.0637


def _is_triclinic_box_array(box: jnp.ndarray | None) -> bool:
    """Return True if ``box`` is a (3, 3) or (B, 3, 3) lower-triangular triclinic matrix."""
    if box is None:
        return False
    b_arr = jnp.asarray(box)
    if b_arr.ndim == 1:
        return False
    if b_arr.ndim == 3 and b_arr.shape[-2:] == (3, 3):
        return True
    if b_arr.ndim == 2 and b_arr.shape == (3, 3):
        if not isinstance(b_arr, jax.core.Tracer):
            b_np = np.asarray(b_arr)
            return bool(
                b_np[0, 1] == 0.0 and b_np[0, 2] == 0.0 and b_np[1, 2] == 0.0
            )
        return True
    return False


def _is_ortho_box(positions: jnp.ndarray, box_size: jnp.ndarray) -> bool:
    """Return True if ``box_size`` is orthorhombic ``(3,)`` or batched ``(B, 3)``."""
    box_arr = jnp.asarray(box_size)
    if box_arr.ndim == 1:
        return True
    if (
        positions.ndim == 3
        and box_arr.ndim == 2
        and box_arr.shape[0] == positions.shape[0]
        and box_arr.shape[-1] == 3
    ):
        return not _is_triclinic_box_array(box_arr)
    return False


def _resolve_env_pme_grid_points(
    n_atoms: int | None = None, default: int | None = None
) -> int | None:
    """Resolve ``PROLIX_PME_GRID_POINTS`` (scalar int or bucket map like ``'8192:48,25000:64'``)."""
    raw = os.environ.get("PROLIX_PME_GRID_POINTS", "").strip()
    if not raw:
        return default
    if ":" in raw or "," in raw:
        mapping: list[tuple[int, int]] = []
        fallback = default
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                k_s, v_s = part.split(":", 1)
                mapping.append((int(k_s.strip()), int(v_s.strip())))
            else:
                fallback = int(part)
        if n_atoms is not None and mapping:
            mapping.sort(key=lambda kv: kv[0])
            for max_n, gp in mapping:
                if int(n_atoms) <= max_n and gp > 0:
                    return gp
            if mapping[-1][1] > 0:
                return mapping[-1][1]
        return fallback
    gp = int(raw)
    return gp if gp > 0 else default


def _spread_charges_reference(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    *,
    grid_dims: tuple[int, int, int],
    order: int = 4,
) -> jnp.ndarray:
    """Spread charges onto the PME mesh in plain JAX."""

    def _spread_single(pos_1, q_1, mask_1, box_1):
        st = _bspline4_stencil(pos_1, q_1, mask_1, box_1, grid_dims, order)
        Q_1 = jnp.zeros(grid_dims, dtype=st["work_dtype"])
        with jax.named_scope("pme_charge_spread"):
            Q_1 = Q_1.at[st["ix"], st["iy"], st["iz"]].add(st["w"])
        return Q_1

    if positions.ndim == 3:
        B = positions.shape[0]
        b_q = charges.ndim == 2
        b_m = atom_mask.ndim == 2
        box_arr = jnp.asarray(box_size)
        b_box = (
            box_arr.ndim >= 2
            and box_arr.shape[0] == B
            and (_is_ortho_box(positions, box_arr) or box_arr.ndim == 3)
        )
        return jax.vmap(
            lambda b: _spread_single(
                positions[b],
                charges[b] if b_q else charges,
                atom_mask[b] if b_m else atom_mask,
                box_size[b] if b_box else box_size,
            )
        )(jnp.arange(B))
    return _spread_single(positions, charges, atom_mask, box_size)


def _gather_reference(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    theta_norm: jnp.ndarray,
    *,
    grid_dims: tuple[int, int, int],
    order: int = 4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gather ``(forces, potentials)`` from ``theta_norm`` in plain JAX."""

    def _gather_single(pos_1, q_1, mask_1, box_1, theta_1):
        dtype = pos_1.dtype
        box_arr = jnp.asarray(box_1, dtype=dtype)
        st = _bspline4_stencil(pos_1, q_1, mask_1, box_arr, grid_dims, order)
        K_arr = st["K_arr"]
        wx, wy, wz = st["wx"], st["wy"], st["wz"]
        ox, oy, oz = st["ox"], st["oy"], st["oz"]
        u = st["u"]
        dwx = _bspline4_deriv(u[:, 0])
        dwy = _bspline4_deriv(u[:, 1])
        dwz = _bspline4_deriv(u[:, 2])
        q_masked = st["q"].astype(dtype)

        with jax.named_scope("pme_bwd_gather"):
            t_val = theta_1[st["ix"], st["iy"], st["iz"]]
            w_unit = jnp.transpose(wx[ox] * wy[oy] * wz[oz], (1, 0))
            pot_1 = jnp.where(mask_1 > 0, jnp.sum(w_unit * t_val, axis=1), 0.0)
            du_x = q_masked[:, None] * jnp.transpose(dwx[ox] * wy[oy] * wz[oz], (1, 0)) * t_val
            du_y = q_masked[:, None] * jnp.transpose(wx[ox] * dwy[oy] * wz[oz], (1, 0)) * t_val
            du_z = q_masked[:, None] * jnp.transpose(wx[ox] * wy[oy] * dwz[oz], (1, 0)) * t_val
            if box_arr.ndim == 2:
                B_mat = invert_box_lower_triangular(box_arr)
                dE_ds = jnp.stack(
                    [
                        jnp.sum(du_x, axis=1) * K_arr[0],
                        jnp.sum(du_y, axis=1) * K_arr[1],
                        jnp.sum(du_z, axis=1) * K_arr[2],
                    ],
                    axis=-1,
                )
                f_1 = dE_ds @ B_mat.T
            else:
                scale = K_arr / box_arr
                f_1 = jnp.stack(
                    [
                        jnp.sum(du_x, axis=1) * scale[0],
                        jnp.sum(du_y, axis=1) * scale[1],
                        jnp.sum(du_z, axis=1) * scale[2],
                    ],
                    axis=-1,
                )
        return f_1.astype(dtype), pot_1.astype(dtype)

    if positions.ndim == 3:
        B = positions.shape[0]
        b_q = charges.ndim == 2
        b_m = atom_mask.ndim == 2
        box_arr_top = jnp.asarray(box_size)
        b_box = (
            box_arr_top.ndim >= 2
            and box_arr_top.shape[0] == B
            and (_is_ortho_box(positions, box_arr_top) or box_arr_top.ndim == 3)
        )
        b_theta = theta_norm.ndim == 4
        return jax.vmap(
            lambda b: _gather_single(
                positions[b],
                charges[b] if b_q else charges,
                atom_mask[b] if b_m else atom_mask,
                box_size[b] if b_box else box_size,
                theta_norm[b] if b_theta else theta_norm,
            )
        )(jnp.arange(B))
    return _gather_single(positions, charges, atom_mask, box_size, theta_norm)

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
    if box_size_np.ndim == 2:
        if _is_triclinic_box_array(box_size_np):
            box_size_np = np.diag(box_size_np)
        else:
            box_size_np = box_size_np[0]
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
    box_arr = jnp.asarray(box_size, dtype=work_dtype)
    if box_arr.ndim == 2:
        B = invert_box_lower_triangular(box_arr)
        s = positions @ B
        s_wrap = s - jnp.floor(s)
        frac = s_wrap * K_arr
    else:
        frac = (positions / box_arr) * K_arr
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


def resolve_use_pallas_pme(
    explicit: bool | None = None,
    *,
    box: jnp.ndarray | None = None,
    is_triclinic: bool = False,
) -> bool:
    """Host-side opt-in (construction time). Never call inside a jitted body."""

    if in_export_context():
        return False
    if is_triclinic or _is_triclinic_box_array(box):
        return False
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("PROLIX_USE_PALLAS_PME", "").strip().lower()
    if not raw:
        raw = os.environ.get("PROLIX_USE_PALLAS", "").strip().lower()
    return raw in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pallas_interpret() -> bool:
    from prolix.pallas_kernels import _triton_gpu_available

    return not _triton_gpu_available()


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

    if use_pallas_pme and not in_export_context() and _is_ortho_box(positions, box_size):
        from xtrax.intent import Context

        from prolix.physics.pme_pallas import make_spread_op

        backend = "gpu" if not _pallas_interpret() else "cpu"
        spread_op = make_spread_op(
            grid_dims, order, context=Context(kernels=True, backend=backend)
        )
        with jax.named_scope("pme_charge_spread"):
            return spread_op(positions, charges, atom_mask, box_size)
    return _spread_charges_reference(
        positions, charges, atom_mask, box_size, grid_dims=grid_dims, order=order
    )


def influence_function(grid_dims: tuple[int, int, int], box_size: jnp.ndarray, alpha: float, order: int = 4) -> jnp.ndarray:
    # Note: pme_greens_setup_standalone scope removed (#4330 Fix 2a) - influence_function
    # is only called from spme_reciprocal_energy (primal energy path), never from the
    # differentiable force/gradient path that runs in production.
    box_arr = jnp.asarray(box_size)
    if box_arr.ndim == 2 and not _is_triclinic_box_array(box_arr):
        return jax.vmap(lambda b: influence_function(grid_dims, b, alpha, order))(box_arr)
    Kx, Ky, Kz = grid_dims
    V = box_volume(box_arr)

    if box_arr.ndim == 2:
        nx = jnp.fft.fftfreq(Kx, dtype=box_arr.dtype) * Kx
        ny = jnp.fft.fftfreq(Ky, dtype=box_arr.dtype) * Ky
        nz = jnp.fft.rfftfreq(Kz, dtype=box_arr.dtype) * Kz
        n_grid = jnp.stack(jnp.meshgrid(nx, ny, nz, indexing="ij"), axis=0)
        B = invert_box_lower_triangular(box_arr)
        # Row-vector lattice A (r = s @ A, s = r @ B): phase invariance exp(2*pi*i * m . (r + a_i))
        # requires A @ m = n => m = B @ n (NOT B.T @ n).
        m_vecs = jnp.einsum("ij,jxyz->ixyz", B, n_grid)
        m_sq = jnp.sum(m_vecs ** 2, axis=0)
    else:
        mx = jnp.fft.fftfreq(Kx, dtype=box_arr.dtype) * Kx / box_arr[0]
        my = jnp.fft.fftfreq(Ky, dtype=box_arr.dtype) * Ky / box_arr[1]
        mz = jnp.fft.rfftfreq(Kz, dtype=box_arr.dtype) * Kz / box_arr[2]
        m_sq = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2

    gauss = jnp.exp(-jnp.pi**2 * m_sq / alpha**2)
    m_sq_safe = jnp.where(m_sq > 0, m_sq, jnp.asarray(1.0, dtype=m_sq.dtype))
    denom = jnp.pi * m_sq_safe * V

    bx = _bspline_modulation(jnp.fft.fftfreq(Kx, dtype=box_arr.dtype), Kx, order)
    by = _bspline_modulation(jnp.fft.fftfreq(Ky, dtype=box_arr.dtype), Ky, order)
    bz = _bspline_modulation(jnp.fft.rfftfreq(Kz, dtype=box_arr.dtype), Kz, order)
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
    orig_dtype = positions.dtype
    is_ortho = _is_ortho_box(positions, box_size)
    if use_pallas_pme and not in_export_context() and is_ortho:
        pos_w = positions.astype(jnp.float32)
        q_w = charges.astype(jnp.float32)
        box_w = jnp.asarray(box_size, dtype=jnp.float32)
    else:
        pos_w, q_w, box_w = positions, charges, jnp.asarray(box_size)
    Q = spread_charges(
        pos_w, q_w, atom_mask, box_w, grid_dims, order, use_pallas_pme
    )
    if use_pallas_pme and not in_export_context() and is_ortho:
        Q = jnp.asarray(Q, dtype=jnp.float32)
    Q_hat = jnp.fft.rfftn(Q, s=grid_dims, axes=(-3, -2, -1))
    if is_ortho and box_w.ndim == 2:
        G = jax.vmap(lambda b: influence_function(grid_dims, b, alpha, order))(box_w).astype(Q.dtype)
    else:
        G = influence_function(grid_dims, box_w, alpha, order).astype(Q.dtype)
    theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims, axes=(-3, -2, -1)).astype(Q.dtype)
    N_grid = jnp.asarray(float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=Q.dtype)
    return (jnp.asarray(0.5 * COULOMB_CONSTANT, dtype=Q.dtype) * jnp.sum(Q * theta, axis=(-3, -2, -1)) * N_grid).astype(orig_dtype)


def spme_self_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float = 0.34) -> jnp.ndarray:
    q_masked = charges * atom_mask.astype(charges.dtype)
    return (-alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * jnp.sum(q_masked ** 2, axis=-1)).astype(charges.dtype)


def spme_background_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float, box_size: jnp.ndarray) -> jnp.ndarray:
    Q = jnp.sum(charges * atom_mask.astype(charges.dtype), axis=-1)
    box_arr = jnp.asarray(box_size)
    if box_arr.ndim == 2 and not _is_triclinic_box_array(box_arr):
        V = jnp.prod(box_arr, axis=-1)
    else:
        V = box_volume(box_arr)
    return (-jnp.pi * Q**2 / (2.0 * alpha**2 * V) * COULOMB_CONSTANT).astype(charges.dtype)


def _unwrap_primal(x):
    return x.value if hasattr(x, "perturbed") else x


def _is_perturbed(x, default: bool = True) -> bool:
    return bool(x.perturbed) if hasattr(x, "perturbed") else bool(default)


@functools.lru_cache(maxsize=64)
def _cached_host_influence_function_np(
    grid_dims: tuple[int, int, int],
    box_key: tuple[float, ...],
    box_shape: tuple[int, ...],
    box_dtype_str: str,
    alpha: float,
    order: int,
) -> np.ndarray:
    with jax.ensure_compile_time_eval():
        box_arr = jnp.asarray(
            np.asarray(box_key, dtype=np.dtype(box_dtype_str)).reshape(box_shape)
        )
        if box_arr.ndim == 2 and not _is_triclinic_box_array(box_arr):
            G = jax.vmap(lambda b: influence_function(grid_dims, b, alpha, order))(box_arr)
        else:
            G = influence_function(grid_dims, box_arr, alpha, order)
        return np.asarray(G)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
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
        + spme_self_energy(charges, atom_mask, alpha).astype(positions.dtype)
    ).astype(positions.dtype)


def _spme_fwd(positions, charges, atom_mask, box_size, grid_dims, alpha, order, use_pallas_pme):
    positions_v = _unwrap_primal(positions)
    charges_v = _unwrap_primal(charges)
    atom_mask_v = _unwrap_primal(atom_mask)
    box_size_v = _unwrap_primal(box_size)
    box_perturbed = _is_perturbed(box_size)
    charges_perturbed = _is_perturbed(charges)

    Kx, Ky, Kz = grid_dims
    orig_dtype = positions_v.dtype
    is_ortho = _is_ortho_box(positions_v, box_size_v)
    if use_pallas_pme and not in_export_context() and is_ortho:
        pos_w = positions_v.astype(jnp.float32)
        q_w = charges_v.astype(jnp.float32)
        box_w = jnp.asarray(box_size_v, dtype=jnp.float32)
    else:
        pos_w, q_w, box_w = positions_v, charges_v, jnp.asarray(box_size_v)
    Q = spread_charges(
        pos_w, q_w, atom_mask_v, box_w, grid_dims, order, use_pallas_pme
    )
    if use_pallas_pme and not in_export_context() and is_ortho:
        Q = jnp.asarray(Q, dtype=jnp.float32)
    with jax.named_scope("pme_fft_forward"):
        Q_hat = jnp.fft.rfftn(Q, s=grid_dims, axes=(-3, -2, -1))

    with jax.named_scope("pme_greens_setup"):
        if is_ortho and box_w.ndim == 2:
            G = jax.vmap(lambda b: influence_function(grid_dims, b, alpha, order))(box_w).astype(Q.dtype)
        else:
            G = influence_function(grid_dims, box_w, alpha, order).astype(Q.dtype)

    # irfftn normalization
    with jax.named_scope("pme_fft_inverse"):
        theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims, axes=(-3, -2, -1)).astype(Q.dtype)
    N_grid = jnp.asarray(float(Kx * Ky * Kz), dtype=Q.dtype)
    theta_norm = (theta * N_grid).astype(Q.dtype)

    e_recip = (jnp.asarray(0.5 * COULOMB_CONSTANT, dtype=Q.dtype) * jnp.sum(Q * theta_norm, axis=(-3, -2, -1))).astype(orig_dtype)
    e_self = spme_self_energy(charges_v, atom_mask_v, alpha).astype(orig_dtype)

    return e_recip + e_self, (
        positions_v,
        charges_v,
        atom_mask_v,
        box_size_v,
        theta_norm,
        box_perturbed,
        charges_perturbed,
    )


def _spme_bwd(grid_dims, alpha, order, use_pallas_pme, residuals, g):
    if len(residuals) == 7:
        positions, charges, atom_mask, box_size, theta_norm, box_perturbed, charges_perturbed = residuals
    elif len(residuals) == 6:
        positions, charges, atom_mask, box_size, theta_norm, box_perturbed = residuals
        charges_perturbed = True
    else:
        positions, charges, atom_mask, box_size, theta_norm = residuals
        box_perturbed = True
        charges_perturbed = True
    dtype = positions.dtype
    box_arr = jnp.asarray(box_size, dtype=dtype)
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return (
            jnp.zeros_like(positions),
            jnp.zeros_like(charges),
            jnp.zeros_like(atom_mask, dtype=dtype),
            jnp.zeros_like(box_arr),
        )

    # StableHLO guardrail: export_context() must force pure XLA in the backward pass too.
    if use_pallas_pme and not in_export_context() and _is_ortho_box(positions, box_arr):
        from xtrax.intent import Context

        from prolix.physics.pme_pallas import make_gather_op

        backend = "gpu" if not _pallas_interpret() else "cpu"
        gather_op = make_gather_op(
            grid_dims,
            order,
            compute_dq=charges_perturbed,
            context=Context(kernels=True, backend=backend),
        )
        with jax.named_scope("pme_bwd_gather"):
            forces, potentials = gather_op(
                positions.astype(jnp.float32),
                charges.astype(jnp.float32),
                atom_mask,
                box_arr.astype(jnp.float32),
                theta_norm.astype(jnp.float32),
            )
            forces = forces.astype(dtype)
            if charges_perturbed:
                potentials = potentials.astype(dtype)
    else:
        forces, potentials = _gather_reference(
            positions,
            charges,
            atom_mask,
            box_size,
            theta_norm.astype(dtype),
            grid_dims=grid_dims,
            order=order,
        )

    g_pos = jnp.expand_dims(g, axis=(-2, -1)) if g.ndim > 0 else g

    # 1. Position gradient
    dE_dpos = (COULOMB_CONSTANT * forces * g_pos).astype(dtype)

    # 2. Charge gradient (MK-38: skip when charges are not perturbed)
    if charges_perturbed:
        g_q = jnp.expand_dims(g, axis=-1) if g.ndim > 0 else g
        q_for_self = charges[None, :] if (potentials.ndim == 2 and charges.ndim == 1) else charges
        m_for_self = atom_mask[None, :] if (potentials.ndim == 2 and atom_mask.ndim == 1) else atom_mask
        dE_dq = COULOMB_CONSTANT * potentials
        dE_dq = dE_dq + (-2.0 * alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * q_for_self)
        dE_dq = (dE_dq * m_for_self.astype(dtype) * g_q).astype(charges.dtype)
        while dE_dq.ndim > charges.ndim:
            dE_dq = jnp.sum(dE_dq, axis=0)
    else:
        dE_dq = jnp.zeros_like(charges)

    # 3. Box gradient via pure-XLA VJP of spme_reciprocal_energy w.r.t. box_size
    # Skip when box_size is not perturbed (e.g. all NVT/NPT MD force evaluations),
    # avoiding a redundant pure-XLA float64 SPME forward+backward pass every step.
    if box_perturbed:
        _, vjp_box = jax.vjp(
            lambda b: spme_reciprocal_energy(
                positions, charges, atom_mask, b, grid_dims, alpha, order, False
            ),
            box_arr,
        )
        (dE_dbox,) = vjp_box(g.astype(dtype))
        dE_dbox = dE_dbox.astype(box_arr.dtype)
    else:
        dE_dbox = jnp.zeros_like(box_arr)

    return dE_dpos, dE_dq, jnp.zeros_like(atom_mask, dtype=dtype), dE_dbox

spme_energy_with_forces.defvjp(_spme_fwd, _spme_bwd, symbolic_zeros=True)

@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _spme_energy_with_forces_cached(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    G_cached: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    alpha: float,
    order: int,
    use_pallas_pme: bool = False,
) -> jnp.ndarray:
    Kx, Ky, Kz = grid_dims
    orig_dtype = positions.dtype
    is_ortho = _is_ortho_box(positions, box_size)
    if use_pallas_pme and not in_export_context() and is_ortho:
        pos_w = positions.astype(jnp.float32)
        q_w = charges.astype(jnp.float32)
        box_w = jnp.asarray(box_size, dtype=jnp.float32)
    else:
        pos_w, q_w, box_w = positions, charges, box_size
    Q = spread_charges(
        pos_w, q_w, atom_mask, box_w, grid_dims, order, use_pallas_pme
    )
    if use_pallas_pme and not in_export_context() and is_ortho:
        Q = jnp.asarray(Q, dtype=jnp.float32)
    Q_hat = jnp.fft.rfftn(Q, s=grid_dims, axes=(-3, -2, -1))
    G = G_cached.astype(Q.dtype)
    theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims, axes=(-3, -2, -1)).astype(Q.dtype)
    N_grid = jnp.asarray(float(Kx * Ky * Kz), dtype=Q.dtype)
    e_recip = (jnp.asarray(0.5 * COULOMB_CONSTANT, dtype=Q.dtype) * jnp.sum(Q * theta, axis=(-3, -2, -1)) * N_grid).astype(orig_dtype)
    return e_recip + spme_self_energy(charges, atom_mask, alpha).astype(orig_dtype)


def _spme_cached_fwd(positions, charges, atom_mask, G_cached, box_size, grid_dims, alpha, order, use_pallas_pme):
    positions_v = _unwrap_primal(positions)
    charges_v = _unwrap_primal(charges)
    atom_mask_v = _unwrap_primal(atom_mask)
    G_cached_v = _unwrap_primal(G_cached)
    box_size_v = _unwrap_primal(box_size)
    box_perturbed = _is_perturbed(box_size)
    charges_perturbed = _is_perturbed(charges)

    Kx, Ky, Kz = grid_dims
    orig_dtype = positions_v.dtype
    is_ortho = _is_ortho_box(positions_v, box_size_v)
    if use_pallas_pme and not in_export_context() and is_ortho:
        pos_w = positions_v.astype(jnp.float32)
        q_w = charges_v.astype(jnp.float32)
        box_w = jnp.asarray(box_size_v, dtype=jnp.float32)
    else:
        pos_w, q_w, box_w = positions_v, charges_v, box_size_v
    Q = spread_charges(
        pos_w, q_w, atom_mask_v, box_w, grid_dims, order, use_pallas_pme
    )
    if use_pallas_pme and not in_export_context() and is_ortho:
        Q = jnp.asarray(Q, dtype=jnp.float32)
    with jax.named_scope("pme_fft_forward"):
        Q_hat = jnp.fft.rfftn(Q, s=grid_dims, axes=(-3, -2, -1))
    G = G_cached_v.astype(Q.dtype)
    with jax.named_scope("pme_fft_inverse"):
        theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims, axes=(-3, -2, -1)).astype(Q.dtype)
    N_grid = jnp.asarray(float(Kx * Ky * Kz), dtype=Q.dtype)
    theta_norm = (theta * N_grid).astype(Q.dtype)
    e_recip = (jnp.asarray(0.5 * COULOMB_CONSTANT, dtype=Q.dtype) * jnp.sum(Q * theta_norm, axis=(-3, -2, -1))).astype(orig_dtype)
    e_self = spme_self_energy(charges_v, atom_mask_v, alpha).astype(orig_dtype)
    return e_recip + e_self, (
        positions_v,
        charges_v,
        atom_mask_v,
        box_size_v,
        theta_norm,
        G_cached_v,
        box_perturbed,
        charges_perturbed,
    )


def _spme_cached_bwd(grid_dims, alpha, order, use_pallas_pme, residuals, g):
    if len(residuals) == 8:
        positions, charges, atom_mask, box_size, theta_norm, G_cached, box_perturbed, charges_perturbed = residuals
    elif len(residuals) == 7:
        positions, charges, atom_mask, box_size, theta_norm, G_cached, box_perturbed = residuals
        charges_perturbed = True
    else:
        positions, charges, atom_mask, box_size, theta_norm, G_cached = residuals
        box_perturbed = True
        charges_perturbed = True
    dE_dpos, dE_dq, d_mask, dE_dbox = _spme_bwd(
        grid_dims,
        alpha,
        order,
        use_pallas_pme,
        (positions, charges, atom_mask, box_size, theta_norm, box_perturbed, charges_perturbed),
        g,
    )
    return dE_dpos, dE_dq, d_mask, jnp.zeros_like(G_cached), dE_dbox


_spme_energy_with_forces_cached.defvjp(_spme_cached_fwd, _spme_cached_bwd, symbolic_zeros=True)


def make_spme_energy_fn(
    box_size: jnp.ndarray,
    alpha: float = 0.34,
    grid_spacing: float = 1.0,
    order: int = 4,
    use_pallas_pme: bool | None = None,
    *,
    grid_dims: tuple[int, int, int] | None = None,
    use_pallas: bool | None = None,
    n_atoms: int | None = None,
):
    gp_env = _resolve_env_pme_grid_points(n_atoms=n_atoms)
    if grid_dims is None and gp_env is not None and gp_env > 0:
        grid_dims = (gp_env, gp_env, gp_env)
    if grid_dims is None:
        grid_dims = compute_pme_grid_dims(box_size, grid_spacing)
    else:
        grid_dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    effective_pallas = use_pallas_pme if use_pallas_pme is not None else use_pallas
    pallas = resolve_use_pallas_pme(effective_pallas, box=box_size)
    if not isinstance(box_size, jax.core.Tracer):
        box_np = np.asarray(box_size)
        is_ortho_host = not _is_triclinic_box_array(box_np)
        G_cached_np = _cached_host_influence_function_np(
            grid_dims,
            tuple(float(x) for x in box_np.reshape(-1)),
            box_np.shape,
            str(box_np.dtype),
            float(alpha),
            int(order),
        )
        G_cached_f32_np = G_cached_np.astype(np.float32)

        def _cached_fn(positions, charges, atom_mask):
            use_p = pallas and not in_export_context()
            g_dtype = jnp.float32 if (use_p and is_ortho_host) else positions.dtype
            g_np = G_cached_f32_np if g_dtype == jnp.float32 else G_cached_np
            return _spme_energy_with_forces_cached(
                positions,
                charges,
                atom_mask,
                jnp.asarray(g_np, dtype=g_dtype),
                box_np.astype(np.dtype(positions.dtype)),
                grid_dims,
                alpha,
                order,
                use_p,
            )

        with jax.ensure_compile_time_eval():
            _cached_fn.G_cached = jnp.asarray(G_cached_np)  # type: ignore[attr-defined]
        return _cached_fn

    return lambda positions, charges, atom_mask: spme_energy_with_forces(
        positions, charges, atom_mask, box_size, grid_dims, alpha, order, pallas
    )

def make_pme_energy_fn(
    charges: jnp.ndarray,
    box_size: jnp.ndarray,
    *,
    grid_points: int = 64,
    alpha: float = 0.34,
    order: int = 4,
    grid_dims: tuple[int, int, int] | None = None,
):
    charges, box_size = jnp.asarray(charges), jnp.asarray(box_size)
    mask = jnp.ones(charges.shape[0], dtype=bool)
    if grid_dims is not None:
        inner = make_spme_energy_fn(box_size, alpha=alpha, order=order, grid_dims=grid_dims)
    else:
        # numpy, not jnp: see compute_pme_grid_dims above (debt 770).
        box_np = np.asarray(box_size)
        diag_np = np.diag(box_np) if box_np.ndim == 2 else box_np
        grid_spacing = float(np.mean(diag_np)) / float(max(grid_points, 1))
        inner = make_spme_energy_fn(box_size, alpha=alpha, grid_spacing=grid_spacing, order=order)
    return lambda positions: inner(positions, charges, mask) + spme_background_energy(charges, mask, alpha, box_size)
